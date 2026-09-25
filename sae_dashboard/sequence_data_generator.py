import importlib
from dataclasses import dataclass
from os import PathLike
from time import perf_counter
from typing import Any, Callable, Literal, TypeVar

import einops
import numpy as np
import torch
from eindex import eindex
from jaxtyping import Float, Int
from torch import Tensor

from sae_dashboard.components import (
    SequenceData,
    SequenceGroupData,
    SequenceMultiGroupData,
)
from sae_dashboard.components_config import SequencesConfig
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.utils_fns import (
    TopK,
    k_largest_indices,
    random_range_indices,
    sample_unique_indices,
)
from sae_dashboard.vector_vis_data import VectorVisConfig

_PositionArrayT = TypeVar("_PositionArrayT", Tensor, "np.ndarray[Any, np.dtype[Any]]")

SequenceSelectionBackend = Literal["legacy", "columnar_gpu"]


def _parse_activation_group_name(group_name: str) -> tuple[float, float, float]:
    bin_min = -1.0
    bin_max = -1.0
    bin_contains = -1.0
    if "TOP ACTIVATIONS" in group_name:
        try:
            bin_max = float(group_name.split(" = ")[-1])
        except ValueError as exc:
            raise ValueError(
                f"Could not parse top-activation group name: {group_name!r}"
            ) from exc
    elif "INTERVAL" in group_name:
        try:
            split = group_name.split("<br>")
            first_split = split[0].split(" ")
            second_split = split[1].split(" ")
            bin_min = float(first_split[1])
            bin_max = float(first_split[-1])
            bin_contains = float(second_split[-1].rstrip("%")) / 100
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Could not parse interval group name: {group_name!r}"
            ) from exc
    return bin_min, bin_max, bin_contains


def _trim_trailing_pad_tokens(
    token_ids: list[int], values: list[float], pad_token_id: int | None
) -> tuple[list[int], list[float]]:
    if pad_token_id is None or not token_ids:
        return token_ids, values
    trimmed_len = len(token_ids)
    while trimmed_len > 0 and token_ids[trimmed_len - 1] == pad_token_id:
        trimmed_len -= 1
    return token_ids[:trimmed_len], values[:trimmed_len]


def _load_sequence_row_pyarrow_modules() -> tuple[Any, Any, Any]:
    try:
        pyarrow = importlib.import_module("pyarrow")
        pyarrow_ipc = importlib.import_module("pyarrow.ipc")
        pyarrow_parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeError(
            "Sequence row Parquet/Arrow writers require pyarrow to be installed."
        ) from exc
    return pyarrow, pyarrow_ipc, pyarrow_parquet


@dataclass(frozen=True)
class SequenceCoordinateTable:
    group_names: list[str]
    group_sizes: list[int]
    original_indices: Int[Tensor, "n_bold"]
    qualifying_token_indices: Int[Tensor, "n_bold"]
    source_token_indices: Int[Tensor, "n_bold buf"]
    token_ids: Int[Tensor, "n_bold buf"]
    feat_acts: np.ndarray
    token_logits: Float[Tensor, "n_bold buf"]

    @property
    def group_offsets(self) -> list[int]:
        return np.cumsum([0] + self.group_sizes).tolist()

    def to_sequence_row_columns(self, feature_index: int) -> dict[str, np.ndarray]:
        if self.token_ids.ndim != 2:
            raise ValueError("token_ids must be a 2D sequence table")
        if self.source_token_indices.shape != self.token_ids.shape:
            raise ValueError("source_token_indices must match token_ids shape")
        if self.token_logits.shape != self.token_ids.shape:
            raise ValueError("token_logits must match token_ids shape")

        n_sequences = int(self.token_ids.shape[0])
        context_width = int(self.token_ids.shape[1])
        if sum(self.group_sizes) != n_sequences:
            raise ValueError("group_sizes must sum to the number of sequence rows")

        row_count = n_sequences * context_width
        per_sequence_group_indices = (
            np.concatenate(
                [
                    np.full(group_size, group_idx, dtype=np.int64)
                    for group_idx, group_size in enumerate(self.group_sizes)
                ]
            )
            if n_sequences
            else np.empty(0, dtype=np.int64)
        )
        per_sequence_group_names = (
            np.concatenate(
                [
                    np.full(group_size, group_name, dtype=object)
                    for group_name, group_size in zip(
                        self.group_names, self.group_sizes, strict=True
                    )
                ]
            )
            if n_sequences
            else np.empty(0, dtype=object)
        )
        per_sequence_group_positions = (
            np.concatenate(
                [
                    np.arange(group_size, dtype=np.int64)
                    for group_size in self.group_sizes
                ]
            )
            if n_sequences
            else np.empty(0, dtype=np.int64)
        )
        sequence_indices = np.arange(n_sequences, dtype=np.int64)
        context_token_indices = np.tile(
            np.arange(context_width, dtype=np.int64), n_sequences
        )

        original_indices = (
            self.original_indices.detach().cpu().to(dtype=torch.long).numpy()
        )
        qualifying_token_indices = (
            self.qualifying_token_indices.detach().cpu().to(dtype=torch.long).numpy()
        )
        source_token_indices = (
            self.source_token_indices.detach().cpu().to(dtype=torch.long).numpy()
        )
        token_ids = self.token_ids.detach().cpu().to(dtype=torch.long).numpy()
        token_logits = self.token_logits.detach().cpu().to(dtype=torch.float32).numpy()
        feat_acts = np.asarray(self.feat_acts, dtype=np.float64)

        return {
            "feature_index": np.full(row_count, feature_index, dtype=np.int64),
            "sequence_index": np.repeat(sequence_indices, context_width),
            "group_index": np.repeat(per_sequence_group_indices, context_width),
            "group_name": np.repeat(per_sequence_group_names, context_width),
            "group_sequence_index": np.repeat(
                per_sequence_group_positions, context_width
            ),
            "context_token_index": context_token_indices,
            "original_index": np.repeat(original_indices, context_width),
            "qualifying_token_index": np.repeat(
                qualifying_token_indices, context_width
            ),
            "source_token_index": source_token_indices.reshape(row_count),
            "token_id": token_ids.reshape(row_count),
            "feat_act": feat_acts.reshape(row_count),
            "token_logit": token_logits.reshape(row_count),
        }

    @staticmethod
    def sequence_row_arrow_schema() -> Any:
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        return pyarrow.schema(
            [
                ("feature_index", pyarrow.int64()),
                ("sequence_index", pyarrow.int64()),
                ("group_index", pyarrow.int64()),
                ("group_name", pyarrow.string()),
                ("group_sequence_index", pyarrow.int64()),
                ("context_token_index", pyarrow.int64()),
                ("original_index", pyarrow.int64()),
                ("qualifying_token_index", pyarrow.int64()),
                ("source_token_index", pyarrow.int64()),
                ("token_id", pyarrow.int64()),
                ("feat_act", pyarrow.float64()),
                ("token_logit", pyarrow.float32()),
            ]
        )

    def to_sequence_row_arrow_record_batch(self, feature_index: int) -> Any:
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        return pyarrow.RecordBatch.from_pydict(
            self.to_sequence_row_columns(feature_index),
            schema=self.sequence_row_arrow_schema(),
        )

    def to_sequence_row_arrow_table(self, feature_index: int) -> Any:
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        return pyarrow.Table.from_batches(
            [self.to_sequence_row_arrow_record_batch(feature_index)]
        )

    @staticmethod
    def activation_row_arrow_schema() -> Any:
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        return pyarrow.schema(
            [
                ("feature_index", pyarrow.int64()),
                ("sequence_index", pyarrow.int64()),
                ("tokens", pyarrow.list_(pyarrow.string())),
                ("max_value", pyarrow.float64()),
                ("max_value_token_index", pyarrow.int64()),
                ("min_value", pyarrow.float64()),
                ("values", pyarrow.list_(pyarrow.float64())),
                ("bin_min", pyarrow.float64()),
                ("bin_max", pyarrow.float64()),
                ("bin_contains", pyarrow.float64()),
                ("qualifying_token_index", pyarrow.int64()),
            ]
        )

    @staticmethod
    def activation_copy_row_arrow_schema() -> Any:
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        return pyarrow.schema(
            [
                ("id", pyarrow.string()),
                ("tokens", pyarrow.list_(pyarrow.string())),
                ("dataIndex", pyarrow.null()),
                ("index", pyarrow.int64()),
                ("layer", pyarrow.string()),
                ("modelId", pyarrow.string()),
                ("dataSource", pyarrow.null()),
                ("maxValue", pyarrow.float64()),
                ("maxValueTokenIndex", pyarrow.int64()),
                ("minValue", pyarrow.float64()),
                ("values", pyarrow.list_(pyarrow.float64())),
                ("dfaValues", pyarrow.list_(pyarrow.float64())),
                ("dfaTargetIndex", pyarrow.null()),
                ("dfaMaxValue", pyarrow.null()),
                ("creatorId", pyarrow.string()),
                ("createdAt", pyarrow.string()),
                ("lossValues", pyarrow.list_(pyarrow.float64())),
                ("logitContributions", pyarrow.null()),
                ("binMin", pyarrow.float64()),
                ("binMax", pyarrow.float64()),
                ("binContains", pyarrow.float64()),
                ("qualifyingTokenIndex", pyarrow.int64()),
            ]
        )

    def to_activation_row_columns(
        self,
        feature_index: int,
        decode_token_ids: Callable[[list[int]], list[str]],
        *,
        pad_token_id: int | None = None,
    ) -> dict[str, list[Any]]:
        if self.token_ids.ndim != 2:
            raise ValueError("token_ids must be a 2D sequence table")
        if self.feat_acts.ndim != 2:
            raise ValueError("feat_acts must be a 2D sequence table")
        if self.feat_acts.shape != tuple(self.token_ids.shape):
            raise ValueError("feat_acts must match token_ids shape")

        n_sequences = int(self.token_ids.shape[0])
        if sum(self.group_sizes) != n_sequences:
            raise ValueError("group_sizes must sum to the number of sequence rows")

        token_ids = self.token_ids.detach().cpu().to(dtype=torch.long).numpy()
        qualifying_token_indices = (
            self.qualifying_token_indices.detach().cpu().to(dtype=torch.long).numpy()
        )
        feat_acts = np.asarray(self.feat_acts, dtype=np.float64)
        group_names_by_sequence = [
            group_name
            for group_name, group_size in zip(
                self.group_names, self.group_sizes, strict=True
            )
            for _ in range(group_size)
        ]

        columns: dict[str, list[Any]] = {
            "feature_index": [],
            "sequence_index": [],
            "tokens": [],
            "max_value": [],
            "max_value_token_index": [],
            "min_value": [],
            "values": [],
            "bin_min": [],
            "bin_max": [],
            "bin_contains": [],
            "qualifying_token_index": [],
        }

        for sequence_index in range(n_sequences):
            token_id_row = token_ids[sequence_index].tolist()
            value_row = np.round(feat_acts[sequence_index], 3).tolist()
            token_id_row, value_row = _trim_trailing_pad_tokens(
                token_id_row,
                value_row,
                pad_token_id,
            )
            decoded_tokens = decode_token_ids(token_id_row)
            if len(decoded_tokens) != len(token_id_row):
                raise ValueError(
                    f"Token decoder returned {len(decoded_tokens)} tokens for {len(token_id_row)} ids."
                )
            max_value = max(value_row) if value_row else 0.0
            min_value = min(value_row) if value_row else 0.0
            max_value_token_index = value_row.index(max_value) if value_row else 0
            bin_min, bin_max, bin_contains = _parse_activation_group_name(
                group_names_by_sequence[sequence_index]
            )

            columns["feature_index"].append(feature_index)
            columns["sequence_index"].append(sequence_index)
            columns["tokens"].append(decoded_tokens)
            columns["max_value"].append(max_value)
            columns["max_value_token_index"].append(max_value_token_index)
            columns["min_value"].append(min_value)
            columns["values"].append(value_row)
            columns["bin_min"].append(bin_min)
            columns["bin_max"].append(bin_max)
            columns["bin_contains"].append(bin_contains)
            columns["qualifying_token_index"].append(
                int(qualifying_token_indices[sequence_index]) - 1
            )

        return columns

    @staticmethod
    def activation_row_arrow_record_batch_from_columns(
        columns: dict[str, list[Any]],
    ) -> Any:
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        return pyarrow.RecordBatch.from_pydict(
            columns,
            schema=SequenceCoordinateTable.activation_row_arrow_schema(),
        )

    @staticmethod
    def activation_copy_row_columns_from_activation_columns(
        columns: dict[str, list[Any]],
        *,
        model_id: str,
        layer: str,
        creator_id: str,
        created_at: str,
        activation_id_prefix: str,
    ) -> dict[str, list[Any]]:
        row_count = len(columns["feature_index"])
        return {
            "id": [
                f"{activation_id_prefix}-{feature_index}-{sequence_index}"
                for feature_index, sequence_index in zip(
                    columns["feature_index"], columns["sequence_index"], strict=True
                )
            ],
            "tokens": columns["tokens"],
            "dataIndex": [None] * row_count,
            "index": columns["feature_index"],
            "layer": [layer] * row_count,
            "modelId": [model_id] * row_count,
            "dataSource": [None] * row_count,
            "maxValue": columns["max_value"],
            "maxValueTokenIndex": columns["max_value_token_index"],
            "minValue": columns["min_value"],
            "values": columns["values"],
            "dfaValues": [[] for _ in range(row_count)],
            "dfaTargetIndex": [None] * row_count,
            "dfaMaxValue": [None] * row_count,
            "creatorId": [creator_id] * row_count,
            "createdAt": [created_at] * row_count,
            "lossValues": [[] for _ in range(row_count)],
            "logitContributions": [None] * row_count,
            "binMin": columns["bin_min"],
            "binMax": columns["bin_max"],
            "binContains": columns["bin_contains"],
            "qualifyingTokenIndex": columns["qualifying_token_index"],
        }

    @staticmethod
    def activation_copy_row_arrow_record_batch_from_activation_columns(
        columns: dict[str, list[Any]],
        *,
        model_id: str,
        layer: str,
        creator_id: str,
        created_at: str,
        activation_id_prefix: str,
    ) -> Any:
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        return pyarrow.RecordBatch.from_pydict(
            SequenceCoordinateTable.activation_copy_row_columns_from_activation_columns(
                columns,
                model_id=model_id,
                layer=layer,
                creator_id=creator_id,
                created_at=created_at,
                activation_id_prefix=activation_id_prefix,
            ),
            schema=SequenceCoordinateTable.activation_copy_row_arrow_schema(),
        )

    @staticmethod
    def activation_copy_row_arrow_record_batch_from_activation_row_record_batch(
        activation_row_record_batch: Any,
        *,
        model_id: str,
        layer: str,
        creator_id: str,
        created_at: str,
        activation_id_prefix: str,
    ) -> Any:
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        required_columns = {
            "feature_index",
            "sequence_index",
            "tokens",
            "max_value",
            "max_value_token_index",
            "min_value",
            "values",
            "bin_min",
            "bin_max",
            "bin_contains",
            "qualifying_token_index",
        }
        missing_columns = required_columns - set(
            activation_row_record_batch.schema.names
        )
        if missing_columns:
            raise ValueError(
                "Activation row record batch is missing required columns for activation_copy conversion: "
                f"{sorted(missing_columns)}"
            )

        row_count = int(activation_row_record_batch.num_rows)
        feature_index_column = activation_row_record_batch.column(
            activation_row_record_batch.schema.get_field_index("feature_index")
        )
        sequence_index_column = activation_row_record_batch.column(
            activation_row_record_batch.schema.get_field_index("sequence_index")
        )
        id_column = pyarrow.array(
            [
                f"{activation_id_prefix}-{feature_index}-{sequence_index}"
                for feature_index, sequence_index in zip(
                    feature_index_column.to_pylist(),
                    sequence_index_column.to_pylist(),
                    strict=True,
                )
            ],
            type=pyarrow.string(),
        )
        null_column = pyarrow.nulls(row_count)
        empty_list_offsets = pyarrow.array([0] * (row_count + 1), type=pyarrow.int32())
        empty_float_values = pyarrow.array([], type=pyarrow.float64())
        empty_float_list_column = pyarrow.ListArray.from_arrays(
            empty_list_offsets,
            empty_float_values,
            type=pyarrow.list_(pyarrow.float64()),
        )

        def _string_column(value: str) -> Any:
            return pyarrow.array([value] * row_count, type=pyarrow.string())

        return pyarrow.RecordBatch.from_arrays(
            [
                id_column,
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index("tokens")
                ),
                null_column,
                feature_index_column,
                _string_column(layer),
                _string_column(model_id),
                null_column,
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index("max_value")
                ),
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index(
                        "max_value_token_index"
                    )
                ),
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index("min_value")
                ),
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index("values")
                ),
                empty_float_list_column,
                null_column,
                null_column,
                _string_column(creator_id),
                _string_column(created_at),
                empty_float_list_column,
                null_column,
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index("bin_min")
                ),
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index("bin_max")
                ),
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index("bin_contains")
                ),
                activation_row_record_batch.column(
                    activation_row_record_batch.schema.get_field_index(
                        "qualifying_token_index"
                    )
                ),
            ],
            schema=SequenceCoordinateTable.activation_copy_row_arrow_schema(),
        )

    def _prepare_activation_row_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate and convert this table's rows into the numpy matrices consumed by the
        shared activation-row RecordBatch core (token ids, rounded values, qualifying
        indices, and the per-row bin columns expanded from the group names)."""
        if self.token_ids.ndim != 2:
            raise ValueError("token_ids must be a 2D sequence table")
        if self.feat_acts.ndim != 2:
            raise ValueError("feat_acts must be a 2D sequence table")
        if self.feat_acts.shape != tuple(self.token_ids.shape):
            raise ValueError("feat_acts must match token_ids shape")
        n_sequences = int(self.token_ids.shape[0])
        if sum(self.group_sizes) != n_sequences:
            raise ValueError("group_sizes must sum to the number of sequence rows")

        token_ids = self.token_ids.detach().cpu().to(dtype=torch.long).numpy()
        qualifying_token_indices = (
            self.qualifying_token_indices.detach().cpu().to(dtype=torch.long).numpy()
        )
        values = np.round(np.asarray(self.feat_acts, dtype=np.float64), 3)
        group_bins = [
            _parse_activation_group_name(group_name) for group_name in self.group_names
        ]
        bin_mins = np.repeat(
            np.array([bins[0] for bins in group_bins], dtype=np.float64),
            self.group_sizes,
        )
        bin_maxes = np.repeat(
            np.array([bins[1] for bins in group_bins], dtype=np.float64),
            self.group_sizes,
        )
        bin_contains = np.repeat(
            np.array([bins[2] for bins in group_bins], dtype=np.float64),
            self.group_sizes,
        )
        return (
            token_ids,
            values,
            qualifying_token_indices,
            bin_mins,
            bin_maxes,
            bin_contains,
        )

    @staticmethod
    def _activation_row_record_batch_from_matrices(
        *,
        token_ids: np.ndarray,
        values: np.ndarray,
        feature_indices_per_row: np.ndarray,
        sequence_indices_per_row: np.ndarray,
        qualifying_token_indices: np.ndarray,
        bin_mins: np.ndarray,
        bin_maxes: np.ndarray,
        bin_contains: np.ndarray,
        decode_token_ids: Callable[[list[int]], list[str]],
        pad_token_id: int | None,
        decode_token_ids_array: "Callable[[np.ndarray], np.ndarray] | None" = None,
    ) -> Any:
        """Shared vectorized activation-row RecordBatch core.

        Operates on ``[rows, buf]`` matrices (single-feature or concatenated across a
        whole feature batch): one vectorized trailing-pad trim, masked row
        max/min/first-max-index, a single unique-id detokenization pass mapped through
        `np.unique(return_inverse=True)`, and `ListArray.from_arrays` nested columns.
        All operations are elementwise or per-row, so per-feature and concatenated
        invocations produce identical rows.
        """
        pyarrow, _, _ = _load_sequence_row_pyarrow_modules()
        n_sequences = int(token_ids.shape[0])
        buffer_width = int(token_ids.shape[1]) if n_sequences else 0

        if pad_token_id is None or buffer_width == 0:
            lengths = np.full(n_sequences, buffer_width, dtype=np.int64)
        else:
            not_pad = token_ids != pad_token_id
            has_content = not_pad.any(axis=1)
            first_content_from_end = np.argmax(not_pad[:, ::-1], axis=1)
            lengths = np.where(
                has_content, buffer_width - first_content_from_end, 0
            ).astype(np.int64)

        if n_sequences and buffer_width:
            valid = np.arange(buffer_width)[None, :] < lengths[:, None]
        else:
            valid = np.zeros((n_sequences, buffer_width), dtype=bool)
        has_values = lengths > 0
        max_values = np.where(
            has_values,
            np.where(valid, values, -np.inf).max(axis=1, initial=-np.inf),
            0.0,
        )
        min_values = np.where(
            has_values,
            np.where(valid, values, np.inf).min(axis=1, initial=np.inf),
            0.0,
        )
        if n_sequences and buffer_width:
            max_value_token_indices = np.where(
                has_values,
                np.argmax(valid & (values == max_values[:, None]), axis=1),
                0,
            ).astype(np.int64)
        else:
            max_value_token_indices = np.zeros(n_sequences, dtype=np.int64)

        flat_token_ids = token_ids[valid]
        flat_values = values[valid]
        offsets = np.zeros(n_sequences + 1, dtype=np.int32)
        np.cumsum(lengths, out=offsets[1:])

        if flat_token_ids.size:
            if decode_token_ids_array is not None:
                # array-level decoder (e.g. a persistent vocab cache): O(n) gathers with
                # no per-batch sort over the concatenated ids
                flat_token_strings = decode_token_ids_array(flat_token_ids)
                if len(flat_token_strings) != flat_token_ids.size:
                    raise ValueError(
                        f"Token array decoder returned {len(flat_token_strings)} tokens "
                        f"for {flat_token_ids.size} ids."
                    )
            else:
                unique_ids, inverse_indices = np.unique(
                    flat_token_ids, return_inverse=True
                )
                decoded_tokens = decode_token_ids(
                    [int(token_id) for token_id in unique_ids.tolist()]
                )
                if len(decoded_tokens) != len(unique_ids):
                    raise ValueError(
                        f"Token decoder returned {len(decoded_tokens)} tokens for "
                        f"{len(unique_ids)} ids."
                    )
                flat_token_strings = np.array(decoded_tokens, dtype=object)[
                    inverse_indices
                ]
        else:
            flat_token_strings = np.array([], dtype=object)

        offsets_array = pyarrow.array(offsets, type=pyarrow.int32())
        arrays = [
            pyarrow.array(feature_indices_per_row.astype(np.int64, copy=False)),
            pyarrow.array(sequence_indices_per_row.astype(np.int64, copy=False)),
            pyarrow.ListArray.from_arrays(
                offsets_array,
                pyarrow.array(flat_token_strings, type=pyarrow.string()),
            ),
            pyarrow.array(max_values, type=pyarrow.float64()),
            pyarrow.array(max_value_token_indices),
            pyarrow.array(min_values, type=pyarrow.float64()),
            pyarrow.ListArray.from_arrays(
                offsets_array,
                pyarrow.array(flat_values, type=pyarrow.float64()),
            ),
            pyarrow.array(bin_mins, type=pyarrow.float64()),
            pyarrow.array(bin_maxes, type=pyarrow.float64()),
            pyarrow.array(bin_contains, type=pyarrow.float64()),
            pyarrow.array(qualifying_token_indices.astype(np.int64) - 1),
        ]
        return pyarrow.RecordBatch.from_arrays(
            arrays, schema=SequenceCoordinateTable.activation_row_arrow_schema()
        )

    def to_activation_row_arrow_record_batch(
        self,
        feature_index: int,
        decode_token_ids: Callable[[list[int]], list[str]],
        *,
        pad_token_id: int | None = None,
    ) -> Any:
        """Vectorized activation-row RecordBatch construction (single feature).

        Produces the same rows as `activation_row_arrow_record_batch_from_columns(
        to_activation_row_columns(...))`, which is retained as the reference
        implementation; see `_activation_row_record_batch_from_matrices` for the
        shared vectorized core.
        """
        (
            token_ids,
            values,
            qualifying_token_indices,
            bin_mins,
            bin_maxes,
            bin_contains,
        ) = self._prepare_activation_row_matrices()
        n_sequences = int(token_ids.shape[0])
        return self._activation_row_record_batch_from_matrices(
            token_ids=token_ids,
            values=values,
            feature_indices_per_row=np.full(n_sequences, feature_index, dtype=np.int64),
            sequence_indices_per_row=np.arange(n_sequences, dtype=np.int64),
            qualifying_token_indices=qualifying_token_indices,
            bin_mins=bin_mins,
            bin_maxes=bin_maxes,
            bin_contains=bin_contains,
            decode_token_ids=decode_token_ids,
            pad_token_id=pad_token_id,
        )

    @staticmethod
    def activation_row_arrow_record_batches_for_features(
        tables_by_feature: "list[tuple[int, SequenceCoordinateTable]]",
        decode_token_ids: Callable[[list[int]], list[str]],
        *,
        pad_token_id: int | None = None,
        decode_token_ids_array: "Callable[[np.ndarray], np.ndarray] | None" = None,
    ) -> list[Any]:
        """Build activation-row RecordBatches for a whole feature batch in one pass.

        Concatenates every feature's row matrices, runs the shared vectorized core once
        (one pad-trim/round/reduction pass and a single unique-id detokenization over the
        entire batch), and returns zero-copy per-feature `RecordBatch.slice` views so the
        per-feature record-batch layout (manifest row groups, copy-row derivation) is
        unchanged. Rows are identical to per-feature construction because every core
        operation is elementwise or per-row.
        """
        if not tables_by_feature:
            return []
        prepared = [
            (feature_index, table._prepare_activation_row_matrices())
            for feature_index, table in tables_by_feature
        ]
        row_counts = [int(matrices[0].shape[0]) for _, matrices in prepared]
        buffer_widths = {
            int(matrices[0].shape[1]) for _, matrices in prepared if matrices[0].size
        }
        if len(buffer_widths) > 1:
            # Mixed sequence widths cannot share one concatenated matrix; fall back.
            return [
                table.to_activation_row_arrow_record_batch(
                    feature_index, decode_token_ids, pad_token_id=pad_token_id
                )
                for feature_index, table in tables_by_feature
            ]

        common_width = buffer_widths.pop() if buffer_widths else 0

        def _normalize_width(matrix: np.ndarray) -> np.ndarray:
            if matrix.shape[0] == 0 and matrix.shape[1] != common_width:
                return matrix.reshape(0, common_width)
            return matrix

        token_ids = np.concatenate(
            [_normalize_width(m[0]) for _, m in prepared], axis=0
        )
        values = np.concatenate([_normalize_width(m[1]) for _, m in prepared], axis=0)
        qualifying = np.concatenate([m[2] for _, m in prepared], axis=0)
        bin_mins = np.concatenate([m[3] for _, m in prepared], axis=0)
        bin_maxes = np.concatenate([m[4] for _, m in prepared], axis=0)
        bin_contains = np.concatenate([m[5] for _, m in prepared], axis=0)
        feature_indices_per_row = np.repeat(
            np.array([feature_index for feature_index, _ in prepared], dtype=np.int64),
            row_counts,
        )
        sequence_indices_per_row = np.concatenate(
            [np.arange(count, dtype=np.int64) for count in row_counts]
        )

        combined = SequenceCoordinateTable._activation_row_record_batch_from_matrices(
            token_ids=token_ids,
            values=values,
            feature_indices_per_row=feature_indices_per_row,
            sequence_indices_per_row=sequence_indices_per_row,
            qualifying_token_indices=qualifying,
            bin_mins=bin_mins,
            bin_maxes=bin_maxes,
            bin_contains=bin_contains,
            decode_token_ids=decode_token_ids,
            pad_token_id=pad_token_id,
            decode_token_ids_array=decode_token_ids_array,
        )
        slices: list[Any] = []
        offset = 0
        for count in row_counts:
            slices.append(combined.slice(offset, count))
            offset += count
        return slices

    def write_sequence_row_arrow_ipc(
        self, path: str | PathLike[str], *, feature_index: int
    ) -> int:
        pyarrow, pyarrow_ipc, _ = _load_sequence_row_pyarrow_modules()
        table = self.to_sequence_row_arrow_table(feature_index)
        with pyarrow.OSFile(str(path), "wb") as sink:
            with pyarrow_ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        return int(table.num_rows)

    def write_sequence_row_parquet(
        self,
        path: str | PathLike[str],
        *,
        feature_index: int,
        **parquet_writer_kwargs: Any,
    ) -> int:
        _, _, pyarrow_parquet = _load_sequence_row_pyarrow_modules()
        table = self.to_sequence_row_arrow_table(feature_index)
        pyarrow_parquet.write_table(table, str(path), **parquet_writer_kwargs)
        return int(table.num_rows)

    def to_sequence_multi_group_data(self) -> SequenceMultiGroupData:
        token_ids_list = self.token_ids.tolist()
        feat_acts_list = self.feat_acts.tolist()
        token_logits_list = self.token_logits.tolist()
        original_indices = self.original_indices.tolist()
        qualifying_token_indices = self.qualifying_token_indices.tolist()
        group_offsets = self.group_offsets

        sequence_groups_data = []
        for group_idx, group_name in enumerate(self.group_names):
            seq_data = [
                SequenceData(
                    original_index=int(original_indices[i]),
                    token_ids=token_ids_list[i],
                    feat_acts=feat_acts_list[i],
                    token_logits=token_logits_list[i],
                    qualifying_token_index=int(qualifying_token_indices[i]),
                )
                for i in range(group_offsets[group_idx], group_offsets[group_idx + 1])
            ]
            sequence_groups_data.append(SequenceGroupData(group_name, seq_data))

        return SequenceMultiGroupData(sequence_groups_data)


class SequenceDataGenerator:
    cfg: SaeVisConfig | VectorVisConfig
    seq_cfg: SequencesConfig

    def __init__(
        self,
        cfg: SaeVisConfig | VectorVisConfig,
        tokens: Int[Tensor, "batch seq"],
        W_U: Float[Tensor, "d_model d_vocab"],
    ):
        self.cfg = cfg
        assert self.cfg.feature_centric_layout.seq_cfg is not None
        self.seq_cfg = self.cfg.feature_centric_layout.seq_cfg
        self.tokens = tokens
        self.W_U = W_U

        self.buffer, self.padded_buffer_width, self.seq_length = (
            self.get_buffer_and_padding(tokens)
        )
        self._candidate_index_cache: dict[
            tuple[object, ...], tuple[Tensor, Tensor, Tensor]
        ] = {}
        self.reset_profile_stats()

    def reset_profile_stats(self) -> None:
        self._profile_totals: dict[str, float] = {
            "feature_calls": 0.0,
            "candidate_token_count_total": 0.0,
            "candidate_positive_count_total": 0.0,
            "candidate_zero_count_total": 0.0,
            "candidate_negative_count_total": 0.0,
            "lowest_interval_count_total": 0.0,
            "largest_interval_count_total": 0.0,
            "interval_candidate_count_total": 0.0,
            "nonempty_interval_group_count_total": 0.0,
            "sampled_index_count_total": 0.0,
            "mask_setup_wall_s": 0.0,
            "candidate_extract_wall_s": 0.0,
            "topk_wall_s": 0.0,
            "interval_scan_wall_s": 0.0,
            "interval_where_wall_s": 0.0,
            "interval_sample_wall_s": 0.0,
            "indices_concat_wall_s": 0.0,
            "get_indices_dict_wall_s": 0.0,
        }
        self._profile_candidate_token_count_min: int | None = None
        self._profile_candidate_token_count_max = 0
        self._profile_lowest_interval_count_min: int | None = None
        self._profile_lowest_interval_count_max = 0

    def consume_profile_stats(self) -> dict[str, float | int]:
        feature_calls = int(self._profile_totals["feature_calls"])
        if feature_calls == 0:
            return {}

        def mean(key: str) -> float:
            return round(self._profile_totals[key] / feature_calls, 6)

        get_indices_dict_wall_s = self._profile_totals["get_indices_dict_wall_s"]
        interval_scan_wall_s = self._profile_totals["interval_scan_wall_s"]
        summary: dict[str, float | int] = {
            "feature_calls": feature_calls,
            "candidate_token_count_mean": mean("candidate_token_count_total"),
            "candidate_token_count_min": int(
                self._profile_candidate_token_count_min or 0
            ),
            "candidate_token_count_max": int(self._profile_candidate_token_count_max),
            "candidate_positive_count_mean": mean("candidate_positive_count_total"),
            "candidate_zero_count_mean": mean("candidate_zero_count_total"),
            "candidate_negative_count_mean": mean("candidate_negative_count_total"),
            "lowest_interval_count_mean": mean("lowest_interval_count_total"),
            "lowest_interval_count_min": int(
                self._profile_lowest_interval_count_min or 0
            ),
            "lowest_interval_count_max": int(self._profile_lowest_interval_count_max),
            "largest_interval_count_mean": mean("largest_interval_count_total"),
            "interval_candidate_count_mean": mean("interval_candidate_count_total"),
            "nonempty_interval_group_count_mean": mean(
                "nonempty_interval_group_count_total"
            ),
            "sampled_index_count_mean": mean("sampled_index_count_total"),
            "mask_setup_wall_s": round(self._profile_totals["mask_setup_wall_s"], 6),
            "candidate_extract_wall_s": round(
                self._profile_totals["candidate_extract_wall_s"], 6
            ),
            "topk_wall_s": round(self._profile_totals["topk_wall_s"], 6),
            "interval_scan_wall_s": round(interval_scan_wall_s, 6),
            "interval_where_wall_s": round(
                self._profile_totals["interval_where_wall_s"], 6
            ),
            "interval_sample_wall_s": round(
                self._profile_totals["interval_sample_wall_s"], 6
            ),
            "indices_concat_wall_s": round(
                self._profile_totals["indices_concat_wall_s"], 6
            ),
            "get_indices_dict_wall_s": round(get_indices_dict_wall_s, 6),
            "interval_scan_share": (
                round(interval_scan_wall_s / get_indices_dict_wall_s, 6)
                if get_indices_dict_wall_s > 0.0
                else 0.0
            ),
        }
        self.reset_profile_stats()
        return summary

    @torch.inference_mode()
    def get_sequences_data(
        self,
        feat_acts: Float[Tensor, "batch seq"],
        feat_logits: Float[Tensor, "d_vocab"],
        resid_post: Float[Tensor, "batch seq d_model"],
        feature_resid_dir: Float[Tensor, "d_model"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
        selection_backend: SequenceSelectionBackend = "legacy",
    ) -> SequenceMultiGroupData:
        sequence_coordinate_table = self.get_sequence_coordinate_table(
            feat_acts=feat_acts,
            feat_logits=feat_logits,
            resid_post=resid_post,
            feature_resid_dir=feature_resid_dir,
            selection_mask=selection_mask,
            selection_backend=selection_backend,
        )
        return sequence_coordinate_table.to_sequence_multi_group_data()

    @torch.inference_mode()
    def get_sequence_coordinate_table(
        self,
        feat_acts: Float[Tensor, "batch seq"],
        feat_logits: Float[Tensor, "d_vocab"],
        resid_post: Float[Tensor, "batch seq d_model"],
        feature_resid_dir: Float[Tensor, "d_model"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
        selection_backend: SequenceSelectionBackend = "legacy",
        precomputed_selection: tuple[dict[str, Tensor], Tensor, int] | None = None,
    ) -> SequenceCoordinateTable:
        """
        This function returns the compact selected-sequence table which underlies the right-hand sequence visualizations.
        It is the deferred-construction boundary for sequence packaging; callers that still need the legacy nested
        dataclass graph can call `to_sequence_multi_group_data()` on the returned table.

        This is a multi-step process (the 4 steps are annotated in the code):

            (1) Find all the token groups (i.e. topk, bottomk, and quantile groups of activations). These are bold tokens.
            (2) Get the indices of all tokens we'll need data from, which includes a buffer around each bold token.
            (3) Extract the token IDs, feature activations & residual stream values for those positions
            (4) Compute the logit effect if this feature is ablated
                (4A) Use this to compute the most affected tokens by this feature (i.e. the vis hoverdata)
                (4B) Use this to compute the loss effect if this feature is ablated (i.e. the blue/red underlining)
            (5) Return all this data as a SequenceCoordinateTable object

        Args:
            tokens:
                The tokens we'll be extracting sequence data from.
            feat_acts:
                The activations of the feature we're interested in, for each token in the batch.
            feat_logits:
                The logit vector for this feature (used to generate histogram, and is needed here for the line-on-hover).
            resid_post:
                The residual stream values before final layernorm, for each token in the batch.
            feature_resid_dir:
                The direction this feature writes to the logit output (i.e. the direction we'll be erasing from resid_post).
            selection_mask:
                Optional mask selecting valid token positions for padded prompt batches.
            selection_backend:
                Candidate-selection backend to use before compact sequence table construction. The default keeps the
                preserved legacy JSON CPU selector; `"columnar_gpu"` enables the guarded candidate-vector substitute path.
            precomputed_selection:
                Optional `(indices_dict, indices_bold, n_bold)` triple produced by
                `get_indices_dicts_columnar_gpu_batched`; when provided, step (1) is skipped and the supplied
                selection is used directly.

        Returns:
            SequenceCoordinateTable
                Compact selected-sequence rows grouped by the original sequence groups. This can be adapted back into the
                legacy `SequenceMultiGroupData` surface when needed.
        """

        # ! (1) Find the tokens from each group
        if precomputed_selection is not None:
            indices_dict, indices_bold, n_bold = precomputed_selection
        else:
            indices_dict, indices_bold, n_bold = self._get_indices_dict_for_backend(
                selection_backend,
                self.buffer,
                feat_acts,
                selection_mask=selection_mask,
            )

        # ! (2) Get the buffer indices
        indices_buf = self.get_indices_buf(
            indices_bold=indices_bold,
            seq_length=self.seq_length,
            n_bold=n_bold,
            padded_buffer_width=self.padded_buffer_width,
        )

        # ! (3) Extract the token IDs, feature activations & residual stream values for those positions
        # Get the tokens which will be in our sequences
        token_ids = eindex(
            self.tokens, indices_buf[:, 1:], "[n_bold seq 0] [n_bold seq 1]"
        )  # shape [batch buf]
        source_token_indices = indices_buf[:, 1:, 1]

        # Now, we split into cases depending on whether we're computing the buffer or not. One kinda weird thing: we get
        # feature acts for 2 different reasons (token coloring & ablation), and in the case where we're computing the buffer
        # we need [:, 1:] for coloring and [:, :-1] for ablation, but when we're not we only need [:, bold] for both. So
        # we split on cases here.
        (
            _,
            feat_acts_coloring,
            #  resid_post_pre_ablation,
            _,
        ) = self.index_objects_for_ablation_experiments(
            token_ids=token_ids,
            tokens=self.tokens,
            feat_acts=feat_acts,
            resid_post=resid_post,
            indices_bold=indices_bold,
            indices_buf=indices_buf,
        )

        if self.cfg.perform_ablation_experiments:
            raise NotImplementedError(
                "We are not supporting ablation experiments for now."
            )
        else:
            # ! (5) Store the results in a compact sequence coordinate table
            sequence_coordinate_table = self.build_sequence_coordinate_table(
                token_ids=token_ids,
                feat_acts_coloring=feat_acts_coloring,
                feat_logits=feat_logits,
                indices_dict=indices_dict,
                indices_bold=indices_bold,
                source_token_indices=source_token_indices,
            )

        return sequence_coordinate_table

    def get_buffer_and_padding(
        self,
        tokens: Int[Tensor, "batch seq"],
    ):
        # Get buffer, s.t. we're looking for bold tokens in the range `buffer[0] : buffer[1]`. For each bold token, we need
        # to see `seq_cfg.buffer[0]+1` behind it (plus 1 because we need the prev token to compute loss effect), and we need
        # to see `seq_cfg.buffer[1]` ahead of it.
        buffer = (
            (self.seq_cfg.buffer[0] + 1, -self.seq_cfg.buffer[1])
            if self.seq_cfg.buffer is not None
            else None
        )
        _batch_size, seq_length = tokens.shape
        padded_buffer_width = (
            self.seq_cfg.buffer[0] + self.seq_cfg.buffer[1] + 2
            if self.seq_cfg.buffer is not None
            else seq_length
        )

        return buffer, padded_buffer_width, seq_length

    def get_indices_dict(
        self,
        buffer: tuple[int, int] | None,
        feat_acts: Float[Tensor, "batch seq"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
    ):
        return self.get_indices_dict_legacy(
            buffer, feat_acts, selection_mask=selection_mask
        )

    def _get_indices_dict_for_backend(
        self,
        selection_backend: SequenceSelectionBackend,
        buffer: tuple[int, int] | None,
        feat_acts: Float[Tensor, "batch seq"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
    ):
        if selection_backend == "legacy":
            return self.get_indices_dict(
                buffer, feat_acts, selection_mask=selection_mask
            )
        if selection_backend == "columnar_gpu":
            return self.get_indices_dict_columnar_gpu(
                buffer, feat_acts, selection_mask=selection_mask
            )
        raise ValueError(f"Unsupported sequence selection backend: {selection_backend}")

    def _get_legacy_selection_views(
        self,
        feat_acts: Float[Tensor, "batch seq"],
        buffer: tuple[int, int] | None,
        selection_mask: Int[Tensor, "batch seq"] | None = None,
    ) -> tuple[Tensor, Tensor | None, int]:
        if buffer is None:
            feat_acts_view = feat_acts
            selection_mask_view = selection_mask
            col_offset = 0
        else:
            feat_acts_view = feat_acts[:, buffer[0] : buffer[1]]
            selection_mask_view = (
                None
                if selection_mask is None
                else selection_mask[:, buffer[0] : buffer[1]]
            )
            col_offset = buffer[0]

        if selection_mask_view is not None:
            selection_mask_view = selection_mask_view.to(
                device=feat_acts_view.device,
                dtype=torch.bool,
            )

        return feat_acts_view, selection_mask_view, col_offset

    def get_indices_dict_legacy(
        self,
        buffer: tuple[int, int] | None,
        feat_acts: Float[Tensor, "batch seq"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
    ):
        profile_enabled = bool(getattr(self.cfg, "log_performance", False))
        get_indices_dict_start = perf_counter() if profile_enabled else 0.0

        feat_acts_view, selection_mask_view, col_offset = (
            self._get_legacy_selection_views(
                feat_acts,
                buffer,
                selection_mask=selection_mask,
            )
        )

        mask_setup_wall_s = 0.0
        candidate_extract_wall_s = 0.0
        if selection_mask is None:
            candidate_values = feat_acts_view.reshape(-1)
            feat_max = float(feat_acts.max().item()) if feat_acts.numel() > 0 else 0.0
        else:
            mask_setup_start = perf_counter() if profile_enabled else 0.0
            _, _, candidate_flat_indices = self._get_candidate_mask_and_indices(
                feat_acts,
                buffer,
                selection_mask,
            )
            mask_setup_wall_s = (
                perf_counter() - mask_setup_start if profile_enabled else 0.0
            )

            candidate_extract_start = perf_counter() if profile_enabled else 0.0
            candidate_values = feat_acts.reshape(-1)[candidate_flat_indices]
            feat_max = (
                float(candidate_values.max().item())
                if candidate_values.numel() > 0
                else 0.0
            )
            candidate_extract_wall_s = (
                perf_counter() - candidate_extract_start if profile_enabled else 0.0
            )

        candidate_token_count = int(candidate_values.numel())
        candidate_positive_count = 0
        candidate_zero_count = 0
        candidate_negative_count = 0
        if profile_enabled and candidate_token_count > 0:
            candidate_positive_count = int((candidate_values > 0).sum().item())
            candidate_zero_count = int((candidate_values == 0).sum().item())
            candidate_negative_count = (
                candidate_token_count - candidate_positive_count - candidate_zero_count
            )

        # Get the top-activating tokens
        topk_start = perf_counter() if profile_enabled else 0.0
        if candidate_values.numel() > 0:
            top_k = min(self.seq_cfg.top_acts_group_size, candidate_values.numel())
            if selection_mask_view is None:
                top_indices = k_largest_indices(
                    feat_acts,
                    k=top_k,
                    buffer=buffer,
                ).cpu()
            else:
                masked_feat_acts = feat_acts_view.masked_fill(
                    ~selection_mask_view,
                    float("-inf"),
                )
                flat_top_indices = (
                    masked_feat_acts.flatten()
                    .topk(
                        k=top_k,
                        largest=True,
                    )
                    .indices
                )
                top_indices = torch.stack(
                    (
                        flat_top_indices // masked_feat_acts.size(1),
                        flat_top_indices % masked_feat_acts.size(1) + col_offset,
                    ),
                    dim=1,
                ).cpu()
        else:
            top_indices = torch.zeros((0, 2), dtype=torch.long)
        topk_wall_s = perf_counter() - topk_start if profile_enabled else 0.0
        top_group_max = (
            float(feat_acts.max().item()) if selection_mask is None else feat_max
        )
        indices_dict = {f"TOP ACTIVATIONS<br>MAX = {top_group_max:.3f}": top_indices}

        # Get all possible indices. Note, we need to be able to look 1 back (feature activation on prev token is needed for
        # computing loss effect on this token)
        interval_scan_wall_s = 0.0
        interval_where_wall_s = 0.0
        interval_sample_wall_s = 0.0
        lowest_interval_count = 0
        largest_interval_count = 0
        interval_candidate_count = 0
        nonempty_interval_group_count = 0
        sampled_index_count = top_indices.shape[0]
        if self.seq_cfg.n_quantiles > 0:
            interval_scan_start = perf_counter() if profile_enabled else 0.0
            quantile_max = top_group_max
            quantiles = self._build_interval_quantiles(quantile_max, feat_acts.device)
            feat_acts_view_for_intervals = feat_acts_view.to(dtype=quantiles.dtype)
            valid_token_count = max(1, candidate_token_count)
            full_feat_acts_for_intervals = feat_acts.to(dtype=quantiles.dtype)
            for i in range(self.seq_cfg.n_quantiles - 1, -1, -1):
                lower = float(quantiles[i].item())
                upper = float(quantiles[i + 1].item())
                upper_closed = self._interval_upper_closed(i)
                interval_where_start = perf_counter() if profile_enabled else 0.0
                interval_member_mask = (feat_acts_view_for_intervals >= lower) & (
                    (feat_acts_view_for_intervals <= upper)
                    if upper_closed
                    else (feat_acts_view_for_intervals < upper)
                )
                if selection_mask_view is not None:
                    interval_member_mask &= selection_mask_view
                interval_count = int(interval_member_mask.sum().item())
                if selection_mask is None:
                    pct = float(
                        (
                            (full_feat_acts_for_intervals >= lower)
                            & (
                                (full_feat_acts_for_intervals <= upper)
                                if upper_closed
                                else (full_feat_acts_for_intervals < upper)
                            )
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    if profile_enabled:
                        interval_where_wall_s += perf_counter() - interval_where_start
                        interval_candidate_count += interval_count
                        largest_interval_count = max(
                            largest_interval_count, interval_count
                        )
                        if i == 0:
                            lowest_interval_count = interval_count
                        if interval_count > 0:
                            nonempty_interval_group_count += 1
                    interval_sample_start = perf_counter() if profile_enabled else 0.0
                    indices = random_range_indices(
                        feat_acts,
                        k=self.seq_cfg.quantile_group_size,
                        bounds=(lower, upper),
                        buffer=buffer,
                        upper_inclusive=upper_closed,
                    ).cpu()
                    if profile_enabled:
                        interval_sample_wall_s += perf_counter() - interval_sample_start
                else:
                    pct = interval_count / valid_token_count
                    indices = torch.stack(torch.where(interval_member_mask), dim=-1)
                    if indices.numel() > 0 and col_offset != 0:
                        indices = indices + torch.tensor(
                            [0, col_offset],
                            device=indices.device,
                        )
                    if profile_enabled:
                        interval_where_wall_s += perf_counter() - interval_where_start
                        interval_candidate_count += interval_count
                        largest_interval_count = max(
                            largest_interval_count, interval_count
                        )
                        if i == 0:
                            lowest_interval_count = interval_count
                        if interval_count > 0:
                            nonempty_interval_group_count += 1
                    if interval_count > self.seq_cfg.quantile_group_size:
                        interval_sample_start = (
                            perf_counter() if profile_enabled else 0.0
                        )
                        indices = indices[
                            sample_unique_indices(
                                interval_count,
                                self.seq_cfg.quantile_group_size,
                            ).to(indices.device)
                        ]
                        if profile_enabled:
                            interval_sample_wall_s += (
                                perf_counter() - interval_sample_start
                            )
                sampled_index_count += int(indices.shape[0])
                indices_dict[
                    f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
                ] = indices.cpu()
            interval_scan_wall_s = (
                perf_counter() - interval_scan_start if profile_enabled else 0.0
            )

        # Concat all the indices together (in the next steps we do all groups at once). Shape of this object is [n_bold 2],
        # i.e. the [i, :]-th element are the batch and sequence dimensions for the i-th bold token.
        indices_concat_start = perf_counter() if profile_enabled else 0.0
        indices_bold = torch.concat(list(indices_dict.values())).cpu()
        n_bold = indices_bold.shape[0]
        indices_concat_wall_s = (
            perf_counter() - indices_concat_start if profile_enabled else 0.0
        )

        if profile_enabled:
            self._profile_totals["feature_calls"] += 1.0
            self._profile_totals["candidate_token_count_total"] += candidate_token_count
            self._profile_totals[
                "candidate_positive_count_total"
            ] += candidate_positive_count
            self._profile_totals["candidate_zero_count_total"] += candidate_zero_count
            self._profile_totals[
                "candidate_negative_count_total"
            ] += candidate_negative_count
            self._profile_totals["lowest_interval_count_total"] += lowest_interval_count
            self._profile_totals[
                "largest_interval_count_total"
            ] += largest_interval_count
            self._profile_totals[
                "interval_candidate_count_total"
            ] += interval_candidate_count
            self._profile_totals[
                "nonempty_interval_group_count_total"
            ] += nonempty_interval_group_count
            self._profile_totals["sampled_index_count_total"] += sampled_index_count
            self._profile_totals["mask_setup_wall_s"] += mask_setup_wall_s
            self._profile_totals["candidate_extract_wall_s"] += candidate_extract_wall_s
            self._profile_totals["topk_wall_s"] += topk_wall_s
            self._profile_totals["interval_scan_wall_s"] += interval_scan_wall_s
            self._profile_totals["interval_where_wall_s"] += interval_where_wall_s
            self._profile_totals["interval_sample_wall_s"] += interval_sample_wall_s
            self._profile_totals["indices_concat_wall_s"] += indices_concat_wall_s
            self._profile_totals["get_indices_dict_wall_s"] += (
                perf_counter() - get_indices_dict_start
            )
            if self._profile_candidate_token_count_min is None:
                self._profile_candidate_token_count_min = candidate_token_count
            else:
                self._profile_candidate_token_count_min = min(
                    self._profile_candidate_token_count_min,
                    candidate_token_count,
                )
            self._profile_candidate_token_count_max = max(
                self._profile_candidate_token_count_max,
                candidate_token_count,
            )
            if self._profile_lowest_interval_count_min is None:
                self._profile_lowest_interval_count_min = lowest_interval_count
            else:
                self._profile_lowest_interval_count_min = min(
                    self._profile_lowest_interval_count_min,
                    lowest_interval_count,
                )
            self._profile_lowest_interval_count_max = max(
                self._profile_lowest_interval_count_max,
                lowest_interval_count,
            )

        return indices_dict, indices_bold, n_bold

    def _sample_dense_interval_candidate_indices(
        self,
        candidate_indices: Tensor,
        interval_member_mask: Tensor,
        interval_count: int,
    ) -> Tensor:
        group_size = self.seq_cfg.quantile_group_size
        candidate_count = int(interval_member_mask.numel())
        if interval_count <= group_size:
            return candidate_indices[interval_member_mask]

        density = interval_count / max(1, candidate_count)
        draw_count = min(
            candidate_count,
            max(group_size * 4, int((group_size / max(density, 1e-6)) * 2) + 1),
        )
        sampled_positions: list[int] = []
        seen_positions: set[int] = set()
        while (
            len(sampled_positions) < group_size
            and len(seen_positions) < candidate_count
        ):
            positions = sample_unique_indices(candidate_count, draw_count).to(
                interval_member_mask.device
            )
            accepted_positions = positions[interval_member_mask[positions]]
            for position in accepted_positions.tolist():
                if position in seen_positions:
                    continue
                seen_positions.add(position)
                sampled_positions.append(position)
                if len(sampled_positions) == group_size:
                    break
            if len(sampled_positions) < group_size:
                if draw_count == candidate_count:
                    break
                draw_count = min(candidate_count, draw_count * 2)

        if len(sampled_positions) < group_size:
            interval_positions = torch.where(interval_member_mask)[0]
            existing = set(sampled_positions)
            for position in interval_positions.tolist():
                if position in existing:
                    continue
                sampled_positions.append(position)
                if len(sampled_positions) == group_size:
                    break

        sampled_positions_tensor = torch.tensor(
            sampled_positions, device=candidate_indices.device
        )
        return candidate_indices[sampled_positions_tensor]

    def _get_candidate_mask_and_indices(
        self,
        feat_acts: Tensor,
        buffer: tuple[int, int] | None,
        selection_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        selection_mask_key: tuple[object, ...]
        if selection_mask is None:
            selection_mask_key = (None,)
        else:
            try:
                selection_mask_version = int(selection_mask._version)
            except RuntimeError:
                selection_mask_version = 0
            selection_mask_key = (
                int(selection_mask.data_ptr()),
                tuple(selection_mask.shape),
                str(selection_mask.device),
                str(selection_mask.dtype),
                selection_mask_version,
            )
        cache_key = (
            tuple(feat_acts.shape),
            str(feat_acts.device),
            buffer,
            *selection_mask_key,
        )
        cached = self._candidate_index_cache.get(cache_key)
        if cached is not None:
            return cached

        if selection_mask is None:
            candidate_mask = torch.ones_like(feat_acts, dtype=torch.bool)
        else:
            candidate_mask = selection_mask.to(
                device=feat_acts.device, dtype=torch.bool
            ).clone()

        if buffer is not None:
            if selection_mask is None:
                candidate_mask.fill_(False)
                candidate_mask[:, buffer[0] : buffer[1]] = True
            else:
                buffer_mask = torch.zeros_like(candidate_mask)
                buffer_mask[:, buffer[0] : buffer[1]] = True
                candidate_mask &= buffer_mask

        candidate_flat_indices = torch.where(candidate_mask.reshape(-1))[0]
        candidate_indices = torch.stack(
            torch.unravel_index(candidate_flat_indices, candidate_mask.shape),
            dim=-1,
        )
        if len(self._candidate_index_cache) >= 4:
            self._candidate_index_cache.clear()
        self._candidate_index_cache[cache_key] = (
            candidate_mask,
            candidate_indices,
            candidate_flat_indices,
        )
        return candidate_mask, candidate_indices, candidate_flat_indices

    def _build_interval_quantiles(
        self, feat_max: float, device: torch.device
    ) -> Tensor:
        return torch.linspace(
            0,
            feat_max,
            self.seq_cfg.n_quantiles + 1,
            device=device,
            dtype=torch.float32,
        )

    def _build_lazy_interval_position_index(
        self,
        candidate_values: Tensor,
        feat_max: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        quantiles = self._build_interval_quantiles(feat_max, candidate_values.device)
        if candidate_values.numel() == 0:
            empty_counts = torch.zeros(
                self.seq_cfg.n_quantiles,
                dtype=torch.long,
                device=candidate_values.device,
            )
            empty_offsets = torch.zeros(
                self.seq_cfg.n_quantiles + 1,
                dtype=torch.long,
                device=candidate_values.device,
            )
            empty_positions = torch.zeros(
                0,
                dtype=torch.long,
                device=candidate_values.device,
            )
            return quantiles, empty_positions, empty_offsets, empty_counts

        candidate_values_for_intervals = candidate_values.to(dtype=quantiles.dtype)
        lower_membership = candidate_values_for_intervals.unsqueeze(0) >= quantiles[
            :-1
        ].unsqueeze(1)
        upper_membership = candidate_values_for_intervals.unsqueeze(0) <= quantiles[
            1:
        ].unsqueeze(1)
        if (
            getattr(self.cfg, "sequence_half_open_interval_bins", False)
            and self.seq_cfg.n_quantiles > 1
        ):
            strict_upper = candidate_values_for_intervals.unsqueeze(0) < quantiles[
                1:
            ].unsqueeze(1)
            upper_membership = torch.cat(
                [strict_upper[:-1], upper_membership[-1:]], dim=0
            )
        interval_membership = lower_membership & upper_membership
        interval_ids, interval_positions = torch.where(interval_membership)
        interval_counts = torch.bincount(
            interval_ids,
            minlength=self.seq_cfg.n_quantiles,
        )
        interval_offsets = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=torch.long,
                    device=candidate_values.device,
                ),
                interval_counts.cumsum(dim=0),
            ]
        )
        return quantiles, interval_positions, interval_offsets, interval_counts

    def _interval_upper_closed(self, interval_index: int) -> bool:
        """Whether interval ``interval_index``'s upper bound is inclusive.

        Historical (default) semantics are double-inclusive on every interval. With
        ``sequence_half_open_interval_bins`` each interval is ``[lower, upper)`` except the
        highest, which stays closed so the feature max remains a member (numpy histogram
        convention) — every value then belongs to exactly one interval.
        """
        if not getattr(self.cfg, "sequence_half_open_interval_bins", False):
            return True
        return interval_index == self.seq_cfg.n_quantiles - 1

    @staticmethod
    def _filter_unselected_positions(
        positions: _PositionArrayT, selected_mask: _PositionArrayT | None
    ) -> _PositionArrayT:
        """Drop positions already claimed by an earlier group (no-op when dedup is off)."""
        if selected_mask is None:
            return positions
        return positions[~selected_mask[positions]]  # pyright: ignore[reportReturnType]

    @staticmethod
    def _build_feature_selected_mask(
        dedup_across_groups: bool,
        interval_positions_flat: Tensor | None,
        candidate_token_count: int,
        top_feature_positions: Tensor,
    ) -> Tensor | None:
        """Per-feature cross-group exclusion mask seeded with the TOP-group selections."""
        if (
            not dedup_across_groups
            or interval_positions_flat is None
            or candidate_token_count == 0
        ):
            return None
        selected_mask = torch.zeros(
            candidate_token_count,
            dtype=torch.bool,
            device=interval_positions_flat.device,
        )
        selected_mask[top_feature_positions.to(interval_positions_flat.device)] = True
        return selected_mask

    @staticmethod
    def _mark_selected_positions(
        positions: _PositionArrayT, selected_mask: _PositionArrayT | None
    ) -> None:
        """Record positions in the running cross-group exclusion mask (no-op when dedup is off)."""
        if selected_mask is not None:
            selected_mask[positions] = True

    def get_indices_dict_columnar_gpu(
        self,
        buffer: tuple[int, int] | None,
        feat_acts: Float[Tensor, "batch seq"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
    ):
        profile_enabled = bool(getattr(self.cfg, "log_performance", False))
        get_indices_dict_start = perf_counter() if profile_enabled else 0.0

        mask_setup_start = perf_counter() if profile_enabled else 0.0
        _candidate_mask, candidate_indices, candidate_flat_indices = (
            self._get_candidate_mask_and_indices(
                feat_acts,
                buffer,
                selection_mask,
            )
        )
        mask_setup_wall_s = (
            perf_counter() - mask_setup_start if profile_enabled else 0.0
        )

        candidate_extract_start = perf_counter() if profile_enabled else 0.0
        candidate_values = feat_acts.reshape(-1)[candidate_flat_indices]
        feat_max = (
            float(candidate_values.max().item())
            if candidate_values.numel() > 0
            else 0.0
        )
        candidate_extract_wall_s = (
            perf_counter() - candidate_extract_start if profile_enabled else 0.0
        )

        top_positive_only = bool(
            getattr(self.cfg, "sequence_top_acts_positive_only", False)
        )
        dedup_across_groups = bool(
            getattr(self.cfg, "sequence_dedup_across_groups", False)
        )
        if getattr(self.cfg, "sequence_skip_dead_features", False) and feat_max <= 0.0:
            empty_indices = torch.zeros((0, 2), dtype=torch.long)
            indices_dict = {f"TOP ACTIVATIONS<br>MAX = {feat_max:.3f}": empty_indices}
            return indices_dict, empty_indices.clone(), 0

        candidate_token_count = int(candidate_values.numel())
        candidate_positive_count = 0
        candidate_zero_count = 0
        candidate_negative_count = 0
        if profile_enabled and candidate_token_count > 0:
            candidate_positive_count = int((candidate_values > 0).sum().item())
            candidate_zero_count = int((candidate_values == 0).sum().item())
            candidate_negative_count = (
                candidate_token_count - candidate_positive_count - candidate_zero_count
            )

        topk_start = perf_counter() if profile_enabled else 0.0
        if candidate_values.numel() > 0:
            top_k = min(self.seq_cfg.top_acts_group_size, candidate_values.numel())
            top_positions = candidate_values.topk(k=top_k, largest=True).indices
            if top_positive_only:
                # topk is value-descending, so the positive selections are exactly the
                # leading min(top_k, n_positive) entries — no zero/negative tie-fill.
                positive_count = int((candidate_values > 0).sum().item())
                top_positions = top_positions[: min(top_k, positive_count)]
            top_indices = candidate_indices[top_positions].cpu()
        else:
            top_positions = torch.zeros(
                0, dtype=torch.long, device=candidate_values.device
            )
            top_indices = torch.zeros((0, 2), dtype=torch.long)
        topk_wall_s = perf_counter() - topk_start if profile_enabled else 0.0
        indices_dict = {f"TOP ACTIVATIONS<br>MAX = {feat_max:.3f}": top_indices}
        selected_position_mask: Tensor | None = None
        selected_position_mask_np: np.ndarray | None = None
        if dedup_across_groups and candidate_token_count > 0:
            selected_position_mask = torch.zeros(
                candidate_token_count, dtype=torch.bool, device=candidate_values.device
            )
            selected_position_mask[top_positions] = True
            selected_position_mask_np = selected_position_mask.cpu().numpy()

        interval_scan_wall_s = 0.0
        interval_where_wall_s = 0.0
        interval_sample_wall_s = 0.0
        lowest_interval_count = 0
        largest_interval_count = 0
        interval_candidate_count = 0
        nonempty_interval_group_count = 0
        sampled_index_count = top_indices.shape[0]
        if self.seq_cfg.n_quantiles > 0:
            interval_scan_start = perf_counter() if profile_enabled else 0.0
            valid_token_count = max(1, candidate_token_count)
            if candidate_values.device.type == "cpu":
                quantile_values = self._build_interval_quantiles(
                    feat_max, candidate_values.device
                ).tolist()
                candidate_values_np = candidate_values.to(torch.float32).numpy()
                for i in range(self.seq_cfg.n_quantiles - 1, -1, -1):
                    lower, upper = quantile_values[i : i + 2]
                    interval_where_start = perf_counter() if profile_enabled else 0.0
                    interval_positions_np = np.flatnonzero(
                        (candidate_values_np >= lower)
                        & (
                            (candidate_values_np <= upper)
                            if self._interval_upper_closed(i)
                            else (candidate_values_np < upper)
                        )
                    )
                    interval_count = int(interval_positions_np.shape[0])
                    pct = interval_count / valid_token_count
                    if profile_enabled:
                        interval_where_wall_s += perf_counter() - interval_where_start
                        interval_candidate_count += interval_count
                        largest_interval_count = max(
                            largest_interval_count, interval_count
                        )
                        if i == 0:
                            lowest_interval_count = interval_count
                        if interval_count > 0:
                            nonempty_interval_group_count += 1
                    interval_positions_np = self._filter_unselected_positions(
                        interval_positions_np, selected_position_mask_np
                    )
                    pool_count = int(interval_positions_np.shape[0])
                    if pool_count > self.seq_cfg.quantile_group_size:
                        interval_sample_start = (
                            perf_counter() if profile_enabled else 0.0
                        )
                        sampled_relative_positions_np: np.ndarray = (
                            sample_unique_indices(
                                pool_count,
                                self.seq_cfg.quantile_group_size,
                            ).numpy()
                        )
                        interval_positions_np = interval_positions_np[
                            sampled_relative_positions_np
                        ]
                        if profile_enabled:
                            interval_sample_wall_s += (
                                perf_counter() - interval_sample_start
                            )
                    self._mark_selected_positions(
                        interval_positions_np, selected_position_mask_np
                    )
                    indices = candidate_indices[
                        torch.as_tensor(interval_positions_np, dtype=torch.long)
                    ]
                    sampled_index_count += int(indices.shape[0])
                    indices_dict[
                        f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
                    ] = indices
            else:
                interval_lookup_start = perf_counter() if profile_enabled else 0.0
                (
                    quantiles,
                    interval_positions_flat,
                    interval_offsets,
                    interval_counts,
                ) = self._build_lazy_interval_position_index(candidate_values, feat_max)
                if profile_enabled:
                    interval_where_wall_s += perf_counter() - interval_lookup_start
                quantile_values = quantiles.detach().cpu().tolist()
                interval_offsets_list = interval_offsets.detach().cpu().tolist()
                interval_counts_list = interval_counts.detach().cpu().tolist()
                for i in range(self.seq_cfg.n_quantiles - 1, -1, -1):
                    lower, upper = quantile_values[i : i + 2]
                    interval_start = int(interval_offsets_list[i])
                    interval_end = int(interval_offsets_list[i + 1])
                    interval_positions = interval_positions_flat[
                        interval_start:interval_end
                    ]
                    interval_count = int(interval_counts_list[i])
                    pct = interval_count / valid_token_count
                    if profile_enabled:
                        interval_candidate_count += interval_count
                        largest_interval_count = max(
                            largest_interval_count, interval_count
                        )
                        if i == 0:
                            lowest_interval_count = interval_count
                        if interval_count > 0:
                            nonempty_interval_group_count += 1
                    interval_positions = self._filter_unselected_positions(
                        interval_positions, selected_position_mask
                    )
                    pool_count = int(interval_positions.numel())
                    if pool_count > self.seq_cfg.quantile_group_size:
                        interval_sample_start = (
                            perf_counter() if profile_enabled else 0.0
                        )
                        sampled_relative_positions_tensor: Tensor = (
                            sample_unique_indices(
                                pool_count,
                                self.seq_cfg.quantile_group_size,
                            ).to(interval_positions.device)
                        )
                        interval_positions = interval_positions[
                            sampled_relative_positions_tensor
                        ]
                        indices = candidate_indices[interval_positions]
                        if profile_enabled:
                            interval_sample_wall_s += (
                                perf_counter() - interval_sample_start
                            )
                    else:
                        indices = candidate_indices[interval_positions]
                    self._mark_selected_positions(
                        interval_positions, selected_position_mask
                    )
                    sampled_index_count += int(indices.shape[0])
                    indices_dict[
                        f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
                    ] = indices.cpu()
            interval_scan_wall_s = (
                perf_counter() - interval_scan_start if profile_enabled else 0.0
            )

        indices_concat_start = perf_counter() if profile_enabled else 0.0
        indices_bold = torch.concat(list(indices_dict.values())).cpu()
        n_bold = indices_bold.shape[0]
        indices_concat_wall_s = (
            perf_counter() - indices_concat_start if profile_enabled else 0.0
        )

        if profile_enabled:
            self._profile_totals["feature_calls"] += 1.0
            self._profile_totals["candidate_token_count_total"] += candidate_token_count
            self._profile_totals[
                "candidate_positive_count_total"
            ] += candidate_positive_count
            self._profile_totals["candidate_zero_count_total"] += candidate_zero_count
            self._profile_totals[
                "candidate_negative_count_total"
            ] += candidate_negative_count
            self._profile_totals["lowest_interval_count_total"] += lowest_interval_count
            self._profile_totals[
                "largest_interval_count_total"
            ] += largest_interval_count
            self._profile_totals[
                "interval_candidate_count_total"
            ] += interval_candidate_count
            self._profile_totals[
                "nonempty_interval_group_count_total"
            ] += nonempty_interval_group_count
            self._profile_totals["sampled_index_count_total"] += sampled_index_count
            self._profile_totals["mask_setup_wall_s"] += mask_setup_wall_s
            self._profile_totals["candidate_extract_wall_s"] += candidate_extract_wall_s
            self._profile_totals["topk_wall_s"] += topk_wall_s
            self._profile_totals["interval_scan_wall_s"] += interval_scan_wall_s
            self._profile_totals["interval_where_wall_s"] += interval_where_wall_s
            self._profile_totals["interval_sample_wall_s"] += interval_sample_wall_s
            self._profile_totals["indices_concat_wall_s"] += indices_concat_wall_s
            self._profile_totals["get_indices_dict_wall_s"] += (
                perf_counter() - get_indices_dict_start
            )
            if self._profile_candidate_token_count_min is None:
                self._profile_candidate_token_count_min = candidate_token_count
            else:
                self._profile_candidate_token_count_min = min(
                    self._profile_candidate_token_count_min,
                    candidate_token_count,
                )
            self._profile_candidate_token_count_max = max(
                self._profile_candidate_token_count_max,
                candidate_token_count,
            )
            if self._profile_lowest_interval_count_min is None:
                self._profile_lowest_interval_count_min = lowest_interval_count
            else:
                self._profile_lowest_interval_count_min = min(
                    self._profile_lowest_interval_count_min,
                    lowest_interval_count,
                )
            self._profile_lowest_interval_count_max = max(
                self._profile_lowest_interval_count_max,
                lowest_interval_count,
            )

        return indices_dict, indices_bold, n_bold

    @torch.inference_mode()
    def get_indices_dicts_columnar_gpu_batched(
        self,
        buffer: tuple[int, int] | None,
        all_feat_acts: Float[Tensor, "batch seq feats"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
        selection_device: str | torch.device | None = None,
        feature_chunk_size: int = 64,
        staged_flat_acts: Tensor | None = None,
    ) -> list[tuple[dict[str, Tensor], Tensor, int]]:
        """Batched-across-features equivalent of `get_indices_dict_columnar_gpu`.

        Returns one ``(indices_dict, indices_bold, n_bold)`` triple per feature column of
        ``all_feat_acts`` while running the selection math (top-k, interval membership,
        interval sampling) batched per feature chunk on ``selection_device``.

        Selection semantics match the sequential per-feature selector exactly:

        - interval boundaries are float32 CPU ``linspace`` values (bitwise-identical to the
          per-feature CPU interval path used in production), and membership uses the same
          closed-interval ``>= lower & <= upper`` comparisons on float32-widened values
          (or numpy-style half-open bounds when ``sequence_half_open_interval_bins`` is set —
          identical semantics in the sequential and batched selectors);
        - ``sample_unique_indices`` is invoked in the same feature-ascending /
          interval-descending order with the same arguments, so the shared RNG stream is
          consumed identically to a sequential feature loop;
        - group labels, per-group index ordering (top-k order for the top group, ascending
          candidate order for non-overfilled intervals, sample order for overfilled ones),
          and the dict insertion order are unchanged.

        The per-feature candidate composition statistics recorded by the sequential selector
        are intentionally not computed here (they are profiling-only and taxed the stage they
        measured); only the cheap wall/feature aggregates are maintained.

        ``staged_flat_acts`` optionally supplies an already device-resident
        ``[features, batch*seq]`` activation matrix (values must equal ``all_feat_acts`` at
        every candidate position — the unmasked flat acts qualify because candidates are a
        subset of valid positions where masking is the identity); when provided, the
        selection device is taken from it and no per-chunk host-to-device uploads occur.
        """
        profile_enabled = bool(getattr(self.cfg, "log_performance", False))
        batched_start = perf_counter() if profile_enabled else 0.0

        top_positive_only = bool(
            getattr(self.cfg, "sequence_top_acts_positive_only", False)
        )
        dedup_across_groups = bool(
            getattr(self.cfg, "sequence_dedup_across_groups", False)
        )
        skip_dead_features = bool(
            getattr(self.cfg, "sequence_skip_dead_features", False)
        )

        n_features = int(all_feat_acts.shape[-1])
        if n_features == 0:
            return []

        _, candidate_indices, candidate_flat_indices = (
            self._get_candidate_mask_and_indices(
                all_feat_acts[..., 0],
                buffer,
                selection_mask,
            )
        )
        candidate_indices_cpu = candidate_indices.cpu()
        candidate_token_count = int(candidate_flat_indices.numel())
        valid_token_count = max(1, candidate_token_count)
        n_quantiles = self.seq_cfg.n_quantiles
        group_size = self.seq_cfg.quantile_group_size
        top_k = min(self.seq_cfg.top_acts_group_size, candidate_token_count)

        if staged_flat_acts is not None:
            device = staged_flat_acts.device
        elif selection_device is not None:
            device = torch.device(selection_device)
        else:
            device = all_feat_acts.device
        flat_acts = all_feat_acts.reshape(-1, n_features)
        candidate_flat_indices_dev = candidate_flat_indices.to(device)

        selections: list[tuple[dict[str, Tensor], Tensor, int]] = []
        for chunk_start in range(0, n_features, feature_chunk_size):
            chunk_end = min(chunk_start + feature_chunk_size, n_features)
            chunk_width = chunk_end - chunk_start

            chunk_vals: Tensor | None = None
            if candidate_token_count > 0:
                if staged_flat_acts is not None:
                    chunk_vals = (
                        staged_flat_acts[chunk_start:chunk_end]
                        .index_select(-1, candidate_flat_indices_dev)
                        .T
                    )
                else:
                    chunk_vals = flat_acts[:, chunk_start:chunk_end].to(device)[
                        candidate_flat_indices_dev
                    ]
                feat_max_chunk = [
                    float(value)
                    for value in chunk_vals.max(dim=0).values.float().cpu().tolist()
                ]
                if top_k > 0:
                    top_positions_chunk = chunk_vals.topk(k=top_k, dim=0).indices.cpu()
                else:
                    top_positions_chunk = torch.zeros(
                        (0, chunk_width), dtype=torch.long
                    )
                if top_positive_only:
                    positive_counts_chunk = [
                        int(count)
                        for count in (chunk_vals > 0).sum(dim=0).cpu().tolist()
                    ]
                else:
                    positive_counts_chunk = None
            else:
                feat_max_chunk = [0.0] * chunk_width
                top_positions_chunk = torch.zeros((0, chunk_width), dtype=torch.long)
                positive_counts_chunk = [0] * chunk_width if top_positive_only else None

            quantile_values_chunk: list[list[float]] = []
            interval_positions_flat: Tensor | None = None
            interval_offsets: list[int] = []
            interval_counts: list[int] = []
            if n_quantiles > 0:
                boundaries = torch.stack(
                    [
                        self._build_interval_quantiles(feat_max, torch.device("cpu"))
                        for feat_max in feat_max_chunk
                    ]
                )
                quantile_values_chunk = boundaries.tolist()
                if chunk_vals is not None:
                    boundaries_dev = boundaries.to(device)
                    vals32 = chunk_vals.to(torch.float32).T
                    lower_membership = vals32.unsqueeze(1) >= boundaries_dev[
                        :, :-1
                    ].unsqueeze(2)
                    upper_membership = vals32.unsqueeze(1) <= boundaries_dev[
                        :, 1:
                    ].unsqueeze(2)
                    if (
                        getattr(self.cfg, "sequence_half_open_interval_bins", False)
                        and n_quantiles > 1
                    ):
                        strict_upper = vals32.unsqueeze(1) < boundaries_dev[
                            :, 1:
                        ].unsqueeze(2)
                        upper_membership = torch.cat(
                            [strict_upper[:, :-1], upper_membership[:, -1:]], dim=1
                        )
                    membership = lower_membership & upper_membership
                    flat_membership = membership.reshape(chunk_width * n_quantiles, -1)
                    nonzero_pairs = flat_membership.nonzero()
                    group_counts = torch.bincount(
                        nonzero_pairs[:, 0],
                        minlength=chunk_width * n_quantiles,
                    ).cpu()
                    interval_counts = [int(count) for count in group_counts.tolist()]
                    interval_offsets = [0]
                    for count in interval_counts:
                        interval_offsets.append(interval_offsets[-1] + count)
                    interval_positions_flat = nonzero_pairs[:, 1]
                    del membership, lower_membership, upper_membership
                    del flat_membership, vals32, nonzero_pairs
                else:
                    interval_counts = [0] * (chunk_width * n_quantiles)
                    interval_offsets = [0] * (chunk_width * n_quantiles + 1)

            # Per-feature assembly. RNG consumption must match the sequential selector:
            # feature ascending, interval index descending, sampling only on overfill.
            chunk_plans: list[list[tuple[str, Tensor | None, int]]] = []
            selected_position_groups: list[Tensor] = []
            for local_index in range(chunk_width):
                feat_max = feat_max_chunk[local_index]
                if skip_dead_features and feat_max <= 0.0:
                    chunk_plans.append(
                        [
                            (
                                f"TOP ACTIVATIONS<br>MAX = {feat_max:.3f}",
                                torch.zeros(0, dtype=torch.long),
                                -1,
                            )
                        ]
                    )
                    continue
                top_feature_positions = (
                    top_positions_chunk[:, local_index]
                    if top_positions_chunk.numel() > 0
                    else torch.zeros(0, dtype=torch.long)
                )
                if positive_counts_chunk is not None:
                    # topk is value-descending, so the positive selections are exactly the
                    # leading min(top_k, n_positive) entries — no zero/negative tie-fill.
                    top_feature_positions = top_feature_positions[
                        : min(top_k, positive_counts_chunk[local_index])
                    ]
                plan: list[tuple[str, Tensor | None, int]] = [
                    (
                        f"TOP ACTIVATIONS<br>MAX = {feat_max:.3f}",
                        top_feature_positions,
                        -1,
                    )
                ]
                feature_selected_mask = self._build_feature_selected_mask(
                    dedup_across_groups,
                    interval_positions_flat,
                    candidate_token_count,
                    top_feature_positions,
                )
                if n_quantiles > 0:
                    quantile_values = quantile_values_chunk[local_index]
                    for i in range(n_quantiles - 1, -1, -1):
                        lower, upper = quantile_values[i : i + 2]
                        group_index = local_index * n_quantiles + i
                        interval_count = (
                            interval_counts[group_index] if interval_counts else 0
                        )
                        pct = interval_count / valid_token_count
                        label = (
                            f"INTERVAL {lower:.3f} - {upper:.3f}"
                            f"<br>CONTAINS {pct:.3%}"
                        )
                        if interval_positions_flat is None or interval_count == 0:
                            plan.append((label, torch.zeros(0, dtype=torch.long), -1))
                            continue
                        start = interval_offsets[group_index]
                        end = interval_offsets[group_index + 1]
                        group_positions = self._filter_unselected_positions(
                            interval_positions_flat[start:end], feature_selected_mask
                        )
                        pool_count = int(group_positions.numel())
                        if pool_count > group_size:
                            sampled_relative = sample_unique_indices(
                                pool_count,
                                group_size,
                            ).to(group_positions.device)
                            group_positions = group_positions[sampled_relative]
                        self._mark_selected_positions(
                            group_positions, feature_selected_mask
                        )
                        plan.append((label, None, len(selected_position_groups)))
                        selected_position_groups.append(group_positions)
                chunk_plans.append(plan)

            if selected_position_groups:
                group_lengths = [
                    int(group.shape[0]) for group in selected_position_groups
                ]
                selected_positions_cpu = torch.cat(selected_position_groups).cpu()
                selected_groups_cpu = list(selected_positions_cpu.split(group_lengths))
            else:
                selected_groups_cpu = []

            for plan in chunk_plans:
                indices_dict: dict[str, Tensor] = {}
                for label, direct_positions, group_ref in plan:
                    positions = (
                        direct_positions
                        if direct_positions is not None
                        else selected_groups_cpu[group_ref]
                    )
                    if positions.numel() > 0:
                        indices_dict[label] = candidate_indices_cpu[positions]
                    else:
                        indices_dict[label] = torch.zeros((0, 2), dtype=torch.long)
                indices_bold = torch.concat(list(indices_dict.values()))
                selections.append(
                    (indices_dict, indices_bold, int(indices_bold.shape[0]))
                )
            del chunk_vals

        if profile_enabled:
            self._profile_totals["feature_calls"] += float(n_features)
            self._profile_totals["candidate_token_count_total"] += float(
                candidate_token_count * n_features
            )
            self._profile_totals["get_indices_dict_wall_s"] += (
                perf_counter() - batched_start
            )

        return selections

    def get_indices_buf(
        self,
        indices_bold: Int[Tensor, "n_bold 2"],
        seq_length: int,
        n_bold: int,
        padded_buffer_width: int,
    ):
        if self.seq_cfg.buffer is not None:
            # Get the buffer indices, by adding a broadcasted arange object. At this point, indices_buf contains 1 more token
            # than the length of the sequences we'll see (because it also contains the token before the sequence starts).
            buffer_tensor = torch.arange(
                -self.seq_cfg.buffer[0] - 1,
                self.seq_cfg.buffer[1] + 1,
                device=indices_bold.device,
            )
            indices_buf = einops.repeat(
                indices_bold,
                "n_bold two -> n_bold seq two",
                seq=self.seq_cfg.buffer[0] + self.seq_cfg.buffer[1] + 2,
            )
            indices_buf = torch.stack(
                [indices_buf[..., 0], indices_buf[..., 1] + buffer_tensor], dim=-1
            )
        else:
            # If we don't specify a sequence, then do all of the indices.
            indices_buf = torch.stack(
                [
                    einops.repeat(
                        indices_bold[:, 0], "n_bold -> n_bold seq", seq=seq_length
                    ),  # batch indices of bold tokens
                    einops.repeat(
                        torch.arange(seq_length), "seq -> n_bold seq", n_bold=n_bold
                    ),  # all sequence indices
                ],
                dim=-1,
            )

        assert indices_buf.shape == (n_bold, padded_buffer_width, 2)

        return indices_buf

    def index_objects_for_ablation_experiments(
        self,
        token_ids: Int[Tensor, "batch seq"],
        tokens: Int[Tensor, "batch seq"],
        feat_acts: Float[Tensor, "batch seq"],
        resid_post: Float[Tensor, "batch seq d_model"],
        indices_bold: Int[Tensor, "n_bold 2"],
        indices_buf: Int[Tensor, "n_bold buf 2"],
    ):
        if self.seq_cfg.compute_buffer:
            feat_acts_buf = eindex(
                feat_acts,
                indices_buf,
                "[n_bold buf_plus1 0] [n_bold buf_plus1 1] -> n_bold buf_plus1",
            )
            feat_acts_pre_ablation = feat_acts_buf[:, :-1]
            feat_acts_coloring = feat_acts_buf[:, 1:]
            # resid_post_pre_ablation = eindex(
            #     resid_post, indices_buf[:, :-1], "[n_bold buf 0] [n_bold buf 1] d_model"
            # )
            # The tokens we'll use to index correct logits are the same as the ones which will be in our sequence
            correct_tokens = token_ids
        else:
            feat_acts_pre_ablation = eindex(
                feat_acts, indices_bold, "[n_bold 0] [n_bold 1]"
            ).unsqueeze(1)
            feat_acts_coloring = feat_acts_pre_ablation
            # resid_post_pre_ablation = eindex(
            #     resid_post, indices_bold, "[n_bold 0] [n_bold 1] d_model"
            # ).unsqueeze(1)
            # The tokens we'll use to index correct logits are the ones after bold
            indices_bold_next = torch.stack(
                [indices_bold[:, 0], indices_bold[:, 1] + 1], dim=-1
            )
            correct_tokens = eindex(
                tokens, indices_bold_next, "[n_bold 0] [n_bold 1]"
            ).unsqueeze(1)

        return (
            feat_acts_pre_ablation,
            feat_acts_coloring,
            # resid_post_pre_ablation,
            correct_tokens,
        )

    def get_feature_ablation_statistics(
        self,
        feat_acts_pre_ablation: Float[Tensor, "n_bold buf"],
        contribution_to_logprobs: Float[Tensor, "n_bold d_vocab"],
        correct_tokens: Int[Tensor, "n_bold 1"],
    ):
        acts_nonzero = feat_acts_pre_ablation.abs() > 1e-5  # shape [batch buf]
        top_contribution_to_logits = TopK(
            contribution_to_logprobs,
            k=self.seq_cfg.top_logits_hoverdata,
            largest=True,
            tensor_mask=acts_nonzero,
        )
        bottom_contribution_to_logits = TopK(
            contribution_to_logprobs,
            k=self.seq_cfg.top_logits_hoverdata,
            largest=False,
            tensor_mask=acts_nonzero,
        )
        loss_contribution = eindex(
            -contribution_to_logprobs, correct_tokens, "batch seq [batch seq]"
        )

        return (
            top_contribution_to_logits,
            bottom_contribution_to_logits,
            loss_contribution,
        )

    def package_sequences_data(
        self,
        token_ids: Int[Tensor, "n_bold buf"],
        feat_acts_coloring: Float[Tensor, "n_bold buf"],
        feat_logits: Float[Tensor, "d_vocab"],
        indices_dict: dict[str, Int[Tensor, "n_bold 2"]],
        indices_bold: Int[Tensor, "n_bold"],
        loss_contribution: Float[Tensor, "n_bold 1"] | None = None,
        top_contribution_to_logits: TopK | None = None,
        bottom_contribution_to_logits: TopK | None = None,
    ):
        if self.cfg.perform_ablation_experiments:
            raise NotImplementedError(
                "We are not supporting ablation experiments for now."
            )
            # assert isinstance(loss_contribution, torch.Tensor)
            # assert top_contribution_to_logits is not None
            # assert bottom_contribution_to_logits is not None
            # for group_idx, group_name in enumerate(indices_dict.keys()):
            #     seq_data = [
            #         SequenceData(
            #             token_ids=token_ids[i].tolist(),
            #             feat_acts=[round(f, 4) for f in feat_acts_coloring[i].tolist()],
            #             loss_contribution=loss_contribution[i].tolist(),
            #             token_logits=feat_logits[token_ids[i]].tolist(),
            #             top_token_ids=top_contribution_to_logits.indices[i].tolist(),
            #             top_logits=top_contribution_to_logits.values[i].tolist(),
            #             bottom_token_ids=bottom_contribution_to_logits.indices[
            #                 i
            #             ].tolist(),
            #             bottom_logits=bottom_contribution_to_logits.values[i].tolist(),
            #         )
            #         for i in range(
            #             group_sizes_cumsum[group_idx], group_sizes_cumsum[group_idx + 1]
            #         )
            #     ]
            #     sequence_groups_data.append(SequenceGroupData(group_name, seq_data))

        sequence_coordinate_table = self.build_sequence_coordinate_table(
            token_ids=token_ids,
            feat_acts_coloring=feat_acts_coloring,
            feat_logits=feat_logits,
            indices_dict=indices_dict,
            indices_bold=indices_bold,
        )
        return sequence_coordinate_table.to_sequence_multi_group_data()

    def supports_batched_coordinate_tables(self) -> bool:
        """The cross-feature coordinate-table fast path covers the production columnar
        configuration: full-sequence context (``buffer is None``) with buffer computation
        enabled and no ablation experiments."""
        return (
            self.seq_cfg.buffer is None
            and self.seq_cfg.compute_buffer
            and not self.cfg.perform_ablation_experiments
        )

    @torch.inference_mode()
    def build_sequence_coordinate_tables_batched(
        self,
        all_feat_acts: Float[Tensor, "batch seq feats"],
        feat_logits_batch: Float[Tensor, "feats d_vocab"],
        selections: list[tuple[dict[str, Tensor], Tensor, int]],
        ignore_tokens_mask: Tensor | None = None,
    ) -> list[SequenceCoordinateTable]:
        """Cross-feature equivalent of per-feature `get_sequence_coordinate_table` for the
        ``buffer is None`` / ``compute_buffer`` configuration.

        With full-sequence context every selected row's buffered window is simply its whole
        sequence, so the per-feature `get_indices_buf` + `eindex` gathers reduce to row
        selections. This method concatenates all features' selected rows, performs one token
        row-gather, one per-row-feature activation gather, one vectorized ``np.around`` pass,
        and one flat logits gather (on `feat_logits_batch`'s device, transferring only the
        gathered ``[rows, seq-1]`` block), then splits the results back into one
        `SequenceCoordinateTable` per feature. Values are identical to the per-feature path:
        every operation is a pure gather or the same elementwise rounding.
        """
        if not self.supports_batched_coordinate_tables():
            raise ValueError(
                "build_sequence_coordinate_tables_batched requires buffer=None, "
                "compute_buffer=True, and no ablation experiments."
            )
        if not selections:
            return []

        n_bold_per_feature = [selection[2] for selection in selections]
        all_bold = torch.cat([selection[1] for selection in selections]).cpu()
        total_rows = int(all_bold.shape[0])
        seq_length = int(self.tokens.shape[1])
        n_features = len(selections)

        feature_row_ids = torch.repeat_interleave(
            torch.arange(n_features, dtype=torch.long),
            torch.tensor(n_bold_per_feature, dtype=torch.long),
        )

        if total_rows:
            bold_batch_indices = all_bold[:, 0]
            # token ids for positions 1..seq-1 of each selected sequence
            token_ids_all = self.tokens.cpu()[bold_batch_indices][:, 1:]
            # coloring values: the selected feature's activations at positions 1..seq-1.
            # `all_feat_acts` may be unmasked (device-resident staging); applying the
            # ignore mask to the gathered rows is the same elementwise multiply the
            # per-feature path applies before gathering.
            acts_device = all_feat_acts.device
            coloring_all = all_feat_acts[
                bold_batch_indices.to(acts_device),
                :,
                feature_row_ids.to(acts_device),
            ]
            if ignore_tokens_mask is not None:
                coloring_all = coloring_all * ignore_tokens_mask.to(acts_device)[
                    bold_batch_indices.to(acts_device)
                ].to(dtype=coloring_all.dtype)
            coloring_all = coloring_all[:, 1:]
            feat_acts_all = np.around(
                coloring_all.to(dtype=torch.float32)
                .cpu()
                .numpy()
                .astype(np.float64, copy=False),
                4,
            )
            logits_device = feat_logits_batch.device
            d_vocab = int(feat_logits_batch.shape[-1])
            flat_logit_indices = feature_row_ids.to(logits_device)[
                :, None
            ] * d_vocab + token_ids_all.to(logits_device)
            token_logits_all = feat_logits_batch.reshape(-1)[flat_logit_indices].cpu()
            source_token_indices_all = torch.arange(
                1, seq_length, dtype=torch.long
            ).expand(total_rows, seq_length - 1)
        else:
            token_ids_all = torch.zeros((0, seq_length - 1), dtype=torch.long)
            feat_acts_all = np.zeros((0, seq_length - 1), dtype=np.float64)
            token_logits_all = torch.zeros(
                (0, seq_length - 1), dtype=feat_logits_batch.dtype
            )
            source_token_indices_all = torch.zeros(
                (0, seq_length - 1), dtype=torch.long
            )

        token_ids_split = torch.split(token_ids_all, n_bold_per_feature)
        token_logits_split = torch.split(token_logits_all, n_bold_per_feature)
        source_indices_split = torch.split(source_token_indices_all, n_bold_per_feature)
        value_offsets = np.cumsum(n_bold_per_feature)[:-1]
        feat_acts_split = np.split(feat_acts_all, value_offsets)

        tables: list[SequenceCoordinateTable] = []
        for feature_position, (indices_dict, indices_bold, _) in enumerate(selections):
            indices_bold_cpu = indices_bold.cpu()
            tables.append(
                SequenceCoordinateTable(
                    group_names=list(indices_dict.keys()),
                    group_sizes=[len(indices) for indices in indices_dict.values()],
                    original_indices=indices_bold_cpu[:, 0],
                    qualifying_token_indices=indices_bold_cpu[:, 1],
                    source_token_indices=source_indices_split[feature_position],
                    token_ids=token_ids_split[feature_position],
                    feat_acts=feat_acts_split[feature_position],
                    token_logits=token_logits_split[feature_position],
                )
            )
        return tables

    def build_sequence_coordinate_table(
        self,
        token_ids: Int[Tensor, "n_bold buf"],
        feat_acts_coloring: Float[Tensor, "n_bold buf"],
        feat_logits: Float[Tensor, "d_vocab"],
        indices_dict: dict[str, Int[Tensor, "n_bold 2"]],
        indices_bold: Int[Tensor, "n_bold"],
        source_token_indices: Int[Tensor, "n_bold buf"] | None = None,
    ) -> SequenceCoordinateTable:
        if self.cfg.perform_ablation_experiments:
            raise NotImplementedError(
                "We are not supporting ablation experiments for now."
            )

        token_ids = token_ids.cpu()
        if source_token_indices is None:
            source_token_indices = torch.arange(
                token_ids.shape[1], device=token_ids.device
            ).repeat(token_ids.shape[0], 1)
        source_token_indices = source_token_indices.cpu()
        feat_acts_coloring = feat_acts_coloring.cpu()
        feat_logits = feat_logits.cpu()
        indices_bold = indices_bold.cpu()
        feat_acts = np.around(
            feat_acts_coloring.to(dtype=torch.float32)
            .numpy()
            .astype(np.float64, copy=False),
            4,
        )

        return SequenceCoordinateTable(
            group_names=list(indices_dict.keys()),
            group_sizes=[len(indices) for indices in indices_dict.values()],
            original_indices=indices_bold[:, 0],
            qualifying_token_indices=indices_bold[:, 1],
            source_token_indices=source_token_indices,
            token_ids=token_ids,
            feat_acts=feat_acts,
            token_logits=feat_logits[token_ids],
        )
