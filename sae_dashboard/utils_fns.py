import importlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import torch
from dataclasses_json import dataclass_json
from eindex import eindex
from jaxtyping import Bool, Float, Int
from sae_lens import ActivationsStore
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from sae_dashboard.perf_logging import (
    log_perf_event,
    tensor_runtime_metadata,
    timed_stage,
)
from sae_dashboard.tl_compat import to_numpy

T = TypeVar("T")
HistogramDataT = TypeVar("HistogramDataT", bound="HistogramData")

# from rich.progress import ProgressColumn, Task # MofNCompleteColumn
# from rich.text import Text
# from rich.table import Column


def get_tokens(
    activations_store: ActivationsStore,
    n_prompts: int,
) -> Tensor:
    all_tokens_list = []
    pbar = tqdm(range(n_prompts))
    for _ in pbar:
        batch_tokens = activations_store.get_batch_tokens()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0)
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens


def has_duplicate_rows(tensor: torch.Tensor) -> bool:
    """
    Check if a 2D tensor has any duplicate rows, with special handling for MPS devices.

    Args:
        tensor (torch.Tensor): A 2D tensor to check for duplicate rows.

    Returns:
        bool: True if there are duplicate rows, False otherwise.

    Raises:
        ValueError: If the input tensor is not 2D.
    """
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")

    if tensor.device.type == "mps":
        # Alternative strategy for MPS devices
        # Convert to CPU and use a different approach
        tensor_cpu = tensor.cpu()

        # Convert each row to a tuple (hashable) and count occurrences
        row_tuples = [tuple(row.tolist()) for row in tensor_cpu]
        from collections import Counter

        counts = Counter(row_tuples)

        # Check if any row appears more than once
        return any(count > 1 for count in counts.values())
    else:
        # Original strategy for other devices
        _, counts = torch.unique(tensor, dim=0, return_counts=True)
        return bool(torch.any(counts > 1))


def get_device() -> torch.device:
    """
    Helper function to return the correct device (cuda, mps, or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


# # Depreciated - we no longer use a global device variable
# device = get_device()

Arr = np.ndarray

MAIN = __name__ == "__main__"


def create_iterator(
    iterator: Iterable[T], verbose: bool, desc: str | None = None
) -> Iterable[T]:
    """
    Returns an iterator, useful for reducing code repetition.
    """
    return tqdm(iterator, desc=desc, leave=False) if verbose else iterator


def k_largest_indices(
    x: Float[Tensor, "rows cols"],
    k: int,
    largest: bool = True,
    buffer: tuple[int, int] | None = (5, -5),
) -> Int[Tensor, "k 2"]:
    """
    Args:
        x:
            2D array of floats (these will be the values of feature activations or losses for each
            token in our batch)
        k:
            Number of indices to return
        largest:
            Whether to return the indices for the largest or smallest values
        buffer:
            Positions to avoid at the start / end of the sequence, i.e. we can include the slice buffer[0]: buffer[1].
            If None, then we use all sequences

    Returns:
        The indices of the top or bottom `k` elements in `x`. In other words, output[i, :] is the (row, column) index of
        the i-th largest/smallest element in `x`.
    """
    if buffer is None:
        buffer = (0, x.size(1))
    x = x[:, buffer[0] : buffer[1]]
    indices = x.flatten().topk(k=k, largest=largest).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer[0]
    return torch.stack((rows, cols), dim=1)


def sample_unique_indices(
    large_number: int, small_number: int
) -> Int[Tensor, "small_number"]:
    """
    Samples a small number of unique indices from a large number of indices.

    This is more efficient than using `torch.permutation`, because we don't need to shuffle everything.
    """
    sampled_indices = random.sample(range(large_number), small_number)
    return torch.Tensor(sampled_indices).to(torch.int64)


def random_range_indices(
    x: Float[Tensor, "batch seq"],
    k: int,
    bounds: tuple[float, float],
    buffer: tuple[int, int] | None = (5, -5),
    upper_inclusive: bool = True,
) -> Int[Tensor, "k 2"]:
    """
    Args:
        x:
            2D array of floats (these will be the values of feature activations or losses for each
            token in our batch)
        k:
            Number of indices to return
        bounds:
            The range of values to consider (so we can get quantiles)
        buffer:
            Positions to avoid at the start / end of the sequence, i.e. we can include the slice buffer[0]: buffer[1]
        upper_inclusive:
            Whether the upper bound is inclusive (historical default). False gives numpy-histogram-style
            half-open membership ``[lower, upper)``.

    Returns:
        Same thing as k_largest_indices, but the difference is that we're using quantiles rather than top/bottom k.
    """
    if buffer is None:
        buffer = (0, x.size(1))

    # Limit x, because our indices (bolded words) shouldn't be too close to the left/right of sequence
    x = x[:, buffer[0] : buffer[1]]

    # Creat a mask for where x is in range, and get the indices as a tensor of shape (k, 2)
    mask = (bounds[0] <= x) & ((x <= bounds[1]) if upper_inclusive else (x < bounds[1]))
    indices = torch.stack(torch.where(mask), dim=-1)

    # If we have more indices than we need, randomly select k of them

    if len(indices) > k:
        indices = indices[sample_unique_indices(len(indices), k)]

    # Adjust indices to account for the buffer
    return indices + torch.tensor([0, buffer[0]]).to(indices.device)


# TODO - solve the `get_decode_html_safe_fn` issue
# The verion using `tokenizer.decode` is much slower, but Stefan's raised issues about it not working correctly for e.g.
# Cyrillic characters. I think patching the `vocab_dict` in some way is the best solution.

# def get_decode_html_safe_fn(tokenizer, html: bool = False) -> Callable[[int | list[int]], str | list[str]]:
#     '''
#     Creates a tokenization function on single integer token IDs, which is HTML-friendly.
#     '''
#     def decode(token_id: int | list[int]) -> str | list[str]:
#         '''
#         Check this is a single token
#         '''
#         if isinstance(token_id, int):
#             str_tok = tokenizer.decode(token_id)
#             return process_str_tok(str_tok, html=html)
#         else:
#             str_toks = tokenizer.batch_decode(token_id)
#             return [process_str_tok(str_tok, html=html) for str_tok in str_toks]

#     return decode


def get_decode_html_safe_fn(
    tokenizer: PreTrainedTokenizerBase, html: bool = False
) -> Callable[[int | list[int]], str | list[str]]:
    vocab_dict = {v: k for k, v in tokenizer.vocab.items()}  # type: ignore

    def decode(token_id: int | list[int]) -> str | list[str]:
        """
        Check this is a single token
        """
        if isinstance(token_id, int):
            str_tok = vocab_dict.get(token_id, "UNK")
            return process_str_tok(str_tok, html=html)
        else:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.tolist()
            return [decode(tok) for tok in token_id]  # type: ignore

    return decode


# # Code to test this function:
# from transformer_lens import HookedTransformer
# model = HookedTransformer.from_pretrained("gelu-1l")
# unsafe_token = "<"
# unsafe_token_id = model.tokenizer.encode(unsafe_token, return_tensors="pt")[0].item() # type: ignore
# assert get_decode_html_safe_fn(model.tokenizer)(unsafe_token_id) == "<"
# assert get_decode_html_safe_fn(model.tokenizer, html=True)(unsafe_token_id) == "&lt;"


HTML_CHARS = {
    "\\": "&bsol;",
    "<": "&lt;",
    ">": "&gt;",
    ")": "&#41;",
    "(": "&#40;",
    "[": "&#91;",
    "]": "&#93;",
    "{": "&#123;",
    "}": "&#125;",
}
HTML_ANOMALIES = {
    "âĢĶ": "&mdash;",
    "âĢĵ": "&ndash;",
    "âĢĭ": "&#8203;",
    "âĢľ": "&ldquo;",
    "âĢĿ": "&rdquo;",
    "âĢĺ": "&lsquo;",
    "âĢĻ": "&rsquo;",
    "Ġ": "&nbsp;",
    "Ċ": "&bsol;n",
    "ĉ": "&bsol;t",
}
HTML_ANOMALIES_REVERSED = {
    "&mdash": "—",
    "&ndash": "–",
    # "&#8203": "​", # TODO: this is actually zero width space character. what's the best way to represent it?
    "&ldquo": "“",
    "&rdquo": "”",
    "&lsquo": "‘",
    "&rsquo": "’",
    "&nbsp;": " ",
    "&bsol;": "\\",
}
HTML_QUOTES = {
    "'": "&apos;",
    '"': "&quot;",
}
HTML_ALL = {**HTML_CHARS, **HTML_QUOTES, " ": "&nbsp;"}

HTML_ALL_REVERSED = {
    **{v: k for k, v in HTML_CHARS.items()},
    **HTML_ANOMALIES_REVERSED,
}


def process_str_tok(str_tok: str, html: bool = True) -> str:
    """
    Takes a string token, and does the necessary formatting to produce the right HTML output. There are 2 things that
    might need to be changed:

        (1) Anomalous chars like Ġ should be replaced with their normal Python string representations
            e.g. "Ġ" -> " "
        (2) Special HTML characters like "<" should be replaced with their HTML representations
            e.g. "<" -> "&lt;", or " " -> "&nbsp;"

    We always do (1), the argument `html` determines whether we do (2) as well.
    """
    for k, v in HTML_ANOMALIES.items():
        str_tok = str_tok.replace(k, v)

    if html:
        # Get rid of the quotes and apostrophes, and replace them with their HTML representations
        for k, v in HTML_QUOTES.items():
            str_tok = str_tok.replace(k, v)
        # repr turns \n into \\n, while slicing removes the quotes from the repr
        str_tok = repr(str_tok)[1:-1]

        # Apply the map from special characters to their HTML representations
        for k, v in HTML_CHARS.items():
            str_tok = str_tok.replace(k, v)

    return str_tok


def unprocess_str_tok(str_tok: str) -> str:
    """
    Performs the reverse of the `process_str_tok` function, i.e. maps from HTML representations back to their original
    characters. This is useful when e.g. our string is inside a <code>...</code> element, because then we have to use
    the literal characters.
    """
    for k, v in HTML_ALL_REVERSED.items():
        str_tok = str_tok.replace(k, v)

    return str_tok


@overload
def to_str_tokens(
    decode_fn: Callable[[int | list[int]], str | list[str]],
    tokens: int,
) -> str: ...


@overload
def to_str_tokens(
    decode_fn: Callable[[int | list[int]], str | list[str]],
    tokens: list[int],
) -> list[str]: ...


def to_str_tokens(
    decode_fn: Callable[[int | list[int]], str | list[str]],
    tokens: int | list[int] | torch.Tensor,
) -> str | Any:
    """
    Helper function which converts tokens to their string representations, but (if tokens is a tensor) keeps
    them in the same shape as the original tensor (i.e. nested lists).
    """
    # Deal with the int case separately
    if isinstance(tokens, int):
        return decode_fn(tokens)

    # If the tokens are a (possibly nested) list, turn them into a tensor
    if isinstance(tokens, list):
        tokens = torch.tensor(tokens)

    # Get flattened list of tokens
    str_tokens = [decode_fn(t) for t in tokens.flatten().tolist()]

    # Reshape
    return np.reshape(str_tokens, tokens.shape).tolist()


class TopK:
    """
    This function implements a version of torch.topk over the last dimension. It offers the following:

        (1) Nicer type signatures (the default obj returned by torck.topk isn't well typed)
        (2) Helper functions for indexing & other standard tensor operations like .ndim, .shape, etc.
        (3) An efficient topk calculation, which doesn't bother applying it to the zero elements of a tensor.
    """

    values: Arr
    indices: Arr

    def __init__(
        self,
        tensor: Float[Tensor, "... d"],
        k: int,
        largest: bool = True,
        tensor_mask: Bool[Tensor, "..."] | None = None,
    ):
        self.k = k
        self.largest = largest
        self.values, self.indices = self.topk(tensor, tensor_mask)

    def __getitem__(self, item: int) -> "TopK":
        new_topk = TopK.__new__(TopK)
        new_topk.k = self.k
        new_topk.largest = self.largest
        new_topk.values = self.values[item]
        new_topk.indices = self.indices[item]
        return new_topk

    def __len__(self) -> int:
        return len(self.values)

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def shape(self) -> tuple[int]:
        return tuple(self.values.shape)  # type: ignore

    def numel(self) -> int:
        return self.values.size

    def topk(  # type: ignore
        self,
        tensor: Float[Tensor, "... d"],
        tensor_mask: Bool[Tensor, "..."] | None = None,
    ) -> tuple[Arr, Arr]:
        """
        This is an efficient version of `torch.topk(..., dim=-1)`. It saves time by only doing the topk calculation over
        the bits of `tensor` where `tensor_mask=True`. This is useful when `tensor` is very sparse, e.g. it has shape
        (batch, seq, d_vocab) and its elements are zero if the corresponding token has feature activation zero. In this
        case, we don't want to waste time taking topk over a tensor of zeros.
        """
        # If no tensor mask is provided, then we just return the topk values and indices
        if tensor_mask is None or not tensor_mask.any():
            k = min(self.k, tensor.shape[-1])
            topk = tensor.topk(k=k, largest=self.largest)
            return to_numpy(topk.values), to_numpy(topk.indices)

        # Get the topk of the tensor, but only computed over the values of the tensor which are nontrivial
        assert (
            tensor_mask.shape == tensor.shape[:-1]
        ), "Error: unexpected shape for tensor mask."
        tensor_nontrivial_values = tensor[tensor_mask]  # shape [rows d]
        k = min(self.k, tensor_nontrivial_values.shape[-1])
        k = self.k
        topk = tensor_nontrivial_values.topk(
            k=k, largest=self.largest
        )  # shape [rows k]

        # Get an array of indices and values (with unimportant elements) which we'll index into using the topk object
        topk_shape = (*tensor_mask.shape, k)
        topk_indices = torch.zeros(
            topk_shape, device=tensor.device, dtype=torch.long
        )  # ).long()  # shape [... k]
        topk_indices[tensor_mask] = topk.indices
        topk_values = torch.zeros(
            topk_shape, device=tensor.device, dtype=tensor.dtype
        )  # shape [... k]
        topk_values[tensor_mask] = topk.values

        return to_numpy(topk_values), to_numpy(topk_indices)


def merge_lists(*lists: Iterable[T]) -> list[T]:
    """
    Merges a bunch of lists into a single list.
    """
    return [item for sublist in lists for item in sublist]


def extract_and_remove_scripts(html_content: str) -> tuple[str, str]:
    """
    Extracts JavaScript from script tags in the HTML content, and returns it as a single string,
    along with the original content with the script tags removed.
    """
    # Pattern to find <script>...</script> tags and capture content inside
    pattern = r"<script[^>]*>(.*?)</script>"

    # Find all script tags and extract content
    scripts = re.findall(pattern, html_content, re.DOTALL)

    # Remove script tags from the original content
    html_without_scripts = re.sub(pattern, "", html_content, flags=re.DOTALL)

    # Join extracted JavaScript code
    javascript = "\n".join(scripts)

    return javascript, html_without_scripts


def pad_with_zeros(
    x: list[float],
    n: int,
    side: Literal["left", "right"] = "left",
) -> list[float]:
    """
    Pads a list with zeros to make it the correct length.
    """
    assert len(x) <= n, "Error: x must have fewer than n elements."

    if side == "right":
        return x + [0.0] * (n - len(x))
    else:
        return [0.0] * (n - len(x)) + x


# This defines the number of decimal places we'll use. It's assumed to refer to values in the range [0, 1] rather than
# pct, e.g. precision of 5 would be 99.497% = 0.99497. In other words, decimal_places = precision - 2.

SYMMETRIC_RANGES_AND_PRECISIONS: list[tuple[list[float], int]] = [
    ([0.0, 0.01], 5),
    ([0.01, 0.05], 4),
    ([0.05, 0.95], 3),
    ([0.95, 0.99], 4),
    ([0.99, 1.0], 5),
]
ASYMMETRIC_RANGES_AND_PRECISIONS: list[tuple[list[float], int]] = [
    ([0.0, 0.95], 3),
    ([0.95, 0.99], 4),
    ([0.99, 1.0], 5),
]


@dataclass_json
@dataclass
class FeatureStatistics:
    """
    This object (which used to be called QuantileCalculator) stores stats about a dataset.

    The quantiles are a bit complicated because we store a higher precision for values closer to 100%. Most of the
    other stats are pretty straightforward.

    We create these objects using the `create` method. We assume data supplid is 2D, where each row is a different
    dataset that we want to compute the stats for.
    """

    # Stats: max, frac_nonzero, skew, kurtosis
    max: list[float] = field(default_factory=list)
    frac_nonzero: list[float] = field(default_factory=list)
    skew: list[float] = field(default_factory=list)
    kurtosis: list[float] = field(default_factory=list)

    # Quantile data
    quantile_data: list[list[float]] = field(default_factory=list)
    quantiles: list[float] = field(default_factory=list)
    ranges_and_precisions: list[tuple[list[float], int]] = field(
        default_factory=lambda: ASYMMETRIC_RANGES_AND_PRECISIONS
    )

    @property
    def aggdata(
        self,
        precision: int = 5,
    ) -> dict[str, list[float]]:
        return {
            "max": [round(x, precision) for x in self.max],
            "frac_nonzero": [round(x, precision) for x in self.frac_nonzero],
            "skew": [round(x, precision) for x in self.skew],
            "kurtosis": [round(x, precision) for x in self.kurtosis],
        }

    @classmethod
    def create(
        cls,
        data: Optional[torch.Tensor] = None,
        ranges_and_precisions: list[
            tuple[list[float], int]
        ] = ASYMMETRIC_RANGES_AND_PRECISIONS,
        batch_size: Optional[int] = None,
        use_sparse_quantiles: bool = False,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> "FeatureStatistics":
        """Calculates various statistics for a tensor of activations.

        Args:
            data: A tensor of activations; should be shape (n_features, n_samples (n_prompts * n_prompt_tokens)).
            ranges_and_precisions: A list of tuples of the form (range, precision).
            batch_size: The feature batch size to use for processing the acts. Reduce this if you encounter OOM errors.
            use_sparse_quantiles: If True, approximate quantiles from each feature's nonzero activations after
                accounting for the zero mass. This is much cheaper for sparse non-negative activation tensors used in
                dashboards.
            valid_mask: Optional boolean mask over the sample dimension. When provided, statistics are computed only
                over valid positions. This is used by per-example padded prompt mode so ignored pad tokens don't distort
                density or quantile estimates.

        Returns:
            A FeatureStatistics object.
        """
        payload = cls._compute_statistics_payload(
            data=data,
            ranges_and_precisions=ranges_and_precisions,
            batch_size=batch_size,
            use_sparse_quantiles=use_sparse_quantiles,
            valid_mask=valid_mask,
        )

        return cls(
            max=payload["max"],
            frac_nonzero=payload["frac_nonzero"],
            skew=payload["skew"],
            kurtosis=payload["kurtosis"],
            quantile_data=payload["quantile_data"],
            quantiles=payload["quantiles"],
            ranges_and_precisions=payload["ranges_and_precisions"],
        )

    @classmethod
    def _compute_statistics_payload(
        cls,
        data: Optional[torch.Tensor] = None,
        ranges_and_precisions: list[
            tuple[list[float], int]
        ] = ASYMMETRIC_RANGES_AND_PRECISIONS,
        batch_size: Optional[int] = None,
        use_sparse_quantiles: bool = False,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if not batch_size:
            batch_size = 0 if data is None else data.shape[0]

        # Generate quantiles from the ranges_and_precisions list
        quantiles = []
        for r, p in ranges_and_precisions:
            start, end = r
            step = 10**-p
            quantiles.extend(np.arange(start, end - 0.5 * step, step))

        # If data is None, then set the quantiles and quantile_data to None, and return
        if data is None:
            return {
                "max": [],
                "frac_nonzero": [],
                "skew": [],
                "kurtosis": [],
                "quantile_data": [],
                "quantiles": [round(q, 6) for q in quantiles + [1.0]],
                "ranges_and_precisions": ranges_and_precisions,
            }

        if valid_mask is not None:
            if valid_mask.ndim == 1:
                if valid_mask.shape[0] != data.shape[-1]:
                    raise ValueError(
                        "valid_mask length must match the sample dimension of data"
                    )
            elif valid_mask.shape != data.shape:
                raise ValueError(
                    "valid_mask must be 1D over samples or match data shape"
                )

        # Process data in batches
        n_features = data.shape[0]
        _max = []
        frac_nonzero = []
        positive_density = []
        quantile_data = []

        for i in range(0, n_features, batch_size):
            batch = data[i : min(i + batch_size, n_features)]
            batch_mask = None
            if valid_mask is not None:
                if valid_mask.ndim == 1:
                    batch_mask = valid_mask.unsqueeze(0).expand(batch.shape[0], -1)
                else:
                    batch_mask = valid_mask[i : min(i + batch_size, n_features)]
                batch_mask = batch_mask.to(device=batch.device, dtype=torch.bool)

            if batch_mask is None:
                _max.extend(batch.max(dim=-1).values.tolist())
                frac_nonzero.extend((batch.abs() > 1e-6).float().mean(dim=-1).tolist())
                positive_density.extend((batch > 0).float().mean(dim=-1).tolist())
            else:
                valid_counts = batch_mask.sum(dim=-1)
                safe_valid_counts = valid_counts.clamp(min=1)
                masked_for_max = batch.masked_fill(~batch_mask, float("-inf"))
                batch_max = masked_for_max.max(dim=-1).values
                batch_max = torch.where(
                    valid_counts > 0, batch_max, torch.zeros_like(batch_max)
                )
                _max.extend(batch_max.tolist())
                nonzero_counts = ((batch.abs() > 1e-6) & batch_mask).sum(dim=-1)
                frac_nonzero.extend(
                    (
                        nonzero_counts.to(torch.float32)
                        / safe_valid_counts.to(torch.float32)
                    ).tolist()
                )
                positive_counts = ((batch > 0) & batch_mask).sum(dim=-1)
                positive_density.extend(
                    (
                        positive_counts.to(torch.float32)
                        / safe_valid_counts.to(torch.float32)
                    ).tolist()
                )

            quantiles_tensor = torch.tensor(
                quantiles, dtype=batch.dtype, device=batch.device
            )

            if use_sparse_quantiles and not (
                ((batch < -1e-6) & batch_mask).any()
                if batch_mask is not None
                else (batch < -1e-6).any()
            ):
                quantile_data.extend(
                    cls._sparse_quantile_rows(
                        batch.to(torch.float32),
                        quantiles_tensor.to(torch.float32),
                        valid_mask=batch_mask,
                    )
                )
            elif batch_mask is not None:
                quantile_data.extend(
                    cls._masked_quantile_rows(
                        batch.to(torch.float32),
                        quantiles_tensor.to(torch.float32),
                        batch_mask,
                    )
                )
            else:
                batch_quantile_data = torch.quantile(
                    batch.to(torch.float32),
                    quantiles_tensor.to(torch.float32),
                    dim=-1,
                )
                quantile_data.extend(batch_quantile_data.T.tolist())

        quantiles = [round(q, 6) for q in quantiles + [1.0]]
        quantile_data = [[round(q, 6) for q in qd] for qd in quantile_data]

        # Strip out the quantile data prefixes which are all zeros
        for i, qd in enumerate(quantile_data):
            first_nonzero = next(
                (i for i, x in enumerate(qd) if abs(x) > 1e-6), len(qd)
            )
            quantile_data[i] = qd[first_nonzero:]

        return {
            "max": _max,
            "frac_nonzero": frac_nonzero,
            "positive_density": positive_density,
            "skew": [],  # Placeholder for skew calculation
            "kurtosis": [],  # Placeholder for kurtosis calculation
            "quantile_data": quantile_data,
            "quantiles": quantiles,
            "ranges_and_precisions": ranges_and_precisions,
        }

    @classmethod
    def create_arrow_table(
        cls,
        data: Optional[torch.Tensor] = None,
        ranges_and_precisions: list[
            tuple[list[float], int]
        ] = ASYMMETRIC_RANGES_AND_PRECISIONS,
        batch_size: Optional[int] = None,
        use_sparse_quantiles: bool = False,
        valid_mask: Optional[torch.Tensor] = None,
        include_quantiles: bool = True,
    ) -> Any:
        pyarrow = importlib.import_module("pyarrow")

        if not include_quantiles:
            return cls._create_scalar_arrow_table(pyarrow, data, valid_mask)

        payload = cls._compute_statistics_payload(
            data=data,
            ranges_and_precisions=ranges_and_precisions,
            batch_size=batch_size,
            use_sparse_quantiles=use_sparse_quantiles,
            valid_mask=valid_mask,
        )

        feature_count = len(cast(list[float], payload["max"]))
        table = pyarrow.table(
            {
                "feature_index": pyarrow.array(range(feature_count)),
                "max": pyarrow.array(payload["max"]),
                "frac_nonzero": pyarrow.array(payload["frac_nonzero"]),
                "positive_density": pyarrow.array(payload["positive_density"]),
                "quantile_data": pyarrow.array(payload["quantile_data"]),
            }
        )
        metadata = {
            b"quantiles": json.dumps(payload["quantiles"]).encode("utf-8"),
            b"ranges_and_precisions": json.dumps(
                payload["ranges_and_precisions"]
            ).encode("utf-8"),
        }
        return table.replace_schema_metadata(metadata)

    @staticmethod
    def _create_scalar_arrow_table(
        pyarrow: Any,
        data: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> Any:
        if data is None:
            return pyarrow.table(
                {
                    "feature_index": pyarrow.array([]),
                    "max": pyarrow.array([]),
                    "frac_nonzero": pyarrow.array([]),
                    "positive_density": pyarrow.array([]),
                    "positive_count": pyarrow.array([]),
                    "nonzero_count": pyarrow.array([]),
                    "valid_count": pyarrow.array([]),
                }
            )

        if valid_mask is None:
            valid_data = data
            valid_counts = torch.full(
                (data.shape[0],),
                data.shape[-1],
                dtype=torch.int64,
                device=data.device,
            )
            max_values = data.max(dim=-1).values
            nonzero_counts = (data.abs() > 1e-6).sum(dim=-1)
            positive_counts = (data > 0).sum(dim=-1)
        elif valid_mask.ndim == 1:
            if valid_mask.shape[0] != data.shape[-1]:
                raise ValueError(
                    "A 1D valid_mask must have the same sample dimension as data"
                )
            valid_mask = valid_mask.to(device=data.device, dtype=torch.bool)
            valid_count = int(valid_mask.sum().item())
            valid_counts = torch.full(
                (data.shape[0],),
                valid_count,
                dtype=torch.int64,
                device=data.device,
            )
            if valid_count == data.shape[-1]:
                valid_data = data
            elif valid_count > 0:
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
                valid_data = data.index_select(-1, valid_indices)
            else:
                valid_data = data.new_empty((data.shape[0], 0))

            if valid_count > 0:
                max_values = valid_data.max(dim=-1).values
                nonzero_counts = (valid_data.abs() > 1e-6).sum(dim=-1)
                positive_counts = (valid_data > 0).sum(dim=-1)
            else:
                max_values = torch.zeros(
                    data.shape[0], dtype=data.dtype, device=data.device
                )
                nonzero_counts = torch.zeros(
                    data.shape[0], dtype=torch.int64, device=data.device
                )
                positive_counts = torch.zeros_like(nonzero_counts)
        elif valid_mask.ndim == 2:
            if tuple(valid_mask.shape) != tuple(data.shape):
                raise ValueError("A 2D valid_mask must have the same shape as data")
            valid_mask = valid_mask.to(device=data.device, dtype=torch.bool)
            valid_counts = valid_mask.sum(dim=-1)
            masked_for_max = data.masked_fill(~valid_mask, float("-inf"))
            max_values = masked_for_max.max(dim=-1).values
            max_values = torch.where(
                valid_counts > 0,
                max_values,
                torch.zeros_like(max_values),
            )
            nonzero_counts = ((data.abs() > 1e-6) & valid_mask).sum(dim=-1)
            positive_counts = ((data > 0) & valid_mask).sum(dim=-1)
        else:
            raise ValueError("valid_mask must be 1D or 2D when provided")

        safe_valid_counts = valid_counts.clamp(min=1).to(torch.float32)
        frac_nonzero = nonzero_counts.to(torch.float32) / safe_valid_counts
        positive_density = positive_counts.to(torch.float32) / safe_valid_counts
        float_columns = (
            torch.stack(
                (
                    max_values.to(torch.float32),
                    frac_nonzero,
                    positive_density,
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        count_columns = (
            torch.stack(
                (
                    positive_counts.to(torch.int64),
                    nonzero_counts.to(torch.int64),
                    valid_counts.to(torch.int64),
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        return pyarrow.table(
            {
                "feature_index": pyarrow.array(
                    np.arange(data.shape[0], dtype=np.int64), type=pyarrow.int64()
                ),
                "max": pyarrow.array(np.ascontiguousarray(float_columns[:, 0])),
                "frac_nonzero": pyarrow.array(
                    np.ascontiguousarray(float_columns[:, 1])
                ),
                "positive_density": pyarrow.array(
                    np.ascontiguousarray(float_columns[:, 2])
                ),
                "positive_count": pyarrow.array(
                    np.ascontiguousarray(count_columns[:, 0])
                ),
                "nonzero_count": pyarrow.array(
                    np.ascontiguousarray(count_columns[:, 1])
                ),
                "valid_count": pyarrow.array(np.ascontiguousarray(count_columns[:, 2])),
            }
        )

    @classmethod
    def create_scalar_arrow_table_from_flat_valid(
        cls,
        flat_data: torch.Tensor,
        valid_flat_indices: torch.Tensor,
        compute_device: str | torch.device | None = None,
        row_chunk_size: int = 256,
    ) -> Any:
        """Chunked-device equivalent of the runner's gather + `_create_scalar_arrow_table` path.

        ``flat_data`` is the ``[rows, samples_total]`` activation matrix (any device/dtype);
        ``valid_flat_indices`` selects the valid sample columns. Row chunks are staged onto
        ``compute_device`` (defaults to ``flat_data.device``), the valid columns gathered
        there, and the same exact reductions applied (row max, ``abs > 1e-6`` counts,
        ``> 0`` counts, exact integer/float32 divisions), so the resulting table is
        identical to gathering on the host first. When no valid columns exist, the
        zero-substitution semantics of the previous runner path are reproduced
        (all-zero statistics with ``valid_count == 1``).
        """
        pyarrow = importlib.import_module("pyarrow")
        device = (
            torch.device(compute_device)
            if compute_device is not None
            else flat_data.device
        )
        n_rows = int(flat_data.shape[0])
        n_valid = int(valid_flat_indices.numel())
        valid_indices_dev = valid_flat_indices.to(device)

        max_chunks: list[torch.Tensor] = []
        nonzero_chunks: list[torch.Tensor] = []
        positive_chunks: list[torch.Tensor] = []
        row_chunk_size = max(1, row_chunk_size)
        for row_start in range(0, n_rows, max(1, row_chunk_size)):
            row_end = min(row_start + row_chunk_size, n_rows)
            chunk = flat_data[row_start:row_end].to(device)
            if n_valid > 0:
                valid_chunk = chunk.index_select(-1, valid_indices_dev)
            else:
                valid_chunk = torch.zeros(
                    (row_end - row_start, 1),
                    dtype=chunk.dtype,
                    device=device,
                )
            max_chunks.append(valid_chunk.max(dim=-1).values.to(torch.float32))
            nonzero_chunks.append((valid_chunk.abs() > 1e-6).sum(dim=-1))
            positive_chunks.append((valid_chunk > 0).sum(dim=-1))
            del chunk, valid_chunk

        max_values = torch.cat(max_chunks) if max_chunks else torch.zeros(0)
        nonzero_counts = (
            torch.cat(nonzero_chunks)
            if nonzero_chunks
            else torch.zeros(0, dtype=torch.int64)
        )
        positive_counts = (
            torch.cat(positive_chunks)
            if positive_chunks
            else torch.zeros(0, dtype=torch.int64)
        )
        valid_counts = torch.full(
            (n_rows,),
            n_valid if n_valid > 0 else 1,
            dtype=torch.int64,
            device=max_values.device,
        )

        safe_valid_counts = valid_counts.clamp(min=1).to(torch.float32)
        frac_nonzero = nonzero_counts.to(torch.float32) / safe_valid_counts
        positive_density = positive_counts.to(torch.float32) / safe_valid_counts
        float_columns = (
            torch.stack((max_values, frac_nonzero, positive_density), dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        count_columns = (
            torch.stack(
                (
                    positive_counts.to(torch.int64),
                    nonzero_counts.to(torch.int64),
                    valid_counts.to(torch.int64),
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        return pyarrow.table(
            {
                "feature_index": pyarrow.array(
                    np.arange(n_rows, dtype=np.int64), type=pyarrow.int64()
                ),
                "max": pyarrow.array(np.ascontiguousarray(float_columns[:, 0])),
                "frac_nonzero": pyarrow.array(
                    np.ascontiguousarray(float_columns[:, 1])
                ),
                "positive_density": pyarrow.array(
                    np.ascontiguousarray(float_columns[:, 2])
                ),
                "positive_count": pyarrow.array(
                    np.ascontiguousarray(count_columns[:, 0])
                ),
                "nonzero_count": pyarrow.array(
                    np.ascontiguousarray(count_columns[:, 1])
                ),
                "valid_count": pyarrow.array(np.ascontiguousarray(count_columns[:, 2])),
            }
        )

    @classmethod
    def from_arrow_table(cls, table: Any) -> "FeatureStatistics":
        columns = table.to_pydict()
        metadata = table.schema.metadata or {}
        quantiles_raw = metadata.get(b"quantiles", b"[]")
        ranges_raw = metadata.get(b"ranges_and_precisions", b"[]")
        quantile_data_column = columns.get("quantile_data", [])
        return cls(
            max=[float(value) for value in columns.get("max", [])],
            frac_nonzero=[float(value) for value in columns.get("frac_nonzero", [])],
            skew=[],
            kurtosis=[],
            quantile_data=[
                [float(value) for value in row] for row in quantile_data_column
            ],
            quantiles=[float(value) for value in json.loads(quantiles_raw)],
            ranges_and_precisions=[
                ([float(bound) for bound in pair[0]], int(pair[1]))
                for pair in json.loads(ranges_raw)
            ],
        )

    @staticmethod
    def _masked_quantile_rows(
        batch: torch.Tensor,
        quantiles_tensor: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> list[list[float]]:
        quantile_rows = []
        for row, row_mask in zip(batch, valid_mask):
            valid_values = row[row_mask]
            if valid_values.numel() == 0:
                quantile_rows.append([])
                continue
            quantile_rows.append(
                torch.quantile(valid_values, quantiles_tensor, dim=-1).tolist()
            )
        return quantile_rows

    @staticmethod
    def _sparse_quantile_rows(
        batch: torch.Tensor,
        quantiles_tensor: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> list[list[float]]:
        quantile_rows = []
        for row_index, row in enumerate(batch):
            row_mask = (
                valid_mask[row_index]
                if valid_mask is not None
                else torch.ones_like(row, dtype=torch.bool)
            )
            sample_count = int(row_mask.sum().item())
            if sample_count == 0:
                quantile_rows.append([])
                continue
            valid_values = row[row_mask]
            nonzero_values = valid_values[valid_values.abs() > 1e-6]
            if nonzero_values.numel() == 0:
                quantile_rows.append([])
                continue
            frac_nonzero = nonzero_values.numel() / sample_count
            zero_fraction = 1.0 - frac_nonzero
            active_quantiles = quantiles_tensor[quantiles_tensor > zero_fraction]
            if active_quantiles.numel() == 0:
                quantile_rows.append([])
                continue
            nonzero_quantiles = (
                (active_quantiles - zero_fraction) / frac_nonzero
            ).clamp(0.0, 1.0)
            quantile_rows.append(
                torch.quantile(nonzero_values, nonzero_quantiles, dim=-1).tolist()
            )
        return quantile_rows

    def update(self, other: "FeatureStatistics"):
        """
        Merges two FeatureStatistics objects together (changing self inplace). This is useful when we're batching our
        calculations over different groups of features, and we want to merge them together at the end.

        Note, we also deal with the special case where self has no data.
        """
        assert (
            self.ranges_and_precisions == other.ranges_and_precisions
        ), "Error: can't merge two FeatureStatistics objects with different ranges."

        self.max.extend(other.max)
        self.frac_nonzero.extend(other.frac_nonzero)
        self.skew.extend(other.skew)
        self.kurtosis.extend(other.kurtosis)
        self.quantiles.extend(other.quantiles)
        self.quantile_data.extend(other.quantile_data)

    def get_quantile(
        self,
        values: Float[Tensor, "batch *data_dim"],
        batch_indices: list[int] | None = None,
    ) -> tuple[Float[Tensor, "batch *data_dim"], Int[Tensor, "batch *data_dim"]]:
        """
        Args:
            values:
                Tensor of values for which we want to compute the quantiles. If this is 1D then it is interpreted as a
                single value for each dataset (i.e. for each row of the reference data), if it's 2D then it's a row of
                values for each dataset.
            batch_indices:
                If not None, then this should be a list of batch indices we're actually using, in other words we should
                index `self.quantiles` down to only these indices. This is useful because often we're only doing this
                calculation on a small set of features (the ones which are non-zero).

        Returns:
            quantiles:
                The quantiles of `values` within the respective rows of the reference data.
            precisions:
                The precision of the quantiles (i.e. how many decimal places we're accurate to).
        """
        rp = self.ranges_and_precisions
        ranges = torch.tensor([r[0] for (r, _p) in rp] + [1.0]).to(values.device)
        precisions = torch.tensor([rp[0][1]] + [p for (_r, p) in rp] + [rp[-1][1]]).to(
            values.device
        )

        # For efficient storage, we remove the zeros from quantile_data (it may start with zeros). So when converting it
        # back to a tensor, we need to pad it with zeros again.
        n_buckets = len(self.quantiles) - 1
        quantiles = torch.tensor(self.quantiles).to(values.device)
        quantile_data = torch.tensor(
            [pad_with_zeros(x, n_buckets) for x in self.quantile_data]
        ).to(values.device)

        values_is_1d = values.ndim == 1
        if values_is_1d:
            values = values.unsqueeze(1)

        # Get an object to slice into the tensor (along batch dimension)
        my_slice = slice(None) if batch_indices is None else batch_indices

        # Find the quantiles of these values (i.e. the values between 0 and 1)
        quantile_indices = torch.searchsorted(
            quantile_data[my_slice], values
        )  # shape [batch data_dim]
        quantiles = quantiles[quantile_indices]

        # Also get the precisions (which we do using a separate searchsorted, only over the range dividers)
        precision_indices = torch.searchsorted(
            ranges, quantiles
        )  # shape [batch data_dim]
        precisions = precisions[precision_indices]

        # If values was 1D, we want to return the result as 1D also (for convenience)
        if values_is_1d:
            quantiles = quantiles.squeeze(1)
            precisions = precisions.squeeze(1)

        return quantiles, precisions


# Example usage
if MAIN:
    # 2D data: each row represents the activations data of a different feature. We set some of it to zero, so we can
    # test the "JSON doesn't store zeros" feature of the FeatureStatistics class.
    device = get_device()
    N = 100_000
    data = torch.stack(
        [torch.rand(N).masked_fill(torch.rand(N) < 0.5, 0.0), torch.rand(N)]
    ).to(device)
    qc = FeatureStatistics.create(data)
    print(f"Total datapoints stored = {sum(len(x) for x in qc.quantile_data):_}")
    print(f"Total datapoints used to compute quantiles = {data.numel():_}\n")

    # 2D values tensor: each row applies to a different dataset
    values = torch.tensor([[0.0, 0.005, 0.02, 0.25], [0.75, 0.98, 0.995, 1.0]]).to(
        device
    )
    quantiles, precisions = qc.get_quantile(values)

    print("When 50% of data is 0, and 50% is Unif[0, 1]")
    for v, q, p in zip(values[0], quantiles[0], precisions[0]):
        print(f"Value: {v:.3f}, Precision: {p}, Quantile: {q:.{p - 2}%}")
    print("\nWhen 100% of data is Unif[0, 1]")
    for v, q, p in zip(values[1], quantiles[1], precisions[1]):
        print(f"Value: {v:.3f}, Precision: {p}, Quantile: {q:.{p - 2}%}")


def split_string(
    input_string: str,
    str1: str,
    str2: str,
) -> tuple[str, str]:
    assert (
        str1 in input_string and str2 in input_string
    ), "Error: str1 and str2 must be in input_string"
    pattern = f"({re.escape(str1)}.*?){re.escape(str2)}"
    match = re.search(pattern, input_string, flags=re.DOTALL)
    if match:
        between_str1_str2 = match.group(1)
        remaining_string = input_string.replace(between_str1_str2, "")
        return between_str1_str2, remaining_string
    else:
        return "", input_string


# Example usage
if MAIN:
    input_string = "The quick brown fox jumps over the lazy dog"
    str1 = "quick"
    str2 = "jumps"
    print(split_string(input_string, str1, str2))

    input_string = (
        "Before table <!-- Logits table --> Table <!-- Logits histogram --> After table"
    )
    str1 = r"<!-- Logits table -->"
    str2 = r"<!-- Logits histogram -->"
    print(split_string(input_string, str1, str2))


def apply_indent(
    text: str,
    prefix: str,
    first_line_indented: bool = True,
) -> str:
    """
    Indents a string at every new line (e.g. by spaces or tabs). This is useful for formatting when we're dumping things
    into an HTML file.

    Args:
        text:
            The text to indent
        prefix:
            The string to add at the start of each line
        first_line_indented:
            Whether the first line should be indented. If False, then the first line will be left as it is.
    """
    text_indented = "\n".join(prefix + line for line in text.strip().split("\n"))
    if not first_line_indented:
        text_indented = text_indented[len(prefix) :]

    return text_indented


def deep_union(
    dict1: dict[Any, Any], dict2: dict[Any, Any], path: str = ""
) -> dict[Any, Any]:
    """
    Returns a deep union of dictionaries (recursive operation). In other words, if `dict1` and `dict2` have the same
    keys then the value of that key will be the deep union of the values.

    Also, base case where one of the values is a list: we concatenate the lists together

    Examples:
        # Normal union
        deep_union({1: 2}, {3: 4}) == {1: 2, 3: 4}

        # 1-deep union
        deep_union(
            {1: {2: [3, 4]}},
            {1: {3: [3, 4]}}
        ) == {1: {2: [3, 4], 3: [3, 4]}}

        # 2-deep union
        assert deep_union(
            {"x": {"y": {"z": 1}}},
            {"x": {"y": {"w": 2}}},
        ) == {"x": {"y": {"z": 1, "w": 2}}}

        # list concatenation
        assert deep_union(
            {"x": [1, 2]},
            {"x": [3, 4]},
        ) == {"x": [1, 2, 3, 4]}

    The `path` accumulates the key/value paths from the recursive calls, so that we can see the full dictionary path
    which caused problems (not just the end-nodes).
    """
    result = dict1.copy()

    # For each new key & value in dict2
    for key2, value2 in dict2.items():
        # If key not in result, then we have a simple case: just add it to the result
        if key2 not in result:
            result[key2] = value2

        # If key in result, both should values be either dicts (then we recursively merge) or lists (then we concat). If
        # not, then we throw an error unconditionally (even if values are the same).
        else:
            value1 = result[key2]

            # Both dicts
            if isinstance(value1, dict) and isinstance(value2, dict):
                result[key2] = deep_union(value1, value2, path=f"{path}[{key2!r}]")

            # Both lists
            elif isinstance(value1, list) and isinstance(value2, list):
                result[key2] = value1 + value2

            # Error
            else:
                path1 = f"dict{path}[{key2!r}] = {value1!r}"
                path2 = f"dict{path}[{key2!r}] = {value2!r}"
                raise ValueError(f"Merge failed. Conflicting paths:\n{path1}\n{path2}")

    return result


if MAIN:
    # Normal union
    assert deep_union({1: 2}, {3: 4}) == {1: 2, 3: 4}

    # 1-deep union
    assert deep_union({1: {2: [3, 4]}}, {1: {3: [3, 4]}}) == {1: {2: [3, 4], 3: [3, 4]}}

    # 2-deep union
    assert deep_union(
        {"x": {"y": {"z": 1}}},
        {"x": {"y": {"w": 2}}},
    ) == {"x": {"y": {"z": 1, "w": 2}}}

    # list concatenation
    assert deep_union(
        {"x": [1, 2]},
        {"x": [3, 4]},
    ) == {"x": [1, 2, 3, 4]}


# class RollingStats:
#     '''
#     This class helps us compute rolling stats of a dataset as we feed in activations, without ever having to store the
#     entire batch in data.
#     '''
#     def __init__(self):
#         self.n = 0
#         self.x_sum = 0.0
#         self.x2_sum = 0.0
#         self.x3_sum = 0.0
#         self.x4_sum = 0.0
#         self.frac_nonzero = 0.0
#         self.max = 0.0

#     def update(self, x: Tensor):
#         x_frac_nonzero = x.nonzero().size(0) / x.numel()
#         x_n = x.numel()
#         self.frac_nonzero = (self.n * self.frac_nonzero + x_n * x_frac_nonzero) / (self.n + x_n)
#         self.n += x.numel()
#         self.x_sum += x.sum().item()
#         self.x2_sum += x.pow(2).sum().item()
#         self.x3_sum += x.pow(3).sum().item()
#         self.x4_sum += x.pow(4).sum().item()
#         self.max = max(self.max, x.max().item())

#     @property
#     def skew(self) -> float:
#         raise NotImplementedError

#     @property
#     def kurtosis(self) -> float:
#         raise NotImplementedError


def resolve_correlation_accumulation_device(
    device: str, policy: str = "auto"
) -> torch.device:
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "correlation_accumulation_device='cuda' requires CUDA to be available"
            )
        return torch.device("cuda")
    if policy != "auto":
        raise ValueError(
            "correlation_accumulation_device must be one of 'auto', 'cpu', or 'cuda'"
        )
    correlation_device = torch.device(device)
    if correlation_device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return correlation_device


class RollingCorrCoef:
    """
    This class helps compute corrcoef (Pearson & cosine sim) between 2 batches of vectors, without having to store the
    entire batch in memory.

    How exactly does it work? We exploit the formula below (x, y assumed to be vectors here), which writes corrcoefs in
    terms of scalars which can be computed on a rolling basis.

        cos_sim(x, y) = xy_sum / ((x2_sum ** 0.5) * (y2_sum ** 0.5))

        pearson_corrcoef(x, y) = num / denom
            num = n * xy_sum - x_sum * y_sum
            denom = (n * x2_sum - x_sum ** 2) ** 0.5 * (n * y2_sum - y_sum ** 2) ** 0.5

    This class batches this computation, i.e. x.shape = (X, N), y.shape = (Y, N), where (for example) we have:
        N = batch_size * seq_len, i.e. it's the number of datapoints we have
        x = features of our original encoder
        y = features of our encoder-B, or neurons of our original model (the thing we're topk-ing over)

    So we can e.g. compute the correlation coefficients for every combination of feature in encoder and model neurons,
    then take topk to find the most correlated neurons for each feature.
    """

    def __init__(
        self,
        indices: list[int] | None = None,
        with_self: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Args:
            indices: list[int]
                If supplied, we map y indices (from 0 to y.shape) to these values. Useful when we're working with e.g.
                a dataset which didn't start from 0, and we want the "true indices".
            with_self: bool
                If True, then we take X and Y as coming from the same dataset. This saves us some computation, and it
                also means we exclude the diagonal from final topk (since correlation with self is always 1).
        """
        self.n = 0
        self.X = None
        self.Y = None
        self.indices = indices
        self.with_self = with_self
        self.dtype = dtype
        self.device = device

    def update(
        self,
        x: Float[Tensor, "X N"],
        y: Float[Tensor, "Y N"],
        *,
        perf_enabled: bool = False,
        perf_label: str = "corrcoef",
        perf_context: dict[str, Any] | None = None,
    ) -> None:
        # Get values of x and y, and check for consistency with each other & with previous values
        assert x.ndim == 2 and y.ndim == 2, "Both x and y should be 2D"
        X, Nx = x.shape
        Y, Ny = y.shape
        assert (
            Nx == Ny
        ), "Error: x and y should have the same size in the last dimension"
        if self.with_self:
            assert X == Y, "If with_self is True, then x and y should be the same shape"
        if self.X is not None:
            assert (
                X == self.X
            ), "Error: updating a corrcoef object with different sized dataset."
        if self.Y is not None:
            assert (
                Y == self.Y
            ), "Error: updating a corrcoef object with different sized dataset."
        self.X = X
        self.Y = Y

        perf_fields = dict(perf_context or {})
        perf_fields.update(
            {
                "corrcoef_label": perf_label,
                "corrcoef_device": str(self.device),
                "corrcoef_dtype": str(self.dtype),
                "corrcoef_with_self": self.with_self,
                "input_x_layout": tensor_runtime_metadata(x) if perf_enabled else None,
                "input_y_layout": tensor_runtime_metadata(y) if perf_enabled else None,
            }
        )
        same_input = x is y
        with timed_stage(
            perf_enabled,
            f"rolling_{perf_label}_materialize_x",
            device=str(self.device),
            capture_runtime_metrics=True,
            **perf_fields,
            same_input=same_input,
        ):
            x = x.to(dtype=self.dtype, device=self.device)
        if same_input:
            y = x
        else:
            with timed_stage(
                perf_enabled,
                f"rolling_{perf_label}_materialize_y",
                device=str(self.device),
                capture_runtime_metrics=True,
                **perf_fields,
                same_input=same_input,
            ):
                y = y.to(dtype=self.dtype, device=self.device)

        if perf_enabled:
            log_perf_event(
                "rolling_tensor_layout",
                stage=f"rolling_{perf_label}_materialized_inputs",
                **perf_fields,
                same_input=same_input,
                materialized_x_layout=tensor_runtime_metadata(x),
                materialized_y_layout=tensor_runtime_metadata(y),
            )

        # If this is the first update step, then we need to initialise the sums
        if self.n == 0:
            with timed_stage(
                perf_enabled,
                f"rolling_{perf_label}_init_accumulators",
                device=str(x.device),
                capture_runtime_metrics=True,
                **perf_fields,
                x_rows=X,
                y_rows=Y,
                n_cols=Nx,
            ):
                self.x_sum = torch.zeros(X, device=x.device, dtype=self.dtype)
                self.xy_sum = torch.zeros(X, Y, device=x.device, dtype=self.dtype)
                self.x2_sum = torch.zeros(X, device=x.device, dtype=self.dtype)
                if not self.with_self:
                    self.y_sum = torch.zeros(Y, device=y.device, dtype=self.dtype)
                    self.y2_sum = torch.zeros(Y, device=y.device, dtype=self.dtype)

        # Next, update the sums
        self.n += x.shape[-1]
        with timed_stage(
            perf_enabled,
            f"rolling_{perf_label}_x_sum_update",
            device=str(x.device),
            capture_runtime_metrics=True,
            **perf_fields,
        ):
            self.x_sum += x.sum(dim=-1)
        with timed_stage(
            perf_enabled,
            f"rolling_{perf_label}_xy_sum_update",
            device=str(x.device),
            capture_runtime_metrics=True,
            **perf_fields,
        ):
            self.xy_sum.addmm_(x, y.mT)
        with timed_stage(
            perf_enabled,
            f"rolling_{perf_label}_x2_sum_update",
            device=str(x.device),
            capture_runtime_metrics=True,
            **perf_fields,
        ):
            self.x2_sum += (x * x).sum(dim=-1)
        if not self.with_self:
            with timed_stage(
                perf_enabled,
                f"rolling_{perf_label}_y_sum_update",
                device=str(y.device),
                capture_runtime_metrics=True,
                **perf_fields,
            ):
                self.y_sum += y.sum(dim=-1)
            with timed_stage(
                perf_enabled,
                f"rolling_{perf_label}_y2_sum_update",
                device=str(y.device),
                capture_runtime_metrics=True,
                **perf_fields,
            ):
                self.y2_sum += (y * y).sum(dim=-1)

    def corrcoef(
        self,
    ) -> tuple[Float[Tensor, "X Y"], Float[Tensor, "X Y"]]:
        """
        Computes the correlation coefficients between x and y, using the formulae given in the class docstring.
        """
        # Get y_sum and y2_sum (to deal with the cases when with_self is True/False)
        if self.with_self:
            self.y_sum = self.x_sum
            self.y2_sum = self.x2_sum

        # Compute cosine sim
        cossim_numer = self.xy_sum
        cossim_denom = torch.sqrt(torch.outer(self.x2_sum, self.y2_sum)) + 1e-6
        cossim = cossim_numer / cossim_denom

        # Compute pearson corrcoef
        pearson_numer = self.n * self.xy_sum - torch.outer(self.x_sum, self.y_sum)
        pearson_denom = (
            torch.sqrt(
                torch.outer(
                    self.n * self.x2_sum - self.x_sum**2,
                    self.n * self.y2_sum - self.y_sum**2,
                )
            )
            + 1e-6
        )
        pearson = pearson_numer / pearson_denom

        # If with_self, we exclude the diagonal
        if self.with_self:
            d = cossim.shape[0]
            cossim[range(d), range(d)] = 0.0
            pearson[range(d), range(d)] = 0.0

        return pearson, cossim

    def topk_pearson(
        self,
        k: int,
        largest: bool = True,
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
        """
        Takes topk of the pearson corrcoefs over the y-dimension (e.g. giving us the most correlated neurons or most
        correlated encoder-B features for each encoder feature).

        Args:
            k: int
                Number of top indices to take (usually 3, for the left-hand tables)
            largest: bool
                If True, then we take the largest k indices. If False, then we take the smallest k indices.

        Returns:
            pearson_indices: list[list[int]]
                y-indices which are most correlated with each x-index (in terms of pearson corrcoef)
            pearson_values: list[list[float]]
                Values of pearson corrcoef for each of the topk indices
            cossim_values: list[list[float]]
                Values of cosine similarity for each of the topk indices
        """
        # Get correlation coefficient, using the formula from corrcoef method
        pearson, cossim = self.corrcoef()

        # Get the top pearson values
        pearson_topk = TopK(tensor=pearson, k=k, largest=largest)  # shape (X, k)

        # Get the cossim values for the top pearson values, i.e. cossim_values[X, k] = cossim[X, pearson_indices[X, k]]
        cossim_values = eindex(cossim, pearson_topk.indices, "X [X k]")

        # If we've supplied indices, use them to offset the returned pearson topk indices
        indices = pearson_topk.indices.tolist()
        if self.indices is not None:
            indices = [[self.indices[i] for i in x] for x in indices]

        return indices, pearson_topk.values.tolist(), cossim_values.tolist()


def build_activation_histogram_titles(
    data: Tensor,
    valid_mask: Bool[Tensor, "samples"] | Bool[Tensor, "items samples"] | None = None,
) -> list[str]:
    """Build activation-density titles using valid-token counts when a mask is provided."""
    if data.ndim != 2:
        raise ValueError(
            "build_activation_histogram_titles expects a 2D tensor with shape (items, samples)"
        )

    positive_mask = data > 0
    if valid_mask is None:
        positive_counts = positive_mask.sum(dim=-1)
        valid_counts = torch.full(
            (data.shape[0],),
            data.shape[-1],
            dtype=torch.int64,
            device=data.device,
        )
    else:
        if valid_mask.ndim == 1:
            if valid_mask.shape[0] != data.shape[1]:
                raise ValueError(
                    "A 1D valid_mask must have the same sample dimension as data"
                )
            valid_mask = valid_mask.unsqueeze(0).expand(data.shape[0], -1)
        elif valid_mask.ndim == 2:
            if tuple(valid_mask.shape) != tuple(data.shape):
                raise ValueError("A 2D valid_mask must have the same shape as data")
        else:
            raise ValueError("valid_mask must be 1D or 2D when provided")

        valid_mask = valid_mask.to(device=data.device, dtype=torch.bool)
        positive_counts = (positive_mask & valid_mask).sum(dim=-1)
        valid_counts = valid_mask.sum(dim=-1).clamp_min(1)

    densities = [
        float(nonzero_count) / float(valid_count)
        for nonzero_count, valid_count in zip(
            positive_counts.tolist(),
            valid_counts.tolist(),
        )
    ]
    return build_activation_histogram_titles_from_densities(densities)


def build_activation_histogram_titles_from_densities(
    densities: Iterable[float],
) -> list[str]:
    """Build activation histogram titles from precomputed per-feature densities."""
    return [f"ACTIVATIONS<br>DENSITY = {float(density):.3%}" for density in densities]


@dataclass_json
@dataclass
class HistogramData:
    """
    This class contains all the data necessary to construct a single histogram (e.g. the logits or feat acts histogram).
    See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    We don't need to store the entire `data` tensor, so we initialize instances of this class using the `from_data`
    method, which computes statistics from the input data tensor then discards it.

        bar_heights: The height of each bar in the histogram
        bar_values: The value of each bar in the histogram
        tick_vals: The tick values we want to use for the histogram
    """

    bar_heights: list[float] = field(default_factory=list)
    bar_values: list[float] = field(default_factory=list)
    tick_vals: list[float] = field(default_factory=list)
    title: str | None = None

    def to_row_dict(self) -> dict[str, object]:
        return {
            "bar_heights": self.bar_heights,
            "bar_values": self.bar_values,
            "tick_vals": self.tick_vals,
            "title": self.title,
        }

    @classmethod
    def from_data(
        cls: Type[HistogramDataT],
        data: Tensor,
        n_bins: int,
        tickmode: Literal["ints", "5 ticks"],
        title: str | None,
    ) -> HistogramDataT:
        """
        Args:
            data: 1D tensor of data which will be turned into histogram
            n_bins: Number of bins in the histogram
            line_posn: list of possible positions of vertical lines we want to put on the histogram

        Returns a HistogramData object, with data computed from the inputs. This is to support the goal of only storing
        the minimum necessary data (and making it serializable, for JSON saving).
        """
        # There might be no data, if the feature never activates
        if data.numel() == 0:
            return cls()

        # Get min and max of data
        max_value = data.max().item()
        min_value = data.min().item()

        # Divide range up into 40 bins
        bin_size = (max_value - min_value) / n_bins
        bin_edges = torch.linspace(min_value, max_value, n_bins + 1)
        # Calculate the heights of each bin
        bar_heights = torch.histc(data, bins=n_bins).int().tolist()
        bar_values = [round(x, 5) for x in (bin_edges[:-1] + bin_size / 2).tolist()]

        tick_vals = cls._tick_values(max_value, min_value, tickmode)

        return cls(  # type: ignore
            bar_heights=bar_heights,  # type: ignore
            bar_values=bar_values,  # type: ignore
            tick_vals=tick_vals,  # type: ignore
            title=title,  # type: ignore
        )

    @classmethod
    def from_data_batch(
        cls: Type[HistogramDataT],
        data: Tensor,
        n_bins: int,
        tickmode: Literal["ints", "5 ticks"],
        title: str | None,
        row_batch_size: int = 64,
        positive_only: bool = False,
        titles: Sequence[str | None] | None = None,
        backend: Literal["torch"] = "torch",
    ) -> list[HistogramDataT]:
        """Create one histogram per row of a 2D tensor using batched binning."""
        histogram_rows = cls._from_data_batch_rows(
            data=data,
            n_bins=n_bins,
            tickmode=tickmode,
            title=title,
            row_batch_size=row_batch_size,
            positive_only=positive_only,
            titles=titles,
            backend=backend,
        )
        return [cls(**histogram_row) for histogram_row in histogram_rows]  # type: ignore[arg-type]

    @classmethod
    def from_data_batch_arrow_table(
        cls: Type[HistogramDataT],
        data: Tensor,
        n_bins: int,
        tickmode: Literal["ints", "5 ticks"],
        title: str | None,
        row_batch_size: int = 64,
        positive_only: bool = False,
        titles: Sequence[str | None] | None = None,
        backend: Literal["torch"] = "torch",
    ) -> Any:
        pyarrow = importlib.import_module("pyarrow")
        if not positive_only:
            tensor_table = cls._from_dense_data_batch_arrow_table(
                data=data,
                n_bins=n_bins,
                tickmode=tickmode,
                title=title,
                row_batch_size=row_batch_size,
                titles=titles,
            )
            if tensor_table is not None:
                return tensor_table
            # The dense lane computes in float32; keep the rare constant-row fallback
            # on the same dtype when the caller staged reduced-precision data (callers
            # historically pre-cast to float32 themselves).
            if data.dtype in (torch.bfloat16, torch.float16):
                data = data.to(torch.float32)

        histogram_rows = cls._from_data_batch_rows(
            data=data,
            n_bins=n_bins,
            tickmode=tickmode,
            title=title,
            row_batch_size=row_batch_size,
            positive_only=positive_only,
            titles=titles,
            backend=backend,
        )
        return pyarrow.table(
            {
                "row_index": pyarrow.array(range(len(histogram_rows))),
                "bar_heights": pyarrow.array(
                    [row["bar_heights"] for row in histogram_rows]
                ),
                "bar_values": pyarrow.array(
                    [row["bar_values"] for row in histogram_rows]
                ),
                "tick_vals": pyarrow.array(
                    [row["tick_vals"] for row in histogram_rows]
                ),
                "title": pyarrow.array([row["title"] for row in histogram_rows]),
            }
        )

    @classmethod
    def from_flat_valid_data_batch_arrow_table(
        cls: Type[HistogramDataT],
        flat_data: Tensor,
        valid_flat_indices: Tensor,
        n_bins: int,
        tickmode: Literal["ints", "5 ticks"],
        titles: Sequence[str | None] | None,
        compute_device: str | torch.device | None = None,
        row_chunk_size: int = 128,
        row_batch_size: int = 64,
    ) -> Any:
        """Chunked-device positive-only histogram table from a flat activation matrix.

        Equivalent to masking ``flat_data`` with the valid-position mask and calling
        `from_data_batch_arrow_table(..., positive_only=True)`: for the positive-only
        lane, gathering the valid columns yields identical histograms to zero-masking
        the invalid columns (only ``> 0`` values participate, and rows whose valid
        values are all non-positive fall back to the same default/constant-row paths).
        Row chunks are staged onto ``compute_device`` so the binning runs there and
        only compact per-row outputs are transferred back.
        """
        pyarrow = importlib.import_module("pyarrow")
        device = (
            torch.device(compute_device)
            if compute_device is not None
            else flat_data.device
        )
        n_rows = int(flat_data.shape[0])
        n_valid = int(valid_flat_indices.numel())
        if titles is not None and len(titles) != n_rows:
            raise ValueError("titles must match the number of data rows")
        valid_indices_dev = valid_flat_indices.to(device)

        histogram_rows: list[dict[str, object]] = []
        row_chunk_size = max(1, row_chunk_size)
        for row_start in range(0, n_rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, n_rows)
            chunk = flat_data[row_start:row_end].to(device)
            if n_valid > 0:
                valid_chunk = chunk.index_select(-1, valid_indices_dev)
            else:
                # Reproduce the previous zero-masked behavior: rows with no valid
                # positions produce the default (empty) histogram rows.
                valid_chunk = torch.zeros(
                    (row_end - row_start, 1),
                    dtype=chunk.dtype,
                    device=device,
                )
            histogram_rows.extend(
                cls._from_data_batch_rows(
                    data=valid_chunk,
                    n_bins=n_bins,
                    tickmode=tickmode,
                    title=None,
                    row_batch_size=row_batch_size,
                    positive_only=True,
                    titles=(
                        list(titles[row_start:row_end]) if titles is not None else None
                    ),
                )
            )
            del chunk, valid_chunk

        return pyarrow.table(
            {
                "row_index": pyarrow.array(range(len(histogram_rows))),
                "bar_heights": pyarrow.array(
                    [row["bar_heights"] for row in histogram_rows]
                ),
                "bar_values": pyarrow.array(
                    [row["bar_values"] for row in histogram_rows]
                ),
                "tick_vals": pyarrow.array(
                    [row["tick_vals"] for row in histogram_rows]
                ),
                "title": pyarrow.array([row["title"] for row in histogram_rows]),
            }
        )

    @classmethod
    def _from_dense_data_batch_arrow_table(
        cls: Type[HistogramDataT],
        data: Tensor,
        n_bins: int,
        tickmode: Literal["ints", "5 ticks"],
        title: str | None,
        row_batch_size: int,
        titles: Sequence[str | None] | None,
    ) -> Any | None:
        """Build a histogram Arrow table directly from dense row-batched tensors.

        This is the deferred logits-histogram fast path. It avoids materializing the
        intermediate Python row dictionaries that the eager object path needs.
        Constant rows are rare for logits and fall back to the reference row path.
        """
        pyarrow = importlib.import_module("pyarrow")
        if data.ndim != 2:
            raise ValueError(
                "from_data_batch expects a 2D tensor with shape (items, samples)"
            )
        if data.numel() == 0:
            return pyarrow.table(
                {
                    "row_index": pyarrow.array([], type=pyarrow.int64()),
                    "bar_heights": pyarrow.array([]),
                    "bar_values": pyarrow.array([]),
                    "tick_vals": pyarrow.array([]),
                    "title": pyarrow.array([], type=pyarrow.string()),
                }
            )
        if titles is not None and len(titles) != data.shape[0]:
            raise ValueError("titles must match the number of data rows")

        # Cast per row batch instead of materializing a full float32 copy of the
        # input (a (feats, d_vocab) logits matrix costs gigabytes at large vocabs).
        # float32 embeds bf16/f16 exactly, so per-batch casting is bit-identical to
        # the previous whole-tensor cast; max/min are element selections, so casting
        # the reduced vectors matches reducing the cast tensor.
        compute_dtype = torch.float32
        max_values = data.max(dim=-1).values.to(compute_dtype)
        min_values = data.min(dim=-1).values.to(compute_dtype)
        if not bool((max_values != min_values).all().item()):
            return None

        row_batch_size = max(1, row_batch_size)
        bar_heights_chunks = []
        bar_values_chunks = []

        bin_offsets = (
            torch.arange(n_bins, dtype=compute_dtype, device=data.device) + 0.5
        )
        scatter_source = torch.ones((), dtype=torch.int32, device=data.device)
        for row_start in range(0, data.shape[0], row_batch_size):
            row_end = min(row_start + row_batch_size, data.shape[0])
            batch = data[row_start:row_end].to(compute_dtype)
            row_max = max_values[row_start:row_end]
            row_min = min_values[row_start:row_end]
            row_range = row_max - row_min

            scaled_bins = torch.floor(
                (batch - row_min[:, None]) * n_bins / row_range[:, None]
            ).to(torch.int64)
            scaled_bins = scaled_bins.clamp_(0, n_bins - 1)
            bar_heights = torch.zeros(
                (batch.shape[0], n_bins), dtype=torch.int32, device=batch.device
            )
            bar_heights.scatter_add_(
                1,
                scaled_bins,
                scatter_source.expand_as(scaled_bins),
            )

            bin_size = row_range / n_bins
            bar_values = row_min[:, None] + bin_offsets[None, :] * bin_size[:, None]
            bar_heights_chunks.append(bar_heights.cpu())
            bar_values_chunks.append(bar_values.cpu())

        bar_heights_array = np.ascontiguousarray(torch.cat(bar_heights_chunks).numpy())
        bar_values_array = np.ascontiguousarray(
            np.round(torch.cat(bar_values_chunks).numpy().astype(np.float64), 5)
        )
        title_values = list(titles) if titles is not None else [title] * data.shape[0]
        tick_values_array = cls._tick_values_arrow_array(
            pyarrow,
            max_values=max_values,
            min_values=min_values,
            tickmode=tickmode,
        )

        return pyarrow.table(
            {
                "row_index": pyarrow.array(
                    np.arange(data.shape[0], dtype=np.int64), type=pyarrow.int64()
                ),
                "bar_heights": pyarrow.FixedSizeListArray.from_arrays(
                    pyarrow.array(bar_heights_array.reshape(-1), type=pyarrow.int32()),
                    n_bins,
                ),
                "bar_values": pyarrow.FixedSizeListArray.from_arrays(
                    pyarrow.array(bar_values_array.reshape(-1), type=pyarrow.float64()),
                    n_bins,
                ),
                "tick_vals": tick_values_array,
                "title": pyarrow.array(title_values),
            }
        )

    @classmethod
    def _tick_values_arrow_array(
        cls,
        pyarrow: Any,
        max_values: Tensor,
        min_values: Tensor,
        tickmode: Literal["ints", "5 ticks"],
    ) -> Any:
        if tickmode == "ints":
            return pyarrow.array(
                [
                    cls._tick_values(float(max_value), float(min_value), tickmode)
                    for max_value, min_value in zip(
                        max_values.detach().cpu().tolist(),
                        min_values.detach().cpu().tolist(),
                    )
                ]
            )

        max_array = max_values.detach().cpu().numpy().astype(np.float64, copy=False)
        min_array = min_values.detach().cpu().numpy().astype(np.float64, copy=False)
        positive_dominant = max_array > -min_array
        tick_step_tenths = np.empty(max_array.shape[0], dtype=np.int64)
        tick_step_tenths[positive_dominant] = (
            1e-4 + max_array[positive_dominant] / 0.3
        ).astype(np.int64)
        tick_step_tenths[~positive_dominant] = (
            1e-4 + -min_array[~positive_dominant] / 0.3
        ).astype(np.int64)
        tick_step_tenths = np.maximum(tick_step_tenths, 1)

        tick_step_values = tick_step_tenths.astype(np.float64) / 10.0
        denominator = tick_step_values + 1e-6
        negative_counts = np.empty(max_array.shape[0], dtype=np.int64)
        positive_counts = np.empty(max_array.shape[0], dtype=np.int64)
        negative_counts[positive_dominant] = (
            -min_array[positive_dominant] / denominator[positive_dominant]
        ).astype(np.int64)
        negative_counts[~positive_dominant] = 3
        positive_counts[positive_dominant] = 3
        positive_counts[~positive_dominant] = (
            max_array[~positive_dominant] / denominator[~positive_dominant]
        ).astype(np.int64)
        negative_counts = np.maximum(negative_counts, 0)
        positive_counts = np.maximum(positive_counts, 0)

        lengths = negative_counts + 1 + positive_counts
        offsets = np.empty(max_array.shape[0] + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        values = np.zeros(int(offsets[-1]), dtype=np.float64)
        row_indices = np.arange(max_array.shape[0], dtype=np.int64)

        total_negative = int(negative_counts.sum())
        if total_negative:
            negative_rows = np.repeat(row_indices, negative_counts)
            negative_starts = np.repeat(offsets[:-1], negative_counts)
            negative_row_starts = np.repeat(
                np.cumsum(np.concatenate(([0], negative_counts[:-1]))),
                negative_counts,
            )
            negative_positions = (
                np.arange(total_negative, dtype=np.int64) - negative_row_starts
            )
            values[negative_starts + negative_positions] = (
                -tick_step_tenths[negative_rows]
                * (negative_counts[negative_rows] - negative_positions)
                / 10
            )

        zero_positions = offsets[:-1] + negative_counts
        values[zero_positions] = 0.0

        total_positive = int(positive_counts.sum())
        if total_positive:
            positive_rows = np.repeat(row_indices, positive_counts)
            positive_starts = np.repeat(
                offsets[:-1] + negative_counts + 1, positive_counts
            )
            positive_row_starts = np.repeat(
                np.cumsum(np.concatenate(([0], positive_counts[:-1]))),
                positive_counts,
            )
            positive_positions = (
                np.arange(total_positive, dtype=np.int64) - positive_row_starts
            )
            values[positive_starts + positive_positions] = (
                tick_step_tenths[positive_rows] * (positive_positions + 1) / 10
            )

        return pyarrow.ListArray.from_arrays(
            pyarrow.array(offsets, type=pyarrow.int32()),
            pyarrow.array(values, type=pyarrow.float64()),
        )

    @classmethod
    def from_arrow_table(cls: Type[HistogramDataT], table: Any) -> list[HistogramDataT]:
        columns = table.to_pydict()
        row_count = len(columns.get("row_index", []))
        return [
            cls(
                bar_heights=[
                    float(value) for value in columns["bar_heights"][row_index]
                ],  # type: ignore[index]
                bar_values=[float(value) for value in columns["bar_values"][row_index]],  # type: ignore[index]
                tick_vals=[float(value) for value in columns["tick_vals"][row_index]],  # type: ignore[index]
                title=columns["title"][row_index],  # type: ignore[index]
            )
            for row_index in range(row_count)
        ]

    @classmethod
    def _from_data_batch_rows(
        cls: Type[HistogramDataT],
        data: Tensor,
        n_bins: int,
        tickmode: Literal["ints", "5 ticks"],
        title: str | None,
        row_batch_size: int = 64,
        positive_only: bool = False,
        titles: Sequence[str | None] | None = None,
        backend: Literal["torch"] = "torch",
    ) -> list[dict[str, object]]:
        """Create one histogram row dict per row of a 2D tensor using batched binning."""
        if data.ndim != 2:
            raise ValueError(
                "from_data_batch expects a 2D tensor with shape (items, samples)"
            )
        if data.numel() == 0:
            return []
        if titles is not None and len(titles) != data.shape[0]:
            raise ValueError("titles must match the number of data rows")
        data = data.to(torch.float32)

        histograms: list[dict[str, object] | None] = [None] * data.shape[0]
        row_batch_size = max(1, row_batch_size)

        if positive_only:
            max_values = data.max(dim=-1).values
            row_has_values = max_values > 0

            positive_row_indices = torch.nonzero(
                row_has_values, as_tuple=False
            ).flatten()
            for row_index_batch in positive_row_indices.split(row_batch_size):
                batch = data.index_select(0, row_index_batch)
                positive_coords = torch.nonzero(batch > 0, as_tuple=False)
                if positive_coords.numel() == 0:
                    continue

                valid_rows = positive_coords[:, 0]
                valid_cols = positive_coords[:, 1]
                valid_values = batch[valid_rows, valid_cols]

                row_max = max_values.index_select(0, row_index_batch)
                row_min = torch.full_like(row_max, float("inf"))
                row_min.scatter_reduce_(
                    0,
                    valid_rows,
                    valid_values,
                    reduce="amin",
                    include_self=True,
                )
                row_range = row_max - row_min
                nonconstant_mask = row_range != 0

                if nonconstant_mask.any():
                    nonconstant_valid = nonconstant_mask.index_select(0, valid_rows)
                    valid_rows_nc = valid_rows[nonconstant_valid]
                    valid_values_nc = valid_values[nonconstant_valid]
                    row_min_nc = row_min.index_select(0, valid_rows_nc)
                    row_range_nc = row_range.index_select(0, valid_rows_nc)

                    scaled_bins = torch.floor(
                        (valid_values_nc - row_min_nc) * n_bins / row_range_nc
                    ).to(torch.int64)
                    scaled_bins = scaled_bins.clamp_(0, n_bins - 1)

                    flat_bin_indices = valid_rows_nc * n_bins + scaled_bins
                    flat_bar_heights = torch.zeros(
                        batch.shape[0] * n_bins,
                        dtype=torch.int64,
                        device=batch.device,
                    )
                    flat_bar_heights.scatter_add_(
                        0,
                        flat_bin_indices,
                        torch.ones_like(flat_bin_indices, dtype=torch.int64),
                    )
                    bar_heights = flat_bar_heights.view(batch.shape[0], n_bins)

                    bin_size = row_range / n_bins
                    bin_offsets = (
                        torch.arange(n_bins, dtype=batch.dtype, device=batch.device)
                        + 0.5
                    )
                    bar_values = (
                        row_min[:, None] + bin_offsets[None, :] * bin_size[:, None]
                    )

                    bar_heights_lists = bar_heights.cpu().tolist()
                    bar_values_lists = [
                        [round(float(value), 5) for value in row]
                        for row in bar_values.cpu().tolist()
                    ]
                    max_list = row_max.cpu().tolist()
                    min_list = row_min.cpu().tolist()
                    nonconstant_list = nonconstant_mask.cpu().tolist()
                    for output_index, source_index in enumerate(
                        row_index_batch.cpu().tolist()
                    ):
                        if not nonconstant_list[output_index]:
                            continue
                        histograms[source_index] = {
                            "bar_heights": bar_heights_lists[output_index],
                            "bar_values": bar_values_lists[output_index],
                            "tick_vals": cls._tick_values(
                                max_list[output_index],
                                min_list[output_index],
                                tickmode,
                            ),
                            "title": (
                                titles[source_index] if titles is not None else title
                            ),
                        }

            row_has_values_list = row_has_values.cpu().tolist()
            for row_index, histogram in enumerate(histograms):
                if histogram is None:
                    if not row_has_values_list[row_index]:
                        histograms[row_index] = cls().to_row_dict()
                        continue
                    row_data = data[row_index]
                    row_data = row_data[row_data > 0]
                    histograms[row_index] = cls.from_data(
                        row_data,
                        n_bins=n_bins,
                        tickmode=tickmode,
                        title=titles[row_index] if titles is not None else title,
                    ).to_row_dict()

            return cast(list[dict[str, object]], histograms)

        value_mask = torch.ones_like(data, dtype=torch.bool)
        row_has_values = value_mask.any(dim=-1)
        max_values = data.max(dim=-1).values
        min_values = data.min(dim=-1).values
        nonconstant_mask = max_values != min_values

        batchable_mask = row_has_values & nonconstant_mask
        if batchable_mask.any():
            row_indices = torch.nonzero(batchable_mask, as_tuple=False).flatten()
            for row_index_batch in row_indices.split(row_batch_size):
                batch = data.index_select(0, row_index_batch)
                batch_mask = value_mask.index_select(0, row_index_batch)
                row_max = max_values.index_select(0, row_index_batch)
                row_min = min_values.index_select(0, row_index_batch)
                row_range = row_max - row_min

                scaled_bins = torch.floor(
                    (batch - row_min[:, None]) * n_bins / row_range[:, None]
                ).to(torch.int64)
                scaled_bins = scaled_bins.clamp_(0, n_bins - 1)
                bar_heights = torch.zeros(
                    (batch.shape[0], n_bins), dtype=torch.int64, device=batch.device
                )
                bar_heights.scatter_add_(
                    1,
                    scaled_bins,
                    batch_mask.to(torch.int64),
                )

                bin_size = row_range / n_bins
                bin_offsets = (
                    torch.arange(n_bins, dtype=batch.dtype, device=batch.device) + 0.5
                )
                bar_values = row_min[:, None] + bin_offsets[None, :] * bin_size[:, None]

                bar_heights_lists = bar_heights.cpu().tolist()
                bar_values_lists = [
                    [round(float(value), 5) for value in row]
                    for row in bar_values.cpu().tolist()
                ]
                max_list = row_max.cpu().tolist()
                min_list = row_min.cpu().tolist()
                for output_index, source_index in enumerate(
                    row_index_batch.cpu().tolist()
                ):
                    histograms[source_index] = {
                        "bar_heights": bar_heights_lists[output_index],
                        "bar_values": bar_values_lists[output_index],
                        "tick_vals": cls._tick_values(
                            max_list[output_index], min_list[output_index], tickmode
                        ),
                        "title": titles[source_index] if titles is not None else title,
                    }

        for row_index, histogram in enumerate(histograms):
            if histogram is None:
                if not row_has_values[row_index]:
                    histograms[row_index] = cls().to_row_dict()
                    continue
                row_data = data[row_index]
                if positive_only:
                    row_data = row_data[value_mask[row_index]]
                histograms[row_index] = cls.from_data(
                    row_data,
                    n_bins=n_bins,
                    tickmode=tickmode,
                    title=titles[row_index] if titles is not None else title,
                ).to_row_dict()

        return cast(list[dict[str, object]], histograms)

    @staticmethod
    def _tick_values(
        max_value: float,
        min_value: float,
        tickmode: Literal["ints", "5 ticks"],
    ) -> list[float]:
        assert tickmode in ["ints", "5 ticks"]
        if tickmode == "ints":
            top_tickval = int(max_value)
            return list(range(top_tickval + 1))

        if max_value > -min_value:
            tick_step_tenths = max(1, int(1e-4 + max_value / (3 * 0.1)))
            num_positive_ticks = 3
            num_negative_ticks = int(-min_value / (tick_step_tenths / 10 + 1e-6))
        else:
            tick_step_tenths = max(1, int(1e-4 + -min_value / (3 * 0.1)))
            num_negative_ticks = 3
            num_positive_ticks = int(max_value / (tick_step_tenths / 10 + 1e-6))

        tick_vals = []
        for i in range(num_negative_ticks, 0, -1):
            tick_vals.append(-tick_step_tenths * i / 10)
        tick_vals.append(0)
        for i in range(1, 1 + num_positive_ticks):
            tick_vals.append(tick_step_tenths * i / 10)
        return tick_vals


def max_or_1(mylist: Sequence[float | int], abs: bool = False) -> float | int:
    """
    Returns max of a list, or 1 if the list is empty.

    Args:
        mylist: list of numbers
        abs: If True, then we take the max of the absolute values of the list
    """
    if len(mylist) == 0:
        return 1

    if abs:
        return max(max(x, -x) for x in mylist)
    else:
        return max(mylist)
