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
from sae_dashboard.utils_fns import TopK, k_largest_indices, random_range_indices
from sae_dashboard.vector_vis_data import VectorVisConfig

Arr = np.ndarray


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
        if self.cfg.feature_centric_layout.seq_cfg is None:
            raise ValueError(
                "Feature centric layout sequence config is required but not provided"
            )
        self.seq_cfg = self.cfg.feature_centric_layout.seq_cfg
        self.tokens = tokens
        self.W_U = W_U

        self.buffer, self.padded_buffer_width, self.seq_length = (
            self.get_buffer_and_padding(tokens)
        )

    @torch.inference_mode()
    def get_sequences_data(
        self,
        feat_acts: Float[Tensor, "batch seq"],
        feat_logits: Float[Tensor, "d_vocab"],
        resid_post: Float[Tensor, "batch seq d_model"],
        feature_resid_dir: Float[Tensor, "d_model"],
    ) -> SequenceMultiGroupData:
        """
        This function returns the data which is used to create the sequence visualizations (i.e. the right-hand column of
        the visualization). This is a multi-step process (the 4 steps are annotated in the code):

            (1) Find all the token groups (i.e. topk, bottomk, and quantile groups of activations). These are bold tokens.
            (2) Get the indices of all tokens we'll need data from, which includes a buffer around each bold token.
            (3) Extract the token IDs, feature activations & residual stream values for those positions
            (4) Compute the logit effect if this feature is ablated
                (4A) Use this to compute the most affected tokens by this feature (i.e. the vis hoverdata)
                (4B) Use this to compute the loss effect if this feature is ablated (i.e. the blue/red underlining)
            (5) Return all this data as a SequenceMultiGroupData object

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
            W_U:
                The model's unembedding matrix, which we'll use to get the logits.
            cfg:
                Feature visualization parameters, containing some important params e.g. num sequences per group.

        Returns:
            SequenceMultiGroupData
                This is a dataclass which contains a dict of SequenceGroupData objects, where each SequenceGroupData object
                contains the data for a particular group of sequences (i.e. the top-k, bottom-k, and the quantile groups).
        """

        # ! (1) Find the tokens from each group
        indices_dict, indices_bold, n_bold = self.get_indices_dict(
            self.buffer, feat_acts
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
            # ! (5) Store the results in a SequenceMultiGroupData object
            # Now that we've indexed everything, construct the batch of SequenceData objects
            sequence_multigroup_data = self.package_sequences_data(
                token_ids=token_ids,
                feat_acts_coloring=feat_acts_coloring,
                feat_logits=feat_logits,
                indices_dict=indices_dict,
                indices_bold=indices_bold,
            )

        return sequence_multigroup_data

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
        self, buffer: tuple[int, int] | None, feat_acts: Float[Tensor, "batch seq"]
    ):
        # Get the top-activating tokens
        indices = k_largest_indices(
            feat_acts, k=self.seq_cfg.top_acts_group_size, buffer=buffer
        ).cpu()
        indices_dict = {f"TOP ACTIVATIONS<br>MAX = {feat_acts.max():.3f}": indices}

        # Get all possible indices. Note, we need to be able to look 1 back (feature activation on prev token is needed for
        # computing loss effect on this token)
        if self.seq_cfg.n_quantiles > 0:
            quantiles = torch.linspace(
                0, feat_acts.max().item(), self.seq_cfg.n_quantiles + 1
            )
            for i in range(self.seq_cfg.n_quantiles - 1, -1, -1):
                lower, upper = quantiles[i : i + 2].tolist()
                pct = ((feat_acts >= lower) & (feat_acts <= upper)).float().mean()
                indices = random_range_indices(
                    feat_acts,
                    k=self.seq_cfg.quantile_group_size,
                    bounds=(lower, upper),
                    buffer=buffer,
                )
                indices_dict[
                    f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
                ] = indices.cpu()

        # Concat all the indices together (in the next steps we do all groups at once). Shape of this object is [n_bold 2],
        # i.e. the [i, :]-th element are the batch and sequence dimensions for the i-th bold token.
        indices_bold = torch.concat(list(indices_dict.values())).cpu()
        n_bold = indices_bold.shape[0]

        return indices_dict, indices_bold, n_bold

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

        if indices_buf.shape != (n_bold, padded_buffer_width, 2):
            raise ValueError(
                f"Expected indices_buf shape {(n_bold, padded_buffer_width, 2)}, got {indices_buf.shape}"
            )

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
        sequence_groups_data = []
        group_sizes_cumsum = np.cumsum(
            [0] + [len(indices) for indices in indices_dict.values()]
        ).tolist()

        feat_logits = feat_logits.cpu()
        feat_acts_coloring = feat_acts_coloring.cpu()
        token_ids = token_ids.cpu()
        indices_bold = indices_bold.cpu()

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

        else:
            for group_idx, group_name in enumerate(indices_dict.keys()):
                seq_data = [
                    SequenceData(
                        original_index=int(indices_bold[i, 0].item()),
                        token_ids=token_ids[i].tolist(),
                        feat_acts=[round(f, 4) for f in feat_acts_coloring[i].tolist()],
                        token_logits=feat_logits[token_ids[i]].tolist(),
                        qualifying_token_index=int(indices_bold[i, 1].item()),
                    )
                    for i in range(
                        group_sizes_cumsum[group_idx], group_sizes_cumsum[group_idx + 1]
                    )
                ]
                sequence_groups_data.append(SequenceGroupData(group_name, seq_data))

        return SequenceMultiGroupData(sequence_groups_data)
