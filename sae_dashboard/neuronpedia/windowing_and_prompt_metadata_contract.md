# Windowing And Prompt-Length Metadata Contract

Purpose: canonical upstream reference for the Neuronpedia prompt windowing modes implemented in
`sae_dashboard.neuronpedia.prompt_pretokenization` and consumed by the dashboard-generation flow.

## Ownership And Scope

- SAEDashboard owns the prompt windowing contract, the emitted prompt-length metadata, and the pretokenization CLI
  behavior.
- Interpretune custom modules should only handle task-specific prompt rendering or task-specific metadata. They should
  not redefine windowing semantics.
- Neuronpedia runner flows consume this metadata contract when they stage shared prompt tokens, derive effective
  lengths, and generate prompt-bucket schedules.

## Windowing Modes

| Windowing mode | Family | Built-in generic path | Streaming | Saved width | Emitted prompt metadata | Typical use |
| --- | --- | --- | --- | --- | --- | --- |
| `concatenate` | packed legacy | yes | supported | configured `context_size` | none | dense corpora such as Monology where packing across prompt boundaries is acceptable |
| `filter-truncate` | packed legacy | yes | supported | configured `context_size` | none | prompt datasets that should stay fixed-width without example-aligned padding |
| `max-prompt-pad` | example-aligned pad-enabled | yes | not yet supported | longest observed prompt | `attention_mask`, `prompt_lengths`, `effective_context_size` | one-row-per-prompt caches where every prompt must survive intact |
| `fixed-context-pad` | example-aligned pad-enabled | yes | not yet supported | configured `context_size` | `attention_mask`, `prompt_lengths`, `effective_context_size` | one-row-per-prompt caches that must preserve a fixed width |

## Generic Versus Custom Prompt Rendering

The built-in SAEDashboard path is sufficient when one dataset column already represents the prompt input.

- Use `--column-name` to point at the prompt source column.
- Use `--use-chat-formatting` when that column should be passed through `tokenizer.apply_chat_template(...)`.
- String rows with `--use-chat-formatting` are wrapped as single user messages before chat templating.
- Conversation-style rows can be passed directly through `apply_chat_template(...)` when the column already contains
  structured messages.
- Example-aligned modes do not require a custom dataset module when the dataset already conforms to one of those
  single-column shapes.

Use `--custom-dataset-module` only when prompt rendering needs more than one source column or extra task-specific
logic.

- RTE/BoolQ is the current example: the prompt must be assembled from multiple fields plus prompt-config settings
  before tokenization.
- Custom modules may also attach extra metadata fields, but they should still hand the resulting token sequences to
  `pretokenize_prompt_token_sequences(...)` so the upstream windowing contract stays authoritative.

## Saved Dataset Shape

Packed modes emit only `input_ids` because saved rows no longer correspond 1:1 to source prompts.

- `concatenate`
- `filter-truncate`

Example-aligned pad-enabled modes emit `input_ids` plus `attention_mask` because each saved row still maps to one
source prompt.

- `max-prompt-pad`
- `fixed-context-pad`

For example-aligned modes, prompts are never truncated by the upstream helper.

- `max-prompt-pad` sets the saved width to the longest observed prompt.
- `fixed-context-pad` requires every prompt length to be `<= context_size`; longer prompts raise an error.

## Metadata Fields

The generated `sae_lens.json` metadata carries the windowing contract forward into runner execution.

- `windowing_mode`: canonical selected mode.
- `prompt_windowing_family`: `packed_legacy` for packed modes, `example_aligned_pad_enabled` for padded example-aligned modes.
- `effective_context_size`: saved row width after windowing. This is the maximum observed prompt length for
  `max-prompt-pad`; otherwise it is the configured `context_size`.
- `prompt_lengths_available`: whether per-prompt lengths remain meaningful for the saved dataset.
- `prompt_length_min`, `prompt_length_max`, `prompt_length_mean`: aggregate statistics derived from prompt-aligned
  lengths when available.
- `pad_token_id`: pad token used for example-aligned padded modes.
- `disable_concat_sequences`: whether prompt packing was disabled in the saved artifact.
- `streaming_supported`: `true` for packed modes and `false` for the current example-aligned modes.

## How Dashboard Generation Uses This Contract

The downstream dashboard flow relies on the emitted metadata rather than re-deriving windowing behavior heuristically.

- `shared_tokens_file` stores the staged `tokens_<n_prompts_total>.pt` tensor reused across layer runs.
- The staged effective-length sidecar is derived from prompt-aligned `attention_mask` metadata when available; packed
  modes fall back to fixed-width assumptions.
- `prompt_bucket_schedule_file` is an explicit bucket schedule artifact supplied beside a run.
- `auto_prompt_bucket_schedule` derives the same schedule structure from staged effective lengths when no explicit
  schedule file is supplied.
- `prompt_bucket_ceilings` are optional explicit inclusive ceilings. When omitted, the runner derives them from the
  effective-length distribution via quantiles instead of hardcoded defaults.

## Recommended Usage Patterns

- Dense corpora such as Monology: use `concatenate`, usually with `--column-name text`; add `--use-chat-formatting`
  when the corpus should be templated into one user message per row.
- Single-column prompt datasets that need one row per prompt: use `max-prompt-pad` or `fixed-context-pad` directly
  without a custom dataset module.
- Multi-field or task-specific prompt rendering: use a custom dataset module, then delegate the resulting token
  sequences back to `pretokenize_prompt_token_sequences(...)`.
- Full-prompt RTE/BoolQ caches: continue using the Interpretune custom module because the prompt text is assembled from
  multiple fields before chat templating.

## Streaming Boundary

Streaming currently remains limited to the packed family.

- `concatenate`
- `filter-truncate`

Example-aligned modes are intentionally non-streaming for now because final saved width, prompt-length aggregates, and
bucket-manifest derivation all depend on whole-dataset finalization semantics. That follow-up belongs to the separate
streaming PR set rather than this contract-cleanup slice.