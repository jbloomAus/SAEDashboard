"""Rename the model name in an existing Neuronpedia export.

Use this when you ran ``neuronpedia-runner --output-neuronpedia-exports``
without ``--neuronpedia-model-name`` and want to change the model name on the
already-converted files instead of re-generating from scratch.

The script rewrites every place the old model name appears:
    {model_dir}/
        {source_id}/
            model.jsonl       -> Model.id
            source.jsonl      -> Source.modelId
            sourceset.jsonl   -> SourceSet.modelId
            features/*.jsonl.gz       -> Feature.modelId on every row
            activations/*.jsonl.gz    -> Activation.modelId on every row
            explanations/*.jsonl.gz   -> Explanation.modelId on every row
                                         (only processed if the directory
                                         exists)

It then renames the parent directory itself
(``.../<old_model_name>`` -> ``.../<new_model_name>``).

Outputs are written to a sibling temp directory and atomically swapped at the
end so a crash mid-run won't corrupt the source files. ``release.jsonl`` is
left untouched (it has no model ID).

Example:
    poetry run python scripts/rename_neuronpedia_export_model.py \\
        --model-dir outputs/neuronpedia_exports/gemma-4-e2b-internal \\
        --new-model-name gemma-4-e2b
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tempfile
from typing import Any, List

import orjson

# Match neuronpedia_export.py for binary parity with the runner's gzip output.
_GZIP_COMPRESSLEVEL = 5

# JSON files that live directly under each source dir and contain a single
# JSON line. Map of filename -> field whose value should be set to the new
# model name.
_SINGLE_LINE_JSONL_FIELDS = {
    "model.jsonl": "id",
    "source.jsonl": "modelId",
    "sourceset.jsonl": "modelId",
}

# Per-batch gzipped jsonl directories. The field on every row whose value
# should be set to the new model name. Each subdir is optional — missing
# directories are silently skipped (e.g. ``explanations/`` only exists once
# auto-interp has been run).
_BATCH_JSONL_DIRS = {
    "features": "modelId",
    "activations": "modelId",
    "explanations": "modelId",
}


def _rewrite_single_line_jsonl(path: str, field: str, new_value: str) -> int:
    """Rewrite a single-line jsonl file in place, updating ``field``.

    Returns the number of rows updated (0 if the file is missing or empty).
    """
    if not os.path.isfile(path):
        return 0

    with open(path, "rb") as f:
        raw = f.read()
    if not raw.strip():
        return 0

    rows: List[bytes] = []
    n_updated = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj: dict[str, Any] = orjson.loads(line)
        if field in obj and obj[field] != new_value:
            obj[field] = new_value
            n_updated += 1
        elif field not in obj:
            print(
                f"  WARNING: field {field!r} missing from row in {path}; "
                "leaving unchanged."
            )
        rows.append(orjson.dumps(obj))

    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        for row in rows:
            f.write(row)
            f.write(b"\n")
    os.replace(tmp_path, path)
    return n_updated


def _rewrite_gzip_jsonl(path: str, field: str, new_value: str) -> int:
    """Stream-rewrite a gzipped jsonl file in place, updating ``field`` on
    every row. Returns the number of rows touched.

    Decompresses the input and writes a fresh ``.gz`` next to it before
    atomically replacing the original — never holds the full uncompressed
    payload in memory beyond a single line at a time.
    """
    tmp_path = path + ".tmp.gz"
    n_updated = 0
    n_total = 0
    try:
        with gzip.open(path, "rb") as f_in, gzip.open(
            tmp_path, "wb", compresslevel=_GZIP_COMPRESSLEVEL
        ) as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                n_total += 1
                obj: dict[str, Any] = orjson.loads(line)
                if field in obj:
                    if obj[field] != new_value:
                        n_updated += 1
                    obj[field] = new_value
                else:
                    print(
                        f"  WARNING: field {field!r} missing from a row in "
                        f"{path}; leaving unchanged."
                    )
                f_out.write(orjson.dumps(obj))  # type: ignore[arg-type]
                f_out.write(b"\n")  # type: ignore[arg-type]
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"  {os.path.basename(path)}: {n_updated}/{n_total} rows updated")
    return n_updated


def _process_source_dir(source_dir: str, new_model_name: str) -> None:
    print(f"Processing source dir: {source_dir}")

    for filename, field in _SINGLE_LINE_JSONL_FIELDS.items():
        path = os.path.join(source_dir, filename)
        n = _rewrite_single_line_jsonl(path, field, new_model_name)
        if os.path.isfile(path):
            print(f"  {filename}: {n} row(s) updated")

    for subdir, field in _BATCH_JSONL_DIRS.items():
        full_subdir = os.path.join(source_dir, subdir)
        if not os.path.isdir(full_subdir):
            continue
        gz_files = sorted(
            f for f in os.listdir(full_subdir) if f.endswith(".jsonl.gz")
        )
        if not gz_files:
            continue
        print(f"  Rewriting {len(gz_files)} {subdir} file(s)...")
        for fn in gz_files:
            _rewrite_gzip_jsonl(
                os.path.join(full_subdir, fn), field, new_model_name
            )


def _atomic_dir_swap(src_dir: str, dst_dir: str) -> None:
    """Atomically move ``src_dir`` -> ``dst_dir``.

    On the same filesystem this is a single ``rename(2)``. If something
    already exists at ``dst_dir`` we refuse — the caller should have caught
    that earlier with a clearer message.
    """
    if os.path.exists(dst_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing directory: {dst_dir}"
        )
    os.rename(src_dir, dst_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rename the model in an existing Neuronpedia export "
            "(written by neuronpedia-runner --output-neuronpedia-exports)."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Path to the existing model export directory, i.e. "
            "'<exports_dir>/<old_model_name>'. All source-set "
            "subdirectories underneath will be rewritten."
        ),
    )
    parser.add_argument(
        "--new-model-name",
        required=True,
        help=(
            "New model name to write into model.jsonl / source.jsonl / "
            "sourceset.jsonl / features/*.jsonl.gz / activations/*.jsonl.gz, "
            "and to use when renaming the parent directory."
        ),
    )
    parser.add_argument(
        "--no-rename-dir",
        action="store_true",
        help=(
            "Skip renaming the parent directory after rewriting files. "
            "Use this if you only want to fix the model ID inside the JSON "
            "and will rename the directory yourself."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir.rstrip(os.sep))
    if not os.path.isdir(model_dir):
        print(f"ERROR: model dir does not exist: {model_dir}", file=sys.stderr)
        return 2

    old_model_name = os.path.basename(model_dir)
    new_model_name = args.new_model_name

    if old_model_name == new_model_name:
        print(
            f"Old and new model name are both {old_model_name!r}; nothing to do."
        )
        return 0

    parent_dir = os.path.dirname(model_dir)
    new_model_dir = os.path.join(parent_dir, new_model_name)
    if (not args.no_rename_dir) and os.path.exists(new_model_dir):
        print(
            f"ERROR: target directory already exists: {new_model_dir}\n"
            "Refusing to overwrite. Move it out of the way and re-run.",
            file=sys.stderr,
        )
        return 2

    source_dirs = sorted(
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, name))
    )
    if not source_dirs:
        print(f"ERROR: no source subdirectories found under {model_dir}", file=sys.stderr)
        return 2

    print(f"Old model name (from --model-dir basename): {old_model_name}")
    print(f"New model name:                              {new_model_name}")
    print(f"Source dirs to rewrite ({len(source_dirs)}):")
    for sd in source_dirs:
        print(f"  - {os.path.basename(sd)}")
    if not args.no_rename_dir:
        print(f"After rewrite the parent dir will be renamed to: {new_model_dir}")
    if not args.yes:
        resp = input("\nProceed? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1

    # Stage rewrites in a sibling temp directory so a crash mid-run doesn't
    # leave the source dir half-modified. We copy first, mutate the copy,
    # then atomically swap the trees at the end.
    staging_parent = tempfile.mkdtemp(
        prefix=f"np_rename_{new_model_name}_", dir=parent_dir
    )
    staging_model_dir = os.path.join(staging_parent, new_model_name)

    print(f"\nStaging copy at: {staging_model_dir}")
    try:
        # copytree preserves gzip files byte-for-byte; we'll mutate them
        # in-place under the staging copy.
        shutil.copytree(model_dir, staging_model_dir)
        for sd in sorted(
            os.path.join(staging_model_dir, name)
            for name in os.listdir(staging_model_dir)
            if os.path.isdir(os.path.join(staging_model_dir, name))
        ):
            _process_source_dir(sd, new_model_name)

        if args.no_rename_dir:
            # Replace the original dir contents only, keeping the old name.
            backup = model_dir + ".old"
            if os.path.exists(backup):
                shutil.rmtree(backup)
            os.rename(model_dir, backup)
            os.rename(staging_model_dir, model_dir)
            shutil.rmtree(backup)
            shutil.rmtree(staging_parent)
            print(
                f"\nDone. Files in {model_dir} now reference "
                f"model name {new_model_name!r}. Directory left in place "
                "(--no-rename-dir)."
            )
        else:
            backup = model_dir + ".old"
            if os.path.exists(backup):
                shutil.rmtree(backup)
            os.rename(model_dir, backup)
            try:
                _atomic_dir_swap(staging_model_dir, new_model_dir)
            except BaseException:
                # Roll back: put the original back where it was.
                os.rename(backup, model_dir)
                raise
            shutil.rmtree(backup)
            shutil.rmtree(staging_parent)
            print(
                f"\nDone. Renamed {model_dir} -> {new_model_dir} with all "
                f"model IDs rewritten to {new_model_name!r}."
            )
    except BaseException:
        # Clean up the staging tree on any failure; the original is
        # untouched (other than the .old swap above which gets undone in
        # that branch).
        if os.path.exists(staging_parent):
            shutil.rmtree(staging_parent, ignore_errors=True)
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
