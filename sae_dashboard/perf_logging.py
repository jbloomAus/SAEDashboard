import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import torch

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix platforms
    resource = None

_TIMED_STAGE_DEPTH: ContextVar[int] = ContextVar("timed_stage_depth", default=0)


def _local_log_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def _format_perf_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def log_perf_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(
        f"{_local_log_timestamp()} [runner_perf] "
        + " ".join(
            f"{key}={_format_perf_value(value)}" for key, value in payload.items()
        ),
        flush=True,
    )


def cpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        load1, load5, load15 = os.getloadavg()
        snapshot.update({"load1": load1, "load5": load5, "load15": load15})
    except OSError:
        pass

    loadavg_path = Path("/proc/loadavg")
    if loadavg_path.exists():
        parts = loadavg_path.read_text(encoding="utf-8", errors="replace").split()
        if len(parts) >= 4 and "/" in parts[3]:
            runnable_threads, total_threads = parts[3].split("/", 1)
            snapshot["runnable_threads"] = int(runnable_threads)
            snapshot["system_threads"] = int(total_threads)

    task_dir = Path("/proc/self/task")
    if task_dir.exists():
        snapshot["process_threads"] = sum(1 for _ in task_dir.iterdir())
    return snapshot


def runtime_snapshot() -> dict[str, Any]:
    snapshot = cpu_snapshot()
    snapshot["torch_num_threads"] = torch.get_num_threads()
    try:
        snapshot["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except RuntimeError:
        pass
    if hasattr(os, "sched_getaffinity"):
        try:
            snapshot["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return snapshot


def process_memory_snapshot() -> dict[str, int]:
    statm_path = Path("/proc/self/statm")
    if not statm_path.exists():
        return {}

    parts = statm_path.read_text(encoding="utf-8", errors="replace").split()
    if len(parts) < 7:
        return {}

    page_size = os.sysconf("SC_PAGE_SIZE")
    names = ("vsz", "rss", "shared", "text", "lib", "data", "dirty")
    snapshot = {"page_size_bytes": int(page_size)}
    for name, raw_value in zip(names, parts):
        if raw_value.isdigit():
            pages = int(raw_value)
            snapshot[f"statm_{name}_pages"] = pages
            snapshot[f"statm_{name}_bytes"] = pages * int(page_size)
    return snapshot


def rusage_snapshot() -> dict[str, float | int]:
    if resource is None:
        return {}

    usage = resource.getrusage(resource.RUSAGE_SELF)
    snapshot = {
        "user_cpu_s": usage.ru_utime,
        "system_cpu_s": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }
    snapshot.update(process_memory_snapshot())
    return snapshot


def rusage_delta(
    start: dict[str, float | int],
    end: dict[str, float | int],
) -> dict[str, float | int]:
    return {
        key: end.get(key, 0) - start.get(key, 0)
        for key in sorted(set(start) | set(end))
    }


def _thread_stat_snapshot(tid_path: Path) -> dict[str, int | str]:
    stat_path = tid_path / "stat"
    raw_stat = stat_path.read_text(encoding="utf-8", errors="replace")
    close_paren = raw_stat.rfind(")")
    if close_paren < 0:
        return {}
    prefix = raw_stat[: close_paren + 1]
    suffix = raw_stat[close_paren + 2 :].split()
    comm_start = prefix.find("(")
    comm = prefix[comm_start + 1 : -1] if comm_start >= 0 else ""
    if len(suffix) < 13:
        return {"name": comm}
    return {
        "name": comm,
        "minor_faults": int(suffix[7]),
        "major_faults": int(suffix[9]),
        "user_ticks": int(suffix[11]),
        "system_ticks": int(suffix[12]),
    }


def _thread_context_switch_snapshot(tid_path: Path) -> dict[str, int]:
    status_path = tid_path / "status"
    if not status_path.exists():
        return {}

    snapshot: dict[str, int] = {}
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("voluntary_ctxt_switches:") or line.startswith(
            "nonvoluntary_ctxt_switches:"
        ):
            key, raw_value = line.split(":", 1)
            raw_value = raw_value.strip()
            if raw_value.isdigit():
                snapshot[key] = int(raw_value)
    return snapshot


def thread_fault_snapshot() -> dict[str, dict[str, int | str]]:
    task_dir = Path("/proc/self/task")
    if not task_dir.exists():
        return {}

    snapshot: dict[str, dict[str, int | str]] = {}
    for tid_path in task_dir.iterdir():
        if not tid_path.name.isdigit():
            continue
        try:
            thread_snapshot = _thread_stat_snapshot(tid_path)
            thread_snapshot.update(_thread_context_switch_snapshot(tid_path))
        except OSError:
            continue
        if thread_snapshot:
            snapshot[tid_path.name] = thread_snapshot
    return snapshot


def thread_fault_delta(
    start: dict[str, dict[str, int | str]],
    end: dict[str, dict[str, int | str]],
) -> dict[str, Any]:
    active_threads: list[dict[str, int | str]] = []
    for tid in sorted(set(start) | set(end), key=int):
        start_values = start.get(tid, {})
        end_values = end.get(tid, {})
        delta: dict[str, int | str] = {
            "tid": int(tid),
            "name": str(end_values.get("name", start_values.get("name", ""))),
        }
        for key in (
            "minor_faults",
            "major_faults",
            "user_ticks",
            "system_ticks",
            "voluntary_ctxt_switches",
            "nonvoluntary_ctxt_switches",
        ):
            start_value = start_values.get(key, 0)
            end_value = end_values.get(key, 0)
            if isinstance(start_value, int) and isinstance(end_value, int):
                delta[key] = end_value - start_value
        if any(
            isinstance(value, int) and value
            for key, value in delta.items()
            if key != "tid"
        ):
            active_threads.append(delta)

    active_threads.sort(key=lambda item: int(item.get("minor_faults", 0)), reverse=True)
    total_minor_faults = sum(
        int(item.get("minor_faults", 0)) for item in active_threads
    )
    total_major_faults = sum(
        int(item.get("major_faults", 0)) for item in active_threads
    )
    return {
        "thread_count_start": len(start),
        "thread_count_end": len(end),
        "active_thread_count": len(active_threads),
        "total_minor_faults": total_minor_faults,
        "total_major_faults": total_major_faults,
        "max_thread_minor_faults": (
            int(active_threads[0].get("minor_faults", 0)) if active_threads else 0
        ),
        "active_threads": active_threads[:64],
        "active_threads_truncated": len(active_threads) > 64,
    }


def tensor_runtime_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    storage_nbytes: int | None
    try:
        storage_nbytes = tensor.untyped_storage().nbytes()
    except RuntimeError:
        storage_nbytes = None
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "is_contiguous": tensor.is_contiguous(),
        "storage_offset": tensor.storage_offset(),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "logical_nbytes": tensor.numel() * tensor.element_size(),
        "storage_nbytes": storage_nbytes,
        "data_ptr": tensor.data_ptr() if tensor.device.type == "cpu" else None,
    }


def _flatten_numeric_mapping(
    prefix: str, value: Any, output: dict[str, int | float]
) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else str(key)
            _flatten_numeric_mapping(nested_prefix, nested_value, output)
    elif isinstance(value, (int, float)):
        output[prefix] = value


def torch_host_allocator_snapshot() -> dict[str, int | float]:
    host_stats_fn = getattr(torch._C, "_cuda_hostMemoryStats", None)
    if host_stats_fn is None:
        return {}
    try:
        host_stats = host_stats_fn()
    except RuntimeError:
        return {}

    snapshot: dict[str, int | float] = {}
    _flatten_numeric_mapping("", host_stats, snapshot)
    return snapshot


def process_io_snapshot() -> dict[str, int]:
    io_path = Path("/proc/self/io")
    if not io_path.exists():
        return {}

    snapshot: dict[str, int] = {}
    for line in io_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        raw_value = raw_value.strip()
        if raw_value.isdigit():
            snapshot[key] = int(raw_value)
    return snapshot


def io_delta(start: dict[str, int], end: dict[str, int]) -> dict[str, int]:
    return {
        key: end.get(key, 0) - start.get(key, 0)
        for key in sorted(set(start) | set(end))
    }


@contextmanager
def temporary_torch_num_threads(num_threads: int | None) -> Generator[None, None, None]:
    if num_threads is None:
        yield
        return
    if num_threads < 1:
        raise ValueError("temporary_torch_num_threads requires num_threads >= 1")

    previous_num_threads = torch.get_num_threads()
    if num_threads == previous_num_threads:
        yield
        return

    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous_num_threads)


def _cuda_memory_fields(device: torch.device | None) -> dict[str, float]:
    """Allocator snapshot in GiB for per-stage peak-memory attribution. max_* are the
    process-wide (unreset) peaks: a stage that raises max_* between its start and end
    snapshots is the stage that set the current peak."""
    gib = 1024**3
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


@contextmanager
def timed_stage(
    enabled: bool,
    stage: str,
    *,
    device: str | torch.device | None = None,
    capture_runtime_metrics: bool = False,
    **fields: Any,
) -> Generator[None, None, None]:
    if not enabled:
        yield
        return

    stage_depth = _TIMED_STAGE_DEPTH.get()
    token = _TIMED_STAGE_DEPTH.set(stage_depth + 1)
    torch_device = torch.device(device) if device is not None else None
    use_cuda_events = (
        stage_depth == 0
        and torch_device is not None
        and torch_device.type == "cuda"
        and torch.cuda.is_available()
    )
    start_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None

    capture_cuda_memory = (
        torch_device is not None
        and torch_device.type == "cuda"
        and torch.cuda.is_available()
    )
    start_cuda_memory = (
        _cuda_memory_fields(torch_device) if capture_cuda_memory else None
    )

    if use_cuda_events and start_event is not None:
        torch.cuda.synchronize(torch_device)
        start_event.record(torch.cuda.current_stream(torch_device))
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(stage)
    start_runtime = runtime_snapshot() if capture_runtime_metrics else None
    start_io = process_io_snapshot() if capture_runtime_metrics else None
    start_rusage = rusage_snapshot() if capture_runtime_metrics else None
    start_thread_faults = thread_fault_snapshot() if capture_runtime_metrics else None
    start_torch_host_allocator = (
        torch_host_allocator_snapshot() if capture_runtime_metrics else None
    )
    start_process_time = time.process_time() if capture_runtime_metrics else None
    start_time = time.perf_counter()
    try:
        with torch.profiler.record_function(stage):
            yield
    finally:
        try:
            wall_seconds = time.perf_counter() - start_time
            cuda_ms: float | None = None
            if use_cuda_events and start_event is not None and end_event is not None:
                end_event.record(torch.cuda.current_stream(torch_device))
                torch.cuda.synchronize(torch_device)
                cuda_ms = float(start_event.elapsed_time(end_event))
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
            log_fields = dict(fields)
            log_fields.update({"stage": stage, "wall_s": wall_seconds})
            if cuda_ms is not None:
                log_fields["cuda_ms"] = cuda_ms
            if capture_cuda_memory and start_cuda_memory is not None:
                log_fields["cuda_memory_start"] = start_cuda_memory
                log_fields["cuda_memory_end"] = _cuda_memory_fields(torch_device)
            if capture_runtime_metrics:
                end_runtime = runtime_snapshot()
                end_io = process_io_snapshot()
                end_rusage = rusage_snapshot()
                end_thread_faults = thread_fault_snapshot()
                end_torch_host_allocator = torch_host_allocator_snapshot()
                if start_process_time is not None:
                    log_fields["process_time_s"] = (
                        time.process_time() - start_process_time
                    )
                if start_runtime is not None:
                    log_fields["runtime_start"] = start_runtime
                log_fields["runtime_end"] = end_runtime
                if start_rusage is not None:
                    runtime_rusage_delta = rusage_delta(start_rusage, end_rusage)
                    if runtime_rusage_delta:
                        log_fields["rusage_delta"] = runtime_rusage_delta
                if start_thread_faults is not None:
                    runtime_thread_fault_delta = thread_fault_delta(
                        start_thread_faults, end_thread_faults
                    )
                    if runtime_thread_fault_delta:
                        log_fields["thread_fault_delta"] = runtime_thread_fault_delta
                if start_torch_host_allocator is not None:
                    runtime_allocator_delta = rusage_delta(
                        start_torch_host_allocator, end_torch_host_allocator
                    )
                    if runtime_allocator_delta:
                        log_fields["torch_host_allocator_delta"] = (
                            runtime_allocator_delta
                        )
                if start_io is not None:
                    runtime_io_delta = io_delta(start_io, end_io)
                    if runtime_io_delta:
                        log_fields["process_io_delta"] = runtime_io_delta
            log_perf_event("stage_timing", **log_fields)
        finally:
            _TIMED_STAGE_DEPTH.reset(token)


@contextmanager
def elapsed_timer() -> Generator[dict[str, float], None, None]:
    timing: dict[str, float] = {}
    start_time = time.perf_counter()
    try:
        yield timing
    finally:
        timing["wall_s"] = time.perf_counter() - start_time
