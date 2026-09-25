# pyright: basic, reportPrivateImportUsage=false
import re
from contextlib import nullcontext

import sae_dashboard.perf_logging as perf_logging


def test_log_perf_event_prefixes_local_timestamp(capsys) -> None:
    perf_logging.log_perf_event("batch_total", batch=3, wall_s=1.25)

    captured = capsys.readouterr().out.strip()

    assert re.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[runner_perf\] event=batch_total batch=3 wall_s=1\.250000$",
        captured,
    )


def test_timed_stage_skips_nested_cuda_synchronization(
    monkeypatch,
) -> None:
    perf_events: list[dict[str, object]] = []
    sync_calls: list[object | None] = []
    event_instances: list[object] = []

    class _FakeEvent:
        def record(self, stream=None) -> None:
            del stream

        def elapsed_time(self, other) -> float:
            del other
            return 1.25

    monkeypatch.setattr(
        perf_logging,
        "log_perf_event",
        lambda event, /, **fields: perf_events.append({"event": event, **fields}),
    )
    monkeypatch.setattr(perf_logging.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        perf_logging.torch.cuda,
        "synchronize",
        lambda device=None: sync_calls.append(device),
    )
    monkeypatch.setattr(
        perf_logging.torch.cuda, "current_stream", lambda device: device
    )
    monkeypatch.setattr(
        perf_logging.torch.cuda,
        "Event",
        lambda enable_timing=True: event_instances.append(_FakeEvent())
        or event_instances[-1],
    )
    monkeypatch.setattr(
        perf_logging.torch.cuda.nvtx,
        "range_push",
        lambda stage: None,
    )
    monkeypatch.setattr(
        perf_logging.torch.cuda.nvtx,
        "range_pop",
        lambda: None,
    )
    monkeypatch.setattr(
        perf_logging.torch.profiler,
        "record_function",
        lambda stage: nullcontext(),
    )

    with perf_logging.timed_stage(True, "outer", device="cuda"):
        with perf_logging.timed_stage(True, "inner", device="cuda"):
            pass

    assert len(sync_calls) == 2
    assert len(event_instances) == 2

    outer_event = next(
        event
        for event in perf_events
        if event["event"] == "stage_timing" and event["stage"] == "outer"
    )
    inner_event = next(
        event
        for event in perf_events
        if event["event"] == "stage_timing" and event["stage"] == "inner"
    )

    assert outer_event["cuda_ms"] == 1.25
    assert "cuda_ms" not in inner_event


def test_timed_stage_emits_runtime_metrics_when_enabled(
    monkeypatch,
) -> None:
    perf_events: list[dict[str, object]] = []
    runtime_snapshots = [
        {"torch_num_threads": 4, "cpu_affinity": [0, 1, 2, 3]},
        {"torch_num_threads": 4, "process_threads": 8, "cpu_affinity": [0, 1, 2, 3]},
    ]
    io_snapshots = [
        {"read_bytes": 10, "write_bytes": 20},
        {"read_bytes": 15, "write_bytes": 32},
    ]
    rusage_snapshots = [
        {"user_cpu_s": 1.0, "system_cpu_s": 0.5, "minor_faults": 10},
        {
            "user_cpu_s": 1.25,
            "system_cpu_s": 0.75,
            "minor_faults": 16,
            "voluntary_context_switches": 4,
        },
    ]
    process_times = iter([2.0, 2.75])

    monkeypatch.setattr(
        perf_logging,
        "log_perf_event",
        lambda event, /, **fields: perf_events.append({"event": event, **fields}),
    )
    monkeypatch.setattr(
        perf_logging, "runtime_snapshot", lambda: runtime_snapshots.pop(0)
    )
    monkeypatch.setattr(
        perf_logging, "process_io_snapshot", lambda: io_snapshots.pop(0)
    )
    monkeypatch.setattr(
        perf_logging, "rusage_snapshot", lambda: rusage_snapshots.pop(0)
    )
    monkeypatch.setattr(perf_logging.time, "process_time", lambda: next(process_times))
    monkeypatch.setattr(perf_logging.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        perf_logging.torch.profiler,
        "record_function",
        lambda stage: nullcontext(),
    )

    with perf_logging.timed_stage(True, "rolling", capture_runtime_metrics=True):
        pass

    event = perf_events[0]
    assert event["event"] == "stage_timing"
    assert event["stage"] == "rolling"
    assert event["process_time_s"] == 0.75
    assert event["runtime_start"] == {
        "torch_num_threads": 4,
        "cpu_affinity": [0, 1, 2, 3],
    }
    assert event["runtime_end"] == {
        "torch_num_threads": 4,
        "process_threads": 8,
        "cpu_affinity": [0, 1, 2, 3],
    }
    assert event["rusage_delta"] == {
        "minor_faults": 6,
        "system_cpu_s": 0.25,
        "user_cpu_s": 0.25,
        "voluntary_context_switches": 4,
    }
    assert event["process_io_delta"] == {"read_bytes": 5, "write_bytes": 12}


def test_timed_stage_preserves_zero_rusage_fields(
    monkeypatch,
) -> None:
    perf_events: list[dict[str, object]] = []
    runtime_snapshots = [
        {"torch_num_threads": 8},
        {"torch_num_threads": 8},
    ]
    io_snapshots = [
        {"read_bytes": 10, "write_bytes": 20},
        {"read_bytes": 10, "write_bytes": 20},
    ]
    rusage_snapshots = [
        {"user_cpu_s": 1.0, "system_cpu_s": 0.5, "minor_faults": 10},
        {"user_cpu_s": 1.25, "system_cpu_s": 0.5, "minor_faults": 10},
    ]
    process_times = iter([4.0, 4.25])

    monkeypatch.setattr(
        perf_logging,
        "log_perf_event",
        lambda event, /, **fields: perf_events.append({"event": event, **fields}),
    )
    monkeypatch.setattr(
        perf_logging, "runtime_snapshot", lambda: runtime_snapshots.pop(0)
    )
    monkeypatch.setattr(
        perf_logging, "process_io_snapshot", lambda: io_snapshots.pop(0)
    )
    monkeypatch.setattr(
        perf_logging, "rusage_snapshot", lambda: rusage_snapshots.pop(0)
    )
    monkeypatch.setattr(perf_logging.time, "process_time", lambda: next(process_times))
    monkeypatch.setattr(perf_logging.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        perf_logging.torch.profiler,
        "record_function",
        lambda stage: nullcontext(),
    )

    with perf_logging.timed_stage(True, "rolling", capture_runtime_metrics=True):
        pass

    event = perf_events[0]
    assert event["rusage_delta"] == {
        "minor_faults": 0,
        "system_cpu_s": 0.0,
        "user_cpu_s": 0.25,
    }


def test_temporary_torch_num_threads_restores_previous_value(
    monkeypatch,
) -> None:
    current_num_threads = {"value": 8}
    set_calls: list[int] = []

    monkeypatch.setattr(
        perf_logging.torch, "get_num_threads", lambda: current_num_threads["value"]
    )

    def _set_num_threads(value: int) -> None:
        set_calls.append(value)
        current_num_threads["value"] = value

    monkeypatch.setattr(perf_logging.torch, "set_num_threads", _set_num_threads)

    with perf_logging.temporary_torch_num_threads(4):
        assert current_num_threads["value"] == 4

    assert set_calls == [4, 8]
