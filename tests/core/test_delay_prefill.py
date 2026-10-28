"""CPU scheduling regressions; model execution and CUDA events are replaced by doubles."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from minisgl.core import Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import ForwardOutput
from minisgl.message import AbortBackendMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
from minisgl.scheduler.table import TableManager


def make_config(**kwargs):
    return SchedulerConfig(
        model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.float32, **kwargs
    )


def make_req(uid=0, *, output_len=10):
    return Req(torch.tensor([10]), uid, 0, output_len, uid, SamplingParams(), None)


def add_pending(scheduler, uid=1, *, age=0.0, length=2, output_len=4):
    scheduler.prefill_manager.add_one_req(
        UserMsg(uid, torch.arange(10, 10 + length), SamplingParams(max_tokens=output_len))
    )
    pending = scheduler.prefill_manager.pending_list[-1]
    pending.queued_at = 100.0 - age
    return pending


def make_scheduler(config=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = config or make_config()
    scheduler.decode_manager = DecodeManager(page_size=1)
    scheduler.table_manager = TableManager(8, torch.zeros((8, 128), dtype=torch.int32))
    scheduler.cache_manager = CacheManager(1024, 1, scheduler.table_manager.page_table, "naive")
    scheduler.prefill_manager = PrefillManager(
        scheduler.cache_manager, scheduler.table_manager, scheduler.decode_manager
    )
    scheduler.prefill_budget = 64
    scheduler._prepare_batch = lambda batch: batch
    return scheduler


@pytest.fixture
def cpu_scheduler(monkeypatch):
    # The production allocator copies pinned CPU tensors to CUDA. Only disable pinning here.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr("minisgl.scheduler.scheduler.monotonic", lambda: 100.0)
    scheduler = make_scheduler(
        make_config(delay_prefill_wait_sec=1, delay_prefill_min_batch_size=2)
    )
    scheduler.decode_manager.running_reqs.add(make_req())
    return scheduler


@pytest.mark.parametrize(
    "wait_sec,min_batch,count,age,expected",
    [
        (0, 0, 1, 0, "prefill"),
        (1, 2, 1, 0, "decode"),
        (1, 2, 1, 1, "prefill"),
        (1, 2, 2, 0, "prefill"),
        (1, 0, 1, 0.5, "decode"),
        (1, 0, 1, 1, "prefill"),
        (0, 2, 1, 100, "decode"),
        (0, 2, 2, 0, "prefill"),
        (1, 2, 0, 0, "decode"),
    ],
)
def test_admission(cpu_scheduler, wait_sec, min_batch, count, age, expected):
    scheduler = cpu_scheduler
    scheduler.config = make_config(
        delay_prefill_wait_sec=wait_sec, delay_prefill_min_batch_size=min_batch
    )
    pending = [add_pending(scheduler, uid=i + 1, age=age) for i in range(count)]
    running = set(scheduler.decode_manager.running_reqs)
    batch = scheduler._schedule_next_batch()
    assert batch.phase == expected
    assert scheduler.decode_manager.running_reqs == running
    if expected == "decode":
        assert scheduler.prefill_manager.pending_list == pending
        assert scheduler.table_manager.available_size == 8


def test_idle_admits_immediately(cpu_scheduler):
    cpu_scheduler.decode_manager.running_reqs.clear()
    add_pending(cpu_scheduler)
    assert cpu_scheduler._schedule_next_batch().is_prefill


def test_abort_delayed_request(cpu_scheduler):
    add_pending(cpu_scheduler, uid=1, age=1.25)
    add_pending(cpu_scheduler, uid=2, age=0.5)
    assert cpu_scheduler.prefill_manager.new_request_stats() == (2, 98.75)
    cpu_scheduler._process_one_msg(AbortBackendMsg(uid=1))
    assert cpu_scheduler.prefill_manager.new_request_stats() == (1, 99.5)
    assert cpu_scheduler._schedule_next_batch().is_decode
    assert [req.uid for req in cpu_scheduler.prefill_manager.pending_list] == [2]


def test_chunk_continues_without_admitting_fresh_requests(cpu_scheduler):
    scheduler = cpu_scheduler
    scheduler.prefill_budget = 2
    assert scheduler.prefill_manager.new_request_stats() == (0, None)
    pending = add_pending(scheduler, length=5, age=1)
    first = scheduler._schedule_next_batch()
    assert isinstance(first.reqs[0], ChunkedReq)
    assert scheduler.prefill_manager.new_request_stats() == (0, None)
    first.reqs[0].complete_one()
    fresh = add_pending(scheduler, uid=2)
    assert scheduler.prefill_manager.new_request_stats() == (1, fresh.queued_at)
    second = scheduler._schedule_next_batch()
    assert [req.uid for req in second.reqs] == [pending.uid]
    assert isinstance(second.reqs[0], ChunkedReq)
    second.reqs[0].complete_one()
    third = scheduler._schedule_next_batch()
    assert [req.uid for req in third.reqs] == [pending.uid]
    assert not isinstance(third.reqs[0], ChunkedReq)
    assert scheduler.prefill_manager.pending_list == [fresh]
    assert scheduler.prefill_manager.new_request_stats() == (1, fresh.queued_at)
    assert scheduler._schedule_next_batch().is_decode


@pytest.mark.parametrize("resource", ["table", "kv_cache"])
def test_prefill_resource_shortage_falls_back_to_decode(cpu_scheduler, resource):
    pending = add_pending(cpu_scheduler, age=1)
    if resource == "table":
        while cpu_scheduler.table_manager.available_size:
            cpu_scheduler.table_manager.allocate()
    else:
        cpu_scheduler.cache_manager.free_slots = torch.empty(0, dtype=torch.int32)
    assert cpu_scheduler._schedule_next_batch().is_decode
    assert cpu_scheduler.prefill_manager.pending_list == [pending]


def test_disabled_policy_needs_no_tp_collective(cpu_scheduler):
    cpu_scheduler.config = replace(make_config(), tp_info=DistributedInfo(1, 2))
    cpu_scheduler.tp_cpu_group = Mock()
    add_pending(cpu_scheduler)
    assert cpu_scheduler._schedule_next_batch().is_prefill
    cpu_scheduler.tp_cpu_group.broadcast.assert_not_called()


@pytest.mark.parametrize("chunk_count", [0, 1, 2])
@pytest.mark.parametrize("fresh_count", [1, 10000])
def test_new_request_stats_only_reads_chunked_prefix(cpu_scheduler, chunk_count, fresh_count):
    accesses = []

    class PrefixOnlyList(list):
        def __iter__(self):
            raise AssertionError("must not iterate over the pending queue")

        def __getitem__(self, index):
            assert isinstance(index, int) and 0 <= index <= chunk_count
            accesses.append(index)
            return super().__getitem__(index)

    queue = [SimpleNamespace(chunked_req=object()) for _ in range(chunk_count)]
    queue.append(SimpleNamespace(chunked_req=None, queued_at=0.0))
    queue.extend([object()] * (fresh_count - 1))
    cpu_scheduler.prefill_manager.pending_list = PrefixOnlyList(queue)
    assert cpu_scheduler.prefill_manager.new_request_stats() == (fresh_count, 0.0)
    assert accesses == list(range(chunk_count + 1))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "wait_sec,min_batch,count,chunked,decoding,expected",
    [
        (0, 0, 1, False, True, True),
        (1, 2, 0, False, True, True),
        (1, 2, 1, False, False, True),
        (1, 2, 1, True, True, True),
        (1, 2, 2, False, True, True),
        (0, 2, 2, False, True, True),
        (0, 2, 1, False, True, False),
    ],
)
def test_deterministic_admission_needs_no_clock_or_collective(
    cpu_scheduler, monkeypatch, rank, wait_sec, min_batch, count, chunked, decoding, expected
):
    scheduler = cpu_scheduler
    scheduler.config = replace(
        make_config(delay_prefill_wait_sec=wait_sec, delay_prefill_min_batch_size=min_batch),
        tp_info=DistributedInfo(rank, 2),
    )
    scheduler.tp_cpu_group = Mock()
    for uid in range(count):
        pending = add_pending(scheduler, uid=uid)
        if chunked:
            pending.chunked_req = make_req(uid)
    if not decoding:
        scheduler.decode_manager.running_reqs.clear()
    clock = Mock(side_effect=AssertionError("deterministic admission must not read the clock"))
    monkeypatch.setattr("minisgl.scheduler.scheduler.monotonic", clock)
    assert scheduler._admit_new_prefills() is expected
    clock.assert_not_called()
    scheduler.tp_cpu_group.broadcast.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize(
    "wait_sec,min_batch,age,expected",
    [(0, 2, 0.5, False), (1, 1, 0.5, True), (1, 0, 1.25, True), (1, 0, 0.5, False)],
)
def test_admission_clock_and_debug_logging(
    cpu_scheduler, monkeypatch, rank, debug, wait_sec, min_batch, age, expected
):
    scheduler = cpu_scheduler
    scheduler.config = replace(
        make_config(
            delay_prefill_wait_sec=wait_sec,
            delay_prefill_min_batch_size=min_batch,
            delay_prefill_debug=debug,
        ),
        tp_info=DistributedInfo(rank, 2),
    )
    add_pending(scheduler, age=age)
    clock = Mock(return_value=100.0)
    log = Mock()
    work = Mock()
    scheduler.tp_cpu_group = Mock()

    def broadcast(decision, root):
        assert root == 0
        decision.fill_(int(expected))
        return work

    scheduler.tp_cpu_group.broadcast.side_effect = broadcast
    monkeypatch.setattr("minisgl.scheduler.scheduler.monotonic", clock)
    monkeypatch.setattr("minisgl.scheduler.scheduler.logger.info_rank0", log)
    assert scheduler._admit_new_prefills() is expected
    needs_time_check = wait_sec > 0 and min_batch != 1
    assert clock.call_count == int(rank == 0 and (debug or needs_time_check))
    assert scheduler.tp_cpu_group.broadcast.call_count == int(needs_time_check)
    assert work.wait.call_count == int(needs_time_check)
    if debug and rank == 0:
        log.assert_called_once_with(
            "Delay prefill: admit=%s waiting=%d oldest_wait=%.3fs decoding=%d",
            expected,
            1,
            age,
            1,
        )
    else:
        log.assert_not_called()


def test_rejected_fresh_requests_skip_resource_estimation(cpu_scheduler, monkeypatch):
    pending = add_pending(cpu_scheduler)
    inflight_tokens = Mock(side_effect=AssertionError("must not estimate decode resources"))
    adder = Mock(side_effect=AssertionError("must not construct a prefill adder"))
    monkeypatch.setattr(DecodeManager, "inflight_tokens", property(inflight_tokens))
    monkeypatch.setattr("minisgl.scheduler.prefill.PrefillAdder", adder)
    assert cpu_scheduler._schedule_next_batch().is_decode
    inflight_tokens.assert_not_called()
    adder.assert_not_called()
    assert cpu_scheduler.prefill_manager.pending_list == [pending]
    assert cpu_scheduler.table_manager.available_size == 8


@pytest.mark.parametrize(
    "field,value",
    [
        ("delay_prefill_wait_sec", -1),
        ("delay_prefill_wait_sec", float("nan")),
        ("delay_prefill_wait_sec", float("inf")),
        ("delay_prefill_min_batch_size", -1),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError, match=field):
        make_config(**{field: value})


def test_cli_options():
    from minisgl.server.args import parse_args

    config, _ = parse_args(
        [
            "--model",
            "unused",
            "--dtype",
            "float32",
            "--delay-prefill-wait-sec",
            "0.05",
            "--delay-prefill-min-batch-size",
            "8",
            "--delay-prefill-debug",
        ]
    )
    assert config.delay_prefill_wait_sec == 0.05
    assert config.delay_prefill_min_batch_size == 8
    assert config.delay_prefill_debug


def _tp_worker(rank, path):
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        scheduler = make_scheduler(make_config(delay_prefill_wait_sec=1))
        scheduler.config = replace(scheduler.config, tp_info=DistributedInfo(rank, 2))
        group = dist.distributed_c10d._get_default_group()
        scheduler.tp_cpu_group = Mock(wraps=group)
        scheduler.decode_manager.running_reqs.add(make_req())
        pending = add_pending(scheduler)
        # Rank 1's age is deliberately the opposite of rank 0's on each decision.
        from time import monotonic

        for step, admit in enumerate([False, True]):
            scheduler.config = replace(
                scheduler.config, delay_prefill_wait_sec=1, delay_prefill_min_batch_size=1
            )
            assert scheduler._admit_new_prefills()
            scheduler.config = replace(
                scheduler.config, delay_prefill_wait_sec=0, delay_prefill_min_batch_size=2
            )
            assert not scheduler._admit_new_prefills()
            assert scheduler.tp_cpu_group.broadcast.call_count == step

            scheduler.config = replace(
                scheduler.config, delay_prefill_wait_sec=1, delay_prefill_min_batch_size=0
            )
            local_age = 10 if (admit == (rank == 0)) else -1000
            pending.queued_at = monotonic() - local_age
            assert scheduler._admit_new_prefills() == admit
            assert scheduler.tp_cpu_group.broadcast.call_count == step + 1

            # The next I/O collective must not consume an admission decision.
            marker = torch.tensor(100 + step if rank == 0 else -1)
            group.broadcast(marker, root=0).wait()
            assert marker.item() == 100 + step
    finally:
        dist.destroy_process_group()


def test_tp_ranks_follow_primary_clock(tmp_path):
    mp.spawn(_tp_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


@pytest.mark.parametrize("overlap", [False, True])
def test_decode_resumes_with_all_tokens_and_resources(cpu_scheduler, monkeypatch, overlap):
    scheduler = cpu_scheduler
    scheduler.decode_manager.running_reqs.clear()
    scheduler.device = torch.device("cpu")
    scheduler.token_pool = scheduler.table_manager.token_pool
    scheduler.finished_reqs = set()
    scheduler.eos_token_id = -1
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.stream = Mock()
    scheduler._prepare_batch = Scheduler._prepare_batch.__get__(scheduler)
    replies = []
    phases = []
    scheduler.send_result = replies.extend
    scheduler.receive_msg = lambda blocking: []
    # Exercise the real input/output mapping and page allocator on CPU.
    for name in ["empty", "tensor"]:
        factory = getattr(torch, name)

        def unpinned(*args, _factory=factory, **kwargs):
            kwargs.pop("pin_memory", None)
            return _factory(*args, **kwargs)

        monkeypatch.setattr(torch, name, unpinned)

    def forward(batch, sample_args):
        phases.append((batch.phase, [req.uid for req in batch.reqs]))
        tokens = []
        offset = 0
        for req in batch.reqs:
            offset += req.extend_len
            tokens.append(int(batch.input_ids[offset - 1]) + 1)
            req.complete_one()
        output = torch.tensor(tokens, dtype=torch.int32)
        return ForwardOutput(output, output.clone(), Mock())

    scheduler.engine = SimpleNamespace(
        graph_runner=SimpleNamespace(
            pad_batch=lambda batch: setattr(batch, "padded_reqs", batch.reqs)
        ),
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
        sampler=SimpleNamespace(prepare=lambda batch: None),
        forward_batch=forward,
        stream=Mock(),
        page_table=scheduler.table_manager.page_table,
    )
    last_data = None

    def step():
        nonlocal last_data
        if overlap:
            last_data = scheduler.overlap_loop(last_data)
        else:
            scheduler.normal_loop()

    add_pending(scheduler, uid=0, output_len=8)
    step()  # Idle bypass: start the first request.
    add_pending(scheduler, uid=1, output_len=4)
    step()  # Below threshold: request 0 decodes, request 1 stays queued.
    add_pending(scheduler, uid=2, output_len=4)
    step()  # Batch threshold: temporarily pause decode and run both new prefills.
    step()  # All decode requests resume, including request 0.
    assert phases[:4] == [
        ("prefill", [0]),
        ("decode", [0]),
        ("prefill", [1, 2]),
        ("decode", [0, 1, 2]),
    ]
    for _ in range(12):
        step()
    for uid, length in [(0, 8), (1, 4), (2, 4)]:
        outputs = [reply for reply in replies if reply.uid == uid]
        assert [reply.next_token for reply in outputs] == list(range(12, 12 + length))
        assert outputs[-1].finished
    assert last_data is None
    assert not scheduler.decode_manager.runnable
    assert not scheduler.prefill_manager.runnable
    assert scheduler.table_manager.available_size == 8
    scheduler.cache_manager.check_integrity()
