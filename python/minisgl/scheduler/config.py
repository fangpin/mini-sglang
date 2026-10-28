from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    delay_prefill_wait_sec: float = 0.0
    delay_prefill_min_batch_size: int = 0
    delay_prefill_max_passes: int = 0
    delay_prefill_debug: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def __post_init__(self) -> None:
        if not isfinite(self.delay_prefill_wait_sec) or self.delay_prefill_wait_sec < 0:
            raise ValueError("delay_prefill_wait_sec must be finite and non-negative")
        if self.delay_prefill_min_batch_size < 0:
            raise ValueError("delay_prefill_min_batch_size must be non-negative")
        if self.delay_prefill_max_passes < 0:
            raise ValueError("delay_prefill_max_passes must be non-negative")

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
