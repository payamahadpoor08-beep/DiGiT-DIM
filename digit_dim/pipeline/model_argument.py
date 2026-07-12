"""ModelArgument: the context/payload carrier of the pipeline.

Position in pipeline: the unit of data that flows through every Block,
routed by a Transflow. A Block reads and returns a ModelArgument; a
Transflow never inspects payload contents, only metadata/trace/errors.

Design choice: ModelArgument is a plain, mutable dataclass rather than a
frozen one. Blocks are expected to treat the instance they receive as
owned by them for the duration of process() and hand off a single
instance to the next stage (single-writer discipline) -- this avoids the
copy overhead a frozen dataclass would force on every hop. The one place
copies are unavoidable is fan-out, where multiple Blocks run concurrently
against the same logical argument; clone() exists for exactly that case.
Thread-safety: a ModelArgument instance itself performs no locking. It is
safe to share read-only *across* threads only after clone()-ing per
branch; concurrent in-place mutation of one instance from multiple
threads is not supported and not needed given the single-writer rule.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArgument:
    """Carries payload, metadata, execution trace, and errors through a pipeline run."""

    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    trace: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)

    def with_payload(self, payload: Any) -> "ModelArgument":
        """Return self with payload replaced, after recording no trace entry.

        Mutates in place and returns self so Blocks can chain calls without
        an extra copy; callers that need isolation should clone() first.
        """
        self.payload = payload
        return self

    def record(self, stage_name: str) -> "ModelArgument":
        """Append a stage name to the execution trace."""
        self.trace.append(stage_name)
        return self

    def fail(self, stage_name: str, error: BaseException) -> "ModelArgument":
        """Record a non-fatal error against this argument (graceful-degradation path)."""
        self.errors.append(f"{stage_name}: {error!r}")
        return self

    def clone(self) -> "ModelArgument":
        """Deep-copy payload/metadata/context, but start a fresh trace/errors list.

        Used for fan-out: each branch gets an independent ModelArgument so
        concurrent Blocks cannot observe each other's mutations. Trace and
        errors are reset per branch and expected to be merged back by the
        Transflow's fan-in step rather than copied forward blindly.
        """
        clone = ModelArgument(
            payload=copy.deepcopy(self.payload),
            metadata=copy.deepcopy(self.metadata),
            context=copy.deepcopy(self.context),
        )
        return clone

    def elapsed(self) -> float:
        """Seconds since this ModelArgument was created."""
        return time.monotonic() - self.created_at

    def has_errors(self) -> bool:
        return bool(self.errors)
