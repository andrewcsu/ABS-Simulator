"""CSV telemetry logger.

Writes one row per control step. Per-wheel data is unrolled with suffix
_FL/_FR/_RL/_RR. Opens the file lazily at first log and closes at close().
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, List, Optional


@dataclass
class TelemetryLogger:
    path: Optional[Path] = None
    _fh: Optional[IO] = None
    _writer: Optional[csv.DictWriter] = None
    _fields: List[str] = field(default_factory=list)
    _rows: List[Dict[str, Any]] = field(default_factory=list)
    in_memory: bool = False

    def log(self, row: Dict[str, Any]) -> None:
        if self.in_memory or self.path is None:
            self._rows.append(row)
            return
        if self._writer is None:
            self._fields = list(row.keys())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)
