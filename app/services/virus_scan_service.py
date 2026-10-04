"""Thin client for a ClamAV daemon (PRD Section 2 Stage 1: "virus scan before the file reaches any
parsing worker"). Not named in PRD Section 7's module list, but it is a genuine external-system
call (a clamd socket), so it gets the same services/ treatment as every other external dependency
(code-quality rule 4) rather than being inlined into the upload route.

Gated by `settings.virus_scan_enabled` (default off) so local/dev environments without a clamd
daemon running don't hard-fail every upload — this is a deployment dependency, not something the
app can bootstrap itself.
"""

from __future__ import annotations

import clamd

from app.config import settings


class InfectedFileError(Exception):
    def __init__(self, signature: str) -> None:
        self.signature = signature
        super().__init__(f"Upload rejected: virus scan matched signature '{signature}'")


class VirusScanService:
    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self._host = host or settings.clamd_host
        self._port = port or settings.clamd_port

    def scan_file(self, file_path: str) -> None:
        """Raises InfectedFileError if the file at `file_path` matches a known signature. No-op if
        virus scanning is disabled (see module docstring).

        Takes a path and streams it to clamd chunk-by-chunk (clamd's `instream` reads from a
        file-like object in bounded chunks) rather than a `bytes` argument — Section 8 says never
        load a full PDF into memory, and accepting `bytes` here would force every caller to have
        already done exactly that.
        """
        if not settings.virus_scan_enabled:
            return
        client = clamd.ClamdNetworkSocket(host=self._host, port=self._port)
        with open(file_path, "rb") as f:
            result = client.instream(f)
        status, signature = result.get("stream", ("OK", None))
        if status == "FOUND":
            raise InfectedFileError(signature or "unknown")


virus_scan_service = VirusScanService()
