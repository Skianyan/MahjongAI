"""Akochan (MJAI subprocess) opponent for local RiichiEnv training.

Akochan expects each stdin line to be a JSON object with the MJAI event plus a
``can_act`` boolean (only the last event in a batch should have ``can_act: true``).
See upstream ``critter-mj/akochan`` and ``system.exe pipe <tactics> <seat>``.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
from pathlib import Path
from typing import Any

from riichienv import Action, Observation


def _prepend_ld_library_path(akochan_dir: str) -> str:
    """Ensure ``libai.so`` in *akochan_dir* is found when spawning ``system.exe``."""
    existing = os.environ.get("LD_LIBRARY_PATH", "").strip()
    if not existing:
        return akochan_dir
    parts = existing.split(os.pathsep)
    if akochan_dir in parts:
        return existing
    return f"{akochan_dir}{os.pathsep}{existing}"


def build_akochan_argv(
    *,
    player_id: int,
    akochan_dir: str | Path,
    tactics_path: str | Path | None = None,
) -> tuple[list[str], dict[str, str] | None]:
    """Build ``argv`` and optional extra ``env`` for spawning Akochan for *player_id* (0--3).

    If *tactics_path* is ``None`` and ``<akochan_dir>/akochan_pipe.sh`` exists and is
    executable, that script is used (typically sets ``LD_LIBRARY_PATH`` and runs
    ``system.exe pipe tactics.json <seat>``).

    Otherwise ``system.exe pipe <tactics> <seat>`` is used with ``LD_LIBRARY_PATH``
    prepended so ``libai.so`` in *akochan_dir* resolves. A non-``None`` *tactics_path*
    always selects this path so a custom tactics file is respected.
    """
    if not (0 <= player_id <= 3):
        raise ValueError(f"player_id must be 0..3, got {player_id}")
    root = Path(akochan_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"akochan_dir is not a directory: {root}")

    wrapper = root / "akochan_pipe.sh"
    exe = root / "system.exe"
    use_wrapper = (
        tactics_path is None
        and wrapper.is_file()
        and os.access(wrapper, os.X_OK)
    )
    if use_wrapper:
        return [str(wrapper), str(player_id)], None

    if not exe.is_file():
        raise FileNotFoundError(
            f"Neither {wrapper} (executable, for default tactics) nor {exe} found under akochan_dir={root}"
        )
    tactics = (
        Path(tactics_path).expanduser().resolve()
        if tactics_path is not None
        else (root / "tactics.json")
    )
    if not tactics.is_file():
        raise FileNotFoundError(
            f"tactics file not found: {tactics}. Pass tactics_path= or place tactics.json in {root}"
        )
    argv = [str(exe), "pipe", str(tactics), str(player_id)]
    extra_env = {"LD_LIBRARY_PATH": _prepend_ld_library_path(str(root))}
    return argv, extra_env


class AkochanAgent:
    """Drive an Akochan ``system.exe pipe`` process over stdin/stdout (one MJAI JSON per line).

    Requires ``Observation.new_events()`` from riichienv (same incremental contract as
    :class:`MortalAgent`).
    """

    def __init__(
        self,
        player_id: int,
        argv: list[str],
        *,
        timeout: float = 30.0,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.player_id = player_id
        self._argv = argv
        self._timeout = timeout
        self._extra_env = extra_env
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        env = os.environ.copy()
        if self._extra_env:
            env.update(self._extra_env)
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )

    def reset_between_games(self) -> None:
        """Drain residual output from the previous game without restarting the process."""
        proc = self._proc
        if proc is None:
            return
        while True:
            readable, _, _ = select.select([proc.stdout], [], [], 0.0)
            if not readable:
                break
            line = proc.stdout.readline()
            if not line:
                break

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def __enter__(self) -> AkochanAgent:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _require_proc(self) -> subprocess.Popen[str]:
        if self._proc is None:
            raise RuntimeError("AkochanAgent.start() must be called before act/observe")
        return self._proc

    def _new_events_payload(self, observation: Observation) -> list[str]:
        new_events = getattr(observation, "new_events", None)
        if not callable(new_events):
            raise RuntimeError(
                "Akochan opponent requires riichienv.Observation.new_events(). "
                "Upgrade riichienv to a version that exposes incremental MJAI lines."
            )
        raw_events = new_events()
        lines: list[str] = []
        for raw in raw_events:
            if isinstance(raw, dict):
                lines.append(json.dumps(raw, ensure_ascii=False, separators=(",", ":")))
            else:
                s = str(raw).strip()
                if s:
                    lines.append(s)
        return lines

    def _lines_with_can_act(self, lines: list[str], *, expect_decision: bool) -> list[str]:
        """Attach ``can_act`` to each JSON object.

        When *expect_decision* is true (``act``), only the last event is ``can_act: true`` so
        akochan emits one reply. When false (``observe``), every event is ``can_act: false`` so
        we only sync state; otherwise akochan may exit when asked to act on another player's
        turn (broken pipe on the next write).
        """
        events: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        out: list[str] = []
        last = len(events) - 1
        for i, obj in enumerate(events):
            payload = dict(obj)
            payload["can_act"] = bool(expect_decision and i == last)
            out.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return out

    def _write_line(self, proc: subprocess.Popen[str], line: str) -> None:
        if not line.endswith("\n"):
            line += "\n"
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(
                "Akochan subprocess closed stdin (likely crashed). "
                "Check akochan build, tactics.json, and that MJAI events match the pipe protocol."
            ) from exc

    def _read_available_json_lines(self, proc: subprocess.Popen[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        while True:
            readable, _, _ = select.select([proc.stdout], [], [], 0.05)
            if not readable:
                break
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def _blocking_read_one(self, proc: subprocess.Popen[str]) -> dict[str, Any] | None:
        readable, _, _ = select.select([proc.stdout], [], [], self._timeout)
        if not readable:
            return None
        line = proc.stdout.readline()
        if not line or not line.strip():
            return None
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None

    def _push_events(self, observation: Observation, *, expect_decision: bool) -> list[dict[str, Any]]:
        proc = self._require_proc()
        outputs: list[dict[str, Any]] = []
        payload = self._lines_with_can_act(
            self._new_events_payload(observation),
            expect_decision=expect_decision,
        )
        for line in payload:
            self._write_line(proc, line)
            outputs.extend(self._read_available_json_lines(proc))
        if expect_decision and not outputs:
            extra = self._blocking_read_one(proc)
            if extra is not None:
                outputs.append(extra)
        return outputs

    def observe(self, observation: Observation) -> None:
        """Feed MJAI deltas for this seat without treating Akochan's reply as an action."""
        self._push_events(observation, expect_decision=False)

    def act(self, observation: Observation) -> Action:
        """Feed MJAI deltas with ``can_act`` and map the last JSON reply to a riichienv ``Action``."""
        outputs = self._push_events(observation, expect_decision=True)
        from mahjong_ai.agents import FallbackAgent

        fb = FallbackAgent()
        if not outputs:
            return fb.act(observation)
        last = outputs[-1]
        try:
            chosen = observation.select_action_from_mjai(last)
            if chosen is not None:
                return chosen
        except Exception:
            pass
        return fb.act(observation)
