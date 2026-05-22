"""Mortal (MJAI subprocess) opponent for local RiichiEnv training."""

from __future__ import annotations

import json
import select
import subprocess
from pathlib import Path
from typing import Any

from riichienv import Action, Observation


def build_mortal_argv(
    *,
    player_id: int,
    mortal_binary: str,
    model_dir: Path | None,
    docker_image: str,
) -> list[str]:
    """Build argv to spawn Mortal for *player_id* (0--3 POV)."""
    if mortal_binary.strip().lower() == "docker":
        if model_dir is None:
            raise ValueError("mortal_model_dir is required when mortal_binary is 'docker'")
        return [
            "docker",
            "run",
            "-i",
            "--rm",
            "-v",
            f"{model_dir.resolve()}:/mnt",
            docker_image,
            str(player_id),
        ]
    return [str(Path(mortal_binary).expanduser().resolve()), str(player_id)]


class MortalAgent:
    """Drive a Mortal process over stdin/stdout MJAI (one line per JSON event).

    Requires ``Observation.new_events()`` from riichienv (incremental MJAI strings
    for this player's POV). See Mortal docs: https://mortal.ekyu.moe/
    """

    def __init__(
        self,
        player_id: int,
        argv: list[str],
        *,
        timeout: float = 30.0,
    ) -> None:
        self.player_id = player_id
        self._argv = argv
        self._timeout = timeout
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
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
                proc.stdin.write('{"type":"end_game"}\n')
                proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass
            try:
                proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def __enter__(self) -> MortalAgent:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _require_proc(self) -> subprocess.Popen[str]:
        if self._proc is None:
            raise RuntimeError("MortalAgent.start() must be called before act/observe")
        return self._proc

    def _new_events_payload(self, observation: Observation) -> list[str]:
        new_events = getattr(observation, "new_events", None)
        if not callable(new_events):
            raise RuntimeError(
                "Mortal opponent requires riichienv.Observation.new_events(). "
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

    def _write_line(self, proc: subprocess.Popen[str], line: str) -> None:
        if not line.endswith("\n"):
            line += "\n"
        proc.stdin.write(line)
        proc.stdin.flush()

    def _read_available_json_lines(self, proc: subprocess.Popen[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
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
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

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

    def _push_events(self, observation: Observation) -> list[dict[str, Any]]:
        proc = self._require_proc()
        outputs: list[dict[str, Any]] = []
        for line in self._new_events_payload(observation):
            self._write_line(proc, line)
            outputs.extend(self._read_available_json_lines(proc))
        if not outputs:
            extra = self._blocking_read_one(proc)
            if extra is not None:
                outputs.append(extra)
        return outputs

    def observe(self, observation: Observation) -> None:
        """Feed MJAI deltas for this seat without using Mortal's last reply as an action."""
        self._push_events(observation)

    def act(self, observation: Observation) -> Action:
        """Feed MJAI deltas and map Mortal's last JSON reply to a riichienv ``Action``."""
        outputs = self._push_events(observation)
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
