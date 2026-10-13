from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


BLACK = 0
BRIDGE_COLORS = (1, 2, 3, 4)
PIPE = 5
TOKEN_COLORS = (6, 7, 8, 9)
GAP = 4  # interior columns differ by four cells in the generated layout

_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


@dataclass
class TokenState:
    color: int
    row: int
    pipe: int
    mode: str = "falling"
    delay: int = 0
    target_pipe: int = -1


def _grid_sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _token_mask(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, TOKEN_COLORS)


def _pipe_columns(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.any((col == PIPE) | np.isin(col, TOKEN_COLORS)):
            cols.append(int(c))
    assert 3 <= len(cols) <= 4, f"expected 3-4 pipes, got {cols}"
    return cols


def _parse_bridges(grid: np.ndarray, pipe_cols: list[int]) -> tuple[dict[tuple[int, int], dict[str, int | bool]], dict[tuple[int, int], tuple[int, int]]]:
    bridges: dict[tuple[int, int], dict[str, int | bool]] = {}
    incident: dict[tuple[int, int], tuple[int, int]] = {}
    for left in range(len(pipe_cols) - 1):
        c0 = pipe_cols[left]
        c1 = pipe_cols[left + 1]
        assert c1 > c0
        for r in range(grid.shape[0]):
            segment = grid[r, c0 + 1 : c1]
            if segment.size == 0:
                continue
            color = int(segment[0])
            if color not in BRIDGE_COLORS:
                continue
            if not np.all(segment == color):
                continue
            key = (int(r), int(left))
            bridges[key] = {"row": int(r), "left": int(left), "right": int(left + 1), "color": color, "used": False}
            for pipe in (left, left + 1):
                slot = (int(r), int(pipe))
                assert slot not in incident, "a pipe-row may not expose two competing bridges"
                incident[slot] = key
    return bridges, incident


def _simulate(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pipe_cols = _pipe_columns(g)
    token_positions = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(_token_mask(g))]
    token_colors = [color for _, _, color in token_positions]
    assert len(token_colors) == len(set(token_colors)), "token colors must be unique"
    assert 2 <= len(token_positions) <= 4

    for c0, c1 in zip(pipe_cols, pipe_cols[1:]):
        assert c1 - c0 >= 2

    base = g.copy()
    for r, c, _color in token_positions:
        base[r, c] = PIPE

    bridges, incident = _parse_bridges(g, pipe_cols)
    tokens: list[TokenState] = []
    for r, c, color in sorted(token_positions, key=lambda item: item[2]):
        pipe = pipe_cols.index(int(c))
        tokens.append(TokenState(color=int(color), row=int(r), pipe=int(pipe)))

    used_bridges: list[tuple[int, int]] = []
    blocked_attempts = 0
    same_tick_conflicts = 0
    per_token_crossings = {token.color: 0 for token in tokens}

    max_ticks = g.shape[0] * g.shape[1] * 4
    for _tick in range(max_ticks):
        start_requests: dict[tuple[int, int], list[TokenState]] = {}
        moved_this_tick = False

        for token in tokens:
            if token.mode == "crossing":
                continue
            key = incident.get((token.row, token.pipe))
            if key is not None:
                bridge = bridges[key]
                if not bool(bridge["used"]):
                    start_requests.setdefault(key, []).append(token)
                else:
                    blocked_attempts += 1

        for requesters in start_requests.values():
            if len(requesters) > 1:
                same_tick_conflicts += 1

        if same_tick_conflicts:
            if return_meta:
                return None, {"valid": False, "reason": "simultaneous bridge request"}
            raise AssertionError("ambiguous simultaneous bridge request")

        for token in tokens:
            if token.mode == "crossing":
                token.delay -= 1
                moved_this_tick = True
                if token.delay == 0:
                    token.mode = "falling"
                    token.pipe = token.target_pipe
                continue

            key = incident.get((token.row, token.pipe))
            if key is not None and key in start_requests:
                bridge = bridges[key]
                bridge["used"] = True
                used_bridges.append(key)
                left = int(bridge["left"])
                right = int(bridge["right"])
                token.mode = "crossing"
                token.target_pipe = right if token.pipe == left else left
                token.delay = abs(pipe_cols[token.target_pipe] - pipe_cols[token.pipe])
                per_token_crossings[token.color] += 1
                moved_this_tick = True
                continue

            if token.row + 1 < g.shape[0] and int(base[token.row + 1, pipe_cols[token.pipe]]) == PIPE:
                token.row += 1
                moved_this_tick = True

        if not moved_this_tick:
            break
    else:
        if return_meta:
            return None, {"valid": False, "reason": "simulation timeout"}
        raise AssertionError("simulation timeout")

    out = base.copy()
    for row, left in used_bridges:
        c0 = pipe_cols[left]
        c1 = pipe_cols[left + 1]
        out[row, c0 + 1 : c1] = BLACK

    final_positions: dict[int, tuple[int, int]] = {}
    for token in tokens:
        assert token.mode == "falling"
        c = pipe_cols[token.pipe]
        final_positions[token.color] = (int(token.row), int(c))
        out[token.row, c] = token.color

    meta = {
        "valid": True,
        "pipe_cols": pipe_cols,
        "used_bridges": tuple(used_bridges),
        "unused_count": len(bridges) - len(set(used_bridges)),
        "blocked_attempts": blocked_attempts,
        "same_tick_conflicts": same_tick_conflicts,
        "per_token_crossings": per_token_crossings,
        "final_positions": final_positions,
        "start_pipes": {color: pipe_cols.index(c) for _r, c, color in token_positions},
        "final_pipes": {color: pipe for color, (_r, c) in final_positions.items() for pipe in [pipe_cols.index(c)]},
    }
    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Simulate falling tokens that can reroute once through same-row colored pipe connectors.

    Input structure:
    - Black cells form pipe walls, with 3-4 narrow vertical gray shafts as the passable pipe interiors.
    - Unique token colors from 6-9 start at pipe entrances, and connector colors 1-4 appear as horizontal wall bridges between adjacent pipes.
    - Each bridge is a same-colored wall segment on one row between two neighboring pipe shafts.

    Rule:
    - All tokens fall downward through the gray shafts.
    - When a token reaches a row where its current pipe touches an unused colored bridge, it traverses horizontally into the neighboring pipe, then resumes falling.
    - Each bridge may be used only once; after the first traversal it closes back to black, so later tokens on that row continue straight down instead.
    - Tokens keep their colors, and the output shows their final resting cells plus any bridges that were consumed and closed.

    Key insight:
    - The connector colors do not recolor tokens; they are one-shot routing instructions whose consumption changes the later paths of other tokens.
    """
    out, meta = _simulate(grid, return_meta=True)
    assert meta["valid"], meta.get("reason", "invalid input")
    assert out is not None
    return out


def _build_canvas(height: int, num_pipes: int, left_margin: int, right_margin: int) -> tuple[np.ndarray, list[int]]:
    width = left_margin + (4 * num_pipes - 1) + right_margin
    grid = np.full((height, width), BLACK, dtype=int)
    pipe_cols = [left_margin + 1 + 4 * idx for idx in range(num_pipes)]
    for c in pipe_cols:
        grid[: height - 1, c] = PIPE
    return grid, pipe_cols


def _add_bridge(grid: np.ndarray, pipe_cols: list[int], row: int, left: int, color: int) -> None:
    c0 = pipe_cols[left]
    c1 = pipe_cols[left + 1]
    grid[row, c0 + 1 : c1] = color


def _random_bridge_color(rng: np.random.Generator) -> int:
    return int(rng.choice(np.array(BRIDGE_COLORS)))


def _reserve_token_pipes(
    num_pipes: int,
    blocked_mode: bool,
    rng: np.random.Generator,
) -> tuple[list[int], int | None]:
    token_count = int(rng.integers(2, min(4, num_pipes) + 1))
    if not blocked_mode:
        pipes = sorted(rng.choice(np.arange(num_pipes), size=token_count, replace=False).tolist())
        return pipes, None

    trio_start = int(rng.integers(0, num_pipes - 2))
    required = [trio_start, trio_start + 1]
    remaining = [pipe for pipe in range(num_pipes) if pipe not in required]
    extra_count = token_count - len(required)
    extras = []
    if extra_count > 0:
        extras = rng.choice(np.array(remaining), size=extra_count, replace=False).tolist()
    pipes = sorted(required + extras)
    return pipes, trio_start


def _scatter_tokens(grid: np.ndarray, pipe_cols: list[int], token_pipes: list[int], rng: np.random.Generator) -> None:
    colors = rng.permutation(np.array(TOKEN_COLORS))[: len(token_pipes)]
    for pipe, color in zip(token_pipes, colors, strict=True):
        grid[0, pipe_cols[pipe]] = int(color)


def _bridge_ok(existing: dict[tuple[int, int], int], row: int, left: int) -> bool:
    if (row, left) in existing:
        return False
    for pipe in (left, left + 1):
        for delta_left in (-1, 0):
            key = (row, pipe + delta_left)
            if key in existing and pipe in (key[1], key[1] + 1):
                return False
    return True


def _semantic_checks(inp: np.ndarray, out: np.ndarray, blocked_mode: bool, require_unused: bool) -> bool:
    sim_out, meta = _simulate(inp, return_meta=True)
    if not meta["valid"] or sim_out is None or not np.array_equal(sim_out, out):
        return False
    if len(set(meta["final_pipes"].values())) != len(meta["final_pipes"]):
        return False
    moved = sum(int(meta["start_pipes"][color] != meta["final_pipes"][color]) for color in meta["start_pipes"])
    if moved == 0:
        return False
    if len(meta["used_bridges"]) == 0:
        return False
    if meta["same_tick_conflicts"] != 0:
        return False
    if require_unused and meta["unused_count"] <= 0:
        return False
    if blocked_mode and meta["blocked_attempts"] <= 0:
        return False
    if not blocked_mode and max(meta["per_token_crossings"].values(), default=0) <= 0:
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    blocked_mode = bool(rng.random() < 0.45)
    num_pipes = 4 if blocked_mode else int(rng.choice(np.array([3, 4])))
    height = int(rng.integers(17, 24))
    left_margin = int(rng.integers(0, 3))
    right_margin = int(rng.integers(0, 3))
    grid, pipe_cols = _build_canvas(height, num_pipes, left_margin, right_margin)

    require_unused = blocked_mode
    if blocked_mode:
        token_pipes = [0, 1]
        trio_start = 0
    else:
        token_pipes = [0, 1]
        if num_pipes == 4 and rng.random() < 0.65:
            token_pipes.append(3)
        trio_start = None
    _scatter_tokens(grid, pipe_cols, token_pipes, rng)

    bridges: dict[tuple[int, int], int] = {}

    if blocked_mode:
        row_unused = int(rng.integers(2, 5))
        row_ab = int(rng.integers(row_unused + 2, min(height // 2, row_unused + 6)))
        row_bc = int(rng.integers(row_ab + 3, height - 4))
        if not (row_unused < row_ab < row_bc):
            return None
        bridges[(row_ab, trio_start)] = _random_bridge_color(rng)
        bridges[(row_bc, trio_start + 1)] = _random_bridge_color(rng)
        bridges[(row_unused, 2)] = _random_bridge_color(rng)
        if row_bc + 2 < height - 2 and rng.random() < 0.7:
            row_cd = int(rng.integers(row_bc + 2, height - 2))
            bridges[(row_cd, 2)] = _random_bridge_color(rng)
    else:
        row_12 = int(rng.integers(2, max(5, height // 3)))
        row_01 = int(rng.integers(row_12 + 2, height - 4))
        bridges[(row_12, 1)] = _random_bridge_color(rng)
        bridges[(row_01, 0)] = _random_bridge_color(rng)
        if num_pipes == 4 and 3 not in token_pipes:
            if rng.random() < 0.75 and row_12 + 2 < height - 2:
                row_23 = int(rng.integers(row_12 + 2, height - 2))
                bridges[(row_23, 2)] = _random_bridge_color(rng)
            if rng.random() < 0.6:
                row_unused = int(rng.integers(2, row_12 + 1))
                bridges[(row_unused, 2)] = _random_bridge_color(rng)
                require_unused = True

    for (row, left), color in bridges.items():
        if not _bridge_ok({k: v for k, v in bridges.items() if k != (row, left)}, row, left):
            return None
        _add_bridge(grid, pipe_cols, row, left, color)

    out, meta = _simulate(grid, return_meta=True)
    if out is None or not meta["valid"]:
        return None
    if not _semantic_checks(grid, out, blocked_mode, require_unused):
        return None

    in_sig = _grid_sig(grid)
    out_sig = _grid_sig(out)
    if in_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(in_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routed-pipe input")
