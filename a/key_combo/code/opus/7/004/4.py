from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC = SourceFileLoader(
    "_arc_utils_opus_7_004_4",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
get_bg = _ARC.get_bg
in_bounds = _ARC.in_bounds
neighbors4 = _ARC.neighbors4


_RNG = random.Random()
_GEN_RETRIES = 400
_SIZE_RANGE = (16, 24)
_CONTROL_MARGIN = 5
_LOCAL_RADIUS = 4


def _candidate_specs(cr: int, cc: int) -> list[tuple[int, int, int, str]]:
    specs: list[tuple[int, int, int, str]] = []
    for dc in range(-2, 3):
        specs.append((cr - 2, cc + dc, Color.GREY, "up"))
    for dr in (-1, 0, 1):
        specs.append((cr + dr, cc - 2, Color.RED, "left"))
        specs.append((cr + dr, cc + 2, Color.RED, "right"))
    specs.append((cr - 1, cc, Color.GREY, "up"))
    for dc in range(-2, 3):
        specs.append((cr + 2, cc + dc, Color.TEAL, "down"))
    return specs


def _reachable_black(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for nr, nc in neighbors4(*start):
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.BLACK and not seen[nr, nc]:
            seen[nr, nc] = True
            q.append((int(nr), int(nc)))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if seen[nr, nc] or int(grid[nr, nc]) != Color.BLACK:
                continue
            seen[nr, nc] = True
            q.append((int(nr), int(nc)))
    return seen


def _axial_run_lengths(grid: np.ndarray, cr: int, cc: int) -> dict[str, int]:
    dirs = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    out: dict[str, int] = {}
    for name, (dr, dc) in dirs.items():
        dist = 0
        r, c = cr, cc
        while True:
            r += dr
            c += dc
            if not in_bounds(grid.shape, r, c) or int(grid[r, c]) != Color.BLACK:
                break
            dist += 1
        out[name] = dist
    return out


def _analyze_control(grid: np.ndarray, cr: int, cc: int) -> dict[str, object]:
    reach = _reachable_black(grid, (cr, cc))
    per_group = {"up": 0, "down": 0, "left": 0, "right": 0}
    reachable = 0
    blocked = 0
    signature: list[str] = []
    for r, c, _color, group in _candidate_specs(cr, cc):
        assert in_bounds(grid.shape, r, c), "candidate must be in-bounds"
        is_reachable = int(grid[r, c]) == Color.BLACK and bool(reach[r, c])
        signature.append("1" if is_reachable else "0")
        if is_reachable:
            reachable += 1
            per_group[group] += 1
        else:
            blocked += 1
    return {
        "reachable": reachable,
        "blocked": blocked,
        "per_group": per_group,
        "signature": "".join(signature),
        "runs": _axial_run_lengths(grid, cr, cc),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a scaffold template around each control point, but only onto maze cells reachable through black corridors.

    Input structure:
    - Black (0) pixels are maze corridors, teal (8) pixels are maze walls, and one or two green (3) singleton pixels are control points.
    - Each control lies at a corridor intersection with room around it for the fixed scaffold candidate offsets.
    - Candidate scaffold cells may coincide with walls or with disconnected corridor stubs.

    Rule:
    - For each green control point, form the fixed scaffold candidate set: a grey five-pixel cap at row -2, red side columns at column offsets +/-2 for rows -1..1, one grey pixel at (-1,0), and a teal five-pixel baseline at row +2.
    - A candidate is drawn only if that target cell is a black corridor cell reachable from the control by a 4-connected path that stays entirely on black cells after leaving the control.
    - Unreachable candidates are omitted; teal wall cells keep their original wall color; control points stay green.

    Key insight:
    - The scaffold geometry alone is insufficient: each candidate pixel must also survive a maze-connectivity test, so the wall topology clips the shape into a location-specific fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.array(g, copy=True)
    controls = np.argwhere(g == Color.GREEN)
    for cr, cc in controls:
        cr = int(cr)
        cc = int(cc)
        reach = _reachable_black(g, (cr, cc))
        for r, c, color, _group in _candidate_specs(cr, cc):
            if not in_bounds(g.shape, r, c):
                continue
            if int(g[r, c]) != Color.BLACK:
                continue
            if reach[r, c]:
                out[r, c] = int(color)

    out[g == Color.GREEN] = Color.GREEN
    out[g == Color.TEAL] = Color.TEAL
    return out


def _choose_controls(h: int, w: int, n_controls: int) -> list[tuple[int, int]]:
    candidates = [
        (cr, cc)
        for cr in range(_CONTROL_MARGIN, h - _CONTROL_MARGIN)
        for cc in range(_CONTROL_MARGIN, w - _CONTROL_MARGIN)
    ]
    _RNG.shuffle(candidates)

    def compatible(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return (
            abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 8
            and (abs(a[0] - b[0]) > 2 * _LOCAL_RADIUS or abs(a[1] - b[1]) > 2 * _LOCAL_RADIUS)
        )

    if n_controls == 1:
        if not candidates:
            raise RuntimeError("no valid center positions")
        return [candidates[0]]

    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            if compatible(first, second):
                return [first, second]
    raise RuntimeError("failed to place controls")


def _sample_lengths() -> dict[str, int]:
    while True:
        lengths = {name: _RNG.randint(2, 4) for name in ("up", "down", "left", "right")}
        if sum(v >= 3 for v in lengths.values()) < 2:
            continue
        if sum(v == 2 for v in lengths.values()) < 1:
            continue
        return lengths


def _sample_pattern() -> dict[str, tuple[int, int]]:
    while True:
        pattern = {
            "head": (_RNG.randint(0, 2), _RNG.randint(0, 2)),
            "base": (_RNG.randint(0, 2), _RNG.randint(0, 2)),
            "left_col": (_RNG.randint(0, 1), _RNG.randint(0, 1)),
            "right_col": (_RNG.randint(0, 1), _RNG.randint(0, 1)),
        }
        counts = [
            sum(pattern["head"]) + 1,
            sum(pattern["base"]) + 1,
            sum(pattern["left_col"]) + 1,
            sum(pattern["right_col"]) + 1,
        ]
        if max(counts) >= 3 and min(counts) <= 2:
            return pattern


def _carve_local_maze(
    grid: np.ndarray,
    cr: int,
    cc: int,
    lengths: dict[str, int],
    pattern: dict[str, tuple[int, int]],
) -> None:
    dirs = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    for name, (dr, dc) in dirs.items():
        for step in range(1, lengths[name] + 1):
            grid[cr + dr * step, cc + dc * step] = Color.BLACK

    head_left, head_right = pattern["head"]
    for dc in range(-head_left, head_right + 1):
        grid[cr - 2, cc + dc] = Color.BLACK

    base_left, base_right = pattern["base"]
    for dc in range(-base_left, base_right + 1):
        grid[cr + 2, cc + dc] = Color.BLACK

    left_up, left_down = pattern["left_col"]
    for dr in range(-left_up, left_down + 1):
        grid[cr + dr, cc - 2] = Color.BLACK

    right_up, right_down = pattern["right_col"]
    for dr in range(-right_up, right_down + 1):
        grid[cr + dr, cc + 2] = Color.BLACK

    for name, (dr, dc) in dirs.items():
        if lengths[name] < 4 or _RNG.random() < 0.45:
            continue
        end_r = cr + dr * lengths[name]
        end_c = cc + dc * lengths[name]
        lateral = (0, _RNG.choice([-1, 1])) if dr else (_RNG.choice([-1, 1]), 0)
        nr = end_r + lateral[0]
        nc = end_c + lateral[1]
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.TEAL:
            grid[nr, nc] = Color.BLACK

    grid[cr, cc] = Color.GREEN


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if get_bg(grid) != Color.TEAL:
        return False

    controls = [tuple(map(int, pt)) for pt in np.argwhere(grid == Color.GREEN)]
    if len(controls) not in (1, 2):
        return False

    if len(controls) == 2:
        (r1, c1), (r2, c2) = controls
        if abs(r1 - r2) + abs(c1 - c2) < 8:
            return False

    signatures = []
    any_blocked = False
    for cr, cc in controls:
        runs = _axial_run_lengths(grid, cr, cc)
        if min(runs.values()) < 2:
            return False
        if sum(v >= 3 for v in runs.values()) < 2:
            return False
        if sum(v == 2 for v in runs.values()) < 1:
            return False

        info = _analyze_control(grid, cr, cc)
        if not (7 <= int(info["reachable"]) <= 14):
            return False
        if int(info["blocked"]) < 3:
            return False
        if sum(v > 0 for v in info["per_group"].values()) < 3:
            return False
        signatures.append(str(info["signature"]))
        any_blocked = any_blocked or int(info["blocked"]) > 0

    if len(signatures) == 2 and signatures[0] == signatures[1]:
        return False
    if not any_blocked:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = _RNG.randint(*_SIZE_RANGE)
    w = _RNG.randint(*_SIZE_RANGE)
    grid = np.full((h, w), Color.TEAL, dtype=int)

    n_controls = 2 if _RNG.random() < 0.55 else 1
    try:
        controls = _choose_controls(h, w, n_controls)
    except RuntimeError:
        return None

    patterns: list[str] = []
    for cr, cc in controls:
        lengths = _sample_lengths()
        pattern = _sample_pattern()
        _carve_local_maze(grid, cr, cc, lengths, pattern)
        patterns.append(_analyze_control(grid, cr, cc)["signature"])

    if len(patterns) == 2 and patterns[0] == patterns[1]:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scaffold-maze input")
