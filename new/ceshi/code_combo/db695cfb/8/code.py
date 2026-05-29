# Combo variant of db695cfb (index 8)
# combo_spec: D8/1+D9/0
# Name: 触发交互+迭代传播

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original task: blue endpoints define one diagonal segment; magenta pixels on that
#    segment emit one pair of perpendicular rays and then stop. Fused task: the emitted
#    rays are themselves active carriers. If a slash ray hits another magenta pixel, that
#    pixel fires a backslash ray; if a backslash ray later crosses a slash ray at a blank
#    cell, that crossing cell also becomes a new trigger. Example sketch:
#    - Seed on a backslash blue segment emits a slash line.
#    - That slash line hits an off-segment magenta relay, so a new backslash line appears.
#    - Another seed emits a backslash line; the two families then cross and create fresh
#      magenta trigger cells. Propagation alternates orientations until no unseen
#      (cell, incoming-orientation) event remains.
# b) Complexity calibration:
#    Only two diagonal orientations exist, so the state space stays small and visually
#    legible. The generator repeats the same alternation pattern across examples:
#    seed -> relay -> crossing. This makes the cascade observable without introducing
#    extra hidden categories.
# c) Broken implicit assumptions from the original solver:
#    - Triggers are no longer "only on blue diagonals".
#    - A trigger is no longer one-shot; the same cell may need separate handling for the
#      two incoming diagonal orientations.
#    - Crossings are no longer passive overlap; they create new work.
#    Therefore the solver must track orientation-tagged events, not a flat trigger list.
# d) Placement legality:
#    The generator keeps blue segments disjoint, keeps all chosen trigger/relay cells
#    in-bounds, and rejects layouts whose required intersections land on occupied cells.
# e) Visual observability:
#    Every generated sample contains at least two initial blue diagonals, two off-diagonal
#    magenta relays, and at least one fresh crossing trigger. The output therefore shows
#    more magenta structure than the one-step original, and the extra structure is tied to
#    visible ray interactions rather than to invisible color ordering.
# f) Visual cleanliness:
#    Segment paths are kept disjoint and relay/crossing cells are forced away from segment
#    cells when possible. This avoids ambiguous blue/magenta stacking on the same local
#    area and keeps the network sparse enough to read by eye.
# g) New solver assumptions checked against the generator:
#    - There is still at least one diagonal-aligned blue pair; generator always creates
#      exactly two main segments with color 1.
#    - Non-blue cells intended to matter use a single trigger color 6; generator enforces
#      that and keeps the background different from 1 and 6.
#    - Useful cascades require opposite-orientation rays and later relays; generator
#      explicitly constructs both and rejects samples whose iterative result matches the
#      one-step baseline.


LINE_COLOR = 1
TRIGGER_COLOR = 6

H_RANGE = (11, 22)
W_RANGE = (11, 22)
SEG_LEN_RANGE = (5, 11)
GEN_RETRIES = 500
SEGMENT_PLACE_RETRIES = 120


Coord = tuple[int, int]
Event = tuple[int, int, int, int]  # r, c, incoming_orientation, color


def _diag_path(p: Coord, q: Coord) -> list[Coord]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _orientation_dirs(orientation: int) -> tuple[Coord, Coord]:
    if orientation == 1:
        return ((1, 1), (-1, -1))
    return ((1, -1), (-1, 1))


def _full_diagonal_cells(origin: Coord, orientation: int, H: int, W: int) -> list[Coord]:
    cells = [origin]
    for dr, dc in _orientation_dirs(orientation):
        r, c = origin
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _line_constants(cell: Coord) -> tuple[int, int]:
    r, c = cell
    return r - c, r + c


def _intersect(backslash_const: int, slash_const: int, H: int, W: int) -> Coord | None:
    total = backslash_const + slash_const
    diff = slash_const - backslash_const
    if total % 2 != 0 or diff % 2 != 0:
        return None
    r = total // 2
    c = diff // 2
    if 0 <= r < H and 0 <= c < W:
        return (r, c)
    return None


def _find_line_color(non_bg: list[tuple[int, int, int]]) -> int:
    by_color: dict[int, list[Coord]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best: tuple[int, int] | None = None
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"
    return best[1]


def _collect_segments(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    H, W = grid.shape
    non_bg = [(r, c, int(grid[r, c])) for r in range(H) for c in range(W) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    line_color = _find_line_color(non_bg)
    pts = [(r, c) for r, c, col in non_bg if col == line_color]

    groups_a: dict[int, list[Coord]] = {}
    groups_b: dict[int, list[Coord]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    segments: list[dict[str, object]] = []
    seen_pairs: set[tuple[Coord, Coord]] = set()

    def add_segment(group: list[Coord], orientation: int) -> None:
        if len(group) < 2:
            return
        ordered = sorted(group)
        p, q = ordered[0], ordered[-1]
        key = (p, q)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        segments.append(
            {
                "endpoints": (p, q),
                "path": _diag_path(p, q),
                "orientation": orientation,
            }
        )

    for group in groups_a.values():
        add_segment(group, 1)
    for group in groups_b.values():
        add_segment(group, -1)

    return line_color, segments


def _simulate(grid: np.ndarray, *, iterative: bool) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    line_color, segments = _collect_segments(g, bg)
    input_trigger_cells = {
        (r, c)
        for r in range(H)
        for c in range(W)
        if int(g[r, c]) != bg and int(g[r, c]) != line_color
    }

    seed_trigger_cells: set[Coord] = set()
    queue: deque[Event] = deque()
    queued: set[Event] = set()
    processed: set[Event] = set()
    seen_by_orientation: dict[int, set[Coord]] = {1: set(), -1: set()}

    metrics = {
        "seed_event_count": 0,
        "relay_activation_cells": set(),
        "cross_activation_cells": set(),
        "input_trigger_cells": set(input_trigger_cells),
        "segment_trigger_cells": seed_trigger_cells,
    }

    def enqueue(event: Event, cause: str) -> None:
        if event in processed or event in queued:
            return
        queued.add(event)
        queue.append(event)
        r, c, _incoming_orientation, _col = event
        if cause == "seed":
            metrics["seed_event_count"] += 1
        elif cause == "relay" and (r, c) not in seed_trigger_cells:
            metrics["relay_activation_cells"].add((r, c))
        elif cause == "cross" and (r, c) not in input_trigger_cells:
            metrics["cross_activation_cells"].add((r, c))

    for segment in segments:
        path = segment["path"]
        orientation = int(segment["orientation"])
        for r, c in path:
            v = int(g[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                seed_trigger_cells.add((r, c))
                enqueue((r, c, orientation, v), "seed")

    while queue:
        r0, c0, incoming_orientation, tcol = queue.popleft()
        event = (r0, c0, incoming_orientation, tcol)
        queued.remove(event)
        if event in processed:
            continue
        processed.add(event)

        emit_orientation = -incoming_orientation

        if int(out[r0, c0]) == bg:
            out[r0, c0] = tcol

        if (r0, c0) not in seen_by_orientation[emit_orientation]:
            if iterative and (r0, c0) in seen_by_orientation[-emit_orientation]:
                enqueue((r0, c0, emit_orientation, tcol), "cross")
                enqueue((r0, c0, -emit_orientation, tcol), "cross")
            seen_by_orientation[emit_orientation].add((r0, c0))

        for dr, dc in _orientation_dirs(emit_orientation):
            r, c = r0, c0
            while True:
                r += dr
                c += dc
                if not (0 <= r < H and 0 <= c < W):
                    break

                if int(out[r, c]) == bg:
                    out[r, c] = tcol

                if iterative and (r, c) in input_trigger_cells:
                    enqueue((r, c, emit_orientation, int(g[r, c])), "relay")

                if iterative and (r, c) in seen_by_orientation[-emit_orientation]:
                    enqueue((r, c, emit_orientation, tcol), "cross")
                    enqueue((r, c, -emit_orientation, tcol), "cross")

                seen_by_orientation[emit_orientation].add((r, c))

    metrics["processed_event_count"] = len(processed)
    metrics["magenta_count"] = int(np.sum(out == TRIGGER_COLOR))
    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the blue diagonals, then let magenta triggers keep spawning perpendicular diagonal rays until the network stabilizes.

    Input structure:
    - The background is the dominant color. Blue pixels still appear as diagonal endpoint pairs that define one or more hidden 45-degree segments.
    - Magenta pixels can sit on those blue diagonals as seed triggers, and additional magenta pixels may appear off the diagonals as relays waiting to be hit later.

    Rule:
    - Fill each blue segment between its extreme same-diagonal blue endpoints.
    - Every non-blue pixel touched by a filled blue segment emits magenta rays along the perpendicular diagonal.
    - If a magenta ray passes through another magenta pixel, that pixel also fires and emits along the opposite diagonal orientation.
    - If opposite-orientation magenta rays cross, the crossing cell becomes a fresh magenta trigger as well.
    - Repeat this propagation until no new trigger-orientation event remains.

    Key insight:
    - The important state is not just "which magenta pixels exist" but "which diagonal orientation reached each magenta cell". The cascade alternates between the two diagonal families and must be simulated to closure.
    """

    out, _metrics = _simulate(grid, iterative=True)
    return out


def _try_place_segment(slope_type: str, seg_len: int, H: int, W: int, blocked: set[Coord]) -> list[Coord] | None:
    for _ in range(SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if seg_len >= H or seg_len >= W:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
        else:
            if seg_len >= H or seg_len >= W:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(cell in blocked for cell in path):
            continue
        return path
    return None


def _build_candidate_input(
    H: int,
    W: int,
    bg: int,
    path_a: list[Coord],
    path_b: list[Coord],
    seed_a: Coord,
    seed_b: Coord,
    relay_a: Coord,
    relay_b: Coord,
) -> np.ndarray:
    grid = np.full((H, W), bg, dtype=int)
    for cell in (path_a[0], path_a[-1], path_b[0], path_b[-1]):
        grid[cell] = LINE_COLOR
    for cell in (seed_a, seed_b, relay_a, relay_b):
        grid[cell] = TRIGGER_COLOR
    return grid


def _one_step_magenta_count(grid: np.ndarray) -> int:
    out, _metrics = _simulate(grid, iterative=False)
    return int(np.sum(out == TRIGGER_COLOR))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (LINE_COLOR, TRIGGER_COLOR)])

    len_a = random.randint(*SEG_LEN_RANGE)
    len_b = random.randint(*SEG_LEN_RANGE)
    max_len = min(H - 1, W - 1)
    if min(len_a, len_b) > max_len:
        return None
    len_a = min(len_a, max_len)
    len_b = min(len_b, max_len)

    path_a = _try_place_segment("backslash", len_a, H, W, set())
    if path_a is None:
        return None
    blocked = set(path_a)
    path_b = _try_place_segment("slash", len_b, H, W, blocked)
    if path_b is None:
        return None

    interior_a = path_a[1:-1]
    interior_b = path_b[1:-1]
    if len(interior_a) < 2 or len(interior_b) < 2:
        return None

    random.shuffle(interior_a)
    random.shuffle(interior_b)

    occupied = set(path_a) | set(path_b)

    for seed_a in interior_a:
        s_a = seed_a[0] + seed_a[1]
        for seed_b in interior_b:
            k_b = seed_b[0] - seed_b[1]
            cross0 = _intersect(k_b, s_a, H, W)
            if cross0 is None or cross0 in occupied:
                continue

            line_a = [
                cell
                for cell in _full_diagonal_cells(seed_a, -1, H, W)
                if cell not in occupied and cell != cross0 and cell != seed_b
            ]
            line_b = [
                cell
                for cell in _full_diagonal_cells(seed_b, 1, H, W)
                if cell not in occupied and cell != cross0 and cell != seed_a
            ]
            random.shuffle(line_a)
            random.shuffle(line_b)

            for relay_a in line_a:
                k_1, _ = _line_constants(relay_a)
                if k_1 == k_b:
                    continue
                emit_a = set(_full_diagonal_cells(relay_a, 1, H, W))
                if len(emit_a - set(path_a) - set(path_b)) < 4:
                    continue

                for relay_b in line_b:
                    if relay_b == relay_a:
                        continue
                    _, s_2 = _line_constants(relay_b)
                    if s_2 == s_a:
                        continue
                    cross2 = _intersect(k_1, s_2, H, W)
                    if cross2 is None:
                        continue
                    if cross2 in occupied or cross2 in {seed_a, seed_b, relay_a, relay_b, cross0}:
                        continue
                    if cross2 in _full_diagonal_cells(seed_a, -1, H, W) or cross2 in _full_diagonal_cells(seed_b, 1, H, W):
                        continue

                    inp = _build_candidate_input(H, W, bg, path_a, path_b, seed_a, seed_b, relay_a, relay_b)
                    out, metrics = _simulate(inp, iterative=True)
                    base_magenta = _one_step_magenta_count(inp)
                    cascade_gain = int(np.sum(out == TRIGGER_COLOR)) - base_magenta

                    if cascade_gain <= 0:
                        continue
                    if len(metrics["relay_activation_cells"]) < 2:
                        continue
                    if len(metrics["cross_activation_cells"]) < 2:
                        continue
                    if cross0 not in metrics["cross_activation_cells"] and cross2 not in metrics["cross_activation_cells"]:
                        continue
                    return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"Failed to generate valid fused input after {GEN_RETRIES} attempts")
