# Combo variant of e12f9a14 (index 1)
# combo_spec: D1/0+D2/0
# Name: 正交射线+碰撞即停

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The framed 2x2 blocks and shared frame color stay intact, but only side gaps are
#    meaningful. A top gap emits upward, a bottom gap downward, a left gap leftward,
#    and a right gap rightward. The second change is that collisions no longer create
#    holes or turns: beams are drawn in visible row-major order of the gap cells, and a
#    later beam stops as soon as its next cell is already part of an earlier beam.
#    Example: a right-side gap at seed (6, 5) shoots right, and a bottom-side gap at
#    seed (3, 11) shoots downward; they would meet at (6, 11). In the source task a
#    cardinal/cardinal collision would leave a blank center and both continue. Here the
#    earlier seed in row-major order wins: if (3, 11) is earlier, the vertical ray
#    paints through (6, 11) and the horizontal ray stops at (6, 10).
# b) Complexity calibration:
#    Replacing diagonal geometry and redirection with straight-stop behavior can become
#    too weak if samples contain only isolated rays. The generator therefore uses only
#    side gaps, enforces at least one horizontal ray and one vertical ray, and rejects
#    instances with zero actual orthogonal stop events under the new solver.
# c) Hidden assumptions audited from the original main():
#    - 8-direction inference no longer applies; we intentionally discard corner gaps.
#    - Collision-class branching (45 vs 90 degrees) and all redirection state vanish.
#    - Retroactive edits such as truncation, unpainting, blocked-hole bookkeeping, and
#      synchronized merged beams are invalid under the new rules because history never
#      rewrites; once an earlier beam paints a cell it permanently blocks later beams.
#    - Event-time simulation is replaced by a stronger, explicit assumption: gap order
#      itself is the priority rule, so each beam can be drawn to completion before the
#      next one starts.
# d) Placement legality:
#    Frames still occupy disjoint 4x4 boxes with >=2 cells of clearance and stay away
#    from the border. Candidate side gaps are validated on a full-frame obstacle grid so
#    every ray has unobstructed straight runway to the boundary unless another beam later
#    blocks it. The generator also forbids ray paths from passing through another gap
#    seed, preventing hidden seed-overwrite ambiguities.
# e) Visual observability:
#    Humans can see both ingredients directly. Side openings visibly imply orthogonal
#    travel, and at least one sample ray is truncated exactly where it reaches a fully
#    drawn perpendicular ray. Because seed order is row-major on visible gap locations,
#    the "winner" is inferable without relying on color ids or hidden metadata.
# f) Visual cleanliness:
#    The generator rejects duplicate horizontal rows and duplicate vertical columns, so
#    same-axis overlaps and head-on stacks do not clutter the board. Collisions are
#    mostly clean T-junctions or plus-like attempted crossings where only one branch
#    survives, which is easier to read than the original redirected bands.
# g) Color inference:
#    No color remapping is introduced. Every ray uses the color of its own inner 2x2
#    block, so the new reasoning burden is geometric and order-based rather than palette
#    based.
# New solver assumptions checked against generator:
#    - every emitted gap is a side gap with one cardinal outward normal;
#    - no earlier beam path runs through a later gap seed;
#    - at least one horizontal ray and one vertical ray are present;
#    - the accepted instance contains at least one actual orthogonal stop event under
#      the fused solver;
#    - same-axis duplicate rows/columns are absent in generated instances.


_SIDE_GAPS = (
    (0, 1), (0, 2),
    (1, 3), (2, 3),
    (3, 2), (3, 1),
    (2, 0), (1, 0),
)

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))
    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        if dr == 0 and dc == 0:
            continue
        if dr != 0 and dc != 0:
            continue
        ok = True
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        else:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        if ok:
            out.append((r, c))
    return out


def _gap_dir_for_block(br: int, bc: int, gr: int, gc: int) -> tuple[int, int] | None:
    dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
    dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
    if dr == 0 and dc == 0:
        return None
    if dr != 0 and dc != 0:
        return None
    return dr, dc


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    if gr == 0:
        return (-1, 0)
    if gr == 3:
        return (1, 0)
    if gc == 0:
        return (0, -1)
    if gc == 3:
        return (0, 1)
    raise AssertionError("expected side gap")


def _axis(direction: tuple[int, int]) -> str:
    return "h" if direction[0] == 0 else "v"


def _collect_beam_specs(g: np.ndarray) -> tuple[int, list[tuple[tuple[int, int], int, tuple[int, int]]]]:
    bg = get_bg(g)
    specs: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for color, br, bc in _find_blocks_2x2(g, bg):
        frame_color = _infer_frame_color(g, bg, br, bc, color)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_color, br, bc):
            direction = _gap_dir_for_block(br, bc, gr, gc)
            if direction is None:
                continue
            specs.append(((gr, gc), int(color), direction))
    specs.sort(key=lambda item: item[0])
    return bg, specs


def _simulate(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, specs = _collect_beam_specs(g)
    assert specs, "expected at least one side-gap beam"

    out = g.copy()
    painted_axis: dict[tuple[int, int], str] = {}

    stats = {
        "beam_count": len(specs),
        "edge_stops": 0,
        "frame_stops": 0,
        "collision_stops": 0,
        "orthogonal_stops": 0,
        "same_axis_stops": 0,
        "move_steps": 0,
    }

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def paint(cell: tuple[int, int], color: int, axis: str) -> None:
        out[cell[0], cell[1]] = int(color)
        painted_axis[cell] = axis

    for seed, color, direction in specs:
        assert int(g[seed[0], seed[1]]) == bg, "expected a background-valued gap seed"
        assert seed not in painted_axis, "later seed was blocked by an earlier beam path"

        axis = _axis(direction)
        paint(seed, color, axis)
        pos = seed

        while True:
            nxt = (pos[0] + direction[0], pos[1] + direction[1])
            if not in_bounds(nxt):
                stats["edge_stops"] += 1
                break
            if int(g[nxt[0], nxt[1]]) != bg:
                stats["frame_stops"] += 1
                break
            if nxt in painted_axis:
                stats["collision_stops"] += 1
                if painted_axis[nxt] == axis:
                    stats["same_axis_stops"] += 1
                else:
                    stats["orthogonal_stops"] += 1
                break
            paint(nxt, color, axis)
            pos = nxt
            stats["move_steps"] += 1

    stats["painted_cells"] = int(np.count_nonzero(out != g))
    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw orthogonal rays from framed 2x2 blocks, stopping later rays at earlier paths.

    Input structure:
    - The dominant color is background. Several monochrome 2x2 blocks sit inside 4x4 borders of one shared frame color.
    - Only side openings on those borders matter here; each such background-colored side gap is a ray source.

    Rule:
    - A top, bottom, left, or right gap emits one ray of the inner block's color in that outward cardinal direction.
    - Gap sources are processed in row-major order of their grid coordinates. Each ray paints straight through background cells until it leaves the grid, hits a frame/block cell, or its next cell is already part of an earlier ray.
    - When a later ray reaches another ray's path, it does not punch a hole or turn; it simply stops immediately before the occupied cell.

    Key insight:
    - The puzzle fuses two changes at once: openings now encode orthogonal direction, and collisions are resolved entirely by visible drawing priority. Solving therefore depends on reading both the side-gap orientation and the row-major seed order.
    """
    return _simulate(np.asarray(grid, dtype=int), return_stats=False)


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (2, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 12, 4: 18}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 3
MIN_CROSSING_DEPTH = 2
MIN_TOTAL_GAPS = 3
MIN_BEAM_PIXELS_PER_GAP = 4
MIN_GEOMETRIC_ORDERED_CROSSINGS = 1
MIN_ORTHOGONAL_STOPS = 1
GEN_RETRIES = 500
GAP_PICK_RETRIES = 160
GAP_COUNT_WEIGHTS = {1: 0.10, 2: 0.35, 3: 0.35, 4: 0.20}


def _gap_combos(size: int) -> list[tuple[tuple[int, int], ...]]:
    combos = []
    for combo in itertools.combinations(_SIDE_GAPS, size):
        idxs = sorted(_BORDER_INDEX[p] for p in combo)
        if any((idxs[(i + 1) % size] - idxs[i]) % 12 == 1 for i in range(size)):
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 5)}


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    margin = PLACEMENT_MARGIN
    all_positions = [(r, c) for r in range(margin, n - 5) for c in range(margin, n - 5)]
    selected: list[tuple[int, int]] = []
    for _ in range(count):
        viable = [p for p in all_positions if all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])
    return selected


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _ray_cells(
    grid: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    cells = [seed]
    r, c = seed
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(grid[r, c]) != bg:
            return None
        cells.append((r, c))
    return cells


def _clear_gap_combos(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            ok = True
            for gr, gc in combo:
                ray = _ray_cells(obstacles, bg, (r0 + gr, c0 + gc), _gap_dir((gr, gc)))
                if ray is None or len(ray) - 1 < MIN_RAY_STEPS:
                    ok = False
                    break
            if ok:
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 10
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _sample_gap_count(available: dict[int, list[tuple[tuple[int, int], ...]]], rng: np.random.Generator) -> int:
    counts = sorted(available)
    weights = np.array([GAP_COUNT_WEIGHTS[c] for c in counts], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(counts, p=weights))


def _build_grid(
    n: int,
    bg: int,
    frame_color: int,
    positions: list[tuple[int, int]],
    block_colors: list[int],
    chosen_gaps: list[tuple[tuple[int, int], ...]],
) -> tuple[np.ndarray, int]:
    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
    return grid, gap_total


def _evaluate_layout(
    obstacles: np.ndarray,
    bg: int,
    positions: list[tuple[int, int]],
    chosen_gaps: list[tuple[tuple[int, int], ...]],
) -> dict[str, int] | None:
    rays: list[dict[str, object]] = []
    seeds: set[tuple[int, int]] = set()
    rows_h: set[int] = set()
    cols_v: set[int] = set()

    for (r0, c0), gaps in zip(positions, chosen_gaps):
        for gap in gaps:
            seed = (r0 + gap[0], c0 + gap[1])
            direction = _gap_dir(gap)
            cells = _ray_cells(obstacles, bg, seed, direction)
            if cells is None or len(cells) - 1 < MIN_RAY_STEPS:
                return None
            if seed in seeds:
                return None
            seeds.add(seed)
            axis = _axis(direction)
            if axis == "h":
                if seed[0] in rows_h:
                    return None
                rows_h.add(seed[0])
            else:
                if seed[1] in cols_v:
                    return None
                cols_v.add(seed[1])
            rays.append(
                {
                    "seed": seed,
                    "dir": direction,
                    "axis": axis,
                    "cells": cells,
                    "set": set(cells),
                    "index": {cell: idx for idx, cell in enumerate(cells)},
                }
            )

    if not rows_h or not cols_v:
        return None

    for ray in rays:
        tail = set(ray["cells"][1:])
        if tail & (seeds - {ray["seed"]}):
            return None

    ordered_crossings = 0
    for hray in rays:
        if hray["axis"] != "h":
            continue
        for vray in rays:
            if vray["axis"] != "v":
                continue
            cell = (hray["seed"][0], vray["seed"][1])
            if cell in seeds:
                continue
            if cell not in hray["set"] or cell not in vray["set"]:
                continue
            if int(obstacles[cell[0], cell[1]]) != bg:
                continue
            ih = hray["index"][cell]
            iv = vray["index"][cell]
            if ih < MIN_CROSSING_DEPTH or iv < MIN_CROSSING_DEPTH:
                continue
            winner = hray if hray["seed"] < vray["seed"] else vray
            if winner["index"][cell] >= len(winner["cells"]) - 1:
                continue
            ordered_crossings += 1

    if ordered_crossings < MIN_GEOMETRIC_ORDERED_CROSSINGS:
        return None

    return {
        "ray_count": len(rays),
        "ordered_crossings": ordered_crossings,
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    available_per_block: list[dict[int, list[tuple[tuple[int, int], ...]]]] = []
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        available_per_block.append(available)

    for _ in range(GAP_PICK_RETRIES):
        chosen_gaps: list[tuple[tuple[int, int], ...]] = []
        for available in available_per_block:
            gap_count = _sample_gap_count(available, rng)
            combos = available[gap_count]
            chosen_gaps.append(combos[int(rng.integers(len(combos)))])

        layout = _evaluate_layout(obstacles, bg, positions, chosen_gaps)
        if layout is None:
            continue

        grid, gap_total = _build_grid(n, bg, frame_color, positions, block_colors, chosen_gaps)
        if gap_total < MIN_TOTAL_GAPS:
            continue
        if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
            continue

        try:
            out, stats = _simulate(grid, return_stats=True)
        except Exception:
            continue

        if stats["orthogonal_stops"] < MIN_ORTHOGONAL_STOPS:
            continue
        if stats["same_axis_stops"] != 0:
            continue
        if stats["painted_cells"] < gap_total * MIN_BEAM_PIXELS_PER_GAP:
            continue
        if layout["ordered_crossings"] < MIN_GEOMETRIC_ORDERED_CROSSINGS:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
