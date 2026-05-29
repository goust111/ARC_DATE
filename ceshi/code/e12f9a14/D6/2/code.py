# Variant of e12f9a14
# Dimension: D6 — 射线终止条件 (dst)
# Change: 射线延伸到网格边界或碰到另一条射线的路径时终止... → 到达目标标记像素终止

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D6 variant
# ----------------------------------
# concreteization:
# - We keep the original framed 2x2 colored source blocks.
# - To make "color-matched target" unambiguous, each frame now has exactly one gap,
#   hence each block emits exactly one ray.
# - Each source color also appears once as a lone 1x1 target marker placed farther
#   along that ray's outward direction. The output fills the corridor from the gap
#   to that target and stops there, leaving background cells beyond the target.
# - Small trace example:
#   original: a right-facing gap would keep painting to the grid edge.
#   variant: 4x4 frame with a right gap and a lone color-4 pixel five cells away
#   produces a color-4 segment that ends exactly at that marker, even if there are
#   still empty cells to the border.
#
# complexity calibration:
# - Minimum block count is raised to 2 so the same-color source/target pairing is
#   supported by repeated evidence in every sample.
# - We intentionally restrict to one gap per frame. Multiple rays of the same color
#   would require extra hidden matching rules between same-color targets, which is
#   too ambiguous for ARC-style induction.
# - Targets are always interior: each ray has at least MIN_TARGET_STEPS cells before
#   the target and at least MIN_TAIL_AFTER_TARGET unused cells after the target.
#
# implicit assumption review vs. the original main():
# - Original code assumes any non-background cell in front of a beam is either an
#   obstacle or part of a collision event. That is no longer true: the matching
#   target is a valid terminal non-background cell.
# - Original collision bookkeeping exists because beams can meet, truncate, and
#   redirect. In this variant the generator guarantees pairwise disjoint corridors,
#   so the solver can assert that no ray overlaps another ray or another target.
# - Original frames may contain 1-4 gaps. That assumption conflicts with a 1:1
#   source-target pairing, so the solver explicitly requires one structural gap.
#
# placement legality:
# - Frames keep the original non-overlap and margin constraints.
# - A target is only placed on the clear outward runway of its own gap.
# - Candidate plans are rejected if their visible cells overlap or if two diagonal
#   unit segments would geometrically cross inside the same 2x2 square.
# - The generator validates every produced input by running main().
#
# visual observability:
# - The new behavior is visible because each output line stops at an already-present
#   singleton marker of the same color instead of continuing to the border.
# - The rule does not depend on color ids or ordering. The only cue is the visible
#   same-color pairing between a 2x2 source block and a 1x1 target.
#
# visual cleanliness:
# - Rays are collision-free and targets are isolated, so the output does not stack
#   multiple interaction rules in the same local area.
# - Interior targets guarantee a clean "stopped early" look rather than a border hit.
#
# color inference:
# - No color remapping is introduced. The target marker uses exactly the same color
#   as its source block, which is the only stable, human-visible matching signal.


_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)


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
    assert len(frame_colors) == 1, "expected one frame color around each 2x2 block"
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
        assert dr != 0 or dc != 0, "expected gap on a frame border"
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    return out


def _block_cells(br: int, bc: int) -> set[tuple[int, int]]:
    return {(br, bc), (br, bc + 1), (br + 1, bc), (br + 1, bc + 1)}


def _gap_dir_from_abs(br: int, bc: int, gr: int, gc: int) -> tuple[int, int]:
    dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
    dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
    d = (dr, dc)
    assert d != (0, 0), "expected each frame gap to point outward"
    return d


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit rays from framed 2x2 blocks toward same-colored endpoint markers.

    Input structure:
    - The background is the most frequent color. A shared frame color surrounds each
      monochrome 2x2 source block with a 4x4 border.
    - Every frame has exactly one structural gap, and each source color also appears
      once as a lone 1x1 target pixel elsewhere in the grid.

    Rule:
    - A ray starts at the gap and moves outward in the direction implied by the gap
      position on the frame border.
    - The ray paints the gap and every background cell along that straight corridor.
    - Painting stops as soon as the next step would land on the matching same-colored
      target pixel; the target itself is already present in the input and serves as
      the visible endpoint.

    Key insight: The task is no longer "extend until the world stops you". The lone
    same-colored marker determines where each ray ends, so the solver must pair each
    framed source with its matching target before tracing the line.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one framed 2x2 block"

    seen_colors: set[int] = set()
    block_cells_by_color: dict[int, set[tuple[int, int]]] = {}
    beams: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, br, bc in blocks:
        assert color not in seen_colors, "variant expects unique source colors"
        seen_colors.add(color)
        frame_col = _infer_frame_color(g, bg, br, bc, color)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert len(gaps) == 1, "variant expects exactly one gap per frame"
        gap = gaps[0]
        beams.append((color, gap, _gap_dir_from_abs(br, bc, gap[0], gap[1])))
        block_cells_by_color[color] = _block_cells(br, bc)

    targets: dict[int, tuple[int, int]] = {}
    for color, block_cells in block_cells_by_color.items():
        cells = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        extras = [cell for cell in cells if cell not in block_cells]
        assert len(extras) == 1, "expected one lone target pixel per source color"
        targets[color] = extras[0]

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    for color, gap, d in sorted(beams):
        out[gap[0], gap[1]] = color
        pos = gap
        for _ in range(h * w):
            nxt = (pos[0] + d[0], pos[1] + d[1])
            assert in_bounds(nxt), "ray left the grid before reaching its target"
            if nxt == targets[color]:
                break
            assert int(g[nxt[0], nxt[1]]) == bg, "expected a clear runway from gap to target"
            assert int(out[nxt[0], nxt[1]]) == bg, "variant generator must keep rays disjoint"
            out[nxt[0], nxt[1]] = color
            pos = nxt
        else:
            raise AssertionError("ray tracing did not reach a matching target")

    return out


# --------------- generator (D6 variant) ---------------

BLOCK_COUNT_RANGE = (2, 6)              # uniform draw [lo, hi)
GRID_SIZE_MAX = 31                      # exclusive upper bound for square grid side
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
MIN_TARGET_STEPS = 3
MIN_TAIL_AFTER_TARGET = 2
MIN_BEAM_PIXELS_PER_RAY = 3
GEN_RETRIES = 400


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


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


def _ray_runway(
    obstacles: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    d: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = obstacles.shape
    cells: list[tuple[int, int]] = []
    nr, nc = seed[0] + d[0], seed[1] + d[1]
    while 0 <= nr < h and 0 <= nc < w and int(obstacles[nr, nc]) == bg:
        cells.append((nr, nc))
        nr += d[0]
        nc += d[1]
    return cells


def _is_diag_segment(seg: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    (r0, c0), (r1, c1) = seg
    return abs(r1 - r0) == 1 and abs(c1 - c0) == 1


def _diag_signature(seg: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], int] | None:
    if not _is_diag_segment(seg):
        return None
    (r0, c0), (r1, c1) = seg
    slope = (r1 - r0) * (c1 - c0)
    mid = (r0 + r1, c0 + c1)
    return mid, slope


def _candidate_plans(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> list[dict[str, object]]:
    plans: list[dict[str, object]] = []
    for gap in _BORDER_ORDER:
        seed = (r0 + gap[0], c0 + gap[1])
        d = _gap_dir(gap)
        runway = _ray_runway(obstacles, bg, seed, d)
        max_dist = len(runway) - MIN_TAIL_AFTER_TARGET
        if max_dist < MIN_TARGET_STEPS:
            continue
        for dist in range(MIN_TARGET_STEPS, max_dist + 1):
            target = runway[dist - 1]
            points = [seed] + runway[:dist]
            segments = list(zip(points, points[1:]))
            diag_sigs = [sig for sig in (_diag_signature(seg) for seg in segments) if sig is not None]
            plans.append(
                {
                    "gap": gap,
                    "target": target,
                    "cells": frozenset(points),
                    "segments": tuple(segments),
                    "diag_sigs": tuple(diag_sigs),
                    "tail_after": len(runway) - dist,
                }
            )
    return plans


def _compatible(
    plan: dict[str, object],
    used_cells: frozenset[tuple[int, int]],
    diag_map: dict[tuple[int, int], int],
) -> bool:
    if used_cells & plan["cells"]:
        return False
    for mid, slope in plan["diag_sigs"]:
        prev = diag_map.get(mid)
        if prev is not None and prev != slope:
            return False
    return True


def _choose_plans(
    obstacles: np.ndarray,
    bg: int,
    positions: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[dict[str, object]] | None:
    candidates = [_candidate_plans(obstacles, bg, r0, c0) for r0, c0 in positions]
    if any(not plans for plans in candidates):
        return None

    order = sorted(range(len(positions)), key=lambda i: len(candidates[i]))
    shuffled = []
    for plans in candidates:
        order_idx = rng.permutation(len(plans))
        shuffled.append([plans[int(i)] for i in order_idx])

    chosen: list[dict[str, object] | None] = [None] * len(positions)

    def backtrack(
        depth: int,
        used_cells: frozenset[tuple[int, int]],
        diag_map: dict[tuple[int, int], int],
    ) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        for plan in shuffled[idx]:
            if not _compatible(plan, used_cells, diag_map):
                continue
            next_cells = used_cells | plan["cells"]
            next_diag = dict(diag_map)
            for mid, slope in plan["diag_sigs"]:
                next_diag[mid] = slope
            chosen[idx] = plan
            if backtrack(depth + 1, next_cells, next_diag):
                return True
            chosen[idx] = None
        return False

    if not backtrack(0, frozenset(), {}):
        return None
    return [plan for plan in chosen if plan is not None]


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

    plans = _choose_plans(obstacles, bg, positions, rng)
    if plans is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, plan in zip(positions, block_colors, plans):
        gap = plan["gap"]
        target = plan["target"]
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), (gap,))
        tr, tc = target
        assert int(grid[tr, tc]) == bg, "target placement must stay off structures"
        grid[tr, tc] = int(color)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < block_count * MIN_BEAM_PIXELS_PER_RAY:
        return None

    for _pos, _color, plan in zip(positions, block_colors, plans):
        d = _gap_dir(plan["gap"])
        tail = int(plan["tail_after"])
        if tail < MIN_TAIL_AFTER_TARGET:
            return None
        end = plan["target"]
        nr, nc = end[0] + d[0], end[1] + d[1]
        for _ in range(tail):
            if not (0 <= nr < n and 0 <= nc < n) or int(grid[nr, nc]) != bg:
                return None
            nr += d[0]
            nc += d[1]

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D6 input")
