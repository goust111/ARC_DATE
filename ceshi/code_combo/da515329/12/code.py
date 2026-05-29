# Combo variant of da515329 (index 12)
# combo_spec: D6/3+D17/2
# Name: 反弹路径 + 镜面障碍物

import math
import random

import numpy as np


# Design analysis for the fused variant
# a) Behavior trace:
#    Original task: each teal arm tip turns into a clockwise orthogonal spiral with a 2/1
#    step cadence and a 1-cell spacing constraint. This produces rectangular rings.
#    Fused task: each tip instead launches a diagonal ray in the "outward + clockwise"
#    direction: N->NE, E->SE, S->SW, W->NW. A boundary acts like a mirror wall just
#    outside the grid; a grey obstacle cell acts like an internal mirror wall. Hitting a
#    diagonal obstacle cell flips both axes, while side-touching a mirror or a border
#    flips the touched axis only. Example: from the north tip a ray starts NE, bounces on
#    the right border to NW, and if a vertical grey bar sits one column further west it
#    flips horizontal direction again. The difference from the original is therefore
#    global: no spiral turning priority, no 2/1 stride logic, and no parity cleanup pass.
# b) Complexity calibration:
#    To keep the rule visible but still learnable, inputs use one teal cross plus only
#    2-3 grey mirror bars, all with the same mirror color. The bars are length-3 and
#    axis-aligned, so the solver sees repeated evidence for "adjacent mirror -> axis flip"
#    without a large obstacle vocabulary.
# c) Hidden assumptions audit:
#    The original main() assumed: only teal exists, the center can be recovered from teal
#    counts, paths must avoid orthogonal adjacency, and post-processing depends on center
#    parity and arm-length parity. In the fused task only the first assumption changes:
#    grey mirrors are present, but center recovery via teal row/column maxima still holds
#    because mirrors never use teal. Spiral-specific spacing and parity assumptions are no
#    longer valid and are removed entirely.
# d) Placement legality:
#    The generator places mirrors only on background cells, keeps them away from the cross
#    and from each other, and validates them by re-running the same tracer used by main().
# e) Visual observability:
#    Human-solvable cues are explicit: teal plus-sign source, grey mirror bars, and teal
#    zigzag rays that bounce at both borders and mirrors. The output differs measurably
#    from the no-mirror version before acceptance.
# f) Visual cleanliness:
#    The generator enforces small obstacle count, non-overlap, and a one-cell buffer
#    around the cross, which avoids noisy clutter near the source.
# g) New solver assumptions:
#    main() assumes exactly one teal plus-sign with four arms and grey cells meaning
#    mirrors. generate_input() guarantees that structure and also guarantees that at least
#    one ray bounces on a mirror and on a boundary, so neither fusion dimension degenerates.

_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIAG_FROM_OUTWARD = {
    (-1, 0): (-1, 1),
    (0, 1): (1, 1),
    (1, 0): (1, -1),
    (0, -1): (-1, -1),
}

CROSS_COLOR = 8
MIRROR_COLOR = 5

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
MIRROR_BARS_MIN, MIRROR_BARS_MAX = 2, 3
MAX_OUTER_RETRIES = 400
MAX_PLACEMENT_RETRIES = 80
FILL_LO, FILL_HI = 0.28, 0.58


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _build_cross(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _find_cross(grid: np.ndarray):
    teal = grid == CROSS_COLOR
    row_counts = np.sum(teal, axis=1)
    col_counts = np.sum(teal, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    arms = []
    for outward in _URDL:
        dr, dc = outward
        r, c = cr, cc
        cells = []
        while _inb(*grid.shape, r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((outward, cells))
    return cr, cc, arms


def _trace(grid: np.ndarray, capture: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    mirrors = g == MIRROR_COLOR
    _, _, arms = _find_cross(g)

    metrics = {
        "boundary_bounces": 0,
        "mirror_side_bounces": 0,
        "mirror_corner_bounces": 0,
        "loops": 0,
        "rays_with_mirror": 0,
        "rays_with_boundary": 0,
        "trace_steps": 0,
    }
    ray_infos = []
    step_guard = max(32, h * w * 8)

    for outward, cells in arms:
        cur = cells[-1]
        dr, dc = _DIAG_FROM_OUTWARD[outward]
        seen = set()
        touched_mirror = False
        touched_boundary = False
        states = []

        for _ in range(step_guard):
            state = (int(cur[0]), int(cur[1]), int(dr), int(dc))
            if state in seen:
                metrics["loops"] += 1
                break
            seen.add(state)
            metrics["trace_steps"] += 1

            nr, nc = cur[0] + dr, cur[1] + dc
            reflected = False
            if nr < 0 or nr >= h:
                dr = -dr
                metrics["boundary_bounces"] += 1
                touched_boundary = True
                reflected = True
            if nc < 0 or nc >= w:
                dc = -dc
                metrics["boundary_bounces"] += 1
                touched_boundary = True
                reflected = True
            if reflected:
                continue

            if mirrors[nr, nc]:
                dr = -dr
                dc = -dc
                metrics["mirror_corner_bounces"] += 1
                touched_mirror = True
                continue

            cur = (int(nr), int(nc))
            if out[cur] == 0:
                out[cur] = CROSS_COLOR
            if capture:
                states.append((int(cur[0]), int(cur[1]), int(dr), int(dc)))

            vr = cur[0] + dr
            if vr < 0 or vr >= h:
                dr = -dr
                metrics["boundary_bounces"] += 1
                touched_boundary = True
            elif mirrors[vr, cur[1]]:
                dr = -dr
                metrics["mirror_side_bounces"] += 1
                touched_mirror = True

            hc = cur[1] + dc
            if hc < 0 or hc >= w:
                dc = -dc
                metrics["boundary_bounces"] += 1
                touched_boundary = True
            elif mirrors[cur[0], hc]:
                dc = -dc
                metrics["mirror_side_bounces"] += 1
                touched_mirror = True

        if touched_mirror:
            metrics["rays_with_mirror"] += 1
        if touched_boundary:
            metrics["rays_with_boundary"] += 1
        ray_infos.append({"start": cells[-1], "states": states})

    metrics["mirror_bounces"] = metrics["mirror_side_bounces"] + metrics["mirror_corner_bounces"]
    metrics["teal_count"] = int(np.count_nonzero(out == CROSS_COLOR))
    return out, metrics, ray_infos


def _cross_cells(cr: int, cc: int, arm: int) -> set[tuple[int, int]]:
    cells = set()
    for k in range(1, arm + 1):
        cells.add((cr - k, cc))
        cells.add((cr + k, cc))
        cells.add((cr, cc - k))
        cells.add((cr, cc + k))
    return cells


def _candidate_bars(base_states, h: int, w: int):
    seen = set()
    candidates = []
    for ray_idx, info in enumerate(base_states):
        states = info["states"][:]
        random.shuffle(states)
        for r, c, dr, dc in states:
            # Horizontal mirror bar one row ahead: flips vertical motion.
            cells = [(r + dr, c - 1), (r + dr, c), (r + dr, c + 1)]
            if all(_inb(h, w, rr, cc) for rr, cc in cells):
                key = tuple(cells)
                if key not in seen:
                    seen.add(key)
                    candidates.append({"cells": cells, "ray_idx": ray_idx})

            # Vertical mirror bar one column ahead: flips horizontal motion.
            cells = [(r - 1, c + dc), (r, c + dc), (r + 1, c + dc)]
            if all(_inb(h, w, rr, cc) for rr, cc in cells):
                key = tuple(cells)
                if key not in seen:
                    seen.add(key)
                    candidates.append({"cells": cells, "ray_idx": ray_idx})
    random.shuffle(candidates)
    return candidates


def _can_place_bar(cells, forbidden: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        if not _inb(h, w, r, c):
            return False
        if (r, c) in forbidden:
            return False
    return True


def _build_forbidden(cross_cells: set[tuple[int, int]], mirror_cells: set[tuple[int, int]]):
    forbidden = set(cross_cells) | set(mirror_cells)
    for r, c in list(cross_cells) + list(mirror_cells):
        forbidden.add((r, c))
        for rr, cc in _neighbors8(r, c):
            forbidden.add((rr, cc))
    return forbidden


def _grid_without_mirrors(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    g[g == MIRROR_COLOR] = 0
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire four teal diagonal rays from the tips of a teal cross, letting borders and grey mirrors reflect them.

    Input structure:
    - Background is black (0).
    - A single teal (8) plus-sign with a center hole defines the source object.
    - Grey (5) short bars are mirror obstacles placed on background cells.

    Rule:
    - Each arm tip launches one ray along the diagonal formed by its outward direction plus a clockwise turn.
    - A boundary behaves like a mirror wall outside the grid: touching the top/bottom edge flips the vertical component, touching the left/right edge flips the horizontal component.
    - A diagonal hit into a grey mirror cell flips both components; side-touching a mirror after a move flips only the touched axis.
    - Rays keep painting visited non-mirror cells teal until their (position, direction) state repeats.

    Key insight:
    - The process is a billiard-style reflection system on a checkerboard: diagonal motion preserves cell parity, so the four rays together cover one parity class while mirror bars carve local zigzag deviations from the border-only pattern.
    """

    out, _, _ = _trace(np.asarray(grid, dtype=int), capture=False)
    return out


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    centers = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(centers)

    for cr, cc in centers:
        base = _build_cross(arm, size, cr, cc)
        _, base_metrics, ray_infos = _trace(base, capture=True)
        if base_metrics["trace_steps"] < size:
            continue

        cross = _cross_cells(cr, cc, arm)
        candidates = _candidate_bars(ray_infos, size, size)
        if len(candidates) < MIRROR_BARS_MIN:
            continue

        for _ in range(MAX_PLACEMENT_RETRIES):
            n_bars = random.randint(MIRROR_BARS_MIN, MIRROR_BARS_MAX)
            grid = base.copy()
            mirror_cells = set()
            used_rays = set()
            pool = candidates[:]
            random.shuffle(pool)

            for cand in pool:
                if len(used_rays) < MIRROR_BARS_MIN and cand["ray_idx"] in used_rays:
                    continue
                forbidden = _build_forbidden(cross, mirror_cells)
                if not _can_place_bar(cand["cells"], forbidden, size, size):
                    continue
                for r, c in cand["cells"]:
                    grid[r, c] = MIRROR_COLOR
                    mirror_cells.add((r, c))
                used_rays.add(cand["ray_idx"])
                if len(mirror_cells) >= 3 * n_bars:
                    break

            if len(mirror_cells) < 3 * MIRROR_BARS_MIN:
                continue

            out, metrics, _ = _trace(grid, capture=False)
            if metrics["mirror_bounces"] < 2:
                continue
            if metrics["boundary_bounces"] < 4:
                continue
            if metrics["rays_with_mirror"] < 2:
                continue
            if metrics["rays_with_boundary"] < 2:
                continue

            fill = float(np.mean(out == CROSS_COLOR))
            if not (FILL_LO <= fill <= FILL_HI):
                continue

            baseline, _, _ = _trace(_grid_without_mirrors(grid), capture=False)
            diff = int(np.count_nonzero((out == CROSS_COLOR) != (baseline == CROSS_COLOR)))
            if diff < max(6, arm * 2):
                continue

            if np.array_equal(main(grid), out):
                return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
