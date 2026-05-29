# Combo variant of 28a6681f (index 21)
# combo_spec: D1/0+D14/0
# Name: 悬空水像素+纯垂直下落

from __future__ import annotations

import random

import numpy as np


"""
variant_design:
  concrete_visualization:
    grid: fixed 10x10, matching the original task family
    structures:
      - 2-3 non-blue colors still build bottom-anchored walls, steps, and neighboring basin-like grooves
      - terrain is encoded as a skyline, so each column has a visible landing height
    water:
      - blue pixels appear as scattered suspended droplets in open air
      - droplets are placed in several columns, often with repeated counts, so the solver must track more than one ray

  behavior_trace:
    small_example:
      surface: [10, 8, 5, 5, 8, 9, 6, 8, 7, 10]
      source: one droplet above column 3, plus extra droplets above columns 5 and 7
      original_task:
        - the column-3 droplet would fall to row 4, notice the lower basin on its right, and slide diagonally away
        - further sideways motion could continue toward a deeper neighbor
      combo_variant:
        - the same droplet is allowed to start unsupported in mid-air
        - after falling straight down, it must stop in column 3 as soon as the cell below is blocked
      visible_difference:
        - deeper neighboring basins no longer attract off-column water
        - the output keeps each source column's water count in the same column

  complexity_calibration:
    keep_it_nontrivial:
      - every sample uses 3-5 rain columns rather than a single drop
      - at least one source column is a ridge edge where old diagonal flow would have been possible
      - some columns receive 2-3 droplets so vertical stacking is visible
    keep_it_readable:
      - terrain comes from a small template family with repeated motifs
      - empty edge columns and sparse upper-air droplets prevent clutter

  implicit_assumptions_review:
    broken_old_assumptions:
      - original input validation required every water cell to be supported; this is intentionally removed
      - original settle() relied on diagonal and horizontal branches; those branches are invalid under pure vertical fall
      - original output stability required left/below/right support; sideways openings are now legal and expected
      - original guide-color scan is removed because upper nonzero pixels are no longer special guides
    new_assumptions:
      - motion is completely column-local, so water counts are conserved per column
      - each column segment between solids can be solved by vertical compression only
    generator_guarantees:
      - all placed droplets are genuinely suspended, with background directly below in the input
      - at least one final droplet sits at a cell that would have had a unique diagonal escape in the original task

  placement_legality:
    - structure templates are prevalidated 10x10 skylines and never leave the grid
    - water is placed only on background cells strictly above the local surface
    - every-other-row source slots avoid overlap and keep droplets visibly separated
    - generate_input() always reruns main() and checks conservation / suspension / ridge-edge exposure

  visual_observability:
    - the input looks like rain above basins, not like pre-settled water stuck to walls
    - the output packs blue pixels directly downward in the same columns
    - repeated ridge-edge examples make the "no sideways spill" change inferable from 2-4 samples

  neatness_and_inferability:
    - only color 1 moves; structure colors stay fixed and few in number
    - no hidden ordering depends on color ids, only on visible columns and visible support
    - terrain, air, and water occupy separate regions, keeping the grids clean
"""


BG = 0
WATER = 1
PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)


TEMPLATES: tuple[dict[str, tuple[int, ...]], ...] = (
    {
        "surface": (10, 8, 5, 5, 8, 9, 6, 8, 7, 10),
        "rain_cols": (1, 2, 3, 4, 5, 6, 7, 8),
        "pedestals": (2, 3, 6),
    },
    {
        "surface": (10, 7, 9, 6, 5, 5, 8, 9, 7, 10),
        "rain_cols": (1, 2, 3, 4, 5, 6, 7, 8),
        "pedestals": (4, 5),
    },
    {
        "surface": (10, 6, 8, 5, 5, 7, 9, 8, 6, 10),
        "rain_cols": (1, 2, 3, 4, 5, 6, 7, 8),
        "pedestals": (3, 4),
    },
    {
        "surface": (10, 8, 6, 9, 5, 5, 8, 7, 9, 10),
        "rain_cols": (1, 2, 3, 4, 5, 6, 7, 8),
        "pedestals": (4, 5, 7),
    },
)


def _source_rows(surface_top: int) -> list[int]:
    return [int(r) for r in range(0, max(0, int(surface_top) - 1), 2)]


def _column_water_counts(grid: np.ndarray) -> tuple[int, ...]:
    return tuple(int(np.count_nonzero(grid[:, c] == int(WATER))) for c in range(grid.shape[1]))


def _floating_water_count(grid: np.ndarray) -> int:
    h, _w = grid.shape
    count = 0
    for r, c in np.argwhere(grid == int(WATER)):
        rr = int(r)
        cc = int(c)
        if rr + 1 < h and int(grid[rr + 1, cc]) == int(BG):
            count += 1
    return int(count)


def _total_fall_distance(inp: np.ndarray, out: np.ndarray) -> int:
    total = 0
    for c in range(inp.shape[1]):
        src_rows = [int(r) for r in np.flatnonzero(inp[:, int(c)] == int(WATER))]
        dst_rows = [int(r) for r in np.flatnonzero(out[:, int(c)] == int(WATER))]
        assert len(src_rows) == len(dst_rows), "column water count must be conserved"
        total += sum(int(dr - sr) for sr, dr in zip(src_rows, dst_rows))
    return int(total)


def _slip_exposure_count(grid: np.ndarray) -> int:
    h, w = grid.shape
    exposed = 0
    for r, c in np.argwhere(grid == int(WATER)):
        rr = int(r)
        cc = int(c)
        if rr + 1 >= h or int(grid[rr + 1, cc]) == int(BG):
            continue
        options = 0
        if (
            cc - 1 >= 0
            and rr + 1 < h
            and int(grid[rr, cc - 1]) == int(BG)
            and int(grid[rr + 1, cc - 1]) == int(BG)
        ):
            options += 1
        if (
            cc + 1 < w
            and rr + 1 < h
            and int(grid[rr, cc + 1]) == int(BG)
            and int(grid[rr + 1, cc + 1]) == int(BG)
        ):
            options += 1
        if options == 1:
            exposed += 1
    return int(exposed)


def _paint_surface(grid: np.ndarray, surface: tuple[int, ...], rng: random.Random) -> None:
    colors = list(rng.sample(PALETTE, 3))
    if rng.random() < 0.5:
        colors[2] = colors[1]

    zones = (
        (0, 3, colors[0]),
        (3, 7, colors[1]),
        (7, 10, colors[2]),
    )
    for c, top in enumerate(surface):
        if int(top) >= grid.shape[0]:
            continue
        fill_color = int(colors[1])
        for left, right, color in zones:
            if left <= c < right:
                fill_color = int(color)
                break
        grid[int(top) :, int(c)] = int(fill_color)


def _mirror_template(template: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    surface = tuple(int(x) for x in reversed(template["surface"]))
    rain_cols = tuple(sorted(9 - int(c) for c in template["rain_cols"]))
    pedestals = tuple(sorted(9 - int(c) for c in template["pedestals"]))
    return {"surface": surface, "rain_cols": rain_cols, "pedestals": pedestals}


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Let suspended blue droplets fall straight down and stack within the same columns above the static terrain.

    Input structure:
    - bg = 0. Water pixels are color 1 and may start unsupported anywhere in open air.
    - Other non-zero colors are fixed bottom-anchored walls, steps, and basin-like terrain.
    - The input may contain several rain columns; there is no guide object and no side-attached source requirement.

    Rule:
    - Water only moves vertically downward through background cells.
    - When a droplet meets the grid bottom, a solid, or previously settled water in its own column, it stops immediately.
    - Equivalently, within each column segment between solids, all water compresses to the lowest available cells of that segment.

    Key insight: The fusion removes both sideways flow and support constraints, so solving the task becomes a per-column rain compression problem rather than a basin-search simulation.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    if check_input:
        assert g.shape == (10, 10), "grid must stay 10x10"

    initial_water = int(np.count_nonzero(g == int(WATER)))

    for c in range(w):
        col = g[:, int(c)].copy()
        solid_rows = [int(r) for r in range(h) if int(col[r]) not in (int(BG), int(WATER))]
        prev = -1
        for solid_r in solid_rows + [h]:
            lo = int(prev + 1)
            hi = int(solid_r - 1)
            if lo <= hi:
                segment = col[lo : hi + 1].copy()
                water_count = int(np.count_nonzero(segment == int(WATER)))
                segment[:] = int(BG)
                if water_count:
                    segment[-water_count:] = int(WATER)
                col[lo : hi + 1] = segment
            prev = int(solid_r)
        g[:, int(c)] = col

    if check_output:
        assert int(np.count_nonzero(g == int(WATER))) == initial_water, "water count must be conserved"
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "output water must be vertically supported"

    return g


def _build_case(rng: random.Random) -> np.ndarray | None:
    template = rng.choice(TEMPLATES)
    if rng.random() < 0.5:
        template = _mirror_template(template)

    surface = tuple(int(x) for x in template["surface"])
    rain_cols = [int(c) for c in template["rain_cols"] if _source_rows(int(surface[int(c)]))]
    pedestals = [int(c) for c in template["pedestals"] if _source_rows(int(surface[int(c)]))]
    if len(rain_cols) < 4 or not pedestals:
        return None

    grid = np.zeros((10, 10), dtype=int)
    _paint_surface(grid, surface, rng)

    forced_col = int(rng.choice(pedestals))
    other_cols = [int(c) for c in rain_cols if int(c) != int(forced_col)]
    if len(other_cols) < 3:
        return None

    total_cols = rng.randint(4, min(5, len(rain_cols)))
    chosen_cols = [int(forced_col)] + rng.sample(other_cols, int(total_cols - 1))

    counts: dict[int, int] = {int(forced_col): 1}
    for col in chosen_cols[1:]:
        capacity = min(3, len(_source_rows(int(surface[int(col)]))))
        counts[int(col)] = int(rng.randint(1, int(capacity)))

    while sum(counts.values()) < 5:
        expandable = [
            int(c)
            for c in chosen_cols[1:]
            if counts[int(c)] < min(3, len(_source_rows(int(surface[int(c)]))))
        ]
        if not expandable:
            break
        pick = int(rng.choice(expandable))
        counts[pick] += 1

    while sum(counts.values()) > 9:
        reducible = [int(c) for c in chosen_cols[1:] if counts[int(c)] > 1]
        if not reducible:
            break
        pick = int(rng.choice(reducible))
        counts[pick] -= 1

    for col in chosen_cols:
        rows = _source_rows(int(surface[int(col)]))
        qty = int(counts[int(col)])
        if len(rows) < qty:
            return None
        selected_rows = sorted(rng.sample(rows, qty))
        for row in selected_rows:
            if int(grid[int(row), int(col)]) != int(BG):
                return None
            grid[int(row), int(col)] = int(WATER)

    out = main(grid)

    if int(np.count_nonzero(grid == int(WATER))) < 5:
        return None
    if _floating_water_count(grid) != int(np.count_nonzero(grid == int(WATER))):
        return None
    if _column_water_counts(grid) != _column_water_counts(out):
        return None
    if _slip_exposure_count(out) < 1:
        return None
    if _total_fall_distance(grid, out) < int(np.count_nonzero(grid == int(WATER))) + 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        case = _build_case(rng)
        if case is not None:
            return case
    raise RuntimeError("generate_input: failed after 200 attempts")
