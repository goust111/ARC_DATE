# Combo variant of 28a6681f (index 34)
# combo_spec: D8/1+D9/2+D13/0
# Name: 多盆地+单向溢流+大网格

from __future__ import annotations

import random

import numpy as np

# Design analysis
# ---------------
# a) Concrete fused behavior and traced example:
#    - The source task uses local gravity-style settling on a 10x10 board with 2-3 adjacent basins.
#      Water can fall, take a unique diagonal slip, then crawl horizontally to the nearest drop.
#    - This combo keeps the same local motion vocabulary but changes three coupled parts together:
#      1. the board is larger (15x15 or 20x20),
#      2. the bottom structure is a chain of 4-5 small basins,
#      3. horizontal overflow is globally one-way, inferred from which side of the basin chain the
#         supported source water sits on.
#    - Concrete 20x20 trace:
#      * five basins span columns 3..16, with the source water packed in basin 2 on the left half;
#      * under the old bidirectional rule, once the source basin and its upstream neighbor connect,
#        later water can keep spilling back into earlier basins, so the far-right basin remains dry;
#      * under the new rule, a supported water cell may only walk horizontally toward drops farther
#        away from the source side, so the same volume reaches the final downstream basin;
#      * visible difference: the output shows a longer occupied cascade chain, often one extra basin
#        farther downstream than the old solver would reach.
# b) Complexity calibration:
#    - The generator uses only three hand-vetted geometry families. They vary grid size, basin count,
#      source height, and wall spacing, but they all keep one coherent visual story: a long basin
#      chain with the source clearly on one side.
#    - Basin widths stay small (1-2 cells) so 4-5 basins fit cleanly on 15x15 or 20x20 boards
#      without creating wide flat lakes that would weaken the overflow reading.
#    - Every accepted sample must occupy the final downstream basin and must differ from a
#      bidirectional reference solver, so the fusion does not collapse back to the original task.
# c) Hidden assumption review against the original solver:
#    - The original code silently assumes a fixed 10x10 board. That is replaced by explicit checks
#      for 15x15 or 20x20 square boards.
#    - The original horizontal step considers drops on both sides. That assumption is exactly what
#      D9/2 breaks: once the flow direction is inferred, opposite-direction drops are ignored.
#    - The original output validator also required every water cell to have non-background on the
#      left, below, and right. That no longer matches the fused rule: a stable surface cell may keep
#      air on its forbidden upstream side because reverse overflow is globally disabled.
#    - The original guide-color parsing over `g[:7]` is no longer reliable on larger boards and is
#      unnecessary here, because direction is read from visible source placement rather than from a
#      color identity hidden in the top rows.
#    - A new assumption is introduced: the source water bbox must sit strictly on one side of the
#      basin-chain center. `generate_input()` guarantees this, and `main()` asserts it when checking.
# d) Placement legality:
#    - Each template is built from explicit wall columns and wall heights, then validated. Walls are
#      single-column solids rooted in the floor, so there are no overlaps or out-of-bounds writes.
#    - Source water is always a 2-column supported block inside one basin on the source side. Every
#      water cell is supported from below by a wall or another water cell, and no cell starts with
#      background on both horizontal sides.
# e) Visual observability:
#    - A human sees three repeated cues directly in the pixels:
#      1. the board is larger than the original 10x10,
#      2. there are 4-5 basins rather than 2-3,
#      3. the occupied chain extends only away from the source side.
#    - The variant-gap filter compares the combo solver against a bidirectional baseline on the same
#      input, so accepted samples visibly demonstrate the directional restriction rather than merely
#      looking like oversized originals.
# f) Visual cleanliness:
#    - The whole lower structure is one compact comb-like chain, with large empty space above. That
#      keeps the scene readable even on 20x20 boards.
#    - Only water plus 2-3 solid colors appear. There are no auxiliary markers, no hidden numeric
#      ordering by color, and no overlapping semantic layers.
# g) Inferability of the change:
#    - Colors are decorative only; they are randomly remapped after geometry is fixed.
#    - Direction is inferred from source-side placement relative to the basin chain, which is a
#      fully visible geometric cue. The solver never branches on hidden palette order or on the
#      numeric value of a color.
# 9) New solver assumptions audited against the generator:
#    - Assumption: there is exactly one bottom-anchored basin chain. Guarantee: every template draws
#      one connected floor-plus-walls structure and no extra solids.
#    - Assumption: the basin chain contains 4 or 5 basins. Guarantee: templates hard-code those
#      counts, and `main()` verifies them from the wall columns.
#    - Assumption: the source sits on one side of the chain center so direction is inferable.
#      Guarantee: all templates place the water block in basin 2 or basin 3 from the source side.
#    - Assumption: bidirectional comparison stays unambiguous on accepted samples. Guarantee:
#      `generate_input()` rejects any candidate whose bidirectional reference asserts or whose gap is
#      too small.

BG = 0
WATER = 1
SOLID_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
SIZE_OPTIONS = (15, 20)
MAX_GENERATION_ATTEMPTS = 300

TEMPLATE_SPECS: tuple[dict[str, object], ...] = (
    {
        "size": 15,
        "left": 1,
        "widths": (1, 2, 1, 1),
        "heights": (5, 5, 5),
        "source_basin": 1,
        "source_height_options": (10, 11),
        "source_width": 2,
    },
    {
        "size": 20,
        "left": 2,
        "widths": (2, 2, 2, 2),
        "heights": (2, 2, 5),
        "source_basin": 1,
        "source_height_options": (15, 16),
        "source_width": 2,
    },
    {
        "size": 20,
        "left": 3,
        "widths": (2, 2, 2, 1, 1),
        "heights": (5, 5, 5, 5),
        "source_basin": 1,
        "source_height_options": (15,),
        "source_width": 2,
    },
)


def _solid_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((int(rr), int(cc)))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _bottom_anchored_mask(grid: np.ndarray) -> np.ndarray:
    solids = (grid != int(BG)) & (grid != int(WATER))
    mask = np.zeros_like(solids, dtype=bool)
    for comp in _solid_components(solids):
        if any(r == grid.shape[0] - 1 for r, _c in comp):
            for r, c in comp:
                mask[r, c] = True
    return mask


def _basin_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = _bottom_anchored_mask(grid)
    pts = np.argwhere(mask)
    assert pts.size, "grid must contain a bottom-anchored basin structure"
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return int(r0), int(c0), int(r1), int(c1)


def _wall_columns(grid: np.ndarray) -> list[int]:
    mask = _bottom_anchored_mask(grid)
    floor = grid.shape[0] - 1
    _r0, c0, _r1, c1 = _basin_bbox(grid)
    return [int(c) for c in range(int(c0), int(c1) + 1) if bool(np.any(mask[:floor, int(c)]))]


def _infer_flow_direction(grid: np.ndarray) -> int:
    waters = np.argwhere(grid == int(WATER))
    assert waters.size, "grid must contain source water"
    _r0, c0, _r1, c1 = _basin_bbox(grid)
    basin_center = (int(c0) + int(c1)) / 2.0
    water_center = (int(waters[:, 1].min()) + int(waters[:, 1].max())) / 2.0
    if abs(water_center - basin_center) < 1e-9:
        raise AssertionError("source water must sit on one side of the basin-chain center")
    return 1 if water_center < basin_center else -1


def _settle(g: np.ndarray, r: int, c: int, *, directional: bool, flow_dir: int) -> tuple[int, int]:
    h, w = g.shape
    rr = int(r)
    cc = int(c)
    for _ in range(h * w):
        if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
            rr += 1
            continue

        opts: list[tuple[int, int]] = []
        if (
            rr + 1 < h
            and cc - 1 >= 0
            and int(g[rr, cc - 1]) == int(BG)
            and int(g[rr + 1, cc - 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc - 1)))
        if (
            rr + 1 < h
            and cc + 1 < w
            and int(g[rr, cc + 1]) == int(BG)
            and int(g[rr + 1, cc + 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc + 1)))
        if opts:
            assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
            rr, cc = opts[0]
            continue

        if rr + 1 < h and int(g[rr + 1, cc]) != int(BG):
            left = int(cc)
            while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                left -= 1
            right = int(cc)
            while right + 1 < w and int(g[rr, right + 1]) == int(BG):
                right += 1

            drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
            if directional:
                drops = [int(x) for x in drops if int((int(x) - int(cc)) * int(flow_dir)) > 0]
            if not drops:
                break

            min_dist = min(abs(int(x - cc)) for x in drops)
            nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
            assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
            best = nearest[0]
            assert best != int(cc)
            step = -1 if best < int(cc) else 1
            nx = int(cc + step)
            assert int(g[rr, nx]) == int(BG)
            cc = nx
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")

    return int(rr), int(cc)


def _simulate(
    grid: np.ndarray,
    *,
    directional: bool,
    check_input: bool = True,
    check_output: bool = True,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    assert h == w and h in SIZE_OPTIONS, "grid must be 15x15 or 20x20"
    flow_dir = _infer_flow_direction(g) if directional else 0

    if check_input:
        walls = _wall_columns(g)
        basin_count = len(walls) - 1
        assert basin_count in (4, 5), "grid must contain 4 or 5 basins"
        assert np.any(g == int(WATER)), "input must contain water"
        _infer_flow_direction(g)
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    for _pass in range(h * w):
        moved = False
        pts = sorted(((int(r), int(c)) for r, c in np.argwhere(g == int(WATER))), reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = _settle(g, int(r), int(c), directional=directional, flow_dir=int(flow_dir))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG)

    return g


def _occupied_basin_count(grid: np.ndarray) -> int:
    walls = _wall_columns(grid)
    occupied = 0
    for left, right in zip(walls, walls[1:]):
        if np.any(grid[:, int(left) + 1 : int(right)] == int(WATER)):
            occupied += 1
    return int(occupied)


def _last_basin_occupied(grid: np.ndarray) -> bool:
    walls = _wall_columns(grid)
    assert len(walls) >= 2
    return bool(np.any(grid[:, int(walls[-2]) + 1 : int(walls[-1])] == int(WATER)))


def _variant_gap(grid: np.ndarray) -> int:
    combo_out = _simulate(grid, directional=True, check_input=True, check_output=True)
    ref_out = _simulate(grid, directional=False, check_input=True, check_output=True)
    return int(np.count_nonzero(combo_out != ref_out))


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate long-chain basin settling on a larger board, with overflow constrained to one direction.

    Input structure:
    - The grid is 15x15 or 20x20, with bg = 0 and water = 1.
    - Other colors form one bottom-anchored chain of 4-5 small basins spread across the lower part
      of the board.
    - Source water appears as a supported block on one side of that chain, which is the global cue
      for the allowed overflow direction.

    Rule:
    - Water still falls straight down, uses a unique open diagonal slip when forced, and on a
      supported row crawls step by step toward a reachable drop.
    - The difference is that horizontal overflow searches only in the downstream direction inferred
      from the source side: left-source samples may spill only to the right, and right-source samples
      may spill only to the left.
    - Repeat bottom-first passes until no water pixel moves; walls stay fixed and water count is
      conserved.

    Key insight: The local falling physics is mostly unchanged, but basin-to-basin transfer is no
    longer symmetric. One must read the source-side cue first, then trace a long one-way cascade
    across 4-5 basins on the enlarged grid.
    """
    return _simulate(
        grid,
        directional=True,
        check_input=check_input,
        check_output=check_output,
    )


def _build_template(spec: dict[str, object], rng: random.Random) -> np.ndarray:
    size = int(spec["size"])
    left = int(spec["left"])
    widths = tuple(int(x) for x in spec["widths"])
    heights = tuple(int(x) for x in spec["heights"])
    source_basin = int(spec["source_basin"])
    source_height = int(rng.choice(tuple(int(x) for x in spec["source_height_options"])))
    source_width = int(spec["source_width"])

    g = np.zeros((size, size), dtype=int)
    floor = size - 1
    wall_cols = [int(left)]
    x = int(left)
    for width in widths:
        x += int(width) + 1
        wall_cols.append(int(x))
    assert wall_cols[-1] < size - 1, "template basin chain must fit inside the board"

    outer_height = max(7, size // 2)
    all_heights = [int(outer_height), *heights, int(outer_height)]

    # Base / outer walls share one label; internal walls alternate between two others.
    g[floor, wall_cols[0] : wall_cols[-1] + 1] = 2
    for idx, (col, height) in enumerate(zip(wall_cols, all_heights)):
        if idx in (0, len(wall_cols) - 1):
            label = 2
        else:
            label = 3 if idx % 2 else 4
        g[max(0, floor - int(height) + 1) : floor + 1, int(col)] = int(label)

    source_left = int(wall_cols[source_basin] + 1)
    source_right = int(wall_cols[source_basin + 1])
    assert source_left + source_width <= source_right, "source block must fit inside its basin"
    g[max(0, floor - source_height) : floor, source_left : source_left + source_width] = int(WATER)
    return g


def _recolor_solids(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    labels = sorted(set(map(int, g.flatten())) - {int(BG), int(WATER)})
    palette = rng.sample(SOLID_POOL, len(labels))
    mapping = {label: color for label, color in zip(labels, palette)}
    for label, color in mapping.items():
        g[g == int(label)] = int(color)
    return g


def generate_input() -> np.ndarray:
    """
    Generate a large-board one-way overflow input whose combo behavior is visible.

    Strategy:
    - Sample one of a few hand-vetted 4- or 5-basin templates.
    - Recolor the static structure for visual variety.
    - Keep only cases where the combo solver reaches the final downstream basin and differs from a
      bidirectional reference solver on the same input.
    """
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        spec = rng.choice(TEMPLATE_SPECS)
        grid = _recolor_solids(_build_template(spec, rng), rng)
        try:
            combo_out = main(grid, check_input=True, check_output=True)
            gap = _variant_gap(grid)
        except AssertionError:
            continue

        basin_count = len(_wall_columns(grid)) - 1
        if basin_count not in (4, 5):
            continue
        if _occupied_basin_count(combo_out) != basin_count:
            continue
        if not _last_basin_occupied(combo_out):
            continue
        if gap < 4:
            continue
        assert int(np.count_nonzero(combo_out == int(WATER))) == int(np.count_nonzero(grid == int(WATER)))
        return grid

    raise AssertionError("failed to generate a directional multi-basin large-grid sample")
