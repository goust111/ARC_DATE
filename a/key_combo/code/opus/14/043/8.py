from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
WATER = 7

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = NEIGHBORS8 if conn8 else NEIGHBORS4
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_blue_bands(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    bands = _components(grid == BLUE, conn8=False)
    bands.sort(key=lambda comp: min(r for r, _ in comp))
    return bands


def _find_defect_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    defects: list[list[tuple[int, int]]] = []
    for comp in _components(grid == BG, conn8=True):
        if len(comp) > 2:
            continue
        comp_set = set(comp)
        ok = True
        for r, c in comp:
            blue_neighbors = 0
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                if (nr, nc) in comp_set:
                    continue
                if grid[nr, nc] != BLUE:
                    ok = False
                    break
                blue_neighbors += 1
            if not ok or blue_neighbors == 0:
                ok = False
                break
        if ok:
            defects.append(sorted(comp))
    defects.sort(key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp)))
    return defects


def _blue_run(grid: np.ndarray, row: int, col: int) -> tuple[int, int]:
    left = col
    while left - 1 >= 0 and grid[row, left - 1] == BLUE:
        left -= 1
    right = col
    while right + 1 < grid.shape[1] and grid[row, right + 1] == BLUE:
        right += 1
    return left, right


def _simulate(
    grid: np.ndarray,
    defect_components: list[list[tuple[int, int]]] | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    defects = defect_components if defect_components is not None else _find_defect_components(g)
    defect_cells = {cell for comp in defects for cell in comp}

    out = g.copy()
    launches = deque((r, c, 0) for comp in defects for r, c in comp)
    seen_launches: set[tuple[int, int]] = set()
    meta = {
        "spread_events": 0,
        "overflow_launches": 0,
        "merge_hits": 0,
        "max_spread_depth": -1,
        "vertical_cells": set(),
        "spread_cells": set(),
        "events": [],
    }

    while launches:
        r, c, depth = launches.popleft()
        if (r, c) in seen_launches:
            continue
        seen_launches.add((r, c))

        rr, cc = r, c
        while True:
            nr = rr + 1
            if nr >= h:
                break
            if g[nr, cc] == BLUE:
                left, right = _blue_run(g, nr, cc)
                for x in range(left, right + 1):
                    if g[rr, x] == BG and (rr, x) not in defect_cells:
                        out[rr, x] = WATER
                        meta["spread_cells"].add((rr, x))
                spill_points: list[tuple[int, int]] = []
                for oc in (left - 1, right + 1):
                    if 0 <= oc < w and g[rr, oc] == BG:
                        if (rr, oc) not in defect_cells:
                            out[rr, oc] = WATER
                            meta["spread_cells"].add((rr, oc))
                        spill_points.append((rr, oc))
                meta["spread_events"] += 1
                meta["overflow_launches"] += len(spill_points)
                meta["max_spread_depth"] = max(meta["max_spread_depth"], depth)
                meta["events"].append(
                    {
                        "spread_row": rr,
                        "band_top_row": nr,
                        "band_span": (left, right),
                        "depth": depth,
                        "spill_points": tuple(spill_points),
                    }
                )
                for sr, sc in spill_points:
                    launches.append((sr, sc, depth + 1))
                break

            rr = nr
            if out[rr, cc] == WATER:
                meta["merge_hits"] += 1
                break
            if (rr, cc) not in defect_cells:
                out[rr, cc] = WATER
                meta["vertical_cells"].add((rr, cc))

    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black defects inside blue bands into orange waterfall sources that cascade over lower blue bands.

    Input structure:
    - The grid uses black (0) background and 2-3 separated blue (1) horizontal band regions.
    - Each band contains 1-2 internal black defect holes of size 1-2 pixels, fully surrounded in 8-neighborhood by blue pixels.
    - Lower blue bands sit below upper defects with enough black space for downward flow and overflow.

    Rule:
    - Detect each black defect component of size at most 2 whose surrounding 8-neighbor cells, excluding the defect itself, are all blue.
    - From every defect pixel, let water fall downward through black cells; paint visited cells orange (7), but keep defect pixels black.
    - If falling water sees a blue cell directly below, it spreads horizontally across the full contiguous blue top span underneath it and paints the two one-cell overflow positions just beyond that span.
    - From each valid overflow position, water resumes vertical falling; if two flows meet, their already-painted continuation is shared.

    Key insight:
    - The same blue bands both define the hidden black defect sources and act as shelves whose top boundaries split those sources into cascading overflow branches.
    """
    return _simulate(np.asarray(grid, dtype=int))


def _choose_span(
    width: int,
    required_cols: list[int],
    min_margin: int,
    min_width: int,
    max_width: int,
) -> tuple[int, int] | None:
    req = sorted(required_cols)
    needed = 0 if not req else req[-1] - req[0] + 3
    lo_w = max(min_width, needed)
    hi_w = max_width
    if lo_w > hi_w:
        return None
    band_w = random.randint(lo_w, hi_w)

    if not req:
        left_lo = min_margin
        left_hi = width - min_margin - band_w
    else:
        left_lo = max(min_margin, req[-1] - band_w + 2)
        left_hi = min(req[0] - 1, width - min_margin - band_w)
    if left_lo > left_hi:
        return None
    left = random.randint(left_lo, left_hi)
    return left, left + band_w - 1


def _pick_rows(num_bands: int) -> tuple[list[int], list[int], int] | None:
    tops = [random.randint(1, 2)]
    heights = [random.randint(3, 4) for _ in range(num_bands)]
    for i in range(1, num_bands):
        tops.append(tops[-1] + heights[i - 1] + random.randint(3, 5))
    bottom_margin = random.randint(4, 6)
    h = tops[-1] + heights[-1] + bottom_margin
    if h > 30:
        return None
    return tops, heights, h


def _paint_band(grid: np.ndarray, top: int, height: int, left: int, right: int) -> None:
    grid[top : top + height, left : right + 1] = BLUE


def _add_defect_component(
    grid: np.ndarray,
    top: int,
    height: int,
    left: int,
    right: int,
    used_cells: set[tuple[int, int]],
    used_cols: set[int],
    avoid_cols: set[int],
    forced_col: int | None = None,
    prefer_size2: bool = False,
) -> list[tuple[int, int]] | None:
    interior_rows = list(range(top + 1, top + height - 1))
    if not interior_rows:
        return None

    candidates: list[list[tuple[int, int]]] = []
    if forced_col is not None:
        cols = [forced_col]
    else:
        cols = list(range(left + 1, right))
        random.shuffle(cols)

    for row in interior_rows:
        for col in cols:
            if col in avoid_cols:
                continue
            single = [(row, col)]
            if (row, col) not in used_cells and col not in used_cols:
                candidates.append(single)
            if col + 1 < right and col not in avoid_cols and col + 1 not in avoid_cols:
                pair = [(row, col), (row, col + 1)]
                if all(cell not in used_cells for cell in pair) and all(x not in used_cols for _, x in pair):
                    candidates.append(pair)

    if not candidates:
        return None
    if prefer_size2:
        candidates.sort(key=lambda comp: (len(comp) != 2, comp[0][0], comp[0][1]))
    else:
        random.shuffle(candidates)

    for comp in candidates:
        ok = True
        comp_set = set(comp)
        for r, c in comp:
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if (nr, nc) in comp_set:
                    continue
                if not _in_bounds(grid.shape, nr, nc) or grid[nr, nc] != BLUE:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for r, c in comp:
            grid[r, c] = BG
            used_cells.add((r, c))
            used_cols.add(c)
        return comp
    return None


def _band_index_for_comp(
    comp: list[tuple[int, int]],
    band_boxes: list[tuple[int, int, int, int]],
) -> int | None:
    for idx, (r0, c0, r1, c1) in enumerate(band_boxes):
        if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in comp):
            return idx
    return None


def _vertical_only_output(grid: np.ndarray, defects: list[list[tuple[int, int]]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    defect_cells = {cell for comp in defects for cell in comp}
    h, _ = out.shape
    for comp in defects:
        for r, c in comp:
            rr = r
            while rr + 1 < h and grid[rr + 1, c] == BG:
                rr += 1
                if (rr, c) not in defect_cells:
                    out[rr, c] = WATER
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    bands = _find_blue_bands(grid)
    if len(bands) not in (2, 3):
        return False

    defects = _find_defect_components(grid)
    defect_cells = {cell for comp in defects for cell in comp}

    band_boxes = [_bbox(band) for band in bands]
    for band, box in zip(bands, band_boxes):
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 3:
            return False
        if c1 - c0 + 1 < max(10, grid.shape[1] // 2):
            return False
        allowed = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if grid[r, c] == BLUE or (r, c) in defect_cells
        }
        if allowed != set(band) | (defect_cells & {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}):
            return False
    if not defects:
        return False

    per_band = [0] * len(bands)
    for comp in defects:
        idx = _band_index_for_comp(comp, band_boxes)
        if idx is None:
            return False
        per_band[idx] += 1
    if any(count < 1 or count > 2 for count in per_band):
        return False

    out, meta = _simulate(grid, defects, return_meta=True)
    if np.array_equal(out, grid):
        return False
    if meta["spread_events"] < len(bands) - 1:
        return False
    if meta["overflow_launches"] < 2:
        return False
    if len(meta["vertical_cells"]) < 6 or len(meta["spread_cells"]) < 4:
        return False
    if np.count_nonzero(out == WATER) > grid.size // 3:
        return False

    naive = _vertical_only_output(grid, defects)
    diff = np.count_nonzero((out == WATER) & (naive != WATER))
    if diff < 4:
        return False
    if len(bands) == 3 and meta["max_spread_depth"] < 1:
        return False

    for comp in defects:
        muted = grid.copy()
        for r, c in comp:
            muted[r, c] = BLUE
        if np.array_equal(main(muted), out):
            return False

    if meta["events"]:
        top_hit = meta["events"][0]["band_top_row"]
        hit_comp = None
        for band in bands:
            if any(r == top_hit for r, _ in band):
                hit_comp = band
                break
        if hit_comp is None:
            return False
        altered = grid.copy()
        for r, c in hit_comp:
            altered[r, c] = BG
        if np.array_equal(main(altered), out):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(20, 28)
    num_bands = random.choice([2, 3])

    rows = _pick_rows(num_bands)
    if rows is None:
        return None
    tops, heights, height = rows

    min_band_w = max(11, int(round(width * 0.55)))
    max_band_w = min(width - 4, max(min_band_w, int(round(width * 0.82))))
    if min_band_w > max_band_w:
        return None

    if num_bands == 3:
        middle = _choose_span(width, [], 3, min_band_w, max_band_w)
        if middle is None:
            return None
        l2, r2 = middle
        spill_choices = []
        if l2 - 1 >= 2:
            spill_choices.append(l2 - 1)
        if r2 + 1 <= width - 3:
            spill_choices.append(r2 + 1)
        if not spill_choices:
            return None
        spill_target = random.choice(spill_choices)
        bottom = _choose_span(width, [spill_target], 2, min_band_w, max_band_w)
        if bottom is None:
            return None
        l3, r3 = bottom
        top_sources = random.sample(list(range(l2 + 2, r2 - 1)), random.choice([1, 2]))
        top_band = _choose_span(width, top_sources, 2, min_band_w, max_band_w)
        if top_band is None:
            return None
        l1, r1 = top_band
        spans = [(l1, r1), (l2, r2), (l3, r3)]
    else:
        bottom = _choose_span(width, [], 2, min_band_w, max_band_w)
        if bottom is None:
            return None
        l2, r2 = bottom
        top_sources = random.sample(list(range(l2 + 2, r2 - 1)), random.choice([1, 2]))
        top_band = _choose_span(width, top_sources, 2, min_band_w, max_band_w)
        if top_band is None:
            return None
        l1, r1 = top_band
        spans = [(l1, r1), (l2, r2)]

    if any(abs(a0 - b0) < 2 and abs(a1 - b1) < 2 for (a0, a1), (b0, b1) in zip(spans, spans[1:])):
        return None

    grid = np.zeros((height, width), dtype=int)
    for (top, band_h), (left, right) in zip(zip(tops, heights), spans):
        _paint_band(grid, top, band_h, left, right)

    band_used_cells: list[set[tuple[int, int]]] = [set() for _ in spans]
    band_used_cols: list[set[int]] = [set() for _ in spans]
    defect_components: list[list[tuple[int, int]]] = []

    incoming_cols: list[set[int]] = [set() for _ in spans]
    incoming_cols[0] = set()
    if len(spans) >= 2:
        incoming_cols[1] = set(top_sources)
    if len(spans) == 3:
        incoming_cols[2] = {spans[1][0] - 1, spans[1][1] + 1}

    for idx, ((top, band_h), (left, right)) in enumerate(zip(zip(tops, heights), spans)):
        avoid = set(incoming_cols[idx])
        forced_cols: list[int] = []
        if idx == 0:
            forced_cols = top_sources
        elif idx < len(spans) - 1:
            next_left, next_right = spans[idx + 1]
            choices = [c for c in range(next_left + 2, next_right - 1) if c not in avoid]
            if not choices:
                return None
            forced_cols = [random.choice(choices)]
        else:
            last_choices = [c for c in range(left + 2, right - 1) if c not in avoid]
            if not last_choices:
                return None
            forced_cols = [random.choice(last_choices)]

        for forced_col in forced_cols:
            comp = _add_defect_component(
                grid,
                top,
                band_h,
                left,
                right,
                band_used_cells[idx],
                band_used_cols[idx],
                avoid,
                forced_col=forced_col,
            )
            if comp is None:
                return None
            defect_components.append(comp)
            avoid.update(c for _, c in comp)

        need_extra = 1
        if (right - left + 1) >= 14 and random.random() < 0.55:
            need_extra = 2
        while len([comp for comp in defect_components if _band_index_for_comp(comp, [_bbox(b) for b in _find_blue_bands(grid)]) == idx]) < need_extra:
            comp = _add_defect_component(
                grid,
                top,
                band_h,
                left,
                right,
                band_used_cells[idx],
                band_used_cols[idx],
                avoid,
                forced_col=None,
                prefer_size2=random.random() < 0.35,
            )
            if comp is None:
                break
            defect_components.append(comp)
            avoid.update(c for _, c in comp)

    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
