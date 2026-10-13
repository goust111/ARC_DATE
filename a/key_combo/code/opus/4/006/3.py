from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SEED_COLORS = (BLUE, GREEN, YELLOW)

Coord = tuple[int, int]
Region = tuple[int, int]


def _neighbors(r: int, c: int, connectivity: int) -> list[Coord]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    ]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[Coord] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _single_run(indices: np.ndarray) -> tuple[int, int]:
    vals = [int(x) for x in indices.tolist()]
    assert vals, "expected a nonempty gap"
    assert vals == list(range(vals[0], vals[-1] + 1)), "gap should be contiguous"
    return vals[0], vals[-1]


def _line_rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_counts = np.sum(grid == GREY, axis=1)
    col_counts = np.sum(grid == GREY, axis=0)
    line_rows = [int(r) for r in np.where(row_counts >= w - 3)[0]]
    line_cols = [int(c) for c in np.where(col_counts >= h - 3)[0]]
    assert len(line_rows) == 2 and len(line_cols) == 2, "expected two horizontal and two vertical dividers"
    return sorted(line_rows), sorted(line_cols)


def _bands_from_lines(h: int, w: int, line_rows: list[int], line_cols: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0, r1 = line_rows
    c0, c1 = line_cols
    row_bands = [(0, r0 - 1), (r0 + 1, r1 - 1), (r1 + 1, h - 1)]
    col_bands = [(0, c0 - 1), (c0 + 1, c1 - 1), (c1 + 1, w - 1)]
    assert all(a <= b for a, b in row_bands + col_bands), "all regions must be nonempty"
    return row_bands, col_bands


def _band_index(start: int, end: int, bands: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(bands):
        if lo <= start and end <= hi:
            return idx
    raise AssertionError("interval should lie inside exactly one region band")


def _region_of_point(r: int, c: int, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> Region:
    return _band_index(r, r, row_bands), _band_index(c, c, col_bands)


def _region_slice(region: Region, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> tuple[slice, slice]:
    rr, cc = region
    r0, r1 = row_bands[rr]
    c0, c1 = col_bands[cc]
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _chebyshev_distance(a: list[Coord], b: list[Coord]) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            best = min(best, max(abs(r0 - r1), abs(c0 - c1)))
    return best


def _is_diagonal_chain(comp: list[Coord]) -> bool:
    if len(comp) <= 1:
        return True
    ordered = sorted(comp)
    rows = [r for r, _ in ordered]
    cols = [c for _, c in ordered]
    drows = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    dcols = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    return all(step == 1 for step in drows) and (all(step == 1 for step in dcols) or all(step == -1 for step in dcols))


def _extract_gap_info(grid: np.ndarray) -> tuple[list[int], list[int], list[dict], Region, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    line_rows, line_cols = _line_rows_cols(g)
    row_bands, col_bands = _bands_from_lines(h, w, line_rows, line_cols)

    red_components = [sorted(comp) for comp in _components(g == RED, connectivity=8)]
    assert len(red_components) == 4, "expected four red diagonal noise chains"
    assert all(_is_diagonal_chain(comp) for comp in red_components), "red components should be diagonal chains"
    unused = set(range(len(red_components)))
    gap_infos: list[dict] = []

    for line_idx, r in enumerate(line_rows):
        c0, c1 = _single_run(np.where(g[r] != GREY)[0])
        gap_cells = [(r, c) for c in range(c0, c1 + 1)]
        seg = _band_index(c0, c1, col_bands)
        visible_len = len(gap_cells)
        matches = []
        for comp_idx in unused:
            comp = red_components[comp_idx]
            if len(comp) != visible_len:
                continue
            dist = _chebyshev_distance(gap_cells, comp)
            matches.append((dist, comp_idx))
        assert matches, "missing red chain for a horizontal gap"
        best_dist = min(dist for dist, _ in matches)
        best_ids = [idx for dist, idx in matches if dist == best_dist]
        assert best_dist == 1 and len(best_ids) == 1, "horizontal gap should have one adjacent matching chain"
        comp = red_components[best_ids[0]]
        unused.remove(best_ids[0])
        gap_infos.append(
            {
                "axis": "h",
                "gap_cells": gap_cells,
                "chain": comp,
                "edge": ((line_idx, seg), (line_idx + 1, seg)),
            }
        )

    for line_idx, c in enumerate(line_cols):
        r0, r1 = _single_run(np.where(g[:, c] != GREY)[0])
        gap_cells = [(r, c) for r in range(r0, r1 + 1)]
        seg = _band_index(r0, r1, row_bands)
        visible_len = len(gap_cells)
        matches = []
        for comp_idx in unused:
            comp = red_components[comp_idx]
            if len(comp) != visible_len:
                continue
            dist = _chebyshev_distance(gap_cells, comp)
            matches.append((dist, comp_idx))
        assert matches, "missing red chain for a vertical gap"
        best_dist = min(dist for dist, _ in matches)
        best_ids = [idx for dist, idx in matches if dist == best_dist]
        assert best_dist == 1 and len(best_ids) == 1, "vertical gap should have one adjacent matching chain"
        comp = red_components[best_ids[0]]
        unused.remove(best_ids[0])
        gap_infos.append(
            {
                "axis": "v",
                "gap_cells": gap_cells,
                "chain": comp,
                "edge": ((seg, line_idx), (seg, line_idx + 1)),
            }
        )

    assert not unused, "all red chains should be consumed by one gap each"
    seed_coords = np.argwhere(~np.isin(g, [BG, RED, GREY]))
    assert seed_coords.shape == (1, 2), "expected a unique seed pixel"
    seed_r, seed_c = map(int, seed_coords[0])
    seed_color = int(g[seed_r, seed_c])
    assert seed_color in SEED_COLORS, "unexpected seed color"
    seed_region = _region_of_point(seed_r, seed_c, row_bands, col_bands)
    return line_rows, line_cols, gap_infos, seed_region, seed_color


def _reachable_regions(start: Region, permeable_edges: list[tuple[Region, Region]]) -> set[Region]:
    adj = {(r, c): set() for r in range(3) for c in range(3)}
    for a, b in permeable_edges:
        adj[a].add(b)
        adj[b].add(a)
    reach = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in reach:
                reach.add(nxt)
                q.append(nxt)
    return reach


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the diagonal noise, repair the broken dividers, then fill only the regions reachable through repaired divider segments.

    Input structure:
    - The grid has black background, two almost-complete horizontal grey dividers, and two almost-complete vertical grey dividers that split the canvas into a 3x3 region layout.
    - Each grey divider has exactly one contiguous 1-3 cell gap, and each gap has one adjacent red diagonal chain whose length matches that missing segment.
    - Exactly one region contains a single seed pixel colored blue, green, or yellow.

    Rule:
    - Delete all red diagonal-noise pixels back to background.
    - Repair every grey divider gap with grey cells, using the nearby red chain to identify a repaired segment that becomes permeable.
    - Find the seed's region, fill all background cells in that region with the seed color, and then propagate to orthogonally adjacent regions only across shared divider segments that contain repaired pixels.
    - Keep all grey divider pixels grey in the output.

    Key insight:
    - The red diagonal noise is not decoration: it simultaneously marks which divider segment was broken, how long the repair must be, and which inter-region boundaries can later transmit the fill.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    line_rows, line_cols, gap_infos, seed_region, seed_color = _extract_gap_info(g)
    row_bands, col_bands = _bands_from_lines(h, w, line_rows, line_cols)
    reach = _reachable_regions(seed_region, [gap["edge"] for gap in gap_infos])

    out = g.copy()
    out[out == RED] = BG
    for gap in gap_infos:
        for r, c in gap["gap_cells"]:
            out[r, c] = GREY

    for region in reach:
        rs, cs = _region_slice(region, row_bands, col_bands)
        block = out[rs, cs]
        block[block == BG] = seed_color

    return out


def _bands_from_sizes(sizes: list[int]) -> tuple[list[int], list[tuple[int, int]], int]:
    a, b, c = sizes
    total = a + b + c + 2
    line0 = a
    line1 = a + b + 1
    bands = [(0, a - 1), (a + 1, a + b), (a + b + 2, total - 1)]
    return [line0, line1], bands, total


def _horizontal_candidates(line_idx: int, line_row: int, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> list[dict]:
    candidates: list[dict] = []
    for seg_idx, (c_lo, c_hi) in enumerate(col_bands):
        seg_len = c_hi - c_lo + 1
        max_len = min(3, seg_len)
        for length in range(1, max_len + 1):
            for c0 in range(c_lo, c_hi - length + 2):
                c1 = c0 + length - 1
                gap_cells = [(line_row, c) for c in range(c0, c1 + 1)]
                for side in (-1, 1):
                    for anchor_right in (False, True):
                        if side < 0:
                            rows_ok = row_bands[line_idx][1] - row_bands[line_idx][0] + 1 >= length
                            if not rows_ok:
                                continue
                            row_fn = lambda k: line_row - 1 - k
                        else:
                            rows_ok = row_bands[line_idx + 1][1] - row_bands[line_idx + 1][0] + 1 >= length
                            if not rows_ok:
                                continue
                            row_fn = lambda k: line_row + 1 + k
                        col_fn = (lambda k, c1=c1: c1 - k) if anchor_right else (lambda k, c0=c0: c0 + k)
                        chain = [(row_fn(k), col_fn(k)) for k in range(length)]
                        candidates.append(
                            {
                                "axis": "h",
                                "gap_cells": gap_cells,
                                "chain": chain,
                                "edge": ((line_idx, seg_idx), (line_idx + 1, seg_idx)),
                            }
                        )
    return candidates


def _vertical_candidates(line_idx: int, line_col: int, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> list[dict]:
    candidates: list[dict] = []
    for seg_idx, (r_lo, r_hi) in enumerate(row_bands):
        seg_len = r_hi - r_lo + 1
        max_len = min(3, seg_len)
        for length in range(1, max_len + 1):
            for r0 in range(r_lo, r_hi - length + 2):
                r1 = r0 + length - 1
                gap_cells = [(r, line_col) for r in range(r0, r1 + 1)]
                for side in (-1, 1):
                    width = col_bands[line_idx][1] - col_bands[line_idx][0] + 1 if side < 0 else col_bands[line_idx + 1][1] - col_bands[line_idx + 1][0] + 1
                    if width < length:
                        continue
                    for anchor_bottom in (False, True):
                        row_fn = (lambda k, r1=r1: r1 - k) if anchor_bottom else (lambda k, r0=r0: r0 + k)
                        col_fn = (lambda k, c=line_col: c - 1 - k) if side < 0 else (lambda k, c=line_col: c + 1 + k)
                        chain = [(row_fn(k), col_fn(k)) for k in range(length)]
                        candidates.append(
                            {
                                "axis": "v",
                                "gap_cells": gap_cells,
                                "chain": chain,
                                "edge": ((seg_idx, line_idx), (seg_idx, line_idx + 1)),
                            }
                        )
    return candidates


def _can_place_chain(chain: list[Coord], occupied_red: set[Coord], h: int, w: int) -> bool:
    for r, c in chain:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied_red:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied_red:
                    return False
    return True


def _build_input(spec: dict) -> np.ndarray:
    h = spec["height"]
    w = spec["width"]
    grid = np.full((h, w), BG, dtype=int)
    for r in spec["line_rows"]:
        grid[r, :] = GREY
    for c in spec["line_cols"]:
        grid[:, c] = GREY
    for gap in spec["gaps"]:
        for r, c in gap["gap_cells"]:
            grid[r, c] = BG
        for r, c in gap["chain"]:
            grid[r, c] = RED
    sr, sc = spec["seed_pos"]
    grid[sr, sc] = spec["seed_color"]
    return grid


def _build_output(spec: dict) -> np.ndarray:
    out = _build_input(spec)
    out[out == RED] = BG
    for gap in spec["gaps"]:
        for r, c in gap["gap_cells"]:
            out[r, c] = GREY
    reach = _reachable_regions(spec["seed_region"], [gap["edge"] for gap in spec["gaps"]])
    for region in reach:
        rs, cs = _region_slice(region, spec["row_bands"], spec["col_bands"])
        block = out[rs, cs]
        block[block == BG] = spec["seed_color"]
    return out


def _seed_candidates(region: Region, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], red_cells: set[Coord]) -> list[Coord]:
    rs, cs = _region_slice(region, row_bands, col_bands)
    cells = [(r, c) for r in range(rs.start, rs.stop) for c in range(cs.start, cs.stop) if (r, c) not in red_cells]
    preferred = []
    for r, c in cells:
        if all((nr, nc) not in red_cells for nr in range(r - 1, r + 2) for nc in range(c - 1, c + 2)):
            preferred.append((r, c))
    return preferred or cells


def _graph_components(edges: list[tuple[Region, Region]]) -> list[set[Region]]:
    unseen = {(r, c) for r in range(3) for c in range(3)}
    comps: list[set[Region]] = []
    while unseen:
        start = next(iter(unseen))
        comp = _reachable_regions(start, edges)
        comps.append(comp)
        unseen -= comp
    return comps


def _sample_spec() -> dict | None:
    row_sizes = [random.randint(3, 7) for _ in range(3)]
    col_sizes = [random.randint(3, 7) for _ in range(3)]
    line_rows, row_bands, h = _bands_from_sizes(row_sizes)
    line_cols, col_bands, w = _bands_from_sizes(col_sizes)

    gap_choices = [
        ("h", 0, _horizontal_candidates(0, line_rows[0], row_bands, col_bands)),
        ("h", 1, _horizontal_candidates(1, line_rows[1], row_bands, col_bands)),
        ("v", 0, _vertical_candidates(0, line_cols[0], row_bands, col_bands)),
        ("v", 1, _vertical_candidates(1, line_cols[1], row_bands, col_bands)),
    ]
    random.shuffle(gap_choices)

    occupied_red: set[Coord] = set()
    chosen: dict[tuple[str, int], dict] = {}
    for axis, idx, candidates in gap_choices:
        random.shuffle(candidates)
        placed = None
        for cand in candidates:
            if _can_place_chain(cand["chain"], occupied_red, h, w):
                placed = cand
                break
        if placed is None:
            return None
        chosen[(axis, idx)] = placed
        occupied_red.update(placed["chain"])

    gaps = [chosen[("h", 0)], chosen[("h", 1)], chosen[("v", 0)], chosen[("v", 1)]]
    edges = [gap["edge"] for gap in gaps]
    comps = _graph_components(edges)
    big_comps = [comp for comp in comps if len(comp) > 1]
    if big_comps and random.random() < 0.7:
        seed_region = random.choice(sorted(random.choice(big_comps)))
    else:
        seed_region = random.choice(sorted(random.choice(comps)))
    seed_cells = _seed_candidates(seed_region, row_bands, col_bands, occupied_red)
    if not seed_cells:
        return None
    seed_pos = random.choice(seed_cells)
    spec = {
        "height": h,
        "width": w,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "gaps": gaps,
        "seed_region": seed_region,
        "seed_pos": seed_pos,
        "seed_color": random.choice(SEED_COLORS),
    }
    return spec


def _validate_spec(spec: dict) -> bool:
    inp = _build_input(spec)
    out = _build_output(spec)
    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return False

    row_counts = np.sum(inp == GREY, axis=1)
    col_counts = np.sum(inp == GREY, axis=0)
    if list(np.where(row_counts >= inp.shape[1] - 3)[0]) != spec["line_rows"]:
        return False
    if list(np.where(col_counts >= inp.shape[0] - 3)[0]) != spec["line_cols"]:
        return False

    red_components = _components(inp == RED, connectivity=8)
    if len(red_components) != 4 or not all(_is_diagonal_chain(comp) for comp in red_components):
        return False

    edges = [gap["edge"] for gap in spec["gaps"]]
    reach = _reachable_regions(spec["seed_region"], edges)
    if len(reach) == 1 and random.random() < 0.6:
        return False
    seed_only = _build_input(spec)
    seed_only[seed_only == RED] = BG
    for gap in spec["gaps"]:
        for r, c in gap["gap_cells"]:
            seed_only[r, c] = GREY
    rs, cs = _region_slice(spec["seed_region"], spec["row_bands"], spec["col_bands"])
    block = seed_only[rs, cs]
    block[block == BG] = spec["seed_color"]
    if len(reach) > 1 and np.array_equal(seed_only, out):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(400):
        spec = _sample_spec()
        if spec is None:
            continue
        if _validate_spec(spec):
            return _build_input(spec)
    raise RuntimeError("failed to generate a valid diagonal-noise divider puzzle")
