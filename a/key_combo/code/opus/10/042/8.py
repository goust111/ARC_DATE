from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
HINGE = 2
ROD = 3
ANCHOR = 7

MAX_GENERATION_ATTEMPTS = 500
LOCAL_PLACEMENT_ATTEMPTS = 250
SIDE_PAIRS_BY_LENGTH = {
    4: ((1, 2), (2, 1)),
    5: ((1, 3), (3, 1)),
    6: ((1, 4), (4, 1), (2, 3), (3, 2)),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _separator_intervals(lines: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(lines), length]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]


def _compressed_intervals(lines: list[int], length: int) -> tuple[list[tuple[int, int]], list[int]]:
    raw_intervals = _separator_intervals(lines, length)
    spans: list[tuple[int, int]] = []
    sizes: list[int] = []
    cursor = 0
    for start, end in raw_intervals:
        size = end - start
        sizes.append(size)
        spans.append((cursor, cursor + size))
        cursor += size
    return spans, sizes


def _axis_block_index(coord: float, spans: list[tuple[int, int]]) -> int:
    for idx, (start, end) in enumerate(spans):
        if start <= coord < end:
            return idx
    raise ValueError(f"coordinate {coord} fell on no compressed block interval")


def _block_distance(coord: float, anchor_coord: float, spans: list[tuple[int, int]]) -> int:
    return abs(_axis_block_index(coord, spans) - _axis_block_index(anchor_coord, spans))


def _compress_grid(grid: np.ndarray):
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BG)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BG)]
    keep_rows = [r for r in range(grid.shape[0]) if r not in set(row_lines)]
    keep_cols = [c for c in range(grid.shape[1]) if c not in set(col_lines)]
    row_map = {r: i for i, r in enumerate(keep_rows)}
    col_map = {c: i for i, c in enumerate(keep_cols)}
    compressed = grid[np.ix_(keep_rows, keep_cols)]
    row_spans, row_sizes = _compressed_intervals(row_lines, grid.shape[0])
    col_spans, col_sizes = _compressed_intervals(col_lines, grid.shape[1])
    return compressed, keep_rows, keep_cols, row_map, col_map, row_spans, col_spans, row_sizes, col_sizes


def _anchor_info(compressed: np.ndarray) -> tuple[list[tuple[int, int]], float, float]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(compressed == ANCHOR)]
    assert len(coords) == 4, "expected a unique 2x2 anchor"
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    assert len(rows) == 2 and len(cols) == 2
    assert rows[1] == rows[0] + 1 and cols[1] == cols[0] + 1
    for r in rows:
        for c in cols:
            assert compressed[r, c] == ANCHOR
    return coords, (rows[0] + rows[1]) / 2.0, (cols[0] + cols[1]) / 2.0


def _rod_records(grid: np.ndarray) -> list[dict]:
    compressed, keep_rows, keep_cols, row_map, col_map, row_spans, col_spans, row_sizes, col_sizes = _compress_grid(grid)
    _anchor_cells, anchor_r, anchor_c = _anchor_info(compressed)
    narrow_axis = 1 if sum(col_sizes) / len(col_sizes) < sum(row_sizes) / len(row_sizes) else 0

    mask = np.isin(compressed, [HINGE, ROD])
    comps = _components(mask)
    rods: list[dict] = []
    for comp in comps:
        hinge_cells = [p for p in comp if compressed[p] == HINGE]
        rod_cells = [p for p in comp if compressed[p] == ROD]
        assert len(hinge_cells) == 1 and len(rod_cells) >= 3
        hinge = hinge_cells[0]
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        assert len(rows) == 1 or len(cols) == 1
        if len(rows) == 1:
            orientation = "h"
            ordered = sorted(comp, key=lambda p: p[1])
            axis_mid = (ordered[0][1] + ordered[-1][1]) / 2.0
            mid_r = float(rows[0])
            mid_c = axis_mid
        else:
            orientation = "v"
            ordered = sorted(comp, key=lambda p: p[0])
            axis_mid = (ordered[0][0] + ordered[-1][0]) / 2.0
            mid_r = axis_mid
            mid_c = float(cols[0])

        hinge_index = ordered.index(hinge)
        left_side = ordered[:hinge_index]
        right_side = ordered[hinge_index + 1 :]
        assert left_side and right_side and len(left_side) != len(right_side)

        dh = _block_distance(mid_c, anchor_c, col_spans)
        dv = _block_distance(mid_r, anchor_r, row_spans)
        narrow_dist = dh if narrow_axis == 1 else dv
        open_dist = dv if narrow_axis == 1 else dh
        selected = dh > 0 and dv > 0 and narrow_dist <= open_dist

        if len(left_side) > len(right_side):
            moving = left_side
            fixed = [hinge, *right_side]
        else:
            moving = right_side
            fixed = [*left_side, hinge]

        rods.append(
            {
                "orientation": orientation,
                "ordered": ordered,
                "hinge": hinge,
                "moving": moving,
                "fixed": fixed,
                "selected": selected,
                "dh": dh,
                "dv": dv,
                "anchor_center": (anchor_r, anchor_c),
                "compressed_shape": compressed.shape,
                "keep_rows": keep_rows,
                "keep_cols": keep_cols,
            }
        )
    return rods


def _fold_targets(rod: dict) -> list[tuple[int, int]]:
    hinge_r, hinge_c = rod["hinge"]
    h, w = rod["compressed_shape"]
    tail_len = len(rod["moving"])
    if rod["orientation"] == "h":
        up_space = hinge_r
        down_space = h - 1 - hinge_r
        step = (-1, 0) if up_space > down_space else (1, 0)
        assert up_space != down_space
    else:
        left_space = hinge_c
        right_space = w - 1 - hinge_c
        step = (0, -1) if left_space > right_space else (0, 1)
        assert left_space != right_space
    return [(hinge_r + step[0] * k, hinge_c + step[1] * k) for k in range(1, tail_len + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress away the full-black separator lines, select rods inside the anchor-centered hourglass, and fold only those rods at their red hinge.

    Input structure:
    - The background is black, with full-black separator rows and columns that partition the scene into 4x4 macro-blocks.
    - One unique orange 2x2 anchor and several straight green rods appear on the non-separator cells.
    - Each rod is a single 4-connected horizontal or vertical segment containing exactly one red hinge marker.

    Rule:
    - Remove every full-black row and column to obtain compressed coordinates, while remembering how to map back to the original grid.
    - In compressed coordinates, locate the anchor center and compute each rod midpoint's horizontal and vertical block distances from it.
    - The separator-denser axis is the hourglass narrow axis; a rod is selected exactly when both block distances are nonzero and the narrow-axis distance is no greater than the opening-axis distance.
    - For each selected rod, keep the hinge and shorter side fixed, rotate the longer side 90 degrees toward the perpendicular side with more empty space to the compressed-grid edge, then restore the original canvas size with separator lines unchanged.

    Key insight:
    - The full-black separators play two roles at once: they define the compressed measurement frame and they determine which distance axis is the hourglass bottleneck, so the fold targets can only be identified after coordinate compression.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rods = _rod_records(g)
    if not rods:
        return out

    keep_rows = rods[0]["keep_rows"]
    keep_cols = rods[0]["keep_cols"]
    clear_raw: list[tuple[int, int]] = []
    paint_raw: list[tuple[int, int]] = []

    for rod in rods:
        if not rod["selected"]:
            continue
        clear_raw.extend((keep_rows[r], keep_cols[c]) for r, c in rod["moving"])
        paint_raw.extend((keep_rows[r], keep_cols[c]) for r, c in _fold_targets(rod))

    for r, c in clear_raw:
        out[r, c] = BG
    for r, c in paint_raw:
        out[r, c] = ROD
    return out


def _partitions(total: int) -> list[tuple[int, int, int, int]]:
    out = []
    for a in (3, 4):
        for b in (3, 4):
            for c in (3, 4):
                for d in (3, 4):
                    if a + b + c + d == total:
                        out.append((a, b, c, d))
    return out


ROW_PARTITIONS = _partitions(13) + _partitions(14) + _partitions(15)
COL_PARTITIONS = _partitions(12) + _partitions(13)


def _build_axis_map(sizes: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    comp_to_raw: list[int] = []
    separators: list[int] = []
    raw = 0
    comp = 0
    for idx, size in enumerate(sizes):
        for _ in range(size):
            comp_to_raw.append(raw)
            raw += 1
            comp += 1
        if idx != len(sizes) - 1:
            separators.append(raw)
            raw += 1
    return comp_to_raw, separators


def _block_spans_from_sizes(sizes: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for size in sizes:
        spans.append((cursor, cursor + size))
        cursor += size
    return spans


def _expand_4(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _sample_anchor(block_rows: tuple[int, int, int, int], block_cols: tuple[int, int, int, int]):
    row_spans = _block_spans_from_sizes(block_rows)
    col_spans = _block_spans_from_sizes(block_cols)
    br = random.choice((1, 2))
    bc = random.choice((1, 2))
    rs, re = row_spans[br]
    cs, ce = col_spans[bc]
    top = random.randint(rs, re - 2)
    left = random.randint(cs, ce - 2)
    cells = {(top, left), (top + 1, left), (top, left + 1), (top + 1, left + 1)}
    return {"block": (br, bc), "top": top, "left": left, "cells": cells}


def _render_raw(comp: np.ndarray, row_map: list[int], col_map: list[int], raw_shape: tuple[int, int]) -> np.ndarray:
    raw = np.zeros(raw_shape, dtype=int)
    for r in range(comp.shape[0]):
        for c in range(comp.shape[1]):
            raw[row_map[r], col_map[c]] = comp[r, c]
    return raw


def _rod_midpoint(orientation: str, cells: list[tuple[int, int]]) -> tuple[float, float]:
    if orientation == "h":
        ordered = sorted(cells, key=lambda p: p[1])
        return float(ordered[0][0]), (ordered[0][1] + ordered[-1][1]) / 2.0
    ordered = sorted(cells, key=lambda p: p[0])
    return (ordered[0][0] + ordered[-1][0]) / 2.0, float(ordered[0][1])


def _rod_fold_dir(orientation: str, hinge: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int] | None:
    r, c = hinge
    h, w = shape
    if orientation == "h":
        up_space, down_space = r, h - 1 - r
        if up_space == down_space:
            return None
        return (-1, 0) if up_space > down_space else (1, 0)
    left_space, right_space = c, w - 1 - c
    if left_space == right_space:
        return None
    return (0, -1) if left_space > right_space else (0, 1)


def _candidate_rods(
    shape: tuple[int, int],
    anchor: dict,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    want_orientation: str,
    prefer_rows: set[int],
    prefer_cols: set[int],
):
    h, w = shape
    anchor_center = (anchor["top"] + 0.5, anchor["left"] + 0.5)
    target_rows = list(prefer_rows) if prefer_rows else list(range(h))
    target_cols = list(prefer_cols) if prefer_cols else list(range(w))
    random.shuffle(target_rows)
    random.shuffle(target_cols)

    candidates = []
    for _ in range(LOCAL_PLACEMENT_ATTEMPTS):
        length = random.randint(4, 6)
        side_a, side_b = random.choice(SIDE_PAIRS_BY_LENGTH[length])

        if want_orientation == "h":
            row = random.choice(target_rows)
            hinge_c = random.randint(side_a, w - 1 - side_b)
            hinge = (row, hinge_c)
            cells = [(row, c) for c in range(hinge_c - side_a, hinge_c + side_b + 1)]
        else:
            col = random.choice(target_cols)
            hinge_r = random.randint(side_a, h - 1 - side_b)
            hinge = (hinge_r, col)
            cells = [(r, col) for r in range(hinge_r - side_a, hinge_r + side_b + 1)]

        cell_set = set(cells)
        if cell_set & anchor["cells"]:
            continue
        if want_orientation == "h" and prefer_cols and not ({c for _, c in cells} & prefer_cols):
            continue
        if want_orientation == "v" and prefer_rows and not ({r for r, _ in cells} & prefer_rows):
            continue
        midpoint_r, midpoint_c = _rod_midpoint(want_orientation, cells)
        try:
            dh = _block_distance(midpoint_c, anchor_center[1], col_spans)
            dv = _block_distance(midpoint_r, anchor_center[0], row_spans)
        except ValueError:
            continue
        selected = dh > 0 and dv > 0 and dh <= dv

        fold_dir = _rod_fold_dir(want_orientation, hinge, shape)
        if selected and fold_dir is None:
            continue
        if selected:
            moving_left = side_a > side_b
            tail_len = max(side_a, side_b)
            moving = (
                [(hinge[0], c) for c in range(hinge[1] - side_a, hinge[1])]
                if moving_left and want_orientation == "h"
                else [(hinge[0], c) for c in range(hinge[1] + 1, hinge[1] + side_b + 1)]
                if want_orientation == "h"
                else [(r, hinge[1]) for r in range(hinge[0] - side_a, hinge[0])]
                if moving_left
                else [(r, hinge[1]) for r in range(hinge[0] + 1, hinge[0] + side_b + 1)]
            )
            folded = [(hinge[0] + fold_dir[0] * k, hinge[1] + fold_dir[1] * k) for k in range(1, tail_len + 1)]
            if any(not (0 <= r < h and 0 <= c < w) for r, c in folded):
                continue
        else:
            moving = []
            folded = []

        candidates.append(
            {
                "orientation": want_orientation,
                "hinge": hinge,
                "cells": cells,
                "cell_set": cell_set,
                "side_a": side_a,
                "side_b": side_b,
                "selected": selected,
                "dh": dh,
                "dv": dv,
                "moving": moving,
                "folded": folded,
                "midpoint": (midpoint_r, midpoint_c),
            }
        )
    random.shuffle(candidates)
    return candidates


def _raw_midpoint(midpoint: tuple[float, float], row_map: list[int], col_map: list[int]) -> tuple[float, float]:
    r, c = midpoint
    if abs(r - round(r)) < 1e-9:
        rr = row_map[int(round(r))]
    else:
        lo = int(np.floor(r))
        hi = int(np.ceil(r))
        rr = (row_map[lo] + row_map[hi]) / 2.0
    if abs(c - round(c)) < 1e-9:
        cc = col_map[int(round(c))]
    else:
        lo = int(np.floor(c))
        hi = int(np.ceil(c))
        cc = (col_map[lo] + col_map[hi]) / 2.0
    return float(rr), float(cc)


def _place_rods():
    row_sizes = random.choice(ROW_PARTITIONS)
    col_sizes = random.choice(COL_PARTITIONS)
    if sum(row_sizes) <= sum(col_sizes):
        return None
    row_spans = _block_spans_from_sizes(row_sizes)
    col_spans = _block_spans_from_sizes(col_sizes)
    h, w = sum(row_sizes), sum(col_sizes)
    row_map, row_lines = _build_axis_map(row_sizes)
    col_map, col_lines = _build_axis_map(col_sizes)
    anchor = _sample_anchor(row_sizes, col_sizes)

    total_rods = random.choice((5, 6))
    selected_target = random.choice((2, 3))
    if total_rods - selected_target not in (2, 3):
        total_rods = 6
    h_count = random.choice((2, 3, 4))
    v_count = total_rods - h_count
    if h_count <= 0 or v_count <= 0:
        return None
    orientations = ["h"] * h_count + ["v"] * v_count
    random.shuffle(orientations)

    rods = []
    occupied = set(anchor["cells"])
    forbidden = _expand_4(occupied, (h, w))
    reserved_fold: set[tuple[int, int]] = set()
    covered_rows = {r for r, _ in anchor["cells"]}
    covered_cols = {c for _, c in anchor["cells"]}

    for idx, orientation in enumerate(orientations):
        remaining = len(orientations) - idx - 1
        current_selected = sum(1 for rod in rods if rod["selected"])
        min_possible = current_selected
        max_possible = current_selected + remaining + 1
        prefer_rows = set(range(h)) - covered_rows
        prefer_cols = set(range(w)) - covered_cols
        candidates = _candidate_rods((h, w), anchor, row_spans, col_spans, orientation, prefer_rows, prefer_cols)
        placed = False
        for cand in candidates:
            next_selected = current_selected + int(cand["selected"])
            if next_selected > selected_target:
                continue
            if next_selected + remaining < selected_target:
                continue
            if cand["cell_set"] & forbidden:
                continue
            if cand["selected"]:
                folded = set(cand["folded"])
                if folded & occupied:
                    continue
                if folded & reserved_fold:
                    continue
            rods.append(cand)
            occupied |= cand["cell_set"]
            forbidden = _expand_4(occupied, (h, w))
            if cand["selected"]:
                reserved_fold |= set(cand["folded"])
            covered_rows |= {r for r, _ in cand["cells"]}
            covered_cols |= {c for _, c in cand["cells"]}
            placed = True
            break
        if not placed:
            return None

    if sum(1 for rod in rods if rod["selected"]) != selected_target:
        return None
    if not ({rod["orientation"] for rod in rods} == {"h", "v"}):
        return None
    if covered_rows != set(range(h)) or covered_cols != set(range(w)):
        return None

    comp = np.zeros((h, w), dtype=int)
    for r, c in anchor["cells"]:
        comp[r, c] = ANCHOR
    for rod in rods:
        for r, c in rod["cells"]:
            comp[r, c] = ROD
        hr, hc = rod["hinge"]
        comp[hr, hc] = HINGE

    raw_shape = (h + len(row_lines), w + len(col_lines))
    raw_in = _render_raw(comp, row_map, col_map, raw_shape)

    anchor_mid_raw = (
        (row_map[anchor["top"]] + row_map[anchor["top"] + 1]) / 2.0,
        (col_map[anchor["left"]] + col_map[anchor["left"] + 1]) / 2.0,
    )
    naive_diff = 0
    for rod in rods:
        raw_mid = _raw_midpoint(rod["midpoint"], row_map, col_map)
        raw_dx = abs(raw_mid[1] - anchor_mid_raw[1])
        raw_dy = abs(raw_mid[0] - anchor_mid_raw[0])
        naive = raw_dx > 0 and raw_dy > 0 and raw_dx <= raw_dy
        naive_diff += int(naive != rod["selected"])
    if naive_diff == 0:
        return None

    expected = comp.copy()
    for rod in rods:
        if not rod["selected"]:
            continue
        for r, c in rod["moving"]:
            expected[r, c] = BG
        for r, c in rod["folded"]:
            expected[r, c] = ROD
    raw_out = _render_raw(expected, row_map, col_map, raw_shape)

    if not np.array_equal(main(raw_in), raw_out):
        return None
    if len([r for r in range(raw_in.shape[0]) if np.all(raw_in[r, :] == 0)]) != 3:
        return None
    if len([c for c in range(raw_in.shape[1]) if np.all(raw_in[:, c] == 0)]) != 3:
        return None

    return raw_in


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _place_rods()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
