import random
from typing import Iterable

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_nonblack(arr: np.ndarray) -> np.ndarray:
    rows, cols = np.where(arr != BLACK)
    assert rows.size > 0
    return arr[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _stack_pieces(pieces: list[np.ndarray], order: list[int]) -> np.ndarray:
    out_h = max(int(pieces[i].shape[0]) for i in order)
    out_w = max(int(pieces[i].shape[1]) for i in order)
    out = np.full((out_h, out_w), BLACK, dtype=int)
    for idx in order:
        piece = pieces[idx]
        h, w = piece.shape
        region = out[:h, :w]
        mask = piece != BLACK
        region[mask] = piece[mask]
    return out


def _parse_scene(grid: np.ndarray) -> tuple[dict, list[dict]]:
    comps = _components(grid != BLACK)
    assert comps

    panel_idx = None
    panel_key = None
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(sub != BLACK):
            key = (c0, -(r1 - r0 + 1) * (c1 - c0 + 1))
            if panel_key is None or key < panel_key:
                panel_key = key
                panel_idx = idx
    assert panel_idx is not None

    pr0, pc0, pr1, pc1 = _bbox(comps[panel_idx])
    panel = {
        "bbox": (pr0, pc0, pr1, pc1),
        "sub": grid[pr0 : pr1 + 1, pc0 : pc1 + 1],
    }

    molds: list[dict] = []
    for idx, comp in enumerate(comps):
        if idx == panel_idx:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape[0] < 3 or sub.shape[1] < 3:
            continue
        if not (
            np.all(sub[0, :] == GRAY)
            and np.all(sub[-1, :] == GRAY)
            and np.all(sub[:, 0] == GRAY)
            and np.all(sub[:, -1] == GRAY)
        ):
            continue
        inner = sub[1:-1, 1:-1]
        hole = inner == BLACK
        marker = (inner != BLACK) & (inner != GRAY)
        if int(marker.sum()) != 1 or int(hole.sum()) == 0:
            continue
        marker_color = int(inner[marker][0])
        molds.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_h": int(inner.shape[0]),
                "inner_w": int(inner.shape[1]),
                "hole": hole,
                "marker_color": marker_color,
            }
        )

    assert molds
    return panel, molds


def _parse_bands(panel: dict) -> list[dict]:
    sub = panel["sub"]
    r0 = panel["bbox"][0]
    is_sep = np.all(sub == GRAY, axis=1)
    bands: list[dict] = []
    start = 0
    while start < sub.shape[0]:
        while start < sub.shape[0] and is_sep[start]:
            start += 1
        if start >= sub.shape[0]:
            break
        end = start
        while end < sub.shape[0] and not is_sep[end]:
            end += 1
        band_rows = sub[start:end]
        assert np.all((band_rows != BLACK) & (band_rows != GRAY))
        bands.append(
            {
                "top": r0 + start,
                "height": int(end - start),
                "rows": band_rows,
            }
        )
        start = end
    assert bands
    heights = [band["height"] for band in bands]
    assert len(set(heights)) == len(heights)
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp colored cutouts from left-side stripe bands using right-side molds, then
    stack the cutouts in the order encoded by marker-color first appearances.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panel, molds = _parse_scene(g)
    bands = _parse_bands(panel)
    band_by_height = {band["height"]: band for band in bands}
    assert len(band_by_height) == len(bands)

    panel_rows = panel["sub"]
    pieces: list[np.ndarray] = []
    first_rows: list[int] = []

    for mold in molds:
        inner_h = mold["inner_h"]
        inner_w = mold["inner_w"]
        assert inner_h in band_by_height
        band = band_by_height[inner_h]["rows"]
        assert band.shape[1] >= inner_w

        hole = mold["hole"]
        source = band[:, :inner_w]
        piece = np.full((inner_h, inner_w), BLACK, dtype=int)
        piece[hole] = source[hole]
        piece = _crop_nonblack(piece)
        pieces.append(piece)

        marker_color = mold["marker_color"]
        rows = np.where(np.any(panel_rows == marker_color, axis=1))[0]
        assert rows.size > 0
        first_rows.append(int(panel["bbox"][0] + rows[0]))

    assert len(set(first_rows)) == len(first_rows)
    order = [idx for idx, _ in sorted(enumerate(first_rows), key=lambda item: item[1])]
    return _stack_pieces(pieces, order)


def _random_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    area = h * w
    if area < 4:
        return None
    target = rng.randint(max(3, area // 4), max(3, area - 1))
    seed = (rng.randrange(h), rng.randrange(w))
    cells = {seed}
    attempts = 0
    while len(cells) < target and attempts < area * 20:
        attempts += 1
        cr, cc = rng.choice(tuple(cells))
        nbrs = []
        for dr, dc in DIR4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if nbrs:
            cells.add(rng.choice(nbrs))
    if len(cells) < 3 or len(cells) >= area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    rows, cols = np.where(mask)
    if len(set(rows.tolist())) < 2 or len(set(cols.tolist())) < 2:
        return None
    return mask


def _build_band_rows(
    rng: random.Random,
    height: int,
    width: int,
    filler_pool: list[int],
    hosted_markers: dict[int, int],
) -> np.ndarray:
    filler_count = 2 if len(filler_pool) == 2 else rng.choice((2, 3))
    fillers = rng.sample(filler_pool, filler_count)
    cycle = fillers[:]
    rng.shuffle(cycle)

    rows = np.empty((height, width), dtype=int)
    for r in range(height):
        rows[r, :] = cycle[(r + rng.randrange(len(cycle))) % len(cycle)]
    for r in range(1, height):
        if rows[r, 0] == rows[r - 1, 0]:
            rows[r, :] = cycle[(cycle.index(int(rows[r, 0])) + 1) % len(cycle)]

    for row_idx, marker_color in hosted_markers.items():
        rows[row_idx, :] = marker_color

    return rows


def _column_partition(rng: random.Random, sizes: list[tuple[int, int]]) -> list[list[int]]:
    n = len(sizes)
    cols = 1
    if n >= 3:
        cols = 2
    elif n == 2 and rng.random() < 0.55:
        cols = 2

    order = list(range(n))
    rng.shuffle(order)
    groups = [[] for _ in range(cols)]
    heights = [0] * cols
    for idx in order:
        col = min(range(cols), key=lambda k: heights[k])
        groups[col].append(idx)
        heights[col] += sizes[idx][0] + 1
    return groups


def _compose_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    count = rng.randint(2, 4)
    height_pool = list(range(3, 8))
    heights = rng.sample(height_pool, count)
    marker_colors = rng.sample(PALETTE, count)
    filler_pool = [color for color in PALETTE if color not in marker_colors]

    inner_ws = [rng.randint(4, 8) for _ in range(count)]
    left_w = max(inner_ws) + rng.randint(2, 5)

    band_specs = [
        {
            "height": heights[i],
            "source_marker": marker_colors[i],
            "inner_w": inner_ws[i],
        }
        for i in range(count)
    ]
    rng.shuffle(band_specs)

    band_host_slots: list[tuple[int, int]] = []
    for band_idx, spec in enumerate(band_specs):
        for row_idx in range(spec["height"]):
            band_host_slots.append((band_idx, row_idx))
    chosen_slots = rng.sample(band_host_slots, count)
    rng.shuffle(chosen_slots)

    host_row_by_marker: dict[int, tuple[int, int]] = {}
    for spec, slot in zip(band_specs, chosen_slots):
        host_row_by_marker[spec["source_marker"]] = slot

    band_arrays: list[np.ndarray] = []
    panel_rows: list[np.ndarray] = []
    first_row_by_marker: dict[int, int] = {}
    cursor = 0
    for band_idx, spec in enumerate(band_specs):
        hosted = {
            row_idx: marker
            for marker, (host_band_idx, row_idx) in host_row_by_marker.items()
            if host_band_idx == band_idx
        }
        band = _build_band_rows(rng, spec["height"], left_w, filler_pool, hosted)
        band_arrays.append(band)
        for row_idx in range(spec["height"]):
            color = int(band[row_idx, 0])
            if color in marker_colors and color not in first_row_by_marker:
                first_row_by_marker[color] = cursor + row_idx
        panel_rows.append(band)
        cursor += spec["height"]
        if band_idx != count - 1:
            panel_rows.append(np.full((1, left_w), GRAY, dtype=int))
            cursor += 1

    panel = np.vstack(panel_rows)
    if set(first_row_by_marker) != set(marker_colors):
        return None

    mold_arrays: list[np.ndarray] = []
    pieces: list[np.ndarray] = []
    for spec, band in zip(band_specs, band_arrays):
        h = spec["height"]
        w = spec["inner_w"]
        hole = _random_connected_mask(rng, h, w)
        if hole is None:
            return None

        mold = np.full((h + 2, w + 2), GRAY, dtype=int)
        mold[1:-1, 1:-1][hole] = BLACK
        candidates = np.argwhere(~hole)
        if candidates.size == 0:
            return None
        mr, mc = candidates[rng.randrange(len(candidates))]
        mold[1 + mr, 1 + mc] = spec["source_marker"]
        mold_arrays.append(mold)

        piece = np.full((h, w), BLACK, dtype=int)
        source = band[:, :w]
        piece[hole] = source[hole]
        pieces.append(_crop_nonblack(piece))

    order = [
        idx
        for idx, _ in sorted(
            enumerate([first_row_by_marker[spec["source_marker"]] for spec in band_specs]),
            key=lambda item: item[1],
        )
    ]
    expected = _stack_pieces(pieces, order)
    reverse = _stack_pieces(pieces, list(reversed(order)))
    if np.array_equal(expected, reverse):
        return None

    sizes = [(int(arr.shape[0]), int(arr.shape[1])) for arr in mold_arrays]
    groups = _column_partition(rng, sizes)
    col_gap = rng.randint(2, 4)
    row_gap = 1

    col_widths: list[int] = []
    col_heights: list[int] = []
    for group in groups:
        width = max(sizes[idx][1] for idx in group)
        height = sum(sizes[idx][0] for idx in group) + row_gap * (len(group) - 1)
        col_widths.append(width)
        col_heights.append(height)

    panel_h = int(panel.shape[0])
    right_w = sum(col_widths) + col_gap * (len(groups) - 1)
    right_h = max(col_heights)
    grid_h = max(panel_h, right_h) + rng.randint(0, 3)
    panel_top = rng.randint(0, grid_h - panel_h)

    sep_w = rng.randint(2, 4)
    margin_r = rng.randint(1, 3)
    grid_w = left_w + sep_w + right_w + margin_r
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[panel_top : panel_top + panel_h, :left_w] = panel

    slack = grid_w - (left_w + sep_w + right_w)
    col_x = left_w + sep_w + (0 if slack == 0 else rng.randint(0, slack))
    for col_idx, group in enumerate(groups):
        if not group:
            continue
        y = rng.randint(0, grid_h - col_heights[col_idx])
        column = group[:]
        rng.shuffle(column)
        for local_idx, mold_idx in enumerate(column):
            mold = mold_arrays[mold_idx]
            mh, mw = mold.shape
            grid[y : y + mh, col_x : col_x + mw] = mold
            y += mh
            if local_idx != len(column) - 1:
                y += row_gap
        col_x += col_widths[col_idx] + col_gap

    main_out = main(grid)
    if not np.array_equal(main_out, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _compose_case(rng)
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} retries")
