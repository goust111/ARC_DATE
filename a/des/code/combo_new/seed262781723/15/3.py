from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BORDER = 5
MARKER = 9
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8)

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PLACE_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _boundary_mask(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros_like(filled, dtype=bool)
    for r, c in np.argwhere(filled):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not filled[nr, nc]:
                boundary[rr, cc] = True
                break
    return boundary


def _extract_palette(top_row: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    c = 0
    w = top_row.shape[0]
    while c < w:
        if int(top_row[c]) == bg:
            c += 1
            continue
        color = int(top_row[c])
        start = c
        while c < w and int(top_row[c]) == color:
            c += 1
        assert c - start == 2, "palette blocks must be 2 cells wide"
        colors.append(color)
    assert len(colors) >= 2, "expected a palette row with at least two colors"
    assert len(set(colors)) == len(colors), "palette colors must be distinct"
    return colors


def _role_colors(grid: np.ndarray, palette: list[int], bg: int) -> tuple[int, int]:
    body = grid[1:]
    remaining = [int(v) for v in np.unique(body) if int(v) != bg and int(v) not in palette]
    assert len(remaining) == 2, "expected exactly one border color and one marker color below the palette"

    marker_candidates = []
    for color in remaining:
        comps = _components(body == color)
        if comps and all(len(comp) == 1 for comp in comps):
            marker_candidates.append(int(color))
    assert len(marker_candidates) == 1, "expected a unique singleton-marker color"
    marker_color = int(marker_candidates[0])
    border_color = next(int(color) for color in remaining if int(color) != marker_color)
    return border_color, marker_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the palette color indexed by its marker count.

    Input structure:
    - The background is black, and the very top row is a horizontal palette made of
      distinct 2-cell color blocks separated by black gaps.
    - Below the palette, gray pixels form several disjoint 1-pixel-thick closed
      boundaries whose interiors are otherwise black.
    - Each interior contains one or more scattered maroon single-pixel markers.

    Rule:
    - Read the palette colors from left to right as positions 1, 2, 3, ...
    - For every closed gray boundary, count how many maroon markers lie inside it.
    - Fill only the black interior cells of that region with the Nth palette color,
      leaving the gray boundary, maroon markers, and palette row unchanged.

    Key insight:
    - The top row is an ordered legend, and the markers inside each cavity act as a
      count-based lookup key into that legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    palette = _extract_palette(g[0], bg)
    border_color, marker_color = _role_colors(g, palette, bg)

    body = g[1:]
    out = g.copy()
    enclosed_count = 0

    for comp in _components(body != border_color):
        if _touches_border(comp, body.shape[0], body.shape[1]):
            continue
        marker_count = sum(int(body[r, c]) == marker_color for r, c in comp)
        assert 1 <= marker_count <= len(palette), "marker count must index the palette"
        fill_color = int(palette[marker_count - 1])
        for r, c in comp:
            if int(body[r, c]) == bg:
                out[r + 1, c] = fill_color
        enclosed_count += 1

    assert enclosed_count > 0, "expected at least one enclosed region"
    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _choose_marker_cells(interior: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    cells = list(interior)
    for _ in range(80):
        random.shuffle(cells)
        picked: list[tuple[int, int]] = []
        for r, c in cells:
            if all(abs(r - pr) + abs(c - pc) != 1 for pr, pc in picked):
                picked.append((int(r), int(c)))
                if len(picked) == count:
                    return picked
    return None


def _notch_top(mask: np.ndarray) -> None:
    h, w = mask.shape
    depth = random.randint(1, h - 3)
    span = random.randint(1, w - 2)
    start = random.randint(0, w - span)
    mask[:depth, start : start + span] = False


def _notch_bottom(mask: np.ndarray) -> None:
    h, w = mask.shape
    depth = random.randint(1, h - 3)
    span = random.randint(1, w - 2)
    start = random.randint(0, w - span)
    mask[h - depth :, start : start + span] = False


def _notch_left(mask: np.ndarray) -> None:
    h, w = mask.shape
    depth = random.randint(1, w - 3)
    span = random.randint(1, h - 2)
    start = random.randint(0, h - span)
    mask[start : start + span, :depth] = False


def _notch_right(mask: np.ndarray) -> None:
    h, w = mask.shape
    depth = random.randint(1, w - 3)
    span = random.randint(1, h - 2)
    start = random.randint(0, h - span)
    mask[start : start + span, w - depth :] = False


def _make_region_shape(marker_count: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        min_side = 5 if marker_count <= 3 else 6
        h = random.randint(min_side, 9)
        w = random.randint(min_side, 10)

        filled = np.ones((h, w), dtype=bool)
        notch_ops = [_notch_top, _notch_bottom, _notch_left, _notch_right]
        random.shuffle(notch_ops)
        for op in notch_ops[: random.randint(0, 3)]:
            op(filled)

        if not filled.any():
            continue
        filled = _crop_mask(filled)
        if filled.shape[0] < 5 or filled.shape[1] < 5:
            continue
        if not _is_connected(filled):
            continue

        boundary = _boundary_mask(filled)
        interior_mask = filled & ~boundary
        interior_comps = _components(interior_mask)
        if len(interior_comps) != 1:
            continue

        interior = interior_comps[0]
        if len(interior) < marker_count + 1:
            continue

        markers = _choose_marker_cells(interior, marker_count)
        if markers is None:
            continue
        return boundary, interior, markers
    return None


def _paint_palette(grid: np.ndarray, palette: list[int], start_col: int) -> None:
    for idx, color in enumerate(palette):
        c0 = start_col + idx * 3
        grid[0, c0 : c0 + 2] = int(color)


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occ.shape[0], r0 + h + 1)
    cc1 = min(occ.shape[1], c0 + w + 1)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_region(
    grid: np.ndarray,
    occ: np.ndarray,
    boundary: np.ndarray,
    interior: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = boundary.shape
    max_r0 = grid.shape[0] - h - 1
    max_c0 = grid.shape[1] - w - 1
    if max_r0 < 2 or max_c0 < 1:
        return None

    for _ in range(PLACE_ATTEMPTS):
        r0 = random.randint(2, max_r0)
        c0 = random.randint(1, max_c0)
        if not _can_place(occ, r0, c0, h, w):
            continue

        for r, c in np.argwhere(boundary):
            grid[r0 + int(r), c0 + int(c)] = BORDER
        for r, c in markers:
            grid[r0 + int(r), c0 + int(c)] = MARKER

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(grid.shape[0], r0 + h + 1)
        cc1 = min(grid.shape[1], c0 + w + 1)
        occ[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _expected_output(
    inp: np.ndarray,
    regions: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int]],
    palette: list[int],
) -> np.ndarray:
    out = inp.copy()
    for r0, c0, interior, markers, count in regions:
        marker_set = set(markers)
        fill = int(palette[count - 1])
        for r, c in interior:
            if (r, c) not in marker_set:
                out[r0 + int(r), c0 + int(c)] = fill
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        palette_len = random.randint(3, 6)
        palette = random.sample(PALETTE_POOL, palette_len)
        palette_width = 3 * palette_len - 1

        region_count = random.randint(2, min(4, palette_len))
        counts = random.sample(range(1, palette_len + 1), region_count)

        min_h = max(12, 6 + region_count * 4)
        min_w = max(palette_width + 2, 8 + region_count * 4)
        if min_h > 30 or min_w > 30:
            continue

        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        grid = np.full((height, width), BG, dtype=int)

        palette_start = random.randint(0, width - palette_width)
        _paint_palette(grid, palette, palette_start)

        occ = np.zeros_like(grid, dtype=bool)
        occ[0, :] = True
        occ[1, :] = True

        regions: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int]] = []
        ok = True
        for count in sorted(counts, reverse=True):
            shape = _make_region_shape(count)
            if shape is None:
                ok = False
                break
            boundary, interior, markers = shape
            placed = _place_region(grid, occ, boundary, interior, markers)
            if placed is None:
                ok = False
                break
            r0, c0 = placed
            regions.append((r0, c0, interior, markers, int(count)))

        if not ok:
            continue

        expected = _expected_output(grid, regions, palette)
        if np.array_equal(grid, expected):
            continue
        if np.array_equal(main(grid), expected):
            return grid

    raise RuntimeError("failed to generate a valid input")
