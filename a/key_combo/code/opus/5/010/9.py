from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
ORANGE = 7
SOLID_COLORS = [1, 2, 3, 6, 8, 9]
PATTERN_COLORS = [1, 2, 3, 6, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(comp, dtype=int))
    return out


def _separator_indices(grid: np.ndarray) -> tuple[list[int], list[int]]:
    sep_rows = [
        r
        for r in range(grid.shape[0])
        if np.all(np.isin(grid[r, :], [GREY, YELLOW])) and np.count_nonzero(grid[r, :] == GREY) >= grid.shape[1] - 3
    ]
    sep_cols = [
        c
        for c in range(grid.shape[1])
        if np.all(np.isin(grid[:, c], [GREY, YELLOW])) and np.count_nonzero(grid[:, c] == GREY) >= grid.shape[0] - 3
    ]
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected a 3x3 separator lattice"
    return sep_rows, sep_cols


def _cell_slices(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows, sep_cols = _separator_indices(grid)
    return _bands(grid.shape[0], sep_rows), _bands(grid.shape[1], sep_cols)


def _solid_cell_color(cell: np.ndarray) -> int | None:
    mask = ~np.isin(cell, [GREY, YELLOW, ORANGE])
    vals = sorted(set(int(v) for v in cell[mask].flatten()))
    if len(vals) == 1:
        return int(vals[0])
    return None


def _segments_1d(indices: Iterable[int]) -> list[tuple[int, int]]:
    pts = [int(x) for x in indices]
    if not pts:
        return []
    segs: list[tuple[int, int]] = []
    start = prev = pts[0]
    for x in pts[1:]:
        if x == prev + 1:
            prev = x
        else:
            segs.append((start, prev))
            start = prev = x
    segs.append((start, prev))
    return segs


def _complete_solid_cell(cell: np.ndarray, base_color: int) -> np.ndarray:
    work = cell.copy()
    orange_mask = work == ORANGE
    spans: list[int] = []
    for comp in _components_mask(orange_mask, connectivity=8):
        r0, c0, r1, c1 = _bbox_coords(comp)
        spans.append(max(r1 - r0 + 1, c1 - c0 + 1))
    work[orange_mask] = base_color
    if spans:
        cap = max(spans)
        for r in range(work.shape[0]):
            cols = np.where(work[r, :] == YELLOW)[0]
            for (a0, a1), (b0, b1) in zip(_segments_1d(cols), _segments_1d(cols)[1:]):
                gap0 = a1 + 1
                gap1 = b0 - 1
                if gap1 >= gap0 and gap1 - gap0 + 1 <= cap:
                    work[r, gap0 : gap1 + 1] = YELLOW
        for c in range(work.shape[1]):
            rows = np.where(work[:, c] == YELLOW)[0]
            for (a0, a1), (b0, b1) in zip(_segments_1d(rows), _segments_1d(rows)[1:]):
                gap0 = a1 + 1
                gap1 = b0 - 1
                if gap1 >= gap0 and gap1 - gap0 + 1 <= cap:
                    work[gap0 : gap1 + 1, c] = YELLOW
    work[work == YELLOW] = base_color
    return work


def _periodic_candidates(cell: np.ndarray) -> list[tuple[int, np.ndarray]]:
    visible_mask = ~np.isin(cell, [YELLOW, ORANGE])
    visible = np.argwhere(visible_mask)
    candidates: list[tuple[int, np.ndarray]] = []
    for ph in (1, 2, 3):
        for pw in (1, 2, 3):
            table: dict[tuple[int, int], int] = {}
            ok = True
            for r, c in visible:
                key = (int(r % ph), int(c % pw))
                val = int(cell[r, c])
                prev = table.get(key)
                if prev is None:
                    table[key] = val
                elif prev != val:
                    ok = False
                    break
            if not ok:
                continue
            full = np.full(cell.shape, -1, dtype=int)
            for r in range(cell.shape[0]):
                for c in range(cell.shape[1]):
                    key = (r % ph, c % pw)
                    if key not in table:
                        ok = False
                        break
                    full[r, c] = table[key]
                if not ok:
                    break
            if not ok:
                continue
            if not np.array_equal(full[visible_mask], cell[visible_mask]):
                continue
            if len(set(int(v) for v in full.flatten())) < 2:
                continue
            candidates.append((ph * pw, full))
    return candidates


def _restore_pattern_cell(cell: np.ndarray) -> np.ndarray:
    out = cell.copy()
    out[out == ORANGE] = BG
    masked = np.isin(cell, [YELLOW, ORANGE])
    if not masked.any():
        return out
    candidates = _periodic_candidates(cell)
    if not candidates:
        out[masked] = BG
        return out
    min_area = min(area for area, _cand in candidates)
    best = [cand for area, cand in candidates if area == min_area]
    restored = best[0].copy()
    for cand in best[1:]:
        same = restored == cand
        restored[~same] = -1
    out[masked] = np.where(restored[masked] >= 0, restored[masked], BG)
    return out


def _cell_base_kind(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_bands, col_bands = _cell_slices(grid)
    kinds = np.full((3, 3), "", dtype=object)
    base = np.full((3, 3), -1, dtype=int)
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            color = _solid_cell_color(cell)
            if color is None:
                kinds[ri, ci] = "pattern"
            else:
                kinds[ri, ci] = "solid"
                base[ri, ci] = color
    return kinds, base


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair line-marked solid cells, erase line overlays from patterned cells, and preserve separator-link hints.

    Input structure:
    - Black background with two full grey separator rows and two full grey separator columns forming a 3x3 lattice of rectangular cells.
    - Some cells are solid monochrome interiors; the others are multicolor patterned interiors.
    - Yellow horizontal or vertical 1-pixel lines cross multiple cells and stay yellow on separator rows or columns.
    - In solid cells only, some yellow line pieces are replaced by short orange diagonal chains that mark the missing span.

    Rule:
    - Split the grid by the full grey separator rows and columns.
    - Classify each cell after ignoring grey separators, yellow line pixels, and orange noise pixels: one remaining color means a solid cell, otherwise a patterned cell.
    - In solid cells, erase orange noise, complete any short horizontal or vertical yellow gap whose length is bounded by the erased diagonal span, then recolor all yellow line pixels to the cell's solid color.
    - In patterned cells, erase yellow and orange overlays and restore the covered pixel colors from the cell's visible periodic pattern; if a covered pixel is not uniquely recoverable, paint it black.
    - Leave separator-row and separator-column yellow pixels unchanged.

    Key insight:
    - The separator lattice is the gating structure: inside solid cells the line is repaired and absorbed into the cell color, while inside patterned cells the same line is deleted and the hidden pattern is reconstructed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    row_bands, col_bands = _cell_slices(g)
    kinds, base = _cell_base_kind(g)

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            if kinds[ri, ci] == "solid":
                out[r0 : r1 + 1, c0 : c1 + 1] = _complete_solid_cell(cell, int(base[ri, ci]))
            else:
                out[r0 : r1 + 1, c0 : c1 + 1] = _restore_pattern_cell(cell)
    return out


def _choose_periodic_pattern(h: int, w: int) -> np.ndarray:
    while True:
        ph, pw = random.choice([(2, 2), (2, 3), (3, 2)])
        n_colors = random.choice([2, 3])
        colors = random.sample(PATTERN_COLORS, n_colors)
        tile = np.empty((ph, pw), dtype=int)
        for r in range(ph):
            for c in range(pw):
                tile[r, c] = colors[(r + 2 * c + random.randrange(n_colors)) % n_colors]
        if len(set(int(v) for v in tile.flatten())) < 2:
            continue
        full = np.tile(tile, (h // ph + 1, w // pw + 1))[:h, :w]
        if len(set(int(v) for v in full.flatten())) < 2:
            continue
        return full


def _pattern_restorable(pattern: np.ndarray, axis: str, line_pos: int) -> bool:
    masked = pattern.copy()
    if axis == "h":
        masked[line_pos, :] = YELLOW
    else:
        masked[:, line_pos] = YELLOW
    restored = _restore_pattern_cell(masked)
    return np.array_equal(restored, pattern)


def _noise_chain_h(cell_h: int, row: int, start: int, gap_len: int) -> list[tuple[int, int]] | None:
    options = []
    for anchor in range(-1, cell_h):
        pts = [(anchor + i, start + i) for i in range(gap_len)]
        if all(0 <= r < cell_h for r, _ in pts):
            options.append(pts)
        pts = [(anchor - i, start + i) for i in range(gap_len)]
        if all(0 <= r < cell_h for r, _ in pts):
            options.append(pts)
    best = []
    for pts in options:
        if any(r == row for r, _ in pts):
            best.append(pts)
    if best:
        return random.choice(best)
    return random.choice(options) if options else None


def _noise_chain_v(cell_w: int, col: int, start: int, gap_len: int) -> list[tuple[int, int]] | None:
    options = []
    for anchor in range(-1, cell_w):
        pts = [(start + i, anchor + i) for i in range(gap_len)]
        if all(0 <= c < cell_w for _, c in pts):
            options.append(pts)
        pts = [(start + i, anchor - i) for i in range(gap_len)]
        if all(0 <= c < cell_w for _, c in pts):
            options.append(pts)
    best = []
    for pts in options:
        if any(c == col for _, c in pts):
            best.append(pts)
    if best:
        return random.choice(best)
    return random.choice(options) if options else None


def _expected_output(base_cells: dict[tuple[int, int], np.ndarray], row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    h = sum(row_sizes) + 2
    w = sum(col_sizes) + 2
    out = np.full((h, w), BG, dtype=int)
    sep_rows = [row_sizes[0], row_sizes[0] + 1 + row_sizes[1]]
    sep_cols = [col_sizes[0], col_sizes[0] + 1 + col_sizes[1]]
    out[sep_rows, :] = GREY
    out[:, sep_cols] = GREY
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            out[r0 : r1 + 1, c0 : c1 + 1] = base_cells[(ri, ci)]
    return out


def _paint_separator_hints(
    grid: np.ndarray,
    row_sizes: list[int],
    col_sizes: list[int],
    orientation: str,
    bands: list[int],
    line_offsets: dict[int, int],
) -> None:
    sep_rows = [row_sizes[0], row_sizes[0] + 1 + row_sizes[1]]
    sep_cols = [col_sizes[0], col_sizes[0] + 1 + col_sizes[1]]
    row_bands = _bands(grid.shape[0], sep_rows)
    col_bands = _bands(grid.shape[1], sep_cols)
    if orientation == "h":
        for ri in bands:
            rr = row_bands[ri][0] + line_offsets[ri]
            for cc in sep_cols:
                grid[rr, cc] = YELLOW
    else:
        for ci in bands:
            cc = col_bands[ci][0] + line_offsets[ci]
            for rr in sep_rows:
                grid[rr, cc] = YELLOW


def _mixed_band_indices(mask: np.ndarray, axis: int) -> list[int]:
    mixed: list[int] = []
    for idx in range(3):
        vals = mask[idx, :] if axis == 0 else mask[:, idx]
        if vals.any() and (~vals).any():
            mixed.append(idx)
    return mixed


def _sample_solid_layout() -> np.ndarray:
    while True:
        n_solid = random.randint(4, 6)
        flat = np.array([True] * n_solid + [False] * (9 - n_solid), dtype=bool)
        np.random.shuffle(flat)
        solid_mask = flat.reshape(3, 3)
        if len(_mixed_band_indices(solid_mask, axis=0)) >= 2 and len(_mixed_band_indices(solid_mask, axis=1)) >= 2:
            return solid_mask


def _build_base_cells(row_sizes: list[int], col_sizes: list[int], solid_mask: np.ndarray, orientation: str, bands: list[int], line_offsets: dict[int, int]) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], int]]:
    base_cells: dict[tuple[int, int], np.ndarray] = {}
    solid_colors: dict[tuple[int, int], int] = {}
    for ri, rh in enumerate(row_sizes):
        for ci, cw in enumerate(col_sizes):
            if solid_mask[ri, ci]:
                color = random.choice(SOLID_COLORS)
                base_cells[(ri, ci)] = np.full((rh, cw), color, dtype=int)
                solid_colors[(ri, ci)] = color
            else:
                while True:
                    pattern = _choose_periodic_pattern(rh, cw)
                    good = True
                    if orientation == "h" and ri in bands:
                        if not _pattern_restorable(pattern, "h", line_offsets[ri]):
                            good = False
                    if orientation == "v" and ci in bands:
                        if not _pattern_restorable(pattern, "v", line_offsets[ci]):
                            good = False
                    if good:
                        base_cells[(ri, ci)] = pattern
                        break
    return base_cells, solid_colors


def _add_lines_and_gaps(
    grid: np.ndarray,
    solid_mask: np.ndarray,
    solid_colors: dict[tuple[int, int], int],
    row_sizes: list[int],
    col_sizes: list[int],
    orientation: str,
    bands: list[int],
    line_offsets: dict[int, int],
) -> dict[str, object]:
    sep_rows = [row_sizes[0], row_sizes[0] + 1 + row_sizes[1]]
    sep_cols = [col_sizes[0], col_sizes[0] + 1 + col_sizes[1]]
    row_bands = _bands(grid.shape[0], sep_rows)
    col_bands = _bands(grid.shape[1], sep_cols)
    gap_records: list[tuple[int, int]] = []

    if orientation == "h":
        for ri in bands:
            r0, _r1 = row_bands[ri]
            rr = r0 + line_offsets[ri]
            grid[rr, :] = YELLOW
            solid_cols = [ci for ci in range(3) if solid_mask[ri, ci]]
            ci = random.choice(solid_cols)
            c0, c1 = col_bands[ci]
            cell_w = c1 - c0 + 1
            gap_len = random.randint(2, cell_w - 2)
            start = random.randint(1, cell_w - gap_len - 1)
            grid[rr, c0 + start : c0 + start + gap_len] = int(solid_colors[(ri, ci)])
            chain = _noise_chain_h(row_sizes[ri], line_offsets[ri], start, gap_len)
            assert chain is not None
            for dr, dc in chain:
                grid[r0 + dr, c0 + dc] = ORANGE
            gap_records.append((ri, ci))
    else:
        for ci in bands:
            c0, _c1 = col_bands[ci]
            cc = c0 + line_offsets[ci]
            grid[:, cc] = YELLOW
            solid_rows = [ri for ri in range(3) if solid_mask[ri, ci]]
            ri = random.choice(solid_rows)
            r0, r1 = row_bands[ri]
            cell_h = r1 - r0 + 1
            gap_len = random.randint(2, cell_h - 2)
            start = random.randint(1, cell_h - gap_len - 1)
            grid[r0 + start : r0 + start + gap_len, cc] = int(solid_colors[(ri, ci)])
            chain = _noise_chain_v(col_sizes[ci], line_offsets[ci], start, gap_len)
            assert chain is not None
            for dr, dc in chain:
                grid[r0 + dr, c0 + dc] = ORANGE
            gap_records.append((ri, ci))
    return {"gap_records": gap_records}


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, solid_mask: np.ndarray, orientation: str, bands: list[int], line_offsets: dict[int, int]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    row_bands, col_bands = _cell_slices(inp)
    kinds, _base = _cell_base_kind(inp)
    for ri in range(3):
        for ci in range(3):
            want = "solid" if solid_mask[ri, ci] else "pattern"
            if kinds[ri, ci] != want:
                return False

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell_in = inp[r0 : r1 + 1, c0 : c1 + 1]
            cell_out = expected[r0 : r1 + 1, c0 : c1 + 1]
            if solid_mask[ri, ci]:
                if not np.all(cell_out == cell_out[0, 0]):
                    return False
                if np.any(cell_in == ORANGE):
                    if np.any(cell_out == ORANGE):
                        return False
            else:
                if np.any(cell_in == YELLOW):
                    masked = np.where(cell_in == YELLOW)
                    if np.array_equal(cell_out[masked], cell_in[masked]):
                        return False

    if orientation == "h":
        for ri in bands:
            rr = row_bands[ri][0] + line_offsets[ri]
            if not np.any(inp[rr, :] == YELLOW):
                return False
    else:
        for ci in bands:
            cc = col_bands[ci][0] + line_offsets[ci]
            if not np.any(inp[:, cc] == YELLOW):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_sizes = [random.randint(4, 5) for _ in range(3)]
    col_sizes = [random.randint(4, 5) for _ in range(3)]
    solid_mask = _sample_solid_layout()
    orientation = random.choice(["h", "v"])
    candidate_bands = _mixed_band_indices(solid_mask, axis=0 if orientation == "h" else 1)
    if len(candidate_bands) < 2:
        return None
    n_lines = random.randint(2, min(3, len(candidate_bands)))
    bands = random.sample(candidate_bands, n_lines)
    line_offsets = {}
    if orientation == "h":
        for ri in bands:
            line_offsets[ri] = random.randint(1, row_sizes[ri] - 2)
    else:
        for ci in bands:
            line_offsets[ci] = random.randint(1, col_sizes[ci] - 2)

    base_cells, solid_colors = _build_base_cells(row_sizes, col_sizes, solid_mask, orientation, bands, line_offsets)
    expected = _expected_output(base_cells, row_sizes, col_sizes)
    _paint_separator_hints(expected, row_sizes, col_sizes, orientation, bands, line_offsets)
    inp = expected.copy()
    sep_rows = [row_sizes[0], row_sizes[0] + 1 + row_sizes[1]]
    sep_cols = [col_sizes[0], col_sizes[0] + 1 + col_sizes[1]]
    inp[sep_rows, :] = GREY
    inp[:, sep_cols] = GREY

    _add_lines_and_gaps(inp, solid_mask, solid_colors, row_sizes, col_sizes, orientation, bands, line_offsets)

    if not _validate_semantics(inp, expected, solid_mask, orientation, bands, line_offsets):
        return None
    if len(set(int(v) for v in inp.flatten())) < 5:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
