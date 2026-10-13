from __future__ import annotations

import numpy as np


BG = 0
GRID_COLOR = 3
OBJECT_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]

CELL_ROW_RANGE = (2, 5)
CELL_COL_RANGE = (2, 5)
CELL_H_RANGE = (4, 9)
CELL_W_RANGE = (4, 9)
MAX_TOTAL_CELLS = 9
MAX_GENERATION_ATTEMPTS = 400


def _grid_structure(g: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, int, int]:
    h, w = g.shape
    candidates = []
    for color in map(int, np.unique(g)):
        mask = g == color
        full_rows = np.flatnonzero(mask.all(axis=1))
        full_cols = np.flatnonzero(mask.all(axis=0))
        if len(full_rows) < 2 or len(full_cols) < 2:
            continue

        lattice = np.zeros_like(mask, dtype=bool)
        lattice[full_rows, :] = True
        lattice[:, full_cols] = True
        if not np.array_equal(mask, lattice):
            continue

        if full_rows[0] != 0 or full_rows[-1] != h - 1:
            continue
        if full_cols[0] != 0 or full_cols[-1] != w - 1:
            continue

        row_steps = np.diff(full_rows)
        col_steps = np.diff(full_cols)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue

        row_step = int(row_steps[0])
        col_step = int(col_steps[0])
        if row_step < 2 or col_step < 2:
            continue

        candidates.append((color, full_rows, full_cols, row_step - 1, col_step - 1))

    assert len(candidates) == 1, "expected one regular separator lattice"
    return candidates[0]


def _cell_spans(line_rows: np.ndarray, line_cols: np.ndarray) -> list[tuple[int, int, int, int]]:
    spans = []
    for i in range(len(line_rows) - 1):
        for j in range(len(line_cols) - 1):
            r0 = int(line_rows[i]) + 1
            r1 = int(line_rows[i + 1])
            c0 = int(line_cols[j]) + 1
            c1 = int(line_cols[j + 1])
            spans.append((r0, r1, c0, c1))
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct the full local shape by unioning incomplete copies across grid cells."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    separator, line_rows, line_cols, cell_h, cell_w = _grid_structure(g)
    spans = _cell_spans(line_rows, line_cols)
    assert len(spans) >= 2, "expected multiple cells"

    content_colors = {int(v) for v in np.unique(g) if int(v) not in {BG, separator}}
    assert len(content_colors) == 1, "expected a single pattern color"
    (obj_color,) = tuple(content_colors)

    full_mask = np.zeros((cell_h, cell_w), dtype=bool)
    for r0, r1, c0, c1 in spans:
        cell = g[r0:r1, c0:c1]
        assert cell.shape == (cell_h, cell_w)
        assert not np.any(cell == separator), "separator color must stay on grid lines"
        assert np.all((cell == BG) | (cell == obj_color)), "cells must contain only background and pattern"

        mask = cell == obj_color
        assert mask.any(), "each cell should keep some visible pixels"
        full_mask |= mask

    assert full_mask.any(), "expected a non-empty reconstructed shape"

    out = g.copy()
    for r0, r1, c0, c1 in spans:
        patch = out[r0:r1, c0:c1]
        patch[:] = BG
        patch[full_mask] = obj_color
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _grow_connected_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        choice = tuple(list(frontier)[int(rng.integers(0, len(frontier)))])
        cells.add(choice)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _shape_blob(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    total = h * w
    min_area = max(6, int(np.ceil(total * 0.25)))
    max_area = min(total - 1, max(min_area, int(np.floor(total * 0.65))))
    target = int(rng.integers(min_area, max_area + 1))
    return _grow_connected_mask(rng, h, w, target)


def _shape_bars(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)

    v_h = int(rng.integers(max(2, h // 2), h + 1))
    v_w = int(rng.integers(1, min(3, w) + 1))
    vr0 = int(rng.integers(0, h - v_h + 1))
    vc0 = int(rng.integers(0, w - v_w + 1))
    mask[vr0 : vr0 + v_h, vc0 : vc0 + v_w] = True

    h_h = int(rng.integers(1, min(3, h) + 1))
    shared_r = int(rng.integers(vr0, vr0 + v_h))
    hr0 = min(max(0, shared_r - int(rng.integers(0, h_h))), h - h_h)
    h_w = int(rng.integers(max(2, (w + 1) // 2), w + 1))
    c_lo = max(0, vc0 - h_w + 1)
    c_hi = min(vc0 + v_w - 1, w - h_w)
    if c_lo > c_hi:
        hc0 = max(0, min(vc0, w - h_w))
    else:
        hc0 = int(rng.integers(c_lo, c_hi + 1))
    mask[hr0 : hr0 + h_h, hc0 : hc0 + h_w] = True

    if rng.random() < 0.6:
        coords = np.argwhere(mask)
        ar, ac = map(int, coords[int(rng.integers(0, len(coords)))])
        ext_h = int(rng.integers(1, min(3, h) + 1))
        ext_w = int(rng.integers(1, min(3, w) + 1))
        er0 = max(0, min(ar - int(rng.integers(0, ext_h)), h - ext_h))
        ec0 = max(0, min(ac - int(rng.integers(0, ext_w)), w - ext_w))
        mask[er0 : er0 + ext_h, ec0 : ec0 + ext_w] = True

    return mask


def _shape_frame(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    rect_h = int(rng.integers(3, h + 1))
    rect_w = int(rng.integers(3, w + 1))
    r0 = int(rng.integers(0, h - rect_h + 1))
    c0 = int(rng.integers(0, w - rect_w + 1))
    r1 = r0 + rect_h
    c1 = c0 + rect_w

    mask[r0, c0:c1] = True
    mask[r1 - 1, c0:c1] = True
    mask[r0:r1, c0] = True
    mask[r0:r1, c1 - 1] = True

    if rng.random() < 0.5:
        if rect_h > rect_w and rect_w >= 4:
            cc = int(rng.integers(c0 + 1, c1 - 1))
            mask[r0:r1, cc] = True
        elif rect_h >= 4:
            rr = int(rng.integers(r0 + 1, r1 - 1))
            mask[rr, c0:c1] = True
    return mask


def _make_shape(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    makers = [_shape_blob, _shape_bars]
    if h >= 4 and w >= 4:
        makers.append(_shape_frame)

    for _ in range(20):
        maker = makers[int(rng.integers(0, len(makers)))]
        mask = maker(rng, h, w)
        area = int(mask.sum())
        if 6 <= area < h * w:
            return mask
    return None


def _sample_rect_chunk(rng: np.random.Generator, allowed: np.ndarray) -> np.ndarray:
    h, w = allowed.shape
    for _ in range(20):
        r0 = int(rng.integers(0, h))
        r1 = int(rng.integers(r0 + 1, h + 1))
        c0 = int(rng.integers(0, w))
        c1 = int(rng.integers(c0 + 1, w + 1))
        chunk = np.zeros_like(allowed, dtype=bool)
        chunk[r0:r1, c0:c1] = True
        chunk &= allowed
        if chunk.any():
            return chunk

    coords = np.argwhere(allowed)
    rr, cc = map(int, coords[int(rng.integers(0, len(coords)))])
    chunk = np.zeros_like(allowed, dtype=bool)
    chunk[rr, cc] = True
    return chunk


def _sample_blob_chunk(rng: np.random.Generator, allowed: np.ndarray, target: int) -> np.ndarray:
    h, w = allowed.shape
    coords = np.argwhere(allowed)
    sr, sc = map(int, coords[int(rng.integers(0, len(coords)))])
    cells = {(sr, sc)}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if allowed[nr, nc] and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        choice = tuple(list(frontier)[int(rng.integers(0, len(frontier)))])
        cells.add(choice)

    chunk = np.zeros_like(allowed, dtype=bool)
    for r, c in cells:
        chunk[r, c] = True
    return chunk


def _make_partial_mask(rng: np.random.Generator, full_mask: np.ndarray) -> np.ndarray | None:
    area = int(full_mask.sum())
    min_hidden = max(1, int(np.ceil(area * 0.18)))
    max_hidden = min(area - 1, max(min_hidden, int(np.floor(area * 0.45))))
    min_visible = max(2, int(np.ceil(area * 0.35)))
    max_hidden = min(max_hidden, area - min_visible)
    if min_hidden > max_hidden:
        return None

    for _ in range(40):
        hidden = np.zeros_like(full_mask, dtype=bool)
        chunk_count = int(rng.integers(1, 4))
        for _ in range(chunk_count):
            allowed = full_mask & ~hidden
            if not allowed.any():
                break

            hidden_now = int(hidden.sum())
            remaining_cap = max_hidden - hidden_now
            if remaining_cap <= 0:
                break

            if rng.random() < 0.5:
                chunk = _sample_rect_chunk(rng, allowed)
            else:
                target = int(rng.integers(1, min(remaining_cap, max(2, area // 4)) + 1))
                chunk = _sample_blob_chunk(rng, allowed, target)
            hidden |= chunk

            hidden_now = int(hidden.sum())
            if hidden_now >= min_hidden and rng.random() < 0.5:
                break

        hidden_count = int(hidden.sum())
        if min_hidden <= hidden_count <= max_hidden:
            visible = full_mask & ~hidden
            if int(visible.sum()) >= min_visible:
                return visible
    return None


def _restore_coverage(
    full_mask: np.ndarray,
    cell_masks: list[np.ndarray],
) -> None:
    coverage = np.zeros_like(full_mask, dtype=bool)
    for mask in cell_masks:
        coverage |= mask

    missing = np.argwhere(full_mask & ~coverage)
    if not len(missing):
        return

    hidden_counts = [int(full_mask.sum() - mask.sum()) for mask in cell_masks]
    for r, c in missing:
        rr, cc = map(int, (r, c))
        options = [i for i, mask in enumerate(cell_masks) if not mask[rr, cc] and hidden_counts[i] > 1]
        if not options:
            options = [i for i, mask in enumerate(cell_masks) if not mask[rr, cc]]
        assert options, "every uncovered pixel should fit back into some cell"
        idx = max(options, key=lambda i: hidden_counts[i])
        cell_masks[idx][rr, cc] = True
        hidden_counts[idx] -= 1


def _render_grid(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    obj_color: int,
    masks: list[np.ndarray],
) -> np.ndarray:
    height = n_rows * cell_h + n_rows + 1
    width = n_cols * cell_w + n_cols + 1
    grid = np.full((height, width), BG, dtype=int)

    line_rows = np.arange(0, height, cell_h + 1)
    line_cols = np.arange(0, width, cell_w + 1)
    grid[line_rows, :] = GRID_COLOR
    grid[:, line_cols] = GRID_COLOR

    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            r0 = 1 + i * (cell_h + 1)
            c0 = 1 + j * (cell_w + 1)
            patch = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            patch[masks[idx]] = obj_color
            idx += 1
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rows = int(rng.integers(*CELL_ROW_RANGE))
    n_cols = int(rng.integers(*CELL_COL_RANGE))
    if not (4 <= n_rows * n_cols <= MAX_TOTAL_CELLS):
        return None

    cell_h = int(rng.integers(*CELL_H_RANGE))
    cell_w = int(rng.integers(*CELL_W_RANGE))
    obj_color = int(rng.choice(OBJECT_COLORS))

    full_mask = _make_shape(rng, cell_h, cell_w)
    if full_mask is None:
        return None
    if int(full_mask.sum()) < n_rows * n_cols + 2:
        return None

    k = n_rows * n_cols
    cell_masks: list[np.ndarray] = []
    for _ in range(k):
        mask = _make_partial_mask(rng, full_mask)
        if mask is None:
            return None
        cell_masks.append(mask)

    _restore_coverage(full_mask, cell_masks)

    if any((not mask.any()) or np.array_equal(mask, full_mask) for mask in cell_masks):
        return None
    if len({mask.tobytes() for mask in cell_masks}) < 2:
        return None

    union = np.zeros_like(full_mask, dtype=bool)
    for mask in cell_masks:
        union |= mask
    if not np.array_equal(union, full_mask):
        return None

    inp = _render_grid(n_rows, n_cols, cell_h, cell_w, obj_color, cell_masks)
    out = _render_grid(n_rows, n_cols, cell_h, cell_w, obj_color, [full_mask] * k)
    if np.array_equal(inp, out):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
