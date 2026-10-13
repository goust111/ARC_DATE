import random

import numpy as np


BACKGROUND = 0
MASK = 5
PALETTE = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo the diagonal shear, repair grey mask pixels by left-right symmetry, then shear the pattern back.

    Input structure:
    - The background is black.
    - A single non-black parallelogram occupies consecutive rows; each occupied row is one contiguous run of equal width.
    - The run start column changes by one step per row, encoding a shear of an originally rectangular pattern.
    - Grey pixels mark damaged cells inside the sheared pattern.

    Rule:
    - Find the occupied rows and the contiguous foreground run on each such row.
    - Undo the shear by aligning those runs into an HxW rectangle in row order.
    - Repair each grey cell from its left-right mirror mate within that rectangle.
    - Reapply the original row offsets and place the repaired rows back into a same-size black canvas.

    Key insight:
    - The visible parallelogram is a sheared view of a hidden rectangle; only after undoing the row offsets does the mirror symmetry needed for repair become explicit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rows, starts, width = _sheared_runs(g)
    rect = np.zeros((len(rows), width), dtype=int)
    for out_r, (src_r, start) in enumerate(zip(rows, starts)):
        rect[out_r] = g[src_r, start : start + width]

    repaired = _repair_by_lr_symmetry(rect)

    out = np.zeros_like(g)
    for src_r, start, row_vals in zip(rows, starts, repaired):
        out[src_r, start : start + width] = row_vals
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/12/009/1")


def _sheared_runs(grid: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    rows = np.flatnonzero(np.any(grid != BACKGROUND, axis=1))
    assert rows.size > 0, "expected foreground rows"
    assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)), "foreground rows must be consecutive"

    starts: list[int] = []
    widths: list[int] = []
    for r in rows:
        cols = np.flatnonzero(grid[r] != BACKGROUND)
        assert cols.size > 0
        assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)), "each occupied row must be contiguous"
        starts.append(int(cols[0]))
        widths.append(int(cols.size))

    width = widths[0]
    assert all(w == width for w in widths), "all rows must share the same sheared width"
    diffs = np.diff(starts)
    assert diffs.size == 0 or np.all(np.abs(diffs) == 1), "row offsets must change by one column per row"
    assert diffs.size == 0 or len(set(map(int, diffs))) == 1, "shear direction must be consistent"
    assert rows.size >= 2 and width >= 2
    return rows, starts, width


def _repair_by_lr_symmetry(rect: np.ndarray) -> np.ndarray:
    out = rect.copy()
    h, w = out.shape
    for r in range(h):
        for c in range((w + 1) // 2):
            mc = w - 1 - c
            a = int(out[r, c])
            b = int(out[r, mc])
            if c == mc:
                assert a != MASK, "center-column mask would be ambiguous"
                continue
            visible = {v for v in (a, b) if v != MASK}
            assert visible, "each symmetry orbit must expose at least one visible color"
            assert len(visible) == 1, "mirror pair must agree on its visible color"
            fill = visible.pop()
            if a == MASK:
                out[r, c] = fill
            if b == MASK:
                out[r, mc] = fill
    assert not np.any(out == MASK), "all mask pixels must be repaired"
    return out


def _gen_attempt() -> np.ndarray | None:
    rect = _make_symmetric_rectangle()
    masked_rect = _mask_rectangle(rect)
    if masked_rect is None:
        return None

    clean_sheared = _shear_rectangle(rect)
    masked_sheared = _shear_rectangle(masked_rect)
    canvas = _place_on_canvas(masked_sheared, clean_sheared)
    if canvas is None:
        return None

    input_grid, expected_output = canvas

    if np.array_equal(input_grid, expected_output):
        return None
    if not np.array_equal(main(input_grid), expected_output):
        return None
    if len(set(map(int, np.unique(rect))) - {BACKGROUND}) < 2:
        return None
    if _mask_count(masked_rect) < 2:
        return None
    if np.array_equal(masked_rect, np.fliplr(masked_rect)):
        return None
    return input_grid


def _make_symmetric_rectangle() -> np.ndarray:
    h = random.randint(6, 9)
    w = random.randint(5, 8) if random.random() < 0.5 else random.randint(6, 9)
    half = (w + 1) // 2
    palette = random.sample(PALETTE, k=random.randint(2, min(4, len(PALETTE))))

    mode = random.choice(("noise", "row_bands", "diag", "patches"))
    left = np.zeros((h, half), dtype=int)

    if mode == "noise":
        left[:] = np.random.choice(palette, size=(h, half))
    elif mode == "row_bands":
        row_colors = np.random.choice(palette, size=h)
        left[:] = row_colors[:, None]
        for _ in range(random.randint(2, h + half)):
            r = random.randrange(h)
            c = random.randrange(half)
            left[r, c] = random.choice(palette)
    elif mode == "diag":
        base = random.randint(0, len(palette) - 1)
        step_r = random.randint(1, len(palette))
        step_c = random.randint(1, len(palette))
        for r in range(h):
            for c in range(half):
                left[r, c] = palette[(base + r * step_r + c * step_c) % len(palette)]
    else:
        fill = random.choice(palette)
        left[:] = fill
        for _ in range(random.randint(3, 6)):
            color = random.choice(palette)
            r0 = random.randint(0, h - 1)
            c0 = random.randint(0, half - 1)
            r1 = random.randint(r0, h - 1)
            c1 = random.randint(c0, half - 1)
            left[r0 : r1 + 1, c0 : c1 + 1] = color

    rect = np.zeros((h, w), dtype=int)
    rect[:, :half] = left
    for c in range(half):
        rect[:, w - 1 - c] = left[:, c]

    if _is_degenerate_rectangle(rect):
        return _make_symmetric_rectangle()
    return rect


def _is_degenerate_rectangle(rect: np.ndarray) -> bool:
    colors = set(map(int, np.unique(rect)))
    if len(colors) < 2:
        return True
    if len({tuple(row.tolist()) for row in rect}) < 2:
        return True
    if len({tuple(col.tolist()) for col in rect.T}) < 2:
        return True
    if np.array_equal(rect, np.roll(rect, 1, axis=0)):
        return True
    return False


def _mask_rectangle(rect: np.ndarray) -> np.ndarray | None:
    masked = rect.copy()
    h, w = rect.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w // 2):
            candidates.append((r, c))
    random.shuffle(candidates)

    max_masks = max(2, int(rect.size * 0.3))
    target = random.randint(max(2, rect.size // 10), max_masks)
    target = min(target, len(candidates))
    if target < 2:
        return None

    chosen = candidates[:target]
    for r, c in chosen:
        mc = w - 1 - c
        if random.random() < 0.5:
            masked[r, c] = MASK
        else:
            masked[r, mc] = MASK

    # Keep the odd-width center column visible so every orbit has a witness.
    if w % 2 == 1 and np.any(masked[:, w // 2] == MASK):
        return None

    mask_total = _mask_count(masked)
    if mask_total < 2 or mask_total > max_masks:
        return None

    rows_with_masks = np.flatnonzero(np.any(masked == MASK, axis=1))
    if rows_with_masks.size < max(2, h // 3):
        return None
    if len({int(c) for _r, c in chosen}) < max(2, min(3, w // 2)):
        return None

    repaired = _repair_by_lr_symmetry(masked)
    if not np.array_equal(repaired, rect):
        return None
    return masked


def _mask_count(grid: np.ndarray) -> int:
    return int(np.sum(grid == MASK))


def _shear_rectangle(rect: np.ndarray) -> np.ndarray:
    h, w = rect.shape
    out = np.zeros((h, w + h - 1), dtype=int)
    for r in range(h):
        start = h - 1 - r
        out[r, start : start + w] = rect[r]
    return out


def _place_on_canvas(masked_local: np.ndarray, clean_local: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = masked_local.shape
    canvas_h = random.randint(max(h + 2, 10), min(18, 30))
    canvas_w = random.randint(max(w + 2, 14), min(24, 30))
    if canvas_h < h or canvas_w < w:
        return None

    r0 = random.randint(0, canvas_h - h)
    c0 = random.randint(0, canvas_w - w)
    inp = np.zeros((canvas_h, canvas_w), dtype=int)
    out = np.zeros((canvas_h, canvas_w), dtype=int)
    inp[r0 : r0 + h, c0 : c0 + w] = masked_local
    out[r0 : r0 + h, c0 : c0 + w] = clean_local
    return inp, out

