from __future__ import annotations

import random

import numpy as np


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    vals = [int(x) for x in indices]
    assert vals, "expected at least one active index"
    out: list[tuple[int, int]] = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    out.append((start, prev))
    return out


def _separator_color(grid: np.ndarray) -> int:
    candidates: list[int] = []
    for color in np.unique(grid):
        color = int(color)
        if np.any(np.all(grid == color, axis=1)) and np.any(np.all(grid == color, axis=0)):
            candidates.append(color)
    assert len(candidates) == 1, f"expected exactly one separator color, got {candidates}"
    return candidates[0]


def _base_and_mask(panel: np.ndarray) -> tuple[int, np.ndarray]:
    colors, counts = np.unique(panel, return_counts=True)
    assert len(colors) == 2, f"panel must have exactly two colors, got {colors.tolist()}"
    order = np.argsort(counts)[::-1]
    base = int(colors[int(order[0])])
    motif_count = int(counts[int(order[1])])
    assert int(counts[int(order[0])]) > motif_count, "base color must be uniquely dominant"
    return base, panel != base


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap motif masks between horizontally paired panels and repaint them with the source panel base colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep = _separator_color(g)
    active_rows = np.where(~np.all(g == sep, axis=1))[0]
    active_cols = np.where(~np.all(g == sep, axis=0))[0]
    row_bands = _bands(active_rows)
    col_bands = _bands(active_cols)
    assert len(col_bands) == 2, f"expected 2 column bands, got {len(col_bands)}"

    out = g.copy()
    (lc0, lc1), (rc0, rc1) = col_bands
    for r0, r1 in row_bands:
        left = g[r0 : r1 + 1, lc0 : lc1 + 1]
        right = g[r0 : r1 + 1, rc0 : rc1 + 1]
        assert left.shape == right.shape, "paired panels must have equal shapes"

        left_base, left_mask = _base_and_mask(left)
        right_base, right_mask = _base_and_mask(right)

        new_left = np.full(left.shape, left_base, dtype=int)
        new_right = np.full(right.shape, right_base, dtype=int)
        new_left[right_mask] = right_base
        new_right[left_mask] = left_base

        out[r0 : r1 + 1, lc0 : lc1 + 1] = new_left
        out[r0 : r1 + 1, rc0 : rc1 + 1] = new_right
    return out


PALETTE = tuple(range(10))
MAX_RETRIES = 400
MAX_SIDE = 30


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _random_blob_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r = random.randrange(h)
    c = random.randrange(w)
    mask[r, c] = True
    frontier = {(r, c)}
    while int(mask.sum()) < target:
        candidates = set()
        for rr, cc in frontier:
            for nr, nc in _neighbors4(rr, cc, h, w):
                if not mask[nr, nc]:
                    candidates.add((nr, nc))
        if not candidates:
            candidates = {(rr, cc) for rr in range(h) for cc in range(w) if not mask[rr, cc]}
        nr, nc = random.choice(list(candidates))
        mask[nr, nc] = True
        frontier.add((nr, nc))
    return mask


def _random_rect_union_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    pieces = random.randint(1, 3)
    for _ in range(pieces):
        rh = random.randint(1, max(1, min(h, target)))
        rw = random.randint(1, max(1, min(w, target)))
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
        if int(mask.sum()) >= target:
            break
    cells = np.argwhere(mask)
    if len(cells) > target:
        keep = set(map(tuple, random.sample([tuple(x) for x in cells], target)))
        trimmed = np.zeros_like(mask)
        for rr, cc in keep:
            trimmed[rr, cc] = True
        mask = trimmed
    while int(mask.sum()) < target:
        rr = random.randrange(h)
        cc = random.randrange(w)
        mask[rr, cc] = True
    return mask


def _structured_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    style = random.choice(("cross", "x", "stripes", "frame"))
    if style == "cross":
        rr = random.randrange(h)
        cc = random.randrange(w)
        mask[rr, :] = True
        mask[:, cc] = True
    elif style == "x":
        for r in range(h):
            c1 = round(r * (w - 1) / max(1, h - 1))
            c2 = (w - 1) - c1
            mask[r, c1] = True
            mask[r, c2] = True
    elif style == "stripes":
        axis = random.choice((0, 1))
        step = random.choice((2, 3))
        offset = random.randrange(step)
        if axis == 0:
            mask[offset::step, :] = True
        else:
            mask[:, offset::step] = True
    else:
        if h >= 3 and w >= 3:
            mask[[0, -1], :] = True
            mask[:, [0, -1]] = True
        else:
            mask[:, :] = True
    return mask


def _random_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    max_cells = max(1, area // 3)
    min_cells = 1
    target = random.randint(min_cells, max_cells)
    for _ in range(40):
        kind = random.choice(("blob", "rects", "structured"))
        if kind == "blob":
            mask = _random_blob_mask(h, w, target)
        elif kind == "rects":
            mask = _random_rect_union_mask(h, w, target)
        else:
            mask = _structured_mask(h, w)
            cells = int(mask.sum())
            if cells > max_cells:
                coords = [tuple(x) for x in np.argwhere(mask)]
                keep = set(random.sample(coords, target))
                trimmed = np.zeros_like(mask)
                for rr, cc in keep:
                    trimmed[rr, cc] = True
                mask = trimmed
            elif cells < min_cells:
                continue
        count = int(mask.sum())
        if 1 <= count < area - count:
            return mask
    return None


def _color_quad() -> tuple[int, int, int, int]:
    colors = list(PALETTE)
    sep = random.choice(colors)
    colors.remove(sep)
    same_base = random.random() < 0.5
    if same_base:
        base = random.choice(colors)
        rest = [c for c in colors if c != base]
        left_motif = random.choice(rest)
        right_motif = random.choice(rest)
        return sep, base, base, left_motif if left_motif != base else rest[0], right_motif if right_motif != base else rest[-1]
    left_base, right_base = random.sample(colors, 2)
    left_motif = random.choice([c for c in colors if c != left_base])
    right_motif = random.choice([c for c in colors if c != right_base])
    return sep, left_base, right_base, left_motif, right_motif


def _panel_pair(h: int, w: int, sep: int) -> tuple[np.ndarray, np.ndarray] | None:
    available = [c for c in PALETTE if c != sep]
    same_base = random.random() < 0.5
    if same_base:
        base = random.choice(available)
        left_base = right_base = base
    else:
        left_base, right_base = random.sample(available, 2)
    left_motif_color = random.choice([c for c in available if c != left_base])
    right_motif_color = random.choice([c for c in available if c != right_base])

    left_mask = _random_mask(h, w)
    right_mask = _random_mask(h, w)
    if left_mask is None or right_mask is None:
        return None

    left = np.full((h, w), left_base, dtype=int)
    right = np.full((h, w), right_base, dtype=int)
    left[left_mask] = left_motif_color
    right[right_mask] = right_motif_color
    return left, right


def _gen_attempt() -> np.ndarray | None:
    num_bands = random.randint(2, 4)
    panel_h = random.randint(4, 8)
    panel_w = random.randint(4, 9)
    sep_h = random.randint(1, 3)
    sep_w = random.randint(1, 3)
    pad_top = random.randint(0, 2)
    pad_bottom = random.randint(0, 2)
    pad_left = random.randint(0, 2)
    pad_right = random.randint(0, 2)

    height = pad_top + num_bands * panel_h + (num_bands - 1) * sep_h + pad_bottom
    width = pad_left + panel_w + sep_w + panel_w + pad_right
    if height > MAX_SIDE or width > MAX_SIDE:
        return None

    sep = random.randrange(10)
    grid = np.full((height, width), sep, dtype=int)
    left_c0 = pad_left
    right_c0 = pad_left + panel_w + sep_w

    for band in range(num_bands):
        r0 = pad_top + band * (panel_h + sep_h)
        pair = _panel_pair(panel_h, panel_w, sep)
        if pair is None:
            return None
        left, right = pair
        _paste(grid, left, r0, left_c0)
        _paste(grid, right, r0, right_c0)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
