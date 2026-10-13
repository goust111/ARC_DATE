from __future__ import annotations

import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _try_parse_separator(
    grid: np.ndarray, bg: int, sep_row: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], int] | None:
    if sep_row <= 0 or sep_row >= grid.shape[0] - 1:
        return None

    upper = grid[:sep_row, :]
    lower = grid[sep_row + 1 :, :]

    upper_bbox = _bbox(upper != bg)
    lower_bbox = _bbox(lower != bg)
    if upper_bbox is None or lower_bbox is None:
        return None

    ur0, uc0, ur1, uc1 = upper_bbox
    lr0, lc0, lr1, lc1 = lower_bbox
    palette = upper[ur0 : ur1 + 1, uc0 : uc1 + 1]
    mask_region = lower[lr0 : lr1 + 1, lc0 : lc1 + 1]

    if palette.shape[0] < 2 or palette.shape[1] < 2:
        return None
    if np.any(palette == bg) or np.unique(palette).size < 2:
        return None

    lower_non_bg = [int(v) for v in np.unique(lower) if int(v) != bg]
    if len(lower_non_bg) != 1:
        return None
    gray = lower_non_bg[0]
    if np.any((mask_region != bg) & (mask_region != gray)):
        return None

    ph, pw = palette.shape
    mh, mw = mask_region.shape
    if mh % ph != 0 or mw % pw != 0:
        return None

    return palette, lower, (lr0, lc0, lr1, lc1), gray


def _parse_grid(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], int]:
    bg = _dominant_color(grid)
    candidates = []
    for sep_row in range(grid.shape[0]):
        row = grid[sep_row, :]
        if row[0] == bg or not np.all(row == row[0]):
            continue
        parsed = _try_parse_separator(grid, bg, sep_row)
        if parsed is not None:
            candidates.append(parsed)

    assert len(candidates) == 1, "expected exactly one valid separator row"
    return candidates[0]


def _apply_palette(
    lower: np.ndarray,
    palette: np.ndarray,
    mask_bbox: tuple[int, int, int, int],
    gray: int,
) -> np.ndarray:
    out = lower.copy()
    r0, c0, r1, c1 = mask_bbox
    region = out[r0 : r1 + 1, c0 : c1 + 1]
    ph, pw = palette.shape
    bh = region.shape[0] // ph
    bw = region.shape[1] // pw

    for pr in range(ph):
        for pc in range(pw):
            block = region[pr * bh : (pr + 1) * bh, pc * bw : (pc + 1) * bw]
            block[block == gray] = int(palette[pr, pc])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the upper multicolor grid as a compact color index table and recolor the gray mask below block by block.

    Input structure:
    - A dominant background fills most cells.
    - A full separator-color row splits the grid into an upper and lower region.
    - The upper region contains a small non-background NxM palette.
    - The lower region contains a larger pattern made from background and one gray color.

    Rule:
    - Divide the lower pattern into NxM blocks, using the palette dimensions.
    - Each block inherits the color from the palette cell at the same row/column.
    - Replace gray pixels inside a block with that palette color.
    - Keep background pixels unchanged.
    - Remove the upper region and separator row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D grid with two regions"

    palette, lower, mask_bbox, gray = _parse_grid(g)
    return _apply_palette(lower, palette, mask_bbox, gray)


# --------------- generator ---------------

PALETTE_H_RANGE = (2, 4)
PALETTE_W_RANGE = (2, 5)
BLOCK_H_RANGE = (3, 5)
BLOCK_W_RANGE = (3, 5)
SIDE_MARGIN_RANGE = (1, 2)
GENERATION_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _blob_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], h, w))

    while int(mask.sum()) < target and frontier:
        rr, cc = random.choice(tuple(frontier))
        frontier.remove((rr, cc))
        mask[rr, cc] = True
        for nxt in _neighbors4(rr, cc, h, w):
            if not mask[nxt]:
                frontier.add(nxt)

    return mask


def _random_block_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    for _ in range(100):
        mode = random.choice(("rect", "plus", "diag", "corner", "blob", "compound"))
        mask = np.zeros((h, w), dtype=bool)

        if mode == "rect":
            rh = random.randint(1, h - 1)
            rw = random.randint(1, w - 1)
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            mask[r0 : r0 + rh, c0 : c0 + rw] = True
        elif mode == "plus":
            rr = random.randrange(h)
            cc = random.randrange(w)
            mask[rr, :] = True
            mask[:, cc] = True
            if h >= 5 and random.random() < 0.35:
                rr2 = min(h - 1, max(0, rr + random.choice((-1, 1))))
                mask[rr2, max(0, cc - 1) : min(w, cc + 2)] = True
            if w >= 5 and random.random() < 0.35:
                cc2 = min(w - 1, max(0, cc + random.choice((-1, 1))))
                mask[max(0, rr - 1) : min(h, rr + 2), cc2] = True
        elif mode == "diag":
            if random.random() < 0.5:
                cols = [round(r * (w - 1) / max(1, h - 1)) for r in range(h)]
            else:
                cols = [(w - 1) - round(r * (w - 1) / max(1, h - 1)) for r in range(h)]
            for rr, cc in enumerate(cols):
                mask[rr, int(cc)] = True
            if random.random() < 0.5:
                rr = random.randrange(h)
                cc = int(cols[rr])
                if cc + 1 < w:
                    mask[rr, cc + 1] = True
                elif cc - 1 >= 0:
                    mask[rr, cc - 1] = True
        elif mode == "corner":
            top = random.random() < 0.5
            left = random.random() < 0.5
            arm_h = random.randint(2, h)
            arm_w = random.randint(2, w)
            row_slice = slice(0, arm_h) if top else slice(h - arm_h, h)
            col_slice = slice(0, arm_w) if left else slice(w - arm_w, w)
            anchor_r = 0 if top else h - 1
            anchor_c = 0 if left else w - 1
            mask[row_slice, anchor_c] = True
            mask[anchor_r, col_slice] = True
        elif mode == "blob":
            target = random.randint(3, area - 3)
            mask = _blob_mask(h, w, target)
        else:
            for _ in range(random.randint(2, 3)):
                rh = random.randint(1, h - 1)
                rw = random.randint(1, w - 1)
                r0 = random.randint(0, h - rh)
                c0 = random.randint(0, w - rw)
                mask[r0 : r0 + rh, c0 : c0 + rw] = True

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)

        count = int(mask.sum())
        if 3 <= count <= area - 3:
            return np.array(mask, dtype=bool)

    target = random.randint(3, area - 3)
    mask = np.zeros((h, w), dtype=bool)
    mask.flat[random.sample(range(area), k=target)] = True
    return mask


def _compose_mask(ph: int, pw: int, bh: int, bw: int) -> np.ndarray:
    mask = np.zeros((ph * bh, pw * bw), dtype=bool)
    for pr in range(ph):
        for pc in range(pw):
            block = _random_block_mask(bh, bw)
            mask[pr * bh : (pr + 1) * bh, pc * bw : (pc + 1) * bw] = block

    if not mask[0, :].any():
        mask[0, random.randrange(mask.shape[1])] = True
    if not mask[-1, :].any():
        mask[-1, random.randrange(mask.shape[1])] = True
    if not mask[:, 0].any():
        mask[random.randrange(mask.shape[0]), 0] = True
    if not mask[:, -1].any():
        mask[random.randrange(mask.shape[0]), -1] = True

    return mask


def _random_palette(h: int, w: int, colors: list[int]) -> np.ndarray:
    max_colors = min(6, len(colors), h * w)
    min_colors = 3 if max_colors >= 3 else 2
    count = random.randint(min_colors, max_colors)
    chosen = random.sample(colors, count)
    flat = chosen + random.choices(chosen, k=h * w - count)
    random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(h, w)


def _expected_output(
    lower: np.ndarray,
    mask_left: int,
    palette: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    out = lower.copy()
    region = out[:, mask_left : mask_left + mask.shape[1]]
    ph, pw = palette.shape
    bh = mask.shape[0] // ph
    bw = mask.shape[1] // pw

    for pr in range(ph):
        for pc in range(pw):
            block_mask = mask[pr * bh : (pr + 1) * bh, pc * bw : (pc + 1) * bw]
            region_block = region[pr * bh : (pr + 1) * bh, pc * bw : (pc + 1) * bw]
            region_block[block_mask] = int(palette[pr, pc])

    return out


def _gen_attempt() -> np.ndarray | None:
    ph = random.randint(*PALETTE_H_RANGE)
    pw = random.randint(*PALETTE_W_RANGE)
    bh = random.randint(*BLOCK_H_RANGE)
    bw = random.randint(*BLOCK_W_RANGE)
    mask_h = ph * bh
    mask_w = pw * bw

    bg, sep_color, gray = random.sample(range(10), 3)
    palette_colors = [c for c in range(10) if c not in (bg, sep_color, gray)]
    palette = _random_palette(ph, pw, palette_colors)
    mask = _compose_mask(ph, pw, bh, bw)

    left_margin = random.randint(*SIDE_MARGIN_RANGE)
    right_margin = random.randint(*SIDE_MARGIN_RANGE)
    grid_w = left_margin + mask_w + right_margin
    palette_x = random.randint(0, grid_w - pw)

    top = np.full((ph, grid_w), bg, dtype=int)
    top[:, palette_x : palette_x + pw] = palette

    lower = np.full((mask_h, grid_w), bg, dtype=int)
    lower_region = lower[:, left_margin : left_margin + mask_w]
    lower_region[mask] = gray

    separator = np.full((1, grid_w), sep_color, dtype=int)
    grid = np.vstack([top, separator, lower])

    if _dominant_color(grid) != bg:
        return None

    full_rows = [
        r
        for r in range(grid.shape[0])
        if grid[r, 0] != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    if full_rows != [ph]:
        return None

    expected = _expected_output(lower, left_margin, palette, mask)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    non_bg_out = np.unique(actual[actual != bg])
    if non_bg_out.size < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
