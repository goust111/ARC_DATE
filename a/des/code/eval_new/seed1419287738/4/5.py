import random

import numpy as np


BG = 0
GREY = 5
TEMPLATE_COLORS = [1, 2, 3, 4]
FRAME_COLORS = [6, 7, 8, 9]


def _separator_row(grid):
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    return rows[0] if rows else None


def _template_map(grid, sep):
    mapping = {}
    for r in range(sep):
        vals = [int(v) for v in grid[r] if v not in (BG, GREY)]
        if not vals:
            continue
        color = vals[0]
        cols = np.where(grid[r] == color)[0]
        if len(cols) == 0:
            continue
        c0, c1 = int(cols.min()), int(cols.max())
        mask = tuple(int(grid[r, c] == color) for c in range(c0, c1 + 1))
        mapping[mask] = color
    return mapping


def _frames(grid, sep):
    h, w = grid.shape
    found = []
    for color in range(1, 10):
        if color == GREY:
            continue
        coords = np.argwhere(grid[sep + 1:] == color)
        if coords.size == 0:
            continue
        coords[:, 0] += sep + 1
        rows = sorted(set(int(r) for r, _ in coords))
        cols = sorted(set(int(c) for _, c in coords))
        for r0 in rows:
            for r1 in rows:
                if r1 <= r0 + 2:
                    continue
                for c0 in cols:
                    for c1 in cols:
                        if c1 <= c0 + 3:
                            continue
                        if np.all(grid[r0, c0:c1 + 1] == color) and np.all(grid[r1, c0:c1 + 1] == color):
                            if np.all(grid[r0:r1 + 1, c0] == color) and np.all(grid[r0:r1 + 1, c1] == color):
                                found.append((r0, r1, c0, c1, color))
    # Remove duplicate candidates that share the same border.
    unique = []
    seen = set()
    for frame in found:
        key = frame[:4]
        if key not in seen:
            seen.add(key)
            unique.append(frame)
    return unique


def main(grid: np.ndarray) -> np.ndarray:
    """Fill framed rows by matching grey fingerprints to colored template stripes.

    Input structure:
    A full grey separator splits a top template panel from a lower target panel.
    The top panel has one-row colored binary stripes.  The lower panel has colored
    rectangular frames whose interiors contain grey fingerprint rows and black rows.

    Rule:
    Each grey interior row is matched against the top template stripe with the same
    binary pattern, and the grey pixels are recolored to that template's color.
    Interior rows that were entirely black are filled across the frame interior with
    the frame border color.  The top panel, separator, and frame borders remain.

    Key insight:
    Grey pixels encode a shape fingerprint, while the frame color supplies the fill
    for empty rows; those two color sources are independent.
    """
    out = np.array(grid, copy=True)
    sep = _separator_row(grid)
    if sep is None:
        return out
    templates = _template_map(grid, sep)
    for r0, r1, c0, c1, frame_color in _frames(grid, sep):
        for r in range(r0 + 1, r1):
            row = grid[r, c0 + 1:c1]
            if np.all(row == BG):
                out[r, c0 + 1:c1] = frame_color
            elif np.any(row == GREY):
                mask = tuple(int(v == GREY) for v in row)
                template_color = templates.get(mask)
                if template_color is not None:
                    for offset, bit in enumerate(mask):
                        if bit:
                            out[r, c0 + 1 + offset] = template_color
    return out


def _unique_masks(width, n):
    masks = set()
    while len(masks) < n:
        mask = tuple(1 if random.random() < 0.45 else 0 for _ in range(width))
        if 2 <= sum(mask) <= width - 1:
            masks.add(mask)
    return list(masks)


def _draw_frame(grid, r0, c0, ih, iw, color):
    r1, c1 = r0 + ih + 1, c0 + iw + 1
    if r1 >= grid.shape[0] or c1 >= grid.shape[1]:
        return False
    if np.any(grid[r0:r1 + 1, c0:c1 + 1] != BG):
        return False
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color
    return True


def generate_input() -> np.ndarray:
    for _ in range(1000):
        width = random.randint(5, 8)
        n_templates = random.randint(3, 4)
        masks = _unique_masks(width, n_templates)
        h = random.randint(20, 29)
        w = max(width + 16, random.randint(18, 29))
        sep = random.randint(n_templates * 2 + 1, n_templates * 2 + 3)
        grid = np.zeros((h, w), dtype=int)
        for i, mask in enumerate(masks):
            r = 1 + i * 2
            color = TEMPLATE_COLORS[i]
            for j, bit in enumerate(mask):
                if bit:
                    grid[r, 2 + j] = color
        grid[sep, :] = GREY

        n_frames = random.randint(2, 3)
        frame_rows = [sep + 2 + i * 5 for i in range(n_frames)]
        ok = True
        for i, r0 in enumerate(frame_rows):
            if r0 + 4 >= h:
                ok = False
                break
            c0 = random.randint(2, max(2, w - width - 4))
            ih = random.randint(3, 4)
            if not _draw_frame(grid, r0, c0, ih, width, FRAME_COLORS[i]):
                ok = False
                break
            interior_rows = list(range(r0 + 1, r0 + ih + 1))
            random.shuffle(interior_rows)
            chosen = random.sample(masks, random.randint(1, min(2, len(masks))))
            for rr, mask in zip(interior_rows, chosen):
                for j, bit in enumerate(mask):
                    if bit:
                        grid[rr, c0 + 1 + j] = GREY
        if not ok:
            continue
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid fingerprint-frame puzzle")
