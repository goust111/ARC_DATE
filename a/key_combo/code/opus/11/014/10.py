from __future__ import annotations

import random

import numpy as np


BG = 0
CENTER_COLOR = 1
SPRITE_COLOR = 2
GEN_RETRIES = 400
EXACT_DIMS = ((3, 3), (4, 2), (6, 2))
FALLBACK_DIMS = ((4, 3), (5, 2), (5, 3))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flat))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _resize_nearest(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = map(int, grid.shape)
    row_idx = (np.arange(out_h) * in_h) // out_h
    col_idx = (np.arange(out_w) * in_w) // out_w
    return grid[row_idx[:, None], col_idx[None, :]]


def _center_sprite(sprite: np.ndarray, bg: int) -> np.ndarray:
    h, w = map(int, sprite.shape)
    if h % w == 0:
        scaled = np.repeat(sprite, h // w, axis=1)
    else:
        scaled = _resize_nearest(sprite, h, h)

    center = np.full((h, h), bg, dtype=int)
    center[scaled != bg] = CENTER_COLOR
    return center


def _ring_output(sprite: np.ndarray, bg: int) -> np.ndarray:
    h, w = map(int, sprite.shape)
    out = np.full((h + 2 * w, h + 2 * w), bg, dtype=int)
    _paste(out, np.rot90(sprite, -1), 0, w)
    _paste(out, sprite, w, 0)
    _paste(out, np.fliplr(sprite), w, w + h)
    _paste(out, np.rot90(sprite, 1), w + h, w)
    _paste(out, _center_sprite(sprite, bg), w, w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop one foreground sprite, build the four-way ring around it, and fill the center with a square resize of the same sprite.

    Input structure:
    - The input is a black-dominant grid containing exactly one non-background color.
    - The colored pixels form one asymmetric connected sprite away from the grid border.
    - The sprite is interpreted through its tight bounding box, preserving any background holes inside that box.

    Rule:
    - Crop the sprite to its tight bounding box with height h and width w.
    - Create a square output of side h + 2*w and place four oriented copies around the center:
      clockwise rotation on top, original on the left, horizontal mirror on the right, and counterclockwise rotation below.
    - The center hole is an h x h square.
    - Fill that center with a blue square resize of the original sprite: if h is an integer multiple of w, use exact column replication; otherwise use nearest-neighbor resizing to h x h.

    Key insight:
    - The same cropped sprite serves two roles at once: it creates the outer ring by rotation/reflection, and it also generates the inner image by resizing itself to the hole produced by that ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    fg_colors = _colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    box = _bbox(g != bg)
    assert box is not None, "expected a non-empty sprite"
    sprite = _crop_bbox(g, box)
    return _ring_output(sprite, bg)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _random_monotone_path(h: int, w: int) -> set[tuple[int, int]]:
    starts_and_ends = (
        ((0, 0), (h - 1, w - 1)),
        ((0, w - 1), (h - 1, 0)),
        ((h - 1, 0), (0, w - 1)),
        ((h - 1, w - 1), (0, 0)),
    )
    (r, c), (tr, tc) = random.choice(starts_and_ends)
    cells = {(r, c)}
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        cells.add((r, c))
    return cells


def _mask_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=int)
    for r, c in cells:
        mask[r, c] = 1
    return mask


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _asymmetric(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.rot90(mask, 2)):
        return False
    return True


def _gen_sprite_mask(h: int, w: int, target_cells: int) -> np.ndarray | None:
    for _ in range(200):
        cells = set(_random_monotone_path(h, w))
        while len(cells) < target_cells:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = _mask_from_cells(cells, h, w)
        box = _bbox(mask.astype(bool))
        if box != (0, 0, h - 1, w - 1):
            continue
        if int(mask.sum()) != target_cells:
            continue
        if not _connected(mask):
            continue
        if not _asymmetric(mask):
            continue
        if np.all(mask == 1):
            continue
        return mask
    return None


def _embed_sprite(sprite: np.ndarray) -> np.ndarray:
    h, w = map(int, sprite.shape)
    rows = random.randint(h + 2, min(12, h + 5))
    cols = random.randint(w + 2, min(10, w + 5))
    grid = np.full((rows, cols), BG, dtype=int)
    r0 = random.randint(1, rows - h - 1)
    c0 = random.randint(1, cols - w - 1)
    _paste(grid, sprite, r0, c0)
    return grid


def _build_sprite(exact_branch: bool) -> np.ndarray | None:
    dims = random.choice(EXACT_DIMS if exact_branch else FALLBACK_DIMS)
    h, w = dims
    min_cells = max(4, h + w - 1)
    max_cells = min(7, h * w)
    if min_cells > max_cells:
        return None
    target_cells = random.randint(min_cells, max_cells)
    mask = _gen_sprite_mask(h, w, target_cells)
    if mask is None:
        return None
    return (mask * SPRITE_COLOR).astype(int)


def _valid_example(inp: np.ndarray, branch_is_exact: bool) -> bool:
    try:
        out = main(inp)
    except AssertionError:
        return False

    bg = _get_bg(inp)
    box = _bbox(inp != bg)
    if box is None:
        return False
    sprite = _crop_bbox(inp, box)
    h, w = sprite.shape
    if (h % w == 0) != branch_is_exact:
        return False
    expected = _ring_output(sprite, bg)
    if not np.array_equal(out, expected):
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    center = out[w : w + h, w : w + h]
    if not np.any(center == CENTER_COLOR):
        return False
    if np.array_equal(center, sprite):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    exact_branch = random.random() < 0.5
    sprite = _build_sprite(exact_branch)
    if sprite is None:
        return None
    inp = _embed_sprite(sprite)
    if _valid_example(inp, exact_branch):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
