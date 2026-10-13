from __future__ import annotations

import numpy as np


BACKGROUND = 0
SIZE_OPTIONS = ((4, 4), (4, 4), (2, 4), (4, 2))
PALETTE = np.arange(1, 10, dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _orientation_variants(sprite: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(sprite, dtype=int)
    lr = np.fliplr(base)
    ud = np.flipud(base)
    return base, lr, ud, np.flipud(lr)


def _quadrant_index(r: int, c: int, h: int, w: int) -> int:
    return 2 * int(r >= h // 2) + int(c >= w // 2)


def _stamp_with_mode(sprite: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(sprite, dtype=int)
    h, w = g.shape
    out = np.zeros((h * h, w * w), dtype=int)
    variants = _orientation_variants(g)

    for r in range(h):
        rr = slice(r * h, (r + 1) * h)
        for c in range(w):
            if int(g[r, c]) == BACKGROUND:
                continue
            if mode == "quadrant":
                idx = _quadrant_index(r, c, h, w)
            elif mode == "row":
                idx = 2 * int(r >= h // 2)
            elif mode == "col":
                idx = int(c >= w // 2)
            else:
                idx = int(mode)
            out[rr, c * w : (c + 1) * w] = variants[idx]
    return out


def _quadrant_cells(h: int, w: int) -> list[list[tuple[int, int]]]:
    r_mid = h // 2
    c_mid = w // 2
    bounds = (
        (0, r_mid, 0, c_mid),
        (0, r_mid, c_mid, w),
        (r_mid, h, 0, c_mid),
        (r_mid, h, c_mid, w),
    )
    return [[(r, c) for r in range(r0, r1) for c in range(c0, c1)] for r0, r1, c0, c1 in bounds]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    neighbors = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            neighbors.append((nr, nc))
    return neighbors


def _has_adjacent_foreground(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c, h, w):
                if mask[nr, nc]:
                    return True
    return False


def _is_good_example(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False
    h, w = g.shape
    if h == 0 or w == 0 or h % 2 or w % 2:
        return False
    if h * h > 30 or w * w > 30:
        return False

    fg_mask = g != BACKGROUND
    fg_count = int(fg_mask.sum())
    if fg_count < 4 or fg_count > h * w - 2:
        return False
    if not _has_adjacent_foreground(fg_mask):
        return False

    colors = np.unique(g[fg_mask])
    if colors.size < 2:
        return False

    for cells in _quadrant_cells(h, w):
        if not any(int(g[r, c]) != BACKGROUND for r, c in cells):
            return False

    variants = _orientation_variants(g)
    if len({variant.tobytes() for variant in variants}) != 4:
        return False

    correct = _stamp_with_mode(g, "quadrant")
    simpler_modes = ("row", "col", "0", "1", "2", "3")
    if any(np.array_equal(correct, _stamp_with_mode(g, mode)) for mode in simpler_modes):
        return False

    return True


def _generate_candidate() -> np.ndarray:
    h, w = SIZE_OPTIONS[int(np.random.randint(len(SIZE_OPTIONS)))]
    max_nonblack = min(h * w - 2, int(np.ceil(0.7 * h * w)))
    target_count = int(np.random.randint(4, max_nonblack + 1))

    chosen: set[tuple[int, int]] = set()
    for cells in _quadrant_cells(h, w):
        chosen.add(cells[int(np.random.randint(len(cells)))])

    seed_r, seed_c = list(chosen)[int(np.random.randint(len(chosen)))]
    seed_neighbors = [cell for cell in _neighbors4(seed_r, seed_c, h, w) if cell not in chosen]
    if seed_neighbors and len(chosen) < target_count:
        chosen.add(seed_neighbors[int(np.random.randint(len(seed_neighbors)))])

    remaining = [(r, c) for r in range(h) for c in range(w) if (r, c) not in chosen]
    np.random.shuffle(remaining)
    while len(chosen) < target_count and remaining:
        chosen.add(remaining.pop())

    chosen_cells = list(chosen)
    np.random.shuffle(chosen_cells)

    palette_size = int(np.random.randint(2, min(4, len(chosen_cells)) + 1))
    palette = np.random.choice(PALETTE, size=palette_size, replace=False)

    grid = np.zeros((h, w), dtype=int)
    for idx, (r, c) in enumerate(chosen_cells):
        if idx < palette_size:
            color = int(palette[idx])
        else:
            color = int(palette[int(np.random.randint(palette_size))])
        grid[r, c] = color

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the sprite into a block stamp pattern whose copy orientation depends on each source cell's quadrant.

    Input structure:
    - The input is a small even-height, even-width sprite on a black background.
    - Every input cell corresponds to one output block with the same height and width as the sprite.

    Rule:
    - Create an output of shape (input_height^2, input_width^2) and partition it into input-sized blocks.
    - If an input cell is black, leave its block black.
    - If an input cell is non-black, stamp the whole sprite into that block.
    - Choose the stamped sprite orientation from the source cell's quadrant relative to the sprite center:
      upper-left uses the original sprite, upper-right uses a left-right mirror,
      lower-left uses an up-down mirror, and lower-right uses both mirrors.

    Key insight:
    - This is self-stamping like 007bbfb7, but each pasted copy must be mirrored from the
      source cell's quadrant, so the quadrant reference frame is essential.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2-D grid"
    h, w = g.shape
    assert h > 0 and w > 0 and h % 2 == 0 and w % 2 == 0, "expected an even-sized sprite"
    assert h * h <= 30 and w * w <= 30, "output must stay within ARC size limits"
    return _stamp_with_mode(g, "quadrant")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate()
        if _is_good_example(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid quadrant-reflection stamp example")
