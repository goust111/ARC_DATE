import random
from typing import Tuple

import numpy as np


AZURE = 8
BLACK = 0


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("expected at least one marked cell")
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _target_bbox(grid: np.ndarray) -> Tuple[int, int, int, int]:
    r0, r1, c0, c1 = _bbox(grid == AZURE)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("target frame must enclose a non-empty interior")
    border = np.zeros_like(grid, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    if not np.all(grid[border] == AZURE):
        raise ValueError("azure cells must form a rectangular target frame")
    return r0, r1, c0, c1


def _seed_bbox(grid: np.ndarray, target: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    tr0, tr1, tc0, tc1 = target
    seed_mask = (grid != BLACK) & (grid != AZURE)
    seed_mask[tr0 : tr1 + 1, tc0 : tc1 + 1] = False
    return _bbox(seed_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the seed into the framed target using the seed's reflection symmetry.

    Input structure:
    A black grid contains one small colored rectangular seed pattern on the left
    and one larger empty target rectangle on the right. The target rectangle is
    marked by an azure border, and its black interior has height and width that
    are integer multiples of the seed height and width.

    Rule:
    Find the seed pattern and the azure target frame. If the seed is left-right
    symmetric, fill alternating tile columns with the original seed and its
    horizontal mirror. Otherwise, if the seed is top-bottom symmetric, fill
    alternating tile rows with the original seed and its vertical mirror. If the
    seed has neither reflection symmetry, fill the target interior in a
    checkerboard pattern alternating between the original seed and its 180 degree
    rotation. The azure frame is preserved.

    Key insight:
    The tiling operation is not indicated by a marker in the target; it is chosen
    by first classifying the visible symmetry of the seed pattern.
    """
    out = np.array(grid, copy=True)
    tr0, tr1, tc0, tc1 = _target_bbox(out)
    sr0, sr1, sc0, sc1 = _seed_bbox(out, (tr0, tr1, tc0, tc1))
    seed = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    sh, sw = seed.shape
    ih, iw = tr1 - tr0 - 1, tc1 - tc0 - 1
    if ih % sh or iw % sw:
        raise ValueError("target interior must be an integer tiling of the seed")

    left_right = np.array_equal(seed, np.fliplr(seed))
    top_bottom = np.array_equal(seed, np.flipud(seed))
    rows, cols = ih // sh, iw // sw

    for rr in range(rows):
        for cc in range(cols):
            tile = seed
            if left_right:
                if cc % 2 == 1:
                    tile = np.fliplr(seed)
            elif top_bottom:
                if rr % 2 == 1:
                    tile = np.flipud(seed)
            elif (rr + cc) % 2 == 1:
                tile = np.rot90(seed, 2)
            r = tr0 + 1 + rr * sh
            c = tc0 + 1 + cc * sw
            out[r : r + sh, c : c + sw] = tile
    return out


def _has_colored_edges(pattern: np.ndarray) -> bool:
    return (
        np.any(pattern[0] != BLACK)
        and np.any(pattern[-1] != BLACK)
        and np.any(pattern[:, 0] != BLACK)
        and np.any(pattern[:, -1] != BLACK)
    )


def _make_seed(kind: str, height: int, width: int, colors: list[int]) -> np.ndarray:
    for _ in range(1000):
        base = np.random.choice(colors + [BLACK], size=(height, width), p=[0.24, 0.24, 0.24, 0.28])
        if kind == "horizontal":
            for c in range(width // 2):
                base[:, width - 1 - c] = base[:, c]
            if not np.array_equal(base, np.fliplr(base)) or np.array_equal(base, np.flipud(base)):
                continue
        elif kind == "vertical":
            for r in range(height // 2):
                base[height - 1 - r, :] = base[r, :]
            if not np.array_equal(base, np.flipud(base)) or np.array_equal(base, np.fliplr(base)):
                continue
        else:
            if np.array_equal(base, np.fliplr(base)) or np.array_equal(base, np.flipud(base)):
                continue
        if _has_colored_edges(base):
            return base.astype(int)

    fallback = np.zeros((height, width), dtype=int)
    a, b, c = colors
    if kind == "horizontal":
        fallback[:, 0] = a
        fallback[:, -1] = a
        fallback[0, width // 2] = b
        fallback[-1, width // 2] = c
        if height > 2:
            fallback[1, width // 2] = b
    elif kind == "vertical":
        fallback[0, :] = a
        fallback[-1, :] = a
        fallback[height // 2, 0] = b
        fallback[height // 2, -1] = c
        if width > 2:
            fallback[height // 2, 1] = b
    else:
        fallback[0, 0] = a
        fallback[0, -1] = b
        fallback[-1, 0] = c
        fallback[-1, -1] = a
        fallback[height // 2, width // 2] = b
    return fallback


def generate_input() -> np.ndarray:
    kind = random.choice(["horizontal", "vertical", "rotated"])
    seed_h = random.randint(3, 5)
    seed_w = random.randint(3, 5)
    tile_rows = random.randint(2, 3)
    tile_cols = random.randint(2, 3)

    palette = [1, 2, 3, 4, 5, 6, 7, 9]
    colors = random.sample(palette, 3)
    seed = _make_seed(kind, seed_h, seed_w, colors)

    target_h = seed_h * tile_rows + 2
    target_w = seed_w * tile_cols + 2
    gap = random.randint(2, 4)
    margin_l = random.randint(1, 3)
    margin_r = random.randint(1, 3)
    total_w = margin_l + seed_w + gap + target_w + margin_r
    total_h = max(seed_h, target_h) + random.randint(2, 6)
    total_w = min(total_w, 30)
    total_h = min(total_h, 30)

    grid = np.zeros((total_h, total_w), dtype=int)
    seed_r = random.randint(0, total_h - seed_h)
    seed_c = margin_l
    target_c = seed_c + seed_w + gap
    if target_c + target_w > total_w:
        target_c = total_w - target_w
        seed_c = max(0, target_c - gap - seed_w)
    target_r = random.randint(0, total_h - target_h)

    grid[seed_r : seed_r + seed_h, seed_c : seed_c + seed_w] = seed
    r0, r1 = target_r, target_r + target_h - 1
    c0, c1 = target_c, target_c + target_w - 1
    grid[r0, c0 : c1 + 1] = AZURE
    grid[r1, c0 : c1 + 1] = AZURE
    grid[r0 : r1 + 1, c0] = AZURE
    grid[r0 : r1 + 1, c1] = AZURE
    return grid
