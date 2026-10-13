from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
ORIENTATIONS = ("tl", "tr", "bl", "br")
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 400
PLACEMENT_RETRIES = 2000
HALF_SIZE_CHOICES = (5, 7, 9, 11, 13)

SPRITE_MASKS = {
    "tl": np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "tr": np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "bl": np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    "br": np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_sprites(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    sprites: list[dict[str, int]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)

            assert len(coords) == 8, "each sprite must contain exactly eight pixels"
            assert r1 - r0 == 2 and c1 - c0 == 2, "each sprite must occupy a 3x3 box"

            local = np.zeros((3, 3), dtype=bool)
            for rr, cc in coords:
                local[rr - r0, cc - c0] = True

            orient = None
            for name, mask in SPRITE_MASKS.items():
                if np.array_equal(local, mask):
                    orient = name
                    break
            assert orient is not None, "foreground objects must be 3x3 corner-missing sprites"

            sprites.append({"row": r0, "col": c0, "color": color, "orient": orient})

    sprites.sort(key=lambda item: (item["row"], item["col"]))
    return sprites


def _quadrant_anchor(shape: tuple[int, int], orient: str) -> tuple[int, int]:
    h, w = shape
    assert h % 2 == 0 and w % 2 == 0, "grid must split into four equal quadrants"
    qh, qw = h // 2, w // 2
    assert qh >= 3 and qw >= 3, "quadrants must be large enough for a 3x3 sprite"

    r0 = 0 if orient in ("tl", "tr") else qh
    c0 = 0 if orient in ("tl", "bl") else qw
    center_r = r0 + qh // 2
    center_c = c0 + qw // 2
    return center_r - 1, center_c - 1


def _render_output(shape: tuple[int, int], ordered_sprites: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for sprite in ordered_sprites:
        top, left = _quadrant_anchor(shape, sprite["orient"])
        view = out[top : top + 3, left : left + 3]
        view[SPRITE_MASKS[sprite["orient"]]] = int(sprite["color"])
    return out


def _place_input_sprite(grid: np.ndarray, sprite: dict[str, int]) -> None:
    top, left = int(sprite["row"]), int(sprite["col"])
    view = grid[top : top + 3, left : left + 3]
    view[SPRITE_MASKS[sprite["orient"]]] = int(sprite["color"])


def _input_quadrant(sprite: dict[str, int], shape: tuple[int, int]) -> str:
    h, w = shape
    center_r = int(sprite["row"]) + 1
    center_c = int(sprite["col"]) + 1
    top = center_r < h // 2
    left = center_c < w // 2
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group corner-missing sprites by the direction of their missing corner and merge each group into the matching quadrant.

    Input structure:
    - The background is black.
    - Several monochromatic sprites are scattered on the grid.
    - Each sprite occupies a 3x3 box and contains exactly eight pixels: one of the four corners is missing.

    Rule:
    - Clear the output grid to black while keeping the same size as the input.
    - Split the grid into four equal quadrants: top-left, top-right, bottom-left, and bottom-right.
    - For each sprite, detect which corner is missing and assign it to the quadrant with the same corner direction.
    - Within each quadrant, overlay that group's sprites with their 3x3 masks centered on the quadrant center, drawing sprites in top-to-bottom then left-to-right input order so later sprites overwrite earlier ones.

    Key insight:
    - The missing corner is both the grouping label and the destination cue, while the output color in each quadrant is decided by overlay order among same-orientation sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sprites = _extract_sprites(g)
    assert sprites, "expected at least one sprite"
    return _render_output(g.shape, sprites)


def _gen_attempt() -> np.ndarray | None:
    half_h = random.choice(HALF_SIZE_CHOICES)
    half_w = random.choice(HALF_SIZE_CHOICES)
    h, w = 2 * half_h, 2 * half_w

    max_sprites = max(4, min(10, (h * w) // 25))
    total = random.randint(4, max_sprites)

    palette_size = random.randint(3, min(7, total))
    palette = random.sample(COLOR_POOL, palette_size)

    orientations = list(ORIENTATIONS)
    for _ in range(total - 4):
        orientations.append(random.choice(ORIENTATIONS))
    random.shuffle(orientations)

    blocked = np.zeros((h, w), dtype=bool)
    sprites: list[dict[str, int]] = []
    for orient in orientations:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            row = random.randint(0, h - 3)
            col = random.randint(0, w - 3)
            if blocked[row : row + 3, col : col + 3].any():
                continue
            sprites.append(
                {
                    "row": row,
                    "col": col,
                    "color": random.choice(palette),
                    "orient": orient,
                }
            )
            blocked[max(0, row - 1) : min(h, row + 4), max(0, col - 1) : min(w, col + 4)] = True
            placed = True
            break
        if not placed:
            return None

    sprites.sort(key=lambda item: (item["row"], item["col"]))

    if all(_input_quadrant(sprite, (h, w)) == sprite["orient"] for sprite in sprites):
        return None

    counts = Counter(sprite["orient"] for sprite in sprites)
    if any(counts[orient] > 1 for orient in ORIENTATIONS):
        if not any(
            len({sprite["color"] for sprite in sprites if sprite["orient"] == orient}) > 1
            for orient in ORIENTATIONS
            if counts[orient] > 1
        ):
            return None

    grid = np.full((h, w), BG, dtype=int)
    for sprite in sprites:
        _place_input_sprite(grid, sprite)

    extracted = _extract_sprites(grid)
    if extracted != sprites:
        return None

    out = _render_output((h, w), extracted)
    if np.array_equal(grid, out):
        return None

    visible_colors = {int(v) for v in np.unique(out) if int(v) != BG}
    if len(visible_colors) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid corner-missing sprite input")
