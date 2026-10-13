from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coord_array = np.array(coords, dtype=int)
            r0 = int(coord_array[:, 0].min())
            c0 = int(coord_array[:, 1].min())
            r1 = int(coord_array[:, 0].max())
            c1 = int(coord_array[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )
    return components


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _stamp_mask(target: np.ndarray, sprite_mask: np.ndarray, top: int, left: int) -> None:
    h, w = target.shape
    sh, sw = sprite_mask.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h, top + sh)
    c1 = min(w, left + sw)
    if r0 >= r1 or c0 >= c1:
        return
    sub = sprite_mask[r0 - top : r1 - top, c0 - left : c1 - left]
    target[r0:r1, c0:c1] |= sub


def _tile_sprite_mask(
    grid_shape: tuple[int, int],
    sprite_mask: np.ndarray,
    origin: tuple[int, int],
    step: tuple[int, int],
) -> np.ndarray:
    h, w = grid_shape
    sh, sw = sprite_mask.shape
    dr, dc = step
    r, c = origin
    out = np.zeros((h, w), dtype=bool)

    while r < h and c < w and r + sh - 1 >= 0 and c + sw - 1 >= 0:
        _stamp_mask(out, sprite_mask, r, c)
        r += dr
        c += dc
    return out


def _classify_components(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    h, w = grid.shape
    bars = []
    sprites = []
    for comp in _find_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        ch = r1 - r0 + 1
        cw = c1 - c0 + 1
        touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        solid_line = ((ch == 1) ^ (cw == 1)) and comp["area"] == ch * cw
        if touches_edge and solid_line:
            bars.append(comp)
        else:
            sprites.append(comp)
    return bars, sprites


def _pair_sprite_with_bar(
    sprite: dict,
    bars: list[dict],
    horizontal_bars: bool,
) -> dict:
    sr0, sc0, sr1, sc1 = sprite["bbox"]
    if horizontal_bars:
        matches = [bar for bar in bars if bar["bbox"][1] == sc0 and bar["bbox"][3] == sc1]
    else:
        matches = [bar for bar in bars if bar["bbox"][0] == sr0 and bar["bbox"][2] == sr1]
    assert len(matches) == 1
    return matches[0]


def _solve_masks(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    bars, sprites = _classify_components(g, bg)
    assert len(bars) == 2
    assert len(sprites) == 2

    horizontal_bars = all(bar["bbox"][0] == bar["bbox"][2] for bar in bars)
    vertical_bars = all(bar["bbox"][1] == bar["bbox"][3] for bar in bars)
    assert horizontal_bars ^ vertical_bars

    if horizontal_bars:
        assert {bar["bbox"][0] for bar in bars} == {0, g.shape[0] - 1}
    else:
        assert {bar["bbox"][1] for bar in bars} == {0, g.shape[1] - 1}

    tilings = []
    for sprite in sprites:
        paired_bar = _pair_sprite_with_bar(sprite, bars, horizontal_bars)
        sr0, sc0, sr1, sc1 = sprite["bbox"]
        sprite_crop = _crop(g, sprite["bbox"])
        sprite_mask = sprite_crop != bg
        sh, sw = sprite_mask.shape

        if horizontal_bars:
            br = paired_bar["bbox"][0]
            step = (-sh, 0) if br == 0 else (sh, 0)
        else:
            bc = paired_bar["bbox"][1]
            step = (0, -sw) if bc == 0 else (0, sw)

        tilings.append(
            (
                _tile_sprite_mask(g.shape, sprite_mask, (sr0, sc0), step),
                int(sprite["color"]),
            )
        )

    return tilings[0][0], tilings[0][1], tilings[1][0], tilings[1][1], bg


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the opposed edge bars, tile each matched sprite toward its own bar, and cancel the overlaps.

    Input structure:
    - The grid has a dominant background, exactly two monochromatic sprites, and exactly two one-cell-thick control bars on opposite edges.
    - The two bars share the same orientation: either horizontal on the top and bottom edges, or vertical on the left and right edges.
    - Each sprite has the same span as exactly one opposite-edge bar on the perpendicular axis, so a horizontal bar matches one sprite's column span and a vertical bar matches one sprite's row span.

    Rule:
    - Remove both control bars.
    - For each sprite, repeatedly place its own bbox crop from the original location toward the matched bar, stepping by one full sprite height or width; background cells inside the sprite stay transparent.
    - Treat the two tiled sprite families, including the originals, as two sets of painted cells: keep cells belonging to exactly one family in that sprite's color, and set cells reached by both families back to background.

    Key insight:
    - The bars are only direction markers; the real transformation is the span-based pairing from each sprite to the opposite edge, followed by a symmetric-difference combination of the two opposed tilings.
    """
    mask_a, color_a, mask_b, color_b, bg = _solve_masks(grid)
    out = np.full(np.asarray(grid).shape, bg, dtype=int)
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a
    out[only_a] = color_a
    out[only_b] = color_b
    return out


_GEN_RETRY_LIMIT = 400
_GEN_BG = 0
_GEN_BAR_COLOR = 1
_GEN_SPRITE_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)


def _random_sprite_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if height >= 3 and width >= 3:
        variant = random.choice(("frame", "holes", "slit"))
        if variant == "frame":
            mask[1:-1, 1:-1] = False
            if height >= 5 and width >= 5 and random.random() < 0.6:
                rr = random.randint(1, height - 2)
                cc = random.randint(1, width - 2)
                mask[rr, cc] = True
        elif variant == "holes":
            interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
            random.shuffle(interior)
            hole_count = random.randint(1, max(1, min(len(interior), (height * width) // 5)))
            for r, c in interior[:hole_count]:
                mask[r, c] = False
        else:
            if height >= width and height >= 4:
                rr = random.randint(1, height - 2)
                mask[rr, 1:-1] = False
                mask[rr, random.randint(1, width - 2)] = True
            elif width >= 4:
                cc = random.randint(1, width - 2)
                mask[1:-1, cc] = False
                mask[random.randint(1, height - 2), cc] = True
            else:
                mask[1:-1, 1:-1] = False
    return mask


def _sprite_array(height: int, width: int, color: int) -> np.ndarray:
    mask = _random_sprite_mask(height, width)
    sprite = np.full((height, width), _GEN_BG, dtype=int)
    sprite[mask] = color
    return sprite


def _paste_sprite(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    sh, sw = sprite.shape
    patch = sprite != _GEN_BG
    canvas[top : top + sh, left : left + sw][patch] = sprite[patch]


def _build_input(
    grid_shape: tuple[int, int],
    orientation: str,
    sprite_a: np.ndarray,
    sprite_b: np.ndarray,
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
) -> np.ndarray:
    h, w = grid_shape
    grid = np.full((h, w), _GEN_BG, dtype=int)
    _paste_sprite(grid, sprite_a, pos_a[0], pos_a[1])
    _paste_sprite(grid, sprite_b, pos_b[0], pos_b[1])

    if orientation == "tb":
        _, left_a = pos_a
        _, left_b = pos_b
        grid[h - 1, left_a : left_a + sprite_a.shape[1]] = _GEN_BAR_COLOR
        grid[0, left_b : left_b + sprite_b.shape[1]] = _GEN_BAR_COLOR
    else:
        top_a, _ = pos_a
        top_b, _ = pos_b
        grid[top_a : top_a + sprite_a.shape[0], w - 1] = _GEN_BAR_COLOR
        grid[top_b : top_b + sprite_b.shape[0], 0] = _GEN_BAR_COLOR
    return grid


def _choose_column_starts(width: int, w_a: int, w_b: int) -> tuple[int, int] | None:
    max_overlap = min(w_a - 1, w_b - 1, 3)
    if max_overlap < 1:
        return None
    overlap = random.randint(1, max_overlap)
    total = w_a + w_b - overlap
    if total > width - 2:
        return None
    left = random.randint(1, width - total - 1)
    if random.random() < 0.5:
        return left, left + w_a - overlap
    return left + w_b - overlap, left


def _choose_row_starts(height: int, h_a: int, h_b: int) -> tuple[int, int] | None:
    max_overlap = min(h_a - 1, h_b - 1, 3)
    if max_overlap < 1:
        return None
    overlap = random.randint(1, max_overlap)
    total = h_a + h_b - overlap
    if total > height - 2:
        return None
    top = random.randint(1, height - total - 1)
    if random.random() < 0.5:
        return top, top + h_a - overlap
    return top + h_b - overlap, top


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("tb", "lr"))
    grid_h = random.randint(14, 24)
    grid_w = random.randint(14, 24)
    color_a, color_b = random.sample(_GEN_SPRITE_COLORS, 2)

    if orientation == "tb":
        sprite_h_a = random.randint(3, 5)
        sprite_w_a = random.randint(3, 6)
        sprite_h_b = random.randint(3, 5)
        sprite_w_b = random.randint(3, 6)
        col_starts = _choose_column_starts(grid_w, sprite_w_a, sprite_w_b)
        if col_starts is None:
            return None

        top_a = random.randint(1, max(1, grid_h // 3))
        bottom_limit = grid_h - sprite_h_b - 2
        bottom_start = max(grid_h // 2, top_a + sprite_h_a + 2)
        if bottom_start > bottom_limit:
            return None
        top_b = random.randint(bottom_start, bottom_limit)

        sprite_a = _sprite_array(sprite_h_a, sprite_w_a, color_a)
        sprite_b = _sprite_array(sprite_h_b, sprite_w_b, color_b)
        grid = _build_input(
            (grid_h, grid_w),
            orientation,
            sprite_a,
            sprite_b,
            (top_a, col_starts[0]),
            (top_b, col_starts[1]),
        )
    else:
        sprite_h_a = random.randint(3, 6)
        sprite_w_a = random.randint(3, 5)
        sprite_h_b = random.randint(3, 6)
        sprite_w_b = random.randint(3, 5)
        row_starts = _choose_row_starts(grid_h, sprite_h_a, sprite_h_b)
        if row_starts is None:
            return None

        left_a = random.randint(1, max(1, grid_w // 3))
        right_limit = grid_w - sprite_w_b - 2
        right_start = max(grid_w // 2, left_a + sprite_w_a + 2)
        if right_start > right_limit:
            return None
        left_b = random.randint(right_start, right_limit)

        sprite_a = _sprite_array(sprite_h_a, sprite_w_a, color_a)
        sprite_b = _sprite_array(sprite_h_b, sprite_w_b, color_b)
        grid = _build_input(
            (grid_h, grid_w),
            orientation,
            sprite_a,
            sprite_b,
            (row_starts[0], left_a),
            (row_starts[1], left_b),
        )

    try:
        mask_a, color_a, mask_b, color_b, bg = _solve_masks(grid)
    except AssertionError:
        return None

    overlap = mask_a & mask_b
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a
    if not overlap.any() or not only_a.any() or not only_b.any():
        return None

    out = np.full(grid.shape, bg, dtype=int)
    out[only_a] = color_a
    out[only_b] = color_b
    if np.array_equal(grid, out):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opposed tiling cancellation")
