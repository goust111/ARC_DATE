from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GRID_SIZE = 30
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _extract_sprites(grid: np.ndarray) -> list[np.ndarray]:
    groups: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for coords in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        sprite = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sprite.shape
        if int(coords.shape[0]) != h * w:
            continue
        groups.setdefault((int(h), int(w)), []).append((int(c0), int(r0), sprite.copy()))

    assert groups, "expected rectangular sprite components"

    best_shape, best_group = max(
        groups.items(),
        key=lambda item: (len(item[1]), item[0][0] * item[0][1]),
    )
    assert len(best_group) >= 3, "expected at least three equal-size sprites"

    h, w = best_shape
    assert h > 0 and w > 0
    best_group.sort(key=lambda item: (item[0], item[1]))
    return [sprite for _c0, _r0, sprite in best_group]


def _special_colors(sprites: np.ndarray) -> list[int]:
    n, h, w = sprites.shape
    special: list[int | None] = [None] * n

    for r in range(h):
        for c in range(w):
            values = sprites[:, r, c]
            colors, counts = np.unique(values, return_counts=True)
            if colors.size == 1:
                continue

            order = np.argsort(counts)
            counts = counts[order]
            colors = colors[order]
            assert colors.size == 2, "each varying position must have one majority and one outlier"
            assert int(counts[0]) == 1 and int(counts[1]) == n - 1, "expected a single outlier"

            outlier_color = int(colors[0])
            outlier_indices = np.flatnonzero(values == outlier_color)
            assert outlier_indices.size == 1
            idx = int(outlier_indices[0])
            assert special[idx] is None, "each sprite must contribute exactly one outlier pixel"
            special[idx] = outlier_color

    assert all(color is not None for color in special), "every sprite needs one special pixel"
    return [int(color) for color in special]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare repeated sprites and extract the color of each sprite's unique outlier pixel.

    Input structure:
    - The background is black.
    - Several non-overlapping rectangular sprite regions of the same size are laid out left to right.
    - The sprites share a common base pattern, but each sprite has exactly one pixel whose color differs
      from the majority color seen at that internal coordinate across the sprite set.

    Rule:
    - Align the sprites by their shared rectangle size.
    - For each internal coordinate, find whether exactly one sprite differs from the majority.
    - Each sprite has exactly one such outlier coordinate.
    - Output a 1 x N row whose entries are the outlier colors, ordered left to right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprites = _extract_sprites(g)
    stack = np.stack(sprites, axis=0)
    colors = _special_colors(stack)
    return np.asarray([colors], dtype=int)


def _build_base_sprite(height: int, width: int, palette: list[int]) -> np.ndarray:
    sprite = np.full((height, width), int(palette[0]), dtype=int)
    mode = random.choice(("cross", "bands", "blocks"))

    if mode == "cross":
        band_h = random.randint(1, height - 1)
        band_w = random.randint(1, width - 1)
        r0 = random.randint(0, height - band_h)
        c0 = random.randint(0, width - band_w)
        sprite[r0 : r0 + band_h, :] = int(palette[1])
        sprite[:, c0 : c0 + band_w] = int(palette[2])
        if height >= 3 and width >= 3:
            inner_h = random.randint(1, max(1, height // 2))
            inner_w = random.randint(1, max(1, width // 2))
            ir = random.randint(0, height - inner_h)
            ic = random.randint(0, width - inner_w)
            sprite[ir : ir + inner_h, ic : ic + inner_w] = int(palette[3])
    elif mode == "bands":
        cut_r = random.randint(1, height - 1)
        cut_c = random.randint(1, width - 1)
        sprite[:cut_r, :] = int(palette[1])
        sprite[:, :cut_c] = int(palette[2])
        block_h = random.randint(1, max(1, height - cut_r))
        block_w = random.randint(1, max(1, width - cut_c))
        br = random.randint(cut_r - 1, height - block_h)
        bc = random.randint(cut_c - 1, width - block_w)
        sprite[br : br + block_h, bc : bc + block_w] = int(palette[3])
    else:
        h1 = random.randint(1, height - 1)
        w1 = random.randint(1, width - 1)
        sprite[:h1, :w1] = int(palette[1])

        h2 = random.randint(1, height - 1)
        w2 = random.randint(1, width - 1)
        r2 = random.randint(height - h2, height - 1)
        c2 = random.randint(width - w2, width - 1)
        sprite[r2 - h2 + 1 : r2 + 1, c2 - w2 + 1 : c2 + 1] = int(palette[2])

        if height >= 4 and width >= 4:
            mid_h = random.randint(1, height - 2)
            mid_w = random.randint(1, width - 2)
            mr = random.randint(1, height - mid_h)
            mc = random.randint(1, width - mid_w)
            sprite[mr : mr + mid_h, mc : mc + mid_w] = int(palette[3])

    if np.unique(sprite).size < 3:
        sprite[0, :] = int(palette[1])
        sprite[:, 0] = int(palette[2])
        sprite[-1, -1] = int(palette[3])
    return sprite


def _layout_options(n_sprites: int) -> list[tuple[int, int, int, int, int, int]]:
    options: list[tuple[int, int, int, int, int, int]] = []
    for height in range(4, 9):
        for width in range(4, 9):
            if height * width < n_sprites:
                continue
            for gap in range(1, 5):
                for top in range(1, 5):
                    for bottom in range(1, 5):
                        for left in range(1, 5):
                            for right in range(1, 5):
                                total_h = top + height + bottom
                                total_w = left + n_sprites * width + (n_sprites - 1) * gap + right
                                if total_h <= MAX_GRID_SIZE and total_w <= MAX_GRID_SIZE:
                                    options.append((height, width, gap, top, left, total_w))
    return options


def _choose_special_positions(height: int, width: int, n_sprites: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    return random.sample(cells, n_sprites)


def _render_instance() -> tuple[np.ndarray, np.ndarray]:
    n_sprites = random.randint(3, 5)
    layout = random.choice(_layout_options(n_sprites))
    height, width, gap, top, left, total_w = layout
    bottom = random.randint(1, MAX_GRID_SIZE - top - height)
    total_h = top + height + bottom
    total_w = max(total_w, left + n_sprites * width + (n_sprites - 1) * gap + 1)

    palette = random.sample(COLOR_POOL, random.randint(4, 6))
    base = _build_base_sprite(height, width, palette)
    special_positions = _choose_special_positions(height, width, n_sprites)

    sprites = []
    out_colors: list[int] = []
    for idx, (r, c) in enumerate(special_positions):
        sprite = base.copy()
        base_color = int(sprite[r, c])
        alternatives = [color for color in palette if color != base_color]
        if not alternatives:
            alternatives = [color for color in COLOR_POOL if color != base_color]
        special_color = random.choice(alternatives)
        sprite[r, c] = int(special_color)
        sprites.append(sprite)
        out_colors.append(int(special_color))

    grid = np.zeros((total_h, total_w), dtype=int)
    col = left
    for sprite in sprites:
        grid[top : top + height, col : col + width] = sprite
        col += width + gap

    expected = np.asarray([out_colors], dtype=int)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    grid, expected = _render_instance()
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
