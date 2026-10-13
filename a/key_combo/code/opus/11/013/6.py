from __future__ import annotations

import random

import numpy as np


BLACK = 0
BAR_COLORS = (1, 2, 3, 4)
RANK_COLORS = (7, 6, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                }
            )

    return components


def _paint(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Dock each monochrome sprite onto its same-colored bar, rank the docked sprites by height, and recolor each bar-group by rank.

    Input structure:
    - The grid has a black background, four long 1-cell-high horizontal color bars in the lower half, and four small monochrome sprites above them.
    - Each sprite color matches exactly one bar color, each sprite is 4-connected, and the four sprite bounding-box heights are all different.
    - Every sprite already lies horizontally inside the column span of its matching bar.

    Rule:
    - Identify the long horizontal bars and the small sprites, then pair them by equal color.
    - Move each sprite straight down so its bbox bottom becomes flush with the row just above its matched bar, keeping all columns unchanged, and erase the original sprite.
    - Rank the four sprites globally by bbox height from shortest to tallest.
    - Recolor each bar together with its docked sprite using rank 1 -> orange, rank 2 -> magenta, rank 3 -> teal, rank 4 -> maroon.

    Key insight:
    - Color only establishes which sprite docks to which bar; the final output color is decided later by a global comparison of sprite heights.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_monochrome_components(g, bg)
    width_threshold = max(6, g.shape[1] // 2)
    bars = [comp for comp in components if int(comp["height"]) == 1 and int(comp["width"]) >= width_threshold]
    sprites = [comp for comp in components if not (int(comp["height"]) == 1 and int(comp["width"]) >= width_threshold)]

    assert len(bars) == 4, "expected four horizontal bars"
    assert len(sprites) == 4, "expected four sprites"

    bars_by_color = {int(comp["color"]): comp for comp in bars}
    sprites_by_color = {int(comp["color"]): comp for comp in sprites}
    assert set(bars_by_color) == set(sprites_by_color) == set(BAR_COLORS)

    sprite_heights = sorted(int(comp["height"]) for comp in sprites)
    assert len(set(sprite_heights)) == 4, "sprite heights must be unique"
    height_to_rank_color = {height: RANK_COLORS[i] for i, height in enumerate(sprite_heights)}

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for color in BAR_COLORS:
        bar = bars_by_color[color]
        sprite = sprites_by_color[color]
        bar_r, bar_c0, _bar_r1, bar_c1 = bar["bbox"]
        sprite_r0, sprite_c0, sprite_r1, sprite_c1 = sprite["bbox"]
        assert bar_c0 <= sprite_c0 <= sprite_c1 <= bar_c1

        dr = int(bar_r - sprite_r1 - 1)
        moved = np.array(sprite["coords"], dtype=int).copy()
        moved[:, 0] += dr
        assert moved[:, 0].min() >= 0

        rank_color = height_to_rank_color[int(sprite["height"])]
        _paint(out, np.array(bar["coords"], dtype=int), rank_color)
        _paint(out, moved, rank_color)

    return out


SPRITE_SHAPES = {
    1: (
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1, 1, 1, 1]], dtype=int),
        np.array([[1, 1, 1, 1, 1]], dtype=int),
    ),
    2: (
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    ),
    3: (
        np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=int),
    ),
    4: (
        np.array([[1, 0], [1, 1], [1, 0], [1, 0]], dtype=int),
        np.array([[1, 1], [1, 0], [1, 1], [1, 0]], dtype=int),
        np.array([[1, 0], [1, 1], [0, 1], [0, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
    ),
    5: (
        np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]], dtype=int),
        np.array([[1, 0], [1, 1], [1, 0], [1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [1, 0], [1, 0], [1, 1], [0, 1]], dtype=int),
    ),
}

GRID_HEIGHT_RANGE = (20, 28)
GRID_WIDTH_RANGE = (14, 20)
GEN_ATTEMPTS = 400


def _random_sprite_mask(height: int) -> np.ndarray:
    mask = np.array(random.choice(SPRITE_SHAPES[height]), dtype=int)
    if random.choice((False, True)):
        mask = np.fliplr(mask)
    return mask


def _rect_clear(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _choose_gaps() -> list[int]:
    while True:
        gaps = [random.choice((2, 3)) for _ in range(3)]
        if 3 in gaps:
            return gaps


def _assign_heights_to_bars(gaps: list[int]) -> list[int]:
    top_height = random.choice((4, 5))
    lower = [1, 2, 3]
    while True:
        random.shuffle(lower)
        if all(lower[i] <= gaps[i] for i in range(3)):
            return [top_height, *lower]


def _place_sprites(
    top_limit: int, bars: list[dict[str, int]], sprites: list[dict[str, object]]
) -> list[tuple[int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []

    order = list(range(len(sprites)))
    random.shuffle(order)
    slots: list[tuple[int, int] | None] = [None] * len(sprites)

    for idx in order:
        sprite = sprites[idx]
        bar = bars[idx]
        mask = sprite["mask"]
        sh, sw = mask.shape
        min_c = int(bar["c0"])
        max_c = int(bar["c1"]) - sw + 1
        max_r = top_limit - sh
        if max_c < min_c or max_r < 0:
            return None

        candidates: list[tuple[int, int]] = []
        for r0 in range(max_r + 1):
            for c0 in range(min_c, max_c + 1):
                if _rect_clear(rects, r0, c0, sh, sw):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        rects.append((r0, c0, r0 + sh - 1, c0 + sw - 1))
        slots[idx] = (r0, c0)

    return [slot for slot in slots if slot is not None]


def _build_candidate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((height, width), dtype=int)

    gaps = _choose_gaps()
    sprite_heights = _assign_heights_to_bars(gaps)
    stack_height = 4 + sum(gaps)
    min_top_bar = max(sprite_heights[0], height // 2 - 3)
    max_top_bar = height - stack_height
    if min_top_bar > max_top_bar:
        return None
    top_bar_row = random.randint(min_top_bar, max_top_bar)

    rows = [top_bar_row]
    for gap in gaps:
        rows.append(rows[-1] + gap + 1)

    colors = list(BAR_COLORS)
    random.shuffle(colors)
    bars: list[dict[str, int]] = []
    sprites: list[dict[str, object]] = []
    for row, color, sprite_height in zip(rows, colors, sprite_heights):
        left = random.randint(0, 2)
        right = random.randint(max(left + width - 5, width - 3), width - 1)
        grid[row, left : right + 1] = color
        bars.append({"color": color, "row": row, "c0": left, "c1": right})

        mask = _random_sprite_mask(sprite_height)
        sprites.append({"color": color, "mask": mask, "height": sprite_height})

    top_limit = top_bar_row - 1
    placements = _place_sprites(top_limit, bars, sprites)
    if placements is None:
        return None

    for sprite, (r0, c0) in zip(sprites, placements):
        mask = sprite["mask"]
        color = int(sprite["color"])
        sh, sw = mask.shape
        patch = np.where(mask, color, BLACK)
        region = grid[r0 : r0 + sh, c0 : c0 + sw]
        region[patch != BLACK] = patch[patch != BLACK]

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue

        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, grid):
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue

        components = _find_monochrome_components(grid, BLACK)
        if len(components) != 8:
            continue

        output_colors = {int(v) for v in np.unique(out) if int(v) != BLACK}
        if output_colors != set(RANK_COLORS):
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
