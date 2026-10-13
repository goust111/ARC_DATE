from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SPRITE_COLORS = (RED, GREEN, YELLOW)
CONTROL_COLORS = (1, 6, 7, 8, 9)
PATTERN_TO_SPRITE = {"A": RED, "B": GREEN, "C": YELLOW}

GLYPH_PATTERNS = {
    "A": np.array(
        [
            [0, 0, 5, 5],
            [0, 0, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
        ],
        dtype=int,
    ),
    "B": np.array(
        [
            [0, 5, 5, 5],
            [5, 0, 5, 5],
            [5, 5, 0, 5],
            [5, 5, 5, 0],
        ],
        dtype=int,
    ),
    "C": np.array(
        [
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [0, 0, 0, 5],
        ],
        dtype=int,
    ),
}

SPRITE_LIBRARY = [
    np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=int),
]

GEN_ATTEMPTS = 300


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _match_glyph(patch: np.ndarray) -> str:
    for name, pattern in GLYPH_PATTERNS.items():
        if np.array_equal(patch, pattern):
            return name
    raise ValueError("unknown glyph pattern")


def _find_glyph_rows(grid: np.ndarray) -> list[int]:
    glyphs = []
    for coords in _components(grid == GREY, connectivity=8):
        box = _bbox(coords)
        patch = _crop(grid, box)
        if patch.shape == (4, 4):
            glyphs.append((box[0], _match_glyph(patch)))
    if len(glyphs) != 3:
        raise ValueError("expected exactly three 4x4 glyphs")
    glyphs.sort(key=lambda item: item[0])
    return [PATTERN_TO_SPRITE[name] for _, name in glyphs]


def _find_control_grid(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    hits = []
    for r in range(h - 2):
        for c in range(w - 2):
            patch = grid[r : r + 3, c : c + 3]
            if np.any(patch == BLACK):
                continue
            if np.unique(patch).size != 3:
                continue
            hits.append((r, c))
    if len(hits) != 1:
        raise ValueError("expected exactly one 3x3 control grid")
    r, c = hits[0]
    return grid[r : r + 3, c : c + 3].copy()


def _least_freq_positions(control: np.ndarray) -> np.ndarray:
    values, counts = np.unique(control, return_counts=True)
    idx = int(np.argmin(counts))
    if int((counts == counts[idx]).sum()) != 1:
        raise ValueError("least frequent control color must be unique")
    return np.argwhere(control == int(values[idx]))


def _extract_sprite(grid: np.ndarray, color: int) -> np.ndarray:
    comps = _components(grid == color, connectivity=4)
    if len(comps) != 1:
        raise ValueError(f"expected one sprite for color {color}")
    box = _bbox(comps[0])
    patch = np.zeros((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=int)
    for r, c in comps[0]:
        patch[r - box[0], c - box[1]] = color
    if patch.shape[0] > 3 or patch.shape[1] > 3:
        raise ValueError("sprite bbox must fit inside a 3x3 output block")
    return patch


def _center_in_block(sprite: np.ndarray, size: int = 3) -> np.ndarray:
    h, w = sprite.shape
    top = (size - h) // 2
    left = (size - w) // 2
    out = np.zeros((size, size), dtype=int)
    mask = sprite != BLACK
    out[top : top + h, left : left + w][mask] = sprite[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the least-frequent color in a control grid to choose output blocks and fill them with sprite content decoded from stacked glyphs.

    Input structure:
    - A black background contains three separate regions: three 4x4 grey/black glyphs stacked vertically on the left, a 3x3 colored control grid near the upper right, and three compact monochrome sprites colored red, green, and yellow near the lower right.
    - Each glyph is one of three fixed black-mask patterns, and each sprite is a 4-connected shape whose tight bounding box fits within 3x3.

    Rule:
    - Read the left glyphs from top to bottom; pattern A selects the red sprite, pattern B selects the green sprite, and pattern C selects the yellow sprite.
    - In the 3x3 control grid, find the unique least-frequent color and record every cell using that color.
    - Create a 9x9 black canvas split into 3x3 output blocks. For each marked control position (r, c), copy the sprite chosen by glyph row r into output block (r, c), centering it if its tight bounding box is smaller than 3x3.

    Key insight:
    - The control grid decides where to stamp, while the vertically ordered glyphs independently decide which color-specific sprite belongs to each output row.
    """
    g = np.asarray(grid, dtype=int)
    row_sprites = _find_glyph_rows(g)
    control = _find_control_grid(g)
    sprites = {color: _center_in_block(_extract_sprite(g, color)) for color in SPRITE_COLORS}

    out = np.zeros((9, 9), dtype=int)
    for r, c in _least_freq_positions(control):
        tile = sprites[row_sprites[int(r)]]
        out[int(r) * 3 : int(r + 1) * 3, int(c) * 3 : int(c + 1) * 3] = tile
    return out


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for _ in range(random.randrange(4)):
        out = np.rot90(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return _trim_mask(out.astype(bool)).astype(int)


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=int)
    out[mask != 0] = color
    return out


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BLACK
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _choose_low_positions() -> list[tuple[int, int]]:
    mode = random.choice(("same_row", "same_col", "scattered"))
    if mode == "same_row":
        r = random.randrange(3)
        cols = random.sample(range(3), 2)
        return [(r, cols[0]), (r, cols[1])]
    if mode == "same_col":
        c = random.randrange(3)
        rows = random.sample(range(3), 2)
        return [(rows[0], c), (rows[1], c)]
    while True:
        p1 = (random.randrange(3), random.randrange(3))
        p2 = (random.randrange(3), random.randrange(3))
        if p1 != p2 and p1[0] != p2[0] and p1[1] != p2[1]:
            return [p1, p2]


def _build_control() -> np.ndarray:
    colors = random.sample(CONTROL_COLORS, 3)
    low_color, mid_color, high_color = colors
    low_positions = _choose_low_positions()
    remaining = [(r, c) for r in range(3) for c in range(3) if (r, c) not in low_positions]
    random.shuffle(remaining)

    control = np.empty((3, 3), dtype=int)
    for r, c in low_positions:
        control[r, c] = low_color
    for r, c in remaining[:3]:
        control[r, c] = mid_color
    for r, c in remaining[3:]:
        control[r, c] = high_color
    return control


def _sample_sprites() -> dict[int, np.ndarray]:
    for _ in range(100):
        chosen = random.sample(SPRITE_LIBRARY, 3)
        sprites = {}
        sigs = set()
        ok = True
        for color, base in zip(SPRITE_COLORS, chosen):
            mask = _transform_mask(base)
            sig = tuple(map(tuple, mask.tolist()))
            if sig in sigs:
                ok = False
                break
            sigs.add(sig)
            sprites[color] = _colorize(mask, color)
        if ok:
            return sprites
    raise RuntimeError("failed to sample distinct sprites")


def _place_sprites(grid: np.ndarray, sprites: dict[int, np.ndarray], zone_top: int, zone_left: int) -> None:
    slots = [(0, 0), (0, 5), (5, 2)]
    assignments = list(zip(random.sample(SPRITE_COLORS, 3), random.sample(slots, 3)))
    for color, (sr, sc) in assignments:
        patch = sprites[color]
        h, w = patch.shape
        top = zone_top + sr + random.randint(0, 3 - h)
        left = zone_left + sc + random.randint(0, 3 - w)
        _place_patch(grid, patch, top, left)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 20)
    w = random.randint(15, 20)
    grid = np.zeros((h, w), dtype=int)

    glyph_col = random.randint(0, 2)
    gap1 = random.randint(1, 2)
    gap2 = random.randint(1, 2)
    stack_h = 12 + gap1 + gap2
    glyph_top = random.randint(0, h - stack_h)
    glyph_rows = [glyph_top, glyph_top + 4 + gap1, glyph_top + 8 + gap1 + gap2]
    glyph_order = random.sample(list(GLYPH_PATTERNS.keys()), 3)
    for top, name in zip(glyph_rows, glyph_order):
        _place_patch(grid, GLYPH_PATTERNS[name], top, glyph_col)

    control = _build_control()
    control_top = random.randint(0, 2)
    control_left = random.randint(w - 4, w - 3)
    _place_patch(grid, control, control_top, control_left)

    sprites = _sample_sprites()
    _place_sprites(grid, sprites, h - 8, w - 8)

    # Oracle check and anti-degeneracy checks.
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != (9, 9):
        return None
    if Counter(map(int, out.flatten()))[BLACK] == out.size:
        return None
    if len(_least_freq_positions(control)) != 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
