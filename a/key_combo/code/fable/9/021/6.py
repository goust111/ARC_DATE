import random
from collections import Counter

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))

SPRITE_SIZE_RANGE = (3, 5)
HOLE_MODE_CHOICES = (0, 0, 1, 1, 2)
GEN_RETRY_LIMIT = 500


def _border_background(grid: np.ndarray) -> int:
    top = grid[0, :]
    bottom = grid[-1, :]
    left = grid[:, 0]
    right = grid[:, -1]
    border = np.concatenate((top, bottom, left, right))
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected at least one foreground cell")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _asymmetry_ok(sprite: np.ndarray) -> bool:
    transforms = [
        np.rot90(sprite, 1),
        np.rot90(sprite, 2),
        np.rot90(sprite, 3),
        np.fliplr(sprite),
        np.flipud(sprite),
        sprite.T,
        np.fliplr(np.rot90(sprite, 1)),
    ]
    return all(not np.array_equal(sprite, other) for other in transforms)


def _color_spread_ok(sprite: np.ndarray, color: int) -> bool:
    coords = np.argwhere(sprite == int(color))
    if coords.shape[0] < 2:
        return False
    rows = {int(r) for r, _ in coords}
    cols = {int(c) for _, c in coords}
    return len(rows) >= 2 and len(cols) >= 2


def _has_cross_color_edge(sprite: np.ndarray, a: int, b: int) -> bool:
    h, w = sprite.shape
    for r in range(h):
        for c in range(w):
            if int(sprite[r, c]) not in (int(a), int(b)):
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    pair = {int(sprite[r, c]), int(sprite[nr, nc])}
                    if pair == {int(a), int(b)}:
                        return True
    return False


def _instance_parts(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _border_background(g)
    h, w = g.shape

    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    control_candidates = [(r, c) for r, c in corners if int(g[r, c]) != int(bg)]
    assert len(control_candidates) == 1, "expected exactly one non-background corner control pixel"
    control_r, control_c = control_candidates[0]
    recursive_color = int(g[control_r, control_c])

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = control_r + dr, control_c + dc
            if 0 <= nr < h and 0 <= nc < w:
                assert int(g[nr, nc]) == int(bg), "control pixel must be isolated"

    fg = g != int(bg)
    fg[control_r, control_c] = False
    r0, c0, r1, c1 = _bbox(fg)
    assert r1 - r0 == c1 - c0, "sprite bbox must be square"

    outside = fg.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    assert not outside.any(), "all non-control foreground pixels must lie inside the sprite square"

    sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = sorted(set(map(int, np.unique(sprite))) - {int(bg)})
    assert len(colors) == 2, "sprite must use exactly two non-background colors"
    assert recursive_color in colors, "control color must be one of the sprite colors"
    return int(bg), recursive_color, sprite


def _render_output(sprite: np.ndarray, recursive_color: int, bg: int) -> np.ndarray:
    n = int(sprite.shape[0])
    out = np.full((n * n, n * n), int(bg), dtype=int)
    for r in range(n):
        for c in range(n):
            value = int(sprite[r, c])
            block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
            if value == int(bg):
                continue
            if value == int(recursive_color):
                block[:] = sprite
            else:
                block[:] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge a square sprite into an N-by-N block grid, using one sprite color as a self-stamping opcode.

    Input structure:
    - The background fills most of the canvas.
    - One square N×N sprite sits away from the border and uses exactly two non-background colors, with optional background holes inside the square.
    - Exactly one isolated corner pixel has one of the sprite's two colors and acts as the control pixel.

    Rule:
    - Read the corner control pixel to decide which sprite color is the recursive color.
    - Produce an output of size N²×N² by replacing every sprite cell with an N×N block.
    - A background sprite cell becomes an all-background block, a non-recursive colored cell becomes a solid block of its own color, and a recursive-color cell becomes a full copy of the original N×N sprite.

    Key insight:
    - The two visible sprite colors are not just paint: one means "fill this block solid", while the control pixel selects the other to mean "stamp the whole sprite here again".
    """
    bg, recursive_color, sprite = _instance_parts(np.asarray(grid, dtype=int))
    return _render_output(sprite, recursive_color, bg)


def _build_sprite(n: int, colors: tuple[int, int]) -> np.ndarray | None:
    a, b = map(int, colors)
    total = n * n
    max_holes = min(2, total - 4)
    hole_count = min(random.choice(HOLE_MODE_CHOICES), max_holes)

    positions = list(range(total))
    hole_positions = set(random.sample(positions, hole_count))
    filled_positions = [idx for idx in positions if idx not in hole_positions]
    filled_count = len(filled_positions)
    if filled_count < 4:
        return None

    low = max(2, int(np.ceil(filled_count * 0.30)))
    high = min(filled_count - 2, int(np.floor(filled_count * 0.70)))
    if low > high:
        return None

    a_count = random.randint(low, high)
    a_positions = set(random.sample(filled_positions, a_count))

    sprite = np.full((n, n), BACKGROUND, dtype=int)
    for idx in filled_positions:
        r, c = divmod(idx, n)
        sprite[r, c] = a if idx in a_positions else b

    non_bg = sprite != BACKGROUND
    if _component_count(non_bg) != 1:
        return None
    if not _color_spread_ok(sprite, a) or not _color_spread_ok(sprite, b):
        return None
    if not _has_cross_color_edge(sprite, a, b):
        return None
    if not _asymmetry_ok(sprite):
        return None
    return sprite


def _layout_signature(sprite: np.ndarray, recursive_color: int) -> int:
    weights = np.arange(1, sprite.size + 1, dtype=int).reshape(sprite.shape)
    return int(np.sum(sprite * weights) + recursive_color * 97 + sprite.shape[0] * 131)


def _place_input(sprite: np.ndarray, recursive_color: int) -> np.ndarray:
    n = int(sprite.shape[0])
    sig = _layout_signature(sprite, recursive_color)
    corner = sig % 4

    near_row = 2 + (sig // 4) % 3
    near_col = 2 + (sig // 13) % 3
    far_row = 1 + (sig // 39) % 4
    far_col = 1 + (sig // 97) % 4

    if corner == 0:  # top-left control
        top, left, bottom, right = near_row, near_col, far_row, far_col
        control = (0, 0)
        anchor = (top, left)
    elif corner == 1:  # top-right control
        top, right, bottom, left = near_row, near_col, far_row, far_col
        h = top + n + bottom
        w = left + n + right
        control = (0, w - 1)
        anchor = (top, left)
        grid = np.full((h, w), BACKGROUND, dtype=int)
        r0, c0 = anchor
        grid[r0 : r0 + n, c0 : c0 + n] = sprite
        grid[control] = recursive_color
        return grid
    elif corner == 2:  # bottom-left control
        bottom, left, top, right = near_row, near_col, far_row, far_col
        h = top + n + bottom
        w = left + n + right
        control = (h - 1, 0)
        anchor = (top, left)
        grid = np.full((h, w), BACKGROUND, dtype=int)
        r0, c0 = anchor
        grid[r0 : r0 + n, c0 : c0 + n] = sprite
        grid[control] = recursive_color
        return grid
    else:  # bottom-right control
        bottom, right, top, left = near_row, near_col, far_row, far_col
        h = top + n + bottom
        w = left + n + right
        control = (h - 1, w - 1)
        anchor = (top, left)
        grid = np.full((h, w), BACKGROUND, dtype=int)
        r0, c0 = anchor
        grid[r0 : r0 + n, c0 : c0 + n] = sprite
        grid[control] = recursive_color
        return grid

    h = top + n + bottom
    w = left + n + right
    grid = np.full((h, w), BACKGROUND, dtype=int)
    r0, c0 = anchor
    grid[r0 : r0 + n, c0 : c0 + n] = sprite
    grid[control] = recursive_color
    return grid


def _semantic_checks(sprite: np.ndarray, recursive_color: int) -> bool:
    colors = sorted(set(map(int, np.unique(sprite))) - {BACKGROUND})
    if len(colors) != 2:
        return False
    other_color = colors[0] if colors[1] == recursive_color else colors[1]

    out = _render_output(sprite, recursive_color, BACKGROUND)
    alt = _render_output(sprite, other_color, BACKGROUND)
    if np.array_equal(out, alt):
        return False

    recursive_pixels = int(np.count_nonzero(sprite == recursive_color))
    solid_pixels = int(np.count_nonzero(sprite == other_color))
    if recursive_pixels < 2 or solid_pixels < 2:
        return False

    if np.any(sprite == BACKGROUND):
        bg_blocks = 0
        n = int(sprite.shape[0])
        for r in range(n):
            for c in range(n):
                block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
                if int(sprite[r, c]) == BACKGROUND and np.any(block != BACKGROUND):
                    return False
                if int(sprite[r, c]) == BACKGROUND:
                    bg_blocks += 1
        if bg_blocks == 0:
            return False

    return True


def _attempt() -> np.ndarray | None:
    n = random.randint(*SPRITE_SIZE_RANGE)
    colors = tuple(random.sample(COLOR_POOL, 2))
    sprite = _build_sprite(n, colors)
    if sprite is None:
        return None

    recursive_color = int(random.choice(colors))
    if not _semantic_checks(sprite, recursive_color):
        return None

    grid = _place_input(sprite, recursive_color)
    if max(grid.shape) > 30:
        return None

    try:
        bg, found_recursive, found_sprite = _instance_parts(grid)
        expected = _render_output(sprite, recursive_color, BACKGROUND)
        actual = main(grid)
    except Exception:
        return None

    if bg != BACKGROUND:
        return None
    if found_recursive != recursive_color or not np.array_equal(found_sprite, sprite):
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
