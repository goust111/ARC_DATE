from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

_HOLE_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _uniform_separator_row(g: np.ndarray, bg: int) -> tuple[int, int]:
    rows = []
    for r in range(g.shape[0]):
        color = int(g[r, 0])
        if color != bg and bool(np.all(g[r, :] == color)):
            rows.append((r, color))
    assert len(rows) == 1, "expected exactly one solid non-background separator row"
    return rows[0]


def _connected_components(panel: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(panel[r, c]) == bg:
                continue

            color = int(panel[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                assert int(panel[rr, cc]) == color
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(panel[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                }
            )

    return components


def _count_internal_holes(patch: np.ndarray, bg: int) -> int:
    h, w = patch.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(patch[r, c]) != bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)

            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(patch[nr, nc]) != bg:
                        continue
                    seen[nr, nc] = True
                    touches_border |= nr in (0, h - 1) or nc in (0, w - 1)
                    stack.append((nr, nc))

            if not touches_border:
                holes += 1

    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the upper-panel sprites by internal hole count and redraw them in the lower panel.

    Input structure:
    - The background is black, and a single full-width grey row splits the grid into upper and lower panels.
    - The upper panel contains 3-5 separate monochrome rectangular sprites arranged left-to-right with one background column between consecutive sprites.
    - All sprites share the same bounding-box size, each sprite may contain 0-3 enclosed black holes, and the lower panel is otherwise blank.

    Rule:
    - For each upper-panel sprite, crop its bounding box and count the number of 4-connected black background regions that stay strictly inside that box.
    - Stable-sort the sprites from fewer holes to more holes.
    - Clear the upper panel to black, keep the grey separator row unchanged, and redraw the sorted sprites top-aligned in the lower panel.
    - The lower panel reuses the same leftmost slot footprint as the source row: same sprite width and one-column gaps.

    Key insight:
    - The sorting key is the number of enclosed holes inside each sprite, not the sprite color or its original position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG
    sep_row, sep_color = _uniform_separator_row(g, bg)
    assert sep_row > 0 and sep_row < g.shape[0] - 1, "separator must leave non-empty panels"

    top = g[:sep_row, :]
    sprites = _connected_components(top, bg)
    assert 3 <= len(sprites) <= 5, "expected 3-5 upper-panel sprites"
    sprites.sort(key=lambda sprite: (sprite["bbox"][1], sprite["bbox"][0]))

    first_r0, first_c0, first_r1, first_c1 = sprites[0]["bbox"]
    sprite_h = first_r1 - first_r0 + 1
    sprite_w = first_c1 - first_c0 + 1
    assert g.shape[0] - sep_row - 1 >= sprite_h, "lower panel must fit the sorted row"

    start_col = first_c0
    for i, sprite in enumerate(sprites):
        r0, c0, r1, c1 = sprite["bbox"]
        assert r1 - r0 + 1 == sprite_h and c1 - c0 + 1 == sprite_w, "all sprites must share one bbox size"
        assert c0 == start_col + i * (sprite_w + 1), "sprites must be laid out with one-column gaps"
        sprite["holes"] = _count_internal_holes(sprite["patch"], bg)

    ordered = sorted(sprites, key=lambda sprite: int(sprite["holes"]))
    out = np.full_like(g, bg)
    out[sep_row, :] = sep_color

    base_row = sep_row + 1
    for i, sprite in enumerate(ordered):
        c0 = start_col + i * (sprite_w + 1)
        out[base_row : base_row + sprite_h, c0 : c0 + sprite_w] = sprite["patch"]

    return out


def _hole_cells_ok(existing: set[tuple[int, int]], new_cells: set[tuple[int, int]]) -> bool:
    if existing & new_cells:
        return False
    for r, c in new_cells:
        for rr, cc in existing:
            if abs(r - rr) + abs(c - cc) <= 1:
                return False
    return True


def _hole_shape_pool(hole_count: int) -> tuple[np.ndarray, ...]:
    if hole_count >= 3:
        return _HOLE_SHAPES[:5]
    return _HOLE_SHAPES


def _build_sprite(height: int, width: int, color: int, hole_count: int) -> np.ndarray:
    if hole_count == 0:
        return np.full((height, width), color, dtype=int)

    for _ in range(250):
        holes: set[tuple[int, int]] = set()
        success = True
        for _hole_idx in range(hole_count):
            placements: list[set[tuple[int, int]]] = []
            for shape in _hole_shape_pool(hole_count):
                sh, sw = shape.shape
                if height - sh < 2 or width - sw < 2:
                    continue
                for r0 in range(1, height - sh):
                    for c0 in range(1, width - sw):
                        cells = {
                            (r0 + dr, c0 + dc)
                            for dr in range(sh)
                            for dc in range(sw)
                            if bool(shape[dr, dc])
                        }
                        if _hole_cells_ok(holes, cells):
                            placements.append(cells)
            if not placements:
                success = False
                break
            holes |= random.choice(placements)

        if not success:
            continue

        sprite = np.full((height, width), color, dtype=int)
        for r, c in holes:
            sprite[r, c] = BG
        if _count_internal_holes(sprite, BG) == hole_count:
            return sprite

    raise RuntimeError("failed to sample a sprite with the requested hole count")


def _sample_hole_counts(count: int) -> list[int]:
    for _ in range(200):
        values = [random.randint(0, 3) for _ in range(count)]
        if len(set(values)) < 2:
            continue
        if values == sorted(values):
            continue
        return values
    raise RuntimeError("failed to sample a non-degenerate hole-count sequence")


def _render_expected(
    height: int,
    width: int,
    sep_row: int,
    start_col: int,
    sprites: list[np.ndarray],
) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    out[sep_row, :] = SEP
    sprite_h, sprite_w = sprites[0].shape
    ordered = sorted(sprites, key=lambda patch: _count_internal_holes(patch, BG))
    for i, patch in enumerate(ordered):
        c0 = start_col + i * (sprite_w + 1)
        out[sep_row + 1 : sep_row + 1 + sprite_h, c0 : c0 + sprite_w] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.randint(3, 5)
    max_width = min(7, (30 - (sprite_count - 1)) // sprite_count)
    if max_width < 5:
        return None

    sprite_w = random.randint(5, max_width)
    sprite_h = random.randint(7, 10)
    row_span = sprite_count * sprite_w + (sprite_count - 1)
    extra_width = random.randint(0, min(6, 30 - row_span))
    width = row_span + extra_width
    start_col = random.randint(0, width - row_span)

    top_offset = random.randint(0, 3)
    top_tail = random.randint(0, 3)
    bottom_tail = random.randint(0, 3)
    sep_row = top_offset + sprite_h + top_tail
    height = sep_row + 1 + sprite_h + bottom_tail
    if height > 30:
        return None

    hole_counts = _sample_hole_counts(sprite_count)
    colors = random.sample(SPRITE_COLORS, sprite_count)
    sprites = []
    for color, hole_count in zip(colors, hole_counts):
        sprites.append(_build_sprite(sprite_h, sprite_w, color, hole_count))

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_row, :] = SEP
    for i, patch in enumerate(sprites):
        c0 = start_col + i * (sprite_w + 1)
        grid[top_offset : top_offset + sprite_h, c0 : c0 + sprite_w] = patch

    expected = _render_expected(height, width, sep_row, start_col, sprites)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
