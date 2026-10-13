import random
from collections import deque

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PATTERN_TYPES = ("checker", "vertical", "horizontal")
GEN_RETRIES = 400


def _bg_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _make_pattern(size, pattern_type, color_a, color_b):
    pattern = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if pattern_type == "checker":
                use_a = (r + c) % 2 == 0
            elif pattern_type == "vertical":
                use_a = c % 2 == 0
            else:
                use_a = r % 2 == 0
            pattern[r, c] = color_a if use_a else color_b
    return pattern


def _is_valid_pattern(block):
    colors = list(dict.fromkeys(int(v) for v in block.flatten()))
    if len(colors) != 2:
        return False
    color_a = int(block[0, 0])
    color_b = next(color for color in colors if color != color_a)
    for pattern_type in PATTERN_TYPES:
        if np.array_equal(block, _make_pattern(block.shape[0], pattern_type, color_a, color_b)):
            return True
    return False


def _border_positions(grid, bg):
    return {
        "top": [int(c) for c in np.flatnonzero(grid[0, :] != bg)],
        "bottom": [int(c) for c in np.flatnonzero(grid[-1, :] != bg)],
        "left": [int(r) for r in np.flatnonzero(grid[:, 0] != bg)],
        "right": [int(r) for r in np.flatnonzero(grid[:, -1] != bg)],
    }


def _palette_piece(grid, bg, side, pos, size):
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        if pos + size > w or size + 1 > h:
            return None
        if np.any(grid[0, pos + 1 : pos + size] != bg):
            return None
        marker = int(grid[0, pos])
        block = grid[1 : size + 1, pos : pos + size]
        mask[0, pos] = True
        mask[1 : size + 1, pos : pos + size] = True
    elif side == "bottom":
        if pos + size > w or size + 1 > h:
            return None
        if np.any(grid[h - 1, pos + 1 : pos + size] != bg):
            return None
        marker = int(grid[h - 1, pos])
        block = grid[h - size - 1 : h - 1, pos : pos + size]
        mask[h - 1, pos] = True
        mask[h - size - 1 : h - 1, pos : pos + size] = True
    elif side == "left":
        if pos + size > h or size + 1 > w:
            return None
        if np.any(grid[pos + 1 : pos + size, 0] != bg):
            return None
        marker = int(grid[pos, 0])
        block = grid[pos : pos + size, 1 : size + 1]
        mask[pos, 0] = True
        mask[pos : pos + size, 1 : size + 1] = True
    else:
        if pos + size > h or size + 1 > w:
            return None
        if np.any(grid[pos + 1 : pos + size, w - 1] != bg):
            return None
        marker = int(grid[pos, w - 1])
        block = grid[pos : pos + size, w - size - 1 : w - 1]
        mask[pos, w - 1] = True
        mask[pos : pos + size, w - size - 1 : w - 1] = True

    if marker == bg or block.shape != (size, size) or np.any(block == bg) or not _is_valid_pattern(block):
        return None
    return marker, block.copy(), mask


def _parse_palette(grid, bg):
    border = _border_positions(grid, bg)
    active = [side for side, positions in border.items() if positions]
    assert len(active) == 1, "expected exactly one palette-bearing edge"
    side = active[0]
    positions = border[side]
    assert len(positions) >= 2, "palette must contain at least two entries"

    candidate_sizes = []
    if len(positions) >= 2:
        diffs = [b - a for a, b in zip(positions, positions[1:])]
        if len(set(diffs)) == 1 and diffs[0] - 1 in (2, 3):
            candidate_sizes.append(diffs[0] - 1)
    for size in (2, 3):
        if size not in candidate_sizes:
            candidate_sizes.append(size)

    for size in candidate_sizes:
        if any(b - a != size + 1 for a, b in zip(positions, positions[1:])):
            continue
        mapping = {}
        palette_mask = np.zeros_like(grid, dtype=bool)
        ok = True
        for pos in positions:
            piece = _palette_piece(grid, bg, side, pos, size)
            if piece is None:
                ok = False
                break
            marker, block, piece_mask = piece
            if marker in mapping:
                ok = False
                break
            mapping[marker] = block
            palette_mask |= piece_mask
        if ok and mapping:
            return mapping, palette_mask

    raise AssertionError("unable to parse palette")


def _rectangular_fence_bbox(grid, mask, color, bg):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros(sub.shape, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(sub == color, expected):
        return None
    if np.any(sub[1:-1, 1:-1] != bg):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _fill_with_pattern(out, top, left, bottom, right, pattern):
    ph, pw = pattern.shape
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            out[r, c] = int(pattern[(r - top) % ph, (c - left) % pw])


def _solve(grid):
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mapping, palette_mask = _parse_palette(g, bg)
    out = g.copy()

    for fence_color, pattern in mapping.items():
        mask = (g == fence_color) & (~palette_mask)
        for comp in _components(mask):
            comp_mask = np.zeros_like(g, dtype=bool)
            for r, c in comp:
                comp_mask[r, c] = True
            bbox = _rectangular_fence_bbox(g, comp_mask, fence_color, bg)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            _fill_with_pattern(out, r0 + 1, c0 + 1, r1 - 1, c1 - 1, pattern)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced rectangle with the tiled texture selected by its border color from an edge palette.

    Input structure:
    - The background is the dominant single color.
    - Several 1-pixel-wide monochrome rectangular fences sit away from the grid edge, and each fence encloses only background.
    - Exactly one grid edge contains a palette region made of repeated cells; each cell has a single border-color marker on the edge and a 2x2 or 3x3 two-color texture patch inward from that marker.

    Rule:
    - Locate the unique edge palette and read each marker color as the key for the adjacent texture patch.
    - Find every closed rectangular fence in the non-palette area and determine its fence color.
    - Look up that fence color in the palette and tile the matched texture across the fence interior, anchored at the interior's top-left cell.
    - Leave the fences themselves and the palette region unchanged.

    Key insight:
    The fence color is only an index into a visible texture legend on the border; solving requires parsing the palette structure first and then repeating the retrieved 2D template inside the matching enclosure.
    """
    return _solve(grid)


def _draw_palette(grid, side, start, fence_colors, patterns, size):
    palette_mask = np.zeros_like(grid, dtype=bool)
    for idx, (fence_color, pattern) in enumerate(zip(fence_colors, patterns)):
        pos = start + idx * (size + 1)
        if side == "top":
            grid[0, pos] = fence_color
            grid[1 : size + 1, pos : pos + size] = pattern
            palette_mask[0, pos] = True
            palette_mask[1 : size + 1, pos : pos + size] = True
        elif side == "bottom":
            grid[-1, pos] = fence_color
            grid[-size - 1 : -1, pos : pos + size] = pattern
            palette_mask[-1, pos] = True
            palette_mask[-size - 1 : -1, pos : pos + size] = True
        elif side == "left":
            grid[pos, 0] = fence_color
            grid[pos : pos + size, 1 : size + 1] = pattern
            palette_mask[pos, 0] = True
            palette_mask[pos : pos + size, 1 : size + 1] = True
        else:
            grid[pos, -1] = fence_color
            grid[pos : pos + size, -size - 1 : -1] = pattern
            palette_mask[pos, -1] = True
            palette_mask[pos : pos + size, -size - 1 : -1] = True
    return palette_mask


def _choose_patterns(rng, size, count, texture_pool):
    patterns = []
    used = set()
    for _ in range(200):
        if len(patterns) == count:
            return patterns
        color_a, color_b = rng.sample(texture_pool, 2)
        pattern_type = rng.choice(PATTERN_TYPES)
        pattern = _make_pattern(size, pattern_type, color_a, color_b)
        key = tuple(int(v) for v in pattern.flatten())
        if key in used:
            continue
        used.add(key)
        patterns.append(pattern)
    return None


def _place_rectangles(rng, grid, reserved, side, size, fence_colors, patterns):
    h, w = grid.shape
    if side == "top":
        r_min, r_max = size + 2, h - 2
        c_min, c_max = 1, w - 2
    elif side == "bottom":
        r_min, r_max = 1, h - size - 3
        c_min, c_max = 1, w - 2
    elif side == "left":
        r_min, r_max = 1, h - 2
        c_min, c_max = size + 2, w - 2
    else:
        r_min, r_max = 1, h - 2
        c_min, c_max = 1, w - size - 3

    if r_min > r_max or c_min > c_max:
        return None

    occupancy = reserved.copy()
    rectangles = []
    for fence_color, pattern in zip(fence_colors, patterns):
        placed = False
        for _ in range(300):
            inner_h = rng.randint(2, 6)
            inner_w = rng.randint(2, 6)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if r_min + outer_h - 1 > r_max or c_min + outer_w - 1 > c_max:
                continue
            r0 = rng.randint(r_min, r_max - outer_h + 1)
            c0 = rng.randint(c_min, c_max - outer_w + 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h - 1, r1 + 1)
            pc1 = min(w - 1, c1 + 1)
            if np.any(occupancy[pr0 : pr1 + 1, pc0 : pc1 + 1]):
                continue

            grid[r0, c0 : c1 + 1] = fence_color
            grid[r1, c0 : c1 + 1] = fence_color
            grid[r0 : r1 + 1, c0] = fence_color
            grid[r0 : r1 + 1, c1] = fence_color
            occupancy[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            rectangles.append((r0, c0, r1, c1, pattern))
            placed = True
            break
        if not placed:
            return None
    return rectangles


def _expected_output(grid, rectangles):
    out = grid.copy()
    for r0, c0, r1, c1, pattern in rectangles:
        _fill_with_pattern(out, r0 + 1, c0 + 1, r1 - 1, c1 - 1, pattern)
    return out


def _gen_attempt(rng):
    size = rng.choice([2, 3])
    side = rng.choice(SIDES)
    count = rng.randint(2, 4)
    palette_span = count * size + (count - 1)
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    if side in ("top", "bottom") and width < palette_span + 2:
        width = palette_span + 2
    if side in ("left", "right") and height < palette_span + 2:
        height = palette_span + 2
    if height > 30 or width > 30:
        return None

    colors = list(range(10))
    bg = rng.choice(colors)
    remaining = [color for color in colors if color != bg]
    fence_colors = rng.sample(remaining, count)
    texture_pool = [color for color in remaining if color not in fence_colors]
    patterns = _choose_patterns(rng, size, count, texture_pool)
    if patterns is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    start_limit = (width if side in ("top", "bottom") else height) - palette_span - 1
    if start_limit < 1:
        return None
    start = rng.randint(1, start_limit)
    palette_mask = _draw_palette(grid, side, start, fence_colors, patterns, size)
    rectangles = _place_rectangles(rng, grid, palette_mask, side, size, fence_colors, patterns)
    if rectangles is None:
        return None

    expected = _expected_output(grid, rectangles)
    solved = _solve(grid)
    if not np.array_equal(expected, solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
