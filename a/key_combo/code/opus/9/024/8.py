from __future__ import annotations

import random

import numpy as np


SCALE = 5
INNER_SIZE = 3
GRID_SIZE_CHOICES = (3, 4, 5, 6)
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _neighbor_palette(grid: np.ndarray, r: int, c: int) -> tuple[list[int], list[int]]:
    """Return raw and de-duplicated neighbor colors in up-right-down-left order."""
    h, w = grid.shape
    raw: list[int] = []
    for rr, cc in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
        if 0 <= rr < h and 0 <= cc < w:
            raw.append(int(grid[rr, cc]))

    palette: list[int] = []
    seen: set[int] = set()
    for color in raw:
        if color not in seen:
            seen.add(color)
            palette.append(color)
    return raw, palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each input pixel to a 5x5 block whose interior encodes that pixel's neighbor colors as concentric bands.

    Input structure:
    - The input is a small full-color grid with no black background; every cell is a colored pixel.
    - The grid is at least 3x3 and at most 6x6 so the fixed 5x enlargement still fits inside ARC's 30x30 limit.
    - Each pixel is interpreted inside the original grid's row-column coordinate system, where valid neighbors are checked in up-right-down-left order.

    Rule:
    - Replace every input pixel by a solid 5x5 block of its own color.
    - Keep the outer one-cell border of each block unchanged.
    - For the inner 3x3 area, collect the original pixel's existing neighbors in up-right-down-left order, skip missing directions, then remove repeated colors while preserving first appearance.
    - Fill the inner 3x3 by distance from its boundary: the inner ring uses palette[0], and the center uses palette[1 mod len(palette)].

    Key insight:
    - A block's interior is not determined by the block's own color; it is determined by the surrounding topology of the corresponding pixel in the original grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert 3 <= h <= 6 and 3 <= w <= 6, "input must stay within the puzzle's 3x3..6x6 range"
    assert np.all(g != 0), "input may not use black"

    out = np.kron(g, np.ones((SCALE, SCALE), dtype=int))

    for r in range(h):
        for c in range(w):
            _raw, palette = _neighbor_palette(g, r, c)
            assert palette, "every cell must have at least one valid neighbor"

            br = r * SCALE
            bc = c * SCALE
            out[br + 1 : br + 4, bc + 1 : bc + 4] = palette[0]
            out[br + 2, bc + 2] = palette[1 % len(palette)]

    return out


def _all_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _interior_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]


def _edge_noncorner_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        coords.append((0, c))
        coords.append((h - 1, c))
    for r in range(1, h - 1):
        coords.append((r, 0))
        coords.append((r, w - 1))
    return coords


def _ensure_every_color_used_twice(grid: np.ndarray, colors: list[int]) -> None:
    """Improve visual clarity by avoiding near-singleton colors after motif injection."""
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    needy = [color for color, count in counts.items() if count < 2]
    spare_cells = [(r, c) for r, c in _all_coords(*grid.shape) if counts[int(grid[r, c])] > 2]
    random.shuffle(spare_cells)

    for color in needy:
        if not spare_cells:
            break
        r, c = spare_cells.pop()
        counts[int(grid[r, c])] -= 1
        grid[r, c] = color
        counts[color] += 1


def _inject_len4_palette(grid: np.ndarray, colors: list[int]) -> bool:
    if len(colors) < 4:
        return False
    candidates = _interior_coords(*grid.shape)
    if not candidates:
        return False

    r, c = random.choice(candidates)
    up, right, down, left = random.sample(colors, 4)
    grid[r - 1, c] = up
    grid[r, c + 1] = right
    grid[r + 1, c] = down
    grid[r, c - 1] = left
    grid[r, c] = random.choice(colors)
    return True


def _inject_len3_palette(grid: np.ndarray, colors: list[int]) -> bool:
    candidates = _interior_coords(*grid.shape)
    if not candidates:
        return False
    if len(colors) >= 3:
        a, b, ccol = random.sample(colors, 3)
    else:
        return False

    r, c = random.choice(candidates)
    grid[r - 1, c] = a
    grid[r, c + 1] = b
    grid[r + 1, c] = ccol
    grid[r, c - 1] = a
    return True


def _inject_edge_dedup(grid: np.ndarray, colors: list[int]) -> bool:
    candidates = _edge_noncorner_coords(*grid.shape)
    if not candidates or len(colors) < 2:
        return False

    r, c = random.choice(candidates)
    a, b = random.sample(colors, 2)
    h, w = grid.shape

    if r == 0:
        grid[r, c + 1] = a
        grid[r + 1, c] = b
        grid[r, c - 1] = a
    elif r == h - 1:
        grid[r - 1, c] = a
        grid[r, c + 1] = b
        grid[r, c - 1] = a
    elif c == 0:
        grid[r - 1, c] = a
        grid[r, c + 1] = b
        grid[r + 1, c] = a
    else:
        grid[r - 1, c] = a
        grid[r + 1, c] = b
        grid[r, c - 1] = a
    return True


def _inject_len1_palette(grid: np.ndarray, colors: list[int]) -> bool:
    corners = [(0, 0), (0, grid.shape[1] - 1), (grid.shape[0] - 1, 0), (grid.shape[0] - 1, grid.shape[1] - 1)]
    r, c = random.choice(corners)
    a = random.choice(colors)
    if r == 0 and c == 0:
        grid[0, 1] = a
        grid[1, 0] = a
    elif r == 0:
        grid[0, c - 1] = a
        grid[1, c] = a
    elif c == 0:
        grid[r - 1, 0] = a
        grid[r, 1] = a
    else:
        grid[r - 1, c] = a
        grid[r, c - 1] = a
    return True


def _make_base_grid(h: int, w: int, colors: list[int]) -> np.ndarray:
    flat = colors[:]
    while len(flat) < h * w:
        flat.append(random.choice(colors))
    random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(h, w)


def _semantic_features(grid: np.ndarray, out: np.ndarray) -> dict[str, object]:
    h, w = grid.shape
    if out.shape != (h * SCALE, w * SCALE):
        return {"valid": False}

    palette_lengths: list[int] = []
    raw_lengths: list[int] = []
    has_ring_change = False
    has_center_change = False
    has_len1 = False
    has_len4 = False
    has_missing_neighbor = False

    for r in range(h):
        for c in range(w):
            raw, palette = _neighbor_palette(grid, r, c)
            raw_lengths.append(len(raw))
            palette_lengths.append(len(palette))
            has_len1 |= len(palette) == 1
            has_len4 |= len(palette) == 4
            has_missing_neighbor |= len(raw) < 4

            br = r * SCALE
            bc = c * SCALE
            block = out[br : br + SCALE, bc : bc + SCALE]
            border = np.zeros((SCALE, SCALE), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(block[border] == int(grid[r, c])):
                return {"valid": False}

            expected = np.full((INNER_SIZE, INNER_SIZE), palette[0], dtype=int)
            expected[1, 1] = palette[1 % len(palette)]
            actual = block[1:4, 1:4]
            if not np.array_equal(actual, expected):
                return {"valid": False}

            has_ring_change |= palette[0] != int(grid[r, c])
            has_center_change |= palette[1 % len(palette)] != palette[0]

    return {
        "valid": True,
        "palette_lengths": palette_lengths,
        "raw_lengths": raw_lengths,
        "has_ring_change": has_ring_change,
        "has_center_change": has_center_change,
        "has_len1": has_len1,
        "has_len4": has_len4,
        "has_missing_neighbor": has_missing_neighbor,
        "has_dedup": any(raw_len > palette_len for raw_len, palette_len in zip(raw_lengths, palette_lengths)),
        "max_palette_len": max(palette_lengths),
        "n_colors": len(np.unique(grid)),
    }


def _make_candidate() -> tuple[np.ndarray, str]:
    mode = random.choices(
        ["len4_focus", "len1_focus", "mixed"],
        weights=[4, 3, 4],
        k=1,
    )[0]

    if mode == "len4_focus":
        h = random.choice((4, 5, 6))
        w = random.choice((4, 5, 6))
        n_colors = 4
    else:
        h = random.choice(GRID_SIZE_CHOICES)
        w = random.choice(GRID_SIZE_CHOICES)
        n_colors = random.choices([3, 4], weights=[3, 2], k=1)[0]

    colors = random.sample(COLOR_POOL, n_colors)
    grid = _make_base_grid(h, w, colors)

    if mode == "len4_focus":
        _inject_len4_palette(grid, colors)
        _inject_edge_dedup(grid, colors)
        if random.random() < 0.55:
            _inject_len1_palette(grid, colors)
    elif mode == "len1_focus":
        _inject_len1_palette(grid, colors)
        _inject_edge_dedup(grid, colors)
        _inject_len3_palette(grid, colors)
    else:
        _inject_edge_dedup(grid, colors)
        if n_colors == 4 and random.random() < 0.65:
            _inject_len4_palette(grid, colors)
        else:
            _inject_len3_palette(grid, colors)
        if random.random() < 0.4:
            _inject_len1_palette(grid, colors)

    grid[random.randrange(h), random.randrange(w)] = random.choice(colors)
    _ensure_every_color_used_twice(grid, colors)
    return grid, mode


def _is_good_input(grid: np.ndarray, mode: str) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if not (3 <= h <= 6 and 3 <= w <= 6):
        return False
    if max(h * SCALE, w * SCALE) > 30:
        return False
    if np.any(grid == 0):
        return False

    colors = np.unique(grid)
    if not (3 <= len(colors) <= 4):
        return False
    counts = np.bincount(grid.ravel(), minlength=10)
    present_counts = [int(counts[color]) for color in colors]
    if min(present_counts) < 2:
        return False

    out = main(grid)
    feats = _semantic_features(grid, out)
    if not feats["valid"]:
        return False
    if not feats["has_missing_neighbor"]:
        return False
    if not feats["has_dedup"]:
        return False
    if not feats["has_ring_change"]:
        return False
    if not feats["has_center_change"]:
        return False
    if int(feats["max_palette_len"]) < min(3, len(colors)):
        return False
    if mode == "len4_focus" and not feats["has_len4"]:
        return False
    if mode == "len1_focus" and not feats["has_len1"]:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, mode = _make_candidate()
        if _is_good_input(grid, mode):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
