from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 5
SQUARE_COLORS = [1, 2, 3, 4, 7]
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _scan_monochrome_2x2(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Return all pure-color 2x2 windows as (top_row, left_col, color) in scan order."""
    h, w = grid.shape
    squares: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color in (BACKGROUND, ANCHOR_COLOR):
                continue
            block = grid[r : r + 2, c : c + 2]
            if np.all(block == color):
                squares.append((r, c, color))
    return squares


def _valid_squares(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Keep the first pure-color 2x2 encountered for each color."""
    kept: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for r, c, color in _scan_monochrome_2x2(grid):
        if color not in seen_colors:
            seen_colors.add(color)
            kept.append((r, c, color))
    return sorted(kept, key=lambda item: (item[0], item[1]))


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == ANCHOR_COLOR)
    assert coords.shape[0] == 1, "expected exactly one grey anchor pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _render_rings(shape: tuple[int, int], anchor: tuple[int, int], colors: list[int]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    ar, ac = anchor
    layer_count = len(colors)
    for depth, color in enumerate(colors):
        radius = layer_count - 1 - depth
        r0, r1 = ar - radius, ar + radius
        c0, c1 = ac - radius, ac + radius
        assert 0 <= r0 <= r1 < shape[0]
        assert 0 <= c0 <= c1 < shape[1]
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Generate anchor-centered concentric rings whose depth count and colors come from unique 2x2 squares.

    Input structure:
    - A black background contains several solid 2x2 monochrome squares chosen from blue, red, green, yellow, and orange.
    - Some single-pixel noise may appear, and some extra 2x2 squares may duplicate colors already used by earlier squares.
    - A single grey anchor pixel marks the center that will be overwritten in the output.

    Rule:
    - Scan every 2x2 window from top to bottom, then left to right, and collect pure-color 2x2 squares.
    - Keep only the first square of each color, then sort the kept squares by their top-left coordinates.
    - Let N be the number of kept squares; draw N concentric square rings centered on the grey anchor, using the sorted square colors from outermost to innermost.

    Key insight:
    - The same detection pass provides both the ring count and the ordered color sequence, while duplicate-color 2x2 distractors only matter through scan-order filtering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor = _find_anchor(g)
    squares = _valid_squares(g)
    assert 2 <= len(squares) <= 5, "expected 2 to 5 valid square colors"

    colors = [color for _, _, color in squares]
    return _render_rings(g.shape, anchor, colors)


def _mark_square_padding(mask: np.ndarray, r: int, c: int) -> None:
    h, w = mask.shape
    r0, r1 = max(0, r - 1), min(h - 1, r + 2)
    c0, c1 = max(0, c - 1), min(w - 1, c + 2)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _reserve_anchor_area(shape: tuple[int, int], anchor: tuple[int, int], layers: int) -> np.ndarray:
    h, w = shape
    ar, ac = anchor
    reserve = np.zeros((h, w), dtype=bool)
    pad = layers
    r0, r1 = max(0, ar - pad), min(h - 1, ar + pad)
    c0, c1 = max(0, ac - pad), min(w - 1, ac + pad)
    reserve[r0 : r1 + 1, c0 : c1 + 1] = True
    return reserve


def _candidate_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [(r, c) for r in range(h - 1) for c in range(w - 1) if not mask[r : r + 2, c : c + 2].any()]


def _choose_square_positions(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    total_squares: int,
    layer_count: int,
    require_same_row: bool,
) -> list[tuple[int, int]] | None:
    occupied = _reserve_anchor_area(shape, anchor, layer_count)
    chosen: list[tuple[int, int]] = []

    if require_same_row:
        row_groups: dict[int, list[int]] = {}
        for r, c in _candidate_positions(occupied):
            row_groups.setdefault(r, []).append(c)
        pair_options = []
        for r, cols in row_groups.items():
            cols = sorted(cols)
            valid_pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols)) if cols[j] - cols[i] >= 3]
            if valid_pairs:
                pair_options.append((r, random.choice(valid_pairs)))
        if not pair_options:
            return None
        row, (c0, c1) = random.choice(pair_options)
        for c in (c0, c1):
            chosen.append((row, c))
            _mark_square_padding(occupied, row, c)

    while len(chosen) < total_squares:
        candidates = _candidate_positions(occupied)
        if not candidates:
            return None
        r, c = random.choice(candidates)
        chosen.append((r, c))
        _mark_square_padding(occupied, r, c)

    return chosen


def _assign_square_colors(
    positions: list[tuple[int, int]],
    valid_colors: list[int],
    duplicate_count: int,
) -> dict[tuple[int, int], int]:
    sorted_positions = sorted(positions)
    color_multiset = list(valid_colors) + random.sample(valid_colors, duplicate_count)
    random.shuffle(color_multiset)

    if len(valid_colors) >= 3:
        attempt = 0
        while attempt < 40:
            visible_order = [color for _, color in sorted(zip(sorted_positions, color_multiset))]
            first_order = []
            seen = set()
            for color in visible_order:
                if color not in seen:
                    seen.add(color)
                    first_order.append(color)
            if first_order != sorted(first_order) and first_order != sorted(first_order, reverse=True):
                break
            random.shuffle(color_multiset)
            attempt += 1

    return {pos: color for pos, color in zip(sorted_positions, color_multiset)}


def _can_place_noise(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    if grid[r, c] != BACKGROUND:
        return False
    h, w = grid.shape
    for r0 in range(max(0, r - 1), min(h - 2, r) + 1):
        for c0 in range(max(0, c - 1), min(w - 2, c) + 1):
            block = grid[r0 : r0 + 2, c0 : c0 + 2].copy()
            block[r - r0, c - c0] = color
            first = int(block[0, 0])
            if first not in (BACKGROUND, ANCHOR_COLOR) and np.all(block == first):
                return False
    return True


def _place_noise(grid: np.ndarray, forbidden: np.ndarray, count: int) -> None:
    palette = SQUARE_COLORS + [6, 8, 9]
    candidates = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if not forbidden[r, c]]
    random.shuffle(candidates)
    placed = 0
    for r, c in candidates:
        color = random.choice(palette)
        if _can_place_noise(grid, r, c, color):
            grid[r, c] = color
            placed += 1
            if placed == count:
                return
    assert placed >= max(1, count // 2), "expected at least a few noise pixels"


def _analyze_output(grid: np.ndarray, out: np.ndarray) -> None:
    anchor = _find_anchor(grid)
    squares = _valid_squares(grid)
    colors = [color for _, _, color in squares]
    expected = _render_rings(grid.shape, anchor, colors)
    assert np.array_equal(out, expected), "output does not match the ring construction rule"

    ar, ac = anchor
    layer_count = len(colors)
    for depth, color in enumerate(colors):
        radius = layer_count - 1 - depth
        rr, cc = np.where(out == color)
        assert rr.size > 0
        assert max(max(abs(int(r) - ar), abs(int(c) - ac)) for r, c in zip(rr, cc)) >= radius


def _generate_attempt(profile: dict[str, int | bool]) -> np.ndarray | None:
    layer_count = random.randint(2, 5)
    h = random.randint(13, 18)
    w = random.randint(13, 18)

    min_anchor_r = max(layer_count - 1, h // 2)
    max_anchor_r = h - layer_count
    min_anchor_c = max(layer_count - 1, w // 2)
    max_anchor_c = w - layer_count
    if min_anchor_r > max_anchor_r or min_anchor_c > max_anchor_c:
        return None

    anchor = (
        random.randint(min_anchor_r, max_anchor_r),
        random.randint(min_anchor_c, max_anchor_c),
    )
    duplicate_count = int(profile["duplicate_count"])
    total_squares = layer_count + duplicate_count
    require_same_row = bool(profile["same_row"])

    positions = _choose_square_positions((h, w), anchor, total_squares, layer_count, require_same_row)
    if positions is None:
        return None

    valid_colors = random.sample(SQUARE_COLORS, layer_count)
    position_colors = _assign_square_colors(positions, valid_colors, duplicate_count)

    grid = np.zeros((h, w), dtype=int)
    for (r, c), color in position_colors.items():
        grid[r : r + 2, c : c + 2] = color
    grid[anchor] = ANCHOR_COLOR

    forbidden = np.zeros_like(grid, dtype=bool)
    forbidden[anchor] = True
    reserve = _reserve_anchor_area(grid.shape, anchor, layer_count)
    forbidden |= reserve
    noise_count = random.randint(4, 12) + int(profile["extra_noise"])
    _place_noise(grid, forbidden, noise_count)

    squares = _scan_monochrome_2x2(grid)
    valid = _valid_squares(grid)
    if len(valid) != layer_count:
        return None
    if any(color not in SQUARE_COLORS for _, _, color in valid):
        return None
    if duplicate_count > 0:
        all_colors = [color for _, _, color in squares]
        if max(Counter(all_colors).values()) < 2:
            return None
    if require_same_row:
        valid_rows = [r for r, _, _ in valid]
        if len(valid_rows) == len(set(valid_rows)):
            return None
    if len(valid) >= 3:
        seq = [color for _, _, color in valid]
        if seq == sorted(seq) or seq == sorted(seq, reverse=True):
            return None

    out = main(grid)
    _analyze_output(grid, out)

    signature = (out.shape, out.tobytes())
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    profiles = [
        {"duplicate_count": 0, "same_row": False, "extra_noise": 0},
        {"duplicate_count": 1, "same_row": True, "extra_noise": 1},
        {"duplicate_count": 2, "same_row": False, "extra_noise": 2},
        {"duplicate_count": 1, "same_row": False, "extra_noise": 0},
    ]
    profile = profiles[_GEN_CALL_COUNT % len(profiles)]
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(profile)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
