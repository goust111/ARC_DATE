from __future__ import annotations

from collections import Counter
import random

import numpy as np


FRAME_COLOR = 6
DIAGONAL_COLORS = (1, 2, 3)
BLACK = 0
MAX_GEN_ATTEMPTS = 800

_SEEN_OUTPUTS: set[bytes] = set()
_GEN_CALLS = 0


def _frame_candidates(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Return (area, r0, c0, r1, c1, color) for every monochrome rectangular frame."""
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    color = int(g[r0, c0])
                    if (
                        int(g[r0, c1]) != color
                        or int(g[r1, c0]) != color
                        or int(g[r1, c1]) != color
                    ):
                        continue
                    if (
                        np.all(g[r0, c0 : c1 + 1] == color)
                        and np.all(g[r1, c0 : c1 + 1] == color)
                        and np.all(g[r0 : r1 + 1, c0] == color)
                        and np.all(g[r0 : r1 + 1, c1] == color)
                    ):
                        area = int((r1 - r0 - 1) * (c1 - c0 - 1))
                        candidates.append((area, int(r0), int(c0), int(r1), int(c1), color))
    return candidates


def _find_largest_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = _frame_candidates(grid)
    assert candidates, "expected at least one monochrome rectangular frame"
    max_area = max(area for area, *_rest in candidates)
    largest = [item for item in candidates if item[0] == max_area]
    assert len(largest) == 1, "expected a unique largest rectangular frame"
    _area, r0, c0, r1, c1, _color = largest[0]
    return int(r0), int(c0), int(r1), int(c1)


def _diag_length_slash(height: int, width: int, total: int) -> int:
    r_min = max(0, total - (width - 1))
    r_max = min(height - 1, total)
    return max(0, r_max - r_min + 1)


def _diag_length_backslash(height: int, width: int, diff: int) -> int:
    r_min = max(0, diff)
    r_max = min(height - 1, width - 1 + diff)
    return max(0, r_max - r_min + 1)


def _detect_diagonal_pattern(canvas: np.ndarray) -> tuple[str, dict[int, int]]:
    g = np.asarray(canvas, dtype=int)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != BLACK)
    assert len(colors) == 3, "expected exactly three diagonal colors"

    def valid(orientation: str) -> dict[int, int] | None:
        mapping: dict[int, int] = {}
        for color in colors:
            coords = np.argwhere(g == color)
            if orientation == "backslash":
                invariants = {int(r - c) for r, c in coords}
            else:
                invariants = {int(r + c) for r, c in coords}
            if len(invariants) != 1:
                return None
            invariant = next(iter(invariants))
            residue = invariant % 3
            if residue in mapping:
                return None
            mapping[residue] = int(color)

            if orientation == "backslash":
                for r in range(g.shape[0]):
                    c = int(r - invariant)
                    if 0 <= c < g.shape[1] and int(g[r, c]) != int(color):
                        return None
            else:
                for r in range(g.shape[0]):
                    c = int(invariant - r)
                    if 0 <= c < g.shape[1] and int(g[r, c]) != int(color):
                        return None
        if set(mapping) != {0, 1, 2}:
            return None
        return mapping

    options = []
    for orientation in ("backslash", "slash"):
        mapping = valid(orientation)
        if mapping is not None:
            options.append((orientation, mapping))
    assert len(options) == 1, "expected a unique diagonal orientation"
    return options[0]


def _fill_from_pattern(shape: tuple[int, int], orientation: str, mapping: dict[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    out = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            residue = (r - c) % 3 if orientation == "backslash" else (r + c) % 3
            out[r, c] = int(mapping[residue])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop inside the largest rectangular frame, then expand three seed diagonals into a modulo-3 tiling.

    Input structure:
    - A noisy grid contains one unique largest 1-pixel-thick monochrome rectangular frame.
    - The desired working canvas is strictly inside that frame.
    - Inside the frame, black background contains exactly three complete diagonals in three distinct colors.
    - The three diagonals share one slope direction and occupy the three modulo-3 diagonal classes.

    Rule:
    - Find the unique largest monochrome rectangular frame and crop to its strict interior.
    - Detect whether the three colored diagonals are parallel to the main or anti-diagonal direction.
    - Read which color occupies each modulo-3 diagonal class in that direction.
    - Fill every cell of the cropped canvas with the color of its matching modulo-3 class.

    Key insight:
    - The outer frame is only a spatial filter: once cropped, each visible seed diagonal identifies one congruence class,
      and repeating the three classes fills the entire canvas without ambiguity.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _find_largest_frame_bbox(g)
    canvas = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    orientation, mapping = _detect_diagonal_pattern(canvas)
    return _fill_from_pattern(canvas.shape, orientation, mapping)


def _choose_sizes(rng: random.Random) -> tuple[int, int, int, int]:
    if _GEN_CALLS % 4 == 0:
        inner_h = rng.randint(7, 9)
        inner_w = rng.randint(10, 13)
    elif _GEN_CALLS % 4 == 1:
        inner_h = rng.randint(10, 13)
        inner_w = rng.randint(7, 9)
    else:
        inner_h = rng.randint(7, 12)
        inner_w = rng.randint(7, 12)

    outer_h = rng.randint(inner_h + 4, min(30, inner_h + 12))
    outer_w = rng.randint(inner_w + 4, min(30, inner_w + 12))
    return inner_h, inner_w, outer_h, outer_w


def _pick_invariants(height: int, width: int, orientation: str, rng: random.Random) -> dict[int, int] | None:
    selected: dict[int, int] = {}
    min_length = max(4, min(height, width) // 2)
    if orientation == "backslash":
        pool = list(range(-(width - 1), height))
        rng.shuffle(pool)
        for diff in pool:
            residue = diff % 3
            if residue in selected:
                continue
            if _diag_length_backslash(height, width, diff) < min_length:
                continue
            selected[residue] = int(diff)
            if len(selected) == 3:
                return selected
        return None

    pool = list(range(height + width - 1))
    rng.shuffle(pool)
    for total in pool:
        residue = total % 3
        if residue in selected:
            continue
        if _diag_length_slash(height, width, total) < min_length:
            continue
        selected[residue] = int(total)
        if len(selected) == 3:
            return selected
    return None


def _paint_seed_canvas(
    height: int,
    width: int,
    orientation: str,
    residue_to_color: dict[int, int],
    residue_to_invariant: dict[int, int],
) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=int)
    for residue, color in residue_to_color.items():
        invariant = residue_to_invariant[residue]
        if orientation == "backslash":
            for r in range(height):
                c = int(r - invariant)
                if 0 <= c < width:
                    canvas[r, c] = int(color)
        else:
            for r in range(height):
                c = int(invariant - r)
                if 0 <= c < width:
                    canvas[r, c] = int(color)
    return canvas


def _add_noise(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], rng: random.Random) -> None:
    h, w = map(int, grid.shape)
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (r0 <= r <= r1 and c0 <= c <= c1)
    ]
    rng.shuffle(outside)
    palette = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    edge_targets = []
    for c in range(w):
        edge_targets.append((0, c))
        edge_targets.append((h - 1, c))
    for r in range(1, h - 1):
        edge_targets.append((r, 0))
        edge_targets.append((r, w - 1))
    edge_targets = list(dict.fromkeys(edge_targets))
    for r, c in edge_targets:
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        grid[r, c] = int(palette[(r + 2 * c) % len(palette)])

    noise_count = rng.randint(max(8, len(outside) // 6), max(12, len(outside) // 3))
    for r, c in outside[:noise_count]:
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        choices = [color for color in palette if color != int(grid[r, c])]
        grid[r, c] = int(rng.choice(choices))


def _validate_generated_case(
    grid: np.ndarray,
    expected_output: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    orientation: str,
    residue_to_color: dict[int, int],
) -> bool:
    try:
        result = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(result, expected_output):
        return False

    candidates = _frame_candidates(grid)
    main_matches = [
        item
        for item in candidates
        if item[1:5] == frame_bbox and int(item[5]) == FRAME_COLOR
    ]
    if len(main_matches) != 1:
        return False
    if sum(1 for item in candidates if item[0] >= main_matches[0][0]) != 1:
        return False

    r0, c0, r1, c1 = map(int, frame_bbox)
    canvas = grid[r0 + 1 : r1, c0 + 1 : c1]
    try:
        found_orientation, mapping = _detect_diagonal_pattern(canvas)
    except AssertionError:
        return False
    if found_orientation != orientation or mapping != residue_to_color:
        return False

    if set(map(int, np.unique(expected_output))) != set(DIAGONAL_COLORS):
        return False
    if np.any(expected_output == BLACK):
        return False

    signature = expected_output.tobytes() + bytes(expected_output.shape)
    if signature in _SEEN_OUTPUTS:
        return False
    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    global _GEN_CALLS

    rng = random.Random((random.randrange(1 << 30) << 8) ^ _GEN_CALLS)
    inner_h, inner_w, grid_h, grid_w = _choose_sizes(rng)
    orientation = "backslash" if _GEN_CALLS % 2 == 0 else "slash"
    if rng.random() < 0.4:
        orientation = "slash" if orientation == "backslash" else "backslash"

    residue_to_invariant = _pick_invariants(inner_h, inner_w, orientation, rng)
    if residue_to_invariant is None:
        return None
    colors = list(DIAGONAL_COLORS)
    rng.shuffle(colors)
    residue_to_color = {residue: int(color) for residue, color in zip((0, 1, 2), colors)}

    seed_canvas = _paint_seed_canvas(inner_h, inner_w, orientation, residue_to_color, residue_to_invariant)
    expected_output = _fill_from_pattern(seed_canvas.shape, orientation, residue_to_color)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if grid_h < outer_h or grid_w < outer_w:
        return None
    r0 = rng.randint(0, grid_h - outer_h)
    c0 = rng.randint(0, grid_w - outer_w)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 + 1 : r1, c0 + 1 : c1] = seed_canvas
    _add_noise(grid, (r0, c0, r1, c1), rng)

    if not _validate_generated_case(
        grid,
        expected_output,
        (r0, c0, r1, c1),
        orientation,
        residue_to_color,
    ):
        return None

    _SEEN_OUTPUTS.add(expected_output.tobytes() + bytes(expected_output.shape))
    _GEN_CALLS += 1
    return grid, expected_output


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _generate_attempt()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
