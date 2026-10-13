from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

VALID_MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
CORNER_LAYOUT = (
    ((0, 0), BLUE),
    ((0, 1), RED),
    ((1, 0), GREEN),
    ((1, 1), YELLOW),
)

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
RECT_SIZE_RANGE = (2, 6)
DECOY_OUTER_SIZE_RANGE = (4, 8)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 800

_RECENT_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _find_valid_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    blue_positions = {tuple(map(int, coord)) for coord in np.argwhere(grid == BLUE)}
    red_positions = {tuple(map(int, coord)) for coord in np.argwhere(grid == RED)}
    green_positions = {tuple(map(int, coord)) for coord in np.argwhere(grid == GREEN)}
    yellow_positions = {tuple(map(int, coord)) for coord in np.argwhere(grid == YELLOW)}

    rectangles = set()
    for top_r, left_c in blue_positions:
        for bottom_r, right_c in yellow_positions:
            if bottom_r - top_r < 3 or right_c - left_c < 3:
                continue
            if (top_r, right_c) not in red_positions:
                continue
            if (bottom_r, left_c) not in green_positions:
                continue
            rectangles.add((top_r + 1, left_c + 1, bottom_r - 1, right_c - 1))
    return sorted(rectangles)


def _apply_rectangles(grid: np.ndarray, rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r0, c0, r1, c1 in rectangles:
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BLACK] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct grey solid rectangles from complete color-coded corner remnants.

    Input structure:
    - The grid uses black as background and contains only scattered single-pixel
      blue, red, green, and yellow marker clues.
    - A legal clue group has blue, red, green, and yellow at the northwest,
      northeast, southwest, and southeast diagonal outside corners of one erased
      axis-aligned rectangle, with all four markers aligned by rows and columns.
    - Additional markers may be decoys: missing one corner, misaligned, or using
      the right geometry with the wrong color-role assignment.

    Rule:
    - Find every four-marker group that simultaneously satisfies the required
      color-role placement and row/column alignment.
    - For each legal group, contract the marker bbox inward by one cell on all
      sides and fill that interior rectangle with grey.
    - Preserve every original marker pixel and leave all other background cells
      unchanged.

    Key insight:
    - The visible colored pixels are not the target objects themselves; they are
      directional corner residues whose colors identify which aligned quartets
      really encode erased rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rectangles = _find_valid_rectangles(g)
    return _apply_rectangles(g, rectangles)


def _boxes_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_boxes(
    height: int,
    width: int,
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    placed: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        box_h, box_w = sizes[idx]
        candidates = [
            (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            for r0 in range(height - box_h + 1)
            for c0 in range(width - box_w + 1)
        ]
        random.shuffle(candidates)
        chosen = None
        for candidate in candidates:
            if all(_boxes_apart(candidate, other) for other in placed.values()):
                chosen = candidate
                break
        if chosen is None:
            return None
        placed[idx] = chosen

    return [placed[idx] for idx in range(len(sizes))]


def _identity_colors() -> list[int]:
    return [BLUE, RED, GREEN, YELLOW]


def _wrong_role_colors() -> list[int]:
    colors = _identity_colors()
    while True:
        shuffled = random.sample(colors, len(colors))
        if shuffled != colors:
            return shuffled


def _render_valid_markers(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    top, left, bottom, right = box
    grid[top, left] = BLUE
    grid[top, right] = RED
    grid[bottom, left] = GREEN
    grid[bottom, right] = YELLOW
    return (top + 1, left + 1, bottom - 1, right - 1)


def _render_wrong_role_decoy(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = box
    corners = [(top, left), (top, right), (bottom, left), (bottom, right)]
    for (r, c), color in zip(corners, _wrong_role_colors()):
        grid[r, c] = color


def _render_missing_corner_decoy(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = box
    correct_markers = [
        (top, left, BLUE),
        (top, right, RED),
        (bottom, left, GREEN),
        (bottom, right, YELLOW),
    ]
    keep_count = random.choice((2, 3, 3))
    for r, c, color in random.sample(correct_markers, keep_count):
        grid[r, c] = color


def _render_misaligned_decoy(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = box
    grid[top, left] = BLUE
    grid[bottom, right] = YELLOW
    if random.random() < 0.5:
        grid[top + 1, right] = RED
        grid[bottom, left] = GREEN
    else:
        grid[top, right] = RED
        grid[bottom, left + 1] = GREEN


def _add_single_marker_decoys(grid: np.ndarray, count: int) -> None:
    free_cells = [tuple(map(int, coord)) for coord in np.argwhere(grid == BLACK)]
    random.shuffle(free_cells)
    for (r, c), color in zip(free_cells, random.choices(VALID_MARKER_COLORS, k=count)):
        grid[r, c] = color


def _expected_from_truth(grid: np.ndarray, truth_rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    return _apply_rectangles(grid, sorted(truth_rectangles))


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (tuple(map(int, grid.shape)), grid.tobytes())


def _validate_scene(
    grid: np.ndarray,
    truth_rectangles: list[tuple[int, int, int, int]],
    required_decoys: set[str],
) -> bool:
    if not truth_rectangles:
        return False

    found_rectangles = _find_valid_rectangles(grid)
    if set(found_rectangles) != set(truth_rectangles):
        return False

    marker_count = int(np.count_nonzero(np.isin(grid, VALID_MARKER_COLORS)))
    if marker_count <= 4 * len(truth_rectangles):
        return False

    expected = _expected_from_truth(grid, truth_rectangles)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    for r0, c0, r1, c1 in truth_rectangles:
        if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
            return False
        if np.any(np.isin(grid[r0 : r1 + 1, c0 : c1 + 1], VALID_MARKER_COLORS)):
            return False

    if "wrong_role" in required_decoys:
        has_wrong_role = False
        for top in range(grid.shape[0]):
            for left in range(grid.shape[1]):
                for bottom in range(top + 3, grid.shape[0]):
                    for right in range(left + 3, grid.shape[1]):
                        colors = [
                            int(grid[top, left]),
                            int(grid[top, right]),
                            int(grid[bottom, left]),
                            int(grid[bottom, right]),
                        ]
                        if sorted(colors) == sorted(_identity_colors()) and colors != _identity_colors():
                            has_wrong_role = True
                            break
                    if has_wrong_role:
                        break
                if has_wrong_role:
                    break
            if has_wrong_role:
                break
        if not has_wrong_role:
            return False

    if "all_corners_required" in required_decoys:
        has_incomplete = False
        for top in range(grid.shape[0]):
            for left in range(grid.shape[1]):
                for bottom in range(top + 3, grid.shape[0]):
                    for right in range(left + 3, grid.shape[1]):
                        present = [
                            int(grid[top, left]) == BLUE,
                            int(grid[top, right]) == RED,
                            int(grid[bottom, left]) == GREEN,
                            int(grid[bottom, right]) == YELLOW,
                        ]
                        if sum(present) in (2, 3):
                            has_incomplete = True
                            break
                    if has_incomplete:
                        break
                if has_incomplete:
                    break
            if has_incomplete:
                break
        if not has_incomplete:
            return False

    if _output_signature(expected) in _RECENT_OUTPUT_SIGNATURES:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    valid_count = random.choices([1, 2, 3, 4], weights=[2, 3, 3, 1], k=1)[0]
    extra_decoy_count = random.randint(0, 3)
    decoy_types = ["wrong_role", random.choice(["missing", "misaligned"])]
    decoy_types.extend(random.choice(["wrong_role", "missing", "misaligned"]) for _ in range(extra_decoy_count))

    specs: list[tuple[str, tuple[int, int]]] = []
    for _ in range(valid_count):
        rect_h = random.randint(*RECT_SIZE_RANGE)
        rect_w = random.randint(*RECT_SIZE_RANGE)
        specs.append(("valid", (rect_h + 2, rect_w + 2)))
    for decoy_type in decoy_types:
        outer_h = random.randint(*DECOY_OUTER_SIZE_RANGE)
        outer_w = random.randint(*DECOY_OUTER_SIZE_RANGE)
        specs.append((decoy_type, (outer_h, outer_w)))

    placements = _place_boxes(height, width, [size for _, size in specs])
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    truth_rectangles: list[tuple[int, int, int, int]] = []
    required_decoys = {"wrong_role", "all_corners_required"}

    for (kind, _size), box in zip(specs, placements):
        if kind == "valid":
            truth_rectangles.append(_render_valid_markers(grid, box))
        elif kind == "wrong_role":
            _render_wrong_role_decoy(grid, box)
        elif kind == "missing":
            _render_missing_corner_decoy(grid, box)
        else:
            _render_misaligned_decoy(grid, box)

    _add_single_marker_decoys(grid, random.randint(0, 3))

    if not _validate_scene(grid, truth_rectangles, required_decoys):
        return None

    output = _expected_from_truth(grid, truth_rectangles)
    _RECENT_OUTPUT_SIGNATURES.add(_output_signature(output))
    if len(_RECENT_OUTPUT_SIGNATURES) > 256:
        _RECENT_OUTPUT_SIGNATURES.pop()
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
