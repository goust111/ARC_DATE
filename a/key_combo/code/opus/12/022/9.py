from __future__ import annotations

import itertools
import random
from collections.abc import Iterable

import numpy as np


BACKGROUND = 0
MARKER_COLORS = {
    "ul": 1,
    "ur": 2,
    "lr": 3,
    "ll": 4,
}
MARKER_PALETTES = (
    {"ul": 1, "ur": 2, "lr": 3, "ll": 4},
    {"ul": 1, "ur": 3, "lr": 5, "ll": 7},
    {"ul": 2, "ur": 4, "lr": 6, "ll": 9},
    {"ul": 3, "ur": 5, "lr": 7, "ll": 1},
    {"ul": 4, "ur": 6, "lr": 9, "ll": 2},
    {"ul": 5, "ur": 1, "lr": 6, "ll": 9},
    {"ul": 1, "ur": 2, "lr": 5, "ll": 6},
    {"ul": 2, "ur": 3, "lr": 6, "ll": 7},
    {"ul": 1, "ur": 4, "lr": 6, "ll": 9},
    {"ul": 2, "ur": 5, "lr": 7, "ll": 9},
    {"ul": 3, "ur": 4, "lr": 6, "ll": 7},
    {"ul": 1, "ur": 5, "lr": 7, "ll": 9},
)
SPRITE_COLOR = 8
CLOCKWISE = ("ul", "ur", "lr", "ll")
OUTPUT_ANCHORS = {
    "ul": (0, 0),
    "ur": (0, 3),
    "ll": (3, 0),
    "lr": (3, 3),
}
MASK_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
MASKS_BY_AREA = {
    area: [tuple(mask) for mask in itertools.combinations(MASK_OFFSETS, area)]
    for area in (1, 2, 3)
}

INPUT_HEIGHTS = (6, 7, 8)
INPUT_WIDTHS = (6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400
AREA_WEIGHTS = (12, 3, 3)  # counterbalance rejection so 1/2/3-cell masks all appear often


def _corner_slice(name: str, height: int, width: int) -> tuple[slice, slice]:
    if name == "ul":
        return slice(0, 2), slice(0, 2)
    if name == "ur":
        return slice(0, 2), slice(width - 2, width)
    if name == "ll":
        return slice(height - 2, height), slice(0, 2)
    if name == "lr":
        return slice(height - 2, height), slice(width - 2, width)
    raise ValueError(f"unknown corner: {name}")


def _marker_colors(grid: np.ndarray) -> dict[str, int]:
    height, width = grid.shape
    colors: dict[str, int] = {}
    for name in CLOCKWISE:
        rs, cs = _corner_slice(name, height, width)
        region = grid[rs, cs]
        marker_cells = np.argwhere((region != BACKGROUND) & (region != SPRITE_COLOR))
        assert len(marker_cells) == 1, f"expected one marker in {name} corner"
        r, c = map(int, marker_cells[0])
        colors[name] = int(region[r, c])
    return colors


def _teal_cells(grid: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(map(int, rc)) for rc in np.argwhere(grid == SPRITE_COLOR))


def _candidate_placements(
    fg: frozenset[tuple[int, int]],
    height: int,
    width: int,
    mask: Iterable[tuple[int, int]],
) -> list[frozenset[tuple[int, int]]]:
    placements: list[frozenset[tuple[int, int]]] = []
    offsets = tuple(mask)
    for r0 in range(height - 1):
        for c0 in range(width - 1):
            cells = frozenset((r0 + dr, c0 + dc) for dr, dc in offsets)
            if cells <= fg:
                placements.append(cells)
    return placements


def _cover_exists(
    fg: frozenset[tuple[int, int]],
    placements: list[frozenset[tuple[int, int]]],
    area: int,
) -> bool:
    by_cell = {cell: [] for cell in fg}
    for placement in placements:
        for cell in placement:
            by_cell[cell].append(placement)

    dead_states: set[tuple[frozenset[tuple[int, int]], int]] = set()

    def backtrack(covered: frozenset[tuple[int, int]], chosen: int) -> bool:
        state = (covered, chosen)
        if state in dead_states:
            return False
        if chosen == 4:
            return covered == fg

        remaining = fg - covered
        if len(remaining) != (4 - chosen) * area:
            dead_states.add(state)
            return False

        pivot = min(
            remaining,
            key=lambda cell: sum(1 for p in by_cell[cell] if not (p & covered)),
        )
        options = [placement for placement in by_cell[pivot] if not (placement & covered)]
        random.shuffle(options)
        for placement in options:
            if backtrack(covered | placement, chosen + 1):
                return True

        dead_states.add(state)
        return False

    return backtrack(frozenset(), 0)


def _find_sprite_masks(grid: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    fg = _teal_cells(grid)
    if not fg or len(fg) % 4 != 0:
        return []

    area = len(fg) // 4
    if area not in MASKS_BY_AREA:
        return []

    height, width = grid.shape
    valid_masks: list[tuple[tuple[int, int], ...]] = []
    for mask in MASKS_BY_AREA[area]:
        placements = _candidate_placements(fg, height, width, mask)
        if placements and _cover_exists(fg, placements, area):
            valid_masks.append(tuple(sorted(mask)))
    return valid_masks


def _render_output(
    mask: Iterable[tuple[int, int]],
    corner_colors: dict[str, int],
) -> np.ndarray:
    mask_set = set(mask)
    out = np.zeros((5, 5), dtype=int)
    for index, name in enumerate(CLOCKWISE):
        own = int(corner_colors[name])
        borrow = int(corner_colors[CLOCKWISE[(index + 1) % 4]])
        r0, c0 = OUTPUT_ANCHORS[name]
        for dr in range(2):
            for dc in range(2):
                out[r0 + dr, c0 + dc] = own if (dr, dc) in mask_set else borrow
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Separate a hidden 2x2 sprite into four corners, then fill each copy's gaps with the next corner's color.

    Input structure:
    - The grid is mostly black, with teal pixels that are exactly the union of four non-overlapping translated copies of one 2x2 sprite mask.
    - Each input corner 2x2 region contains exactly one non-teal marker pixel.
    - The four marker pixels identify the corner colors that will recolor the separated sprite copies.

    Rule:
    - Infer the unique 2x2 sprite mask whose four translated copies cover all teal pixels.
    - Build a 5x5 output with that mask stamped into the four 2x2 corners, leaving the middle row and column black.
    - Recolor each corner copy's sprite cells with its own corner marker color.
    - Fill that corner copy's missing mask cells with the clockwise next corner's marker color: upper-left uses upper-right, upper-right uses lower-right, lower-right uses lower-left, and lower-left uses upper-left.

    Key insight:
    - The input combines two independent clues: teal pixels reveal the shared sprite mask, while the four corner markers provide a cyclic color legend used to fill the separated copies' internal gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 4

    masks = _find_sprite_masks(g)
    assert len(masks) == 1, "expected a unique sprite mask"

    corner_colors = _marker_colors(g)
    return _render_output(masks[0], corner_colors)


def _mask_cells(anchor: tuple[int, int], mask: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0 = anchor
    return frozenset((r0 + dr, c0 + dc) for dr, dc in mask)


def _choose_anchors(height: int, width: int, mask: Iterable[tuple[int, int]]) -> list[tuple[int, int]] | None:
    anchors = [(r, c) for r in range(height - 1) for c in range(width - 1)]
    cell_map = {anchor: _mask_cells(anchor, mask) for anchor in anchors}

    def backtrack(chosen: list[tuple[int, int]], occupied: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == 4:
            return chosen

        candidates = [anchor for anchor in anchors if not (cell_map[anchor] & occupied)]
        random.shuffle(candidates)
        for anchor in candidates:
            result = backtrack(chosen + [anchor], occupied | cell_map[anchor])
            if result is not None:
                return result
        return None

    return backtrack([], frozenset())


def _place_markers(grid: np.ndarray, marker_colors: dict[str, int]) -> bool:
    height, width = grid.shape
    for name, color in marker_colors.items():
        rs, cs = _corner_slice(name, height, width)
        region = grid[rs, cs]
        empties = [tuple(map(int, rc)) for rc in np.argwhere(region == BACKGROUND)]
        if not empties:
            return False
        r, c = random.choice(empties)
        region[r, c] = color
    return True


def _spans_both_axes(grid: np.ndarray) -> bool:
    coords = np.argwhere(grid == SPRITE_COLOR)
    if len(coords) == 0:
        return False
    span = coords.max(axis=0) - coords.min(axis=0)
    return bool(span[0] >= 2 and span[1] >= 2)


def _is_valid_generated_input(
    grid: np.ndarray,
    mask: tuple[tuple[int, int], ...],
    marker_colors: dict[str, int],
) -> bool:
    if set(map(int, np.unique(grid))) - ({BACKGROUND, SPRITE_COLOR} | set(marker_colors.values())):
        return False

    if np.count_nonzero(grid == SPRITE_COLOR) != 4 * len(mask):
        return False

    for color in marker_colors.values():
        if np.count_nonzero(grid == color) != 1:
            return False

    height, width = grid.shape
    for name, color in marker_colors.items():
        rs, cs = _corner_slice(name, height, width)
        region = grid[rs, cs]
        if np.count_nonzero(region == color) != 1:
            return False

    masks = _find_sprite_masks(grid)
    if masks != [tuple(sorted(mask))]:
        return False

    expected = _render_output(mask, marker_colors)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    height = random.choice(INPUT_HEIGHTS)
    width = random.choice(INPUT_WIDTHS)
    area = random.choices((1, 2, 3), weights=AREA_WEIGHTS, k=1)[0]
    mask = tuple(sorted(random.choice(MASKS_BY_AREA[area])))
    marker_colors = dict(random.choice(MARKER_PALETTES))

    anchors = _choose_anchors(height, width, mask)
    if anchors is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for anchor in anchors:
        for r, c in _mask_cells(anchor, mask):
            grid[r, c] = SPRITE_COLOR

    if not _spans_both_axes(grid):
        return None
    if not _place_markers(grid, marker_colors):
        return None
    if not _is_valid_generated_input(grid, mask, marker_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
