from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DELTA = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}
DELTA_TO_SIDE = {v: k for k, v in SIDE_TO_DELTA.items()}


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_unique_marker(grid: np.ndarray) -> set[tuple[int, int]]:
    comps = _component_sets(grid == MAROON)
    assert len(comps) == 1, "expected exactly one maroon marker component"
    marker = comps[0]
    assert 4 <= len(marker) <= 8, "marker must have 4-8 cells"
    return marker


def _find_blue_objects(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    comps = _component_sets(grid == BLUE)
    assert 2 <= len(comps) <= 3, "expected 2-3 blue objects"
    for comp in comps:
        assert 3 <= len(comp) <= 5, "blue objects must have 3-5 cells"
    return comps


def _contact_side(
    obj: set[tuple[int, int]], marker: set[tuple[int, int]]
) -> str:
    sides: set[str] = set()
    for r, c in obj:
        if (r + 1, c) in marker:
            sides.add("north")
        if (r - 1, c) in marker:
            sides.add("south")
        if (r, c + 1) in marker:
            sides.add("west")
        if (r, c - 1) in marker:
            sides.add("east")
    assert len(sides) == 1, "each blue object must touch the marker from exactly one side"
    return next(iter(sides))


def _shift_to_boundary(
    obj: set[tuple[int, int]], side: str, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    h, w = shape
    rows = [r for r, _ in obj]
    cols = [c for _, c in obj]
    dr, dc = SIDE_TO_DELTA[side]
    if dr == -1:
        amount = min(rows)
    elif dr == 1:
        amount = (h - 1) - max(rows)
    elif dc == -1:
        amount = min(cols)
    else:
        amount = (w - 1) - max(cols)
    return {(r + dr * amount, c + dc * amount) for r, c in obj}


def _enclosed_black_regions(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    blocked = grid != BLACK
    seen = np.zeros_like(blocked, dtype=bool)
    regions: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                regions.append(comp)
    return regions


def _move_all_blue_objects(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[set[tuple[int, int]], set[tuple[int, int]], str]]]:
    marker = _find_unique_marker(grid)
    blue_objects = _find_blue_objects(grid)
    sides = [_contact_side(obj, marker) for obj in blue_objects]
    assert len(set(sides)) == len(sides), "blue objects must touch different marker sides"

    moved_info = []
    occupied = set(marker)
    out = grid.copy()
    out[out == BLUE] = BLACK
    for obj, side in zip(blue_objects, sides):
        moved = _shift_to_boundary(obj, side, grid.shape)
        assert not (moved & occupied), "moved objects must not overlap marker or each other"
        occupied |= moved
        moved_info.append((obj, moved, side))
    for _, moved, _ in moved_info:
        for r, c in moved:
            out[r, c] = BLUE
    return out, moved_info


def _expected_output(grid: np.ndarray) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    moved_grid, moved_info = _move_all_blue_objects(grid)
    input_closed = {frozenset(region) for region in _enclosed_black_regions(grid)}
    output_closed = _enclosed_black_regions(moved_grid)
    to_fill = [region for region in output_closed if frozenset(region) not in input_closed]
    out = moved_grid.copy()
    for region in to_fill:
        for r, c in region:
            out[r, c] = YELLOW
    return out, to_fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each blue object away from the maroon marker to the grid edge, then fill only the black regions that become newly enclosed.

    Input structure:
    - The grid is a black background with one small maroon 4-connected marker component and 2-3 blue 4-connected objects.
    - Every blue object touches the marker from exactly one cardinal side, and different blue objects use different sides.
    - The marker itself is an open boundary fragment; after the blue objects are pushed to the border, some black regions may become enclosed by the moved blue cells, the maroon cells, and the grid boundary.

    Rule:
    - Keep the maroon marker fixed.
    - For each blue object, infer which side of the marker it touches and translate the whole object in the opposite direction until it first becomes flush with the corresponding grid edge.
    - Erase the original blue positions and draw all moved blue objects simultaneously.
    - Find every 4-connected black region that is enclosed after the moves, but ignore any region that was already enclosed before the moves.
    - Fill each newly enclosed black region with yellow.

    Key insight:
    - The fill decision must be delayed until all pushes are finished: only the post-move arrangement reveals which black regions are newly sealed by the marker, the moved blue objects, and the grid boundary together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    out, _ = _expected_output(g)
    return out


MARKER_TEMPLATE = {
    (0, 0),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
}
BLUE_TEMPLATES = {
    "east": {(-1, 2), (-1, 3), (0, 1), (0, 2)},
    "north": {(1, 4), (1, 5), (1, 6), (1, 7), (2, 6)},
    "south": {(4, 3), (4, 4), (4, 5), (5, 5), (5, 6)},
}


def _rotate_coords(
    coords: set[tuple[int, int]], shape: tuple[int, int], turns: int
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    h, w = shape
    out = set(coords)
    turns %= 4
    for _ in range(turns):
        out = {(c, h - 1 - r) for r, c in out}
        h, w = w, h
    return out, (h, w)


def _paint(
    grid: np.ndarray, coords: set[tuple[int, int]], color: int
) -> None:
    for r, c in coords:
        grid[r, c] = color


def _build_canonical_scene(rng: random.Random) -> np.ndarray:
    h = rng.randint(12, 16)
    w = rng.randint(12, 16)
    grid = np.zeros((h, w), dtype=int)

    anchor_top = 2
    anchor_left = w - 10
    marker = {(anchor_top + r, anchor_left + c) for r, c in MARKER_TEMPLATE}
    _paint(grid, marker, MAROON)

    sides = ["east", "north"]
    if rng.choice([False, True]):
        sides.append("south")
    for side in sides:
        coords = {(anchor_top + r, anchor_left + c) for r, c in BLUE_TEMPLATES[side]}
        assert all(0 <= r < h and 0 <= c < w for r, c in coords)
        assert not any(grid[r, c] != BLACK for r, c in coords)
        _paint(grid, coords, BLUE)

    marker_component = _find_unique_marker(grid)
    blue_objects = _find_blue_objects(grid)
    contact_sides = {_contact_side(obj, marker_component) for obj in blue_objects}
    assert contact_sides == set(sides)
    return grid


def _semantic_check(grid: np.ndarray) -> None:
    marker = _find_unique_marker(grid)
    blue_objects = _find_blue_objects(grid)
    sides = [_contact_side(obj, marker) for obj in blue_objects]
    assert len(set(sides)) == len(sides)

    moved_grid, moved_info = _move_all_blue_objects(grid)
    input_closed = {frozenset(region) for region in _enclosed_black_regions(grid)}
    output_closed = [frozenset(region) for region in _enclosed_black_regions(moved_grid)]
    new_regions = [region for region in output_closed if region not in input_closed]

    assert new_regions, "generator must create at least one newly enclosed region"
    assert any(
        any(r == 0 or r == grid.shape[0] - 1 or c == 0 or c == grid.shape[1] - 1 for r, c in moved)
        for _, moved, _ in moved_info
    ), "at least one object must end on the boundary"

    # Ensure the fill depends on the push, not on a pre-existing enclosure.
    assert not all(region in input_closed for region in output_closed)

    # Ensure at least one object's move is essential for producing the new enclosure.
    essential = False
    original_marker = marker
    original_objects = [(obj, side) for obj, _, side in moved_info]
    for held_obj, held_side in original_objects:
        test = grid.copy()
        test[test == BLUE] = BLACK
        occupied = set(original_marker)
        for obj, side in original_objects:
            placed = obj if obj is held_obj else _shift_to_boundary(obj, side, grid.shape)
            if placed & occupied:
                break
            occupied |= placed
            _paint(test, placed, BLUE)
        else:
            test_new = {
                frozenset(region)
                for region in _enclosed_black_regions(test)
                if frozenset(region) not in input_closed
            }
            if not test_new:
                essential = True
                break
    assert essential, "at least one moved object must be necessary for the new enclosure"

    expected, to_fill = _expected_output(grid)
    assert to_fill
    actual = main(grid)
    assert np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _build_canonical_scene(rng)
        turns = rng.randrange(4)
        rotated = np.rot90(grid, turns).copy()
        _semantic_check(rotated)
        return rotated
    raise RuntimeError("failed to generate a valid input")
