from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)

CORNER_SPECS = {
    "NW": (-1, -1, BLUE),
    "NE": (-1, 2, RED),
    "SW": (2, -1, GREEN),
    "SE": (2, 2, YELLOW),
}
SIDE_TO_SUPPRESSED = {
    "north": ("NW", "NE"),
    "south": ("SW", "SE"),
    "west": ("NW", "SW"),
    "east": ("NE", "SE"),
}
SIDE_TO_GROW = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}
SIDE_ORDER = ("north", "south", "west", "east")
SIZE_RANGE = (14, 19)
MAX_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _marker_positions_for_side(r0: int, c0: int, side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "north":
        return (r0 - 1, c0), (r0 - 1, c0 + 1)
    if side == "south":
        return (r0 + 2, c0), (r0 + 2, c0 + 1)
    if side == "west":
        return (r0, c0 - 1), (r0 + 1, c0 - 1)
    if side == "east":
        return (r0, c0 + 2), (r0 + 1, c0 + 2)
    raise AssertionError(f"unknown side {side!r}")


def _extract_blocks(grid: np.ndarray) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []

    for coords in _components_of_color(grid, GREY):
        assert len(coords) == 4, "each grey component must be a 2x2 square"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "grey component must have bbox 2x2"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey bbox must be solid"

        marker_hits: list[tuple[str, int, int, int]] = []
        for side in SIDE_ORDER:
            for mr, mc in _marker_positions_for_side(r0, c0, side):
                if _in_bounds(grid.shape, mr, mc) and int(grid[mr, mc]) in MARKER_COLORS:
                    marker_hits.append((side, mr, mc, int(grid[mr, mc])))

        assert len(marker_hits) == 1, "each grey square must have exactly one orthogonally adjacent colored marker"
        side, mr, mc, marker_color = marker_hits[0]
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "side": side,
                "marker": (mr, mc),
                "marker_color": marker_color,
            }
        )

    assert blocks, "expected at least one grey block"
    blocks.sort(key=lambda item: item["bbox"])
    return blocks


def _apply_rule(
    grid: np.ndarray,
    blocks: list[dict[str, object]],
    *,
    suppress: bool = True,
    allow_exception: bool = True,
    grow: bool = True,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)

    for block in blocks:
        r0, c0, _r1, _c1 = block["bbox"]
        side = str(block["side"])
        mr, mc = block["marker"]
        marker_color = int(block["marker_color"])

        if grow:
            dr, dc = SIDE_TO_GROW[side]
            gr, gc = mr + dr, mc + dc
            assert _in_bounds(out.shape, gr, gc), "growth target must stay on-grid"
            out[gr, gc] = marker_color

        suppressed = set(SIDE_TO_SUPPRESSED[side]) if suppress else set()
        for corner_name, (dr, dc, color) in CORNER_SPECS.items():
            rr = r0 + dr
            cc = c0 + dc
            assert _in_bounds(out.shape, rr, cc), "corner marker must stay on-grid"
            if corner_name in suppressed and not (allow_exception and color == marker_color):
                continue
            out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each side marker outward and decorate each grey 2x2 square with side-suppressed corner colors plus a color-matching exception.

    Input structure:
    - The grid has black background, 3-4 isolated grey 2x2 squares, and one colored marker pixel orthogonally adjacent to each square.
    - Each marker sits on exactly one side of its square (north, south, west, or east) and uses one of blue/red/green/yellow.
    - Around every square, the diagonal corner cells and the one-step outward growth cells are empty in the input.

    Rule:
    - Keep every grey square and every original marker unchanged.
    - For each square, determine which side contains its adjacent marker, then add one same-colored pixel one step farther away from the square in that direction.
    - Also decorate the four diagonal outside corners with fixed colors: northwest=blue, northeast=red, southwest=green, southeast=yellow.
    - Suppress the two corners on the same side as the marker, except that any suppressed corner whose fixed color matches the marker color is restored and drawn normally.

    Key insight:
    - The marker's side controls both the growth direction and which pair of corner decorations is suppressed, while the marker's color can selectively cancel one suppressed corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(Counter(map(int, g.flat)).most_common(1)[0][0]) == BLACK, "background must be black"
    blocks = _extract_blocks(g)
    return _apply_rule(g, blocks)


def _suppressed_colors(side: str) -> tuple[int, int]:
    return tuple(CORNER_SPECS[name][2] for name in SIDE_TO_SUPPRESSED[side])


def _non_suppressed_marker_colors(side: str) -> list[int]:
    blocked = set(_suppressed_colors(side))
    return [color for color in MARKER_COLORS if color not in blocked]


def _zone_disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) >= 6 or abs(a[1] - b[1]) >= 6


def _sample_positions(h: int, w: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(2, h - 3) for c in range(2, w - 3)]
    placed: list[tuple[int, int]] = []

    for idx in range(count):
        available = [pos for pos in candidates if all(_zone_disjoint(pos, other) for other in placed)]
        if not available:
            return None
        if idx + 1 < count:
            viable = []
            remaining = count - idx - 1
            for pos in available:
                future = sum(
                    all(_zone_disjoint(candidate, other) for other in placed + [pos]) for candidate in candidates
                )
                if future >= remaining:
                    viable.append(pos)
            if viable:
                available = viable
        placed.append(available[int(rng.integers(len(available)))])
    return placed


def _build_input_grid(
    h: int,
    w: int,
    placements: list[tuple[int, int]],
    sides: list[str],
    marker_colors: list[int],
    offsets: list[int],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    grid = np.full((h, w), BLACK, dtype=int)
    blocks: list[dict[str, object]] = []

    for (r0, c0), side, marker_color, offset in zip(placements, sides, marker_colors, offsets):
        grid[r0 : r0 + 2, c0 : c0 + 2] = GREY
        marker = _marker_positions_for_side(r0, c0, side)[offset]
        grid[marker] = marker_color
        blocks.append(
            {
                "bbox": (r0, c0, r0 + 1, c0 + 1),
                "side": side,
                "marker": marker,
                "marker_color": marker_color,
            }
        )

    blocks.sort(key=lambda item: item["bbox"])
    return grid, blocks


def _validate_case(grid: np.ndarray, blocks: list[dict[str, object]]) -> bool:
    parsed = _extract_blocks(grid)
    if parsed != blocks:
        return False

    if len(blocks) not in (3, 4):
        return False
    sides = [str(block["side"]) for block in blocks]
    if len(set(sides)) != len(sides):
        return False

    exception_flags = []
    for block in blocks:
        side = str(block["side"])
        marker_color = int(block["marker_color"])
        exception_flags.append(marker_color in _suppressed_colors(side))
    if not any(exception_flags) or all(exception_flags):
        return False

    if len(set(int(block["marker_color"]) for block in blocks)) < 3:
        return False

    expected = _apply_rule(grid, blocks)
    if np.array_equal(expected, grid):
        return False
    if np.array_equal(_apply_rule(grid, blocks, suppress=False), expected):
        return False
    if np.array_equal(_apply_rule(grid, blocks, allow_exception=False), expected):
        return False
    if np.array_equal(_apply_rule(grid, blocks, grow=False), expected):
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    for block in blocks:
        r0, c0, _r1, _c1 = block["bbox"]
        side = str(block["side"])
        marker_color = int(block["marker_color"])
        drawn_same_side = 0
        suppressed_same_side = 0
        for name in SIDE_TO_SUPPRESSED[side]:
            dr, dc, color = CORNER_SPECS[name]
            rr, cc = r0 + dr, c0 + dc
            if int(expected[rr, cc]) == color:
                drawn_same_side += 1
            else:
                suppressed_same_side += 1
        if marker_color in _suppressed_colors(side):
            if drawn_same_side != 1 or suppressed_same_side != 1:
                return False
        else:
            if drawn_same_side != 0 or suppressed_same_side != 2:
                return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*SIZE_RANGE))
    w = int(rng.integers(*SIZE_RANGE))
    count = 3 if rng.random() < 0.4 else 4
    placements = _sample_positions(h, w, count, rng)
    if placements is None:
        return None

    sides = list(rng.choice(np.array(SIDE_ORDER), size=count, replace=False))
    rng.shuffle(sides)

    exception_count = int(rng.integers(1, count))
    exception_indices = set(rng.choice(count, size=exception_count, replace=False).tolist())

    marker_colors: list[int] = []
    offsets: list[int] = []
    for idx, side in enumerate(sides):
        if idx in exception_indices:
            options = list(_suppressed_colors(side))
        else:
            options = _non_suppressed_marker_colors(side)
        marker_colors.append(int(rng.choice(np.array(options))))
        offsets.append(int(rng.integers(0, 2)))

    grid, blocks = _build_input_grid(h, w, placements, sides, marker_colors, offsets)
    if not _validate_case(grid, blocks):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
