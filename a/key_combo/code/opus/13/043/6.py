from __future__ import annotations

import random
from typing import Any

import numpy as np


PALETTE = (1, 3, 4, 6)
BACKGROUND = 0
SQUARE_SIZE = 2
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIR_TO_MARKER_OFFSET = {
    (-1, -1): (-1, -1),
    (-1, 1): (-1, 2),
    (1, -1): (2, -1),
    (1, 1): (2, 2),
}
DIR_TO_ORIGIN_OFFSET = {
    (-1, -1): (1, 1),
    (-1, 1): (1, -2),
    (1, -1): (-2, 1),
    (1, 1): (-2, -2),
}
ANCHOR_PAIRS = (
    ((1, 1), (1, -1)),
    ((-1, 1), (-1, -1)),
    ((1, 1), (-1, 1)),
    ((1, -1), (-1, -1)),
)
GRID_RANGE = (18, 24)
EXTRA_MARKER_PROB = 0.45
GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _square_cells(origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = origin
    return [
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0 + 1, c0 + 1),
    ]


def _marker_cell(origin: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = origin
    dr, dc = DIR_TO_MARKER_OFFSET[direction]
    return r0 + dr, c0 + dc


def _ray_cells(
    shape: tuple[int, int],
    marker: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = marker[0] + dr, marker[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _dist2_chebyshev(cell: tuple[int, int], center2: tuple[int, int]) -> int:
    r, c = cell
    return max(abs(2 * r - center2[0]), abs(2 * c - center2[1]))


def _separated_from(
    new_cells: list[tuple[int, int]] | set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    gap: int = 1,
) -> bool:
    for r, c in new_cells:
        for rr, cc in occupied:
            if max(abs(r - rr), abs(c - cc)) <= gap:
                return False
    return True


def _make_block(
    block_id: int,
    color: int,
    origin: tuple[int, int],
    directions: list[tuple[int, int]],
) -> dict[str, Any]:
    return {
        "id": int(block_id),
        "color": int(color),
        "origin": tuple(map(int, origin)),
        "center2": (2 * origin[0] + 1, 2 * origin[1] + 1),
        "directions": [tuple(map(int, d)) for d in directions],
    }


def _object_cells(block: dict[str, Any]) -> set[tuple[int, int]]:
    cells = set(_square_cells(block["origin"]))
    for direction in block["directions"]:
        cells.add(_marker_cell(block["origin"], direction))
    return cells


def _parse_blocks(grid: np.ndarray) -> list[dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND)
    assert 2 <= len(colors) <= 4, "expected 2-4 colored source squares"
    assert all(color in PALETTE for color in colors), "unexpected source color"

    blocks: list[dict[str, Any]] = []
    seen_origins: set[tuple[int, int]] = set()
    for block_id, color in enumerate(colors):
        color_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        origins: list[tuple[int, int]] = []
        for r in range(g.shape[0] - 1):
            for c in range(g.shape[1] - 1):
                if np.all(g[r : r + 2, c : c + 2] == color):
                    origins.append((r, c))
        assert len(origins) == 1, f"color {color} must contain exactly one 2x2 square"
        origin = origins[0]
        assert origin not in seen_origins
        seen_origins.add(origin)

        markers: list[tuple[int, int]] = []
        for direction in DIAGONALS:
            marker = _marker_cell(origin, direction)
            if _in_bounds(g.shape, marker[0], marker[1]) and int(g[marker]) == color:
                markers.append(marker)
        assert 1 <= len(markers) <= 2, f"color {color} must have 1-2 diagonal markers"

        expected_cells = set(_square_cells(origin)) | set(markers)
        assert color_cells == expected_cells, f"color {color} contains extra pixels"

        directions = [
            direction
            for direction in DIAGONALS
            if _marker_cell(origin, direction) in markers
        ]
        blocks.append(_make_block(block_id, color, origin, directions))
    return blocks


def _collect_claims(
    grid: np.ndarray,
    blocks: list[dict[str, Any]],
) -> dict[tuple[int, int], set[int]]:
    g = np.asarray(grid, dtype=int)
    claims: dict[tuple[int, int], set[int]] = {}
    for block in blocks:
        for direction in block["directions"]:
            marker = _marker_cell(block["origin"], direction)
            for cell in _ray_cells(g.shape, marker, direction):
                assert int(g[cell]) == BACKGROUND, "ray continuation must traverse only background"
                claims.setdefault(cell, set()).add(int(block["id"]))
    return claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend rays from multiple marked 2x2 squares and color each contested diagonal cell by its nearest source square.

    Input structure:
    - The grid has black background and 2-4 solid 2x2 source squares in distinct colors from {1,3,4,6}.
    - Each square has 1-2 same-color diagonal marker pixels, each sitting exactly one diagonal step outside one corner.
    - No other foreground pixels exist; the square cells and marker cells are the full input content.

    Rule:
    - Keep every original square cell and marker cell unchanged.
    - For each marker, continue outward in the same diagonal direction to the grid edge, claiming only background cells on that ray.
    - If a claimed background cell belongs to one square, color it with that square's color; if several squares claim it, choose the square with minimum Chebyshev distance to its 2x2 center, breaking ties by smaller center row then smaller center column.

    Key insight:
    - The markers define diagonal rays exactly as in the single-source task, but crossing cells are resolved by nearest-source ownership rather than a fixed layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks = _parse_blocks(g)
    claims = _collect_claims(g, blocks)

    blocks_by_id = {int(block["id"]): block for block in blocks}
    out = g.copy()
    for cell, claimant_ids in claims.items():
        winner_id = min(
            claimant_ids,
            key=lambda block_id: (
                _dist2_chebyshev(cell, blocks_by_id[block_id]["center2"]),
                blocks_by_id[block_id]["center2"][0],
                blocks_by_id[block_id]["center2"][1],
            ),
        )
        out[cell] = blocks_by_id[winner_id]["color"]
    return out


def _valid_anchor_steps(
    shape: tuple[int, int],
    cross: tuple[int, int],
    direction: tuple[int, int],
) -> list[int]:
    h, w = shape
    out: list[int] = []
    dr, dc = direction
    for steps in range(3, max(h, w)):
        marker = (cross[0] - steps * dr, cross[1] - steps * dc)
        origin = (
            marker[0] + DIR_TO_ORIGIN_OFFSET[direction][0],
            marker[1] + DIR_TO_ORIGIN_OFFSET[direction][1],
        )
        if not (1 <= origin[0] <= h - 4 and 1 <= origin[1] <= w - 4):
            continue
        ray = _ray_cells(shape, marker, direction)
        if cross not in ray:
            continue
        if len(ray) - steps < 2:
            continue
        out.append(steps)
    return out


def _build_anchor_blocks(
    shape: tuple[int, int],
    colors: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    h, w = shape
    cross = (random.randint(6, h - 7), random.randint(6, w - 7))
    pair = random.choice(ANCHOR_PAIRS)
    tie_mode = random.random() < 0.45

    steps_a = _valid_anchor_steps(shape, cross, pair[0])
    steps_b = _valid_anchor_steps(shape, cross, pair[1])
    if not steps_a or not steps_b:
        return None

    if tie_mode:
        common = sorted(set(steps_a) & set(steps_b))
        if not common:
            return None
        step1 = step2 = random.choice(common)
    else:
        candidates = [(a, b) for a in steps_a for b in steps_b if abs(a - b) >= 1]
        if not candidates:
            return None
        step1, step2 = random.choice(candidates)

    blocks: list[dict[str, Any]] = []
    occupied: set[tuple[int, int]] = set()
    for block_id, (color, direction, steps) in enumerate(
        ((colors[0], pair[0], step1), (colors[1], pair[1], step2))
    ):
        marker = (cross[0] - steps * direction[0], cross[1] - steps * direction[1])
        origin = (
            marker[0] + DIR_TO_ORIGIN_OFFSET[direction][0],
            marker[1] + DIR_TO_ORIGIN_OFFSET[direction][1],
        )
        block = _make_block(block_id, color, origin, [direction])
        cells = _object_cells(block)
        if not _separated_from(cells, occupied, gap=1):
            return None
        if any(cell in occupied for cell in _ray_cells(shape, marker, direction)):
            return None
        blocks.append(block)
        occupied |= cells

    return blocks, {
        "cross": cross,
        "anchor_colors": (colors[0], colors[1]),
        "tie_mode": tie_mode,
        "pair": pair,
        "steps": (step1, step2),
    }


def _candidate_extra_directions(
    shape: tuple[int, int],
    origin: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for direction in DIAGONALS:
        marker = _marker_cell(origin, direction)
        if not _in_bounds(shape, marker[0], marker[1]):
            continue
        if not _separated_from([marker], occupied, gap=1):
            continue
        ray = _ray_cells(shape, marker, direction)
        if len(ray) < 2:
            continue
        if any(cell in occupied for cell in ray):
            continue
        options.append(direction)
    return options


def _add_extra_block(
    shape: tuple[int, int],
    block_id: int,
    color: int,
    occupied: set[tuple[int, int]],
) -> dict[str, Any] | None:
    h, w = shape
    for _ in range(120):
        origin = (random.randint(1, h - 4), random.randint(1, w - 4))
        square = _square_cells(origin)
        if not _separated_from(square, occupied, gap=1):
            continue
        directions = _candidate_extra_directions(shape, origin, occupied)
        if not directions:
            continue
        primary = random.choice(directions)
        chosen = [primary]

        remaining = [direction for direction in directions if direction != primary]
        if remaining and random.random() < EXTRA_MARKER_PROB:
            secondary = random.choice(remaining)
            chosen.append(secondary)

        block = _make_block(block_id, color, origin, chosen)
        if not _separated_from(_object_cells(block), occupied, gap=1):
            continue
        return block
    return None


def _render_input(blocks: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for block in blocks:
        for r, c in _square_cells(block["origin"]):
            grid[r, c] = block["color"]
        for direction in block["directions"]:
            marker = _marker_cell(block["origin"], direction)
            grid[marker] = block["color"]
    return grid


def _validate_generated(grid: np.ndarray, meta: dict[str, Any]) -> bool:
    try:
        blocks = _parse_blocks(grid)
        claims = _collect_claims(grid, blocks)
        if not claims:
            return False
        contested = {cell: ids for cell, ids in claims.items() if len(ids) >= 2}
        if not contested:
            return False
        if not any(len(ids) == 1 for ids in claims.values()):
            return False

        cross = tuple(meta["cross"])
        if cross not in contested:
            return False

        blocks_by_color = {int(block["color"]): block for block in blocks}
        color_a, color_b = meta["anchor_colors"]
        block_a = blocks_by_color[color_a]
        block_b = blocks_by_color[color_b]
        cross_ids = contested[cross]
        if block_a["id"] not in cross_ids or block_b["id"] not in cross_ids:
            return False

        dist_a = _dist2_chebyshev(cross, block_a["center2"])
        dist_b = _dist2_chebyshev(cross, block_b["center2"])
        if meta["tie_mode"]:
            if dist_a != dist_b:
                return False
        else:
            if dist_a == dist_b:
                return False

        solved = main(grid)
        if np.array_equal(solved, grid):
            return False
        if solved.shape != grid.shape:
            return False

        winner_color = int(solved[cross])
        expected_winner = min(
            (block_a, block_b),
            key=lambda block: (
                _dist2_chebyshev(cross, block["center2"]),
                block["center2"][0],
                block["center2"][1],
            ),
        )["color"]
        if winner_color != expected_winner:
            return False

        new_pixels = int(np.count_nonzero((grid == BACKGROUND) & (solved != BACKGROUND)))
        if new_pixels < 8:
            return False
        return True
    except Exception:
        return False


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    n_blocks = random.randint(2, 4)
    colors = random.sample(list(PALETTE), n_blocks)

    anchor_result = _build_anchor_blocks((h, w), colors)
    if anchor_result is None:
        return None
    blocks, meta = anchor_result

    occupied = set().union(*(_object_cells(block) for block in blocks))
    for block_id in range(2, n_blocks):
        block = _add_extra_block((h, w), block_id, colors[block_id], occupied)
        if block is None:
            return None
        blocks.append(block)
        occupied |= _object_cells(block)

    for block in blocks:
        if len(block["directions"]) == 1 and random.random() < EXTRA_MARKER_PROB:
            occupied_without_self = occupied - _object_cells(block)
            options = [
                direction
                for direction in _candidate_extra_directions(
                    (h, w), block["origin"], occupied_without_self
                )
                if direction not in block["directions"]
            ]
            if options:
                block["directions"].append(random.choice(options))
                occupied |= _object_cells(block)

    grid = _render_input(blocks, (h, w))
    return grid if _validate_generated(grid, meta) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
