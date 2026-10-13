from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

DIRECTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

CLAIM_PRIORITY = {
    "down": 0,
    "right": 1,
    "left": 2,
    "up": 3,
}

ARROW_MASKS = {
    "up": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=bool,
    ),
}

MAX_GENERATION_ATTEMPTS = 400
CELL_SIZE_MIN = 5
CELL_SIZE_MAX = 7


def _axis_ranges(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    ranges: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for i, size in enumerate(sizes):
        ranges.append((pos, pos + size))
        pos += size
        if i != len(sizes) - 1:
            separators.append(pos)
            pos += 1
    return ranges, separators, pos


def _cell_ranges(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    sep_rows = [-1] + [r for r in range(h) if bool(np.all(grid[r] == GREY))] + [h]
    sep_cols = [-1] + [c for c in range(w) if bool(np.all(grid[:, c] == GREY))] + [w]

    row_ranges = [(a + 1, b) for a, b in zip(sep_rows, sep_rows[1:]) if a + 1 < b]
    col_ranges = [(a + 1, b) for a, b in zip(sep_cols, sep_cols[1:]) if a + 1 < b]
    return row_ranges, col_ranges


def _detect_arrow(patch: np.ndarray) -> dict[str, int | str] | None:
    fg = (patch != BLACK) & (patch != GREY)
    if not bool(np.any(fg)):
        return None

    colors = np.unique(patch[fg])
    if len(colors) != 1:
        return None

    color = int(colors[0])
    mask = patch == color
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    for direction, template in ARROW_MASKS.items():
        if sub.shape == template.shape and bool(np.array_equal(sub, template)):
            return {
                "color": color,
                "direction": direction,
                "area": int(mask.sum()),
            }
    return None


def _step(
    cell: tuple[int, int],
    direction: str,
    row_count: int,
    col_count: int,
) -> tuple[int, int] | None:
    dr, dc = DIRECTION_DELTAS[direction]
    rr = cell[0] + dr
    cc = cell[1] + dc
    if 0 <= rr < row_count and 0 <= cc < col_count:
        return rr, cc
    return None


def _incoming_sources(
    cell: tuple[int, int],
    row_count: int,
    col_count: int,
) -> list[tuple[tuple[int, int], str]]:
    r, c = cell
    sources: list[tuple[tuple[int, int], str]] = []
    if r > 0:
        sources.append(((r - 1, c), "down"))
    if r + 1 < row_count:
        sources.append(((r + 1, c), "up"))
    if c > 0:
        sources.append(((r, c - 1), "right"))
    if c + 1 < col_count:
        sources.append(((r, c + 1), "left"))
    return sources


def _outgoing_directions(cell: tuple[int, int], row_count: int, col_count: int) -> list[str]:
    r, c = cell
    directions: list[str] = []
    if r > 0:
        directions.append("up")
    if r + 1 < row_count:
        directions.append("down")
    if c > 0:
        directions.append("left")
    if c + 1 < col_count:
        directions.append("right")
    return directions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill empty table cells that are pointed to by colored arrow sprites in neighboring cells.

    Input structure:
    - The background is black.
    - Full grey rows and columns partition the canvas into rectangular subregions.
    - Some subregions contain a monochrome arrow: a solid 3x3 square with a
      2-pixel single-cell tip extending from one side center.
    - Other subregions are empty black rectangles.

    Rule:
    - Detect each arrow direction from its protruding 2-pixel tip and map it to
      the immediately adjacent subregion across the grey separator.
    - If that target subregion is empty, fill the whole target rectangle with
      the arrow color.
    - If multiple arrows target the same empty subregion, choose the arrow with
      the largest foreground area; ties use source-side priority
      up > left > right > down.
    - Subregions that already contain an arrow or any non-empty content stay
      unchanged, and arrow cells themselves are never modified.

    Key insight:
    - The grey lines define a region graph, while the arrow tip encodes which
      neighboring region receives the color propagation.
    """
    g = np.asarray(grid, dtype=int)
    row_ranges, col_ranges = _cell_ranges(g)
    row_count = len(row_ranges)
    col_count = len(col_ranges)

    cells: dict[tuple[int, int], dict[str, object]] = {}
    claims: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)

    for rr, (r0, r1) in enumerate(row_ranges):
        for cc, (c0, c1) in enumerate(col_ranges):
            patch = g[r0:r1, c0:c1]
            arrow = _detect_arrow(patch)
            if bool(np.all(patch == BLACK)):
                kind = "empty"
            elif arrow is not None:
                kind = "arrow"
            else:
                kind = "occupied"

            cells[(rr, cc)] = {
                "bounds": (r0, r1, c0, c1),
                "kind": kind,
                "arrow": arrow,
            }

            if arrow is None:
                continue
            direction = str(arrow["direction"])
            target = _step((rr, cc), direction, row_count, col_count)
            if target is None:
                continue
            claims[target].append(
                (
                    int(arrow["area"]),
                    CLAIM_PRIORITY[direction],
                    int(arrow["color"]),
                )
            )

    out = g.copy()
    for target, options in claims.items():
        cell = cells.get(target)
        if cell is None or cell["kind"] != "empty":
            continue
        best_area, best_priority, best_color = min(options, key=lambda item: (-item[0], item[1]))
        _ = (best_area, best_priority)
        r0, r1, c0, c1 = cell["bounds"]
        out[r0:r1, c0:c1] = best_color

    return out


def _sample_axis_sizes(count: int) -> list[int]:
    for _ in range(200):
        sizes = [random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX) for _ in range(count)]
        if sum(sizes) + count - 1 <= 30:
            return sizes
    raise RuntimeError("failed to sample valid axis sizes")


def _simulate_plan(
    row_count: int,
    col_count: int,
    arrows: dict[tuple[int, int], dict[str, int | str]],
) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    claims: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    winners: dict[tuple[int, int], int] = {}

    for cell, data in arrows.items():
        direction = str(data["direction"])
        target = _step(cell, direction, row_count, col_count)
        if target is None:
            continue
        claims[target].append((11, CLAIM_PRIORITY[direction], int(data["color"])))

    single_targets = 0
    competition_targets = 0
    blocked_claims = 0

    for target, options in claims.items():
        if target in arrows:
            blocked_claims += len(options)
            continue
        if len(options) == 1:
            single_targets += 1
        else:
            competition_targets += 1
        winners[target] = min(options, key=lambda item: (-item[0], item[1]))[2]

    stats = {
        "single_targets": single_targets,
        "competition_targets": competition_targets,
        "blocked_claims": blocked_claims,
    }
    return winners, stats


def _paint_arrow(
    grid: np.ndarray,
    row_bounds: tuple[int, int],
    col_bounds: tuple[int, int],
    color: int,
    direction: str,
) -> None:
    mask = ARROW_MASKS[direction]
    r0, r1 = row_bounds
    c0, c1 = col_bounds
    cell_h = r1 - r0
    cell_w = c1 - c0
    mask_h, mask_w = mask.shape

    top = random.randint(0, cell_h - mask_h)
    left = random.randint(0, cell_w - mask_w)
    patch = grid[r0 + top : r0 + top + mask_h, c0 + left : c0 + left + mask_w]
    patch[mask] = color


def _render_input(
    row_heights: list[int],
    col_widths: list[int],
    arrows: dict[tuple[int, int], dict[str, int | str]],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_ranges, sep_rows, height = _axis_ranges(row_heights)
    col_ranges, sep_cols, width = _axis_ranges(col_widths)

    grid = np.full((height, width), BLACK, dtype=int)
    for r in sep_rows:
        grid[r, :] = GREY
    for c in sep_cols:
        grid[:, c] = GREY

    for (rr, cc), data in arrows.items():
        _paint_arrow(
            grid,
            row_ranges[rr],
            col_ranges[cc],
            int(data["color"]),
            str(data["direction"]),
        )

    return grid, row_ranges, col_ranges


def _build_expected_output(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    winners: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for (rr, cc), color in winners.items():
        r0, r1 = row_ranges[rr]
        c0, c1 = col_ranges[cc]
        out[r0:r1, c0:c1] = color
    return out


def _plan_case() -> tuple[list[int], list[int], dict[tuple[int, int], dict[str, int | str]]] | None:
    row_count = random.randint(3, 4)
    col_count = random.randint(3, 4)
    row_heights = _sample_axis_sizes(row_count)
    col_widths = _sample_axis_sizes(col_count)

    cells = [(r, c) for r in range(row_count) for c in range(col_count)]
    arrows: dict[tuple[int, int], dict[str, int | str]] = {}
    reserved_empty: set[tuple[int, int]] = set()

    competition_candidates = [cell for cell in cells if len(_incoming_sources(cell, row_count, col_count)) >= 2]
    if not competition_candidates:
        return None
    competition_target = random.choice(competition_candidates)
    reserved_empty.add(competition_target)
    claimants = _incoming_sources(competition_target, row_count, col_count)
    claimant_count = random.randint(2, min(4, len(claimants)))
    for source, direction in random.sample(claimants, claimant_count):
        arrows[source] = {"direction": direction}

    used = set(arrows) | reserved_empty
    blocked_options: list[tuple[tuple[int, int], list[tuple[tuple[int, int], str]]]] = []
    for cell in cells:
        if cell in used:
            continue
        incoming = [item for item in _incoming_sources(cell, row_count, col_count) if item[0] not in used]
        if incoming and _outgoing_directions(cell, row_count, col_count):
            blocked_options.append((cell, incoming))
    if not blocked_options:
        return None

    blocked_target, _ = random.choice(blocked_options)
    arrows[blocked_target] = {"direction": random.choice(_outgoing_directions(blocked_target, row_count, col_count))}
    used.add(blocked_target)

    blocked_sources = [item for item in _incoming_sources(blocked_target, row_count, col_count) if item[0] not in used]
    if not blocked_sources:
        return None
    source, direction = random.choice(blocked_sources)
    arrows[source] = {"direction": direction}
    used.add(source)

    free_options: list[tuple[tuple[int, int], list[tuple[tuple[int, int], str]]]] = []
    for cell in cells:
        if cell in used:
            continue
        incoming = [item for item in _incoming_sources(cell, row_count, col_count) if item[0] not in used]
        if incoming:
            free_options.append((cell, incoming))
    if not free_options:
        return None

    free_target, incoming = random.choice(free_options)
    reserved_empty.add(free_target)
    used.add(free_target)
    source, direction = random.choice(incoming)
    arrows[source] = {"direction": direction}
    used.add(source)

    remaining = [cell for cell in cells if cell not in used]
    random.shuffle(remaining)
    extra_count = random.randint(0, min(2, len(remaining)))
    extras_added = 0
    for cell in remaining:
        if extras_added >= extra_count:
            break
        if random.random() < 0.65:
            arrows[cell] = {"direction": random.choice(_outgoing_directions(cell, row_count, col_count))}
            extras_added += 1

    if len(arrows) > len(PALETTE):
        return None

    colors = random.sample(PALETTE, len(arrows))
    arrow_cells = list(arrows)
    random.shuffle(arrow_cells)
    for cell, color in zip(arrow_cells, colors):
        arrows[cell]["color"] = color

    winners, stats = _simulate_plan(row_count, col_count, arrows)
    if not winners:
        return None
    if stats["single_targets"] == 0 or stats["competition_targets"] == 0 or stats["blocked_claims"] == 0:
        return None

    return row_heights, col_widths, arrows


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        plan = _plan_case()
        if plan is None:
            continue

        row_heights, col_widths, arrows = plan
        row_count = len(row_heights)
        col_count = len(col_widths)
        winners, stats = _simulate_plan(row_count, col_count, arrows)
        if stats["single_targets"] == 0 or stats["competition_targets"] == 0 or stats["blocked_claims"] == 0:
            continue

        grid, row_ranges, col_ranges = _render_input(row_heights, col_widths, arrows)
        expected = _build_expected_output(grid, row_ranges, col_ranges, winners)

        try:
            actual = main(grid)
        except Exception:
            continue

        if not bool(np.array_equal(actual, expected)):
            continue
        if actual.shape != grid.shape or max(grid.shape) > 30 or max(actual.shape) > 30:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for task 4")
