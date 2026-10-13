from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


# The helper conventions here follow arc_utils.py: coordinates are (row, column),
# connectivity is 4-way, and component coordinates are returned as (N, 2) arrays.
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = tuple(range(10))
_MAX_GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "background must be uniquely most frequent"
    return int(winners[0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return the 4-connected True components of a boolean mask."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for start_r in range(height):
        for start_c in range(width):
            if seen[start_r, start_c] or not mask[start_r, start_c]:
                continue
            queue = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append(np.asarray(cells, dtype=int))
    return result


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == width - 1)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every enclosed chamber with the code-row color whose frequency equals its area.

    Input structure:
    - The grid body contains a uniquely dominant background and one wall color; the walls
      enclose several completely empty 4-connected background chambers.
    - The full bottom row is a detached code row made from several colors other than the
      background and wall colors.

    Rule:
    - Within the body above the code row, separate 4-connected background components that
      touch the body boundary from those fully enclosed by walls.  For an enclosed chamber
      of area k, find the unique code-row color occurring exactly k times and flood-fill the
      entire chamber with it.  Preserve the body otherwise and clear the code row to background.

    Key insight:
    - Chamber position and code-row order are distractors: chamber area is the frequency key
      that uniquely pairs each enclosed region with its fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 4, "expected a 2D ARC grid"

    background = _background_color(g)
    body = g[:-1, :]
    code_row = g[-1, :]
    assert np.all(code_row != background), "the bottom code row must be fully colored"

    body_colors = set(map(int, np.unique(body))) - {background}
    assert len(body_colors) == 1, "the body must contain exactly one wall color"
    wall = body_colors.pop()
    assert wall not in set(map(int, code_row)), "the code row must be detached from the wall"

    code_counts = Counter(map(int, code_row))
    assert len(code_counts) >= 2, "the code row must contain several colors"

    enclosed: list[np.ndarray] = []
    for component in _components(body == background):
        if not _touches_boundary(component, body.shape):
            enclosed.append(component)
    assert enclosed, "expected at least one enclosed chamber"

    out = g.copy()
    for chamber in enclosed:
        area = int(chamber.shape[0])
        matches = [color for color, frequency in code_counts.items() if frequency == area]
        assert len(matches) == 1, "each chamber area must select one unique code color"
        out[chamber[:, 0], chamber[:, 1]] = int(matches[0])
    out[-1, :] = background
    return out


def _random_polyomino(area: int, require_irregular: bool) -> np.ndarray | None:
    """Grow a compact connected chamber mask with exactly the requested area."""
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in _NEIGHBORS4
                if (r + dr, c + dc) not in cells
            }
            candidates = []
            for candidate in frontier:
                trial = cells | {candidate}
                rows = [r for r, _c in trial]
                cols = [c for _r, c in trial]
                if max(rows) - min(rows) + 1 <= 4 and max(cols) - min(cols) + 1 <= 4:
                    candidates.append(candidate)
            if not candidates:
                break
            # Favor cells with two existing neighbors often enough to produce compact shapes,
            # while retaining concave and bent polyominoes as counterexamples.
            weights = []
            for r, c in candidates:
                adjacent = sum((r + dr, c + dc) in cells for dr, dc in _NEIGHBORS4)
                weights.append(1 + 2 * adjacent)
            cells.add(random.choices(candidates, weights=weights, k=1)[0])
        if len(cells) != area:
            continue

        min_r = min(r for r, _c in cells)
        min_c = min(c for _r, c in cells)
        normalized = {(r - min_r, c - min_c) for r, c in cells}
        height = max(r for r, _c in normalized) + 1
        width = max(c for _r, c in normalized) + 1
        if area >= 3 and (height < 2 or width < 2):
            continue
        if require_irregular and height * width == area:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in normalized:
            mask[r, c] = True
        return mask
    return None


def _sample_frequency_plan() -> tuple[list[int], list[int]] | None:
    """Choose distinct chamber areas and nonmatching distractor frequencies."""
    chamber_count = random.choice((2, 3, 4))
    areas = random.sample(range(2, 9), chamber_count)
    distractor_count = random.choice((1, 2))
    nonmatching = [count for count in range(1, 11) if count not in areas]
    if len(nonmatching) < distractor_count:
        return None
    distractor_frequencies = random.sample(nonmatching, distractor_count)
    width = sum(areas) + sum(distractor_frequencies)
    if not 16 <= width <= 30:
        return None
    return areas, distractor_frequencies


def _place_chambers(
    body_height: int, width: int, areas: list[int]
) -> tuple[np.ndarray, list[set[tuple[int, int]]]] | None:
    """Place separated polyomino chambers and paint their one-cell wall boundaries."""
    wall_mask = np.zeros((body_height, width), dtype=bool)
    reserved = np.zeros_like(wall_mask)
    chambers: list[set[tuple[int, int]]] = []
    irregular_index = random.choice([i for i, area in enumerate(areas) if area >= 3])

    for index, area in enumerate(areas):
        chamber_mask = _random_polyomino(area, require_irregular=(index == irregular_index))
        if chamber_mask is None:
            return None
        chamber_height, chamber_width = chamber_mask.shape
        patch_height, patch_width = chamber_height + 2, chamber_width + 2
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, body_height - patch_height):
            for c0 in range(1, width - patch_width):
                if not reserved[r0 : r0 + patch_height, c0 : c0 + patch_width].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)

        local_cells = {
            (int(r) + 1, int(c) + 1) for r, c in np.argwhere(chamber_mask)
        }
        local_wall = {
            (r + dr, c + dc)
            for r, c in local_cells
            for dr, dc in _NEIGHBORS4
            if (r + dr, c + dc) not in local_cells
        }
        for r, c in local_wall:
            wall_mask[r0 + r, c0 + c] = True
        chambers.append({(r0 + r, c0 + c) for r, c in local_cells})

        rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
        rr1 = min(body_height, r0 + patch_height + 1)
        cc1 = min(width, c0 + patch_width + 1)
        reserved[rr0:rr1, cc0:cc1] = True

    return wall_mask, chambers


def _first_occurrence_order(code_row: np.ndarray, colors: list[int]) -> list[int]:
    return sorted(colors, key=lambda color: int(np.flatnonzero(code_row == color)[0]))


def _coord_set(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _is_nonrectangular(cells: set[tuple[int, int]]) -> bool:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1) > len(cells)


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    chambers: list[set[tuple[int, int]]],
    area_to_color: dict[int, int],
    background: int,
    wall: int,
) -> bool:
    """Check puzzle semantics, uniqueness, anti-shortcuts, and the solver oracle."""
    if inp.ndim != 2 or max(inp.shape) > 30 or inp.shape != expected.shape:
        return False
    if _background_color(inp) != background:
        return False
    body, code_row = inp[:-1], inp[-1]
    if set(map(int, np.unique(body))) != {background, wall}:
        return False
    if np.any(code_row == background) or np.any(code_row == wall):
        return False
    if not (
        np.all(body[0, :] == background)
        and np.all(body[-1, :] == background)
        and np.all(body[:, 0] == background)
        and np.all(body[:, -1] == background)
    ):
        return False

    components = _components(body == background)
    exterior = [comp for comp in components if _touches_boundary(comp, body.shape)]
    enclosed = [comp for comp in components if not _touches_boundary(comp, body.shape)]
    if len(exterior) != 1 or len(enclosed) != len(chambers):
        return False
    if {_coord_set(comp) for comp in enclosed} != {frozenset(cells) for cells in chambers}:
        return False

    areas = [len(cells) for cells in chambers]
    if len(set(areas)) != len(areas) or set(areas) != set(area_to_color):
        return False
    if not any(_is_nonrectangular(cells) for cells in chambers):
        return False

    frequencies = Counter(map(int, code_row))
    for area, color in area_to_color.items():
        if frequencies[color] != area:
            return False
        if [c for c, count in frequencies.items() if count == area] != [color]:
            return False

    matching_colors = list(area_to_color.values())
    left_to_right = sorted(
        chambers,
        key=lambda cells: sum(c for _r, c in cells) / len(cells),
    )
    centroids = [sum(c for _r, c in cells) / len(cells) for cells in left_to_right]
    if len(set(centroids)) != len(centroids):
        return False
    left_colors = [area_to_color[len(cells)] for cells in left_to_right]
    if _first_occurrence_order(code_row, matching_colors) == left_colors:
        return False

    independently_expected = inp.copy()
    independently_expected[-1, :] = background
    for cells in chambers:
        color = area_to_color[len(cells)]
        for r, c in cells:
            independently_expected[r, c] = color
    if not np.array_equal(independently_expected, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        solved = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    return bool(np.array_equal(solved, expected))


def _generate_attempt() -> np.ndarray | None:
    plan = _sample_frequency_plan()
    if plan is None:
        return None
    areas, distractor_frequencies = plan
    width = sum(areas) + sum(distractor_frequencies)
    body_height = random.randint(15, 27)

    palette = random.sample(_PALETTE, 2 + len(areas) + len(distractor_frequencies))
    background, wall = palette[:2]
    matching_colors = palette[2 : 2 + len(areas)]
    distractor_colors = palette[2 + len(areas) :]
    area_to_color = dict(zip(areas, matching_colors))

    placed = _place_chambers(body_height, width, areas)
    if placed is None:
        return None
    wall_mask, chambers = placed

    code_values: list[int] = []
    for area, color in area_to_color.items():
        code_values.extend([color] * area)
    for frequency, color in zip(distractor_frequencies, distractor_colors):
        code_values.extend([color] * frequency)

    left_to_right = sorted(
        chambers,
        key=lambda cells: sum(c for _r, c in cells) / len(cells),
    )
    left_colors = [area_to_color[len(cells)] for cells in left_to_right]
    code_row = np.asarray(code_values, dtype=int)
    for _ in range(80):
        np.random.shuffle(code_row)
        transitions = int(np.count_nonzero(code_row[1:] != code_row[:-1]))
        if (
            _first_occurrence_order(code_row, matching_colors) != left_colors
            and transitions >= len(set(code_values))
        ):
            break
    else:
        return None

    inp = np.full((body_height + 1, width), background, dtype=int)
    inp[:-1, :][wall_mask] = wall
    inp[-1, :] = code_row
    expected = inp.copy()
    expected[-1, :] = background
    for cells in chambers:
        color = area_to_color[len(cells)]
        for r, c in cells:
            expected[r, c] = color

    if not _validate_generated_case(
        inp, expected, chambers, area_to_color, background, wall
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-frequency flood-fill puzzle")
