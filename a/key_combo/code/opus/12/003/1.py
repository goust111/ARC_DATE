from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
YELLOW = 4
WALL = 5
SEPARATOR = 6
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    """Return the most frequent color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    """Yield 4-neighbors using ARC's (row, col) convention."""
    for dr, dc in N4:
        yield r + dr, c + dc


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _separator_column(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    assert h > 0 and w >= 3 and w % 2 == 1, "expected two equal panels plus one divider column"
    candidates: list[int] = []
    for col in range(w):
        color = int(grid[0, col])
        if color == bg:
            continue
        if not np.all(grid[:, col] == color):
            continue
        if col == 0 or col == w - 1 or col != w - col - 1:
            continue
        without = np.concatenate((grid[:, :col], grid[:, col + 1 :]), axis=1)
        if np.any(without == color):
            continue
        candidates.append(col)
    assert len(candidates) == 1, "expected exactly one solid divider column"
    return candidates[0]


def _comparison_map(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_fg = left != BACKGROUND
    right_fg = right != BACKGROUND
    overlap = left_fg & right_fg
    assert np.all(left[overlap] == right[overlap]), "shared panel pixels must agree"
    assert np.all(left[left_fg] == YELLOW), "left panel foreground must be yellow"
    assert np.all(right[right_fg] == YELLOW), "right panel foreground must be yellow"

    out = np.full(left.shape, BACKGROUND, dtype=int)
    out[overlap] = WALL
    out[left_fg & ~right_fg] = RED
    out[right_fg & ~left_fg] = BLUE
    return out


def _flood_fill_from_comparison(compared: np.ndarray) -> np.ndarray:
    out = np.asarray(compared, dtype=int).copy()
    h, w = out.shape
    seen = np.zeros((h, w), dtype=bool)

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or out[sr, sc] == WALL:
                continue

            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            seeds: set[int] = set()

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                if out[r, c] in (RED, BLUE):
                    seeds.add(int(out[r, c]))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and out[nr, nc] != WALL:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            assert len(seeds) <= 1, "each chamber may contain at most one seed color"
            if seeds:
                fill = next(iter(seeds))
                for r, c in cells:
                    out[r, c] = fill

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compare the two yellow panels to turn shared pixels into walls and panel differences into flood-fill seeds.

    Input structure:
    - A black background contains two equal-size side panels separated by one solid vertical divider column.
    - The divider is a unique non-background color, and both panels use the same yellow foreground color.
    - Most yellow pixels coincide between the two panels, with some left-only and right-only differences.

    Rule:
    - Remove the divider and align the two panels cell by cell.
    - When both panels have yellow at a position, write gray there to create an impassable wall; when both are background, keep black.
    - When only the left panel has yellow, write red; when only the right panel has yellow, write blue.
    - Treat gray pixels as 4-connected barriers, then for every non-wall region fill the whole region with its seed color if that region contains red seeds or blue seeds. Regions without seeds stay black.

    Key insight:
    - The panel comparison is only an intermediate structure: shared pixels become walls, while mismatches become the seeds that determine each chamber's flood-filled color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    sep_col = _separator_column(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape, "divider must split two equal-size panels"

    compared = _comparison_map(left, right)
    return _flood_fill_from_comparison(compared)


def _sample_line_positions(length: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    choices = list(range(2, length - 2))
    if len(choices) < count:
        return None
    for _ in range(100):
        picked = sorted(random.sample(choices, count))
        if all(b - a >= 3 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in map(tuple, coords))


def _choose_seed_cells(coords: np.ndarray, count: int) -> list[tuple[int, int]]:
    cells = [tuple(map(int, rc)) for rc in coords]
    random.shuffle(cells)
    for _ in range(80):
        chosen: list[tuple[int, int]] = []
        pool = cells[:]
        random.shuffle(pool)
        for cell in pool:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return cells[:count]


def _derive_panels(compared: np.ndarray) -> np.ndarray:
    left = np.zeros_like(compared)
    right = np.zeros_like(compared)
    left[(compared == WALL) | (compared == RED)] = YELLOW
    right[(compared == WALL) | (compared == BLUE)] = YELLOW
    divider = np.full((compared.shape[0], 1), SEPARATOR, dtype=int)
    return np.concatenate((left, divider, right), axis=1)


def _independent_expected_from_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = g.shape[1] // 2
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    compared = np.zeros(left.shape, dtype=int)
    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            lv = int(left[r, c] != BACKGROUND)
            rv = int(right[r, c] != BACKGROUND)
            if lv and rv:
                compared[r, c] = WALL
            elif lv:
                compared[r, c] = RED
            elif rv:
                compared[r, c] = BLUE
    expected = compared.copy()
    for comp in components_mask(compared != WALL):
        colors = {int(compared[r, c]) for r, c in comp if compared[r, c] in (RED, BLUE)}
        assert len(colors) <= 1
        if colors:
            fill = next(iter(colors))
            expected[comp[:, 0], comp[:, 1]] = fill
    return expected


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(9, 13)
    w = random.randint(9, 13)
    n_h = random.randint(1, 2)
    n_v = random.randint(1, 2)

    wall_rows = _sample_line_positions(h, n_h)
    wall_cols = _sample_line_positions(w, n_v)
    if wall_rows is None or wall_cols is None:
        return None

    wall_mask = np.zeros((h, w), dtype=bool)
    wall_mask[wall_rows, :] = True
    wall_mask[:, wall_cols] = True
    regions = components_mask(~wall_mask)
    if len(regions) < 4:
        return None

    border_regions = [i for i, comp in enumerate(regions) if _touches_border(comp, h, w)]
    interior_regions = [i for i, comp in enumerate(regions) if i not in border_regions]
    if not border_regions:
        return None

    region_ids = list(range(len(regions)))
    random.shuffle(region_ids)
    max_colored = min(4, len(regions) - 1)
    colored_count = random.randint(2, max_colored)

    colored: list[int] = []
    red_anchor = random.choice(border_regions)
    colored.append(red_anchor)
    if interior_regions:
        blue_anchor = random.choice(interior_regions)
    else:
        blue_anchor = random.choice([idx for idx in region_ids if idx != red_anchor])
    colored.append(blue_anchor)
    remaining = [idx for idx in region_ids if idx not in colored]
    random.shuffle(remaining)
    colored.extend(remaining[: colored_count - 2])

    compared = np.full((h, w), BACKGROUND, dtype=int)
    compared[wall_mask] = WALL
    expected = np.full((h, w), BACKGROUND, dtype=int)
    expected[wall_mask] = WALL

    color_by_region: dict[int, int] = {}
    extra_colors = [RED, BLUE] * 4
    random.shuffle(extra_colors)
    color_by_region[red_anchor] = RED
    color_by_region[blue_anchor] = BLUE
    for idx in colored[2:]:
        color_by_region[idx] = extra_colors.pop()

    multi_seed_candidates = [
        idx for idx in colored if regions[idx].shape[0] >= 5
    ]
    multi_seed_region = random.choice(multi_seed_candidates) if multi_seed_candidates and random.random() < 0.7 else None

    for idx, comp in enumerate(regions):
        if idx not in color_by_region:
            continue
        fill = color_by_region[idx]
        expected[comp[:, 0], comp[:, 1]] = fill
        max_seed_count = 2 if comp.shape[0] >= 5 else 1
        seed_count = 2 if idx == multi_seed_region else random.randint(1, max_seed_count)
        seeds = _choose_seed_cells(comp, seed_count)
        for r, c in seeds:
            compared[r, c] = fill

    if not np.any(compared == RED) or not np.any(compared == BLUE):
        return None
    if not np.any(expected == BACKGROUND):
        return None
    if not np.any(expected == RED) or not np.any(expected == BLUE):
        return None
    if not np.any(expected == WALL):
        return None

    inp = _derive_panels(compared)
    if inp.shape[1] > 30 or inp.shape[0] > 30:
        return None
    if get_bg(inp) != BACKGROUND:
        return None
    if not np.all(inp[:, w] == SEPARATOR):
        return None

    actual = main(inp)
    independent = _independent_expected_from_input(inp)
    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(actual, independent):
        return None

    open_components = components_mask(actual != WALL)
    seedless_regions = 0
    for comp in open_components:
        colors = {int(actual[r, c]) for r, c in comp if actual[r, c] in (RED, BLUE)}
        if not colors:
            seedless_regions += 1
    if seedless_regions == 0:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _generate_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid panel-diff flood-fill puzzle")
