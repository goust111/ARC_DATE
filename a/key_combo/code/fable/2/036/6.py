from __future__ import annotations

import itertools
import random

import numpy as np


DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE: tuple[int, ...] = tuple(range(1, 10))
BASE_GROUPS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (2, 0), (4, 0), (2, 4)),
    ((1, 1), (1, 3), (3, 1), (3, 3), (2, 2)),
)
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4_values(grid: np.ndarray, r: int, c: int) -> list[int]:
    return [int(grid[r - 1, c]), int(grid[r + 1, c]), int(grid[r, c - 1]), int(grid[r, c + 1])]


def _transform_index(rc: tuple[int, int], code: int) -> tuple[int, int]:
    r, c = rc
    if code == 0:
        return r, c
    if code == 1:
        return c, 4 - r
    if code == 2:
        return 4 - r, 4 - c
    if code == 3:
        return 4 - c, r
    if code == 4:
        return r, 4 - c
    if code == 5:
        return 4 - r, c
    if code == 6:
        return c, r
    return 4 - c, 4 - r


def _first_nonblack(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != 0:
            return r, c
        r += dr
        c += dc
    return None


def _find_examples(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    examples: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center == 0:
                continue
            neighbors = _neighbors4_values(grid, r, c)
            if any(value == 0 for value in neighbors):
                continue
            if len(set(neighbors)) != 4:
                continue
            legend = {idx: neighbors[idx] for idx in range(4)}
            examples.append({"center": (r, c), "color": center, "legend": legend})
    return examples


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    examples = _find_examples(g)
    assert len(examples) == 2, "expected exactly two example markers"
    marker_colors = [int(example["color"]) for example in examples]
    assert marker_colors[0] != marker_colors[1], "example marker colors must differ"

    legends = {int(example["color"]): dict(example["legend"]) for example in examples}
    example_centers = {tuple(map(int, example["center"])) for example in examples}

    bare_by_color: dict[int, set[tuple[int, int]]] = {color: set() for color in marker_colors}
    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color not in bare_by_color or (r, c) in example_centers:
                continue
            if all(value == 0 for value in _neighbors4_values(g, r, c)):
                bare_by_color[color].add((r, c))

    marker_positions = {
        color: {center for center, example_color in ((tuple(map(int, e["center"])), int(e["color"])) for e in examples) if example_color == color}
        | bare_by_color[color]
        for color in bare_by_color
    }

    assert all(bare_by_color[color] for color in bare_by_color), "each marker color needs at least one bare marker"
    return {
        "grid": g,
        "examples": examples,
        "legends": legends,
        "marker_colors": tuple(marker_colors),
        "bare_by_color": bare_by_color,
        "marker_positions": marker_positions,
        "example_centers": example_centers,
    }


def _lit_directions(scene: dict, color: int, pos: tuple[int, int]) -> tuple[int, ...]:
    grid = scene["grid"]
    same_color_markers = scene["marker_positions"][color]
    lit: list[int] = []
    for idx, direction in enumerate(DIRS):
        first = _first_nonblack(grid, pos, direction)
        if first is not None and first in same_color_markers:
            lit.append(idx)
    return tuple(lit)


def _solve_from_scene(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    for color in sorted(scene["bare_by_color"]):
        legend = scene["legends"][color]
        for r, c in sorted(scene["bare_by_color"][color]):
            for dir_idx in _lit_directions(scene, color, (r, c)):
                dr, dc = DIRS[dir_idx]
                out[r + dr, c + dc] = int(legend[dir_idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each example cross as a per-color direction legend, then light only the visible arms around each bare marker.

    Input structure:
    - A black background contains marker pixels of exactly two marker colors.
    - For each marker color, exactly one example marker has four orthogonal neighbors already colored with four distinct legend colors.
    - Other markers of those two colors are bare single pixels whose four orthogonal neighbors are black.

    Rule:
    - Read the up, down, left, and right legend colors from each example marker separately.
    - For every bare marker, look outward one direction at a time; if the first non-black pixel on that ray is another marker of the same color, color the adjacent cell in that direction with that color's legend entry.
    - Otherwise leave that arm dark, and preserve all original example crosses and marker pixels unchanged.

    Key insight:
    - The example halos are data rather than decoration, and bare markers do not copy a full cross: each arm is gated by first-hit line-of-sight to a same-color peer.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    return _solve_from_scene(scene)


def _sample_spaced_values(low: int, high: int, count: int, min_gap: int) -> list[int]:
    span = high - low + 1 - (count - 1) * (min_gap - 1)
    assert span >= count
    base = sorted(random.sample(range(span), count))
    return [low + base[idx] + idx * (min_gap - 1) for idx in range(count)]


def _sample_legends(marker_colors: tuple[int, int]) -> dict[int, tuple[int, int, int, int]]:
    halo_palette = random.sample([color for color in PALETTE if color not in marker_colors], 4)
    first = tuple(random.sample(halo_palette, 4))
    derangements = [
        tuple(perm)
        for perm in itertools.permutations(halo_palette)
        if any(perm[idx] != first[idx] for idx in range(4)) and all(perm[idx] != first[idx] for idx in range(4))
    ]
    second = random.choice(derangements)
    return {marker_colors[0]: first, marker_colors[1]: second}


def _corner_centers(height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if random.random() < 0.5:
        return (
            (random.randint(2, 3), random.randint(2, 3)),
            (random.randint(height - 4, height - 3), random.randint(width - 4, width - 3)),
        )
    return (
        (random.randint(2, 3), random.randint(width - 4, width - 3)),
        (random.randint(height - 4, height - 3), random.randint(2, 3)),
    )


def _stamp_example(grid: np.ndarray, center: tuple[int, int], marker_color: int, legend: tuple[int, int, int, int]) -> None:
    r, c = center
    grid[r, c] = int(marker_color)
    for dir_idx, (dr, dc) in enumerate(DIRS):
        grid[r + dr, c + dc] = int(legend[dir_idx])


def _any_same_color_further(markers: set[tuple[int, int]], start: tuple[int, int], direction: tuple[int, int]) -> bool:
    r, c = start
    dr, dc = direction
    for rr, cc in markers:
        if dr != 0 and cc == c and (rr - r) * dr > 0:
            return True
        if dc != 0 and rr == r and (cc - c) * dc > 0:
            return True
    return False


def _solve_with_single_legend(scene: dict, chosen_color: int) -> np.ndarray:
    out = scene["grid"].copy()
    legend = scene["legends"][chosen_color]
    for color in sorted(scene["bare_by_color"]):
        for r, c in sorted(scene["bare_by_color"][color]):
            for dir_idx in _lit_directions(scene, color, (r, c)):
                dr, dc = DIRS[dir_idx]
                out[r + dr, c + dc] = int(legend[dir_idx])
    return out


def _solve_ignore_blockers(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    for color in sorted(scene["bare_by_color"]):
        legend = scene["legends"][color]
        markers = scene["marker_positions"][color]
        for r, c in sorted(scene["bare_by_color"][color]):
            for dir_idx, direction in enumerate(DIRS):
                if _any_same_color_further(markers, (r, c), direction):
                    dr, dc = direction
                    out[r + dr, c + dc] = int(legend[dir_idx])
    return out


def _solve_full_cross(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    for color in sorted(scene["bare_by_color"]):
        legend = scene["legends"][color]
        for r, c in sorted(scene["bare_by_color"][color]):
            for dir_idx, (dr, dc) in enumerate(DIRS):
                out[r + dr, c + dc] = int(legend[dir_idx])
    return out


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        scene = _analyze_scene(grid)
    except AssertionError:
        return False

    out = _solve_from_scene(scene)
    if np.array_equal(out, scene["grid"]):
        return False

    lit_masks: list[int] = []
    lit_cells: set[tuple[int, int]] = set()
    has_blocked_same_color = False
    directions_seen = set()
    for color in sorted(scene["bare_by_color"]):
        markers = scene["marker_positions"][color]
        for pos in sorted(scene["bare_by_color"][color]):
            lit = _lit_directions(scene, color, pos)
            mask = 0
            r, c = pos
            for dir_idx in lit:
                mask |= 1 << dir_idx
                directions_seen.add(dir_idx)
                dr, dc = DIRS[dir_idx]
                target = (r + dr, c + dc)
                if target in lit_cells:
                    return False
                lit_cells.add(target)
            lit_masks.append(mask)
            for dir_idx, direction in enumerate(DIRS):
                if not _any_same_color_further(markers, pos, direction):
                    continue
                first = _first_nonblack(scene["grid"], pos, direction)
                if first is not None and first not in markers:
                    has_blocked_same_color = True

    if not lit_masks or 0 not in lit_masks or not any(bin(mask).count("1") >= 2 for mask in lit_masks):
        return False
    if len(set(lit_masks)) < 3 or directions_seen != {0, 1, 2, 3}:
        return False
    if not has_blocked_same_color:
        return False

    marker_colors = scene["marker_colors"]
    if np.array_equal(out, _solve_with_single_legend(scene, marker_colors[0])):
        return False
    if np.array_equal(out, _solve_with_single_legend(scene, marker_colors[1])):
        return False
    if np.array_equal(out, _solve_ignore_blockers(scene)):
        return False
    if np.array_equal(out, _solve_full_cross(scene)):
        return False

    try:
        actual = main(scene["grid"])
    except AssertionError:
        return False
    return np.array_equal(actual, out)


def _build_candidate() -> np.ndarray:
    height = random.randint(19, 26)
    width = random.randint(19, 26)
    grid = np.zeros((height, width), dtype=int)

    marker_colors = tuple(random.sample(PALETTE, 2))
    legends = _sample_legends(marker_colors)

    row_values = _sample_spaced_values(5, height - 6, 5, 2)
    col_values = _sample_spaced_values(5, width - 6, 5, 2)
    transform_code = random.randrange(8)
    group_colors = marker_colors if random.random() < 0.5 else marker_colors[::-1]

    used_cells: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for group_idx, group in enumerate(BASE_GROUPS):
        color = int(group_colors[group_idx])
        for logical_pos in group:
            rr_idx, cc_idx = _transform_index(logical_pos, transform_code)
            cell = (row_values[rr_idx], col_values[cc_idx])
            assert cell not in used_cells
            grid[cell] = color
            used_cells.add(cell)
            used_rows.add(cell[0])
            used_cols.add(cell[1])

    example_centers = _corner_centers(height, width)
    random_order = list(marker_colors)
    random.shuffle(random_order)
    for center, color in zip(example_centers, random_order):
        _stamp_example(grid, center, int(color), legends[int(color)])

    extra_rows = [r for r in range(5, height - 5) if r not in used_rows]
    extra_cols = [c for c in range(5, width - 5) if c not in used_cols]
    extra_count = random.randint(0, min(3, len(extra_rows), len(extra_cols)))
    random.shuffle(extra_rows)
    random.shuffle(extra_cols)
    for idx in range(extra_count):
        grid[extra_rows[idx], extra_cols[idx]] = int(random.choice(marker_colors))

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if _validate_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid visibility-legend input")
