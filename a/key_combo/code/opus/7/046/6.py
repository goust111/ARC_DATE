from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PALETTE = (1, 2, 3, 4, 6)
BACKGROUND = 0
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _neighbors8(r: int, c: int):
    for dr, dc in NEIGHBORS8:
        yield r + dr, c + dc


def _find_monochrome_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append({"color": color, "coords": np.array(cells, dtype=int)})

    return components


def _majority_colors(grid: np.ndarray) -> set[int]:
    counts = Counter(int(v) for v in grid.flat if int(v) != BACKGROUND)
    assert 4 <= len(counts) <= 5, "expected 4-5 non-background colors"

    ranked = counts.most_common()
    top_count = ranked[0][1]
    top_colors = {color for color, count in ranked if count == top_count}
    if len(top_colors) == 2:
        return top_colors

    assert len(top_colors) == 1, "top frequency rank must be unambiguous or a 2-way tie"
    lower_counts = [count for _, count in ranked if count < top_count]
    assert lower_counts, "expected a second frequency rank"
    second_count = max(lower_counts)
    second_colors = {color for color, count in ranked if count == second_count}
    assert len(second_colors) == 1, "second frequency rank must be unique when the top rank is not tied"
    return top_colors | second_colors


def _fill_outline_rows(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    for r in sorted({int(rr) for rr in coords[:, 0]}):
        row_cols = coords[coords[:, 0] == r, 1]
        assert row_cols.size == 2, "each occupied row must contain exactly two boundary pixels"
        c0 = int(row_cols.min())
        c1 = int(row_cols.max())
        out[r, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the minority-color hollow outlines while keeping the two most frequent outline colors hollow.

    Input structure:
    - The grid has a black background and 4-5 differently colored monochrome hollow outlines.
    - Each outline occupies consecutive rows, and on every occupied row it has exactly two same-color boundary pixels.
    - Outline colors are unique per object, and the two highest-frequency colors tie for the top frequency rank.

    Rule:
    - Count the total number of pixels of each non-background color.
    - Mark the two highest-frequency colors as majority colors and all remaining colors as minority colors.
    - For each outline whose color is a minority color, fill every occupied row from its left boundary pixel to its right boundary pixel, inclusive.
    - Leave the black background and the majority-color outlines unchanged.

    Key insight:
    - The geometry only provides row-wise fill limits; the decision to fill comes from the global color-frequency ranking, not from position, width, or a fixed color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    majority = _majority_colors(g)
    components = _find_monochrome_components(g)
    assert len(components) >= 4, "expected multiple outlines"

    out = g.copy()
    for component in components:
        color = int(component["color"])
        coords = component["coords"]
        if color not in majority:
            _fill_outline_rows(out, coords, color)
    return out


MOVE_OPTIONS = [(-1, 0), (0, 1), (-1, 1), (0, 0)]
MOVE_WEIGHTS = [3, 3, 5, 1]
MAX_GENERATION_ATTEMPTS = 300


def _outline_from_height(height: int) -> dict | None:
    for _ in range(120):
        bridge_row = random.randrange(height)
        left = [0] * height
        right = [1] * height
        expansions = 0

        for rows in (range(bridge_row - 1, -1, -1), range(bridge_row + 1, height)):
            cur_left, cur_right = 0, 1
            for r in rows:
                dl, dr = random.choices(MOVE_OPTIONS, weights=MOVE_WEIGHTS, k=1)[0]
                cur_left += dl
                cur_right += dr
                left[r] = cur_left
                right[r] = cur_right
                expansions += int(dl != 0 or dr != 0)

        widths = [rr - ll + 1 for ll, rr in zip(left, right)]
        if max(widths) < 4:
            continue
        if expansions < max(2, height // 2):
            continue

        shift = -min(left)
        left = [value + shift for value in left]
        right = [value + shift for value in right]
        width = max(right) + 1
        if width > 10:
            continue

        boundary: list[tuple[int, int]] = []
        filled: list[tuple[int, int]] = []
        for r, (c0, c1) in enumerate(zip(left, right)):
            boundary.append((r, c0))
            boundary.append((r, c1))
            for c in range(c0, c1 + 1):
                filled.append((r, c))

        if random.random() < 0.5:
            boundary = [(r, width - 1 - c) for r, c in boundary]
            filled = [(r, width - 1 - c) for r, c in filled]

        return {
            "height": height,
            "width": width,
            "boundary": boundary,
            "filled": filled,
        }
    return None


def _dilate_cells(cells: list[tuple[int, int]], shape: tuple[int, int], margin: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _candidate_positions(shape: dict, blocked: set[tuple[int, int]], grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid_shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - shape["height"] + 1):
        for c0 in range(w - shape["width"] + 1):
            cells = [(r0 + r, c0 + c) for r, c in shape["filled"]]
            if all(cell not in blocked for cell in cells):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _place_shapes(shapes: list[dict], grid_shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda idx: (-len(shapes[idx]["filled"]), -shapes[idx]["height"]))
    placements: list[tuple[int, int] | None] = [None] * len(shapes)

    def dfs(pos: int, blocked: set[tuple[int, int]]) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        shape = shapes[idx]
        for r0, c0 in _candidate_positions(shape, blocked, grid_shape):
            cells = [(r0 + r, c0 + c) for r, c in shape["filled"]]
            placements[idx] = (r0, c0)
            if dfs(pos + 1, blocked | _dilate_cells(cells, grid_shape)):
                return True
            placements[idx] = None
        return False

    if not dfs(0, set()):
        return None
    return [placement for placement in placements if placement is not None]


def _render_input(colors: list[int], shapes: list[dict], placements: list[tuple[int, int]], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for color, shape, (r0, c0) in zip(colors, shapes, placements):
        for r, c in shape["boundary"]:
            grid[r0 + r, c0 + c] = color
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    majority = _majority_colors(grid)
    for component in _find_monochrome_components(grid):
        color = int(component["color"])
        if color not in majority:
            _fill_outline_rows(out, component["coords"], color)
    return out


def _sample_heights(count: int) -> list[int]:
    major_height = random.randint(max(5, count + 1), 8)
    lower_heights = random.sample(list(range(3, major_height)), count - 2)
    return [major_height, major_height] + sorted(lower_heights, reverse=True)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(4, 5)
    grid_h = random.randint(15, 19)
    grid_w = random.randint(15, 19)
    heights = _sample_heights(count)

    shapes: list[dict] = []
    for height in heights:
        shape = _outline_from_height(height)
        if shape is None:
            return None
        shapes.append(shape)

    placements = _place_shapes(shapes, (grid_h, grid_w))
    if placements is None:
        return None

    shape_placements = list(zip(shapes, placements))
    random.shuffle(shape_placements)
    shapes = [shape for shape, _ in shape_placements]
    placements = [placement for _, placement in shape_placements]
    colors = random.sample(list(PALETTE), count)

    grid = _render_input(colors, shapes, placements, (grid_h, grid_w))
    if len(_majority_colors(grid)) != 2:
        return None
    if not np.array_equal(main(grid), _expected_output(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
