from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


Grid = np.ndarray
Coord = tuple[int, int]

BACKGROUND = 0
GEN_RETRY_LIMIT = 400
COLOR_POOL = tuple(range(1, 10))


@dataclass(frozen=True)
class Container:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def opening_columns(self) -> list[int]:
        return list(range(self.c0 + 1, self.c1))


def _get_bg(grid: Grid) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: Iterable[Coord]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _components_for_color(grid: Grid, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r, c in np.argwhere(grid == color):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[Coord] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _is_u_container(coords: list[Coord], shape: tuple[int, int]) -> bool:
    h, _w = shape
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if r1 != h - 1 or height < 2 or width < 3:
        return False
    expected = {(r1, c) for c in range(c0, c1 + 1)}
    expected |= {(r, c0) for r in range(r0, r1 + 1)}
    expected |= {(r, c1) for r in range(r0, r1 + 1)}
    return set(coords) == expected


def _parse_scene(grid: Grid) -> tuple[int, list[Container], set[Coord], list[Coord]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    containers: list[Container] = []
    remaining: list[tuple[int, list[Coord]]] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        for comp in _components_for_color(g, color):
            if _is_u_container(comp, g.shape):
                r0, c0, r1, c1 = _bbox(comp)
                containers.append(Container(int(color), r0, c0, r1, c1))
            else:
                remaining.append((int(color), comp))

    containers.sort(key=lambda item: item.c0)
    assert len(containers) >= 2, "expected multiple bottom U-containers"
    assert len({c.color for c in containers}) == len(containers), "container colors must be unique"
    for left, right in zip(containers, containers[1:]):
        assert left.c1 < right.c0, "containers must be separated"

    obstacles: set[Coord] = set()
    fallers: list[Coord] = []
    for _color, comp in remaining:
        r0, c0, r1, c1 = _bbox(comp)
        area = len(comp)
        if r0 == r1 and area == c1 - c0 + 1 and area >= 2:
            obstacles.update(comp)
            continue
        assert area == 1, "non-container objects must be single fallers or horizontal obstacles"
        fallers.append(comp[0])

    assert fallers, "expected at least one falling pixel"
    return bg, containers, obstacles, fallers


def _counts_from_scene(containers: list[Container], obstacles: set[Coord], fallers: list[Coord]) -> list[int]:
    obstacle_rows_by_col: dict[int, list[int]] = {}
    for r, c in obstacles:
        obstacle_rows_by_col.setdefault(c, []).append(r)
    for rows in obstacle_rows_by_col.values():
        rows.sort()

    counts = [0] * len(containers)
    for r, c in fallers:
        blocked = any(obs_r > r for obs_r in obstacle_rows_by_col.get(c, ()))
        hits = [idx for idx, box in enumerate(containers) if box.c0 < c < box.c1]
        if blocked:
            assert len(hits) <= 1, "blocked faller column should not cross multiple openings"
            continue
        assert len(hits) == 1, "every surviving faller must belong to exactly one container opening"
        counts[hits[0]] += 1

    assert max(counts) > 0, "expected at least one surviving faller"
    return counts


def _histogram(bg: int, containers: list[Container], counts: list[int]) -> Grid:
    height = max(counts)
    out = np.full((height, len(containers)), bg, dtype=int)
    for c, (box, count) in enumerate(zip(containers, counts)):
        if count:
            out[height - count :, c] = box.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the falling pixels that reach each bottom container and rewrite the counts as a color-coded histogram.

    Input structure:
    - Several differently colored open-upward U-shaped containers sit along the bottom edge.
    - Above them are isolated falling pixels and one or more horizontal obstacle segments.
    - Containers are distinguished by color and by their left-to-right order.

    Rule:
    - Each falling pixel moves straight downward within its own column.
    - If some horizontal obstacle lies below that pixel in the same column, the pixel is blocked and discarded.
    - Otherwise the pixel falls into the unique container whose opening contains that column.
    - Output a grid whose width is the number of containers and whose height is the largest surviving count; column i is filled bottom-up with the i-th container's color for exactly its count cells.

    Key insight:
    - The task is a pipeline: detect which falling pixels survive the vertical obstacle filter, bucket survivors by container-opening column membership, then compress those counts into a left-to-right histogram keyed by container colors.
    """
    bg, containers, obstacles, fallers = _parse_scene(np.asarray(grid, dtype=int))
    counts = _counts_from_scene(containers, obstacles, fallers)
    return _histogram(bg, containers, counts)


def _draw_container(grid: Grid, box: Container) -> None:
    grid[box.r1, box.c0 : box.c1 + 1] = box.color
    grid[box.r0 : box.r1 + 1, box.c0] = box.color
    grid[box.r0 : box.r1 + 1, box.c1] = box.color


def _sample_layout(rng: np.random.Generator) -> tuple[int, int, list[Container]]:
    n = int(rng.integers(2, 6))
    widths = [int(rng.integers(3, 7)) for _ in range(n)]
    heights = [int(rng.integers(3, 7)) for _ in range(n)]
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    gaps = [int(rng.integers(1, 4)) for _ in range(n - 1)]
    width = left_margin + right_margin + sum(widths) + sum(gaps)
    if width > 30:
        raise ValueError("layout too wide")

    top_band_min = 7
    top_band_max = min(13, 30 - max(heights))
    if top_band_max < top_band_min:
        raise ValueError("layout too tall")
    top_band = int(rng.integers(top_band_min, top_band_max + 1))
    height = top_band + max(heights)

    palette = rng.permutation(COLOR_POOL)[:n]
    containers: list[Container] = []
    c0 = left_margin
    for idx, (w, h, color) in enumerate(zip(widths, heights, palette)):
        r0 = height - h
        box = Container(int(color), r0, c0, height - 1, c0 + w - 1)
        containers.append(box)
        c0 += w
        if idx < n - 1:
            c0 += gaps[idx]
    return height, width, containers


def _interval_from_focus(rng: np.random.Generator, column: int, width: int) -> tuple[int, int]:
    start = column - int(rng.integers(0, 2))
    end = column + int(rng.integers(1, 3))
    start = max(0, start)
    end = min(width - 1, end)
    if end - start + 1 < 2:
        if end < width - 1:
            end += 1
        else:
            start -= 1
    return start, end


def _sample_obstacles(
    rng: np.random.Generator,
    shape: tuple[int, int],
    containers: list[Container],
) -> tuple[set[Coord], int, int]:
    h, w = shape
    min_r0 = min(box.r0 for box in containers)
    focus_candidates = [idx for idx, box in enumerate(containers) if len(box.opening_columns) >= 1]
    focus_idx = int(rng.choice(focus_candidates))
    focus_box = containers[focus_idx]
    focus_col = int(rng.choice(focus_box.opening_columns))
    focus_row = int(rng.integers(2, focus_box.r0 - 2))

    segments: list[tuple[int, int, int]] = []
    start, end = _interval_from_focus(rng, focus_col, w)
    segments.append((focus_row, start, end))

    extra_count = int(rng.integers(0, len(containers) + 1))
    candidate_rows = list(range(1, min_r0 - 1))
    for _ in range(extra_count):
        row = int(rng.choice(candidate_rows))
        if rng.random() < 0.65:
            box = containers[int(rng.integers(0, len(containers)))]
            col = int(rng.choice(box.opening_columns))
            start, end = _interval_from_focus(rng, col, w)
        else:
            seg_len = int(rng.integers(2, min(8, w) + 1))
            start = int(rng.integers(0, w - seg_len + 1))
            end = start + seg_len - 1
        if start <= focus_col <= end and row > focus_box.r0 - 3:
            continue
        segments.append((row, start, end))

    obstacles: set[Coord] = set()
    for row, start, end in segments:
        for col in range(start, end + 1):
            obstacles.add((row, col))

    covered_openings = sum(
        any((r, c) in obstacles for r in range(h))
        for box in containers
        for c in box.opening_columns
    )
    assert covered_openings >= 1
    assert 0 <= focus_row < h and 0 <= focus_col < w
    return obstacles, focus_col, focus_row


def _rows_for_column(
    container: Container,
    column: int,
    obstacle_rows_by_col: dict[int, list[int]],
) -> tuple[list[int], list[int]]:
    blocked_rows = []
    safe_rows = []
    obs_rows = obstacle_rows_by_col.get(column, [])
    highest = max(obs_rows) if obs_rows else None
    for row in range(container.r0):
        if row in obs_rows:
            continue
        if highest is None or row > highest:
            safe_rows.append(row)
        else:
            blocked_rows.append(row)
    return blocked_rows, safe_rows


def _place_fallers(
    rng: np.random.Generator,
    containers: list[Container],
    obstacles: set[Coord],
    focus_col: int,
) -> tuple[list[Coord], list[int], list[int], bool]:
    obstacle_rows_by_col: dict[int, list[int]] = {}
    for r, c in obstacles:
        obstacle_rows_by_col.setdefault(c, []).append(r)
    for rows in obstacle_rows_by_col.values():
        rows.sort()

    focus_idx = next(idx for idx, box in enumerate(containers) if focus_col in box.opening_columns)
    focus_box = containers[focus_idx]
    focus_blocked_rows, focus_safe_rows = _rows_for_column(focus_box, focus_col, obstacle_rows_by_col)
    assert focus_blocked_rows and len(focus_safe_rows) >= 2

    fallers: set[Coord] = set()
    actual_counts = [0] * len(containers)
    raw_counts = [0] * len(containers)
    survivor_in_covered_column = False

    blocked_row = int(rng.choice(focus_blocked_rows))
    fallers.add((blocked_row, focus_col))
    raw_counts[focus_idx] += 1

    safe_choices = rng.choice(focus_safe_rows, size=2, replace=False)
    for row in safe_choices:
        fallers.add((int(row), focus_col))
        raw_counts[focus_idx] += 1
        actual_counts[focus_idx] += 1
        survivor_in_covered_column = True

    for idx, box in enumerate(containers):
        column_data = {
            c: _rows_for_column(box, c, obstacle_rows_by_col) for c in box.opening_columns
        }
        safe_columns = [c for c, (_blocked, safe) in column_data.items() if safe]
        blocked_columns = [c for c, (blocked, _safe) in column_data.items() if blocked]
        covered_safe_columns = [
            c for c in safe_columns if obstacle_rows_by_col.get(c)
        ]

        extra_survivors = int(rng.integers(0, 4))
        extra_blocked = int(rng.integers(0, 3))
        if idx == focus_idx:
            extra_survivors = int(rng.integers(0, 3))
            extra_blocked = int(rng.integers(0, 2))

        while extra_survivors > 0 and safe_columns:
            if idx != focus_idx and covered_safe_columns and rng.random() < 0.45:
                col = int(rng.choice(covered_safe_columns))
            else:
                col = int(rng.choice(safe_columns))
            _blocked_rows, safe_rows = column_data[col]
            free_rows = [row for row in safe_rows if (row, col) not in fallers]
            if not free_rows:
                safe_columns = [c for c in safe_columns if any((row, c) not in fallers for row in column_data[c][1])]
                covered_safe_columns = [c for c in covered_safe_columns if any((row, c) not in fallers for row in column_data[c][1])]
                continue
            row = int(rng.choice(free_rows))
            fallers.add((row, col))
            raw_counts[idx] += 1
            actual_counts[idx] += 1
            survivor_in_covered_column |= bool(obstacle_rows_by_col.get(col))
            extra_survivors -= 1

        while extra_blocked > 0 and blocked_columns:
            col = int(rng.choice(blocked_columns))
            blocked_rows, _safe_rows = column_data[col]
            free_rows = [row for row in blocked_rows if (row, col) not in fallers]
            if not free_rows:
                blocked_columns = [c for c in blocked_columns if any((row, c) not in fallers for row in column_data[c][0])]
                continue
            row = int(rng.choice(free_rows))
            fallers.add((row, col))
            raw_counts[idx] += 1
            extra_blocked -= 1

    positive_indices = [idx for idx, count in enumerate(actual_counts) if count > 0]
    zero_indices = [idx for idx, count in enumerate(actual_counts) if count == 0]
    if len(positive_indices) < 2:
        for idx in zero_indices:
            box = containers[idx]
            for col in box.opening_columns:
                _blocked_rows, safe_rows = _rows_for_column(box, col, obstacle_rows_by_col)
                free_rows = [row for row in safe_rows if (row, col) not in fallers]
                if free_rows:
                    row = int(rng.choice(free_rows))
                    fallers.add((row, col))
                    raw_counts[idx] += 1
                    actual_counts[idx] += 1
                    survivor_in_covered_column |= bool(obstacle_rows_by_col.get(col))
                    positive_indices.append(idx)
                    break
            if len(positive_indices) >= 2:
                break

    return sorted(fallers), actual_counts, raw_counts, survivor_in_covered_column


def _render_scene(
    shape: tuple[int, int],
    containers: list[Container],
    obstacles: set[Coord],
    fallers: list[Coord],
    obstacle_color: int,
    faller_color: int,
) -> Grid:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for box in containers:
        _draw_container(grid, box)
    for r, c in obstacles:
        grid[r, c] = obstacle_color
    for r, c in fallers:
        grid[r, c] = faller_color
    return grid


def _stacked_column_exists(
    containers: list[Container],
    obstacles: set[Coord],
    fallers: list[Coord],
) -> bool:
    obstacle_rows_by_col: dict[int, list[int]] = {}
    for r, c in obstacles:
        obstacle_rows_by_col.setdefault(c, []).append(r)
    survivors_by_container_col: dict[tuple[int, int], int] = {}
    for r, c in fallers:
        if any(obs_r > r for obs_r in obstacle_rows_by_col.get(c, ())):
            continue
        for idx, box in enumerate(containers):
            if box.c0 < c < box.c1:
                survivors_by_container_col[(idx, c)] = survivors_by_container_col.get((idx, c), 0) + 1
                break
    return any(count >= 2 for count in survivors_by_container_col.values())


def _scene_oracle(
    containers: list[Container],
    obstacles: set[Coord],
    fallers: list[Coord],
) -> tuple[list[int], Grid]:
    counts = _counts_from_scene(containers, obstacles, fallers)
    return counts, _histogram(BACKGROUND, containers, counts)


def _valid_scene(
    grid: Grid,
    containers: list[Container],
    obstacles: set[Coord],
    fallers: list[Coord],
    true_counts: list[int],
    raw_counts: list[int],
    survivor_in_covered_column: bool,
) -> bool:
    try:
        solved = main(grid)
    except Exception:
        return False

    expected = _histogram(BACKGROUND, containers, true_counts)
    if not np.array_equal(solved, expected):
        return False
    if true_counts == raw_counts:
        return False
    if len({count for count in true_counts}) == 1:
        return False
    if sum(count > 0 for count in true_counts) < 2:
        return False
    if not any(count == 0 for count in true_counts):
        return False
    if not survivor_in_covered_column:
        return False
    if not _stacked_column_exists(containers, obstacles, fallers):
        return False
    if max(true_counts) > 6:
        return False
    return True


def _generate_once() -> Grid | None:
    rng = np.random.default_rng()
    try:
        height, width, containers = _sample_layout(rng)
    except ValueError:
        return None

    obstacle_color, faller_color = rng.permutation(
        [c for c in COLOR_POOL if c not in {box.color for box in containers}]
    )[:2]
    obstacles, focus_col, _focus_row = _sample_obstacles(rng, (height, width), containers)
    fallers, true_counts, raw_counts, survivor_in_covered_column = _place_fallers(
        rng, containers, obstacles, focus_col
    )
    grid = _render_scene(
        (height, width),
        containers,
        obstacles,
        fallers,
        int(obstacle_color),
        int(faller_color),
    )
    if not _valid_scene(
        grid,
        containers,
        obstacles,
        fallers,
        true_counts,
        raw_counts,
        survivor_in_covered_column,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid histogram-from-falling-pixels scene")
