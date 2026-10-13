from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _reflection_targets(r: int, c: int, axis_r: int, axis_c: int) -> dict[str, tuple[int, int]]:
    return {
        "v": (int(r), int(2 * axis_c - c)),
        "h": (int(2 * axis_r - r), int(c)),
        "vh": (int(2 * axis_r - r), int(2 * axis_c - c)),
    }


def _foreground_points(grid: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    points: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(grid != bg):
        points[(int(r), int(c))] = int(grid[r, c])
    return points


def _infer_axes(grid: np.ndarray) -> tuple[int, int, int, int, dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)

    by_color: dict[int, list[tuple[int, int]]] = {}
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) in (1, 2), "only the two axis-defining colors may repeat"
        by_color[int(color)] = coords

    pair_colors = [color for color, coords in by_color.items() if len(coords) == 2]
    assert len(pair_colors) == 2, "expected exactly two repeated foreground colors"

    horizontal_candidates: list[tuple[int, int]] = []
    vertical_candidates: list[tuple[int, int]] = []
    for color in pair_colors:
        (r0, c0), (r1, c1) = sorted(by_color[color])
        if r0 == r1 and c0 != c1 and abs(c1 - c0) % 2 == 0:
            horizontal_candidates.append((int(color), int((c0 + c1) // 2)))
        if c0 == c1 and r0 != r1 and abs(r1 - r0) % 2 == 0:
            vertical_candidates.append((int(color), int((r0 + r1) // 2)))

    assert len(horizontal_candidates) == 1, "expected one same-row color pair to define the vertical axis"
    assert len(vertical_candidates) == 1, "expected one same-column color pair to define the horizontal axis"

    x_color, axis_c = horizontal_candidates[0]
    y_color, axis_r = vertical_candidates[0]
    assert x_color != y_color, "the two axis-defining colors must be distinct"

    points = _foreground_points(g, bg)
    for r, c in points:
        assert r != axis_r and c != axis_c, "foreground pixels may not lie on the hidden axes"
    return bg, x_color, y_color, axis_r, axis_c, points


def _apply_rule(
    shape: tuple[int, int],
    points: dict[tuple[int, int], int],
    x_color: int,
    y_color: int,
    axis_r: int,
    axis_c: int,
) -> dict[tuple[int, int], int]:
    assignments = dict(points)
    for (r, c), source_color in points.items():
        for kind, (rr, cc) in _reflection_targets(r, c, axis_r, axis_c).items():
            if not _in_bounds(shape, rr, cc):
                continue
            target_color = {"v": x_color, "h": y_color, "vh": source_color}[kind]
            previous = assignments.get((rr, cc))
            assert previous in (None, target_color), "reflection targets must not conflict in color"
            assignments[(rr, cc)] = int(target_color)
    return assignments


def _render(shape: tuple[int, int], bg: int, assignments: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for (r, c), color in assignments.items():
        out[int(r), int(c)] = int(color)
    return out


def _same_color_completion(
    shape: tuple[int, int],
    bg: int,
    points: dict[tuple[int, int], int],
    axis_r: int,
    axis_c: int,
) -> np.ndarray:
    assignments = dict(points)
    for (r, c), source_color in points.items():
        for rr, cc in _reflection_targets(r, c, axis_r, axis_c).values():
            if not _in_bounds(shape, rr, cc):
                continue
            previous = assignments.get((rr, cc))
            assert previous in (None, source_color)
            assignments[(rr, cc)] = int(source_color)
    return _render(shape, bg, assignments)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a hidden cross of mirror axes from two repeated colors and complete the reflections with axis-coded recoloring.

    Input structure:
    - A dominant background contains only sparse foreground pixels.
    - Exactly one color X appears twice on one row with an even horizontal gap, which fixes the hidden vertical mirror axis.
    - Exactly one other color Y appears twice in one column with an even vertical gap, which fixes the hidden horizontal mirror axis.
    - Every remaining foreground color appears once, and all foreground pixels sit off the hidden axes.

    Rule:
    - Recover the two hidden mirror axes from the two repeated same-color pairs.
    - Keep every input pixel.
    - For every foreground pixel, add its reflection across the vertical axis, across the horizontal axis, and across both axes when the reflected cell stays inside the grid.
    - A vertical-only reflection is recolored to X, a horizontal-only reflection is recolored to Y, and a double reflection keeps the source pixel's original color.

    Key insight:
    - The axes are implicit rather than drawn, and the color depends on which axis or axes were crossed, so naive same-color symmetry completion is the wrong rule.
    """
    bg, x_color, y_color, axis_r, axis_c, points = _infer_axes(np.asarray(grid, dtype=int))
    assignments = _apply_rule(grid.shape, points, x_color, y_color, axis_r, axis_c)
    return _render(grid.shape, bg, assignments)


GEN_RETRIES = 500
GRID_MIN = 9
GRID_MAX = 18
SINGLETON_COUNT = {
    False: (3, 5),
    True: (4, 6),
}
ALL_COLORS = tuple(range(10))


def _far_enough(pos: tuple[int, int], used: set[tuple[int, int]], gap: int = 1) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) > gap for rr, cc in used)


def _quadrant(pos: tuple[int, int], axis_r: int, axis_c: int) -> tuple[int, int]:
    r, c = pos
    return (0 if r < axis_r else 1, 0 if c < axis_c else 1)


def _sample_axis(length: int, require_off_center: bool) -> int:
    choices = list(range(2, length - 2))
    if require_off_center:
        off_center = [a for a in choices if abs((length - 1) - 2 * a) >= 2]
        if off_center:
            choices = off_center
    return int(random.choice(choices))


def _sample_pair_points(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
) -> dict[tuple[int, int], str] | None:
    for _ in range(80):
        used: set[tuple[int, int]] = set()

        dx = random.randint(1, min(axis_c, w - 1 - axis_c))
        x_row = random.choice([r for r in range(h) if r != axis_r])
        x_points = {(x_row, axis_c - dx), (x_row, axis_c + dx)}
        if not all(_far_enough(pt, used) for pt in x_points):
            continue
        used |= x_points

        dy = random.randint(1, min(axis_r, h - 1 - axis_r))
        y_col = random.choice([c for c in range(w) if c != axis_c])
        y_points = {(axis_r - dy, y_col), (axis_r + dy, y_col)}
        if x_points & y_points:
            continue
        if not all(_far_enough(pt, used) for pt in y_points):
            continue
        used |= y_points

        quadrants = {_quadrant(pt, axis_r, axis_c) for pt in used}
        if len(quadrants) < 3:
            continue
        labels = {pt: "x" for pt in x_points}
        labels.update({pt: "y" for pt in y_points})
        return labels
    return None


def _candidate_cells(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    used: set[tuple[int, int]],
    predicate,
) -> list[tuple[int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if r == axis_r or c == axis_c or pos in used or not _far_enough(pos, used):
                continue
            if predicate(pos):
                cells.append(pos)
    random.shuffle(cells)
    return cells


def _open_targets(
    pos: tuple[int, int],
    shape: tuple[int, int],
    axis_r: int,
    axis_c: int,
    used: set[tuple[int, int]],
) -> bool:
    all_coords = {pos}
    for rr, cc in _reflection_targets(pos[0], pos[1], axis_r, axis_c).values():
        if _in_bounds(shape, rr, cc):
            if (rr, cc) in used:
                return False
            all_coords.add((rr, cc))
    return len(all_coords) >= 3


def _pick_required_singletons(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    used: set[tuple[int, int]],
    require_clip: bool,
) -> list[tuple[int, int]] | None:
    shape = (h, w)
    selected: list[tuple[int, int]] = []

    def add_from(predicate) -> bool:
        cells = _candidate_cells(h, w, axis_r, axis_c, used, predicate)
        if not cells:
            return False
        choice = cells[0]
        selected.append(choice)
        used.add(choice)
        return True

    if not add_from(
        lambda pos: all(
            _in_bounds(shape, rr, cc) for rr, cc in _reflection_targets(pos[0], pos[1], axis_r, axis_c).values()
        )
        and _open_targets(pos, shape, axis_r, axis_c, used)
    ):
        return None

    if require_clip:
        if not add_from(
            lambda pos: not _in_bounds(shape, *_reflection_targets(pos[0], pos[1], axis_r, axis_c)["v"])
            and any(
                _in_bounds(shape, rr, cc)
                for name, (rr, cc) in _reflection_targets(pos[0], pos[1], axis_r, axis_c).items()
                if name != "v"
            )
            and _open_targets(pos, shape, axis_r, axis_c, used)
        ):
            return None
        if not add_from(
            lambda pos: not _in_bounds(shape, *_reflection_targets(pos[0], pos[1], axis_r, axis_c)["h"])
            and any(
                _in_bounds(shape, rr, cc)
                for name, (rr, cc) in _reflection_targets(pos[0], pos[1], axis_r, axis_c).items()
                if name != "h"
            )
            and _open_targets(pos, shape, axis_r, axis_c, used)
        ):
            return None

    return selected


def _build_input() -> np.ndarray | None:
    require_clip = random.random() < 0.55
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    axis_r = _sample_axis(h, require_off_center=require_clip or random.random() < 0.65)
    axis_c = _sample_axis(w, require_off_center=require_clip or random.random() < 0.65)
    if h % 2 == 1 and w % 2 == 1 and axis_r == h // 2 and axis_c == w // 2:
        return None

    pair_labels = _sample_pair_points(h, w, axis_r, axis_c)
    if pair_labels is None:
        return None

    used = set(pair_labels)
    required_singletons = _pick_required_singletons(h, w, axis_r, axis_c, used, require_clip)
    if required_singletons is None:
        return None

    total_singletons = random.randint(*SINGLETON_COUNT[require_clip])
    while len(required_singletons) < total_singletons:
        candidates = _candidate_cells(
            h,
            w,
            axis_r,
            axis_c,
            used,
            lambda pos: _open_targets(pos, (h, w), axis_r, axis_c, used),
        )
        if not candidates:
            return None
        choice = candidates[0]
        required_singletons.append(choice)
        used.add(choice)

    bg = random.choice(ALL_COLORS)
    colors = random.sample([c for c in ALL_COLORS if c != bg], 2 + len(required_singletons))
    x_color, y_color = colors[:2]
    singleton_colors = colors[2:]

    grid = np.full((h, w), int(bg), dtype=int)
    for pos, label in pair_labels.items():
        grid[pos] = x_color if label == "x" else y_color
    for pos, color in zip(required_singletons, singleton_colors):
        grid[pos] = int(color)

    if _validate_generated_instance(grid, require_clip=require_clip):
        return grid
    return None


def _validate_generated_instance(grid: np.ndarray, require_clip: bool) -> bool:
    try:
        bg, x_color, y_color, axis_r, axis_c, points = _infer_axes(grid)
        expected_assignments = _apply_rule(grid.shape, points, x_color, y_color, axis_r, axis_c)
        expected = _render(grid.shape, bg, expected_assignments)
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False

    if int(np.count_nonzero(actual != grid)) < 6:
        return False

    has_vertical = False
    has_horizontal = False
    has_diagonal = False
    has_clip = False
    for (r, c), source_color in points.items():
        for kind, (rr, cc) in _reflection_targets(r, c, axis_r, axis_c).items():
            if not _in_bounds(grid.shape, rr, cc):
                has_clip = True
                continue
            if (rr, cc) in points:
                continue
            if kind == "v" and source_color != x_color and actual[rr, cc] == x_color:
                has_vertical = True
            elif kind == "h" and source_color != y_color and actual[rr, cc] == y_color:
                has_horizontal = True
            elif kind == "vh" and source_color not in (x_color, y_color) and actual[rr, cc] == source_color:
                has_diagonal = True

    if not (has_vertical and has_horizontal and has_diagonal):
        return False
    if require_clip and not has_clip:
        return False

    try:
        same_color = _same_color_completion(grid.shape, bg, points, axis_r, axis_c)
    except AssertionError:
        return False
    if np.array_equal(actual, same_color):
        return False

    if grid.shape[0] % 2 == 1 and grid.shape[1] % 2 == 1:
        center_r = grid.shape[0] // 2
        center_c = grid.shape[1] // 2
        if (center_r, center_c) != (axis_r, axis_c):
            try:
                center_assignments = _apply_rule(grid.shape, points, x_color, y_color, center_r, center_c)
                center_out = _render(grid.shape, bg, center_assignments)
            except AssertionError:
                center_out = None
            if center_out is not None and np.array_equal(actual, center_out):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-axis reflection input")
