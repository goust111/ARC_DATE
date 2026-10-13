from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 16
GRID_MAX = 26
AXIS_MARGIN = 3
MIN_AXIS_GAP = 4
MIN_ARROW_COUNT = 2
MAX_ARROW_COUNT = 4
MAX_OBJECTS = 6
MAX_GENERATION_ATTEMPTS = 400
MAX_AXIS_ATTEMPTS = 100
MAX_SHAPE_ATTEMPTS = 80
MAX_PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_grey_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != GREY:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_center(coords: list[tuple[int, int]], kind: str) -> float:
    if kind == "vertical":
        values = [c for _r, c in coords]
    else:
        values = [r for r, _c in coords]
    return (min(values) + max(values)) / 2.0


def _detect_arrows(grid: np.ndarray) -> tuple[str, list[dict[str, int | str]]]:
    h, w = grid.shape
    sides: dict[str, list[tuple[int, int]]] = {"top": [], "bottom": [], "left": [], "right": []}

    if h >= 2:
        for c in range(w):
            color = int(grid[0, c])
            if color not in (BLACK, GREY) and int(grid[1, c]) == color:
                sides["top"].append((c, color))
            color = int(grid[h - 1, c])
            if color not in (BLACK, GREY) and int(grid[h - 2, c]) == color:
                sides["bottom"].append((c, color))

    if w >= 2:
        for r in range(h):
            color = int(grid[r, 0])
            if color not in (BLACK, GREY) and int(grid[r, 1]) == color:
                sides["left"].append((r, color))
            color = int(grid[r, w - 1])
            if color not in (BLACK, GREY) and int(grid[r, w - 2]) == color:
                sides["right"].append((r, color))

    active = [(side, items) for side, items in sides.items() if items]
    assert len(active) == 1, "expected arrows on exactly one border side"
    side, raw = active[0]
    kind = "vertical" if side in ("top", "bottom") else "horizontal"

    arrows: list[dict[str, int | str]] = []
    seen_colors: set[int] = set()
    for axis, color in sorted(raw):
        assert color not in seen_colors, "arrow colors must be unique"
        seen_colors.add(color)
        arrows.append({"axis": int(axis), "color": int(color), "kind": kind, "side": side})
    return side, arrows


def _nearest_arrow_index(coords: list[tuple[int, int]], arrows: list[dict[str, int | str]]) -> int:
    center = _bbox_center(coords, str(arrows[0]["kind"]))
    distances = [abs(center - float(arrow["axis"])) for arrow in arrows]
    best = min(distances)
    return min(i for i, dist in enumerate(distances) if dist == best)


def _reflect_coords(
    coords: list[tuple[int, int]],
    axis: int,
    kind: str,
) -> list[tuple[int, int]]:
    out = []
    if kind == "vertical":
        for r, c in coords:
            out.append((r, 2 * axis - c))
    else:
        for r, c in coords:
            out.append((2 * axis - r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each grey object across the row or column indicated by the nearest border arrow, recoloring the reflected copy by that arrow.

    Input structure:
    - The background is Black.
    - One border side contains several two-pixel arrows: one colored pixel on the border and one same-colored pixel one step inward.
    - Each arrow marks a mirror axis: top/bottom arrows mark columns, left/right arrows mark rows.
    - The interior contains several 4-connected Grey objects, separated from the arrows.

    Rule:
    - Detect every arrow and infer its row/column axis from its border side and aligned position.
    - For each Grey object, measure the distance from the object's center to each candidate axis and choose the nearest arrow.
    - Reflect the whole Grey object across that axis onto the opposite side, coloring the reflected pixels with the chosen arrow's color.
    - Keep the original Grey objects and the border arrows unchanged.

    Key insight:
    - The arrows do not move anything directly; they encode hidden mirror axes, and each object's nearest axis determines both where its copy appears and which color that copy receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _side, arrows = _detect_arrows(g)
    kind = str(arrows[0]["kind"])
    out = g.copy()

    for comp in _find_grey_components(g):
        arrow_index = _nearest_arrow_index(comp, arrows)
        arrow = arrows[arrow_index]
        reflected = _reflect_coords(comp, int(arrow["axis"]), kind)
        for rr, cc in reflected:
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = int(arrow["color"])

    return out


def _tighten(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape() -> list[tuple[int, int]] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        box_h = random.randint(1, 4)
        box_w = random.randint(1, 4)
        max_area = min(6, box_h * box_w)
        min_area = 2 if max_area >= 2 else 1
        area = random.randint(min_area, max_area)
        cells = {(random.randrange(box_h), random.randrange(box_w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        shape = _tighten(cells)
        rows = {r for r, _c in shape}
        cols = {c for _r, c in shape}
        if len(rows) >= 1 and len(cols) >= 1:
            return shape
    return None


def _dims(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _halo4(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            out.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _halo8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            out.add((r, c))
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _axis_positions(length: int, count: int) -> list[int] | None:
    choices = list(range(AXIS_MARGIN, length - AXIS_MARGIN))
    if len(choices) < count:
        return None
    for _ in range(MAX_AXIS_ATTEMPTS):
        axes = sorted(random.sample(choices, count))
        if all(b - a >= MIN_AXIS_GAP for a, b in zip(axes, axes[1:])):
            return axes
    return None


def _distinct_nearest(center: float, axes: list[int]) -> tuple[int, bool]:
    dists = [abs(center - axis) for axis in axes]
    best = min(dists)
    winners = [i for i, dist in enumerate(dists) if dist == best]
    return winners[0], len(winners) == 1


def _candidate_range_vertical(axis: int, width: int, total_width: int, side: str) -> tuple[int, int] | None:
    if side == "left":
        lo = max(1, 2 * axis - (total_width - 2))
        hi = axis - width
    else:
        lo = axis + 1
        hi = min(total_width - 1 - width, 2 * axis - width)
    if lo > hi:
        return None
    return lo, hi


def _candidate_range_horizontal(axis: int, height: int, total_height: int, side: str) -> tuple[int, int] | None:
    if side == "up":
        lo = max(1, 2 * axis - (total_height - 2))
        hi = axis - height
    else:
        lo = axis + 1
        hi = min(total_height - 1 - height, 2 * axis - height)
    if lo > hi:
        return None
    return lo, hi


def _render_arrow_cells(side: str, axis: int, h: int, w: int) -> set[tuple[int, int]]:
    if side == "top":
        return {(0, axis), (1, axis)}
    if side == "bottom":
        return {(h - 1, axis), (h - 2, axis)}
    if side == "left":
        return {(axis, 0), (axis, 1)}
    return {(axis, w - 1), (axis, w - 2)}


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    side = random.choice(("top", "bottom", "left", "right"))
    primary_len = w if side in ("top", "bottom") else h

    max_count = min(MAX_ARROW_COUNT, 1 + (primary_len - 2 * AXIS_MARGIN - 1) // MIN_AXIS_GAP)
    if max_count < MIN_ARROW_COUNT:
        return None
    arrow_count = random.randint(MIN_ARROW_COUNT, max_count)
    axes = _axis_positions(primary_len, arrow_count)
    if axes is None:
        return None

    arrow_colors = random.sample(ARROW_COLORS, arrow_count)
    object_count = random.randint(arrow_count, MAX_OBJECTS)
    assignments = list(range(arrow_count))
    assignments.extend(random.randrange(arrow_count) for _ in range(object_count - arrow_count))
    random.shuffle(assignments)

    grid = np.full((h, w), BLACK, dtype=int)
    expected = np.full((h, w), BLACK, dtype=int)

    arrow_cells: set[tuple[int, int]] = set()
    input_halo = set()
    output_occ = set()
    output_halo = set()
    for axis, color in zip(axes, arrow_colors):
        cells = _render_arrow_cells(side, axis, h, w)
        for r, c in cells:
            grid[r, c] = color
            expected[r, c] = color
        arrow_cells |= cells
    input_halo |= _halo8(arrow_cells, h, w)
    output_occ |= set(arrow_cells)
    output_halo |= _halo4(arrow_cells, h, w)

    kind = "vertical" if side in ("top", "bottom") else "horizontal"
    placements: list[tuple[set[tuple[int, int]], set[tuple[int, int]], int]] = []

    for arrow_index in assignments:
        shape = _random_shape()
        if shape is None:
            return None
        shape_h, shape_w = _dims(shape)
        axis = axes[arrow_index]
        axis_color = arrow_colors[arrow_index]

        if kind == "vertical":
            feasible_sides = []
            if _candidate_range_vertical(axis, shape_w, w, "left") is not None:
                feasible_sides.append("left")
            if _candidate_range_vertical(axis, shape_w, w, "right") is not None:
                feasible_sides.append("right")
        else:
            feasible_sides = []
            if _candidate_range_horizontal(axis, shape_h, h, "up") is not None:
                feasible_sides.append("up")
            if _candidate_range_horizontal(axis, shape_h, h, "down") is not None:
                feasible_sides.append("down")

        if not feasible_sides:
            return None
        random.shuffle(feasible_sides)

        placed = False
        for preferred_side in feasible_sides:
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                if kind == "vertical":
                    col_range = _candidate_range_vertical(axis, shape_w, w, preferred_side)
                    assert col_range is not None
                    row_lo = 1
                    row_hi = h - shape_h - 1
                    if row_lo > row_hi:
                        break
                    r0 = random.randint(row_lo, row_hi)
                    c0 = random.randint(col_range[0], col_range[1])
                else:
                    row_range = _candidate_range_horizontal(axis, shape_h, h, preferred_side)
                    assert row_range is not None
                    col_lo = 1
                    col_hi = w - shape_w - 1
                    if col_lo > col_hi:
                        break
                    r0 = random.randint(row_range[0], row_range[1])
                    c0 = random.randint(col_lo, col_hi)

                original = {(r0 + dr, c0 + dc) for dr, dc in shape}
                reflected = set(_reflect_coords(sorted(original), axis, kind))
                all_cells = original | reflected

                if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in all_cells):
                    continue
                if original & input_halo:
                    continue
                if all_cells & output_halo:
                    continue
                if all_cells & output_occ:
                    continue
                if kind == "vertical":
                    if any(c in axes for _r, c in all_cells):
                        continue
                else:
                    if any(r in axes for r, _c in all_cells):
                        continue

                nearest, unique = _distinct_nearest(_bbox_center(sorted(original), kind), axes)
                if not unique or nearest != arrow_index:
                    continue

                placed = True
                placements.append((original, reflected, axis_color))
                input_halo |= _halo8(original, h, w)
                output_occ |= all_cells
                output_halo |= _halo4(all_cells, h, w)
                for r, c in original:
                    grid[r, c] = GREY
                    expected[r, c] = GREY
                for r, c in reflected:
                    expected[r, c] = axis_color
                break
            if placed:
                break
        if not placed:
            return None

    if not placements:
        return None
    if len({_nearest_arrow_index(sorted(list(original)), [{"axis": a, "color": col, "kind": kind, "side": side} for a, col in zip(axes, arrow_colors)]) for original, _reflected, _color in placements}) < min(arrow_count, 2):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is None:
            continue
        grid, expected = candidate
        try:
            actual = main(grid)
        except Exception:
            continue
        if actual.shape != expected.shape:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
