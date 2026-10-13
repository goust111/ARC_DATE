from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


ARROW_SHAPES = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}
HORIZONTAL_DIRS = {"left", "right"}
VERTICAL_DIRS = {"up", "down"}


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)

    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_direction(cells: list[tuple[int, int]]) -> str | None:
    if len(cells) != 5:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None

    norm = frozenset((r - r0, c - c0) for r, c in cells)
    for direction, shape in ARROW_SHAPES.items():
        if norm == shape:
            return direction
    return None


def _arrow_distance(
    bbox: tuple[int, int, int, int], direction: str, height: int, width: int
) -> int:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return int(r0)
    if direction == "down":
        return int(height - 1 - r1)
    if direction == "left":
        return int(c0)
    if direction == "right":
        return int(width - 1 - c1)
    raise AssertionError(direction)


def _analyze_grid(grid: np.ndarray, strict_counts: bool = False) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    bg = _background_color(g)
    height, width = g.shape
    infos: list[dict[str, object]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        mask = g == color
        components = _connected_components(mask)
        arrow_matches: list[tuple[list[tuple[int, int]], str]] = []
        for comp in components:
            direction = _arrow_direction(comp)
            if direction is not None:
                arrow_matches.append((comp, direction))

        if len(arrow_matches) != 1:
            raise AssertionError(f"color {color} must contain exactly one arrow")

        arrow_cells, direction = arrow_matches[0]
        arrow_set = set(arrow_cells)
        body_pixels = [(int(r), int(c)) for r, c in map(tuple, np.argwhere(mask)) if (int(r), int(c)) not in arrow_set]

        if strict_counts and not (3 <= len(body_pixels) <= 8):
            raise AssertionError(f"color {color} must have 3-8 non-arrow pixels")

        bbox = _bbox(arrow_cells)
        infos.append(
            {
                "color": int(color),
                "direction": direction,
                "arrow_cells": arrow_cells,
                "body_pixels": body_pixels,
                "bbox": bbox,
                "distance": _arrow_distance(bbox, direction, height, width),
            }
        )

    if not infos:
        raise AssertionError("expected at least one colored arrow")

    return bg, infos


def _compacted_positions(
    body_pixels: list[tuple[int, int]], direction: str, height: int, width: int
) -> list[tuple[int, int]]:
    if not body_pixels:
        return []

    lanes: dict[int, list[int]] = defaultdict(list)
    out: list[tuple[int, int]] = []

    if direction in HORIZONTAL_DIRS:
        for r, c in body_pixels:
            lanes[int(r)].append(int(c))
        for r, cols in lanes.items():
            cols.sort()
            start = 0 if direction == "left" else width - len(cols)
            for i, _col in enumerate(cols):
                out.append((int(r), int(start + i)))
        return out

    for r, c in body_pixels:
        lanes[int(c)].append(int(r))
    for c, rows in lanes.items():
        rows.sort()
        start = 0 if direction == "up" else height - len(rows)
        for i, _row in enumerate(rows):
            out.append((int(start + i), int(c)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each color's unique 3x3 arrow, remove it, and compact the remaining
    same-color pixels toward the indicated border while preserving order within
    each row or column. Collisions are resolved by arrow-to-border distance.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg, infos = _analyze_grid(g, strict_counts=False)

    out = np.full((height, width), int(bg), dtype=int)

    # Paint farthest stacks first so arrows closest to their target edge end up on top.
    paint_order = sorted(
        infos,
        key=lambda info: (
            -int(info["distance"]),
            int(info["bbox"][0]),
            int(info["bbox"][1]),
        ),
    )

    for info in paint_order:
        positions = _compacted_positions(
            info["body_pixels"],
            str(info["direction"]),
            height,
            width,
        )
        color = int(info["color"])
        for r, c in positions:
            out[int(r), int(c)] = color

    return out


GEN_BG = 0
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_DIRECTIONS = ["up", "down", "left", "right"]
GEN_HEIGHT_RANGE = (16, 24)
GEN_WIDTH_RANGE = (16, 24)
GEN_COLOR_COUNT_CHOICES = [2, 3, 4, 4]
GEN_SCATTER_RANGE = (3, 8)
GEN_OVERLAP_PROB = 0.55
GEN_MAX_ATTEMPTS = 600


def _can_place(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], height: int, width: int) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False

    for r, c in cell_set:
        if not (0 <= r < height and 0 <= c < width):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _stamp(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _all_arrow_positions(height: int, width: int, direction: str) -> list[list[tuple[int, int]]]:
    shape = ARROW_SHAPES[direction]
    placements: list[list[tuple[int, int]]] = []
    for r0 in range(height - 2):
        for c0 in range(width - 2):
            placements.append([(r0 + dr, c0 + dc) for dr, dc in shape])
    random.shuffle(placements)
    return placements


def _place_arrow(
    occupied: set[tuple[int, int]], height: int, width: int, direction: str
) -> list[tuple[int, int]] | None:
    for cells in _all_arrow_positions(height, width, direction):
        if _can_place(cells, occupied, height, width):
            return cells
    return None


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        values.append(int(cut - prev))
        prev = cut
    random.shuffle(values)
    return values


def _line_specs(axis_size: int, total: int, fixed_line: int | None = None, fixed_count: int | None = None) -> dict[int, int]:
    if fixed_line is not None:
        if fixed_count is None or fixed_count > total:
            raise AssertionError("invalid fixed line request")
        specs = {int(fixed_line): int(fixed_count)}
        remaining = total - fixed_count
        if remaining == 0:
            return specs
        available = [i for i in range(axis_size) if i != fixed_line]
        extra_parts = random.randint(1, min(2, remaining, len(available)))
        counts = _split_total(remaining, extra_parts)
        for line, count in zip(random.sample(available, extra_parts), counts, strict=False):
            specs[int(line)] = int(count)
        return specs

    max_parts = min(3, total - 1, axis_size)
    parts = 1 if max_parts <= 1 else random.randint(1, max_parts)
    counts = _split_total(total, parts)
    return {int(line): int(count) for line, count in zip(random.sample(range(axis_size), parts), counts, strict=False)}


def _place_scatter(
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
    direction: str,
    specs: dict[int, int],
) -> list[tuple[int, int]] | None:
    temp_occupied = set(occupied)
    points: list[tuple[int, int]] = []
    items = list(specs.items())
    random.shuffle(items)

    if direction in HORIZONTAL_DIRS:
        for row, count in items:
            candidates = list(range(width))
            random.shuffle(candidates)
            placed = 0
            for col in candidates:
                cell = (int(row), int(col))
                if _can_place([cell], temp_occupied, height, width):
                    points.append(cell)
                    temp_occupied.add(cell)
                    placed += 1
                    if placed == count:
                        break
            if placed != count:
                return None
        return points

    for col, count in items:
        candidates = list(range(height))
        random.shuffle(candidates)
        placed = 0
        for row in candidates:
            cell = (int(row), int(col))
            if _can_place([cell], temp_occupied, height, width):
                points.append(cell)
                temp_occupied.add(cell)
                placed += 1
                if placed == count:
                    break
        if placed != count:
            return None
    return points


def _forced_overlap_plan(directions: list[str], height: int, width: int) -> dict[int, tuple[int, int]] | None:
    horizontal = [i for i, direction in enumerate(directions) if direction in HORIZONTAL_DIRS]
    vertical = [i for i, direction in enumerate(directions) if direction in VERTICAL_DIRS]
    if not horizontal or not vertical:
        return None

    h_idx = random.choice(horizontal)
    v_idx = random.choice(vertical)
    h_dir = directions[h_idx]
    v_dir = directions[v_idx]

    h_count = random.randint(2, min(4, width))
    v_count = random.randint(2, min(4, height))

    if h_dir == "left":
        overlap_col = random.randint(0, h_count - 1)
    else:
        overlap_col = random.randint(width - h_count, width - 1)

    if v_dir == "up":
        overlap_row = random.randint(0, v_count - 1)
    else:
        overlap_row = random.randint(height - v_count, height - 1)

    return {
        int(h_idx): (int(overlap_row), int(h_count)),
        int(v_idx): (int(overlap_col), int(v_count)),
    }


def _claim_map(infos: list[dict[str, object]], height: int, width: int) -> dict[tuple[int, int], list[int]]:
    claims: dict[tuple[int, int], list[int]] = defaultdict(list)
    for info in infos:
        positions = _compacted_positions(
            info["body_pixels"],
            str(info["direction"]),
            height,
            width,
        )
        for pos in positions:
            claims[pos].append(int(info["color"]))
    return claims


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_MAX_ATTEMPTS):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)
        n_colors = random.choice(GEN_COLOR_COUNT_CHOICES)
        colors = random.sample(GEN_COLORS, n_colors)
        directions = random.sample(GEN_DIRECTIONS, n_colors)

        force_overlap = random.random() < GEN_OVERLAP_PROB
        forced_plan = _forced_overlap_plan(directions, height, width) if force_overlap else None
        if force_overlap and forced_plan is None:
            force_overlap = False

        grid = np.full((height, width), GEN_BG, dtype=int)
        occupied: set[tuple[int, int]] = set()
        plans: list[dict[str, object]] = []
        failed = False

        for color, direction in zip(colors, directions, strict=False):
            arrow_cells = _place_arrow(occupied, height, width, direction)
            if arrow_cells is None:
                failed = True
                break
            occupied.update(arrow_cells)
            _stamp(grid, arrow_cells, color)
            plans.append(
                {
                    "color": int(color),
                    "direction": direction,
                    "arrow_cells": arrow_cells,
                }
            )

        if failed:
            continue

        for idx, plan in enumerate(plans):
            direction = str(plan["direction"])
            total = random.randint(*GEN_SCATTER_RANGE)
            if forced_plan is not None and idx in forced_plan:
                fixed_line, fixed_count = forced_plan[idx]
                total = random.randint(max(total, fixed_count), GEN_SCATTER_RANGE[1])
                axis_size = height if direction in HORIZONTAL_DIRS else width
                specs = _line_specs(axis_size, total, fixed_line=fixed_line, fixed_count=fixed_count)
            else:
                axis_size = height if direction in HORIZONTAL_DIRS else width
                specs = _line_specs(axis_size, total)

            points = _place_scatter(occupied, height, width, direction, specs)
            if points is None:
                failed = True
                break
            occupied.update(points)
            _stamp(grid, points, int(plan["color"]))
            plan["body_pixels"] = points

        if failed:
            continue

        bg, infos = _analyze_grid(grid, strict_counts=True)
        assert int(bg) == GEN_BG
        assert len(infos) == n_colors

        if force_overlap:
            claims = _claim_map(infos, height, width)
            if not any(len(colors_here) > 1 for colors_here in claims.values()):
                continue

        out = main(grid)
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
