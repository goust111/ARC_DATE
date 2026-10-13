from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np


LINE_COLOR = 5
ALL_COLORS = tuple(range(10))
AREA_CANDIDATES = (5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _closest_factor_pair(area: int) -> tuple[int, int]:
    root = int(math.isqrt(area))
    for w in range(root, 0, -1):
        if area % w == 0:
            return w, area // w
    raise AssertionError("unreachable")


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == LINE_COLOR)]
    assert len(rows) == 1, "expected one full-width grey separator row"
    return rows[0]


def _background_color(grid: np.ndarray, sep_row: int) -> int:
    mask = np.ones_like(grid, dtype=bool)
    mask[sep_row, :] = False
    values = [int(v) for v in grid[mask].reshape(-1) if int(v) != LINE_COLOR]
    assert values, "non-separator cells must exist"
    bg, _ = Counter(values).most_common(1)[0]
    return int(bg)


def _shape_dims(cells: list[tuple[int, int]]) -> tuple[int, int]:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    return h, w


def _normalize_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = sorted((r - min_r, c - min_c) for r, c in cells)
    return normalized


def _transform_cells(cells: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    out = list(cells)
    for _ in range(rng.randrange(4)):
        out = [(c, -r) for r, c in out]
    if rng.random() < 0.5:
        out = [(r, -c) for r, c in out]
    if rng.random() < 0.5:
        out = [(-r, c) for r, c in out]
    return _normalize_cells(out)


def _random_connected_shape(size: int, rng: random.Random) -> list[tuple[int, int]]:
    last: list[tuple[int, int]] | None = None
    for _ in range(80):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < size:
            choice = rng.choice(tuple(frontier))
            frontier.remove(choice)
            cells.add(choice)
            for nxt in _neighbors4(*choice):
                if nxt not in cells:
                    frontier.add(nxt)
        shape = _transform_cells(list(cells), rng)
        last = shape
        h, w = _shape_dims(shape)
        if size <= 2:
            return shape
        if h == 1 or w == 1 or h * w != size:
            return shape
    assert last is not None
    return last


def _partition_total(total: int, parts: int, min_each: int, rng: random.Random) -> list[int]:
    assert min_each * parts <= total
    sizes = [min_each] * parts
    for _ in range(total - min_each * parts):
        sizes[rng.randrange(parts)] += 1
    rng.shuffle(sizes)
    return sizes


def _sample_areas(n_colors: int, rng: random.Random) -> list[int]:
    tie_mode = n_colors >= 2 and rng.random() < 0.5
    if not tie_mode:
        areas = rng.sample(list(AREA_CANDIDATES), n_colors)
        rng.shuffle(areas)
        return areas

    if n_colors == 4 and rng.random() < 0.3:
        a, b = rng.sample(list(AREA_CANDIDATES), 2)
        areas = [a, a, b, b]
        rng.shuffle(areas)
        return areas

    tie_size = 3 if n_colors >= 3 and rng.random() < 0.3 else 2
    tie_size = min(tie_size, n_colors)
    repeated = rng.choice(AREA_CANDIDATES)
    pool = [area for area in AREA_CANDIDATES if area != repeated]
    areas = [repeated] * tie_size + rng.sample(pool, n_colors - tie_size)
    rng.shuffle(areas)
    return areas


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    occupied[r0:r1, c0:c1] = True


def _valid_positions(
    occupied: np.ndarray,
    piece_h: int,
    piece_w: int,
    row_min: int,
    exact_row: int | None = None,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    if piece_h > h or piece_w > w:
        return []
    rows = [exact_row] if exact_row is not None else range(row_min, h - piece_h + 1)
    positions: list[tuple[int, int]] = []
    for r in rows:
        if r is None or r < 0 or r + piece_h > h:
            continue
        for c in range(0, w - piece_w + 1):
            r0 = max(0, r - 1)
            r1 = min(h, r + piece_h + 1)
            c0 = max(0, c - 1)
            c1 = min(w, c + piece_w + 1)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _paint_piece(grid: np.ndarray, color: int, top: int, left: int, cells: list[tuple[int, int]]) -> None:
    for dr, dc in cells:
        grid[top + dr, left + dc] = color


def _component_count(grid: np.ndarray, color: int, row_limit: int) -> int:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    count = 0
    for r in range(row_limit):
        for c in range(w):
            if int(grid[r, c]) != color or (r, c) in seen:
                continue
            count += 1
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < row_limit and 0 <= nc < w and int(grid[nr, nc]) == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered same-color fragments by total area, infer the nearest-square
    rectangle size for each color, then render solid rectangles below the grey divider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    sep_row = _find_separator_row(g)
    bg = _background_color(g, sep_row)
    assert bg != LINE_COLOR

    color_info: dict[int, dict[str, int]] = {}
    for r in range(sep_row):
        for c in range(w):
            value = int(g[r, c])
            if value in (bg, LINE_COLOR):
                continue
            if value not in color_info:
                color_info[value] = {
                    "area": 0,
                    "first_row": r,
                    "first_col": c,
                }
            color_info[value]["area"] += 1

    assert color_info, "expected at least one fragment color above the separator"

    ordered = sorted(
        (
            (
                color,
                info["area"],
                info["first_row"],
                info["first_col"],
            )
            for color, info in color_info.items()
        ),
        key=lambda item: (item[1], item[2], item[3]),
    )

    out = np.full((h, w), bg, dtype=int)
    out[sep_row, :] = LINE_COLOR

    cursor = 0
    out_h = h - sep_row - 1
    for color, area, _first_row, _first_col in ordered:
        rect_w, rect_h = _closest_factor_pair(area)
        assert rect_h <= out_h, "output area is too short for an inferred rectangle"
        assert cursor + rect_w <= w, "output area is too narrow for packed rectangles"
        out[sep_row + 1 : sep_row + 1 + rect_h, cursor : cursor + rect_w] = int(color)
        cursor += rect_w + 1

    return out


def _render_expected_output(
    shape: tuple[int, int],
    bg: int,
    specs: list[dict[str, object]],
    sep_row: int,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    out[sep_row, :] = LINE_COLOR
    cursor = 0
    for spec in sorted(specs, key=lambda item: (item["area"], item["first_row"], item["first_col"])):
        rect_w, rect_h = _closest_factor_pair(int(spec["area"]))
        color = int(spec["color"])
        out[sep_row + 1 : sep_row + 1 + rect_h, cursor : cursor + rect_w] = color
        cursor += rect_w + 1
    return out


def _generate_group_specs(rng: random.Random) -> list[dict[str, object]]:
    bg = rng.choice([c for c in ALL_COLORS if c != LINE_COLOR])
    n_colors = rng.randint(2, 4)
    colors = rng.sample([c for c in ALL_COLORS if c not in (bg, LINE_COLOR)], n_colors)
    areas = _sample_areas(n_colors, rng)

    specs: list[dict[str, object]] = []
    for color, area in zip(colors, areas):
        piece_count = rng.randint(2, 4)
        min_each = 2 if area >= 2 * piece_count and rng.random() < 0.75 else 1
        sizes = _partition_total(area, piece_count, min_each, rng)
        pieces = []
        for size in sizes:
            cells = _random_connected_shape(size, rng)
            piece_h, piece_w = _shape_dims(cells)
            pieces.append(
                {
                    "size": size,
                    "cells": cells,
                    "height": piece_h,
                    "width": piece_w,
                }
            )
        anchor_height = min(int(piece["height"]) for piece in pieces)
        specs.append(
            {
                "color": color,
                "area": area,
                "pieces": pieces,
                "anchor_height": anchor_height,
            }
        )
    return [dict(spec, bg=bg) for spec in specs]


def _assign_layout(specs: list[dict[str, object]], rng: random.Random) -> tuple[int, int, int] | None:
    total_area = sum(int(spec["area"]) for spec in specs)
    rect_pairs = [_closest_factor_pair(int(spec["area"])) for spec in specs]
    required_output_w = sum(pair[0] for pair in rect_pairs) + len(specs) - 1
    required_output_h = max(pair[1] for pair in rect_pairs)

    output_h = required_output_h + rng.randint(0, 2)
    min_width = max(required_output_w, 12, math.ceil(math.sqrt(total_area * 3.5)))
    if min_width > 30:
        return None

    width = rng.randint(min_width, min(30, min_width + 8))
    input_h_min = max(8, 2 * len(specs) + 4, math.ceil(total_area * 2.4 / width))
    input_h_max = 30 - 1 - output_h
    if input_h_min > input_h_max:
        return None

    input_h = rng.randint(input_h_min, min(input_h_max, input_h_min + 6))
    return input_h, width, output_h


def _assign_first_rows(specs: list[dict[str, object]], input_h: int, rng: random.Random) -> bool:
    max_anchor_h = max(int(spec["anchor_height"]) for spec in specs)
    top_limit = min(input_h - max_anchor_h, input_h // 2 + 2)
    choices = list(range(0, top_limit + 1, 2))
    if len(choices) < len(specs):
        choices = list(range(0, input_h - max_anchor_h + 1))
    if len(choices) < len(specs):
        return False

    rows = rng.sample(choices, len(specs))
    rng.shuffle(rows)
    for spec, row in zip(specs, rows):
        spec["first_row"] = int(row)
    return True


def _place_fragments(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    sep_row: int,
    rng: random.Random,
) -> bool:
    input_h, width = sep_row, grid.shape[1]
    occupied = np.zeros((input_h, width), dtype=bool)

    anchor_tasks = []
    remaining_tasks = []
    for spec in specs:
        pieces = list(spec["pieces"])
        min_anchor_h = min(int(piece["height"]) for piece in pieces)
        anchor_candidates = [piece for piece in pieces if int(piece["height"]) == min_anchor_h]
        anchor_piece = rng.choice(anchor_candidates)
        anchor_id = id(anchor_piece)
        for piece in pieces:
            task = {
                "color": int(spec["color"]),
                "first_row": int(spec["first_row"]),
                "cells": piece["cells"],
                "height": int(piece["height"]),
                "width": int(piece["width"]),
                "size": int(piece["size"]),
                "piece_count": len(pieces),
            }
            if id(piece) == anchor_id:
                anchor_tasks.append(task)
            else:
                remaining_tasks.append(task)

    anchor_tasks.sort(key=lambda task: (-task["size"], -task["width"]))
    remaining_tasks.sort(key=lambda task: (-task["size"], -task["width"]))

    for task in anchor_tasks:
        positions = _valid_positions(
            occupied,
            int(task["height"]),
            int(task["width"]),
            row_min=int(task["first_row"]),
            exact_row=int(task["first_row"]),
        )
        if not positions:
            return False
        top, left = rng.choice(positions)
        _paint_piece(grid, int(task["color"]), top, left, list(task["cells"]))
        _mark_occupied(occupied, top, left, int(task["height"]), int(task["width"]))

    for task in remaining_tasks:
        positions = _valid_positions(
            occupied,
            int(task["height"]),
            int(task["width"]),
            row_min=int(task["first_row"]),
        )
        if not positions:
            return False
        top, left = rng.choice(positions)
        _paint_piece(grid, int(task["color"]), top, left, list(task["cells"]))
        _mark_occupied(occupied, top, left, int(task["height"]), int(task["width"]))

    return True


def _spec_first_positions(grid: np.ndarray, specs: list[dict[str, object]], sep_row: int) -> bool:
    for spec in specs:
        color = int(spec["color"])
        pts = np.argwhere(grid[:sep_row] == color)
        if len(pts) != int(spec["area"]):
            return False
        top_row = int(pts[:, 0].min())
        top_cols = pts[pts[:, 0] == top_row][:, 1]
        first_col = int(top_cols.min())
        spec["first_row"] = top_row
        spec["first_col"] = first_col
        if _component_count(grid, color, sep_row) != len(spec["pieces"]):
            return False
    return True


def _background_dominant(grid: np.ndarray, bg: int, sep_row: int) -> bool:
    mask = np.ones_like(grid, dtype=bool)
    mask[sep_row, :] = False
    values = [int(v) for v in grid[mask].reshape(-1)]
    counts = Counter(values)
    if not counts:
        return False
    color, _ = counts.most_common(1)[0]
    return int(color) == int(bg)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _generate_group_specs(rng)
    bg = int(specs[0]["bg"])
    layout = _assign_layout(specs, rng)
    if layout is None:
        return None
    input_h, width, output_h = layout
    if not _assign_first_rows(specs, input_h, rng):
        return None

    height = input_h + 1 + output_h
    sep_row = input_h
    grid = np.full((height, width), bg, dtype=int)
    grid[sep_row, :] = LINE_COLOR

    if not _place_fragments(grid, specs, sep_row, rng):
        return None
    if not _spec_first_positions(grid, specs, sep_row):
        return None
    if not _background_dominant(grid, bg, sep_row):
        return None

    expected = _render_expected_output(grid.shape, bg, specs, sep_row)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
