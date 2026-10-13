import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

OBJECT_COLORS = (RED, GREEN, YELLOW, AZURE)
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 400
GEN_GRID_MIN = 22
GEN_GRID_MAX = 28
GEN_BAND_SPAN = 6
GEN_BAND_GAP = 1
GEN_MAIN_DIMS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3))
GEN_EXTRA_DIMS = ((1, 1), (1, 2), (2, 1), (2, 2))


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(
    grid: np.ndarray,
    *,
    include_colors: set[int] | None = None,
    exclude_colors: set[int] | None = None,
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c]:
                continue
            if include_colors is not None and color not in include_colors:
                continue
            if exclude_colors is not None and color in exclude_colors:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _straight_triplet(coords: np.ndarray) -> tuple[str, list[tuple[int, int]]] | None:
    if int(coords.shape[0]) != 3:
        return None
    pts = sorted((int(r), int(c)) for r, c in coords)
    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    if len(rows) == 1:
        row = pts[0][0]
        ordered = sorted(pts, key=lambda p: p[1])
        if [c for _, c in ordered] == [ordered[0][1] + i for i in range(3)]:
            return "h", [(row, ordered[0][1] + i) for i in range(3)]
    if len(cols) == 1:
        col = pts[0][1]
        ordered = sorted(pts, key=lambda p: p[0])
        if [r for r, _ in ordered] == [ordered[0][0] + i for i in range(3)]:
            return "v", [(ordered[0][0] + i, col) for i in range(3)]
    return None


def _is_isolated_marker(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    color = int(grid[r, c])
    if color not in OBJECT_COLORS:
        return False
    for nr, nc in _neighbors4(r, c, h, w):
        if int(grid[nr, nc]) == color:
            return False
    return True


def _find_instructions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    components = _find_components(g, exclude_colors={BLACK, GREY})
    instructions: list[dict[str, object]] = []

    for comp in components:
        if int(comp["area"]) != 3:
            continue
        line = _straight_triplet(np.asarray(comp["coords"], dtype=int))
        if line is None:
            continue
        axis, cells = line
        marker: tuple[int, int] | None = None
        direction: str | None = None
        if axis == "h":
            row = cells[0][0]
            left_marker = (row, cells[0][1] - 1)
            right_marker = (row, cells[-1][1] + 1)
            has_left = _is_isolated_marker(g, *left_marker)
            has_right = _is_isolated_marker(g, *right_marker)
            if has_left ^ has_right:
                marker = left_marker if has_left else right_marker
                direction = "right" if has_left else "left"
        else:
            col = cells[0][1]
            top_marker = (cells[0][0] - 1, col)
            bottom_marker = (cells[-1][0] + 1, col)
            has_top = _is_isolated_marker(g, *top_marker)
            has_bottom = _is_isolated_marker(g, *bottom_marker)
            if has_top ^ has_bottom:
                marker = top_marker if has_top else bottom_marker
                direction = "down" if has_top else "up"
        if marker is None or direction is None:
            continue
        instructions.append(
            {
                "target_color": int(g[marker]),
                "direction": direction,
                "marker": marker,
                "arrow_cells": tuple(cells),
            }
        )

    instructions.sort(key=lambda inst: (inst["marker"][0], inst["marker"][1]))  # type: ignore[index]
    return instructions


def _stack_color(original_color: int, object_count: int) -> int:
    if object_count <= 1:
        return int(original_color)
    if object_count == 2:
        return MAROON
    if object_count == 3:
        return MAGENTA
    return ORANGE


def _sort_key(comp: dict[str, object], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
    if direction == "up":
        return int(r0), int(c0)
    if direction == "down":
        return -int(r1), int(c0)
    if direction == "left":
        return int(c0), int(r0)
    return -int(c1), int(r0)


def _can_translate(
    coords: np.ndarray,
    dr: int,
    dc: int,
    occupied: np.ndarray,
) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        nr = int(r) + int(dr)
        nc = int(c) + int(dc)
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        if occupied[nr, nc]:
            return False
    return True


def _apply_instruction(grid: np.ndarray, target_color: int, direction: str) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    components = _find_components(out, include_colors={int(target_color)})
    if not components:
        return out

    for comp in components:
        coords = np.asarray(comp["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = BLACK

    occupied = out != BLACK
    dr, dc = DIRECTION_VECTORS[direction]
    paint_color = _stack_color(int(target_color), len(components))

    for comp in sorted(components, key=lambda item: _sort_key(item, direction)):
        coords = np.asarray(comp["coords"], dtype=int)
        shift = 0
        while _can_translate(coords, dr * (shift + 1), dc * (shift + 1), occupied):
            shift += 1
        moved = coords + np.array([dr * shift, dc * shift], dtype=int)
        out[moved[:, 0], moved[:, 1]] = int(paint_color)
        occupied[moved[:, 0], moved[:, 1]] = True

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each arrow as “push the color on its tail toward that wall”, then
    recolor the packed chain by the number of merged same-color objects.
    """
    g = np.asarray(grid, dtype=int)
    instructions = _find_instructions(g)
    out = g.copy()

    seen_colors: set[int] = set()
    ordered: list[dict[str, object]] = []
    for inst in instructions:
        color = int(inst["target_color"])
        if color in seen_colors:
            continue
        seen_colors.add(color)
        ordered.append(inst)
        mr, mc = inst["marker"]  # type: ignore[misc]
        out[mr, mc] = BLACK
        for ar, ac in inst["arrow_cells"]:  # type: ignore[misc]
            out[int(ar), int(ac)] = BLACK

    for inst in ordered:
        out = _apply_instruction(out, int(inst["target_color"]), str(inst["direction"]))

    return out


def _sample_object_count() -> int:
    branch = random.choice((1, 2, 3, 4))
    if branch < 4:
        return branch
    return random.choice((4, 5))


def _band_starts(length: int, count: int) -> list[int] | None:
    required = count * GEN_BAND_SPAN + (count - 1) * GEN_BAND_GAP
    if required > length:
        return None
    slack = length - required
    gaps = [0] * (count + 1)
    for _ in range(slack):
        gaps[random.randrange(count + 1)] += 1
    starts: list[int] = []
    cursor = 1 + gaps[0]
    for idx in range(count):
        starts.append(cursor)
        cursor += GEN_BAND_SPAN
        if idx + 1 < count:
            cursor += GEN_BAND_GAP + gaps[idx + 1]
    return starts


def _cells_clear(cells: list[tuple[int, int]], occupied: np.ndarray, moat: int = 0) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if occupied[r, c]:
            return False
        if moat <= 0:
            continue
        for rr in range(max(1, r - moat), min(h - 1, r + moat + 1)):
            for cc in range(max(1, c - moat), min(w - 1, c + moat + 1)):
                if occupied[rr, cc]:
                    return False
    return True


def _rect_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr in range(height) for dc in range(width)]


def _place_horizontal_instruction(
    grid: np.ndarray,
    occupied: np.ndarray,
    band_start: int,
    color: int,
    direction: str,
    count: int,
) -> bool:
    h, w = grid.shape
    arrow_row = band_start
    zone_top = band_start + 2
    zone_bottom = band_start + 4
    anchor_row = band_start + 3

    for _ in range(60):
        dims = [random.choice(GEN_MAIN_DIMS) for _ in range(count)]
        object_cells: list[list[tuple[int, int]]] = []
        if direction == "left":
            left = 1 + random.randint(3, 6)
            ok = True
            for obj_h, obj_w in dims:
                top_lo = max(zone_top, anchor_row - obj_h + 1)
                top_hi = min(anchor_row, zone_bottom - obj_h + 1)
                if top_lo > top_hi:
                    ok = False
                    break
                top = random.randint(top_lo, top_hi)
                cells = _rect_cells(top, left, obj_h, obj_w)
                if not _cells_clear(cells, occupied):
                    ok = False
                    break
                object_cells.append(cells)
                left += obj_w + random.randint(1, 3)
                if left >= w - 1:
                    ok = False
                    break
            if not ok:
                continue
        else:
            right = w - 2 - random.randint(3, 6)
            ok = True
            for obj_h, obj_w in dims:
                top_lo = max(zone_top, anchor_row - obj_h + 1)
                top_hi = min(anchor_row, zone_bottom - obj_h + 1)
                if top_lo > top_hi:
                    ok = False
                    break
                top = random.randint(top_lo, top_hi)
                left = right - obj_w + 1
                cells = _rect_cells(top, left, obj_h, obj_w)
                if not _cells_clear(cells, occupied):
                    ok = False
                    break
                object_cells.append(cells)
                right = left - random.randint(1, 3) - 1
                if right <= 0:
                    ok = False
                    break
            if not ok:
                continue

        span_start = random.randint(2, w - 6)
        if direction == "right":
            marker = (arrow_row, span_start)
            arrow_cells = [(arrow_row, span_start + i) for i in range(1, 4)]
        else:
            marker = (arrow_row, span_start + 3)
            arrow_cells = [(arrow_row, span_start + i) for i in range(3)]
        control_cells = [marker, *arrow_cells]
        if not _cells_clear(control_cells, occupied):
            continue

        for cells in object_cells:
            rows, cols = zip(*cells)
            if anchor_row not in rows:
                break
            grid[list(rows), list(cols)] = int(color)
            occupied[list(rows), list(cols)] = True
        else:
            grid[marker[0], marker[1]] = int(color)
            occupied[marker[0], marker[1]] = True
            for ar, ac in arrow_cells:
                grid[ar, ac] = BLUE
                occupied[ar, ac] = True
            return True
    return False


def _place_vertical_instruction(
    grid: np.ndarray,
    occupied: np.ndarray,
    band_start: int,
    color: int,
    direction: str,
    count: int,
) -> bool:
    h, w = grid.shape
    arrow_col = band_start
    zone_left = band_start + 2
    zone_right = band_start + 4
    anchor_col = band_start + 3

    for _ in range(60):
        dims = [random.choice(GEN_MAIN_DIMS) for _ in range(count)]
        object_cells: list[list[tuple[int, int]]] = []
        if direction == "up":
            top = 1 + random.randint(3, 6)
            ok = True
            for obj_h, obj_w in dims:
                left_lo = max(zone_left, anchor_col - obj_w + 1)
                left_hi = min(anchor_col, zone_right - obj_w + 1)
                if left_lo > left_hi:
                    ok = False
                    break
                left = random.randint(left_lo, left_hi)
                cells = _rect_cells(top, left, obj_h, obj_w)
                if not _cells_clear(cells, occupied):
                    ok = False
                    break
                object_cells.append(cells)
                top += obj_h + random.randint(1, 3)
                if top >= h - 1:
                    ok = False
                    break
            if not ok:
                continue
        else:
            bottom = h - 2 - random.randint(3, 6)
            ok = True
            for obj_h, obj_w in dims:
                left_lo = max(zone_left, anchor_col - obj_w + 1)
                left_hi = min(anchor_col, zone_right - obj_w + 1)
                if left_lo > left_hi:
                    ok = False
                    break
                left = random.randint(left_lo, left_hi)
                top = bottom - obj_h + 1
                cells = _rect_cells(top, left, obj_h, obj_w)
                if not _cells_clear(cells, occupied):
                    ok = False
                    break
                object_cells.append(cells)
                bottom = top - random.randint(1, 3) - 1
                if bottom <= 0:
                    ok = False
                    break
            if not ok:
                continue

        span_start = random.randint(2, h - 6)
        if direction == "down":
            marker = (span_start, arrow_col)
            arrow_cells = [(span_start + i, arrow_col) for i in range(1, 4)]
        else:
            marker = (span_start + 3, arrow_col)
            arrow_cells = [(span_start + i, arrow_col) for i in range(3)]
        control_cells = [marker, *arrow_cells]
        if not _cells_clear(control_cells, occupied):
            continue

        for cells in object_cells:
            rows, cols = zip(*cells)
            if anchor_col not in cols:
                break
            grid[list(rows), list(cols)] = int(color)
            occupied[list(rows), list(cols)] = True
        else:
            grid[marker[0], marker[1]] = int(color)
            occupied[marker[0], marker[1]] = True
            for ar, ac in arrow_cells:
                grid[ar, ac] = BLUE
                occupied[ar, ac] = True
            return True
    return False


def _place_extras(
    grid: np.ndarray,
    occupied: np.ndarray,
    colors: list[int],
    *,
    free_rows: set[int] | None = None,
    free_cols: set[int] | None = None,
) -> None:
    h, w = grid.shape
    for color in colors:
        for _ in range(80):
            obj_h, obj_w = random.choice(GEN_EXTRA_DIMS)
            top = random.randint(1, h - 1 - obj_h)
            left = random.randint(1, w - 1 - obj_w)
            cells = _rect_cells(top, left, obj_h, obj_w)
            if free_rows is not None and any(r not in free_rows for r, _ in cells):
                continue
            if free_cols is not None and any(c not in free_cols for _, c in cells):
                continue
            if not _cells_clear(cells, occupied, moat=1):
                continue
            rows, cols = zip(*cells)
            grid[list(rows), list(cols)] = int(color)
            occupied[list(rows), list(cols)] = True
            break


def _validate_scene(grid: np.ndarray, target_counts: dict[int, int], instruction_count: int) -> bool:
    instructions = _find_instructions(grid)
    if len(instructions) != instruction_count:
        return False
    if {int(inst["target_color"]) for inst in instructions} != set(target_counts):
        return False
    cleaned = np.asarray(grid, dtype=int).copy()
    for inst in instructions:
        mr, mc = inst["marker"]  # type: ignore[misc]
        cleaned[mr, mc] = BLACK
        for ar, ac in inst["arrow_cells"]:  # type: ignore[misc]
            cleaned[int(ar), int(ac)] = BLACK
    for color, expected in target_counts.items():
        comps = _find_components(cleaned, include_colors={int(color)})
        if len(comps) != int(expected):
            return False
    try:
        out = main(grid)
    except Exception:
        return False
    if np.any(out == BLUE):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(("horizontal", "vertical"))
    instruction_count = random.randint(1, 3)
    target_colors = random.sample(list(OBJECT_COLORS), instruction_count)
    target_counts = {color: _sample_object_count() for color in target_colors}

    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    if axis == "horizontal":
        starts = _band_starts(h - 2, instruction_count)
    else:
        starts = _band_starts(w - 2, instruction_count)
    if starts is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[h - 1, :] = GREY
    grid[:, 0] = GREY
    grid[:, w - 1] = GREY
    occupied = grid != BLACK

    if axis == "horizontal":
        reserved_rows: set[int] = set()
        for band_start, color in zip(starts, target_colors):
            reserved_rows.update(range(band_start, band_start + GEN_BAND_SPAN))
            direction = random.choice(("left", "right"))
            if not _place_horizontal_instruction(
                grid,
                occupied,
                band_start,
                color,
                direction,
                target_counts[color],
            ):
                return None
        free_rows = {r for r in range(1, h - 1) if r not in reserved_rows}
        remaining = [c for c in OBJECT_COLORS if c not in target_colors]
        extra_colors = random.sample(remaining, random.randint(0, len(remaining)))
        _place_extras(grid, occupied, extra_colors, free_rows=free_rows)
    else:
        reserved_cols: set[int] = set()
        for band_start, color in zip(starts, target_colors):
            reserved_cols.update(range(band_start, band_start + GEN_BAND_SPAN))
            direction = random.choice(("up", "down"))
            if not _place_vertical_instruction(
                grid,
                occupied,
                band_start,
                color,
                direction,
                target_counts[color],
            ):
                return None
        free_cols = {c for c in range(1, w - 1) if c not in reserved_cols}
        remaining = [c for c in OBJECT_COLORS if c not in target_colors]
        extra_colors = random.sample(remaining, random.randint(0, len(remaining)))
        _place_extras(grid, occupied, extra_colors, free_cols=free_cols)

    if not _validate_scene(grid, target_counts, instruction_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
