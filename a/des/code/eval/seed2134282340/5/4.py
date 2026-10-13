from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
GENERATE_RETRIES = 400
PLACE_RETRIES = 250

ARROW_PATTERNS = {
    "right": frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}),
    "left": frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}),
    "down": frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}),
    "up": frozenset({(0, 1), (1, 1), (2, 1), (0, 0), (0, 2)}),
}

UNIT_VECTORS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _direction_from_arrow(coords: np.ndarray) -> str:
    normalized = frozenset(_normalize_cells([tuple(map(int, cell)) for cell in coords]))
    for direction, pattern in ARROW_PATTERNS.items():
        if normalized == pattern:
            return direction
    raise AssertionError("invalid arrow component")


def _ruler_info(grid: np.ndarray) -> tuple[int, list[int]]:
    rows = np.flatnonzero(np.any(grid != BG, axis=1))
    assert rows.size > 0, "expected at least one non-background row"
    ruler_row = int(rows[-1])
    cols = np.flatnonzero(grid[ruler_row] != BG)
    assert cols.size > 0, "expected ruler pixels"
    assert not np.any(grid[ruler_row, cols] == MAROON), "ruler cannot use maroon"
    colors = [int(grid[ruler_row, c]) for c in cols]
    return ruler_row, colors


def _touches_object(arrow_coords: np.ndarray, object_cells: set[tuple[int, int]]) -> bool:
    for r, c in arrow_coords:
        for nr, nc in _neighbors4(int(r), int(c)):
            if (nr, nc) in object_cells:
                return True
    return False


def _arrow_relative_cells(direction: str, attach: int, shape_h: int, shape_w: int) -> set[tuple[int, int]]:
    if direction == "right":
        return {
            (attach, shape_w),
            (attach, shape_w + 1),
            (attach, shape_w + 2),
            (attach - 1, shape_w + 2),
            (attach + 1, shape_w + 2),
        }
    if direction == "left":
        return {
            (attach, -1),
            (attach, -2),
            (attach, -3),
            (attach - 1, -3),
            (attach + 1, -3),
        }
    if direction == "down":
        return {
            (shape_h, attach),
            (shape_h + 1, attach),
            (shape_h + 2, attach),
            (shape_h + 2, attach - 1),
            (shape_h + 2, attach + 1),
        }
    assert direction == "up"
    return {
        (-1, attach),
        (-2, attach),
        (-3, attach),
        (-3, attach - 1),
        (-3, attach + 1),
    }


def _attachment_values(shape_cells: tuple[tuple[int, int], ...], direction: str) -> list[int]:
    height = max(r for r, _ in shape_cells) + 1
    width = max(c for _, c in shape_cells) + 1
    if direction == "right":
        return sorted({r for r, c in shape_cells if c == width - 1})
    if direction == "left":
        return sorted({r for r, c in shape_cells if c == 0})
    if direction == "down":
        return sorted({c for r, c in shape_cells if r == height - 1})
    return sorted({c for r, c in shape_cells if r == 0})


def _direction_step(direction: str, shape_h: int, shape_w: int) -> tuple[int, int]:
    unit_r, unit_c = UNIT_VECTORS[direction]
    step = shape_w if direction in {"left", "right"} else shape_h
    return unit_r * step, unit_c * step


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    ruler_row, ruler_colors = _ruler_info(g)

    arrow_mask = g == MAROON
    arrow_mask[ruler_row, :] = False
    arrows = []
    for coords in _components_from_mask(arrow_mask):
        arrows.append({"coords": coords, "direction": _direction_from_arrow(coords)})
    assert arrows, "expected at least one arrow"

    object_mask = (g != BG) & (g != MAROON)
    object_mask[ruler_row, :] = False
    objects = []
    for coords in _components_from_mask(object_mask):
        objects.append(
            {
                "coords": coords,
                "cells": {tuple(map(int, cell)) for cell in coords},
                "bbox": _bbox(coords),
            }
        )
    assert len(objects) == len(arrows), "every object must have exactly one arrow"

    assignments: dict[int, dict[str, object]] = {}
    for arrow in arrows:
        touched = [idx for idx, obj in enumerate(objects) if _touches_object(arrow["coords"], obj["cells"])]
        assert len(touched) == 1, "arrow/object pairing must be unambiguous"
        obj_idx = touched[0]
        assert obj_idx not in assignments, "each object must have only one arrow"
        assignments[obj_idx] = arrow
    assert len(assignments) == len(objects), "every object must be matched"

    out = g.copy()
    for arrow in arrows:
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for obj_idx, obj in enumerate(objects):
        direction = str(assignments[obj_idx]["direction"])
        r0, c0, r1, c1 = obj["bbox"]
        step_dr, step_dc = _direction_step(direction, r1 - r0 + 1, c1 - c0 + 1)
        coords = obj["coords"]
        for copy_idx, color in enumerate(ruler_colors, start=1):
            rr = coords[:, 0] + step_dr * copy_idx
            cc = coords[:, 1] + step_dc * copy_idx
            in_bounds = (0 <= rr) & (rr < height) & (0 <= cc) & (cc < width)
            out[rr[in_bounds], cc[in_bounds]] = int(color)

    return out


def _random_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(100):
        box_h = random.randint(2, 5)
        box_w = random.randint(2, 5)
        target_area = random.randint(3, min(box_h * box_w, 8))
        cells = {(random.randrange(box_h), random.randrange(box_w))}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        if random.random() < 0.5:
            max_r = max(r for r, _ in cells)
            cells = {(max_r - r, c) for r, c in cells}
        if random.random() < 0.5:
            max_c = max(c for _, c in cells)
            cells = {(r, max_c - c) for r, c in cells}
        if random.random() < 0.5:
            cells = {(c, r) for r, c in cells}

        return _normalize_cells(cells)

    raise RuntimeError("failed to sample a shape")


def _paint_shape(shape_cells: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], int]:
    if len(shape_cells) >= 4 and random.random() < 0.5:
        primary, secondary = random.sample(PALETTE, 2)
        accent_count = random.randint(1, len(shape_cells) - 1)
        accent_cells = set(random.sample(list(shape_cells), accent_count))
        return {cell: (secondary if cell in accent_cells else primary) for cell in shape_cells}
    color = random.choice(PALETTE)
    return {cell: color for cell in shape_cells}


def _sample_ruler_colors(count: int) -> list[int]:
    colors = [random.choice(PALETTE) for _ in range(count)]
    if count > 1 and len(set(colors)) == 1:
        idx = random.randrange(count)
        alternatives = [c for c in PALETTE if c != colors[idx]]
        colors[idx] = random.choice(alternatives)
    return colors


def _clear_of(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], radius: int = 0) -> bool:
    if radius <= 0:
        return not bool(cells & occupied)
    for r, c in cells:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _sample_source_position(
    direction: str,
    clip_mode: bool,
    attach: int,
    shape_h: int,
    shape_w: int,
    step: int,
    copy_count: int,
    scene_h: int,
    width: int,
) -> tuple[int, int] | None:
    if direction in {"right", "left"}:
        row_lo = max(0, 1 - attach)
        row_hi = min(scene_h - shape_h, scene_h - 2 - attach)
        if row_lo > row_hi:
            return None
        src_r = random.randint(row_lo, row_hi)

        if direction == "right":
            arrow_hi = width - shape_w - 3
            if clip_mode:
                col_lo = max(0, width - shape_w - copy_count * step + 1)
                col_hi = arrow_hi
            else:
                col_lo = 0
                col_hi = min(arrow_hi, width - shape_w - copy_count * step)
        else:
            if clip_mode:
                col_lo = 3
                col_hi = min(width - shape_w, copy_count * step - 1)
            else:
                col_lo = max(3, copy_count * step)
                col_hi = width - shape_w
        if col_lo > col_hi:
            return None
        src_c = random.randint(col_lo, col_hi)
        return src_r, src_c

    col_lo = max(0, 1 - attach)
    col_hi = min(width - shape_w, width - 2 - attach)
    if col_lo > col_hi:
        return None
    src_c = random.randint(col_lo, col_hi)

    if direction == "down":
        arrow_hi = scene_h - shape_h - 3
        if clip_mode:
            row_lo = max(0, scene_h - shape_h - copy_count * step + 1)
            row_hi = arrow_hi
        else:
            row_lo = 0
            row_hi = min(arrow_hi, scene_h - shape_h - copy_count * step)
    else:
        if clip_mode:
            row_lo = 3
            row_hi = min(scene_h - shape_h, copy_count * step - 1)
        else:
            row_lo = max(3, copy_count * step)
            row_hi = scene_h - shape_h
    if row_lo > row_hi:
        return None
    src_r = random.randint(row_lo, row_hi)
    return src_r, src_c


def _place_object(
    spec: dict[str, object],
    scene_h: int,
    width: int,
    copy_count: int,
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> dict[str, object] | None:
    shape_cells = tuple(spec["shape_cells"])
    colors = dict(spec["colors"])
    direction = str(spec["direction"])
    clip_mode = bool(spec["clip"])
    shape_h = max(r for r, _ in shape_cells) + 1
    shape_w = max(c for _, c in shape_cells) + 1
    step = shape_w if direction in {"left", "right"} else shape_h
    attach_values = _attachment_values(shape_cells, direction)

    for _ in range(PLACE_RETRIES):
        attach = random.choice(attach_values)
        source_pos = _sample_source_position(
            direction,
            clip_mode,
            attach,
            shape_h,
            shape_w,
            step,
            copy_count,
            scene_h,
            width,
        )
        if source_pos is None:
            continue
        src_r, src_c = source_pos

        source_cells = {(src_r + r, src_c + c) for r, c in shape_cells}
        arrow_cells = {(src_r + r, src_c + c) for r, c in _arrow_relative_cells(direction, attach, shape_h, shape_w)}
        if not all(0 <= r < scene_h and 0 <= c < width for r, c in arrow_cells):
            continue
        group_cells = source_cells | arrow_cells
        if not _clear_of(group_cells, input_occupied, radius=1):
            continue

        step_dr, step_dc = _direction_step(direction, shape_h, shape_w)
        all_output_cells = set(source_cells)
        copy_cells: list[set[tuple[int, int]]] = []
        saw_partial = False
        valid = True
        for copy_idx in range(1, copy_count + 1):
            shifted = {
                (src_r + r + step_dr * copy_idx, src_c + c + step_dc * copy_idx)
                for r, c in shape_cells
            }
            visible = {(r, c) for r, c in shifted if 0 <= r < scene_h and 0 <= c < width}
            if copy_idx == 1 and not visible:
                valid = False
                break
            if clip_mode:
                if visible and len(visible) < len(shape_cells):
                    saw_partial = True
            else:
                if len(visible) != len(shape_cells):
                    valid = False
                    break
            if visible & output_occupied or visible & all_output_cells:
                valid = False
                break
            all_output_cells |= visible
            copy_cells.append(visible)
        if not valid:
            continue
        if clip_mode and not saw_partial:
            continue
        if arrow_cells & output_occupied:
            continue

        return {
            "shape_cells": shape_cells,
            "colors": colors,
            "direction": direction,
            "source_pos": (src_r, src_c),
            "source_cells": source_cells,
            "arrow_cells": arrow_cells,
            "copy_cells": copy_cells,
        }

    return None


def _sample_scene_specs() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(16, 24)
    width = random.randint(16, 24)
    scene_h = height - 2

    copy_count = random.randint(2, 5)
    ruler_colors = _sample_ruler_colors(copy_count)
    ruler_start = random.randint(1, width - copy_count - 1)

    object_count = random.randint(1, 4)
    if object_count <= 4 and random.random() < 0.7:
        directions = random.sample(list(ARROW_PATTERNS), k=object_count)
    else:
        directions = [random.choice(list(ARROW_PATTERNS)) for _ in range(object_count)]

    clip_index = random.randrange(object_count) if random.random() < 0.5 else None
    specs = []
    for idx in range(object_count):
        shape_cells = _random_shape()
        specs.append(
            {
                "shape_cells": shape_cells,
                "colors": _paint_shape(shape_cells),
                "direction": directions[idx],
                "clip": idx == clip_index,
            }
        )

    specs.sort(
        key=lambda spec: (
            0 if bool(spec["clip"]) else 1,
            -(len(tuple(spec["shape_cells"]))),
        )
    )

    input_occupied: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    for spec in specs:
        placed = _place_object(spec, scene_h, width, copy_count, input_occupied, output_occupied)
        if placed is None:
            return None
        placements.append(placed)
        input_occupied |= placed["source_cells"] | placed["arrow_cells"]
        output_occupied |= placed["source_cells"]
        for cells in placed["copy_cells"]:
            output_occupied |= cells

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    inp[height - 1, ruler_start : ruler_start + copy_count] = np.asarray(ruler_colors, dtype=int)
    out[height - 1, ruler_start : ruler_start + copy_count] = np.asarray(ruler_colors, dtype=int)

    for placed in placements:
        src_r, src_c = placed["source_pos"]
        colors = dict(placed["colors"])
        for rel_cell, color in colors.items():
            r = src_r + rel_cell[0]
            c = src_c + rel_cell[1]
            inp[r, c] = int(color)
            out[r, c] = int(color)
        for r, c in placed["arrow_cells"]:
            inp[r, c] = MAROON
        for color, cells in zip(ruler_colors, placed["copy_cells"]):
            for r, c in cells:
                out[r, c] = int(color)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        sample = _sample_scene_specs()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
