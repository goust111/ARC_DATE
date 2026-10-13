from __future__ import annotations

import random
from collections import deque

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _non_bg_runs(row: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(row.tolist()):
        if int(value) != int(bg):
            if start is None:
                start = int(idx)
        elif start is not None:
            runs.append((start, int(idx) - 1))
            start = None
    if start is not None:
        runs.append((start, len(row) - 1))
    return runs


def _find_palette(grid: np.ndarray, bg: int) -> tuple[int, int, np.ndarray]:
    assert grid.shape[0] >= 2, "expected space above the palette row"
    last_row = grid[-1]
    candidates: list[tuple[int, int, int]] = []
    for c0, c1 in _non_bg_runs(last_row, bg):
        colors = [int(x) for x in last_row[c0 : c1 + 1].tolist()]
        if len(colors) < 2:
            continue
        if len(colors) != len(set(colors)):
            continue
        if not np.all(grid[-2, c0 : c1 + 1] == int(bg)):
            continue
        candidates.append((c1 - c0 + 1, c0, c1))

    assert len(candidates) == 1, "expected exactly one distinct-color palette strip on the bottom row"
    _length, c0, c1 = candidates[0]
    return int(c0), int(c1), grid[-1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _rectangle_components(grid: np.ndarray, bg: int, palette_c0: int, palette_c1: int) -> list[dict]:
    mask = grid != int(bg)
    mask[-1, palette_c0 : palette_c1 + 1] = False
    comps = _components4(mask)
    assert comps, "expected at least one rectangle component"

    objects: list[dict] = []
    source_colors: set[int] = set()
    for comp in comps:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "rectangle bbox must have an interior"

        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local_mask[r - r0, c - c0] = True

        assert bool(np.all(local_mask[0, :])), "top border must be closed"
        assert bool(np.all(local_mask[-1, :])), "bottom border must be closed"
        assert bool(np.all(local_mask[:, 0])), "left border must be closed"
        assert bool(np.all(local_mask[:, -1])), "right border must be closed"

        cell_colors = {int(grid[r, c]) for r, c in comp}
        assert len(cell_colors) == 1, "each rectangle must be monochrome in the input"
        source_colors |= cell_colors

        hole_count = int(np.count_nonzero(~local_mask[1:-1, 1:-1]))
        assert hole_count >= 1, "each rectangle must contain at least one background hole pixel"

        objects.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "hole_count": hole_count,
            }
        )

    assert len(source_colors) == 1, "all rectangles must share one source color"
    hole_counts = [obj["hole_count"] for obj in objects]
    assert len(hole_counts) == len(set(hole_counts)), "hole counts must be unique across rectangles"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed rectangle according to the number of black hole pixels inside it.

    Input structure:
    - The background is black.
    - Several separated monochrome rectangles are drawn with a one-pixel-thick closed border, and their interiors stay filled except for some black hole pixels.
    - A horizontal palette strip sits on the bottom row; its left-to-right single-pixel colors encode counts 1, 2, 3, 4, ...

    Rule:
    - Find the bottom palette strip and read its colors by position.
    - For each rectangle, count how many background pixels lie strictly inside its rectangular border.
    - Recolor every non-background pixel of that rectangle with the palette color at the matching 1-based position, leaving the hole pixels black.

    Key insight:
    - The rectangles are keyed by an internal quantity, not by their original color or position: the number of interior black holes is the index used to query the palette strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    palette_c0, palette_c1, palette = _find_palette(g, bg)
    objects = _rectangle_components(g, bg, palette_c0, palette_c1)

    out = g.copy()
    for obj in objects:
        hole_count = int(obj["hole_count"])
        assert hole_count <= len(palette), "hole count must fit within the palette length"
        new_color = int(palette[hole_count - 1])
        for r, c in obj["coords"]:
            out[r, c] = new_color
    return out


# --------------- generator ---------------

BG = 0
COLOR_POOL = tuple(range(1, 10))
OBJECT_COUNT_RANGE = (2, 4)
PALETTE_LEN_RANGE = (4, 7)
INNER_H_RANGE = (2, 6)
INNER_W_RANGE = (2, 7)
SIDE_PAD_RANGE = (1, 3)
TOP_PAD_RANGE = (1, 3)
INTER_OBJECT_GAP_RANGE = (1, 3)
INTER_ROW_GAP_RANGE = (1, 3)
PALETTE_GAP_RANGE = (1, 3)
GRID_SIDE_MAX = 30
GENERATION_ATTEMPTS = 300
OBJECT_ATTEMPTS = 200


def _feasible_inner_sizes(hole_count: int) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for inner_h in range(INNER_H_RANGE[0], INNER_H_RANGE[1] + 1):
        for inner_w in range(INNER_W_RANGE[0], INNER_W_RANGE[1] + 1):
            area = inner_h * inner_w
            checker_capacity = (area + 1) // 2
            if checker_capacity >= int(hole_count) and area > int(hole_count):
                sizes.append((inner_h, inner_w))
    return sizes


def _sample_holes(inner_h: int, inner_w: int, hole_count: int) -> list[tuple[int, int]]:
    parity_cells = [
        [(r, c) for r in range(inner_h) for c in range(inner_w) if (r + c) % 2 == parity]
        for parity in (0, 1)
    ]
    choices = [cells for cells in parity_cells if len(cells) >= int(hole_count)]
    cells = random.choice(choices)
    return [(int(r), int(c)) for r, c in random.sample(cells, int(hole_count))]


def _make_rectangle(source_color: int, hole_count: int) -> dict:
    sizes = _feasible_inner_sizes(hole_count)
    for _ in range(OBJECT_ATTEMPTS):
        inner_h, inner_w = random.choice(sizes)
        patch = np.full((inner_h + 2, inner_w + 2), int(source_color), dtype=int)
        holes = _sample_holes(inner_h, inner_w, hole_count)
        for hr, hc in holes:
            patch[hr + 1, hc + 1] = BG
        if len(_components4(patch != BG)) != 1:
            continue
        return {
            "array": patch,
            "hole_count": int(hole_count),
            "height": int(patch.shape[0]),
            "width": int(patch.shape[1]),
        }
    raise RuntimeError("failed to sample a connected rectangle with the requested hole count")


def _row_width(objs: list[dict], gaps: list[int]) -> int:
    width = sum(int(obj["width"]) for obj in objs)
    width += sum(int(g) for g in gaps)
    return int(width)


def _partition_rows(objects: list[dict]) -> list[list[dict]]:
    shuffled = objects[:]
    random.shuffle(shuffled)
    min_single_width = sum(int(obj["width"]) for obj in shuffled) + INTER_OBJECT_GAP_RANGE[0] * (len(shuffled) - 1)
    single_row_feasible = min_single_width + 2 * SIDE_PAD_RANGE[1] <= GRID_SIDE_MAX

    if single_row_feasible and len(shuffled) <= 3 and random.random() < 0.45:
        return [shuffled]

    rows: list[list[dict]] = [[], []]
    row_loads = [0, 0]
    for obj in sorted(shuffled, key=lambda item: int(item["width"]), reverse=True):
        idx = 0 if row_loads[0] <= row_loads[1] else 1
        rows[idx].append(obj)
        row_loads[idx] += int(obj["width"])

    if not rows[0]:
        rows[0].append(rows[1].pop())
    if not rows[1]:
        rows[1].append(rows[0].pop())
    for row in rows:
        random.shuffle(row)
    return rows


def _compose_scene(objects: list[dict], palette_colors: list[int]) -> np.ndarray | None:
    rows = _partition_rows(objects)
    row_heights = [max(int(obj["height"]) for obj in row) for row in rows]
    row_gaps = [random.randint(*INTER_OBJECT_GAP_RANGE) for row in rows for _ in range(max(0, len(row) - 1))]

    row_gap_lists: list[list[int]] = []
    idx = 0
    for row in rows:
        count = max(0, len(row) - 1)
        row_gap_lists.append(row_gaps[idx : idx + count])
        idx += count

    content_width = max(
        max(_row_width(row, gaps) for row, gaps in zip(rows, row_gap_lists, strict=False)),
        len(palette_colors),
    )
    left_pad = random.randint(*SIDE_PAD_RANGE)
    right_pad = random.randint(*SIDE_PAD_RANGE)
    min_width = content_width + left_pad + right_pad
    if min_width > GRID_SIDE_MAX:
        return None
    grid_w = min_width + random.randint(0, min(4, GRID_SIDE_MAX - min_width))

    top_pad = random.randint(*TOP_PAD_RANGE)
    palette_gap = random.randint(*PALETTE_GAP_RANGE)
    inter_row_gaps = [random.randint(*INTER_ROW_GAP_RANGE) for _ in range(max(0, len(rows) - 1))]
    min_height = top_pad + sum(row_heights) + sum(inter_row_gaps) + palette_gap + 1
    if min_height > GRID_SIDE_MAX:
        return None
    grid_h = min_height + random.randint(0, min(4, GRID_SIDE_MAX - min_height))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    current_top = top_pad
    for row_idx, (row, row_h, gaps) in enumerate(zip(rows, row_heights, row_gap_lists, strict=False)):
        row_width = _row_width(row, gaps)
        row_left = random.randint(left_pad, grid_w - right_pad - row_width)
        cursor = row_left
        for obj_idx, obj in enumerate(row):
            obj_h = int(obj["height"])
            vertical_offset = random.randint(0, row_h - obj_h)
            r0 = current_top + vertical_offset
            c0 = cursor
            patch = obj["array"]
            grid[r0 : r0 + obj_h, c0 : c0 + int(obj["width"])] = patch
            if obj_idx < len(gaps):
                cursor += int(obj["width"]) + int(gaps[obj_idx])
        current_top += row_h
        if row_idx < len(inter_row_gaps):
            current_top += int(inter_row_gaps[row_idx])

    palette_left = random.randint(left_pad, grid_w - right_pad - len(palette_colors))
    grid[-1, palette_left : palette_left + len(palette_colors)] = np.array(palette_colors, dtype=int)

    if not np.all(grid[-2, palette_left : palette_left + len(palette_colors)] == BG):
        return None
    return grid


def _validate_generated(grid: np.ndarray, expected_object_count: int, palette_len: int) -> None:
    assert grid.ndim == 2
    assert max(grid.shape) <= GRID_SIDE_MAX
    bg = _background_color(grid)
    c0, c1, palette = _find_palette(grid, bg)
    assert len(palette) == int(palette_len)
    objects = _rectangle_components(grid, bg, c0, c1)
    assert len(objects) == int(expected_object_count)
    counts = sorted(int(obj["hole_count"]) for obj in objects)
    assert counts == sorted(set(counts))
    assert counts[-1] <= len(palette)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        object_count = random.randint(*OBJECT_COUNT_RANGE)
        palette_len = random.randint(max(PALETTE_LEN_RANGE[0], object_count), PALETTE_LEN_RANGE[1])
        hole_counts = random.sample(range(1, palette_len + 1), object_count)

        source_color = random.choice(COLOR_POOL)
        remaining_colors = [color for color in COLOR_POOL if color != source_color]
        palette_colors = random.sample(remaining_colors, palette_len)

        objects = [_make_rectangle(source_color, hole_count) for hole_count in hole_counts]
        grid = _compose_scene(objects, palette_colors)
        if grid is None:
            continue

        _validate_generated(grid, object_count, palette_len)
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
