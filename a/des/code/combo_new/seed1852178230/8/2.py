from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400


def _make_shape(direction: str, cells: tuple[tuple[int, int], ...], tail: tuple[int, int]) -> dict:
    arr = np.asarray(cells, dtype=int)
    return {
        "direction": direction,
        "cells": arr,
        "tail": np.asarray(tail, dtype=int),
        "height": int(arr[:, 0].max()) + 1,
        "width": int(arr[:, 1].max()) + 1,
    }


SHAPES = (
    _make_shape("up", ((0, 0), (1, 0), (1, 1)), (0, 0)),
    _make_shape("up", ((0, 1), (1, 0), (1, 1)), (0, 1)),
    _make_shape("up", ((0, 1), (1, 0), (1, 1), (1, 2)), (0, 1)),
    _make_shape("up", ((0, 0), (1, 0), (2, 0), (2, 1)), (0, 0)),
    _make_shape("up", ((0, 1), (1, 1), (2, 0), (2, 1)), (0, 1)),
    _make_shape("down", ((0, 0), (0, 1), (1, 0)), (1, 0)),
    _make_shape("down", ((0, 0), (0, 1), (1, 1)), (1, 1)),
    _make_shape("down", ((0, 0), (0, 1), (0, 2), (1, 1)), (1, 1)),
    _make_shape("down", ((0, 0), (0, 1), (1, 1), (2, 1)), (2, 1)),
    _make_shape("down", ((0, 1), (0, 2), (1, 1), (2, 1)), (2, 1)),
)
SHAPES_BY_DIR = {
    direction: [shape for shape in SHAPES if shape["direction"] == direction]
    for direction in ("up", "down")
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _row_uniform_color(row: np.ndarray) -> int | None:
    return int(row[0]) if bool(np.all(row == row[0])) else None


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    bg = _most_common_color(g)
    row_colors = [_row_uniform_color(g[r]) for r in range(h)]

    top_color = row_colors[0]
    bottom_color = row_colors[-1]
    assert top_color is not None and top_color != bg
    assert bottom_color is not None and bottom_color != bg and bottom_color != top_color

    top_end = 0
    while top_end + 1 < h and row_colors[top_end + 1] == top_color:
        top_end += 1

    bottom_start = h - 1
    while bottom_start - 1 >= 0 and row_colors[bottom_start - 1] == bottom_color:
        bottom_start -= 1

    grey_rows: list[int] = []
    grey_color: int | None = None
    for r in range(top_end + 1, bottom_start):
        color = row_colors[r]
        if color is None or color == bg:
            continue
        grey_rows.append(r)
        if grey_color is None:
            grey_color = color
        else:
            assert color == grey_color

    assert grey_rows
    assert top_end < grey_rows[0] < bottom_start
    assert grey_color is not None and grey_color not in {bg, top_color, bottom_color}
    return {
        "bg": bg,
        "top_color": int(top_color),
        "bottom_color": int(bottom_color),
        "grey_color": int(grey_color),
        "top_end": int(top_end),
        "bottom_start": int(bottom_start),
        "grey_rows": grey_rows,
    }


def _find_objects(grid: np.ndarray, structure: dict) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocked_rows = set(range(structure["top_end"] + 1))
    blocked_rows.update(structure["grey_rows"])
    blocked_rows.update(range(structure["bottom_start"], h))
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or r in blocked_rows or int(g[r, c]) == structure["bg"]:
                visited[r, c] = visited[r, c] or (r in blocked_rows)
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or nr in blocked_rows:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            objects.append({"color": color, "coords": arr, "bbox": _bbox(arr)})
    return objects


def _tail_info(coords: np.ndarray) -> tuple[tuple[int, int], str]:
    coord_set = {tuple(map(int, cell)) for cell in coords}
    vertical_leaves: list[tuple[tuple[int, int], str]] = []
    for r, c in coord_set:
        neighbors = []
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in coord_set:
                neighbors.append((nr, nc))
        if len(neighbors) != 1:
            continue
        nr, nc = neighbors[0]
        if nc != c:
            continue
        direction = "up" if r < nr else "down"
        vertical_leaves.append(((r, c), direction))

    assert len(vertical_leaves) == 1, "each object must have one vertical tail pixel"
    return vertical_leaves[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each small object vertically according to its tail and leave a colored trail through the separator rows.

    Input structure:
    - Black background with several full-width grey separator rows that split the grid into horizontal bands.
    - The top band and bottom band are solid edge bars of two distinct colors.
    - The middle bands contain small 4-connected monochrome objects whose colors match one of the edge-bar colors and whose unique vertical leaf pixel acts as a tail.

    Rule:
    - For each object, find the tail pixel: the only degree-1 pixel whose lone same-color 4-neighbor lies directly above or below it.
    - If the tail sits above the rest of the object, slide the whole object upward until its bbox top is on the row just below the top band; if the tail sits below, slide it downward until its bbox bottom is on the row just above the bottom band.
    - Remove the source object, draw the translated object at its destination, and paint the full vertical segment between the old and new tail positions in the object's color.
    - Wherever that segment crosses a grey separator row, the separator pixel at that column becomes the object's color.

    Key insight:
    - The motion is encoded by the object's tail geometry, not by the object color; the separator rows simply record the crossings as colored markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    structure = _extract_structure(g)
    top_target = structure["top_end"] + 1
    bottom_target = structure["bottom_start"] - 1
    out = g.copy()
    objects = _find_objects(g, structure)

    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = structure["bg"]

    for obj in objects:
        coords = obj["coords"]
        r0, _, r1, _ = obj["bbox"]
        tail, direction = _tail_info(coords)
        dr = top_target - r0 if direction == "up" else bottom_target - r1
        shifted = coords + np.array([dr, 0], dtype=int)
        out[shifted[:, 0], shifted[:, 1]] = obj["color"]

        old_tail_r, old_tail_c = tail
        new_tail_r = old_tail_r + dr
        for r in range(min(old_tail_r, new_tail_r), max(old_tail_r, new_tail_r) + 1):
            out[r, old_tail_c] = obj["color"]

    return out


def _build_band_layout(
    top_h: int, middle_heights: list[int], bottom_h: int
) -> tuple[int, int, int, list[int], list[tuple[int, int]]]:
    grey_rows: list[int] = []
    middle_bands: list[tuple[int, int]] = []
    row = top_h
    top_end = top_h - 1
    for band_h in middle_heights:
        grey_rows.append(row)
        row += 1
        middle_bands.append((row, row + band_h - 1))
        row += band_h
    grey_rows.append(row)
    bottom_start = row + 1
    height = bottom_start + bottom_h
    return height, top_end, bottom_start, grey_rows, middle_bands


def _plan_columns(rng: np.random.Generator, specs: list[dict], width: int) -> bool:
    min_width = sum(spec["shape"]["width"] for spec in specs) + max(0, len(specs) - 1)
    if min_width > width:
        return False
    slack = width - min_width
    extras = rng.multinomial(slack, np.full(len(specs) + 1, 1.0 / (len(specs) + 1)))
    col = int(extras[0])
    for i, spec in enumerate(specs):
        spec["col"] = col
        col += int(spec["shape"]["width"])
        if i < len(specs) - 1:
            col += 1 + int(extras[i + 1])
    return True


def _paint_shape(grid: np.ndarray, cells: np.ndarray, color: int, row0: int, col0: int) -> None:
    coords = cells + np.array([row0, col0], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    top_color, bottom_color = map(int, rng.choice(PALETTE, size=2, replace=False))
    middle_count = int(rng.integers(1, 4))
    top_h = int(rng.integers(1, 4))
    bottom_h = int(rng.integers(1, 4))
    middle_heights = [int(x) for x in rng.integers(3, 6, size=middle_count)]
    height, top_end, bottom_start, grey_rows, middle_bands = _build_band_layout(
        top_h, middle_heights, bottom_h
    )
    if height > 30:
        return None

    total_objects = int(rng.integers(4, 6))
    up_count = total_objects // 2
    if total_objects % 2 == 1:
        up_count += int(rng.integers(0, 2))
    down_count = total_objects - up_count
    if up_count == 0 or down_count == 0:
        return None

    specs: list[dict] = []
    for direction, count in (("up", up_count), ("down", down_count)):
        pool = SHAPES_BY_DIR[direction]
        for _ in range(count):
            shape = pool[int(rng.integers(0, len(pool)))]
            specs.append({"direction": direction, "shape": shape})
    rng.shuffle(specs)

    min_width = sum(spec["shape"]["width"] for spec in specs) + len(specs) - 1
    width = min(30, min_width + int(rng.integers(1, 8)))
    if width < min_width or width > 30:
        return None
    if not _plan_columns(rng, specs, width):
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:top_h, :] = top_color
    grid[bottom_start:, :] = bottom_color
    for row in grey_rows:
        grid[row, :] = GREY

    top_target = top_end + 1
    bottom_target = bottom_start - 1

    for spec in specs:
        direction = spec["direction"]
        shape = spec["shape"]
        band_candidates = [band for band in middle_bands if band[1] - band[0] + 1 >= shape["height"]]
        if not band_candidates:
            return None
        band_index = int(rng.integers(0, len(band_candidates)))
        band_r0, band_r1 = band_candidates[band_index]
        max_offset = band_r1 - band_r0 + 1 - int(shape["height"])
        row0 = band_r0 + int(rng.integers(0, max_offset + 1))

        spec["row"] = row0
        spec["color"] = top_color if direction == "up" else bottom_color
        spec["tail_col"] = int(spec["col"] + shape["tail"][1])
        spec["tail_row"] = int(row0 + shape["tail"][0])
        if direction == "up":
            spec["final_row"] = top_target
        else:
            spec["final_row"] = bottom_target - int(shape["height"]) + 1

    for spec in specs:
        _paint_shape(grid, spec["shape"]["cells"], spec["color"], spec["row"], spec["col"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
