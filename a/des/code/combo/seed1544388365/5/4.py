from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
AXIS_BY_SIDE = {
    "top": "horizontal",
    "bottom": "horizontal",
    "left": "vertical",
    "right": "vertical",
}
GENERATION_RETRIES = 400


def _normalize(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cells = tuple(cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(cells: Iterable[tuple[int, int]]) -> tuple[int, int]:
    cells = tuple(cells)
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _reflect_shape(
    cells: Iterable[tuple[int, int]], axis: str
) -> tuple[tuple[int, int], ...]:
    cells = tuple(cells)
    h, w = _shape_dims(cells)
    if axis == "horizontal":
        flipped = ((h - 1 - r, c) for r, c in cells)
    else:
        assert axis == "vertical"
        flipped = ((r, w - 1 - c) for r, c in cells)
    return _normalize(flipped)


def _reflect_component(
    coords: Iterable[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    axis: str,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if axis == "horizontal":
        return {(r0 + r1 - r, c) for r, c in coords}
    assert axis == "vertical"
    return {(r, c0 + c1 - c) for r, c in coords}


def _edge_foreground(g: np.ndarray) -> set[tuple[int, int]]:
    h, w = g.shape
    cells: set[tuple[int, int]] = set()
    for c in range(w):
        if int(g[0, c]) != 0:
            cells.add((0, c))
        if int(g[h - 1, c]) != 0:
            cells.add((h - 1, c))
    for r in range(1, h - 1):
        if int(g[r, 0]) != 0:
            cells.add((r, 0))
        if int(g[r, w - 1]) != 0:
            cells.add((r, w - 1))
    return cells


def _extract_edge_pairs(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    pairs: list[dict[str, object]] = []
    covered: set[tuple[int, int]] = set()

    def register(side: str, cells: tuple[tuple[int, int], tuple[int, int]]) -> None:
        colors = tuple(int(g[r, c]) for r, c in cells)
        assert colors[0] != 0 and colors[1] != 0
        assert colors[0] != colors[1], "marker pair must use two different colors"
        cell_set = set(cells)
        assert not (covered & cell_set), "edge marker cells may belong to only one pair"
        covered.update(cell_set)
        pairs.append(
            {
                "side": side,
                "axis": AXIS_BY_SIDE[side],
                "cells": cells,
                "colors": colors,
            }
        )

    for c in range(w - 1):
        if (
            int(g[0, c]) != 0
            and int(g[0, c + 1]) != 0
            and (c == 0 or int(g[0, c - 1]) == 0)
            and (c + 2 == w or int(g[0, c + 2]) == 0)
        ):
            register("top", ((0, c), (0, c + 1)))
        if (
            int(g[h - 1, c]) != 0
            and int(g[h - 1, c + 1]) != 0
            and (c == 0 or int(g[h - 1, c - 1]) == 0)
            and (c + 2 == w or int(g[h - 1, c + 2]) == 0)
        ):
            register("bottom", ((h - 1, c), (h - 1, c + 1)))

    for r in range(h - 1):
        if (
            int(g[r, 0]) != 0
            and int(g[r + 1, 0]) != 0
            and (r == 0 or int(g[r - 1, 0]) == 0)
            and (r + 2 == h or int(g[r + 2, 0]) == 0)
        ):
            register("left", ((r, 0), (r + 1, 0)))
        if (
            int(g[r, w - 1]) != 0
            and int(g[r + 1, w - 1]) != 0
            and (r == 0 or int(g[r - 1, w - 1]) == 0)
            and (r + 2 == h or int(g[r + 2, w - 1]) == 0)
        ):
            register("right", ((r, w - 1), (r + 1, w - 1)))

    edge_fg = _edge_foreground(g)
    assert pairs, "expected at least one boundary marker pair"
    assert covered == edge_fg, "every edge foreground cell must belong to a length-2 pair"
    return pairs


def _find_internal_components(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color == 0 or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        1 <= nr < h - 1
                        and 1 <= nc < w - 1
                        and not visited[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    assert comps, "expected at least one internal object"
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each internal object along the axis encoded by its same-color edge marker,
    then recolor it with the adjacent marker color from that same boundary pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5
    assert int(g.min()) >= 0

    pairs = _extract_edge_pairs(g)
    markers: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        cells = pair["cells"]
        colors = pair["colors"]
        axis = pair["axis"]
        assert isinstance(cells, tuple) and isinstance(colors, tuple)
        for i in range(2):
            markers.append(
                {
                    "pair_index": pair_index,
                    "coord": cells[i],
                    "color": int(colors[i]),
                    "axis": axis,
                    "mate_color": int(colors[1 - i]),
                }
            )

    comps = _find_internal_components(g)
    out = g.copy()

    old_cells: set[tuple[int, int]] = set()
    paints: list[tuple[tuple[int, int], int]] = []

    for comp in comps:
        color = int(comp["color"])
        coords = tuple(comp["coords"])
        bbox = comp["bbox"]

        candidates: list[tuple[int, dict[str, object]]] = []
        for marker in markers:
            if int(marker["color"]) != color:
                continue
            mr, mc = marker["coord"]
            dist = min(abs(r - mr) + abs(c - mc) for r, c in coords)
            candidates.append((dist, marker))

        assert candidates, f"object color {color} has no same-color boundary marker"
        best_dist = min(dist for dist, _marker in candidates)
        best = [marker for dist, marker in candidates if dist == best_dist]
        assert len(best) == 1, f"object color {color} has ambiguous nearest marker"
        marker = best[0]

        axis = str(marker["axis"])
        replace_color = int(marker["mate_color"])
        new_cells = _reflect_component(coords, bbox, axis)

        old_cells.update(coords)
        for r, c in new_cells:
            assert 1 <= r < h - 1 and 1 <= c < w - 1
            paints.append(((r, c), replace_color))

    for r, c in old_cells:
        out[r, c] = 0
    for (r, c), color in paints:
        assert int(out[r, c]) == 0, "transformed objects must stay disjoint"
        out[r, c] = color

    return out


def _random_shape(rng: random.Random, axis: str) -> tuple[tuple[int, int], ...] | None:
    for _ in range(250):
        target_area = rng.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if -3 <= nr <= 3 and -3 <= nc <= 3 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(rng.choice(tuple(frontier)))

        shape = _normalize(cells)
        h, w = _shape_dims(shape)
        if h > 4 or w > 4:
            continue
        if axis == "horizontal" and h < 2:
            continue
        if axis == "vertical" and w < 2:
            continue
        if _reflect_shape(shape, "horizontal") == shape:
            continue
        if _reflect_shape(shape, "vertical") == shape:
            continue
        if _reflect_shape(shape, axis) == shape:
            continue
        return shape
    return None


def _pair_candidates(grid: np.ndarray, side: str) -> list[int]:
    h, w = grid.shape
    starts: list[int] = []
    if side == "top":
        row = 0
        for c in range(1, w - 2):
            if (
                int(grid[row, c]) == 0
                and int(grid[row, c + 1]) == 0
                and int(grid[row, c - 1]) == 0
                and (c + 2 == w or int(grid[row, c + 2]) == 0)
            ):
                starts.append(c)
    elif side == "bottom":
        row = h - 1
        for c in range(1, w - 2):
            if (
                int(grid[row, c]) == 0
                and int(grid[row, c + 1]) == 0
                and int(grid[row, c - 1]) == 0
                and (c + 2 == w or int(grid[row, c + 2]) == 0)
            ):
                starts.append(c)
    elif side == "left":
        col = 0
        for r in range(1, h - 2):
            if (
                int(grid[r, col]) == 0
                and int(grid[r + 1, col]) == 0
                and int(grid[r - 1, col]) == 0
                and (r + 2 == h or int(grid[r + 2, col]) == 0)
            ):
                starts.append(r)
    else:
        assert side == "right"
        col = w - 1
        for r in range(1, h - 2):
            if (
                int(grid[r, col]) == 0
                and int(grid[r + 1, col]) == 0
                and int(grid[r - 1, col]) == 0
                and (r + 2 == h or int(grid[r + 2, col]) == 0)
            ):
                starts.append(r)
    return starts


def _place_pair(
    grid: np.ndarray,
    rng: random.Random,
    side: str,
    select_color: int,
    replace_color: int,
) -> tuple[int, int] | None:
    starts = _pair_candidates(grid, side)
    if not starts:
        return None

    order = [select_color, replace_color]
    if rng.random() < 0.5:
        order.reverse()

    if side == "top":
        row = 0
        c = rng.choice(starts)
        grid[row, c] = order[0]
        grid[row, c + 1] = order[1]
        return (row, c) if order[0] == select_color else (row, c + 1)
    if side == "bottom":
        row = grid.shape[0] - 1
        c = rng.choice(starts)
        grid[row, c] = order[0]
        grid[row, c + 1] = order[1]
        return (row, c) if order[0] == select_color else (row, c + 1)
    if side == "left":
        col = 0
        r = rng.choice(starts)
        grid[r, col] = order[0]
        grid[r + 1, col] = order[1]
        return (r, col) if order[0] == select_color else (r + 1, col)

    assert side == "right"
    col = grid.shape[1] - 1
    r = rng.choice(starts)
    grid[r, col] = order[0]
    grid[r + 1, col] = order[1]
    return (r, col) if order[0] == select_color else (r + 1, col)


def _can_place(
    occupied: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    top: int,
    left: int,
) -> bool:
    coords = [(top + r, left + c) for r, c in cells]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = max(0, min(rows) - 1)
    r1 = min(occupied.shape[0] - 1, max(rows) + 1)
    c0 = max(0, min(cols) - 1)
    c1 = min(occupied.shape[1] - 1, max(cols) + 1)
    return not occupied[r0 : r1 + 1, c0 : c1 + 1].any()


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    cells: tuple[tuple[int, int], ...],
    color: int,
    anchor: tuple[int, int],
) -> dict[str, object] | None:
    h, w = _shape_dims(cells)
    row_starts = list(range(2, grid.shape[0] - h - 1))
    col_starts = list(range(2, grid.shape[1] - w - 1))
    if not row_starts or not col_starts:
        return None

    positions = [(r, c) for r in row_starts for c in col_starts]
    rng.shuffle(positions)
    ar, ac = anchor
    positions.sort(
        key=lambda pos: abs((2 * pos[0] + h - 1) - 2 * ar)
        + abs((2 * pos[1] + w - 1) - 2 * ac)
    )

    for top, left in positions:
        if not _can_place(occupied, cells, top, left):
            continue
        coords = tuple((top + r, left + c) for r, c in cells)
        for r, c in coords:
            grid[r, c] = color
            occupied[r, c] = True
        return {
            "coords": coords,
            "bbox": (top, left, top + h - 1, left + w - 1),
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    object_count = rng.randint(3, 4)

    min_size = 12 + 2 * (object_count - 3)
    h = rng.randint(min_size, min_size + 4)
    w = rng.randint(min_size, min_size + 4)
    grid = np.zeros((h, w), dtype=int)

    horizontal_count = rng.randint(1, object_count - 1)
    sides = [rng.choice(["top", "bottom"]) for _ in range(horizontal_count)]
    sides += [rng.choice(["left", "right"]) for _ in range(object_count - horizontal_count)]
    rng.shuffle(sides)

    palette = rng.sample(list(range(1, 10)), 2 * object_count)
    select_colors = palette[:object_count]
    replace_colors = palette[object_count:]

    specs: list[dict[str, object]] = []
    for select_color, replace_color, side in zip(select_colors, replace_colors, sides):
        select_marker = _place_pair(grid, rng, side, select_color, replace_color)
        if select_marker is None:
            return None
        axis = AXIS_BY_SIDE[side]
        shape = _random_shape(rng, axis)
        if shape is None:
            return None
        specs.append(
            {
                "select": select_color,
                "replace": replace_color,
                "side": side,
                "axis": axis,
                "shape": shape,
                "anchor": select_marker,
            }
        )

    assert any(spec["axis"] == "horizontal" for spec in specs)
    assert any(spec["axis"] == "vertical" for spec in specs)

    occupied = np.zeros_like(grid, dtype=bool)
    specs.sort(key=lambda spec: -len(spec["shape"]))
    for spec in specs:
        placement = _place_shape(
            grid,
            occupied,
            rng,
            spec["shape"],
            int(spec["select"]),
            spec["anchor"],
        )
        if placement is None:
            return None
        spec.update(placement)

    expected = grid.copy()
    for spec in specs:
        coords = tuple(spec["coords"])
        bbox = spec["bbox"]
        axis = str(spec["axis"])
        replace_color = int(spec["replace"])
        new_cells = _reflect_component(coords, bbox, axis)
        for r, c in coords:
            expected[r, c] = 0
        for r, c in new_cells:
            if int(expected[r, c]) != 0:
                return None
            expected[r, c] = replace_color

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
