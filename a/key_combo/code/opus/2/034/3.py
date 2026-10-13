from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_034_3",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4
paste = _ARC_UTILS.paste


BACKGROUND = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
GREY = int(Color.GREY)

GRID_HEIGHT_RANGE = (15, 25)
GRID_WIDTH_RANGE = (16, 26)
SPRITE_AREA_RANGE = (6, 10)
SPRITE_MAX_SPAN = 5
TARGET_COUNT_RANGE = (3, 5)
GENERATION_RETRIES = 500
PLACEMENT_RETRIES = 80
SELECTION_RETRIES = 30


def _expand4(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    halo = set(cells)
    for r, c in cells:
        halo.update(neighbors4(r, c))
    return halo


def _source_from_grid(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], tuple[int, int], list[tuple[int, int]]]:
    components = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )

    source_cells: list[tuple[int, int, int]] | None = None
    source_anchor: tuple[int, int] | None = None
    targets: list[tuple[int, int]] = []
    for comp in components:
        coords = comp["coords"]
        colors = grid[coords[:, 0], coords[:, 1]]
        if np.any(colors == GREY) and np.any(colors != GREY):
            assert source_cells is None, "expected exactly one source sprite"
            anchor_coords = coords[colors == GREY]
            assert anchor_coords.shape == (1, 2), "source sprite must contain one gray anchor"
            source_anchor = tuple(map(int, anchor_coords[0]))
            source_cells = [
                (int(r), int(c), int(grid[int(r), int(c)]))
                for r, c in coords
            ]
        else:
            assert np.all(colors == GREY), "non-source components must be isolated gray anchors"
            assert coords.shape == (1, 2), "target anchors must be isolated gray pixels"
            targets.append(tuple(map(int, coords[0])))

    assert source_cells is not None and source_anchor is not None, "missing source sprite"
    assert targets, "expected at least one target anchor"
    targets.sort()
    return source_cells, source_anchor, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the two-color source sprite onto each gray target anchor and recolor the copy based on alignment direction.

    Input structure:
    - The grid has a black background, one 4-connected multicolor sprite made of blue, red, and exactly one gray anchor pixel, and several isolated gray target anchors.
    - Every target anchor lies either in the same row as the source anchor or in the same column as the source anchor.

    Rule:
    - Keep the source sprite unchanged.
    - For each target anchor from top to bottom, then left to right, erase that gray pixel and translate every non-gray source pixel so the source anchor would land on the target anchor.
    - If the target is horizontally aligned with the source anchor, recolor blue to yellow and keep red unchanged; if it is vertically aligned, recolor red to green and keep blue unchanged.

    Key insight:
    - The anchor alignment determines where each copy goes, and the row-vs-column alignment also selects which color channel gets replaced in that copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source_cells, source_anchor, targets = _source_from_grid(g)
    sr, sc = source_anchor
    out = g.copy()

    for tr, tc in targets:
        assert (tr == sr) ^ (tc == sc), "each target must align horizontally or vertically with the source anchor"
        out[tr, tc] = BACKGROUND
        horizontal = tr == sr
        for r, c, color in source_cells:
            if color == GREY:
                continue
            rr = tr + (r - sr)
            cc = tc + (c - sc)
            assert in_bounds(g.shape, rr, cc), "translated copy must stay in bounds"
            if horizontal and color == BLUE:
                color = YELLOW
            elif (not horizontal) and color == RED:
                color = GREEN
            out[rr, cc] = color

    return out


def _make_sprite(rng: np.random.Generator) -> dict | None:
    area = int(rng.integers(SPRITE_AREA_RANGE[0], SPRITE_AREA_RANGE[1] + 1))
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            frontier_list = list(frontier)
            choice = frontier_list[int(rng.integers(len(frontier_list)))]
            cells.add(choice)

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        if max_r - min_r + 1 > SPRITE_MAX_SPAN or max_c - min_c + 1 > SPRITE_MAX_SPAN:
            continue
        if min_r == max_r or min_c == max_c:
            continue
        anchor_degree = sum((nr, nc) in cells for nr, nc in neighbors4(0, 0))
        if anchor_degree < 2:
            continue

        non_anchor = sorted(cell for cell in cells if cell != (0, 0))
        blue_count = int(rng.integers(2, len(non_anchor) - 1))
        while len(non_anchor) - blue_count < 2:
            blue_count = int(rng.integers(2, len(non_anchor) - 1))
        order = list(range(len(non_anchor)))
        rng.shuffle(order)
        blue_indices = set(order[:blue_count])

        patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
        cells_colored = [(0, 0, GREY)]
        for idx, (r, c) in enumerate(non_anchor):
            color = BLUE if idx in blue_indices else RED
            patch[r - min_r, c - min_c] = color
            cells_colored.append((r, c, color))
        patch[-min_r, -min_c] = GREY
        return {
            "cells": cells_colored,
            "non_anchor": [(r, c, color) for r, c, color in cells_colored if color != GREY],
            "full_offsets": {(r, c) for r, c, _color in cells_colored},
            "bbox": (min_r, min_c, max_r, max_c),
            "patch": patch,
            "origin": (min_r, min_c),
        }
    return None


def _candidate_records(
    shape: tuple[int, int],
    sprite: dict,
    source_anchor: tuple[int, int],
) -> tuple[list[dict], list[dict]]:
    height, width = shape
    sr, sc = source_anchor
    min_r, min_c, max_r, max_c = sprite["bbox"]
    source_full = {(sr + r, sc + c) for r, c in sprite["full_offsets"]}
    source_halo = _expand4(source_full)

    horizontal: list[dict] = []
    vertical: list[dict] = []

    for shift_c in range(-sc - min_c, width - sc - max_c):
        if shift_c == 0:
            continue
        anchor = (sr, sc + shift_c)
        if anchor in source_halo:
            continue
        copy_cells = {(sr + r, sc + c + shift_c) for r, c, _color in sprite["non_anchor"]}
        if copy_cells & source_halo:
            continue
        horizontal.append(
            {
                "axis": "h",
                "anchor": anchor,
                "cells": copy_cells,
                "halo": _expand4(copy_cells),
                "distance": abs(shift_c),
            }
        )

    for shift_r in range(-sr - min_r, height - sr - max_r):
        if shift_r == 0:
            continue
        anchor = (sr + shift_r, sc)
        if anchor in source_halo:
            continue
        copy_cells = {(sr + r + shift_r, sc + c) for r, c, _color in sprite["non_anchor"]}
        if copy_cells & source_halo:
            continue
        vertical.append(
            {
                "axis": "v",
                "anchor": anchor,
                "cells": copy_cells,
                "halo": _expand4(copy_cells),
                "distance": abs(shift_r),
            }
        )

    return horizontal, vertical


def _compatible(candidate: dict, selected: list[dict]) -> bool:
    ar, ac = candidate["anchor"]
    for other in selected:
        br, bc = other["anchor"]
        if abs(ar - br) + abs(ac - bc) <= 1:
            return False
        if candidate["cells"] & other["halo"]:
            return False
        if other["cells"] & candidate["halo"]:
            return False
        if candidate["anchor"] in other["cells"] or other["anchor"] in candidate["cells"]:
            return False
    return True


def _pick_targets(
    rng: np.random.Generator,
    horizontal: list[dict],
    vertical: list[dict],
    h_needed: int,
    v_needed: int,
) -> list[dict] | None:
    if len(horizontal) < h_needed or len(vertical) < v_needed:
        return None

    for _ in range(SELECTION_RETRIES):
        selected: list[dict] = []
        pools = [
            ("h", sorted(horizontal, key=lambda item: (-item["distance"], rng.random()))),
            ("v", sorted(vertical, key=lambda item: (-item["distance"], rng.random()))),
        ]
        if rng.random() < 0.5:
            pools.reverse()

        needed = {"h": h_needed, "v": v_needed}
        success = True
        for axis, pool in pools:
            chosen = 0
            for candidate in pool:
                if _compatible(candidate, selected):
                    selected.append(candidate)
                    chosen += 1
                    if chosen == needed[axis]:
                        break
            if chosen < needed[axis]:
                success = False
                break

        if success:
            selected.sort(key=lambda item: item["anchor"])
            return selected
    return None


def _render_input(shape: tuple[int, int], sprite: dict, source_anchor: tuple[int, int], targets: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    sr, sc = source_anchor
    min_r, min_c = sprite["origin"]
    paste(grid, sprite["patch"], sr + min_r, sc + min_c, transparent=BACKGROUND)
    for target in targets:
        tr, tc = target["anchor"]
        grid[tr, tc] = GREY
    return grid


def _attempt_input(rng: np.random.Generator) -> np.ndarray | None:
    sprite = _make_sprite(rng)
    if sprite is None:
        return None

    height = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    min_r, min_c, max_r, max_c = sprite["bbox"]
    total_targets = int(rng.integers(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    h_needed = int(rng.integers(1, total_targets))
    v_needed = total_targets - h_needed

    row_candidates = list(range(-min_r, height - max_r))
    col_candidates = list(range(-min_c, width - max_c))
    if not row_candidates or not col_candidates:
        return None
    rng.shuffle(row_candidates)
    rng.shuffle(col_candidates)

    for sr in row_candidates[:PLACEMENT_RETRIES]:
        for sc in col_candidates[:PLACEMENT_RETRIES]:
            horizontal, vertical = _candidate_records((height, width), sprite, (sr, sc))
            chosen = _pick_targets(rng, horizontal, vertical, h_needed, v_needed)
            if chosen is None:
                continue
            grid = _render_input((height, width), sprite, (sr, sc), chosen)
            try:
                out = main(grid)
            except AssertionError:
                continue
            if int(np.count_nonzero(grid == GREY)) != total_targets + 1:
                continue
            if np.any(out == YELLOW) and np.any(out == GREEN):
                return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 034/3")
