from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg, neighbors4


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BACKGROUND = Color.BLACK
GRID_RANGE = (12, 24)
BOX_SIDE_RANGE = (6, 13)
VALID_PIXEL_RANGE = (4, 8)
DECOY_PIXEL_RANGE = (1, 4)
MAX_GEN_ATTEMPTS = 400


def _border_coords(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    coords: set[Coord] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _strictly_inside(box: BBox, cell: Coord) -> bool:
    r0, c0, r1, c1 = box
    r, c = cell
    return r0 < r < r1 and c0 < c < c1


def _inside_or_on(box: BBox, cell: Coord) -> bool:
    r0, c0, r1, c1 = box
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _point_reflect(center: Coord, cell: Coord) -> Coord:
    return 2 * center[0] - cell[0], 2 * center[1] - cell[1]


def _gap_side(box: BBox, gap: Coord) -> str:
    r0, c0, r1, c1 = box
    r, c = gap
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on the box border")


def _axis_reflect(box: BBox, gap: Coord, cell: Coord) -> Coord:
    side = _gap_side(box, gap)
    r, c = cell
    if side in ("top", "bottom"):
        return 2 * gap[0] - r, c
    return r, 2 * gap[1] - c


def _candidate_box(grid: np.ndarray, component: dict, bg: int) -> dict | None:
    coords = component["coords"]
    box = tuple(map(int, component["bbox"]))
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _border_coords(box)
    cells = {tuple(map(int, rc)) for rc in coords}
    if len(cells) != len(border) - 1:
        return None
    if not cells.issubset(border):
        return None

    missing = list(border - cells)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if int(grid[gap]) != int(bg):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != int(bg)):
        return None

    return {
        "bbox": box,
        "gap": gap,
        "color": int(component["color"]),
        "cells": cells,
    }


def _find_box(grid: np.ndarray) -> tuple[int, dict]:
    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    candidates = []
    for component in components:
        candidate = _candidate_box(grid, component, bg)
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected exactly one hollow rectangle box with one gap"
    return bg, candidates[0]


def _projection_result(
    grid: np.ndarray,
    *,
    use_axis_reflection: bool = False,
) -> tuple[np.ndarray, dict]:
    bg, box_info = _find_box(grid)
    box = box_info["bbox"]
    gap = box_info["gap"]
    box_cells = box_info["cells"]

    out = np.asarray(grid, dtype=int).copy()
    valid_sources: list[tuple[Coord, Coord, int]] = []
    clipped_sources: list[tuple[Coord, Coord, int]] = []

    for r, c in np.argwhere(grid != bg):
        src = (int(r), int(c))
        if src in box_cells:
            continue
        assert not _strictly_inside(box, src), "box interior must be empty in the input"
        target = _axis_reflect(box, gap, src) if use_axis_reflection else _point_reflect(gap, src)
        if _strictly_inside(box, target):
            color = int(grid[src])
            if int(out[target]) not in (bg, color):
                raise AssertionError("multiple source pixels map to the same target with different colors")
            out[target] = color
            valid_sources.append((src, target, color))
        else:
            clipped_sources.append((src, target, int(grid[src])))

    return out, {
        "bg": bg,
        "bbox": box,
        "gap": gap,
        "valid_sources": valid_sources,
        "clipped_sources": clipped_sources,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the outside pattern through the box's single gap into the hollow rectangle interior by point symmetry.

    Input structure:
    - The grid has a black background, one one-cell-thick hollow rectangular box, and exactly one missing wall pixel on that box.
    - The box interior is empty in the input, while colored pixels of one or more outside patterns lie elsewhere on the background.
    - The missing wall pixel is the only gap on the rectangle border and acts as the projection center.

    Rule:
    - Identify the unique hollow rectangle whose full border is present except for one background gap pixel.
    - For every colored pixel outside the box, reflect it through the gap by point symmetry.
    - If the reflected location falls strictly inside the box interior, paint that interior cell with the source pixel's color; otherwise discard it.
    - Keep the outside pixels, the box wall, and the gap unchanged.

    Key insight:
    - The gap is a pinhole, so the image is a 180-degree point reflection around that single pixel, not a mirror across the whole wall.
    """
    out, _ = _projection_result(np.asarray(grid, dtype=int), use_axis_reflection=False)
    return out


def _projection_pairs(shape: tuple[int, int], box: BBox, gap: Coord) -> dict[Coord, Coord]:
    pairs: dict[Coord, Coord] = {}
    r0, c0, r1, c1 = box
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            dst = (r, c)
            src = _point_reflect(gap, dst)
            if _in_bounds(shape, src) and not _inside_or_on(box, src):
                pairs[dst] = src
    return pairs


def _tangential_offset(box: BBox, gap: Coord, cell: Coord) -> bool:
    side = _gap_side(box, gap)
    if side in ("top", "bottom"):
        return cell[1] != gap[1]
    return cell[0] != gap[0]


def _grow_target_shape(available: set[Coord], box: BBox, gap: Coord) -> list[Coord] | None:
    tangential = [cell for cell in available if _tangential_offset(box, gap, cell)]
    if not tangential:
        return None

    max_size = min(VALID_PIXEL_RANGE[1], len(available))
    min_size = min(VALID_PIXEL_RANGE[0], max_size)
    if max_size < VALID_PIXEL_RANGE[0]:
        return None

    for _ in range(80):
        size = random.randint(min_size, max_size)
        start = random.choice(tangential)
        chosen = {start}
        frontier = {n for n in neighbors4(*start) if n in available}
        while len(chosen) < size and frontier:
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            if nxt in chosen:
                continue
            chosen.add(nxt)
            frontier.update(n for n in neighbors4(*nxt) if n in available and n not in chosen)

        if len(chosen) != size:
            continue
        chosen_list = sorted(chosen)
        bbox = bbox_coords(np.array(chosen_list, dtype=int))
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            continue
        return chosen_list

    return None


def _choose_colors(count: int, box_color: int) -> list[int]:
    palette = [color for color in Color.NOT_BLACK if color != box_color]
    n_colors = random.randint(2, min(4, count, len(palette)))
    chosen = random.sample(palette, n_colors)
    values = chosen[:]
    while len(values) < count:
        values.append(random.choice(chosen))
    random.shuffle(values)
    return values


def _decoy_sources(
    shape: tuple[int, int],
    box: BBox,
    gap: Coord,
    used_sources: set[Coord],
) -> list[Coord]:
    r0, c0, r1, c1 = box
    border = _border_coords(box) - {gap}
    border_hits: list[Coord] = []
    near_hits: list[Coord] = []

    for r in range(max(0, r0 - 2), min(shape[0], r1 + 3)):
        for c in range(max(0, c0 - 2), min(shape[1], c1 + 3)):
            target = (r, c)
            if _strictly_inside(box, target) or target == gap:
                continue
            src = _point_reflect(gap, target)
            if not _in_bounds(shape, src) or _inside_or_on(box, src) or src in used_sources:
                continue
            if target in border:
                border_hits.append(src)
            elif r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1):
                near_hits.append(src)

    random.shuffle(border_hits)
    random.shuffle(near_hits)
    picks: list[Coord] = []
    if border_hits:
        picks.append(border_hits.pop())

    pool = border_hits + near_hits
    random.shuffle(pool)
    want = random.randint(*DECOY_PIXEL_RANGE)
    for src in pool:
        if src not in picks:
            picks.append(src)
        if len(picks) >= want:
            break
    return picks


def _random_box(shape: tuple[int, int]) -> tuple[BBox, Coord]:
    h, w = shape
    box_h = random.randint(BOX_SIDE_RANGE[0], min(BOX_SIDE_RANGE[1], h - 4))
    box_w = random.randint(BOX_SIDE_RANGE[0], min(BOX_SIDE_RANGE[1], w - 4))
    r0 = random.randint(2, h - box_h - 2)
    c0 = random.randint(2, w - box_w - 2)
    box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        gap = (r0, random.randint(c0 + 1, c0 + box_w - 2))
    elif side == "bottom":
        gap = (box[2], random.randint(c0 + 1, c0 + box_w - 2))
    elif side == "left":
        gap = (random.randint(r0 + 1, r0 + box_h - 2), c0)
    else:
        gap = (random.randint(r0 + 1, r0 + box_h - 2), box[3])
    return box, gap


def _paint_box(grid: np.ndarray, box: BBox, gap: Coord, color: int) -> None:
    for cell in _border_coords(box):
        if cell != gap:
            grid[cell] = color


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, info = _projection_result(grid, use_axis_reflection=False)
        wrong, _ = _projection_result(grid, use_axis_reflection=True)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, wrong):
        return False

    valid_sources = info["valid_sources"]
    clipped_sources = info["clipped_sources"]
    box = info["bbox"]
    gap = info["gap"]

    if len(valid_sources) < VALID_PIXEL_RANGE[0]:
        return False
    if len(clipped_sources) < 1:
        return False

    targets = [dst for _src, dst, _color in valid_sources]
    if len(set(targets)) != len(targets):
        return False

    colors = {color for _src, _dst, color in valid_sources}
    if len(colors) < 2:
        return False

    if not any(_tangential_offset(box, gap, dst) for _src, dst, _color in valid_sources):
        return False

    motif_bbox = bbox_coords(np.array(targets, dtype=int))
    if motif_bbox[0] == motif_bbox[2] or motif_bbox[1] == motif_bbox[3]:
        return False

    src_rows = {src[0] for src, _dst, _color in valid_sources}
    src_cols = {src[1] for src, _dst, _color in valid_sources}
    if len(src_rows) == 1 or len(src_cols) == 1:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_RANGE),
        random.randint(*GRID_RANGE),
    )
    grid = np.full(shape, BACKGROUND, dtype=int)
    box, gap = _random_box(shape)
    pairs = _projection_pairs(shape, box, gap)
    targets = _grow_target_shape(set(pairs), box, gap)
    if targets is None:
        return None

    used_sources = {pairs[target] for target in targets}
    decoys = _decoy_sources(shape, box, gap, used_sources)
    if not decoys:
        return None

    box_color = random.choice(Color.NOT_BLACK)
    _paint_box(grid, box, gap, box_color)

    valid_colors = _choose_colors(len(targets), box_color)
    for target, color in zip(targets, valid_colors):
        grid[pairs[target]] = color

    decoy_palette = [color for color in Color.NOT_BLACK if color != box_color]
    for src in decoys:
        grid[src] = random.choice(decoy_palette)

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pinhole-projection ARC input")
