from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


GREY = Color.GREY
RED = Color.RED

DIR_CONFIG = {
    "top": {"phase": 0, "mode": "row"},
    "bottom": {"phase": 1, "mode": "row"},
    "left": {"phase": 0, "mode": "col"},
    "right": {"phase": 1, "mode": "col"},
}

GEN_GRID_ROWS = (13, 18)
GEN_GRID_COLS = (13, 18)
GEN_FRAME_COUNT = (2, 3)
GEN_INTERIOR_HEIGHTS = (3, 4, 5, 6)
GEN_INTERIOR_WIDTHS = (3, 4, 5, 6, 7)
GEN_MAX_ATTEMPTS = 400
GEN_DIRECTIONS = ("top", "bottom", "left", "right")


def _frame_bbox_from_component_bbox(
    bbox: tuple[int, int, int, int], direction: str
) -> tuple[int, int, int, int] | None:
    br0, bc0, br1, bc1 = map(int, bbox)
    if direction == "top":
        candidate = (br0 + 1, bc0, br1, bc1)
    elif direction == "bottom":
        candidate = (br0, bc0, br1 - 1, bc1)
    elif direction == "left":
        candidate = (br0, bc0 + 1, br1, bc1)
    else:
        candidate = (br0, bc0, br1, bc1 - 1)

    r0, c0, r1, c1 = candidate
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None
    return candidate


def _open_frame_cells(bbox: tuple[int, int, int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cells: set[tuple[int, int]] = set()

    cells.update((r0, c) for c in range(c0, c1 + 1))
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))

    if direction == "top":
        cells.difference_update((r0, c) for c in range(c0 + 1, c1))
    elif direction == "bottom":
        cells.difference_update((r1, c) for c in range(c0 + 1, c1))
    elif direction == "left":
        cells.difference_update((r, c0) for r in range(r0 + 1, r1))
    else:
        cells.difference_update((r, c1) for r in range(r0 + 1, r1))
    return cells


def _hat_options(bbox: tuple[int, int, int, int], direction: str) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, bbox)
    if direction == "top":
        return [
            {(r0 - 1, c0), (r0 - 1, c0 + 1)},
            {(r0 - 1, c1 - 1), (r0 - 1, c1)},
        ]
    if direction == "bottom":
        return [
            {(r1 + 1, c0), (r1 + 1, c0 + 1)},
            {(r1 + 1, c1 - 1), (r1 + 1, c1)},
        ]
    if direction == "left":
        return [
            {(r0, c0 - 1), (r0 + 1, c0 - 1)},
            {(r1 - 1, c0 - 1), (r1, c0 - 1)},
        ]
    return [
        {(r0, c1 + 1), (r0 + 1, c1 + 1)},
        {(r1 - 1, c1 + 1), (r1, c1 + 1)},
    ]


def _fill_linear(
    out: np.ndarray, bg: int, bbox: tuple[int, int, int, int], direction: str
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    phase = int(DIR_CONFIG[direction]["phase"])
    mode = str(DIR_CONFIG[direction]["mode"])

    for dr in range(ih):
        for dc in range(iw):
            rr = r0 + 1 + dr
            cc = c0 + 1 + dc
            if int(out[rr, cc]) != int(bg):
                continue
            idx = dr * iw + dc if mode == "row" else dc * ih + dr
            if idx % 2 == phase:
                out[rr, cc] = RED


def _close_frame(out: np.ndarray, bbox: tuple[int, int, int, int], direction: str) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    if direction == "top":
        out[r0, c0 + 1 : c1] = GREY
    elif direction == "bottom":
        out[r1, c0 + 1 : c1] = GREY
    elif direction == "left":
        out[r0 + 1 : r1, c0] = GREY
    else:
        out[r0 + 1 : r1, c1] = GREY


def _parse_frame_component(g: np.ndarray, comp: dict, bg: int) -> dict:
    comp_cells = {tuple(map(int, rc)) for rc in comp["coords"]}
    matches: list[dict] = []
    for direction in GEN_DIRECTIONS:
        frame_bbox = _frame_bbox_from_component_bbox(comp["bbox"], direction)
        if frame_bbox is None:
            continue

        r0, c0, r1, c1 = map(int, frame_bbox)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not bool(np.all(interior == int(bg))):
            continue

        frame_cells = _open_frame_cells(frame_bbox, direction)
        for hat_cells in _hat_options(frame_bbox, direction):
            if frame_cells | hat_cells != comp_cells:
                continue
            matches.append(
                {
                    "direction": direction,
                    "frame_bbox": frame_bbox,
                    "hat_cells": tuple(sorted(hat_cells)),
                }
            )

    assert len(matches) == 1, "each grey component must be one uniquely decodable open frame"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each open rectangular frame and fill its interior with a direction-coded alternating red pattern.

    Input structure:
    - The background is the dominant color and the only foreground color is grey.
    - Each grey 4-connected component is a one-cell-thick rectangular frame with one side open.
    - The open side keeps its two corner pixels, while its interior edge segment is missing.
    - A two-pixel grey hat sits just outside one end of the opening and stays attached to the frame.

    Rule:
    - For every grey component, recover the underlying rectangle and determine whether the open side is top, bottom, left, or right.
    - Complete the missing edge segment in grey, leaving the hat unchanged.
    - Fill only the rectangle interior's background cells in red using an alternating pattern:
      top/bottom openings use row-major alternation with opposite start phases, and
      left/right openings use column-major alternation with opposite start phases.

    Key insight:
    - The frame must first be normalized back to its full rectangle; then the original opening direction selects both the fill phase and whether the interior alternation is read by rows or by columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert set(map(int, np.unique(g))) <= {bg, int(GREY)}

    comps = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, colors=[GREY], return_format="dict"
    )
    assert comps, "expected at least one grey open frame"

    parsed = [_parse_frame_component(g, comp, bg) for comp in comps]

    out = g.copy()
    for info in parsed:
        frame_bbox = info["frame_bbox"]
        direction = str(info["direction"])
        _close_frame(out, frame_bbox, direction)
        _fill_linear(out, bg, frame_bbox, direction)
    return out


def _sample_direction_multiset(frame_count: int) -> list[str]:
    distinct_count = frame_count if frame_count == 2 or random.random() < 0.7 else 2
    distinct = random.sample(GEN_DIRECTIONS, distinct_count)
    directions = distinct[:]
    while len(directions) < frame_count:
        directions.append(random.choice(distinct))
    random.shuffle(directions)
    return directions


def _sample_interior(direction: str, used_areas: set[int]) -> tuple[int, int]:
    for _ in range(200):
        ih = random.choice(GEN_INTERIOR_HEIGHTS)
        iw = random.choice(GEN_INTERIOR_WIDTHS)
        if direction in ("left", "right") and random.random() < 0.75:
            even_heights = [v for v in GEN_INTERIOR_HEIGHTS if v % 2 == 0]
            ih = random.choice(even_heights)
        area = ih * iw
        if area in used_areas:
            continue
        return ih, iw
    raise RuntimeError("failed to sample distinct interior areas")


def _frame_input_cells(frame: dict) -> set[tuple[int, int]]:
    return _open_frame_cells(frame["bbox"], frame["direction"]) | set(frame["hat_cells"])


def _touches_existing(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _all_positions(h: int, w: int, oh: int, ow: int, direction: str) -> list[tuple[int, int]]:
    if direction == "top":
        r_values = range(1, h - oh + 1)
        c_values = range(0, w - ow + 1)
    elif direction == "bottom":
        r_values = range(0, h - oh)
        c_values = range(0, w - ow + 1)
    elif direction == "left":
        r_values = range(0, h - oh + 1)
        c_values = range(1, w - ow + 1)
    else:
        r_values = range(0, h - oh + 1)
        c_values = range(0, w - ow)
    return [(int(r0), int(c0)) for r0 in r_values for c0 in c_values]


def _make_frame(r0: int, c0: int, ih: int, iw: int, direction: str) -> dict:
    bbox = (r0, c0, r0 + ih + 1, c0 + iw + 1)
    hat_cells = random.choice(_hat_options(bbox, direction))
    return {"bbox": bbox, "direction": direction, "hat_cells": tuple(sorted(hat_cells))}


def _place_frames(h: int, w: int, directions: list[str]) -> list[dict] | None:
    frames: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    used_areas: set[int] = set()

    for direction in sorted(directions, key=lambda d: 0 if d in ("left", "right") else 1):
        ih, iw = _sample_interior(direction, used_areas)
        positions = _all_positions(h, w, ih + 2, iw + 2, direction)
        random.shuffle(positions)

        placed = None
        for r0, c0 in positions:
            frame = _make_frame(r0, c0, ih, iw, direction)
            cells = _frame_input_cells(frame)
            if _touches_existing(cells, occupied):
                continue
            placed = frame
            break

        if placed is None:
            return None

        frames.append(placed)
        used_areas.add(ih * iw)
        occupied.update(_frame_input_cells(placed))

    return frames


def _render_input(h: int, w: int, frames: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        for r, c in _frame_input_cells(frame):
            grid[int(r), int(c)] = GREY
    return grid


def _validate_generated_input(inp: np.ndarray, frames: list[dict]) -> None:
    parsed = find_components(
        inp, bg=0, connectivity=4, monochromatic=True, colors=[GREY], return_format="dict"
    )
    assert len(parsed) == len(frames)

    recovered = [_parse_frame_component(inp, comp, 0) for comp in parsed]
    recovered_dirs = sorted(str(info["direction"]) for info in recovered)
    expected_dirs = sorted(str(frame["direction"]) for frame in frames)
    assert recovered_dirs == expected_dirs

    areas = []
    for info in recovered:
        r0, c0, r1, c1 = map(int, info["frame_bbox"])
        areas.append((r1 - r0 - 1) * (c1 - c0 - 1))
    assert len(set(areas)) == len(areas)
    assert len(set(recovered_dirs)) >= 2

    out = main(inp)
    assert out.shape == inp.shape
    assert not np.array_equal(inp, out)


def _generate_once() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_ROWS)
    w = random.randint(*GEN_GRID_COLS)
    frame_count = random.randint(*GEN_FRAME_COUNT)
    directions = _sample_direction_multiset(frame_count)
    frames = _place_frames(h, w, directions)
    if frames is None:
        return None

    inp = _render_input(h, w, frames)
    try:
        _validate_generated_input(inp, frames)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
