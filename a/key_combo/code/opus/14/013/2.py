from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _is_rect_border(coords: np.ndarray) -> bool:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if coords.shape[0] != expected or r1 - r0 < 2 or c1 - c0 < 2:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    return True


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _component_coords(grid == GREY):
        if not _is_rect_border(coords):
            continue
        frames.append({"bbox": _bbox(coords), "coords": coords, "children": [], "parent": None, "depth": None})
    assert frames
    for i, frame in enumerate(frames):
        containers = [j for j, other in enumerate(frames) if i != j and _strictly_contains(other["bbox"], frame["bbox"])]
        if containers:
            parent = min(containers, key=lambda j: _bbox_area(frames[j]["bbox"]))
            frame["parent"] = parent
            frames[parent]["children"].append(i)
    roots = [i for i, frame in enumerate(frames) if frame["parent"] is None]
    for root in roots:
        stack = [(root, 0)]
        while stack:
            idx, depth = stack.pop()
            frames[idx]["depth"] = depth
            for child in frames[idx]["children"]:
                stack.append((child, depth + 1))
    frames.sort(key=lambda frame: (frame["depth"], -_bbox_area(frame["bbox"])))
    return frames


def _depth_map(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), -1, dtype=int)
    ordered = sorted(frames, key=lambda frame: frame["depth"])
    for r in range(h):
        for c in range(w):
            best = -1
            for frame in ordered:
                r0, c0, r1, c1 = frame["bbox"]
                if r0 < r < r1 and c0 < c < c1:
                    best = int(frame["depth"])
            out[r, c] = best
    return out


def _find_markers(grid: np.ndarray, depth_map: np.ndarray) -> dict[int, tuple[int, int]]:
    markers: dict[int, tuple[int, int]] = {}
    for coords in _component_coords(grid == GREEN):
        if coords.shape[0] != 4:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN):
            continue
        depths = set(int(depth_map[r, c]) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
        assert len(depths) == 1
        depth = depths.pop()
        assert depth >= 0 and depth not in markers
        markers[depth] = (r0, c0)
    return markers


def _mirror_cell(r: int, c: int, marker_top: int, marker_left: int, horizontal: bool) -> tuple[int, int]:
    if horizontal:
        return marker_top + marker_top + 1 - r, c
    return r, marker_left + marker_left + 1 - c


def _apply_rule(grid: np.ndarray, mode: str = "parity") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _extract_frames(g)
    depth_map = _depth_map(g.shape, frames)
    markers = _find_markers(g, depth_map)
    h, w = g.shape
    for r, c in np.argwhere(g == RED):
        r = int(r)
        c = int(c)
        depth = int(depth_map[r, c])
        if depth < 0:
            continue
        marker_top, marker_left = markers[depth]
        if mode == "parity":
            horizontal = depth % 2 == 0
        elif mode == "all_horizontal":
            horizontal = True
        elif mode == "all_vertical":
            horizontal = False
        elif mode == "inverted":
            horizontal = depth % 2 == 1
        else:
            raise ValueError(f"unknown mode: {mode}")
        rr, cc = _mirror_cell(r, c, marker_top, marker_left, horizontal)
        if 0 <= rr < h and 0 <= cc < w and depth_map[rr, cc] == depth and out[rr, cc] == BLACK:
            out[rr, cc] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each red shape with a single mirror whose axis is chosen by nested-frame depth.

    Input structure:
    - A black background contains 2 or 3 nested grey single-pixel rectangular frames.
    - Each ring region and the innermost region contains exactly one green 2x2 marker square.
    - Some regions also contain a small red 4-connected shape that stays on one side of the marker's future mirror axis.

    Rule:
    - Build the containment chain of grey rectangular frames and assign depths: outermost frame depth 0, its child depth 1, and so on.
    - For every cell inside a frame but not inside any deeper frame, use that region's green 2x2 square as the mirror center.
    - In even-depth regions reflect each red pixel across the marker's horizontal center line; in odd-depth regions reflect each red pixel across the marker's vertical center line.
    - Paint only the reflected red pixels onto black cells, keeping the grey frames, the green markers, and existing red pixels unchanged.

    Key insight:
    - The green square gives the symmetry center, but the topology of nested containment selects which single axis to use in that region.
    """
    return _apply_rule(grid, mode="parity")


def _min_outer_size(frame_count: int) -> int:
    return 7 + 8 * (frame_count - 1)


def _sample_frame_chain(rng: np.random.Generator) -> list[tuple[int, int, int, int]] | None:
    frame_count = int(rng.choice([2, 3], p=[0.45, 0.55]))
    margins = [int(rng.integers(1, 3)) for _ in range(4)]
    top, bottom, left, right = margins
    max_h = 30 - top - bottom
    max_w = 30 - left - right
    min_outer = _min_outer_size(frame_count)
    if max_h < min_outer or max_w < min_outer:
        return None
    outer_h = int(rng.integers(min_outer, max_h + 1))
    outer_w = int(rng.integers(min_outer, max_w + 1))
    boxes = [(top, left, top + outer_h - 1, left + outer_w - 1)]
    for depth in range(1, frame_count):
        r0, c0, r1, c1 = boxes[-1]
        cur_h = r1 - r0 + 1
        cur_w = c1 - c0 + 1
        min_child = _min_outer_size(frame_count - depth)
        max_row_slack = cur_h - min_child
        max_col_slack = cur_w - min_child
        if max_row_slack < 8 or max_col_slack < 8:
            return None
        row_slack = int(rng.integers(8, max_row_slack + 1))
        col_slack = int(rng.integers(8, max_col_slack + 1))
        top_inset = int(rng.integers(4, row_slack - 3))
        bottom_inset = row_slack - top_inset
        left_inset = int(rng.integers(4, col_slack - 3))
        right_inset = col_slack - left_inset
        boxes.append((r0 + top_inset, c0 + left_inset, r1 - bottom_inset, c1 - right_inset))
    return boxes


def _draw_frames(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, list[dict], np.ndarray]:
    grid = np.full(shape, BLACK, dtype=int)
    for r0, c0, r1, c1 in boxes:
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
    frames = _extract_frames(grid)
    depth_map = _depth_map(shape, frames)
    return grid, frames, depth_map


def _random_connected_subset(
    rng: np.random.Generator,
    allowed: set[tuple[int, int]],
    size_min: int = 3,
    size_max: int = 7,
) -> set[tuple[int, int]] | None:
    if len(allowed) < size_min:
        return None
    starts = list(allowed)
    rng.shuffle(starts)
    for start in starts[: min(len(starts), 18)]:
        target = int(rng.integers(size_min, min(size_max, len(allowed)) + 1))
        chosen = {start}
        frontier = {cell for cell in _neighbors4(*start) if cell in allowed}
        while frontier and len(chosen) < target:
            frontier_list = list(frontier)
            cell = frontier_list[int(rng.integers(len(frontier_list)))]
            frontier.remove(cell)
            if cell in chosen:
                continue
            chosen.add(cell)
            for nxt in _neighbors4(*cell):
                if nxt in allowed and nxt not in chosen:
                    frontier.add(nxt)
        if size_min <= len(chosen) <= size_max:
            return chosen
    return None


def _shape_candidates(
    depth_map: np.ndarray,
    depth: int,
    marker_top: int,
    marker_left: int,
    horizontal: bool,
    side: str,
    quadrant_mode: str,
) -> set[tuple[int, int]]:
    allowed: set[tuple[int, int]] = set()
    h, w = depth_map.shape
    for r in range(h):
        for c in range(w):
            if depth_map[r, c] != depth:
                continue
            if marker_top <= r <= marker_top + 1 and marker_left <= c <= marker_left + 1:
                continue
            rr, cc = _mirror_cell(r, c, marker_top, marker_left, horizontal)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if depth_map[rr, cc] != depth or (rr == r and cc == c):
                continue
            if horizontal:
                if side == "above" and not (r < marker_top):
                    continue
                if side == "below" and not (r > marker_top + 1):
                    continue
                if quadrant_mode == "left" and not (c < marker_left):
                    continue
                if quadrant_mode == "right" and not (c > marker_left + 1):
                    continue
            else:
                if side == "left" and not (c < marker_left):
                    continue
                if side == "right" and not (c > marker_left + 1):
                    continue
                if quadrant_mode == "above" and not (r < marker_top):
                    continue
                if quadrant_mode == "below" and not (r > marker_top + 1):
                    continue
            allowed.add((r, c))
    return allowed


def _place_region_content(
    rng: np.random.Generator,
    grid: np.ndarray,
    depth_map: np.ndarray,
    depth: int,
    must_have_red: bool,
) -> tuple[int, int, set[tuple[int, int]]] | None:
    positions = []
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(depth_map[r : r + 2, c : c + 2] == depth):
                positions.append((r, c))
    rng.shuffle(positions)
    horizontal = depth % 2 == 0
    side_options = ["above", "below"] if horizontal else ["left", "right"]
    ortho_options = ["both", "left", "right"] if horizontal else ["both", "above", "below"]
    for marker_top, marker_left in positions[: min(len(positions), 60)]:
        if not must_have_red:
            grid[marker_top : marker_top + 2, marker_left : marker_left + 2] = GREEN
            return marker_top, marker_left, set()
        side_order = side_options.copy()
        ortho_order = ortho_options.copy()
        rng.shuffle(side_order)
        rng.shuffle(ortho_order)
        for side in side_order:
            for quadrant_mode in ortho_order:
                candidates = _shape_candidates(depth_map, depth, marker_top, marker_left, horizontal, side, quadrant_mode)
                shape = _random_connected_subset(rng, candidates)
                if shape is None:
                    continue
                grid[marker_top : marker_top + 2, marker_left : marker_left + 2] = GREEN
                for r, c in shape:
                    grid[r, c] = RED
                return marker_top, marker_left, shape
    return None


def _active_depths(frame_count: int, rng: np.random.Generator) -> dict[int, bool]:
    active = {0: True, 1: True}
    if frame_count == 3:
        active[2] = bool(rng.random() < 0.65)
    return active


def _validate_example(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    frames = _extract_frames(grid)
    if len(frames) not in {2, 3}:
        return False
    depths = sorted(int(frame["depth"]) for frame in frames)
    if depths != list(range(len(frames))):
        return False
    depth_map = _depth_map(grid.shape, frames)
    markers = _find_markers(grid, depth_map)
    if set(markers) != set(range(len(frames))):
        return False
    correct = main(grid)
    if np.array_equal(correct, grid):
        return False
    active = {int(depth_map[r, c]) for r, c in np.argwhere(grid == RED) if int(depth_map[r, c]) >= 0}
    if not active or not any(depth % 2 == 0 for depth in active) or not any(depth % 2 == 1 for depth in active):
        return False
    for depth, marker in markers.items():
        marker_top, marker_left = marker
        cells = {(int(r), int(c)) for r, c in np.argwhere((grid == RED) & (depth_map == depth))}
        if not cells:
            continue
        horizontal = depth % 2 == 0
        added = 0
        for r, c in cells:
            rr, cc = _mirror_cell(r, c, marker_top, marker_left, horizontal)
            if depth_map[rr, cc] != depth or correct[rr, cc] != RED:
                return False
            if grid[rr, cc] == BLACK:
                added += 1
        if added == 0:
            return False
    if np.array_equal(correct, _apply_rule(grid, "all_horizontal")):
        return False
    if np.array_equal(correct, _apply_rule(grid, "all_vertical")):
        return False
    if np.array_equal(correct, _apply_rule(grid, "inverted")):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    boxes = _sample_frame_chain(rng)
    if boxes is None:
        return None
    outer = boxes[0]
    h = outer[2] + 1 + int(rng.integers(1, 3))
    w = outer[3] + 1 + int(rng.integers(1, 3))
    if h > 30 or w > 30:
        return None
    grid, _frames, depth_map = _draw_frames((h, w), boxes)
    frame_count = len(boxes)
    active = _active_depths(frame_count, rng)
    for depth in range(frame_count):
        placed = _place_region_content(rng, grid, depth_map, depth, must_have_red=active.get(depth, False))
        if placed is None:
            return None
    try:
        if not _validate_example(grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth mirror example")
