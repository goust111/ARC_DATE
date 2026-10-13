from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
COLORS = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")

GEN_MAX_ATTEMPTS = 400
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_FRAME_COUNT_CHOICES = (1, 2, 2, 3, 3)
GEN_FRAME_MARGIN = 1
GEN_MIN_GAP = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != int(bg)]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _component_border_points(coords: np.ndarray, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    return [
        (int(r), int(c))
        for r, c in coords
        if int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
    ]


def _extract_corridor(subgrid: np.ndarray, bg: int) -> list[Coord]:
    bg_components = _find_components(subgrid == bg)
    candidates: list[tuple[np.ndarray, list[Coord]]] = []
    for coords in bg_components:
        border_points = _component_border_points(coords, subgrid.shape)
        if border_points:
            candidates.append((coords, border_points))

    assert len(candidates) == 1, "expected exactly one corridor component touching the frame border"
    coords, border_points = candidates[0]
    assert len(border_points) == 1, "expected exactly one border gap"

    path_cells = {tuple(map(int, rc)) for rc in coords}
    gap = border_points[0]
    order = [gap]
    prev: Coord | None = None
    current = gap

    while True:
        next_steps = [nbr for nbr in _neighbors4(*current) if nbr in path_cells and nbr != prev]
        if not next_steps:
            break
        assert len(next_steps) == 1, "corridor must be a single unbranched path"
        prev, current = current, next_steps[0]
        order.append(current)

    assert len(order) == len(path_cells), "corridor traversal did not cover the whole path"
    return order[1:]


def _classify_color_components(grid: np.ndarray, color: int, bg: int) -> tuple[BBox, list[Coord], int, list[np.ndarray]]:
    components = _find_components(grid == color)
    assert components, "expected at least one component for every foreground color"

    frame_bbox: BBox | None = None
    corridor: list[Coord] | None = None
    fill_amount = 0
    bar_components: list[np.ndarray] = []

    for coords in components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(coords.shape[0])
        subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]

        if area < h * w and np.any(subgrid == bg):
            candidate_corridor = _extract_corridor(subgrid, bg)
            assert frame_bbox is None, "expected one frame per color"
            frame_bbox = (r0, c0, r1, c1)
            corridor = candidate_corridor
        else:
            assert h == 1, "bars must be horizontal segments"
            fill_amount += area
            bar_components.append(coords)

    assert frame_bbox is not None and corridor is not None, "expected a same-colored frame"
    assert fill_amount > 0, "expected at least one same-colored bar segment"
    return frame_bbox, corridor, fill_amount, bar_components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inject each bar's length into the same-colored cavity corridor from the gap inward.

    Input structure:
    - The background is black.
    - Each foreground color appears as one sealed frame-like component plus one or more horizontal bar segments.
    - Inside each frame is a one-cell-wide black corridor, and exactly one border cell of the frame is missing as the corridor's gap.

    Rule:
    - Erase every horizontal bar segment.
    - For each color, find the frame's black corridor and start at the first corridor cell just inside the unique gap.
    - Follow the corridor's unique path and recolor its first N cells with the frame color, where N is the total length of the same-colored bar segments; leave the gap and the remaining corridor cells black.

    Key insight:
    - The bar encodes only a quantity. The colored result is the corridor-path prefix measured from the gap, not a shape copied from the bar and not a gravity-based fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()

    for color in _colors_present(g, bg):
        frame_bbox, corridor, fill_amount, bar_components = _classify_color_components(g, color, bg)
        assert fill_amount <= len(corridor), "bar length cannot exceed the corridor length"

        for coords in bar_components:
            out[coords[:, 0], coords[:, 1]] = bg

        r0, c0, _r1, _c1 = frame_bbox
        for rr, cc in corridor[:fill_amount]:
            out[r0 + rr, c0 + cc] = color

    return out


def _in_inner_bounds(h: int, w: int, cell: Coord) -> bool:
    r, c = cell
    return 1 <= r <= h - 2 and 1 <= c <= w - 2


def _side_spec(h: int, w: int, side: str) -> tuple[Coord, Coord, Coord]:
    if side == "top":
        c = random.randint(1, w - 2)
        return (1, c), (0, c), (1, 0)
    if side == "bottom":
        c = random.randint(1, w - 2)
        return (h - 2, c), (h - 1, c), (-1, 0)
    if side == "left":
        r = random.randint(1, h - 2)
        return (r, 1), (r, 0), (0, 1)
    if side == "right":
        r = random.randint(1, h - 2)
        return (r, w - 2), (r, w - 1), (0, -1)
    raise ValueError(f"unknown side {side}")


def _can_extend_path(h: int, w: int, used: set[Coord], current: Coord, nxt: Coord) -> bool:
    if not _in_inner_bounds(h, w, nxt) or nxt in used:
        return False
    for nbr in _neighbors4(*nxt):
        if nbr in used and nbr != current:
            return False
    return True


def _dfs_path(h: int, w: int, path: list[Coord], used: set[Coord], target: int, turns: int, require_turn: bool) -> list[Coord] | None:
    if len(path) == target:
        if not require_turn or turns > 0:
            return path.copy()
        return None

    current = path[-1]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    prev_dir = None
    if len(path) >= 2:
        prev = path[-2]
        prev_dir = (current[0] - prev[0], current[1] - prev[1])

    for dr, dc in dirs:
        nxt = (current[0] + dr, current[1] + dc)
        if not _can_extend_path(h, w, used, current, nxt):
            continue
        turn_inc = int(prev_dir is not None and prev_dir != (dr, dc))
        path.append(nxt)
        used.add(nxt)
        result = _dfs_path(h, w, path, used, target, turns + turn_inc, require_turn)
        if result is not None:
            return result
        used.remove(nxt)
        path.pop()
    return None


def _straight_path(h: int, w: int, side: str) -> tuple[Coord, list[Coord]]:
    start, gap, inward = _side_spec(h, w, side)
    max_len = h - 2 if inward[0] != 0 else w - 2
    length = random.randint(3, max_len)
    path = [(start[0] + inward[0] * i, start[1] + inward[1] * i) for i in range(length)]
    return gap, path


def _bent_path(h: int, w: int, side: str) -> tuple[Coord, list[Coord]] | None:
    start, gap, inward = _side_spec(h, w, side)
    second = (start[0] + inward[0], start[1] + inward[1])
    if not _in_inner_bounds(h, w, second):
        return None

    inner_area = (h - 2) * (w - 2)
    min_len = 4
    max_len = min(12, inner_area // 2 + 1)
    if max_len < min_len:
        return None

    lengths = list(range(min_len, max_len + 1))
    random.shuffle(lengths)
    for target in lengths:
        path = [start, second]
        used = {start, second}
        result = _dfs_path(h, w, path, used, target, turns=0, require_turn=True)
        if result is not None:
            return gap, result
    return None


def _build_frame_spec(color: int, bent: bool) -> dict | None:
    for _ in range(120):
        if bent:
            h = random.randint(6, 10)
            w = random.randint(6, 10)
        else:
            h = random.randint(5, 9)
            w = random.randint(5, 9)

        side = random.choice(SIDES)
        corridor_info = _bent_path(h, w, side) if bent else _straight_path(h, w, side)
        if corridor_info is None:
            continue
        gap, corridor = corridor_info
        corridor_len = len(corridor)
        if corridor_len < 3:
            continue

        fill_choices = {1, corridor_len, random.randint(1, corridor_len)}
        if corridor_len >= 4:
            fill_choices.add(random.randint(2, corridor_len - 1))
        fill_len = random.choice(sorted(fill_choices))

        inp_patch = np.full((h, w), color, dtype=int)
        inp_patch[gap] = BG
        for cell in corridor:
            inp_patch[cell] = BG

        if len(_find_components(inp_patch == color)) != 1:
            continue
        try:
            extracted = _extract_corridor(inp_patch, BG)
        except AssertionError:
            continue
        if len(extracted) != corridor_len:
            continue

        out_patch = inp_patch.copy()
        for cell in corridor[:fill_len]:
            out_patch[cell] = color

        return {
            "color": int(color),
            "input_patch": inp_patch,
            "output_patch": out_patch,
            "fill_len": int(fill_len),
            "corridor_len": int(corridor_len),
            "bent": bool(bent),
            "gap_side": side,
        }
    return None


def _boxes_separated(a: BBox, b: BBox, gap: int) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_rect(h: int, w: int, item_h: int, item_w: int, occupied: list[BBox], gap: int, margin: int) -> tuple[int, int] | None:
    row_stop = h - item_h - margin
    col_stop = w - item_w - margin
    if row_stop < margin or col_stop < margin:
        return None

    candidates = [
        (r, c)
        for r in range(margin, row_stop + 1)
        for c in range(margin, col_stop + 1)
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        bbox = (r, c, r + item_h - 1, c + item_w - 1)
        if all(_boxes_separated(bbox, other, gap) for other in occupied):
            return r, c
    return None


def _coverage_ok(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    fg = int(np.count_nonzero(grid != BG))
    if counts[BG] <= int(max(counts[1:])):
        return False
    coverage = fg / float(grid.size)
    return 0.08 <= coverage <= 0.42


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    frame_count = random.choice(GEN_FRAME_COUNT_CHOICES)
    colors = random.sample(COLORS, frame_count)

    bent_flags = [random.random() < 0.65 for _ in range(frame_count)]
    if not any(bent_flags):
        bent_flags[random.randrange(frame_count)] = True
    if frame_count > 1 and all(bent_flags):
        bent_flags[random.randrange(frame_count)] = False

    specs: list[dict] = []
    for color, bent in zip(colors, bent_flags):
        spec = _build_frame_spec(color, bent)
        if spec is None:
            return None
        specs.append(spec)

    occupied: list[BBox] = []
    frame_specs = sorted(specs, key=lambda spec: spec["input_patch"].size, reverse=True)
    for spec in frame_specs:
        patch_h, patch_w = spec["input_patch"].shape
        pos = _place_rect(h, w, patch_h, patch_w, occupied, GEN_MIN_GAP, GEN_FRAME_MARGIN)
        if pos is None:
            return None
        r0, c0 = pos
        spec["frame_pos"] = pos
        occupied.append((r0, c0, r0 + patch_h - 1, c0 + patch_w - 1))

    for spec in sorted(specs, key=lambda spec: spec["fill_len"], reverse=True):
        bar_pos = _place_rect(h, w, 1, spec["fill_len"], occupied, GEN_MIN_GAP, GEN_FRAME_MARGIN)
        if bar_pos is None:
            return None
        r0, c0 = bar_pos
        spec["bar_pos"] = bar_pos
        occupied.append((r0, c0, r0, c0 + spec["fill_len"] - 1))

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    for spec in specs:
        fr, fc = spec["frame_pos"]
        patch_h, patch_w = spec["input_patch"].shape
        inp[fr : fr + patch_h, fc : fc + patch_w] = spec["input_patch"]
        expected[fr : fr + patch_h, fc : fc + patch_w] = spec["output_patch"]

        br, bc = spec["bar_pos"]
        inp[br, bc : bc + spec["fill_len"]] = spec["color"]

    if not _coverage_ok(inp):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        pair = _generate_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid cavity-injection input")
