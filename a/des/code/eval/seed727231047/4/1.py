from __future__ import annotations

import random
from collections import Counter

import numpy as np


SIDES = ("top", "right", "bottom", "left")
SIDE_TO_SHIFT = {"top": 0, "right": 1, "bottom": 2, "left": 3}
NON_BG_COLORS = list(range(1, 10))
GEN_RETRIES = 500
PLACEMENT_TRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_palette(grid: np.ndarray, bg: int) -> list[int]:
    row = grid[-1]
    cols = np.flatnonzero(row != bg)
    assert len(cols) == 4, "bottom palette row must contain exactly four colored cells"
    assert all(int(cols[i + 1] - cols[i]) == 2 for i in range(3)), "palette cells must be spaced by one gap"
    palette = [int(row[c]) for c in cols]
    assert len(set(palette)) == 4, "palette colors must be distinct"
    return palette


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _parse_frame(coords: list[tuple[int, int]], color: int) -> dict | None:
    if len(coords) < 7:
        return None
    pts = {(int(r), int(c)) for r, c in coords}
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    border = _border_cells(r0, c0, r1, c1)
    if not pts.issubset(border):
        return None
    missing = list(border - pts)
    if len(missing) != 1 or len(pts) != len(border) - 1:
        return None
    gap = missing[0]
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        side = "top"
    elif gc == c1 and r0 < gr < r1:
        side = "right"
    elif gr == r1 and c0 < gc < c1:
        side = "bottom"
    elif gc == c0 and r0 < gr < r1:
        side = "left"
    else:
        return None
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gr), int(gc)),
        "side": side,
        "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
    }


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components_of_color(grid, color):
            frame = _parse_frame(comp, color)
            if frame is not None:
                frame["id"] = len(frames)
                frames.append(frame)
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1], -f["area"]))
    for idx, frame in enumerate(frames):
        frame["id"] = idx
    return frames


def _contains_box(outer: dict, inner: dict) -> bool:
    r0, c0, r1, c1 = outer["bbox"]
    rr0, cc0, rr1, cc1 = inner["bbox"]
    return r0 < rr0 and c0 < cc0 and rr1 < r1 and cc1 < c1


def _nesting_info(frames: list[dict]) -> tuple[dict[int, int | None], dict[int, int]]:
    parent: dict[int, int | None] = {}
    for child in frames:
        containers = [outer for outer in frames if outer["id"] != child["id"] and _contains_box(outer, child)]
        if not containers:
            parent[child["id"]] = None
            continue
        parent_frame = min(containers, key=lambda f: f["area"])
        parent[child["id"]] = int(parent_frame["id"])

    root_of: dict[int, int] = {}
    for frame in frames:
        cur = frame["id"]
        while parent[cur] is not None:
            cur = int(parent[cur])
        root_of[frame["id"]] = int(cur)
    return parent, root_of


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color nested rectangular regions by depth, then rotate the depth->color
    mapping according to the outermost frame's gap direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)
    palette = _parse_palette(g, bg)
    frames = _extract_frames(g, bg)
    assert frames, "expected at least one gapped rectangular frame"

    _, root_of = _nesting_info(frames)
    frame_by_id = {frame["id"]: frame for frame in frames}
    gap_cells = {frame["gap"] for frame in frames}
    out = g.copy()

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != bg or (r, c) in gap_cells:
                continue
            containing = [
                frame
                for frame in frames
                if frame["bbox"][0] < r < frame["bbox"][2] and frame["bbox"][1] < c < frame["bbox"][3]
            ]
            if not containing:
                continue
            roots = {root_of[frame["id"]] for frame in containing}
            assert len(roots) == 1, "a region cell must belong to exactly one nested group"
            depth = len(containing)
            assert 1 <= depth <= 4, "nesting depth must be between 1 and 4"
            root = frame_by_id[int(next(iter(roots)))]
            shift = SIDE_TO_SHIFT[root["side"]]
            out[r, c] = palette[(depth - 1 - shift) % 4]

    return out


def _choose_gap(bbox: tuple[int, int, int, int], side: str | None = None) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    side = random.choice(SIDES) if side is None else side
    if side == "top":
        return side, (r0, random.randint(c0 + 1, c1 - 1))
    if side == "right":
        return side, (random.randint(r0 + 1, r1 - 1), c1)
    if side == "bottom":
        return side, (r1, random.randint(c0 + 1, c1 - 1))
    return side, (random.randint(r0 + 1, r1 - 1), c0)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = 0


def _build_relative_boxes(depth: int) -> list[tuple[int, int, int, int]]:
    inner_h = random.randint(3, 6)
    inner_w = random.randint(3, 7)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    cur = boxes[0]
    for _ in range(depth - 1):
        top = random.randint(2, 4)
        bottom = random.randint(2, 4)
        left = random.randint(2, 4)
        right = random.randint(2, 4)
        cur = (cur[0] - top, cur[1] - left, cur[2] + bottom, cur[3] + right)
        boxes.append(cur)
    outer = boxes[-1]
    shift_r = -outer[0]
    shift_c = -outer[1]
    normalized = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
        for r0, c0, r1, c1 in reversed(boxes)
    ]
    return normalized


def _shift_boxes(
    boxes: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _rectangles_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _make_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    grid = np.zeros((h, w), dtype=int)

    frame_color, *palette = random.sample(NON_BG_COLORS, 5)
    palette_start = random.randint(0, w - 7)
    palette_cols = [palette_start + 2 * i for i in range(4)]
    for c, color in zip(palette_cols, palette):
        grid[h - 1, c] = color

    num_groups = random.randint(1, 3)
    placed_outer_boxes: list[tuple[int, int, int, int]] = []
    specs = []

    for _ in range(num_groups):
        depth = random.randint(1, 4)
        rel_boxes = _build_relative_boxes(depth)
        outer = rel_boxes[0]
        group_h = outer[2] - outer[0] + 1
        group_w = outer[3] - outer[1] + 1

        max_r0 = h - 2 - group_h
        max_c0 = w - 1 - group_w
        if max_r0 < 1 or max_c0 < 1:
            return None

        placement = None
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            abs_boxes = _shift_boxes(rel_boxes, r0, c0)
            outer_abs = abs_boxes[0]
            if outer_abs[2] >= h - 2:
                continue
            if any(_rectangles_too_close(outer_abs, other) for other in placed_outer_boxes):
                continue
            placement = abs_boxes
            break
        if placement is None:
            return None

        placed_outer_boxes.append(placement[0])
        specs.append({"boxes": placement})

    expected_frames = 0
    for spec in specs:
        boxes = spec["boxes"]
        outer_side = random.choice(SIDES)
        for idx, bbox in enumerate(boxes):
            side = outer_side if idx == 0 else random.choice(SIDES)
            _, gap = _choose_gap(bbox, side)
            _draw_frame(grid, bbox, frame_color, gap)
            expected_frames += 1

    try:
        bg = _most_common_color(grid)
        parsed_palette = _parse_palette(grid, bg)
        parsed_frames = _extract_frames(grid, bg)
        if parsed_palette != palette:
            return None
        if len(parsed_frames) != expected_frames:
            return None
        _ = main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
