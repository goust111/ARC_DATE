from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PARENT_COLORS = (BLUE, RED, GREEN, MAGENTA, ORANGE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == GREY):
        pts = np.asarray(comp, dtype=int)
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        if len(comp) != 2 * (height + width) - 4:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(box[border] == GREY):
            continue
        if height > 2 and width > 2 and np.any(box[1:-1, 1:-1] == GREY):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "border": _border_cells((r0, c0, r1, c1)),
            }
        )
    return frames


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _bbox_gap(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> int:
    ra0, ca0, ra1, ca1 = bbox_a
    rb0, cb0, rb1, cb1 = bbox_b
    return max(_interval_gap(ra0, ra1, rb0, rb1), _interval_gap(ca0, ca1, cb0, cb1))


def _overlay(mask_a: np.ndarray, color_a: int, mask_b: np.ndarray, color_b: int) -> np.ndarray:
    out = np.full(mask_a.shape, BLACK, dtype=int)
    both = mask_a & mask_b
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a
    out[only_a] = int(color_a)
    out[only_b] = int(color_b)
    out[both] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine each connected parent pair inside the matching offspring frame.

    The separator row, grey frames, azure connectors, and two-pixel color markers
    are all read directly from the grid. The offspring inherits each parent's
    pixels at matching coordinates; overlaps become yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows = np.where(np.all(g == MAROON, axis=1))[0]
    assert len(sep_rows) == 1
    sep = int(sep_rows[0])

    frames = _find_frames(g)
    parent_frames = [frame for frame in frames if frame["bbox"][2] < sep]
    child_frames = [frame for frame in frames if frame["bbox"][0] > sep]
    assert parent_frames and child_frames

    parent_by_color: dict[int, dict] = {}
    for frame in parent_frames:
        r0, c0, r1, c1 = frame["inner_bbox"]
        inner = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, inner.reshape(-1))) - {BLACK})
        assert len(colors) == 1
        color = int(colors[0])
        assert color not in parent_by_color
        frame["color"] = color
        frame["mask"] = inner == color
        parent_by_color[color] = frame

    pair_by_colors: dict[tuple[int, int], tuple[dict, dict]] = {}
    for comp in _components(g == AZURE):
        touched: list[dict] = []
        for frame in parent_frames:
            border = frame["border"]
            if any((nr, nc) in border for r, c in comp for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))):
                touched.append(frame)
        assert len(touched) in (0, 2)
        if len(touched) == 2:
            key = tuple(sorted((int(touched[0]["color"]), int(touched[1]["color"]))))
            assert key not in pair_by_colors
            pair_by_colors[key] = (touched[0], touched[1])

    assert pair_by_colors

    marker_mask = np.zeros_like(g, dtype=bool)
    lower = g[sep + 1 :]
    marker_mask[sep + 1 :] = ~np.isin(lower, np.array([BLACK, GREY, AZURE, MAROON], dtype=int))
    markers: list[dict] = []
    for comp in _components(marker_mask):
        assert len(comp) == 2
        colors = sorted(set(int(g[r, c]) for r, c in comp))
        assert len(colors) == 2
        pts = np.asarray(comp, dtype=int)
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        markers.append({"bbox": (r0, c0, r1, c1), "pair": tuple(colors)})

    out = g.copy()
    assert len(child_frames) == len(pair_by_colors) == len(markers)
    used_markers: set[int] = set()
    for frame in child_frames:
        candidates = [
            (idx, _bbox_gap(frame["bbox"], marker["bbox"]))
            for idx, marker in enumerate(markers)
            if idx not in used_markers and _bbox_gap(frame["bbox"], marker["bbox"]) <= 1
        ]
        assert len(candidates) == 1
        marker_idx = candidates[0][0]
        used_markers.add(marker_idx)

        pair_key = markers[marker_idx]["pair"]
        assert pair_key in pair_by_colors
        parent_a, parent_b = pair_by_colors[pair_key]
        assert parent_a["mask"].shape == parent_b["mask"].shape

        r0, c0, r1, c1 = frame["inner_bbox"]
        target_shape = (r1 - r0 + 1, c1 - c0 + 1)
        assert parent_a["mask"].shape == target_shape
        out[r0 : r1 + 1, c0 : c1 + 1] = _overlay(
            parent_a["mask"],
            int(parent_a["color"]),
            parent_b["mask"],
            int(parent_b["color"]),
        )

    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top, left : left + width] = GREY
    grid[top + height - 1, left : left + width] = GREY
    grid[top : top + height, left] = GREY
    grid[top : top + height, left + width - 1] = GREY


def _random_connected_cells(
    rng: random.Random,
    height: int,
    width: int,
    size: int,
) -> set[tuple[int, int]] | None:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    frontier = set()
    for dr, dc in DIR4:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))
    while len(cells) < size and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in DIR4:
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells if len(cells) == size else None


def _grow_attached_cells(
    rng: random.Random,
    height: int,
    width: int,
    base: set[tuple[int, int]],
    size: int,
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    cells: set[tuple[int, int]] = set()
    frontier = set()
    for r, c in base:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            cell = (nr, nc)
            if 0 <= nr < height and 0 <= nc < width and cell not in forbidden:
                frontier.add(cell)
    while len(cells) < size and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in forbidden or cell in cells:
            continue
        cells.add(cell)
        for dr, dc in DIR4:
            nr = cell[0] + dr
            nc = cell[1] + dc
            nxt = (nr, nc)
            if 0 <= nr < height and 0 <= nc < width and nxt not in forbidden and nxt not in cells:
                frontier.add(nxt)
    return cells if len(cells) == size else None


def _make_pair_masks(rng: random.Random, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    area = height * width
    for _ in range(100):
        overlap_size = rng.randint(2, min(6, area - 2))
        max_extra = min(5, area - overlap_size - 1)
        if max_extra < 1:
            continue
        extra_a = rng.randint(1, max_extra)
        extra_b_limit = min(5, area - overlap_size - extra_a)
        if extra_b_limit < 1:
            continue
        extra_b = rng.randint(1, extra_b_limit)

        overlap = _random_connected_cells(rng, height, width, overlap_size)
        if overlap is None:
            continue
        only_a = _grow_attached_cells(rng, height, width, overlap, extra_a, overlap)
        if only_a is None:
            continue
        only_b = _grow_attached_cells(rng, height, width, overlap, extra_b, overlap | only_a)
        if only_b is None:
            continue

        mask_a = np.zeros((height, width), dtype=bool)
        mask_b = np.zeros((height, width), dtype=bool)
        for r, c in overlap | only_a:
            mask_a[r, c] = True
        for r, c in overlap | only_b:
            mask_b[r, c] = True
        return mask_a, mask_b
    raise RuntimeError("failed to generate parent masks")


def _child_structure(frame_h: int, frame_w: int, side: str) -> dict:
    if side == "left":
        frame_top = 0
        frame_left = 2
        marker = [(frame_h // 2 - 1, 0), (frame_h // 2, 0)]
        return {
            "height": frame_h,
            "width": frame_w + 2,
            "frame_top": frame_top,
            "frame_left": frame_left,
            "marker": marker,
        }
    if side == "right":
        frame_top = 0
        frame_left = 0
        marker = [(frame_h // 2 - 1, frame_w + 1), (frame_h // 2, frame_w + 1)]
        return {
            "height": frame_h,
            "width": frame_w + 2,
            "frame_top": frame_top,
            "frame_left": frame_left,
            "marker": marker,
        }
    if side == "top":
        frame_top = 2
        frame_left = 0
        mid = frame_w // 2 - 1
        marker = [(0, mid), (0, mid + 1)]
        return {
            "height": frame_h + 2,
            "width": frame_w,
            "frame_top": frame_top,
            "frame_left": frame_left,
            "marker": marker,
        }
    frame_top = 0
    frame_left = 0
    mid = frame_w // 2 - 1
    marker = [(frame_h + 1, mid), (frame_h + 1, mid + 1)]
    return {
        "height": frame_h + 2,
        "width": frame_w,
        "frame_top": frame_top,
        "frame_left": frame_left,
        "marker": marker,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = 1 if rng.random() < 0.35 else 2
    colors = rng.sample(PARENT_COLORS, 2 * pair_count)
    pairs: list[dict] = []
    for idx in range(pair_count):
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(4, 7)
        mask_a, mask_b = _make_pair_masks(rng, inner_h, inner_w)
        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        gap = rng.randint(2, 4)
        orient = rng.choice(("horizontal", "vertical"))
        if orient == "horizontal":
            top_h = frame_h
            top_w = frame_w * 2 + gap
            parents = [(0, 0), (0, frame_w + gap)]
            line = [(1 + inner_h // 2, frame_w + i) for i in range(gap)]
        else:
            top_h = frame_h * 2 + gap
            top_w = frame_w
            parents = [(0, 0), (frame_h + gap, 0)]
            line = [(frame_h + i, 1 + inner_w // 2) for i in range(gap)]

        child_side = rng.choice(("left", "right", "top", "bottom"))
        child = _child_structure(frame_h, frame_w, child_side)
        marker_colors = (color_a, color_b) if rng.random() < 0.5 else (color_b, color_a)

        pairs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "frame_h": frame_h,
                "frame_w": frame_w,
                "mask_a": mask_a,
                "mask_b": mask_b,
                "color_a": color_a,
                "color_b": color_b,
                "top_h": top_h,
                "top_w": top_w,
                "parents": parents,
                "line": line,
                "child": child,
                "marker_colors": marker_colors,
            }
        )

    top_order = list(range(pair_count))
    bottom_order = list(range(pair_count))
    rng.shuffle(top_order)
    rng.shuffle(bottom_order)

    top_positions: dict[int, tuple[int, int]] = {}
    cur_y = rng.randint(1, 3)
    max_w = 0
    for idx in top_order:
        left = rng.randint(1, 4)
        top_positions[idx] = (cur_y, left)
        cur_y += pairs[idx]["top_h"] + rng.randint(1, 3)
        max_w = max(max_w, left + pairs[idx]["top_w"])

    sep = cur_y
    child_positions: dict[int, tuple[int, int]] = {}
    cur_y = sep + 1 + rng.randint(1, 3)
    for idx in bottom_order:
        left = rng.randint(1, 4)
        child_positions[idx] = (cur_y, left)
        cur_y += pairs[idx]["child"]["height"] + rng.randint(1, 3)
        max_w = max(max_w, left + pairs[idx]["child"]["width"])

    height = cur_y + rng.randint(1, 3)
    width = max_w + rng.randint(1, 4)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[sep, :] = MAROON

    for idx, pair in enumerate(pairs):
        top, left = top_positions[idx]
        (pa_top, pa_left), (pb_top, pb_left) = pair["parents"]

        _draw_frame(grid, top + pa_top, left + pa_left, pair["frame_h"], pair["frame_w"])
        _draw_frame(grid, top + pb_top, left + pb_left, pair["frame_h"], pair["frame_w"])

        a_inner = grid[
            top + pa_top + 1 : top + pa_top + 1 + pair["inner_h"],
            left + pa_left + 1 : left + pa_left + 1 + pair["inner_w"],
        ]
        b_inner = grid[
            top + pb_top + 1 : top + pb_top + 1 + pair["inner_h"],
            left + pb_left + 1 : left + pb_left + 1 + pair["inner_w"],
        ]
        a_inner[pair["mask_a"]] = pair["color_a"]
        b_inner[pair["mask_b"]] = pair["color_b"]

        for dr, dc in pair["line"]:
            grid[top + dr, left + dc] = AZURE

    for idx, pair in enumerate(pairs):
        struct = pair["child"]
        top, left = child_positions[idx]
        frame_top = top + struct["frame_top"]
        frame_left = left + struct["frame_left"]
        _draw_frame(grid, frame_top, frame_left, pair["frame_h"], pair["frame_w"])
        for (mr, mc), color in zip(struct["marker"], pair["marker_colors"]):
            grid[top + mr, left + mc] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
