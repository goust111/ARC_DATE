from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 400


def _gap_side(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top"
    if gc == c1 and r0 < gr < r1:
        return "right"
    if gr == r1 and c0 < gc < c1:
        return "bottom"
    if gc == c0 and r0 < gr < r1:
        return "left"
    return None


def _frame_area(frame: dict) -> int:
    r0, c0, r1, c1 = frame["bbox"]
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _frame_from_color(grid: np.ndarray, color: int) -> dict | None:
    pts = np.argwhere(grid == int(color))
    if len(pts) == 0:
        return None

    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    expected = 2 * h + 2 * w - 5
    if len(pts) != expected:
        return None

    gap = None
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            v = int(grid[r, c])
            if on_border:
                if v == int(color):
                    continue
                if gap is not None or v != BG:
                    return None
                gap = (r, c)
            elif v == int(color):
                return None

    if gap is None:
        return None
    side = _gap_side((r0, c0, r1, c1), gap)
    if side is None:
        return None

    axis = int(gap[1] if side in {"top", "bottom"} else gap[0])
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap": (int(gap[0]), int(gap[1])),
        "side": side,
        "axis": axis,
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != BG)
    frames = []
    for color in colors:
        frame = _frame_from_color(grid, color)
        if frame is not None:
            frames.append(frame)

    assert len(frames) in {2, 3}, "expected 2 or 3 rectangular frames"
    frames.sort(key=_frame_area, reverse=True)

    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        inset = ir0 - or0
        assert inset >= 2, "nested frames need at least one background shell between them"
        assert ic0 - oc0 == inset
        assert or1 - ir1 == inset
        assert oc1 - ic1 == inset

    return frames


def _extract_pattern(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    frame_mask = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        frame_mask |= grid == int(frame["color"])

    mask = (grid != BG) & (~frame_mask)
    pts = np.argwhere(mask)
    assert len(pts) > 0, "expected one outside pattern"
    pr0, pc0 = pts.min(axis=0)
    pr1, pc1 = pts.max(axis=0)
    pr0 = int(pr0)
    pc0 = int(pc0)
    pr1 = int(pr1)
    pc1 = int(pc1)

    block_mask = mask[pr0 : pr1 + 1, pc0 : pc1 + 1]
    assert bool(np.all(block_mask)), "pattern must be a solid rectangle"

    pattern = grid[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    assert 2 <= pattern.shape[0] <= 4 and 2 <= pattern.shape[1] <= 4

    outer = frames[0]
    or0, oc0, or1, oc1 = outer["bbox"]
    gr, gc = outer["gap"]
    side = outer["side"]
    if side == "top":
        assert pr1 == or0 - 1 and pc0 <= gc <= pc1
    elif side == "right":
        assert pc0 == oc1 + 1 and pr0 <= gr <= pr1
    elif side == "bottom":
        assert pr0 == or1 + 1 and pc0 <= gc <= pc1
    else:
        assert pc1 == oc0 - 1 and pr0 <= gr <= pr1

    return pattern


def _start_cell(side: str, axis: int, top: int, left: int, bottom: int, right: int) -> tuple[int, int]:
    if side == "top":
        return top, axis
    if side == "right":
        return axis, right
    if side == "bottom":
        return bottom, axis
    return axis, left


def _rotate_to_start(cells: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    idx = cells.index(start)
    return cells[idx:] + cells[:idx]


def _shell_order(side: str, axis: int, top: int, left: int, bottom: int, right: int) -> list[tuple[int, int]]:
    assert top <= bottom and left <= right
    start = _start_cell(side, axis, top, left, bottom, right)

    if top == bottom and left == right:
        assert start == (top, left)
        return [start]

    if top == bottom:
        if side in {"top", "left"}:
            cells = [(top, c) for c in range(left, right + 1)]
        else:
            cells = [(top, c) for c in range(right, left - 1, -1)]
        return _rotate_to_start(cells, start)

    if left == right:
        if side in {"top", "right"}:
            cells = [(r, left) for r in range(top, bottom + 1)]
        else:
            cells = [(r, left) for r in range(bottom, top - 1, -1)]
        return _rotate_to_start(cells, start)

    cells = [(top, c) for c in range(left, right + 1)]
    cells += [(r, right) for r in range(top + 1, bottom + 1)]
    cells += [(bottom, c) for c in range(right - 1, left - 1, -1)]
    cells += [(r, left) for r in range(bottom - 1, top, -1)]
    return _rotate_to_start(cells, start)


def _paint_shells(out: np.ndarray, side: str, axis: int, bounds: list[tuple[int, int, int, int]], tile: np.ndarray) -> None:
    th, tw = tile.shape
    for depth, (top, left, bottom, right) in enumerate(bounds):
        shell = _shell_order(side, axis, top, left, bottom, right)
        for offset, (r, c) in enumerate(shell):
            assert int(out[r, c]) == BG
            out[r, c] = int(tile[depth % th, offset % tw])


def _annulus_bounds(outer: dict, inner: dict) -> list[tuple[int, int, int, int]]:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, _ir1, _ic1 = inner["bbox"]
    inset = ir0 - or0
    return [
        (or0 + 1 + depth, oc0 + 1 + depth, or1 - 1 - depth, oc1 - 1 - depth)
        for depth in range(inset - 1)
    ]


def _interior_bounds(frame: dict) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    top = r0 + 1
    left = c0 + 1
    bottom = r1 - 1
    right = c1 - 1
    bounds = []
    while top <= bottom and left <= right:
        bounds.append((top, left, bottom, right))
        top += 1
        left += 1
        bottom -= 1
        right -= 1
    return bounds


def _apply_fill(grid: np.ndarray, frames: list[dict], pattern: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for layer, frame in enumerate(frames):
        tile = np.rot90(pattern, -layer)
        if layer + 1 < len(frames):
            bounds = _annulus_bounds(frame, frames[layer + 1])
        else:
            bounds = _interior_bounds(frame)
        _paint_shells(out, frame["side"], frame["axis"], bounds, tile)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular layers by injecting an outside pattern through each gap.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    pattern = _extract_pattern(g, frames)
    return _apply_fill(g, frames, pattern)


def _interior_shell_count(inner_h: int, inner_w: int) -> int:
    return (min(int(inner_h), int(inner_w)) + 1) // 2


def _build_bboxes(inner_h: int, inner_w: int, insets: list[int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = [(0, 0, inner_h + 1, inner_w + 1)]
    for inset in reversed(insets):
        r0, c0, r1, c1 = boxes[0]
        boxes.insert(0, (r0 - inset, c0 - inset, r1 + inset, c1 + inset))
    return boxes


def _gap_choices(bbox: tuple[int, int, int, int], shell_count: int, side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    choices = []
    if side in {"top", "bottom"}:
        row = r0 if side == "top" else r1
        for col in range(c0 + shell_count, c1 - shell_count + 1):
            if c0 < col < c1:
                choices.append((row, col))
    else:
        col = c0 if side == "left" else c1
        for row in range(r0 + shell_count, r1 - shell_count + 1):
            if r0 < row < r1:
                choices.append((row, col))
    return choices


def _random_pattern(colors: list[int]) -> np.ndarray:
    while True:
        ph = random.randint(2, 4)
        pw = random.randint(2, 4)
        palette_size = random.randint(2, min(4, len(colors), ph * pw))
        palette = random.sample(colors, palette_size)
        pattern = np.array(
            [[random.choice(palette) for _ in range(pw)] for _ in range(ph)],
            dtype=int,
        )
        if len(set(map(int, pattern.ravel()))) < 2:
            continue
        if pattern.shape == np.rot90(pattern, -1).shape and np.array_equal(pattern, np.rot90(pattern, -1)):
            continue
        return pattern


def _pattern_bbox(frame: dict, pattern: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    gr, gc = frame["gap"]
    ph, pw = pattern.shape
    side = frame["side"]
    if side == "top":
        offset = random.randint(0, pw - 1)
        pc0 = gc - offset
        return r0 - ph, pc0, r0 - 1, pc0 + pw - 1
    if side == "right":
        offset = random.randint(0, ph - 1)
        pr0 = gr - offset
        return pr0, c1 + 1, pr0 + ph - 1, c1 + pw
    if side == "bottom":
        offset = random.randint(0, pw - 1)
        pc0 = gc - offset
        return r1 + 1, pc0, r1 + ph, pc0 + pw - 1
    offset = random.randint(0, ph - 1)
    pr0 = gr - offset
    return pr0, c0 - pw, pr0 + ph - 1, c0 - 1


def _shift_frame(frame: dict, dr: int, dc: int) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    gr, gc = frame["gap"]
    shifted = dict(frame)
    shifted["bbox"] = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
    shifted["gap"] = (gr + dr, gc + dc)
    if frame["side"] in {"top", "bottom"}:
        shifted["axis"] = int(frame["axis"]) + dc
    else:
        shifted["axis"] = int(frame["axis"]) + dr
    return shifted


def _paint_frame(grid: np.ndarray, frame: dict) -> None:
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    gr, gc = frame["gap"]
    grid[gr, gc] = BG


def _paint_pattern(grid: np.ndarray, pattern: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    assert pattern.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = pattern


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice([2, 3])
    inner_h = random.randint(3, 7)
    inner_w = random.randint(3, 7)
    insets = [random.randint(2, 4) for _ in range(n_frames - 1)]
    boxes = _build_bboxes(inner_h, inner_w, insets)

    color_pool = list(range(1, 10))
    random.shuffle(color_pool)
    frame_colors = color_pool[:n_frames]
    pattern_colors = color_pool[n_frames:]
    pattern = _random_pattern(pattern_colors)

    frames = []
    for idx, bbox in enumerate(boxes):
        shell_count = insets[idx] - 1 if idx < len(insets) else _interior_shell_count(inner_h, inner_w)
        side_options = []
        for side in SIDES:
            gaps = _gap_choices(bbox, shell_count, side)
            if gaps:
                side_options.append((side, gaps))
        if not side_options:
            return None
        side, gaps = random.choice(side_options)
        gap = random.choice(gaps)
        axis = int(gap[1] if side in {"top", "bottom"} else gap[0])
        frames.append(
            {
                "color": int(frame_colors[idx]),
                "bbox": bbox,
                "gap": (int(gap[0]), int(gap[1])),
                "side": side,
                "axis": axis,
            }
        )

    pat_bbox = _pattern_bbox(frames[0], pattern)
    all_boxes = [frame["bbox"] for frame in frames] + [pat_bbox]
    min_r = min(box[0] for box in all_boxes)
    min_c = min(box[1] for box in all_boxes)
    max_r = max(box[2] for box in all_boxes)
    max_c = max(box[3] for box in all_boxes)

    pad_top = random.randint(1, 4)
    pad_bottom = random.randint(1, 4)
    pad_left = random.randint(1, 4)
    pad_right = random.randint(1, 4)
    dr = pad_top - min_r
    dc = pad_left - min_c
    height = max_r - min_r + 1 + pad_top + pad_bottom
    width = max_c - min_c + 1 + pad_left + pad_right

    shifted_frames = [_shift_frame(frame, dr, dc) for frame in frames]
    spr0, spc0, spr1, spc1 = pat_bbox
    shifted_pat_bbox = (spr0 + dr, spc0 + dc, spr1 + dr, spc1 + dc)

    grid = np.full((height, width), BG, dtype=int)
    for frame in shifted_frames:
        _paint_frame(grid, frame)
    _paint_pattern(grid, pattern, shifted_pat_bbox)

    expected = _apply_fill(grid, shifted_frames, pattern)
    if np.array_equal(expected, grid):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
