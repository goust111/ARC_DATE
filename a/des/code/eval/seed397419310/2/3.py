from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
FRAME_COLOR_POOL = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 300
SHAPE_RETRIES = 80


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _legend_colors(grid: np.ndarray) -> list[int]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "missing grey legend border"
    r0, c0, r1, c1 = _bbox_from_coords(grey_coords)
    assert c1 - c0 == 2, "legend must be three columns wide"
    legend_h = r1 - r0 + 1
    assert 3 <= legend_h <= 5, "legend must have 3-5 entries"
    assert np.count_nonzero(grid == GREY) == 2 * legend_h, "grey should appear only in the legend border"
    assert np.all(grid[r0 : r1 + 1, c0] == GREY) and np.all(grid[r0 : r1 + 1, c1] == GREY)

    colors = list(map(int, grid[r0 : r1 + 1, c0 + 1]))
    assert len(set(colors)) == legend_h, "legend colors must be unique"
    assert all(color not in (BLACK, GREY, AZURE) for color in colors), "legend colors must be non-special"
    return colors


def _is_perfect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _detect_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.flatten())) - {BLACK, GREY, AZURE}):
        for coords in _components_of_color(grid, color):
            if _is_perfect_frame(coords):
                r0, c0, r1, c1 = _bbox_from_coords(coords)
                frames.append(
                    {
                        "color": int(color),
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                    }
                )
    frames.sort(key=lambda item: (-int(item["area"]), item["bbox"]))
    assert 2 <= len(frames) <= 4, "expected one nested chain of 2-4 frames"
    for outer, inner in zip(frames, frames[1:]):
        ro0, co0, ro1, co1 = outer["bbox"]  # type: ignore[index]
        ri0, ci0, ri1, ci1 = inner["bbox"]  # type: ignore[index]
        assert ri0 > ro0 and ci0 > co0 and ri1 < ro1 and ci1 < co1, "frames must be strictly nested"
    return frames


def _apply_ops(content: np.ndarray, frame_colors: list[int], legend: list[int]) -> np.ndarray:
    pos = {color: i + 1 for i, color in enumerate(legend)}
    out = np.asarray(content, dtype=int).copy()
    for color in frame_colors:
        idx = pos[int(color)]
        out = np.fliplr(out) if idx % 2 == 1 else np.flipud(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    In layered nested colored frames, the frame color's position in the left legend determines
    whether that layer flips its interior horizontally or vertically, accumulated from outer to inner.
    """
    g = np.asarray(grid, dtype=int)
    legend = _legend_colors(g)
    frames = _detect_frames(g)

    inner = frames[-1]
    r0, c0, r1, c1 = inner["bbox"]  # type: ignore[index]
    content = g[r0 + 1 : r1, c0 + 1 : c1]
    transformed = _apply_ops(content, [int(frame["color"]) for frame in frames], legend)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    inner_color = int(inner["color"])
    out[0, :] = inner_color
    out[-1, :] = inner_color
    out[:, 0] = inner_color
    out[:, -1] = inner_color
    out[1:-1, 1:-1] = transformed
    return out


def _random_connected_mask(h: int, w: int, target: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    pts = np.asarray(sorted(cells), dtype=int)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    return _bbox_from_coords(coords)


def _random_shape(h: int, w: int) -> np.ndarray | None:
    area = h * w
    lo = max(4, int(np.ceil(area * 0.35)))
    hi = max(lo, area - 2)
    for _ in range(SHAPE_RETRIES):
        target = random.randint(lo, hi)
        mask = _random_connected_mask(h, w, target)
        if mask is None:
            continue
        if _tight_bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue
        return mask
    return None


def _non_identity_ops(depth: int) -> list[str] | None:
    for _ in range(SHAPE_RETRIES):
        ops = [random.choice(("H", "V")) for _ in range(depth)]
        h_count = sum(op == "H" for op in ops) % 2
        v_count = sum(op == "V" for op in ops) % 2
        if h_count == 0 and v_count == 0:
            continue
        return ops
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _make_expected(legend: list[int], frame_colors: list[int], inner_color: int, content: np.ndarray) -> np.ndarray:
    transformed = _apply_ops(content, frame_colors, legend)
    out = np.full((content.shape[0] + 2, content.shape[1] + 2), BLACK, dtype=int)
    out[0, :] = inner_color
    out[-1, :] = inner_color
    out[:, 0] = inner_color
    out[:, -1] = inner_color
    out[1:-1, 1:-1] = transformed
    return out


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.randint(3, 5)
    depth = random.randint(2, 4)
    legend = random.sample(FRAME_COLOR_POOL, legend_len)
    odd_colors = [color for i, color in enumerate(legend, start=1) if i % 2 == 1]
    even_colors = [color for i, color in enumerate(legend, start=1) if i % 2 == 0]
    ops = _non_identity_ops(depth)
    if ops is None:
        return None
    frame_colors = [random.choice(odd_colors if op == "H" else even_colors) for op in ops]

    shape_h = random.randint(3, 5)
    shape_w = random.randint(3, 5)
    shape = _random_shape(shape_h, shape_w)
    if shape is None:
        return None

    inner_h = random.randint(shape_h, min(shape_h + 3, 7))
    inner_w = random.randint(shape_w, min(shape_w + 3, 7))
    sr = random.randint(0, inner_h - shape_h)
    sc = random.randint(0, inner_w - shape_w)
    content = np.full((inner_h, inner_w), BLACK, dtype=int)
    content[sr : sr + shape_h, sc : sc + shape_w][shape] = AZURE

    boxes = [(0, 0, inner_h + 1, inner_w + 1)]
    cur_h = inner_h + 2
    cur_w = inner_w + 2
    for _ in range(depth - 1):
        top_gap = random.randint(1, 2)
        bottom_gap = random.randint(1, 2)
        left_gap = random.randint(1, 2)
        right_gap = random.randint(1, 2)
        shift_r = top_gap + 1
        shift_c = left_gap + 1
        boxes = [
            (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
            for r0, c0, r1, c1 in boxes
        ]
        cur_h = cur_h + top_gap + bottom_gap + 2
        cur_w = cur_w + left_gap + right_gap + 2
        boxes.insert(0, (0, 0, cur_h - 1, cur_w - 1))

    outer_h = cur_h
    outer_w = cur_w
    legend_top = random.randint(0, 2)
    legend_left = random.randint(0, 1)
    main_top = random.randint(0, 2)
    main_left = legend_left + 3 + random.randint(2, 4)
    bottom_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)
    grid_h = max(legend_top + legend_len, main_top + outer_h) + bottom_pad
    grid_w = main_left + outer_w + right_pad

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[legend_top : legend_top + legend_len, legend_left] = GREY
    grid[legend_top : legend_top + legend_len, legend_left + 2] = GREY
    for i, color in enumerate(legend):
        grid[legend_top + i, legend_left + 1] = int(color)

    for bbox, color in zip(boxes, frame_colors):
        r0, c0, r1, c1 = bbox
        _draw_frame(grid, (main_top + r0, main_left + c0, main_top + r1, main_left + c1), int(color))

    inner_r0, inner_c0, inner_r1, inner_c1 = boxes[-1]
    ir0 = main_top + inner_r0 + 1
    ic0 = main_left + inner_c0 + 1
    grid[ir0 : ir0 + inner_h, ic0 : ic0 + inner_w] = content

    expected = _make_expected(legend, frame_colors, frame_colors[-1], content)
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
