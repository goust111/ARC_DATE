from __future__ import annotations

import random
from collections import deque

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(points) != expected:
        return False
    pset = set(points)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pset or (r1, c) not in pset:
            return False
    for r in range(r0 + 1, r1):
        if (r, c0) not in pset or (r, c1) not in pset:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(grid == color):
            if _is_rect_frame(pts):
                frames.append({"color": color, "bbox": _bbox(pts)})

    frames.sort(key=lambda item: (_bbox_area(item["bbox"]), item["bbox"][0], item["bbox"][1]))
    assert 3 <= len(frames) <= 5, "expected 3-5 nested frames"

    colors = [int(item["color"]) for item in frames]
    assert len(set(colors)) == len(colors), "frame colors must be unique"

    for i in range(1, len(frames)):
        ir0, ic0, ir1, ic1 = frames[i - 1]["bbox"]
        or0, oc0, or1, oc1 = frames[i]["bbox"]
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "frames must be strictly nested"

    return frames


def _find_legend(
    grid: np.ndarray, bg: int, frame_colors: set[int]
) -> tuple[list[int], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[list[int], list[tuple[int, int]]]] = []

    edge_specs = [
        [(0, c) for c in range(w)],
        [(h - 1, c) for c in range(w)],
        [(r, 0) for r in range(h)],
        [(r, w - 1) for r in range(h)],
    ]
    for coords in edge_specs:
        pts = [(r, c) for r, c in coords if int(grid[r, c]) != bg]
        if not pts:
            continue
        colors = [int(grid[r, c]) for r, c in pts]
        if len(colors) == len(frame_colors) and set(colors) == frame_colors:
            candidates.append((colors, pts))

    assert candidates, "expected one edge legend using all frame colors"
    return candidates[0]


def _paint_ring(
    out: np.ndarray,
    outer_box: tuple[int, int, int, int],
    inner_box: tuple[int, int, int, int] | None,
    color: int,
) -> None:
    r0, c0, r1, c1 = outer_box
    mask = np.zeros(out.shape, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if inner_box is not None:
        ir0, ic0, ir1, ic1 = inner_box
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    out[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each band between nested rectangular frames using an indirect legend-driven color lookup.

    Input structure:
    - The background is black and the main object is one set of 3-5 concentric hollow rectangular frames.
    - Every frame has a distinct color, frame borders are 1 cell thick, and neighboring frames are separated by 1-2 background cells.
    - One grid edge contains a one-dimensional legend whose pixels use exactly the same colors as the frames but in a different order.

    Rule:
    - Detect the nested frames and sort them from inner to outer.
    - Read the legend order along its edge; a frame color maps to the layer whose index matches that color's legend position.
    - Fill each frame's interior band, excluding any more-inner frame and its enclosed area, with the mapped layer color, keep frame borders unchanged, and erase the legend to background.

    Key insight:
    - The legend does not directly name fill colors; it first assigns each frame color a position, and that position points to another frame color by layer depth.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    frames = _find_frames(g, bg)
    frame_colors = {int(item["color"]) for item in frames}
    legend_order, legend_pts = _find_legend(g, bg, frame_colors)

    inner_to_outer_colors = [int(item["color"]) for item in frames]
    index_by_color = {color: idx for idx, color in enumerate(legend_order)}

    out = g.copy()
    for i, frame in enumerate(frames):
        frame_color = int(frame["color"])
        fill_color = inner_to_outer_colors[index_by_color[frame_color]]
        inner_box = None if i == 0 else frames[i - 1]["bbox"]
        _paint_ring(out, frame["bbox"], inner_box, fill_color)

    for r, c in legend_pts:
        out[r, c] = bg

    return out


# --------------- generator ---------------

MAX_TRIES = 300
FRAME_LAYER_RANGE = (3, 5)
FRAME_GAP_RANGE = (1, 2)
OUTER_H_MAX = 27
OUTER_W_MAX = 28
LEGEND_STEP_CHOICES = (1, 2)
LEGEND_SIDES = ("top", "bottom")


def _split_padding(total: int, low_a: int, low_b: int, rng: random.Random) -> tuple[int, int]:
    a = rng.randint(low_a, total - low_b)
    return a, total - a


def _build_boxes(
    outer_box: tuple[int, int, int, int], gaps: list[int]
) -> list[tuple[int, int, int, int]]:
    boxes = [outer_box]
    for gap in gaps:
        r0, c0, r1, c1 = boxes[-1]
        inset = gap + 1
        boxes.append((r0 + inset, c0 + inset, r1 - inset, c1 - inset))
    return boxes


def _legend_permutation(colors: list[int], rng: random.Random) -> list[int]:
    perm = colors[:]
    while True:
        rng.shuffle(perm)
        if perm != colors:
            return perm[:]


def _render_input(
    h: int,
    w: int,
    boxes_outer_to_inner: list[tuple[int, int, int, int]],
    inner_to_outer_colors: list[int],
    legend_side: str,
    legend_start: int,
    legend_step: int,
    legend_order: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)

    for box, color in zip(reversed(boxes_outer_to_inner), inner_to_outer_colors):
        r0, c0, r1, c1 = box
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

    row = 0 if legend_side == "top" else h - 1
    for i, color in enumerate(legend_order):
        grid[row, legend_start + i * legend_step] = color

    return grid


def _expected_output(
    inp: np.ndarray,
    boxes_outer_to_inner: list[tuple[int, int, int, int]],
    inner_to_outer_colors: list[int],
    legend_side: str,
    legend_start: int,
    legend_step: int,
    legend_order: list[int],
) -> np.ndarray:
    out = inp.copy()
    boxes_inner_to_outer = list(reversed(boxes_outer_to_inner))
    index_by_color = {color: idx for idx, color in enumerate(legend_order)}

    for i, (box, frame_color) in enumerate(zip(boxes_inner_to_outer, inner_to_outer_colors)):
        fill_color = inner_to_outer_colors[index_by_color[frame_color]]
        inner_box = None if i == 0 else boxes_inner_to_outer[i - 1]
        _paint_ring(out, box, inner_box, fill_color)

    row = 0 if legend_side == "top" else inp.shape[0] - 1
    for i in range(len(legend_order)):
        out[row, legend_start + i * legend_step] = 0

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_layers = rng.randint(*FRAME_LAYER_RANGE)
    gaps = [rng.randint(*FRAME_GAP_RANGE) for _ in range(n_layers - 1)]
    total_inset = sum(gap + 1 for gap in gaps)

    outer_h_min = 3 + 2 * total_inset
    outer_w_min = 3 + 2 * total_inset
    outer_h = rng.randint(outer_h_min, OUTER_H_MAX)
    outer_w = rng.randint(outer_w_min, OUTER_W_MAX)

    legend_side = rng.choice(LEGEND_SIDES)
    legend_step = rng.choice(LEGEND_STEP_CHOICES)
    legend_w = 1 + (n_layers - 1) * legend_step

    min_h = outer_h + 3
    min_w = max(outer_w + 2, legend_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)

    total_vpad = h - outer_h
    if legend_side == "top":
        top_pad, bottom_pad = _split_padding(total_vpad, 2, 1, rng)
    else:
        top_pad, bottom_pad = _split_padding(total_vpad, 1, 2, rng)
    total_hpad = w - outer_w
    left_pad, right_pad = _split_padding(total_hpad, 1, 1, rng)

    outer_box = (top_pad, left_pad, top_pad + outer_h - 1, left_pad + outer_w - 1)
    boxes_outer_to_inner = _build_boxes(outer_box, gaps)
    inner_box = boxes_outer_to_inner[-1]
    assert inner_box[2] - inner_box[0] + 1 >= 3 and inner_box[3] - inner_box[1] + 1 >= 3

    colors = rng.sample(list(range(1, 10)), n_layers)
    inner_to_outer_colors = colors[:]
    legend_order = _legend_permutation(inner_to_outer_colors, rng)

    legend_start = rng.randint(1, w - legend_w - 1)
    inp = _render_input(
        h,
        w,
        boxes_outer_to_inner,
        inner_to_outer_colors,
        legend_side,
        legend_start,
        legend_step,
        legend_order,
    )
    expected = _expected_output(
        inp,
        boxes_outer_to_inner,
        inner_to_outer_colors,
        legend_side,
        legend_start,
        legend_step,
        legend_order,
    )

    if _mode_color(inp) != 0:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame sample")
