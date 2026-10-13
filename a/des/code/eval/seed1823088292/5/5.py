from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MIN = 12
GRID_MAX = 20
LAYER_MIN = 3
LAYER_MAX = 5
BOTTOM_SIDE_MIN = 7
BOTTOM_SIDE_MAX = 11
MAX_MARGIN = 2
GEN_ATTEMPTS = 600


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _rect_output(order: list[int], info: dict[int, dict[str, int]], bg: int) -> np.ndarray:
    heights = [info[color]["h"] for color in order]
    widths = [info[color]["w"] for color in order]
    out_h = max(heights)
    out_w = sum(widths) + max(0, len(order) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for idx, color in enumerate(order):
        h = info[color]["h"]
        w = info[color]["w"]
        out[:h, col : col + w] = color
        col += w
        if idx + 1 < len(order):
            out[:, col] = SEP
            col += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each overlapped solid rectangle from its visible bounding box and
    arrange the rectangles from bottom layer to top layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors

    info: dict[int, dict[str, int]] = {}
    for color in colors:
        mask = g == color
        r0, c0, r1, c1 = _bbox_of(mask)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        flat_first = int(np.flatnonzero(mask)[0])
        info[color] = {
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "h": h,
            "w": w,
            "visible": int(mask.sum()),
            "full": int(h * w),
            "first": flat_first,
        }

    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for lower in colors:
        r0 = info[lower]["r0"]
        c0 = info[lower]["c0"]
        r1 = info[lower]["r1"]
        c1 = info[lower]["c1"]
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        blockers = {int(color) for color in np.unique(window)}
        blockers.discard(bg)
        blockers.discard(lower)
        for upper in blockers:
            if upper not in above[lower]:
                above[lower].add(upper)
                indegree[upper] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert ready, "layer relation must be acyclic"
        ready.sort(
            key=lambda color: (
                -info[color]["full"],
                info[color]["r0"],
                info[color]["c0"],
                info[color]["first"],
            )
        )
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for upper in above[color]:
            indegree[upper] -= 1

    return _rect_output(order, info, bg)


def _visible_bbox_preserved(h: int, w: int, t: int, b: int, l: int, r: int) -> bool:
    inner_h = h - t - b
    inner_w = w - l - r
    if inner_h < 2 or inner_w < 2:
        return False
    if t + b + l + r == 0:
        return False

    visible = np.ones((h, w), dtype=bool)
    visible[t : t + inner_h, l : l + inner_w] = False
    return bool(
        visible[0, :].any()
        and visible[-1, :].any()
        and visible[:, 0].any()
        and visible[:, -1].any()
    )


def _margin_candidates(h: int, w: int, mode: str) -> list[tuple[int, int, int, int]]:
    t_max = min(MAX_MARGIN, max(0, h - 2))
    b_max = min(MAX_MARGIN, max(0, h - 2))
    l_max = min(MAX_MARGIN, max(0, w - 2))
    r_max = min(MAX_MARGIN, max(0, w - 2))
    candidates: list[tuple[int, int, int, int]] = []

    for t in range(t_max + 1):
        for b in range(b_max + 1):
            for l in range(l_max + 1):
                for r in range(r_max + 1):
                    if not _visible_bbox_preserved(h, w, t, b, l, r):
                        continue
                    is_frame = min(t, b, l, r) >= 1
                    is_hstrip = t >= 1 and b >= 1 and l == 0 and r == 0
                    is_vstrip = l >= 1 and r >= 1 and t == 0 and b == 0
                    is_touch = 0 in (t, b, l, r) and not (is_frame or is_hstrip or is_vstrip)

                    if mode == "frame" and is_frame:
                        candidates.append((t, b, l, r))
                    elif mode == "hstrip" and is_hstrip:
                        candidates.append((t, b, l, r))
                    elif mode == "vstrip" and is_vstrip:
                        candidates.append((t, b, l, r))
                    elif mode == "touch" and is_touch:
                        candidates.append((t, b, l, r))

    return candidates


def _build_chain(n_layers: int) -> list[tuple[int, int, int, int]] | None:
    min_side = max(BOTTOM_SIDE_MIN, n_layers + 2)
    if min_side > BOTTOM_SIDE_MAX:
        return None

    bottom_h = random.randint(min_side, BOTTOM_SIDE_MAX)
    bottom_w = random.randint(min_side, BOTTOM_SIDE_MAX)
    boxes = [(0, 0, bottom_h - 1, bottom_w - 1)]

    for _ in range(n_layers - 1):
        r0, c0, r1, c1 = boxes[-1]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        by_mode = {
            mode: _margin_candidates(h, w, mode)
            for mode in ("frame", "hstrip", "vstrip", "touch")
        }
        valid_modes = [mode for mode, choices in by_mode.items() if choices]
        if not valid_modes:
            return None
        mode = random.choice(valid_modes)
        t, b, l, r = random.choice(by_mode[mode])
        boxes.append((r0 + t, c0 + l, r1 - b, c1 - r))

    return boxes


def _place_chain(
    boxes: list[tuple[int, int, int, int]]
) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    bottom_r0, bottom_c0, bottom_r1, bottom_c1 = boxes[0]
    bottom_h = bottom_r1 - bottom_r0 + 1
    bottom_w = bottom_c1 - bottom_c0 + 1
    bottom_area = bottom_h * bottom_w

    valid_shapes = [
        (h, w)
        for h in range(max(GRID_MIN, bottom_h + 2), GRID_MAX + 1)
        for w in range(max(GRID_MIN, bottom_w + 2), GRID_MAX + 1)
        if h * w > 2 * bottom_area
    ]
    if not valid_shapes:
        return None

    grid_h, grid_w = random.choice(valid_shapes)
    dr = random.randint(1, grid_h - bottom_h - 1)
    dc = random.randint(1, grid_w - bottom_w - 1)
    shifted = [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]
    return grid_h, grid_w, shifted


def _render_input(
    grid_h: int,
    grid_w: int,
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, boxes, strict=False):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _expected_output(
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    info: dict[int, dict[str, int]] = {}
    for color, (r0, c0, r1, c1) in zip(colors, boxes, strict=False):
        info[color] = {
            "h": int(r1 - r0 + 1),
            "w": int(c1 - c0 + 1),
        }
    return _rect_output(colors, info, BG)


def _visible_bboxes_match(grid: np.ndarray, boxes: list[tuple[int, int, int, int]], colors: list[int]) -> bool:
    for color, expected_box in zip(colors, boxes, strict=False):
        mask = grid == color
        if not mask.any():
            return False
        if _bbox_of(mask) != expected_box:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.randint(LAYER_MIN, LAYER_MAX)
    colors = random.sample(PALETTE, n_layers)
    boxes = _build_chain(n_layers)
    if boxes is None:
        return None

    placed = _place_chain(boxes)
    if placed is None:
        return None
    grid_h, grid_w, shifted = placed
    grid = _render_input(grid_h, grid_w, shifted, colors)

    if not _visible_bboxes_match(grid, shifted, colors):
        return None

    expected = _expected_output(shifted, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
