from __future__ import annotations

import numpy as np


FRAME_COLOR = 5
PIECE_COLORS = tuple(color for color in range(1, 10) if color != FRAME_COLOR)
CORNERS = ("br", "bl", "tr", "tl")

# Generator configuration.
PIECE_COUNT_RANGE = (3, 4)
RECT_HEIGHT_RANGE = (3, 5)
RECT_WIDTH_RANGE = (3, 5)
OVERLAP_SIZE_RANGE = (1, 2)
INNER_MARGIN_RANGE = (1, 3)
OUTER_MARGIN_TOTAL_RANGE = (0, 4)
MAX_UNION_SPAN = 20
GENERATION_RETRY_LIMIT = 500

_RNG = np.random.default_rng()


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.asarray(coords, dtype=int)
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _intersection_box(
    a: dict[str, int], b: dict[str, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(int(a["r"]), int(b["r"]))
    c0 = max(int(a["c"]), int(b["c"]))
    r1 = min(int(a["r"]) + int(a["h"]) - 1, int(b["r"]) + int(b["h"]) - 1)
    c1 = min(int(a["c"]) + int(a["w"]) - 1, int(b["c"]) + int(b["w"]) - 1)
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _draw_frame(
    grid: np.ndarray, bbox: tuple[int, int, int, int], color: int
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == 0:
            continue
        coords = np.argwhere(grid == color)
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        rel = coords - np.array([r0, c0], dtype=int)
        on_border = (
            (rel[:, 0] == 0)
            | (rel[:, 0] == h - 1)
            | (rel[:, 1] == 0)
            | (rel[:, 1] == w - 1)
        )
        if int(coords.shape[0]) == 2 * h + 2 * w - 4 and bool(np.all(on_border)):
            candidates.append((color, (int(r0), int(c0), int(r1), int(c1))))

    assert len(candidates) == 1, "expected exactly one rectangular frame"
    return candidates[0]


def _recover_rectangles(inner: np.ndarray) -> list[dict[str, int]]:
    rects = []
    for color in np.unique(inner):
        color = int(color)
        if color == 0:
            continue
        coords = np.argwhere(inner == color)
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = h * w
        rects.append(
            {
                "color": color,
                "r": int(r0),
                "c": int(c0),
                "h": h,
                "w": w,
                "visible": int(coords.shape[0]),
                "hidden": int(area - coords.shape[0]),
            }
        )
    assert rects, "expected at least one interior rectangle"
    return rects


def _stack_order(inner: np.ndarray, rects: list[dict[str, int]]) -> list[dict[str, int]]:
    n = len(rects)
    above_to_below = [set() for _ in range(n)]
    indegree = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            inter = _intersection_box(rects[i], rects[j])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            patch = inner[r0 : r1 + 1, c0 : c1 + 1]
            sees_i = bool(np.any(patch == int(rects[i]["color"])))
            sees_j = bool(np.any(patch == int(rects[j]["color"])))
            assert sees_i != sees_j, "overlap must reveal exactly one upper rectangle"
            if sees_i:
                above_to_below[i].add(j)
                indegree[j] += 1
            else:
                above_to_below[j].add(i)
                indegree[i] += 1

    order: list[int] = []
    ready = [idx for idx, deg in enumerate(indegree) if deg == 0]
    while ready:
        assert len(ready) == 1, "stack order must be uniquely determined"
        idx = ready.pop()
        order.append(idx)
        for nxt in sorted(above_to_below[idx]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    assert len(order) == n, "overlap relations must form a complete acyclic order"
    return [rects[idx] for idx in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden rectangles and line them up by cover depth with spacing taken from hidden area.

    Input structure:
    - The background is black.
    - A single gray one-pixel rectangular frame encloses the meaningful region.
    - Inside the frame, each non-black color belongs to one solid axis-aligned rectangle, but lower rectangles may be partially hidden by higher ones.

    Rule:
    - Recover each rectangle's full size from the bbox of its visible pixels.
    - For every pair of recovered rectangles whose bboxes overlap, inspect that overlap zone: the color visible there is the upper rectangle, which yields the full top-to-bottom stacking order.
    - Count how many pixels each rectangle has lost to higher layers.
    - Output a new black grid with the recovered full rectangles arranged left to right from top layer to bottom layer, top-aligned, and with the gap after each rectangle equal to that rectangle's hidden-pixel count.

    Key insight:
    - The bbox of each visible fragment restores its original rectangle, and the visible color inside each bbox intersection reveals both depth order and the spacing value encoded as hidden area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color, (fr0, fc0, fr1, fc1) = _find_frame(g)
    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    assert frame_color == FRAME_COLOR or FRAME_COLOR not in inner, "frame color must stay distinct"

    rects = _recover_rectangles(inner)
    ordered = _stack_order(inner, rects)

    out_h = max(int(rect["h"]) for rect in ordered)
    out_w = sum(int(rect["w"]) for rect in ordered) + sum(
        int(rect["hidden"]) for rect in ordered[:-1]
    )
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for idx, rect in enumerate(ordered):
        h = int(rect["h"])
        w = int(rect["w"])
        out[0:h, col : col + w] = int(rect["color"])
        col += w
        if idx + 1 < len(ordered):
            col += int(rect["hidden"])

    return out


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _sample_colors(count: int) -> list[int]:
    picks = _RNG.choice(np.array(PIECE_COLORS, dtype=int), size=int(count), replace=False)
    return [int(color) for color in picks]


def _place_corner_overlap(
    prev: dict[str, int], h: int, w: int, corner: str, overlap_h: int, overlap_w: int
) -> tuple[int, int]:
    if corner == "br":
        return int(prev["r"] + prev["h"] - overlap_h), int(prev["c"] + prev["w"] - overlap_w)
    if corner == "bl":
        return int(prev["r"] + prev["h"] - overlap_h), int(prev["c"] - (w - overlap_w))
    if corner == "tr":
        return int(prev["r"] - (h - overlap_h)), int(prev["c"] + prev["w"] - overlap_w)
    assert corner == "tl"
    return int(prev["r"] - (h - overlap_h)), int(prev["c"] - (w - overlap_w))


def _normalize_rects(rects: list[dict[str, int]]) -> list[dict[str, int]]:
    min_r = min(int(rect["r"]) for rect in rects)
    min_c = min(int(rect["c"]) for rect in rects)
    normalized = []
    for rect in rects:
        normalized.append(
            {
                "r": int(rect["r"] - min_r),
                "c": int(rect["c"] - min_c),
                "h": int(rect["h"]),
                "w": int(rect["w"]),
                "color": int(rect["color"]),
            }
        )
    return normalized


def _scene_shape(rects: list[dict[str, int]]) -> tuple[int, int]:
    height = max(int(rect["r"] + rect["h"]) for rect in rects)
    width = max(int(rect["c"] + rect["w"]) for rect in rects)
    return height, width


def _render_scene(rects: list[dict[str, int]]) -> np.ndarray:
    h, w = _scene_shape(rects)
    scene = np.zeros((h, w), dtype=int)
    for rect in rects:
        r = int(rect["r"])
        c = int(rect["c"])
        scene[r : r + int(rect["h"]), c : c + int(rect["w"])] = int(rect["color"])
    return scene


def _valid_chain(rects: list[dict[str, int]]) -> bool:
    n = len(rects)
    for i in range(n):
        for j in range(i + 1, n):
            inter = _intersection_box(rects[i], rects[j])
            if j == i + 1:
                if inter is None:
                    return False
                r0, c0, r1, c1 = inter
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if not (1 <= area <= 4):
                    return False
            elif inter is not None:
                return False

    scene = _render_scene(rects)
    for rect in rects:
        coords = np.argwhere(scene == int(rect["color"]))
        if coords.size == 0:
            return False
        r0, c0, r1, c1 = _bbox(coords)
        full_bbox = (
            int(rect["r"]),
            int(rect["c"]),
            int(rect["r"] + rect["h"] - 1),
            int(rect["c"] + rect["w"] - 1),
        )
        if (r0, c0, r1, c1) != full_bbox:
            return False

    union_h, union_w = _scene_shape(rects)
    return union_h <= MAX_UNION_SPAN and union_w <= MAX_UNION_SPAN


def _expected_output(rects: list[dict[str, int]]) -> np.ndarray:
    scene = _render_scene(rects)
    ordered = list(reversed(rects))
    out_h = max(int(rect["h"]) for rect in ordered)
    out_w = sum(int(rect["w"]) for rect in ordered)

    hidden_counts = []
    for rect in ordered:
        visible = int(np.count_nonzero(scene == int(rect["color"])))
        hidden = int(rect["h"] * rect["w"] - visible)
        hidden_counts.append(hidden)
    out_w += sum(hidden_counts[:-1])

    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for idx, rect in enumerate(ordered):
        h = int(rect["h"])
        w = int(rect["w"])
        out[0:h, col : col + w] = int(rect["color"])
        col += w
        if idx + 1 < len(ordered):
            col += int(hidden_counts[idx])
    return out


def _split_margin(total: int) -> tuple[int, int]:
    left = int(total // 2)
    right = int(total - left)
    return left, right


def _random_chain() -> list[dict[str, int]] | None:
    count = _randint(*PIECE_COUNT_RANGE)
    colors = _sample_colors(count)

    rects = [
        {
            "r": 0,
            "c": 0,
            "h": _randint(*RECT_HEIGHT_RANGE),
            "w": _randint(*RECT_WIDTH_RANGE),
            "color": int(colors[0]),
        }
    ]

    for idx in range(1, count):
        prev = rects[-1]
        h = _randint(*RECT_HEIGHT_RANGE)
        w = _randint(*RECT_WIDTH_RANGE)
        max_oh = min(OVERLAP_SIZE_RANGE[1], int(prev["h"]) - 1, h)
        max_ow = min(OVERLAP_SIZE_RANGE[1], int(prev["w"]) - 1, w)
        if max_oh < OVERLAP_SIZE_RANGE[0] or max_ow < OVERLAP_SIZE_RANGE[0]:
            return None
        overlap_h = _randint(OVERLAP_SIZE_RANGE[0], max_oh)
        overlap_w = _randint(OVERLAP_SIZE_RANGE[0], max_ow)
        corner = CORNERS[_randint(0, len(CORNERS) - 1)]
        r, c = _place_corner_overlap(prev, h, w, corner, overlap_h, overlap_w)
        rects.append(
            {
                "r": int(r),
                "c": int(c),
                "h": h,
                "w": w,
                "color": int(colors[idx]),
            }
        )

    rects = _normalize_rects(rects)
    if not _valid_chain(rects):
        return None
    return rects


def _render_input(rects: list[dict[str, int]]) -> np.ndarray | None:
    union_h, union_w = _scene_shape(rects)

    inner_top = _randint(*INNER_MARGIN_RANGE)
    inner_bottom = _randint(*INNER_MARGIN_RANGE)
    inner_left = _randint(*INNER_MARGIN_RANGE)
    inner_right = _randint(*INNER_MARGIN_RANGE)

    inner_h = union_h + inner_top + inner_bottom
    inner_w = union_w + inner_left + inner_right

    outer_h_total = _randint(*OUTER_MARGIN_TOTAL_RANGE)
    outer_w_total = _randint(*OUTER_MARGIN_TOTAL_RANGE)
    outer_top, outer_bottom = _split_margin(outer_h_total)
    outer_left, outer_right = _split_margin(outer_w_total)

    grid_h = inner_h + 2 + outer_top + outer_bottom
    grid_w = inner_w + 2 + outer_left + outer_right
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    frame_bbox = (
        int(outer_top),
        int(outer_left),
        int(outer_top + inner_h + 1),
        int(outer_left + inner_w + 1),
    )
    _draw_frame(grid, frame_bbox, FRAME_COLOR)

    shift_r = int(outer_top + 1 + inner_top)
    shift_c = int(outer_left + 1 + inner_left)
    for rect in rects:
        r = int(rect["r"] + shift_r)
        c = int(rect["c"] + shift_c)
        grid[r : r + int(rect["h"]), c : c + int(rect["w"])] = int(rect["color"])

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        rects = _random_chain()
        if rects is None:
            continue

        expected = _expected_output(rects)
        if max(expected.shape) > 30:
            continue

        grid = _render_input(rects)
        if grid is None:
            continue

        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError(
        f"failed to generate a valid input after {GENERATION_RETRY_LIMIT} attempts"
    )
