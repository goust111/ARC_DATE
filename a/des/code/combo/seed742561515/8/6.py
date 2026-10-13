import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)
RECT_COUNT_WEIGHTS = (2, 3, 3, 4, 4)
GEN_RETRIES = 500


def _present_frame_colors(grid: np.ndarray) -> list[int]:
    colors = []
    for value in np.unique(grid):
        color = int(value)
        if color not in (BG, GREY, MAROON):
            colors.append(color)
    return colors


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rectangles = []
    marker_positions = set()

    for color in _present_frame_colors(g):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue

        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "each filter must be at least 3x3"

        pts = {(int(r), int(c)) for r, c in coords}
        assert any(r == r0 for r, _ in pts), "top border must be visible"
        assert any(r == r1 for r, _ in pts), "bottom border must be visible"
        assert any(c == c0 for _, c in pts), "left border must be visible"
        assert any(c == c1 for _, c in pts), "right border must be visible"
        for r, c in pts:
            assert r in (r0, r1) or c in (c0, c1), "frame pixels must lie on the bbox border"

        marker_pos = (r0 + 1, c0 + 1)
        marker = int(g[marker_pos])
        assert marker in (GREY, MAROON), "missing depth marker at top-left interior"
        marker_positions.add(marker_pos)

        rectangles.append(
            {
                "color": int(color),
                "top": r0,
                "left": c0,
                "bottom": r1,
                "right": c1,
                "height": height,
                "width": width,
                "area": height * width,
                "marker": marker,
            }
        )

    actual_markers = {
        (int(r), int(c))
        for r, c in np.argwhere((g == GREY) | (g == MAROON))
    }
    assert marker_positions == actual_markers, "every marker must belong to exactly one rectangle"
    assert rectangles, "expected at least one filter rectangle"
    return rectangles


def _draw_order(rectangles: list[dict]) -> list[dict]:
    return sorted(
        rectangles,
        key=lambda rect: (
            rect["marker"] == MAROON,
            -rect["area"],
            rect["top"],
            rect["left"],
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow filter frames and redraw them in depth order.

    Grey markers indicate bottom-layer filters, maroon markers indicate top-layer
    filters. Within the same layer, larger rectangles are underneath smaller ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _extract_rectangles(g)
    out = np.full_like(g, BG)
    for rect in _draw_order(rectangles):
        out[rect["top"] : rect["bottom"] + 1, rect["left"] : rect["right"] + 1] = rect["color"]
    return out


def _build_chain(count: int, rng: random.Random) -> tuple[tuple[int, int], list[dict]]:
    min_outer_h = 3 + 4 * (count - 1)
    min_outer_w = 3 + 4 * (count - 1)
    h = rng.randint(max(GRID_H_RANGE[0], min_outer_h + 2), GRID_H_RANGE[1])
    w = rng.randint(max(GRID_W_RANGE[0], min_outer_w + 2), GRID_W_RANGE[1])

    outer_h = rng.randint(min_outer_h, h - 2)
    outer_w = rng.randint(min_outer_w, w - 2)
    outer_top = rng.randint(1, h - outer_h - 1)
    outer_left = rng.randint(1, w - outer_w - 1)

    chain = [
        {
            "index": 0,
            "top": outer_top,
            "left": outer_left,
            "height": outer_h,
            "width": outer_w,
        }
    ]

    for idx in range(1, count):
        prev = chain[-1]
        remaining = count - idx - 1
        min_h = 3 + 4 * remaining
        min_w = 3 + 4 * remaining
        inner_h = rng.randint(min_h, prev["height"] - 4)
        inner_w = rng.randint(min_w, prev["width"] - 4)
        top_inset = rng.randint(2, prev["height"] - inner_h - 2)
        left_inset = rng.randint(2, prev["width"] - inner_w - 2)
        chain.append(
            {
                "index": idx,
                "top": prev["top"] + top_inset,
                "left": prev["left"] + left_inset,
                "height": inner_h,
                "width": inner_w,
            }
        )

    for rect in chain:
        rect["bottom"] = rect["top"] + rect["height"] - 1
        rect["right"] = rect["left"] + rect["width"] - 1
        rect["area"] = rect["height"] * rect["width"]

    return (h, w), chain


def _sample_markers(count: int, mode: str, rng: random.Random) -> list[int] | None:
    for _ in range(200):
        markers = [rng.choice((GREY, MAROON)) for _ in range(count)]
        if len(set(markers)) < 2:
            continue
        order = sorted(range(count), key=lambda idx: (markers[idx] == MAROON, idx))
        if mode == "marker_matters" and order == list(range(count)):
            continue
        return markers
    return None


def _render_input(shape: tuple[int, int], rectangles: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rectangles:
        r0 = rect["top"]
        c0 = rect["left"]
        r1 = rect["bottom"]
        c1 = rect["right"]
        color = rect["color"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0 + 1, c0 + 1] = rect["marker"]
    return grid


def _render_output(shape: tuple[int, int], rectangles: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for rect in _draw_order(rectangles):
        out[rect["top"] : rect["bottom"] + 1, rect["left"] : rect["right"] + 1] = rect["color"]
    return out


def _non_bg_colors(grid: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(grid) if int(v) != BG}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(("aligned", "marker_matters"))
    if mode == "marker_matters":
        count = rng.choice((3, 3, 4, 4))
    else:
        count = rng.choice(RECT_COUNT_WEIGHTS)

    shape, rectangles = _build_chain(count, rng)
    colors = rng.sample(FRAME_COLORS, count)
    markers = _sample_markers(count, mode, rng)
    if markers is None:
        return None

    for rect, color, marker in zip(rectangles, colors, markers):
        rect["color"] = int(color)
        rect["marker"] = int(marker)

    inp = _render_input(shape, rectangles)
    out = _render_output(shape, rectangles)

    if len(_non_bg_colors(out)) < 2:
        return None
    if np.array_equal(inp, out):
        return None
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
