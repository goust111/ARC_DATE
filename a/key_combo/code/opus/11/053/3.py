import random

import numpy as np


BACKGROUND = 5
RECTANGLE_COLORS = (1, 2, 3, 4, 6)
GEN_HEIGHT_RANGE = (14, 19)
GEN_WIDTH_RANGE = (14, 19)
GEN_RECT_COUNT_RANGE = (3, 5)
GEN_MAX_SIDE = 7
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    rectangles: list[dict] = []

    for r in range(height):
        for c in range(width):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            box = grid[r0 : r1 + 1, c0 : c1 + 1]
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1

            assert rect_h >= 3 and rect_w >= 3, "each object must be at least 3x3"
            assert len(coords) == rect_h * rect_w, "foreground objects must be solid rectangles"
            assert np.all(box == color), "rectangle bbox must be monochrome and filled"

            rectangles.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "height": rect_h,
                    "width": rect_w,
                }
            )

    assert rectangles, "expected at least one rectangle"
    return rectangles


def _paint_wedge(interior: np.ndarray, color: int, bg: int) -> None:
    inner_h, inner_w = interior.shape
    interior[:, :] = bg
    for local_r in range(inner_h):
        left = local_r
        right = inner_w - 1 - local_r
        if left > right:
            break
        interior[local_r, left : right + 1] = color


def _paint_checker(interior: np.ndarray, color: int, bg: int) -> None:
    rows, cols = np.indices(interior.shape)
    interior[:, :] = color
    interior[(rows + cols) % 2 == 0] = bg


def _apply_rectangle_fill(
    out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int, bg: int, mode: str | None = None
) -> None:
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = interior.shape
    interior_area = inner_h * inner_w

    if mode is None:
        mode = "wedge" if interior_area % 2 == 1 else "checker"

    if mode == "wedge":
        _paint_wedge(interior, color, bg)
        return
    if mode == "checker":
        _paint_checker(interior, color, bg)
        return
    raise ValueError(f"unknown fill mode: {mode}")


def _solve_with_mode(grid: np.ndarray, mode: str | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    rectangles = _extract_rectangles(g, bg)

    out = g.copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        _apply_rectangle_fill(out, r0, c0, r1, c1, rect["color"], bg, mode=mode)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each solid rectangle interior with a parity-selected wedge or checkerboard pattern.

    Input structure:
    - The background is the most common color.
    - Every non-background object is a monochrome filled axis-aligned rectangle with a 1-cell border and a nonempty interior.
    - Rectangles are separated by background, so each rectangle can be identified independently.

    Rule:
    - Preserve the 1-cell border of every rectangle.
    - Compute the interior area `(height - 2) * (width - 2)` for each rectangle.
    - If that area is odd, repaint the interior as a shrinking top-down wedge: interior row `r` keeps only local columns `r .. inner_width - 1 - r`, and all trimmed cells become background.
    - If that area is even, carve a checkerboard through the interior by turning cells with even `(local_row + local_col)` parity relative to the interior top-left corner into background.

    Key insight:
    - A rectangle's visible color or outer footprint does not choose the fill rule; you must isolate the interior, count its cells, and use that parity to dispatch to two different fill algorithms.
    """
    return _solve_with_mode(grid)


def _all_sizes() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    odd_sizes = []
    even_sizes = []
    for rect_h in range(3, GEN_MAX_SIDE + 1):
        for rect_w in range(3, GEN_MAX_SIDE + 1):
            inner_h = rect_h - 2
            inner_w = rect_w - 2
            inner_area = inner_h * inner_w
            if inner_area % 2 == 1:
                if inner_h >= 2:
                    odd_sizes.append((rect_h, rect_w))
            else:
                even_sizes.append((rect_h, rect_w))
    return odd_sizes, even_sizes


ODD_SIZES, EVEN_SIZES = _all_sizes()
ODD_EDGE_SIZES = [size for size in ODD_SIZES if size[1] == 3]
EVEN_EDGE_SIZES = [size for size in EVEN_SIZES if min(size) <= 4]


def _boxes_have_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _place_rectangles(
    grid_h: int, grid_w: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    used_boxes: list[tuple[int, int, int, int]] = []

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        rect_h, rect_w = sizes[idx]
        positions = [(r, c) for r in range(grid_h - rect_h + 1) for c in range(grid_w - rect_w + 1)]
        random.shuffle(positions)

        for top, left in positions:
            box = (top, left, top + rect_h - 1, left + rect_w - 1)
            if all(_boxes_have_gap(box, other) for other in used_boxes):
                used_boxes.append(box)
                placed[idx] = box
                if backtrack(pos + 1):
                    return True
                used_boxes.pop()
                placed[idx] = None
        return False

    if not backtrack(0):
        return None
    return [box for box in placed if box is not None]


def _sample_sizes(rect_count: int, grid_area: int) -> list[tuple[int, int]] | None:
    mandatory_odd = random.choice(ODD_EDGE_SIZES if random.random() < 0.55 else ODD_SIZES)
    mandatory_even = random.choice(EVEN_EDGE_SIZES if random.random() < 0.55 else EVEN_SIZES)
    sizes = [mandatory_odd, mandatory_even]

    while len(sizes) < rect_count:
        if random.random() < 0.5:
            sizes.append(random.choice(ODD_SIZES))
        else:
            sizes.append(random.choice(EVEN_SIZES))

    random.shuffle(sizes)
    total_area = sum(rect_h * rect_w for rect_h, rect_w in sizes)
    if total_area >= grid_area // 2:
        return None
    return sizes


def _render_input(
    grid_h: int, grid_w: int, boxes: list[tuple[int, int, int, int]], colors: list[int]
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    for (r0, c0, r1, c1), color in zip(boxes, colors):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _validate_case(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    bg = _get_bg(inp)
    assert bg == BACKGROUND
    rectangles = _extract_rectangles(inp, bg)
    assert 3 <= len(rectangles) <= 5
    colors = [rect["color"] for rect in rectangles]
    assert len(colors) == len(set(colors))
    assert set(colors).issubset(set(RECTANGLE_COLORS))

    odd_count = 0
    even_count = 0
    for index, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        for other in rectangles[index + 1 :]:
            assert _boxes_have_gap(rect["bbox"], other["bbox"]), "rectangles must have a background gap"

        inner_h = rect["height"] - 2
        inner_w = rect["width"] - 2
        inner_area = inner_h * inner_w
        input_patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        output_patch = out[r0 : r1 + 1, c0 : c1 + 1]

        assert np.array_equal(input_patch[0], output_patch[0])
        assert np.array_equal(input_patch[-1], output_patch[-1])
        assert np.array_equal(input_patch[:, 0], output_patch[:, 0])
        assert np.array_equal(input_patch[:, -1], output_patch[:, -1])

        expected = input_patch.copy()
        _apply_rectangle_fill(expected, 0, 0, rect["height"] - 1, rect["width"] - 1, rect["color"], bg)
        assert np.array_equal(expected, output_patch)
        assert np.any(input_patch[1:-1, 1:-1] != output_patch[1:-1, 1:-1]), "each rectangle must visibly change"

        if inner_area % 2 == 1:
            odd_count += 1
        else:
            even_count += 1

    assert odd_count >= 1 and even_count >= 1

    all_checker = _solve_with_mode(inp, mode="checker")
    all_wedge = _solve_with_mode(inp, mode="wedge")
    assert not np.array_equal(out, all_checker), "examples must require the odd-area branch"
    assert not np.array_equal(out, all_wedge), "examples must require the even-area branch"


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GEN_HEIGHT_RANGE)
    grid_w = random.randint(*GEN_WIDTH_RANGE)
    rect_count = random.randint(*GEN_RECT_COUNT_RANGE)

    sizes = _sample_sizes(rect_count, grid_h * grid_w)
    if sizes is None:
        return None

    boxes = _place_rectangles(grid_h, grid_w, sizes)
    if boxes is None:
        return None

    colors = random.sample(RECTANGLE_COLORS, rect_count)
    inp = _render_input(grid_h, grid_w, boxes, colors)
    if np.count_nonzero(inp == BACKGROUND) <= inp.size // 2:
        return None

    out = main(inp)
    try:
        _validate_case(inp, out)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-rectangle input")
