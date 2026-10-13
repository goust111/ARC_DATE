import numpy as np
from collections import deque


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

CONTROL_COLORS = (BLUE, RED, GREEN, YELLOW)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _analyze_controls(grid: np.ndarray) -> dict:
    colored = np.argwhere((grid != BLACK) & (grid != GREY))
    assert len(colored) == 5, "expected exactly four controls plus one pivot"

    points = [(int(r), int(c), int(grid[r, c])) for r, c in colored]
    top = min(points, key=lambda x: x[0])
    bottom = max(points, key=lambda x: x[0])
    left = min(points, key=lambda x: x[1])
    right = max(points, key=lambda x: x[1])
    extreme_positions = {(top[0], top[1]), (bottom[0], bottom[1]), (left[0], left[1]), (right[0], right[1])}
    assert len(extreme_positions) == 4, "control extrema must be unique"

    pivot_candidates = [(r, c, color) for r, c, color in points if (r, c) not in extreme_positions]
    assert len(pivot_candidates) == 1, "expected exactly one interior pivot candidate"
    pivot = pivot_candidates[0]

    top_r = top[0]
    bottom_r = bottom[0]
    left_c = left[1]
    right_c = right[1]
    assert top_r < pivot[0] < bottom_r
    assert left_c < pivot[1] < right_c

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "pivot": pivot,
        "rect": (top_r, left_c, bottom_r, right_c),
    }


def _hole_data(grid: np.ndarray, rect: tuple[int, int, int, int]) -> list[dict]:
    top_r, left_c, bottom_r, right_c = rect
    interior = grid[top_r + 1 : bottom_r, left_c + 1 : right_c]
    holes = []
    for coords in _components(interior == BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        holes.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the control-defined rectangle and fill each interior hole with stripes chosen by hole geometry.

    Input structure:
    - The grid has a black background, four colored control pixels, one red pivot pixel, and gray obstacle walls inside the control-defined rectangle span.
    - The four control pixels have unique topmost, bottommost, leftmost, and rightmost positions among all colored non-gray pixels.
    - Inside the future rectangle interior, the gray walls split the black area into 2-4 disconnected holes.

    Rule:
    - Use the extreme control rows and columns to define a rectangle, then draw its top, bottom, left, and right edges in the corresponding control colors, with corners taking the horizontal edge colors.
    - Keep the gray obstacle pixels and the red pivot unchanged.
    - Find each 4-connected black hole inside the rectangle interior; if its bbox is wider than tall, fill that hole with alternating top/bottom-colored horizontal stripes, if taller than wide fill it with alternating left/right-colored vertical stripes, and if square fill it solid with the pivot color.

    Key insight:
    - The same control pixels provide both rectangle geometry and the palette for the hole fills, while each hole's own bbox aspect ratio chooses which fill mode applies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _analyze_controls(g)
    top_r, left_c, bottom_r, right_c = info["rect"]
    top_color = info["top"][2]
    bottom_color = info["bottom"][2]
    left_color = info["left"][2]
    right_color = info["right"][2]
    pivot_color = info["pivot"][2]

    out = g.copy()
    out[top_r, left_c : right_c + 1] = top_color
    out[bottom_r, left_c : right_c + 1] = bottom_color
    out[top_r + 1 : bottom_r, left_c] = left_color
    out[top_r + 1 : bottom_r, right_c] = right_color

    for hole in _hole_data(g, info["rect"]):
        r0, c0, r1, c1 = hole["bbox"]
        height = hole["height"]
        width = hole["width"]
        for rr, cc in hole["coords"]:
            gr = top_r + 1 + int(rr)
            gc = left_c + 1 + int(cc)
            if width > height:
                out[gr, gc] = top_color if (int(rr) - r0) % 2 == 0 else bottom_color
            elif height > width:
                out[gr, gc] = left_color if (int(cc) - c0) % 2 == 0 else right_color
            else:
                out[gr, gc] = pivot_color

    out[info["pivot"][0], info["pivot"][1]] = pivot_color
    return out


def _classify(height: int, width: int) -> str:
    if width > height:
        return "wide"
    if height > width:
        return "tall"
    return "square"


def _choose_layout(rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    family = "three" if rng.random() < 0.5 else "four"

    if family == "three":
        s = int(rng.integers(3, 6))
        bottom_h = int(rng.integers(2, s))
        ih = s + 1 + bottom_h
        right_w = int(rng.integers(2, ih))

        interior = np.zeros((ih, s + 1 + right_w), dtype=int)
        interior[:, s] = GREY
        interior[s, :s] = GREY
    else:
        while True:
            s = int(rng.integers(4, 6))
            bottom_h = int(rng.integers(2, s))
            ih = s + 1 + bottom_h
            max_right_w = ih - 4
            if max_right_w < 3:
                continue
            right_w = int(rng.integers(3, max_right_w + 1))
            max_rt = min(right_w - 1, ih - right_w - 2)
            if max_rt < 2:
                continue
            top_right_h = int(rng.integers(2, max_rt + 1))
            break

        interior = np.zeros((ih, s + 1 + right_w), dtype=int)
        interior[:, s] = GREY
        interior[s, :s] = GREY
        interior[top_right_h, s + 1 :] = GREY

    pivot_r = int(rng.integers(1, s - 1))
    pivot_c = int(rng.integers(1, s - 1))
    interior[pivot_r, pivot_c] = RED
    return interior, (pivot_r, pivot_c)


def _place_controls(grid: np.ndarray, rect: tuple[int, int, int, int], rng: np.random.Generator) -> None:
    top_r, left_c, bottom_r, right_c = rect
    role_colors = list(rng.permutation(np.array(CONTROL_COLORS, dtype=int)))
    top_color, bottom_color, left_color, right_color = role_colors

    top_col = int(rng.integers(left_c + 1, right_c))
    bottom_col = int(rng.integers(left_c + 1, right_c))
    left_row = int(rng.integers(top_r + 1, bottom_r))
    right_row = int(rng.integers(top_r + 1, bottom_r))

    grid[top_r, top_col] = top_color
    grid[bottom_r, bottom_col] = bottom_color
    grid[left_row, left_c] = left_color
    grid[right_row, right_c] = right_color


def _apply_symmetry(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    choice = int(rng.integers(0, 6))
    if choice == 0:
        return grid
    if choice == 1:
        return np.flipud(grid)
    if choice == 2:
        return np.fliplr(grid)
    if choice == 3:
        return np.flipud(np.fliplr(grid))
    if choice == 4:
        return grid.T.copy()
    return np.flipud(grid.T).copy()


def _validate_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BLACK, BLUE, RED, GREEN, YELLOW, GREY})

    info = _analyze_controls(g)
    top_r, left_c, bottom_r, right_c = info["rect"]
    assert top_r >= 1 and left_c >= 1
    assert bottom_r <= g.shape[0] - 2 and right_c <= g.shape[1] - 2

    holes = _hole_data(g, info["rect"])
    assert 2 <= len(holes) <= 4

    classes = []
    square_count = 0
    pivot_r, pivot_c, _ = info["pivot"]
    pivot_local = (pivot_r - top_r - 1, pivot_c - left_c - 1)
    pivot_in_square = False

    for hole in holes:
        r0, c0, r1, c1 = hole["bbox"]
        height = hole["height"]
        width = hole["width"]
        assert height >= 2 and width >= 2
        cls = _classify(height, width)
        classes.append(cls)
        if cls == "square":
            square_count += 1
            if r0 <= pivot_local[0] <= r1 and c0 <= pivot_local[1] <= c1:
                pivot_in_square = True

    assert "wide" in classes and "tall" in classes and "square" in classes
    assert square_count == 1
    assert pivot_in_square, "pivot must sit inside the unique square-hole bbox"
    return {"controls": info, "holes": holes}


def _validate_output(inp: np.ndarray, out: np.ndarray, analysis: dict) -> None:
    info = analysis["controls"]
    top_r, left_c, bottom_r, right_c = info["rect"]
    top_color = info["top"][2]
    bottom_color = info["bottom"][2]
    left_color = info["left"][2]
    right_color = info["right"][2]
    pivot = info["pivot"]

    assert np.array_equal(out[:top_r, :], inp[:top_r, :]) or top_r == 0
    assert np.array_equal(out[bottom_r + 1 :, :], inp[bottom_r + 1 :, :]) or bottom_r == inp.shape[0] - 1
    assert np.all(out[top_r, left_c : right_c + 1] == top_color)
    assert np.all(out[bottom_r, left_c : right_c + 1] == bottom_color)
    assert np.all(out[top_r + 1 : bottom_r, left_c] == left_color)
    assert np.all(out[top_r + 1 : bottom_r, right_c] == right_color)
    assert out[pivot[0], pivot[1]] == pivot[2] == RED
    assert np.array_equal(inp == GREY, out == GREY)

    for hole in analysis["holes"]:
        r0, c0, _, _ = hole["bbox"]
        height = hole["height"]
        width = hole["width"]
        for rr, cc in hole["coords"]:
            gr = top_r + 1 + int(rr)
            gc = left_c + 1 + int(cc)
            if width > height:
                expected = top_color if (int(rr) - r0) % 2 == 0 else bottom_color
            elif height > width:
                expected = left_color if (int(cc) - c0) % 2 == 0 else right_color
            else:
                expected = pivot[2]
            assert out[gr, gc] == expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        interior, _ = _choose_layout(rng)
        ih, iw = interior.shape
        rect_h = ih + 2
        rect_w = iw + 2

        top_margin = int(rng.integers(1, 4))
        left_margin = int(rng.integers(1, 4))
        bottom_margin = int(rng.integers(1, 4))
        right_margin = int(rng.integers(1, 4))

        grid_h = rect_h + top_margin + bottom_margin
        grid_w = rect_w + left_margin + right_margin
        if max(grid_h, grid_w) > 30:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        top_r = top_margin
        left_c = left_margin
        bottom_r = top_r + rect_h - 1
        right_c = left_c + rect_w - 1

        grid[top_r + 1 : bottom_r, left_c + 1 : right_c] = interior
        _place_controls(grid, (top_r, left_c, bottom_r, right_c), rng)
        grid = _apply_symmetry(grid, rng)

        try:
            analysis = _validate_input(grid)
            out = main(grid)
            _validate_output(grid, out, analysis)
            return grid
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid input for puzzle 10")
