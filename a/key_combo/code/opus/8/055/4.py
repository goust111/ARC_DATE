import random
from collections import deque

import numpy as np


BACKGROUND = 8
RECTANGLE = 1
FILL_COLORS = (2, 3, 4, 7)
CROSS = 5
INNER_SIZES = (5, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _quadrant_from_marker(marker_r: int, marker_c: int, center_r: int, center_c: int) -> str:
    if marker_r < center_r and marker_c < center_c:
        return "tl"
    if marker_r < center_r and marker_c > center_c:
        return "tr"
    if marker_r > center_r and marker_c < center_c:
        return "bl"
    if marker_r > center_r and marker_c > center_c:
        return "br"
    raise AssertionError("marker must not lie on the center row or center column")


def _canonical_local_marker(inner_h: int, inner_w: int, quadrant: str) -> tuple[int, int]:
    center_r = inner_h // 2
    center_c = inner_w // 2
    mapping = {
        "tl": (center_r - 1, center_c - 1),
        "tr": (center_r - 1, center_c + 1),
        "bl": (center_r + 1, center_c - 1),
        "br": (center_r + 1, center_c + 1),
    }
    return mapping[quadrant]


def _analyze_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all(np.isin(g, (BACKGROUND, RECTANGLE, CROSS, *FILL_COLORS))) or not np.any(
        g == CROSS
    )
    blue_components = _components(g == RECTANGLE)
    assert 2 <= len(blue_components) <= 4

    rectangles: list[dict] = []
    marker_positions: set[tuple[int, int]] = set()
    for cells in blue_components:
        r0, c0, r1, c1 = _bbox(cells)
        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        assert inner_h in INNER_SIZES and inner_w in INNER_SIZES
        assert inner_h % 2 == 1 and inner_w % 2 == 1

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch[0, :] == RECTANGLE)
        assert np.all(patch[-1, :] == RECTANGLE)
        assert np.all(patch[:, 0] == RECTANGLE)
        assert np.all(patch[:, -1] == RECTANGLE)
        assert np.all(patch[1:-1, 1:-1] != RECTANGLE)
        assert len(cells) == 2 * outer_h + 2 * outer_w - 4

        interior = patch[1:-1, 1:-1]
        marker_locs = np.argwhere(np.isin(interior, FILL_COLORS))
        assert len(marker_locs) == 1
        local_r, local_c = map(int, marker_locs[0])
        marker_color = int(interior[local_r, local_c])
        center_r = r0 + 1 + inner_h // 2
        center_c = c0 + 1 + inner_w // 2
        marker_r = r0 + 1 + local_r
        marker_c = c0 + 1 + local_c
        quadrant = _quadrant_from_marker(marker_r, marker_c, center_r, center_c)
        assert (local_r, local_c) == _canonical_local_marker(inner_h, inner_w, quadrant)

        interior_without_marker = interior.copy()
        interior_without_marker[local_r, local_c] = BACKGROUND
        assert np.all(interior_without_marker == BACKGROUND)

        marker_positions.add((marker_r, marker_c))
        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": (center_r, center_c),
                "marker": (marker_r, marker_c),
                "color": marker_color,
                "quadrant": quadrant,
                "inner_shape": (inner_h, inner_w),
            }
        )

    colored_positions = {
        (int(r), int(c))
        for r, c in np.argwhere(np.isin(g, FILL_COLORS))
    }
    assert colored_positions == marker_positions

    rectangles.sort(key=lambda rect: rect["bbox"])
    return rectangles


def _quadrant_slices(rect: dict) -> tuple[slice, slice]:
    r0, c0, r1, c1 = rect["bbox"]
    center_r, center_c = rect["center"]
    marker_r, marker_c = rect["marker"]
    if marker_r < center_r:
        row_slice = slice(r0 + 1, center_r)
    else:
        row_slice = slice(center_r + 1, r1)
    if marker_c < center_c:
        col_slice = slice(c0 + 1, center_c)
    else:
        col_slice = slice(center_c + 1, c1)
    return row_slice, col_slice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an internal grey cross in each blue hollow rectangle and fill only the marker's quadrant.

    Input structure:
    - The background is teal.
    - The grid contains 2-4 separated blue hollow rectangles with odd-sized interiors.
    - Each rectangle interior has exactly one marker pixel in red, green, yellow, or orange.

    Rule:
    - For each blue rectangle, find the center row and center column of its interior.
    - Draw a grey horizontal segment across the interior center row and a grey vertical segment
      across the interior center column.
    - Determine which of the four quadrants contains the marker pixel.
    - Fill that quadrant with the marker's color, leaving the other three quadrants teal.

    Key insight:
    - The marker color chooses the fill color, but the grey cross is the real structure:
      it partitions one hollow interior into four candidate regions, and only one is selected.
    """
    g = np.asarray(grid, dtype=int)
    rectangles = _analyze_rectangles(g)
    out = g.copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        center_r, center_c = rect["center"]
        out[center_r, c0 + 1 : c1] = CROSS
        out[r0 + 1 : r1, center_c] = CROSS
        row_slice, col_slice = _quadrant_slices(rect)
        out[row_slice, col_slice] = rect["color"]
    return out


def _expanded_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _choose_quadrants(count: int) -> list[str]:
    all_quadrants = ["tl", "tr", "bl", "br"]
    if count == 2:
        return list(random.choice((("tl", "br"), ("tr", "bl"))))
    return random.sample(all_quadrants, count)


def _choose_colors(count: int) -> list[int]:
    if count == 2:
        return random.sample(FILL_COLORS, 2)
    if random.random() < 0.35:
        colors = [random.choice(FILL_COLORS) for _ in range(count)]
        if len(set(colors)) == 1:
            colors[random.randrange(count)] = random.choice([c for c in FILL_COLORS if c != colors[0]])
        return colors
    return random.sample(FILL_COLORS, count)


def _paint_rectangle(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    quadrant: str,
    color: int,
) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = RECTANGLE
    grid[top : bottom + 1, right] = RECTANGLE
    grid[top, left : right + 1] = RECTANGLE
    grid[bottom, left : right + 1] = RECTANGLE

    local_r, local_c = _canonical_local_marker(inner_h, inner_w, quadrant)
    grid[top + 1 + local_r, left + 1 + local_c] = color


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    count = random.choice((2, 3, 4))
    quadrants = _choose_quadrants(count)
    colors = _choose_colors(count)
    objects = []
    for i in range(count):
        inner_h = random.choice(INNER_SIZES)
        inner_w = random.choice(INNER_SIZES)
        objects.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "quadrant": quadrants[i],
                "color": colors[i],
            }
        )

    order = sorted(range(count), key=lambda i: objects[i]["outer_h"] * objects[i]["outer_w"], reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: dict[int, tuple[int, int]] = {}
    for index in order:
        obj = objects[index]
        max_top = height - obj["outer_h"] - 1
        max_left = width - obj["outer_w"] - 1
        if max_top < 1 or max_left < 1:
            return None
        candidates = []
        for top in range(1, max_top + 1):
            for left in range(1, max_left + 1):
                box = (top, left, top + obj["outer_h"] - 1, left + obj["outer_w"] - 1)
                if any(_expanded_conflict(box, other) for other in placed_boxes):
                    continue
                candidates.append((top, left, box))
        if not candidates:
            return None
        top, left, box = random.choice(candidates)
        placements[index] = (top, left)
        placed_boxes.append(box)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for index, obj in enumerate(objects):
        top, left = placements[index]
        _paint_rectangle(grid, top, left, obj["inner_h"], obj["inner_w"], obj["quadrant"], obj["color"])
    return grid


def _validate_generated_pair(inp: np.ndarray, out: np.ndarray) -> None:
    rectangles = _analyze_rectangles(inp)
    assert inp.shape == out.shape
    assert inp.ndim == 2 and out.ndim == 2
    assert np.all(inp != CROSS)
    assert np.any(inp != out)

    quadrants = {rect["quadrant"] for rect in rectangles}
    assert len(quadrants) >= 2
    assert any(q in {"tl", "tr"} for q in quadrants)
    assert any(q in {"bl", "br"} for q in quadrants)
    assert any(q in {"tl", "bl"} for q in quadrants)
    assert any(q in {"tr", "br"} for q in quadrants)

    colors = {rect["color"] for rect in rectangles}
    assert len(colors) >= 2

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        center_r, center_c = rect["center"]
        row_slice, col_slice = _quadrant_slices(rect)

        assert np.all(out[r0 : r1 + 1, c0] == RECTANGLE)
        assert np.all(out[r0 : r1 + 1, c1] == RECTANGLE)
        assert np.all(out[r0, c0 : c1 + 1] == RECTANGLE)
        assert np.all(out[r1, c0 : c1 + 1] == RECTANGLE)
        assert np.all(out[center_r, c0 + 1 : c1] == CROSS)
        assert np.all(out[r0 + 1 : r1, center_c] == CROSS)
        assert np.all(out[row_slice, col_slice] == rect["color"])

        tl = out[r0 + 1 : center_r, c0 + 1 : center_c]
        tr = out[r0 + 1 : center_r, center_c + 1 : c1]
        bl = out[center_r + 1 : r1, c0 + 1 : center_c]
        br = out[center_r + 1 : r1, center_c + 1 : c1]
        for quadrant, patch in {"tl": tl, "tr": tr, "bl": bl, "br": br}.items():
            expected = rect["color"] if quadrant == rect["quadrant"] else BACKGROUND
            assert np.all(patch == expected)

    outside_mask = np.ones_like(inp, dtype=bool)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    assert np.array_equal(out[outside_mask], inp[outside_mask])


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is None:
            continue
        out = main(grid)
        _validate_generated_pair(grid, out)
        return grid
    raise RuntimeError("failed to generate a valid input grid")
