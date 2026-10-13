from __future__ import annotations

import numpy as np


BG = 0
COLOR_COUNT = 10
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 300
SEGMENT_PADDING = 1
RECT_PADDING = 1


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    height, width = mask.shape

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)

    return comps


def _line_info(coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    if r0 == r1 and c0 < c1:
        expected = {(r0, c) for c in range(c0, c1 + 1)}
        assert set(coords) == expected, "horizontal pieces must be solid line segments"
        return {"kind": "h", "length": c1 - c0 + 1, "start": (r0, c0)}

    if c0 == c1 and r0 < r1:
        expected = {(r, c0) for r in range(r0, r1 + 1)}
        assert set(coords) == expected, "vertical pieces must be solid line segments"
        return {"kind": "v", "length": r1 - r0 + 1, "start": (r0, c0)}

    raise AssertionError("each foreground component must be one horizontal or vertical segment of length at least 2")


def _draw_frame(out: np.ndarray, color: int, r0: int, c0: int, height: int, width: int) -> None:
    rows, cols = out.shape
    assert height >= 2 and width >= 2
    assert 0 <= r0 and 0 <= c0 and r0 + height <= rows and c0 + width <= cols

    cells = set()
    for c in range(c0, c0 + width):
        cells.add((r0, c))
        cells.add((r0 + height - 1, c))
    for r in range(r0, r0 + height):
        cells.add((r, c0))
        cells.add((r, c0 + width - 1))

    for r, c in cells:
        assert int(out[r, c]) in (BG, int(color)), "assembled rectangles must not overlap"
        out[r, c] = int(color)


def _present_colors(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen = {BG}
    for value in grid.ravel():
        color = int(value)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered same-color line segments into complete rectangular frames.

    Input structure:
    - The background is black.
    - Each non-background color appears as exactly four disconnected one-pixel-thick line segments.
    - For each color, two segments are horizontal with the same length and two are vertical with the same length.

    Rule:
    - Group the segments by color and identify the two horizontal and two vertical pieces in each group.
    - The horizontal length gives the rectangle width, and the vertical length gives the rectangle height.
    - Draw that color's hollow rectangular border with its upper-left corner at the start endpoint of the most upper-left segment in the group.
    - Clear the scattered fragments; the rectangle interiors and all other cells remain background.

    Key insight:
    Same-colored fragments are not independent objects: they are the four translated sides of one hidden rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full_like(g, BG)
    for color in _present_colors(g):
        segments = [_line_info(comp) for comp in _components_for_color(g, color)]
        assert len(segments) == 4, "each color must have exactly four line segments"

        horizontals = [seg for seg in segments if seg["kind"] == "h"]
        verticals = [seg for seg in segments if seg["kind"] == "v"]
        assert len(horizontals) == 2 and len(verticals) == 2
        assert len({int(seg["length"]) for seg in horizontals}) == 1
        assert len({int(seg["length"]) for seg in verticals}) == 1

        width = int(horizontals[0]["length"])
        height = int(verticals[0]["length"])
        anchor_r, anchor_c = min((tuple(seg["start"]) for seg in segments), key=lambda pt: (pt[0], pt[1]))
        _draw_frame(out, color, int(anchor_r), int(anchor_c), height, width)

    return out


def _segment_cells(kind: str, r0: int, c0: int, length: int) -> list[tuple[int, int]]:
    if kind == "h":
        return [(int(r0), int(c)) for c in range(int(c0), int(c0) + int(length))]
    return [(int(r), int(c0)) for r in range(int(r0), int(r0) + int(length))]


def _cells_fit(cells: list[tuple[int, int]], rows: int, cols: int) -> bool:
    return all(0 <= r < rows and 0 <= c < cols for r, c in cells)


def _can_place_segment(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    rows, cols = occupied.shape
    if not _cells_fit(cells, rows, cols):
        return False
    for r, c in cells:
        r0 = max(0, r - SEGMENT_PADDING)
        r1 = min(rows, r + SEGMENT_PADDING + 1)
        c0 = max(0, c - SEGMENT_PADDING)
        c1 = min(cols, c + SEGMENT_PADDING + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _paint_cells(grid: np.ndarray, occupied: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)
        occupied[r, c] = True


def _place_segment(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: np.random.Generator,
    color: int,
    kind: str,
    length: int,
    min_r: int = 0,
    min_c: int = 0,
) -> bool:
    rows, cols = grid.shape
    max_r = rows - 1 if kind == "h" else rows - int(length)
    max_c = cols - int(length) if kind == "h" else cols - 1
    if min_r > max_r or min_c > max_c:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = int(rng.integers(min_r, max_r + 1))
        c0 = int(rng.integers(min_c, max_c + 1))
        cells = _segment_cells(kind, r0, c0, length)
        if _can_place_segment(occupied, cells):
            _paint_cells(grid, occupied, cells, color)
            return True

    candidates = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        cells = _segment_cells(kind, r0, c0, length)
        if _can_place_segment(occupied, cells):
            _paint_cells(grid, occupied, cells, color)
            return True
    return False


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _place_rectangles(
    rng: np.random.Generator, rows: int, cols: int, colors: list[int]
) -> list[dict] | None:
    rects: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            height = int(rng.integers(3, 8))
            width = int(rng.integers(3, 8))
            max_r = rows - height - 2
            max_c = cols - width - 2
            if max_r < 0 or max_c < 0:
                continue
            r0 = int(rng.integers(0, max_r + 1))
            c0 = int(rng.integers(0, max_c + 1))
            bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
            if any(_bboxes_overlap(bbox, old["bbox"], RECT_PADDING) for old in rects):
                continue
            rects.append(
                {
                    "color": int(color),
                    "r0": r0,
                    "c0": c0,
                    "height": height,
                    "width": width,
                    "bbox": bbox,
                }
            )
            placed = True
            break
        if not placed:
            return None

    return rects


def _build_expected(rows: int, cols: int, rects: list[dict]) -> np.ndarray:
    out = np.full((rows, cols), BG, dtype=int)
    for rect in rects:
        _draw_frame(out, rect["color"], rect["r0"], rect["c0"], rect["height"], rect["width"])
    return out


def _derive_input(rng: np.random.Generator, rows: int, cols: int, rects: list[dict]) -> np.ndarray | None:
    grid = np.full((rows, cols), BG, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    anchors = []
    for rect in rects:
        anchor_kind = "h" if bool(rng.integers(0, 2)) else "v"
        anchor_length = rect["width"] if anchor_kind == "h" else rect["height"]
        cells = _segment_cells(anchor_kind, rect["r0"], rect["c0"], anchor_length)
        if not _can_place_segment(occupied, cells):
            return None
        _paint_cells(grid, occupied, cells, rect["color"])
        anchors.append(anchor_kind)

    order = list(range(len(rects)))
    rng.shuffle(order)
    for idx in order:
        rect = rects[idx]
        remaining = ["h", "h", "v", "v"]
        remaining.remove(anchors[idx])
        rng.shuffle(remaining)

        min_r = int(rect["r0"]) + 2
        min_c = int(rect["c0"]) + 2
        for kind in remaining:
            length = rect["width"] if kind == "h" else rect["height"]
            if not _place_segment(grid, occupied, rng, rect["color"], kind, length, min_r=min_r, min_c=min_c):
                return None

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rects = int(rng.integers(2, 5))
    min_side = {2: 18, 3: 22, 4: 26}[n_rects]
    rows = int(rng.integers(min_side, 31))
    cols = int(rng.integers(min_side, 31))

    colors = [int(c) for c in rng.choice(np.arange(1, COLOR_COUNT), size=n_rects, replace=False)]
    rects = _place_rectangles(rng, rows, cols, colors)
    if rects is None:
        return None

    expected = _build_expected(rows, cols, rects)
    grid = _derive_input(rng, rows, cols, rects)
    if grid is None:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
