from __future__ import annotations

import numpy as np


BG = 0
RECT = 1
MARKER_COLORS = (2, 3, 4)
MIN_BARRIER_GAP = 4  # ensures every sub-rectangle interior side is at least 3
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _collapse_adjacent(colors: list[int]) -> list[int]:
    palette: list[int] = []
    for color in colors:
        if not palette or palette[-1] != int(color):
            palette.append(int(color))
    return palette


def _extract_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    r0, c0, r1, c1 = _bbox(g != BG)
    if np.any(g[:r0, :] != BG) or np.any(g[r1 + 1 :, :] != BG):
        raise AssertionError("non-black cells must belong to the main rectangle")
    if np.any(g[r0 : r1 + 1, :c0] != BG) or np.any(g[r0 : r1 + 1, c1 + 1 :] != BG):
        raise AssertionError("non-black cells must belong to the main rectangle")

    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(rect == BG):
        raise AssertionError("rectangle must be fully filled by blue or marker colors")
    if not (
        np.all(rect[0, :] == RECT)
        and np.all(rect[-1, :] == RECT)
        and np.all(rect[:, 0] == RECT)
        and np.all(rect[:, -1] == RECT)
    ):
        raise AssertionError("rectangle border must stay blue")

    marker_local = np.argwhere(rect != RECT)
    if marker_local.shape[0] not in (2, 3):
        raise AssertionError("expected 2 or 3 isolated markers")

    marker_rows = [int(r0 + r) for r, _c in marker_local]
    marker_cols = [int(c0 + c) for _r, c in marker_local]
    if len(set(marker_rows)) != marker_local.shape[0] or len(set(marker_cols)) != marker_local.shape[0]:
        raise AssertionError("markers must occupy distinct rows and columns")

    markers = []
    seen_colors = set()
    for lr, lc in marker_local:
        color = int(rect[int(lr), int(lc)])
        if color not in MARKER_COLORS:
            raise AssertionError("unexpected marker color")
        if color in seen_colors:
            raise AssertionError("marker colors must be distinct")
        seen_colors.add(color)
        markers.append((int(r0 + lr), int(c0 + lc), color))

    row_to_color = {r: color for r, _c, color in markers}
    col_to_color = {c: color for _r, c, color in markers}

    return {
        "grid": g,
        "bbox": (r0, c0, r1, c1),
        "markers": markers,
        "row_to_color": row_to_color,
        "col_to_color": col_to_color,
    }


def _expanded_lines(layout: dict) -> np.ndarray:
    g = layout["grid"]
    r0, c0, r1, c1 = layout["bbox"]
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = RECT

    for c, color in layout["col_to_color"].items():
        out[r0 : r1 + 1, int(c)] = int(color)
    for r, color in layout["row_to_color"].items():
        out[int(r), c0 : c1 + 1] = int(color)

    return out


def _subrectangles(layout: dict) -> list[dict]:
    r0, c0, r1, c1 = layout["bbox"]
    row_to_color = layout["row_to_color"]
    col_to_color = layout["col_to_color"]

    barriers_r = [r0, *sorted(row_to_color), r1]
    barriers_c = [c0, *sorted(col_to_color), c1]
    regions: list[dict] = []

    for i in range(len(barriers_r) - 1):
        top = int(barriers_r[i])
        bottom = int(barriers_r[i + 1])
        for j in range(len(barriers_c) - 1):
            left = int(barriers_c[j])
            right = int(barriers_c[j + 1])
            region = {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "ir0": top + 1,
                "ir1": bottom - 1,
                "ic0": left + 1,
                "ic1": right - 1,
                "raw_palette": [
                    RECT if top == r0 else int(row_to_color[top]),
                    RECT if right == c1 else int(col_to_color[right]),
                    RECT if bottom == r1 else int(row_to_color[bottom]),
                    RECT if left == c0 else int(col_to_color[left]),
                ],
            }
            region["palette"] = _collapse_adjacent(region["raw_palette"])
            regions.append(region)

    return regions


def _paint_regions(layout: dict) -> np.ndarray:
    out = _expanded_lines(layout)
    for region in _subrectangles(layout):
        palette = region["palette"]
        top = region["top"]
        bottom = region["bottom"]
        left = region["left"]
        right = region["right"]
        for r in range(region["ir0"], region["ir1"] + 1):
            dr = min(r - top, bottom - r)
            for c in range(region["ic0"], region["ic1"] + 1):
                d = min(dr, c - left, right - c)
                out[r, c] = int(palette[d % len(palette)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand sparse markers inside a blue rectangle into partition lines, then fill each partition with concentric bands from its boundary colors.

    Input structure:
    - A black background surrounds one large filled blue rectangle.
    - Inside the rectangle sit 2-3 isolated non-blue marker pixels with distinct colors and distinct rows/columns.
    - Markers are away from the rectangle border so their expanded rows and columns split the rectangle into visible sub-rectangles.

    Rule:
    - Treat each marker as a control for its whole rectangle row and whole rectangle column.
    - Expand every marker across the full rectangle width and height; when a horizontal and vertical expansion cross, the row color wins.
    - For every open sub-rectangle between consecutive expansion lines or rectangle edges, read the top/right/bottom/left boundary colors, remove only consecutive duplicates in that clockwise list, and color each interior cell by the resulting palette indexed with its Chebyshev distance to the sub-rectangle boundary.

    Key insight:
    - The markers do not directly define the final pattern; they first create the partition topology, and each partition then derives its own cyclic concentric palette from the colors of the specific lines that surround it.
    """
    return _paint_regions(_extract_layout(grid))


def _sample_barriers(length: int, count: int, rng: np.random.Generator) -> list[int]:
    base_total = MIN_BARRIER_GAP * (count + 1)
    extra = length - 1 - base_total
    if extra < 0:
        raise ValueError("length too small for the requested marker count")
    extras = rng.multinomial(extra, np.full(count + 1, 1.0 / (count + 1)))
    gaps = [MIN_BARRIER_GAP + int(v) for v in extras]

    pos = 0
    barriers: list[int] = []
    for gap in gaps[:-1]:
        pos += int(gap)
        barriers.append(pos)
    return barriers


def _build_input(
    rect_h: int,
    rect_w: int,
    marker_count: int,
    pads: tuple[int, int, int, int],
    marker_rows_local: list[int],
    marker_cols_local: list[int],
    marker_colors: list[int],
) -> np.ndarray:
    top_pad, bottom_pad, left_pad, right_pad = pads
    grid = np.zeros((top_pad + rect_h + bottom_pad, left_pad + rect_w + right_pad), dtype=int)
    r0 = top_pad
    c0 = left_pad
    grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = RECT
    for idx in range(marker_count):
        grid[r0 + marker_rows_local[idx], c0 + marker_cols_local[idx]] = int(marker_colors[idx])
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        layout = _extract_layout(grid)
    except AssertionError:
        return False

    r0, c0, r1, c1 = layout["bbox"]
    rows = [r0, *sorted(layout["row_to_color"]), r1]
    cols = [c0, *sorted(layout["col_to_color"]), c1]
    if any(b - a < MIN_BARRIER_GAP for a, b in zip(rows, rows[1:])):
        return False
    if any(b - a < MIN_BARRIER_GAP for a, b in zip(cols, cols[1:])):
        return False

    regions = _subrectangles(layout)
    if not regions:
        return False
    if any(region["ir0"] > region["ir1"] or region["ic0"] > region["ic1"] for region in regions):
        return False
    if any((region["ir1"] - region["ir0"] + 1) < 3 or (region["ic1"] - region["ic0"] + 1) < 3 for region in regions):
        return False

    collapsed = any(len(region["palette"]) < len(region["raw_palette"]) for region in regions)
    full_cycle = any(len(region["palette"]) == 4 for region in regions)
    if not collapsed or not full_cycle:
        return False

    skeleton = _expanded_lines(layout)
    out = _paint_regions(layout)
    if np.array_equal(out, grid) or np.array_equal(out, skeleton):
        return False

    line_mask = np.zeros_like(grid, dtype=bool)
    for r in layout["row_to_color"]:
        line_mask[int(r), c0 : c1 + 1] = True
    for c in layout["col_to_color"]:
        line_mask[r0 : r1 + 1, int(c)] = True
    if not np.all(out[line_mask] == skeleton[line_mask]):
        return False

    changed_fill = False
    for region in regions:
        palette = region["palette"]
        top = region["top"]
        bottom = region["bottom"]
        left = region["left"]
        right = region["right"]
        for r in range(region["ir0"], region["ir1"] + 1):
            dr = min(r - top, bottom - r)
            for c in range(region["ic0"], region["ic1"] + 1):
                d = min(dr, c - left, right - c)
                expected = int(palette[d % len(palette)])
                if int(out[r, c]) != expected:
                    return False
                if int(out[r, c]) != int(skeleton[r, c]):
                    changed_fill = True
    return changed_fill


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    marker_count = 2 if rng.random() < 0.45 else 3
    min_h = MIN_BARRIER_GAP * (marker_count + 1) + 1
    min_w = MIN_BARRIER_GAP * (marker_count + 1) + 1
    rect_h = int(rng.integers(min_h, min(min_h + 5, 24) + 1))
    rect_w = int(rng.integers(min_w, min(min_w + 5, 24) + 1))

    marker_rows_local = _sample_barriers(rect_h, marker_count, rng)
    marker_cols_local = _sample_barriers(rect_w, marker_count, rng)
    col_perm = list(rng.permutation(marker_count))
    marker_cols_local = [marker_cols_local[idx] for idx in col_perm]
    marker_colors = list(rng.permutation(np.array(MARKER_COLORS, dtype=int))[:marker_count])

    max_extra_h = 30 - rect_h
    max_extra_w = 30 - rect_w
    top_pad = int(rng.integers(1, min(3, max_extra_h - 1) + 1))
    bottom_pad = int(rng.integers(1, min(3, max_extra_h - top_pad) + 1))
    left_pad = int(rng.integers(1, min(3, max_extra_w - 1) + 1))
    right_pad = int(rng.integers(1, min(3, max_extra_w - left_pad) + 1))

    grid = _build_input(
        rect_h,
        rect_w,
        marker_count,
        (top_pad, bottom_pad, left_pad, right_pad),
        marker_rows_local,
        marker_cols_local,
        marker_colors,
    )
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partition-and-bands ARC input")
