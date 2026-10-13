from __future__ import annotations

import random

import numpy as np


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    color = int(comp["color"])
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1) and bool(
        np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    )


def _center(comp: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _ordered_components(components: list[dict], orientation: str) -> list[dict]:
    if orientation == "horizontal":
        return sorted(components, key=lambda comp: (_center(comp)[1], _center(comp)[0]))
    return sorted(components, key=lambda comp: (_center(comp)[0], _center(comp)[1]))


def _pair_is_valid(a: dict, b: dict, orientation: str) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])

    if orientation == "horizontal":
        gap = bc0 - ac1 - 1
        overlap = min(ar1, br1) - max(ar0, br0) + 1
    else:
        gap = br0 - ar1 - 1
        overlap = min(ac1, bc1) - max(ac0, bc0) + 1

    return gap >= 1 and overlap >= 1


def _infer_orientation(components: list[dict]) -> tuple[str, list[dict]]:
    ordered_h = _ordered_components(components, "horizontal")
    ordered_v = _ordered_components(components, "vertical")
    fits_h = all(_pair_is_valid(a, b, "horizontal") for a, b in zip(ordered_h, ordered_h[1:]))
    fits_v = all(_pair_is_valid(a, b, "vertical") for a, b in zip(ordered_v, ordered_v[1:]))

    if fits_h and not fits_v:
        return "horizontal", ordered_h
    if fits_v and not fits_h:
        return "vertical", ordered_v

    centers = np.asarray([_center(comp) for comp in components], dtype=float)
    row_span = float(centers[:, 0].max() - centers[:, 0].min())
    col_span = float(centers[:, 1].max() - centers[:, 1].min())
    orientation = "horizontal" if col_span >= row_span else "vertical"
    ordered = ordered_h if orientation == "horizontal" else ordered_v
    assert all(_pair_is_valid(a, b, orientation) for a, b in zip(ordered, ordered[1:]))
    return orientation, ordered


def _extract_scene(grid: np.ndarray) -> tuple[int, str, list[dict]]:
    bg = _majority_color(grid)
    components = _find_components(grid, bg)
    assert len(components) >= 2, "expected at least two foreground rectangles"
    assert len({int(comp["color"]) for comp in components}) == len(components), "foreground colors must be distinct"
    for comp in components:
        assert _is_solid_rectangle(grid, comp), "each foreground object must be a solid rectangle"
    orientation, ordered = _infer_orientation(components)
    return bg, orientation, ordered


def _connector_slice(a: dict, b: dict, orientation: str) -> tuple[slice, slice]:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])

    if orientation == "horizontal":
        overlap0 = max(ar0, br0)
        overlap1 = min(ar1, br1)
        overlap = overlap1 - overlap0 + 1
        thickness = min(int(a["area"]), int(b["area"]), overlap)
        start = overlap0 + (overlap - thickness) // 2
        return slice(start, start + thickness), slice(ac1 + 1, bc0)

    overlap0 = max(ac0, bc0)
    overlap1 = min(ac1, bc1)
    overlap = overlap1 - overlap0 + 1
    thickness = min(int(a["area"]), int(b["area"]), overlap)
    start = overlap0 + (overlap - thickness) // 2
    return slice(ar1 + 1, br0), slice(start, start + thickness)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect adjacent aligned rectangles with centered bars.

    The rectangles are ordered by the dominant arrangement axis. Between each
    consecutive pair, draw a connector through the gap using the color of the
    larger rectangle and a thickness based on the smaller area, capped by the
    available overlap on the perpendicular axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, orientation, objects = _extract_scene(g)
    out = g.copy()

    for a, b in zip(objects, objects[1:]):
        rr, cc = _connector_slice(a, b, orientation)
        assert rr.stop > rr.start and cc.stop > cc.start
        assert bool(np.all(g[rr, cc] == bg)), "connectors must lie entirely in background gaps"
        color = int(a["color"]) if int(a["area"]) > int(b["area"]) else int(b["color"])
        out[rr, cc] = color

    return out


# --------------- generator ---------------

BG = 5
COLOR_POOL = (0, 1, 2, 3, 4, 6, 7, 8, 9)
ORIENTATIONS = ("horizontal", "vertical")

OBJECT_COUNT_RANGE = (3, 5)
ALONG_SIZE_RANGE = (1, 3)
SHARED_BAND_RANGE = (1, 3)
EXTRA_RANGE = (0, 2)
GAP_RANGE = (1, 2)
MARGIN_RANGE = (1, 3)
MAX_SIDE = 29
GEN_ATTEMPTS = 400


def _sample_rectangles(orientation: str) -> tuple[list[dict], tuple[int, int]] | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    shared = random.randint(*SHARED_BAND_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]
    colors = random.sample(COLOR_POOL, count)

    along_sizes = [random.randint(*ALONG_SIZE_RANGE) for _ in range(count)]
    extra_a = [random.randint(*EXTRA_RANGE) for _ in range(count)]
    extra_b = [random.randint(*EXTRA_RANGE) for _ in range(count)]

    areas: list[int] = []
    rect_sizes: list[tuple[int, int]] = []
    for along, ea, eb in zip(along_sizes, extra_a, extra_b):
        cross = shared + ea + eb
        if orientation == "horizontal":
            height, width = cross, along
        else:
            height, width = along, cross
        rect_sizes.append((height, width))
        areas.append(height * width)

    if any(areas[i] == areas[i + 1] for i in range(count - 1)):
        return None

    if orientation == "horizontal":
        max_up = max(extra_a)
        max_down = max(extra_b)
        shared_top = top_margin + max_up
        grid_h = shared_top + shared + max_down + bottom_margin
        grid_w = left_margin + sum(width for _height, width in rect_sizes) + sum(gaps) + right_margin
        if grid_h > MAX_SIDE or grid_w > MAX_SIDE:
            return None

        rects: list[dict] = []
        col = left_margin
        for i, ((height, width), color, up, down) in enumerate(zip(rect_sizes, colors, extra_a, extra_b)):
            r0 = shared_top - up
            r1 = shared_top + shared - 1 + down
            rects.append({"bbox": (r0, col, r1, col + width - 1), "color": int(color)})
            col += width
            if i < count - 1:
                col += gaps[i]
    else:
        max_left = max(extra_a)
        max_right = max(extra_b)
        shared_left = left_margin + max_left
        grid_h = top_margin + sum(height for height, _width in rect_sizes) + sum(gaps) + bottom_margin
        grid_w = shared_left + shared + max_right + right_margin
        if grid_h > MAX_SIDE or grid_w > MAX_SIDE:
            return None

        rects = []
        row = top_margin
        for i, ((height, width), color, left, right) in enumerate(zip(rect_sizes, colors, extra_a, extra_b)):
            c0 = shared_left - left
            c1 = shared_left + shared - 1 + right
            rects.append({"bbox": (row, c0, row + height - 1, c1), "color": int(color)})
            row += height
            if i < count - 1:
                row += gaps[i]

    fg_area = sum((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) for bbox in (rect["bbox"] for rect in rects))
    if grid_h * grid_w <= 2 * fg_area:
        return None

    return rects, (grid_h, grid_w)


def _render_rectangles(rects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return grid


def _generated_scene_ok(grid: np.ndarray, orientation: str) -> bool:
    try:
        bg, inferred_orientation, objects = _extract_scene(grid)
        out = main(grid)
    except Exception:
        return False

    if bg != BG or inferred_orientation != orientation:
        return False
    if out.shape != grid.shape:
        return False
    if int(np.count_nonzero(out != BG)) <= int(np.count_nonzero(grid != BG)):
        return False
    if len(objects) < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        orientation = random.choice(ORIENTATIONS)
        sampled = _sample_rectangles(orientation)
        if sampled is None:
            continue
        rects, shape = sampled
        grid = _render_rectangles(rects, shape)
        if _generated_scene_ok(grid, orientation):
            return grid
    raise RuntimeError("failed to generate a valid input")
