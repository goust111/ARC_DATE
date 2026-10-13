from __future__ import annotations

from collections import deque
import random

import numpy as np


RED = 2
BLUE = 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append({"color": color, "coords": np.array(coords, dtype=int)})
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _major_axis(coords: np.ndarray) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return "h" if w >= h else "v"


def _sorted_unique(values: list[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def _component_skeleton(coords: np.ndarray) -> set[tuple[int, int]]:
    axis = _major_axis(coords)
    pts = {(int(r), int(c)) for r, c in coords}
    skeleton: set[tuple[int, int]] = set()

    if axis == "h":
        cols = _sorted_unique([c for _r, c in pts])
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal ribbon must occupy every column in its span"
        centers: dict[int, int] = {}
        for c in cols:
            rows = sorted(r for r, cc in pts if cc == c)
            assert rows[-1] - rows[0] + 1 == len(rows), "column slice must be contiguous"
            centers[c] = int(rows[(len(rows) - 1) // 2])
        prev_r = centers[cols[0]]
        skeleton.add((prev_r, cols[0]))
        for c in cols[1:]:
            cur_r = centers[c]
            for r in range(min(prev_r, cur_r), max(prev_r, cur_r) + 1):
                skeleton.add((r, c))
            prev_r = cur_r
    else:
        rows = _sorted_unique([r for r, _c in pts])
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical ribbon must occupy every row in its span"
        centers: dict[int, int] = {}
        for r in rows:
            cols = sorted(c for rr, c in pts if rr == r)
            assert cols[-1] - cols[0] + 1 == len(cols), "row slice must be contiguous"
            centers[r] = int(cols[(len(cols) - 1) // 2])
        prev_c = centers[rows[0]]
        skeleton.add((rows[0], prev_c))
        for r in rows[1:]:
            cur_c = centers[r]
            for c in range(min(prev_c, cur_c), max(prev_c, cur_c) + 1):
                skeleton.add((r, c))
            prev_c = cur_c

    return skeleton


def _skeleton_endpoints(skeleton: set[tuple[int, int]]) -> list[tuple[int, int]]:
    endpoints = []
    for r, c in skeleton:
        degree = sum((nr, nc) in skeleton for nr, nc in _neighbors4(r, c))
        if degree == 1:
            endpoints.append((int(r), int(c)))
    assert len(endpoints) == 2, "expected a path skeleton with two endpoints"
    return endpoints


def _edge_choice(point: tuple[int, int], shape: tuple[int, int]) -> tuple[tuple[str, int, tuple[int, int], int], bool]:
    r, c = point
    h, w = shape
    candidates = [
        ("left", int(c), (0, -1), RED),
        ("right", int(w - 1 - c), (0, 1), RED),
        ("top", int(r), (-1, 0), BLUE),
        ("bottom", int(h - 1 - r), (1, 0), BLUE),
    ]
    priority = {"left": 0, "right": 1, "top": 2, "bottom": 3}
    ordered = sorted(candidates, key=lambda item: (item[1], priority[item[0]]))
    unique = bool(len(ordered) == 1 or ordered[0][1] < ordered[1][1])
    return ordered[0], unique


def _ray_cells(point: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = step
    rr = int(point[0]) + int(dr)
    cc = int(point[1]) + int(dc)
    out = []
    while 0 <= rr < h and 0 <= cc < w:
        out.append((int(rr), int(cc)))
        rr += int(dr)
        cc += int(dc)
    return out


def _output_footprint(coords: set[tuple[int, int]], shape: tuple[int, int]) -> dict[str, object]:
    arr = np.array(sorted(coords), dtype=int)
    skeleton = _component_skeleton(arr)
    endpoints = _skeleton_endpoints(skeleton)
    rays: list[tuple[list[tuple[int, int]], int]] = []
    footprint = set(skeleton)
    for endpoint in endpoints:
        edge, unique = _edge_choice(endpoint, shape)
        if not unique or edge[1] < 2:
            raise AssertionError("endpoint must have a unique nearest edge with a visible ray")
        ray = _ray_cells(endpoint, edge[2], shape)
        if not ray:
            raise AssertionError("ray must have at least one cell")
        footprint.update(ray)
        rays.append((ray, int(edge[3])))
    return {"skeleton": skeleton, "rays": rays, "footprint": footprint}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each thin object's 1-pixel skeleton, keep it in the object's color,
    then extend each skeleton endpoint to its nearest grid edge.

    Left/right connections are painted red; top/bottom connections are painted blue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for comp in _find_components(g, bg):
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        skeleton = _component_skeleton(coords)
        for r, c in skeleton:
            out[r, c] = color
        for endpoint in _skeleton_endpoints(skeleton):
            edge, _unique = _edge_choice(endpoint, g.shape)
            for rr, cc in _ray_cells(endpoint, edge[2], g.shape):
                out[rr, cc] = int(edge[3])

    return out


GEN_ATTEMPTS = 400
GRID_MIN = 16
GRID_MAX = 24
OBJECT_COUNT_CHOICES = (2, 3, 4)
PALETTE = tuple(range(10))
BG_CHOICES = tuple(c for c in PALETTE if c not in (RED, BLUE))
OBJECT_COLORS = tuple(c for c in PALETTE if c not in (RED, BLUE))
MAX_SHAPE_ATTEMPTS = 120
MAX_PLACE_ATTEMPTS = 160
MAX_LOCAL_SPAN = 6
MIN_MAJOR_LEN = 6
MARGIN_TO_EDGE = 2


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _ribbon_from_segments(segments: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set()
    for c, (lo, width) in enumerate(segments):
        for r in range(lo, lo + width):
            cells.add((int(r), int(c)))
    return _normalize(cells)


def _generate_horizontal_ribbon(max_len: int) -> set[tuple[int, int]]:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        length = random.randint(MIN_MAJOR_LEN, max(MIN_MAJOR_LEN, max_len))
        lo = random.randint(0, 2)
        width = random.randint(1, 3)
        segments = [(lo, width)]
        min_lo = lo
        max_hi = lo + width - 1
        changed = False

        for _step in range(1, length):
            prev_lo, prev_width = segments[-1]
            prev_hi = prev_lo + prev_width - 1
            options: list[tuple[int, int, int, int]] = []
            for next_width in (1, 2, 3):
                for next_lo in range(prev_lo - 2, prev_hi + 1):
                    next_hi = next_lo + next_width - 1
                    if next_hi < prev_lo or next_lo > prev_hi:
                        continue
                    new_min = min(min_lo, next_lo)
                    new_max = max(max_hi, next_hi)
                    if new_max - new_min + 1 > MAX_LOCAL_SPAN:
                        continue
                    options.append((next_lo, next_width, new_min, new_max))
            if not options:
                break

            options.sort(key=lambda item: (abs(item[0] - prev_lo) + abs(item[1] - prev_width), random.random()))
            next_lo, next_width, min_lo, max_hi = random.choice(options[: min(8, len(options))])
            changed |= next_lo != prev_lo or next_width != prev_width
            segments.append((next_lo, next_width))

        if len(segments) != length:
            continue

        cells = _ribbon_from_segments(segments)
        arr = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(arr)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if w < h + 3:
            continue
        if not changed and random.random() < 0.65:
            continue
        return cells

    raise RuntimeError("failed to sample a horizontal ribbon")


def _transpose(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(c), int(r)) for r, c in cells}


def _generate_ribbon(axis: str, max_major: int) -> set[tuple[int, int]]:
    cells = _generate_horizontal_ribbon(max_major)
    return cells if axis == "h" else _transpose(cells)


def _shift(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in cells}


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    arr = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _chebyshev_buffer(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_object(
    shape: tuple[int, int],
    local_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
    prefer_axis: str,
) -> tuple[set[tuple[int, int]], dict[str, object]] | None:
    h, w = shape
    obj_h, obj_w = _shape_dims(local_cells)
    row_candidates = list(range(MARGIN_TO_EDGE, h - obj_h - MARGIN_TO_EDGE + 1))
    col_candidates = list(range(MARGIN_TO_EDGE, w - obj_w - MARGIN_TO_EDGE + 1))
    if not row_candidates or not col_candidates:
        return None

    if prefer_axis == "h":
        weighted_rows = row_candidates
        weighted_cols = col_candidates[: min(3, len(col_candidates))] + col_candidates[max(0, len(col_candidates) - 3) :]
    else:
        weighted_rows = row_candidates[: min(3, len(row_candidates))] + row_candidates[max(0, len(row_candidates) - 3) :]
        weighted_cols = col_candidates

    for _ in range(MAX_PLACE_ATTEMPTS):
        dr = random.choice(weighted_rows if weighted_rows and random.random() < 0.7 else row_candidates)
        dc = random.choice(weighted_cols if weighted_cols and random.random() < 0.7 else col_candidates)
        cells = _shift(local_cells, dr, dc)
        if cells & blocked:
            continue

        try:
            info = _output_footprint(cells, shape)
        except AssertionError:
            continue

        skeleton = set(info["skeleton"])
        footprint = set(info["footprint"])
        ray_cells = footprint - skeleton

        if footprint & reserved_output:
            continue
        if ray_cells & occupied:
            continue

        return cells, info

    return None


def _orientation_schedule(count: int) -> list[str]:
    base = ["h", "v"] * 2
    if count == 2:
        out = ["h", "v"]
    elif count == 3:
        out = ["h", "v", random.choice(("h", "v"))]
    else:
        out = base[:count]
    random.shuffle(out)
    return out


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)

    count = random.choice(OBJECT_COUNT_CHOICES)
    orientations = _orientation_schedule(count)
    colors = [c for c in OBJECT_COLORS if c != bg]
    random.shuffle(colors)
    colors = colors[:count]

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()

    for idx, (axis, color) in enumerate(zip(orientations, colors)):
        max_major = (w - 2 * MARGIN_TO_EDGE - 2) if axis == "h" else (h - 2 * MARGIN_TO_EDGE - 2)
        if max_major < MIN_MAJOR_LEN:
            return None
        local_cells = _generate_ribbon(axis, max_major)
        prefer_axis = "h" if idx % 2 == 0 else "v"
        placed = _place_object((h, w), local_cells, occupied, blocked, reserved_output, prefer_axis)
        if placed is None:
            return None
        cells, info = placed
        for r, c in cells:
            grid[r, c] = int(color)
        occupied |= cells
        blocked |= _chebyshev_buffer(cells, (h, w))
        reserved_output |= set(info["footprint"])

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == RED)) == 0 or int(np.count_nonzero(out == BLUE)) == 0:
        return None
    if int(np.count_nonzero(grid != bg)) < 18:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid skeleton-to-edge puzzle instance")
