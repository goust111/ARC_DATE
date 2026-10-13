from __future__ import annotations

import numpy as np


# --------------- solver ---------------


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(values[counts.argmax()])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    boundary = _boundary_cells(r0, c0, r1, c1)
    if not coords.issubset(boundary):
        return None

    missing = list(boundary - coords)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None

    if gap_r == r0:
        gap_side = "top"
    elif gap_r == r1:
        gap_side = "bottom"
    elif gap_c == c0:
        gap_side = "left"
    else:
        gap_side = "right"

    interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    if interior.size == 0:
        return None

    line_points = np.argwhere(interior != bg)
    if len(line_points) < 2:
        return None
    line_colors = {int(interior[r, c]) for r, c in line_points}
    if len(line_colors) != 1 or color in line_colors:
        return None
    line_color = int(next(iter(line_colors)))

    rel_points = [(int(r), int(c)) for r, c in line_points]
    row_ids = {r for r, _c in rel_points}
    col_ids = {c for _r, c in rel_points}
    if len(row_ids) == 1:
        if gap_side not in {"top", "bottom"}:
            return None
        orient = "h"
        cols = sorted(c for _r, c in rel_points)
        if cols != list(range(cols[0], cols[0] + len(cols))):
            return None
    elif len(col_ids) == 1:
        if gap_side not in {"left", "right"}:
            return None
        orient = "v"
        rows = sorted(r for r, _c in rel_points)
        if rows != list(range(rows[0], rows[0] + len(rows))):
            return None
    else:
        return None

    line_coords = {(r0 + 1 + r, c0 + 1 + c) for r, c in rel_points}
    return {
        "frame_color": color,
        "line_color": line_color,
        "bbox": (r0, c0, r1, c1),
        "gap_side": gap_side,
        "line_orient": orient,
        "line_coords": line_coords,
    }


def _scan_order(bbox: tuple[int, int, int, int], gap_side: str, line_orient: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rows = range(r0 + 1, r1)
    cols = range(c0 + 1, c1)
    order: list[tuple[int, int]] = []
    if line_orient == "h":
        row_iter = rows if gap_side == "top" else range(r1 - 1, r0, -1)
        for r in row_iter:
            for c in cols:
                order.append((r, c))
    else:
        col_iter = cols if gap_side == "left" else range(c1 - 1, c0, -1)
        for c in col_iter:
            for r in rows:
                order.append((r, c))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Absorb matching exterior pixels into open rectangular fences and stack them inside.

    Input structure:
    - The background is a uniform color.
    - Several colored hollow rectangular fences appear, each with a single-pixel gap on one side.
    - Inside each fence there is one short straight line segment of another color, horizontal or vertical.
    - Outside the fence bboxes there are scattered singleton pixels whose colors match one of the interior line colors.

    Rule:
    - Detect each fence bbox, its gap side, and the unique interior line segment.
    - The line orientation chooses whether the interior is scanned row-by-row (horizontal line) or column-by-column (vertical line).
    - The gap side chooses the scan direction: top to bottom, bottom to top, left to right, or right to left.
    - Count same-color pixels outside all fence bboxes, erase them, then place that many pixels inside the matching fence along the scan order, skipping over the original line cells.

    Key insight:
    - The line and the gap encode two independent parts of one traversal order: the line chooses the stacking axis, while the gap chooses which side the fill advances from.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = []
    for comp in _components(g, bg):
        info = _frame_info(g, comp, bg)
        if info is not None:
            frames.append(info)
    if not frames:
        return g.copy()

    frames.sort(key=lambda item: item["bbox"])
    bbox_mask = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    out = g.copy()
    for frame in frames:
        line_coords = frame["line_coords"]
        r0, c0, r1, c1 = frame["bbox"]
        line_color = int(frame["line_color"])
        outside_mask = (g == line_color) & (~bbox_mask)
        count = int(outside_mask.sum())
        out[outside_mask] = bg

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if (r, c) not in line_coords:
                    out[r, c] = bg

        remaining = count
        for r, c in _scan_order(frame["bbox"], frame["gap_side"], frame["line_orient"]):
            if (r, c) in line_coords:
                continue
            if remaining <= 0:
                break
            out[r, c] = line_color
            remaining -= 1

    return out


# --------------- generator ---------------


MAX_GEN_ATTEMPTS = 400
FRAME_COUNT_RANGE = (1, 5)
FRAME_SIDE_MIN = 5
FRAME_SIDE_MAX_BY_COUNT = {1: 11, 2: 10, 3: 8, 4: 7}
GRID_RANGE_BY_COUNT = {
    1: (12, 18, 12, 18),
    2: (15, 22, 15, 24),
    3: (18, 26, 18, 26),
    4: (22, 30, 22, 30),
}
FRAME_PLACEMENT_TRIES = 250
SINGLETON_PLACEMENT_TRIES = 600


def _line_cells(spec: dict) -> list[tuple[int, int]]:
    if spec["line_orient"] == "h":
        return [(spec["line_row"], spec["line_col0"] + dc) for dc in range(spec["line_len"])]
    return [(spec["line_row0"] + dr, spec["line_col"]) for dr in range(spec["line_len"])]


def _sample_specs(rng: np.random.Generator) -> list[dict]:
    n_frames = int(rng.integers(*FRAME_COUNT_RANGE))
    palette = rng.choice(np.arange(1, 10), size=2 * n_frames, replace=False)
    frame_colors = [int(v) for v in palette[:n_frames]]
    line_colors = [int(v) for v in palette[n_frames:]]
    max_side = FRAME_SIDE_MAX_BY_COUNT[n_frames]

    specs: list[dict] = []
    for frame_color, line_color in zip(frame_colors, line_colors, strict=False):
        outer_h = int(rng.integers(FRAME_SIDE_MIN, max_side + 1))
        outer_w = int(rng.integers(FRAME_SIDE_MIN, max_side + 1))
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        orient = "h" if rng.random() < 0.5 else "v"
        if orient == "h":
            line_len = int(rng.integers(2, inner_w + 1))
            line_row = int(rng.integers(0, inner_h))
            line_col0 = int(rng.integers(0, inner_w - line_len + 1))
            gap_side = "top" if rng.random() < 0.5 else "bottom"
            gap_index = int(rng.integers(1, outer_w - 1))
            line_area = line_len
            spec = {
                "frame_color": frame_color,
                "line_color": line_color,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "gap_side": gap_side,
                "gap_index": gap_index,
                "line_orient": orient,
                "line_row": line_row,
                "line_col0": line_col0,
                "line_len": line_len,
            }
        else:
            line_len = int(rng.integers(2, inner_h + 1))
            line_row0 = int(rng.integers(0, inner_h - line_len + 1))
            line_col = int(rng.integers(0, inner_w))
            gap_side = "left" if rng.random() < 0.5 else "right"
            gap_index = int(rng.integers(1, outer_h - 1))
            line_area = line_len
            spec = {
                "frame_color": frame_color,
                "line_color": line_color,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "gap_side": gap_side,
                "gap_index": gap_index,
                "line_orient": orient,
                "line_row0": line_row0,
                "line_col": line_col,
                "line_len": line_len,
            }

        capacity = inner_h * inner_w - line_area
        if capacity >= 5 and rng.random() < 0.5:
            low = max(1, capacity // 3)
            high = min(capacity, low + 4)
            count = int(rng.integers(low, high + 1))
        else:
            count = int(rng.integers(1, min(capacity, 6) + 1))
        spec["count"] = count
        specs.append(spec)
    return specs


def _place_frames(rng: np.random.Generator, grid_h: int, grid_w: int, specs: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        h = int(spec["outer_h"])
        w = int(spec["outer_w"])
        placed_here = False
        for _ in range(FRAME_PLACEMENT_TRIES):
            r0 = int(rng.integers(0, grid_h - h + 1))
            c0 = int(rng.integers(0, grid_w - w + 1))
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            spec["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
            blocked[rr0:rr1, cc0:cc1] = True
            placed.append(spec)
            placed_here = True
            break
        if not placed_here:
            return None
    return placed, blocked


def _draw_frame(grid: np.ndarray, spec: dict, bg: int) -> None:
    r0, c0, r1, c1 = map(int, spec["bbox"])
    frame_color = int(spec["frame_color"])
    line_color = int(spec["line_color"])
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    if spec["gap_side"] == "top":
        grid[r0, c0 + int(spec["gap_index"])] = bg
    elif spec["gap_side"] == "bottom":
        grid[r1, c0 + int(spec["gap_index"])] = bg
    elif spec["gap_side"] == "left":
        grid[r0 + int(spec["gap_index"]), c0] = bg
    else:
        grid[r0 + int(spec["gap_index"]), c1] = bg

    for rr, cc in _line_cells(spec):
        grid[r0 + 1 + rr, c0 + 1 + cc] = line_color


def _place_singletons(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    specs: list[dict],
) -> list[tuple[int, int, int]] | None:
    h, w = grid.shape
    colors = [int(spec["line_color"]) for spec in specs for _ in range(int(spec["count"]))]
    rng.shuffle(colors)
    placed: list[tuple[int, int, int]] = []
    for color in colors:
        success = False
        for _ in range(SINGLETON_PLACEMENT_TRIES):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            if blocked[r, c]:
                continue
            grid[r, c] = color
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(h, r + 2)
            cc1 = min(w, c + 2)
            blocked[rr0:rr1, cc0:cc1] = True
            placed.append((r, c, color))
            success = True
            break
        if not success:
            return None
    return placed


def _expected_output(inp: np.ndarray, specs: list[dict], singletons: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for r, c, _color in singletons:
        out[int(r), int(c)] = bg

    for spec in specs:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        line_abs = {(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in _line_cells(spec)}
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if (r, c) not in line_abs:
                    out[r, c] = bg

        remaining = int(spec["count"])
        for r, c in _scan_order((r0, c0, r1, c1), str(spec["gap_side"]), str(spec["line_orient"])):
            if (r, c) in line_abs:
                continue
            if remaining <= 0:
                break
            out[r, c] = int(spec["line_color"])
            remaining -= 1
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _sample_specs(rng)
    n_frames = len(specs)
    h_lo, h_hi, w_lo, w_hi = GRID_RANGE_BY_COUNT[n_frames]
    grid_h = int(rng.integers(h_lo, h_hi + 1))
    grid_w = int(rng.integers(w_lo, w_hi + 1))
    placed = _place_frames(rng, grid_h, grid_w, specs)
    if placed is None:
        return None
    placed_specs, blocked = placed

    inp = np.zeros((grid_h, grid_w), dtype=int)
    for spec in placed_specs:
        _draw_frame(inp, spec, bg=0)

    singletons = _place_singletons(rng, inp, blocked, placed_specs)
    if singletons is None:
        return None

    expected = _expected_output(inp, placed_specs, singletons, bg=0)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
