import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
NON_SPECIAL_COLORS = (2, 3, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _analyze_grid(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    colors = [int(c) for c in np.unique(g) if int(c) not in (BLACK, BLUE, YELLOW)]
    if len(colors) != 2:
        raise ValueError("expected exactly two non-special rectangle colors")

    outer = None
    bbox = None
    for color in colors:
        box = _bbox_of_mask(g == color)
        r0, c0, r1, c1 = box
        if (
            g[r0, c0] == color
            and g[r0, c1] == color
            and g[r1, c0] == color
            and g[r1, c1] == color
        ):
            outer = color
            bbox = box
            break
    if outer is None or bbox is None:
        raise ValueError("failed to identify the outer frame color")

    inner = colors[0] if colors[1] == outer else colors[1]
    r0, c0, r1, c1 = bbox

    protrusions: list[dict[str, Any]] = []
    for r, c in np.argwhere(g == inner):
        rr = int(r)
        cc = int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        if rr < r0:
            side = "top"
        elif rr > r1:
            side = "bottom"
        elif cc < c0:
            side = "left"
        elif cc > c1:
            side = "right"
        else:
            continue

        same_neighbors = []
        for nr, nc in _neighbors4(rr, cc):
            if _in_bounds(g.shape, nr, nc) and g[nr, nc] == inner:
                same_neighbors.append((nr, nc))
        if len(same_neighbors) != 1:
            continue

        nr, nc = same_neighbors[0]
        if nr != rr:
            axis = "col"
            line = cc
        else:
            axis = "row"
            line = rr
        protrusions.append(
            {
                "pos": (rr, cc),
                "neighbor": (int(nr), int(nc)),
                "side": side,
                "axis": axis,
                "line": int(line),
            }
        )

    if not protrusions:
        raise ValueError("no protrusions found")

    row_lines = {p["line"] for p in protrusions if p["axis"] == "row"}
    col_lines = {p["line"] for p in protrusions if p["axis"] == "col"}
    if len(row_lines) != sum(p["axis"] == "row" for p in protrusions):
        raise ValueError("duplicate selected rows")
    if len(col_lines) != sum(p["axis"] == "col" for p in protrusions):
        raise ValueError("duplicate selected cols")

    return {
        "grid": g,
        "outer": outer,
        "inner": inner,
        "bbox": bbox,
        "protrusions": protrusions,
    }


def _build_rays(info: dict[str, Any]) -> list[dict[str, Any]]:
    r0, c0, r1, c1 = info["bbox"]
    rays: list[dict[str, Any]] = []
    for protrusion in info["protrusions"]:
        rr, cc = protrusion["pos"]
        axis = protrusion["axis"]
        side = protrusion["side"]
        if axis == "row":
            row = protrusion["line"]
            left_anchor = (row, cc) if side == "left" else (row, c0)
            right_anchor = (row, cc) if side == "right" else (row, c1)
            rays.append({"anchor": left_anchor, "dir": (0, -1), "axis": "row", "line": row})
            rays.append({"anchor": right_anchor, "dir": (0, 1), "axis": "row", "line": row})
        else:
            col = protrusion["line"]
            top_anchor = (rr, col) if side == "top" else (r0, col)
            bottom_anchor = (rr, col) if side == "bottom" else (r1, col)
            rays.append({"anchor": top_anchor, "dir": (-1, 0), "axis": "col", "line": col})
            rays.append({"anchor": bottom_anchor, "dir": (1, 0), "axis": "col", "line": col})
    rays.sort(key=lambda ray: (ray["axis"], ray["line"], ray["dir"][0], ray["dir"][1]))
    return rays


def _aligned_blue_maps(grid: np.ndarray) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    blue_by_col: dict[int, list[int]] = {}
    blue_by_row: dict[int, list[int]] = {}
    for r, c in np.argwhere(grid == BLUE):
        rr = int(r)
        cc = int(c)
        blue_by_col.setdefault(cc, []).append(rr)
        blue_by_row.setdefault(rr, []).append(cc)
    for vals in blue_by_col.values():
        vals.sort()
    for vals in blue_by_row.values():
        vals.sort()
    return blue_by_col, blue_by_row


def _nearest_signed_delta(current: int, aligned_positions: list[int]) -> int | None:
    candidates = []
    for pos in aligned_positions:
        delta = int(pos) - current
        if delta != 0:
            candidates.append(delta)
    if not candidates:
        return None
    candidates.sort(key=lambda delta: (abs(delta), delta))
    return candidates[0]


def _trace_rays(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    info = _analyze_grid(grid)
    out = np.asarray(grid, dtype=int).copy()
    rays = _build_rays(info)
    blue_by_col, blue_by_row = _aligned_blue_maps(out)
    ray_infos: list[dict[str, Any]] = []

    for ray in rays:
        r, c = ray["anchor"]
        dr, dc = ray["dir"]
        deflections = 0
        painted = 0
        stop_reason = "boundary"

        while True:
            nr = r + dr
            nc = c + dc
            if not _in_bounds(out.shape, nr, nc):
                stop_reason = "boundary"
                break
            if out[nr, nc] == YELLOW:
                stop_reason = "collision"
                break
            if out[nr, nc] != BLACK:
                stop_reason = "blocked"
                break

            r, c = nr, nc
            out[r, c] = YELLOW
            painted += 1

            if ray["axis"] == "row":
                delta = _nearest_signed_delta(r, blue_by_col.get(c, []))
                if delta is not None:
                    shift_r = r - 1 if delta > 0 else r + 1
                    shift_c = c
                    if not _in_bounds(out.shape, shift_r, shift_c):
                        stop_reason = "boundary"
                        break
                    if out[shift_r, shift_c] == YELLOW:
                        stop_reason = "collision"
                        break
                    if out[shift_r, shift_c] != BLACK:
                        stop_reason = "blocked"
                        break
                    r, c = shift_r, shift_c
                    out[r, c] = YELLOW
                    painted += 1
                    deflections += 1
            else:
                delta = _nearest_signed_delta(c, blue_by_row.get(r, []))
                if delta is not None:
                    shift_r = r
                    shift_c = c - 1 if delta > 0 else c + 1
                    if not _in_bounds(out.shape, shift_r, shift_c):
                        stop_reason = "boundary"
                        break
                    if out[shift_r, shift_c] == YELLOW:
                        stop_reason = "collision"
                        break
                    if out[shift_r, shift_c] != BLACK:
                        stop_reason = "blocked"
                        break
                    r, c = shift_r, shift_c
                    out[r, c] = YELLOW
                    painted += 1
                    deflections += 1

        ray_infos.append(
            {
                "ray": ray,
                "deflections": deflections,
                "painted": painted,
                "stop_reason": stop_reason,
            }
        )
    return out, ray_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast yellow rays from leaf-selected rows and columns, bending each step away from aligned blue control pixels.

    Input structure:
    - A black grid contains one central bicolor rectangle with an outer frame color and an inner fill color.
    - Several single-cell inner-color leaf protrusions extend one cell outside the rectangle, with each leaf's only same-color neighbor lying back inside the rectangle border.
    - Separate blue pixels lie away from the rectangle and act as off-path deflection controls.

    Rule:
    - A protrusion whose only same-color neighbor is left or right selects its row; a protrusion whose neighbor is above or below selects its column.
    - For each selected row, cast a yellow ray outward from both the left and right exterior sides of the rectangle; for each selected column, cast a yellow ray outward from both the top and bottom exterior sides.
    - After each forward step, if the ray is aligned with any blue pixel on the perpendicular axis, shift the ray one cell away from the nearest such blue pixel, then continue in the original travel direction.
    - A ray stops when its next forward step would leave the grid or enter an already yellow cell; the rectangle, the protrusions, and the blue pixels remain unchanged.

    Key insight:
    - The leaves decide which rows and columns are active, but the blue pixels decide the staircase shape, so neither cue alone is sufficient to reconstruct the output.
    """
    out, _ = _trace_rays(np.asarray(grid, dtype=int))
    return out


def _choose_dims() -> tuple[int, int, int, int]:
    while True:
        height = random.randint(18, 24)
        width = random.randint(18, 24)
        rect_h = random.randint(7, 9)
        rect_w = random.randint(8, 10)
        if height - rect_h >= 8 and width - rect_w >= 8:
            return height, width, rect_h, rect_w


def _choose_two_with_gap(values: list[int], min_gap: int) -> tuple[int, int]:
    pairs = [(a, b) for i, a in enumerate(values) for b in values[i + 1 :] if b - a >= min_gap]
    if not pairs:
        raise ValueError("no feasible pair")
    return random.choice(pairs)


def _place_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    rect_h: int,
    rect_w: int,
    outer: int,
    inner: int,
) -> tuple[int, int]:
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = outer
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner
    return r1, c1


def _add_protrusions(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    inner: int,
    row_lines: tuple[int, int],
    col_lines: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    left_row, right_row = random.sample(list(row_lines), 2)
    top_col, bottom_col = random.sample(list(col_lines), 2)

    grid[left_row, c0] = inner
    grid[left_row, c0 - 1] = inner

    grid[right_row, c1] = inner
    grid[right_row, c1 + 1] = inner

    grid[r0, top_col] = inner
    grid[r0 - 1, top_col] = inner

    grid[r1, bottom_col] = inner
    grid[r1 + 1, bottom_col] = inner


def _add_blue_controls(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    row_lines: tuple[int, int],
    col_lines: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    low_row, high_row = sorted(row_lines)
    low_col, high_col = sorted(col_lines)

    left_blue_row = random.randint(low_row + 1, high_row - 1)
    right_blue_row = random.randint(low_row + 1, high_row - 1)
    top_blue_col = random.randint(low_col + 1, high_col - 1)
    bottom_blue_col = random.randint(low_col + 1, high_col - 1)

    left_blue_col = random.randint(0, c0 - 3)
    right_blue_col = random.randint(c1 + 3, grid.shape[1] - 1)
    top_blue_row = random.randint(0, r0 - 3)
    bottom_blue_row = random.randint(r1 + 3, grid.shape[0] - 1)

    grid[left_blue_row, left_blue_col] = BLUE
    grid[right_blue_row, right_blue_col] = BLUE
    grid[top_blue_row, top_blue_col] = BLUE
    grid[bottom_blue_row, bottom_blue_col] = BLUE


def _candidate_grid() -> np.ndarray:
    height, width, rect_h, rect_w = _choose_dims()
    grid = np.zeros((height, width), dtype=int)
    outer, inner = random.sample(list(NON_SPECIAL_COLORS), 2)

    top_margin = random.randint(4, height - rect_h - 4)
    left_margin = random.randint(4, width - rect_w - 4)
    r0 = top_margin
    c0 = left_margin
    r1, c1 = _place_rectangle(grid, r0, c0, rect_h, rect_w, outer, inner)
    bbox = (r0, c0, r1, c1)

    row_values = list(range(r0 + 1, r1))
    col_values = list(range(c0 + 1, c1))
    row_lines = _choose_two_with_gap(row_values, 3)
    col_lines = _choose_two_with_gap(col_values, 3)

    _add_protrusions(grid, bbox, inner, row_lines, col_lines)
    _add_blue_controls(grid, bbox, row_lines, col_lines)
    return grid


def _is_adjacent_to_protrusion(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    info = _analyze_grid(grid)
    protrusion_positions = {p["pos"] for p in info["protrusions"]}
    r, c = pos
    for nr, nc in _neighbors4(r, c):
        if (nr, nc) in protrusion_positions:
            return True
    return False


def _valid_candidate(grid: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
    except Exception:
        return False

    if len(info["protrusions"]) != 4:
        return False
    if sum(p["axis"] == "row" for p in info["protrusions"]) != 2:
        return False
    if sum(p["axis"] == "col" for p in info["protrusions"]) != 2:
        return False

    blue_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    if len(blue_cells) != 4:
        return False
    for pos in blue_cells:
        if _is_adjacent_to_protrusion(grid, pos):
            return False

    try:
        out, ray_infos = _trace_rays(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == YELLOW) < 16:
        return False

    for ray_info in ray_infos:
        if ray_info["deflections"] < 1:
            return False
        if ray_info["painted"] < 2:
            return False
        if ray_info["stop_reason"] != "boundary":
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _candidate_grid()
        if _valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
