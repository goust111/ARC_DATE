from __future__ import annotations

import numpy as np

BG = 0
BAR = 5
CONTROL_COLORS = (1, 2, 3, 4)
ALL_SIDES = ("left", "right", "top", "bottom")
RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bbox")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _contiguous(values: np.ndarray) -> bool:
    return bool(values.size) and np.all(np.diff(values) == 1)


def _cross_footprint(
    shape: tuple[int, int],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    hr0: int,
    hr1: int,
    vc0: int,
    vc1: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[hr0 : hr1 + 1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, vc0 : vc1 + 1] = True
    return mask


def _infer_structure(grid: np.ndarray) -> dict:
    gray_mask = grid == BAR
    if not np.any(gray_mask):
        raise AssertionError("expected a gray bar")

    r0, c0, r1, c1 = _bbox(gray_mask)
    sub = gray_mask[r0 : r1 + 1, c0 : c1 + 1]
    row_counts = sub.sum(axis=1)
    col_counts = sub.sum(axis=0)

    horizontal_rows_rel = np.flatnonzero(row_counts > sub.shape[1] // 2)
    vertical_cols_rel = np.flatnonzero(col_counts > sub.shape[0] // 2)
    assert _contiguous(horizontal_rows_rel)
    assert _contiguous(vertical_cols_rel)

    hr0 = r0 + int(horizontal_rows_rel[0])
    hr1 = r0 + int(horizontal_rows_rel[-1])
    vc0 = c0 + int(vertical_cols_rel[0])
    vc1 = c0 + int(vertical_cols_rel[-1])

    footprint = _cross_footprint(grid.shape, r0, c0, r1, c1, hr0, hr1, vc0, vc1)
    notch_mask = footprint & (grid == BG)
    notches = {(int(r), int(c)) for r, c in np.argwhere(notch_mask)}

    row_bounds: dict[int, tuple[int, int]] = {}
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(footprint[r])
        if cols.size:
            row_bounds[int(r)] = (int(cols[0]), int(cols[-1]))

    col_bounds: dict[int, tuple[int, int]] = {}
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(footprint[:, c])
        if rows.size:
            col_bounds[int(c)] = (int(rows[0]), int(rows[-1]))

    return {
        "bbox": (r0, c0, r1, c1),
        "hr0": hr0,
        "hr1": hr1,
        "vc0": vc0,
        "vc1": vc1,
        "footprint": footprint,
        "notches": notches,
        "row_bounds": row_bounds,
        "col_bounds": col_bounds,
    }


def _trace_control(
    grid: np.ndarray,
    footprint: np.ndarray,
    notches: set[tuple[int, int]],
    row_bounds: dict[int, tuple[int, int]],
    col_bounds: dict[int, tuple[int, int]],
    r: int,
    c: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], str]:
    if r in row_bounds:
        left, right = row_bounds[r]
        if c < left:
            path = [(r, cc) for cc in range(c + 1, left)]
            assert all(int(grid[rr, cc]) == BG for rr, cc in path)
            return path, (r, left), "horizontal"
        if c > right:
            path = [(r, cc) for cc in range(right + 1, c)]
            assert all(int(grid[rr, cc]) == BG for rr, cc in path)
            return path, (r, right), "horizontal"

    if c in col_bounds:
        top, bottom = col_bounds[c]
        if r < top:
            path = [(rr, c) for rr in range(r + 1, top)]
            assert all(int(grid[rr, cc]) == BG for rr, cc in path)
            return path, (top, c), "vertical"
        if r > bottom:
            path = [(rr, c) for rr in range(bottom + 1, r)]
            assert all(int(grid[rr, cc]) == BG for rr, cc in path)
            return path, (bottom, c), "vertical"

    raise AssertionError("control pixel is not aligned with the bar")


def _fill_chain(
    out: np.ndarray,
    footprint: np.ndarray,
    notches: set[tuple[int, int]],
    notch: tuple[int, int],
    color: int,
    incident_axis: str,
) -> None:
    r, c = notch
    out[r, c] = int(color)
    directions = [(-1, 0), (1, 0)] if incident_axis == "horizontal" else [(0, -1), (0, 1)]

    for dr, dc in directions:
        rr, cc = r + dr, c + dc
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and footprint[rr, cc]:
            if (rr, cc) in notches:
                break
            out[rr, cc] = int(color)
            rr += dr
            cc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """Extend control-pixel rays into a notched central cross, then fill each hit notch along the perpendicular axis.

    Input structure:
    - A black background contains one large gray cross-shaped bar made from one horizontal strip and one vertical strip.
    - Some 1x1 black notches are carved from the outer boundary of that gray bar.
    - Several colored control pixels lie outside the bar and align by row or column with specific notches.

    Rule:
    - Infer the gray cross footprint and its boundary notches.
    - For each non-black, non-gray control pixel, extend a straight ray toward the bar, coloring only the background cells before the bar boundary.
    - If the first boundary cell on that ray is a notch, activate the notch.
    - Fill the activated notch and continue inside the bar along the axis perpendicular to the ray until another notch or the end of that strip.
    - Leave untouched notches black and keep all other cells unchanged.

    Key insight:
    - The growth direction is not determined by the notch geometry alone: a horizontal incoming ray forces a vertical fill, while a vertical incoming ray forces a horizontal fill.
    """
    g = np.asarray(grid, dtype=int)
    structure = _infer_structure(g)

    out = g.copy()
    control_coords = np.argwhere((g != BG) & (g != BAR))
    for rr, cc in control_coords:
        r = int(rr)
        c = int(cc)
        color = int(g[r, c])
        path, boundary, incident_axis = _trace_control(
            g,
            structure["footprint"],
            structure["notches"],
            structure["row_bounds"],
            structure["col_bounds"],
            r,
            c,
        )
        for pr, pc in path:
            out[pr, pc] = color
        if boundary in structure["notches"]:
            _fill_chain(out, structure["footprint"], structure["notches"], boundary, color, incident_axis)

    return out


def _build_plan() -> dict | None:
    height = int(RNG.integers(17, 24))
    width = int(RNG.integers(17, 24))
    thickness = int(RNG.integers(3, 5))

    left_margin = int(RNG.integers(2, 6))
    right_margin = int(RNG.integers(2, 6))
    top_margin = int(RNG.integers(2, 6))
    bottom_margin = int(RNG.integers(2, 6))

    if width - left_margin - right_margin < thickness + 4:
        return None
    if height - top_margin - bottom_margin < thickness + 4:
        return None

    c0 = left_margin
    c1 = width - 1 - right_margin
    r0 = top_margin
    r1 = height - 1 - bottom_margin

    vc0_lo = c0 + 1
    vc0_hi = c1 - thickness
    hr0_lo = r0 + 1
    hr0_hi = r1 - thickness
    if vc0_lo > vc0_hi or hr0_lo > hr0_hi:
        return None

    vc0 = int(RNG.integers(vc0_lo, vc0_hi + 1))
    vc1 = vc0 + thickness - 1
    hr0 = int(RNG.integers(hr0_lo, hr0_hi + 1))
    hr1 = hr0 + thickness - 1

    footprint = _cross_footprint((height, width), r0, c0, r1, c1, hr0, hr1, vc0, vc1)

    total_notches = 4 if RNG.random() < 0.7 else 3
    if total_notches == 4:
        notch_sides = list(ALL_SIDES)
    else:
        while True:
            notch_sides = list(RNG.choice(np.array(ALL_SIDES), size=3, replace=False))
            if any(side in ("left", "right") for side in notch_sides) and any(
                side in ("top", "bottom") for side in notch_sides
            ):
                break

    if total_notches == 4 and RNG.random() < 0.55:
        active_sides = [side for side in ALL_SIDES if side != str(RNG.choice(np.array(ALL_SIDES)))]
    else:
        active_sides = notch_sides[:]

    if not any(side in ("left", "right") for side in active_sides):
        return None
    if not any(side in ("top", "bottom") for side in active_sides):
        return None

    h_rows = list(range(hr0, hr1 + 1))
    v_cols = list(range(vc0, vc1 + 1))
    RNG.shuffle(h_rows)
    RNG.shuffle(v_cols)

    notch_positions: dict[str, tuple[int, int]] = {}
    if "left" in notch_sides:
        notch_positions["left"] = (h_rows.pop(), c0)
    if "right" in notch_sides:
        row = h_rows.pop() if h_rows else int(RNG.integers(hr0, hr1 + 1))
        notch_positions["right"] = (row, c1)
    if "top" in notch_sides:
        notch_positions["top"] = (r0, v_cols.pop())
    if "bottom" in notch_sides:
        col = v_cols.pop() if v_cols else int(RNG.integers(vc0, vc1 + 1))
        notch_positions["bottom"] = (r1, col)

    colors = [int(v) for v in RNG.choice(np.array(CONTROL_COLORS), size=len(active_sides), replace=False)]
    RNG.shuffle(colors)

    controls: dict[str, dict] = {}
    for side, color in zip(active_sides, colors):
        nr, nc = notch_positions[side]
        if side == "left":
            if c0 < 2:
                return None
            cc = int(RNG.integers(0, c0 - 1))
            controls[side] = {"color": color, "pos": (nr, cc)}
        elif side == "right":
            if c1 + 2 >= width:
                return None
            cc = int(RNG.integers(c1 + 2, width))
            controls[side] = {"color": color, "pos": (nr, cc)}
        elif side == "top":
            if r0 < 2:
                return None
            rr = int(RNG.integers(0, r0 - 1))
            controls[side] = {"color": color, "pos": (rr, nc)}
        else:
            if r1 + 2 >= height:
                return None
            rr = int(RNG.integers(r1 + 2, height))
            controls[side] = {"color": color, "pos": (rr, nc)}

    plan = {
        "shape": {
            "height": height,
            "width": width,
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "hr0": hr0,
            "hr1": hr1,
            "vc0": vc0,
            "vc1": vc1,
        },
        "footprint": footprint,
        "notches": notch_positions,
        "controls": controls,
    }
    if _validate_plan(plan):
        return plan
    return None


def _render_input(plan: dict) -> np.ndarray:
    shape = plan["shape"]
    grid = np.full((shape["height"], shape["width"]), BG, dtype=int)
    grid[shape["hr0"] : shape["hr1"] + 1, shape["c0"] : shape["c1"] + 1] = BAR
    grid[shape["r0"] : shape["r1"] + 1, shape["vc0"] : shape["vc1"] + 1] = BAR

    for r, c in plan["notches"].values():
        grid[r, c] = BG
    for info in plan["controls"].values():
        r, c = info["pos"]
        grid[r, c] = int(info["color"])
    return grid


def _plan_fill_cells(plan: dict, side: str) -> set[tuple[int, int]]:
    shape = plan["shape"]
    notch_set = set(plan["notches"].values())
    r, c = plan["notches"][side]
    filled = {(r, c)}

    if side in ("left", "right"):
        for step in (-1, 1):
            rr = r + step
            while shape["hr0"] <= rr <= shape["hr1"]:
                if (rr, c) in notch_set:
                    break
                filled.add((rr, c))
                rr += step
    else:
        for step in (-1, 1):
            cc = c + step
            while shape["vc0"] <= cc <= shape["vc1"]:
                if (r, cc) in notch_set:
                    break
                filled.add((r, cc))
                cc += step

    return filled


def _render_output_from_plan(plan: dict) -> np.ndarray:
    shape = plan["shape"]
    out = _render_input(plan)
    for side, info in plan["controls"].items():
        color = int(info["color"])
        cr, cc = info["pos"]
        nr, nc = plan["notches"][side]

        if side == "left":
            for c in range(cc + 1, nc):
                out[cr, c] = color
        elif side == "right":
            for c in range(nc + 1, cc):
                out[cr, c] = color
        elif side == "top":
            for r in range(cr + 1, nr):
                out[r, cc] = color
        else:
            for r in range(nr + 1, cr):
                out[r, cc] = color

        for rr, rc in _plan_fill_cells(plan, side):
            out[rr, rc] = color

    return out


def _validate_plan(plan: dict) -> bool:
    shape = plan["shape"]
    height = shape["height"]
    width = shape["width"]
    if max(height, width) > 30:
        return False

    notches = plan["notches"]
    controls = plan["controls"]
    if not (3 <= len(notches) <= 4 and 3 <= len(controls) <= 4):
        return False
    if not set(controls).issubset(notches):
        return False
    if not any(side in ("left", "right") for side in controls):
        return False
    if not any(side in ("top", "bottom") for side in controls):
        return False

    notch_set = set(notches.values())
    if len(notch_set) != len(notches):
        return False
    colors = [int(info["color"]) for info in controls.values()]
    if len(set(colors)) != len(colors):
        return False
    control_positions = [tuple(info["pos"]) for info in controls.values()]
    if len(set(control_positions)) != len(control_positions):
        return False

    footprint = plan["footprint"]
    ray_cells: list[set[tuple[int, int]]] = []
    fill_cells: list[set[tuple[int, int]]] = []
    for side, info in controls.items():
        r, c = info["pos"]
        if footprint[r, c]:
            return False

        nr, nc = notches[side]
        if not footprint[nr, nc]:
            return False

        if side == "left":
            if c >= nc - 1:
                return False
            path = {(r, cc) for cc in range(c + 1, nc)}
        elif side == "right":
            if c <= nc + 1:
                return False
            path = {(r, cc) for cc in range(nc + 1, c)}
        elif side == "top":
            if r >= nr - 1:
                return False
            path = {(rr, c) for rr in range(r + 1, nr)}
        else:
            if r <= nr + 1:
                return False
            path = {(rr, c) for rr in range(nr + 1, r)}

        if path & notch_set:
            return False
        ray_cells.append(path)
        fill_cells.append(_plan_fill_cells(plan, side))

    for i, cells_a in enumerate(ray_cells):
        for cells_b in ray_cells[i + 1 :]:
            if cells_a & cells_b:
                return False

    for i, cells_a in enumerate(fill_cells):
        for cells_b in fill_cells[i + 1 :]:
            if cells_a & cells_b:
                return False

    if set().union(*ray_cells) & set().union(*fill_cells):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    plan = _build_plan()
    if plan is None:
        return None

    inp = _render_input(plan)
    expected = _render_output_from_plan(plan)

    structure = _infer_structure(inp)
    if structure["notches"] != set(plan["notches"].values()):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-notch chain-fill input")
