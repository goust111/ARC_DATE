import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
STRIP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_of_value(grid: np.ndarray, value: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(value):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(value):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _line_orientation(coords: list[tuple[int, int]]) -> tuple[str, int]:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1 and len(cols) >= 2:
        return "h", next(iter(rows))
    if len(cols) == 1 and len(rows) >= 2:
        return "v", next(iter(cols))
    raise AssertionError("expected a horizontal or vertical line object")


def _sorted_axis(coords: list[tuple[int, int]], orient: str) -> list[int]:
    vals = [c if orient == "h" else r for r, c in coords]
    vals = sorted(set(int(v) for v in vals))
    assert vals
    return vals


def _is_consecutive(vals: list[int]) -> bool:
    return all(vals[i + 1] - vals[i] == 1 for i in range(len(vals) - 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each gray cargo segment along its colored conveyor toward the matching count markers.

    Input structure:
    - Black background.
    - Several 1-cell-wide colored strips, each horizontal or vertical and color-unique.
    - One gray 4-connected cargo on each strip, covering part of the strip.
    - Same-color count markers sit beyond one strip end on the same line, with a single black gap.

    Rule:
    - The markers indicate both the move direction (their side) and the move distance (their count).
    - Shift the gray cargo rigidly along the strip toward the markers.
    - If the requested shift would overshoot the strip, stop at the strip end instead.
    - Restore vacated strip cells to the strip color, and keep the markers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    gray_comps = _components_of_value(g, GRAY)
    non_special_colors = {int(v) for v in np.unique(g)} - {BG, GRAY}
    cargo_by_color: dict[int, list[tuple[int, int]]] = {}

    for comp in gray_comps:
        neigh_colors = set()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not _in_bounds(g.shape, nr, nc):
                    continue
                val = int(g[nr, nc])
                if val not in (BG, GRAY):
                    neigh_colors.add(val)
        assert len(neigh_colors) == 1, "each gray cargo must belong to exactly one strip color"
        color = next(iter(neigh_colors))
        assert color not in cargo_by_color, "each strip color must have exactly one cargo"
        cargo_by_color[color] = comp

    assert set(cargo_by_color) == non_special_colors, "every strip color must have exactly one gray cargo"

    for color, gray_comp in cargo_by_color.items():
        color_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
        assert color_coords, "strip color must remain visible somewhere"

        merged_coords = color_coords + gray_comp
        orient, fixed = _line_orientation(merged_coords)
        color_axis = _sorted_axis(color_coords, orient)
        gray_axis = _sorted_axis(gray_comp, orient)
        all_axis = sorted(set(color_axis) | set(gray_axis))

        assert _is_consecutive(gray_axis), "gray cargo must be a contiguous line segment"

        gap_indices = [i for i in range(len(all_axis) - 1) if all_axis[i + 1] - all_axis[i] == 2]
        assert len(gap_indices) == 1, "strip body and markers must be separated by one black cell"
        split = gap_indices[0] + 1
        left_part = all_axis[:split]
        right_part = all_axis[split:]
        assert left_part and right_part
        assert _is_consecutive(left_part) and _is_consecutive(right_part)

        gray_set = set(gray_axis)
        if gray_set.issubset(set(left_part)):
            body_axis = left_part
            marker_axis = right_part
            direction = 1
        elif gray_set.issubset(set(right_part)):
            body_axis = right_part
            marker_axis = left_part
            direction = -1
        else:
            raise AssertionError("gray cargo must lie on the strip body side, not the marker side")

        body_lo, body_hi = body_axis[0], body_axis[-1]
        cargo_len = len(gray_axis)
        old_start = gray_axis[0]
        marker_count = len(marker_axis)

        if direction > 0:
            new_start = min(old_start + marker_count, body_hi - cargo_len + 1)
        else:
            new_start = max(old_start - marker_count, body_lo)
        new_axis = list(range(new_start, new_start + cargo_len))

        for r, c in gray_comp:
            out[r, c] = int(color)

        for axis in new_axis:
            if orient == "h":
                out[fixed, axis] = GRAY
            else:
                out[axis, fixed] = GRAY

    return out


GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
STRIP_COUNT_RANGE = (1, 4)
STRIP_LEN_RANGE = (4, 10)
CARGO_LEN_RANGE = (1, 4)
MARKER_COUNT_RANGE = (1, 5)
ATTEMPTS = 400
PLACE_ATTEMPTS = 250


def _clearance_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _choose_orientation_sequence(n: int) -> list[str]:
    if n == 1:
        return [random.choice(("h", "v"))]
    if random.random() < 0.7:
        seq = ["h", "v"] + [random.choice(("h", "v")) for _ in range(n - 2)]
        random.shuffle(seq)
        return seq
    return [random.choice(("h", "v")) for _ in range(n)]


def _choose_branch_sequence(n: int) -> list[str]:
    if n == 1:
        return [random.choice(("free", "clipped"))]
    if random.random() < 0.7:
        seq = ["free", "clipped"] + [random.choice(("free", "clipped")) for _ in range(n - 2)]
        random.shuffle(seq)
        return seq
    return [random.choice(("free", "clipped")) for _ in range(n)]


def _pick_cargo_start(body_len: int, cargo_len: int, marker_count: int, marker_side: int, branch: str) -> int | None:
    max_shift = body_len - cargo_len
    if max_shift < 1:
        return None

    if branch == "free":
        if max_shift < marker_count:
            return None
        avail = random.randint(marker_count, max_shift)
    else:
        if marker_count <= 1:
            return None
        max_avail = min(marker_count - 1, max_shift)
        if max_avail < 1:
            return None
        avail = random.randint(1, max_avail)

    if marker_side < 0:
        return avail
    return max_shift - avail


def _strip_geometry(
    orient: str,
    fixed: int,
    body_start: int,
    body_len: int,
    cargo_start: int,
    cargo_len: int,
    marker_side: int,
    marker_count: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    body_cells: set[tuple[int, int]] = set()
    cargo_cells: set[tuple[int, int]] = set()
    marker_cells: set[tuple[int, int]] = set()
    gap_cells: set[tuple[int, int]] = set()

    for offset in range(body_len):
        axis = body_start + offset
        cell = (fixed, axis) if orient == "h" else (axis, fixed)
        body_cells.add(cell)

    for offset in range(cargo_len):
        axis = body_start + cargo_start + offset
        cell = (fixed, axis) if orient == "h" else (axis, fixed)
        cargo_cells.add(cell)

    if marker_side < 0:
        gap_axis = body_start - 1
        marker_axes = [body_start - 2 - i for i in range(marker_count)]
    else:
        gap_axis = body_start + body_len
        marker_axes = [body_start + body_len + 1 + i for i in range(marker_count)]

    gap_cells.add((fixed, gap_axis) if orient == "h" else (gap_axis, fixed))
    for axis in marker_axes:
        cell = (fixed, axis) if orient == "h" else (axis, fixed)
        marker_cells.add(cell)

    return body_cells, cargo_cells, marker_cells, gap_cells


def _place_one_strip(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    color: int,
    orient: str,
    branch: str,
) -> bool:
    h, w = grid.shape
    span_limit = w if orient == "h" else h
    fixed_limit = h if orient == "h" else w

    for _ in range(PLACE_ATTEMPTS):
        marker_count = random.randint(*MARKER_COUNT_RANGE)
        body_len = random.randint(STRIP_LEN_RANGE[0], min(STRIP_LEN_RANGE[1], span_limit - 4))
        cargo_len = random.randint(CARGO_LEN_RANGE[0], min(CARGO_LEN_RANGE[1], body_len - 1))
        marker_side = random.choice((-1, 1))
        cargo_start = _pick_cargo_start(body_len, cargo_len, marker_count, marker_side, branch)
        if cargo_start is None:
            continue

        if marker_side < 0:
            min_body_start = marker_count + 2
            max_body_start = span_limit - body_len - 2
        else:
            min_body_start = 1
            max_body_start = span_limit - (body_len + marker_count + 2)
        if min_body_start > max_body_start:
            continue

        body_start = random.randint(min_body_start, max_body_start)
        fixed = random.randint(1, fixed_limit - 2)

        body_cells, cargo_cells, marker_cells, gap_cells = _strip_geometry(
            orient, fixed, body_start, body_len, cargo_start, cargo_len, marker_side, marker_count
        )
        footprint = body_cells | marker_cells | gap_cells
        if not all(_in_bounds(grid.shape, r, c) for r, c in footprint):
            continue

        blocked = _clearance_cells(footprint, grid.shape)
        if any(cell in reserved for cell in blocked):
            continue

        for r, c in body_cells:
            grid[r, c] = int(color)
        for r, c in cargo_cells:
            grid[r, c] = GRAY
        for r, c in marker_cells:
            grid[r, c] = int(color)
        reserved.update(blocked)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    max_strips = min(STRIP_COUNT_RANGE[1], len(STRIP_COLORS))
    strip_count = random.randint(STRIP_COUNT_RANGE[0], max_strips)
    colors = random.sample(STRIP_COLORS, strip_count)
    orientations = _choose_orientation_sequence(strip_count)
    branches = _choose_branch_sequence(strip_count)

    reserved: set[tuple[int, int]] = set()

    for color, orient, branch in zip(colors, orientations, branches):
        if not _place_one_strip(grid, reserved, color, orient, branch):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {ATTEMPTS} attempts")
