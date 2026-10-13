from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
TEAL = 8
GROUP_COLORS = (2, 3, 1)
PATTERNS = (("H", "H"), ("H", "V"), ("V", "H"), ("V", "V"))
MAX_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    if lo > hi:
        return None
    return int(lo), int(hi)


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _outline_from_missing_corner(
    r0: int, c0: int, r1: int, c1: int, missing_corner: tuple[int, int]
) -> set[tuple[int, int]]:
    span = int(r1 - r0)
    outline: set[tuple[int, int]] = set()
    if missing_corner == (r0, c0):
        outline.update((r0, c) for c in range(c0, c1 + 1))
        outline.update((r, c0) for r in range(r0, r1 + 1))
        outline.update((r0 + i, c1 - i) for i in range(span + 1))
    elif missing_corner == (r0, c1):
        outline.update((r0, c) for c in range(c0, c1 + 1))
        outline.update((r, c1) for r in range(r0, r1 + 1))
        outline.update((r0 + i, c0 + i) for i in range(span + 1))
    elif missing_corner == (r1, c0):
        outline.update((r1, c) for c in range(c0, c1 + 1))
        outline.update((r, c0) for r in range(r0, r1 + 1))
        outline.update((r0 + i, c0 + i) for i in range(span + 1))
    else:
        assert missing_corner == (r1, c1)
        outline.update((r1, c) for c in range(c0, c1 + 1))
        outline.update((r, c1) for r in range(r0, r1 + 1))
        outline.update((r0 + i, c1 - i) for i in range(span + 1))
    return outline


def _choose_bridge(
    src_bbox: tuple[int, int, int, int], dst_bbox: tuple[int, int, int, int]
) -> tuple[str, list[tuple[int, int]]]:
    sr0, sc0, sr1, sc1 = src_bbox
    tr0, tc0, tr1, tc1 = dst_bbox
    candidates: list[tuple[int, int, str, list[tuple[int, int]]]] = []

    row_band = _interval_overlap(sr0, sr1, tr0, tr1)
    if row_band is not None:
        row = (row_band[0] + row_band[1]) // 2
        if sc1 < tc0:
            cols = list(range(sc1, tc0 + 1))
            candidates.append((tc0 - sc1, 0, "H", [(row, c) for c in cols]))
        elif tc1 < sc0:
            cols = list(range(tc1, sc0 + 1))
            candidates.append((sc0 - tc1, 0, "H", [(row, c) for c in cols]))

    col_band = _interval_overlap(sc0, sc1, tc0, tc1)
    if col_band is not None:
        col = (col_band[0] + col_band[1]) // 2
        if sr1 < tr0:
            rows = list(range(sr1, tr0 + 1))
            candidates.append((tr0 - sr1, 1, "V", [(r, col) for r in rows]))
        elif tr1 < sr0:
            rows = list(range(tr1, sr0 + 1))
            candidates.append((sr0 - tr1, 1, "V", [(r, col) for r in rows]))

    if not candidates:
        raise ValueError("Triangles are not bridge-alignable")
    candidates.sort(key=lambda item: (item[0], item[1]))
    _gap, _axis_rank, axis, cells = candidates[0]
    return axis, cells


def _parse_scene(grid: np.ndarray) -> tuple[int, list[int], dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rows_with_fg = [r for r in range(g.shape[0]) if np.any(g[r] != bg)]
    if not rows_with_fg:
        raise ValueError("No foreground found")
    ctrl_r = rows_with_fg[-1]
    control = [int(g[ctrl_r, c]) for c in range(g.shape[1]) if int(g[ctrl_r, c]) != bg]
    if len(control) != 2 or control[0] == control[1]:
        raise ValueError("Control row must contain two distinct colors")

    groups: dict[int, dict[str, object]] = {}
    body = g[:ctrl_r]
    for color in GROUP_COLORS:
        coords = np.argwhere(body == int(color))
        if coords.shape != (4, 2):
            raise ValueError(f"Color {color} must have exactly four marker pixels")
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r1 - r0 != c1 - c0 or r1 - r0 < 3:
            raise ValueError(f"Color {color} does not form a square bbox")
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        pts = {tuple(map(int, p)) for p in coords}
        hit_corners = pts & corners
        missing = list(corners - hit_corners)
        interior = [pt for pt in pts if pt not in corners]
        if len(hit_corners) != 3 or len(missing) != 1 or len(interior) != 1:
            raise ValueError(f"Color {color} markers are malformed")
        dr, dc = interior[0]
        if not (r0 < dr < r1 and c0 < dc < c1):
            raise ValueError(f"Color {color} decoy is not strictly interior")
        groups[int(color)] = {
            "bbox": (r0, c0, r1, c1),
            "missing": missing[0],
            "decoy": interior[0],
        }
    return bg, control, groups


def main(input_grid: np.ndarray) -> np.ndarray:
    """Construct teal right-triangle outlines from missing-corner marker groups, then connect them by control-sequence bridges.

    Input structure:
    - Black background with exactly three colored marker groups in red, green, and blue, plus a two-pixel control sequence on the bottom-most foreground row.
    - Each group contains four same-color pixels whose minimal bounding box is a square: three pixels occupy square corners and the fourth is a strict interior decoy.
    - The three square bboxes are disjoint, and every pair that must be connected shares either a row band or a column band.

    Rule:
    - For each color group, find the unique empty corner of its square bbox and treat it as the right-angle vertex of a right isosceles triangle whose other vertices are the two adjacent marked corners.
    - Draw that triangle outline in teal using the two axis-aligned legs and the diagonal hypotenuse, while leaving all original non-background pixels unchanged.
    - Read the bottom control pixels from left to right as [c0, c1], connect c0's triangle to c1's triangle with a bridge colored c0, then connect c1's triangle to the remaining third triangle with a bridge colored c1.
    - Each bridge is a one-pixel-wide straight segment along the middle row or column of the shared band between the two bboxes, choosing the shorter valid axis if both exist, and only blank cells are filled.

    Key insight:
    - The connectable objects are not explicitly present in the input: they must first be reconstructed from the missing bbox corners, and only then can the control sequence determine which inferred triangles get bridged and from which sides.
    """
    g = np.asarray(input_grid, dtype=int)
    out = g.copy()
    bg, control, groups = _parse_scene(g)

    for color in GROUP_COLORS:
        bbox = groups[color]["bbox"]
        missing = groups[color]["missing"]
        r0, c0, r1, c1 = bbox
        for r, c in _outline_from_missing_corner(r0, c0, r1, c1, missing):
            if out[r, c] == bg:
                out[r, c] = TEAL

    c0, c1 = control
    c2 = next(color for color in GROUP_COLORS if color not in (c0, c1))
    for src, dst in ((c0, c1), (c1, c2)):
        _axis, cells = _choose_bridge(groups[src]["bbox"], groups[dst]["bbox"])
        for r, c in cells:
            if out[r, c] == bg:
                out[r, c] = int(src)

    return out


def _sample_relative_bbox(
    base: tuple[int, int, int, int], span: int, mode: str, side: str, h: int, w: int, max_r1: int
) -> tuple[int, int, int, int] | None:
    br0, bc0, br1, bc1 = base
    if mode == "H":
        min_r0 = max(0, br0 - span)
        max_r0 = min(max_r1 - span, br1)
        if min_r0 > max_r0:
            return None
        if side == "left":
            max_gap = bc0 - span - 1
            if max_gap < 1:
                return None
            gap = random.randint(1, min(5, max_gap))
            c1 = bc0 - gap - 1
            c0 = c1 - span
        else:
            max_gap = w - bc1 - span - 2
            if max_gap < 1:
                return None
            gap = random.randint(1, min(5, max_gap))
            c0 = bc1 + gap + 1
            c1 = c0 + span
        r0 = random.randint(min_r0, max_r0)
        return (r0, c0, r0 + span, c1)

    min_c0 = max(0, bc0 - span)
    max_c0 = min(w - 1 - span, bc1)
    if min_c0 > max_c0:
        return None
    c0 = random.randint(min_c0, max_c0)
    if side == "up":
        max_gap = br0 - span - 1
        if max_gap < 1:
            return None
        gap = random.randint(1, min(5, max_gap))
        r1 = br0 - gap - 1
        r0 = r1 - span
    else:
        max_gap = max_r1 - br1 - span - 1
        if max_gap < 1:
            return None
        gap = random.randint(1, min(5, max_gap))
        r0 = br1 + gap + 1
        r1 = r0 + span
    return (r0, c0, r1, c0 + span)


def _pick_directions(pattern: tuple[str, str]) -> tuple[str, str]:
    first, second = pattern
    if pattern == ("H", "H"):
        if random.random() < 0.5:
            return "left", "right"
        return "right", "left"
    if pattern == ("V", "V"):
        if random.random() < 0.5:
            return "up", "down"
        return "down", "up"
    first_dir = random.choice(["left", "right"]) if first == "H" else random.choice(["up", "down"])
    second_dir = random.choice(["left", "right"]) if second == "H" else random.choice(["up", "down"])
    return first_dir, second_dir


def _place_chain(
    h: int, w: int, order: tuple[int, int, int], spans: dict[int, int], pattern: tuple[str, str]
) -> dict[int, tuple[int, int, int, int]] | None:
    middle = order[1]
    max_r1 = h - 3
    if max_r1 < 6:
        return None
    dir1, dir2 = _pick_directions(pattern)
    for _ in range(250):
        sb = spans[middle]
        max_r0 = max_r1 - sb
        max_c0 = w - 1 - sb
        if max_r0 < 0 or max_c0 < 0:
            return None
        br0 = random.randint(0, max_r0)
        bc0 = random.randint(0, max_c0)
        base = (br0, bc0, br0 + sb, bc0 + sb)
        first = _sample_relative_bbox(base, spans[order[0]], pattern[0], dir1, h, w, max_r1)
        second = _sample_relative_bbox(base, spans[order[2]], pattern[1], dir2, h, w, max_r1)
        if first is None or second is None:
            continue
        if _rects_intersect(first, second):
            continue
        boxes = {order[0]: first, order[1]: base, order[2]: second}
        if sum(_rects_intersect(boxes[a], boxes[b]) for a in GROUP_COLORS for b in GROUP_COLORS if a < b):
            continue
        return boxes
    return None


def _marker_points(
    bbox: tuple[int, int, int, int], missing: tuple[int, int], decoy: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    return [corner for corner in corners if corner != missing] + [decoy]


def _render_input(spec: dict[str, object]) -> np.ndarray:
    h = int(spec["height"])
    w = int(spec["width"])
    grid = np.zeros((h, w), dtype=int)
    boxes = spec["boxes"]
    for color in GROUP_COLORS:
        info = boxes[color]
        for r, c in _marker_points(info["bbox"], info["missing"], info["decoy"]):
            grid[r, c] = int(color)
    ctrl_positions = spec["control_positions"]
    for col, color in zip(ctrl_positions, spec["control"]):
        grid[h - 1, int(col)] = int(color)
    return grid


def _render_output_from_spec(spec: dict[str, object]) -> np.ndarray:
    out = _render_input(spec)
    boxes = spec["boxes"]
    for color in GROUP_COLORS:
        bbox = boxes[color]["bbox"]
        missing = boxes[color]["missing"]
        for r, c in _outline_from_missing_corner(*bbox, missing):
            if out[r, c] == BG:
                out[r, c] = TEAL
    order = spec["visit_order"]
    for src, dst in zip(order, order[1:]):
        _axis, cells = _choose_bridge(boxes[src]["bbox"], boxes[dst]["bbox"])
        for r, c in cells:
            if out[r, c] == BG:
                out[r, c] = int(src)
    return out


def _validate_case(spec: dict[str, object], grid: np.ndarray) -> bool:
    if max(grid.shape) > 30:
        return False
    if not np.array_equal(grid, _render_input(spec)):
        return False
    boxes = spec["boxes"]
    order = spec["visit_order"]
    pattern = spec["pattern"]

    for color in GROUP_COLORS:
        bbox = boxes[color]["bbox"]
        r0, c0, r1, c1 = bbox
        if r1 - r0 != c1 - c0 or r1 >= grid.shape[0] - 1:
            return False
        decoy = boxes[color]["decoy"]
        if decoy in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            return False

    for idx, (src, dst) in enumerate(zip(order, order[1:])):
        axis, cells = _choose_bridge(boxes[src]["bbox"], boxes[dst]["bbox"])
        if axis != pattern[idx]:
            return False
        if not any(grid[r, c] == BG for r, c in cells):
            return False

    expected = _render_output_from_spec(spec)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _random_spec() -> dict[str, object] | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    control = tuple(random.sample(list(GROUP_COLORS), 2))
    remaining = next(color for color in GROUP_COLORS if color not in control)
    order = (control[0], control[1], remaining)
    pattern = random.choice(PATTERNS)
    spans = {color: random.randint(3, 6) for color in GROUP_COLORS}
    boxes = _place_chain(h, w, order, spans, pattern)
    if boxes is None:
        return None

    box_info: dict[int, dict[str, object]] = {}
    for color in GROUP_COLORS:
        bbox = boxes[color]
        r0, c0, r1, c1 = bbox
        missing = random.choice([(r0, c0), (r0, c1), (r1, c0), (r1, c1)])
        decoy = (random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1))
        box_info[color] = {"bbox": bbox, "missing": missing, "decoy": decoy}

    x0 = random.randint(0, w - 4)
    x1 = random.randint(x0 + 2, w - 1)
    return {
        "height": h,
        "width": w,
        "control": control,
        "visit_order": order,
        "pattern": pattern,
        "control_positions": (x0, x1),
        "boxes": box_info,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        spec = _random_spec()
        if spec is None:
            continue
        grid = _render_input(spec)
        try:
            if _validate_case(spec, grid):
                return grid
        except Exception:
            continue
    raise RuntimeError(f"Failed to generate a valid ARC example after {MAX_RETRIES} attempts")
