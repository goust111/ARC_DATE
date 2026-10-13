from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 5
OBJECT_COLOR_POOL = (1, 2, 3, 7)
QUADRANT_COLOR_POOL = (4, 6, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
QUADRANTS = ("ul", "ur", "ll", "lr")

GRID_HEIGHT_RANGE = (12, 19)
GRID_WIDTH_RANGE = (12, 19)
ANCHOR_OFFSET_RANGE = (2, 4)
SINGLETON_OFFSET_RANGE = (1, 4)
HALF_HEIGHT_CHOICES = (2, 3)
OBJECT_WIDTH_RANGE = (4, 8)
MAX_ROW_RUN = 4
GENERATION_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components_from_mask(grid == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _quadrant(row: float, col: float, center_row: float, center_col: float) -> str:
    assert row != center_row and col != center_col, "centroid or marker lies on anchor axis"
    if row < center_row:
        return "ul" if col < center_col else "ur"
    return "ll" if col < center_col else "lr"


def _centroid(coords: list[tuple[int, int]]) -> tuple[float, float]:
    rows = np.array([r for r, _ in coords], dtype=float)
    cols = np.array([c for _, c in coords], dtype=float)
    return float(rows.mean()), float(cols.mean())


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = get_bg(g)

    mono = _monochrome_components(g, bg)

    anchors = []
    for comp in mono:
        if comp["color"] != ANCHOR_COLOR or comp["area"] != 4:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == ANCHOR_COLOR):
            anchors.append(comp)
    assert len(anchors) == 1, "expected exactly one solid 2x2 gray anchor block"
    anchor = anchors[0]
    ar0, ac0, ar1, ac1 = anchor["bbox"]
    anchor_center = (ar0 + 0.5, ac0 + 0.5)

    quadrant_to_color: dict[str, int] = {}
    singletons = [
        comp
        for comp in mono
        if comp["area"] == 1 and comp["color"] in QUADRANT_COLOR_POOL
    ]
    assert len(singletons) == 4, "expected four scattered quadrant singletons"
    for comp in singletons:
        r, c = comp["coords"][0]
        q = _quadrant(float(r), float(c), anchor_center[0], anchor_center[1])
        assert q not in quadrant_to_color, "expected one singleton per anchor quadrant"
        quadrant_to_color[q] = int(comp["color"])
    assert set(quadrant_to_color) == set(QUADRANTS)

    object_mask = np.isin(g, OBJECT_COLOR_POOL)
    object_components = _components_from_mask(object_mask)
    assert len(object_components) == 1, "expected exactly one bi-color foreground object"
    object_coords = sorted(object_components[0])
    area = len(object_coords)
    assert 8 <= area <= 14, "object area must stay within the intended range"
    or0, oc0, or1, oc1 = _bbox(object_coords)
    obj_height = or1 - or0 + 1
    assert obj_height % 2 == 0, "object bbox height must be even"
    split_row = or0 + obj_height // 2

    upper = [(r, c) for r, c in object_coords if r < split_row]
    lower = [(r, c) for r, c in object_coords if r >= split_row]
    assert upper and lower, "split must produce two non-empty halves"

    upper_colors = {int(g[r, c]) for r, c in upper}
    lower_colors = {int(g[r, c]) for r, c in lower}
    assert len(upper_colors) == 2 and len(lower_colors) == 2, "each half must visibly contain both object colors"

    upper_centroid = _centroid(upper)
    lower_centroid = _centroid(lower)
    upper_quad = _quadrant(*upper_centroid, *anchor_center)
    lower_quad = _quadrant(*lower_centroid, *anchor_center)
    assert upper_quad != lower_quad, "upper and lower halves must land in different anchor quadrants"

    upper_left_col = min(c for _, c in upper)
    upper_diag_row = max(r for r, c in upper if c == upper_left_col)
    lower_right_col = max(c for _, c in lower)
    lower_diag_row = min(r for r, c in lower if c == lower_right_col)
    assert upper_diag_row + 1 == lower_diag_row, "diagonal join must be achievable by horizontal shifting only"

    shift = lower_right_col + 1 - upper_left_col
    assert shift > 0, "upper half must shift to the right"

    shifted_upper = [(r, c + shift) for r, c in upper]
    assert max(c for _, c in shifted_upper) < w, "shifted upper half must fit in the grid"
    non_object_mask = (g != bg) & ~object_mask
    assert not any(non_object_mask[r, c] for r, c in shifted_upper), "shifted upper half may not cover anchor or singleton clues"

    bbox_width = oc1 - oc0 + 1
    assert shift != bbox_width, "generator should avoid the trivial bbox-width shortcut"

    return {
        "bg": bg,
        "anchor_center": anchor_center,
        "quadrant_to_color": quadrant_to_color,
        "object_coords": object_coords,
        "object_mask": object_mask,
        "upper": upper,
        "lower": lower,
        "upper_quad": upper_quad,
        "lower_quad": lower_quad,
        "upper_color": quadrant_to_color[upper_quad],
        "lower_color": quadrant_to_color[lower_quad],
        "shift": shift,
        "shifted_upper": shifted_upper,
        "bbox_width": bbox_width,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the two-color object horizontally, recolor each half by anchor-quadrant lookup, and rejoin it diagonally.

    Input structure:
    - The grid has black background, one 4-connected two-color object with an even-height bounding box, one solid gray 2x2 anchor block, and four colored singleton clues.
    - Each clue singleton lies in a different diagonal quadrant around the anchor block, and the object colors are drawn from a disjoint palette from the clue colors.

    Rule:
    - Split the two-color object at the horizontal midpoint of its bounding box into upper and lower halves.
    - Compute each half's centroid; relative to the anchor-block center, use that centroid's quadrant to pick the matching singleton color.
    - Recolor the entire upper half to its looked-up quadrant color, recolor the lower half likewise, keep the lower half in place, and shift the upper half right until its lower-left boundary pixel is diagonally adjacent to the lower half's upper-right boundary pixel.

    Key insight:
    - The split is essential because the two resulting half-centroids usually fall in different anchor quadrants, so each half receives a different clue color before the upper half is moved into the final diagonal join.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)

    out = g.copy()
    out[info["object_mask"]] = info["bg"]
    for r, c in info["lower"]:
        out[r, c] = info["lower_color"]
    for r, c in info["shifted_upper"]:
        out[r, c] = info["upper_color"]
    return out


def _all_intervals(width: int, min_len: int = 1, max_len: int | None = None) -> list[tuple[int, int]]:
    if max_len is None:
        max_len = width
    intervals = []
    for left in range(width):
        for right in range(left, width):
            if min_len <= right - left + 1 <= max_len:
                intervals.append((left, right))
    return intervals


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _generate_upper_intervals(half_h: int, width: int, bridge_col: int) -> list[tuple[int, int]] | None:
    intervals = _all_intervals(width, min_len=1, max_len=min(MAX_ROW_RUN, width))
    base_choices = [iv for iv in intervals if bridge_col >= iv[0] and bridge_col <= iv[1] and (iv[1] - iv[0] + 1) >= 2]
    random.shuffle(base_choices)
    for bottom in base_choices:
        rows: list[tuple[int, int]] = [bottom]
        min_left = bottom[0]
        ok = True
        for _ in range(half_h - 1):
            candidates = [
                iv
                for iv in intervals
                if iv[0] >= min_left and _overlap(iv, rows[-1])
            ]
            if not candidates:
                ok = False
                break
            rows.append(random.choice(candidates))
        if ok:
            return list(reversed(rows))
    return None


def _generate_lower_intervals(half_h: int, width: int, bridge_col: int) -> list[tuple[int, int]] | None:
    intervals = _all_intervals(width, min_len=1, max_len=min(MAX_ROW_RUN, width))
    base_choices = [iv for iv in intervals if bridge_col >= iv[0] and bridge_col <= iv[1] and (iv[1] - iv[0] + 1) >= 2]
    random.shuffle(base_choices)
    for top in base_choices:
        rows: list[tuple[int, int]] = [top]
        max_right = top[1]
        ok = True
        for _ in range(half_h - 1):
            candidates = [
                iv
                for iv in intervals
                if iv[1] <= max_right and _overlap(iv, rows[-1])
            ]
            if not candidates:
                ok = False
                break
            rows.append(random.choice(candidates))
        if ok:
            return rows
    return None


def _coords_from_intervals(intervals: list[tuple[int, int]], row_offset: int = 0) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, (left, right) in enumerate(intervals):
        for c in range(left, right + 1):
            coords.append((row_offset + r, c))
    return coords


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _checkerboard_coloring(
    coords: list[tuple[int, int]],
    color_a: int,
    color_b: int,
    phase: int,
) -> dict[tuple[int, int], int]:
    colored = {}
    for r, c in coords:
        colored[(r, c)] = color_a if (r + c + phase) % 2 == 0 else color_b
    return colored


def _generate_object_shape() -> dict | None:
    half_h = random.choice(HALF_HEIGHT_CHOICES)
    width = random.randint(*OBJECT_WIDTH_RANGE)
    bridge_col = random.randint(1, width - 2)

    upper_intervals = _generate_upper_intervals(half_h, width, bridge_col)
    lower_intervals = _generate_lower_intervals(half_h, width, bridge_col)
    if upper_intervals is None or lower_intervals is None:
        return None

    upper = _coords_from_intervals(upper_intervals, row_offset=0)
    lower = _coords_from_intervals(lower_intervals, row_offset=half_h)
    full = upper + lower
    if len(_components_from_mask(_coords_to_mask(full))) != 1:
        return None

    area = len(full)
    if not (8 <= area <= 14):
        return None

    normalized = _normalize_coords(full)
    upper_norm = [coord for coord in normalized if coord[0] < half_h]
    lower_norm = [coord for coord in normalized if coord[0] >= half_h]
    if not upper_norm or not lower_norm:
        return None

    _, _, max_r, max_c = _bbox(normalized)
    bbox_width = max_c + 1

    upper_left = min(c for _, c in upper_norm)
    lower_right = max(c for _, c in lower_norm)
    if upper_left == 0 and lower_right == bbox_width - 1:
        return None

    upper_diag_row = max(r for r, c in upper_norm if c == upper_left)
    lower_diag_row = min(r for r, c in lower_norm if c == lower_right)
    if upper_diag_row + 1 != lower_diag_row:
        return None

    shift = lower_right + 1 - upper_left
    if shift <= 0 or shift == bbox_width:
        return None

    color_a, color_b = random.sample(list(OBJECT_COLOR_POOL), 2)
    phase = random.randint(0, 1)
    colors = _checkerboard_coloring(normalized, color_a, color_b, phase)
    upper_colors = {colors[coord] for coord in upper_norm}
    lower_colors = {colors[coord] for coord in lower_norm}
    if len(upper_colors) != 2 or len(lower_colors) != 2:
        return None

    shifted_upper = [(r, c + shift) for r, c in upper_norm]
    return {
        "coords": normalized,
        "upper": upper_norm,
        "lower": lower_norm,
        "shifted_upper": shifted_upper,
        "colors": colors,
        "bbox_height": max_r + 1,
        "bbox_width": bbox_width,
        "shift": shift,
    }


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    if not coords:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _anchor_position(h: int, w: int) -> tuple[int, int]:
    corner = random.choice(("tl", "tr", "bl", "br"))
    roff = random.randint(*ANCHOR_OFFSET_RANGE)
    coff = random.randint(*ANCHOR_OFFSET_RANGE)
    if corner == "tl":
        return roff, coff
    if corner == "tr":
        return roff, w - 2 - coff
    if corner == "bl":
        return h - 2 - roff, coff
    return h - 2 - roff, w - 2 - coff


def _singleton_layout(anchor_r: int, anchor_c: int) -> dict[str, tuple[int, int]]:
    r_off = {q: random.randint(*SINGLETON_OFFSET_RANGE) for q in QUADRANTS}
    c_off = {q: random.randint(*SINGLETON_OFFSET_RANGE) for q in QUADRANTS}
    return {
        "ul": (anchor_r - r_off["ul"], anchor_c - c_off["ul"]),
        "ur": (anchor_r - r_off["ur"], anchor_c + 1 + c_off["ur"]),
        "ll": (anchor_r + 1 + r_off["ll"], anchor_c - c_off["ll"]),
        "lr": (anchor_r + 1 + r_off["lr"], anchor_c + 1 + c_off["lr"]),
    }


def _valid_singletons(h: int, w: int, positions: dict[str, tuple[int, int]]) -> bool:
    coords = list(positions.values())
    if len(set(coords)) != 4:
        return False
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _padded_mask(h: int, w: int, coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True
    return mask


def _absolute_coords(coords: list[tuple[int, int]], row0: int, col0: int) -> list[tuple[int, int]]:
    return [(row0 + r, col0 + c) for r, c in coords]


def _feasible_positions(
    h: int,
    w: int,
    anchor_center: tuple[float, float],
    forbidden: np.ndarray,
    shape: dict,
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for row0 in range(h - shape["bbox_height"] + 1):
        for col0 in range(w - shape["bbox_width"] - shape["shift"] + 1):
            abs_object = _absolute_coords(shape["coords"], row0, col0)
            abs_shifted_upper = _absolute_coords(shape["shifted_upper"], row0, col0)
            if any(forbidden[r, c] for r, c in abs_object):
                continue
            if any(forbidden[r, c] for r, c in abs_shifted_upper):
                continue

            abs_upper = _absolute_coords(shape["upper"], row0, col0)
            abs_lower = _absolute_coords(shape["lower"], row0, col0)
            upper_centroid = _centroid(abs_upper)
            lower_centroid = _centroid(abs_lower)
            try:
                upper_quad = _quadrant(*upper_centroid, *anchor_center)
                lower_quad = _quadrant(*lower_centroid, *anchor_center)
            except AssertionError:
                continue
            if upper_quad == lower_quad:
                continue
            groups.setdefault((upper_quad, lower_quad), []).append((row0, col0))
    return groups


def _render_case(
    h: int,
    w: int,
    anchor_r: int,
    anchor_c: int,
    singleton_colors: dict[str, int],
    singleton_positions: dict[str, tuple[int, int]],
    shape: dict,
    row0: int,
    col0: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((h, w), dtype=int)
    inp[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = ANCHOR_COLOR
    for quadrant, color in singleton_colors.items():
        r, c = singleton_positions[quadrant]
        inp[r, c] = color

    abs_object = _absolute_coords(shape["coords"], row0, col0)
    abs_upper = _absolute_coords(shape["upper"], row0, col0)
    abs_lower = _absolute_coords(shape["lower"], row0, col0)
    abs_shifted_upper = _absolute_coords(shape["shifted_upper"], row0, col0)

    for local_coord, color in shape["colors"].items():
        inp[row0 + local_coord[0], col0 + local_coord[1]] = color

    anchor_center = (anchor_r + 0.5, anchor_c + 0.5)
    upper_quad = _quadrant(*_centroid(abs_upper), *anchor_center)
    lower_quad = _quadrant(*_centroid(abs_lower), *anchor_center)
    upper_color = singleton_colors[upper_quad]
    lower_color = singleton_colors[lower_quad]

    out = inp.copy()
    for r, c in abs_object:
        out[r, c] = BACKGROUND
    for r, c in abs_lower:
        out[r, c] = lower_color
    for r, c in abs_shifted_upper:
        out[r, c] = upper_color
    return inp, out


def _semantic_shortcut_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    info = _analyze(inp)

    alt_same_color = inp.copy()
    alt_same_color[info["object_mask"]] = info["bg"]
    whole_centroid = _centroid(info["object_coords"])
    if whole_centroid[0] != info["anchor_center"][0] and whole_centroid[1] != info["anchor_center"][1]:
        whole_quad = _quadrant(*whole_centroid, *info["anchor_center"])
        whole_color = info["quadrant_to_color"][whole_quad]
        for r, c in info["lower"]:
            alt_same_color[r, c] = whole_color
        for r, c in info["shifted_upper"]:
            alt_same_color[r, c] = whole_color
        if np.array_equal(alt_same_color, expected):
            return False

    bbox_shift = info["bbox_width"]
    max_shifted_col = max(c + bbox_shift for r, c in info["upper"])
    if max_shifted_col < inp.shape[1]:
        alt_bbox_shift = inp.copy()
        alt_bbox_shift[info["object_mask"]] = info["bg"]
        for r, c in info["lower"]:
            alt_bbox_shift[r, c] = info["lower_color"]
        for r, c in info["upper"]:
            alt_bbox_shift[r, c + bbox_shift] = info["upper_color"]
        if np.array_equal(alt_bbox_shift, expected):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)

    anchor_r, anchor_c = _anchor_position(h, w)
    singleton_positions = _singleton_layout(anchor_r, anchor_c)
    if not _valid_singletons(h, w, singleton_positions):
        return None

    shape = _generate_object_shape()
    if shape is None:
        return None

    singleton_palette = random.sample(list(QUADRANT_COLOR_POOL), 4)
    singleton_colors = {quadrant: color for quadrant, color in zip(QUADRANTS, singleton_palette)}

    reserved = {
        (r, c)
        for r in range(anchor_r, anchor_r + 2)
        for c in range(anchor_c, anchor_c + 2)
    }
    reserved.update(singleton_positions.values())
    forbidden = _padded_mask(h, w, reserved)
    groups = _feasible_positions(h, w, (anchor_r + 0.5, anchor_c + 0.5), forbidden, shape)
    if not groups:
        return None

    key = random.choice(sorted(groups))
    row0, col0 = random.choice(groups[key])
    inp, expected = _render_case(
        h,
        w,
        anchor_r,
        anchor_c,
        singleton_colors,
        singleton_positions,
        shape,
        row0,
        col0,
    )

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_shortcut_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
