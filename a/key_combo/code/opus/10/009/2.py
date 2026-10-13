import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
EDGE_BUFFER = 3
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _component_side(comp: dict, h: int, w: int) -> str:
    coords = comp["coords"]
    sides = []
    if np.any(coords[:, 0] == 0):
        sides.append("top")
    if np.any(coords[:, 0] == h - 1):
        sides.append("bottom")
    if np.any(coords[:, 1] == 0):
        sides.append("left")
    if np.any(coords[:, 1] == w - 1):
        sides.append("right")
    assert len(sides) == 1, f"component must touch exactly one edge, got {sides}"
    return sides[0]


def _trace_pair(color: int, comp_a: dict, comp_b: dict, h: int, w: int) -> dict:
    side_a = _component_side(comp_a, h, w)
    side_b = _component_side(comp_b, h, w)
    sides = {side_a, side_b}
    coverage: set[tuple[int, int]] = set()
    final_cells: set[tuple[int, int]] = set()
    initial_cells = {tuple(map(int, xy)) for xy in comp_a["coords"]} | {tuple(map(int, xy)) for xy in comp_b["coords"]}

    if sides == {"top", "bottom"}:
        upper = comp_a if side_a == "top" else comp_b
        lower = comp_b if side_a == "top" else comp_a
        gap = int(lower["bbox"][0] - upper["bbox"][2] - 1)
        assert gap >= 0 and gap % 2 == 0, f"vertical gap must be even, got {gap}"
        steps = gap // 2
        for r, c in map(tuple, upper["coords"]):
            for t in range(steps + 1):
                coverage.add((int(r + t), int(c)))
            final_cells.add((int(r + steps), int(c)))
        for r, c in map(tuple, lower["coords"]):
            for t in range(steps + 1):
                coverage.add((int(r - t), int(c)))
            final_cells.add((int(r - steps), int(c)))
        orientation = "vertical"
    elif sides == {"left", "right"}:
        left = comp_a if side_a == "left" else comp_b
        right = comp_b if side_a == "left" else comp_a
        gap = int(right["bbox"][1] - left["bbox"][3] - 1)
        assert gap >= 0 and gap % 2 == 0, f"horizontal gap must be even, got {gap}"
        steps = gap // 2
        for r, c in map(tuple, left["coords"]):
            for t in range(steps + 1):
                coverage.add((int(r), int(c + t)))
            final_cells.add((int(r), int(c + steps)))
        for r, c in map(tuple, right["coords"]):
            for t in range(steps + 1):
                coverage.add((int(r), int(c - t)))
            final_cells.add((int(r), int(c - steps)))
        orientation = "horizontal"
    else:
        raise AssertionError(f"pair color {color} is not on opposite edges: {side_a}, {side_b}")

    return {
        "color": color,
        "orientation": orientation,
        "coverage": coverage,
        "final": final_cells,
        "initial": initial_cells,
        "steps": steps,
    }


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _find_components(g)
    assert comps, "expected foreground sprites"
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    assert 2 <= len(by_color) <= 4, f"expected 2-4 color pairs, got {len(by_color)}"
    pairs = []
    h, w = g.shape
    for color, group in sorted(by_color.items()):
        assert len(group) == 2, f"color {color} must appear in exactly two components"
        for comp in group:
            assert 2 <= comp["area"] <= 5, f"sprite area out of range for color {color}: {comp['area']}"
        pairs.append(_trace_pair(int(color), group[0], group[1], h, w))

    orientations = {pair["orientation"] for pair in pairs}
    assert "vertical" in orientations and "horizontal" in orientations, "need at least one pair of each orientation"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide opposite-edge same-color sprites toward each other and gray out cross-pair trail intersections.

    Input structure:
    - A black background contains 2-4 colors, and each color appears in exactly two 4-connected monochrome sprites.
    - Each sprite touches exactly one border; for a given color the two sprites touch opposite borders, either top/bottom or left/right.
    - Sprites are small irregular bodies rather than single pixels, and their paths are otherwise unobstructed.

    Rule:
    - Group components by color to form pairs.
    - For each top/bottom pair, slide both sprites vertically toward each other; for each left/right pair, slide both sprites horizontally toward each other.
    - Each sprite moves one cell per step, leaves its own full swept trail, and stops when the two paired bounding boxes become edge-adjacent.
    - After all pairs are traced, recolor any cell covered by two or more different pairs to gray (5); overlap created only within one pair keeps that pair's color.

    Key insight:
    - Pair motion is local to each color, but the final gray cells are a global effect that only appears after comparing the complete swept regions of all pairs together.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _analyze_grid(g)
    out = np.zeros_like(g)
    painted_by: dict[tuple[int, int], set[int]] = defaultdict(set)

    for pair in pairs:
        color = int(pair["color"])
        for cell in pair["coverage"]:
            out[cell] = color
            painted_by[cell].add(color)

    for cell, colors in painted_by.items():
        if len(colors) >= 2:
            out[cell] = GREY
    return out


BASE_SHAPES = [
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
]


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _rotate_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = 1 + max(r for r, _ in coords)
    return _normalize_shape([(c, h - 1 - r) for r, c in coords])


def _flip_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    w = 1 + max(c for _, c in coords)
    return _normalize_shape([(r, w - 1 - c) for r, c in coords])


def _random_shape() -> dict:
    coords = tuple(BASE_SHAPES[random.randrange(len(BASE_SHAPES))])
    if random.random() < 0.5:
        coords = _flip_shape(coords)
    for _ in range(random.randrange(4)):
        coords = _rotate_shape(coords)
    coords = _normalize_shape(coords)
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    return {
        "coords": coords,
        "h": 1 + max(r for r, _ in coords),
        "w": 1 + max(c for _, c in coords),
        "rows": rows,
        "cols": cols,
    }


def _final_rows_for_vertical(shape: dict, height: int) -> set[int]:
    steps = (height - 2 * shape["h"]) // 2
    top_rows = {steps + r for r in shape["rows"]}
    bottom_base = height - shape["h"] - steps
    bottom_rows = {bottom_base + r for r in shape["rows"]}
    return top_rows | bottom_rows


def _final_cols_for_horizontal(shape: dict, width: int) -> set[int]:
    steps = (width - 2 * shape["w"]) // 2
    left_cols = {steps + c for c in shape["cols"]}
    right_base = width - shape["w"] - steps
    right_cols = {right_base + c for c in shape["cols"]}
    return left_cols | right_cols


def _shifted(values: list[int], start: int) -> set[int]:
    return {start + v for v in values}


def _axis_candidates(
    occupied: list[int],
    span: int,
    total: int,
    blocked: set[int],
    *,
    must_intersect: set[int] | None = None,
    must_avoid: set[int] | None = None,
) -> list[int]:
    starts = []
    lo = EDGE_BUFFER
    hi = total - span - EDGE_BUFFER
    for start in range(lo, hi + 1):
        actual = _shifted(occupied, start)
        if actual & blocked:
            continue
        if must_intersect is not None and not (actual & must_intersect):
            continue
        if must_avoid is not None and (actual & must_avoid):
            continue
        starts.append(start)
    random.shuffle(starts)
    return starts


def _block_with_padding(blocked: set[int], actual: set[int], total: int) -> None:
    for value in actual:
        for delta in (-1, 0, 1):
            nv = value + delta
            if 0 <= nv < total:
                blocked.add(nv)


def _stamp_shape(grid: np.ndarray, shape: dict, top: int, left: int, color: int) -> None:
    for dr, dc in shape["coords"]:
        grid[top + dr, left + dc] = color


def _target_overlap_mode(v_pair: dict, h_pair: dict) -> str:
    overlap = v_pair["coverage"] & h_pair["coverage"]
    assert overlap, "targeted vertical/horizontal pair must overlap"
    if overlap & v_pair["final"]:
        return "vertical_final"
    if overlap & h_pair["final"]:
        return "horizontal_final"
    return "trail_only"


def _validate_generated(grid: np.ndarray, target_v_color: int, target_h_color: int, requested_mode: str) -> bool:
    try:
        pairs = _analyze_grid(grid)
        out = main(grid)
    except Exception:
        return False

    if int(np.count_nonzero(out == GREY)) == 0:
        return False

    pair_by_color = {int(pair["color"]): pair for pair in pairs}
    if target_v_color not in pair_by_color or target_h_color not in pair_by_color:
        return False

    all_initial = {int(pair["color"]): set(pair["initial"]) for pair in pairs}
    for pair in pairs:
        foreign_initial = set()
        for color, initial in all_initial.items():
            if color != int(pair["color"]):
                foreign_initial |= initial
        if pair["coverage"] & foreign_initial:
            return False
        if pair["steps"] <= 0:
            return False

    overlap_counts: dict[tuple[int, int], set[int]] = defaultdict(set)
    for pair in pairs:
        for cell in pair["coverage"]:
            overlap_counts[cell].add(int(pair["color"]))
    if not any(len(colors) >= 2 for colors in overlap_counts.values()):
        return False

    mode = _target_overlap_mode(pair_by_color[target_v_color], pair_by_color[target_h_color])
    if requested_mode == "final" and mode == "trail_only":
        return False
    if requested_mode == "trail" and mode != "trail_only":
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.choice([16, 18, 20, 22, 24])
    width = random.choice([16, 18, 20, 22, 24])
    n_vertical, n_horizontal = random.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
    shapes_v = [_random_shape() for _ in range(n_vertical)]
    shapes_h = [_random_shape() for _ in range(n_horizontal)]
    colors = random.sample(PALETTE, n_vertical + n_horizontal)

    target_mode = random.choice(["trail", "final"])
    target_v_shape = shapes_v[0]
    target_h_shape = shapes_h[0]
    target_final_rows = _final_rows_for_vertical(target_v_shape, height)
    target_final_cols = _final_cols_for_horizontal(target_h_shape, width)

    blocked_cols: set[int] = set()
    v_starts: list[int] = []
    first_v_candidates = _axis_candidates(
        target_v_shape["cols"],
        target_v_shape["w"],
        width,
        blocked_cols,
        must_avoid=target_final_cols if target_mode == "trail" else None,
    )
    if not first_v_candidates:
        return None
    first_v = first_v_candidates[0]
    v_starts.append(first_v)
    _block_with_padding(blocked_cols, _shifted(target_v_shape["cols"], first_v), width)
    for shape in shapes_v[1:]:
        candidates = _axis_candidates(shape["cols"], shape["w"], width, blocked_cols)
        if not candidates:
            return None
        start = candidates[0]
        v_starts.append(start)
        _block_with_padding(blocked_cols, _shifted(shape["cols"], start), width)

    blocked_rows: set[int] = set()
    h_starts: list[int] = []
    first_h_candidates = _axis_candidates(
        target_h_shape["rows"],
        target_h_shape["h"],
        height,
        blocked_rows,
        must_intersect=target_final_rows if target_mode == "final" else None,
        must_avoid=target_final_rows if target_mode == "trail" else None,
    )
    if not first_h_candidates:
        return None
    first_h = first_h_candidates[0]
    h_starts.append(first_h)
    _block_with_padding(blocked_rows, _shifted(target_h_shape["rows"], first_h), height)
    for shape in shapes_h[1:]:
        candidates = _axis_candidates(shape["rows"], shape["h"], height, blocked_rows)
        if not candidates:
            return None
        start = candidates[0]
        h_starts.append(start)
        _block_with_padding(blocked_rows, _shifted(shape["rows"], start), height)

    grid = np.zeros((height, width), dtype=int)
    target_v_color = colors[0]
    target_h_color = colors[n_vertical]

    for color, shape, start in zip(colors[:n_vertical], shapes_v, v_starts):
        _stamp_shape(grid, shape, 0, start, color)
        _stamp_shape(grid, shape, height - shape["h"], start, color)
    for color, shape, start in zip(colors[n_vertical:], shapes_h, h_starts):
        _stamp_shape(grid, shape, start, 0, color)
        _stamp_shape(grid, shape, start, width - shape["w"], color)

    if not _validate_generated(grid, target_v_color, target_h_color, target_mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
