from __future__ import annotations

import random

import numpy as np


UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

DIRECTIONS = {
    "up": UP,
    "down": DOWN,
    "left": LEFT,
    "right": RIGHT,
}

MARKER_OFFSETS = [UP, DOWN, LEFT, RIGHT]
CENTER = (2, 2)
INNER_CORNERS = [(1, 1), (1, 3), (3, 1), (3, 3)]

PERIMETER = {
    (r, c)
    for r in range(5)
    for c in range(5)
    if r in (0, 4) or c in (0, 4)
}

GAP_CELLS = {
    "up": {(0, 1), (0, 2), (0, 3)},
    "down": {(4, 1), (4, 2), (4, 3)},
    "left": {(1, 0), (2, 0), (3, 0)},
    "right": {(1, 4), (2, 4), (3, 4)},
}

RING_CELLS = {name: sorted(PERIMETER - cells) for name, cells in GAP_CELLS.items()}

PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
BG0 = 0
BG1 = 5

GRID_MIN = 18
GRID_MAX = 30
RING_COUNT_MIN = 2
RING_COUNT_MAX = 4
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
MIN_BRANCH_PIXELS = 4


def _rotate_left(vec: tuple[int, int]) -> tuple[int, int]:
    dr, dc = vec
    return -dc, dr


def _rotate_right(vec: tuple[int, int]) -> tuple[int, int]:
    dr, dc = vec
    return dc, -dr


def _checker_expected(shape: tuple[int, int], even_color: int, odd_color: int, phase: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where(((rows + cols + phase) % 2) == 0, even_color, odd_color)


def _best_checkerboard(g: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    colors = [int(v) for v in np.unique(g)]
    best_score = -1
    best_grid: np.ndarray | None = None
    best_meta: tuple[int, int, int] | None = None
    for even_color in colors:
        for odd_color in colors:
            if odd_color == even_color:
                continue
            for phase in (0, 1):
                expected = _checker_expected(g.shape, even_color, odd_color, phase)
                score = int(np.sum(g == expected))
                if score > best_score:
                    best_score = score
                    best_grid = expected
                    best_meta = (even_color, odd_color, phase)
    assert best_grid is not None and best_meta is not None
    return best_grid, best_meta


def _branch_side(growth: tuple[int, int], marker_offset: tuple[int, int]) -> tuple[int, int]:
    dot = growth[0] * marker_offset[0] + growth[1] * marker_offset[1]
    if dot == 0:
        return marker_offset
    if marker_offset == growth:
        return _rotate_left(growth)
    return _rotate_right(growth)


def _match_ring_window(g: np.ndarray, bg: np.ndarray, r0: int, c0: int) -> dict[str, object] | None:
    sub = g[r0 : r0 + 5, c0 : c0 + 5]
    bg_sub = bg[r0 : r0 + 5, c0 : c0 + 5]
    if int(np.count_nonzero(sub != bg_sub)) != 14:
        return None

    for gap_name, growth in DIRECTIONS.items():
        ring_cells = RING_CELLS[gap_name]
        gap_cells = GAP_CELLS[gap_name]

        ring_vals = [int(sub[r, c]) for r, c in ring_cells]
        ring_color = ring_vals[0]
        if any(v != ring_color for v in ring_vals):
            continue
        if any(sub[r, c] != bg_sub[r, c] for r, c in gap_cells):
            continue
        if any(sub[r, c] != bg_sub[r, c] for r, c in INNER_CORNERS):
            continue
        if int(sub[CENTER]) != int(bg_sub[CENTER]):
            continue

        markers = []
        for dr, dc in MARKER_OFFSETS:
            rr = CENTER[0] + dr
            cc = CENTER[1] + dc
            if int(sub[rr, cc]) != int(bg_sub[rr, cc]):
                markers.append((dr, dc))
        if len(markers) != 1:
            continue

        marker_offset = markers[0]
        mr = CENTER[0] + marker_offset[0]
        mc = CENTER[1] + marker_offset[1]
        marker_color = int(sub[mr, mc])
        if marker_color == ring_color:
            continue
        for dr, dc in MARKER_OFFSETS:
            if (dr, dc) == marker_offset:
                continue
            rr = CENTER[0] + dr
            cc = CENTER[1] + dc
            if int(sub[rr, cc]) != int(bg_sub[rr, cc]):
                break
        else:
            center_r = r0 + 2
            center_c = c0 + 2
            gap_center = (center_r + 2 * growth[0], center_c + 2 * growth[1])
            side = _branch_side(growth, marker_offset)
            return {
                "top": int(r0),
                "left": int(c0),
                "gap_name": gap_name,
                "growth": growth,
                "side": side,
                "ring_color": ring_color,
                "marker_color": marker_color,
                "marker_offset": marker_offset,
                "gap_center": gap_center,
                "ring_pixels": [(r0 + r, c0 + c) for r, c in ring_cells],
                "bbox": (r0, c0, r0 + 4, c0 + 4),
            }
    return None


def _find_rings(g: np.ndarray, bg: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    used = np.zeros((h, w), dtype=bool)
    rings: list[dict[str, object]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            if used[r0 : r0 + 5, c0 : c0 + 5].any():
                continue
            ring = _match_ring_window(g, bg, r0, c0)
            if ring is None:
                continue
            used[r0 : r0 + 5, c0 : c0 + 5] = True
            rings.append(ring)
    return rings


def _trace_branch(
    shape: tuple[int, int],
    occupied: np.ndarray,
    start: tuple[int, int],
    growth: tuple[int, int],
    side: tuple[int, int],
) -> tuple[list[tuple[int, int]], str]:
    h, w = shape
    cells = [start]
    r, c = start
    while True:
        for _ in range(2):
            nr = r + growth[0]
            nc = c + growth[1]
            if not (0 <= nr < h and 0 <= nc < w):
                return cells, "boundary"
            if occupied[nr, nc]:
                return cells, "occupied"
            cells.append((nr, nc))
            r, c = nr, nc
        nr = r + side[0]
        nc = c + side[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return cells, "boundary"
        if occupied[nr, nc]:
            return cells, "occupied"
        cells.append((nr, nc))
        r, c = nr, nc


def _occupied_mask(g: np.ndarray, bg: np.ndarray) -> np.ndarray:
    return g != bg


def _paint_solution(g: np.ndarray) -> np.ndarray:
    bg, _meta = _best_checkerboard(g)
    rings = _find_rings(g, bg)
    if not rings:
        return g.copy()

    occupied = _occupied_mask(g, bg)
    out = g.copy()
    for ring in rings:
        cells, _reason = _trace_branch(
            g.shape,
            occupied,
            ring["gap_center"],
            ring["growth"],
            ring["side"],
        )
        for r, c in cells:
            out[r, c] = int(ring["marker_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored staircase branches from 5x5 broken square rings on a checkerboard.

    The gap side gives the growth direction, the inner marker color gives the branch color,
    and the marker position gives the sideways bend. The staircase is operationalized as
    two forward cells followed by one lateral cell, repeated until the next step would leave
    the grid or hit another pre-existing non-background structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _paint_solution(g)


def _checker_canvas(h: int, w: int, phase: int) -> np.ndarray:
    return _checker_expected((h, w), BG0, BG1, phase).astype(int, copy=False)


def _ring_box(spec: dict[str, object]) -> tuple[int, int, int, int]:
    top = int(spec["top"])
    left = int(spec["left"])
    return top, left, top + 4, left + 4


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _cell_in_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _render_input(h: int, w: int, phase: int, specs: list[dict[str, object]]) -> np.ndarray:
    g = _checker_canvas(h, w, phase).copy()
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        gap_name = str(spec["gap_name"])
        ring_color = int(spec["ring_color"])
        for dr, dc in RING_CELLS[gap_name]:
            g[top + dr, left + dc] = ring_color
        mr, mc = spec["marker_offset"]
        g[top + 2 + mr, left + 2 + mc] = int(spec["marker_color"])
    return g


def _make_specs(count: int) -> list[dict[str, object]]:
    colors = PALETTE[:]
    random.shuffle(colors)
    ring_colors = colors[:count]
    marker_colors = colors[count : 2 * count]
    specs: list[dict[str, object]] = []
    for idx in range(count):
        gap_name = random.choice(tuple(DIRECTIONS))
        marker_offset = random.choice(MARKER_OFFSETS)
        specs.append(
            {
                "top": -1,
                "left": -1,
                "gap_name": gap_name,
                "growth": DIRECTIONS[gap_name],
                "side": _branch_side(DIRECTIONS[gap_name], marker_offset),
                "ring_color": ring_colors[idx],
                "marker_color": marker_colors[idx],
                "marker_offset": marker_offset,
            }
        )
    return specs


def _place_random_box(
    h: int,
    w: int,
    specs: list[dict[str, object]],
    idx: int,
    placed_boxes: list[tuple[int, int, int, int]],
) -> bool:
    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(1, h - 6)
        left = random.randint(1, w - 6)
        box = (top, left, top + 4, left + 4)
        if any(_box_overlap(box, other) for other in placed_boxes):
            continue
        specs[idx]["top"] = top
        specs[idx]["left"] = left
        placed_boxes.append(box)
        return True
    return False


def _trace_from_spec(h: int, w: int, g: np.ndarray, spec: dict[str, object]) -> tuple[list[tuple[int, int]], str]:
    bg = _checker_expected((h, w), BG0, BG1, int(spec["phase"]))
    occupied = _occupied_mask(g, bg)
    center_r = int(spec["top"]) + 2
    center_c = int(spec["left"]) + 2
    growth = DIRECTIONS[str(spec["gap_name"])]
    start = (center_r + 2 * growth[0], center_c + 2 * growth[1])
    side = _branch_side(growth, spec["marker_offset"])
    return _trace_branch((h, w), occupied, start, growth, side)


def _free_path_to_boundary(h: int, w: int, spec: dict[str, object]) -> list[tuple[int, int]]:
    phase = int(spec["phase"])
    g = _render_input(h, w, phase, [spec])
    cells, _reason = _trace_from_spec(h, w, g, spec)
    return cells


def _align_blocker(
    h: int,
    w: int,
    emitter: dict[str, object],
    blocker: dict[str, object],
) -> bool:
    emitter_path = _free_path_to_boundary(h, w, emitter)
    if len(emitter_path) < 8:
        return False
    emitter_box = _ring_box(emitter)
    ring_offsets = RING_CELLS[str(blocker["gap_name"])][:]
    random.shuffle(ring_offsets)
    target_cells = emitter_path[4:-1]
    random.shuffle(target_cells)
    for target in target_cells:
        prefix = emitter_path[: emitter_path.index(target)]
        for dr, dc in ring_offsets:
            top = target[0] - dr
            left = target[1] - dc
            box = (top, left, top + 4, left + 4)
            if not (1 <= top <= h - 6 and 1 <= left <= w - 6):
                continue
            if _box_overlap(box, emitter_box):
                continue
            if any(_cell_in_box(cell, box) for cell in prefix):
                continue
            blocker["top"] = top
            blocker["left"] = left
            return True
    return False


def _validate_sample(g: np.ndarray, intended_count: int, mode: str) -> bool:
    try:
        bg, _meta = _best_checkerboard(g)
        rings = _find_rings(g, bg)
    except Exception:
        return False
    if len(rings) != intended_count:
        return False

    occupied = _occupied_mask(g, bg)
    ring_boxes = [tuple(ring["bbox"]) for ring in rings]
    all_branch_cells: set[tuple[int, int]] = set()
    reasons = []
    for idx, ring in enumerate(rings):
        cells, reason = _trace_branch(g.shape, occupied, ring["gap_center"], ring["growth"], ring["side"])
        if len(cells) < MIN_BRANCH_PIXELS:
            return False
        for other_idx, box in enumerate(ring_boxes):
            if other_idx == idx:
                continue
            if any(_cell_in_box(cell, box) for cell in cells[1:]):
                return False
        cell_set = set(cells)
        if all_branch_cells & cell_set:
            return False
        all_branch_cells |= cell_set
        reasons.append(reason)

    if mode == "boundary" and any(reason != "boundary" for reason in reasons):
        return False
    if mode == "blocker" and all(reason != "occupied" for reason in reasons):
        return False

    try:
        out = main(g)
    except Exception:
        return False
    if not isinstance(out, np.ndarray) or out.shape != g.shape:
        return False
    if not np.any(out != g):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    phase = random.randint(0, 1)
    mode = random.choice(["boundary", "blocker"])
    count = random.randint(RING_COUNT_MIN, RING_COUNT_MAX)
    specs = _make_specs(count)
    for spec in specs:
        spec["phase"] = phase

    placed_boxes: list[tuple[int, int, int, int]] = []

    if mode == "blocker":
        if count < 2:
            return None
        if not _place_random_box(h, w, specs, 0, placed_boxes):
            return None
        if not _align_blocker(h, w, specs[0], specs[1]):
            return None
        placed_boxes.append(_ring_box(specs[1]))
        for idx in range(2, count):
            if not _place_random_box(h, w, specs, idx, placed_boxes):
                return None
    else:
        for idx in range(count):
            if not _place_random_box(h, w, specs, idx, placed_boxes):
                return None

    g = _render_input(h, w, phase, specs)
    if not _validate_sample(g, count, mode):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _generate_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate a valid staircase-ring input")
