from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, bbox_coords, components_mask


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
GREY = Color.GREY
MAGENTA = Color.PINK
ORANGE = Color.ORANGE

PAIR_COLORS = (BLUE, RED, GREEN, YELLOW)
TRAIL_VALUES = {BLUE, RED, GREEN, YELLOW, MAGENTA}
OUTER_SHAPES = ((4, 6), (6, 4))
SIDE_PRIORITY = ("top", "right", "bottom", "left")
GEN_RETRIES = 400


def _rect_perimeter(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _detect_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for coords in components_mask(grid == GREY, connectivity=4):
        if coords.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(coords)
        if not np.all(grid[ir0 : ir1 + 1, ic0 : ic1 + 1] == GREY):
            continue
        r0, c0, r1, c1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        if min(r0, c0) < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
            continue
        border = _rect_perimeter(r0, c0, r1, c1)
        if any(int(grid[r, c]) != BG for r, c in border):
            continue
        containers.append(
            {
                "outer": (int(r0), int(c0), int(r1), int(c1)),
                "inner": (int(ir0), int(ic0), int(ir1), int(ic1)),
                "area": int(coords.shape[0]),
            }
        )
    containers.sort(key=lambda item: item["outer"])
    return containers


def _detect_pairs(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    pairs: list[dict] = []
    for color in PAIR_COLORS:
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        assert coords.shape[0] == 2, f"color {color} must appear exactly twice"
        pts = [tuple(map(int, pt)) for pt in coords]
        (r0, c0), (r1, c1) = pts
        if r0 == 0 and r1 == h - 1 and c0 == c1:
            pairs.append(
                {
                    "color": int(color),
                    "axis": "vertical",
                    "start_a": (0, c0),
                    "start_b": (h - 1, c0),
                    "dir_a": (1, 0),
                    "dir_b": (-1, 0),
                }
            )
        elif c0 == 0 and c1 == w - 1 and r0 == r1:
            pairs.append(
                {
                    "color": int(color),
                    "axis": "horizontal",
                    "start_a": (r0, 0),
                    "start_b": (r0, w - 1),
                    "dir_a": (0, 1),
                    "dir_b": (0, -1),
                }
            )
        else:
            raise AssertionError(f"color {color} is not a valid opposite-side pair")
    pairs.sort(key=lambda item: item["color"])
    return pairs


def _build_border_mask(shape: tuple[int, int], containers: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for cont in containers:
        r0, c0, r1, c1 = cont["outer"]
        for r, c in _rect_perimeter(r0, c0, r1, c1):
            mask[r, c] = True
    return mask


def _trail_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, list(TRAIL_VALUES))


def _mark_trail(out: np.ndarray, trail_mask: np.ndarray, cell: tuple[int, int], color: int) -> None:
    r, c = cell
    out[r, c] = int(color)
    trail_mask[r, c] = True


def _trace_pairs(
    grid: np.ndarray,
    containers: list[dict],
    *,
    stop_on_borders: bool,
    stop_on_trails: bool,
) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    border_mask = _build_border_mask(out.shape, containers)
    trail_mask = np.zeros(out.shape, dtype=bool)
    pairs = _detect_pairs(out)
    pair_meta: list[dict] = []

    def is_blocked(next_cell: tuple[int, int]) -> tuple[bool, str | None]:
        r, c = next_cell
        if stop_on_borders and border_mask[r, c]:
            return True, "border"
        if stop_on_trails and trail_mask[r, c]:
            return True, "trail"
        return False, None

    for pair in pairs:
        color = int(pair["color"])
        pos_a = pair["start_a"]
        pos_b = pair["start_b"]
        dir_a = pair["dir_a"]
        dir_b = pair["dir_b"]
        active_a = True
        active_b = True
        blocked_by_border = False
        blocked_by_trail = False
        met = False
        met_mode: str | None = None

        _mark_trail(out, trail_mask, pos_a, color)
        _mark_trail(out, trail_mask, pos_b, color)

        while active_a or active_b:
            next_a = (pos_a[0] + dir_a[0], pos_a[1] + dir_a[1]) if active_a else pos_a
            next_b = (pos_b[0] + dir_b[0], pos_b[1] + dir_b[1]) if active_b else pos_b

            if active_a and active_b:
                if next_a == next_b and not is_blocked(next_a)[0]:
                    _mark_trail(out, trail_mask, next_a, MAGENTA)
                    met = True
                    met_mode = "same_cell"
                    break
                if next_a == pos_b and next_b == pos_a:
                    _mark_trail(out, trail_mask, pos_a, MAGENTA)
                    met = True
                    met_mode = "cross_swap"
                    break

            if active_a and (not active_b) and next_a == pos_b:
                _mark_trail(out, trail_mask, pos_b, MAGENTA)
                met = True
                met_mode = "into_stopped_b"
                break
            if active_b and (not active_a) and next_b == pos_a:
                _mark_trail(out, trail_mask, pos_a, MAGENTA)
                met = True
                met_mode = "into_stopped_a"
                break

            blocked_a, reason_a = (False, None)
            blocked_b, reason_b = (False, None)
            if active_a:
                blocked_a, reason_a = is_blocked(next_a)
            if active_b:
                blocked_b, reason_b = is_blocked(next_b)

            if active_a and not blocked_a:
                pos_a = next_a
                _mark_trail(out, trail_mask, pos_a, color)
            elif active_a:
                active_a = False
                blocked_by_border |= reason_a == "border"
                blocked_by_trail |= reason_a == "trail"

            if active_b and not blocked_b:
                pos_b = next_b
                _mark_trail(out, trail_mask, pos_b, color)
            elif active_b:
                active_b = False
                blocked_by_border |= reason_b == "border"
                blocked_by_trail |= reason_b == "trail"

            if active_a and active_b and pos_a == pos_b:
                _mark_trail(out, trail_mask, pos_a, MAGENTA)
                met = True
                met_mode = "post_step_same"
                break

        pair_meta.append(
            {
                "color": color,
                "blocked_by_border": blocked_by_border,
                "blocked_by_trail": blocked_by_trail,
                "met": met,
                "met_mode": met_mode,
            }
        )

    return out, pair_meta


def _side_counts(track_grid: np.ndarray, cont: dict) -> dict[str, int]:
    h, w = track_grid.shape
    r0, c0, r1, c1 = cont["outer"]
    counts = {side: 0 for side in SIDE_PRIORITY}
    if r0 - 1 >= 0:
        counts["top"] = int(sum(int(track_grid[r0 - 1, c]) in TRAIL_VALUES for c in range(c0, c1 + 1)))
    if c1 + 1 < w:
        counts["right"] = int(sum(int(track_grid[r, c1 + 1]) in TRAIL_VALUES for r in range(r0, r1 + 1)))
    if r1 + 1 < h:
        counts["bottom"] = int(sum(int(track_grid[r1 + 1, c]) in TRAIL_VALUES for c in range(c0, c1 + 1)))
    if c0 - 1 >= 0:
        counts["left"] = int(sum(int(track_grid[r, c0 - 1]) in TRAIL_VALUES for r in range(r0, r1 + 1)))
    return counts


def _fill_order(inner: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = inner
    order: list[tuple[int, int]] = []
    if side == "top":
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 + 1):
                order.append((r, c))
    elif side == "bottom":
        for r in range(ir1, ir0 - 1, -1):
            for c in range(ic0, ic1 + 1):
                order.append((r, c))
    elif side == "left":
        for c in range(ic0, ic1 + 1):
            for r in range(ir0, ir1 + 1):
                order.append((r, c))
    elif side == "right":
        for c in range(ic1, ic0 - 1, -1):
            for r in range(ir0, ir1 + 1):
                order.append((r, c))
    else:
        raise ValueError(f"unknown side: {side}")
    return order


def _apply_charge(track_grid: np.ndarray, containers: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.array(track_grid, dtype=int, copy=True)
    charge_meta: list[dict] = []
    for cont in containers:
        counts = _side_counts(track_grid, cont)
        total = int(sum(counts.values()))
        charge = int(min(total, cont["area"]))
        best_side = max(SIDE_PRIORITY, key=lambda side: (counts[side], -SIDE_PRIORITY.index(side)))
        order = _fill_order(cont["inner"], best_side)
        for r, c in order[:charge]:
            out[r, c] = ORANGE
        charge_meta.append(
            {
                "outer": cont["outer"],
                "counts": counts,
                "total": total,
                "charge": charge,
                "side": best_side,
            }
        )
    return out, charge_meta


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    containers = _detect_containers(g)
    assert containers, "expected at least one container"
    track_grid, pair_meta = _trace_pairs(g, containers, stop_on_borders=True, stop_on_trails=True)
    out, charge_meta = _apply_charge(track_grid, containers)
    return out, {"containers": containers, "pairs": pair_meta, "track_grid": track_grid, "charge": charge_meta}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide opposite-side color pairs inward, let bordered containers stop the trails, then charge each container from the most-adjacent side.

    Input structure:
    - Black background contains several closed black rectangular frames whose interiors are solid grey.
    - Colors 1/2/3/4 appear only as isolated single pixels on opposite outer edges, with each color appearing exactly twice on one shared row or one shared column.
    - The containers lie fully inside the grid and act as the only black obstacles distinct from the background.

    Rule:
    - Match each color to its opposite-edge partner, then process the pairs in color order 1 -> 2 -> 3 -> 4.
    - For one pair, both endpoints slide one cell per step toward each other along their shared row or column, painting their color as a trail; if the next cell is a container border or any already-drawn trail cell, that mover stops.
    - When the two movers reach the same meeting cell, or one moves onto the other stopped mover, mark that meeting cell magenta.
    - After all trails are drawn, count trail cells 4-neighbor-adjacent to each container side; the total gives the charge amount capped by the container's grey area, and the side with the largest adjacent count chooses the inward fill direction with tie priority top -> right -> bottom -> left.
    - Replace that many grey interior cells by orange in the chosen side-first scan order, while keeping all borders and trails unchanged.

    Key insight:
    - The same container has two causal roles at once: its black frame truncates the sliding trails, and the resulting trail distribution around that frame determines both how much energy the container receives and from which side it fills.
    """
    out, _meta = _solve(grid)
    return out


def _draw_container(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 + 1 : r1, c0 + 1 : c1] = GREY


def _sample_box(
    h: int,
    w: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    existing: list[tuple[int, int, int, int]],
    *,
    padding: int = 2,
) -> tuple[int, int, int, int] | None:
    if row_lo > row_hi or col_lo > col_hi:
        return None
    tries = 80
    for _ in range(tries):
        oh, ow = random.choice(OUTER_SHAPES)
        if row_lo + oh - 1 > row_hi or col_lo + ow - 1 > col_hi:
            continue
        r0 = random.randint(row_lo, row_hi - oh + 1)
        c0 = random.randint(col_lo, col_hi - ow + 1)
        box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
        if any(_boxes_overlap(box, other, padding=padding) for other in existing):
            continue
        return box
    return None


def _build_input(spec: dict) -> np.ndarray:
    h, w = spec["shape"]
    grid = np.zeros((h, w), dtype=int)
    for cont in spec["containers"]:
        _draw_container(grid, cont["outer"])
    grid[0, spec["v1_col"]] = BLUE
    grid[h - 1, spec["v1_col"]] = BLUE
    grid[spec["h2_row"], 0] = RED
    grid[spec["h2_row"], w - 1] = RED
    grid[0, spec["v3_col"]] = GREEN
    grid[h - 1, spec["v3_col"]] = GREEN
    grid[spec["h4_row"], 0] = YELLOW
    grid[spec["h4_row"], w - 1] = YELLOW
    return grid


def _attempt_layout() -> np.ndarray | None:
    h = random.choice((21, 23, 25, 27, 29))
    w = random.choice((21, 23, 25, 27, 29))
    top_split = random.choice((True, False))

    a_shape = random.choice(OUTER_SHAPES)
    a_row_lo = 8 if top_split else 5
    a_row_hi = h - a_shape[0] - 8 if not top_split else h - a_shape[0] - 5
    a_col_lo = 7
    a_col_hi = w - a_shape[1] - 8
    if a_row_lo > a_row_hi or a_col_lo > a_col_hi:
        return None
    a_r0 = random.randint(a_row_lo, a_row_hi)
    a_c0 = random.randint(a_col_lo, a_col_hi)
    a_outer = (a_r0, a_c0, a_r0 + a_shape[0] - 1, a_c0 + a_shape[1] - 1)
    a_inner = (a_r0 + 1, a_c0 + 1, a_outer[2] - 1, a_outer[3] - 1)
    v1_col = random.randint(a_outer[1], a_outer[3])
    h2_row = a_outer[0] - 1 if top_split else a_outer[2] + 1

    containers = [{"outer": a_outer, "inner": a_inner, "area": 8, "tag": "A"}]
    boxes = [a_outer]

    side_b = random.choice(("left", "right"))
    b_row_lo = 2 if top_split else h2_row + 2
    b_row_hi = h2_row - 3 if top_split else h - 3
    if top_split:
        b_box = _sample_box(h, w, b_row_lo, b_row_hi, 2, w - 3, boxes)
    else:
        b_box = _sample_box(h, w, b_row_lo, b_row_hi, 2, w - 3, boxes)
    if b_box is None:
        return None
    if side_b == "left":
        if b_box[1] < 2:
            return None
        v3_col = b_box[1] - 1
    else:
        if b_box[3] > w - 3:
            return None
        v3_col = b_box[3] + 1
    if not (1 <= v3_col <= w - 2) or abs(v3_col - v1_col) < 2:
        return None
    containers.append({"outer": b_box, "inner": (b_box[0] + 1, b_box[1] + 1, b_box[2] - 1, b_box[3] - 1), "area": 8, "tag": "B"})
    boxes.append(b_box)

    left_limit = min(v1_col, v3_col) - 2
    right_start = max(v1_col, v3_col) + 2
    c_side_options = []
    if left_limit >= 6:
        c_side_options.append("left_region")
    if right_start <= w - 7:
        c_side_options.append("right_region")
    if not c_side_options:
        return None
    c_region = random.choice(c_side_options)
    c_top_bottom = random.choice(("top", "bottom"))
    if c_region == "left_region":
        c_box = _sample_box(h, w, 2, h - 3, 2, left_limit, boxes)
    else:
        c_box = _sample_box(h, w, 2, h - 3, right_start, w - 3, boxes)
    if c_box is None:
        return None
    h4_row = c_box[0] - 1 if c_top_bottom == "top" else c_box[2] + 1
    if not (1 <= h4_row <= h - 2) or abs(h4_row - h2_row) < 2:
        return None
    containers.append({"outer": c_box, "inner": (c_box[0] + 1, c_box[1] + 1, c_box[2] - 1, c_box[3] - 1), "area": 8, "tag": "C"})
    boxes.append(c_box)

    want_four = random.random() < 0.55
    if want_four:
        d_mode = random.choice(("h2_adjacent", "v3_through"))
        d_box: tuple[int, int, int, int] | None = None
        if d_mode == "h2_adjacent":
            d_shape = random.choice(OUTER_SHAPES)
            if top_split:
                d_r0 = h2_row - d_shape[0]
            else:
                d_r0 = h2_row + 1
            if 2 <= d_r0 and d_r0 + d_shape[0] - 1 <= h - 3:
                for _ in range(60):
                    d_c0 = random.randint(2, w - d_shape[1] - 3)
                    cand = (d_r0, d_c0, d_r0 + d_shape[0] - 1, d_c0 + d_shape[1] - 1)
                    if any(_boxes_overlap(cand, other, padding=2) for other in boxes):
                        continue
                    d_box = cand
                    break
        else:
            d_shape = random.choice(OUTER_SHAPES)
            d_row_lo = 2 if top_split else h2_row + 2
            d_row_hi = h2_row - 3 if top_split else h - d_shape[0] - 3
            if d_row_lo <= d_row_hi:
                for _ in range(60):
                    d_r0 = random.randint(d_row_lo, d_row_hi)
                    d_c0_lo = max(2, v3_col - d_shape[1] + 1)
                    d_c0_hi = min(v3_col, w - d_shape[1] - 3)
                    if d_c0_lo > d_c0_hi:
                        continue
                    d_c0 = random.randint(d_c0_lo, d_c0_hi)
                    cand = (d_r0, d_c0, d_r0 + d_shape[0] - 1, d_c0 + d_shape[1] - 1)
                    if any(_boxes_overlap(cand, other, padding=2) for other in boxes):
                        continue
                    d_box = cand
                    break
        if d_box is not None:
            containers.append({"outer": d_box, "inner": (d_box[0] + 1, d_box[1] + 1, d_box[2] - 1, d_box[3] - 1), "area": 8, "tag": "D"})
            boxes.append(d_box)

    rows = {h2_row, h4_row}
    cols = {v1_col, v3_col}
    if len(rows) != 2 or len(cols) != 2:
        return None
    if abs(h2_row - h4_row) < 2 or abs(v1_col - v3_col) < 2:
        return None

    spec = {
        "shape": (h, w),
        "containers": containers,
        "v1_col": v1_col,
        "h2_row": h2_row,
        "v3_col": v3_col,
        "h4_row": h4_row,
        "top_split": top_split,
    }
    inp = _build_input(spec)
    if len(_detect_containers(inp)) != len(containers):
        return None
    return inp


def _semantic_ok(grid: np.ndarray) -> bool:
    containers = _detect_containers(grid)
    if len(containers) < 3 or len(containers) > 4:
        return False

    try:
        solved, meta = _solve(grid)
    except AssertionError:
        return False

    track_grid = meta["track_grid"]
    pair_meta = meta["pairs"]
    charge_meta = meta["charge"]

    try:
        no_border_track, _ = _trace_pairs(grid, containers, stop_on_borders=False, stop_on_trails=True)
        no_trail_track, _ = _trace_pairs(grid, containers, stop_on_borders=True, stop_on_trails=False)
    except AssertionError:
        return False

    if np.array_equal(track_grid, no_border_track):
        return False
    if np.array_equal(track_grid, no_trail_track):
        return False
    if not any(info["blocked_by_border"] for info in pair_meta):
        return False
    if not any(info["blocked_by_trail"] for info in pair_meta):
        return False
    if not all(info["total"] > 0 for info in charge_meta):
        return False
    if not any(0 < info["charge"] < 8 for info in charge_meta):
        return False
    if len({info["side"] for info in charge_meta}) < 2:
        return False
    if not any(sum(count > 0 for count in info["counts"].values()) >= 2 for info in charge_meta):
        return False

    orange_count = int(np.sum(solved == ORANGE))
    if orange_count <= 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_layout()
        if grid is None:
            continue
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
