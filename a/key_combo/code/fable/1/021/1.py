from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# Fixed vote plan. It guarantees:
# - exclusive winner is rectangle A
# - naive "count every interior point" winner is rectangle B
WINNER_MAJOR_COUNT = 5
WINNER_MINOR_COUNT = 2
EX_B_COUNT = 3
EX_C_COUNT = 2
AB_COUNT = 1
BC_B_COUNT = 3
BC_C_COUNT = 2
OUTSIDE_EXTRA_COUNT = 2
DECOY_COUNT = 4

GEN_ATTEMPTS = 400


BBox = tuple[int, int, int, int]
Point = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_points(points: np.ndarray) -> BBox:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_from_four(coords: np.ndarray) -> BBox | None:
    if coords.shape != (4, 2):
        return None
    r0, c0, r1, c1 = _bbox_from_points(coords)
    if r0 == r1 or c0 == c1:
        return None
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if set(map(tuple, coords)) != expected:
        return None
    return r0, c0, r1, c1


def _box_corners(box: BBox) -> list[Point]:
    r0, c0, r1, c1 = box
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _shift_box(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _inside_strict(box: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _box_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _render_output(box: BBox, border_color: int, fill_color: int) -> np.ndarray:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    out = np.full((h, w), fill_color, dtype=int)
    out[0, :] = border_color
    out[-1, :] = border_color
    out[:, 0] = border_color
    out[:, -1] = border_color
    return out


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        coords = np.argwhere(grid == color)
        box = _rectangle_from_four(coords)
        if box is not None:
            rectangles.append({"color": int(color), "bbox": box})
    rectangles.sort(key=lambda rect: (rect["bbox"], rect["color"]))
    return rectangles


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rectangles = _detect_rectangles(g)
    assert len(rectangles) >= 2, "expected multiple hidden rectangles"

    corner_colors = {int(rect["color"]) for rect in rectangles}
    ballots: list[list[int]] = [[] for _ in rectangles]
    ballot_counts = [0] * len(rectangles)

    for r, c in np.argwhere(g != BG):
        color = int(g[r, c])
        if color in corner_colors:
            continue
        owners = [i for i, rect in enumerate(rectangles) if _inside_strict(rect["bbox"], int(r), int(c))]
        if len(owners) == 1:
            idx = owners[0]
            ballots[idx].append(color)
            ballot_counts[idx] += 1

    best = max(ballot_counts)
    winners = [i for i, count in enumerate(ballot_counts) if count == best]
    assert len(winners) == 1 and best > 0, "expected one unique vote winner"
    winner = winners[0]

    color_counts = Counter(ballots[winner])
    most_common = color_counts.most_common()
    assert len(most_common) >= 1, "winner must receive at least one valid ballot"
    if len(most_common) >= 2:
        assert most_common[0][1] > most_common[1][1], "winner ballots must have a unique majority color"
    majority_color = int(most_common[0][0])
    return {
        "rectangles": rectangles,
        "corner_colors": corner_colors,
        "ballots": ballots,
        "ballot_counts": ballot_counts,
        "winner": winner,
        "majority_color": majority_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct all hidden corner-defined rectangles, discard overlap ballots, and output the winning frame filled with its ballot-majority color.

    Input structure:
    - The background is black.
    - Some colors appear exactly four times and may or may not form the four corners of an axis-aligned rectangle.
    - Every true corner color uses all of its pixels as one hidden rectangle's four corners; the rectangle borders themselves are absent.
    - Other colored singleton pixels are scattered inside one rectangle, inside overlap zones, or outside every rectangle.

    Rule:
    - Identify every color whose full pixel set is exactly the four corners of an axis-aligned rectangle and reconstruct those rectangles.
    - A scattered pixel counts as a ballot only when it lies strictly inside exactly one reconstructed rectangle.
    - Choose the unique rectangle with the most valid ballots.
    - Return a grid with that rectangle's size: paint the outer border in its corner color and fill the interior with the unique majority ballot color from that rectangle.

    Key insight:
    - The rectangles are invisible and must be recovered from per-color geometry, and overlap pixels are invalid ballots even though they lie inside visible rectangle interiors.
    """
    details = _solve_details(grid)
    winner_rect = details["rectangles"][details["winner"]]
    return _render_output(winner_rect["bbox"], winner_rect["color"], details["majority_color"])


def _transform_point(
    point: Point,
    shape: tuple[int, int],
    *,
    transpose: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> Point:
    h, w = shape
    r, c = point
    if transpose:
        r, c = c, r
        h, w = w, h
    if flip_rows:
        r = h - 1 - r
    if flip_cols:
        c = w - 1 - c
    return int(r), int(c)


def _transform_box(
    box: BBox,
    shape: tuple[int, int],
    *,
    transpose: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> BBox:
    pts = [
        _transform_point(pt, shape, transpose=transpose, flip_rows=flip_rows, flip_cols=flip_cols)
        for pt in _box_corners(box)
    ]
    arr = np.array(pts, dtype=int)
    return _bbox_from_points(arr)


def _candidate_cells(mask: np.ndarray, blocked: list[Point], parity: tuple[int, int]) -> list[Point]:
    pr, pc = parity
    coords: list[Point] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr % 2 != pr or cc % 2 != pc:
            continue
        if any(abs(rr - br) <= 1 and abs(cc - bc) <= 1 for br, bc in blocked):
            continue
        coords.append((rr, cc))
    random.shuffle(coords)
    return coords


def _take(coords: list[Point], n: int) -> list[Point]:
    if len(coords) < n:
        raise ValueError("not enough candidate cells")
    picked = coords[:n]
    del coords[:n]
    return picked


def _choose_decoy(coords: list[Point]) -> list[Point]:
    if len(coords) < DECOY_COUNT:
        raise ValueError("not enough outside cells for decoy")
    for _ in range(300):
        sample = random.sample(coords, DECOY_COUNT)
        if _rectangle_from_four(np.array(sample, dtype=int)) is None:
            return sample
    raise ValueError("failed to find a non-rectangle four-point decoy")


def _build_geometry() -> tuple[tuple[int, int], list[BBox], list[BBox]]:
    a_h = random.randint(7, 9)
    a_w = random.randint(7, 9)
    ab_h = random.randint(2, 3)
    ab_w = random.randint(2, 3)
    bc_h = random.randint(4, 5)
    bc_w = random.randint(4, 5)
    b_h = random.randint(max(10, ab_h + bc_h + 1), 12)
    b_w = random.randint(max(10, ab_w + bc_w + 1), 12)
    c_h = random.randint(7, 9)
    c_w = random.randint(7, 9)

    a_int = (2, 2, 2 + a_h - 1, 2 + a_w - 1)
    a_ir1 = a_int[2]
    a_ic1 = a_int[3]
    b_int = (
        a_ir1 - ab_h + 1,
        a_ic1 - ab_w + 1,
        a_ir1 - ab_h + b_h,
        a_ic1 - ab_w + b_w,
    )
    b_ir1 = b_int[2]
    b_ic1 = b_int[3]
    c_int = (
        b_ir1 - bc_h + 1,
        b_ic1 - bc_w + 1,
        b_ir1 - bc_h + c_h,
        b_ic1 - bc_w + c_w,
    )

    interior_boxes = [a_int, b_int, c_int]
    full_boxes = [(r0 - 1, c0 - 1, r1 + 1, c1 + 1) for r0, c0, r1, c1 in interior_boxes]

    base_h = full_boxes[-1][2] + 3
    base_w = full_boxes[-1][3] + 3
    if base_h > 30 or base_w > 30:
        raise ValueError("geometry exceeds ARC bounds")

    h = random.randint(base_h, min(30, base_h + 3))
    w = random.randint(base_w, min(30, base_w + 3))
    dr = random.randint(0, h - base_h)
    dc = random.randint(0, w - base_w)
    interior_boxes = [_shift_box(box, dr, dc) for box in interior_boxes]
    full_boxes = [_shift_box(box, dr, dc) for box in full_boxes]
    return (h, w), interior_boxes, full_boxes


def _plan_instance() -> dict | None:
    try:
        shape, interior_boxes, full_boxes = _build_geometry()
    except ValueError:
        return None

    blocked = [pt for box in full_boxes for pt in _box_corners(box)]
    masks = [_box_mask(shape, box) for box in interior_boxes]
    a_mask, b_mask, c_mask = masks
    ex_a = a_mask & ~b_mask & ~c_mask
    ex_b = b_mask & ~a_mask & ~c_mask
    ex_c = c_mask & ~a_mask & ~b_mask
    ab = a_mask & b_mask & ~c_mask
    bc = b_mask & c_mask & ~a_mask
    outside = ~(a_mask | b_mask | c_mask)

    parities = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(parities)

    parity = None
    pools: dict[str, list[Point]] = {}
    for candidate_parity in parities:
        trial = {
            "ex_a": _candidate_cells(ex_a, blocked, candidate_parity),
            "ex_b": _candidate_cells(ex_b, blocked, candidate_parity),
            "ex_c": _candidate_cells(ex_c, blocked, candidate_parity),
            "ab": _candidate_cells(ab, blocked, candidate_parity),
            "bc": _candidate_cells(bc, blocked, candidate_parity),
            "outside": _candidate_cells(outside, blocked, candidate_parity),
        }
        if (
            len(trial["ex_a"]) >= WINNER_MAJOR_COUNT + WINNER_MINOR_COUNT
            and len(trial["ex_b"]) >= EX_B_COUNT
            and len(trial["ex_c"]) >= EX_C_COUNT
            and len(trial["ab"]) >= AB_COUNT
            and len(trial["bc"]) >= BC_B_COUNT + BC_C_COUNT
            and len(trial["outside"]) >= OUTSIDE_EXTRA_COUNT + DECOY_COUNT
        ):
            parity = candidate_parity
            pools = trial
            break
    if parity is None:
        return None

    colors = random.sample(PALETTE, 8)
    corner_colors = colors[:3]
    decoy_color = colors[3]
    major_color, minor_color, b_color, c_color = colors[4:]

    points: list[dict] = []
    for idx, box in enumerate(full_boxes):
        for pt in _box_corners(box):
            points.append({"pos": pt, "color": corner_colors[idx], "tag": f"corner{idx}"})

    ex_a_cells = pools["ex_a"].copy()
    ex_b_cells = pools["ex_b"].copy()
    ex_c_cells = pools["ex_c"].copy()
    ab_cells = pools["ab"].copy()
    bc_cells = pools["bc"].copy()
    outside_cells = pools["outside"].copy()

    for pt in _take(ex_a_cells, WINNER_MAJOR_COUNT):
        points.append({"pos": pt, "color": major_color, "tag": "ex0"})
    for pt in _take(ex_a_cells, WINNER_MINOR_COUNT):
        points.append({"pos": pt, "color": minor_color, "tag": "ex0"})
    for pt in _take(ex_b_cells, EX_B_COUNT):
        points.append({"pos": pt, "color": b_color, "tag": "ex1"})
    for pt in _take(ex_c_cells, EX_C_COUNT):
        points.append({"pos": pt, "color": c_color, "tag": "ex2"})
    for pt in _take(ab_cells, AB_COUNT):
        points.append({"pos": pt, "color": minor_color, "tag": "ab"})
    for pt in _take(bc_cells, BC_B_COUNT):
        points.append({"pos": pt, "color": b_color, "tag": "bc"})
    for pt in _take(bc_cells, BC_C_COUNT):
        points.append({"pos": pt, "color": c_color, "tag": "bc"})

    decoy = _choose_decoy(outside_cells)
    decoy_set = set(decoy)
    outside_cells = [pt for pt in outside_cells if pt not in decoy_set]
    for pt in decoy:
        points.append({"pos": pt, "color": decoy_color, "tag": "decoy"})

    outside_pick = _take(outside_cells, OUTSIDE_EXTRA_COUNT)
    points.append({"pos": outside_pick[0], "color": major_color, "tag": "outside"})
    points.append({"pos": outside_pick[1], "color": c_color, "tag": "outside"})

    transpose = random.choice([False, True])
    flip_rows = random.choice([False, True])
    flip_cols = random.choice([False, True])
    out_shape = (shape[1], shape[0]) if transpose else shape

    transformed_rectangles = [
        {
            "color": int(color),
            "bbox": _transform_box(
                box,
                shape,
                transpose=transpose,
                flip_rows=flip_rows,
                flip_cols=flip_cols,
            ),
        }
        for color, box in zip(corner_colors, full_boxes)
    ]
    transformed_points = [
        {
            "pos": _transform_point(
                item["pos"],
                shape,
                transpose=transpose,
                flip_rows=flip_rows,
                flip_cols=flip_cols,
            ),
            "color": int(item["color"]),
            "tag": str(item["tag"]),
        }
        for item in points
    ]
    return {
        "shape": out_shape,
        "rectangles": transformed_rectangles,
        "points": transformed_points,
        "winner": 0,
        "majority_color": int(major_color),
    }


def _render_grid(plan: dict) -> np.ndarray:
    grid = np.zeros(plan["shape"], dtype=int)
    for item in plan["points"]:
        r, c = item["pos"]
        grid[r, c] = int(item["color"])
    return grid


def _validate_plan(plan: dict, grid: np.ndarray) -> bool:
    if _most_common_color(grid) != BG:
        return False

    seen: set[Point] = set()
    for item in plan["points"]:
        pos = tuple(item["pos"])
        if pos in seen:
            return False
        seen.add(pos)

    detected = _detect_rectangles(grid)
    expected_rects = sorted(
        [{"color": int(rect["color"]), "bbox": tuple(rect["bbox"])} for rect in plan["rectangles"]],
        key=lambda rect: (rect["bbox"], rect["color"]),
    )
    if detected != expected_rects:
        return False

    exclusive_counts = [0, 0, 0]
    exclusive_colors = [[], [], []]
    naive_counts = [0, 0, 0]
    corner_positions = {pt for rect in plan["rectangles"] for pt in _box_corners(rect["bbox"])}
    for item in plan["points"]:
        pos = tuple(item["pos"])
        if pos in corner_positions:
            continue
        r, c = pos
        owners = [i for i, rect in enumerate(plan["rectangles"]) if _inside_strict(rect["bbox"], r, c)]
        tag = item["tag"]
        if tag == "ex0" and owners != [0]:
            return False
        if tag == "ex1" and owners != [1]:
            return False
        if tag == "ex2" and owners != [2]:
            return False
        if tag == "ab" and owners != [0, 1]:
            return False
        if tag == "bc" and owners != [1, 2]:
            return False
        if tag in {"outside", "decoy"} and owners:
            return False
        if len(owners) == 1:
            exclusive_counts[owners[0]] += 1
            exclusive_colors[owners[0]].append(int(item["color"]))
        for idx in owners:
            naive_counts[idx] += 1

    if exclusive_counts != [WINNER_MAJOR_COUNT + WINNER_MINOR_COUNT, EX_B_COUNT, EX_C_COUNT]:
        return False
    if naive_counts != [
        WINNER_MAJOR_COUNT + WINNER_MINOR_COUNT + AB_COUNT,
        EX_B_COUNT + AB_COUNT + BC_B_COUNT + BC_C_COUNT,
        EX_C_COUNT + BC_B_COUNT + BC_C_COUNT,
    ]:
        return False
    if not (naive_counts[1] > naive_counts[0] > exclusive_counts[1]):
        return False

    winner_colors = Counter(exclusive_colors[plan["winner"]])
    major = winner_colors.most_common(2)
    if len(major) < 1:
        return False
    if len(major) > 1 and major[0][1] <= major[1][1]:
        return False
    if int(major[0][0]) != int(plan["majority_color"]):
        return False

    decoy_points = [item["pos"] for item in plan["points"] if item["tag"] == "decoy"]
    if len(decoy_points) != DECOY_COUNT:
        return False
    if _rectangle_from_four(np.array(decoy_points, dtype=int)) is not None:
        return False

    expected = _render_output(plan["rectangles"][plan["winner"]]["bbox"], plan["rectangles"][0]["color"], plan["majority_color"])
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        plan = _plan_instance()
        if plan is None:
            continue
        grid = _render_grid(plan)
        if _validate_plan(plan, grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
