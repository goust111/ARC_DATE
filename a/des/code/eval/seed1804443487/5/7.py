from __future__ import annotations

import random

import numpy as np


BG = 0
ARROW = 7
PIXEL_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
GEN_RETRIES = 400
SCENARIOS = ("mixed", "cross", "tie", "truncate")

PATTERNS = {
    "right": np.array([[0, 0, ARROW], [ARROW, ARROW, ARROW], [0, 0, ARROW]], dtype=int),
    "left": np.array([[ARROW, 0, 0], [ARROW, ARROW, ARROW], [ARROW, 0, 0]], dtype=int),
    "down": np.array([[0, ARROW, 0], [0, ARROW, 0], [ARROW, ARROW, ARROW]], dtype=int),
    "up": np.array([[ARROW, ARROW, ARROW], [0, ARROW, 0], [0, ARROW, 0]], dtype=int),
}

DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _points_inward(name: str, r0: int, c0: int, h: int, w: int) -> bool:
    left_gap = c0
    right_gap = w - c0 - 3
    top_gap = r0
    bottom_gap = h - r0 - 3
    if name == "right":
        return left_gap <= right_gap
    if name == "left":
        return right_gap <= left_gap
    if name == "down":
        return top_gap <= bottom_gap
    return bottom_gap <= top_gap


def _arrow_from_bbox(name: str, r0: int, c0: int, h: int, w: int) -> dict:
    center = (r0 + 1, c0 + 1)
    if name == "right":
        tip = (r0 + 1, c0 + 2)
        back_capacity = c0
        front_capacity = w - c0 - 3
        edge = "left"
    elif name == "left":
        tip = (r0 + 1, c0)
        back_capacity = w - c0 - 3
        front_capacity = c0
        edge = "right"
    elif name == "down":
        tip = (r0 + 2, c0 + 1)
        back_capacity = r0
        front_capacity = h - r0 - 3
        edge = "top"
    else:
        tip = (r0, c0 + 1)
        back_capacity = h - r0 - 3
        front_capacity = r0
        edge = "bottom"
    return {
        "name": name,
        "bbox": (r0, c0),
        "center": center,
        "tip": tip,
        "dir": DIRS[name],
        "back_capacity": back_capacity,
        "front_capacity": front_capacity,
        "edge": edge,
    }


def _arrow_from_edge(edge: str, center: int, gap: int, h: int, w: int) -> dict | None:
    if edge == "left":
        arrow = _arrow_from_bbox("right", center - 1, gap, h, w)
    elif edge == "right":
        arrow = _arrow_from_bbox("left", center - 1, w - 3 - gap, h, w)
    elif edge == "top":
        arrow = _arrow_from_bbox("down", gap, center - 1, h, w)
    else:
        arrow = _arrow_from_bbox("up", h - 3 - gap, center - 1, h, w)
    r0, c0 = arrow["bbox"]
    if not (0 <= r0 and 0 <= c0 and r0 + 2 < h and c0 + 2 < w):
        return None
    if not _points_inward(arrow["name"], r0, c0, h, w):
        return None
    if arrow["front_capacity"] < 2:
        return None
    return arrow


def _detect_arrows(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows: list[dict] = []
    covered: set[tuple[int, int]] = set()

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            block = g[r0 : r0 + 3, c0 : c0 + 3]
            if int(np.count_nonzero(block == ARROW)) != 5:
                continue
            for name, pattern in PATTERNS.items():
                if not np.array_equal(block, pattern):
                    continue
                if not _points_inward(name, r0, c0, h, w):
                    continue
                arrow = _arrow_from_bbox(name, r0, c0, h, w)
                arrows.append(arrow)
                for rr in range(r0, r0 + 3):
                    for cc in range(c0, c0 + 3):
                        if int(g[rr, cc]) == ARROW:
                            covered.add((rr, cc))
                break

    orange_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == ARROW)}
    assert covered == orange_cells, "all orange pixels must belong to arrows"
    arrows.sort(key=lambda arrow: (arrow["center"][0], arrow["center"][1], arrow["name"]))
    return arrows


def _claim_distance(arrow: dict, r: int, c: int) -> int | None:
    center_r, center_c = arrow["center"]
    tip_r, tip_c = arrow["tip"]
    name = arrow["name"]
    if name == "right" and r == center_r and c > tip_c:
        return c - tip_c
    if name == "left" and r == center_r and c < tip_c:
        return tip_c - c
    if name == "down" and c == center_c and r > tip_r:
        return r - tip_r
    if name == "up" and c == center_c and r < tip_r:
        return tip_r - r
    return None


def _claimants(arrows: list[dict], r: int, c: int) -> list[tuple[int, int]]:
    claimants: list[tuple[int, int]] = []
    for idx, arrow in enumerate(arrows):
        dist = _claim_distance(arrow, r, c)
        if dist is not None:
            claimants.append((idx, dist))
    return claimants


def _collect_assignments(grid: np.ndarray, arrows: list[dict]) -> tuple[dict[int, list[tuple[int, int, int, int]]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assignments: dict[int, list[tuple[int, int, int, int]]] = {idx: [] for idx in range(len(arrows))}
    owners: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value in (BG, ARROW):
                continue
            claimants = _claimants(arrows, r, c)
            if not claimants:
                continue
            best_idx = min(
                claimants,
                key=lambda item: (item[1], arrows[item[0]]["center"][0], arrows[item[0]]["center"][1]),
            )[0]
            dist = next(dist for idx, dist in claimants if idx == best_idx)
            assignments[best_idx].append((dist, r, c, value))
            owners[(r, c)] = best_idx

    for idx in assignments:
        assignments[idx].sort(key=lambda item: (item[0], item[1], item[2]))
    return assignments, owners


def _iter_back_positions(arrow: dict, h: int, w: int):
    center_r, center_c = arrow["center"]
    name = arrow["name"]
    if name == "right":
        for c in range(arrow["bbox"][1] - 1, -1, -1):
            yield (center_r, c)
    elif name == "left":
        start = arrow["bbox"][1] + 3
        for c in range(start, w):
            yield (center_r, c)
    elif name == "down":
        for r in range(arrow["bbox"][0] - 1, -1, -1):
            yield (r, center_c)
    else:
        start = arrow["bbox"][0] + 3
        for r in range(start, h):
            yield (r, center_c)


def _line_cell(arrow: dict, dist: int) -> tuple[int, int]:
    tip_r, tip_c = arrow["tip"]
    dr, dc = arrow["dir"]
    return (tip_r + dr * dist, tip_c + dc * dist)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows = _detect_arrows(g)
    assert arrows, "expected at least one arrow"

    assignments, _ = _collect_assignments(g, arrows)
    out = g.copy()

    for items in assignments.values():
        for _dist, r, c, _value in items:
            out[r, c] = BG

    for idx, arrow in enumerate(arrows):
        for (_dist, _r, _c, color), (rr, cc) in zip(assignments[idx], _iter_back_positions(arrow, h, w)):
            out[rr, cc] = color

    return out


def _gap_choices(edge: str, h: int, w: int) -> list[int]:
    span = w if edge in ("left", "right") else h
    max_gap = max(1, min(3, span - 6))
    return list(range(1, max_gap + 1))


def _paint_arrow(grid: np.ndarray, arrow: dict) -> None:
    r0, c0 = arrow["bbox"]
    grid[r0 : r0 + 3, c0 : c0 + 3] = PATTERNS[arrow["name"]]


def _try_place_arrow(grid: np.ndarray, reserved: set[tuple[int, int]], edge: str, center: int, gap: int) -> dict | None:
    h, w = grid.shape
    arrow = _arrow_from_edge(edge, center, gap, h, w)
    if arrow is None:
        return None
    r0, c0 = arrow["bbox"]
    for rr in range(max(0, r0 - 1), min(h, r0 + 4)):
        for cc in range(max(0, c0 - 1), min(w, c0 + 4)):
            if (rr, cc) in reserved:
                return None
    if np.any(grid[r0 : r0 + 3, c0 : c0 + 3] != BG):
        return None
    _paint_arrow(grid, arrow)
    for rr in range(max(0, r0 - 1), min(h, r0 + 4)):
        for cc in range(max(0, c0 - 1), min(w, c0 + 4)):
            reserved.add((rr, cc))
    return arrow


def _register_arrow(arrow: dict, used_rows: set[int], used_cols: set[int]) -> None:
    if arrow["name"] in ("right", "left"):
        used_rows.add(arrow["center"][0])
    else:
        used_cols.add(arrow["center"][1])


def _try_random_arrow(
    rng: random.Random,
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
    banned_rows: set[int] | None = None,
    banned_cols: set[int] | None = None,
) -> dict | None:
    h, w = grid.shape
    banned_rows = banned_rows or set()
    banned_cols = banned_cols or set()
    edges = ["left", "right", "top", "bottom"]
    rng.shuffle(edges)

    for edge in edges:
        if edge in ("left", "right"):
            centers = [value for value in range(2, h - 2) if value not in used_rows and value not in banned_rows]
        else:
            centers = [value for value in range(2, w - 2) if value not in used_cols and value not in banned_cols]
        if not centers:
            continue
        gaps = _gap_choices(edge, h, w)
        for _ in range(12):
            center = rng.choice(centers)
            gap = rng.choice(gaps)
            arrow = _try_place_arrow(grid, reserved, edge, center, gap)
            if arrow is not None:
                return arrow
    return None


def _available_line_distances(grid: np.ndarray, arrow: dict, max_span: int | None = None) -> list[int]:
    upper = arrow["front_capacity"] if max_span is None else min(arrow["front_capacity"], max_span)
    available: list[int] = []
    for dist in range(1, upper + 1):
        rr, cc = _line_cell(arrow, dist)
        if int(grid[rr, cc]) == BG:
            available.append(dist)
    return available


def _place_line_pixels(
    rng: random.Random,
    grid: np.ndarray,
    arrow: dict,
    distances: list[int],
    palette: list[int],
) -> bool:
    positions = [_line_cell(arrow, dist) for dist in distances]
    if len(set(positions)) != len(positions):
        return False
    if any(int(grid[r, c]) != BG for r, c in positions):
        return False
    for r, c in positions:
        grid[r, c] = rng.choice(palette)
    return True


def _all_back_cells(arrows: list[dict], h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for arrow in arrows:
        cells.update(_iter_back_positions(arrow, h, w))
    return cells


def _place_unclaimed_pixels(
    rng: random.Random,
    grid: np.ndarray,
    arrows: list[dict],
    reserved: set[tuple[int, int]],
    palette: list[int],
    count: int,
) -> bool:
    h, w = grid.shape
    blocked = reserved | _all_back_cells(arrows, h, w)
    candidates: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if int(grid[r, c]) != BG:
                continue
            if (r, c) in blocked:
                continue
            if _claimants(arrows, r, c):
                continue
            candidates.append((r, c))
    if len(candidates) < count:
        return False
    for r, c in rng.sample(candidates, count):
        grid[r, c] = rng.choice(palette)
    return True


def _generate_tie_case(rng: random.Random, h: int, w: int, palette: list[int]) -> tuple[np.ndarray, dict] | None:
    grid = np.zeros((h, w), dtype=int)
    reserved: set[tuple[int, int]] = set()

    if rng.random() < 0.5:
        row = rng.randint(3, h - 4)
        gap_pairs = []
        for left_gap in _gap_choices("left", h, w):
            for right_gap in _gap_choices("right", h, w):
                diff = (w - 3 - right_gap) - (left_gap + 2)
                if diff >= 4 and diff % 2 == 0:
                    gap_pairs.append((left_gap, right_gap))
        if not gap_pairs:
            return None
        left_gap, right_gap = rng.choice(gap_pairs)
        left_arrow = _try_place_arrow(grid, reserved, "left", row, left_gap)
        right_arrow = _try_place_arrow(grid, reserved, "right", row, right_gap)
        if left_arrow is None or right_arrow is None:
            return None
        tip_left = left_arrow["tip"][1]
        tip_right = right_arrow["tip"][1]
        tie_col = (tip_left + tip_right) // 2
        if not _place_line_pixels(rng, grid, left_arrow, [tie_col - tip_left, tie_col - tip_left - 1], palette):
            return None
        if not _place_line_pixels(rng, grid, right_arrow, [tip_right - (tie_col + 1)], palette):
            return None
        arrows = [left_arrow, right_arrow]
        if not _place_unclaimed_pixels(rng, grid, arrows, reserved, palette, 2):
            return None
        return grid, {"tie_cell": (row, tie_col)}

    col = rng.randint(3, w - 4)
    gap_pairs = []
    for top_gap in _gap_choices("top", h, w):
        for bottom_gap in _gap_choices("bottom", h, w):
            diff = (h - 3 - bottom_gap) - (top_gap + 2)
            if diff >= 4 and diff % 2 == 0:
                gap_pairs.append((top_gap, bottom_gap))
    if not gap_pairs:
        return None
    top_gap, bottom_gap = rng.choice(gap_pairs)
    top_arrow = _try_place_arrow(grid, reserved, "top", col, top_gap)
    bottom_arrow = _try_place_arrow(grid, reserved, "bottom", col, bottom_gap)
    if top_arrow is None or bottom_arrow is None:
        return None
    tip_top = top_arrow["tip"][0]
    tip_bottom = bottom_arrow["tip"][0]
    tie_row = (tip_top + tip_bottom) // 2
    if not _place_line_pixels(rng, grid, top_arrow, [tie_row - tip_top, tie_row - tip_top - 1], palette):
        return None
    if not _place_line_pixels(rng, grid, bottom_arrow, [tip_bottom - (tie_row + 1)], palette):
        return None
    arrows = [top_arrow, bottom_arrow]
    if not _place_unclaimed_pixels(rng, grid, arrows, reserved, palette, 2):
        return None
    return grid, {"tie_cell": (tie_row, col)}


def _generate_cross_case(rng: random.Random, h: int, w: int, palette: list[int]) -> tuple[np.ndarray, dict] | None:
    grid = np.zeros((h, w), dtype=int)
    reserved: set[tuple[int, int]] = set()

    row = rng.randint(4, h - 5)
    col = rng.randint(4, w - 5)
    horizontal_edge = rng.choice(["left", "right"])
    vertical_edge = rng.choice(["top", "bottom"])

    h_arrow = _try_place_arrow(grid, reserved, horizontal_edge, row, rng.choice(_gap_choices(horizontal_edge, h, w)))
    v_arrow = _try_place_arrow(grid, reserved, vertical_edge, col, rng.choice(_gap_choices(vertical_edge, h, w)))
    if h_arrow is None or v_arrow is None:
        return None

    cross_cell = (row, col)
    h_dist = _claim_distance(h_arrow, *cross_cell)
    v_dist = _claim_distance(v_arrow, *cross_cell)
    if h_dist is None or v_dist is None or h_dist == v_dist:
        return None
    if int(grid[cross_cell]) != BG:
        return None
    grid[cross_cell] = rng.choice(palette)

    extra_h = [dist for dist in _available_line_distances(grid, h_arrow, 8) if dist != h_dist]
    extra_v = [dist for dist in _available_line_distances(grid, v_arrow, 8) if dist != v_dist]
    if not extra_h or not extra_v:
        return None
    if not _place_line_pixels(rng, grid, h_arrow, [rng.choice(extra_h)], palette):
        return None
    if not _place_line_pixels(rng, grid, v_arrow, [rng.choice(extra_v)], palette):
        return None

    arrows = [h_arrow, v_arrow]
    extra_arrow = _try_random_arrow(rng, grid, reserved, {row}, {col}, {row}, {col})
    if extra_arrow is not None:
        arrows.append(extra_arrow)
        extra_distances = _available_line_distances(grid, extra_arrow, 6)
        if extra_distances:
            _place_line_pixels(rng, grid, extra_arrow, [rng.choice(extra_distances)], palette)

    if not _place_unclaimed_pixels(rng, grid, arrows, reserved, palette, 1):
        return None
    return grid, {"cross_cell": cross_cell}


def _generate_truncate_case(rng: random.Random, h: int, w: int, palette: list[int]) -> tuple[np.ndarray, dict] | None:
    grid = np.zeros((h, w), dtype=int)
    reserved: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    core_edge = rng.choice(["left", "right", "top", "bottom"])
    if core_edge in ("left", "right"):
        center = rng.randint(3, h - 4)
    else:
        center = rng.randint(3, w - 4)
    core_gap = rng.choice([1, 2])
    core_arrow = _try_place_arrow(grid, reserved, core_edge, center, core_gap)
    if core_arrow is None:
        return None
    _register_arrow(core_arrow, used_rows, used_cols)
    if core_arrow["front_capacity"] <= core_arrow["back_capacity"] + 1:
        return None

    count = min(core_arrow["front_capacity"], core_arrow["back_capacity"] + rng.randint(2, 4))
    if not _place_line_pixels(rng, grid, core_arrow, list(range(1, count + 1)), palette):
        return None

    arrows = [core_arrow]
    extra_arrow = _try_random_arrow(rng, grid, reserved, used_rows, used_cols)
    if extra_arrow is not None:
        arrows.append(extra_arrow)
        _register_arrow(extra_arrow, used_rows, used_cols)
        extra_distances = _available_line_distances(grid, extra_arrow, 7)
        if extra_distances:
            take = min(len(extra_distances), rng.randint(1, min(2, len(extra_distances))))
            _place_line_pixels(rng, grid, extra_arrow, sorted(rng.sample(extra_distances, take)), palette)

    if not _place_unclaimed_pixels(rng, grid, arrows, reserved, palette, 2):
        return None
    return grid, {}


def _generate_mixed_case(rng: random.Random, h: int, w: int, palette: list[int]) -> tuple[np.ndarray, dict] | None:
    grid = np.zeros((h, w), dtype=int)
    reserved: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    arrows: list[dict] = []

    target = rng.randint(3, 5)
    for _ in range(2):
        arrow = _try_random_arrow(rng, grid, reserved, used_rows, used_cols)
        if arrow is None:
            return None
        arrows.append(arrow)
        _register_arrow(arrow, used_rows, used_cols)

    while len(arrows) < target:
        arrow = _try_random_arrow(rng, grid, reserved, used_rows, used_cols)
        if arrow is None:
            break
        arrows.append(arrow)
        _register_arrow(arrow, used_rows, used_cols)

    if len(arrows) < 2:
        return None

    movers = rng.sample(arrows, k=rng.randint(2, len(arrows)))
    active = 0
    for arrow in movers:
        available = _available_line_distances(grid, arrow, 8)
        if not available:
            continue
        take = min(len(available), rng.randint(1, min(3, len(available))))
        if _place_line_pixels(rng, grid, arrow, sorted(rng.sample(available, take)), palette):
            active += 1

    if active < 2:
        return None
    if not _place_unclaimed_pixels(rng, grid, arrows, reserved, palette, rng.randint(1, 3)):
        return None
    return grid, {}


def _generate_case(rng: random.Random, scenario: str) -> tuple[np.ndarray, dict] | None:
    h = rng.randint(12, 20)
    w = rng.randint(12, 20)
    palette = rng.sample(list(PIXEL_COLORS), rng.randint(3, 5))

    if scenario == "tie":
        return _generate_tie_case(rng, h, w, palette)
    if scenario == "cross":
        return _generate_cross_case(rng, h, w, palette)
    if scenario == "truncate":
        return _generate_truncate_case(rng, h, w, palette)
    return _generate_mixed_case(rng, h, w, palette)


def _validate_generated(grid: np.ndarray, scenario: str, meta: dict) -> bool:
    try:
        arrows = _detect_arrows(grid)
        if not arrows:
            return False
        assignments, owners = _collect_assignments(grid, arrows)
        if sum(len(items) for items in assignments.values()) == 0:
            return False
        main(grid)
    except Exception:
        return False

    if scenario == "tie":
        cell = meta["tie_cell"]
        claimants = _claimants(arrows, *cell)
        if len(claimants) < 2:
            return False
        claimants.sort(key=lambda item: (item[1], arrows[item[0]]["center"][0], arrows[item[0]]["center"][1]))
        if claimants[0][1] != claimants[1][1]:
            return False
        owner = owners.get(cell)
        if owner is None or owner != claimants[0][0]:
            return False

    if scenario == "cross":
        cell = meta["cross_cell"]
        claimants = sorted(_claimants(arrows, *cell), key=lambda item: item[1])
        if len(claimants) < 2 or claimants[0][1] == claimants[1][1]:
            return False

    if scenario == "truncate":
        if not any(len(assignments[idx]) > arrows[idx]["back_capacity"] for idx in range(len(arrows))):
            return False

    if scenario == "mixed":
        if sum(1 for items in assignments.values() if items) < 2:
            return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        scenario = rng.choice(SCENARIOS)
        generated = _generate_case(rng, scenario)
        if generated is None:
            continue
        grid, meta = generated
        if _validate_generated(grid, scenario, meta):
            return grid
    raise RuntimeError("failed to generate a valid input")
