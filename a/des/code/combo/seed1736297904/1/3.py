from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

# --------------- solver helpers ---------------


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []

    for r in range(h):
        vals = np.unique(grid[r, :])
        if vals.size == 1 and int(vals[0]) != bg:
            candidates.append(("horizontal", int(r), int(vals[0])))

    for c in range(w):
        vals = np.unique(grid[:, c])
        if vals.size == 1 and int(vals[0]) != bg:
            candidates.append(("vertical", int(c), int(vals[0])))

    assert len(candidates) == 1, "expected exactly one full separator line"
    return candidates[0]


def _side_mask(shape: tuple[int, int], orient: str, axis: int, side: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    if orient == "vertical":
        return cc < axis if side < 0 else cc > axis
    assert orient == "horizontal"
    return rr < axis if side < 0 else rr > axis


def _extract_side_objects(
    grid: np.ndarray, bg: int, sep_color: int, orient: str, axis: int, side: int
) -> dict[int, dict]:
    allowed = _side_mask(grid.shape, orient, axis, side) & (grid != bg) & (grid != sep_color)
    visited = np.zeros_like(allowed, dtype=bool)
    h, w = grid.shape
    objects: list[dict] = []

    for r_raw, c_raw in np.argwhere(allowed):
        r = int(r_raw)
        c = int(c_raw)
        if visited[r, c]:
            continue

        color = int(grid[r, c])
        queue = deque([(r, c)])
        visited[r, c] = True
        coords: list[tuple[int, int]] = []

        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for dr, dc in DIRS4:
                nr = rr + dr
                nc = cc + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and not visited[nr, nc]
                    and allowed[nr, nc]
                    and int(grid[nr, nc]) == color
                ):
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        arr = np.array(coords, dtype=int)
        objects.append(
            {
                "color": color,
                "side": int(side),
                "coords": arr,
                "area": int(len(coords)),
                "bbox": _bbox(arr),
            }
        )

    by_color: dict[int, dict] = {}
    for obj in objects:
        color = int(obj["color"])
        assert color not in by_color, "each side must contain exactly one object per color"
        by_color[color] = obj
    return by_color


def _distance_num(obj: dict, orient: str, axis: int) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    if orient == "vertical":
        return abs(int(c0 + c1 - 2 * axis))
    return abs(int(r0 + r1 - 2 * axis))


def _reflect_coords(coords: np.ndarray, orient: str, axis: int) -> np.ndarray:
    mirrored = coords.copy()
    if orient == "vertical":
        mirrored[:, 1] = 2 * int(axis) - mirrored[:, 1]
    else:
        mirrored[:, 0] = 2 * int(axis) - mirrored[:, 0]
    return mirrored


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    orient, axis, sep_color = _find_separator(g, bg)

    side_objects = {
        -1: _extract_side_objects(g, bg, sep_color, orient, axis, -1),
        1: _extract_side_objects(g, bg, sep_color, orient, axis, 1),
    }
    left_colors = set(side_objects[-1])
    right_colors = set(side_objects[1])
    assert left_colors and left_colors == right_colors, "both sides must contain the same color set"

    pairs: list[dict] = []
    winners: list[dict] = []

    serial = 0
    for side in (-1, 1):
        ordered_side = sorted(
            side_objects[side].values(),
            key=lambda obj: tuple(map(int, obj["bbox"][:2])),
        )
        for obj in ordered_side:
            obj["serial"] = serial
            serial += 1
            obj["distance_num"] = _distance_num(obj, orient, axis)

    ordered_colors = list(side_objects[-1].keys())
    ordered_colors.sort(key=lambda color: tuple(map(int, side_objects[-1][color]["bbox"][:2])))

    for color in ordered_colors:
        a = side_objects[-1][color]
        b = side_objects[1][color]
        assert int(a["area"]) != int(b["area"]), "paired objects must have different areas"
        winner = a if int(a["area"]) > int(b["area"]) else b
        loser = b if winner is a else a
        pairs.append({"color": int(color), "winner": winner, "loser": loser})
        winners.append(winner)

    return {
        "grid": g,
        "bg": int(bg),
        "orient": orient,
        "axis": int(axis),
        "sep_color": int(sep_color),
        "pairs": pairs,
        "winners": winners,
    }


def _build_candidates(info: dict) -> list[dict]:
    h, w = info["grid"].shape
    candidates: list[dict] = []

    for pair in info["pairs"]:
        winner = pair["winner"]
        reflected = _reflect_coords(winner["coords"], info["orient"], info["axis"])
        assert np.all((0 <= reflected[:, 0]) & (reflected[:, 0] < h))
        assert np.all((0 <= reflected[:, 1]) & (reflected[:, 1] < w))

        for kind_rank, coords in ((0, winner["coords"]), (1, reflected)):
            candidates.append(
                {
                    "color": int(winner["color"]),
                    "coords": coords,
                    "distance_num": int(winner["distance_num"]),
                    "kind_rank": int(kind_rank),
                    "source_serial": int(winner["serial"]),
                    "bbox": tuple(map(int, winner["bbox"])),
                }
            )

    return candidates


def _contest_stats(candidates: list[dict]) -> tuple[int, int]:
    contests: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for cand in candidates:
        color = int(cand["color"])
        dist = int(cand["distance_num"])
        for r_raw, c_raw in cand["coords"]:
            key = (int(r_raw), int(c_raw))
            contests.setdefault(key, []).append((dist, color))

    overlap_pixels = 0
    ambiguous_pixels = 0
    for contenders in contests.values():
        if len({color for _dist, color in contenders}) < 2:
            continue
        overlap_pixels += 1
        best_dist = min(dist for dist, _color in contenders)
        if len({color for dist, color in contenders if dist == best_dist}) >= 2:
            ambiguous_pixels += 1

    return int(overlap_pixels), int(ambiguous_pixels)


def _render_output(info: dict) -> np.ndarray:
    g = info["grid"]
    out = np.full_like(g, int(info["bg"]))

    if info["orient"] == "vertical":
        out[:, int(info["axis"])] = int(info["sep_color"])
    else:
        out[int(info["axis"]), :] = int(info["sep_color"])

    best: dict[tuple[int, int], tuple[tuple[int, int, int, int, int, int, int], int]] = {}
    for cand in _build_candidates(info):
        r0, c0, r1, c1 = cand["bbox"]
        priority = (
            int(cand["distance_num"]),
            int(cand["kind_rank"]),
            int(r0),
            int(c0),
            int(r1 - r0),
            int(c1 - c0),
            int(cand["source_serial"]),
        )
        color = int(cand["color"])
        for r_raw, c_raw in cand["coords"]:
            key = (int(r_raw), int(c_raw))
            prev = best.get(key)
            if prev is None or priority < prev[0]:
                best[key] = (priority, color)

    for (r, c), (_priority, color) in best.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare each cross-separator color pair by area, erase the smaller object,
    and reflect the larger object across the separator while keeping the
    original separator line in place.
    """
    info = _analyze(grid)
    return _render_output(info)


# --------------- generator helpers ---------------

GEN_ATTEMPTS = 400
OBJ_RETRIES = 100
PLACE_RETRIES = 120

HEIGHT_RANGE = (12, 18)
SIDE_WIDTH_RANGE = (8, 11)
PAIR_COUNT_RANGE = (2, 4)
WINNER_AREA_RANGE = (5, 11)
LOSER_AREA_RANGE = (3, 8)
OVERLAP_PROB = 0.35


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr, dc in DIRS8:
        rs_src = slice(max(0, -dr), mask.shape[0] - max(0, dr))
        cs_src = slice(max(0, -dc), mask.shape[1] - max(0, dc))
        rs_dst = slice(max(0, dr), mask.shape[0] - max(0, -dr))
        cs_dst = slice(max(0, dc), mask.shape[1] - max(0, -dc))
        out[rs_dst, cs_dst] |= mask[rs_src, cs_src]
    return out


def _make_connected_mask(
    h: int, w: int, area: int, required_cell: tuple[int, int] | None = None
) -> np.ndarray | None:
    base = h + w - 1
    if not (base <= area <= h * w):
        return None

    rr = random.randrange(h) if required_cell is None else int(required_cell[0])
    cc = random.randrange(w) if required_cell is None else int(required_cell[1])
    if not (0 <= rr < h and 0 <= cc < w):
        return None

    selected = {(rr, c) for c in range(w)}
    selected.update({(r, cc) for r in range(h)})
    frontier: set[tuple[int, int]] = set()

    def extend_frontier(cell_r: int, cell_c: int) -> None:
        for dr, dc in DIRS4:
            nr = cell_r + dr
            nc = cell_c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in selected:
                frontier.add((nr, nc))

    for r, c in tuple(selected):
        extend_frontier(r, c)

    while len(selected) < area:
        if not frontier:
            return None
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        selected.add(pick)
        extend_frontier(*pick)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in selected:
        mask[r, c] = True
    return mask


def _sample_mask(
    max_h: int,
    max_w: int,
    min_area: int,
    max_area: int,
    *,
    fixed_shape: tuple[int, int] | None = None,
    required_cell: tuple[int, int] | None = None,
) -> np.ndarray | None:
    for _ in range(OBJ_RETRIES):
        if fixed_shape is None:
            h = random.randint(2, min(5, max_h))
            w = random.randint(2, min(5, max_w))
        else:
            h, w = map(int, fixed_shape)

        lo = max(int(min_area), int(h + w - 1))
        hi = min(int(max_area), int(h * w))
        if lo > hi:
            continue

        area = random.randint(lo, hi)
        mask = _make_connected_mask(h, w, area, required_cell=required_cell)
        if mask is None:
            continue
        return mask
    return None


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask).astype(int)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    blocked = _dilate8(occupied)
    return not bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any())


def _paint_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)
    occupied[r0 : r0 + h, c0 : c0 + w] |= mask


def _side_col_start(axis: int, side: int, local_c0: int) -> int:
    if side < 0:
        return int(local_c0)
    return int(axis + 1 + local_c0)


def _random_position(occupied: np.ndarray, axis: int, side: int, mask: np.ndarray) -> tuple[int, int] | None:
    h_grid, w_grid = occupied.shape
    h, w = mask.shape
    side_width = axis if side < 0 else w_grid - axis - 1
    starts: list[tuple[int, int]] = []
    for r0 in range(h_grid - h + 1):
        for local_c0 in range(side_width - w + 1):
            c0 = _side_col_start(axis, side, local_c0)
            starts.append((r0, c0))
    random.shuffle(starts)

    for r0, c0 in starts:
        if _can_place(occupied, mask, r0, c0):
            return int(r0), int(c0)
    return None


def _sample_winner_sides(pair_count: int, force_overlap: bool) -> list[int]:
    if force_overlap:
        left_count = random.randint(1, pair_count - 1)
        sides = [-1] * left_count + [1] * (pair_count - left_count)
        random.shuffle(sides)
        return sides

    mode = random.choice(("mixed", "all_left", "all_right"))
    if mode == "mixed":
        left_count = random.randint(1, pair_count - 1)
        sides = [-1] * left_count + [1] * (pair_count - left_count)
        random.shuffle(sides)
        return sides
    return ([-1] * pair_count) if mode == "all_left" else ([1] * pair_count)


def _place_forced_overlap_winners(
    grid: np.ndarray,
    occupied: np.ndarray,
    axis: int,
    pair_records: list[dict],
    left_idx: int,
    right_idx: int,
) -> bool:
    h_grid, w_grid = grid.shape
    side_width = axis
    deltas = [(-1, -1), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 1)]

    for _ in range(PLACE_RETRIES):
        mask_a = _sample_mask(
            h_grid,
            side_width,
            WINNER_AREA_RANGE[0],
            WINNER_AREA_RANGE[1],
        )
        if mask_a is None:
            continue

        h_a, w_a = mask_a.shape
        local_left_choices = [
            lc for lc in range(side_width - w_a + 1) if 0 < side_width - lc - w_a < side_width - w_a
        ]
        if not local_left_choices:
            continue
        random.shuffle(local_left_choices)

        row_choices = list(range(h_grid - h_a + 1))
        random.shuffle(row_choices)
        placed = False

        for r_a in row_choices:
            for lc_a in local_left_choices:
                c_a = lc_a
                if not _can_place(occupied, mask_a, r_a, c_a):
                    continue

                rc_ref = side_width - lc_a - w_a
                random.shuffle(deltas)
                for dr, dc in deltas:
                    r_b = r_a + dr
                    rc_b = rc_ref + dc
                    c_b = axis + 1 + rc_b
                    if not (0 <= r_b <= h_grid - h_a and 0 <= rc_b <= side_width - w_a):
                        continue
                    if not _can_place(occupied, mask_a, r_b, c_b):
                        continue

                    a_coords = _coords_from_mask(mask_a, r_a, c_a)
                    a_reflected = _reflect_coords(a_coords, "vertical", axis)
                    overlap_cells = [
                        (int(r), int(c))
                        for r, c in a_reflected
                        if r_b <= int(r) < r_b + h_a and c_b <= int(c) < c_b + w_a
                    ]
                    if not overlap_cells:
                        continue

                    required_global = random.choice(overlap_cells)
                    required_local = (required_global[0] - r_b, required_global[1] - c_b)
                    mask_b = _sample_mask(
                        h_grid,
                        side_width,
                        WINNER_AREA_RANGE[0],
                        WINNER_AREA_RANGE[1],
                        fixed_shape=(h_a, w_a),
                        required_cell=required_local,
                    )
                    if mask_b is None:
                        continue
                    if not _can_place(occupied, mask_b, r_b, c_b):
                        continue

                    left_rec = pair_records[left_idx]
                    right_rec = pair_records[right_idx]
                    left_rec["winner_mask"] = mask_a
                    left_rec["winner_pos"] = (int(r_a), int(c_a))
                    right_rec["winner_mask"] = mask_b
                    right_rec["winner_pos"] = (int(r_b), int(c_b))
                    _paint_mask(grid, occupied, mask_a, left_rec["color"], r_a, c_a)
                    _paint_mask(grid, occupied, mask_b, right_rec["color"], r_b, c_b)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break

        if placed:
            return True
    return False


def _build_vertical_input(force_overlap: bool) -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    height = random.randint(*HEIGHT_RANGE)
    side_width = random.randint(*SIDE_WIDTH_RANGE)
    width = 2 * side_width + 1
    axis = side_width

    palette = random.sample(range(10), pair_count + 2)
    bg = int(palette[0])
    sep_color = int(palette[1])
    pair_colors = [int(c) for c in palette[2:]]
    winner_sides = _sample_winner_sides(pair_count, force_overlap)

    grid = np.full((height, width), bg, dtype=int)
    grid[:, axis] = sep_color
    occupied = np.zeros((height, width), dtype=bool)

    pair_records = [{"color": color, "winner_side": side} for color, side in zip(pair_colors, winner_sides)]

    forced_pair_indices: set[int] = set()
    if force_overlap:
        left_choices = [i for i, side in enumerate(winner_sides) if side < 0]
        right_choices = [i for i, side in enumerate(winner_sides) if side > 0]
        left_idx = random.choice(left_choices)
        right_idx = random.choice(right_choices)
        if not _place_forced_overlap_winners(grid, occupied, axis, pair_records, left_idx, right_idx):
            return None
        forced_pair_indices = {left_idx, right_idx}

    remaining = [i for i in range(pair_count) if i not in forced_pair_indices]
    random.shuffle(remaining)

    for idx in remaining:
        rec = pair_records[idx]
        mask = _sample_mask(height, side_width, WINNER_AREA_RANGE[0], WINNER_AREA_RANGE[1])
        if mask is None:
            return None
        pos = _random_position(occupied, axis, rec["winner_side"], mask)
        if pos is None:
            return None
        rec["winner_mask"] = mask
        rec["winner_pos"] = pos
        _paint_mask(grid, occupied, mask, rec["color"], *pos)

    loser_order = list(range(pair_count))
    loser_order.sort(key=lambda idx: int(pair_records[idx]["winner_mask"].sum()), reverse=True)

    for idx in loser_order:
        rec = pair_records[idx]
        winner_area = int(rec["winner_mask"].sum())
        loser_max_area = min(LOSER_AREA_RANGE[1], winner_area - 1)
        if loser_max_area < LOSER_AREA_RANGE[0]:
            return None
        loser_mask = _sample_mask(height, side_width, LOSER_AREA_RANGE[0], loser_max_area)
        if loser_mask is None:
            return None
        loser_side = -int(rec["winner_side"])
        pos = _random_position(occupied, axis, loser_side, loser_mask)
        if pos is None:
            return None
        _paint_mask(grid, occupied, loser_mask, rec["color"], *pos)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        force_overlap = random.random() < OVERLAP_PROB
        grid = _build_vertical_input(force_overlap=force_overlap)
        if grid is None:
            continue
        if random.random() < 0.5:
            grid = grid.T.copy()

        try:
            info = _analyze(grid)
            candidates = _build_candidates(info)
            overlap_pixels, ambiguous_pixels = _contest_stats(candidates)
            if ambiguous_pixels > 0:
                continue
            if force_overlap and overlap_pixels == 0:
                continue
            out = main(grid)
            if np.array_equal(out, grid):
                continue
            return grid
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid input")
