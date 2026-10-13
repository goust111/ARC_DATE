import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
DIR_VECS = {
    "R": (0, 1),
    "L": (0, -1),
    "D": (1, 0),
    "U": (-1, 0),
}
ARROW_PATTERNS = {
    "R": {(0, 2), (1, 3), (2, 2)},
    "L": {(0, -1), (1, -2), (2, -1)},
    "D": {(2, 0), (3, 1), (2, 2)},
    "U": {(-1, 0), (-2, 1), (-1, 2)},
}
GEN_ATTEMPTS = 600


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    offsets = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in offsets:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_frame(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components_of_color(grid, color, connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 10 or width < 10:
                continue
            border_size = 2 * (height + width) - 4
            if len(comp) != border_size:
                continue
            ok = True
            for rr in range(r0, r1 + 1):
                if int(grid[rr, c0]) != color or int(grid[rr, c1]) != color:
                    ok = False
                    break
            if not ok:
                continue
            for cc in range(c0, c1 + 1):
                if int(grid[r0, cc]) != color or int(grid[r1, cc]) != color:
                    ok = False
                    break
            if not ok:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or np.any(interior != bg):
                continue
            candidates.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "area": border_size,
                }
            )
    assert candidates
    candidates.sort(key=lambda item: (-item["area"], item["bbox"]))
    return candidates[0]


def _parse_object_component(
    coords: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
) -> dict | None:
    if len(coords) != 7:
        return None
    pts = set(coords)
    rmin, cmin, rmax, cmax = _bbox(coords)
    matches = []
    for br in range(rmin - 1, rmax + 1):
        for bc in range(cmin - 1, cmax + 1):
            block = {
                (br, bc),
                (br, bc + 1),
                (br + 1, bc),
                (br + 1, bc + 1),
            }
            if not block <= pts:
                continue
            arrow = pts - block
            for direction, pattern in ARROW_PATTERNS.items():
                expected = {(br + dr, bc + dc) for dr, dc in pattern}
                if arrow == expected:
                    matches.append((br, bc, direction))
    if len(matches) != 1:
        return None

    br, bc, direction = matches[0]
    r0, c0, r1, c1 = frame_bbox
    if direction == "R":
        dist = c0 - (bc + 1) - 1
        if not (bc + 1 < c0 and r0 + 1 <= br <= r1 - 2):
            return None
    elif direction == "L":
        dist = bc - c1 - 1
        if not (bc > c1 and r0 + 1 <= br <= r1 - 2):
            return None
    elif direction == "D":
        dist = r0 - (br + 1) - 1
        if not (br + 1 < r0 and c0 + 1 <= bc <= c1 - 2):
            return None
    else:
        dist = br - r1 - 1
        if not (br > r1 and c0 + 1 <= bc <= c1 - 2):
            return None
    if dist < 1:
        return None

    return {
        "block_r": br,
        "block_c": bc,
        "dir": direction,
        "dist": int(dist),
    }


def _detect_objects(grid: np.ndarray, bg: int, frame: dict) -> list[dict]:
    frame_color = int(frame["color"])
    frame_bbox = frame["bbox"]
    objects: list[dict] = []
    seen_colors: set[int] = set()
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame_color):
            continue
        comps = _components_of_color(grid, color, connectivity=8)
        for comp in comps:
            parsed = _parse_object_component(comp, frame_bbox)
            if parsed is None:
                continue
            assert color not in seen_colors
            seen_colors.add(color)
            parsed["color"] = color
            objects.append(parsed)
    assert 3 <= len(objects) <= 6
    return objects


def _entry_position(obj: dict, frame_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    if obj["dir"] == "R":
        return obj["block_r"], c0 + 1
    if obj["dir"] == "L":
        return obj["block_r"], c1 - 2
    if obj["dir"] == "D":
        return r0 + 1, obj["block_c"]
    return r1 - 2, obj["block_c"]


def _rect_fits(r: int, c: int, inner_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = inner_bbox
    return r0 <= r and c0 <= c and r + 1 <= r1 and c + 1 <= c1


def _simulate_objects(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    objects: list[dict],
    preserve_order: bool = False,
) -> list[tuple[dict, int, int]] | None:
    h, w = shape
    _ = (h, w)
    r0, c0, r1, c1 = frame_bbox
    inner_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    occ = np.zeros(shape, dtype=bool)
    if preserve_order:
        ordered = list(objects)
    else:
        ordered = sorted(objects, key=lambda obj: (-obj["dist"], obj["block_r"], obj["block_c"], obj["color"]))
    placements: list[tuple[dict, int, int]] = []
    for obj in ordered:
        sr, sc = _entry_position(obj, frame_bbox)
        if not _rect_fits(sr, sc, inner_bbox):
            return None
        if np.any(occ[sr : sr + 2, sc : sc + 2]):
            return None
        r, c = sr, sc
        dr, dc = DIR_VECS[obj["dir"]]
        while True:
            nr = r + dr
            nc = c + dc
            if not _rect_fits(nr, nc, inner_bbox):
                break
            if np.any(occ[nr : nr + 2, nc : nc + 2]):
                break
            r, c = nr, nc
        occ[r : r + 2, c : c + 2] = True
        placements.append((obj, r, c))
    return placements


def _render_output(shape: tuple[int, int], bg: int, frame: dict, placements: list[tuple[dict, int, int]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    frame_color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color
    for obj, r, c in placements:
        out[r : r + 2, c : c + 2] = int(obj["color"])
    return out


def _parse_scene(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    bg = _mode_color(grid)
    frame = _detect_frame(grid, bg)
    objects = _detect_objects(grid, bg, frame)
    return bg, frame, objects


def _solve_from_grid(grid: np.ndarray) -> np.ndarray:
    bg, frame, objects = _parse_scene(grid)
    placements = _simulate_objects(grid.shape, frame["bbox"], objects)
    assert placements is not None
    return _render_output(grid.shape, bg, frame, placements)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored 2x2 block into the central frame along its arrow direction.
    Farther blocks move first; once inside, they slide until hitting the opposite wall
    or an already placed block, so later blocks stack behind earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    return _solve_from_grid(g)


def _build_side_candidates(start: int, stop: int) -> list[int]:
    return list(range(start, stop))


def _pick_separated_values(rng: np.random.Generator, candidates: list[int], count: int, min_gap: int = 3, reserved: tuple[int, ...] = ()) -> list[int] | None:
    if count == 0:
        return []
    reserved_list = list(reserved)
    if len(set(reserved_list)) != len(reserved_list):
        return None
    if any(value not in candidates for value in reserved_list):
        return None
    if any(abs(a - b) < min_gap for i, a in enumerate(reserved_list) for b in reserved_list[i + 1 :]):
        return None
    remaining = [v for v in candidates if v not in reserved_list]
    for _ in range(200):
        order = remaining[:]
        rng.shuffle(order)
        chosen = reserved_list[:]
        for value in order:
            if all(abs(value - prev) >= min_gap for prev in chosen):
                chosen.append(value)
            if len(chosen) == count:
                return chosen
    return None


def _order_sensitive(shape: tuple[int, int], frame: dict, objects: list[dict], bg: int) -> bool:
    desc = _simulate_objects(shape, frame["bbox"], objects)
    if desc is None:
        return False
    asc_objects = sorted(objects, key=lambda obj: (obj["dist"], obj["block_r"], obj["block_c"], obj["color"]))
    asc = _simulate_objects(shape, frame["bbox"], asc_objects, preserve_order=True)
    if asc is None:
        return True
    out_desc = _render_output(shape, bg, frame, desc)
    out_asc = _render_output(shape, bg, frame, asc)
    return not np.array_equal(out_desc, out_asc)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(20, 29))
    w = int(rng.integers(20, 29))

    max_inner_h = min(14, h - 10)
    max_inner_w = min(14, w - 10)
    if max_inner_h < 8 or max_inner_w < 8:
        return None
    inner_h = int(rng.integers(8, max_inner_h + 1))
    inner_w = int(rng.integers(8, max_inner_w + 1))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    top = int(rng.integers(4, h - frame_h - 3))
    left = int(rng.integers(4, w - frame_w - 3))
    bottom_margin = h - (top + frame_h)
    right_margin = w - (left + frame_w)
    frame = {
        "color": FRAME,
        "bbox": (top, left, top + frame_h - 1, left + frame_w - 1),
    }

    r0, c0, r1, c1 = frame["bbox"]
    dmax = {
        "R": left - 2,
        "L": right_margin - 2,
        "D": top - 2,
        "U": bottom_margin - 2,
    }
    maxd = max(dmax.values())
    if maxd < 4:
        return None

    row_lanes = _build_side_candidates(r0 + 1, r1 - 1)
    col_lanes = _build_side_candidates(c0 + 1, c1 - 1)
    row_cap = (len(row_lanes) + 2) // 3
    col_cap = (len(col_lanes) + 2) // 3
    side_cap = {
        "R": row_cap,
        "L": row_cap,
        "D": col_cap,
        "U": col_cap,
    }

    pair_options = []
    if dmax["R"] >= 2 and dmax["L"] >= 2 and row_cap >= 1:
        pair_options.append("H")
    if dmax["D"] >= 2 and dmax["U"] >= 2 and col_cap >= 1:
        pair_options.append("V")
    if not pair_options:
        return None
    pair_axis = str(rng.choice(pair_options))

    nmax = min(6, len(PALETTE), maxd - 1)
    if nmax < 3:
        return None
    n = int(rng.integers(3, nmax + 1))

    if pair_axis == "H":
        d_r = int(rng.integers(2, dmax["R"] + 1))
        options_l = [d for d in range(2, dmax["L"] + 1) if d != d_r]
        if not options_l:
            return None
        d_l = int(rng.choice(options_l))
        forced = [(d_r, "R"), (d_l, "L")]
        reserved_rows = {
            "R": int(rng.choice(row_lanes)),
            "L": None,
        }
        reserved_rows["L"] = reserved_rows["R"]
        reserved_cols: dict[str, int | None] = {"D": None, "U": None}
    else:
        d_d = int(rng.integers(2, dmax["D"] + 1))
        options_u = [d for d in range(2, dmax["U"] + 1) if d != d_d]
        if not options_u:
            return None
        d_u = int(rng.choice(options_u))
        forced = [(d_d, "D"), (d_u, "U")]
        reserved_cols = {
            "D": int(rng.choice(col_lanes)),
            "U": None,
        }
        reserved_cols["U"] = reserved_cols["D"]
        reserved_rows = {"R": None, "L": None}

    used_distances = {distance for distance, _ in forced}
    available_distances = [d for d in range(2, maxd + 1) if d not in used_distances]
    if len(available_distances) < n - 2:
        return None
    extra_distances = rng.choice(np.array(available_distances, dtype=int), size=n - 2, replace=False).tolist()
    distances = forced + [(int(d), "") for d in extra_distances]

    counts = {side: 0 for side in "RLDU"}
    for _, side in forced:
        counts[side] += 1
        side_cap[side] -= 1
        if side_cap[side] < 0:
            return None

    extra_assignments = []
    eligible_sides = [side for side in "RLDU" if dmax[side] >= 2]
    target_distinct = min(len(eligible_sides), n, 3 if rng.random() < 0.65 else 2)
    for _ in range(300):
        temp_counts = counts.copy()
        remaining_cap = side_cap.copy()
        temp_assignments: list[tuple[int, str]] = []
        ok = True
        for distance, _ in sorted((pair for pair in distances if pair[1] == ""), reverse=True):
            candidates = [side for side in "RLDU" if remaining_cap[side] > 0 and distance <= dmax[side]]
            if not candidates:
                ok = False
                break
            min_count = min(temp_counts[side] for side in candidates)
            balanced = [side for side in candidates if temp_counts[side] == min_count]
            side = str(rng.choice(balanced))
            temp_assignments.append((int(distance), side))
            temp_counts[side] += 1
            remaining_cap[side] -= 1
        if ok and sum(temp_counts[side] > 0 for side in "RLDU") >= target_distinct:
            counts = temp_counts
            side_cap = remaining_cap
            extra_assignments = temp_assignments
            break
    if not extra_assignments and n > 2:
        return None

    objects_spec = forced + extra_assignments

    rows_by_side: dict[str, list[int]] = {"R": [], "L": []}
    cols_by_side: dict[str, list[int]] = {"D": [], "U": []}
    for side in ("R", "L"):
        reserved = ()
        if reserved_rows[side] is not None:
            reserved = (int(reserved_rows[side]),)
        picked = _pick_separated_values(rng, row_lanes, counts[side], reserved=reserved)
        if picked is None:
            return None
        rows_by_side[side] = picked
    for side in ("D", "U"):
        reserved = ()
        if reserved_cols[side] is not None:
            reserved = (int(reserved_cols[side]),)
        picked = _pick_separated_values(rng, col_lanes, counts[side], reserved=reserved)
        if picked is None:
            return None
        cols_by_side[side] = picked

    side_lane_index = {side: 0 for side in "RLDU"}
    side_assignments = sorted(objects_spec, key=lambda item: (-item[0], item[1]))
    colors = rng.choice(np.array(PALETTE, dtype=int), size=n, replace=False).tolist()
    grid = np.full((h, w), BG, dtype=int)
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    objects: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for idx, (distance, side) in enumerate(side_assignments):
        color = int(colors[idx])
        if side in ("R", "L"):
            br = int(rows_by_side[side][side_lane_index[side]])
            side_lane_index[side] += 1
            if side == "R":
                bc = c0 - distance - 2
            else:
                bc = c1 + distance + 1
        else:
            bc = int(cols_by_side[side][side_lane_index[side]])
            side_lane_index[side] += 1
            if side == "D":
                br = r0 - distance - 2
            else:
                br = r1 + distance + 1
        cells = {
            (br, bc),
            (br, bc + 1),
            (br + 1, bc),
            (br + 1, bc + 1),
        }
        cells |= {(br + dr, bc + dc) for dr, dc in ARROW_PATTERNS[side]}
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells):
            return None
        if any((rr, cc) in occupied for rr, cc in cells):
            return None
        if any(int(grid[rr, cc]) != BG for rr, cc in cells):
            return None
        for rr, cc in cells:
            grid[rr, cc] = color
            occupied.add((rr, cc))
        objects.append(
            {
                "color": color,
                "block_r": int(br),
                "block_c": int(bc),
                "dir": side,
                "dist": int(distance),
            }
        )

    try:
        parsed_bg, parsed_frame, parsed_objects = _parse_scene(grid)
    except AssertionError:
        return None
    if parsed_bg != BG or parsed_frame["bbox"] != frame["bbox"] or parsed_frame["color"] != FRAME:
        return None
    if len(parsed_objects) != len(objects):
        return None

    want = {
        (obj["color"], obj["block_r"], obj["block_c"], obj["dir"], obj["dist"])
        for obj in objects
    }
    got = {
        (obj["color"], obj["block_r"], obj["block_c"], obj["dir"], obj["dist"])
        for obj in parsed_objects
    }
    if want != got:
        return None

    if len({obj["dist"] for obj in parsed_objects}) != len(parsed_objects):
        return None

    placements = _simulate_objects(grid.shape, frame["bbox"], parsed_objects)
    if placements is None:
        return None
    if not _order_sensitive(grid.shape, frame, parsed_objects, BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
