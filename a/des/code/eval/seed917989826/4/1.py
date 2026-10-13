import numpy as np


SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DELTA = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _frame_side_positions(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0 + 1, c1):
        yield "top", (r0, c), (r0 - 1, c)
    for c in range(c0 + 1, c1):
        yield "bottom", (r1, c), (r1 + 1, c)
    for r in range(r0 + 1, r1):
        yield "left", (r, c0), (r, c0 - 1)
    for r in range(r0 + 1, r1):
        yield "right", (r, c1), (r, c1 + 1)


def _find_frame(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        coords = np.argwhere(grid == color)
        if coords.shape[0] < 8:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
            continue

        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        perimeter = 2 * (box_h + box_w) - 4
        if coords.shape[0] < max(10, perimeter // 2):
            continue
        if not all((int(r) in (r0, r1)) or (int(c) in (c0, c1)) for r, c in coords):
            continue

        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if any(int(grid[r, c]) != color for r, c in corners):
            continue

        holes = []
        valid = True
        for side, (r, c), outside in _frame_side_positions(r0, c0, r1, c1):
            cell = int(grid[r, c])
            if cell == color:
                continue
            if cell != bg:
                valid = False
                break
            orow, ocol = outside
            marker = int(grid[orow, ocol])
            if marker == bg or marker == color:
                valid = False
                break
            holes.append((side, (r, c), outside, marker))
        if not valid or not holes:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if not np.any((interior != bg) & (interior != color)):
            continue
        candidates.append((int(coords.shape[0]), color, (r0, c0, r1, c1), holes))

    assert candidates, "expected a rectangular frame"
    candidates.sort(reverse=True)
    _, color, bbox, holes = candidates[0]
    return color, bbox, holes


def _collect_seeds(grid: np.ndarray, bg: int, frame_color: int, bbox):
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere((interior != bg) & (interior != frame_color))
    seeds = {}
    for rr, cc in coords:
        r = int(r0 + 1 + rr)
        c = int(c0 + 1 + cc)
        color = int(grid[r, c])
        assert color not in seeds, "each seed color must appear exactly once inside the frame"
        seeds[color] = (r, c)
    assert seeds, "expected at least one seed"
    return seeds


def _trace(seed, side: str, bbox):
    sr, sc = seed
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [((r, sc), sr - r) for r in range(sr, r0 - 2, -1)]
    if side == "bottom":
        return [((r, sc), r - sr) for r in range(sr, r1 + 2)]
    if side == "left":
        return [((sr, c), sc - c) for c in range(sc, c0 - 2, -1)]
    assert side == "right"
    return [((sr, c), c - sc) for c in range(sc, c1 + 2)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow straight seed-colored branches toward frame sides chosen by same-color outside markers.

    Each border hole is paired to the unique interior seed of the same color by the
    outside marker adjacent to that hole. Every paired hole contributes its side
    direction (top/bottom/left/right); the seed then emits one straight branch for
    each owned side. Branches paint only background cells (plus their own seed cell),
    so the frame, the outside markers, and all seed pixels stay unchanged. When
    different seeds compete for one pixel, the shorter source-to-pixel distance wins;
    ties use the smaller seed row, then seed column for determinism.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    frame_color, bbox, holes = _find_frame(g, bg)
    seeds = _collect_seeds(g, bg, frame_color, bbox)

    seed_sides = {color: set() for color in seeds}
    for side, _hole, _outside, marker_color in holes:
        assert marker_color in seeds, "each outside marker color must match an interior seed"
        seed_sides[marker_color].add(side)

    out = g.copy()
    candidates = {}
    for color, seed in seeds.items():
        sr, sc = seed
        for side in sorted(seed_sides.get(color, ())):
            for (r, c), dist in _trace(seed, side, bbox):
                cell = int(g[r, c])
                if cell != bg and cell != color:
                    continue
                candidates.setdefault((r, c), []).append((dist, sr, sc, color))

    for (r, c), contest in candidates.items():
        contest.sort(key=lambda item: (item[0], item[1], item[2]))
        out[r, c] = contest[0][3]
    return out


# --------------- generator ---------------

MIN_H, MAX_H = 12, 20
MIN_W, MAX_W = 12, 20
MIN_SEEDS, MAX_SEEDS = 2, 4
MAX_ATTEMPTS = 400


def _sample_colors(rng: np.random.Generator, count: int):
    return [int(v) for v in rng.choice(np.arange(10, dtype=int), size=count, replace=False)]


def _sample_spaced(rng: np.random.Generator, low: int, high: int, k: int, min_gap: int):
    values = list(range(low, high + 1))
    for _ in range(80):
        picked = sorted(int(v) for v in rng.choice(values, size=k, replace=False))
        if all(b - a > min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return sorted(int(v) for v in rng.choice(values, size=k, replace=False))


def _random_nonempty_side_set(rng: np.random.Generator):
    count = int(rng.integers(1, 4))
    return set(str(v) for v in rng.choice(np.array(SIDES, dtype=object), size=count, replace=False))


def _aligned_slot(seed, side: str):
    r, c = seed
    return c if side in ("top", "bottom") else r


def _free_side_slots(side: str, bbox, used):
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        return [c for c in range(c0 + 1, c1) if c not in used[side]]
    return [r for r in range(r0 + 1, r1) if r not in used[side]]


def _hole_cell(side: str, slot: int, bbox):
    r0, c0, r1, c1 = bbox
    if side == "top":
        return (r0, slot)
    if side == "bottom":
        return (r1, slot)
    if side == "left":
        return (slot, c0)
    return (slot, c1)


def _outside_cell(side: str, slot: int, bbox):
    r0, c0, r1, c1 = bbox
    if side == "top":
        return (r0 - 1, slot)
    if side == "bottom":
        return (r1 + 1, slot)
    if side == "left":
        return (slot, c0 - 1)
    return (slot, c1 + 1)


def _add_hole_record(hole_records, used, bbox, side: str, slot: int, color: int, aligned: bool):
    used[side].add(slot)
    hole_records.append(
        {
            "side": side,
            "slot": int(slot),
            "color": int(color),
            "aligned": bool(aligned),
            "hole": _hole_cell(side, int(slot), bbox),
            "outside": _outside_cell(side, int(slot), bbox),
        }
    )


def _pick_slot(
    rng: np.random.Generator,
    side: str,
    bbox,
    used,
    preferred: int,
    require_aligned: bool,
):
    free = _free_side_slots(side, bbox, used)
    if not free:
        return None, None
    if require_aligned and preferred in free:
        return preferred, True
    if require_aligned:
        return None, None
    misaligned = [slot for slot in free if slot != preferred]
    if misaligned and rng.random() < 0.55:
        return int(rng.choice(misaligned)), False
    if preferred in free:
        return preferred, True
    return int(rng.choice(free)), False


def _place_cross_pair(rng: np.random.Generator, row_lo: int, row_hi: int, col_lo: int, col_hi: int):
    max_dr = row_hi - row_lo
    max_dc = col_hi - col_lo
    if max_dr < 2 or max_dc < 2:
        return None

    tie_mode = bool(rng.integers(0, 2))
    deltas = []
    for dr in range(2, max_dr + 1):
        for dc in range(2, max_dc + 1):
            if tie_mode and dr == dc:
                deltas.append((dr, dc))
            if (not tie_mode) and dr != dc:
                deltas.append((dr, dc))
    if not deltas:
        return None

    dr, dc = deltas[int(rng.integers(len(deltas)))]
    horizontal_sign = -1 if bool(rng.integers(0, 2)) else 1

    if horizontal_sign > 0:
        ra_max = row_hi - dr
        ca_min = col_lo
        ca_max = col_hi - dc
    else:
        ra_max = row_hi - dr
        ca_min = col_lo + dc
        ca_max = col_hi
    if ra_max < row_lo or ca_max < ca_min:
        return None

    ra = int(rng.integers(row_lo, ra_max + 1))
    ca = int(rng.integers(ca_min, ca_max + 1))
    rb = ra + dr
    cb = ca + horizontal_sign * dc
    return (ra, ca), (rb, cb)


def _gen_attempt(rng: np.random.Generator):
    h = int(rng.integers(MIN_H, MAX_H + 1))
    w = int(rng.integers(MIN_W, MAX_W + 1))
    seed_count = int(rng.integers(MIN_SEEDS, MAX_SEEDS + 1))

    min_frame_h = max(8, seed_count + 4)
    min_frame_w = max(8, seed_count + 4)
    if h < min_frame_h + 2 or w < min_frame_w + 2:
        return None

    frame_h = int(rng.integers(min_frame_h, h - 1))
    frame_w = int(rng.integers(min_frame_w, w - 1))
    r0 = int(rng.integers(1, h - frame_h))
    c0 = int(rng.integers(1, w - frame_w))
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    bbox = (r0, c0, r1, c1)

    core_row_lo, core_row_hi = r0 + 2, r1 - 2
    core_col_lo, core_col_hi = c0 + 2, c1 - 2
    if core_row_hi - core_row_lo + 1 < seed_count or core_col_hi - core_col_lo + 1 < seed_count:
        return None

    cross_mode = bool(rng.integers(0, 2))
    seed_positions = []
    cross_pair = None
    if cross_mode:
        cross_pair = _place_cross_pair(rng, core_row_lo, core_row_hi, core_col_lo, core_col_hi)
        if cross_pair is not None:
            seed_positions.extend(cross_pair)
    used_rows = {r for r, _ in seed_positions}
    used_cols = {c for _, c in seed_positions}

    remaining = seed_count - len(seed_positions)
    if remaining > 0:
        free_rows = [r for r in range(core_row_lo, core_row_hi + 1) if r not in used_rows]
        free_cols = [c for c in range(core_col_lo, core_col_hi + 1) if c not in used_cols]
        if len(free_rows) < remaining or len(free_cols) < remaining:
            return None
        more_rows = _sample_spaced(rng, min(free_rows), max(free_rows), remaining, min_gap=0)
        more_rows = [r for r in more_rows if r not in used_rows]
        if len(more_rows) < remaining:
            more_rows = [int(v) for v in rng.choice(np.array(free_rows, dtype=int), size=remaining, replace=False)]
        more_cols = _sample_spaced(rng, min(free_cols), max(free_cols), remaining, min_gap=0)
        more_cols = [c for c in more_cols if c not in used_cols]
        if len(more_cols) < remaining:
            more_cols = [int(v) for v in rng.choice(np.array(free_cols, dtype=int), size=remaining, replace=False)]
        rng.shuffle(more_cols)
        seed_positions.extend(list(zip(more_rows[:remaining], more_cols[:remaining])))

    if len(seed_positions) != seed_count:
        return None
    if len({r for r, _ in seed_positions}) != seed_count or len({c for _, c in seed_positions}) != seed_count:
        return None

    colors = _sample_colors(rng, seed_count + 2)
    bg = colors[0]
    frame_color = colors[1]
    seed_colors = colors[2:]

    side_sets = [_random_nonempty_side_set(rng) for _ in range(seed_count)]
    forced_aligned = set()
    if cross_pair is not None:
        a = 0
        b = 1
        ra, ca = seed_positions[a]
        rb, cb = seed_positions[b]
        vertical = "bottom" if rb > ra else "top"
        horizontal = "right" if ca > cb else "left"
        side_sets[a] = {vertical}
        side_sets[b] = {horizontal}
        forced_aligned.add((a, vertical))
        forced_aligned.add((b, horizontal))
        if rng.random() < 0.35:
            extra = {"left", "right"} if vertical in ("top", "bottom") else {"top", "bottom"}
            side_sets[a].add(str(rng.choice(np.array(sorted(extra), dtype=object))))
        if rng.random() < 0.35:
            extra = {"top", "bottom"} if horizontal in ("left", "right") else {"left", "right"}
            side_sets[b].add(str(rng.choice(np.array(sorted(extra), dtype=object))))

    used_slots = {side: set() for side in SIDES}
    hole_records = []
    misaligned_seen = False
    side_has_aligned = {}

    for idx, (seed, color, sides) in enumerate(zip(seed_positions, seed_colors, side_sets)):
        for side in sorted(sides):
            preferred = _aligned_slot(seed, side)
            want_aligned = (idx, side) in forced_aligned or rng.random() < 0.7
            slot, aligned = _pick_slot(rng, side, bbox, used_slots, preferred, want_aligned)
            if slot is None:
                return None
            _add_hole_record(hole_records, used_slots, bbox, side, slot, color, aligned)
            side_has_aligned[(idx, side)] = aligned
            misaligned_seen |= not aligned

            if rng.random() < 0.35:
                extras = [s for s in _free_side_slots(side, bbox, used_slots) if s != preferred]
                if extras:
                    extra_slot = int(rng.choice(np.array(extras, dtype=int)))
                    _add_hole_record(hole_records, used_slots, bbox, side, extra_slot, color, aligned=False)
                    misaligned_seen = True

    if not misaligned_seen:
        expandable = []
        for idx, (seed, _color, sides) in enumerate(zip(seed_positions, seed_colors, side_sets)):
            for side in sides:
                preferred = _aligned_slot(seed, side)
                extras = [s for s in _free_side_slots(side, bbox, used_slots) if s != preferred]
                if extras:
                    expandable.append((idx, side, extras))
        if expandable:
            idx, side, extras = expandable[int(rng.integers(len(expandable)))]
            extra_slot = int(rng.choice(np.array(extras, dtype=int)))
            _add_hole_record(hole_records, used_slots, bbox, side, extra_slot, seed_colors[idx], aligned=False)
            misaligned_seen = True

    grid = np.full((h, w), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    for (r, c), color in zip(seed_positions, seed_colors):
        grid[r, c] = color

    for record in hole_records:
        hr, hc = record["hole"]
        orow, ocol = record["outside"]
        grid[hr, hc] = bg
        grid[orow, ocol] = record["color"]

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if _bg_color(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
