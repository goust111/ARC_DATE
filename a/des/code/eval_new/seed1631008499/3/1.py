import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = tuple(range(10))
SPECIAL_POOL = tuple(c for c in COLOR_POOL if c not in {BG, FRAME})

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
PAIR_COUNT_RANGE = (2, 4)
KEY_LEN_RANGE = (2, 4)
LOCK_INNER_H_RANGE = (2, 6)
LOCK_INNER_W_RANGE = (2, 7)
PLACEMENT_TRIES = 200
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    perim: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        perim.add((r0, c))
        perim.add((r1, c))
    for r in range(r0, r1 + 1):
        perim.add((r, c0))
        perim.add((r, c1))
    return perim


def _gap_neighbors(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0:
        return [(gr, gc - 1), (gr, gc + 1)]
    if gr == r1:
        return [(gr, gc - 1), (gr, gc + 1)]
    if gc == c0:
        return [(gr - 1, gc), (gr + 1, gc)]
    if gc == c1:
        return [(gr - 1, gc), (gr + 1, gc)]
    return []


def _parse_key(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    if not (2 <= len(coords) <= 4):
        return None

    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        ordered = sorted(coords, key=lambda p: p[1])
        if ordered[-1][1] - ordered[0][1] + 1 != len(ordered):
            return None
        axis = 1
    elif len(cols) == 1:
        ordered = sorted(coords, key=lambda p: p[0])
        if ordered[-1][0] - ordered[0][0] + 1 != len(ordered):
            return None
        axis = 0
    else:
        return None

    colors = [int(grid[r, c]) for r, c in ordered]
    if len(set(colors)) != len(colors):
        return None
    return {"coords": ordered, "colors": colors, "axis": axis}


def _parse_lock(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    coord_set = set(coords)
    perim = _perimeter_coords(bbox)
    if not coord_set.issubset(perim):
        return None
    if len(coord_set) != len(perim) - 1:
        return None

    missing = list(perim - coord_set)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    counts = Counter(int(grid[r, c]) for r, c in coords)
    frame_color, frame_count = counts.most_common(1)[0]
    if frame_count != len(coords) - 1:
        return None

    specials = [(r, c) for r, c in coords if int(grid[r, c]) != frame_color]
    if len(specials) != 1:
        return None

    special = specials[0]
    if special not in _gap_neighbors(bbox, gap):
        return None

    return {
        "bbox": bbox,
        "gap": gap,
        "special": special,
        "match_color": int(grid[special]),
        "frame_color": int(frame_color),
    }


def _sequence_from_match(key: dict, match_color: int) -> list[int]:
    colors = key["colors"]
    if colors[0] == match_color:
        return colors[:]
    if colors[-1] == match_color:
        return list(reversed(colors))
    raise AssertionError("matching color must appear at one key endpoint")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each color key to the rectangular lock with the same special border pixel and fill the lock interior row by row.

    Input structure:
    - The background is black, with several hollow axis-aligned rectangular lock frames on the grid.
    - Each lock is mostly grey, has a one-cell gap on one side, and has one adjacent border pixel replaced by a special non-grey color.
    - Separate short horizontal or vertical key bars contain 2-4 distinct colors, with exactly one endpoint color matching one lock's special border pixel.

    Rule:
    - Detect each lock by its nearly complete rectangular border and record its unique special border color.
    - Detect each straight key bar and pair it with the unique lock whose special color appears on that key.
    - Read the key's ordered color sequence starting at the matching endpoint, then fill the matched lock interior from top row to bottom row, cycling through the sequence if needed.
    - Remove the key bars from the output by restoring those cells to the background.

    Key insight:
    - The colored border pixel beside a lock's gap identifies which key belongs to that lock, and the matching endpoint of the key fixes the reading direction of the fill sequence.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    locks: dict[int, dict] = {}
    keys: list[dict] = []
    for comp in _components(g != bg):
        key = _parse_key(g, comp)
        if key is not None:
            keys.append(key)
            continue
        lock = _parse_lock(g, comp)
        if lock is not None:
            locks[lock["match_color"]] = lock

    assert locks and keys, "expected at least one lock and one key"
    assert len(locks) == len(keys), "expected one key per lock"

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = bg

        matched = [color for color in key["colors"] if color in locks]
        assert len(matched) == 1, "key must match exactly one lock color"
        match_color = matched[0]
        seq = _sequence_from_match(key, match_color)

        r0, c0, r1, c1 = locks[match_color]["bbox"]
        for idx, r in enumerate(range(r0 + 1, r1)):
            out[r, c0 + 1 : c1] = seq[idx % len(seq)]

    return out


def _sequence_physical_order(sequence: list[int], match_at_low_end: bool) -> list[int]:
    return sequence[:] if match_at_low_end else list(reversed(sequence))


def _expanded_slices(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[slice, slice]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return slice(rr0, rr1 + 1), slice(cc0, cc1 + 1)


def _bbox_is_free(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    rs, cs = _expanded_slices(bbox, blocked.shape, pad=1)
    return not blocked[rs, cs].any()


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    rs, cs = _expanded_slices(bbox, blocked.shape, pad=1)
    blocked[rs, cs] = True


def _place_bbox(
    blocked: np.ndarray, obj_h: int, obj_w: int, rng: random.Random
) -> tuple[int, int, int, int] | None:
    h, w = blocked.shape
    if obj_h > h or obj_w > w:
        return None
    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randint(0, h - obj_h)
        c0 = rng.randint(0, w - obj_w)
        bbox = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
        if _bbox_is_free(blocked, bbox):
            return bbox
    return None


def _draw_lock(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    match_color: int,
    gap_side: str,
    favor_first_neighbor: bool,
    rng: random.Random,
) -> dict:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    if gap_side == "top":
        gc = rng.randint(c0 + 1, c1 - 1)
        gap = (r0, gc)
    elif gap_side == "bottom":
        gc = rng.randint(c0 + 1, c1 - 1)
        gap = (r1, gc)
    elif gap_side == "left":
        gr = rng.randint(r0 + 1, r1 - 1)
        gap = (gr, c0)
    else:
        gr = rng.randint(r0 + 1, r1 - 1)
        gap = (gr, c1)

    neighbors = _gap_neighbors(bbox, gap)
    special = neighbors[0] if favor_first_neighbor else neighbors[1]
    grid[gap] = BG
    grid[special] = match_color
    return {"bbox": bbox, "gap": gap, "special": special, "match_color": match_color}


def _draw_key(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    orientation: str,
    physical_colors: list[int],
    sequence: list[int],
) -> dict:
    r0, c0, r1, c1 = bbox
    coords: list[tuple[int, int]] = []
    if orientation == "h":
        for offset, color in enumerate(physical_colors):
            coord = (r0, c0 + offset)
            coords.append(coord)
            grid[coord] = color
    else:
        for offset, color in enumerate(physical_colors):
            coord = (r0 + offset, c0)
            coords.append(coord)
            grid[coord] = color
    return {"coords": coords, "sequence": sequence[:]}


def _expected_from_specs(
    grid: np.ndarray, locks: list[dict], keys: list[dict]
) -> np.ndarray:
    out = grid.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = BG
    for lock, key in zip(locks, keys):
        r0, c0, r1, c1 = lock["bbox"]
        seq = key["sequence"]
        for idx, r in enumerate(range(r0 + 1, r1)):
            out[r, c0 + 1 : c1] = seq[idx % len(seq)]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    if pair_count > len(SPECIAL_POOL):
        return None

    special_colors = rng.sample(list(SPECIAL_POOL), pair_count)
    other_pool = [c for c in SPECIAL_POOL if c not in special_colors]
    if len(other_pool) < KEY_LEN_RANGE[1] - 1:
        return None

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    locks: list[dict] = []
    keys: list[dict] = []

    lock_specs = []
    for match_color in special_colors:
        key_len = rng.randint(*KEY_LEN_RANGE)
        wrap = bool(rng.randint(0, 1))
        if wrap:
            inner_h = rng.randint(key_len + 1, min(LOCK_INNER_H_RANGE[1], key_len + 2))
        else:
            inner_h = rng.randint(LOCK_INNER_H_RANGE[0], min(key_len, LOCK_INNER_H_RANGE[1]))
        inner_w = rng.randint(*LOCK_INNER_W_RANGE)
        lock_specs.append(
            {
                "match_color": match_color,
                "key_len": key_len,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "gap_side": rng.choice(["top", "bottom", "left", "right"]),
                "special_side": bool(rng.randint(0, 1)),
                "orientation": rng.choice(["h", "v"]),
                "match_at_low_end": bool(rng.randint(0, 1)),
            }
        )

    lock_specs.sort(key=lambda spec: (spec["inner_h"] + 2) * (spec["inner_w"] + 2), reverse=True)
    for spec in lock_specs:
        bbox = _place_bbox(blocked, spec["inner_h"] + 2, spec["inner_w"] + 2, rng)
        if bbox is None:
            return None
        lock = _draw_lock(
            grid, bbox, spec["match_color"], spec["gap_side"], spec["special_side"], rng
        )
        lock["sequence_len"] = spec["key_len"]
        lock["orientation"] = spec["orientation"]
        lock["match_at_low_end"] = spec["match_at_low_end"]
        _mark_blocked(blocked, bbox)
        locks.append(lock)

    rng.shuffle(locks)
    for lock in locks:
        key_len = lock["sequence_len"]
        extras = rng.sample(other_pool, key_len - 1)
        sequence = [lock["match_color"], *extras]
        physical_colors = _sequence_physical_order(sequence, lock["match_at_low_end"])
        if lock["orientation"] == "h":
            bbox = _place_bbox(blocked, 1, key_len, rng)
        else:
            bbox = _place_bbox(blocked, key_len, 1, rng)
        if bbox is None:
            return None
        key = _draw_key(grid, bbox, lock["orientation"], physical_colors, sequence)
        key["match_color"] = lock["match_color"]
        _mark_blocked(blocked, bbox)
        keys.append(key)

    expected = _expected_from_specs(grid, locks, keys)
    solved = main(grid)
    if not np.array_equal(expected, solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key ARC grid")
