import random
from collections import deque

import numpy as np


BG = 0
DIRECTIONS = ("up", "down", "left", "right")
DIR_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (3, 5)
RECT_HEIGHT_RANGE = (4, 8)
RECT_WIDTH_RANGE = (4, 8)
DISTANCE_POOL_MAX = 8
GEN_RETRIES = 400
PLACE_RETRIES = 250
INPUT_MARGIN = 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _make_c_shape(
    h: int, w: int, direction: str, gap_start: int, gap_end: int, depth: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = np.ones((h, w), dtype=bool)
    if direction == "up":
        mask[:depth, gap_start : gap_end + 1] = False
        cavity = (0, gap_start, depth - 1, gap_end)
    elif direction == "down":
        mask[h - depth :, gap_start : gap_end + 1] = False
        cavity = (h - depth, gap_start, h - 1, gap_end)
    elif direction == "left":
        mask[gap_start : gap_end + 1, :depth] = False
        cavity = (gap_start, 0, gap_end, depth - 1)
    else:
        mask[gap_start : gap_end + 1, w - depth :] = False
        cavity = (gap_start, w - depth, gap_end, w - 1)
    return mask, cavity


def _analyze_object(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    vals = set()
    for r, c in coords:
        local[r - r0, c - c0] = True
        vals.add(int(grid[r, c]))
    assert len(vals) == 1
    color = vals.pop()

    missing = ~local
    assert missing.any()
    miss_coords = np.argwhere(missing)
    mr0, mc0, mr1, mc1 = _bbox(miss_coords)
    cavity = missing[mr0 : mr1 + 1, mc0 : mc1 + 1]
    assert cavity.all()
    assert int(missing.sum()) == int(cavity.size)

    touches = {
        "up": mr0 == 0,
        "down": mr1 == h - 1,
        "left": mc0 == 0,
        "right": mc1 == w - 1,
    }
    assert sum(int(v) for v in touches.values()) == 1
    direction = next(name for name, flag in touches.items() if flag)

    if direction in ("up", "down"):
        assert 0 < mc0 and mc1 < w - 1
    else:
        assert 0 < mr0 and mr1 < h - 1

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "h": h,
        "w": w,
        "mask": local,
        "cavity": (mr0, mc0, mr1, mc1),
        "direction": direction,
    }


def _shifted_origin(obj: dict, steps: int) -> tuple[int, int]:
    r0, c0, _, _ = obj["bbox"]
    dr, dc = DIR_DELTA[obj["direction"]]
    return int(r0 + dr * steps), int(c0 + dc * steps)


def _solid_clear(obj: dict, r0: int, c0: int, occupied: np.ndarray) -> bool:
    h, w = obj["h"], obj["w"]
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    return not bool((occupied[r0 : r0 + h, c0 : c0 + w] & obj["mask"]).any())


def _full_bbox_clear(obj: dict, r0: int, c0: int, occupied: np.ndarray) -> bool:
    h, w = obj["h"], obj["w"]
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    return not bool(occupied[r0 : r0 + h, c0 : c0 + w].any())


def _closure_satisfied(obj: dict, r0: int, c0: int, occupied: np.ndarray) -> bool:
    h, w = obj["h"], obj["w"]
    mr0, mc0, mr1, mc1 = obj["cavity"]
    H, W = occupied.shape
    direction = obj["direction"]

    if direction == "up":
        if r0 == 0:
            return True
        row = r0 - 1
        return bool(occupied[row, c0 + mc0 : c0 + mc1 + 1].all())
    if direction == "down":
        if r0 + h == H:
            return True
        row = r0 + h
        return bool(occupied[row, c0 + mc0 : c0 + mc1 + 1].all())
    if direction == "left":
        if c0 == 0:
            return True
        col = c0 - 1
        return bool(occupied[r0 + mr0 : r0 + mr1 + 1, col].all())

    if c0 + w == W:
        return True
    col = c0 + w
    return bool(occupied[r0 + mr0 : r0 + mr1 + 1, col].all())


def _first_closure_distance(obj: dict, occupied: np.ndarray) -> int | None:
    max_steps = occupied.shape[0] + occupied.shape[1]
    for steps in range(max_steps + 1):
        r0, c0 = _shifted_origin(obj, steps)
        if not _solid_clear(obj, r0, c0, occupied):
            break
        if _closure_satisfied(obj, r0, c0, occupied) and _full_bbox_clear(obj, r0, c0, occupied):
            return steps
    return None


def _simulate(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    out = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    remaining = list(range(len(objects)))

    while remaining:
        candidates = []
        for idx in remaining:
            dist = _first_closure_distance(objects[idx], occupied)
            assert dist is not None
            r0, c0, _, _ = objects[idx]["bbox"]
            candidates.append((dist, r0, c0, idx))
        _, _, _, chosen = min(candidates)
        obj = objects[chosen]
        dist = _first_closure_distance(obj, occupied)
        assert dist is not None
        r0, c0 = _shifted_origin(obj, dist)
        assert _full_bbox_clear(obj, r0, c0, occupied)
        out[r0 : r0 + obj["h"], c0 : c0 + obj["w"]] = obj["color"]
        occupied[r0 : r0 + obj["h"], c0 : c0 + obj["w"]] = True
        remaining.remove(chosen)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each C-shaped notch object until its opening is sealed, then fill the sealed cavity.

    Input structure:
    - The background is black.
    - Each foreground object is a single-color 4-connected rectangle with one rectangular notch cut from exactly one side.
    - Different objects use different colors and are separated in the input, while their openings may face up, down, left, or right.

    Rule:
    - For each object, compare its occupied pixels with its bounding box to locate the rectangular missing notch and therefore the opening direction.
    - Rebuild the output on an empty canvas: repeatedly pick the remaining object whose opening can be sealed after the shortest translation along that direction, using only the grid boundary and already placed objects as blockers.
    - Move that object to its sealing position and fill the entire bounding box with the object's own color, because the sealed notch becomes an enclosed rectangular cavity.

    Key insight:
    - The same missing rectangle inside each bounding box determines all three steps: which way the object moves, when it must stop, and which area becomes filled after stopping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components(g != BG)
    objects = [_analyze_object(g, comp) for comp in comps]
    assert objects
    return _simulate(objects, tuple(map(int, g.shape)))


def _sample_directions(rng: random.Random, count: int) -> list[str]:
    dirs = list(DIRECTIONS)
    rng.shuffle(dirs)
    out = dirs[: min(count, len(dirs))]
    while len(out) < count:
        extra = list(DIRECTIONS)
        rng.shuffle(extra)
        out.extend(extra)
    return out[:count]


def _sample_closure_types(rng: random.Random, count: int) -> list[str]:
    if count == 1:
        return ["boundary"]
    later = ["obstacle"] * ((count - 1 + 1) // 2) + ["boundary"] * ((count - 1) // 2)
    later = later[: count - 1]
    rng.shuffle(later)
    return ["boundary"] + later


def _choose_size(rng: random.Random, direction: str) -> tuple[int, int]:
    h = rng.randint(*RECT_HEIGHT_RANGE)
    w = rng.randint(*RECT_WIDTH_RANGE)
    if direction in ("up", "down"):
        w = max(w, 4)
    else:
        h = max(h, 4)
    return h, w


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _ranges_overlap(ar0, ar1, br0, br1) and _ranges_overlap(ac0, ac1, bc0, bc1)


def _rect_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 + margin < br0 or br1 + margin < ar0:
        return False
    if ac1 + margin < bc0 or bc1 + margin < ac0:
        return False
    return True


def _boundary_final_bbox(
    rng: random.Random, H: int, W: int, h: int, w: int, dist: int, direction: str
) -> tuple[int, int] | None:
    if direction == "up":
        if h + dist > H:
            return None
        return 0, rng.randint(0, W - w)
    if direction == "down":
        if h + dist > H:
            return None
        return H - h, rng.randint(0, W - w)
    if direction == "left":
        if w + dist > W:
            return None
        return rng.randint(0, H - h), 0
    if w + dist > W:
        return None
    return rng.randint(0, H - h), W - w


def _obstacle_final_bbox(
    rng: random.Random,
    blocker: dict,
    H: int,
    W: int,
    h: int,
    w: int,
    dist: int,
    direction: str,
) -> tuple[int, int] | None:
    br0, bc0, br1, bc1 = blocker["bbox"]
    if direction == "up":
        fr = br1 + 1
        lo = max(0, bc0 - w + 2)
        hi = min(W - w, bc1 - 1)
        if fr + h + dist > H or lo > hi:
            return None
        return fr, rng.randint(lo, hi)
    if direction == "down":
        fr = br0 - h
        lo = max(0, bc0 - w + 2)
        hi = min(W - w, bc1 - 1)
        if fr - dist < 0 or lo > hi or fr < 0:
            return None
        return fr, rng.randint(lo, hi)
    if direction == "left":
        fc = bc1 + 1
        lo = max(0, br0 - h + 2)
        hi = min(H - h, br1 - 1)
        if fc + w + dist > W or lo > hi:
            return None
        return rng.randint(lo, hi), fc
    fc = bc0 - w
    lo = max(0, br0 - h + 2)
    hi = min(H - h, br1 - 1)
    if fc - dist < 0 or lo > hi or fc < 0:
        return None
    return rng.randint(lo, hi), fc


def _support_runs(
    direction: str, r0: int, c0: int, h: int, w: int, occupied: np.ndarray
) -> list[tuple[int, int]]:
    H, W = occupied.shape
    if direction == "up":
        flags = np.ones(w, dtype=bool) if r0 == 0 else occupied[r0 - 1, c0 : c0 + w]
    elif direction == "down":
        flags = np.ones(w, dtype=bool) if r0 + h == H else occupied[r0 + h, c0 : c0 + w]
    elif direction == "left":
        flags = np.ones(h, dtype=bool) if c0 == 0 else occupied[r0 : r0 + h, c0 - 1]
    else:
        flags = np.ones(h, dtype=bool) if c0 + w == W else occupied[r0 : r0 + h, c0 + w]

    runs = []
    start = None
    for idx in range(1, len(flags) - 1):
        if bool(flags[idx]):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 2))
    return runs


def _original_origin(final_r: int, final_c: int, dist: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return final_r + dist, final_c
    if direction == "down":
        return final_r - dist, final_c
    if direction == "left":
        return final_r, final_c + dist
    return final_r, final_c - dist


def _candidate_object(
    color: int,
    direction: str,
    dist: int,
    final_r: int,
    final_c: int,
    h: int,
    w: int,
    gap_start: int,
    gap_end: int,
    depth: int,
) -> dict:
    mask, cavity = _make_c_shape(h, w, direction, gap_start, gap_end, depth)
    orig_r, orig_c = _original_origin(final_r, final_c, dist, direction)
    return {
        "color": color,
        "bbox": (orig_r, orig_c, orig_r + h - 1, orig_c + w - 1),
        "h": h,
        "w": w,
        "mask": mask,
        "cavity": cavity,
        "direction": direction,
        "final_bbox": (final_r, final_c, final_r + h - 1, final_c + w - 1),
        "dist": dist,
    }


def _path_is_valid(obj: dict, occupied: np.ndarray) -> bool:
    dist = obj["dist"]
    for step in range(dist + 1):
        r0, c0 = _shifted_origin(obj, step)
        if not _full_bbox_clear(obj, r0, c0, occupied):
            return False
        closed = _closure_satisfied(obj, r0, c0, occupied)
        if step < dist and closed:
            return False
        if step == dist and not closed:
            return False
    return True


def _pick_gap_and_depth(
    rng: random.Random,
    direction: str,
    runs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, int, int] | None:
    if not runs:
        return None
    run0, run1 = rng.choice(runs)
    gap_start = rng.randint(run0, run1)
    gap_end = rng.randint(gap_start, run1)
    max_depth = h - 1 if direction in ("up", "down") else w - 1
    depth = rng.randint(1, max_depth)
    return gap_start, gap_end, depth


def _place_one(
    rng: random.Random,
    H: int,
    W: int,
    color: int,
    direction: str,
    closure_type: str,
    dist: int,
    final_rects: list[dict],
    input_bboxes: list[tuple[int, int, int, int]],
    final_occ: np.ndarray,
) -> dict | None:
    for _ in range(PLACE_RETRIES):
        h, w = _choose_size(rng, direction)
        if closure_type == "boundary":
            pos = _boundary_final_bbox(rng, H, W, h, w, dist, direction)
        else:
            if not final_rects:
                continue
            blocker = rng.choice(final_rects)
            pos = _obstacle_final_bbox(rng, blocker, H, W, h, w, dist, direction)
        if pos is None:
            continue

        final_r, final_c = pos
        final_bbox = (final_r, final_c, final_r + h - 1, final_c + w - 1)
        if any(_rect_overlap(final_bbox, prev["bbox"]) for prev in final_rects):
            continue

        runs = _support_runs(direction, final_r, final_c, h, w, final_occ)
        gap = _pick_gap_and_depth(rng, direction, runs, h, w)
        if gap is None:
            continue
        gap_start, gap_end, depth = gap
        obj = _candidate_object(
            color, direction, dist, final_r, final_c, h, w, gap_start, gap_end, depth
        )

        orig_bbox = obj["bbox"]
        if any(_rect_too_close(orig_bbox, prev, INPUT_MARGIN) for prev in input_bboxes):
            continue
        if not _path_is_valid(obj, final_occ):
            continue
        return obj
    return None


def _build_input(H: int, W: int, objects: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for obj in objects:
        r0, c0, _, _ = obj["bbox"]
        h, w = obj["h"], obj["w"]
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        assert not bool((sub != BG).any())
        sub[obj["mask"]] = obj["color"]
    return grid


def _build_expected(H: int, W: int, objects: list[dict]) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["final_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = obj["color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(GRID_MIN, GRID_MAX)
    W = rng.randint(GRID_MIN, GRID_MAX)
    count = rng.randint(*OBJECT_COUNT_RANGE)
    if count > DISTANCE_POOL_MAX:
        return None

    distances = sorted(rng.sample(range(1, DISTANCE_POOL_MAX + 1), count))
    directions = _sample_directions(rng, count)
    closure_types = _sample_closure_types(rng, count)
    colors = rng.sample(list(range(1, 10)), count)

    objects = []
    final_rects = []
    input_bboxes = []
    final_occ = np.zeros((H, W), dtype=bool)

    for i in range(count):
        obj = _place_one(
            rng,
            H,
            W,
            colors[i],
            directions[i],
            closure_types[i],
            distances[i],
            final_rects,
            input_bboxes,
            final_occ,
        )
        if obj is None:
            return None
        objects.append(obj)
        input_bboxes.append(obj["bbox"])
        fr0, fc0, fr1, fc1 = obj["final_bbox"]
        final_rects.append({"bbox": (fr0, fc0, fr1, fc1), "color": obj["color"]})
        final_occ[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    grid = _build_input(H, W, objects)
    expected = _build_expected(H, W, objects)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
