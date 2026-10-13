import numpy as np


BG = 0
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
SIDE_TO_DIR = {
    "top": "S",
    "bottom": "N",
    "left": "E",
    "right": "W",
}
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_GEN_ATTEMPTS = 300


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    comps.sort(key=lambda obj: obj["bbox"])
    return comps


def _lock_relative_coords(frame_h: int, frame_w: int, side: str) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]:
    rel: set[tuple[int, int]] = set()
    if side == "top":
        frame_r0, frame_c0 = 2, 0
        outer_h, outer_w = frame_h + 2, frame_w
    elif side == "bottom":
        frame_r0, frame_c0 = 0, 0
        outer_h, outer_w = frame_h + 2, frame_w
    elif side == "left":
        frame_r0, frame_c0 = 0, 2
        outer_h, outer_w = frame_h, frame_w + 2
    else:
        frame_r0, frame_c0 = 0, 0
        outer_h, outer_w = frame_h, frame_w + 2

    for r in range(frame_h):
        for c in range(frame_w):
            if r in (0, frame_h - 1) or c in (0, frame_w - 1):
                rel.add((frame_r0 + r, frame_c0 + c))

    if side in ("top", "bottom"):
        mid_c = frame_w // 2
        gap = (frame_r0 + (0 if side == "top" else frame_h - 1), frame_c0 + mid_c)
        rel.remove(gap)
        groove_rows = [0, 1] if side == "top" else [frame_h, frame_h + 1]
        for rr in groove_rows:
            rel.add((rr, frame_c0 + mid_c - 1))
            rel.add((rr, frame_c0 + mid_c + 1))
    else:
        mid_r = frame_h // 2
        gap = (frame_r0 + mid_r, frame_c0 + (0 if side == "left" else frame_w - 1))
        rel.remove(gap)
        groove_cols = [0, 1] if side == "left" else [frame_w, frame_w + 1]
        for cc in groove_cols:
            rel.add((frame_r0 + mid_r - 1, cc))
            rel.add((frame_r0 + mid_r + 1, cc))

    return rel, gap, (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)


def _parse_lock(comp: dict) -> dict | None:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    rel = {(r - r0, c - c0) for r, c in coords}
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    candidates: list[tuple[str, int, int]] = []
    if h >= 5 and w >= 5 and w % 2 == 1:
        candidates.append(("top", h - 2, w))
        candidates.append(("bottom", h - 2, w))
    if h >= 5 and w >= 5 and h % 2 == 1:
        candidates.append(("left", h, w - 2))
        candidates.append(("right", h, w - 2))

    for side, frame_h, frame_w in candidates:
        if frame_h < 4 or frame_w < 5:
            continue
        expected_rel, gap_rel, frame_bbox_rel = _lock_relative_coords(frame_h, frame_w, side)
        if rel != expected_rel:
            continue
        fr0, fc0, fr1, fc1 = frame_bbox_rel
        gap = (r0 + gap_rel[0], c0 + gap_rel[1])
        frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
        ir0, ic0, ir1, ic1 = frame_bbox[0] + 1, frame_bbox[1] + 1, frame_bbox[2] - 1, frame_bbox[3] - 1
        return {
            "kind": "lock",
            "color": int(comp["color"]),
            "coords": list(coords),
            "bbox": comp["bbox"],
            "side": side,
            "gap": gap,
            "frame_bbox": frame_bbox,
            "interior_bbox": (ir0, ic0, ir1, ic1),
        }
    return None


def _parse_key(comp: dict) -> dict | None:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    rel = {(r - r0, c - c0) for r, c in coords}
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if h == 2 and w >= 3:
        for handle_row in (0, 1):
            if not all((handle_row, c) in rel for c in range(w)):
                continue
            tooth_row = 1 - handle_row
            teeth_cols = [c for c in range(w) if (tooth_row, c) in rel]
            if not teeth_cols or len(rel) != w + len(teeth_cols):
                continue
            teeth = len(teeth_cols)
            east = [2 * i for i in range(teeth)]
            west_start = w - 1 - 2 * (teeth - 1)
            west = [west_start + 2 * i for i in range(teeth)] if west_start >= 0 else []
            if teeth_cols == east:
                return {
                    "kind": "key",
                    "color": int(comp["color"]),
                    "coords": list(coords),
                    "bbox": comp["bbox"],
                    "direction": "E",
                    "teeth": teeth,
                    "tip": (r0 + handle_row, c1),
                }
            if teeth_cols == west:
                return {
                    "kind": "key",
                    "color": int(comp["color"]),
                    "coords": list(coords),
                    "bbox": comp["bbox"],
                    "direction": "W",
                    "teeth": teeth,
                    "tip": (r0 + handle_row, c0),
                }

    if w == 2 and h >= 3:
        for handle_col in (0, 1):
            if not all((r, handle_col) in rel for r in range(h)):
                continue
            tooth_col = 1 - handle_col
            teeth_rows = [r for r in range(h) if (r, tooth_col) in rel]
            if not teeth_rows or len(rel) != h + len(teeth_rows):
                continue
            teeth = len(teeth_rows)
            south = [2 * i for i in range(teeth)]
            north_start = h - 1 - 2 * (teeth - 1)
            north = [north_start + 2 * i for i in range(teeth)] if north_start >= 0 else []
            if teeth_rows == south:
                return {
                    "kind": "key",
                    "color": int(comp["color"]),
                    "coords": list(coords),
                    "bbox": comp["bbox"],
                    "direction": "S",
                    "teeth": teeth,
                    "tip": (r1, c0 + handle_col),
                }
            if teeth_rows == north:
                return {
                    "kind": "key",
                    "color": int(comp["color"]),
                    "coords": list(coords),
                    "bbox": comp["bbox"],
                    "direction": "N",
                    "teeth": teeth,
                    "tip": (r0, c0 + handle_col),
                }

    return None


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _bg_color(grid)
    locks: list[dict] = []
    keys: list[dict] = []
    for comp in _components(grid, bg):
        lock = _parse_lock(comp)
        if lock is not None:
            locks.append(lock)
            continue
        key = _parse_key(comp)
        if key is not None:
            keys.append(key)
    locks.sort(key=lambda obj: obj["bbox"])
    keys.sort(key=lambda obj: obj["bbox"])
    return bg, locks, keys


def _forward_distance(start: tuple[int, int], end: tuple[int, int], direction: str) -> int | None:
    sr, sc = start
    er, ec = end
    if direction == "E" and sr == er and ec > sc:
        return ec - sc
    if direction == "W" and sr == er and ec < sc:
        return sc - ec
    if direction == "S" and sc == ec and er > sr:
        return er - sr
    if direction == "N" and sc == ec and er < sr:
        return sr - er
    return None


def _path_clear(grid: np.ndarray, bg: int, start: tuple[int, int], direction: str, dist: int) -> bool:
    dr, dc = DIRS[direction]
    r, c = start
    for step in range(1, dist + 1):
        rr = r + dr * step
        cc = c + dc * step
        if int(grid[rr, cc]) != bg:
            return False
    return True


def _pair_objects(grid: np.ndarray, bg: int, locks: list[dict], keys: list[dict]) -> list[tuple[dict, dict, int]] | None:
    pairs: list[tuple[dict, dict, int]] = []
    used_locks: set[int] = set()
    for key in keys:
        compatible_side = None
        for side, direction in SIDE_TO_DIR.items():
            if direction == key["direction"]:
                compatible_side = side
                break
        candidates: list[tuple[int, int]] = []
        for idx, lock in enumerate(locks):
            if idx in used_locks or lock["side"] != compatible_side:
                continue
            dist = _forward_distance(key["tip"], lock["gap"], key["direction"])
            if dist is None:
                continue
            if not _path_clear(grid, bg, key["tip"], key["direction"], dist):
                continue
            candidates.append((dist, idx))
        if not candidates:
            return None
        candidates.sort()
        min_dist = candidates[0][0]
        nearest = [idx for dist, idx in candidates if dist == min_dist]
        if len(nearest) != 1:
            return None
        lock_idx = nearest[0]
        if lock_idx in used_locks:
            return None
        used_locks.add(lock_idx)
        pairs.append((key, locks[lock_idx], min_dist))
    if len(pairs) != len(keys):
        return None
    return pairs


def _fill_lock(out: np.ndarray, lock: dict, fill_color: int, teeth: int, bg: int) -> None:
    ir0, ic0, ir1, ic1 = lock["interior_bbox"]
    if ir0 > ir1 or ic0 > ic1:
        return
    height = ir1 - ir0 + 1
    slots = 2 * teeth - 1
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg
    if height % slots == 0:
        band = height // slots
        for i in range(teeth):
            r0 = ir0 + 2 * i * band
            out[r0 : r0 + band, ic0 : ic1 + 1] = fill_color
    else:
        row = ir0
        for idx in range(slots):
            if row > ir1:
                break
            if idx % 2 == 0:
                out[row, ic0 : ic1 + 1] = fill_color
            row += 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, locks, keys = _parse_scene(g)
    pairs = _pair_objects(g, bg, locks, keys)
    if not locks or not keys or pairs is None:
        return g.copy()

    out = g.copy()
    for key, lock, dist in pairs:
        for r, c in key["coords"]:
            out[r, c] = bg
        dr, dc = DIRS[key["direction"]]
        for r, c in key["coords"]:
            out[r + dr * dist, c + dc * dist] = int(key["color"])
        _fill_lock(out, lock, int(key["color"]), int(key["teeth"]), bg)
    return out


def _lock_bitmap(frame_h: int, frame_w: int, side: str, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    rel, gap, _frame_bbox = _lock_relative_coords(frame_h, frame_w, side)
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    arr = np.zeros((h, w), dtype=int)
    for r, c in rel:
        arr[r, c] = color
    return arr, gap


def _key_bitmap(length: int, teeth: int, direction: str, tooth_side: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    if direction in ("E", "W"):
        handle_row = 1 if tooth_side < 0 else 0
        tooth_row = 1 - handle_row
        arr = np.zeros((2, length), dtype=int)
        arr[handle_row, :] = color
        if direction == "E":
            cols = [2 * i for i in range(teeth)]
            tip = (handle_row, length - 1)
        else:
            cols = [length - 1 - 2 * i for i in range(teeth)]
            tip = (handle_row, 0)
        arr[tooth_row, cols] = color
        return arr, tip

    handle_col = 1 if tooth_side < 0 else 0
    tooth_col = 1 - handle_col
    arr = np.zeros((length, 2), dtype=int)
    arr[:, handle_col] = color
    if direction == "S":
        rows = [2 * i for i in range(teeth)]
        tip = (length - 1, handle_col)
    else:
        rows = [length - 1 - 2 * i for i in range(teeth)]
        tip = (0, handle_col)
    arr[rows, tooth_col] = color
    return arr, tip


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    sh, sw = src.shape
    region = dst[top : top + sh, left : left + sw]
    mask = src != BG
    region[mask] = src[mask]


def _build_pair_cluster(
    rng: np.random.Generator, lock_color: int, key_color: int, teeth: int
) -> tuple[np.ndarray, dict]:
    side = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    direction = SIDE_TO_DIR[side]
    divisible = True if teeth == 1 else bool(rng.integers(0, 2))
    slots = 2 * teeth - 1

    if side in ("left", "right"):
        if divisible:
            q = int(rng.choice(np.array([1, 3], dtype=int)))
            interior_h = slots * q
        else:
            interior_h = slots + int(rng.choice(np.array([2, 4], dtype=int)))
        frame_h = interior_h + 2
        frame_w = int(rng.integers(5, 10))
    else:
        if divisible:
            interior_h = slots * int(rng.integers(1, 4))
        else:
            interior_h = slots + int(rng.integers(1, 5))
        frame_h = interior_h + 2
        frame_w = int(rng.choice(np.array([5, 7, 9, 11], dtype=int)))

    lock_arr, gap = _lock_bitmap(frame_h, frame_w, side, lock_color)
    tail = int(rng.integers(1, 5))
    key_length = 2 * teeth + tail
    tooth_side = -1 if bool(rng.integers(0, 2)) else 1
    key_arr, key_tip = _key_bitmap(key_length, teeth, direction, tooth_side, key_color)
    gap_dist = int(rng.integers(3, 9))
    dr, dc = DIRS[direction]
    target_tip = np.array(gap, dtype=int)
    source_tip = target_tip - gap_dist * np.array([dr, dc], dtype=int)
    key_origin = source_tip - np.array(key_tip, dtype=int)

    lock_cells = np.argwhere(lock_arr != BG)
    key_cells = np.argwhere(key_arr != BG) + key_origin
    all_cells = np.vstack([lock_cells, key_cells])
    min_r = int(all_cells[:, 0].min())
    min_c = int(all_cells[:, 1].min())
    max_r = int(all_cells[:, 0].max())
    max_c = int(all_cells[:, 1].max())

    shift = np.array([-min_r, -min_c], dtype=int)
    cluster = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    _paste_nonzero(cluster, lock_arr, int(shift[0]), int(shift[1]))
    _paste_nonzero(cluster, key_arr, int(key_origin[0] + shift[0]), int(key_origin[1] + shift[1]))

    meta = {
        "side": side,
        "direction": direction,
        "teeth": teeth,
        "divisible": divisible,
    }
    return cluster, meta


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, margin: int) -> bool:
    rr0 = max(0, top - margin)
    cc0 = max(0, left - margin)
    rr1 = min(reserved.shape[0], top + h + margin)
    cc1 = min(reserved.shape[1], left + w + margin)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _place_clusters(rng: np.random.Generator, clusters: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(arr.shape[0] for arr in clusters)
    max_w = max(arr.shape[1] for arr in clusters)
    h_low = max(24, max_h + 4)
    w_low = max(24, max_w + 4)
    h = int(rng.integers(h_low, h_low + 13))
    w = int(rng.integers(w_low, w_low + 13))
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    margin = 2

    order = list(rng.permutation(len(clusters)))
    for idx in order:
        arr = clusters[idx]
        ah, aw = arr.shape
        placed = False
        for _ in range(300):
            top = int(rng.integers(0, h - ah + 1))
            left = int(rng.integers(0, w - aw + 1))
            if not _can_place(reserved, top, left, ah, aw, margin):
                continue
            _paste_nonzero(grid, arr, top, left)
            rr0 = max(0, top - margin)
            cc0 = max(0, left - margin)
            rr1 = min(h, top + ah + margin)
            cc1 = min(w, left + aw + margin)
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _scene_valid(grid: np.ndarray, expected_pairs: int) -> bool:
    bg, locks, keys = _parse_scene(grid)
    if bg != BG:
        return False
    if len(locks) != expected_pairs or len(keys) != expected_pairs:
        return False
    if len({int(key["teeth"]) for key in keys}) != expected_pairs:
        return False
    pairs = _pair_objects(grid, bg, locks, keys)
    if pairs is None or len(pairs) != expected_pairs:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(2, 5))
    teeth_counts = list(rng.choice(np.array([1, 2, 3, 4], dtype=int), size=pair_count, replace=False))
    rng.shuffle(teeth_counts)
    colors = list(rng.choice(COLOR_POOL, size=2 * pair_count, replace=False))
    clusters: list[np.ndarray] = []
    for i, teeth in enumerate(teeth_counts):
        cluster, _meta = _build_pair_cluster(rng, int(colors[2 * i]), int(colors[2 * i + 1]), int(teeth))
        clusters.append(cluster)
    grid = _place_clusters(rng, clusters)
    if grid is None:
        return None
    if not _scene_valid(grid, pair_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
