import numpy as np
from collections import deque


BLACK = 0
GREY = 5
AZURE = 8
MAROON = 9
SEGMENT_COLORS = (1, 2, 3, 4, 6, 7)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 500
_RNG = np.random.default_rng()


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample(seq, k: int):
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [seq[int(i)] for i in idx]


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _normalize_coords(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=int)
    origin = coords.min(axis=0)
    return coords - origin, origin


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BLACK)
    if pts.size == 0:
        return np.zeros((1, 1), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, pt)) for pt in np.asarray(coords, dtype=int)}


def _choose_anchor_point(anchor_pixels: np.ndarray, travel_dir: tuple[int, int]) -> np.ndarray:
    pts = [tuple(map(int, pt)) for pt in np.asarray(anchor_pixels, dtype=int)]
    dr, dc = travel_dir
    if dr == 0 and dc == 1:
        best = max(pts, key=lambda rc: (rc[1], -rc[0]))
    elif dr == 0 and dc == -1:
        best = min(pts, key=lambda rc: (rc[1], rc[0]))
    elif dr == 1 and dc == 0:
        best = max(pts, key=lambda rc: (rc[0], -rc[1]))
    else:
        best = min(pts, key=lambda rc: (rc[0], rc[1]))
    return np.array(best, dtype=int)


def _parse_segment(grid: np.ndarray, color: int, comp: np.ndarray) -> dict:
    h, w = grid.shape
    pts = _coord_set(comp)
    endpoints: list[tuple[int, int]] = []
    for r, c in pts:
        deg = 0
        for dr, dc in DIRS:
            if (r + dr, c + dc) in pts:
                deg += 1
        assert deg in (1, 2), "segment must be a simple path"
        if deg == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "segment must have exactly two endpoints"

    entry_pt = None
    exit_pt = None
    entry_dir = None
    exit_dir = None
    for r, c in endpoints:
        maroon_dirs = []
        grey_dirs = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            val = int(grid[nr, nc])
            if val == MAROON:
                maroon_dirs.append((dr, dc))
            elif val == GREY:
                grey_dirs.append((dr, dc))
        if len(maroon_dirs) == 1 and not grey_dirs:
            entry_pt = np.array((r, c), dtype=int)
            entry_dir = maroon_dirs[0]
        elif len(grey_dirs) == 1 and not maroon_dirs:
            exit_pt = np.array((r, c), dtype=int)
            exit_dir = grey_dirs[0]
    assert entry_pt is not None and exit_pt is not None, "each segment needs one maroon and one grey endpoint marker"

    pixels_local, origin = _normalize_coords(comp)
    return {
        "color": int(color),
        "pixels": pixels_local,
        "entry": entry_pt - origin,
        "exit": exit_pt - origin,
        "entry_dir": tuple(map(int, entry_dir)),
        "exit_dir": tuple(map(int, exit_dir)),
        "size": int(len(comp)),
        "top_row": int(comp[:, 0].min()),
        "left_col": int(comp[:, 1].min()),
    }


def _extract_segments(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in (BLACK, GREY, AZURE, MAROON):
            continue
        comps = _connected_components(grid == color)
        for comp in comps:
            infos.append(_parse_segment(grid, color, comp))
    assert infos, "at least one segment is required"
    return infos


def _assemble_chain(segments: list[dict], anchor_pixels: np.ndarray, *, return_layout: bool = False):
    anchor_pixels = np.asarray(anchor_pixels, dtype=int)
    objects = [{"color": AZURE, "pixels": anchor_pixels.copy()}]
    occupied: set[tuple[int, int]] = _coord_set(anchor_pixels)

    current_exit = None
    current_dir = None
    for idx, seg in enumerate(segments):
        seg_pixels = np.asarray(seg["pixels"], dtype=int)
        entry = np.asarray(seg["entry"], dtype=int)
        exit_pt = np.asarray(seg["exit"], dtype=int)
        entry_dir = tuple(map(int, seg["entry_dir"]))
        exit_dir = tuple(map(int, seg["exit_dir"]))
        start_dir = (-entry_dir[0], -entry_dir[1])

        if idx == 0:
            anchor_pt = _choose_anchor_point(anchor_pixels, start_dir)
            target_entry = anchor_pt + 2 * np.array(start_dir, dtype=int)
        else:
            assert start_dir == tuple(map(int, current_dir)), "segment entrance must face the previous exit direction"
            target_entry = current_exit + 2 * current_dir

        shift = target_entry - entry
        abs_pixels = seg_pixels + shift
        for cell in _coord_set(abs_pixels):
            assert cell not in occupied, "assembled segments must not overlap"
        occupied |= _coord_set(abs_pixels)
        objects.append({"color": int(seg["color"]), "pixels": abs_pixels})
        current_exit = exit_pt + shift
        current_dir = np.array(exit_dir, dtype=int)

    all_pts = np.vstack([obj["pixels"] for obj in objects])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    out = np.zeros((int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1)), dtype=int)
    for obj in objects:
        shifted = obj["pixels"] - mins
        out[shifted[:, 0], shifted[:, 1]] = int(obj["color"])
    if return_layout:
        return out, objects
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor_comps = _connected_components(g == AZURE)
    assert anchor_comps, "azure anchor is required"
    anchor = max(anchor_comps, key=len)
    anchor_local, _ = _normalize_coords(anchor)

    segments = _extract_segments(g)
    segments.sort(key=lambda seg: (int(seg["size"]), int(seg["top_row"]), int(seg["left_col"])))
    return _crop_nonblack(_assemble_chain(segments, anchor_local))


def _build_segment(color: int, size: int, start_dir: tuple[int, int]) -> dict:
    assert 2 <= size <= 5
    straight_only = size == 2 or bool(_RNG.integers(0, 2))
    coords = [(0, 0)]

    if straight_only:
        end_dir = start_dir
        for _ in range(size - 1):
            last = coords[-1]
            coords.append((last[0] + start_dir[0], last[1] + start_dir[1]))
    else:
        turn_dir = _choice([(start_dir[1], -start_dir[0]), (-start_dir[1], start_dir[0])])
        first_leg = _randint(2, size - 1)
        second_leg = size - first_leg + 1
        end_dir = turn_dir
        for _ in range(first_leg - 1):
            last = coords[-1]
            coords.append((last[0] + start_dir[0], last[1] + start_dir[1]))
        for _ in range(second_leg - 1):
            last = coords[-1]
            coords.append((last[0] + end_dir[0], last[1] + end_dir[1]))

    path = np.array(coords, dtype=int)
    pixels_local, origin = _normalize_coords(path)
    entry = path[0] - origin
    exit_pt = path[-1] - origin
    return {
        "color": int(color),
        "pixels": pixels_local,
        "entry": entry,
        "exit": exit_pt,
        "entry_dir": (-start_dir[0], -start_dir[1]),
        "exit_dir": end_dir,
        "size": int(size),
    }


def _build_scatter_object(seg: dict) -> dict:
    colored = np.asarray(seg["pixels"], dtype=int)
    entry = np.asarray(seg["entry"], dtype=int)
    exit_pt = np.asarray(seg["exit"], dtype=int)
    entry_marker = entry + np.array(seg["entry_dir"], dtype=int)
    exit_marker = exit_pt + np.array(seg["exit_dir"], dtype=int)
    full = np.vstack([colored, entry_marker[None, :], exit_marker[None, :]])
    full_local, origin = _normalize_coords(full)
    colored_local = colored - origin
    entry_marker_local = entry_marker - origin
    exit_marker_local = exit_marker - origin
    return {
        "color": int(seg["color"]),
        "colored": colored_local,
        "maroon": entry_marker_local,
        "grey": exit_marker_local,
        "footprint": full_local,
        "height": int(full_local[:, 0].max() + 1),
        "width": int(full_local[:, 1].max() + 1),
        "color_top_offset": int(colored_local[:, 0].min()),
        "size": int(seg["size"]),
    }


def _mark_blocked(blocked: np.ndarray, pts: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in np.asarray(pts, dtype=int):
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                blocked[rr, cc] = True


def _objects_have_clearance(objects: list[dict]) -> bool:
    sets = [_coord_set(obj["pixels"]) for obj in objects]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                return False
            for r, c in sets[i]:
                for dr, dc in DIRS:
                    if (r + dr, c + dc) in sets[j]:
                        return False
    return True


def _sample_sizes() -> list[int]:
    use_tie = bool(_RNG.integers(0, 2))
    if not use_tie:
        count = _randint(3, 4)
        return sorted(int(v) for v in _sample(list(range(2, 6)), count))

    count = _randint(4, 6)
    for _ in range(100):
        vals = sorted(int(v) for v in _RNG.integers(2, 6, size=count))
        if len(set(vals)) < len(vals):
            return vals
    raise RuntimeError("failed to sample tied sizes")


def _place_anchor(grid: np.ndarray, blocked: np.ndarray) -> None:
    h, w = grid.shape
    corner = _choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        pos = np.array([[0, 0]], dtype=int)
    elif corner == "tr":
        pos = np.array([[0, w - 1]], dtype=int)
    elif corner == "bl":
        pos = np.array([[h - 1, 0]], dtype=int)
    else:
        pos = np.array([[h - 1, w - 1]], dtype=int)
    grid[pos[:, 0], pos[:, 1]] = AZURE
    _mark_blocked(blocked, pos)


def _gen_attempt() -> np.ndarray | None:
    sizes = _sample_sizes()
    count = len(sizes)
    colors = [int(v) for v in _sample(list(SEGMENT_COLORS), count)]
    first_dir = _choice(DIRS)

    segments: list[dict] = []
    current_dir = first_dir
    for color, size in zip(colors, sizes):
        seg = _build_segment(color, int(size), current_dir)
        segments.append(seg)
        current_dir = tuple(map(int, seg["exit_dir"]))

    try:
        expected, layout = _assemble_chain(segments, np.array([[0, 0]], dtype=int), return_layout=True)
    except AssertionError:
        return None
    if not _objects_have_clearance(layout):
        return None

    h = _randint(22, 30)
    w = _randint(22, 30)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    _place_anchor(grid, blocked)

    scatter_objects = [_build_scatter_object(seg) for seg in segments]
    last_top_by_size: dict[int, int] = {}
    for obj in scatter_objects:
        placed = False
        prev_top = last_top_by_size.get(int(obj["size"]), -1)
        for _ in range(400):
            r0 = _randint(0, h - int(obj["height"]))
            c0 = _randint(0, w - int(obj["width"]))
            color_top = int(r0 + obj["color_top_offset"])
            if color_top <= prev_top:
                continue
            pts = obj["footprint"] + np.array([r0, c0], dtype=int)
            if bool(blocked[pts[:, 0], pts[:, 1]].any()):
                continue
            colored = obj["colored"] + np.array([r0, c0], dtype=int)
            maroon = obj["maroon"] + np.array([r0, c0], dtype=int)
            grey = obj["grey"] + np.array([r0, c0], dtype=int)
            grid[colored[:, 0], colored[:, 1]] = int(obj["color"])
            grid[int(maroon[0]), int(maroon[1])] = MAROON
            grid[int(grey[0]), int(grey[1])] = GREY
            _mark_blocked(blocked, pts)
            last_top_by_size[int(obj["size"])] = color_top
            placed = True
            break
        if not placed:
            return None

    top_rows = {}
    for seg in _extract_segments(grid):
        key = (int(seg["color"]), int(seg["size"]), int(seg["left_col"]))
        top_rows[key] = int(seg["top_row"])
    for size in sorted(set(sizes)):
        group = [seg for seg in _extract_segments(grid) if int(seg["size"]) == int(size)]
        if len(group) <= 1:
            continue
        rows = [int(seg["top_row"]) for seg in sorted(group, key=lambda seg: int(seg["top_row"]))]
        if len(rows) != len(set(rows)):
            return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
