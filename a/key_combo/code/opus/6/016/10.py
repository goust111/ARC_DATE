import random
from collections import deque

import numpy as np


BG = 0
PATH = 7
FILL = 8
OBJECT_COLORS = [1, 2, 3, 4, 5, 6, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in neigh:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    pad = np.pad(mask.astype(bool), 1)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= pad[dr : dr + h, dc : dc + w]
    return out


def _channel_from_grid(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    path_mask = grid == PATH
    comps = _components_mask(path_mask, connectivity=4)
    assert len(comps) == 2, "expected exactly two orange path components"
    masks = []
    for coords in comps:
        comp_mask = np.zeros_like(path_mask, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        masks.append(comp_mask)
    channel = _dilate8(masks[0]) & _dilate8(masks[1]) & ~path_mask
    return channel, comps


def _extract_objects(grid: np.ndarray, channel: np.ndarray) -> list[dict]:
    objects = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, PATH)):
        for coords in _components_mask(grid == color, connectivity=4):
            if not np.all(channel[coords[:, 0], coords[:, 1]]):
                continue
            max_r = int(coords[:, 0].max())
            min_r = int(coords[:, 0].min())
            min_c = int(coords[:, 1].min())
            objects.append(
                {
                    "color": int(color),
                    "coords": coords.copy(),
                    "area": int(coords.shape[0]),
                    "max_r": max_r,
                    "min_r": min_r,
                    "min_c": min_c,
                }
            )
    objects.sort(key=lambda obj: (-obj["max_r"], -obj["min_r"], obj["min_c"], obj["color"]))
    return objects


def _top_connected_empty(channel: np.ndarray, grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    passable = channel & (grid == BG)
    out = np.zeros((h, w), dtype=bool)
    q = deque()
    for c in range(w):
        if passable[0, c]:
            out[0, c] = True
            q.append((0, c))
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _simulate(grid: np.ndarray, *, use_channel: bool, use_collision: bool, paint_fill: bool) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    channel, _ = _channel_from_grid(g)
    objects = _extract_objects(g, channel)
    assert 2 <= len(objects) <= 3, "expected two or three channel objects"

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    occupied = out == PATH
    info = []
    for obj in objects:
        coords = obj["coords"]
        color = obj["color"]
        area = obj["area"]

        best = coords.copy()
        best_shift = 0
        collided = False

        # Evaluate the downward slide one row at a time so collision-stop can
        # leave the object at the last non-colliding shifted position.
        for shift in range(1, area + 1):
            shifted = coords.copy()
            shifted[:, 0] += shift
            in_bounds = (0 <= shifted[:, 0]) & (shifted[:, 0] < h) & (0 <= shifted[:, 1]) & (shifted[:, 1] < w)
            if use_channel:
                keep = in_bounds.copy()
                if np.any(keep):
                    kk = np.flatnonzero(keep)
                    keep[kk] &= channel[shifted[kk, 0], shifted[kk, 1]]
            else:
                keep = in_bounds
            visible = shifted[keep]
            if use_collision and visible.size:
                if occupied[visible[:, 0], visible[:, 1]].any():
                    collided = True
                    break
            best = visible.copy()
            best_shift = shift

        if best.size:
            out[best[:, 0], best[:, 1]] = color
            occupied[best[:, 0], best[:, 1]] = True

        info.append(
            {
                "color": color,
                "area": area,
                "initial": coords.copy(),
                "final": best.copy(),
                "shift": best_shift,
                "collided": collided,
                "clipped": int(best.shape[0]) < area,
                "changed": not _same_cells(coords, best),
            }
        )

    fill_before = _top_connected_empty(channel, g)
    fill_after = _top_connected_empty(channel, out)
    if paint_fill:
        out[fill_after] = FILL

    meta = {
        "channel": channel,
        "objects": info,
        "fill_before": fill_before,
        "fill_after": fill_after,
        "cyan_count": int(fill_after.sum()),
    }
    return out, meta


def _same_cells(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if a.size == 0 and b.size == 0:
        return True
    aa = sorted(map(tuple, a.tolist()))
    bb = sorted(map(tuple, b.tolist()))
    return aa == bb


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide channel objects downward inside the overlap corridor and mark the remaining top-reachable corridor.

    Input structure:
    - Black background contains exactly two orange 4-connected one-pixel-wide path components.
    - The middle corridor is defined by the non-orange intersection of the two paths' 8-neighborhood dilations.
    - Two or three small monochromatic 4-connected objects lie entirely inside that corridor.

    Rule:
    - Compute the corridor as the intersection of the two orange dilation halos, excluding orange path pixels themselves.
    - Process the corridor objects from lowest to highest; each object tries to translate straight downward by a number of rows equal to its area.
    - For every trial row shift, keep only pixels whose shifted locations still lie in the corridor; if the shifted visible pixels would hit an orange pixel or a previously stopped object, stop one row earlier.
    - After all objects stop, recolor every still-black corridor pixel that is 4-connected to the top edge through still-black corridor cells to cyan.

    Key insight:
    - The dilation-overlap corridor simultaneously defines the legal motion lane, the effective collision geometry, and the final region whose top-reachable residue is painted.
    """
    out, _ = _simulate(np.asarray(grid, dtype=int), use_channel=True, use_collision=True, paint_fill=True)
    return out


def _left_normal(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _wall_sets_from_path(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    left: set[tuple[int, int]] = set()
    right: set[tuple[int, int]] = set()
    channel = set(path)
    for i, (r, c) in enumerate(path):
        dirs = []
        if i > 0:
            pr, pc = path[i - 1]
            dirs.append((r - pr, c - pc))
        if i + 1 < len(path):
            nr, nc = path[i + 1]
            dirs.append((nr - r, nc - c))
        for dr, dc in dirs:
            lr, lc = _left_normal(dr, dc)
            if (r + lr, c + lc) not in channel:
                left.add((r + lr, c + lc))
            if (r - lr, c - lc) not in channel:
                right.add((r - lr, c - lc))
        if len(dirs) == 2 and dirs[0] != dirs[1]:
            for side in (1, -1):
                offsets = []
                for dr, dc in dirs:
                    lr, lc = _left_normal(dr, dc)
                    offsets.append((side * lr, side * lc))
                if all((r + dr, c + dc) not in channel for dr, dc in offsets):
                    er = r + offsets[0][0] + offsets[1][0]
                    ec = c + offsets[0][1] + offsets[1][1]
                    if side == 1:
                        left.add((er, ec))
                    else:
                        right.add((er, ec))
    return left, right


def _path_mask(path: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in path:
        mask[r, c] = True
    return mask


def _build_path_attempt() -> tuple[list[tuple[int, int]], list[tuple[str, int, int]], tuple[int, int]] | None:
    h = random.randint(18, 24)
    w = random.randint(16, 22)
    h_count = random.choice([1, 1, 1, 2, 2])
    bottom_gap = random.randint(0, 2)
    vertical_total = h - 1 - bottom_gap
    seg_count = h_count + 1
    mins = [4] + [3] * max(0, seg_count - 2) + [7]
    if sum(mins) > vertical_total:
        return None
    extra = vertical_total - sum(mins)
    v_lengths = mins[:]
    for _ in range(extra):
        v_lengths[random.randrange(seg_count)] += 1

    h_lengths = [random.randint(1, 3) for _ in range(h_count)]
    hdir = random.choice([-1, 1])
    total_h = sum(h_lengths)
    min_c = 3
    max_c = w - 4
    if hdir > 0:
        start_min = min_c
        start_max = max_c - total_h
    else:
        start_min = min_c + total_h
        start_max = max_c
    if start_min > start_max:
        return None
    c = random.randint(start_min, start_max)
    r = 0
    path = [(r, c)]
    segments: list[tuple[str, int, int]] = []
    for i, vlen in enumerate(v_lengths):
        s = len(path) - 1
        for _ in range(vlen):
            r += 1
            path.append((r, c))
        e = len(path) - 1
        segments.append(("V", s, e))
        if i < h_count:
            s = len(path) - 1
            for _ in range(h_lengths[i]):
                c += hdir
                path.append((r, c))
            e = len(path) - 1
            segments.append(("H", s, e))

    if r != h - 1 - bottom_gap:
        return None
    if len(path) != len(set(path)):
        return None
    return path, segments, (h, w)


def _build_empty_grid_from_path(path: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    left, right = _wall_sets_from_path(path)
    channel_cells = set(path)
    all_wall = left | right
    if left & right or left & channel_cells or right & channel_cells:
        return None
    for r, c in all_wall:
        if not (0 <= r < h and 0 <= c < w):
            return None
        grid[r, c] = PATH
    comps = _components_mask(grid == PATH, connectivity=4)
    if len(comps) != 2:
        return None
    channel, _ = _channel_from_grid(grid)
    if not np.array_equal(channel, _path_mask(path, grid.shape)):
        return None
    return grid


def _turn_indices(path: list[tuple[int, int]]) -> list[int]:
    turns = []
    for i in range(1, len(path) - 1):
        d0 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d1 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d0 != d1:
            turns.append(i)
    return turns


def _slice_has_turn(path: list[tuple[int, int]], a: int, b: int) -> bool:
    for i in range(a + 1, b):
        d0 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d1 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d0 != d1:
            return True
    return False


def _paint_object(grid: np.ndarray, path: list[tuple[int, int]], start: int, end: int, color: int) -> None:
    for idx in range(start, end + 1):
        r, c = path[idx]
        grid[r, c] = color


def _candidate_bent_slices(path: list[tuple[int, int]], turns: list[int]) -> list[tuple[int, int]]:
    cand = []
    for turn in turns:
        for length in (3, 4, 5):
            for left in range(1, length):
                a = turn - left
                b = a + length - 1
                if a < 0 or b >= len(path):
                    continue
                if _slice_has_turn(path, a, b):
                    cand.append((a, b))
    uniq = []
    seen = set()
    for item in cand:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _place_objects(grid: np.ndarray, path: list[tuple[int, int]], segments: list[tuple[str, int, int]]) -> bool:
    vertical_segments = [seg for seg in segments if seg[0] == "V"]
    last_v = vertical_segments[-1]
    _, last_start, last_end = last_v
    turns = _turn_indices(path)
    bent_candidates = _candidate_bent_slices(path, turns)
    if not bent_candidates:
        return False

    for _ in range(80):
        object_count = random.choice([3, 3, 2])
        colors = random.sample(OBJECT_COLORS, object_count)

        gap = random.randint(1, 2)
        lower_area = random.randint(2, 3)
        upper_min = max(4, lower_area + gap + 1)
        if upper_min > 5:
            continue
        upper_area = random.randint(upper_min, 5)

        lower_min = last_start + upper_area + gap
        lower_max = last_end - 2 * lower_area + 1
        if lower_min > lower_max:
            continue
        lower_start = random.randint(lower_min, lower_max)
        lower = (lower_start, lower_start + lower_area - 1)

        upper_end = lower_start - gap - 1
        upper_start = upper_end - upper_area + 1
        upper = (upper_start, upper_end)
        if upper_start < last_start:
            continue

        if object_count == 3:
            eligible_bent = [
                cand
                for cand in bent_candidates
                if cand[1] < upper_start - 1 and cand[1] - cand[0] + 1 <= 5
            ]
            if not eligible_bent:
                continue
            specs = [random.choice(eligible_bent), upper, lower]
        else:
            eligible_bent = [
                cand
                for cand in bent_candidates
                if cand[1] < lower_start - 1
                and cand[1] >= last_start - 3
                and cand[1] - cand[0] + 1 >= 4
            ]
            if not eligible_bent:
                continue
            bent = random.choice(eligible_bent)
            specs = [bent, lower]

        specs.sort(key=lambda ab: ab[0])
        if any(b0 >= a1 - 1 for (_, b0), (a1, _) in zip(specs, specs[1:])):
            continue

        trial = grid.copy()
        for color, (a, b) in zip(colors, specs):
            _paint_object(trial, path, a, b, color)
        if _accept_generated(trial):
            grid[:] = trial
            return True
    return False


def _accept_generated(grid: np.ndarray) -> bool:
    try:
        out, meta = _simulate(grid, use_channel=True, use_collision=True, paint_fill=True)
        out_no_channel, _ = _simulate(grid, use_channel=False, use_collision=True, paint_fill=True)
        out_no_collision, _ = _simulate(grid, use_channel=True, use_collision=False, paint_fill=True)
    except Exception:
        return False

    if np.array_equal(grid, out):
        return False
    if meta["cyan_count"] <= 0:
        return False
    if np.array_equal(meta["fill_before"], meta["fill_after"]):
        return False
    if np.array_equal(out, out_no_channel):
        return False
    if np.array_equal(out, out_no_collision):
        return False

    objects = meta["objects"]
    if not any(obj["collided"] for obj in objects):
        return False
    if not any(obj["clipped"] for obj in objects):
        return False
    if not all(obj["final"].size > 0 for obj in objects):
        return False
    if not any(obj["changed"] for obj in objects):
        return False

    channel = meta["channel"]
    if int(channel.sum()) <= sum(obj["area"] for obj in objects) + 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    built = _build_path_attempt()
    if built is None:
        return None
    path, segments, shape = built
    grid = _build_empty_grid_from_path(path, shape)
    if grid is None:
        return None
    if not _place_objects(grid, path, segments):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 10")
