import random
from collections import deque

import numpy as np


BG = 0
MARKERS = {"H": 2, "V": 1, "R": 4}
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors(point):
    r, c = point
    return ((r + dr, c + dc) for dr, dc in DIRS)


def _extract_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or visited[r0, c0]:
                continue
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for nr, nc in _neighbors((r, c)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mirror_point(p, mode: str, axis_r2: int | None, axis_c2: int | None):
    r, c = p
    if mode == "H":
        return r, axis_c2 - c  # type: ignore[arg-type]
    if mode == "V":
        return axis_r2 - r, c  # type: ignore[arg-type]
    return axis_r2 - r, axis_c2 - c  # type: ignore[arg-type]


def _reflect_set(points: set[tuple[int, int]], mode: str, h: int, w: int, axis_r2: int | None, axis_c2: int | None):
    out = set()
    for p in points:
        mr, mc = _mirror_point(p, mode, axis_r2, axis_c2)
        if not _in_bounds(mr, mc, h, w):
            return None
        out.add((mr, mc))
    return out


def _component_count(points: set[tuple[int, int]]) -> int:
    if not points:
        return 0
    seen = set()
    count = 0
    for start in points:
        if start in seen:
            continue
        count += 1
        q = deque([start])
        seen.add(start)
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors((r, c)):
                if (nr, nc) in points and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return count


def _perimeter(points: set[tuple[int, int]]) -> int:
    if not points:
        return 0
    per = 0
    for r, c in points:
        for nr, nc in _neighbors((r, c)):
            if (nr, nc) not in points:
                per += 1
    return per


def _best_anomaly(points: set[tuple[int, int]], delta: set[tuple[int, int]], mode: str, axis_r2: int | None, axis_c2: int | None, h: int, w: int):
    if len(delta) != 2:
        return None

    cands = list(delta)
    best = None
    for p in cands:
        restored = set(points)
        if p in restored:
            restored.remove(p)
        else:
            restored.add(p)

        reflected = _reflect_set(restored, mode, h, w, axis_r2, axis_c2)
        if reflected is None:
            continue
        if reflected != restored:
            continue

        comps = _component_count(restored)
        per = _perimeter(restored)
        score = (comps, per, -len(restored))
        if best is None or score < best[0]:
            best = (score, p)

    if best is None:
        in_obs = [p for p in cands if p in points]
        out_obs = [p for p in cands if p not in points]
        if len(in_obs) == 1 and len(out_obs) == 1:
            return in_obs[0]
        return None
    return best[1]


def _axis_candidates_for_range(lo: int, hi: int):
    return range(lo - 1, hi + 2)


def _analyze_component(points: set[tuple[int, int]], h: int, w: int):
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    r_min, r_max = min(rs), max(rs)
    c_min, c_max = min(cs), max(cs)

    best_key = None
    best = None

    # Horizontal symmetry: mirror on a vertical axis.
    for axis2 in _axis_candidates_for_range(c_min * 2, c_max * 2):
        reflected = _reflect_set(points, "H", h, w, None, axis2)
        if reflected is None:
            continue
        delta = points.symmetric_difference(reflected)
        anomaly = _best_anomaly(points, delta, "H", None, axis2, h, w)
        valid = anomaly is not None and len(delta) == 2
        key = (len(delta), 0 if valid else 1, axis2)
        if best_key is None or key < best_key:
            best_key, best = key, ("H", None, axis2, anomaly)

    # Vertical symmetry: mirror on a horizontal axis.
    for axis2 in _axis_candidates_for_range(r_min * 2, r_max * 2):
        reflected = _reflect_set(points, "V", h, w, axis2, None)
        if reflected is None:
            continue
        delta = points.symmetric_difference(reflected)
        anomaly = _best_anomaly(points, delta, "V", axis2, None, h, w)
        valid = anomaly is not None and len(delta) == 2
        key = (len(delta), 0 if valid else 1, axis2)
        if best_key is None or key < best_key:
            best_key, best = key, ("V", axis2, None, anomaly)

    # 180° rotation symmetry.
    for axis_r2 in _axis_candidates_for_range(r_min * 2, r_max * 2):
        for axis_c2 in _axis_candidates_for_range(c_min * 2, c_max * 2):
            reflected = _reflect_set(points, "R", h, w, axis_r2, axis_c2)
            if reflected is None:
                continue
            delta = points.symmetric_difference(reflected)
            anomaly = _best_anomaly(points, delta, "R", axis_r2, axis_c2, h, w)
            valid = anomaly is not None and len(delta) == 2
            key = (len(delta), 0 if valid else 1, axis_r2, axis_c2)
            if best_key is None or key < best_key:
                best_key, best = key, ("R", axis_r2, axis_c2, anomaly)

    return best


def _is_fixed_point(p: tuple[int, int], mode: str, axis_r2: int | None, axis_c2: int | None) -> bool:
    r, c = p
    if mode == "H":
        return axis_c2 is not None and c * 2 == axis_c2
    if mode == "V":
        return axis_r2 is not None and r * 2 == axis_r2
    return axis_r2 is not None and axis_c2 is not None and 2 * r == axis_r2 and 2 * c == axis_c2


def _generate_symmetric_pattern(mode: str, h: int, w: int):
    steps = random.randint(max(8, (h * w) // 8), max(10, (h * w) // 4))
    if mode == "H":
        axis_c = random.randint(1, w - 2)
        axis_c2 = axis_c * 2
        start = (random.randint(0, h - 1), axis_c)
        points = {start}
        for _ in range(steps):
            r, c = random.choice(list(points))
            dr, dc = random.choice(DIRS)
            nr, nc = r + dr, c + dc
            mc = axis_c2 - nc
            if 0 <= nr < h and 0 <= nc < w and 0 <= mc < w:
                points.add((nr, nc))
                points.add((nr, mc))
        return points, ("H", None, axis_c2)

    if mode == "V":
        axis_r = random.randint(1, h - 2)
        axis_r2 = axis_r * 2
        start = (axis_r, random.randint(0, w - 1))
        points = {start}
        for _ in range(steps):
            r, c = random.choice(list(points))
            dr, dc = random.choice(DIRS)
            nr, nc = r + dr, c + dc
            mr = axis_r2 - nr
            if 0 <= nr < h and 0 <= nc < w and 0 <= mr < h:
                points.add((nr, nc))
                points.add((mr, nc))
        return points, ("V", axis_r2, None)

    axis_r = random.randint(1, h - 2)
    axis_c = random.randint(1, w - 2)
    axis_r2 = axis_r * 2
    axis_c2 = axis_c * 2
    start = (axis_r, axis_c)
    points = {start}
    for _ in range(steps):
        r, c = random.choice(list(points))
        dr, dc = random.choice(DIRS)
        nr, nc = r + dr, c + dc
        mr, mc = axis_r2 - nr, axis_c2 - nc
        if 0 <= nr < h and 0 <= nc < w and 0 <= mr < h and 0 <= mc < w:
            points.add((nr, nc))
            points.add((mr, mc))
    return points, ("R", axis_r2, axis_c2)


def _add_anomaly(points: set[tuple[int, int]], mode_info: tuple[str, int | None, int | None], h: int, w: int):
    mode, axis_r2, axis_c2 = mode_info
    if not points:
        return points

    if random.random() < 0.5:
        removable = [p for p in points if not _is_fixed_point(p, mode, axis_r2, axis_c2)]
        if len(removable) <= 1:
            return points
        for _ in range(80):
            p = random.choice(removable)
            points = set(points)
            points.remove(p)
            if not points:
                continue
            return points
        return points

    candidates = []
    for r, c in points:
        for nr, nc in _neighbors((r, c)):
            if not _in_bounds(nr, nc, h, w):
                continue
            if (nr, nc) in points:
                continue
            mr, mc = _mirror_point((nr, nc), mode, axis_r2, axis_c2)
            if _in_bounds(mr, mc, h, w) and (mr, mc) not in points:
                candidates.append((nr, nc))

    if not candidates:
        return points

    points = set(points)
    points.add(random.choice(candidates))
    return points


def generate_input() -> np.ndarray:
    """
    1) Grid setup
    --------------
    Creates a 2D numpy ARC-like grid with background black and several single-color patterns.

    2) Symmetric pattern synthesis
    -------------------------------
    For each pattern chooses one target symmetry type (H/V/R), builds a connected synthetic shape
    with exact symmetry, then injects exactly one pixel defect (add or remove).

    3) Placement strategy
    ---------------------
    Randomly places defect patterns onto a background canvas while avoiding overlaps and keeping
    all coordinates in bounds.

    4) Diversity
    -------------
    Random pattern count, sizes, colors, centers, symmetry axis positions, and defect type give
    broad variety while respecting the required size bound (<= 30).
    """

    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.zeros((h, w), dtype=np.int64)
    occupied = np.zeros((h, w), dtype=bool)

    palette = [3, 5, 6, 7, 8, 9]
    random.shuffle(palette)
    pattern_count = random.randint(1, 4)
    symm = ["H", "V", "R"]

    for _ in range(pattern_count):
        mode = random.choice(symm)
        ph = random.randint(5, min(12, h))
        pw = random.randint(5, min(12, w))

        points, mode_info = _generate_symmetric_pattern(mode, ph, pw)
        points = _add_anomaly(points, mode_info, ph, pw)
        if len(points) < 3:
            continue

        min_r = min(r for r, _ in points)
        max_r = max(r for r, _ in points)
        min_c = min(c for _, c in points)
        max_c = max(c for _, c in points)
        bh = max_r - min_r + 1
        bw = max_c - min_c + 1

        if bh > h or bw > w:
            continue

        placed = False
        for _ in range(200):
            base_r = random.randint(0, h - bh)
            base_c = random.randint(0, w - bw)

            translated = {(r - min_r + base_r, c - min_c + base_c) for r, c in points}
            if any(occupied[r, c] for r, c in translated):
                continue
            color = palette[random.randrange(len(palette))]
            for r, c in translated:
                grid[r, c] = color
                occupied[r, c] = True
            placed = True
            break
        if not placed:
            continue

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    1) Input handling
    ------------------
    Accept a 2D numpy integer grid and keep background/size unchanged.

    2) Component isolation
    ----------------------
    Extract every connected non-background color component and estimate its intended symmetry.

    3) Symmetry inference
    ---------------------
    For each component test H/V/180° candidates, choose the symmetry with the smallest
    disagreement and locate the single pixel whose correction makes symmetry hold.

    4) Output labeling
    -------------------
    Replace that pixel with marker colors by symmetry type:
    horizontal -> red(2), vertical -> blue(1), rotation -> yellow(4).
    """

    grid = np.asarray(grid)
    if grid.ndim != 2:
        return np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    h, w = grid.shape

    comps = _extract_components(grid != BG)
    for comp in comps:
        points = set(comp)
        result = _analyze_component(points, h, w)
        if not result:
            continue
        mode, axis_r2, axis_c2, anomaly = result
        if anomaly is None:
            continue
        ar, ac = anomaly
        if _in_bounds(ar, ac, h, w):
            out[ar, ac] = MARKERS[mode]

    return out

