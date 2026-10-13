from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PINK = 6

SEGMENT_MIN_LEN = 5
SEGMENT_MAX_LEN = 10
GRID_MIN = 14
GRID_MAX = 20
GEN_ATTEMPTS = 500
SEGMENT_ATTEMPTS = 300
COUNT_ATTEMPTS = 200


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = r1 - r0
    dc = c1 - c0
    assert abs(dr) == abs(dc) and dr != 0
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    steps = abs(dr)
    return [(r0 + k * sr, c0 + k * sc) for k in range(steps + 1)]


def _bbox(cells: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(cells, np.ndarray):
        rows = cells[:, 0]
        cols = cells[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _orientation_dirs(orientation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if orientation == "backslash":
        return (-1, 1), (1, -1)
    if orientation == "slash":
        return (-1, -1), (1, 1)
    raise ValueError(f"unknown orientation: {orientation}")


def _collect_blue_segments(grid: np.ndarray) -> list[dict]:
    blue_pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    assert len(blue_pts) >= 4 and len(blue_pts) % 2 == 0

    segments: list[dict] = []
    for orientation in ("backslash", "slash"):
        buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r, c in blue_pts:
            key = r - c if orientation == "backslash" else r + c
            buckets[key].append((r, c))
        for pts in buckets.values():
            if len(pts) == 1:
                continue
            assert len(pts) == 2, "expected exactly one blue pair on each used diagonal"
            p, q = sorted(pts)
            assert abs(p[0] - q[0]) == abs(p[1] - q[1])
            assert SEGMENT_MIN_LEN <= abs(p[0] - q[0]) <= SEGMENT_MAX_LEN
            segments.append(
                {
                    "orientation": orientation,
                    "endpoints": (p, q),
                    "path": _diag_path(p, q),
                }
            )

    counts = Counter()
    for segment in segments:
        for pt in segment["endpoints"]:
            counts[pt] += 1
    assert len(segments) * 2 == len(blue_pts)
    assert all(counts[pt] == 1 for pt in blue_pts), "each blue endpoint must belong to exactly one pair"
    return segments


def _render_completed_grid(
    base: np.ndarray,
    segments: list[dict],
    triggers: dict[tuple[int, int], str],
) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    h, w = out.shape

    for segment in segments:
        for r, c in segment["path"]:
            out[r, c] = BLUE

    for trigger in triggers:
        out[trigger] = PINK

    for trigger, orientation in triggers.items():
        for dr, dc in _orientation_dirs(orientation):
            r, c = trigger[0] + dr, trigger[1] + dc
            while 0 <= r < h and 0 <= c < w:
                if out[r, c] == BLUE:
                    break
                out[r, c] = PINK
                r += dr
                c += dc
    return out


def _region_output(completed: np.ndarray) -> np.ndarray:
    passable = (completed == BLACK) | (completed == RED)
    regions = _components4(passable)
    assert regions, "expected at least one open region"

    scored: list[tuple[int, int, int, np.ndarray]] = []
    for region in regions:
        red_count = int(np.count_nonzero(completed[region[:, 0], region[:, 1]] == RED))
        region_sorted = region[np.lexsort((region[:, 1], region[:, 0]))]
        anchor_r, anchor_c = map(int, region_sorted[0])
        scored.append((red_count, -anchor_r, -anchor_c, region))

    _, _, _, best = max(scored)
    r0, c0, r1, c1 = _bbox(best)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in best:
        out[r - r0, c - c0] = int(completed[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each blue diagonal pair, shoot perpendicular magenta rays from trigger pixels, then crop the region with the most red markers.

    Input structure:
    - The background is black, with 2-3 blue endpoint pairs lying on distinct 45-degree diagonals.
    - One or two magenta trigger pixels lie strictly between the endpoints of some blue pair, and red marker pixels lie in the open space.
    - Blue and magenta pixels act as future walls; red markers belong to the open regions unless a ray overwrites them.

    Rule:
    - Fill every blue endpoint pair into a solid blue diagonal segment.
    - For each magenta trigger, find the blue segment containing it and extend magenta rays in the perpendicular diagonal directions until the boundary, stopping just before any blue cell.
    - Treat only black and red cells as 4-connected open regions, count the red markers in each region, choose the region with the largest count, and break ties by the lexicographically smallest region cell.
    - Return the tight bounding-box crop of that chosen region, keeping only its own black/red cells and turning everything else in the crop black.

    Key insight:
    - The answer is determined only after reconstructing the full diagonal wall network, because those blue and magenta diagonals implicitly partition the grid into irregular regions whose red-marker counts must then be compared.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    segments = _collect_blue_segments(g)
    trigger_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == PINK)]
    assert 1 <= len(trigger_cells) <= 2

    trigger_to_orientation: dict[tuple[int, int], str] = {}
    for trigger in trigger_cells:
        hosts = [segment for segment in segments if trigger in set(segment["path"])]
        assert len(hosts) == 1, "each trigger must lie on exactly one blue diagonal pair"
        trigger_to_orientation[trigger] = str(hosts[0]["orientation"])

    completed = _render_completed_grid(g, segments, trigger_to_orientation)
    return _region_output(completed)


def _sample_segment(
    shape: tuple[int, int],
    existing_blue: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    require_trigger: bool,
    min_ray_len: int,
) -> tuple[dict, tuple[int, int] | None, set[tuple[int, int]]] | None:
    h, w = shape
    max_len = min(SEGMENT_MAX_LEN, h - 1, w - 1)
    if max_len < SEGMENT_MIN_LEN:
        return None

    for _ in range(SEGMENT_ATTEMPTS):
        orientation = random.choice(["backslash", "slash"])
        length = random.randint(SEGMENT_MIN_LEN, max_len)
        if orientation == "backslash":
            r0 = random.randint(0, h - 1 - length)
            c0 = random.randint(0, w - 1 - length)
            path = [(r0 + k, c0 + k) for k in range(length + 1)]
        else:
            r0 = random.randint(0, h - 1 - length)
            c0 = random.randint(length, w - 1)
            path = [(r0 + k, c0 - k) for k in range(length + 1)]

        path_set = set(path)
        if path_set & occupied:
            continue

        endpoints = [path[0], path[-1]]
        if any(abs(er - br) == abs(ec - bc) for er, ec in endpoints for br, bc in existing_blue):
            continue

        trigger = None
        ray_cells: set[tuple[int, int]] = set()
        if require_trigger:
            interior = path[2:-2]
            random.shuffle(interior)
            chosen = None
            chosen_rays: set[tuple[int, int]] | None = None
            for cell in interior:
                rays: set[tuple[int, int]] = set()
                lengths = []
                ok = True
                for dr, dc in _orientation_dirs(orientation):
                    rr, cc = cell[0] + dr, cell[1] + dc
                    run = []
                    while 0 <= rr < h and 0 <= cc < w:
                        if (rr, cc) in occupied or (rr, cc) in path_set:
                            ok = False
                            break
                        run.append((rr, cc))
                        rr += dr
                        cc += dc
                    if not ok:
                        break
                    lengths.append(len(run))
                    rays.update(run)
                if ok and min(lengths) >= min_ray_len:
                    chosen = cell
                    chosen_rays = rays
                    break
            if chosen is None or chosen_rays is None:
                continue
            trigger = chosen
            ray_cells = chosen_rays

        segment = {
            "orientation": orientation,
            "endpoints": (path[0], path[-1]),
            "path": path,
        }
        return segment, trigger, ray_cells
    return None


def _walls_only_grid(
    shape: tuple[int, int],
    segments: list[dict],
    triggers: dict[tuple[int, int], str],
) -> np.ndarray:
    return _render_completed_grid(np.zeros(shape, dtype=int), segments, triggers)


def _choose_marker_counts(regions: list[np.ndarray]) -> tuple[dict[int, int], int] | None:
    anchors = []
    sizes = []
    for region in regions:
        region_sorted = region[np.lexsort((region[:, 1], region[:, 0]))]
        anchors.append((int(region_sorted[0, 0]), int(region_sorted[0, 1])))
        sizes.append(int(region.shape[0]))

    candidates = [i for i, size in enumerate(sizes) if size >= 2]
    if len(candidates) < 2:
        return None

    for _ in range(COUNT_ATTEMPTS):
        tie_branch = random.random() < 0.35
        active_n = random.randint(2, min(4, len(candidates)))
        active = random.sample(candidates, active_n)
        counts = {i: 0 for i in range(len(regions))}

        if tie_branch and active_n >= 2:
            a, b = random.sample(active, 2)
            top_cap = min(sizes[a], sizes[b], 4)
            if top_cap < 2:
                continue
            top = random.randint(2, top_cap)
            counts[a] = top
            counts[b] = top
            for idx in active:
                if idx in (a, b):
                    continue
                cap = min(sizes[idx], top - 1, 3)
                counts[idx] = 0 if cap <= 0 else random.randint(0, cap)
            if sum(v > 0 for v in counts.values()) < 2:
                continue
            total = sum(counts.values())
            if 5 <= total <= 10:
                winner = min([a, b], key=lambda idx: anchors[idx])
                return counts, winner
            continue

        winner = random.choice(active)
        top_cap = min(sizes[winner], 5)
        if top_cap < 3:
            continue
        top = random.randint(3, top_cap)
        counts[winner] = top
        ok = True
        for idx in active:
            if idx == winner:
                continue
            cap = min(sizes[idx], top - 1, 3)
            if cap < 1:
                ok = False
                break
            counts[idx] = random.randint(1, cap)
        if not ok:
            continue
        total = sum(counts.values())
        if 5 <= total <= 10:
            return counts, winner
    return None


def _blue_pairs_are_unambiguous(segments: list[dict]) -> bool:
    endpoints = []
    for segment in segments:
        endpoints.extend(segment["endpoints"])
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if abs(endpoints[i][0] - endpoints[j][0]) != abs(endpoints[i][1] - endpoints[j][1]):
                continue
            same_segment = any(
                endpoints[i] in segment["endpoints"] and endpoints[j] in segment["endpoints"]
                for segment in segments
            )
            if not same_segment:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)

    segment_count = random.choice([2, 3])
    trigger_count = random.choice([1, 2]) if segment_count == 3 else random.choice([1, 2])
    trigger_count = min(trigger_count, segment_count)

    segments: list[dict] = []
    trigger_to_orientation: dict[tuple[int, int], str] = {}
    occupied: set[tuple[int, int]] = set()
    blue_pts: list[tuple[int, int]] = []

    for idx in range(segment_count):
        require_trigger = idx < trigger_count
        min_ray_len = 3 if idx == 0 else 2
        sampled = _sample_segment(shape, blue_pts, occupied, require_trigger, min_ray_len)
        if sampled is None:
            return None
        segment, trigger, ray_cells = sampled
        segments.append(segment)
        blue_pts.extend(segment["endpoints"])
        occupied.update(segment["path"])
        if trigger is not None:
            trigger_to_orientation[trigger] = str(segment["orientation"])
            occupied.add(trigger)
            occupied.update(ray_cells)

    if not _blue_pairs_are_unambiguous(segments):
        return None

    walls = _walls_only_grid(shape, segments, trigger_to_orientation)
    passable = (walls == BLACK)
    regions = _components4(passable)
    if len(regions) < 3:
        return None

    non_tiny = [region for region in regions if region.shape[0] >= 4]
    if len(non_tiny) < 2:
        return None

    plan = _choose_marker_counts(regions)
    if plan is None:
        return None
    counts, planned_winner = plan

    final_grid = walls.copy()
    for idx, count in counts.items():
        if count <= 0:
            continue
        cells = [tuple(map(int, rc)) for rc in regions[idx]]
        if len(cells) < count:
            return None
        for r, c in random.sample(cells, count):
            final_grid[r, c] = RED

    input_grid = np.zeros(shape, dtype=int)
    input_grid[final_grid == RED] = RED
    for segment in segments:
        for r, c in segment["endpoints"]:
            input_grid[r, c] = BLUE
    for trigger in trigger_to_orientation:
        input_grid[trigger] = PINK

    expected = _region_output(final_grid)
    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    completed = _render_completed_grid(input_grid, _collect_blue_segments(input_grid), trigger_to_orientation)
    selected = _region_output(completed)
    if not np.array_equal(selected, expected):
        return None

    completed_passable = (completed == BLACK) | (completed == RED)
    completed_regions = _components4(completed_passable)
    actual_winner_anchor = None
    max_red = -1
    for region in completed_regions:
        region_sorted = region[np.lexsort((region[:, 1], region[:, 0]))]
        anchor = (int(region_sorted[0, 0]), int(region_sorted[0, 1]))
        red_count = int(np.count_nonzero(completed[region[:, 0], region[:, 1]] == RED))
        if red_count > max_red or (red_count == max_red and (actual_winner_anchor is None or anchor < actual_winner_anchor)):
            max_red = red_count
            actual_winner_anchor = anchor
    planned_region_sorted = regions[planned_winner][np.lexsort((regions[planned_winner][:, 1], regions[planned_winner][:, 0]))]
    planned_anchor = (int(planned_region_sorted[0, 0]), int(planned_region_sorted[0, 1]))
    if counts[planned_winner] != max_red or planned_anchor != actual_winner_anchor:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
