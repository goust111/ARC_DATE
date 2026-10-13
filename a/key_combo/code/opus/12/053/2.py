from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
MIRROR = 5
SOURCE_COLORS = (2, 3, 4, 6, 7, 8, 9)
LOOP_TEMPLATES = ((3, 4), (3, 5), (4, 4), (4, 5))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        deltas = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _enclosed_regions(wall_coords: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = map(int, box)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    wall = np.zeros((h, w), dtype=bool)
    local = wall_coords - np.array([r0, c0], dtype=int)
    wall[local[:, 0], local[:, 1]] = True
    passable = ~wall

    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    return [coords + np.array([r0, c0], dtype=int) for coords in _components_mask(enclosed, connectivity=4)]


def _extract_mirrors(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    comps = _components_mask(grid == MIRROR, connectivity=4)
    mirror_map = np.full(grid.shape, -1, dtype=int)
    mirrors: list[dict] = []
    for idx, coords in enumerate(comps):
        rows = sorted(set(int(r) for r in coords[:, 0]))
        assert len(rows) == 1, "mirror must be horizontal"
        row = rows[0]
        cols = sorted(int(c) for c in coords[:, 1])
        assert cols[-1] - cols[0] + 1 == len(cols), "mirror must be contiguous"
        mirror_map[coords[:, 0], coords[:, 1]] = idx
        mirrors.append(
            {
                "id": idx,
                "row": row,
                "c0": cols[0],
                "c1": cols[-1],
                "length": len(cols),
            }
        )
    return mirrors, mirror_map


def _extract_contours(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    comps = _components_mask(grid == BLUE, connectivity=4)
    blue_map = np.full(grid.shape, -1, dtype=int)
    contours: list[dict] = []
    for idx, coords in enumerate(comps):
        blue_map[coords[:, 0], coords[:, 1]] = idx
        box = _bbox_coords(coords)
        enclosed = _enclosed_regions(coords, box)
        assert len(enclosed) == 1, "each contour must enclose exactly one region"
        region = enclosed[0]
        interior_mask = np.zeros(grid.shape, dtype=bool)
        interior_mask[region[:, 0], region[:, 1]] = True
        contours.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": box,
                "interior_coords": region,
                "interior_mask": interior_mask,
            }
        )
    return contours, blue_map


def _accessible_targets(
    grid: np.ndarray,
    contours: list[dict],
    blue_map: np.ndarray,
    row: int,
    start_col: int,
    dc: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    targets: list[tuple[int, int]] = []
    seen_ids: set[int] = set()
    c = int(start_col) + int(dc)
    while 0 <= c < w:
        if int(grid[row, c]) == BLUE:
            comp_id = int(blue_map[row, c])
            nc = c + int(dc)
            if (
                comp_id >= 0
                and comp_id not in seen_ids
                and 0 <= nc < w
                and contours[comp_id]["interior_mask"][row, nc]
            ):
                seen_ids.add(comp_id)
                targets.append((comp_id, c))
        c += int(dc)
    return targets


def _trace_ray(
    grid: np.ndarray,
    contours: list[dict],
    blue_map: np.ndarray,
    source_r: int,
    source_c: int,
) -> dict:
    h, w = grid.shape
    hit_r = None
    for rr in range(source_r + 1, h):
        if int(grid[rr, source_c]) != BG:
            hit_r = rr
            break
    assert hit_r is not None and int(grid[hit_r, source_c]) == MIRROR, "source must hit a mirror below"

    mirrors, mirror_map = _extract_mirrors(grid)
    mirror_id = int(mirror_map[hit_r, source_c])
    assert mirror_id >= 0
    mirror = mirrors[mirror_id]
    hit_index = int(source_c - mirror["c0"])
    midpoint = (mirror["length"] - 1) // 2
    dc = -1 if hit_index <= midpoint else 1

    targets = _accessible_targets(grid, contours, blue_map, hit_r, source_c, dc)
    assert len(targets) >= 1, "ray must reach a contour"
    target_id, boundary_c = targets[0]

    path: list[tuple[int, int]] = []
    c = source_c + dc
    while c != boundary_c:
        value = int(grid[hit_r, c])
        if int(mirror_map[hit_r, c]) == mirror_id:
            c += dc
            continue
        assert value == BG, "path after leaving the mirror and before the target must be clear"
        path.append((hit_r, c))
        c += dc

    entry_c = boundary_c + dc
    total_length = (hit_r - source_r) + abs(entry_c - source_c)
    return {
        "source": (source_r, source_c),
        "mirror_id": mirror_id,
        "mirror_row": hit_r,
        "dc": dc,
        "path": path,
        "target_id": target_id,
        "boundary_c": boundary_c,
        "length": total_length,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each source pixel through a horizontal mirror and fill the unique closed region that the reflected ray enters.

    Input structure:
    - The grid uses black background, blue closed contour loops, and gray horizontal mirror segments.
    - Two or three colored single-pixel sources sit above the mirrors, each in a distinct column.
    - Each source has exactly one mirror segment directly below it, and each contour encloses one black interior region.

    Rule:
    - For every colored source, cast a vertical ray downward until it hits the gray mirror in the same column.
    - Convert that hit into a horizontal ray: hits on the midpoint-or-left half turn left, hits on the right half turn right.
    - Keep the horizontal path in the source color, and when the ray crosses a blue contour boundary into its enclosed region, fill that entire enclosed black region with the source color.
    - Erase the original source pixels; keep mirror pixels and blue contour borders unchanged. If two rays target the same contour, only the shorter full path fills it.

    Key insight:
    - The mirror is a router, not a reflected shape: the source column determines which half of the mirror is hit, which fixes the horizontal direction and therefore which enclosed region gets filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    contours, blue_map = _extract_contours(g)
    source_coords = np.argwhere((g != BG) & (g != BLUE) & (g != MIRROR))
    assert 2 <= len(source_coords) <= 3, "expected two or three sources"

    rays: list[dict] = []
    for r_raw, c_raw in source_coords:
        r = int(r_raw)
        c = int(c_raw)
        ray = _trace_ray(g, contours, blue_map, r, c)
        ray["color"] = int(g[r, c])
        rays.append(ray)

    winners: dict[int, dict] = {}
    for ray in rays:
        current = winners.get(ray["target_id"])
        if current is None or ray["length"] < current["length"]:
            winners[ray["target_id"]] = ray

    out = g.copy()
    for ray in rays:
        sr, sc = ray["source"]
        out[sr, sc] = BG
        for rr, cc in ray["path"]:
            out[rr, cc] = ray["color"]

    for target_id, ray in winners.items():
        out[contours[target_id]["interior_mask"]] = ray["color"]

    return out


def _sample_spaced_rows(low: int, high: int, count: int, gap: int) -> list[int] | None:
    candidates = list(range(low, high + 1))
    for _ in range(200):
        rows = sorted(random.sample(candidates, count))
        if all(rows[i + 1] - rows[i] >= gap for i in range(count - 1)):
            return rows
    return None


def _choose_directions(count: int) -> list[int]:
    if count == 2:
        dirs = [-1, 1]
        random.shuffle(dirs)
        return dirs
    patterns = ([-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [1, -1, 1])
    return list(random.choice(patterns))


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    dr = 0
    if a[2] < b[0]:
        dr = b[0] - a[2]
    elif b[2] < a[0]:
        dr = a[0] - b[2]
    dc = 0
    if a[3] < b[1]:
        dc = b[1] - a[3]
    elif b[3] < a[1]:
        dc = a[1] - b[3]
    return dr + dc


def _build_expected(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        sr, sc = pair["source"]
        out[sr, sc] = BG
        row = pair["mirror_row"]
        dc = pair["dir"]
        boundary_c = pair["loop_c1"] if dc == -1 else pair["loop_c0"]
        start_c = pair["mirror_c0"] - 1 if dc == -1 else pair["mirror_c1"] + 1
        for c in range(start_c, boundary_c, dc):
            out[row, c] = pair["color"]
        out[pair["loop_r0"] + 1 : pair["loop_r0"] + pair["loop_h"] - 1, pair["loop_c0"] + 1 : pair["loop_c0"] + pair["loop_w"] - 1] = pair["color"]
    return out


def _validate_generated(grid: np.ndarray, pairs: list[dict], expected: np.ndarray) -> bool:
    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False

    contours, blue_map = _extract_contours(grid)
    mirrors, _mirror_map = _extract_mirrors(grid)
    sources = np.argwhere((grid != BG) & (grid != BLUE) & (grid != MIRROR))

    if len(sources) != len(pairs) or len(mirrors) != len(pairs) or len(contours) != len(pairs):
        return False
    if len({tuple(map(int, p)) for p in sources}) != len(pairs):
        return False

    source_lookup = {tuple(pair["source"]): pair for pair in pairs}
    seen_targets: set[int] = set()
    for contour in contours:
        area = int(contour["interior_mask"].sum())
        if not (2 <= area <= 6):
            return False

    for mirror in mirrors:
        if not (5 <= mirror["length"] <= 7):
            return False

    for src in sources:
        key = tuple(map(int, src))
        if key not in source_lookup:
            return False
        planned = source_lookup[key]
        traced = _trace_ray(grid, contours, blue_map, key[0], key[1])
        if traced["dc"] != planned["dir"]:
            return False
        if traced["mirror_row"] != planned["mirror_row"]:
            return False
        if traced["boundary_c"] != (planned["loop_c1"] if planned["dir"] == -1 else planned["loop_c0"]):
            return False
        targets = _accessible_targets(grid, contours, blue_map, traced["mirror_row"], key[1], traced["dc"])
        if len(targets) != 1:
            return False
        if traced["target_id"] in seen_targets:
            return False
        seen_targets.add(traced["target_id"])

    actual = main(grid)
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    count = random.choice((2, 3))
    h = random.randint(18, 24) if count == 2 else random.randint(22, 26)
    w = random.randint(20, 26)
    mirror_rows = _sample_spaced_rows(7, h - 5, count, gap=5)
    if mirror_rows is None:
        return None

    dirs = _choose_directions(count)
    colors = random.sample(SOURCE_COLORS, count)
    grid = np.zeros((h, w), dtype=int)
    used_source_cols: set[int] = set()
    mirror_intervals: list[tuple[int, int]] = []
    pairs: list[dict] = []

    for row, direction, color in zip(mirror_rows, dirs, colors):
        placed = False
        for _ in range(200):
            loop_h, loop_w = random.choice(LOOP_TEMPLATES)
            entry_offset = 1 if loop_h == 3 else random.choice((1, 2))
            loop_r0 = row - entry_offset
            if loop_r0 < 5 or loop_r0 + loop_h > h - 1:
                continue
            loop_box = (loop_r0, 0, loop_r0 + loop_h - 1, 0)

            mirror_len = random.randint(5, 7)
            midpoint = (mirror_len - 1) // 2
            if direction == -1:
                left_hits = list(range(midpoint + 1))
                hit_index = midpoint if random.random() < 0.4 else random.choice(left_hits)
                max_loop_c0 = w // 2 - loop_w - 2
                if max_loop_c0 < 1:
                    continue
                loop_c0 = random.randint(1, max_loop_c0)
                loop_c1 = loop_c0 + loop_w - 1
                loop_box = (loop_r0, loop_c0, loop_r0 + loop_h - 1, loop_c1)
                src_min = max(w // 2, loop_c1 + 3 + hit_index)
                src_max = w - mirror_len + hit_index - 1
            else:
                right_hits = list(range(midpoint + 1, mirror_len))
                if not right_hits:
                    continue
                hit_index = random.choice(right_hits)
                min_loop_c0 = w // 2 + 2
                max_loop_c0 = w - loop_w - 2
                if min_loop_c0 > max_loop_c0:
                    continue
                loop_c0 = random.randint(min_loop_c0, max_loop_c0)
                loop_c1 = loop_c0 + loop_w - 1
                loop_box = (loop_r0, loop_c0, loop_r0 + loop_h - 1, loop_c1)
                src_min = 1 + hit_index
                src_max = min(w // 2 - 1, loop_c0 - mirror_len + hit_index - 2)

            if src_min > src_max:
                continue
            if any(
                _bbox_distance(loop_box, (pair["loop_r0"], pair["loop_c0"], pair["loop_r0"] + pair["loop_h"] - 1, pair["loop_c1"])) < 2
                for pair in pairs
            ):
                continue

            source_c = random.randint(src_min, src_max)
            mirror_c0 = source_c - hit_index
            mirror_c1 = mirror_c0 + mirror_len - 1
            if source_c in used_source_cols:
                continue
            if any(pair["loop_c0"] <= source_c <= pair["loop_c1"] for pair in pairs):
                continue
            if any(c0 <= source_c <= c1 for c0, c1 in mirror_intervals):
                continue
            if any(mirror_c0 <= old_source <= mirror_c1 for old_source in used_source_cols):
                continue

            source_r = random.randint(1, min(4, row - 4))
            grid[source_r, source_c] = color
            grid[row, mirror_c0 : mirror_c1 + 1] = MIRROR
            grid[loop_r0, loop_c0 : loop_c1 + 1] = BLUE
            grid[loop_r0 + loop_h - 1, loop_c0 : loop_c1 + 1] = BLUE
            grid[loop_r0 : loop_r0 + loop_h, loop_c0] = BLUE
            grid[loop_r0 : loop_r0 + loop_h, loop_c1] = BLUE

            used_source_cols.add(source_c)
            mirror_intervals.append((mirror_c0, mirror_c1))
            pairs.append(
                {
                    "source": (source_r, source_c),
                    "color": color,
                    "mirror_row": row,
                    "mirror_c0": mirror_c0,
                    "mirror_c1": mirror_c1,
                    "dir": direction,
                    "loop_r0": loop_r0,
                    "loop_c0": loop_c0,
                    "loop_c1": loop_c1,
                    "loop_h": loop_h,
                    "loop_w": loop_w,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _build_expected(grid, pairs)
    if not _validate_generated(grid, pairs, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-ray fill input")
