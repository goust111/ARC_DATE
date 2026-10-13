import random
from collections import deque

import numpy as np


BG = 0
SOURCE = 2
MARKER = 3
OBSTACLE = 7

QUADRANTS = ("TL", "TR", "BL", "BR")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_RANGE = (13, 19)
MARKER_MARGIN = 4
SOURCE_SIZE_RANGE = (4, 8)
SOURCE_BOX_RANGE = (3, 5)
BAR_LENGTH_RANGE = (3, 6)


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        cells = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        components.append(np.array(cells, dtype=int))
    return components


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MARKER)
    assert len(coords) == 4, "expected exactly four marker pixels"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert (r1 - r0, c1 - c0) == (1, 1), "marker must be a 2x2 square"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == MARKER), "marker square must be solid"
    return r0, c0


def _quadrant_of(r: int, c: int, mr: int, mc: int) -> str | None:
    if r < mr and c < mc:
        return "TL"
    if r < mr and c > mc + 1:
        return "TR"
    if r > mr + 1 and c < mc:
        return "BL"
    if r > mr + 1 and c > mc + 1:
        return "BR"
    return None


def _reflect_targets(r: int, c: int, mr: int, mc: int) -> list[tuple[int, int]]:
    rr = 2 * mr + 1 - r
    cc = 2 * mc + 1 - c
    return [(r, cc), (rr, c), (rr, cc)]


def _ray_path(mr: int, mc: int, target: tuple[int, int]) -> list[tuple[int, int]]:
    """Discrete sight line from the 2x2 marker toward a target quadrant cell.

    The ray family is quantized to the closest of:
    - vertical: same target column
    - horizontal: same target row
    - 45-degree diagonal: same gap in row/col from the marker corner
    """
    r, c = map(int, target)
    quadrant = _quadrant_of(r, c, mr, mc)
    assert quadrant is not None, "target must lie in a strict quadrant"

    vr = -1 if r < mr else 1
    vc = -1 if c < mc else 1
    row_gap = mr - r if r < mr else r - (mr + 1)
    col_gap = mc - c if c < mc else c - (mc + 1)
    assert row_gap >= 1 and col_gap >= 1

    if row_gap == col_gap:
        rr = mr - 1 if vr < 0 else mr + 2
        cc = mc - 1 if vc < 0 else mc + 2
        path = []
        while True:
            path.append((rr, cc))
            if (rr, cc) == (r, c):
                return path
            rr += vr
            cc += vc

    if row_gap > col_gap:
        rr = mr - 1 if vr < 0 else mr + 2
        path = []
        while True:
            path.append((rr, c))
            if rr == r:
                return path
            rr += vr

    cc = mc - 1 if vc < 0 else mc + 2
    path = []
    while True:
        path.append((r, cc))
        if cc == c:
            return path
        cc += vc


def _reachable(grid: np.ndarray, mr: int, mc: int, target: tuple[int, int]) -> bool:
    return all(int(grid[r, c]) != OBSTACLE for r, c in _ray_path(mr, mc, target)[:-1])


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the mirror copies of the red shape, but only where the marker-centered sight line is not blocked by orange bars.

    Input structure:
    - The background is black.
    - A unique solid 2x2 green square marks the mirror center.
    - A red source shape appears in exactly one strict quadrant around the marker, while orange one-cell-wide bars occupy the other quadrants.

    Rule:
    - Reflect every red source pixel across the marker's horizontal axis, vertical axis, and both axes together.
    - For each reflected candidate, trace a quantized straight sight line from the marker center toward that target using the closest horizontal, vertical, or 45-degree diagonal family.
    - Paint the reflected pixel red iff that sight line reaches the target without crossing an orange obstacle before the endpoint; keep the marker, the original red shape, and the orange bars unchanged.

    Key insight:
    - The green square defines all three mirror candidates, but the orange bars decide which of those candidates are actually visible from the center, so symmetry completion is filtered by local line-of-sight.
    """
    g = np.asarray(grid, dtype=int)
    mr, mc = _find_marker(g)
    source_coords = [(int(r), int(c)) for r, c in np.argwhere(g == SOURCE)]
    source_quadrants = {
        _quadrant_of(r, c, mr, mc)
        for r, c in source_coords
    }
    assert None not in source_quadrants and len(source_quadrants) == 1, "source must occupy one quadrant"

    out = g.copy()
    for r, c in source_coords:
        for target in _reflect_targets(r, c, mr, mc):
            tr, tc = target
            if int(g[tr, tc]) == MARKER:
                continue
            if _reachable(g, mr, mc, target):
                out[tr, tc] = SOURCE
    return out


def _random_connected_mask(height: int, width: int, count: int) -> np.ndarray | None:
    for _ in range(100):
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True
        cells = {start}
        frontier = set()
        frontier.update(
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < height and 0 <= nc < width
        )
        while len(cells) < count and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            mask[cell] = True
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != count:
            continue
        rr, cc = np.argwhere(mask).T
        if (rr.max() - rr.min() + 1) < 2 or (cc.max() - cc.min() + 1) < 2:
            continue
        if count == height * width:
            continue
        return mask
    return None


def _source_bbox(quadrant: str, mr: int, mc: int, h: int, w: int, gap_r: int, gap_c: int) -> tuple[int, int]:
    if quadrant == "TL":
        return mr - gap_r - h, mc - gap_c - w
    if quadrant == "TR":
        return mr - gap_r - h, mc + 2 + gap_c
    if quadrant == "BL":
        return mr + 2 + gap_r, mc - gap_c - w
    return mr + 2 + gap_r, mc + 2 + gap_c


def _quadrant_bounds(quadrant: str, h: int, w: int, mr: int, mc: int) -> tuple[int, int, int, int]:
    if quadrant == "TL":
        return 0, mr - 1, 0, mc - 1
    if quadrant == "TR":
        return 0, mr - 1, mc + 2, w - 1
    if quadrant == "BL":
        return mr + 2, h - 1, 0, mc - 1
    return mr + 2, h - 1, mc + 2, w - 1


def _ray_family(mr: int, mc: int, cell: tuple[int, int]) -> str:
    r, c = cell
    row_gap = mr - r if r < mr else r - (mr + 1)
    col_gap = mc - c if c < mc else c - (mc + 1)
    if row_gap == col_gap:
        return "diag"
    if row_gap > col_gap:
        return "vert"
    return "horiz"


def _build_candidate_map(source_cells: list[tuple[int, int]], mr: int, mc: int) -> dict[str, list[dict]]:
    out = {q: [] for q in QUADRANTS}
    for src in source_cells:
        for target in _reflect_targets(src[0], src[1], mr, mc):
            q = _quadrant_of(target[0], target[1], mr, mc)
            assert q is not None
            out[q].append(
                {
                    "src": src,
                    "target": target,
                    "path": tuple(_ray_path(mr, mc, target)),
                    "family": _ray_family(mr, mc, target),
                }
            )
    return out


def _enumerate_bars(
    quadrant: str,
    grid_shape: tuple[int, int],
    mr: int,
    mc: int,
    forbidden: set[tuple[int, int]],
    candidates: list[dict],
) -> list[dict]:
    h, w = grid_shape
    r0, r1, c0, c1 = _quadrant_bounds(quadrant, h, w, mr, mc)
    bars = []
    seen = set()
    candidate_paths = [set(item["path"][:-1]) for item in candidates]
    for orient in ("H", "V"):
        max_len = min(BAR_LENGTH_RANGE[1], (c1 - c0 + 1) if orient == "H" else (r1 - r0 + 1))
        for length in range(BAR_LENGTH_RANGE[0], max_len + 1):
            if orient == "H":
                for r in range(r0, r1 + 1):
                    for start_c in range(c0, c1 - length + 2):
                        cells = tuple((r, start_c + k) for k in range(length))
                        if any(cell in forbidden for cell in cells):
                            continue
                        blocks = frozenset(i for i, path in enumerate(candidate_paths) if any(cell in path for cell in cells))
                        if 1 <= len(blocks) < len(candidates):
                            key = (cells, blocks)
                            if key not in seen:
                                seen.add(key)
                                bars.append({"cells": cells, "blocked": blocks})
            else:
                for c in range(c0, c1 + 1):
                    for start_r in range(r0, r1 - length + 2):
                        cells = tuple((start_r + k, c) for k in range(length))
                        if any(cell in forbidden for cell in cells):
                            continue
                        blocks = frozenset(i for i, path in enumerate(candidate_paths) if any(cell in path for cell in cells))
                        if 1 <= len(blocks) < len(candidates):
                            key = (cells, blocks)
                            if key not in seen:
                                seen.add(key)
                                bars.append({"cells": cells, "blocked": blocks})
    random.shuffle(bars)
    return bars


def _bars_compatible(chosen: list[dict], bar: dict) -> bool:
    occupied = set()
    halo = set()
    for item in chosen:
        for r, c in item["cells"]:
            occupied.add((r, c))
            halo.add((r, c))
            halo.update(_neighbors4(r, c))
    return not any(cell in occupied or cell in halo for cell in bar["cells"])


def _select_bars_for_quadrant(
    bars: list[dict],
    candidate_count: int,
) -> list[dict] | None:
    for first in bars:
        blocked = set(first["blocked"])
        if len(blocked) < candidate_count:
            selected = [first]
            if random.random() < 0.45:
                for second in bars:
                    if second is first or not _bars_compatible(selected, second):
                        continue
                    union = blocked | set(second["blocked"])
                    if 1 <= len(union) < candidate_count:
                        selected.append(second)
                        blocked = union
                        break
            if 1 <= len(blocked) < candidate_count:
                return selected
    return None


def _validate_obstacle_components(grid: np.ndarray, source_quadrant: str, mr: int, mc: int) -> bool:
    comps = _component_coords(grid == OBSTACLE)
    if len(comps) < 3:
        return False
    for comp in comps:
        rs = comp[:, 0]
        cs = comp[:, 1]
        h = int(rs.max() - rs.min() + 1)
        w = int(cs.max() - cs.min() + 1)
        if not ((h == 1 and BAR_LENGTH_RANGE[0] <= w <= BAR_LENGTH_RANGE[1]) or (w == 1 and BAR_LENGTH_RANGE[0] <= h <= BAR_LENGTH_RANGE[1])):
            return False
        for r, c in comp:
            if _quadrant_of(int(r), int(c), mr, mc) == source_quadrant:
                return False
    return True


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    mr, mc = _find_marker(grid)
    reds = [(int(r), int(c)) for r, c in np.argwhere(grid == SOURCE)]
    if not (SOURCE_SIZE_RANGE[0] <= len(reds) <= SOURCE_SIZE_RANGE[1]):
        return False
    red_quads = {_quadrant_of(r, c, mr, mc) for r, c in reds}
    if None in red_quads or len(red_quads) != 1:
        return False
    if len(_component_coords(grid == SOURCE)) != 1:
        return False
    source_quadrant = next(iter(red_quads))
    if not _validate_obstacle_components(grid, source_quadrant, mr, mc):
        return False

    candidates = _build_candidate_map(reds, mr, mc)
    for quadrant in QUADRANTS:
        if quadrant == source_quadrant:
            continue
        blocked = 0
        clear = 0
        for item in candidates[quadrant]:
            if _reachable(grid, mr, mc, item["target"]):
                clear += 1
            else:
                blocked += 1
        if blocked == 0 or clear == 0:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    full_added = 3 * len(reds)
    added = int(np.count_nonzero((out == SOURCE) & (grid != SOURCE)))
    if not (1 <= added < full_added):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    mr = random.randint(MARKER_MARGIN, height - MARKER_MARGIN - 2)
    mc = random.randint(MARKER_MARGIN, width - MARKER_MARGIN - 2)

    source_quadrant = random.choice(QUADRANTS)
    vertical_room = min(mr, height - (mr + 2))
    horizontal_room = min(mc, width - (mc + 2))
    row_options = [
        (box_h, gap_r)
        for box_h in range(SOURCE_BOX_RANGE[0], min(SOURCE_BOX_RANGE[1], vertical_room) + 1)
        for gap_r in range(0, min(2, vertical_room - box_h) + 1)
    ]
    col_options = [
        (box_w, gap_c)
        for box_w in range(SOURCE_BOX_RANGE[0], min(SOURCE_BOX_RANGE[1], horizontal_room) + 1)
        for gap_c in range(0, min(2, horizontal_room - box_w) + 1)
    ]
    if not row_options or not col_options:
        return None
    box_h, gap_r = random.choice(row_options)
    box_w, gap_c = random.choice(col_options)
    top, left = _source_bbox(source_quadrant, mr, mc, box_h, box_w, gap_r, gap_c)
    if top < 0 or left < 0 or top + box_h > height or left + box_w > width:
        return None

    source_count = random.randint(SOURCE_SIZE_RANGE[0], min(SOURCE_SIZE_RANGE[1], box_h * box_w - 1))
    mask = _random_connected_mask(box_h, box_w, source_count)
    if mask is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[mr : mr + 2, mc : mc + 2] = MARKER
    source_cells = []
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        if _quadrant_of(r, c, mr, mc) != source_quadrant:
            return None
        source_cells.append((r, c))
        grid[r, c] = SOURCE

    families = {_ray_family(mr, mc, cell) for cell in source_cells}
    if len(families) < 2:
        return None

    candidate_map = _build_candidate_map(source_cells, mr, mc)
    endpoint_forbidden = set(source_cells)
    endpoint_forbidden.update((r, c) for r in range(mr, mr + 2) for c in range(mc, mc + 2))
    for quadrant in QUADRANTS:
        for item in candidate_map[quadrant]:
            endpoint_forbidden.add(item["target"])

    selected_bars = []
    target_quadrants = [q for q in QUADRANTS if q != source_quadrant]
    random.shuffle(target_quadrants)
    for quadrant in target_quadrants:
        bars = _enumerate_bars(quadrant, grid.shape, mr, mc, endpoint_forbidden, candidate_map[quadrant])
        chosen = _select_bars_for_quadrant(bars, len(candidate_map[quadrant]))
        if chosen is None:
            return None
        selected_bars.extend(chosen)
        for item in chosen:
            endpoint_forbidden.update(item["cells"])

    # Paint bars after all quadrant choices. Different quadrants are already separated by the marker bands.
    for bar in selected_bars:
        for r, c in bar["cells"]:
            grid[r, c] = OBSTACLE

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid visibility-gated symmetry puzzle")
