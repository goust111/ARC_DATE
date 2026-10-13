import random
from collections import deque

import numpy as np


BG = 0
TRACK = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATE_ATTEMPTS = 400
MAX_HOLES = 3
TRACK_COUNT = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components_mask(mask)) == 1


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    holes = 0
    for comp in _components_mask(bg_mask):
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp):
            holes += 1
    return holes


def _is_boundary_pixel(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    if not mask[r, c]:
        return False
    for nr, nc in _neighbors4(r, c):
        if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
            return True
    return False


def _random_blob_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        target = random.randint(max(5, (h * w) // 2), h * w)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if _is_connected(mask) and _count_holes(mask) == 0:
            return mask
    raise RuntimeError("failed to generate zero-hole object")


def _random_holey_mask(holes: int) -> np.ndarray:
    assert 1 <= holes <= MAX_HOLES
    for _ in range(300):
        h = random.randint(max(5, holes + 3), 7)
        w = random.randint(max(5, holes + 4), 8)
        mask = np.ones((h, w), dtype=bool)
        candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for cell in candidates:
            if all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in chosen):
                chosen.append(cell)
            if len(chosen) == holes:
                break
        if len(chosen) != holes:
            continue
        for r, c in chosen:
            mask[r, c] = False

        for _ in range(random.randint(0, 4)):
            boundary = [(r, c) for r in range(h) for c in range(w) if _is_boundary_pixel(mask, r, c)]
            random.shuffle(boundary)
            updated = False
            for r, c in boundary:
                if mask.sum() <= holes + 5:
                    break
                trial = mask.copy()
                trial[r, c] = False
                if _is_connected(trial) and _count_holes(trial) == holes:
                    mask = trial
                    updated = True
                    break
            if not updated:
                break

        mask = _trim_mask(mask)
        if _is_connected(mask) and _count_holes(mask) == holes:
            k = random.randrange(4)
            if random.random() < 0.5:
                mask = np.fliplr(mask)
            if random.random() < 0.5:
                mask = np.flipud(mask)
            return np.rot90(mask, k)
    raise RuntimeError("failed to generate holey object")


def _generate_object_mask(holes: int) -> np.ndarray:
    if holes == 0:
        mask = _random_blob_mask()
    else:
        mask = _random_holey_mask(holes)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return np.rot90(mask, random.randrange(4))


def _perimeter_clockwise(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        pts.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def _ray_start_pixel(
    bbox: tuple[int, int, int, int],
    center: tuple[float, float],
    marker: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    cr, cc = center
    mr, mc = marker
    dy = mr - cr
    dx = mc - cc
    assert dy != 0 or dx != 0

    candidates: list[tuple[float, str, float]] = []
    if dy < 0:
        t = (r0 - cr) / dy
        c = cc + t * dx
        if t > 0 and c0 <= c <= c1:
            candidates.append((t, "top", c))
    if dy > 0:
        t = (r1 - cr) / dy
        c = cc + t * dx
        if t > 0 and c0 <= c <= c1:
            candidates.append((t, "bottom", c))
    if dx < 0:
        t = (c0 - cc) / dx
        r = cr + t * dy
        if t > 0 and r0 <= r <= r1:
            candidates.append((t, "left", r))
    if dx > 0:
        t = (c1 - cc) / dx
        r = cr + t * dy
        if t > 0 and r0 <= r <= r1:
            candidates.append((t, "right", r))

    assert candidates, "marker ray must hit track"
    t, side, other = min(candidates, key=lambda item: item[0])
    if side == "top":
        return r0, max(c0, min(c1, int(round(other))))
    if side == "bottom":
        return r1, max(c0, min(c1, int(round(other))))
    if side == "left":
        return max(r0, min(r1, int(round(other)))), c0
    return max(r0, min(r1, int(round(other)))), c1


def _equally_spaced_positions(
    perimeter: list[tuple[int, int]],
    start: tuple[int, int],
    count: int,
) -> list[tuple[int, int]]:
    length = len(perimeter)
    start_idx = perimeter.index(start)
    if count == 1:
        return [start]
    if length % count == 0:
        step = length // count
        return [perimeter[(start_idx + i * step) % length] for i in range(count)]

    positions = []
    used = set()
    for i in range(count):
        offset = int(round(i * length / count)) % length
        pos = perimeter[(start_idx + offset) % length]
        if pos in used:
            raise AssertionError("failed to place equally spaced copies uniquely")
        positions.append(pos)
        used.add(pos)
    return positions


def _split_object_and_marker(
    grid: np.ndarray,
    color: int,
    center: tuple[float, float],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    mask = grid == color
    comps = _components_mask(mask)
    singletons = [comp for comp in comps if len(comp) == 1]
    nonsingle = [comp for comp in comps if len(comp) > 1]
    if len(singletons) == 1 and len(nonsingle) == 1:
        return singletons[0][0], nonsingle[0]

    if len(comps) == 1:
        coords = comps[0]
        ordered = sorted(
            coords,
            key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2,
        )
        for marker in ordered:
            trial = mask.copy()
            trial[marker] = False
            if not trial.any():
                continue
            if len(_components_mask(trial)) == 1:
                obj_coords = [(r, c) for r, c in coords if (r, c) != marker]
                return marker, obj_coords

    raise AssertionError("each color must contain one object and one marker")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    track_info = sorted(
        ((comp, _bbox(comp)) for comp in _components_mask(g == TRACK)),
        key=lambda item: (item[1][2] - item[1][0] + 1) * (item[1][3] - item[1][1] + 1),
    )
    assert track_info, "expected grey tracks"
    track_bboxes = [bbox for _, bbox in track_info]
    for comp, bbox in track_info:
        expected = set(_perimeter_clockwise(bbox))
        assert expected == set(comp), "grey component must be a 1-pixel rectangular ring"

    outer = track_bboxes[-1]
    center = ((outer[0] + outer[2]) / 2.0, (outer[1] + outer[3]) / 2.0)
    perimeters = [_perimeter_clockwise(bbox) for bbox in track_bboxes]

    out = g.copy()
    out[(g != BG) & (g != TRACK)] = BG

    colors = [int(c) for c in np.unique(g) if c not in (BG, TRACK)]
    for color in colors:
        marker, obj_coords = _split_object_and_marker(g, color, center)
        r0, c0, r1, c1 = _bbox(obj_coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in obj_coords:
            mask[r - r0, c - c0] = True
        holes = _count_holes(mask)
        track_index = 1 if holes == 0 else holes
        assert 1 <= track_index <= len(track_bboxes)
        start = _ray_start_pixel(track_bboxes[track_index - 1], center, marker)
        positions = _equally_spaced_positions(perimeters[track_index - 1], start, holes + 1)
        for pr, pc in positions:
            out[pr, pc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _sample_track_inner_size() -> tuple[int, int]:
    return random.choice([(5, 7), (6, 6), (7, 5)])


def _draw_tracks(grid: np.ndarray, outer_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = outer_bbox
    bboxes = []
    for i in range(TRACK_COUNT):
        bbox = (r0 + 2 * (TRACK_COUNT - 1 - i), c0 + 2 * (TRACK_COUNT - 1 - i), r1 - 2 * (TRACK_COUNT - 1 - i), c1 - 2 * (TRACK_COUNT - 1 - i))
        bboxes.append(bbox)
        rr0, cc0, rr1, cc1 = bbox
        grid[rr0, cc0 : cc1 + 1] = TRACK
        grid[rr1, cc0 : cc1 + 1] = TRACK
        grid[rr0 : rr1 + 1, cc0] = TRACK
        grid[rr0 : rr1 + 1, cc1] = TRACK
    return list(reversed(bboxes))


def _line_points(a: tuple[float, float], b: tuple[float, float]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    steps = int(max(abs(br - ar), abs(bc - ac)) * 6) + 2
    pts: list[tuple[int, int]] = []
    seen = set()
    for i in range(steps + 1):
        t = i / steps
        r = int(round(ar + (br - ar) * t))
        c = int(round(ac + (bc - ac) * t))
        if (r, c) not in seen:
            pts.append((r, c))
            seen.add((r, c))
    return pts


def _centroid(cells: list[tuple[int, int]]) -> tuple[float, float]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return sum(rows) / len(rows), sum(cols) / len(cols)


def _choose_marker(
    object_cells: list[tuple[int, int]],
    center: tuple[float, float],
    outer_bbox: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = outer_bbox
    centroid = _centroid(object_cells)
    object_set = set(object_cells)
    candidates = _line_points(centroid, center)
    valid: list[tuple[int, int]] = []
    for cell in candidates:
        rr, cc = cell
        if cell in object_set or cell in occupied:
            continue
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        valid.append(cell)
    if not valid:
        return None
    return min(valid, key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)


def _place_object(
    grid: np.ndarray,
    color: int,
    mask: np.ndarray,
    corner: str,
    outer_bbox: tuple[int, int, int, int],
    center: tuple[float, float],
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    or0, oc0, or1, oc1 = outer_bbox

    if corner == "tl":
        row_range = range(1, or0 - mh)
        col_range = range(1, oc0 - mw)
    elif corner == "tr":
        row_range = range(1, or0 - mh)
        col_range = range(oc1 + 2, w - mw - 1)
    elif corner == "bl":
        row_range = range(or1 + 2, h - mh - 1)
        col_range = range(1, oc0 - mw)
    else:
        row_range = range(or1 + 2, h - mh - 1)
        col_range = range(oc1 + 2, w - mw - 1)

    starts = [(r, c) for r in row_range for c in col_range]
    random.shuffle(starts)
    for sr, sc in starts:
        cells = [(sr + r, sc + c) for r, c in zip(*np.where(mask))]
        if any(cell in occupied for cell in cells):
            continue
        marker = _choose_marker(cells, center, outer_bbox, occupied, grid.shape)
        if marker is None:
            continue
        if marker in occupied:
            continue
        for r, c in cells:
            grid[r, c] = color
            occupied.add((r, c))
        grid[marker] = color
        occupied.add(marker)
        return marker, cells
    return None


def _track_positions_for_validation(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    out = _solve(grid)
    positions: dict[int, list[tuple[int, int]]] = {}
    for color in [int(c) for c in np.unique(grid) if c not in (BG, TRACK)]:
        positions[color] = [tuple(p) for p in np.argwhere(out == color)]
    return positions


def _valid_generated_instance(grid: np.ndarray) -> bool:
    try:
        positions = _track_positions_for_validation(grid)
    except AssertionError:
        return False

    all_positions = [pos for pts in positions.values() for pos in pts]
    if len(all_positions) != len(set(all_positions)):
        return False

    track_components = _components_mask(grid == TRACK)
    track_bboxes = sorted(
        (_bbox(comp) for comp in track_components),
        key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1),
    )
    outer = track_bboxes[-1]
    center = ((outer[0] + outer[2]) / 2.0, (outer[1] + outer[3]) / 2.0)
    colors = [int(c) for c in np.unique(grid) if c not in (BG, TRACK)]
    for color in colors:
        marker, obj_coords = _split_object_and_marker(grid, color, center)
        r0, c0, r1, c1 = _bbox(obj_coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in obj_coords:
            mask[r - r0, c - c0] = True
        holes = _count_holes(mask)
        track_index = 1 if holes == 0 else holes
        start = _ray_start_pixel(track_bboxes[track_index - 1], center, marker)
        if start in {
            (track_bboxes[track_index - 1][0], track_bboxes[track_index - 1][1]),
            (track_bboxes[track_index - 1][0], track_bboxes[track_index - 1][3]),
            (track_bboxes[track_index - 1][2], track_bboxes[track_index - 1][1]),
            (track_bboxes[track_index - 1][2], track_bboxes[track_index - 1][3]),
        }:
            return False
    return True


def _generate_hole_counts() -> list[int]:
    while True:
        holes = [random.randint(0, MAX_HOLES) for _ in range(4)]
        if max(holes) >= 2 and len(set(holes)) >= 2:
            return holes


def _gen_attempt() -> np.ndarray | None:
    hole_counts = _generate_hole_counts()
    masks = [_generate_object_mask(h) for h in hole_counts]

    inner_h, inner_w = _sample_track_inner_size()
    outer_h = inner_h + 4 * (TRACK_COUNT - 1)
    outer_w = inner_w + 4 * (TRACK_COUNT - 1)

    margin_y = max(max(mask.shape[0] for mask in masks), 6) + random.randint(2, 4)
    margin_x = max(max(mask.shape[1] for mask in masks), 6) + random.randint(2, 4)
    grid_h = outer_h + 2 * margin_y
    grid_w = outer_w + 2 * margin_x
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    outer_bbox = (margin_y, margin_x, margin_y + outer_h - 1, margin_x + outer_w - 1)
    _draw_tracks(grid, outer_bbox)
    center = ((outer_bbox[0] + outer_bbox[2]) / 2.0, (outer_bbox[1] + outer_bbox[3]) / 2.0)

    colors = random.sample(OBJECT_COLORS, 4)
    corners = ["tl", "tr", "bl", "br"]
    occupied = set(tuple(p) for p in np.argwhere(grid == TRACK))

    for color, mask, corner in zip(colors, masks, corners):
        placement = _place_object(grid, color, mask, corner, outer_bbox, center, occupied)
        if placement is None:
            return None

    if not _valid_generated_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
