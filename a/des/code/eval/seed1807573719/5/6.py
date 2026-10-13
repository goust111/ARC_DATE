import random
from collections import Counter

import numpy as np


BG = 0
SOURCE = 1
GRAY = 5
DYE = 7
PAINT_CHOICES = (2, 3, 4, 6, 8, 9)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 300
REGION_PLACE_RETRIES = 80
MARKER_MIN_SEGMENT = 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0)


def _ordered_wall_cells_ccw(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    edges: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or not mask[r - 1, c]:
                edges[(r, c + 1)] = ((r, c), (r, c))
            if c == w - 1 or not mask[r, c + 1]:
                edges[(r + 1, c + 1)] = ((r, c + 1), (r, c))
            if r == h - 1 or not mask[r + 1, c]:
                edges[(r + 1, c)] = ((r + 1, c + 1), (r, c))
            if c == 0 or not mask[r, c - 1]:
                edges[(r, c)] = ((r + 1, c), (r, c))

    assert edges, "region must have a non-empty wall contour"

    start = min(edges)
    cur = start
    visited = set()
    walk: list[tuple[int, int]] = []
    while True:
        assert cur in edges, "wall contour must be closed"
        assert cur not in visited, "wall contour must be a single simple loop"
        visited.add(cur)
        nxt, cell = edges[cur]
        walk.append(cell)
        cur = nxt
        if cur == start:
            break

    assert len(visited) == len(edges), "region wall should have exactly one contour"

    cycle: list[tuple[int, int]] = []
    for cell in walk:
        if not cycle or cycle[-1] != cell:
            cycle.append(cell)
    if len(cycle) > 1 and cycle[0] == cycle[-1]:
        cycle.pop()
    assert cycle, "compressed wall cycle must be non-empty"
    return cycle


def _diffusion_direction(source: tuple[int, int], dye: tuple[int, int]) -> str:
    sr, sc = source
    dr, dc = dye
    clockwise = dr < sr or dc < sc
    counterclockwise = dr > sr or dc > sc
    assert clockwise != counterclockwise, "dye position relative to source is ambiguous"
    return "cw" if clockwise else "ccw"


def _rotate_cycle(cycle: list[tuple[int, int]], start_cell: tuple[int, int]) -> list[tuple[int, int]]:
    idx = cycle.index(start_cell)
    return cycle[idx:] + cycle[:idx]


def _color_region(
    out: np.ndarray,
    region_coords: list[tuple[int, int]],
    source: tuple[int, int],
    dye: tuple[int, int],
    fill_a: int,
    fill_b: int,
) -> None:
    mask, (r0, c0) = _coords_to_mask(region_coords)
    cycle_local = _ordered_wall_cells_ccw(mask)
    cycle = [(r + r0, c + c0) for r, c in cycle_local]
    counts = Counter(cycle)
    assert counts[source] == 1 and counts[dye] == 1, "markers must lie on unique wall positions"

    direction = _diffusion_direction(source, dye)
    if direction == "cw":
        cycle = list(reversed(cycle))

    cycle = _rotate_cycle(cycle, source)
    dye_idx = cycle.index(dye)
    assert 1 <= dye_idx < len(cycle), "dye must be distinct from source on the wall cycle"

    for cell in cycle[1:dye_idx]:
        r, c = cell
        assert int(out[r, c]) == BG
        out[r, c] = fill_a
    for cell in cycle[dye_idx + 1:]:
        r, c = cell
        assert int(out[r, c]) == BG
        out[r, c] = fill_b


def _find_palette_and_regions(grid: np.ndarray) -> tuple[tuple[int, int], list[list[tuple[int, int]]]]:
    h, w = grid.shape
    enclosed = [
        comp
        for comp in _components(grid != GRAY)
        if not _touches_border(comp, h, w)
    ]
    assert enclosed, "expected enclosed non-gray regions"

    palette_idx = None
    palette_colors = None
    palette_score = None
    for idx, comp in enumerate(enclosed):
        mask, (r0, c0) = _coords_to_mask(comp)
        if mask.shape != (2, 2) or not mask.all():
            continue
        patch = grid[r0:r0 + 2, c0:c0 + 2]
        if int(patch[0, 0]) != SOURCE or int(patch[0, 1]) != DYE:
            continue
        if int(patch[1, 0]) in {BG, SOURCE, GRAY, DYE}:
            continue
        if int(patch[1, 1]) in {BG, SOURCE, GRAY, DYE}:
            continue
        if int(patch[1, 0]) == int(patch[1, 1]):
            continue
        score = r0 + c0
        if palette_score is None or score > palette_score:
            palette_idx = idx
            palette_score = score
            palette_colors = (int(patch[1, 0]), int(patch[1, 1]))

    assert palette_idx is not None and palette_colors is not None, "palette must exist"
    regions = [comp for idx, comp in enumerate(enclosed) if idx != palette_idx]
    assert regions, "expected at least one enclosed room"
    return palette_colors, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each enclosed room's inner wall in two segments using the palette legend in the bottom-right.

    The palette is a gray-bordered 2x2 interior whose first row is [blue source, orange dye]
    and whose second row is the two output colors. Each room contains exactly one blue source
    and one orange dye on the wall ring. Dye above/left of the source means clockwise; dye
    below/right means counterclockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (fill_a, fill_b), regions = _find_palette_and_regions(g)
    out = g.copy()
    for comp in regions:
        vals = [int(g[r, c]) for r, c in comp]
        assert vals.count(SOURCE) == 1 and vals.count(DYE) == 1, "each room needs one source and one dye"
        source = next((r, c) for r, c in comp if int(g[r, c]) == SOURCE)
        dye = next((r, c) for r, c in comp if int(g[r, c]) == DYE)
        _color_region(out, comp, source, dye, fill_a, fill_b)
    return out


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _candidate_marker_pairs(
    wall_cycle: list[tuple[int, int]],
    direction: str,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    counts = Counter(wall_cycle)
    unique_cells = [cell for cell in wall_cycle if counts[cell] == 1]
    seq = wall_cycle if direction == "ccw" else list(reversed(wall_cycle))
    positions = {cell: idx for idx, cell in enumerate(seq)}

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for source in unique_cells:
        sr, sc = source
        for dye in unique_cells:
            if dye == source:
                continue
            dr, dc = dye
            if sr != dr and sc != dc:
                continue
            if _diffusion_direction(source, dye) != direction:
                continue
            s_idx = positions[source]
            d_idx = positions[dye]
            if d_idx <= s_idx:
                continue
            seg1 = d_idx - s_idx - 1
            seg2 = len(seq) - d_idx - 1
            if min(seg1, seg2) < MARKER_MIN_SEGMENT:
                continue
            pairs.append((source, dye))
    return pairs


def _place_palette(grid: np.ndarray, fill_a: int, fill_b: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    r0, c0 = h - 4, w - 4
    grid[r0:r0 + 4, c0:c0 + 4] = GRAY
    grid[r0 + 1:r0 + 3, c0 + 1:c0 + 3] = np.array(
        [[SOURCE, DYE], [fill_a, fill_b]],
        dtype=int,
    )
    return r0, c0, r0 + 3, c0 + 3


def _draw_room(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    source: tuple[int, int],
    dye: tuple[int, int],
) -> list[tuple[int, int]]:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top:bottom + 1, left:right + 1] = GRAY
    grid[top + 1:bottom, left + 1:right] = BG

    region_coords: list[tuple[int, int]] = []
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            region_coords.append((r, c))

    sr, sc = source
    dr, dc = dye
    grid[sr, sc] = SOURCE
    grid[dr, dc] = DYE
    return region_coords


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choice([2, 3, 4, 4])
    side_min = 18 + 2 * (n_regions - 2)
    h = random.randint(side_min, 30)
    w = random.randint(side_min, 30)

    grid = np.full((h, w), BG, dtype=int)
    fill_a, fill_b = random.sample(PAINT_CHOICES, 2)
    palette_box = _place_palette(grid, fill_a, fill_b)

    direction_pool = ["cw", "ccw"] * (n_regions // 2)
    if n_regions % 2 == 1:
        direction_pool.append(random.choice(["cw", "ccw"]))
    random.shuffle(direction_pool)

    occupied = [_expand_box(palette_box, 1, h, w)]
    rooms: list[tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]] = []

    for direction in direction_pool:
        placed = False
        for _ in range(REGION_PLACE_RETRIES):
            inner_h = random.randint(4, 8)
            inner_w = random.randint(4, 8)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h >= h - 2 or outer_w >= w - 2:
                continue

            top = random.randint(1, h - outer_h - 1)
            left = random.randint(1, w - outer_w - 1)
            box = (top, left, top + outer_h - 1, left + outer_w - 1)
            padded = _expand_box(box, 1, h, w)
            if any(_box_overlap(padded, other) for other in occupied):
                continue

            local_mask = np.ones((inner_h, inner_w), dtype=bool)
            wall_cycle_local = _ordered_wall_cells_ccw(local_mask)
            candidates = _candidate_marker_pairs(wall_cycle_local, direction)
            if not candidates:
                continue
            source_local, dye_local = random.choice(candidates)
            source = (top + 1 + source_local[0], left + 1 + source_local[1])
            dye = (top + 1 + dye_local[0], left + 1 + dye_local[1])

            region_coords = _draw_room(grid, top, left, inner_h, inner_w, source, dye)
            rooms.append((region_coords, source, dye))
            occupied.append(padded)
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for region_coords, source, dye in rooms:
        _color_region(expected, region_coords, source, dye, fill_a, fill_b)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
