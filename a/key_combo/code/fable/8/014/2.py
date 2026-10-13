from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


GREY = 5
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
SOURCE_COUNT_RANGE = (2, 4)
EXTRA_RECT_COUNT_RANGE = (0, 2)
RECT_HEIGHT_RANGE = (4, 7)
RECT_WIDTH_RANGE = (4, 8)
RECT_PADDING = 2
STEP_RANGE = (4, 8)


@dataclass(frozen=True)
class Rectangle:
    bbox: tuple[int, int, int, int]

    @property
    def corners(self) -> dict[tuple[int, int], tuple[int, int]]:
        r0, c0, r1, c1 = self.bbox
        return {
            (1, 1): (r0, c0),
            (1, -1): (r0, c1),
            (-1, 1): (r1, c0),
            (-1, -1): (r1, c1),
        }

    def border_cells(self) -> set[tuple[int, int]]:
        r0, c0, r1, c1 = self.bbox
        cells = set()
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        return cells

    def zone_cells(self) -> set[tuple[int, int]]:
        r0, c0, r1, c1 = self.bbox
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[dict] = []
    h, w = grid.shape
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == bg:
                continue
            color = int(grid[sr, sc])
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if not in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": len(cells),
                }
            )
    return comps


def _missing_corner_direction(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "L markers must fit a 2x2 box"
    cells = {(int(r), int(c)) for r, c in coords}
    for rr in (r0, r1):
        for cc in (c0, c1):
            if (rr, cc) not in cells:
                dr = -1 if rr == r0 else 1
                dc = -1 if cc == c0 else 1
                return (rr, cc), (dr, dc)
    raise AssertionError("expected exactly one missing corner")


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return {(int(r), int(c)) for r, c in coords} == expected


def _diag_segment(
    start: tuple[int, int], end: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    dr, dc = direction
    delta_r = er - sr
    delta_c = ec - sc
    assert delta_r * dr >= 0 and delta_c * dc >= 0
    assert abs(delta_r) == abs(delta_c)
    steps = abs(delta_r)
    assert (delta_r, delta_c) == (steps * dr, steps * dc)
    return [(sr + i * dr, sc + i * dc) for i in range(steps + 1)]


def _l_cells_from_start(
    start: tuple[int, int], direction: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int, int, int]]:
    r, c = start
    dr, dc = direction
    r0 = r - (1 if dr > 0 else 0)
    c0 = c - (1 if dc > 0 else 0)
    cells = []
    for rr in (r0, r0 + 1):
        for cc in (c0, c0 + 1):
            if (rr, cc) != start:
                cells.append((rr, cc))
    return tuple(sorted(cells)), (r0, c0, r0 + 1, c0 + 1)


def _valid_prism_hit_cells(
    start: tuple[int, int], out_dir: tuple[int, int], in_dir: tuple[int, int]
) -> list[tuple[int, int]]:
    cells, _ = _l_cells_from_start(start, out_dir)
    cell_set = set(cells)
    valid = []
    for cell in cells:
        prev = (cell[0] - in_dir[0], cell[1] - in_dir[1])
        if prev not in cell_set:
            valid.append(cell)
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each colored L source through grey relay prisms and fill the first hit rectangle interior.

    Input structure:
    - The grid has a dominant background, several one-cell-thick hollow rectangles
      of one repeated wall color, and several 2x2-minus-one-cell L markers.
    - Grey L markers are relay prisms. Every other L marker has a unique source
      color and sits outside all rectangles.

    Rule:
    - Read each L marker's missing corner as a diagonal direction and start the
      ray from that empty missing-corner cell.
    - When a source ray touches any grey L cell, restart the ray from that grey
      L's own missing corner and continue along its missing-corner direction.
    - The first hollow rectangle border hit by the ray is that source's target;
      recolor the full interior of that rectangle with the source color.

    Key insight:
    - The L marker does not point with an arm; its absent 2x2 corner is the
      direction cue, and grey L markers redirect the ray before the rectangle fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    comps = find_components(g, bg)
    assert comps, "expected foreground objects"

    rectangles: list[Rectangle] = []
    sources: list[dict] = []
    prisms: list[dict] = []

    for comp in comps:
        coords = comp["coords"]
        area = int(comp["area"])
        color = int(comp["color"])
        if area == 3 and bbox_coords(coords)[2] - bbox_coords(coords)[0] == 1 and bbox_coords(coords)[3] - bbox_coords(coords)[1] == 1:
            start, direction = _missing_corner_direction(coords)
            record = {"color": color, "start": start, "direction": direction, "coords": coords}
            if color == GREY:
                prisms.append(record)
            else:
                sources.append(record)
        elif _is_hollow_rectangle(coords):
            rectangles.append(Rectangle(comp["bbox"]))
        else:
            raise AssertionError(f"unexpected object shape with color {color}")

    assert rectangles, "expected hollow rectangles"
    assert sources, "expected colored source L markers"
    assert len({src["color"] for src in sources}) == len(sources), "source colors must be unique"

    prism_lookup: dict[tuple[int, int], dict] = {}
    for prism in prisms:
        for r, c in prism["coords"]:
            key = (int(r), int(c))
            assert key not in prism_lookup, "prisms must not overlap"
            prism_lookup[key] = prism

    rect_lookup: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rectangles):
        for cell in rect.border_cells():
            assert cell not in rect_lookup, "rectangles must not touch"
            rect_lookup[cell] = idx

    out = g.copy()
    filled: set[int] = set()

    for src in sources:
        current = src["start"]
        direction = src["direction"]
        seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        hit_rect = None
        while in_bounds(g.shape, current[0], current[1]):
            state = (current, direction)
            assert state not in seen_states, "ray loop detected"
            seen_states.add(state)

            if current in rect_lookup:
                hit_rect = rect_lookup[current]
                break
            if current in prism_lookup:
                prism = prism_lookup[current]
                current = prism["start"]
                direction = prism["direction"]
                continue
            current = (current[0] + direction[0], current[1] + direction[1])

        assert hit_rect is not None, "each source must hit exactly one rectangle"
        filled.add(hit_rect)
        r0, c0, r1, c1 = rectangles[hit_rect].bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = int(src["color"])

    return out


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_chain_lengths(rng: random.Random, count: int) -> list[int]:
    lengths = [rng.choices([0, 1, 2], weights=[3, 4, 3], k=1)[0] for _ in range(count)]
    if count >= 2 and all(length == 0 for length in lengths):
        lengths[rng.randrange(count)] = rng.choice([1, 2])
    if count >= 3 and max(lengths) < 2 and rng.random() < 0.45:
        lengths[rng.randrange(count)] = 2
    if all(length > 0 for length in lengths) and rng.random() < 0.5:
        lengths[rng.randrange(count)] = 0
    return lengths


def _pick_colors(rng: random.Random, count: int) -> tuple[int, int, list[int]]:
    palette = list(range(10))
    palette.remove(GREY)
    bg = rng.choice(palette)
    palette.remove(bg)
    wall = rng.choice(palette)
    palette.remove(wall)
    colors = rng.sample(palette, count)
    return bg, wall, colors


def _place_rectangles(
    rng: random.Random, total: int
) -> list[Rectangle] | None:
    rects: list[Rectangle] = []
    for _ in range(total * 80):
        if len(rects) == total:
            return rects
        h = rng.randint(*RECT_HEIGHT_RANGE)
        w = rng.randint(*RECT_WIDTH_RANGE)
        r0 = rng.randint(0, CANVAS_SIZE - h)
        c0 = rng.randint(0, CANVAS_SIZE - w)
        rect = Rectangle((r0, c0, r0 + h - 1, c0 + w - 1))
        if any(_bbox_overlaps(rect.bbox, other.bbox, RECT_PADDING) for other in rects):
            continue
        rects.append(rect)
    return None


def _can_place_l(
    start: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    bg_blocked: set[tuple[int, int]],
    rect_zones: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int, int, int]] | None:
    cells, bbox = _l_cells_from_start(start, direction)
    all_cells = set(cells) | {start}
    for r, c in all_cells:
        if not in_bounds((CANVAS_SIZE, CANVAS_SIZE), r, c):
            return None
        if (r, c) in occupied or (r, c) in bg_blocked or (r, c) in rect_zones:
            return None
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in occupied:
                return None
    return cells, bbox


def _segment_clear(
    start: tuple[int, int],
    end: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    bg_blocked: set[tuple[int, int]],
    rect_zones: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells = _diag_segment(start, end, direction)
    for cell in cells[:-1]:
        if cell in occupied or cell in bg_blocked or cell in rect_zones:
            return None
    return cells


def _build_one_source(
    rng: random.Random,
    target_rect: Rectangle,
    color: int,
    chain_len: int,
    occupied: set[tuple[int, int]],
    bg_blocked: set[tuple[int, int]],
    rect_zones: set[tuple[int, int]],
) -> tuple[list[tuple[int, tuple[tuple[int, int], ...]]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(160):
        local_occupied: set[tuple[int, int]] = set()
        local_bg: set[tuple[int, int]] = set()
        placements: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        cur_dir = rng.choice(DIAGONALS)
        cur_hit = target_rect.corners[cur_dir]
        occ_union = occupied | local_occupied
        bg_union = bg_blocked | local_bg
        success = True

        for prism_index in range(chain_len):
            placed = False
            distances = list(range(STEP_RANGE[0], STEP_RANGE[1] + 1))
            rng.shuffle(distances)
            for dist in distances:
                start = (cur_hit[0] - dist * cur_dir[0], cur_hit[1] - dist * cur_dir[1])
                placement = _can_place_l(start, cur_dir, occ_union, bg_union, rect_zones)
                if placement is None:
                    continue
                cells, _bbox = placement
                segment = _segment_clear(start, cur_hit, cur_dir, occ_union, bg_union, rect_zones)
                if segment is None:
                    continue
                prev_dirs = DIAGONALS[:]
                rng.shuffle(prev_dirs)
                prev_dirs.sort(key=lambda d: d == cur_dir)
                for prev_dir in prev_dirs:
                    hit_cells = _valid_prism_hit_cells(start, cur_dir, prev_dir)
                    if not hit_cells:
                        continue
                    rng.shuffle(hit_cells)
                    hit = hit_cells[0]
                    placements.append((GREY, cells))
                    local_occupied.update(cells)
                    local_bg.update(segment[:-1])
                    occ_union = occupied | local_occupied
                    bg_union = bg_blocked | local_bg
                    cur_hit = hit
                    cur_dir = prev_dir
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                success = False
                break

        if not success:
            continue

        distances = list(range(STEP_RANGE[0], STEP_RANGE[1] + 1))
        rng.shuffle(distances)
        for dist in distances:
            start = (cur_hit[0] - dist * cur_dir[0], cur_hit[1] - dist * cur_dir[1])
            placement = _can_place_l(start, cur_dir, occ_union, bg_union, rect_zones)
            if placement is None:
                continue
            cells, _bbox = placement
            segment = _segment_clear(start, cur_hit, cur_dir, occ_union, bg_union, rect_zones)
            if segment is None:
                continue
            placements.append((color, cells))
            local_occupied.update(cells)
            local_bg.update(segment[:-1])
            return placements, local_occupied, local_bg

    return None


def _crop_with_padding(grid: np.ndarray, bg: int, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = rng.randint(0, min(2, int(r0)))
    left = rng.randint(0, min(2, int(c0)))
    bottom = rng.randint(0, min(2, int(grid.shape[0] - 1 - r1)))
    right = rng.randint(0, min(2, int(grid.shape[1] - 1 - c1)))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    source_count = rng.randint(*SOURCE_COUNT_RANGE)
    extra_rect_count = rng.randint(*EXTRA_RECT_COUNT_RANGE)
    total_rects = source_count + extra_rect_count
    rectangles = _place_rectangles(rng, total_rects)
    if rectangles is None:
        return None

    bg, wall, source_colors = _pick_colors(rng, source_count)
    chain_lengths = _sample_chain_lengths(rng, source_count)
    active_rects = rng.sample(rectangles, source_count)

    occupied: set[tuple[int, int]] = set()
    bg_blocked: set[tuple[int, int]] = set()
    rect_zones: set[tuple[int, int]] = set()
    for rect in rectangles:
        occupied.update(rect.border_cells())
        rect_zones.update(rect.zone_cells())

    l_objects: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    order = list(range(source_count))
    rng.shuffle(order)
    for idx in order:
        built = _build_one_source(
            rng,
            active_rects[idx],
            source_colors[idx],
            chain_lengths[idx],
            occupied,
            bg_blocked,
            rect_zones,
        )
        if built is None:
            return None
        placements, occ_add, bg_add = built
        l_objects.extend(placements)
        occupied.update(occ_add)
        bg_blocked.update(bg_add)

    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), bg, dtype=int)
    for rect in rectangles:
        for r, c in rect.border_cells():
            grid[r, c] = wall
    for color, cells in l_objects:
        for r, c in cells:
            grid[r, c] = color

    cropped = _crop_with_padding(grid, bg, rng)
    if cropped.ndim != 2 or max(cropped.shape) > 30:
        return None

    try:
        out = main(cropped)
    except AssertionError:
        return None

    if np.array_equal(out, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prism-relay input")
