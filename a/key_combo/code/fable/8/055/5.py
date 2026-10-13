from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BACKGROUND = 0
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
EIGHT_DIRS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
INPUT_SIDE_OPTIONS = (15, 17, 19, 21, 23, 25, 27, 29)
OUTPUT_SIDE_OPTIONS = (5, 7)
MAX_GENERATION_RETRIES = 600
MAX_ANCHOR_ATTEMPTS = 160
MAX_GROW_ATTEMPTS = 120
_SEEN_INPUT_SIGS: set[bytes] = set()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _spiral_coords(side: int) -> list[tuple[int, int]]:
    assert side % 2 == 1 and side > 0
    center = side // 2
    r = center
    c = center
    coords = [(r, c)]
    step_len = 1
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    while len(coords) < side * side:
        for turn_idx, (dr, dc) in enumerate(dirs):
            for _ in range(step_len):
                r += dr
                c += dc
                if 0 <= r < side and 0 <= c < side:
                    coords.append((r, c))
                    if len(coords) == side * side:
                        return coords
            if turn_idx % 2 == 1:
                step_len += 1
    return coords


def _spiral_index_map(side: int) -> dict[tuple[int, int], int]:
    return {coord: idx for idx, coord in enumerate(_spiral_coords(side))}


def _components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in FOUR_DIRS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "area": int(coords_arr.shape[0]),
                }
            )
    return comps


def _output_signature(grid: np.ndarray) -> bytes:
    return grid.shape[0].to_bytes(1, "little") + grid.shape[1].to_bytes(1, "little") + grid.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse scattered objects into one solid color spiral ordered by first encounter on a hidden central spiral.

    Input structure:
    - The input is an odd-by-odd black canvas containing several monochromatic
      4-connected objects, with a different non-black color for each object.
    - The sum of all object areas is an odd square.
    - Multiple objects may first appear on the same square spiral ring, so ring
      distance alone does not fix their order.

    Rule:
    - Start at the canvas center and scan cells along the clockwise square
      spiral: right 1, down 1, left 2, up 2, right 3, and so on.
    - Record the first time this spiral touches each object; that encounter
      order is the object order.
    - Let each object's area be its run length. Build an odd square whose side
      is the square root of the total area, then fill its cells from the center
      along the same clockwise spiral using consecutive monochrome runs in the
      recorded encounter order.

    Key insight:
    - The input keeps only two invariants from each object: when the shared
      center spiral meets it, and how many cells it contains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1
    side = int(g.shape[0])
    bg = _get_bg(g)
    assert bg == BACKGROUND, "expected black background"

    comps = _components(g, bg=bg)
    assert len(comps) >= 2, "expected multiple objects"
    colors = [int(comp["color"]) for comp in comps]
    assert len(colors) == len(set(colors)), "each object color must be unique"
    area_by_color = {int(comp["color"]): int(comp["area"]) for comp in comps}

    order: list[int] = []
    seen_colors: set[int] = set()
    for r, c in _spiral_coords(side):
        color = int(g[r, c])
        if color == bg or color in seen_colors:
            continue
        seen_colors.add(color)
        order.append(color)

    assert set(order) == set(colors), "spiral should eventually encounter every object"
    total_area = sum(area_by_color.values())
    out_side = math.isqrt(total_area)
    assert out_side * out_side == total_area and out_side % 2 == 1, "total area must be an odd square"

    out = np.full((out_side, out_side), bg, dtype=int)
    out_coords = _spiral_coords(out_side)
    cursor = 0
    for color in order:
        run_len = area_by_color[color]
        for _ in range(run_len):
            r, c = out_coords[cursor]
            out[r, c] = color
            cursor += 1
    assert cursor == len(out_coords)
    return out


def _random_partition(total: int, count: int, max_part: int) -> list[int] | None:
    if count <= 0 or count > total or count * max_part < total:
        return None
    parts = [1] * count
    remain = total - count
    for _ in range(remain):
        candidates = [i for i, part in enumerate(parts) if part < max_part]
        if not candidates:
            return None
        parts[random.choice(candidates)] += 1
    random.shuffle(parts)
    if max(parts) == 1:
        return None
    if len(set(parts)) == 1:
        idx = random.randrange(count)
        jdx = (idx + 1) % count
        if parts[jdx] > 1 and parts[idx] < max_part:
            parts[idx] += 1
            parts[jdx] -= 1
    if len(set(parts)) == 1:
        return None
    return parts


def _ring(center: int, coord: tuple[int, int]) -> int:
    r, c = coord
    return max(abs(r - center), abs(c - center))


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _neighbors4_in_bounds(side: int, coord: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = coord
    out = []
    for dr, dc in FOUR_DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < side and 0 <= nc < side:
            out.append((nr, nc))
    return out


def _halo(coords: set[tuple[int, int]], side: int) -> set[tuple[int, int]]:
    halo = set(coords)
    for r, c in coords:
        for dr, dc in EIGHT_DIRS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < side and 0 <= nc < side:
                halo.add((nr, nc))
    return halo


def _grow_blob(
    side: int,
    anchor: tuple[int, int],
    min_index: int,
    area: int,
    index_map: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if anchor in blocked:
        return None
    if area == 1:
        return {anchor}
    radius_choices = [2, 3, 4]
    random.shuffle(radius_choices)
    for radius in radius_choices:
        allowed = {
            (r, c)
            for r in range(max(0, anchor[0] - radius), min(side, anchor[0] + radius + 1))
            for c in range(max(0, anchor[1] - radius), min(side, anchor[1] + radius + 1))
            if index_map[(r, c)] >= min_index and (r, c) not in blocked
        }
        if len(allowed) < area:
            continue
        for _ in range(MAX_GROW_ATTEMPTS):
            cells = {anchor}
            frontier = [nb for nb in _neighbors4_in_bounds(side, anchor) if nb in allowed]
            seen_frontier = set(frontier)
            while len(cells) < area and frontier:
                frontier.sort(key=lambda coord: (index_map[coord], random.random()))
                pick_from = frontier[: min(len(frontier), 5)]
                cell = random.choice(pick_from)
                frontier.remove(cell)
                seen_frontier.discard(cell)
                if cell in cells:
                    continue
                if not any(nb in cells for nb in _neighbors4_in_bounds(side, cell)):
                    continue
                cells.add(cell)
                for nb in _neighbors4_in_bounds(side, cell):
                    if nb in allowed and nb not in cells and nb not in seen_frontier:
                        frontier.append(nb)
                        seen_frontier.add(nb)
            if len(cells) == area:
                return cells
    return None


def _choose_anchor_and_blob(
    side: int,
    prev_index: int,
    area: int,
    index_map: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    chosen_anchors: list[tuple[int, int]],
    forced_ring: int | None,
    latest_index: int,
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    center = side // 2
    coords = list(index_map.keys())
    random.shuffle(coords)
    candidates = []
    for coord in coords:
        idx = index_map[coord]
        if idx <= prev_index + 2 or idx > latest_index:
            continue
        if coord in blocked:
            continue
        ring = _ring(center, coord)
        if ring == 0:
            continue
        if forced_ring is not None and ring != forced_ring:
            continue
        if any(_chebyshev(coord, other) < 5 for other in chosen_anchors):
            continue
        candidates.append(coord)
    candidates.sort(key=lambda coord: index_map[coord])
    for anchor in candidates[:MAX_ANCHOR_ATTEMPTS]:
        blob = _grow_blob(side, anchor, index_map[anchor], area, index_map, blocked)
        if blob is None:
            continue
        return anchor, blob
    return None


def _build_input_from_plan(
    side: int,
    areas: list[int],
    colors: list[int],
    tie_pair: tuple[int, int],
) -> np.ndarray | None:
    path = _spiral_coords(side)
    index_map = {coord: idx for idx, coord in enumerate(path)}
    latest_index = int(side * side * 0.8)
    blocked: set[tuple[int, int]] = set()
    anchors: list[tuple[int, int]] = []
    blobs: list[set[tuple[int, int]]] = []
    prev_index = -1
    shared_ring: int | None = None

    for obj_idx, area in enumerate(areas):
        forced_ring = None
        if obj_idx == tie_pair[1]:
            forced_ring = shared_ring
        result = _choose_anchor_and_blob(
            side=side,
            prev_index=prev_index,
            area=area,
            index_map=index_map,
            blocked=blocked,
            chosen_anchors=anchors,
            forced_ring=forced_ring,
            latest_index=latest_index,
        )
        if result is None:
            return None
        anchor, blob = result
        if obj_idx == tie_pair[0]:
            shared_ring = _ring(side // 2, anchor)
        if obj_idx == tie_pair[1]:
            if shared_ring is None or _ring(side // 2, anchor) != shared_ring:
                return None
        anchors.append(anchor)
        blobs.append(blob)
        prev_index = index_map[anchor]
        blocked |= _halo(blob, side)

    if shared_ring is None:
        return None
    grid = np.zeros((side, side), dtype=int)
    for color, blob in zip(colors, blobs):
        for r, c in blob:
            grid[r, c] = color
    return grid


def _object_summary(grid: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    comps = _components(grid, bg=BACKGROUND)
    center = grid.shape[0] // 2
    index_map = _spiral_index_map(grid.shape[0])
    by_color = {int(comp["color"]): comp for comp in comps}
    encounter = []
    seen = set()
    for coord in _spiral_coords(grid.shape[0]):
        color = int(grid[coord])
        if color != BACKGROUND and color not in seen:
            seen.add(color)
            encounter.append(color)
    areas = [int(by_color[color]["area"]) for color in encounter]
    rings = [
        min(_ring(center, (int(r), int(c))) for r, c in by_color[color]["coords"])
        for color in encounter
    ]
    first_indices = [
        min(index_map[(int(r), int(c))] for r, c in by_color[color]["coords"])
        for color in encounter
    ]
    return areas, rings, first_indices


def _semantic_checks(grid: np.ndarray, planned_areas: list[int], planned_colors: list[int]) -> bool:
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] % 2 != 1 or max(grid.shape) > 30:
        return False
    if _get_bg(grid) != BACKGROUND:
        return False
    comps = _components(grid, bg=BACKGROUND)
    if len(comps) != len(planned_areas):
        return False
    if len({int(comp["color"]) for comp in comps}) != len(comps):
        return False

    areas, rings, first_indices = _object_summary(grid)
    if areas != planned_areas:
        return False
    if len(set(rings)) == len(rings):
        return False
    if not any(rings[i] == rings[i + 1] for i in range(len(rings) - 1)):
        return False
    if not all(first_indices[i] < first_indices[i + 1] for i in range(len(first_indices) - 1)):
        return False
    if len(set(areas)) == 1:
        return False
    if max(areas) < 4:
        return False

    out = main(grid)
    if out.shape[0] not in OUTPUT_SIDE_OPTIONS:
        return False
    flat_spiral_colors = [int(out[r, c]) for r, c in _spiral_coords(out.shape[0])]
    expected = []
    for color, area in zip(planned_colors, planned_areas):
        expected.extend([color] * area)
    if flat_spiral_colors != expected:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    out_side = random.choice(OUTPUT_SIDE_OPTIONS)
    total_area = out_side * out_side
    min_count = 3 if out_side == 5 else 4
    max_count = 5 if out_side == 5 else 7
    count = random.randint(min_count, max_count)
    max_part = 10 if out_side == 5 else 13
    areas = _random_partition(total_area, count, max_part=max_part)
    if areas is None:
        return None
    if sum(areas) != total_area:
        return None

    colors = random.sample(list(range(1, 10)), count)
    side = random.choice(INPUT_SIDE_OPTIONS)
    tie_start = random.randint(0, count - 2)
    grid = _build_input_from_plan(side=side, areas=areas, colors=colors, tie_pair=(tie_start, tie_start + 1))
    if grid is None:
        return None

    if not _semantic_checks(grid, areas, colors):
        return None

    out = main(grid)
    out_sig = _output_signature(out)
    inp_sig = _output_signature(grid)
    if out_sig in _SEEN_OUTPUT_SIGS or inp_sig in _SEEN_INPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(out_sig)
    _SEEN_INPUT_SIGS.add(inp_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-collapse input")
