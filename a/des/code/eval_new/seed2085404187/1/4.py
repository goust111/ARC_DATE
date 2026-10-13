from __future__ import annotations

import random
from collections import deque

import numpy as np


RED = 2
GREY = 5
PALETTE = tuple(range(10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _bg_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _wall_orientations(grid: np.ndarray) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == r1:
            orient = "h"
            assert coords.shape[0] == c1 - c0 + 1, "horizontal wall must be contiguous"
        elif c0 == c1:
            orient = "v"
            assert coords.shape[0] == r1 - r0 + 1, "vertical wall must be contiguous"
        else:
            raise AssertionError("grey walls must be straight line segments")
        for r, c in coords:
            orientations[(int(r), int(c))] = orient
    return orientations


def _source_side(red_rc: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    rr, rc = red_rc
    r0, c0, r1, c1 = bbox
    if rr < r0:
        return "up"
    if rr > r1:
        return "down"
    if rc < c0:
        return "left"
    if rc > c1:
        return "right"
    raise AssertionError("red marker must sit outside the source bbox")


def _find_sources(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    source_colors = [int(c) for c in np.unique(g) if int(c) not in {bg, RED, GREY}]
    sources: list[dict] = []
    red_owner: dict[tuple[int, int], int] = {}
    for color in source_colors:
        for coords in _components(g == color):
            assert 1 <= coords.shape[0] <= 4, "sources must contain 1 to 4 cells"
            bbox = _bbox(coords)
            red_neighbors: set[tuple[int, int]] = set()
            for r, c in coords:
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == RED:
                        red_neighbors.add((nr, nc))
            assert len(red_neighbors) == 1, "each source must have exactly one adjacent red marker"
            red_rc = next(iter(red_neighbors))
            if red_rc in red_owner:
                raise AssertionError("a red marker cannot serve multiple sources")
            red_owner[red_rc] = len(sources)
            sources.append(
                {
                    "color": color,
                    "coords": np.asarray(sorted(map(tuple, coords.tolist())), dtype=int),
                    "bbox": bbox,
                    "red": red_rc,
                    "side": _source_side(red_rc, bbox),
                }
            )
    sources.sort(key=lambda src: (src["bbox"][0], src["bbox"][1], src["color"]))
    return sources


def _initial_direction(pixel: tuple[int, int], source: dict) -> tuple[int, int]:
    r, c = pixel
    r0, c0, r1, c1 = source["bbox"]
    center_r = 0.5 * (r0 + r1)
    center_c = 0.5 * (c0 + c1)
    side = source["side"]
    if side == "up":
        return 1, -1 if c < center_c else 1
    if side == "down":
        return -1, -1 if c < center_c else 1
    if side == "left":
        return (-1 if r < center_r else 1), 1
    if side == "right":
        return (-1 if r < center_r else 1), -1
    raise AssertionError("unknown source side")


def _trace_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    source_cells: set[tuple[int, int]],
    wall_orient: dict[tuple[int, int], str],
) -> tuple[list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cur_r, cur_c = start
    dr, dc = direction
    seen: set[tuple[int, int, int, int]] = set()
    path: list[tuple[int, int]] = []
    reflections = 0
    while True:
        state = (cur_r, cur_c, dr, dc)
        if state in seen:
            break
        seen.add(state)
        nr, nc = cur_r + dr, cur_c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if (nr, nc) in source_cells or int(g[nr, nc]) == RED:
            break
        if int(g[nr, nc]) == GREY:
            orient = wall_orient.get((nr, nc))
            if orient == "h":
                dr = -dr
            elif orient == "v":
                dc = -dc
            else:
                break
            reflections += 1
            continue
        cur_r, cur_c = nr, nc
        path.append((cur_r, cur_c))
    return path, reflections


def _project(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    wall_orient = _wall_orientations(g)
    sources = _find_sources(g, bg)
    source_cells = {tuple(map(int, cell)) for src in sources for cell in src["coords"]}

    out = g.copy()
    total_reflections = 0
    for source in sources:
        for cell in source["coords"]:
            start = (int(cell[0]), int(cell[1]))
            direction = _initial_direction(start, source)
            path, reflections = _trace_ray(g, start, direction, source_cells, wall_orient)
            total_reflections += reflections
            for r, c in path:
                out[r, c] = source["color"]
    return out, total_reflections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each small colored source away from its adjacent red marker and bend the diagonal shadows with grey reflector walls.

    Input structure:
    - The most common color is the background.
    - Each source is a 1-to-4-cell 4-connected component in its own color, and exactly one red pixel sits immediately outside one side of that source.
    - Grey pixels form straight horizontal or vertical wall segments, while all remaining non-background, non-red, non-grey pixels belong to sources.

    Rule:
    - Sort the sources from top to bottom, then left to right.
    - The red marker chooses the emitting side: markers above or below make the source cast downward or upward, and markers left or right make it cast rightward or leftward.
    - Every source pixel emits one diagonal ray away from that side; pixels on opposite halves of the source bbox use opposite lateral signs, so the source fans out into a slanted shadow.
    - Rays recolor background or earlier shadow cells with the source color. Hitting a horizontal grey wall flips the row step, and hitting a vertical grey wall flips the column step; rays stop at the grid edge, at a red marker, or at any source cell.
    - Grey walls and red markers remain unchanged, and later sources overwrite earlier shadow pixels where their paths overlap.

    Key insight: The red pixel selects which side of a source emits the shadow, the source pixel's position inside the tiny shape fixes the diagonal branch, and each grey wall bends that branch by flipping one signed step.
    """
    out, _ = _project(np.asarray(grid, dtype=int))
    return out


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rotated = tuple((c, -r) for r, c in shape)
    return _normalize_shape(rotated)


def _build_shapes() -> list[tuple[tuple[int, int], ...]]:
    bases = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
    ]
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        cur = _normalize_shape(base)
        for _ in range(4):
            shapes.add(cur)
            cur = _rotate_shape(cur)
    return sorted(shapes)


SHAPES = _build_shapes()


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _valid_sides(shape: tuple[tuple[int, int], ...]) -> list[str]:
    h, w = _shape_dims(shape)
    sides: list[str] = []
    if w > 1:
        sides.extend(["up", "down"])
    if h > 1:
        sides.extend(["left", "right"])
    return sides


def _edge_cells(shape: tuple[tuple[int, int], ...], side: str) -> list[tuple[int, int]]:
    if side == "up":
        edge_value = min(r for r, _ in shape)
        return [(r, c) for r, c in shape if r == edge_value]
    if side == "down":
        edge_value = max(r for r, _ in shape)
        return [(r, c) for r, c in shape if r == edge_value]
    if side == "left":
        edge_value = min(c for _, c in shape)
        return [(r, c) for r, c in shape if c == edge_value]
    edge_value = max(c for _, c in shape)
    return [(r, c) for r, c in shape if c == edge_value]


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _place_source(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    side: str,
    color: int,
    blocked: set[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    sh, sw = _shape_dims(shape)
    dr, dc = SIDE_TO_DELTA[side]
    margin = 2
    for _ in range(80):
        r0 = random.randint(margin, h - sh - margin - 1)
        c0 = random.randint(margin, w - sw - margin - 1)
        cells = {(r0 + r, c0 + c) for r, c in shape}
        marker_base = random.choice(_edge_cells(shape, side))
        mr = r0 + marker_base[0] + dr
        mc = c0 + marker_base[1] + dc
        footprint = set(cells)
        footprint.add((mr, mc))
        if not (0 <= mr < h and 0 <= mc < w):
            continue
        if any(cell in blocked for cell in footprint):
            continue
        if any(int(grid[r, c]) != _bg_color(grid) for r, c in footprint):
            continue
        for r, c in cells:
            grid[r, c] = color
        grid[mr, mc] = RED
        coords = np.asarray(sorted(footprint - {(mr, mc)}), dtype=int)
        return {
            "color": color,
            "coords": coords,
            "bbox": _bbox(coords),
            "red": (mr, mc),
            "side": side,
        }
    return None


def _step_candidates(
    grid: np.ndarray,
    source: dict,
    source_cells: set[tuple[int, int]],
    wall_orient: dict[tuple[int, int], str],
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = []
    for cell in source["coords"]:
        cur_r, cur_c = int(cell[0]), int(cell[1])
        dr, dc = _initial_direction((cur_r, cur_c), source)
        seen: set[tuple[int, int, int, int]] = set()
        advance_count = 0
        while True:
            state = (cur_r, cur_c, dr, dc)
            if state in seen:
                break
            seen.add(state)
            nr, nc = cur_r + dr, cur_c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if (nr, nc) in source_cells or int(g[nr, nc]) == RED:
                break
            if int(g[nr, nc]) == GREY:
                orient = wall_orient.get((nr, nc))
                if orient == "h":
                    dr = -dr
                elif orient == "v":
                    dc = -dc
                else:
                    break
                continue
            candidates.append(((cur_r, cur_c), (dr, dc), (nr, nc), advance_count))
            cur_r, cur_c = nr, nc
            advance_count += 1
    return candidates


def _segment_cells(
    hit: tuple[int, int],
    orient: str,
    length: int,
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    idx = random.randrange(length)
    hr, hc = hit
    if orient == "h":
        c0 = hc - idx
        cells = [(hr, c0 + k) for k in range(length)]
    else:
        r0 = hr - idx
        cells = [(r0 + k, hc) for k in range(length)]
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None
    return cells


def _place_wall(
    grid: np.ndarray,
    sources: list[dict],
    wall_blocked: set[tuple[int, int]],
    target: dict,
) -> bool:
    h, w = grid.shape
    source_cells = {tuple(map(int, cell)) for src in sources for cell in src["coords"]}
    wall_orient = _wall_orientations(grid)
    candidates = _step_candidates(grid, target, source_cells, wall_orient)
    random.shuffle(candidates)
    for _cur, _dir, hit, distance in candidates:
        if distance < 2:
            continue
        orientations = ["h", "v"]
        random.shuffle(orientations)
        for orient in orientations:
            for length in random.sample([2, 3, 4, 5], 4):
                cells = _segment_cells(hit, orient, length, h, w)
                if cells is None:
                    continue
                if any(int(grid[r, c]) != _bg_color(grid) for r, c in cells):
                    continue
                if any(cell in wall_blocked for cell in cells):
                    continue
                for r, c in cells:
                    grid[r, c] = GREY
                wall_blocked |= _expand_cells(set(cells), h, w, pad=1)
                return True
    return False


def _strict_bg(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    others = [int(v) for i, v in enumerate(counts) if i != bg]
    return counts[bg] > max(others, default=0)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    bg = random.choice([c for c in PALETTE if c not in {RED, GREY}])
    grid = np.full((h, w), bg, dtype=int)

    source_count = random.randint(2, 5)
    colors = random.sample([c for c in PALETTE if c not in {bg, RED, GREY}], source_count)

    sources: list[dict] = []
    source_blocked: set[tuple[int, int]] = set()
    for color in colors:
        shape = random.choice(SHAPES)
        sides = _valid_sides(shape)
        if not sides:
            return None
        side = random.choice(sides)
        source = _place_source(grid, shape, side, color, source_blocked)
        if source is None:
            return None
        sources.append(source)
        occupied = {tuple(map(int, cell)) for cell in source["coords"]}
        occupied.add(source["red"])
        source_blocked |= _expand_cells(occupied, h, w, pad=1)

    wall_blocked = set(source_blocked)
    wall_targets = random.randint(1, min(4, source_count + 1))
    placed_walls = 0
    for _ in range(wall_targets * 3):
        if placed_walls >= wall_targets:
            break
        target = random.choice(sources)
        if _place_wall(grid, sources, wall_blocked, target):
            placed_walls += 1
    if placed_walls == 0:
        return None

    try:
        out, reflections = _project(grid)
    except AssertionError:
        return None

    if reflections == 0:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < max(6, source_count * 3):
        return None
    if not _strict_bg(grid, bg):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflection-shadow ARC input")
