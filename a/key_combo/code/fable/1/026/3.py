from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _install_arc_utils() -> None:
    here = Path(__file__).resolve()
    for base in here.parents:
        candidate = base / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            sys.path.append(str(candidate.parent))
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_install_arc_utils()

from arc_utils import find_components, get_bg, in_bounds, neighbors4, neighbors8


Coord = tuple[int, int]
Direction = tuple[int, int]

BG = 0
DIRS4: list[Direction] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PERP_DIRS: dict[Direction, list[Direction]] = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}

_TRIANGLE_BASES: dict[str, tuple[Coord, ...]] = {
    "filled4": ((0, 0), (1, -1), (1, 0), (1, 1)),
    "midfilled6": ((0, 0), (1, -1), (1, 0), (1, 1), (2, -1), (2, 1)),
    "full9": ((0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)),
}


def _rotate_cw(offset: Coord) -> Coord:
    r, c = offset
    return c, -r


def _rotate_offsets(offsets: tuple[Coord, ...], turns: int) -> tuple[Coord, ...]:
    out: list[Coord] = []
    for r, c in offsets:
        rr, cc = r, c
        for _ in range(turns % 4):
            rr, cc = _rotate_cw((rr, cc))
        out.append((rr, cc))
    return tuple(out)


TRIANGLE_TEMPLATES: dict[str, dict[Direction, tuple[Coord, ...]]] = {
    name: {DIRS4[k]: _rotate_offsets(base, k) for k in range(4)}
    for name, base in _TRIANGLE_BASES.items()
}


GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_TURN_COUNTS = [0, 1, 2, 3, 4]
GEN_TURN_WEIGHTS = [1, 3, 4, 3, 2]
GEN_MIN_SEG = 2
GEN_MAX_SEG = 7
GEN_MIN_FINAL = 3
GEN_DISTRACTOR_RANGE = (1, 4)
GEN_MAX_ATTEMPTS = 500


def _add(a: Coord, b: Coord) -> Coord:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _mul(v: Direction, k: int) -> Coord:
    return int(v[0] * k), int(v[1] * k)


def _coord_list(coords: np.ndarray) -> list[Coord]:
    return [tuple(map(int, p)) for p in coords]


def _perp(dir_: Direction) -> Direction:
    return -dir_[1], dir_[0]


def _triangle_offsets(name: str, direction: Direction) -> tuple[Coord, ...]:
    return TRIANGLE_TEMPLATES[name][direction]


def _triangle_cells(name: str, tip: Coord, direction: Direction) -> list[Coord]:
    return [_add(tip, off) for off in _triangle_offsets(name, direction)]


def _infer_triangle(coords: list[Coord]) -> tuple[Coord, Direction]:
    pts = set(coords)
    candidates: list[tuple[tuple[int, int, int], Coord, Direction]] = []

    for tip in pts:
        nbrs = [nb for nb in neighbors4(*tip) if nb in pts]
        if len(nbrs) != 1:
            continue

        inner = nbrs[0]
        direction = (tip[0] - inner[0], tip[1] - inner[1])
        if direction not in PERP_DIRS:
            continue

        back = (-direction[0], -direction[1])
        side = _perp(direction)
        by_depth: dict[int, list[int]] = {}
        ok = True

        for r, c in pts:
            rel_r = r - tip[0]
            rel_c = c - tip[1]
            depth = rel_r * back[0] + rel_c * back[1]
            if depth < 0:
                ok = False
                break
            offset = rel_r * side[0] + rel_c * side[1]
            by_depth.setdefault(int(depth), []).append(int(offset))

        if not ok or len(by_depth.get(0, [])) != 1:
            continue

        max_depth = max(by_depth)
        deep_offsets = by_depth[max_depth]
        if not any(v < 0 for v in deep_offsets) or not any(v > 0 for v in deep_offsets):
            continue

        symmetric_layers = sum(
            any(v < 0 for v in offsets) and any(v > 0 for v in offsets)
            for offsets in by_depth.values()
        )
        candidates.append(((max_depth, symmetric_layers, len(pts)), tip, direction))

    if len(candidates) == 1:
        return candidates[0][1], candidates[0][2]

    pts_set = set(coords)
    for tip in coords:
        for direction in DIRS4:
            for template_map in TRIANGLE_TEMPLATES.values():
                shape = {_add(tip, off) for off in template_map[direction]}
                if shape == pts_set:
                    return tip, direction

    raise AssertionError("failed to infer triangle tip and direction")


def _parse_scene(grid: np.ndarray) -> tuple[int, int, Coord, Direction, dict[Coord, tuple[Coord, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    triangle_comps = [comp for comp in comps if int(comp["area"]) > 2]
    assert len(triangle_comps) == 1, "expected exactly one triangle component"
    tri_coords = _coord_list(triangle_comps[0]["coords"])
    tri_colors = {int(g[r, c]) for r, c in tri_coords}
    assert len(tri_colors) == 1, "triangle must be monochromatic"
    tri_color = tri_colors.pop()
    tip, direction = _infer_triangle(tri_coords)

    domino_map: dict[Coord, tuple[Coord, int]] = {}
    for comp in comps:
        if int(comp["area"]) != 2:
            continue
        cells = _coord_list(comp["coords"])
        if len(cells) != 2:
            continue
        a, b = cells
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            continue
        color_a = int(g[a])
        color_b = int(g[b])
        if color_a == color_b:
            continue
        domino_map[a] = (b, color_b)
        domino_map[b] = (a, color_a)

    return bg, tri_color, tip, direction, domino_map


def _trace_ray(
    grid: np.ndarray,
    tip: Coord,
    direction: Direction,
    start_color: int,
    domino_map: dict[Coord, tuple[Coord, int]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    current_color = int(start_color)
    r, c = _add(tip, direction)
    events: list[dict[str, object]] = []

    while in_bounds(g.shape, r, c):
        pos = (r, c)
        if pos in domino_map:
            partner, next_color = domino_map[pos]
            new_dir = (partner[0] - r, partner[1] - c)
            if abs(new_dir[0]) + abs(new_dir[1]) != 1 or (new_dir[0] * direction[0] + new_dir[1] * direction[1]) != 0:
                break
            events.append(
                {
                    "hit": pos,
                    "partner": partner,
                    "incoming": direction,
                    "outgoing": new_dir,
                    "color_in": current_color,
                    "color_out": int(next_color),
                }
            )
            direction = new_dir
            current_color = int(next_color)
            r, c = _add(partner, direction)
            continue

        if int(g[r, c]) != bg:
            break
        if int(out[r, c]) == bg:
            out[r, c] = current_color
        r += direction[0]
        c += direction[1]

    return out, events


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    bg, tri_color, tip, direction, domino_map = _parse_scene(np.asarray(grid, dtype=int))
    _ = bg
    return _trace_ray(np.asarray(grid, dtype=int), tip, direction, tri_color, domino_map)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a color-changing ray from the triangle tip through hit dominoes until it exits the grid.

    Input structure:
    - A black background contains one solid triangular arrow sprite whose tip faces one cardinal direction.
    - The other foreground objects are isolated bicolor dominoes: exactly two orthogonally adjacent cells of different colors.
    - Some dominoes lie on the ray route and some are distractors that are never reached.

    Rule:
    - Start the ray one cell in front of the triangle tip and let the initial ray color be the triangle color.
    - While the ray moves through background cells, paint those traversed background cells with the current ray color.
    - When the ray first enters one half of a domino, turn to the direction from the struck half toward the other half, then continue from the cell just beyond that other half; from then on the ray color becomes the other half's color.
    - Leave the triangle and every domino cell unchanged; stop when the ray leaves the grid.

    Key insight:
    - Each hit domino encodes both instructions geometrically: its axis gives the 90-degree deflection, and its far half gives the next segment color.
    """
    out, _events = _solve(np.asarray(grid, dtype=int))
    return out


def _random_color(exclude: set[int] | None = None) -> int:
    ban = set() if exclude is None else set(map(int, exclude))
    pool = [c for c in GEN_COLORS if c not in ban]
    if not pool:
        pool = GEN_COLORS[:]
    return int(random.choice(pool))


def _ray_cells_to_edge(shape: tuple[int, int], start: Coord, direction: Direction) -> list[Coord]:
    cells: list[Coord] = []
    r, c = start
    while in_bounds(shape, r, c):
        cells.append((r, c))
        r += direction[0]
        c += direction[1]
    return cells


def _plan_route(
    shape: tuple[int, int],
    start: Coord,
    direction: Direction,
    occupied: set[Coord],
    turns_left: int,
) -> tuple[list[list[Coord]], list[tuple[Coord, Coord]]] | None:
    reserved = set(occupied)
    object_cells = set(occupied)
    segments: list[list[Coord]] = []
    dominoes: list[tuple[Coord, Coord]] = []

    def domino_isolated(a: Coord, b: Coord) -> bool:
        pair = {a, b}
        for cell in pair:
            for nb in neighbors4(*cell):
                if nb in object_cells and nb not in pair:
                    return False
        return True

    def dfs(curr: Coord, dir_: Direction, remaining: int) -> bool:
        if curr in reserved or not in_bounds(shape, *curr):
            return False

        if remaining == 0:
            cells = _ray_cells_to_edge(shape, curr, dir_)
            if len(cells) < GEN_MIN_FINAL or any(cell in reserved for cell in cells):
                return False
            reserved.update(cells)
            segments.append(cells)
            return True

        turn_dirs = PERP_DIRS[dir_][:]
        random.shuffle(turn_dirs)

        distances = list(range(GEN_MIN_SEG, GEN_MAX_SEG + 1))
        random.shuffle(distances)

        for new_dir in turn_dirs:
            for dist in distances:
                cells = [_add(curr, _mul(dir_, k)) for k in range(dist)]
                hit = _add(curr, _mul(dir_, dist))
                partner = _add(hit, new_dir)
                next_curr = _add(partner, new_dir)
                needed = cells + [hit, partner]

                if not all(in_bounds(shape, *cell) for cell in needed):
                    continue
                if next_curr in reserved or not in_bounds(shape, *next_curr):
                    continue
                if any(cell in reserved for cell in needed):
                    continue
                if not domino_isolated(hit, partner):
                    continue

                reserved.update(needed)
                object_cells.update((hit, partner))
                segments.append(cells)
                dominoes.append((hit, partner))
                if dfs(next_curr, new_dir, remaining - 1):
                    return True
                dominoes.pop()
                segments.pop()
                object_cells.remove(hit)
                object_cells.remove(partner)
                for cell in needed:
                    reserved.remove(cell)

        return False

    if dfs(start, direction, turns_left):
        return segments, dominoes
    return None


def _halo(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors8(r, c):
            if in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _place_distractors(grid: np.ndarray, forbidden: set[Coord]) -> list[tuple[Coord, Coord, int, int]] | None:
    shape = grid.shape
    target = random.randint(*GEN_DISTRACTOR_RANGE)
    blocked = _halo(set(forbidden), shape)
    candidates: list[tuple[Coord, Coord]] = []

    for r in range(shape[0]):
        for c in range(shape[1]):
            for dr, dc in ((0, 1), (1, 0)):
                a = (r, c)
                b = (r + dr, c + dc)
                if not in_bounds(shape, *b):
                    continue
                if a in blocked or b in blocked:
                    continue
                candidates.append((a, b))

    random.shuffle(candidates)
    placed: list[tuple[Coord, Coord, int, int]] = []
    for a, b in candidates:
        if len(placed) >= target:
            break
        if int(grid[a]) != BG or int(grid[b]) != BG:
            continue
        color_a = _random_color()
        color_b = _random_color({color_a})
        grid[a] = color_a
        grid[b] = color_b
        placed.append((a, b, color_a, color_b))
        blocked |= _halo({a, b}, shape)

    if len(placed) < target:
        return None
    return placed


def _build_expected(
    inp: np.ndarray,
    segments: list[list[Coord]],
    segment_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for cells, color in zip(segments, segment_colors):
        for r, c in cells:
            assert int(inp[r, c]) == BG
            out[r, c] = int(color)
    return out


def _validate_constructed(
    inp: np.ndarray,
    expected: np.ndarray,
    segments: list[list[Coord]],
    path_dominoes: list[tuple[Coord, Coord, int, int]],
    distractors: list[tuple[Coord, Coord, int, int]],
) -> bool:
    try:
        out, events = _solve(inp)
    except AssertionError:
        return False
    if not np.array_equal(out, expected):
        return False

    if len(events) != len(path_dominoes):
        return False

    changed = {tuple(map(int, p)) for p in np.argwhere(out != inp)}
    painted = {cell for seg in segments for cell in seg}
    if changed != painted:
        return False

    for event, spec in zip(events, path_dominoes):
        hit, partner, hit_color, partner_color = spec
        if event["hit"] != hit or event["partner"] != partner:
            return False
        if event["outgoing"] != (partner[0] - hit[0], partner[1] - hit[1]):
            return False
        if int(event["color_out"]) != int(partner_color):
            return False
        if int(inp[hit]) != int(hit_color) or int(inp[partner]) != int(partner_color):
            return False

    for a, b, _, _ in path_dominoes + distractors:
        if int(out[a]) != int(inp[a]) or int(out[b]) != int(inp[b]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    turn_count = random.choices(GEN_TURN_COUNTS, weights=GEN_TURN_WEIGHTS, k=1)[0]
    min_dim = 15 + 2 * turn_count
    h = random.randint(min_dim, 26)
    w = random.randint(min_dim, 26)
    shape = (h, w)

    tri_name = random.choice(list(TRIANGLE_TEMPLATES))
    direction = random.choice(DIRS4)

    offsets = _triangle_offsets(tri_name, direction)
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    tip_r_lo = -min(rows)
    tip_r_hi = h - 1 - max(rows)
    tip_c_lo = -min(cols)
    tip_c_hi = w - 1 - max(cols)
    if tip_r_lo > tip_r_hi or tip_c_lo > tip_c_hi:
        return None

    tips = [(r, c) for r in range(tip_r_lo, tip_r_hi + 1) for c in range(tip_c_lo, tip_c_hi + 1)]
    random.shuffle(tips)

    for tip in tips[:120]:
        start = _add(tip, direction)
        if not in_bounds(shape, *start):
            continue

        tri_cells = _triangle_cells(tri_name, tip, direction)
        occupied = set(tri_cells)
        route = _plan_route(shape, start, direction, occupied, turn_count)
        if route is None:
            continue
        segments, domino_locs = route

        triangle_color = _random_color()
        segment_colors = [triangle_color]
        path_dominoes: list[tuple[Coord, Coord, int, int]] = []
        current_color = triangle_color
        for hit, partner in domino_locs:
            partner_color = _random_color({current_color})
            hit_color = _random_color({current_color, partner_color})
            path_dominoes.append((hit, partner, hit_color, partner_color))
            current_color = partner_color
            segment_colors.append(current_color)

        inp = np.full(shape, BG, dtype=int)
        for r, c in tri_cells:
            inp[r, c] = triangle_color
        for hit, partner, hit_color, partner_color in path_dominoes:
            inp[hit] = hit_color
            inp[partner] = partner_color

        forbidden = set(tri_cells)
        forbidden.update(cell for seg in segments for cell in seg)
        for hit, partner, _, _ in path_dominoes:
            forbidden.add(hit)
            forbidden.add(partner)

        distractors = _place_distractors(inp, forbidden)
        if distractors is None:
            continue

        expected = _build_expected(inp, segments, segment_colors)
        if not _validate_constructed(inp, expected, segments, path_dominoes, distractors):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-ray input")
