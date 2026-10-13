from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 120


@dataclass(frozen=True)
class Marker:
    host_color: int
    target_color: int
    size: int
    outdir: tuple[int, int]
    cells: tuple[tuple[int, int], ...]
    attach: tuple[tuple[int, int], ...]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def bbox(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def solid_square_size(cells: list[tuple[int, int]]) -> int:
    if not cells:
        return 0
    r0, r1, c0, c1 = bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h not in (1, 2):
        return 0
    expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return h if expected == set(cells) else 0


def sort_strip(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], outdir: tuple[int, int]):
    if outdir[0] == 0:
        return tuple(sorted(cells, key=lambda p: (p[0], p[1])))
    return tuple(sorted(cells, key=lambda p: (p[1], p[0])))


def marker_sides(
    cells: list[tuple[int, int]], outdir: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    r0, r1, c0, c1 = bbox(cells)
    if outdir == (0, 1):
        inner_face = [(r, c0) for r in range(r0, r1 + 1)]
        outer_face = [(r, c1) for r in range(r0, r1 + 1)]
    elif outdir == (0, -1):
        inner_face = [(r, c1) for r in range(r0, r1 + 1)]
        outer_face = [(r, c0) for r in range(r0, r1 + 1)]
    elif outdir == (1, 0):
        inner_face = [(r0, c) for c in range(c0, c1 + 1)]
        outer_face = [(r1, c) for c in range(c0, c1 + 1)]
    else:
        inner_face = [(r1, c) for c in range(c0, c1 + 1)]
        outer_face = [(r0, c) for c in range(c0, c1 + 1)]
    inside = [(r - outdir[0], c - outdir[1]) for r, c in inner_face]
    attach = [(r + outdir[0], c + outdir[1]) for r, c in outer_face]
    return inner_face, inside, attach


def detect_marker(
    grid: np.ndarray,
    bg: int,
    islands: dict[int, set[tuple[int, int]]],
    marker_color: int,
    cells: list[tuple[int, int]],
) -> Marker | None:
    h, w = grid.shape
    size = solid_square_size(cells)
    if size == 0:
        return None

    candidates: list[Marker] = []
    cell_set = set(cells)
    for outdir in DIRS4:
        _face, inside, attach = marker_sides(cells, outdir)
        if not all(0 <= r < h and 0 <= c < w for r, c in inside):
            continue
        if not all(0 <= r < h and 0 <= c < w for r, c in attach):
            continue
        host_colors = {int(grid[r, c]) for r, c in inside}
        if len(host_colors) != 1:
            continue
        host_color = next(iter(host_colors))
        if host_color in (bg, marker_color):
            continue
        island = islands.get(host_color)
        if island is None or not set(inside).issubset(island):
            continue
        if any(int(grid[r, c]) != bg for r, c in attach):
            continue

        touchers = set()
        for r, c in cell_set:
            for nr, nc in neighbors4(r, c, h, w):
                if (nr, nc) in island:
                    touchers.add((nr, nc))
        if touchers != set(inside):
            continue

        candidates.append(
            Marker(
                host_color=host_color,
                target_color=marker_color,
                size=size,
                outdir=outdir,
                cells=tuple(sorted(cell_set)),
                attach=sort_strip(attach, outdir),
            )
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def detect_islands_and_markers(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    color_comps = {color: components_of_color(g, color) for color in colors}

    islands: dict[int, set[tuple[int, int]]] = {}
    markers: list[Marker] = []
    for color, comps in color_comps.items():
        comps = sorted(comps, key=lambda comp: (-len(comp), bbox(comp)))
        if not comps:
            continue
        islands[color] = set(comps[0])

    for color, comps in color_comps.items():
        comps = sorted(comps, key=lambda comp: (-len(comp), bbox(comp)))
        for comp in comps[1:]:
            marker = detect_marker(g, bg, islands, color, comp)
            if marker is not None:
                markers.append(marker)
    return bg, islands, markers


def straight_bridge(start: Marker, end: Marker) -> set[tuple[int, int]] | None:
    if start.outdir != (-end.outdir[0], -end.outdir[1]):
        return None
    if start.size != end.size:
        return None

    if start.outdir[0] == 0:
        start_rows = [r for r, _ in start.attach]
        end_rows = [r for r, _ in end.attach]
        if start_rows != end_rows:
            return None
        c1 = start.attach[0][1]
        c2 = end.attach[0][1]
        if start.outdir[1] == 1 and c1 > c2:
            return None
        if start.outdir[1] == -1 and c1 < c2:
            return None
        return {
            (r, c)
            for r in start_rows
            for c in range(min(c1, c2), max(c1, c2) + 1)
        }

    start_cols = [c for _, c in start.attach]
    end_cols = [c for _, c in end.attach]
    if start_cols != end_cols:
        return None
    r1 = start.attach[0][0]
    r2 = end.attach[0][0]
    if start.outdir[0] == 1 and r1 > r2:
        return None
    if start.outdir[0] == -1 and r1 < r2:
        return None
    return {
        (r, c)
        for r in range(min(r1, r2), max(r1, r2) + 1)
        for c in start_cols
    }


def l_bridge(start: Marker, end: Marker) -> set[tuple[int, int]] | None:
    if start.size != end.size:
        return None
    finish = (-end.outdir[0], -end.outdir[1])
    if start.outdir[0] * finish[0] + start.outdir[1] * finish[1] != 0:
        return None

    if start.outdir[0] == 0 and finish[1] == 0:
        if len({c for _, c in start.attach}) != 1 or len({r for r, _ in end.attach}) != 1:
            return None
        rows = sorted(r for r, _ in start.attach)
        start_col = start.attach[0][1]
        end_cols = sorted(c for _, c in end.attach)
        end_row = end.attach[0][0]

        if start.outdir[1] == 1 and start_col > min(end_cols):
            return None
        if start.outdir[1] == -1 and start_col < max(end_cols):
            return None
        if finish[0] == 1 and max(rows) > end_row:
            return None
        if finish[0] == -1 and min(rows) < end_row:
            return None

        h_cols = range(min(start_col, min(end_cols)), max(start_col, max(end_cols)) + 1)
        v_rows = range(min(min(rows), end_row), max(max(rows), end_row) + 1)
        return (
            {(r, c) for r in rows for c in h_cols}
            | {(r, c) for r in v_rows for c in end_cols}
        )

    if start.outdir[1] == 0 and finish[0] == 0:
        if len({r for r, _ in start.attach}) != 1 or len({c for _, c in end.attach}) != 1:
            return None
        start_row = start.attach[0][0]
        start_cols = sorted(c for _, c in start.attach)
        end_rows = sorted(r for r, _ in end.attach)
        end_col = end.attach[0][1]

        if start.outdir[0] == 1 and start_row > min(end_rows):
            return None
        if start.outdir[0] == -1 and start_row < max(end_rows):
            return None
        if finish[1] == 1 and max(start_cols) > end_col:
            return None
        if finish[1] == -1 and min(start_cols) < end_col:
            return None

        v_rows = range(min(start_row, min(end_rows)), max(start_row, max(end_rows)) + 1)
        h_cols = range(min(min(start_cols), end_col), max(max(start_cols), end_col) + 1)
        return (
            {(r, c) for r in v_rows for c in start_cols}
            | {(r, c) for r in end_rows for c in h_cols}
        )

    return None


def bridge_cells_for_pair(
    marker_a: Marker, marker_b: Marker, grid: np.ndarray, bg: int
) -> set[tuple[int, int]]:
    candidates: list[frozenset[tuple[int, int]]] = []
    for start, end in ((marker_a, marker_b), (marker_b, marker_a)):
        for builder in (straight_bridge, l_bridge):
            cells = builder(start, end)
            if not cells:
                continue
            if all(int(grid[r, c]) == bg for r, c in cells):
                candidates.append(frozenset(cells))
    unique = sorted(set(candidates), key=lambda s: (len(s), sorted(s)))
    assert unique, "no valid bridge path for reciprocal marker pair"
    return set(unique[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect mutually referenced islands with width-coded orthogonal grey bridges.

    Each island is the largest component of its color; smaller 1x1 or 2x2 square
    components of that color are treated as markers on other islands. If island A
    hosts a marker colored like island B, and island B hosts a reciprocal marker
    colored like island A, add a grey bridge between the two marker faces. The
    bridge width equals the marker size, and the route is either straight or a
    single shortest right-angle turn.
    """
    g = np.asarray(grid, dtype=int)
    bg, _islands, markers = detect_islands_and_markers(g)

    by_key: dict[tuple[int, int], list[Marker]] = {}
    for marker in markers:
        by_key.setdefault((marker.host_color, marker.target_color), []).append(marker)

    out = g.copy()
    done_pairs: set[frozenset[int]] = set()
    for (a, b), ab_markers in sorted(by_key.items()):
        pair_key = frozenset((a, b))
        if a == b or pair_key in done_pairs:
            continue
        ba_markers = by_key.get((b, a), [])
        if not ba_markers:
            continue

        bridge = None
        for marker_a in ab_markers:
            for marker_b in ba_markers:
                if marker_a.size != marker_b.size:
                    continue
                try:
                    candidate = bridge_cells_for_pair(marker_a, marker_b, g, bg)
                except AssertionError:
                    continue
                if bridge is None or len(candidate) < len(bridge):
                    bridge = candidate

        assert bridge is not None, f"reciprocal pair {a}-{b} has no valid bridge"
        for r, c in bridge:
            out[r, c] = GREY
        done_pairs.add(pair_key)

    return out


def rotate_point(point: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = point
    for _ in range(k % 4):
        r, c = c, -r
    return r, c


def rotate_dir(direction: tuple[int, int], k: int) -> tuple[int, int]:
    return rotate_point(direction, k)


def marker_layout_from_attach(
    attach: tuple[tuple[int, int], ...],
    outdir: tuple[int, int],
    depth: int,
    pad_lo: int,
    pad_hi: int,
):
    size = len(attach)
    if outdir == (0, 1):
        rows = sorted(r for r, _ in attach)
        ac = attach[0][1]
        marker = {(r, c) for r in rows for c in range(ac - size, ac)}
        seeds = {(r, ac - size - 1) for r in rows}
        zone = {
            (r, c)
            for r in range(rows[0] - pad_lo, rows[-1] + pad_hi + 1)
            for c in range(ac - size - depth, ac - size)
        }
    elif outdir == (0, -1):
        rows = sorted(r for r, _ in attach)
        ac = attach[0][1]
        marker = {(r, c) for r in rows for c in range(ac + 1, ac + size + 1)}
        seeds = {(r, ac + size + 1) for r in rows}
        zone = {
            (r, c)
            for r in range(rows[0] - pad_lo, rows[-1] + pad_hi + 1)
            for c in range(ac + size + 1, ac + size + depth + 1)
        }
    elif outdir == (1, 0):
        cols = sorted(c for _, c in attach)
        ar = attach[0][0]
        marker = {(r, c) for r in range(ar - size, ar) for c in cols}
        seeds = {(ar - size - 1, c) for c in cols}
        zone = {
            (r, c)
            for r in range(ar - size - depth, ar - size)
            for c in range(cols[0] - pad_lo, cols[-1] + pad_hi + 1)
        }
    else:
        cols = sorted(c for _, c in attach)
        ar = attach[0][0]
        marker = {(r, c) for r in range(ar + 1, ar + size + 1) for c in cols}
        seeds = {(ar + size + 1, c) for c in cols}
        zone = {
            (r, c)
            for r in range(ar + size + 1, ar + size + depth + 1)
            for c in range(cols[0] - pad_lo, cols[-1] + pad_hi + 1)
        }
    return marker, seeds, zone


def grow_blob(
    zone: set[tuple[int, int]],
    seeds: set[tuple[int, int]],
    target_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    if not seeds or not seeds.issubset(zone):
        return None

    blob = set(seeds)
    frontier = set()
    for r, c in blob:
        for dr, dc in DIRS4:
            cand = (r + dr, c + dc)
            if cand in zone and cand not in blob:
                frontier.add(cand)

    while len(blob) < target_size and frontier:
        ranked = []
        for cand in frontier:
            deg = sum((cand[0] + dr, cand[1] + dc) in blob for dr, dc in DIRS4)
            score = (
                2 if deg == 1 else 1 if deg == 2 else 0,
                rng.random(),
                -abs(cand[0] - next(iter(seeds))[0]) - abs(cand[1] - next(iter(seeds))[1]),
            )
            ranked.append((score, cand))
        ranked.sort(reverse=True)
        choice = ranked[0][1]
        blob.add(choice)
        frontier.remove(choice)
        for dr, dc in DIRS4:
            cand = (choice[0] + dr, choice[1] + dc)
            if cand in zone and cand not in blob:
                frontier.add(cand)

    if len(blob) < target_size:
        return None

    r0, r1, c0, c1 = bbox(blob)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if len(blob) == area:
        removable = []
        for cell in sorted(blob):
            if cell in seeds:
                continue
            rest = blob - {cell}
            start = next(iter(rest))
            q = deque([start])
            seen = {start}
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nxt = (cr + dr, cc + dc)
                    if nxt in rest and nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            if seen == rest:
                removable.append(cell)
        if removable:
            blob.remove(rng.choice(removable))
    return blob


def dilate_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1):
    out = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def shift_cells(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...], dr: int, dc: int):
    return {(r + dr, c + dc) for r, c in cells}


def build_local_pair(
    size: int, branch: str, rng: random.Random
) -> tuple[
    Marker,
    Marker,
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
] | None:
    if branch == "straight":
        length = rng.randint(6, 11)
        attach_a = tuple((r, 0) for r in range(size))
        attach_b = tuple((r, length - 1) for r in range(size))
        dir_a = (0, 1)
        dir_b = (0, -1)
    elif rng.random() < 0.5:
        dx = rng.randint(3, 8)
        dy = rng.randint(size + 1, size + 7)
        attach_a = tuple((r, 0) for r in range(size))
        attach_b = tuple((dy, dx + i) for i in range(size))
        dir_a = (0, 1)
        dir_b = (-1, 0)
    else:
        dx = rng.randint(3, 8)
        dy = rng.randint(size + 1, size + 7)
        attach_a = tuple((dy + r, 0) for r in range(size))
        attach_b = tuple((0, dx + i) for i in range(size))
        dir_a = (0, 1)
        dir_b = (1, 0)

    rot = rng.randrange(4)
    attach_a = sort_strip([rotate_point(p, rot) for p in attach_a], rotate_dir(dir_a, rot))
    attach_b = sort_strip([rotate_point(p, rot) for p in attach_b], rotate_dir(dir_b, rot))
    dir_a = rotate_dir(dir_a, rot)
    dir_b = rotate_dir(dir_b, rot)

    marker_a = Marker(1, 2, size, dir_a, (), attach_a)
    marker_b = Marker(2, 1, size, dir_b, (), attach_b)
    bridge = straight_bridge(marker_a, marker_b)
    if bridge is None:
        bridge = l_bridge(marker_a, marker_b)
    if bridge is None:
        bridge = straight_bridge(marker_b, marker_a)
    if bridge is None:
        bridge = l_bridge(marker_b, marker_a)
    if bridge is None:
        return None

    depth_a = rng.randint(4, 7)
    depth_b = rng.randint(4, 7)
    pad_a0, pad_a1 = rng.randint(1, 3), rng.randint(1, 3)
    pad_b0, pad_b1 = rng.randint(1, 3), rng.randint(1, 3)
    marker_cells_a, seeds_a, zone_a = marker_layout_from_attach(attach_a, dir_a, depth_a, pad_a0, pad_a1)
    marker_cells_b, seeds_b, zone_b = marker_layout_from_attach(attach_b, dir_b, depth_b, pad_b0, pad_b1)
    all_cells = bridge | marker_cells_a | marker_cells_b | zone_a | zone_b
    return (
        marker_a,
        marker_b,
        bridge,
        marker_cells_a,
        marker_cells_b,
        seeds_a,
        seeds_b,
        zone_a,
        zone_b,
        all_cells,
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(26, 34)
    w = rng.randint(26, 34)
    bg = rng.choice(COLOR_POOL)
    grid = np.full((h, w), bg, dtype=int)
    forbidden: set[tuple[int, int]] = set()

    pair_count = rng.choice([2, 3])
    island_colors = rng.sample([c for c in COLOR_POOL if c != bg], 2 * pair_count)
    pair_specs = []

    for pair_idx in range(pair_count):
        placed = False
        branch = rng.choice(["straight", "L"])
        size = rng.choice([1, 2])
        color_a = island_colors[2 * pair_idx]
        color_b = island_colors[2 * pair_idx + 1]

        for _ in range(PAIR_ATTEMPTS):
            built = build_local_pair(size, branch, rng)
            if built is None:
                continue
            (
                marker_a,
                marker_b,
                bridge_local,
                marker_cells_a_local,
                marker_cells_b_local,
                seeds_a_local,
                seeds_b_local,
                zone_a_local,
                zone_b_local,
                all_local,
            ) = built

            r0, r1, c0, c1 = bbox(all_local)
            min_dr = 1 - r0
            max_dr = (h - 2) - r1
            min_dc = 1 - c0
            max_dc = (w - 2) - c1
            if min_dr > max_dr or min_dc > max_dc:
                continue
            dr = rng.randint(min_dr, max_dr)
            dc = rng.randint(min_dc, max_dc)

            bridge = shift_cells(bridge_local, dr, dc)
            marker_cells_a = shift_cells(marker_cells_a_local, dr, dc)
            marker_cells_b = shift_cells(marker_cells_b_local, dr, dc)
            seeds_a = shift_cells(seeds_a_local, dr, dc)
            seeds_b = shift_cells(seeds_b_local, dr, dc)
            zone_a = shift_cells(zone_a_local, dr, dc)
            zone_b = shift_cells(zone_b_local, dr, dc)

            reserved = shift_cells(all_local, dr, dc)
            if any(cell in forbidden for cell in reserved):
                continue
            if len(zone_a) < len(seeds_a) + 5 or len(zone_b) < len(seeds_b) + 5:
                continue

            blob_a = grow_blob(zone_a, seeds_a, rng.randint(len(seeds_a) + 6, min(len(zone_a), 14)), rng)
            blob_b = grow_blob(zone_b, seeds_b, rng.randint(len(seeds_b) + 6, min(len(zone_b), 14)), rng)
            if blob_a is None or blob_b is None:
                continue

            island_a = blob_a
            island_b = blob_b
            if island_a & forbidden or island_b & forbidden:
                continue
            if island_a & (marker_cells_a | marker_cells_b | bridge):
                continue
            if island_b & (marker_cells_a | marker_cells_b | bridge):
                continue
            if island_a & island_b:
                continue

            pair_specs.append(
                {
                    "bridge": bridge,
                    "marker_a": marker_cells_a,
                    "marker_b": marker_cells_b,
                    "island_a": island_a,
                    "island_b": island_b,
                    "color_a": color_a,
                    "color_b": color_b,
                }
            )
            forbidden |= dilate_cells(reserved | island_a | island_b, h, w, pad=1)
            placed = True
            break

        if not placed:
            return None

    for spec in pair_specs:
        for r, c in spec["island_a"]:
            grid[r, c] = spec["color_a"]
        for r, c in spec["island_b"]:
            grid[r, c] = spec["color_b"]
        for r, c in spec["marker_a"]:
            grid[r, c] = spec["color_b"]
        for r, c in spec["marker_b"]:
            grid[r, c] = spec["color_a"]

    if np.count_nonzero(grid != bg) < 30:
        return None
    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == GREY)) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-island input")
