from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GRAY = 5
TOWER_PALETTE = (1, 2, 3, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (12, 19)
TOWER_COUNT_CHOICES = (2, 3)
MIRROR_COUNT_RANGE = (1, 4)
PLACEMENT_RETRIES = 160
GENERATE_RETRIES = 400
MIN_TOWER_MARGIN = 2
MIN_TOWER_GAP = 4
MAX_PAINT_FRAC_NUM = 9
MAX_PAINT_FRAC_DEN = 20


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _reflect(direction: tuple[int, int], mirror_kind: str) -> tuple[int, int]:
    dr, dc = direction
    if mirror_kind == "\\":
        return dc, dr
    assert mirror_kind == "/"
    return -dc, -dr


def _find_mirror_kinds(g: np.ndarray) -> tuple[dict[tuple[int, int], str], list[tuple[tuple[int, int], tuple[int, int], str]]]:
    mirror_map: dict[tuple[int, int], str] = {}
    mirror_pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for comp in _components(g == GRAY, connectivity=8):
        assert len(comp) == 2, "each gray mirror must be a 2-pixel component"
        (r0, c0), (r1, c1) = sorted(comp)
        dr = r1 - r0
        dc = c1 - c0
        assert abs(dr) == 1 and abs(dc) == 1, "mirror pixels must be diagonal neighbors"
        kind = "\\" if dr == dc else "/"
        mirror_map[(r0, c0)] = kind
        mirror_map[(r1, c1)] = kind
        mirror_pairs.append(((r0, c0), (r1, c1), kind))
    return mirror_map, mirror_pairs


def _find_towers(g: np.ndarray) -> list[dict[str, object]]:
    towers: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BLACK, GRAY}):
        coords = np.argwhere(g == int(color))
        assert coords.shape == (1, 2), "each tower color must appear exactly once"
        r, c = map(int, coords[0])
        towers.append({"pos": (r, c), "color": int(color)})
    assert 2 <= len(towers) <= 3, "expected 2 or 3 towers"
    return towers


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(g.min()) >= 0
    h, w = g.shape

    towers = _find_towers(g)
    tower_positions = {tuple(tower["pos"]) for tower in towers}
    mirror_map, mirror_pairs = _find_mirror_kinds(g)

    visits: dict[tuple[int, int], set[int]] = defaultdict(set)
    mirror_hit_cells: set[tuple[int, int]] = set()
    mirror_hit_kinds: set[str] = set()
    tower_stops = 0
    boundary_stops = 0
    loops = 0

    for tower in towers:
        start = tuple(tower["pos"])
        color = int(tower["color"])
        for dr0, dc0 in DIRS4:
            r, c = start
            dr, dc = dr0, dc0
            seen_states = {(r, c, dr, dc)}

            while True:
                nr = r + dr
                nc = c + dc
                if not _in_bounds(h, w, nr, nc):
                    boundary_stops += 1
                    break
                if (nr, nc) in tower_positions:
                    tower_stops += 1
                    break

                r, c = nr, nc
                if int(g[r, c]) == GRAY:
                    kind = mirror_map[(r, c)]
                    mirror_hit_cells.add((r, c))
                    mirror_hit_kinds.add(kind)
                    dr, dc = _reflect((dr, dc), kind)
                else:
                    visits[(r, c)].add(color)

                state = (r, c, dr, dc)
                if state in seen_states:
                    loops += 1
                    break
                seen_states.add(state)

    out = g.copy()
    for (r, c), colors in visits.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        else:
            out[r, c] = YELLOW

    stats = {
        "tower_count": len(towers),
        "mirror_count": len(mirror_pairs),
        "mirror_hit_cells": mirror_hit_cells,
        "mirror_hit_kinds": mirror_hit_kinds,
        "painted_cells": len(visits),
        "yellow_cells": int(sum(1 for colors in visits.values() if len(colors) > 1)),
        "tower_stops": tower_stops,
        "boundary_stops": boundary_stops,
        "loops": loops,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four axial rays from each singleton tower, reflecting 90 degrees on gray diagonal mirrors.

    Input structure:
    - Background is black.
    - There are 2-3 single-pixel towers of distinct colors.
    - Gray pixels come in isolated diagonal pairs; each pair is a "\" or "/" mirror.

    Rule:
    - Each tower fires rays up, down, left, and right.
    - Rays advance cell by cell until the boundary or another tower.
    - Entering a gray mirror cell turns the ray 90 degrees according to the mirror orientation.
    - Mirror cells stay gray.
    - Background cells visited by one tower take that tower's color; cells visited by multiple tower colors become yellow.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int))
    return out


def _candidate_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(MIN_TOWER_MARGIN, h - MIN_TOWER_MARGIN)
        for c in range(MIN_TOWER_MARGIN, w - MIN_TOWER_MARGIN)
    ]


def _place_scattered_towers(h: int, w: int, tower_count: int) -> tuple[list[tuple[int, int]], dict[str, object]] | None:
    cells = _candidate_positions(h, w)
    for _ in range(PLACEMENT_RETRIES):
        random.shuffle(cells)
        towers: list[tuple[int, int]] = []
        for cell in cells:
            if any(
                cell[0] == other[0] or cell[1] == other[1] or _chebyshev(cell, other) < MIN_TOWER_GAP
                for other in towers
            ):
                continue
            towers.append(cell)
            if len(towers) == tower_count:
                break
        if len(towers) == tower_count:
            return towers, {"aligned": False, "blocked_rays": set()}
    return None


def _place_aligned_towers(h: int, w: int, tower_count: int) -> tuple[list[tuple[int, int]], dict[str, object]] | None:
    axis = random.choice(["row", "col"])
    for _ in range(PLACEMENT_RETRIES):
        if axis == "row":
            if h < 2 * MIN_TOWER_MARGIN + 1 or w < 2 * MIN_TOWER_MARGIN + MIN_TOWER_GAP + 1:
                return None
            row = random.randint(MIN_TOWER_MARGIN, h - MIN_TOWER_MARGIN - 1)
            c1 = random.randint(MIN_TOWER_MARGIN, w - MIN_TOWER_MARGIN - MIN_TOWER_GAP - 1)
            c2 = random.randint(c1 + MIN_TOWER_GAP, w - MIN_TOWER_MARGIN - 1)
            left = (row, c1)
            right = (row, c2)
            towers = [left, right]
            blocked_rays = {(0, (0, 1)), (1, (0, -1))}
        else:
            if w < 2 * MIN_TOWER_MARGIN + 1 or h < 2 * MIN_TOWER_MARGIN + MIN_TOWER_GAP + 1:
                return None
            col = random.randint(MIN_TOWER_MARGIN, w - MIN_TOWER_MARGIN - 1)
            r1 = random.randint(MIN_TOWER_MARGIN, h - MIN_TOWER_MARGIN - MIN_TOWER_GAP - 1)
            r2 = random.randint(r1 + MIN_TOWER_GAP, h - MIN_TOWER_MARGIN - 1)
            top = (r1, col)
            bottom = (r2, col)
            towers = [top, bottom]
            blocked_rays = {(0, (1, 0)), (1, (-1, 0))}

        if tower_count == 3:
            choices = _candidate_positions(h, w)
            random.shuffle(choices)
            third = None
            for cell in choices:
                if any(
                    cell[0] == other[0] or cell[1] == other[1] or _chebyshev(cell, other) < MIN_TOWER_GAP
                    for other in towers
                ):
                    continue
                third = cell
                break
            if third is None:
                continue
            towers.append(third)

        return towers, {"aligned": True, "axis": axis, "blocked_rays": blocked_rays}
    return None


def _neighboring_feature(
    cell: tuple[int, int],
    tower_positions: set[tuple[int, int]],
    gray_cells: set[tuple[int, int]],
    allow: set[tuple[int, int]] | None = None,
) -> bool:
    allow = allow or set()
    r, c = cell
    for nr in range(r - 1, r + 2):
        for nc in range(c - 1, c + 2):
            pos = (nr, nc)
            if pos in allow:
                continue
            if pos in tower_positions or pos in gray_cells:
                return True
    return False


def _pair_clear(
    pair: tuple[tuple[int, int], tuple[int, int]],
    h: int,
    w: int,
    tower_positions: set[tuple[int, int]],
    gray_cells: set[tuple[int, int]],
) -> bool:
    cells = set(pair)
    for r, c in cells:
        if not _in_bounds(h, w, r, c):
            return False
        if (r, c) in tower_positions or (r, c) in gray_cells:
            return False
    for cell in cells:
        if _neighboring_feature(cell, tower_positions, gray_cells, allow=cells):
            return False
    return True


def _free_run_len(
    start: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    tower_positions: set[tuple[int, int]],
    gray_cells: set[tuple[int, int]],
) -> int:
    r, c = start
    dr, dc = direction
    steps = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(h, w, r, c):
            return steps
        if (r, c) in tower_positions or (r, c) in gray_cells:
            return steps
        steps += 1


def _partner_candidates(mirror_cell: tuple[int, int], kind: str) -> list[tuple[int, int]]:
    r, c = mirror_cell
    if kind == "\\":
        return [(r - 1, c - 1), (r + 1, c + 1)]
    return [(r - 1, c + 1), (r + 1, c - 1)]


def _place_mirror_on_ray(
    h: int,
    w: int,
    tower_pos: tuple[int, int],
    direction: tuple[int, int],
    tower_positions: set[tuple[int, int]],
    gray_cells: set[tuple[int, int]],
    forced_kind: str | None,
) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r, c = tower_pos
    while True:
        r += dr
        c += dc
        if not _in_bounds(h, w, r, c):
            break
        if (r, c) in tower_positions or (r, c) in gray_cells:
            break
        cells.append((r, c))

    if len(cells) < 3:
        return None

    random.shuffle(cells)
    for mirror_cell in cells:
        if abs(mirror_cell[0] - tower_pos[0]) + abs(mirror_cell[1] - tower_pos[1]) < 2:
            continue
        kinds = [forced_kind] if forced_kind is not None else ["\\", "/"]
        random.shuffle(kinds)
        for kind in kinds:
            out_dir = _reflect(direction, kind)
            if _free_run_len(mirror_cell, out_dir, h, w, tower_positions, gray_cells) < 2:
                continue
            partners = _partner_candidates(mirror_cell, kind)
            random.shuffle(partners)
            for partner in partners:
                pair = (mirror_cell, partner)
                if not _pair_clear(pair, h, w, tower_positions, gray_cells):
                    continue
                return mirror_cell, partner, kind
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    tower_count = random.choice(TOWER_COUNT_CHOICES)
    want_aligned = random.random() < 0.5

    placed = _place_aligned_towers(h, w, tower_count) if want_aligned else _place_scattered_towers(h, w, tower_count)
    if placed is None:
        return None

    tower_positions, layout = placed
    colors = random.sample(TOWER_PALETTE, tower_count)
    grid = np.full((h, w), BLACK, dtype=int)
    for pos, color in zip(tower_positions, colors):
        grid[pos] = int(color)

    tower_pos_set = set(tower_positions)
    gray_cells: set[tuple[int, int]] = set()
    mirror_pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []

    ray_specs = [
        (tower_index, direction)
        for tower_index in range(tower_count)
        for direction in DIRS4
        if (tower_index, direction) not in layout["blocked_rays"]
    ]
    random.shuffle(ray_specs)

    target_mirrors = random.randint(*MIRROR_COUNT_RANGE)
    target_mirrors = min(target_mirrors, len(ray_specs))
    need_both_kinds = target_mirrors >= 2 and random.random() < 0.6
    forced_kinds: list[str | None] = []
    if need_both_kinds:
        forced_kinds = ["\\", "/"]
    while len(forced_kinds) < target_mirrors:
        forced_kinds.append(None)
    random.shuffle(forced_kinds)

    for tower_index, direction in ray_specs:
        if len(mirror_pairs) >= target_mirrors:
            break
        forced_kind = forced_kinds[len(mirror_pairs)]
        result = _place_mirror_on_ray(
            h=h,
            w=w,
            tower_pos=tower_positions[tower_index],
            direction=direction,
            tower_positions=tower_pos_set,
            gray_cells=gray_cells,
            forced_kind=forced_kind,
        )
        if result is None:
            continue
        a, b, kind = result
        mirror_pairs.append((a, b, kind))
        gray_cells.add(a)
        gray_cells.add(b)

    if not mirror_pairs:
        return None

    for a, b, _kind in mirror_pairs:
        grid[a] = GRAY
        grid[b] = GRAY

    out, stats = _simulate(grid)
    if bool(np.array_equal(out, grid)):
        return None
    if stats["loops"] != 0:
        return None
    if stats["painted_cells"] < 10:
        return None
    if stats["painted_cells"] * MAX_PAINT_FRAC_DEN > h * w * MAX_PAINT_FRAC_NUM:
        return None
    if len(stats["mirror_hit_cells"]) < len(mirror_pairs):
        return None
    if want_aligned:
        if stats["yellow_cells"] == 0 or stats["tower_stops"] < 2:
            return None
    if need_both_kinds and stats["mirror_hit_kinds"] != {"\\", "/"}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray ARC input")
