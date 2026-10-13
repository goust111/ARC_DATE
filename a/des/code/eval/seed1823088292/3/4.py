from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
MAROON = 9
PORTAL_COLOR_POOL = [1, 2, 3, 5, 6, 7, 8]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
PERPENDICULARS = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}

MIN_PORTAL_PAIRS = 2
MAX_PORTAL_PAIRS = 4
MAX_USED_PORTALS = 3
MIN_HEIGHT = 10
MAX_HEIGHT = 18
MIN_FINAL_GAP = 3
MAX_FINAL_GAP = 5
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _monochrome_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(g == int(color)):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            comp.append((int(r), int(c)))
            for rr, cc in _neighbors4(int(r), int(c)):
                if _in_bounds((h, w), int(rr), int(cc)) and not seen[rr, cc] and int(g[rr, cc]) == int(color):
                    seen[rr, cc] = True
                    queue.append((int(rr), int(cc)))
        comps.append(comp)
    return comps


def _portal_dict(color: int, idx: int, comp: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert len(comp) == 4, "portal must have area 4"
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "portal must be a solid 2x2 block"
    cells = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
    assert set(comp) == cells, "portal component must be solid"
    return {"color": int(color), "idx": int(idx), "r0": int(r0), "c0": int(c0)}


def _extract_portals(grid: np.ndarray) -> tuple[dict[int, list[dict]], dict[tuple[int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) not in (BLACK, YELLOW, MAROON)]
    portals_by_color: dict[int, list[dict]] = {}
    cell_to_portal: dict[tuple[int, int], dict] = {}
    for color in colors:
        comps = _monochrome_components(g, color)
        assert len(comps) == 2, "each portal color must form exactly two components"
        entries = [_portal_dict(color, idx, comp) for idx, comp in enumerate(comps)]
        portals_by_color[int(color)] = entries
        for portal in entries:
            for rr in range(int(portal["r0"]), int(portal["r0"]) + 2):
                for cc in range(int(portal["c0"]), int(portal["c0"]) + 2):
                    cell_to_portal[(int(rr), int(cc))] = portal
    assert MIN_PORTAL_PAIRS <= len(portals_by_color) <= MAX_PORTAL_PAIRS
    return portals_by_color, cell_to_portal


def _find_launcher(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    pts = np.argwhere(g == MAROON)
    assert pts.shape[0] == 1, "expected exactly one maroon launcher"
    r, c = map(int, pts[0])
    h, w = map(int, g.shape)
    assert r in (0, h - 1) or c in (0, w - 1), "launcher must lie on the border"
    return int(r), int(c)


def _arrow_templates(launcher: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    lr, lc = launcher
    dr, dc = direction
    pr, pc = shape
    return {
        (lr + dr, lc + dc),
        (lr + 2 * dr, lc + 2 * dc),
        (lr + 2 * dr + pr, lc + 2 * dc + pc),
    }


def _find_direction(grid: np.ndarray, launcher: tuple[int, int]) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    yellow = {tuple(map(int, pt)) for pt in np.argwhere(g == YELLOW)}
    assert len(yellow) == 3, "expected a single 3-cell yellow arrow"

    for direction in DIRS:
        dr, dc = direction
        straight = {(launcher[0] + k * dr, launcher[1] + k * dc) for k in (1, 2, 3)}
        if yellow == straight:
            return direction
        for perp in PERPENDICULARS[direction]:
            template = _arrow_templates(launcher, direction, perp)
            if yellow == template:
                return direction
    raise AssertionError("yellow arrow does not match any supported direction template")


def _portal_exit(
    portal: dict,
    partner: dict,
    hit: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    hit_r, hit_c = map(int, hit)
    r0, c0 = int(portal["r0"]), int(portal["c0"])
    pr0, pc0 = int(partner["r0"]), int(partner["c0"])

    if direction == RIGHT:
        assert hit_c == c0
        offset = hit_r - r0
        assert offset in (0, 1)
        return int(pr0 + offset), int(pc0 + 1)
    if direction == LEFT:
        assert hit_c == c0 + 1
        offset = hit_r - r0
        assert offset in (0, 1)
        return int(pr0 + offset), int(pc0)
    if direction == DOWN:
        assert hit_r == r0
        offset = hit_c - c0
        assert offset in (0, 1)
        return int(pr0 + 1), int(pc0 + offset)

    assert direction == UP
    assert hit_r == r0 + 1
    offset = hit_c - c0
    assert offset in (0, 1)
    return int(pr0), int(pc0 + offset)


def _simulate(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    portals_by_color, cell_to_portal = _extract_portals(g)
    launcher = _find_launcher(g)
    direction = _find_direction(g, launcher)
    dr, dc = map(int, direction)
    h, w = map(int, g.shape)

    out = g.copy()
    r, c = launcher
    ray_color = MAROON
    seen: set[tuple[int, int]] = set()

    while True:
        state = (int(r), int(c))
        assert state not in seen, "ray path must not loop forever"
        seen.add(state)

        nr, nc = int(r + dr), int(c + dc)
        if not _in_bounds((h, w), nr, nc):
            break

        portal = cell_to_portal.get((int(nr), int(nc)))
        if portal is not None:
            partner = portals_by_color[int(portal["color"])][1 - int(portal["idx"])]
            r, c = _portal_exit(portal, partner, (int(nr), int(nc)), direction)
            ray_color = int(partner["color"])
            continue

        if int(g[nr, nc]) == BLACK:
            out[nr, nc] = int(ray_color)
        r, c = int(nr), int(nc)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify same-color 2x2 portal pairs and trace a ray that teleports between them.

    The launcher is a single maroon border pixel. The yellow 3-cell marker next to it
    encodes one of four cardinal directions using either a straight 3-pixel line or an
    L shape. The ray keeps its direction through every teleport, exits from the paired
    portal's opposite side, and changes color to the destination portal color.
    """
    return _simulate(np.asarray(grid, dtype=int))


def _portal_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _can_place_portal(occupied: set[tuple[int, int]], h: int, w: int, r0: int, c0: int) -> bool:
    if not (1 <= r0 <= h - 3 and 1 <= c0 <= w - 3):
        return False
    for rr in range(int(r0) - 1, int(r0) + 3):
        for cc in range(int(c0) - 1, int(c0) + 3):
            if (int(rr), int(cc)) in occupied:
                return False
    return True


def _choose_rows(height: int, used_count: int) -> list[int]:
    rows = [random.randint(2, height - 3)]
    for _ in range(used_count):
        pool = list(range(2, height - 2))
        if len(pool) > 1 and random.random() < 0.75:
            pool = [r for r in pool if r != rows[-1]]
        rows.append(random.choice(pool))
    return rows


def _build_base_grid() -> np.ndarray | None:
    pair_count = random.randint(MIN_PORTAL_PAIRS, MAX_PORTAL_PAIRS)
    used_count = random.randint(1, min(pair_count, MAX_USED_PORTALS))
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    rows = _choose_rows(height, used_count)
    offsets = [random.randint(0, 1) for _ in range(used_count)]
    colors = random.sample(PORTAL_COLOR_POOL, pair_count)

    arrow_is_straight = random.random() < 0.5
    bend = random.choice((-1, 1))
    arrow_cells = [(rows[0], 1), (rows[0], 2)]
    if arrow_is_straight:
        arrow_cells.append((rows[0], 3))
        cursor = 3
    else:
        arrow_cells.append((rows[0] + bend, 2))
        cursor = 2

    used_portals: list[tuple[int, int, int, int, int]] = []
    for i in range(used_count):
        gap_before = random.randint(3, 4) if i == 0 else random.randint(2, 3)
        gap_between = random.randint(2, 4)
        a_c0 = cursor + gap_before
        b_c0 = a_c0 + 2 + gap_between
        used_portals.append((rows[i], rows[i + 1], offsets[i], a_c0, b_c0))
        cursor = b_c0 + 1

    final_gap = random.randint(MIN_FINAL_GAP, MAX_FINAL_GAP)
    width = cursor + final_gap + 1
    if width > 36:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    launcher = (rows[0], 0)
    grid[launcher] = MAROON
    _paint_cells(grid, arrow_cells, YELLOW)

    occupied = {launcher, *arrow_cells}
    active_rows = set(rows)
    for color, (entry_row, exit_row, offset, a_c0, b_c0) in zip(colors[:used_count], used_portals):
        a_r0 = entry_row - offset
        b_r0 = exit_row - offset
        if not _can_place_portal(occupied, height, width, a_r0, a_c0):
            return None
        a_cells = _portal_cells(a_r0, a_c0)
        _paint_cells(grid, a_cells, color)
        occupied.update(a_cells)

        if not _can_place_portal(occupied, height, width, b_r0, b_c0):
            return None
        b_cells = _portal_cells(b_r0, b_c0)
        _paint_cells(grid, b_cells, color)
        occupied.update(b_cells)

    invalid_tops = {r for r in active_rows if 1 <= r <= height - 3}
    invalid_tops |= {r - 1 for r in active_rows if 1 <= r - 1 <= height - 3}
    candidate_tops = [r for r in range(1, height - 2) if r not in invalid_tops]
    if len(candidate_tops) < 2 and pair_count > used_count:
        return None

    for color in colors[used_count:]:
        pair_cells: list[tuple[int, int]] = []
        for _ in range(2):
            candidates: list[tuple[int, int]] = []
            for r0 in candidate_tops:
                for c0 in range(4, width - 2):
                    if _can_place_portal(occupied, height, width, r0, c0):
                        candidates.append((int(r0), int(c0)))
            if not candidates:
                return None
            r0, c0 = random.choice(candidates)
            cells = _portal_cells(r0, c0)
            _paint_cells(grid, cells, color)
            occupied.update(cells)
            pair_cells.extend(cells)
        assert pair_cells

    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        out = _simulate(grid)
    except AssertionError:
        return False

    changed = out != grid
    if int(changed.sum()) < 6:
        return False

    changed_colors = {int(v) for v in out[changed]}
    if MAROON not in changed_colors:
        return False
    if len(changed_colors - {MAROON}) < 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        base = _build_base_grid()
        if base is None:
            continue
        rotated = np.rot90(base, random.randint(0, 3)).copy()
        if _valid_generated_grid(rotated):
            return rotated
    raise RuntimeError(f"failed to generate a valid portal-ray input after {MAX_GENERATION_ATTEMPTS} attempts")
