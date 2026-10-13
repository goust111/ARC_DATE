from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
ORANGE = 7
CYAN = 8

OCCLUDER_COLORS = (YELLOW, ORANGE)
PATTERN_COLORS = (1, 3, 5, 6, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros_like(g, dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(g == int(color)):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q: deque[tuple[int, int]] = deque([(rr, cc)])
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc2 = q.popleft()
            coords.append((cr, cc2))
            for nr, nc in neighbors4(cr, cc2):
                if in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append({"color": int(color), "coords": coords, "bbox": _bbox_from_coords(coords), "area": len(coords)})
    return comps


def _cell_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (int(r), int(c)),
        (int(c), int(r)),
        (int(n - 1 - c), int(n - 1 - r)),
        (int(n - 1 - r), int(n - 1 - c)),
    }
    return tuple(sorted(pts))


def _square_orbit_toplefts(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (int(r), int(c)),
        (int(c), int(r)),
        (int(n - 2 - c), int(n - 2 - r)),
        (int(n - 2 - r), int(n - 2 - c)),
    }
    return tuple(sorted(pts))


def _square_cells(tl: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = tl
    return [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]


def _touches_target(red_coords: list[tuple[int, int]], target_cells: set[tuple[int, int]]) -> bool:
    for r, c in red_coords:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in target_cells:
                return True
    return False


def _find_occluders(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    out: list[dict] = []
    for color in np.unique(g):
        col = int(color)
        if col in {int(bg), RED, CYAN}:
            continue
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == col)]
        if len(coords) < 4:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(coords) != area:
            continue
        if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == col):
            continue
        out.append({"color": col, "bbox": (r0, c0, r1, c1), "area": area})
    out.sort(key=lambda item: (-item["area"], item["color"]))
    assert len(out) == 2, "expected exactly two solid occluder rectangles"
    return out


def _restore_pattern(grid: np.ndarray, occluders: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n, m = map(int, g.shape)
    assert n == m, "grid must be square"
    occluder_colors = {int(item["color"]) for item in occluders}
    restored = g.copy()
    for item in occluders:
        r0, c0, r1, c1 = item["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                values = {int(g[rr, cc]) for rr, cc in _cell_orbit(n, r, c) if int(g[rr, cc]) not in occluder_colors}
                assert len(values) == 1, "every occluded orbit must have a unique visible value"
                restored[r, c] = values.pop()
    return restored


def _find_anchor_squares(grid: np.ndarray) -> list[dict]:
    anchors = _components_of_color(grid, CYAN)
    assert anchors, "expected at least one cyan anchor"
    for anchor in anchors:
        r0, c0, r1, c1 = anchor["bbox"]
        assert anchor["area"] == 4 and (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 2, "cyan anchors must be 2x2 squares"
    return anchors


def _unique_red_component(grid: np.ndarray) -> dict:
    reds = _components_of_color(grid, RED)
    assert len(reds) == 1, "expected exactly one red object"
    return reds[0]


def _best_slide_to_target(
    grid: np.ndarray, red_coords: list[tuple[int, int]], target_cells: set[tuple[int, int]], bg: int
) -> tuple[int, list[tuple[int, int]]] | None:
    obstacle = np.asarray(grid != int(bg), dtype=bool)
    for r, c in red_coords:
        obstacle[r, c] = False

    best: tuple[int, list[tuple[int, int]]] | None = None
    current0 = [(int(r), int(c)) for r, c in red_coords]
    for dr, dc in DIRS4:
        current = current0
        steps = 0
        while True:
            shifted = [(r + dr, c + dc) for r, c in current]
            if not all(in_bounds(grid.shape, r, c) for r, c in shifted):
                break
            if any((r, c) in target_cells for r, c in shifted):
                break
            if any(bool(obstacle[r, c]) for r, c in shifted):
                break
            steps += 1
            current = shifted
            if _touches_target(current, target_cells):
                candidate = (steps, current)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                break
    return best


def _solve_slide(grid: np.ndarray) -> tuple[dict, list[tuple[int, int]]]:
    bg = get_bg(grid)
    red = _unique_red_component(grid)
    anchors = _find_anchor_squares(grid)
    candidates: list[tuple[dict, int, list[tuple[int, int]]]] = []
    for anchor in anchors:
        target_cells = set(anchor["coords"])
        move = _best_slide_to_target(grid, red["coords"], target_cells, bg)
        if move is not None:
            steps, placed = move
            candidates.append((anchor, steps, placed))
    assert len(candidates) == 1, "there must be exactly one reachable cyan anchor"
    anchor, _steps, placed = candidates[0]
    return anchor, placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct both occluded patches by diagonal reflections, reveal the hidden cyan anchor, and slide the red object onto the only reachable one.

    Input structure:
    - A square grid contains a sparse colored pattern that is symmetric across both diagonals.
    - The pattern includes several cyan 2x2 anchor squares, two solid-color occluder rectangles, and one red free object.
    - The red object and the occluders are not themselves part of the symmetric pattern.

    Rule:
    - Identify the two solid occluder rectangles and refill every covered cell from its unique orbit under main-diagonal and anti-diagonal reflection.
    - In the restored pattern, treat every cyan 2x2 square as a candidate target.
    - Slide the whole red object in one cardinal direction by the shortest amount that makes it 4-neighbor adjacent to the unique candidate whose straight path is unobstructed.

    Key insight:
    - The only valid target is revealed by completing the double-diagonal symmetry first; visible aligned cyan squares are decoys because some foreground cell blocks the slide path.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    occluders = _find_occluders(g, bg)
    restored = _restore_pattern(g, occluders)
    _anchor, placed_red = _solve_slide(restored)

    out = restored.copy()
    for r, c in _unique_red_component(restored)["coords"]:
        out[r, c] = bg
    for r, c in placed_red:
        out[r, c] = RED
    return out


def _dilate(cells: set[tuple[int, int]], n: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    out.add((nr, nc))
    return out


def _all_square_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(1, n - 2):
        for c in range(1, n - 2):
            orbit = _square_orbit_toplefts(n, r, c)
            if len(orbit) != 4:
                continue
            if orbit in seen:
                continue
            seen.add(orbit)
            out.append(orbit)
    random.shuffle(out)
    return out


def _paint_square_orbit(grid: np.ndarray, orbit: tuple[tuple[int, int], ...], color: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tl in orbit:
        for r, c in _square_cells(tl):
            grid[r, c] = int(color)
            cells.add((r, c))
    return cells


def _paint_cell_orbit(grid: np.ndarray, orbit: tuple[tuple[int, int], ...], color: int) -> set[tuple[int, int]]:
    cells = set(orbit)
    for r, c in cells:
        grid[r, c] = int(color)
    return cells


def _orbit_cells_for_square_orbit(orbit: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tl in orbit:
        cells.update(_square_cells(tl))
    return cells


def _valid_square_orbit(
    orbit: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]], n: int
) -> bool:
    cells = _orbit_cells_for_square_orbit(orbit)
    return not (cells & occupied) and not (_dilate(cells, n, 1) & reserved_bg)


def _hidden_configs(
    orbit: tuple[tuple[int, int], ...], n: int
) -> list[tuple[tuple[int, int], str, int, int, tuple[int, int], set[tuple[int, int]]]]:
    configs: list[tuple[tuple[int, int], str, int, int, tuple[int, int], set[tuple[int, int]]]] = []
    for tl in orbit:
        tr, tc = tl
        for lane in (0, 1):
            for gap in (2, 3, 4, 5):
                rr = tr + lane
                left_red = (rr, tc - gap)
                if tc - gap >= 1:
                    reserved = {(rr, cc) for cc in range(tc - gap, tc)}
                    configs.append((tl, "R", lane, gap, left_red, reserved))
                right_red = (rr, tc + 1 + gap)
                if tc + 1 + gap < n - 1:
                    reserved = {(rr, cc) for cc in range(tc + 2, tc + 2 + gap)}
                    configs.append((tl, "L", lane, gap, right_red, reserved | {right_red}))
                cc = tc + lane
                up_red = (tr - gap, cc)
                if tr - gap >= 1:
                    reserved = {(rr2, cc) for rr2 in range(tr - gap, tr)}
                    configs.append((tl, "D", lane, gap, up_red, reserved))
                down_red = (tr + 1 + gap, cc)
                if tr + 1 + gap < n - 1:
                    reserved = {(rr2, cc) for rr2 in range(tr + 2, tr + 2 + gap)}
                    configs.append((tl, "U", lane, gap, down_red, reserved | {down_red}))
    random.shuffle(configs)
    return configs


def _decoy_info(
    orbit: tuple[tuple[int, int], ...], red_pos: tuple[int, int], hidden_dir: str
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    rr, rc = red_pos
    for tl in orbit:
        tr, tc = tl
        if hidden_dir == "R":
            if rr not in {tr, tr + 1} or tc + 1 >= rc - 1:
                continue
            path = [(rr, cc) for cc in range(tc + 2, rc)]
        elif hidden_dir == "L":
            if rr not in {tr, tr + 1} or tc <= rc + 1:
                continue
            path = [(rr, cc) for cc in range(rc + 1, tc)]
        elif hidden_dir == "D":
            if rc not in {tc, tc + 1} or tr + 1 >= rr - 1:
                continue
            path = [(rr2, rc) for rr2 in range(tr + 2, rr)]
        else:
            if rc not in {tc, tc + 1} or tr <= rr + 1:
                continue
            path = [(rr2, rc) for rr2 in range(rr + 1, tr)]
        if path:
            return tl, path
    return None


def _candidate_cell_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            orbit = _cell_orbit(n, r, c)
            if len(orbit) < 2 or orbit in seen:
                continue
            seen.add(orbit)
            out.append(orbit)
    random.shuffle(out)
    return out


def _pick_blocker_orbit(
    n: int,
    path_cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]] | None:
    for cell in path_cells:
        orbit = _cell_orbit(n, cell[0], cell[1])
        orbit_cells = set(orbit)
        if orbit_cells & occupied:
            continue
        if orbit_cells & reserved_bg:
            continue
        return orbit, cell
    return None


def _pick_extra_square_orbits(
    all_orbits: list[tuple[tuple[int, int], ...]],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    n: int,
    count: int,
) -> list[tuple[tuple[int, int], ...]]:
    chosen: list[tuple[tuple[int, int], ...]] = []
    for orbit in all_orbits:
        if len(chosen) >= count:
            break
        if not _valid_square_orbit(orbit, occupied, reserved_bg, n):
            continue
        cells = _orbit_cells_for_square_orbit(orbit)
        if _dilate(cells, n, 1) & occupied:
            continue
        chosen.append(orbit)
        occupied.update(cells)
    return chosen


def _pick_decor_orbits(
    n: int,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    count: int,
) -> list[tuple[tuple[tuple[int, int], ...], int]]:
    chosen: list[tuple[tuple[tuple[int, int], ...], int]] = []
    for orbit in _candidate_cell_orbits(n):
        if len(chosen) >= count:
            break
        cells = set(orbit)
        if cells & occupied or cells & reserved_bg:
            continue
        if _dilate(cells, n, 1) & occupied:
            continue
        color = random.choice(PATTERN_COLORS)
        chosen.append((orbit, color))
        occupied.update(cells)
    return chosen


def _pick_second_occluder(
    completed: np.ndarray,
    hidden_bbox: tuple[int, int, int, int],
    red_pos: tuple[int, int],
    blocker_cells: set[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    n = int(completed.shape[0])
    hidden_keys = {_cell_orbit(n, r, c) for r in range(hidden_bbox[0], hidden_bbox[2] + 1) for c in range(hidden_bbox[1], hidden_bbox[3] + 1)}
    for h, w in random.sample([(2, 2), (2, 3), (3, 2)], 3):
        positions = [(r0, c0) for r0 in range(1, n - h) for c0 in range(1, n - w)]
        random.shuffle(positions)
        for r0, c0 in positions:
            cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
            if red_pos in cells or cells & blocker_cells:
                continue
            if any(int(completed[r, c]) == CYAN for r, c in cells):
                continue
            if not any(int(completed[r, c]) != BG for r, c in cells):
                continue
            keys = [_cell_orbit(n, r, c) for r, c in cells]
            if len(set(keys)) != len(keys):
                continue
            if any(key in hidden_keys for key in keys):
                continue
            if any(red_pos in key for key in keys):
                continue
            return (r0, c0, r0 + h - 1, c0 + w - 1)
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(18, 26)
    completed = np.full((n, n), BG, dtype=int)
    square_orbits = _all_square_orbits(n)
    if len(square_orbits) < 3:
        return None

    for hidden_orbit in square_orbits:
        hidden_cells_all = _orbit_cells_for_square_orbit(hidden_orbit)
        for hidden_tl, hidden_dir, _lane, _gap, red_pos, reserved in _hidden_configs(hidden_orbit, n):
            if not in_bounds((n, n), red_pos[0], red_pos[1]):
                continue
            reserved_bg = set(reserved)
            reserved_bg.add(red_pos)
            if hidden_cells_all & reserved_bg:
                continue
            occupied = set(hidden_cells_all)

            decoy_orbit = None
            decoy_path: list[tuple[int, int]] | None = None
            for orbit in square_orbits:
                if orbit == hidden_orbit:
                    continue
                if not _valid_square_orbit(orbit, occupied, reserved_bg, n):
                    continue
                info = _decoy_info(orbit, red_pos, hidden_dir)
                if info is None:
                    continue
                decoy_tl, path = info
                cells = _orbit_cells_for_square_orbit(orbit)
                if cells & reserved_bg:
                    continue
                if _dilate(cells, n, 1) & hidden_cells_all:
                    continue
                if not path:
                    continue
                decoy_orbit = orbit
                decoy_path = path
                occupied.update(cells)
                break
            if decoy_orbit is None or decoy_path is None:
                continue

            blocker_pick = _pick_blocker_orbit(n, decoy_path, occupied, reserved_bg)
            if blocker_pick is None:
                continue
            blocker_orbit, blocker_cell = blocker_pick
            blocker_cells = set(blocker_orbit)
            occupied.update(blocker_cells)

            extra_orbits = _pick_extra_square_orbits(square_orbits, occupied, reserved_bg, n, random.randint(0, 1))
            decor_orbits = _pick_decor_orbits(n, occupied, reserved_bg, random.randint(4, 8))

            completed.fill(BG)
            _paint_square_orbit(completed, hidden_orbit, CYAN)
            _paint_square_orbit(completed, decoy_orbit, CYAN)
            for orbit in extra_orbits:
                _paint_square_orbit(completed, orbit, CYAN)
            _paint_cell_orbit(completed, blocker_orbit, random.choice((1, 3, 5)))
            for orbit, color in decor_orbits:
                _paint_cell_orbit(completed, orbit, color)

            hidden_bbox = (hidden_tl[0], hidden_tl[1], hidden_tl[0] + 1, hidden_tl[1] + 1)
            second_bbox = _pick_second_occluder(completed, hidden_bbox, red_pos, blocker_cells)
            if second_bbox is None:
                continue

            inp = completed.copy()
            r0, c0, r1, c1 = hidden_bbox
            inp[r0 : r1 + 1, c0 : c1 + 1] = YELLOW
            r0, c0, r1, c1 = second_bbox
            inp[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
            if int(inp[red_pos]) != BG:
                continue
            inp[red_pos] = RED

            try:
                out = main(inp)
            except Exception:
                continue

            restored = _restore_pattern(inp, _find_occluders(inp, BG))
            anchor, placed_red = _solve_slide(restored)
            if anchor["bbox"] != hidden_bbox:
                continue

            expected = completed.copy()
            expected[red_pos] = BG
            for r, c in placed_red:
                expected[r, c] = RED
            if np.array_equal(out, expected):
                return inp
        if random.random() < 0.7:
            break
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
