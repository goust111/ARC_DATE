import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_RETRIES = 400
SCENE_RETRIES = 120
EXTRA_SCREEN_RETRIES = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _parse_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 8 and w >= 8, "grid too small"

    screens = []
    screen_by_cell: dict[tuple[int, int], int] = {}
    for coords in _components(g == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        assert hh == ww and hh in (2, 3), "screen must be 2x2 or 3x3"
        assert len(coords) == hh * ww, "screen must be a solid rectangle"
        idx = len(screens)
        for cell in coords:
            screen_by_cell[cell] = idx
        screens.append({"bbox": (r0, c0, r1, c1), "size": hh})

    mirrors = []
    for coords in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        assert hh == 1 or ww == 1, "mirror must be a straight segment"
        assert len(coords) == max(hh, ww), "mirror must be contiguous"
        length = max(hh, ww)
        assert 3 <= length <= 6, "mirror length out of range"
        orient = "v" if ww == 1 else "h"
        mirrors.append({"bbox": (r0, c0, r1, c1), "orientation": orient, "length": length})

    tower_mask = ~np.isin(g, [BLACK, GREY, MAROON])
    tower_comps = _components(tower_mask)
    towers = []
    tower_colors = set()
    for coords in tower_comps:
        assert len(coords) == 1, "each tower must be a single pixel"
        r, c = coords[0]
        color = int(g[r, c])
        assert color not in tower_colors, "tower colors must be unique"
        tower_colors.add(color)
        towers.append({"pos": (r, c), "color": color})

    assert towers, "expected at least one signal tower"
    assert mirrors, "expected at least one mirror"
    assert screens, "expected at least one screen"
    return towers, mirrors, screens, screen_by_cell


def _clear_segment(g: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return bool(np.all(g[r0, lo + 1 : hi] == BLACK))
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return bool(np.all(g[lo + 1 : hi, c0] == BLACK))
    raise AssertionError("segment must be axis-aligned")


def _candidate_mirrors(g: np.ndarray, tower: dict, mirrors: list[dict]):
    r, c = tower["pos"]
    candidates = []
    for mirror in mirrors:
        r0, c0, r1, c1 = mirror["bbox"]
        if mirror["orientation"] == "v" and r0 <= r <= r1 and c != c0:
            hit = (r, c0)
            if _clear_segment(g, (r, c), hit):
                candidates.append(
                    {"distance": abs(c - c0), "incoming": "h", "hit": hit, "mirror": mirror}
                )
        elif mirror["orientation"] == "h" and c0 <= c <= c1 and r != r0:
            hit = (r0, c)
            if _clear_segment(g, (r, c), hit):
                candidates.append(
                    {"distance": abs(r - r0), "incoming": "v", "hit": hit, "mirror": mirror}
                )
    candidates.sort(key=lambda item: (item["distance"], item["hit"][0], item["hit"][1]))
    return candidates


def _exit_direction(mirror: dict, hit: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = mirror["bbox"]
    hr, hc = hit
    if mirror["orientation"] == "v":
        up_len = hr - r0
        down_len = r1 - hr
        assert up_len != down_len, "ambiguous vertical reflection"
        return (-1, 0) if up_len > down_len else (1, 0)
    left_len = hc - c0
    right_len = c1 - hc
    assert left_len != right_len, "ambiguous horizontal reflection"
    return (0, -1) if left_len > right_len else (0, 1)


def _trace_tower(g: np.ndarray, tower: dict, mirrors: list[dict], screen_by_cell: dict):
    candidates = _candidate_mirrors(g, tower, mirrors)
    assert len(candidates) == 1, "each tower must align with exactly one reachable mirror"
    hit = candidates[0]["hit"]
    dr, dc = _exit_direction(candidates[0]["mirror"], hit)
    h, w = g.shape
    r, c = hit
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None, candidates[0]["incoming"]
        screen_idx = screen_by_cell.get((r, c))
        if screen_idx is not None:
            return screen_idx, candidates[0]["incoming"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored tower ray to its aligned grey mirror, reflect it by 90 degrees,
    and recolor the first maroon screen hit by the reflected ray.
    """
    g = np.asarray(grid, dtype=int)
    towers, mirrors, screens, screen_by_cell = _parse_grid(g)

    out = g.copy()
    for tower in sorted(towers, key=lambda item: item["pos"]):
        screen_idx, _incoming = _trace_tower(g, tower, mirrors, screen_by_cell)
        if screen_idx is None:
            continue
        r0, c0, r1, c1 = screens[screen_idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = tower["color"]
    return out


def _line_cells_exclusive(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo + 1, hi)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo + 1, hi)]
    raise AssertionError("line must be axis-aligned")


def _first_hit_ray_cells(
    hit: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    stop_bbox: tuple[int, int, int, int] | None,
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = hit
    cells = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return cells
        cells.append((r, c))
        if stop_bbox is not None:
            r0, c0, r1, c1 = stop_bbox
            if r0 <= r <= r1 and c0 <= c <= c1:
                return cells


def _object_cells(scene: dict) -> set[tuple[int, int]]:
    cells = {scene["tower"]["pos"]}
    cells.update(scene["mirror_cells"])
    if scene["screen_bbox"] is not None:
        cells.update(_rect_cells(scene["screen_bbox"]))
    return cells


def _reserved_cells(scene: dict, h: int, w: int) -> set[tuple[int, int]]:
    reserved = set(_object_cells(scene))
    reserved.update(_line_cells_exclusive(scene["tower"]["pos"], scene["hit"]))
    reserved.update(_first_hit_ray_cells(scene["hit"], scene["exit_dir"], h, w, scene["screen_bbox"]))
    return reserved


def _touches_occupied(cells: set[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return True
        for nr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for nc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                if occupied[nr, nc]:
                    return True
    return False


def _scene_fits(scene: dict, occupied: np.ndarray, forbidden: np.ndarray) -> bool:
    h, w = occupied.shape
    obj_cells = _object_cells(scene)
    if _touches_occupied(obj_cells, occupied):
        return False
    for r, c in obj_cells:
        if forbidden[r, c]:
            return False
    for r, c in _reserved_cells(scene, h, w):
        if occupied[r, c]:
            return False
    return True


def _place_scene(grid: np.ndarray, scene: dict, occupied: np.ndarray, forbidden: np.ndarray) -> None:
    for r, c in scene["mirror_cells"]:
        grid[r, c] = GREY
    tr, tc = scene["tower"]["pos"]
    grid[tr, tc] = scene["tower"]["color"]
    if scene["screen_bbox"] is not None:
        r0, c0, r1, c1 = scene["screen_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = MAROON

    for r, c in _object_cells(scene):
        occupied[r, c] = True
        forbidden[r, c] = True
    for r, c in _reserved_cells(scene, grid.shape[0], grid.shape[1]):
        forbidden[r, c] = True


def _sample_scene(
    rng: random.Random,
    h: int,
    w: int,
    direction: str,
    color: int,
    will_hit: bool,
) -> dict | None:
    for _ in range(SCENE_RETRIES):
        mirror_len = rng.randint(3, 6)
        tower_gap = rng.randint(2, 6)
        screen_size = rng.choice((2, 3))
        screen_gap = rng.randint(1, 4)

        if direction == "up":
            tower_side = rng.choice(("left", "right"))
            c_lo = tower_gap if tower_side == "left" else 0
            c_hi = (w - 1) if tower_side == "left" else (w - 1 - tower_gap)
            r_lo = mirror_len - 1
            if will_hit:
                r_lo = max(r_lo, mirror_len + screen_gap + screen_size - 1)
            r_hi = h - 1
            if c_lo > c_hi or r_lo > r_hi:
                continue
            hit_r = rng.randint(r_lo, r_hi)
            hit_c = rng.randint(c_lo, c_hi)
            tower_pos = (hit_r, hit_c - tower_gap) if tower_side == "left" else (hit_r, hit_c + tower_gap)
            mirror_rows = list(range(hit_r - mirror_len + 1, hit_r + 1))
            mirror_cells = {(r, hit_c) for r in mirror_rows}
            screen_bbox = None
            if will_hit:
                sc_lo = max(0, hit_c - screen_size + 1)
                sc_hi = min(hit_c, w - screen_size)
                if sc_lo > sc_hi:
                    continue
                sc0 = rng.randint(sc_lo, sc_hi)
                sr0 = hit_r - mirror_len + 1 - screen_gap - screen_size
                screen_bbox = (sr0, sc0, sr0 + screen_size - 1, sc0 + screen_size - 1)
            return {
                "tower": {"pos": tower_pos, "color": color},
                "hit": (hit_r, hit_c),
                "mirror_cells": mirror_cells,
                "screen_bbox": screen_bbox,
                "exit_dir": (-1, 0),
            }

        if direction == "down":
            tower_side = rng.choice(("left", "right"))
            c_lo = tower_gap if tower_side == "left" else 0
            c_hi = (w - 1) if tower_side == "left" else (w - 1 - tower_gap)
            r_lo = 0
            r_hi = h - mirror_len
            if will_hit:
                r_hi = min(r_hi, h - mirror_len - screen_gap - screen_size)
            if c_lo > c_hi or r_lo > r_hi:
                continue
            hit_r = rng.randint(r_lo, r_hi)
            hit_c = rng.randint(c_lo, c_hi)
            tower_pos = (hit_r, hit_c - tower_gap) if tower_side == "left" else (hit_r, hit_c + tower_gap)
            mirror_rows = list(range(hit_r, hit_r + mirror_len))
            mirror_cells = {(r, hit_c) for r in mirror_rows}
            screen_bbox = None
            if will_hit:
                sc_lo = max(0, hit_c - screen_size + 1)
                sc_hi = min(hit_c, w - screen_size)
                if sc_lo > sc_hi:
                    continue
                sc0 = rng.randint(sc_lo, sc_hi)
                sr0 = hit_r + mirror_len + screen_gap
                screen_bbox = (sr0, sc0, sr0 + screen_size - 1, sc0 + screen_size - 1)
            return {
                "tower": {"pos": tower_pos, "color": color},
                "hit": (hit_r, hit_c),
                "mirror_cells": mirror_cells,
                "screen_bbox": screen_bbox,
                "exit_dir": (1, 0),
            }

        if direction == "left":
            tower_side = rng.choice(("up", "down"))
            r_lo = tower_gap if tower_side == "up" else 0
            r_hi = (h - 1) if tower_side == "up" else (h - 1 - tower_gap)
            c_lo = mirror_len - 1
            if will_hit:
                c_lo = max(c_lo, mirror_len + screen_gap + screen_size - 1)
            c_hi = w - 1
            if r_lo > r_hi or c_lo > c_hi:
                continue
            hit_r = rng.randint(r_lo, r_hi)
            hit_c = rng.randint(c_lo, c_hi)
            tower_pos = (hit_r - tower_gap, hit_c) if tower_side == "up" else (hit_r + tower_gap, hit_c)
            mirror_cols = list(range(hit_c - mirror_len + 1, hit_c + 1))
            mirror_cells = {(hit_r, c) for c in mirror_cols}
            screen_bbox = None
            if will_hit:
                sr_lo = max(0, hit_r - screen_size + 1)
                sr_hi = min(hit_r, h - screen_size)
                if sr_lo > sr_hi:
                    continue
                sr0 = rng.randint(sr_lo, sr_hi)
                sc0 = hit_c - mirror_len + 1 - screen_gap - screen_size
                screen_bbox = (sr0, sc0, sr0 + screen_size - 1, sc0 + screen_size - 1)
            return {
                "tower": {"pos": tower_pos, "color": color},
                "hit": (hit_r, hit_c),
                "mirror_cells": mirror_cells,
                "screen_bbox": screen_bbox,
                "exit_dir": (0, -1),
            }

        if direction == "right":
            tower_side = rng.choice(("up", "down"))
            r_lo = tower_gap if tower_side == "up" else 0
            r_hi = (h - 1) if tower_side == "up" else (h - 1 - tower_gap)
            c_lo = 0
            c_hi = w - mirror_len
            if will_hit:
                c_hi = min(c_hi, w - mirror_len - screen_gap - screen_size)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            hit_r = rng.randint(r_lo, r_hi)
            hit_c = rng.randint(c_lo, c_hi)
            tower_pos = (hit_r - tower_gap, hit_c) if tower_side == "up" else (hit_r + tower_gap, hit_c)
            mirror_cols = list(range(hit_c, hit_c + mirror_len))
            mirror_cells = {(hit_r, c) for c in mirror_cols}
            screen_bbox = None
            if will_hit:
                sr_lo = max(0, hit_r - screen_size + 1)
                sr_hi = min(hit_r, h - screen_size)
                if sr_lo > sr_hi:
                    continue
                sr0 = rng.randint(sr_lo, sr_hi)
                sc0 = hit_c + mirror_len + screen_gap
                screen_bbox = (sr0, sc0, sr0 + screen_size - 1, sc0 + screen_size - 1)
            return {
                "tower": {"pos": tower_pos, "color": color},
                "hit": (hit_r, hit_c),
                "mirror_cells": mirror_cells,
                "screen_bbox": screen_bbox,
                "exit_dir": (0, 1),
            }

    return None


def _place_extra_screen(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    forbidden: np.ndarray,
) -> bool:
    h, w = grid.shape
    for _ in range(EXTRA_SCREEN_RETRIES):
        size = rng.choice((2, 3))
        r0 = rng.randint(0, h - size)
        c0 = rng.randint(0, w - size)
        bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
        cells = set(_rect_cells(bbox))
        if _touches_occupied(cells, occupied):
            continue
        if any(forbidden[r, c] for r, c in cells):
            continue
        grid[r0 : r0 + size, c0 : c0 + size] = MAROON
        for r, c in cells:
            occupied[r, c] = True
            forbidden[r, c] = True
        return True
    return False


def _choose_directions(rng: random.Random, count: int) -> list[str]:
    horizontal_count = rng.randint(max(1, count - 2), min(2, count - 1))
    horizontal_in = rng.sample(["up", "down"], k=horizontal_count)
    vertical_count = count - len(horizontal_in)
    vertical_in = rng.sample(["left", "right"], k=vertical_count)
    directions = horizontal_in + vertical_in
    rng.shuffle(directions)
    return directions


def _choose_hit_flags(rng: random.Random, count: int) -> list[bool]:
    hit_count = rng.randint(1, count - 1)
    flags = [True] * hit_count + [False] * (count - hit_count)
    rng.shuffle(flags)
    return flags


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        towers, mirrors, screens, screen_by_cell = _parse_grid(grid)
    except AssertionError:
        return False

    incoming_seen = set()
    hit_screens = []
    for tower in towers:
        candidates = _candidate_mirrors(grid, tower, mirrors)
        if len(candidates) != 1:
            return False
        try:
            screen_idx, incoming = _trace_tower(grid, tower, mirrors, screen_by_cell)
        except AssertionError:
            return False
        incoming_seen.add(incoming)
        if screen_idx is not None:
            hit_screens.append(screen_idx)

    if incoming_seen != {"h", "v"}:
        return False
    if not hit_screens or len(hit_screens) == len(towers):
        return False
    if len(set(hit_screens)) != len(hit_screens):
        return False
    if len(set(hit_screens)) >= len(screens):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    changed = np.argwhere(out != grid)
    if changed.size == 0:
        return False
    for r, c in changed:
        if int(grid[r, c]) != MAROON:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 24)
    w = rng.randint(16, 24)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    forbidden = np.zeros((h, w), dtype=bool)

    tower_count = rng.randint(2, 4)
    colors = rng.sample(list(TOWER_COLORS), k=tower_count)
    directions = _choose_directions(rng, tower_count)
    hit_flags = _choose_hit_flags(rng, tower_count)

    for color, direction, will_hit in zip(colors, directions, hit_flags):
        scene = _sample_scene(rng, h, w, direction, color, will_hit)
        if scene is None or not _scene_fits(scene, occupied, forbidden):
            for _ in range(SCENE_RETRIES):
                scene = _sample_scene(rng, h, w, direction, color, will_hit)
                if scene is not None and _scene_fits(scene, occupied, forbidden):
                    break
            else:
                return None
        _place_scene(grid, scene, occupied, forbidden)

    extra_screens = rng.randint(1, 3)
    for _ in range(extra_screens):
        if not _place_extra_screen(rng, grid, occupied, forbidden):
            return None

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
