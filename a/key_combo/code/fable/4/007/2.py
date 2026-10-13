from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GRAY = 5
YELLOW = 4
MAROON = 9

AXIS_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ALL_DIRS = DIAG_DIRS + AXIS_DIRS


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    remaining = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            r, c = stack.pop()
            for nb in _neighbors8(r, c):
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
                    comp.append(nb)
        comps.append(comp)
    return comps


def _triangle_info(cells: list[tuple[int, int]]) -> dict | None:
    if len(cells) != 3:
        return None
    s = set(cells)
    deg4 = {p: sum(1 for n in _neighbors4(*p) if n in s) for p in cells}
    deg8 = {p: sum(1 for n in _neighbors8(*p) if n in s) for p in cells}

    corners = [p for p, d in deg4.items() if d == 2]
    if len(corners) == 1:
        apex = corners[0]
        ar, ac = apex
        arms = []
        for r, c in cells:
            if (r, c) != apex:
                dr, dc = r - ar, c - ac
                if abs(dr) + abs(dc) == 1:
                    arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr = -(arms[0][0] + arms[1][0])
        dc = -(arms[0][1] + arms[1][1])
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        return {"apex": apex, "dir": (dr, dc), "kind": "L"}

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in cells if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    if abs(d1[0]) != 1 or abs(d1[1]) != 1 or abs(d2[0]) != 1 or abs(d2[1]) != 1:
        return None
    dr = -((d1[0] + d2[0]) // 2)
    dc = -((d1[1] + d2[1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return {"apex": apex, "dir": (dr, dc), "kind": "V"}


def _parse(grid: np.ndarray) -> tuple[int, list[dict], int]:
    bg = _background(grid)
    marker = np.argwhere(grid == GRAY)
    assert marker.shape == (1, 2), "expected exactly one grey marker pixel"
    marker_rc = tuple(map(int, marker[0]))

    sprites: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, GRAY}):
        for comp in _components8(grid == color):
            info = _triangle_info(comp)
            assert info is not None, "every non-marker component must be a 3-pixel triangle"
            sid = len(sprites)
            sprites.append(
                {
                    "id": sid,
                    "color": int(color),
                    "cells": tuple(sorted(comp)),
                    "cell_set": frozenset(comp),
                    "apex": info["apex"],
                    "dir": info["dir"],
                    "kind": info["kind"],
                }
            )

    assert sprites, "expected at least one triangle sprite"
    touching = []
    mr, mc = marker_rc
    for sprite in sprites:
        if any(abs(r - mr) + abs(c - mc) == 1 for r, c in sprite["cells"]):
            touching.append(int(sprite["id"]))
    assert len(touching) == 1, "grey marker must touch exactly one sprite by edge"
    return bg, sprites, touching[0]


def _simulate_fire(
    grid: np.ndarray,
    bg: int,
    sprites: list[dict],
    start_id: int,
) -> dict:
    cell_to_sprite = {}
    for sprite in sprites:
        for cell in sprite["cells"]:
            assert cell not in cell_to_sprite, "sprites must not overlap"
            cell_to_sprite[cell] = int(sprite["id"])

    fired = {int(start_id)}
    queue = deque([int(start_id)])
    order: list[int] = []
    trail: set[tuple[int, int]] = set()
    hit_new = False
    hit_old = False
    crossed_non_bg: set[tuple[int, int]] = set()

    while queue:
        sid = queue.popleft()
        order.append(sid)
        sprite = sprites[sid]
        r, c = sprite["apex"]
        dr, dc = sprite["dir"]
        r += dr
        c += dc
        while _in_bounds(grid.shape, r, c):
            cell = (r, c)
            if cell in cell_to_sprite:
                target = cell_to_sprite[cell]
                if target not in fired:
                    fired.add(target)
                    queue.append(target)
                    hit_new = True
                else:
                    hit_old = True
                break
            if int(grid[r, c]) == bg:
                trail.add(cell)
            else:
                crossed_non_bg.add(cell)
            r += dr
            c += dc

    return {
        "fired": fired,
        "order": order,
        "trail": trail,
        "hit_new": hit_new,
        "hit_old": hit_old,
        "crossed_non_bg": crossed_non_bg,
    }


def _render(grid: np.ndarray, sprites: list[dict], sim: dict) -> np.ndarray:
    out = grid.copy()
    for r, c in sim["trail"]:
        out[r, c] = YELLOW
    for sid in sim["fired"]:
        for r, c in sprites[sid]["cells"]:
            out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite the grey-marked triangle and propagate straight projectile fire through hit sprites.

    Input structure:
    - The background is the most common color, with several separated 3-pixel triangle sprites shaped as axis-aligned L's or diagonal V's.
    - Exactly one grey pixel is edge-adjacent to one sprite and edge-adjacent to no other sprite.

    Rule:
    - The marked sprite fires first: each L fires diagonally away from its missing 2x2 corner, and each V fires axially away from its two diagonal legs.
    - A projectile starts one cell outside the sprite apex, paints only background cells yellow, and stops when it hits any sprite pixel or leaves the grid.
    - A newly hit sprite immediately fires once in its own encoded direction; a sprite that has already fired only blocks later projectiles.
    - Every sprite that fires is recolored maroon, while untouched sprites and the grey marker keep their input colors.

    Key insight:
    - The grey marker only selects the first shooter; the rest of the maroon sprites and yellow paths are determined by orientation-coded ray hits.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    bg, sprites, start_id = _parse(g)
    sim = _simulate_fire(g, bg, sprites, start_id)
    return _render(g, sprites, sim)


# ---------------- generator ----------------

GEN_ATTEMPTS = 500
MIN_SIZE = 13
MAX_SIZE = 27
SPRITE_COUNT_RANGE = (4, 9)
FIRED_COUNT_RANGE = (3, 6)
BACKGROUND_COLORS = (0, 1, 2, 3, 6, 7, 8)
SPRITE_PALETTE = tuple(c for c in range(10) if c not in {GRAY, YELLOW, MAROON})


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _halo(cells: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(h, r + 2))
        for cc in range(max(0, c - 1), min(w, c + 2))
    )


def _placement(apex: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> dict | None:
    cells = _sprite_cells(apex, direction)
    if not all(_in_bounds(shape, r, c) for r, c in cells):
        return None
    return {
        "apex": apex,
        "dir": direction,
        "kind": "L" if abs(direction[0]) + abs(direction[1]) == 2 else "V",
        "cells": cells,
        "cell_set": frozenset(cells),
        "halo": _halo(cells, shape),
    }


def _all_placements(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    out = []
    for r in range(h):
        for c in range(w):
            for direction in ALL_DIRS:
                p = _placement((r, c), direction, shape)
                if p is not None:
                    out.append(p)
    return out


def _ray_cells(shape: tuple[int, int], placement: dict) -> list[tuple[int, int]]:
    r, c = placement["apex"]
    dr, dc = placement["dir"]
    cells = []
    r += dr
    c += dc
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _ray_hit_distance(shape: tuple[int, int], shooter: dict, target: dict) -> int | None:
    target_cells = target["cell_set"]
    for dist, cell in enumerate(_ray_cells(shape, shooter), start=1):
        if cell in target_cells:
            return dist
    return None


def _path_before_hit(shape: tuple[int, int], shooter: dict, hit_dist: int) -> set[tuple[int, int]]:
    ray = _ray_cells(shape, shooter)
    return set(ray[: max(0, hit_dist - 1)])


def _fits(candidate: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (candidate["halo"] & occupied) and not (candidate["cell_set"] & reserved_bg)


def _clear_before_hit(
    shape: tuple[int, int],
    shooter: dict,
    target: dict,
    occupied: set[tuple[int, int]],
    hit_dist: int,
) -> bool:
    target_cells = target["cell_set"]
    for dist, cell in enumerate(_ray_cells(shape, shooter), start=1):
        if dist >= hit_dist:
            return cell in target_cells
        if cell in occupied:
            return False
    return False


def _choose_next_hit(
    rng: random.Random,
    shape: tuple[int, int],
    placements: list[dict],
    shooter: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[dict, int] | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    for cand in candidates:
        if not _fits(cand, occupied, reserved_bg):
            continue
        hit_dist = _ray_hit_distance(shape, shooter, cand)
        if hit_dist is None or hit_dist < 3:
            continue
        if _clear_before_hit(shape, shooter, cand, occupied, hit_dist):
            return cand, hit_dist
    return None


def _choose_free_sprite(
    rng: random.Random,
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    required_kind: str | None = None,
) -> dict | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    for cand in candidates:
        if required_kind is not None and cand["kind"] != required_kind:
            continue
        if _fits(cand, occupied, reserved_bg):
            return cand
    return None


def _build_linear(
    rng: random.Random,
    shape: tuple[int, int],
    placements: list[dict],
) -> tuple[list[dict], set[tuple[int, int]], bool] | None:
    fired_target = rng.randint(*FIRED_COUNT_RANGE)
    start = rng.choice(placements)
    sprites = [start]
    occupied = set(start["cells"])
    reserved_bg: set[tuple[int, int]] = set()

    for _ in range(fired_target - 1):
        chosen = _choose_next_hit(rng, shape, placements, sprites[-1], occupied, reserved_bg)
        if chosen is None:
            return None
        nxt, hit_dist = chosen
        reserved_bg.update(_path_before_hit(shape, sprites[-1], hit_dist))
        sprites.append(nxt)
        occupied.update(nxt["cells"])

    return sprites, reserved_bg, False


def _build_mutual_pair(
    rng: random.Random,
    shape: tuple[int, int],
) -> tuple[list[dict], set[tuple[int, int]], bool] | None:
    h, w = shape
    options: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for r in range(2, h - 2):
        for c1 in range(2, w - 5):
            for dist in range(4, min(12, w - c1 - 1)):
                c2 = c1 + dist
                options.append(((r, c1), (0, 1), (r, c2), (0, -1)))

    for c in range(2, w - 2):
        for r1 in range(2, h - 5):
            for dist in range(4, min(12, h - r1 - 1)):
                r2 = r1 + dist
                options.append(((r1, c), (1, 0), (r2, c), (-1, 0)))

    for dr, dc in DIAG_DIRS:
        for r in range(2, h - 2):
            for c in range(2, w - 2):
                for dist in range(4, 10):
                    r2 = r + dr * dist
                    c2 = c + dc * dist
                    if 2 <= r2 < h - 2 and 2 <= c2 < w - 2:
                        options.append(((r, c), (dr, dc), (r2, c2), (-dr, -dc)))

    rng.shuffle(options)
    for a1, d1, a2, d2 in options:
        p1 = _placement(a1, d1, shape)
        p2 = _placement(a2, d2, shape)
        if p1 is None or p2 is None:
            continue
        if p1["halo"] & set(p2["cells"]):
            continue
        hit12 = _ray_hit_distance(shape, p1, p2)
        hit21 = _ray_hit_distance(shape, p2, p1)
        if hit12 is None or hit21 is None or min(hit12, hit21) < 3:
            continue
        reserved = _path_before_hit(shape, p1, hit12) | _path_before_hit(shape, p2, hit21)
        return [p1, p2], reserved, True
    return None


def _render_input(
    rng: random.Random,
    shape: tuple[int, int],
    bg: int,
    sprites: list[dict],
    reserved_bg: set[tuple[int, int]],
) -> np.ndarray | None:
    grid = np.full(shape, bg, dtype=int)
    colors = [c for c in SPRITE_PALETTE if c != bg]
    if rng.random() < 0.55:
        chosen = [rng.choice(colors)] * len(sprites)
    else:
        chosen = [rng.choice(colors) for _ in sprites]

    for sprite, color in zip(sprites, chosen, strict=False):
        for r, c in sprite["cells"]:
            grid[r, c] = int(color)

    start_cells = set(sprites[0]["cells"])
    other_cells = set().union(*(set(s["cells"]) for s in sprites[1:])) if len(sprites) > 1 else set()
    marker_options = []
    for r, c in start_cells:
        for rr, cc in _neighbors4(r, c):
            if not _in_bounds(shape, rr, cc):
                continue
            if grid[rr, cc] != bg or (rr, cc) in reserved_bg:
                continue
            if any(abs(rr - r2) + abs(cc - c2) == 1 for r2, c2 in other_cells):
                continue
            marker_options.append((rr, cc))
    if not marker_options:
        return None
    mr, mc = rng.choice(marker_options)
    grid[mr, mc] = GRAY
    return grid


def _add_distractors(
    rng: random.Random,
    shape: tuple[int, int],
    placements: list[dict],
    sprites: list[dict],
    reserved_bg: set[tuple[int, int]],
) -> list[dict] | None:
    occupied = set().union(*(set(s["cells"]) for s in sprites))
    target_total = rng.randint(max(len(sprites) + 1, SPRITE_COUNT_RANGE[0]), SPRITE_COUNT_RANGE[1])

    kinds = {s["kind"] for s in sprites}
    if kinds != {"L", "V"}:
        missing = ({"L", "V"} - kinds).pop()
        extra = _choose_free_sprite(rng, placements, occupied, reserved_bg, missing)
        if extra is None:
            return None
        sprites.append(extra)
        occupied.update(extra["cells"])

    while len(sprites) < target_total:
        extra = _choose_free_sprite(rng, placements, occupied, reserved_bg)
        if extra is None:
            return None
        sprites.append(extra)
        occupied.update(extra["cells"])
    return sprites


def _start_only_output(grid: np.ndarray) -> np.ndarray | None:
    try:
        bg, sprites, start_id = _parse(grid)
        cell_to_sprite = {cell: int(s["id"]) for s in sprites for cell in s["cells"]}
        start = sprites[start_id]
        trail = set()
        r, c = start["apex"]
        dr, dc = start["dir"]
        r += dr
        c += dc
        while _in_bounds(grid.shape, r, c):
            if (r, c) in cell_to_sprite:
                break
            if int(grid[r, c]) == bg:
                trail.add((r, c))
            r += dr
            c += dc
        return _render(grid, sprites, {"trail": trail, "fired": {start_id}})
    except Exception:
        return None


def _semantically_valid(grid: np.ndarray, require_old_hit: bool) -> bool:
    try:
        bg, sprites, start_id = _parse(grid)
        sim = _simulate_fire(grid, bg, sprites, start_id)
        out = _render(grid, sprites, sim)
    except Exception:
        return False

    if max(grid.shape) > 30 or grid.shape != out.shape:
        return False
    if len(sprites) < SPRITE_COUNT_RANGE[0]:
        return False
    if {s["kind"] for s in sprites} != {"L", "V"}:
        return False
    if len(sim["fired"]) < 2 or len(sim["fired"]) >= len(sprites):
        return False
    if not sim["hit_new"] or (require_old_hit and not sim["hit_old"]):
        return False
    if len(sim["trail"]) < len(sim["fired"]) + 1:
        return False
    if sim["crossed_non_bg"]:
        return False

    shortcut = _start_only_output(grid)
    if shortcut is None or np.array_equal(shortcut, out):
        return False

    all_maroon = grid.copy()
    for sprite in sprites:
        for r, c in sprite["cells"]:
            all_maroon[r, c] = MAROON
    if np.array_equal(all_maroon, out):
        return False

    return bool(np.array_equal(main(grid), out))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(MIN_SIZE, MAX_SIZE)
    w = rng.randint(MIN_SIZE, MAX_SIZE)
    if rng.random() < 0.4:
        w = rng.randint(max(MIN_SIZE, h - 4), min(MAX_SIZE, h + 5))
    shape = (h, w)
    bg = rng.choice(BACKGROUND_COLORS)
    placements = _all_placements(shape)

    if rng.random() < 0.35:
        built = _build_mutual_pair(rng, shape)
    else:
        built = _build_linear(rng, shape, placements)
    if built is None:
        return None
    sprites, reserved_bg, require_old_hit = built

    sprites = _add_distractors(rng, shape, placements, sprites, reserved_bg)
    if sprites is None:
        return None
    grid = _render_input(rng, shape, bg, sprites, reserved_bg)
    if grid is None:
        return None
    if not _semantically_valid(grid, require_old_hit):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
