from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


HIT_COLOR = 9
COLLISION_COLOR = 1
INPUT_PALETTE = [0, 2, 3, 4, 5, 6, 7, 8]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    steps = neighbors4 if connectivity == 4 else neighbors8
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in steps(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    if len(comp) != 3:
        return None
    cells = set(comp)
    deg4 = {cell: sum(nei in cells for nei in neighbors4(*cell)) for cell in comp}
    deg8 = {cell: sum(nei in cells for nei in neighbors8(*cell)) for cell in comp}

    corners = [cell for cell, deg in deg4.items() if deg == 2]
    if corners:
        apex = corners[0]
        ar, ac = apex
        arms = []
        for r, c in comp:
            if (r, c) == apex:
                continue
            dr, dc = r - ar, c - ac
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr = -(arms[0][0] + arms[1][0])
        dc = -(arms[0][1] + arms[1][1])
        dr = 0 if dr == 0 else (1 if dr > 0 else -1)
        dc = 0 if dc == 0 else (1 if dc > 0 else -1)
        return apex, (dr, dc), "L"

    apexes = [cell for cell, deg in deg8.items() if deg == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [cell for cell in comp if cell != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    dr = -((d1[0] + d2[0]) // 2)
    dc = -((d1[1] + d2[1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc), "V"


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _halo(cells: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    H, W = shape
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    triangles: list[dict] = []
    singletons_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    other_components: list[dict] = []
    cell_to_triangle: dict[tuple[int, int], int] = {}

    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        for coords in _components_mask(g == color, connectivity=8):
            cells = [(int(r), int(c)) for r, c in coords]
            if len(cells) == 1:
                singletons_by_color[color].append(cells[0])
                continue
            tri = _triangle_apex_and_dir(cells) if len(cells) == 3 else None
            if tri is None:
                other_components.append({"color": color, "cells": cells})
                continue
            apex, direction, kind = tri
            tid = len(triangles)
            triangles.append(
                {
                    "id": tid,
                    "color": color,
                    "cells": tuple(cells),
                    "cell_set": frozenset(cells),
                    "apex": apex,
                    "dir": direction,
                    "kind": kind,
                }
            )
            for cell in cells:
                cell_to_triangle[cell] = tid

    return {
        "grid": g,
        "bg": bg,
        "triangles": triangles,
        "singletons_by_color": {int(k): list(v) for k, v in singletons_by_color.items()},
        "other_components": other_components,
        "cell_to_triangle": cell_to_triangle,
    }


def _simulate(grid: np.ndarray) -> dict:
    scene = _extract_scene(grid)
    g = scene["grid"]
    bg = scene["bg"]
    out = g.copy()
    triangle_colors = {tri["color"] for tri in scene["triangles"]}

    consumed_singletons: list[tuple[int, int]] = []
    unowned_singletons: list[tuple[int, int]] = []
    for color, coords in scene["singletons_by_color"].items():
        if color in triangle_colors:
            for rc in coords:
                out[rc] = bg
                consumed_singletons.append(rc)
        else:
            unowned_singletons.extend(coords)

    fuel_counts = {
        color: len(scene["singletons_by_color"].get(color, []))
        for color in triangle_colors
    }
    bullets: list[dict] = []
    for tri in scene["triangles"]:
        k = fuel_counts[tri["color"]]
        dr, dc = tri["dir"]
        ar, ac = tri["apex"]
        dest = (ar + k * dr, ac + k * dc)
        bullets.append({"shooter": tri["id"], "dest": dest, "range": k})

    bg_dest_counts: Counter[tuple[int, int]] = Counter()
    hit_triangle_ids: set[int] = set()
    H, W = g.shape
    for bullet in bullets:
        r, c = bullet["dest"]
        if not (0 <= r < H and 0 <= c < W):
            continue
        target_id = scene["cell_to_triangle"].get((r, c))
        if target_id is not None and target_id != bullet["shooter"]:
            hit_triangle_ids.add(target_id)
        elif target_id is None:
            bg_dest_counts[(r, c)] += 1

    for tid in hit_triangle_ids:
        for r, c in scene["triangles"][tid]["cells"]:
            out[r, c] = HIT_COLOR

    for (r, c), count in bg_dest_counts.items():
        out[r, c] = COLLISION_COLOR if count >= 2 else HIT_COLOR

    return {
        "scene": scene,
        "out": out,
        "fuel_counts": fuel_counts,
        "bullets": bullets,
        "bg_dest_counts": bg_dest_counts,
        "hit_triangle_ids": hit_triangle_ids,
        "consumed_singletons": consumed_singletons,
        "unowned_singletons": unowned_singletons,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each triangle once; its same-color fuel dots decide how far the projectile travels.

    Input structure:
    - The dominant color is the background.
    - Foreground objects are isolated 3-pixel triangle sprites plus isolated singleton dots.
    - A sprite is either an axis-aligned L that fires diagonally away from its missing corner, or a diagonal V that fires axially away from its open side.
    - Singleton dots whose color matches a triangle color are that color's fuel; singleton dots of colors with no triangle are unrelated debris.

    Rule:
    - For every triangle, count all singleton dots in the whole grid that share its color.
    - Starting from the triangle apex, move exactly that many cells along the triangle's firing direction and land one projectile there.
    - After counting, erase every singleton fuel dot whose color belongs to at least one triangle; unrelated singleton dots stay unchanged.
    - All projectiles resolve simultaneously from the input scene: a projectile landing on another triangle recolors that whole triangle to maroon 9; a projectile landing on empty non-triangle space paints maroon 9; if multiple projectiles land on the same empty cell, that cell becomes blue 1.

    Key insight:
    - The shooting distance is not encoded by shape or a fixed constant; it is the per-color count of matching singleton fuel dots, and consuming only the matched colors disambiguates fuel from unrelated debris.
    """
    return _simulate(np.asarray(grid, dtype=int))["out"]


def _build_placements(H: int, W: int, max_range: int) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                cells = _sprite_cells((ar, ac), direction)
                if not all(0 <= r < H and 0 <= c < W for r, c in cells):
                    continue
                halo = _halo(cells, (H, W))
                kind = "L" if abs(direction[0]) + abs(direction[1]) == 2 else "V"
                for k in range(1, max_range + 1):
                    dest = (ar + k * direction[0], ac + k * direction[1])
                    if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                        continue
                    placement = {
                        "apex": (ar, ac),
                        "dir": direction,
                        "kind": kind,
                        "k": k,
                        "dest": dest,
                        "cells": cells,
                        "cell_set": frozenset(cells),
                        "halo": halo,
                    }
                    placements.append(placement)
                    by_dest[dest].append(placement)
    return placements, by_dest


def _fits_triangle(placement: dict, occupied: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied)


def _choose_collision_pair(
    rng: random.Random,
    by_dest: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(by_dest)
    rng.shuffle(dests)
    for dest in dests:
        if dest in occupied or dest in reserved_bg:
            continue
        candidates = [p for p in by_dest[dest] if _fits_triangle(p, occupied)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in occupied or placement["dest"] in reserved_bg:
            continue
        if _fits_triangle(placement, occupied):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    by_dest: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        candidates = [p for p in by_dest[cell] if _fits_triangle(p, occupied)]
        rng.shuffle(candidates)
        for placement in candidates:
            if placement["cell_set"] & target_sprite["placement"]["cell_set"]:
                continue
            return placement, cell
    return None


def _forbidden_singletons(
    H: int,
    W: int,
    triangle_cells: set[tuple[int, int]],
    singleton_cells: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    forbidden = set(reserved_bg) | triangle_cells | singleton_cells
    for r, c in triangle_cells | singleton_cells:
        for rr in range(max(0, r - 1), min(H, r + 2)):
            for cc in range(max(0, c - 1), min(W, c + 2)):
                forbidden.add((rr, cc))
    return forbidden


def _place_singletons(
    rng: random.Random,
    H: int,
    W: int,
    counts_by_color: dict[int, int],
    triangle_cells: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict[int, list[tuple[int, int]]] | None:
    placed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    singleton_cells: set[tuple[int, int]] = set()
    color_order = list(counts_by_color.items())
    rng.shuffle(color_order)
    for color, count in color_order:
        for _ in range(count):
            forbidden = _forbidden_singletons(H, W, triangle_cells, singleton_cells, reserved_bg)
            pool = [
                (r, c)
                for r in range(H)
                for c in range(W)
                if (r, c) not in forbidden
            ]
            if not pool:
                return None
            cell = rng.choice(pool)
            placed[color].append(cell)
            singleton_cells.add(cell)
    return {int(k): list(v) for k, v in placed.items()}


def _render_input(
    H: int,
    W: int,
    bg: int,
    sprites: list[dict],
    singletons_by_color: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((H, W), bg, dtype=int)
    for sprite in sprites:
        for r, c in sprite["placement"]["cells"]:
            grid[r, c] = sprite["color"]
    for color, coords in singletons_by_color.items():
        for r, c in coords:
            grid[r, c] = color
    return grid


def _semantic_ok(inp: np.ndarray) -> bool:
    sim = _simulate(inp)
    scene = sim["scene"]
    if scene["other_components"]:
        return False
    if len(scene["triangles"]) < 5:
        return False
    if {tri["kind"] for tri in scene["triangles"]} != {"L", "V"} and not (
        {"L", "V"} <= {tri["kind"] for tri in scene["triangles"]}
    ):
        return False
    fuel_values = [sim["fuel_counts"][tri["color"]] for tri in scene["triangles"]]
    if any(v < 1 for v in fuel_values):
        return False
    if len(set(fuel_values)) < 2:
        return False
    if not sim["consumed_singletons"] or not sim["unowned_singletons"]:
        return False
    if not sim["hit_triangle_ids"]:
        return False
    if not any(count >= 2 for count in sim["bg_dest_counts"].values()):
        return False
    if not any(count == 1 for count in sim["bg_dest_counts"].values()):
        return False
    singleton_coords = {
        rc
        for coords in scene["singletons_by_color"].values()
        for rc in coords
    }
    if any(dest in singleton_coords for dest in sim["bg_dest_counts"]):
        return False
    out = sim["out"]
    if not np.any(out == COLLISION_COLOR) or not np.any(out == HIT_COLOR):
        return False
    for rc in sim["consumed_singletons"]:
        if out[rc] != scene["bg"]:
            return False
    for rc in sim["unowned_singletons"]:
        if out[rc] != inp[rc]:
            return False
    return np.array_equal(main(inp), out)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(18, 24)
    W = rng.randint(18, 24)
    max_range = rng.randint(3, 5)
    bg = rng.choice(INPUT_PALETTE)
    usable_colors = [c for c in INPUT_PALETTE if c != bg]
    n_triangles = rng.randint(5, 6)
    if len(usable_colors) < n_triangles + 1:
        return None

    placements, by_dest = _build_placements(H, W, max_range)
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    sprites: list[dict] = []

    collision = _choose_collision_pair(rng, by_dest, occupied, reserved_bg)
    if collision is None:
        return None
    dest, pair = collision
    reserved_bg.add(dest)
    for placement in pair:
        sprites.append({"placement": placement, "target_kind": "bg", "target": dest, "role": "collision"})
        occupied.update(placement["cells"])

    target_placement = _choose_unique(rng, placements, occupied, reserved_bg)
    if target_placement is None:
        return None
    target_idx = len(sprites)
    sprites.append(
        {
            "placement": target_placement,
            "target_kind": "bg",
            "target": target_placement["dest"],
            "role": "target",
        }
    )
    occupied.update(target_placement["cells"])
    reserved_bg.add(target_placement["dest"])

    hit = _choose_hit(rng, sprites[target_idx], by_dest, occupied)
    if hit is None:
        return None
    hit_placement, hit_cell = hit
    sprites.append(
        {
            "placement": hit_placement,
            "target_kind": "hit",
            "target": hit_cell,
            "target_sprite": target_idx,
            "role": "hit",
        }
    )
    occupied.update(hit_placement["cells"])

    while len(sprites) < n_triangles:
        placement = _choose_unique(rng, placements, occupied, reserved_bg)
        if placement is None:
            return None
        sprites.append({"placement": placement, "target_kind": "bg", "target": placement["dest"], "role": "unique"})
        occupied.update(placement["cells"])
        reserved_bg.add(placement["dest"])

    if not {"L", "V"} <= {sprite["placement"]["kind"] for sprite in sprites}:
        return None
    k_values = [sprite["placement"]["k"] for sprite in sprites]
    if len(set(k_values)) < 2:
        return None

    rng.shuffle(usable_colors)
    triangle_colors = usable_colors[:n_triangles]
    extra_color = usable_colors[n_triangles]
    for sprite, color in zip(sprites, triangle_colors):
        sprite["color"] = color

    counts_by_color = {sprite["color"]: sprite["placement"]["k"] for sprite in sprites}
    counts_by_color[extra_color] = rng.randint(1, max_range)

    singletons_by_color = _place_singletons(rng, H, W, counts_by_color, occupied, reserved_bg)
    if singletons_by_color is None:
        return None

    inp = _render_input(H, W, bg, sprites, singletons_by_color)
    if get_bg(inp) != bg:
        return None
    if not _semantic_ok(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
