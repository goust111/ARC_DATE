from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
SPRITE_COLORS = (1, 2)
OBSTACLE = 5
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _normalize_coords(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _dihedral_variants(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    base = _coords_to_mask(_normalize_coords(coords))
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for mask in (rot, np.fliplr(rot)):
            pts = _normalize_coords(_mask_to_coords(mask))
            if pts not in seen:
                seen.add(pts)
                out.append(pts)
    return out


def _is_transpose_symmetric(coords: tuple[tuple[int, int], ...]) -> bool:
    pts = set(_normalize_coords(coords))
    return pts == {(c, r) for r, c in pts}


def _shape_size(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _bbox_of_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_shape(coords: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in coords]


def _chebyshev_adjacent_to_any(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in others:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _find_components_of_colors(grid: np.ndarray, colors: tuple[int, ...] = SPRITE_COLORS) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in colors or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ORTHO4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            bbox = _bbox_of_cells(cells)
            r0, c0, _, _ = bbox
            local = _normalize_coords([(rr - r0, cc - c0) for rr, cc in cells])
            comps.append(
                {
                    "color": color,
                    "cells": tuple(sorted(cells)),
                    "cell_set": set(cells),
                    "bbox": bbox,
                    "local": local,
                    "size": len(cells),
                    "symmetric": _is_transpose_symmetric(local),
                }
            )
    return comps


def _trace_rays(
    shape: tuple[int, int],
    emitters: list[dict],
    obstacle_cells: set[tuple[int, int]],
    *,
    block_obstacles: bool = True,
    block_emitters: bool = True,
) -> dict:
    h, w = shape
    emitter_cells = set().union(*(e["cell_set"] for e in emitters))
    ray_records: list[dict] = []
    ray_cells_by_color: dict[tuple[int, int], int] = {}
    ray_color_sets: dict[tuple[int, int], set[int]] = {}

    ordered_emitters = sorted(emitters, key=lambda e: (e["bbox"], int(e.get("color", 1)), e["cells"]))
    for emitter_index, emitter in enumerate(ordered_emitters):
        for start in sorted(emitter["cells"]):
            for direction in DIAGONALS:
                dr, dc = direction
                path: list[tuple[int, int]] = []
                r, c = start[0] + dr, start[1] + dc
                stop_reason = "edge"
                stop_cell = None
                while 0 <= r < h and 0 <= c < w:
                    cell = (r, c)
                    if block_obstacles and cell in obstacle_cells:
                        stop_reason = "obstacle"
                        stop_cell = cell
                        break
                    if block_emitters and cell in emitter_cells:
                        stop_reason = "emitter"
                        stop_cell = cell
                        break
                    path.append(cell)
                    ray_color = int(emitter.get("color", 1))
                    ray_cells_by_color.setdefault(cell, ray_color)
                    ray_color_sets.setdefault(cell, set()).add(ray_color)
                    r += dr
                    c += dc
                ray_records.append(
                    {
                        "emitter_index": emitter_index,
                        "start": start,
                        "direction": direction,
                        "path": tuple(path),
                        "stop_reason": stop_reason,
                        "stop_cell": stop_cell,
                        "color": int(emitter.get("color", 1)),
                    }
                )
    return {"ray_map": ray_cells_by_color, "ray_color_sets": ray_color_sets, "records": ray_records}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify sprites by transpose symmetry, let the symmetric ones fire diagonal rays, and erase every hit asymmetric sprite.

    Input structure:
    - The background is black, with 5-7 blue/red monochrome 4-connected sprites and 2-4 gray singleton blockers.
    - Sprite color does not determine its role; role depends on each sprite's local shape inside its own bbox.
    - Blockers are isolated pixels and are not part of any sprite.

    Rule:
    - Split the blue/red sprites into emitters and passives by testing whether each sprite's local coordinate set is invariant under main-diagonal transpose.
    - From every pixel of every emitter, cast rays in the four diagonal directions; each ray stops before the grid edge, before a gray blocker, or before any emitter pixel.
    - If any passive sprite pixel lies on any ray path, erase that entire passive sprite; keep all emitters, all blockers, all unhit passives, and paint the ray paths in the emitter color.

    Key insight:
    - The decisive cue is geometric transpose symmetry of each sprite, not its color; color only distracts from which sprites emit and which are erasable targets.
    """
    g = np.asarray(grid, dtype=int)
    emitters = []
    passives = []
    for comp in _find_components_of_colors(g):
        (emitters if comp["symmetric"] else passives).append(comp)

    obstacle_cells = {(int(r), int(c)) for r, c in np.argwhere(g == OBSTACLE)}
    trace = _trace_rays(g.shape, emitters, obstacle_cells, block_obstacles=True, block_emitters=True)
    ray_map = trace["ray_map"]

    hit_passives = [comp for comp in passives if any(cell in ray_map for cell in comp["cells"])]

    out = g.copy()
    for comp in hit_passives:
        for r, c in comp["cells"]:
            out[r, c] = BACKGROUND
    for (r, c), color in ray_map.items():
        out[r, c] = int(color)
    return out


SYMMETRIC_BASES = [
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
]
ASYMMETRIC_BASES = [
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
]
SYMMETRIC_SHAPES = [
    variant
    for base in SYMMETRIC_BASES
    for variant in _dihedral_variants(base)
    if _is_transpose_symmetric(variant)
]
ASYMMETRIC_SHAPES = [
    variant
    for base in ASYMMETRIC_BASES
    for variant in _dihedral_variants(base)
    if not _is_transpose_symmetric(variant)
]

GEN_ATTEMPTS = 400


def _choose_emitter_pair(h: int, w: int) -> list[dict] | None:
    for _ in range(200):
        s1 = random.choice(SYMMETRIC_SHAPES)
        s2 = random.choice(SYMMETRIC_SHAPES)
        p1 = random.choice(s1)
        p2 = random.choice(s2)
        delta = random.randint(4, 6)
        dh = p1[0] - p2[0] + delta
        dw = p1[1] - p2[1] + delta
        h1, w1 = _shape_size(s1)
        h2, w2 = _shape_size(s2)

        r_lo = max(1, -dh)
        c_lo = max(1, -dw)
        r_hi = min(h - h1 - 2, h - h2 - 2 - dh)
        c_hi = min(w - w1 - 2, w - w2 - 2 - dw)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        top1 = random.randint(r_lo, r_hi)
        left1 = random.randint(c_lo, c_hi)
        top2 = top1 + dh
        left2 = left1 + dw
        cells1 = _place_shape(s1, top1, left1)
        cells2 = _place_shape(s2, top2, left2)
        bbox1 = _bbox_of_cells(cells1)
        bbox2 = _bbox_of_cells(cells2)
        if _bbox_too_close(bbox1, bbox2, gap=1):
            continue

        block_target_r = top2 + p2[0]
        block_target_c = left2 + p2[1]
        if block_target_r + 1 >= h or block_target_c + 1 >= w:
            continue

        emitters = [
            {"shape": s1, "cells": tuple(cells1), "cell_set": set(cells1), "bbox": bbox1, "symmetric": True},
            {"shape": s2, "cells": tuple(cells2), "cell_set": set(cells2), "bbox": bbox2, "symmetric": True},
        ]
        trace = _trace_rays((h, w), emitters, set(), block_obstacles=True, block_emitters=True)
        no_block = _trace_rays((h, w), emitters, set(), block_obstacles=True, block_emitters=False)
        if set(no_block["ray_map"]) - set(trace["ray_map"]):
            return emitters
    return None


def _pick_obstacles(h: int, w: int, emitters: list[dict]) -> list[tuple[int, int]] | None:
    emitter_cells = set().union(*(e["cell_set"] for e in emitters))
    trace = _trace_rays((h, w), emitters, set(), block_obstacles=True, block_emitters=True)
    protected = {
        cell
        for rec in trace["records"]
        if rec["stop_reason"] == "emitter"
        for cell in rec["path"]
    }

    clip_candidates: list[tuple[int, int]] = []
    seen_clip: set[tuple[int, int]] = set()
    for rec in trace["records"]:
        if rec["stop_reason"] != "edge" or len(rec["path"]) < 3:
            continue
        for idx, cell in enumerate(rec["path"][1:-1], start=1):
            if cell in protected or cell in seen_clip:
                continue
            if idx >= len(rec["path"]) - 1:
                continue
            if _chebyshev_adjacent_to_any(cell, emitter_cells):
                continue
            seen_clip.add(cell)
            clip_candidates.append(cell)
    if not clip_candidates:
        return None

    for _ in range(100):
        obstacles = [random.choice(clip_candidates)]
        wanted = random.randint(2, 4)
        available = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in emitter_cells
            and not _chebyshev_adjacent_to_any((r, c), emitter_cells)
            and (r, c) not in protected
        ]
        random.shuffle(available)
        for cell in available:
            if cell in obstacles:
                continue
            if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in obstacles):
                continue
            obstacles.append(cell)
            if len(obstacles) == wanted:
                break
        if len(obstacles) != wanted:
            continue

        final_trace = _trace_rays((h, w), emitters, set(obstacles), block_obstacles=True, block_emitters=True)
        no_obstacles = _trace_rays((h, w), emitters, set(), block_obstacles=False, block_emitters=True)
        if set(no_obstacles["ray_map"]) - set(final_trace["ray_map"]):
            return obstacles
    return None


def _enumerate_passive_candidates(
    h: int,
    w: int,
    shape: tuple[tuple[int, int], ...],
    used_bboxes: list[tuple[int, int, int, int]],
    sprite_cells: set[tuple[int, int]],
    obstacle_cells: set[tuple[int, int]],
    ray_map: dict[tuple[int, int], int],
    *,
    require_hit: bool,
) -> list[dict]:
    sh, sw = _shape_size(shape)
    out: list[dict] = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            cells = _place_shape(shape, top, left)
            bbox = _bbox_of_cells(cells)
            if any(_bbox_too_close(bbox, other, gap=1) for other in used_bboxes):
                continue
            if any(cell in obstacle_cells or cell in sprite_cells for cell in cells):
                continue
            if any(_chebyshev_adjacent_to_any(cell, obstacle_cells) for cell in cells):
                continue
            hit_cells = [cell for cell in cells if cell in ray_map]
            if require_hit:
                if not (1 <= len(hit_cells) < len(cells)):
                    continue
                if len({ray_map[cell] for cell in hit_cells}) != 1:
                    continue
            elif hit_cells:
                continue
            out.append({"shape": shape, "cells": tuple(cells), "bbox": bbox, "symmetric": False})
    random.shuffle(out)
    return out


def _assign_sprite_colors(emitters: list[dict], passives: list[dict]) -> None:
    emitter_color = random.choice(SPRITE_COLORS)
    other_color = 1 if emitter_color == 2 else 2
    for emitter in emitters:
        emitter["color"] = emitter_color
        emitter["cell_set"] = set(emitter["cells"])

    passive_colors = [emitter_color, other_color]
    while len(passive_colors) < len(passives):
        passive_colors.append(random.choice(SPRITE_COLORS))
    random.shuffle(passive_colors)
    for passive, color in zip(passives, passive_colors):
        passive["color"] = color
        passive["cell_set"] = set(passive["cells"])


def _build_grid(h: int, w: int, emitters: list[dict], passives: list[dict], obstacles: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    for obj in emitters + passives:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["color"])
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    comps = _find_components_of_colors(grid)
    if not (5 <= len(comps) <= 7):
        return False
    if not all(3 <= comp["size"] <= 6 for comp in comps):
        return False

    emitters = [comp for comp in comps if comp["symmetric"]]
    passives = [comp for comp in comps if not comp["symmetric"]]
    if len(emitters) < 2 or len(passives) < 2:
        return False

    obstacle_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == OBSTACLE)}
    if not (2 <= len(obstacle_cells) <= 4):
        return False

    for cell in obstacle_cells:
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for comp in comps for rr, cc in comp["cells"]):
            return False
        if sum(1 for other in obstacle_cells if other != cell and max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) <= 1):
            return False

    bboxes = [comp["bbox"] for comp in comps]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _bbox_too_close(bboxes[i], bboxes[j], gap=1):
                return False

    sprite_colors = {int(comp["color"]) for comp in comps}
    emitter_colors = {int(comp["color"]) for comp in emitters}
    passive_colors = {int(comp["color"]) for comp in passives}
    if sprite_colors != {1, 2}:
        return False
    if len(emitter_colors) != 1:
        return False
    if passive_colors != {1, 2}:
        return False
    if not emitter_colors & passive_colors:
        return False
    if passive_colors == emitter_colors:
        return False

    final_trace = _trace_rays(grid.shape, emitters, obstacle_cells, block_obstacles=True, block_emitters=True)
    no_obstacles = _trace_rays(grid.shape, emitters, obstacle_cells, block_obstacles=False, block_emitters=True)
    no_emitters = _trace_rays(grid.shape, emitters, obstacle_cells, block_obstacles=True, block_emitters=False)
    if any(len(colors) > 1 for colors in final_trace["ray_color_sets"].values()):
        return False
    if not (set(no_obstacles["ray_map"]) - set(final_trace["ray_map"])):
        return False
    if not (set(no_emitters["ray_map"]) - set(final_trace["ray_map"])):
        return False

    hit_passives = [comp for comp in passives if any(cell in final_trace["ray_map"] for cell in comp["cells"])]
    if not hit_passives or len(hit_passives) == len(passives):
        return False
    if not any(0 < sum(cell in final_trace["ray_map"] for cell in comp["cells"]) < comp["size"] for comp in hit_passives):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out == OBSTACLE)) != len(obstacle_cells):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 17)
    w = random.randint(13, 17)
    emitters = _choose_emitter_pair(h, w)
    if emitters is None:
        return None

    obstacles = _pick_obstacles(h, w, emitters)
    if obstacles is None:
        return None
    obstacle_cells = set(obstacles)

    trace = _trace_rays((h, w), emitters, obstacle_cells, block_obstacles=True, block_emitters=True)
    ray_map = trace["ray_map"]
    sprite_cells = set().union(*(e["cell_set"] for e in emitters))
    used_bboxes = [e["bbox"] for e in emitters]

    passive_total = random.randint(3, 5)
    hit_needed = 1 if passive_total == 3 else random.randint(1, 2)
    safe_needed = passive_total - hit_needed
    passives: list[dict] = []

    for _ in range(hit_needed):
        candidates: list[dict] = []
        shapes = ASYMMETRIC_SHAPES[:]
        random.shuffle(shapes)
        for shape in shapes:
            candidates.extend(
                _enumerate_passive_candidates(
                    h,
                    w,
                    shape,
                    used_bboxes,
                    sprite_cells,
                    obstacle_cells,
                    ray_map,
                    require_hit=True,
                )
            )
            if candidates:
                break
        if not candidates:
            return None
        choice = random.choice(candidates)
        passives.append(choice)
        used_bboxes.append(choice["bbox"])
        sprite_cells.update(choice["cells"])

    for _ in range(safe_needed):
        candidates = []
        shapes = ASYMMETRIC_SHAPES[:]
        random.shuffle(shapes)
        for shape in shapes:
            candidates.extend(
                _enumerate_passive_candidates(
                    h,
                    w,
                    shape,
                    used_bboxes,
                    sprite_cells,
                    obstacle_cells,
                    ray_map,
                    require_hit=False,
                )
            )
            if candidates:
                break
        if not candidates:
            return None
        choice = random.choice(candidates)
        passives.append(choice)
        used_bboxes.append(choice["bbox"])
        sprite_cells.update(choice["cells"])

    _assign_sprite_colors(emitters, passives)
    grid = _build_grid(h, w, emitters, passives, obstacles)
    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
