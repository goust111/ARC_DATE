from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, find_components, in_bounds, neighbors4


def _coord_key(coord: tuple[int, int]) -> tuple[int, int]:
    return int(coord[0]), int(coord[1])


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.asarray(coords, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return [tuple(map(int, cell)) for cell in arr]


def _rotate_coords(coords: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = [tuple(map(int, cell)) for cell in coords]
    for _ in range(int(turns) % 4):
        max_c = max(c for _, c in out)
        out = [(c, max_c - r) for r, c in out]
        out = _normalize_coords(out)
    return out


def _translate_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords}


def _min_manhattan(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> int:
    best = 10**9
    for ra, ca in coords_a:
        for rb, cb in coords_b:
            best = min(best, abs(int(ra) - int(rb)) + abs(int(ca) - int(cb)))
    return int(best)


def _extract_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(np.asarray(grid, dtype=int), bg=0, connectivity=4, monochromatic=True)
    out = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        out.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "coord_set": _coords_to_set(coords),
                "bbox": tuple(map(int, comp["bbox"])),
            }
        )
    return out


def _dilation_zone(grid: np.ndarray, coord_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in coord_set:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == 0:
                zone.add((int(nr), int(nc)))
    return zone


def _hole_mask_for_component(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = bbox_coords(coords)
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    black = region == 0
    if not black.any():
        return out
    h, w = black.shape
    holes = np.zeros_like(black, dtype=bool)
    for comp in components_mask(black, connectivity=4):
        touches_border = False
        for rr, cc in comp:
            rr = int(rr)
            cc = int(cc)
            if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                touches_border = True
                break
        if not touches_border:
            holes[comp[:, 0], comp[:, 1]] = True
    out[r0 : r1 + 1, c0 : c1 + 1] = holes
    return out


def _connectable(comp_a: dict, comp_b: dict) -> bool:
    zone_a = comp_a["dilation"]
    zone_b = comp_b["dilation"]
    return bool((zone_a & zone_b) or (zone_a & comp_b["coord_set"]) or (zone_b & comp_a["coord_set"]))


def _closest_pair(comp_a: dict, comp_b: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    best = None
    best_pair = None
    for ra, ca in comp_a["coord_set"]:
        for rb, cb in comp_b["coord_set"]:
            key = (abs(ra - rb) + abs(ca - cb), ra, ca, rb, cb)
            if best is None or key < best:
                best = key
                best_pair = ((ra, ca), (rb, cb))
    assert best_pair is not None
    return best_pair


def _path_points(p_a: tuple[int, int], p_b: tuple[int, int]) -> list[tuple[int, int]]:
    ra, ca = map(int, p_a)
    rb, cb = map(int, p_b)
    pts: list[tuple[int, int]] = []
    if ra == rb:
        step = 1 if cb >= ca else -1
        for c in range(ca, cb + step, step):
            pts.append((ra, int(c)))
        return pts
    if ca == cb:
        step = 1 if rb >= ra else -1
        for r in range(ra, rb + step, step):
            pts.append((int(r), ca))
        return pts

    step_c = 1 if cb >= ca else -1
    for c in range(ca, cb + step_c, step_c):
        pts.append((ra, int(c)))
    step_r = 1 if rb >= ra else -1
    for r in range(ra + step_r, rb + step_r, step_r):
        pts.append((int(r), cb))
    return pts


def _analyze_scene(grid: np.ndarray) -> dict:
    comps = _extract_components(grid)
    hole_mask = np.zeros_like(grid, dtype=bool)
    for comp in comps:
        comp["dilation"] = _dilation_zone(grid, comp["coord_set"])
        comp_holes = _hole_mask_for_component(grid, comp["coords"])
        comp["holes"] = comp_holes
        hole_mask |= comp_holes

    pairs = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i]["color"] != comps[j]["color"]:
                continue
            if _connectable(comps[i], comps[j]):
                p_a, p_b = _closest_pair(comps[i], comps[j])
                pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "color": comps[i]["color"],
                        "p_a": p_a,
                        "p_b": p_b,
                        "path": _path_points(p_a, p_b),
                    }
                )
    return {"components": comps, "hole_mask": hole_mask, "pairs": pairs}


def _apply_connections(grid: np.ndarray, scene: dict, *, use_hole_exemption: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    hole_mask = scene["hole_mask"]
    for pair in scene["pairs"]:
        color = int(pair["color"])
        for r, c in pair["path"]:
            if int(out[r, c]) != 0:
                continue
            if use_hole_exemption and bool(hole_mask[r, c]):
                continue
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect only the near-enough same-color objects, but leave enclosed hole pixels unfilled.

    Input structure:
    - A black background contains 4-connected single-color foreground objects in
      2 or 3 non-black colors.
    - Some same-color objects sit one dilation step apart, while other same-color
      objects remain farther away.
    - At least one object may contain a fully enclosed black hole inside its bbox.

    Rule:
    - Find every 4-connected foreground component and group them by color.
    - For each component, mark its one-step 4-neighbor dilation zone on black
      cells only; two same-color components are connectable iff those marked zones
      touch (or, equivalently here, a marked cell reaches the other component).
    - For every connectable same-color pair, choose the closest pixel pair by
      Manhattan distance with row-major tie-breaking, then draw the prescribed
      axis-aligned shortest path between them.
    - Recolor only black path cells to that shared color, except any path cell
      that lies inside an enclosed black hole of any foreground object remains black.

    Key insight:
    - The dilation is not drawn in the output; it is only the proximity test that
      decides which same-color objects get linked, and the hole mask can veto an
      otherwise valid fill cell on that path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    scene = _analyze_scene(g)
    return _apply_connections(g, scene, use_hole_exemption=True)


OUTER_FRAME_SIZE = 6
MAX_ATTEMPTS = 400
GRID_SIZE_RANGE = (18, 25)
COLOR_POOL = tuple(range(1, 10))
MIN_SEPARATION_DIFF_COLOR = 4
MIN_SEPARATION_SAME_COLOR_FAR = 4


def _base_frame_coords() -> list[tuple[int, int]]:
    coords = []
    for r in range(OUTER_FRAME_SIZE):
        for c in range(OUTER_FRAME_SIZE):
            if r in (0, OUTER_FRAME_SIZE - 1) or c in (0, OUTER_FRAME_SIZE - 1):
                coords.append((r, c))
    return coords


def _make_hole_pair(rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    outer = _base_frame_coords()
    side = rng.choice(["top", "bottom", "left", "right"])
    if side == "top":
        outer.append((-1, 2))
    elif side == "bottom":
        outer.append((OUTER_FRAME_SIZE, 3))
    elif side == "left":
        outer.append((2, -1))
    else:
        outer.append((3, OUTER_FRAME_SIZE))
    inner = [(2, 2), (2, 3), (3, 2)]
    all_outer = _normalize_coords(outer)
    min_r = min(r for r, _ in outer)
    min_c = min(c for _, c in outer)
    inner = [(r - min_r, c - min_c) for r, c in inner]
    return all_outer, inner


def _make_visible_pair(rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]], str]:
    a = [(0, 0), (1, 0), (1, 1)]
    b = [(0, 0), (0, 1), (1, 1)]
    branch = rng.choice(["row", "col", "l"])
    if branch == "row":
        b = _translate_coords(b, 0, 2)
        turns = rng.choice([0, 2])
    elif branch == "col":
        b = _translate_coords(b, 0, 2)
        turns = rng.choice([1, 3])
    else:
        b = _translate_coords(b, 2, 2)
        turns = rng.randrange(4)
    a = _rotate_coords(a, turns)
    b = _rotate_coords(b, turns)
    return a, b, branch


def _random_polyomino(rng: random.Random, area: int) -> list[tuple[int, int]]:
    area = int(area)
    for _ in range(100):
        cells = {(0, 0)}
        frontier = {(0, 0)}
        while len(cells) < area:
            base = rng.choice(tuple(frontier))
            candidates = []
            for nr, nc in neighbors4(base[0], base[1]):
                if (nr, nc) not in cells:
                    candidates.append((int(nr), int(nc)))
            if not candidates:
                frontier = set(cells)
                continue
            cells.add(rng.choice(candidates))
            frontier = set(cells)
        coords = _normalize_coords(sorted(cells))
        rs = {r for r, _ in coords}
        cs = {c for _, c in coords}
        if len(rs) > 1 and len(cs) > 1:
            return coords
    return _normalize_coords(sorted(cells))


def _globalize(local: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in local]


def _place_cluster(
    rng: random.Random,
    shape_sets: list[list[tuple[int, int]]],
    color: int,
    existing: list[dict],
    shape: tuple[int, int],
    *,
    same_color_far: bool = False,
) -> list[dict] | None:
    H, W = map(int, shape)
    union = sorted({cell for coords in shape_sets for cell in coords})
    max_r = max(r for r, _ in union)
    max_c = max(c for _, c in union)
    for _ in range(120):
        top = rng.randint(1, H - max_r - 2)
        left = rng.randint(1, W - max_c - 2)
        placed = [_globalize(coords, top, left) for coords in shape_sets]
        ok = True
        for new_coords in placed:
            new_set = set(new_coords)
            if len(new_set) != len(new_coords):
                ok = False
                break
            for obj in existing:
                old = obj["coords"]
                if new_set & set(old):
                    ok = False
                    break
                min_dist = _min_manhattan(new_coords, old)
                if obj["color"] == color and same_color_far:
                    if min_dist < MIN_SEPARATION_SAME_COLOR_FAR:
                        ok = False
                        break
                elif obj["color"] != color and min_dist < MIN_SEPARATION_DIFF_COLOR:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return [{"color": int(color), "coords": coords} for coords in placed]
    return None


def _build_grid(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["coords"]:
            grid[int(r), int(c)] = color
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    scene = _analyze_scene(grid)
    comps = scene["components"]
    if not (4 <= len(comps) <= 6):
        return False

    colors = sorted({int(comp["color"]) for comp in comps})
    if not (2 <= len(colors) <= 3):
        return False

    if not bool(scene["hole_mask"].any()):
        return False

    same_color_pairs = 0
    nonconnect_pairs = 0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i]["color"] != comps[j]["color"]:
                continue
            same_color_pairs += 1
            if not _connectable(comps[i], comps[j]):
                nonconnect_pairs += 1
    if same_color_pairs < 3 or nonconnect_pairs < 1:
        return False

    if len(scene["pairs"]) != 2:
        return False

    out = main(grid)
    naive = _apply_connections(grid, scene, use_hole_exemption=False)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, naive):
        return False

    changed = (out != grid)
    if not bool((changed & (~scene["hole_mask"])).any()):
        return False
    if not bool(((naive != out) & scene["hole_mask"]).any()):
        return False

    if any(int(grid[r, c]) != 0 for r, c in np.argwhere(changed)):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(*GRID_SIZE_RANGE)
    W = rng.randint(*GRID_SIZE_RANGE)
    colors = list(rng.sample(COLOR_POOL, 3))
    hole_color, near_color, solo_color = colors

    objects: list[dict] = []

    outer, inner = _make_hole_pair(rng)
    hole_objs = _place_cluster(rng, [outer, inner], hole_color, objects, (H, W))
    if hole_objs is None:
        return None
    objects.extend(hole_objs)

    vis_a, vis_b, _branch = _make_visible_pair(rng)
    near_objs = _place_cluster(rng, [vis_a, vis_b], near_color, objects, (H, W))
    if near_objs is None:
        return None
    objects.extend(near_objs)

    far_shape = _random_polyomino(rng, rng.randint(3, 6))
    far_obj = _place_cluster(rng, [far_shape], near_color, objects, (H, W), same_color_far=True)
    if far_obj is None:
        return None
    objects.extend(far_obj)

    if rng.random() < 0.7:
        solo_shape = _random_polyomino(rng, rng.randint(3, 7))
        solo_obj = _place_cluster(rng, [solo_shape], solo_color, objects, (H, W))
        if solo_obj is None:
            return None
        objects.extend(solo_obj)

    grid = _build_grid(objects, (H, W))
    if _validate_generated(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for seed in range(MAX_ATTEMPTS):
        rng = random.Random(seed * 1009 + random.randrange(10**9))
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid puzzle instance")
