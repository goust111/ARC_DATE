from __future__ import annotations

import random
from collections import Counter

import numpy as np


ROT_REFLECTS = (
    (1, 0, 0, 1),
    (0, 1, -1, 0),
    (-1, 0, 0, -1),
    (0, -1, 1, 0),
    (1, 0, 0, -1),
    (0, 1, 1, 0),
    (-1, 0, 0, 1),
    (0, -1, -1, 0),
)

PALETTE = tuple(range(10))
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _normalize_shape(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _transform_pair(
    body: tuple[tuple[int, int], ...],
    lock: tuple[int, int],
    tfm: tuple[int, int, int, int],
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    rr_r, rr_c, cc_r, cc_c = tfm
    body_t = []
    for r, c in body:
        body_t.append((rr_r * r + rr_c * c, cc_r * r + cc_c * c))
    lock_t = (rr_r * lock[0] + rr_c * lock[1], cc_r * lock[0] + cc_c * lock[1])
    min_r = min(r for r, _ in body_t)
    min_c = min(c for _, c in body_t)
    body_n = tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in body_t))
    lock_n = (int(lock_t[0] - min_r), int(lock_t[1] - min_c))
    return body_n, lock_n


def _pair_variants(
    body: tuple[tuple[int, int], ...],
    lock: tuple[int, int],
) -> set[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    return {_transform_pair(body, lock, tfm) for tfm in ROT_REFLECTS}


def _pair_canonical(
    body: tuple[tuple[int, int], ...],
    lock: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    return min(_pair_variants(body, lock))


def _shape_canonical(body: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    variants = []
    for tfm in ROT_REFLECTS:
        rr_r, rr_c, cc_r, cc_c = tfm
        pts = []
        for r, c in body:
            pts.append((rr_r * r + rr_c * c, cc_r * r + cc_c * c))
        variants.append(_normalize_shape(pts))
    return min(variants)


def _grow_polyomino(h: int, w: int, area: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = {start}
    while len(cells) < area:
        base = rng.choice(tuple(frontier))
        nbrs = []
        for nr, nc in _neighbors4(*base):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove(base)
            if not frontier:
                frontier = set(cells)
            continue
        nxt = rng.choice(nbrs)
        cells.add(nxt)
        frontier.add(nxt)
    return _normalize_shape(tuple(cells))


def _lock_candidates(body: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    body_set = set(body)
    h = max(r for r, _ in body) + 1
    w = max(c for _, c in body) + 1
    cand = set()
    for r, c in body:
        if r == 0 and (r - 1, c) not in body_set:
            cand.add((-1, c))
        if r == h - 1 and (r + 1, c) not in body_set:
            cand.add((h, c))
        if c == 0 and (r, c - 1) not in body_set:
            cand.add((r, -1))
        if c == w - 1 and (r, c + 1) not in body_set:
            cand.add((r, w))
    return sorted(cand)


def _make_shape_with_locks(group_size: int, used_shapes: set[tuple[tuple[int, int], ...]], rng: random.Random):
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 5)
        upper = min(7, h * w - 1)
        if upper < 4:
            continue
        area = rng.randint(4, upper)
        body = _grow_polyomino(h, w, area, rng)
        shape_key = _shape_canonical(body)
        if shape_key in used_shapes:
            continue
        lock_options = _lock_candidates(body)
        keyed = {}
        for lock in lock_options:
            keyed[_pair_canonical(body, lock)] = lock
        if len(keyed) < group_size:
            continue
        chosen_keys = rng.sample(list(keyed.keys()), group_size)
        return body, [keyed[key] for key in chosen_keys], shape_key
    return None


def _render_sprite_patch(
    body: tuple[tuple[int, int], ...],
    lock: tuple[int, int],
    body_color: int,
    lock_color: int,
    bg: int,
) -> np.ndarray:
    pts = list(body) + [lock]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - min_r, c - min_c] = int(body_color)
    patch[lock[0] - min_r, lock[1] - min_c] = int(lock_color)
    return patch


def _collect_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _border_background(g)
    fg_components = _components(g != bg)
    assert len(fg_components) >= 4

    comp_infos = []
    color_sets = []
    for coords in fg_components:
        colors = set(map(int, g[coords[:, 0], coords[:, 1]]))
        comp_infos.append({"coords": coords, "colors": colors})
        color_sets.append(colors)

    common = set.intersection(*color_sets)
    common.discard(int(bg))
    assert len(common) == 1
    lock_color = next(iter(common))

    wall_info = max(comp_infos, key=lambda item: int(item["coords"].shape[0]))
    wall_coords = wall_info["coords"]
    r0, c0, r1, c1 = _bbox(wall_coords)
    wall_crop = g[r0 : r1 + 1, c0 : c1 + 1]
    wall_color_counts = Counter(
        int(v)
        for v in wall_crop.ravel()
        if int(v) != int(bg) and int(v) != int(lock_color)
    )
    assert wall_color_counts
    wall_color = int(wall_color_counts.most_common(1)[0][0])

    holes = []
    for hole_coords in _components(wall_crop == bg):
        hr0, hc0, hr1, hc1 = _bbox(hole_coords)
        body = tuple(sorted((int(r - hr0), int(c - hc0)) for r, c in hole_coords))
        lock_hits = set()
        hole_set = {(int(r), int(c)) for r, c in hole_coords}
        for rr, cc in hole_set:
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < wall_crop.shape[0] and 0 <= nc < wall_crop.shape[1] and int(wall_crop[nr, nc]) == int(lock_color):
                    lock_hits.add((int(nr), int(nc)))
        assert len(lock_hits) == 1
        lock_abs = next(iter(lock_hits))
        holes.append(
            {
                "coords_local": tuple(sorted(hole_set)),
                "body": body,
                "lock": (int(lock_abs[0] - hr0), int(lock_abs[1] - hc0)),
                "pair": (body, (int(lock_abs[0] - hr0), int(lock_abs[1] - hc0))),
            }
        )

    sprites = []
    for info in comp_infos:
        if info is wall_info:
            continue
        colors = set(info["colors"])
        assert int(lock_color) in colors and len(colors) == 2
        body_color = next(iter(colors - {int(lock_color)}))
        coords = info["coords"]
        body_coords = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == int(body_color)]
        lock_coords = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == int(lock_color)]
        assert len(lock_coords) == 1
        sr0 = min(r for r, _ in body_coords)
        sc0 = min(c for _, c in body_coords)
        body = tuple(sorted((int(r - sr0), int(c - sc0)) for r, c in body_coords))
        lock = (int(lock_coords[0][0] - sr0), int(lock_coords[0][1] - sc0))
        sprites.append({"body": body, "lock": lock, "body_color": int(body_color)})

    assert len(holes) == len(sprites)
    return {
        "bg": int(bg),
        "lock_color": int(lock_color),
        "wall_color": int(wall_color),
        "wall_bbox": (int(r0), int(c0), int(r1), int(c1)),
        "holes": holes,
        "sprites": sprites,
    }


def _solve_output(grid: np.ndarray) -> np.ndarray:
    scene = _collect_scene(grid)
    r0, c0, r1, c1 = scene["wall_bbox"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(scene["wall_color"]), dtype=int)

    hole_to_color = {}
    for sprite_idx, sprite in enumerate(scene["sprites"]):
        candidates = []
        variants = _pair_variants(sprite["body"], sprite["lock"])
        for hole_idx, hole in enumerate(scene["holes"]):
            if hole["pair"] in variants:
                candidates.append(hole_idx)
        assert len(candidates) == 1
        hole_idx = candidates[0]
        assert hole_idx not in hole_to_color
        hole_to_color[hole_idx] = int(sprite["body_color"])

    assert len(hole_to_color) == len(scene["holes"])
    for idx, hole in enumerate(scene["holes"]):
        color = hole_to_color[idx]
        for r, c in hole["coords_local"]:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate and reflect loose sprites into the congruent wall holes, then return the repaired wall.

    Input structure:
    - The grid has a background, one large rectangular wall component, and several detached sprite components outside the wall.
    - Inside the wall rectangle, each background hole is a polyomino-shaped cavity; each hole has exactly one lock pixel of a shared lock color on the adjacent wall side.
    - Each outside sprite has one body color plus exactly one attached lock pixel of that same shared lock color.

    Rule:
    - For every sprite, search over the eight rotation/reflection poses and find the unique hole whose background shape is exactly filled by the sprite body while the sprite lock lands on that hole's wall lock.
    - Fill each matched hole with the sprite's body color.
    - Recolor every used lock position back to the wall color, erase all outside sprites, and tight-crop to the wall rectangle.

    Key insight:
    - The holes are matched as negative shapes under dihedral symmetry, but shape alone is insufficient because some holes are congruent; the shared lock pixel fixes both the target hole and the final pose.
    """
    return _solve_output(np.asarray(grid, dtype=int))


def _group_pattern(count: int, rng: random.Random) -> list[int]:
    if count == 3:
        return [2, 1]
    return rng.choice(([2, 1, 1], [2, 2], [3, 1]))


def _place_holes(
    items: list[dict],
    wall_h: int,
    wall_w: int,
    rng: random.Random,
):
    blocked = np.zeros((wall_h, wall_w), dtype=bool)
    placed = []
    order = sorted(
        range(len(items)),
        key=lambda idx: (
            -(max(r for r, _ in items[idx]["body"]) + 1) * (max(c for _, c in items[idx]["body"]) + 1)
        ),
    )
    for idx in order:
        item = items[idx]
        body = item["body"]
        lock = item["lock"]
        h = max(r for r, _ in body) + 1
        w = max(c for _, c in body) + 1
        candidates = []
        for r0 in range(1, wall_h - h):
            for c0 in range(1, wall_w - w):
                cells = {(r0 + r, c0 + c) for r, c in body}
                lock_cell = (r0 + lock[0], c0 + lock[1])
                if not (0 <= lock_cell[0] < wall_h and 0 <= lock_cell[1] < wall_w):
                    continue
                if lock_cell in cells:
                    continue
                rr0 = max(0, r0 - 1)
                rr1 = min(wall_h, r0 + h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(wall_w, c0 + w + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0, cells, lock_cell, rr0, rr1, cc0, cc1))
        if not candidates:
            return None
        r0, c0, cells, lock_cell, rr0, rr1, cc0, cc1 = rng.choice(candidates)
        blocked[rr0:rr1, cc0:cc1] = True
        placed.append({"index": idx, "cells": cells, "lock_cell": lock_cell, "top_left": (r0, c0)})
    placed.sort(key=lambda item: item["index"])
    return placed


def _build_expected(
    wall_h: int,
    wall_w: int,
    wall_color: int,
    placements: list[dict],
    items: list[dict],
) -> np.ndarray:
    out = np.full((wall_h, wall_w), int(wall_color), dtype=int)
    for placement in placements:
        item = items[placement["index"]]
        for r, c in placement["cells"]:
            out[r, c] = int(item["body_color"])
    return out


def _build_wall_input(
    wall_h: int,
    wall_w: int,
    bg: int,
    wall_color: int,
    lock_color: int,
    placements: list[dict],
) -> np.ndarray:
    wall = np.full((wall_h, wall_w), int(wall_color), dtype=int)
    for placement in placements:
        for r, c in placement["cells"]:
            wall[r, c] = int(bg)
        lr, lc = placement["lock_cell"]
        wall[lr, lc] = int(lock_color)
    return wall


def _sprite_patches(items: list[dict], rng: random.Random) -> list[np.ndarray]:
    tfm_ids = [rng.randrange(len(ROT_REFLECTS)) for _ in items]
    if all(idx < 4 for idx in tfm_ids):
        tfm_ids[0] = rng.randrange(4, 8)
    patches = []
    for item, tfm_idx in zip(items, tfm_ids):
        body_t, lock_t = _transform_pair(item["body"], item["lock"], ROT_REFLECTS[tfm_idx])
        patches.append(_render_sprite_patch(body_t, lock_t, item["body_color"], item["lock_color"], item["bg"]))
    return patches


def _compose_canvas(
    wall_patch: np.ndarray,
    sprites: list[np.ndarray],
    bg: int,
    rng: random.Random,
) -> np.ndarray | None:
    wall_h, wall_w = wall_patch.shape
    zones = ["top", "bottom", "left", "right"]
    rng.shuffle(zones)
    zones = zones[: len(sprites)]
    assign = dict(zip(zones, sprites))

    top_space = max([patch.shape[0] + 2 for zone, patch in assign.items() if zone == "top"] + [3])
    bottom_space = max([patch.shape[0] + 2 for zone, patch in assign.items() if zone == "bottom"] + [3])
    left_space = max([patch.shape[1] + 2 for zone, patch in assign.items() if zone == "left"] + [3])
    right_space = max([patch.shape[1] + 2 for zone, patch in assign.items() if zone == "right"] + [3])

    canvas_h = wall_h + top_space + bottom_space
    canvas_w = wall_w + left_space + right_space
    if max(canvas_h, canvas_w) > 30:
        return None

    grid = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    wall_r = top_space
    wall_c = left_space
    grid[wall_r : wall_r + wall_h, wall_c : wall_c + wall_w] = wall_patch

    for zone, patch in assign.items():
        ph, pw = patch.shape
        if zone == "top":
            r0 = rng.randint(1, top_space - ph - 1)
            c0 = rng.randint(0, canvas_w - pw)
        elif zone == "bottom":
            r0 = rng.randint(wall_r + wall_h + 1, canvas_h - ph - 1)
            c0 = rng.randint(0, canvas_w - pw)
        elif zone == "left":
            r0 = rng.randint(wall_r, wall_r + wall_h - ph)
            c0 = rng.randint(1, left_space - pw - 1)
        else:
            r0 = rng.randint(wall_r, wall_r + wall_h - ph)
            c0 = rng.randint(wall_c + wall_w + 1, canvas_w - pw - 1)
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any(view != int(bg)):
            return None
        mask = patch != int(bg)
        view[mask] = patch[mask]
    return grid


def _semantic_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        scene = _collect_scene(inp)
        solved = _solve_output(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    shape_keys = [_shape_canonical(hole["body"]) for hole in scene["holes"]]
    counts = Counter(shape_keys)
    if len(counts) < 2:
        return False
    if max(counts.values()) < 2:
        return False
    for sprite in scene["sprites"]:
        hits = 0
        variants = _pair_variants(sprite["body"], sprite["lock"])
        for hole in scene["holes"]:
            if hole["pair"] in variants:
                hits += 1
        if hits != 1:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(3, 4)
    bg, wall_color, lock_color, *body_colors = rng.sample(PALETTE, count + 3)

    groups = _group_pattern(count, rng)
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    items = []
    color_idx = 0
    for group_size in groups:
        built = _make_shape_with_locks(group_size, used_shapes, rng)
        if built is None:
            return None
        body, locks, shape_key = built
        used_shapes.add(shape_key)
        for lock in locks:
            items.append(
                {
                    "body": body,
                    "lock": lock,
                    "body_color": int(body_colors[color_idx]),
                    "lock_color": int(lock_color),
                    "bg": int(bg),
                }
            )
            color_idx += 1
    rng.shuffle(items)

    max_h = max(max(r for r, _ in item["body"]) + 1 for item in items)
    max_w = max(max(c for _, c in item["body"]) + 1 for item in items)
    wall_h = rng.randint(max(10, max_h + 5), 15)
    wall_w = rng.randint(max(11, max_w + 6), 16)

    placements = _place_holes(items, wall_h, wall_w, rng)
    if placements is None:
        return None

    wall_input = _build_wall_input(wall_h, wall_w, bg, wall_color, lock_color, placements)
    expected = _build_expected(wall_h, wall_w, wall_color, placements, items)

    patches = _sprite_patches(items, rng)
    inp = _compose_canvas(wall_input, patches, bg, rng)
    if inp is None:
        return None
    if not _semantic_ok(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-hole docking task")
