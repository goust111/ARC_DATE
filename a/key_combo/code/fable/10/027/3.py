from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
RING_BORDER_OFFSETS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
BODY_OFFSETS_BY_RAY = {
    (-1, 1): [(1, 0), (0, -1), (1, -1), (2, -1), (1, -2)],
    (-1, -1): [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
    (1, 1): [(-1, 0), (0, -1), (-1, -1), (-2, -1), (-1, -2)],
    (1, -1): [(-1, 0), (0, 1), (-1, 1), (-2, 1), (-1, 2)],
}

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
SPRITE_COUNT_OPTIONS = (2, 3)
EXTRA_RING_OPTIONS = (1, 2)
GENERATION_RETRIES = 800


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = np.array(list(coords), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _ring_cells_from_tl(top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in RING_BORDER_OFFSETS]


def _detect_rings(grid: np.ndarray, bg: int) -> list[dict]:
    rings = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components(grid == color, connectivity=8):
            if int(coords.shape[0]) != 8:
                continue
            r0, c0, r1, c1 = _bbox_from_coords(map(tuple, coords.tolist()))
            if (r1 - r0, c1 - c0) != (2, 2):
                continue
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            if int(window[1, 1]) != bg:
                continue
            if any(int(window[dr, dc]) != color for dr, dc in RING_BORDER_OFFSETS):
                continue
            rings.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "cells": _ring_cells_from_tl(r0, c0),
                    "center": (r0 + 1, c0 + 1),
                }
            )
    return rings


def _parse_sprites(grid: np.ndarray, bg: int, rings: list[dict]) -> list[dict]:
    ring_colors = {ring["color"] for ring in rings}
    colors = sorted(set(map(int, grid.ravel())) - {bg} - ring_colors)
    by_color = {color: _components(grid == color, connectivity=8) for color in colors}
    body_candidates = [color for color, comps in by_color.items() if len(comps) >= 2 and all(int(comp.shape[0]) > 1 for comp in comps)]
    assert len(body_candidates) == 1, "expected exactly one repeated body color"
    body_color = int(body_candidates[0])
    body_components = by_color[body_color]

    markers = []
    for color in colors:
        if color == body_color:
            continue
        comps = by_color[color]
        assert len(comps) == 1 and int(comps[0].shape[0]) == 1, "marker colors must be singleton pixels"
        rr, cc = map(int, comps[0][0])
        markers.append({"color": int(color), "coord": (rr, cc)})

    sprites = []
    used_markers = set()
    for comp in body_components:
        body_cells = {(int(r), int(c)) for r, c in comp}
        attached = []
        for marker in markers:
            if marker["coord"] in used_markers:
                continue
            mr, mc = marker["coord"]
            if any(abs(br - mr) <= 1 and abs(bc - mc) <= 1 for br, bc in body_cells):
                attached.append(marker)
        assert len(attached) == 1, "each body component must own one marker"
        marker = attached[0]
        used_markers.add(marker["coord"])

        mr, mc = marker["coord"]
        bias_r = float(comp[:, 0].mean()) - mr
        bias_c = float(comp[:, 1].mean()) - mc
        sr, sc = _sign(bias_r), _sign(bias_c)
        assert sr != 0 and sc != 0, "body centroid must encode a diagonal bias"
        direction = (-sr, -sc)
        assert direction in BODY_OFFSETS_BY_RAY
        sprites.append(
            {
                "marker_color": marker["color"],
                "marker": marker["coord"],
                "body_color": body_color,
                "body_coords": comp.copy(),
                "direction": direction,
            }
        )

    assert len(sprites) == len(markers)
    return sprites


def _trace_ray(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    ring_lookup: dict[tuple[int, int], int] | None = None,
) -> list[dict] | tuple[int, int]:
    h, w = map(int, shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    reflections = 0
    seen = set()
    path = []

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray cycled before hitting a ring"
        seen.add(state)

        nr, nc = r + dr, c + dc
        bounce_r = nr < 0 or nr >= h
        bounce_c = nc < 0 or nc >= w
        assert not (bounce_r and bounce_c), "ray entered a corner"
        if bounce_r:
            dr = -dr
            nr = r + dr
            reflections += 1
        if bounce_c:
            dc = -dc
            nc = c + dc
            reflections += 1

        r, c = nr, nc
        step = {"pos": (r, c), "dir": (dr, dc), "reflections": reflections}
        path.append(step)
        if ring_lookup is not None and (r, c) in ring_lookup:
            return int(ring_lookup[(r, c)]), reflections
        if ring_lookup is None and len(path) > 4 * h * w:
            raise AssertionError("unexpectedly long reflection cycle")
        if ring_lookup is None and (r, c, dr, dc) in seen:
            return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the first 3x3 ring hit by each sprite's invisible diagonal ray after edge reflections.

    Input structure:
    - A dominant background surrounds several same-body-color directional sprites, each with a unique singleton marker color.
    - The body pixels of a sprite sit behind its marker, so their centroid bias points opposite the emitted diagonal direction.
    - Several distinct-color 3x3 hollow square rings are scattered elsewhere on the grid.

    Rule:
    - From each marker pixel, advance one diagonal cell at a time; when the next step would leave the grid, mirror-reflect the perpendicular motion component at that edge.
    - The ray itself is never drawn. As soon as it first enters any ring border pixel, stop that ray and recolor the entire ring to the marker color.
    - Sprites and untouched rings remain unchanged.

    Key insight:
    - The answer is determined by mentally simulating an invisible billiard-style ray whose diagonal launch direction is encoded only by the sprite body's centroid bias.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rings = _detect_rings(g, bg)
    assert rings, "expected at least one ring"
    ring_lookup = {}
    for idx, ring in enumerate(rings):
        for cell in ring["cells"]:
            ring_lookup[cell] = idx

    sprites = _parse_sprites(g, bg, rings)
    out = g.copy()
    hit_rings = set()
    for sprite in sprites:
        ring_idx, _ = _trace_ray(g.shape, sprite["marker"], sprite["direction"], ring_lookup)
        hit_rings.add(ring_idx)
        for rr, cc in rings[ring_idx]["cells"]:
            out[rr, cc] = int(sprite["marker_color"])
    assert len(hit_rings) == len(sprites), "each ray must hit a different ring"
    return out


def _sprite_cells(marker: tuple[int, int], direction: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    mr, mc = marker
    body = [(mr + dr, mc + dc) for dr, dc in BODY_OFFSETS_BY_RAY[direction]]
    bbox = _bbox_from_coords(body + [marker])
    return body, bbox


def _bbox_clear(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_sprite(grid: np.ndarray, blocked: np.ndarray, marker: tuple[int, int], direction: tuple[int, int], body_color: int, marker_color: int) -> None:
    body, bbox = _sprite_cells(marker, direction)
    assert _bbox_clear(blocked, bbox, pad=1)
    for rr, cc in body:
        grid[rr, cc] = int(body_color)
        blocked[rr, cc] = True
    mr, mc = marker
    grid[mr, mc] = int(marker_color)
    blocked[mr, mc] = True


def _place_ring(grid: np.ndarray, blocked: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    bbox = (top, left, top + 2, left + 2)
    assert _bbox_clear(blocked, bbox, pad=1)
    cells = _ring_cells_from_tl(top, left)
    for rr, cc in cells:
        grid[rr, cc] = int(color)
        blocked[rr, cc] = True
    return cells


def _unreflected_prefix(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    out = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return out
        out.append((r, c))


def _choose_target_ring(
    shape: tuple[int, int],
    path: list[dict],
    min_reflections: int,
    blocked: np.ndarray,
    object_cells: set[tuple[int, int]],
    prior_prefix_cells: set[tuple[int, int]],
) -> tuple[int, int, int, list[tuple[int, int]]] | None:
    indices = list(range(4, len(path)))
    random.shuffle(indices)
    for idx in indices:
        step = path[idx]
        if int(step["reflections"]) < int(min_reflections):
            continue
        hit_r, hit_c = step["pos"]
        prefix = {tuple(item["pos"]) for item in path[:idx]}
        if prefix & object_cells:
            continue
        offsets = RING_BORDER_OFFSETS[:]
        random.shuffle(offsets)
        for off_r, off_c in offsets:
            top = hit_r - off_r
            left = hit_c - off_c
            if not (0 <= top and 0 <= left and top + 2 < shape[0] and left + 2 < shape[1]):
                continue
            bbox = (top, left, top + 2, left + 2)
            if not _bbox_clear(blocked, bbox, pad=1):
                continue
            ring_cells = set(_ring_cells_from_tl(top, left))
            if ring_cells & prefix:
                continue
            if ring_cells & prior_prefix_cells:
                continue
            return idx, top, left, sorted(ring_cells)
    return None


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _candidate_ring_spots(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    spots = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    random.shuffle(spots)
    return spots


def _find_decoy_ring(
    shape: tuple[int, int],
    blocked: np.ndarray,
    forbidden_cells: set[tuple[int, int]],
    sprite: dict,
    target_center: tuple[int, int],
) -> tuple[int, int] | None:
    marker = sprite["marker"]
    dr, dc = sprite["direction"]
    target_dist = _distance(marker, target_center)
    for top, left in _candidate_ring_spots(shape):
        bbox = (top, left, top + 2, left + 2)
        if not _bbox_clear(blocked, bbox, pad=1):
            continue
        cells = set(_ring_cells_from_tl(top, left))
        if cells & forbidden_cells:
            continue
        center = (top + 1, left + 1)
        if (center[0] - marker[0]) * dr < 0 or (center[1] - marker[1]) * dc < 0:
            continue
        if _distance(marker, center) >= target_dist:
            continue
        return top, left
    return None


def _find_any_extra_ring(
    shape: tuple[int, int],
    blocked: np.ndarray,
    forbidden_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for top, left in _candidate_ring_spots(shape):
        bbox = (top, left, top + 2, left + 2)
        if not _bbox_clear(blocked, bbox, pad=1):
            continue
        cells = set(_ring_cells_from_tl(top, left))
        if cells & forbidden_cells:
            continue
        return top, left
    return None


def _semantic_checks(grid: np.ndarray, expected_sprite_count: int, require_decoy: bool) -> bool:
    try:
        bg = _get_bg(grid)
        rings = _detect_rings(grid, bg)
        if len(rings) <= expected_sprite_count:
            return False
        sprites = _parse_sprites(grid, bg, rings)
        if len(sprites) != expected_sprite_count:
            return False
        ring_lookup = {cell: idx for idx, ring in enumerate(rings) for cell in ring["cells"]}

        hits = []
        reflections = []
        target_centers = []
        for sprite in sprites:
            ring_idx, refl = _trace_ray(grid.shape, sprite["marker"], sprite["direction"], ring_lookup)
            hits.append(ring_idx)
            reflections.append(refl)
            target_centers.append(rings[ring_idx]["center"])

        if len(set(hits)) != len(hits):
            return False
        if max(reflections) == 0:
            return False
        if len(rings) - len(set(hits)) < 1:
            return False

        out = main(grid)
        for sprite, ring_idx in zip(sprites, hits):
            ring = rings[ring_idx]
            for rr, cc in ring["cells"]:
                if int(out[rr, cc]) != int(sprite["marker_color"]):
                    return False
            if int(ring["color"]) == int(sprite["marker_color"]):
                return False
        untouched = set(range(len(rings))) - set(hits)
        for ring_idx in untouched:
            for rr, cc in rings[ring_idx]["cells"]:
                if int(out[rr, cc]) != int(grid[rr, cc]):
                    return False

        if require_decoy:
            closer = False
            for sprite, ring_idx in zip(sprites, hits):
                target_center = rings[ring_idx]["center"]
                target_dist = _distance(sprite["marker"], target_center)
                for other_idx, ring in enumerate(rings):
                    if other_idx == ring_idx:
                        continue
                    if _distance(sprite["marker"], ring["center"]) < target_dist:
                        closer = True
                        break
                if closer:
                    break
            if not closer:
                return False

            reflected_sprite_ok = False
            for sprite, refl in zip(sprites, reflections):
                if refl == 0:
                    continue
                prefix = _unreflected_prefix(grid.shape, sprite["marker"], sprite["direction"])
                target_cells = set(rings[hits[sprites.index(sprite)]]["cells"])
                if not any(cell in target_cells for cell in prefix):
                    reflected_sprite_ok = True
                    break
            if not reflected_sprite_ok:
                return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_sprites = random.choice(SPRITE_COUNT_OPTIONS)
    extra_rings = random.choice(EXTRA_RING_OPTIONS)
    total_colors_needed = 2 + n_sprites + (n_sprites + extra_rings)
    if total_colors_needed > 10:
        return None

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette.pop()
    body_color = palette.pop()
    marker_colors = [palette.pop() for _ in range(n_sprites)]
    ring_colors = [palette.pop() for _ in range(n_sprites + extra_rings)]

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    directions = random.sample(DIAGONALS, n_sprites)
    reflect_idx = random.randrange(n_sprites)
    min_reflections = [0] * n_sprites
    min_reflections[reflect_idx] = 2 if random.random() < 0.35 else 1
    for i in range(n_sprites):
        if i != reflect_idx and random.random() < 0.3:
            min_reflections[i] = 1

    sprites = []
    candidate_markers = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
    random.shuffle(candidate_markers)
    for idx, direction in enumerate(directions):
        placed = False
        for marker in candidate_markers:
            body, bbox = _sprite_cells(marker, direction)
            if not (0 <= bbox[0] and 0 <= bbox[1] and bbox[2] < h and bbox[3] < w):
                continue
            if not _bbox_clear(blocked, bbox, pad=1):
                continue
            _place_sprite(grid, blocked, marker, direction, body_color, marker_colors[idx])
            sprites.append(
                {
                    "marker": marker,
                    "marker_color": marker_colors[idx],
                    "direction": direction,
                    "min_reflections": min_reflections[idx],
                }
            )
            placed = True
            break
        if not placed:
            return None

    traces = []
    for sprite in sprites:
        try:
            trace = _trace_ray((h, w), sprite["marker"], sprite["direction"])
        except AssertionError:
            return None
        if len(trace) < 10:
            return None
        traces.append(trace)

    order = sorted(range(n_sprites), key=lambda i: (-sprites[i]["min_reflections"], len(traces[i])))
    all_sprite_cells = {(r, c) for r in range(h) for c in range(w) if grid[r, c] != bg}
    prior_prefix_cells: set[tuple[int, int]] = set()
    target_steps: dict[int, int] = {}
    target_ring_cells: dict[int, set[tuple[int, int]]] = {}
    ring_centers = {}

    for color_idx, sprite_idx in enumerate(order):
        choice = _choose_target_ring(
            (h, w),
            traces[sprite_idx],
            sprites[sprite_idx]["min_reflections"],
            blocked,
            all_sprite_cells,
            prior_prefix_cells,
        )
        if choice is None:
            return None
        step_idx, top, left, ring_cells = choice
        _place_ring(grid, blocked, top, left, ring_colors[color_idx])
        ring_cells_set = set(ring_cells)
        all_sprite_cells |= ring_cells_set
        prior_prefix_cells |= {tuple(item["pos"]) for item in traces[sprite_idx][:step_idx]}
        target_steps[sprite_idx] = step_idx
        target_ring_cells[sprite_idx] = ring_cells_set
        ring_centers[sprite_idx] = (top + 1, left + 1)

    prefix_forbidden = set()
    for sprite_idx, trace in enumerate(traces):
        if sprite_idx not in target_steps:
            return None
        step_idx = target_steps[sprite_idx]
        prefix = [tuple(item["pos"]) for item in trace[:step_idx]]
        if set(prefix) & ({(r, c) for r in range(h) for c in range(w) if grid[r, c] != bg} - target_ring_cells[sprite_idx]):
            return None
        prefix_forbidden |= set(prefix)
        prefix_forbidden |= target_ring_cells[sprite_idx]

    if reflect_idx not in ring_centers:
        return None
    decoy_pos = _find_decoy_ring((h, w), blocked, prefix_forbidden, sprites[reflect_idx], ring_centers[reflect_idx])
    if decoy_pos is None:
        return None
    _place_ring(grid, blocked, decoy_pos[0], decoy_pos[1], ring_colors[n_sprites])

    for extra_idx in range(1, extra_rings):
        extra_pos = _find_any_extra_ring((h, w), blocked, prefix_forbidden)
        if extra_pos is None:
            return None
        _place_ring(grid, blocked, extra_pos[0], extra_pos[1], ring_colors[n_sprites + extra_idx])

    if not _semantic_checks(grid, n_sprites, require_decoy=True):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for invisible diagonal reflected rays")
