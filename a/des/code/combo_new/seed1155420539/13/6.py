from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a marker-colored outer halo around each closed object, with thickness set by the number of interior marker pixels.

    Input structure:
    - The background is black.
    - The grid contains several separated closed objects; each object is one
      4-connected non-black component whose main body uses one color.
    - Some objects contain interior marker pixels of one shared marker color,
      different from the host object's body color and located strictly inside it.
    - Different objects are separated by black space, so each object-plus-markers
      bundle is its own non-black component.

    Rule:
    - Identify every non-black 4-connected component as one object.
    - Within each object, treat the outer contour color as the body color; the
      remaining pixels are marker pixels, whose count is N and whose color is the
      halo color.
    - Draw halos in synchronous layers. Layer 1 claims black pixels 8-neighboring
      the object; layer k+1 claims black pixels 8-neighboring that object's
      successful layer-k halo pixels. Only objects with at least k marker pixels
      may emit layer k.
    - If multiple objects try to claim the same black pixel in the same layer,
      that pixel stays black and does not become halo for any object. Earlier
      layers always beat later layers because only still-black pixels can be
      claimed.
    - Keep the original body and marker pixels unchanged.

    Key insight:
    - The hidden instruction is inside each object: marker color chooses the halo
      color, and marker count chooses how many outward 8-neighbor layers are
      allowed to grow.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    return _simulate_halos(g, objects)


def generate_input() -> np.ndarray:
    rng = random.Random()
    modes = ("none", "early", "tie")
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng, rng.choice(modes))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid halo puzzle input")


def _extract_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == BG:
                continue
            comp = _collect_component(grid, visited, r, c)
            colors = Counter(int(grid[rr, cc]) for rr, cc in comp)
            boundary_counts = Counter()
            comp_set = set(comp)
            for rr, cc in comp:
                col = int(grid[rr, cc])
                if any((nr, nc) not in comp_set for nr, nc in _neighbors4(rr, cc, h, w)):
                    boundary_counts[col] += 1
            body_color = max(colors, key=lambda col: (boundary_counts[col], colors[col], -col))
            marker_pixels = [(rr, cc) for rr, cc in comp if int(grid[rr, cc]) != body_color]
            marker_counts = Counter(int(grid[rr, cc]) for rr, cc in marker_pixels)
            marker_color = None if not marker_counts else marker_counts.most_common(1)[0][0]
            objects.append(
                {
                    "cells": set(comp),
                    "body_color": int(body_color),
                    "marker_color": None if marker_color is None else int(marker_color),
                    "marker_count": len(marker_pixels),
                }
            )

    return objects


def _collect_component(
    grid: np.ndarray, visited: np.ndarray, start_r: int, start_c: int
) -> list[tuple[int, int]]:
    h, w = grid.shape
    q = deque([(start_r, start_c)])
    visited[start_r, start_c] = True
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if visited[nr, nc] or int(grid[nr, nc]) == BG:
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
    return comp


def _simulate_halos(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    max_layers = max((obj["marker_count"] for obj in objects), default=0)
    frontiers = [set(obj["cells"]) for obj in objects]

    for layer in range(1, max_layers + 1):
        base_black = out == BG
        proposals: dict[tuple[int, int], int] = {}
        next_frontiers = [set() for _ in objects]

        for obj_idx, obj in enumerate(objects):
            if obj["marker_count"] < layer or not frontiers[obj_idx]:
                continue
            for r, c in frontiers[obj_idx]:
                for nr, nc in _neighbors8(r, c, out.shape[0], out.shape[1]):
                    if not base_black[nr, nc]:
                        continue
                    prev = proposals.get((nr, nc))
                    if prev is None:
                        proposals[(nr, nc)] = obj_idx
                    elif prev != obj_idx:
                        proposals[(nr, nc)] = -1

        for (r, c), owner in proposals.items():
            if owner < 0:
                continue
            marker_color = objects[owner]["marker_color"]
            if marker_color is None:
                continue
            out[r, c] = int(marker_color)
            next_frontiers[owner].add((r, c))

        frontiers = next_frontiers

    return out


def _gen_attempt(rng: random.Random, mode: str) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    n_objects = rng.choice([2, 3, 3, 4])
    colors = rng.sample(PALETTE, n_objects + 1)
    body_colors = colors[:n_objects]
    marker_color = colors[-1]

    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    specs: list[dict] = []

    pair_orientation = rng.choice(["horizontal", "vertical"])
    pair_h = rng.randint(4, 7)
    pair_w = rng.randint(4, 7)
    pair_mask = np.ones((pair_h, pair_w), dtype=bool)

    if mode == "tie":
        gap = 3
        pair_count = rng.randint(2, 4)
        pair_counts = [pair_count, pair_count]
    elif mode == "early":
        gap = 4
        high_count = 4
        low_count = rng.randint(1, 2)
        pair_counts = [low_count, high_count]
        if rng.random() < 0.5:
            pair_counts.reverse()
    else:
        gap = rng.randint(9, 12)
        pair_counts = [rng.randint(1, 4), rng.randint(1, 4)]

    pair_positions = _pair_positions(h, w, pair_mask.shape, gap, pair_orientation, rng)
    if pair_positions is None:
        return None

    for idx in range(2):
        patch = _make_patch(pair_mask, body_colors[idx], marker_color, pair_counts[idx], rng)
        if patch is None:
            return None
        top, left = pair_positions[idx]
        coords = _global_coords(patch != BG, top, left)
        if not _can_place(coords, occupied, h, w):
            return None
        _stamp_patch(grid, patch, top, left)
        occupied.update(coords)
        specs.append(
            {
                "top": top,
                "left": left,
                "patch": patch,
                "mask": patch != BG,
                "marker_count": pair_counts[idx],
            }
        )

    for idx in range(2, n_objects):
        placed = False
        for _ in range(80):
            mask = _random_mask(rng)
            max_markers = min(4, len(_interior_cells(mask)))
            if max_markers == 0:
                continue
            marker_count = rng.randint(0, max_markers)
            if idx == 2 and n_objects >= 3 and rng.random() < 0.55:
                marker_count = 0
            patch = _make_patch(mask, body_colors[idx], marker_color, marker_count, rng)
            if patch is None:
                continue
            pos = _random_position(h, w, patch.shape, rng)
            if pos is None:
                continue
            top, left = pos
            coords = _global_coords(patch != BG, top, left)
            if not _can_place(coords, occupied, h, w):
                continue
            _stamp_patch(grid, patch, top, left)
            occupied.update(coords)
            specs.append(
                {
                    "top": top,
                    "left": left,
                    "patch": patch,
                    "mask": patch != BG,
                    "marker_count": marker_count,
                }
            )
            placed = True
            break
        if not placed:
            return None

    objects = _extract_objects(grid)
    if len(objects) != n_objects:
        return None
    if max(obj["marker_count"] for obj in objects) == 0:
        return None

    profile = _overlap_profile(grid, objects)
    if mode == "none" and (profile["early"] or profile["tie"]):
        return None
    if mode == "early" and (not profile["early"] or profile["tie"]):
        return None
    if mode == "tie" and not profile["tie"]:
        return None

    if np.array_equal(main(grid), grid):
        return None
    return grid


def _pair_positions(
    h: int,
    w: int,
    shape: tuple[int, int],
    gap: int,
    orientation: str,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    oh, ow = shape
    if orientation == "horizontal":
        total_w = 2 * ow + gap
        if total_w > w - 2 or oh > h - 2:
            return None
        top = rng.randint(1, h - oh - 1)
        left = rng.randint(1, w - total_w - 1)
        return [(top, left), (top, left + ow + gap)]
    total_h = 2 * oh + gap
    if total_h > h - 2 or ow > w - 2:
        return None
    top = rng.randint(1, h - total_h - 1)
    left = rng.randint(1, w - ow - 1)
    return [(top, left), (top + oh + gap, left)]


def _make_patch(
    mask: np.ndarray,
    body_color: int,
    marker_color: int,
    marker_count: int,
    rng: random.Random,
) -> np.ndarray | None:
    interior = _interior_cells(mask)
    if marker_count > len(interior):
        return None

    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(body_color)
    if marker_count > 0:
        for r, c in rng.sample(interior, marker_count):
            patch[r, c] = int(marker_color)
    return patch


def _random_mask(rng: random.Random) -> np.ndarray:
    kind = rng.choice(["rect", "rect", "l", "t", "plus", "blob"])
    if kind == "rect":
        h = rng.randint(4, 8)
        w = rng.randint(4, 8)
        mask = np.ones((h, w), dtype=bool)
    elif kind == "l":
        h = rng.randint(6, 9)
        w = rng.randint(6, 9)
        t = rng.randint(3, min(h, w) - 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :t] = True
        mask[h - t :, :] = True
    elif kind == "t":
        h = rng.randint(7, 9)
        w = rng.randint(7, 9)
        t = 3
        stem = (w - t) // 2
        mask = np.zeros((h, w), dtype=bool)
        mask[:t, :] = True
        mask[:, stem : stem + t] = True
    elif kind == "plus":
        h = rng.choice([7, 9])
        w = rng.choice([7, 9])
        t = 3
        vr = (w - t) // 2
        hr = (h - t) // 2
        mask = np.zeros((h, w), dtype=bool)
        mask[:, vr : vr + t] = True
        mask[hr : hr + t, :] = True
    else:
        h = rng.randint(6, 8)
        w = rng.randint(6, 8)
        cut_h = rng.randint(2, h - 3)
        cut_w = rng.randint(2, w - 3)
        mask = np.ones((h, w), dtype=bool)
        corner = rng.choice(["tr", "tl"])
        if corner == "tr":
            mask[:cut_h, w - cut_w :] = False
        else:
            mask[:cut_h, :cut_w] = False

    k = rng.randrange(4)
    mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            ok = True
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    ok = False
                    break
            if ok:
                cells.append((r, c))
    return cells


def _random_position(
    h: int, w: int, shape: tuple[int, int], rng: random.Random
) -> tuple[int, int] | None:
    oh, ow = shape
    if oh > h or ow > w:
        return None
    return (rng.randint(0, h - oh), rng.randint(0, w - ow))


def _global_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        coords.append((top + int(r), left + int(c)))
    return coords


def _can_place(
    coords: Iterable[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    coords = list(coords)
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr, cc in occupied:
            if max(abs(r - rr), abs(c - cc)) <= 2:
                return False
    return True


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    region[patch != BG] = patch[patch != BG]


def _overlap_profile(grid: np.ndarray, objects: list[dict]) -> dict[str, bool]:
    reaches = [_reach_layers(grid, obj["cells"], obj["marker_count"]) for obj in objects]
    by_pixel: dict[tuple[int, int], list[int]] = {}
    for idx, reach in enumerate(reaches):
        for pix, layer in reach.items():
            by_pixel.setdefault(pix, []).append((layer, idx))

    early = False
    tie = False
    for entries in by_pixel.values():
        if len(entries) < 2:
            continue
        min_layer = min(layer for layer, _ in entries)
        if sum(layer == min_layer for layer, _ in entries) >= 2:
            tie = True
        else:
            early = True
    return {"early": early, "tie": tie}


def _reach_layers(
    grid: np.ndarray, start_cells: set[tuple[int, int]], max_layers: int
) -> dict[tuple[int, int], int]:
    if max_layers <= 0:
        return {}

    h, w = grid.shape
    visited = set(start_cells)
    frontier = set(start_cells)
    layers: dict[tuple[int, int], int] = {}

    for layer in range(1, max_layers + 1):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in _neighbors8(r, c, h, w):
                if (nr, nc) in visited or int(grid[nr, nc]) != BG:
                    continue
                visited.add((nr, nc))
                layers[(nr, nc)] = layer
                nxt.add((nr, nc))
        frontier = nxt
        if not frontier:
            break
    return layers


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _neighbors8(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc
