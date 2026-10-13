from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GRID_H_RANGE = (17, 23)
GRID_W_RANGE = (17, 23)
FRAME_SIZES = [(3, 3), (3, 4), (3, 5), (4, 4), (4, 5)]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return comps


def _emit_ray_mask(sources: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for sr, sc in sources:
        sr = int(sr)
        sc = int(sc)
        for dr, dc in DIAG4:
            r, c = sr, sc
            while 0 <= r < h and 0 <= c < w:
                mask[r, c] = True
                r += dr
                c += dc
    return mask


def _expand4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _component_has_hole(coords: np.ndarray) -> bool:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
    mask[coords[:, 0] - r0 + 1, coords[:, 1] - c0 + 1] = True

    seen = np.zeros_like(mask, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and not seen[nr, nc] and not mask[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any((~mask) & (~seen)))


def _analyze_blue_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray, list[dict]]:
    comps = _find_components_of_color(grid, BLUE)
    sources = np.argwhere(grid == RED)
    ray_mask = _emit_ray_mask(sources, grid.shape)
    ray_touch = _expand4(ray_mask)
    info: list[dict] = []
    for comp in comps:
        coords = comp["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        overlap = bool(ray_mask[rows, cols].any())
        contact = bool(ray_touch[rows, cols].any())
        hole = _component_has_hole(coords)
        info.append(
            {
                "coords": coords,
                "area": comp["area"],
                "contact": contact,
                "overlap": overlap,
                "hole": hole,
                "select": contact and hole,
            }
        )
    return comps, ray_mask, ray_touch, info


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor only the hollow blue components that are contacted by diagonal rays from red source pixels.

    Input structure:
    - A black background contains 4-connected blue components and one or more isolated red source pixels near the border.
    - Blue components may be hollow closed outlines or open non-hollow outlines.
    - The red pixels are emitters; the input does not yet contain the full rays.

    Rule:
    - From every red source pixel, extend red rays in all four diagonal directions until they leave the grid.
    - For each original blue component, test two independent predicates: whether the component overlaps a ray or is 4-neighbor adjacent to a ray, and whether the component encloses at least one black hole.
    - Recolor a blue component to green iff both predicates are true, then draw the red rays on top so every ray-covered pixel remains red.

    Key insight:
    - The deciding signal is a logical AND between a geometric contact test against generated diagonal rays and a topological hole test on each component in isolation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _, ray_mask, ray_touch, info = _analyze_blue_components(g)
    out = g.copy()
    for comp in info:
        if comp["select"]:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = GREEN
    out[ray_mask] = RED
    return out


def _frame_sprite(height: int, width: int) -> np.ndarray:
    sprite = np.zeros((height, width), dtype=int)
    sprite[0, :] = 1
    sprite[-1, :] = 1
    sprite[:, 0] = 1
    sprite[:, -1] = 1
    return sprite


def _random_hole_sprite(rng: np.random.Generator) -> np.ndarray:
    h, w = FRAME_SIZES[int(rng.integers(0, len(FRAME_SIZES)))]
    return _frame_sprite(h, w)


def _random_open_sprite(rng: np.random.Generator) -> np.ndarray:
    h, w = FRAME_SIZES[int(rng.integers(0, len(FRAME_SIZES)))]
    sprite = _frame_sprite(h, w)
    boundary = [(r, c) for r in range(h) for c in range(w) if sprite[r, c] == 1]
    while True:
        rr, cc = boundary[int(rng.integers(0, len(boundary)))]
        sprite2 = sprite.copy()
        sprite2[rr, cc] = 0
        coords = np.argwhere(sprite2 == 1)
        if coords.shape[0] < 6 or coords.shape[0] > 15:
            continue
        if len(_find_components_of_color(sprite2, 1)) != 1:
            continue
        if _component_has_hole(coords):
            continue
        return sprite2


def _random_transform(sprite: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(sprite, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    return out.copy()


def _pick_sources(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int]]:
    edge_cells: list[tuple[int, int]] = []
    edge_cells.extend((0, c) for c in range(1, w - 1))
    edge_cells.extend((h - 1, c) for c in range(1, w - 1))
    edge_cells.extend((r, 0) for r in range(1, h - 1))
    edge_cells.extend((r, w - 1) for r in range(1, h - 1))
    rng.shuffle(edge_cells)

    target = 1 if rng.random() < 0.58 else 2
    chosen: list[tuple[int, int]] = []
    for cell in edge_cells:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) >= 3 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == target:
                break
    return chosen


def _source_forbidden(shape: tuple[int, int], sources: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for sr, sc in sources:
        for r in range(max(0, sr - 2), min(h, sr + 3)):
            for c in range(max(0, sc - 2), min(w, sc + 3)):
                mask[r, c] = True
    return mask


def _placement_candidates(
    sprite: np.ndarray,
    grid_shape: tuple[int, int],
    blocked: np.ndarray,
    source_blocked: np.ndarray,
    ray_mask: np.ndarray,
    ray_touch: np.ndarray,
    *,
    want_touch: bool,
    touch_mode: str | None,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    sh, sw = sprite.shape
    rel = np.argwhere(sprite == 1)
    area = int(rel.shape[0])
    overlap_cap = min(max(2, area // 2), 4)
    candidates: list[tuple[int, int]] = []

    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            rows = rel[:, 0] + r0
            cols = rel[:, 1] + c0
            if blocked[rows, cols].any() or source_blocked[rows, cols].any():
                continue

            overlap_count = int(ray_mask[rows, cols].sum())
            touched = bool(ray_touch[rows, cols].any())
            if want_touch:
                if not touched:
                    continue
                if touch_mode == "overlap":
                    if overlap_count == 0 or overlap_count > overlap_cap:
                        continue
                elif touch_mode == "adjacent":
                    if overlap_count != 0:
                        continue
                else:
                    raise AssertionError("touch mode required for touched components")
            else:
                if touched:
                    continue

            candidates.append((r0, c0))
    return candidates


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                blocked[nr, nc] = True


def _render_component(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> np.ndarray:
    rel = np.argwhere(sprite == 1)
    abs_coords = rel + np.array([[top, left]])
    grid[abs_coords[:, 0], abs_coords[:, 1]] = BLUE
    return abs_coords


def _component_specs(rng: np.random.Generator) -> list[dict]:
    n_components = int(rng.integers(4, 7))
    touch_modes = ["overlap", "adjacent"]
    rng.shuffle(touch_modes)
    specs = [
        {"hole": True, "touch": True, "mode": touch_modes[0]},
        {"hole": False, "touch": True, "mode": touch_modes[1]},
        {"hole": True, "touch": False, "mode": None},
        {"hole": False, "touch": False, "mode": None},
    ]
    extra_pool = [
        {"hole": True, "touch": True},
        {"hole": False, "touch": True},
        {"hole": True, "touch": False},
        {"hole": False, "touch": False},
    ]
    while len(specs) < n_components:
        base = extra_pool[int(rng.integers(0, len(extra_pool)))]
        spec = dict(base)
        spec["mode"] = ["overlap", "adjacent"][int(rng.integers(0, 2))] if spec["touch"] else None
        specs.append(spec)

    def priority(spec: dict) -> tuple[int, int]:
        if spec["touch"] and spec["mode"] == "overlap":
            return (0, 0)
        if spec["touch"] and spec["mode"] == "adjacent":
            return (0, 1)
        return (1, 0 if spec["hole"] else 1)

    rng.shuffle(specs)
    specs.sort(key=priority)
    return specs


def _expected_output_from_info(grid: np.ndarray, ray_mask: np.ndarray, info: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for comp in info:
        if comp["select"]:
            coords = comp["coords"]
            expected[coords[:, 0], coords[:, 1]] = GREEN
    expected[ray_mask] = RED
    return expected


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    blue_comps, ray_mask, _, info = _analyze_blue_components(grid)
    red_comps = _find_components_of_color(grid, RED)
    if not (4 <= len(blue_comps) <= 6):
        return False
    if len(red_comps) not in (1, 2):
        return False
    if any(comp["area"] != 1 for comp in red_comps):
        return False
    if any(not (r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)) for comp in red_comps for r, c in comp["coords"]):
        return False

    category_counts = {
        (True, True): 0,
        (True, False): 0,
        (False, True): 0,
        (False, False): 0,
    }
    for comp in info:
        category_counts[(comp["contact"], comp["hole"])] += 1
    if category_counts[(True, True)] < 1:
        return False
    if category_counts[(True, False)] < 1:
        return False
    if category_counts[(False, True)] < 1:
        return False
    if category_counts[(False, False)] < 1:
        return False

    out = main(grid)
    expected = _expected_output_from_info(grid, ray_mask, info)
    if not np.array_equal(out, expected):
        return False
    if not np.all(out[ray_mask] == RED):
        return False

    for comp in info:
        coords = comp["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        non_ray = ~ray_mask[rows, cols]
        if not np.any(non_ray):
            return False
        visible = out[rows[non_ray], cols[non_ray]]
        target = GREEN if comp["select"] else BLUE
        if not np.all(visible == target):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    sources = _pick_sources(rng, h, w)
    if len(sources) not in (1, 2):
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in sources:
        grid[r, c] = RED

    ray_mask = _emit_ray_mask(np.asarray(sources, dtype=int), (h, w))
    ray_touch = _expand4(ray_mask)
    blocked = np.zeros((h, w), dtype=bool)
    source_blocked = _source_forbidden((h, w), sources)
    specs = _component_specs(rng)

    for spec in specs:
        placed = False
        for _ in range(40):
            base = _random_hole_sprite(rng) if spec["hole"] else _random_open_sprite(rng)
            sprite = _random_transform(base, rng)
            coords = np.argwhere(sprite == 1)
            if not (6 <= coords.shape[0] <= 15):
                continue
            if _component_has_hole(coords) != bool(spec["hole"]):
                continue

            candidates = _placement_candidates(
                sprite,
                (h, w),
                blocked,
                source_blocked,
                ray_mask,
                ray_touch,
                want_touch=bool(spec["touch"]),
                touch_mode=spec["mode"],
            )
            if not candidates:
                continue

            top, left = candidates[int(rng.integers(0, len(candidates)))]
            abs_coords = _render_component(grid, sprite, top, left)
            _mark_blocked(blocked, abs_coords)
            placed = True
            break
        if not placed:
            return None

    return grid if _validate_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 2")
