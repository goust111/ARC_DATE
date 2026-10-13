from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _has_enclosed_background(mask: np.ndarray) -> bool:
    if np.all(mask):
        return False
    h, w = mask.shape
    for comp in _components(~mask):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            return True
    return False


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _extract_regions(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    regions: list[dict[str, object]] = []
    for comp in _components(grid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        adjacency = Counter()
        boundary_cells: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                color = int(grid[nr, nc])
                if color == bg:
                    continue
                adjacency[color] += 1
                boundary_cells[color].add((nr, nc))
        if len(adjacency) != 2:
            continue
        wall_color, _ = adjacency.most_common(1)[0]
        seed_colors = [color for color in adjacency if color != wall_color]
        if len(seed_colors) != 1:
            continue
        seed_color = int(seed_colors[0])
        seed_cells = sorted(boundary_cells[seed_color])
        if len(seed_cells) != 1:
            continue
        regions.append(
            {
                "cells": comp,
                "wall_color": int(wall_color),
                "seed_color": seed_color,
                "seed": seed_cells[0],
            }
        )
    assert regions, "expected at least one enclosed region"
    wall_colors = {int(region["wall_color"]) for region in regions}
    assert len(wall_colors) == 1, "all regions must share one wall color"
    seed_colors = [int(region["seed_color"]) for region in regions]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be unique"
    return regions


def _extract_templates(
    grid: np.ndarray, regions: list[dict[str, object]]
) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for region in regions:
        seed_color = int(region["seed_color"])
        seed_cell = tuple(region["seed"])
        comps = _components(grid == seed_color)
        candidates = [comp for comp in comps if seed_cell not in comp]
        assert candidates, "each seed color must have a separate template"
        template_cells = max(candidates, key=len)
        assert len(template_cells) > 1, "template must be larger than the one-pixel seed"
        r0, c0, r1, c1 = _bbox(template_cells)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in template_cells:
            mask[r - r0, c - c0] = True
        templates[seed_color] = mask
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed region with the periodic template keyed by its seed color.

    The seed color identifies which edge template to use, while the seed position
    sets the tiling phase by aligning the template's top-left corner to the seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    regions = _extract_regions(g, bg)
    templates = _extract_templates(g, regions)

    out = g.copy()
    for region in regions:
        color = int(region["seed_color"])
        seed_r, seed_c = map(int, region["seed"])
        template = templates[color]
        th, tw = template.shape
        for r, c in region["cells"]:
            if template[(r - seed_r) % th, (c - seed_c) % tw]:
                out[r, c] = color
    return out


# --------------- generator ---------------

BG = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
N_REGIONS_RANGE = (1, 4)
TEMPLATE_EDGE_CHOICES = ("top", "bottom", "left", "right")
TEMPLATE_GAP_RANGE = (1, 2)

ROOM_SHAPES = ("rect", "L", "T", "plus")
ROOM_PLACEMENT_RETRIES = 200
GEN_ATTEMPTS = 400


def _random_connected_template(rng: random.Random) -> np.ndarray | None:
    for _ in range(100):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        max_area = max(2, min(h * w - 1, 6))
        area = rng.randint(2, max_area)
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), h, w):
                    if not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(frontier)
            mask[nr, nc] = True
        if int(mask.sum()) != area:
            continue
        mask = _crop_mask(mask)
        if _has_enclosed_background(mask):
            continue
        return mask
    return None


def _room_shape(kind: str, rng: random.Random) -> np.ndarray:
    if kind == "rect":
        return np.ones((rng.randint(2, 6), rng.randint(2, 6)), dtype=bool)

    if kind == "L":
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        thickness = rng.randint(1, min(2, h - 1, w - 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :thickness] = True
        mask[-thickness:, :] = True
        return np.rot90(mask, rng.randrange(4))

    if kind == "T":
        h = rng.randint(4, 7)
        w = rng.randint(3, 7)
        bar = rng.randint(1, min(2, h - 1))
        stem_w = rng.randint(1, min(2, w))
        if w - stem_w >= 2:
            stem_c = rng.randint(1, w - stem_w - 1)
        else:
            stem_c = rng.randint(0, w - stem_w)
        mask = np.zeros((h, w), dtype=bool)
        mask[:bar, :] = True
        mask[:, stem_c : stem_c + stem_w] = True
        return np.rot90(mask, rng.randrange(4))

    assert kind == "plus"
    h = rng.randint(3, 7)
    w = rng.randint(3, 7)
    band_h = rng.randint(1, min(2, h))
    band_w = rng.randint(1, min(2, w))
    band_r = rng.randint(0, h - band_h)
    band_c = rng.randint(0, w - band_w)
    mask = np.zeros((h, w), dtype=bool)
    mask[band_r : band_r + band_h, :] = True
    mask[:, band_c : band_c + band_w] = True
    return mask


def _build_room_patch(
    interior: np.ndarray, wall_color: int, seed_color: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    h, w = interior.shape
    full_interior = np.zeros((h + 2, w + 2), dtype=bool)
    full_interior[1 : h + 1, 1 : w + 1] = interior

    wall = np.zeros_like(full_interior)
    for r, c in np.argwhere(full_interior):
        for nr, nc in _neighbors4(int(r), int(c), h + 2, w + 2):
            if not full_interior[nr, nc]:
                wall[nr, nc] = True

    preferred_seed_cells = []
    for r, c in np.argwhere(wall):
        adj = 0
        for nr, nc in _neighbors4(int(r), int(c), h + 2, w + 2):
            if full_interior[nr, nc]:
                adj += 1
        if adj == 1:
            preferred_seed_cells.append((int(r), int(c)))
    seed = rng.choice(preferred_seed_cells or [tuple(map(int, rc)) for rc in np.argwhere(wall)])

    patch = np.full((h + 2, w + 2), BG, dtype=int)
    patch[wall] = wall_color
    patch[seed] = seed_color
    return patch, full_interior, seed


def _mark_occupied(occ: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> None:
    H, W = occ.shape
    occ[
        max(0, r0 - padding) : min(H, r0 + h + padding),
        max(0, c0 - padding) : min(W, c0 + w + padding),
    ] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = color


def _place_templates(
    grid: np.ndarray,
    occ: np.ndarray,
    templates: dict[int, np.ndarray],
    side: str,
    rng: random.Random,
) -> bool:
    items = list(templates.items())
    rng.shuffle(items)
    gaps = [rng.randint(*TEMPLATE_GAP_RANGE) for _ in range(max(0, len(items) - 1))]
    H, W = grid.shape

    if side in ("top", "bottom"):
        total = sum(mask.shape[1] for _, mask in items) + sum(gaps)
        if total > W:
            return False
        c = rng.randint(0, W - total)
        row_positions = []
        for idx, (_, mask) in enumerate(items):
            row_positions.append(c)
            c += mask.shape[1]
            if idx < len(gaps):
                c += gaps[idx]
        for (color, mask), c0 in zip(items, row_positions):
            r0 = 0 if side == "top" else H - mask.shape[0]
            _paint_mask(grid, mask, color, r0, c0)
            _mark_occupied(occ, r0, c0, mask.shape[0], mask.shape[1])
        return True

    total = sum(mask.shape[0] for _, mask in items) + sum(gaps)
    if total > H:
        return False
    r = rng.randint(0, H - total)
    col_positions = []
    for idx, (_, mask) in enumerate(items):
        col_positions.append(r)
        r += mask.shape[0]
        if idx < len(gaps):
            r += gaps[idx]
    for (color, mask), r0 in zip(items, col_positions):
        c0 = 0 if side == "left" else W - mask.shape[1]
        _paint_mask(grid, mask, color, r0, c0)
        _mark_occupied(occ, r0, c0, mask.shape[0], mask.shape[1])
    return True


def _room_position_candidates(occ: np.ndarray, patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = occ.shape
    h, w = patch_shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            if occ[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue
            candidates.append((r0, c0))
    return candidates


def _apply_periodic_fill(
    base: np.ndarray,
    interior_cells: list[tuple[int, int]],
    seed: tuple[int, int],
    color: int,
    template: np.ndarray,
) -> None:
    seed_r, seed_c = seed
    th, tw = template.shape
    for r, c in interior_cells:
        if template[(r - seed_r) % th, (c - seed_c) % tw]:
            base[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    H = rng.randint(*GRID_H_RANGE)
    W = rng.randint(*GRID_W_RANGE)
    n_regions = rng.randint(*N_REGIONS_RANGE)
    colors = rng.sample(COLOR_POOL, n_regions)
    side = rng.choice(TEMPLATE_EDGE_CHOICES)

    templates: dict[int, np.ndarray] = {}
    for color in colors:
        template = _random_connected_template(rng)
        if template is None:
            return None
        templates[color] = np.rot90(template, rng.randrange(4))

    inp = np.full((H, W), BG, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    if not _place_templates(inp, occ, templates, side, rng):
        return None

    rooms = []
    for color in colors:
        patch = None
        interior = None
        seed = None
        for _ in range(ROOM_PLACEMENT_RETRIES):
            shape = _room_shape(rng.choice(ROOM_SHAPES), rng)
            patch, interior, seed = _build_room_patch(shape, WALL, color, rng)
            candidates = _room_position_candidates(occ, patch.shape)
            if not candidates:
                continue
            r0, c0 = rng.choice(candidates)
            ph, pw = patch.shape
            region = inp[r0 : r0 + ph, c0 : c0 + pw]
            mask = patch != BG
            region[mask] = patch[mask]
            _mark_occupied(occ, r0, c0, ph, pw)
            interior_cells = [
                (r0 + int(r), c0 + int(c)) for r, c in np.argwhere(interior)
            ]
            rooms.append(
                {
                    "color": color,
                    "seed": (r0 + seed[0], c0 + seed[1]),
                    "interior_cells": interior_cells,
                }
            )
            break
        else:
            return None

    expected = inp.copy()
    for room in rooms:
        _apply_periodic_fill(
            expected,
            room["interior_cells"],
            room["seed"],
            room["color"],
            templates[int(room["color"])],
        )

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
