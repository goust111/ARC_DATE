from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
ShapeSig = tuple[Coord, ...]

BG = 0
WALL = 5
SPRITE_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

CHAMBER_COUNT_RANGE = (2, 4)
CHAMBER_AREA_RANGE = (4, 7)
DECOY_COUNT_RANGE = (1, 3)
MAX_SHAPE_SPAN = 5
MAX_GRID_SPAN = 30
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            seen[sr, sc] = True
            queue = deque([(sr, sc)])
            coords: list[Coord] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _normalize(coords: Iterable[Coord] | np.ndarray) -> ShapeSig:
    arr = np.asarray(list(coords), dtype=int)
    arr = arr - arr.min(axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return tuple((int(r), int(c)) for r, c in arr)


def _shape_dims(sig: ShapeSig) -> tuple[int, int]:
    rows = [r for r, _ in sig]
    cols = [c for _, c in sig]
    return max(rows) + 1, max(cols) + 1


def _shape_area(sig: ShapeSig) -> int:
    return len(sig)


def _shape_mask(sig: ShapeSig) -> np.ndarray:
    h, w = _shape_dims(sig)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in sig:
        mask[r, c] = True
    return mask


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    colors = {int(v) for v in np.unique(grid)}
    if WALL in colors and WALL != bg:
        return WALL
    counts = Counter(map(int, grid.ravel()))
    candidates = [c for c in colors if c != bg and counts[c] > 1]
    assert candidates, "expected a repeated wall color"
    return max(candidates, key=lambda c: counts[c])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each enclosed chamber of the grey wall with the congruent outside sprite, fill the chamber, and erase the claimed sprite.

    Input structure:
    - The dominant background surrounds a grey wall structure placed on one side of the grid.
    - Inside the wall are several enclosed background chambers with pairwise distinct shapes.
    - Outside the wall are monochrome solid sprite components; each chamber has exactly one translation-congruent sprite, plus extra non-matching decoys.

    Rule:
    - Treat each enclosed background chamber as a negative-shape object.
    - Find the outside monochrome sprite whose filled shape is congruent to that chamber by translation only, without rotation or reflection.
    - Fill the whole chamber with that sprite's color and erase the matched sprite to background; leave decoys and the grey wall unchanged.

    Key insight:
    - The meaningful objects inside the wall are enclosed background holes rather than painted foreground pieces, so the task is solved by figure-ground reversal followed by exact shape matching.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    wall = _infer_wall_color(g, bg)
    h, w = g.shape

    chambers: dict[ShapeSig, np.ndarray] = {}
    for comp in _components(g == bg):
        if _touches_border(comp, (h, w)):
            continue
        sig = _normalize(comp)
        assert sig not in chambers, "chamber shapes should be unique"
        chambers[sig] = comp

    sprite_matches: dict[ShapeSig, tuple[int, np.ndarray]] = {}
    for color in sorted({int(v) for v in np.unique(g)} - {bg, wall}):
        for comp in _components(g == color):
            sig = _normalize(comp)
            if sig in chambers:
                assert sig not in sprite_matches, "each chamber should have a unique matching sprite"
                sprite_matches[sig] = (color, comp)

    assert chambers, "expected at least one enclosed chamber"
    assert set(sprite_matches) == set(chambers), "every chamber must have exactly one congruent sprite"

    out = g.copy()
    for sig, chamber in chambers.items():
        color, sprite = sprite_matches[sig]
        out[chamber[:, 0], chamber[:, 1]] = int(color)
        out[sprite[:, 0], sprite[:, 1]] = bg
    return out


def _random_polyomino(rng: random.Random, area: int) -> ShapeSig:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nbr = (r + dr, c + dc)
                    if nbr not in cells:
                        frontier.add(nbr)
            candidate = rng.choice(tuple(frontier))
            trial = _normalize(cells | {candidate})
            th, tw = _shape_dims(trial)
            if th <= MAX_SHAPE_SPAN and tw <= MAX_SHAPE_SPAN:
                cells = set(trial)
        sig = _normalize(cells)
        sh, sw = _shape_dims(sig)
        if sh >= 2 and sw >= 2:
            return sig
    raise RuntimeError("failed to sample a compact polyomino")


def _transformed_variants(sig: ShapeSig) -> list[ShapeSig]:
    mask = _shape_mask(sig)
    variants: set[ShapeSig] = set()
    for rot in range(4):
        rotated = np.rot90(mask, rot)
        for arr in (rotated, np.fliplr(rotated)):
            coords = np.argwhere(arr)
            variants.add(_normalize(coords))
    variants.discard(sig)
    return sorted(variants)


def _sample_chambers(rng: random.Random) -> list[ShapeSig]:
    count = rng.randint(*CHAMBER_COUNT_RANGE)
    areas = [rng.randint(*CHAMBER_AREA_RANGE) for _ in range(count)]
    if count >= 3 and rng.random() < 0.7:
        i, j = rng.sample(range(count), 2)
        areas[j] = areas[i]

    used: set[ShapeSig] = set()
    shapes: list[ShapeSig] = []
    for area in areas:
        for _ in range(200):
            sig = _random_polyomino(rng, area)
            if sig not in used:
                used.add(sig)
                shapes.append(sig)
                break
        else:
            raise RuntimeError("failed to sample distinct chamber shapes")
    return shapes


def _sample_decoys(rng: random.Random, chambers: list[ShapeSig]) -> list[ShapeSig]:
    max_decoys = min(DECOY_COUNT_RANGE[1], len(SPRITE_PALETTE) - len(chambers))
    count = rng.randint(DECOY_COUNT_RANGE[0], max_decoys)
    used = set(chambers)
    decoys: list[ShapeSig] = []

    anchor = rng.choice(chambers)
    variants = [v for v in _transformed_variants(anchor) if v not in used]
    if variants and rng.random() < 0.7:
        first = rng.choice(variants)
    else:
        area = _shape_area(anchor)
        for _ in range(200):
            first = _random_polyomino(rng, area)
            if first not in used:
                break
        else:
            raise RuntimeError("failed to sample first decoy")
    decoys.append(first)
    used.add(first)

    while len(decoys) < count:
        area = _shape_area(rng.choice(chambers)) if rng.random() < 0.8 else rng.randint(*CHAMBER_AREA_RANGE)
        for _ in range(200):
            sig = _random_polyomino(rng, area)
            if sig not in used:
                decoys.append(sig)
                used.add(sig)
                break
        else:
            raise RuntimeError("failed to sample extra decoy")
    return decoys


def _pack_sizes(
    rng: random.Random,
    sizes: list[tuple[int, int]],
    *,
    gap_range: tuple[int, int],
    max_row_cap: int,
    pad_range: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    gap = rng.randint(*gap_range)
    max_item_w = max(w for _, w in sizes)
    flat_w = sum(w for _, w in sizes) + gap * (len(sizes) - 1)
    max_row_w = rng.randint(max_item_w, min(flat_w, max_row_cap))

    positions: list[tuple[int, int]] = []
    x = 0
    y = 0
    row_h = 0
    used_w = 0
    for h, w in sizes:
        if positions and x + w > max_row_w:
            y += row_h + gap
            x = 0
            row_h = 0
        positions.append((y, x))
        x += w + gap
        row_h = max(row_h, h)
        used_w = max(used_w, x - gap)

    used_h = y + row_h
    pad_top = rng.randint(*pad_range)
    pad_left = rng.randint(*pad_range)
    pad_bottom = rng.randint(*pad_range)
    pad_right = rng.randint(*pad_range)
    shifted = [(r + pad_top, c + pad_left) for r, c in positions]
    total_h = used_h + pad_top + pad_bottom
    total_w = used_w + pad_left + pad_right
    return shifted, (total_h, total_w)


def _paste_shape(canvas: np.ndarray, sig: ShapeSig, top: int, left: int, color: int) -> np.ndarray:
    coords = np.asarray(sig, dtype=int)
    canvas[top + coords[:, 0], left + coords[:, 1]] = int(color)
    return coords + np.array([top, left], dtype=int)


def _build_wall_patch(
    rng: random.Random,
    chambers: list[ShapeSig],
) -> tuple[np.ndarray, list[np.ndarray]]:
    order = chambers[:]
    rng.shuffle(order)
    sizes = [_shape_dims(sig) for sig in order]
    positions, (inner_h, inner_w) = _pack_sizes(
        rng,
        sizes,
        gap_range=(1, 2),
        max_row_cap=14,
        pad_range=(0, 1),
    )

    patch = np.full((inner_h + 2, inner_w + 2), WALL, dtype=int)
    chamber_coords: list[np.ndarray] = []
    for sig, (r0, c0) in zip(order, positions):
        chamber_coords.append(_paste_shape(patch, sig, r0 + 1, c0 + 1, BG))
    return patch, chamber_coords


def _build_sprite_zone(
    rng: random.Random,
    items: list[tuple[str, ShapeSig, int]],
) -> tuple[np.ndarray, list[tuple[str, ShapeSig, int, np.ndarray]]]:
    order = items[:]
    rng.shuffle(order)
    sizes = [_shape_dims(sig) for _, sig, _ in order]
    positions, zone_shape = _pack_sizes(
        rng,
        sizes,
        gap_range=(2, 3),
        max_row_cap=13,
        pad_range=(0, 1),
    )

    zone = np.full(zone_shape, BG, dtype=int)
    placed: list[tuple[str, ShapeSig, int, np.ndarray]] = []
    for (kind, sig, color), (r0, c0) in zip(order, positions):
        coords = _paste_shape(zone, sig, r0, c0, color)
        placed.append((kind, sig, color, coords))
    return zone, placed


def _compose_scene(
    rng: random.Random,
    wall_patch: np.ndarray,
    chamber_coords: list[np.ndarray],
    zone: np.ndarray,
    placed_sprites: list[tuple[str, ShapeSig, int, np.ndarray]],
    chamber_colors: dict[ShapeSig, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    wall_h, wall_w = wall_patch.shape
    zone_h, zone_w = zone.shape

    for _ in range(40):
        side = rng.choice(["left", "right", "top", "bottom"])
        top = rng.randint(1, 3)
        bottom = rng.randint(1, 3)
        left = rng.randint(1, 3)
        right = rng.randint(1, 3)
        sep = rng.randint(2, 4)

        if side in ("left", "right"):
            height = max(wall_h, zone_h) + top + bottom
            width = wall_w + zone_w + sep + left + right
            if height > MAX_GRID_SPAN or width > MAX_GRID_SPAN:
                continue
            wall_y = top + rng.randint(0, height - top - bottom - wall_h)
            zone_y = top + rng.randint(0, height - top - bottom - zone_h)
            if side == "left":
                wall_x = left
                zone_x = width - right - zone_w
            else:
                zone_x = left
                wall_x = width - right - wall_w
        else:
            height = wall_h + zone_h + sep + top + bottom
            width = max(wall_w, zone_w) + left + right
            if height > MAX_GRID_SPAN or width > MAX_GRID_SPAN:
                continue
            wall_x = left + rng.randint(0, width - left - right - wall_w)
            zone_x = left + rng.randint(0, width - left - right - zone_w)
            if side == "top":
                wall_y = top
                zone_y = height - bottom - zone_h
            else:
                zone_y = top
                wall_y = height - bottom - wall_h

        inp = np.full((height, width), BG, dtype=int)
        inp[wall_y : wall_y + wall_h, wall_x : wall_x + wall_w] = wall_patch
        zone_mask = zone != BG
        sub = inp[zone_y : zone_y + zone_h, zone_x : zone_x + zone_w]
        if np.any(zone_mask & (sub != BG)):
            continue
        sub[zone_mask] = zone[zone_mask]

        if get_bg(inp) != BG:
            continue

        out = inp.copy()
        for local_coords in chamber_coords:
            abs_coords = local_coords + np.array([wall_y, wall_x], dtype=int)
            sig = _normalize(local_coords)
            out[abs_coords[:, 0], abs_coords[:, 1]] = chamber_colors[sig]

        for kind, _sig, _color, local_coords in placed_sprites:
            if kind != "match":
                continue
            abs_coords = local_coords + np.array([zone_y, zone_x], dtype=int)
            out[abs_coords[:, 0], abs_coords[:, 1]] = BG

        return inp, out
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            chambers = _sample_chambers(rng)
            decoys = _sample_decoys(rng, chambers)
            total_sprites = len(chambers) + len(decoys)
            sprite_colors = rng.sample(SPRITE_PALETTE, total_sprites)

            chamber_colors = {sig: color for sig, color in zip(chambers, sprite_colors[: len(chambers)])}
            sprite_items: list[tuple[str, ShapeSig, int]] = [
                ("match", sig, chamber_colors[sig]) for sig in chambers
            ]
            sprite_items.extend(
                ("decoy", sig, color)
                for sig, color in zip(decoys, sprite_colors[len(chambers) :])
            )

            wall_patch, chamber_coords = _build_wall_patch(rng, chambers)
            zone, placed_sprites = _build_sprite_zone(rng, sprite_items)
            scene = _compose_scene(rng, wall_patch, chamber_coords, zone, placed_sprites, chamber_colors)
            if scene is None:
                continue
            inp, expected = scene

            chamber_areas = Counter(_shape_area(sig) for sig in chambers)
            assert any(_shape_area(sig) in chamber_areas for sig in decoys), "need a same-area decoy"
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("generate_input: failed to produce a valid sample")
