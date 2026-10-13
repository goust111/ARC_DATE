from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
WALL_THICKNESS = 2
MAX_GEN_ATTEMPTS = 800


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            cur = np.rot90(base, k)
            key = (cur.shape, cur.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                out.append(cur.copy())
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((variant.shape, variant.astype(np.uint8).tobytes()) for variant in _mask_variants(mask))


def _ring_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], thickness: int = WALL_THICKNESS) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mask[r0 + thickness : r1 - thickness + 1, c0 + thickness : c1 - thickness + 1] = False
    return mask


def _find_wall(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]], tuple[int, int, int, int]]:
    best_color = None
    best_total = -1
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        total = int(np.sum(grid == color))
        if total > best_total:
            best_total = total
            best_color = color
    assert best_color is not None
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == best_color)]
    return best_color, coords, _bbox(coords)


def _flood_fill(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    sr, sc = start
    assert 0 <= sr < h and 0 <= sc < w and passable[sr, sc]
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Seal wall holes with shape-matched outside sprites, then flood the seed color through the remaining openings.

    Input structure:
    - A large monochrome wall ring encloses one interior region, and some connected hole shapes are carved through the wall band.
    - Inside the enclosure there is exactly one single-cell fill seed.
    - Outside the wall there are several monochrome sprites; some match hole shapes up to rotation or reflection, and some are distractors.

    Rule:
    - Identify the wall, reconstruct its rectangular ring, and extract each hole as a connected shape inside that ring band.
    - For every hole whose shape has exactly one matching outside sprite, erase that sprite and fill the hole with the sprite's color.
    - Starting from the seed, flood-fill in 4-connectivity through all non-blocked cells; the wall, sealed holes, and unmatched outside sprites block the flood.

    Key insight:
    - Shape matching decides which openings remain as connectivity gates between the inside and the outside, so matching changes the flood's reach rather than directly defining the final colored area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BACKGROUND if np.any(g == BACKGROUND) else int(np.bincount(g.ravel()).argmax())

    wall_color, wall_comp, wall_bbox = _find_wall(g, bg)
    r0, c0, r1, c1 = wall_bbox
    assert r1 - r0 + 1 >= 2 * WALL_THICKNESS + 3
    assert c1 - c0 + 1 >= 2 * WALL_THICKNESS + 3

    ideal_ring = _ring_mask(g.shape, wall_bbox)
    hole_mask = ideal_ring & (g != wall_color)
    hole_components = _components(hole_mask)
    assert hole_components

    inner_r0 = r0 + WALL_THICKNESS
    inner_c0 = c0 + WALL_THICKNESS
    inner_r1 = r1 - WALL_THICKNESS
    inner_c1 = c1 - WALL_THICKNESS

    other_mask = (g != bg) & (g != wall_color)
    other_components = _components(other_mask)
    seed_pos = None
    seed_color = None
    sprites: list[dict[str, object]] = []
    for comp in other_components:
        inside = [(r, c) for r, c in comp if inner_r0 <= r <= inner_r1 and inner_c0 <= c <= inner_c1]
        color = int(g[comp[0][0], comp[0][1]])
        if inside:
            assert len(comp) == 1
            assert seed_pos is None
            seed_pos = comp[0]
            seed_color = color
            continue
        shape, bbox = _shape_from_coords(comp)
        sprites.append(
            {
                "coords": comp,
                "color": color,
                "shape": shape,
                "key": _canonical_key(shape),
                "bbox": bbox,
            }
        )

    assert seed_pos is not None and seed_color is not None

    sprites_by_key: dict[tuple[tuple[int, int], bytes], list[dict[str, object]]] = {}
    for sprite in sprites:
        sprites_by_key.setdefault(sprite["key"], []).append(sprite)

    out = g.copy()
    matched_sprite_ids: set[int] = set()
    for hole_comp in hole_components:
        hole_shape, _ = _shape_from_coords(hole_comp)
        hole_key = _canonical_key(hole_shape)
        matches = sprites_by_key.get(hole_key, [])
        if len(matches) == 1:
            sprite = matches[0]
            matched_sprite_ids.add(id(sprite))
            out[[r for r, _ in hole_comp], [c for _, c in hole_comp]] = int(sprite["color"])
        else:
            out[[r for r, _ in hole_comp], [c for _, c in hole_comp]] = bg

    for sprite in sprites:
        if id(sprite) in matched_sprite_ids:
            coords = sprite["coords"]
            out[[r for r, _ in coords], [c for _, c in coords]] = bg

    blocked = out != bg
    sr, sc = seed_pos
    blocked[sr, sc] = False
    flooded = _flood_fill(~blocked, seed_pos)
    out[flooded] = seed_color
    return out


HOLE_LIBRARY = [
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]

DISTRACTOR_LIBRARY = [
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
]

SCENARIO_BAG: list[str] = []


def _next_scenario() -> str:
    global SCENARIO_BAG
    if not SCENARIO_BAG:
        SCENARIO_BAG = random.sample(["mixed", "mixed", "mixed", "all", "none"], 5)
    return SCENARIO_BAG.pop()


def _oriented_hole_variants(mask: np.ndarray, side: str) -> list[np.ndarray]:
    variants = []
    for variant in _mask_variants(mask):
        if side in ("top", "bottom"):
            if variant.shape[0] != WALL_THICKNESS:
                continue
            if not np.any(variant[0]) or not np.any(variant[-1]):
                continue
        else:
            if variant.shape[1] != WALL_THICKNESS:
                continue
            if not np.any(variant[:, 0]) or not np.any(variant[:, -1]):
                continue
        variants.append(variant)
    return variants


def _place_hole(side: str, mask: np.ndarray, outer_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = outer_bbox
    h, w = mask.shape
    side_gap = WALL_THICKNESS + 1
    if side == "top":
        start_r = r0
        start_c = random.randint(c0 + side_gap, c1 - side_gap - w + 1)
    elif side == "bottom":
        start_r = r1 - WALL_THICKNESS + 1
        start_c = random.randint(c0 + side_gap, c1 - side_gap - w + 1)
    elif side == "left":
        start_r = random.randint(r0 + side_gap, r1 - side_gap - h + 1)
        start_c = c0
    else:
        start_r = random.randint(r0 + side_gap, r1 - side_gap - h + 1)
        start_c = c1 - WALL_THICKNESS + 1
    coords: list[tuple[int, int]] = []
    for rr in range(h):
        for cc in range(w):
            if mask[rr, cc]:
                coords.append((start_r + rr, start_c + cc))
    return coords


def _can_place_sprite(
    grid: np.ndarray,
    sprite: np.ndarray,
    top: int,
    left: int,
    outer_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    sh, sw = sprite.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    r0, c0, r1, c1 = outer_bbox
    coords = [(top + rr, left + cc) for rr in range(sh) for cc in range(sw) if sprite[rr, cc]]
    for r, c in coords:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return False
        if grid[r, c] != BACKGROUND:
            return False
    pad_r0 = max(0, top - 1)
    pad_c0 = max(0, left - 1)
    pad_r1 = min(h - 1, top + sh)
    pad_c1 = min(w - 1, left + sw)
    if np.any(grid[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] != BACKGROUND):
        return False
    return True


def _draw_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, color: int) -> None:
    sh, sw = sprite.shape
    for rr in range(sh):
        for cc in range(sw):
            if sprite[rr, cc]:
                grid[top + rr, left + cc] = color


def _place_sprite_random(
    grid: np.ndarray,
    sprite: np.ndarray,
    color: int,
    outer_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    sh, sw = sprite.shape
    positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if _can_place_sprite(grid, sprite, top, left, outer_bbox):
            _draw_sprite(grid, sprite, top, left, color)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    colors = list(range(1, 10))
    random.shuffle(colors)
    wall_color = colors.pop()
    seed_color = colors.pop()

    outer_h = random.randint(10, min(14, h - 8))
    outer_w = random.randint(10, min(14, w - 8))
    r0 = random.randint(4, h - outer_h - 4)
    c0 = random.randint(4, w - outer_w - 4)
    outer_bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    r1 = outer_bbox[2]
    c1 = outer_bbox[3]

    ring = _ring_mask(grid.shape, outer_bbox)
    grid[ring] = wall_color

    n_holes = random.randint(2, 3)
    sides = random.sample(["top", "bottom", "left", "right"], n_holes)
    base_shapes = random.sample(HOLE_LIBRARY, n_holes)
    holes: list[dict[str, object]] = []
    for side, base_mask in zip(sides, base_shapes):
        variants = _oriented_hole_variants(base_mask, side)
        if not variants:
            return None
        hole_mask = random.choice(variants)
        coords = _place_hole(side, hole_mask, outer_bbox)
        for r, c in coords:
            grid[r, c] = BACKGROUND
        holes.append(
            {
                "coords": coords,
                "shape": hole_mask,
                "key": _canonical_key(hole_mask),
            }
        )

    inner_r0 = r0 + WALL_THICKNESS
    inner_c0 = c0 + WALL_THICKNESS
    inner_r1 = r1 - WALL_THICKNESS
    inner_c1 = c1 - WALL_THICKNESS
    seed_r = random.randint(inner_r0 + 1, inner_r1 - 1)
    seed_c = random.randint(inner_c0 + 1, inner_c1 - 1)
    grid[seed_r, seed_c] = seed_color

    scenario = _next_scenario()
    if scenario == "all":
        matched_indices = set(range(n_holes))
    elif scenario == "none":
        matched_indices = set()
    else:
        matched_count = random.randint(1, n_holes - 1)
        matched_indices = set(random.sample(range(n_holes), matched_count))

    used_keys = {hole["key"] for hole in holes}
    distractor_candidates = [mask for mask in DISTRACTOR_LIBRARY if _canonical_key(mask) not in used_keys]
    if not distractor_candidates:
        return None

    sprite_specs: list[tuple[np.ndarray, int]] = []
    needed_sprite_count = len(matched_indices) + random.randint(1, 2)
    if len(colors) < needed_sprite_count:
        return None

    for idx in matched_indices:
        hole_shape = holes[idx]["shape"]
        sprite_shape = random.choice(_mask_variants(hole_shape))
        sprite_specs.append((sprite_shape, colors.pop()))

    distractor_count = needed_sprite_count - len(matched_indices)
    for _ in range(distractor_count):
        base = random.choice(distractor_candidates)
        sprite_shape = random.choice(_mask_variants(base))
        sprite_specs.append((sprite_shape, colors.pop()))

    random.shuffle(sprite_specs)
    for sprite_shape, sprite_color in sprite_specs:
        if not _place_sprite_random(grid, sprite_shape, sprite_color, outer_bbox):
            return None

    # Verify one-to-one matching by canonical key.
    sprite_comps = _components((grid != BACKGROUND) & (grid != wall_color) & ~(
        (np.arange(h)[:, None] == seed_r) & (np.arange(w)[None, :] == seed_c)
    ))
    sprite_key_counts: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in sprite_comps:
        if (seed_r, seed_c) in comp:
            continue
        if any(inner_r0 <= r <= inner_r1 and inner_c0 <= c <= inner_c1 for r, c in comp):
            continue
        shape, _ = _shape_from_coords(comp)
        key = _canonical_key(shape)
        sprite_key_counts[key] = sprite_key_counts.get(key, 0) + 1

    expected_unique_matches = 0
    for idx, hole in enumerate(holes):
        count = sprite_key_counts.get(hole["key"], 0)
        if idx in matched_indices:
            if count != 1:
                return None
            expected_unique_matches += 1
        elif count != 0:
            return None

    if expected_unique_matches != len(matched_indices):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid shape-key flood input")
