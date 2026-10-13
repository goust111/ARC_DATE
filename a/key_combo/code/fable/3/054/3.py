from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
EDGE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SPRITE_TYPES = ("vertical", "horizontal", "diag_main", "diag_anti")
PAIR_TO_SYMMETRY = {
    frozenset({"left", "right"}): "vertical",
    frozenset({"top", "bottom"}): "horizontal",
    frozenset({"top", "left"}): "diag_main",
    frozenset({"bottom", "right"}): "diag_main",
    frozenset({"top", "right"}): "diag_anti",
    frozenset({"bottom", "left"}): "diag_anti",
}
EDGE_PAIRS_BY_SYMMETRY = {
    "vertical": (("left", "right"),),
    "horizontal": (("top", "bottom"),),
    "diag_main": (("top", "left"), ("bottom", "right")),
    "diag_anti": (("top", "right"), ("bottom", "left")),
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
MAX_GENERATION_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("cannot crop an empty mask")
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _iter_components(mask: np.ndarray, *, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _is_connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    return len(_iter_components(mask, connectivity=4)) == 1


def _reflect_cell(symmetry: str, r: int, c: int, h: int, w: int) -> tuple[int, int]:
    if symmetry == "vertical":
        return r, w - 1 - c
    if symmetry == "horizontal":
        return h - 1 - r, c
    if symmetry == "diag_main":
        return c, r
    if symmetry == "diag_anti":
        return w - 1 - c, h - 1 - r
    raise ValueError(f"unknown symmetry {symmetry}")


def _symmetry_types(mask: np.ndarray) -> set[str]:
    h, w = mask.shape
    types: set[str] = set()
    if np.array_equal(mask, np.fliplr(mask)):
        types.add("vertical")
    if np.array_equal(mask, np.flipud(mask)):
        types.add("horizontal")
    if h == w and np.array_equal(mask, mask.T):
        types.add("diag_main")
    anti = np.flipud(np.fliplr(mask)).T
    if h == w and np.array_equal(mask, anti):
        types.add("diag_anti")
    return types


def _classify_mask(mask: np.ndarray) -> str | None:
    kinds = _symmetry_types(mask)
    if len(kinds) == 1:
        return next(iter(kinds))
    return None


def _detect_side_colors(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    assert h >= 3 and w >= 3, "grid is too small to carry border-side markers"

    def uniform_non_bg(values: np.ndarray) -> int | None:
        unique = {int(v) for v in values}
        if len(unique) == 1:
            color = unique.pop()
            if color != BG:
                return color
        return None

    probes = {
        "top": grid[0, 1:-1],
        "bottom": grid[-1, 1:-1],
        "left": grid[1:-1, 0],
        "right": grid[1:-1, -1],
    }
    side_colors: dict[str, int] = {}
    for side, values in probes.items():
        color = uniform_non_bg(values)
        if color is not None:
            side_colors[side] = color

    assert len(side_colors) == 2, f"expected exactly two colored sides, got {side_colors}"
    pair = frozenset(side_colors)
    assert pair in PAIR_TO_SYMMETRY, f"invalid side pair {pair}"
    return side_colors


def _side_distance(side: str, r: int, c: int, h: int, w: int) -> int:
    if side == "top":
        return r
    if side == "bottom":
        return h - 1 - r
    if side == "left":
        return c
    if side == "right":
        return w - 1 - c
    raise ValueError(f"unknown side {side}")


def _recolor_crop(crop: np.ndarray, side_colors: dict[str, int]) -> np.ndarray:
    h, w = crop.shape
    sides = list(side_colors)
    assert len(sides) == 2
    out = np.full((h, w), BG, dtype=int)
    mask = crop == GREY
    out[mask] = GREY

    for r, c in np.argwhere(mask):
        a, b = sides
        da = _side_distance(a, int(r), int(c), h, w)
        db = _side_distance(b, int(r), int(c), h, w)
        if da < db:
            out[r, c] = side_colors[a]
        elif db < da:
            out[r, c] = side_colors[b]
        else:
            out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Select the sprite whose symmetry matches the border-coded reflection axis and recolor its two sides.

    Input structure:
    - A black background contains several disconnected grey sprites, each having exactly one nontrivial reflection symmetry in its tight bbox.
    - Exactly two border sides are fully colored (excluding the four corners), and the two side colors differ.
    - The colored-side pair encodes one reflection axis: left/right means vertical, top/bottom means horizontal, and adjacent sides mean one of the two diagonals.

    Rule:
    - Detect which reflection swaps the two colored border sides.
    - Among the grey connected sprites, crop the unique sprite whose mask has exactly that symmetry and no other one from the allowed set.
    - In the cropped sprite, recolor each non-axis grey pixel by the color of the closer encoded border side; pixels exactly on the symmetry axis stay grey and background stays black.

    Key insight:
    - The colored border sides do not name a sprite directly; they name a reflection axis, and that same axis both selects the correct symmetric sprite and determines how to split its two colored halves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    side_colors = _detect_side_colors(g)
    target_symmetry = PAIR_TO_SYMMETRY[frozenset(side_colors)]

    matches: list[np.ndarray] = []
    for coords in _iter_components(g == GREY, connectivity=8):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        if _classify_mask(crop == GREY) == target_symmetry:
            matches.append(crop)

    assert len(matches) == 1, f"expected one {target_symmetry} sprite, got {len(matches)}"
    return _recolor_crop(matches[0], side_colors)


def _all_orbits(symmetry: str, h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = _reflect_cell(symmetry, r, c, h, w)
            orbit = tuple(sorted({(r, c), (int(rr), int(cc))}))
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _is_axis_orbit(symmetry: str, orbit: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    if symmetry == "vertical":
        return len(orbit) == 1 and orbit[0][1] == w // 2
    if symmetry == "horizontal":
        return len(orbit) == 1 and orbit[0][0] == h // 2
    if symmetry == "diag_main":
        return len(orbit) == 1 and orbit[0][0] == orbit[0][1]
    if symmetry == "diag_anti":
        return len(orbit) == 1 and orbit[0][0] + orbit[0][1] == w - 1
    return False


def _orbit_adjacent(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> bool:
    for r, c in orbit:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                return True
    return False


def _axis_pixel_count(mask: np.ndarray, symmetry: str) -> int:
    h, w = mask.shape
    if symmetry == "vertical":
        return int(np.count_nonzero(mask[:, w // 2]))
    if symmetry == "horizontal":
        return int(np.count_nonzero(mask[h // 2, :]))
    if symmetry == "diag_main":
        return int(np.count_nonzero(np.diag(mask)))
    if symmetry == "diag_anti":
        return int(np.count_nonzero(np.diag(np.fliplr(mask))))
    raise ValueError(f"unknown symmetry {symmetry}")


def _off_axis_count(mask: np.ndarray, symmetry: str) -> int:
    return int(np.count_nonzero(mask)) - _axis_pixel_count(mask, symmetry)


def _sample_dims(symmetry: str) -> tuple[int, int]:
    if symmetry == "vertical":
        return random.randint(5, 7), random.choice((5, 7))
    if symmetry == "horizontal":
        return random.choice((5, 7)), random.randint(5, 7)
    side = random.randint(5, 7)
    return side, side


def _make_sprite_mask(symmetry: str) -> np.ndarray:
    for _ in range(200):
        h, w = _sample_dims(symmetry)
        orbits = _all_orbits(symmetry, h, w)
        axis_orbits = [orbit for orbit in orbits if _is_axis_orbit(symmetry, orbit, h, w)]
        min_area = max(6, int(np.ceil(0.28 * h * w)))
        max_area = max(min_area + 2, int(np.floor(0.58 * h * w)))
        target_area = random.randint(min_area, max_area)

        mask = np.zeros((h, w), dtype=bool)
        used: set[tuple[tuple[int, int], ...]] = set()
        seed = random.choice(axis_orbits)
        used.add(seed)
        for r, c in seed:
            mask[r, c] = True

        while int(np.count_nonzero(mask)) < target_area:
            frontier = [orbit for orbit in orbits if orbit not in used and _orbit_adjacent(mask, orbit)]
            if not frontier:
                break
            orbit = random.choice(frontier)
            used.add(orbit)
            for r, c in orbit:
                mask[r, c] = True
            if int(np.count_nonzero(mask)) >= min_area and random.random() < 0.18:
                break

        if int(np.count_nonzero(mask)) < min_area:
            continue

        crop = _tight_crop_mask(mask)
        if min(crop.shape) < 3:
            continue
        if not _is_connected(crop):
            continue
        if _classify_mask(crop) != symmetry:
            continue
        if _axis_pixel_count(crop, symmetry) == 0 or _off_axis_count(crop, symmetry) == 0:
            continue
        return crop
    raise RuntimeError(f"failed to build a {symmetry} sprite")


def _render_sprite(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, GREY, BG).astype(int)


def _quadrant_slots(size: int) -> list[tuple[int, int, int, int]]:
    margin = 2
    gap = 2
    inner = size - 2 * margin - gap
    top_h = inner // 2
    bottom_h = inner - top_h
    left_w = inner // 2
    right_w = inner - left_w
    split_r = margin + top_h
    split_c = margin + left_w
    return [
        (margin, margin, split_r - 1, split_c - 1),
        (margin, split_c + gap, split_r - 1, size - margin - 1),
        (split_r + gap, margin, size - margin - 1, split_c - 1),
        (split_r + gap, split_c + gap, size - margin - 1, size - margin - 1),
    ]


def _place_sprite(grid: np.ndarray, sprite: np.ndarray, slot: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = slot
    h, w = sprite.shape
    slot_h = r1 - r0 + 1
    slot_w = c1 - c0 + 1
    if h > slot_h or w > slot_w:
        return None
    rr = random.randint(r0, r1 - h + 1)
    cc = random.randint(c0, c1 - w + 1)
    patch = grid[rr : rr + h, cc : cc + w]
    if np.any((sprite != BG) & (patch != BG)):
        return None
    patch[sprite != BG] = sprite[sprite != BG]
    return rr, cc


def _paint_border_sides(grid: np.ndarray, side_colors: dict[str, int]) -> None:
    if "top" in side_colors:
        grid[0, 1:-1] = side_colors["top"]
    if "bottom" in side_colors:
        grid[-1, 1:-1] = side_colors["bottom"]
    if "left" in side_colors:
        grid[1:-1, 0] = side_colors["left"]
    if "right" in side_colors:
        grid[1:-1, -1] = side_colors["right"]
    grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = BG


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(22, 28)
    grid = np.full((size, size), BG, dtype=int)

    sprites = {symmetry: _render_sprite(_make_sprite_mask(symmetry)) for symmetry in SPRITE_TYPES}
    slots = _quadrant_slots(size)
    random.shuffle(slots)

    placements: dict[str, tuple[int, int]] = {}
    for symmetry, slot in zip(random.sample(SPRITE_TYPES, len(SPRITE_TYPES)), slots):
        origin = _place_sprite(grid, sprites[symmetry], slot)
        if origin is None:
            return None
        placements[symmetry] = origin

    del placements

    target_symmetry = random.choice(SPRITE_TYPES)
    side_a, side_b = random.choice(EDGE_PAIRS_BY_SYMMETRY[target_symmetry])
    color_a, color_b = random.sample(EDGE_COLORS, 2)
    side_colors = {side_a: color_a, side_b: color_b}
    _paint_border_sides(grid, side_colors)

    expected = _recolor_crop(sprites[target_symmetry], side_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
