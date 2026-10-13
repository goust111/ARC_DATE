from __future__ import annotations

import itertools
import random

import numpy as np


SPRITE_COLORS = (1, 2, 3, 4)
DIVIDER_COLOR = 5
BG_COLOR = 0

QUADRANTS = ("UL", "UR", "LL", "LR")
ROTATION_K = {
    "UL": 2,  # 180
    "UR": 3,  # 90 CW
    "LL": 1,  # 90 CCW
    "LR": 0,  # identity
}

SHIFT_MODES = ("none", "row", "col", "both")
SIDE_SLOTS = ("top", "bottom", "left", "right")

GRID_ODD_SIZES = (17, 19, 21, 23)
MAX_GENERATION_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (r,c) coords."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.astype(int).ravel())


def _build_shape_library() -> list[np.ndarray]:
    """Enumerate asymmetric 3x3 4-connected masks used by the generator."""
    cells = [(r, c) for r in range(3) for c in range(3)]
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for size in range(3, 7):
        for combo in itertools.combinations(cells, size):
            if (1, 1) not in combo:
                continue
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            comps = _components4(mask)
            if len(comps) != 1:
                continue
            if _bbox(np.argwhere(mask)) != (0, 0, 2, 2):
                continue
            rotations = [np.rot90(mask, k) for k in range(4)]
            if len({_mask_key(rot) for rot in rotations}) != 4:
                continue
            canon = min(_mask_key(rot) for rot in rotations)
            unique.setdefault(canon, mask)
    library = list(unique.values())
    assert len(library) >= 4
    return library


SHAPE_LIBRARY = _build_shape_library()


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    grey_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER_COLOR)]
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER_COLOR)]
    assert len(grey_rows) == 1 and len(grey_cols) == 1
    return int(grey_rows[0]), int(grey_cols[0])


def _quadrant_of(r: int, c: int, cross_r: int, cross_c: int) -> str:
    if r < cross_r and c < cross_c:
        return "UL"
    if r < cross_r and c > cross_c:
        return "UR"
    if r > cross_r and c < cross_c:
        return "LL"
    if r > cross_r and c > cross_c:
        return "LR"
    raise AssertionError("marker cannot lie on the grey divider")


def _quadrant_bounds(name: str, grid_shape: tuple[int, int], cross_r: int, cross_c: int) -> tuple[int, int, int, int]:
    h, w = grid_shape
    if name == "UL":
        return 0, cross_r - 1, 0, cross_c - 1
    if name == "UR":
        return 0, cross_r - 1, cross_c + 1, w - 1
    if name == "LL":
        return cross_r + 1, h - 1, 0, cross_c - 1
    if name == "LR":
        return cross_r + 1, h - 1, cross_c + 1, w - 1
    raise ValueError(name)


def _place_in_quadrant(
    sprite_mask: np.ndarray,
    marker: tuple[int, int],
    quadrant: str,
    grid_shape: tuple[int, int],
    cross_r: int,
    cross_c: int,
) -> tuple[np.ndarray, str]:
    """Rotate to the target quadrant, center on the marker, then shift minimally inward."""
    rotated = np.rot90(sprite_mask, ROTATION_K[quadrant])
    coords = np.argwhere(rotated)
    anchor_r = rotated.shape[0] // 2
    anchor_c = rotated.shape[1] // 2
    mr, mc = marker
    placed = coords + np.array([mr - anchor_r, mc - anchor_c], dtype=int)

    rmin, rmax, cmin, cmax = _quadrant_bounds(quadrant, grid_shape, cross_r, cross_c)
    min_r = int(placed[:, 0].min())
    max_r = int(placed[:, 0].max())
    min_c = int(placed[:, 1].min())
    max_c = int(placed[:, 1].max())

    shift_r = 0
    if min_r < rmin:
        shift_r = rmin - min_r
    elif max_r > rmax:
        shift_r = rmax - max_r

    shift_c = 0
    if min_c < cmin:
        shift_c = cmin - min_c
    elif max_c > cmax:
        shift_c = cmax - max_c

    placed = placed + np.array([shift_r, shift_c], dtype=int)

    assert np.all(placed[:, 0] >= rmin) and np.all(placed[:, 0] <= rmax)
    assert np.all(placed[:, 1] >= cmin) and np.all(placed[:, 1] <= cmax)
    assert np.all(placed[:, 0] != cross_r) and np.all(placed[:, 1] != cross_c)

    if shift_r and shift_c:
        mode = "both"
    elif shift_r:
        mode = "row"
    elif shift_c:
        mode = "col"
    else:
        mode = "none"
    return placed.astype(int), mode


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    cross_r, cross_c = _find_cross(g)

    markers: dict[int, tuple[int, int]] = {}
    sprites: dict[int, np.ndarray] = {}
    source_coords: dict[int, np.ndarray] = {}

    present = {int(color) for color in np.unique(g) if color not in {BG_COLOR, DIVIDER_COLOR}}
    assert present == set(SPRITE_COLORS)

    for color in SPRITE_COLORS:
        components = _components4(g == color)
        assert len(components) == 2
        components.sort(key=len)
        marker_cc, sprite_cc = components
        assert len(marker_cc) == 1
        assert 3 <= len(sprite_cc) <= 6
        markers[color] = tuple(map(int, marker_cc[0]))
        sprite_mask = _crop_mask(sprite_cc)
        assert sprite_mask.shape[0] <= 3 and sprite_mask.shape[1] <= 3
        sprites[color] = sprite_mask
        source_coords[color] = sprite_cc.astype(int)

    quadrants = {_quadrant_of(r, c, cross_r, cross_c) for r, c in markers.values()}
    assert quadrants == set(QUADRANTS)

    placements: dict[int, np.ndarray] = {}
    shift_modes: dict[int, str] = {}
    for color, marker in markers.items():
        quadrant = _quadrant_of(marker[0], marker[1], cross_r, cross_c)
        placed, shift_mode = _place_in_quadrant(sprites[color], marker, quadrant, g.shape, cross_r, cross_c)
        placements[color] = placed
        shift_modes[color] = shift_mode

    return {
        "cross": (cross_r, cross_c),
        "markers": markers,
        "sprites": sprites,
        "source_coords": source_coords,
        "placements": placements,
        "shift_modes": shift_modes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each edge sprite to the same-colored quadrant marker, rotate by target quadrant, then stamp it at the marker.

    Input structure:
    - A black grid is split by one grey horizontal line and one grey vertical line into four quadrants.
    - Each quadrant contains exactly one isolated marker pixel, and the four marker colors are the distinct set {1,2,3,4}.
    - Four external single-color 4-connected sprites of those same colors sit against the outer edges without crossing the grey lines.

    Rule:
    - For each color, pair the edge sprite with the quadrant marker of the same color.
    - Rotate that sprite according to the marker's quadrant: upper-left 180 degrees, upper-right 90 degrees clockwise, lower-left 90 degrees counterclockwise, lower-right unchanged.
    - Center the rotated sprite's bbox on the marker pixel; if any sprite cell would land on a grey line or outside its quadrant, shift it by the smallest integer amount toward the quadrant interior until it fits.
    - Draw the rotated sprite at that final position, erase the sprite from its original edge location, and keep the grey cross unchanged.

    Key insight:
    - Color decides which sprite travels to which quadrant, and the destination quadrant itself uniquely determines the required rotation.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_input(g)
    out = g.copy()

    for color in SPRITE_COLORS:
        src = scene["source_coords"][color]
        out[src[:, 0], src[:, 1]] = BG_COLOR

    for color in SPRITE_COLORS:
        dst = scene["placements"][color]
        out[dst[:, 0], dst[:, 1]] = color

    cross_r, cross_c = scene["cross"]
    out[cross_r, :] = DIVIDER_COLOR
    out[:, cross_c] = DIVIDER_COLOR
    return out


def _slot_origins(slot: str, h: int, w: int, cross_r: int, cross_c: int) -> list[tuple[int, int]]:
    if slot == "top":
        choices = [(0, c0) for c0 in list(range(3, cross_c - 3)) + list(range(cross_c + 2, w - 5))]
    elif slot == "bottom":
        choices = [(h - 3, c0) for c0 in list(range(3, cross_c - 3)) + list(range(cross_c + 2, w - 5))]
    elif slot == "left":
        choices = [(r0, 0) for r0 in list(range(3, cross_r - 3)) + list(range(cross_r + 2, h - 5))]
    elif slot == "right":
        choices = [(r0, w - 3) for r0 in list(range(3, cross_r - 3)) + list(range(cross_r + 2, h - 5))]
    else:
        raise ValueError(slot)
    assert choices
    return choices


def _pick_marker(mode: str, quadrant: str, h: int, w: int, cross_r: int, cross_c: int) -> tuple[int, int]:
    rmin, rmax, cmin, cmax = _quadrant_bounds(quadrant, (h, w), cross_r, cross_c)
    safe_rmin = max(rmin, 3)
    safe_rmax = min(rmax, h - 4)
    safe_cmin = max(cmin, 3)
    safe_cmax = min(cmax, w - 4)
    interior_rows = list(range(safe_rmin, safe_rmax + 1))
    interior_cols = list(range(safe_cmin, safe_cmax + 1))
    assert interior_rows and interior_cols
    row_near_cross = rmax if quadrant in {"UL", "UR"} else rmin
    col_near_cross = cmax if quadrant in {"UL", "LL"} else cmin
    assert row_near_cross in interior_rows
    assert col_near_cross in interior_cols
    if mode in {"row", "both"}:
        row = row_near_cross
    else:
        row = random.choice(interior_rows)
    if mode in {"col", "both"}:
        col = col_near_cross
    else:
        col = random.choice(interior_cols)
    return int(row), int(col)


def _touches_exactly_one_outer_edge(coords: np.ndarray, h: int, w: int) -> bool:
    touches = {
        "top": bool(np.any(coords[:, 0] == 0)),
        "bottom": bool(np.any(coords[:, 0] == h - 1)),
        "left": bool(np.any(coords[:, 1] == 0)),
        "right": bool(np.any(coords[:, 1] == w - 1)),
    }
    return sum(int(v) for v in touches.values()) == 1


def _validate_generated_input(grid: np.ndarray) -> None:
    scene = _analyze_input(grid)
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cross_r, cross_c = scene["cross"]

    shape_keys = set()
    for color in SPRITE_COLORS:
        sprite = scene["sprites"][color]
        assert sprite.shape == (3, 3)
        assert len({_mask_key(np.rot90(sprite, k)) for k in range(4)}) == 4
        key = min(_mask_key(np.rot90(sprite, k)) for k in range(4))
        assert key not in shape_keys
        shape_keys.add(key)

        src = scene["source_coords"][color]
        assert _touches_exactly_one_outer_edge(src, h, w)
        assert np.all(np.abs(src[:, 0] - cross_r) > 1)
        assert np.all(np.abs(src[:, 1] - cross_c) > 1)

    assert set(scene["shift_modes"].values()) == set(SHIFT_MODES)

    out = main(g)
    assert np.all(out[cross_r, :] == DIVIDER_COLOR)
    assert np.all(out[:, cross_c] == DIVIDER_COLOR)
    for color in SPRITE_COLORS:
        src = scene["source_coords"][color]
        assert np.all(out[src[:, 0], src[:, 1]] == BG_COLOR)
        comps = _components4(out == color)
        assert len(comps) == 1
        expected = {tuple(map(int, rc)) for rc in scene["placements"][color]}
        actual = {tuple(map(int, rc)) for rc in comps[0]}
        assert actual == expected


def _generate_attempt() -> np.ndarray | None:
    h = random.choice(GRID_ODD_SIZES)
    w = random.choice(GRID_ODD_SIZES)
    cross_r = h // 2
    cross_c = w // 2

    grid = np.zeros((h, w), dtype=int)
    grid[cross_r, :] = DIVIDER_COLOR
    grid[:, cross_c] = DIVIDER_COLOR

    chosen_shapes = [shape.copy() for shape in random.sample(SHAPE_LIBRARY, 4)]
    random.shuffle(chosen_shapes)

    colors = list(SPRITE_COLORS)
    random.shuffle(colors)
    color_by_quadrant = dict(zip(QUADRANTS, colors))

    shift_modes = list(SHIFT_MODES)
    random.shuffle(shift_modes)
    marker_by_quadrant = {
        quadrant: _pick_marker(mode, quadrant, h, w, cross_r, cross_c)
        for quadrant, mode in zip(QUADRANTS, shift_modes)
    }

    for quadrant, color in color_by_quadrant.items():
        mr, mc = marker_by_quadrant[quadrant]
        grid[mr, mc] = color

    side_order = list(SIDE_SLOTS)
    random.shuffle(side_order)
    for quadrant, slot, base_mask in zip(QUADRANTS, side_order, chosen_shapes):
        color = color_by_quadrant[quadrant]
        sprite = np.rot90(base_mask, random.randrange(4))
        r0, c0 = random.choice(_slot_origins(slot, h, w, cross_r, cross_c))
        coords = np.argwhere(sprite) + np.array([r0, c0], dtype=int)
        assert np.all(grid[coords[:, 0], coords[:, 1]] == BG_COLOR)
        grid[coords[:, 0], coords[:, 1]] = color

    try:
        _validate_generated_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
