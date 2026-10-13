from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


GRID_SHAPE = (9, 9)
SPRITE_TOP_LEFT = (3, 3)
CORNERS = ("tl", "tr", "bl", "br")
CORNER_MARKER_POS = {
    "tl": (1, 1),
    "tr": (1, GRID_SHAPE[1] - 2),
    "bl": (GRID_SHAPE[0] - 2, 1),
    "br": (GRID_SHAPE[0] - 2, GRID_SHAPE[1] - 2),
}
TRANSFORM_COLORS = (Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW)

_RNG = np.random.default_rng()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SPRITE_ORDER: list[int] = []
_SPRITE_CURSOR = 0
_PERM_ORDER: list[int] = []
_PERM_CURSOR = 0


def _apply_transform(sprite: np.ndarray, color: int) -> np.ndarray:
    color = int(color)
    if color == Color.RED:
        return np.array(sprite, dtype=int, copy=True)
    if color == Color.BLUE:
        return np.fliplr(sprite).copy()
    if color == Color.GREEN:
        return np.flipud(sprite).copy()
    if color == Color.YELLOW:
        return np.rot90(sprite, 2).copy()
    raise ValueError(f"unexpected indicator color {color}")


def _coord_neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _coord_neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]):
                nxt = (int(rr), int(cc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _mask_full_bbox(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    return (
        rows.size > 0
        and int(rows.min()) == 0
        and int(rows.max()) == mask.shape[0] - 1
        and int(cols.min()) == 0
        and int(cols.max()) == mask.shape[1] - 1
    )


def _mask_signature(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).tobytes()


def _build_sprite_catalog() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    all_cells = range(9)
    for area in range(4, 7):
        for picks in itertools.combinations(all_cells, area):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in picks:
                mask[idx // 3, idx % 3] = True
            if not _mask_connected(mask):
                continue
            if not _mask_full_bbox(mask):
                continue

            variants = [mask, np.fliplr(mask), np.flipud(mask), np.rot90(mask, 2)]
            sigs = {_mask_signature(variant) for variant in variants}
            if len(sigs) != 4:
                continue

            orbit_min = min(sigs)
            if _mask_signature(mask) != orbit_min:
                continue

            sprite = np.full((3, 3), Color.BLACK, dtype=int)
            sprite[mask] = Color.GREY
            masks.append(sprite)

    masks.sort(key=lambda arr: arr.tobytes())
    return masks


VALID_SPRITES = _build_sprite_catalog()
ALL_PERMUTATIONS = list(itertools.permutations(TRANSFORM_COLORS))


def _next_sprite() -> np.ndarray:
    global _SPRITE_ORDER, _SPRITE_CURSOR
    if not VALID_SPRITES:
        raise RuntimeError("no valid sprites available")
    if _SPRITE_CURSOR >= len(_SPRITE_ORDER):
        _SPRITE_ORDER = list(map(int, _RNG.permutation(len(VALID_SPRITES))))
        _SPRITE_CURSOR = 0
    sprite = VALID_SPRITES[int(_SPRITE_ORDER[_SPRITE_CURSOR])]
    _SPRITE_CURSOR += 1
    return np.array(sprite, dtype=int, copy=True)


def _next_permutation() -> tuple[int, int, int, int]:
    global _PERM_ORDER, _PERM_CURSOR
    if _PERM_CURSOR >= len(_PERM_ORDER):
        _PERM_ORDER = list(map(int, _RNG.permutation(len(ALL_PERMUTATIONS))))
        _PERM_CURSOR = 0
    perm = ALL_PERMUTATIONS[int(_PERM_ORDER[_PERM_CURSOR])]
    _PERM_CURSOR += 1
    return tuple(map(int, perm))


def _corner_from_coord(r: int, c: int, shape: tuple[int, int]) -> str:
    h, w = map(int, shape)
    top = int(r) < h // 2
    left = int(c) < w // 2
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _extract_sprite_and_markers(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)

    sprite_comps = find_components(
        g,
        bg=Color.BLACK,
        colors=[Color.GREY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(sprite_comps) == 1
    sprite = crop_bbox(g, sprite_comps[0]["bbox"])

    marker_comps = find_components(
        g,
        bg=Color.BLACK,
        colors=list(TRANSFORM_COLORS),
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(marker_comps) == 4

    markers: dict[str, int] = {}
    seen_colors: set[int] = set()
    for comp in marker_comps:
        assert int(comp["area"]) == 1
        color = int(comp["color"])
        assert color not in seen_colors
        seen_colors.add(color)
        r, c = map(int, comp["coords"][0])
        corner = _corner_from_coord(r, c, g.shape)
        assert corner not in markers
        markers[corner] = color

    assert set(markers) == set(CORNERS)
    return sprite, markers


def _compose_output(sprite: np.ndarray, markers: dict[str, int]) -> np.ndarray:
    h, w = map(int, sprite.shape)
    out = np.full((2 * h + 1, 2 * w + 1), Color.BLACK, dtype=int)
    anchors = {
        "tl": (0, 0),
        "tr": (0, w + 1),
        "bl": (h + 1, 0),
        "br": (h + 1, w + 1),
    }
    for corner, (r0, c0) in anchors.items():
        paste(out, _apply_transform(sprite, markers[corner]), r0, c0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the central grey sprite into the four output corners using color-coded mirror/rotation rules.

    Input structure:
    - Black background with one central 4-connected grey sprite.
    - Four isolated corner-region indicator pixels use the four distinct colors blue, red, green, and yellow.
    - The indicator colors encode geometric transforms for the grey sprite; the indicator pixels themselves are not part of the sprite.

    Rule:
    - Extract the grey sprite by its tight bounding box.
    - Read the single indicator color in each input corner quadrant.
    - Map red to identity, blue to left-right mirror, green to up-down mirror, and yellow to 180-degree rotation.
    - Build a black output canvas of shape (2H+1, 2W+1), keep the middle row and middle column black, and paste the transformed sprite into the matching output quadrant.

    Key insight:
    - Corner position chooses which output quadrant to fill, while indicator color independently chooses which of the four transforms to apply.
    """
    sprite, markers = _extract_sprite_and_markers(np.asarray(grid, dtype=int))
    return _compose_output(sprite, markers)


def _build_input(sprite: np.ndarray, corner_colors: dict[str, int]) -> np.ndarray:
    grid = np.full(GRID_SHAPE, Color.BLACK, dtype=int)
    paste(grid, sprite, *SPRITE_TOP_LEFT)
    for corner, color in corner_colors.items():
        r, c = CORNER_MARKER_POS[corner]
        grid[r, c] = int(color)
    return grid


def _validate_generated_case(grid: np.ndarray, expected_markers: dict[str, int]) -> None:
    sprite, markers = _extract_sprite_and_markers(grid)
    assert markers == expected_markers
    assert sprite.shape == (3, 3)

    mask = sprite == Color.GREY
    assert 4 <= int(mask.sum()) <= 6
    assert _mask_connected(mask)
    assert _mask_full_bbox(mask)

    variants = [_apply_transform(sprite, color) for color in TRANSFORM_COLORS]
    assert len({variant.tobytes() for variant in variants}) == 4

    expected = _compose_output(sprite, markers)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    swapped = dict(markers)
    swapped["tl"], swapped["tr"] = swapped["tr"], swapped["tl"]
    assert not np.array_equal(_compose_output(sprite, swapped), expected)

    alt_sprite = np.fliplr(sprite)
    assert not np.array_equal(_compose_output(alt_sprite, markers), expected)


def generate_input() -> np.ndarray:
    attempts = 0
    while attempts < 512:
        attempts += 1
        sprite = _next_sprite()
        perm = _next_permutation()
        corner_colors = {corner: int(color) for corner, color in zip(CORNERS, perm)}
        grid = _build_input(sprite, corner_colors)
        out = main(grid)
        sig = (tuple(map(int, out.shape)), out.tobytes())
        if sig in _SEEN_OUTPUTS:
            continue
        _validate_generated_case(grid, corner_colors)
        _SEEN_OUTPUTS.add(sig)
        return grid
    raise RuntimeError("failed to generate a fresh non-degenerate sample")
