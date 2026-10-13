from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
SPRITE_SIZE = 2
SEED_SIZE = 4
TILE_REPEAT = 3
OUTPUT_SIZE = SEED_SIZE * TILE_REPEAT
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_RANGE = (13, 18)
MIN_BBOX_GAP = 2

MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
MISSING_TO_DEST = {
    (1, 1): (0, 0),  # missing BR -> place in UL quadrant
    (1, 0): (0, 2),  # missing BL -> place in UR quadrant
    (0, 1): (2, 0),  # missing TR -> place in LL quadrant
    (0, 0): (2, 2),  # missing TL -> place in LR quadrant
}
MISSING_TO_OUTPUT_QUADRANT = {
    (1, 1): (0, 0),
    (1, 0): (0, 1),
    (0, 1): (1, 0),
    (0, 0): (1, 1),
}
ROW_MAJOR_OUTPUT_ORIENTATION_ORDER = ((1, 1), (1, 0), (0, 1), (0, 0))


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords_arr.shape[0]),
                }
            )

    return components


def _extract_sprite_info(grid: np.ndarray, component: dict, bg: int) -> dict:
    r0, c0, r1, c1 = component["bbox"]
    assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "each sprite must fit a 2x2 bbox"
    assert int(component["area"]) == 3, "each sprite must contain exactly three cells"

    sprite = np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()
    color = int(component["color"])
    assert set(map(int, np.unique(sprite))).issubset({bg, color}), "each component must be monochrome"
    assert int(np.count_nonzero(sprite == color)) == 3, "each 2x2 sprite must have three colored cells"

    missing = np.argwhere(sprite == bg)
    assert missing.shape == (1, 2), "each sprite must have exactly one missing corner"
    missing_corner = tuple(map(int, missing[0]))
    assert missing_corner in MISSING_TO_DEST, "missing corner must be one of the four 2x2 corners"

    return {
        "color": color,
        "bbox": component["bbox"],
        "sprite": sprite,
        "missing": missing_corner,
    }


def _build_seed(sprite_infos: list[dict], bg: int) -> np.ndarray:
    assert len(sprite_infos) == 4, "expected exactly four sprites"
    assert {info["missing"] for info in sprite_infos} == set(MISSING_CORNERS), "all four missing-corner orientations must appear exactly once"
    assert {int(info["color"]) for info in sprite_infos} == set(PALETTE), "the four sprite colors must be blue, red, green, and yellow"

    seed = np.full((SEED_SIZE, SEED_SIZE), bg, dtype=int)
    used_destinations: set[tuple[int, int]] = set()
    for info in sprite_infos:
        dst_r, dst_c = MISSING_TO_DEST[info["missing"]]
        assert (dst_r, dst_c) not in used_destinations, "each orientation must map to a distinct quadrant"
        used_destinations.add((dst_r, dst_c))
        seed[dst_r : dst_r + SPRITE_SIZE, dst_c : dst_c + SPRITE_SIZE] = info["sprite"]

    assert np.all(seed[1:3, 1:3] == bg), "the assembled seed must leave a central 2x2 hole"
    return seed


def _tile_output(seed: np.ndarray) -> np.ndarray:
    mirrored = np.fliplr(seed)
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), int(seed[1, 1]), dtype=int)
    for tile_r in range(TILE_REPEAT):
        for tile_c in range(TILE_REPEAT):
            tile = seed if (tile_r + tile_c) % 2 == 0 else mirrored
            r0 = tile_r * SEED_SIZE
            c0 = tile_c * SEED_SIZE
            out[r0 : r0 + SEED_SIZE, c0 : c0 + SEED_SIZE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered L sprites into a 4x4 seed tile, then fill a 3x3 canvas with alternating left-right mirrors of that seed.

    Input structure:
    - The background is black.
    - There are exactly four disconnected monochrome foreground objects.
    - Each object is a three-cell L sprite occupying a 2x2 bounding box with one missing corner.
    - The four colors are blue, red, green, and yellow, and the four missing-corner orientations appear exactly once each.

    Rule:
    - Crop each sprite to its 2x2 bounding box.
    - Map missing BR, BL, TR, TL to the seed tile quadrants UL, UR, LL, LR respectively, preserving each sprite's orientation.
    - The four missing corners must face inward, so the assembled 4x4 seed contains a central 2x2 black hole.
    - Produce a 12x12 output by tiling the seed over a 3x3 tile grid; use the original seed when tile-row + tile-col is even and its left-right mirror when that sum is odd.

    Key insight:
    - Absolute input positions are irrelevant: the missing corner determines the sprite's seed quadrant, and the seed itself is then the unit that alternates with its horizontal mirror across the final tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    bg = _background_color(g)
    assert bg == BG, "expected black background"

    components = _find_monochrome_components(g, bg)
    assert len(components) == 4, "expected exactly four foreground sprites"
    sprite_infos = [_extract_sprite_info(g, component, bg) for component in components]

    seed = _build_seed(sprite_infos, bg)
    assert not np.array_equal(seed, np.fliplr(seed)), "seed must not be left-right symmetric"
    return _tile_output(seed)


def _make_sprite(color: int, missing_corner: tuple[int, int]) -> np.ndarray:
    sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), int(color), dtype=int)
    sprite[missing_corner] = BG
    return sprite


def _boxes_far_enough(a: tuple[int, int], b: tuple[int, int], gap: int = MIN_BBOX_GAP) -> bool:
    ar, ac = a
    br, bc = b
    return (
        ar + SPRITE_SIZE - 1 + gap < br
        or br + SPRITE_SIZE - 1 + gap < ar
        or ac + SPRITE_SIZE - 1 + gap < bc
        or bc + SPRITE_SIZE - 1 + gap < ac
    )


def _sample_positions(height: int, width: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(height - SPRITE_SIZE + 1) for c in range(width - SPRITE_SIZE + 1)]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for pos in candidates:
        if all(_boxes_far_enough(pos, other) for other in chosen):
            chosen.append(pos)
            if len(chosen) == 4:
                return chosen
    return None


def _placement_breaks_positional_shortcuts(
    placement_by_orientation: dict[tuple[int, int], tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    row_major = tuple(
        orientation
        for orientation, _pos in sorted(
            placement_by_orientation.items(),
            key=lambda item: (item[1][0], item[1][1]),
        )
    )
    if row_major == ROW_MAJOR_OUTPUT_ORIENTATION_ORDER:
        return False

    quadrant_matches = 0
    for orientation, (r0, c0) in placement_by_orientation.items():
        center_r = r0 + 0.5
        center_c = c0 + 0.5
        input_quadrant = (
            0 if center_r < height / 2.0 else 1,
            0 if center_c < width / 2.0 else 1,
        )
        if input_quadrant == MISSING_TO_OUTPUT_QUADRANT[orientation]:
            quadrant_matches += 1

    return quadrant_matches <= 2


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    positions = _sample_positions(height, width)
    if positions is None:
        return None

    random.shuffle(positions)
    placement_by_orientation = dict(zip(MISSING_CORNERS, positions))
    if not _placement_breaks_positional_shortcuts(placement_by_orientation, height, width):
        return None

    colors = list(PALETTE)
    random.shuffle(colors)
    color_by_orientation = dict(zip(MISSING_CORNERS, colors))

    grid = np.full((height, width), BG, dtype=int)
    for orientation in MISSING_CORNERS:
        r0, c0 = placement_by_orientation[orientation]
        sprite = _make_sprite(color_by_orientation[orientation], orientation)
        grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = sprite

    components = _find_monochrome_components(grid, BG)
    if len(components) != 4:
        return None

    sprite_infos = [_extract_sprite_info(grid, component, BG) for component in components]
    seed = _build_seed(sprite_infos, BG)
    if np.array_equal(seed, np.fliplr(seed)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
