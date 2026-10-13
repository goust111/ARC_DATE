from __future__ import annotations

import random
import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

BODY_COLORS = (GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON)
SIZE_RANGE = (3, 5)
TILE_COUNT_RANGE = (3, 6)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _filled_square_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for coords in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        h, w = patch.shape
        if h != w:
            continue
        if len(coords) != h * w:
            continue
        values, counts = np.unique(patch, return_counts=True)
        dominant = int(values[counts.argmax()])
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "size": int(h),
                "dominant": dominant,
            }
        )
    return objects


def _single_position(patch: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(patch == color)
    assert pts.shape[0] == 1, f"expected exactly one color-{color} marker"
    return int(pts[0, 0]), int(pts[0, 1])


def _body_color(patch: np.ndarray) -> int:
    values, counts = np.unique(patch, return_counts=True)
    keep = [(int(v), int(n)) for v, n in zip(values, counts, strict=False) if int(v) not in (RED, BLUE)]
    assert keep, "expected a body color"
    keep.sort(key=lambda item: item[1], reverse=True)
    return keep[0][0]


def _restore_tile(patch: np.ndarray, target_red: tuple[int, int]) -> np.ndarray:
    matches: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(patch, k)
        if _single_position(rotated, RED) == target_red:
            matches.append(rotated)
    assert len(matches) == 1, "tile orientation should be uniquely determined by the red marker"
    return matches[0]


def _clean_tile(patch: np.ndarray) -> tuple[int, np.ndarray]:
    body = _body_color(patch)
    blue_r, blue_c = _single_position(patch, BLUE)
    clean = patch.copy()
    clean[clean == RED] = body
    clean[clean == BLUE] = body
    return int(blue_r * patch.shape[1] + blue_c), clean


def main(grid: np.ndarray) -> np.ndarray:
    """Restore rotated marker-tiles to a common orientation and pack them by an internal order marker.

    Input structure:
    - The background is black.
    - The top-left object is a grey square reference tile containing one red pixel that marks the canonical orientation.
    - Elsewhere are several same-sized filled square tiles, each with one red orientation pixel and one blue ordering pixel on a single body color.

    Rule:
    - Find every filled square component and use the grey reference tile's red-pixel position as the target orientation.
    - Rotate each colored tile by 0/90/180/270 degrees until its red pixel matches the reference red position.
    - After restoration, sort the tiles by the blue marker's row-major position, replace both markers with the tile's body color, and place the cleaned tiles in one tight left-to-right row.

    Key insight:
    - The grey tile is not output content; it defines the canonical orientation, and the blue marker becomes meaningful only after the tile has been rotated back into that shared frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _filled_square_objects(g)
    assert objects, "expected at least one square object"

    sizes = {obj["size"] for obj in objects}
    assert len(sizes) == 1, "all square objects must share one size"

    references = [obj for obj in objects if obj["dominant"] == GREY]
    assert len(references) == 1, "expected exactly one grey reference tile"
    ref = references[0]
    ref_r0, ref_c0, _ref_r1, _ref_c1 = ref["bbox"]
    assert (ref_r0, ref_c0) == (0, 0), "reference tile must be in the top-left corner"

    target_red = _single_position(ref["patch"], RED)
    tiles = [obj for obj in objects if obj is not ref]
    assert tiles, "expected at least one colored tile"

    restored: list[tuple[int, np.ndarray]] = []
    for tile in tiles:
        patch = _restore_tile(tile["patch"], target_red)
        restored.append(_clean_tile(patch))

    keys = [key for key, _patch in restored]
    assert len(keys) == len(set(keys)), "ordering markers must be unique"
    restored.sort(key=lambda item: item[0])

    return np.concatenate([patch for _key, patch in restored], axis=1)


def _orbit4_positions(size: int) -> list[tuple[int, int]]:
    center = size // 2
    out = []
    for r in range(size):
        for c in range(size):
            if size % 2 == 1 and r == center and c == center:
                continue
            out.append((r, c))
    return out


def _canonical_tile(size: int, body: int, red_pos: tuple[int, int], blue_pos: tuple[int, int]) -> np.ndarray:
    tile = np.full((size, size), int(body), dtype=int)
    tile[red_pos] = RED
    tile[blue_pos] = BLUE
    return tile


def _reserve(occ: np.ndarray, r0: int, c0: int, size: int) -> None:
    h, w = occ.shape
    occ[max(0, r0 - 1) : min(h, r0 + size + 1), max(0, c0 - 1) : min(w, c0 + size + 1)] = True


def _free_positions(occ: np.ndarray, size: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    spots: list[tuple[int, int]] = []
    for r0 in range(h - size + 1):
        for c0 in range(w - size + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + size + 1), max(0, c0 - 1) : min(w, c0 + size + 1)].any():
                continue
            spots.append((r0, c0))
    return spots


def _expected_output(size: int, bodies: list[int], red_pos: tuple[int, int], blue_positions: list[tuple[int, int]]) -> np.ndarray:
    ordered = sorted(zip(blue_positions, bodies), key=lambda item: (item[0][0], item[0][1]))
    patches = [
        np.full((size, size), int(body), dtype=int)
        for _blue_pos, body in ordered
    ]
    return np.concatenate(patches, axis=1)


def _sample_grid() -> np.ndarray | None:
    size = random.randint(*SIZE_RANGE)
    red_pos = random.choice(_orbit4_positions(size))

    all_positions = [(r, c) for r in range(size) for c in range(size) if (r, c) != red_pos]
    max_tiles = min(TILE_COUNT_RANGE[1], len(BODY_COLORS), len(all_positions))
    count = random.randint(TILE_COUNT_RANGE[0], max_tiles)

    blue_positions = random.sample(all_positions, count)
    bodies = random.sample(list(BODY_COLORS), count)
    rotations = [random.randrange(4) for _ in range(count)]

    min_side = max(size + 2, int(np.ceil(np.sqrt(count + 1))) * (size + 1) + 1)
    if min_side > 30:
        return None
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    grid = np.full((height, width), BLACK, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    ref = np.full((size, size), GREY, dtype=int)
    ref[red_pos] = RED
    grid[:size, :size] = ref
    _reserve(occ, 0, 0, size)

    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        spots = _free_positions(occ, size)
        if not spots:
            return None
        r0, c0 = random.choice(spots)
        tile = _canonical_tile(size, bodies[idx], red_pos, blue_positions[idx])
        tile = np.rot90(tile, rotations[idx])
        grid[r0 : r0 + size, c0 : c0 + size] = tile
        _reserve(occ, r0, c0, size)

    expected = _expected_output(size, bodies, red_pos, blue_positions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _sample_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tile-rotation input")
