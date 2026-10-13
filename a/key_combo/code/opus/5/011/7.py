from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
DIVIDER = 4
CENTER = 6
COLOR_POOL = [1, 2, 3, 7, 8, 9]
SOURCE_DIRECTIONS = ("left", "up", "down")
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
CROSS_MASK = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)
CROSS_CELLS = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
CROSS_PATTERNS = [
    ("a", "a", "b", "b", "a"),
    ("a", "b", "a", "a", "b"),
    ("b", "a", "a", "b", "a"),
    ("a", "b", "b", "a", "a"),
]
GEN_ATTEMPTS = 500


def _swap_two_colors(patch: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = patch.copy()
    mask_a = out == int(color_a)
    mask_b = out == int(color_b)
    out[mask_a] = int(color_b)
    out[mask_b] = int(color_a)
    return out


def _paste_checked(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    assert target.shape == patch.shape, "patch must fit inside the destination grid"
    assert not np.any((target != int(bg)) & (patch != int(bg))), "generated copy overlap"
    paste(dst, patch, int(r0), int(c0), transparent=int(bg))


def _parse_object_patch(patch: np.ndarray, bg: int) -> dict:
    mask = patch != int(bg)
    assert int(mask.sum()) == 9, "each composite object must occupy 9 pixels"

    colors = sorted(set(int(v) for v in patch[mask]))
    assert len(colors) == 2, "each composite object must use exactly two colors"

    h, w = patch.shape
    candidates = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not np.array_equal(mask[r0 : r0 + 3, c0 : c0 + 3], CROSS_MASK):
                continue

            remainder = mask.copy()
            window = remainder[r0 : r0 + 3, c0 : c0 + 3].copy()
            window[CROSS_MASK] = False
            remainder[r0 : r0 + 3, c0 : c0 + 3] = window
            rem_coords = np.argwhere(remainder)
            if rem_coords.shape[0] != 4:
                continue

            mr0, mc0 = rem_coords.min(axis=0)
            mr1, mc1 = rem_coords.max(axis=0)
            if (int(mr1 - mr0), int(mc1 - mc0)) != (1, 1):
                continue
            if not remainder[int(mr0) : int(mr1) + 1, int(mc0) : int(mc1) + 1].all():
                continue

            marker_values = patch[int(mr0) : int(mr1) + 1, int(mc0) : int(mc1) + 1]
            if len(set(int(v) for v in marker_values.ravel())) != 1:
                continue

            if mc1 < c0:
                direction = "left"
            elif mc0 > c0 + 2:
                direction = "right"
            elif mr1 < r0:
                direction = "up"
            elif mr0 > r0 + 2:
                direction = "down"
            else:
                continue

            candidates.append(
                {
                    "direction": direction,
                    "colors": tuple(colors),
                    "cross_bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2)),
                    "marker_bbox": (int(mr0), int(mc0), int(mr1), int(mc1)),
                }
            )

    assert len(candidates) == 1, "expected one cross block and one attached 2x2 marker block"
    return candidates[0]


def _extract_sources(grid: np.ndarray) -> tuple[list[dict], int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cr, cc = h // 2, w // 2
    bg = int(get_bg(g))

    assert h % 2 == 1 and w % 2 == 1
    assert int(g[cr, cc]) == CENTER, "center marker must be magenta"
    assert np.all(g[:, cc][np.arange(h) != cr] == DIVIDER), "divider column must be yellow"
    assert not np.any(g[:, cc + 1 :] != bg), "input right side must be empty"

    work = g.copy()
    work[:, cc] = bg
    work[cr, cc] = bg

    components = find_components(work, bg=bg, connectivity=8, monochromatic=False)
    assert 2 <= len(components) <= 3, "expected 2 or 3 source objects"

    sources = []
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert c1 < cc, "all source objects must lie strictly left of the divider"
        patch = crop_bbox(work, comp["bbox"])
        info = _parse_object_patch(patch, bg)
        info.update({"bbox": (r0, c0, r1, c1), "patch": patch})
        sources.append(info)

    sources.sort(key=lambda item: item["bbox"])
    return sources, bg, cr, cc


def _copy_target_top_left(
    start_r: int,
    start_c: int,
    shape: tuple[int, int],
    direction: str,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    h, w = map(int, shape)
    gh, gw = map(int, grid_shape)
    if direction == "right":
        return int(start_r), int(gw - w)
    if direction == "left":
        return int(start_r), 0
    if direction == "up":
        return 0, int(start_c)
    if direction == "down":
        return int(gh - h), int(start_c)
    raise ValueError(f"unknown direction: {direction}")


def _apply_rule(
    grid: np.ndarray,
    *,
    swap_colors: bool = True,
    slide: bool = True,
    rotate_cross: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sources, bg, cr, cc = _extract_sources(g)

    for source in sources:
        r0, c0, r1, c1 = source["bbox"]
        reflected = np.rot90(source["patch"], 2)
        copy_r0 = int(2 * cr - r1)
        copy_c0 = int(2 * cc - c1)
        reflected_info = _parse_object_patch(reflected, bg)

        transformed = reflected
        if swap_colors:
            color_a, color_b = source["colors"]
            transformed = _swap_two_colors(transformed, color_a, color_b)

        if rotate_cross:
            xr0, xc0, xr1, xc1 = reflected_info["cross_bbox"]
            transformed[xr0 : xr1 + 1, xc0 : xc1 + 1] = np.rot90(
                transformed[xr0 : xr1 + 1, xc0 : xc1 + 1],
                2,
            )

        target_r0, target_c0 = copy_r0, copy_c0
        if slide:
            target_r0, target_c0 = _copy_target_top_left(
                copy_r0,
                copy_c0,
                transformed.shape,
                reflected_info["direction"],
                g.shape,
            )

        _paste_checked(out, transformed, target_r0, target_c0, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each left-side cross object by point reflection, then swap its two colors and slide the reflected copy to the edge in the marker direction.

    Input structure:
    - The background is black, the center column is a yellow divider, and the
      center cell is a magenta rotation marker.
    - Two or three 8-connected foreground composites lie entirely on the left of
      the divider; the right side is otherwise empty.
    - Each composite is a 9-pixel two-color object made from a 5-cell cross in a
      3x3 window plus a solid 2x2 marker block attached on one side.

    Rule:
    - Keep the divider column, center marker, and all left-side source objects unchanged.
    - For each source object, create its 180-degree point-reflected copy across the
      center marker on the right side.
    - In that reflected copy, swap the object's two colors, rotate only the 3x3
      cross block by 180 degrees, and then translate the whole reflected object in
      the marker-block direction until its bbox touches the corresponding grid edge.

    Key insight:
    The same 180-degree completion step both creates the right-side object and flips
    its marker direction, so the reflected copy is the only object that gets color
    swapping, payload rotation, and edge-directed sliding.
    """
    return _apply_rule(grid)


def _build_cross_block(color_a: int, color_b: int, pattern: tuple[str, ...]) -> np.ndarray:
    block = np.full((3, 3), BG, dtype=int)
    for (r, c), token in zip(CROSS_CELLS, pattern):
        block[r, c] = int(color_a if token == "a" else color_b)
    assert not np.array_equal(block, np.rot90(block, 2)), "cross pattern must make 180-degree rotation visible"
    return block


def _build_source_patch(direction: str, color_a: int, color_b: int, pattern: tuple[str, ...]) -> np.ndarray:
    cross = _build_cross_block(color_a, color_b, pattern)
    if direction == "left":
        patch = np.full((3, 5), BG, dtype=int)
        patch[:, 2:5] = cross
        patch[0:2, 0:2] = int(color_b)
        return patch
    if direction == "up":
        patch = np.full((5, 3), BG, dtype=int)
        patch[2:5, :] = cross
        patch[0:2, 0:2] = int(color_b)
        return patch
    if direction == "down":
        patch = np.full((5, 3), BG, dtype=int)
        patch[0:3, :] = cross
        patch[3:5, 0:2] = int(color_b)
        return patch
    raise ValueError(f"unsupported source direction: {direction}")


def _padded_clear(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = patch.shape
    oh, ow = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > oh or c0 + w > ow:
        return False
    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - pad)
        r2 = min(oh, rr + pad + 1)
        c1 = max(0, cc - pad)
        c2 = min(ow, cc + pad + 1)
        if np.any(occupied[r1:r2, c1:c2]):
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, int(r0), int(c0), transparent=BG)


def _right_component_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int).copy()
    h, w = g.shape
    cc = w // 2
    g[:, : cc + 1] = BG
    return len(find_components(g, bg=BG, connectivity=8, monochromatic=False))


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        sources, _bg, _cr, cc = _extract_sources(inp)
        out = main(inp)
        if np.array_equal(out, inp):
            return False
        if not np.array_equal(out[:, :cc], inp[:, :cc]):
            return False
        if not np.array_equal(out[:, cc], inp[:, cc]):
            return False
        if _right_component_count(out) != len(sources):
            return False
        for source in sources:
            xr0, xc0, xr1, xc1 = source["cross_bbox"]
            cross = source["patch"][xr0 : xr1 + 1, xc0 : xc1 + 1]
            if np.array_equal(cross, np.rot90(cross, 2)):
                return False
            if len(set(int(v) for v in cross[cross != BG])) != 2:
                return False

        for alt in (
            _apply_rule(inp, swap_colors=False),
            _apply_rule(inp, slide=False),
            _apply_rule(inp, rotate_cross=False),
        ):
            if np.array_equal(out, alt):
                return False
        return True
    except AssertionError:
        return False


def _random_grid_shape() -> tuple[int, int]:
    return random.choice([13, 15, 17, 19]), random.choice([19, 21, 23, 25])


def _random_specs() -> list[dict]:
    n_obj = random.choice([2, 3])
    directions = [random.choice(SOURCE_DIRECTIONS) for _ in range(n_obj)]
    if len(set(directions)) == 1:
        directions[random.randrange(n_obj)] = random.choice([d for d in SOURCE_DIRECTIONS if d != directions[0]])

    pairs = random.sample([(a, b) for a in COLOR_POOL for b in COLOR_POOL if a != b], n_obj)
    specs = []
    for direction, (color_a, color_b) in zip(directions, pairs):
        specs.append(
            {
                "direction": direction,
                "color_a": int(color_a),
                "color_b": int(color_b),
                "pattern": random.choice(CROSS_PATTERNS),
            }
        )
    return specs


def _candidate_positions(shape: tuple[int, int], direction: str, grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    gh, gw = map(int, grid_shape)
    cc = gw // 2

    rows = list(range(gh - h + 1))
    cols = list(range(cc - w))
    if direction == "left":
        cols = [c for c in cols if c > 0]
    elif direction == "up":
        rows = [r for r in rows if r > 0]
    elif direction == "down":
        rows = [r for r in rows if r < gh - h]

    positions = [(r, c) for r in rows for c in cols]
    random.shuffle(positions)
    return positions


def _gen_attempt() -> np.ndarray | None:
    h, w = _random_grid_shape()
    grid = np.full((h, w), BG, dtype=int)
    cr, cc = h // 2, w // 2
    grid[:, cc] = DIVIDER
    grid[cr, cc] = CENTER

    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, cc] = True

    specs = _random_specs()
    for spec in specs:
        patch = _build_source_patch(spec["direction"], spec["color_a"], spec["color_b"], spec["pattern"])
        positions = _candidate_positions(patch.shape, spec["direction"], grid.shape)
        placed = False
        for r0, c0 in positions:
            if not _padded_clear(occupied, patch, r0, c0, pad=1):
                continue
            _place_patch(grid, patch, r0, c0)
            occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch != BG
            placed = True
            break
        if not placed:
            return None

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid point-symmetry slide puzzle")
