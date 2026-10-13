from __future__ import annotations

import random
import sys
from itertools import combinations

import numpy as np


ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import bbox, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


MASK_DIM_RANGE = range(2, 5)
MASK_AREA_RANGE = range(3, 7)
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
COUNT_OPTIONS = [(2, 3), (3, 2), (2, 4), (4, 2)]
GENERATION_RETRIES = 500


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    return crop_bbox(mask, box).astype(bool)


def _symmetry_kind(mask: np.ndarray) -> str | None:
    trimmed = _trim_mask(np.asarray(mask, dtype=bool))
    horizontal = bool(np.array_equal(trimmed, trimmed[::-1, :]))
    vertical = bool(np.array_equal(trimmed, trimmed[:, ::-1]))
    if horizontal == vertical:
        return None
    return "H" if horizontal else "V"


def _component_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _swap_two_colors(grid: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, colors)
    out = np.array(grid, copy=True)
    a_mask = out == a
    b_mask = out == b
    out[a_mask] = b
    out[b_mask] = a
    return out


def _build_mask_bank(kind: str) -> list[np.ndarray]:
    bank: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for height in MASK_DIM_RANGE:
        for width in MASK_DIM_RANGE:
            cell_count = height * width
            for area in range(3, min(6, cell_count) + 1):
                for cells in combinations(range(cell_count), area):
                    mask = np.zeros((height, width), dtype=bool)
                    mask.flat[list(cells)] = True
                    if len(components_mask(mask, connectivity=4)) != 1:
                        continue
                    trimmed = _trim_mask(mask)
                    if _symmetry_kind(trimmed) != kind:
                        continue
                    key = (tuple(map(int, trimmed.shape)), trimmed.tobytes())
                    bank[key] = trimmed
    result = list(bank.values())
    assert result, f"no masks generated for {kind}"
    return result


MASK_BANKS = {kind: _build_mask_bank(kind) for kind in ("H", "V")}


def _analyze_objects(grid: np.ndarray) -> tuple[int, tuple[int, int], int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = tuple(sorted(int(color) for color in np.unique(g) if int(color) != bg))
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=list(fg_colors),
        return_format="dict",
    )
    assert objects, "expected foreground objects"
    assert sum(int(obj["area"]) for obj in objects) == int(np.count_nonzero(g != bg))

    horizontal_count = 0
    vertical_count = 0
    for obj in objects:
        mask = _component_mask(obj["coords"], obj["bbox"])
        kind = _symmetry_kind(mask)
        assert kind is not None, "each object must have exactly one reflection axis"
        if kind == "H":
            horizontal_count += 1
        else:
            vertical_count += 1

    return bg, fg_colors, horizontal_count, vertical_count, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count horizontal- and vertical-symmetric objects to set the tiling size, then swap the two foreground colors on odd checkerboard tiles.

    Input structure:
    - A black background contains 4-connected monochrome foreground objects in exactly two non-background colors.
    - Each object has exactly one reflection symmetry category: horizontal or vertical.
    - The whole input grid is the tile that will later be repeated.

    Rule:
    - Detect every 4-connected foreground object and classify it by horizontal or vertical reflection symmetry.
    - Let H be the number of horizontally symmetric objects and V be the number of vertically symmetric objects.
    - Tile the full input grid into an output with H columns and V rows.
    - On tile positions (i, j) where i + j is odd, swap the two foreground colors everywhere in that tile; keep even positions unchanged.

    Key insight:
    - The symmetry test does not recolor the objects directly; it supplies the two repetition counts that determine output geometry, while a second checkerboard rule controls where the two foreground colors are exchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, fg_colors, horizontal_count, vertical_count, _objects = _analyze_objects(g)
    assert horizontal_count > 0 and vertical_count > 0, "expected both symmetry types"

    tile_h, tile_w = map(int, g.shape)
    out = np.full((tile_h * vertical_count, tile_w * horizontal_count), bg, dtype=int)
    swapped = _swap_two_colors(g, fg_colors)

    for row in range(vertical_count):
        for col in range(horizontal_count):
            tile = swapped if (row + col) % 2 else g
            paste(out, tile, row * tile_h, col * tile_w)
    return out


def _sample_group_colors(count: int, colors: tuple[int, int]) -> list[int]:
    a, b = map(int, colors)
    group = [a, b]
    while len(group) < count:
        group.append(random.choice([a, b]))
    random.shuffle(group)
    return group


def _choose_canvas_shape(horizontal_count: int, vertical_count: int) -> tuple[int, int]:
    max_rows = 30 // vertical_count
    max_cols = 30 // horizontal_count
    row_hi = min(12, max_rows)
    col_hi = min(12, max_cols)
    row_lo = min(row_hi, 7 if row_hi >= 7 else max(5, row_hi))
    col_lo = min(col_hi, 7 if col_hi >= 7 else max(5, col_hi))
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _colored_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _expected_output(grid: np.ndarray, horizontal_count: int, vertical_count: int, colors: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = map(int, grid.shape)
    out = np.zeros((tile_h * vertical_count, tile_w * horizontal_count), dtype=int)
    swapped = _swap_two_colors(grid, colors)
    for row in range(vertical_count):
        for col in range(horizontal_count):
            tile = swapped if (row + col) % 2 else grid
            paste(out, tile, row * tile_h, col * tile_w)
    return out


def _validate_semantics(
    grid: np.ndarray,
    out: np.ndarray,
    *,
    expected_counts: tuple[int, int] | None = None,
) -> None:
    bg, fg_colors, horizontal_count, vertical_count, objects = _analyze_objects(grid)
    assert bg == 0, "generator must keep black as the unique background"
    assert 4 <= len(objects) <= 6, "generator must create 4-6 objects"
    assert horizontal_count >= 2 and vertical_count >= 2 and horizontal_count != vertical_count
    assert all(3 <= int(obj["area"]) <= 6 for obj in objects)

    if expected_counts is not None:
        assert (horizontal_count, vertical_count) == tuple(map(int, expected_counts))

    expected = _expected_output(grid, horizontal_count, vertical_count, fg_colors)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)
    assert not np.array_equal(_swap_two_colors(grid, fg_colors), grid), "odd tiles must visibly differ"


def _gen_attempt() -> np.ndarray | None:
    horizontal_count, vertical_count = random.choice(COUNT_OPTIONS)
    height, width = _choose_canvas_shape(horizontal_count, vertical_count)
    colors = tuple(random.sample(COLOR_POOL, 2))

    specs: list[tuple[str, int, np.ndarray]] = []
    for color in _sample_group_colors(horizontal_count, colors):
        specs.append(("H", color, random.choice(MASK_BANKS["H"])))
    for color in _sample_group_colors(vertical_count, colors):
        specs.append(("V", color, random.choice(MASK_BANKS["V"])))
    random.shuffle(specs)

    grid = np.zeros((height, width), dtype=int)
    for _kind, color, mask in specs:
        patch = _colored_patch(mask, color)
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=0)

    if int(np.count_nonzero(grid == 0)) <= int(np.count_nonzero(grid != 0)):
        return None
    if int(get_bg(grid)) != 0:
        return None

    try:
        out = main(grid)
        _validate_semantics(grid, out, expected_counts=(horizontal_count, vertical_count))
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
