from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
MASK_COLOR = 1
RED = 2
GREEN = 3
YELLOW = 4
SEPARATOR = 5
PALETTE_COLORS = (RED, GREEN, YELLOW)
TILE = 3
MASK_SIDE = 3
REGION_SIDE = TILE * MASK_SIDE

GRID_H_RANGE = (15, 20)
GRID_W_RANGE = (11, 18)
MIN_PADDING = 1
GEN_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_mask(region: np.ndarray) -> np.ndarray:
    assert region.shape == (REGION_SIDE, REGION_SIDE), "expected a 9x9 encoded region"
    mask = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            tile = region[br * TILE : (br + 1) * TILE, bc * TILE : (bc + 1) * TILE]
            value = int(tile[0, 0])
            assert value in {BG, MASK_COLOR}, "source tiles may use only black and blue"
            assert np.all(tile == value), "each 3x3 source tile must be uniform"
            mask[br, bc] = value == MASK_COLOR
    assert int(mask.sum()) >= 3, "mask must contain at least three ON cells"
    return mask


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_box = _bbox(g == SEPARATOR)
    assert sep_box is not None, "expected one grey separator"
    r0, c0, r1, c1 = sep_box
    assert (r0, c0, r1, c1) == (r0, c0, r0, c0 + REGION_SIDE - 1), "separator must be a 1x9 segment"
    assert np.count_nonzero(g == SEPARATOR) == REGION_SIDE, "grey must appear only in the separator"
    assert r0 >= REGION_SIDE, "separator must have a 9x9 region above it"
    assert r0 + 1 + MASK_SIDE <= g.shape[0], "separator must have a 3x3 palette below it"
    assert c0 + MASK_SIDE <= g.shape[1], "palette must fit under the separator"

    region = g[r0 - REGION_SIDE : r0, c0 : c1 + 1]
    palette = g[r0 + 1 : r0 + 1 + MASK_SIDE, c0 : c0 + MASK_SIDE]
    assert region.shape == (REGION_SIDE, REGION_SIDE)
    assert palette.shape == (MASK_SIDE, MASK_SIDE)
    assert set(map(int, np.unique(palette))).issubset(set((BG, *PALETTE_COLORS))), "palette uses invalid colors"

    mask = _decode_mask(region)
    active_colors = [int(v) for v in palette.flatten() if int(v) != BG]
    assert len(active_colors) >= 2, "palette must activate at least two cells"
    assert len(set(active_colors)) == len(active_colors), "active palette colors must be distinct"
    for rr in range(MASK_SIDE):
        for cc in range(MASK_SIDE):
            if int(palette[rr, cc]) != BG:
                assert mask[rr, cc], "palette may be non-black only on ON cells of the mask"
    return mask, palette


def _render_output(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = np.zeros((REGION_SIDE, REGION_SIDE), dtype=int)
    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            color = int(palette[br, bc])
            if mask[br, bc] and color != BG:
                out[br * TILE : (br + 1) * TILE, bc * TILE : (bc + 1) * TILE] = np.where(mask, color, BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the enlarged binary mask, use the palette to select active macro cells, and stamp the same mask inside each active block.

    Input structure:
    - A black background contains a 9x9 blue/black block-encoded region made of nine uniform 3x3 tiles.
    - A single grey horizontal separator of length 9 sits directly below that region.
    - Directly under the separator's left edge is a 3x3 palette using black, red, green, and yellow.

    Rule:
    - Compress the 9x9 upper region into a 3x3 binary mask M by reading each uniform 3x3 tile as ON for blue and OFF for black.
    - Read the 3x3 palette C below the separator.
    - For each macro position (r,c), activate it only when M[r,c] is ON and C[r,c] is non-black.
    - Build a 9x9 output whose active 3x3 macro blocks contain the shape of M colored by C[r,c], while silent blocks stay black.

    Key insight:
    - The decoded mask plays two roles at once: it filters which macro positions can be active and it is also the micro-level stamp repeated inside every active block.
    """
    mask, palette = _parse_grid(grid)
    return _render_output(mask, palette)


def _mask_library() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (MASK_SIDE * MASK_SIDE)):
        mask = np.array([(bits >> i) & 1 for i in range(MASK_SIDE * MASK_SIDE)], dtype=bool).reshape(MASK_SIDE, MASK_SIDE)
        on_count = int(mask.sum())
        if not 4 <= on_count <= 7:
            continue
        if not np.any(mask[0]) or not np.any(mask[2]) or not np.any(mask[:, 0]) or not np.any(mask[:, 2]):
            continue
        masks.append(mask)
    return masks


MASK_LIBRARY = _mask_library()


def _valid_active_sets(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    on_positions = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    active_sets: list[list[tuple[int, int]]] = []
    for size in (2, 3):
        if size > len(on_positions):
            continue
        for combo in itertools.combinations(on_positions, size):
            rows = {r for r, _ in combo}
            cols = {c for _, c in combo}
            if len(rows) < 2 or len(cols) < 2:
                continue
            active_sets.append(list(combo))
    return active_sets


def _count_output_explanations(out: np.ndarray) -> int:
    out = np.asarray(out, dtype=int)
    visible_colors = {int(v) for v in np.unique(out) if int(v) != BG}
    if not 2 <= len(visible_colors) <= 3:
        return 0

    solutions = 0
    for bits in range(1 << (MASK_SIDE * MASK_SIDE)):
        mask = np.array([(bits >> i) & 1 for i in range(MASK_SIDE * MASK_SIDE)], dtype=bool).reshape(MASK_SIDE, MASK_SIDE)
        if int(mask.sum()) < 3:
            continue

        used_colors: set[int] = set()
        valid = True
        for br in range(MASK_SIDE):
            for bc in range(MASK_SIDE):
                block = out[br * TILE : (br + 1) * TILE, bc * TILE : (bc + 1) * TILE]
                if not np.any(block):
                    continue
                matched_color = None
                for color in visible_colors:
                    if np.array_equal(block, np.where(mask, color, BG)):
                        matched_color = int(color)
                        break
                if matched_color is None or matched_color in used_colors or not mask[br, bc]:
                    valid = False
                    break
                used_colors.add(matched_color)
            if not valid:
                break
        if valid and used_colors == visible_colors:
            solutions += 1
    return solutions


def _build_instance(mask: np.ndarray, active_cells: list[tuple[int, int]]) -> np.ndarray:
    palette = np.zeros((MASK_SIDE, MASK_SIDE), dtype=int)
    colors = random.sample(list(PALETTE_COLORS), k=len(active_cells))
    random.shuffle(colors)
    for (r, c), color in zip(active_cells, colors):
        palette[r, c] = int(color)

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    top = random.randint(MIN_PADDING, h - (REGION_SIDE + 1 + MASK_SIDE) - MIN_PADDING)
    left = random.randint(MIN_PADDING, w - REGION_SIDE - MIN_PADDING)

    grid = np.zeros((h, w), dtype=int)
    region = np.kron(mask.astype(int), np.ones((TILE, TILE), dtype=int)) * MASK_COLOR
    grid[top : top + REGION_SIDE, left : left + REGION_SIDE] = region
    grid[top + REGION_SIDE, left : left + REGION_SIDE] = SEPARATOR
    grid[top + REGION_SIDE + 1 : top + REGION_SIDE + 1 + MASK_SIDE, left : left + MASK_SIDE] = palette
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        mask, palette = _parse_grid(grid)
    except AssertionError:
        return False

    active_cells = [tuple(map(int, rc)) for rc in np.argwhere(palette != BG)]
    if len(active_cells) not in {2, 3}:
        return False
    rows = {r for r, _ in active_cells}
    cols = {c for _, c in active_cells}
    if len(rows) < 2 or len(cols) < 2:
        return False
    if not np.any(mask == 0):
        return False

    # Most examples include ON-but-silent cells so palette filtering cannot be skipped.
    silent_on_count = int(mask.sum()) - len(active_cells)
    if silent_on_count < 0:
        return False

    out = main(grid)
    if out.shape != (REGION_SIDE, REGION_SIDE):
        return False
    if _count_output_explanations(out) != 1:
        return False

    # Reject shortcuts: solid active blocks would produce a different result,
    # and in most cases promoting every ON macro cell would also differ.
    solid = np.zeros_like(out)
    for r, c in active_cells:
        solid[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = int(palette[r, c])
    if np.array_equal(out, solid):
        return False
    if silent_on_count > 0:
        forced = np.zeros_like(out)
        for r, c in np.argwhere(mask):
            color = int(palette[r, c]) if int(palette[r, c]) != BG else YELLOW
            forced[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = np.where(mask, color, BG)
        if np.array_equal(out, forced):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        mask = random.choice(MASK_LIBRARY)
        active_sets = _valid_active_sets(mask)
        if not active_sets:
            continue

        silent_candidates = [cells for cells in active_sets if len(cells) < int(mask.sum())]
        if silent_candidates and random.random() < 0.8:
            active_cells = random.choice(silent_candidates)
        else:
            active_cells = random.choice(active_sets)

        grid = _build_instance(mask, active_cells)
        if _semantic_checks(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
