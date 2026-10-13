from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
PALETTE = (1, 2, 3, 6, 7, 8, 9)

GRID_SIDE_RANGE = (11, 15)
SPRITE_CELL_RANGE = (8, 15)
SPRITE_BBOX_SIDES = range(3, 7)
COLOR_COUNT_OPTIONS = (2, 3)
OVERLAP_RATIO_MIN = 0.30
OVERLAP_RATIO_MAX = 0.70
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components4(mask)
    return len(comps) == 1


def _bbox_shape(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _bbox_overlap_ratio(h: int, w: int, shift: tuple[int, int]) -> float:
    dr, dc = shift
    overlap_h = h - abs(dr)
    overlap_w = w - abs(dc)
    if overlap_h <= 0 or overlap_w <= 0:
        return 0.0
    return (overlap_h * overlap_w) / float(h * w)


def _parse_input(grid: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int], dict[tuple[int, int], int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    gray_coords = np.argwhere(g == GRAY)
    assert gray_coords.shape[0] == 2, "expected exactly two gray markers"
    comps = _components4(g != bg)
    assert len(comps) == 2, "expected exactly one sprite component and one isolated target marker"

    source_anchor: tuple[int, int] | None = None
    target_anchor: tuple[int, int] | None = None
    source_pixels: dict[tuple[int, int], int] | None = None

    for comp in comps:
        coords = [tuple(map(int, rc)) for rc in comp]
        gray_cells = [cell for cell in coords if int(g[cell]) == GRAY]
        non_gray_cells = [cell for cell in coords if int(g[cell]) != GRAY]

        if gray_cells and non_gray_cells:
            assert source_anchor is None, "expected a single mixed source component"
            assert len(gray_cells) == 1, "source component must contain one gray anchor"
            source_anchor = gray_cells[0]
            source_pixels = {cell: int(g[cell]) for cell in non_gray_cells}
        elif len(coords) == 1 and gray_cells:
            assert target_anchor is None, "expected a single isolated target anchor"
            target_anchor = gray_cells[0]
        else:
            raise AssertionError("unexpected non-background component structure")

    assert source_anchor is not None and source_pixels is not None, "missing source sprite"
    assert target_anchor is not None, "missing isolated target anchor"

    source_occupied = set(source_pixels) | {source_anchor}
    assert any(nei in source_pixels for nei in _neighbors4(*source_anchor)), "source anchor must sit inside the sprite"
    assert target_anchor not in source_occupied, "target anchor must not overlap the source sprite"
    assert all(nei not in source_occupied for nei in _neighbors4(*target_anchor)), "target anchor must be isolated"

    dr = int(target_anchor[0] - source_anchor[0])
    dc = int(target_anchor[1] - source_anchor[1])
    return bg, source_anchor, target_anchor, source_pixels, (dr, dc)


def _paint_symmetric_difference(
    shape: tuple[int, int],
    source_pixels: dict[tuple[int, int], int],
    shift: tuple[int, int],
    bg: int,
) -> np.ndarray:
    dr, dc = shift
    translated: dict[tuple[int, int], int] = {}
    for (r, c), color in source_pixels.items():
        rr = r + dr
        cc = c + dc
        assert _in_bounds(shape, rr, cc), "translated sprite left the grid"
        translated[(rr, cc)] = color

    out = np.full(shape, int(bg), dtype=int)
    source_positions = set(source_pixels)
    translated_positions = set(translated)
    for pos in source_positions | translated_positions:
        if pos in source_positions and pos in translated_positions:
            continue
        if pos in source_pixels:
            out[pos] = int(source_pixels[pos])
        else:
            out[pos] = int(translated[pos])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the sprite by the gray-anchor offset, then erase every pixel where the original and copy overlap.

    Input structure:
    - Black is the background.
    - One 4-connected multicolor sprite contains exactly one gray anchor pixel.
    - A second gray pixel is isolated elsewhere on the background.

    Rule:
    - Treat the gray pixel inside the sprite as the source anchor and the isolated gray pixel as the target anchor.
    - Compute the translation from source anchor to target anchor.
    - Erase both gray anchors to black.
    - Keep the source sprite's non-gray pixels in place, copy those same non-gray pixels by the anchor offset, and erase any position occupied by both the original and the translated copy.

    Key insight:
    - The two gray pixels define a partial-overlap translation, and the output is the symmetric difference of the original sprite and its translated copy.
    """
    g = np.asarray(grid, dtype=int)
    bg, _source_anchor, _target_anchor, source_pixels, shift = _parse_input(g)
    translated_positions = {(r + shift[0], c + shift[1]) for r, c in source_pixels}
    assert set(source_pixels) & translated_positions, "the copy must partially overlap the original"
    return _paint_symmetric_difference(g.shape, source_pixels, shift, bg)


def _candidate_shifts(h: int, w: int) -> list[tuple[int, int]]:
    shifts: list[tuple[int, int]] = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            ratio = _bbox_overlap_ratio(h, w, (dr, dc))
            if OVERLAP_RATIO_MIN <= ratio <= OVERLAP_RATIO_MAX:
                shifts.append((dr, dc))
    return shifts


def _gen_connected_mask(h: int, w: int, total: int, rng: random.Random) -> np.ndarray | None:
    mask = np.ones((h, w), dtype=bool)
    while int(mask.sum()) > total:
        removable: list[tuple[int, int]] = []
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        rng.shuffle(coords)
        for r, c in coords:
            trial = mask.copy()
            trial[r, c] = False
            if not trial[0].any() or not trial[-1].any() or not trial[:, 0].any() or not trial[:, -1].any():
                continue
            if _is_connected(trial):
                removable.append((r, c))
        if not removable:
            return None
        rr, cc = rng.choice(removable)
        mask[rr, cc] = False
    return mask


def _overlap_pairs(cells: set[tuple[int, int]], shift: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    dr, dc = shift
    return sorted(
        [((r, c), (r + dr, c + dc)) for r, c in cells if (r + dr, c + dc) in cells]
    )


def _anchor_options(mask: np.ndarray, shift: tuple[int, int]) -> list[dict[str, object]]:
    occupied = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    options: list[dict[str, object]] = []
    for anchor in sorted(occupied):
        if sum(nei in occupied for nei in _neighbors4(*anchor)) < 2:
            continue
        target = (anchor[0] + shift[0], anchor[1] + shift[1])
        if target in occupied:
            continue
        if any(nei in occupied for nei in _neighbors4(*target)):
            continue

        non_gray = occupied - {anchor}
        pairs = _overlap_pairs(non_gray, shift)
        if len(pairs) < 2:
            continue

        copied = {(r + shift[0], c + shift[1]) for r, c in non_gray}
        overlap = non_gray & copied
        orig_only = non_gray - copied
        copy_only = copied - non_gray
        if len(overlap) < 2 or len(orig_only) < 2 or len(copy_only) < 2:
            continue

        options.append(
            {
                "anchor": anchor,
                "pairs": pairs,
                "overlap": overlap,
                "orig_only": orig_only,
                "copy_only": copy_only,
            }
        )
    return options


def _assign_colors(
    non_gray: set[tuple[int, int]],
    overlap_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    rng: random.Random,
) -> dict[tuple[int, int], int]:
    cells = sorted(non_gray)
    for _ in range(80):
        color_count = rng.choice(COLOR_COUNT_OPTIONS)
        palette = rng.sample(PALETTE, color_count)
        colors = {cell: rng.choice(palette) for cell in cells}
        if len(set(colors.values())) < 2:
            continue
        same_flags = [colors[a] == colors[b] for a, b in overlap_pairs]
        if any(same_flags) and not all(same_flags):
            return colors

    palette = rng.sample(PALETTE, 3)
    colors = {cell: palette[i % len(palette)] for i, cell in enumerate(cells)}
    colors[overlap_pairs[0][0]] = palette[0]
    colors[overlap_pairs[0][1]] = palette[0]
    colors[overlap_pairs[1][0]] = palette[1]
    colors[overlap_pairs[1][1]] = palette[2]
    return colors


def _grid_shape_and_origin(h: int, w: int, shift: tuple[int, int], rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]] | None:
    dr, dc = shift
    min_h = max(GRID_SIDE_RANGE[0], h + abs(dr))
    min_w = max(GRID_SIDE_RANGE[0], w + abs(dc))
    max_h = max(min_h, GRID_SIDE_RANGE[1] - 1)
    max_w = max(min_w, GRID_SIDE_RANGE[1] - 1)
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = rng.randint(min_h, max_h)
    grid_w = rng.randint(min_w, max_w)

    r0_min = max(0, -dr)
    r0_max = min(grid_h - h, grid_h - h - dr)
    c0_min = max(0, -dc)
    c0_max = min(grid_w - w, grid_w - w - dc)
    if r0_min > r0_max or c0_min > c0_max:
        return None

    origin = (rng.randint(r0_min, r0_max), rng.randint(c0_min, c0_max))
    return (grid_h, grid_w), origin


def _build_input(
    shape: tuple[int, int],
    origin: tuple[int, int],
    anchor: tuple[int, int],
    colors: dict[tuple[int, int], int],
    shift: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    r0, c0 = origin
    for (r, c), color in colors.items():
        grid[r0 + r, c0 + c] = int(color)
    grid[r0 + anchor[0], c0 + anchor[1]] = GRAY
    grid[r0 + anchor[0] + shift[0], c0 + anchor[1] + shift[1]] = GRAY
    return grid


def _semantic_validation(grid: np.ndarray) -> bool:
    bg, source_anchor, target_anchor, source_pixels, shift = _parse_input(grid)
    assert bg == BG
    assert source_anchor != target_anchor

    source_occupied = set(source_pixels) | {source_anchor}
    assert SPRITE_CELL_RANGE[0] <= len(source_occupied) <= SPRITE_CELL_RANGE[1]
    assert 2 <= len(set(source_pixels.values())) <= 3
    assert sum(nei in source_occupied for nei in _neighbors4(*source_anchor)) >= 2
    bbox_h, bbox_w = _bbox_shape(source_occupied)
    ratio = _bbox_overlap_ratio(bbox_h, bbox_w, shift)
    assert OVERLAP_RATIO_MIN <= ratio <= OVERLAP_RATIO_MAX

    copied = {(r + shift[0], c + shift[1]): color for (r, c), color in source_pixels.items()}
    source_positions = set(source_pixels)
    copied_positions = set(copied)
    overlap = source_positions & copied_positions
    orig_only = source_positions - copied_positions
    copy_only = copied_positions - source_positions
    assert len(overlap) >= 2
    assert len(orig_only) >= 2
    assert len(copy_only) >= 2

    pair_colors = [source_pixels[a] == source_pixels[b] for a, b in _overlap_pairs(source_positions, shift)]
    assert any(pair_colors) and not all(pair_colors)

    out = _paint_symmetric_difference(grid.shape, source_pixels, shift, bg)
    union = np.full(grid.shape, bg, dtype=int)
    for (r, c), color in source_pixels.items():
        union[r, c] = color
        union[r + shift[0], c + shift[1]] = color
    original_only = np.full(grid.shape, bg, dtype=int)
    translated_only = np.full(grid.shape, bg, dtype=int)
    for (r, c), color in source_pixels.items():
        original_only[r, c] = color
        translated_only[r + shift[0], c + shift[1]] = color

    assert not np.array_equal(out, union)
    assert not np.array_equal(out, original_only)
    assert not np.array_equal(out, translated_only)
    assert np.array_equal(main(grid), out)
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    total_cells = rng.randint(SPRITE_CELL_RANGE[0], SPRITE_CELL_RANGE[1])
    bbox_options = [
        (h, w)
        for h in SPRITE_BBOX_SIDES
        for w in SPRITE_BBOX_SIDES
        if total_cells <= h * w and _candidate_shifts(h, w)
    ]
    rng.shuffle(bbox_options)

    for h, w in bbox_options:
        mask = _gen_connected_mask(h, w, total_cells, rng)
        if mask is None:
            continue

        shifts = _candidate_shifts(h, w)
        rng.shuffle(shifts)
        for shift in shifts:
            anchor_choices = _anchor_options(mask, shift)
            if not anchor_choices:
                continue
            choice = rng.choice(anchor_choices)
            anchor = choice["anchor"]
            colors = _assign_colors(mask_to_non_gray(mask, anchor), choice["pairs"], rng)
            placed = _grid_shape_and_origin(h, w, shift, rng)
            if placed is None:
                continue
            shape, origin = placed
            grid = _build_input(shape, origin, anchor, colors, shift)
            try:
                if _semantic_validation(grid):
                    return grid
            except AssertionError:
                continue
    return None


def mask_to_non_gray(mask: np.ndarray, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    occupied = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    return occupied - {anchor}


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
