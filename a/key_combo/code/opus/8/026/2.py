from __future__ import annotations

from collections import Counter
import random

import numpy as np


MASK_COLORS = (1, 2, 3)
REPLACEMENT_COLORS = (4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _parse_scene(grid: np.ndarray) -> tuple[int, np.ndarray, list[tuple[int, int]], dict[int, int], list[tuple[int, int]]]:
    bg = _get_bg(grid)
    comps = _components_from_mask(grid != bg)
    assert comps, "expected at least one foreground component"

    comps.sort(key=len, reverse=True)
    mask_coords = comps[0]
    assert len(mask_coords) > 2, "expected one large foreground mask"

    mask_bool = np.zeros_like(grid, dtype=bool)
    for r, c in mask_coords:
        mask_bool[r, c] = True

    mapping: dict[int, int] = {}
    legend_cells: list[tuple[int, int]] = []
    for coords in comps[1:]:
        assert len(coords) == 2, "all non-mask components must be two-pixel legends"
        r0, c0, r1, c1 = _bbox_coords(coords)
        assert r0 == r1 and c1 == c0 + 1, "legends must be horizontal dominos"
        replacement = int(grid[r0, c0])
        source = int(grid[r0, c1])
        assert replacement != source, "legend colors must differ"
        assert source not in mapping, "legend source colors must be unique"
        mapping[source] = replacement
        legend_cells.extend(coords)

    return bg, mask_bool, mask_coords, mapping, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor only the horizontally mirrored same-color mask pixels using a legend, then erase the legend.

    Input structure:
    - The grid has a black background, one large 4-connected multicolor foreground mask,
      and separate horizontal 1x2 legend dominos outside that mask.
    - The mask uses colors 1/2/3, the grid width is even, and each legend reads
      left-to-right as replacement color then source color.

    Rule:
    - Use the whole grid's vertical midline as the reflection axis.
    - For each pixel of the large mask, check its same-row mirror position.
    - If the mirror position is also inside the large mask and has the same color,
      that pixel is paired; paired pixels are recolored only when their color appears
      as a legend source, otherwise they stay unchanged.
    - Unpaired mask pixels stay unchanged, and every legend pixel is erased to black.

    Key insight:
    - Recoloring is doubly gated: geometry decides which mask pixels qualify, and the
      legends only choose the replacement color for already-qualified pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w % 2 == 0, "expected an even-width grid"

    bg, mask_bool, mask_coords, mapping, legend_cells = _parse_scene(g)

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = bg

    for r, c in mask_coords:
        mirror_c = w - 1 - c
        color = int(g[r, c])
        if mask_bool[r, mirror_c] and int(g[r, mirror_c]) == color:
            out[r, c] = int(mapping.get(color, color))

    return out


def _generate_mask_rows(width: int, rng: random.Random) -> list[tuple[int, int]]:
    half = width // 2
    max_len = min(half, 5)
    for _ in range(200):
        row_count = rng.randint(5, 8)
        left = rng.randint(2, max_len)
        right = rng.randint(2, max_len)
        rows: list[tuple[int, int]] = []
        for _row in range(row_count):
            left = max(1, min(max_len, left + rng.choice((-1, 0, 1))))
            right = max(1, min(max_len, right + rng.choice((-1, 0, 1))))
            rows.append((left, right))

        area = sum(left_len + right_len for left_len, right_len in rows)
        pair_count = sum(min(left_len, right_len) for left_len, right_len in rows)
        unpaired_count = sum(abs(left_len - right_len) for left_len, right_len in rows)
        asymmetric_rows = sum(left_len != right_len for left_len, right_len in rows)
        if not (25 <= area <= 40):
            continue
        if pair_count < 5 or unpaired_count < 3 or asymmetric_rows == 0:
            continue
        return rows
    raise RuntimeError("failed to generate a non-degenerate mask shape")


def _build_mask(shape: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    h, w = shape
    rows = _generate_mask_rows(w, rng)
    top = rng.randint(1, h - len(rows) - 1)
    mid_left = w // 2 - 1
    mid_right = w // 2

    mask = np.zeros((h, w), dtype=bool)
    for offset, (left_len, right_len) in enumerate(rows):
        r = top + offset
        c0 = mid_left - left_len + 1
        c1 = mid_right + right_len - 1
        mask[r, c0 : c1 + 1] = True

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    pair_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    unpaired: list[tuple[int, int]] = []
    for r, c in coords:
        mc = w - 1 - c
        if mask[r, mc]:
            if c < mc:
                pair_pairs.append(((r, c), (r, mc)))
        else:
            unpaired.append((r, c))

    assert len(_components_from_mask(mask)) == 1
    assert len(pair_pairs) >= 5 and len(unpaired) >= 3
    return mask, coords, pair_pairs, unpaired


def _assign_mask_colors(
    mask: np.ndarray,
    pair_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    unpaired: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, dict[int, int]]:
    grid = np.zeros(mask.shape, dtype=int)
    grid[mask] = rng.choice(MASK_COLORS)

    shuffled_pairs = pair_pairs[:]
    rng.shuffle(shuffled_pairs)
    shuffled_unpaired = unpaired[:]
    rng.shuffle(shuffled_unpaired)

    unmapped_color = rng.choice(MASK_COLORS)
    mapped_colors = [color for color in MASK_COLORS if color != unmapped_color]
    fixed: dict[tuple[int, int], int] = {}

    for color, pair in zip(MASK_COLORS, shuffled_pairs[:3]):
        a, b = pair
        fixed[a] = color
        fixed[b] = color

    mismatch_pair = shuffled_pairs[3]
    mismatch_colors = list(MASK_COLORS)
    rng.shuffle(mismatch_colors)
    fixed[mismatch_pair[0]] = mismatch_colors[0]
    fixed[mismatch_pair[1]] = mismatch_colors[1]

    for color, coord in zip(mapped_colors, shuffled_unpaired[:2]):
        fixed[coord] = color

    for coord, color in fixed.items():
        grid[coord] = color

    remaining = [coord for coord in np.argwhere(mask) if tuple(map(int, coord)) not in fixed]
    for coord in remaining:
        r, c = map(int, coord)
        grid[r, c] = rng.choice(MASK_COLORS)

    mapped_sources = mapped_colors[:]
    rng.shuffle(mapped_sources)
    replacements = rng.sample([c for c in REPLACEMENT_COLORS if c not in MASK_COLORS], len(mapped_sources))
    mapping = {int(source): int(replacement) for source, replacement in zip(mapped_sources, replacements)}

    paired_same = {color: 0 for color in MASK_COLORS}
    unpaired_counts = {color: 0 for color in MASK_COLORS}
    mismatched_pairs = 0
    w = mask.shape[1]
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        mc = w - 1 - cc
        color = int(grid[rr, cc])
        if mask[rr, mc]:
            if int(grid[rr, mc]) == color:
                paired_same[color] += 1
            elif cc < mc:
                mismatched_pairs += 1
        else:
            unpaired_counts[color] += 1

    for color in mapped_sources:
        assert paired_same[color] >= 2, "mapped color needs at least one same-color mirrored pair"
        assert unpaired_counts[color] >= 1, "mapped color needs at least one unpaired pixel"
    assert paired_same[unmapped_color] >= 2, "unmapped color needs a paired branch"
    assert mismatched_pairs >= 1, "need a mirrored presence pair with mismatched colors"
    assert set(int(v) for v in grid[mask]) == set(MASK_COLORS), "all three mask colors must appear"
    return grid, mapping


def _legend_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    occupied = grid != 0
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w - 1):
            if occupied[r, c] or occupied[r, c + 1]:
                continue
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 3)
            window = occupied[r0:r1, c0:c1].copy()
            window[r - r0, c - c0] = False
            window[r - r0, c + 1 - c0] = False
            if not window.any():
                candidates.append((r, c))
    return candidates


def _place_legends(grid: np.ndarray, mapping: dict[int, int], rng: random.Random) -> bool:
    items = list(mapping.items())
    rng.shuffle(items)
    for source, replacement in items:
        candidates = _legend_positions(grid)
        if not candidates:
            return False
        r, c = rng.choice(candidates)
        grid[r, c] = int(replacement)
        grid[r, c + 1] = int(source)
    return True


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    bg, mask_bool, mask_coords, mapping, legend_cells = _parse_scene(inp)
    h, w = inp.shape

    for r, c in legend_cells:
        assert int(out[r, c]) == bg, "legend cells must be erased in the output"

    paired_recolored = 0
    paired_unmapped = 0
    mapped_unpaired_unchanged = {color: False for color in mapping}
    mismatched_unchanged = False

    for r, c in mask_coords:
        mc = w - 1 - c
        color = int(inp[r, c])
        mirror_inside = bool(mask_bool[r, mc])
        same_color = mirror_inside and int(inp[r, mc]) == color

        if same_color and color in mapping:
            assert int(out[r, c]) == int(mapping[color]), "paired mapped pixels must recolor"
            paired_recolored += 1
        else:
            assert int(out[r, c]) == color, "non-trigger pixels must stay unchanged"
            if same_color and color not in mapping:
                paired_unmapped += 1
            if not mirror_inside and color in mapping:
                mapped_unpaired_unchanged[color] = True
            if mirror_inside and int(inp[r, mc]) != color:
                mismatched_unchanged = True

    assert paired_recolored > 0, "need at least one recolored pixel"
    assert paired_unmapped > 0, "need paired pixels with no legend mapping"
    assert all(mapped_unpaired_unchanged.values()), "each mapped color needs an unchanged unpaired example"
    assert mismatched_unchanged, "need a same-row mirror-presence counterexample with mismatched colors"


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = rng.randint(10, 16)
        w = rng.choice((12, 14, 16, 18))
        mask, _coords, pair_pairs, unpaired = _build_mask((h, w), rng)
        grid, mapping = _assign_mask_colors(mask, pair_pairs, unpaired, rng)
        if not _place_legends(grid, mapping, rng):
            continue

        out = _expected_output(grid)
        try:
            _semantic_checks(grid, out)
        except AssertionError:
            continue
        return grid.astype(int)

    raise RuntimeError("failed to generate a valid mirror-legend ARC input")
