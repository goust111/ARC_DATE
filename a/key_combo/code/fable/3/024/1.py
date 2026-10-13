from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
SEP = 5
LEGEND_KEY = 8
OCCLUDER = 9
SYMBOL_SIZE = 3
LEGEND_W = 7
ENTRY_H = 3
ENTRY_GAP = 1

OUTPUT_COLORS = [1, 2, 3, 4, 6, 7]
MAIN_SYMBOL_COLORS = [1, 2, 3, 4, 6, 7]
MAIN_SIZES = [13, 17, 21]
HEIGHTS = [17, 21, 25, 29]
MAX_GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            coords: list[Coord] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append({"color": color, "coords": coords, "bbox": _bbox_from_coords(coords)})
    return out


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == int(col[0])) and int(col[0]) != bg:
            candidates.append(c)
    assert len(candidates) == 1, "expected one full-height separator column"
    return int(candidates[0])


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    assert mask.shape == (SYMBOL_SIZE, SYMBOL_SIZE)
    return tuple(int(v) for v in mask.astype(int).ravel())


def _parse_legend(legend: np.ndarray, bg: int) -> dict[tuple[int, ...], int]:
    assert legend.shape[1] >= LEGEND_W, "legend must have room for symbol, gap, and color block"
    lookup: dict[tuple[int, ...], int] = {}
    used_rows: set[int] = set()

    for r in range(0, legend.shape[0] - ENTRY_H + 1):
        key = legend[r : r + ENTRY_H, 0:SYMBOL_SIZE]
        gap = legend[r : r + ENTRY_H, SYMBOL_SIZE]
        block = legend[r : r + ENTRY_H, SYMBOL_SIZE + 1 : SYMBOL_SIZE + 1 + SYMBOL_SIZE]
        if np.any(gap != bg):
            continue
        if not np.any(key != bg):
            continue
        if block.shape != (SYMBOL_SIZE, SYMBOL_SIZE):
            continue
        block_color = int(block[0, 0])
        if block_color == bg or not np.all(block == block_color):
            continue
        if any(rr in used_rows for rr in range(r, r + ENTRY_H)):
            continue

        shape = key != bg
        shape_key = _mask_key(shape)
        assert shape_key not in lookup, "legend key shapes must be unique"
        lookup[shape_key] = block_color
        used_rows.update(range(r, r + ENTRY_H))

    assert lookup, "legend must contain at least one lookup entry"
    return lookup


def _find_occluder_bbox(main: np.ndarray, bg: int) -> BBox:
    candidates: list[BBox] = []
    for comp in _components_by_color(main, bg):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        if color == SEP:
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) != (SYMBOL_SIZE, SYMBOL_SIZE):
            continue
        if len(comp["coords"]) != SYMBOL_SIZE * SYMBOL_SIZE:
            continue
        if np.all(main[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one solid 3x3 occluder block in the main panel"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden rotated lookup key and return the color named by the legend.

    Input structure:
    - A full-height separator column divides a left main panel from a right legend panel.
    - The main panel is an odd-sized canvas with separated 3x3 symbols arranged in 180-degree rotational pairs.
    - Exactly one 3x3 solid occluder block covers a symbol; the corresponding opposite symbol remains visible.
    - The legend panel lists rows of 3x3 symbol keys followed by same-height solid color blocks, including both a key and its 180-degree rotation with different colors.

    Rule:
    - Locate the solid 3x3 occluder in the main panel.
    - Crop the 3x3 symbol at the occluder's 180-degree corresponding location and rotate that visible partner 180 degrees to reconstruct the hidden key shape.
    - Look up that reconstructed key in the legend and output the associated color as a 1x1 grid.

    Key insight:
    - The visible opposite symbol is a trap key; the legend distinguishes it from its half-turn rotation, so the hidden key must be inferred before lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    bg = _most_common_color(g)
    sep_c = _find_separator_col(g, bg)
    main_panel = g[:, :sep_c]
    legend_panel = g[:, sep_c + 1 :]
    assert main_panel.shape[0] % 2 == 1 and main_panel.shape[1] % 2 == 1
    assert legend_panel.shape[1] >= LEGEND_W

    lookup = _parse_legend(legend_panel, bg)
    r0, c0, _r1, _c1 = _find_occluder_bbox(main_panel, bg)

    h, w = main_panel.shape
    partner_r0 = h - r0 - SYMBOL_SIZE
    partner_c0 = w - c0 - SYMBOL_SIZE
    assert 0 <= partner_r0 <= h - SYMBOL_SIZE
    assert 0 <= partner_c0 <= w - SYMBOL_SIZE
    assert (partner_r0, partner_c0) != (r0, c0), "occluder must have a distinct visible partner"

    partner = main_panel[partner_r0 : partner_r0 + SYMBOL_SIZE, partner_c0 : partner_c0 + SYMBOL_SIZE]
    assert not np.any(partner == OCCLUDER), "partner symbol must be visible"
    hidden_shape = np.rot90(partner != bg, 2)
    key = _mask_key(hidden_shape)
    assert key in lookup, "recovered hidden key is missing from the legend"
    return np.asarray([[lookup[key]]], dtype=int)


# --------------- generator ---------------


BASE_SHAPES = [
    ((1, 0, 1), (1, 1, 0), (0, 1, 0)),
    ((0, 1, 0), (1, 1, 1), (1, 0, 0)),
    ((1, 1, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 1, 1), (1, 1, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 1, 0), (0, 1, 1)),
    ((0, 0, 1), (1, 1, 1), (0, 1, 0)),
    ((1, 0, 1), (0, 1, 1), (1, 0, 0)),
    ((0, 1, 0), (1, 1, 0), (1, 0, 1)),
    ((1, 1, 0), (1, 0, 1), (0, 0, 1)),
    ((0, 1, 1), (1, 0, 1), (1, 0, 0)),
]


def _shape_array(shape: tuple[tuple[int, ...], ...]) -> np.ndarray:
    arr = np.asarray(shape, dtype=bool)
    assert arr.shape == (SYMBOL_SIZE, SYMBOL_SIZE)
    assert np.any(arr)
    assert not np.array_equal(arr, np.rot90(arr, 2))
    return arr


SHAPES = [_shape_array(shape) for shape in BASE_SHAPES]


def _shape_orbit_key(mask: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    a = _mask_key(mask)
    b = _mask_key(np.rot90(mask, 2))
    return tuple(sorted((a, b)))


def _anchor_orbits(h: int, w: int) -> list[tuple[Coord, Coord]]:
    anchors = [(r, c) for r in range(1, h - SYMBOL_SIZE + 1, 4) for c in range(1, w - SYMBOL_SIZE + 1, 4)]
    seen: set[Coord] = set()
    orbits: list[tuple[Coord, Coord]] = []
    for r, c in anchors:
        if (r, c) in seen:
            continue
        mate = (h - r - SYMBOL_SIZE, w - c - SYMBOL_SIZE)
        if mate == (r, c) or mate not in anchors:
            continue
        seen.add((r, c))
        seen.add(mate)
        pair = ((r, c), mate)
        orbits.append(pair)
    return orbits


def _paint_symbol(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    patch = grid[r0 : r0 + SYMBOL_SIZE, c0 : c0 + SYMBOL_SIZE]
    patch[mask] = int(color)


def _draw_legend_entry(grid: np.ndarray, top: int, shape: np.ndarray, color: int, legend_c0: int) -> None:
    key_patch = grid[top : top + ENTRY_H, legend_c0 : legend_c0 + SYMBOL_SIZE]
    key_patch[shape] = LEGEND_KEY
    block_c0 = legend_c0 + SYMBOL_SIZE + 1
    grid[top : top + ENTRY_H, block_c0 : block_c0 + SYMBOL_SIZE] = int(color)


def _render_expected_lookup(legend_entries: list[tuple[np.ndarray, int]]) -> dict[tuple[int, ...], int]:
    lookup = {}
    for shape, color in legend_entries:
        key = _mask_key(shape)
        assert key not in lookup
        lookup[key] = int(color)
    return lookup


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    main_h = random.choice(HEIGHTS)
    main_w = random.choice(MAIN_SIZES)
    entry_count = random.randint(4, min(6, len(OUTPUT_COLORS)))
    legend_h = entry_count * ENTRY_H + (entry_count - 1) * ENTRY_GAP
    if legend_h > main_h:
        return None

    orbits = _anchor_orbits(main_h, main_w)
    if len(orbits) < 3:
        return None
    pair_count = random.randint(3, min(6, len(orbits), len(SHAPES)))
    chosen_orbits = random.sample(orbits, pair_count)

    available_shapes = SHAPES[:]
    random.shuffle(available_shapes)
    pair_shapes: list[np.ndarray] = []
    used_shape_orbits: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for shape in available_shapes:
        orbit_key = _shape_orbit_key(shape)
        if orbit_key in used_shape_orbits:
            continue
        pair_shapes.append(shape)
        used_shape_orbits.add(orbit_key)
        if len(pair_shapes) == pair_count:
            break
    if len(pair_shapes) != pair_count:
        return None

    grid = np.full((main_h, main_w + 1 + LEGEND_W), BG, dtype=int)
    sep_c = main_w
    grid[:, sep_c] = SEP

    hidden_pair_i = random.randrange(pair_count)
    hidden_end = random.randrange(2)
    hidden_anchor = chosen_orbits[hidden_pair_i][hidden_end]
    partner_anchor = chosen_orbits[hidden_pair_i][1 - hidden_end]
    hidden_shape = pair_shapes[hidden_pair_i]
    partner_shape = np.rot90(hidden_shape, 2)

    colors = random.choices(MAIN_SYMBOL_COLORS, k=pair_count)
    for i, ((a, b), shape, color) in enumerate(zip(chosen_orbits, pair_shapes, colors)):
        if i == hidden_pair_i:
            _paint_symbol(grid, partner_anchor[0], partner_anchor[1], partner_shape, color)
            hr, hc = hidden_anchor
            grid[hr : hr + SYMBOL_SIZE, hc : hc + SYMBOL_SIZE] = OCCLUDER
        else:
            _paint_symbol(grid, a[0], a[1], shape, color)
            _paint_symbol(grid, b[0], b[1], np.rot90(shape, 2), color)

    target_color, trap_color = random.sample(OUTPUT_COLORS, 2)
    extra_colors = [c for c in OUTPUT_COLORS if c not in (target_color, trap_color)]
    random.shuffle(extra_colors)

    legend_entries: list[tuple[np.ndarray, int]] = [(hidden_shape, target_color), (partner_shape, trap_color)]
    extra_shapes: list[np.ndarray] = []
    for i, shape in enumerate(pair_shapes):
        if i == hidden_pair_i:
            continue
        extra_shapes.extend([shape, np.rot90(shape, 2)])
    random.shuffle(extra_shapes)
    for shape in extra_shapes:
        if len(legend_entries) >= entry_count:
            break
        if _mask_key(shape) not in {_mask_key(s) for s, _ in legend_entries}:
            legend_entries.append((shape, extra_colors[(len(legend_entries) - 2) % len(extra_colors)]))

    if len(legend_entries) < entry_count:
        return None
    random.shuffle(legend_entries)

    top_margin = random.randint(0, main_h - legend_h)
    legend_c0 = sep_c + 1
    for i, (shape, color) in enumerate(legend_entries):
        top = top_margin + i * (ENTRY_H + ENTRY_GAP)
        _draw_legend_entry(grid, top, shape, color, legend_c0)

    expected = np.asarray([[target_color]], dtype=int)
    lookup = _render_expected_lookup(legend_entries)
    if lookup[_mask_key(hidden_shape)] == lookup[_mask_key(partner_shape)]:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _build_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate valid hidden lookup-key input")
