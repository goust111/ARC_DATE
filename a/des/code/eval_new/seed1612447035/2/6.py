import random
from collections import Counter

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
BASE = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (18, 30)
PIECE_COUNT_RANGE = (3, 5)
BASE_HEIGHT_CHOICES = (2, 3)
BAR_ROW_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize(coords)
    h = 1 + max(r for r, _ in norm)
    w = 1 + max(c for _, c in norm)
    patch = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        patch[r, c] = True
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _find_base(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], dict[int, int]]:
    h, w = grid.shape
    best = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if r1 != h - 1 or height < 2 or width < max(8, w // 2):
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(sub[1:, :] == color):
                continue
            top = [int(v) for v in sub[0]]
            if any(v == bg for v in top):
                continue
            slot_colors = [v for v in top if v != color]
            if len(slot_colors) < 2 or len(slot_colors) != len(set(slot_colors)):
                continue
            slot_map = {int(sub[0, dc]): c0 + dc for dc in range(width) if int(sub[0, dc]) != color}
            score = (width * height, width, -r0)
            if best is None or score > best[0]:
                best = (score, color, (r0, c0, r1, c1), slot_map)
    assert best is not None, "failed to locate the slotted base"
    return int(best[1]), best[2], best[3]


def _find_layer_bar(
    grid: np.ndarray, bg: int, slot_colors: set[int], base_bbox: tuple[int, int, int, int]
) -> tuple[int, int, int, list[int]]:
    h, w = grid.shape
    base_r0, _base_c0, base_r1, _base_c1 = base_bbox
    n = len(slot_colors)
    candidates: list[tuple[int, int, int, list[int]]] = []
    for r in range(h):
        if base_r0 <= r <= base_r1:
            continue
        c = 0
        while c < w:
            if int(grid[r, c]) == bg:
                c += 1
                continue
            c0 = c
            while c < w and int(grid[r, c]) != bg:
                c += 1
            c1 = c - 1
            seq = [int(v) for v in grid[r, c0 : c1 + 1]]
            if len(seq) == n and len(set(seq)) == n and set(seq) == slot_colors:
                candidates.append((r, c0, c1, seq))
    assert len(candidates) == 1, "expected a unique layer bar"
    return candidates[0]


def _extract_piece(
    grid: np.ndarray, color: int, base_bbox: tuple[int, int, int, int], bar: tuple[int, int, int, list[int]]
) -> list[tuple[int, int]]:
    mask = grid == color
    r0, c0, r1, c1 = base_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    bar_r, bar_c0, bar_c1, _ = bar
    mask[bar_r, bar_c0 : bar_c1 + 1] = False
    comps = _components(mask)
    assert len(comps) == 1, f"expected exactly one scattered piece for color {color}"
    return comps[0]


def _bottom_midpoint(coords: list[tuple[int, int]]) -> int:
    bottom = max(r for r, _ in coords)
    bottom_cols = [c for r, c in coords if r == bottom]
    mid2 = min(bottom_cols) + max(bottom_cols)
    assert mid2 % 2 == 0, "piece bottom midpoint must be integral"
    return mid2 // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered colored fragments onto the slotted base using the bar-defined layering order.

    Input structure:
    - bg = most common color.
    - A large solid rectangular base sits on the bottom edge; its top row is mostly base-colored but includes several distinct colored slot markers.
    - A separate one-row color bar elsewhere in the grid contains exactly the slot-marker colors, once each, in adjacent cells.
    - Above the base are monochrome 4-connected fragments, one per slot color, scattered apart from the base and the bar.

    Rule:
    - Identify the bottom base and read each slot marker's column from the colored cells embedded in its top row.
    - Match every scattered fragment to the slot marker of the same color.
    - Translate each fragment so its lowest occupied row lands exactly one row above the base top, and so the horizontal midpoint of its lowest row aligns with its slot column.
    - Clear the separate color bar, then place the translated fragments in the left-to-right order shown by that bar; later fragments overwrite earlier ones where they overlap.

    Key insight: Two independent cues must be combined: slot-marker colors encode each fragment's destination column, while the detached color bar encodes the only valid front-to-back drawing order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)

    _base_color, base_bbox, slot_map = _find_base(g, bg)
    slot_colors = set(slot_map)
    bar = _find_layer_bar(g, bg, slot_colors, base_bbox)
    base_r0, _base_c0, _base_r1, _base_c1 = base_bbox

    pieces: dict[int, list[tuple[int, int]]] = {}
    original_piece_cells: list[tuple[int, int]] = []
    for color in slot_map:
        coords = _extract_piece(g, color, base_bbox, bar)
        pieces[color] = coords
        original_piece_cells.extend(coords)

    out = g.copy()
    bar_r, bar_c0, bar_c1, order = bar
    out[bar_r, bar_c0 : bar_c1 + 1] = bg
    for r, c in original_piece_cells:
        out[r, c] = bg

    for color in order:
        coords = pieces[int(color)]
        piece_bottom = max(r for r, _ in coords)
        dr = (base_r0 - 1) - piece_bottom
        dc = slot_map[int(color)] - _bottom_midpoint(coords)
        for r, c in coords:
            rr = r + dr
            cc = c + dc
            assert 0 <= rr < h and 0 <= cc < w
            assert rr < base_r0
            out[rr, cc] = int(color)

    return out


def _make_piece_patch() -> np.ndarray:
    for _ in range(200):
        bottom_w = random.choices([1, 3, 5], weights=[2, 4, 2])[0]
        height = random.randint(1, 4)
        left_pad = random.randint(0, 1)
        right_pad = random.randint(0, 1)
        width = bottom_w + left_pad + right_pad

        cells = {(height - 1, left_pad + dc) for dc in range(bottom_w)}
        target = random.randint(bottom_w, min(width * height, bottom_w + 4))
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if nr == height - 1 and not (left_pad <= nc < left_pad + bottom_w):
                        continue
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        norm = _normalize(cells)
        patch = _mask_from_coords(norm)
        if random.random() < 0.5:
            patch = np.fliplr(patch)

        bottom = patch.shape[0] - 1
        bottom_cols = list(np.flatnonzero(patch[bottom]))
        if not bottom_cols:
            continue
        if bottom_cols != list(range(bottom_cols[0], bottom_cols[-1] + 1)):
            continue
        if (bottom_cols[0] + bottom_cols[-1]) % 2 != 0:
            continue
        return patch
    raise RuntimeError("failed to generate a valid fragment patch")


def _piece_cells_from_patch(patch: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(patch)]


def _placement_cells(piece: dict, slot_col: int, base_top: int) -> set[tuple[int, int]]:
    top = base_top - piece["h"]
    left = slot_col - piece["mid"]
    return {(top + r, left + c) for r, c in piece["cells"]}


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sample_slot_columns(
    base_c0: int, base_c1: int, pieces: list[dict], want_overlap: bool
) -> dict[int, int] | None:
    usable = list(range(base_c0 + 3, base_c1 - 2))
    if len(usable) < len(pieces):
        return None

    colors = [piece["color"] for piece in pieces]
    piece_by_color = {piece["color"]: piece for piece in pieces}

    if want_overlap:
        pair_colors = random.sample(colors, 2)
        a = piece_by_color[pair_colors[0]]
        b = piece_by_color[pair_colors[1]]
        max_gap = a["bottom_radius"] + b["bottom_radius"]
        pair_options = []
        for col_a in usable:
            for gap in range(1, max_gap + 1):
                for sign in (-1, 1):
                    col_b = col_a + sign * gap
                    if col_b in usable and col_b != col_a:
                        pair_options.append((col_a, col_b))
        if not pair_options:
            return None

        for _ in range(120):
            col_a, col_b = random.choice(pair_options)
            mapping = {pair_colors[0]: col_a, pair_colors[1]: col_b}
            remaining = [color for color in colors if color not in mapping]
            free = [c for c in usable if c not in mapping.values()]
            if len(free) < len(remaining):
                continue
            random.shuffle(free)
            for color, col in zip(remaining, free):
                mapping[color] = col
            if _has_overlap(mapping, piece_by_color):
                return mapping
        return None

    for _ in range(160):
        cols = random.sample(usable, len(pieces))
        mapping = {color: col for color, col in zip(colors, cols)}
        if not _has_overlap(mapping, piece_by_color):
            return mapping
    return None


def _has_overlap(slot_map: dict[int, int], piece_by_color: dict[int, dict]) -> bool:
    counts: Counter[tuple[int, int]] = Counter()
    base_top = piece_by_color[next(iter(piece_by_color))]["base_top"]
    for color, slot_col in slot_map.items():
        for cell in _placement_cells(piece_by_color[color], slot_col, base_top):
            counts[cell] += 1
            if counts[cell] > 1:
                return True
    return False


def _place_scattered_pieces(
    grid: np.ndarray, pieces: list[dict], reserved: np.ndarray, base_top: int
) -> bool:
    occupied = reserved.copy()
    h, w = grid.shape
    order = sorted(pieces, key=lambda piece: piece["area"], reverse=True)
    for piece in order:
        patch = piece["patch"]
        ph, pw = patch.shape
        min_r0 = piece["bar_row"] + 2
        max_r0 = base_top - ph - 2
        if max_r0 < min_r0:
            return False
        candidates = [
            (r0, c0)
            for r0 in range(min_r0, max_r0 + 1)
            for c0 in range(0, w - pw + 1)
            if not np.any(_dilate8(occupied)[r0 : r0 + ph, c0 : c0 + pw] & patch)
        ]
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        coords = {(r0 + r, c0 + c) for r, c in piece["cells"]}
        piece["scatter_coords"] = coords
        _paint_cells(grid, coords, piece["color"])
        for r, c in coords:
            occupied[r, c] = True
    return True


def _build_expected_output(
    h: int,
    w: int,
    base_bbox: tuple[int, int, int, int],
    slot_map: dict[int, int],
    pieces: list[dict],
    layer_order: list[int],
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    r0, c0, r1, c1 = base_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BASE
    for color, col in slot_map.items():
        out[r0, col] = color
    piece_by_color = {piece["color"]: piece for piece in pieces}
    for color in layer_order:
        coords = _placement_cells(piece_by_color[color], slot_map[color], r0)
        _paint_cells(out, coords, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    base_h = random.choice(BASE_HEIGHT_CHOICES)
    base_w = random.randint(max(12, w - 8), w - 1)
    base_c0 = random.randint(0, w - base_w)
    base_c1 = base_c0 + base_w - 1
    base_r0 = h - base_h
    base_bbox = (base_r0, base_c0, h - 1, base_c1)

    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    if n_pieces > len(PIECE_COLORS):
        return None
    colors = random.sample(PIECE_COLORS, n_pieces)
    want_overlap = random.random() < 0.5

    pieces = []
    for color in colors:
        patch = _make_piece_patch()
        cells = _piece_cells_from_patch(patch)
        bottom_cols = list(np.flatnonzero(patch[-1]))
        mid = (bottom_cols[0] + bottom_cols[-1]) // 2
        pieces.append(
            {
                "color": color,
                "patch": patch,
                "cells": cells,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "mid": int(mid),
                "bottom_radius": (bottom_cols[-1] - bottom_cols[0]) // 2,
                "area": int(patch.sum()),
                "base_top": int(base_r0),
            }
        )

    slot_map = _sample_slot_columns(base_c0, base_c1, pieces, want_overlap)
    if slot_map is None:
        return None

    layer_order = colors[:]
    random.shuffle(layer_order)

    bar_row = random.randint(*BAR_ROW_RANGE)
    bar_side_left = random.random() < 0.5
    bar_c0 = 0 if bar_side_left else w - n_pieces
    bar_c1 = bar_c0 + n_pieces - 1

    grid = np.full((h, w), BG, dtype=int)
    grid[base_r0:h, base_c0 : base_c1 + 1] = BASE
    for color, col in slot_map.items():
        grid[base_r0, col] = color
    for idx, color in enumerate(layer_order):
        grid[bar_row, bar_c0 + idx] = color

    reserved = np.zeros((h, w), dtype=bool)
    reserved[base_r0:h, base_c0 : base_c1 + 1] = True
    reserved[bar_row, bar_c0 : bar_c1 + 1] = True
    for piece in pieces:
        piece["bar_row"] = int(bar_row)

    if not _place_scattered_pieces(grid, pieces, reserved, base_r0):
        return None

    expected = _build_expected_output(h, w, base_bbox, slot_map, pieces, layer_order)
    if _has_overlap(slot_map, {piece["color"]: piece for piece in pieces}) != want_overlap:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
