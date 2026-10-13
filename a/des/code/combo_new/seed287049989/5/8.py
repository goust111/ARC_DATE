from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_MISSING_CORNER_TO_TURNS = {
    (0, 0): 0,  # open to top-left
    (0, 1): 1,  # open to top-right
    (1, 1): 2,  # open to bottom-right
    (1, 0): 3,  # open to bottom-left
}
_TURN_TO_MISSING_CORNER = {v: k for k, v in _MISSING_CORNER_TO_TURNS.items()}
_GREY = 5
_MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rotate_offset(offset: tuple[int, int], turns: int) -> tuple[int, int]:
    dr, dc = offset
    turns %= 4
    if turns == 0:
        return dr, dc
    if turns == 1:
        return dc, -dr
    if turns == 2:
        return -dr, -dc
    return -dc, dr


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _has_nontrivial_symmetry(mask: np.ndarray) -> bool:
    variants = []
    for rot in range(4):
        arr = np.rot90(mask, rot)
        variants.append(arr)
        variants.append(np.fliplr(arr))
    original = mask
    for variant in variants[1:]:
        if variant.shape == original.shape and np.array_equal(variant, original):
            return True
    return False


def _find_l_marker(grid: np.ndarray, bg: int, pattern_color: int) -> tuple[int, list[tuple[int, int]], int]:
    for color in map(int, np.unique(grid)):
        if color in (bg, pattern_color):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(coords) != 3:
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        rel = {(r - r0, c - c0) for r, c in coords}
        missing = ({(0, 0), (0, 1), (1, 0), (1, 1)} - rel)
        if len(missing) != 1:
            continue
        turns = _MISSING_CORNER_TO_TURNS[missing.pop()]
        return color, coords, turns
    raise AssertionError("expected a unique 3-cell L marker")


def _pattern_color(grid: np.ndarray, bg: int) -> int:
    counts = {int(color): int(np.sum(grid == color)) for color in np.unique(grid)}
    candidates = [(count, color) for color, count in counts.items() if color != bg and count >= 4]
    assert candidates, "expected one pattern color with at least four pixels"
    return max(candidates)[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use same-color anchor pairs and an L marker to place rotated copies of the main sprite.

    Input structure:
    - The background is the most common color.
    - One connected non-background color forms the main sprite, and its bounding box also contains several empty cells.
    - Each anchor-pair color appears in exactly two singleton pixels: one lies inside the main sprite's bounding box, and the other lies outside it on blank background.
    - A separate 3-pixel L marker sits outside the main sprite; its missing corner encodes a global rotation.

    Rule:
    - Read the L marker's opening direction to get a clockwise rotation of 0, 90, 180, or 270 degrees.
    - For each anchor pair, treat the anchor inside the sprite bbox as the reference origin and rotate every sprite-pixel offset around that origin.
    - Repaint the rotated sprite using the outside anchor as the new origin.
    - Keep the original sprite, all anchor pixels, and the L marker unchanged; later copies may overwrite earlier copies only on originally blank cells.

    Key insight:
    - The task combines two independent clues: same-color singleton pairing tells you which blank location receives a copy, and the L marker's missing corner tells you how every copied offset must be rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    sprite_color = _pattern_color(g, bg)
    sprite_mask = g == sprite_color
    assert _is_connected(sprite_mask), "main sprite must stay connected"

    sprite_coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite_mask)]
    rows = [r for r, _ in sprite_coords]
    cols = [c for _, c in sprite_coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    l_color, _, turns = _find_l_marker(g, bg, sprite_color)

    anchor_pairs = []
    for color in map(int, np.unique(g)):
        if color in (bg, sprite_color, l_color):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, f"anchor color {color} must appear twice"
        inside = [pos for pos in coords if r0 <= pos[0] <= r1 and c0 <= pos[1] <= c1]
        outside = [pos for pos in coords if pos not in inside]
        assert len(inside) == 1 and len(outside) == 1, "each anchor color needs one inside and one outside pixel"
        first_row, first_col = min(coords)
        anchor_pairs.append((first_row, first_col, inside[0], outside[0]))
    anchor_pairs.sort()

    protected = g != bg
    out = g.copy()
    for _, _, inside, outside in anchor_pairs:
        ar, ac = inside
        tr, tc = outside
        for sr, sc in sprite_coords:
            dr, dc = sr - ar, sc - ac
            rr_off, cc_off = _rotate_offset((dr, dc), turns)
            rr, cc = tr + rr_off, tc + cc_off
            assert 0 <= rr < h and 0 <= cc < w
            if not protected[rr, cc]:
                out[rr, cc] = sprite_color
    return out


def _sample_pattern(anchor_count: int) -> np.ndarray:
    for _ in range(300):
        ph = int(_RNG.integers(3, 7))
        pw = int(_RNG.integers(3, 7))
        bbox_area = ph * pw
        max_area = bbox_area - anchor_count
        if max_area < 4:
            continue
        target_area = int(_RNG.integers(4, max_area + 1))
        start = (int(_RNG.integers(ph)), int(_RNG.integers(pw)))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < ph and 0 <= cc < pw and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(tuple(frontier.pop()) if len(frontier) == 1 else tuple(list(frontier)[int(_RNG.integers(len(frontier)))]))
        mask = np.zeros((ph, pw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not (rows[0] and rows[-1] and cols[0] and cols[-1]):
            continue
        if int((~mask).sum()) < anchor_count:
            continue
        if _has_nontrivial_symmetry(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a suitable pattern")


def _choose_internal_anchors(mask: np.ndarray, anchor_count: int) -> list[tuple[int, int]]:
    empties = [tuple(map(int, rc)) for rc in np.argwhere(~mask)]
    assert len(empties) >= anchor_count
    _RNG.shuffle(empties)
    by_row: dict[int, list[tuple[int, int]]] = {}
    for cell in empties:
        by_row.setdefault(cell[0], []).append(cell)
    if len(by_row) >= anchor_count:
        rows = list(by_row)
        _RNG.shuffle(rows)
        chosen = []
        for row in rows:
            options = by_row[row]
            chosen.append(options[int(_RNG.integers(len(options)))])
            if len(chosen) == anchor_count:
                return chosen
    return empties[:anchor_count]


def _copy_offsets(mask: np.ndarray, anchor: tuple[int, int], turns: int) -> list[tuple[int, int]]:
    ar, ac = anchor
    offsets = []
    for r, c in np.argwhere(mask):
        dr, dc = int(r) - ar, int(c) - ac
        offsets.append(_rotate_offset((dr, dc), turns))
    return offsets


def _padded_box_mask(shape: tuple[int, int], r0: int, c0: int, h: int, w: int, pad: int = 1) -> np.ndarray:
    grid_h, grid_w = shape
    mask = np.zeros(shape, dtype=bool)
    rr0 = max(0, r0 - pad)
    rr1 = min(grid_h, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(grid_w, c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True
    return mask


def _feasible_external_anchors(
    grid_shape: tuple[int, int],
    offsets: list[tuple[int, int]],
    forbidden_region: np.ndarray,
    protected: np.ndarray,
    occupied_copy: np.ndarray,
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = grid_shape
    drs = [dr for dr, _ in offsets]
    dcs = [dc for _, dc in offsets]
    row_lo, row_hi = -min(drs), h - 1 - max(drs)
    col_lo, col_hi = -min(dcs), w - 1 - max(dcs)
    candidates = []
    for ar in range(row_lo, row_hi + 1):
        for ac in range(col_lo, col_hi + 1):
            if forbidden_region[ar, ac] or protected[ar, ac] or occupied_copy[ar, ac]:
                continue
            cells = []
            ok = True
            for dr, dc in offsets:
                rr, cc = ar + dr, ac + dc
                if forbidden_region[rr, cc] or protected[rr, cc] or occupied_copy[rr, cc]:
                    ok = False
                    break
                cells.append((rr, cc))
            if ok:
                candidates.append(((ar, ac), cells))
    return candidates


def _l_offsets(turns: int) -> list[tuple[int, int]]:
    missing = _TURN_TO_MISSING_CORNER[turns % 4]
    return [
        (r, c)
        for r in range(2)
        for c in range(2)
        if (r, c) != missing
    ]


def _place_l_marker(
    grid_shape: tuple[int, int],
    turns: int,
    forbidden_region: np.ndarray,
    protected: np.ndarray,
    occupied_copy: np.ndarray,
) -> list[tuple[int, int]] | None:
    offsets = _l_offsets(turns)
    h, w = grid_shape
    candidates = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            box_cells = [(r0 + r, c0 + c) for r in range(2) for c in range(2)]
            if any(forbidden_region[r, c] or protected[r, c] or occupied_copy[r, c] for r, c in box_cells):
                continue
            candidates.append([(r0 + dr, c0 + dc) for dr, dc in offsets])
    if not candidates:
        return None
    return candidates[int(_RNG.integers(len(candidates)))]


def _gen_attempt() -> np.ndarray | None:
    anchor_count = int(_RNG.integers(2, 5))
    pattern = _sample_pattern(anchor_count)
    ph, pw = pattern.shape

    min_side = max(16, 2 * max(ph, pw) + 8 + anchor_count)
    max_side = min(30, min_side + 8)
    grid_h = int(_RNG.integers(min_side, max_side + 1))
    grid_w = int(_RNG.integers(min_side, max_side + 1))

    turns = int(_RNG.integers(4))

    bg_choices = [color for color in range(10) if color != _GREY]
    bg = int(bg_choices[int(_RNG.integers(len(bg_choices)))])
    available = [color for color in range(10) if color not in (bg, _GREY)]
    _RNG.shuffle(available)
    pattern_color = int(available[0])
    anchor_colors = [int(color) for color in available[1 : 1 + anchor_count]]

    src_r0 = int(_RNG.integers(2, grid_h - ph - 1))
    src_c0 = int(_RNG.integers(2, grid_w - pw - 1))
    source_box_forbidden = _padded_box_mask((grid_h, grid_w), src_r0, src_c0, ph, pw, pad=1)

    internal_anchors_rel = _choose_internal_anchors(pattern, anchor_count)
    source_cells = [(src_r0 + int(r), src_c0 + int(c)) for r, c in np.argwhere(pattern)]
    internal_anchors = [(src_r0 + r, src_c0 + c) for r, c in internal_anchors_rel]

    protected = np.zeros((grid_h, grid_w), dtype=bool)
    for r, c in source_cells + internal_anchors:
        protected[r, c] = True

    occupied_copy = np.zeros((grid_h, grid_w), dtype=bool)
    external_anchors: list[tuple[int, int]] = []
    copy_footprints: list[list[tuple[int, int]]] = []

    for anchor_rel in internal_anchors_rel:
        offsets = _copy_offsets(pattern, anchor_rel, turns)
        candidates = _feasible_external_anchors(
            (grid_h, grid_w),
            offsets,
            source_box_forbidden,
            protected,
            occupied_copy,
        )
        if not candidates:
            return None
        idx = int(_RNG.integers(len(candidates)))
        anchor_abs, cells = candidates[idx]
        external_anchors.append(anchor_abs)
        copy_footprints.append(cells)
        protected[anchor_abs] = True
        for rr, cc in cells:
            occupied_copy[rr, cc] = True

    l_cells = _place_l_marker((grid_h, grid_w), turns, source_box_forbidden, protected, occupied_copy)
    if l_cells is None:
        return None
    for r, c in l_cells:
        protected[r, c] = True

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for r, c in source_cells:
        grid[r, c] = pattern_color
    for color, inside, outside in zip(anchor_colors, internal_anchors, external_anchors):
        grid[inside] = color
        grid[outside] = color
    for r, c in l_cells:
        grid[r, c] = _GREY

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
