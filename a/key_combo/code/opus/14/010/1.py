from __future__ import annotations

import random

import numpy as np


BG = 0
STAMP = 2
SQUARE = 3
DIVIDER = 5

LEGEND_H = 3
SLOT_W = 3
SEP_W = 1
LEGEND_GAP = 1

GLYPHS: dict[int, np.ndarray] = {
    3: np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    5: np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    7: np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    9: np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_legend(g: np.ndarray) -> tuple[int, int, dict[int, np.ndarray]]:
    gray_rows = np.where(np.any(g == DIVIDER, axis=1))[0]
    assert gray_rows.size >= 1, "legend dividers missing"
    assert gray_rows[-1] == g.shape[0] - 1, "legend must touch bottom edge"
    r0 = int(gray_rows[0])
    r1 = int(gray_rows[-1])
    assert r1 - r0 + 1 == LEGEND_H, "legend band must be 3 rows tall"

    boundaries = [
        c for c in range(g.shape[1]) if np.all(g[r0 : r1 + 1, c] == DIVIDER)
    ]
    assert len(boundaries) in (4, 5), "legend must contain 3 or 4 slots"
    assert all(boundaries[i + 1] - boundaries[i] == SLOT_W + SEP_W for i in range(len(boundaries) - 1))

    glyphs: dict[int, np.ndarray] = {}
    for idx in range(len(boundaries) - 1):
        c0 = boundaries[idx] + 1
        c1 = boundaries[idx + 1] - 1
        assert c1 - c0 + 1 == SLOT_W
        glyphs[2 * idx + 3] = (g[r0 : r1 + 1, c0 : c1 + 1] == STAMP)
    return r0, r1, glyphs


def _solid_green_squares(g: np.ndarray) -> list[tuple[int, int, int]]:
    squares: list[tuple[int, int, int]] = []
    for coords in _components(g == SQUARE):
        r0, c0, r1, c1 = _bbox(coords)
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        assert side_r == side_c, "green objects must be square"
        assert coords.shape[0] == side_r * side_c, "green squares must be solid"
        squares.append((r0, c0, side_r))
    squares.sort()
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each odd-sized green square by the 3x3 legend glyph indexed by its side length, anchored at the square center.

    Input structure:
    - A mostly black grid contains several separated green solid square components of mixed odd and even side lengths.
    - A legend strip touches the bottom edge, using gray vertical dividers to split 3 or 4 equal 3x3 slots.
    - Each legend slot contains a distinct red 3x3 glyph; from left to right the slots encode side lengths 3, 5, 7, and optionally 9.

    Rule:
    - Detect every green 4-connected solid square and measure its side length.
    - Even-sized squares remain unchanged.
    - For each odd-sized square, erase the whole square to black, then stamp the red glyph whose legend slot matches that side length with the glyph centered on the square center; glyph background is transparent.
    - Erase the entire bottom legend strip after all stamping is finished.

    Key insight:
    - The same geometric attribute is reused twice: odd/even parity decides whether a square transforms, while the exact odd side length selects which glyph to stamp at the square center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG

    legend_r0, legend_r1, glyphs = _extract_legend(g)
    out = g.copy()

    for r0, c0, side in _solid_green_squares(g):
        if side % 2 == 0:
            continue
        assert side in glyphs, "odd square side length must appear in the legend"
        out[r0 : r0 + side, c0 : c0 + side] = BG
        cr = r0 + side // 2
        cc = c0 + side // 2
        glyph = glyphs[side]
        for dr in range(SLOT_W):
            for dc in range(SLOT_W):
                if glyph[dr, dc]:
                    out[cr - 1 + dr, cc - 1 + dc] = STAMP

    out[legend_r0 : legend_r1 + 1, :] = BG
    return out


def _legend_width(slot_count: int) -> int:
    return slot_count * SLOT_W + (slot_count + 1) * SEP_W


def _weighted_choice(options: list[int], weights: list[float]) -> int:
    return random.choices(options, weights=weights, k=1)[0]


def _sample_sides(slot_count: int) -> list[int]:
    n_objects = random.randint(4, 6)
    n_odd = random.randint(1, n_objects - 1)
    n_even = n_objects - n_odd

    odd_pool = [3, 5, 7] + ([9] if slot_count == 4 else [])
    odd_weights = [0.40, 0.30, 0.20] + ([0.10] if slot_count == 4 else [])
    even_pool = [4, 6, 8]
    even_weights = [0.45, 0.35, 0.20]

    odd_sides: list[int] = []
    if slot_count == 4 and n_odd >= 1 and random.random() < 0.55:
        odd_sides.append(9)
    if n_odd >= 2:
        distinct = min(2, len(odd_pool), n_odd - len(odd_sides))
        remaining_choices = [x for x in odd_pool if x not in odd_sides]
        odd_sides.extend(random.sample(remaining_choices, k=distinct))
    while len(odd_sides) < n_odd:
        odd_sides.append(_weighted_choice(odd_pool, odd_weights))

    even_sides = [_weighted_choice(even_pool, even_weights) for _ in range(n_even)]
    sides = odd_sides + even_sides
    random.shuffle(sides)
    sides.sort(reverse=True)
    return sides


def _can_place(occ: np.ndarray, r0: int, c0: int, side: int) -> bool:
    h, w = occ.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + side + 1)
    cc1 = min(w, c0 + side + 1)
    return not bool(np.any(occ[rr0:rr1, cc0:cc1]))


def _place_squares(scene_h: int, width: int, sides: list[int]) -> list[tuple[int, int, int]] | None:
    occ = np.zeros((scene_h, width), dtype=bool)
    placed: list[tuple[int, int, int]] = []
    for side in sides:
        candidates: list[tuple[int, int]] = []
        for r0 in range(scene_h - side + 1):
            for c0 in range(width - side + 1):
                if _can_place(occ, r0, c0, side):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occ[r0 : r0 + side, c0 : c0 + side] = True
        placed.append((r0, c0, side))
    return placed


def _draw_legend(inp: np.ndarray, slot_count: int, start_col: int) -> None:
    row0 = inp.shape[0] - LEGEND_H
    width = _legend_width(slot_count)
    inp[row0:, start_col : start_col + width] = BG
    for idx in range(slot_count + 1):
        col = start_col + idx * (SLOT_W + SEP_W)
        inp[row0:, col] = DIVIDER
    for idx, side in enumerate(range(3, 2 * slot_count + 3, 2)):
        c0 = start_col + idx * (SLOT_W + SEP_W) + SEP_W
        mask = GLYPHS[side]
        inp[row0 : row0 + SLOT_W, c0 : c0 + SLOT_W][mask] = STAMP


def _expected_from_layout(
    inp: np.ndarray,
    squares: list[tuple[int, int, int]],
    slot_count: int,
) -> np.ndarray:
    out = inp.copy()
    row0 = inp.shape[0] - LEGEND_H
    glyph_map = {side: GLYPHS[side] for side in range(3, 2 * slot_count + 3, 2)}
    for r0, c0, side in sorted(squares):
        if side % 2 == 0:
            continue
        out[r0 : r0 + side, c0 : c0 + side] = BG
        cr = r0 + side // 2
        cc = c0 + side // 2
        mask = glyph_map[side]
        for dr in range(SLOT_W):
            for dc in range(SLOT_W):
                if mask[dr, dc]:
                    out[cr - 1 + dr, cc - 1 + dc] = STAMP
    out[row0:, :] = BG
    return out


def _attempt_generate() -> np.ndarray | None:
    slot_count = random.choice([3, 4])
    legend_w = _legend_width(slot_count)

    height = random.randint(18, 24)
    width = random.randint(max(18, legend_w), 28)
    scene_h = height - LEGEND_H - LEGEND_GAP
    if scene_h < 9:
        return None

    sides = _sample_sides(slot_count)
    if max(sides) > min(scene_h, width):
        return None

    squares = _place_squares(scene_h, width, sides)
    if squares is None:
        return None

    inp = np.full((height, width), BG, dtype=int)
    for r0, c0, side in squares:
        inp[r0 : r0 + side, c0 : c0 + side] = SQUARE

    start_col = random.randint(0, width - legend_w)
    _draw_legend(inp, slot_count, start_col)

    expected = _expected_from_layout(inp, squares, slot_count)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    odd_sides = {side for _r, _c, side in squares if side % 2 == 1}
    even_sides = {side for _r, _c, side in squares if side % 2 == 0}
    if not odd_sides or not even_sides:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
