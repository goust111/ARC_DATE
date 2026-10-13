from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


_TOP_ACCENTS = (PINK, TEAL)
_BOTTOM_ACCENTS = (GREY, ORANGE)
_COUNT_PAIRS = ((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))
_MAX_ATTEMPTS = 400

_TEMPLATES = [
    ("##.", "#..", "#.."),
    (".#.", "###"),
    ("##.", ".##"),
    ("###", "..#"),
    (".##", "##."),
    ("###", "#..", "#.."),
    (".#.", "###", ".#."),
    ("###", ".##"),
    ("####", ".#.."),
    ("###", "##.", "#.."),
    ("###", "###"),
    ("##.", "###", ".#."),
]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the half-specific indicator pixels, remove them, and stitch the two halves onto a maroon canvas using those counts as row and column offsets.

    Input structure:
    - The grid has an even number of rows and a black background.
    - The top half contains a blue-dominant pattern plus 1-3 red indicator pixels that are 4-neighbor-adjacent to that pattern.
    - The bottom half contains a yellow-dominant pattern plus 1-3 green indicator pixels that are 4-neighbor-adjacent to that pattern.

    Rule:
    - Count the red pixels in the top half as R and the green pixels in the bottom half as G.
    - Remove those indicator pixels by turning them black.
    - Create a maroon output canvas with height `top_half_height + R + bottom_half_height` and width `max(top_half_width, bottom_half_width + G)`.
    - Copy the cleaned top half to the upper-left corner and the cleaned bottom half to row `top_half_height + R`, column `G`, including black pixels.

    Key insight:
    - The indicators do not encode direction by position; their counts independently encode the vertical gap and horizontal offset used for the final stitched layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 0, "input must split into two equal-height halves"
    assert not bool(np.any(g == MAROON)), "maroon is reserved for the output canvas"

    half = h // 2
    top = g[:half, :].copy()
    bottom = g[half:, :].copy()

    row_gap = int(np.count_nonzero(top == RED))
    col_gap = int(np.count_nonzero(bottom == GREEN))
    assert 1 <= row_gap <= 3 and 1 <= col_gap <= 3, "indicator counts must be in 1..3"

    top[top == RED] = BLACK
    bottom[bottom == GREEN] = BLACK

    out_h = half + row_gap + bottom.shape[0]
    out_w = max(top.shape[1], bottom.shape[1] + col_gap)
    out = np.full((out_h, out_w), MAROON, dtype=int)
    out[: top.shape[0], : top.shape[1]] = top
    out[half + row_gap : half + row_gap + bottom.shape[0], col_gap : col_gap + bottom.shape[1]] = bottom
    return out


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return ()
    coords = coords - coords.min(axis=0, keepdims=True)
    return tuple((int(r), int(c)) for r, c in coords)


def _parse_template(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _sample_mask(rng: np.random.Generator) -> np.ndarray:
    mask = _parse_template(_TEMPLATES[int(rng.integers(len(_TEMPLATES)))])
    mask = np.rot90(mask, int(rng.integers(4)))
    if bool(rng.integers(2)):
        mask = np.fliplr(mask)
    if bool(rng.integers(2)):
        mask = np.flipud(mask)
    return _trim_mask(mask)


def _candidate_slots(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    slots = set()
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                slots.add((nr, nc))
    return sorted(slots)


def _choose_slots(
    candidates: list[tuple[int, int]],
    count: int,
    *,
    reverse: bool,
) -> list[tuple[int, int]] | None:
    ordered = sorted(candidates, reverse=reverse)
    chosen: list[tuple[int, int]] = []
    for cell in ordered:
        if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _paint_shape(
    canvas: np.ndarray,
    shape_mask: np.ndarray,
    *,
    anchor_r: int,
    anchor_c: int,
    main_color: int,
    accent_colors: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    coords = [(anchor_r + int(r), anchor_c + int(c)) for r, c in np.argwhere(shape_mask)]
    for r, c in coords:
        canvas[r, c] = main_color

    area = len(coords)
    max_accents = min(2, area // 3)
    if max_accents == 0:
        return
    accent_count = int(rng.integers(max_accents + 1))
    if accent_count == 0:
        return
    picked = rng.choice(len(coords), size=accent_count, replace=False)
    for i, idx in enumerate(np.atleast_1d(picked).tolist()):
        r, c = coords[int(idx)]
        canvas[r, c] = accent_colors[i % len(accent_colors)]


def _make_half(
    rng: np.random.Generator,
    *,
    half_h: int,
    width: int,
    main_color: int,
    accent_colors: tuple[int, int],
    indicator_color: int,
    indicator_count: int,
    top_half: bool,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        shape_mask = _sample_mask(rng)
        sh, sw = shape_mask.shape
        if sh > half_h or sw > width:
            continue

        row_min = 0 if top_half else 1
        row_max = half_h - sh - (1 if top_half else 0)
        if row_min > row_max:
            continue
        col_max = width - sw
        if col_max < 0:
            continue

        anchor_r = int(rng.integers(row_min, row_max + 1))
        anchor_c = int(rng.integers(col_max + 1))

        placed_mask = np.zeros((half_h, width), dtype=bool)
        placed_mask[anchor_r : anchor_r + sh, anchor_c : anchor_c + sw] = shape_mask
        slots = _candidate_slots(placed_mask)
        chosen = _choose_slots(slots, indicator_count, reverse=not top_half)
        if chosen is None:
            continue

        canvas = np.zeros((half_h, width), dtype=int)
        _paint_shape(
            canvas,
            shape_mask,
            anchor_r=anchor_r,
            anchor_c=anchor_c,
            main_color=main_color,
            accent_colors=accent_colors,
            rng=rng,
        )
        for r, c in chosen:
            canvas[r, c] = indicator_color
        return canvas, placed_mask

    raise RuntimeError("failed to construct a half-region with enough indicator slots")


def _render_output(top: np.ndarray, bottom: np.ndarray, row_gap: int, col_gap: int) -> np.ndarray:
    out_h = top.shape[0] + row_gap + bottom.shape[0]
    out_w = max(top.shape[1], bottom.shape[1] + col_gap)
    out = np.full((out_h, out_w), MAROON, dtype=int)
    out[: top.shape[0], : top.shape[1]] = top
    out[top.shape[0] + row_gap : top.shape[0] + row_gap + bottom.shape[0], col_gap : col_gap + bottom.shape[1]] = bottom
    return out


def _adjacent_to_nonblack(grid: np.ndarray, r: int, c: int, banned: int) -> bool:
    h, w = grid.shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] not in (BLACK, banned):
            return True
    return False


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30 or g.shape[0] % 2 != 0:
        return False
    if np.any(g == MAROON):
        return False

    half = g.shape[0] // 2
    top = g[:half, :].copy()
    bottom = g[half:, :].copy()
    row_gap = int(np.count_nonzero(top == RED))
    col_gap = int(np.count_nonzero(bottom == GREEN))
    if (row_gap, col_gap) not in _COUNT_PAIRS:
        return False
    if np.any(top == GREEN) or np.any(bottom == RED):
        return False

    if np.count_nonzero(top == BLUE) == 0 or np.count_nonzero(bottom == YELLOW) == 0:
        return False

    for r, c in np.argwhere(top == RED):
        if not _adjacent_to_nonblack(top, int(r), int(c), RED):
            return False
    for r, c in np.argwhere(bottom == GREEN):
        if not _adjacent_to_nonblack(bottom, int(r), int(c), GREEN):
            return False

    clean_top = top.copy()
    clean_bottom = bottom.copy()
    clean_top[clean_top == RED] = BLACK
    clean_bottom[clean_bottom == GREEN] = BLACK

    top_key = _shape_key(clean_top != BLACK)
    bottom_key = _shape_key(clean_bottom != BLACK)
    if top_key == bottom_key:
        return False

    if np.count_nonzero(clean_top == BLUE) <= max(np.count_nonzero(clean_top == color) for color in _TOP_ACCENTS):
        return False
    if np.count_nonzero(clean_bottom == YELLOW) <= max(np.count_nonzero(clean_bottom == color) for color in _BOTTOM_ACCENTS):
        return False

    top_shape = clean_top != BLACK
    bottom_shape = clean_bottom != BLACK
    for c in range(g.shape[1]):
        if top_shape[-1, c] and (bottom == GREEN)[0, c]:
            return False
        if (top == RED)[-1, c] and bottom_shape[0, c]:
            return False
        if top_shape[-1, c] and bottom_shape[0, c]:
            return False

    correct = _render_output(clean_top, clean_bottom, row_gap, col_gap)
    if np.array_equal(correct, _render_output(top, bottom, row_gap, col_gap)):
        return False
    if np.array_equal(correct, _render_output(clean_top, clean_bottom, col_gap, row_gap)):
        return False
    if np.array_equal(correct, _render_output(clean_top, clean_bottom, 1, 1)):
        return False

    if not np.array_equal(correct, main(g)):
        return False
    return max(correct.shape) <= 30


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    half_h = int(rng.integers(5, 7))
    width = int(rng.integers(8, 13))
    row_gap, col_gap = _COUNT_PAIRS[int(rng.integers(len(_COUNT_PAIRS)))]

    try:
        top, top_mask = _make_half(
            rng,
            half_h=half_h,
            width=width,
            main_color=BLUE,
            accent_colors=_TOP_ACCENTS,
            indicator_color=RED,
            indicator_count=row_gap,
            top_half=True,
        )
        for _ in range(80):
            bottom, bottom_mask = _make_half(
                rng,
                half_h=half_h,
                width=width,
                main_color=YELLOW,
                accent_colors=_BOTTOM_ACCENTS,
                indicator_color=GREEN,
                indicator_count=col_gap,
                top_half=False,
            )
            if _shape_key(top_mask) != _shape_key(bottom_mask):
                grid = np.vstack([top, bottom])
                return grid if _validate_instance(grid) else None
    except RuntimeError:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
