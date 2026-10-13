from __future__ import annotations

import random

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(1, 10))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _inside_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _missing_rect(grid: np.ndarray, axis: int) -> tuple[tuple[int, int, int, int], str]:
    black = grid == BLACK
    box = _bbox(black)
    assert box is not None, "expected one black missing rectangle"

    r0, c0, r1, c1 = box
    assert np.all(black[r0 : r1 + 1, c0 : c1 + 1]), "black cells must form a solid rectangle"
    assert int(np.count_nonzero(black)) == (r1 - r0 + 1) * (c1 - c0 + 1), "no black outside the missing rectangle"

    if c1 < axis:
        side = "left"
    elif c0 > axis:
        side = "right"
    else:
        raise AssertionError("missing rectangle must lie strictly on one side of the separator")
    return box, side


def _learn_left_to_right(
    grid: np.ndarray, axis: int, hole: tuple[int, int, int, int]
) -> tuple[dict[int, int], dict[int, int]]:
    h, _w = grid.shape
    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}

    for r in range(h):
        for lc in range(axis):
            rc = 2 * axis - lc
            if _inside_box(r, lc, hole) or _inside_box(r, rc, hole):
                continue

            left = int(grid[r, lc])
            right = int(grid[r, rc])
            assert left != BLACK and right != BLACK, "only the missing rectangle may be black"

            if left in mapping:
                assert mapping[left] == right, "left-to-right color mapping must be consistent"
            if right in inverse:
                assert inverse[right] == left, "color mapping must be one-to-one"
            mapping[left] = right
            inverse[right] = left

    assert mapping, "expected observed mirror pairs"
    assert set(mapping) == set(inverse), "observed colors must form a closed permutation"
    assert any(mapping[mapping[color]] != color for color in mapping), "permutation must not be self-inverse"
    return mapping, inverse


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the blacked-out mirror patch using the learned negative color permutation.

    Input structure:
    - A one-pixel vertical separator column splits the grid into two equal-width halves.
    - The two halves are horizontal mirror images across that separator, but every
      left-half color is changed by one fixed bijective, non-self-inverse color permutation
      when it appears on the right half.
    - Exactly one side contains a solid black rectangular missing region; all other
      mirrored pixel pairs are visible and non-black.

    Rule:
    - Treat the central separator as the mirror axis and locate the only black rectangle.
    - From all visible mirrored pairs outside that rectangle, infer the color permutation
      from the left half to the right half.
    - For every missing pixel, read the pixel mirrored across the separator. If the
      missing rectangle is on the right, apply the learned permutation; if it is on
      the left, apply the inverse permutation.
    - Return only the recovered rectangular patch.

    Key insight:
    - The geometric mirror and the color remapping are separate steps, and the correct
      direction of the non-self-inverse permutation is determined by which side is missing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width >= 3 and width % 2 == 1, "grid must have two equal halves and a separator"

    axis = width // 2
    sep_col = g[:, axis]
    assert np.all(sep_col == sep_col[0]) and int(sep_col[0]) != BLACK, "expected a non-black vertical separator"

    hole, side = _missing_rect(g, axis)
    left_to_right, right_to_left = _learn_left_to_right(g, axis, hole)

    r0, c0, r1, c1 = hole
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            mirror_c = 2 * axis - c
            src = int(g[r, mirror_c])
            assert src != BLACK, "the mirror source for a missing pixel must be visible"
            if side == "right":
                assert src in left_to_right, "source color must have an observed forward mapping"
                out[r - r0, c - c0] = left_to_right[src]
            else:
                assert src in right_to_left, "source color must have an observed inverse mapping"
                out[r - r0, c - c0] = right_to_left[src]
    return out


# ------------------------------ generator ------------------------------

GEN_ATTEMPTS = 500
MIN_H, MAX_H = 9, 24
MIN_HALF_W, MAX_HALF_W = 5, 13
MIN_PALETTE, MAX_PALETTE = 3, 5
MIN_HOLE_H, MAX_HOLE_H = 2, 7
MIN_HOLE_W, MAX_HOLE_W = 2, 5
EXTRA_SHAPE_COUNT_RANGE = (4, 9)

_GEN_CALL_COUNT = 0


def _cycle_permutation(colors: list[int]) -> dict[int, int]:
    order = random.sample(colors, len(colors))
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


def _inverse_perm(perm: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in perm.items()}


def _apply_perm(arr: np.ndarray, perm: dict[int, int]) -> np.ndarray:
    out = np.empty_like(arr, dtype=int)
    for color, mapped in perm.items():
        out[arr == color] = mapped
    return out


def _build_left_pattern(height: int, width: int, palette: list[int]) -> np.ndarray:
    n = len(palette)
    block_r = random.randint(1, 3)
    block_c = random.randint(1, 3)
    stride_r = random.choice([1, 2])
    stride_c = random.choice([1, 2])
    phase = random.randrange(n)

    left = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            idx = (phase + stride_r * (r // block_r) + stride_c * (c // block_c)) % n
            left[r, c] = palette[idx]

    for color in palette:
        rr = random.randrange(height)
        cc = random.randrange(width)
        left[rr, cc] = color

    for _ in range(random.randint(*EXTRA_SHAPE_COUNT_RANGE)):
        color = random.choice(palette)
        if random.random() < 0.55:
            rh = random.randint(1, min(4, height))
            rw = random.randint(1, min(4, width))
            r0 = random.randint(0, height - rh)
            c0 = random.randint(0, width - rw)
            left[r0 : r0 + rh, c0 : c0 + rw] = color
        elif random.random() < 0.5:
            r = random.randrange(height)
            c0 = random.randint(0, width - 2)
            c1 = random.randint(c0 + 1, width - 1)
            left[r, c0 : c1 + 1] = color
        else:
            c = random.randrange(width)
            r0 = random.randint(0, height - 2)
            r1 = random.randint(r0 + 1, height - 1)
            left[r0 : r1 + 1, c] = color

    return left


def _full_grid(
    left: np.ndarray, sep: int, perm: dict[int, int]
) -> np.ndarray:
    right = _apply_perm(np.fliplr(left), perm)
    sep_col = np.full((left.shape[0], 1), int(sep), dtype=int)
    return np.concatenate((left, sep_col, right), axis=1)


def _sample_hole(height: int, half_w: int, side: str) -> tuple[int, int, int, int]:
    hh = random.randint(MIN_HOLE_H, min(MAX_HOLE_H, height))
    hw = random.randint(MIN_HOLE_W, min(MAX_HOLE_W, half_w))

    row_choices = list(range(0, height - hh + 1))
    col_choices = list(range(0, half_w - hw + 1))
    if height - hh > 0 and random.random() < 0.45:
        row_choices = [0, height - hh]
    if half_w - hw > 0 and random.random() < 0.45:
        col_choices = [0, half_w - hw]

    r0 = random.choice(row_choices)
    local_c0 = random.choice(col_choices)
    r1 = r0 + hh - 1
    local_c1 = local_c0 + hw - 1
    axis = half_w
    if side == "left":
        return r0, local_c0, r1, local_c1
    return r0, axis + 1 + local_c0, r1, axis + 1 + local_c1


def _source_mirror_patch(full: np.ndarray, hole: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = hole
    axis = full.shape[1] // 2
    patch = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            patch[r - r0, c - c0] = full[r, 2 * axis - c]
    return patch


def _wrong_direction_patch(
    source_patch: np.ndarray, side: str, perm: dict[int, int], inv: dict[int, int]
) -> np.ndarray:
    return _apply_perm(source_patch, inv if side == "right" else perm)


def _mask_hole(full: np.ndarray, hole: tuple[int, int, int, int]) -> np.ndarray:
    inp = full.copy()
    r0, c0, r1, c1 = hole
    inp[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return inp


def _semantic_valid(
    inp: np.ndarray,
    full: np.ndarray,
    hole: tuple[int, int, int, int],
    side: str,
    perm: dict[int, int],
) -> bool:
    if max(inp.shape) > 30:
        return False
    expected = _crop(full, hole)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return False
    if len(set(map(int, expected.flat))) < 2:
        return False

    source = _source_mirror_patch(full, hole)
    if np.array_equal(source, expected):
        return False

    inv = _inverse_perm(perm)
    wrong = _wrong_direction_patch(source, side, perm, inv)
    if np.array_equal(wrong, expected):
        return False

    visible = inp.copy()
    r0, c0, r1, c1 = hole
    visible[r0 : r1 + 1, c0 : c1 + 1] = -1
    for color in perm:
        if not np.any(visible == color):
            return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt(side: str) -> np.ndarray | None:
    height = random.randint(MIN_H, MAX_H)
    half_w = random.randint(MIN_HALF_W, MAX_HALF_W)
    palette_size = random.randint(MIN_PALETTE, MAX_PALETTE)
    palette = random.sample(list(ARC_COLORS), palette_size)
    sep_choices = [color for color in ARC_COLORS if color not in palette]
    if not sep_choices:
        return None
    sep = random.choice(sep_choices)
    perm = _cycle_permutation(palette)

    left = _build_left_pattern(height, half_w, palette)
    if set(map(int, np.unique(left))) != set(palette):
        return None

    full = _full_grid(left, sep, perm)
    for _ in range(60):
        hole = _sample_hole(height, half_w, side)
        inp = _mask_hole(full, hole)
        if _semantic_valid(inp, full, hole, side, perm):
            return inp
    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    side = "left" if _GEN_CALL_COUNT % 2 == 0 else "right"
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(side)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid negative mirror patch input")
