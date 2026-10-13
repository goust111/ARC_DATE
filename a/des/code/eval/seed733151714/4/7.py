from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400

COMMON_SIDE_RANGE = (8, 16)
SMALL_SIDE_RANGE = (3, 7)
EXTRA_SIDE_RANGE = (1, 5)
BIG_OBJECT_COUNT_RANGE = (2, 5)
SMALL_OBJECT_COUNT_RANGE = (1, 4)

SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _merge_with_priority(primary: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    out = primary.copy()
    take_overlay = (out == BLACK) & (overlay != BLACK)
    out[take_overlay] = overlay[take_overlay]
    return out


def _full_nonblack_rows(g: np.ndarray) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if int(g[r, 0]) != BLACK and bool(np.all(g[r, :] == g[r, 0]))
    ]


def _full_nonblack_cols(g: np.ndarray) -> list[int]:
    return [
        int(c)
        for c in range(g.shape[1])
        if int(g[0, c]) != BLACK and bool(np.all(g[:, c] == g[0, c]))
    ]


def _find_fold_axis(g: np.ndarray) -> tuple[str, int]:
    row_axes = _full_nonblack_rows(g)
    col_axes = _full_nonblack_cols(g)
    assert len(row_axes) + len(col_axes) == 1, "expected exactly one fold band"
    if row_axes:
        return "horizontal", row_axes[0]
    return "vertical", col_axes[0]


def _prefer_first(first_len: int, second_len: int, first_nonblack: int, second_nonblack: int) -> bool:
    if first_len != second_len:
        return first_len > second_len
    return first_nonblack >= second_nonblack


def _solve_horizontal(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    top_big = _prefer_first(
        int(top.shape[0]),
        int(bottom.shape[0]),
        int(np.count_nonzero(top != BLACK)),
        int(np.count_nonzero(bottom != BLACK)),
    )
    if top_big:
        out = top.copy()
        folded = np.flipud(bottom)
        out[-bottom.shape[0] :, :] = _merge_with_priority(out[-bottom.shape[0] :, :], folded)
        return out

    out = bottom.copy()
    folded = np.flipud(top)
    out[: top.shape[0], :] = _merge_with_priority(out[: top.shape[0], :], folded)
    return out


def _solve_vertical(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_big = _prefer_first(
        int(left.shape[1]),
        int(right.shape[1]),
        int(np.count_nonzero(left != BLACK)),
        int(np.count_nonzero(right != BLACK)),
    )
    if left_big:
        out = left.copy()
        folded = np.fliplr(right)
        out[:, -right.shape[1] :] = _merge_with_priority(out[:, -right.shape[1] :], folded)
        return out

    out = right.copy()
    folded = np.fliplr(left)
    out[:, : left.shape[1]] = _merge_with_priority(out[:, : left.shape[1]], folded)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the smaller side across the single full-row/full-column color band and
    keep the larger side's colors wherever both sides are non-black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orient, axis = _find_fold_axis(g)
    if orient == "horizontal":
        top = g[:axis, :]
        bottom = g[axis + 1 :, :]
        assert top.size and bottom.size
        return _solve_horizontal(top, bottom)

    left = g[:, :axis]
    right = g[:, axis + 1 :]
    assert left.size and right.size
    return _solve_vertical(left, right)


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    return int(h), int(w)


def _place_shape(
    canvas: np.ndarray,
    blocked: np.ndarray,
    allowed: np.ndarray,
    palette: list[int],
) -> bool:
    h, w = map(int, canvas.shape)
    occupied = canvas != BLACK

    for _ in range(80):
        shape = random.choice(SHAPES)
        sh, sw = _shape_size(shape)
        if sh > h or sw > w:
            continue

        r0 = random.randint(0, h - sh)
        c0 = random.randint(0, w - sw)
        coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if not all(bool(allowed[r, c]) for r, c in coords):
            continue
        if any(bool(blocked[r, c]) for r, c in coords):
            continue

        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + sh + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + sw + 1)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            continue

        color = random.choice(palette)
        for r, c in coords:
            canvas[r, c] = int(color)
        return True

    return False


def _random_true_cell(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r, c = coords[random.randrange(len(coords))]
    return int(r), int(c)


def _seed_overlap_cases(
    overlap: np.ndarray,
    overlay: np.ndarray,
    overlap_blocked: np.ndarray,
    overlay_blocked: np.ndarray,
    palette: list[int],
) -> None:
    h, w = map(int, overlay.shape)
    points = random.sample([(r, c) for r in range(h) for c in range(w)], 4)
    conflict, big_only, small_only, both_black = points

    big_color = random.choice(palette)
    small_color = random.choice([c for c in palette if c != big_color])
    overlap[conflict] = int(big_color)
    overlay[conflict] = int(small_color)

    overlap[big_only] = int(random.choice(palette))
    overlay[big_only] = BLACK

    overlap[small_only] = BLACK
    overlay[small_only] = int(random.choice(palette))

    overlap[both_black] = BLACK
    overlay[both_black] = BLACK

    for point in points:
        overlap_blocked[point] = True
        overlay_blocked[point] = True


def _overlap_view(arr: np.ndarray, span: int, axis: int, leading: bool) -> np.ndarray:
    if axis == 0:
        return arr[:span, :] if leading else arr[-span:, :]
    return arr[:, :span] if leading else arr[:, -span:]


def _extra_mask(shape: tuple[int, int], span: int, axis: int, leading: bool) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if axis == 0:
        if leading:
            mask[:span, :] = False
        else:
            mask[-span:, :] = False
    else:
        if leading:
            mask[:, :span] = False
        else:
            mask[:, -span:] = False
    return mask


def _boost_big_side(
    big: np.ndarray,
    big_blocked: np.ndarray,
    extra_allowed: np.ndarray,
    palette: list[int],
) -> None:
    preferred = extra_allowed & (big == BLACK) & (~big_blocked)
    cell = _random_true_cell(preferred)
    if cell is None:
        cell = _random_true_cell((big == BLACK) & (~big_blocked))
    if cell is not None:
        big[cell] = int(random.choice(palette))


def _compose_case(case: str, band_color: int, big: np.ndarray, overlay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if case == "top_big":
        band = np.full((1, big.shape[1]), int(band_color), dtype=int)
        grid = np.vstack([big, band, np.flipud(overlay)])
        expected = big.copy()
        expected[-overlay.shape[0] :, :] = _merge_with_priority(expected[-overlay.shape[0] :, :], overlay)
        return grid, expected

    if case == "bottom_big":
        band = np.full((1, big.shape[1]), int(band_color), dtype=int)
        grid = np.vstack([np.flipud(overlay), band, big])
        expected = big.copy()
        expected[: overlay.shape[0], :] = _merge_with_priority(expected[: overlay.shape[0], :], overlay)
        return grid, expected

    if case == "left_big":
        band = np.full((big.shape[0], 1), int(band_color), dtype=int)
        grid = np.hstack([big, band, np.fliplr(overlay)])
        expected = big.copy()
        expected[:, -overlay.shape[1] :] = _merge_with_priority(expected[:, -overlay.shape[1] :], overlay)
        return grid, expected

    band = np.full((big.shape[0], 1), int(band_color), dtype=int)
    grid = np.hstack([np.fliplr(overlay), band, big])
    expected = big.copy()
    expected[:, : overlay.shape[1]] = _merge_with_priority(expected[:, : overlay.shape[1]], overlay)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    case = random.choice(("top_big", "bottom_big", "left_big", "right_big"))
    band_color = random.choice(COLORS)
    palette = random.sample([c for c in COLORS if c != band_color], random.randint(3, 6))

    if case in {"top_big", "bottom_big"}:
        width = random.randint(*COMMON_SIDE_RANGE)
        small_h = random.randint(*SMALL_SIDE_RANGE)
        extra_h = random.randint(*EXTRA_SIDE_RANGE)
        big = np.zeros((small_h + extra_h, width), dtype=int)
        overlay = np.zeros((small_h, width), dtype=int)
        axis = 0
        span = small_h
        leading = case == "bottom_big"
    else:
        height = random.randint(*COMMON_SIDE_RANGE)
        small_w = random.randint(*SMALL_SIDE_RANGE)
        extra_w = random.randint(*EXTRA_SIDE_RANGE)
        big = np.zeros((height, small_w + extra_w), dtype=int)
        overlay = np.zeros((height, small_w), dtype=int)
        axis = 1
        span = small_w
        leading = case == "right_big"

    big_blocked = np.zeros_like(big, dtype=bool)
    overlay_blocked = np.zeros_like(overlay, dtype=bool)
    overlap = _overlap_view(big, span, axis, leading)
    overlap_blocked = _overlap_view(big_blocked, span, axis, leading)
    extra_allowed = _extra_mask(big.shape, span, axis, leading)

    _seed_overlap_cases(overlap, overlay, overlap_blocked, overlay_blocked, palette)

    extra_cell = _random_true_cell(extra_allowed & (~big_blocked))
    if extra_cell is not None:
        big[extra_cell] = int(random.choice(palette))
        big_blocked[extra_cell] = True

    allowed_big = np.ones_like(big, dtype=bool)
    allowed_overlay = np.ones_like(overlay, dtype=bool)
    for _ in range(random.randint(*BIG_OBJECT_COUNT_RANGE)):
        _place_shape(big, big_blocked, allowed_big, palette)
    for _ in range(random.randint(*SMALL_OBJECT_COUNT_RANGE)):
        _place_shape(overlay, overlay_blocked, allowed_overlay, palette)

    while int(np.count_nonzero(big != BLACK)) <= int(np.count_nonzero(overlay != BLACK)):
        before = int(np.count_nonzero(big != BLACK))
        _boost_big_side(big, big_blocked, extra_allowed, palette)
        if int(np.count_nonzero(big != BLACK)) == before:
            break

    grid, expected = _compose_case(case, band_color, big, overlay)

    row_axes = _full_nonblack_rows(grid)
    col_axes = _full_nonblack_cols(grid)
    if case in {"top_big", "bottom_big"}:
        if len(row_axes) != 1 or col_axes:
            return None
    else:
        if len(col_axes) != 1 or row_axes:
            return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding input")
