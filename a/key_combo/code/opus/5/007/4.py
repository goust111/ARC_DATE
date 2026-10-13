from __future__ import annotations

import random

import numpy as np


BG = 0
PATTERN_COLORS = (1, 2, 3, 4)
GREY = 5

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
MAX_GENERATION_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _complete_by_axis(grid: np.ndarray, orientation: str, axis: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    if orientation == "vertical":
        for c in range(W):
            mirror_c = 2 * int(axis) - int(c)
            if mirror_c < 0 or mirror_c >= W or c > mirror_c:
                continue
            for r in range(H):
                a = int(g[r, c])
                b = int(g[r, mirror_c])
                if a != BG and b != BG:
                    assert a == b, f"conflicting mirrored colors at ({r},{c}) and ({r},{mirror_c})"
                elif a == BG and b != BG:
                    out[r, c] = b
                elif b == BG and a != BG:
                    out[r, mirror_c] = a
    else:
        for r in range(H):
            mirror_r = 2 * int(axis) - int(r)
            if mirror_r < 0 or mirror_r >= H or r > mirror_r:
                continue
            for c in range(W):
                a = int(g[r, c])
                b = int(g[mirror_r, c])
                if a != BG and b != BG:
                    assert a == b, f"conflicting mirrored colors at ({r},{c}) and ({mirror_r},{c})"
                elif a == BG and b != BG:
                    out[r, c] = b
                elif b == BG and a != BG:
                    out[mirror_r, c] = a
    return out


def _helper_info(grid: np.ndarray) -> tuple[str, int, np.ndarray]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.shape == (2, 2), f"expected exactly one 2-cell grey helper, got {grey_coords.shape[0]} cells"

    r0, c0, r1, c1 = _bbox_coords(grey_coords)
    if r0 == r1 and c1 == c0 + 1:
        assert c0 == 0 or c1 == grid.shape[1] - 1, "horizontal helper must touch left or right edge"
        return "horizontal", int(r0), grey_coords
    if c0 == c1 and r1 == r0 + 1:
        assert r0 == 0 or r1 == grid.shape[0] - 1, "vertical helper must touch top or bottom edge"
        return "vertical", int(c0), grey_coords
    raise AssertionError("grey helper must be 1x2 or 2x1")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the edge helper rectangle to recover the reflection axis and complete the missing half of the pattern.

    Input structure:
    - Black background with one multicolor pattern made from colors 1-4.
    - The pattern is intended to be mirror-symmetric about one single grid row or one single grid column, but one rectangular area on exactly one side has been erased to background.
    - A grey 2-cell rectangle touches one outer edge; a horizontal 1x2 helper encodes a row axis, while a vertical 2x1 helper encodes a column axis.

    Rule:
    - Read the helper rectangle's row or column as the reflection axis implied by the erased slide trail.
    - Remove the grey helper from consideration.
    - For every pair of positions mirrored across that axis, if one side is colored and the other side is background, copy the color into the background side; if both sides are colored they must already agree.
    - Return the completed pattern with the helper removed.

    Key insight:
    - The grey edge rectangle is not an object to preserve; it is a compact code for the hidden mirror axis that uniquely determines how to refill the erased rectangle.
    """
    g = np.asarray(grid, dtype=int)
    orientation, axis, grey_coords = _helper_info(g)

    work = g.copy()
    work[grey_coords[:, 0], grey_coords[:, 1]] = BG
    return _complete_by_axis(work, orientation, axis)


def _build_half_patch(height: int, width: int, colors: tuple[int, ...]) -> np.ndarray | None:
    for _ in range(150):
        patch = np.zeros((height, width), dtype=int)
        n_shapes = random.randint(len(colors) + 2, len(colors) + 5)
        color_sequence = list(colors) + [random.choice(colors) for _ in range(n_shapes - len(colors))]
        random.shuffle(color_sequence)

        for color in color_sequence:
            placed = False
            for _ in range(40):
                rh = random.randint(1, min(4, height))
                rw = random.randint(1, min(3, width))
                rr = random.randint(0, height - rh)
                cc = random.randint(0, width - rw)
                region = patch[rr : rr + rh, cc : cc + rw]
                if np.count_nonzero(region) > (2 * rh * rw) // 3:
                    continue
                patch[rr : rr + rh, cc : cc + rw] = int(color)
                placed = True
                break
            if not placed:
                break

        for _ in range(random.randint(0, 2)):
            if height < 3 or width < 3:
                break
            rh = random.randint(1, min(2, height - 1))
            rw = random.randint(1, min(2, width - 1))
            rr = random.randint(0, height - rh)
            cc = random.randint(0, width - rw)
            region = patch[rr : rr + rh, cc : cc + rw]
            if np.count_nonzero(region) >= max(2, rh * rw - 1):
                patch[rr : rr + rh, cc : cc + rw] = BG

        present = tuple(sorted(int(v) for v in np.unique(patch) if v != BG))
        filled = int(np.count_nonzero(patch))
        row_hits = int(np.count_nonzero(np.any(patch != BG, axis=1)))
        col_hits = int(np.count_nonzero(np.any(patch != BG, axis=0)))
        if present != tuple(sorted(colors)):
            continue
        if filled < max(14, height * width // 3) or filled > int(0.78 * height * width):
            continue
        if row_hits < max(5, height - 2) or col_hits < max(3, width - 1):
            continue
        if not np.any(patch[:, 0] != BG) or not np.any(patch[:, -1] != BG):
            continue
        if min(int(np.count_nonzero(patch == color)) for color in colors) < 2:
            continue
        return patch
    return None


def _render_vertical_pattern(shape: tuple[int, int], half: np.ndarray, axis: int, top: int) -> np.ndarray:
    H, W = shape
    out = np.zeros((H, W), dtype=int)
    ph, pw = half.shape
    for rr in range(ph):
        for jj in range(pw):
            color = int(half[rr, jj])
            if color == BG:
                continue
            gr = int(top + rr)
            left_c = int(axis - (jj + 1))
            right_c = int(axis + (jj + 1))
            out[gr, left_c] = color
            out[gr, right_c] = color
    return out


def _clear_vertical_hole(grid: np.ndarray, axis: int, top: int, rect: tuple[int, int, int, int], side: str) -> np.ndarray:
    out = grid.copy()
    rr0, cc0, hh, ww = rect
    for rr in range(rr0, rr0 + hh):
        gr = int(top + rr)
        for jj in range(cc0, cc0 + ww):
            gc = int(axis - (jj + 1)) if side == "left" else int(axis + (jj + 1))
            out[gr, gc] = BG
    return out


def _choose_hole(full: np.ndarray, half: np.ndarray, axis: int, top: int, side: str) -> tuple[int, int, int, int] | None:
    ph, pw = half.shape
    candidates: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    max_h = min(5, ph)
    max_w = min(4, pw)

    for hh in range(2, max_h + 1):
        for ww in range(2, max_w + 1):
            for rr0 in range(0, ph - hh + 1):
                for cc0 in range(0, pw - ww + 1):
                    ref = half[rr0 : rr0 + hh, cc0 : cc0 + ww]
                    filled = int(np.count_nonzero(ref))
                    colors = sorted(int(v) for v in np.unique(ref) if v != BG)
                    if filled < 4 or len(colors) < 2:
                        continue

                    trial = _clear_vertical_hole(full, axis, top, (rr0, cc0, hh, ww), side)
                    bbox = _bbox(trial != BG)
                    if bbox is None:
                        continue
                    if bbox[1] + bbox[3] == 2 * axis:
                        continue
                    if np.array_equal(trial, full):
                        continue

                    mixed_bg = int(np.any(ref == BG))
                    shift = abs((bbox[1] + bbox[3]) - 2 * axis)
                    score = (mixed_bg, shift, len(colors), filled)
                    candidates.append(((rr0, cc0, hh, ww), score))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_score = candidates[0][1]
    best_rects = [rect for rect, score in candidates if score == best_score]
    return random.choice(best_rects)


def _add_vertical_helper(grid: np.ndarray, axis: int) -> tuple[np.ndarray, str]:
    out = grid.copy()
    if random.random() < 0.5:
        out[0:2, axis] = GREY
        return out, "top"
    out[-2:, axis] = GREY
    return out, "bottom"


def _wrong_bbox_axis_differs(inp: np.ndarray, expected: np.ndarray, true_orientation: str, true_axis: int) -> bool:
    clean = inp.copy()
    clean[clean == GREY] = BG
    bbox = _bbox(clean != BG)
    if bbox is None:
        return False

    if true_orientation == "vertical":
        if (bbox[1] + bbox[3]) % 2 != 0:
            return True
        wrong_axis = (bbox[1] + bbox[3]) // 2
    else:
        if (bbox[0] + bbox[2]) % 2 != 0:
            return True
        wrong_axis = (bbox[0] + bbox[2]) // 2

    if wrong_axis == true_axis:
        return False
    try:
        wrong = _complete_by_axis(clean, true_orientation, int(wrong_axis))
    except AssertionError:
        return True
    return not np.array_equal(wrong, expected)


def _validate_example(
    inp: np.ndarray,
    expected: np.ndarray,
    orientation: str,
    axis: int,
    hole_side: str,
    hole_rect: tuple[int, int, int, int],
) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    if np.any(expected == GREY):
        return False
    if orientation == "vertical":
        if np.any(expected[:, axis] != BG):
            return False
        for c in range(expected.shape[1]):
            mc = 2 * axis - c
            if 0 <= mc < expected.shape[1]:
                if not np.array_equal(expected[:, c], expected[:, mc]):
                    return False
    else:
        if np.any(expected[axis, :] != BG):
            return False
        for r in range(expected.shape[0]):
            mr = 2 * axis - r
            if 0 <= mr < expected.shape[0]:
                if not np.array_equal(expected[r, :], expected[mr, :]):
                    return False

    clean = inp.copy()
    clean[clean == GREY] = BG
    delta = expected != clean
    if int(np.count_nonzero(delta)) < 4:
        return False

    bbox = _bbox(clean != BG)
    if bbox is None:
        return False
    if orientation == "vertical":
        if bbox[1] + bbox[3] == 2 * axis:
            return False
    else:
        if bbox[0] + bbox[2] == 2 * axis:
            return False

    if not _wrong_bbox_axis_differs(inp, expected, orientation, axis):
        return False

    colors = sorted(int(v) for v in np.unique(expected) if v != BG)
    if len(colors) not in (3, 4):
        return False

    rr0, cc0, hh, ww = hole_rect
    if hh < 2 or ww < 2:
        return False
    if orientation == "vertical" and hole_side not in ("left", "right"):
        return False
    if orientation == "horizontal" and hole_side not in ("top", "bottom"):
        return False
    return True


def _generate_vertical_example() -> tuple[np.ndarray, np.ndarray, int, str, tuple[int, int, int, int]]:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    axis = random.randint(4, W - 5)

    max_side = min(7, axis - 1, W - axis - 2)
    if max_side < 4:
        raise RuntimeError("insufficient width for vertical symmetry")
    half_w = random.randint(4, max_side)

    min_half_h = max(7, H - 6)
    if min_half_h > H - 2:
        raise RuntimeError("insufficient height for pattern")
    half_h = random.randint(min_half_h, H - 2)
    top = random.randint(1, H - half_h - 1)

    n_colors = random.randint(3, 4)
    colors = tuple(sorted(random.sample(PATTERN_COLORS, n_colors)))
    half = _build_half_patch(half_h, half_w, colors)
    if half is None:
        raise RuntimeError("failed to build half patch")

    expected = _render_vertical_pattern((H, W), half, axis, top)
    missing_side = random.choice(("left", "right"))
    hole_rect = _choose_hole(expected, half, axis, top, missing_side)
    if hole_rect is None:
        missing_side = "left" if missing_side == "right" else "right"
        hole_rect = _choose_hole(expected, half, axis, top, missing_side)
    if hole_rect is None:
        raise RuntimeError("failed to find informative hole")

    inp = _clear_vertical_hole(expected, axis, top, hole_rect, missing_side)
    inp, _edge = _add_vertical_helper(inp, axis)
    return inp, expected, axis, missing_side, hole_rect


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected, axis, side, hole_rect = _generate_vertical_example()
        except RuntimeError:
            continue

        orientation = "vertical"
        if random.random() < 0.5:
            inp = inp.T.copy()
            expected = expected.T.copy()
            orientation = "horizontal"
            axis = int(axis)
            side = "top" if side == "left" else "bottom"
            hole_rect = hole_rect

        if _validate_example(inp, expected, orientation, axis, side, hole_rect):
            return inp
    raise RuntimeError("failed to generate a valid input")
