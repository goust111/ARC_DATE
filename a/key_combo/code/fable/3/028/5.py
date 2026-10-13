from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 600


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rot_cw(patch: np.ndarray) -> np.ndarray:
    return np.rot90(patch, -1).copy()


def _render_band(
    final_patch: np.ndarray,
    width: int,
    bg: int,
    sign: int,
    stamp_count: int,
    first_gap: int,
) -> np.ndarray:
    side = int(final_patch.shape[0])
    gaps = [int(first_gap) + i for i in range(int(stamp_count) - 1)]
    span = sum(gaps)
    if sign == 1:
        anchors = [0]
        for gap in gaps:
            anchors.append(anchors[-1] + gap)
    else:
        anchors = [span]
        for gap in gaps:
            anchors.append(anchors[-1] - gap)

    out = np.full((side, int(width)), int(bg), dtype=int)
    for index, anchor in enumerate(anchors):
        patch = np.rot90(final_patch, int(stamp_count) - 1 - index).copy()
        out[:, anchor : anchor + side] = patch
    return out


def _find_solutions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    box = _bbox(g != bg)
    assert box is not None, "expected a colored stamp trail"
    r0, c0, r1, c1 = box
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    side, width = map(int, crop.shape)

    assert 3 <= side <= width, "expected one horizontal band of square stamps"
    assert np.all(crop != bg), "the visible stamp band must be solidly covered"
    assert width > side, "expected more than one stamp in the trail"

    span = width - side
    solutions: list[dict] = []
    for sign, final_patch, final_anchor in (
        (1, crop[:, width - side : width].copy(), span),
        (-1, crop[:, :side].copy(), 0),
    ):
        if np.any(final_patch == bg):
            continue
        for stamp_count in range(4, width + 1):
            gap_count = stamp_count - 1
            base = gap_count * (gap_count - 1) // 2
            numerator = span - base
            if numerator < gap_count or numerator % gap_count != 0:
                continue
            first_gap = numerator // gap_count
            last_gap = first_gap + gap_count - 1
            if first_gap < 1 or last_gap >= side:
                continue
            rendered = _render_band(final_patch, width, bg, sign, stamp_count, first_gap)
            if np.array_equal(rendered, crop):
                solutions.append(
                    {
                        "bg": bg,
                        "r0": int(r0),
                        "c0": int(c0),
                        "side": side,
                        "width": width,
                        "sign": int(sign),
                        "final_patch": final_patch,
                        "last_anchor_c": int(c0 + final_anchor),
                        "stamp_count": int(stamp_count),
                        "first_gap": int(first_gap),
                        "last_gap": int(last_gap),
                    }
                )
    return solutions


def _continue_from_solution(grid: np.ndarray, solution: dict) -> tuple[np.ndarray, int]:
    out = np.asarray(grid, dtype=int).copy()
    side = int(solution["side"])
    r0 = int(solution["r0"])
    c = int(solution["last_anchor_c"])
    sign = int(solution["sign"])
    gap = int(solution["last_gap"]) + 1
    patch = _rot_cw(solution["final_patch"])
    added = 0

    while True:
        c_next = c + sign * gap
        if r0 < 0 or r0 + side > out.shape[0] or c_next < 0 or c_next + side > out.shape[1]:
            break
        out[r0 : r0 + side, c_next : c_next + side] = patch
        added += 1
        c = c_next
        gap += 1
        patch = _rot_cw(patch)
    return out, added


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the occluded rotating stamp trail and extend it with increasing gaps.

    Input structure:
    - A black-background grid contains one horizontal band made by repeatedly stamping the same solid
      non-symmetric square color sprite.
    - Consecutive stamp anchors lie on the same row; their horizontal gaps increase by one cell.
    - Every input gap is smaller than the sprite side, so each early stamp is partly covered by the next
      stamp, while the final stamp remains fully visible at one horizontal end of the band.

    Rule:
    - Keep the input image, including all visible covered-trail pixels.
    - Use the foreground band's height as the square sprite side and test the left and right end squares
      as possible final complete stamps.
    - The unique end square whose counter-rotated copies reproduce the whole band with a +1 gap sequence
      fixes the time direction, the last gap, and the final rotation.
    - Continue stamping from that end: rotate the sprite clockwise once, increase the gap by one, paint the
      full square over existing pixels, and stop before the first stamp that would not fit.

    Key insight:
    - The last uncovered square is the only complete sprite view; the overlapped color trace determines
      which end is later and which increasing-gap, four-rotation sequence produced it.
    """
    solutions = _find_solutions(np.asarray(grid, dtype=int))
    assert len(solutions) == 1, "expected a unique rotating stamp reconstruction"
    out, _added = _continue_from_solution(np.asarray(grid, dtype=int), solutions[0])
    return out


def _rotations_unique(sprite: np.ndarray) -> bool:
    variants = {tuple(np.rot90(sprite, k).ravel().tolist()) for k in range(4)}
    return len(variants) == 4


def _make_sprite(side: int) -> np.ndarray:
    for _ in range(300):
        palette = random.sample(COLORS, random.choice((3, 3, 4)))
        weights = [0.52, 0.27, 0.16, 0.05][: len(palette)]
        flat = random.choices(palette, weights=weights, k=side * side)
        sprite = np.array(flat, dtype=int).reshape((side, side))

        # Force an asymmetric, easy-to-see colored frame and diagonal accent.
        sprite[0, : side - 1] = palette[1]
        sprite[1:, 1] = palette[1]
        sprite[side - 2, 2:] = palette[2]
        sprite[2:, side - 1] = palette[2]
        sprite[side - 1, 0] = palette[-1]
        sprite[side // 2, side // 2] = palette[-1]
        for _accent in range(random.randint(side // 2, side + 1)):
            sprite[random.randrange(side), random.randrange(side)] = random.choice(palette)

        sprite = np.rot90(sprite, random.randrange(4)).copy()
        if random.random() < 0.5:
            sprite = np.fliplr(sprite).copy()
        if len(set(map(int, sprite.ravel()))) == len(palette) and _rotations_unique(sprite):
            return sprite
    raise RuntimeError("failed to build an asymmetric sprite")


def _sequence_options(side: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for stamp_count in range(4, side + 2):
        for first_gap in range(1, side):
            last_gap = first_gap + stamp_count - 2
            if last_gap >= side:
                continue
            span = sum(first_gap + i for i in range(stamp_count - 1))
            if span + (last_gap + 1) + side <= 30:
                options.append((stamp_count, first_gap))
    longer = [item for item in options if item[0] >= 5]
    if longer and random.random() < 0.8:
        return longer
    return options


def _anchors(first_c: int, sign: int, stamp_count: int, first_gap: int) -> list[int]:
    anchors = [int(first_c)]
    for i in range(int(stamp_count) - 1):
        anchors.append(anchors[-1] + int(sign) * (int(first_gap) + i))
    return anchors


def _render_input(
    rows: int,
    cols: int,
    row0: int,
    anchors: list[int],
    sprite: np.ndarray,
) -> np.ndarray:
    side = int(sprite.shape[0])
    grid = np.full((int(rows), int(cols)), BG, dtype=int)
    patch = sprite.copy()
    for c in anchors:
        grid[row0 : row0 + side, c : c + side] = patch
        patch = _rot_cw(patch)
    return grid


def _expected_from_metadata(
    grid: np.ndarray,
    row0: int,
    final_c: int,
    sign: int,
    last_gap: int,
    final_patch: np.ndarray,
) -> tuple[np.ndarray, int]:
    out = grid.copy()
    side = int(final_patch.shape[0])
    c = int(final_c)
    gap = int(last_gap) + 1
    patch = _rot_cw(final_patch)
    added = 0
    while True:
        c_next = c + int(sign) * gap
        if c_next < 0 or c_next + side > out.shape[1]:
            break
        out[row0 : row0 + side, c_next : c_next + side] = patch
        added += 1
        c = c_next
        gap += 1
        patch = _rot_cw(patch)
    return out, added


def _semantic_accepts(
    grid: np.ndarray,
    row0: int,
    anchors: list[int],
    sign: int,
    first_gap: int,
    sprite: np.ndarray,
) -> bool:
    side = int(sprite.shape[0])
    stamp_count = len(anchors)
    last_gap = first_gap + stamp_count - 2
    final_c = anchors[-1]
    final_patch = np.rot90(sprite, -(stamp_count - 1)).copy()
    expected, added = _expected_from_metadata(grid, row0, final_c, sign, last_gap, final_patch)
    if added < 1:
        return False

    try:
        solutions = _find_solutions(grid)
        actual = main(grid)
    except Exception:
        return False
    if len(solutions) != 1:
        return False
    sol = solutions[0]
    if not (
        sol["sign"] == sign
        and sol["stamp_count"] == stamp_count
        and sol["first_gap"] == first_gap
        and sol["last_gap"] == last_gap
        and sol["r0"] == row0
        and sol["last_anchor_c"] == final_c
        and np.array_equal(sol["final_patch"], final_patch)
    ):
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    # Every non-final input stamp must lose cells to its immediate successor.
    for a, b in zip(anchors, anchors[1:]):
        if abs(a - b) >= side:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    side = random.choice((5, 5, 6))
    options = _sequence_options(side)
    if not options:
        return None
    stamp_count, first_gap = random.choice(options)
    gaps = [first_gap + i for i in range(stamp_count - 1)]
    span = sum(gaps)
    last_gap = gaps[-1]
    next_gap = last_gap + 1
    sign = random.choice((-1, 1))

    min_cols = span + next_gap + side
    if min_cols > 30:
        return None
    cols = random.randint(max(22, min_cols), 30)
    rows = random.randint(max(side + 6, 14), 30)
    row0 = random.randint(1, rows - side - 1)

    if sign == 1:
        first_c = random.randint(0, cols - min_cols)
    else:
        final_min = next_gap
        final_max = cols - span - side
        if final_min > final_max:
            return None
        final_c = random.randint(final_min, final_max)
        first_c = final_c + span

    sprite = _make_sprite(side)
    anchors = _anchors(first_c, sign, stamp_count, first_gap)
    if min(anchors) < 0 or max(c + side for c in anchors) > cols:
        return None

    grid = _render_input(rows, cols, row0, anchors, sprite)
    if not _semantic_accepts(grid, row0, anchors, sign, first_gap, sprite):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotating stamp trail")
