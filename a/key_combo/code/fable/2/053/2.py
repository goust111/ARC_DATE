from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


# Minimal helpers adapted from arc_utils: (r, c) coords, inclusive bbox semantics,
# and 4-connected component extraction on boolean masks.


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((int(cr), int(cc2)))
            for nr, nc in ((cr - 1, cc2), (cr + 1, cc2), (cr, cc2 - 1), (cr, cc2 + 1)):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(np.array(coords, dtype=int))
    return comps


def _inside_square(pos: tuple[int, int], size: int, shape: tuple[int, int]) -> bool:
    r, c = map(int, pos)
    h, w = map(int, shape)
    s = int(size)
    return 0 <= r and 0 <= c and r + s <= h and c + s <= w


def _rectangles_overlap(a: tuple[int, int], b: tuple[int, int], size: int) -> bool:
    ar, ac = map(int, a)
    br, bc = map(int, b)
    s = int(size)
    return not (ar + s <= br or br + s <= ar or ac + s <= bc or bc + s <= ac)


def _paint_square(grid: np.ndarray, pos: tuple[int, int], size: int, color: int) -> None:
    r, c = map(int, pos)
    s = int(size)
    grid[r : r + s, c : c + s] = int(color)


def _phase_color(index: int, phase: int, color_a: int, color_b: int) -> int:
    return int(color_a if (int(index) + int(phase)) % 2 == 0 else color_b)


def _infer_initial_sequence(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], tuple[int, int], int, int, int]:
    bg = _get_bg(grid)
    comps: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components_mask(grid == int(color)):
            r0, c0, r1, c1 = _bbox_coords(coords)
            size_r = int(r1 - r0 + 1)
            size_c = int(c1 - c0 + 1)
            assert size_r == size_c, "each foreground component must be a square"
            assert int(coords.shape[0]) == size_r * size_r, "square must be solid"
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert bool((patch == int(color)).all()), "square must be monochrome"
            comps.append({"top_left": (int(r0), int(c0)), "color": int(color), "size": int(size_r)})

    assert len(comps) == 3, "expected exactly three square markers"
    sizes = {int(comp["size"]) for comp in comps}
    assert len(sizes) == 1, "all markers must have the same size"
    size = int(next(iter(sizes)))

    color_counts = Counter(int(comp["color"]) for comp in comps)
    assert sorted(color_counts.values()) == [1, 2], "colors must follow A,B,A with two colors"

    matches: list[tuple[list[tuple[int, int]], tuple[int, int], int, int]] = []
    for perm in itertools.permutations(comps):
        p0 = tuple(perm[0]["top_left"])
        p1 = tuple(perm[1]["top_left"])
        p2 = tuple(perm[2]["top_left"])
        v1 = (int(p1[0] - p0[0]), int(p1[1] - p0[1]))
        v2 = (int(p2[0] - p1[0]), int(p2[1] - p1[1]))
        if v1 != v2:
            continue
        if v1[0] == 0 or v1[1] == 0 or abs(v1[0]) != abs(v1[1]):
            continue
        c0 = int(perm[0]["color"])
        c1 = int(perm[1]["color"])
        c2 = int(perm[2]["color"])
        prev_pos = (int(p0[0] - v1[0]), int(p0[1] - v1[1]))
        if c0 == c2 and c0 != c1 and not _inside_square(prev_pos, size, grid.shape):
            matches.append(([p0, p1, p2], v1, c0, c1))

    assert len(matches) == 1, "initial order must be uniquely determined"
    positions, step, color_a, color_b = matches[0]
    return int(bg), positions, (int(step[0]), int(step[1])), int(color_a), int(color_b), int(size)


def _simulate_sequence(
    shape: tuple[int, int],
    size: int,
    initial_positions: list[tuple[int, int]],
    step: tuple[int, int],
    color_a: int,
    color_b: int,
    bg: int,
    *,
    swap_phase_on_bounce: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)

    positions = [tuple(map(int, pos)) for pos in initial_positions]
    colors = [int(color_a), int(color_b), int(color_a)]
    for pos, color in zip(positions, colors):
        _paint_square(out, pos, size, color)

    cur = positions[-1]
    vr, vc = map(int, step)
    phase = 0
    bounce_count = 0
    placed_after_bounce = 0
    first_post_bounce_color: int | None = None
    max_squares = (h * w) // max(1, int(size) * int(size)) + 4

    while len(positions) <= max_squares:
        next_index = len(positions)
        cand = (int(cur[0] + vr), int(cur[1] + vc))
        if not _inside_square(cand, size, shape):
            new_vr = -vr if cand[0] < 0 or cand[0] + size > h else vr
            new_vc = -vc if cand[1] < 0 or cand[1] + size > w else vc
            assert (new_vr, new_vc) != (vr, vc), "bounce must flip at least one axis"
            vr, vc = int(new_vr), int(new_vc)
            if swap_phase_on_bounce:
                phase ^= 1
            bounce_count += 1
            cand = (int(cur[0] + vr), int(cur[1] + vc))
            if not _inside_square(cand, size, shape):
                return out, {
                    "positions": tuple(positions),
                    "colors": tuple(colors),
                    "bounces": int(bounce_count),
                    "placed_after_bounce": int(placed_after_bounce),
                    "first_post_bounce_color": first_post_bounce_color,
                    "term": "oob_after_reflect",
                }

        next_color = _phase_color(next_index, phase, color_a, color_b)
        if any(_rectangles_overlap(cand, old, size) for old in positions):
            return out, {
                "positions": tuple(positions),
                "colors": tuple(colors),
                "bounces": int(bounce_count),
                "placed_after_bounce": int(placed_after_bounce),
                "first_post_bounce_color": first_post_bounce_color,
                "term": "overlap",
            }

        _paint_square(out, cand, size, next_color)
        positions.append(cand)
        colors.append(int(next_color))
        if bounce_count > 0:
            placed_after_bounce += 1
            if first_post_bounce_color is None:
                first_post_bounce_color = int(next_color)
        cur = cand

    raise AssertionError("sequence failed to terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a diagonal square sequence by bouncing it off the canvas while swapping the two-color phase.

    Input structure:
    - A background canvas contains exactly three equal solid squares on a 45-degree arithmetic progression.
    - The three marker squares use two colors in an A, B, A order.
    - All three initial squares fit strictly inside the canvas, and only one end could not be preceded by another same-step square without leaving the canvas.

    Rule:
    - Infer the step vector from the first two squares in the unique diagonal A, B, A ordering.
    - Repeatedly try to place the next square at the current square plus that vector, continuing the A/B alternation.
    - If the next square would leave the canvas, flip every out-of-bounds vector component, swap the A/B phase, and recompute the landing from the same current square.
    - Stop without placing a square if the reflected landing is still out of bounds or if it overlaps any previously placed square; otherwise paint the new square and continue.

    Key insight:
    - The path behaves like a billiard in square-sized top-left coordinates, and every bounce also swaps which parity uses color A versus color B.
    """
    g = np.asarray(grid, dtype=int)
    bg, positions, step, color_a, color_b, size = _infer_initial_sequence(g)
    out, _meta = _simulate_sequence(
        g.shape,
        size,
        positions,
        step,
        color_a,
        color_b,
        bg,
        swap_phase_on_bounce=True,
    )
    return out


GEN_RETRIES = 1200
_GEN_MODE = 0


def _sample_palette() -> tuple[int, int, int]:
    bg = random.randrange(10)
    color_a, color_b = random.sample([c for c in range(10) if c != bg], 2)
    return int(bg), int(color_a), int(color_b)


def _axis_start_candidates(max_top: int, size: int, step_mag: int, pre_bounce_steps: int, sign: int, should_bounce_next: bool) -> list[int]:
    vals: list[int] = []
    for p0 in range(1, int(max_top)):
        p2 = int(p0 + sign * 2 * step_mag)
        if not (1 <= p2 <= int(max_top) - 1):
            continue
        current = int(p0 + sign * (2 + pre_bounce_steps) * step_mag)
        nxt = int(p0 + sign * (3 + pre_bounce_steps) * step_mag)
        if not (0 <= current <= int(max_top)):
            continue
        will_bounce = not (0 <= nxt <= int(max_top))
        if bool(will_bounce) == bool(should_bounce_next):
            vals.append(int(p0))
    return vals


def _build_case(
    shape: tuple[int, int],
    size: int,
    step: tuple[int, int],
    start: tuple[int, int],
    color_a: int,
    color_b: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object], np.ndarray]:
    positions = [
        (int(start[0]), int(start[1])),
        (int(start[0] + step[0]), int(start[1] + step[1])),
        (int(start[0] + 2 * step[0]), int(start[1] + 2 * step[1])),
    ]
    inp = np.full(shape, int(bg), dtype=int)
    for pos, color in zip(positions, (color_a, color_b, color_a)):
        _paint_square(inp, pos, size, color)

    expected, meta = _simulate_sequence(
        shape,
        size,
        positions,
        step,
        color_a,
        color_b,
        bg,
        swap_phase_on_bounce=True,
    )
    counterfactual, no_swap_meta = _simulate_sequence(
        shape,
        size,
        positions,
        step,
        color_a,
        color_b,
        bg,
        swap_phase_on_bounce=False,
    )
    return inp, expected, meta, no_swap_meta, counterfactual


def _sample_corner_stop_case() -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object], np.ndarray] | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    size = random.randint(2, 4)
    max_d = min(8, h - size - 2, w - size - 2)
    if max_d < size + 1:
        return None

    step_mag = random.randint(size + 1, max_d)
    sr = random.choice([-1, 1])
    sc = random.choice([-1, 1])
    pre_bounce_steps = random.randint(1, 4)
    row_vals = _axis_start_candidates(h - size, size, step_mag, pre_bounce_steps, sr, True)
    col_vals = _axis_start_candidates(w - size, size, step_mag, pre_bounce_steps, sc, True)
    if not row_vals or not col_vals:
        return None

    bg, color_a, color_b = _sample_palette()
    start = (random.choice(row_vals), random.choice(col_vals))
    return _build_case((h, w), size, (sr * step_mag, sc * step_mag), start, color_a, color_b, bg)


def _sample_side_bounce_case() -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object], np.ndarray] | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    size = random.randint(2, 4)
    max_d = min(8, h - size - 2, w - size - 2)
    if max_d < size + 1:
        return None

    step_mag = random.randint(size + 1, max_d)
    sr = random.choice([-1, 1])
    sc = random.choice([-1, 1])
    pre_bounce_steps = random.randint(0, 3)
    first_bounce_axis = random.choice([0, 1])  # 0=row, 1=col
    row_vals = _axis_start_candidates(h - size, size, step_mag, pre_bounce_steps, sr, first_bounce_axis == 0)
    col_vals = _axis_start_candidates(w - size, size, step_mag, pre_bounce_steps, sc, first_bounce_axis == 1)
    if not row_vals or not col_vals:
        return None

    bg, color_a, color_b = _sample_palette()
    start = (random.choice(row_vals), random.choice(col_vals))
    return _build_case((h, w), size, (sr * step_mag, sc * step_mag), start, color_a, color_b, bg)


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    meta: dict[str, object],
    no_swap_meta: dict[str, object],
    counterfactual: np.ndarray,
    mode: str,
) -> bool:
    try:
        bg, positions, step, color_a, color_b, size = _infer_initial_sequence(inp)
    except AssertionError:
        return False

    del bg, step, color_a, color_b, size  # validated by successful inference
    if list(positions) != list(meta["positions"][:3]):
        return False
    if meta["term"] != "overlap":
        return False
    if int(len(meta["positions"])) <= 3:
        return False
    if int((expected != _get_bg(inp)).sum()) <= int((inp != _get_bg(inp)).sum()):
        return False

    if mode == "corner_stop":
        return int(meta["bounces"]) == 1 and int(meta["placed_after_bounce"]) == 0 and int(len(meta["positions"])) >= 5

    if np.array_equal(expected, counterfactual):
        return False
    if meta["first_post_bounce_color"] is None or no_swap_meta["first_post_bounce_color"] is None:
        return False
    if int(meta["first_post_bounce_color"]) == int(no_swap_meta["first_post_bounce_color"]):
        return False

    if mode == "bounce_chain":
        return int(meta["bounces"]) >= 2 and int(meta["placed_after_bounce"]) >= 2 and 5 <= int(len(meta["positions"])) <= 9
    if mode == "long_chain":
        return int(meta["bounces"]) >= 3 and int(meta["placed_after_bounce"]) >= 4 and 7 <= int(len(meta["positions"])) <= 12
    raise AssertionError(f"unknown generation mode: {mode}")


def _gen_attempt(mode: str) -> np.ndarray | None:
    if mode == "corner_stop":
        built = _sample_corner_stop_case()
    else:
        built = _sample_side_bounce_case()
    if built is None:
        return None

    inp, expected, meta, no_swap_meta, counterfactual = built
    if not _validate_case(inp, expected, meta, no_swap_meta, counterfactual, mode):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_MODE
    mode = ("bounce_chain", "long_chain", "corner_stop")[_GEN_MODE % 3]
    _GEN_MODE += 1
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample for mode {mode}")
