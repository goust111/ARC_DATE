from __future__ import annotations

import hashlib
import random

import numpy as np


BLACK = 0
GREY = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BRANCHES = ("right", "left", "inside")
GENERATION_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_rails(g: np.ndarray) -> tuple[int, int, int, int]:
    grey = g == GREY
    cols = [int(c) for c in np.flatnonzero(grey.any(axis=0))]
    assert len(cols) == 2, "expected exactly two grey rail columns"

    spans = []
    for c in cols:
        rows = np.flatnonzero(grey[:, c])
        assert rows.size >= 2, "each rail must be a vertical short line"
        assert int(rows[-1] - rows[0] + 1) == int(rows.size), "rail pixels must be contiguous"
        spans.append((int(rows[0]), int(rows[-1])))

    assert spans[0] == spans[1], "the two rails must cover the same row range"
    c0, c1 = sorted(cols)
    return spans[0][0], spans[0][1], c0, c1


def _phase_to_unit_col(x: int, period_w: int) -> int:
    x %= 2 * period_w
    return x if x < period_w else 2 * period_w - 1 - x


def _infer_glide_candidates(strip: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    h, w = strip.shape
    known = strip != GREY
    candidates: list[tuple[int, int, np.ndarray]] = []

    for period_w in range(2, w // 2 + 1):
        for shift in range(2 * period_w):
            unit = np.full((h, period_w), -1, dtype=int)
            ok = True
            for c in range(w):
                unit_c = _phase_to_unit_col(c + shift, period_w)
                for r in np.flatnonzero(known[:, c]):
                    v = int(strip[int(r), c])
                    old = int(unit[int(r), unit_c])
                    if old == -1:
                        unit[int(r), unit_c] = v
                    elif old != v:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or np.any(unit < 0):
                continue
            if np.array_equal(unit, np.fliplr(unit)):
                continue
            candidates.append((period_w, shift, unit))

    assert candidates, "expected a uniquely recoverable glide-reflection period"
    return candidates


def _render_candidate(
    rail_r0: int,
    rail_r1: int,
    rail_c0: int,
    rail_c1: int,
    pattern_r0: int,
    pattern_c0: int,
    period_w: int,
    shift: int,
    unit: np.ndarray,
) -> np.ndarray:
    out_h = rail_r1 - rail_r0 + 1
    out_w = rail_c1 - rail_c0 - 1
    assert out_h > 0 and out_w > 0, "rails must enclose a non-empty rectangle"

    out = np.full((out_h, out_w), BLACK, dtype=int)
    for out_r, grid_r in enumerate(range(rail_r0, rail_r1 + 1)):
        unit_r = grid_r - pattern_r0
        if not 0 <= unit_r < unit.shape[0]:
            continue
        for out_c, grid_c in enumerate(range(rail_c0 + 1, rail_c1)):
            unit_c = _phase_to_unit_col(grid_c - pattern_c0 + shift, period_w)
            out[out_r, out_c] = int(unit[unit_r, unit_c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a horizontally glide-reflected pattern and crop the rectangle between two rails.

    Input structure:
    - The grid has a black background, one adjacent band of non-grey colored pattern rows, and two vertical grey rail lines.
    - Along the pattern band, a hidden unit repeats left-to-right as unit, mirrored unit, unit, mirrored unit, possibly beginning at any phase.
    - The grey rails share the same row range and may lie inside the painted segment or in blank space to its left or right.

    Rule:
    - Detect the two grey rail columns and the painted non-grey pattern band.
    - Infer the smallest visible glide period by aligning every observed column to a unit column under alternating horizontal reflection.
    - Conceptually extend that periodic glide pattern left and right, then output only the rows and columns strictly between the rails.
    - Rows of the rail rectangle outside the pattern band remain black, and the rail columns themselves are excluded.

    Key insight:
    - The repeated motif is not a plain translation: every step flips left-right, so the crop can be correct beyond the painted area only after recovering the glide phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    rail_r0, rail_r1, rail_c0, rail_c1 = _find_rails(g)
    pattern_box = _bbox((g != BLACK) & (g != GREY))
    assert pattern_box is not None, "expected a colored pattern"
    pattern_r0, pattern_c0, pattern_r1, pattern_c1 = pattern_box

    strip = g[pattern_r0 : pattern_r1 + 1, pattern_c0 : pattern_c1 + 1]
    candidates = _infer_glide_candidates(strip)

    outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for period_w, shift, unit in candidates:
        out = _render_candidate(
            rail_r0,
            rail_r1,
            rail_c0,
            rail_c1,
            pattern_r0,
            pattern_c0,
            period_w,
            shift,
            unit,
        )
        outputs[(out.shape, out.tobytes())] = out

    assert len(outputs) == 1, "the observed glide evidence must determine one output"
    return next(iter(outputs.values())).copy()


def _render_strip(unit: np.ndarray, width: int, shift: int) -> np.ndarray:
    h, period_w = unit.shape
    out = np.full((h, width), BLACK, dtype=int)
    for c in range(width):
        out[:, c] = unit[:, _phase_to_unit_col(c + shift, period_w)]
    return out


def _candidate_min_period(unit: np.ndarray) -> int:
    period_w = unit.shape[1]
    probe = _render_strip(unit, 6 * period_w, 0)
    return min(p for p, _s, _u in _infer_glide_candidates(probe))


def _random_unit(height: int, period_w: int) -> np.ndarray | None:
    for _ in range(200):
        colors = random.sample(PATTERN_COLORS, random.randint(2, min(4, len(PATTERN_COLORS))))
        unit = np.full((height, period_w), BLACK, dtype=int)
        density = random.uniform(0.62, 0.86)
        for r in range(height):
            for c in range(period_w):
                if random.random() < density:
                    unit[r, c] = random.choice(colors)

        for r in range(height):
            if not np.any(unit[r] != BLACK):
                unit[r, random.randrange(period_w)] = random.choice(colors)
        for c in range(period_w):
            if not np.any(unit[:, c] != BLACK):
                unit[random.randrange(height), c] = random.choice(colors)

        if np.array_equal(unit, np.fliplr(unit)):
            continue
        if len(set(map(int, unit[unit != BLACK]))) < 2:
            continue
        try:
            if _candidate_min_period(unit) != period_w:
                continue
        except AssertionError:
            continue
        return unit
    return None


def _layout_columns(
    branch: str, paint_w: int, crop_w: int
) -> tuple[int, int, int, int] | None:
    options: list[tuple[int, int, int, int]] = []
    for grid_w in range(max(paint_w, crop_w + 2), 31):
        for paint_c0 in range(0, grid_w - paint_w + 1):
            paint_c1 = paint_c0 + paint_w - 1
            for rail_c0 in range(0, grid_w - crop_w - 1):
                rail_c1 = rail_c0 + crop_w + 1
                if rail_c0 in {paint_c0, paint_c1} or rail_c1 in {paint_c0, paint_c1}:
                    continue

                left_inside = paint_c0 < rail_c0 < paint_c1
                right_inside = paint_c0 < rail_c1 < paint_c1
                if branch == "right":
                    if rail_c0 > paint_c1 and 1 <= rail_c0 - paint_c1 <= 4:
                        options.append((grid_w, paint_c0, rail_c0, rail_c1))
                elif branch == "left":
                    if rail_c1 < paint_c0 and 1 <= paint_c0 - rail_c1 <= 4:
                        options.append((grid_w, paint_c0, rail_c0, rail_c1))
                else:
                    both_inside = left_inside and right_inside
                    straddle_left = rail_c0 < paint_c0 and right_inside
                    straddle_right = left_inside and rail_c1 > paint_c1
                    if both_inside or straddle_left or straddle_right:
                        options.append((grid_w, paint_c0, rail_c0, rail_c1))

    if not options:
        return None
    return random.choice(options)


def _expected_from_spec(
    unit: np.ndarray,
    shift: int,
    pattern_r0: int,
    pattern_c0: int,
    rail_r0: int,
    rail_r1: int,
    rail_c0: int,
    rail_c1: int,
) -> np.ndarray:
    return _render_candidate(
        rail_r0,
        rail_r1,
        rail_c0,
        rail_c1,
        pattern_r0,
        pattern_c0,
        unit.shape[1],
        shift,
        unit,
    )


def _naive_visible_crop(grid: np.ndarray, rail_r0: int, rail_r1: int, rail_c0: int, rail_c1: int) -> np.ndarray:
    return grid[rail_r0 : rail_r1 + 1, rail_c0 + 1 : rail_c1].copy()


def _gen_attempt(branch: str) -> np.ndarray | None:
    period_w = random.randint(2, 5)
    pattern_h = random.randint(2, 5)
    unit = _random_unit(pattern_h, period_w)
    if unit is None:
        return None

    min_paint_w = 3 * period_w + 2
    max_paint_w = min(20, 5 * period_w + 2)
    if min_paint_w > max_paint_w:
        return None
    paint_w = random.randint(min_paint_w, max_paint_w)
    crop_w = random.randint(3, 9)
    layout = _layout_columns(branch, paint_w, crop_w)
    if layout is None:
        return None
    grid_w, paint_c0, rail_c0, rail_c1 = layout
    paint_c1 = paint_c0 + paint_w - 1

    top_margin = random.randint(0, 3)
    bottom_margin = random.randint(0, 3)
    rail_h = pattern_h + top_margin + bottom_margin
    if rail_h > 16:
        return None
    grid_h = random.randint(rail_h, min(30, rail_h + 8))
    rail_r0 = random.randint(0, grid_h - rail_h)
    rail_r1 = rail_r0 + rail_h - 1
    pattern_r0 = rail_r0 + top_margin
    pattern_r1 = pattern_r0 + pattern_h - 1

    shift = random.randrange(2 * period_w)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[pattern_r0 : pattern_r1 + 1, paint_c0 : paint_c1 + 1] = _render_strip(unit, paint_w, shift)

    grid[rail_r0 : rail_r1 + 1, rail_c0] = GREY
    grid[rail_r0 : rail_r1 + 1, rail_c1] = GREY

    expected = _expected_from_spec(unit, shift, pattern_r0, paint_c0, rail_r0, rail_r1, rail_c0, rail_c1)
    if not np.any(expected != BLACK):
        return None

    naive = _naive_visible_crop(grid, rail_r0, rail_r1, rail_c0, rail_c1)
    if branch in {"left", "right"} and np.array_equal(naive, expected):
        return None
    if branch == "inside":
        outside_cols = [c for c in range(rail_c0 + 1, rail_c1) if c < paint_c0 or c > paint_c1]
        if outside_cols and np.array_equal(naive, expected):
            return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    preferred = BRANCHES[_GEN_CALL_COUNT % len(BRANCHES)]
    _GEN_CALL_COUNT += 1
    branches = [preferred] + random.sample(list(BRANCHES), len(BRANCHES))
    for branch in branches:
        for _ in range(GENERATION_ATTEMPTS):
            grid = _gen_attempt(branch)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid glide-period rail crop input")


if __name__ == "__main__":
    sigs = set()
    for _ in range(10):
        inp = generate_input()
        out = main(inp)
        sigs.add(hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest())
        assert inp.ndim == 2 and out.ndim == 2
        assert max(inp.shape) <= 30 and max(out.shape) <= 30
    print("ok", len(sigs))
