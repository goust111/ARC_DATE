from __future__ import annotations

import numpy as np


def _find_sample_stencil(g: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(g.shape[0] - 3):
        for c in range(g.shape[1] - 3):
            if np.all(g[r : r + 3, c + 3] == 5) and np.all(g[r + 3, c : c + 4] == 5):
                candidates.append((r, c))

    assert len(candidates) == 1, "expected one grey L-stencil around the sample"
    return candidates[0]


def _find_matching_sample(g: np.ndarray, sample: np.ndarray, source: tuple[int, int]) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for r in range(g.shape[0] - sample.shape[0] + 1):
        for c in range(g.shape[1] - sample.shape[1] + 1):
            if (r, c) == source:
                continue
            if np.array_equal(g[r : r + sample.shape[0], c : c + sample.shape[1]], sample):
                matches.append((r, c))

    assert len(matches) == 1, "expected exactly one duplicate of the sample"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the duplicate of the 3x3 sample marked by a grey L-stencil and draw a grey frame around it.

    Input structure:
    - A 3x3 sample region is marked by grey pixels immediately to its right and below, forming an L-stencil.
    - The same 3x3 color pattern appears exactly once elsewhere in the grid.
    - The duplicate has a one-cell margin available on every side.

    Rule:
    - Read the 3x3 sample from the upper-left side of the grey L-stencil.
    - Locate the unique other exact occurrence of that 3x3 sample.
    - Paint a one-cell-thick grey rectangular frame around the duplicate sample, preserving its interior.

    Key insight:
    The grey L is not the target frame itself; it identifies the small sample whose matching copy should be framed.
    """
    g = np.asarray(grid, dtype=int)
    source = _find_sample_stencil(g)
    sr, sc = source
    sample = g[sr : sr + 3, sc : sc + 3]
    tr, tc = _find_matching_sample(g, sample, source)

    assert tr > 0 and tc > 0, "duplicate sample must have top and left frame margins"
    assert tr + 3 < g.shape[0] and tc + 3 < g.shape[1], "duplicate sample must have bottom and right frame margins"

    out = g.copy()
    out[tr - 1 : tr + 4, tc - 1] = 5
    out[tr - 1 : tr + 4, tc + 3] = 5
    out[tr - 1, tc - 1 : tc + 4] = 5
    out[tr + 3, tc - 1 : tc + 4] = 5
    return out

# --------------- generator (o5x) ---------------

# Canvas geometry, colors, and sample content.
_GEN_GRID_SHAPE, _GEN_SOURCE_POS = (15, 15), (0, 0)
_GEN_BACKGROUND, _GEN_FRAME_COLOR = 0, 5
_GEN_SAMPLE_SIZE, _GEN_STENCIL_SIZE, _GEN_FRAME_PAD = 3, 4, 1
_GEN_SAMPLE_CELLS = _GEN_SAMPLE_SIZE * _GEN_SAMPLE_SIZE
_GEN_SAMPLE_NONZERO_RANGE = (4, 6)
_GEN_SAMPLE_DISTINCT_MIN_MAX = (3, 4)
_GEN_SAMPLE_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

# Duplicate top-left locations, excluding overlap with the source L-stencil frame.
_GEN_DUPLICATE_MIN_RC, _GEN_DUPLICATE_STOP_RC = 2, 11
_GEN_DUPLICATE_STENCIL_CLEARANCE = 5
_GEN_DUPLICATE_POSITIONS = tuple(
    (r, c)
    for r in range(_GEN_DUPLICATE_MIN_RC, _GEN_DUPLICATE_STOP_RC)
    for c in range(_GEN_DUPLICATE_MIN_RC, _GEN_DUPLICATE_STOP_RC)
    if r >= _GEN_DUPLICATE_STENCIL_CLEARANCE or c >= _GEN_DUPLICATE_STENCIL_CLEARANCE
)

# Noise density, repair palettes, and acceptance bounds.
_GEN_NOISE_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_GEN_REPAIR_VALUES = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_GEN_NOISE_NZ_MIN, _GEN_NOISE_NZ_MAX_AFTER_REPAIR = 74, 82
_GEN_NOISE_NZ_MAX_BY_SAMPLE_NZ = {4: 81, 5: 79}
_GEN_BG_FRAC_MIN_MAX = (0.57, 0.63)

# Retry budgets.
_GEN_L_REPAIR_ATTEMPTS, _GEN_MATCH_REPAIR_ATTEMPTS, _GEN_GENERATE_RETRIES = 60, 120, 300


def _gen_write_frame(g: np.ndarray, tr: int, tc: int, value: int | bool) -> None:
    r0 = tr - _GEN_FRAME_PAD
    r1 = tr + _GEN_SAMPLE_SIZE + _GEN_FRAME_PAD
    c0 = tc - _GEN_FRAME_PAD
    c1 = tc + _GEN_SAMPLE_SIZE + _GEN_FRAME_PAD
    g[r0:r1, c0] = value
    g[r0:r1, c1 - 1] = value
    g[r0, c0:c1] = value
    g[r1 - 1, c0:c1] = value


def _gen_paint_frame(g: np.ndarray, tr: int, tc: int) -> None:
    _gen_write_frame(g, tr, tc, _GEN_FRAME_COLOR)


def _gen_frame_mask(tr: int, tc: int) -> np.ndarray:
    mask = np.zeros(_GEN_GRID_SHAPE, dtype=bool)
    _gen_write_frame(mask, tr, tc, True)
    return mask


def _gen_make_sample(rng: np.random.Generator) -> np.ndarray:
    nz_min, nz_stop = _GEN_SAMPLE_NONZERO_RANGE
    distinct_min, distinct_max = _GEN_SAMPLE_DISTINCT_MIN_MAX
    nonzero_count = int(rng.integers(nz_min, nz_stop))
    distinct_count = int(rng.integers(distinct_min, min(distinct_max, nonzero_count) + 1))
    colors = rng.choice(_GEN_SAMPLE_COLORS, size=distinct_count, replace=False)
    extra = rng.choice(colors, size=nonzero_count - distinct_count, replace=True)
    values = np.concatenate([colors, extra])
    rng.shuffle(values)

    flat = np.zeros(_GEN_SAMPLE_CELLS, dtype=int)
    positions = rng.choice(_GEN_SAMPLE_CELLS, size=nonzero_count, replace=False)
    flat[positions] = values
    return flat.reshape((_GEN_SAMPLE_SIZE, _GEN_SAMPLE_SIZE))


def _gen_pick_duplicate_position(rng: np.random.Generator) -> tuple[int, int]:
    return _GEN_DUPLICATE_POSITIONS[int(rng.integers(len(_GEN_DUPLICATE_POSITIONS)))]


def _gen_l_stencils(g: np.ndarray) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for r in range(g.shape[0] - _GEN_SAMPLE_SIZE):
        for c in range(g.shape[1] - _GEN_SAMPLE_SIZE):
            right_col = g[r : r + _GEN_SAMPLE_SIZE, c + _GEN_SAMPLE_SIZE]
            bottom_row = g[r + _GEN_SAMPLE_SIZE, c : c + _GEN_STENCIL_SIZE]
            if np.all(right_col == _GEN_FRAME_COLOR) and np.all(bottom_row == _GEN_FRAME_COLOR):
                found.append((r, c))
    return found


def _gen_sample_matches(g: np.ndarray, sample: np.ndarray) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    sample_h, sample_w = sample.shape
    for r in range(g.shape[0] - sample_h + 1):
        for c in range(g.shape[1] - sample_w + 1):
            if np.array_equal(g[r : r + sample_h, c : c + sample_w], sample):
                found.append((r, c))
    return found


def _gen_window_equals_after(
    g: np.ndarray, sample: np.ndarray, wr: int, wc: int, rr: int, cc: int, value: int
) -> bool:
    sample_h, sample_w = sample.shape
    for dr in range(sample_h):
        for dc in range(sample_w):
            got = value if (wr + dr == rr and wc + dc == cc) else g[wr + dr, wc + dc]
            if got != sample[dr, dc]:
                return False
    return True


def _gen_safe_values_for_cell(
    g: np.ndarray,
    sample: np.ndarray,
    rr: int,
    cc: int,
    expected_matches: set[tuple[int, int]],
) -> list[int]:
    safe: list[int] = []
    sample_h, sample_w = sample.shape
    for value in _GEN_REPAIR_VALUES:
        ok = True
        for wr in range(max(0, rr - sample_h + 1), min(rr, g.shape[0] - sample_h) + 1):
            for wc in range(max(0, cc - sample_w + 1), min(cc, g.shape[1] - sample_w) + 1):
                if (wr, wc) in expected_matches:
                    continue
                if _gen_window_equals_after(g, sample, wr, wc, rr, cc, int(value)):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            safe.append(int(value))
    return safe


def _gen_choose_repair_value(rng: np.random.Generator, values: list[int], preserve_nonzero: bool) -> int:
    preferred = [v for v in values if (v != 0) == preserve_nonzero]
    if not preferred:
        preferred = values
    return int(preferred[int(rng.integers(len(preferred)))])


def _gen_repair_l_stencils(
    g: np.ndarray,
    sample: np.ndarray,
    fixed: np.ndarray,
    expected_matches: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    for _ in range(_GEN_L_REPAIR_ATTEMPTS):
        bad = [pos for pos in _gen_l_stencils(g) if pos != _GEN_SOURCE_POS]
        if not bad:
            return True

        r, c = bad[0]
        cells = [(r + i, c + _GEN_SAMPLE_SIZE) for i in range(_GEN_SAMPLE_SIZE)] + [
            (r + _GEN_SAMPLE_SIZE, c + j) for j in range(_GEN_STENCIL_SIZE)
        ]
        rng.shuffle(cells)
        for rr, cc in cells:
            if fixed[rr, cc]:
                continue
            values = _gen_safe_values_for_cell(g, sample, rr, cc, expected_matches)
            if values:
                g[rr, cc] = _gen_choose_repair_value(rng, values, preserve_nonzero=True)
                break
        else:
            return False
    return False


def _gen_repair_sample_matches(
    g: np.ndarray,
    sample: np.ndarray,
    fixed: np.ndarray,
    expected_matches: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    sample_h, sample_w = sample.shape
    for _ in range(_GEN_MATCH_REPAIR_ATTEMPTS):
        bad = [pos for pos in _gen_sample_matches(g, sample) if pos not in expected_matches]
        if not bad:
            return True

        r, c = bad[0]
        cells = [
            (r + dr, c + dc)
            for dr in range(sample_h)
            for dc in range(sample_w)
            if not fixed[r + dr, c + dc]
        ]
        if not cells:
            return False

        nonzero_cells = [(rr, cc) for rr, cc in cells if sample[rr - r, cc - c] != 0]
        zero_cells = [(rr, cc) for rr, cc in cells if sample[rr - r, cc - c] == 0]
        rng.shuffle(nonzero_cells)
        rng.shuffle(zero_cells)
        for rr, cc in nonzero_cells + zero_cells:
            values = _gen_safe_values_for_cell(g, sample, rr, cc, expected_matches)
            if values:
                g[rr, cc] = _gen_choose_repair_value(rng, values, preserve_nonzero=g[rr, cc] != 0)
                break
        else:
            return False
    return False


def _gen_build_candidate_input(
    rng: np.random.Generator,
    sample: np.ndarray,
    tr: int,
    tc: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = _gen_frame_mask(tr, tc)
    out = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    sr, sc = _GEN_SOURCE_POS
    out[sr : sr + _GEN_SAMPLE_SIZE, sc : sc + _GEN_SAMPLE_SIZE] = sample
    out[sr : sr + _GEN_SAMPLE_SIZE, sc + _GEN_SAMPLE_SIZE] = _GEN_FRAME_COLOR
    out[sr + _GEN_SAMPLE_SIZE, sc : sc + _GEN_STENCIL_SIZE] = _GEN_FRAME_COLOR
    out[tr : tr + _GEN_SAMPLE_SIZE, tc : tc + _GEN_SAMPLE_SIZE] = sample
    _gen_paint_frame(out, tr, tc)

    fixed = np.zeros(_GEN_GRID_SHAPE, dtype=bool)
    fixed[sr : sr + _GEN_STENCIL_SIZE, sc : sc + _GEN_STENCIL_SIZE] = True
    fixed[tr : tr + _GEN_SAMPLE_SIZE, tc : tc + _GEN_SAMPLE_SIZE] = True
    free = ~fixed
    free_coords = np.argwhere(free)
    max_noise = _GEN_NOISE_NZ_MAX_BY_SAMPLE_NZ[int(np.count_nonzero(sample))]
    noise_count = int(rng.integers(_GEN_NOISE_NZ_MIN, max_noise + 1))
    chosen = rng.choice(len(free_coords), size=noise_count, replace=False)
    values = rng.choice(_GEN_NOISE_COLORS, size=noise_count, replace=True)

    coords = free_coords[chosen]
    on_frame = frame[coords[:, 0], coords[:, 1]]
    frame_restore = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    frame_coords = coords[on_frame]
    noise_coords = coords[~on_frame]
    frame_restore[frame_coords[:, 0], frame_coords[:, 1]] = values[on_frame]
    out[noise_coords[:, 0], noise_coords[:, 1]] = values[~on_frame]

    inp = out.copy()
    inp[frame] = frame_restore[frame]
    return inp, fixed


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    sample = _gen_make_sample(rng)
    tr, tc = _gen_pick_duplicate_position(rng)
    inp, fixed = _gen_build_candidate_input(rng, sample, tr, tc)
    expected_matches = {_GEN_SOURCE_POS, (tr, tc)}

    if not _gen_repair_l_stencils(inp, sample, fixed, expected_matches, rng):
        return None
    if not _gen_repair_sample_matches(inp, sample, fixed, expected_matches, rng):
        return None
    noise_nz = int(np.count_nonzero(inp[~fixed]))
    bg_frac = float(np.mean(inp == _GEN_BACKGROUND))
    bg_min, bg_max = _GEN_BG_FRAC_MIN_MAX
    if not (_GEN_NOISE_NZ_MIN <= noise_nz <= _GEN_NOISE_NZ_MAX_AFTER_REPAIR and bg_min <= bg_frac <= bg_max):
        return None

    expected = inp.copy()
    _gen_paint_frame(expected, tr, tc)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e2092e0c input")
