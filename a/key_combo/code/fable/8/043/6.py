import random
from collections import defaultdict

import numpy as np


BLACK = 0
GEN_ATTEMPTS = 600


def _tile_fill(tile: np.ndarray, height: int, width: int, row_shift: int = 0, col_shift: int = 0) -> np.ndarray:
    rows = (np.arange(height)[:, None] - int(row_shift)) % tile.shape[0]
    cols = (np.arange(width)[None, :] - int(col_shift)) % tile.shape[1]
    return tile[rows, cols].astype(int, copy=True)


def _phase_map(grid: np.ndarray, row_period: int, col_period: int, hole: int = BLACK) -> dict[tuple[int, int], int] | None:
    phase_colors: dict[tuple[int, int], int] = {}
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            value = int(grid[r, c])
            if value == int(hole):
                continue
            key = (r % int(row_period), c % int(col_period))
            prev = phase_colors.get(key)
            if prev is not None and prev != value:
                return None
            phase_colors[key] = value
    return phase_colors


def _infer_period(grid: np.ndarray) -> tuple[int, int]:
    height, width = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            if _phase_map(grid, row_period, col_period) is not None:
                candidates.append((row_period * col_period, row_period, col_period))
    assert candidates, "visible pixels must define at least one period"
    best_area = min(area for area, _, _ in candidates)
    best = [(rp, cp) for area, rp, cp in candidates if area == best_area]
    assert len(best) == 1, "minimal period must be unique"
    return best[0]


def _shift_score(top: np.ndarray, bottom: np.ndarray, dr: int, dc: int) -> tuple[int, int] | None:
    height, width = top.shape
    r0 = max(0, -int(dr))
    r1 = min(height, height - int(dr))
    c0 = max(0, -int(dc))
    c1 = min(width, width - int(dc))
    if r0 >= r1 or c0 >= c1:
        return None

    compared = 0
    for r in range(r0, r1):
        br = r + int(dr)
        for c in range(c0, c1):
            bc = c + int(dc)
            top_value = int(top[r, c])
            bottom_value = int(bottom[br, bc])
            if top_value == BLACK or bottom_value == BLACK:
                continue
            if top_value != bottom_value:
                return None
            compared += 1
    if compared == 0:
        return None
    overlap_area = (r1 - r0) * (c1 - c0)
    return compared, overlap_area


def _infer_shift(top: np.ndarray, bottom: np.ndarray) -> tuple[int, int]:
    height, width = top.shape
    candidates: list[tuple[tuple[int, int], int, int]] = []
    for dr in range(-(height - 1), height):
        for dc in range(-(width - 1), width):
            score = _shift_score(top, bottom, dr, dc)
            if score is None:
                continue
            candidates.append((score, dr, dc))
    assert candidates, "expected at least one alignment candidate"
    best_score = max(score for score, _, _ in candidates)
    best = [(dr, dc) for score, dr, dc in candidates if score == best_score]
    assert len(best) == 1, "alignment shift must be unique"
    return best[0]


def _fill_by_period(grid: np.ndarray, row_period: int, col_period: int) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int, copy=True)
    phase_colors = _phase_map(out, row_period, col_period)
    assert phase_colors is not None, "partially repaired wallpaper must remain period-consistent"

    filled = 0
    height, width = out.shape
    for r in range(height):
        for c in range(width):
            if int(out[r, c]) != BLACK:
                continue
            key = (r % int(row_period), c % int(col_period))
            assert key in phase_colors, "every remaining hole must have a visible phase mate"
            out[r, c] = int(phase_colors[key])
            filled += 1
    return out, filled


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 1, "expected a vertical stack split by one separator row"
    split = g.shape[0] // 2
    top = g[:split, :]
    bottom = g[split + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0
    assert np.all(g[split, :] == g[split, 0]), "separator row must be uniform"

    row_period, col_period = _infer_period(top)
    dr, dc = _infer_shift(top, bottom)

    partial = np.array(top, dtype=int, copy=True)
    stage1 = 0
    height, width = partial.shape
    for r in range(height):
        for c in range(width):
            if int(partial[r, c]) != BLACK:
                continue
            br = r + int(dr)
            bc = c + int(dc)
            if 0 <= br < height and 0 <= bc < width and int(bottom[br, bc]) != BLACK:
                partial[r, c] = int(bottom[br, bc])
                stage1 += 1

    repaired, stage2 = _fill_by_period(partial, row_period, col_period)
    assert np.array_equal(repaired[top != BLACK], top[top != BLACK]), "non-black top cells must stay fixed"

    if return_meta:
        return repaired, {
            "period": (row_period, col_period),
            "shift": (dr, dc),
            "stage1": stage1,
            "stage2": stage2,
            "partial": partial,
        }
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the upper wallpaper window by first aligning the lower shifted copy, then using the wallpaper's own period.

    Input structure:
    - A single separator row splits the grid into two equal-size windows, one above and one below.
    - Both windows come from the same infinite periodic wallpaper but from different translated positions.
    - Black cells are missing pixels; black is not part of the wallpaper palette.

    Rule:
    - Infer the unique smallest row/column period consistent with the visible pixels in the upper window.
    - Find the unique translation that makes every overlapping pair of non-black cells agree between the two windows.
    - Output the repaired upper window: each black cell is filled first from the aligned lower window, and any hole still black after that is filled from the upper window's periodic phase.

    Key insight:
    - The lower window is not a direct answer mask; it is a shifted copy of the same wallpaper, so the shift must be recovered before it can reveal missing phases that the upper window alone does not show.
    """
    return _solve(grid, return_meta=False)


def _has_exact_period(tile: np.ndarray, row_period: int, col_period: int) -> bool:
    for rp in range(1, row_period + 1):
        for cp in range(1, col_period + 1):
            if rp == row_period and cp == col_period:
                continue
            ok = True
            for r in range(tile.shape[0]):
                for c in range(tile.shape[1]):
                    if int(tile[r, c]) != int(tile[r % rp, c % cp]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return False
    return True


def _gen_tile(row_period: int, col_period: int, rng: random.Random) -> np.ndarray:
    palette_size = rng.randint(3, min(6, row_period * col_period, 9))
    palette = rng.sample(list(range(1, 10)), palette_size)
    for _ in range(300):
        tile = np.array([rng.choice(palette) for _ in range(row_period * col_period)], dtype=int).reshape(row_period, col_period)
        if len(set(map(int, tile.ravel()))) < 3:
            continue
        if not _has_exact_period(tile, row_period, col_period):
            continue
        symmetric = False
        for dr in range(row_period):
            for dc in range(col_period):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(np.roll(tile, -dr, axis=0), -dc, axis=1)
                if int(np.count_nonzero(tile == shifted)) >= tile.size - 1:
                    symmetric = True
                    break
            if symmetric:
                break
        if symmetric:
            continue
        return tile
    raise RuntimeError("failed to generate a tile with an exact period")


def _non_self_inverse_steps(modulus: int) -> list[int]:
    return [step for step in range(1, int(modulus)) if (2 * step) % int(modulus) != 0]


def _choose_shift(row_period: int, col_period: int, height: int, width: int, rng: random.Random) -> tuple[int, int]:
    row_steps = [0] + [step for step in _non_self_inverse_steps(row_period) if step < height]
    col_steps = [0] + [step for step in _non_self_inverse_steps(col_period) if step < width]

    options: list[tuple[int, int]] = []
    if len(row_steps) > 1:
        options.extend((step, 0) for step in row_steps if step != 0)
    if len(col_steps) > 1:
        options.extend((0, step) for step in col_steps if step != 0)
    if len(row_steps) > 1 and len(col_steps) > 1:
        options.extend((r, c) for r in row_steps if r != 0 for c in col_steps if c != 0)
    if not options:
        raise RuntimeError("no usable non-zero shift for the chosen periods")
    return rng.choice(options)


def _phase_coords(height: int, width: int, row_period: int, col_period: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    out: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for r in range(height):
        for c in range(width):
            out[(r % row_period, c % col_period)].append((r, c))
    return out


def _pick_special_phase(
    height: int,
    width: int,
    row_period: int,
    col_period: int,
    dr: int,
    dc: int,
    prefer_out_of_overlap: bool,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], bool]:
    by_phase = _phase_coords(height, width, row_period, col_period)
    branch_candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], bool]] = []
    fallback_candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], bool]] = []
    for phase, coords in by_phase.items():
        if len(coords) < 2:
            continue
        overlap = [(r, c) for r, c in coords if r + dr < height and c + dc < width]
        if len(overlap) < 1:
            continue
        outside = [(r, c) for r, c in coords if (r, c) not in overlap]
        for recover in overlap:
            if prefer_out_of_overlap and outside:
                remain = rng.choice(outside)
                branch_candidates.append((phase, recover, remain, True))
            nonrecover = [(r, c) for r, c in overlap if (r, c) != recover]
            if nonrecover:
                remain = rng.choice(nonrecover)
                fallback_candidates.append((phase, recover, remain, False))
    pool = branch_candidates if branch_candidates else fallback_candidates
    if not pool:
        raise RuntimeError("failed to find a phase that forces both repair stages")
    return rng.choice(pool)


def _add_top_noise(top: np.ndarray, protected_phase: tuple[int, int], row_period: int, col_period: int, rng: random.Random) -> None:
    visible_counts: dict[tuple[int, int], int] = defaultdict(int)
    height, width = top.shape
    for r in range(height):
        for c in range(width):
            if int(top[r, c]) != BLACK:
                visible_counts[(r % row_period, c % col_period)] += 1

    extra_holes = rng.randint(1, max(2, min(5, (height * width) // 12)))
    candidates = [(r, c) for r in range(height) for c in range(width) if int(top[r, c]) != BLACK]
    rng.shuffle(candidates)
    added = 0
    for r, c in candidates:
        phase = (r % row_period, c % col_period)
        if phase == protected_phase:
            continue
        if visible_counts[phase] <= 1:
            continue
        top[r, c] = BLACK
        visible_counts[phase] -= 1
        added += 1
        if added >= extra_holes:
            break


def _add_bottom_noise(bottom: np.ndarray, protected: set[tuple[int, int]], rng: random.Random) -> None:
    height, width = bottom.shape
    extra_holes = rng.randint(1, max(2, min(6, (height * width) // 10)))
    cells = [(r, c) for r in range(height) for c in range(width) if int(bottom[r, c]) != BLACK and (r, c) not in protected]
    rng.shuffle(cells)
    for r, c in cells[:extra_holes]:
        bottom[r, c] = BLACK


def _top_alone_is_insufficient(top: np.ndarray, row_period: int, col_period: int) -> bool:
    phase_colors = _phase_map(top, row_period, col_period)
    assert phase_colors is not None
    for r in range(top.shape[0]):
        for c in range(top.shape[1]):
            if int(top[r, c]) == BLACK and (r % row_period, c % col_period) not in phase_colors:
                return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_period = rng.randint(2, 5)
    col_period = rng.randint(2, 5)
    height = rng.randint(max(7, row_period * 2), min(14, row_period * 3 + 2))
    width = rng.randint(max(7, col_period * 2), min(14, col_period * 3 + 2))

    try:
        dr, dc = _choose_shift(row_period, col_period, height, width, rng)
    except RuntimeError:
        return None

    tile = _gen_tile(row_period, col_period, rng)
    top_clean = _tile_fill(tile, height, width)
    bottom_clean = _tile_fill(tile, height, width, row_shift=dr, col_shift=dc)

    top = np.array(top_clean, copy=True)
    bottom = np.array(bottom_clean, copy=True)

    prefer_out = rng.random() < 0.5
    try:
        phase, recover_cell, remain_cell, uses_out_of_overlap = _pick_special_phase(
            height, width, row_period, col_period, dr, dc, prefer_out, rng
        )
    except RuntimeError:
        return None

    phase_cells = _phase_coords(height, width, row_period, col_period)[phase]
    for r, c in phase_cells:
        top[r, c] = BLACK

    protected_bottom = {(recover_cell[0] + dr, recover_cell[1] + dc)}
    if not uses_out_of_overlap:
        br = remain_cell[0] + dr
        bc = remain_cell[1] + dc
        bottom[br, bc] = BLACK
        protected_bottom.add((br, bc))

    _add_top_noise(top, phase, row_period, col_period, rng)
    _add_bottom_noise(bottom, protected_bottom, rng)

    if int(bottom[recover_cell[0] + dr, recover_cell[1] + dc]) == BLACK:
        return None
    if not _top_alone_is_insufficient(top, row_period, col_period):
        return None

    period = _infer_period(top)
    if period != (row_period, col_period):
        return None

    shift = _infer_shift(top, bottom)
    if shift != (dr, dc):
        return None

    sep_choices = [color for color in range(1, 10) if color not in set(map(int, tile.ravel()))]
    if not sep_choices:
        return None
    separator_color = rng.choice(sep_choices)
    grid = np.vstack(
        [
            top,
            np.full((1, width), separator_color, dtype=int),
            bottom,
        ]
    )

    try:
        output, meta = _solve(grid, return_meta=True)
    except Exception:
        return None
    if not np.array_equal(output, top_clean):
        return None
    if meta["period"] != (row_period, col_period):
        return None
    if meta["shift"] != (dr, dc):
        return None
    if int(meta["stage1"]) <= 0 or int(meta["stage2"]) <= 0:
        return None
    if uses_out_of_overlap:
        rr, cc = remain_cell
        if rr + dr < height and cc + dc < width:
            return None
    else:
        if int(bottom[remain_cell[0] + dr, remain_cell[1] + dc]) != BLACK:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
