# Combo variant of 7b80bb43 (index 1)
# combo_spec: D1/0+D3/0
# Name: 对称修复保留提示

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the fused variant:
# - Concrete behavior: in the original task, a row like `###...###` with a nearby
#   diagonal 2-pixel staircase became `#########` because the staircase was deleted
#   and the 3-cell gap was filled. Here the staircase survives. The repaired output
#   is instead `###/..../###` -> `#########` plus the same staircase pixels still
#   present, but the fill is allowed only when the same 3-cell gap appears at the
#   mirrored location across the network's vertical symmetry axis.
# - Complexity calibration: a single global vertical axis keeps the rule visible,
#   while mixing horizontal and vertical repairs prevents the task from collapsing
#   into a trivial one-row mirror puzzle. Each sample contains at least one paired
#   horizontal repair and one paired vertical repair so the symmetry control
#   variable has repeated support.
# - Implicit assumptions changed from the source solver:
#   1. Diagonal-only pixels are no longer disposable; they must be preserved.
#   2. Local "hint nearby => fill" is no longer valid; mirrored partner gaps are
#      now required.
#   3. The solver therefore needs a stable symmetry axis. The generator enforces an
#      odd-width grid and a single integer vertical axis.
#   4. The generator avoids self-mirrored gaps on the axis and blocks gap placement
#      through junction cells, so each repaired gap has an unambiguous mirrored mate.
# - Placement legality and cleanliness: the generator builds a clean symmetric
#   line network first, then enumerates only gap sites whose staircase hints fit
#   in empty background, stay in-bounds, and remain 4-disconnected from the line
#   before repair. This keeps the final images readable.


GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_INSET = 2

GRID_H_RANGE = (18, 28)
GRID_W_CHOICES = [w for w in range(19, 30) if w % 2 == 1]
N_COLORS = 10
ROW_SPACING = 3
GEN_RETRIES = 500


def _deg4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    deg = 0
    if r > 0 and bool(mask[r - 1, c]):
        deg += 1
    if r + 1 < h and bool(mask[r + 1, c]):
        deg += 1
    if c > 0 and bool(mask[r, c - 1]):
        deg += 1
    if c + 1 < w and bool(mask[r, c + 1]):
        deg += 1
    return deg


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr in (-1, 1):
        for dc in (-1, 1):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                return True
    return False


def _diag_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, r, c) == 0 and _has_diag(mask, r, c):
                diag[r, c] = True
    return diag


def _best_vertical_axis(line_mask: np.ndarray) -> int:
    h, w = line_mask.shape
    pts = np.argwhere(line_mask)
    assert pts.size > 0, "line network must be non-empty"
    best_axis = 0
    best_score = -1
    for axis in range(w):
        score = 0
        for r, c in pts:
            cc = 2 * axis - int(c)
            if 0 <= cc < w and bool(line_mask[int(r), cc]):
                score += 1
        if score > best_score:
            best_score = int(score)
            best_axis = int(axis)
    return best_axis


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    vals = [int(x) for x in indices]
    out: list[tuple[int, int]] = []
    start = vals[0]
    prev = vals[0]
    for cur in vals[1:]:
        if cur == prev + 1:
            prev = cur
        else:
            out.append((int(start), int(prev)))
            start = prev = cur
    out.append((int(start), int(prev)))
    return out


def _is_row_gap(line_mask: np.ndarray, row: int, c0: int, c1: int) -> bool:
    h, w = line_mask.shape
    if not (0 <= row < h and 0 < c0 <= c1 < w - 1):
        return False
    if bool(line_mask[row, c0 : c1 + 1].any()):
        return False
    return bool(line_mask[row, c0 - 1]) and bool(line_mask[row, c1 + 1])


def _is_col_gap(line_mask: np.ndarray, col: int, r0: int, r1: int) -> bool:
    h, w = line_mask.shape
    if not (0 <= col < w and 0 < r0 <= r1 < h - 1):
        return False
    if bool(line_mask[r0 : r1 + 1, col].any()):
        return False
    return bool(line_mask[r0 - 1, col]) and bool(line_mask[r1 + 1, col])


def _hint_near_h(diag: np.ndarray, row: int, c0: int, c1: int) -> bool:
    h, w = diag.shape
    rr0 = max(0, row - 1)
    rr1 = min(h - 1, row + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w - 1, c1 + 1)
    return bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _hint_near_v(diag: np.ndarray, r0: int, r1: int, col: int) -> bool:
    h, w = diag.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc0 = max(0, col - 1)
    cc1 = min(w - 1, col + 1)
    return bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _draw_h(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    grid[row, c0 : c1 + 1] = int(color)


def _draw_v(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    grid[r0 : r1 + 1, col] = int(color)


def _pick_spaced(rng: random.Random, values: list[int], k: int, min_gap: int) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - p) >= min_gap for p in picked):
            picked.append(int(v))
            if len(picked) == k:
                return sorted(picked)
    return []


def _h_chain_coords(row: int, gap_start: int, anchor: str, dr: int) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + dr * i, gap_start - 1 + i) for i in range(1, CHAIN_LENGTH + 1)]
    return [(row + dr * i, gap_start + GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]
    return [(gap_start + GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]


def _cells_are_bg(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(bg):
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int) -> bool:
    h, w = grid.shape
    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in coord_set:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg):
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg)):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _blocked_h_cols(clean: np.ndarray, row: int, fg: int) -> set[int]:
    h, w = clean.shape
    blocked: set[int] = set()
    for c in range(w):
        if int(clean[row, c]) != int(fg):
            continue
        if (row > 0 and int(clean[row - 1, c]) == int(fg)) or (row + 1 < h and int(clean[row + 1, c]) == int(fg)):
            blocked.add(int(c))
    return blocked


def _blocked_v_rows(clean: np.ndarray, col: int, fg: int) -> set[int]:
    h, w = clean.shape
    blocked: set[int] = set()
    for r in range(h):
        if int(clean[r, col]) != int(fg):
            continue
        if (col > 0 and int(clean[r, col - 1]) == int(fg)) or (col + 1 < w and int(clean[r, col + 1]) == int(fg)):
            blocked.add(int(r))
    return blocked


def _horizontal_pair_options(clean: np.ndarray, axis: int, fg: int, bg: int) -> list[dict]:
    h, _w = clean.shape
    options: list[dict] = []
    for row in range(h):
        blocked = _blocked_h_cols(clean, row, fg)
        for c0, c1 in _runs(np.where(clean[row, :] == int(fg))[0]):
            if not (c0 < axis < c1):
                continue
            if c1 - c0 + 1 < 2 * GAP_SIZE + 5:
                continue
            for gap_start in range(c0 + GAP_INSET, axis - GAP_SIZE):
                mirror_start = 2 * axis - (gap_start + GAP_SIZE - 1)
                if mirror_start <= axis or mirror_start + GAP_SIZE - 1 > c1:
                    continue
                left_gap = range(gap_start, gap_start + GAP_SIZE)
                right_gap = range(mirror_start, mirror_start + GAP_SIZE)
                if any(col in blocked for col in left_gap) or any(col in blocked for col in right_gap):
                    continue
                temp = clean.copy()
                temp[row, gap_start : gap_start + GAP_SIZE] = int(bg)
                temp[row, mirror_start : mirror_start + GAP_SIZE] = int(bg)
                for dr in (-1, 1):
                    left_chain = _h_chain_coords(row, gap_start, "left", dr)
                    right_chain = _h_chain_coords(row, mirror_start, "right", dr)
                    all_chain = left_chain + right_chain
                    if not _cells_are_bg(temp, all_chain, bg):
                        continue
                    if not _validate_chain(temp, left_chain, fg):
                        continue
                    if not _validate_chain(temp, right_chain, fg):
                        continue
                    options.append(
                        {
                            "kind": "h",
                            "row": int(row),
                            "left_start": int(gap_start),
                            "right_start": int(mirror_start),
                            "left_chain": left_chain,
                            "right_chain": right_chain,
                        }
                    )
    return options


def _vertical_pair_options(clean: np.ndarray, axis: int, fg: int, bg: int) -> list[dict]:
    h, w = clean.shape
    options: list[dict] = []
    for col in range(axis):
        mirror_col = 2 * axis - col
        if mirror_col >= w or mirror_col <= axis:
            continue
        left_runs = _runs(np.where(clean[:, col] == int(fg))[0])
        right_run_set = set(_runs(np.where(clean[:, mirror_col] == int(fg))[0]))
        blocked = _blocked_v_rows(clean, col, fg) | _blocked_v_rows(clean, mirror_col, fg)
        for r0, r1 in left_runs:
            if (r0, r1) not in right_run_set:
                continue
            if r1 - r0 + 1 < GAP_SIZE + 4:
                continue
            for gap_start in range(r0 + GAP_INSET, r1 - GAP_SIZE):
                if any(row in blocked for row in range(gap_start, gap_start + GAP_SIZE)):
                    continue
                temp = clean.copy()
                temp[gap_start : gap_start + GAP_SIZE, col] = int(bg)
                temp[gap_start : gap_start + GAP_SIZE, mirror_col] = int(bg)
                for anchor in ("top", "bottom"):
                    left_chain = _v_chain_coords(col, gap_start, anchor, 1)
                    right_chain = _v_chain_coords(mirror_col, gap_start, anchor, -1)
                    all_chain = left_chain + right_chain
                    if not _cells_are_bg(temp, all_chain, bg):
                        continue
                    if not _validate_chain(temp, left_chain, fg):
                        continue
                    if not _validate_chain(temp, right_chain, fg):
                        continue
                    options.append(
                        {
                            "kind": "v",
                            "top": int(gap_start),
                            "left_col": int(col),
                            "right_col": int(mirror_col),
                            "left_chain": left_chain,
                            "right_chain": right_chain,
                        }
                    )
    return options


def _impact_cells(option: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if option["kind"] == "h":
        row = int(option["row"])
        for c in range(int(option["left_start"]), int(option["left_start"]) + GAP_SIZE):
            cells.add((row, c))
        for c in range(int(option["right_start"]), int(option["right_start"]) + GAP_SIZE):
            cells.add((row, c))
    else:
        top = int(option["top"])
        for r in range(top, top + GAP_SIZE):
            cells.add((r, int(option["left_col"])))
            cells.add((r, int(option["right_col"])))
    for cell in option["left_chain"] + option["right_chain"]:
        cells.add((int(cell[0]), int(cell[1])))
    return cells


def _build_clean_network(rng: random.Random, h: int, w: int, bg: int, fg: int) -> tuple[np.ndarray, int] | None:
    axis = w // 2
    half_max = axis - 2
    if half_max < 6:
        return None

    clean = np.full((h, w), int(bg), dtype=int)
    trunk_r0 = rng.randint(1, 3)
    trunk_r1 = rng.randint(h - 5, h - 2)
    _draw_v(clean, axis, trunk_r0, trunk_r1, fg)

    n_crossbars = rng.choice([2, 3])
    row_pool = list(range(trunk_r0 + 3, trunk_r1 - 2))
    cross_rows = _pick_spaced(rng, row_pool, n_crossbars, ROW_SPACING)
    if len(cross_rows) != n_crossbars:
        return None

    half_lengths: dict[int, int] = {}
    for row in cross_rows:
        half = rng.randint(6, half_max)
        half_lengths[int(row)] = int(half)
        _draw_h(clean, int(row), axis - half, axis + half, fg)

    n_outer_pairs = 2 if n_crossbars == 2 else rng.choice([1, 2])
    used_offsets: set[int] = set()
    for _ in range(n_outer_pairs):
        eligible = [(row, half) for row, half in half_lengths.items() if half >= 4]
        if not eligible:
            return None
        row, half = rng.choice(eligible)
        offset_choices = [d for d in range(3, half - 1) if d not in used_offsets]
        if not offset_choices:
            return None
        offset = rng.choice(offset_choices)
        used_offsets.add(int(offset))
        left_col = axis - offset
        right_col = axis + offset
        if rng.choice([True, False]):
            _draw_v(clean, left_col, 0, row, fg)
            _draw_v(clean, right_col, 0, row, fg)
        else:
            _draw_v(clean, left_col, row, h - 1, fg)
            _draw_v(clean, right_col, row, h - 1, fg)

    return clean, int(axis)


def _apply_option(output: np.ndarray, input_grid: np.ndarray, option: dict, fg: int, bg: int) -> None:
    if option["kind"] == "h":
        row = int(option["row"])
        ls = int(option["left_start"])
        rs = int(option["right_start"])
        input_grid[row, ls : ls + GAP_SIZE] = int(bg)
        input_grid[row, rs : rs + GAP_SIZE] = int(bg)
    else:
        top = int(option["top"])
        lc = int(option["left_col"])
        rc = int(option["right_col"])
        input_grid[top : top + GAP_SIZE, lc] = int(bg)
        input_grid[top : top + GAP_SIZE, rc] = int(bg)

    for r, c in option["left_chain"] + option["right_chain"]:
        output[int(r), int(c)] = int(fg)
        input_grid[int(r), int(c)] = int(fg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve diagonal stair hints while repairing only symmetry-matched breaks.

    Input structure:
    - One foreground color forms an axis-aligned line network plus short 45-degree hint staircases.
    - The orthogonal line network is organized around a single vertical mirror axis, and some mirrored
      left/right locations contain 3-cell horizontal or vertical gaps.

    Rule:
    - Keep all foreground pixels, including diagonal-only hint pixels.
    - Infer the network's vertical symmetry axis from the non-diagonal line pixels.
    - Fill a 3-cell gap only when the mirrored location across that axis is the same kind of gap and
      both sides have nearby diagonal hints.

    Key insight: The stair pixels are now persistent decorations rather than noise. Symmetry decides
    which breaks are genuine repairs, so "hint nearby" is insufficient without a mirrored partner.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors, "grid must contain one foreground color"
    fg = int(next(iter(colors)))

    mask = g == int(fg)
    diag = _diag_mask(mask)
    line = mask & (~diag)
    assert bool(line.any()), "input must contain orthogonal line pixels"

    h, w = g.shape
    axis = _best_vertical_axis(line)
    out = g.copy()

    seen_h: set[tuple[int, int, int]] = set()
    for row in range(h):
        for (a1, b1), (a2, _b2) in zip(_runs(np.where(line[row, :])[0]), _runs(np.where(line[row, :])[0])[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != GAP_SIZE:
                continue
            if not _hint_near_h(diag, row, gap0, gap1):
                continue
            mirror0 = 2 * axis - gap1
            mirror1 = 2 * axis - gap0
            if mirror0 == gap0 and mirror1 == gap1:
                continue
            if not _is_row_gap(line, row, mirror0, mirror1):
                continue
            if not _hint_near_h(diag, row, mirror0, mirror1):
                continue
            key = (int(row), min(gap0, mirror0), max(gap1, mirror1))
            if key in seen_h:
                continue
            seen_h.add(key)
            out[row, gap0 : gap1 + 1] = int(fg)
            out[row, mirror0 : mirror1 + 1] = int(fg)

    seen_v: set[tuple[int, int, int]] = set()
    for col in range(w):
        runs = _runs(np.where(line[:, col])[0])
        for (a1, b1), (a2, _b2) in zip(runs, runs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != GAP_SIZE:
                continue
            if not _hint_near_v(diag, gap0, gap1, col):
                continue
            mirror_col = 2 * axis - col
            if mirror_col == col:
                continue
            if not _is_col_gap(line, mirror_col, gap0, gap1):
                continue
            if not _hint_near_v(diag, gap0, gap1, mirror_col):
                continue
            key = (int(gap0), int(gap1), min(col, mirror_col))
            if key in seen_v:
                continue
            seen_v.add(key)
            out[gap0 : gap1 + 1, col] = int(fg)
            out[gap0 : gap1 + 1, mirror_col] = int(fg)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.choice(GRID_W_CHOICES)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])

    built = _build_clean_network(rng, h, w, bg, fg)
    if built is None:
        return None
    clean, axis = built

    h_options = _horizontal_pair_options(clean, axis, fg, bg)
    v_options = _vertical_pair_options(clean, axis, fg, bg)
    if not h_options or not v_options:
        return None

    rng.shuffle(h_options)
    rng.shuffle(v_options)
    output = clean.copy()
    input_grid = clean.copy()
    reserved: set[tuple[int, int]] = set()
    selected: list[dict] = []

    for pool in (h_options, v_options):
        picked = None
        for option in pool:
            impact = _impact_cells(option)
            if impact.isdisjoint(reserved):
                picked = option
                reserved |= impact
                selected.append(option)
                break
        if picked is None:
            return None

    extra_pool = h_options + v_options
    rng.shuffle(extra_pool)
    for option in extra_pool:
        impact = _impact_cells(option)
        if impact.isdisjoint(reserved):
            reserved |= impact
            selected.append(option)
            break

    for option in selected:
        _apply_option(output, input_grid, option, fg, bg)

    try:
        if not np.array_equal(main(input_grid), output):
            return None
    except Exception:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo variant")
