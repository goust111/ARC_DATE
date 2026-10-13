from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


def _infer_axis_period(is_gap_line: np.ndarray) -> tuple[int, int]:
    """Return (period, gap_residue) from full-background rows or columns."""
    flags = np.asarray(is_gap_line, dtype=bool)
    n = int(flags.size)
    candidates: list[tuple[int, int, int]] = []
    for period in range(2, n):
        for gap_residue in range(period):
            indices = np.arange(gap_residue, n, period, dtype=int)
            if indices.size < 2:
                continue
            if bool(flags[indices].all()):
                candidates.append((period, int(indices.size), gap_residue))

    assert candidates, "failed to infer periodic gap lines"

    best_period = min(period for period, _count, _gap in candidates)
    period_candidates = [item for item in candidates if item[0] == best_period]
    best_count = max(count for _period, count, _gap in period_candidates)
    winners = [item for item in period_candidates if item[1] == best_count]
    assert len(winners) == 1, "ambiguous periodic gap axis"
    period, _count, gap_residue = winners[0]
    return int(period), int(gap_residue)


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values.astype(int), return_counts=True)
    order = np.argsort(-counts)
    assert order.size > 0
    if order.size >= 2:
        assert int(counts[order[0]]) > int(counts[order[1]]), "majority must be unique"
    return int(colors[order[0]])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique stamp template by majority-voting aligned periodic copies.

    Input structure:
    - A dominant background contains an all-grid periodic array of one unknown template,
      with exactly one background row and one background column separating neighboring copies.
    - The array is clipped by the grid borders, so edge copies may show only partial fragments.
    - Every visible copy is damaged in a few cells, but the gap rows and gap columns remain pure background.

    Rule:
    - Infer the vertical and horizontal repeat periods from the globally background-only separator rows and columns.
    - Treat every non-gap residue pair modulo those periods as one template coordinate, aligning full and clipped copies together.
    - For each aligned template coordinate, output the unique majority color seen across all covering copies.

    Key insight:
    - The separators reveal the period and phase directly; once those are known, the puzzle is an inverse stamping problem solved by per-cell majority vote over all aligned fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    gap_rows = np.all(g == bg, axis=1)
    gap_cols = np.all(g == bg, axis=0)

    period_h, gap_r = _infer_axis_period(gap_rows)
    period_w, gap_c = _infer_axis_period(gap_cols)
    template_h = period_h - 1
    template_w = period_w - 1
    assert template_h >= 1 and template_w >= 1

    out = np.empty((template_h, template_w), dtype=int)
    row_mod = np.arange(g.shape[0], dtype=int) % period_h
    col_mod = np.arange(g.shape[1], dtype=int) % period_w

    for tr in range(template_h):
        row_residue = (gap_r + 1 + tr) % period_h
        row_idx = np.flatnonzero(row_mod == row_residue)
        assert row_idx.size > 0
        for tc in range(template_w):
            col_residue = (gap_c + 1 + tc) % period_w
            col_idx = np.flatnonzero(col_mod == col_residue)
            assert col_idx.size > 0
            values = g[np.ix_(row_idx, col_idx)].reshape(-1)
            out[tr, tc] = _majority_color(values)

    return out


_RNG = np.random.default_rng()
_BG = 0
_PALETTE = tuple(range(10))
_NON_BG = tuple(range(1, 10))
_TEMPLATE_HW_RANGE = (3, 7)
_FG_COLOR_COUNT_RANGE = (2, 4)
_PERIOD_MULTIPLIER_RANGE = (3, 5)
_MAX_GENERATION_ATTEMPTS = 400


def _mask_connected_8(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, coords[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    reached = 0
    while stack:
        r, c = stack.pop()
        reached += 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return reached == int(mask.sum())


def _has_smaller_axis_period(arr: np.ndarray, axis: int) -> bool:
    size = int(arr.shape[axis])
    for period in range(1, size):
        if axis == 0:
            ok = all(np.array_equal(arr[r], arr[r % period]) for r in range(size))
        else:
            ok = all(np.array_equal(arr[:, c], arr[:, c % period]) for c in range(size))
        if ok:
            return True
    return False


def _build_template() -> np.ndarray:
    while True:
        h = int(_RNG.integers(*_TEMPLATE_HW_RANGE))
        w = int(_RNG.integers(*_TEMPLATE_HW_RANGE))
        area = h * w
        fg_cells = int(_RNG.integers(int(np.ceil(area * 0.5)), int(np.floor(area * 0.8)) + 1))

        mask = np.zeros((h, w), dtype=bool)
        mandatory = {(r, int(_RNG.integers(w))) for r in range(h)}
        mandatory |= {(int(_RNG.integers(h)), c) for c in range(w)}
        for r in (0, h - 1):
            mandatory.add((r, int(_RNG.integers(w))))
        for c in (0, w - 1):
            mandatory.add((int(_RNG.integers(h)), c))

        mandatory_list = list(mandatory)
        if len(mandatory_list) > fg_cells:
            continue
        for r, c in mandatory_list:
            mask[r, c] = True

        extra_slots = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
        _RNG.shuffle(extra_slots)
        for r, c in extra_slots[: fg_cells - len(mandatory_list)]:
            mask[r, c] = True

        if not _mask_connected_8(mask):
            continue
        if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
            continue

        fg_color_count = int(_RNG.integers(*_FG_COLOR_COUNT_RANGE))
        colors = _RNG.choice(_NON_BG, size=fg_color_count, replace=False)
        template = np.full((h, w), _BG, dtype=int)
        filled = np.argwhere(mask)
        color_ids = _RNG.integers(0, fg_color_count, size=len(filled))
        for idx, (r, c) in enumerate(filled):
            template[int(r), int(c)] = int(colors[int(color_ids[idx])])

        present = set(map(int, template[template != _BG]))
        if len(present) < 2:
            continue
        if _has_smaller_axis_period(template, axis=0) or _has_smaller_axis_period(template, axis=1):
            continue
        return template


def _render_clean_grid(template: np.ndarray, grid_h: int, grid_w: int, gap_r: int, gap_c: int) -> np.ndarray:
    period_h = template.shape[0] + 1
    period_w = template.shape[1] + 1
    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    row_mod = np.arange(grid_h, dtype=int) % period_h
    col_mod = np.arange(grid_w, dtype=int) % period_w

    for r in range(grid_h):
        if row_mod[r] == gap_r:
            continue
        tr = (r - gap_r - 1) % period_h
        for c in range(grid_w):
            if col_mod[c] == gap_c:
                continue
            tc = (c - gap_c - 1) % period_w
            grid[r, c] = int(template[tr, tc])
    return grid


def _segment_bounds(gap_indices: np.ndarray, n: int) -> list[tuple[int, int]]:
    bounds = [-1, *map(int, gap_indices), int(n)]
    segments: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        start = left + 1
        stop = right
        if start < stop:
            segments.append((start, stop))
    return segments


def _damage_color(original: int, fg_colors: list[int], extra_colors: list[int], force_to_bg: bool | None = None) -> int:
    if force_to_bg is True and original != _BG:
        return _BG
    choices = list(dict.fromkeys(fg_colors + extra_colors))
    if original == _BG:
        return int(_RNG.choice(choices))
    if force_to_bg is False:
        non_bg_choices = [c for c in choices if c != original]
        return int(_RNG.choice(non_bg_choices))
    if _RNG.random() < 0.5:
        return _BG
    non_bg_choices = [c for c in choices if c != original]
    return int(_RNG.choice(non_bg_choices))


def _template_coord(r: int, c: int, period_h: int, period_w: int, gap_r: int, gap_c: int) -> tuple[int, int]:
    return (int((r - gap_r - 1) % period_h), int((c - gap_c - 1) % period_w))


def _validate_generated(
    grid: np.ndarray,
    clean: np.ndarray,
    template: np.ndarray,
    period_h: int,
    period_w: int,
    gap_r: int,
    gap_c: int,
    copy_bounds: list[tuple[int, int, int, int]],
) -> bool:
    gap_rows = np.flatnonzero(np.all(grid == _BG, axis=1))
    gap_cols = np.flatnonzero(np.all(grid == _BG, axis=0))
    expected_gap_rows = np.arange(gap_r, grid.shape[0], period_h, dtype=int)
    expected_gap_cols = np.arange(gap_c, grid.shape[1], period_w, dtype=int)
    if not np.array_equal(gap_rows, expected_gap_rows):
        return False
    if not np.array_equal(gap_cols, expected_gap_cols):
        return False

    for r0, r1, c0, c1 in copy_bounds:
        if np.array_equal(grid[r0:r1, c0:c1], clean[r0:r1, c0:c1]):
            return False

    votes: dict[tuple[int, int], Counter[int]] = {
        (tr, tc): Counter() for tr in range(template.shape[0]) for tc in range(template.shape[1])
    }
    for r in range(grid.shape[0]):
        if r % period_h == gap_r:
            continue
        for c in range(grid.shape[1]):
            if c % period_w == gap_c:
                continue
            votes[_template_coord(r, c, period_h, period_w, gap_r, gap_c)][int(grid[r, c])] += 1

    for key, counter in votes.items():
        ranked = counter.most_common(2)
        if len(ranked) == 0:
            return False
        if len(ranked) >= 2 and ranked[0][1] <= ranked[1][1]:
            return False
        if int(ranked[0][0]) != int(template[key]):
            return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, template)


def _gen_attempt() -> np.ndarray | None:
    template = _build_template()
    period_h = int(template.shape[0] + 1)
    period_w = int(template.shape[1] + 1)

    max_k_h = min(_PERIOD_MULTIPLIER_RANGE[1], (30 + 1) // period_h)
    max_k_w = min(_PERIOD_MULTIPLIER_RANGE[1], (30 + 1) // period_w)
    if max_k_h < _PERIOD_MULTIPLIER_RANGE[0] or max_k_w < _PERIOD_MULTIPLIER_RANGE[0]:
        return None

    k_h = int(_RNG.integers(_PERIOD_MULTIPLIER_RANGE[0], max_k_h + 1))
    k_w = int(_RNG.integers(_PERIOD_MULTIPLIER_RANGE[0], max_k_w + 1))
    grid_h = int(k_h * period_h - 1)
    grid_w = int(k_w * period_w - 1)
    gap_r = int(_RNG.integers(1, period_h - 1))
    gap_c = int(_RNG.integers(1, period_w - 1))

    clean = _render_clean_grid(template, grid_h, grid_w, gap_r, gap_c)
    grid = clean.copy()

    gap_rows = np.flatnonzero(np.all(clean == _BG, axis=1))
    gap_cols = np.flatnonzero(np.all(clean == _BG, axis=0))
    row_segments = _segment_bounds(gap_rows, grid_h)
    col_segments = _segment_bounds(gap_cols, grid_w)
    copy_bounds = [(r0, r1, c0, c1) for r0, r1 in row_segments for c0, c1 in col_segments]

    fg_colors = sorted(set(map(int, template[template != _BG])))
    extra_pool = [c for c in _NON_BG if c not in fg_colors]
    extra_count = min(len(extra_pool), int(_RNG.integers(1, 3)))
    extra_colors = list(map(int, _RNG.choice(extra_pool, size=extra_count, replace=False))) if extra_count else []

    forced_modes = ["to_bg", "to_color"]
    shuffled_bounds = copy_bounds.copy()
    _RNG.shuffle(shuffled_bounds)
    for idx, (r0, r1, c0, c1) in enumerate(shuffled_bounds):
        positions = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
        if not positions:
            return None

        max_damage = min(4, len(positions))
        damage_count = int(_RNG.integers(1, max_damage + 1))
        _RNG.shuffle(positions)
        chosen = positions[:damage_count]

        if idx < len(forced_modes):
            mode = forced_modes[idx]
            if mode == "to_bg":
                non_bg_sites = [(r, c) for r, c in positions if clean[r, c] != _BG]
                if not non_bg_sites:
                    return None
                chosen[0] = non_bg_sites[0]

        for j, (r, c) in enumerate(chosen):
            if idx == 0 and j == 0:
                grid[r, c] = _damage_color(int(clean[r, c]), fg_colors, extra_colors, force_to_bg=True)
            elif idx == 1 and j == 0:
                grid[r, c] = _damage_color(int(clean[r, c]), fg_colors, extra_colors, force_to_bg=False)
            else:
                grid[r, c] = _damage_color(int(clean[r, c]), fg_colors, extra_colors)

    if not _validate_generated(grid, clean, template, period_h, period_w, gap_r, gap_c, copy_bounds):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reverse-stamp voting input")
