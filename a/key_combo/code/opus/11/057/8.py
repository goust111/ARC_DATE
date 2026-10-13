from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 7
PATTERN = 2
OCCLUDER = 3

GRID_H_RANGE = (10, 15)
PERIOD_CHOICES = (3, 4)
PERIOD_COUNT_CHOICES = (4, 5, 6)
BAND_HEIGHT_RANGE = (3, 6)
GEN_ATTEMPTS = 600


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solid_rectangle_component(grid: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = _components_of_color(grid, color)
    assert len(comps) == 1, "expected exactly one occluder component"
    coords = comps[0]
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    assert int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1), "occluder must be a filled rectangle"
    return coords, box


def _pattern_span(grid: np.ndarray) -> tuple[int, int]:
    red = np.argwhere(grid == PATTERN)
    assert red.size > 0, "expected visible red pattern"
    return int(red[:, 1].min()), int(red[:, 1].max())


def _period_support_rows(grid: np.ndarray, occ_mask: np.ndarray) -> list[int]:
    visible_rows = sorted(set(int(r) for r, _ in np.argwhere(grid == PATTERN)))
    assert visible_rows, "expected visible pattern rows"
    rows = [r for r in range(visible_rows[0], visible_rows[-1] + 1) if np.any((grid[r] == PATTERN) | occ_mask[r])]
    assert rows, "expected non-empty pattern band"
    return rows


def _period_matches(
    grid: np.ndarray,
    occ_mask: np.ndarray,
    rows: list[int],
    period: int,
    c0: int,
    c1: int,
) -> bool:
    support = 0
    for r in rows:
        for c in range(c0, c1 - period + 1):
            if occ_mask[r, c] or occ_mask[r, c + period]:
                continue
            support += 1
            if int(grid[r, c]) != int(grid[r, c + period]):
                return False
    return support > 0


def _detect_period(grid: np.ndarray, occ_mask: np.ndarray) -> int:
    rows = _period_support_rows(grid, occ_mask)
    c0, c1 = _pattern_span(grid)
    width = c1 - c0 + 1
    for period in range(1, width):
        if _period_matches(grid, occ_mask, rows, period, c0, c1):
            return int(period)
    raise AssertionError("no horizontal period found")


def _repair_occlusion(
    grid: np.ndarray,
    occ_mask: np.ndarray,
    period: int,
    bg: int,
    c0: int,
    c1: int,
) -> np.ndarray:
    out = grid.copy()
    coords = np.argwhere(occ_mask)
    for r, c in coords:
        value = None
        for step in range(1, c1 - c0 + 2):
            for sign in (-1, 1):
                cc = int(c) + sign * step * int(period)
                if c0 <= cc <= c1 and not occ_mask[int(r), cc]:
                    value = int(out[int(r), cc])
                    break
            if value is not None:
                break
        out[int(r), int(c)] = int(bg if value is None else value)
    return out


def _extend_right(grid: np.ndarray, period: int, bg: int) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h, w + period), int(bg), dtype=int)
    out[:, :w] = grid
    out[:, w:] = grid[:, w - period : w]
    return out


def _extend_left(grid: np.ndarray, period: int, bg: int) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h, w + period), int(bg), dtype=int)
    out[:, period:] = grid
    out[:, :period] = grid[:, :period]
    return out


def _build_output_from_repaired(grid: np.ndarray, period: int, occ_center: float) -> np.ndarray:
    red = np.argwhere(grid == PATTERN)
    assert red.size > 0, "expected repaired red pattern"
    pat_c0 = int(red[:, 1].min())
    pat_c1 = int(red[:, 1].max())
    pat_center = (pat_c0 + pat_c1) / 2.0
    bg = _border_background(grid)
    if occ_center < pat_center:
        return _extend_right(grid, period, bg)
    return _extend_left(grid, period, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the green rectangular occluder, repair the hidden translational pattern, and extend it one full period opposite the occluder side.

    Input structure:
    - The border-dominant background is orange, a red foreground pattern occupies only a middle band of rows, and that pattern repeats horizontally with a fixed period.
    - A single green 4-connected filled rectangle lies inside the pattern band and hides some of the red pattern cells.
    - The remaining visible red cells still expose the horizontal translational symmetry.

    Rule:
    - Identify the unique solid green rectangle, record its horizontal center, and erase it to the orange background.
    - Detect the smallest positive horizontal translation period of the red pattern from the still-visible cells.
    - Use that period to copy values horizontally and repair every erased occluded cell.
    - Compute the repaired red pattern's horizontal center: if the occluder center is left of it, append one period on the right; otherwise prepend one period on the left.

    Key insight:
    - The occluder is both noise and a control signal: after its hidden pixels are reconstructed by translation symmetry, its left-vs-right position uniquely determines the direction of the final growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_background(g)
    occ_coords, occ_box = _solid_rectangle_component(g, OCCLUDER)
    occ_mask = np.zeros(g.shape, dtype=bool)
    occ_mask[occ_coords[:, 0], occ_coords[:, 1]] = True

    erased = g.copy()
    erased[occ_mask] = int(bg)

    period = _detect_period(erased, occ_mask)
    pat_c0, pat_c1 = _pattern_span(erased)
    repaired = _repair_occlusion(erased, occ_mask, period, bg, pat_c0, pat_c1)

    occ_center = (int(occ_box[1]) + int(occ_box[3])) / 2.0
    return _build_output_from_repaired(repaired, period, occ_center)


def _is_strict_period(tile: np.ndarray, period: int) -> bool:
    strip = np.tile(tile, (1, 4))
    for smaller in range(1, period):
        if np.array_equal(strip[:, :-smaller], strip[:, smaller:]):
            return False
    return True


def _random_tile(period: int, band_h: int) -> np.ndarray | None:
    for _ in range(200):
        tile = np.zeros((band_h, period), dtype=int)
        for r in range(band_h):
            ones = random.randint(1, period - 1)
            cols = random.sample(range(period), ones)
            tile[r, cols] = 1

        if np.any(tile.sum(axis=0) == 0):
            continue
        if not tile[:, 0].any() or not tile[:, -1].any():
            continue
        if tile.sum() <= band_h or tile.sum() >= band_h * period:
            continue
        if not _is_strict_period(tile, period):
            continue
        return tile
    return None


def _render_clean_case(
    period: int,
    n_periods: int,
    band_h: int,
    top: int,
    tile: np.ndarray,
    left_margin: int,
    right_margin: int,
) -> np.ndarray:
    h = top + band_h + random.randint(2, 4)
    pattern_w = period * n_periods
    w = left_margin + pattern_w + right_margin
    grid = np.full((h, w), BG, dtype=int)
    band = np.tile(tile, (1, n_periods))
    grid[top : top + band_h, left_margin : left_margin + pattern_w] = np.where(band == 1, PATTERN, BG)
    return grid


def _occluder_candidates(
    clean: np.ndarray,
    period: int,
    n_periods: int,
    top: int,
    band_h: int,
    extend_dir: str,
) -> list[tuple[int, int, int, int]]:
    h, w = clean.shape
    occ_h_choices = [v for v in range(2, min(3, band_h) + 1)]
    occ_w_choices = [v for v in range(2, min(4, period) + 1)]
    assert h > 0 and w > 0

    if extend_dir == "right":
        period_indices = list(range(0, n_periods // 2))
    else:
        period_indices = list(range((n_periods + 1) // 2, n_periods))

    pat_c0, pat_c1 = _pattern_span(clean)
    candidates: list[tuple[int, int, int, int]] = []
    for occ_h in occ_h_choices:
        for occ_w in occ_w_choices:
            for period_idx in period_indices:
                base_c = pat_c0 + period_idx * period
                for dc in range(0, period - occ_w + 1):
                    c0 = base_c + dc
                    c1 = c0 + occ_w - 1
                    if not (pat_c0 <= c0 <= c1 <= pat_c1):
                        continue
                    for dr in range(0, band_h - occ_h + 1):
                        r0 = top + dr
                        r1 = r0 + occ_h - 1
                        patch = clean[r0 : r1 + 1, c0 : c1 + 1]
                        if np.count_nonzero(patch == PATTERN) == 0:
                            continue
                        candidates.append((r0, c0, r1, c1))
    return candidates


def _apply_occluder(clean: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    inp = clean.copy()
    r0, c0, r1, c1 = box
    inp[r0 : r1 + 1, c0 : c1 + 1] = OCCLUDER
    return inp


def _expected_from_clean(clean: np.ndarray, period: int, extend_dir: str) -> np.ndarray:
    if extend_dir == "right":
        return _extend_right(clean, period, BG)
    return _extend_left(clean, period, BG)


def _naive_without_repair(inp: np.ndarray, period: int, extend_dir: str) -> np.ndarray:
    erased = inp.copy()
    erased[erased == OCCLUDER] = BG
    if extend_dir == "right":
        return _extend_right(erased, period, BG)
    return _extend_left(erased, period, BG)


def _opposite_direction(clean: np.ndarray, period: int, extend_dir: str) -> np.ndarray:
    if extend_dir == "right":
        return _extend_left(clean, period, BG)
    return _extend_right(clean, period, BG)


def _validate_case(
    inp: np.ndarray,
    clean: np.ndarray,
    expected: np.ndarray,
    period: int,
    extend_dir: str,
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    occ_coords, occ_box = _solid_rectangle_component(inp, OCCLUDER)
    occ_mask = np.zeros(inp.shape, dtype=bool)
    occ_mask[occ_coords[:, 0], occ_coords[:, 1]] = True
    if np.count_nonzero(clean[occ_mask] == PATTERN) == 0:
        return False

    erased = inp.copy()
    erased[occ_mask] = BG
    detected = _detect_period(erased, occ_mask)
    if detected != int(period):
        return False
    for smaller in range(1, period):
        pat_c0, pat_c1 = _pattern_span(erased)
        if _period_matches(erased, occ_mask, _period_support_rows(erased, occ_mask), smaller, pat_c0, pat_c1):
            return False

    repaired = _repair_occlusion(erased, occ_mask, period, BG, pat_c0, pat_c1)
    if not np.array_equal(repaired, clean):
        return False

    red = np.argwhere(clean == PATTERN)
    pat_center = (int(red[:, 1].min()) + int(red[:, 1].max())) / 2.0
    occ_center = (int(occ_box[1]) + int(occ_box[3])) / 2.0
    if (extend_dir == "right") != (occ_center < pat_center):
        return False

    if np.array_equal(_naive_without_repair(inp, period, extend_dir), expected):
        return False
    if np.array_equal(_opposite_direction(clean, period, extend_dir), expected):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    period = random.choice(PERIOD_CHOICES)
    n_periods = random.choice(PERIOD_COUNT_CHOICES)

    band_h = random.randint(*BAND_HEIGHT_RANGE)
    tile = _random_tile(period, band_h)
    if tile is None:
        return None

    top = random.randint(2, 5)
    left_margin = random.randint(1, period - 1)
    right_margin = random.randint(1, period - 1)
    width = left_margin + period * n_periods + right_margin
    if width + period > 30:
        return None
    clean = _render_clean_case(period, n_periods, band_h, top, tile, left_margin, right_margin)
    if max(clean.shape) > 30:
        return None

    extend_dir = random.choice(("left", "right"))
    candidates = _occluder_candidates(clean, period, n_periods, top, band_h, extend_dir)
    if not candidates:
        return None

    random.shuffle(candidates)
    for box in candidates:
        inp = _apply_occluder(clean, box)
        expected = _expected_from_clean(clean, period, extend_dir)
        if _validate_case(inp, clean, expected, period, extend_dir):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
