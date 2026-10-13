from __future__ import annotations

import random

import numpy as np


BLACK = 0
INPUT_COLORS = list(range(1, 10))
PERIOD_SEARCH_CAP = 8

PERIOD_MIN = 2
PERIOD_MAX = 5
REPEAT_MIN = 3
REPEAT_MAX = 5
TILE_COLOR_COUNT_MIN = 4
TILE_COLOR_COUNT_MAX = 6

STAMP_MIN_SIZE = 4
STAMP_MAX_SIZE = 9
VISIBLE_FRAC_MIN = 0.45
VISIBLE_FRAC_MAX = 0.9

OUTER_ATTEMPTS = 500
INNER_ATTEMPTS = 300
SIGNAL_GAP = 0.12
DIVISOR_GAP = 0.08


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % th, c % tw])
    return out


def _is_primitive_tile(tile: np.ndarray) -> bool:
    pr, pc = tile.shape
    for shift in range(1, pr):
        if np.array_equal(tile[:-shift, :], tile[shift:, :]):
            return False
    for shift in range(1, pc):
        if np.array_equal(tile[:, :-shift], tile[:, shift:]):
            return False
    return True


def _period_match_scores(grid: np.ndarray, axis: int) -> list[float]:
    n = grid.shape[axis]
    limit = min(PERIOD_SEARCH_CAP, max(1, n // 2))
    scores: list[float] = []
    for shift in range(1, limit + 1):
        if axis == 0:
            same = grid[:-shift, :] == grid[shift:, :]
        else:
            same = grid[:, :-shift] == grid[:, shift:]
        scores.append(float(np.mean(same)))
    return scores


def _majority_color(values: np.ndarray) -> tuple[int, int]:
    uniq, counts = np.unique(values, return_counts=True)
    order = np.argsort(-counts)
    color = int(uniq[order[0]])
    second = int(counts[order[1]]) if len(order) > 1 else 0
    margin = int(counts[order[0]] - second)
    return color, margin


def _rebuild_wallpaper(grid: np.ndarray, pr: int, pc: int) -> tuple[np.ndarray, int, int]:
    h, w = grid.shape
    wallpaper = np.zeros_like(grid)
    min_margin: int | None = None
    for rr in range(pr):
        rows = np.arange(rr, h, pr)
        for cc in range(pc):
            cols = np.arange(cc, w, pc)
            color, margin = _majority_color(grid[np.ix_(rows, cols)].ravel())
            wallpaper[np.ix_(rows, cols)] = color
            min_margin = margin if min_margin is None else min(min_margin, margin)
    mismatches = int(np.count_nonzero(wallpaper != grid))
    return wallpaper, mismatches, int(min_margin or 0)


def _infer_periods_and_wallpaper(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, w = grid.shape
    row_limit = min(PERIOD_SEARCH_CAP, max(1, h // 2))
    col_limit = min(PERIOD_SEARCH_CAP, max(1, w // 2))
    best: tuple[tuple[int, int, int, int], int, int, np.ndarray] | None = None
    for pr in range(1, row_limit + 1):
        for pc in range(1, col_limit + 1):
            wallpaper, mismatches, min_margin = _rebuild_wallpaper(grid, pr, pc)
            score = (mismatches, -min_margin, pr * pc, pr + pc)
            if best is None or score < best[0]:
                best = (score, pr, pc, wallpaper)
    assert best is not None
    return best[1], best[2], best[3]


def _complete_left_right(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    h, w = out.shape
    for r in range(h):
        for c in range((w + 1) // 2):
            mc = w - 1 - c
            left = int(out[r, c])
            right = int(out[r, mc])
            if c == mc:
                continue
            if left == BLACK and right != BLACK:
                out[r, c] = right
            elif right == BLACK and left != BLACK:
                out[r, mc] = left
            elif left != BLACK and right != BLACK:
                assert left == right, "mirror mates must share one color"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the periodic wallpaper, extract the residual stamp, then mirror-complete its hidden cells.

    Input structure:
    - Most of the grid follows a two-direction translation wallpaper whose row and
      column periods can be recovered by majority agreement.
    - One contiguous stamp with left-right mirror symmetry is overlaid on top of
      that wallpaper.
    - Some stamp cells are invisible in the residual because they happen to match
      the wallpaper color underneath.

    Rule:
    - Infer the wallpaper periods and reconstruct the ideal wallpaper by taking
      the majority color in each period equivalence class.
    - Compare the input to that wallpaper cell-by-cell and keep only the
      mismatching cells as the visible residual of the stamp.
    - Crop to the residual bounding box, then use left-right mirror symmetry to
      fill missing stamp cells from their reflected mates.

    Key insight:
    - Translational symmetry isolates the stamp from the wallpaper, but that
      residual is still incomplete; the stamp's own mirror symmetry is needed to
      restore cells that were swallowed because stamp and wallpaper shared a color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _pr, _pc, wallpaper = _infer_periods_and_wallpaper(g)
    residual = np.where(g != wallpaper, g, BLACK)
    box = _bbox(residual != BLACK)
    assert box is not None, "expected a non-empty residual stamp"
    visible = _crop_bbox(residual, box)
    return _complete_left_right(visible)


def _period_signal_ok(wallpaper: np.ndarray, true_period: int, axis: int) -> bool:
    scores = _period_match_scores(wallpaper, axis)
    true_score = scores[true_period - 1]
    for shift, score in enumerate(scores, start=1):
        if shift == true_period or shift % true_period == 0:
            continue
        if true_score < score + SIGNAL_GAP:
            return False
    for shift in range(1, true_period):
        if true_period % shift == 0 and true_score < scores[shift - 1] + DIVISOR_GAP:
            return False
    return True


def _random_wallpaper() -> tuple[np.ndarray, int, int]:
    for _ in range(INNER_ATTEMPTS):
        pr = random.randint(PERIOD_MIN, PERIOD_MAX)
        pc = random.randint(PERIOD_MIN, PERIOD_MAX)
        h = random.randint(pr * REPEAT_MIN, min(30, pr * REPEAT_MAX))
        w = random.randint(pc * REPEAT_MIN, min(30, pc * REPEAT_MAX))
        color_count = random.randint(TILE_COLOR_COUNT_MIN, TILE_COLOR_COUNT_MAX)
        palette = random.sample(INPUT_COLORS, color_count)
        tile = np.array(
            [[random.choice(palette) for _ in range(pc)] for _ in range(pr)],
            dtype=int,
        )
        if not _is_primitive_tile(tile):
            continue
        wallpaper = _tile_fill((h, w), tile)
        if _period_signal_ok(wallpaper, pr, 0) and _period_signal_ok(wallpaper, pc, 1):
            return wallpaper, pr, pc
    raise RuntimeError("failed to generate a wallpaper with a clear primitive period")


def _make_stamp_mask(h: int, w: int) -> np.ndarray:
    half = w // 2
    extents: list[int] = []
    cur = random.randint(max(1, half // 2), half)
    for _ in range(h):
        cur = max(1, min(half, cur + random.choice((-1, 0, 1))))
        extents.append(cur)
    if max(extents) < half:
        extents[random.randrange(h)] = half
    if len(set(extents)) == 1 and half > 1:
        idx = random.randrange(h)
        extents[idx] = half - 1 if extents[idx] == half else half

    mask = np.zeros((h, w), dtype=bool)
    for r, extent in enumerate(extents):
        left = half - extent
        right = w - left
        mask[r, left:right] = True
    return mask


def _color_stamp_against_wallpaper(
    mask: np.ndarray, wallpaper_window: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    h, w = mask.shape
    stamp = np.zeros((h, w), dtype=int)
    visible = np.zeros((h, w), dtype=bool)
    hidden_cells = 0
    asymmetric_pairs = 0

    palette = sorted(set(map(int, wallpaper_window[mask].ravel())))
    for r in range(h):
        for c in range((w + 1) // 2):
            mc = w - 1 - c
            if not mask[r, c] and not mask[r, mc]:
                continue

            if c == mc:
                wall_color = int(wallpaper_window[r, c])
                choices = [color for color in palette if color != wall_color]
                if not choices:
                    return stamp, visible, 0, 0, 0
                stamp[r, c] = random.choice(choices)
                visible[r, c] = True
                continue

            left_wall = int(wallpaper_window[r, c])
            right_wall = int(wallpaper_window[r, mc])
            if left_wall != right_wall and random.random() < 0.6:
                color = random.choice((left_wall, right_wall))
            else:
                choices = [color for color in palette if color != left_wall and color != right_wall]
                if not choices:
                    choices = [color for color in palette if color != left_wall or color != right_wall]
                color = random.choice(choices)

            stamp[r, c] = color
            stamp[r, mc] = color
            left_visible = color != left_wall
            right_visible = color != right_wall
            visible[r, c] = left_visible
            visible[r, mc] = right_visible
            hidden_cells += int(not left_visible) + int(not right_visible)
            asymmetric_pairs += int(left_visible != right_visible)

    return stamp, visible, hidden_cells, int(np.count_nonzero(visible)), asymmetric_pairs


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    wallpaper: np.ndarray,
    pr: int,
    pc: int,
    mask: np.ndarray,
    stamp: np.ndarray,
    visible: np.ndarray,
    top: int,
    left: int,
) -> bool:
    try:
        pred = main(inp)
    except Exception:
        return False
    if not np.array_equal(pred, expected):
        return False

    got_pr, got_pc, rebuilt = _infer_periods_and_wallpaper(inp)
    if (got_pr, got_pc) != (pr, pc):
        return False
    if not np.array_equal(rebuilt, wallpaper):
        return False

    residual = np.where(inp != wallpaper, inp, BLACK)
    box = _bbox(residual != BLACK)
    if box != (top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1):
        return False

    crop = _crop_bbox(residual, box)
    if np.array_equal(crop, expected):
        return False
    if not np.array_equal(_complete_left_right(crop), expected):
        return False
    if not np.array_equal(expected, np.fliplr(expected)):
        return False

    if not np.any(mask & (stamp == wallpaper[top : top + mask.shape[0], left : left + mask.shape[1]])):
        return False
    if not np.any(visible):
        return False

    for r in range(mask.shape[0]):
        for c in range((mask.shape[1] + 1) // 2):
            mc = mask.shape[1] - 1 - c
            if not mask[r, c] and not mask[r, mc]:
                continue
            if c == mc:
                if not visible[r, c]:
                    return False
            elif not (visible[r, c] or visible[r, mc]):
                return False

    return True


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    wallpaper, pr, pc = _random_wallpaper()
    h_grid, w_grid = wallpaper.shape

    for _ in range(INNER_ATTEMPTS):
        stamp_h = random.randint(STAMP_MIN_SIZE, min(STAMP_MAX_SIZE, h_grid - 2))
        stamp_w = random.randint(STAMP_MIN_SIZE, min(STAMP_MAX_SIZE, w_grid - 2))
        top = random.randint(1, h_grid - stamp_h - 1)
        left = random.randint(1, w_grid - stamp_w - 1)

        mask = _make_stamp_mask(stamp_h, stamp_w)
        window = wallpaper[top : top + stamp_h, left : left + stamp_w]
        stamp, visible, hidden_cells, visible_cells, asymmetric_pairs = _color_stamp_against_wallpaper(mask, window)

        if hidden_cells <= 0 or asymmetric_pairs <= 0:
            continue
        if visible_cells < int(mask.sum() * VISIBLE_FRAC_MIN):
            continue
        if visible_cells > int(mask.sum() * VISIBLE_FRAC_MAX):
            continue
        if len(set(map(int, stamp[mask].ravel()))) < 2:
            continue

        visible_box = _bbox(visible)
        if visible_box != (0, 0, stamp_h - 1, stamp_w - 1):
            continue

        inp = wallpaper.copy()
        inp[top : top + stamp_h, left : left + stamp_w][mask] = stamp[mask]
        expected = stamp.copy()
        expected[~mask] = BLACK

        if _semantic_ok(inp, expected, wallpaper, pr, pc, mask, stamp, visible, top, left):
            return inp, expected

    raise RuntimeError("failed to place a valid mirror-repair stamp")


def generate_input() -> np.ndarray:
    for _ in range(OUTER_ATTEMPTS):
        inp, _expected = _build_instance()
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
