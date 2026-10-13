from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 400
MIN_SCORE_MARGIN = 0.50


def _separator_row(grid: np.ndarray) -> int:
    rows = [
        int(r)
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == YELLOW)
    ]
    assert len(rows) == 1, "expected exactly one yellow separator row"
    sep = rows[0]
    assert sep > 0 and sep < grid.shape[0] - 1, "separator must be internal"
    assert sep == grid.shape[0] - sep - 1, "top and bottom halves must have equal height"
    return sep


def _class_sizes(length: int, period: int) -> np.ndarray:
    return np.array(
        [((length - 1 - offset) // period) + 1 for offset in range(period)],
        dtype=int,
    )


def _period_candidates(red_mask: np.ndarray) -> list[tuple[float, int, int]]:
    h, w = map(int, red_mask.shape)
    coords = np.argwhere(red_mask)
    rankings: list[tuple[float, int, int]] = []

    for row_period in range(1, h + 1):
        row_sizes = _class_sizes(h, row_period)
        for col_period in range(1, w + 1):
            col_sizes = _class_sizes(w, col_period)
            counts = np.zeros((row_period, col_period), dtype=int)
            for r, c in coords:
                counts[int(r) % row_period, int(c) % col_period] += 1

            score = 0.0
            for rr in range(row_period):
                for cc in range(col_period):
                    n = int(counts[rr, cc])
                    if n == 0:
                        continue
                    class_size = int(row_sizes[rr] * col_sizes[cc])
                    score += n * (n - 1) / class_size

            rankings.append((float(score), int(row_period), int(col_period)))

    rankings.sort(key=lambda item: (-item[0], item[1] * item[2], item[1], item[2]))
    return rankings


def _infer_periods(red_mask: np.ndarray) -> tuple[int, int]:
    rankings = _period_candidates(red_mask)
    assert rankings and int(red_mask.sum()) > 0, "expected visible red evidence"
    _score, row_period, col_period = rankings[0]
    return int(row_period), int(col_period)


def _repair_top(top: np.ndarray, row_period: int | None = None, col_period: int | None = None) -> np.ndarray:
    g = np.asarray(top, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({BLACK, RED}), "top half must be red/black only"

    visible = g == RED
    if row_period is None or col_period is None:
        row_period, col_period = _infer_periods(visible)

    repaired = np.full_like(g, BLACK)
    for rr in range(int(row_period)):
        for cc in range(int(col_period)):
            if bool(visible[rr::row_period, cc::col_period].any()):
                repaired[rr::row_period, cc::col_period] = RED
    return repaired


def _xor_output(repaired_top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    top_red = np.asarray(repaired_top, dtype=int) == RED
    bottom_red = np.asarray(bottom, dtype=int) == RED
    return np.where(np.logical_xor(top_red, bottom_red), GREEN, BLACK).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the corrupted periodic top half, then XOR it against the lower red mask.

    Input structure:
    - One full yellow row splits the grid into equal-height upper and lower halves.
    - The upper half is a red/black periodic wallpaper where some red cells were erased to black.
    - The lower half is a red/black mask of the same size.

    Rule:
    - Infer the smallest row and column periods supported by the visible red evidence in the upper half.
    - Repair the upper half by filling every residue class that still contains at least one visible red cell.
    - Output a grid with the half-size shape: cells are green exactly where repaired-top red-ness and lower-mask red-ness differ, otherwise black.

    Key insight:
    - The visible top half is damaged periodic structure, so the missing red cells must be restored before the final cellwise XOR; using the damaged pixels directly flips the answer at every erased location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep = _separator_row(g)

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape == bottom.shape
    assert set(np.unique(bottom)).issubset({BLACK, RED}), "bottom half must be red/black only"

    repaired = _repair_top(top)
    return _xor_output(repaired, bottom)


def _has_subperiod(tile: np.ndarray) -> bool:
    h, w = map(int, tile.shape)
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            if row_period == h and col_period == w:
                continue
            if all(
                bool(tile[r, c] == tile[r % row_period, c % col_period])
                for r in range(h)
                for c in range(w)
            ):
                return True
    return False


def _valid_tile(tile: np.ndarray) -> bool:
    h, w = map(int, tile.shape)
    red_total = int(tile.sum())
    area = h * w
    density = red_total / area
    if not (3 <= red_total <= area - 3):
        return False
    if not (0.28 <= density <= 0.60):
        return False
    if _has_subperiod(tile):
        return False
    if any(int(tile[r, :].sum()) == 0 or int(tile[r, :].sum()) == w for r in range(h)):
        return False
    if any(int(tile[:, c].sum()) == 0 or int(tile[:, c].sum()) == h for c in range(w)):
        return False
    row_signatures = {tuple(int(x) for x in tile[r, :]) for r in range(h)}
    col_signatures = {tuple(int(x) for x in tile[:, c]) for c in range(w)}
    if len(row_signatures) < 2 or len(col_signatures) < 2:
        return False
    return True


def _random_tile(rng: np.random.Generator, row_period: int, col_period: int) -> np.ndarray | None:
    for _ in range(300):
        density = float(rng.uniform(0.30, 0.56))
        tile = (rng.random((row_period, col_period)) < density)
        if _valid_tile(tile):
            return tile
    return None


def _repeat_tile(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    th, tw = map(int, tile.shape)
    for r in range(h):
        for c in range(w):
            if bool(tile[r % th, c % tw]):
                out[r, c] = RED
    return out


def _rectangle_mask(
    mask: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = True


def _bottom_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    for _ in range(150):
        mask = np.zeros((h, w), dtype=bool)
        rect_count = int(rng.integers(2, 5))
        for _rect in range(rect_count):
            rh = int(rng.integers(1, max(2, h // 2) + 1))
            rw = int(rng.integers(1, max(2, w // 2) + 1))
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
            _rectangle_mask(mask, r0, c0, rh, rw)

        if bool(rng.integers(0, 2)):
            row = int(rng.integers(0, h))
            thickness = int(rng.integers(1, min(3, h - row) + 1))
            _rectangle_mask(mask, row, 0, thickness, w)
        if bool(rng.integers(0, 2)):
            col = int(rng.integers(0, w))
            thickness = int(rng.integers(1, min(3, w - col) + 1))
            _rectangle_mask(mask, 0, col, h, thickness)

        density = float(mask.mean())
        if 0.18 <= density <= 0.62 and mask.any() and (~mask).any():
            return np.where(mask, RED, BLACK).astype(int)

    raise RuntimeError("failed to generate lower mask")


def _class_coord_lists(full_top: np.ndarray, row_period: int, col_period: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    classes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    h, w = map(int, full_top.shape)
    for r in range(h):
        for c in range(w):
            if int(full_top[r, c]) != RED:
                continue
            key = (r % row_period, c % col_period)
            classes.setdefault(key, []).append((r, c))
    return classes


def _erase_reds(
    rng: np.random.Generator,
    full_top: np.ndarray,
    bottom: np.ndarray,
    row_period: int,
    col_period: int,
) -> np.ndarray | None:
    classes = _class_coord_lists(full_top, row_period, col_period)
    visible = full_top.copy()
    class_lists = {key: coords[:] for key, coords in classes.items()}
    for coords in class_lists.values():
        rng.shuffle(coords)

    keep_floor = {
        key: max(2, int(np.ceil(0.65 * len(coords))))
        for key, coords in class_lists.items()
    }
    erased: list[tuple[int, int]] = []
    visible_counts = {key: len(coords) for key, coords in class_lists.items()}

    def can_erase(coord: tuple[int, int]) -> bool:
        key = (coord[0] % row_period, coord[1] % col_period)
        return visible_counts[key] > keep_floor[key]

    red_on_red = [coord for coords in class_lists.values() for coord in coords if int(bottom[coord]) == RED]
    red_on_black = [coord for coords in class_lists.values() for coord in coords if int(bottom[coord]) == BLACK]
    rng.shuffle(red_on_red)
    rng.shuffle(red_on_black)

    mandatory: list[tuple[int, int]] = []
    for pool in (red_on_red, red_on_black):
        chosen = next((coord for coord in pool if can_erase(coord)), None)
        if chosen is None:
            return None
        mandatory.append(chosen)

    affected_classes: set[tuple[int, int]] = set()
    for coord in mandatory:
        if not can_erase(coord):
            return None
        visible[coord] = BLACK
        erased.append(coord)
        key = (coord[0] % row_period, coord[1] % col_period)
        affected_classes.add(key)
        visible_counts[key] -= 1

    all_candidates = [coord for coords in class_lists.values() for coord in coords]
    rng.shuffle(all_candidates)
    red_total = int(np.count_nonzero(full_top == RED))
    target = int(rng.integers(max(3, len(classes) // 2), max(5, red_total // 6) + 1))
    target = min(target, sum(visible_counts[key] - keep_floor[key] for key in visible_counts) + len(erased))

    for coord in all_candidates:
        if len(erased) >= target:
            break
        if not can_erase(coord):
            continue
        key = (coord[0] % row_period, coord[1] % col_period)
        visible[coord] = BLACK
        visible_counts[key] -= 1
        erased.append(coord)
        affected_classes.add(key)

    if len(erased) < 2:
        return None
    if len(affected_classes) < 2:
        return None
    rows = {r for r, _c in erased}
    cols = {c for _r, c in erased}
    if len(rows) < 2 or len(cols) < 2:
        return None
    return visible


def _validate_instance(
    inp: np.ndarray,
    repaired_top: np.ndarray,
    bottom: np.ndarray,
    row_period: int,
    col_period: int,
) -> bool:
    sep = _separator_row(inp)
    top = inp[:sep, :]
    rankings = _period_candidates(top == RED)
    best_score, best_r, best_c = rankings[0]
    second_score = rankings[1][0] if len(rankings) > 1 else -1.0
    if (best_r, best_c) != (row_period, col_period):
        return False
    if best_score - second_score < MIN_SCORE_MARGIN:
        return False

    expected = _xor_output(repaired_top, bottom)
    if not np.array_equal(main(inp), expected):
        return False

    direct = _xor_output(top, bottom)
    if np.array_equal(direct, expected):
        return False
    if int(np.count_nonzero(direct != expected)) < 2:
        return False

    if len(set(np.unique(expected)) & {BLACK, GREEN}) < 2:
        return False
    if np.array_equal(expected, bottom) or np.array_equal(expected, repaired_top):
        return False

    return True


def _generate_instance(rng: np.random.Generator) -> np.ndarray | None:
    row_period = int(rng.integers(2, 5))
    col_period = int(rng.integers(3, 7))

    row_repeats_max = 14 // row_period
    if row_repeats_max < 3:
        return None
    row_repeats = int(rng.integers(3, row_repeats_max + 1))
    h_base = row_period * row_repeats
    h_extra = int(rng.integers(0, min(row_period, 15 - h_base)))
    h = h_base + h_extra

    col_repeats_max = 30 // col_period
    if col_repeats_max < 3:
        return None
    col_repeats = int(rng.integers(3, min(6, col_repeats_max + 1)))
    w_base = col_period * col_repeats
    w_extra = int(rng.integers(0, min(col_period, 31 - w_base)))
    w = w_base + w_extra

    tile = _random_tile(rng, row_period, col_period)
    if tile is None:
        return None
    repaired_top = _repeat_tile(tile, h, w)
    if int(np.count_nonzero(repaired_top == RED)) < max(18, h * w // 5):
        return None

    bottom = _bottom_mask(rng, h, w)
    if not np.any((repaired_top == RED) & (bottom == RED)):
        return None
    if not np.any((repaired_top == RED) & (bottom == BLACK)):
        return None

    top = _erase_reds(rng, repaired_top, bottom, row_period, col_period)
    if top is None:
        return None

    inp = np.vstack(
        [
            top,
            np.full((1, w), YELLOW, dtype=int),
            bottom,
        ]
    )
    if not _validate_instance(inp, repaired_top, bottom, row_period, col_period):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
