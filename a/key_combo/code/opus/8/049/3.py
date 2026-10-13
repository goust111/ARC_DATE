import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
RED = 2
PERIOD_RANGE = range(3, 6)
PERIODIC_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 500


def _color_counts(grid: np.ndarray) -> Counter:
    return Counter(map(int, grid.ravel()))


def _periodic_palette(grid: np.ndarray) -> set[int]:
    counts = _color_counts(grid)
    palette = {color for color, count in counts.items() if color not in {BLACK, RED} and count >= 3}
    assert len(palette) >= 3, "need at least three reliable periodic colors"
    return palette


def _axis_period(grid: np.ndarray, palette: set[int], axis: int) -> int:
    support = np.isin(grid, list(palette))
    limit = grid.shape[axis]
    for period in range(1, limit):
        if axis == 0:
            left = grid[:-period, :]
            right = grid[period:, :]
            mask = support[:-period, :] & support[period:, :]
        else:
            left = grid[:, :-period]
            right = grid[:, period:]
            mask = support[:, :-period] & support[:, period:]
        if int(mask.sum()) < max(8, (left.size // 6)):
            continue
        if np.all(left[mask] == right[mask]):
            return period
    raise AssertionError(f"no valid period found on axis {axis}")


def _same_residue_coords(shape: tuple[int, int], pr: int, pc: int, r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    rr0 = r % pr
    cc0 = c % pc
    return [(rr, cc) for rr in range(rr0, h, pr) for cc in range(cc0, w, pc)]


def _repair_grid(grid: np.ndarray, pr: int, pc: int, palette: set[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r, c in np.argwhere(g == BLACK):
        orbit_colors = {
            int(g[rr, cc])
            for rr, cc in _same_residue_coords(g.shape, pr, pc, int(r), int(c))
            if int(g[rr, cc]) != BLACK
        }
        assert orbit_colors, "every black cell must have visible orbit witnesses"
        outliers = orbit_colors - palette
        periodic = orbit_colors & palette
        if outliers:
            assert outliers == {RED}, "only red may appear outside the periodic palette"
            out[int(r), int(c)] = RED
        else:
            assert len(periodic) == 1, "ordinary occlusions must resolve to one periodic color"
            out[int(r), int(c)] = next(iter(periodic))
    return out


def _marker_positions(grid: np.ndarray) -> np.ndarray:
    markers = np.argwhere(grid == RED)
    assert markers.shape == (2, 2), "repaired grid must contain exactly two red markers"
    return markers


def _crop_on_markers(grid: np.ndarray) -> np.ndarray:
    markers = _marker_positions(grid)
    (r0, c0), (r1, c1) = [tuple(map(int, rc)) for rc in markers]
    h, w = grid.shape
    out = None
    if r0 == r1:
        mid = w // 2
        same_left = c0 < mid and c1 < mid
        same_right = c0 >= mid and c1 >= mid
        assert same_left ^ same_right, "row-aligned markers must lie in one horizontal half"
        out = grid[:, :mid].copy() if same_left else grid[:, mid:].copy()
    elif c0 == c1:
        mid = h // 2
        same_top = r0 < mid and r1 < mid
        same_bottom = r0 >= mid and r1 >= mid
        assert same_top ^ same_bottom, "column-aligned markers must lie in one vertical half"
        out = grid[:mid, :].copy() if same_top else grid[mid:, :].copy()
    else:
        raise AssertionError("markers must share a row or a column")
    out[out == RED] = BLACK
    return out


def _infer_periods(grid: np.ndarray) -> tuple[int, int, set[int]]:
    palette = _periodic_palette(grid)
    pr = _axis_period(grid, palette, axis=0)
    pc = _axis_period(grid, palette, axis=1)
    assert pr in PERIOD_RANGE and pc in PERIOD_RANGE, "periods must stay in the intended ARC range"
    return pr, pc, palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a periodic image, reveal the hidden red crop marker, then crop the marked half and erase the markers.

    Input structure:
    - The grid is a mostly periodic colored tiling with rectangular row/column periods between 3 and 5.
    - Black cells are occlusions, not background; one red marker is visible and its partner is hidden under black.
    - The two red markers belong to one period residue class, share one row or one column, and lie in the same half along that direction.

    Rule:
    - Infer the smallest vertical and horizontal periods from the repeated non-black, non-marker colors.
    - For each black cell, inspect all visible cells in the same period residue class.
    - If that class shows a non-periodic visible color, fill the black cell with that outlier color; otherwise fill it with the class's unique periodic color.
    - After all occlusions are repaired, find the two red markers, crop the left/right or top/bottom half that contains them, and erase the red markers back to black.

    Key insight:
    - Periodic residue classes recover ordinary missing cells, and the single visible red outlier propagates through its class to expose the second crop marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    pr, pc, palette = _infer_periods(g)
    repaired = _repair_grid(g, pr, pc, palette)
    return _crop_on_markers(repaired)


def _has_smaller_axis_period(tile: np.ndarray, axis: int) -> bool:
    limit = tile.shape[axis]
    for period in range(1, limit):
        if axis == 0:
            if np.all(tile[:-period, :] == tile[period:, :]):
                return True
        else:
            if np.all(tile[:, :-period] == tile[:, period:]):
                return True
    return False


def _random_tile(pr: int, pc: int, palette: tuple[int, ...]) -> np.ndarray:
    for _ in range(200):
        tile = np.array([[random.choice(palette) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if set(map(int, tile.ravel())) != set(palette):
            continue
        if _has_smaller_axis_period(tile, 0) or _has_smaller_axis_period(tile, 1):
            continue
        if any(len(set(map(int, tile[r, :]))) == 1 for r in range(pr)):
            continue
        if any(len(set(map(int, tile[:, c]))) == 1 for c in range(pc)):
            continue
        return tile
    raise RuntimeError("failed to generate a non-degenerate tile")


def _tile_to_shape(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    pr, pc = tile.shape
    rr = np.arange(h) % pr
    cc = np.arange(w) % pc
    return tile[rr[:, None], cc[None, :]].copy()


def _marker_pair_candidates(shape: tuple[int, int], pr: int, pc: int, mode: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if mode == "row":
        mid = w // 2
        for r in range(h):
            for c in range(0, w - pc):
                c2 = c + pc
                if not ((c < mid and c2 < mid) or (c >= mid and c2 >= mid)):
                    continue
                if len(_same_residue_coords(shape, pr, pc, r, c)) < 3:
                    continue
                candidates.append(((r, c), (r, c2)))
    else:
        mid = h // 2
        for r in range(0, h - pr):
            r2 = r + pr
            for c in range(w):
                if not ((r < mid and r2 < mid) or (r >= mid and r2 >= mid)):
                    continue
                if len(_same_residue_coords(shape, pr, pc, r, c)) < 3:
                    continue
                candidates.append(((r, c), (r2, c)))
    return candidates


def _keep_half_signature(grid: np.ndarray, markers: tuple[tuple[int, int], tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    marked = grid.copy()
    for r, c in markers:
        marked[r, c] = RED
    out = _crop_on_markers(marked)
    h, w = marked.shape
    (r0, c0), (r1, c1) = markers
    if r0 == r1:
        mid = w // 2
        discarded = marked[:, mid:].copy() if c0 < mid and c1 < mid else marked[:, :mid].copy()
    else:
        mid = h // 2
        discarded = marked[mid:, :].copy() if r0 < mid and r1 < mid else marked[:mid, :].copy()
    discarded[discarded == RED] = BLACK
    return out, discarded


def _pick_markers(shape: tuple[int, int], pr: int, pc: int, mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates = _marker_pair_candidates(shape, pr, pc, mode)
    random.shuffle(candidates)
    for pair in candidates:
        keep, discard = _keep_half_signature(np.zeros(shape, dtype=int), pair)
        if keep.shape == discard.shape and np.array_equal(keep, discard):
            continue
        return pair
    raise RuntimeError("failed to place marker pair")


def _class_key(r: int, c: int, pr: int, pc: int) -> tuple[int, int]:
    return (r % pr, c % pc)


def _choose_extra_blacks(
    solved: np.ndarray,
    pr: int,
    pc: int,
    marker_pair: tuple[tuple[int, int], tuple[int, int]],
    hidden_marker: tuple[int, int],
    total_black: int,
) -> list[tuple[int, int]]:
    h, w = solved.shape
    marker_class = _class_key(hidden_marker[0], hidden_marker[1], pr, pc)
    class_sizes: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            key = _class_key(r, c, pr, pc)
            class_sizes[key] = class_sizes.get(key, 0) + 1

    used = {hidden_marker}
    black_per_class = defaultdict(int)
    black_per_class[marker_class] = 1
    extras: list[tuple[int, int]] = []
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    marker_cells = set(marker_pair)
    for cell in candidates:
        if len(extras) >= total_black - 1:
            break
        if cell in used or cell in marker_cells:
            continue
        key = _class_key(cell[0], cell[1], pr, pc)
        if key == marker_class:
            continue
        if black_per_class[key] >= class_sizes[key] - 1:
            continue
        used.add(cell)
        extras.append(cell)
        black_per_class[key] += 1
    if len(extras) != total_black - 1:
        raise RuntimeError("could not place enough extra occlusions")
    return extras


def _expected_output_from_repaired(repaired: np.ndarray) -> np.ndarray:
    return _crop_on_markers(repaired)


def _validate_generated(grid: np.ndarray, expected: np.ndarray, pr: int, pc: int, mode: str) -> bool:
    counts = _color_counts(grid)
    if counts[BLACK] < 6 or counts[BLACK] > 12 or counts[RED] != 1:
        return False
    try:
        got = main(grid)
        inferred_pr, inferred_pc, palette = _infer_periods(grid)
        repaired = _repair_grid(grid, inferred_pr, inferred_pc, palette)
        markers = _marker_positions(repaired)
    except AssertionError:
        return False
    if inferred_pr != pr or inferred_pc != pc:
        return False
    if not np.array_equal(got, expected):
        return False
    if repaired.shape[0] > 30 or repaired.shape[1] > 30:
        return False
    if markers[0, 0] == markers[1, 0]:
        if mode != "row":
            return False
    elif markers[0, 1] == markers[1, 1]:
        if mode != "col":
            return False
    else:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["row", "col"])
    pr = random.choice(tuple(PERIOD_RANGE))
    pc = random.choice(tuple(PERIOD_RANGE))
    palette = tuple(random.sample(PERIODIC_COLORS, k=random.choice((3, 4))))

    rep_r = random.randint(3, 4) if mode == "col" else random.randint(2, 4)
    rep_c = random.randint(3, 4) if mode == "row" else random.randint(2, 4)
    extra_r = random.randint(0, pr - 1)
    extra_c = random.randint(0, pc - 1)
    h = pr * rep_r + extra_r
    w = pc * rep_c + extra_c
    if not (9 <= h <= 18 and 9 <= w <= 18):
        return None

    tile = _random_tile(pr, pc, palette)
    solved = _tile_to_shape(tile, h, w)

    candidates = _marker_pair_candidates((h, w), pr, pc, mode)
    random.shuffle(candidates)
    marker_pair = None
    for pair in candidates:
        tmp = solved.copy()
        tmp[pair[0]] = RED
        tmp[pair[1]] = RED
        keep, discard = _keep_half_signature(tmp, pair)
        if keep.shape == discard.shape and np.array_equal(keep, discard):
            continue
        marker_pair = pair
        break
    if marker_pair is None:
        return None

    repaired = solved.copy()
    repaired[marker_pair[0]] = RED
    repaired[marker_pair[1]] = RED

    grid = repaired.copy()
    hidden_marker = random.choice(marker_pair)
    grid[hidden_marker] = BLACK
    total_black = random.randint(6, 12)
    try:
        for r, c in _choose_extra_blacks(repaired, pr, pc, marker_pair, hidden_marker, total_black):
            grid[r, c] = BLACK
    except RuntimeError:
        return None

    visible_palette_counts = [count for color, count in _color_counts(grid).items() if color not in {BLACK, RED}]
    if min(visible_palette_counts, default=0) < 3:
        return None

    expected = _expected_output_from_repaired(repaired)
    if not _validate_generated(grid, expected, pr, pc, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after many attempts")
