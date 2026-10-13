from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 7
PATTERN_COLORS = (1, 2, 3)
GEN_RETRIES = 400

DIRS = (
    ("up", (-1, 0)),
    ("right", (0, 1)),
    ("down", (1, 0)),
    ("left", (0, -1)),
)


def _visible_tile_for_period(
    grid: np.ndarray, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    tile: dict[tuple[int, int], int] = {}
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color in (BACKGROUND, MARKER):
                continue
            key = (r % row_period, c % col_period)
            prev = tile.get(key)
            if prev is not None and prev != color:
                return None
            tile[key] = color
    return tile


def _infer_period(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int]]:
    height, width = grid.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            tile = _visible_tile_for_period(grid, row_period, col_period)
            if tile is not None:
                candidates.append((row_period, col_period, tile))

    assert candidates, "expected at least one consistent period"
    min_area = min(row_period * col_period for row_period, col_period, _ in candidates)
    best = [
        candidate
        for candidate in candidates
        if candidate[0] * candidate[1] == min_area
    ]
    assert len(best) == 1, "expected a unique smallest period"
    return best[0]


def _restore_pattern(grid: np.ndarray) -> np.ndarray:
    row_period, col_period, tile = _infer_period(grid)
    restored = np.zeros_like(grid)
    for (rr, cc), color in tile.items():
        restored[rr::row_period, cc::col_period] = color
    return restored


def _marker_choice(shape: tuple[int, int], r: int, c: int) -> tuple[str, tuple[int, int], list[tuple[int, int]], bool]:
    height, width = shape
    options = (
        ("up", r, (-1, 0)),
        ("right", width - 1 - c, (0, 1)),
        ("down", height - 1 - r, (1, 0)),
        ("left", c, (0, -1)),
    )
    min_dist = min(dist for _, dist, _ in options)
    ties = sum(dist == min_dist for _, dist, _ in options) > 1
    name, _dist, (dr, dc) = next(item for item in options if item[1] == min_dist)

    path: list[tuple[int, int]] = []
    rr, cc = r, c
    while 0 <= rr < height and 0 <= cc < width:
        path.append((rr, cc))
        rr += dr
        cc += dc
    return name, (dr, dc), path, ties


def _paint_marker_rays(restored: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = restored.copy()
    occupied: set[tuple[int, int]] = set()
    for r, c in sorted(markers):
        _name, _direction, path, _ties = _marker_choice(out.shape, r, c)
        for cell in path:
            assert cell not in occupied, "marker rays must not overlap"
            occupied.add(cell)
            out[cell] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the periodic wallpaper, then shoot edge-directed orange rays from the marker pixels.

    Input structure
    - The grid contains a repeating wallpaper built from colors 0, 1, 2, and 3, with a smallest tile of at least 3x3.
    - Some non-zero wallpaper cells are erased to black 0, and 2-3 isolated orange 7 markers replace original non-zero wallpaper cells.
    - Marker rays are determined only by the marker positions, not by the wallpaper colors.

    Rule
    - Ignore black 0 erasures and orange 7 markers, infer the unique smallest row and column period consistent with the visible non-zero non-marker cells, and restore the full wallpaper.
    - For each orange marker, compare its distances to the top, right, bottom, and left grid edges; pick the smallest distance, breaking ties by the clockwise priority up > right > down > left.
    - Starting at the marker itself, paint a straight ray in that chosen direction until the grid boundary, overwriting the restored wallpaper with orange 7.

    Key insight
    - The orange pixels are disguised replacements for missing wallpaper cells, so the periodic wallpaper must be reconstructed first; only then can the markers be recognized as control signals that emit rays based on absolute position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    restored = _restore_pattern(g)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    assert 2 <= len(markers) <= 3, "expected 2-3 markers"
    return _paint_marker_rays(restored, markers)


def _wallpaper_from_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols]


def _sample_tile(row_period: int, col_period: int) -> np.ndarray | None:
    area = row_period * col_period
    zero_count = random.randint(max(1, area // 8), max(1, area // 3))
    values = [BACKGROUND] * zero_count
    for color in PATTERN_COLORS:
        values.extend([color, color])
    while len(values) < area:
        values.append(random.choice(PATTERN_COLORS))
    random.shuffle(values)
    tile = np.array(values[:area], dtype=int).reshape(row_period, col_period)
    if not all(np.any(tile == color) for color in PATTERN_COLORS):
        return None
    if not np.any(tile == BACKGROUND):
        return None
    return tile


def _pattern_residue_map(tile: np.ndarray) -> dict[tuple[int, int], int]:
    return {
        (r, c): int(tile[r, c])
        for r in range(tile.shape[0])
        for c in range(tile.shape[1])
        if int(tile[r, c]) != BACKGROUND
    }


def _choose_markers(
    full: np.ndarray, row_period: int, col_period: int
) -> list[tuple[int, int]] | None:
    candidates_by_dir: dict[str, list[tuple[int, int, bool, set[tuple[int, int]]]]] = {
        name: [] for name, _ in DIRS
    }
    height, width = full.shape
    for r in range(height):
        for c in range(width):
            if int(full[r, c]) == BACKGROUND:
                continue
            name, _direction, path, ties = _marker_choice(full.shape, r, c)
            if len(path) < 2:
                continue
            candidates_by_dir[name].append((r, c, ties, set(path)))

    available_dirs = [name for name, items in candidates_by_dir.items() if items]
    if len(available_dirs) < 2:
        return None

    count = random.randint(2, min(3, len(available_dirs)))
    require_tie = random.random() < 0.5
    random.shuffle(available_dirs)
    wanted_dirs = available_dirs[:count]

    def backtrack(
        index: int,
        chosen: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
        used_tie: bool,
    ) -> list[tuple[int, int]] | None:
        if index == len(wanted_dirs):
            return chosen if (used_tie or not require_tie) else None

        options = candidates_by_dir[wanted_dirs[index]][:]
        random.shuffle(options)
        for r, c, ties, path in options[:80]:
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
                continue
            if occupied & path:
                continue
            result = backtrack(
                index + 1,
                chosen + [(r, c)],
                occupied | path,
                used_tie or ties,
            )
            if result is not None:
                return result
        return None

    return backtrack(0, [], set(), False)


def _choose_erasure_rectangles(
    full: np.ndarray, markers: set[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    height, width = full.shape
    rects: list[tuple[int, int, int, int]] = []
    blocked = np.zeros_like(full, dtype=bool)
    target_count = random.randint(2, 4)
    for _ in range(target_count):
        choices = []
        for rh in range(2, 5):
            for rw in range(2, 5):
                for r0 in range(0, height - rh + 1):
                    for c0 in range(0, width - rw + 1):
                        if blocked[max(0, r0 - 1) : min(height, r0 + rh + 1), max(0, c0 - 1) : min(width, c0 + rw + 1)].any():
                            continue
                        cells = [
                            (r, c)
                            for r in range(r0, r0 + rh)
                            for c in range(c0, c0 + rw)
                            if int(full[r, c]) != BACKGROUND and (r, c) not in markers
                        ]
                        if len(cells) >= 2:
                            choices.append((r0, c0, rh, rw))
        if not choices:
            break
        rect = random.choice(choices)
        rects.append(rect)
        r0, c0, rh, rw = rect
        blocked[max(0, r0 - 1) : min(height, r0 + rh + 1), max(0, c0 - 1) : min(width, c0 + rw + 1)] = True
    return rects if len(rects) >= 2 else None


def _apply_erasures(
    full: np.ndarray,
    row_period: int,
    col_period: int,
    markers: list[tuple[int, int]],
) -> np.ndarray | None:
    erased = full.copy()
    marker_set = set(markers)
    rects = _choose_erasure_rectangles(full, marker_set)
    if rects is None:
        return None

    for r0, c0, rh, rw in rects:
        patch = erased[r0 : r0 + rh, c0 : c0 + rw]
        patch[(patch != BACKGROUND)] = BACKGROUND

    total_erased = int(np.count_nonzero((full != BACKGROUND) & (erased == BACKGROUND)))
    total_pattern = int(np.count_nonzero(full != BACKGROUND))
    if not (max(6, total_pattern // 12) <= total_erased <= max(8, total_pattern // 3)):
        return None

    residue_colors = _pattern_residue_map(full[:row_period, :col_period])
    visible_by_residue: dict[tuple[int, int], int] = {key: 0 for key in residue_colors}
    height, width = full.shape
    for r in range(height):
        for c in range(width):
            key = (r % row_period, c % col_period)
            if key not in residue_colors:
                continue
            if int(erased[r, c]) == residue_colors[key]:
                visible_by_residue[key] += 1

    if any(count == 0 for count in visible_by_residue.values()):
        return None
    for r, c in markers:
        if visible_by_residue[(r % row_period, c % col_period)] < 2:
            return None

    return erased


def _make_candidate() -> np.ndarray | None:
    height = random.randint(12, 18)
    width = random.randint(12, 18)
    row_period = random.randint(3, min(6, height // 2))
    col_period = random.randint(3, min(6, width // 2))

    tile = _sample_tile(row_period, col_period)
    if tile is None:
        return None
    full = _wallpaper_from_tile(tile, height, width)

    try:
        inferred_row_period, inferred_col_period, _ = _infer_period(full)
    except AssertionError:
        return None
    if (inferred_row_period, inferred_col_period) != (row_period, col_period):
        return None

    markers = _choose_markers(full, row_period, col_period)
    if markers is None:
        return None

    erased = _apply_erasures(full, row_period, col_period, markers)
    if erased is None:
        return None

    inp = erased.copy()
    for r, c in markers:
        inp[r, c] = MARKER

    expected = _paint_marker_rays(full, markers)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
