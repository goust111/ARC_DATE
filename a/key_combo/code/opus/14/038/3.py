import random
from itertools import product

import numpy as np


MARKER = 5
LEFT_RECOLOR = 4
MID_RECOLOR = 7
RIGHT_RECOLOR = 6
WALLPAPER_COLORS = (1, 2, 3)
ZONE_NAMES = ("left", "mid", "right")
MAX_GENERATION_ATTEMPTS = 600


def _foreground_tile_for_period(
    grid: np.ndarray, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    tile: dict[tuple[int, int], int] = {}
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color in (0, MARKER):
                continue
            key = (r % row_period, c % col_period)
            prev = tile.get(key)
            if prev is not None and prev != color:
                return None
            tile[key] = color
    return tile


def _infer_unique_period_and_tile(
    grid: np.ndarray,
) -> tuple[int, int, dict[tuple[int, int], int]]:
    height, width = grid.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            tile = _foreground_tile_for_period(grid, row_period, col_period)
            if tile:
                candidates.append((row_period, col_period, tile))

    assert candidates, "expected at least one valid period candidate"
    min_area = min(row_period * col_period for row_period, col_period, _ in candidates)
    best = [
        candidate
        for candidate in candidates
        if candidate[0] * candidate[1] == min_area
    ]
    assert len(best) == 1, "expected a unique smallest period"
    return best[0]


def _zone_from_col(col: int, width: int) -> str:
    cut1 = width // 3
    cut2 = (2 * width) // 3
    if col < cut1:
        return "left"
    if col < cut2:
        return "mid"
    return "right"


def _recolor_for_zone(zone: str) -> int:
    if zone == "left":
        return LEFT_RECOLOR
    if zone == "mid":
        return MID_RECOLOR
    return RIGHT_RECOLOR


def _apply_rule_with_trace(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int],
    dict[tuple[int, int], int],
    dict[int, tuple[int, str]],
]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    row_period, col_period, tile = _infer_unique_period_and_tile(g)
    out = g.copy()
    completed = np.zeros_like(g, dtype=bool)

    for r in range(height):
        for c in range(width):
            if int(g[r, c]) == 0:
                key = (r % row_period, c % col_period)
                if key in tile:
                    out[r, c] = tile[key]
                    completed[r, c] = True

    row_markers: dict[int, tuple[int, str]] = {}
    for r in range(height):
        marker_cols = np.flatnonzero(g[r] == MARKER)
        assert len(marker_cols) <= 1, "expected at most one marker per row"
        if len(marker_cols) == 0:
            continue
        marker_col = int(marker_cols[0])
        zone = _zone_from_col(marker_col, width)
        row_markers[r] = (marker_col, zone)
        out[r, completed[r]] = _recolor_for_zone(zone)
        out[r, marker_col] = MARKER

    return out, completed, (row_period, col_period), tile, row_markers


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the periodic wallpaper, then selectively recolor only the restored pixels using row markers.

    Input structure:
    - The grid contains a periodic wallpaper made from black background cells and 2-3 foreground colors.
    - Some foreground wallpaper cells have been erased to black.
    - Grey marker pixels may appear on black wallpaper holes, with at most one marker per row.

    Rule:
    - Ignore grey markers and infer the unique smallest row/column period whose residue classes assign every visible
      non-black wallpaper pixel one consistent color.
    - Restore every black cell that belongs to an observed foreground residue class, and record exactly which cells
      were restored by that completion step.
    - For each row with a grey marker, read the marker column zone: left third -> yellow, middle third -> orange,
      right third -> magenta. Recolor only that row's restored cells to the zone color; surviving original wallpaper
      pixels stay unchanged, and marker pixels remain grey.

    Key insight:
    - The important hidden intermediate result is not just the completed wallpaper, but the set of cells that had to
      be restored; the marker controls act only on those restored cells, not on the whole row.
    """
    out, _, _, _, _ = _apply_rule_with_trace(grid)
    return out


def _full_tile_min_period(tile: np.ndarray) -> tuple[int, int] | None:
    rows, cols = tile.shape
    candidates: list[tuple[int, int]] = []
    for row_period in range(1, rows + 1):
        for col_period in range(1, cols + 1):
            ok = True
            for r in range(rows):
                for c in range(cols):
                    if int(tile[r, c]) != int(tile[r % row_period, c % col_period]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((row_period, col_period))
    if not candidates:
        return None
    min_area = min(rp * cp for rp, cp in candidates)
    best = [candidate for candidate in candidates if candidate[0] * candidate[1] == min_area]
    if len(best) != 1:
        return None
    return best[0]


def _build_wallpaper(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    row_idx = np.arange(height)[:, None] % tile.shape[0]
    col_idx = np.arange(width)[None, :] % tile.shape[1]
    return tile[row_idx, col_idx].astype(int)


def _tile_foreground_map(tile: np.ndarray) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for r, c in product(range(tile.shape[0]), range(tile.shape[1])):
        color = int(tile[r, c])
        if color != 0:
            out[(r, c)] = color
    return out


def _generate_tile(row_period: int, col_period: int) -> np.ndarray | None:
    for _ in range(200):
        fg_count = random.choice((2, 3))
        fg_colors = random.sample(WALLPAPER_COLORS, fg_count)
        zero_prob = random.uniform(0.28, 0.45)
        tile = np.zeros((row_period, col_period), dtype=int)
        for r in range(row_period):
            for c in range(col_period):
                if random.random() < zero_prob:
                    tile[r, c] = 0
                else:
                    tile[r, c] = random.choice(fg_colors)

        if any(np.all(tile[r] == 0) for r in range(row_period)):
            continue
        if any(not np.any(tile[r] == 0) for r in range(row_period)):
            continue
        if any(np.all(tile[:, c] == 0) for c in range(col_period)):
            continue
        if any(not np.any(tile[r] != 0) for r in range(row_period)):
            continue
        if any(not np.any(tile[:, c] != 0) for c in range(col_period)):
            continue
        if any(not np.any(tile == color) for color in fg_colors):
            continue
        if int(np.count_nonzero(tile == 0)) < row_period:
            continue
        if _full_tile_min_period(tile) != (row_period, col_period):
            continue
        return tile
    return None


def _pick_distinct_zone_rows(
    zero_by_row_zone: dict[int, dict[str, list[int]]], fg_by_row: dict[int, list[int]]
) -> dict[str, int] | None:
    rows = list(fg_by_row)
    for _ in range(200):
        used: set[int] = set()
        picked: dict[str, int] = {}
        zones = list(ZONE_NAMES)
        random.shuffle(zones)
        ok = True
        for zone in zones:
            candidates = [
                row
                for row in rows
                if row not in used
                and len(fg_by_row[row]) >= 2
                and zero_by_row_zone[row][zone]
            ]
            if not candidates:
                ok = False
                break
            row = random.choice(candidates)
            used.add(row)
            picked[zone] = row
        if ok:
            return picked
    return None


def _safe_row_erasure_limit(row: int, row_fg_total: dict[int, int], row_erased: dict[int, int]) -> bool:
    return row_erased[row] < row_fg_total[row] - 1


def _try_erase(
    row: int,
    col: int,
    erased: set[tuple[int, int]],
    visible_by_residue: dict[tuple[int, int], int],
    row_erased: dict[int, int],
    row_fg_total: dict[int, int],
    row_period: int,
    col_period: int,
) -> bool:
    if (row, col) in erased:
        return False
    residue = (row % row_period, col % col_period)
    if visible_by_residue[residue] <= 1:
        return False
    if not _safe_row_erasure_limit(row, row_fg_total, row_erased):
        return False
    erased.add((row, col))
    visible_by_residue[residue] -= 1
    row_erased[row] += 1
    return True


def _construct_expected(
    wallpaper: np.ndarray,
    erased: set[tuple[int, int]],
    markers: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = wallpaper.copy()
    pure_completion = wallpaper.copy()
    naive_row_fill = wallpaper.copy()
    for row, col in markers.items():
        pure_completion[row, col] = MARKER
        expected[row, col] = MARKER
        naive_row_fill[row, col] = MARKER

    for row, marker_col in markers.items():
        zone = _zone_from_col(marker_col, wallpaper.shape[1])
        recolor = _recolor_for_zone(zone)
        row_erased_cols = [col for r, col in erased if r == row]
        for col in row_erased_cols:
            expected[row, col] = recolor
        naive_mask = naive_row_fill[row] != 0
        naive_mask[marker_col] = False
        naive_row_fill[row, naive_mask] = recolor

    return expected, pure_completion, naive_row_fill


def _attempt_instance() -> np.ndarray | None:
    height = random.randint(10, 14)
    width = random.randint(10, 14)
    row_period = random.randint(2, 4)
    col_period = random.randint(2, 4)

    tile = _generate_tile(row_period, col_period)
    if tile is None:
        return None

    wallpaper = _build_wallpaper(tile, height, width)
    fg_map = _tile_foreground_map(tile)
    if len(fg_map) < 4:
        return None

    fg_by_row = {
        r: [int(c) for c in np.flatnonzero(wallpaper[r] != 0)]
        for r in range(height)
    }
    zero_by_row_zone = {
        r: {zone: [] for zone in ZONE_NAMES}
        for r in range(height)
    }
    for r in range(height):
        for c in np.flatnonzero(wallpaper[r] == 0):
            zone = _zone_from_col(int(c), width)
            zero_by_row_zone[r][zone].append(int(c))

    mandatory_rows = _pick_distinct_zone_rows(zero_by_row_zone, fg_by_row)
    if mandatory_rows is None:
        return None

    residue_positions: dict[tuple[int, int], list[tuple[int, int]]] = {
        residue: [] for residue in fg_map
    }
    for r in range(height):
        for c in fg_by_row[r]:
            residue_positions[(r % row_period, c % col_period)].append((r, c))
    visible_by_residue = {
        residue: len(coords) for residue, coords in residue_positions.items()
    }
    if any(count <= 1 for count in visible_by_residue.values()):
        return None

    row_fg_total = {row: len(cols) for row, cols in fg_by_row.items()}
    row_erased = {row: 0 for row in range(height)}
    erased: set[tuple[int, int]] = set()
    markers: dict[int, int] = {}

    for zone, row in mandatory_rows.items():
        candidates = fg_by_row[row][:]
        random.shuffle(candidates)
        wanted = 2 if len(candidates) >= 4 and random.random() < 0.7 else 1
        erased_here = 0
        for col in candidates:
            if _try_erase(
                row,
                col,
                erased,
                visible_by_residue,
                row_erased,
                row_fg_total,
                row_period,
                col_period,
            ):
                erased_here += 1
                if erased_here >= wanted:
                    break
        if erased_here == 0:
            return None
        markers[row] = random.choice(zero_by_row_zone[row][zone])

    unmarked_rows = [row for row in range(height) if row not in markers]
    random.shuffle(unmarked_rows)
    unmarked_effect_rows = 0
    for row in unmarked_rows:
        candidates = fg_by_row[row][:]
        random.shuffle(candidates)
        for col in candidates:
            if _try_erase(
                row,
                col,
                erased,
                visible_by_residue,
                row_erased,
                row_fg_total,
                row_period,
                col_period,
            ):
                unmarked_effect_rows += 1
                break
        if unmarked_effect_rows >= 1:
            break
    if unmarked_effect_rows == 0:
        return None

    min_erased = max(5, len(fg_map) + 1)
    max_erased = min(int(np.count_nonzero(wallpaper) * 0.18), 16)
    if min_erased > max_erased:
        return None
    target_erased = random.randint(min_erased, max_erased)
    all_fg_positions = [(r, c) for r in range(height) for c in fg_by_row[r]]
    random.shuffle(all_fg_positions)
    for row, col in all_fg_positions:
        if len(erased) >= target_erased:
            break
        _try_erase(
            row,
            col,
            erased,
            visible_by_residue,
            row_erased,
            row_fg_total,
            row_period,
            col_period,
        )

    if random.random() < 0.45:
        candidate_rows = [
            row
            for row in range(height)
            if row not in markers and any(zero_by_row_zone[row][zone] for zone in ZONE_NAMES)
        ]
        if candidate_rows:
            row = random.choice(candidate_rows)
            usable_zones = [zone for zone in ZONE_NAMES if zero_by_row_zone[row][zone]]
            zone = random.choice(usable_zones)
            markers[row] = random.choice(zero_by_row_zone[row][zone])

    if len(erased) < 4 or len(markers) < 3:
        return None

    inp = wallpaper.copy()
    erased_mask = np.zeros_like(inp, dtype=bool)
    for row, col in erased:
        inp[row, col] = 0
        erased_mask[row, col] = True
    for row, col in markers.items():
        assert int(wallpaper[row, col]) == 0
        inp[row, col] = MARKER

    expected, pure_completion, naive_row_fill = _construct_expected(
        wallpaper, erased, markers
    )

    try:
        actual, completed_mask, inferred_period, inferred_tile, row_markers = _apply_rule_with_trace(inp)
    except AssertionError:
        return None

    if inferred_period != (row_period, col_period):
        return None
    if inferred_tile != fg_map:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(completed_mask, erased_mask):
        return None
    if np.array_equal(actual, pure_completion):
        return None
    if np.array_equal(actual, naive_row_fill):
        return None

    marker_effect_rows = [row for row in markers if row_erased[row] > 0]
    if len(marker_effect_rows) < 3:
        return None
    if {_zone_from_col(markers[row], width) for row in marker_effect_rows} != set(ZONE_NAMES):
        return None
    if not any(row not in markers and row_erased[row] > 0 for row in range(height)):
        return None
    if not all(row_erased[row] < row_fg_total[row] for row in marker_effect_rows):
        return None
    if not any(
        row_erased[row] > 0 and row not in markers and np.any(actual[row] == wallpaper[row])
        for row in range(height)
    ):
        return None
    if not any(
        row in markers and row_erased[row] > 0 and np.any((wallpaper[row] != 0) & (actual[row] == wallpaper[row]))
        for row in range(height)
    ):
        return None
    if set(row_markers) != set(markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
