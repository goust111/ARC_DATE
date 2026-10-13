import random
from collections import Counter

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _collapse_runs(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        value = int(value)
        if not out or value != out[-1]:
            out.append(value)
    return out


def _segment_kind(src: tuple[int, int], dst: tuple[int, int]) -> str:
    if int(src[0]) == int(dst[0]):
        return "H"
    if int(src[1]) == int(dst[1]):
        return "V"
    return "L"


def _path_cells(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = map(int, src)
    tr, tc = map(int, dst)

    cells: list[tuple[int, int]] = []
    if sc <= tc:
        cols = range(sc, tc + 1)
    else:
        cols = range(sc, tc - 1, -1)
    for cc in cols:
        cells.append((sr, int(cc)))

    if sr == tr:
        return cells

    if sr < tr:
        rows = range(sr + 1, tr + 1)
    else:
        rows = range(sr - 1, tr - 1, -1)
    for rr in rows:
        cells.append((int(rr), tc))
    return cells


def _find_divider(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, int]] = []

    for divider_c in range(1, width - 1):
        column = g[:, divider_c]
        if not bool(np.all(column == column[0])):
            continue

        left = g[:, :divider_c]
        right = g[:, divider_c + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        if not all(bool(np.all(left[:, c] == left[0, c])) for c in range(left.shape[1])):
            continue

        bg = _most_common_color(right)
        if int(column[0]) == int(bg):
            continue
        foreground = [(int(r), int(c)) for r, c in np.argwhere(right != bg)]
        if len(foreground) < 2:
            continue

        per_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in foreground:
            per_color.setdefault(int(right[r, c]), []).append((r, c))
        if any(len(coords) != 1 for coords in per_color.values()):
            continue

        candidates.append((int(divider_c), int(column[0]), int(bg)))

    assert len(candidates) == 1, "expected a unique divider column"
    return candidates[0]


def _extract_stations(
    right: np.ndarray,
    bg: int,
    *,
    col_offset: int,
) -> dict[int, tuple[int, int]]:
    stations: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(right != int(bg)):
        color = int(right[r, c])
        assert color not in stations, "each station color must appear exactly once"
        stations[color] = (int(r), int(c + col_offset))
    assert stations, "expected at least one station"
    return stations


def _render_output(
    grid: np.ndarray,
    sequence: list[int],
    stations: dict[int, tuple[int, int]],
    *,
    bg: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    station_cells = set(stations.values())

    for i in range(len(sequence) - 1):
        source_color = int(sequence[i])
        target_color = int(sequence[i + 1])
        src = stations[source_color]
        dst = stations[target_color]
        for r, c in _path_cells(src, dst):
            if (r, c) in station_cells:
                continue
            if int(out[r, c]) == int(bg):
                out[r, c] = source_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the left stripe widths into a visit order and draw same-source L trips between the matching right-side stations.

    Input structure:
    - A single full-height gray divider column splits the grid into a left stripe panel and a right black field.
    - Every left column belongs to a full-height vertical stripe; stripe thickness varies, but each stripe keeps one color throughout its width.
    - On the right field, each stripe color appears as exactly one single-pixel station.

    Rule:
    - Read the left stripe colors from left to right and collapse consecutive duplicate columns to recover the visit sequence.
    - Start at the station of the first color; for each consecutive pair, move first horizontally to the target column and then vertically to the target row, using a straight segment if the pair is already row- or column-aligned.
    - Color only empty traversed right-field cells with the departure station color, while preserving all stations, the divider, and the left stripe panel.

    Key insight:
    - Stripe thickness is a decoy: the actionable information is only the deduplicated color order, and each segment records where the trip starts rather than where it ends.
    """
    g = np.asarray(grid, dtype=int)
    divider_c, _divider_color, bg = _find_divider(g)
    sequence = _collapse_runs([int(g[0, c]) for c in range(divider_c)])
    right = g[:, divider_c + 1 :]
    stations = _extract_stations(right, bg, col_offset=divider_c + 1)

    assert set(sequence) == set(stations), "stations must match the stripe colors exactly"
    return _render_output(g, sequence, stations, bg=bg)


def _build_sequence(label_colors: dict[str, int]) -> list[int]:
    cycle = ["A", "B", "C", "D"]
    start = random.randrange(len(cycle))
    rotated = cycle[start:] + cycle[:start]
    rotated.append(rotated[0])
    return [int(label_colors[label]) for label in rotated]


def _sample_station_layout(height: int, width: int) -> dict[str, tuple[int, int]]:
    rd = random.randint(1, height - 5)
    ra = random.randint(rd + 2, height - 3)
    rc = random.randint(ra + 2, height - 1)

    cd = random.randint(1, width - 5)
    ca = random.randint(cd + 2, width - 3)
    cb = random.randint(ca + 2, width - 1)

    return {
        "A": (ra, ca),
        "B": (ra, cb),
        "C": (rc, cb),
        "D": (rd, cd),
    }


def _station_paths_ok(
    sequence: list[int],
    stations: dict[int, tuple[int, int]],
) -> bool:
    station_cells = set(stations.values())
    used_interior: set[tuple[int, int]] = set()
    kinds = set()

    for i in range(len(sequence) - 1):
        src = stations[int(sequence[i])]
        dst = stations[int(sequence[i + 1])]
        kinds.add(_segment_kind(src, dst))
        cells = _path_cells(src, dst)
        interior = set(cells) - {src, dst}
        if any(cell in station_cells for cell in interior):
            return False
        if used_interior & interior:
            return False
        used_interior |= interior

    return kinds == {"H", "V", "L"}


def _validate_generated_input(
    grid: np.ndarray,
    sequence: list[int],
    stations: dict[int, tuple[int, int]],
    stripe_widths: list[int],
) -> None:
    divider_c, divider_color, bg = _find_divider(grid)
    assert divider_color == DIVIDER
    assert bg == BG
    assert any(width > 1 for width in stripe_widths)
    assert any(width == 1 for width in stripe_widths)

    raw_left = [int(grid[0, c]) for c in range(divider_c)]
    parsed_sequence = _collapse_runs(raw_left)
    assert parsed_sequence == sequence
    assert len(raw_left) > len(parsed_sequence), "compression must matter"
    assert len(set(parsed_sequence)) == 4
    assert parsed_sequence[0] == parsed_sequence[-1]
    assert set(parsed_sequence) == set(stations)

    right = grid[:, divider_c + 1 :]
    parsed_stations = _extract_stations(right, bg, col_offset=divider_c + 1)
    assert parsed_stations == stations
    assert _station_paths_ok(sequence, stations)

    expected = _render_output(grid, sequence, stations, bg=bg)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert int(np.sum(actual != grid)) > 0


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(8, 13)
        right_width = random.randint(8, 12)

        stripe_widths = [random.randint(1, 3) for _ in range(5)]
        if not any(width > 1 for width in stripe_widths):
            continue
        if not any(width == 1 for width in stripe_widths):
            continue

        left_width = sum(stripe_widths)
        total_width = left_width + 1 + right_width
        if total_width > 30:
            continue

        label_colors = dict(zip(["A", "B", "C", "D"], random.sample(PALETTE, 4)))
        sequence = _build_sequence(label_colors)
        stations = {
            int(label_colors[label]): position
            for label, position in _sample_station_layout(height, right_width).items()
        }
        if not _station_paths_ok(sequence, stations):
            continue

        grid = np.full((height, total_width), BG, dtype=int)
        c0 = 0
        for color, width in zip(sequence, stripe_widths):
            grid[:, c0 : c0 + width] = int(color)
            c0 += width
        grid[:, left_width] = DIVIDER

        for color, (r, c) in stations.items():
            grid[int(r), int(left_width + 1 + c)] = int(color)

        global_stations = {
            int(color): (int(r), int(left_width + 1 + c))
            for color, (r, c) in stations.items()
        }
        _validate_generated_input(grid, sequence, global_stations, stripe_widths)
        return grid

    raise RuntimeError("failed to generate a valid striped-itinerary input")
