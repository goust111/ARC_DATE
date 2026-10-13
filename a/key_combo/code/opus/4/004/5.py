from __future__ import annotations

from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PAIR_COLORS = (BLUE, RED, GREEN)

MIN_TRACK_LEN = 7
MAX_TRACKS = 6
MAX_OCCUPIED_TRACKS = 4
MAX_GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _diag_cells(height: int, width: int, diag_offset: int) -> list[tuple[int, int]]:
    """Return the full NW-SE diagonal with constant (c - r) == diag_offset."""
    cells: list[tuple[int, int]] = []
    r = max(0, -diag_offset)
    c = r + diag_offset
    while 0 <= r < height and 0 <= c < width:
        cells.append((int(r), int(c)))
        r += 1
        c += 1
    return cells


def _extract_tracks(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Group non-black cells into contiguous NW-SE diagonal runs."""
    coords = np.argwhere(grid != BLACK)
    by_diag: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rr, cc in coords:
        r, c = int(rr), int(cc)
        by_diag[c - r].append((r, c))

    tracks: list[list[tuple[int, int]]] = []
    for cells in by_diag.values():
        cells.sort()
        run = [cells[0]]
        for r, c in cells[1:]:
            pr, pc = run[-1]
            if r == pr + 1 and c == pc + 1:
                run.append((r, c))
            else:
                tracks.append(run)
                run = [(r, c)]
        tracks.append(run)

    tracks.sort(key=lambda track: (track[0][0], track[0][1]))
    return tracks


def _track_values(grid: np.ndarray, track: list[tuple[int, int]]) -> list[int]:
    return [int(grid[r, c]) for r, c in track]


def _pair_info(grid: np.ndarray, track: list[tuple[int, int]]) -> tuple[int, int, int] | None:
    """Return (left_index, right_index, color) for an occupied track, else None."""
    values = _track_values(grid, track)
    color_indices = [idx for idx, value in enumerate(values) if value != GREY]
    if not color_indices:
        assert all(value == GREY for value in values)
        return None

    assert len(color_indices) == 2, "an occupied track must contain exactly two colored pixels"
    i, j = color_indices
    color = values[i]
    assert color in PAIR_COLORS, "track colors must come from the visible foreground palette"
    assert values[j] == color, "the two moving pixels must have the same color"
    for idx, value in enumerate(values):
        if idx in (i, j):
            continue
        assert value == GREY, "all other track cells must stay grey"
    return i, j, color


def _adjacent_stop_indices(i: int, j: int) -> tuple[int, int]:
    """
    Resolve the final adjacent positions after repeated single-cell inward moves.

    The upper-left pixel moves first in each round; when the initial gap has even
    size, this is equivalent to the simultaneous interpretation used by the
    generator's unambiguous samples.
    """
    assert i < j
    diff = j - i
    left_moves = diff // 2
    right_moves = (diff - 1) // 2
    return i + left_moves, j - right_moves


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each same-color diagonal track pair inward along its grey lattice until the two pixels touch.

    Input structure:
    - The grid has a black background.
    - Several NW-SE diagonal tracks are drawn with grey pixels; each track is one contiguous run on a constant (c-r) diagonal.
    - Some tracks are empty grey rails, while each occupied track contains exactly two same-color pixels replacing two grey lattice points.

    Rule:
    - Detect every grey-and-colored NW-SE track independently.
    - On each occupied track, move the upper-left colored pixel one lattice step toward the lower-right and move the lower-right pixel one lattice step toward the upper-left, repeating until the two become adjacent on the track.
    - Restore the vacated cells to grey, keep the moved pair in its original color, and leave black off-track cells unchanged.

    Key insight:
    - The grey diagonal lattice is not decoration: it is the only legal motion path, so pairing and stopping must be solved separately on each diagonal run rather than across the whole canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for track in _extract_tracks(g):
        info = _pair_info(g, track)
        if info is None:
            continue

        i, j, color = info
        new_i, new_j = _adjacent_stop_indices(i, j)
        out[track[i]] = GREY
        out[track[j]] = GREY
        out[track[new_i]] = color
        out[track[new_j]] = color

    return out


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    """Independent stepwise oracle used by the generator and semantic self-checks."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for track in _extract_tracks(g):
        info = _pair_info(g, track)
        if info is None:
            continue

        left, right, color = info
        cur_left, cur_right = left, right
        move_left = True
        while cur_right - cur_left > 1:
            if move_left:
                cur_left += 1
            else:
                cur_right -= 1
            move_left = not move_left

        out[track[left]] = GREY
        out[track[right]] = GREY
        out[track[cur_left]] = color
        out[track[cur_right]] = color

    return out


def _validate_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BLACK, GREY, *PAIR_COLORS})

    tracks = _extract_tracks(g)
    assert 3 <= len(tracks) <= MAX_TRACKS
    offsets = sorted(track[0][1] - track[0][0] for track in tracks)
    assert all((b - a) in (2, 3) for a, b in zip(offsets, offsets[1:])), "track spacing must stay 2 or 3"

    occupied_count = 0
    occupied_colors: list[int] = []
    grey_count = 0
    for track in tracks:
        assert len(track) >= MIN_TRACK_LEN
        values = _track_values(g, track)
        grey_count += sum(value == GREY for value in values)
        info = _pair_info(g, track)
        if info is None:
            continue
        i, j, color = info
        occupied_count += 1
        occupied_colors.append(color)
        assert j - i >= 3, "there must be at least two grey cells between the pair"
        assert (j - i - 1) % 2 == 0, "generator keeps the visible motion semantics unambiguous"

    assert grey_count > 0
    assert 2 <= occupied_count <= min(len(tracks), MAX_OCCUPIED_TRACKS)
    return {
        "tracks": tracks,
        "occupied_count": occupied_count,
        "occupied_colors": occupied_colors,
    }


def _validate_solution(inp: np.ndarray, out: np.ndarray) -> None:
    g = np.asarray(inp, dtype=int)
    o = np.asarray(out, dtype=int)
    assert g.shape == o.shape
    assert np.array_equal(g == BLACK, o == BLACK), "off-track black cells must remain black"

    expected = _oracle_output(g)
    assert np.array_equal(o, expected), "main() must match the stepwise task semantics"

    for track in _extract_tracks(g):
        before = _pair_info(g, track)
        after = _pair_info(o, track)
        if before is None:
            assert after is None
            continue
        assert after is not None
        _i, _j, color = before
        ai, aj, acolor = after
        assert acolor == color
        assert aj - ai == 1, "each moved pair must stop at diagonal adjacency"


def _sample_track_offsets(height: int, width: int) -> list[int]:
    min_diag = -(height - MIN_TRACK_LEN)
    max_diag = width - MIN_TRACK_LEN
    max_span = max_diag - min_diag
    max_tracks = min(MAX_TRACKS, max_span // 2 + 1)
    assert max_tracks >= 3

    for _ in range(100):
        track_count = int(_RNG.integers(3, max_tracks + 1))
        gaps = [int(x) for x in _RNG.choice([2, 3], size=track_count - 1, replace=True)]
        if track_count >= 4 and _RNG.random() < 0.6 and len(set(gaps)) == 1:
            continue
        span = sum(gaps)
        if span > max_span:
            continue
        start = int(_RNG.integers(min_diag, max_diag - span + 1))
        offsets = [start]
        cur = start
        for gap in gaps:
            cur += gap
            offsets.append(cur)
        return offsets

    raise RuntimeError("failed to sample valid diagonal offsets")


def _sample_colors(occupied_count: int) -> list[int]:
    if occupied_count == 2:
        if _RNG.random() < 0.5:
            color = int(_RNG.choice(PAIR_COLORS))
            return [color, color]
        colors = [int(x) for x in _RNG.choice(PAIR_COLORS, size=2, replace=False)]
        return colors

    for _ in range(100):
        colors = [int(x) for x in _RNG.choice(PAIR_COLORS, size=occupied_count, replace=True)]
        unique = len(set(colors))
        if unique == 1:
            continue
        if occupied_count == 3 and _RNG.random() < 0.6 and unique == 3:
            continue
        return colors

    raise RuntimeError("failed to sample a nondegenerate color pattern")


def _sample_pair_indices(track_len: int, prefer_long: bool) -> tuple[int, int]:
    separations = [sep for sep in range(3, track_len, 2)]
    assert separations
    if prefer_long:
        long_options = [sep for sep in separations if sep >= 5]
        if long_options:
            separations = long_options
    elif 3 in separations and _RNG.random() < 0.55:
        separations = [3]

    sep = int(_RNG.choice(separations))
    left = int(_RNG.integers(0, track_len - sep))
    return left, left + sep


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(12, 19))
    width = int(_RNG.integers(12, 19))
    offsets = _sample_track_offsets(height, width)
    tracks = [_diag_cells(height, width, offset) for offset in offsets]

    track_count = len(tracks)
    max_occupied = min(track_count, MAX_OCCUPIED_TRACKS)
    occupied_count = int(_RNG.integers(2, max_occupied + 1))
    occupied_indices = sorted(int(x) for x in _RNG.choice(track_count, size=occupied_count, replace=False))
    colors = _sample_colors(occupied_count)

    grid = np.zeros((height, width), dtype=int)
    for track in tracks:
        for r, c in track:
            grid[r, c] = GREY

    long_slot = None
    long_candidates = [slot for slot, track_idx in enumerate(occupied_indices) if len(tracks[track_idx]) >= 8]
    if long_candidates and _RNG.random() < 0.75:
        long_slot = int(_RNG.choice(long_candidates))

    for slot, track_idx in enumerate(occupied_indices):
        track = tracks[track_idx]
        left, right = _sample_pair_indices(len(track), prefer_long=slot == long_slot)
        color = colors[slot]
        grid[track[left]] = color
        grid[track[right]] = color

    info = _validate_input(grid)
    occupied_colors = info["occupied_colors"]
    assert isinstance(occupied_colors, list)
    assert len(occupied_colors) == occupied_count

    out = main(grid)
    _validate_solution(grid, out)
    if not np.array_equal(out, _oracle_output(grid)):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-lattice sliding input")
