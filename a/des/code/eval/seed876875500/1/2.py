from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BRANCHES = ("none", "single", "multi")
GENERATION_RETRIES = 400
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (14, 20)
PALETTE = tuple(range(10))


@dataclass(frozen=True)
class Track:
    color: int
    orientation: str  # "h" or "v"
    fixed: int
    lo: int
    hi: int
    direction: int  # +1 means lo -> hi, -1 means hi -> lo
    piece_bbox: tuple[int, int, int, int]

    def start_coord(self) -> int:
        return self.lo if self.direction > 0 else self.hi

    def end_coord(self) -> int:
        return self.hi if self.direction > 0 else self.lo

    def start_cell(self) -> tuple[int, int]:
        if self.orientation == "h":
            return self.fixed, self.start_coord()
        return self.start_coord(), self.fixed

    def piece_cells(self) -> list[tuple[int, int]]:
        return list(_bbox_cells(self.piece_bbox))

    def track_cells(self) -> list[tuple[int, int]]:
        if self.orientation == "h":
            return [(self.fixed, c) for c in range(self.lo, self.hi + 1)]
        return [(r, self.fixed) for r in range(self.lo, self.hi + 1)]

    def final_bbox(self) -> tuple[int, int, int, int]:
        r0, c0, r1, c1 = self.piece_bbox
        end = self.end_coord()
        if self.orientation == "h":
            if self.direction > 0:
                return r0, end - 1, r1, end
            return r0, end, r1, end + 1
        if self.direction > 0:
            return end - 1, c0, end, c1
        return end, c0, end + 1, c1

    def axis_value(self, cell: tuple[int, int]) -> int:
        return cell[1] if self.orientation == "h" else cell[0]


def _bbox_cells(bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield (r, c)


def _bbox_in_bounds(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _most_frequent_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _piece_windows(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    h, w = grid.shape
    pieces: list[tuple[int, tuple[int, int, int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            color = int(block[0, 0])
            if color == bg or not np.all(block == color):
                continue
            cells = set(_bbox_cells((r, c, r + 1, c + 1)))
            assert not (cells & occupied), "overlapping 2x2 fragments are not allowed"
            pieces.append((color, (r, c, r + 1, c + 1)))
            occupied |= cells
    assert pieces, "expected at least one 2x2 fragment"
    return pieces


def _track_from_piece(grid: np.ndarray, bg: int, color: int, bbox: tuple[int, int, int, int]) -> Track:
    h, w = grid.shape
    piece_cells = set(_bbox_cells(bbox))
    neighbors: set[tuple[int, int]] = set()
    for r, c in piece_cells:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in piece_cells and int(grid[nr, nc]) == color:
                neighbors.add((nr, nc))

    assert len(neighbors) == 1, "each fragment must touch exactly one same-color track endpoint"
    nr, nc = next(iter(neighbors))
    r0, c0, r1, c1 = bbox

    if nc == c1 + 1 and nr in (r0, r1):
        end = nc
        while end + 1 < w and int(grid[nr, end + 1]) != bg:
            end += 1
        return Track(color, "h", nr, nc, end, 1, bbox)

    if nc == c0 - 1 and nr in (r0, r1):
        start = nc
        while start - 1 >= 0 and int(grid[nr, start - 1]) != bg:
            start -= 1
        return Track(color, "h", nr, start, nc, -1, bbox)

    if nr == r1 + 1 and nc in (c0, c1):
        end = nr
        while end + 1 < h and int(grid[end + 1, nc]) != bg:
            end += 1
        return Track(color, "v", nc, nr, end, 1, bbox)

    if nr == r0 - 1 and nc in (c0, c1):
        start = nr
        while start - 1 >= 0 and int(grid[start - 1, nc]) != bg:
            start -= 1
        return Track(color, "v", nc, start, nr, -1, bbox)

    raise AssertionError("fragment adjacency does not define a valid straight track")


def _intersections(tracks: list[Track]) -> list[list[tuple[tuple[int, int], int]]]:
    result: list[list[tuple[tuple[int, int], int]]] = [[] for _ in tracks]
    for i, first in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            second = tracks[j]
            if first.orientation == second.orientation:
                continue
            if first.orientation == "h":
                horizontal, vertical = first, second
                h_idx, v_idx = i, j
            else:
                horizontal, vertical = second, first
                h_idx, v_idx = j, i

            row = horizontal.fixed
            col = vertical.fixed
            if vertical.lo <= row <= vertical.hi and horizontal.lo <= col <= horizontal.hi:
                cell = (row, col)
                result[h_idx].append((cell, vertical.color))
                result[v_idx].append((cell, horizontal.color))
    return result


def _piece_color(track: Track, hits: list[tuple[tuple[int, int], int]], color_counts: dict[int, int]) -> int:
    final_bbox = track.final_bbox()
    covered = [
        (abs(track.axis_value(cell) - track.end_coord()), other_color)
        for cell, other_color in hits
        if _in_bbox(cell, final_bbox)
    ]
    if not covered:
        return track.color

    nearest_by_color: dict[int, int] = {}
    for dist, other_color in sorted(covered, key=lambda item: item[0]):
        nearest_by_color.setdefault(other_color, dist)

    if len(nearest_by_color) == 1:
        return next(iter(nearest_by_color))

    min_count = min(color_counts.get(color, 0) for color in nearest_by_color)
    tied = [color for color in nearest_by_color if color_counts.get(color, 0) == min_count]
    if len(tied) == 1:
        return tied[0]

    # Visible tie-break: prefer the color whose covered intersection is closest to the terminal cell.
    for color, _dist in nearest_by_color.items():
        if color in tied:
            return color
    raise AssertionError("unreachable")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 2x2 fragment along its straight same-color track to the far endpoint.
    If the final 2x2 position covers intersections with other tracks, recolor the whole
    fragment to the crossed track color; with multiple different crossed colors, choose
    the least frequent visible color in the input grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_frequent_color(g)

    pieces = _piece_windows(g, bg)
    tracks = [_track_from_piece(g, bg, color, bbox) for color, bbox in pieces]
    color_counts = {int(color): int(count) for color, count in zip(*np.unique(g, return_counts=True))}
    track_hits = _intersections(tracks)

    out = g.copy()
    for track in tracks:
        for r, c in track.piece_cells():
            out[r, c] = bg

    for track, hits in zip(tracks, track_hits):
        final_color = _piece_color(track, hits, color_counts)
        r0, c0, r1, c1 = track.final_bbox()
        assert _bbox_in_bounds(g.shape, (r0, c0, r1, c1)), "moved fragment must stay in bounds"
        out[r0 : r1 + 1, c0 : c1 + 1] = final_color

    return out


def _make_horizontal(
    color: int,
    row: int,
    start: int,
    end: int,
    direction: int,
    piece_above: bool,
) -> Track:
    if direction > 0:
        lo, hi = start, end
        bbox = (row - 1 if piece_above else row, start - 2, row if piece_above else row + 1, start - 1)
    else:
        lo, hi = end, start
        bbox = (row - 1 if piece_above else row, start + 1, row if piece_above else row + 1, start + 2)
    return Track(color, "h", row, lo, hi, direction, bbox)


def _make_vertical(
    color: int,
    col: int,
    start: int,
    end: int,
    direction: int,
    piece_left: bool,
) -> Track:
    if direction > 0:
        lo, hi = start, end
        bbox = (start - 2, col - 1 if piece_left else col, start - 1, col if piece_left else col + 1)
    else:
        lo, hi = end, start
        bbox = (start + 1, col - 1 if piece_left else col, start + 2, col if piece_left else col + 1)
    return Track(color, "v", col, lo, hi, direction, bbox)


def _validate_layout(shape: tuple[int, int], tracks: list[Track]) -> bool:
    h, w = shape
    line_orientations: dict[tuple[int, int], str] = {}
    track_cells: set[tuple[int, int]] = set()
    for track in tracks:
        if track.lo >= track.hi:
            return False
        if not _bbox_in_bounds(shape, track.piece_bbox) or not _bbox_in_bounds(shape, track.final_bbox()):
            return False
        for cell in track.track_cells():
            if not (0 <= cell[0] < h and 0 <= cell[1] < w):
                return False
            if cell in line_orientations and line_orientations[cell] == track.orientation:
                return False
            line_orientations.setdefault(cell, track.orientation)
            track_cells.add(cell)

    all_piece_cells: set[tuple[int, int]] = set()
    final_piece_cells: set[tuple[int, int]] = set()
    for track in tracks:
        piece_cells = set(track.piece_cells())
        if piece_cells & track_cells or piece_cells & all_piece_cells:
            return False
        all_piece_cells |= piece_cells

        surrounding: set[tuple[int, int]] = set()
        for r, c in piece_cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in piece_cells:
                    if (nr, nc) in track_cells or (nr, nc) in (all_piece_cells - piece_cells):
                        surrounding.add((nr, nc))
        if surrounding != {track.start_cell()}:
            return False

        final_cells = set(_bbox_cells(track.final_bbox()))
        if final_cells & final_piece_cells:
            return False
        final_piece_cells |= final_cells

    return True


def _render_input(shape: tuple[int, int], bg: int, tracks: list[Track]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), bg, dtype=int)
    occupancy: dict[tuple[int, int], list[int]] = {}
    for track in tracks:
        for cell in track.track_cells():
            occupancy.setdefault(cell, []).append(track.color)

    for cell, colors in occupancy.items():
        grid[cell] = random.choice(colors)

    for track in tracks:
        r0, c0, r1, c1 = track.piece_bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = track.color

    return grid


def _focus_branch(grid: np.ndarray, tracks: list[Track]) -> str | None:
    hits = _intersections(tracks)[0]
    counts = {int(color): int(count) for color, count in zip(*np.unique(grid, return_counts=True))}
    bbox = tracks[0].final_bbox()
    colors = []
    for cell, other_color in hits:
        if _in_bbox(cell, bbox) and other_color not in colors:
            colors.append(other_color)
    if not colors:
        return "none"
    if len(colors) == 1:
        return "single"
    min_count = min(counts.get(color, 0) for color in colors)
    if sum(counts.get(color, 0) == min_count for color in colors) != 1:
        return None
    return "multi"


def _vertical_pairs(height: int, focus_row: int, bias: str) -> list[tuple[int, int]]:
    pairs = [
        (lo, hi)
        for lo in range(2, focus_row - 1)
        for hi in range(focus_row + 2, height - 2)
        if hi - lo + 1 >= 5
    ]
    if not pairs:
        return []
    pairs.sort(key=lambda pair: pair[1] - pair[0])
    band = max(1, len(pairs) // 3)
    if bias == "short":
        return pairs[:band]
    if bias == "long":
        return pairs[-band:]
    middle = pairs[band:-band]
    return middle if middle else pairs[band : band + 1]


def _sample_vertical(
    color: int,
    height: int,
    col: int,
    focus_row: int,
    bias: str,
    direction: int | None = None,
    piece_left: bool | None = None,
) -> Track | None:
    pairs = _vertical_pairs(height, focus_row, bias)
    if not pairs:
        return None
    lo, hi = random.choice(pairs)
    if direction is None:
        direction = random.choice((-1, 1))
    if piece_left is None:
        piece_left = random.choice((False, True))
    start, end = (lo, hi) if direction > 0 else (hi, lo)
    return _make_vertical(color, col, start, end, direction, piece_left)


def _sample_focus_track(color: int, height: int, width: int, extra_columns: int) -> tuple[Track, list[int], list[int]] | None:
    row = random.randint(4, height - 5)
    direction = random.choice((-1, 1))
    piece_above = random.choice((False, True))
    min_len = 7 + max(0, extra_columns - 1)

    if direction > 0:
        start_max = width - min_len - 2
        if start_max < 2:
            return None
        start = random.randint(2, start_max)
        end = random.randint(start + min_len - 1, width - 3)
        track = _make_horizontal(color, row, start, end, direction, piece_above)
        final_columns = [end - 1, end]
        blocked = {start, start + 1}
    else:
        start_min = min_len + 1
        if start_min > width - 3:
            return None
        start = random.randint(start_min, width - 3)
        end = random.randint(2, start - min_len + 1)
        track = _make_horizontal(color, row, start, end, direction, piece_above)
        final_columns = [end, end + 1]
        blocked = {start, start - 1}

    available = [c for c in range(track.lo, track.hi + 1) if c not in set(final_columns) | blocked]
    if len(available) < extra_columns:
        return None
    return track, final_columns, available


def _gen_attempt(branch: str) -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    n_near = {"none": 0, "single": 1, "multi": 2}[branch]
    if branch == "none":
        n_extra = random.choice((1, 2))
    elif branch == "single":
        n_extra = random.choice((0, 1))
    else:
        n_extra = 0
    focus_info = _sample_focus_track(0, height, width, n_extra)
    if focus_info is None:
        return None
    focus_track, final_columns, spare_columns = focus_info

    if n_near == 0:
        near_columns: list[int] = []
    elif n_near == 1:
        near_columns = [random.choice(final_columns)]
    else:
        near_columns = sorted(final_columns)

    if len(spare_columns) < n_extra:
        return None
    extra_columns = random.sample(spare_columns, n_extra)
    all_columns = near_columns + extra_columns

    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], 1 + len(all_columns))
    focus_track = Track(colors[0], focus_track.orientation, focus_track.fixed, focus_track.lo, focus_track.hi, focus_track.direction, focus_track.piece_bbox)

    tracks = [focus_track]
    next_color = 1
    if branch == "multi":
        short_first = random.choice((False, True))
        near_specs = [
            (near_columns[0], "short" if short_first else "long"),
            (near_columns[1], "long" if short_first else "short"),
        ]
        direction_pattern = random.choice(((1, -1), (-1, 1)))
        side_pattern = random.choice(((True, False), (False, True)))
        for (col, bias), direction, piece_left in zip(near_specs, direction_pattern, side_pattern):
            track = _sample_vertical(
                colors[next_color],
                height,
                col,
                focus_track.fixed,
                bias,
                direction=direction,
                piece_left=piece_left,
            )
            next_color += 1
            if track is None:
                return None
            tracks.append(track)
    else:
        for col in near_columns:
            track = _sample_vertical(colors[next_color], height, col, focus_track.fixed, random.choice(("short", "medium", "long")))
            next_color += 1
            if track is None:
                return None
            tracks.append(track)

    for col in extra_columns:
        bias = random.choice(("medium", "long")) if branch != "multi" else random.choice(("short", "medium", "long"))
        track = _sample_vertical(colors[next_color], height, col, focus_track.fixed, bias)
        next_color += 1
        if track is None:
            return None
        tracks.append(track)

    if not _validate_layout((height, width), tracks):
        return None

    grid = _render_input((height, width), bg, tracks)
    if _most_frequent_color(grid) != bg:
        return None

    if _focus_branch(grid, tracks) != branch:
        return None

    if random.choice((False, True)):
        grid = grid.T.copy()

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    branch = random.choice(BRANCHES)
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid {branch} input")
