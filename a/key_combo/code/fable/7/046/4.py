from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
WHITE = 10
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MODE_SEQUENCE = ("clean", "overlap", "clip")

_GENERATE_CALL_COUNT = 0


Coord = tuple[int, int]
Marker = tuple[Coord, Coord, int, int]


def _row_major_key(coord: Coord) -> tuple[int, int]:
    return int(coord[0]), int(coord[1])


def _row_major_gt(a: Coord, b: Coord | None) -> bool:
    if b is None:
        return True
    return _row_major_key(a) > _row_major_key(b)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _rotate_rel(rel: list[Coord], turns: int) -> list[Coord]:
    turns %= 4
    out: list[Coord] = []
    for dr, dc in rel:
        if turns == 0:
            out.append((dr, dc))
        elif turns == 1:
            out.append((dc, -dr))
        elif turns == 2:
            out.append((-dr, -dc))
        else:
            out.append((-dc, dr))
    return out


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == GREY, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full grey separator row"
    return int(rows[0])


def _find_seed_relcoords(grid: np.ndarray, sep: int) -> list[Coord]:
    upper = np.asarray(grid[:sep, :], dtype=int)
    seed_colors = sorted({int(v) for v in upper.ravel()} - {BLACK})
    assert len(seed_colors) == 1, "expected exactly one seed color above the separator"
    seed_color = seed_colors[0]

    coords = np.argwhere(upper == seed_color)
    assert coords.size > 0, "expected a seed object"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    crop = upper[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    mask = crop == seed_color

    hole = _find_unique_enclosed_hole(mask)
    hole_r = int(r0) + hole[0]
    hole_c = int(c0) + hole[1]

    rel = [(int(r) - hole_r, int(c) - hole_c) for r, c in coords]
    return sorted(rel)


def _find_unique_enclosed_hole(mask: np.ndarray) -> Coord:
    h, w = map(int, mask.shape)
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[Coord] = deque()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    holes = [(r, c) for r in range(h) for c in range(w) if not mask[r, c] and not visited[r, c]]
    assert len(holes) == 1, "expected a unique enclosed hole"
    return holes[0]


def _extract_markers(grid: np.ndarray, sep: int) -> list[Marker]:
    whites = [tuple(map(int, coord)) for coord in np.argwhere(grid == WHITE) if int(coord[0]) > sep]
    markers: list[Marker] = []
    seen_colors: set[int] = set()
    for white in sorted(whites, key=_row_major_key):
        wr, wc = white
        options: list[tuple[Coord, int, int]] = []
        for turns, (dr, dc) in enumerate(DIRS):
            rr, cc = wr + dr, wc + dc
            if rr <= sep or not _in_bounds(grid.shape, rr, cc):
                continue
            color = int(grid[rr, cc])
            if color not in (BLACK, GREY, WHITE):
                options.append(((rr, cc), color, turns))
        assert len(options) == 1, "each white anchor must have exactly one colored partner"
        colored, color, turns = options[0]
        assert color not in seen_colors, "marker colors must be unique"
        seen_colors.add(color)
        markers.append((white, colored, color, turns))
    assert markers, "expected at least one marker pair"
    return markers


def _apply_stamp(out: np.ndarray, sep: int, rel: list[Coord], marker: Marker) -> None:
    (wr, wc), _colored, color, turns = marker
    for dr, dc in _rotate_rel(rel, turns):
        rr, cc = wr + dr, wc + dc
        if rr <= sep or not _in_bounds(out.shape, rr, cc):
            continue
        if int(out[rr, cc]) == BLACK:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the seed around its enclosed hole and stamp recolored copies onto the marker anchors.

    Input structure:
    - A full grey horizontal line splits the grid into a narrow seed area above and a wider black main area below.
    - The upper area contains one monochrome 4-connected seed object whose bbox encloses exactly one black hole.
    - The lower area contains several isolated marker pairs, each made of one white cell and one orthogonally adjacent colored cell.

    Rule:
    - Process marker pairs in row-major order of their white cells.
    - Let the colored cell direction relative to the white cell be up, right, down, or left; rotate the seed around its enclosed hole so the original up-facing pose points in that direction.
    - Translate the rotated seed so the hole lands on the white cell, then paint the seed foreground with the marker color only onto still-black cells in the lower area, clipping anything out of bounds.

    Key insight:
    - The white cell gives the hidden hole anchor, the colored partner gives both the stamp color and the rotation, and the black-only overwrite rule makes earlier row-major stamps win overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    rel = _find_seed_relcoords(g, sep)
    markers = _extract_markers(g, sep)

    out = g.copy()
    for marker in markers:
        _apply_stamp(out, sep, rel, marker)
    return out


def _normalize_rel(coords: set[Coord]) -> list[Coord]:
    return sorted((int(r), int(c)) for r, c in coords)


def _make_seed_relcoords() -> list[Coord]:
    base = {
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    }
    side_specs = [
        ((-1, 0), (-1, 0), (0, 1)),
        ((0, 1), (0, 1), (1, 0)),
        ((1, 0), (1, 0), (0, 1)),
        ((0, -1), (0, -1), (1, 0)),
    ]

    for _ in range(200):
        coords = set(base)
        lengths = [random.randint(0, 3) for _ in range(4)]
        if max(lengths) == 0:
            continue

        for side, length in enumerate(lengths):
            (sr, sc), (dr, dc), _orth = side_specs[side]
            for step in range(1, length + 1):
                coords.add((sr + dr * step, sc + dc * step))

        flag_count = random.randint(1, 3)
        viable_sides = [side for side, length in enumerate(lengths) if length > 0]
        random.shuffle(viable_sides)
        for side in viable_sides[:flag_count]:
            (sr, sc), (dr, dc), (or_, oc_) = side_specs[side]
            tip = (sr + dr * lengths[side], sc + dc * lengths[side])
            sign = random.choice((-1, 1))
            coords.add((tip[0] + or_ * sign, tip[1] + oc_ * sign))

        rel = _normalize_rel(coords)
        rel_set = set(rel)
        if any(set(_rotate_rel(rel, turns)) == rel_set for turns in (1, 2, 3)):
            continue

        rows = [r for r, _ in rel]
        cols = [c for _, c in rel]
        if max(rows) - min(rows) + 1 > 8 or max(cols) - min(cols) + 1 > 8:
            continue
        return rel

    raise RuntimeError("failed to build an asymmetric seed shape")


def _seed_array(rel: list[Coord], color: int) -> np.ndarray:
    rows = [r for r, _ in rel] + [0]
    cols = [c for _, c in rel] + [0]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in rel:
        arr[r - r0, c - c0] = color
    return arr


def _stamp_metrics(
    shape: tuple[int, int],
    sep: int,
    rel: list[Coord],
    white: Coord,
    colored: Coord,
    turns: int,
    input_grid: np.ndarray,
    current_output: np.ndarray,
) -> dict[str, int]:
    total = 0
    clipped = 0
    blocked_by_input = 0
    blocked_by_prior = 0
    painted = 0
    colored_covered = False

    wr, wc = white
    for dr, dc in _rotate_rel(rel, turns):
        total += 1
        rr, cc = wr + dr, wc + dc
        if rr <= sep or not _in_bounds(shape, rr, cc):
            clipped += 1
            continue
        if (rr, cc) == colored:
            blocked_by_input += 1
            colored_covered = True
            continue
        if int(input_grid[rr, cc]) != BLACK:
            blocked_by_input += 1
            continue
        if int(current_output[rr, cc]) != BLACK:
            blocked_by_prior += 1
            continue
        painted += 1

    assert colored_covered, "rotated seed must cover the colored marker cell"
    return {
        "total": total,
        "clipped": clipped,
        "blocked_by_input": blocked_by_input,
        "blocked_by_prior": blocked_by_prior,
        "painted": painted,
    }


def _is_pair_isolated(occupied: set[Coord], white: Coord, colored: Coord) -> bool:
    pair = {white, colored}
    if occupied & pair:
        return False
    for pr, pc in pair:
        for qr, qc in occupied:
            if abs(pr - qr) + abs(pc - qc) == 1:
                return False
    return True


def _place_marker(grid: np.ndarray, output: np.ndarray, marker: Marker, rel: list[Coord]) -> None:
    white, colored, color, _turns = marker
    grid[white] = WHITE
    grid[colored] = color
    output[white] = WHITE
    output[colored] = color
    _apply_stamp(output, _find_separator_row(grid), rel, marker)


def _find_candidates(
    input_grid: np.ndarray,
    current_output: np.ndarray,
    rel: list[Coord],
    sep: int,
    occupied: set[Coord],
    last_white: Coord | None,
    require_overlap: bool,
    require_clip: bool,
) -> list[tuple[Coord, Coord, int, dict[str, int]]]:
    h, w = map(int, input_grid.shape)
    candidates: list[tuple[Coord, Coord, int, dict[str, int]]] = []
    for wr in range(sep + 1, h):
        for wc in range(w):
            white = (wr, wc)
            if not _row_major_gt(white, last_white):
                continue
            if int(current_output[white]) != BLACK:
                continue
            for turns, (dr, dc) in enumerate(DIRS):
                colored = (wr + dr, wc + dc)
                if colored[0] <= sep or not _in_bounds((h, w), colored[0], colored[1]):
                    continue
                if int(current_output[colored]) != BLACK:
                    continue
                if not _is_pair_isolated(occupied, white, colored):
                    continue

                metrics = _stamp_metrics(
                    (h, w),
                    sep,
                    rel,
                    white,
                    colored,
                    turns,
                    input_grid,
                    current_output,
                )
                if metrics["blocked_by_input"] != 1:
                    continue
                if metrics["painted"] < 4:
                    continue
                if require_overlap and metrics["blocked_by_prior"] == 0:
                    continue
                if not require_overlap and metrics["blocked_by_prior"] != 0:
                    continue
                if require_clip and metrics["clipped"] == 0:
                    continue
                if not require_clip and metrics["clipped"] != 0:
                    continue
                candidates.append((white, colored, turns, metrics))
    return candidates


def _trace_output(grid: np.ndarray, rel: list[Coord]) -> tuple[np.ndarray, list[dict[str, int]]]:
    sep = _find_separator_row(grid)
    markers = _extract_markers(grid, sep)
    out = grid.copy()
    stats: list[dict[str, int]] = []
    for white, colored, color, turns in markers:
        metrics = _stamp_metrics(out.shape, sep, rel, white, colored, turns, grid, out)
        stats.append(metrics)
        _apply_stamp(out, sep, rel, (white, colored, color, turns))
    return out, stats


def _build_canvas(seed_arr: np.ndarray, mode: str) -> tuple[np.ndarray, int]:
    seed_h, seed_w = map(int, seed_arr.shape)
    top_margin = random.randint(1, 2)
    seed_bottom_gap = random.randint(1, 2)
    top_h = top_margin + seed_h + seed_bottom_gap
    if mode == "overlap":
        lower_h = random.randint(9, 11)
        width = random.randint(max(seed_w + 4, 12), 17)
    elif mode == "clip":
        lower_h = random.randint(10, 13)
        width = random.randint(max(seed_w + 4, 12), 18)
    else:
        lower_h = random.randint(11, 15)
        width = random.randint(max(seed_w + 5, 14), 22)

    height = top_h + 1 + lower_h
    assert height <= 30 and width <= 30

    grid = np.zeros((height, width), dtype=int)
    sep = top_h
    grid[sep, :] = GREY

    left = random.randint(1, width - seed_w - 1)
    grid[top_margin : top_margin + seed_h, left : left + seed_w] = seed_arr
    return grid, sep


def _final_quality(input_grid: np.ndarray, output_grid: np.ndarray, stats: list[dict[str, int]], mode: str) -> bool:
    sep = _find_separator_row(input_grid)
    markers = _extract_markers(input_grid, sep)
    turns_used = {turns for _white, _colored, _color, turns in markers}
    if len(turns_used) < 2:
        return False

    added = int(np.sum((output_grid != input_grid) & (input_grid == BLACK)))
    lower_area = int((input_grid.shape[0] - sep - 1) * input_grid.shape[1])
    if added < 12 or added > int(lower_area * 0.38):
        return False

    any_overlap = any(item["blocked_by_prior"] > 0 for item in stats)
    any_clip = any(item["clipped"] > 0 for item in stats)

    if mode == "clean":
        return not any_overlap and not any_clip
    if mode == "overlap":
        return any_overlap and not any_clip
    return any_clip and not any_overlap


def _generate_one(mode: str) -> np.ndarray | None:
    rel = _make_seed_relcoords()
    seed_color = random.choice(MARKER_COLORS)
    seed_arr = _seed_array(rel, seed_color)
    base_grid, sep = _build_canvas(seed_arr, mode)

    palette = [color for color in MARKER_COLORS if color != seed_color]
    marker_count = random.randint(3, min(5, len(palette)))
    marker_colors = random.sample(palette, marker_count)

    input_grid = base_grid.copy()
    current_output = base_grid.copy()
    occupied: set[Coord] = set()
    markers: list[Marker] = []
    last_white: Coord | None = None

    base_count = marker_count if mode == "clean" else marker_count - 1
    for idx in range(base_count):
        candidates = _find_candidates(
            input_grid,
            current_output,
            rel,
            sep,
            occupied,
            last_white,
            require_overlap=False,
            require_clip=False,
        )
        if not candidates:
            return None
        white, colored, turns, _metrics = random.choice(candidates)
        marker = (white, colored, marker_colors[idx], turns)
        markers.append(marker)
        _place_marker(input_grid, current_output, marker, rel)
        occupied.update((white, colored))
        last_white = white

    if mode != "clean":
        candidates = _find_candidates(
            input_grid,
            current_output,
            rel,
            sep,
            occupied,
            last_white,
            require_overlap=(mode == "overlap"),
            require_clip=(mode == "clip"),
        )
        if not candidates:
            return None
        white, colored, turns, _metrics = random.choice(candidates)
        marker = (white, colored, marker_colors[-1], turns)
        markers.append(marker)
        _place_marker(input_grid, current_output, marker, rel)

    traced_output, stats = _trace_output(input_grid, rel)
    if not np.array_equal(traced_output, current_output):
        return None
    if not np.array_equal(traced_output, main(input_grid)):
        return None
    if not _final_quality(input_grid, traced_output, stats, mode):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT

    mode = MODE_SEQUENCE[_GENERATE_CALL_COUNT % len(MODE_SEQUENCE)]
    _GENERATE_CALL_COUNT += 1

    for _ in range(400):
        grid = _generate_one(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input in mode={mode}")
