from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MARKER_COLORS = (GREEN, YELLOW, ORANGE)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        steps = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _label_black_components(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    comps = _components_mask(grid == BLACK, connectivity=4)
    labels = np.zeros(grid.shape, dtype=int)
    for idx, coords in enumerate(comps, start=1):
        labels[coords[:, 0], coords[:, 1]] = idx
    return labels, comps


def _is_contiguous_diagonal(coords: np.ndarray) -> bool:
    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) < 2:
        return False
    if len({r - c for r, c in pts}) == 1:
        rows = sorted(r for r, _ in pts)
        cols = sorted(c for _, c in pts)
    elif len({r + c for r, c in pts}) == 1:
        rows = sorted(r for r, _ in pts)
        cols = sorted(c for _, c in pts)
    else:
        return False
    return (
        rows == list(range(rows[0], rows[-1] + 1))
        and cols == list(range(cols[0], cols[-1] + 1))
        and len(pts) == rows[-1] - rows[0] + 1
    )


def _find_seed(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for coords in _components_mask(grid == BLUE, connectivity=8):
        if len(coords) not in (2, 3) or not _is_contiguous_diagonal(coords):
            continue

        pts = {(int(r), int(c)) for r, c in coords}
        if len({r - c for r, c in pts}) == 1:
            first = min(pts)
            last = max(pts)
            endpoints = ((first, (-1, -1)), (last, (1, 1)))
        else:
            top = min(pts)
            bottom = max(pts)
            endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

        states = []
        for (r, c), (dr, dc) in endpoints:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc):
                states.append((r, c, dr, dc))
        if len(states) == 1:
            candidates.append((coords, states[0]))

    assert len(candidates) == 1, "expected one diagonal blue seed with one inward extension"
    return candidates[0]


def _mirror_orientations(grid: np.ndarray) -> tuple[dict[tuple[int, int], str], list[np.ndarray]]:
    mirrors: dict[tuple[int, int], str] = {}
    comps = _components_mask(grid == RED, connectivity=4)
    assert 2 <= len(comps) <= 3, "expected 2-3 red mirrors"
    for coords in comps:
        rows = sorted({int(r) for r, _ in coords})
        cols = sorted({int(c) for _, c in coords})
        assert len(coords) in (2, 3), "mirror length must be 2 or 3"
        if len(rows) == 1:
            assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal mirror must be contiguous"
            orientation = "horizontal"
        else:
            assert len(cols) == 1, "mirror must be horizontal or vertical"
            assert rows == list(range(rows[0], rows[-1] + 1)), "vertical mirror must be contiguous"
            orientation = "vertical"
        for r, c in coords:
            mirrors[(int(r), int(c))] = orientation
    return mirrors, comps


def _trace_ray(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    _seed_coords, (r, c, dr, dc) = _find_seed(out)
    mirrors, _mirror_comps = _mirror_orientations(out)

    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray must not loop forever"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(out.shape, nr, nc):
            break

        if int(out[nr, nc]) == RED:
            orientation = mirrors.get((nr, nc))
            assert orientation is not None, "red pixel must belong to one mirror"
            if orientation == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue

        r, c = nr, nc
        if int(out[r, c]) != GREY:
            out[r, c] = BLUE

    return out


def _markers_by_chamber(grid: np.ndarray, labels: np.ndarray) -> dict[int, list[tuple[int, tuple[int, int]]]]:
    by_chamber: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for color in MARKER_COLORS:
        for r, c in np.argwhere(grid == color):
            rr, cc = int(r), int(c)
            adjacent = {
                int(labels[nr, nc])
                for nr, nc in _neighbors4(rr, cc)
                if _in_bounds(grid.shape, nr, nc) and int(labels[nr, nc]) > 0
            }
            assert len(adjacent) <= 1, "marker must belong to at most one chamber"
            if adjacent:
                by_chamber[next(iter(adjacent))].append((int(color), (rr, cc)))
    return by_chamber


def _choose_fill_color(entries: list[tuple[int, tuple[int, int]]]) -> int | None:
    if not entries:
        return None
    counts = Counter(color for color, _pos in entries)
    best_count = max(counts.values())
    candidates = [color for color, cnt in counts.items() if cnt == best_count]
    if len(candidates) == 1:
        return int(candidates[0])
    best_pos = {}
    for color in candidates:
        best_pos[color] = min(pos for col, pos in entries if col == color)
    return min(candidates, key=lambda color: best_pos[color])


def _fill_from_traced(traced: np.ndarray, *, wall_mask: np.ndarray | None = None) -> np.ndarray:
    out = traced.copy()
    if wall_mask is None:
        wall_mask = (traced == BLUE) | (traced == GREY)
    work = traced.copy()
    work[wall_mask] = GREY
    labels, comps = _label_black_components(work)
    markers = _markers_by_chamber(traced, labels)
    for chamber_id, coords in enumerate(comps, start=1):
        fill = _choose_fill_color(markers.get(chamber_id, []))
        if fill is not None:
            out[coords[:, 0], coords[:, 1]] = fill
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Extend the diagonal ray, let the traced blue path and gray wall segments partition the black background, then fill each chamber by its internal marker vote.

    Input structure:
    - A black grid contains one short diagonal blue seed, 2-3 red straight mirror segments, several gray straight wall segments, and scattered green/yellow/orange singleton markers.
    - The blue seed is an 8-connected diagonal segment of length 2-3 with exactly one endpoint whose diagonal continuation stays inside the grid.
    - Gray walls are horizontal or vertical contiguous segments, and red mirrors are horizontal or vertical contiguous segments.

    Rule:
    - Start from the inward-pointing seed endpoint and advance one diagonal step at a time.
    - If the next cell is red, do not paint it; reflect by flipping the row direction on a horizontal mirror or the column direction on a vertical mirror.
    - Otherwise move into the next cell and paint it blue unless it is gray, then continue until the ray leaves the grid.
    - Treat the completed blue trace together with the gray cells as walls, find 4-connected black chambers, assign each marker to the unique chamber touching it, and fill each chamber with its majority marker color.
    - Chambers with no markers stay black; ties are broken by the tied color whose topmost marker is highest, then leftmost if needed. Non-black input pixels stay unchanged.

    Key insight:
    - The ray is not the final answer by itself: its reflected path dynamically completes the wall structure, and only that completed partition tells you which markers vote for each chamber.
    """
    traced = _trace_ray(np.asarray(input_grid, dtype=int))
    return _fill_from_traced(traced)


def _build_path_case() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(13, 16)
    top_len = random.randint(2, 5)
    bot_len = random.randint(2, 5)
    rt = top_len
    rb = h - 1 - bot_len
    span = rb - rt
    assert 4 <= span <= 8

    mirror_count = random.choice([2, 2, 3])
    seed_len = random.choice([2, 3])
    start_top = random.choice([True, False])
    r0 = rt if start_top else rb
    dr0 = 1 if start_top else -1
    boundary_r = r0 - dr0 * (seed_len - 1)
    assert 0 <= boundary_r < h

    points = [(r0, seed_len - 1)]
    cur_r, cur_c, cur_dr = r0, seed_len - 1, dr0
    for _ in range(mirror_count):
        nxt_r = rb if cur_dr == 1 else rt
        delta = abs(nxt_r - cur_r)
        cur_c += delta
        cur_r = nxt_r
        points.append((cur_r, cur_c))
        cur_dr = -cur_dr

    max_tail = cur_r if cur_dr == -1 else (h - 1 - cur_r)
    assert max_tail >= 2
    tail_len = random.randint(2, min(6, max_tail))
    w = cur_c + tail_len + 1
    assert 13 <= w <= 30

    path = []
    for k in range(seed_len):
        path.append((boundary_r + dr0 * k, k))
    cur_r, cur_c = points[0]
    cur_dr = dr0
    for nxt_r, nxt_c in points[1:]:
        seg_len = nxt_c - cur_c
        for step in range(1, seg_len + 1):
            path.append((cur_r + cur_dr * step, cur_c + step))
        cur_r, cur_c = nxt_r, nxt_c
        cur_dr = -cur_dr
    for step in range(1, tail_len + 1):
        path.append((cur_r + cur_dr * step, cur_c + step))

    grid = np.zeros((h, w), dtype=int)
    seed = path[:seed_len]
    for r, c in seed:
        grid[r, c] = BLUE
    path_set = set(path)
    wall_points = points[1:]
    reserved_wall_cells = set()
    for wr, wc in wall_points:
        if wr == rt:
            for rr in range(0, wr):
                reserved_wall_cells.add((rr, wc))
        else:
            for rr in range(wr + 1, h):
                reserved_wall_cells.add((rr, wc))

    incoming_dr = dr0
    for turn_r, turn_c in points[1:]:
        mr = turn_r + incoming_dr
        mc = turn_c + 1
        length = random.choice([2, 3])
        starts = []
        for start_c in range(max(0, mc - length + 1), min(mc, w - length) + 1):
            cells = [(mr, cc) for cc in range(start_c, start_c + length)]
            if any(cell in path_set or cell in reserved_wall_cells or grid[cell] != BLACK for cell in cells):
                continue
            starts.append(start_c)
        assert starts
        start_c = random.choice(starts)
        for cc in range(start_c, start_c + length):
            grid[mr, cc] = RED
        incoming_dr = -incoming_dr

    for wr, wc in wall_points:
        if wr == rt:
            for rr in range(0, wr):
                grid[rr, wc] = GREY
        else:
            for rr in range(wr + 1, h):
                grid[rr, wc] = GREY

    return grid, np.array(path, dtype=int)


def _transform(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = out.T.copy()
    if random.random() < 0.5:
        out = np.flipud(out).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    return out


def _candidate_marker_cells(labels: np.ndarray, chamber_id: int) -> list[tuple[int, int]]:
    candidates = []
    for r, c in np.argwhere(labels == chamber_id):
        rr, cc = int(r), int(c)
        orth = []
        ok = True
        for nr, nc in _neighbors4(rr, cc):
            if not _in_bounds(labels.shape, nr, nc) or int(labels[nr, nc]) != chamber_id:
                ok = False
                break
            orth.append((nr, nc))
        if ok:
            candidates.append((rr, cc))
    return candidates


def _spaced_sample(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    pool = candidates[:]
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if all(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) >= 2 for other in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _assign_markers(grid: np.ndarray) -> np.ndarray:
    traced = _trace_ray(grid)
    labels, comps = _label_black_components(traced)
    path_only = traced.copy()
    path_only[traced == GREY] = BLACK
    path_only[traced == BLUE] = GREY
    path_labels, _path_comps = _label_black_components(path_only)

    chamber_candidates = []
    for chamber_id, coords in enumerate(comps, start=1):
        candidates = _candidate_marker_cells(labels, chamber_id)
        if len(candidates) >= 2:
            path_groups = {int(path_labels[r, c]) for r, c in coords}
            path_groups.discard(0)
            assert len(path_groups) == 1
            chamber_candidates.append((chamber_id, candidates, len(coords), next(iter(path_groups))))
    assert len(chamber_candidates) >= 2

    by_path_group: dict[int, list[tuple[int, list[tuple[int, int]], int, int]]] = defaultdict(list)
    for item in chamber_candidates:
        by_path_group[item[3]].append(item)
    merged_options = [items for items in by_path_group.values() if len(items) >= 2]
    assert merged_options, "grey walls must split at least one path-only chamber"

    primary_group = random.choice(merged_options)
    primary_group = sorted(primary_group, key=lambda item: item[2], reverse=True)
    chosen = primary_group[:2]

    extras = [item for item in chamber_candidates if item[0] not in {chosen[0][0], chosen[1][0]}]
    random.shuffle(extras)
    extra_limit = random.randint(0, min(2, len(extras)))
    chosen.extend(extras[:extra_limit])

    preferred = random.sample(list(MARKER_COLORS), 3)
    target_winners = [preferred[0], preferred[1]]
    while len(target_winners) < len(chosen):
        target_winners.append(random.choice(MARKER_COLORS))

    out = grid.copy()
    tie_used = False
    want_tie = random.random() < 0.4 and len(chosen) >= 3

    for index, ((chamber_id, candidates, _area, _path_group), winner) in enumerate(zip(chosen, target_winners)):
        max_needed = 3 if index < 2 else 4
        min_needed = 2
        cells = None
        for needed in range(max_needed, min_needed - 1, -1):
            cells = _spaced_sample(candidates, needed)
            if cells is not None:
                break
        assert cells is not None
        cells.sort()
        other_colors = [color for color in MARKER_COLORS if color != winner]
        do_tie = want_tie and not tie_used and len(cells) >= 2 and index >= 2
        if do_tie:
            loser = random.choice(other_colors)
            out[cells[0]] = winner
            out[cells[1]] = loser
            tie_used = True
            continue

        if index < 2:
            marker_total = len(cells)
            layout = [winner] * marker_total
        else:
            marker_total = random.randint(2, len(cells))
            minority_count = 1 if marker_total >= 3 else 0
            layout = [winner] * (marker_total - minority_count)
            if minority_count:
                layout.append(random.choice(other_colors))
            random.shuffle(layout)
            layout[0] = winner
        for (r, c), color in zip(cells[:marker_total], layout):
            out[r, c] = color

    return out


def _fill_under_variant(grid: np.ndarray, *, blue_walls: bool, grey_walls: bool) -> np.ndarray:
    base = np.asarray(grid, dtype=int).copy()
    if not grey_walls:
        base[base == GREY] = BLACK
    wall_mask = np.zeros(base.shape, dtype=bool)
    if blue_walls:
        wall_mask |= base == BLUE
    if grey_walls:
        wall_mask |= base == GREY
    return _fill_from_traced(base, wall_mask=wall_mask)


def _validate_case(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        traced = _trace_ray(grid)
        out = main(grid)
        _mirrors, mirror_comps = _mirror_orientations(grid)
    except AssertionError:
        return False

    grey_comps = _components_mask(grid == GREY, connectivity=4)
    if not (2 <= len(mirror_comps) <= 3 and 2 <= len(grey_comps) <= 3):
        return False
    for comp in grey_comps:
        rows = {int(r) for r, _ in comp}
        cols = {int(c) for _, c in comp}
        if len(comp) < 2 or len(comp) > 5:
            return False
        if len(rows) != 1 and len(cols) != 1:
            return False

    labels, comps = _label_black_components(traced)
    if len(comps) < 3:
        return False
    if np.array_equal(out, traced):
        return False

    markers = _markers_by_chamber(traced, labels)
    winners = []
    for chamber_id in range(1, len(comps) + 1):
        fill = _choose_fill_color(markers.get(chamber_id, []))
        if fill is not None:
            winners.append(fill)
    if len(winners) < 2 or len(set(winners)) < 2:
        return False

    try:
        no_ray = _fill_under_variant(grid, blue_walls=False, grey_walls=True)
        if np.array_equal(no_ray, out):
            return False
    except AssertionError:
        pass

    try:
        no_grey = _fill_under_variant(traced, blue_walls=True, grey_walls=False)
        if np.array_equal(no_grey, out):
            return False
    except AssertionError:
        pass

    traced_black = np.count_nonzero(traced == BLACK)
    out_black = np.count_nonzero(out == BLACK)
    if out_black >= traced_black:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, _path = _build_path_case()
            grid = _assign_markers(grid)
            grid = _transform(grid)
        except AssertionError:
            continue
        if _validate_case(grid):
            return grid
    raise AssertionError("failed to generate a valid non-degenerate puzzle instance")
