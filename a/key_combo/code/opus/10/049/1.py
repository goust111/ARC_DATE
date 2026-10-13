from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 1
GREY = 5
MARKER_PALETTE = (2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _canonical_shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    out = []
    for rr, cc in (
        (arr[:, 0], arr[:, 1]),
        (arr[:, 0], -arr[:, 1]),
        (-arr[:, 0], arr[:, 1]),
        (-arr[:, 0], -arr[:, 1]),
        (arr[:, 1], arr[:, 0]),
        (arr[:, 1], -arr[:, 0]),
        (-arr[:, 1], arr[:, 0]),
        (-arr[:, 1], -arr[:, 0]),
    ):
        pts = np.stack([rr, cc], axis=1)
        out.append(_normalize(pts))
    return min(out)


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.asarray(coords, dtype=int) + np.array([dr, dc], dtype=int)


def _min_sqdist_point_to_fragment(cell: tuple[int, int], coords: np.ndarray) -> int:
    diffs = coords - np.array(cell, dtype=int)
    return int((diffs * diffs).sum(axis=1).min())


def _marker_choice(coords: np.ndarray, markers: list[tuple[int, int, int]]) -> tuple[int, int]:
    scored = []
    for idx, (r, c, _color) in enumerate(markers):
        scored.append((_min_sqdist_point_to_fragment((r, c), coords), int(r), int(c), idx))
    scored.sort()
    return int(scored[0][3]), int(scored[0][0])


def _simulate_fragments(
    fragments: list[np.ndarray],
    shape: tuple[int, int],
    *,
    reverse_row_tie: bool = False,
) -> tuple[list[np.ndarray], list[tuple[str, int]]]:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, 0] = True

    order = []
    for idx, coords in enumerate(fragments):
        r0, c0, _r1, _c1 = _bbox(coords)
        row_key = -r0 if reverse_row_tie else r0
        order.append((c0, row_key, idx))
    order.sort()

    final = [None] * len(fragments)
    supports: list[tuple[str, int]] = [("wall", -1)] * len(fragments)

    for _c0, _row_key, idx in order:
        coords = np.asarray(fragments[idx], dtype=int).copy()
        while True:
            nxt = coords.copy()
            nxt[:, 1] -= 1
            if np.any(nxt[:, 1] < 0):
                break
            if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt

        blocker = ("wall", -1)
        for r, c in coords:
            if c == 1:
                blocker = ("wall", -1)
                break
            if occupied[r, c - 1]:
                hit = None
                for other_idx, other in enumerate(final):
                    if other_idx == idx or other is None:
                        continue
                    if np.any((other[:, 0] == r) & (other[:, 1] == c - 1)):
                        hit = other_idx
                        break
                if hit is not None:
                    blocker = ("fragment", int(hit))
                    break

        final[idx] = coords
        occupied[coords[:, 0], coords[:, 1]] = True
        supports[idx] = blocker

    return [np.asarray(x, dtype=int) for x in final], supports


def _render_output(
    shape: tuple[int, int],
    final_fragments: list[np.ndarray],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[:, 0] = WALL
    for coords in final_fragments:
        marker_idx, _best = _marker_choice(coords, markers)
        color = int(markers[marker_idx][2])
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _solve_from_objects(
    shape: tuple[int, int],
    fragments: list[np.ndarray],
    markers: list[tuple[int, int, int]],
    *,
    reverse_row_tie: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[str, int]]]:
    final_fragments, supports = _simulate_fragments(
        fragments,
        shape,
        reverse_row_tie=reverse_row_tie,
    )
    return _render_output(shape, final_fragments, markers), final_fragments, supports


def _parse_grid(grid: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 4
    assert np.all(g[:, 0] == WALL), "expected a complete blue wall in the leftmost column"
    assert np.count_nonzero(g == WALL) == h, "blue should appear only as the wall"

    fragments = _components(g == GREY)
    assert fragments, "expected grey fragments"

    markers: list[tuple[int, int, int]] = []
    seen_colors = set()
    for r in range(h):
        for c in range(1, w):
            color = int(g[r, c])
            if color in (BG, WALL, GREY):
                continue
            assert color not in seen_colors, "marker colors must be unique"
            seen_colors.add(color)
            markers.append((r, c, color))

    assert markers, "expected marker pixels"
    return fragments, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop grey fragments left against the blue wall, then recolor each settled fragment by its closest marker.

    Input structure
    - Black background fills most cells.
    - The full leftmost column is a blue edge wall.
    - Grey pixels form several 4-connected fragments, and every other non-black non-blue pixel is an isolated colored marker.

    Rule
    - Slide every grey fragment rigidly left toward the wall.
    - Settle fragments by increasing leftmost column; if two fragments start equally close to the wall, settle the one with the smaller top row first.
    - During motion, only the wall, the grid boundary, and already settled fragments block movement; markers stay fixed but do not block.
    - After all fragments settle, recolor each fragment with the color of the marker whose pixel has the smallest squared Euclidean distance to that fragment. Break marker ties by smaller row, then smaller column.
    - Erase all markers to black and keep the blue wall unchanged.

    Key insight
    - The marker choice must be evaluated after the gravity step, because the nearest landmark can change once the fragments stack against the wall.
    """
    g = np.asarray(grid, dtype=int)
    fragments, markers = _parse_grid(g)
    out, _final_fragments, _supports = _solve_from_objects(g.shape, fragments, markers)
    return out


def _random_shape(rng: random.Random) -> np.ndarray:
    area = rng.randint(3, 7)
    cells = {(0, 0)}
    target_h = rng.randint(2, 4)
    target_w = rng.randint(2, 4)

    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cells:
                    continue
                if nr < 0 or nc < 0 or nr >= target_h or nc >= target_w:
                    continue
                frontier.append((nr, nc))
        if not frontier:
            return _random_shape(rng)
        cells.add(rng.choice(frontier))
    return np.array(sorted(cells), dtype=int)


def _generate_shapes(count: int, rng: random.Random) -> list[np.ndarray]:
    shapes = []
    keys = set()
    while len(shapes) < count:
        shape = _random_shape(rng)
        key = _canonical_shape_key(shape)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(np.array(sorted(key), dtype=int))
    return shapes


def _mark_blocked(mask: np.ndarray, coords: np.ndarray) -> None:
    h, w = mask.shape
    for r, c in coords:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _place_random_fragments(
    shapes: list[np.ndarray],
    shape: tuple[int, int],
    rng: random.Random,
    *,
    want_order_tie: bool,
) -> list[np.ndarray] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, 0] = True
    placed: list[np.ndarray | None] = [None] * len(shapes)

    indices = list(range(len(shapes)))
    rng.shuffle(indices)

    if want_order_tie and len(shapes) >= 2:
        i0, i1 = indices[:2]
        s0, s1 = shapes[i0], shapes[i1]
        h0, w0 = _bbox(s0)[2] + 1, _bbox(s0)[3] + 1
        h1, w1 = _bbox(s1)[2] + 1, _bbox(s1)[3] + 1
        for _ in range(250):
            c0 = rng.randint(4, max(4, w - max(w0, w1) - 2))
            r0 = rng.randint(0, h - h0)
            r1 = rng.randint(0, h - h1)
            coords0 = _translate(s0, r0, c0)
            coords1 = _translate(s1, r1, c0)
            rows0 = set(map(int, coords0[:, 0]))
            rows1 = set(map(int, coords1[:, 0]))
            if not (rows0 & rows1):
                continue
            if np.any(blocked[coords0[:, 0], coords0[:, 1]]):
                continue
            if np.any(blocked[coords1[:, 0], coords1[:, 1]]):
                continue
            tmp = blocked.copy()
            _mark_blocked(tmp, coords0)
            if np.any(tmp[coords1[:, 0], coords1[:, 1]]):
                continue
            placed[i0] = coords0
            placed[i1] = coords1
            _mark_blocked(blocked, coords0)
            _mark_blocked(blocked, coords1)
            break
        if placed[i0] is None or placed[i1] is None:
            return None

    for idx in range(len(shapes)):
        if placed[idx] is not None:
            continue
        shp = shapes[idx]
        sh, sw = _bbox(shp)[2] + 1, _bbox(shp)[3] + 1
        ok = False
        for _ in range(250):
            r0 = rng.randint(0, h - sh)
            c0 = rng.randint(3, w - sw - 1)
            coords = _translate(shp, r0, c0)
            if np.any(blocked[coords[:, 0], coords[:, 1]]):
                continue
            placed[idx] = coords
            _mark_blocked(blocked, coords)
            ok = True
            break
        if not ok:
            return None

    return [np.asarray(coords, dtype=int) for coords in placed]


def _available_marker_cells(
    shape: tuple[int, int],
    fragments: list[np.ndarray],
    final_fragments: list[np.ndarray],
) -> list[tuple[int, int]]:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, 0] = True
    for coords in fragments:
        _mark_blocked(blocked, coords)
    for coords in final_fragments:
        blocked[coords[:, 0], coords[:, 1]] = True
    cells = []
    for r in range(h):
        for c in range(1, w):
            if not blocked[r, c]:
                cells.append((r, c))
    return cells


def _marker_spacing_ok(cells: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = candidate
    return all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in cells)


def _candidate_marker_cells(
    frag_idx: int,
    final_fragments: list[np.ndarray],
    available: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    target = final_fragments[frag_idx]
    out = []
    for cell in available:
        d_target = _min_sqdist_point_to_fragment(cell, target)
        if not (1 <= d_target <= 8):
            continue
        d_other = min(
            _min_sqdist_point_to_fragment(cell, other)
            for j, other in enumerate(final_fragments)
            if j != frag_idx
        )
        if d_target >= d_other:
            continue
        out.append((d_target, -(d_other - d_target), cell[1], abs(cell[0] - int(target[:, 0].mean())), cell))
    out.sort()
    return [cell for *_score, cell in out[:18]]


def _winners_by_fragment(
    fragments: list[np.ndarray],
    markers: list[tuple[int, int, int]],
) -> tuple[list[int], list[int]]:
    winner_indices = []
    winner_dists = []
    for coords in fragments:
        idx, dist = _marker_choice(coords, markers)
        winner_indices.append(idx)
        winner_dists.append(dist)
    return winner_indices, winner_dists


def _assign_primary_markers(
    fragments: list[np.ndarray],
    final_fragments: list[np.ndarray],
    colors: list[int],
    available: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int]] | None:
    candidates = [_candidate_marker_cells(i, final_fragments, available) for i in range(len(fragments))]
    if any(not pool for pool in candidates):
        return None

    order = sorted(range(len(fragments)), key=lambda idx: len(candidates[idx]))
    used: list[tuple[int, int]] = []
    chosen: list[tuple[int, int, int] | None] = [None] * len(fragments)

    def rec(pos: int) -> bool:
        if pos == len(order):
            markers = [chosen[i] for i in range(len(chosen))]
            assert all(m is not None for m in markers)
            markers2 = [m for m in markers if m is not None]
            winner_indices, _winner_dists = _winners_by_fragment(final_fragments, markers2)
            return all(markers2[winner_indices[i]][2] == colors[i] for i in range(len(fragments)))

        frag_idx = order[pos]
        pool = candidates[frag_idx][:]
        rng.shuffle(pool)
        pool = sorted(
            pool,
            key=lambda cell: (
                _min_sqdist_point_to_fragment(cell, final_fragments[frag_idx]),
                cell[1],
                cell[0],
            ),
        )
        for cell in pool[:10]:
            if not _marker_spacing_ok(used, cell):
                continue
            chosen[frag_idx] = (cell[0], cell[1], colors[frag_idx])
            used.append(cell)
            if rec(pos + 1):
                return True
            used.pop()
            chosen[frag_idx] = None
        return False

    if not rec(0):
        return None
    return [chosen[i] for i in range(len(chosen)) if chosen[i] is not None]


def _add_extra_tie_marker(
    initial_fragments: list[np.ndarray],
    final_fragments: list[np.ndarray],
    markers: list[tuple[int, int, int]],
    extra_color: int,
    available: list[tuple[int, int]],
    rng: random.Random,
    *,
    same_row_tie: bool,
) -> list[tuple[int, int, int]] | None:
    used = [(r, c) for r, c, _color in markers]
    winner_indices, winner_dists = _winners_by_fragment(final_fragments, markers)
    frag_order = list(range(len(final_fragments)))
    rng.shuffle(frag_order)

    for frag_idx in frag_order:
        win_marker = markers[winner_indices[frag_idx]]
        desired = winner_dists[frag_idx]
        pool = []
        for cell in available:
            if not _marker_spacing_ok(used, cell):
                continue
            if _min_sqdist_point_to_fragment(cell, final_fragments[frag_idx]) != desired:
                continue
            if same_row_tie and cell[0] != win_marker[0]:
                continue
            if not same_row_tie and cell[0] == win_marker[0]:
                continue
            if (cell[0], cell[1]) <= (win_marker[0], win_marker[1]):
                continue
            pool.append(cell)

        rng.shuffle(pool)
        for cell in pool:
            trial = markers + [(cell[0], cell[1], extra_color)]
            trial_winners, _trial_dists = _winners_by_fragment(final_fragments, trial)
            if any(trial[trial_winners[i]][2] != markers[winner_indices[i]][2] for i in range(len(final_fragments))):
                continue
            best = min(
                (_min_sqdist_point_to_fragment((r, c), final_fragments[frag_idx]), r, c)
                for r, c, _color in trial
            )
            tie_count = sum(
                (_min_sqdist_point_to_fragment((r, c), final_fragments[frag_idx]), r, c)[0] == best[0]
                for r, c, _color in trial
            )
            if tie_count >= 2:
                return trial
    return None


def _add_extra_change_or_distractor(
    initial_fragments: list[np.ndarray],
    final_fragments: list[np.ndarray],
    markers: list[tuple[int, int, int]],
    extra_color: int,
    available: list[tuple[int, int]],
    rng: random.Random,
    *,
    require_change: bool,
) -> list[tuple[int, int, int]] | None:
    used = [(r, c) for r, c, _color in markers]
    initial_winners_before, _ = _winners_by_fragment(initial_fragments, markers)
    final_winners_before, _ = _winners_by_fragment(final_fragments, markers)
    if any(
        markers[initial_winners_before[i]][2] != markers[final_winners_before[i]][2]
        for i in range(len(final_fragments))
    ):
        require_change = False

    scored = []
    for cell in available:
        if not _marker_spacing_ok(used, cell):
            continue
        min_final = min(_min_sqdist_point_to_fragment(cell, frag) for frag in final_fragments)
        min_initial = min(_min_sqdist_point_to_fragment(cell, frag) for frag in initial_fragments)
        scored.append((-(min_final - min_initial), -min_final, min_initial, cell))
    rng.shuffle(scored)
    scored.sort()

    fallback = None
    for _score0, _score1, _score2, cell in scored[:80]:
        trial = markers + [(cell[0], cell[1], extra_color)]
        trial_initial_winners, _ = _winners_by_fragment(initial_fragments, trial)
        trial_final_winners, _ = _winners_by_fragment(final_fragments, trial)
        final_same = all(
            trial[trial_final_winners[i]][2] == markers[final_winners_before[i]][2]
            for i in range(len(final_fragments))
        )
        if not final_same:
            continue
        if fallback is None:
            fallback = trial
        if any(trial[trial_initial_winners[i]][2] != trial[trial_final_winners[i]][2] for i in range(len(final_fragments))):
            return trial

    if require_change:
        return None
    return fallback


def _build_input_grid(
    shape: tuple[int, int],
    fragments: list[np.ndarray],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[:, 0] = WALL
    for r, c, color in markers:
        assert grid[r, c] == BG
        grid[r, c] = color
    for coords in fragments:
        assert not np.any(grid[coords[:, 0], coords[:, 1]] != BG)
        grid[coords[:, 0], coords[:, 1]] = GREY
    return grid


def _semantic_checks(
    grid: np.ndarray,
    fragments: list[np.ndarray],
    final_fragments: list[np.ndarray],
    markers: list[tuple[int, int, int]],
    *,
    want_order_tie: bool,
    want_distance_tie: bool,
) -> bool:
    h, _w = grid.shape
    if not np.all(grid[:, 0] == WALL):
        return False
    if np.count_nonzero(grid == WALL) != h:
        return False

    if len(fragments) not in (4, 5):
        return False
    if len(markers) not in (4, 5):
        return False
    if len(fragments) > len(markers):
        return False

    if len({_canonical_shape_key(coords) for coords in fragments}) != len(fragments):
        return False

    marker_cells = {(r, c) for r, c, _color in markers}
    if len(marker_cells) != len(markers):
        return False
    if len({color for _r, _c, color in markers}) != len(markers):
        return False

    for r, c, _color in markers:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == GREY:
                return False

    supports = _solve_from_objects(grid.shape, fragments, markers)[2]
    if not any(kind == "fragment" for kind, _idx in supports):
        return False

    initial_winners, _ = _winners_by_fragment(fragments, markers)
    final_winners, _final_dists = _winners_by_fragment(final_fragments, markers)
    if not any(markers[initial_winners[i]][2] != markers[final_winners[i]][2] for i in range(len(fragments))):
        return False

    if want_order_tie:
        min_cols = [_bbox(coords)[1] for coords in fragments]
        if len(min_cols) == len(set(min_cols)):
            return False
        normal = _solve_from_objects(grid.shape, fragments, markers)[0]
        reversed_tie = _solve_from_objects(
            grid.shape,
            fragments,
            markers,
            reverse_row_tie=True,
        )[0]
        if np.array_equal(normal, reversed_tie):
            return False

    if want_distance_tie:
        saw_tie = False
        for frag_idx, coords in enumerate(final_fragments):
            scored = sorted(
                (_min_sqdist_point_to_fragment((r, c), coords), r, c, color)
                for r, c, color in markers
            )
            if len(scored) >= 2 and scored[0][0] == scored[1][0]:
                saw_tie = True
                break
        if not saw_tie:
            return False

    out = main(grid)
    expected, _final2, _supports2 = _solve_from_objects(grid.shape, fragments, markers)
    if not np.array_equal(out, expected):
        return False

    if np.any(out == GREY):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1400):
        h = rng.randint(14, 18)
        w = rng.randint(14, 18)

        n_fragments = 5 if rng.random() < 0.30 else 4
        if n_fragments == 5:
            n_markers = 5
        else:
            n_markers = 5 if rng.random() < 0.70 else 4

        want_order_tie = rng.random() < 0.35
        want_distance_tie = n_fragments == 4 and n_markers == 5 and rng.random() < 0.45
        same_row_tie = rng.random() < 0.5

        shapes = _generate_shapes(n_fragments, rng)
        fragments = _place_random_fragments(
            shapes,
            (h, w),
            rng,
            want_order_tie=want_order_tie,
        )
        if fragments is None:
            continue

        final_fragments, supports = _simulate_fragments(fragments, (h, w))
        if not any(kind == "fragment" for kind, _idx in supports):
            continue

        colors = rng.sample(list(MARKER_PALETTE), n_markers)
        primary_colors = colors[:n_fragments]
        extra_colors = colors[n_fragments:]
        available = _available_marker_cells((h, w), fragments, final_fragments)

        markers = _assign_primary_markers(
            fragments,
            final_fragments,
            primary_colors,
            available,
            rng,
        )
        if markers is None:
            continue

        if extra_colors:
            extra_color = extra_colors[0]
            if want_distance_tie:
                tied = _add_extra_tie_marker(
                    fragments,
                    final_fragments,
                    markers,
                    extra_color,
                    available,
                    rng,
                    same_row_tie=same_row_tie,
                )
                if tied is None:
                    continue
                markers = tied
                initial_winners, _ = _winners_by_fragment(fragments, markers)
                final_winners, _ = _winners_by_fragment(final_fragments, markers)
                if not any(markers[initial_winners[i]][2] != markers[final_winners[i]][2] for i in range(len(fragments))):
                    continue
            else:
                with_extra = _add_extra_change_or_distractor(
                    fragments,
                    final_fragments,
                    markers,
                    extra_color,
                    available,
                    rng,
                    require_change=True,
                )
                if with_extra is None:
                    continue
                markers = with_extra
        else:
            initial_winners, _ = _winners_by_fragment(fragments, markers)
            final_winners, _ = _winners_by_fragment(final_fragments, markers)
            if not any(markers[initial_winners[i]][2] != markers[final_winners[i]][2] for i in range(len(fragments))):
                continue

        if len(markers) != n_markers:
            continue

        grid = _build_input_grid((h, w), fragments, markers)
        if not _semantic_checks(
            grid,
            fragments,
            final_fragments,
            markers,
            want_order_tie=want_order_tie,
            want_distance_tie=want_distance_tie,
        ):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
