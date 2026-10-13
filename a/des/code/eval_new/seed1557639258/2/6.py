from __future__ import annotations

import heapq
import random
from collections import Counter

import numpy as np


PIPE = 5
TABLE_WIDTH = 6
GEN_ATTEMPTS = 400
MODE_CHOICES = ("h_segment", "h_cross_turn", "v_cross_turn")
COLOR_POOL = [c for c in range(10) if c != PIPE]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _table_candidate(
    grid: np.ndarray,
    bg: int,
    rows: tuple[int, int],
    *,
    from_left: bool,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    cols = list(range(w)) if from_left else list(range(w - 1, -1, -1))
    width = 0
    for c in cols:
        if int(grid[rows[0], c]) in (bg, PIPE) or int(grid[rows[1], c]) in (bg, PIPE):
            break
        width += 1

    if width < 2 or width % 2 != 0:
        return None

    if from_left:
        c0, c1 = 0, width - 1
    else:
        c0, c1 = w - width, w - 1

    block = grid[rows[0] : rows[1] + 1, c0 : c1 + 1]
    if bool(np.isin(block, [bg, PIPE]).any()):
        return None
    return int(rows[0]), int(rows[1]), int(c0), int(c1)


def _find_table_and_mix_map(
    grid: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, dict[frozenset[int], int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    if h >= 2:
        for rows, from_left in (
            ((0, 1), True),
            ((0, 1), False),
            ((h - 2, h - 1), True),
            ((h - 2, h - 1), False),
        ):
            cand = _table_candidate(grid, bg, rows, from_left=from_left)
            if cand is not None:
                candidates.append(cand)

    assert len(candidates) == 1, "expected exactly one corner mixing table"
    r0, r1, c0, c1 = candidates[0]

    table_mask = np.zeros(grid.shape, dtype=bool)
    table_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    table = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert table.shape[0] == 2 and table.shape[1] % 2 == 0

    mix_map: dict[frozenset[int], int] = {}
    for j in range(0, int(table.shape[1]), 2):
        a = int(table[0, j])
        b = int(table[0, j + 1])
        m0 = int(table[1, j])
        m1 = int(table[1, j + 1])
        assert m0 == m1, "each table entry must repeat the mixed color"
        assert a != b, "table entries must encode two different input colors"
        mix_map[frozenset((a, b))] = int(m0)
    return table_mask, mix_map


def _mix_colors(colors: list[int], mix_map: dict[frozenset[int], int]) -> int:
    uniq = list(dict.fromkeys(int(c) for c in colors))
    assert uniq, "need at least one arriving color"
    cur = uniq[0]
    for nxt in uniq[1:]:
        if nxt == cur:
            continue
        key = frozenset((int(cur), int(nxt)))
        assert key in mix_map, "encountered an unmapped color pair"
        cur = int(mix_map[key])
    return int(cur)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate endpoint colors through a grey pipe network and apply table-driven mixing when opposing signals meet.

    Input structure:
    - The background is the dominant color.
    - Grey (5) pixels form a 1-pixel-wide orthogonal pipe network, and a corner 2xN table
      stores color-pair mixing rules.
    - Some degree-1 pipe endpoints are replaced by colored source pixels.
    - Cross-shaped degree-4 junctions may connect horizontal and vertical pipe segments.

    Rule:
    - Each source sends its color inward through the adjacent pipe at one-cell-per-step speed.
    - A pipe cell takes the color of the earliest signal(s) reaching it; simultaneous different
      colors are replaced by the table's mixed color.
    - On an ordinary segment, propagation continues forward along the pipe.
    - At a degree-4 cross, a signal continues straight if the opposite side is free; if an
      opposite signal blocks that straight-through move, the signal turns onto the perpendicular branch.

    Key insight:
    - The final coloring is determined by synchronized arrival times on the pipe graph, with
      equal-time collisions producing a visible mixed color and equal-time collisions at crosses
      forcing the mixed signal to leave along the perpendicular branch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    table_mask, mix_map = _find_table_and_mix_map(g, bg)

    network_mask = (g != bg) & (~table_mask)
    assert bool(network_mask.any()), "expected a pipe network outside the table"

    network_cells = {tuple(map(int, rc)) for rc in np.argwhere(network_mask)}
    pipe_cells = {rc for rc in network_cells if int(g[rc]) == PIPE}
    source_cells = {rc for rc in network_cells if int(g[rc]) != PIPE}
    assert len(source_cells) >= 2, "expected at least two colored sources"

    def network_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = cell
        return [nb for nb in _neighbors4(r, c) if nb in network_cells]

    source_specs: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for src in sorted(source_cells):
        nbrs = network_neighbors(src)
        assert len(nbrs) == 1, "sources must sit at degree-1 endpoints"
        source_specs.append((src, int(g[src]), nbrs[0]))

    out = g.copy()
    first_time: dict[tuple[int, int], int] = {}
    scheduled_min: dict[tuple[int, int], int] = {}
    heap: list[tuple[int, int, int, int, int, int]] = []

    for src, color, nxt in source_specs:
        if nxt in source_cells:
            continue
        heapq.heappush(heap, (1, int(nxt[0]), int(nxt[1]), int(src[0]), int(src[1]), int(color)))
        scheduled_min[nxt] = min(scheduled_min.get(nxt, 10**9), 1)

    while heap:
        time, r, c, pr, pc, color = heapq.heappop(heap)
        cell = (int(r), int(c))
        if cell in first_time:
            continue

        batch = [(int(pr), int(pc), int(color))]
        while heap and (int(heap[0][0]), int(heap[0][1]), int(heap[0][2])) == (int(time), int(r), int(c)):
            _, _, _, pr2, pc2, color2 = heapq.heappop(heap)
            batch.append((int(pr2), int(pc2), int(color2)))

        result_color = _mix_colors([item[2] for item in batch], mix_map)
        out[cell] = int(result_color)
        first_time[cell] = int(time)

        pipe_nbrs = [nb for nb in network_neighbors(cell) if nb in pipe_cells]
        if not pipe_nbrs:
            continue

        incoming = {(int(pr0), int(pc0)) for pr0, pc0, _ in batch}
        next_targets: set[tuple[int, int]] = set()

        if len(pipe_nbrs) == 4:
            for prev_r, prev_c, _ in batch:
                dr = int(r - prev_r)
                dc = int(c - prev_c)
                straight = (int(r + dr), int(c + dc))
                blocked = (
                    straight not in pipe_cells
                    or straight in incoming
                    or scheduled_min.get(straight, 10**9) <= int(time)
                )

                if not blocked:
                    if scheduled_min.get(straight, 10**9) >= int(time + 1):
                        next_targets.add(straight)
                    continue

                for nb in pipe_nbrs:
                    if nb == straight:
                        continue
                    if (prev_r, prev_c) in pipe_cells and nb == (prev_r, prev_c):
                        continue
                    if nb in incoming:
                        continue
                    if scheduled_min.get(nb, 10**9) < int(time + 1):
                        continue
                    next_targets.add(nb)
        else:
            for prev_r, prev_c, _ in batch:
                prev = (int(prev_r), int(prev_c))
                prev_pipe = prev if prev in pipe_cells else None
                candidates = [nb for nb in pipe_nbrs if nb != prev_pipe]
                candidates = [
                    nb
                    for nb in candidates
                    if nb not in incoming and scheduled_min.get(nb, 10**9) >= int(time + 1)
                ]
                if len(candidates) == 1:
                    next_targets.add(candidates[0])
                elif len(candidates) > 1:
                    next_targets.update(candidates)

        for nr, nc in sorted(next_targets):
            heapq.heappush(
                heap,
                (int(time + 1), int(nr), int(nc), int(r), int(c), int(result_color)),
            )
            scheduled_min[(int(nr), int(nc))] = min(
                scheduled_min.get((int(nr), int(nc)), 10**9),
                int(time + 1),
            )

    return out


def _sample_palette() -> tuple[int, list[int], dict[tuple[int, int], int]]:
    bg = int(random.choice(COLOR_POOL))
    pool = [c for c in COLOR_POOL if c != bg]
    chosen = random.sample(pool, 6)
    bases = [int(c) for c in chosen[:3]]
    mixes = [int(c) for c in chosen[3:]]
    pair_map = {
        tuple(sorted((bases[0], bases[1]))): mixes[0],
        tuple(sorted((bases[0], bases[2]))): mixes[1],
        tuple(sorted((bases[1], bases[2]))): mixes[2],
    }
    return int(bg), bases, pair_map


def _paint_table(
    grid: np.ndarray,
    bg: int,
    bases: list[int],
    pair_map: dict[tuple[int, int], int],
    corner: str,
) -> None:
    assert TABLE_WIDTH == 6
    pairs = list(pair_map.items())
    random.shuffle(pairs)

    top_row: list[int] = []
    bottom_row: list[int] = []
    for (a, b), mix in pairs:
        top_row.extend([int(a), int(b)])
        bottom_row.extend([int(mix), int(mix)])

    h, w = grid.shape
    if corner.startswith("T"):
        r0 = 0
    else:
        r0 = h - 2
    if corner.endswith("L"):
        c0 = 0
    else:
        c0 = w - TABLE_WIDTH

    grid[r0, c0 : c0 + TABLE_WIDTH] = np.array(top_row, dtype=int)
    grid[r0 + 1, c0 : c0 + TABLE_WIDTH] = np.array(bottom_row, dtype=int)


def _choose_spaced_columns(
    lo: int,
    hi: int,
    required: list[int],
    forbidden: set[int],
    total: int,
) -> list[int] | None:
    chosen = sorted(required)
    if any(c < lo or c > hi for c in chosen):
        return None
    if any(abs(a - b) < 3 for a, b in zip(chosen, chosen[1:])):
        return None

    while len(chosen) < total:
        candidates = [
            c
            for c in range(lo, hi + 1)
            if c not in forbidden and c not in chosen and all(abs(c - x) >= 3 for x in chosen)
        ]
        if not candidates:
            return None
        chosen.append(int(random.choice(candidates)))
        chosen.sort()

    return chosen


def _render_network(
    grid: np.ndarray,
    *,
    hr: int,
    left_c: int,
    right_c: int,
    columns: list[int],
    top_r: int,
    bottom_r: int,
) -> None:
    grid[hr, left_c : right_c + 1] = PIPE
    for vc in columns:
        grid[top_r : bottom_r + 1, vc] = PIPE


def _build_scene() -> np.ndarray | None:
    mode = random.choice(MODE_CHOICES)
    h = random.randint(13, 24)
    w = random.randint(13, 28)
    bg, bases, pair_map = _sample_palette()
    src_colors = random.sample(bases, 2)
    corner = random.choice(["TL", "TR", "BL", "BR"])

    grid = np.full((h, w), bg, dtype=int)
    _paint_table(grid, bg, bases, pair_map, corner)

    min_top = 3 if corner.startswith("T") else 1
    max_bottom = h - 4 if corner.startswith("B") else h - 2
    if max_bottom - min_top < 6:
        return None

    left_bound = 1
    right_bound = w - 2
    if right_bound - left_bound < 8:
        return None

    if mode == "v_cross_turn":
        hr_lo = min_top + 2
        hr_hi = max_bottom - 2
        if hr_lo > hr_hi:
            return None
        hr = random.randint(hr_lo, hr_hi)
        max_d = min(hr - min_top, max_bottom - hr)
        if max_d < 2:
            return None
        d = random.randint(2, max_d)
        top_r = hr - d
        bottom_r = hr + d
        left_c = random.randint(1, 3)
        right_c = random.randint(max(left_c + 8, w - 5), w - 2)
        if right_c >= w:
            return None
        vc = random.randint(left_c + 3, right_c - 3)
        total_cols = random.randint(1, 3)
        columns = _choose_spaced_columns(
            left_c + 2,
            right_c - 2,
            [vc],
            set(),
            total_cols,
        )
        if columns is None:
            return None
        _render_network(
            grid,
            hr=hr,
            left_c=left_c,
            right_c=right_c,
            columns=columns,
            top_r=top_r,
            bottom_r=bottom_r,
        )
        grid[top_r, vc] = int(src_colors[0])
        grid[bottom_r, vc] = int(src_colors[1])
    else:
        hr_lo = min_top + 2
        hr_hi = max_bottom - 2
        if hr_lo > hr_hi:
            return None
        hr = random.randint(hr_lo, hr_hi)
        top_r = random.randint(min_top, hr - 2)
        bottom_r = random.randint(hr + 2, max_bottom)

        center_lo = 4
        center_hi = w - 5
        if center_lo > center_hi:
            return None
        center = random.randint(center_lo, center_hi)
        max_d = min(center - left_bound, right_bound - center)
        if max_d < 4:
            return None
        d = random.randint(4, max_d)
        left_c = center - d
        right_c = center + d

        total_cols = random.randint(1, 3)
        if mode == "h_cross_turn":
            columns = _choose_spaced_columns(
                left_c + 2,
                right_c - 2,
                [center],
                set(),
                total_cols,
            )
        else:
            columns = _choose_spaced_columns(
                left_c + 2,
                right_c - 2,
                [],
                {center - 1, center, center + 1},
                total_cols,
            )
        if columns is None:
            return None

        _render_network(
            grid,
            hr=hr,
            left_c=left_c,
            right_c=right_c,
            columns=columns,
            top_r=top_r,
            bottom_r=bottom_r,
        )
        grid[hr, left_c] = int(src_colors[0])
        grid[hr, right_c] = int(src_colors[1])

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    mix_colors = set(pair_map.values())
    if not any(int(color) in mix_colors for color in np.unique(out)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-signal puzzle")
