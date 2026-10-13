from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_regions(grid: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=int)
    regions: dict[int, list[tuple[int, int]]] = {}
    next_id = 1

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == GREY or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if int(grid[nr, nc]) == GREY or labels[nr, nc] != 0:
                        continue
                    labels[nr, nc] = next_id
                    q.append((nr, nc))
            regions[next_id] = cells
            next_id += 1

    return labels, regions


def _touches_grey(grid: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
            return True
    return False


def _cell_targets(grid: np.ndarray, labels: np.ndarray, rid: int, r: int, c: int) -> set[int]:
    targets = set()
    for dr, dc in DIR4:
        wr = r + dr
        wc = c + dc
        tr = r + 2 * dr
        tc = c + 2 * dc
        if not _in_bounds(grid.shape, wr, wc) or not _in_bounds(grid.shape, tr, tc):
            continue
        if int(grid[wr, wc]) != GREY or int(grid[tr, tc]) == GREY:
            continue
        tgt = int(labels[tr, tc])
        if tgt != 0 and tgt != rid:
            targets.add(tgt)
    return targets


def _region_infos(
    grid: np.ndarray,
    regions: dict[int, list[tuple[int, int]]],
) -> dict[int, dict[str, object]]:
    infos: dict[int, dict[str, object]] = {}
    for rid, cells in regions.items():
        counts: Counter[int] = Counter()
        top: dict[int, tuple[int, int]] = {}
        votes: list[tuple[int, int, int]] = []
        black_cells = 0
        for r, c in cells:
            color = int(grid[r, c])
            if color == BLACK:
                black_cells += 1
                continue
            if color == GREY:
                continue
            counts[color] += 1
            pos = (r, c)
            if color not in top or pos < top[color]:
                top[color] = pos
            votes.append((r, c, color))
        infos[rid] = {
            "cells": cells,
            "counts": counts,
            "top": top,
            "votes": votes,
            "black_cells": black_cells,
        }
    return infos


def _minority_colors(counts: Counter[int]) -> set[int]:
    if not counts:
        return set()
    max_count = max(counts.values())
    return {color for color, count in counts.items() if count < max_count}


def _virtual_votes(
    grid: np.ndarray,
    labels: np.ndarray,
    infos: dict[int, dict[str, object]],
) -> tuple[dict[int, Counter[int]], list[tuple[int, int, int, tuple[int, int]]]]:
    extras: dict[int, Counter[int]] = defaultdict(Counter)
    details: list[tuple[int, int, int, tuple[int, int]]] = []
    for rid, info in infos.items():
        counts = info["counts"]
        minorities = _minority_colors(counts)
        for r, c, color in info["votes"]:
            if color not in minorities:
                continue
            targets = _cell_targets(grid, labels, rid, r, c)
            for tgt in sorted(targets):
                extras[tgt][color] += 1
                details.append((rid, tgt, color, (r, c)))
    return extras, details


def _winner_color(counts: Counter[int], top: dict[int, tuple[int, int]]) -> int:
    max_count = max(counts.values())
    tied = [color for color, count in counts.items() if count == max_count]
    return min(
        tied,
        key=lambda color: (
            top.get(color, (10**9, 10**9))[0],
            top.get(color, (10**9, 10**9))[1],
            color,
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each non-grey region using its final majority color.

    Majority is computed from the region's vote pixels. If there is a strict
    minority, its boundary-touching pixels each send one virtual vote of their
    color across the adjacent grey wall into the neighboring region. This
    propagation happens once, based only on the original in-region counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    labels, regions = _label_regions(g)
    infos = _region_infos(g, regions)

    for info in infos.values():
        assert info["counts"], "each region must contain at least one vote pixel"

    extras, _details = _virtual_votes(g, labels, infos)

    out = g.copy()
    for rid, info in infos.items():
        final_counts = Counter(info["counts"])
        final_counts.update(extras.get(rid, Counter()))
        fill_color = _winner_color(final_counts, info["top"])
        for r, c in info["cells"]:
            if int(g[r, c]) == BLACK:
                out[r, c] = fill_color
    return out


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _line_positions(sizes: list[int]) -> list[int]:
    pos = [0]
    cur = 0
    for size in sizes:
        cur += size + 1
        pos.append(cur)
    return pos


def _build_partition(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(2, 4)
    n_cols = rng.randint(2, 4)
    row_sizes = [rng.randint(4, 6) for _ in range(n_rows)]
    col_sizes = [rng.randint(4, 6) for _ in range(n_cols)]
    row_lines = _line_positions(row_sizes)
    col_lines = _line_positions(col_sizes)
    h = row_lines[-1] + 1
    w = col_lines[-1] + 1

    grid = np.full((h, w), BLACK, dtype=int)
    for rr in row_lines:
        grid[rr, :] = GREY
    for cc in col_lines:
        grid[:, cc] = GREY

    cell_count = n_rows * n_cols
    target_regions = rng.randint(3, min(cell_count, 7))
    dsu = _DSU(cell_count)
    edges: list[tuple[str, int, int]] = []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            edges.append(("right", r, c))
    for r in range(n_rows - 1):
        for c in range(n_cols):
            edges.append(("down", r, c))
    rng.shuffle(edges)

    current_regions = cell_count
    removed_edges: list[tuple[str, int, int]] = []
    for kind, r, c in edges:
        if current_regions <= target_regions:
            break
        a = r * n_cols + c
        b = a + (1 if kind == "right" else n_cols)
        if not dsu.union(a, b):
            continue
        removed_edges.append((kind, r, c))
        current_regions -= 1

    for kind, r, c in removed_edges:
        if kind == "right":
            wall_c = col_lines[c + 1]
            r0 = row_lines[r] + 1
            r1 = row_lines[r + 1] - 1
            grid[r0 : r1 + 1, wall_c] = BLACK
        else:
            wall_r = row_lines[r + 1]
            c0 = col_lines[c] + 1
            c1 = col_lines[c + 1] - 1
            grid[wall_r, c0 : c1 + 1] = BLACK

    labels, regions = _label_regions(grid)
    if len(regions) < 3:
        return None
    if np.any(labels == 0) and np.any(grid != GREY):
        return grid
    return grid


def _region_geometry(
    grid: np.ndarray,
    labels: np.ndarray,
    regions: dict[int, list[tuple[int, int]]],
) -> tuple[
    dict[int, list[tuple[int, int]]],
    dict[int, list[tuple[int, int]]],
    dict[tuple[int, int], list[tuple[int, int]]],
]:
    deep: dict[int, list[tuple[int, int]]] = {}
    all_cells: dict[int, list[tuple[int, int]]] = {}
    pair_cells: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    for rid, cells in regions.items():
        ordered = sorted(cells)
        all_cells[rid] = ordered
        deep[rid] = [cell for cell in ordered if not _touches_grey(grid, cell[0], cell[1])]
        for r, c in ordered:
            targets = _cell_targets(grid, labels, rid, r, c)
            if len(targets) == 1:
                tgt = next(iter(targets))
                pair_cells[(rid, tgt)].append((r, c))

    return deep, all_cells, pair_cells


def _pick_top(cells: list[tuple[int, int]], used: set[tuple[int, int]]) -> tuple[int, int] | None:
    for cell in cells:
        if cell not in used:
            return cell
    return None


def _pick_random(
    cells: list[tuple[int, int]],
    used: set[tuple[int, int]],
    count: int,
    rng: random.Random,
    after: tuple[int, int] | None = None,
) -> list[tuple[int, int]] | None:
    avail = [cell for cell in cells if cell not in used and (after is None or cell > after)]
    if len(avail) < count:
        return None
    return rng.sample(avail, count)


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _build_votes(
    base: np.ndarray,
    labels: np.ndarray,
    regions: dict[int, list[tuple[int, int]]],
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, object]] | None:
    deep, all_cells, pair_cells = _region_geometry(base, labels, regions)
    scenario = rng.choice(("flip", "stable", "stable_tie"))
    target_need = 4 if scenario == "stable" else 3

    candidate_pairs = [
        (src, tgt)
        for (src, tgt), cells in pair_cells.items()
        if len(cells) >= 1
        and len(all_cells[src]) >= 3
        and len(deep[tgt]) >= target_need
        and len(deep[src]) >= 1
    ]
    if not candidate_pairs:
        return None

    src, tgt = rng.choice(candidate_pairs)
    other_regions = [rid for rid in regions if rid not in {src, tgt}]
    tie_rid = rng.choice(other_regions) if other_regions and rng.random() < 0.5 else None
    tie_winner_first = rng.choice((True, False))

    transfer_color, src_major, tgt_major, tie_a, tie_b, spare = rng.sample(COLOR_POOL, 6)

    grid = base.copy()
    used: dict[int, set[tuple[int, int]]] = {rid: set() for rid in regions}

    src_boundary = rng.choice(pair_cells[(src, tgt)])
    used[src].add(src_boundary)
    _paint(grid, [src_boundary], transfer_color)

    src_major_cells = _pick_random(all_cells[src], used[src], 2, rng)
    if src_major_cells is None:
        return None
    used[src].update(src_major_cells)
    _paint(grid, src_major_cells, src_major)

    target_expected = tgt_major
    if scenario == "flip":
        top_b = _pick_top(deep[tgt], used[tgt])
        if top_b is None:
            return None
        used[tgt].add(top_b)
        later_a = _pick_random(deep[tgt], used[tgt], 2, rng, after=top_b)
        if later_a is None:
            return None
        used[tgt].update(later_a)
        _paint(grid, [top_b], transfer_color)
        _paint(grid, later_a, tgt_major)
        target_expected = transfer_color
    elif scenario == "stable_tie":
        top_a = _pick_top(deep[tgt], used[tgt])
        if top_a is None:
            return None
        used[tgt].add(top_a)
        b_cell = _pick_random(deep[tgt], used[tgt], 1, rng, after=top_a)
        if b_cell is None:
            return None
        used[tgt].update(b_cell)
        other_a = _pick_random(deep[tgt], used[tgt], 1, rng, after=top_a)
        if other_a is None:
            return None
        used[tgt].update(other_a)
        _paint(grid, [top_a], tgt_major)
        _paint(grid, b_cell, transfer_color)
        _paint(grid, other_a, tgt_major)
    else:
        major_cells = _pick_random(deep[tgt], used[tgt], 3, rng)
        if major_cells is None:
            return None
        used[tgt].update(major_cells)
        minor_cell = _pick_random(deep[tgt], used[tgt], 1, rng)
        if minor_cell is None:
            return None
        used[tgt].update(minor_cell)
        _paint(grid, major_cells, tgt_major)
        _paint(grid, minor_cell, transfer_color)

    for rid in regions:
        if rid in {src, tgt}:
            continue
        if rid == tie_rid:
            top_cell = _pick_top(deep[rid], used[rid])
            if top_cell is None:
                return None
            used[rid].add(top_cell)
            other = _pick_random(deep[rid], used[rid], 1, rng, after=top_cell)
            if other is None:
                return None
            used[rid].update(other)
            first_color = tie_a if tie_winner_first else tie_b
            second_color = tie_b if tie_winner_first else tie_a
            _paint(grid, [top_cell], first_color)
            _paint(grid, other, second_color)
            continue

        style = rng.choice(("solid", "solid", "mixed"))
        if style == "solid":
            color = rng.choice(COLOR_POOL)
            count = rng.randint(2, min(4, len(all_cells[rid]) - 1))
            cells = _pick_random(all_cells[rid], used[rid], count, rng)
            if cells is None:
                return None
            used[rid].update(cells)
            _paint(grid, cells, color)
        else:
            colors = rng.sample(COLOR_POOL, 2)
            major_count = rng.choice((2, 3))
            if len(all_cells[rid]) < major_count + 2 or len(deep[rid]) < 1:
                return None
            minor_cells = _pick_random(deep[rid], used[rid], 1, rng)
            if minor_cells is None:
                return None
            used[rid].update(minor_cells)
            major_cells = _pick_random(all_cells[rid], used[rid], major_count, rng)
            if major_cells is None:
                return None
            used[rid].update(major_cells)
            _paint(grid, major_cells, colors[0])
            _paint(grid, minor_cells, colors[1])

    meta = {
        "src": src,
        "tgt": tgt,
        "transfer_color": transfer_color,
        "src_major": src_major,
        "tgt_major": tgt_major,
        "target_expected": target_expected,
        "scenario": scenario,
        "tie_rid": tie_rid,
        "tie_expected": tie_a if tie_rid is not None and tie_winner_first else tie_b,
    }
    return grid, meta


def _validate_generated(grid: np.ndarray, meta: dict[str, object]) -> bool:
    labels, regions = _label_regions(grid)
    infos = _region_infos(grid, regions)
    if len(regions) < 3:
        return False
    for info in infos.values():
        counts = info["counts"]
        if sum(counts.values()) < 2:
            return False
        if len(counts) > 2:
            return False
        if info["black_cells"] < 1:
            return False

    extras, details = _virtual_votes(grid, labels, infos)
    if len(details) != 1:
        return False
    src, tgt, color, _cell = details[0]
    if src != meta["src"] or tgt != meta["tgt"] or color != meta["transfer_color"]:
        return False

    target_counts = Counter(infos[meta["tgt"]]["counts"])
    initial_target = _winner_color(target_counts, infos[meta["tgt"]]["top"])
    target_counts.update(extras.get(meta["tgt"], Counter()))
    final_target = _winner_color(target_counts, infos[meta["tgt"]]["top"])
    if final_target != meta["target_expected"]:
        return False
    if meta["scenario"] == "flip" and final_target == initial_target:
        return False
    if meta["scenario"] != "flip" and final_target != initial_target:
        return False

    if meta["tie_rid"] is not None:
        rid = meta["tie_rid"]
        tie_counts = Counter(infos[rid]["counts"])
        if len(tie_counts) != 2 or len(set(tie_counts.values())) != 1:
            return False
        tie_winner = _winner_color(tie_counts, infos[rid]["top"])
        if tie_winner != meta["tie_expected"]:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    base = _build_partition(rng)
    if base is None:
        return None
    labels, regions = _label_regions(base)
    built = _build_votes(base, labels, regions, rng)
    if built is None:
        return None
    grid, meta = built
    if not _validate_generated(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
