from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400

TWO_COLOR_COUNTS = {
    3: ((2, 1),),
    4: ((3, 1),),
    5: ((3, 2), (4, 1)),
    6: ((4, 2), (5, 1)),
    7: ((4, 3), (5, 2)),
}
THREE_COLOR_COUNTS = {
    6: ((3, 2, 1),),
    7: ((4, 2, 1),),
}


def _detect_walls(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = map(int, g.shape)
    row_colors = [int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])]
    col_colors = [int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])]

    candidates = [
        color
        for color in set(row_colors) & set(col_colors)
        if row_colors.count(color) >= 2 and col_colors.count(color) >= 2
    ]
    assert candidates, "expected a separator color spanning rows and columns"
    wall = max(candidates, key=lambda color: row_colors.count(color) + col_colors.count(color))

    row_lines = [r for r in range(h) if np.all(g[r, :] == wall)]
    col_lines = [c for c in range(w) if np.all(g[:, c] == wall)]
    assert row_lines[0] == 0 and row_lines[-1] == h - 1, "expected outer horizontal border"
    assert col_lines[0] == 0 and col_lines[-1] == w - 1, "expected outer vertical border"
    assert all(b - a >= 2 for a, b in zip(row_lines, row_lines[1:], strict=False))
    assert all(b - a >= 2 for a, b in zip(col_lines, col_lines[1:], strict=False))
    return int(wall), row_lines, col_lines


def _region_defs(row_lines: list[int], col_lines: list[int]) -> tuple[list[dict[str, int]], list[list[int]], int, int]:
    nr = len(row_lines) - 1
    nc = len(col_lines) - 1
    regions: list[dict[str, int]] = []
    neighbors = [[] for _ in range(nr * nc)]

    for i in range(nr):
        for j in range(nc):
            rid = i * nc + j
            regions.append(
                {
                    "id": rid,
                    "ri": i,
                    "ci": j,
                    "top": row_lines[i],
                    "bottom": row_lines[i + 1],
                    "left": col_lines[j],
                    "right": col_lines[j + 1],
                }
            )
            if i > 0:
                neighbors[rid].append((i - 1) * nc + j)
            if i + 1 < nr:
                neighbors[rid].append((i + 1) * nc + j)
            if j > 0:
                neighbors[rid].append(i * nc + (j - 1))
            if j + 1 < nc:
                neighbors[rid].append(i * nc + (j + 1))

    return regions, neighbors, nr, nc


def _ranked_votes(patch: np.ndarray, wall: int) -> tuple[list[int], int]:
    counts: Counter[int] = Counter()
    order: dict[int, int] = {}
    next_order = 0
    for value in patch.ravel():
        color = int(value)
        if color in (BLACK, wall):
            continue
        counts[color] += 1
        if color not in order:
            order[color] = next_order
            next_order += 1

    assert counts, "each region must contain at least one vote pixel"
    ranked = sorted(counts, key=lambda color: (-counts[color], order[color]))
    return [int(color) for color in ranked], int(sum(counts.values()))


def _weaker_than(a: int, b: int, totals: list[int]) -> bool:
    return totals[a] < totals[b] or (totals[a] == totals[b] and a > b)


def _solve_region_colors(rankings: list[list[int]], totals: list[int], neighbors: list[list[int]]) -> list[int]:
    choice_idx = [0 for _ in rankings]
    weak_to_strong = sorted(range(len(rankings)), key=lambda rid: (totals[rid], -rid))
    max_changes = sum(len(options) - 1 for options in rankings)
    changes_used = 0

    while True:
        changed = False
        for rid in weak_to_strong:
            while True:
                current = rankings[rid][choice_idx[rid]]
                losing_conflict = False
                for nid in neighbors[rid]:
                    if rankings[nid][choice_idx[nid]] != current:
                        continue
                    if _weaker_than(rid, nid, totals):
                        losing_conflict = True
                        break
                if not losing_conflict:
                    break
                choice_idx[rid] += 1
                changes_used += 1
                assert choice_idx[rid] < len(rankings[rid]), "ran out of fallback colors for a region"
                assert changes_used <= max_changes, "conflict resolution exceeded fallback budget"
                changed = True

        has_conflict = False
        for rid, nbrs in enumerate(neighbors):
            for nid in nbrs:
                if nid <= rid:
                    continue
                if rankings[rid][choice_idx[rid]] == rankings[nid][choice_idx[nid]]:
                    has_conflict = True
                    break
            if has_conflict:
                break
        if not has_conflict:
            return [rankings[rid][choice_idx[rid]] for rid in range(len(rankings))]
        assert changed, "stalled with unresolved adjacent-color conflicts"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    wall, row_lines, col_lines = _detect_walls(g)
    regions, neighbors, _nr, _nc = _region_defs(row_lines, col_lines)

    rankings: list[list[int]] = []
    totals: list[int] = []
    for region in regions:
        patch = g[region["top"] + 1 : region["bottom"], region["left"] + 1 : region["right"]]
        assert patch.size > 0, "region interiors must be nonempty"
        assert not np.any(patch == wall), "separator color should not appear inside a region"
        ranked, total_votes = _ranked_votes(patch, wall)
        rankings.append(ranked)
        totals.append(total_votes)

    final_colors = _solve_region_colors(rankings, totals, neighbors)

    out = g.copy()
    for region, fill_color in zip(regions, final_colors, strict=False):
        patch = out[region["top"] + 1 : region["bottom"], region["left"] + 1 : region["right"]]
        patch[patch == BLACK] = int(fill_color)
    return out


def _build_layout() -> tuple[list[int], list[int], list[dict[str, int]], list[list[int]], int, int]:
    nr = random.randint(2, 3)
    nc = random.randint(2, 3)
    row_heights = [random.randint(4, 6) for _ in range(nr)]
    col_widths = [random.randint(4, 6) for _ in range(nc)]

    row_lines = [0]
    for height in row_heights:
        row_lines.append(row_lines[-1] + height + 1)
    col_lines = [0]
    for width in col_widths:
        col_lines.append(col_lines[-1] + width + 1)

    regions, neighbors, _, _ = _region_defs(row_lines, col_lines)
    return row_lines, col_lines, regions, neighbors, nr, nc


def _assign_final_colors(
    n_regions: int,
    neighbors: list[list[int]],
    palette: list[int],
    preassign: dict[int, int] | None = None,
    forbidden: dict[int, set[int]] | None = None,
) -> list[int] | None:
    preassign = {} if preassign is None else {int(k): int(v) for k, v in preassign.items()}
    forbidden = {} if forbidden is None else {int(k): {int(v) for v in vals} for k, vals in forbidden.items()}
    assigned: dict[int, int] = {}
    order = sorted(range(n_regions), key=lambda rid: (0 if rid in preassign else 1, -len(neighbors[rid]), rid))

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        rid = order[pos]
        blocked = forbidden.get(rid, set())
        if rid in preassign:
            choices = [preassign[rid]]
        else:
            choices = list(palette)
            random.shuffle(choices)
        for color in choices:
            if color in blocked:
                continue
            if any(assigned.get(nid) == color for nid in neighbors[rid]):
                continue
            assigned[rid] = int(color)
            if dfs(pos + 1):
                return True
            del assigned[rid]
        return False

    if dfs(0):
        return [assigned[rid] for rid in range(n_regions)]
    return None


def _default_votes(final_color: int, palette: list[int]) -> tuple[list[int], int]:
    if random.random() < 0.55:
        total = random.randint(3, 7)
        colors_used = 2
    else:
        total = random.randint(6, 7)
        colors_used = 3
    others = [color for color in palette if color != final_color]
    random.shuffle(others)
    return [int(final_color)] + others[: colors_used - 1], int(total)


def _counts_for(total_votes: int, n_colors: int) -> list[int]:
    if n_colors == 2:
        return list(random.choice(TWO_COLOR_COUNTS[total_votes]))
    if n_colors == 3:
        return list(random.choice(THREE_COLOR_COUNTS[total_votes]))
    raise AssertionError("regions must use two or three vote colors")


def _place_votes(
    grid: np.ndarray,
    region: dict[str, int],
    ranked_colors: list[int],
    total_votes: int,
) -> None:
    counts = _counts_for(total_votes, len(ranked_colors))
    cells = [
        (r, c)
        for r in range(region["top"] + 1, region["bottom"])
        for c in range(region["left"] + 1, region["right"])
    ]
    assert len(cells) > total_votes, "each region needs visible empty space after voting"
    positions = random.sample(cells, total_votes)
    random.shuffle(positions)

    offset = 0
    for color, count in zip(ranked_colors, counts, strict=False):
        for r, c in positions[offset : offset + count]:
            grid[r, c] = int(color)
        offset += count


def _corner_ids(nr: int, nc: int) -> list[int]:
    return sorted({0, nc - 1, (nr - 1) * nc, nr * nc - 1})


def _single_conflict_case(
    neighbors: list[list[int]],
    palette: list[int],
    nr: int,
    nc: int,
) -> tuple[list[int], dict[int, list[int]], dict[int, int]] | None:
    n_regions = nr * nc
    weaker = random.choice(_corner_ids(nr, nc))
    stronger = random.choice(neighbors[weaker])
    conflict_color, weaker_final = random.sample(palette, 2)

    forbidden: dict[int, set[int]] = {}
    for nid in neighbors[weaker]:
        if nid != stronger:
            forbidden.setdefault(nid, set()).add(conflict_color)

    final_colors = _assign_final_colors(
        n_regions=n_regions,
        neighbors=neighbors,
        palette=palette,
        preassign={stronger: conflict_color, weaker: weaker_final},
        forbidden=forbidden,
    )
    if final_colors is None:
        return None

    prefs: dict[int, list[int]] = {}
    totals: dict[int, int] = {}
    prefs[stronger] = [conflict_color] + [color for color in palette if color != conflict_color][:1]
    totals[stronger] = 7
    prefs[weaker] = [conflict_color, weaker_final]
    totals[weaker] = random.randint(3, 6)

    for rid in range(n_regions):
        if rid in prefs:
            continue
        pref, total = _default_votes(final_colors[rid], palette)
        prefs[rid] = pref
        totals[rid] = total
    return final_colors, prefs, totals


def _cascade_conflict_case(
    neighbors: list[list[int]],
    palette: list[int],
    nr: int,
    nc: int,
) -> tuple[list[int], dict[int, list[int]], dict[int, int]] | None:
    n_regions = nr * nc
    middle = random.choice(_corner_ids(nr, nc))
    first_neighbor, second_neighbor = random.sample(neighbors[middle], 2)
    first_color, second_color, middle_final = random.sample(palette, 3)

    final_colors = _assign_final_colors(
        n_regions=n_regions,
        neighbors=neighbors,
        palette=palette,
        preassign={
            first_neighbor: first_color,
            second_neighbor: second_color,
            middle: middle_final,
        },
    )
    if final_colors is None:
        return None

    prefs: dict[int, list[int]] = {}
    totals: dict[int, int] = {}
    prefs[first_neighbor] = [first_color] + [color for color in palette if color != first_color][:1]
    prefs[second_neighbor] = [second_color] + [color for color in palette if color != second_color][:1]
    totals[first_neighbor] = 7
    totals[second_neighbor] = 7
    prefs[middle] = [first_color, second_color, middle_final]
    totals[middle] = 6

    for rid in range(n_regions):
        if rid in prefs:
            continue
        pref, total = _default_votes(final_colors[rid], palette)
        prefs[rid] = pref
        totals[rid] = total
    return final_colors, prefs, totals


def _no_conflict_case(
    neighbors: list[list[int]],
    palette: list[int],
    n_regions: int,
) -> tuple[list[int], dict[int, list[int]], dict[int, int]] | None:
    final_colors = _assign_final_colors(n_regions=n_regions, neighbors=neighbors, palette=palette)
    if final_colors is None:
        return None

    prefs: dict[int, list[int]] = {}
    totals: dict[int, int] = {}
    for rid in range(n_regions):
        pref, total = _default_votes(final_colors[rid], palette)
        prefs[rid] = pref
        totals[rid] = total
    return final_colors, prefs, totals


def _branch_payload(
    mode: str,
    neighbors: list[list[int]],
    palette: list[int],
    nr: int,
    nc: int,
) -> tuple[list[int], dict[int, list[int]], dict[int, int]] | None:
    if mode == "single":
        return _single_conflict_case(neighbors, palette, nr, nc)
    if mode == "cascade":
        return _cascade_conflict_case(neighbors, palette, nr, nc)
    return _no_conflict_case(neighbors, palette, nr * nc)


def _gen_attempt() -> np.ndarray | None:
    row_lines, col_lines, regions, neighbors, nr, nc = _build_layout()
    h, w = row_lines[-1] + 1, col_lines[-1] + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[row_lines, :] = WALL
    grid[:, col_lines] = WALL

    palette = random.sample(list(COLOR_POOL), random.randint(4, 5))
    mode = random.choice(("none", "single", "cascade"))
    payload = _branch_payload(mode, neighbors, palette, nr, nc)
    if payload is None:
        return None
    final_colors, prefs, totals = payload

    for region in regions:
        ranked_colors = prefs[region["id"]]
        total_votes = totals[region["id"]]
        _place_votes(grid, region, ranked_colors, total_votes)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    if mode == "none":
        expected = final_colors
    else:
        wall, parsed_rows, parsed_cols = _detect_walls(grid)
        parsed_regions, parsed_neighbors, _r, _c = _region_defs(parsed_rows, parsed_cols)
        rankings = []
        totals_seen = []
        for region in parsed_regions:
            patch = grid[region["top"] + 1 : region["bottom"], region["left"] + 1 : region["right"]]
            ranked, total_votes = _ranked_votes(patch, wall)
            rankings.append(ranked)
            totals_seen.append(total_votes)
        expected = _solve_region_colors(rankings, totals_seen, parsed_neighbors)
    if expected != final_colors:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid region-voting puzzle")
