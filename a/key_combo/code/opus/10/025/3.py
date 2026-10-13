from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
SEP = 5
OBSTACLE = 7
SEED_PALETTE = (1, 2, 3, 4, 6, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _solid_rows(grid: np.ndarray) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]):
            rows.append(r)
    return rows


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [r for r in _solid_rows(grid) if int(grid[r, 0]) == SEP and int(grid[r, 0]) != bg]
    assert len(rows) == 1, "expected exactly one gray separator row"
    return rows[0]


def _find_code_row(grid: np.ndarray, sep_r: int, bg: int) -> int:
    candidates = [r for r in range(sep_r + 1, grid.shape[0]) if np.all(grid[r] != bg)]
    assert len(candidates) == 1, "expected exactly one full non-background code row"
    return candidates[0]


def _find_seeds(grid: np.ndarray, sep_r: int, bg: int) -> list[tuple[int, int]]:
    seeds = []
    for r in range(sep_r):
        color = int(grid[r, 0])
        if color != bg:
            seeds.append((r, color))
    assert 2 <= len(seeds) <= 3, "expected 2-3 left-edge seeds"
    assert len({color for _, color in seeds}) == len(seeds), "seed colors must be distinct"
    return seeds


def _top_obstacles(grid: np.ndarray, sep_r: int) -> set[tuple[int, int]]:
    coords = np.argwhere(grid[:sep_r] == OBSTACLE)
    return {tuple(map(int, rc)) for rc in coords}


def _simulate_seed(
    base: np.ndarray,
    out: np.ndarray,
    seed_row: int,
    seed_color: int,
    obstacle_cells: set[tuple[int, int]],
    *,
    write_output: bool,
) -> dict:
    h, w = base.shape
    activated: set[tuple[int, int]] = set()
    local_hits: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    full_hits: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    full_beams: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    branch_count = 0
    local_hit_count = 0
    painted: set[tuple[int, int]] = set()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_obstacle(r: int, c: int) -> bool:
        return (r, c) in obstacle_cells

    def paint(r: int, c: int) -> None:
        if not in_bounds(r, c) or int(base[r, c]) != BG:
            return
        painted.add((r, c))
        if write_output and int(out[r, c]) == BG:
            out[r, c] = seed_color

    def hit_obstacle(
        r: int,
        c: int,
        *,
        full: bool,
        blocked_entries: frozenset[tuple[int, int]] = frozenset(),
    ) -> list[tuple[int, int, frozenset[tuple[int, int]]]]:
        nonlocal branch_count, local_hit_count

        key = (r, c, tuple(sorted(blocked_entries)))
        if full:
            if key in full_hits:
                return []
            full_hits.add(key)
        else:
            if key in local_hits:
                return []
            local_hits.add(key)
            local_hit_count += 1

        activated.add((r, c))
        side_obstacles: list[tuple[int, int]] = []
        side_open = False

        for dr in (-1, 1):
            rr = r + dr
            if not (0 <= rr < h):
                continue

            entry = (rr, c - 1)
            if entry in blocked_entries:
                continue
            if c - 1 >= 0 and is_obstacle(*entry) and entry not in activated:
                continue

            side_open = True
            for cc in range(max(0, c - 1), min(w, c + 2)):
                paint(rr, cc)
                if is_obstacle(rr, cc) and (rr, cc) != (r, c):
                    side_obstacles.append((rr, cc))

        if not full and side_open:
            paint(r, c + 1)

        if full:
            up = is_obstacle(r - 1, c + 1)
            down = is_obstacle(r + 1, c + 1)
            if up and down:
                branch_count += 1
                parent = frozenset({(r, c)})
                return [(r - 1, c, parent), (r + 1, c, parent)]

        for rr, cc in side_obstacles:
            hit_obstacle(rr, cc, full=False, blocked_entries=frozenset({(r, c)}))
        return []

    def run_full_beam(
        r: int,
        start_c: int,
        blocked_entries: frozenset[tuple[int, int]] = frozenset(),
    ) -> None:
        key = (r, start_c, tuple(sorted(blocked_entries)))
        if key in full_beams or not (0 <= r < h):
            return
        full_beams.add(key)

        c = start_c + 1
        while c < w:
            if is_obstacle(r, c):
                branches = hit_obstacle(r, c, full=True, blocked_entries=blocked_entries)
                if branches:
                    for rr, cc, blocked in branches:
                        run_full_beam(rr, cc, blocked)
                    return
                c += 1
                continue

            if int(base[r, c]) == BG:
                paint(r, c)
            c += 1

    run_full_beam(seed_row, 0)
    direct_row_hits = sum(1 for r, _ in activated if r == seed_row)
    return {
        "count": len(activated),
        "activated": activated,
        "branches": branch_count,
        "local_hits": local_hit_count,
        "painted": painted,
        "direct_row_hits": direct_row_hits,
        "complex": branch_count > 0 or local_hit_count > 0,
    }


def _preview_seed_infos(grid: np.ndarray, sep_r: int) -> list[dict]:
    obstacle_cells = _top_obstacles(grid, sep_r)
    seeds = _find_seeds(grid, sep_r, BG)
    dummy_out = grid.copy()
    infos = []
    for seed_row, seed_color in seeds:
        info = _simulate_seed(grid, dummy_out, seed_row, seed_color, obstacle_cells, write_output=False)
        info["row"] = seed_row
        info["color"] = seed_color
        infos.append(info)
    return infos


def _solve_details(grid: np.ndarray, *, write_output: bool) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    sep_r = _find_separator_row(g, bg)
    code_r = _find_code_row(g, sep_r, bg)
    seeds = _find_seeds(g, sep_r, bg)
    obstacle_cells = _top_obstacles(g, sep_r)
    assert obstacle_cells, "expected at least one obstacle"

    assert np.all(g[sep_r + 1 : code_r, :] == bg), "lower middle band must be background"
    out = g.copy()

    seed_infos = []
    counts = []
    for seed_row, seed_color in seeds:
        info = _simulate_seed(g, out, seed_row, seed_color, obstacle_cells, write_output=write_output)
        assert info["count"] >= 1, "each seed must collide with at least one obstacle"
        info["row"] = seed_row
        info["color"] = seed_color
        seed_infos.append(info)
        counts.append(info["count"])

    code = np.array(g[code_r], dtype=int)
    code_counts = Counter(map(int, code))
    matched_counts = set(counts)
    matched_cols = [c for c, color in enumerate(map(int, code)) if code_counts[color] in matched_counts]
    assert matched_cols, "expected at least one code color whose frequency matches a collision count"

    if write_output:
        for c in sorted(matched_cols):
            color = int(code[c])
            height = int(code_counts[color])
            for r in range(sep_r - 1, sep_r - height - 1, -1):
                if int(out[r, c]) == bg:
                    out[r, c] = color

    return {
        "input": g,
        "output": out,
        "separator_row": sep_r,
        "code_row": code_r,
        "seed_infos": seed_infos,
        "counts": counts,
        "code_counts": code_counts,
        "matched_cols": matched_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate seed rays above the separator, count each seed's obstacle collisions, and turn matching code-row frequencies into vertical bars.

    Input structure:
    - A black background contains one full-width gray separator row, 2-3 colored seed pixels in the leftmost column above it, and orange single-pixel obstacles in the upper zone.
    - The bottom row is a fully non-background code row with repeated colors; the rows between separator and code row are empty.
    - Obstacles may lie on a seed row, on wake rows, or on both forward diagonals of a hit obstacle so that beams can split.

    Rule:
    - Each seed emits a horizontal ray to the right in its own color. When the ray hits an obstacle, the obstacle stays fixed while side wakes of length up to 3 are painted above and below on background cells.
    - If both forward diagonals of a hit obstacle are also obstacles, the parent beam stops and two child horizontal beams continue on the adjacent rows. Obstacles touched by wakes activate locally but do not become full beams unless reached by such a split.
    - Count, for each seed separately, how many distinct obstacles are activated by that seed's full beam plus all descendant wakes and branches. In the bottom code row, every color whose frequency equals any seed collision count is selected, and each of its columns is filled upward from the separator by that many cells.

    Key insight:
    - The histogram height and color selection threshold are both determined by a dynamic ray-collision count, so the solver must simulate the beam branching process before it can interpret the code row.
    """
    return _solve_details(grid, write_output=True)["output"]


TEMPLATES = {
    "simple1": {(0, 0)},
    "wake2u": {(0, 0), (-1, 1)},
    "wake2d": {(0, 0), (1, 1)},
    "split3": {(0, 0), (-1, 1), (1, 1)},
    "split4u": {(0, 0), (-1, 1), (1, 1), (-2, 2)},
    "split4d": {(0, 0), (-1, 1), (1, 1), (2, 2)},
    "blocked2": {(0, 0), (-1, -1), (1, 1)},
}


def _choose_seed_rows(sep_r: int, seed_count: int) -> list[int]:
    top_margin = 2
    bottom_limit = sep_r - 7
    candidates = list(range(top_margin, bottom_limit + 1))
    rows = [
        combo
        for combo in itertools.combinations(candidates, seed_count)
        if all(b - a >= 5 for a, b in zip(combo, combo[1:]))
    ]
    if not rows:
        raise ValueError("no feasible seed rows")
    return list(random.choice(rows))


def _place_template(
    grid: np.ndarray,
    seed_row: int,
    base_col: int,
    offsets: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for dr, dc in offsets:
        r = seed_row + dr
        c = base_col + dc
        if not (0 <= r < h and 1 <= c < w):
            return False
        if grid[r, c] != BG:
            return False
    for dr, dc in offsets:
        grid[seed_row + dr, base_col + dc] = OBSTACLE
    return True


def _sample_code_frequencies(
    width: int,
    matched_freqs: list[int],
    forbidden: set[int],
    max_parts: int,
) -> list[int]:
    remaining = width - sum(matched_freqs)
    assert remaining >= 0
    if remaining == 0:
        return []

    choices = [parts for parts in (2, 3, 4) if parts <= max_parts]
    random.shuffle(choices)
    for extra_colors in choices:
        if extra_colors > remaining:
            continue
        for _ in range(200):
            cuts = sorted(random.sample(range(1, remaining), extra_colors - 1)) if extra_colors > 1 else []
            parts = []
            prev = 0
            for cut in cuts + [remaining]:
                parts.append(cut - prev)
                prev = cut
            if all(part not in forbidden for part in parts):
                return parts
    raise RuntimeError("failed to sample non-matching code frequencies")


def _build_code_row(width: int, seed_counts: list[int], available_colors: list[int], complex_counts: set[int]) -> np.ndarray:
    unique_counts = sorted(set(seed_counts))
    matchable_counts = [c for c in unique_counts if c in complex_counts]
    if not matchable_counts:
        matchable_counts = unique_counts

    match_count = random.randint(1, min(3, len(unique_counts)))
    chosen = set(random.sample(unique_counts, match_count))
    chosen.add(random.choice(matchable_counts))
    chosen_counts = sorted(chosen)

    max_extra = len(available_colors) - len(chosen_counts)
    if max_extra < 0:
        raise RuntimeError("not enough colors for matched frequencies")
    nonmatch_parts = _sample_code_frequencies(width, chosen_counts, set(unique_counts), max_extra)
    total_colors = len(chosen_counts) + len(nonmatch_parts)
    assert total_colors <= len(available_colors)
    colors = random.sample(available_colors, total_colors)

    row_values = []
    color_cursor = 0
    for freq in chosen_counts:
        color = colors[color_cursor]
        color_cursor += 1
        row_values.extend([color] * freq)
    for freq in nonmatch_parts:
        color = colors[color_cursor]
        color_cursor += 1
        row_values.extend([color] * freq)

    assert len(row_values) == width
    random.shuffle(row_values)
    return np.array(row_values, dtype=int)


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        details = _solve_details(grid, write_output=False)
    except Exception:
        return False

    sep_r = details["separator_row"]
    code = details["input"][details["code_row"]]
    code_counts = details["code_counts"]
    seed_infos = details["seed_infos"]

    if not any(info["complex"] for info in seed_infos):
        return False
    if not any(info["complex"] and info["count"] != info["direct_row_hits"] for info in seed_infos):
        return False
    if len(set(details["counts"])) < 2:
        return False

    matched_complex = {
        info["count"]
        for info in seed_infos
        if info["complex"] and info["count"] in code_counts.values()
    }
    if not matched_complex:
        return False

    for c in details["matched_cols"]:
        color = int(code[c])
        height = int(code_counts[color])
        if sep_r - height < 0:
            return False
        if np.any(grid[sep_r - height : sep_r, c] != BG):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(18, 24)
    lower_gap = random.randint(3, 5)
    sep_r = h - lower_gap - 2
    if sep_r < 9:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[sep_r, :] = SEP

    seed_count = random.randint(2, 3)
    try:
        seed_rows = _choose_seed_rows(sep_r, seed_count)
    except ValueError:
        return None

    seed_colors = random.sample(SEED_PALETTE, seed_count)
    for row, color in zip(seed_rows, seed_colors):
        grid[row, 0] = color

    template_pool = ["simple1", "wake2u", "wake2d", "split3", "split4u", "split4d", "blocked2"]
    complex_used = False
    base_cols = random.sample(range(3, min(w - 3, 10)), seed_count)

    for i, (seed_row, base_col) in enumerate(zip(seed_rows, base_cols)):
        if i == seed_count - 1 and not complex_used:
            name = random.choice(["split3", "split4u", "split4d", "blocked2", "wake2u", "wake2d"])
        else:
            name = random.choice(template_pool)
        if name != "simple1":
            complex_used = True
        if not _place_template(grid, seed_row, base_col, TEMPLATES[name]):
            return None

    seed_infos = _preview_seed_infos(grid, sep_r)
    seed_counts = [info["count"] for info in seed_infos]
    complex_counts = {info["count"] for info in seed_infos if info["complex"]}

    code_palette = [c for c in SEED_PALETTE if c not in seed_colors]
    if len(code_palette) < 4:
        return None
    code_row = _build_code_row(w, seed_counts, code_palette, complex_counts)
    grid[h - 1, :] = code_row

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
