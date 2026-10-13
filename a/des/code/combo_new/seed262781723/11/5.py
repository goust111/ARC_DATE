from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator palette: black background, grey channels, distinct seed colors.
GEN_BG = 0
GEN_CHANNEL = 5
GEN_SEEDS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values), return_counts=True)
    return int(vals[counts.argmax()])


def _channel_color(grid: np.ndarray, bg: int) -> int | None:
    non_bg = grid[grid != bg]
    if non_bg.size == 0:
        return None
    return _mode(non_bg)


def _trace_seed_paths(
    grid: np.ndarray, bg: int, channel: int
) -> tuple[dict[tuple[int, int], list[tuple[int, int, str]]], list[tuple[int, int, int, str]]]:
    h, w = grid.shape
    claims: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
    seeds: list[tuple[int, int, int, str]] = []

    for r, c in np.argwhere((grid != bg) & (grid != channel)):
        r = int(r)
        c = int(c)
        color = int(grid[r, c])
        neighbors: list[tuple[int, int, int, int]] = []
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == channel:
                neighbors.append((dr, dc, nr, nc))

        assert len(neighbors) == 1, "each seed must touch exactly one channel endpoint"
        dr, dc, nr, nc = neighbors[0]
        orientation = "h" if dc != 0 else "v"
        seeds.append((r, c, color, orientation))

        step = 1
        rr, cc = nr, nc
        while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == channel:
            claims.setdefault((rr, cc), []).append((step, color, orientation))
            rr += dr
            cc += dc
            step += 1

    return claims, seeds


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    channel = _channel_color(g, bg)
    if channel is None:
        return g.copy()

    claims, _ = _trace_seed_paths(g, bg, channel)
    out = g.copy()
    for r, c in np.argwhere(g == channel):
        key = (int(r), int(c))
        if key not in claims:
            continue
        best_dist = min(dist for dist, _color, _ori in claims[key])
        winners = [color for dist, color, _ori in claims[key] if dist == best_dist]
        if len(winners) == 1:
            out[key] = winners[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood each seeded grey channel while awarding shared crossroads to the nearer seed.

    Input structure:
    - The background is a dominant blank color.
    - One structural color forms straight one-pixel horizontal and vertical channels that may cross.
    - Distinct non-structural single pixels sit just outside some channel endpoints as seeds.

    Rule:
    - For each seed, step into its adjacent channel cell and continue straight through that whole row or column segment until the channel ends.
    - Every channel pixel claimed by exactly one seed takes that seed's color.
    - If a crossroad pixel is claimed by both a horizontal seed and a vertical seed, compare their step counts to that pixel and keep the closer seed's color.
    - Channel pixels never reached by any seed stay in the channel color, and the seed pixels themselves remain in place.

    Key insight:
    The spread itself is just straight-line filling, but intersections are special: only those shared pixels need conflict resolution, and the winner is determined by visible path distance rather than by color or update order.
    """
    return _solve(grid)


def _distance_to_cell(coord: int, start: int, end: int, side: str) -> int:
    if side in {"left", "top"}:
        return coord - start + 1
    return end - coord + 1


def _choose_network() -> tuple[int, int, int, int, int, int, list[int], list[int]] | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)

    top_choices = [r for r in range(2, h - 7) if r % 2 == 0]
    left_choices = [c for c in range(2, w - 7) if c % 2 == 0]
    if not top_choices or not left_choices:
        return None

    top = random.choice(top_choices)
    left = random.choice(left_choices)

    bottom_choices = [r for r in range(top + 8, h - 1) if r % 2 == 0 and r <= h - 2]
    right_choices = [c for c in range(left + 6, w - 1) if c % 2 == 0 and c <= w - 2]
    if not bottom_choices or not right_choices:
        return None

    bottom = random.choice(bottom_choices)
    right = random.choice(right_choices)

    row_parity = random.choice([0, 1])
    col_parity = 1 - row_parity

    rows = [r for r in range(top + 1, bottom) if r % 2 == row_parity]
    cols = [c for c in range(left + 1, right) if c % 2 == col_parity]
    if len(rows) < 2 or len(cols) < 2:
        return None

    nh = random.randint(2, min(4, len(rows)))
    nv = random.randint(2, min(4, len(cols)))
    chosen_rows = sorted(random.sample(rows, nh))
    chosen_cols = sorted(random.sample(cols, nv))
    return h, w, top, bottom, left, right, chosen_rows, chosen_cols


def _paint_network(
    h: int,
    w: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    rows: list[int],
    cols: list[int],
    h_seeds: dict[int, tuple[str, int]],
    v_seeds: dict[int, tuple[str, int]],
) -> np.ndarray:
    grid = np.full((h, w), GEN_BG, dtype=int)
    for row in rows:
        grid[row, left : right + 1] = GEN_CHANNEL
    for col in cols:
        grid[top : bottom + 1, col] = GEN_CHANNEL

    for row, (side, color) in h_seeds.items():
        seed_col = left - 1 if side == "left" else right + 1
        grid[row, seed_col] = color
    for col, (side, color) in v_seeds.items():
        seed_row = top - 1 if side == "top" else bottom + 1
        grid[seed_row, col] = color
    return grid


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        claims, seeds = _trace_seed_paths(grid, GEN_BG, GEN_CHANNEL)
        out = _solve(grid)
    except Exception:
        return False

    if len(seeds) < 2:
        return False
    if np.array_equal(out, grid):
        return False
    if not np.any(out == GEN_CHANNEL):
        return False

    contested: list[tuple[int, int]] = []
    winners: set[str] = set()
    for cell, options in claims.items():
        if len(options) < 2:
            continue
        contested.append(cell)
        best = min(dist for dist, _color, _ori in options)
        best_orientations = {ori for dist, _color, ori in options if dist == best}
        if len(best_orientations) != 1:
            return False
        winners.add(next(iter(best_orientations)))

    if not contested:
        return False
    return bool(winners)


def _duel_scene(horizontal_wins: bool) -> np.ndarray | None:
    network = _choose_network()
    if network is None:
        return None
    h, w, top, bottom, left, right, rows, cols = network

    candidates: list[tuple[int, int, str, str]] = []
    for row in rows:
        for col in cols:
            for h_side in ("left", "right"):
                for v_side in ("top", "bottom"):
                    h_dist = _distance_to_cell(col, left, right, h_side)
                    v_dist = _distance_to_cell(row, top, bottom, v_side)
                    if horizontal_wins and h_dist < v_dist:
                        candidates.append((row, col, h_side, v_side))
                    if (not horizontal_wins) and v_dist < h_dist:
                        candidates.append((row, col, h_side, v_side))
    if not candidates:
        return None

    row, col, h_side, v_side = random.choice(candidates)
    colors = random.sample(GEN_SEEDS, 2)
    h_seeds = {row: (h_side, colors[0])}
    v_seeds = {col: (v_side, colors[1])}
    grid = _paint_network(h, w, top, bottom, left, right, rows, cols, h_seeds, v_seeds)
    return grid if _validate_scene(grid) else None


def _multi_scene() -> np.ndarray | None:
    network = _choose_network()
    if network is None:
        return None
    h, w, top, bottom, left, right, rows, cols = network

    for _ in range(60):
        seed_rows_count = random.randint(1, len(rows))
        seed_cols_count = random.randint(1, len(cols))
        if seed_rows_count + seed_cols_count < 3:
            continue
        if seed_rows_count == len(rows) and seed_cols_count == len(cols):
            if seed_rows_count > 1 and random.random() < 0.5:
                seed_rows_count -= 1
            elif seed_cols_count > 1:
                seed_cols_count -= 1
            else:
                continue

        seed_rows = sorted(random.sample(rows, seed_rows_count))
        seed_cols = sorted(random.sample(cols, seed_cols_count))
        total_seeds = len(seed_rows) + len(seed_cols)
        if total_seeds > len(GEN_SEEDS):
            continue

        colors = random.sample(GEN_SEEDS, total_seeds)
        h_seeds = {
            row: (random.choice(["left", "right"]), color)
            for row, color in zip(seed_rows, colors[: len(seed_rows)], strict=False)
        }
        v_seeds = {
            col: (random.choice(["top", "bottom"]), color)
            for col, color in zip(seed_cols, colors[len(seed_rows) :], strict=False)
        }
        grid = _paint_network(h, w, top, bottom, left, right, rows, cols, h_seeds, v_seeds)

        if not _validate_scene(grid):
            continue

        claims, _ = _trace_seed_paths(grid, GEN_BG, GEN_CHANNEL)
        winner_orientations: set[str] = set()
        for options in claims.values():
            if len(options) < 2:
                continue
            best = min(dist for dist, _color, _ori in options)
            winner_orientations.update(
                ori for dist, _color, ori in options if dist == best
            )
        if winner_orientations == {"h", "v"}:
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    scene_type = random.choice(["duel_h", "duel_v", "multi"])
    if scene_type == "duel_h":
        return _duel_scene(horizontal_wins=True)
    if scene_type == "duel_v":
        return _duel_scene(horizontal_wins=False)
    return _multi_scene()


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid channel-distance puzzle")
