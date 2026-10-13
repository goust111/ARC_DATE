from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 12
GRID_MAX = 24
TRACK_CONFIGS = [(2, 2), (2, 3), (3, 2), (3, 3)]
MIN_TRACK_GAP = 3
GEN_RETRIES = 300
SEED_ASSIGN_RETRIES = 800


def _get_background(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return int(Counter(corners).most_common(1)[0][0])


def _get_track_color(grid: np.ndarray, background: int) -> int:
    counts = Counter(map(int, grid.flatten()))
    candidates = [(count, color) for color, count in counts.items() if int(color) != int(background)]
    assert candidates, "expected at least one non-background color"
    return int(max(candidates)[1])


def _detect_tracks(
    grid: np.ndarray,
    background: int,
    track_color: int,
) -> tuple[list[dict], list[dict], list[list[dict]], list[list[dict]]]:
    h, w = grid.shape
    occupied = grid != int(background)

    horizontal: list[dict] = []
    horizontal_by_row: list[list[dict]] = [[] for _ in range(h)]
    for r in range(h):
        c = 0
        while c < w:
            if not occupied[r, c]:
                c += 1
                continue
            c0 = c
            while c < w and occupied[r, c]:
                c += 1
            c1 = c - 1
            if c1 - c0 + 1 < 2:
                continue
            seed_cols = [
                cc
                for cc in range(c0, c1 + 1)
                if int(grid[r, cc]) not in (int(background), int(track_color))
            ]
            assert len(seed_cols) <= 1, f"ambiguous horizontal track on row {r}"
            if len(seed_cols) != 1:
                continue
            seed_c = int(seed_cols[0])
            line = {
                "axis": "h",
                "r": int(r),
                "c0": int(c0),
                "c1": int(c1),
                "seed": (int(r), int(seed_c)),
                "color": int(grid[r, seed_c]),
            }
            horizontal.append(line)
            horizontal_by_row[r].append(line)

    vertical: list[dict] = []
    vertical_by_col: list[list[dict]] = [[] for _ in range(w)]
    for c in range(w):
        r = 0
        while r < h:
            if not occupied[r, c]:
                r += 1
                continue
            r0 = r
            while r < h and occupied[r, c]:
                r += 1
            r1 = r - 1
            if r1 - r0 + 1 < 2:
                continue
            seed_rows = [
                rr
                for rr in range(r0, r1 + 1)
                if int(grid[rr, c]) not in (int(background), int(track_color))
            ]
            assert len(seed_rows) <= 1, f"ambiguous vertical track on col {c}"
            if len(seed_rows) != 1:
                continue
            seed_r = int(seed_rows[0])
            line = {
                "axis": "v",
                "c": int(c),
                "r0": int(r0),
                "r1": int(r1),
                "seed": (int(seed_r), int(c)),
                "color": int(grid[seed_r, c]),
            }
            vertical.append(line)
            vertical_by_col[c].append(line)

    coverage = np.zeros_like(grid, dtype=int)
    for line in horizontal:
        coverage[int(line["r"]), int(line["c0"]) : int(line["c1"]) + 1] += 1
    for line in vertical:
        coverage[int(line["r0"]) : int(line["r1"]) + 1, int(line["c"])] += 1

    assert np.all(coverage[occupied] >= 1), "every non-background track cell must belong to a line"
    assert int(coverage.max()) <= 2, "a cell can belong to at most one horizontal and one vertical track"

    seen_colors = set()
    seen_seeds = set()
    for line in horizontal + vertical:
        assert int(line["color"]) not in seen_colors, "seed colors must be unique"
        assert tuple(line["seed"]) not in seen_seeds, "seed positions must be unique"
        seen_colors.add(int(line["color"]))
        seen_seeds.add(tuple(line["seed"]))

    return horizontal, vertical, horizontal_by_row, vertical_by_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each straight grey track from its seed, resolving shared crossings by along-track distance.

    Input structure:
    - Background is black. Grey pixels form one-cell-wide straight horizontal and vertical tracks, and tracks may cross.
    - Each track contains exactly one non-black, non-grey seed pixel, and every seed color is different.
    - A crossing is a shared cell where one horizontal track and one vertical track overlap.

    Rule:
    - Treat every maximal horizontal or vertical non-black segment as a track and read its unique seed position.
    - Recolor each track cell by the seed of the closest track passing through that cell, where distance is counted in steps along that seed's own straight track.
    - At a crossing with equal distances, keep the color of the seed whose coordinates come first in row-major order.
    - Leave black background unchanged; every grey track cell becomes one of the seed colors.

    Key insight:
    - Seeds spread only along their own straight line, so the only competition happens on shared crossing cells, where the winner is chosen by along-track distance instead of paint order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    background = _get_background(g)
    track_color = _get_track_color(g, background)
    occupied = g != int(background)

    horizontal, vertical, horizontal_by_row, vertical_by_col = _detect_tracks(g, background, track_color)
    assert horizontal or vertical, "expected at least one track"

    out = np.array(g, copy=True)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if not occupied[r, c]:
                out[r, c] = int(background)
                continue

            best_key = None
            best_color = None

            for line in horizontal_by_row[r]:
                if int(line["c0"]) <= c <= int(line["c1"]):
                    seed_r, seed_c = line["seed"]
                    key = (abs(c - int(seed_c)), int(seed_r), int(seed_c))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_color = int(line["color"])

            for line in vertical_by_col[c]:
                if int(line["r0"]) <= r <= int(line["r1"]):
                    seed_r, seed_c = line["seed"]
                    key = (abs(r - int(seed_r)), int(seed_r), int(seed_c))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_color = int(line["color"])

            assert best_color is not None, f"unassigned track cell at {(r, c)}"
            out[r, c] = int(best_color)

    assert not np.any(out == int(track_color)), "all grey track cells should be recolored"
    return out


def _pick_spaced_values(
    rng: random.Random,
    low: int,
    high: int,
    count: int,
    min_gap: int,
) -> list[int] | None:
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(300):
        picks = sorted(rng.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return [int(v) for v in picks]
    return None


def _build_layout(rng: random.Random) -> dict | None:
    h_count, v_count = rng.choice(TRACK_CONFIGS)

    min_h = max(GRID_MIN, 6 + (h_count - 1) * MIN_TRACK_GAP)
    min_w = max(GRID_MIN, 6 + (v_count - 1) * MIN_TRACK_GAP)
    height = rng.randint(min_h, GRID_MAX)
    width = rng.randint(min_w, GRID_MAX)

    rows = _pick_spaced_values(rng, 2, height - 3, h_count, MIN_TRACK_GAP)
    cols = _pick_spaced_values(rng, 2, width - 3, v_count, MIN_TRACK_GAP)
    if rows is None or cols is None:
        return None

    top_limit = min(rows) - 1
    bottom_limit = max(rows) + 1
    left_limit = min(cols) - 1
    right_limit = max(cols) + 1

    horizontal = []
    for r in rows:
        c0 = rng.randint(1, left_limit)
        c1 = rng.randint(right_limit, width - 2)
        horizontal.append({"r": int(r), "c0": int(c0), "c1": int(c1)})

    vertical = []
    for c in cols:
        r0 = rng.randint(1, top_limit)
        r1 = rng.randint(bottom_limit, height - 2)
        vertical.append({"c": int(c), "r0": int(r0), "r1": int(r1)})

    return {
        "shape": (int(height), int(width)),
        "rows": [int(r) for r in rows],
        "cols": [int(c) for c in cols],
        "horizontal": horizontal,
        "vertical": vertical,
    }


def _available_seed_sites(layout: dict) -> tuple[list[list[int]], list[list[int]]]:
    cols = set(layout["cols"])
    rows = set(layout["rows"])

    horizontal_sites = []
    for line in layout["horizontal"]:
        sites = [c for c in range(int(line["c0"]), int(line["c1"]) + 1) if c not in cols]
        if not sites:
            return [], []
        horizontal_sites.append(sites)

    vertical_sites = []
    for line in layout["vertical"]:
        sites = [r for r in range(int(line["r0"]), int(line["r1"]) + 1) if r not in rows]
        if not sites:
            return [], []
        vertical_sites.append(sites)

    return horizontal_sites, vertical_sites


def _intersection_categories(
    rows: list[int],
    cols: list[int],
    h_seed_cols: list[int],
    v_seed_rows: list[int],
) -> tuple[set[str], set[str]]:
    categories: set[str] = set()
    tie_winners: set[str] = set()
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            dh = abs(int(h_seed_cols[i]) - int(c))
            dv = abs(int(v_seed_rows[j]) - int(r))
            if dh < dv:
                categories.add("horizontal")
            elif dv < dh:
                categories.add("vertical")
            else:
                categories.add("tie")
                horizontal_seed = (int(r), int(h_seed_cols[i]))
                vertical_seed = (int(v_seed_rows[j]), int(c))
                if horizontal_seed < vertical_seed:
                    tie_winners.add("horizontal")
                else:
                    tie_winners.add("vertical")
    return categories, tie_winners


def _assign_seed_sites(rng: random.Random, layout: dict) -> tuple[list[int], list[int]] | None:
    horizontal_sites, vertical_sites = _available_seed_sites(layout)
    if not horizontal_sites or not vertical_sites:
        return None

    best: tuple[list[int], list[int]] | None = None
    best_score = -1
    for _ in range(SEED_ASSIGN_RETRIES):
        h_seed_cols = [int(rng.choice(sites)) for sites in horizontal_sites]
        v_seed_rows = [int(rng.choice(sites)) for sites in vertical_sites]
        categories, tie_winners = _intersection_categories(
            layout["rows"],
            layout["cols"],
            h_seed_cols,
            v_seed_rows,
        )
        score = 10 * len(categories) + len(tie_winners)
        if score > best_score:
            best = (h_seed_cols, v_seed_rows)
            best_score = score
        if categories == {"horizontal", "vertical", "tie"}:
            return h_seed_cols, v_seed_rows

    return best


def _paint_input(layout: dict, h_seed_cols: list[int], v_seed_rows: list[int], colors: list[int]) -> np.ndarray:
    height, width = layout["shape"]
    grid = np.full((height, width), BLACK, dtype=int)

    for line in layout["horizontal"]:
        grid[int(line["r"]), int(line["c0"]) : int(line["c1"]) + 1] = GREY
    for line in layout["vertical"]:
        grid[int(line["r0"]) : int(line["r1"]) + 1, int(line["c"])] = GREY

    color_iter = iter(colors)
    for line, seed_c in zip(layout["horizontal"], h_seed_cols):
        grid[int(line["r"]), int(seed_c)] = int(next(color_iter))
    for line, seed_r in zip(layout["vertical"], v_seed_rows):
        grid[int(seed_r), int(line["c"])] = int(next(color_iter))

    return grid


def _scene_is_valid(grid: np.ndarray) -> bool:
    background = _get_background(grid)
    track_color = _get_track_color(grid, background)
    horizontal, vertical, _by_row, _by_col = _detect_tracks(grid, background, track_color)
    if not horizontal or not vertical:
        return False

    rows = [int(line["r"]) for line in horizontal]
    cols = [int(line["c"]) for line in vertical]
    h_seed_cols = [int(line["seed"][1]) for line in horizontal]
    v_seed_rows = [int(line["seed"][0]) for line in vertical]
    categories, _tie_winners = _intersection_categories(rows, cols, h_seed_cols, v_seed_rows)
    if categories != {"horizontal", "vertical", "tie"}:
        return False

    out = main(grid)
    return not np.any(out == GREY)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _build_layout(rng)
    if layout is None:
        return None

    assigned = _assign_seed_sites(rng, layout)
    if assigned is None:
        return None
    h_seed_cols, v_seed_rows = assigned

    categories, _tie_winners = _intersection_categories(layout["rows"], layout["cols"], h_seed_cols, v_seed_rows)
    if categories != {"horizontal", "vertical", "tie"}:
        return None

    total_tracks = len(layout["horizontal"]) + len(layout["vertical"])
    colors = rng.sample(SEED_COLORS, total_tracks)
    grid = _paint_input(layout, h_seed_cols, v_seed_rows, colors)
    if not _scene_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
