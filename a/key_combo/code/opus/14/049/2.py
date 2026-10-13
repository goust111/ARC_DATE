from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
SEED = 5
LINE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(int(x) for x in np.unique(grid))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_segments(grid: np.ndarray) -> tuple[list[dict[str, int | str]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    horizontal_owner = np.full(g.shape, -1, dtype=int)
    vertical_owner = np.full(g.shape, -1, dtype=int)
    segments: list[dict[str, int | str]] = []

    for color in _colors_present(g, exclude={bg, SEED}):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each line color must appear exactly twice as endpoints"
        (r0, c0), (r1, c1) = coords.tolist()
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        assert r0 == r1 or c0 == c1, "same-color endpoints must share a row or a column"

        if r0 == r1:
            c_start, c_end = sorted((c0, c1))
            for c in range(c_start, c_end + 1):
                assert horizontal_owner[r0, c] == -1, "horizontal lines must not overlap each other"
                horizontal_owner[r0, c] = color
            segments.append(
                {"orient": "horizontal", "color": color, "r0": r0, "c0": c_start, "r1": r1, "c1": c_end}
            )
        else:
            r_start, r_end = sorted((r0, r1))
            for r in range(r_start, r_end + 1):
                assert vertical_owner[r, c0] == -1, "vertical lines must not overlap each other"
                vertical_owner[r, c0] = color
            segments.append(
                {"orient": "vertical", "color": color, "r0": r_start, "c0": c0, "r1": r_end, "c1": c1}
            )

    assert segments, "expected at least one colored endpoint pair"
    return segments, bg


def _segment_cells(segment: dict[str, int | str]) -> list[tuple[int, int]]:
    orient = str(segment["orient"])
    color = int(segment["color"])
    _ = color
    r0 = int(segment["r0"])
    c0 = int(segment["c0"])
    r1 = int(segment["r1"])
    c1 = int(segment["c1"])
    if orient == "horizontal":
        return [(r0, c) for c in range(c0, c1 + 1)]
    return [(r, c0) for r in range(r0, r1 + 1)]


def _line_geometry(
    shape: tuple[int, int], segments: list[dict[str, int | str]]
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], list[tuple[str, int]]], int]:
    line_mask = np.zeros(shape, dtype=bool)
    visible = np.zeros(shape, dtype=int)
    pos_to_incidence: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)

    for segment in segments:
        if segment["orient"] != "horizontal":
            continue
        color = int(segment["color"])
        for pos in _segment_cells(segment):
            line_mask[pos] = True
            visible[pos] = color
            pos_to_incidence[pos].append(("horizontal", color))

    for segment in segments:
        if segment["orient"] != "vertical":
            continue
        color = int(segment["color"])
        for pos in _segment_cells(segment):
            line_mask[pos] = True
            visible[pos] = color
            pos_to_incidence[pos].append(("vertical", color))

    crossing_count = sum(1 for incidences in pos_to_incidence.values() if len(incidences) > 1)
    return line_mask, visible, pos_to_incidence, crossing_count


def _region_components(line_mask: np.ndarray) -> list[dict[str, object]]:
    h, w = line_mask.shape
    seen = np.zeros_like(line_mask, dtype=bool)
    regions: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if line_mask[r0, c0] or seen[r0, c0]:
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            touches_border = False

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not line_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            boundary_positions: set[tuple[int, int]] = set()
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and line_mask[nr, nc]:
                        boundary_positions.add((nr, nc))

            regions.append(
                {
                    "coords": coords,
                    "closed": not touches_border,
                    "boundary_positions": boundary_positions,
                }
            )

    return regions


def _nearest_line_choice(
    seed: tuple[int, int],
    boundary_positions: set[tuple[int, int]],
    pos_to_incidence: dict[tuple[int, int], list[tuple[str, int]]],
) -> tuple[int, str]:
    best_distance: int | None = None
    best_hits: list[tuple[str, int]] = []

    for r, c in boundary_positions:
        dist = abs(seed[0] - r) + abs(seed[1] - c)
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_hits = list(pos_to_incidence[(r, c)])
        elif dist == best_distance:
            best_hits.extend(pos_to_incidence[(r, c)])

    assert best_distance is not None and best_hits, "closed seeded region must have boundary line pixels"

    vertical_colors = {color for orient, color in best_hits if orient == "vertical"}
    horizontal_colors = {color for orient, color in best_hits if orient == "horizontal"}

    if vertical_colors:
        assert len(vertical_colors) == 1, "nearest-distance tie among multiple vertical colors is ambiguous"
        if horizontal_colors:
            return next(iter(vertical_colors)), "tie_vertical"
        return next(iter(vertical_colors)), "vertical"

    assert len(horizontal_colors) == 1, "nearest-distance tie among multiple horizontal colors is ambiguous"
    return next(iter(horizontal_colors)), "horizontal"


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    segments, bg = _extract_segments(g)
    line_mask, visible_lines, pos_to_incidence, crossing_count = _line_geometry(g.shape, segments)
    regions = _region_components(line_mask)

    out = g.copy()
    out[line_mask] = visible_lines[line_mask]

    closed_seeded = 0
    for region in regions:
        coords = list(region["coords"])
        seeds = [(r, c) for r, c in coords if int(g[r, c]) == SEED]
        region["seeds"] = seeds
        if not bool(region["closed"]):
            region["fill_color"] = None
            region["fill_mode"] = None
            continue
        assert len(seeds) <= 1, "each closed region must contain at most one seed"
        if not seeds:
            region["fill_color"] = None
            region["fill_mode"] = None
            continue
        fill_color, fill_mode = _nearest_line_choice(
            seeds[0],
            set(region["boundary_positions"]),
            pos_to_incidence,
        )
        for r, c in coords:
            out[r, c] = fill_color
        region["fill_color"] = fill_color
        region["fill_mode"] = fill_mode
        closed_seeded += 1

    return {
        "output": out,
        "segments": segments,
        "bg": bg,
        "line_mask": line_mask,
        "visible_lines": visible_lines,
        "pos_to_incidence": pos_to_incidence,
        "crossing_count": crossing_count,
        "regions": regions,
        "closed_seeded": closed_seeded,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned color pairs into crossing lines, then flood-fill each enclosed seeded region using the nearest boundary segment color.

    Input structure:
    - The background is black, with several non-gray colors each appearing exactly twice as aligned endpoints.
    - Each endpoint pair defines a horizontal or vertical one-pixel-wide segment, and gray pixels are fill seeds.
    - After the segments are drawn, some non-line cells become fully enclosed regions while other cells remain connected to the grid boundary.

    Rule:
    - Draw every endpoint pair as an inclusive straight segment of its own color.
    - If a horizontal and a vertical segment cross on the same cell, the visible crossing color is the vertical one.
    - Find every region of non-line cells that is completely cut off from the grid boundary; for each such region with a gray seed, compute the seed's minimum Manhattan distance to the region's boundary line pixels and use the corresponding segment color as the fill color, breaking distance ties in favor of vertical segments.
    - Flood-fill the entire enclosed region, including the seed cell, with that chosen color; leave non-enclosed gray seeds unchanged.

    Key insight:
    - The fillable areas do not exist in the input directly; they emerge only after the aligned endpoint pairs are connected into a crossing line network, and the same vertical-priority logic governs both crossings and nearest-boundary tie resolution.
    """
    return np.asarray(_solve_details(grid)["output"], dtype=int)


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[rng.randrange(parts)] += 1
    return buckets


def _sample_axis(limit: int, count: int, rng: random.Random) -> list[int] | None:
    slack = limit - 5 - 4 * (count - 1)
    if slack < 0:
        return None
    parts = _random_partition(slack, count + 1, rng)
    positions = [2 + parts[0]]
    for gap_extra in parts[1:count]:
        positions.append(positions[-1] + 4 + gap_extra)
    if positions[-1] > limit - 3:
        return None
    return positions


def _build_framework(rng: random.Random) -> tuple[np.ndarray, list[dict[str, int | str]], list[int], list[int]] | None:
    height = rng.randint(14, 22)
    width = rng.randint(14, 22)
    possible_h = [count for count in (3, 4) if _sample_axis(height, count, rng) is not None]
    possible_v = [count for count in (3, 4) if _sample_axis(width, count, rng) is not None]
    if not possible_h or not possible_v:
        return None

    n_h = rng.choice(possible_h)
    n_v = rng.choice(possible_v)
    rows = _sample_axis(height, n_h, rng)
    cols = _sample_axis(width, n_v, rng)
    if rows is None or cols is None:
        return None

    colors = list(LINE_COLORS)
    rng.shuffle(colors)
    if n_h + n_v > len(colors):
        return None

    segments: list[dict[str, int | str]] = []
    color_index = 0
    for row in rows:
        c0 = rng.randint(1, cols[0] - 1)
        c1 = rng.randint(cols[-1] + 1, width - 2)
        segments.append({"orient": "horizontal", "color": colors[color_index], "r0": row, "c0": c0, "r1": row, "c1": c1})
        color_index += 1
    for col in cols:
        r0 = rng.randint(1, rows[0] - 1)
        r1 = rng.randint(rows[-1] + 1, height - 2)
        segments.append({"orient": "vertical", "color": colors[color_index], "r0": r0, "c0": col, "r1": r1, "c1": col})
        color_index += 1

    grid = np.zeros((height, width), dtype=int)
    for segment in segments:
        color = int(segment["color"])
        r0 = int(segment["r0"])
        c0 = int(segment["c0"])
        r1 = int(segment["r1"])
        c1 = int(segment["c1"])
        if segment["orient"] == "horizontal":
            if grid[r0, c0] != BG or grid[r1, c1] != BG:
                return None
            grid[r0, c0] = color
            grid[r1, c1] = color
        else:
            if grid[r0, c0] != BG or grid[r1, c1] != BG:
                return None
            grid[r0, c0] = color
            grid[r1, c1] = color
    return grid, segments, rows, cols


def _region_candidates(
    region: dict[str, object], pos_to_incidence: dict[tuple[int, int], list[tuple[str, int]]]
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    boundary_positions = set(region["boundary_positions"])
    for cell in region["coords"]:
        try:
            fill_color, fill_mode = _nearest_line_choice(cell, boundary_positions, pos_to_incidence)
        except AssertionError:
            continue
        candidates.append({"cell": cell, "fill_color": fill_color, "fill_mode": fill_mode})
    return candidates


def _choose_seed_layout(
    closed_regions: list[dict[str, object]],
    pos_to_incidence: dict[tuple[int, int], list[tuple[str, int]]],
    rng: random.Random,
    force_tie: bool,
) -> tuple[dict[int, tuple[int, int]], dict[int, str], dict[int, int]] | None:
    candidate_map: dict[int, list[dict[str, object]]] = {}
    mode_map: dict[int, dict[str, list[dict[str, object]]]] = {}
    for idx, region in enumerate(closed_regions):
        region_candidates = _region_candidates(region, pos_to_incidence)
        if not region_candidates:
            return None
        candidate_map[idx] = region_candidates
        modes: dict[str, list[dict[str, object]]] = defaultdict(list)
        for cand in region_candidates:
            modes[str(cand["fill_mode"])].append(cand)
        mode_map[idx] = dict(modes)

    assignments: dict[int, tuple[int, int]] = {}
    modes: dict[int, str] = {}
    colors: dict[int, int] = {}

    if force_tie:
        tie_regions = [idx for idx, grouped in mode_map.items() if "tie_vertical" in grouped]
        if not tie_regions:
            return None
        tie_idx = rng.choice(tie_regions)
        cand = rng.choice(mode_map[tie_idx]["tie_vertical"])
        assignments[tie_idx] = tuple(cand["cell"])
        modes[tie_idx] = "tie_vertical"
        colors[tie_idx] = int(cand["fill_color"])

    horizontal_regions = [idx for idx, grouped in mode_map.items() if "horizontal" in grouped and idx not in assignments]
    if not horizontal_regions:
        return None
    horizontal_idx = rng.choice(horizontal_regions)
    cand = rng.choice(mode_map[horizontal_idx]["horizontal"])
    assignments[horizontal_idx] = tuple(cand["cell"])
    modes[horizontal_idx] = "horizontal"
    colors[horizontal_idx] = int(cand["fill_color"])

    if not force_tie:
        vertical_regions = [idx for idx, grouped in mode_map.items() if "vertical" in grouped and idx not in assignments]
        if not vertical_regions:
            return None
        vertical_idx = rng.choice(vertical_regions)
        cand = rng.choice(mode_map[vertical_idx]["vertical"])
        assignments[vertical_idx] = tuple(cand["cell"])
        modes[vertical_idx] = "vertical"
        colors[vertical_idx] = int(cand["fill_color"])

    for idx in range(len(closed_regions)):
        if idx in assignments:
            continue
        options = list(candidate_map[idx])
        if force_tie and "tie_vertical" in mode_map[idx] and rng.random() < 0.35:
            options = list(mode_map[idx]["tie_vertical"])
        elif "vertical" in mode_map[idx] and "horizontal" in mode_map[idx]:
            preferred = "vertical" if rng.random() < 0.5 else "horizontal"
            options = list(mode_map[idx][preferred])
        cand = rng.choice(options)
        assignments[idx] = tuple(cand["cell"])
        modes[idx] = str(cand["fill_mode"])
        colors[idx] = int(cand["fill_color"])

    if len(set(colors.values())) < 2:
        return None
    return assignments, modes, colors


def _place_open_seeds(
    grid: np.ndarray, open_regions: list[dict[str, object]], line_mask: np.ndarray, rng: random.Random
) -> list[tuple[int, int]]:
    placed: list[tuple[int, int]] = []
    if not open_regions or rng.random() >= 0.45:
        return placed

    candidates_by_region: list[list[tuple[int, int]]] = []
    for region in open_regions:
        cells = []
        for r, c in region["coords"]:
            if grid[r, c] != BG:
                continue
            if any(0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and line_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                continue
            cells.append((r, c))
        if cells:
            candidates_by_region.append(cells)

    if not candidates_by_region:
        return placed

    seed_count = 1 if rng.random() < 0.7 else 2
    rng.shuffle(candidates_by_region)
    for cells in candidates_by_region[:seed_count]:
        pos = rng.choice(cells)
        grid[pos] = SEED
        placed.append(pos)
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    built = _build_framework(rng)
    if built is None:
        return None
    grid, _segments, _rows, _cols = built

    base_details = _solve_details(grid)
    if int(base_details["crossing_count"]) < 3:
        return None

    closed_regions = [region for region in base_details["regions"] if bool(region["closed"])]
    open_regions = [region for region in base_details["regions"] if not bool(region["closed"])]
    if len(closed_regions) < 2:
        return None

    force_tie = rng.random() < 0.55
    chosen = _choose_seed_layout(
        closed_regions,
        base_details["pos_to_incidence"],
        rng,
        force_tie,
    )
    if chosen is None:
        return None
    assignments, chosen_modes, chosen_colors = chosen

    for idx, region in enumerate(closed_regions):
        seed = assignments[idx]
        if grid[seed] != BG:
            return None
        grid[seed] = SEED
        region["expected_seed"] = seed
        region["expected_mode"] = chosen_modes[idx]
        region["expected_color"] = chosen_colors[idx]

    outside_seeds = _place_open_seeds(grid, open_regions, base_details["line_mask"], rng)

    solved = _solve_details(grid)
    out = np.asarray(solved["output"], dtype=int)

    solved_closed = [region for region in solved["regions"] if bool(region["closed"])]
    if len(solved_closed) != len(closed_regions):
        return None
    if any(len(region["seeds"]) != 1 for region in solved_closed):
        return None
    if int(solved["crossing_count"]) < 3:
        return None

    fill_colors = [int(region["fill_color"]) for region in solved_closed if region["fill_color"] is not None]
    fill_modes = [str(region["fill_mode"]) for region in solved_closed if region["fill_mode"] is not None]
    if len(set(fill_colors)) < 2:
        return None
    if "horizontal" not in fill_modes:
        return None
    if force_tie and "tie_vertical" not in fill_modes:
        return None
    if not force_tie and "vertical" not in fill_modes:
        return None

    for pos in outside_seeds:
        if int(out[pos]) != SEED:
            return None

    if not np.array_equal(main(grid), out):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return np.asarray(grid, dtype=int)
    raise RuntimeError("failed to generate a valid enclosed-line flood-fill input")
