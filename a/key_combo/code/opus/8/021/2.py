from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
LINE = 3
SEED_COLORS = (1, 2, 4, 6)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _mirror_quarter(quarter: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [np.flipud(np.fliplr(quarter)), np.flipud(quarter)],
            [np.fliplr(quarter), quarter],
        ]
    )


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _enclosed_cavities(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    cavities: list[list[tuple[int, int]]] = []
    for pts in _components(grid == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        cavities.append(pts)
    return cavities


def _cavity_label_map(cavities: list[list[tuple[int, int]]], shape: tuple[int, int]) -> np.ndarray:
    labels = np.full(shape, -1, dtype=int)
    for idx, pts in enumerate(cavities):
        for r, c in pts:
            labels[r, c] = idx
    return labels


def _boundary_seeds(grid: np.ndarray, cavity: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    seeds: set[tuple[int, int]] = set()
    for r, c in cavity:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                value = int(grid[nr, nc])
                if value not in (BG, LINE):
                    seeds.add((nr, nc))
    return sorted(seeds)


def _choose_fill_color(grid: np.ndarray, cavity: list[tuple[int, int]], seeds: list[tuple[int, int]]) -> int:
    counts = Counter(int(grid[r, c]) for r, c in seeds)
    max_count = max(counts.values())
    tied = {color for color, count in counts.items() if count == max_count}
    if len(tied) == 1:
        return int(next(iter(tied)))

    centroid_r = sum(r for r, _ in cavity) / len(cavity)
    centroid_c = sum(c for _, c in cavity) / len(cavity)
    best = min(
        (
            abs(r - centroid_r) + abs(c - centroid_c),
            r,
            c,
            int(grid[r, c]),
        )
        for r, c in seeds
        if int(grid[r, c]) in tied
    )
    return int(best[3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the lower-right line drawing into four quadrants, then fill each enclosed cavity by the boundary seed majority.

    Input structure:
    - The grid has even height and width and uses black as background.
    - Only the lower-right quadrant contains non-background pixels: a green orthogonal line drawing plus 2-4 distinct one-pixel seeds.
    - Seed pixels use colors chosen from blue, red, yellow, and magenta, and each seed is 8-neighbor-adjacent to the green drawing.

    Rule:
    - Reflect the entire lower-right quadrant across the vertical and horizontal center lines to populate the other three quadrants.
    - In the mirrored full grid, find every 4-connected background cavity that does not touch the outer border.
    - For each cavity, collect all non-green seed pixels on its 4-neighbor boundary and count seed colors.
    - Fill the whole cavity with the unique most frequent color; if there is a tie, choose the tied color whose boundary seed is closest to the cavity centroid by Manhattan distance, then by smaller row.
    - Cavities whose boundary touches no seeds stay black, and all green lines and seed pixels remain unchanged.

    Key insight:
    - The lower-right drawing is only a quarter blueprint: mirroring changes the topology by creating new closed chambers, and the mirrored seed copies then vote locally on each new chamber.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 0 and w % 2 == 0, "expected even grid dimensions"

    quarter = g[h // 2 :, w // 2 :].copy()
    full = _mirror_quarter(quarter)
    out = full.copy()

    for cavity in _enclosed_cavities(full):
        seeds = _boundary_seeds(full, cavity)
        if not seeds:
            continue
        color = _choose_fill_color(full, cavity, seeds)
        for r, c in cavity:
            out[r, c] = color

    return out


def _build_quarter(side: int, inner_row: int, inner_col: int, outer_row: int, outer_col: int) -> np.ndarray:
    quarter = np.zeros((side, side), dtype=int)
    quarter[0, 1 : outer_col + 1] = LINE
    quarter[1 : outer_row + 1, 0] = LINE
    corners = [(inner_row, inner_col), (inner_row, inner_col + 1), (outer_row, outer_col)]
    for r, c in corners:
        quarter[r, : c + 1] = LINE
        quarter[: r + 1, c] = LINE
    return quarter


def _mirror_seed_positions(side: int, pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, pos)
    return [
        (side - 1 - r, side - 1 - c),
        (side - 1 - r, side + c),
        (side + r, side - 1 - c),
        (side + r, side + c),
    ]


def _candidate_touch_sets(
    quarter: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], np.ndarray, dict[tuple[int, int], tuple[int, ...]], int]:
    side = int(quarter.shape[0])
    mirrored = _mirror_quarter(quarter)
    cavities = _enclosed_cavities(mirrored)
    labels = _cavity_label_map(cavities, mirrored.shape)

    center_r = (mirrored.shape[0] - 1) / 2.0
    center_c = (mirrored.shape[1] - 1) / 2.0
    central_idx = min(
        range(len(cavities)),
        key=lambda idx: abs(np.mean([r for r, _ in cavities[idx]]) - center_r)
        + abs(np.mean([c for _, c in cavities[idx]]) - center_c),
    )

    touch_sets: dict[tuple[int, int], tuple[int, ...]] = {}
    for r in range(side):
        for c in range(side):
            if int(quarter[r, c]) != BG:
                continue
            if not any(
                0 <= nr < side and 0 <= nc < side and int(quarter[nr, nc]) == LINE
                for nr, nc in _neighbors8(r, c)
            ):
                continue

            touched: set[int] = set()
            for mr, mc in _mirror_seed_positions(side, (r, c)):
                for nr, nc in _neighbors4(mr, mc):
                    if 0 <= nr < labels.shape[0] and 0 <= nc < labels.shape[1] and labels[nr, nc] >= 0:
                        touched.add(int(labels[nr, nc]))
            touch_sets[(r, c)] = tuple(sorted(touched))

    return cavities, labels, touch_sets, int(central_idx)


def _pick_outer_pair(candidates: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(candidates) < 2:
        return None
    best_pair = None
    best_key = None
    for i, p0 in enumerate(candidates):
        for p1 in candidates[i + 1 :]:
            dist = abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])
            same_axis = int(p0[0] == p1[0]) + int(p0[1] == p1[1])
            key = (dist, -same_axis, abs(p0[0] - p1[0]), abs(p0[1] - p1[1]))
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (p0, p1)
    return best_pair


def _cavity_vote_data(
    mirrored_with_seeds: np.ndarray, cavities: list[list[tuple[int, int]]]
) -> list[dict[str, object]]:
    data: list[dict[str, object]] = []
    for cavity in cavities:
        seeds = _boundary_seeds(mirrored_with_seeds, cavity)
        counts = Counter(int(mirrored_with_seeds[r, c]) for r, c in seeds)
        data.append({"cells": cavity, "seeds": seeds, "counts": counts})
    return data


def _semantic_checks(inp: np.ndarray, mode: str) -> bool:
    side = inp.shape[0] // 2
    quarter = inp[side:, side:]
    line_only = np.where(quarter == LINE, LINE, BG)

    quarter_cavities = _enclosed_cavities(line_only)
    full_line = _mirror_quarter(line_only)
    full_cavities = _enclosed_cavities(full_line)
    if len(full_cavities) < 5 or len(full_cavities) <= len(quarter_cavities):
        return False

    cavities, _labels, touch_sets, central_idx = _candidate_touch_sets(line_only)
    mirrored_with_seeds = _mirror_quarter(quarter)
    vote_data = _cavity_vote_data(mirrored_with_seeds, cavities)

    if not any(len(info["counts"]) >= 2 for info in vote_data):
        return False

    tied_multi = False
    unique_multi = False
    untouched = False
    for idx, info in enumerate(vote_data):
        counts: Counter[int] = info["counts"]  # type: ignore[assignment]
        if not counts:
            untouched = True
            continue
        values = sorted(counts.values(), reverse=True)
        if len(counts) >= 2 and values[0] == values[1]:
            tied_multi = True
        if len(counts) == 1 or (len(values) >= 2 and values[0] > values[1]):
            unique_multi = True
        if idx == central_idx and mode == "without_central" and counts:
            return False

    if mode != "without_central" and not unique_multi:
        return False
    if not tied_multi:
        return False
    if mode == "without_central" and not untouched:
        return False

    out = main(inp)
    if np.array_equal(out, inp):
        return False

    fill_colors = []
    for cavity in full_cavities:
        r, c = cavity[0]
        fill_colors.append(int(out[r, c]))
    if len(set(fill_colors)) < 2:
        return False

    if mode == "without_central":
        central_cells = cavities[central_idx]
        if any(int(out[r, c]) != BG for r, c in central_cells):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    side = random.randint(8, 12)
    inner_row = random.choice((1, 2))
    inner_col = random.choice((1, 2))
    if side < max(inner_row + 4, inner_col + 5):
        return None
    outer_row = random.randint(inner_row + 2, side - 3)
    outer_col = random.randint(inner_col + 2, side - 3)

    line_only = _build_quarter(side, inner_row, inner_col, outer_row, outer_col)
    cavities, _labels, touch_sets, central_idx = _candidate_touch_sets(line_only)

    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for pos, touched in touch_sets.items():
        groups.setdefault(touched, []).append(pos)

    central_positions = groups.get((central_idx,), [])
    inert_positions = groups.get((), [])
    outer_groups = [(ts, pts) for ts, pts in groups.items() if ts and ts != (central_idx,)]
    if not central_positions or not outer_groups:
        return None

    outer_touch_set, outer_positions = max(outer_groups, key=lambda item: (len(item[0]), len(item[1])))
    outer_pair = _pick_outer_pair(outer_positions)
    if outer_pair is None:
        return None

    mode = random.choices(
        ["with_central", "without_central", "with_central_and_inert"],
        weights=[5, 2, 3],
        k=1,
    )[0]

    n_colors = 4 if mode == "with_central_and_inert" else 3
    colors = random.sample(SEED_COLORS, n_colors)
    seeds: list[tuple[tuple[int, int], int]] = []

    if mode != "without_central":
        seeds.append((random.choice(central_positions), colors[0]))
        outer_colors = colors[1:3]
        next_color = 3
    else:
        outer_colors = colors[0:2]
        next_color = 2

    outer_a, outer_b = outer_pair
    if random.random() < 0.5:
        outer_a, outer_b = outer_b, outer_a
    seeds.append((outer_a, outer_colors[0]))
    seeds.append((outer_b, outer_colors[1]))

    if mode in ("without_central", "with_central_and_inert"):
        if not inert_positions:
            return None
        seeds.append((random.choice(inert_positions), colors[next_color]))

    quarter = line_only.copy()
    for (r, c), color in seeds:
        if int(quarter[r, c]) != BG:
            return None
        quarter[r, c] = int(color)

    inp = np.zeros((2 * side, 2 * side), dtype=int)
    inp[side:, side:] = quarter

    if not _semantic_checks(inp, mode):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
