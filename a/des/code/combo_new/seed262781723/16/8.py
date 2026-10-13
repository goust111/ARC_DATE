from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
FILL_POOL = (3, 4, 5, 6, 7, 8, 9)

LEGEND_COUNTS = 6
LEGEND_CELL_W = 3
LEGEND_H = 5
LEGEND_W = 1 + LEGEND_COUNTS * LEGEND_CELL_W + (LEGEND_COUNTS - 1) + 1

MIN_REGION_SIDE = 3
MIN_REGIONS = 6
MAX_REGIONS = 9
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _parse_legend(
    grid: np.ndarray, box: tuple[int, int, int, int], bg: int
) -> dict[int, int] | None:
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape != (LEGEND_H, LEGEND_W):
        return None
    if not np.all(patch[0, :] == BLUE) or not np.all(patch[-1, :] == BLUE):
        return None

    sep_cols = [c for c in range(patch.shape[1]) if np.all(patch[:, c] == BLUE)]
    if len(sep_cols) != LEGEND_COUNTS + 1:
        return None
    if sep_cols[0] != 0 or sep_cols[-1] != patch.shape[1] - 1:
        return None

    spans: list[tuple[int, int]] = []
    for left, right in zip(sep_cols, sep_cols[1:]):
        if right - left != LEGEND_CELL_W + 1:
            return None
        spans.append((left + 1, right - 1))

    mapping: dict[int, int] = {}
    for s0, s1 in spans:
        cell = patch[1:-1, s0 : s1 + 1]
        if cell.shape != (3, 3):
            return None
        marker_block = cell[:2, :]
        if np.any(~np.isin(marker_block, [bg, RED])):
            return None
        color_row = cell[2, :]
        color_values = sorted(set(map(int, color_row)))
        if len(color_values) != 1:
            return None
        fill_color = color_values[0]
        if fill_color in (bg, BLUE, RED):
            return None
        count = int(np.count_nonzero(marker_block == RED))
        if count in mapping:
            return None
        mapping[count] = fill_color

    if set(mapping) != set(range(LEGEND_COUNTS)):
        return None
    return mapping


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    h, _w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], dict[int, int]]] = []
    for comp in _components(grid == BLUE):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        if r1 != h - 1:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        extra_colors = set(map(int, np.unique(patch))) - {bg, BLUE, RED}
        if len(extra_colors) < LEGEND_COUNTS:
            continue
        mapping = _parse_legend(grid, box, bg)
        if mapping is not None:
            candidates.append((box, mapping))
    assert len(candidates) == 1, "expected exactly one parseable legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the legend color keyed by its red marker count.

    Input structure:
    - The background is the dominant color.
    - One blue rectilinear line structure partitions the upper area into closed regions.
    - A separate blue-framed legend on one side contains cells with 0..5 red dots and one fill color strip per cell.

    Rule:
    - Read the legend to build a mapping from red-dot count to fill color.
    - Treat blue pixels as barriers and split the remaining background/red cells into 4-connected regions.
    - Ignore components that touch the grid border and ignore the legend area itself.
    - For every enclosed region, count its red marker pixels and fill that region's background pixels with the mapped legend color, leaving blue lines, red markers, and the legend unchanged.

    Key insight:
    - The relevant property is not a region's shape or size but the number of red marker pixels inside it, and the legend supplies the color corresponding to that count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    legend_box, mapping = _find_legend(g, bg)
    lr0, lc0, lr1, lc1 = legend_box

    region_mask = (g == bg) | (g == RED)
    region_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = False

    out = g.copy()
    for comp in _components(region_mask):
        touches_border = np.any(
            (comp[:, 0] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == 0) | (comp[:, 1] == w - 1)
        )
        if touches_border:
            continue
        count = int(np.count_nonzero(g[comp[:, 0], comp[:, 1]] == RED))
        assert count in mapping, "region count missing from legend"
        fill_color = int(mapping[count])
        bg_mask = g[comp[:, 0], comp[:, 1]] == bg
        out[comp[bg_mask, 0], comp[bg_mask, 1]] = fill_color

    return out


def _split_orientations(region: tuple[int, int, int, int]) -> list[str]:
    r0, c0, r1, c1 = region
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    choices: list[str] = []
    if width >= 2 * MIN_REGION_SIDE + 1:
        choices.append("v")
    if height >= 2 * MIN_REGION_SIDE + 1:
        choices.append("h")
    return choices


def _partition_patch(board_h: int, board_w: int, rng: random.Random) -> np.ndarray | None:
    patch = np.full((board_h, board_w), BG, dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE

    regions = [(1, 1, board_h - 2, board_w - 2)]
    target = rng.randint(MIN_REGIONS, MAX_REGIONS)

    while len(regions) < target:
        candidates = [(idx, _split_orientations(region)) for idx, region in enumerate(regions)]
        candidates = [(idx, choices) for idx, choices in candidates if choices]
        if not candidates:
            break

        weights = []
        for idx, _choices in candidates:
            r0, c0, r1, c1 = regions[idx]
            weights.append((r1 - r0 + 1) * (c1 - c0 + 1))
        chosen_idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        region_idx, orientations = candidates[chosen_idx]
        r0, c0, r1, c1 = regions.pop(region_idx)
        orient = rng.choice(orientations)

        if orient == "v":
            split_c = rng.randint(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE)
            patch[r0 - 1 : r1 + 2, split_c] = BLUE
            regions.append((r0, c0, r1, split_c - 1))
            regions.append((r0, split_c + 1, r1, c1))
        else:
            split_r = rng.randint(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE)
            patch[split_r, c0 - 1 : c1 + 2] = BLUE
            regions.append((r0, c0, split_r - 1, c1))
            regions.append((split_r + 1, c0, r1, c1))

    comps = _components(patch == BG)
    if len(comps) < MIN_REGIONS:
        return None
    if any(len(comp) < LEGEND_COUNTS for comp in comps):
        return None
    return patch


def _spread_points(coords: np.ndarray, count: int, rng: random.Random) -> list[tuple[int, int]]:
    points = [tuple(map(int, xy)) for xy in coords]
    if count == 0:
        return []
    assert count <= len(points)

    chosen = [rng.choice(points)]
    remaining = [pt for pt in points if pt != chosen[0]]
    while len(chosen) < count:
        best_score = -1
        best: list[tuple[int, int]] = []
        for pt in remaining:
            score = min(abs(pt[0] - rr) + abs(pt[1] - cc) for rr, cc in chosen)
            if score > best_score:
                best_score = score
                best = [pt]
            elif score == best_score:
                best.append(pt)
        pick = rng.choice(best)
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def _legend_patch(mapping: dict[int, int], rng: random.Random) -> np.ndarray:
    patch = np.full((LEGEND_H, LEGEND_W), BG, dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE

    order = list(range(LEGEND_COUNTS))
    rng.shuffle(order)
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for idx, count in enumerate(order):
        start = 1 + idx * (LEGEND_CELL_W + 1)
        end = start + LEGEND_CELL_W
        patch[:, start - 1] = BLUE
        patch[:, end] = BLUE

        cell = patch[1:-1, start:end]
        cell[2, :] = int(mapping[count])
        marker_positions = rng.sample(slots, k=count)
        for rr, cc in marker_positions:
            cell[rr, cc] = RED

    return patch


def _build_expected(
    inp: np.ndarray,
    region_comps: list[np.ndarray],
    counts: list[int],
    mapping: dict[int, int],
    board_top: int,
    board_left: int,
) -> np.ndarray:
    out = inp.copy()
    for comp, count in zip(region_comps, counts):
        full_rows = comp[:, 0] + board_top
        full_cols = comp[:, 1] + board_left
        bg_mask = inp[full_rows, full_cols] == BG
        out[full_rows[bg_mask], full_cols[bg_mask]] = int(mapping[count])
    return out


def _generate_attempt() -> np.ndarray | None:
    rng = random.Random()

    width = rng.randint(25, 30)
    top_margin = rng.randint(1, 2)
    gap = rng.randint(1, 2)
    max_board_h = 30 - LEGEND_H - top_margin - gap
    if max_board_h < 14:
        return None

    board_h = rng.randint(14, max_board_h)
    board_w = rng.randint(16, width - 2)
    board_patch = _partition_patch(board_h, board_w, rng)
    if board_patch is None:
        return None

    region_comps = _components(board_patch == BG)
    if len(region_comps) < MIN_REGIONS:
        return None

    height = top_margin + board_h + gap + LEGEND_H
    board_top = top_margin
    board_left = rng.randint(1, width - board_w - 1)
    legend_top = height - LEGEND_H
    legend_left = rng.randint(0, width - LEGEND_W)

    fill_colors = rng.sample(FILL_POOL, k=LEGEND_COUNTS)
    counts_to_color = {count: color for count, color in zip(range(LEGEND_COUNTS), fill_colors)}

    counts = list(range(LEGEND_COUNTS))
    while len(counts) < len(region_comps):
        counts.append(rng.randint(0, LEGEND_COUNTS - 1))
    rng.shuffle(counts)

    inp = np.full((height, width), BG, dtype=int)
    inp[board_top : board_top + board_h, board_left : board_left + board_w] = board_patch
    inp[legend_top : legend_top + LEGEND_H, legend_left : legend_left + LEGEND_W] = _legend_patch(
        counts_to_color, rng
    )

    for comp, count in zip(region_comps, counts):
        marker_cells = _spread_points(comp, count, rng)
        for rr, cc in marker_cells:
            inp[board_top + rr, board_left + cc] = RED

    expected = _build_expected(inp, region_comps, counts, counts_to_color, board_top, board_left)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
