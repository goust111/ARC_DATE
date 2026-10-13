from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


MIN_RECT_SIDE = 3
BOARD_SIDE_RANGE = (13, 24)
REGION_COUNT_RANGE = (4, 6)
MARGIN_RANGE = (1, 3)
GEN_RETRIES = 300

BG_COLOR = 0
LINE_COLOR = 5
STRIPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int, int]:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if lo > hi:
        return 0, 0, -1
    return hi - lo + 1, lo, hi


def _infer_pattern(patch: np.ndarray, r0: int, c0: int) -> dict[str, object]:
    colors = np.unique(patch)
    assert colors.size == 2, "each region must contain exactly two stripe colors"

    row_constant = bool(np.all(patch == patch[:, [0]]))
    col_constant = bool(np.all(patch == patch[[0], :]))
    assert row_constant != col_constant, "stripes must be unambiguously horizontal or vertical"

    if row_constant:
        row_vals = patch[:, 0]
        first = int(row_vals[0])
        second = int(row_vals[1])
        assert first != second, "horizontal stripes need two alternating row colors"
        expected = np.array([first if i % 2 == 0 else second for i in range(len(row_vals))], dtype=int)
        assert np.array_equal(row_vals, expected), "rows must alternate cleanly"
        return {"orientation": "h", "anchor": int(r0), "colors": (first, second)}

    col_vals = patch[0, :]
    first = int(col_vals[0])
    second = int(col_vals[1])
    assert first != second, "vertical stripes need two alternating column colors"
    expected = np.array([first if i % 2 == 0 else second for i in range(len(col_vals))], dtype=int)
    assert np.array_equal(col_vals, expected), "columns must alternate cleanly"
    return {"orientation": "v", "anchor": int(c0), "colors": (first, second)}


def _extract_regions(grid: np.ndarray) -> tuple[int, int, tuple[int, int, int, int], list[dict[str, object]]]:
    bg = int(Counter(map(int, (grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]))).most_common(1)[0][0])
    board_bbox = _bbox_from_mask(grid != bg)
    assert board_bbox is not None, "expected a framed board inside the background"
    br0, bc0, br1, bc1 = board_bbox
    board = grid[br0 : br1 + 1, bc0 : bc1 + 1]

    line_color = int(board[0, 0])
    assert np.all(board[0, :] == line_color)
    assert np.all(board[-1, :] == line_color)
    assert np.all(board[:, 0] == line_color)
    assert np.all(board[:, -1] == line_color)

    regions: list[dict[str, object]] = []
    for comp in _components(board != line_color):
        rr0, cc0 = comp.min(axis=0)
        rr1, cc1 = comp.max(axis=0)
        height = int(rr1 - rr0 + 1)
        width = int(cc1 - cc0 + 1)
        assert len(comp) == height * width, "each non-line component must be a solid rectangle"

        r0 = br0 + int(rr0)
        c0 = bc0 + int(cc0)
        r1 = br0 + int(rr1)
        c1 = bc0 + int(cc1)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "area": height * width,
                "pattern": _infer_pattern(patch, r0, c0),
            }
        )

    assert regions, "expected at least one striped rectangle"
    return bg, line_color, board_bbox, regions


def _build_adjacency(
    grid: np.ndarray, line_color: int, regions: list[dict[str, object]]
) -> dict[int, list[dict[str, int | str]]]:
    adjacency: dict[int, list[dict[str, int | str]]] = {idx: [] for idx in range(len(regions))}
    for i in range(len(regions)):
        ar0, ac0, ar1, ac1 = regions[i]["bbox"]
        for j in range(i + 1, len(regions)):
            br0, bc0, br1, bc1 = regions[j]["bbox"]

            if ar1 + 2 == br0:
                shared, lo, hi = _interval_overlap(ac0, ac1, bc0, bc1)
                if shared:
                    assert np.all(grid[ar1 + 1, lo : hi + 1] == line_color)
                    adjacency[i].append({"neighbor": j, "dir": "down", "shared": shared})
                    adjacency[j].append({"neighbor": i, "dir": "up", "shared": shared})
            elif br1 + 2 == ar0:
                shared, lo, hi = _interval_overlap(ac0, ac1, bc0, bc1)
                if shared:
                    assert np.all(grid[br1 + 1, lo : hi + 1] == line_color)
                    adjacency[i].append({"neighbor": j, "dir": "up", "shared": shared})
                    adjacency[j].append({"neighbor": i, "dir": "down", "shared": shared})

            if ac1 + 2 == bc0:
                shared, lo, hi = _interval_overlap(ar0, ar1, br0, br1)
                if shared:
                    assert np.all(grid[lo : hi + 1, ac1 + 1] == line_color)
                    adjacency[i].append({"neighbor": j, "dir": "right", "shared": shared})
                    adjacency[j].append({"neighbor": i, "dir": "left", "shared": shared})
            elif bc1 + 2 == ac0:
                shared, lo, hi = _interval_overlap(ar0, ar1, br0, br1)
                if shared:
                    assert np.all(grid[lo : hi + 1, bc1 + 1] == line_color)
                    adjacency[i].append({"neighbor": j, "dir": "left", "shared": shared})
                    adjacency[j].append({"neighbor": i, "dir": "right", "shared": shared})
    return adjacency


def _pattern_values(pattern: dict[str, object], axis: np.ndarray) -> np.ndarray:
    first, second = pattern["colors"]
    anchor = int(pattern["anchor"])
    return np.where((axis - anchor) % 2 == 0, int(first), int(second)).astype(int)


def _paint_from_neighbor(
    out: np.ndarray,
    target: dict[str, object],
    source: dict[str, object],
    source_pattern: dict[str, object],
    direction: str,
) -> None:
    tr0, tc0, tr1, tc1 = target["bbox"]
    sr0, sc0, sr1, sc1 = source["bbox"]

    if source_pattern["orientation"] == "h":
        rows = np.arange(tr0, tr1 + 1, dtype=int)
        if direction == "up":
            sample_rows = sr1 - (rows - tr0)
        elif direction == "down":
            sample_rows = sr0 + (tr1 - rows)
        else:
            sample_rows = rows
        row_colors = _pattern_values(source_pattern, sample_rows)
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = row_colors[:, None]
        return

    cols = np.arange(tc0, tc1 + 1, dtype=int)
    if direction == "left":
        sample_cols = sc1 - (cols - tc0)
    elif direction == "right":
        sample_cols = sc0 + (tc1 - cols)
    else:
        sample_cols = cols
    col_colors = _pattern_values(source_pattern, sample_cols)
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = col_colors[None, :]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each smaller striped rectangle by the pattern of its largest processed neighbor while continuing the stripe phase across their shared border.

    Input structure:
    - A uniform background surrounds one framed board made of one-pixel separator lines.
    - The separator color forms the outer border and internal rectilinear grid lines that partition the board into rectangles.
    - Every rectangle interior is fully filled by exactly one two-color stripe pattern, either horizontal or vertical.

    Rule:
    - Identify every rectangular interior separated by the line color and compute its area.
    - Process rectangles from largest area to smallest.
    - The largest rectangle keeps its own pattern. Each later rectangle looks at already processed adjacent rectangles that share one separator edge with it.
    - If such larger neighbors exist, pick the one with the longest shared edge and repaint the current rectangle using that neighbor's current stripe pattern.
    - Keep stripe phase continuous from the shared edge: the copied pattern is aligned so the boundary-adjacent stripes match across the separator line.
    - Background and separator lines stay unchanged.

    Key insight:
    - The task is not simple pattern copying by rectangle identity; it requires reconstructing the rectangle adjacency graph, ordering rectangles by area, and extending a neighbor's stripe phase from the common border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    _bg, line_color, _board_bbox, regions = _extract_regions(g)
    adjacency = _build_adjacency(g, line_color, regions)

    areas = [int(region["area"]) for region in regions]
    current_patterns = {idx: regions[idx]["pattern"] for idx in range(len(regions))}
    order = sorted(
        range(len(regions)),
        key=lambda idx: (-areas[idx], regions[idx]["bbox"][0], regions[idx]["bbox"][1]),
    )

    processed: set[int] = set()
    for idx in order:
        candidates = [
            item
            for item in adjacency[idx]
            if item["neighbor"] in processed and areas[item["neighbor"]] > areas[idx]
        ]
        if candidates:
            best = max(
                candidates,
                key=lambda item: (int(item["shared"]), areas[item["neighbor"]], -int(item["neighbor"])),
            )
            source_idx = int(best["neighbor"])
            _paint_from_neighbor(out, regions[idx], regions[source_idx], current_patterns[source_idx], str(best["dir"]))
            r0, c0, r1, c1 = regions[idx]["bbox"]
            current_patterns[idx] = _infer_pattern(out[r0 : r1 + 1, c0 : c1 + 1], r0, c0)
        processed.add(idx)

    return out


def _rect_area(region: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = region
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _possible_splits(region: tuple[int, int, int, int]) -> list[str]:
    r0, c0, r1, c1 = region
    options: list[str] = []
    if r1 - r0 + 1 >= 2 * MIN_RECT_SIDE + 1:
        options.append("h")
    if c1 - c0 + 1 >= 2 * MIN_RECT_SIDE + 1:
        options.append("v")
    return options


def _split_region(rng: random.Random, region: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    options = _possible_splits(region)
    if not options:
        return None

    r0, c0, r1, c1 = region
    orientation = rng.choice(options)
    if orientation == "h":
        height = r1 - r0 + 1
        choices = list(range(MIN_RECT_SIDE, height - MIN_RECT_SIDE))
        small = rng.choice(choices)
        if rng.random() < 0.5:
            cut = r0 + small
        else:
            cut = r1 - small
        upper = (r0, c0, cut - 1, c1)
        lower = (cut + 1, c0, r1, c1)
        return upper, lower

    width = c1 - c0 + 1
    choices = list(range(MIN_RECT_SIDE, width - MIN_RECT_SIDE))
    small = rng.choice(choices)
    if rng.random() < 0.5:
        cut = c0 + small
    else:
        cut = c1 - small
    left = (r0, c0, r1, cut - 1)
    right = (r0, cut + 1, r1, c1)
    return left, right


def _adjacency_from_bboxes(regions: list[tuple[int, int, int, int]]) -> dict[int, list[dict[str, int | str]]]:
    adjacency: dict[int, list[dict[str, int | str]]] = {idx: [] for idx in range(len(regions))}
    for i in range(len(regions)):
        ar0, ac0, ar1, ac1 = regions[i]
        for j in range(i + 1, len(regions)):
            br0, bc0, br1, bc1 = regions[j]

            if ar1 + 2 == br0:
                shared, _, _ = _interval_overlap(ac0, ac1, bc0, bc1)
                if shared:
                    adjacency[i].append({"neighbor": j, "dir": "down", "shared": shared})
                    adjacency[j].append({"neighbor": i, "dir": "up", "shared": shared})
            elif br1 + 2 == ar0:
                shared, _, _ = _interval_overlap(ac0, ac1, bc0, bc1)
                if shared:
                    adjacency[i].append({"neighbor": j, "dir": "up", "shared": shared})
                    adjacency[j].append({"neighbor": i, "dir": "down", "shared": shared})

            if ac1 + 2 == bc0:
                shared, _, _ = _interval_overlap(ar0, ar1, br0, br1)
                if shared:
                    adjacency[i].append({"neighbor": j, "dir": "right", "shared": shared})
                    adjacency[j].append({"neighbor": i, "dir": "left", "shared": shared})
            elif bc1 + 2 == ac0:
                shared, _, _ = _interval_overlap(ar0, ar1, br0, br1)
                if shared:
                    adjacency[i].append({"neighbor": j, "dir": "left", "shared": shared})
                    adjacency[j].append({"neighbor": i, "dir": "right", "shared": shared})
    return adjacency


def _valid_partition(regions: list[tuple[int, int, int, int]]) -> bool:
    areas = [_rect_area(region) for region in regions]
    if len(set(areas)) != len(areas):
        return False

    largest = max(areas)
    if areas.count(largest) != 1:
        return False

    adjacency = _adjacency_from_bboxes(regions)
    for idx, area in enumerate(areas):
        larger = [item for item in adjacency[idx] if areas[item["neighbor"]] > area]
        if area != largest and not larger:
            return False
        if larger:
            best_shared = max(int(item["shared"]) for item in larger)
            if sum(int(item["shared"]) == best_shared for item in larger) != 1:
                return False
    return True


def _build_partition(
    rng: random.Random, board_h: int, board_w: int, region_count: int
) -> list[tuple[int, int, int, int]] | None:
    regions = [(1, 1, board_h - 2, board_w - 2)]
    for _ in range(region_count - 1):
        candidates = [
            (idx, _rect_area(region))
            for idx, region in enumerate(regions)
            if _possible_splits(region)
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        pick_count = min(len(candidates), 3)
        chosen_idx = rng.choice([item[0] for item in candidates[:pick_count]])
        region = regions.pop(chosen_idx)
        split = _split_region(rng, region)
        if split is None:
            return None
        regions.extend(split)

    return regions if _valid_partition(regions) else None


def _paint_initial_pattern(
    board: np.ndarray,
    region: tuple[int, int, int, int],
    orientation: str,
    pair: tuple[int, int],
    first_index: int,
) -> None:
    r0, c0, r1, c1 = region
    first = pair[first_index]
    second = pair[1 - first_index]

    if orientation == "h":
        colors = np.array(
            [first if offset % 2 == 0 else second for offset in range(r1 - r0 + 1)],
            dtype=int,
        )
        board[r0 : r1 + 1, c0 : c1 + 1] = colors[:, None]
        return

    colors = np.array(
        [first if offset % 2 == 0 else second for offset in range(c1 - c0 + 1)],
        dtype=int,
    )
    board[r0 : r1 + 1, c0 : c1 + 1] = colors[None, :]


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    board_h = rng.randint(*BOARD_SIDE_RANGE)
    board_w = rng.randint(*BOARD_SIDE_RANGE)
    region_count = rng.randint(*REGION_COUNT_RANGE)

    regions = _build_partition(rng, board_h, board_w, region_count)
    if regions is None:
        return None

    margin_top = rng.randint(*MARGIN_RANGE)
    margin_bottom = rng.randint(*MARGIN_RANGE)
    margin_left = rng.randint(*MARGIN_RANGE)
    margin_right = rng.randint(*MARGIN_RANGE)
    total_h = board_h + margin_top + margin_bottom
    total_w = board_w + margin_left + margin_right
    if total_h > 30 or total_w > 30:
        return None

    board = np.full((board_h, board_w), LINE_COLOR, dtype=int)

    pairs = rng.sample(list(combinations(STRIPE_COLORS, 2)), len(regions))
    orientations = [rng.choice(["h", "v"]) for _ in regions]
    if len(set(orientations)) == 1:
        orientations[rng.randrange(len(orientations))] = "h" if orientations[0] == "v" else "v"

    for region, pair, orientation in zip(regions, pairs, orientations):
        _paint_initial_pattern(board, region, orientation, pair, rng.randrange(2))

    grid = np.full((total_h, total_w), BG_COLOR, dtype=int)
    grid[margin_top : margin_top + board_h, margin_left : margin_left + board_w] = board

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
