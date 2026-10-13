import random
from collections import deque

import numpy as np


BACKGROUND = 8  # azure
ORANGE = 7
ISLAND_COLORS = (1, 2, 3, 4, 6)  # blue, red, green, yellow, magenta

GEN_RETRIES = 400
PAIR_RETRIES = 150
EXTRA_RETRIES = 80
MIN_REGION_SPAN = 8


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_islands(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    mask = (grid != BACKGROUND) & (grid != ORANGE)
    comps = _components(mask)
    owner = np.full(grid.shape, -1, dtype=int)
    islands: list[dict] = []

    for island_id, coords in enumerate(comps):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(grid[coords[0]])
        assert all(int(grid[r, c]) == color for r, c in coords), "island colors must be uniform"
        area = len(coords)
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "islands must be solid rectangles"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "islands must be filled rectangles"
        for r, c in coords:
            owner[r, c] = island_id
        islands.append(
            {
                "id": island_id,
                "color": color,
                "area": area,
                "bbox": (r0, c0, r1, c1),
            }
        )

    return islands, owner


def _scan_to_island(
    grid: np.ndarray,
    owner: np.ndarray,
    start: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[int, tuple[int, int]] | None:
    h, w = grid.shape
    r, c = start
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value != BACKGROUND and value != ORANGE:
            return int(owner[r, c]), (r, c)
        r += dr
        c += dc
    return None


def _bridge_spec_for_marker(grid: np.ndarray, islands: list[dict], owner: np.ndarray, marker: tuple[int, int]) -> dict:
    r, c = marker
    hits = {
        "up": _scan_to_island(grid, owner, marker, -1, 0),
        "down": _scan_to_island(grid, owner, marker, 1, 0),
        "left": _scan_to_island(grid, owner, marker, 0, -1),
        "right": _scan_to_island(grid, owner, marker, 0, 1),
    }

    horizontal = hits["left"] is not None and hits["right"] is not None
    vertical = hits["up"] is not None and hits["down"] is not None
    assert horizontal ^ vertical, "each marker must align with exactly one island pair"

    islands_by_id = {island["id"]: island for island in islands}

    if horizontal:
        assert hits["up"] is None and hits["down"] is None, "horizontal bridge marker must not align vertically"
        left_id, left_edge = hits["left"]
        right_id, right_edge = hits["right"]
        assert left_id != right_id, "marker must connect two distinct islands"
        cells = [(r, cc) for cc in range(left_edge[1] + 1, right_edge[1])]
        pair = (islands_by_id[left_id], islands_by_id[right_id])
        orientation = "horizontal"
    else:
        assert hits["left"] is None and hits["right"] is None, "vertical bridge marker must not align horizontally"
        up_id, up_edge = hits["up"]
        down_id, down_edge = hits["down"]
        assert up_id != down_id, "marker must connect two distinct islands"
        cells = [(rr, c) for rr in range(up_edge[0] + 1, down_edge[0])]
        pair = (islands_by_id[up_id], islands_by_id[down_id])
        orientation = "vertical"

    assert cells and marker in cells, "bridge path must contain the marker itself"
    area_a = int(pair[0]["area"])
    area_b = int(pair[1]["area"])
    assert area_a != area_b, "bridge color must be determined by a strict area comparison"
    color = int(pair[0]["color"]) if area_a < area_b else int(pair[1]["color"])

    return {
        "marker": marker,
        "orientation": orientation,
        "color": color,
        "cells": cells,
    }


def _bridge_specs(grid: np.ndarray) -> list[dict]:
    islands, owner = _extract_islands(grid)
    assert islands, "expected at least one island"
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)]
    assert markers, "expected at least one bridge marker"

    claimed: dict[tuple[int, int], int] = {}
    specs: list[dict] = []
    for marker in sorted(markers):
        spec = _bridge_spec_for_marker(grid, islands, owner, marker)
        for cell in spec["cells"]:
            assert int(grid[cell]) in (BACKGROUND, ORANGE), "bridge path must be empty in the input"
            assert cell not in claimed, "bridge paths must not overlap"
            claimed[cell] = int(spec["color"])
        specs.append(spec)
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build one-cell-wide bridges between aligned colored islands using the smaller island's color.

    Input structure:
    - The background is azure.
    - Several solid rectangular islands of distinct colors appear on the grid.
    - Orange single-pixel bridge markers sit in the empty gap between two islands
      that line up in the same row or the same column.

    Rule:
    - For each orange marker, find the nearest island on both sides along its
      aligned row or column.
    - Fill the straight one-pixel-wide gap from the marker out to both island
      edges without overwriting the islands themselves.
    - Replace the marker with the bridge color.
    - The bridge color is the color of the smaller-area island among the two
      connected islands.

    Key insight:
    - The orange pixel identifies which two islands should be paired, while the
      bridge color is decided by a visual size comparison between those islands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for spec in _bridge_specs(g):
        for r, c in spec["cells"]:
            out[r, c] = int(spec["color"])
    return out


def _render_scene(height: int, width: int, islands: list[dict], markers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for island in islands:
        r0, c0, r1, c1 = island["bbox"]
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BACKGROUND), "islands must not overlap"
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(island["color"])
    for r, c in markers:
        assert int(grid[r, c]) == BACKGROUND, "marker must be placed on background"
        grid[r, c] = ORANGE
    return grid


def _scene_is_valid(height: int, width: int, islands: list[dict], markers: list[tuple[int, int]]) -> bool:
    try:
        grid = _render_scene(height, width, islands, markers)
        specs = _bridge_specs(grid)
        return len(specs) == len(markers)
    except AssertionError:
        return False


def _make_regions(height: int, width: int, pair_count: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    if pair_count == 1:
        return [(1, 1, height - 2, width - 2)]

    if rng.random() < 0.5:
        gap = rng.randint(2, 4)
        max_top = height - 2 - gap - MIN_REGION_SPAN
        if max_top < MIN_REGION_SPAN:
            return None
        top_h = rng.randint(MIN_REGION_SPAN, max_top)
        r1_end = top_h
        r2_start = r1_end + gap + 1
        return [
            (1, 1, r1_end, width - 2),
            (r2_start, 1, height - 2, width - 2),
        ]

    gap = rng.randint(2, 4)
    max_left = width - 2 - gap - MIN_REGION_SPAN
    if max_left < MIN_REGION_SPAN:
        return None
    left_w = rng.randint(MIN_REGION_SPAN, max_left)
    c1_end = left_w
    c2_start = c1_end + gap + 1
    return [
        (1, 1, height - 2, c1_end),
        (1, c2_start, height - 2, width - 2),
    ]


def _pick_dims(
    rng: random.Random,
    max_h: int,
    max_w: int,
    smaller_first: bool,
) -> tuple[int, int, int, int] | None:
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(80):
        h1 = rng.randint(2, max_h)
        w1 = rng.randint(2, max_w)
        h2 = rng.randint(2, max_h)
        w2 = rng.randint(2, max_w)
        area1 = h1 * w1
        area2 = h2 * w2
        if area1 == area2:
            continue
        if (area1 < area2) == smaller_first:
            return h1, w1, h2, w2
    return None


def _place_horizontal_pair(
    region: tuple[int, int, int, int],
    color_a: int,
    color_b: int,
    smaller_first: bool,
    rng: random.Random,
) -> dict | None:
    r0, c0, r1, c1 = region
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    max_h = min(5, region_h)
    max_w = min(6, region_w)

    for _ in range(120):
        dims = _pick_dims(rng, max_h, max_w, smaller_first)
        if dims is None:
            return None
        h1, w1, h2, w2 = dims
        max_gap = min(9, region_w - w1 - w2)
        gap_choices = [gap for gap in range(3, max_gap + 1, 2)]
        if not gap_choices:
            continue
        gap = rng.choice(gap_choices)
        total_w = w1 + gap + w2
        start_c = rng.randint(c0, c1 - total_w + 1)
        bridge_r = rng.randint(r0, r1)

        top1_min = max(r0, bridge_r - h1 + 1)
        top1_max = min(bridge_r, r1 - h1 + 1)
        top2_min = max(r0, bridge_r - h2 + 1)
        top2_max = min(bridge_r, r1 - h2 + 1)
        if top1_min > top1_max or top2_min > top2_max:
            continue
        top1 = rng.randint(top1_min, top1_max)
        top2 = rng.randint(top2_min, top2_max)

        left_bbox = (top1, start_c, top1 + h1 - 1, start_c + w1 - 1)
        right_start = start_c + w1 + gap
        right_bbox = (top2, right_start, top2 + h2 - 1, right_start + w2 - 1)
        marker = (bridge_r, start_c + w1 + gap // 2)

        return {
            "islands": [
                {"color": color_a, "bbox": left_bbox},
                {"color": color_b, "bbox": right_bbox},
            ],
            "marker": marker,
        }
    return None


def _place_vertical_pair(
    region: tuple[int, int, int, int],
    color_a: int,
    color_b: int,
    smaller_first: bool,
    rng: random.Random,
) -> dict | None:
    r0, c0, r1, c1 = region
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    max_h = min(6, region_h)
    max_w = min(5, region_w)

    for _ in range(120):
        dims = _pick_dims(rng, max_h, max_w, smaller_first)
        if dims is None:
            return None
        h1, w1, h2, w2 = dims
        max_gap = min(9, region_h - h1 - h2)
        gap_choices = [gap for gap in range(3, max_gap + 1, 2)]
        if not gap_choices:
            continue
        gap = rng.choice(gap_choices)
        total_h = h1 + gap + h2
        start_r = rng.randint(r0, r1 - total_h + 1)
        bridge_c = rng.randint(c0, c1)

        left1_min = max(c0, bridge_c - w1 + 1)
        left1_max = min(bridge_c, c1 - w1 + 1)
        left2_min = max(c0, bridge_c - w2 + 1)
        left2_max = min(bridge_c, c1 - w2 + 1)
        if left1_min > left1_max or left2_min > left2_max:
            continue
        left1 = rng.randint(left1_min, left1_max)
        left2 = rng.randint(left2_min, left2_max)

        top_bbox = (start_r, left1, start_r + h1 - 1, left1 + w1 - 1)
        bottom_start = start_r + h1 + gap
        bottom_bbox = (bottom_start, left2, bottom_start + h2 - 1, left2 + w2 - 1)
        marker = (start_r + h1 + gap // 2, bridge_c)

        return {
            "islands": [
                {"color": color_a, "bbox": top_bbox},
                {"color": color_b, "bbox": bottom_bbox},
            ],
            "marker": marker,
        }
    return None


def _place_pair(
    region: tuple[int, int, int, int],
    orientation: str,
    color_a: int,
    color_b: int,
    smaller_first: bool,
    rng: random.Random,
) -> dict | None:
    if orientation == "horizontal":
        return _place_horizontal_pair(region, color_a, color_b, smaller_first, rng)
    return _place_vertical_pair(region, color_a, color_b, smaller_first, rng)


def _forbidden_axes(markers: list[tuple[int, int]], orientations: list[str]) -> tuple[set[int], set[int]]:
    forbidden_rows: set[int] = set()
    forbidden_cols: set[int] = set()
    for (r, c), orientation in zip(markers, orientations):
        if orientation == "vertical":
            forbidden_rows.add(r)
        else:
            forbidden_cols.add(c)
    return forbidden_rows, forbidden_cols


def _try_place_extra_island(
    height: int,
    width: int,
    islands: list[dict],
    markers: list[tuple[int, int]],
    orientations: list[str],
    color: int,
    rng: random.Random,
) -> dict | None:
    grid = _render_scene(height, width, islands, markers)
    forbidden_rows, forbidden_cols = _forbidden_axes(markers, orientations)

    for _ in range(EXTRA_RETRIES):
        box_h = rng.randint(2, min(5, height - 2))
        box_w = rng.randint(2, min(5, width - 2))
        top = rng.randint(1, height - box_h - 1)
        left = rng.randint(1, width - box_w - 1)
        bbox = (top, left, top + box_h - 1, left + box_w - 1)

        if any(top <= row <= top + box_h - 1 for row in forbidden_rows):
            continue
        if any(left <= col <= left + box_w - 1 for col in forbidden_cols):
            continue

        pad_r0 = max(0, top - 1)
        pad_c0 = max(0, left - 1)
        pad_r1 = min(height - 1, top + box_h)
        pad_c1 = min(width - 1, left + box_w)
        if np.any(grid[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] != BACKGROUND):
            continue
        return {"color": color, "bbox": bbox}
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([1, 2])
    height = rng.randint(14 if pair_count == 1 else 18, 30)
    width = rng.randint(14 if pair_count == 1 else 18, 30)
    regions = _make_regions(height, width, pair_count, rng)
    if regions is None:
        return None

    colors = list(ISLAND_COLORS)
    rng.shuffle(colors)

    islands: list[dict] = []
    markers: list[tuple[int, int]] = []
    orientations: list[str] = []

    for region in regions:
        orientation = rng.choice(["horizontal", "vertical"])
        smaller_first = rng.choice([True, False])
        color_a = colors.pop()
        color_b = colors.pop()

        placed = None
        for _ in range(PAIR_RETRIES):
            candidate = _place_pair(region, orientation, color_a, color_b, smaller_first, rng)
            if candidate is None:
                continue
            trial_islands = islands + candidate["islands"]
            trial_markers = markers + [candidate["marker"]]
            if _scene_is_valid(height, width, trial_islands, trial_markers):
                placed = candidate
                break
        if placed is None:
            return None

        islands.extend(placed["islands"])
        markers.append(placed["marker"])
        orientations.append(orientation)

    if colors and rng.random() < 0.65:
        extra = _try_place_extra_island(height, width, islands, markers, orientations, colors.pop(), rng)
        if extra is not None:
            trial_islands = islands + [extra]
            if _scene_is_valid(height, width, trial_islands, markers):
                islands = trial_islands

    if not _scene_is_valid(height, width, islands, markers):
        return None
    return _render_scene(height, width, islands, markers)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid bridge puzzle after {GEN_RETRIES} attempts")
