from __future__ import annotations

from collections import defaultdict, deque
import itertools
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, label_mask, neighbors4


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


# --------------- solver helpers ---------------


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _crop(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _discover_carriers(grid: np.ndarray) -> list[dict[str, object]]:
    bg = int(get_bg(grid))
    carriers: list[dict[str, object]] = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    ):
        box = tuple(map(int, comp["bbox"]))
        crop = _crop(grid, box)
        assert int(comp["area"]) == _bbox_area(box), "each carrier must be a solid rectangle"
        assert bool(np.all(crop != bg)), "carrier rectangles may not contain background holes"

        carrier_colors = sorted(set(map(int, np.unique(crop))) - {Color.RED, Color.YELLOW})
        assert len(carrier_colors) == 1, "each carrier must have exactly one base color"
        carrier_color = int(carrier_colors[0])
        assert carrier_color in {Color.BLUE, Color.TEAL}, "carrier color must be blue or teal"

        r0, c0, _r1, _c1 = box
        markers = sorted(
            [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(crop == Color.RED)],
            key=lambda rc: (rc[0], rc[1]),
        )
        obstacles = sorted(
            [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(crop == Color.YELLOW)],
            key=lambda rc: (rc[0], rc[1]),
        )
        assert markers, "every carrier must contain at least one red marker"
        marker_rows = [r for r, _c in markers]
        assert len(marker_rows) == len(set(marker_rows)), "carrier markers must lie on distinct rows"

        carriers.append(
            {
                "bbox": box,
                "color": carrier_color,
                "markers": markers,
                "obstacles": obstacles,
            }
        )

    carriers.sort(key=lambda carrier: carrier["bbox"])
    return carriers


def _direction_for_color(color: int) -> int:
    if int(color) == Color.BLUE:
        return 1
    assert int(color) == Color.TEAL
    return -1


def _extend_carrier_rays(out: np.ndarray, carrier: dict[str, object]) -> None:
    r0, c0, r1, c1 = carrier["bbox"]
    step = _direction_for_color(int(carrier["color"]))
    for r, c in carrier["markers"]:
        cc = c + step
        while c0 < cc < c1:
            if int(out[r, cc]) == Color.YELLOW:
                break
            out[r, cc] = Color.RED
            cc += step


def _region_order(labels: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for r in range(labels.shape[0]):
        for c in range(labels.shape[1]):
            label_id = int(labels[r, c])
            if label_id > 0 and label_id not in seen:
                seen.add(label_id)
                order.append(label_id)
    return order


def _region_graph(crop: np.ndarray, carrier_color: int) -> tuple[np.ndarray, dict[int, set[int]], list[int], list[int]]:
    h, w = crop.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    fillable = (crop == int(carrier_color)) & ~boundary
    labels, n_regions = label_mask(fillable, connectivity=4)
    assert int(n_regions) >= 1, "each carrier must retain at least one fillable region"

    graph: dict[int, set[int]] = {region_id: set() for region_id in range(1, int(n_regions) + 1)}
    wall_labels, n_walls = label_mask(crop == Color.RED, connectivity=4)
    separating_walls: list[int] = []

    for wall_id in range(1, int(n_walls) + 1):
        adjacent_regions: set[int] = set()
        for r, c in np.argwhere(wall_labels == wall_id):
            rr = int(r)
            cc = int(c)
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w:
                    region_id = int(labels[nr, nc])
                    if region_id > 0:
                        adjacent_regions.add(region_id)
        if len(adjacent_regions) == 2:
            a, b = tuple(sorted(adjacent_regions))
            graph[a].add(b)
            graph[b].add(a)
            separating_walls.append(wall_id)
    return labels, graph, _region_order(labels), separating_walls


def _region_parity(labels: np.ndarray, graph: dict[int, set[int]], order: list[int]) -> dict[int, int]:
    parity: dict[int, int] = {}
    for root in order:
        if root in parity:
            continue
        parity[root] = 0
        queue = deque([root])
        while queue:
            region_id = queue.popleft()
            for neighbor in graph[region_id]:
                wanted = 1 - parity[region_id]
                if neighbor in parity:
                    assert parity[neighbor] == wanted, "region graph must stay bipartite"
                else:
                    parity[neighbor] = wanted
                    queue.append(neighbor)
    return parity


def _paint_carrier_regions(out: np.ndarray, carrier: dict[str, object]) -> None:
    box = carrier["bbox"]
    carrier_color = int(carrier["color"])
    crop = _crop(out, box)
    labels, graph, order, _separating_walls = _region_graph(crop, carrier_color)
    parity = _region_parity(labels, graph, order)

    solved = crop.copy()
    for region_id, region_parity in parity.items():
        solved[labels == region_id] = Color.GREY if region_parity == 0 else Color.GREEN

    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend color-directed marker rays inside each carrier rectangle and alternately fill the ray-separated subregions.

    Input structure:
    - The background is black.
    - The foreground consists of 2-3 disjoint rectangular carrier regions whose base color is blue or teal.
    - Each carrier contains one or more red marker pixels and optional yellow blocker pixels, all strictly inside the rectangle.

    Rule:
    - Treat each multicolor rectangle as one carrier; blue carriers send every red marker horizontally to the right and teal carriers send every red marker horizontally to the left.
    - Draw each red ray one cell at a time, stopping before the carrier boundary column or before a yellow blocker on the same row.
    - Keep the carrier boundary, all yellow blockers, and all red ray pixels unchanged; then label the remaining 4-connected interior carrier cells.
    - Starting from the upper-leftmost remaining subregion of each carrier, color regions grey and flip to green across every separating red ray wall.

    Key insight:
    - The carrier color is an implicit direction code: those directed horizontal rays become internal walls, and the final grey/green pattern is determined by the region graph induced by those walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    carriers = _discover_carriers(g)
    out = g.copy()
    for carrier in carriers:
        _extend_carrier_rays(out, carrier)
    for carrier in carriers:
        _paint_carrier_regions(out, carrier)
    return out


# --------------- generator helpers ---------------


GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (15, 22)
CARRIER_COUNT_CHOICES = (2, 3)
CARRIER_H_RANGE = (4, 7)
CARRIER_W_RANGE = (5, 8)
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400


def _overlaps_with_gap(a: BBox, b: BBox, gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_rectangles(height: int, width: int, count: int) -> list[BBox] | None:
    dims = [
        (random.randint(*CARRIER_H_RANGE), random.randint(*CARRIER_W_RANGE))
        for _ in range(int(count))
    ]
    order = sorted(range(count), key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)
    placed: list[BBox | None] = [None] * count

    for idx in order:
        h, w = dims[idx]
        if height - h - 1 < 1 or width - w - 1 < 1:
            return None

        choices = [
            (r0, c0)
            for r0 in range(1, height - h)
            for c0 in range(1, width - w)
            if all(
                not _overlaps_with_gap((r0, c0, r0 + h - 1, c0 + w - 1), other, gap=1)
                for other in placed
                if other is not None
            )
        ]
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        placed[idx] = (r0, c0, r0 + h - 1, c0 + w - 1)

    return [box for box in placed if box is not None]


def _carrier_colors(count: int) -> list[int]:
    colors = [Color.BLUE, Color.TEAL]
    if count == 3:
        colors.append(random.choice([Color.BLUE, Color.TEAL]))
    random.shuffle(colors)
    return colors


def _available_marker_rows(box: BBox) -> list[int]:
    r0, _c0, r1, _c1 = box
    return list(range(r0 + 1, r1))


def _spaced_marker_rows(rows: list[int], count: int) -> list[int] | None:
    combos = [
        list(combo)
        for combo in itertools.combinations(rows, int(count))
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1))
    ]
    if not combos:
        return None
    return random.choice(combos)


def _decorate_structural_row(
    grid: np.ndarray,
    box: BBox,
    color: int,
    row: int,
    obstacle_budget: int,
) -> tuple[int, bool]:
    _r0, c0, _r1, c1 = box
    if int(color) == Color.BLUE:
        marker_col = c0 + 1
        blocker_col = c1 - 1
    else:
        marker_col = c1 - 1
        blocker_col = c0 + 1

    truncated = obstacle_budget > 0 and random.random() < 0.6
    grid[row, marker_col] = Color.RED
    used = 0
    if truncated:
        grid[row, blocker_col] = Color.YELLOW
        used = 1
    return used, truncated


def _decorate_decoy_row(
    grid: np.ndarray,
    box: BBox,
    color: int,
    row: int,
    obstacle_budget: int,
) -> tuple[int, bool]:
    _r0, c0, _r1, c1 = box
    width = c1 - c0 + 1
    used = 0
    truncated = False

    if int(color) == Color.BLUE:
        marker_choices = [c0 + dc for dc in range(2, width - 1)]
        marker_col = random.choice(marker_choices)
        grid[row, marker_col] = Color.RED
        blocker_choices = [cc for cc in range(marker_col + 2, c1)]
        if obstacle_budget > 0 and blocker_choices and random.random() < 0.65:
            blocker_col = random.choice(blocker_choices)
            grid[row, blocker_col] = Color.YELLOW
            used = 1
            truncated = True
    else:
        marker_choices = [c0 + dc for dc in range(1, width - 2)]
        marker_col = random.choice(marker_choices)
        grid[row, marker_col] = Color.RED
        blocker_choices = [cc for cc in range(c0 + 1, marker_col - 1)]
        if obstacle_budget > 0 and blocker_choices and random.random() < 0.65:
            blocker_col = random.choice(blocker_choices)
            grid[row, blocker_col] = Color.YELLOW
            used = 1
            truncated = True

    return used, truncated


def _decorate_carrier(grid: np.ndarray, box: BBox, color: int) -> dict[str, bool | int]:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    rows = _available_marker_rows(box)
    feasible_counts = [count for count in range(1, min(3, len(rows)) + 1) if _spaced_marker_rows(rows, count)]
    marker_count = random.choice(feasible_counts)
    marker_rows = _spaced_marker_rows(rows, marker_count)
    assert marker_rows is not None
    marker_rows = list(marker_rows)
    random.shuffle(marker_rows)

    if marker_count == 1:
        structural_count = 1
    else:
        structural_count = random.randint(1, marker_count)

    obstacle_budget = 2
    had_decoy = structural_count < marker_count
    had_truncation = False

    for row in marker_rows[:structural_count]:
        used, truncated = _decorate_structural_row(grid, box, color, row, obstacle_budget)
        obstacle_budget -= used
        had_truncation |= truncated

    for row in marker_rows[structural_count:]:
        used, truncated = _decorate_decoy_row(grid, box, color, row, obstacle_budget)
        obstacle_budget -= used
        had_truncation |= truncated

    return {
        "had_decoy": had_decoy,
        "had_truncation": had_truncation,
        "marker_count": marker_count,
    }


def _ray_lengths(grid: np.ndarray, carrier: dict[str, object]) -> list[int]:
    lengths: list[int] = []
    _r0, c0, _r1, c1 = carrier["bbox"]
    step = _direction_for_color(int(carrier["color"]))
    for r, c in carrier["markers"]:
        length = 1
        cc = c + step
        while c0 < cc < c1:
            if int(grid[r, cc]) == Color.YELLOW:
                break
            length += 1
            cc += step
        lengths.append(length)
    return lengths


def _semantic_report(grid: np.ndarray) -> dict[str, object]:
    carriers = _discover_carriers(grid)
    after_rays = np.array(grid, dtype=int, copy=True)
    for carrier in carriers:
        _extend_carrier_rays(after_rays, carrier)

    report: dict[str, object] = {
        "colors": {int(carrier["color"]) for carrier in carriers},
        "has_truncation": False,
        "has_short_ray": False,
        "has_multi_region": False,
        "carrier_regions": [],
    }

    for carrier in carriers:
        crop = _crop(after_rays, carrier["bbox"])
        labels, _graph, _order, separating_walls = _region_graph(crop, int(carrier["color"]))
        n_regions = int(labels.max())
        report["has_multi_region"] |= n_regions > 1
        report["carrier_regions"].append(n_regions)

        lengths = _ray_lengths(grid, carrier)
        max_full_length = carrier["bbox"][3] - carrier["bbox"][1] - 1
        report["has_short_ray"] |= any(length < max_full_length for length in lengths)

        r0, c0, r1, c1 = carrier["bbox"]
        step = _direction_for_color(int(carrier["color"]))
        for r, c in carrier["markers"]:
            blocker_col = None
            cc = c + step
            while c0 < cc < c1:
                if int(grid[r, cc]) == Color.YELLOW:
                    blocker_col = cc
                    break
                cc += step
            if blocker_col is not None:
                report["has_truncation"] = True

        assert n_regions == len(separating_walls) + 1, "structural walls should form a simple stacked partition"

    return report


def _build_candidate() -> np.ndarray | None:
    count = random.choice(CARRIER_COUNT_CHOICES)
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    boxes = None
    for _ in range(PLACEMENT_RETRIES):
        boxes = _place_rectangles(height, width, count)
        if boxes is not None:
            break
    if boxes is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    color_seq = _carrier_colors(count)
    meta = []
    for box, color in zip(boxes, color_seq):
        meta.append(_decorate_carrier(grid, box, color))

    report = _semantic_report(grid)
    if report["colors"] != {Color.BLUE, Color.TEAL}:
        return None
    if not bool(report["has_multi_region"]):
        return None
    if not any(bool(info["had_decoy"]) for info in meta):
        return None
    if not bool(report["has_short_ray"]):
        return None
    if not bool(report["has_truncation"]):
        return None

    solved = main(grid)
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid carrier-ray partition puzzle")
