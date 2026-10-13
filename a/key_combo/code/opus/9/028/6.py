from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ORTH_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _expand_markers(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    markers = np.argwhere(out == BLUE)
    for r, c in markers:
        for rr in range(max(0, int(r) - 1), min(out.shape[0], int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(out.shape[1], int(c) + 2)):
                if int(out[rr, cc]) == BLACK:
                    out[rr, cc] = BLUE
    return out


def _area_color(area: int) -> int:
    if area <= 4:
        return GREEN
    if area <= 7:
        return YELLOW
    return RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each blue marker through its 3x3 neighborhood, then recolor each merged blue component by area.

    Input structure:
    - The grid uses black as background, blue single-cell markers as the expansion sources, and gray single-cell blockers.
    - Blue markers lie away from the border so each has a centered 3x3 neighborhood.
    - Some markers are close enough that their expanded neighborhoods can merge, while some gray blockers sit inside those neighborhoods and prevent part of the fill.

    Rule:
    - For every blue marker, fill every black cell in its centered 3x3 neighborhood with blue; existing blue or gray cells stay unchanged.
    - After all markers expand, find every 4-connected blue component.
    - Recolor each blue component by its area: 1-4 to green, 5-7 to yellow, and 8 or more to red.
    - Keep gray blockers and black background unchanged.

    Key insight:
    - The final color is determined only after local 3x3 expansions interact: blockers can both carve holes inside one marker's patch and stop nearby markers from merging into a larger area tier.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    expanded = _expand_markers(g)
    out = expanded.copy()
    for comp in _components_of_color(expanded, BLUE):
        color = _area_color(len(comp))
        for r, c in comp:
            out[r, c] = color
    return out


def _make_split_cluster() -> dict:
    # Two adjacent orthogonal blockers isolate exactly one corner cell.
    variants = [
        {"blockers": [(0, 1), (1, 0)], "isolated": (0, 0)},
        {"blockers": [(0, 1), (1, 2)], "isolated": (0, 2)},
        {"blockers": [(1, 0), (2, 1)], "isolated": (2, 0)},
        {"blockers": [(1, 2), (2, 1)], "isolated": (2, 2)},
    ]
    choice = random.choice(variants)
    return {
        "kind": "split",
        "h": 3,
        "w": 3,
        "markers": [(1, 1)],
        "blockers": list(choice["blockers"]),
        "isolated": choice["isolated"],
    }


def _make_merged_cluster() -> dict:
    orientation = random.choice(["h", "v"])
    distance = random.choice([2, 3])
    if orientation == "h":
        return {
            "kind": "merged",
            "h": 3,
            "w": distance + 3,
            "markers": [(1, 1), (1, 1 + distance)],
            "blockers": [],
        }
    return {
        "kind": "merged",
        "h": distance + 3,
        "w": 3,
        "markers": [(1, 1), (1 + distance, 1)],
        "blockers": [],
    }


def _make_barrier_cluster() -> dict:
    orientation = random.choice(["h", "v"])
    blocked_first = random.choice([True, False])
    if orientation == "h":
        markers = [(1, 1), (1, 4)]
        barrier = [(0, 2), (1, 2), (2, 2)] if blocked_first else [(0, 3), (1, 3), (2, 3)]
        return {
            "kind": "barrier",
            "h": 3,
            "w": 6,
            "markers": markers,
            "blockers": barrier,
            "blocked_marker_index": 0 if blocked_first else 1,
        }
    markers = [(1, 1), (4, 1)]
    barrier = [(2, 0), (2, 1), (2, 2)] if blocked_first else [(3, 0), (3, 1), (3, 2)]
    return {
        "kind": "barrier",
        "h": 6,
        "w": 3,
        "markers": markers,
        "blockers": barrier,
        "blocked_marker_index": 0 if blocked_first else 1,
    }


def _make_isolated_cluster() -> dict:
    return {
        "kind": "extra",
        "h": 3,
        "w": 3,
        "markers": [(1, 1)],
        "blockers": [],
    }


def _rect_clear(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int = 1) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        separated = r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0
        if not separated:
            return False
    return True


def _place_clusters(h: int, w: int, clusters: list[dict]) -> list[dict] | None:
    placed_rects: list[tuple[int, int, int, int]] = []
    placed_clusters: list[dict] = []
    order = sorted(clusters, key=lambda item: item["h"] * item["w"], reverse=True)

    for cluster in order:
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - cluster["h"]):
            for c0 in range(1, w - cluster["w"]):
                rect = (r0, c0, r0 + cluster["h"] - 1, c0 + cluster["w"] - 1)
                if _rect_clear(rect, placed_rects, gap=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        rect = (r0, c0, r0 + cluster["h"] - 1, c0 + cluster["w"] - 1)
        placed_rects.append(rect)

        placed = dict(cluster)
        placed["top_left"] = (r0, c0)
        placed["bbox"] = rect
        placed["abs_markers"] = [(r0 + rr, c0 + cc) for rr, cc in cluster["markers"]]
        placed["abs_blockers"] = [(r0 + rr, c0 + cc) for rr, cc in cluster["blockers"]]
        if cluster["kind"] == "split":
            iso_r, iso_c = cluster["isolated"]
            placed["isolated_abs"] = (r0 + iso_r, c0 + iso_c)
        placed_clusters.append(placed)
    return placed_clusters


def _outside_neighborhood(cell: tuple[int, int], markers: list[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - mr), abs(c - mc)) > 1 for mr, mc in markers)


def _add_far_blocker(grid: np.ndarray, markers: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in occupied and _outside_neighborhood((r, c), markers)
    ]
    if not candidates:
        return None
    cell = random.choice(candidates)
    grid[cell] = GREY
    return cell


def _component_map(grid: np.ndarray, color: int) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps = _components_of_color(grid, color)
    by_cell: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for cell in comp:
            by_cell[cell] = idx
    return comps, by_cell


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    markers = list(map(tuple, np.argwhere(grid == BLUE)))
    blockers = list(map(tuple, np.argwhere(grid == GREY)))
    if not 5 <= len(markers) <= 8:
        return False
    if not 4 <= len(blockers) <= 6:
        return False
    if any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in markers):
        return False

    expanded = _expand_markers(grid)
    comps, by_cell = _component_map(expanded, BLUE)
    areas = sorted(len(comp) for comp in comps)
    if not any(area <= 4 for area in areas):
        return False
    if not any(5 <= area <= 7 for area in areas):
        return False
    if not any(area >= 8 for area in areas):
        return False

    split_marker = meta["split_marker"]
    split_comp_area = len(comps[by_cell[split_marker]])
    if split_comp_area != 6:
        return False
    isolated_corner = meta["split_isolated_corner"]
    if isolated_corner not in by_cell or len(comps[by_cell[isolated_corner]]) != 1:
        return False

    merged_markers = meta["merged_markers"]
    merged_ids = {by_cell[cell] for cell in merged_markers}
    if len(merged_ids) != 1 or len(comps[next(iter(merged_ids))]) < 12:
        return False

    barrier_markers = meta["barrier_markers"]
    barrier_ids = [by_cell[cell] for cell in barrier_markers]
    if barrier_ids[0] == barrier_ids[1]:
        return False
    blocked_area = len(comps[barrier_ids[meta["barrier_blocked_index"]]])
    open_area = len(comps[barrier_ids[1 - meta["barrier_blocked_index"]]])
    if blocked_area != 6 or open_area != 9:
        return False

    extra_blocker = meta["extra_blocker"]
    if extra_blocker is None or not _outside_neighborhood(extra_blocker, markers):
        return False

    out = main(grid)
    expected = expanded.copy()
    for comp in comps:
        color = _area_color(len(comp))
        for r, c in comp:
            expected[r, c] = color
    if not np.array_equal(out, expected):
        return False

    if {GREEN, YELLOW, RED} - set(map(int, out.ravel())):
        return False

    return True


def _attempt_generate() -> tuple[np.ndarray, dict] | None:
    extra_count = random.randint(0, 3)
    clusters = [_make_split_cluster(), _make_merged_cluster(), _make_barrier_cluster()]
    clusters.extend(_make_isolated_cluster() for _ in range(extra_count))

    total_bbox_area = sum(cluster["h"] * cluster["w"] for cluster in clusters)
    for _ in range(40):
        h = random.randint(12, 16 + (1 if extra_count >= 2 else 0))
        w = random.randint(12, 16 + (1 if total_bbox_area >= 30 else 0))
        placed = _place_clusters(h, w, clusters)
        if placed is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        occupied: set[tuple[int, int]] = set()
        meta: dict = {"extra_blocker": None}
        all_markers: list[tuple[int, int]] = []

        for cluster in placed:
            for cell in cluster["abs_markers"]:
                if cell in occupied:
                    return None
                grid[cell] = BLUE
                occupied.add(cell)
                all_markers.append(cell)
            for cell in cluster["abs_blockers"]:
                if cell in occupied:
                    return None
                grid[cell] = GREY
                occupied.add(cell)

            if cluster["kind"] == "split":
                meta["split_marker"] = cluster["abs_markers"][0]
                meta["split_isolated_corner"] = cluster["isolated_abs"]
            elif cluster["kind"] == "merged":
                meta["merged_markers"] = list(cluster["abs_markers"])
            elif cluster["kind"] == "barrier":
                meta["barrier_markers"] = list(cluster["abs_markers"])
                meta["barrier_blocked_index"] = int(cluster["blocked_marker_index"])

        meta["extra_blocker"] = _add_far_blocker(grid, all_markers, occupied)
        if meta["extra_blocker"] is None:
            continue

        if _validate_instance(grid, meta):
            return grid, meta
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        generated = _attempt_generate()
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid ARC input for puzzle 6")
