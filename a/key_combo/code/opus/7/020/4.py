from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds, label_mask, neighbors4


GROW_SHAPES = [
    ((0, 1), (1, 0), (1, 2), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 2), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 2), (2, 1)),
    ((0, 1), (1, 0), (1, 2), (2, 1), (2, 2)),
]
BIG_LOW_SHAPES = [
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (2, 1)),
]
SMALL_LOW_SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 2), (2, 1)),
]
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]
CORNERS = ("TL", "TR", "BL", "BR")
GEN_ATTEMPTS = 1000


def _objects(grid: np.ndarray) -> list[dict]:
    objects = find_components(grid, bg=Color.BLACK, connectivity=8, monochromatic=True)
    return sorted(objects, key=lambda obj: int(obj["color"]))


def _region_count_touching_object(grid: np.ndarray, coords: np.ndarray, bg_lab: np.ndarray) -> int:
    regions: set[int] = set()
    for r, c in coords:
        r = int(r)
        c = int(c)
        for rr, cc in neighbors4(r, c):
            if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == Color.BLACK:
                regions.add(int(bg_lab[rr, cc]))
    return len(regions)


def _growth_candidate_score(work: np.ndarray, r: int, c: int) -> int:
    bg_mask = work == Color.BLACK
    bg_mask[r, c] = False
    bg_lab, _ = label_mask(bg_mask, connectivity=4)
    neighbor_regions: set[int] = set()
    for rr, cc in neighbors4(r, c):
        if in_bounds(work.shape, rr, cc) and bool(bg_mask[rr, cc]):
            neighbor_regions.add(int(bg_lab[rr, cc]))
    return len(neighbor_regions)


def _simulate_growth(grid: np.ndarray, color: int, steps: int) -> tuple[np.ndarray, bool]:
    work = grid.copy()
    varied = False
    for _ in range(int(steps)):
        candidates: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c in np.argwhere(work == color):
            r = int(r)
            c = int(c)
            for rr, cc in neighbors4(r, c):
                if in_bounds(work.shape, rr, cc) and int(work[rr, cc]) == Color.BLACK and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    candidates.append((rr, cc))
        if not candidates:
            break

        scored = [(_growth_candidate_score(work, r, c), r, c) for r, c in candidates]
        if len({score for score, _, _ in scored}) > 1:
            varied = True
        score, rr, cc = max(scored, key=lambda item: (item[0], -item[1], -item[2]))
        _ = score
        work[rr, cc] = color
    return work, varied


def _simulate_shrink(grid: np.ndarray, color: int, steps: int) -> tuple[np.ndarray, bool]:
    work = grid.copy()
    varied = False
    for _ in range(int(steps)):
        scored: list[tuple[int, int, int]] = []
        for r, c in np.argwhere(work == color):
            r = int(r)
            c = int(c)
            bg_neighbors = 0
            for rr, cc in neighbors4(r, c):
                if in_bounds(work.shape, rr, cc) and int(work[rr, cc]) == Color.BLACK:
                    bg_neighbors += 1
            if bg_neighbors:
                scored.append((bg_neighbors, r, c))
        if not scored:
            break
        if len({score for score, _, _ in scored}) > 1:
            varied = True
        _, rr, cc = max(scored, key=lambda item: (item[0], item[1], item[2]))
        work[rr, cc] = Color.BLACK
    return work, varied


def _solve(grid: np.ndarray, *, exempt_largest: bool) -> tuple[np.ndarray, dict[int, dict], bool, bool, int]:
    g = np.asarray(grid, dtype=int)
    objects = _objects(g)
    assert objects, "expected foreground objects"

    areas = [int(obj["area"]) for obj in objects]
    largest_area = max(areas)
    assert areas.count(largest_area) == 1, "largest object must be unique"
    largest_idx = areas.index(largest_area)

    bg_lab, n_bg = label_mask(g == Color.BLACK, connectivity=4)
    metadata: dict[int, dict] = {}
    finals: list[tuple[int, set[tuple[int, int]]]] = []
    grow_score_varied = False
    shrink_score_varied = False

    for idx, obj in enumerate(objects):
        color = int(obj["color"])
        coords = obj["coords"]
        B = _region_count_touching_object(g, coords, bg_lab)
        metadata[color] = {"area": int(obj["area"]), "B": B, "largest": idx == largest_idx}

        if B >= 3:
            work, varied = _simulate_growth(g, color, B)
            grow_score_varied |= varied
        elif idx != largest_idx or not exempt_largest:
            work, varied = _simulate_shrink(g, color, B)
            shrink_score_varied |= varied
        else:
            work = g.copy()
        finals.append((color, set(map(tuple, np.argwhere(work == color)))))

    out = np.zeros_like(g)
    for color, coords in finals:
        for r, c in coords:
            if int(out[r, c]) not in (Color.BLACK, color):
                raise AssertionError("simultaneous edits overlap")
            out[r, c] = color
    return out, metadata, grow_score_varied, shrink_score_varied, int(n_bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow or shrink each colored object according to how many background regions touch it.

    Input structure:
    - Black is the background.
    - There are several single-color 8-connected foreground objects, each color used by only one object.
    - Treating all foreground pixels as barriers splits the black cells into multiple 4-connected background regions.

    Rule:
    - Flood-fill the black background with 4-connectivity and count, for each object, how many distinct background
      regions touch it through 4-neighbor contacts.
    - If an object touches at least three regions, grow it by exactly that many steps; each step paints the boundary
      background pixel whose temporary removal would separate the most local background regions, breaking ties by
      choosing the topmost then leftmost pixel.
    - If an object touches at most two regions and it is not the unique largest-area object, shrink it by exactly
      that many steps; each step removes the boundary object pixel with the most 4-neighbor background contacts,
      breaking ties by choosing the bottommost then rightmost pixel.
    - The unique largest-area object is exempt from shrinking when its region count is at most two.
    - Every object computes its own grow or shrink result from the original input, then all edits are written at once.

    Key insight:
    - The decisive feature is topological adjacency to distinct background regions, not the object's color, position,
      or raw shape; the largest object matters only as a special exemption inside the low-adjacency branch.
    """
    out, _, _, _, _ = _solve(np.asarray(grid, dtype=int), exempt_largest=True)
    return out


def _place_corner(shape: tuple[tuple[int, int], ...], corner: str, h: int, w: int) -> set[tuple[int, int]]:
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    r0 = 0 if "T" in corner else h - sh
    c0 = 0 if "L" in corner else w - sw
    return {(r0 + r, c0 + c) for r, c in shape}


def _objects_are_separated(objects: list[set[tuple[int, int]]]) -> bool:
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            for r, c in obj_a:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if (rr, cc) in obj_b:
                            return False
    return True


def _build_candidate() -> np.ndarray:
    h = random.randint(12, 15)
    w = random.randint(12, 15)
    grow_corners = set(random.sample(CORNERS, 2))
    low_corners = [corner for corner in CORNERS if corner not in grow_corners]
    grow_list = list(grow_corners)

    objects = [
        _place_corner(random.choice(BIG_LOW_SHAPES), low_corners[0], h, w),
        _place_corner(random.choice(SMALL_LOW_SHAPES), low_corners[1], h, w),
        _place_corner(random.choice(GROW_SHAPES), grow_list[0], h, w),
        _place_corner(random.choice(GROW_SHAPES), grow_list[1], h, w),
    ]
    if not _objects_are_separated(objects):
        raise ValueError("objects too close")

    colors = random.sample(PALETTE, 4)
    grid = np.zeros((h, w), dtype=int)
    for color, obj in zip(colors, objects):
        for r, c in obj:
            grid[r, c] = color
    return grid


def _valid_generated_example(grid: np.ndarray) -> bool:
    try:
        out, metadata, grow_varied, shrink_varied, n_bg = _solve(grid, exempt_largest=True)
        alt_out, _, _, _, _ = _solve(grid, exempt_largest=False)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, alt_out):
        return False
    if not grow_varied or not shrink_varied:
        return False
    if not (3 <= n_bg <= 5):
        return False

    b_values = [int(info["B"]) for info in metadata.values()]
    if sum(B >= 3 for B in b_values) != 2:
        return False
    if sum(B <= 2 for B in b_values) != 2:
        return False

    largest = [color for color, info in metadata.items() if bool(info["largest"])]
    if len(largest) != 1:
        return False
    if int(metadata[largest[0]]["B"]) > 2:
        return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if _valid_generated_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid region-adjacency growth/shrink input")
