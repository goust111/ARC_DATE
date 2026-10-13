from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1

GRID_HEIGHT_RANGE = (8, 13)  # inclusive of 8..12 via randrange
GRID_WIDTH_RANGE = (7, 11)   # inclusive of 7..10 via randrange
RECT_SPECS = ((2, 3), (3, 2))
RECT_COUNT_CHOICES = (2, 3)
NOISE_COUNT_RANGE = (4, 9)   # inclusive of 4..8 via randrange
GEN_ATTEMPTS = 500


def _neighbor_counts(mask: np.ndarray) -> np.ndarray:
    """Return 4-neighbor counts for a boolean mask."""
    m = mask.astype(int)
    p = np.pad(m, 1)
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def _stack_and_filter(grid: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    composite = np.vstack((np.flipud(g), g))
    blue_mask = composite == BLUE
    counts = _neighbor_counts(blue_mask)
    out = composite.copy()
    out[blue_mask & (counts < int(min_neighbors))] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack a vertical mirror above the input, then erase weakly supported blue pixels in the stacked grid.

    Input structure:
    - A black background contains 2-3 small blue rectangles, each either 2x3 or 3x2.
    - Additional blue single-pixel noise is scattered around the grid; some noise may sit near the top seam-creating edge.
    - In the original input, rectangle pixels already have strong 4-neighbor support, while every noise pixel has at most one blue 4-neighbor.

    Rule:
    - Vertically flip the whole input grid.
    - Stack that flipped copy above the unchanged input, doubling the height and keeping the width.
    - In the stacked grid, count blue 4-neighbors for every blue pixel.
    - Keep blue pixels with at least two blue 4-neighbors; erase every blue pixel with fewer than two by turning it black.

    Key insight:
    - The mirror-stack is not the goal by itself: it creates new 4-neighbor support across the join, so some top-edge noise survives only after stacking, while unsupported noise elsewhere is still deleted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(np.unique(g)).issubset({BLACK, BLUE}), "expected a binary black/blue grid"
    return _stack_and_filter(g, min_neighbors=2)


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    total = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            total += 1
    return total


def _rect_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, h, w = rect
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _stamp_rectangles(shape: tuple[int, int], rects: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        for r, c in _rect_cells(rect):
            grid[r, c] = BLUE
    return grid


def _rect_ok(rects: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, rh, rw = candidate
    r1 = r0 + rh - 1
    c1 = c0 + rw - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    for or0, oc0, oh, ow in rects:
        or1 = or0 + oh - 1
        oc1 = oc0 + ow - 1
        # Keep a one-cell moat for visual separation and to prevent accidental adjacency.
        if not (r1 < or0 - 2 or or1 < r0 - 2 or c1 < oc0 - 2 or oc1 < c0 - 2):
            return False
    return True


def _place_rectangles(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    n_rects = random.choice(RECT_COUNT_CHOICES)

    seam_specs = list(RECT_SPECS)
    random.shuffle(seam_specs)
    seam_rect = None
    for rh, rw in seam_specs:
        cols = list(range(0, w - rw + 1))
        random.shuffle(cols)
        for c0 in cols:
            cand = (1, c0, rh, rw)
            if _rect_ok(rects, cand, h, w):
                seam_rect = cand
                rects.append(cand)
                break
        if seam_rect is not None:
            break
    if seam_rect is None:
        return None

    attach_rect = None
    attach_specs = list(RECT_SPECS)
    random.shuffle(attach_specs)
    for rh, rw in attach_specs:
        rows = list(range(2, h - rh + 1))
        cols = list(range(0, w - rw + 1))
        random.shuffle(rows)
        random.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                cand = (r0, c0, rh, rw)
                if _rect_ok(rects, cand, h, w):
                    attach_rect = cand
                    rects.append(cand)
                    break
            if attach_rect is not None:
                break
        if attach_rect is not None:
            break
    if attach_rect is None:
        return None

    while len(rects) < n_rects:
        specs = list(RECT_SPECS)
        random.shuffle(specs)
        placed = False
        for rh, rw in specs:
            rows = list(range(0, h - rh + 1))
            cols = list(range(0, w - rw + 1))
            random.shuffle(rows)
            random.shuffle(cols)
            for r0 in rows:
                for c0 in cols:
                    cand = (r0, c0, rh, rw)
                    if _rect_ok(rects, cand, h, w):
                        rects.append(cand)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return rects


def _noise_candidates(grid: np.ndarray, rect_grid: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = grid.shape
    seam: list[tuple[int, int]] = []
    attached: list[tuple[int, int]] = []
    isolated: list[tuple[int, int]] = []
    edge_isolated: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK:
                continue
            deg_total = _degree(grid == BLUE, r, c)
            deg_rect = _degree(rect_grid == BLUE, r, c)
            if deg_total == 1 and deg_rect == 1:
                if r == 0:
                    seam.append((r, c))
                else:
                    attached.append((r, c))
            elif deg_total == 0:
                isolated.append((r, c))
                if r in {h - 1, h - 2}:
                    edge_isolated.append((r, c))
    return {
        "seam": seam,
        "attached": attached,
        "isolated": isolated,
        "edge_isolated": edge_isolated,
    }


def _add_noise(grid: np.ndarray, pos: tuple[int, int], role: str, noise_roles: dict[tuple[int, int], str]) -> None:
    r, c = pos
    grid[r, c] = BLUE
    noise_roles[(r, c)] = role


def _validate_sample(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    noise_roles: dict[tuple[int, int], str],
) -> bool:
    h, w = grid.shape
    rect_grid = _stamp_rectangles((h, w), rects)
    if not (2 <= len(rects) <= 3):
        return False
    if not (4 <= len(noise_roles) <= 8):
        return False
    if not any(role == "seam" for role in noise_roles.values()):
        return False
    if not any(role == "attached" for role in noise_roles.values()):
        return False
    if not any(r in {0, h - 1} for r, _c in noise_roles):
        return False

    rect_mask = rect_grid == BLUE
    full_mask = grid == BLUE

    for rect in rects:
        rh, rw = rect[2], rect[3]
        if (rh, rw) not in RECT_SPECS:
            return False
        for r, c in _rect_cells(rect):
            if _degree(rect_mask, r, c) < 2:
                return False

    for (r, c), role in noise_roles.items():
        if role == "seam":
            if r != 0:
                return False
        else:
            if r == 0 and role != "seam":
                return False
        if _degree(full_mask, r, c) > 1:
            return False

    out = main(grid)
    composite = np.vstack((np.flipud(grid), grid))
    counts = _neighbor_counts(composite == BLUE)

    if np.array_equal(out, composite):
        return False
    if not np.array_equal(out, _stack_and_filter(grid, min_neighbors=2)):
        return False
    if np.array_equal(out, _stack_and_filter(grid, min_neighbors=1)):
        return False
    if np.array_equal(out, _stack_and_filter(grid, min_neighbors=3)):
        return False

    filtered_original = grid.copy()
    orig_counts = _neighbor_counts(grid == BLUE)
    filtered_original[(grid == BLUE) & (orig_counts < 2)] = BLACK
    if np.array_equal(out, np.vstack((np.flipud(filtered_original), filtered_original))):
        return False

    seam_success = False
    attached_deleted = False
    for (r, c), role in noise_roles.items():
        mirror_r = h - 1 - r
        orig_r = h + r
        if role == "seam":
            if (
                counts[mirror_r, c] >= 2
                and counts[orig_r, c] >= 2
                and out[mirror_r, c] == BLUE
                and out[orig_r, c] == BLUE
                and filtered_original[r, c] == BLACK
            ):
                seam_success = True
        elif role == "attached":
            if (
                counts[mirror_r, c] == 1
                and counts[orig_r, c] == 1
                and out[mirror_r, c] == BLACK
                and out[orig_r, c] == BLACK
            ):
                attached_deleted = True
    if not seam_success or not attached_deleted:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randrange(*GRID_HEIGHT_RANGE)
    w = random.randrange(*GRID_WIDTH_RANGE)
    rects = _place_rectangles(h, w)
    if rects is None:
        return None

    rect_grid = _stamp_rectangles((h, w), rects)
    grid = rect_grid.copy()
    noise_roles: dict[tuple[int, int], str] = {}

    target_noise = random.randrange(*NOISE_COUNT_RANGE)

    candidates = _noise_candidates(grid, rect_grid)
    if not candidates["seam"] or not candidates["attached"]:
        return None

    _add_noise(grid, random.choice(candidates["seam"]), "seam", noise_roles)

    candidates = _noise_candidates(grid, rect_grid)
    if not candidates["attached"]:
        return None
    _add_noise(grid, random.choice(candidates["attached"]), "attached", noise_roles)

    if len(noise_roles) < target_noise:
        candidates = _noise_candidates(grid, rect_grid)
        edge_choices = candidates["edge_isolated"]
        if edge_choices and random.random() < 0.8:
            _add_noise(grid, random.choice(edge_choices), "isolated", noise_roles)

    while len(noise_roles) < target_noise:
        candidates = _noise_candidates(grid, rect_grid)
        choices: list[tuple[str, list[tuple[int, int]]]] = []
        if candidates["seam"]:
            choices.extend([("seam", candidates["seam"])] * 3)
        if candidates["attached"]:
            choices.extend([("attached", candidates["attached"])] * 2)
        if candidates["edge_isolated"]:
            choices.extend([("isolated", candidates["edge_isolated"])] * 2)
        if candidates["isolated"]:
            choices.append(("isolated", candidates["isolated"]))
        if not choices:
            break
        role, pool = random.choice(choices)
        _add_noise(grid, random.choice(pool), role, noise_roles)

    if not _validate_sample(grid, rects, noise_roles):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input for puzzle 3")
