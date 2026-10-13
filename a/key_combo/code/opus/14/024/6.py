from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0

# Fixed visible overlay priority, from strongest to weakest.
PRIORITY_COLORS = (1, 2, 3, 4, 6, 7)
PRIORITY_RANK = {color: rank for rank, color in enumerate(PRIORITY_COLORS)}

# Canonical 5x5 pattern sketches; each produces exactly four pixels once cropped to its bbox.
_PATTERN_CANONICAL = (
    ("corners33", ((1, 1), (1, 3), (3, 1), (3, 3))),
    ("diamond33", ((1, 2), (2, 1), (2, 3), (3, 2))),
    ("outer35", ((1, 0), (1, 4), (3, 0), (3, 4))),
    ("outer53", ((0, 1), (0, 3), (4, 1), (4, 3))),
    ("diamond55", ((0, 2), (2, 0), (2, 4), (4, 2))),
    ("zig53_a", ((0, 2), (1, 1), (3, 3), (4, 2))),
    ("zig53_b", ((0, 2), (1, 3), (3, 1), (4, 2))),
    ("zig35_a", ((1, 1), (2, 0), (2, 4), (3, 3))),
    ("zig35_b", ((1, 3), (2, 0), (2, 4), (3, 1))),
)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _make_pattern(name: str, canonical_cells: Iterable[tuple[int, int]]) -> dict:
    coords = np.array(list(canonical_cells), dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return {
        "name": name,
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "rel": rel,
        "signature": rel,
    }


PATTERNS = tuple(_make_pattern(name, cells) for name, cells in _PATTERN_CANONICAL)


def _extract_groups(grid: np.ndarray) -> list[dict]:
    groups = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND):
        coords = np.argwhere(grid == color)
        assert coords.shape == (4, 2), "each foreground color must appear exactly four times"
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
        groups.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "rel": rel,
            }
        )
    assert 3 <= len(groups) <= 4, "expected 3-4 color groups"
    rel_signatures = [group["rel"] for group in groups]
    assert len(set(rel_signatures)) == len(rel_signatures), "every group must use a distinct relative mask"
    return groups


def _overlay_specs(specs: list[dict]) -> tuple[np.ndarray, dict[tuple[int, int], list[int]]]:
    assert specs, "expected at least one group"
    H = max(int(spec["h"]) for spec in specs)
    W = max(int(spec["w"]) for spec in specs)
    interior = np.full((H, W), BACKGROUND, dtype=int)
    best_rank = np.full((H, W), len(PRIORITY_COLORS) + 1, dtype=int)
    coverage: dict[tuple[int, int], list[int]] = {}

    for spec in specs:
        h = int(spec["h"])
        w = int(spec["w"])
        color = int(spec["color"])
        assert color in PRIORITY_RANK, "unexpected foreground color"
        row_extra = H - h
        col_extra = W - w
        assert row_extra % 2 == 0 and col_extra % 2 == 0, "all boxes must center on the shared canvas"
        ro = row_extra // 2
        co = col_extra // 2
        rank = PRIORITY_RANK[color]
        for rr, cc in spec["rel"]:
            r = int(rr + ro)
            c = int(cc + co)
            coverage.setdefault((r, c), []).append(color)
            if rank < best_rank[r, c]:
                best_rank[r, c] = rank
                interior[r, c] = color

    return interior, coverage


def _transform_from_groups(groups: list[dict], frame_color: int) -> np.ndarray:
    interior, _coverage = _overlay_specs(groups)
    out = np.full((interior.shape[0] + 2, interior.shape[1] + 2), frame_color, dtype=int)
    out[1:-1, 1:-1] = interior
    out[1:-1, 1:-1][interior == frame_color] = BACKGROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Center-overlay the scattered four-pixel masks, frame the result with the input-center color, and erase that color inside.

    Input structure:
    - The grid is odd-sized with black background and contains 3-4 distinct foreground colors.
    - Each foreground color forms exactly one isolated four-pixel mask inside a small non-overlapping bbox.
    - Exactly one foreground pixel sits at the geometric center of the whole input grid.

    Rule:
    - For each color group, crop to its bbox and keep the four pixels at their bbox-relative positions.
    - Build a shared black canvas whose height and width are the largest bbox height and width across all groups, then center-align every cropped mask onto that canvas.
    - If several centered masks land on the same cell, keep the color with higher fixed visual priority: blue, then red, then green, then yellow, then magenta, then orange.
    - Let F be the color of the unique pixel at the input grid center; add a one-cell border of color F around the overlaid canvas, then erase every interior F-colored cell back to black.

    Key insight:
    - The same center-bearing color plays two roles at once: one scattered mask participates in the centered overlay, and that mask's center pixel also controls the border color that must then be removed from the interior to disambiguate it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)
    assert H % 2 == 1 and W % 2 == 1, "expected a unique geometric center"
    assert max(H, W) <= 30, "grid exceeds ARC size limits"
    center = (H // 2, W // 2)
    frame_color = int(g[center])
    assert frame_color in PRIORITY_RANK, "expected a foreground control pixel at the grid center"
    groups = _extract_groups(g)
    return _transform_from_groups(groups, frame_color)


def _expanded_rect(rect: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_groups(
    group_specs: list[dict], frame_color: int, grid_shape: tuple[int, int], rng: random.Random
) -> np.ndarray | None:
    H, W = grid_shape
    center = (H // 2, W // 2)
    grid = np.zeros((H, W), dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []

    frame_group = next(spec for spec in group_specs if int(spec["color"]) == frame_color)
    anchor_r, anchor_c = rng.choice(list(frame_group["rel"]))
    top = center[0] - anchor_r
    left = center[1] - anchor_c
    rect = (top, left, top + frame_group["h"] - 1, left + frame_group["w"] - 1)
    if top < 0 or left < 0 or rect[2] >= H or rect[3] >= W:
        return None
    placed_rects.append(rect)
    for rr, cc in frame_group["rel"]:
        grid[top + rr, left + cc] = frame_color

    others = [spec for spec in group_specs if int(spec["color"]) != frame_color]
    rng.shuffle(others)
    for spec in others:
        positions = [
            (r0, c0)
            for r0 in range(H - spec["h"] + 1)
            for c0 in range(W - spec["w"] + 1)
        ]
        rng.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            rect = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
            if any(_rects_overlap(_expanded_rect(rect), _expanded_rect(prev)) for prev in placed_rects):
                continue
            if center[0] >= rect[0] and center[0] <= rect[2] and center[1] >= rect[1] and center[1] <= rect[3]:
                continue
            cells = [(r0 + rr, c0 + cc) for rr, cc in spec["rel"]]
            if any(grid[r, c] != BACKGROUND for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = int(spec["color"])
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    assert int(np.count_nonzero(grid == frame_color)) == 4
    assert int(grid[center]) == frame_color
    return grid


def _semantic_ok(group_specs: list[dict], frame_color: int) -> bool:
    interior, coverage = _overlay_specs(group_specs)
    if not np.any(interior == frame_color):
        return False
    overlap_cells = [cell for cell, colors in coverage.items() if len(set(colors)) > 1]
    if not overlap_cells:
        return False

    visible_priority_overlap = False
    for cell in overlap_cells:
        colors = list(dict.fromkeys(coverage[cell]))
        winner = int(interior[cell])
        if winner != frame_color and any(color != winner for color in colors):
            visible_priority_overlap = True
            break
    if not visible_priority_overlap:
        return False

    expected = _transform_from_groups(group_specs, frame_color)
    no_erase = np.full_like(expected, frame_color)
    no_erase[1:-1, 1:-1] = interior
    if np.array_equal(expected, no_erase):
        return False

    interior_nonzero = int(np.count_nonzero(expected[1:-1, 1:-1]))
    if interior_nonzero < 3:
        return False

    alt_union = np.zeros_like(interior)
    for spec in group_specs:
        h = int(spec["h"])
        w = int(spec["w"])
        ro = (interior.shape[0] - h) // 2
        co = (interior.shape[1] - w) // 2
        for rr, cc in spec["rel"]:
            alt_union[ro + rr, co + cc] = int(spec["color"])
    if np.array_equal(alt_union, interior):
        return False

    return True


def _choose_specs(rng: random.Random) -> tuple[list[dict], int] | None:
    n_groups = rng.choice((3, 4))
    patterns = list(rng.sample(PATTERNS, n_groups))
    colors = list(rng.sample(PRIORITY_COLORS, n_groups))
    rng.shuffle(patterns)
    rng.shuffle(colors)
    specs = [
        {
            "name": pattern["name"],
            "h": int(pattern["h"]),
            "w": int(pattern["w"]),
            "rel": tuple(pattern["rel"]),
            "color": int(color),
        }
        for pattern, color in zip(patterns, colors)
    ]

    candidates = []
    for color in colors:
        if _semantic_ok(specs, color):
            candidates.append(int(color))
    if not candidates:
        return None

    frame_color = int(rng.choice(candidates))
    if not _semantic_ok(specs, frame_color):
        return None
    return specs, frame_color


def _validate_input(grid: np.ndarray) -> None:
    H, W = grid.shape
    center = (H // 2, W // 2)
    assert H % 2 == 1 and W % 2 == 1
    assert int(grid[center]) in PRIORITY_RANK
    groups = _extract_groups(grid)
    rects = []
    for group in groups:
        rect = group["bbox"]
        rects.append(rect)
        assert group["h"] <= 5 and group["w"] <= 5
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert not _rects_overlap(_expanded_rect(a), _expanded_rect(b)), "bboxes too close"


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    chosen = _choose_specs(rng)
    if chosen is None:
        return None
    specs, frame_color = chosen

    size_options = (15, 17, 19)
    if len(specs) == 4:
        size_options = (17, 19)
    grid_shape = (rng.choice(size_options), rng.choice(size_options))
    grid = _place_groups(specs, frame_color, grid_shape, rng)
    if grid is None:
        return None

    try:
        _validate_input(grid)
        expected = _transform_from_groups(_extract_groups(grid), frame_color)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid center-overlay frame task")
