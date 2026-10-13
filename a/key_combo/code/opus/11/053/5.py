from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GRAY = 5
ORANGE = 7
MARKER_COLORS = (1, 2, 3, 4, 6)
TEMPLATE_PATTERNS = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=bool),
)
CORNER_OFFSETS = {
    "tl": (0, 0),
    "tr": (0, -3),
    "bl": (-3, 0),
    "br": (-3, -3),
}


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_depth(radius2: int, ring_radii2: list[int]) -> int:
    enclosing = sum(radius2 < ring_radius2 for ring_radius2 in ring_radii2)
    if enclosing <= 0:
        raise ValueError("marker must lie strictly inside the outermost gray frame")
    return enclosing - 1


def _gray_structure(grid: np.ndarray) -> tuple[tuple[int, int], list[int], tuple[int, int, int, int]]:
    gray_coords = np.argwhere(grid == GRAY)
    if gray_coords.size == 0:
        raise ValueError("expected at least one gray concentric frame")
    r0, c0 = gray_coords.min(axis=0)
    r1, c1 = gray_coords.max(axis=0)
    if int(r1 - r0) != int(c1 - c0):
        raise ValueError("gray structure must have a square outer bbox")
    center2 = (int(r0 + r1), int(c0 + c1))
    ring_radii2 = sorted(
        {
            max(abs(2 * int(r) - center2[0]), abs(2 * int(c) - center2[1]))
            for r, c in gray_coords
        },
        reverse=True,
    )
    if len(ring_radii2) < 2:
        raise ValueError("expected multiple concentric layers")
    return center2, ring_radii2, (int(r0), int(c0), int(r1), int(c1))


def _extract_template(grid: np.ndarray) -> np.ndarray:
    orange_coords = np.argwhere(grid == ORANGE)
    if orange_coords.size == 0:
        raise ValueError("expected an orange template")
    h, w = grid.shape
    corner_boxes = (
        (0, 0),
        (0, w - 3),
        (h - 3, 0),
        (h - 3, w - 3),
    )
    for r0, c0 in corner_boxes:
        block = grid[r0 : r0 + 3, c0 : c0 + 3]
        if np.any(block == ORANGE):
            mask = np.zeros_like(grid, dtype=bool)
            mask[r0 : r0 + 3, c0 : c0 + 3] = block == ORANGE
            if np.array_equal(np.argwhere(mask), orange_coords):
                return (block == ORANGE).astype(bool)
    raise ValueError("expected all orange template pixels to live in one corner 3x3 block")


def _footprint_cells(
    center: tuple[int, int], template: np.ndarray, shape: tuple[int, int]
) -> tuple[frozenset[tuple[int, int]], int]:
    h, w = shape
    r, c = center
    cells: set[tuple[int, int]] = set()
    clipped = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if not template[dr + 1, dc + 1]:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
            else:
                clipped += 1
    return frozenset(cells), clipped


def _paint_from_markers(
    shape: tuple[int, int],
    template: np.ndarray,
    markers: Iterable[dict],
    *,
    mode: str,
) -> np.ndarray:
    marker_list = [dict(marker) for marker in markers]
    if mode == "actual":
        marker_list.sort(key=lambda m: (m["depth"], m["r"], m["c"]))
    elif mode == "scan_only":
        marker_list.sort(key=lambda m: (m["r"], m["c"]))
    elif mode == "reverse_depth":
        marker_list.sort(key=lambda m: (-m["depth"], m["r"], m["c"]))
    elif mode == "reverse_same_depth":
        marker_list.sort(key=lambda m: (m["depth"], -m["r"], -m["c"]))
    else:
        raise ValueError(f"unknown paint mode: {mode}")

    out = np.zeros(shape, dtype=int)
    for marker in marker_list:
        color = int(marker["color"])
        r = int(marker["r"])
        c = int(marker["c"])
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if not template[dr + 1, dc + 1]:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored copies of a corner template at marker pixels, breaking overlaps by concentric depth.

    Input structure:
    - Black background with a central gray concentric-square frame structure made of 3-4 one-pixel-thick layers.
    - Several single-pixel colored markers lie strictly inside the outer gray square but never on gray pixels.
    - One corner contains a 3x3 orange template pattern with black holes.

    Rule:
    - Infer the concentric structure center and order the gray square layers from outside to inside.
    - For each colored marker, place the 3x3 template centered on that marker and recolor every orange template pixel to the marker color.
    - If stamps overlap, markers in deeper inner bands override shallower ones; within the same depth, later row-major markers override earlier ones.
    - Remove the original gray frames, the raw marker pixels, and the corner template, keeping only the stamped result on black.

    Key insight:
    - The gray concentric frames are not copied to the output; they exist only to assign each marker a depth level that decides which stamp wins in overlap conflicts.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    center2, ring_radii2, _outer_bbox = _gray_structure(g)
    template = _extract_template(g)

    marker_coords = np.argwhere((g != BG) & (g != GRAY) & (g != ORANGE))
    markers: list[dict] = []
    for r, c in marker_coords:
        radius2 = max(abs(2 * int(r) - center2[0]), abs(2 * int(c) - center2[1]))
        markers.append(
            {
                "r": int(r),
                "c": int(c),
                "color": int(g[r, c]),
                "depth": _ring_depth(radius2, ring_radii2),
            }
        )

    return _paint_from_markers(g.shape, template, markers, mode="actual")


def _draw_square_frame(grid: np.ndarray, top: int, left: int, side: int) -> None:
    bottom = top + side - 1
    right = left + side - 1
    grid[top, left : right + 1] = GRAY
    grid[bottom, left : right + 1] = GRAY
    grid[top : bottom + 1, left] = GRAY
    grid[top : bottom + 1, right] = GRAY


def _place_template(grid: np.ndarray, template: np.ndarray, corner: str) -> None:
    h, w = grid.shape
    dr, dc = CORNER_OFFSETS[corner]
    r0 = 0 if dr >= 0 else h + dr
    c0 = 0 if dc >= 0 else w + dc
    grid[r0 : r0 + 3, c0 : c0 + 3][template] = ORANGE


def _candidate_markers(
    grid: np.ndarray,
    center2: tuple[int, int],
    ring_radii2: list[int],
    top: int,
    left: int,
    side: int,
    template: np.ndarray,
) -> list[dict]:
    infos: list[dict] = []
    for r in range(top, top + side):
        for c in range(left, left + side):
            if int(grid[r, c]) != BG:
                continue
            radius2 = max(abs(2 * r - center2[0]), abs(2 * c - center2[1]))
            depth = _ring_depth(radius2, ring_radii2)
            cells, clipped = _footprint_cells((r, c), template, grid.shape)
            infos.append(
                {
                    "r": r,
                    "c": c,
                    "depth": depth,
                    "cells": cells,
                    "clipped": clipped > 0,
                }
            )
    return infos


def _pick_layout() -> tuple[int, int, int, int, int, str]:
    layer_count = random.choice((3, 4))
    inner_radius = random.randint(1, 3 if layer_count == 3 else 2)
    outer_radius = inner_radius + 2 * (layer_count - 1)
    side = 2 * outer_radius + 1
    template_corner = random.choice(("tl", "tr", "bl", "br"))
    tight_margin_branch = random.random() < 0.45

    close_lo, close_hi = (0, 2) if tight_margin_branch else (1, 3)
    far_lo, far_hi = 3, 5

    if template_corner == "tl":
        top = random.randint(far_lo, far_hi)
        left = random.randint(far_lo, far_hi)
        bottom = random.randint(close_lo, close_hi)
        right = random.randint(close_lo, close_hi)
    elif template_corner == "tr":
        top = random.randint(far_lo, far_hi)
        right = random.randint(far_lo, far_hi)
        bottom = random.randint(close_lo, close_hi)
        left = random.randint(close_lo, close_hi)
    elif template_corner == "bl":
        bottom = random.randint(far_lo, far_hi)
        left = random.randint(far_lo, far_hi)
        top = random.randint(close_lo, close_hi)
        right = random.randint(close_lo, close_hi)
    else:
        bottom = random.randint(far_lo, far_hi)
        right = random.randint(far_lo, far_hi)
        top = random.randint(close_lo, close_hi)
        left = random.randint(close_lo, close_hi)

    h = side + top + bottom
    w = side + left + right
    if max(h, w) > 30:
        raise ValueError("layout exceeded ARC size bound")
    return h, w, top, left, side, template_corner


def _choose_markers(candidates: list[dict], template: np.ndarray, shape: tuple[int, int]) -> list[dict] | None:
    same_pairs: list[tuple[int, int]] = []
    diff_pairs: list[tuple[int, int]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if not (candidates[i]["cells"] & candidates[j]["cells"]):
                continue
            if candidates[i]["depth"] == candidates[j]["depth"]:
                same_pairs.append((i, j))
            else:
                diff_pairs.append((i, j))

    if not same_pairs or not diff_pairs:
        return None

    for _ in range(2500):
        k = random.randint(3, 5)
        same_pair = random.choice(same_pairs)
        diff_pair = random.choice(diff_pairs)
        chosen = set(same_pair) | set(diff_pair)
        if len(chosen) > k:
            continue

        remaining = [i for i in range(len(candidates)) if i not in chosen]
        random.shuffle(remaining)

        while len(chosen) < k and remaining:
            chosen.add(remaining.pop())
        if len(chosen) != k:
            continue

        chosen_infos = [dict(candidates[i]) for i in sorted(chosen, key=lambda idx: (candidates[idx]["r"], candidates[idx]["c"]))]
        if len({info["depth"] for info in chosen_infos}) < 2:
            continue

        overlap_exists = False
        same_overlap = False
        diff_overlap = False
        for i in range(len(chosen_infos)):
            for j in range(i + 1, len(chosen_infos)):
                if not (chosen_infos[i]["cells"] & chosen_infos[j]["cells"]):
                    continue
                overlap_exists = True
                if chosen_infos[i]["depth"] == chosen_infos[j]["depth"]:
                    same_overlap = True
                else:
                    diff_overlap = True
        if not (overlap_exists and same_overlap and diff_overlap):
            continue

        colors = random.sample(MARKER_COLORS, len(chosen_infos))
        for info, color in zip(chosen_infos, colors):
            info["color"] = color

        actual = _paint_from_markers(shape, template, chosen_infos, mode="actual")
        scan_only = _paint_from_markers(shape, template, chosen_infos, mode="scan_only")
        reverse_depth = _paint_from_markers(shape, template, chosen_infos, mode="reverse_depth")
        reverse_same = _paint_from_markers(shape, template, chosen_infos, mode="reverse_same_depth")

        if np.array_equal(actual, scan_only):
            continue
        if np.array_equal(actual, reverse_depth):
            continue
        if np.array_equal(actual, reverse_same):
            continue
        if np.count_nonzero(actual) < int(template.sum()):
            continue

        return chosen_infos

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            h, w, top, left, side, template_corner = _pick_layout()
        except ValueError:
            continue

        template = random.choice(TEMPLATE_PATTERNS)
        grid = np.zeros((h, w), dtype=int)
        layer_count = None

        outer_radius = side // 2
        ring_sides: list[int] = []
        for frame_side in range(side, 0, -4):
            if frame_side < 3:
                break
            ring_sides.append(frame_side)
        layer_count = len(ring_sides)
        if layer_count not in (3, 4):
            continue

        for frame_side in ring_sides:
            inset = (side - frame_side) // 2
            _draw_square_frame(grid, top + inset, left + inset, frame_side)

        _place_template(grid, template, template_corner)

        center2 = (2 * (top + outer_radius), 2 * (left + outer_radius))
        ring_radii2 = sorted(
            {
                max(abs(2 * r - center2[0]), abs(2 * c - center2[1]))
                for r, c in np.argwhere(grid == GRAY)
            },
            reverse=True,
        )
        candidates = _candidate_markers(grid, center2, ring_radii2, top, left, side, template)
        if len(candidates) < 5:
            continue

        markers = _choose_markers(candidates, template, grid.shape)
        if markers is None:
            continue

        for marker in markers:
            grid[marker["r"], marker["c"]] = marker["color"]

        expected = _paint_from_markers(grid.shape, template, markers, mode="actual")
        solved = main(grid)
        if not np.array_equal(expected, solved):
            continue

        if np.any((solved == GRAY) | (solved == ORANGE)):
            continue

        marker_mask = (grid != BG) & (grid != GRAY) & (grid != ORANGE)
        if np.count_nonzero(marker_mask) != len(markers):
            continue

        return grid

    raise RuntimeError("failed to generate a valid non-degenerate input")
