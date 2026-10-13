from __future__ import annotations

from collections import Counter
import random

import numpy as np


_DIR_SPECS = {
    "right": {
        "head": [(0, 3), (1, 4), (2, 3)],
        "arrow_bbox": (0, 0, 2, 4),
        "project_delta": (0, 6),
        "reserve_bbox": (0, 0, 2, 8),
    },
    "left": {
        "head": [(0, -1), (1, -2), (2, -1)],
        "arrow_bbox": (0, -2, 2, 2),
        "project_delta": (0, -6),
        "reserve_bbox": (0, -6, 2, 2),
    },
    "down": {
        "head": [(3, 0), (4, 1), (3, 2)],
        "arrow_bbox": (0, 0, 4, 2),
        "project_delta": (6, 0),
        "reserve_bbox": (0, 0, 8, 2),
    },
    "up": {
        "head": [(-1, 0), (-2, 1), (-1, 2)],
        "arrow_bbox": (-2, 0, 2, 2),
        "project_delta": (-6, 0),
        "reserve_bbox": (-6, 0, 2, 2),
    },
}

_BODY_COLORS = [1, 2, 3, 4, 5]
_HEAD_COLORS = [6, 7, 8, 9]
_MAX_GENERATION_ATTEMPTS = 500
_MIN_SIDE_BY_COUNT = {1: 15, 2: 18, 3: 22, 4: 24}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.flat).most_common(1)[0][0])


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _rect_from_rel(base_r: int, base_c: int, rel_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rel_bbox
    return int(base_r + r0), int(base_c + c0), int(base_r + r1), int(base_c + c1)


def _mask_cells(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _build_expected_arrow_patch(
    body_color: int,
    head_color: int,
    direction: str,
) -> np.ndarray:
    spec = _DIR_SPECS[direction]
    r0, c0, r1, c1 = spec["arrow_bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    patch = np.zeros((h, w), dtype=int)

    for dr in range(3):
        for dc in range(3):
            patch[int(dr - r0), int(dc - c0)] = int(body_color)
    for dr, dc in spec["head"]:
        patch[int(dr - r0), int(dc - c0)] = int(head_color)
    return patch


def _detect_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    used = np.zeros((h, w), dtype=bool)
    arrows: list[dict] = []

    for r in range(h - 2):
        for c in range(w - 2):
            if bool(used[r : r + 3, c : c + 3].any()):
                continue
            body_color = int(g[r, c])
            if body_color == int(bg):
                continue
            body = g[r : r + 3, c : c + 3]
            if not bool((body == int(body_color)).all()):
                continue

            matched = None
            for direction, spec in _DIR_SPECS.items():
                head_vals = []
                head_coords_abs = []
                ok = True
                for dr, dc in spec["head"]:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if not _in_bounds(h, w, rr, cc):
                        ok = False
                        break
                    head_vals.append(int(g[rr, cc]))
                    head_coords_abs.append((rr, cc))
                if not ok:
                    continue

                head_color = head_vals[0]
                if head_color in (int(bg), int(body_color)):
                    continue
                if any(v != head_color for v in head_vals):
                    continue

                rr0, cc0, rr1, cc1 = _rect_from_rel(r, c, spec["arrow_bbox"])
                patch = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
                expected = _build_expected_arrow_patch(int(body_color), int(head_color), direction)
                if patch.shape != expected.shape or not np.array_equal(patch, expected):
                    continue

                matched = {
                    "direction": direction,
                    "body_color": int(body_color),
                    "head_color": int(head_color),
                    "body_r": int(r),
                    "body_c": int(c),
                    "cells": {(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3)} | set(head_coords_abs),
                }
                break

            if matched is not None:
                arrows.append(matched)
                for rr, cc in matched["cells"]:
                    used[int(rr), int(cc)] = True

    assert arrows, "expected at least one arrow"
    body_colors = [arrow["body_color"] for arrow in arrows]
    assert len(set(body_colors)) == len(body_colors), "body colors must pair arrows uniquely to patterns"
    return arrows


def _extract_pattern_for_color(
    grid: np.ndarray,
    bg: int,
    body_color: int,
    arrow_mask: np.ndarray,
) -> np.ndarray:
    coords = np.argwhere((grid == int(body_color)) & ~arrow_mask)
    assert coords.size > 0, f"missing pattern for color {body_color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0 + 1) == 3 and int(c1 - c0 + 1) == 3, "pattern bbox must be 3x3"
    patch = np.asarray(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int).copy()
    assert set(map(int, np.unique(patch))) <= {int(bg), int(body_color)}
    assert bool((patch == int(body_color)).any()) and bool((patch == int(bg)).any())
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect color-paired arrows and project the matched 3x3 pattern in the arrow's facing direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    arrows = _detect_arrows(g, bg)
    arrow_mask = np.zeros((h, w), dtype=bool)
    for arrow in arrows:
        for rr, cc in arrow["cells"]:
            arrow_mask[int(rr), int(cc)] = True

    patterns = {
        arrow["body_color"]: _extract_pattern_for_color(g, bg, arrow["body_color"], arrow_mask)
        for arrow in arrows
    }

    out = g.copy()
    for arrow in arrows:
        dr, dc = _DIR_SPECS[arrow["direction"]]["project_delta"]
        r0 = int(arrow["body_r"] + dr)
        c0 = int(arrow["body_c"] + dc)
        patch = patterns[arrow["body_color"]]
        assert 0 <= r0 and r0 + 3 <= h and 0 <= c0 and c0 + 3 <= w
        target = out[r0 : r0 + 3, c0 : c0 + 3]
        paint = patch != int(bg)
        assert not bool(np.any((target != int(bg)) & paint)), "projection must land on blank cells"
        target[paint] = patch[paint]

    return out


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((int(r + dr), int(c + dc)))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors8(r, c):
            if 0 <= rr < 3 and 0 <= cc < 3 and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _random_pattern_mask() -> np.ndarray:
    while True:
        count = random.randint(3, 6)
        picks = random.sample(range(9), count)
        mask = np.zeros((3, 3), dtype=bool)
        for idx in picks:
            mask[idx // 3, idx % 3] = True
        if any(not bool(mask[r, :].any()) for r in range(3)):
            continue
        if any(not bool(mask[:, c].any()) for c in range(3)):
            continue
        if not _is_connected(mask):
            continue
        return mask


def _paste_pattern(arr: np.ndarray, r0: int, c0: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    view = arr[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    paint = patch != 0
    view[paint] = patch[paint]


def _rect_fits(h: int, w: int, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return 0 <= int(r0) <= int(r1) < int(h) and 0 <= int(c0) <= int(c1) < int(w)


def _can_place_rect(
    occupied: np.ndarray,
    rect: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = rect
    if not _rect_fits(h, w, rect):
        return False
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(h - 1, int(r1 + pad))
    cc1 = min(w - 1, int(c1 + pad))
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_rect(occupied: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    occupied[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True


def _candidate_body_positions(h: int, w: int, direction: str) -> list[tuple[int, int]]:
    positions = []
    for r in range(h - 2):
        for c in range(w - 2):
            rect = _rect_from_rel(r, c, _DIR_SPECS[direction]["reserve_bbox"])
            if _rect_fits(h, w, rect):
                positions.append((int(r), int(c)))
    random.shuffle(positions)
    return positions


def _render_arrow(grid: np.ndarray, body_r: int, body_c: int, body_color: int, head_color: int, direction: str) -> None:
    grid[int(body_r) : int(body_r) + 3, int(body_c) : int(body_c) + 3] = int(body_color)
    for dr, dc in _DIR_SPECS[direction]["head"]:
        grid[int(body_r + dr), int(body_c + dc)] = int(head_color)


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    n_arrows = random.randint(1, 4)
    min_side = _MIN_SIDE_BY_COUNT[int(n_arrows)]
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    body_colors = random.sample(_BODY_COLORS, n_arrows)
    head_colors = [random.choice(_HEAD_COLORS) for _ in range(n_arrows)]
    directions = random.sample(list(_DIR_SPECS.keys()), n_arrows)
    pattern_masks = [_random_pattern_mask() for _ in range(n_arrows)]

    occupied = np.zeros((h, w), dtype=bool)
    arrows: list[dict] = []
    for body_color, head_color, direction, mask in zip(body_colors, head_colors, directions, pattern_masks):
        del mask
        placed = False
        for body_r, body_c in _candidate_body_positions(h, w, direction):
            reserve = _rect_from_rel(body_r, body_c, _DIR_SPECS[direction]["reserve_bbox"])
            if not _can_place_rect(occupied, reserve, pad=1):
                continue
            _mark_rect(occupied, reserve)
            arrows.append(
                {
                    "body_r": int(body_r),
                    "body_c": int(body_c),
                    "body_color": int(body_color),
                    "head_color": int(head_color),
                    "direction": direction,
                }
            )
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place arrow")

    patterns: list[dict] = []
    for body_color, mask in zip(body_colors, pattern_masks):
        patch = np.zeros((3, 3), dtype=int)
        patch[mask] = int(body_color)
        cells = [(r, c) for r in range(h - 2) for c in range(w - 2)]
        random.shuffle(cells)
        placed = False
        for r0, c0 in cells:
            rect = (int(r0), int(c0), int(r0 + 2), int(c0 + 2))
            if not _can_place_rect(occupied, rect, pad=1):
                continue
            _mark_rect(occupied, rect)
            patterns.append({"color": int(body_color), "r0": int(r0), "c0": int(c0), "patch": patch})
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place pattern")

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros_like(inp)
    for arrow in arrows:
        _render_arrow(
            inp,
            arrow["body_r"],
            arrow["body_c"],
            arrow["body_color"],
            arrow["head_color"],
            arrow["direction"],
        )
        _render_arrow(
            out,
            arrow["body_r"],
            arrow["body_c"],
            arrow["body_color"],
            arrow["head_color"],
            arrow["direction"],
        )

    for pat in patterns:
        _paste_pattern(inp, pat["r0"], pat["c0"], pat["patch"])
        _paste_pattern(out, pat["r0"], pat["c0"], pat["patch"])

    for arrow in arrows:
        patch = next(p["patch"] for p in patterns if p["color"] == arrow["body_color"])
        dr, dc = _DIR_SPECS[arrow["direction"]]["project_delta"]
        _paste_pattern(out, int(arrow["body_r"] + dr), int(arrow["body_c"] + dc), patch)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected = _build_scene()
        except RuntimeError:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
