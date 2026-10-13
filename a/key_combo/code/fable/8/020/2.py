from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        return [(r0, c) for c in range(c0, c1 + 1)]
    if c0 == c1:
        return [(r, c0) for r in range(r0, r1 + 1)]
    out = []
    for c in range(c0, c1 + 1):
        out.append((r0, c))
        out.append((r1, c))
    for r in range(r0 + 1, r1):
        out.append((r, c0))
        out.append((r, c1))
    return out


def _parse_concentric_legend(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[list[int], np.ndarray] | None:
    r0, c0, r1, c1 = _bbox(coords)
    actual = set(coords)
    colors: list[int] = []
    used = set()
    covered: set[tuple[int, int]] = set()
    depth = 0
    while r0 + depth <= r1 - depth and c0 + depth <= c1 - depth:
        rr0 = r0 + depth
        cc0 = c0 + depth
        rr1 = r1 - depth
        cc1 = c1 - depth
        if int(grid[rr0, cc0]) == BLACK:
            for r in range(rr0, rr1 + 1):
                for c in range(cc0, cc1 + 1):
                    if (r, c) in actual:
                        return None
            break
        border = _border_coords(rr0, cc0, rr1, cc1)
        color = int(grid[rr0, cc0])
        if color in used:
            return None
        if any(int(grid[r, c]) != color for r, c in border):
            return None
        colors.append(color)
        used.add(color)
        covered.update(border)
        depth += 1
    if covered != actual or len(colors) < 2:
        return None
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return colors, mask


def _is_hollow_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    return set(coords) == set(_border_coords(r0, c0, r1, c1))


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    candidates = []
    for coords in _components(grid != BLACK):
        parsed = _parse_concentric_legend(grid, coords)
        if parsed is not None:
            candidates.append(parsed)
    assert len(candidates) == 1, "expected exactly one multicolor concentric legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray hollow frame from the concentric ring legend using its interior teal-count index.

    Input structure:
    - The background is black.
    - Exactly one foreground object is a multicolor concentric rectangular legend made of one-cell-thick nested rings.
    - Other foreground objects are gray hollow rectangular frames plus teal single-pixel markers, some inside frames and some outside as distractors.

    Rule:
    - Read the legend colors from outer ring to inner ring.
    - For each gray frame, count only teal pixels strictly inside its border, then recolor the whole frame with the legend color at that count modulo the number of rings.
    - Erase every teal marker and leave the concentric legend unchanged.

    Key insight:
    - The multicolor nested object is not transformed; it is an ordered lookup table whose ring depth indexes the recolor choice for each separate gray frame.
    """
    g = np.asarray(grid, dtype=int)
    legend_colors, legend_mask = _find_legend(g)

    out = g.copy()
    teal_mask = (g == TEAL) & ~legend_mask

    grey_mask = (g == GREY) & ~legend_mask
    for coords in _components(grey_mask):
        assert _is_hollow_frame(coords), "gray objects must be hollow rectangular frames"
        r0, c0, r1, c1 = _bbox(coords)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(inner == TEAL))
        new_color = legend_colors[count % len(legend_colors)]
        for r, c in coords:
            out[r, c] = new_color

    out[teal_mask] = BLACK
    return out


def _paint_nested_legend(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    colors: list[int],
) -> None:
    for depth, color in enumerate(colors):
        r0 = top + depth
        c0 = left + depth
        r1 = top + height - 1 - depth
        c1 = left + width - 1 - depth
        for r, c in _border_coords(r0, c0, r1, c1):
            grid[r, c] = color


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _random_positions(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    return random.sample(cells, count)


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        grid = np.full((h, w), BLACK, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        ring_count = random.randint(3, 5)
        legend_h = random.randint(2 * ring_count + 1, min(11, h - 6))
        legend_w = random.randint(2 * ring_count + 1, min(11, w - 6))
        legend_colors = random.sample([1, 2, 3, 4, 6, 7, 9], ring_count)

        placed = False
        for _ in range(80):
            lr = random.randint(1, h - legend_h - 1)
            lc = random.randint(1, w - legend_w - 1)
            if _can_place(occupied, lr, lc, legend_h, legend_w, padding=1):
                _paint_nested_legend(grid, lr, lc, legend_h, legend_w, legend_colors)
                _mark_occupied(occupied, lr, lc, legend_h, legend_w)
                placed = True
                break
        if not placed:
            continue

        frame_count = random.randint(2, 4)
        target_counts = [0, ring_count + random.randint(1, max(2, ring_count))]
        while len(target_counts) < frame_count:
            target_counts.append(random.randint(1, ring_count + 2))
        random.shuffle(target_counts)

        frame_boxes: list[tuple[int, int, int, int, int]] = []
        ok = True
        for teal_count in target_counts:
            placed = False
            for _ in range(120):
                inner_h = random.randint(2, 5)
                inner_w = random.randint(2, 6)
                if inner_h * inner_w < teal_count:
                    continue
                fh = inner_h + 2
                fw = inner_w + 2
                fr = random.randint(1, h - fh - 1)
                fc = random.randint(1, w - fw - 1)
                if not _can_place(occupied, fr, fc, fh, fw, padding=1):
                    continue
                inside_cells = [(r, c) for r in range(fr + 1, fr + fh - 1) for c in range(fc + 1, fc + fw - 1)]
                marker_positions = _random_positions(inside_cells, teal_count)
                patch = np.full((fh, fw), BLACK, dtype=int)
                border_local = _border_coords(0, 0, fh - 1, fw - 1)
                for r, c in border_local:
                    patch[r, c] = GREY
                for r, c in marker_positions:
                    patch[r - fr, c - fc] = TEAL
                patch_coords = [(r, c) for r in range(fh) for c in range(fw) if patch[r, c] != BLACK]
                if _parse_concentric_legend(patch, patch_coords) is not None:
                    continue
                for r, c in _border_coords(fr, fc, fr + fh - 1, fc + fw - 1):
                    grid[r, c] = GREY
                for r, c in marker_positions:
                    grid[r, c] = TEAL
                _mark_occupied(occupied, fr, fc, fh, fw)
                frame_boxes.append((fr, fc, fh, fw, teal_count))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        distractor_count = random.randint(2, 6)
        free_cells = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if grid[r, c] != BLACK:
                    continue
                inside_any_frame = False
                for fr, fc, fh, fw, _count in frame_boxes:
                    if fr < r < fr + fh - 1 and fc < c < fc + fw - 1:
                        inside_any_frame = True
                        break
                if not inside_any_frame:
                    if any(grid[nr, nc] != BLACK for nr, nc in _neighbors4(r, c)):
                        continue
                    free_cells.append((r, c))
        if len(free_cells) < distractor_count:
            continue
        for r, c in random.sample(free_cells, distractor_count):
            grid[r, c] = TEAL

        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
