"""Repair masked rectangles by restoring each occluder region with the higher-priority color.

Input structure:
- A 2D grid with a dominant background color.
- Several solid single-color rectangles are placed above the last row.
- One rectangle may be fully or partially hidden when a smaller single-color rectangle sits on top of it inside its bounds.
- The last row is a color-priority bar: contiguous 2-cell color blocks ordered from left (higher priority) to right (lower priority).

Rule:
- Keep the priority bar (the last row) unchanged.
- Detect masking rectangles as solid rectangles whose four sides are strictly surrounded by one single non-background color on all four neighboring side strips.
- For each detected mask rectangle, look up its color in the priority bar and take the immediate higher-priority color (its left neighbor on the bar).
- Repaint the whole mask rectangle with that higher-priority color.
- Leave all other cells unchanged.

Key insight:
- Reconstruction depends on geometric mask detection plus bar-encoded precedence; the hidden color is read from the bar, not inferred from nearby visible region color only."""

from __future__ import annotations

from collections import deque
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _extract_priority_order(bottom_row: Sequence[int]) -> List[int]:
    if len(bottom_row) == 0:
        return []
    order = []
    i = 0
    w = len(bottom_row)
    while i < w:
        c = int(bottom_row[i])
        j = i + 1
        while j < w and bottom_row[j] == c:
            j += 1
        order.append(c)
        i = j
    return order


def _largest_solid_rectangles(grid: np.ndarray) -> List[Dict[str, int]]:
    """Return all 4-connected solid-rectangular components in rows excluding the bar row."""
    h, w = grid.shape
    if h <= 1:
        return []

    visited = np.zeros((h - 1, w), dtype=bool)
    out: List[Dict[str, int]] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(h - 1):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            min_r, max_r, min_c, max_c = r, r, c, c

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                min_r = min(min_r, rr)
                max_r = max(max_r, rr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= h - 1 or nc < 0 or nc >= w:
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            area = len(cells)
            bbox_h = max_r - min_r + 1
            bbox_w = max_c - min_c + 1
            if area == bbox_h * bbox_w:
                out.append(
                    {
                        "color": color,
                        "min_r": min_r,
                        "max_r": max_r,
                        "min_c": min_c,
                        "max_c": max_c,
                    }
                )

    return out


def _uniform_value(values: Sequence[int]) -> int | None:
    if not values:
        return None
    first = int(values[0])
    for v in values[1:]:
        if int(v) != first:
            return None
    return first


def _is_mask_rectangle(grid: np.ndarray, obj: Dict[str, int], bg: int) -> Tuple[bool, int | None]:
    """Check whether obj is a top-layer mask rectangle and return (is_mask, underlying_color)."""
    h, w = grid.shape
    min_r, max_r = obj["min_r"], obj["max_r"]
    min_c, max_c = obj["min_c"], obj["max_c"]
    color = obj["color"]

    if min_r <= 0 or min_c <= 0 or max_r >= h - 2 or max_c >= w - 1:
        return False, None

    top = _uniform_value([int(v) for v in grid[min_r - 1, min_c : max_c + 1]])
    bottom = _uniform_value([int(v) for v in grid[max_r + 1, min_c : max_c + 1]])
    left = _uniform_value([int(v) for v in grid[min_r : max_r + 1, min_c - 1]])
    right = _uniform_value([int(v) for v in grid[min_r : max_r + 1, max_c + 1]])

    if top is None or bottom is None or left is None or right is None:
        return False, None

    if top != bottom or top != left or top != right:
        return False, None

    if top == color:
        return False, None
    if top == bg:
        return False, None

    return True, top


def main(grid: np.ndarray) -> np.ndarray:
    """Restore occluded rectangle interiors based on the top-level priority bar."""
    arr = np.array(grid, copy=True)
    if arr.ndim != 2 or arr.size == 0:
        return arr

    if arr.shape[0] < 2:
        return arr

    h, w = arr.shape
    if w == 0:
        return arr

    bg = int(np.bincount(arr[:-1, :].ravel()).argmax()) if h > 1 else int(np.bincount(arr.ravel()).argmax())

    if h < 2:
        return arr

    priority = _extract_priority_order(arr[h - 1])
    if len(priority) < 2:
        return arr

    higher_map: Dict[int, int] = {}
    for i in range(1, len(priority)):
        higher_map[int(priority[i])] = int(priority[i - 1])

    # Only recolor top-layer solids that are strict inner masks.
    rectangles = _largest_solid_rectangles(arr)
    for obj in rectangles:
        is_mask, _ = _is_mask_rectangle(arr, obj, bg)
        if not is_mask:
            continue
        mask_color = obj["color"]
        if mask_color not in higher_map:
            continue
        repl = higher_map[mask_color]
        arr[obj["min_r"] : obj["max_r"] + 1, obj["min_c"] : obj["max_c"] + 1] = repl

    return arr


def _random_rect_within(
    h: int, w: int,
    existing: np.ndarray,
    min_h: int,
    min_w: int,
    margin: int,
) -> Tuple[int, int, int, int] | None:
    hmax = h - margin
    wmax = w - margin
    if hmax < min_h or wmax < min_w:
        return None

    attempts = 200
    for _ in range(attempts):
        rh = random.randint(min_h, min(10, hmax))
        rw = random.randint(min_w, min(10, wmax))
        r0 = random.randint(0, h - rh - margin)
        c0 = random.randint(0, w - rw - margin)
        if existing[r0 : r0 + rh, c0 : c0 + rw].any():
            continue
        return r0, c0, rh, rw
    return None


def generate_input() -> np.ndarray:
    """Generate ARC-like inputs with optional masked-rectangle occlusion and a bottom priority bar."""
    while True:
        bar_len = random.randint(4, 12)
        w = bar_len * 2

        h = random.randint(9, 30)
        # keep enough room above the bar
        if h <= 4:
            continue

        # Build a 2-cell-per-color priority bar in the bottom row.\n
        palette = list(range(1, 21))
        bar_colors = random.sample(palette, bar_len)
        bg = 0
        grid = np.full((h, w), bg, dtype=np.int64)

        for i, color in enumerate(bar_colors):
            grid[h - 1, 2 * i : 2 * i + 2] = color

        fill_area = h - 1
        occupied = np.zeros((fill_area, w), dtype=bool)
        # Reserve a small perimeter to avoid edge-cases that make masks ambiguous.
        available_margin = 1

        max_objects = random.randint(1, 5)
        available_colors = [c for c in palette if c not in bar_colors]
        random.shuffle(available_colors)

        objects = []
        mask_count = 0

        for obj_idx in range(max_objects):
            color = available_colors[obj_idx % len(available_colors)]
            # Keep enough room for future masks inside an object.
            can_be_masked = random.random() < 0.55 and random.random() < 0.8
            min_h = 3 if can_be_masked else 1
            min_w = 3 if can_be_masked else 1
            rect = _random_rect_within(fill_area, w, occupied, min_h=min_h, min_w=min_w, margin=available_margin)
            if rect is None:
                continue
            r0, c0, rh, rw = rect
            grid[r0 : r0 + rh, c0 : c0 + rw] = color
            occupied[r0 : r0 + rh, c0 : c0 + rw] = True
            objects.append((r0, c0, rh, rw, color))

        # Add one branch where no masks are used.
        if random.random() > 0.55 or not objects:
            # Keep the scene clean, only non-overlapped full rectangles.
            return grid

        # Place 1..2 masks per selected object when room allows.
        used_mask_area = np.zeros((fill_area, w), dtype=bool)

        for r0, c0, rh, rw, _ in objects:
            if rh < 3 or rw < 3:
                continue

            if random.random() > 0.7:
                continue

            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                if random.random() > 0.85:
                    continue

                m_h = random.randint(1, max(1, rh - 2))
                m_w = random.randint(1, max(1, rw - 2))
                m_r0_min = r0 + 1
                m_r0_max = r0 + rh - m_h - 1
                m_c0_min = c0 + 1
                m_c0_max = c0 + rw - m_w - 1
                if m_r0_min > m_r0_max or m_c0_min > m_c0_max:
                    continue

                m_r0 = random.randint(m_r0_min, m_r0_max)
                m_c0 = random.randint(m_c0_min, m_c0_max)

                if used_mask_area[m_r0 : m_r0 + m_h, m_c0 : m_c0 + m_w].any():
                    continue

                # mask color must have a higher-priority color to the left.
                idx = random.randint(1, len(bar_colors) - 1)
                mask_color = bar_colors[idx]
                if used_mask_area[m_r0 : m_r0 + m_h, m_c0 : m_c0 + m_w].any():
                    continue

                grid[m_r0 : m_r0 + m_h, m_c0 : m_c0 + m_w] = mask_color
                used_mask_area[m_r0 : m_r0 + m_h, m_c0 : m_c0 + m_w] = True
                mask_count += 1

        # ensure at least one mask appears in this branch
        if mask_count == 0:
            continue

        return grid
