from __future__ import annotations

import random

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_bbox(component: np.ndarray) -> tuple[int, int, int, int] | None:
    if component.size == 0:
        return None
    r0, c0 = component.min(axis=0)
    r1, c1 = component.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    expected = 2 * (h + w) - 4
    if int(component.shape[0]) != expected:
        return None
    for r, c in component:
        rr = int(r)
        cc = int(c)
        if rr not in (r0, r1) and cc not in (c0, c1):
            return None
    return r0, c0, r1, c1


def _detect_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    frame_candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for comp in _components(grid == color):
            box = _frame_bbox(comp)
            if box is None:
                ok = False
                break
            boxes.append(box)
        if ok and boxes:
            frame_candidates.append((color, boxes))
    assert len(frame_candidates) == 1, "expected a unique frame color"
    frame_color, frame_boxes = frame_candidates[0]
    return int(frame_color), frame_boxes


def _max_layers(inner_h: int, inner_w: int) -> int:
    assert inner_h >= 1 and inner_w >= 1
    return (min(inner_h, inner_w) + 1) // 2


def _paint_rings(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    layers: int,
    fill_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    usable = min(max(0, int(layers)), _max_layers(inner_h, inner_w))
    for depth in range(usable):
        top = r0 + 1 + depth
        bottom = r1 - 1 - depth
        left = c0 + 1 + depth
        right = c1 - 1 - depth
        out[top, left : right + 1] = fill_color
        out[bottom, left : right + 1] = fill_color
        out[top : bottom + 1, left] = fill_color
        out[top : bottom + 1, right] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert interior dot counts into concentric shrinking fill rings.

    Input structure:
    - A dominant background color.
    - Several disjoint one-pixel-thick rectangular frames of one color.
    - A second foreground color used only for interior dots.

    Rule:
    - For each frame, count the dots strictly inside it.
    - Fill exactly that many shrinking inner rings with the dot color.
    - If the count exceeds the available ring depth, fill the whole interior.
    - Remove the original dots and preserve the frame borders.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    frame_color, frame_boxes = _detect_frames(g, bg)

    interior_mask = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    dot_candidates: list[int] = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, frame_color):
            continue
        coords = np.argwhere(g == color)
        if coords.size > 0 and bool(np.all(interior_mask[coords[:, 0], coords[:, 1]])):
            dot_candidates.append(color)

    if not dot_candidates:
        return g.copy()
    assert len(dot_candidates) == 1, "expected a unique dot color"
    dot_color = int(dot_candidates[0])

    out = g.copy()
    for r0, c0, r1, c1 in frame_boxes:
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == dot_color] = bg

    for box in frame_boxes:
        r0, c0, r1, c1 = box
        count = int(np.count_nonzero(g[r0 + 1 : r1, c0 + 1 : c1] == dot_color))
        _paint_rings(out, box, count, dot_color)

    assert bool(np.all(out[g == frame_color] == frame_color))
    return out


def _dot_pool(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
) -> list[tuple[int, int]]:
    row_offset = random.randint(0, min(1, inner_h - 1))
    col_offset = random.randint(0, min(1, inner_w - 1))
    cells: list[tuple[int, int]] = []
    for dr in range(row_offset, inner_h, 2):
        for dc in range(col_offset, inner_w, 2):
            cells.append((r0 + 1 + dr, c0 + 1 + dc))
    random.shuffle(cells)
    return cells


def _sample_count(max_depth: int, capacity: int) -> int:
    assert capacity >= 1
    choices = ["normal"]
    if capacity > max_depth:
        choices.append("overflow")
    choices.append("zero")
    kind = random.choice(choices)
    if kind == "zero":
        return 0
    if kind == "normal":
        return random.randint(1, max_depth)
    return random.randint(max_depth + 1, capacity)


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h > hh - 1 or c0 + w > ww - 1:
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(hh, r0 + h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(ww, c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_frames(h: int, w: int, specs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for outer_h, outer_w in sorted(specs, key=lambda item: item[0] * item[1], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - outer_h):
            for c0 in range(1, w - outer_w):
                if _can_place(occupied, r0, c0, outer_h, outer_w):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _mark_occupied(occupied, r0, c0, outer_h, outer_w)
        boxes.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))
    return boxes


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_candidate() -> np.ndarray | None:
    h = random.randint(14, 28)
    w = random.randint(14, 28)
    n_frames = random.randint(1, 4)

    specs: list[tuple[int, int]] = []
    for _ in range(n_frames):
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 8)
        specs.append((inner_h + 2, inner_w + 2))

    boxes = _place_frames(h, w, specs)
    if boxes is None:
        return None

    bg, frame_color, dot_color = random.sample(list(range(10)), 3)
    grid = np.full((h, w), bg, dtype=int)
    for box in boxes:
        _draw_frame(grid, box, frame_color)

    counts: list[int] = []
    dot_slots: list[list[tuple[int, int]]] = []
    for r0, c0, r1, c1 in boxes:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        pool = _dot_pool(r0, c0, inner_h, inner_w)
        if not pool:
            return None
        max_depth = _max_layers(inner_h, inner_w)
        count = _sample_count(max_depth, len(pool))
        counts.append(count)
        dot_slots.append(pool)

    if not any(counts):
        idx = random.randrange(len(counts))
        r0, c0, r1, c1 = boxes[idx]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        capacity = len(dot_slots[idx])
        counts[idx] = random.randint(1, min(capacity, _max_layers(inner_h, inner_w)))

    for count, slots in zip(counts, dot_slots):
        for r, c in slots[:count]:
            grid[r, c] = dot_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
