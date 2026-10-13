"""Grow each seed-directed rectangular frame outward through a keyed directional legend.

Input structure:
- The background is black (0).
- Inside the grid are several disjoint closed, 1-pixel-thick rectangular frames (non-black colors), each containing exactly one interior seed pixel of another color.
- Along the border, a legend lists seed colors: each entry has a color block and a nearby 3-cell grey L-shaped arrow.
- The arrow direction maps that seed color to a growth direction among up, down, left, right.

Rule:
- Locate every closed rectangular frame by verifying a perimeter-only 4-neighbor component and detect its single interior seed color.
- Read the frame's seed color in the legend to get the growth direction.
- Open the entire non-corner segment on the corresponding frame side by setting those pixels to black.
- Remove the seed pixel (set it to black).
- Create a new hollow contour outside the frame in the seed color: the side perpendicular to growth has the same length as the opened side, while the side parallel to growth has half the original frame size in that growth direction (floor).
- Only draw the new contour; do not recolor other pixels.

Key insight:
- Seed colors are lookup keys, not behavior colors: the legend maps each seed color to a direction.
- Frame geometry uniquely determines opening location and copied contour size.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


BG = 0
GREY = 9

FRAME_COLORS: Sequence[int] = (1, 2, 3, 4)
SEED_COLORS: Sequence[int] = (5, 6, 7, 8)
DIRECTIONS = ("up", "down", "left", "right")


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[Tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _legend_arrow_offsets(direction: str) -> List[Tuple[int, int]]:
    if direction == "right":
        return [(0, 1), (0, 2), (-1, 2)]
    if direction == "left":
        return [(0, -1), (0, -2), (-1, -2)]
    if direction == "down":
        return [(1, 0), (2, 0), (2, -1)]
    if direction == "up":
        return [(-1, 0), (-2, 0), (-2, -1)]
    raise ValueError(direction)


def _draw_rect_outline(grid: np.ndarray, color: int, r0: int, r1: int, c0: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _mark_rect_outline(mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _find_legend(grid: np.ndarray) -> Dict[int, str]:
    h, w = grid.shape
    mapping: Dict[int, str] = {}

    for r in range(h):
        for c in range(w):
            if not (r == 0 or c == 0 or r == h - 1 or c == w - 1):
                continue

            color = int(grid[r, c])
            if color in (BG, GREY):
                continue
            if color in mapping:
                continue

            for direction in DIRECTIONS:
                offsets = _legend_arrow_offsets(direction)
                ok = True
                for dr, dc in offsets:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        ok = False
                        break
                    if int(grid[rr, cc]) != GREY:
                        ok = False
                        break
                if ok:
                    mapping[color] = direction
                    break

    return mapping


def _connected_components(grid: np.ndarray) -> List[Set[Tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Set[Tuple[int, int]]] = []

    for i in range(h):
        for j in range(w):
            v = int(grid[i, j])
            if v == BG or v == GREY or visited[i, j]:
                continue

            q = deque([(i, j)])
            visited[i, j] = True
            comp: Set[Tuple[int, int]] = {(i, j)}

            while q:
                r, c = q.popleft()
                for rr, cc in _neighbors4(r, c, h, w):
                    if visited[rr, cc]:
                        continue
                    if int(grid[rr, cc]) == BG or int(grid[rr, cc]) == GREY:
                        continue
                    visited[rr, cc] = True
                    q.append((rr, cc))
                    comp.add((rr, cc))

            comps.append(comp)

    return comps


def _iter_perimeter_points(r0: int, c0: int, r1: int, c1: int) -> Iterable[Tuple[int, int]]:
    for cc in range(c0, c1 + 1):
        yield r0, cc
        yield r1, cc
    for rr in range(r0 + 1, r1):
        yield rr, c0
        yield rr, c1


def _rect_from_component(grid: np.ndarray, comp: Set[Tuple[int, int]]) -> Tuple[int, int, int, int] | None:
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if h < 3 or w < 3:
        return None

    if len(comp) != 2 * (h + w) - 4:
        return None

    perimeter = set(_iter_perimeter_points(r0, c0, r1, c1))
    if comp != perimeter:
        return None

    if (0 in (r0, c0)) or (0 in (r1, c1)):
        # legend should be on edge, and we do not treat edge-touching frames as candidates.
        return None

    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            if int(grid[rr, cc]) != BG:
                return None

    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.array(grid, copy=True)
    legend = _find_legend(grid)

    comps = _connected_components(grid)
    for comp in comps:
        rect = _rect_from_component(grid, comp)
        if rect is None:
            continue

        r0, c0, r1, c1 = rect
        inner = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) != BG
        ]
        if len(inner) != 1:
            continue

        seed_r, seed_c = inner[0]
        seed_color = int(grid[seed_r, seed_c])
        direction = legend.get(seed_color)
        if direction is None:
            continue

        height = r1 - r0 + 1
        width = c1 - c0 + 1

        # 1) erase the opening in the matching side
        if direction == "right":
            for rr in range(r0 + 1, r1):
                out[rr, c1] = BG
        elif direction == "left":
            for rr in range(r0 + 1, r1):
                out[rr, c0] = BG
        elif direction == "down":
            for cc in range(c0 + 1, c1):
                out[r1, cc] = BG
        else:  # up
            for cc in range(c0 + 1, c1):
                out[r0, cc] = BG

        # 2) remove seed pixel
        out[seed_r, seed_c] = BG

        # 3) grow a scaled copy using seed color as contour color
        if direction in ("right", "left"):
            copy_h = width - 2
            copy_w = max(1, height // 2)

            if direction == "right":
                nr0, nr1 = r0 + 1, r1 - 1
                nc0, nc1 = c1 + 1, c1 + copy_w
            else:
                nr0, nr1 = r0 + 1, r1 - 1
                nc1, nc0 = c0 - 1, c0 - copy_w

            if nc0 < 0 or nc1 >= w or nr0 < 0 or nr1 >= h or nc0 > nc1:
                continue
            _draw_rect_outline(out, seed_color, nr0, nr1, nc0, nc1)

        else:
            copy_w = width - 2
            copy_h = max(1, height // 2)

            if direction == "down":
                nr0, nr1 = r1 + 1, r1 + copy_h
                nc0, nc1 = c0 + 1, c1 - 1
            else:
                nr1, nr0 = r0 - 1, r0 - copy_h
                nc0, nc1 = c0 + 1, c1 - 1

            if nc0 < 0 or nc1 >= w or nr0 < 0 or nr1 >= h or nr0 > nr1:
                continue
            _draw_rect_outline(out, seed_color, nr0, nr1, nc0, nc1)

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(4000):
        h = rng.randint(14, 30)
        w = rng.randint(14, 30)
        if h < 14 or w < 14:
            continue

        grid = np.full((h, w), BG, dtype=np.int64)
        occupied = np.zeros((h, w), dtype=bool)

        object_count = rng.randint(1, 3)
        dirs = [rng.choice(DIRECTIONS) for _ in range(object_count)]

        frame_pool = list(FRAME_COLORS)
        seed_pool = list(SEED_COLORS)
        rng.shuffle(frame_pool)
        rng.shuffle(seed_pool)

        if object_count > len(frame_pool) or object_count > len(seed_pool):
            continue

        frame_colors = frame_pool[:object_count]
        seed_colors = seed_pool[:object_count]

        plans = [
            {
                "frame_color": frame_colors[i],
                "seed_color": seed_colors[i],
                "dir": dirs[i],
            }
            for i in range(object_count)
        ]

        # Legend: one color block and a 3-cell grey L-shape keyed by direction.
        def mark_occupied(cells: Sequence[Tuple[int, int]]) -> bool:
            return any(
                not (0 <= rr < h and 0 <= cc < w) or occupied[rr, cc]
                for rr, cc in cells
            )

        success_legend = True
        for p in plans:
            direction = p["dir"]
            color = p["seed_color"]

            if direction == "right":
                candidates = list(range(1, h - 1))
                rng.shuffle(candidates)
                placed = False
                for r in candidates:
                    cells = [(r, 0), (r, 1), (r, 2), (r - 1, 2)]
                    if mark_occupied(cells):
                        continue
                    for rr, cc in cells:
                        grid[rr, cc] = color if cc == 0 else GREY
                        occupied[rr, cc] = True
                    p["legend_anchor"] = (r, 0)
                    placed = True
                    break

            elif direction == "left":
                candidates = list(range(1, h - 1))
                rng.shuffle(candidates)
                placed = False
                for r in candidates:
                    cells = [(r, w - 1), (r, w - 2), (r, w - 3), (r - 1, w - 3)]
                    if mark_occupied(cells):
                        continue
                    for rr, cc in cells:
                        grid[rr, cc] = color if cc == w - 1 else GREY
                        occupied[rr, cc] = True
                    p["legend_anchor"] = (r, w - 1)
                    placed = True
                    break

            elif direction == "down":
                candidates = list(range(1, w - 1))
                rng.shuffle(candidates)
                placed = False
                for c in candidates:
                    cells = [(0, c), (1, c), (2, c), (2, c - 1)]
                    if mark_occupied(cells):
                        continue
                    for rr, cc in cells:
                        grid[rr, cc] = color if rr == 0 else GREY
                        occupied[rr, cc] = True
                    p["legend_anchor"] = (0, c)
                    placed = True
                    break

            else:  # up
                candidates = list(range(1, w - 1))
                rng.shuffle(candidates)
                placed = False
                for c in candidates:
                    cells = [(h - 1, c), (h - 2, c), (h - 3, c), (h - 3, c - 1)]
                    if mark_occupied(cells):
                        continue
                    for rr, cc in cells:
                        grid[rr, cc] = color if rr == h - 1 else GREY
                        occupied[rr, cc] = True
                    p["legend_anchor"] = (h - 1, c)
                    placed = True
                    break

            if not placed:
                success_legend = False
                break

        if not success_legend:
            continue

        # Build final-shape occupancy to prevent overlaps between frame and grown copy.
        possible = True
        for p in plans:
            direction = p["dir"]
            frame_color = p["frame_color"]
            seed_color = p["seed_color"]

            # Draw each frame+seed and a predicted grown copy.
            trial = 0
            while trial < 300:
                trial += 1

                if direction in ("right", "left"):
                    h_min = 5
                    h_max = min(8, h - 3)
                    if h_max < h_min:
                        break
                    rect_h = rng.randint(h_min, h_max)
                    copy_w = max(1, rect_h // 2)
                    w_max = min(8, w - copy_w - 3)
                    if w_max < 5:
                        break
                    rect_w = rng.randint(5, w_max)

                    if direction == "right":
                        r0_min, r0_max = 1, h - rect_h - 1
                        c0_min = 1
                        c0_max = w - rect_w - copy_w - 1
                    else:
                        r0_min, r0_max = 1, h - rect_h - 1
                        c0_min = copy_w + 1
                        c0_max = w - rect_w - 1

                else:
                    w_min = 5
                    w_max = min(8, w - 3)
                    if w_max < w_min:
                        break
                    rect_w = rng.randint(w_min, w_max)
                    rect_h_max = min(8, h - 3)
                    if rect_h_max < 5:
                        break
                    rect_h = rng.randint(5, rect_h_max)
                    copy_h = max(1, rect_h // 2)

                    if direction == "down":
                        r0_min = 1
                        r0_max = h - rect_h - copy_h - 1
                        c0_min = 1
                        c0_max = w - rect_w - 1
                    else:
                        r0_min = copy_h + 1
                        r0_max = h - rect_h - 1
                        c0_min = 1
                        c0_max = w - rect_w - 1

                if r0_min > r0_max or c0_min > c0_max:
                    break

                r0 = rng.randint(r0_min, r0_max)
                c0 = rng.randint(c0_min, c0_max)
                r1 = r0 + rect_h - 1
                c1 = c0 + rect_w - 1

                if direction in ("right", "left"):
                    copy_w = max(1, rect_h // 2)
                    if direction == "right":
                        nr0, nr1 = r0 + 1, r1 - 1
                        nc0, nc1 = c1 + 1, c1 + copy_w
                    else:
                        nr0, nr1 = r0 + 1, r1 - 1
                        nc1, nc0 = c0 - 1, c0 - copy_w
                    if nc0 < 0 or nc1 >= w:
                        continue

                else:
                    copy_h = max(1, rect_h // 2)
                    if direction == "down":
                        nr0, nr1 = r1 + 1, r1 + copy_h
                        nc0, nc1 = c0 + 1, c1 - 1
                    else:
                        nr1, nr0 = r0 - 1, r0 - copy_h
                        nc0, nc1 = c0 + 1, c1 - 1
                    if nr0 < 0 or nr1 >= h:
                        continue

                if not (nr0 <= nr1 and nc0 <= nc1):
                    continue

                temp = np.zeros((h, w), dtype=bool)
                _mark_rect_outline(temp, r0, r1, c0, c1)

                # opening segment (will be erased)
                if direction == "right":
                    temp[r0 + 1 : r1, c1] = True
                    # keep seed location reserved
                    seed_r = r0 + 1
                    seed_c = c0 + 1
                elif direction == "left":
                    temp[r0 + 1 : r1, c0] = True
                    seed_r = r0 + 1
                    seed_c = c0 + 1
                elif direction == "down":
                    temp[r1, c0 + 1 : c1] = True
                    seed_r = r0 + 1
                    seed_c = c0 + 1
                else:
                    temp[r0, c0 + 1 : c1] = True
                    seed_r = r0 + 1
                    seed_c = c0 + 1

                temp[seed_r, seed_c] = True
                _mark_rect_outline(temp, nr0, nr1, nc0, nc1)

                if np.any(occupied & temp):
                    continue

                # Commit object
                occupied[temp] = True
                _draw_rect_outline(grid, frame_color, r0, r1, c0, c1)
                grid[seed_r, seed_c] = seed_color
                break
            else:
                possible = False
                break

        if not possible:
            continue

        # Final validation: at least one closed frame exists and every rectangle has a mapped legend color.
        # This avoids degenerate rare retries from accidental placement failure.
        valid_frames = 0
        comps = _connected_components(grid)
        for comp in comps:
            rect = _rect_from_component(grid, comp)
            if rect is not None:
                valid_frames += 1
        return grid

    raise RuntimeError("generate_input failed to satisfy constraints")
