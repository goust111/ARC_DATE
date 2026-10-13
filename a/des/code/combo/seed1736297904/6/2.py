from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
UP, DOWN, LEFT, RIGHT = DIRS
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(g.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            out.append({"color": color, "coords": coords})
    return out


def _is_arrow(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 2:
        return False
    (r0, c0), (r1, c1) = coords
    return abs(r0 - r1) + abs(c0 - c1) == 1


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = int(np.sign(r1 - r0))
    dc = int(np.sign(c1 - c0))
    assert (dr == 0) ^ (dc == 0), "segment must be axis-aligned"
    out = [(r0, c0)]
    r, c = r0, c0
    while (r, c) != (r1, c1):
        r += dr
        c += dc
        out.append((r, c))
    return out


def _zone3(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _follow_chain(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g)
    arrows = [comp for comp in comps if _is_arrow(comp["coords"])]  # type: ignore[index]
    blobs = [comp for comp in comps if not _is_arrow(comp["coords"])]  # type: ignore[index]
    assert arrows, "expected at least one arrow"
    assert len(blobs) == 1, "expected exactly one non-arrow blob"
    blob = blobs[0]
    blob_set = set(blob["coords"])  # type: ignore[arg-type]

    arrow_cells: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(arrows):
        for cell in comp["coords"]:  # type: ignore[index]
            key = tuple(cell)
            assert key not in arrow_cells, "arrow cells must be disjoint"
            arrow_cells[key] = idx

    first_candidates: list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for idx, comp in enumerate(arrows):
        cells = [tuple(cell) for cell in comp["coords"]]  # type: ignore[index]
        for center, head in (cells, cells[::-1]):
            dr = head[0] - center[0]
            dc = head[1] - center[1]
            assert (dr, dc) in DIRS
            spacer = (center[0] - dr, center[1] - dc)
            anchor = (center[0] - 2 * dr, center[1] - 2 * dc)
            if not _in_bounds(g.shape, spacer[0], spacer[1]):
                continue
            if not _in_bounds(g.shape, anchor[0], anchor[1]):
                continue
            if int(g[spacer]) != BG:
                continue
            if anchor not in blob_set:
                continue
            first_candidates.append((idx, center, (dr, dc), anchor))

    assert len(first_candidates) == 1, "blob must identify a unique first arrow"
    first_id, first_center, first_dir, blob_anchor = first_candidates[0]

    order: list[dict[str, object]] = []
    remaining = set(range(len(arrows)))
    cur_id = first_id
    cur_center = first_center
    cur_dir = first_dir

    while True:
        comp = arrows[cur_id]
        order.append(
            {
                "id": cur_id,
                "center": cur_center,
                "direction": cur_dir,
                "color": int(comp["color"]),  # type: ignore[index]
                "cells": [tuple(cell) for cell in comp["coords"]],  # type: ignore[index]
            }
        )
        remaining.remove(cur_id)

        r, c = cur_center
        dr, dc = cur_dir
        next_id = None
        next_center = None
        while True:
            r += dr
            c += dc
            if not _in_bounds(g.shape, r, c):
                break
            hit = arrow_cells.get((r, c))
            if hit is None:
                continue
            if hit == cur_id:
                continue
            assert hit in remaining, "chain cannot revisit an earlier arrow"
            next_id = hit
            next_center = (r, c)
            break

        if next_id is None:
            break

        next_cells = [tuple(cell) for cell in arrows[next_id]["coords"]]  # type: ignore[index]
        assert next_center in next_cells
        other = next_cells[0] if next_cells[1] == next_center else next_cells[1]
        next_dir = (other[0] - next_center[0], other[1] - next_center[1])
        assert next_dir in DIRS
        assert next_dir != (-cur_dir[0], -cur_dir[1]), "cannot hit an arrow head before its center"

        cur_id = next_id
        cur_center = next_center
        cur_dir = next_dir

    assert not remaining, "all arrows must belong to the same chain"
    return order, arrows, blob, blob_anchor


def _translated_blob(
    shape: tuple[int, int],
    blob_coords: list[tuple[int, int]],
    anchor: tuple[int, int],
    tail_center: tuple[int, int],
    tail_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    offsets = [(r - anchor[0], c - anchor[1]) for r, c in blob_coords]
    dr, dc = tail_dir
    best: list[tuple[int, int]] | None = None
    step = 1
    while True:
        anchor_out = (tail_center[0] + dr * step, tail_center[1] + dc * step)
        coords = [(anchor_out[0] + rr, anchor_out[1] + cc) for rr, cc in offsets]
        if all(_in_bounds(shape, r, c) for r, c in coords):
            best = coords
            step += 1
            continue
        break
    assert best is not None, "tail must leave room for the translated blob"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat 3x3 arrow glyphs as teleport instructions, color the travel rays,
    erase the arrows, and rebuild the blob near the chain tail.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    order, arrows, blob, blob_anchor = _follow_chain(g)
    out = g.copy()

    for idx, arrow in enumerate(order):
        center = tuple(arrow["center"])  # type: ignore[arg-type]
        direction = tuple(arrow["direction"])  # type: ignore[arg-type]
        color = int(arrow["color"])  # type: ignore[arg-type]
        dr, dc = direction
        stop = tuple(order[idx + 1]["center"]) if idx + 1 < len(order) else None  # type: ignore[arg-type]

        r, c = center
        while True:
            r += dr
            c += dc
            if not _in_bounds(g.shape, r, c):
                break
            if int(g[r, c]) == BG:
                out[r, c] = color
            if stop is not None and (r, c) == stop:
                break

    for r, c in blob["coords"]:  # type: ignore[index]
        out[r, c] = BG
    for comp in arrows:
        for r, c in comp["coords"]:  # type: ignore[index]
            out[r, c] = BG

    tail = order[-1]
    tail_center = tuple(tail["center"])  # type: ignore[arg-type]
    tail_dir = tuple(tail["direction"])  # type: ignore[arg-type]
    tail_color = int(tail["color"])  # type: ignore[arg-type]
    moved_blob = _translated_blob(
        g.shape,
        [tuple(cell) for cell in blob["coords"]],  # type: ignore[index]
        blob_anchor,
        tail_center,
        tail_dir,
    )

    for r, c in moved_blob:
        assert int(out[r, c]) in (BG, tail_color), "blob may only cover background or the tail ray"
    for r, c in moved_blob:
        out[r, c] = int(blob["color"])  # type: ignore[index]

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
ARROW_COUNT_RANGE = (2, 5)
SEG_LEN_RANGE = (4, 8)
MAX_GENERATE_ATTEMPTS = 400
START_LIMIT = 90

CANONICAL_BLOBS = [
    [(0, 0), (0, -1), (-1, -1)],
    [(0, 0), (0, -1), (1, -1)],
    [(0, 0), (0, -1), (-1, -1), (1, -1)],
    [(0, 0), (0, -1), (0, -2), (-1, -2)],
    [(0, 0), (0, -1), (0, -2), (1, -2)],
    [(0, 0), (0, -1), (-1, -1), (-1, -2)],
    [(0, 0), (0, -1), (1, -1), (1, -2)],
]


def _rotate_from_right(offset: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r, c = offset
    if direction == RIGHT:
        return (r, c)
    if direction == DOWN:
        return (c, -r)
    if direction == LEFT:
        return (-r, -c)
    assert direction == UP
    return (-c, r)


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    return {center, (center[0] + direction[0], center[1] + direction[1])}


def _center_ok(shape: tuple[int, int], center: tuple[int, int], direction: tuple[int, int]) -> bool:
    r, c = center
    hr, hc = r + direction[0], c + direction[1]
    return 1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1 and _in_bounds(shape, hr, hc)


def _sample_directions(count: int) -> list[tuple[int, int]]:
    dirs = [random.choice(DIRS)]
    for i in range(1, count):
        prev = dirs[-1]
        options = [d for d in DIRS if d != (-prev[0], -prev[1])]
        if i == count - 1 and dirs[0] in options and random.random() < 0.45:
            dirs.append(dirs[0])
            continue
        turns = [d for d in options if d != prev]
        if turns and random.random() < 0.65:
            dirs.append(random.choice(turns))
        else:
            dirs.append(random.choice(options))
    return dirs


def _blob_offsets(direction: tuple[int, int]) -> list[tuple[int, int]]:
    shape = random.choice(CANONICAL_BLOBS)
    return [_rotate_from_right(offset, direction) for offset in shape]


def _try_build_centers(
    shape: tuple[int, int],
    dirs: list[tuple[int, int]],
    blob_cells: set[tuple[int, int]],
    center0: tuple[int, int],
) -> list[tuple[int, int]] | None:
    count = len(dirs)
    occupied = set(blob_cells) | _arrow_cells(center0, dirs[0])
    if occupied & (_zone3(center0) - _arrow_cells(center0, dirs[0])):
        return None
    blocked = set(occupied)
    centers = [center0]

    def dfs(i: int) -> bool:
        if i == count - 1:
            return True

        cur = centers[i]
        cur_dir = dirs[i]
        next_dir = dirs[i + 1]
        cur_arrow = _arrow_cells(cur, cur_dir)
        lengths = list(range(SEG_LEN_RANGE[0], SEG_LEN_RANGE[1] + 1))
        random.shuffle(lengths)

        for dist in lengths:
            nxt = (cur[0] + cur_dir[0] * dist, cur[1] + cur_dir[1] * dist)
            if not _center_ok(shape, nxt, next_dir):
                continue
            next_arrow = _arrow_cells(nxt, next_dir)
            if next_dir == (-cur_dir[0], -cur_dir[1]):
                continue
            if any(cell in blocked for cell in next_arrow):
                continue
            if occupied & (_zone3(nxt) - next_arrow):
                continue

            seg = _segment_cells(cur, nxt)
            interior = seg[1:-1]
            if any(cell in blocked and cell not in cur_arrow for cell in interior):
                continue
            if nxt in blocked:
                continue
            if next_arrow - {nxt} & set(seg[:-1]):
                continue

            centers.append(nxt)
            occupied_update = next_arrow
            blocked_update = set(seg[1:]) | next_arrow
            occupied.update(occupied_update)
            blocked.update(blocked_update)
            if dfs(i + 1):
                return True
            centers.pop()
            occupied.difference_update(occupied_update)
            blocked.difference_update(blocked_update)

        return False

    if dfs(0):
        return centers
    return None


def _tail_ray(shape: tuple[int, int], center: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    dr, dc = direction
    out = set()
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            break
        out.add((r, c))
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)
    arrow_count = random.randint(*ARROW_COUNT_RANGE)
    dirs = _sample_directions(arrow_count)
    blob_offsets = _blob_offsets(dirs[0])

    colors = random.sample(PALETTE, arrow_count + 1)
    blob_color = colors[0]
    arrow_colors = colors[1:]

    starts: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center0 = (r, c)
            if not _center_ok(shape, center0, dirs[0]):
                continue
            anchor = (center0[0] - 2 * dirs[0][0], center0[1] - 2 * dirs[0][1])
            blob_cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in blob_offsets}
            if not all(_in_bounds(shape, rr, cc) for rr, cc in blob_cells):
                continue
            if blob_cells & _zone3(center0):
                continue
            starts.append((center0, blob_cells))

    random.shuffle(starts)
    for center0, blob_cells in starts[:START_LIMIT]:
        centers = _try_build_centers(shape, dirs, blob_cells, center0)
        if centers is None:
            continue

        original_blob = set(blob_cells)
        occupied = set(original_blob)
        for center, direction in zip(centers, dirs):
            occupied.update(_arrow_cells(center, direction))

        try:
            tail_blob = _translated_blob(
                shape,
                list(original_blob),
                (center0[0] - 2 * dirs[0][0], center0[1] - 2 * dirs[0][1]),
                centers[-1],
                dirs[-1],
            )
        except AssertionError:
            continue
        tail_ray = _tail_ray(shape, centers[-1], dirs[-1])
        allowed = tail_ray | _arrow_cells(centers[-1], dirs[-1]) | original_blob
        if any(cell in occupied and cell not in allowed for cell in tail_blob):
            continue

        grid = np.zeros(shape, dtype=int)
        for r, c in original_blob:
            grid[r, c] = blob_color
        for color, center, direction in zip(arrow_colors, centers, dirs):
            cr, cc = center
            hr, hc = center[0] + direction[0], center[1] + direction[1]
            grid[cr, cc] = color
            grid[hr, hc] = color

        try:
            out = main(grid)
        except AssertionError:
            continue

        if out.shape != grid.shape:
            continue
        if not np.any(out == blob_color):
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-arrow input")
