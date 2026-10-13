from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
PALETTE_POOL = [2, 3, 4, 5, 6, 7, 8, 9]

GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 200
CANVAS_ATTEMPTS = 80


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    next_id = 0

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or labels[r, c] != -1:
                continue
            labels[r, c] = next_id
            queue = deque([(r, c)])
            coords = [(r, c)]
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(mask.shape, nr, nc):
                        continue
                    if not bool(mask[nr, nc]) or labels[nr, nc] != -1:
                        continue
                    labels[nr, nc] = next_id
                    queue.append((nr, nc))
                    coords.append((nr, nc))
            components.append(coords)
            next_id += 1

    return labels, components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _boundary_mask(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    ring = np.zeros_like(solid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(solid[r, c]):
                continue
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(solid.shape, nr, nc) or not bool(solid[nr, nc]):
                    ring[r, c] = True
                    break
    return ring


def _count_enclosed_false_components(mask: np.ndarray) -> int:
    inv_labels, inv_components = _label_components(~mask)
    del inv_labels
    h, w = mask.shape
    count = 0
    for coords in inv_components:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        count += 1
    return count


def _is_single_cycle(ring: np.ndarray) -> bool:
    _, components = _label_components(ring)
    if len(components) != 1:
        return False
    for r, c in components[0]:
        degree = 0
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(ring.shape, nr, nc) and bool(ring[nr, nc]):
                degree += 1
        if degree != 2:
            return False
    return True


def _loop_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    black_labels, black_components = _label_components(g == BLACK)
    black_touches_edge = []
    black_sizes = []
    h, w = g.shape
    for coords in black_components:
        black_touches_edge.append(
            any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)
        )
        black_sizes.append(len(coords))

    _, blue_components = _label_components(g == BLUE)
    records = []
    for coords in blue_components:
        inside_ids: set[int] = set()
        outside_cells: set[tuple[int, int]] = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(g.shape, nr, nc) or int(g[nr, nc]) != BLACK:
                    continue
                bid = int(black_labels[nr, nc])
                if bid < 0:
                    continue
                if black_touches_edge[bid]:
                    outside_cells.add((nr, nc))
                else:
                    inside_ids.add(bid)
        area = sum(black_sizes[bid] for bid in inside_ids)
        records.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "inside_ids": inside_ids,
                "outside_cells": outside_cells,
                "area": int(area),
            }
        )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each loop's outside contour by the rank of the black area enclosed by that loop.

    Input structure:
    - Row 0 is a palette row whose non-black cells form a left-to-right color sequence,
      separated by black cells.
    - Rows 1+ contain several disjoint blue one-pixel-thick closed loops on a black background.
    - Each loop encloses a different amount of black interior area and does not touch the palette row.

    Rule:
    - Read the palette as the non-black colors in row 0 from left to right.
    - For each blue loop, count the black cells enclosed by that loop: the adjacent black
      component(s) that do not reach the grid boundary.
    - Sort loops by enclosed area from smallest to largest, erase all blue loop pixels to black,
      and recolor only the black cells 4-adjacent to the loop on its exterior side with the
      palette color of the same rank.
    - Keep the palette row and the grid size unchanged.

    Key insight:
    - The loop geometry only tells you where to paint; the palette order is assigned by ranking
      enclosed black area rather than by the loops' positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = [int(v) for v in g[0] if int(v) != BLACK]
    records = _loop_records(g)
    assert palette, "expected a non-empty palette row"
    assert len(records) == len(palette), "expected one palette color per loop"

    out = g.copy()
    out[g == BLUE] = BLACK

    ranked = sorted(records, key=lambda rec: rec["area"])
    for color, rec in zip(palette, ranked):
        for r, c in rec["outside_cells"]:
            out[r, c] = int(color)
    return out


def _trace_to(path: list[tuple[int, int]], end: tuple[int, int]) -> None:
    r, c = path[-1]
    er, ec = end
    dr = 0 if er == r else (1 if er > r else -1)
    dc = 0 if ec == c else (1 if ec > c else -1)
    assert dr == 0 or dc == 0
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))


def _shape_record() -> dict | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(6, 10)
        w = random.randint(6, 10)
        side = random.choice(["top", "bottom", "left", "right"])
        path = [(0, 0)]

        if side in {"top", "bottom"}:
            if h < 6 or w < 6:
                continue
            depth = random.randint(1, min(3, h - 4))
            cut_w = random.randint(1, w - 4)
            c0 = random.randint(1, w - cut_w - 3)
            c1 = c0 + cut_w + 1
            notch = {"side": side, "depth": depth, "c0": c0, "c1": c1}
        else:
            if h < 6 or w < 6:
                continue
            depth = random.randint(1, min(3, w - 4))
            cut_h = random.randint(1, h - 4)
            r0 = random.randint(1, h - cut_h - 3)
            r1 = r0 + cut_h + 1
            notch = {"side": side, "depth": depth, "r0": r0, "r1": r1}

        if notch["side"] == "top":
            _trace_to(path, (0, notch["c0"]))
            _trace_to(path, (notch["depth"], notch["c0"]))
            _trace_to(path, (notch["depth"], notch["c1"]))
            _trace_to(path, (0, notch["c1"]))
        _trace_to(path, (0, w - 1))

        if notch["side"] == "right":
            _trace_to(path, (notch["r0"], w - 1))
            _trace_to(path, (notch["r0"], w - 1 - notch["depth"]))
            _trace_to(path, (notch["r1"], w - 1 - notch["depth"]))
            _trace_to(path, (notch["r1"], w - 1))
        _trace_to(path, (h - 1, w - 1))

        if notch["side"] == "bottom":
            _trace_to(path, (h - 1, notch["c1"]))
            _trace_to(path, (h - 1 - notch["depth"], notch["c1"]))
            _trace_to(path, (h - 1 - notch["depth"], notch["c0"]))
            _trace_to(path, (h - 1, notch["c0"]))
        _trace_to(path, (h - 1, 0))

        if notch["side"] == "left":
            _trace_to(path, (notch["r1"], 0))
            _trace_to(path, (notch["r1"], notch["depth"]))
            _trace_to(path, (notch["r0"], notch["depth"]))
            _trace_to(path, (notch["r0"], 0))
        _trace_to(path, (0, 0))
        path.pop()

        if len(path) != len(set(path)):
            continue

        ring = np.zeros((h, w), dtype=bool)
        for r, c in path:
            ring[r, c] = True
        if not _is_single_cycle(ring):
            continue

        probe = np.zeros((h + 2, w + 2), dtype=int)
        probe[1 : h + 1, 1 : w + 1][ring] = BLUE
        records = _loop_records(probe)
        if len(records) != 1:
            continue
        rec = records[0]
        if len(rec["inside_ids"]) != 1 or int(rec["area"]) < 2:
            continue

        outer_offsets = sorted((r - 1, c - 1) for r, c in rec["outside_cells"])
        return {
            "ring": ring,
            "area": int(rec["area"]),
            "h": int(h),
            "w": int(w),
            "outer_offsets": outer_offsets,
        }
    return None


def _expanded_rect(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0 - 2, c0 - 2, r0 + h + 1, c0 + w + 1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_records(records: list[dict], grid_shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    order = sorted(range(len(records)), key=lambda i: records[i]["h"] * records[i]["w"], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(records)
    occupied: list[tuple[int, int, int, int]] = []

    for idx in order:
        rec = records[idx]
        rh = int(rec["h"])
        rw = int(rec["w"])
        candidates = []
        for r0 in range(3, h - rh - 1):
            for c0 in range(2, w - rw - 1):
                rect = _expanded_rect(r0, c0, rh, rw)
                if any(_rects_overlap(rect, prev) for prev in occupied):
                    continue
                candidates.append((r0, c0, rect))
        if not candidates:
            return None
        r0, c0, rect = random.choice(candidates)
        placements[idx] = (r0, c0)
        occupied.append(rect)

    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]


def _palette_positions(width: int, n_colors: int) -> list[int] | None:
    for _ in range(100):
        cols = sorted(random.sample(range(1, width - 1), n_colors))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(n_colors - 1)):
            return cols
    return None


def _expected_output(inp: np.ndarray, records: list[dict], placements: list[tuple[int, int]], palette: list[int]) -> np.ndarray:
    out = inp.copy()
    out[out == BLUE] = BLACK

    ranked = sorted(range(len(records)), key=lambda i: records[i]["area"])
    for color, idx in zip(palette, ranked):
        rec = records[idx]
        r0, c0 = placements[idx]
        for dr, dc in rec["outer_offsets"]:
            gr = r0 + int(dr)
            gc = c0 + int(dc)
            if _in_bounds(out.shape, gr, gc) and int(inp[gr, gc]) == BLACK:
                out[gr, gc] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_loops = random.choice([2, 3, 3, 4])
    records: list[dict] = []
    used_areas: set[int] = set()

    for _ in range(200):
        if len(records) == n_loops:
            break
        rec = _shape_record()
        if rec is None or rec["area"] in used_areas:
            continue
        used_areas.add(int(rec["area"]))
        records.append(rec)
    if len(records) != n_loops:
        return None

    for _ in range(CANVAS_ATTEMPTS):
        grid_h = random.randint(18, 28)
        grid_w = random.randint(18, 28)
        placements = _place_records(records, (grid_h, grid_w))
        if placements is None:
            continue

        palette_positions = _palette_positions(grid_w, n_loops)
        if palette_positions is None:
            continue

        palette = random.sample(PALETTE_POOL, n_loops)
        inp = np.zeros((grid_h, grid_w), dtype=int)
        for color, c in zip(palette, palette_positions):
            inp[0, c] = int(color)

        for rec, (r0, c0) in zip(records, placements):
            ring = rec["ring"]
            rr, cc = np.where(ring)
            inp[r0 + rr, c0 + cc] = BLUE

        expected = _expected_output(inp, records, placements, palette)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
