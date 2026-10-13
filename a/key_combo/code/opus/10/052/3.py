from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BARRIER = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 500
GRID_SIZES = [17, 19, 21, 23, 25, 27]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _cross_from_grid(grid: np.ndarray) -> dict[str, object]:
    candidates = []
    for comp in _components(grid != BG):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) < 2 or colors == {BARRIER}:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        rows = np.zeros(h, dtype=int)
        cols = np.zeros(w, dtype=int)
        rel = set()
        for r, c in comp:
            rr = r - r0
            cc = c - c0
            rows[rr] += 1
            cols[cc] += 1
            rel.add((rr, cc))
        row_hits = np.flatnonzero(rows == w)
        col_hits = np.flatnonzero(cols == h)
        if row_hits.size != 1 or col_hits.size != 1:
            continue
        rr = int(row_hits[0])
        cc = int(col_hits[0])
        if any(v not in (0, 1, w) for v in rows) or any(v not in (0, 1, h) for v in cols):
            continue
        expected_size = h + w - 1
        if len(comp) != expected_size:
            continue
        if h % 2 != 1 or w % 2 != 1 or rr != h // 2 or cc != w // 2:
            continue
        center = (r0 + rr, c0 + cc)
        hseq = [int(grid[center[0], center[1] + d]) for d in range(w // 2 + 1)]
        vseq = [int(grid[center[0] + d, center[1]]) for d in range(h // 2 + 1)]
        if any(int(grid[center[0], center[1] - d]) != hseq[d] for d in range(w // 2 + 1)):
            continue
        if any(int(grid[center[0] - d, center[1]]) != vseq[d] for d in range(h // 2 + 1)):
            continue
        if hseq[0] != vseq[0]:
            continue
        candidates.append(
            {
                "center": center,
                "hseq": hseq,
                "vseq": vseq,
                "bbox": (r0, c0, r1, c1),
            }
        )
    assert len(candidates) == 1, "expected exactly one multicolor symmetric cross"
    return candidates[0]


def _barrier_info(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], list[dict[str, object]]]:
    comps = _components(grid == BARRIER)
    cell_to_idx: dict[tuple[int, int], int] = {}
    info: list[dict[str, object]] = []
    for idx, comp in enumerate(comps):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert 2 <= len(comp) <= 4, "barriers must be short segments"
        if len(rows) == 1:
            orient = "H"
        elif len(cols) == 1:
            orient = "V"
        else:
            raise AssertionError("barrier must be straight")
        for cell in comp:
            cell_to_idx[cell] = idx
        info.append({"cells": comp, "orient": orient})
    return cell_to_idx, info


def _trace_axis(
    out: np.ndarray,
    src: np.ndarray,
    center: tuple[int, int],
    direction: str,
    seq: list[int],
    barrier_cells: dict[tuple[int, int], int],
    barrier_info: list[dict[str, object]],
) -> None:
    h, w = src.shape
    dr, dc = DIRS[direction]
    expect_orient = "V" if dc else "H"
    step = 1
    while True:
        r = center[0] + dr * step
        c = center[1] + dc * step
        if not (0 <= r < h and 0 <= c < w):
            break
        if (r, c) in barrier_cells:
            barrier = barrier_info[barrier_cells[(r, c)]]
            assert barrier["orient"] == expect_orient, "ray must meet a perpendicular barrier"
            out[r, c] = seq[step % len(seq)]
            break
        out[r, c] = seq[step % len(seq)]
        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the central cross periodically and recolor the contacted barrier pixel where each ray is stopped.

    Input structure:
    - The grid has a black background, one connected multicolor width-1 cross, and 2-4 maroon barrier segments.
    - The cross is symmetric around its center, so the center row and center column encode outward color sequences.
    - Each barrier is a straight 2-4 cell segment; ray-blocking barriers intersect the center row or center column and are perpendicular to that ray.

    Rule:
    - Read the horizontal and vertical color sequences from the cross center outward.
    - From the center, extend those sequences periodically left/right and up/down across the canvas.
    - If a ray reaches a perpendicular maroon barrier, stop before passing through it and recolor only the contacted barrier pixel with the color due at that distance.
    - If a direction has no barrier, extend that ray to the grid edge.

    Key insight:
    - The obstacle color is not fixed: the contacted barrier pixel takes the periodic ray color determined by its exact distance from the cross center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross = _cross_from_grid(g)
    center = cross["center"]
    barrier_cells, barrier_info = _barrier_info(g)

    out = g.copy()
    for direction in ("left", "right"):
        _trace_axis(
            out,
            g,
            center,
            direction,
            list(cross["hseq"]),
            barrier_cells,
            barrier_info,
        )
    for direction in ("up", "down"):
        _trace_axis(
            out,
            g,
            center,
            direction,
            list(cross["vseq"]),
            barrier_cells,
            barrier_info,
        )
    return out


def _draw_cross(grid: np.ndarray, center: tuple[int, int], hseq: list[int], vseq: list[int]) -> None:
    cr, cc = center
    for d, color in enumerate(hseq):
        grid[cr, cc + d] = color
        grid[cr, cc - d] = color
    for d, color in enumerate(vseq):
        grid[cr + d, cc] = color
        grid[cr - d, cc] = color


def _segment_cells(
    center: tuple[int, int],
    direction: str,
    distance: int,
    length: int,
    anchor: int,
) -> list[tuple[int, int]]:
    cr, cc = center
    if direction in ("left", "right"):
        c = cc - distance if direction == "left" else cc + distance
        top = cr - anchor
        return [(top + i, c) for i in range(length)]
    r = cr - distance if direction == "up" else cr + distance
    left = cc - anchor
    return [(r, left + i) for i in range(length)]


def _fits_segment(
    cells: list[tuple[int, int]],
    n: int,
    occupied: set[tuple[int, int]],
    cross_bbox: tuple[int, int, int, int],
    center: tuple[int, int],
) -> bool:
    r0, c0, r1, c1 = cross_bbox
    for r, c in cells:
        if not (0 <= r < n and 0 <= c < n):
            return False
        if (r, c) in occupied:
            return False
        if (r, c) == center:
            return False
        if r0 - 2 <= r <= r1 + 2 and c0 - 2 <= c <= c1 + 2:
            return False
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def _pick_axis_sequences() -> tuple[list[int], list[int]]:
    hlen = random.randint(3, 5)
    vlen = random.randint(3, 5)
    center_color = random.choice(PALETTE)
    h_rest = random.sample([c for c in PALETTE if c != center_color], hlen - 1)
    v_rest = random.sample([c for c in PALETTE if c != center_color], vlen - 1)
    return [center_color] + h_rest, [center_color] + v_rest


def _active_directions() -> list[str]:
    if random.random() < 0.5:
        return [random.choice(["left", "right"]), random.choice(["up", "down"])]
    free = random.choice(list(DIRS))
    return [d for d in DIRS if d != free]


def _place_active_barriers(
    n: int,
    center: tuple[int, int],
    cross_bbox: tuple[int, int, int, int],
    hseq: list[int],
    vseq: list[int],
    directions: list[str],
) -> list[dict[str, object]] | None:
    active: list[dict[str, object]] = []
    occupied = set()
    cr, cc = center

    limits = {
        "left": cc - 1,
        "right": n - cc - 2,
        "up": cr - 1,
        "down": n - cr - 2,
    }
    mins = {
        "left": len(hseq) + 1,
        "right": len(hseq) + 1,
        "up": len(vseq) + 1,
        "down": len(vseq) + 1,
    }
    for direction in directions:
        if mins[direction] > limits[direction]:
            return None

    order = directions[:]
    random.shuffle(order)
    for direction in order:
        seq = hseq if direction in ("left", "right") else vseq
        distances = list(range(mins[direction], limits[direction] + 1))
        random.shuffle(distances)
        placed = None
        for distance in distances:
            length = random.randint(2, 4)
            anchors = list(range(length))
            random.shuffle(anchors)
            for anchor in anchors:
                cells = _segment_cells(center, direction, distance, length, anchor)
                if not _fits_segment(cells, n, occupied, cross_bbox, center):
                    continue
                placed = {
                    "direction": direction,
                    "distance": distance,
                    "length": length,
                    "cells": cells,
                    "contact": (cr + DIRS[direction][0] * distance, cc + DIRS[direction][1] * distance),
                    "color": seq[distance % len(seq)],
                }
                break
            if placed is not None:
                break
        if placed is None:
            return None
        occupied.update(placed["cells"])
        active.append(placed)

    if len({barrier["color"] for barrier in active}) < 2:
        return None
    return active


def _place_distractors(
    n: int,
    center: tuple[int, int],
    cross_bbox: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    count: int,
) -> list[list[tuple[int, int]]] | None:
    distractors: list[list[tuple[int, int]]] = []
    cr, cc = center
    for _ in range(count):
        placed = None
        for _ in range(80):
            orient = random.choice(["H", "V"])
            length = random.randint(2, 4)
            if orient == "H":
                r = random.randrange(n)
                if r == cr:
                    continue
                c0 = random.randrange(n - length + 1)
                cells = [(r, c0 + i) for i in range(length)]
            else:
                c = random.randrange(n)
                if c == cc:
                    continue
                r0 = random.randrange(n - length + 1)
                cells = [(r0 + i, c) for i in range(length)]
            if _fits_segment(cells, n, occupied, cross_bbox, center):
                placed = cells
                break
        if placed is None:
            return None
        occupied.update(placed)
        distractors.append(placed)
    return distractors


def _render_input(spec: dict[str, object]) -> np.ndarray:
    n = int(spec["size"])
    grid = np.zeros((n, n), dtype=int)
    _draw_cross(grid, spec["center"], list(spec["hseq"]), list(spec["vseq"]))
    for barrier in spec["active"]:
        for r, c in barrier["cells"]:
            grid[r, c] = BARRIER
    for cells in spec["distractors"]:
        for r, c in cells:
            grid[r, c] = BARRIER
    return grid


def _render_output(spec: dict[str, object], inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    center = tuple(spec["center"])
    hseq = list(spec["hseq"])
    vseq = list(spec["vseq"])
    contacts = {tuple(barrier["contact"]): int(barrier["color"]) for barrier in spec["active"]}

    for direction in ("left", "right"):
        dr, dc = DIRS[direction]
        step = 1
        while True:
            r = center[0] + dr * step
            c = center[1] + dc * step
            if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                break
            if (r, c) in contacts:
                out[r, c] = contacts[(r, c)]
                break
            out[r, c] = hseq[step % len(hseq)]
            step += 1

    for direction in ("up", "down"):
        dr, dc = DIRS[direction]
        step = 1
        while True:
            r = center[0] + dr * step
            c = center[1] + dc * step
            if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                break
            if (r, c) in contacts:
                out[r, c] = contacts[(r, c)]
                break
            out[r, c] = vseq[step % len(vseq)]
            step += 1
    return out


def _validate_structure(spec: dict[str, object], inp: np.ndarray, out: np.ndarray) -> bool:
    cross = _cross_from_grid(inp)
    if tuple(cross["center"]) != tuple(spec["center"]):
        return False
    if list(cross["hseq"]) != list(spec["hseq"]) or list(cross["vseq"]) != list(spec["vseq"]):
        return False

    barrier_cells, barrier_info = _barrier_info(inp)
    expected_barriers = len(spec["active"]) + len(spec["distractors"])
    if len(barrier_info) != expected_barriers:
        return False

    active_by_direction = {barrier["direction"]: barrier for barrier in spec["active"]}
    if len(active_by_direction) != len(spec["active"]):
        return False
    if len(spec["active"]) not in (2, 3):
        return False
    if expected_barriers not in (2, 3, 4):
        return False
    if len(set(DIRS) - set(active_by_direction)) < 1:
        return False

    center = tuple(spec["center"])
    hseq = list(spec["hseq"])
    vseq = list(spec["vseq"])
    for direction in DIRS:
        dr, dc = DIRS[direction]
        seq = hseq if direction in ("left", "right") else vseq
        seen_contact = None
        step = 1
        while True:
            r = center[0] + dr * step
            c = center[1] + dc * step
            if not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]):
                break
            if (r, c) in barrier_cells:
                orient = barrier_info[barrier_cells[(r, c)]]["orient"]
                if direction in ("left", "right") and orient != "V":
                    return False
                if direction in ("up", "down") and orient != "H":
                    return False
                seen_contact = (r, c, seq[step % len(seq)])
                break
            step += 1
        if direction in active_by_direction:
            if seen_contact is None:
                return False
            if tuple(active_by_direction[direction]["contact"]) != seen_contact[:2]:
                return False
            if int(active_by_direction[direction]["color"]) != int(seen_contact[2]):
                return False
            if out[seen_contact[0], seen_contact[1]] != seen_contact[2]:
                return False
        else:
            if seen_contact is not None:
                return False

    if len({int(barrier["color"]) for barrier in spec["active"]}) < 2:
        return False
    return True


def _build_spec() -> dict[str, object] | None:
    n = random.choice(GRID_SIZES)
    center = (n // 2, n // 2)
    hseq, vseq = _pick_axis_sequences()
    cross_bbox = (
        center[0] - (len(vseq) - 1),
        center[1] - (len(hseq) - 1),
        center[0] + (len(vseq) - 1),
        center[1] + (len(hseq) - 1),
    )
    active_dirs = _active_directions()
    active = _place_active_barriers(n, center, cross_bbox, hseq, vseq, active_dirs)
    if active is None:
        return None
    occupied = {cell for barrier in active for cell in barrier["cells"]}
    distractor_count = random.choice([0, 0, 1, 1 if len(active) == 3 else 0])
    distractors = _place_distractors(n, center, cross_bbox, occupied, distractor_count)
    if distractors is None:
        return None
    return {
        "size": n,
        "center": center,
        "hseq": hseq,
        "vseq": vseq,
        "active": active,
        "distractors": distractors,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        spec = _build_spec()
        if spec is None:
            continue
        inp = _render_input(spec)
        expected = _render_output(spec, inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if not _validate_structure(spec, inp, expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid periodic-cross barrier input")
