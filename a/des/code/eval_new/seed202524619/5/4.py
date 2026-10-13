from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    if vals.size == 0:
        raise ValueError("empty grid")
    return int(vals[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr_i, sc_i = int(sr), int(sc)
        if seen[sr_i, sc_i]:
            continue
        stack = [(sr_i, sc_i)]
        seen[sr_i, sc_i] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_of(points: np.ndarray) -> tuple[int, int, int, int]:
    if points.size == 0:
        raise ValueError("empty component")
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0))
    return r0, c0, r1, c1


def _collect_markers_and_frame(g: np.ndarray, bg: int) -> list[dict[str, Any]]:
    h, w = map(int, g.shape)
    frames: list[dict[str, Any]] = []
    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        for comp in _connected_components(g == color):
            if comp.shape[0] < 8:
                continue
            r0, c0, r1, c1 = _bbox_of(comp)
            hh = int(r1 - r0 + 1)
            ww = int(c1 - c0 + 1)
            if hh < 3 or ww < 3:
                continue

            area = int(comp.shape[0])
            expected_area = 2 * (hh + ww) - 4
            if area != expected_area:
                continue

            comp_set = {tuple(map(int, p)) for p in comp}
            ok = True
            for rr in range(r0, r1 + 1):
                if (rr, c0) not in comp_set or (rr, c1) not in comp_set:
                    ok = False
                    break
            if not ok:
                continue
            for cc in range(c0, c1 + 1):
                if (r0, cc) not in comp_set or (r1, cc) not in comp_set:
                    ok = False
                    break
            if not ok:
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or int(np.any(inner == color)):
                continue

            inner_markers = [
                (int(rr), int(cc))
                for rr in range(r0 + 1, r1)
                for cc in range(c0 + 1, c1)
                if int(g[rr, cc]) != bg
            ]
            if not inner_markers:
                continue

            frames.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "markers": inner_markers,
                }
            )
    frames.sort(key=lambda f: (f["r0"], f["c0"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw outward rays from the outer side of each hollow rectangular frame based on marker positions.

    Input structure:
    - bg is the most common color.
    - Several non-overlapping 1-pixel-thick hollow rectangles exist, each with its own border color.
    - Inside each rectangle, one-color 1x1 marker cells lie on one inner side only.

    Rule:
    - For each frame, infer the side where its markers lie (top, bottom, left, or right).
    - Emit one ray per marker, parallel to each other, outward from the corresponding frame side.
    - Every ray has length equal to the frame's marker count and color equal to the frame border color.
    - Rays start at the pixel just outside the frame edge and stop at grid boundary.

    Key insight:
    - Marker-side gives the ray direction, marker count gives the ray multiplicity and length, and frame color gives the stroke color; these three channels are independent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _most_common_color(g)

    frames = _collect_markers_and_frame(g, bg)
    assert frames, "no valid frame detected"

    out = g.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        r1 = int(frame["r1"])
        c1 = int(frame["c1"])
        frame_color = int(frame["color"])
        markers: list[tuple[int, int]] = frame["markers"]

        side_counts: dict[str, int] = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        for r, c in markers:
            if r == r0 + 1:
                side_counts["top"] += 1
            if r == r1 - 1:
                side_counts["bottom"] += 1
            if c == c0 + 1:
                side_counts["left"] += 1
            if c == c1 - 1:
                side_counts["right"] += 1

        side = None
        for k, v in side_counts.items():
            if v:
                assert side is None, "markers must touch exactly one frame edge per frame"
                side = k
        assert side is not None, "invalid marker placement"

        if side == "top":
            dr, dc = -1, 0
        elif side == "bottom":
            dr, dc = 1, 0
        elif side == "left":
            dr, dc = 0, -1
        else:
            dr, dc = 0, 1

        ray_len = len(markers)
        assert ray_len > 0, "frame must contain at least one marker"
        for mr, mc in markers:
            for step in range(1, ray_len + 1):
                rr = int(mr + dr * step)
                cc = int(mc + dc * step)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = frame_color

    return out


def _is_clear(
    occupied: np.ndarray,
    cells: list[tuple[int, int]],
    margin: int = 1,
) -> bool:
    h, w = map(int, occupied.shape)
    for r, c in cells:
        for rr in range(r - margin, r + margin + 1):
            for cc in range(c - margin, c + margin + 1):
                if 0 <= rr < h and 0 <= cc < w and bool(occupied[rr, cc]):
                    return False
    return True


def _frame_pixels_and_rays(
    H: int,
    W: int,
    h: int,
    w: int,
    orientation: str,
    n_markers: int,
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    if h < 3 or w < 3 or H < h or W < w:
        return None

    r0 = rng.randint(0, H - h)
    c0 = rng.randint(0, W - w)

    if orientation == "top" and rng.random() < 0.75 and H - h >= 1:
        r0 = max(r0, 1)
    if orientation == "bottom" and rng.random() < 0.75 and H - h >= 1:
        r0 = min(r0, H - h - 1)
    if orientation == "left" and rng.random() < 0.75 and W - w >= 1:
        c0 = max(c0, 1)
    if orientation == "right" and rng.random() < 0.75 and W - w >= 1:
        c0 = min(c0, W - w - 1)

    r1 = r0 + h - 1
    c1 = c0 + w - 1

    frame_cells: list[tuple[int, int]] = []
    for cc in range(c0, c1 + 1):
        frame_cells.append((r0, cc))
        frame_cells.append((r1, cc))
    for rr in range(r0 + 1, r1):
        frame_cells.append((rr, c0))
        frame_cells.append((rr, c1))

    if orientation == "top":
        row = r0 + 1
        candidates = list(range(c0 + 1, c1))
        if n_markers > len(candidates):
            return None
        cols = sorted(rng.sample(candidates, n_markers))
        marker_cells = [(row, cc) for cc in cols]
        dr, dc = -1, 0
        max_len = r0
    elif orientation == "bottom":
        row = r1 - 1
        candidates = list(range(c0 + 1, c1))
        if n_markers > len(candidates):
            return None
        cols = sorted(rng.sample(candidates, n_markers))
        marker_cells = [(row, cc) for cc in cols]
        dr, dc = 1, 0
        max_len = H - 1 - r1
    elif orientation == "left":
        col = c0 + 1
        candidates = list(range(r0 + 1, r1))
        if n_markers > len(candidates):
            return None
        rows = sorted(rng.sample(candidates, n_markers))
        marker_cells = [(rr, col) for rr in rows]
        dr, dc = 0, -1
        max_len = c0
    else:
        col = c1 - 1
        candidates = list(range(r0 + 1, r1))
        if n_markers > len(candidates):
            return None
        rows = sorted(rng.sample(candidates, n_markers))
        marker_cells = [(rr, col) for rr in rows]
        dr, dc = 0, 1
        max_len = W - 1 - c1

    if max_len < 1:
        return None

    ray_cells: list[tuple[int, int]] = []
    for mr, mc in marker_cells:
        for step in range(1, n_markers + 1):
            rr = int(mr + dr * step)
            cc = int(mc + dc * step)
            if 0 <= rr < H and 0 <= cc < W:
                ray_cells.append((rr, cc))

    if not ray_cells:
        return None

    all_cells = frame_cells + marker_cells + ray_cells
    return (r0, c0, r1, c1), marker_cells, all_cells


def generate_input() -> np.ndarray:
    rng = random.Random()

    GRID_H_RANGE = (12, 30)
    GRID_W_RANGE = (12, 30)
    FRAME_NUM_RANGE = (1, 3)
    MARKER_COLOR = 4
    MAX_TRIES = 1200
    MAX_FRAME_PLACEMENT_TRIES = 120
    PALETTE = tuple(range(10))

    for _ in range(MAX_TRIES):
        H = rng.randint(*GRID_H_RANGE)
        W = rng.randint(*GRID_W_RANGE)
        bg = rng.choice(PALETTE)
        marker = int(MARKER_COLOR if MARKER_COLOR != bg else rng.choice([c for c in PALETTE if c != bg]))

        available_colors = [c for c in PALETTE if c not in (bg, marker)]
        if len(available_colors) < FRAME_NUM_RANGE[0]:
            continue

        n_frames = rng.randint(*FRAME_NUM_RANGE)
        n_frames = min(n_frames, len(available_colors))
        frame_colors = rng.sample(available_colors, n_frames)

        grid = np.full((H, W), bg, dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        orientations = ("top", "bottom", "left", "right")
        placed = 0

        for fi in range(n_frames):
            frame_color = frame_colors[fi]
            frame_placed = False

            for _ in range(MAX_FRAME_PLACEMENT_TRIES):
                orientation = rng.choice(orientations)
                marker_count = rng.randint(1, 4)

                if orientation in ("top", "bottom"):
                    if W < 6 or H < 6:
                        continue
                    inner_w = rng.randint(max(4, marker_count + 1), min(8, W - 2))
                    inner_h = rng.randint(3, min(8, H - 2))
                else:
                    if W < 6 or H < 6:
                        continue
                    inner_h = rng.randint(max(4, marker_count + 1), min(8, H - 2))
                    inner_w = rng.randint(3, min(8, W - 2))

                if inner_w < 3 or inner_h < 3:
                    continue

                h = inner_h + 2
                w = inner_w + 2
                if h > H or w > W or h > H - 1 or w > W - 1:
                    continue

                proposal = _frame_pixels_and_rays(H, W, h, w, orientation, marker_count, rng)
                if proposal is None:
                    continue

                _bbox, marker_cells, all_cells = proposal
                r0, c0, r1, c1 = _bbox
                if not _is_clear(occupied, all_cells, margin=0):
                    continue

                for cc in range(c0, c1 + 1):
                    grid[r0, cc] = frame_color
                    grid[r1, cc] = frame_color
                for rr in range(r0 + 1, r1):
                    grid[rr, c0] = frame_color
                    grid[rr, c1] = frame_color
                for mr, mc in marker_cells:
                    grid[mr, mc] = marker

                for rr, cc in all_cells:
                    occupied[rr, cc] = True

                frame_placed = True
                placed += 1
                break

            if not frame_placed:
                break

        if placed != n_frames:
            continue

        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("generate_input: failed to produce a valid task instance")
