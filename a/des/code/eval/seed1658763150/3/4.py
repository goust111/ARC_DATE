import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
BAND_COLORS = (8, 7, 6)
PIECE_COLORS = (1, 2, 3, 4, 9)
DIRS = {
    "down": (1, 0),
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _detect_bands(g: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    row_bands: dict[int, int] = {}
    col_bands: dict[int, int] = {}
    for r in range(g.shape[0]):
        vals = set(map(int, g[r, :]))
        if len(vals) == 1:
            color = next(iter(vals))
            if color not in (BG, GREY):
                row_bands[r] = color
    for c in range(g.shape[1]):
        vals = set(map(int, g[:, c]))
        if len(vals) == 1:
            color = next(iter(vals))
            if color not in (BG, GREY):
                col_bands[c] = color
    return row_bands, col_bands


def _frame_notch(g: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    block = g[r0 : r1 + 1, c0 : c1 + 1]
    missing = block != GREY
    if not missing.any():
        return None

    h, w = missing.shape
    edge_starts = []
    for r in range(h):
        for c in range(w):
            if missing[r, c] and (r == 0 or c == 0 or r == h - 1 or c == w - 1):
                edge_starts.append((r, c))
    if not edge_starts:
        return None

    seen = set(edge_starts)
    q = deque(edge_starts)
    notch_local: set[tuple[int, int]] = set()
    while q:
        r, c = q.popleft()
        notch_local.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and missing[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    if not (1 <= len(notch_local) <= 8):
        return None

    side_counts = {
        "top": sum(r == 0 for r, _ in notch_local),
        "bottom": sum(r == h - 1 for r, _ in notch_local),
        "left": sum(c == 0 for _, c in notch_local),
        "right": sum(c == w - 1 for _, c in notch_local),
    }
    side = max(side_counts, key=side_counts.get)
    if side_counts[side] == 0:
        return None

    cells = {(r0 + r, c0 + c) for r, c in notch_local}
    return {"cells": cells, "shape": _norm(cells), "side": side}


def _candidate_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(g == GREY):
        frame = _frame_notch(g, comp)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda f: _bbox(f["cells"]))
    return frames


def _piece_components(g: np.ndarray, band_colors: set[int]) -> list[dict]:
    pieces: list[dict] = []
    colors = sorted(set(map(int, g.ravel())) - {BG, GREY} - set(band_colors))
    for color in colors:
        for comp in _components(g == color):
            cells = set(comp)
            pieces.append({"color": color, "cells": cells, "shape": _norm(cells)})
    return pieces


def _movement(piece_cells: set[tuple[int, int]], notch_cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    pr0, pc0, _, _ = _bbox(piece_cells)
    nr0, nc0, _, _ = _bbox(notch_cells)
    dr = nr0 - pr0
    dc = nc0 - pc0
    if dr == 0 and dc == 0:
        return None
    if dr != 0 and dc != 0:
        return None
    return dr, dc


def _band_color_after(
    r: int,
    c: int,
    dr: int,
    dc: int,
    target_cells: set[tuple[int, int]],
    row_bands: dict[int, int],
    col_bands: dict[int, int],
    piece_color: int,
) -> int:
    tr0, tc0, tr1, tc1 = _bbox(target_cells)
    if dr > 0:
        candidates = [rr for rr in row_bands if r < rr < tr0]
        return row_bands[max(candidates)] if candidates else piece_color
    if dr < 0:
        candidates = [rr for rr in row_bands if tr1 < rr < r]
        return row_bands[min(candidates)] if candidates else piece_color
    if dc > 0:
        candidates = [cc for cc in col_bands if c < cc < tc0]
        return col_bands[max(candidates)] if candidates else piece_color
    candidates = [cc for cc in col_bands if tc1 < cc < c]
    return col_bands[min(candidates)] if candidates else piece_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the same-shaped notch cut from a grey block.
    Cells swept before the notch become a trail colored by the last dye band still
    ahead on that straight path; dye bands themselves remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    row_bands, col_bands = _detect_bands(g)
    band_colors = set(row_bands.values()) | set(col_bands.values())
    frames = _candidate_frames(g)
    pieces = _piece_components(g, band_colors)

    out = g.copy()
    used_pieces: set[int] = set()
    band_cells = {(r, c) for r in row_bands for c in range(g.shape[1])}
    band_cells |= {(r, c) for c in col_bands for r in range(g.shape[0])}

    for frame in frames:
        match_idx = None
        move = None
        for idx, piece in enumerate(pieces):
            if idx in used_pieces or piece["shape"] != frame["shape"]:
                continue
            delta = _movement(piece["cells"], frame["cells"])
            if delta is None:
                continue
            dr, dc = delta
            side = frame["side"]
            if (side == "top" and dr <= 0) or (side == "bottom" and dr >= 0):
                continue
            if (side == "left" and dc <= 0) or (side == "right" and dc >= 0):
                continue
            match_idx = idx
            move = delta
            break

        if match_idx is None or move is None:
            continue

        piece = pieces[match_idx]
        used_pieces.add(match_idx)
        dr_total, dc_total = move
        steps = abs(dr_total + dc_total)
        step_r = 0 if dr_total == 0 else (1 if dr_total > 0 else -1)
        step_c = 0 if dc_total == 0 else (1 if dc_total > 0 else -1)

        for r, c in piece["cells"]:
            out[r, c] = BG

        for step in range(1, steps):
            for r, c in piece["cells"]:
                rr = r + step_r * step
                cc = c + step_c * step
                if (rr, cc) in band_cells:
                    continue
                if int(g[rr, cc]) == BG:
                    out[rr, cc] = _band_color_after(
                        rr,
                        cc,
                        step_r,
                        step_c,
                        frame["cells"],
                        row_bands,
                        col_bands,
                        int(piece["color"]),
                    )

        for r, c in frame["cells"]:
            out[r, c] = int(piece["color"])

    return out


def _rotate_shape(cells: set[tuple[int, int]], side: str) -> set[tuple[int, int]]:
    if side == "top":
        return cells
    if side == "bottom":
        return {(-r, c) for r, c in cells}
    if side == "left":
        return {(c, r) for r, c in cells}
    if side == "right":
        return {(c, -r) for r, c in cells}
    raise ValueError(side)


def _normalized_from_anchor(cells: set[tuple[int, int]], side: str) -> set[tuple[int, int]]:
    rotated = _rotate_shape(cells, side)
    min_r = min(r for r, _ in rotated)
    min_c = min(c for _, c in rotated)
    return {(r - min_r, c - min_c) for r, c in rotated}


def _template(side: str) -> set[tuple[int, int]]:
    base_shapes = [
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0), (2, 0)},
        {(0, 0), (0, 1), (1, 1)},
        {(0, 0), (0, 1), (0, 2), (1, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2)},
    ]
    return _normalized_from_anchor(random.choice(base_shapes), side)


def _side_delta(side: str) -> tuple[int, int]:
    return {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}[side]


def _rand_between(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _place_frame_and_piece() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    side = random.choice(("top", "bottom", "left", "right"))
    fh = random.randint(5, 8)
    fw = random.randint(5, 8)
    dist = random.randint(5, 8)
    notch = _template(side)

    nr0, nc0, nr1, nc1 = _bbox(notch)
    notch_h = nr1 - nr0 + 1
    notch_w = nc1 - nc0 + 1
    if notch_h >= fh or notch_w >= fw:
        return None

    if side in ("top", "bottom"):
        start_c = random.randint(1, fw - notch_w - 1)
        start_r = 0 if side == "top" else fh - notch_h
    else:
        start_r = random.randint(1, fh - notch_h - 1)
        start_c = 0 if side == "left" else fw - notch_w

    local_notch = {(start_r + r, start_c + c) for r, c in notch}
    if side == "top":
        fr = _rand_between(dist + 1, h - fh - 1)
        fc = _rand_between(1, w - fw - 1)
    elif side == "bottom":
        fr = _rand_between(1, h - fh - dist - 2)
        fc = _rand_between(1, w - fw - 1)
    elif side == "left":
        fr = _rand_between(1, h - fh - 1)
        fc = _rand_between(dist + 1, w - fw - 1)
    else:
        fr = _rand_between(1, h - fh - 1)
        fc = _rand_between(1, w - fw - dist - 2)
    if fr is None or fc is None:
        return None

    target = {(fr + r, fc + c) for r, c in local_notch}
    drow, dcol = _side_delta(side)
    piece = {(r + drow * dist, c + dcol * dist) for r, c in target}

    if not all(0 <= r < h and 0 <= c < w for r, c in piece):
        return None

    grid = np.full((h, w), BG, dtype=int)
    for r in range(fr, fr + fh):
        for c in range(fc, fc + fw):
            grid[r, c] = GREY
    for r, c in target:
        grid[r, c] = BG

    band_positions = list(range(2, dist - 1))
    if not band_positions:
        return None
    band_count = random.randint(1, min(3, len(band_positions)))
    band_offsets = sorted(random.sample(band_positions, band_count))
    band_palette = random.sample(BAND_COLORS, band_count)
    if side == "top":
        for offset, color in zip(band_offsets, band_palette):
            grid[min(r for r, _ in piece) + offset, :] = color
    elif side == "bottom":
        for offset, color in zip(band_offsets, band_palette):
            grid[max(r for r, _ in piece) - offset, :] = color
    elif side == "left":
        for offset, color in zip(band_offsets, band_palette):
            grid[:, min(c for _, c in piece) + offset] = color
    else:
        for offset, color in zip(band_offsets, band_palette):
            grid[:, max(c for _, c in piece) - offset] = color

    piece_color = random.choice(PIECE_COLORS)
    for r, c in piece:
        if int(grid[r, c]) != BG:
            return None
        grid[r, c] = piece_color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _place_frame_and_piece()
        if grid is None:
            continue
        out = main(grid)
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate seed1658763150/3/4 input")
