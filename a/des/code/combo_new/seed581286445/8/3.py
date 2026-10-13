from __future__ import annotations

import random

import numpy as np


BG = 0
ORANGE = 7
MAROON = 9
SEED_PALETTE = (1, 2, 3, 4, 5, 6, 8)
PANEL_SIZES = (4, 6, 8, 10, 12, 14)
GENERATION_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for color in sorted({int(v) for v in grid.ravel()} - {bg}):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) != 1 or len(cols) != 1:
            continue
        mid_r = rows[0]
        mid_c = cols[0]
        if 0 < mid_r < h - 1 and 0 < mid_c < w - 1 and h == 2 * mid_r + 1 and w == 2 * mid_c + 1:
            candidates.append((int(color), int(mid_r), int(mid_c)))
    assert len(candidates) == 1, "expected one full-row/full-column divider cross"
    return candidates[0]


def _panel_offsets(mid_r: int, mid_c: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, mid_c + 1),
        "bl": (mid_r + 1, 0),
        "br": (mid_r + 1, mid_c + 1),
    }


def _extract_seed(panel: np.ndarray, bg: int, divider: int) -> tuple[int, int, int]:
    seed_rc = np.argwhere((panel != bg) & (panel != divider))
    assert seed_rc.shape[0] == 1, "each panel must contain exactly one seed pixel"
    r, c = map(int, seed_rc[0])
    return r, c, int(panel[r, c])


def _quadrant_dirs(local_r: int, local_c: int, panel_h: int, panel_w: int) -> tuple[int, int]:
    hdir = 1 if local_c < panel_w / 2 else -1
    vdir = 1 if local_r < panel_h / 2 else -1
    return hdir, vdir


def _l_points(local_r: int, local_c: int, hdir: int, vdir: int, panel_h: int, panel_w: int) -> list[tuple[int, int]]:
    pts = {(local_r, local_c)}
    for step in (1, 2):
        pts.add((local_r, local_c + hdir * step))
        pts.add((local_r + vdir * step, local_c))
    return [(r, c) for r, c in sorted(pts) if 0 <= r < panel_h and 0 <= c < panel_w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow quadrant-coded seed pixels into L shapes and recolor divider segments when neighboring shapes face each other.

    Input structure:
    - The background is black.
    - A single maroon full-width row and matching full-height column split the grid into four equal panels.
    - Each panel contains exactly one non-divider colored seed pixel.
    - The seed position lies in one of the panel's four quadrants relative to that panel's center.

    Rule:
    - In each panel, grow the seed into an L shape whose horizontal and vertical arms point away from the panel center:
      top-left seeds grow right and down, top-right seeds left and down, bottom-left seeds right and up, and bottom-right seeds left and up.
    - Each arm extends two cells from the seed, clipped to stay inside the panel, and uses the seed's own color.
    - For each left-right neighboring panel pair, if the left L points right and the right L points left, recolor the divider column segment between the two seed rows to orange.
    - For each top-bottom neighboring panel pair, if the top L points down and the bottom L points up, recolor the divider row segment between the two seed columns to orange.

    Key insight:
    - The seed's quadrant determines the local L orientation, and only pairs whose Ls point inward toward their shared divider cause an orange connector segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    divider, mid_r, mid_c = _find_divider(g, bg)
    panel_h, panel_w = mid_r, mid_c
    offsets = _panel_offsets(mid_r, mid_c)

    out = g.copy()
    info: dict[str, dict[str, int | tuple[int, int]]] = {}

    for name, (r0, c0) in offsets.items():
        panel = g[r0 : r0 + panel_h, c0 : c0 + panel_w]
        local_r, local_c, color = _extract_seed(panel, bg, divider)
        hdir, vdir = _quadrant_dirs(local_r, local_c, panel_h, panel_w)
        for pr, pc in _l_points(local_r, local_c, hdir, vdir, panel_h, panel_w):
            out[r0 + pr, c0 + pc] = color
        info[name] = {
            "seed_r": r0 + local_r,
            "seed_c": c0 + local_c,
            "hdir": hdir,
            "vdir": vdir,
        }

    if int(info["tl"]["hdir"]) == 1 and int(info["tr"]["hdir"]) == -1:
        r0 = int(info["tl"]["seed_r"])
        r1 = int(info["tr"]["seed_r"])
        out[min(r0, r1) : max(r0, r1) + 1, mid_c] = ORANGE
    if int(info["bl"]["hdir"]) == 1 and int(info["br"]["hdir"]) == -1:
        r0 = int(info["bl"]["seed_r"])
        r1 = int(info["br"]["seed_r"])
        out[min(r0, r1) : max(r0, r1) + 1, mid_c] = ORANGE
    if int(info["tl"]["vdir"]) == 1 and int(info["bl"]["vdir"]) == -1:
        c0 = int(info["tl"]["seed_c"])
        c1 = int(info["bl"]["seed_c"])
        out[mid_r, min(c0, c1) : max(c0, c1) + 1] = ORANGE
    if int(info["tr"]["vdir"]) == 1 and int(info["br"]["vdir"]) == -1:
        c0 = int(info["tr"]["seed_c"])
        c1 = int(info["br"]["seed_c"])
        out[mid_r, min(c0, c1) : max(c0, c1) + 1] = ORANGE

    return out


def _half_range(size: int, use_first_half: bool) -> list[int]:
    split = size // 2
    if use_first_half:
        return list(range(0, split))
    return list(range(split, size))


def _sample_pair(face: bool) -> tuple[bool, bool]:
    return (True, False) if face else (False, True)


def _place_seed(grid: np.ndarray, r0: int, c0: int, panel_h: int, panel_w: int, top_half: bool, left_half: bool, color: int) -> None:
    rows = _half_range(panel_h, top_half)
    cols = _half_range(panel_w, left_half)
    r = r0 + random.choice(rows)
    c = c0 + random.choice(cols)
    grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.choice(PANEL_SIZES)
    panel_w = random.choice(PANEL_SIZES)
    h = 2 * panel_h + 1
    w = 2 * panel_w + 1

    pair_faces = {
        "top_lr": bool(random.getrandbits(1)),
        "bottom_lr": bool(random.getrandbits(1)),
        "left_tb": bool(random.getrandbits(1)),
        "right_tb": bool(random.getrandbits(1)),
    }
    if len(set(pair_faces.values())) == 1:
        return None

    tl_left, tr_left = _sample_pair(pair_faces["top_lr"])
    bl_left, br_left = _sample_pair(pair_faces["bottom_lr"])
    tl_top, bl_top = _sample_pair(pair_faces["left_tb"])
    tr_top, br_top = _sample_pair(pair_faces["right_tb"])

    panel_bits = {
        "tl": (tl_top, tl_left),
        "tr": (tr_top, tr_left),
        "bl": (bl_top, bl_left),
        "br": (br_top, br_left),
    }
    colors = random.sample(SEED_PALETTE, 4)

    grid = np.full((h, w), BG, dtype=int)
    mid_r = panel_h
    mid_c = panel_w
    grid[mid_r, :] = MAROON
    grid[:, mid_c] = MAROON

    offsets = _panel_offsets(mid_r, mid_c)
    for name, color in zip(("tl", "tr", "bl", "br"), colors, strict=False):
        top_half, left_half = panel_bits[name]
        r0, c0 = offsets[name]
        _place_seed(grid, r0, c0, panel_h, panel_w, top_half, left_half, color)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == ORANGE) <= np.count_nonzero(grid == ORANGE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
