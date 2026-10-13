from __future__ import annotations

import random

import numpy as np


Rect = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _prefix(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    out[1:, 1:] = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return out


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _border_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = _rect_sum(ps, r0, c0, r0, c1)
    if r1 > r0:
        total += _rect_sum(ps, r1, c0, r1, c1)
    if r1 - r0 >= 2:
        total += _rect_sum(ps, r0 + 1, c0, r1 - 1, c0)
        if c1 > c0:
            total += _rect_sum(ps, r0 + 1, c1, r1 - 1, c1)
    return total


def _frame_candidates(grid: np.ndarray, bg: int, color: int) -> list[tuple[str, Rect]]:
    h, w = grid.shape
    color_mask = grid == color
    row_idx = np.flatnonzero(color_mask.any(axis=1))
    col_idx = np.flatnonzero(color_mask.any(axis=0))
    if row_idx.size < 2 or col_idx.size < 2:
        return []
    bg_mask = grid == bg
    rows, cols = np.indices(grid.shape)
    parity0 = (rows + cols) % 2 == 0
    parity1 = ~parity0

    ps_color = _prefix(color_mask)
    ps_bg = _prefix(bg_mask)
    ps_parity0 = _prefix(parity0)
    ps_parity1 = _prefix(parity1)
    ps_color_p0 = _prefix(color_mask & parity0)
    ps_color_p1 = _prefix(color_mask & parity1)
    ps_bg_p0 = _prefix(bg_mask & parity0)
    ps_bg_p1 = _prefix(bg_mask & parity1)

    out: list[tuple[str, Rect]] = []
    for i, r0 in enumerate(row_idx[:-1]):
        for r1 in row_idx[i + 1 :]:
            if r1 - r0 < 2:
                continue
            for j, c0 in enumerate(col_idx[:-1]):
                for c1 in col_idx[j + 1 :]:
                    if c1 - c0 < 2:
                        continue
                    if _rect_sum(ps_color, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue

                    perim = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
                    border_color = _border_sum(ps_color, r0, c0, r1, c1)
                    if border_color == perim:
                        out.append(("solid", (r0, c0, r1, c1)))
                        continue

                    need_p0 = _border_sum(ps_parity0, r0, c0, r1, c1)
                    need_p1 = _border_sum(ps_parity1, r0, c0, r1, c1)
                    have_color_p0 = _border_sum(ps_color_p0, r0, c0, r1, c1)
                    have_color_p1 = _border_sum(ps_color_p1, r0, c0, r1, c1)
                    have_bg_p0 = _border_sum(ps_bg_p0, r0, c0, r1, c1)
                    have_bg_p1 = _border_sum(ps_bg_p1, r0, c0, r1, c1)

                    if have_color_p0 == need_p0 and have_bg_p1 == need_p1:
                        out.append(("dashed", (r0, c0, r1, c1)))
                    elif have_color_p1 == need_p1 and have_bg_p0 == need_p0:
                        out.append(("dashed", (r0, c0, r1, c1)))

    return out


def _centered_transfer(src: np.ndarray, dst: np.ndarray) -> None:
    sh, sw = src.shape
    dh, dw = dst.shape
    dr0 = max(0, (dh - sh) // 2)
    dc0 = max(0, (dw - sw) // 2)
    sr0 = max(0, (sh - dh) // 2)
    sc0 = max(0, (sw - dw) // 2)
    oh = min(sh, dh)
    ow = min(sw, dw)
    dst[dr0 : dr0 + oh, dc0 : dc0 + ow] = src[sr0 : sr0 + oh, sc0 : sc0 + ow]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each dashed frame's interior into the solid frame of the same color, then erase the dashed frame.

    Input structure:
    - The dominant color is the background.
    - Some one-pixel-thick rectangular frames are solid.
    - Some one-pixel-thick rectangular frames are dashed, with border pixels alternating between
      the frame color and the background.
    - Each frame color appears exactly twice: once as a dashed source frame and once as a solid target frame.
    - Dashed frames contain colored content; solid-frame interiors are empty background.

    Rule:
    - Distinguish dashed versus solid frames from their border continuity.
    - Copy the complete dashed-frame interior pattern into the same-colored solid frame interior.
    - When source and target interiors differ in size, align their centers.
    - Erase every dashed frame together with everything inside it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]

    pairs: list[tuple[int, Rect, Rect]] = []
    for color in colors:
        candidates = _frame_candidates(g, bg, color)
        solids = [rect for kind, rect in candidates if kind == "solid"]
        dashed = [rect for kind, rect in candidates if kind == "dashed"]
        if not solids and not dashed:
            continue
        assert len(solids) == 1 and len(dashed) == 1, f"ambiguous frames for color {color}"
        pairs.append((color, dashed[0], solids[0]))

    assert pairs, "expected at least one dashed/solid color pair"

    out = g.copy()
    for _color, src_rect, dst_rect in pairs:
        sr0, sc0, sr1, sc1 = src_rect
        dr0, dc0, dr1, dc1 = dst_rect
        src_inner = g[sr0 + 1 : sr1, sc0 + 1 : sc1].copy()
        dst_inner = out[dr0 + 1 : dr1, dc0 + 1 : dc1]
        _centered_transfer(src_inner, dst_inner)
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg

    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 30
PAIR_CHOICES = (1, 2, 3)
INNER_MIN = 3
INNER_MAX = 7
PAD = 1
PLACEMENT_PAD = 2
RETRIES = 400

CONTENT_MODES = ("rects", "noise", "walk")


def _rand_content(h: int, w: int, bg: int, colors: list[int]) -> np.ndarray:
    arr = np.full((h, w), bg, dtype=int)
    mode = random.choice(CONTENT_MODES)

    if mode == "rects":
        for _ in range(random.randint(1, 3)):
            rh = random.randint(1, max(1, h - 1))
            rw = random.randint(1, max(1, w - 1))
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            arr[r0 : r0 + rh, c0 : c0 + rw] = random.choice(colors)
    elif mode == "noise":
        count = random.randint(1, max(1, (h * w) // 2))
        picks = random.sample(range(h * w), count)
        for idx in picks:
            r, c = divmod(idx, w)
            arr[r, c] = random.choice(colors)
    else:
        steps = random.randint(max(2, min(h, w)), max(3, h * w // 2))
        r = random.randrange(h)
        c = random.randrange(w)
        for _ in range(steps):
            arr[r, c] = random.choice(colors)
            dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            r = min(h - 1, max(0, r + dr))
            c = min(w - 1, max(0, c + dc))

    if not np.any(arr != bg):
        arr[random.randrange(h), random.randrange(w)] = random.choice(colors)
    return arr


def _make_solid_patch(h: int, w: int, frame_color: int, bg: int) -> np.ndarray:
    patch = np.full((h + 2, w + 2), frame_color, dtype=int)
    patch[1:-1, 1:-1] = bg
    return patch


def _make_dashed_patch(inner: np.ndarray, frame_color: int, bg: int, phase: int) -> np.ndarray:
    h, w = inner.shape
    patch = np.full((h + 2, w + 2), bg, dtype=int)
    patch[1:-1, 1:-1] = inner
    hh, ww = patch.shape
    for r in range(hh):
        for c in range(ww):
            if r not in (0, hh - 1) and c not in (0, ww - 1):
                continue
            if (r + c + phase) % 2 == 0:
                patch[r, c] = frame_color
    return patch


def _draw_patch(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, margin: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 * margin > h or pw + 2 * margin > w:
        return False

    positions = [
        (r, c)
        for r in range(margin, h - ph - margin + 1)
        for c in range(margin, w - pw - margin + 1)
    ]
    random.shuffle(positions)
    for r0, c0 in positions:
        rr0 = max(0, r0 - PLACEMENT_PAD)
        cc0 = max(0, c0 - PLACEMENT_PAD)
        rr1 = min(h, r0 + ph + PLACEMENT_PAD)
        cc1 = min(w, c0 + pw + PLACEMENT_PAD)
        if occ[rr0:rr1, cc0:cc1].any():
            continue
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        view[:, :] = patch
        occ[r0 : r0 + ph, c0 : c0 + pw] = True
        return True
    return False


def _generate_case() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randrange(10)
    pair_count = random.choice(PAIR_CHOICES)

    palette = [c for c in range(10) if c != bg]
    if len(palette) < pair_count + 2:
        return None
    frame_colors = random.sample(palette, pair_count)
    content_palette = [c for c in palette if c not in frame_colors]
    if not content_palette:
        return None

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    specs = []
    for frame_color in frame_colors:
        sh = random.randint(INNER_MIN, INNER_MAX)
        sw = random.randint(INNER_MIN, INNER_MAX)

        if random.random() < 0.5:
            dh, dw = sh, sw
        else:
            dh = sh + random.randint(0, 3)
            dw = sw + random.randint(0, 3)
            if dh == sh and dw == sw:
                if random.random() < 0.5:
                    dh += 1
                else:
                    dw += 1

        inner = _rand_content(sh, sw, bg, content_palette)
        dashed = _make_dashed_patch(inner, frame_color, bg, random.randint(0, 1))
        solid = _make_solid_patch(dh, dw, frame_color, bg)
        specs.append(dashed)
        specs.append(solid)

    specs.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    for patch in specs:
        if not _draw_patch(grid, occ, patch, PAD):
            return None

    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
