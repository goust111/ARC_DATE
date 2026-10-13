from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GAP_SIDES = ("top", "right", "bottom", "left")
ROT_BY_SIDE = {"top": 0, "right": 3, "bottom": 2, "left": 1}


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else DIRS4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _bottom_band(grid: np.ndarray) -> tuple[int, dict[int, int]]:
    h, w = grid.shape
    row = grid[-1]
    assert np.all(row != BG), "bottom band row must be non-background"

    band_rows = 1
    while band_rows < h and np.array_equal(grid[h - 1 - band_rows], row):
        band_rows += 1
    band_top = h - band_rows
    assert band_top > 0

    segments: list[int] = []
    c = 0
    while c < w:
        color = int(row[c])
        assert color != BG
        c2 = c + 1
        while c2 < w and int(row[c2]) == color:
            c2 += 1
        segments.append(color)
        c = c2

    assert len(segments) >= 2
    mapping = {segments[i]: segments[(i + 1) % len(segments)] for i in range(len(segments))}
    return band_top, mapping


def _frame_info(coords: list[tuple[int, int]], color: int) -> dict[str, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    pts = set(coords)
    for r, c in pts:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    side_arrays = {
        "top": [((r0, c) in pts) for c in range(c0, c1 + 1)],
        "bottom": [((r1, c) in pts) for c in range(c0, c1 + 1)],
        "left": [((r, c0) in pts) for r in range(r0, r1 + 1)],
        "right": [((r, c1) in pts) for r in range(r0, r1 + 1)],
    }
    missing = {
        side: [i for i, present in enumerate(arr) if not present]
        for side, arr in side_arrays.items()
    }

    gap_side: str | None = None
    gap_len = 0
    for side, miss in missing.items():
        if not miss:
            continue
        if len(miss) > 3 or miss != list(range(miss[0], miss[-1] + 1)):
            return None
        arr_len = len(side_arrays[side])
        if miss[0] == 0 or miss[-1] == arr_len - 1:
            return None
        if any(missing[other] for other in side_arrays if other != side):
            return None
        gap_side = side
        gap_len = len(miss)

    if gap_side is None:
        return None

    expected = 2 * (h + w) - 4 - gap_len
    if len(pts) != expected:
        return None

    return {
        "color": color,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "gap_side": gap_side,
    }


def _find_frames(grid: np.ndarray, band_top: int) -> list[dict[str, int]]:
    work = grid[:band_top]
    frames: list[dict[str, int]] = []
    for color in sorted({int(v) for v in work.ravel() if int(v) != BG}):
        comps = _components(work == color, conn8=False)
        for comp in comps:
            info = _frame_info(comp, color)
            if info is not None:
                frames.append(info)
    assert frames, "expected at least one frame"
    return frames


def _seed_mask(grid: np.ndarray, frame: dict[str, int]) -> np.ndarray:
    inner = grid[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]]
    fg = inner != BG
    comps = _components(fg, conn8=True)
    assert len(comps) == 1, "each frame interior must contain exactly one seed component"
    comp = comps[0]

    h, w = inner.shape
    for r, c in comp:
        assert 0 < r < h - 1 and 0 < c < w - 1, "seed must not touch the frame"

    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp:
        mask[r - r0, c - c0] = True
    return mask


def _line_positions(limit: int, size: int, base: int, step: int, bidir: bool) -> list[int]:
    assert 0 <= base <= limit - size
    vals = [base]
    cur = base + step * size
    while 0 <= cur <= limit - size:
        vals.append(cur)
        cur += step * size
    if bidir:
        cur = base - step * size
        while 0 <= cur <= limit - size:
            vals.append(cur)
            cur -= step * size
    return sorted(set(vals))


def _stamped_cover(inner_h: int, inner_w: int, seed: np.ndarray, gap_side: str) -> np.ndarray:
    rot = np.rot90(seed, ROT_BY_SIDE[gap_side])
    sh, sw = rot.shape
    assert sh <= inner_h and sw <= inner_w

    if gap_side == "top":
        row_positions = _line_positions(inner_h, sh, inner_h - sh, -1, bidir=False)
        col_positions = _line_positions(inner_w, sw, (inner_w - sw) // 2, 1, bidir=True)
    elif gap_side == "bottom":
        row_positions = _line_positions(inner_h, sh, 0, 1, bidir=False)
        col_positions = _line_positions(inner_w, sw, (inner_w - sw) // 2, 1, bidir=True)
    elif gap_side == "left":
        row_positions = _line_positions(inner_h, sh, (inner_h - sh) // 2, 1, bidir=True)
        col_positions = _line_positions(inner_w, sw, inner_w - sw, -1, bidir=False)
    else:
        row_positions = _line_positions(inner_h, sh, (inner_h - sh) // 2, 1, bidir=True)
        col_positions = _line_positions(inner_w, sw, 0, 1, bidir=False)

    cover = np.zeros((inner_h, inner_w), dtype=bool)
    for r0 in row_positions:
        for c0 in col_positions:
            cover[r0 : r0 + sh, c0 : c0 + sw] |= rot
    return cover


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    band_top, palette = _bottom_band(g)
    frames = _find_frames(g, band_top)

    out = g.copy()
    for frame in frames:
        fill_color = palette[frame["color"]]
        seed = _seed_mask(g, frame)
        inner_h = frame["r1"] - frame["r0"] - 1
        inner_w = frame["c1"] - frame["c0"] - 1
        cover = _stamped_cover(inner_h, inner_w, seed, frame["gap_side"])
        inner = np.full((inner_h, inner_w), fill_color, dtype=int)
        inner[cover] = BG
        out[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]] = inner
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each frame's interior seed by the border-gap direction, then fill the remaining interior from the bottom palette band.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular frames appear in the upper area; each frame uses a single color and has one contiguous gap on exactly one side.
    - Inside each frame is one small connected seed pattern of another color, separated from the frame.
    - A horizontal multi-segment color band occupies the bottom edge; each segment is a solid color block, and the band encodes a cyclic right-neighbor mapping.

    Rule:
    - Detect the side of each frame's gap.
    - Rotate the seed mask inside that frame according to the gap direction: top -> no rotation, right -> 90 degrees clockwise, bottom -> 180 degrees, left -> 90 degrees counterclockwise.
    - Treat the rotated seed as a stamp. Starting from the center of the side opposite the gap, tile that stamp across the frame interior without overlap and without crossing the border.
    - Stamp-covered cells become black background, while every other interior cell becomes the color of the segment immediately to the right of the frame color in the bottom band, with wraparound from the last segment to the first.

    Key insight:
    - The frame gap controls two different things at once: the seed rotation and the phase of the stamp tiling.
    - The fill color does not come from the seed or the frame itself; it is read indirectly from the cyclic neighbor relation in the bottom palette band.
    """
    return _solve(grid)


# --------------- generator ---------------

FRAME_COUNT_CHOICES = (2, 3, 4)
BAND_HEIGHT_CHOICES = (1, 2)
MIN_HW_BY_FRAMES = {2: (16, 18), 3: (20, 22), 4: (23, 24)}
MAX_SIDE = 30
FRAME_PADDING = 1
FRAME_PLACE_TRIES = 150
GEN_RETRIES = 400


def _random_seed_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        target = rng.randint(2, h * w - 1)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            r, c = rng.choice(tuple(cells))
            nbrs = []
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nbrs.append((nr, nc))
            cells.add(rng.choice(nbrs))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 and mask.shape[1] < 2:
            continue

        signatures = set()
        for k in range(4):
            rot = np.rot90(mask, k)
            signatures.add((rot.shape, tuple(bool(v) for v in rot.ravel())))
        if len(signatures) < 4:
            continue
        return mask
    raise AssertionError("failed to generate a non-symmetric seed")


def _random_widths(total: int, count: int, min_width: int, rng: random.Random) -> list[int]:
    assert total >= count * min_width
    extra = total - count * min_width
    widths = [min_width] * count
    for _ in range(extra):
        widths[rng.randrange(count)] += 1
    rng.shuffle(widths)
    if len(set(widths)) == 1 and extra > 0:
        i = rng.randrange(count)
        j = (i + 1) % count
        if widths[j] > min_width:
            widths[i] += 1
            widths[j] -= 1
    return widths


def _draw_frame(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, color: int, side: str, gap_start: int, gap_len: int) -> None:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif side == "bottom":
        grid[r1, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_len, c0] = BG
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_len, c1] = BG


def _place_seed_with_rng(grid: np.ndarray, frame: dict[str, int], mask: np.ndarray, color: int, rng: random.Random) -> None:
    inner_h = frame["r1"] - frame["r0"] - 1
    inner_w = frame["c1"] - frame["c0"] - 1
    sh, sw = mask.shape
    seed_r = frame["r0"] + 1 + rng.randint(1, inner_h - sh - 1)
    seed_c = frame["c0"] + 1 + rng.randint(1, inner_w - sw - 1)
    for rr in range(sh):
        for cc in range(sw):
            if mask[rr, cc]:
                grid[seed_r + rr, seed_c + cc] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    band_h = rng.choice(BAND_HEIGHT_CHOICES)
    min_h, min_w = MIN_HW_BY_FRAMES[frame_count]
    h = rng.randint(min_h, MAX_SIDE)
    w = rng.randint(min_w, MAX_SIDE)
    if h - band_h < 8:
        return None

    frame_colors = rng.sample(range(1, 10), frame_count)
    band_order = frame_colors[:]
    rng.shuffle(band_order)
    widths = _random_widths(w, frame_count, 2, rng)
    palette_map = {band_order[i]: band_order[(i + 1) % frame_count] for i in range(frame_count)}

    grid = np.zeros((h, w), dtype=int)
    cursor = 0
    for color, width in zip(band_order, widths):
        grid[h - band_h :, cursor : cursor + width] = color
        cursor += width

    directions = list(rng.sample(GAP_SIDES, frame_count))
    top_h = h - band_h
    occupied = np.zeros((top_h, w), dtype=bool)
    specs: list[dict[str, int]] = []

    size_bonus = {2: 6, 3: 4, 4: 3}[frame_count]
    for frame_color, gap_side in zip(frame_colors, directions):
        placed = False
        for _ in range(FRAME_PLACE_TRIES):
            seed = _random_seed_mask(rng)
            seed_h, seed_w = seed.shape
            seed_max = max(seed_h, seed_w)
            inner_min = seed_max + 2
            inner_h_hi = min(inner_min + size_bonus, 10)
            inner_w_hi = min(inner_min + size_bonus + 1, 11)
            if inner_min > inner_h_hi or inner_min > inner_w_hi:
                continue
            inner_h = rng.randint(inner_min, inner_h_hi)
            inner_w = rng.randint(inner_min, inner_w_hi)
            outer_h = inner_h + 2
            outer_w = inner_w + 2

            if top_h < outer_h or w < outer_w:
                continue

            r0 = rng.randint(0, top_h - outer_h)
            c0 = rng.randint(0, w - outer_w)
            rr0 = max(0, r0 - FRAME_PADDING)
            rr1 = min(top_h, r0 + outer_h + FRAME_PADDING)
            cc0 = max(0, c0 - FRAME_PADDING)
            cc1 = min(w, c0 + outer_w + FRAME_PADDING)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            side_len = outer_w if gap_side in ("top", "bottom") else outer_h
            gap_len = rng.randint(1, min(3, side_len - 2))
            gap_start = rng.randint(1, side_len - gap_len - 1)
            seed_color = rng.choice([c for c in range(1, 10) if c != frame_color])

            frame = {
                "color": frame_color,
                "r0": r0,
                "c0": c0,
                "r1": r0 + outer_h - 1,
                "c1": c0 + outer_w - 1,
                "gap_side": gap_side,
            }
            _draw_frame(grid, r0, c0, outer_h, outer_w, frame_color, gap_side, gap_start, gap_len)
            _place_seed_with_rng(grid, frame, seed, seed_color, rng)
            occupied[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
            specs.append(frame)
            placed = True
            break
        if not placed:
            return None

    try:
        out = _solve(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    for frame in specs:
        inner = out[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]]
        fill_color = palette_map[frame["color"]]
        vals = {int(v) for v in inner.ravel()}
        if vals - {BG, fill_color}:
            return None
        if BG not in vals or fill_color not in vals:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("generate_input failed after retry budget")
