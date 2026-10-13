from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
WINDOW_SIZE = 4
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Interior cell offsets, then the corresponding horizontal and vertical ray directions.
CELL_SPECS = (
    ((1, 1), (0, -1), (-1, 0)),  # top-left
    ((1, 2), (0, 1), (-1, 0)),   # top-right
    ((2, 1), (0, -1), (1, 0)),   # bottom-left
    ((2, 2), (0, 1), (1, 0)),    # bottom-right
)

# --------------- solver ---------------


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_windows(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    windows: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _mask_components(grid == color):
            if len(comp) != 12:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 != WINDOW_SIZE or c1 - c0 + 1 != WINDOW_SIZE:
                continue
            ok = True
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    on_border = r in (r0, r1) or c in (c0, c1)
                    if on_border:
                        ok = ok and int(grid[r, c]) == color
                    else:
                        ok = ok and int(grid[r, c]) != color
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                windows.append((r0, c0, r1, c1, color))
    windows.sort()
    return windows


def _scan_ray(
    grid: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    rr = start_r + dr
    cc = start_c + dc
    dist = 1
    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value != BG:
            return rr, cc, value, dist
        rr += dr
        cc += dc
        dist += 1
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each missing 2x2 window cell with the nearer of its two directional paint buckets,
    then erase the bucket that was used.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1, frame_color in _find_windows(g):
        for (roff, coff), (hdr, hdc), (vdr, vdc) in CELL_SPECS:
            ir = r0 + roff
            ic = c0 + coff
            if int(g[ir, ic]) != BG:
                continue

            if hdc < 0:
                h_hit = _scan_ray(g, ir, c0, hdr, hdc)
            else:
                h_hit = _scan_ray(g, ir, c1, hdr, hdc)

            if vdr < 0:
                v_hit = _scan_ray(g, r0, ic, vdr, vdc)
            else:
                v_hit = _scan_ray(g, r1, ic, vdr, vdc)

            candidates: list[tuple[int, int, int, int]] = []
            for hit in (h_hit, v_hit):
                if hit is None:
                    continue
                rr, cc, value, dist = hit
                if value == frame_color:
                    continue
                candidates.append((dist, rr, cc, value))

            if not candidates:
                continue

            dist, br, bc, fill_color = min(candidates, key=lambda item: item[0])
            _ = dist
            out[ir, ic] = fill_color
            out[br, bc] = BG

    return out


# --------------- generator ---------------

GENERATE_ATTEMPTS = 300
WINDOWS_RANGE = (2, 5)
ZONE_H_RANGE = (10, 14)
ZONE_W_RANGE = (10, 14)
GAP_RANGE = (1, 4)
PAD_RANGE = (1, 4)


def _paint_window(grid: np.ndarray, r0: int, c0: int) -> None:
    grid[r0, c0 : c0 + WINDOW_SIZE] = GREY
    grid[r0 + WINDOW_SIZE - 1, c0 : c0 + WINDOW_SIZE] = GREY
    grid[r0 : r0 + WINDOW_SIZE, c0] = GREY
    grid[r0 : r0 + WINDOW_SIZE, c0 + WINDOW_SIZE - 1] = GREY


def _step(pos: tuple[int, int], direction: tuple[int, int], dist: int) -> tuple[int, int]:
    return pos[0] + direction[0] * dist, pos[1] + direction[1] * dist


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _far_enough(new_points: list[tuple[int, int]], existing: list[tuple[int, int]]) -> bool:
    for i, p in enumerate(new_points):
        for q in new_points[i + 1 :]:
            if _chebyshev(p, q) <= 1:
                return False
        for q in existing:
            if _chebyshev(p, q) <= 1:
                return False
    return True


def _ray_specs(
    zone_h: int,
    zone_w: int,
    r0: int,
    c0: int,
    cell_idx: int,
) -> tuple[tuple[int, int], tuple[int, int], int, tuple[int, int], tuple[int, int], int]:
    (roff, coff), hdir, vdir = CELL_SPECS[cell_idx]
    ir = r0 + roff
    ic = c0 + coff

    if hdir[1] < 0:
        h_start = (ir, c0)
        h_max = c0
    else:
        h_start = (ir, c0 + WINDOW_SIZE - 1)
        h_max = zone_w - (c0 + WINDOW_SIZE)

    if vdir[0] < 0:
        v_start = (r0, ic)
        v_max = r0
    else:
        v_start = (r0 + WINDOW_SIZE - 1, ic)
        v_max = zone_h - (r0 + WINDOW_SIZE)

    return h_start, hdir, h_max, v_start, vdir, v_max


def _pick_bucket_pair(
    rng: random.Random,
    zone_h: int,
    zone_w: int,
    r0: int,
    c0: int,
    cell_idx: int,
    existing: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    h_start, hdir, h_max, v_start, vdir, v_max = _ray_specs(zone_h, zone_w, r0, c0, cell_idx)
    combos: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for hdist in range(1, h_max + 1):
        h_pos = _step(h_start, hdir, hdist)
        for vdist in range(1, v_max + 1):
            if hdist == vdist:
                continue
            v_pos = _step(v_start, vdir, vdist)
            if not _far_enough([h_pos, v_pos], existing):
                continue
            winner = "h" if hdist < vdist else "v"
            combos.append((h_pos, v_pos, winner))

    if not combos:
        return None

    preferred = rng.choice(("h", "v"))
    filtered = [combo for combo in combos if combo[2] == preferred]
    pool = filtered if filtered else combos
    return rng.choice(pool)


def _random_color(rng: random.Random) -> int:
    return int(rng.choice(PALETTE))


def _distinct_pair(rng: random.Random) -> tuple[int, int]:
    a, b = rng.sample(PALETTE, 2)
    return int(a), int(b)


def _build_zone(rng: random.Random, zone_h: int, zone_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((zone_h, zone_w), dtype=int)

    r0 = rng.randint(3, zone_h - WINDOW_SIZE - 3)
    c0 = rng.randint(3, zone_w - WINDOW_SIZE - 3)
    _paint_window(inp, r0, c0)

    missing_count = rng.randint(1, 3)
    missing_ids = set(rng.sample(range(4), missing_count))
    bucket_positions: list[tuple[int, int]] = []
    fills: list[tuple[int, int, tuple[int, int], int]] = []

    for cell_idx, ((roff, coff), _, _) in enumerate(CELL_SPECS):
        ir = r0 + roff
        ic = c0 + coff
        if cell_idx not in missing_ids:
            inp[ir, ic] = _random_color(rng)

    order = list(missing_ids)
    rng.shuffle(order)
    for cell_idx in order:
        picked = _pick_bucket_pair(rng, zone_h, zone_w, r0, c0, cell_idx, bucket_positions)
        if picked is None:
            return None
        h_pos, v_pos, winner = picked
        h_color, v_color = _distinct_pair(rng)
        inp[h_pos] = h_color
        inp[v_pos] = v_color
        bucket_positions.extend([h_pos, v_pos])

        roff, coff = CELL_SPECS[cell_idx][0]
        ir = r0 + roff
        ic = c0 + coff
        if winner == "h":
            fills.append((ir, ic, h_pos, h_color))
        else:
            fills.append((ir, ic, v_pos, v_color))

    out = inp.copy()
    for ir, ic, used_pos, fill_color in fills:
        out[ir, ic] = fill_color
        out[used_pos] = BG

    return inp, out


def _layout_choices(n_windows: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for rows in range(1, 4):
        for cols in range(1, 4):
            if rows * cols >= n_windows:
                choices.append((rows, cols))
    return choices


def _assemble_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_windows = rng.randint(*WINDOWS_RANGE)
    rows, cols = rng.choice(_layout_choices(n_windows))

    row_heights = [rng.randint(*ZONE_H_RANGE) for _ in range(rows)]
    col_widths = [rng.randint(*ZONE_W_RANGE) for _ in range(cols)]
    row_gaps = [rng.randint(*GAP_RANGE) for _ in range(max(0, rows - 1))]
    col_gaps = [rng.randint(*GAP_RANGE) for _ in range(max(0, cols - 1))]
    top_pad = rng.randint(*PAD_RANGE)
    bottom_pad = rng.randint(*PAD_RANGE)
    left_pad = rng.randint(*PAD_RANGE)
    right_pad = rng.randint(*PAD_RANGE)

    row_starts: list[int] = []
    cur = top_pad
    for idx, height in enumerate(row_heights):
        row_starts.append(cur)
        cur += height
        if idx < len(row_gaps):
            cur += row_gaps[idx]
    grid_h = cur + bottom_pad

    col_starts: list[int] = []
    cur = left_pad
    for idx, width in enumerate(col_widths):
        col_starts.append(cur)
        cur += width
        if idx < len(col_gaps):
            cur += col_gaps[idx]
    grid_w = cur + right_pad

    inp = np.zeros((grid_h, grid_w), dtype=int)
    out = np.zeros((grid_h, grid_w), dtype=int)

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    active_slots = rng.sample(slots, n_windows)
    for r_idx, c_idx in active_slots:
        zone_h = row_heights[r_idx]
        zone_w = col_widths[c_idx]
        built = _build_zone(rng, zone_h, zone_w)
        if built is None:
            return None
        zone_inp, zone_out = built
        r0 = row_starts[r_idx]
        c0 = col_starts[c_idx]
        inp[r0 : r0 + zone_h, c0 : c0 + zone_w] = zone_inp
        out[r0 : r0 + zone_h, c0 : c0 + zone_w] = zone_out

    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_ATTEMPTS):
        built = _assemble_attempt(rng)
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_ATTEMPTS} attempts")
