from __future__ import annotations

import random

import numpy as np


BG_COLOR = 8
OCCLUDER_COLORS = (9, 7, 6)
PATTERN_COLORS = (1, 2, 3, 4, 5)
GEN_RETRIES = 400


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _parse_horizontal_band(strip: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray] | None:
    if strip.shape[0] != 2:
        return None
    active_cols = np.flatnonzero(np.any(strip != bg, axis=0))
    if active_cols.size == 0:
        return None
    start = int(active_cols[0])
    end = int(active_cols[-1]) + 1
    if (end - start) % 2 != 0:
        return None

    mapping: dict[int, int] = {}
    for c in range(start, end, 2):
        block = strip[:, c : c + 2]
        if block.shape != (2, 2):
            return None
        if not (np.all(block[0] == block[0, 0]) and np.all(block[1] == block[1, 0])):
            return None
        occ = int(block[0, 0])
        src = int(block[1, 0])
        if occ == bg or src == bg or occ == src:
            return None
        if occ in mapping and mapping[occ] != src:
            return None
        mapping[occ] = src

    if not mapping:
        return None
    mask = np.zeros(strip.shape, dtype=bool)
    mask[:, start:end] = True
    return mapping, mask


def _parse_vertical_band(strip: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray] | None:
    if strip.shape[1] != 2:
        return None
    active_rows = np.flatnonzero(np.any(strip != bg, axis=1))
    if active_rows.size == 0:
        return None
    start = int(active_rows[0])
    end = int(active_rows[-1]) + 1
    if (end - start) % 2 != 0:
        return None

    mapping: dict[int, int] = {}
    for r in range(start, end, 2):
        block = strip[r : r + 2, :]
        if block.shape != (2, 2):
            return None
        if not (np.all(block[:, 0] == block[0, 0]) and np.all(block[:, 1] == block[0, 1])):
            return None
        occ = int(block[0, 0])
        src = int(block[0, 1])
        if occ == bg or src == bg or occ == src:
            return None
        if occ in mapping and mapping[occ] != src:
            return None
        mapping[occ] = src

    if not mapping:
        return None
    mask = np.zeros(strip.shape, dtype=bool)
    mask[start:end, :] = True
    return mapping, mask


def _detect_band(grid: np.ndarray, bg: int) -> tuple[str, dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[str, dict[int, int], np.ndarray]] = []

    if h >= 2:
        parsed = _parse_horizontal_band(grid[:2, :], bg)
        if parsed is not None:
            mapping, local_mask = parsed
            mask = np.zeros_like(grid, dtype=bool)
            mask[:2, :] = local_mask
            candidates.append(("top", mapping, mask))

        parsed = _parse_horizontal_band(grid[h - 2 :, :], bg)
        if parsed is not None:
            mapping, local_mask = parsed
            mask = np.zeros_like(grid, dtype=bool)
            mask[h - 2 :, :] = local_mask
            candidates.append(("bottom", mapping, mask))

    if w >= 2:
        parsed = _parse_vertical_band(grid[:, :2], bg)
        if parsed is not None:
            mapping, local_mask = parsed
            mask = np.zeros_like(grid, dtype=bool)
            mask[:, :2] = local_mask
            candidates.append(("left", mapping, mask))

        parsed = _parse_vertical_band(grid[:, w - 2 :], bg)
        if parsed is not None:
            mapping, local_mask = parsed
            mask = np.zeros_like(grid, dtype=bool)
            mask[:, w - 2 :] = local_mask
            candidates.append(("right", mapping, mask))

    assert candidates
    candidates.sort(key=lambda item: int(item[2].sum()), reverse=True)
    side, mapping, mask = candidates[0]
    if len(candidates) > 1:
        assert int(candidates[0][2].sum()) > int(candidates[1][2].sum())
    return side, mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a partially occluded 180-degree symmetric pattern and selectively
    recolor restored pixels using the edge mapping band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG_COLOR if np.any(g == BG_COLOR) else _majority_color(g)
    _side, mapping, band_mask = _detect_band(g, bg)

    pattern_mask = (g != bg) & ~band_mask
    coords = np.argwhere(pattern_mask)
    assert coords.size > 0
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))

    occluder_colors = np.array(sorted(mapping), dtype=int)
    occluded = np.isin(g, occluder_colors) & ~band_mask

    out = g.copy()
    for r, c in np.argwhere(occluded):
        rr = r0 + r1 - int(r)
        cc = c0 + c1 - int(c)
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
        assert not band_mask[rr, cc]
        src = int(g[rr, cc])
        occ = int(g[r, c])
        assert src != bg and src not in mapping
        out[r, c] = occ if src == int(mapping[occ]) else src

    return out


def _choose_block_size(length: int) -> int:
    options = [d for d in (3, 2) if length % d == 0 and length // d >= 3]
    if options and random.random() < 0.8:
        return random.choice(options)
    return 1


def _paint_rot_rect(arr: np.ndarray, r: int, c: int, h: int, w: int, color: int) -> None:
    arr[r : r + h, c : c + w] = int(color)
    rr = arr.shape[0] - r - h
    cc = arr.shape[1] - c - w
    arr[rr : rr + h, cc : cc + w] = int(color)


def _paint_orbit(arr: np.ndarray, r: int, c: int, color: int) -> None:
    rr = arr.shape[0] - 1 - r
    cc = arr.shape[1] - 1 - c
    arr[r, c] = int(color)
    arr[rr, cc] = int(color)


def _ensure_palette_visibility(arr: np.ndarray, palette: list[int]) -> None:
    for color in palette:
        if np.any(arr == int(color)):
            continue
        r = random.randrange(arr.shape[0])
        c = random.randrange(arr.shape[1])
        _paint_orbit(arr, r, c, color)


def _make_coarse_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    mode = random.choice(("orbits", "rings", "patches"))
    k = len(palette)

    if mode == "orbits":
        arr = np.empty((h, w), dtype=int)
        reps: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                rr = h - 1 - r
                cc = w - 1 - c
                if r > rr or (r == rr and c > cc):
                    continue
                reps.append((r, c))
        random.shuffle(reps)
        forced = palette.copy()
        random.shuffle(forced)
        for i, (r, c) in enumerate(reps):
            color = forced[i] if i < len(forced) else random.choice(palette)
            _paint_orbit(arr, r, c, color)
    elif mode == "rings":
        rows, cols = np.indices((h, w))
        dist = np.minimum(rows, h - 1 - rows) + np.minimum(cols, w - 1 - cols)
        arr = np.asarray(palette, dtype=int)[(dist + random.randrange(k)) % k]
    else:
        arr = np.full((h, w), random.choice(palette), dtype=int)
        for _ in range(random.randint(max(3, k), max(5, k + 2))):
            color = random.choice(palette)
            rh = random.randint(1, min(3, h))
            rw = random.randint(1, min(3, w))
            r = random.randint(0, h - rh)
            c = random.randint(0, w - rw)
            _paint_rot_rect(arr, r, c, rh, rw, color)

    for _ in range(random.randint(1, 4)):
        color = random.choice(palette)
        rh = random.randint(1, min(2, h))
        rw = random.randint(1, min(2, w))
        r = random.randint(0, h - rh)
        c = random.randint(0, w - rw)
        _paint_rot_rect(arr, r, c, rh, rw, color)

    _ensure_palette_visibility(arr, palette)
    return arr


def _make_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    block_h = _choose_block_size(h)
    block_w = _choose_block_size(w)
    coarse = _make_coarse_pattern(h // block_h, w // block_w, palette)
    pattern = np.kron(coarse, np.ones((block_h, block_w), dtype=int)).astype(int)
    return pattern[:h, :w]


def _valid_occlusion(mask: np.ndarray) -> bool:
    return not np.any(mask & mask[::-1, ::-1])


def _find_rect_for_color(
    pattern: np.ndarray,
    mapped_color: int,
    used_mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = pattern.shape
    sizes = [
        (rh, rw)
        for rh in range(2, min(5, h) + 1)
        for rw in range(2, min(5, w) + 1)
        if rh < h and rw < w
    ]
    random.shuffle(sizes)

    for _ in range(200):
        rh, rw = random.choice(sizes)
        r_lo = 1 if h - rh >= 2 else 0
        c_lo = 1 if w - rw >= 2 else 0
        r_hi = h - rh - 1 if h - rh >= 2 else h - rh
        c_hi = w - rw - 1 if w - rw >= 2 else w - rw
        r = random.randint(r_lo, r_hi)
        c = random.randint(c_lo, c_hi)

        rect_mask = np.zeros((h, w), dtype=bool)
        rect_mask[r : r + rh, c : c + rw] = True
        union = used_mask | rect_mask
        if np.any(used_mask & rect_mask):
            continue
        if not _valid_occlusion(union):
            continue

        colors = pattern[r : r + rh, c : c + rw]
        if not np.any(colors == int(mapped_color)):
            continue
        if not np.any(colors != int(mapped_color)):
            continue
        return r, c, rh, rw

    return None


def _layout_geometry(side: str, ph: int, pw: int) -> tuple[int, int, int, int]:
    parallel_a = random.randint(2, 4)
    parallel_b = max(1, min(5, parallel_a + random.randint(-1, 1)))
    near_gap = random.randint(1, 3)
    far_gap = max(1, min(5, near_gap + random.randint(-1, 1)))

    if side == "top":
        gh = 2 + near_gap + ph + far_gap
        gw = parallel_a + pw + parallel_b
        return gh, gw, 2 + near_gap, parallel_a
    if side == "bottom":
        gh = far_gap + ph + near_gap + 2
        gw = parallel_a + pw + parallel_b
        return gh, gw, far_gap, parallel_a
    if side == "left":
        gh = parallel_a + ph + parallel_b
        gw = 2 + near_gap + pw + far_gap
        return gh, gw, parallel_a, 2 + near_gap
    gh = parallel_a + ph + parallel_b
    gw = far_gap + pw + near_gap + 2
    return gh, gw, parallel_a, far_gap


def _paint_band(grid: np.ndarray, side: str, entries: list[tuple[int, int]]) -> None:
    h, w = grid.shape
    if side in ("top", "bottom"):
        run = 2 * len(entries)
        start = random.randint(1, w - run - 1)
        r0 = 0 if side == "top" else h - 2
        for i, (occ, src) in enumerate(entries):
            c0 = start + 2 * i
            grid[r0, c0 : c0 + 2] = int(occ)
            grid[r0 + 1, c0 : c0 + 2] = int(src)
        return

    run = 2 * len(entries)
    start = random.randint(1, h - run - 1)
    c0 = 0 if side == "left" else w - 2
    for i, (occ, src) in enumerate(entries):
        r0 = start + 2 * i
        grid[r0 : r0 + 2, c0] = int(occ)
        grid[r0 : r0 + 2, c0 + 1] = int(src)


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(("top", "bottom", "left", "right"))
    occluder_count = random.randint(1, 3)
    ph = random.randint(8, 14)
    pw = random.randint(8, 14)

    min_palette = max(3, occluder_count + 1)
    palette = random.sample(PATTERN_COLORS, random.randint(min_palette, len(PATTERN_COLORS)))
    pattern = _make_pattern(ph, pw, palette)

    mapped_colors = random.sample(palette, occluder_count)
    occluders = random.sample(OCCLUDER_COLORS, occluder_count)
    entries = list(zip(occluders, mapped_colors))
    random.shuffle(entries)

    used_mask = np.zeros((ph, pw), dtype=bool)
    rects: list[tuple[int, int, int, int, int, int]] = []
    for occ, src in entries:
        rect = _find_rect_for_color(pattern, src, used_mask)
        if rect is None:
            return None
        r, c, rh, rw = rect
        used_mask[r : r + rh, c : c + rw] = True
        rects.append((r, c, rh, rw, occ, src))

    gh, gw, pr0, pc0 = _layout_geometry(side, ph, pw)
    grid = np.full((gh, gw), BG_COLOR, dtype=int)
    grid[pr0 : pr0 + ph, pc0 : pc0 + pw] = pattern
    _paint_band(grid, side, entries)

    clean = grid.copy()
    inp = grid.copy()
    out = grid.copy()

    replaced_any = False
    kept_any = False
    for r, c, rh, rw, occ, src in rects:
        region = clean[pr0 + r : pr0 + r + rh, pc0 + c : pc0 + c + rw]
        inp[pr0 + r : pr0 + r + rh, pc0 + c : pc0 + c + rw] = int(occ)
        out_region = np.where(region == int(src), int(occ), region)
        out[pr0 + r : pr0 + r + rh, pc0 + c : pc0 + c + rw] = out_region
        replaced_any |= bool(np.any(region == int(src)))
        kept_any |= bool(np.any(region != int(src)))

    if not (replaced_any and kept_any):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 4")
