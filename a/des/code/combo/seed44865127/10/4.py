import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    mask = grid != bg
    candidates: list[tuple[int, int, int, int]] = []
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 1 or height < 4 or height % 2 != 0:
            continue
        if c0 not in (0, w - 1):
            continue
        if len(pts) != height:
            continue
        candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected exactly one edge palette column"
    r0, c0, r1, _ = candidates[0]
    mapping: dict[int, int] = {}
    for r in range(r0, r1 + 1, 2):
        mapping[int(grid[r, c0])] = int(grid[r + 1, c0])

    pal_mask = np.zeros_like(mask, dtype=bool)
    pal_mask[r0 : r1 + 1, c0] = True
    return mapping, pal_mask


def _filled_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        box = mask[r0 : r1 + 1, c0 : c1 + 1]
        if bool(box.all()) and len(pts) == box.size:
            rects.append((r0, c0, r1, c1))
    return rects


def _longest_run_color(block: np.ndarray) -> int:
    best_color = int(block[0, 0])
    best_len = 0
    for row in block:
        start = 0
        while start < len(row):
            end = start + 1
            while end < len(row) and int(row[end]) == int(row[start]):
                end += 1
            run_len = end - start
            if run_len > best_len:
                best_len = run_len
                best_color = int(row[start])
            start = end
    return best_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each noisy solid rectangle using the fill color paired with the
    rectangle's longest horizontal-run color in the edge palette column.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mapping, pal_mask = _find_palette(g, bg)

    rect_mask = (g != bg) & (~pal_mask)
    rects = _filled_rectangles(rect_mask)
    assert rects, "expected at least one solid rectangle"

    out = g.copy()
    for r0, c0, r1, c1 in rects:
        key = _longest_run_color(g[r0 : r1 + 1, c0 : c1 + 1])
        assert key in mapping, "rectangle key color must appear in the palette"
        out[r0 : r1 + 1, c0 : c1 + 1] = mapping[key]
    return out


GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
RECT_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (3, 6)
RECT_W_RANGE = (4, 8)
MAX_RETRIES = 400
PATTERN_RETRIES = 120


def _max_horizontal_run(block: np.ndarray) -> tuple[int, int]:
    best_len = 0
    best_color = int(block[0, 0])
    for row in block:
        start = 0
        while start < len(row):
            end = start + 1
            while end < len(row) and int(row[end]) == int(row[start]):
                end += 1
            run_len = end - start
            if run_len > best_len:
                best_len = run_len
                best_color = int(row[start])
            start = end
    return best_color, best_len


def _limit_runs(row: np.ndarray, max_run: int, key: int, alt: int) -> np.ndarray:
    fixed = row.copy()
    run_color = int(fixed[0])
    run_len = 1
    for idx in range(1, len(fixed)):
        val = int(fixed[idx])
        if val == run_color:
            run_len += 1
            if run_len > max_run:
                fixed[idx] = alt if val == key else key
                run_color = int(fixed[idx])
                run_len = 1
        else:
            run_color = val
            run_len = 1
    return fixed


def _make_rectangle_pattern(
    h: int, w: int, key: int, alt: int, tricky: bool, rng: random.Random
) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        run_len = rng.randint(max(2, w // 2 + 1), w)
        anchor_row = rng.randrange(h)
        start = rng.randint(0, w - run_len)
        p_key = 0.78 if not tricky else 0.18

        block = np.empty((h, w), dtype=int)
        for r in range(h):
            if r == anchor_row:
                continue
            raw = np.array(
                [key if rng.random() < p_key else alt for _ in range(w)], dtype=int
            )
            block[r] = _limit_runs(raw, run_len - 1, key, alt)

        block[anchor_row] = alt
        block[anchor_row, start : start + run_len] = key

        longest_color, longest_len = _max_horizontal_run(block)
        if longest_color != key or longest_len != run_len:
            continue

        key_count = int(np.sum(block == key))
        alt_count = int(np.sum(block == alt))
        if tricky and key_count >= alt_count:
            continue
        if (not tricky) and key_count <= alt_count:
            continue
        if alt_count == 0:
            continue
        return block
    return None


def _rect_positions(
    occ: np.ndarray, rect_h: int, rect_w: int, c_lo: int, c_hi: int
) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - rect_h):
        for c0 in range(c_lo, c_hi - rect_w + 1):
            r_lo = max(0, r0 - 1)
            r_hi = min(h, r0 + rect_h + 1)
            c_left = max(0, c0 - 1)
            c_right = min(w, c0 + rect_w + 1)
            if occ[r_lo:r_hi, c_left:c_right].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_palette(
    grid: np.ndarray,
    occ: np.ndarray | None,
    keys: list[int],
    fills: list[int],
    side: str,
    r0: int,
) -> None:
    h, w = grid.shape
    c0 = 0 if side == "left" else w - 1
    for idx, (key, fill) in enumerate(zip(keys, fills)):
        grid[r0 + 2 * idx, c0] = key
        grid[r0 + 2 * idx + 1, c0] = fill

    if occ is not None:
        palette_h = 2 * len(keys)
        c_lo = 0 if side == "left" else w - 2
        c_hi = 2 if side == "left" else w
        occ[max(0, r0 - 1) : min(h, r0 + palette_h + 1), c_lo:c_hi] = True


def _bg_is_unique_majority(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    bg_count = counts.pop(bg, 0)
    return bg_count > max(counts.values(), default=-1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rect_count = rng.randint(*RECT_COUNT_RANGE)
    palette_size = rng.randint(rect_count, min(4, rect_count + 1))

    h = rng.randint(max(GRID_H_RANGE[0], 2 * palette_size + 1), GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    bg = rng.randrange(10)

    usable = [c for c in range(10) if c != bg]
    rng.shuffle(usable)
    keys = usable[:palette_size]
    fills = usable[palette_size : 2 * palette_size]
    mapping = dict(zip(keys, fills))

    grid = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    side = rng.choice(["left", "right"])
    palette_r0 = rng.randint(0, h - 2 * palette_size)
    _place_palette(grid, occ, keys, fills, side, palette_r0)
    _place_palette(out, None, keys, fills, side, palette_r0)

    used_keys = rng.sample(keys, rect_count)
    if rect_count == 1:
        tricky_flags = [bool(rng.randrange(2))]
    else:
        tricky_n = rng.randint(1, rect_count - 1)
        tricky_flags = [True] * tricky_n + [False] * (rect_count - tricky_n)
        rng.shuffle(tricky_flags)

    c_lo = 2 if side == "left" else 1
    c_hi = w - 2 if side == "right" else w - 1
    if c_hi - c_lo < RECT_W_RANGE[0]:
        return None

    for key, tricky in zip(used_keys, tricky_flags):
        rect_h = rng.randint(*RECT_H_RANGE)
        rect_w = rng.randint(*RECT_W_RANGE)
        if rect_h >= h - 1 or rect_w > c_hi - c_lo:
            return None

        positions = _rect_positions(occ, rect_h, rect_w, c_lo, c_hi)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)

        alt_pool = [c for c in range(10) if c not in {bg, key, mapping[key]}]
        alt = rng.choice(alt_pool)
        block = _make_rectangle_pattern(rect_h, rect_w, key, alt, tricky, rng)
        if block is None:
            return None

        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = block
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = mapping[key]
        occ[max(0, r0 - 1) : min(h, r0 + rect_h + 1), max(0, c0 - 1) : min(w, c0 + rect_w + 1)] = True

    if not _bg_is_unique_majority(grid, bg):
        return None
    try:
        if not np.array_equal(main(grid), out):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
