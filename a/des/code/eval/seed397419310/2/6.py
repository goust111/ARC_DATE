import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5

BORDER_COLOR_POOL = (1, 2, 3, 6, 7, 8, 9)

MIN_PAIRS = 1
MAX_PAIRS = 3
MIN_INTERIOR = 2
MAX_INTERIOR = 5
MIN_BRIDGE = 2
MAX_BRIDGE_EXTRA = 3
GRID_MIN = 14
GRID_MAX = 30
MAX_PATTERN_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _frame_from_component(coords: np.ndarray, color: int) -> dict | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None
    actual = {(int(r), int(c)) for r, c in coords}
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if actual != expected:
        return None
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
    }


def _extract_frame_pairs(grid: np.ndarray) -> dict[int, list[dict]]:
    frames_by_color: dict[int, list[dict]] = {}
    for color in map(int, np.unique(grid)):
        if color in {BG, YELLOW, GREY}:
            continue
        frames = []
        for comp in _components4(grid == color):
            frame = _frame_from_component(comp, color)
            if frame is not None:
                frames.append(frame)
        if len(frames) == 2:
            frames_by_color[color] = sorted(frames, key=lambda f: f["bbox"])
    assert frames_by_color, "expected at least one frame pair"
    return frames_by_color


def _find_bridge(grid: np.ndarray, color: int, a: dict, b: dict) -> dict:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    bridge_colors = np.array([GREY, YELLOW], dtype=int)
    options: list[dict] = []

    if ac1 < bc0:
        left, right = a, b
    elif bc1 < ac0:
        left, right = b, a
    else:
        left = right = None

    if left is not None:
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        c_start = lc1 + 1
        c_end = rc0 - 1
        if c_start <= c_end:
            for r in range(max(lr0, rr0), min(lr1, rr1) + 1):
                seg = grid[r, c_start : c_end + 1]
                if seg.size == 0:
                    continue
                if np.all(np.isin(seg, bridge_colors)):
                    options.append(
                        {
                            "orientation": "h",
                            "row": int(r),
                            "c_start": int(c_start),
                            "c_end": int(c_end),
                            "markers": int(np.sum(seg == YELLOW)),
                        }
                    )

    if ar1 < br0:
        top, bottom = a, b
    elif br1 < ar0:
        top, bottom = b, a
    else:
        top = bottom = None

    if top is not None:
        tr0, tc0, tr1, tc1 = top["bbox"]
        br0, bc0, br1, bc1 = bottom["bbox"]
        r_start = tr1 + 1
        r_end = br0 - 1
        if r_start <= r_end:
            for c in range(max(tc0, bc0), min(tc1, bc1) + 1):
                seg = grid[r_start : r_end + 1, c]
                if seg.size == 0:
                    continue
                if np.all(np.isin(seg, bridge_colors)):
                    options.append(
                        {
                            "orientation": "v",
                            "col": int(c),
                            "r_start": int(r_start),
                            "r_end": int(r_end),
                            "markers": int(np.sum(seg == YELLOW)),
                        }
                    )

    assert len(options) == 1, f"expected a unique bridge for color {color}"
    return options[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match same-colored framed regions, count yellow bridge markers, and copy the
    sender interior into the empty receiver after clockwise rotation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color, frames in _extract_frame_pairs(g).items():
        bridge = _find_bridge(g, color, frames[0], frames[1])
        interiors = []
        filled_flags = []
        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            interiors.append(interior)
            filled_flags.append(bool(np.any(interior != BG)))

        assert filled_flags.count(True) == 1, f"expected one sender and one receiver for color {color}"
        src_i = 0 if filled_flags[0] else 1
        dst_i = 1 - src_i

        rotated = np.rot90(interiors[src_i], -bridge["markers"])
        dst_r0, dst_c0, dst_r1, dst_c1 = frames[dst_i]["bbox"]
        dst_shape = (dst_r1 - dst_r0 - 1, dst_c1 - dst_c0 - 1)
        assert rotated.shape == dst_shape, f"rotated interior shape mismatch for color {color}"
        out[dst_r0 + 1 : dst_r1, dst_c0 + 1 : dst_c1] = rotated

    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def _grow_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    frontier = {seed}
    while int(mask.sum()) < area:
        candidates: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    candidates.add((int(nr), int(nc)))
        nr, nc = random.choice(tuple(candidates))
        mask[nr, nc] = True
        frontier.add((nr, nc))
    return mask


def _random_pattern(h: int, w: int, palette: list[int], k: int) -> np.ndarray:
    max_area = h * w - 1 if h * w > 2 else h * w
    for _ in range(MAX_PATTERN_ATTEMPTS):
        area = random.randint(2, max_area)
        mask = _grow_mask(h, w, area)
        pat = np.zeros((h, w), dtype=int)
        color_count = random.randint(1, min(3, len(palette), area))
        colors = random.sample(palette, color_count)

        cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(cells)
        for idx, (r, c) in enumerate(cells):
            if idx < color_count:
                pat[r, c] = colors[idx]
            else:
                pat[r, c] = random.choice(colors)

        empty_cells = [tuple(map(int, rc)) for rc in np.argwhere(~mask)]
        if empty_cells and random.random() < 0.35:
            extra_count = random.randint(1, min(len(empty_cells), max(1, (h * w) // 4)))
            for r, c in random.sample(empty_cells, extra_count):
                pat[r, c] = random.choice(colors)

        if k and np.array_equal(np.rot90(pat, -k), pat):
            continue
        return pat
    raise RuntimeError("failed to create a non-degenerate pattern")


def _make_pair(border_color: int, pattern_palette: list[int], k: int) -> tuple[np.ndarray, np.ndarray]:
    orientation = random.choice(("h", "v"))
    if k % 2 == 1:
        side = random.randint(MIN_INTERIOR, MAX_INTERIOR)
        inner_h = side
        inner_w = side
    else:
        inner_h = random.randint(MIN_INTERIOR, MAX_INTERIOR)
        inner_w = random.randint(MIN_INTERIOR, MAX_INTERIOR)

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    bridge_min = max(MIN_BRIDGE, k + 1 if k else MIN_BRIDGE)
    bridge_len = random.randint(bridge_min, bridge_min + MAX_BRIDGE_EXTRA)
    sender = _random_pattern(inner_h, inner_w, pattern_palette, k)
    rotated = np.rot90(sender, -k)
    assert rotated.shape == (inner_h, inner_w)

    if orientation == "h":
        patch_h = frame_h
        patch_w = frame_w * 2 + bridge_len
        inp = np.zeros((patch_h, patch_w), dtype=int)
        out = np.zeros_like(inp)
        left_c = 0
        right_c = frame_w + bridge_len
        _draw_frame(inp, 0, left_c, frame_h, frame_w, border_color)
        _draw_frame(inp, 0, right_c, frame_h, frame_w, border_color)
        _draw_frame(out, 0, left_c, frame_h, frame_w, border_color)
        _draw_frame(out, 0, right_c, frame_h, frame_w, border_color)

        bridge_row = random.randint(1, frame_h - 2)
        bridge_cols = list(range(frame_w, frame_w + bridge_len))
        inp[bridge_row, bridge_cols] = GREY
        out[bridge_row, bridge_cols] = GREY
        for col in random.sample(bridge_cols, k):
            inp[bridge_row, col] = YELLOW
            out[bridge_row, col] = YELLOW

        sender_on_left = bool(random.getrandbits(1))
        if sender_on_left:
            inp[1 : 1 + inner_h, 1 : 1 + inner_w] = sender
            out[1 : 1 + inner_h, 1 : 1 + inner_w] = sender
            out[1 : 1 + inner_h, right_c + 1 : right_c + 1 + inner_w] = rotated
        else:
            inp[1 : 1 + inner_h, right_c + 1 : right_c + 1 + inner_w] = sender
            out[1 : 1 + inner_h, right_c + 1 : right_c + 1 + inner_w] = sender
            out[1 : 1 + inner_h, 1 : 1 + inner_w] = rotated
        return inp, out

    patch_h = frame_h * 2 + bridge_len
    patch_w = frame_w
    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros_like(inp)
    top_r = 0
    bottom_r = frame_h + bridge_len
    _draw_frame(inp, top_r, 0, frame_h, frame_w, border_color)
    _draw_frame(inp, bottom_r, 0, frame_h, frame_w, border_color)
    _draw_frame(out, top_r, 0, frame_h, frame_w, border_color)
    _draw_frame(out, bottom_r, 0, frame_h, frame_w, border_color)

    bridge_col = random.randint(1, frame_w - 2)
    bridge_rows = list(range(frame_h, frame_h + bridge_len))
    inp[bridge_rows, bridge_col] = GREY
    out[bridge_rows, bridge_col] = GREY
    for row in random.sample(bridge_rows, k):
        inp[row, bridge_col] = YELLOW
        out[row, bridge_col] = YELLOW

    sender_on_top = bool(random.getrandbits(1))
    if sender_on_top:
        inp[1 : 1 + inner_h, 1 : 1 + inner_w] = sender
        out[1 : 1 + inner_h, 1 : 1 + inner_w] = sender
        out[bottom_r + 1 : bottom_r + 1 + inner_h, 1 : 1 + inner_w] = rotated
    else:
        inp[bottom_r + 1 : bottom_r + 1 + inner_h, 1 : 1 + inner_w] = sender
        out[bottom_r + 1 : bottom_r + 1 + inner_h, 1 : 1 + inner_w] = sender
        out[1 : 1 + inner_h, 1 : 1 + inner_w] = rotated
    return inp, out


def _positive_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _scatter(total: int, buckets: int) -> list[int]:
    values = [0] * buckets
    for _ in range(total):
        values[random.randrange(buckets)] += 1
    return values


def _assemble_scene(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n = len(shuffled)

    for _ in range(100):
        row_count = random.randint(1, min(3, n))
        counts = _positive_partition(n, row_count)
        rows: list[list[tuple[np.ndarray, np.ndarray]]] = []
        idx = 0
        for count in counts:
            row = shuffled[idx : idx + count]
            idx += count
            row = row[:]
            random.shuffle(row)
            rows.append(row)

        top = random.randint(1, 3)
        bottom = random.randint(1, 3)
        left = random.randint(1, 3)
        right = random.randint(1, 3)
        row_gaps = [random.randint(2, 4) for _ in range(row_count - 1)]

        row_heights: list[int] = []
        row_widths: list[int] = []
        col_gaps: list[list[int]] = []
        for row in rows:
            gaps = [random.randint(2, 4) for _ in range(len(row) - 1)]
            col_gaps.append(gaps)
            row_heights.append(max(int(inp.shape[0]) for inp, _ in row))
            row_widths.append(sum(int(inp.shape[1]) for inp, _ in row) + sum(gaps))

        min_h = top + bottom + sum(row_heights) + sum(row_gaps)
        min_w = left + right + max(row_widths)
        if min_h > GRID_MAX or min_w > GRID_MAX:
            continue

        h_lo = max(GRID_MIN, min_h)
        w_lo = max(GRID_MIN, min_w)
        h_hi = min(GRID_MAX, max(h_lo, min_h + 8))
        w_hi = min(GRID_MAX, max(w_lo, min_w + 10))
        if h_lo > h_hi or w_lo > w_hi:
            continue

        h = random.randint(h_lo, h_hi)
        w = random.randint(w_lo, w_hi)
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)

        extra_v = h - min_h
        v_slots = _scatter(extra_v, row_count + 1)
        y = top + v_slots[0]
        for row_idx, row in enumerate(rows):
            row_h = row_heights[row_idx]
            row_w = row_widths[row_idx]
            x = random.randint(left, w - right - row_w)
            cur_x = x
            for item_idx, (pair_inp, pair_out) in enumerate(row):
                ph, pw = pair_inp.shape
                py = y + random.randint(0, row_h - ph)
                inp[py : py + ph, cur_x : cur_x + pw] = pair_inp
                out[py : py + ph, cur_x : cur_x + pw] = pair_out
                cur_x += pw
                if item_idx < len(row) - 1:
                    cur_x += col_gaps[row_idx][item_idx]
            y += row_h
            if row_idx < row_count - 1:
                y += row_gaps[row_idx] + v_slots[row_idx + 1]

        return inp, out

    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(MIN_PAIRS, MAX_PAIRS)
    border_colors = random.sample(BORDER_COLOR_POOL, pair_count)
    pattern_palette = [c for c in range(10) if c not in {BG, YELLOW, GREY} and c not in border_colors]
    ks = random.sample([0, 1, 2, 3], pair_count)

    pairs = [_make_pair(border_color, pattern_palette, k) for border_color, k in zip(border_colors, ks)]
    scene = _assemble_scene(pairs)
    if scene is None:
        return None
    inp, expected = scene

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
