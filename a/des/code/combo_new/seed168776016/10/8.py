import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points):
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _frame_border_cells(r0: int, c0: int, r1: int, c1: int):
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _find_receivers(grid: np.ndarray, bg: int, grey: int):
    receivers = []
    for comp in _components(grid == bg):
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        fr0, fc0, fr1, fc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
        if fr0 < 0 or fc0 < 0 or fr1 >= grid.shape[0] or fc1 >= grid.shape[1]:
            continue
        if all(int(grid[r, c]) == grey for r, c in _frame_border_cells(fr0, fc0, fr1, fc1)):
            receivers.append(
                {
                    "inner_bbox": (r0, c0, r1, c1),
                    "frame_bbox": (fr0, fc0, fr1, fc1),
                }
            )
    receivers.sort(key=lambda x: x["frame_bbox"])
    return receivers


def _attachment_candidates(grid: np.ndarray, grey: int, frame_bbox):
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    candidates = []

    if r0 > 0:
        for c in range(c0, c1 + 1):
            if int(grid[r0 - 1, c]) == grey:
                candidates.append(((r0, c), (r0 - 1, c)))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            if int(grid[r1 + 1, c]) == grey:
                candidates.append(((r1, c), (r1 + 1, c)))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            if int(grid[r, c0 - 1]) == grey:
                candidates.append(((r, c0), (r, c0 - 1)))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            if int(grid[r, c1 + 1]) == grey:
                candidates.append(((r, c1), (r, c1 + 1)))

    uniq = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _trace_signal(grid: np.ndarray, bg: int, grey: int, contact, first_path):
    h, w = grid.shape
    path = [contact, first_path]
    prev = contact
    cur = first_path
    visited = {contact, first_path}

    while True:
        grey_next = []
        source_next = []
        for nr, nc in _neighbors4(cur[0], cur[1], h, w):
            if (nr, nc) == prev:
                continue
            val = int(grid[nr, nc])
            if val == grey:
                grey_next.append((nr, nc))
            elif val != bg:
                source_next.append((nr, nc))

        if source_next:
            assert len(source_next) == 1 and not grey_next, "path endpoint must touch exactly one source"
            source = source_next[0]
            path.append(source)
            break

        assert len(grey_next) == 1, f"path must continue uniquely, found {len(grey_next)} options"
        nxt = grey_next[0]
        assert nxt not in visited, "signal path must not loop"
        visited.add(nxt)
        path.append(nxt)
        prev, cur = cur, nxt

    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = sum(d1 != d0 for d0, d1 in zip(dirs, dirs[1:]))
    color = int(grid[source[0], source[1]])
    assert color not in {bg, grey}, "source must be a colored endpoint"
    return color, turns


def _resolve_signal(grid: np.ndarray, bg: int, grey: int, frame_bbox):
    valid = []
    for contact, first_path in _attachment_candidates(grid, grey, frame_bbox):
        try:
            color, turns = _trace_signal(grid, bg, grey, contact, first_path)
        except AssertionError:
            continue
        valid.append((contact, first_path, color, turns))
    assert len(valid) == 1, f"expected one valid signal path, found {len(valid)}"
    return valid[0]


def _fill_pattern(out: np.ndarray, inner_bbox, color: int, turns: int):
    r0, c0, r1, c1 = inner_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if turns == 0:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        return

    if turns == 1:
        split = (h + 1) // 2
        out[r0 : r0 + split, c0 : c1 + 1] = color
        return

    if turns == 2:
        for rr, cc in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            out[rr, cc] = color
        return

    center_rows = [r0 + h // 2] if h % 2 else [r0 + h // 2 - 1, r0 + h // 2]
    center_cols = [c0 + w // 2] if w % 2 else [c0 + w // 2 - 1, c0 + w // 2]
    for rr in center_rows:
        for cc in center_cols:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color through its grey wire and fill the linked receiver according to the wire's turn count.

    Input structure:
    - The background is black.
    - Several hollow rectangular receivers are drawn with one-pixel-thick grey borders, and their interiors are black.
    - Each receiver is connected to exactly one single colored source pixel by a one-pixel-wide grey orthogonal path.
    - Different receiver-wire-source groups are disjoint, and each wire touches its receiver border at one point.

    Rule:
    - Find every enclosed black rectangle whose surrounding ring is grey; this is a receiver interior.
    - From the receiver's unique external grey attachment, follow the grey path to its colored source and record the source color and the number of 90-degree turns in the path.
    - Fill the receiver interior with that color using one of four patterns: all cells for 0 turns, the top half for 1 turn, only the four interior corners for 2 turns, and only the center cell or center block for 3 or more turns.
    - Leave the grey frame, grey wire, and source pixel unchanged.

    Key insight:
    - The wire carries two independent signals at once: the source color identifies the fill color, while the wire geometry encodes the interior fill pattern through its turn count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    non_bg = g[g != bg]
    if non_bg.size == 0:
        return g.copy()
    grey = _most_common_color(non_bg)

    receivers = _find_receivers(g, bg, grey)
    out = g.copy()
    for receiver in receivers:
        _, _, color, turns = _resolve_signal(g, bg, grey, receiver["frame_bbox"])
        _fill_pattern(out, receiver["inner_bbox"], color, turns)
    return out


def _make_group(turn_class: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        inner_h = int(rng.integers(2, 6))
        inner_w = int(rng.integers(2, 6))
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        attach_row = int(rng.integers(1, outer_h - 1))

        if turn_class >= 3:
            turns = int(rng.choice([3, 4]))
        else:
            turns = int(turn_class)

        lengths = []
        for i in range(turns + 1):
            if turn_class >= 3:
                lo, hi = 1, 3
            else:
                lo, hi = 2, 4
            if i == turns:
                lo = max(lo, 2)
            lengths.append(int(rng.integers(lo, hi + 1)))

        path = [(0, 0)]
        r = 0
        c = 0
        for i, length in enumerate(lengths):
            if i % 2 == 0:
                dr, dc = 0, -1
            else:
                dr = int(rng.choice([-1, 1]))
                dc = 0
            for _ in range(length):
                r += dr
                c += dc
                path.append((r, c))

        source = path[-1]
        grey_path = set(path[:-1])

        frame_r0 = -attach_row
        frame_c0 = 0
        frame_r1 = frame_r0 + outer_h - 1
        frame_c1 = outer_w - 1
        frame = set(_frame_border_cells(frame_r0, frame_c0, frame_r1, frame_c1))

        all_pts = list(frame | grey_path | {source})
        min_r = min(r0 for r0, _ in all_pts)
        max_r = max(r0 for r0, _ in all_pts)
        min_c = min(c0 for _, c0 in all_pts)
        max_c = max(c0 for _, c0 in all_pts)
        arr = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)

        for rr, cc in frame:
            arr[rr - min_r, cc - min_c] = GREY
        for rr, cc in grey_path:
            arr[rr - min_r, cc - min_c] = GREY
        arr[source[0] - min_r, source[1] - min_c] = color

        k = int(rng.integers(0, 4))
        arr = np.rot90(arr, k)
        receivers = _find_receivers(arr, BG, GREY)
        if len(receivers) != 1:
            continue
        try:
            _resolve_signal(arr, BG, GREY, receivers[0]["frame_bbox"])
        except AssertionError:
            continue
        return arr
    raise RuntimeError("failed to construct a valid receiver group")


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int):
    h, w = src.shape
    region = dst[top : top + h, left : left + w]
    assert np.all((region == BG) | (src == BG)), "objects must not overlap"
    mask = src != BG
    region[mask] = src[mask]


def _gen_attempt(rng: np.random.Generator):
    colors = list(rng.choice(COLOR_POOL, size=4, replace=False))
    try:
        groups = [_make_group(turn_class, color, rng) for turn_class, color in zip((0, 1, 2, 3), colors)]
    except RuntimeError:
        return None
    rng.shuffle(groups)

    top_a, top_b, bot_a, bot_b = groups
    hgap_top = int(rng.integers(1, 3))
    hgap_bot = int(rng.integers(1, 3))
    vgap = int(rng.integers(1, 3))

    row1_h = max(top_a.shape[0], top_b.shape[0])
    row2_h = max(bot_a.shape[0], bot_b.shape[0])
    row1_w = top_a.shape[1] + hgap_top + top_b.shape[1]
    row2_w = bot_a.shape[1] + hgap_bot + bot_b.shape[1]
    base_h = row1_h + vgap + row2_h
    base_w = max(row1_w, row2_w)

    if base_h > 30 or base_w > 30:
        return None

    grid_h = int(rng.integers(base_h, 31))
    grid_w = int(rng.integers(base_w, 31))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    row1_y = int(rng.integers(0, grid_h - base_h + 1))
    row2_y = row1_y + row1_h + vgap
    row1_x = int(rng.integers(0, grid_w - row1_w + 1))
    row2_x = int(rng.integers(0, grid_w - row2_w + 1))

    top_a_y = row1_y + int(rng.integers(0, row1_h - top_a.shape[0] + 1))
    top_b_y = row1_y + int(rng.integers(0, row1_h - top_b.shape[0] + 1))
    bot_a_y = row2_y + int(rng.integers(0, row2_h - bot_a.shape[0] + 1))
    bot_b_y = row2_y + int(rng.integers(0, row2_h - bot_b.shape[0] + 1))

    _paste(grid, top_a, top_a_y, row1_x)
    _paste(grid, top_b, top_b_y, row1_x + top_a.shape[1] + hgap_top)
    _paste(grid, bot_a, bot_a_y, row2_x)
    _paste(grid, bot_b, bot_b_y, row2_x + bot_a.shape[1] + hgap_bot)

    receivers = _find_receivers(grid, BG, GREY)
    if len(receivers) != 4:
        return None
    for receiver in receivers:
        try:
            _resolve_signal(grid, BG, GREY, receiver["frame_bbox"])
        except AssertionError:
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
