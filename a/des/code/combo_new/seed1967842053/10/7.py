import numpy as np


GREY = 5
BLACK = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            comps.append((cells, (min(rows), max(rows) + 1, min(cols), max(cols) + 1)))
    return comps


def _find_canvas(grid):
    candidates = []
    for cells, (r0, r1, c0, c1) in _components(grid != BLACK):
        sub = grid[r0:r1, c0:c1]
        if len(cells) == sub.size and np.all(sub != BLACK):
            all_grey = bool(np.all(sub == GREY))
            candidates.append((all_grey, c0, -sub.size, r0, r1, c0, c1))
    if not candidates:
        raise ValueError("no solid canvas found")
    _, _, _, r0, r1, c0, c1 = min(candidates)
    return r0, r1, c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Extract canvas colors through black holes in the grey templates.

    Input structure:
    A black background contains one solid non-black rectangular canvas on the
    left and several grey rectangular templates on the right.  Each template is
    the same size as the canvas, is separated from the other objects by black
    background, and contains black holes inside the grey material.

    Rule:
    Find the canvas rectangle and each same-sized grey template.  Align every
    template with the canvas by top-left corner: grey cells block the view and
    become black in the output, while black cells in the template copy the
    canvas color at the matching relative position.  Stack the results in the
    templates' top-to-bottom order with one black separator row between them.

    Key insight:
    The grey stamps are inverse masks: the black holes, not the grey material,
    reveal the aligned canvas pixels.
    """
    grid = np.asarray(grid)
    cr0, cr1, cc0, cc1 = _find_canvas(grid)
    canvas = grid[cr0:cr1, cc0:cc1]
    h, w = canvas.shape

    stamps = []
    for _, (r0, r1, c0, c1) in _components(grid == GREY):
        if r1 - r0 != h or c1 - c0 != w or c0 < cc1:
            continue
        sub = grid[r0:r1, c0:c1]
        if np.all((sub == BLACK) | (sub == GREY)) and np.any(sub == BLACK):
            stamps.append((r0, c0, sub))

    stamps.sort(key=lambda item: (item[0], item[1]))
    out_h = len(stamps) * h + max(0, len(stamps) - 1)
    out = np.zeros((out_h, w), dtype=grid.dtype)
    for i, (_, _, stamp) in enumerate(stamps):
        start = i * (h + 1)
        out[start:start + h] = np.where(stamp == BLACK, canvas, BLACK)
    return out


def _make_canvas(h, w, rng):
    colors = np.array([1, 2, 3, 4, 6, 7, 8, 9])
    k = int(rng.integers(3, 7))
    palette = rng.choice(colors, size=k, replace=False)
    rr, cc = np.indices((h, w))
    style = int(rng.integers(4))
    if style == 0:
        idx = (rr + 2 * cc + int(rng.integers(k))) % k
    elif style == 1:
        bh = int(rng.integers(1, 3))
        bw = int(rng.integers(1, 3))
        idx = (rr // bh + cc // bw + int(rng.integers(k))) % k
    elif style == 2:
        idx = (rr * rr + 2 * cc + int(rng.integers(k))) % k
    else:
        idx = rng.integers(0, k, size=(h, w))
    return palette[idx].astype(int)


def _make_holes(h, w, rng):
    holes = np.zeros((h, w), dtype=bool)
    style = int(rng.integers(5))
    ih, iw = h - 2, w - 2

    if style == 0:
        max_count = max(2, min(ih * iw // 4, 7))
        count = int(rng.integers(2, max_count + 1))
        spots = rng.choice(ih * iw, size=count, replace=False)
        for spot in spots:
            holes[1 + spot // iw, 1 + spot % iw] = True
    elif style == 1:
        horizontal = bool(rng.integers(2))
        if horizontal:
            r = int(rng.integers(1, h - 1))
            length = int(rng.integers(2, max(3, w - 1)))
            c0 = int(rng.integers(1, w - length))
            holes[r, c0:c0 + length] = True
        else:
            c = int(rng.integers(1, w - 1))
            length = int(rng.integers(2, max(3, h - 1)))
            r0 = int(rng.integers(1, h - length))
            holes[r0:r0 + length, c] = True
    elif style == 2:
        length = min(ih, iw)
        reverse = bool(rng.integers(2))
        for i in range(length):
            r = 1 + i
            c = w - 2 - i if reverse else 1 + i
            holes[r, c] = True
    elif style == 3:
        bh = int(rng.integers(1, max(2, ih // 2 + 1)))
        bw = int(rng.integers(1, max(2, iw // 2 + 1)))
        r0 = int(rng.integers(1, h - bh))
        c0 = int(rng.integers(1, w - bw))
        holes[r0:r0 + bh, c0:c0 + bw] = True
    else:
        r = int(rng.integers(1, h - 1))
        c = int(rng.integers(1, w - 1))
        arm_r = int(rng.integers(1, max(2, ih // 2 + 1)))
        arm_c = int(rng.integers(1, max(2, iw // 2 + 1)))
        holes[max(1, r - arm_r):min(h - 1, r + arm_r + 1), c] = True
        holes[r, max(1, c - arm_c):min(w - 1, c + arm_c + 1)] = True

    return holes


def _make_stamp(h, w, rng):
    stamp = np.full((h, w), GREY, dtype=int)
    stamp[_make_holes(h, w, rng)] = BLACK
    return stamp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    while True:
        h = int(rng.integers(5, 9))
        w = int(rng.integers(5, 11))
        max_n = min(4, (31 // (h + 1)))
        n = int(rng.integers(1, max_n + 1))
        gaps = [int(rng.integers(1, 4)) for _ in range(n - 1)]
        stack_h = n * h + sum(gaps)
        if stack_h <= 30:
            break

    canvas_left = int(rng.integers(0, 3))
    canvas_top = int(rng.integers(0, 31 - h))
    stamp_min_left = canvas_left + w + int(rng.integers(2, 6))
    stamp_max_left = 30 - w
    base_left = int(rng.integers(stamp_min_left, stamp_max_left + 1))

    stamp_lefts = []
    for _ in range(n):
        left = base_left + int(rng.integers(-2, 3))
        left = max(canvas_left + w + 1, min(left, 30 - w))
        stamp_lefts.append(left)

    stamp_top = int(rng.integers(0, 31 - stack_h))
    stamp_tops = []
    cursor = stamp_top
    for i in range(n):
        stamp_tops.append(cursor)
        if i < n - 1:
            cursor += h + gaps[i]

    min_h = max(canvas_top + h, max(stamp_tops) + h)
    min_w = max(canvas_left + w, max(stamp_lefts) + w)
    grid_h = min_h + int(rng.integers(0, 31 - min_h))
    grid_w = min_w + int(rng.integers(0, 31 - min_w))
    grid = np.zeros((grid_h, grid_w), dtype=int)

    grid[canvas_top:canvas_top + h, canvas_left:canvas_left + w] = _make_canvas(h, w, rng)
    for top, left in zip(stamp_tops, stamp_lefts):
        grid[top:top + h, left:left + w] = _make_stamp(h, w, rng)
    return grid
