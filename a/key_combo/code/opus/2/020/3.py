import numpy as np


def _decode_direction(legend):
    cy, cx = 1, 1
    if legend.shape != (3, 3):
        return 0, 1
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        y, x = cy + dy, cx + dx
        if 0 <= y < 3 and 0 <= x < 3 and legend[y, x] == 4:
            return dy, dx
    return 0, 1


def _ordered_iter(h, w, drow, dcol):
    if drow == 0:
        rows = range(h)
        cols = range(w - 1, -1, -1) if dcol == 1 else range(w)
        return [(r, c) for r in rows for c in cols]
    cols = range(w)
    rows = range(h - 1, -1, -1) if drow == 1 else range(h)
    return [(r, c) for c in cols for r in rows]


def main(grid):
    """
    1) Split by separator, decode direction from legend.
    2) Move each missing template pixel to collision-stop position.
    3) Use fixed order depending on direction.
    4) Return transformed grid preserving fixed elements.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r] == 6)]
    if not sep_rows:
        return g
    sep = int(sep_rows[0])
    top = g[:sep]
    bottom = g[sep + 1 :]

    template = top[:5, :5]
    legend = top[:3, 6:9]
    drow, dcol = _decode_direction(legend)

    rows_nonzero = np.where(np.any(bottom != 0, axis=1))[0]
    if len(rows_nonzero) == 0:
        return g
    br0 = int(rows_nonzero[0])

    for sc in (0, 7, 14):
        if sc + 5 > bottom.shape[1] or br0 + 5 > bottom.shape[0]:
            continue
        copy = bottom[br0 : br0 + 5, sc : sc + 5].copy()
        miss = [(y, x) for y in range(5) for x in range(5)
                if template[y, x] != 0 and copy[y, x] == 0]
        if not miss:
            continue

        order = _ordered_iter(5, 5, drow, dcol)
        for my, mx in order:
            if (my, mx) not in miss:
                continue
            color = int(template[my, mx])
            yy, xx = my, mx
            while True:
                ny, nx = yy + drow, xx + dcol
                if not (0 <= ny < 5 and 0 <= nx < 5):
                    break
                if copy[ny, nx] != 0:
                    break
                yy, xx = ny, nx
            copy[yy, xx] = color
            copy[my, mx] = 0

        bottom[br0 : br0 + 5, sc : sc + 5] = copy

    g[sep + 1 : sep + 1 + bottom.shape[0], :bottom.shape[1]] = bottom
    return g


def _random_template(rng):
    cnt = int(rng.integers(8, 13))
    cells = {(2, 2)}
    attempts = 0
    while len(cells) < cnt and attempts < 1000:
        attempts += 1
        cy, cx = list(cells)[-1]
        steps = 0
        moved = False
        for _ in range(10):
            dr, dc = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            ny, nx = cy + dr, cx + dc
            if 0 <= ny < 5 and 0 <= nx < 5 and (ny, nx) not in cells:
                cells.add((ny, nx))
                cy, cx = ny, nx
                moved = True
                break
        if not moved:
            y = int(rng.integers(0, 5))
            x = int(rng.integers(0, 5))
            if (y, x) not in cells:
                cells.add((y, x))
                cy, cx = y, x

    t = np.zeros((5, 5), dtype=int)
    for y, x in cells:
        t[y, x] = int(rng.choice([1, 2]))
    return t


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    rng = np.random.default_rng(int(seed) * 1000003 + int(variant) * 7 + 11)
    g = np.zeros((15, 21), dtype=int)
    template = _random_template(rng)
    g[:5, :5] = template

    # legend 3x3
    g[:3, 6:9] = 6
    g[1, 7] = 0
    dr, dc = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
    g[1 + dr, 7 + dc] = 4

    sep = 8
    g[sep, :] = 6

    n_copies = int(rng.integers(2, 4))
    starts = [0, 7, 14][:n_copies]

    for sc in starts:
        copy = template.copy()
        miss_candidates = np.argwhere(template != 0)
        if len(miss_candidates) == 0:
            continue
        rng.shuffle(miss_candidates)
        miss_count = int(rng.integers(2, min(5, len(miss_candidates) + 1)))
        miss = set((int(y), int(x)) for y, x in miss_candidates[:miss_count])
        for y, x in miss:
            copy[y, x] = 0

        order = _ordered_iter(5, 5, dr, dc)
        for y, x in order:
            if (y, x) not in miss:
                continue
            color = int(template[y, x])
            yy, xx = y, x
            while True:
                ny, nx = yy + dr, xx + dc
                if not (0 <= ny < 5 and 0 <= nx < 5):
                    break
                if copy[ny, nx] != 0:
                    break
                yy, xx = ny, nx
            copy[yy, xx] = color
            copy[y, x] = 0

        g[sep + 1 : sep + 6, sc : sc + 5] = copy

    return g
