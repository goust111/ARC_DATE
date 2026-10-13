import numpy as np
from collections import defaultdict, deque


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIZE_OPTIONS = ((3, 3), (3, 5), (3, 7), (3, 9), (5, 5), (5, 7), (5, 9))
MAX_RETRIES = 300
_RNG = np.random.default_rng()


def _normalize(coords):
    coords = list(coords)
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = {(r - r0, c - c0) for r, c in coords}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, h, w


def _rotate(coords, h, w, turns):
    shape = set(coords)
    hh, ww = int(h), int(w)
    for _ in range(int(turns) % 4):
        shape = {(c, hh - 1 - r) for r, c in shape}
        shape, hh, ww = _normalize(shape)
    return shape, hh, ww


def _rotate180(coords, h, w):
    return {(int(h) - 1 - r, int(w) - 1 - c) for r, c in coords}


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _find_anchor_row(grid):
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(np.count_nonzero(row == MAROON)) != 1:
            continue
        if np.all((row == GREY) | (row == MAROON)):
            candidates.append(int(r))
    assert candidates, "missing grey marker row"
    row = max(candidates)
    anchor_col = int(np.where(grid[row] == MAROON)[0][0])
    return row, anchor_col


def _fit_rectangle_one_way(shape_a, ha, wa, shape_b_input, hb, wb):
    shape_b = _rotate180(shape_b_input, hb, wb)
    dims = set()
    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            shifted = {(r + dr, c + dc) for r, c in shape_b}
            if shape_a & shifted:
                continue
            union = shape_a | shifted
            rs = [r for r, _ in union]
            cs = [c for _, c in union]
            uh = max(rs) - min(rs) + 1
            uw = max(cs) - min(cs) + 1
            if len(union) == uh * uw:
                dims.add((int(uh), int(uw)))
    return dims


def _fit_rectangle(shape1, shape2):
    a, ha, wa = shape1
    b, hb, wb = shape2
    dims = set()
    dims |= _fit_rectangle_one_way(a, ha, wa, b, hb, wb)
    dims |= _fit_rectangle_one_way(b, hb, wb, a, ha, wa)
    assert dims, "pair does not assemble into a rectangle"
    assert len(dims) == 1, f"ambiguous rectangle dimensions: {sorted(dims)}"
    return next(iter(dims))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_row, anchor_col = _find_anchor_row(g)
    work = g[:marker_row, :]

    by_color = defaultdict(list)
    for color in np.unique(work):
        color = int(color)
        if color in (BLACK, GREY, MAROON):
            continue
        comps = _components(work == color)
        assert len(comps) == 2, "each color must appear in exactly two components"
        for comp in comps:
            by_color[color].append(_normalize(comp))

    rects = []
    for color, pair in by_color.items():
        assert len(pair) == 2
        rh, rw = _fit_rectangle(pair[0], pair[1])
        rects.append((int(rh * rw), int(rh), int(rw), int(color)))

    rects.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    out = np.full((h, w), BLACK, dtype=int)
    out[marker_row:, :] = g[marker_row:, :]

    bottom = marker_row - 1
    for _, rh, rw, color in rects:
        r0 = int(bottom - rh + 1)
        c0 = int(anchor_col - rw // 2)
        assert r0 >= 0 and c0 >= 0 and c0 + rw <= w
        out[r0 : bottom + 1, c0 : c0 + rw] = color
        bottom = r0 - 1

    return out


def _partition_rows(height, width):
    full_rows = int(_RNG.integers(1, height - 1))
    tail_len = int(height - full_rows)
    choices = np.sort(_RNG.choice(np.arange(1, width), size=tail_len, replace=False))[::-1]
    rows = [int(width)] * full_rows + [int(x) for x in choices]
    return rows


def _make_partition(height, width):
    rows = _partition_rows(height, width)
    all_cells = {(r, c) for r in range(height) for c in range(width)}
    shape_a = {(r, c) for r, ncols in enumerate(rows) for c in range(int(ncols))}
    shape_b = all_cells - shape_a

    norm_a = _normalize(shape_a)
    norm_b = _normalize(shape_b)

    # Reject near-rectangles so both fragments stay visibly L-like.
    for shape, sh, sw in (norm_a, norm_b):
        assert 2 <= sh and 2 <= sw
        assert len(shape) < sh * sw

    return norm_a, norm_b


def _sample_pair(rect_h, rect_w):
    for _ in range(100):
        part_a, part_b = _make_partition(rect_h, rect_w)
        turns = int(_RNG.integers(0, 4))
        inp_a = _rotate(*part_a, turns)
        inp_b = _rotate(*part_b, turns + 2)
        dims = _fit_rectangle(inp_a, inp_b)
        expected = (rect_h, rect_w) if turns % 2 == 0 else (rect_w, rect_h)
        if dims != expected:
            continue
        if _RNG.random() < 0.5:
            return inp_a, inp_b, expected
        return inp_b, inp_a, expected
    raise RuntimeError("failed to sample an unambiguous complementary pair")


def _paint(grid, coords, color, top, left):
    for r, c in coords:
        grid[top + r, left + c] = int(color)


def _can_place(occupied, ph, pw, top, left):
    h, w = occupied.shape
    r0 = max(0, int(top) - 1)
    r1 = min(h, int(top) + int(ph) + 1)
    c0 = max(0, int(left) - 1)
    c1 = min(w, int(left) + int(pw) + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_pieces(grid, pieces, usable_rows):
    occupied = np.zeros((usable_rows, grid.shape[1]), dtype=bool)
    for color, (coords, ph, pw) in sorted(pieces, key=lambda item: (item[1][1] * item[1][2], item[1][1], item[1][2]), reverse=True):
        placed = False
        for _ in range(300):
            top = int(_RNG.integers(0, usable_rows - ph + 1))
            left = int(_RNG.integers(0, grid.shape[1] - pw + 1))
            if not _can_place(occupied, ph, pw, top, left):
                continue
            _paint(grid, coords, color, top, left)
            for r, c in coords:
                occupied[top + r, left + c] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt():
    pair_count = int(_RNG.integers(2, 5))
    size_ids = _RNG.choice(len(SIZE_OPTIONS), size=pair_count, replace=False)
    colors = _RNG.choice(PALETTE, size=pair_count, replace=False)

    rect_specs = []
    pieces = []
    total_height = 0
    max_width = 0
    max_piece_h = 0

    for idx, color in zip(size_ids, colors):
        rect_h, rect_w = map(int, SIZE_OPTIONS[int(idx)])
        piece1, piece2, out_dims = _sample_pair(rect_h, rect_w)
        out_h, out_w = out_dims
        total_height += int(out_h)
        max_width = max(max_width, int(out_w))
        max_piece_h = max(max_piece_h, int(piece1[1]), int(piece2[1]))
        rect_specs.append((int(out_h * out_w), int(out_h), int(out_w), int(color)))
        pieces.append((int(color), piece1))
        pieces.append((int(color), piece2))

    min_h = max(18, total_height + 2, max_piece_h + 4)
    min_w = max(18, max_width + 4)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = int(_RNG.integers(min_h, 31))
    grid_w = int(_RNG.integers(min_w, 31))
    anchor_lo = max_width // 2
    anchor_hi = grid_w - max_width // 2
    if anchor_lo >= anchor_hi:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    usable_rows = grid_h - 2
    if usable_rows < 1:
        return None
    if not _place_pieces(grid, pieces, usable_rows):
        return None

    anchor_col = int(_RNG.integers(anchor_lo, anchor_hi))
    grid[-1, :] = GREY
    grid[-1, anchor_col] = MAROON

    out = main(grid)
    if not np.array_equal(out[-1], grid[-1]):
        return None
    if int(np.count_nonzero(out[:-1] == BLACK)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
