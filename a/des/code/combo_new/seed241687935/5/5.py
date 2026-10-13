import numpy as np


BG = 0
RED = 2
YELLOW = 4
GREY = 5
PATTERN_COLORS = np.array([1, 3, 6, 7, 8, 9], dtype=int)
MAX_RETRIES = 400
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _same4_count(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    color = int(grid[r, c])
    return sum(int(grid[nr, nc]) == color for nr, nc in _neighbors4(r, c, h, w))


def _same8_count(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    color = int(grid[r, c])
    count = 0
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if nr == r and nc == c:
                continue
            count += int(grid[nr, nc]) == color
    return count


def _same4_if_changed(grid: np.ndarray, r: int, c: int, color: int) -> int:
    h, w = grid.shape
    return sum(int(grid[nr, nc]) == int(color) for nr, nc in _neighbors4(r, c, h, w))


def _consistency_score(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    return _same4_count(grid, r, c), _same8_count(grid, r, c)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_axis(grid: np.ndarray) -> tuple[str, int]:
    yellow_rows = [int(r) for r in np.where(np.all(grid == YELLOW, axis=1))[0]]
    yellow_cols = [int(c) for c in np.where(np.all(grid == YELLOW, axis=0))[0]]
    assert len(yellow_rows) + len(yellow_cols) == 1, "expected one full yellow symmetry axis"
    if yellow_rows:
        return "h", yellow_rows[0]
    return "v", yellow_cols[0]


def _find_helper_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected one grey helper frame"
    r0, c0, r1, c1 = _bbox(coords)
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(frame[0, :] == GREY)
    assert np.all(frame[-1, :] == GREY)
    assert np.all(frame[:, 0] == GREY)
    assert np.all(frame[:, -1] == GREY)
    assert np.all(frame[1:-1, 1:-1] == BG)
    return r0, c0, r1, c1


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = np.sort(RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    edges = np.concatenate(([0], cuts, [total]))
    return [int(edges[i + 1] - edges[i]) for i in range(parts)]


def _make_pattern(h: int, w: int, palette: np.ndarray) -> np.ndarray:
    row_parts = int(RNG.integers(2, min(4, h) + 1))
    col_parts = int(RNG.integers(2, min(4, w) + 1))
    row_sizes = _random_partition(h, row_parts)
    col_sizes = _random_partition(w, col_parts)

    grid = np.empty((h, w), dtype=int)
    r = 0
    for rh in row_sizes:
        c = 0
        for cw in col_sizes:
            color = int(palette[int(RNG.integers(len(palette)))])
            grid[r : r + rh, c : c + cw] = color
            c += cw
        r += rh

    for _ in range(int(RNG.integers(2, 6))):
        color = int(palette[int(RNG.integers(len(palette)))])
        r0 = int(RNG.integers(0, h))
        r1 = int(RNG.integers(r0, h))
        c0 = int(RNG.integers(0, w))
        c1 = int(RNG.integers(c0, w))
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    if RNG.random() < 0.6:
        color = int(palette[int(RNG.integers(len(palette)))])
        if RNG.random() < 0.5:
            rr = int(RNG.integers(0, h))
            grid[rr, :] = color
        else:
            cc = int(RNG.integers(0, w))
            grid[:, cc] = color

    return grid


def _build_pattern_pair(
    orient: str, h: int, w: int, palette: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    base = _make_pattern(h, w, palette)
    if len(set(map(int, np.unique(base)))) < 3:
        return None

    if orient == "v":
        mirrored = np.fliplr(base)
    else:
        mirrored = np.flipud(base)

    palette_used = [int(c) for c in np.unique(base)]
    candidates = []
    for r in range(h):
        for c in range(w):
            mr = r if orient == "v" else h - 1 - r
            mc = w - 1 - c if orient == "v" else c
            assert int(base[r, c]) == int(mirrored[mr, mc])
            support = _same4_count(base, r, c)
            if support < 2:
                continue
            valid_wrong = []
            for color in palette_used:
                if color == int(base[r, c]):
                    continue
                if support > _same4_if_changed(base, r, c, color) and support > _same4_if_changed(
                    mirrored, mr, mc, color
                ):
                    valid_wrong.append(int(color))
            if valid_wrong:
                candidates.append((r, c, valid_wrong))

    if not candidates:
        return None

    shuffled = list(candidates)
    RNG.shuffle(shuffled)
    target = int(RNG.integers(1, min(5, len(shuffled)) + 1))
    chosen = []
    for r, c, valid_wrong in shuffled:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc, _vals in chosen):
            chosen.append((r, c, valid_wrong))
            if len(chosen) == target:
                break
    if len(chosen) != target:
        return None

    left = base.copy()
    right = mirrored.copy()
    marks = np.full((h, w), BG, dtype=int)

    for r, c, valid_wrong in chosen:
        wrong_color = int(valid_wrong[int(RNG.integers(len(valid_wrong)))])
        wrong_on_first = bool(RNG.integers(0, 2))
        if orient == "v":
            rr, cc = r, w - 1 - c
            if wrong_on_first:
                left[r, c] = wrong_color
                marks[r, c] = RED
            else:
                right[rr, cc] = wrong_color
                marks[rr, cc] = RED
        else:
            rr, cc = h - 1 - r, c
            if wrong_on_first:
                left[r, c] = wrong_color
                marks[r, c] = RED
            else:
                right[rr, cc] = wrong_color
                marks[rr, cc] = RED

    return np.array(left, dtype=int), np.array(right, dtype=int), np.array(marks, dtype=int), base, mirrored


def _assemble_instance(
    orient: str,
    first: np.ndarray,
    second: np.ndarray,
    marks: np.ndarray,
    clean_first: np.ndarray,
    clean_second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ph, pw = first.shape
    grid = np.full((2 * ph + 5, 2 * pw + 5), BG, dtype=int)
    out = np.full_like(grid, BG)
    helper_h = ph + 2
    helper_w = pw + 2

    if orient == "v":
        axis = pw + 2
        grid[:, axis] = YELLOW
        out[:, axis] = YELLOW
        helper_top = bool(RNG.integers(0, 2))
        helper_left = bool(RNG.integers(0, 2))
        pattern_r0 = ph + 3 if helper_top else 2
        helper_r0 = 0 if helper_top else ph + 3
        left_c0 = 2
        right_c0 = pw + 3
        helper_c0 = 0 if helper_left else 2 * pw + 5 - helper_w

        grid[pattern_r0 : pattern_r0 + ph, left_c0 : left_c0 + pw] = first
        grid[pattern_r0 : pattern_r0 + ph, right_c0 : right_c0 + pw] = second
        out[pattern_r0 : pattern_r0 + ph, left_c0 : left_c0 + pw] = clean_first
        out[pattern_r0 : pattern_r0 + ph, right_c0 : right_c0 + pw] = clean_second

    else:
        axis = ph + 2
        grid[axis, :] = YELLOW
        out[axis, :] = YELLOW
        helper_top = bool(RNG.integers(0, 2))
        helper_left = bool(RNG.integers(0, 2))
        pattern_c0 = pw + 3 if helper_left else 2
        helper_c0 = 0 if helper_left else pw + 3
        top_r0 = 2
        bottom_r0 = ph + 3
        helper_r0 = 0 if helper_top else 2 * ph + 5 - helper_h

        grid[top_r0 : top_r0 + ph, pattern_c0 : pattern_c0 + pw] = first
        grid[bottom_r0 : bottom_r0 + ph, pattern_c0 : pattern_c0 + pw] = second
        out[top_r0 : top_r0 + ph, pattern_c0 : pattern_c0 + pw] = clean_first
        out[bottom_r0 : bottom_r0 + ph, pattern_c0 : pattern_c0 + pw] = clean_second

    grid[helper_r0 : helper_r0 + helper_h, helper_c0 : helper_c0 + helper_w] = GREY
    out[helper_r0 : helper_r0 + helper_h, helper_c0 : helper_c0 + helper_w] = GREY
    grid[helper_r0 + 1 : helper_r0 + 1 + ph, helper_c0 + 1 : helper_c0 + 1 + pw] = BG
    out[helper_r0 + 1 : helper_r0 + 1 + ph, helper_c0 + 1 : helper_c0 + 1 + pw] = marks

    return grid, out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair asymmetric defect pixels across a yellow mirror axis and mark the repaired offsets in the helper frame.

    Input structure:
    - The background is black.
    - One full yellow row or one full yellow column is the symmetry axis.
    - On opposite sides of that axis sit two same-sized filled multicolor rectangles whose contents should be mirror images, except for a few mismatched defect pixels.
    - One corner contains a grey one-pixel-thick frame with a black interior whose interior size matches the pattern rectangles.

    Rule:
    - Find the two pattern rectangles and compare mirror-corresponding cells across the yellow axis.
    - For each mismatched pair, score both sides by how many orthogonal neighbors on that same side share the pixel color; the higher-scoring side is the correct one.
    - Copy the correct color onto the lower-scoring side so the two rectangles become perfect mirrors.
    - In the grey helper frame, paint red at the repaired pixel's offset inside the half that was corrected; leave other helper cells black.

    Key insight:
    - The task is not just to find reflection mismatches: each mismatch also requires a local coherence judgment, and the helper frame records only the locations of the pixels that were actually changed.
    """
    g = np.asarray(grid, dtype=int)
    orient, axis = _find_axis(g)
    fr0, fc0, fr1, fc1 = _find_helper_frame(g)
    helper = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)

    pattern_mask = ~np.isin(g, [BG, YELLOW, GREY])
    comps = _components(pattern_mask)
    assert len(comps) == 2, "expected exactly two pattern components"

    bboxes = [_bbox(comp) for comp in comps]
    if orient == "v":
        order = np.argsort([box[1] for box in bboxes])
        left_box = bboxes[int(order[0])]
        right_box = bboxes[int(order[1])]
        r0a, c0a, r1a, c1a = left_box
        r0b, c0b, r1b, c1b = right_box
        assert c1a < axis < c0b
        assert r0a == r0b and r1a == r1b
        h = int(r1a - r0a + 1)
        w = int(c1a - c0a + 1)
        assert (helper[2] - helper[0] + 1, helper[3] - helper[1] + 1) == (h, w)

        out = g.copy()
        out[helper[0] : helper[2] + 1, helper[1] : helper[3] + 1] = BG
        for r in range(h):
            for c in range(w):
                lc = int(g[r0a + r, c0a + c])
                rc_local = w - 1 - c
                rc = int(g[r0b + r, c0b + rc_local])
                if lc == rc:
                    continue
                left_score = _consistency_score(g, r0a + r, c0a + c)
                right_score = _consistency_score(g, r0b + r, c0b + rc_local)
                if left_score == right_score:
                    raise AssertionError("defect side must be uniquely determined")
                if left_score > right_score:
                    out[r0b + r, c0b + rc_local] = lc
                    out[helper[0] + r, helper[1] + rc_local] = RED
                else:
                    out[r0a + r, c0a + c] = rc
                    out[helper[0] + r, helper[1] + c] = RED
        return out

    order = np.argsort([box[0] for box in bboxes])
    top_box = bboxes[int(order[0])]
    bottom_box = bboxes[int(order[1])]
    r0a, c0a, r1a, c1a = top_box
    r0b, c0b, r1b, c1b = bottom_box
    assert r1a < axis < r0b
    assert c0a == c0b and c1a == c1b
    h = int(r1a - r0a + 1)
    w = int(c1a - c0a + 1)
    assert (helper[2] - helper[0] + 1, helper[3] - helper[1] + 1) == (h, w)

    out = g.copy()
    out[helper[0] : helper[2] + 1, helper[1] : helper[3] + 1] = BG
    for r in range(h):
        br_local = h - 1 - r
        for c in range(w):
            tv = int(g[r0a + r, c0a + c])
            bv = int(g[r0b + br_local, c0b + c])
            if tv == bv:
                continue
            top_score = _consistency_score(g, r0a + r, c0a + c)
            bottom_score = _consistency_score(g, r0b + br_local, c0b + c)
            if top_score == bottom_score:
                raise AssertionError("defect side must be uniquely determined")
            if top_score > bottom_score:
                out[r0b + br_local, c0b + c] = tv
                out[helper[0] + br_local, helper[1] + c] = RED
            else:
                out[r0a + r, c0a + c] = bv
                out[helper[0] + r, helper[1] + c] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    orient = "v" if bool(RNG.integers(0, 2)) else "h"
    ph = int(RNG.integers(5, 11))
    pw = int(RNG.integers(5, 11))
    palette_size = int(RNG.integers(3, 6))
    palette = np.array(RNG.choice(PATTERN_COLORS, size=palette_size, replace=False), dtype=int)

    built = _build_pattern_pair(orient, ph, pw, palette)
    if built is None:
        return None
    first, second, marks, clean_first, clean_second = built
    inp, expected = _assemble_instance(orient, first, second, marks, clean_first, clean_second)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-defect input")
