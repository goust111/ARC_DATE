import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RNG = np.random.default_rng()


def _neighbors4(r, c):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords):
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask.copy()
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_enclosed_black_holes(crop):
    black = np.asarray(crop == BLACK, dtype=bool)
    h, w = black.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _axis_row_col(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_candidates = []
    for r in range(h):
        row = g[r, :]
        if set(map(int, np.unique(row))) <= {BLUE, RED}:
            blue_count = int(np.sum(row == BLUE))
            red_count = int(np.sum(row == RED))
            if blue_count + red_count == w and blue_count >= w - 1:
                row_candidates.append((blue_count, r))
    assert row_candidates, "expected one horizontal blue axis"
    axis_r = max(row_candidates)[1]

    col_candidates = []
    for c in range(w):
        col = g[:, c]
        if set(map(int, np.unique(col))) <= {BLUE, RED}:
            red_count = int(np.sum(col == RED))
            blue_count = int(np.sum(col == BLUE))
            if red_count + blue_count == h and red_count >= h - 1:
                col_candidates.append((red_count, c))
    assert col_candidates, "expected one vertical red axis"
    axis_c = max(col_candidates)[1]
    return int(axis_r), int(axis_c)


def _mirror_coords(coords, axis_r, axis_c, hole_count):
    if hole_count == 1:
        reflected = [(int(r), int(2 * axis_c - c)) for r, c in coords]
    else:
        reflected = [(int(2 * axis_r - r), int(c)) for r, c in coords]
    return reflected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each green closed object across the axis selected by its black-hole count, then recolor the reflected copy.

    Input structure:
    - The background is black.
    - One red vertical axis and one blue horizontal axis span the grid and cross once.
    - Several 4-connected green closed objects lie away from the axes; each encloses either one or two black holes.

    Rule:
    - Keep the red vertical axis and blue horizontal axis unchanged.
    - Split the green pixels into 4-connected objects and count each object's enclosed black holes.
    - If an object has one hole, reflect it across the red vertical axis and draw the reflected copy in yellow.
    - If an object has two holes, reflect it across the blue horizontal axis and draw the reflected copy in gray.
    - The original green objects stay unchanged, and any reflected pixel that would land on either axis is skipped.

    Key insight:
    - The decisive feature is topological: the number of enclosed holes does not choose a color directly, it chooses which geometric reflection to apply before recoloring the copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    axis_r, axis_c = _axis_row_col(g)
    out = g.copy()

    for comp in _components(g == GREEN):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        holes = _count_enclosed_black_holes(crop)
        assert holes in {1, 2}, f"unsupported hole count: {holes}"
        color = YELLOW if holes == 1 else GRAY
        reflected = _mirror_coords(comp, axis_r, axis_c, holes)
        for rr, cc in reflected:
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                continue
            if rr == axis_r or cc == axis_c:
                continue
            out[rr, cc] = color

    return out


def _carve_rect_hole(mask, rng):
    h, w = mask.shape
    hole_h = int(rng.integers(1, min(3, h - 1)))
    hole_w = int(rng.integers(1, min(3, w - 1)))
    r0 = int(rng.integers(1, h - hole_h))
    c0 = int(rng.integers(1, w - hole_w))
    mask[r0 : r0 + hole_h, c0 : c0 + hole_w] = False


def _carve_boundary_notch(mask, rng):
    h, w = mask.shape
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        depth = int(rng.integers(1, min(3, h)))
        width = int(rng.integers(1, min(4, w + 1)))
        c0 = int(rng.integers(0, w - width + 1))
        if side == "top":
            mask[:depth, c0 : c0 + width] = False
        else:
            mask[h - depth :, c0 : c0 + width] = False
    else:
        depth = int(rng.integers(1, min(3, w)))
        height = int(rng.integers(1, min(4, h + 1)))
        r0 = int(rng.integers(0, h - height + 1))
        if side == "left":
            mask[r0 : r0 + height, :depth] = False
        else:
            mask[r0 : r0 + height, w - depth :] = False


def _make_object_mask(hole_count, rng):
    for _ in range(200):
        if hole_count == 1:
            h = int(rng.integers(4, 7))
            w = int(rng.integers(4, 7))
        else:
            h = int(rng.integers(5, 8))
            w = int(rng.integers(5, 8))

        mask = np.ones((h, w), dtype=bool)
        if hole_count == 1:
            _carve_rect_hole(mask, rng)
        else:
            _carve_rect_hole(mask, rng)
            _carve_rect_hole(mask, rng)

        notch_count = int(rng.integers(0, 3))
        for _ in range(notch_count):
            candidate = mask.copy()
            _carve_boundary_notch(candidate, rng)
            candidate = _trim_mask(candidate)
            if candidate.size == 0:
                continue
            if len(_components(candidate)) != 1:
                continue
            crop = np.where(candidate, GREEN, BLACK)
            if _count_enclosed_black_holes(crop) != hole_count:
                continue
            mask = candidate

        k = int(rng.integers(0, 4))
        mask = np.rot90(mask, k)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = _trim_mask(mask)

        if len(_components(mask)) != 1:
            continue
        crop = np.where(mask, GREEN, BLACK)
        if _count_enclosed_black_holes(crop) != hole_count:
            continue
        area = int(mask.sum())
        if not (8 <= area <= 24):
            continue
        if mask.shape[0] < 4 or mask.shape[1] < 4:
            continue
        return mask.astype(bool)
    raise RuntimeError("failed to build object mask")


def _quadrant_ranges(h, w, axis_r, axis_c, quadrant, obj_h, obj_w):
    if quadrant == "TL":
        return (0, axis_r - obj_h), (0, axis_c - obj_w)
    if quadrant == "TR":
        return (0, axis_r - obj_h), (axis_c + 1, w - obj_w)
    if quadrant == "BL":
        return (axis_r + 1, h - obj_h), (0, axis_c - obj_w)
    if quadrant == "BR":
        return (axis_r + 1, h - obj_h), (axis_c + 1, w - obj_w)
    raise ValueError(quadrant)


def _coords_from_mask(mask, r0, c0):
    coords = np.argwhere(mask)
    return [(int(r0 + r), int(c0 + c)) for r, c in coords]


def _pick_from_options(options, fallback):
    vals = [int(v) for v in options if v is not None]
    if vals:
        return vals
    return [int(fallback)]


def _slot_origin(h, w, axis_r, axis_c, quadrant, hole_count, obj_h, obj_w, rng):
    top_outer = _pick_from_options([1, 2 if 2 <= axis_r - obj_h - 1 else None], 1)
    top_inner = _pick_from_options(
        [axis_r - obj_h - 2, axis_r - obj_h - 1],
        max(1, axis_r - obj_h - 1),
    )
    bottom_inner = _pick_from_options(
        [axis_r + 1, axis_r + 2 if axis_r + 2 <= h - obj_h - 1 else None],
        axis_r + 1,
    )
    bottom_outer = _pick_from_options(
        [h - obj_h - 2, h - obj_h - 1],
        h - obj_h - 1,
    )

    left_outer = _pick_from_options([1, 2 if 2 <= axis_c - obj_w - 1 else None], 1)
    left_inner = _pick_from_options(
        [axis_c - obj_w - 2, axis_c - obj_w - 1],
        max(1, axis_c - obj_w - 1),
    )
    right_inner = _pick_from_options(
        [axis_c + 1, axis_c + 2 if axis_c + 2 <= w - obj_w - 1 else None],
        axis_c + 1,
    )
    right_outer = _pick_from_options(
        [w - obj_w - 2, w - obj_w - 1],
        w - obj_w - 1,
    )

    if quadrant == "TL" and hole_count == 1:
        row_choices, col_choices = top_outer, left_outer
    elif quadrant == "TL" and hole_count == 2:
        row_choices, col_choices = top_inner, left_inner
    elif quadrant == "TR" and hole_count == 1:
        row_choices, col_choices = top_inner, right_outer
    elif quadrant == "TR" and hole_count == 2:
        row_choices, col_choices = top_outer, right_outer
    elif quadrant == "BL" and hole_count == 1:
        row_choices, col_choices = bottom_inner, left_inner
    elif quadrant == "BL" and hole_count == 2:
        row_choices, col_choices = bottom_outer, left_inner
    elif quadrant == "BR" and hole_count == 1:
        row_choices, col_choices = bottom_outer, right_outer
    elif quadrant == "BR" and hole_count == 2:
        row_choices, col_choices = bottom_inner, right_inner
    else:
        raise ValueError((quadrant, hole_count))

    r0 = int(rng.choice(row_choices))
    c0 = int(rng.choice(col_choices))
    return r0, c0


def _slot_choices(h, w, axis_r, axis_c, quadrant, hole_count, obj_h, obj_w, rng):
    top_outer = _pick_from_options([1, 2 if 2 <= axis_r - obj_h - 1 else None], 1)
    top_inner = _pick_from_options(
        [axis_r - obj_h - 2, axis_r - obj_h - 1],
        max(1, axis_r - obj_h - 1),
    )
    bottom_inner = _pick_from_options(
        [axis_r + 1, axis_r + 2 if axis_r + 2 <= h - obj_h - 1 else None],
        axis_r + 1,
    )
    bottom_outer = _pick_from_options(
        [h - obj_h - 2, h - obj_h - 1],
        h - obj_h - 1,
    )

    left_outer = _pick_from_options([1, 2 if 2 <= axis_c - obj_w - 1 else None], 1)
    left_inner = _pick_from_options(
        [axis_c - obj_w - 2, axis_c - obj_w - 1],
        max(1, axis_c - obj_w - 1),
    )
    right_inner = _pick_from_options(
        [axis_c + 1, axis_c + 2 if axis_c + 2 <= w - obj_w - 1 else None],
        axis_c + 1,
    )
    right_outer = _pick_from_options(
        [w - obj_w - 2, w - obj_w - 1],
        w - obj_w - 1,
    )

    if quadrant == "TL" and hole_count == 1:
        row_choices, col_choices = top_outer, left_outer
    elif quadrant == "TL" and hole_count == 2:
        row_choices, col_choices = top_inner, left_inner
    elif quadrant == "TR" and hole_count == 1:
        row_choices, col_choices = top_inner, right_outer
    elif quadrant == "TR" and hole_count == 2:
        row_choices, col_choices = top_outer, right_outer
    elif quadrant == "BL" and hole_count == 1:
        row_choices, col_choices = bottom_inner, left_inner
    elif quadrant == "BL" and hole_count == 2:
        row_choices, col_choices = bottom_outer, left_inner
    elif quadrant == "BR" and hole_count == 1:
        row_choices, col_choices = bottom_outer, right_outer
    elif quadrant == "BR" and hole_count == 2:
        row_choices, col_choices = bottom_inner, right_inner
    else:
        raise ValueError((quadrant, hole_count))

    choices = [(int(r0), int(c0)) for r0 in row_choices for c0 in col_choices]
    order = rng.permutation(len(choices))
    return [choices[int(i)] for i in order]


def _mark_blocked(blocked, coords):
    h, w = blocked.shape
    for r, c in coords:
        for dr, dc in DIR4 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_object(grid, blocked, axis_r, axis_c, quadrant, hole_count, rng):
    h, w = grid.shape
    for _ in range(8):
        try:
            mask = _make_object_mask(hole_count, rng)
        except RuntimeError:
            continue
        obj_h, obj_w = mask.shape
        row_range, col_range = _quadrant_ranges(h, w, axis_r, axis_c, quadrant, obj_h, obj_w)
        r_lo, r_hi = row_range
        c_lo, c_hi = col_range
        if r_lo > r_hi or c_lo > c_hi:
            continue

        for _ in range(200):
            r0 = int(rng.integers(r_lo, r_hi + 1))
            c0 = int(rng.integers(c_lo, c_hi + 1))
            source = _coords_from_mask(mask, r0, c0)
            reflected = _mirror_coords(source, axis_r, axis_c, hole_count)
            all_coords = source + reflected

            ok = True
            for rr, cc in all_coords:
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if rr == axis_r or cc == axis_c:
                    ok = False
                    break
                if blocked[rr, cc]:
                    ok = False
                    break
            if not ok:
                continue

            out_grid = grid.copy()
            for rr, cc in source:
                out_grid[rr, cc] = GREEN
            out_blocked = blocked.copy()
            _mark_blocked(out_blocked, source)
            _mark_blocked(out_blocked, reflected)
            return {
                "grid": out_grid,
                "blocked": out_blocked,
                "source": source,
                "reflected": reflected,
                "hole_count": hole_count,
                "quadrant": quadrant,
            }
    return None


def _expected_output(inp):
    g = inp.copy()
    axis_r, axis_c = _axis_row_col(g)
    for comp in _components(g == GREEN):
        holes = _count_enclosed_black_holes(
            g[
                _bbox_from_coords(comp)[0] : _bbox_from_coords(comp)[2] + 1,
                _bbox_from_coords(comp)[1] : _bbox_from_coords(comp)[3] + 1,
            ]
        )
        color = YELLOW if holes == 1 else GRAY
        for rr, cc in _mirror_coords(comp, axis_r, axis_c, holes):
            if rr == axis_r or cc == axis_c:
                continue
            g[rr, cc] = color
    return g


def _validate_sample(inp, placed):
    assert inp.ndim == 2
    h, w = inp.shape
    assert max(h, w) <= 30
    axis_r, axis_c = _axis_row_col(inp)
    assert 0 < axis_r < h - 1
    assert 0 < axis_c < w - 1

    comps = _components(inp == GREEN)
    assert len(comps) == len(placed)
    hole_counts = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        holes = _count_enclosed_black_holes(inp[r0 : r1 + 1, c0 : c1 + 1])
        assert holes in {1, 2}
        hole_counts.append(holes)
    assert 1 in hole_counts and 2 in hole_counts

    by_hole = {1: set(), 2: set()}
    for item in placed[:4]:
        by_hole[item["hole_count"]].add(item["quadrant"])
    assert len(by_hole[1]) == 2 and len(by_hole[2]) == 2

    expected = inp.copy()
    for item in placed:
        color = YELLOW if item["hole_count"] == 1 else GRAY
        for rr, cc in item["reflected"]:
            assert expected[rr, cc] == BLACK
            expected[rr, cc] = color
    actual = main(inp)
    assert np.array_equal(actual, expected)

    yellow = int(np.sum(actual == YELLOW))
    gray = int(np.sum(actual == GRAY))
    assert yellow > 0 and gray > 0


def generate_input() -> np.ndarray:
    for _ in range(200):
        h = int(_RNG.integers(25, 30))
        w = int(_RNG.integers(25, 30))
        if h % 2 == 0:
            h += 1
        if w % 2 == 0:
            w += 1
        axis_r = h // 2
        axis_c = w // 2

        grid = np.zeros((h, w), dtype=int)
        grid[:, axis_c] = RED
        grid[axis_r, :] = BLUE
        if _RNG.random() < 0.5:
            grid[axis_r, axis_c] = RED

        quadrants = ["TL", "TR", "BL", "BR"]
        hole_pattern = [1, 1, 2, 2]
        _RNG.shuffle(hole_pattern)
        reserved = np.zeros((h, w), dtype=bool)
        reserved[:, axis_c] = True
        reserved[axis_r, :] = True

        placed = []
        ok = True
        for quadrant, hole_count in zip(quadrants, hole_pattern):
            try:
                mask = _make_object_mask(hole_count, _RNG)
            except RuntimeError:
                ok = False
                break

            obj_h, obj_w = mask.shape
            placed_here = False
            for r0, c0 in _slot_choices(h, w, axis_r, axis_c, quadrant, hole_count, obj_h, obj_w, _RNG):
                source = _coords_from_mask(mask, r0, c0)
                reflected = _mirror_coords(source, axis_r, axis_c, hole_count)

                candidate_ok = True
                for rr, cc in source + reflected:
                    if not (0 <= rr < h and 0 <= cc < w):
                        candidate_ok = False
                        break
                    if rr == axis_r or cc == axis_c or reserved[rr, cc]:
                        candidate_ok = False
                        break
                if not candidate_ok:
                    continue

                for rr, cc in source:
                    grid[rr, cc] = GREEN
                    reserved[rr, cc] = True
                for rr, cc in reflected:
                    reserved[rr, cc] = True
                placed.append(
                    {
                        "source": source,
                        "reflected": reflected,
                        "hole_count": hole_count,
                        "quadrant": quadrant,
                    }
                )
                placed_here = True
                break
            if not placed_here:
                ok = False
                break

        if not ok:
            continue

        seen = set()
        for item in placed:
            for coord in item["reflected"]:
                if coord in seen:
                    ok = False
                    break
                seen.add(coord)
            if not ok:
                break
        if not ok:
            continue

        try:
            _validate_sample(grid, placed)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid sample")
