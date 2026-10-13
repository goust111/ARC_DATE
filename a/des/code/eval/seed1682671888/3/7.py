import numpy as np


BG = 0
GREY = 5
ALL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CARD_WIDTH = 5
GEN_RETRIES = 400
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _find_recipe_card(grid: np.ndarray) -> tuple[int, int, int, int]:
    best = None
    for pts in _components(grid == GREY):
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 4:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[pts[:, 0] - r0, pts[:, 1] - c0] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(local, border):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        if not np.any((interior != BG) & (interior != GREY)):
            continue
        score = (int(r1 + c1), int(h * w))
        if best is None or score > best[0]:
            best = (score, (int(r0), int(c0), int(r1), int(c1)))
    assert best is not None, "expected one grey recipe card"
    return best[1]


def _parse_recipe(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[tuple[int, int], int]:
    r0, c0, r1, c1 = bbox
    mapping: dict[tuple[int, int], int] = {}
    for r in range(r0 + 1, r1):
        row = [int(v) for v in grid[r, c0 + 1 : c1] if int(v) not in (BG, GREY)]
        if not row:
            continue
        if len(row) == 2:
            mapping[(row[0], row[0])] = row[1]
        elif len(row) == 3:
            mapping[(row[0], row[1])] = row[2]
        else:
            raise AssertionError("recipe rows must contain 2 or 3 colored pixels")
    assert mapping, "recipe card must define at least one rule"
    return mapping


def _recover_rectangles(
    grid: np.ndarray, card_bbox: tuple[int, int, int, int]
) -> list[dict[str, int | tuple[int, int, int, int]]]:
    r0, c0, r1, c1 = card_bbox
    scene_mask = np.ones(grid.shape, dtype=bool)
    scene_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    colors = sorted(set(map(int, grid[scene_mask].reshape(-1))) - {BG, GREY})
    rects = []
    for color in colors:
        pts = np.argwhere((grid == color) & scene_mask)
        assert len(pts) > 0
        rr0, cc0 = pts.min(axis=0)
        rr1, cc1 = pts.max(axis=0)
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        rects.append(
            {
                "color": int(color),
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "area": int(h * w),
            }
        )
    areas = [int(item["area"]) for item in rects]
    assert len(set(areas)) == len(areas), "rectangle areas must uniquely determine layer order"
    rects.sort(key=lambda item: int(item["area"]), reverse=True)
    return rects


def _composite(
    shape: tuple[int, int],
    rects: list[dict[str, int | tuple[int, int, int, int]]],
    recipe: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        color = int(rect["color"])
        r0, c0, r1, c1 = rect["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        empty = patch == BG
        patch[empty] = color
        for rr, cc in np.argwhere(~empty):
            key = (int(patch[rr, cc]), color)
            assert key in recipe, f"missing recipe for {key}"
            patch[rr, cc] = recipe[key]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    card_bbox = _find_recipe_card(g)
    recipe = _parse_recipe(g, card_bbox)
    rects = _recover_rectangles(g, card_bbox)
    out = _composite(g.shape, rects, recipe)
    r0, c0, r1, c1 = card_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    return out


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _sample_outer_rect(scene_h: int, scene_w: int, margin: int) -> tuple[int, int, int, int] | None:
    h_hi = min(scene_h - 2 * margin, 10)
    w_hi = min(scene_w - 2 * margin, 10)
    if h_hi < 6 or w_hi < 6:
        return None
    h = _randint(6, h_hi)
    w = _randint(6, w_hi)
    r = _randint(margin, scene_h - h - margin)
    c = _randint(margin, scene_w - w - margin)
    return (r, c, h, w)


def _place_inside(outer: tuple[int, int, int, int], min_margin: int, min_size: int) -> tuple[int, int, int, int] | None:
    r0, c0, h0, w0 = outer
    h_hi = h0 - 2 * min_margin
    w_hi = w0 - 2 * min_margin
    if h_hi < min_size or w_hi < min_size:
        return None
    h = _randint(min_size, h_hi)
    w = _randint(min_size, w_hi)
    r_lo = r0 + min_margin
    r_hi = r0 + h0 - min_margin - h
    c_lo = c0 + min_margin
    c_hi = c0 + w0 - min_margin - w
    if r_lo > r_hi or c_lo > c_hi:
        return None
    r = _randint(r_lo, r_hi)
    c = _randint(c_lo, c_hi)
    return (r, c, h, w)


def _bands_between(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    or0, oc0, oh, ow = outer
    ir0, ic0, ih, iw = inner
    or1 = or0 + oh
    oc1 = oc0 + ow
    ir1 = ir0 + ih
    ic1 = ic0 + iw
    bands = []
    if ir0 - or0 >= 3:
        bands.append((or0, oc0, ir0 - or0, ow))
    if or1 - ir1 >= 3:
        bands.append((ir1, oc0, or1 - ir1, ow))
    if ic0 - oc0 >= 3:
        bands.append((or0, oc0, oh, ic0 - oc0))
    if oc1 - ic1 >= 3:
        bands.append((or0, ic1, oh, oc1 - ic1))
    return bands


def _place_in_band(band: tuple[int, int, int, int], max_area: int) -> tuple[int, int, int, int] | None:
    r0, c0, h0, w0 = band
    if h0 < 3 or w0 < 3:
        return None
    for _ in range(40):
        h = _randint(3, h0)
        w = _randint(3, w0)
        if h * w >= max_area:
            continue
        r = _randint(r0, r0 + h0 - h)
        c = _randint(c0, c0 + w0 - w)
        return (r, c, h, w)
    return None


def _place_partial(
    outer: tuple[int, int, int, int], scene_h: int, scene_w: int, min_protrusion: int
) -> tuple[tuple[int, int, int, int], str, int] | None:
    r0, c0, h0, w0 = outer
    orientations = ("right", "left", "down", "up")
    for _ in range(80):
        orient = str(_RNG.choice(orientations))
        if orient in ("right", "left"):
            h_hi = h0 - 2
            if h_hi < 4:
                continue
            h1 = _randint(4, h_hi)
            overlap_w_hi = min(w0 - 2, 5)
            if overlap_w_hi < 2:
                continue
            overlap_w = _randint(2, overlap_w_hi)
            max_protrusion = scene_w - (c0 + w0) if orient == "right" else c0
            if max_protrusion < min_protrusion:
                continue
            protrusion = _randint(min_protrusion, min(max_protrusion, 6))
            w1 = overlap_w + protrusion
            if h1 * w1 >= h0 * w0:
                continue
            r_lo = r0 + 1
            r_hi = r0 + h0 - h1 - 1
            if r_lo > r_hi:
                continue
            r1 = _randint(r_lo, r_hi)
            c1 = c0 + w0 - overlap_w if orient == "right" else c0 - protrusion
            if c1 < 0 or c1 + w1 > scene_w:
                continue
            return (r1, c1, h1, w1), orient, protrusion
        else:
            w_hi = w0 - 2
            if w_hi < 4:
                continue
            w1 = _randint(4, w_hi)
            overlap_h_hi = min(h0 - 2, 5)
            if overlap_h_hi < 2:
                continue
            overlap_h = _randint(2, overlap_h_hi)
            max_protrusion = scene_h - (r0 + h0) if orient == "down" else r0
            if max_protrusion < min_protrusion:
                continue
            protrusion = _randint(min_protrusion, min(max_protrusion, 6))
            h1 = overlap_h + protrusion
            if h1 * w1 >= h0 * w0:
                continue
            c_lo = c0 + 1
            c_hi = c0 + w0 - w1 - 1
            if c_lo > c_hi:
                continue
            c1 = _randint(c_lo, c_hi)
            r1 = r0 + h0 - overlap_h if orient == "down" else r0 - protrusion
            if r1 < 0 or r1 + h1 > scene_h:
                continue
            return (r1, c1, h1, w1), orient, protrusion
    return None


def _lobe_band(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], orient: str, protrusion: int
) -> tuple[int, int, int, int]:
    r1, c1, h1, w1 = inner
    if orient == "right":
        return (r1, c1 + w1 - protrusion, h1, protrusion)
    if orient == "left":
        return (r1, c1, h1, protrusion)
    if orient == "down":
        return (r1 + h1 - protrusion, c1, protrusion, w1)
    return (r1, c1, protrusion, w1)


def _place_in_lobe(
    band: tuple[int, int, int, int], max_area: int
) -> tuple[int, int, int, int] | None:
    r0, c0, h0, w0 = band
    if h0 < 5 or w0 < 5:
        return None
    h_hi = h0 - 2
    w_hi = w0 - 2
    if h_hi < 3 or w_hi < 3:
        return None
    for _ in range(40):
        h = _randint(3, h_hi)
        w = _randint(3, w_hi)
        if h * w >= max_area:
            continue
        r = _randint(r0 + 1, r0 + h0 - h - 1)
        c = _randint(c0 + 1, c0 + w0 - w - 1)
        return (r, c, h, w)
    return None


def _place_crossing_rect(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], orient: str, max_area: int
) -> tuple[int, int, int, int] | None:
    or0, oc0, oh, ow = outer
    ir0, ic0, ih, iw = inner
    or1 = or0 + oh
    oc1 = oc0 + ow
    ir1 = ir0 + ih
    ic1 = ic0 + iw
    for _ in range(60):
        if orient == "right":
            boundary = ic0
            a_span = boundary - oc0
            x_span = oc1 - boundary
            if a_span < 2 or x_span < 2:
                continue
            h_hi = min(ih, 6)
            if h_hi < 3:
                continue
            h = _randint(3, h_hi)
            r = _randint(ir0, ir1 - h)
            left = _randint(1, min(a_span, 3))
            right = _randint(1, min(x_span, 3))
            w = left + right
            if h * w >= max_area:
                continue
            c = boundary - left
            return (r, c, h, w)
        if orient == "left":
            boundary = ic1
            x_span = boundary - oc0
            a_span = oc1 - boundary
            if a_span < 2 or x_span < 2:
                continue
            h_hi = min(ih, 6)
            if h_hi < 3:
                continue
            h = _randint(3, h_hi)
            r = _randint(ir0, ir1 - h)
            left = _randint(1, min(x_span, 3))
            right = _randint(1, min(a_span, 3))
            w = left + right
            if h * w >= max_area:
                continue
            c = boundary - left
            return (r, c, h, w)
        if orient == "down":
            boundary = ir0
            a_span = boundary - or0
            x_span = or1 - boundary
            if a_span < 2 or x_span < 2:
                continue
            w_hi = min(iw, 6)
            if w_hi < 3:
                continue
            w = _randint(3, w_hi)
            c = _randint(ic0, ic1 - w)
            top = _randint(1, min(a_span, 3))
            bottom = _randint(1, min(x_span, 3))
            h = top + bottom
            if h * w >= max_area:
                continue
            r = boundary - top
            return (r, c, h, w)
        boundary = ir1
        x_span = boundary - or0
        a_span = or1 - boundary
        if a_span < 2 or x_span < 2:
            continue
        w_hi = min(iw, 6)
        if w_hi < 3:
            continue
        w = _randint(3, w_hi)
        c = _randint(ic0, ic1 - w)
        top = _randint(1, min(x_span, 3))
        bottom = _randint(1, min(a_span, 3))
        h = top + bottom
        if h * w >= max_area:
            continue
        r = boundary - top
        return (r, c, h, w)
    return None


def _opaque_input(shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for (r, c, h, w), color in zip(rects, colors):
        grid[r : r + h, c : c + w] = int(color)
    return grid


def _rect_states(rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> set[int]:
    if len(rects) != 3:
        return set()
    state = np.zeros(shape, dtype=int)
    r, c, h, w = rects[0]
    state[r : r + h, c : c + w] = 1
    r, c, h, w = rects[1]
    sub = state[r : r + h, c : c + w]
    sub[sub == 0] = 2
    sub[sub == 1] = 3
    r, c, h, w = rects[2]
    vals = state[r : r + h, c : c + w]
    return set(map(int, vals[vals > 0].reshape(-1)))


def _build_recipe_and_output(
    shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]
) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    mix_pool = [color for color in ALL_COLORS if color not in colors]
    _RNG.shuffle(mix_pool)
    recipe: dict[tuple[int, int], int] = {(color, color): color for color in colors}
    out = np.zeros(shape, dtype=int)
    next_mix = 0
    for rect, color in zip(rects, colors):
        r, c, h, w = rect
        patch = out[r : r + h, c : c + w]
        empty = patch == BG
        patch[empty] = color
        for rr, cc in np.argwhere(~empty):
            key = (int(patch[rr, cc]), int(color))
            if key not in recipe:
                assert next_mix < len(mix_pool), "not enough colors for mix states"
                recipe[key] = int(mix_pool[next_mix])
                next_mix += 1
            patch[rr, cc] = recipe[key]
    return recipe, out


def _visible_bboxes_ok(
    grid: np.ndarray, rects: list[tuple[int, int, int, int]], colors: list[int], scene_w: int
) -> bool:
    scene = grid[:, :scene_w]
    for rect, color in zip(rects, colors):
        pts = np.argwhere(scene == int(color))
        if len(pts) == 0:
            return False
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rr, cc, hh, ww = rect
        if (int(r0), int(c0), int(r1), int(c1)) != (rr, cc, rr + hh - 1, cc + ww - 1):
            return False
    return True


def _recipe_rows(recipe: dict[tuple[int, int], int]) -> list[list[int]]:
    rows = []
    for (a, b), out in recipe.items():
        if a == b:
            rows.append([a, out])
        else:
            rows.append([a, b, out])
    _RNG.shuffle(rows)
    return rows


def _add_card(grid: np.ndarray, rows: list[list[int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    card_h = len(rows) + 2
    h, w = grid.shape
    assert card_h <= h
    r0 = h - card_h
    c0 = w - CARD_WIDTH
    r1 = h - 1
    c1 = w - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    for i, row in enumerate(rows, start=1):
        for j, color in enumerate(row, start=1):
            grid[r0 + i, c0 + j] = int(color)
    return grid, (r0, c0, r1, c1)


def _gen_two_contained(scene_h: int, scene_w: int) -> list[tuple[int, int, int, int]] | None:
    outer = _sample_outer_rect(scene_h, scene_w, margin=1)
    if outer is None:
        return None
    inner = _place_inside(outer, min_margin=1, min_size=3)
    if inner is None or inner[2] * inner[3] >= outer[2] * outer[3]:
        return None
    return [outer, inner]


def _gen_two_offset(scene_h: int, scene_w: int) -> list[tuple[int, int, int, int]] | None:
    outer = _sample_outer_rect(scene_h, scene_w, margin=2)
    if outer is None:
        return None
    placed = _place_partial(outer, scene_h, scene_w, min_protrusion=2)
    if placed is None:
        return None
    inner, _orient, _protrusion = placed
    if inner[2] * inner[3] >= outer[2] * outer[3]:
        return None
    return [outer, inner]


def _gen_three_nested(scene_h: int, scene_w: int) -> list[tuple[int, int, int, int]] | None:
    outer = _sample_outer_rect(scene_h, scene_w, margin=1)
    if outer is None:
        return None
    mid = _place_inside(outer, min_margin=1, min_size=5)
    if mid is None or mid[2] * mid[3] >= outer[2] * outer[3]:
        return None
    top = _place_inside(mid, min_margin=1, min_size=3)
    if top is None or top[2] * top[3] >= mid[2] * mid[3]:
        return None
    return [outer, mid, top]


def _gen_three_a_only(scene_h: int, scene_w: int) -> list[tuple[int, int, int, int]] | None:
    outer = _sample_outer_rect(scene_h, scene_w, margin=1)
    if outer is None:
        return None
    mid = _place_inside(outer, min_margin=2, min_size=4)
    if mid is None or mid[2] * mid[3] >= outer[2] * outer[3]:
        return None
    candidates = [_place_in_band(band, mid[2] * mid[3]) for band in _bands_between(outer, mid)]
    candidates = [cand for cand in candidates if cand is not None]
    if not candidates:
        return None
    top = candidates[_randint(0, len(candidates) - 1)]
    if top[2] * top[3] >= mid[2] * mid[3]:
        return None
    return [outer, mid, top]


def _gen_three_b_only(scene_h: int, scene_w: int) -> list[tuple[int, int, int, int]] | None:
    outer = _sample_outer_rect(scene_h, scene_w, margin=2)
    if outer is None:
        return None
    placed = _place_partial(outer, scene_h, scene_w, min_protrusion=5)
    if placed is None:
        return None
    mid, orient, protrusion = placed
    if mid[2] * mid[3] >= outer[2] * outer[3]:
        return None
    top = _place_in_lobe(_lobe_band(outer, mid, orient, protrusion), mid[2] * mid[3])
    if top is None or top[2] * top[3] >= mid[2] * mid[3]:
        return None
    return [outer, mid, top]


def _gen_three_mixed(scene_h: int, scene_w: int) -> list[tuple[int, int, int, int]] | None:
    outer = _sample_outer_rect(scene_h, scene_w, margin=2)
    if outer is None:
        return None
    placed = _place_partial(outer, scene_h, scene_w, min_protrusion=2)
    if placed is None:
        return None
    mid, orient, _protrusion = placed
    if mid[2] * mid[3] >= outer[2] * outer[3]:
        return None
    top = _place_crossing_rect(outer, mid, orient, mid[2] * mid[3])
    if top is None or top[2] * top[3] >= mid[2] * mid[3]:
        return None
    return [outer, mid, top]


def _gen_attempt() -> np.ndarray | None:
    scene_h = _randint(12, 18)
    scene_w = _randint(12, 18)
    family = str(_RNG.choice(("two_contained", "two_offset", "three_nested", "three_a_only", "three_b_only", "three_mixed")))
    if family == "two_contained":
        rects = _gen_two_contained(scene_h, scene_w)
    elif family == "two_offset":
        rects = _gen_two_offset(scene_h, scene_w)
    elif family == "three_nested":
        rects = _gen_three_nested(scene_h, scene_w)
    elif family == "three_a_only":
        rects = _gen_three_a_only(scene_h, scene_w)
    elif family == "three_b_only":
        rects = _gen_three_b_only(scene_h, scene_w)
    else:
        rects = _gen_three_mixed(scene_h, scene_w)
    if rects is None:
        return None

    areas = [h * w for _r, _c, h, w in rects]
    if len(set(areas)) != len(areas) or areas != sorted(areas, reverse=True):
        return None

    if len(rects) == 3:
        states = _rect_states(rects, (scene_h, scene_w))
        if family == "three_nested" and states != {3}:
            return None
        if family == "three_a_only" and states != {1}:
            return None
        if family == "three_b_only" and states != {2}:
            return None
        if family == "three_mixed" and not ({1, 3} <= states or {2, 3} <= states):
            return None

    colors = list(map(int, _RNG.choice(np.asarray(ALL_COLORS), size=len(rects), replace=False)))
    recipe, expected = _build_recipe_and_output((scene_h, scene_w), rects, colors)
    rows = _recipe_rows(recipe)
    full = np.zeros((scene_h, scene_w + CARD_WIDTH + 1), dtype=int)
    opaque = _opaque_input((scene_h, scene_w), rects, colors)
    if not _visible_bboxes_ok(opaque, rects, colors, scene_w):
        return None
    if np.array_equal(opaque, expected):
        return None
    full[:, :scene_w] = opaque
    full, card_bbox = _add_card(full, rows)
    if card_bbox[0] < 0:
        return None
    expected_full = np.zeros_like(full)
    expected_full[:, :scene_w] = expected
    try:
        solved = main(full)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected_full):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
