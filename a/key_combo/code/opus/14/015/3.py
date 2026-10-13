from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

HORIZONTAL_BARS = [(RED, 2), (GREEN, 3)]
VERTICAL_BARS = [(BLUE, 3), (ORANGE, 4)]
BASE_CROSS_DIMS = [(2, 3), (2, 4), (3, 3), (3, 4)]
ROTATABLE_KEYS = {(2, 3), (2, 4), (3, 4)}

GEN_RETRIES = 400
SLOT_CYCLE: list[int] = []
PATTERN_CYCLE: list[int] = []
GEN_CALLS = 0


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in map(int, indices[1:]):
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    return groups


def _analysis(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_rows = np.flatnonzero(np.all(g != BLACK, axis=1))
    full_cols = np.flatnonzero(np.all(g != BLACK, axis=0))
    row_bands = _bands(full_rows)
    col_bands = _bands(full_cols)
    assert len(row_bands) == 2, "expected exactly two full-width row bands"
    assert len(col_bands) == 2, "expected exactly two full-height column bands"

    row_sort = sorted(row_bands)
    col_sort = sorted(col_bands)
    outside_cols = np.ones(g.shape[1], dtype=bool)
    outside_cols[full_cols] = False
    outside_rows = np.ones(g.shape[0], dtype=bool)
    outside_rows[full_rows] = False
    assert np.any(outside_cols), "need columns outside the vertical bars"
    assert np.any(outside_rows), "need rows outside the horizontal bars"

    hbars = []
    for r0, r1 in row_sort:
        band = g[r0 : r1 + 1][:, outside_cols]
        colors = np.unique(band)
        colors = colors[colors != BLACK]
        assert colors.size == 1, "horizontal band must have one own color outside crossings"
        hbars.append(
            {
                "rows": (int(r0), int(r1)),
                "height": int(r1 - r0 + 1),
                "color": int(colors[0]),
            }
        )

    vbars = []
    for c0, c1 in col_sort:
        band = g[outside_rows][:, c0 : c1 + 1]
        colors = np.unique(band)
        colors = colors[colors != BLACK]
        assert colors.size == 1, "vertical band must have one own color outside crossings"
        vbars.append(
            {
                "cols": (int(c0), int(c1)),
                "width": int(c1 - c0 + 1),
                "color": int(colors[0]),
            }
        )

    cross_by_key: dict[tuple[int, int], dict] = {}
    for hi, hbar in enumerate(hbars):
        hr0, hr1 = hbar["rows"]
        for vi, vbar in enumerate(vbars):
            vc0, vc1 = vbar["cols"]
            region = g[hr0 : hr1 + 1, vc0 : vc1 + 1]
            visible_colors = np.unique(region)
            assert visible_colors.size == 1, "crossing must be monochrome"
            visible = int(visible_colors[0])
            hc = int(hbar["color"])
            vc = int(vbar["color"])
            assert visible in {hc, vc}, "crossing color must belong to one of the two bars"
            hidden = vc if visible == hc else hc
            key = tuple(sorted((int(hbar["height"]), int(vbar["width"]))))
            assert key not in cross_by_key, "cross dimensions must be unique"
            cross_by_key[key] = {
                "visible": visible,
                "hidden": hidden,
                "horizontal_on_top": visible == hc,
                "h_index": hi,
                "v_index": vi,
            }

    grey_components = _components(g == GREY)
    assert len(grey_components) == 1, "expected one grey object"
    obj_coords = grey_components[0]
    obj_box = _bbox(obj_coords)
    crop = _crop_bbox(g, obj_box)
    assert np.all((crop == GREY) | (crop == BLACK)), "grey object bbox must contain only grey and black"

    hole_components = []
    for comp in _components(crop == BLACK):
        if (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == crop.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == crop.shape[1] - 1)
        ):
            continue
        hole_components.append(comp)

    holes = []
    seen_keys: set[tuple[int, int]] = set()
    for comp in hole_components:
        r0, c0, r1, c1 = _bbox(comp)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        assert int(comp.shape[0]) == hh * ww, "every hole must be rectangular"
        key = tuple(sorted((hh, ww)))
        assert key not in seen_keys, "hole dimensions must be unique"
        seen_keys.add(key)
        holes.append(
            {
                "coords_local": comp,
                "coords_abs": comp + np.array([[obj_box[0], obj_box[1]]], dtype=int),
                "dims": (hh, ww),
                "key": key,
            }
        )

    return {
        "hbars": hbars,
        "vbars": vbars,
        "cross_by_key": cross_by_key,
        "object_box": obj_box,
        "holes": holes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey rectangular hole with the hidden color from the crossing rectangle that has the same dimensions.

    Input structure:
    - A black background contains two full-width horizontal bars and two full-height vertical bars, each with its own thickness and color.
    - Every bar pair forms one monochrome crossing rectangle whose visible color belongs to the upper bar in that local overlap.
    - Away from the bars there is one connected grey object whose bounding box contains exactly four rectangular black holes.

    Rule:
    - Detect the two horizontal bars from the contiguous all-nonblack row bands and the two vertical bars from the contiguous all-nonblack column bands.
    - For each crossing, record its rectangle size using horizontal height x vertical width, and infer the hidden color as the other bar color not currently visible there.
    - For each rectangular hole in the grey object, measure its height and width and match it to the unique crossing with the same dimensions, allowing a 90-degree rotation.
    - Fill that hole with the matched crossing's hidden color and leave all other pixels unchanged.

    Key insight:
    - The holes are keyed by rectangle dimensions, but the fill color is not the visible crossing color: it is the color of the bar that is occluded at that crossing.
    """
    g = np.asarray(grid, dtype=int)
    info = _analysis(g)

    expected_keys = {tuple(sorted(dim)) for dim in BASE_CROSS_DIMS}
    assert {bar["height"] for bar in info["hbars"]} == {2, 3}
    assert {bar["width"] for bar in info["vbars"]} == {3, 4}
    assert set(info["cross_by_key"]) == expected_keys
    assert len(info["holes"]) == 4
    assert {hole["key"] for hole in info["holes"]} == expected_keys

    out = g.copy()
    for hole in info["holes"]:
        fill_color = int(info["cross_by_key"][hole["key"]]["hidden"])
        coords = hole["coords_abs"]
        out[coords[:, 0], coords[:, 1]] = fill_color
    return out


def _next_cycle(values: list[int], store_name: str) -> int:
    global SLOT_CYCLE, PATTERN_CYCLE
    store = SLOT_CYCLE if store_name == "slot" else PATTERN_CYCLE
    if not store:
        store.extend(values)
        random.shuffle(store)
    return store.pop()


def _choose_orientations(call_index: int) -> list[tuple[int, int]]:
    rotatable = [dim for dim in BASE_CROSS_DIMS if dim in ROTATABLE_KEYS]
    rotate_count = 1 + (call_index % 2)
    rotated_subset = set(random.sample(rotatable, k=rotate_count))
    dims = []
    rotated = 0
    unrotated = 0
    for h, w in BASE_CROSS_DIMS:
        if (h, w) in rotated_subset:
            dims.append((w, h))
            rotated += 1
        else:
            dims.append((h, w))
            if h != w:
                unrotated += 1
    assert rotated >= 1 and unrotated >= 1
    return dims


def _make_object_patch(call_index: int) -> np.ndarray:
    for _ in range(100):
        hole_dims = _choose_orientations(call_index)
        assigned = random.sample(hole_dims, k=4)
        tl, tr, bl, br = assigned

        top_h = max(tl[0], tr[0]) + random.randint(0, 1)
        bottom_h = max(bl[0], br[0]) + random.randint(0, 1)
        left_w = max(tl[1], bl[1]) + random.randint(0, 1)
        right_w = max(tr[1], br[1]) + random.randint(0, 1)

        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 2)
        row_gap = random.randint(1, 2)
        col_gap = random.randint(1, 2)

        h = top_margin + top_h + row_gap + bottom_h + bottom_margin
        w = left_margin + left_w + col_gap + right_w + right_margin
        if not (9 <= h <= 14 and 9 <= w <= 14):
            continue

        patch = np.full((h, w), GREY, dtype=int)
        cells = {
            "tl": (top_margin, left_margin, top_h, left_w, tl),
            "tr": (top_margin, left_margin + left_w + col_gap, top_h, right_w, tr),
            "bl": (top_margin + top_h + row_gap, left_margin, bottom_h, left_w, bl),
            "br": (
                top_margin + top_h + row_gap,
                left_margin + left_w + col_gap,
                bottom_h,
                right_w,
                br,
            ),
        }
        for origin_r, origin_c, cell_h, cell_w, (hole_h, hole_w) in cells.values():
            rr = origin_r + random.randint(0, cell_h - hole_h)
            cc = origin_c + random.randint(0, cell_w - hole_w)
            patch[rr : rr + hole_h, cc : cc + hole_w] = BLACK

        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if random.random() < 0.5:
            patch = np.flipud(patch)
        return patch

    raise RuntimeError("failed to sample a bounded grey object patch")


def _cross_specs(pattern_id: int) -> dict[tuple[int, int], dict[str, int]]:
    pattern = (
        ((True, False), (False, True))
        if pattern_id == 0
        else ((False, True), (True, False))
    )
    out: dict[tuple[int, int], dict[str, int]] = {}
    for hi, (h_color, h_height) in enumerate(HORIZONTAL_BARS):
        for vi, (v_color, v_width) in enumerate(VERTICAL_BARS):
            horizontal_on_top = bool(pattern[hi][vi])
            visible = h_color if horizontal_on_top else v_color
            hidden = v_color if horizontal_on_top else h_color
            out[tuple(sorted((h_height, v_width)))] = {
                "visible": int(visible),
                "hidden": int(hidden),
            }
    return out


def _place_bars_and_object(patch: np.ndarray, slot: int, pattern_id: int) -> np.ndarray | None:
    ph, pw = patch.shape

    top_space = random.randint(1, 3)
    bottom_space = random.randint(1, 3)
    middle_space = ph + random.randint(2, 4)
    H = top_space + 2 + middle_space + 3 + bottom_space

    side_a = random.randint(1, 3)
    side_b = random.randint(1, 3)
    slot_space = pw + random.randint(2, 4)
    if slot == 0:
        left_space, middle_cols, right_space = slot_space, side_a, side_b
    elif slot == 1:
        left_space, middle_cols, right_space = side_a, slot_space, side_b
    else:
        left_space, middle_cols, right_space = side_a, side_b, slot_space
    W = left_space + 3 + middle_cols + 4 + right_space
    if H > 30 or W > 30:
        return None

    r_red = top_space
    r_green = top_space + 2 + middle_space
    c_blue = left_space
    c_orange = left_space + 3 + middle_cols

    row_open_start = r_red + 2
    row_open_height = middle_space
    if slot == 0:
        col_open_start, col_open_width = 0, left_space
    elif slot == 1:
        col_open_start, col_open_width = c_blue + 3, middle_cols
    else:
        col_open_start, col_open_width = c_orange + 4, right_space

    row_slack = row_open_height - ph
    col_slack = col_open_width - pw
    assert row_slack >= 2 and col_slack >= 2
    r_obj = row_open_start + random.randint(1, row_slack - 1)
    c_obj = col_open_start + random.randint(1, col_slack - 1)

    g = np.zeros((H, W), dtype=int)
    g[r_red : r_red + 2, :] = RED
    g[r_green : r_green + 3, :] = GREEN
    g[:, c_blue : c_blue + 3] = BLUE
    g[:, c_orange : c_orange + 4] = ORANGE

    pattern = (
        ((True, False), (False, True))
        if pattern_id == 0
        else ((False, True), (True, False))
    )
    row_starts = [r_red, r_green]
    col_starts = [c_blue, c_orange]
    for hi, (_hc, hh) in enumerate(HORIZONTAL_BARS):
        for vi, (_vc, vw) in enumerate(VERTICAL_BARS):
            visible = HORIZONTAL_BARS[hi][0] if pattern[hi][vi] else VERTICAL_BARS[vi][0]
            rr = row_starts[hi]
            cc = col_starts[vi]
            g[rr : rr + hh, cc : cc + vw] = int(visible)

    g[r_obj : r_obj + ph, c_obj : c_obj + pw] = patch
    return g


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        info = _analysis(grid)
        out = main(grid)
    except Exception:
        return False

    expected_keys = {tuple(sorted(dim)) for dim in BASE_CROSS_DIMS}
    if set(info["cross_by_key"]) != expected_keys:
        return False
    if {bar["height"] for bar in info["hbars"]} != {2, 3}:
        return False
    if {bar["width"] for bar in info["vbars"]} != {3, 4}:
        return False
    if {bar["color"] for bar in info["hbars"]} != {RED, GREEN}:
        return False
    if {bar["color"] for bar in info["vbars"]} != {BLUE, ORANGE}:
        return False
    if len(info["holes"]) != 4 or {hole["key"] for hole in info["holes"]} != expected_keys:
        return False

    if sum(int(meta["horizontal_on_top"]) for meta in info["cross_by_key"].values()) != 2:
        return False
    if {int(meta["hidden"]) for meta in info["cross_by_key"].values()} != {BLUE, RED, GREEN, ORANGE}:
        return False

    rotated = 0
    unrotated = 0
    for hole in info["holes"]:
        key = hole["key"]
        dims = tuple(hole["dims"])
        if key in ROTATABLE_KEYS:
            canonical = next(dim for dim in BASE_CROSS_DIMS if tuple(sorted(dim)) == key)
            if dims == canonical:
                unrotated += 1
            else:
                rotated += 1
    if rotated < 1 or unrotated < 1:
        return False

    changed = out != grid
    object_box = info["object_box"]
    r0, c0, r1, c1 = object_box
    changed_outside = changed.copy()
    changed_outside[r0 : r1 + 1, c0 : c1 + 1] = False
    if np.any(changed_outside):
        return False
    if not np.all(grid[changed] == BLACK):
        return False

    visible_fill = grid.copy()
    for hole in info["holes"]:
        key = hole["key"]
        hidden = int(info["cross_by_key"][key]["hidden"])
        visible = int(info["cross_by_key"][key]["visible"])
        coords = hole["coords_abs"]
        if not np.all(out[coords[:, 0], coords[:, 1]] == hidden):
            return False
        visible_fill[coords[:, 0], coords[:, 1]] = visible
    if np.array_equal(visible_fill, out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    global GEN_CALLS
    GEN_CALLS += 1
    try:
        patch = _make_object_patch(GEN_CALLS)
    except RuntimeError:
        return None
    slot = _next_cycle([0, 1, 2], "slot")
    pattern_id = _next_cycle([0, 1], "pattern")
    grid = _place_bars_and_object(patch, slot, pattern_id)
    if grid is None:
        return None
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-dimension hole-filling input")
