# Variant of 7ed72f31
# Dimension: D7 — 网格输出形态 (grid_shape)
# Change: 输出与输入同尺寸（18-28的正方形网格），精灵原地变换... → 网格扩展以容纳所有反射

from __future__ import annotations

import random
import numpy as np

from arc_utils import get_bg, label_mask

# Design analysis for the D7 variant
# variant:
#   goal: keep the original reflection rule unchanged, but stop clipping reflected pixels
#     at the input border. The output canvas expands with background fill until every
#     reflected pixel fits.
#   concrete_behavior:
#     input:
#       - still a square 18..28 grid.
#       - same object families as the source task:
#         1. embedded red axis line with one-sided monochrome payload
#         2. isolated red singleton with one adjacent monochrome component
#     output:
#       - contains the entire input grid as an unchanged sub-rectangle.
#       - plus all reflected pixels, even when they would land at r<0/c<0 or beyond
#         the old H-1/W-1 boundary.
#       - the new canvas is the minimal rectangle covering both the original input
#         window [0..H-1]x[0..W-1] and every reflected coordinate.
#       - if some reflections go above/left of the old window, the whole input is
#         translated down/right inside the larger output by exactly -min_row/-min_col.
#   trace_example:
#     original:
#       - a horizontal red axis at row 16 with payload above it at rows 14..15.
#       - reflection wants to paint rows 17..18.
#       - on an 18x18 grid row 18 is out of bounds, so the old solver silently drops it.
#     variant:
#       - the same reflected rows are collected first.
#       - bounds become rows 0..18, so the output grows to 19 rows.
#       - the original 18x18 input remains anchored at the top of the output because
#         min_row is still 0; only one extra background row is appended at the bottom.
#   complexity_calibration:
#     - reflection types stay limited to the original small set (row / col / point).
#     - generator forces at least one genuine overflowing reflection so the new shape
#       rule is visible, but keeps only 3..7 markers and uses padding to avoid clutter.
#     - only a few colors (3..4) are used, with embedded sprites getting distinct colors
#       to avoid the source solver's global same-color span ambiguity.
#   implicit_assumptions_review:
#     - old `out = g.copy()` assumed the output size is fixed. Not valid now.
#     - old `inb(rr, cc)` on reflected pixels assumed clipping is harmless. Not valid now;
#       clipping is the dimension being changed.
#     - the component parsing, axis inference, and isolated-red neighborhood lookup still
#       depend only on the input grid, so those parts remain valid.
#     - the original scan-over-whole-grid payload selection for embedded markers is kept,
#       which means generation must still ensure no same-color stray pixels fall inside a
#       component's sensitive row/column span.
#   placement_legality:
#     - generator places input-visible pixels fully inside the square input canvas.
#     - it separately reserves the whole expected output footprint (original + mirror)
#       with one-cell Chebyshev padding so different sprites never merge or overlap.
#     - non-expanding sprites must keep their full reflection inside the input canvas;
#       designated expanding sprites must have at least one mirror pixel beyond a chosen
#       border.
#   visual_observability:
#     - each sample has at least one reflection crossing the old border, so output shape
#       differs from input shape.
#     - overflow side varies across samples; top/left overflow creates visible background
#       bands before the original grid, making the expansion rule inferable.
#   cleanliness:
#     - sparse placement, small payloads, and no overlap keep the enlarged output readable.
#   color_reasoning:
#     - colors do not encode control flow; they only identify sprites exactly as in the
#       source task.

MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 400

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red (2) markers as reflection axes.

    D7 variant:
    - reflection logic is unchanged
    - reflected pixels are no longer clipped at the original grid border
    - output expands with background fill to contain the original grid and all reflections
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    def inb_input(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    draws: list[tuple[int, int, int]] = []
    min_r, max_r = 0, int(H - 1)
    min_c, max_c = 0, int(W - 1)

    def record(rr: int, cc: int, v: int) -> None:
        nonlocal min_r, max_r, min_c, max_c
        rr = int(rr)
        cc = int(cc)
        draws.append((rr, cc, int(v)))
        min_r = min(int(min_r), rr)
        max_r = max(int(max_r), rr)
        min_c = min(int(min_c), cc)
        max_c = max(int(max_c), cc)

    # --- 1) Mirror payload for marker-containing components ---
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return (
                    int(v) in payload
                    and int(rr0) <= int(r) <= int(rr1)
                    and int(cc0) <= int(c) <= int(cc1)
                )

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                record(int(rr), int(cc), int(v))

    # --- 2) Isolated red pixels mirror adjacent color-components by point reflection ---
    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb_input(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    record(int(rr), int(cc), int(v))

    out = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), int(bg), dtype=int)
    roff = -int(min_r)
    coff = -int(min_c)
    out[int(roff) : int(roff + H), int(coff) : int(coff + W)] = g
    for rr, cc, v in draws:
        out[int(rr + roff), int(cc + coff)] = int(v)
    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _build_embedded_sprite(rng: random.Random) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    is_row = rng.choice([True, False])
    original_before_axis = rng.choice([True, False])

    if is_row:
        H, W = 2 * ph + 1, pw
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        red_pixels = [(ph, c) for c in range(pw)]
        if original_before_axis:
            payload[:ph, :] = pmask
            mirror[ph + 1 :, :] = pmask[::-1, :]
        else:
            payload[ph + 1 :, :] = pmask
            mirror[:ph, :] = pmask[::-1, :]
    else:
        H, W = ph, 2 * pw + 1
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        red_pixels = [(r, pw) for r in range(ph)]
        if original_before_axis:
            payload[:, :pw] = pmask
            mirror[:, pw + 1 :] = pmask[:, ::-1]
        else:
            payload[:, pw + 1 :] = pmask
            mirror[:, :pw] = pmask[:, ::-1]

    return {
        "kind": "embedded",
        "payload": payload,
        "mirror": mirror,
        "red_pixels": red_pixels,
    }


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = (
            np.ones((ph, pw), dtype=bool)
            if n_px == ph * pw
            else _random_connected_mask(ph, pw, n_px, rng=rng)
        )
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
    rng.shuffle(pcoords)
    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = int(pr + dr), int(pc + dc)
            refl = [(int(2 * red_r - r), int(2 * red_c - c)) for r, c in pcoords]
            all_pts = pcoords + refl + [(int(red_r), int(red_c))]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1

            payload = np.zeros((H, W), dtype=bool)
            mirror = np.zeros((H, W), dtype=bool)
            for r, c in pcoords:
                payload[int(r - minr), int(c - minc)] = True
            for r, c in refl:
                mirror[int(r - minr), int(c - minc)] = True
            red_pos = (int(red_r - minr), int(red_c - minc))

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {
                "kind": "isolated",
                "payload": payload,
                "mirror": mirror,
                "red_pixels": [red_pos],
            }
    return None


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _bbox_of_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _finalize_sprite(spr: dict) -> dict:
    color = int(spr["color"])
    payload_pts = _mask_coords(spr["payload"])
    mirror_pts = _mask_coords(spr["mirror"])
    red_pts = [(int(r), int(c)) for r, c in spr["red_pixels"]]
    input_pts = payload_pts + red_pts
    full_pts = payload_pts + mirror_pts + red_pts
    in_bbox = _bbox_of_coords(input_pts)
    full_bbox = _bbox_of_coords(full_pts)

    overflow_sides = set()
    if int(full_bbox[0]) < int(in_bbox[0]):
        overflow_sides.add("top")
    if int(full_bbox[2]) > int(in_bbox[2]):
        overflow_sides.add("bottom")
    if int(full_bbox[1]) < int(in_bbox[1]):
        overflow_sides.add("left")
    if int(full_bbox[3]) > int(in_bbox[3]):
        overflow_sides.add("right")

    spr["payload_pts"] = payload_pts
    spr["mirror_pts"] = mirror_pts
    spr["red_pts"] = red_pts
    spr["input_pts"] = input_pts
    spr["full_pts"] = full_pts
    spr["input_bbox"] = in_bbox
    spr["full_bbox"] = full_bbox
    spr["overflow_sides"] = overflow_sides
    spr["full_area"] = len(full_pts)
    spr["input_cells"] = [(int(r), int(c), color) for r, c in payload_pts] + [
        (int(r), int(c), MARK) for r, c in red_pts
    ]
    spr["full_cells"] = (
        [(int(r), int(c), color) for r, c in payload_pts]
        + [(int(r), int(c), color) for r, c in mirror_pts]
        + [(int(r), int(c), MARK) for r, c in red_pts]
    )
    return spr


def _dilate_points(points: set[tuple[int, int]], padding: int) -> set[tuple[int, int]]:
    if padding <= 0:
        return set(points)
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                out.add((int(r + dr), int(c + dc)))
    return out


def _inside_base(N: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(N) and 0 <= int(c) < int(N)


def _full_inside_base(spr: dict, tr: int, tc: int, N: int) -> bool:
    return all(_inside_base(N, int(r + tr), int(c + tc)) for r, c in spr["full_pts"])


def _overflow_on_side(spr: dict, tr: int, tc: int, N: int, side: str) -> bool:
    for r, c in spr["mirror_pts"]:
        rr = int(r + tr)
        cc = int(c + tc)
        if side == "top" and rr < 0:
            return True
        if side == "bottom" and rr >= int(N):
            return True
        if side == "left" and cc < 0:
            return True
        if side == "right" and cc >= int(N):
            return True
    return False


def _find_location(
    spr: dict,
    N: int,
    forbidden_out: set[tuple[int, int]],
    rng: random.Random,
    *,
    must_overflow_side: str | None,
) -> tuple[int, int] | None:
    ir0, ic0, ir1, ic1 = map(int, spr["input_bbox"])
    tr_lo, tr_hi = -int(ir0), int(N - 1 - ir1)
    tc_lo, tc_hi = -int(ic0), int(N - 1 - ic1)
    if tr_lo > tr_hi or tc_lo > tc_hi:
        return None

    candidates = [(int(tr), int(tc)) for tr in range(tr_lo, tr_hi + 1) for tc in range(tc_lo, tc_hi + 1)]
    rng.shuffle(candidates)
    for tr, tc in candidates:
        shifted_full = {(int(r + tr), int(c + tc)) for r, c in spr["full_pts"]}
        if shifted_full & forbidden_out:
            continue
        if must_overflow_side is None:
            if not _full_inside_base(spr, tr, tc, N):
                continue
        else:
            if not _overflow_on_side(spr, tr, tc, N, must_overflow_side):
                continue
        return int(tr), int(tc)
    return None


def _render_input(sprites: list[dict], N: int, bg: int) -> np.ndarray:
    inp = np.full((N, N), int(bg), dtype=int)
    for spr in sprites:
        tr, tc = map(int, spr["shift"])
        for r, c, v in spr["input_cells"]:
            inp[int(r + tr), int(c + tc)] = int(v)
    return inp


def _render_output(sprites: list[dict], N: int, bg: int) -> np.ndarray:
    all_cells: list[tuple[int, int, int]] = []
    for spr in sprites:
        tr, tc = map(int, spr["shift"])
        for r, c, v in spr["full_cells"]:
            all_cells.append((int(r + tr), int(c + tc), int(v)))
    min_r = min([0] + [r for r, _c, _v in all_cells])
    max_r = max([int(N - 1)] + [r for r, _c, _v in all_cells])
    min_c = min([0] + [c for _r, c, _v in all_cells])
    max_c = max([int(N - 1)] + [c for _r, c, _v in all_cells])
    out = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), int(bg), dtype=int)
    roff, coff = -int(min_r), -int(min_c)
    for r, c, v in all_cells:
        out[int(r + roff), int(c + coff)] = int(v)
    return out


def _assign_colors(sprites: list[dict], palette: list[int], rng: random.Random) -> bool:
    embedded = [spr for spr in sprites if spr["kind"] == "embedded"]
    isolated = [spr for spr in sprites if spr["kind"] == "isolated"]
    min_colors = max(PALETTE_COLORS_MIN, len(embedded))
    max_colors = min(PALETTE_COLORS_MAX, len(palette), len(sprites))
    if min_colors > max_colors:
        return False
    n_colors = rng.randint(min_colors, max_colors)
    chosen = rng.sample(palette, n_colors)

    rng.shuffle(embedded)
    rng.shuffle(isolated)
    used = 0
    for spr in embedded:
        spr["color"] = int(chosen[used])
        used += 1
    unused = chosen[used:]
    for spr in isolated:
        if unused:
            spr["color"] = int(unused.pop(0))
        else:
            spr["color"] = int(rng.choice(chosen))
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    sprites: list[dict] = []
    for builder, count in [(_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)]:
        for _ in range(count):
            spr = builder(rng)
            if spr is None:
                return None
            sprites.append(spr)

    if not _assign_colors(sprites, palette, rng):
        return None
    sprites = [_finalize_sprite(spr) for spr in sprites]

    expand_candidates = [spr for spr in sprites if spr["overflow_sides"]]
    if not expand_candidates:
        return None
    rng.shuffle(expand_candidates)
    n_expand = rng.randint(1, min(3, len(expand_candidates)))
    expanding_ids = {id(spr) for spr in expand_candidates[:n_expand]}
    for spr in sprites:
        if id(spr) in expanding_ids:
            spr["must_overflow_side"] = rng.choice(sorted(spr["overflow_sides"]))
        else:
            spr["must_overflow_side"] = None

    place_order = sorted(
        sprites,
        key=lambda spr: (spr["must_overflow_side"] is None, -int(spr["full_area"])),
    )

    occupied_out: set[tuple[int, int]] = set()
    forbidden_out: set[tuple[int, int]] = set()
    for spr in place_order:
        loc = _find_location(
            spr,
            N,
            forbidden_out,
            rng,
            must_overflow_side=spr["must_overflow_side"],
        )
        if loc is None:
            return None
        spr["shift"] = (int(loc[0]), int(loc[1]))
        full_pts_abs = {(int(r + loc[0]), int(c + loc[1])) for r, c in spr["full_pts"]}
        occupied_out |= full_pts_abs
        forbidden_out |= _dilate_points(full_pts_abs, PLACEMENT_PADDING)

    inp = _render_input(sprites, N, bg)
    if (inp == int(bg)).sum() / float(N * N) < BG_COVERAGE_MIN:
        return None

    expected = _render_output(sprites, N, bg)
    if expected.shape == inp.shape:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
