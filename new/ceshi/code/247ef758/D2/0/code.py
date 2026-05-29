# Variant of 247ef758
# Dimension: D2 — 源对象：精灵类型与属性 (src)
# Change: 左侧窄条中2-3个可移动精灵（8连通单色色块，3-9像素，外接矩形≤5x5）加0-1个干扰精灵，颜色各不相同... → 多色精灵
#
# Design analysis
# ===============
# source_task:
#   none
# variant_spec:
#   grid: unchanged from original, 10-15 rows x 14-20 cols
#   layout: unchanged: left sprite strip | divider column | rectangular frame on right
#   sprites:
#     count: 2-3 movers + 0-1 distractors
#     connectivity: each sprite is one 8-connected component in the union-of-non-bg sense
#     colors: each sprite uses exactly 2 colors
#     primary_color: strict majority color within the sprite; unique per sprite
#     accent_color: shared secondary color, never used by divider/frame/markers
#     output rule: match markers by primary_color, but stamp the full two-color pattern
#   frame:
#     unchanged geometry and marker logic
#     markers use only primary colors, so the new required inference step is:
#       identify each multi-color sprite -> count its colors -> choose the majority color -> match that color to the frame
# concrete_trace:
#   original:
#     a 5-colored single-color blob is detected by looking at connected components inside (g == 5),
#     the same color 5 is used both for target matching and for all stamped pixels.
#   variant:
#     suppose a sprite contains 5 blue pixels and 2 yellow pixels.
#     The solver must first detect the whole 7-pixel multi-color component, then infer "blue" as the main color.
#     If the frame has blue row/col markers, the solver stamps the exact blue+yellow pattern at those targets.
#     The yellow pixels appear in the output interior even though yellow is absent from the frame markers.
# complexity_calibration:
#   keeping the frame and target rule unchanged preserves the original puzzle skeleton.
#   The non-trivial change is localized but visible: colors inside the moving objects now carry internal structure.
#   To avoid ambiguity, every sprite has a unique primary color and a strict majority; only the accent color is shared.
#   This keeps the color vocabulary small enough for human inference while making the D2 change obvious.
# hidden_assumption_review:
#   original sprite extraction loop iterates "per color", which only works because each sprite was monochrome.
#   In the variant this would split one sprite into two fake components, so extraction must instead label all
#   non-bg pixels in the left strip at once.
#   original move_colors used the sprite color directly; now it must use the inferred primary_color.
#   original stamping wrote int(col) for every moved pixel; now each stored pixel keeps its own color.
#   original generator could stack sprites flush vertically because per-color extraction kept them separate.
#   In the variant that would merge adjacent two-color sprites into one 8-connected component, so generator
#   must enforce at least one blank row between sprites.
# placement_legality:
#   row/col markers are still chosen so each sprite bbox fits fully when centered on every target.
#   The left-strip placer enforces one blank row between neighboring sprites, preventing accidental component merging.
#   The accent color is never used by frame edges or the divider, so target matching cannot accidentally key off it.
# observability:
#   humans can see the new rule because frame markers show only one of the two sprite colors, while the output
#   reproduces both colors. That makes "majority color decides the destination" the only consistent explanation.
# cleanliness:
#   the shared accent color reduces palette clutter, while the frame/divider remain visually distinct.
# color_inference:
#   no numeric ordering of color ids is used semantically.
#   The only color-dependent decision is visible counting inside each sprite: the more frequent sprite color is the key.
# new_solver_assumptions:
#   every movable/distractor sprite in the left strip is an isolated 8-connected component.
#   every such component has a unique dominant color.
#   generate_input() guarantees both conditions by construction.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _dominant_color(vals: list[int]) -> int:
    counts = Counter(map(int, vals))
    return sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0]


def _extract_sprites(g: np.ndarray, bg: int, fc0: int, divider_mask: np.ndarray) -> list[dict]:
    left_mask = (np.arange(g.shape[1])[None, :] < int(fc0))
    sprite_mask = (g != int(bg)) & left_mask & (~divider_mask)
    slab, sn = label_mask(sprite_mask, connectivity=8)

    sprites: list[dict] = []
    for sid in range(1, int(sn) + 1):
        pts = np.argwhere(slab == int(sid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        comp = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
        patch = np.where(comp, g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], int(bg))
        colors = [int(v) for v in patch[comp]]
        if not colors:
            continue
        sprites.append(
            {
                "main_color": int(_dominant_color(colors)),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "pattern": patch.astype(int),
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Left-strip sprites are two-color 8-connected components.
    - The dominant color (strict majority inside the sprite) is used to match frame markers.
    - The full two-color sprite is stamped at every target; lower sprites stamp first, upper sprites cover later.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    sprites = _extract_sprites(g, bg, fc0, divider_mask)
    move_colors = {int(s["main_color"]) for s in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["main_color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        pat = np.array(s["pattern"], dtype=int)
        for rr, cc in np.argwhere(pat != int(bg)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        key = int(s["main_color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pat = np.array(s["pattern"], dtype=int)
        pix = [(int(rr), int(cc2), int(pat[int(rr), int(cc2)])) for rr, cc2 in np.argwhere(pat != int(bg))]

        for tr, tc in targets[int(key)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for rr0, cc0, color in pix:
                rr = int(s["r0"]) + int(rr0) + int(dr)
                cc3 = int(s["c0"]) + int(cc0) + int(dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(color)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_shape_mask(max_h: int, max_w: int) -> np.ndarray | None:
    sh = min(random.choice([2, 3, 3, 4]), max_h)
    sw = min(random.choice([2, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None

    max_pixels = min(9, sh * sw)
    min_pixels = min(3, max_pixels)
    if max_pixels < 3:
        return None
    n_pixels = random.randint(min_pixels, max_pixels)

    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 40):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            pixels.add((nr, nc))
            r, c = nr, nc

    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for pr, pc in pixels:
        mask[pr - min_r, pc - min_c] = True
    return mask


def _sample_connected_subset(coords: list[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    start = random.choice(coords)
    chosen = {start}
    frontier = [start]
    while len(chosen) < int(k):
        base = random.choice(frontier)
        nbrs = [
            (nr, nc)
            for nr in range(base[0] - 1, base[0] + 2)
            for nc in range(base[1] - 1, base[1] + 2)
            if (nr, nc) in coords and (nr, nc) not in chosen and not (nr == base[0] and nc == base[1])
        ]
        if nbrs:
            nxt = random.choice(nbrs)
            chosen.add(nxt)
            frontier.append(nxt)
            continue
        frontier.append(random.choice(list(chosen)))
        if len(frontier) > len(coords) * 4:
            break
    if len(chosen) < int(k):
        remaining = [p for p in coords if p not in chosen]
        random.shuffle(remaining)
        for p in remaining:
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in chosen):
                chosen.add(p)
                if len(chosen) >= int(k):
                    break
    return chosen


def _make_multicolor_sprite(primary: int, accent: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a two-color 8-connected sprite cropped to bbox; primary color is the strict majority."""
    mask = _make_shape_mask(max_h, max_w)
    if mask is None:
        return None
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    area = len(coords)
    if area < 3:
        return None

    max_accent = min(3, (area - 1) // 2)
    if max_accent < 1:
        return None
    min_accent = 2 if area >= 5 and max_accent >= 2 else 1
    n_accent = random.randint(min_accent, max_accent)
    accent_cells = _sample_connected_subset(coords, n_accent)

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        sprite[int(r), int(c)] = int(accent if (r, c) in accent_cells else primary)
    return sprite


def _sample_gaps(total_extra: int, n_slots: int) -> list[int]:
    gaps = [0] * int(n_slots)
    for _ in range(int(total_extra)):
        gaps[random.randrange(int(n_slots))] += 1
    return gaps


def _variant_secondary_footprint(inp: np.ndarray, out: np.ndarray, accent_color: int) -> int:
    return int(np.count_nonzero((out == int(accent_color)) & (inp != int(accent_color))))


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    n_total = n_movers + n_distractors

    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    accent_color = all_colors[idx]
    idx += 1
    primary_colors = all_colors[idx : idx + n_total]
    mover_colors = primary_colors[:n_movers]
    distractor_colors = primary_colors[n_movers:]

    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_multicolor_sprite(mc, accent_color, 4, sprite_area_w)
        if sp is None:
            return None
        mover_sprites.append(sp)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc, sp in zip(mover_colors, mover_sprites):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None

        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), sh)
        if not row_marks:
            return None
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), sw)
        if not col_marks:
            return None

        mover_targets[int(mc)] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(map(int, row_marks))
        occupied_cols.update(map(int, col_marks))

    all_items: list[tuple[int, np.ndarray, bool]] = [(int(mc), sp, False) for mc, sp in zip(mover_colors, mover_sprites)]
    for dc in distractor_colors:
        ds = _make_multicolor_sprite(int(dc), accent_color, 4, sprite_area_w)
        if ds is None:
            return None
        all_items.append((int(dc), ds, True))

    min_needed_h = sum(int(sp.shape[0]) for _, sp, _ in all_items) + max(0, len(all_items) - 1)
    if min_needed_h > H:
        return None

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - int(fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = int(frame_base)
        for mc, (rmarks, _) in mover_targets.items():
            if int(r) in rmarks:
                val = int(mc)
                break
        inp[r, fc0] = int(val)
        inp[r, fc1] = int(val)

    extra_rows = H - min_needed_h
    gap_extras = _sample_gaps(extra_rows, len(all_items) + 1)
    cur_row = gap_extras[0]
    for idx_item, (_primary, sp, _is_distractor) in enumerate(all_items):
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        inp[cur_row : cur_row + sh, c_off : c_off + sw] = np.where(sp != int(bg), sp, inp[cur_row : cur_row + sh, c_off : c_off + sw])
        cur_row += sh
        if idx_item < len(all_items) - 1:
            cur_row += 1 + gap_extras[idx_item + 1]

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None

    if _variant_secondary_footprint(inp, output, int(accent_color)) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
