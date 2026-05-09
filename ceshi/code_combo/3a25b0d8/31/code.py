# Combo variant of 3a25b0d8 (index 31)
# combo_spec: D7/1+D17/2
# Name: 非方形画布+大量空腔
#
# Design analysis
# ===============
# a) Concrete behavior:
#    The seed task copies colors from src cavities into dst holes by positional order.
#    This combo keeps that matching rule, but changes the scene geometry:
#    - there are now 10-15 cavities
#    - the two objects live in an extreme-aspect-ratio canvas
#    - the cavities are arranged almost linearly inside a long strip object
#    So the reasoning collapses from a 2D arrangement into a near-1D sequence.
# b) Complexity calibration:
#    To keep the rule visible, the generator uses one dominant axis with 10-15 ordered
#    slots and only tiny variation on the minor axis. This makes the task harder by
#    quantity and by elongated geometry, without introducing extra hidden signals.
# c) Hidden assumptions reviewed:
#    The original solver searched rotations/flips to reconcile 2D rank keys. That is
#    unnecessary here: generator guarantees src/dst share the same orientation, and the
#    new solver sorts cavities along the dominant axis directly.
# d) Placement legality:
#    Horizontal scenes use a short-tall canvas around 11x34..38 and stack src/dst
#    vertically; vertical scenes use the transpose and place src/dst side-by-side.
# e) Visual observability:
#    Humans can see a long sequence of colored cavities in src and empty cavities in dst.
#    With 10-15 slots and repeated colors, the only stable cue is the linear order.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste


def _components(mask: np.ndarray, *, connectivity: int) -> list[dict]:
    lab, n = label_mask(mask, connectivity=connectivity)
    comps = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        comps.append(
            {
                "cid": int(cid),
                "pts": pts,
                "size": int(len(pts)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
            }
        )
    comps.sort(key=lambda d: int(d["size"]), reverse=True)
    return comps


def _tight_crop_non_bg(sub: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(sub != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _extract_two_objects(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    comps = _components(grid != int(bg), connectivity=8)
    assert len(comps) >= 2
    objs = []
    for comp in comps[:2]:
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
        objs.append(sub)
    return objs


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = sub.shape
    holes = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 == 0 or c0 == 0 or r1 == int(h - 1) or c1 == int(w - 1)):
            continue
        holes.append(
            {
                "pts": pts,
                "size": int(len(pts)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
            }
        )
    return holes


def _outline_col(sub: np.ndarray, bg: int) -> int:
    border_vals = []
    border_vals.extend(int(v) for v in sub[0, :])
    border_vals.extend(int(v) for v in sub[-1, :])
    border_vals.extend(int(v) for v in sub[:, 0])
    border_vals.extend(int(v) for v in sub[:, -1])
    border_non_bg = [int(v) for v in border_vals if int(v) != int(bg)]
    if border_non_bg:
        return int(Counter(border_non_bg).most_common(1)[0][0])
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int, outline_col: int) -> list[dict]:
    mask = (sub != int(bg)) & (sub != int(outline_col))
    lab, n = label_mask(mask, connectivity=4)
    subs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        color = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        subs.append(
            {
                "pts": pts,
                "size": int(len(pts)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "color": int(color),
            }
        )
    return subs


def _linear_order(items: list[dict]) -> list[dict]:
    rs = [float(it["cent"][0]) for it in items]
    cs = [float(it["cent"][1]) for it in items]
    r_span = (max(rs) - min(rs)) if rs else 0.0
    c_span = (max(cs) - min(cs)) if cs else 0.0
    if c_span >= r_span:
        return sorted(items, key=lambda it: (float(it["cent"][1]), float(it["cent"][0]), int(it["size"])))
    return sorted(items, key=lambda it: (float(it["cent"][0]), float(it["cent"][1]), int(it["size"])))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs = _extract_two_objects(g, bg)

    holes_list = [_enclosed_bg_components(obj, bg) for obj in objs]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    outc = _outline_col(src, bg)
    src_subs = _colored_components(src, bg, outc)
    dest_holes = _enclosed_bg_components(dest, bg)
    if not src_subs or not dest_holes:
        return _tight_crop_non_bg(dest, bg)

    src_list = _linear_order(src_subs)
    dest_list = _linear_order(dest_holes)
    for src_comp, hole in zip(src_list, dest_list):
        col = int(src_comp["color"])
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(col)
    return _tight_crop_non_bg(dest, bg)


CANVAS_LONG_RANGE = (34, 38)
CANVAS_SHORT_RANGE = (10, 12)
N_CAVITIES_RANGE = (10, 15)
FILL_COLORS_RANGE = (3, 6)
SLOT_H = 2
SLOT_W_CHOICES = [1, 2]
SLOT_GAP = 1
MAX_GEN_ATTEMPTS = 600


def _random_shape(h: int, w: int) -> np.ndarray:
    mask = np.ones((int(h), int(w)), dtype=bool)
    if int(h * w) >= 3 and random.random() < 0.35:
        rr = random.randrange(int(h))
        cc = random.randrange(int(w))
        mask[rr, cc] = False
        if not mask.any():
            mask[rr, cc] = True
        _, n = label_mask(mask, connectivity=4)
        if int(n) != 1:
            mask[rr, cc] = True
    return mask


def _make_slot_specs(n: int, *, horizontal: bool) -> list[dict]:
    specs = []
    for _ in range(int(n)):
        if horizontal:
            w = int(random.choice(SLOT_W_CHOICES))
            shape = _random_shape(SLOT_H, w)
        else:
            w = int(random.choice(SLOT_W_CHOICES))
            shape = _random_shape(w, SLOT_H)
        specs.append({"span": int(w), "shape": shape})
    return specs


def _build_strip_object(
    colors: list[int],
    slot_specs: list[dict],
    *,
    horizontal: bool,
    filled: bool,
    bg: int,
    outline: int,
) -> np.ndarray:
    n = len(colors)
    slot_ws = [int(spec["span"]) for spec in slot_specs]
    widths_total = sum(int(w) for w in slot_ws) + SLOT_GAP * (n - 1)
    if horizontal:
        inner_h = SLOT_H
        inner_w = widths_total
        canvas = np.full((inner_h + 2, inner_w + 2), int(outline), dtype=int)
        c = 1
        for color, spec in zip(colors, slot_specs):
            shape = np.array(spec["shape"], dtype=bool)
            rr, cc = np.where(shape)
            if filled:
                canvas[1 + rr, c + cc] = int(color)
            else:
                canvas[1 + rr, c + cc] = int(bg)
            c += int(spec["span"]) + SLOT_GAP
    else:
        inner_h = widths_total
        inner_w = SLOT_H
        canvas = np.full((inner_h + 2, inner_w + 2), int(outline), dtype=int)
        r = 1
        for color, spec in zip(colors, slot_specs):
            shape = np.array(spec["shape"], dtype=bool)
            rr, cc = np.where(shape)
            if filled:
                canvas[r + rr, 1 + cc] = int(color)
            else:
                canvas[r + rr, 1 + cc] = int(bg)
            r += int(spec["span"]) + SLOT_GAP
    return canvas


def _place_scene(src: np.ndarray, dst: np.ndarray, *, horizontal: bool, bg: int) -> np.ndarray | None:
    if horizontal:
        canvas_h = random.randint(*CANVAS_SHORT_RANGE)
        canvas_w = random.randint(*CANVAS_LONG_RANGE)
        if src.shape[1] + 2 > canvas_w or dst.shape[1] + 2 > canvas_w:
            return None
        if src.shape[0] + dst.shape[0] + 3 > canvas_h:
            return None
        canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
        src_r = 1
        dst_r = canvas_h - dst.shape[0] - 1
        src_c = random.randint(1, canvas_w - src.shape[1] - 1)
        dst_c = random.randint(1, canvas_w - dst.shape[1] - 1)
        paste(canvas, src, int(src_r), int(src_c), transparent=bg)
        paste(canvas, dst, int(dst_r), int(dst_c), transparent=bg)
        return canvas
    canvas_h = random.randint(*CANVAS_LONG_RANGE)
    canvas_w = random.randint(*CANVAS_SHORT_RANGE)
    if src.shape[0] + 2 > canvas_h or dst.shape[0] + 2 > canvas_h:
        return None
    if src.shape[1] + dst.shape[1] + 3 > canvas_w:
        return None
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    src_c = 1
    dst_c = canvas_w - dst.shape[1] - 1
    src_r = random.randint(1, canvas_h - src.shape[0] - 1)
    dst_r = random.randint(1, canvas_h - dst.shape[0] - 1)
    paste(canvas, src, int(src_r), int(src_c), transparent=bg)
    paste(canvas, dst, int(dst_r), int(dst_c), transparent=bg)
    return canvas


def _scene_metrics(grid: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(grid))
    objs = _extract_two_objects(grid, bg)
    holes_list = [_enclosed_bg_components(obj, bg) for obj in objs]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    src = objs[src_i]
    dest = objs[dest_i]
    outc = _outline_col(src, bg)
    src_subs = _colored_components(src, bg, outc)
    dest_holes = _enclosed_bg_components(dest, bg)
    rs = [float(it["cent"][0]) for it in src_subs]
    cs = [float(it["cent"][1]) for it in src_subs]
    r_span = (max(rs) - min(rs)) if rs else 0.0
    c_span = (max(cs) - min(cs)) if cs else 0.0
    return {
        "n_cavities": int(len(src_subs)),
        "linear_strength": int(max(r_span, c_span) - min(r_span, c_span)),
        "is_horizontal": int(c_span >= r_span),
        "dst_holes": int(len(dest_holes)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    outline = int(palette[0])
    n_colors = random.randint(*FILL_COLORS_RANGE)
    fill_pool = [int(c) for c in palette[1 : 1 + n_colors]]
    n = random.randint(*N_CAVITIES_RANGE)

    colors = [int(random.choice(fill_pool)) for _ in range(n)]
    if len(set(colors)) == 1 and len(fill_pool) > 1:
        colors[random.randrange(n)] = int(fill_pool[-1])

    horizontal = bool(random.getrandbits(1))
    slot_specs = _make_slot_specs(n, horizontal=horizontal)
    src = _build_strip_object(colors, slot_specs, horizontal=horizontal, filled=True, bg=bg, outline=outline)
    dst = _build_strip_object(colors, slot_specs, horizontal=horizontal, filled=False, bg=bg, outline=outline)
    canvas = _place_scene(src, dst, horizontal=horizontal, bg=bg)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    expected = _tight_crop_non_bg(dst.copy().astype(int), bg)
    ordered_holes = _linear_order(_enclosed_bg_components(dst, bg))
    for color, hole in zip(colors, ordered_holes):
        for rr, cc in hole["pts"]:
            expected[int(rr), int(cc)] = int(color)
    expected = _tight_crop_non_bg(expected, bg)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None

    metrics = _scene_metrics(canvas)
    if metrics["n_cavities"] < 10 or metrics["dst_holes"] < 10:
        return None
    if metrics["linear_strength"] < 6:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
