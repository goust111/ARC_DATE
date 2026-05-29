# Combo variant of 3a25b0d8 (index 8)
# combo_spec: D2/1+D4/0
# Name: 嵌套框架源+颜色映射填充

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


# Design analysis
# =============================================================================
# a) Concrete behavior / trace
#    Base 3a25b0d8:
#    - src = one outlined object containing several colored cavity-fills.
#    - dst = another outlined object with empty holes in the same centroid-rank
#      arrangement.
#    - Solver extracts src colors, matches by (row_rank, col_rank), and copies
#      those colors directly into dst holes.
#
#    Combo D2/1 + D4/0:
#    - src is no longer a set of scattered cavity blobs. It is a nested-frame
#      object: several offset rectangular outline frames are stacked inward, and
#      each inter-frame band is filled by a single color. Because the inner
#      rectangles are offset rather than perfectly concentric, the colored bands
#      have different centroids and remain rank-matchable.
#    - A separate legend object gives a visible color mapping. Each row encodes
#      `outline | src_color | outline | mapped_color | outline`.
#    - dst is an outline-only nested-frame object with the same centroid-rank
#      order of holes. Output fills dst's holes by matched src layer, but after
#      legend remapping.
#
#    Tiny example:
#    - src outer->inner layer colors = [red, green, yellow, teal].
#    - legend rows say red->blue, green->maroon, yellow->orange, teal->pink.
#    - dst holes are matched to those four nested bands by centroid ranks.
#    - Original task would output [red, green, yellow, teal].
#    - Combo output becomes [blue, maroon, orange, pink].
#
# b) Complexity calibration
#    - The source remains one 8-connected object, but the informative parts are
#      now nested color bands rather than isolated interior blobs. This is a
#      visible structural change: the human must read depth/layer order rather
#      than a flat bag of cavities.
#    - The legend uses exactly four rows because four nested layers are the
#      largest layout that fits reliably inside ARC's 30x30 canvas together with
#      dst and a separate legend. Using fewer than four would make the nested
#      change much less salient; using more would force objects too large to
#      place legally.
#    - The mapping is a derangement on the four layer colors, so direct-copy and
#      mapped-fill outputs are always different.
#
# c) Hidden assumptions broken in the original solver
#    - `_extract_two_objects()` assumed only src/dst exist. Broken by the legend.
#      New solver scans all 8-connected components and explicitly parses the
#      legend card.
#    - `_outline_col()` used "most frequent non-bg color". Broken here because a
#      wide nested band can outnumber the outline. New logic reads the dominant
#      non-bg color on the component border; the generator guarantees every
#      object is tight-cropped to an outer outline frame, so the border is pure
#      outline color.
#    - "src informative pieces are generic cavity blobs" no longer holds. The
#      new src pieces are annular bands between nested frames. The good news is
#      that once outline color is identified correctly, the same 4-connected
#      component extraction still recovers one component per band.
#    - "final fill color equals src color" is broken by D4/0; now color must be
#      passed through the legend map before painting dst.
#
# d) Placement legality
#    - Generator places exactly three objects: src, dst, legend.
#    - All are pasted with the shared non-overlap helper, one-cell canvas border,
#      and one-cell inter-object padding, so components cannot merge.
#    - Four-layer nested objects are kept at 12-13 cells per side so that two of
#      them plus a 5-column legend still fit inside a 26-30 canvas.
#
# e) Visual observability
#    - Across examples, src visibly looks like "frame inside frame inside frame"
#      with four colored bands; dst is the same style but hollow.
#    - The legend is a small isolated card, so the remapping cue is explicit.
#    - Because the mapping is always non-identity, the output colors cannot be
#      explained by direct copying alone.
#
# f) Visual cleanliness
#    - No overlay annotations are drawn on top of src/dst. The three semantic
#      regions are spatially separated: src, dst, legend.
#    - Within src/dst, only outline and one layer color appear in each band, so
#      the nested structure remains readable.
#
# g) Change inferability
#    - The nested structure gives a strong structural cue for "outer layer,
#      middle layer, inner layer ...".
#    - The legend gives a strong symbolic cue for color change.
#    - Only four layer colors are used, so each example shows enough repeated
#      evidence without overwhelming the viewer.
#
# 9) New solver assumptions and generator guarantees
#    - The legend follows the fixed 5-column schema and has no background inside.
#      `_parse_legend()` depends on this; generator enforces it.
#    - After removing the legend, exactly two large objects remain. Generator
#      always places one src and one dst only.
#    - The border color of src/dst/legend is the outline color. Generator crops
#      each patch tightly to its outer frame before placement.
#    - Src's non-outline 4-connected components form a strict bbox-containment
#      chain (outer band contains next band, etc.). Generator validates this so
#      the "nested source" behavior cannot silently degenerate back to a flat
#      cavity layout.
#    - All four src colors appear in the legend, and every legend mapping changes
#      color. Generator validates both and also rejects any sample whose mapped
#      output equals the direct-fill output.


CANVAS_SIZE_RANGE = (26, 30)
OBJECT_H_RANGE = (12, 13)
OBJECT_W_RANGE = (12, 13)
LAYER_COUNT = 4
INNER_SIZE_CHOICES = [3, 4]
DRIFT_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
MAX_GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[dict]:
    lab, n = label_mask(mask, connectivity=8)
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
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    comps.sort(key=lambda d: int(d["size"]), reverse=True)
    return comps


def _bbox_of(group: list[dict]) -> tuple[int, int, int, int]:
    r0 = min(int(d["bbox"][0]) for d in group)
    c0 = min(int(d["bbox"][1]) for d in group)
    r1 = max(int(d["bbox"][2]) for d in group)
    c1 = max(int(d["bbox"][3]) for d in group)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_group_subgrid(grid: np.ndarray, group: list[dict], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of(group)
    sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for d in group:
        for rr, cc in d["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
    return sub


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
                "cent": (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return holes


def _outline_col(sub: np.ndarray, bg: int) -> int:
    border = np.concatenate(
        [
            sub[0, :],
            sub[-1, :],
            sub[1:-1, 0] if sub.shape[0] > 2 else np.array([], dtype=int),
            sub[1:-1, -1] if sub.shape[0] > 2 else np.array([], dtype=int),
        ]
    )
    border_cols = [int(c) for c in border if int(c) != int(bg)]
    if border_cols:
        return int(Counter(border_cols).most_common(1)[0][0])
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    parts = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        parts.append(
            {
                "pts": pts,
                "size": int(len(pts)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cent": (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))),
                "color": int(col),
            }
        )
    return parts


def _rank_keys(pts: list[tuple[int, int]]):
    rows = sorted(set(int(r) for r, _ in pts))
    cols = sorted(set(int(c) for _, c in pts))
    rrank = {r: i for i, r in enumerate(rows)}
    crank = {c: i for i, c in enumerate(cols)}
    keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
    return keys, rrank, crank


def _transform_point(p: tuple[int, int], h0: int, w0: int, k: int, flip: bool) -> tuple[int, int]:
    r, c = int(p[0]), int(p[1])
    if int(k) == 0:
        rr, cc = int(r), int(c)
        h2, w2 = int(h0), int(w0)
    elif int(k) == 1:
        rr, cc = int(w0 - 1 - c), int(r)
        h2, w2 = int(w0), int(h0)
    elif int(k) == 2:
        rr, cc = int(h0 - 1 - r), int(w0 - 1 - c)
        h2, w2 = int(h0), int(w0)
    else:
        rr, cc = int(c), int(h0 - 1 - r)
        h2, w2 = int(w0), int(h0)
    if bool(flip):
        cc = int(w2 - 1 - cc)
    return int(rr), int(cc)


def _parse_legend(sub: np.ndarray, bg: int) -> dict[int, int] | None:
    outc = _outline_col(sub, bg)
    h, w = sub.shape
    if int(w) != 5 or int(h) < 5:
        return None
    if np.any(sub == int(bg)):
        return None
    if not np.all(sub[0, :] == int(outc)) or not np.all(sub[-1, :] == int(outc)):
        return None
    if not np.all(sub[:, 0] == int(outc)) or not np.all(sub[:, -1] == int(outc)):
        return None
    if not np.all(sub[:, 2] == int(outc)):
        return None
    mapping: dict[int, int] = {}
    seen_dst: set[int] = set()
    for r in range(1, int(h - 1)):
        src_col = int(sub[int(r), 1])
        dst_col = int(sub[int(r), 3])
        if src_col in (int(bg), int(outc)) or dst_col in (int(bg), int(outc)):
            return None
        if src_col in mapping or dst_col in seen_dst or int(src_col) == int(dst_col):
            return None
        mapping[int(src_col)] = int(dst_col)
        seen_dst.add(int(dst_col))
    return mapping if mapping else None


def _tight_crop_non_bg(sub: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(sub != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _solve_scene(grid: np.ndarray, *, apply_map: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = _components(g != int(bg))
    infos = []
    for comp in comps:
        sub = _extract_group_subgrid(g, [comp], bg)
        infos.append(
            {
                "sub": sub,
                "size": int(comp["size"]),
                "holes": _enclosed_bg_components(sub, bg),
                "legend_map": _parse_legend(sub, bg),
            }
        )

    legend_candidates = [i for i, info in enumerate(infos) if info["legend_map"]]
    legend_idx = None
    if legend_candidates:
        legend_idx = min(
            legend_candidates,
            key=lambda i: (int(infos[i]["size"]), -len(infos[i]["legend_map"])),
        )
    color_map = infos[legend_idx]["legend_map"] if legend_idx is not None else {}

    object_infos = [info for i, info in enumerate(infos) if i != legend_idx]
    object_infos.sort(key=lambda d: int(d["size"]), reverse=True)
    assert len(object_infos) >= 2, "expected src, dst, and legend"

    objs = [object_infos[0]["sub"], object_infos[1]["sub"]]
    holes_list = [object_infos[0]["holes"], object_infos[1]["holes"]]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [
                _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
                for s in src_subs
            ]
            tkeys, _s_rr, _s_cc = _rank_keys(tpts)
            if sorted(tkeys) == sorted(dest_keys):
                best = (int(k), bool(flip))
                break
        if best is not None:
            break
    if best is None:
        best = (0, False)
    k, flip = best

    dest_list = []
    for h in dest_holes:
        key = (int(d_rr[int(h["cent"][0])]), int(d_cc[int(h["cent"][1])]))
        dest_list.append((key, h["cent"], h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [
        _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        for s in src_subs
    ]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for s in src_subs:
        tp = _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(s["color"])))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    for (_k1, _p1, src_col), (_k2, _p2, hole) in zip(src_list, dest_list):
        fill_col = int(color_map.get(int(src_col), int(src_col))) if apply_map else int(src_col)
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(fill_col)

    return _tight_crop_non_bg(dest, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill dst's nested holes using src's nested band colors after legend remapping.
    """
    return _solve_scene(grid, apply_map=True)


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int], *, border_only: bool) -> np.ndarray:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, rect)
    mask = np.zeros((h, w), dtype=bool)
    if border_only:
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    else:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _interior_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, rect)
    mask = np.zeros((h, w), dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _build_nested_rects(inner_h: int, inner_w: int, drift: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    dr, dc = int(drift[0]), int(drift[1])
    top_step, bottom_step = (2, 1) if dr < 0 else (1, 2)
    left_step, right_step = (2, 1) if dc < 0 else (1, 2)

    cur = (0, 0, int(inner_h - 1), int(inner_w - 1))
    rects_inner_out = [cur]
    for _ in range(LAYER_COUNT - 1):
        r0, c0, r1, c1 = cur
        cur = (
            int(r0 - top_step),
            int(c0 - left_step),
            int(r1 + bottom_step),
            int(c1 + right_step),
        )
        rects_inner_out.append(cur)

    rects = rects_inner_out[::-1]
    min_r = min(int(r[0]) for r in rects)
    min_c = min(int(r[1]) for r in rects)
    shifted = []
    for r0, c0, r1, c1 in rects:
        shifted.append((int(r0 - min_r), int(c0 - min_c), int(r1 - min_r), int(c1 - min_c)))
    return shifted


def _render_nested_object(
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    h = int(max(r[2] for r in rects) + 1)
    w = int(max(r[3] for r in rects) + 1)
    canvas = np.full((h, w), int(bg), dtype=int)
    filled_canvas = np.full((h, w), int(bg), dtype=int)

    for rect in rects:
        border = _rect_mask((h, w), rect, border_only=True)
        canvas[border] = int(outline_col)
        filled_canvas[border] = int(outline_col)

    for i, rect in enumerate(rects):
        layer = _interior_mask((h, w), rect)
        if i + 1 < len(rects):
            inner_rect = _rect_mask((h, w), rects[i + 1], border_only=False)
            layer &= ~inner_rect
        if not layer.any():
            return None, None
        filled_canvas[layer] = int(colors[i])
        if filled:
            canvas[layer] = int(colors[i])

    b = bbox(filled_canvas != int(bg))
    if b is None:
        return None, None
    r0, c0, r1, c1 = b
    cropped = crop_bbox(canvas, b)
    filled_cropped = filled_canvas[r0 : r1 + 1, c0 : c1 + 1]
    return cropped, filled_cropped


def _count_4conn_components(mask: np.ndarray) -> int:
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(grid: np.ndarray, bg: int) -> int:
    return len(_enclosed_bg_components(grid, bg))


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def _nested_chain_ok(sub: np.ndarray, bg: int) -> bool:
    parts = _colored_components(sub, bg)
    if len(parts) != LAYER_COUNT:
        return False
    parts.sort(
        key=lambda d: (
            -(d["bbox"][2] - d["bbox"][0] + 1) * (d["bbox"][3] - d["bbox"][1] + 1),
            -d["size"],
        )
    )
    for outer, inner in zip(parts, parts[1:]):
        or0, oc0, or1, oc1 = map(int, outer["bbox"])
        ir0, ic0, ir1, ic1 = map(int, inner["bbox"])
        if not (or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1):
            return False
    return True


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _make_derangement(colors: list[int]) -> dict[int, int]:
    src = list(colors)
    dst = list(colors)
    while True:
        random.shuffle(dst)
        if all(int(a) != int(b) for a, b in zip(src, dst)):
            return {int(a): int(b) for a, b in zip(src, dst)}


def _build_legend(color_map: dict[int, int], outline_col: int) -> np.ndarray:
    rows = list(color_map.items())
    random.shuffle(rows)
    legend = np.full((len(rows) + 2, 5), int(outline_col), dtype=int)
    for i, (src_col, dst_col) in enumerate(rows, start=1):
        legend[int(i), 1] = int(src_col)
        legend[int(i), 3] = int(dst_col)
    return legend


def _build_object_pair(
    src_colors: list[int],
    dst_colors: list[int],
    bg: int,
    outline_col: int,
    drift: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    src_rects = _build_nested_rects(random.choice(INNER_SIZE_CHOICES), random.choice(INNER_SIZE_CHOICES), drift)
    dst_rects = _build_nested_rects(random.choice(INNER_SIZE_CHOICES), random.choice(INNER_SIZE_CHOICES), drift)

    src_cropped, _src_filled = _render_nested_object(src_rects, src_colors, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if not _size_in_spec(src_cropped):
        return None
    if _count_4conn_components((src_cropped != bg) & (src_cropped != outline_col)) != LAYER_COUNT:
        return None
    if not _nested_chain_ok(src_cropped, bg):
        return None

    dst_cropped, expected = _render_nested_object(dst_rects, dst_colors, bg, outline_col, filled=False)
    if dst_cropped is None or expected is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if not _size_in_spec(dst_cropped):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != LAYER_COUNT:
        return None

    src_parts = _colored_components(src_cropped, bg)
    dst_holes = _enclosed_bg_components(dst_cropped, bg)
    if len(src_parts) != LAYER_COUNT or len(dst_holes) != LAYER_COUNT:
        return None
    src_keys = sorted(_rank_keys([p["cent"] for p in src_parts])[0])
    dst_keys = sorted(_rank_keys([h["cent"] for h in dst_holes])[0])
    if src_keys != dst_keys:
        return None

    return src_cropped, dst_cropped, expected


def _place_on_canvas(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    for patch in patches:
        try:
            r, c = random_free_location(
                canvas,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
            )
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=bg)
    return canvas


def _combo_metrics_for_grid(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = _components(g != int(bg))
    infos = []
    for comp in comps:
        sub = _extract_group_subgrid(g, [comp], bg)
        infos.append({"sub": sub, "legend_map": _parse_legend(sub, bg), "holes": _enclosed_bg_components(sub, bg)})
    legend_idxs = [i for i, info in enumerate(infos) if info["legend_map"]]
    legend_idx = legend_idxs[0] if legend_idxs else None
    objects = [info["sub"] for i, info in enumerate(infos) if i != legend_idx]
    objects.sort(key=lambda sub: int(np.count_nonzero(sub != bg)), reverse=True)
    hole_counts = [_count_enclosed_bg_holes(obj, bg) for obj in objects]
    dest = objects[int(np.argmax(hole_counts))]
    src = objects[int(1 - int(np.argmax(hole_counts)))]

    mapped = _solve_scene(grid, apply_map=True)
    direct = _solve_scene(grid, apply_map=False)
    diff_pixels = int(np.count_nonzero(mapped != direct))
    legend_size = len(infos[legend_idx]["legend_map"]) if legend_idx is not None else 0
    layer_count = len(_colored_components(src, bg))
    return {
        "layer_count": int(layer_count),
        "legend_size": int(legend_size),
        "direct_diff_pixels": int(diff_pixels),
        "nested_chain": int(_nested_chain_ok(src, bg)),
        "dst_holes": int(_count_enclosed_bg_holes(dest, bg)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])
    fill_pool = [int(c) for c in palette[1 : 1 + LAYER_COUNT]]
    src_colors = list(fill_pool)
    random.shuffle(src_colors)
    color_map = _make_derangement(fill_pool)
    dst_colors = [int(color_map[int(col)]) for col in src_colors]
    drift = random.choice(DRIFT_DIRECTIONS)

    built = _build_object_pair(src_colors, dst_colors, bg, outline_col, drift)
    if built is None:
        return None
    src, dst, expected = built
    legend = _build_legend(color_map, outline_col)
    canvas = _place_on_canvas([src, dst, legend], bg)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
        direct = _solve_scene(canvas, apply_map=False)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, direct):
        return None

    metrics = _combo_metrics_for_grid(canvas)
    if metrics["layer_count"] != LAYER_COUNT:
        return None
    if metrics["legend_size"] != LAYER_COUNT:
        return None
    if metrics["dst_holes"] != LAYER_COUNT:
        return None
    if metrics["nested_chain"] != 1:
        return None
    if metrics["direct_diff_pixels"] <= 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
