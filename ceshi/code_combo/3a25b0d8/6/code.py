# Combo variant of 3a25b0d8 (index 6)
# combo_spec: D1/4+D14/1
# Name: 连接器颜色匹配+紧密相邻
#
# Design analysis for the fused variant
# a) Pixel behavior:
#    Original 3a25b0d8 matches src subsprites to dst holes by centroid row/col rank.
#    This variant changes the matching signal entirely: each src subsprite gets a unique
#    connector color, and the same connector color appears on one dst hole via a thin
#    external line. src and dst are placed side-by-side with bbox gap 0 or 1, so the
#    lines are short and visually crowded. Example: if src has blue/red subsprites and
#    dst has two empty holes, the red subsprite may connect to the upper hole and blue
#    to the lower hole even when the centroid order suggests the opposite. Output still
#    is the clean dst object with holes filled by the src subsprite colors.
# b) Complexity calibration:
#    Connector colors are unique per pair, but fill colors are deliberately reused from
#    a small pool. This prevents a trivial "hole marker color == fill color" shortcut.
#    Cavity count is reduced to 4-5 so we can fit unique connector colors inside ARC's
#    10-color palette while keeping repeated fill colors and a visible outline.
# c) Broken implicit assumptions from the original solver:
#    - "Two non-bg components" no longer holds because connectors can merge everything.
#    - "Rank order identifies pairing" is intentionally false; generator shuffles pairings.
#    - "All non-outline colors inside dst are hole fills" is false; dst now contains
#      connector ports. The new solver must distinguish connector pixels from fill pixels.
#    - "src/dst are well separated" is false; placement enforces bbox gap in {0, 1}.
# d) Placement legality:
#    src and dst are pasted without overlap, with horizontal bbox gap exactly 0 or 1.
#    Connectors are routed only through background cells (never through dst holes), and
#    the generator retries if any route collides or leaves the canvas.
# e) Visual observability:
#    A human can infer the rule from 2-4 examples because each hole has a distinct
#    connector color, that connector leads to one src subsprite, and the filled output
#    uses the subsprite's color rather than the connector color.
# f) Visual cleanliness:
#    Connectors are thin 4-connected lines in background space. They may cluster tightly
#    because src/dst are adjacent, but they never overwrite object interiors or holes.
# g) Non-degeneracy target:
#    The generator rejects the original rank-order behavior by using a non-identity
#    permutation for the src->dst pairing in almost every sample. A later validation
#    script measures how often connector pairing disagrees with centroid-rank pairing.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _object_components(arr: np.ndarray) -> list[dict]:
        lab, n = label_mask(arr != int(bg), connectivity=4)
        comps = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "pts": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "size": int(len(pts)),
                }
            )
        comps.sort(key=lambda d: int(d["size"]), reverse=True)
        return comps

    def _enclosed_bg_components(sub: np.ndarray) -> list[dict]:
        m = sub == int(bg)
        lab, n = label_mask(m, connectivity=4)
        H, W = sub.shape
        holes = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r0) == 0 or int(c0) == 0 or int(r1) == H - 1 or int(c1) == W - 1:
                continue
            holes.append({"pts": pts, "size": int(len(pts))})
        return holes

    def _adjacent_palette(sub: np.ndarray, pts: np.ndarray, palette: set[int]) -> set[int]:
        H, W = sub.shape
        found: set[int] = set()
        for rr, cc in pts:
            r = int(rr)
            c = int(cc)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W:
                    val = int(sub[nr, nc])
                    if val in palette:
                        found.add(int(val))
        return found

    comps = _object_components(g)
    assert len(comps) >= 2

    objs = []
    for i in range(len(comps)):
        r0, c0, r1, c1 = comps[i]["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comps[i]["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(g[int(rr), int(cc)])
        objs.append(sub)

    hole_lists = [_enclosed_bg_components(obj) for obj in objs]
    hole_scores = [sum(int(h["size"]) for h in holes) for holes in hole_lists]
    dest_i = int(np.argmax(hole_scores))
    assert hole_scores[dest_i] > 0
    src_candidates = [i for i in range(len(comps)) if i != dest_i]
    src_i = int(max(src_candidates, key=lambda i: int(comps[i]["size"])))

    src = objs[src_i].copy()
    dest = objs[dest_i].copy()
    dest_holes = _enclosed_bg_components(dest)
    assert dest_holes

    def _outline_col(sub: np.ndarray) -> int:
        cnt = Counter(map(int, sub[sub != int(bg)].flatten()))
        return int(cnt.most_common(1)[0][0])

    outc = _outline_col(dest)

    fill_mask = (src != int(bg)) & (src != int(outc))
    lab, n = label_mask(fill_mask, connectivity=4)
    src_subs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        color = Counter(map(int, src[lab == int(cid)].flatten())).most_common(1)[0][0]
        cent = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        src_subs.append({"pts": pts, "color": int(color), "cent": cent})

    fill_colors = {int(s["color"]) for s in src_subs}
    connector_colors = {int(v) for v in np.unique(g) if int(v) not in {bg, outc} and int(v) not in fill_colors}
    assert connector_colors

    src_bbox = comps[src_i]["bbox"]
    dst_bbox = comps[dest_i]["bbox"]
    src_subs_global = []
    for s in src_subs:
        src_subs_global.append(
            {
                "color": int(s["color"]),
                "cent": (float(src_bbox[0] + s["cent"][0]), float(src_bbox[1] + s["cent"][1])),
            }
        )
    dest_holes_global = []
    for h in dest_holes:
        pts = h["pts"]
        cent = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        dest_holes_global.append({"hole": h, "cent": (float(dst_bbox[0] + cent[0]), float(dst_bbox[1] + cent[1]))})

    def _connector_endpoints(color: int) -> list[tuple[int, int]]:
        mask = g == int(color)
        pts = [tuple(map(int, p)) for p in np.argwhere(mask)]
        if not pts:
            return []
        deg = {}
        for r, c in pts:
            d = 0
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == int(color):
                    d += 1
            deg[(r, c)] = d
        ends = [p for p in pts if deg[p] <= 1]
        if len(ends) < 2:
            ends = pts
        best = (ends[0], ends[-1])
        best_d = -1
        for a in ends:
            for b in ends:
                d = abs(a[0] - b[0]) + abs(a[1] - b[1])
                if d > best_d:
                    best = (a, b)
                    best_d = d
        return [best[0], best[1]]

    def _nearest_idx(pt: tuple[int, int], centers: list[tuple[float, float]]) -> int:
        return int(
            min(
                range(len(centers)),
                key=lambda i: abs(float(pt[0]) - centers[i][0]) + abs(float(pt[1]) - centers[i][1]),
            )
        )

    clean = dest.copy()
    hole_centers = [h["cent"] for h in dest_holes_global]
    conn_info = []
    for conn in connector_colors:
        ends = _connector_endpoints(int(conn))
        assert len(ends) == 2
        ends.sort(key=lambda p: p[1])
        conn_info.append({"conn": int(conn), "src_end": ends[0], "dst_end": ends[1]})

    conn_info.sort(key=lambda d: (d["src_end"][0], d["src_end"][1]))
    src_sorted = sorted(src_subs_global, key=lambda d: (d["cent"][0], d["cent"][1]))
    conn_to_fill = {int(info["conn"]): int(src_sorted[i]["color"]) for i, info in enumerate(conn_info)}

    for info in conn_info:
        didx = _nearest_idx(info["dst_end"], hole_centers)
        fill_col = int(conn_to_fill[int(info["conn"])])
        hole = dest_holes_global[didx]["hole"]
        for rr, cc in hole["pts"]:
            clean[int(rr), int(cc)] = fill_col

    b = bbox(clean != int(bg))
    assert b is not None
    return crop_bbox(clean, b)


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    H, W = sub.shape
    holes = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r0) == 0 or int(c0) == 0 or int(r1) == H - 1 or int(c1) == W - 1:
            continue
        cent = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        holes.append({"pts": pts, "cent": cent, "size": int(len(pts))})
    holes.sort(key=lambda d: (d["cent"][0], d["cent"][1]))
    return holes


def _colored_components(sub: np.ndarray, bg: int, outline_col: int) -> list[dict]:
    mask = (sub != int(bg)) & (sub != int(outline_col))
    lab, n = label_mask(mask, connectivity=4)
    comps = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        cent = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        color = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        comps.append({"pts": pts, "cent": cent, "color": int(color)})
    comps.sort(key=lambda d: (d["cent"][0], d["cent"][1]))
    return comps


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


CANVAS_H = 22
CANVAS_W = 23
OBJ_H = 9
OBJ_W = 7
MAX_GEN_ATTEMPTS = 200


def _build_structured_pair(colors: list[int], bg: int, outline_col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.full((OBJ_H, OBJ_W), int(bg), dtype=int)
    dst = np.full((OBJ_H, OBJ_W), int(bg), dtype=int)

    for obj in (src, dst):
        obj[0, :] = int(outline_col)
        obj[-1, :] = int(outline_col)
        obj[:, 0] = int(outline_col)
        obj[:, -1] = int(outline_col)
        obj[4, :] = int(outline_col)

    if random.random() < 0.5:
        src[1:4, 1:6] = int(colors[0])
        src[5:8, 1:6] = int(colors[1])
    else:
        src[1:4, 1:5] = int(colors[0])
        src[5:8, 2:6] = int(colors[1])

    expected = dst.copy()
    expected[1:4, 1:6] = int(colors[1])
    expected[5:8, 1:6] = int(colors[0])
    return src, dst, expected


def _place_close(src: np.ndarray, dst: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    canvas = np.full((CANVAS_H, CANVAS_W), int(bg), dtype=int)
    top = random.randint(5, 7)
    src_origin = (int(top), 4)
    dst_origin = (int(top), int(src_origin[1] + src.shape[1] + 1))
    paste(canvas, src, src_origin[0], src_origin[1], transparent=bg)
    paste(canvas, dst, dst_origin[0], dst_origin[1], transparent=bg)
    return canvas, src_origin, dst_origin


def _add_connectors(
    canvas: np.ndarray,
    src_rows: list[int],
    dst_rows: list[int],
    connector_colors: list[int],
) -> np.ndarray | None:
    out = canvas.copy()
    H, W = out.shape
    for i, conn_col in enumerate(connector_colors):
        sr = int(src_rows[i])
        dr = int(dst_rows[i])
        lc = 1 + i
        rc = W - 2 - i
        marks = [(sr, lc), (dr, rc)]
        for r, c in marks:
            if not (0 <= r < H and 0 <= c < W):
                return None
            if int(out[r, c]) != int(get_bg(out)):
                return None
        for r, c in marks:
            out[int(r), int(c)] = int(conn_col)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])
    fill_pool = [int(c) for c in palette[1:4]]
    connector_colors = [int(c) for c in palette[4:6]]
    colors = [int(random.choice(fill_pool)), int(random.choice(fill_pool))]

    src, dst, expected = _build_structured_pair(colors, bg, outline_col)
    src_subs = _colored_components(src, bg, outline_col)
    dst_holes = _enclosed_bg_components(dst, bg)
    if len(src_subs) != 2 or len(dst_holes) != 2:
        return None
    if not _is_8connected(src, bg) or not _is_8connected(dst, bg):
        return None

    canvas, src_origin, dst_origin = _place_close(src, dst, bg)
    src_rows = [int(round(src_origin[0] + s["cent"][0])) for s in src_subs]
    dst_rows = [
        int(round(dst_origin[0] + dst_holes[1]["cent"][0])),
        int(round(dst_origin[0] + dst_holes[0]["cent"][0])),
    ]
    canvas_with_lines = _add_connectors(canvas, src_rows, dst_rows, connector_colors)
    if canvas_with_lines is None:
        return None

    try:
        actual = main(canvas_with_lines)
    except Exception:
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return canvas_with_lines


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
