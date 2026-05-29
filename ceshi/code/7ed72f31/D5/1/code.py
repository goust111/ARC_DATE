# Variant of 7ed72f31
# Dimension: D5 — 标记颜色约定 (marker_axis)
# Change: 标记颜色固定为红色(2)，所有示例一致... → 标记颜色编码反射方向

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask


# Design analysis for the D5 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original: |
#       原题把“变换类型”编码在标记几何里：横红线=沿该行镜像，竖红线=沿该列镜像，孤立红点=点反射。
#     variant: |
#       变体把“变换类型”编码在标记颜色里：红(2)=沿标记所在行反射，蓝(1)=沿标记所在列反射，绿(3)=绕标记做点反射。
#       为了让 D5 真正可观察，三类标记统一为单像素；否则若仍保留“横线/竖线/单点”的旧几何，
#       旧 solver 只靠几何也能做对，颜色语义不会产生任何效果，变体会退化。
#   concrete_appearance:
#     embedded: |
#       输入里仍有“标记 + 单色 payload”的 4 连通混合组件，但标记现在永远只有一个像素。
#       标记像素与 payload 4 邻接，说明它属于同一组件；颜色决定把 payload 沿该像素的行/列/中心点反射。
#     isolated: |
#       也保留“孤立标记”分支：标记像素 4 连通上独立，只与 payload 8 邻接。
#       该标记的颜色同样决定反射方式，然后把相邻的单色组件复制到背景上。
#   behavior_trace:
#     example: |
#       设一个 L 形黄色 payload 在红色单像素标记下方，并与标记 4 邻接。
#       原题逻辑看见“单像素标记”会执行点反射；变体逻辑看见“红色标记”则执行沿标记所在行的上下镜像。
#       同一输入上，两者输出不同：旧逻辑会把 L 形搬到对角位置，新逻辑会把 L 形直接翻到标记上方。
#   complexity_calibration:
#     choices: |
#       操作类别只保留 3 档（row/col/point），并要求每个样本至少出现这三种标记颜色各一次。
#       这样 2-3 个样例就能累积出颜色到操作的对应关系，同时不会引入过多自由度。
#   implicit_assumption_review:
#     main_changes:
#       - 原 `MARK = 2` 假设失效，必须改成 `MARK_OP = {2: row, 1: col, 3: point}`。
#       - 原先 `len(reds)==1 -> point` 的推断失效；现在单像素也可能表示 row/col。
#       - 原先 embedded 的 `eligible(...)` 依赖“整条红线覆盖 payload 全宽/全高”；单像素轴不再满足，
#         因此 embedded 分支必须只反射该组件里的 payload 像素，而不能扫描整张图按 bbox 猜范围。
#       - 原先 isolated 分支只对红点做点反射；现在 isolated 标记也可能是红/蓝/绿三种，需要按颜色分派。
#   placement_legality:
#     guarantees: |
#       generator 继续 output-first。先构造完整输出 patch（payload + marker + reflection），再擦掉 mirror 得到输入。
#       构造时强制：
#       1. marker 不与 payload 或 mirror 重叠；
#       2. mirror 只落在背景上；
#       3. embedded marker 与 payload 4 邻接，isolated marker 只 8 邻接；
#       4. row/col 反射时 payload 全部位于轴的一侧，避免原像与镜像重叠；
#       5. 不同 patch 之间留 1 像素 padding，避免组件合并。
#   visual_observability:
#     notes: |
#       颜色语义是显式可见的；标记总是单像素，几何本身不再告诉你操作类型。
#       红/蓝/绿各自重复出现，且 payload 使用不对称随机形状，避免“对象刚好对称导致看不出差别”。
#   cleanliness:
#     notes: |
#       patch 仍然分散放置，反射目标被擦除前后都不会覆盖其它对象；payload 颜色不使用 1/2/3，
#       避免与三类标记色混淆。


MARK_ROW = 2
MARK_COL = 1
MARK_POINT = 3
MARK_OP = {MARK_ROW: "row", MARK_COL: "col", MARK_POINT: "point"}
OPS = ("row", "col", "point")
MARKER_COLORS = tuple(MARK_OP.keys())

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
MARKER_RETRIES = 80
GEN_RETRIES = 300

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _reflect(op: str, mr: int, mc: int, r: int, c: int) -> tuple[int, int]:
    if op == "row":
        return int(2 * mr - r), int(c)
    if op == "col":
        return int(r), int(2 * mc - c)
    return int(2 * mr - r), int(2 * mc - c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using color-coded single-pixel markers.

    Marker semantics:
    - Red (2): reflect across the marker's row.
    - Blue (1): reflect across the marker's column.
    - Green (3): point-reflect around the marker.

    Marker geometry is no longer informative: every marker is a single pixel.
    The solver must decode color semantics, then apply the corresponding transform
    either to the marker's own 4-connected component (embedded) or to adjacent
    8-connected monochrome components (isolated).
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    marker_colors = set(MARKER_COLORS)
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    lab, n = label_mask(g != int(bg), connectivity=4)
    isolated_markers: list[tuple[int, int, int]] = []

    # Embedded marker handling: reflect only the payload pixels inside the same
    # 4-connected component. With single-pixel markers, bbox-based global scans
    # are unsafe; we stay local to the component.
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        markers = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) in marker_colors]
        if not markers:
            continue
        if len(markers) == 1 and len(coords) == 1:
            isolated_markers.append(markers[0])
            continue
        if len(markers) != 1:
            continue

        mr, mc, mcol = markers[0]
        op = MARK_OP[int(mcol)]
        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) not in marker_colors]
        if not payload:
            continue

        for r, c, v in payload:
            rr, cc = _reflect(op, int(mr), int(mc), int(r), int(c))
            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = int(v)

    # Isolated marker handling: find adjacent 8-connected monochrome components,
    # then reflect those components using the transform encoded by marker color.
    for mr, mc, mcol in isolated_markers:
        op = MARK_OP[int(mcol)]
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr, dc in _DIRS8:
            rr, cc = int(mr + dr), int(mc + dc)
            if not inb(rr, cc):
                continue
            v = int(g[rr, cc])
            if v == int(bg) or v in marker_colors:
                continue
            neigh_colors.add(int(v))
            neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = _reflect(op, int(mr), int(mc), int(r), int(c))
                    if inb(int(rr), int(cc)):
                        out[int(rr), int(cc)] = int(v)

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


def _normalize_sprite(
    pcoords: list[tuple[int, int]],
    refl: list[tuple[int, int]],
    marker: tuple[int, int],
    *,
    marker_color: int,
    op: str,
    kind: str,
) -> dict:
    all_pts = list(pcoords) + list(refl) + [marker]
    minr = min(r for r, _ in all_pts)
    minc = min(c for _, c in all_pts)
    maxr = max(r for r, _ in all_pts)
    maxc = max(c for _, c in all_pts)
    payload = np.zeros((maxr - minr + 1, maxc - minc + 1), dtype=bool)
    mirror = np.zeros_like(payload)
    for r, c in pcoords:
        payload[r - minr, c - minc] = True
    for r, c in refl:
        mirror[r - minr, c - minc] = True
    return {
        "payload": payload,
        "mirror": mirror,
        "marker_pos": (int(marker[0] - minr), int(marker[1] - minc)),
        "marker_color": int(marker_color),
        "op": op,
        "kind": kind,
    }


def _adjacency_flags(pcoords: list[tuple[int, int]], marker: tuple[int, int]) -> tuple[bool, bool]:
    mr, mc = marker
    has4 = False
    has8 = False
    for r, c in pcoords:
        dr = abs(int(r) - int(mr))
        dc = abs(int(c) - int(mc))
        if max(dr, dc) == 1:
            has8 = True
        if dr + dc == 1:
            has4 = True
    return has4, has8


def _candidate_markers(
    pcoords: list[tuple[int, int]],
    *,
    op: str,
    kind: str,
) -> list[tuple[int, int]]:
    rows = [r for r, _ in pcoords]
    cols = [c for _, c in pcoords]
    minr, maxr = min(rows), max(rows)
    minc, maxc = min(cols), max(cols)
    cands: set[tuple[int, int]] = set()

    if op == "row":
        top_cols = [c for r, c in pcoords if r == minr]
        bot_cols = [c for r, c in pcoords if r == maxr]
        if kind == "embedded":
            for c in top_cols:
                cands.add((minr - 1, c))
            for c in bot_cols:
                cands.add((maxr + 1, c))
        else:
            for c in top_cols:
                cands.add((minr - 1, c - 1))
                cands.add((minr - 1, c + 1))
            for c in bot_cols:
                cands.add((maxr + 1, c - 1))
                cands.add((maxr + 1, c + 1))
        return list(cands)

    if op == "col":
        left_rows = [r for r, c in pcoords if c == minc]
        right_rows = [r for r, c in pcoords if c == maxc]
        if kind == "embedded":
            for r in left_rows:
                cands.add((r, minc - 1))
            for r in right_rows:
                cands.add((r, maxc + 1))
        else:
            for r in left_rows:
                cands.add((r - 1, minc - 1))
                cands.add((r + 1, minc - 1))
            for r in right_rows:
                cands.add((r - 1, maxc + 1))
                cands.add((r + 1, maxc + 1))
        return list(cands)

    base_dirs = _DIRS4 if kind == "embedded" else _DIRS_DIAG
    occupied = set(pcoords)
    for r, c in pcoords:
        for dr, dc in base_dirs:
            cand = (int(r + dr), int(c + dc))
            if cand not in occupied:
                cands.add(cand)
    return list(cands)


def _choose_marker_and_reflection(
    pcoords: list[tuple[int, int]],
    *,
    op: str,
    kind: str,
    rng: random.Random,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    pset = set(pcoords)
    cands = _candidate_markers(pcoords, op=op, kind=kind)
    rng.shuffle(cands)

    for marker in cands[:MARKER_RETRIES]:
        has4, has8 = _adjacency_flags(pcoords, marker)
        if kind == "embedded":
            if not has4:
                continue
        else:
            if has4 or not has8:
                continue

        mr, mc = marker
        if op == "row":
            if not (all(r > mr for r, _ in pcoords) or all(r < mr for r, _ in pcoords)):
                continue
        elif op == "col":
            if not (all(c > mc for _, c in pcoords) or all(c < mc for _, c in pcoords)):
                continue

        refl = [_reflect(op, int(mr), int(mc), int(r), int(c)) for r, c in pcoords]
        rset = set(refl)
        if len(rset) != len(pcoords):
            continue
        if marker in rset or marker in pset:
            continue
        if pset & rset:
            continue
        return marker, refl
    return None


def _build_embedded_sprite(op: str, rng: random.Random) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None
    pcoords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
    chosen = _choose_marker_and_reflection(pcoords, op=op, kind="embedded", rng=rng)
    if chosen is None:
        return None
    marker, refl = chosen
    return _normalize_sprite(
        pcoords,
        refl,
        marker,
        marker_color={"row": MARK_ROW, "col": MARK_COL, "point": MARK_POINT}[op],
        op=op,
        kind="embedded",
    )


def _build_isolated_sprite(op: str, rng: random.Random) -> dict | None:
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = np.ones((ph, pw), dtype=bool) if n_px == ph * pw else _random_connected_mask(ph, pw, n_px, rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None
    pcoords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
    chosen = _choose_marker_and_reflection(pcoords, op=op, kind="isolated", rng=rng)
    if chosen is None:
        return None
    marker, refl = chosen
    return _normalize_sprite(
        pcoords,
        refl,
        marker,
        marker_color={"row": MARK_ROW, "col": MARK_COL, "point": MARK_POINT}[op],
        op=op,
        kind="isolated",
    )


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
):
    H, W = canvas.shape
    h, w = patch_mask.shape
    if h > H or w > W:
        return None
    occ = canvas != bg
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (occ[r0:r0 + h, c0:c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0:r0 + h, c0:c0 + w][colored] = spr["color"]
    mr, mc = spr["marker_pos"]
    canvas[r0 + mr, c0 + mc] = spr["marker_color"]


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = spr["r0"], spr["c0"]
    h, w = spr["mirror"].shape
    canvas[r0:r0 + h, c0:c0 + w][spr["mirror"]] = bg


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg_choices = [c for c in range(10) if c not in MARKER_COLORS]
    bg = rng.choice(bg_choices)
    palette = [c for c in range(10) if c not in MARKER_COLORS and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    ops = list(OPS) + [rng.choice(OPS) for _ in range(total_markers - len(OPS))]
    rng.shuffle(ops)
    kinds = ["embedded"] * n_embedded + ["isolated"] * n_isolated
    rng.shuffle(kinds)

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)

    sprites: list[dict] = []
    for kind, op in zip(kinds, ops):
        builder = _build_embedded_sprite if kind == "embedded" else _build_isolated_sprite
        spr = builder(op, rng)
        if spr is None:
            return None
        sprites.append(spr)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, color in zip(sprites, assignment):
        spr["color"] = int(color)

    output = np.full((N, N), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr["payload"] | spr["mirror"]
        mr, mc = spr["marker_pos"]
        footprint = footprint.copy()
        footprint[mr, mc] = True
        loc = _random_free_location(output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = int(r0), int(c0)
        _stamp_sprite(output, spr, int(r0), int(c0))

    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / float(N * N) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
