# Variant of 7ed72f31
# Dimension: D13 — 颜色调色板约束 (meta)
# Change: 3-4种不同的非背景、非红色颜色，每个嵌入式精灵单色... → 颜色决定反射距离倍率

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask

# ---------------------------------------------------------------------------
# Design analysis for the D13 variant
#
# a) 具体化与行为追踪
#    原题里，任意颜色的 payload 都做“标准镜像”，即离轴距离恒为 1 倍。
#    这个变体把“颜色”从纯外观属性改成控制参数：不同颜色的 sprite 在反射时，
#    到红轴/红点的距离要乘上不同倍率。这里固定两档，可直接肉眼比较：
#      - 蓝/黄 (1,4)  -> 倍率 1，仍是贴着轴的普通镜像
#      - 绿/青 (3,8)  -> 倍率 2，镜像会比普通镜像更远，中间多出一格空隙
#    例：若一个绿色像素在红横线上一格上方，原题会落到下一格下方；本变体会落到
#    两格下方，因此红线与镜像之间出现 1 行背景。对点反射同理，绿色/青色点会
#    以红点为中心，沿原向量延长 2 倍。
#
# b) 复杂度校准
#    只保留 1x / 2x 两档，避免 3+ 档导致放置过难、训练样例过 sparse。颜色数仍为
#    3-4 种，但语义颜色只从 {1,3,4,8} 中取，因此每张图都至少同时出现 1x 与 2x
#    两类行为，且多张样例之间颜色语义固定，可由人类归纳。
#
# c) 原 solver 的隐式假设审查
#    原代码在 embedded 分支里会“遍历整张图的所有同色像素，只按 bbox 的一维投影过
#    滤”。这在原题里通常无害，因为所有颜色倍率相同且生成器较疏；但在本变体里，
#    同色对象可能对应不同目标距离，若误吸入别的对象就会把远处像素投到错误位置。
#    因此这里改成：embedded 组件只反射该 4-connected 组件内部、非红色的 payload
#    像素。isolated 分支也改成先找红点 8 邻域接触到的实际组件，再逐像素按颜色倍率
#    反射，而不是对整张图某个颜色做全局扫描。
#
# d) 放置合法性
#    生成器仍采用 output-first：先按倍率构造完整输出 sprite，再整体放置，最后擦掉
#    mirror 区得到输入。这样“远距离镜像”是否越界、是否与别的对象冲突，都能在输出
#    阶段直接检查。padding 仍为 1，避免组件粘连；镜像真实占据的彩色/红色单元都会被
#    footprint 占用，因此不会被其它 sprite 覆盖。
#
# e) 视觉可观察性
#    1x 与 2x 的核心差别是“镜像是否紧贴红轴/红点另一侧”。对 embedded row/col，
#    2x 会在红线与镜像之间留下可见背景带；对 isolated point，2x 会让对称点更远。
#    这在 2-4 组 input/output 中是可直接比较的，而不是依赖不可见的颜色编号排序。
#
# f) 可视化整洁度
#    不引入额外标记，只改变镜像着陆位置。sprite 仍是单色、低密度、彼此有 padding，
#    空隙虽然变大，但网格仍清晰可读，不会在同一区域堆叠多层语义。
#
# g) 颜色变化可推断性
#    颜色不发生“变色”，而是控制“落点距离”。语义颜色集合固定且很小：
#      blue(1), yellow(4) -> 1x
#      green(3), teal(8)  -> 2x
#    训练样例中每张图使用 3-4 个颜色，保证颜色-倍率关系可从重复现象中推断。
#
# h) 新 solver 的额外假设及 generator 保证
#    - 语义颜色只取上面的四种；若出现其它颜色，solver 退化为倍率 1。
#    - embedded red marker 仍要求是单条横/竖红线，或单个红点；generator 只产出横/竖线
#      embedded，与原题一致。
#    - isolated red marker 旁边至少有一个 8 邻接 payload 组件；generator 保证存在且不
#      与其它对象粘连。
# ---------------------------------------------------------------------------

MARK = 2
DISTANCE_BY_COLOR = {1: 1, 4: 1, 3: 2, 8: 2}
SEMANTIC_COLORS = tuple(DISTANCE_BY_COLOR.keys())


def _distance_for_color(v: int) -> int:
    return int(DISTANCE_BY_COLOR.get(int(v), 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red markers as axes, with distance controlled by color.

    Blue/yellow payload mirrors with the usual distance (1x).
    Green/teal payload mirrors twice as far from the same red axis/point (2x).
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
        comps.append({"coords": coords, "vals": vals})

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(center: tuple[int, int], r: int, c: int, mult: int) -> tuple[int, int]:
        pr, pc = map(int, center)
        k = int(mult)
        return int((k + 1) * pr - k * int(r)), int((k + 1) * pc - k * int(c))

    def refl_row(r0: int, r: int, c: int, mult: int) -> tuple[int, int]:
        k = int(mult)
        return int((k + 1) * int(r0) - k * int(r)), int(c)

    def refl_col(c0: int, r: int, c: int, mult: int) -> tuple[int, int]:
        k = int(mult)
        return int(r), int((k + 1) * int(c0) - k * int(c))

    # 1) Reflect payload for marker-containing components.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if MARK not in vals:
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == MARK]
        payload_cells = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != MARK]
        if not reds or not payload_cells:
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

        for r, c, v in payload_cells:
            mult = _distance_for_color(int(v))
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c), mult)
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c), mult)
            else:
                rr, cc = refl_point(axis[1], int(r), int(c), mult)
            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = int(v)

    # 2) Isolated red pixels reflect each adjacent 8-connected payload component by point reflection.
    payload_lab, _ = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=8)
    marker_pts = [comp["coords"][0] for comp in comps if len(comp["coords"]) == 1 and int(comp["vals"][0]) == MARK]

    for pr, pc in marker_pts:
        touching_labels: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                lbl = int(payload_lab[int(rr), int(cc)])
                if lbl != 0:
                    touching_labels.add(int(lbl))
        for lbl in touching_labels:
            for r, c in np.argwhere(payload_lab == int(lbl)):
                v = int(g[int(r), int(c)])
                mult = _distance_for_color(int(v))
                rr, cc = refl_point((int(pr), int(pc)), int(r), int(c), mult)
                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    return out


# --------------- generator ---------------

"""Generator for the D13 variant of 7ed72f31.

Output-first strategy:
  1. Pick grid size, background, semantic colors.
  2. Assign each sprite a color first, so its reflection distance is known.
  3. Build the FULL output sprite using that distance multiplier.
  4. Place sprites without overlap.
  5. Erase the mirror half to derive the input.
  6. Oracle check main(input) == output.
"""

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


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    """Grow a random 4-connected mask with exactly *target_pixels* cells."""
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


def _build_embedded_sprite(rng: random.Random, distance_mult: int) -> dict | None:
    """Build an embedded row/col marker sprite whose mirror distance depends on *distance_mult*."""
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
    side_flip = rng.random() < 0.5

    if is_row:
        H, W = (distance_mult + 1) * ph + 1, pw
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        axis_r = ph
        payload[:ph, :] = pmask
        for r, c in np.argwhere(pmask):
            rr = int((distance_mult + 1) * axis_r - distance_mult * int(r))
            mirror[int(rr), int(c)] = True
        red_pixels = [(axis_r, c) for c in range(W)]
        if side_flip:
            payload = payload[::-1]
            mirror = mirror[::-1]
            red_pixels = [(H - 1 - int(r), int(c)) for r, c in red_pixels]
    else:
        H, W = ph, (distance_mult + 1) * pw + 1
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        axis_c = pw
        payload[:, :pw] = pmask
        for r, c in np.argwhere(pmask):
            cc = int((distance_mult + 1) * axis_c - distance_mult * int(c))
            mirror[int(r), int(cc)] = True
        red_pixels = [(r, axis_c) for r in range(H)]
        if side_flip:
            payload = payload[:, ::-1]
            mirror = mirror[:, ::-1]
            red_pixels = [(int(r), W - 1 - int(c)) for r, c in red_pixels]

    if (payload & mirror).any():
        return None
    return {"payload": payload, "mirror": mirror, "red_pixels": red_pixels, "distance_mult": int(distance_mult)}


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random, distance_mult: int) -> dict | None:
    """Build an isolated-red sprite with point reflection distance controlled by color."""
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

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = int(pr + dr), int(pc + dc)
            refl = [
                (
                    int((distance_mult + 1) * red_r - distance_mult * int(r)),
                    int((distance_mult + 1) * red_c - distance_mult * int(c)),
                )
                for r, c in pcoords
            ]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [int(r) for r, _ in all_pts]
            all_cs = [int(c) for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1

            payload = np.zeros((H, W), dtype=bool)
            mirror = np.zeros((H, W), dtype=bool)
            for r, c in pcoords:
                payload[int(r) - minr, int(c) - minc] = True
            for r, c in refl:
                mirror[int(r) - minr, int(c) - minc] = True
            red_pos = (int(red_r) - minr, int(red_c) - minc)

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {"payload": payload, "mirror": mirror, "red_pixels": [red_pos], "distance_mult": int(distance_mult)}
    return None


def _random_free_location(
    canvas: np.ndarray, patch_mask: np.ndarray, bg: int, rng: random.Random, *, padding: int = PLACEMENT_PADDING
):
    """Find a random location where *patch_mask* fits without touching occupied cells."""
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
            if not (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0 : r0 + h, c0 : c0 + w][colored] = int(spr["color"])
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + int(rr), c0 + int(cc)] = MARK


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = int(spr["r0"]), int(spr["c0"])
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = int(bg)


def _assign_colors(sprite_types: list[str], chosen_colors: list[int], rng: random.Random) -> list[int]:
    """Assign colors so every chosen color appears, and embedded sprites show both multipliers when possible."""
    total = len(sprite_types)
    assigned: list[int | None] = [None] * total
    embedded_idx = [i for i, t in enumerate(sprite_types) if t == "embedded"]
    colors_1x = [c for c in chosen_colors if _distance_for_color(c) == 1]
    colors_2x = [c for c in chosen_colors if _distance_for_color(c) == 2]

    if embedded_idx:
        assigned[embedded_idx[0]] = rng.choice(colors_1x if len(embedded_idx) >= 2 else chosen_colors)
    if len(embedded_idx) >= 2:
        assigned[embedded_idx[1]] = rng.choice(colors_2x)

    used_now = {c for c in assigned if c is not None}
    remaining_slots = [i for i, c in enumerate(assigned) if c is None]
    must_use = [c for c in chosen_colors if c not in used_now]
    rng.shuffle(remaining_slots)
    for idx, color in zip(remaining_slots, must_use):
        assigned[idx] = int(color)

    for i, c in enumerate(assigned):
        if c is None:
            assigned[i] = int(rng.choice(chosen_colors))

    return [int(c) for c in assigned]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK and c not in SEMANTIC_COLORS])

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(SEMANTIC_COLORS), total_markers))
    chosen_colors = rng.sample(list(SEMANTIC_COLORS), n_colors)

    sprite_types = ["embedded"] * n_embedded + ["isolated"] * n_isolated
    rng.shuffle(sprite_types)
    color_assignment = _assign_colors(sprite_types, chosen_colors, rng)

    sprites: list[dict] = []
    for kind, color in zip(sprite_types, color_assignment):
        dist = _distance_for_color(int(color))
        builder = _build_embedded_sprite if kind == "embedded" else _build_isolated_sprite
        spr = builder(rng, dist)
        if spr is None:
            return None
        spr["color"] = int(color)
        spr["kind"] = kind
        sprites.append(spr)

    output = np.full((N, N), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr["payload"] | spr["mirror"]
        for r, c in spr["red_pixels"]:
            footprint[int(r), int(c)] = True
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
