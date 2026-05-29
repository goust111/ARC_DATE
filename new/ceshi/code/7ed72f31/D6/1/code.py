# Variant of 7ed72f31
# Dimension: D6 — 镜像像素的绘制规则 (transform)
# Change: 镜像像素必须落在背景格上才绘制，原始精灵和红色标记保持不变
#         → 原始精灵被擦除，仅保留镜像；红色标记也被擦除

from __future__ import annotations

import os
import sys
import random
import numpy as np
from scipy import ndimage

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import STRUCT8, get_bg, label_mask


"""
design_analysis:
  source_task_read:
    task: 71e489b6
    takeaway: |
      参考任务的复杂度基线不是几何变换，而是“先破坏原图局部状态，再只保留派生结果”。
      这说明 D6 的 erase 变体可以只改输出画布语义，不必新增新的匹配类别或额外符号。
  concrete_behavior:
    input_shape: |
      输入仍沿用原题几何：彩色载荷在红线一侧，或者小彩色块以 8 邻接方式围绕孤立红点。
      红线可以是整行/整列，孤立红点仍然只有一个像素。载荷单色，镜像目标区域预留为空背景。
    output_shape: |
      输出不再保留输入中的彩色原件，也不保留任何红色标记。每个对象只在反射后的目标位置
      留下一份彩色镜像，其他原先非背景单元全部恢复为背景。
    step_trace_example: |
      例：竖直红线在第 c=5 列，左侧有蓝色 L 形载荷。原题输出是“左侧 L + 红线 + 右侧镜像 L”。
      本变体输出改为“只有右侧镜像 L”，左侧原始 L 与红线都擦成背景。
      孤立红点同理：原题是“原块 + 红点 + 点反射副本”，本变体是“只有点反射副本”。
  complexity_calibration:
    kept_dimensions: |
      轴类型、对象数量、尺寸范围、颜色数量、连通性、放置间距全部沿用原题，避免无关创新。
    visibility_guard: |
      擦除原件会让输入/输出差异非常显眼，因此不需要额外引入颜色映射或多阶段规则。
      仍保留 3-7 个标记、1-4 个嵌入式对象，保证样例里有重复支撑，避免只出现单个孤立特例。
  implicit_assumption_audit_main:
    old_out_copy: |
      原代码 `out = g.copy()` 隐含“原件与红标记要保留”。变体中该前提失效，必须改成纯背景画布。
    old_global_scan: |
      原代码对嵌入式对象按整张网格扫描，再把符合 bbox/颜色条件的像素反射到 `out`。
      在 overlay 语义下，误保留原图无害；在 erase 语义下若继续依赖 `g.copy()` 就会错误保留原件。
      因此这里改为只遍历组件内部 payload 像素，直接生成镜像。
    isolated_markers: |
      原代码对孤立红点只追加反射像素，不清空红点与原块。变体中改成背景画布后，这些旧像素自然不会被带入输出。
    overlap_assumption: |
      原题依赖“镜像目标落在背景上”避免覆盖冲突。变体保留这个生成约束，因此 solver 仍可直接写入镜像。
  placement_legality:
    full_footprint: |
      生成器放置时仍使用完整足迹 = payload + red + mirror。这样虽然输出只显示 mirror，
      但输入里的原件与红标记、输出里的镜像都提前保证不越界、不互相重叠。
    derivation: |
      output-first 轻微改写为 footprint-first：先生成完整对象几何，再分别盖到 input/output。
      input 只盖 payload+red；output 只盖 mirror。
  visual_observability:
    human_inference: |
      人类看到样例时，输入中的红线/红点像“移动指令”，输出里对应对象出现在对称位置且原位被清空，
      能直接读出“移动而非复制”的语义。由于红标记也消失，区别于原题的叠加式镜像非常明显。
  cleanliness:
    spacing: |
      保留 1 格膨胀间距，避免不同对象在输入或输出里黏连成更大组件，维持肉眼可分辨性。
    color_policy: |
      不新增颜色变化；颜色只承担“识别同一载荷”的作用，避免不可见排序。
  new_solver_assumptions_and_generator_support:
    assumption_1: |
      嵌入式对象的 payload 与红标记都在同一个 4 连通分量内。生成器继续构造整行/整列红线贴着 payload。
    assumption_2: |
      孤立红点与其对应 payload 仅 8 邻接、不 4 邻接。生成器用 `_is_4connected_to` 显式保证。
    assumption_3: |
      反射目标不与其他对象冲突。生成器用完整足迹放置 + oracle `main(input)==output` 保证。
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red (2) markers as reflection axes, but keep only the mirror.

    Input structure:
    - bg = most common color. Red (2) = marker color.
    - Colored sprites appear near red markers.

    Rule:
    - Red line within a sprite (vertical or horizontal) -> mirror payload across that axis.
    - Single red pixel in a mixed sprite -> point-reflect payload around it.
    - Isolated red pixel (alone, no payload) -> point-reflect each adjacent 8-connected color component.
    - Unlike the original task, the original payload and all red markers are erased; only reflected pixels remain.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    mark = 2

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

    out = np.full_like(g, int(bg))

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    # 1) Marker-containing mixed components: reflect their non-red payload only.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(mark) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(mark)]
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

        payload_pixels = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(mark)]
        if not payload_pixels:
            continue

        for r, c, v in payload_pixels:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))
            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = int(v)

    # 2) Isolated red pixels: reflect adjacent color-components by point reflection.
    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(mark):
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
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(mark):
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
                    if inb(int(rr), int(cc)):
                        out[int(rr), int(cc)] = int(v)

    return out


# --------------- generator ---------------

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
GEN_RETRIES = 300

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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
    side_flip = rng.random() < 0.5

    if is_row:
        H, W = 2 * ph + 1, pw
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        red_pixels = [(ph, c) for c in range(pw)]
        orig_slice, mirr_slice = (np.s_[:ph], np.s_[ph + 1 :]) if not side_flip else (np.s_[ph + 1 :], np.s_[:ph])
        payload[orig_slice] = pmask
        mirror[mirr_slice] = pmask[::-1]
    else:
        H, W = pw, 2 * ph + 1
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        red_pixels = [(r, ph) for r in range(pw)]
        orig_slice, mirr_slice = (np.s_[:, :ph], np.s_[:, ph + 1 :]) if not side_flip else (np.s_[:, ph + 1 :], np.s_[:, :ph])
        payload[orig_slice] = pmask.T
        mirror[mirr_slice] = pmask.T[:, ::-1]

    return {"payload": payload, "mirror": mirror, "red_pixels": red_pixels, "kind": "embedded"}


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
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1

            payload = np.zeros((H, W), dtype=bool)
            mirror = np.zeros((H, W), dtype=bool)
            for r, c in pcoords:
                payload[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red_pos = (red_r - minr, red_c - minc)

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {"payload": payload, "mirror": mirror, "red_pixels": [red_pos], "kind": "isolated"}
    return None


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
            if not (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _footprint(spr: dict) -> np.ndarray:
    mask = spr["payload"] | spr["mirror"]
    for r, c in spr["red_pixels"]:
        mask[int(r), int(c)] = True
    return mask


def _stamp_input(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["payload"]] = spr["color"]
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + rr, c0 + cc] = MARK


def _stamp_output(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = spr["color"]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)

    sprites: list[dict] = []
    for builder, count in [(_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)]:
        for _ in range(count):
            s = builder(rng)
            if s is None:
                return None
            sprites.append(s)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, c in zip(sprites, assignment):
        spr["color"] = c

    layout = np.full((N, N), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = _footprint(spr)
        loc = _random_free_location(layout, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = r0, c0
        layout[r0 : r0 + footprint.shape[0], c0 : c0 + footprint.shape[1]][footprint] = 1

    inp = np.full((N, N), bg, dtype=int)
    out = np.full((N, N), bg, dtype=int)
    for spr in sprites:
        _stamp_input(inp, spr, int(spr["r0"]), int(spr["c0"]))
        _stamp_output(out, spr, int(spr["r0"]), int(spr["c0"]))

    if (inp == bg).sum() / (N * N) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
