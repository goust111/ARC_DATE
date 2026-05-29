# Combo variant of 2c181942 (index 2)
# combo_spec: D1/0+D7/0
# Name: 形状轮廓匹配+锚点消歧

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  fusion_summary: |
    这个融合版同时替换了原题的两条关键捷径：
      1) 不再靠颜色把 sprite 直接配到某一边。hub 与所有 sprite 主体统一为同一种颜色，
         只能看 hub 外侧 dock 的凹口模式与 sprite 接触边的凸起模式是否互补。
      2) 不再要求 sprite 沿接触轴镜像对称。相反，每个 sprite 都有一个特殊颜色的锚点像素，
         正确旋转时该锚点必须位于 sprite bbox 的左上角。
    因此配对由“轮廓形状”负责，旋转消歧由“锚点位置”负责，两者职责明确分离。

  a_concretization_and_trace: |
    采用 D1 单变体的可观察几何：中心是 6x6 hollow core，四边外侧各有 2 格厚 dock，
    最外层用 4-bit 凹口模式编码边身份。示意上，top/bot/left/right 四边分别读出不同的
    4 格缺口序列；散落 sprite 的接触边则做出对应 4 格凸起。

    一个具体例子：
      - top dock 外沿模式是 1000，表示最外侧 4 格里只有第 1 格留空，其余 3 格实心。
      - 与之匹配的 top-sprite 在正确朝向下，最底两行必须是：
          倒数第 2 行：1111   （整条 support 带）
          倒数第 1 行：1000   （恰好补进 hub 的凹口）
      - 但这个 sprite 同时也故意在最上两行做出同样的 1111/1000。
        于是如果只看轮廓，0° 和 180° 都能匹配同一边。
      - 融合后 solver 先用轮廓确认“它属于 top 边”，再检查哪一个旋转让锚点像素落在
        bbox 左上角，只接受那个朝向。

    相比原题：
      - 原题：颜色先决定 side；镜像对称再决定 rotation。
      - 融合：轮廓先决定 side；锚点位置再决定 rotation。
      - 差异在输入里是可见的：hub 只有一种主体色，sprite 都带一个特殊角标像素，而且
        轮廓本身会制造 180° 朝向二义性。

  b_complexity_calibration: |
    仅仅把颜色统一是不够的，因为直边 hub 上没有可见差异，会完全失去可配对性。
    因此保留 D1 的 dock+凹口编码，并把 pattern 固定在 4 个少量档位：
      top   = 1000
      bot   = 1100
      left  = 1010
      right = 1110
    这些模式互不相同，也都不是彼此 reverse，从而边身份稳定。

    同时，锚点不承担配对职责，只承担“在两个轮廓都合法的旋转里选唯一正确朝向”的职责。
    这样不会引入第二套复杂的 side 编码，人类也更容易从 2-4 组样例归纳出规则。

  c_implicit_assumptions_review: |
    原始 main() 中被破坏的隐式前提：
      - `m = (g == col) & (~hub_mask)`：
          依赖“颜色唯一决定 side”。融合后失效，因为所有主体都是同色。
          新 solver 改为：先找包含 core 的 hub 组件，再把其余非背景组件逐个尝试匹配四边。
      - hub 四边颜色互异：
          失效。新 hub 的四边与 dock 全都同色，只剩轮廓差异。
      - “合法旋转 = 接触边长度正确 + 沿接触轴镜像对称”：
          失效。新 solver 完全移除对称性判断，改成
          “合法旋转 = 接触带 pattern 正确 + 锚点在 bbox 左上角”。
      - 干扰物可以靠新颜色直接过滤：
          失效。新 generator 若生成 distractor，也必须和真 sprite 共享主体色与锚点色，
          但轮廓不形成任何合法 contact pattern。

    多项叠加后的新假设：
      - 每个真 sprite 恰有 1 个锚点像素。
      - 每个真 sprite 的 occupancy 轮廓在忽略锚点颜色时至少有 2 个合法 (rotation, side)
        候选，而加上锚点规则后只剩 1 个。
      - generator 必须显式保证这些条件，否则 D7 会退化成“可有可无”。

  d_placement_legality: |
    生成器仍用 output-first：
      - 先画 hub+dock。
      - 再为若干 side 生成 canonical attached sprites，逐个按固定几何贴到输出图。
      - 每次贴附都检查越界与冲突；允许 support 带与 hub 同色 dock 重合，但不允许锚点压到
        hub 已有前景上。
      - 然后把这些 sprites 拆下、随机非零旋转、散落回输入中，并用
        `random_free_location(..., padding=2, connectivity=8)` 保证彼此分离。

  e_visual_observability: |
    人类能直接看到三条稳定线索：
      - hub 只有一种主体色，说明不可能再按颜色配对。
      - hub 四边外沿的缺口模式不同，散落 sprite 的“嘴型”与某一边互补。
      - 每个 sprite 都带一个同色系之外的单像素角标，输出里这些角标都会跑到各自 sprite 的
        左上角，说明角标在决定朝向。
    这三条线索都来自可见像素结构，不依赖颜色编号大小、扫描顺序等不可视因素。

  f_visual_cleanliness: |
    画面只使用三类元素：
      - 同色 hub core+dock
      - 同色 sprite 主体
      - 单像素锚点色
    锚点只占一个像素，不会淹没轮廓；dock 统一厚度 2，pattern 固定长度 4，也避免了 hub
    周围出现过度复杂的层叠。

  g_inference_of_change: |
    这里不存在大规模颜色映射。颜色数量被严格限制为：
      - bg
      - 主体色（hub 与 sprite 共用）
      - 锚点色（所有 sprite 共用）
    人类需要推断的是“哪种轮廓补哪条边”与“锚点左上才是正确朝向”，而不是猜颜色置换表。

  h_post_test_metric: |
    退化检查使用两个量化指标：
      - ambiguous_matchable_components:
          输入中非 hub 组件里，有多少个组件在“忽略锚点颜色时”存在至少 2 个合法轮廓匹配。
      - anchor_resolved_components:
          同一批组件里，有多少个组件在“加入锚点左上约束后”只剩 1 个合法匹配。
    对真 sprite，这两个数应同时成立；若 ambiguous 长期为 0，就说明锚点根本没参与推理，
    该融合已退化。

  i_solver_assumptions_guaranteed_by_generator: |
    main() 新增的输入假设，以及 generator 的保证方式：
      - 假设 hub 是离中心最近的 6x6 hollow core，generator 固定这样构造。
      - 假设 hub 外侧 dock 的 4 个 pattern 都可读，generator 固定画满。
      - 假设每个真 sprite 恰有一个特殊锚点像素，generator 通过重着色一个现有占据格保证。
      - 假设真 sprite 忽略锚点时至少有两种合法旋转、加锚点后唯一，generator 用显式验证
        `_shape_hits(mask)` 与 `_anchor_hits(arr)` 过滤。
      - 假设 distractor 不会误匹配，generator 也用相同验证拒绝任何可匹配的假目标。
"""


HUB_N = 6
EDGE_LEN = HUB_N - 2
DOCK_THICKNESS = 2
BG_COLOR = 8
SIDES = ("top", "bot", "left", "right")
PATTERNS = {
    "top": (1, 0, 0, 0),
    "bot": (1, 1, 0, 0),
    "left": (1, 0, 1, 0),
    "right": (1, 1, 1, 0),
}
GENERATOR_SIDES = ("top", "left")
ANCHOR_CELLS = {(0, 0), (0, 1), (1, 0), (1, 1)}
ANCHOR_POSITIONS = {
    "top": (0, 0),
    "bot": (0, 0),
    "left": (0, 1),
    "right": (0, 0),
}


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _find_core_hub(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    H, W = g.shape
    best = None
    n = HUB_N
    for r in range(2, H - n - 2):
        for c in range(2, W - n - 2):
            if (
                int(g[r, c]) != bg
                or int(g[r, c + n - 1]) != bg
                or int(g[r + n - 1, c]) != bg
                or int(g[r + n - 1, c + n - 1]) != bg
            ):
                continue

            inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
            if not bool((inner == bg).all()):
                continue

            top = g[r, c + 1 : c + n - 1]
            bot = g[r + n - 1, c + 1 : c + n - 1]
            left = g[r + 1 : r + n - 1, c]
            right = g[r + 1 : r + n - 1, c + n - 1]
            if bool((top == bg).any()) or bool((bot == bg).any()) or bool((left == bg).any()) or bool((right == bg).any()):
                continue

            colors = set(map(int, np.concatenate([top, bot, left, right])))
            if len(colors) != 1:
                continue

            color = int(next(iter(colors)))
            if color == bg:
                continue

            cr = r + (n - 1) / 2
            cc = c + (n - 1) / 2
            score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
            cand = (float(score), int(r), int(c), int(color))
            if best is None or cand[0] < best[0]:
                best = cand

    assert best is not None
    _, r, c, color = best
    return r, c, HUB_N, color


def _side_span(r: int, c: int, side: str) -> tuple[int, int]:
    if side in ("top", "bot"):
        return c + 1, c + HUB_N - 2
    return r + 1, r + HUB_N - 2


def _dock_outer_coord(r: int, c: int, side: str) -> int:
    if side == "top":
        return r - 2
    if side == "bot":
        return r + HUB_N + 1
    if side == "left":
        return c - 2
    return c + HUB_N + 1


def _read_hub_patterns(g: np.ndarray, r: int, c: int, bg: int) -> dict[str, tuple[int, ...]]:
    patterns: dict[str, tuple[int, ...]] = {}
    for side in SIDES:
        start, end = _side_span(r, c, side)
        outer = _dock_outer_coord(r, c, side)
        vals: list[int] = []
        if side in ("top", "bot"):
            for cc in range(start, end + 1):
                vals.append(1 if int(g[outer, cc]) == bg else 0)
        else:
            for rr in range(start, end + 1):
                vals.append(1 if int(g[rr, outer]) == bg else 0)
        patterns[side] = tuple(vals)
    return patterns


def _contact_band(mask: np.ndarray, side: str, pattern: tuple[int, ...]) -> int | None:
    h, w = mask.shape
    L = len(pattern)
    hits: list[int] = []

    if side == "top":
        if h < 2 or w < L:
            return None
        for j in range(w - L + 1):
            if bool(mask[h - 2, :j].any()) or bool(mask[h - 2, j + L :].any()):
                continue
            if bool(mask[h - 1, :j].any()) or bool(mask[h - 1, j + L :].any()):
                continue
            if not bool(mask[h - 2, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[h - 1, j : j + L]) != pattern:
                continue
            hits.append(j)
    elif side == "bot":
        if h < 2 or w < L:
            return None
        for j in range(w - L + 1):
            if bool(mask[1, :j].any()) or bool(mask[1, j + L :].any()):
                continue
            if bool(mask[0, :j].any()) or bool(mask[0, j + L :].any()):
                continue
            if not bool(mask[1, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[0, j : j + L]) != pattern:
                continue
            hits.append(j)
    elif side == "left":
        if w < 2 or h < L:
            return None
        for j in range(h - L + 1):
            if bool(mask[:j, w - 2].any()) or bool(mask[j + L :, w - 2].any()):
                continue
            if bool(mask[:j, w - 1].any()) or bool(mask[j + L :, w - 1].any()):
                continue
            if not bool(mask[j : j + L, w - 2].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, w - 1]) != pattern:
                continue
            hits.append(j)
    else:
        if w < 2 or h < L:
            return None
        for j in range(h - L + 1):
            if bool(mask[:j, 1].any()) or bool(mask[j + L :, 1].any()):
                continue
            if bool(mask[:j, 0].any()) or bool(mask[j + L :, 0].any()):
                continue
            if not bool(mask[j : j + L, 1].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, 0]) != pattern:
                continue
            hits.append(j)

    if len(hits) != 1:
        return None
    return hits[0]


def _shape_hits(mask: np.ndarray) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for krot in range(4):
        rot = np.rot90(mask, krot)
        rot = _crop_to_mask(rot.astype(int), rot)[1]
        for side, pattern in PATTERNS.items():
            if _contact_band(rot, side, pattern) is not None:
                hits.append((int(krot), side))
    return hits


def _anchor_hits(arr: np.ndarray, bg: int, body_color: int) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for krot in range(4):
        rot = np.rot90(arr, krot)
        mask = rot != bg
        rot, mask = _crop_to_mask(rot, mask)
        anchor_pts = np.argwhere((rot != bg) & (rot != body_color))
        if anchor_pts.shape[0] != 1:
            continue
        ar, ac = map(int, anchor_pts[0])
        if (ar, ac) not in ANCHOR_CELLS:
            continue
        for side, pattern in PATTERNS.items():
            if _contact_band(mask, side, pattern) is not None:
                hits.append((int(krot), side))
    return hits


def _placeable(out: np.ndarray, arr: np.ndarray, mask: np.ndarray, dr: int, dc: int, bg: int) -> bool:
    H, W = out.shape
    for rr, cc in np.argwhere(mask):
        R = int(rr + dr)
        C = int(cc + dc)
        if not (0 <= R < H and 0 <= C < W):
            return False
        val = int(arr[rr, cc])
        if val == bg:
            continue
        if int(out[R, C]) != bg and int(out[R, C]) != val:
            return False
    return True


def _attachment_shift(mask: np.ndarray, side: str, pattern: tuple[int, ...], hub_r: int, hub_c: int) -> tuple[int, int] | None:
    pos = _contact_band(mask, side, pattern)
    if pos is None:
        return None
    span_start, _span_end = _side_span(hub_r, hub_c, side)
    outer = _dock_outer_coord(hub_r, hub_c, side)
    h, w = mask.shape

    if side == "top":
        return outer - (h - 1), span_start - pos
    if side == "bot":
        return outer, span_start - pos
    if side == "left":
        return span_start - pos, outer - (w - 1)
    return span_start - pos, outer


def _match_component_to_side(
    comp_arr: np.ndarray,
    comp_mask: np.ndarray,
    side: str,
    pattern: tuple[int, ...],
    hub_r: int,
    hub_c: int,
    out: np.ndarray,
    bg: int,
    body_color: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    hits: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for krot in range(4):
        arr = np.rot90(comp_arr, krot)
        mask = np.rot90(comp_mask, krot)
        arr, mask = _crop_to_mask(arr, mask)
        if sorted(_anchor_hits(arr, bg, body_color)) != [(0, side)]:
            continue

        shift = _attachment_shift(mask, side, pattern, hub_r, hub_c)
        if shift is None:
            continue
        dr, dc = shift
        if not _placeable(out, arr, mask, dr, dc, bg):
            continue
        hits.append((arr, mask, int(dr), int(dc)))

    if len(hits) != 1:
        return None
    return hits[0]


def _component_records(g: np.ndarray, bg: int, hub_r: int, hub_c: int) -> tuple[np.ndarray, list[tuple[int, np.ndarray, np.ndarray, int]]]:
    lab, nc = label_mask(g != bg, connectivity=8)
    hub_id = int(lab[hub_r, hub_c + 1])
    assert hub_id > 0
    hub_mask = lab == hub_id

    comps: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for cid in range(1, int(nc) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        comps.append((int(cid), arr, mask, int(mask.sum())))
    comps.sort(key=lambda x: -x[3])
    return hub_mask, comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub_r, hub_c, hub_n, body_color = _find_core_hub(g, bg)
    assert hub_n == HUB_N

    hub_mask, comps = _component_records(g, bg, hub_r, hub_c)
    out = np.full((H, W), bg, dtype=int)
    out[hub_mask] = g[hub_mask]

    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)

    used: set[int] = set()
    for side in SIDES:
        pattern = hub_patterns[side]
        found = None
        found_id = None
        for cid, arr, mask, _sz in comps:
            if cid in used:
                continue
            match = _match_component_to_side(arr, mask, side, pattern, hub_r, hub_c, out, bg, body_color)
            if match is None:
                continue
            found = match
            found_id = cid
            break
        if found is None:
            continue
        arr, mask, dr, dc = found
        paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=True)
        used.add(int(found_id))

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _make_sprite_mask(pattern: tuple[int, ...], side: str, rng: random.Random) -> np.ndarray | None:
    del pattern, rng
    prototypes = {
        "top": np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 0, 0, 0],
            ],
            dtype=bool,
        ),
        "bot": np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
            ],
            dtype=bool,
        ),
        "left": np.array(
            [
                [0, 1, 1],
                [1, 1, 0],
                [0, 1, 1],
                [1, 1, 0],
            ],
            dtype=bool,
        ),
        "right": np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            dtype=bool,
        ),
    }
    mask = prototypes.get(side)
    if mask is None:
        return None
    shape_hits = sorted(_shape_hits(mask))
    expected_second = {"top": 3, "bot": 1, "left": 2, "right": 2}[side]
    if shape_hits != [(0, side), (expected_second, side)]:
        return None
    return mask.copy()


def _paint_anchor(mask: np.ndarray, side: str, body_color: int, anchor_color: int, bg: int) -> np.ndarray:
    arr = np.where(mask, body_color, bg).astype(int)
    ar, ac = ANCHOR_POSITIONS[side]
    arr[ar, ac] = int(anchor_color)
    return arr


def _make_distractor_arr(body_color: int, anchor_color: int, bg: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(500):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        mask = np.zeros((h, w), dtype=bool)
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        k = rng.randint(5, min(10, h * w))
        for rr, cc in cells[:k]:
            mask[rr, cc] = True
        mask[0, 0] = True
        mask = _crop_to_mask(mask.astype(int), mask)[1]
        if not bool(mask[0, 0]):
            continue
        lab, nc = label_mask(mask, connectivity=8)
        if nc != 1:
            continue
        if _shape_hits(mask):
            continue
        side = rng.choice(list(ANCHOR_POSITIONS))
        if not bool(mask[ANCHOR_POSITIONS[side]]):
            continue
        arr = _paint_anchor(mask, side, body_color, anchor_color, bg)
        if _anchor_hits(arr, bg, body_color):
            continue
        return arr
    return None


def _draw_hub(canvas: np.ndarray, hub_r: int, hub_c: int, color: int, bg: int) -> None:
    n = HUB_N
    for cc in range(1, n - 1):
        canvas[hub_r, hub_c + cc] = color
        canvas[hub_r + n - 1, hub_c + cc] = color
    for rr in range(1, n - 1):
        canvas[hub_r + rr, hub_c] = color
        canvas[hub_r + rr, hub_c + n - 1] = color

    span_c0 = hub_c + 1
    span_r0 = hub_r + 1

    canvas[hub_r - 1, span_c0 : span_c0 + EDGE_LEN] = color
    canvas[hub_r + HUB_N, span_c0 : span_c0 + EDGE_LEN] = color
    canvas[span_r0 : span_r0 + EDGE_LEN, hub_c - 1] = color
    canvas[span_r0 : span_r0 + EDGE_LEN, hub_c + HUB_N] = color

    for i, v in enumerate(PATTERNS["top"]):
        if not v:
            canvas[hub_r - 2, span_c0 + i] = color
    for i, v in enumerate(PATTERNS["bot"]):
        if not v:
            canvas[hub_r + HUB_N + 1, span_c0 + i] = color
    for i, v in enumerate(PATTERNS["left"]):
        if not v:
            canvas[span_r0 + i, hub_c - 2] = color
    for i, v in enumerate(PATTERNS["right"]):
        if not v:
            canvas[span_r0 + i, hub_c + HUB_N + 1] = color


def _attach_sprite(
    out: np.ndarray,
    arr: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    mask = arr != bg
    arr, mask = _crop_to_mask(arr, mask)
    shift = _attachment_shift(mask, side, PATTERNS[side], hub_r, hub_c)
    if shift is None:
        return None
    dr, dc = shift
    if not _placeable(out, arr, mask, dr, dc, bg):
        return None
    paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=False)
    return arr, mask, int(dr), int(dc)


def variant_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    hub_r, hub_c, _n, body_color = _find_core_hub(g, bg)
    hub_mask, comps = _component_records(g, bg, hub_r, hub_c)
    _ = hub_mask

    ambiguous = 0
    resolved = 0
    matchable = 0
    anchored_unique = 0
    for _cid, arr, mask, _sz in comps:
        shape_hits = _shape_hits(mask)
        anchor_hits = _anchor_hits(arr, bg, body_color)
        if len(shape_hits) >= 2:
            ambiguous += 1
        if len(shape_hits) > len(anchor_hits):
            resolved += 1
        if shape_hits:
            matchable += 1
        if len(anchor_hits) == 1:
            anchored_unique += 1

    patterns = _read_hub_patterns(g, hub_r, hub_c, bg)
    return {
        "components": int(len(comps)),
        "ambiguous_matchable_components": int(ambiguous),
        "anchor_resolved_components": int(resolved),
        "anchored_unique_components": int(anchored_unique),
        "shape_matchable_components": int(matchable),
        "non_bg_colors": int(len(set(map(int, g[g != bg].flatten())))),
        "unique_hub_patterns": int(len(set(patterns.values()))),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR
    palette = [c for c in range(10) if c != bg]
    body_color = rng.choice(palette)
    anchor_color = rng.choice([c for c in palette if c != body_color])

    H = rng.randint(20, 28)
    W = rng.randint(22, 28)

    sprite_sides = list(GENERATOR_SIDES)
    K = len(sprite_sides)

    sprite_masks: dict[str, np.ndarray] = {}
    sprite_arrs: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(PATTERNS[side], side, rng)
        if mask is None:
            return None
        arr = _paint_anchor(mask, side, body_color, anchor_color, bg)
        if sorted(_anchor_hits(arr, bg, body_color)) != [(0, side)]:
            return None
        sprite_masks[side] = mask
        sprite_arrs[side] = arr

    max_sprite_h = max(arr.shape[0] for arr in sprite_arrs.values())
    max_sprite_w = max(arr.shape[1] for arr in sprite_arrs.values())
    margin = DOCK_THICKNESS + max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    _draw_hub(out, hub_r, hub_c, body_color, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _attach_sprite(out, sprite_arrs[side], side, hub_r, hub_c, bg)
        if placed is None:
            return None
        attached[side] = placed

    inp = np.full((H, W), bg, dtype=int)
    _draw_hub(inp, hub_r, hub_c, body_color, bg)
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached[side]
        k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, k)
        rot_mask = np.rot90(mask, k)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        patch = np.where(rot_mask, rot_arr, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = variant_metrics(inp)
    if metrics["non_bg_colors"] != 2:
        return None
    if metrics["unique_hub_patterns"] != 4:
        return None
    if metrics["ambiguous_matchable_components"] < K:
        return None
    if metrics["anchored_unique_components"] < K:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
