# Variant of 7ed72f31
# Dimension: D16 — 精灵与标记之间的基数关系 (meta)
# Change: 一对一关系：每个标记对应一个精灵（或一组8邻域精灵），每个精灵有且只有一个标记... → 标记之间的配对关系驱动两步反射

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask

# -----------------------------------------------------------------------------
# Design analysis for the D16 variant
# -----------------------------------------------------------------------------
# design_analysis:
#   variant_summary: |
#     原题里每个精灵只看“自己那一个红标记”，做一次镜像/点反射。
#     本变体把 cardinality 改成“两个红标记共同控制一个精灵”：
#     输入里每个精灵现在对应一对平行红线，
#     其中第一条红线与 payload 4 连通（活动标记），第二条红线是纯红孤立线（配对标记）。
#     输出不是只画一步镜像，而是把 payload 先沿第一条红线反射，再把第一步结果沿第二条红线再反射一次。
#     因此输出会同时出现：
#       1. 原始 payload
#       2. 第一跳镜像结果（位于两条红线之间）
#       3. 第二跳镜像结果（位于第二条红线另一侧）
#   concrete_pixel_form: |
#     只使用原题里最稳定、最可见的“嵌入式横/竖红线”子类，不再生成孤立红点。
#     每个对象都是单色不对称小精灵 + 两条同长度、同方向、完全对齐的红线。
#     若第一条是横线，则 payload 全在其上方或下方；第二条横线位于第一次反射后的前进方向上。
#     若第一条是竖线，则 payload 全在其左侧或右侧；第二条竖线位于第一次反射后的前进方向上。
#   behavior_trace: |
#     例子：payload 在横红线 r=5 上方，像素 (4, 10) 为蓝色；第二条配对横红线在 r=10。
#     原题逻辑：只看嵌入的那条横线，于是 (4,10) -> (6,10)，只生成一步。
#     本变体逻辑：
#       第一步：沿 r=5 反射，(4,10) -> (6,10)
#       第二步：把第一步结果再沿 r=10 反射，(6,10) -> (14,10)
#     所以输出中会同时保留原像素 (4,10)，并新增 (6,10) 与 (14,10)。
#     差异的核心不在“反射公式”本身，而在“第二条标记是否参与，并且参与顺序由配对决定”。
#   complexity_calibration: |
#     如果只输出第二步结果，那么两次平行反射会退化成“平移”，解题者未必能意识到是两条红线的链式作用。
#     因此本变体显式保留第一步结果，让“两条红线 + 中间副本 + 远端副本”的结构直接可见。
#     同时只保留横/竖红线两档，避免点反射和线反射混杂导致 2-3 个样例无法归纳。
#     每个网格固定 3 对标记（共 6 条红线），既能反复支撑配对规则，又不会把画面堆得太满。
#   implicit_assumption_review: |
#     原 solver 与变体相关的旧假设主要有四组：
#       1. “含红组件里的红像素自己就定义完整变换”：
#          原代码在发现一个 mixed component 后，立刻从该组件的红像素推断 axis，然后单独完成一次反射。
#          新变体中这不再成立；mixed component 只给出第一条轴，第二条轴来自别的 red-only component。
#       2. “可反射目标只需从当前组件 bbox/局部邻域筛选”：
#          原代码通过 bbox 或 1-cell 扩张邻域来界定 payload。
#          新变体仍然只反射 mixed component 自己的 payload，但第二步位置不再局限在该 bbox 附近。
#          因此 solver 不能继续用“遍历全图 + local eligible”那套隐含单步局部性的写法。
#       3. “孤立红像素一定是点反射中心”：
#          原题第 2 段专门处理 1 像素红点，并把 8 邻域同色连通块做点反射。
#          变体里配对标记是整条纯红线，不是点；保留那段逻辑会把配对线误解成无 payload 的旧标记类型。
#       4. “每个红标记只会被用一次，且不会和别的红标记形成更高层关系”：
#          新变体里必须显式做 pairing。solver 需要检查候选纯红线是否同方向、同跨度、位于第一步前进方向，
#          以及两线之间的条带在输入中为空，从而确保真的是这两个标记配成一对。
#   placement_legality: |
#     生成器按 output-first 构造完整输出 patch：payload + 活动红线 + 第一步镜像 + 配对红线 + 第二步镜像。
#     然后放置完整 patch，确保：
#       - patch 整体不越界
#       - 不与别的 patch 重叠
#       - 反射得到的两个副本都落在背景上
#       - 输入态中删除第一步/第二步副本后，活动红线仍与 payload 4 连通，配对红线仍保持纯红孤立线
#     配对线与活动线之间至少留 1 格空隙，因此输入中二者之间的条带全是背景，solver 可用此性质唯一配对。
#   visual_observability: |
#     解题者从样例里能直接看到：
#       - 一个 payload 旁边不是一条红线，而是“同向的两条红线”
#       - 输出多了两份复制：一份落在两线之间，一份落在第二条线之外
#       - 无论横向还是纵向，复制顺序都与“先过近线，再过远线”一致
#     这些都是纯空间线索，不依赖颜色编号、坐标奇偶、排序编号等不可见因素。
#   visual_cleanliness: |
#     每个 patch 只包含一种 payload 颜色和两条红线，层次固定为 5 层（原图 / 线1 / 中间副本 / 线2 / 远端副本）。
#     patch 之间用 8 邻域 padding 隔开，避免红线与别的对象视觉缠绕。
#   color_reasoning: |
#     颜色不参与配对和排序。颜色只用于把不同 pair 的精灵区分开。
#     每个网格使用 3 个不同 payload 颜色，足以让人看见“同一种颜色沿两条线传播”，
#     又不会因为颜色太多而掩盖真正的 D16 规律。
#   solver_assumptions_backed_by_generator: |
#     新 solver 的关键假设，以及 generator 对它们的保证：
#       - 活动标记一定是连续横线或竖线：生成器只造连续直线。
#       - payload 一定完整地位于活动轴的一侧：canonical patch + 翻转保持此性质。
#       - 配对标记与活动标记同方向、同跨度：生成器直接复制同一条线的长度与对齐。
#       - 真正配对的两线之间在输入态为空：生成器删除中间副本后保留纯背景条带。
#       - 每条纯红配对线只服务一个活动标记：每个 patch 恰有一条活动线和一条配对线。

MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 22, 28
PAIR_COUNT = 3

# Payload geometry
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 4
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 9
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Distance between the first and second axes.
PAIR_GAP_MIN, PAIR_GAP_MAX = 1, 2

# Colors
PALETTE_COLORS = [c for c in range(10) if c != MARK]

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 600

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    """Grow a 4-connected mask with a tight bbox."""
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier: list[tuple[int, int]] = []
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
    if not mask.any(axis=0).all() or not mask.any(axis=1).all():
        return None
    return mask


def _classify_red_line(reds: list[tuple[int, int]]) -> dict | None:
    reds = [(int(r), int(c)) for r, c in reds]
    if len(reds) < 2:
        return None
    rs = {r for r, _ in reds}
    cs = {c for _, c in reds}
    if len(rs) == 1:
        row = next(iter(rs))
        c0, c1 = min(cs), max(cs)
        if len(reds) != c1 - c0 + 1:
            return None
        if cs != set(range(c0, c1 + 1)):
            return None
        return {"kind": "row", "pos": int(row), "span": (int(c0), int(c1))}
    if len(cs) == 1:
        col = next(iter(cs))
        r0, r1 = min(rs), max(rs)
        if len(reds) != r1 - r0 + 1:
            return None
        if rs != set(range(r0, r1 + 1)):
            return None
        return {"kind": "col", "pos": int(col), "span": (int(r0), int(r1))}
    return None


def _analyze_scene(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    lab, n = label_mask(g != int(bg), connectivity=4)
    active: list[dict] = []
    passive: list[dict] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        vals = [int(g[r, c]) for r, c in coords]
        reds = [(r, c) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue
        axis = _classify_red_line(reds)
        if axis is None:
            continue
        payload = [(r, c, int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        rec = {
            "kind": axis["kind"],
            "pos": int(axis["pos"]),
            "span": tuple(map(int, axis["span"])),
            "reds": reds,
            "payload": payload,
        }
        if payload:
            active.append(rec)
        else:
            passive.append(rec)
    active.sort(key=lambda rec: (rec["kind"], rec["pos"], rec["span"]))
    passive.sort(key=lambda rec: (rec["kind"], rec["pos"], rec["span"]))
    return active, passive


def _travel_direction(active: dict) -> int | None:
    if active["kind"] == "row":
        rows = [r for r, _c, _v in active["payload"]]
        if max(rows) < int(active["pos"]):
            return 1
        if min(rows) > int(active["pos"]):
            return -1
        return None
    cols = [c for _r, c, _v in active["payload"]]
    if max(cols) < int(active["pos"]):
        return 1
    if min(cols) > int(active["pos"]):
        return -1
    return None


def _strip_is_bg(g: np.ndarray, bg: int, axis0: dict, axis1: dict) -> bool:
    if axis0["kind"] != axis1["kind"] or axis0["span"] != axis1["span"]:
        return False
    if axis0["kind"] == "row":
        r0, r1 = sorted([int(axis0["pos"]), int(axis1["pos"])])
        c0, c1 = map(int, axis0["span"])
        if r1 - r0 <= 1:
            return True
        return bool(np.all(g[r0 + 1 : r1, c0 : c1 + 1] == int(bg)))
    c0, c1 = sorted([int(axis0["pos"]), int(axis1["pos"])])
    r0, r1 = map(int, axis0["span"])
    if c1 - c0 <= 1:
        return True
    return bool(np.all(g[r0 : r1 + 1, c0 + 1 : c1] == int(bg)))


def _reflect_once(axis: dict, r: int, c: int) -> tuple[int, int]:
    if axis["kind"] == "row":
        return int(2 * int(axis["pos"]) - int(r)), int(c)
    return int(r), int(2 * int(axis["pos"]) - int(c))


def _find_pairs(g: np.ndarray, bg: int) -> list[tuple[dict, dict]]:
    active, passive = _analyze_scene(g, bg)
    pairs: list[tuple[dict, dict]] = []
    used: set[int] = set()
    for act in active:
        travel = _travel_direction(act)
        if travel is None:
            continue
        candidates: list[tuple[int, int, dict]] = []
        for idx, pas in enumerate(passive):
            if idx in used:
                continue
            if pas["kind"] != act["kind"] or pas["span"] != act["span"]:
                continue
            delta = int(pas["pos"]) - int(act["pos"])
            if delta * int(travel) <= 0:
                continue
            if not _strip_is_bg(g, bg, act, pas):
                continue
            candidates.append((abs(delta), idx, pas))
        if not candidates:
            continue
        _, idx, partner = min(candidates, key=lambda item: item[0])
        used.add(int(idx))
        pairs.append((act, partner))
    return pairs


def _variant_metrics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    pairs = _find_pairs(g, bg)
    distances = [abs(int(a["pos"]) - int(b["pos"])) for a, b in pairs]
    return {
        "pair_count": len(pairs),
        "row_pairs": sum(1 for a, _b in pairs if a["kind"] == "row"),
        "col_pairs": sum(1 for a, _b in pairs if a["kind"] == "col"),
        "distances": distances,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Mixed component: payload + one red line = first reflection axis.
    - Pair it with the nearest pure-red line of the same orientation/span on the reflected side,
      with only background between the two lines in the input.
    - Draw both the first reflection result and the second reflection result.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < h and 0 <= int(c) < w

    for active, partner in _find_pairs(g, bg):
        step1: list[tuple[int, int, int]] = []
        for r, c, v in active["payload"]:
            rr, cc = _reflect_once(active, int(r), int(c))
            if inb(rr, cc):
                out[int(rr), int(cc)] = int(v)
                step1.append((int(rr), int(cc), int(v)))
        for r, c, v in step1:
            rr, cc = _reflect_once(partner, int(r), int(c))
            if inb(rr, cc):
                out[int(rr), int(cc)] = int(v)
    return out


def _build_pair_sprite(rng: random.Random) -> dict | None:
    """Build a full output patch for one pair-controlled chain reflection."""
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    gap = rng.randint(PAIR_GAP_MIN, PAIR_GAP_MAX)
    orientation = rng.choice(["row", "col"])

    if orientation == "row":
        h, w = 3 * ph + gap + 1, pw
        payload = np.zeros((h, w), dtype=bool)
        step1 = np.zeros((h, w), dtype=bool)
        step2 = np.zeros((h, w), dtype=bool)
        active_red = np.zeros((h, w), dtype=bool)
        passive_red = np.zeros((h, w), dtype=bool)
        payload[:ph, :] = pmask
        active_red[ph, :] = True
        step1[ph + 1 : 2 * ph + 1, :] = pmask[::-1, :]
        partner_row = 2 * ph + gap
        passive_red[partner_row, :] = True
        step2[partner_row + 1 : partner_row + 1 + ph, :] = pmask
    else:
        h, w = ph, 3 * pw + gap + 1
        payload = np.zeros((h, w), dtype=bool)
        step1 = np.zeros((h, w), dtype=bool)
        step2 = np.zeros((h, w), dtype=bool)
        active_red = np.zeros((h, w), dtype=bool)
        passive_red = np.zeros((h, w), dtype=bool)
        payload[:, :pw] = pmask
        active_red[:, pw] = True
        step1[:, pw + 1 : 2 * pw + 1] = pmask[:, ::-1]
        partner_col = 2 * pw + gap
        passive_red[:, partner_col] = True
        step2[:, partner_col + 1 : partner_col + 1 + pw] = pmask

    if rng.random() < 0.5:
        payload = np.flipud(payload)
        step1 = np.flipud(step1)
        step2 = np.flipud(step2)
        active_red = np.flipud(active_red)
        passive_red = np.flipud(passive_red)
    if rng.random() < 0.5:
        payload = np.fliplr(payload)
        step1 = np.fliplr(step1)
        step2 = np.fliplr(step2)
        active_red = np.fliplr(active_red)
        passive_red = np.fliplr(passive_red)

    return {
        "payload": payload,
        "step1": step1,
        "step2": step2,
        "active_red": active_red,
        "passive_red": passive_red,
        "footprint": payload | step1 | step2 | active_red | passive_red,
    }


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
) -> tuple[int, int] | None:
    h, w = patch_mask.shape
    H, W = canvas.shape
    if h > H or w > W:
        return None
    occ = canvas != int(bg)
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((int(r0), int(c0)))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_output(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    colored = spr["payload"] | spr["step1"] | spr["step2"]
    h, w = colored.shape
    canvas[r0 : r0 + h, c0 : c0 + w][colored] = int(spr["color"])
    canvas[r0 : r0 + h, c0 : c0 + w][spr["active_red"]] = int(MARK)
    canvas[r0 : r0 + h, c0 : c0 + w][spr["passive_red"]] = int(MARK)


def _erase_generated(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = int(spr["r0"]), int(spr["c0"])
    h, w = spr["payload"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["step1"] | spr["step2"]] = int(bg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice(PALETTE_COLORS)
    palette = [c for c in PALETTE_COLORS if c != bg]
    chosen = rng.sample(palette, PAIR_COUNT)

    sprites: list[dict] = []
    for color in chosen:
        spr = _build_pair_sprite(rng)
        if spr is None:
            return None
        spr["color"] = int(color)
        sprites.append(spr)

    output = np.full((n, n), int(bg), dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        loc = _random_free_location(output, spr["footprint"], int(bg), rng)
        if loc is None:
            return None
        spr["r0"], spr["c0"] = map(int, loc)
        _stamp_output(output, spr, int(spr["r0"]), int(spr["c0"]))

    inp = output.copy()
    for spr in sprites:
        _erase_generated(inp, spr, int(bg))

    if (inp == int(bg)).sum() / inp.size < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None

    metrics = _variant_metrics(inp)
    if metrics["pair_count"] != PAIR_COUNT:
        return None
    if not metrics["distances"] or min(metrics["distances"]) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
