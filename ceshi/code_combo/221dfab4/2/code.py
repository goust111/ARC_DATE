# Combo variant of 221dfab4 (index 2)
# combo_spec: D3/3+D8/3
# Name: 颜色配对+碰撞颜色互换

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask


# Design analysis for the combo variant
# a) 具体化与行为追踪
#    - 仍保留原题几何骨架：黄色边缘线段给出推进方向；条纹每隔 1 行/列出现一次；
#      第 0,1,3,4,... 条只画在线段 span 内，第 2,5,8,... 条仍是全行/列 footprint。
#    - 融合后的关键变化是“颜色不再按 Y-Y-G 固定循环”，而是沿轨迹维护一个 carry 颜色状态。
#      初始 carry 为黄色 4。若当前条纹命中某个前景 blob，则该条纹吸收 blob 当前可见颜色：
#      * 条纹本步显示为 blob 颜色；
#      * blob 被本步进入前的 carry 颜色重着色；
#      * 下一步的 carry 更新为 blob 颜色。
#    - 小例子：假设自顶向下推进，step1 的 span-limited 条纹用 4 行进并碰到红 blob。
#      原题会把该行 span 直接涂成 4；D8 单变体会让碰撞格显示红，但下一条仍按固定周期走。
#      本融合变体中，step1 的整条 span 条纹都转成红，blob 的其余像素被改成 4；
#      step2（全行条纹）将带着红继续前进。若 step2 再碰到蓝 blob，则 step2 显示蓝，
#      蓝 blob 变红，step3 以后继续带蓝。解题者必须同时追踪“交换”与“后续继承”。
# b) 复杂度校准
#    - 如果 generator 允许一条 stripe 同时撞多个 blob，carry 的来源会变模糊；因此生成器只把
#      每个目标 stripe 锚定到一个 blob，并用 1-cell padding 保证组件彼此分离。
#    - 为了让 D3/3 真正可见，样例强制至少 5 个碰撞步，其中既有 span-limited，也有 full-row/col，
#      并要求至少 3 种 blob 颜色、至少 3 次 carry 改变。这样不会退化成固定 Y-Y-G 或纯 D8 交换。
#    - blob 用小型 L 形 / 2x2 形状，既足够稳定，又能让“被撞到的格子”和“组件其余像素”同时可见。
# c) 隐式假设审查
#    - 原 solver 的 `cycle = [4,4,3]` 假设条纹颜色完全由步数决定；这里失效，颜色必须由先前碰撞决定。
#    - 原 solver 的 green 处理 `np.where(out[pos,:] != bg, 3, ...)` 假设第三步一定是绿色 flood；
#      这里保留“第三步是 full footprint”的几何语义，但其颜色由 carry / pairing 决定，不再固定为 3。
#    - D8 单变体把对象身份固定在输入上是合理的；本融合也继续固定“blob 的几何组件”来自输入，
#      否则新画出的条纹会和 blob 粘连，后续连通域会失真。变化的是这些固定组件上的当前颜色。
#    - 新 solver 额外假设：当一条 stripe 真正承担教学信号时，只会撞到一个 blob 颜色。
#      代码对多命中仍给出确定结果（按 stripe 上的首次命中决定 carry），而生成器会避免这种歧义。
# d) 放置合法性
#    - 生成器只放置手工构造的小 sprite，逐个检查越界、重叠和 8 邻域 padding，因此组件不会粘连。
#    - span-limited blob 刻意跨越 span 边界，只让 1 个命中单元落在 span 内，其余像素留在 span 外；
#      这样碰撞后能看到“条纹吸色”和“blob 释放旧色”两个方向。
#    - full-footprint blob 放在 span 外，但与目标 full row/column 对齐，确保必须依赖“全行/列”才能碰撞。
# e) 视觉可观察性
#    - 观察者会稳定看到：某条 stripe 撞到某色 blob 后，当前 stripe 和下一条 stripe 都改带该色；
#      同时该 blob 的剩余像素被改成上一条的颜色。这个现象会连续发生 4-5 次，足以从 2-4 个样例中归纳。
#    - 生成器还显式拒绝弱样例：若融合输出与“无 carry 传播”的对照版本太接近，或与原题输出差异过小，
#      就重采样，从而避免退化成单一变体或原任务。


Y = 4
G = 3
FULL_PERIOD = 3


def _segment_info(g: np.ndarray):
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(Y)) & border
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    source_mask = lab == int(best_id)
    pts = np.argwhere(source_mask)
    if pts.size == 0:
        return None

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        cols = sorted(int(c) for c in cs)
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span": list(range(int(cols[0]), int(cols[-1]) + 1)),
            "span0": int(cols[0]),
            "span1": int(cols[-1]),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": int(H),
            "cross_limit": int(W),
            "source_mask": source_mask,
        }

    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span": list(range(int(rows[0]), int(rows[-1]) + 1)),
            "span0": int(rows[0]),
            "span1": int(rows[-1]),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": int(W),
            "cross_limit": int(H),
            "source_mask": source_mask,
        }

    return None


def _stripe_steps(info: dict) -> list[dict[str, int | bool]]:
    steps: list[dict[str, int | bool]] = []
    step = 0
    while True:
        pos = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step))
        if not (0 <= pos < int(info["limit"])):
            break
        steps.append(
            {
                "step": int(step),
                "pos": int(pos),
                "is_full": bool(int(step) % FULL_PERIOD == FULL_PERIOD - 1),
            }
        )
        step += 1
    return steps


def _footprint_and_gap(info: dict, step: int, pos: int, is_full: bool):
    orient = str(info["orient"])
    span = list(info["span"])
    start_pos = int(info["start_pos"])
    dir1 = int(info["dir1"])

    if orient == "h":
        cols = list(range(info["cross_limit"])) if is_full else span
        footprint = [(int(pos), int(c)) for c in cols]
        gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
        gap_cells = [(int(gap), int(c)) for c in span] if 0 <= gap < int(info["limit"]) else []
        return footprint, gap_cells

    rows = list(range(info["cross_limit"])) if is_full else span
    footprint = [(int(r), int(pos)) for r in rows]
    gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
    gap_cells = [(int(r), int(gap)) for r in span] if 0 <= gap < int(info["limit"]) else []
    return footprint, gap_cells


def _ordered_hit_ids(footprint: list[tuple[int, int]], obj_lab: np.ndarray) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for r, c in footprint:
        cid = int(obj_lab[r, c])
        if cid > 0 and cid not in seen:
            ordered.append(int(cid))
            seen.add(int(cid))
    return ordered


def _original_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    info = _segment_info(g)
    if info is None:
        return out

    cycle = [int(Y), int(Y), int(G)]
    for item in _stripe_steps(info):
        step = int(item["step"])
        pos = int(item["pos"])
        col = int(cycle[step % 3])
        footprint, gap_cells = _footprint_and_gap(info, step, pos, bool(item["is_full"]))

        if bool(item["is_full"]):
            if info["orient"] == "h":
                out[pos, :] = np.where(out[pos, :] != int(bg), int(col), out[pos, :])
            else:
                out[:, pos] = np.where(out[:, pos] != int(bg), int(col), out[:, pos])

        for r, c in footprint:
            out[int(r), int(c)] = int(col)
        for r, c in gap_cells:
            out[int(r), int(c)] = int(bg)

    return out


def _apply_pair_swap(grid: np.ndarray, dynamic_carry: bool, collect_stats: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    info = _segment_info(g)
    if info is None:
        return (out, {}) if collect_stats else out

    obj_mask = (g != int(bg)) & (~info["source_mask"])
    obj_lab, n_obj = label_mask(obj_mask, connectivity=4)
    component_masks = [obj_lab == cid for cid in range(1, int(n_obj) + 1)]

    carry = int(Y)
    cycle = [int(Y), int(Y), int(G)]

    collision_steps = 0
    carry_changes = 0
    recolored_object_pixels = 0
    draw_colors: list[int] = []
    incoming_colors: list[int] = []
    hit_colors: list[int] = []
    paired_components: list[int] = []

    for item in _stripe_steps(info):
        step = int(item["step"])
        pos = int(item["pos"])
        is_full = bool(item["is_full"])
        footprint, gap_cells = _footprint_and_gap(info, step, pos, is_full)
        footprint_mask = np.zeros((H, W), dtype=bool)
        for r, c in footprint:
            footprint_mask[int(r), int(c)] = True

        incoming = int(carry if dynamic_carry else cycle[step % 3])
        draw_color = int(incoming)
        hit_ids = _ordered_hit_ids(footprint, obj_lab)
        incoming_colors.append(int(incoming))

        if hit_ids:
            collision_steps += 1
            first_cid = int(hit_ids[0])
            first_cell = next((r, c) for r, c in footprint if int(obj_lab[r, c]) == int(first_cid))
            paired_color = int(out[int(first_cell[0]), int(first_cell[1])])
            draw_color = int(paired_color)
            hit_colors.append(int(paired_color))
            paired_components.append(int(first_cid))
            if int(paired_color) != int(incoming):
                carry_changes += 1

            for cid in hit_ids:
                comp_mask = component_masks[int(cid) - 1]
                recolor_mask = comp_mask & (~footprint_mask)
                recolored_object_pixels += int(np.sum(recolor_mask & (out != int(incoming))))
                out[recolor_mask] = int(incoming)

            if dynamic_carry:
                carry = int(paired_color)
        elif dynamic_carry:
            carry = int(carry)

        for r, c in footprint:
            if int(obj_lab[r, c]) == 0:
                out[int(r), int(c)] = int(draw_color)

        for r, c in gap_cells:
            out[int(r), int(c)] = int(bg)

        draw_colors.append(int(draw_color))

    if not collect_stats:
        return out

    stats = {
        "collision_steps": int(collision_steps),
        "carry_changes": int(carry_changes),
        "recolored_object_pixels": int(recolored_object_pixels),
        "distinct_draw_colors": int(len(set(draw_colors))),
        "draw_colors": list(draw_colors),
        "incoming_colors": list(incoming_colors),
        "hit_colors": list(hit_colors),
        "paired_components": list(paired_components),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source stripes while carrying a mutable trajectory color.

    Combo rule:
    - geometry stays close to the original task: stripes still appear every two cells, and every third
      stripe still uses the full row/column footprint;
    - color is stateful: the trajectory starts yellow, then a collision swaps colors with the hit blob;
    - swap semantics:
      * the stripe footprint keeps the blob's visible color at the overlap cells and shows that color on
        the rest of the current stripe;
      * the remainder of the blob component is recolored to the stripe's incoming color;
      * later stripes inherit the absorbed color, creating the D3+D8 cascade.
    """
    return _apply_pair_swap(grid, dynamic_carry=True, collect_stats=False)


def _can_place(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> bool:
    H, W = canvas.shape
    h, w = sprite.shape
    if top < 1 or left < 1 or top + h > H - 1 or left + w > W - 1:
        return False

    mask = sprite != int(bg)
    occ = np.argwhere(mask)
    for sr, sc in occ:
        gr = int(top + sr)
        gc = int(left + sc)
        for rr in range(max(0, gr - 1), min(H, gr + 2)):
            for cc in range(max(0, gc - 1), min(W, gc + 2)):
                if int(canvas[rr, cc]) == int(bg):
                    continue
                local_r = rr - int(top)
                local_c = cc - int(left)
                if 0 <= local_r < h and 0 <= local_c < w and bool(mask[local_r, local_c]):
                    continue
                return False
    return True


def _paste(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = sprite != int(bg)
    h, w = sprite.shape
    for dr in range(h):
        for dc in range(w):
            if bool(mask[dr, dc]):
                canvas[int(top + dr), int(left + dc)] = int(sprite[dr, dc])


def _limited_sprite(orient: str, side: str, color: int, bg: int) -> np.ndarray:
    sprite = np.full((2, 2), int(bg), dtype=int)
    if orient == "h":
        if side == "left":
            sprite[0, 0] = int(color)
            sprite[0, 1] = int(color)
            sprite[1, 0] = int(color)
        else:
            sprite[0, 0] = int(color)
            sprite[0, 1] = int(color)
            sprite[1, 1] = int(color)
        return sprite

    if side == "top":
        sprite[0, 0] = int(color)
        sprite[1, 0] = int(color)
        sprite[0, 1] = int(color)
    else:
        sprite[0, 0] = int(color)
        sprite[1, 0] = int(color)
        sprite[1, 1] = int(color)
    return sprite


def _full_sprite(color: int, bg: int) -> np.ndarray:
    return np.full((2, 2), int(color), dtype=int)


def _place_limited_blob(
    canvas: np.ndarray,
    info: dict,
    pos: int,
    color: int,
    bg: int,
    rng: random.Random,
) -> bool:
    orient = str(info["orient"])
    span0 = int(info["span0"])
    span1 = int(info["span1"])
    dir1 = int(info["dir1"])
    H, W = canvas.shape

    if orient == "h":
        candidates: list[tuple[str, int, int]] = []
        if span0 - 1 >= 1 and 1 <= pos + dir1 < H - 1:
            candidates.append(("left", int(pos), int(span0 - 1)))
        if span1 + 1 < W - 1 and 1 <= pos + dir1 < H - 1:
            candidates.append(("right", int(pos), int(span1)))
        rng.shuffle(candidates)
        for side, top, left in candidates:
            sprite = _limited_sprite("h", side, color, bg)
            if _can_place(canvas, sprite, top, left, bg):
                _paste(canvas, sprite, top, left, bg)
                return True
        return False

    candidates_v: list[tuple[str, int, int]] = []
    if span0 - 1 >= 1 and 1 <= pos + dir1 < W - 1:
        candidates_v.append(("top", int(span0 - 1), int(pos)))
    if span1 + 1 < H - 1 and 1 <= pos + dir1 < W - 1:
        candidates_v.append(("bottom", int(span1), int(pos)))
    rng.shuffle(candidates_v)
    for side, top, left in candidates_v:
        sprite = _limited_sprite("v", side, color, bg)
        if _can_place(canvas, sprite, top, left, bg):
            _paste(canvas, sprite, top, left, bg)
            return True
    return False


def _outside_ranges(start: int, end: int, max_start: int) -> list[int]:
    vals = [v for v in range(1, max_start + 1) if v + 1 < start or v > end]
    return [int(v) for v in vals]


def _place_full_blob(
    canvas: np.ndarray,
    info: dict,
    pos: int,
    color: int,
    bg: int,
    rng: random.Random,
) -> bool:
    orient = str(info["orient"])
    span0 = int(info["span0"])
    span1 = int(info["span1"])
    dir1 = int(info["dir1"])
    H, W = canvas.shape
    sprite = _full_sprite(color, bg)

    if orient == "h":
        if not (1 <= pos + dir1 < H - 1):
            return False
        candidates = _outside_ranges(span0, span1, W - 3)
        rng.shuffle(candidates)
        for left in candidates:
            if _can_place(canvas, sprite, int(pos), int(left), bg):
                _paste(canvas, sprite, int(pos), int(left), bg)
                return True
        return False

    if not (1 <= pos + dir1 < W - 1):
        return False
    candidates = _outside_ranges(span0, span1, H - 3)
    rng.shuffle(candidates)
    for top in candidates:
        if _can_place(canvas, sprite, int(top), int(pos), bg):
            _paste(canvas, sprite, int(top), int(pos), bg)
            return True
    return False


def _select_target_steps(stripe_steps: list[dict[str, int | bool]]):
    limited = [item for item in stripe_steps if int(item["step"]) > 0 and not bool(item["is_full"])]
    full = [item for item in stripe_steps if int(item["step"]) > 0 and bool(item["is_full"])]
    if len(limited) < 3 or len(full) < 2:
        return None
    return [limited[0], full[0], limited[1], full[1], limited[2]]


def _color_sequence(rng: random.Random, pool: list[int], n: int) -> list[int]:
    palette_size = min(len(pool), max(3, min(5, n)))
    palette = rng.sample(pool, palette_size)
    seq: list[int] = []
    idx = rng.randrange(palette_size)
    for _ in range(n):
        seq.append(int(palette[idx]))
        idx = (idx + rng.choice([1, 2])) % palette_size
    if len(set(seq)) < 3:
        seq[-1] = int(palette[-1])
    for i in range(1, len(seq)):
        if int(seq[i]) == int(seq[i - 1]):
            seq[i] = int(palette[(palette.index(seq[i]) + 1) % palette_size])
    return seq


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    usable = [c for c in range(10) if c not in (Y, G)]
    bg = int(rng.choice(usable))
    blob_pool = [c for c in usable if c != int(bg)]
    if len(blob_pool) < 3:
        return None

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(4, 7)

    grid = np.full((H, W), int(bg), dtype=int)
    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(int(seg_start), int(seg_start + seg_len)))
        edge_row = 0 if edge == "top" else H - 1
        for c in span:
            grid[int(edge_row), int(c)] = int(Y)
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(int(seg_start), int(seg_start + seg_len)))
        edge_col = 0 if edge == "left" else W - 1
        for r in span:
            grid[int(r), int(edge_col)] = int(Y)

    info = _segment_info(grid)
    if info is None:
        return None
    stripe_steps = _stripe_steps(info)
    targets = _select_target_steps(stripe_steps)
    if targets is None:
        return None

    canvas = grid.copy()
    hit_colors = _color_sequence(rng, blob_pool, len(targets))

    for item, color in zip(targets, hit_colors):
        pos = int(item["pos"])
        ok = (
            _place_full_blob(canvas, info, pos, int(color), bg, rng)
            if bool(item["is_full"])
            else _place_limited_blob(canvas, info, pos, int(color), bg, rng)
        )
        if not ok:
            return None

    fg_mask = (canvas != int(bg)) & (~info["source_mask"])
    _lab, n_obj = label_mask(fg_mask, connectivity=4)
    if int(n_obj) != len(targets):
        return None

    inp = canvas.copy()
    out, stats = _apply_pair_swap(inp, dynamic_carry=True, collect_stats=True)
    no_cascade = _apply_pair_swap(inp, dynamic_carry=False, collect_stats=False)
    original = _original_output(inp)

    if int(get_bg(inp)) != int(bg):
        return None
    if stats["collision_steps"] != len(targets):
        return None
    if stats["carry_changes"] < 3:
        return None
    if stats["recolored_object_pixels"] < 5:
        return None
    if len(set(stats["hit_colors"])) < 3:
        return None

    delta_no_cascade = int(np.sum(out != no_cascade))
    delta_original = int(np.sum(out != original))
    if delta_no_cascade < 8:
        return None
    if delta_original < 12:
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
