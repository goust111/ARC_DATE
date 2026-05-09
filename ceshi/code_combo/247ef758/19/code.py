# Combo variant of 247ef758 (index 19)
# combo_spec: D2/1+D4/3+D12/0
# Name: 空心精灵+实心化+左上对齐

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
- 基础布局保持原题结构: 左侧精灵条带 | 单列分隔线 | 右侧矩形框。框四边仍通过“左右同色=目标行、上下同色=目标列”
  给出每个 mover 的目标交点集合。
- 融合后左侧 mover 不再是普通实心 blob, 而是带内部洞的单色 8 连通图形。比如一个 4x4 mover 可以是
  `1111 / 1001 / 1011 / 1111`; 它有至少一个被前景包围的背景像素。
- 原题在目标 `(tr, tc)` 处把精灵按 bbox 中心对齐并只印刻真实前景像素。融合后改成两步联动:
  1. 目标点直接当作精灵 bbox 左上角。
  2. 不再按稀疏 mask 印刻, 而是把该 bbox 的整个矩形区域涂满该色。
- 因而同一个左侧空心 4x4 精灵, 若目标点是 `(5, 9)`, 原题会在 `(3..6, 7..10)` 或 `(4..7, 8..11)`
  一类中心对齐位置印刻空心图案; 本变体则在 `rows 5..8, cols 9..12` 直接得到实心 4x4 矩形。
- 差异是可叠加的而不是各改各的: 输入侧通过“空心”提供强证据, 输出侧同时表现为“洞消失”与“整体偏到目标交点
  的左上角”。

b) 复杂度校准
- 若 mover 是实心矩形, D4/3 会退化得不可见; 因此生成器强制所有 mover 都带洞。
- 若目标仍按中心合法范围采样, D12/0 会被弱化; 因此生成器按左上锚点合法范围选标记。
- 若不同 mover 的目标太近, 实心化后的矩形会大面积互相遮挡, 样例会变乱; 因此行/列标记按 bbox 高宽做跨颜色间距控制。
- 仍只保留 2-3 个 mover 与 0-1 个 distractor, 并且颜色一一对应到标记, 让规则主干保持低熵可推断。

c) 隐式假设审查
- `largest_enclosed_component_bbox(...)`: 仍假设框内部是最大封闭背景区。空心精灵只出现在左条带, 不会干扰该假设。
- `divider_mask` 检测: 仍依赖左侧存在近乎整列的统一非背景色分隔线。生成器保持此结构不变。
- 左侧 sprite 提取: 原代码只要求“按颜色的 8 连通分量”。空心图形虽然含洞, 但前景仍是单个 8 连通分量, 所以这段逻辑仍成立。
- 擦除原精灵: 只能擦除 `mask` 为真的像素, 不能擦 bbox, 否则会错误清掉洞周围本来就是背景的区域。这里保持逐前景擦除。
- 目标印刻: 原题 `dr/dc = target - bbox_center` 依赖中心锚点; 融合后必须改成 `base = target`。
- 绘制内容: 原题遍历前景像素; 融合后必须遍历/切片 bbox 矩形, 否则 D4/3 不生效。
- 覆盖顺序: 仍按输入中“更低的 mover 先画、更高的后画”。这个排序依赖可见的纵向位置, 不依赖颜色 id。

d) 放置合法性
- mover 的目标行列范围按左上锚点计算: `ir0 <= tr <= ir1-h+1`, `ic0 <= tc <= ic1-w+1`, 保证完整实心矩形能落在框内。
- 同一 mover 的多个行标记至少相隔自身高, 多个列标记至少相隔自身宽; 不同 mover 之间也按各自跨度做 clearance。
- 左条带中的所有 sprites 逐个竖向堆叠, 每个至少保留 3 行高度预算, 因为空心图形最小需要 3x3。

e) 视觉可观察性
- 解题者能从输入直接看到“左边是空心图形”; 从输出直接看到“右边复制后洞消失, 变成实心矩形”。
- 左上对齐也可从边缘标记读出: 由边标记投影出的交点正好落在实心矩形左上角, 而不是落在矩形中心。
- 生成器拒绝任何 `combo_output` 与原题、仅 D4、仅 D12 的输出一致的样本, 防止视觉上退化。

f) 可视化整洁度
- 右框内仍只有矩形复制结果, 不再出现“框内标记 + 复杂局部遮挡 + 原地保留 mover”的混杂局面。
- 行/列不复用且留出跨度间距, 降低不同颜色实心矩形互相压住的概率, 保持版面整洁。

g) 变化可推断性
- 不引入新的颜色映射; 颜色仍只表达“左侧哪个 sprite 对应框边哪组标记”。
- D2 改变的是输入形状属性, D4/D12 改变的是输出几何行为, 三者的线索分别落在不同视觉层面, 不会互相遮蔽。

h) main() 的新隐式假设与 generator 保证
- `main()` 新增假设 1: mover 的 bbox 高宽至少能作为左上锚点矩形使用。生成器保证所有 mover bbox 都在 3..5 且目标合法。
- `main()` 新增假设 2: mover 最好是空心, 否则实心化不明显。生成器保证所有 mover 都含至少一个封闭洞。
- `main()` 新增假设 3: 样例需要同时体现 D4 与 D12。生成器用对照求解器检查 `combo != center+fill` 且 `combo != topleft+sparse`。
"""

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """Return bbox of the largest connected component in `mask` that does not touch the grid border."""
    h, w = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))
    return best_bbox


def _extract_scene(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, tuple[int, int, int, int], list[dict], dict[int, list[tuple[int, int]]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((h, w), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(w)[None, :] < int(fc0)

    sprites: list[dict] = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0): int(r1) + 1, int(c0): int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                }
            )

    move_colors = {int(s["color"]) for s in sprites}
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

    return g, bg, (ir0, ic0, ir1, ic1), sprites, targets


def _solve(grid: np.ndarray, *, anchor_mode: str, fill_bbox: bool) -> np.ndarray:
    g, bg, inner_bbox, sprites, targets = _extract_scene(grid)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)

    if anchor_mode not in {"center", "topleft"}:
        raise ValueError(f"unknown anchor_mode: {anchor_mode}")

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        h, w = s["mask"].shape
        if anchor_mode == "center":
            anchor_rr = (int(h) - 1) // 2
            anchor_cc = (int(w) - 1) // 2
        else:
            anchor_rr = 0
            anchor_cc = 0

        for tr, tc in targets[int(col)]:
            base_r = int(tr - anchor_rr)
            base_c = int(tc - anchor_cc)
            if fill_bbox:
                rr0 = max(int(ir0), int(base_r))
                rr1 = min(int(ir1), int(base_r + int(h) - 1))
                cc0 = max(int(ic0), int(base_c))
                cc1 = min(int(ic1), int(base_c + int(w) - 1))
                if rr0 <= rr1 and cc0 <= cc1:
                    out[int(rr0): int(rr1) + 1, int(cc0): int(cc1) + 1] = int(col)
            else:
                for rr, cc in np.argwhere(s["mask"]):
                    pr = int(base_r + int(rr))
                    pc = int(base_c + int(cc))
                    if int(ir0) <= pr <= int(ir1) and int(ic0) <= pc <= int(ic1):
                        out[pr, pc] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - read mover targets from mirrored frame-edge markers
    - erase each moved hollow sprite from the left strip
    - at every target, use the target as the sprite bbox top-left anchor
    - paint the translated bbox as a solid rectangle
    - keep bottom-to-top overwrite order
    """
    return _solve(grid, anchor_mode="topleft", fill_bbox=True)


def _enclosed_hole_pixel_count(mask: np.ndarray) -> int:
    """Count background pixels in enclosed 4-connected components inside a sprite bbox."""
    bg_mask = ~mask.astype(bool)
    lab, n = label_mask(bg_mask, connectivity=4)
    total = 0
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(mask.shape[0] - 1)).any() or (cs == 0).any() or (cs == int(mask.shape[1] - 1)).any()):
            continue
        total += int(coords.shape[0])
    return total


def _is_valid_hollow_mask(mask: np.ndarray) -> bool:
    """A hollow sprite must be a single 8-connected foreground component with at least one enclosed hole pixel."""
    lab, n = label_mask(mask.astype(bool), connectivity=8)
    if int(n) != 1:
        return False
    return int(_enclosed_hole_pixel_count(mask)) >= 1


def _pick_positions_with_clearance(
    candidates: list[int],
    n: int,
    span: int,
    occupied: list[tuple[int, int]],
) -> list[int] | None:
    """Pick `n` anchor positions while keeping bbox-sized clearance from existing anchors."""
    if not candidates:
        return None
    order = candidates[:]
    random.shuffle(order)
    picked: list[int] = []
    local_occupied = occupied[:]
    for v in order:
        if all(abs(int(v) - int(prev)) >= max(int(span), int(prev_span)) for prev, prev_span in local_occupied):
            picked.append(int(v))
            local_occupied.append((int(v), int(span)))
            if len(picked) == int(n):
                return sorted(picked)
    return None


def _make_hollow_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a small hollow 8-connected sprite cropped to its bbox."""
    if max_h < 3 or max_w < 3:
        return None
    for _ in range(120):
        sh = min(random.choice([3, 3, 4, 4, 5]), int(max_h))
        sw = min(random.choice([3, 3, 4, 4, 5]), int(max_w))
        if sh < 3 or sw < 3:
            continue

        mask = np.ones((int(sh), int(sw)), dtype=bool)
        interior = [(r, c) for r in range(1, int(sh) - 1) for c in range(1, int(sw) - 1)]
        min_holes = 1 if len(interior) == 1 else 2
        hole_count = random.randint(min_holes, len(interior))
        for r, c in random.sample(interior, hole_count):
            mask[int(r), int(c)] = False

        corner_budget = random.randint(0, min(2, len(interior)))
        corners = [(0, 0), (0, int(sw) - 1), (int(sh) - 1, 0), (int(sh) - 1, int(sw) - 1)]
        random.shuffle(corners)
        for r, c in corners[:corner_budget]:
            trial = mask.copy()
            trial[int(r), int(c)] = False
            if _is_valid_hollow_mask(trial):
                mask = trial

        if not _is_valid_hollow_mask(mask):
            continue

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = mask[int(r0): int(r1) + 1, int(c0): int(c1) + 1]
        if cropped.shape[0] < 3 or cropped.shape[1] < 3:
            continue
        if int(_enclosed_hole_pixel_count(cropped)) < (1 if cropped.size <= 9 else 2):
            continue

        sprite = np.zeros(cropped.shape, dtype=int)
        sprite[cropped] = int(color)
        return sprite
    return None


def _build_item_sprites(
    colors: list[int],
    *,
    total_rows: int,
    max_w: int,
) -> list[np.ndarray] | None:
    """Build sprites sequentially while reserving the minimum remaining height for later items."""
    sprites: list[np.ndarray] = []
    used_rows = 0
    total_items = len(colors)
    for idx, color in enumerate(colors):
        remaining_items = int(total_items - idx - 1)
        max_h = min(5, int(total_rows - used_rows - 3 * remaining_items))
        if max_h < 3:
            return None
        sprite = _make_hollow_sprite(int(color), max_h, int(max_w))
        if sprite is None:
            return None
        sprites.append(sprite)
        used_rows += int(sprite.shape[0])
    return sprites


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    _, _, _, sprites, targets = _extract_scene(inp)
    movers = [s for s in sprites if int(s["color"]) in targets]
    combo = _solve(inp, anchor_mode="topleft", fill_bbox=True)
    center_sparse = _solve(inp, anchor_mode="center", fill_bbox=False)
    center_fill = _solve(inp, anchor_mode="center", fill_bbox=True)
    topleft_sparse = _solve(inp, anchor_mode="topleft", fill_bbox=False)
    return {
        "movers": int(len(movers)),
        "hollow_movers": int(sum(1 for s in movers if _enclosed_hole_pixel_count(s["mask"]) >= 1)),
        "diff_vs_source": int(np.count_nonzero(combo != center_sparse)),
        "diff_vs_d4": int(np.count_nonzero(combo != center_fill)),
        "diff_vs_d12": int(np.count_nonzero(combo != topleft_sparse)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    h = interior_h + 2
    if not 10 <= h <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    w = sprite_area_w + 1 + frame_w
    if not 14 <= w <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, w - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = h - 2, fc1 - 1

    max_items = h // 3
    if max_items < 2:
        return None

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, min(3, max_items))
    can_have_distractor = max_items > n_movers
    n_distractors = 1 if can_have_distractor and random.choice([0, 0, 1]) else 0

    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx: idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx: idx + n_distractors]

    ordered_colors = mover_colors + distractor_colors
    all_sprites = _build_item_sprites(ordered_colors, total_rows=h, max_w=sprite_area_w)
    if all_sprites is None:
        return None

    mover_sprites = all_sprites[:n_movers]
    distractor_sprites = all_sprites[n_movers:]

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: list[tuple[int, int]] = []
    occupied_cols: list[tuple[int, int]] = []
    for mc, sp in zip(mover_colors, mover_sprites):
        sh, sw = sp.shape
        min_row = ir0
        max_row = ir1 - int(sh) + 1
        min_col = ic0
        max_col = ic1 - int(sw) + 1
        if min_row > max_row or min_col > max_col:
            return None
        row_count = random.randint(1, 2)
        col_count = random.randint(1, 2)
        row_marks = _pick_positions_with_clearance(list(range(min_row, max_row + 1)), row_count, int(sh), occupied_rows)
        col_marks = _pick_positions_with_clearance(list(range(min_col, max_col + 1)), col_count, int(sw), occupied_cols)
        if row_marks is None or col_marks is None:
            return None
        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.extend((int(r), int(sh)) for r in row_marks)
        occupied_cols.extend((int(c), int(sw)) for c in col_marks)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, int(fc0): int(fc0) + frame_w] = top_edge
    inp[int(h - 1), int(fc0): int(fc0) + frame_w] = top_edge

    for r in range(1, int(h) - 1):
        val = int(frame_base)
        for mc, (rmarks, _) in mover_targets.items():
            if int(r) in rmarks:
                val = int(mc)
                break
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    cur_row = 0
    for sprite in mover_sprites + distractor_sprites:
        sh, sw = sprite.shape
        if cur_row + int(sh) > int(h):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(int(sh)):
            for dc in range(int(sw)):
                if int(sprite[dr, dc]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sprite[dr, dc])
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        combo = main(inp)
        metrics = _combo_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, combo):
        return None
    if metrics["movers"] != int(n_movers):
        return None
    if metrics["hollow_movers"] != metrics["movers"]:
        return None
    if metrics["diff_vs_source"] < 8:
        return None
    if metrics["diff_vs_d4"] < 4:
        return None
    if metrics["diff_vs_d12"] < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 600 attempts")
