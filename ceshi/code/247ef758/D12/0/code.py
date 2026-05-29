# Variant of 247ef758
# Dimension: D12 — 精灵印刻对齐方式 (meta)
# Change: 精灵以自身中心点（外接矩形中心）对齐到目标位置... → 精灵以左上角对齐到目标位置
#
# Design analysis
# a) 具体化与行为追踪
# - 输入仍然由三块组成：左侧精灵条带、整列分隔线、右侧矩形框。框的左右边和上下边继续用同色标记给出
#   “目标行”和“目标列”，笛卡尔积交点仍然是复制目标。
# - 原题在每个目标点 `(tr, tc)` 处，把精灵外接框中心对齐到该点；本变体改为把精灵外接框左上角直接放到
#   `(tr, tc)`。因此同一个 3x4 精灵若目标点是 `(5, 9)`，原题会把它放到 `rows 4..6, cols 8..11`，
#   本变体会放到 `rows 5..7, cols 9..12`。
# - 差异发生在“目标坐标 -> 印刻基址”的一步：旧逻辑是减去中心偏移，本变体不减偏移。
#   其余流程：找框、找分隔列、提取左侧精灵、按颜色读取标记、擦除原精灵、按自下而上覆盖，全部保持不变。
#
# b) 复杂度校准
# - 只改锚点，不改颜色语义和匹配方式，避免引入新的配对歧义。
# - 仍保留 2-3 个 mover 和 0-1 个 distractor，且每个 mover 有 1-2 个行标记和 1-2 个列标记，保证样例中
#   会重复出现“同一精灵多次复制”的证据。
# - 为了让 D12 可见，生成器拒绝“新锚点输出 == 旧中心锚点输出”的样本，避免退化。
#
# c) 隐式假设审查
# - 原 `main()` 中 `cr/cc = bbox center` 与 `dr/dc = target - center` 明确依赖“中心对齐”前提；
#   在本变体里该前提失效，必须改成“目标点就是新左上角”。
# - 原生成器用 `min_row = ir0 + sh//2` 等约束，保证中心对齐后不会越界；改成左上角对齐后，合法范围必须变成
#   `ir0 <= tr <= ir1-sh+1` 和 `ic0 <= tc <= ic1-sw+1`。若不改，目标会被无意义地限制在偏内侧，且不能体现
#   新锚点允许贴近框左上边缘的特征。
# - 擦除原精灵、排序覆盖、颜色匹配、clip 到 interior 的代码不依赖旧锚点，保持有效。
#
# d) 放置合法性
# - 生成器按左上角锚点挑选标记，因此每次印刻都能完整落在框内，不依赖裁剪来“救场”。
# - 同一 mover 的多个行标记/列标记仍至少相隔自身高/宽，避免同色副本彼此重叠得过于混乱。
# - 不同 mover 不共享任何行标记或列标记，维持原题的颜色到目标集合的一一对应线索。
#
# e) 视觉可观察性
# - 人类可直接从“标记交点在精灵的左上角，而不是中心”推断新规则。即使精灵是单色，只要外接框尺寸大于 1，
#   这个偏移就是可见的。
# - 生成器要求至少有一个 mover 的外接框在两个维度上都大于 1，并且拒绝旧新输出相同的样本，保证训练样例里
#   能看到实际偏移。
#
# f) 可视化整洁度
# - 仍沿用原题的整洁布局：左条带只有精灵，右框只有标记与复制结果，中间有统一分隔线。
# - 覆盖顺序仍按“输入中更低的精灵先盖，更高的精灵后盖”，因此即便不同颜色副本重叠，解释方式与原题一致。
#
# g) 颜色变化可推断性
# - 本变体不新增颜色映射；颜色继续只承担“哪种精灵对应哪组标记”的作用。
#
# h) main() 的新隐式假设与 generator 对应保证
# - `main()` 现在假设目标点可以作为精灵左上角；`generate_input()` 通过新的合法范围保证这一点。
# - `main()` 仍假设框内最大封闭背景区就是 interior，分隔列是左侧近乎整列的统一非背景色，左条带精灵按颜色是
#   8 连通单色块；这些假设全部沿用原生成策略继续满足。

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _solve_with_anchor(grid: np.ndarray, *, anchor_mode: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

    sprites = []
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
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

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

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    if anchor_mode not in {"center", "topleft"}:
        raise ValueError(f"unknown anchor_mode: {anchor_mode}")

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
            for rr, cc in np.argwhere(s["mask"]):
                pr = int(base_r + int(rr))
                pc = int(base_c + int(cc))
                if int(ir0) <= pr <= int(ir1) and int(ic0) <= pc <= int(ic1):
                    out[pr, pc] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Variant rule:
    - For each color, target rows/cols are where both opposite frame edges share that color.
    - Erase each sprite from the left and stamp it with its bbox top-left corner anchored at every target.
    - Higher sprites cover lower ones.
    """
    return _solve_with_anchor(grid, anchor_mode="topleft")


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    sh = min(random.choice([3, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            pixels.add((nr, nc))
            r, c = nr, nc
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for pr, pc in pixels:
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    if not any(sp.shape[0] > 1 and sp.shape[1] > 1 for sp in sprites):
        return None

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row = ir0
        max_row = ir1 - int(sh) + 1
        min_col = ic0
        max_col = ic1 - int(sw) + 1
        if min_row > max_row or min_col > max_col:
            return None

        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), int(sh))
        if not row_marks:
            return None

        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), int(sw))
        if not col_marks:
            return None

        mover_targets[mc] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - fc0] = mc
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for _color, sp, _is_distractor in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
        center_output = _solve_with_anchor(inp, anchor_mode="center")
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    if np.array_equal(output, center_output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
