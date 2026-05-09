# Combo variant of 247ef758 (index 4)
# combo_spec: D2/0+D4/0
# Name: 多色精灵重着色印刻
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题：左侧每个 mover 是单色 blob。solver 按颜色逐色提取组件，用该颜色去匹配框架四边的同色标记，
#      再把原图案原色印到 row_marks x col_marks 的交点，底部对象先印、顶部对象后盖。
#    - D2/0 改动：左侧每个 sprite 变成双色 8 连通组件，主色是组件内像素数严格更多的颜色。匹配时只看主色。
#      如果仍按“每种颜色各自做连通域”提取，会把一个双色 sprite 错拆成两个组件，因此必须改成：
#      在左侧非背景并排除 divider 的并集上做一次 8 连通标记，再对每个组件统计主色。
#    - D4/0 改动：目标位置仍由主色标记决定，但印刻时不保留原双色图案的颜色；组件内所有前景像素统一改写成
#      frame_base。这样“识别时看双色组成，匹配时看主色，输出时所有颜色都消失”三步必须同时成立。
#    - 小例子：若一个 sprite 有 5 个红像素 + 2 个黄像素，则主色为红。原题会把红色单色块印到红色标记交点；
#      D2-only 会把红黄双色图案印到红色标记交点；本融合变体会把这 7 个像素对应的形状全部改成 frame_base。
#      因此黄像素在输出里不再保留，但它仍参与“这是一个红色主导的 mover”的识别。
# b) 复杂度校准
#    - 所有 sprite 恰好双色，辅色全局共享，避免颜色组合过多导致样例难归纳。
#    - mover 的主色互不相同，且 frame/divider/accent 全部不同，避免匹配歧义。
#    - 仍保留 2-3 个 mover，使“多个不同原色 -> 同一 frame_base 输出色”的重着色规律有重复支撑。
# c) 隐式假设审查
#    - 原 solver 的“按颜色取左侧组件”假设失效，改为按左侧非背景并集提取组件。
#    - 原 solver 的“sprite color == 匹配色 == 输出色”假设失效，拆成：
#      component colors -> dominant main_color -> targets -> recolor to frame_base。
#    - 新增 frame_base 识别步骤，因此 generator 必须保证框架边界上 frame_base 是最常见非背景色。
#    - 原生成器可以把左侧 sprite 竖直紧贴堆放；现在若紧贴会把两个双色组件连成一个 8 连通块，因此必须保证相邻
#      sprite 至少隔 1 行背景。
#    - 原生成器只需避免同一 mover 的多次印刻越界；现在所有 mover 输出都会同色，若不同 mover 的目标 bbox 接触，
#      视觉上会融合成一团，因此还需保证所有目标 bbox 之间至少留 1 格。
# d) 放置合法性
#    - target picker 仅从能让 sprite 完整落入 interior 的中心点中选行列标记。
#    - 不同 mover 不共享 row mark / col mark；并额外检查所有复制体 bbox 两两不接触。
#    - 左侧放置时按最小总高度 + 分配额外空行，保证不越界且组件间至少一行背景。
# e) 视觉可观察性
#    - 输入中框架标记只显示主色，不显示辅色；输出中复制体全变 frame_base。人类可从样例看到：
#      “双色信息用于识别与匹配，但最终输出统一改色”。
#    - 因为辅色在输入可见、在输出消失，且多个 mover 重复出现，这个融合规则能从 2-4 个样例归纳。
# f) 可视化整洁度
#    - 共享辅色减少调色板噪声；frame_base 统一输出色避免内部多色混杂。
#    - 目标复制体不接触，divider/frame/movers/distractors 颜色角色分离，视觉区域清晰。
# g) 变化的可推断性
#    - 颜色变化不是任意映射，而是唯一全局规则：所有 mover 的所有前景像素都改成 frame_base。
#      frame_base 直接由包围 interior 的主边框提供线索，不依赖颜色编号大小等不可视因素。
# h) 新 solver 假设与 generator 对应保证
#    - 每个左侧 sprite 是一个孤立的 8 连通双色组件：由左侧放置器保证。
#    - 每个组件存在唯一严格多数主色：由双色 sprite 生成器保证。
#    - frame_base 是边框最常见非背景色：由边框构造和校验保证。
#    - mover 的主色在四边同时拥有行标记和列标记：由目标构造保证。

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the grid border."""
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


def _find_frame_bounds(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int, int, int, int]:
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    return int(ir0), int(ic0), int(ir1), int(ic1), int(fr0), int(fc0), int(fr1), int(fc1)


def _frame_base_color(g: np.ndarray, bg: int, fr0: int, fc0: int, fr1: int, fc1: int) -> int:
    border_vals: list[int] = []
    border_vals.extend(map(int, g[int(fr0), int(fc0) : int(fc1) + 1]))
    border_vals.extend(map(int, g[int(fr1), int(fc0) : int(fc1) + 1]))
    border_vals.extend(int(g[r, int(fc0)]) for r in range(int(fr0) + 1, int(fr1)))
    border_vals.extend(int(g[r, int(fc1)]) for r in range(int(fr0) + 1, int(fr1)))
    counts = Counter(v for v in border_vals if int(v) != int(bg))
    assert counts
    return int(counts.most_common(1)[0][0])


def _divider_mask(g: np.ndarray, bg: int, fc0: int) -> np.ndarray:
    h, w = g.shape
    mask = np.zeros((h, w), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            mask[:, int(c)] = True
    return mask


def _dominant_color(colors: list[int]) -> int:
    counts = Counter(map(int, colors))
    return sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0]


def _extract_sprites(g: np.ndarray, bg: int, fc0: int, divider_mask: np.ndarray) -> list[dict]:
    left_mask = np.arange(g.shape[1])[None, :] < int(fc0)
    sprite_mask = (g != int(bg)) & left_mask & (~divider_mask)
    slab, sn = label_mask(sprite_mask, connectivity=8)

    sprites: list[dict] = []
    for sid in range(1, int(sn) + 1):
        pts = np.argwhere(slab == int(sid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        comp = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
        patch = np.where(comp, g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], int(bg))
        colors = [int(v) for v in patch[comp]]
        if not colors:
            continue
        sprites.append(
            {
                "main_color": int(_dominant_color(colors)),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "pattern": patch.astype(int),
            }
        )
    return sprites


def _marker_targets(
    g: np.ndarray,
    bg: int,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
    move_colors: set[int],
) -> dict[int, list[tuple[int, int]]]:
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
        targets[int(col)] = sorted({(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]})
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule:
    - Left-strip sprites are two-color 8-connected components.
    - Match frame markers by each sprite's dominant color.
    - Stamp the sprite shape at every target, but recolor every stamped foreground pixel to frame_base.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1 = _find_frame_bounds(g, bg)
    frame_base = _frame_base_color(g, bg, fr0, fc0, fr1, fc1)
    divider_mask = _divider_mask(g, bg, fc0)

    sprites = _extract_sprites(g, bg, fc0, divider_mask)
    move_colors = {int(s["main_color"]) for s in sprites}
    targets = _marker_targets(g, bg, ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1, move_colors)

    out = g.copy()
    movers = [s for s in sprites if int(s["main_color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        pat = np.array(s["pattern"], dtype=int)
        for rr, cc in np.argwhere(pat != int(bg)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        key = int(s["main_color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pat = np.array(s["pattern"], dtype=int)
        pix = [(int(rr), int(cc2)) for rr, cc2 in np.argwhere(pat != int(bg))]

        for tr, tc in targets[int(key)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for rr0, cc0 in pix:
                rr = int(s["r0"]) + int(rr0) + int(dr)
                cc3 = int(s["c0"]) + int(cc0) + int(dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(frame_base)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    if not candidates:
        return None
    if n == 1:
        return [int(random.choice(candidates))]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == int(n):
                return picked
    return picked if picked else None


def _make_shape_mask(max_h: int, max_w: int) -> np.ndarray | None:
    sh = min(random.choice([2, 3, 3, 4]), max_h)
    sw = min(random.choice([2, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None

    max_pixels = min(9, sh * sw)
    if max_pixels < 3:
        return None
    n_pixels = random.randint(3, max_pixels)

    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 40):
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
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for pr, pc in pixels:
        mask[pr - min_r, pc - min_c] = True
    return mask


def _sample_connected_subset(coords: list[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    start = random.choice(coords)
    chosen = {start}
    frontier = [start]
    coord_set = set(coords)

    while len(chosen) < int(k):
        base = random.choice(frontier)
        nbrs = [
            (nr, nc)
            for nr in range(base[0] - 1, base[0] + 2)
            for nc in range(base[1] - 1, base[1] + 2)
            if (nr, nc) in coord_set and (nr, nc) not in chosen and not (nr == base[0] and nc == base[1])
        ]
        if nbrs:
            nxt = random.choice(nbrs)
            chosen.add(nxt)
            frontier.append(nxt)
            continue
        frontier.append(random.choice(list(chosen)))
        if len(frontier) > len(coords) * 4:
            break

    if len(chosen) < int(k):
        remaining = [p for p in coords if p not in chosen]
        random.shuffle(remaining)
        for p in remaining:
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in chosen):
                chosen.add(p)
                if len(chosen) >= int(k):
                    break
    return chosen


def _make_multicolor_sprite(primary: int, accent: int, max_h: int, max_w: int) -> np.ndarray | None:
    mask = _make_shape_mask(max_h, max_w)
    if mask is None:
        return None
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    area = len(coords)
    if area < 3:
        return None

    max_accent = min(3, (area - 1) // 2)
    if max_accent < 1:
        return None
    min_accent = 2 if area >= 5 and max_accent >= 2 else 1
    n_accent = random.randint(min_accent, max_accent)
    accent_cells = _sample_connected_subset(coords, n_accent)

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        sprite[int(r), int(c)] = int(accent if (r, c) in accent_cells else primary)
    return sprite


def _stamp_bbox(sp: np.ndarray, tr: int, tc: int) -> tuple[int, int, int, int]:
    sh, sw = sp.shape
    r0 = int(tr - sh // 2)
    c0 = int(tc - sw // 2)
    return int(r0), int(c0), int(r0 + sh - 1), int(c0 + sw - 1)


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _pick_targets(
    sp: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    occupied_rows: set[int],
    occupied_cols: set[int],
    existing_bboxes: list[tuple[int, int, int, int]],
) -> tuple[list[int], list[int], list[tuple[int, int, int, int]]] | None:
    sh, sw = sp.shape
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    if min_row > max_row or min_col > max_col:
        return None

    avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
    avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
    if not avail_rows or not avail_cols:
        return None

    for _ in range(100):
        want_rows = random.randint(1, 2)
        want_cols = random.randint(1, 2)
        row_marks = _pick_spaced(avail_rows[:], want_rows, sh)
        col_marks = _pick_spaced(avail_cols[:], want_cols, sw)
        if not row_marks or not col_marks:
            continue

        boxes = [_stamp_bbox(sp, r, c) for r in row_marks for c in col_marks]
        ok = True
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _bboxes_too_close(boxes[i], boxes[j]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for box in boxes:
            if any(_bboxes_too_close(box, prev) for prev in existing_bboxes):
                ok = False
                break
        if ok:
            return sorted(row_marks), sorted(col_marks), boxes
    return None


def _sample_gaps(total_extra: int, n_slots: int) -> list[int]:
    gaps = [0] * int(n_slots)
    for _ in range(int(total_extra)):
        gaps[random.randrange(int(n_slots))] += 1
    return gaps


def _frame_base_majority(inp: np.ndarray) -> bool:
    bg = int(get_bg(inp))
    ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1 = _find_frame_bounds(inp, bg)
    _ = (ir0, ic0, ir1, ic1)
    border_vals: list[int] = []
    border_vals.extend(map(int, inp[int(fr0), int(fc0) : int(fc1) + 1]))
    border_vals.extend(map(int, inp[int(fr1), int(fc0) : int(fc1) + 1]))
    border_vals.extend(int(inp[r, int(fc0)]) for r in range(int(fr0) + 1, int(fr1)))
    border_vals.extend(int(inp[r, int(fc1)]) for r in range(int(fr0) + 1, int(fr1)))
    counts = Counter(v for v in border_vals if int(v) != int(bg))
    if not counts:
        return False
    top = counts.most_common(2)
    if len(top) == 1:
        return True
    return int(top[0][1]) > int(top[1][1])


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    """
    Quantify the fused behavior.
    - moved_accent_pixels: how many stamped pixels come from source accent-colored cells.
    - recolored_accent_pixels: how many of those positions became frame_base in output.
    - moved_primary_pixels: same for primary-colored source cells.
    - recolored_primary_pixels: same for primary source cells.
    - non_framebase_new_pixels: any new interior pixels not equal to frame_base.
    """
    g = np.array(inp, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1 = _find_frame_bounds(g, bg)
    frame_base = _frame_base_color(g, bg, fr0, fc0, fr1, fc1)
    divider_mask = _divider_mask(g, bg, fc0)
    sprites = _extract_sprites(g, bg, fc0, divider_mask)
    targets = _marker_targets(
        g,
        bg,
        ir0,
        ic0,
        ir1,
        ic1,
        fr0,
        fc0,
        fr1,
        fc1,
        {int(s["main_color"]) for s in sprites},
    )

    moved_accent_pixels = 0
    recolored_accent_pixels = 0
    moved_primary_pixels = 0
    recolored_primary_pixels = 0
    stamped_positions: set[tuple[int, int]] = set()

    for s in sprites:
        key = int(s["main_color"])
        if key not in targets:
            continue
        pat = np.array(s["pattern"], dtype=int)
        counts = Counter(int(v) for v in pat[pat != int(bg)])
        accent_candidates = [col for col, cnt in counts.items() if int(col) != key and int(cnt) > 0]
        accent = int(accent_candidates[0]) if accent_candidates else None
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2

        for tr, tc in targets[int(key)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for rr0, cc0 in np.argwhere(pat != int(bg)):
                rr = int(s["r0"]) + int(rr0) + int(dr)
                cc3 = int(s["c0"]) + int(cc0) + int(dc)
                if not (int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1)):
                    continue
                stamped_positions.add((int(rr), int(cc3)))
                src_color = int(pat[int(rr0), int(cc0)])
                if src_color == int(key):
                    moved_primary_pixels += 1
                    if int(out[int(rr), int(cc3)]) == int(frame_base):
                        recolored_primary_pixels += 1
                elif accent is not None and src_color == int(accent):
                    moved_accent_pixels += 1
                    if int(out[int(rr), int(cc3)]) == int(frame_base):
                        recolored_accent_pixels += 1

    non_framebase_new_pixels = 0
    for r in range(int(ir0), int(ir1) + 1):
        for c in range(int(ic0), int(ic1) + 1):
            if int(out[r, c]) != int(g[r, c]) and int(out[r, c]) != int(frame_base):
                non_framebase_new_pixels += 1

    return {
        "moved_accent_pixels": int(moved_accent_pixels),
        "recolored_accent_pixels": int(recolored_accent_pixels),
        "moved_primary_pixels": int(moved_primary_pixels),
        "recolored_primary_pixels": int(recolored_primary_pixels),
        "non_framebase_new_pixels": int(non_framebase_new_pixels),
        "stamped_pixels": int(len(stamped_positions)),
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

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    n_total = n_movers + n_distractors

    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    accent_color = int(all_colors[idx])
    idx += 1
    primary_colors = [int(v) for v in all_colors[idx : idx + n_total]]
    mover_colors = primary_colors[:n_movers]
    distractor_colors = primary_colors[n_movers:]

    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_multicolor_sprite(int(mc), int(accent_color), 4, sprite_area_w)
        if sp is None:
            return None
        mover_sprites.append(sp)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    occupied_target_boxes: list[tuple[int, int, int, int]] = []
    for mc, sp in zip(mover_colors, mover_sprites):
        picked = _pick_targets(sp, ir0, ic0, ir1, ic1, occupied_rows, occupied_cols, occupied_target_boxes)
        if picked is None:
            return None
        row_marks, col_marks, boxes = picked
        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.update(map(int, row_marks))
        occupied_cols.update(map(int, col_marks))
        occupied_target_boxes.extend(boxes)

    all_items: list[tuple[int, np.ndarray, bool]] = [(int(mc), sp, False) for mc, sp in zip(mover_colors, mover_sprites)]
    for dc in distractor_colors:
        ds = _make_multicolor_sprite(int(dc), int(accent_color), 4, sprite_area_w)
        if ds is None:
            return None
        all_items.append((int(dc), ds, True))

    min_needed_h = sum(int(sp.shape[0]) for _, sp, _ in all_items) + max(0, len(all_items) - 1)
    if min_needed_h > int(h):
        return None

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - int(fc0)] = int(mc)
    inp[0, int(fc0) : int(fc0 + frame_w)] = top_edge
    inp[int(h - 1), int(fc0) : int(fc0 + frame_w)] = top_edge

    for r in range(1, int(h - 1)):
        val = int(frame_base)
        for mc, (rmarks, _) in mover_targets.items():
            if int(r) in rmarks:
                val = int(mc)
                break
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    extra_rows = int(h) - int(min_needed_h)
    gap_extras = _sample_gaps(extra_rows, len(all_items) + 1)
    cur_row = int(gap_extras[0])
    for idx_item, (_primary, sp, _is_distractor) in enumerate(all_items):
        sh, sw = sp.shape
        if int(sw) > int(sprite_area_w) or int(cur_row + sh) > int(h):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        inp[cur_row : cur_row + sh, c_off : c_off + sw] = np.where(
            sp != int(bg),
            sp,
            inp[cur_row : cur_row + sh, c_off : c_off + sw],
        )
        cur_row += int(sh)
        if idx_item < len(all_items) - 1:
            cur_row += 1 + int(gap_extras[idx_item + 1])

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None
    if not _frame_base_majority(inp):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    metrics = _fusion_metrics(inp, out)
    if metrics["moved_accent_pixels"] <= 0:
        return None
    if metrics["recolored_accent_pixels"] != metrics["moved_accent_pixels"]:
        return None
    if metrics["moved_primary_pixels"] <= 0:
        return None
    if metrics["recolored_primary_pixels"] != metrics["moved_primary_pixels"]:
        return None
    if metrics["non_framebase_new_pixels"] != 0:
        return None
    if metrics["stamped_pixels"] <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
