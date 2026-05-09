# Combo variant of 800d221b (index 3)
# combo_spec: D1/3+D2/0
# Name: 颜色交换+hub取最少色

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import components_mask, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 视觉骨架保持原题：一个 3x3 hub、若干条 1 像素宽路径、以及贴着网格边的矩形种子区。
#      每个种子内部仍然只使用两种颜色并形成多数/少数关系，种子外框与路径在输入里同为 P。
#    - 小例子：某个 3x2 种子内部输入为
#          A A A
#          A B A
#      原题输出会保持这 6 个内部像素不变，同时把连到它的路径和边框改成 A。
#      本融合版输出会同时做两件事：
#      1. 路径和边框仍改成多数色 A；
#      2. 种子内部做双向交换，变成
#             B B B
#             B A B
#      所以解题者既能从路径读出“多数色”，又能从种子内部读出“多数/少数互换”。
#    - hub 方面，若与 hub 相邻的重着色分支颜色计数为 {A: 5, B: 3}，原题中心取 A，
#      本融合版改为取 B。也就是分支仍提供计数线索，但最后要取最少而不是最多。
# b) 复杂度校准
#    - 颜色词汇仍限制为 4 色：bg、路径色 P、以及两种种子色 A/B。这样“交换”是可见的，
#      但不会引入过多颜色映射自由度。
#    - 生成器强制所有样例都至少出现一个 A 多数种子和一个 B 多数种子，避免全体同向交换，
#      使“多数/少数互换”在 2-4 个样例中可以稳定归纳。
#    - hub 中心的新规则若遇到平票会变得含糊，因此生成器只接受 hub 相邻颜色计数严格不等、
#      且至少两种颜色都出现在 hub 周围的样例。
# c) 隐式假设审查
#    - 原 main() 只需要知道种子多数色；融合后还必须唯一确定少数色，因此现在显式要求每个
#      种子区域恰有两种颜色且多数严格大于少数。生成器继续保证这一点。
#    - 分支组件的识别依赖“输入里路径/边框仍统一为 P”，颜色交换只发生在输出端，因此不会
#      破坏 P 网络识别、hub 检测或 branch 连通性假设。
#    - hub 取最少色会把原来对 Counter.most_common() 的默认 tie 行为变成风险点，所以这里
#      既在 solver 中给出确定性最少色选择，又在 generator 中规避平票样例。
#    - 原分支匹配按与哪个种子接触最多来决定所属种子，这个前提在融合后依然成立，因为种子
#      位置、边框和路径拓扑都没有改变。
# d) 放置合法性
#    - 种子仍按贴边矩形放置，边框只出现在非贴边侧，且各 seed zone 之间保留间隔，避免互相
#      重叠或与 hub 冲突。
#    - 路径仍由正交折线雕刻，禁止与已有路径、种子或 hub 形成 T 字分叉；每条路径占用不同的
#      hub 边界起点，因此 hub 相邻计数与“分支条数”一一对应。
# e) 视觉可观察性
#    - 输入输出同时存在两条线索：种子内部被交换、外部路径/边框被重着色。人类可以从同一 seed
#      的“内部变成对方颜色、外框却变成原多数色”反推出双向交换规则。
#    - 生成器拒绝 hub 中心与原题一致的退化样例，确保 D2/0 的“取最少”在样本中真实生效。
# f) 可视化整洁度
#    - 仍使用原题的规则化矩形 seed 与单像素路径，不额外叠加标记。交换仅发生在 seed 内部，
#      不会和路径层覆盖混杂；hub 也只改中心一个像素。
# g) 变化的可推断性
#    - 颜色变化只围绕 A/B 两色展开，且两色同时承担“种子内部互换”和“路径取多数色”两个角色。
#      结构提示足够强，不依赖颜色编号大小或任何人类不可见信息。
# 9) 新 solver 假设与 generator 保证
#    - solver 假设每个 seed interior 是单个 4 连通布尔区域、内部恰两色且存在严格多数；
#      generator 用整块矩形填充并控制 minority 数量来保证。
#    - solver 假设 hub 周围至少有一条分支且最少色唯一；generator 要求至少两种 hub 相邻色，
#      并过滤掉“最少=最多”或平票样例。
#    - solver 假设最大实心 P 方块就是 hub；generator 继续固定为 3x3 实心 hub，并避免在其他
#      位置制造更大的实心 P 方块。


# ── Tunable generation parameters ──────────────────────────────────────────

GRID_SIZE_BASE = 10
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3, 4, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 400
SEED_ZONE_GAP = 1

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
MAX_PATHS_PER_SEED = 3
PATH_FIRST_ROUND_RETRIES = 60
PATH_EXTRA_ROUND_RETRIES = 30
PATH_DIR_RETRIES = 3

GEN_OUTER_RETRIES = 300

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    h, w = g.shape
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors

    def comp_deg_score(mask: np.ndarray) -> float:
        comps = components_mask(mask, connectivity=4)
        if not comps:
            return -1e9
        best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
        best_sz = int(best_pts.shape[0])
        m = np.zeros(mask.shape, dtype=bool)
        m[best_pts[:, 0], best_pts[:, 1]] = True

        deg2 = 0
        deg4 = 0
        for r, c in best_pts:
            d = 0
            if r > 0 and bool(m[int(r - 1), int(c)]):
                d += 1
            if r + 1 < h and bool(m[int(r + 1), int(c)]):
                d += 1
            if c > 0 and bool(m[int(r), int(c - 1)]):
                d += 1
            if c + 1 < w and bool(m[int(r), int(c + 1)]):
                d += 1
            if d == 2:
                deg2 += 1
            elif d == 4:
                deg4 += 1
        frac2 = deg2 / max(1, best_sz)
        frac4 = deg4 / max(1, best_sz)
        return float((frac2 - frac4) * best_sz)

    return int(max(colors, key=lambda c: comp_deg_score(g == int(c))))


def _find_hub(g: np.ndarray, path_color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = g.shape
    m_path = g == int(path_color)
    ps = m_path.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[int(r1), int(c1)])
        if r0 > 0:
            s -= int(ps[int(r0 - 1), int(c1)])
        if c0 > 0:
            s -= int(ps[int(r1), int(c0 - 1)])
        if r0 > 0 and c0 > 0:
            s += int(ps[int(r0 - 1), int(c0 - 1)])
        return int(s)

    best = None
    for k in range(2, int(min(h, w)) + 1):
        found = None
        for r0 in range(0, int(h - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(w - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found

    assert best is not None
    k, r0, c0 = map(int, best)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)
    hub = np.zeros((h, w), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    return hub, (int(r0), int(c0), int(r1), int(c1))


def _collect_seed_info(
    g: np.ndarray, bg: int, path_color: int
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    h, w = g.shape
    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_info: dict[int, dict[str, object]] = {}

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())
        if rn > 0:
            assert all(int(g[int(rn - 1), c]) == int(path_color) for c in range(cn, cx + 1))
        if rx < h - 1:
            assert all(int(g[int(rx + 1), c]) == int(path_color) for c in range(cn, cx + 1))
        if cn > 0:
            assert all(int(g[r, int(cn - 1)]) == int(path_color) for r in range(rn, rx + 1))
        if cx < w - 1:
            assert all(int(g[r, int(cx + 1)]) == int(path_color) for r in range(rn, rx + 1))

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals).most_common()
        assert len(counts) == 2, "swap variant expects exactly two seed colors"
        (major, major_n), (minor, minor_n) = counts[0], counts[1]
        assert int(major_n) > int(minor_n), "seed majority must be unique"
        seed_info[int(sid)] = {
            "pts": pts,
            "major": int(major),
            "minor": int(minor),
            "bbox": (int(rn), int(cn), int(rx), int(cx)),
        }

    assert seed_info, "grid must contain seed regions"
    return seed_lab, seed_info


def _label_branches(g: np.ndarray, path_color: int, hub: np.ndarray, seed_lab: np.ndarray) -> tuple[np.ndarray, int]:
    branch_mask = (g == int(path_color)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    path_mask = g == int(path_color)
    ppad = np.pad(path_mask, 1)
    p_deg = (
        ppad[2:, 1:-1].astype(int)
        + ppad[:-2, 1:-1]
        + ppad[1:-1, 2:]
        + ppad[1:-1, :-2]
    )
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & path_mask & ~hub & ~adj_seed), "path has T-junction"
    return branch_lab, int(branch_n)


def _pick_center_color(touch_cols: Counter[int], mode: str) -> int:
    assert touch_cols
    if mode == "most":
        return int(touch_cols.most_common(1)[0][0])
    assert mode == "least"
    least = min(int(v) for v in touch_cols.values())
    for col, cnt in touch_cols.items():
        if int(cnt) == int(least):
            return int(col)
    raise AssertionError("unreachable")


def _solve_core(
    grid: np.ndarray, *, swap_seed_colors: bool, hub_mode: str, return_meta: bool = False
):
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    h, w = g.shape

    path_color = _infer_path_color(g, bg)
    hub, (r0, c0, r1, c1) = _find_hub(g, path_color)
    seed_lab, seed_info = _collect_seed_info(g, bg, path_color)
    branch_lab, branch_n = _label_branches(g, path_color, hub, seed_lab)

    if swap_seed_colors:
        for info in seed_info.values():
            pts = info["pts"]
            major = int(info["major"])
            minor = int(info["minor"])
            for r, c in pts:
                rv = int(r)
                cv = int(c)
                out[rv, cv] = int(minor if int(g[rv, cv]) == int(major) else major)

    branch_to_seed: dict[int, int] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches: Counter[int] = Counter()
        for r, c in pts:
            rv = int(r)
            cv = int(c)
            for dr, dc in _DIRS:
                rr, cc = int(rv + dr), int(cv + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        if not touches:
            continue
        sid = int(touches.most_common(1)[0][0])
        branch_to_seed[int(bid)] = int(sid)
        col = int(seed_info[int(sid)]["major"])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    touch_cols: Counter[int] = Counter()
    if int(branch_n) > 0:
        for r in range(h):
            for c in range(w):
                if not bool(hub[int(r), int(c)]):
                    continue
                for dr, dc in _DIRS:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and int(branch_lab[rr, cc]) != 0:
                        touch_cols[int(out[rr, cc])] += 1

    cr = int((r0 + r1) // 2)
    cc = int((c0 + c1) // 2)
    center_col = _pick_center_color(touch_cols, hub_mode)
    out[int(cr), int(cc)] = int(center_col)

    if not return_meta:
        return out

    meta = {
        "bg": int(bg),
        "path_color": int(path_color),
        "hub_bbox": (int(r0), int(c0), int(r1), int(c1)),
        "center": (int(cr), int(cc)),
        "seed_lab": seed_lab,
        "seed_info": seed_info,
        "branch_lab": branch_lab,
        "branch_to_seed": branch_to_seed,
        "touch_cols": dict((int(k), int(v)) for k, v in touch_cols.items()),
    }
    return out, meta


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    variant_out, variant_meta = _solve_core(inp, swap_seed_colors=True, hub_mode="least", return_meta=True)
    baseline_out, _ = _solve_core(inp, swap_seed_colors=False, hub_mode="most", return_meta=True)
    seed_mask = variant_meta["seed_lab"] > 0
    cr, cc = variant_meta["center"]
    touch_cols = Counter(variant_meta["touch_cols"])
    counts = list(touch_cols.values())
    return {
        "seed_swap_changed_cells": int(np.sum((variant_out != baseline_out) & seed_mask)),
        "center_changed_vs_original": int(int(variant_out[int(cr), int(cc)]) != int(baseline_out[int(cr), int(cc)])),
        "hub_color_count_span": int(max(counts) - min(counts)) if counts else 0,
        "hub_color_variety": int(len(touch_cols)),
        "variant_center_color": int(variant_out[int(cr), int(cc)]),
        "original_center_color": int(baseline_out[int(cr), int(cc)]),
        "seed_count": int(len(variant_meta["seed_info"])),
        "branch_count": int(len(variant_meta["branch_to_seed"])),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the path network while swapping each seed's internal majority/minority colors.

    Input structure:
    - The most common color is background.
    - One non-background color forms a thin orthogonal path network with a solid square hub.
    - Several edge-touching rectangular seed regions sit behind path-colored borders; each seed
      interior mixes exactly two colors with a strict majority and minority.

    Rule:
    - Infer the path color from the network-like component and locate the largest solid all-path square as the hub.
    - For each branch-plus-seed-border component, find the seed region it touches most and recolor that whole
      branch component to the seed's interior majority color.
    - Inside every seed region, swap the majority and minority colors in place.
    - Recolor the hub center pixel to the least frequent color among recolored branch pixels adjacent to the hub.

    Key insight:
    - The output contains two linked clues at once: branches still advertise each seed's majority color, while the
      seed interior flips majority and minority; the hub center then inverts the original counting rule by choosing
      the least common adjacent branch color instead of the most common one.
    """
    return _solve_core(grid, swap_seed_colors=True, hub_mode="least")


def _try_carve_path(canvas, path_color, start, dest_border_cells, max_bends=MAX_BENDS):
    h, w = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r, c, prev, allow_dest):
        if not (0 <= r < h and 0 <= c < w) or canvas[int(r), int(c)] != 0:
            return False
        for dr, dc in _DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            v = int(canvas[nr, nc])
            if v == int(path_color):
                if allow_dest and (nr, nc) in dest_set:
                    continue
                return False
            if v != 0:
                return False
        return True

    sr, sc = start
    init_dirs = []
    for dr, dc in _DIRS:
        nr, nc = int(sr + dr), int(sc + dc)
        if 0 <= nr < h and 0 <= nc < w and int(canvas[nr, nc]) == 0:
            init_dirs.append((dr, dc))
    if not init_dirs:
        return None
    random.shuffle(init_dirs)

    n_bends = random.randint(0, max_bends)

    def _attempt(initial_dir):
        path = []
        cur = start
        cur_dir = initial_dir
        for seg_i in range(n_bends + 1):
            is_last = seg_i == n_bends
            max_steps = max(h, w) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr, nc = int(cur[0] + cur_dir[0]), int(cur[1] + cur_dir[1])
                if any((int(nr + dr), int(nc + dc)) in dest_set for dr, dc in _DIRS) and _check_neighbors(
                    nr, nc, cur, allow_dest=True
                ):
                    canvas[int(nr), int(nc)] = int(path_color)
                    path.append((int(nr), int(nc)))
                    return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[int(nr), int(nc)] = int(path_color)
                    path.append((int(nr), int(nc)))
                    cur = (int(nr), int(nc))
                else:
                    blocked = True
                    break
            if blocked or is_last:
                for r, c in path:
                    canvas[int(r), int(c)] = 0
                return None
            perps = [(-cur_dir[1], cur_dir[0]), (cur_dir[1], -cur_dir[0])]
            random.shuffle(perps)
            cur_dir = perps[0]
        for r, c in path:
            canvas[int(r), int(c)] = 0
        return None

    for d in init_dirs:
        for _ in range(PATH_DIR_RETRIES):
            res = _attempt(d)
            if res is not None:
                return res
    return None


def _compute_seed_zone(edge, ir0, ic0, ir1, ic1, h, w):
    if edge == "top":
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(w - 1, ic1 + 1))
    if edge == "bottom":
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(w - 1, ic1 + 1))
    if edge == "left":
        return (max(0, ir0 - 1), ic0, min(h - 1, ir1 + 1), ic1 + 1)
    return (max(0, ir0 - 1), max(0, ic0 - 1), min(h - 1, ir1 + 1), ic1)


def _place_seed_border(work, path_color, ir0, ic0, ir1, ic1, h, w):
    border_cells = []
    landing_cells = []
    for cond, cells in [
        (ir0 > 0, [(ir0 - 1, c) for c in range(ic0, ic1 + 1)]),
        (ir1 < h - 1, [(ir1 + 1, c) for c in range(ic0, ic1 + 1)]),
        (ic0 > 0, [(r, ic0 - 1) for r in range(ir0, ir1 + 1)]),
        (ic1 < w - 1, [(r, ic1 + 1) for r in range(ir0, ir1 + 1)]),
    ]:
        if cond:
            for r, c in cells:
                work[int(r), int(c)] = int(path_color)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))

    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < w - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < h - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < h - 1 and ic1 < w - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[int(rc[0]), int(rc[1])] = int(path_color)
            border_cells.append((int(rc[0]), int(rc[1])))
    return border_cells, landing_cells


def _zones_overlap(zone, others, gap):
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge, h, w):
    ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    if edge == "top":
        ir0, ic0 = 0, random.randint(0, w - iw)
    elif edge == "bottom":
        ir0, ic0 = h - ih, random.randint(0, w - iw)
    elif edge == "left":
        ir0, ic0 = random.randint(0, h - ih), 0
    else:
        ir0, ic0 = random.randint(0, h - ih), w - iw
    return ir0, ic0, ir0 + ih - 1, ic0 + iw - 1


def _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid):
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(n_int * MINORITY_FRAC_LO))
    max_minor = max(min_minor, int(n_int * MINORITY_FRAC_HI))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[int(r), int(c)] = int(minority)
        seed_id_grid[int(r), int(c)] = int(sid)
    for r, c in interior_cells[n_minor:]:
        work[int(r), int(c)] = int(majority)
        seed_id_grid[int(r), int(c)] = int(sid)


def _carve_all_paths(work, path_color, hub_perimeter, seeds):
    used_hub_starts = set()
    paths_per_seed = {i: 0 for i in range(len(seeds))}

    seed_order = list(range(len(seeds)))
    random.shuffle(seed_order)
    for si in seed_order:
        ok = False
        for _ in range(PATH_FIRST_ROUND_RETRIES):
            avail = [s for s in hub_perimeter if s not in used_hub_starts]
            if not avail:
                break
            start = random.choice(avail)
            path = _try_carve_path(work, path_color, start, seeds[si]["landing_cells"])
            if path is not None:
                used_hub_starts.add(start)
                paths_per_seed[si] += 1
                ok = True
                break
        if not ok:
            return None

    extra_target = sum(random.randint(0, 2) for _ in seeds)
    for _ in range(extra_target * 2):
        if len(used_hub_starts) >= len(hub_perimeter):
            break
        candidates = [si for si in range(len(seeds)) if paths_per_seed[si] < MAX_PATHS_PER_SEED]
        if not candidates:
            break
        si = random.choice(candidates)
        for _ in range(PATH_EXTRA_ROUND_RETRIES):
            avail = [s for s in hub_perimeter if s not in used_hub_starts]
            if not avail:
                break
            start = random.choice(avail)
            path = _try_carve_path(work, path_color, start, seeds[si]["landing_cells"])
            if path is not None:
                used_hub_starts.add(start)
                paths_per_seed[si] += 1
                break
    return paths_per_seed


def _gen_attempt():
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    h = random.randint(size_lo, GRID_SIZE_MAX)
    w = random.randint(size_lo, GRID_SIZE_MAX)
    bg, path_color, seed_a, seed_b = random.sample(list(range(10)), 4)

    g = np.full((h, w), bg, dtype=int)
    hr0 = random.randint(max(2, h // 3 - 1), min(h - 5, 2 * h // 3 - 2))
    hc0 = random.randint(max(2, w // 3 - 1), min(w - 5, 2 * w // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(path_color)

    hub_perimeter = [
        (r, c)
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == int(bg), 0, g).astype(int)
    seed_id_grid = np.zeros((h, w), dtype=int)
    seeds = []
    used_zones = []
    edges = ["top", "bottom", "left", "right"]
    majority_plan = [seed_a, seed_b] + [random.choice([seed_a, seed_b]) for _ in range(n_seeds_target - 2)]
    random.shuffle(majority_plan)

    def _overlaps_hub(zone):
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    for _ in range(SEED_PLACEMENT_RETRIES):
        if len(seeds) >= n_seeds_target:
            break
        edge = random.choice(edges)
        ir0, ic0, ir1, ic1 = _pick_seed_interior(edge, h, w)
        border_zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, h, w)
        if _overlaps_hub(border_zone):
            continue
        if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
            continue

        majority = majority_plan[len(seeds)]
        minority = int(seed_b if int(majority) == int(seed_a) else seed_a)
        sid = len(seeds) + 1
        _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
        border_cells, landing_cells = _place_seed_border(work, path_color, ir0, ic0, ir1, ic1, h, w)
        used_zones.append(border_zone)
        seeds.append(
            {
                "id": sid,
                "majority": int(majority),
                "minority": int(minority),
                "border_cells": border_cells,
                "landing_cells": landing_cells,
                "bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
                "border_zone": border_zone,
            }
        )

    if len(seeds) < 3:
        return None
    if len({int(seed["majority"]) for seed in seeds}) < 2:
        return None

    path_counts = _carve_all_paths(work, path_color, hub_perimeter, seeds)
    if path_counts is None:
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    try:
        metrics = _behavior_metrics(inp)
    except Exception:
        return None

    if metrics["seed_swap_changed_cells"] <= 0:
        return None
    if metrics["center_changed_vs_original"] != 1:
        return None
    if metrics["hub_color_variety"] < 2:
        return None
    if metrics["hub_color_count_span"] <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        try:
            res = _gen_attempt()
        except Exception:
            res = None
        if res is not None:
            return res
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")
