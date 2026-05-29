# Variant of 7b80bb43
# Dimension: D6 — 干扰对角链的角色 (transform)
# Change: 0-2条独立干扰对角链放置在空白背景区域，不与任何线段相邻；输出中被删除（及其叶子像素被修剪）... → 干扰链连接两条线段形成新路径

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# variant_spec:
#   core_change:
#     original: |
#       额外的对角链出现在纯背景区，不接触线段。solver 可以把所有对角-only 像素都视作
#       “潜在线索”，因为这些干扰链只会导致删除，不会靠近任何可填补的线段。
#     variant: |
#       额外的对角链现在故意放在线段端点之间，看起来像一条可以把两条独立线段桥接起来的
#       45 度路径。它们仍然必须在输出中被删除，而且绝不能触发水平/竖直补线。
#   concrete_objects:
#     true_gap_hint:
#       shape: 长度 2 的 45 度链
#       relation: 只对角接触一条被切断的直线的一端
#       effect: 删除该链，并填回同一行或同一列上的 3 像素缺口
#     bridge_decoy:
#       shape: 长度 2 或 3 的 45 度链
#       relation: 两端分别对角接触两条不同线段的自由端点，形成“看似可桥接”的新路径
#       effect: 只删除链本身；不做任何桥接
#   behavior_trace:
#     original_logic: |
#       原 main() 先找出所有对角-only 像素，再在整张图上扫描任意靠近这些像素的短横/竖向空隙；
#       只要空隙长度 <= MAXG 就直接补上。
#     failure_under_variant: |
#       若桥接诱饵链刚好位于两条线段之间，旧逻辑会把“靠近对角链”误解为“应该补线”，从而在
#       不该连通的地方新增水平/竖直连接。
#     new_logic: |
#       先按 8 连通分解对角链组件，并统计每个组件对角接触到多少个非对角线段像素。
#       只有“长度恰为 2 且仅接触 1 个根像素”的组件才被视为真实 gap hint；它们允许触发
#       3 像素补线。接触 2 个根像素的组件就是桥接诱饵，只删除，不参与补线。
#   complexity_calibration:
#     rationale: |
#       真实提示链固定为长度 2；诱饵链允许长度 2 或 3，但必须连接两条不同组件、不同朝向
#       (H/V) 的线段端点。这样视觉上能重复出现“单根提示”和“双根桥接”两类案例，差异足够明显，
#       又不会引入太多自由度导致样例难归纳。
#   implicit_assumptions_review:
#     main_old_assumptions:
#       - 对角链附近只会有单条待修复直线，不会有“连接两条线”的诱饵。
#       - 所有对角链都可以参与设置 MAXG。
#       - 在对角链邻域内做叶子修剪不会误伤合法端点。
#     adaptation:
#       - 补线资格改为组件级判定，而不是“全体对角像素一视同仁”。
#       - gap 长度直接锁定为 3，不再让桥接链影响阈值。
#       - 叶子修剪只围绕真实 gap hint 的邻域执行；桥接诱饵附近不修剪，以免删掉合法端点。
#   placement_legality:
#     true_hints: |
#       仍沿原题 output-first 流程，在输出网络上切出 3 像素横/竖向缺口，并放置一条长度 2 的
#       提示链。validator 保证链像素在输入中没有正交前景邻居。
#     bridge_decoys: |
#       从输出网络中提取各线段的“内部自由端点”，枚举位于同一对角线上的 H 端点和 V 端点对。
#       只接受链长 2/3、两端属于不同 4 连通组件、链路径完全落在输出背景中且对当前输入仍满足
#       对角-only 条件的方案。
#   visual_observability:
#     note: |
#       真实提示链总是只贴着一条断线的一端；桥接诱饵总是明显横跨两条独立线段端点，且不恢复
#       任何直线。看 2-3 个样例时，人可以对比“单端提示会补 3 格直线”与“双端桥接只删除”。
#   cleanliness:
#     note: |
#       诱饵链只连接不同组件且不同朝向的端点，避免和真实“同一行/列缺口”混在一起；同时不让
#       链覆盖输出中的任何永久线段像素，减少视觉冲突。
#   color_inference:
#     note: |
#       仍只有 1 个前景色和 1 个背景色，没有新增颜色映射规则。
#   new_main_assumptions_to_guarantee:
#     - 真提示链: 长度 2、根像素数 1、目标缺口长度恰为 3
#     - 桥接诱饵: 根像素数 2、连接不同组件的 H/V 自由端点、不会制造需要修剪的残余枝杈
#     - generate_input() 必须持续构造满足以上判定边界的输入
# ---------------------------------------------------------------------------


# --- Configuration constants ---

GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

VALID_CONFIGS = [
    {"segments": 6, "h": 3, "v": 3, "junctions": 2},
    {"segments": 7, "h": 3, "v": 4, "junctions": 2},
    {"segments": 7, "h": 4, "v": 3, "junctions": 2},
    {"segments": 8, "h": 4, "v": 4, "junctions": 3},
]
T_SLOTS = ("LT", "RT", "LB", "RB")

EDGE_MARGIN = 3
ROW_COL_GAP = 2
T_BRANCH_OFFSET_MIN = 3
T_BRANCH_OFFSET_MAX = 9
H_SEG_END_MARGIN = 4
H_SEG_MAX_END = 25

SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

BRIDGE_LENGTHS = (3, 2)

GEN_RETRIES = 2000


def _deg4(m: np.ndarray, r: int, c: int) -> int:
    h, w = m.shape
    d = 0
    if r > 0 and bool(m[r - 1, c]):
        d += 1
    if r + 1 < h and bool(m[r + 1, c]):
        d += 1
    if c > 0 and bool(m[r, c - 1]):
        d += 1
    if c + 1 < w and bool(m[r, c + 1]):
        d += 1
    return int(d)


def _has_diag(m: np.ndarray, r: int, c: int) -> bool:
    h, w = m.shape
    if r > 0 and c > 0 and bool(m[r - 1, c - 1]):
        return True
    if r > 0 and c + 1 < w and bool(m[r - 1, c + 1]):
        return True
    if r + 1 < h and c > 0 and bool(m[r + 1, c - 1]):
        return True
    if r + 1 < h and c + 1 < w and bool(m[r + 1, c + 1]):
        return True
    return False


def _diag_component_infos(mask: np.ndarray, diag: np.ndarray) -> list[dict]:
    h, w = mask.shape
    infos: list[dict] = []
    for coords in components_mask(diag, connectivity=8):
        roots: set[tuple[int, int]] = set()
        for r, c in coords:
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not bool(diag[rr, cc]):
                    roots.add((rr, cc))
        infos.append(
            {
                "coords": coords,
                "size": int(coords.shape[0]),
                "roots": sorted(roots),
                "root_count": int(len(roots)),
            }
        )
    return infos


def _diag_roots_for_coords(grid: np.ndarray, coords: list[tuple[int, int]], fg: int) -> set[tuple[int, int]]:
    gh, gw = grid.shape
    coord_set = set(coords)
    roots: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < gh and 0 <= cc < gw and int(grid[rr, cc]) == int(fg) and (rr, cc) not in coord_set:
                roots.add((rr, cc))
    return roots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise, but only let single-root chains repair a straight 3-pixel gap.

    D6 twist:
    - True hints: a 2-pixel diagonal chain touching exactly one line root, which indicates a broken straight line.
    - Bridge decoys: a 2-3 pixel diagonal chain touching two different line roots, visually spanning between segments.
      These are deleted but must not trigger any connection.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

    mask = g == fg
    h, w = g.shape

    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, r, c) == 0 and _has_diag(mask, r, c):
                diag[r, c] = True

    fill_diag = np.zeros((h, w), dtype=bool)
    near_fill = np.zeros((h, w), dtype=bool)
    for info in _diag_component_infos(mask, diag):
        if info["size"] != CHAIN_LENGTH or info["root_count"] != 1:
            continue
        for r, c in info["coords"]:
            fill_diag[r, c] = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w:
                        near_fill[rr, cc] = True

    out = g.copy()
    out[diag] = bg

    base = mask & (~diag)

    for r in range(h):
        cols = np.where(base[r, :])[0]
        if cols.size == 0:
            continue
        segs: list[tuple[int, int]] = []
        start = int(cols[0])
        prev = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
            else:
                segs.append((start, prev))
                start = prev = c
        segs.append((start, prev))

        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != GAP_SIZE:
                continue
            rr0 = max(0, r - 1)
            rr1 = min(h - 1, r + 1)
            cc0 = max(0, gap0 - 1)
            cc1 = min(w - 1, gap1 + 1)
            if bool(fill_diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[r, gap0 : gap1 + 1] = fg

    for c in range(w):
        rows = np.where(base[:, c])[0]
        if rows.size == 0:
            continue
        segs = []
        start = int(rows[0])
        prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r == prev + 1:
                prev = r
            else:
                segs.append((start, prev))
                start = prev = r
        segs.append((start, prev))

        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != GAP_SIZE:
                continue
            rr0 = max(0, gap0 - 1)
            rr1 = min(h - 1, gap1 + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w - 1, c + 1)
            if bool(fill_diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[gap0 : gap1 + 1, c] = fg

    m = out == fg
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(m[r, c]) or not bool(near_fill[r, c]):
                    continue
                if _deg4(m, r, c) != 1:
                    continue
                if not _has_diag(m, r, c):
                    continue
                to_clear.append((r, c))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out2 = np.full((h, w), bg, dtype=int)
    out2[m] = fg
    return out2


def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int = ROW_COL_GAP) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - p) >= gap for p in picked):
            picked.append(v)
            if len(picked) == k:
                return sorted(picked)
    return []


def _chain_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + i, gap_start - 1 + i) for i in range(1, CHAIN_LENGTH + 1)]
    return [(row + i, gap_start + GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]
    return [(gap_start + GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    gh, gw = grid.shape
    for r, c in coords:
        if not (0 <= r < gh and 0 <= c < gw) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    fg: int,
    occupied: set[tuple[int, int]],
) -> bool:
    gh, gw = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < gh and 0 <= cc < gw and int(grid[rr, cc]) == int(fg):
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < gh and 0 <= cc < gw and int(grid[rr, cc]) == int(fg)):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int, branch_col: int, end_col: int) -> tuple[dict, dict]:
    gh, gw = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else gw - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else gh - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}, "simple": False}
    v_seg = {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}, "simple": False}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    gw = out.shape[1]
    c0 = 0 if side == "left" else gw - length
    c1 = length - 1 if side == "left" else gw - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": row, "c0": c0, "c1": c1, "blocked": set(), "simple": True}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    gh = out.shape[0]
    r0 = 0 if side == "top" else gh - length
    r1 = length - 1 if side == "top" else gh - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": col, "r0": r0, "r1": r1, "blocked": set(), "simple": True}


def _clear_segment(out: np.ndarray, seg: dict, bg: int) -> None:
    if "row" in seg:
        out[int(seg["row"]), int(seg["c0"]) : int(seg["c1"]) + 1] = bg
    else:
        out[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["col"])] = bg


def _force_bridge_alignment(
    rng: random.Random,
    out: np.ndarray,
    h_segments: list[dict],
    v_segments: list[dict],
    fg: int,
    bg: int,
    expected_components: int,
) -> bool:
    simple_h = [seg for seg in h_segments if bool(seg.get("simple"))]
    simple_v = [seg for seg in v_segments if bool(seg.get("simple"))]
    if not simple_h or not simple_v:
        return False

    h_order = simple_h[:]
    v_order = simple_v[:]
    rng.shuffle(h_order)
    rng.shuffle(v_order)
    gh, gw = out.shape

    for h_seg in h_order:
        row = int(h_seg["row"])
        h_left = int(h_seg["c0"]) == 0
        h_sign = 1 if h_left else -1
        for v_seg in v_order:
            col = int(v_seg["col"])
            v_top = int(v_seg["r0"]) == 0
            v_sign = 1 if v_top else -1
            trial_lengths = list(BRIDGE_LENGTHS)
            rng.shuffle(trial_lengths)
            for chain_len in trial_lengths:
                span = int(chain_len + 1)
                desired_h_end = int(col - h_sign * span)
                desired_v_end = int(row - v_sign * span)
                if h_left:
                    new_c0, new_c1 = 0, desired_h_end
                else:
                    new_c0, new_c1 = desired_h_end, gw - 1
                if v_top:
                    new_r0, new_r1 = 0, desired_v_end
                else:
                    new_r0, new_r1 = desired_v_end, gh - 1

                h_len = int(new_c1 - new_c0 + 1)
                v_len = int(new_r1 - new_r0 + 1)
                if not (SIMPLE_H_LEN_MIN <= h_len <= SIMPLE_H_LEN_MAX):
                    continue
                if v_len < 2:
                    continue
                if not (0 <= new_c0 <= new_c1 < gw and 0 <= new_r0 <= new_r1 < gh):
                    continue

                trial = out.copy()
                _clear_segment(trial, h_seg, bg)
                _clear_segment(trial, v_seg, bg)
                if bool((trial[row, new_c0 : new_c1 + 1] != bg).any()):
                    continue
                if bool((trial[new_r0 : new_r1 + 1, col] != bg).any()):
                    continue

                draw_line(trial, row, new_c0, end_r=row, end_c=new_c1, color=fg)
                draw_line(trial, new_r0, col, end_r=new_r1, end_c=col, color=fg)
                comps = find_components(trial, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
                if len(comps) != expected_components:
                    continue

                out[:, :] = trial
                h_seg["c0"], h_seg["c1"] = int(new_c0), int(new_c1)
                v_seg["r0"], v_seg["r1"] = int(new_r0), int(new_r1)
                return True
    return False


def _h_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((g, anchor))
    return options


def _v_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                options.append((g, anchor, dc))
    return options


def _segment_endpoints(out: np.ndarray, h_segments: list[dict], v_segments: list[dict], bg: int) -> list[dict]:
    labels = np.full(out.shape, -1, dtype=int)
    for idx, comp in enumerate(find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")):
        for r, c in comp["coords"]:
            labels[int(r), int(c)] = int(idx)

    gh, gw = out.shape
    endpoints: list[dict] = []
    for seg in h_segments:
        row = int(seg["row"])
        if int(seg["c0"]) == 0:
            col = int(seg["c1"])
            endpoints.append({"coord": (row, col), "axis": "h", "out_sign": 1, "comp": int(labels[row, col])})
        elif int(seg["c1"]) == gw - 1:
            col = int(seg["c0"])
            endpoints.append({"coord": (row, col), "axis": "h", "out_sign": -1, "comp": int(labels[row, col])})
        else:
            raise AssertionError("horizontal segment must touch an edge")

    for seg in v_segments:
        col = int(seg["col"])
        if int(seg["r0"]) == 0:
            row = int(seg["r1"])
            endpoints.append({"coord": (row, col), "axis": "v", "out_sign": 1, "comp": int(labels[row, col])})
        elif int(seg["r1"]) == gh - 1:
            row = int(seg["r0"])
            endpoints.append({"coord": (row, col), "axis": "v", "out_sign": -1, "comp": int(labels[row, col])})
        else:
            raise AssertionError("vertical segment must touch an edge")
    return endpoints


def _diag_chain_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    if abs(dr) != abs(dc):
        return None
    span = abs(dr)
    if span - 1 not in BRIDGE_LENGTHS:
        return None
    sdr = 0 if dr == 0 else int(dr / abs(dr))
    sdc = 0 if dc == 0 else int(dc / abs(dc))
    return [(int(a[0] + k * sdr), int(a[1] + k * sdc)) for k in range(1, span)]


def _bridge_options(
    out: np.ndarray,
    inp: np.ndarray,
    fg: int,
    bg: int,
    h_segments: list[dict],
    v_segments: list[dict],
    occupied: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    endpoints = _segment_endpoints(out, h_segments, v_segments, bg)
    options: list[list[tuple[int, int]]] = []
    for a in endpoints:
        for b in endpoints:
            if a["axis"] == b["axis"]:
                continue
            if a["comp"] == b["comp"]:
                continue
            chain = _diag_chain_between(a["coord"], b["coord"])
            if not chain:
                continue
            dr = int(b["coord"][0] - a["coord"][0])
            dc = int(b["coord"][1] - a["coord"][1])
            sdr = int(dr / abs(dr))
            sdc = int(dc / abs(dc))
            if a["axis"] == "h" and sdc != int(a["out_sign"]):
                continue
            if a["axis"] == "v" and sdr != int(a["out_sign"]):
                continue
            if b["axis"] == "h" and -sdc != int(b["out_sign"]):
                continue
            if b["axis"] == "v" and -sdr != int(b["out_sign"]):
                continue
            if not _cells_are_color(out, chain, bg):
                continue
            if not _cells_are_color(inp, chain, bg):
                continue
            if not _validate_chain(inp, chain, fg, occupied):
                continue
            if len(_diag_roots_for_coords(inp, chain, fg)) != 2:
                continue
            options.append(chain)
    return options


def _build_output(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    fg: int,
    cfg: dict,
) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), bg, dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    h_simple = cfg["h"] - cfg["junctions"]
    v_simple = cfg["v"] - cfg["junctions"]
    t_slots = rng.sample(list(T_SLOTS), cfg["junctions"])

    top_slots = [s for s in t_slots if s[1] == "T"]
    bottom_slots = [s for s in t_slots if s[1] == "B"]
    top_pool = list(range(EDGE_MARGIN, max(EDGE_MARGIN + 1, h // 2 - 2)))
    bottom_pool = list(range(max(h // 2 + 1, EDGE_MARGIN + 2), h - EDGE_MARGIN))
    top_rows = _pick_distinct(rng, top_pool, len(top_slots))
    bottom_rows = _pick_distinct(rng, bottom_pool, len(bottom_slots))
    if len(top_rows) != len(top_slots) or len(bottom_rows) != len(bottom_slots):
        return None
    row_by_slot = dict(zip(top_slots, top_rows)) | dict(zip(bottom_slots, bottom_rows))

    branch_cols = _pick_distinct(rng, list(range(EDGE_MARGIN, w - EDGE_MARGIN)), len(t_slots))
    if len(branch_cols) != len(t_slots):
        return None
    rng.shuffle(branch_cols)
    branch_by_slot = dict(zip(t_slots, branch_cols))

    for slot in t_slots:
        branch_col = branch_by_slot[slot]
        row = row_by_slot[slot]
        if slot[0] == "L":
            min_end = max(SIMPLE_H_LEN_MIN + 1, branch_col + T_BRANCH_OFFSET_MIN)
            max_end = min(w - H_SEG_END_MARGIN, branch_col + T_BRANCH_OFFSET_MAX, H_SEG_MAX_END)
        else:
            min_end = max(2, branch_col - T_BRANCH_OFFSET_MAX, w - H_SEG_MAX_END - 1)
            max_end = min(w - SIMPLE_H_LEN_MIN - 1, branch_col - T_BRANCH_OFFSET_MIN)
        if min_end > max_end:
            return None
        end_col = rng.randint(min_end, max_end)
        h_seg, v_seg = _place_t_component(out, fg, slot, row, branch_col, end_col)
        h_segments.append(h_seg)
        v_segments.append(v_seg)

    center_start = max(top_rows + [1]) + 2
    center_end = min(bottom_rows + [h - 2]) - 2
    if h_simple > 0 and center_start > center_end:
        return None

    simple_h_rows = _pick_distinct(rng, list(range(center_start, center_end + 1)), h_simple) if h_simple else []
    if len(simple_h_rows) != h_simple:
        return None
    if h_simple == 0:
        h_sides: list[str] = []
    elif h_simple == 1:
        h_sides = [rng.choice(["left", "right"])]
    else:
        h_sides = ["left", "right"]
        rng.shuffle(h_sides)

    simple_v_cols: list[int] = []
    used_branch_cols = set(branch_by_slot.values())
    if v_simple:
        has_left = "left" in h_sides
        has_right = "right" in h_sides
        if has_left and has_right:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_NEAR))
        elif has_left:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_FAR))
        elif has_right:
            col_pool = list(range(2, w - SIMPLE_V_MARGIN_NEAR))
        else:
            col_pool = list(range(2, w - 2))
        col_pool = [c for c in col_pool if c not in used_branch_cols]
        simple_v_cols = _pick_distinct(rng, col_pool, v_simple)
        if len(simple_v_cols) != v_simple:
            return None

    for row, side in zip(simple_h_rows, h_sides):
        if side == "left":
            max_len = (min(simple_v_cols) if simple_v_cols else w - EDGE_MARGIN) - 1
        else:
            max_len = w - (max(simple_v_cols) if simple_v_cols else 2) - 2
        max_len = min(SIMPLE_H_LEN_MAX, max_len)
        if max_len < SIMPLE_H_LEN_MIN:
            return None
        h_segments.append(_place_simple_h(out, fg, side, row, rng.randint(SIMPLE_H_LEN_MIN, max_len)))

    top_v = min(2, v_simple)
    top_v_cols = simple_v_cols[:top_v]
    bottom_v_cols = simple_v_cols[top_v:]
    top_limit = min(min(top_rows) - 2 if top_rows else center_start - 2, center_start - 2)
    bottom_start = max(max(bottom_rows) + 2 if bottom_rows else center_end + 2, center_end + 2)
    for col in top_v_cols:
        if top_limit < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "top", col, rng.randint(2, top_limit)))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "bottom", col, rng.randint(2, max_len)))

    if not _force_bridge_alignment(rng, out, h_segments, v_segments, fg, bg, cfg["segments"] - cfg["junctions"]):
        return None

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    return out, h_segments, v_segments


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    bridge_target = 1

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, out, fg, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, out, fg, bg, occupied)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, out, fg, bg, occupied) or h_initial
    gap_start, anchor = rng.choice(h_opts)
    row = int(h_seg["row"])
    chain = _chain_coords(row, gap_start, anchor)
    inp[row, gap_start : gap_start + GAP_SIZE] = bg
    for r, c in chain:
        inp[r, c] = fg
    occupied.update(chain)

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, bg, occupied) or v_initial
        gap_start, anchor, dc = rng.choice(v_opts)
        col = int(v_seg["col"])
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

    bridge_opts = _bridge_options(out, inp, fg, bg, h_segments, v_segments, occupied)
    if len(bridge_opts) < bridge_target:
        return None
    rng.shuffle(bridge_opts)
    placed = 0
    reserved: set[tuple[int, int]] = set()
    for coords in bridge_opts:
        if placed >= bridge_target:
            break
        if any(cell in reserved for cell in coords):
            continue
        if any(cell in occupied for cell in coords):
            continue
        if not _validate_chain(inp, coords, fg, occupied):
            continue
        for r, c in coords:
            inp[r, c] = fg
        occupied.update(coords)
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    reserved.add((int(r + dr), int(c + dc)))
        placed += 1
    if placed != bridge_target:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7b80bb43 D6 variant")
