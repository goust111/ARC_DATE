# Combo variant of 3dc255db (index 24)
# combo_spec: D4/0+D6/3+D8/0
# Name: 障碍停止+碰撞变色+距离排序
#
# Design analysis
# a) 具体化与行为追踪
#    - 保留原题对象：每个杯仍是单色 8 连通边框，bbox 某一条边上有唯一一个非角落单像素尖刺。
#      嘴口内部仍只有一种内容色，内容像素数决定弹射“弹药量”。
#    - 融合后的执行顺序不是逐杯独立，而是先解析全部杯，再按“bbox 中心到网格中心距离”从近到远处理。
#    - 处理某杯时，先擦除它嘴口内的内容像素，然后从尖刺沿尾部方向逐格前进：
#      1. 若下一格是背景，则写入内容色并继续；
#      2. 若下一格已经是非背景，则该格视为障碍，但命中格本身会被改成当前内容色，然后立刻停止；
#         剩余未使用的内容像素丢弃。
#    - 具体小例子：两杯在同一行相向。近中心杯 A 先向右发射，前方若先撞到远杯 B 的尖刺边框，
#      A 会把 B 的尖刺改成 A 的内容色并停止。之后轮到更远的 B 向左发射，B 又会立刻撞到 A
#      已留下的线段，把被撞中的那一格改成 B 的内容色并停止。于是同一个样例里同时出现：
#      “边框障碍停止 + 命中变色 + 先后顺序决定后续障碍”。
# b) 复杂度校准
#    - 生成器固定生成 2-3 个杯，且至少包含一组“同轴相向交互对”，避免退化成原题的单杯独立弹射。
#    - 交互对被设计成：近杯先撞远杯边框，远杯后撞近杯已生成的线。这样三项改动在同一因果链里连续发生，
#      而不是三个互不相干的小技巧。
#    - 仍保留杯方向、尺寸和是否存在第 3 个清爽 filler 杯的自由度，避免样例模板过死。
# c) 原 main() 的隐式假设审查
#    - 原代码边检测边写出，这依赖“杯之间无交互”。融合后不成立，因此必须先用静态输入 `g`
#      检出所有杯，再排序后统一执行。
#    - 原代码默认尾射路径一直可写到边界。融合后不成立：非背景像素会成为障碍，且命中格要改色后停止。
#    - 原代码没有区分“检测阶段看什么”和“执行阶段看什么”。融合后必须：杯识别只看 `g`，
#      障碍判断只看动态 `out`，否则先前生成的线会污染杯检测，或后续杯看不到前序杯造成的阻挡。
#    - 命中格改色后就停止，这是新隐式假设。生成器必须保证路径上不会出现“应该继续穿透多个障碍”的歧义。
# d) 放置合法性
#    - 所有杯 bbox 仍保持至少 1 格间隔，避免不同边框颜色在 8 连通下粘连。
#    - 指定交互对共线相向；近杯到远杯尖刺之间的走廊在输入中必须全空，确保近杯先撞到的就是远杯边框。
#    - 若存在 filler 杯，则它的整条有效路径保持清空，且不与交互走廊或其他 bbox 相交。
# e) 视觉可观察性
#    - 排序依据是空间位置到全局中心的距离，完全由形状位置可见，不依赖颜色编号。
#    - 输出里至少会有一个“边框像素被改色”和一个“已生成线段被后续杯改色”的现象；同时输入中的源内容
#      会被擦除，和原题的“只是画线”形成可见差异。
#    - 交互对共用同一行或列，肉眼容易看出谁更靠近中心，以及后处理杯为什么会被前处理杯挡住。
# f) 可视化整洁度
#    - 关闭 stray 噪声；只允许“线撞边框”和“线撞先前线段”两类交互。
#    - 第三个杯若存在，只承担无冲突的普通弹射，不再叠加额外障碍，避免同一区域覆盖层数过多。
# g) 变化可推断性
#    - 颜色不做隐藏映射：命中谁，就把命中格改成发射线本身的内容色。不同杯尽量使用不同 border/content 色，
#      让解题者能直接追踪哪条线改了哪个像素。
# 9) 新 solver 假设与 generator 保证
#    - 假设：每个有效杯在输入中仍可被原始 mouth-flood 规则唯一识别。
#      保证：每杯 border 颜色唯一，内容色只出现在自己的 mouth 内。
#    - 假设：排序 key 需要稳定；若中心距离相同会产生歧义。
#      保证：生成器拒绝中心距离重复的布局。
#    - 假设：交互只依赖当前 `out` 中可见的非背景像素。
#      保证：交互对走廊在输入中全空，额外 filler 的路径不接触交互走廊。

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood inside bbox, blocked by the cup border."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _cup_order_key(bbox: tuple[int, int, int, int], H: int, W: int) -> tuple[int, int, int]:
    """Visible order: squared distance from bbox-center to grid-center, then top-left."""
    r0, c0, r1, c1 = bbox
    dr2 = (r0 + r1) - (H - 1)
    dc2 = (c0 + c1) - (W - 1)
    return dr2 * dr2 + dc2 * dc2, r0, c0


def _detect_cups(grid: np.ndarray) -> list[dict]:
    """Parse cups once from the immutable input grid."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cups: list[dict] = []

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [
                (d, cells[0]) for d, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]
            cups.append(
                {
                    "border_color": int(border_col),
                    "bbox": bbox,
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "content_color": content_col,
                    "content_cells": list(content_cells),
                }
            )

    return cups


def _emit_with_obstacles(
    out: np.ndarray,
    spike: tuple[int, int],
    tail_dir: str,
    content_color: int,
    content_count: int,
    bg: int,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    """Write one ordered shot onto `out`, stopping after recoloring the first obstacle."""
    H, W = out.shape
    dr, dc = DIR4[tail_dir]
    rr, cc = int(spike[0]) + dr, int(spike[1]) + dc
    painted: list[tuple[int, int]] = []
    while len(painted) < content_count and 0 <= rr < H and 0 <= cc < W:
        if int(out[rr, cc]) == bg:
            out[rr, cc] = int(content_color)
            painted.append((rr, cc))
            rr += dr
            cc += dc
            continue
        out[rr, cc] = int(content_color)
        painted.append((rr, cc))
        return painted, (rr, cc)
    return painted, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Detect all cups from the static input.
    2. Sort cups from near-center to far-center.
    3. For each cup, erase its source content.
    4. Emit outward; bg cells are filled normally, while the first non-bg obstacle cell is
       recolored to the ray color and ends the shot immediately.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    cups = _detect_cups(g)
    cups.sort(key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

        _emit_with_obstacles(
            out,
            cup["spike"],
            cup["tail_dir"],
            int(cup["content_color"]),
            len(cup["content_cells"]),
            bg,
        )

    return out


# ---- generator ----

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical funnel with tail pointing up."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1 : L + hL + 1, 0] = color
    if hR:
        g[R + 1 : R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(
    patch: np.ndarray,
    border_color: int,
    opposite_side: str,
) -> list[tuple[int, int]]:
    """Flood-fill the cup mouth interior from the mouth side."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate canonical tail-up cup to the requested direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Sample a cup shape that respects the original bbox constraints."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    L_arm = int(rng.integers(lo, hi + 1))
    R_arm = int(rng.integers(lo, hi + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    wlo, whi = WALL_EXT_RANGE
    hL = int(rng.integers(wlo, whi + 1))
    hR = int(rng.integers(wlo, whi + 1))
    if hL + hR == 0:
        hL = 1
    while max(L_arm + hL, R_arm + hR) + 1 > max_h_canon:
        if L_arm + hL >= R_arm + hR and hL > 0:
            hL -= 1
        elif hR > 0:
            hR -= 1
        else:
            break
    if hL + hR == 0:
        return None
    return L_arm, R_arm, hL, hR


def _build_cup_spec(
    rng: np.random.Generator,
    border_color: int,
    content_color: int,
    direction: str,
) -> dict | None:
    """Sample one cup blueprint with fixed direction."""
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape
    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None
    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    return {
        "border_color": int(border_color),
        "content_color": int(content_color),
        "direction": direction,
        "patch": patch,
        "spike_rel": spike,
        "mouth_rel": mouth,
        "mouth_size": len(mouth),
        "ph": int(patch.shape[0]),
        "pw": int(patch.shape[1]),
    }


def _ray_from_spike(
    spike_abs: tuple[int, int],
    direction: str,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """All cells from the spike outward to the edge."""
    dr, dc = DIR4[direction]
    rr, cc = spike_abs[0] + dr, spike_abs[1] + dc
    cells = []
    while 0 <= rr < H and 0 <= cc < W:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _patch_border_cells(patch: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    coords = np.argwhere(patch != 0)
    return {(r0 + int(r), c0 + int(c)) for r, c in coords}


def _finalize_cup(
    spec: dict,
    r0: int,
    c0: int,
    content_count: int,
    H: int,
    W: int,
) -> dict:
    """Attach absolute placement data to a cup blueprint."""
    patch = spec["patch"]
    ph, pw = patch.shape
    spike_abs = (r0 + int(spec["spike_rel"][0]), c0 + int(spec["spike_rel"][1]))
    return {
        "border_color": int(spec["border_color"]),
        "content_color": int(spec["content_color"]),
        "direction": str(spec["direction"]),
        "patch": patch.copy(),
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "spike_abs": spike_abs,
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in spec["mouth_rel"]],
        "content_count": int(content_count),
        "border_abs": _patch_border_cells(patch, r0, c0),
        "full_ray": _ray_from_spike(spike_abs, str(spec["direction"]), H, W),
    }


def _distance_value(bbox: tuple[int, int, int, int], H: int, W: int) -> int:
    return _cup_order_key(bbox, H, W)[0]


def _find_pair_placement(
    rng: np.random.Generator,
    H: int,
    W: int,
    spec_a: dict,
    spec_b: dict,
    layout: str,
) -> tuple[dict, dict] | None:
    """
    Place a near-center cup A and a farther cup B on one axis.

    A is processed first and hits B's spike border.
    B is processed later and immediately hits A's already-written line.
    """
    ph_a, pw_a = int(spec_a["ph"]), int(spec_a["pw"])
    ph_b, pw_b = int(spec_b["ph"]), int(spec_b["pw"])
    sr_a, sc_a = spec_a["spike_rel"]
    sr_b, sc_b = spec_b["spike_rel"]
    lim_a = min(CONTENT_COUNT_MAX, int(spec_a["mouth_size"]))
    lim_b = min(CONTENT_COUNT_MAX, int(spec_b["mouth_size"]))
    if lim_b < 2:
        return None

    candidates = []
    if layout in ("right", "left"):
        row_lo = max(int(sr_a), int(sr_b))
        row_hi = min(H - ph_a + int(sr_a), H - ph_b + int(sr_b))
        rows = list(range(row_lo, row_hi + 1))
        rng.shuffle(rows)
        for spike_row in rows:
            r_a = spike_row - int(sr_a)
            r_b = spike_row - int(sr_b)
            a_cols = list(range(W - pw_a + 1))
            b_cols = list(range(W - pw_b + 1))
            rng.shuffle(a_cols)
            rng.shuffle(b_cols)
            for c_a in a_cols:
                sa = c_a + int(sc_a)
                bbox_a = (r_a, c_a, r_a + ph_a - 1, c_a + pw_a - 1)
                for c_b in b_cols:
                    sb = c_b + int(sc_b)
                    bbox_b = (r_b, c_b, r_b + ph_b - 1, c_b + pw_b - 1)
                    if layout == "right":
                        if c_b < c_a + pw_a + 1:
                            continue
                        gap = sb - sa - 1
                    else:
                        if c_a < c_b + pw_b + 1:
                            continue
                        gap = sa - sb - 1
                    if gap < 2:
                        continue
                    if _distance_value(bbox_a, H, W) >= _distance_value(bbox_b, H, W):
                        continue
                    hit_step_a = gap + 1
                    if hit_step_a + 1 > lim_a:
                        continue
                    candidates.append((r_a, c_a, r_b, c_b, hit_step_a))
    else:
        col_lo = max(int(sc_a), int(sc_b))
        col_hi = min(W - pw_a + int(sc_a), W - pw_b + int(sc_b))
        cols = list(range(col_lo, col_hi + 1))
        rng.shuffle(cols)
        for spike_col in cols:
            c_a = spike_col - int(sc_a)
            c_b = spike_col - int(sc_b)
            a_rows = list(range(H - ph_a + 1))
            b_rows = list(range(H - ph_b + 1))
            rng.shuffle(a_rows)
            rng.shuffle(b_rows)
            for r_a in a_rows:
                sa = r_a + int(sr_a)
                bbox_a = (r_a, c_a, r_a + ph_a - 1, c_a + pw_a - 1)
                for r_b in b_rows:
                    sb = r_b + int(sr_b)
                    bbox_b = (r_b, c_b, r_b + ph_b - 1, c_b + pw_b - 1)
                    if layout == "down":
                        if r_b < r_a + ph_a + 1:
                            continue
                        gap = sb - sa - 1
                    else:
                        if r_a < r_b + ph_b + 1:
                            continue
                        gap = sa - sb - 1
                    if gap < 2:
                        continue
                    if _distance_value(bbox_a, H, W) >= _distance_value(bbox_b, H, W):
                        continue
                    hit_step_a = gap + 1
                    if hit_step_a + 1 > lim_a:
                        continue
                    candidates.append((r_a, c_a, r_b, c_b, hit_step_a))

    if not candidates:
        return None

    r_a, c_a, r_b, c_b, hit_step_a = candidates[int(rng.integers(0, len(candidates)))]
    count_a = int(rng.integers(hit_step_a + 1, lim_a + 1))
    count_b = int(rng.integers(2, lim_b + 1))
    return (
        _finalize_cup(spec_a, r_a, c_a, count_a, H, W),
        _finalize_cup(spec_b, r_b, c_b, count_b, H, W),
    )


def _pair_reserved_bg_cells(cup_a: dict, cup_b: dict) -> set[tuple[int, int]]:
    """Input bg cells that must stay empty for the interaction pair to behave as intended."""
    dr, dc = DIR4[cup_a["direction"]]
    rr, cc = cup_a["spike_abs"][0] + dr, cup_a["spike_abs"][1] + dc
    reserved: set[tuple[int, int]] = set()
    while (rr, cc) != cup_b["spike_abs"]:
        reserved.add((rr, cc))
        rr += dr
        cc += dc
    return reserved


def _try_place_clear_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    spec: dict,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Place a filler cup whose whole ray stays clear and independent."""
    ph, pw = int(spec["ph"]), int(spec["pw"])
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue

        trial = _finalize_cup(spec, r0, c0, 1, H, W)
        full_ray = trial["full_ray"]
        if not full_ray:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in full_ray):
            continue
        max_count = min(CONTENT_COUNT_MAX, len(trial["mouth_abs"]), len(full_ray))
        if max_count < 1:
            continue
        trial["content_count"] = int(rng.integers(1, max_count + 1))
        return trial
    return None


def _distance_keys_are_unique(cups_meta: list[dict], H: int, W: int) -> bool:
    vals = [_distance_value(meta["bbox"], H, W) for meta in cups_meta]
    return len(vals) == len(set(vals))


def _draw_borders(canvas: np.ndarray, cups_meta: list[dict]) -> None:
    for meta in cups_meta:
        r0, c0, r1, c1 = meta["bbox"]
        patch = meta["patch"]
        canvas[r0 : r1 + 1, c0 : c1 + 1][patch != 0] = int(meta["border_color"])


def _scatter_content(rng: np.random.Generator, canvas: np.ndarray, cups_meta: list[dict]) -> bool:
    for meta in cups_meta:
        count = int(meta["content_count"])
        mouth_abs = meta["mouth_abs"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas[r, c] = int(meta["content_color"])
    return True


def _simulate_combo(canvas_in: np.ndarray) -> tuple[np.ndarray, dict]:
    """Independent simulation used by the generator and behavior checks."""
    g = np.array(canvas_in, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()
    cups = _detect_cups(g)
    border_owner = {}
    for cup in cups:
        for r, c in np.argwhere(g == int(cup["border_color"])):
            border_owner[(int(r), int(c))] = int(cup["border_color"])
    cups.sort(key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))

    painted_so_far: set[tuple[int, int]] = set()
    metrics = {"collision_hits": 0, "border_hits": 0, "line_hits": 0, "ordered_blocks": 0}

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

        painted, hit_cell = _emit_with_obstacles(
            out,
            cup["spike"],
            cup["tail_dir"],
            int(cup["content_color"]),
            len(cup["content_cells"]),
            bg,
        )
        if hit_cell is not None:
            metrics["collision_hits"] += 1
            metrics["ordered_blocks"] += 1
            if hit_cell in painted_so_far:
                metrics["line_hits"] += 1
            elif hit_cell in border_owner:
                metrics["border_hits"] += 1
        for cell in painted:
            painted_so_far.add(cell)

    return out, metrics


def _sample_colors(rng: np.random.Generator, num_cups: int) -> list[tuple[int, int]]:
    colors = list(range(1, 10))
    rng.shuffle(colors)
    picked = colors[: 2 * num_cups]
    return [(int(picked[2 * i]), int(picked[2 * i + 1])) for i in range(num_cups)]


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Generate one instance with a guaranteed fused causal chain."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    color_pairs = _sample_colors(rng, num_cups)
    layout_to_dirs = {
        "right": ("R", "L"),
        "left": ("L", "R"),
        "down": ("D", "U"),
        "up": ("U", "D"),
    }
    layout = str(rng.choice(list(layout_to_dirs)))
    dir_a, dir_b = layout_to_dirs[layout]

    spec_a = _build_cup_spec(rng, color_pairs[0][0], color_pairs[0][1], dir_a)
    spec_b = _build_cup_spec(rng, color_pairs[1][0], color_pairs[1][1], dir_b)
    if spec_a is None or spec_b is None:
        return None

    pair = _find_pair_placement(rng, H, W, spec_a, spec_b, layout)
    if pair is None:
        return None
    cups_meta = [pair[0], pair[1]]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    for meta in cups_meta:
        r0, c0, r1, c1 = meta["bbox"]
        occ_cup[r0 : r1 + 1, c0 : c1 + 1] = True
    for r, c in _pair_reserved_bg_cells(cups_meta[0], cups_meta[1]):
        occ_path[r, c] = True

    for idx in range(2, num_cups):
        placed = None
        for _ in range(40):
            direction = str(rng.choice(["U", "D", "L", "R"]))
            spec = _build_cup_spec(rng, color_pairs[idx][0], color_pairs[idx][1], direction)
            if spec is None:
                continue
            placed = _try_place_clear_cup(rng, H, W, spec, occ_cup, occ_path)
            if placed is None:
                continue
            cups_meta.append(placed)
            r0, c0, r1, c1 = placed["bbox"]
            occ_cup[r0 : r1 + 1, c0 : c1 + 1] = True
            for r, c in placed["full_ray"]:
                occ_path[r, c] = True
            break
        if placed is None:
            return None

    if not _distance_keys_are_unique(cups_meta, H, W):
        return None

    canvas_in = np.zeros((H, W), dtype=int)
    _draw_borders(canvas_in, cups_meta)
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    try:
        expected, metrics = _simulate_combo(canvas_in)
        produced = main(canvas_in)
    except Exception:
        return None

    if not np.array_equal(produced, expected):
        return None
    if metrics["border_hits"] < 1 or metrics["line_hits"] < 1 or metrics["ordered_blocks"] < 2:
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(
        f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts"
    )
