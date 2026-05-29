# Combo variant of 3dc255db (index 31)
# combo_spec: D6/2+D14/0
# Name: 杯口方向弹射+杯形填充
#
# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 保留原题对象: 每个对象仍是单色杯框, 某一侧 bbox 边上有唯一尖刺像素, 对侧是杯口, 内容色只出现在
#   杯口内部 mouth 中.
# - 原题流程: 用尖刺方向判定 tail, 从 tail 的反侧 flood 出 mouth, 统计 mouth 里内容像素个数 n,
#   擦除这 n 个内容点, 再从尖刺往 tail 外侧画一条长度 n 的直线.
# - 融合流程: 识别杯子的方式不变, 仍以输入里散落内容点的个数 n 决定弹射长度; 但输出不再从尖刺向 tail
#   方向喷出, 而是把尖刺投影到 bbox 的对边, 从杯口外侧沿 mouth-outward 方向喷出. 同时 mouth 不再只
#   擦掉原有内容点, 而是整块填满为该内容色.
# - 小例子: tail=U 的 canonical 杯, spike 在顶部第 3 列, 底边是开口, mouth 共 5 格, 输入里其中 2 格
#   是红色. 原题输出: 顶部尖刺上方出现 2 个红点, 杯内变空. 融合后输出: 底边正下方出现 2 个红点, 且这
#   5 个 mouth 格全部变红. 差异同时体现在“路径锚点/方向反转”和“杯内稀疏->满填充”.
#
# b) 复杂度校准
# - 仍保留 1-3 个杯、4 个朝向、非对称臂长、可复用内容色和可选 stray pixel, 不削弱原题识别难度.
# - 为了让 D6 的填充行为稳定可见, generator 强制每个杯满足 content_count < len(mouth), 避免输入本来
#   就是满杯, 导致“填充”退化不可见.
# - 为了让 D14 的方向反转稳定可见, generator 只保留 mouth-side runway, 不再保留 tail-side runway;
#   因而输出线稳定出现在开口外面而非尖刺外面.
#
# c) 隐式假设审查
# - `_flood_from_bbox_side(...)` 依赖 mouth 能从与 tail 相对的 bbox 边进入, 该几何前提未变.
# - `cnt.most_common(1)` 依赖每个 mouth 只有一种真实内容色; generator 继续保证这一点.
# - 原题 `for r, c in content_cells: out[r, c] = bg` 的前提是“只处理已有内容点”; 融合后不成立, 必须改成
#   对整个 mouth 上色, 否则 D6 行为丢失.
# - 原题 `dr, dc = DIR4[tail_dir]` 默认从 spike 往 tail 外侧弹射; 融合后不成立, 必须改成先把 spike 投影
#   到对侧 mouth edge, 再沿 mouth 外法向前进.
# - 仍以 `len(content_cells)` 作为弹射长度, 而不是按 `len(mouth)` 弹射; 这保证 D6 与 D14 叠加后是
#   “填充 + 以原始内容数计数的杯口外弹射”, 而不是新的第三种计数规则.
#
# d) 放置合法性
# - 杯 bbox 继续保持至少 1 格间隔, 以免边框或填充区视觉粘连.
# - generator 在放置时预留的是 mouth-side 半直线, 该路径必须避开所有 cup bbox 与其他路径.
# - 输出里的 mouth 填充仅发生在各自 bbox 内部; 输入里的内容散点只落在对应 mouth 内部.
#
# e) 视觉可观察性
# - 从 2-4 组样例中, 人会看到: 杯口外侧出现与杯内内容同色的直线, 而杯内从稀疏散点变成整块实心.
# - “线在开口外”与“杯内被填满”两条线索互相支撑, 不依赖颜色编号、遍历顺序或不可视属性.
#
# f) 可视化整洁度
# - 输出中同一区域只承担一种角色: bbox 内是框架+填充, bbox 外的 mouth-side 半直线是弹射线.
# - stray pixel 仍限制为孤立点, 且不得落在任一路径上或贴近已有非背景像素, 避免与弹射线混淆.
#
# g) 变化的可推断性
# - 没有新颜色映射. 每个杯仍只把自己 mouth 中的内容色传播到两个位置: 填满 mouth 与杯口外直线.
# - 内容色种类受原题调色板约束, 结构线索足以让人把“哪种颜色属于哪个杯”与“为什么喷到哪里”对应起来.
#
# 9) 新的 solver 假设与 generator 保证
# - solver 假设 spike 投影到对侧 bbox 边后, 正好落在该杯 mouth 的中心轴上; generator 只生成 canonical
#   funnel/Y 家族并用旋转得到四向杯, 保证该投影成立.
# - solver 假设 mouth 外侧至少有 1 格 runway; generator 的放置函数拒绝没有 mouth-side path 的位置.
# - solver 假设输入内容在 mouth 内仍是稀疏散点而非额外“伪杯”; generator 限制每杯输入内容不满杯, 并用
#   8 连通分量上限过滤过于团块的散点.

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


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> set[tuple[int, int]]:
    """4-connected flood inside bbox, seeded from one edge and blocked by border cells."""
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
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _project_spike_to_mouth(
    bbox: tuple[int, int, int, int], spike: tuple[int, int], tail_dir: str
) -> tuple[tuple[int, int], str]:
    """Project the spike to the opposite bbox edge and return the mouth-outward direction."""
    r0, c0, r1, c1 = bbox
    sr, sc = spike
    if tail_dir == "U":
        return (r1, sc), "D"
    if tail_dir == "D":
        return (r0, sc), "U"
    if tail_dir == "L":
        return (sr, c1), "R"
    return (sr, c0), "L"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - detect the cup and its mouth exactly as in the original task;
    - count the sparse content pixels currently inside the mouth;
    - fill the entire mouth interior with that content color;
    - project the spike to the mouth side and eject a line outward from the cup opening.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

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
                (d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners
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
                passing.append((d, spike, content_col, mouth, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, mouth, content_cells = passing[0]

            for r, c in mouth:
                out[r, c] = content_col

            mouth_edge, path_dir = _project_spike_to_mouth(bbox, spike, tail_dir)
            dr, dc = DIR4[path_dir]
            sr, sc = mouth_edge
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
CONTENT_SCATTER_RETRIES = 80
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(
    L: int, R: int, hL: int, hR: int, color: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical funnel/Y with tail pointing up."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1 : L + hL + 1, 0] = color
    if hR:
        patch[R + 1 : R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(
    patch: np.ndarray, border_color: int, opposite_side: str
) -> list[tuple[int, int]]:
    """Flood-fill the cup interior from the open side."""
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
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical tail-up cup to the target tail direction."""
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
    """Sample canonical cup parameters that still fit the original bbox bounds."""
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


def _path_cells_from_mouth(
    H: int, W: int, bbox: tuple[int, int, int, int], spike_abs: tuple[int, int], tail_dir: str
) -> list[tuple[int, int]]:
    """All cells on the mouth-outward half-line, starting just outside the cup."""
    mouth_edge, path_dir = _project_spike_to_mouth(bbox, spike_abs, tail_dir)
    dr, dc = DIR4[path_dir]
    rr, cc = mouth_edge[0] + dr, mouth_edge[1] + dc
    path_cells = []
    while 0 <= rr < H and 0 <= cc < W:
        path_cells.append((rr, cc))
        rr += dr
        cc += dc
    return path_cells


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup so its bbox stays separated and its mouth-side runway stays clear."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        spike_abs = (r0 + spike[0], c0 + spike[1])
        path_cells = _path_cells_from_mouth(H, W, bbox, spike_abs, direction)
        if not path_cells:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, within the perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0 : r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0 : r1 + 1, c1 + 1 :] = True
    elif tail_dir == "D":
        mask[:r0, c0 : c1 + 1] = True
    else:
        mask[r1 + 1 :, c0 : c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> None:
    """Optionally place one isolated mouth-side stray pixel that survives unchanged."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup_meta in cups_meta:
        br0, bc0, br1, bc1 = cup_meta["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in cup_meta["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _max_8conn_component_size(cells: list[tuple[int, int]]) -> int:
    """Largest 8-connected component size among selected absolute coordinates."""
    remaining = set(cells)
    best = 0
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 1
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nxt = (r + dr, c + dc)
                    if nxt in remaining:
                        remaining.remove(nxt)
                        stack.append(nxt)
                        size += 1
        if size > best:
            best = size
    return best


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build, place, and commit one cup for the filled-mouth + mouth-side-ejection target."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    max_count = min(CONTENT_COUNT_MAX, len(mouth) - 1)
    if max_count < 1:
        return None
    content_count = int(rng.integers(1, max_count + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True

    mouth_abs = [(r0 + r, c0 + c) for r, c in mouth]
    for r, c in mouth_abs:
        canvas_out[r, c] = content_color
    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": mouth_abs,
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter sparse input content inside each mouth without forming large pseudo-components."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) <= count:
            return False
        selected = None
        for _ in range(CONTENT_SCATTER_RETRIES):
            idxs = rng.choice(len(mouth_abs), size=count, replace=False)
            trial = [mouth_abs[int(i)] for i in idxs]
            if _max_8conn_component_size(trial) < 5:
                selected = trial
                break
        if selected is None:
            return False
        for r, c in selected:
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Construct one combo-variant instance and verify it with the solver oracle."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if len(palette) < num_cups * 2:
        return None
    border_palette = [int(c) for c in palette[:num_cups]]
    content_pool = [int(c) for c in palette[num_cups:]]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for border_color in border_palette:
        content_color = int(rng.choice(content_pool))
        meta = _add_cup(
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, canvas_out):
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
