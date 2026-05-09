# Variant of 3dc255db
# Dimension: D6 — 变换操作 (transform)
# Change: erase — 弹射后杯口内的内容像素被擦除(变为背景色) → 杯形填充
#
# Design analysis for this variant
# --------------------------------
# source_task: None
#
# a) 具体化与行为追踪
# - 保留原题对象: 每个杯子仍然是单色边框的漏斗/Y 形框架, 某一侧 bbox 边上有唯一尖刺像素,
#   另一侧是开口. 输入中的内容像素仍然只出现在杯口内部, 且每杯只有一种内容色.
# - 原题输出: 统计杯内内容像素个数 n, 沿尖刺方向画长度 n 的直线, 同时把杯内原有内容擦成背景.
# - 变体输出: 仍然统计输入里内容像素个数 n 并沿尖刺方向画长度 n 的直线, 但杯口内部不再清空,
#   而是整块被该内容色填满. 这里“杯口内部”指从开口侧 bbox 边做 4 连通 flood, 被边框阻挡后
#   到达的整块 mouth 区域, 不只是原先散落的几个内容点.
# - 小例子:
#   输入 mouth 有 5 个内部格, 其中 2 个是红色内容点.
#   原题: 输出中 mouth 的 2 个红点被擦除为背景, 尖刺外画 2 个红点.
#   变体: 输出中同样尖刺外画 2 个红点, 但 mouth 的 5 个格全部变红.
#   差异发生在 solver 的“处理 mouth 内容”环节, generator 也必须把目标输出改为“满杯口”.
#
# b) 复杂度校准
# - 不改杯子识别机制, 仍保留 1-3 个杯、四个方向、非对称臂长、随机内容数与可选 stray pixel.
# - 视觉差异已经很强: 原题输出会留下空心杯口, 变体会留下实心杯口.
# - 为避免退化到“输入杯口本来就已经满了, 填充行为不明显”, 生成器强制每个杯至少留 1 个
#   未填格在输入中, 即 content_count < len(mouth).
#
# c) 隐式假设审查
# - `_flood_from_bbox_side(...)` 依赖: mouth 可由与尾巴相对的 bbox 边进入, 边框会阻挡 flood.
#   变体下仍成立, 因为几何结构没变.
# - `cnt[v] += 1` / `content_col = cnt.most_common(1)[0][0]` 依赖: mouth 内只有一个真正的内容色.
#   变体仍成立; generator 继续保证每杯 mouth 中只散布一种内容色.
# - 原代码 `for r, c in content_cells: out[r, c] = bg` 依赖旧规则“只擦除原有内容格”.
#   这是 D6 需要替换的核心. 新规则不能只改 `content_cells`, 必须对整个 `mouth` 着色,
#   否则会漏掉原先是背景的内部格.
# - 仍以输入中的 `len(content_cells)` 决定弹射长度; 变体没有把“填充后 mouth 大小”用于计数,
#   否则会改成另一维度的任务.
#
# d) 放置合法性
# - 杯子的 bbox 仍需彼此至少隔 1 格, 且弹射路径在输入中保持为空白.
# - 输出中的新 mouth 填充只发生在各自 bbox 内部, 不会越界也不会与其他杯/路径重叠.
# - generator 继续 output-first: 先构造“满杯口 + 弹射线”的期望输出, 再随机散布输入内容,
#   最后用 `main(input)` 回验.
#
# e) 视觉可观察性
# - 解题者从少量样例即可看出: 每个杯都把少量内容“喷”到尾部外面, 同时杯子内部从稀疏点变成
#   整块实心区域. 这是稳定的、方向无关的规则, 不依赖颜色编号或排序.
#
# f) 可视化整洁度
# - 只在原本 hollow 的 mouth 区域做实心填充, 不增加新类型标记.
# - 仍保留路径留空、对象间隔和 stray pixel 的隔离要求, 避免多层覆盖造成视觉混乱.
#
# g) 颜色变化可推断性
# - 没有新增颜色映射. 只使用“每杯原有内容色”去填满对应 mouth.
# - 颜色种类与原题基本一致, 结构线索直接给出“哪种颜色填哪个杯”.
#
# 9) 新的 solver 假设与 generator 保证
# - solver 现在假设: mouth 区域内若出现非边框非背景色, 主色就是该杯内容色.
#   generator 保证每杯 mouth 中只使用一种内容色.
# - solver 现在假设: `mouth` 是希望被整体填满的区域, 而不是若干离散点.
#   generator 的杯型构造仍保证 mouth 是单个可 flood 到的连通区域.
# - solver 处理顺序仍按颜色/连通块遍历, 但所有修改写到 `out` 上且不会影响对 `g` 的识别.
#   generator 保证输入中不同杯的 mouth / 路径 / bbox 不重叠, 所以顺序无关.

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


def _flood_from_bbox_side(border_set: set, bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from non-border cells on the given bbox edge,
    stopping at border cells. Returns the set of reached cells (includes seeds)."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:  # "R"
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - detect each cup exactly as in the original task;
    - count the input content pixels inside its mouth;
    - eject a line of that many pixels from the spike along the tail direction;
    - fill the entire mouth interior with the content color instead of erasing it.
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

            dr, dc = DIR4[tail_dir]
            sr, sc = spike
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


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP (spike at top, row 0)."""
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
    patch: np.ndarray, border_color: int, opposite_side: str
) -> list[tuple[int, int]]:
    """Flood-fill the cup interior from the open side of the bbox."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:  # "R"
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
    """Rotate a canonical (tail=U) patch, its spike and mouth coords to the target tail direction."""
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
    """Pick canonical cup parameters clamped to the original bbox bounds."""
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
    """Place a cup so its bbox stays separated and its ejection path stays clear."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((rr, cc))
            rr += dr
            cc += dc
        if not path_cells:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, limited to the perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0 : r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0 : r1 + 1, c1 + 1 :] = True
    elif tail_dir == "D":
        mask[:r0, c0 : c1 + 1] = True
    else:  # "U"
        mask[r1 + 1 :, c0 : c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> None:
    """Optionally place a mouth-side isolated pixel that survives unchanged."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _max_8conn_component_size(cells: list[tuple[int, int]]) -> int:
    """Largest 8-connected component size among the selected absolute coords."""
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
    """Build one cup and commit the variant's output-first target state."""
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
    """Scatter each cup's sparse input content within its mouth.

    The selected pixels must stay visually sparse enough that no same-color input cluster
    becomes a size>=5 component, because the solver treats such components as potential cups.
    """
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
    """Construct a variant instance and verify it with the solver oracle."""
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
        # Keep border colors unique and disjoint from all content colors. This avoids
        # accidental same-color cross-cup patterns that can look like extra border components.
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
