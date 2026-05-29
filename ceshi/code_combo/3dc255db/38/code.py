# Combo variant of 3dc255db (index 38)
# combo_spec: D3/0+D5/2+D6/0
# Name: 多色内容+周期循环+内容变色保留
#
# Design analysis
# ===============
# fusion_spec:
#   concretization_and_trace:
#     - 保留原题的杯状框架检测、尾刺方向检测和 mouth flood 逻辑；变化只发生在
#       “如何读取杯内内容” 与 “如何写出输出” 两步。
#     - 输入中每个杯子的 mouth 内有 2-3 种内容色。这些颜色不是任意混在一起，
#       而是沿 mouth 的可见阅读轴占据连续切片：
#         tail 为 U/D 时按列从左到右切片；
#         tail 为 L/R 时按行从上到下切片。
#     - 具体例子：某个 tail=R 的杯子，border=7，mouth 中上半部分有 2 个红(2)，
#       下半部分有 3 个青(8)。原题会把 5 个内容全部擦掉，并在右侧画出
#       [2,2,2,2,2]；D3 单变体会画 [2,2,8,8,8]；本融合变体改为：
#         1. 这 5 个 mouth 内容位置不消失，而是都变成边框色 7；
#         2. 右侧弹射线长度仍为 5，但颜色序列按切片顺序循环，得到 [2,8,2,8,2]。
#     - 若 mouth 中有三种颜色且顺序为 [4,1,3]、总内容数为 7，则输出射线是
#       [4,1,3,4,1,3,4]，同时 7 个原始内容位置都改成 border 色。
#   complexity_calibration:
#     - 每杯只用 2-3 种内容色，避免颜色爆炸。
#     - 每种颜色只占一个连续切片，避免 solver 需要处理“同色多段往返出现”的歧义。
#     - 生成器强制 total_content > num_colors，这样输出一定出现“周期重复”，
#       不会退化成只把颜色顺序读一遍。
#   implicit_assumption_audit_main:
#     - 原代码 `Counter(...).most_common(1)` 假设 mouth 内只有单色内容；融合后失效，
#       必须改为按颜色分组并保留全部颜色。
#     - 原代码 `content_cells = [...] == content_col` 只收集/擦除一种颜色；融合后必须
#       收集 mouth 内全部非 bg、非 border 的内容像素。
#     - 原代码 `out[r,c] = bg` 假设内容会被擦除；融合后改为 `out[r,c] = border_col`。
#     - 原代码 `out[rr,cc] = content_col` 假设射线单色；融合后改为“按 mouth 切片顺序
#       形成颜色周期，再重复到 total_content 长度”。
#     - 颜色组排序不能依赖颜色编号。这里只把颜色值当作最后的平局破除器；
#       生成器保证不同组的最早投影不同，因此真正决定顺序的是可见位置。
#   placement_legality:
#     - 继续使用 output-first：先放杯框和预期射线，再回填输入内容，最后用
#       `main(input) == output` 反验。
#     - 每个杯的 bbox 仍需与其他杯保持至少 1-cell gap；所有射线路径仍必须与其他杯
#       bbox 和其他射线路径完全分离。
#     - 为了完整展示周期规律，生成器要求从 spike 到边界的可用路径长度至少覆盖
#       全部内容数，不依赖原题的“越界截断”。
#   visual_observability:
#     - 输入中可见“同一杯内有多色散点，而且这些颜色沿 mouth 方向有稳定的区域顺序”。
#     - 输出中同一批位置改成 border 色，明显表示“原位置被保留但变色”；
#       同时射线出现多色循环，明显不同于原题的单色直线和 D3 的分块直线。
#     - 人类可从 2-4 个样例中观察到：
#         1. 哪些像素被弹射的是 cup mouth 内的内容；
#         2. 这些像素原位置没有消失，而是变成 border 色；
#         3. 射线颜色循环顺序来自 mouth 中颜色区域的顺序。
#   visual_cleanliness:
#     - 内容色池限制为小集合并允许跨杯复用，减少整图噪声。
#     - stray pixel 仍最多 1 个，且永不进入 cup bbox，也不占用任何射线路径。
#     - 不让 border 色参与周期射线，避免输出上 border 框与射线视觉粘连。
#   color_inference:
#     - 颜色变化不是任意映射。border 色只承担“原位置标记”功能；射线只使用输入里
#       已经出现过的内容色，并按 mouth 切片顺序循环。
#   new_solver_assumptions_and_generator_guarantees:
#     - solver 假设同一杯内每种内容色只对应一个连续 mouth 切片；生成器按切片分区放置，
#       保证这一点成立。
#     - solver 假设 actionable cup 至少有 2 种内容色且总内容数大于颜色数；
#       生成器显式构造并验证这些条件。
#     - solver 假设 stray pixel 不会污染任何 cup mouth；生成器只在 bbox 外、mouth 外侧
#       区域放 stray，并与现有非背景像素保持 1-cell 间隔。

from __future__ import annotations

from collections import defaultdict
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
STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood within bbox from one bbox side, stopping at border cells."""
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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _mouth_projection(cell: tuple[int, int], tail_dir: str) -> int:
    """Project a mouth cell onto the human-visible ordering axis."""
    r, c = cell
    return int(c if tail_dir in ("U", "D") else r)


def _content_groups_in_mouth(
    g: np.ndarray,
    mouth: set[tuple[int, int]],
    bg: int,
    border_col: int,
    tail_dir: str,
) -> list[dict]:
    """Group mouth content by color, ordered by mouth slice position."""
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in mouth:
        v = int(g[r, c])
        if v != bg and v != border_col:
            by_color[v].append((int(r), int(c)))
    if len(by_color) < 2:
        return []

    def order_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[float, float, int]:
        color, cells = item
        projections = [_mouth_projection(cell, tail_dir) for cell in cells]
        other_axis = [cell[0] if tail_dir in ("U", "D") else cell[1] for cell in cells]
        return min(projections), sum(projections) / len(projections), min(other_axis) * 100 + color

    ordered = []
    for color, cells in sorted(by_color.items(), key=order_key):
        ordered.append(
            {
                "color": int(color),
                "cells": sorted(cells, key=lambda rc: (_mouth_projection(rc, tail_dir), rc[0], rc[1])),
            }
        )
    return ordered


def _cyclic_eject_sequence(colors: list[int], length: int) -> list[int]:
    """Repeat the ordered colors cyclically until the requested length is reached."""
    assert colors and length >= 0
    return [int(colors[i % len(colors)]) for i in range(length)]


def _detect_cups(g: np.ndarray) -> list[dict]:
    """Detect actionable cups from the input grid using the fused task assumptions."""
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
                (direction, cells[0])
                for direction, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                groups = _content_groups_in_mouth(g, mouth, bg, int(border_col), tail_dir)
                if not groups:
                    continue
                content_count = sum(len(group["cells"]) for group in groups)
                if content_count <= len(groups):
                    continue
                passing.append(
                    {
                        "tail_dir": tail_dir,
                        "spike": spike,
                        "border_col": int(border_col),
                        "groups": groups,
                        "mouth": mouth,
                        "content_count": int(content_count),
                    }
                )

            if len(passing) == 1:
                cups.append(passing[0])
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule for D3/0 + D5/2 + D6/0.

    For each actionable cup:
    - detect the mouth-side ordering of its 2-3 content colors,
    - recolor every original content pixel to the cup's border color,
    - eject a straight ray whose colors cycle through that ordered content-color list,
      one output cell per input content pixel.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for cup in _detect_cups(g):
        border_col = int(cup["border_col"])
        tail_dir = str(cup["tail_dir"])
        sr, sc = cup["spike"]
        groups = cup["groups"]

        content_cells = [cell for group in groups for cell in group["cells"]]
        cycle_colors = [int(group["color"]) for group in groups]

        for r, c in content_cells:
            out[r, c] = border_col

        dr, dc = DIR4[tail_dir]
        for t, color in enumerate(_cyclic_eject_sequence(cycle_colors, len(content_cells)), start=1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = int(color)

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
CONTENT_GROUP_COUNT_RANGE = (2, 3)
GLOBAL_CONTENT_POOL_SIZE = 4

MIN_VISIBLE_EJECT_LONG = 4
MIN_TOTAL_VISIBLE_EJECT = 5

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP."""
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


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the cup interior within the patch."""
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
    """Rotate a canonical UP-facing cup patch into the target tail direction."""
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
    """Pick canonical cup parameters that stay within the original bbox bounds after rotation."""
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


def _split_sorted_values(values: list[int], groups: int, rng: np.random.Generator) -> list[list[int]] | None:
    """Split sorted projection values into contiguous non-empty groups."""
    if len(values) < groups:
        return None
    if groups == 1:
        return [list(values)]
    cut_points = sorted(int(x) for x in rng.choice(np.arange(1, len(values)), size=groups - 1, replace=False))
    result: list[list[int]] = []
    start = 0
    for stop in [*cut_points, len(values)]:
        result.append(list(values[start:stop]))
        start = stop
    return result


def _choose_content_groups(
    rng: np.random.Generator,
    mouth: list[tuple[int, int]],
    direction: str,
    available_colors: list[int],
) -> list[dict] | None:
    """Choose 2-3 ordered content-color groups occupying disjoint mouth slices."""
    unique_proj = sorted({_mouth_projection(cell, direction) for cell in mouth})
    max_total = min(CONTENT_COUNT_MAX, len(mouth))
    max_groups = min(CONTENT_GROUP_COUNT_RANGE[1], len(unique_proj), len(available_colors))
    feasible_group_counts = [
        n
        for n in range(CONTENT_GROUP_COUNT_RANGE[0], max_groups + 1)
        if max_total >= n + 1
    ]
    if not feasible_group_counts:
        return None

    num_groups = int(rng.choice(np.array(feasible_group_counts, dtype=int)))
    proj_groups = _split_sorted_values(unique_proj, num_groups, rng)
    if proj_groups is None:
        return None

    zones: list[list[tuple[int, int]]] = []
    for proj_values in proj_groups:
        proj_set = set(proj_values)
        zone = [cell for cell in mouth if _mouth_projection(cell, direction) in proj_set]
        zone = sorted(zone, key=lambda rc: (_mouth_projection(rc, direction), rc[0], rc[1]))
        if not zone:
            return None
        zones.append(zone)

    total_count = int(rng.integers(num_groups + 1, max_total + 1))
    colors = [int(v) for v in rng.choice(np.array(available_colors, dtype=int), size=num_groups, replace=False)]
    rng.shuffle(colors)

    counts = [1] * num_groups
    capacities = [len(zone) - 1 for zone in zones]
    remaining = total_count - num_groups
    while remaining > 0:
        choices = [i for i, cap in enumerate(capacities) if cap > 0]
        if not choices:
            return None
        idx = int(rng.choice(np.array(choices, dtype=int)))
        counts[idx] += 1
        capacities[idx] -= 1
        remaining -= 1

    return [
        {"color": int(colors[i]), "count": int(counts[i]), "zone": zones[i]}
        for i in range(num_groups)
    ]


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    sequence_len: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup with a clear tail path long enough to show the full cyclic sequence."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=STRUCT8)

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
        if len(path_cells) < sequence_len:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells, path_cells[:sequence_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Mask of cells on the mouth side of ``bbox`` restricted to the cup's perpendicular extent."""
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
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    """Optionally place one isolated stray pixel on the mouth side of a random cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    content_colors = [group["color"] for group in meta["content_groups"]]
    if rng.random() < STRAY_USE_BORDER_PROB:
        stray_color = int(meta["border_color"])
    else:
        stray_color = int(rng.choice(np.array(content_colors, dtype=int)))

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=STRUCT8)
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in cup["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_palette: list[int],
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, choose ordered multicolor content groups, and commit its output skeleton."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if not mouth_canon:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    available_content_colors = [c for c in content_palette if c != border_color]
    content_groups = _choose_content_groups(rng, mouth, direction, available_content_colors)
    if content_groups is None:
        return None

    total_count = sum(group["count"] for group in content_groups)
    cycle_order = [int(group["color"]) for group in content_groups]
    eject_sequence = _cyclic_eject_sequence(cycle_order, total_count)

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, len(eject_sequence), occ_cup, occ_path
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    ph, pw = patch.shape
    patch_mask = patch != 0
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True
    for (r, c), color in zip(ejection_cells, eject_sequence):
        canvas_out[r, c] = int(color)
    for r, c in path_cells:
        occ_path[r, c] = True

    content_groups_abs = [
        {
            "color": int(group["color"]),
            "count": int(group["count"]),
            "zone_abs": [(r0 + r, c0 + c) for (r, c) in group["zone"]],
        }
        for group in content_groups
    ]

    return {
        "border_color": int(border_color),
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "content_groups": content_groups_abs,
        "path_cells": path_cells,
        "visible_eject_len": len(ejection_cells),
        "repeat_margin": int(total_count - len(cycle_order)),
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter input content inside each mouth slice and mark those source cells with border color in output."""
    for meta in cups_meta:
        border_color = int(meta["border_color"])
        for group in meta["content_groups"]:
            zone_abs = group["zone_abs"]
            count = int(group["count"])
            if len(zone_abs) < count:
                return False
            idxs = rng.choice(len(zone_abs), size=count, replace=False)
            for i in idxs:
                r, c = zone_abs[int(i)]
                canvas_in[r, c] = int(group["color"])
                canvas_out[r, c] = border_color
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single generation attempt with output-first verification."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    border_palette = list(range(1, 10))
    rng.shuffle(border_palette)
    content_palette = [int(v) for v in rng.choice(np.arange(1, 10), size=GLOBAL_CONTENT_POOL_SIZE, replace=False)]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if not border_palette:
            return None
        border_color = int(border_palette.pop())
        meta = _add_cup(
            rng, H, W, border_color, content_palette, canvas_in, canvas_out, occ_cup, occ_path
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None

    visible_lengths = [int(meta["visible_eject_len"]) for meta in cups_meta]
    if not visible_lengths:
        return None
    if max(visible_lengths) < MIN_VISIBLE_EJECT_LONG:
        return None
    if sum(visible_lengths) < max(MIN_TOTAL_VISIBLE_EJECT, 3 * len(cups_meta)):
        return None

    if not _scatter_content(rng, canvas_in, canvas_out, cups_meta):
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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
