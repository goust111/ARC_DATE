# Variant of 3dc255db
# Dimension: D3 — 源对象类型 (src)
# Change: 杯口内部散布的单色内容像素(1-7个)... → 多色内容像素
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: 每个杯子 mouth_interior 中只有一种内容色；solver 只需找出现次数最多的单色并整段弹射。
#     variant: 每个杯子 mouth_interior 中有 2-3 种内容色；输出中的弹射线不再单色，而是按颜色分组串接。
#   concretization_and_trace:
#     input_structure:
#       - 杯框形状、尾刺识别、bbox/间距约束保持原题。
#       - 杯口内部仍是散布像素，但这些像素来自 2-3 种颜色。
#       - 为了让人眼可见，每种颜色只占据杯口横向阅读顺序中的一个切片：
#         tail 为 U/D 时按列从左到右切片；tail 为 L/R 时按行从上到下切片。
#       - 切片内部的像素仍随机散布，因此看上去仍是“散点内容”，不是规则条纹。
#     output_structure:
#       - 原逻辑: 若杯内有 [红, 红, 红] 三个像素，则尾巴方向得到 [红, 红, 红]。
#       - 新逻辑: 若杯内左侧切片是 2 个红色、右侧切片是 3 个蓝色，
#         则输出射线为 [红, 红, 蓝, 蓝, 蓝]；杯内原像素全部擦除。
#     affected_pipeline_stage:
#       - solver 中受影响的是 “从 mouth 提取内容 -> 决定输出射线颜色序列” 这一步。
#       - generator 中受影响的是 “内容散布回填” 和 “预期输出射线构造”。
#   complexity_calibration:
#     - 颜色组数限制为 2-3，避免颜色种类过多。
#     - 排序规则只使用杯口中的可见位置顺序，不使用颜色编号。
#     - 同一杯内每种颜色至少 1 像素，且尽量令总内容数 >= 3，使变体差异明显。
#   implicit_assumption_audit_main:
#     - 原代码 `cnt.most_common(1)` 假设 mouth 内只有一种有效内容色；变体下会丢失其余颜色，必须删除。
#     - 原代码 `content_cells = [...] == content_col` 只擦除一种颜色；变体下必须擦除 mouth 内全部内容色。
#     - 原代码 `for t in range(1, len(content_cells)+1): out[...] = content_col` 假设输出整段同色；
#       变体下必须先构造按组展开的颜色序列，再逐格写出，不能无条件复写为同一个颜色。
#     - 候选尾巴判定仍依赖 “从对侧 flood 到的 mouth 中存在内容像素”；这个前提在变体中仍成立。
#   placement_legality:
#     - 继续使用 output-first：先放杯框和预期射线，再回填输入内容，最后用 `main(input)` 反验。
#     - 为确保多色分组完整可见，生成器要求从 spike 到边界的 path 长度至少覆盖全部内容像素，
#       不再依赖原题允许的截断溢出。
#     - 杯 bbox 仍保持 1-cell gap，射线路径仍不可与其他杯或路径重叠。
#   visual_observability:
#     - 规则可由 2-4 组样例看出：不同颜色在杯口中占据从左到右/从上到下的不同区域，
#       输出则按同样顺序变成颜色块。
#     - 排序不依赖颜色值，因此人类可从位置线索推断。
#   visual_cleanliness:
#     - 不在一个杯里使用超过 3 种内容色。
#     - 使用共享的小内容色池，避免整张图颜色爆炸。
#     - stray pixel 仍最多 1 个，且不进入杯 bbox，不与射线路径重叠。
#   color_inference:
#     - 颜色变化不是任意映射；“颜色本身保持不变，只是按口部顺序分组排队弹射”。
#     - 生成器保证多色内容真实出现，因此不会退化回原始单色题。
#   new_solver_assumptions_and_generator_guarantees:
#     - solver 假设同一杯内不同颜色在 mouth-order 轴上的最早占位切片不同；
#       generator 通过把不同颜色放到不重叠的连续切片中来保证。
#     - solver 对同色像素的计数来自 mouth 内全部该色像素；
#       generator 保证 stray pixel 不会落入任何 cup bbox，因此不会污染计数。

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


def _flood_from_bbox_side(border_set: set, bbox: tuple, side: str) -> set:
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


def _mouth_projection(cell: tuple[int, int], tail_dir: str) -> int:
    r, c = cell
    return int(c if tail_dir in ("U", "D") else r)


def _content_groups_in_mouth(
    g: np.ndarray, mouth: set[tuple[int, int]], bg: int, border_col: int, tail_dir: str,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Return mouth content grouped by color and ordered by visible mouth reading order."""
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in mouth:
        v = int(g[r, c])
        if v != bg and v != border_col:
            by_color[v].append((int(r), int(c)))
    if not by_color:
        return []

    def order_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[float, float, int]:
        color, cells = item
        projections = [_mouth_projection(cell, tail_dir) for cell in cells]
        other_axis = [cell[0] if tail_dir in ("U", "D") else cell[1] for cell in cells]
        # `color` is only a deterministic tie-break; generator ensures ties should not matter.
        return min(projections), sum(projections) / len(projections), min(other_axis) * 100 + color

    return sorted(by_color.items(), key=order_key)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from cup-shaped frames outward along each cup's tail direction.

    Variant rule:
    - The mouth may contain 2-3 content colors.
    - Group colors by their occupied mouth slices:
      left->right for vertical tails, top->bottom for horizontal tails.
    - Eject one straight line made of contiguous color blocks in that order, with one pixel
      per input content pixel, then erase all input content from the cup interior.
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
                (d, cells[0]) for d, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                groups = _content_groups_in_mouth(g, mouth, bg, int(border_col), d)
                if not groups:
                    continue
                passing.append((d, spike, groups))

            if len(passing) != 1:
                continue
            tail_dir, spike, groups = passing[0]

            content_cells = [cell for _color, cells in groups for cell in cells]
            for r, c in content_cells:
                out[r, c] = bg

            eject_sequence = [color for color, cells in groups for _ in range(len(cells))]
            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            for t, color in enumerate(eject_sequence, start=1):
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

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
STRUCT8 = np.ones((3, 3), dtype=bool)


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
        g[L + 1:L + hL + 1, 0] = color
    if hR:
        g[R + 1:R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the interior of the cup patch from cells on the given bbox edge."""
    h, w = patch.shape
    border_mask = (patch == border_color)
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
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str,
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
    """Pick canonical cup parameters clamped to bbox bounds after rotation."""
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
    if len(values) < groups:
        return None
    if groups == 1:
        return [list(values)]
    cut_points = sorted(int(x) for x in rng.choice(np.arange(1, len(values)), size=groups - 1, replace=False))
    out: list[list[int]] = []
    start = 0
    for stop in [*cut_points, len(values)]:
        out.append(list(values[start:stop]))
        start = stop
    return out


def _choose_content_groups(
    rng: np.random.Generator, mouth: list[tuple[int, int]], direction: str, available_colors: list[int],
) -> list[dict] | None:
    """Create ordered color groups backed by disjoint mouth slices."""
    unique_proj = sorted({_mouth_projection(cell, direction) for cell in mouth})
    max_groups = min(CONTENT_GROUP_COUNT_RANGE[1], len(unique_proj), len(available_colors))
    if max_groups < CONTENT_GROUP_COUNT_RANGE[0]:
        return None
    num_groups = int(rng.integers(CONTENT_GROUP_COUNT_RANGE[0], max_groups + 1))

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

    max_total = min(CONTENT_COUNT_MAX, sum(len(zone) for zone in zones))
    min_total = max(num_groups, 3)
    if max_total < min_total:
        min_total = num_groups
    if max_total < num_groups:
        return None
    total_count = int(rng.integers(min_total, max_total + 1))

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
        {"color": colors[i], "count": counts[i], "zone": zones[i]}
        for i in range(num_groups)
    ]


def _try_place_cup(
    rng: np.random.Generator, H: int, W: int, patch: np.ndarray, spike: tuple[int, int],
    direction: str, sequence_len: int, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup with a clear tail path long enough for the full ejection sequence."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
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
    """Mask of cells on the mouth side of ``bbox`` restricted to the cup's transverse extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif tail_dir == "D":
        mask[:r0, c0:c1 + 1] = True
    else:
        mask[r1 + 1:, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict],
) -> None:
    """Optionally place one isolated stray pixel on the mouth side of a random cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    content_colors = [group["color"] for group in meta["content_groups"]]
    if rng.random() < STRAY_USE_BORDER_PROB:
        stray_color = meta["border_color"]
    else:
        stray_color = int(rng.choice(np.array(content_colors, dtype=int)))

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator, H: int, W: int, border_color: int, content_palette: list[int],
    canvas_in: np.ndarray, canvas_out: np.ndarray, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, choose ordered multicolor content groups, and commit it to the canvases."""
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
    eject_sequence = [group["color"] for group in content_groups for _ in range(group["count"])]

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, len(eject_sequence), occ_cup, occ_path,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    ph, pw = patch.shape
    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for (r, c), color in zip(ejection_cells, eject_sequence):
        canvas_out[r, c] = int(color)
    for r, c in path_cells:
        occ_path[r, c] = True

    content_groups_abs = [
        {
            "color": int(group["color"]),
            "count": int(group["count"]),
            "zone_abs": [(r0 + r, c0 + c) for r, c in group["zone"]],
        }
        for group in content_groups
    ]

    return {
        "border_color": border_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "content_groups": content_groups_abs,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each color group within its own mouth slice zone."""
    for meta in cups_meta:
        for group in meta["content_groups"]:
            zone_abs = group["zone_abs"]
            count = int(group["count"])
            if len(zone_abs) < count:
                return False
            idxs = rng.choice(len(zone_abs), size=count, replace=False)
            for i in idxs:
                r, c = zone_abs[int(i)]
                canvas_in[r, c] = int(group["color"])
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single generation attempt with output-first verification."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

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
            rng, H, W, border_color, content_palette, canvas_in, canvas_out, occ_cup, occ_path,
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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
