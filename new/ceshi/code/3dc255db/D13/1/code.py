# Variant of 3dc255db
# Dimension: D13 — 弹射线宽度 (trajectory)
# Change: 单像素宽度的射线(1px宽)... → 宽度等于杯口宽度
#
# Design analysis for this variant
# --------------------------------
# spec:
#   variant_core:
#     old_rule: 每个被清空的内容像素只在尾刺方向上贡献 1 个像素，因此输出是一条 1px 宽直线。
#     new_rule: 每个被清空的内容像素改为贡献 1 个“带状截面”。
#       带状截面的宽度 = 杯口开口在 mouth 边上的跨度，沿尾刺方向堆叠后形成矩形轨迹。
#   concrete_geometry:
#     - 先像原题一样用尾刺方向 d 识别杯子，并从 opposite(d) 这一侧 flood 得到 mouth。
#     - mouth 在开口边上的那些 seed cells 就是“杯口开口”。
#     - 若尾刺为 U/D，则杯口宽度是这些开口 cells 的列跨度；输出轨迹每前进 1 行，整段列跨度都被填满。
#     - 若尾刺为 L/R，则杯口宽度是这些开口 cells 的行跨度；输出轨迹每前进 1 列，整段行跨度都被填满。
#     - 轨迹从尖刺外侧第一层开始，因此视觉上像从尖刺“扇形展开”后继续保持固定宽度。
#   behavior_trace:
#     - 原逻辑: content_count=3, tail=R 时，输出只会在 spike 右边放 3 个连续单元。
#     - 新逻辑: 若同一杯子的 mouth opening 高度为 4，则 spike 右边第 1/2/3 列都要填满这 4 行，
#       结果是 4x3 的矩形带，而不是 1x3 的线。
#     - 差异不在“识别哪个杯子/哪个颜色”，而在“如何把 content_count 映射为外部轨迹”。
#   complexity_calibration:
#     - 为了让 D13 可观察，生成器强制 mouth_width >= 2，避免大量样本退化回原题的 1px 射线。
#     - 仍保留 1-3 个杯子、方向 U/D/L/R、内容散点、stray pixel 等原题结构，避免把题目简化成单对象模板。
#     - 轨迹宽度只有一个离散控制量: mouth opening span；不会再引入额外颜色映射或多级规则。
#   main_line_by_line_audit:
#     - 原 `for t in range(1, len(content_cells)+1)` 依赖“每一步只写 1 个 cell”。
#       在变体中该前提失效，必须改成“每一步写一个 band”。
#     - 原实现完全不测 mouth opening，只用 `content_cells` 的数量定长度。
#       变体必须额外从 `mouth` 中抽取开口边界 cells 并计算跨度。
#     - 原实现无条件 `out[rr, cc] = content_col` 在 1px 轨迹下无害；
#       变体若仍逐点写单 cell，会直接丢失宽度信息，因此需要遍历 band 内全部 cells。
#   placement_legality:
#     - 生成器的“clear path”从单条线升级为“到边界的整块矩形带都必须为空”。
#     - `occ_path` 记录整条厚轨迹的所有 cells，保证不同杯子的输出矩形互不相撞，也不会穿过别的 cup bbox。
#     - stray pixel 仍只能放在某个杯子的 mouth side，但也禁止落到厚轨迹保留区里。
#   visual_observability:
#     - 人类可见的控制变量是杯口开口宽度；不同杯子的轨迹宽度会随各自杯口同步变化。
#     - 即使颜色重复，宽轨迹的位置和厚度都由杯体几何决定，而不是由颜色编号决定。
#   cleanliness:
#     - 轨迹始终是单色矩形带，不和杯体重叠，视觉层次仍保持“边框 / 内部内容 / 外部弹射带”三层。
#   new_solver_assumptions_and_generator_guarantees:
#     - 假设 mouth opening 在开口边上形成单一连续区间。生成器使用单口 funnel/Y 形边框并保证这一点。
#     - 假设每个可解杯子的 mouth opening 宽度 >= 2。生成器显式筛掉宽度 1 的形状。
#     - 假设厚轨迹矩形不会与其他对象重叠。生成器用整块 `occ_path` 保证。

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _flood_from_bbox_side(border_set: set, bbox: tuple, side: str) -> set:
    """4-conn flood within bbox, seeded from non-border cells on the given bbox edge."""
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


def _opening_edge_cells(mouth: set[tuple[int, int]] | list[tuple[int, int]], bbox: tuple, side: str) -> list[tuple[int, int]]:
    """Cells of the mouth that lie directly on the cup's opening edge."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        return sorted((r, c) for r, c in mouth if r == r0)
    if side == "D":
        return sorted((r, c) for r, c in mouth if r == r1)
    if side == "L":
        return sorted((r, c) for r, c in mouth if c == c0)
    return sorted((r, c) for r, c in mouth if c == c1)


def _opening_span(edge_cells: list[tuple[int, int]], side: str) -> tuple[int, int] | None:
    """Inclusive span of the opening on the axis perpendicular to the tail direction."""
    if not edge_cells:
        return None
    if side in ("U", "D"):
        vals = [c for _, c in edge_cells]
    else:
        vals = [r for r, _ in edge_cells]
    return min(vals), max(vals)


def _trajectory_bands(
    spike: tuple[int, int], direction: str, span: tuple[int, int], H: int, W: int,
) -> list[list[tuple[int, int]]]:
    """Return consecutive width-wise bands from the spike outward until the grid edge."""
    sr, sc = spike
    dr, dc = DIR4[direction]
    lo, hi = span
    bands: list[list[tuple[int, int]]] = []
    t = 1
    while True:
        rr = sr + dr * t
        cc = sc + dc * t
        if not (0 <= rr < H and 0 <= cc < W):
            break
        if direction in ("U", "D"):
            band = [(rr, x) for x in range(lo, hi + 1)]
        else:
            band = [(x, cc) for x in range(lo, hi + 1)]
        bands.append(band)
        t += 1
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from cup-shaped frames outward along each cup's tail direction.

    Variant D13:
    - The outbound trajectory is no longer 1px wide.
    - Its width equals the cup-mouth opening width on the side opposite the spike.
    - Each erased content pixel contributes one width-wise band, so the output is a rectangle.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

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
                mouth_side = OPPOSITE[d]
                mouth = _flood_from_bbox_side(border_set, bbox, mouth_side)
                edge_cells = _opening_edge_cells(mouth, bbox, mouth_side)
                span = _opening_span(edge_cells, mouth_side)
                if span is None:
                    continue

                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue

                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, span, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, span, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            for band in _trajectory_bands(spike, tail_dir, span, H, W)[: len(content_cells)]:
                for r, c in band:
                    out[r, c] = content_col

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
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

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
        g[L + 1:L + hL + 1, 0] = color
    if hR:
        g[R + 1:R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the interior of the cup patch from its open side."""
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
    """Pick canonical cup parameters that fit the bbox bounds after rotation."""
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


def _trajectory_bands_from_opening(
    spike_abs: tuple[int, int], direction: str, opening_abs: list[tuple[int, int]], H: int, W: int,
) -> list[list[tuple[int, int]]]:
    """Absolute-thickness trajectory bands produced by the cup opening."""
    side = OPPOSITE[direction]
    span = _opening_span(opening_abs, side)
    if span is None:
        return []
    return _trajectory_bands(spike_abs, direction, span, H, W)


def _flatten_bands(bands: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for band in bands for cell in band]


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    opening: list[tuple[int, int]],
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a valid placement with a clear thick trajectory from spike to edge."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        opening_abs = [(r0 + r, c0 + c) for (r, c) in opening]
        path_bands = _trajectory_bands_from_opening((sr_abs, sc_abs), direction, opening_abs, H, W)
        if not path_bands:
            continue

        path_cells = _flatten_bands(path_bands)
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        eject_cells = _flatten_bands(path_bands[:content_count])
        return r0, c0, path_cells, eject_cells
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of bbox, restricted to the cup's perpendicular extent."""
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
    """Optionally place one isolated pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for (r, c) in m["path_cells"]:
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
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, reserve its thick trajectory, and commit it to the canvases."""
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

    open_side = OPPOSITE[direction]
    opening = _opening_edge_cells(mouth, (0, 0, ph - 1, pw - 1), open_side)
    span = _opening_span(opening, open_side)
    if span is None or (span[1] - span[0] + 1) < 2:
        return None

    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, opening, content_count, occ_cup, occ_path,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for (r, c) in ejection_cells:
        canvas_out[r, c] = content_color
    for (r, c) in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "mouth_width": span[1] - span[0] + 1,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's content pixels at random inside its mouth interior."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single end-to-end generation attempt."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if len(palette) < 2:
            return None
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        palette.append(content_color)

        meta = _add_cup(
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path,
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
