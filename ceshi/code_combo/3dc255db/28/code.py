# Combo variant of 3dc255db (index 28)
# combo_spec: D10/1+D13/1
# Name: 边框数量决定长度+杯口宽度射线
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题: 每个杯先找到尾刺方向, 擦除 mouth 内的内容色像素, 然后沿尾刺方向发射一条 1px 宽直线;
#   直线长度 = 内容像素数。
# - 本融合版: 仍先用同样的几何方式识别杯与 mouth, 但输出不再是细线。
#   现在要同时测两个几何量:
#   1. 长度 = 杯形边框像素总数。
#   2. 宽度 = 杯口 opening 在开口边上的宽度。
#   因此每个杯输出的是一个 content 色的矩形带, 尺寸为
#   `mouth_width x border_pixel_count`(或旋转后的等价方向)。
# - 具体小例子: 某个向右弹射的杯子, 边框共有 7 像素, 开口高度为 3。
#   原题会把 4 个内容点擦掉后向右画 4 个单元。
#   D13 单变体会画 3x4 的矩形带。
#   本融合版会画 3x7 的矩形带。
#   差异点是: 输出二维尺寸完全由杯体几何决定, 内容点只决定颜色与“是否激活该杯”。
# b) 复杂度校准
# - 若允许 opening 宽度经常为 1, 视觉上会退化回“只改长度”的 D10, 难以看出融合。
# - 若允许 content_count 常常等于 border_pixel_count, 则长度会退化回 D13 的单变体。
# - 因此生成器强制:
#   1. `mouth_width >= 2`, 保证厚轨迹肉眼可见。
#   2. `border_pixel_count - content_count >= 2`, 保证长度不是旧逻辑。
#   3. spike 到边界的可见 band 数 >= `border_pixel_count`, 避免被裁剪后看不出长度编码。
# - 仍保留 1-2 个杯、四个方向、随机散点内容和少量 stray pixel, 不把题目简化成单模板。
# c) 隐式假设审查
# - 原 `len(content_cells)` 同时承担“擦哪些内容”和“射多长”两件事; 融合后前者仍依赖真实内容格,
#   后者必须改成 `len(border_set)`。
# - 原 solver 不测 mouth opening, 默认宽度恒为 1; 融合后该前提失效, 必须显式从 mouth 开口边提取 span。
# - 原 `out[rr, cc] = content_col` 每步只写一个格; 融合后每步应写整条 band, 否则宽度信息丢失。
# - 候选尾刺识别仍依赖“某 bbox 边恰有一个非角 border 像素”; 该前提未变。
# - 口内内容色仍通过 mouth 内非 bg/非 border 的众数识别; 生成器继续保证每杯只有一种内容色。
# d) 放置合法性
# - cup bbox 之间保留至少 1 格间隙。
# - 厚轨迹的整条矩形走廊从 spike 一直到边界都必须为空, `occ_path` 记录完整 band 走廊,
#   防止后放对象或 stray 像素进入。
# - 生成器只接受能完整显示 `border_pixel_count` 个 band 的放置, 因而不会出现长度被截断的样本。
# e) 视觉可观察性
# - 解题者可直接看到: 杯口越宽, 输出矩形越厚; 杯边框越长, 输出矩形越长。
# - 两个控制量都来自同一个杯的几何, 不依赖颜色编号、扫描顺序或不可见元数据。
# - 内容像素刻意比边框像素少, 因而人眼能排除“长度由内容数量决定”的旧规则。
# f) 可视化整洁度
# - 输出区域始终是单色矩形带, 不与杯框重叠。
# - stray pixel 只放在某个杯的 mouth side, 且禁止接触 cup bbox 或厚轨迹保留区, 避免干扰判读。
# - 为了给长矩形留出空间, 杯数控制在 1-2 个, 并将边框像素数限制在中小档位。
# g) 变化的可推断性
# - 这里只有一种颜色行为: 输出仍沿用该杯内容色, 没有额外颜色映射。
# - 真正变化的是两个可见几何量到矩形尺寸的对应关系, 2-4 组样例足以让人通过对比口宽、
#   边框长度和输出矩形尺寸来归纳规则。
# h) main() 的新假设与 generator 保证
# - 假设 mouth opening 在开口边上形成单一连续区间; 生成器使用单口 funnel/Y 形边框并显式筛选。
# - 假设 `mouth_width >= 2`; 生成器过滤掉 width=1 的形状。
# - 假设 `border_pixel_count` 对应的全部 band 都在网格内可见; 生成器只接受完整可见的放置。
# - 假设每个可解杯的 thick path 不与其他对象冲突; 生成器用 `occ_path` 保证。

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 6
MIN_MOUTH_CELLS = 3
MIN_MOUTH_WIDTH = 2
MIN_BORDER_PIXELS = 6
MAX_BORDER_PIXELS = 8
MIN_BORDER_COUNT_GAP = 2

STRAY_PROB = 0.35
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400
CUP_BUILD_RETRIES = 120

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> set[tuple[int, int]]:
    """4-connected flood inside bbox from one side, blocked by border pixels."""
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


def _opening_edge_cells(
    mouth: set[tuple[int, int]] | list[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
    """Cells of the mouth that sit directly on the cup opening edge."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        return sorted((r, c) for r, c in mouth if r == r0)
    if side == "D":
        return sorted((r, c) for r, c in mouth if r == r1)
    if side == "L":
        return sorted((r, c) for r, c in mouth if c == c0)
    return sorted((r, c) for r, c in mouth if c == c1)


def _opening_span(edge_cells: list[tuple[int, int]], side: str) -> tuple[int, int] | None:
    """Inclusive span on the axis perpendicular to the tail direction."""
    if not edge_cells:
        return None
    if side in ("U", "D"):
        vals = [c for _, c in edge_cells]
    else:
        vals = [r for r, _ in edge_cells]
    return min(vals), max(vals)


def _trajectory_bands(
    spike: tuple[int, int], direction: str, span: tuple[int, int], H: int, W: int
) -> list[list[tuple[int, int]]]:
    """Consecutive width-wise bands from the spike outward until the grid edge."""
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


def _flatten_bands(bands: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for band in bands for cell in band]


def _detect_cups(grid: np.ndarray) -> tuple[int, list[dict]]:
    """Recover cups plus the geometry needed by the combo rule."""
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
                passing.append(
                    {
                        "tail_dir": d,
                        "spike": (int(spike[0]), int(spike[1])),
                        "mouth_side": mouth_side,
                        "mouth": sorted((int(r), int(c)) for r, c in mouth),
                        "opening": edge_cells,
                        "span": span,
                        "content_col": content_col,
                        "content_cells": sorted((int(r), int(c)) for r, c in content_cells),
                    }
                )

            if len(passing) != 1:
                continue

            cup = passing[0]
            cup.update(
                {
                    "border_color": int(border_col),
                    "bbox": bbox,
                    "border_set": border_set,
                    "border_pixels": len(border_set),
                    "mouth_width": cup["span"][1] - cup["span"][0] + 1,
                }
            )
            cups.append(cup)

    return bg, cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - identify each actionable cup as in the base task
    - erase the content-colored pixels inside the mouth
    - emit a rectangular band whose width is the mouth opening width
      and whose length is the number of border pixels of that cup
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, cups = _detect_cups(g)
    out = g.copy()

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

        bands = _trajectory_bands(cup["spike"], cup["tail_dir"], cup["span"], H, W)
        for band in bands[: cup["border_pixels"]]:
            for r, c in band:
                out[r, c] = cup["content_col"]

    return out


def _build_canonical_cup(
    L: int, R: int, hL: int, hR: int, color: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a funnel/Y cup whose tail points UP in canonical orientation."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1:L + hL + 1, 0] = color
    if hR:
        patch[R + 1:R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(
    patch: np.ndarray, border_color: int, opposite_side: str
) -> list[tuple[int, int]]:
    """Flood-fill the mouth interior of a standalone cup patch."""
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
    return sorted((int(r), int(c)) for r, c in visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate canonical UP-tail geometry to a target tail direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Sample canonical cup parameters that fit the original bbox limits."""
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
    spike_abs: tuple[int, int], direction: str, opening_abs: list[tuple[int, int]], H: int, W: int
) -> list[list[tuple[int, int]]]:
    """Absolute thick trajectory determined by the cup opening."""
    side = OPPOSITE[direction]
    span = _opening_span(opening_abs, side)
    if span is None:
        return []
    return _trajectory_bands(spike_abs, direction, span, H, W)


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    opening: list[tuple[int, int]],
    eject_len: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup so the full thick rectangle is visible and unobstructed."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        spike_abs = (r0 + spike[0], c0 + spike[1])
        opening_abs = [(r0 + r, c0 + c) for r, c in opening]
        path_bands = _trajectory_bands_from_opening(spike_abs, direction, opening_abs, H, W)
        if len(path_bands) < eject_len:
            continue

        path_cells = _flatten_bands(path_bands)
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        ejection_cells = _flatten_bands(path_bands[:eject_len])
        return r0, c0, path_cells, ejection_cells
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, clipped to the perpendicular extent."""
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
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> None:
    """Optionally place one isolated pixel on the mouth side of a cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
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
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, reserve its thick path corridor, and commit it to the canvases."""
    for _ in range(CUP_BUILD_RETRIES):
        direction = str(rng.choice(["U", "D", "L", "R"]))
        shape = _sample_cup_shape(rng, direction)
        if shape is None:
            continue

        L_arm, R_arm, hL, hR = shape
        patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
        border_pixels = int(np.count_nonzero(patch_canon == border_color))
        if not (MIN_BORDER_PIXELS <= border_pixels <= MAX_BORDER_PIXELS):
            continue

        mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
        if len(mouth_canon) < MIN_MOUTH_CELLS:
            continue

        patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
        ph, pw = patch.shape
        open_side = OPPOSITE[direction]
        opening = _opening_edge_cells(mouth, (0, 0, ph - 1, pw - 1), open_side)
        span = _opening_span(opening, open_side)
        if span is None:
            continue
        mouth_width = span[1] - span[0] + 1
        if mouth_width < MIN_MOUTH_WIDTH:
            continue

        max_count = min(CONTENT_COUNT_MAX, len(mouth), border_pixels - MIN_BORDER_COUNT_GAP)
        if max_count < 1:
            continue
        content_count = int(rng.integers(1, max_count + 1))

        placement = _try_place_cup(
            rng, H, W, patch, spike, direction, opening, border_pixels, occ_cup, occ_path
        )
        if placement is None:
            continue
        r0, c0, path_cells, ejection_cells = placement

        patch_mask = patch != 0
        canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
        canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
        occ_cup[r0:r0 + ph, c0:c0 + pw] = True
        for r, c in ejection_cells:
            canvas_out[r, c] = content_color
        for r, c in path_cells:
            occ_path[r, c] = True

        return {
            "border_color": border_color,
            "content_color": content_color,
            "direction": direction,
            "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
            "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
            "content_count": content_count,
            "path_cells": path_cells,
            "mouth_width": mouth_width,
            "border_pixels": border_pixels,
        }
    return None


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's sparse content points inside its mouth interior."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for idx in idxs:
            r, c = mouth_abs[int(idx)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single output-first generation attempt."""
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
