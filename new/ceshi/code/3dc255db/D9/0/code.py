# Variant of 3dc255db
# Dimension: D9 — 杯形结构形状 (src)
# Change: Y形/漏斗形 — 两臂对角收敛到单像素尖刺，臂长可不对称，底部开口... → 矩形框架缺口
#
# Design analysis
# ---
# variant_spec:
#   preserved_rule:
#     - 每个杯内只有一种内容色；内容像素会被擦除，并从尖刺朝尾向外弹射成直线。
#     - 仍然允许 1-3 个杯、4 个朝向、0-1 个 stray pixel，仍采用 output-first 生成并用 solver 回验。
#   d9_change:
#     - 原题杯体是 Y/漏斗形，尖刺是 bbox 某边上唯一的边框像素。
#     - 本变体杯体改为规则的三边矩形框：3 边封闭、1 边开口。
#     - 尖刺不再是独立凸出的像素，而是“封闭边中心像素”；弹射方向为该边的外法向。
#   concrete_geometry:
#     - 以 tail=U 为例：top row 全封闭，left/right col 全封闭，bottom side 只保留两个角点，中间开口。
#     - mouth 是从开口边进入、被三边框住的所有非边框格。
#     - spike = top row 正中格；因此封闭边长度必须为奇数，避免中心二义性。
#   behavior_trace:
#     - 原 solver 在候选尖刺检测阶段要求“某 bbox 边恰好只有 1 个边框像素”。
#     - 对新 U 形矩形杯，edge counts 变成例如 top=5, bottom=2, left=4, right=4；原条件完全失效。
#     - 新 solver 先枚举四个方向，检查组件是否严格等于该方向下的三边矩形框，再从开口边洪泛找内容。
#     - 找到含内容的开口后，尖刺位置直接由封闭边中心给出，之后擦除/弹射逻辑与原题一致。
#   complexity_calibration:
#     - 形状规则化后，识别框体更容易；但尖刺不再是独立外凸点，必须定位封闭边中心。
#     - 仍保留朝向、尺寸、内容数量、多杯排布和 stray pixel，因此不会退化成过度简单的单模板任务。
#   implicit_assumption_audit:
#     - 原 main() 的 `len(cells) == 1` 候选边假设在变体中不成立，必须移除。
#     - 原 flood-from-opposite-side 思路仍成立，因为开口边仍是唯一能进入 mouth 的 bbox 边。
#     - 新 main() 新增假设：杯体必须精确匹配三边矩形框，且封闭边长度为奇数。generator 负责保证。
#   placement_legality:
#     - 杯 bbox 之间继续保持至少 1 格间距；弹射路径必须清空到边界。
#     - 由于尖刺位于 bbox 边上，弹射第一格必在 bbox 外；placement 只需像原题一样检查外侧直线路径。
#   visual_observability:
#     - 人类可直接看到“Y/漏斗”被替换成“U/C 形矩形框”，这是样例级别可见的结构变化。
#     - 开口边与封闭边对置，输出中的射线总从封闭边正中射出，规则可从少量样例归纳。
#   neatness:
#     - 仍禁止 bbox 重叠和路径冲突，stray pixel 只落在 mouth 一侧且与已有非背景保持 1 格隔离。

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
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


def _rect_frame_cells(bbox: tuple[int, int, int, int], tail_dir: str) -> set[tuple[int, int]]:
    """Exact cell set of the 3-sided rectangular cup for the given tail direction."""
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    if tail_dir == "U":
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
    elif tail_dir == "D":
        cells.update((r1, c) for c in range(c0, c1 + 1))
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
    elif tail_dir == "L":
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
    else:  # "R"
        cells.update((r, c1) for r in range(r0, r1 + 1))
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
    return cells


def _spike_from_bbox_center(bbox: tuple[int, int, int, int], tail_dir: str) -> tuple[int, int] | None:
    """Center pixel of the closed side; returns None if that side length is even."""
    r0, c0, r1, c1 = bbox
    if tail_dir in ("U", "D"):
        width = c1 - c0 + 1
        if width % 2 == 0:
            return None
        return (r0 if tail_dir == "U" else r1, c0 + width // 2)
    height = r1 - r0 + 1
    if height % 2 == 0:
        return None
    return (r0 + height // 2, c0 if tail_dir == "L" else c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Cups are 3-sided rectangular frames with one open side.
    - The spike is the center pixel on the closed side opposite the opening.
    - Content in the mouth interior is erased and ejected outward from that center.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 7:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue

            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in border_coords}
            passing = []

            for tail_dir in ("U", "D", "L", "R"):
                if border_set != _rect_frame_cells(bbox, tail_dir):
                    continue
                spike = _spike_from_bbox_center(bbox, tail_dir)
                if spike is None or spike not in border_set:
                    continue

                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                if not mouth:
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
                passing.append((tail_dir, spike, content_col, content_cells))

            if len(passing) != 1:
                continue

            tail_dir, spike, content_col, content_cells = passing[0]
            for r, c in content_cells:
                out[r, c] = bg

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

# Canonical shape is tail=U. For L/R, rotation swaps axes, so canonical width may be up to 7
# while canonical height may be up to 7 to respect final bbox bounds from the original task.
CUP_BBOX_MAX_UD = (8, 7)  # (max_h, max_w) when tail is U or D
CUP_BBOX_MAX_LR = (7, 8)  # (max_h, max_w) when tail is L or R

CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(height: int, width: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a 3-sided rectangular cup with tail pointing UP and open side at the bottom."""
    assert height >= 3 and width >= 3 and width % 2 == 1
    g = np.zeros((height, width), dtype=int)
    g[0, :] = color
    g[:, 0] = color
    g[:, width - 1] = color
    return g, (0, width // 2)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the cup mouth interior from the open bbox side."""
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
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
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


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int] | None:
    """Pick canonical (height, width) for a 3-sided rectangular cup."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    shapes = [
        (h, w)
        for h in range(3, max_h_canon + 1)
        for w in range(3, max_w_canon + 1, 2)
        if (h - 1) * (w - 2) >= 3
    ]
    if not shapes:
        return None
    return shapes[int(rng.integers(0, len(shapes)))]


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
    """Find a valid placement offset such that cup bboxes stay separated and the path is clear."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
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
    """Cells on the mouth side of a cup bbox, restricted to the cup's perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif tail_dir == "D":
        mask[:r0, c0:c1 + 1] = True
    else:  # "U"
        mask[r1 + 1:, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    """Optionally place an isolated pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
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
    """Build one cup (shape + placement + ejection line) and commit it to the canvases."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    height, width = shape

    patch_canon, spike_canon = _build_canonical_cup(height, width, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
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
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "closed_side_len": width if direction in ("U", "D") else patch.shape[0],
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

        meta = _add_cup(rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path)
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
