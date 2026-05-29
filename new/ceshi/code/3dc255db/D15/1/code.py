from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage

# Variant of 3dc255db
# Dimension: D15 — 边框颜色唯一性 (meta)
# Change: 每个杯的边框颜色唯一(不同杯颜色不同)... → 边框颜色编码弹射规则
"""
design_analysis:
  concreteization:
    preserved_structure:
      - 仍然是原题的空心杯/Y/漏斗框架，仍然通过唯一尖刺识别 tail 方向。
      - 杯内仍有单色内容像素，输出仍然会先擦除杯内内容，再从尖刺处弹射。
    new_border_color_semantics:
      - 蓝色边框(1): 沿 tail 方向直线弹射，长度 = 内容像素数 n。
      - 红色边框(2): 沿 tail 方向直线弹射，长度 = 2n。
      - 绿色边框(3): 弹射方向从 tail 顺时针偏转 90 度，长度 = n。
    small_trace:
      - 例: 一个 tail=U 的红杯，杯内有 3 个内容像素。
      - 原题: 从尖刺向上画 3 格。
      - 变体: 仍向上，但画 6 格。
      - 例: 一个 tail=L 的绿杯，杯内有 2 个内容像素。
      - 原题: 从尖刺向左画 2 格。
      - 变体: 方向改为顺时针后的 U，从尖刺向上画 2 格。
  complexity_calibration:
    - 只用 3 种边框色和 3 条固定规则，分支足够少，人能在少量样例中归纳。
    - 每个网格至少 2 个杯子，避免单杯样例信息不足。
    - 内容数限制为 2..4，保证“红色 2 倍长度”和“绿色拐弯”都能肉眼看见。
  implicit_assumption_audit:
    - 原 solver 的 `for border_col in colors_present(...)` 默认所有非背景色都有可能是杯框。
      变体后这不再成立；内容色不应被当作杯框，因此 solver 只处理 {1,2,3}。
    - 原 solver 的弹射 `dr,dc = DIR4[tail_dir]`、长度 `len(content_cells)` 依赖
      “边框颜色不携带行为信息”。变体后方向和长度都要先查颜色规则。
    - 原 generator 随机分配任意边框色、并允许内容色回流到调色板。
      变体后必须固定边框色集合，并让内容色与边框色集合分离，避免颜色语义混淆。
  placement_legality:
    - 杯框 bbox 仍然保持至少 1 格间隔，不重叠。
    - 清路检查不再沿原 tail，而是沿“颜色规则作用后的真实弹射方向”。
    - 红杯必须能完整显示 2n 长度，否则会退化成和普通直射近似；生成器直接拒绝这类放置。
    - 绿杯的偏转射线可能贴着自身 bbox 边缘经过，但不会穿过边框，因为 tail 边只有一个 spike。
  visual_observability:
    - 取消边框色 stray，避免孤立的红/蓝/绿像素误导解题者把它们当规则标记。
    - 内容色只从 4..9 里取，边框色只从 1..3 里取，视觉职责分离。
    - 在一个网格里放 2 或 3 个杯，训练样例之间会重复出现同一边框色，支持归纳颜色→行为。
  visual_cleanliness:
    - 不引入额外装饰标记。
    - 仍然使用 output-first 构造，再反推输入，保证每个输入都能被 solver 稳定处理。
  color_inference:
    - 颜色映射是固定全局规则，不依赖排序、数值大小、位置或不可见元信息。
    - 只有三种边框色参与规则，类别数受控。
  new_solver_assumptions_after_change:
    - 真正的杯框颜色只会出现在 {1,2,3} 中。
    - 杯内内容色不会使用 {1,2,3}。
    - 生成器显式保证这两点，因此 solver 可以安全忽略其他非背景色。
"""


RULE_BORDER_COLORS = (1, 2, 3)
CONTENT_COLORS = (4, 5, 6, 7, 8, 9)
DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
CW = {"U": "R", "R": "D", "D": "L", "L": "U"}

# border_color -> (direction_mode, length_multiplier)
RULES_BY_BORDER = {
    1: ("tail", 1),  # blue: original behavior
    2: ("tail", 2),  # red: doubled length
    3: ("cw", 1),    # green: rotate direction clockwise by 90 degrees
}


def _rule_direction(border_color: int, tail_dir: str) -> str:
    mode, _mult = RULES_BY_BORDER[int(border_color)]
    return tail_dir if mode == "tail" else CW[tail_dir]


def _rule_length(border_color: int, content_count: int) -> int:
    _mode, mult = RULES_BY_BORDER[int(border_color)]
    return int(mult) * int(content_count)


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
    - Detect cups exactly as in the original task.
    - Border color now selects the ejection behavior:
      blue(1)  -> straight, length n
      red(2)   -> straight, length 2n
      green(3) -> clockwise 90-degree turn, length n
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        border_col = int(border_col)
        if border_col not in RULES_BY_BORDER:
            continue
        for border_coords in components_mask(g == border_col, connectivity=8):
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
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != border_col:
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

            eject_dir = _rule_direction(border_col, tail_dir)
            eject_len = _rule_length(border_col, len(content_cells))
            dr, dc = DIR4[eject_dir]
            sr, sc = spike
            for t in range(1, eject_len + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MIN = 2
CONTENT_COUNT_MAX = 4

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
    """Flood-fill the mouth-side interior of a cup patch."""
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
    """Rotate canonical (tail=U) patch/spike/mouth to the target tail direction."""
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
    """Pick canonical cup parameters that still fit after rotation."""
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


def _trace_ray(H: int, W: int, start: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR4[direction]
    rr, cc = start[0] + dr, start[1] + dc
    cells = []
    while 0 <= rr < H and 0 <= cc < W:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    tail_dir: str,
    border_color: int,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place one cup so its bbox has a 1-cell gap and its effective ejection ray stays clear."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    actual_dir = _rule_direction(border_color, tail_dir)
    visible_len = _rule_length(border_color, content_count)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        path_cells = _trace_ray(H, W, (sr_abs, sc_abs), actual_dir)
        if len(path_cells) < visible_len:
            continue

        blocked = False
        for r, c in path_cells:
            if occ_cup[r, c] or occ_path[r, c]:
                blocked = True
                break
            lr = r - r0
            lc = c - c0
            if 0 <= lr < ph and 0 <= lc < pw and patch[lr, lc] != 0:
                blocked = True
                break
        if blocked:
            continue

        return r0, c0, path_cells, path_cells[:visible_len]
    return None


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
    """Build one cup, place it, and commit its border/output ray."""
    tail_dir = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, tail_dir)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < CONTENT_COUNT_MIN:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, tail_dir)
    ph, pw = patch.shape

    content_cap = min(CONTENT_COUNT_MAX, len(mouth))
    if content_cap < CONTENT_COUNT_MIN:
        return None
    content_count = int(rng.integers(CONTENT_COUNT_MIN, content_cap + 1))

    placement = _try_place_cup(
        rng, H, W, patch, spike, tail_dir, border_color, content_count, occ_cup, occ_path,
    )
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
        "direction": tail_dir,
        "actual_dir": _rule_direction(border_color, tail_dir),
        "eject_len": len(ejection_cells),
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's input content inside its mouth interior."""
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
    """Output-first generation followed by oracle verification through `main`."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    border_colors = list(rng.choice(RULE_BORDER_COLORS, size=num_cups, replace=False))
    content_palette = list(CONTENT_COLORS)
    rng.shuffle(content_palette)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for border_color in border_colors:
        if not content_palette:
            return None
        content_color = int(content_palette.pop())
        meta = _add_cup(
            rng,
            H,
            W,
            int(border_color),
            content_color,
            canvas_in,
            canvas_out,
            occ_cup,
            occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

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
