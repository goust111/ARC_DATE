# Combo variant of 3dc255db (index 1)
# combo_spec: D1/0+D2/0
# Name: 对角弹射+颜色编码方向

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


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#   - 原题: 杯框是正交 Y/funnel；尾刺在哪条 bbox 边上，就沿该正交方向发射。
#   - D1/0: 几何改成对角 funnel；尾刺落在 bbox 某个角上，嘴朝对角的另一象限，输出沿该角对应
#     的 45 度方向发射。
#   - D2/0: 方向不再来自形状，而来自杯内内容色。
#   - 融合后: 先像 D1 那样通过 diagonal cup 几何识别“哪个 corner 是 spike、mouth 在哪”，
#     但真正绘制射线时不使用 spike corner 对应的方向，而是使用固定的颜色编码:
#       red(2)->NW, yellow(4)->NE, green(3)->SW, blue(1)->SE
#     小例子:
#       1. 一个 spike 在 NE 角的对角杯，mouth 在 SW 象限，里面有 3 个红色(2)内容像素。
#       2. 原 D1 会从 NE 角沿 NE 方向画 3 格。
#       3. 本融合版会仍然从 NE 角识别出同一批 mouth 内容并擦除，但改为按 red->NW，
#          在 NE 角外沿 NW 对角线画 3 格。
#     差异点有两个且协同出现:
#       - 几何是 diagonal cup，不是正交 cup。
#       - 发射方向是颜色控制，不是 shape 控制。
#
# b) 复杂度校准
#   - 方向档位只保留 4 个对角方向，内容色只用 1..4 四种，便于从少量样例归纳。
#   - 每个样例放 2..3 个杯子，保证同图里通常能看到多种颜色方向。
#   - border colors 只用 5..9，避免把方向编码色和框颜色混淆。
#   - 生成器强制“颜色方向 != spike 角的原始 D1 方向”，避免退化成单纯 D1。
#
# c) 隐式假设审查
#   - 原 solver 假设“candidate spike 在哪个边/角，就向哪个方向画线”。这里失效。
#   - D1 solver 仍可复用的前提:
#       * diagonal cup 只有一个有效 spike corner；
#       * 从 spike 对角的两条 bbox 边联合 flood 能拿到 mouth；
#       * mouth 内内容色单一。
#   - 新增前提:
#       * 真正会触发发射的内容色属于 {1,2,3,4}；
#       * 颜色与方向的对应是固定且跨样例一致的。
#     generator 通过内容色池和 oracle 校验保证这些前提。
#
# d) 放置合法性
#   - 杯框 bbox 仍保持至少 1-cell gap。
#   - ejection path 按颜色方向检查清空，而不是按 spike 方向。
#   - 为保证“从 spike 向外”而不是射回杯内，生成器只允许那些会从 spike 角直接离开 bbox
#     的颜色方向，而且还强制不等于该 corner 的原始 D1 方向。
#
# e) 视觉可观察性
#   - 不同朝向的杯子若内容色相同，会朝同一个对角方向发射，足以暴露“颜色编码方向”。
#   - 同时杯框本身明显是 diagonal funnel，能看出不是原题的正交直线弹射。
#   - 生成器避免颜色方向与 shape 方向相同，因此对同一输入，shape-based D1 输出与本题输出
#     必然不同。
#
# f) 可视化整洁度
#   - 路径外部清空，bbox 之间留缝，stray pixel 只用 border color，避免在杯外制造伪内容色。
#   - 内容仍只出现在 mouth 内，边框仍是单色 8-connected component。
#
# g) 变化可推断性
#   - 颜色档位只有四种，且每种颜色控制一个稳定的绝对对角方向。
#   - solver 在 ARC 真正评测时应从训练对中“学到”这个固定映射；此处单文件 `main()` 无法访问
#     训练对，所以直接实现该一致映射，generator 也围绕同一映射构造样例。
#
# post-main assumption review
#   - `_identify_diagonal_cup_payload()` 假设只有一个 corner candidate 的 opposite-corner flood
#     能碰到内容；generator 通过单-corner funnel 保证。
#   - `main()` 假设内容色不与 border 色混用到同一 cup 内；generator 保证每个 cup 双色分离。
#   - `main()` 假设沿颜色方向的对角线无需避障；generator 的 placement 对该方向逐格清场。


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)

CUP_H_RANGE = (4, 7)
CUP_W_RANGE = (4, 7)
CONTENT_COUNT_MAX = 7

STRAY_PROB = 0.4

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIAG_DIRS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}
COLOR_TO_DIAG = {
    2: "NW",
    4: "NE",
    3: "SW",
    1: "SE",
}
MOUTH_SIDES = {
    "NW": ("D", "R"),
    "NE": ("D", "L"),
    "SW": ("U", "R"),
    "SE": ("U", "L"),
}
# Allowed true spike corners for each color-coded ejection direction.
# They are chosen so the first ray step leaves the cup bbox, but does not match the
# original D1 corner->direction behavior.
COMPATIBLE_TAILS = {
    "NW": ("NE", "SW"),
    "NE": ("NW", "SE"),
    "SW": ("NW", "SE"),
    "SE": ("NE", "SW"),
}
_STRUCT8 = np.ones((3, 3), dtype=bool)
BORDER_COLOR_POOL = (5, 6, 7, 8, 9)
CONTENT_COLOR_POOL = (1, 2, 3, 4)


def _corner_coord(bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if corner == "NW":
        return r0, c0
    if corner == "NE":
        return r0, c1
    if corner == "SW":
        return r1, c0
    if corner == "SE":
        return r1, c1
    raise ValueError(corner)


def _edge_seeds(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        return [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    if side == "D":
        return [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    if side == "L":
        return [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    if side == "R":
        return [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]
    raise ValueError(side)


def _flood_from_bbox_sides(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    sides: tuple[str, ...],
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    seeds: list[tuple[int, int]] = []
    for side in sides:
        seeds.extend(_edge_seeds(border_set, bbox, side))
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


def _identify_diagonal_cup_payload(
    g: np.ndarray,
    bg: int,
    border_coords: np.ndarray,
) -> tuple[str, tuple[int, int], int, list[tuple[int, int]]] | None:
    if border_coords.size == 0 or int(border_coords.shape[0]) < 4:
        return None

    bbox = tuple(int(v) for v in bbox_coords(border_coords))
    border_set = {(int(r), int(c)) for r, c in border_coords}
    border_col = int(g[int(border_coords[0, 0]), int(border_coords[0, 1])])

    candidates = [
        (corner, _corner_coord(bbox, corner))
        for corner in ("NW", "NE", "SW", "SE")
        if _corner_coord(bbox, corner) in border_set
    ]
    if not candidates:
        return None

    passing = []
    for tail_dir, spike in candidates:
        mouth = _flood_from_bbox_sides(border_set, bbox, MOUTH_SIDES[tail_dir])
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
        return None
    return passing[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect diagonal cup geometry exactly like the D1 variant: spike corner + opposite-corner mouth.
    - Erase the cup's content pixels from the detected mouth.
    - Emit a 45-degree ray from the spike, but choose the diagonal from the content color instead of
      from the spike corner.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            payload = _identify_diagonal_cup_payload(g, bg, border_coords)
            if payload is None:
                continue
            _, spike, content_col, content_cells = payload
            eject_dir = COLOR_TO_DIAG.get(content_col)
            if eject_dir is None:
                continue

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIAG_DIRS[eject_dir]
            sr, sc = spike
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


def _bresenham_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    pts: list[tuple[int, int]] = []
    r, c = r0, c0
    while True:
        pts.append((int(r), int(c)))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return pts


def _build_canonical_cup(
    h: int,
    w: int,
    bottom_exit: int,
    right_exit: int,
    color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Build a diagonal funnel whose true spike corner is NW.

    One wall runs from NW toward the bottom edge; the other from NW toward the right edge.
    The mouth therefore opens toward the SE region.
    """
    assert h >= 4 and w >= 4
    assert 1 <= bottom_exit <= w - 2
    assert 1 <= right_exit <= h - 2
    patch = np.zeros((h, w), dtype=int)
    for r, c in _bresenham_line((0, 0), (h - 1, bottom_exit)):
        patch[r, c] = color
    for r, c in _bresenham_line((0, 0), (right_exit, w - 1)):
        patch[r, c] = color
    return patch, (0, 0)


def _mouth_interior_coords(
    patch: np.ndarray,
    border_color: int,
    sides: tuple[str, ...],
) -> list[tuple[int, int]]:
    h, w = patch.shape
    border_set = {(int(r), int(c)) for r, c in np.argwhere(patch == int(border_color))}
    bbox = (0, 0, h - 1, w - 1)
    return sorted(_flood_from_bbox_sides(border_set, bbox, sides))


def _flip_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "NW":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "NE":
        return np.fliplr(patch).copy(), (sr, w - 1 - sc), [(r, w - 1 - c) for r, c in mouth]
    if direction == "SW":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "SE":
        return (
            np.flipud(np.fliplr(patch)).copy(),
            (h - 1 - sr, w - 1 - sc),
            [(h - 1 - r, w - 1 - c) for r, c in mouth],
        )
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    h = int(rng.integers(CUP_H_RANGE[0], CUP_H_RANGE[1] + 1))
    w = int(rng.integers(CUP_W_RANGE[0], CUP_W_RANGE[1] + 1))

    max_bottom_exit = min(w - 2, max(1, (w - 1) // 2))
    max_right_exit = min(h - 2, max(1, (h - 1) // 2))
    if max_bottom_exit < 1 or max_right_exit < 1:
        return None

    bottom_exit = int(rng.integers(1, max_bottom_exit + 1))
    right_exit = int(rng.integers(1, max_right_exit + 1))
    return h, w, bottom_exit, right_exit


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    eject_direction: str,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    ph, pw = patch.shape
    dr, dc = DIAG_DIRS[eject_direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells: list[tuple[int, int]] = []
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


def _stray_candidate_mask(
    H: int,
    W: int,
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "NW":
        mask[r1 + 1 :, c1 + 1 :] = True
    elif tail_dir == "NE":
        mask[r1 + 1 :, :c0] = True
    elif tail_dir == "SW":
        mask[:r0, c1 + 1 :] = True
    else:  # "SE"
        mask[:r0, :c0] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["tail_direction"]) & ~forbidden
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
    eject_direction = COLOR_TO_DIAG[content_color]
    tail_direction = str(rng.choice(COMPATIBLE_TAILS[eject_direction]))
    shape = _sample_cup_shape(rng)
    if shape is None:
        return None
    h, w, bottom_exit, right_exit = shape

    patch_canon, spike_canon = _build_canonical_cup(h, w, bottom_exit, right_exit, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, sides=("D", "R"))
    if len(mouth_canon) < 3:
        return None

    patch, spike, mouth = _flip_to_direction(patch_canon, spike_canon, mouth_canon, tail_direction)
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(
        rng, H, W, patch, spike, eject_direction, content_count, occ_cup, occ_path,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement
    ph, pw = patch.shape

    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True
    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "tail_direction": tail_direction,
        "eject_direction": eject_direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
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
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    border_palette = list(BORDER_COLOR_POOL)
    content_palette = list(CONTENT_COLOR_POOL)
    rng.shuffle(border_palette)
    rng.shuffle(content_palette)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if not border_palette or not content_palette:
            return None
        border_color = int(border_palette.pop())
        content_color = int(content_palette.pop())
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
    raise RuntimeError(
        f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts"
    )
