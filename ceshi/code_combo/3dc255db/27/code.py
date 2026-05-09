# Combo variant of 3dc255db (index 27)
# combo_spec: D10/0+D13/0
# Name: 杯口面积决定长度+双像素宽射线
#
# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 原题: 先找出每个杯状边框的尾刺方向, flood 出杯口内部 mouth, 擦掉 mouth 内的内容色像素,
#      然后沿尾刺方向画一条 1px 宽直线, 长度 = 内容像素数。
#    - D10/0 改动: 直线长度不再取内容数, 而是取 mouth_area = flood 到的 mouth 总格数。
#    - D13/0 改动: 每一步不再只画 1 个像素, 而是画一个 2px 宽的截面; 连续截面堆叠成 2px 宽射线。
#    - 融合后: 若某杯 mouth_area=5, 实际内容数=2, 尾刺朝上, 且 2px 宽截面落在列 4 和 5:
#      原题输出会在尾刺上方画 2 个单像素; D10 只会画 5 个单像素; 本融合版会擦掉那 2 个内容像素,
#      再在上方连续 5 层各画 2 个像素, 总计 10 个输出像素。
#    - 可见差异有两层: 输出长度跟杯腔面积走, 同时厚度固定为 2 而不是 1。
# b) 复杂度校准
#    - 若 content_count 经常等于 mouth_area, D10 会退化; 因此生成器强制 content_count <= mouth_area-2。
#    - 若尾刺到边界的可见距离 < mouth_area, 长度会被裁剪, D10 不明显; 因此生成器强制完整可见长度 >= mouth_area。
#    - 若 mouth opening 只有 1 格, 2px 宽无法定义; 因此生成器只接受 opening width >= 2 的杯子。
#    - 不额外引入新颜色/新标记, 仍保留 1-3 个杯子和少量 stray pixel, 让题目保持原题风格而非模板题。
# c) 隐式假设审查
#    - 原 flood 逻辑假设: mouth 由 bbox 某一侧向内 4 连通泛洪定义。融合后仍成立, 因为 mouth_area 就是用它定义的。
#    - 原 content 颜色判定依赖: 每个杯 mouth 内只有一种真正内容色。融合后仍保持, generator 保证这一点。
#    - 原 draw loop 隐含前提: 每一步只写 1 格, 且步数=内容数。融合后这两点都失效:
#      需要把步数改成 mouth_area, 并把每一步改成写 2 格 band。
#    - 宽度 2 还引入了一个新前提: 必须能从 mouth opening edge 上选出一个相邻的 2 格带。
#      solver 显式这样选; generator 则只生成满足该条件的杯子。
# d) 放置合法性
#    - 生成器继续 output-first: 先按 mouth_area 和 2px band 预留完整输出轨迹, 再回填输入内容。
#    - `occ_cup` 保证 cup bbox 之间至少隔 1 格; `occ_path` 记录整条厚轨迹的全部格子,
#      防止不同杯子的 2px 射线相撞, 也防止后续杯子压到已有轨迹上。
#    - stray pixel 仍只允许放在某个杯口外侧, 且不得落到任何 cup bbox 或 thick path 上。
# e) 视觉可观察性
#    - 人类能直接看到: 同色内容点数量比输出射线长度少, 射线长度反而对应杯腔面积;
#      同时射线的每一层都是并排 2 格, 明显不同于原题单像素线。
#    - 宽度规则不依赖颜色值; 颜色只负责“这条射线涂什么色”。
# f) 可视化整洁度
#    - 画面仍只有三层信息: 杯框 / 杯内内容 / 杯外射线。射线始终是单色直带, 不做锥形或多色编码。
#    - 通过路径预留避免厚射线和别的对象重叠, 保持图面可读。
# g) 变化的可推断性
#    - 本融合没有额外颜色映射。唯一新增几何规则是“长度=mouth_area, 宽度=2”。
#    - 为避免把“2px 宽”误看成“宽度等于整个杯口宽度”, 生成器允许 opening width 大于 2,
#      但输出始终只取 opening edge 上最靠近尾刺投影的一对相邻格作为固定宽度带。
#      因此宽度来源是一个稳定的局部几何锚点, 不是颜色或绝对坐标。

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


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


def _opening_edge_cells(
    mouth: set[tuple[int, int]] | list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    """Cells of the mouth that lie directly on the opening edge."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        return sorted((r, c) for r, c in mouth if r == r0)
    if side == "D":
        return sorted((r, c) for r, c in mouth if r == r1)
    if side == "L":
        return sorted((r, c) for r, c in mouth if c == c0)
    return sorted((r, c) for r, c in mouth if c == c1)


def _select_double_width_pair(
    edge_cells: list[tuple[int, int]], side: str, spike: tuple[int, int],
) -> tuple[int, int] | None:
    """Pick the adjacent 2-cell opening segment closest to the spike projection."""
    if side in ("U", "D"):
        vals = sorted({c for _, c in edge_cells})
        anchor = spike[1]
    else:
        vals = sorted({r for r, _ in edge_cells})
        anchor = spike[0]

    pairs = [(a, b) for a, b in zip(vals, vals[1:]) if b == a + 1]
    if not pairs:
        return None

    def score(pair: tuple[int, int]) -> tuple[float, int, int]:
        a, b = pair
        midpoint = (a + b) / 2.0
        return (abs(midpoint - anchor), abs(a - anchor) + abs(b - anchor), a)

    return min(pairs, key=score)


def _trajectory_bands(
    spike: tuple[int, int], direction: str, width_pair: tuple[int, int], H: int, W: int,
) -> list[list[tuple[int, int]]]:
    """Return consecutive 2px-wide bands from the spike outward until the grid edge."""
    sr, sc = spike
    dr, dc = DIR4[direction]
    a, b = width_pair
    bands: list[list[tuple[int, int]]] = []
    t = 1
    while True:
        rr = sr + dr * t
        cc = sc + dc * t
        if not (0 <= rr < H and 0 <= cc < W):
            break
        if direction in ("U", "D"):
            bands.append([(rr, a), (rr, b)])
        else:
            bands.append([(a, cc), (b, cc)])
        t += 1
    return bands


def _flatten_bands(bands: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for band in bands for cell in band]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - detect cups as in the original task
    - erase the mouth's content-colored pixels
    - eject a straight ray whose length = mouth interior area
    - each step of that ray is a fixed 2px-wide band chosen from the mouth opening geometry
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
                opening = _opening_edge_cells(mouth, bbox, mouth_side)
                width_pair = _select_double_width_pair(opening, mouth_side, spike)
                if width_pair is None:
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
                passing.append((d, spike, width_pair, content_col, content_cells, len(mouth)))

            if len(passing) != 1:
                continue
            tail_dir, spike, width_pair, content_col, content_cells, mouth_area = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            for band in _trajectory_bands(spike, tail_dir, width_pair, H, W)[:mouth_area]:
                for r, c in band:
                    out[r, c] = content_col

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
MIN_MOUTH_AREA = 4
MAX_MOUTH_AREA = 8
MIN_AREA_COUNT_GAP = 2

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400
# ==================================


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
    """Flood-fill the cup interior from the open side."""
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
    """Pick canonical cup parameters clamped to the original bbox limits."""
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


def _trajectory_bands_from_rel_pair(
    spike_abs: tuple[int, int],
    direction: str,
    rel_pair: tuple[int, int],
    offset: tuple[int, int],
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    """Convert a relative width pair into absolute 2px bands."""
    r0, c0 = offset
    if direction in ("U", "D"):
        width_pair = (c0 + rel_pair[0], c0 + rel_pair[1])
    else:
        width_pair = (r0 + rel_pair[0], r0 + rel_pair[1])
    return _trajectory_bands(spike_abs, direction, width_pair, H, W)


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    rel_pair: tuple[int, int],
    eject_len: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup so its full 2px-wide, area-driven ejection is visible and unobstructed."""
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
        path_bands = _trajectory_bands_from_rel_pair((sr_abs, sc_abs), direction, rel_pair, (r0, c0), H, W)
        if len(path_bands) < eject_len:
            continue

        path_cells = _flatten_bands(path_bands)
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        return r0, c0, path_cells, _flatten_bands(path_bands[:eject_len])
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Boolean mask of cells on the mouth side of `bbox`, restricted to the perpendicular extent."""
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


def _choose_content_count(rng: np.random.Generator, mouth_area: int) -> int | None:
    """Pick a sparse content count so the area-driven length stays visibly different."""
    hi = min(CONTENT_COUNT_MAX, mouth_area - MIN_AREA_COUNT_GAP)
    if hi < 1:
        return None
    return int(rng.integers(1, hi + 1))


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
    """Build one cup and commit its area-driven 2px-wide ejection band."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    mouth_area = len(mouth_canon)
    if mouth_area < MIN_MOUTH_AREA or mouth_area > MAX_MOUTH_AREA:
        return None

    content_count = _choose_content_count(rng, mouth_area)
    if content_count is None:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    open_side = OPPOSITE[direction]
    opening = _opening_edge_cells(mouth, (0, 0, ph - 1, pw - 1), open_side)
    rel_pair = _select_double_width_pair(opening, open_side, spike)
    if rel_pair is None:
        return None

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, rel_pair, mouth_area, occ_cup, occ_path,
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
        "mouth_area": mouth_area,
        "content_count": content_count,
        "path_cells": path_cells,
        "rel_pair": rel_pair,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's sparse content pixels inside its mouth interior."""
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
    """Single generation attempt with output-first oracle checking."""
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
