# Variant of 3dc255db
# Dimension: D8 — 多步处理顺序 (multi_step)
# Change: ordered_processing — 内容像素逐个移动到尾部堆叠(实际效果等同于一次性放置)... → 迭代弹射

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage

# Design analysis
# ===============
# spec:
#   focus_dimension: D8 / multi_step
#   preserved_structure:
#     - 杯体仍然是单色 8 连通边框，尾刺仍然由 bbox 某一边唯一的 1 个边框像素确定
#     - 内容仍然只来自杯口内部，颜色仍然是“每杯内部单色”
#     - 求解仍然先识别每个杯，再把内部内容擦除
#   changed_behavior:
#     - 原题是“每个杯独立完成弹射”：每杯一次性把 count 个像素放到自己尾部路径的前 count 格
#     - 本变体改成“全局轮转弹射”：所有杯按尾刺阅读顺序轮流行动，每轮每杯至多弹射 1 个像素
#     - 某杯行动时，像素沿尾部方向扫描，落在当前路径上第一个背景格；若整条路径已满，则该像素被丢弃
#     - 因为别的杯可能已经占住交叉点，所以后来的杯会跳过该格，把自己的像素放到更远处，形成交错线
#   concrete_trace:
#     - 例：A 向右，路径是 a1,a2,X,a4；B 向上，路径是 b1,X,b3，且 B 比 A 更早抵达 X
#     - 原题：A 会占 a1,a2,X,a4；B 会占 b1,X,b3（若允许重叠，只是静态覆盖）
#     - 新题：第 1 轮 A→a1，B→b1；第 2 轮 A→a2，B→X；第 3 轮 A 发现 X 已被 B 占据，于是跳到 a4
#     - 可见差异不是“有没有直线”，而是“谁先抢到共享路径格”，所以必须模拟时间步
#   complexity_calibration:
#     - 若保留 1 杯样本，新规则会退化回原题，因此生成器改为 2-3 杯
#     - 若仍禁止路径相交，新规则在视觉上也几乎退化回原题，因此生成器强制至少一处可观测交叉
#     - 为了让交错可读，交叉杯的内容色保持互异；这样人眼可直接看到哪一格被谁抢占
#   implicit_assumptions_review:
#     - 原 solver 中 `for t in range(1, len(content_cells)+1)` 假设“每杯自己的路径永远空着”，交叉后不成立
#     - 原 solver 中逐杯直接写 `out[rr, cc] = content_col` 假设不同杯之间互不影响，交叉后不成立
#     - 原 generator 中 `_try_place_cup` 禁止 `occ_path` 相交，这正好抹掉了 D8 行为，需要放宽为“路径可相交，但杯 bbox 仍不可压到已有路径上”
#     - 原 generator 先静态画出输出线；新规则下输出依赖全局轮转过程，因此改为“先构造输入，再用新 main 做 oracle 校验”
#   placement_legality:
#     - 杯 bbox 之间仍保持至少 1 格间距
#     - 新放置约束是：新杯的 bbox 不能覆盖旧路径；新路径不能穿过已有杯 bbox；路径之间允许共享背景格
#     - 杯口、尾刺、内容散布方式保持原题风格，因此 solver 的杯识别逻辑仍可复用
#   visual_observability:
#     - 交叉必须真实改变某杯的最终落点；因此生成器额外检查：至少 1 个杯在最终输出中的自色路径位置，不等于“若单独弹射时的前缀位置”
#     - 这比只检查“路径相交”更强，因为单纯相交但未产生抢占，视觉上仍会退化
#   cleanliness:
#     - 杯本体仍互不接触，杂点不放在任何路径上，避免把“静态噪声”误读成弹射结果
#     - 杂点概率略降，避免在多杯交错任务中抢走注意力

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.25
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from non-border cells on one bbox edge."""
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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _path_from_spike(shape: tuple[int, int], spike: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    H, W = shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    out: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def _detect_cups(grid: np.ndarray) -> tuple[int, list[dict]]:
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
            candidates = [(d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners]
            if not candidates:
                continue

            passing = []
            for direction, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[direction])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = sorted(p for p in mouth if int(g[p[0], p[1]]) == content_col)
                passing.append(
                    {
                        "border_color": int(border_col),
                        "bbox": bbox,
                        "direction": direction,
                        "spike": (int(spike[0]), int(spike[1])),
                        "mouth": mouth,
                        "content_color": content_col,
                        "content_cells": content_cells,
                    }
                )

            if len(passing) != 1:
                continue
            cup = passing[0]
            cup["path_cells"] = _path_from_spike(g.shape, cup["spike"], cup["direction"])
            cup["content_count"] = len(cup["content_cells"])
            cups.append(cup)

    return bg, cups


def _simulate_iterative(grid: np.ndarray, bg: int, cups: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    ordered_cups = sorted(cups, key=lambda cup: (cup["spike"][0], cup["spike"][1], cup["direction"]))

    for cup in ordered_cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    remaining = [int(cup["content_count"]) for cup in ordered_cups]
    while any(v > 0 for v in remaining):
        for idx, cup in enumerate(ordered_cups):
            if remaining[idx] <= 0:
                continue
            for rr, cc in cup["path_cells"]:
                if int(out[rr, cc]) == bg:
                    out[rr, cc] = int(cup["content_color"])
                    break
            remaining[idx] -= 1
    return out


def _simulate_one_shot(grid: np.ndarray, bg: int, cups: list[dict]) -> np.ndarray:
    """Source-task behavior: each cup fills its own path prefix in detection order."""
    out = np.array(grid, dtype=int, copy=True)
    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg
        for rr, cc in cup["path_cells"][: int(cup["content_count"])]:
            out[rr, cc] = int(cup["content_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each cup exactly as in the source task, but simulate ejection in global rounds.

    Every round processes all cups in spike reading order. A cup ejects at most one pixel:
    the pixel occupies the first currently-empty cell on that cup's tail ray. This makes
    intersecting rays interact, so later cups may skip cells already claimed by earlier rounds.
    """
    g = np.array(grid, dtype=int)
    bg, cups = _detect_cups(g)
    return _simulate_iterative(g, bg, cups)


def analyze_variant(grid: np.ndarray, output: np.ndarray | None = None) -> dict:
    """Quantify whether an input truly exercises the iterative-interaction variant."""
    g = np.array(grid, dtype=int)
    _bg, cups = _detect_cups(g)
    out = main(g) if output is None else np.array(output, dtype=int)

    shared_path_cells = 0
    interacting_pairs = 0
    for i in range(len(cups)):
        set_i = set(cups[i]["path_cells"])
        for j in range(i + 1, len(cups)):
            shared = set_i & set(cups[j]["path_cells"])
            if shared:
                interacting_pairs += 1
                shared_path_cells += len(shared)

    displaced_cups = 0
    for cup in cups:
        path_cells = cup["path_cells"]
        k = min(int(cup["content_count"]), len(path_cells))
        expected = path_cells[:k]
        actual = [cell for cell in path_cells if int(out[cell]) == int(cup["content_color"])]
        if actual != expected:
            displaced_cups += 1

    return {
        "cup_count": len(cups),
        "interacting_pairs": interacting_pairs,
        "shared_path_cells": shared_path_cells,
        "displaced_cups": displaced_cups,
    }


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
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


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
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
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue

        spike_abs = (r0 + spike[0], c0 + spike[1])
        path_cells = _path_from_spike((H, W), spike_abs, direction)
        if not path_cells:
            continue
        if any(occ_cup[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
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
    rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
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


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
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
    placement = _try_place_cup(rng, H, W, patch, spike, direction, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells = placement
    ph, pw = patch.shape

    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "path_cells": path_cells,
        "spike": (r0 + spike[0], c0 + spike[1]),
    }


def _choose_interaction_plan(cups_meta: list[dict]) -> dict | None:
    turn_order = sorted(range(len(cups_meta)), key=lambda idx: (cups_meta[idx]["spike"][0], cups_meta[idx]["spike"][1], idx))
    turn_rank = {cup_idx: rank for rank, cup_idx in enumerate(turn_order)}
    path_pos = [{cell: i + 1 for i, cell in enumerate(cup["path_cells"])} for cup in cups_meta]
    caps = [min(CONTENT_COUNT_MAX, len(cup["mouth_abs"])) for cup in cups_meta]

    best = None
    best_score = None
    for i in range(len(cups_meta)):
        for j in range(i + 1, len(cups_meta)):
            shared = set(path_pos[i]) & set(path_pos[j])
            for cell in shared:
                ti = path_pos[i][cell]
                tj = path_pos[j][cell]
                if (ti, turn_rank[i]) < (tj, turn_rank[j]):
                    claimant, claim_round, delayed, delayed_round = i, ti, j, tj
                else:
                    claimant, claim_round, delayed, delayed_round = j, tj, i, ti
                if claim_round > caps[claimant] or delayed_round > caps[delayed]:
                    continue
                if len(cups_meta[delayed]["path_cells"]) <= delayed_round:
                    continue
                score = (
                    len(cups_meta[delayed]["path_cells"]) - delayed_round,
                    delayed_round - claim_round,
                    -claim_round,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = {
                        "shared_cell": cell,
                        "claimant": claimant,
                        "claim_round": claim_round,
                        "delayed": delayed,
                        "delayed_round": delayed_round,
                    }
    return best


def _assign_content_counts(rng: np.random.Generator, cups_meta: list[dict], plan: dict | None) -> list[int] | None:
    if plan is None:
        return None

    counts: list[int] = []
    for cup in cups_meta:
        cap = min(CONTENT_COUNT_MAX, len(cup["mouth_abs"]))
        low = 2 if cap >= 2 else 1
        counts.append(int(rng.integers(low, cap + 1)))

    for key_idx, key_round in (("claimant", "claim_round"), ("delayed", "delayed_round")):
        idx = int(plan[key_idx])
        required = int(plan[key_round])
        cap = min(CONTENT_COUNT_MAX, len(cups_meta[idx]["mouth_abs"]))
        if required > cap:
            return None
        hi = min(cap, required + 2)
        counts[idx] = int(rng.integers(required, hi + 1))

    return counts


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict], counts: list[int]) -> bool:
    for meta, count in zip(cups_meta, counts, strict=True):
        mouth_abs = meta["mouth_abs"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = meta["content_color"]
        meta["content_count"] = int(count)
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Interaction-first generator: place cups, enforce a usable crossing, then oracle-check main()."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
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
        meta = _add_cup(rng, H, W, border_color, content_color, canvas_in, occ_cup, occ_path)
        if meta is None:
            return None
        cups_meta.append(meta)

    plan = _choose_interaction_plan(cups_meta)
    counts = _assign_content_counts(rng, cups_meta, plan)
    if counts is None:
        return None
    if not _scatter_content(rng, canvas_in, cups_meta, counts):
        return None

    _try_place_stray(rng, canvas_in, cups_meta)

    try:
        produced = main(canvas_in)
    except Exception:
        return None

    bg, detected = _detect_cups(canvas_in)
    legacy = _simulate_one_shot(canvas_in, bg, detected)
    metrics = analyze_variant(canvas_in, produced)
    if metrics["shared_path_cells"] < 1 or metrics["displaced_cups"] < 1:
        return None
    if np.array_equal(produced, legacy):
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
