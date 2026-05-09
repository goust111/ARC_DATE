# Variant of 3dc255db
# Dimension: D17 — 尖刺识别消歧规则 (meta)
# Change: 尖刺 = 某bbox边上恰好只有1个边框像素且不在角落的点；多候选时通过flood-fill验证哪个对侧有内容... → 多尖刺多方向弹射

"""
design_analysis:
  variant_goal:
    original: 每个杯子至多接受 1 个通过 flood-fill 验证的尖刺，内容像素全部沿该单一方向弹射。
    new: 一个杯子允许 3 个同时通过验证的尖刺；它们都发射射线，内容像素按固定顺序均分到多条射线上。
  concrete_shape:
    canonical_shape: 开口朝下的三尖刺框架。顶部 1 个尖刺，左右边各 1 个尖刺，底边无尖刺且保持开口。
    border_geometry:
      - 顶部尖刺沿两条对角臂连到左右边尖刺
      - 左右边尖刺再各自沿对角线回收到底部内侧肩点
      - 因为底部没有闭合，bbox 内非边框区域连通，且能同时从 D/L/R 三侧 flood 进入
    content_region:
      - solver 侧把所有 passing 尖刺看到的同色内容做并集并去重
      - generator 不强求所有尖刺看到完全同一批像素，只保证每个尖刺的对侧 flood 至少能看到一些同色内容
      - 这允许一个杯内内容分散在多个 mouth 子区域中，但输出时仍按“总内容数”均分到多条射线上
  behavior_trace:
    original_example:
      - 若候选尖刺为 U/L/R 三个，原 solver 会把三者都放进 passing
      - 因为 len(passing) != 1，整杯直接跳过，不发生任何弹射
    new_example:
      - 同样的 U/L/R 三个候选都通过
      - 假设内容数为 5，则按 U,D,L,R 顺序对实际存在的尖刺均分为 [2,1,2] 或 [2,0,1,2] 的压缩版；对三尖刺杯即 [2,1,2] 中对应的三个方向
      - 输出中杯内 5 个内容像素被清空，同时三条射线分别长 2/1/2
    key_difference:
      - 原题把“多候选”当成歧义并跳过
      - 变体把“多候选”升级为核心信号，要求并行弹射
  complexity_calibration:
    kept_small:
      - 不引入颜色映射、排序依赖或跨杯交互
      - 只新增“一个杯子可有多个合法尖刺”这一层
    kept_visible:
      - 每个生成杯固定拥有 3 个尖刺，因此 2-3 个样例就能稳定看到“一杯多射线”
      - 内容数量至少等于尖刺数，避免出现某些射线长度为 0 而看不出规则
  old_solver_assumptions_review:
    - candidate collection 仍可沿用：bbox 某边恰好 1 个边框像素且不在角落
    - flood validation 仍可沿用：从候选对侧 bbox 边 flood，寻找内容色
    - `if len(passing) != 1: continue` 依赖“唯一合法尖刺”旧前提；变体下必须删除
    - 原先 `len(content_cells)` 直接等于唯一射线长度；变体下必须先对去重后的总内容数做均分，再分别落到多条路径
    - 原先无须处理多条路径的覆盖冲突；变体下 generator 必须保证同杯多射线和跨杯射线都不重叠
  placement_legality:
    - cup bbox 仍保持至少 1 格间距
    - 每个尖刺向外到网格边缘的整条路径都必须为空，并与其他杯 bbox / 路径不相交
    - content_count 在放置后才决定，确保按均分后的每条射线长度都能完整落下，不依赖 overflow
  visual_observability:
    - 输入里同一个边框对象内部有一团单色内容
    - 输出里这团内容消失，并在同一对象的多个尖刺外侧同时出现同色直线
    - 这是肉眼可见的结构变化，不依赖颜色编号或隐藏排序
  cleanliness:
    - 内容只放在多侧 flood 交集的 strict interior，避免贴着 bbox 边造成“内容不在杯内”的观感
    - stray pixel 仍最多 1 个，且放在某个尖刺对应 mouth side 外部，不与主结构重叠
  new_solver_assumptions:
    - 一个真实杯会产生至少 1 个 passing candidate；generator 固定构造 3 个
    - 同一杯所有 passing candidate 会指向同一种内容色；generator 保证内容只用单色
    - 去重后的内容集合才代表需要均分的总量；generator 保证每个 passing candidate 至少能看到其中的一部分
"""

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


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2)

ARM_LEN_RANGE = (2, 2)
LOWER_RETURN_RANGE = (1, 1)
CONTENT_COUNT_MAX = 7

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 500
GEN_RETRIES_FALLBACK = 500

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
EDGE_ORDER = ("U", "D", "L", "R")
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood inside ``bbox``, seeded from non-border cells on one bbox edge."""
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


def _candidate_spikes(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> list[tuple[str, tuple[int, int]]]:
    """Candidate spikes in fixed U,D,L,R order."""
    r0, c0, r1, c1 = bbox
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    edges = (
        ("U", [p for p in border_set if p[0] == r0]),
        ("D", [p for p in border_set if p[0] == r1]),
        ("L", [p for p in border_set if p[1] == c0]),
        ("R", [p for p in border_set if p[1] == c1]),
    )
    return [
        (d, cells[0]) for d, cells in edges
        if len(cells) == 1 and cells[0] not in corners
    ]


def _split_evenly(total: int, parts: int) -> list[int]:
    """Deterministic near-even split; remainders go to earlier directions in U,D,L,R order."""
    base = total // parts
    extra = total % parts
    return [base + (i < extra) for i in range(parts)]


def _rotate_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction == "U":
        return patch.copy()
    if direction == "D":
        return np.flipud(patch).copy()
    if direction == "L":
        return np.rot90(patch, k=1).copy()
    if direction == "R":
        return np.rot90(patch, k=-1).copy()
    raise ValueError(direction)


def _analyze_patch_shape(patch: np.ndarray, border_color: int) -> dict | None:
    """Extract candidate spikes plus per-spike mouth regions from a border-only patch."""
    border_coords = np.argwhere(patch == border_color)
    if border_coords.size == 0:
        return None
    bbox = tuple(int(v) for v in bbox_coords(border_coords))
    border_set = {(int(r), int(c)) for r, c in border_coords}
    candidates = _candidate_spikes(border_set, bbox)
    if len(candidates) != 3:
        return None

    mouths_by_dir: dict[str, set[tuple[int, int]]] = {}
    for direction, _spike in candidates:
        mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[direction])
        if not mouth:
            return None
        mouths_by_dir[direction] = mouth

    r0, c0, r1, c1 = bbox
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    content_slots = sorted(
        set().union(*(mouths_by_dir[d] for d, _ in candidates)) - corners
    )
    if len(content_slots) < len(candidates):
        return None

    candidate_info = [
        {
            "dir": direction,
            "spike": spike,
            "mouth": sorted(mouths_by_dir[direction] - corners),
        }
        for direction, spike in candidates
    ]
    return {
        "bbox": bbox,
        "candidates": candidate_info,
        "content_slots": content_slots,
    }


def _build_canonical_trident(arm_len: int, lower_return: int, color: int) -> np.ndarray:
    """Canonical 3-spike cup: spikes on U/L/R, open on D."""
    assert arm_len >= 2
    assert 1 <= lower_return < arm_len
    h = arm_len + lower_return + 1
    w = 2 * arm_len + 1
    patch = np.zeros((h, w), dtype=int)

    patch[0, arm_len] = color
    for i in range(1, arm_len + 1):
        patch[i, arm_len - i] = color
        patch[i, arm_len + i] = color
    for j in range(1, lower_return + 1):
        patch[arm_len + j, j] = color
        patch[arm_len + j, w - 1 - j] = color
    return patch


def _sample_cup_patch(rng: np.random.Generator, border_color: int) -> tuple[np.ndarray, dict] | None:
    arm_len = int(rng.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1] + 1))
    lower_hi = min(LOWER_RETURN_RANGE[1], arm_len - 1)
    lower_return = int(rng.integers(LOWER_RETURN_RANGE[0], lower_hi + 1))
    direction = str(rng.choice(list(EDGE_ORDER)))

    patch = _rotate_patch(_build_canonical_trident(arm_len, lower_return, border_color), direction)
    shape = _analyze_patch_shape(patch, border_color)
    if shape is None:
        return None
    return patch, shape


def _path_from_spike(H: int, W: int, spike_abs: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR4[direction]
    rr, cc = spike_abs[0] + dr, spike_abs[1] + dc
    path = []
    while 0 <= rr < H and 0 <= cc < W:
        path.append((rr, cc))
        rr += dr
        cc += dc
    return path


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    candidates: list[dict],
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[list[tuple[int, int]]]] | None:
    """Place the cup so its bbox is separated and all spike rays are clear and non-empty."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        all_paths: list[list[tuple[int, int]]] = []
        used = set()
        ok = True
        for info in candidates:
            sr, sc = info["spike"]
            spike_abs = (r0 + sr, c0 + sc)
            path = _path_from_spike(H, W, spike_abs, info["dir"])
            if not path:
                ok = False
                break
            if any(occ_cup[r, c] or occ_path[r, c] or (r, c) in used for r, c in path):
                ok = False
                break
            used.update(path)
            all_paths.append(path)
        if ok:
            return r0, c0, all_paths
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Mask of cells lying beyond the mouth side for one chosen spike direction."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Multi-spike version of 3dc255db.

    Any bbox edge that has exactly one non-corner border pixel is a candidate spike.
    Every candidate whose opposite-side flood reaches content becomes an active spike.
    The cup's content is erased and split as evenly as possible across all active spikes,
    in U,D,L,R order.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            bbox = tuple(int(v) for v in bbox_coords(border_coords))
            border_set = {(int(r), int(c)) for r, c in border_coords}
            candidates = _candidate_spikes(border_set, bbox)
            if not candidates:
                continue

            passing = []
            for direction, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[direction])
                color_counts = Counter()
                for r, c in mouth:
                    value = int(g[r, c])
                    if value != bg and value != int(border_col):
                        color_counts[value] += 1
                if not color_counts:
                    continue
                content_col = int(color_counts.most_common(1)[0][0])
                content_cells = {
                    (r, c) for (r, c) in mouth
                    if int(g[r, c]) == content_col
                }
                if content_cells:
                    passing.append(
                        {
                            "dir": direction,
                            "spike": spike,
                            "content_col": content_col,
                            "content_cells": content_cells,
                        }
                    )

            if not passing:
                continue

            dominant_col = int(Counter(info["content_col"] for info in passing).most_common(1)[0][0])
            passing = [info for info in passing if info["content_col"] == dominant_col]
            if not passing:
                continue

            content_cells = sorted(set().union(*(info["content_cells"] for info in passing)))
            if not content_cells:
                continue

            for r, c in content_cells:
                out[r, c] = bg

            shares = _split_evenly(len(content_cells), len(passing))
            for share, info in zip(shares, passing):
                dr, dc = DIR4[info["dir"]]
                sr, sc = info["spike"]
                for step in range(1, share + 1):
                    rr = sr + dr * step
                    cc = sc + dc * step
                    if not (0 <= rr < H and 0 <= cc < W):
                        break
                    out[rr, cc] = dominant_col

    return out


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
    sampled = _sample_cup_patch(rng, border_color)
    if sampled is None:
        return None
    patch, shape = sampled
    placement = _try_place_cup(rng, H, W, patch, shape["candidates"], occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, all_paths = placement

    path_lengths = [len(path) for path in all_paths]
    spike_count = len(shape["candidates"])
    content_cap = min(CONTENT_COUNT_MAX, len(shape["content_slots"]))
    feasible_counts = [
        n for n in range(spike_count, content_cap + 1)
        if all(share <= path_len for share, path_len in zip(_split_evenly(n, spike_count), path_lengths))
    ]
    if not feasible_counts:
        return None
    content_count = int(rng.choice(feasible_counts))

    ph, pw = patch.shape
    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True

    shares = _split_evenly(content_count, spike_count)
    for share, path in zip(shares, all_paths):
        for r, c in path[:share]:
            canvas_out[r, c] = content_color
    for path in all_paths:
        for r, c in path:
            occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "content_slots_abs": [(r0 + r, c0 + c) for (r, c) in shape["content_slots"]],
        "candidate_mouths_abs": {
            info["dir"]: [(r0 + r, c0 + c) for (r, c) in info["mouth"]]
            for info in shape["candidates"]
        },
        "content_count": content_count,
        "candidate_dirs": [info["dir"] for info in shape["candidates"]],
        "path_cells": [cell for path in all_paths for cell in path],
    }


def _content_priority(cell: tuple[int, int], bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int, int]:
    """Prefer cells deeper inside the mouth corresponding to ``direction``."""
    r, c = cell
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) / 2.0
    mid_c = (c0 + c1) / 2.0
    if direction == "U":
        return (-r, abs(c - mid_c), abs(r - mid_r))
    if direction == "D":
        return (r, abs(c - mid_c), abs(r - mid_r))
    if direction == "L":
        return (-c, abs(r - mid_r), abs(c - mid_c))
    return (c, abs(r - mid_r), abs(c - mid_c))


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    for meta in cups_meta:
        count = meta["content_count"]
        chosen: list[tuple[int, int]] = []
        used = set()
        bbox = meta["bbox"]

        for direction in meta["candidate_dirs"]:
            mouth = meta["candidate_mouths_abs"][direction]
            ordered = sorted(mouth, key=lambda cell: _content_priority(cell, bbox, direction))
            pick = next((cell for cell in ordered if cell not in used), None)
            if pick is None:
                return False
            chosen.append(pick)
            used.add(pick)

        pool = [cell for cell in meta["content_slots_abs"] if cell not in used]
        extra = count - len(chosen)
        if extra > len(pool):
            return False
        if extra:
            idxs = rng.choice(len(pool), size=extra, replace=False)
            for idx in idxs:
                chosen.append(pool[int(idx)])

        for r, c in chosen:
            canvas_in[r, c] = meta["content_color"]
    return True


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
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]
    chosen_dir = str(rng.choice(meta["candidate_dirs"]))

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup in cups_meta:
        r0, c0, r1, c1 = cup["bbox"]
        forbidden[r0:r1 + 1, c0:c1 + 1] = True
        for r, c in cup["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], chosen_dir) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

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
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path
        )
        if meta is None:
            return None
        cups_meta.append(meta)

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
