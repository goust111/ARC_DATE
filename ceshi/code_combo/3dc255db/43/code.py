# Combo variant of 3dc255db (index 43)
# combo_spec: D3/0+D12/1+D15/0
# Name: 多色内容+多干扰像素+共享边框色
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original:
#       - 每个杯框有自己独立的边框色。
#       - 每个杯 mouth 里只有一种内容色。
#       - 网格里最多只有 0-1 个孤立干扰像素。
#     combo:
#       - 所有杯共享同一个边框色，因此不能再靠颜色区分不同杯，只能靠 8 连通分量拆分。
#       - 每个杯 mouth 里有 2-3 种内容色，输出射线按 mouth 可见顺序分成连续色块。
#       - 网格里固定放置 2-4 个孤立干扰像素，颜色复用共享边框色或某个内容色。
#   concretization_and_trace:
#     - 例子：两个杯都用蓝色边框。左杯 tail=U，mouth 内左列有 2 个红像素、右列有 3 个黄像素；
#       右杯 tail=R，mouth 内上方有 1 个绿像素、下方有 2 个红像素。网格别处再放 3 个孤立像素，
#       其中一个也是蓝色，另两个复用红/黄。
#     - 原题 solver 会按颜色遍历，默认“一个边框色=一个杯”，并在 mouth 中选出现次数最多的单色；
#       这在融合后失效，因为两个杯共享边框色，而且 mouth 里不止一种内容色。
#     - 新 solver 的流程：
#       1. 对每个非背景色做 8 连通分量分解；大且像杯框的分量才作为候选杯。
#       2. 对每个候选杯仍按“某 bbox 边仅有 1 个非角边框像素”找 tail 候选。
#       3. 从对侧 flood 到 mouth，只统计 bbox 内、非背景、非边框的内容像素。
#       4. 按 mouth 顺序分组：tail 为 U/D 时按列从左到右；tail 为 L/R 时按行从上到下。
#       5. 擦除 mouth 内全部内容像素，并沿 tail 方向依次写出分组展开后的颜色序列。
#     - 干扰像素不会被误当作内容，因为它们都在所有 cup bbox 之外；它们只保留在输出中不动。
#   complexity_calibration:
#     - 每杯只用 2-3 种内容色，避免颜色爆炸，但仍明显区别于原题单色内容。
#     - 内容排序只依赖人眼可见的位置顺序，不依赖颜色编号。
#     - 每张图固定 2-3 个杯，避免 1 杯时“共享边框色”这个变化不可见。
#     - 每张图固定 2-4 个干扰像素，并保证至少一个复用边框色、至少一个复用内容色。
#   implicit_assumption_audit_main:
#     - 原代码 `for border_col in colors_present(...)` 默认“一种边框色对应一组杯”；融合后必须改成
#       “遍历所有非背景色的 8 连通分量”，否则多个同色杯会被混在一起。
#     - 原代码 `cnt.most_common(1)` 默认 mouth 内只有一种有效内容色；融合后必须改成多色分组。
#     - 原代码只擦除 `content_col` 对应的像素；融合后必须擦除 mouth 内全部非背景非边框像素。
#     - 原代码整条射线用一个颜色；融合后必须逐格写入展开后的颜色序列。
#     - 原代码把小同色分量自动忽略；这个前提对共享边框色+边框色干扰像素仍成立，因为干扰像素是单点分量。
#   placement_legality:
#     - 继续使用 output-first：先放杯框和目标射线，再回填输入内容，最后用 `main(input)` 反验。
#     - 杯 bbox 之间保持至少 1 个细胞的 8 邻域间隔，防止共享边框色下两个杯被连成一个分量。
#     - 射线路径必须完整可见，长度至少覆盖整条多色序列，避免被截断后丢失分组信息。
#     - 所有干扰像素放在所有 bbox 和射线路径之外，且彼此也保持孤立。
#   visual_observability:
#     - 人类能同时看到三件事：
#       1. 两个或三个同色杯框其实是不同对象，因为它们空间分离且各自弹射。
#       2. cup 内不同颜色占据不同 mouth 切片，输出变成同顺序的连续色块。
#       3. 网格里还有若干孤立像素完全不参与变换，因此需要区分“杯内内容”和“外部噪声”。
#   visual_cleanliness:
#     - 干扰像素不进入杯 bbox，不压到 tail path，也不贴边框，避免视觉叠层混乱。
#     - 内容色池限制在少量复用颜色，降低杂乱度。
#   color_inference:
#     - 颜色不会被重映射；输出只是把杯内已有颜色按位置顺序排成一条线。
#   new_solver_assumptions_and_generator_guarantees:
#     - solver 假设真实杯框是某个非背景色的单个 8 连通分量，且大小 >= 5；
#       generator 用 1-cell gap 保证这一点。
#     - solver 假设同一杯不同颜色在 mouth-order 轴上占据不重叠连续切片；
#       generator 通过按列/按行切片分区来保证这一点。
#     - solver 假设所有干扰像素都在 cup bbox 之外；generator 强制如此构造。

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


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-neighbor flood inside bbox from one bbox edge, stopping at border cells."""
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
    """Group mouth content by color and order those groups by visible mouth direction."""
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
        transverse = [cell[0] if tail_dir in ("U", "D") else cell[1] for cell in cells]
        return min(projections), sum(projections) / len(projections), min(transverse) * 100 + color

    return sorted(by_color.items(), key=order_key)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shared-border multicolor cup variant.

    Each cup is a same-colored frame component. Content inside the cup mouth may use 2-3 colors.
    The output erases all cup-interior content and ejects a straight tail line whose colors are
    grouped by mouth order: left->right for vertical tails, top->bottom for horizontal tails.
    External stray pixels remain unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

    for color in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(color), connectivity=8):
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
                groups = _content_groups_in_mouth(g, mouth, bg, int(color), d)
                if groups:
                    passing.append((d, spike, groups))

            if len(passing) != 1:
                continue
            tail_dir, spike, groups = passing[0]

            content_cells = [cell for _content_color, cells in groups for cell in cells]
            for r, c in content_cells:
                out[r, c] = bg

            eject_sequence = [content_color for content_color, cells in groups for _ in range(len(cells))]
            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            for t, content_color in enumerate(eject_sequence, start=1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = int(content_color)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
CONTENT_GROUP_COUNT_RANGE = (2, 3)
GLOBAL_CONTENT_POOL_SIZE = 4
STRAY_COUNT_RANGE = (2, 4)

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP (spike at row 0)."""
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
    """Flood-fill the cup interior from the mouth side."""
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
    """Rotate canonical tail-UP data into target tail direction."""
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
    """Pick a canonical cup shape that fits bbox limits after rotation."""
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
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, len(values)), size=groups - 1, replace=False))
    out: list[list[int]] = []
    start = 0
    for stop in [*cuts, len(values)]:
        out.append(list(values[start:stop]))
        start = stop
    return out


def _choose_content_groups(
    rng: np.random.Generator, mouth: list[tuple[int, int]], direction: str, available_colors: list[int],
) -> list[dict] | None:
    """Create 2-3 ordered color groups backed by disjoint mouth slices."""
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
    min_total = min(max_total, max(num_groups + 1, 4))
    if min_total < num_groups or max_total < num_groups:
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
    """Place a cup whose full ejection sequence fits to the grid edge."""
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
    """Mask of cells on the mouth side of a cup bbox."""
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


def _build_forbidden_mask(canvas_in: np.ndarray, cups_meta: list[dict]) -> np.ndarray:
    """Cells strays must avoid: anything occupied, any cup bbox, any tail path, and neighbors."""
    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=STRUCT8)
    for meta in cups_meta:
        r0, c0, r1, c1 = meta["bbox"]
        forbidden[r0:r1 + 1, c0:c1 + 1] = True
        for r, c in meta["path_cells"]:
            forbidden[r, c] = True
    return forbidden


def _place_multiple_strays(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict], border_color: int,
) -> bool:
    """Place 2-4 isolated strays, mixing shared-border and content colors."""
    H, W = canvas_in.shape
    used_content_colors = sorted({
        int(group["color"])
        for meta in cups_meta
        for group in meta["content_groups"]
    })
    if not used_content_colors:
        return False

    target = int(rng.integers(STRAY_COUNT_RANGE[0], STRAY_COUNT_RANGE[1] + 1))
    color_plan = [int(border_color), int(rng.choice(np.array(used_content_colors, dtype=int)))]
    color_pool = [int(border_color), *used_content_colors]
    while len(color_plan) < target:
        color_plan.append(int(rng.choice(np.array(color_pool, dtype=int))))

    for stray_color in color_plan:
        forbidden = _build_forbidden_mask(canvas_in, cups_meta)
        masks = []
        cup_indices = list(range(len(cups_meta)))
        rng.shuffle(cup_indices)
        for idx in cup_indices:
            meta = cups_meta[idx]
            mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
            if mask.any():
                masks.append(mask)
        if not masks:
            return False

        candidate_mask = np.zeros((H, W), dtype=bool)
        for mask in masks:
            candidate_mask |= mask
        candidates = np.argwhere(candidate_mask)
        if candidates.size == 0:
            return False
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        canvas_in[r, c] = stray_color
        canvas_out[r, c] = stray_color

    return True


def _add_cup(
    rng: np.random.Generator, H: int, W: int, border_color: int, content_palette: list[int],
    canvas_in: np.ndarray, canvas_out: np.ndarray, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, choose ordered multicolor content groups, and commit it."""
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
        "border_color": int(border_color),
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "content_groups": content_groups_abs,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each color group inside its own mouth slice zone."""
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
    """Single verified generation attempt."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    border_color = int(rng.integers(1, 10))
    content_palette = [int(v) for v in rng.choice(
        np.array([c for c in range(1, 10) if c != border_color], dtype=int),
        size=GLOBAL_CONTENT_POOL_SIZE,
        replace=False,
    )]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
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
    if not _place_multiple_strays(rng, canvas_in, canvas_out, cups_meta, border_color):
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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
