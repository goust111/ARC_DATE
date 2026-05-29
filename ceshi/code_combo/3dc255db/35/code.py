# Combo variant of 3dc255db (index 35)
# combo_spec: D3/0+D14/2
# Name: 多色内容+双向弹射
#
# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 原题：每个杯子的 mouth 里只有 1 种内容色，全部内容都沿尾刺方向变成 1 条单色射线。
#    - 本融合版：每个杯子的 mouth 里恰好有 2 种内容色，并且它们沿 mouth 的可见阅读轴占据两个连续切片。
#      * tail 为 U/D 时，阅读轴是从左到右。
#      * tail 为 L/R 时，阅读轴是从上到下。
#    - 颜色到方向的规则不再靠颜色编号，而是靠位置：
#      * 处于较早切片的颜色向杯口侧弹射。
#      * 处于较晚切片的颜色向尾刺侧弹射。
#    - 同时保留 D14 的数量分配语义：总数 n 被拆成
#        tail_count = ceil(n / 2), mouth_count = floor(n / 2)
#      并由生成器保证“后切片颜色”恰好出现 tail_count 次，“前切片颜色”恰好出现 mouth_count 次。
#    - 例：tail=U，mouth 内左切片有 2 个蓝点，右切片有 3 个红点。
#      原题输出：只能得到 5 个同色点朝上。
#      融合后输出：3 个红点从尾刺向上，2 个蓝点从杯口向下；杯内 5 个点全部擦除。
#
# b) 复杂度校准
#    - 每杯只用 2 种内容色，避免 3 色以上时“哪种颜色去哪个方向”难以从少量样例归纳。
#    - 颜色到方向的映射只依赖位置切片，且每杯都重复同一规则。
#    - 总内容数优先取 3..7，使双向和奇偶分配都更明显；若空间不足才退化到 2。
#
# c) 隐式假设审查
#    - 原 solver 的 `Counter(...).most_common(1)` 假设 mouth 单色；这里必须改为收集全部内容色并排序。
#    - 原 solver 的“只擦除一种颜色”不再成立；这里必须擦除 mouth 内两种颜色的全部像素。
#    - 原 solver 只画 1 条尾刺射线；这里必须额外求 mouth 侧锚点并画第 2 条射线。
#    - D14 单变体默认“两侧同色”；这里不成立，颜色由切片顺序决定。
#    - 新 solver 还隐含假设：mouth 中恰有 2 个位置上可分离的颜色组，且其计数分别等于
#      floor(n/2) 与 ceil(n/2)。生成器明确按此构造。
#
# d) 放置合法性
#    - 继续采用 output-first：先放杯框和两条目标射线，再回填输入内容。
#    - 杯 bbox 之间保持至少 1 格间距。
#    - 尾刺射线与杯口射线都必须完整可见且互不与其他杯/路径重叠。
#    - stray pixel 仍最多 1 个，且不能落入任何 cup bbox 或任一预留射线上。
#
# e) 视觉可观察性
#    - 人类可同时看到两个稳定线索：
#      * input 中两种颜色分处 mouth 的前/后切片；
#      * output 中两条不同方向的单色射线分别继承这两种颜色。
#    - 规则不依赖颜色值大小，因此不需要知道 ARC 颜色编号。
#
# f) 可视化整洁度
#    - 每杯只产生两种内容色与两条直线，避免同一区域颜色过多。
#    - 两条射线都在杯外直行，路径预留避免与其他对象缠绕。
#
# g) 变化的可推断性
#    - 颜色变化不是任意映射，而是“前切片颜色 -> 杯口侧；后切片颜色 -> 尾刺侧”。
#    - D14 的数量规则由两组大小直接支撑：前组 smaller/equal，后组 larger/equal，奇数时尾刺侧多 1。
#
# 新 solver 假设与生成器保证对照
# - `main()` 假设每个被处理的 cup 只有 1 个合法 tail 候选；生成器沿用原题单尾刺构造。
# - `main()` 假设 mouth 侧锚点可由 spike 轴线和 mouth flood 唯一近似定位；生成器保证该口沿位置存在。
# - `main()` 假设 mouth 中恰有 2 个按阅读轴分离的颜色组；生成器用连续切片构造并避免第三种色混入。
# - `main()` 假设两组大小等于 floor/ceil 分裂；生成器在选组大小时显式满足。

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


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
CONTENT_COUNT_MIN_VISIBLE = 3
GLOBAL_CONTENT_POOL_SIZE = 4

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """Flood within bbox from one edge, stopping at border cells."""
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


def _split_content_count(n: int) -> tuple[int, int]:
    """Return (tail_count, mouth_count); odd totals favor the tail side."""
    return (n + 1) // 2, n // 2


def _mouth_projection(cell: tuple[int, int], tail_dir: str) -> int:
    r, c = cell
    return int(c if tail_dir in ("U", "D") else r)


def _content_groups_in_mouth(
    g: np.ndarray,
    mouth: set[tuple[int, int]],
    bg: int,
    border_col: int,
    tail_dir: str,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Return mouth content grouped by color and ordered by visible mouth reading axis."""
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
        return (
            min(projections),
            sum(projections) / len(projections),
            min(other_axis) * 100 + color,
        )

    return sorted(by_color.items(), key=order_key)


def _mouth_anchor_from_spike(
    spike: tuple[int, int],
    bbox: tuple[int, int, int, int],
    tail_dir: str,
    mouth_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find the mouth-edge cell opposite the spike, preferring exact axial alignment."""
    sr, sc = spike
    r0, c0, r1, c1 = bbox
    if tail_dir == "U":
        preferred = (r1, sc)
        edge_cells = [p for p in mouth_cells if p[0] == r1]
        key = lambda p: (abs(p[1] - sc), p[1])
    elif tail_dir == "D":
        preferred = (r0, sc)
        edge_cells = [p for p in mouth_cells if p[0] == r0]
        key = lambda p: (abs(p[1] - sc), p[1])
    elif tail_dir == "L":
        preferred = (sr, c1)
        edge_cells = [p for p in mouth_cells if p[1] == c1]
        key = lambda p: (abs(p[0] - sr), p[0])
    else:
        preferred = (sr, c0)
        edge_cells = [p for p in mouth_cells if p[1] == c0]
        key = lambda p: (abs(p[0] - sr), p[0])

    if preferred in mouth_cells:
        return preferred
    if not edge_cells:
        return None
    return min(edge_cells, key=key)


def _ray_cells_from_anchor(
    anchor: tuple[int, int],
    direction: str,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Cells strictly outside the anchor, continuing to the grid edge."""
    dr, dc = DIR4[direction]
    rr, cc = anchor[0] + dr, anchor[1] + dc
    out: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule for 3dc255db.

    Each cup contains exactly two content colors arranged in ordered mouth slices.
    The earlier slice color ejects toward the mouth side, the later slice color ejects
    toward the tail side, and the two sides receive floor/ceil halves of the total count.
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
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                groups = _content_groups_in_mouth(g, mouth, bg, int(border_col), tail_dir)
                if len(groups) != 2:
                    continue
                total_count = sum(len(cells) for _, cells in groups)
                tail_count, mouth_count = _split_content_count(total_count)
                if len(groups[0][1]) != mouth_count or len(groups[1][1]) != tail_count:
                    continue
                mouth_anchor = _mouth_anchor_from_spike(spike, bbox, tail_dir, mouth)
                if mouth_anchor is None:
                    continue
                passing.append(
                    (
                        tail_dir,
                        spike,
                        mouth_anchor,
                        int(groups[0][0]),
                        int(groups[1][0]),
                        [cell for _color, cells in groups for cell in cells],
                        tail_count,
                        mouth_count,
                    )
                )

            if len(passing) != 1:
                continue

            tail_dir, spike, mouth_anchor, mouth_color, tail_color, content_cells, tail_count, mouth_count = passing[0]
            for r, c in content_cells:
                out[r, c] = bg

            tail_cells = _ray_cells_from_anchor(spike, tail_dir, H, W)
            mouth_cells = _ray_cells_from_anchor(mouth_anchor, OPPOSITE[tail_dir], H, W)

            for rr, cc in tail_cells[:tail_count]:
                out[rr, cc] = tail_color
            for rr, cc in mouth_cells[:mouth_count]:
                out[rr, cc] = mouth_color

    return out


def _build_canonical_cup(
    L: int,
    R: int,
    hL: int,
    hR: int,
    color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP."""
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


def _mouth_interior_coords(
    patch: np.ndarray,
    border_color: int,
    opposite_side: str,
) -> list[tuple[int, int]]:
    """Flood-fill the interior of the cup patch from the given bbox edge."""
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
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical (tail=U) patch, spike, and mouth coords to target direction."""
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


def _zone_order_key(cell: tuple[int, int], direction: str) -> tuple[int, int, int]:
    return (_mouth_projection(cell, direction), int(cell[0]), int(cell[1]))


def _choose_directional_content_groups(
    rng: np.random.Generator,
    mouth: list[tuple[int, int]],
    direction: str,
    available_colors: list[int],
    tail_path_len: int,
    mouth_path_len: int,
) -> list[dict] | None:
    """Choose two ordered mouth slices whose counts match the bidirectional split."""
    unique_proj = sorted({_mouth_projection(cell, direction) for cell in mouth})
    if len(unique_proj) < 2 or len(available_colors) < 2:
        return None

    cut_choices = list(range(1, len(unique_proj)))
    rng.shuffle(cut_choices)
    for cut_idx in cut_choices:
        mouth_proj = set(unique_proj[:cut_idx])
        tail_proj = set(unique_proj[cut_idx:])
        mouth_zone = sorted(
            [cell for cell in mouth if _mouth_projection(cell, direction) in mouth_proj],
            key=lambda rc: _zone_order_key(rc, direction),
        )
        tail_zone = sorted(
            [cell for cell in mouth if _mouth_projection(cell, direction) in tail_proj],
            key=lambda rc: _zone_order_key(rc, direction),
        )
        if not mouth_zone or not tail_zone:
            continue

        hard_max = min(CONTENT_COUNT_MAX, len(mouth), tail_path_len + mouth_path_len)
        feasible_totals = []
        for total_count in range(CONTENT_COUNT_MIN_VISIBLE, hard_max + 1):
            tail_count, mouth_count = _split_content_count(total_count)
            if mouth_count > mouth_path_len or tail_count > tail_path_len:
                continue
            if mouth_count > len(mouth_zone) or tail_count > len(tail_zone):
                continue
            feasible_totals.append(total_count)
        if not feasible_totals and CONTENT_COUNT_MIN_VISIBLE > 2:
            for total_count in range(2, hard_max + 1):
                tail_count, mouth_count = _split_content_count(total_count)
                if mouth_count > mouth_path_len or tail_count > tail_path_len:
                    continue
                if mouth_count > len(mouth_zone) or tail_count > len(tail_zone):
                    continue
                feasible_totals.append(total_count)
        if not feasible_totals:
            continue

        total_count = int(rng.choice(np.array(feasible_totals, dtype=int)))
        tail_count, mouth_count = _split_content_count(total_count)
        colors = [int(v) for v in rng.choice(np.array(available_colors, dtype=int), size=2, replace=False)]
        return [
            {"color": colors[0], "count": mouth_count, "zone": mouth_zone},
            {"color": colors[1], "count": tail_count, "zone": tail_zone},
        ]
    return None


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth_anchor_rel: tuple[int, int],
    direction: str,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a placement whose tail ray and mouth ray are both clear."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=STRUCT8)
    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    mouth_dir = OPPOSITE[direction]
    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        spike_abs = (r0 + spike[0], c0 + spike[1])
        mouth_anchor_abs = (r0 + mouth_anchor_rel[0], c0 + mouth_anchor_rel[1])
        tail_path = _ray_cells_from_anchor(spike_abs, direction, H, W)
        mouth_path = _ray_cells_from_anchor(mouth_anchor_abs, mouth_dir, H, W)
        if not tail_path or not mouth_path:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in tail_path):
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in mouth_path):
            continue
        return r0, c0, tail_path, mouth_path
    return None


def _stray_candidate_mask(
    H: int,
    W: int,
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> np.ndarray:
    """Cells on the mouth side of bbox, restricted to the transverse span."""
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
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    """Optionally place one isolated stray pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    if rng.random() < STRAY_USE_BORDER_PROB:
        stray_color = meta["border_color"]
    else:
        content_colors = [group["color"] for group in meta["content_groups"]]
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
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_palette: list[int],
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, place two directional color groups, and commit the output rays."""
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
    mouth_anchor_rel = _mouth_anchor_from_spike(spike, (0, 0, ph - 1, pw - 1), direction, set(mouth))
    if mouth_anchor_rel is None:
        return None

    placement = _try_place_cup(rng, H, W, patch, spike, mouth_anchor_rel, direction, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, tail_path, mouth_path = placement

    available_content_colors = [c for c in content_palette if c != border_color]
    content_groups = _choose_directional_content_groups(
        rng,
        mouth,
        direction,
        available_content_colors,
        len(tail_path),
        len(mouth_path),
    )
    if content_groups is None:
        return None

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True

    mouth_group, tail_group = content_groups
    for r, c in mouth_path[:int(mouth_group["count"])]:
        canvas_out[r, c] = int(mouth_group["color"])
    for r, c in tail_path[:int(tail_group["count"])]:
        canvas_out[r, c] = int(tail_group["color"])
    for r, c in tail_path:
        occ_path[r, c] = True
    for r, c in mouth_path:
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
        "path_cells": tail_path + mouth_path,
    }


def _scatter_content(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    cups_meta: list[dict],
) -> bool:
    """Scatter each color within its own mouth slice zone."""
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
    """Single end-to-end generation attempt using output-first verification."""
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
            rng,
            H,
            W,
            border_color,
            content_palette,
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
