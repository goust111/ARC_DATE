# Variant of 3dc255db
# Dimension: D1 — 路径类型 (path)
# Change: straight_ray — 内容像素从尖刺沿尾部方向发射为一条直线... → 扩散填充
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: 从尖刺外侧的第一个格子开始，沿 tail_dir 单向直线放置 N 个内容像素。
#     variant: 从同一个外侧起点开始，不再沿单一路径前进，而是在杯体 bbox 外部的背景区域里
#       做 4-邻域 BFS，按固定的 tail-relative 邻居顺序扩散，取前 N 个访问到的背景格着色。
#   concrete_trace:
#     example:
#       tail_dir: U
#       spike: (4, 5)
#       seed: (3, 5)
#       content_count: 5
#       original_output_cells:
#         - (3, 5)
#         - (2, 5)
#         - (1, 5)
#         - (0, 5)
#       variant_output_cells:
#         - (3, 5)
#         - (2, 5)
#         - (3, 4)
#         - (3, 6)
#         - (1, 5)
#       difference: 原题是 1 维射线；变体是从同一起点长出带侧枝的 BFS 云团。
#   complexity_calibration:
#     - 仅改 D1，杯体检测、嘴部内容识别、擦除内容等逻辑保持原样。
#     - 为避免退化成肉眼看不出的“短直线”，生成器要求 content_count >= 4。
#     - 同时要求前 4 个 BFS 填充格里至少有 2 个偏离主轴，确保样例中能看出扩散而非射线。
#   implicit_assumptions_review:
#     - 原 solver 的 `for t in range(...)` 依赖“输出路径是一条全序直线”；变体下不成立，必须换成队列。
#     - 原 solver 的 `if out_of_bounds: break` 对单射线安全；变体若某一支撞边后整体 break 会丢失其他分支。
#     - 原 solver 无需记录 visited；BFS 若不记 visited，会重复入队并可能覆盖成环。
#     - 原 solver 允许只看 tail_dir 一个方向；变体必须保留 tail_dir 仅用于定位 seed 和决定 BFS 的并列扩展顺序。
#     - 原逻辑默认“内容输出区是一条清空的 path”；变体下输出区是一个连通区域，生成器必须预留整块可扩散背景。
#   placement_legality:
#     - 每个杯体 bbox 仍保持至少 1 格间隔。
#     - BFS 只允许走背景格，且不得回到当前杯体 bbox 内，避免内容从外部绕回嘴部。
#     - 生成器顺序放置杯体，并为每个杯体预先计算实际可达的 BFS 填充格；只有可达数足够时才接受放置。
#     - 不同杯体的填充区互不重叠，并与已有填充区保持 1 格缓冲，降低视觉粘连与颜色歧义。
#   visual_observability:
#     - 观察者仍可通过“杯体尖刺 + 内容被擦除 + 外部新图案”识别同一对象关系。
#     - 但输出新图案不再是细长单线，而是以 seed 为根的 T/十字状扩散前沿，2-3 组样例即可看出差异。
#   visual_cleanliness:
#     - 不允许扩散区进入当前杯体 bbox。
#     - stray pixel 仍放在嘴部一侧，但要避开所有填充区及其邻域，避免和 BFS 云团混在一起。
#   color_inference:
#     - 颜色规则不变：内容颜色原样搬运到外部扩散区，没有新颜色映射。
#   generator_guarantees_for_new_solver:
#     - 每个被接受的杯体都有唯一 tail 候选。
#     - spike 外侧 seed 必为背景且可达至少 4 个 BFS 格。
#     - 前 4 个 BFS 格具有明显 off-axis spread，因此同一输入上新旧规则输出不同。

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
LEFT_TURN = {"U": "L", "L": "D", "D": "R", "R": "U"}
RIGHT_TURN = {"U": "R", "R": "D", "D": "L", "L": "U"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _expansion_dirs(tail_dir: str) -> list[tuple[int, int]]:
    """Fixed neighbor priority for BFS: forward first, then the two sideways branches, then back."""
    order = [tail_dir, LEFT_TURN[tail_dir], RIGHT_TURN[tail_dir], OPPOSITE[tail_dir]]
    return [DIR4[d] for d in order]


def _bfs_fill_cells(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    tail_dir: str,
    blocked_mask: np.ndarray,
    limit: int | None = None,
) -> list[tuple[int, int]]:
    """Return BFS visitation order from the spike's outward seed.

    Traversal is limited to background-eligible cells outside the current cup bbox.
    The queue order is deterministic and tail-relative so the first few cells still
    visually emerge from the spike before fanning sideways.
    """
    H, W = shape
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    if not (0 <= sr < H and 0 <= sc < W):
        return []
    if r0 <= sr <= r1 and c0 <= sc <= c1:
        return []
    if bool(blocked_mask[sr, sc]):
        return []

    out: list[tuple[int, int]] = []
    visited = {seed}
    q = deque([seed])
    nbrs = _expansion_dirs(tail_dir)

    while q and (limit is None or len(out) < limit):
        r, c = q.popleft()
        out.append((r, c))
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                continue
            if bool(blocked_mask[nr, nc]):
                continue
            visited.add((nr, nc))
            q.append((nr, nc))
    return out


def _has_off_axis_spread(fill_cells: list[tuple[int, int]], seed: tuple[int, int], tail_dir: str) -> bool:
    """Require at least two painted cells off the main tail axis so the BFS is visually distinct."""
    sr, sc = seed
    if tail_dir in ("U", "D"):
        off_axis = sum(c != sc for _, c in fill_cells)
    else:
        off_axis = sum(r != sr for r, _ in fill_cells)
    return off_axis >= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Detect each cup exactly as in the original task.
    - Erase its mouth content.
    - Start at the cell immediately outside the spike in tail_dir.
    - Perform 4-neighbor BFS over background cells outside that cup bbox.
    - Paint the first N visited cells, where N = number of erased content pixels.
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
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIR4[tail_dir]
            seed = (spike[0] + dr, spike[1] + dc)
            blocked_mask = (g != bg)
            fill_cells = _bfs_fill_cells((H, W), bbox, seed, tail_dir, blocked_mask, limit=len(content_cells))
            for r, c in fill_cells:
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

CONTENT_COUNT_MIN = 4
CONTENT_COUNT_MAX = 7

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 500
GEN_RETRIES_FALLBACK = 500
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
    """Flood-fill the interior of the cup patch from cells on the given bbox edge, stopping at
    the border. Returns the sorted list of interior coordinates (seeds included)."""
    h, w = patch.shape
    border_mask = (patch == border_color)
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
        for dr, dc in DIR4.values():
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
    """Pick (L, R, hL, hR) parameters for a canonical cup, clamped so the patch fits the bbox spec."""
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
    occ_fill: np.ndarray,
) -> tuple[int, int, tuple[int, int], list[tuple[int, int]]] | None:
    """Find a placement whose outward seed supports a visible BFS fill area."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    occ_fill_dilated = ndimage.binary_dilation(occ_fill, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_fill_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue

        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        seed = (r0 + spike[0] + dr, c0 + spike[1] + dc)
        blocked = occ_cup_dilated | occ_fill_dilated
        fill_cells = _bfs_fill_cells((H, W), bbox, seed, direction, blocked, limit=CONTENT_COUNT_MAX)
        if len(fill_cells) < CONTENT_COUNT_MIN:
            continue
        if not _has_off_axis_spread(fill_cells[:CONTENT_COUNT_MIN], seed, direction):
            continue
        return r0, c0, seed, fill_cells
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Boolean mask of cells on the mouth side of ``bbox`` (opposite the tail direction)."""
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
    """Optionally place an isolated stray pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    fill_mask = np.zeros((H, W), dtype=bool)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["fill_cells"]:
            fill_mask[r, c] = True
    forbidden |= ndimage.binary_dilation(fill_mask, structure=_STRUCT8)

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
    occ_fill: np.ndarray,
) -> dict | None:
    """Build one cup, place it, and commit its BFS fill to the output canvas."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < CONTENT_COUNT_MIN:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    placement = _try_place_cup(rng, H, W, patch, spike, direction, occ_cup, occ_fill)
    if placement is None:
        return None
    r0, c0, seed, fill_order = placement

    max_count = min(CONTENT_COUNT_MAX, len(mouth), len(fill_order))
    if max_count < CONTENT_COUNT_MIN:
        return None
    content_count = int(rng.integers(CONTENT_COUNT_MIN, max_count + 1))
    fill_cells = fill_order[:content_count]
    if not _has_off_axis_spread(fill_cells, seed, direction):
        return None

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in fill_cells:
        canvas_out[r, c] = content_color
        occ_fill[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "fill_cells": fill_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's content_count pixels at random inside its mouth interior."""
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
    occ_fill = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if len(palette) < 2:
            return None
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        palette.append(content_color)

        meta = _add_cup(
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_fill,
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
