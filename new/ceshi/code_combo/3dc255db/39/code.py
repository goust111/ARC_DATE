# Combo variant of 3dc255db (index 39)
# combo_spec: D9/0+D10/0+D13/1
# Name: 矩形框架+杯口宽度射线+面积决定长度
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 杯体不再是 Y/漏斗，而是 3 边封闭、1 边开口的矩形框；尖刺定义为封闭边中心像素。
#    - 以 canonical tail=U 的 4x5 杯为例：
#        border = 顶边 5 格 + 左右边各 4 格，底边中间 3 格开口；
#        mouth = rows 1..3, cols 1..3，共 9 格；
#        opening width = 3。
#      输入若只放 3 个红色内容点：
#        原题输出：从尖刺上方射出 3 个单像素红点；
#        D10 输出：射出长度 9 的 1px 红线；
#        D13 输出：射出 3 层、每层宽 3 的红色带；
#        本融合输出：射出 9 层、每层宽 3，得到 3x9 的红色矩形块。
#    - 因而融合后的两条几何测量链必须同时成立：
#        1. 宽度来自杯口 opening span；
#        2. 长度来自 mouth area，而不是内容数量。
# b) 复杂度校准
#    - 若允许 mouth_width=1，D13 会退化回原题；因此只生成 opening width 为 3 或 5 的矩形杯。
#    - 若允许 content_count 接近 mouth_area，D10 会弱化；因此强制 content_count 至少比 area 少 2，
#      且不超过 area 的一半。
#    - 若允许 area 太大，小网格里会被边界截断，看不出“长度=面积”；因此只使用 3 种 canonical 尺寸：
#        3x5 -> area 6, width 3
#        4x5 -> area 9, width 3
#        3x7 -> area 10, width 5
#      这样既有少量离散档位便于归纳，又能保证视觉差异明显。
# c) 隐式假设审查
#    - 原 main() 的“某 bbox 边恰有 1 个 border cell 即 spike”前提被 D9 破坏，必须改成：
#      组件必须精确匹配某个朝向的 3 边矩形框，spike 是封闭边中心。
#    - 原 main() 的 `len(content_cells)` 同时负责擦除与定长；D10 破坏了“定长=内容数”，
#      因此擦除仍按真实 content_cells，长度改为 `len(mouth)`。
#    - 原 main() 默认每步只写 1 个像素；D13 破坏了这一点，因此每个步长要写整条 band。
#    - 新 solver 新增假设：
#        1. 杯体严格是 3 边矩形框；
#        2. 封闭边长度为奇数，中心无歧义；
#        3. opening 在开口边上形成单一连续区间。
#      generator 会显式保证这些条件。
# d) 放置合法性
#    - 杯 bbox 之间保持至少 1 格间距。
#    - `occ_path` 预留的是“整条厚轨迹到边界”的所有格，而不只是可见输出矩形，保证输入里该路径全是 bg。
#    - 杯 bbox 不能覆盖已有路径，路径也不能穿过已有 cup bbox 或路径。
# e) 视觉可观察性
#    - 人类能直接看到杯体变成规则 U/C 形矩形框。
#    - 输出块的厚度与开口宽度同步变化，长度又明显大于输入内容数，并与腔体面积对应。
#    - 3 种离散尺寸会重复出现：同样 width=3 时 area 可为 6 或 9；width=5 时 area 为 10。
#      这样能从少量样例中分离“宽度控制厚度”和“面积控制长度”两条规则。
# f) 可视化整洁度
#    - 轨迹始终是单色矩形块，不与杯体重叠。
#    - stray pixel 仍只放在某个杯的 mouth 一侧，并避开 cup bbox 与预留路径，避免遮挡关键结构。
# g) 变化的可推断性
#    - 颜色规则未新增映射：内容色只负责决定输出色。
#    - 关键控制量都来自可见几何：矩形框尺寸、开口宽度、内部面积；不依赖颜色编号或不可见排序。

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood inside ``bbox`` seeded from non-border cells on one bbox side."""
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


def _rect_frame_cells(
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> set[tuple[int, int]]:
    """Exact 3-sided rectangular-frame cell set for a given tail direction."""
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


def _spike_from_bbox_center(
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> tuple[int, int] | None:
    """Center pixel of the closed side; None when that side length is even."""
    r0, c0, r1, c1 = bbox
    if tail_dir in ("U", "D"):
        closed_len = c1 - c0 + 1
        if closed_len % 2 == 0:
            return None
        return (r0 if tail_dir == "U" else r1, c0 + closed_len // 2)
    closed_len = r1 - r0 + 1
    if closed_len % 2 == 0:
        return None
    return (r0 + closed_len // 2, c0 if tail_dir == "L" else c1)


def _opening_edge_cells(
    mouth: set[tuple[int, int]] | list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    """Cells of the mouth that lie directly on the open bbox edge."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        return sorted((r, c) for r, c in mouth if r == r0)
    if side == "D":
        return sorted((r, c) for r, c in mouth if r == r1)
    if side == "L":
        return sorted((r, c) for r, c in mouth if c == c0)
    return sorted((r, c) for r, c in mouth if c == c1)


def _opening_span(
    edge_cells: list[tuple[int, int]],
    side: str,
) -> tuple[int, int] | None:
    """Inclusive span of the opening on the axis perpendicular to the tail direction."""
    if not edge_cells:
        return None
    vals = [c for _, c in edge_cells] if side in ("U", "D") else [r for r, _ in edge_cells]
    return min(vals), max(vals)


def _trajectory_bands(
    spike: tuple[int, int],
    direction: str,
    span: tuple[int, int],
    H: int,
    W: int,
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
            bands.append([(rr, x) for x in range(lo, hi + 1)])
        else:
            bands.append([(x, cc) for x in range(lo, hi + 1)])
        t += 1
    return bands


def _flatten_bands(bands: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for band in bands for cell in band]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - detect a 3-sided rectangular frame with one open side;
    - spike = center of the closed side;
    - erase the interior content pixels;
    - emit a rectangular block in the tail direction:
        width  = cup opening width
        length = mouth interior area
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 7:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
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

                mouth_side = OPPOSITE[tail_dir]
                mouth = _flood_from_bbox_side(border_set, bbox, mouth_side)
                if len(mouth) < 3:
                    continue
                opening = _opening_edge_cells(mouth, bbox, mouth_side)
                span = _opening_span(opening, mouth_side)
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
                passing.append((tail_dir, spike, span, content_col, content_cells, len(mouth)))

            if len(passing) != 1:
                continue

            tail_dir, spike, span, content_col, content_cells, mouth_area = passing[0]
            for r, c in content_cells:
                out[r, c] = bg
            for band in _trajectory_bands(spike, tail_dir, span, H, W)[:mouth_area]:
                for r, c in band:
                    out[r, c] = content_col

    return out


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

# Original task bbox envelope: rows 3..8, cols 3..7.
# Canonical patch is tail=U; L/R swaps the axes after rotation.
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

# Restrict to a small shape vocabulary so the fused rule is learnable and visible.
# Each tuple is (canonical_height, canonical_width).
RECT_SHAPES = (
    (3, 5),  # mouth width 3, area 6
    (4, 5),  # mouth width 3, area 9
    (3, 7),  # mouth width 5, area 10
)

CONTENT_COUNT_MAX = 7
MIN_AREA_COUNT_GAP = 2

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400
# ==================================


def _build_canonical_cup(height: int, width: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a 3-sided rectangular cup with tail pointing UP and opening at the bottom."""
    assert height >= 3 and width >= 5 and width % 2 == 1
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[:, 0] = color
    patch[:, width - 1] = color
    return patch, (0, width // 2)


def _mouth_interior_coords(
    patch: np.ndarray,
    border_color: int,
    opposite_side: str,
) -> list[tuple[int, int]]:
    """Flood-fill the mouth interior from the open side of the cup patch."""
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
        for dr, dc in DIR4.values():
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
    """Rotate canonical tail=U patch, spike, and mouth coordinates to the target direction."""
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


def _sample_cup_shape(
    rng: np.random.Generator,
    direction: str,
) -> tuple[int, int] | None:
    """Pick one of the allowed rectangular-frame sizes that fits the bbox limits."""
    shapes = []
    for height, width in RECT_SHAPES:
        if direction in ("U", "D"):
            if height <= CUP_BBOX_MAX_UD[0] and width <= CUP_BBOX_MAX_UD[1]:
                shapes.append((height, width))
        else:
            if width <= CUP_BBOX_MAX_LR[0] and height <= CUP_BBOX_MAX_LR[1]:
                shapes.append((height, width))
    if not shapes:
        return None
    return shapes[int(rng.integers(0, len(shapes)))]


def _trajectory_bands_from_opening(
    spike_abs: tuple[int, int],
    direction: str,
    opening_abs: list[tuple[int, int]],
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    """Absolute thick trajectory bands implied by the opening width."""
    side = OPPOSITE[direction]
    span = _opening_span(opening_abs, side)
    if span is None:
        return []
    return _trajectory_bands(spike_abs, direction, span, H, W)


def _choose_content_count(rng: np.random.Generator, mouth_area: int) -> int | None:
    """Pick sparse content so the area-driven length stays visually distinct."""
    hi = min(CONTENT_COUNT_MAX, mouth_area - MIN_AREA_COUNT_GAP, mouth_area // 2)
    if hi < 1:
        return None
    return int(rng.integers(1, hi + 1))


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    opening: list[tuple[int, int]],
    mouth_area: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup with a fully visible thick, area-length trajectory and a 1-cell bbox gap."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords)

    for r0, c0 in coords:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        opening_abs = [(r0 + r, c0 + c) for (r, c) in opening]
        path_bands = _trajectory_bands_from_opening((sr_abs, sc_abs), direction, opening_abs, H, W)
        if len(path_bands) < mouth_area:
            continue

        path_cells = _flatten_bands(path_bands)
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        eject_cells = _flatten_bands(path_bands[:mouth_area])
        return r0, c0, path_cells, eject_cells

    return None


def _stray_candidate_mask(
    H: int,
    W: int,
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, restricted to its perpendicular extent."""
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
    """Optionally place one isolated stray pixel on the mouth side of one cup."""
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
    """Build one cup, reserve its thick path to the edge, and commit it to the canvases."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    height, width = shape

    patch_canon, spike_canon = _build_canonical_cup(height, width, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    mouth_area = len(mouth_canon)
    content_count = _choose_content_count(rng, mouth_area)
    if content_count is None:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    opening_side = OPPOSITE[direction]
    opening = _opening_edge_cells(mouth, (0, 0, ph - 1, pw - 1), opening_side)
    span = _opening_span(opening, opening_side)
    if span is None or (span[1] - span[0] + 1) < 2:
        return None

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, opening, mouth_area, occ_cup, occ_path,
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
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "mouth_area": mouth_area,
        "mouth_width": span[1] - span[0] + 1,
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    cups_meta: list[dict],
) -> bool:
    """Scatter each cup's sparse content pixels inside its mouth interior."""
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
    """Single output-first generation attempt with oracle verification against ``main``."""
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
