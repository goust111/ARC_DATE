# Combo variant of 3dc255db (index 20)
# combo_spec: D6/3+D8/2
# Name: 碰撞变色+链式反应
#
# Design analysis
# a) 具体化与行为追踪
# - 原题: 每个杯只要口内有内容，就各自独立擦除内容并沿尾刺方向画一直线。
# - 本融合版: 输入里只有一个杯带有 starter 像素: 它位于尾刺外第 1 格，颜色等于该杯内容色。
#   只有 starter 杯先弹射；若它的射线在步数耗尽前撞到另一杯的边框，则该命中边框像素
#   被改成射线颜色，同时被撞杯入队并继续按自己的尾刺方向弹射。未被 starter 或碰撞触发的杯
#   保持原状，口内内容不会被擦除。
# - 具体小例子: A 杯带 starter，A 向右射 3 步，第 3 步撞到 B 杯边框。
#   原题会把 A/B/C 都独立发射；D8 单变体会让 A 触发 B 但不会把撞点改色；本题则是
#   A 的第 3 步把 B 的那一格边框改成 A 的内容色，并触发 B。若场上还有 C 且没被命中，
#   C 完全不动。
# b) 复杂度校准
# - 生成器固定做 3 杯场景: 1 个 starter 射手 + 1 个被撞触发杯 + 1 个未触发旁观杯。
#   这样样例里始终同时出现“碰撞改色”“链式触发”“未触发杯保持不动”三种关键信号。
# - 只保留一条显式链 A->B，避免多分支链造成视觉杂乱；第三个杯只用来显示选择性触发。
# c) 隐式假设审查
# - 原 solver 假设“每杯都应弹射”已失效，必须改成 starter/queue 驱动。
# - 原 solver 假设“射线前方只会是背景”已失效，命中别杯边框时必须先改色再停止。
# - 原 solver 对所有杯一律擦内容已失效，现在只能对已激活杯擦除；未触发杯必须保留内容。
# - 杯检测仍依赖输入中的完整边框和口内主导内容色，因此生成器保证所有杯在输入里都保留口内内容。
# d) 放置合法性
# - 所有杯 bbox 之间仍保持至少 1 格间隔。
# - B 杯先放成“可清晰向外弹射”的目标杯；A 杯再放成会首先命中 B 边框的射手；
#   C 杯最后放在不接触前两条可见路径的位置。
# - B 与 C 的未来射线路径都被保留为占位，后放对象不能覆盖这些路径。
# e) 视觉可观察性
# - starter 像素是唯一启动提示，解题者可直接看出不是所有杯同步发射。
# - 输出里 A 的射线在 B 处停下且 B 的某个边框像素变色，然后只有 B 继续发射；C 保持不变。
# - 这些现象都由可见的几何与颜色关系决定，不依赖颜色数值大小或扫描顺序。
# f) 可视化整洁度
# - 不放 stray pixel，避免与 starter 混淆。
# - A 的射线只命中一次并停止；B 的射线通向空白区域；C 与可见路径保持分离。
# g) 变化可推断性
# - 边框色对每杯唯一，内容色也取与各边框不同的少量颜色。
# - 碰撞改色总发生在“射线终点正好落在另一杯边框”这个强结构线索上，人类可从少量样例归纳。

from __future__ import annotations

from collections import Counter, deque

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (3,)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7

GEN_RETRIES_PRIMARY = 500
GEN_RETRIES_FALLBACK = 500
CUP_BUILD_RETRIES = 80

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> set[tuple[int, int]]:
    """4-neighbor flood inside bbox from one bbox edge, blocked by border cells."""
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


def _detect_cups(g: np.ndarray) -> tuple[int, list[dict]]:
    """Recover actionable cups from the immutable input grid."""
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
            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            starter_cell = (sr + dr, sc + dc)
            starter = (
                0 <= starter_cell[0] < g.shape[0]
                and 0 <= starter_cell[1] < g.shape[1]
                and int(g[starter_cell[0], starter_cell[1]]) == content_col
            )

            cups.append(
                {
                    "border_color": int(border_col),
                    "bbox": bbox,
                    "border_set": border_set,
                    "tail_dir": tail_dir,
                    "spike": (int(sr), int(sc)),
                    "content_col": content_col,
                    "content_cells": [(int(r), int(c)) for r, c in content_cells],
                    "starter": starter,
                }
            )

    return bg, cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Exactly the visibly armed cup(s) start: a starter is a same-colored pixel one step beyond the spike.
    - An active cup erases only its own source content, then emits along its tail direction.
    - If the ray hits another cup border, that struck border pixel is recolored to the ray color,
      the ray stops there, and the hit cup is activated.
    - Cups that are never armed or hit remain unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, cups = _detect_cups(g)
    out = g.copy()

    border_lookup: dict[tuple[int, int], int] = {}
    for idx, cup in enumerate(cups):
        for cell in cup["border_set"]:
            border_lookup[cell] = idx

    activated: set[int] = set()
    queue: deque[int] = deque()
    for idx, cup in enumerate(cups):
        if cup["starter"]:
            activated.add(idx)
            queue.append(idx)

    while queue:
        idx = queue.popleft()
        cup = cups[idx]

        for r, c in cup["content_cells"]:
            out[r, c] = bg

        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        for step in range(1, len(cup["content_cells"]) + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < H and 0 <= cc < W):
                break
            hit_idx = border_lookup.get((rr, cc), -1)
            if hit_idx >= 0 and hit_idx != idx:
                out[rr, cc] = cup["content_col"]
                if hit_idx not in activated:
                    activated.add(hit_idx)
                    queue.append(hit_idx)
                break
            out[rr, cc] = cup["content_col"]

    return out


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing up."""
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
    """Flood-fill the patch interior from the mouth side."""
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
    return sorted((int(r), int(c)) for r, c in visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical (tail=U) patch to the target tail direction."""
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
    """Pick canonical cup parameters that fit after rotation."""
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


def _sample_blueprint(rng: np.random.Generator, border_color: int, content_color: int) -> dict | None:
    """Sample one cup shape before placement."""
    for _ in range(CUP_BUILD_RETRIES):
        direction = str(rng.choice(["U", "D", "L", "R"]))
        shape = _sample_cup_shape(rng, direction)
        if shape is None:
            continue
        L_arm, R_arm, hL, hR = shape
        patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
        mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
        if len(mouth_canon) < 3:
            continue
        patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
        return {
            "border_color": border_color,
            "content_color": content_color,
            "direction": direction,
            "patch": patch,
            "spike": spike,
            "mouth": mouth,
        }
    return None


def _path_to_edge(H: int, W: int, spike_abs: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR4[direction]
    rr, cc = spike_abs[0] + dr, spike_abs[1] + dc
    path = []
    while 0 <= rr < H and 0 <= cc < W:
        path.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return path


def _bbox_has_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    return b1[2] < b2[0] - 1 or b2[2] < b1[0] - 1 or b1[3] < b2[1] - 1 or b2[3] < b1[1] - 1


def _all_bboxes_have_gap(bbox: tuple[int, int, int, int], placed: list[dict]) -> bool:
    return all(_bbox_has_gap(bbox, p["bbox"]) for p in placed)


def _cell_in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    return bbox[0] <= r <= bbox[2] and bbox[1] <= c <= bbox[3]


def _occupied_bbox_cell(cell: tuple[int, int], placed: list[dict]) -> bool:
    return any(_cell_in_bbox(cell, p["bbox"]) for p in placed)


def _reserved_cells(placed: list[dict]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for p in placed:
        out.update(p["reserved_path"])
    return out


def _patch_border_cells(patch: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    coords = np.argwhere(patch != 0)
    return {(r0 + int(r), c0 + int(c)) for r, c in coords}


def _commit_placed(
    blueprint: dict,
    r0: int,
    c0: int,
    path_full: list[tuple[int, int]],
    content_count: int,
    reserve_full_path: bool,
) -> dict:
    patch = blueprint["patch"]
    ph, pw = patch.shape
    spike_abs = (r0 + blueprint["spike"][0], c0 + blueprint["spike"][1])
    border_abs = _patch_border_cells(patch, r0, c0)
    active_shot = path_full[:content_count]
    return {
        **blueprint,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "r0": r0,
        "c0": c0,
        "spike_abs": spike_abs,
        "border_abs": border_abs,
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in blueprint["mouth"]],
        "path_full": path_full,
        "content_count": content_count,
        "active_shot": active_shot,
        "reserved_path": set(path_full if reserve_full_path else active_shot),
    }


def _try_place_clear_cup(
    rng: np.random.Generator, H: int, W: int, blueprint: dict, placed: list[dict]
) -> dict | None:
    """Place a cup whose full ray path to the edge stays unobstructed."""
    patch = blueprint["patch"]
    ph, pw = patch.shape
    reserved = _reserved_cells(placed)
    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if not _all_bboxes_have_gap(bbox, placed):
            continue
        border_abs = _patch_border_cells(patch, r0, c0)
        if border_abs & reserved:
            continue
        spike_abs = (r0 + blueprint["spike"][0], c0 + blueprint["spike"][1])
        path_full = _path_to_edge(H, W, spike_abs, blueprint["direction"])
        if not path_full:
            continue
        if any(_occupied_bbox_cell(cell, placed) for cell in path_full):
            continue
        if any(cell in reserved for cell in path_full):
            continue
        max_count = min(CONTENT_COUNT_MAX, len(blueprint["mouth"]), len(path_full))
        if max_count < 1:
            continue
        content_count = int(rng.integers(1, max_count + 1))
        return _commit_placed(blueprint, r0, c0, path_full, content_count, reserve_full_path=True)
    return None


def _try_place_collision_shooter(
    rng: np.random.Generator, H: int, W: int, blueprint: dict, target: dict, placed: list[dict]
) -> dict | None:
    """Place the starter shooter so its first border hit is on the target cup."""
    patch = blueprint["patch"]
    ph, pw = patch.shape
    reserved = _reserved_cells(placed)
    max_count = min(CONTENT_COUNT_MAX, len(blueprint["mouth"]))
    if max_count < 2:
        return None

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)
    for r0, c0 in offsets:
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if not _all_bboxes_have_gap(bbox, placed):
            continue
        border_abs = _patch_border_cells(patch, r0, c0)
        if border_abs & reserved:
            continue
        spike_abs = (r0 + blueprint["spike"][0], c0 + blueprint["spike"][1])
        path_full = _path_to_edge(H, W, spike_abs, blueprint["direction"])
        if not path_full:
            continue

        hit_idx = None
        valid = True
        for idx, cell in enumerate(path_full[:max_count], start=1):
            if cell in target["border_abs"]:
                hit_idx = idx
                break
            if cell in reserved or _occupied_bbox_cell(cell, placed):
                valid = False
                break
        if not valid or hit_idx is None or hit_idx < 2:
            continue
        return _commit_placed(blueprint, r0, c0, path_full, hit_idx, reserve_full_path=False)
    return None


def _render_expected(H: int, W: int, bg: int, canvas_in: np.ndarray, cups: list[dict]) -> np.ndarray:
    """Oracle renderer for the fused rule."""
    out = canvas_in.copy()
    border_lookup: dict[tuple[int, int], int] = {}
    for idx, cup in enumerate(cups):
        for cell in cup["border_abs"]:
            border_lookup[cell] = idx

    activated: set[int] = set()
    queue: deque[int] = deque()
    for idx, cup in enumerate(cups):
        dr, dc = DIR4[cup["direction"]]
        rr = cup["spike_abs"][0] + dr
        cc = cup["spike_abs"][1] + dc
        if 0 <= rr < H and 0 <= cc < W and int(canvas_in[rr, cc]) == int(cup["content_color"]):
            activated.add(idx)
            queue.append(idx)

    while queue:
        idx = queue.popleft()
        cup = cups[idx]
        for r, c in cup["content_cells_abs"]:
            out[r, c] = bg
        dr, dc = DIR4[cup["direction"]]
        sr, sc = cup["spike_abs"]
        for step in range(1, cup["content_count"] + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < H and 0 <= cc < W):
                break
            hit_idx = border_lookup.get((rr, cc), -1)
            if hit_idx >= 0 and hit_idx != idx:
                out[rr, cc] = cup["content_color"]
                if hit_idx not in activated:
                    activated.add(hit_idx)
                    queue.append(hit_idx)
                break
            out[rr, cc] = cup["content_color"]
    return out


def _render_original_baseline(H: int, W: int, bg: int, canvas_in: np.ndarray, cups: list[dict]) -> np.ndarray:
    """Reference output of the original independent-ejection task on the same generated scene."""
    out = canvas_in.copy()
    for cup in cups:
        for r, c in cup["content_cells_abs"]:
            out[r, c] = bg
        dr, dc = DIR4[cup["direction"]]
        sr, sc = cup["spike_abs"]
        for step in range(1, cup["content_count"] + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = cup["content_color"]
    return out


def _starter_count(cups: list[dict], canvas_in: np.ndarray) -> int:
    total = 0
    H, W = canvas_in.shape
    for cup in cups:
        dr, dc = DIR4[cup["direction"]]
        rr = cup["spike_abs"][0] + dr
        cc = cup["spike_abs"][1] + dc
        if 0 <= rr < H and 0 <= cc < W and int(canvas_in[rr, cc]) == int(cup["content_color"]):
            total += 1
    return total


def _sample_color_pairs(rng: np.random.Generator) -> list[tuple[int, int]]:
    """Use distinct border/content colors so the collision recolor is always visible."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_colors = colors[:3]
    content_colors = colors[3:6]
    return list(zip(border_colors, content_colors))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    """Build one scene: shooter A hits B, while cup C stays inert."""
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    color_pairs = _sample_color_pairs(rng)

    target_bp = _sample_blueprint(rng, *color_pairs[0])
    shooter_bp = _sample_blueprint(rng, *color_pairs[1])
    inert_bp = _sample_blueprint(rng, *color_pairs[2])
    if target_bp is None or shooter_bp is None or inert_bp is None:
        return None

    target = _try_place_clear_cup(rng, H, W, target_bp, placed=[])
    if target is None:
        return None

    shooter = _try_place_collision_shooter(rng, H, W, shooter_bp, target=target, placed=[target])
    if shooter is None:
        return None

    inert = _try_place_clear_cup(rng, H, W, inert_bp, placed=[target, shooter])
    if inert is None:
        return None

    cups = [shooter, target, inert]
    canvas_in = np.zeros((H, W), dtype=int)
    for cup in cups:
        for r, c in cup["border_abs"]:
            canvas_in[r, c] = cup["border_color"]

    for cup in cups:
        if len(cup["mouth_abs"]) < cup["content_count"]:
            return None
        idxs = rng.choice(len(cup["mouth_abs"]), size=cup["content_count"], replace=False)
        cup["content_cells_abs"] = [cup["mouth_abs"][int(i)] for i in idxs]
        for r, c in cup["content_cells_abs"]:
            canvas_in[r, c] = cup["content_color"]

    if not shooter["active_shot"]:
        return None
    starter_r, starter_c = shooter["active_shot"][0]
    if canvas_in[starter_r, starter_c] != 0:
        return None
    canvas_in[starter_r, starter_c] = shooter["content_color"]

    if _starter_count(cups, canvas_in) != 1:
        return None
    _, detected_cups = _detect_cups(canvas_in)
    if len(detected_cups) != 3 or sum(int(c["starter"]) for c in detected_cups) != 1:
        return None

    expected = _render_expected(H, W, 0, canvas_in, cups)
    baseline = _render_original_baseline(H, W, 0, canvas_in, cups)

    # Generator-side invariants: exactly one collision recolor, and the inert cup keeps its content.
    collision_pixels = sum(1 for cell in target["border_abs"] if expected[cell] != target["border_color"])
    if collision_pixels != 1:
        return None
    if any(expected[r, c] != inert["content_color"] for r, c in inert["content_cells_abs"]):
        return None
    if any(expected[r, c] != 0 for r, c in shooter["content_cells_abs"]):
        return None
    if any(expected[r, c] != 0 for r, c in target["content_cells_abs"]):
        return None
    if np.array_equal(expected, baseline):
        return None

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
