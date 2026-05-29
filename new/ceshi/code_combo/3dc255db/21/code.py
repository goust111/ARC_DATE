# Combo variant of 3dc255db (index 21)
# combo_spec: D6/3+D8/2+D11/0
# Name: 碰撞变色+链式反应+路径穿越

from __future__ import annotations

from collections import Counter, deque

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


"""
design_analysis:
  a_concrete_behavior_trace:
    - 原题里每个杯只要口内有内容就独立发射；发射时先擦掉内容，再在尾刺外侧连续写 len(content) 个像素。
    - 本融合版保留“杯检测 / 尾刺判定 / 口内内容数决定发射额度 / 已激活杯会清空口内内容”这几件事。
    - 新增一枚显式 starter：只有某个杯在尾刺外第 1 格已有同色像素时，它才是初始激活杯。
    - 激活杯沿尾刺方向逐格前进；若遇到背景，则正常写出内容色并消耗 1 个额度。
    - 若遇到别的杯的边框，则该边框像素改成当前射线颜色，但这一步不消耗额度，射线继续往前；第一次撞到某个未激活杯时，会把该杯加入队列，形成链式反应。
    - 于是“穿越”不是无事发生：边框会变色，目标杯也会被启动，而且射线还会在杯体之后继续画出后续像素。
    - 小例子：A 向右、额度 3。其路径依次看到 [bg, B左边框, B内部bg, B右边框, bg]。
      原题会在前三格写色，既不会区分 B 的边框，也没有启动关系。
      融合版会在第1格写色，第2格把 B 边框改成 A 的颜色并激活 B，第3格写色，第4格再把 B 另一侧边框改色，第5格写色，然后 B 再从自己的尾刺继续发射。
  b_complexity_calibration:
    - 生成器只做线性链，不做分叉传播；这样 2-3 个杯就能稳定展示“starter -> 撞杯 -> 续射 -> 被撞杯再发射”。
    - 每个样例都强制至少一条真实穿越：命中外杯边框后，当前射线还必须再画出至少 1 个非边框像素。
    - 不再放 stray pixel，避免与 starter 混淆；可见符号只剩杯框、口内内容、starter、穿越后的射线和变色边框。
  c_hidden_assumptions_audit_main:
    - 原 solver 的“所有杯都独立发射”失效，因此改成先检测 starter，再用队列传播。
    - 原 solver 的 `for t in range(len(content))` 默认每一步都消耗额度；融合后只有真正画在非边框单元上的步骤才消耗额度。
    - 原 solver 可直接覆写路径上的任意像素；融合后必须先判断该格是否属于别杯边框，因为边框命中要改色并触发，但不能终止前进。
    - 原 solver 边画边依赖 out 不会影响后续识别；融合后仍必须只从 immutable input 抽取杯信息，因此先整体检测 cups，再模拟。
    - 新增假设是“初始发射杯可见且唯一”，这里用 starter 单像素编码，generator 会保证 starter_count == 1。
  d_placement_legality:
    - 杯 bbox 仍保持至少 1 格间隔，且彼此不重叠。
    - 只允许前一杯的 full ray 穿过当前杯；更早的射线不能再碰到后面的杯，避免出现非线性多重触发。
    - 不同杯真正画到非边框上的射线路径互不重叠；后放的杯也不会把自己的边框放到旧的非边框射线上。
    - 被前驱射线穿过的杯，会把这些路径单元从 mouth 可放内容集合里剔除，保证输入合法、输出也不含歧义覆盖。
  e_visual_observability:
    - 人类可见的顺序线索是 starter，不依赖扫描顺序或颜色数值大小。
    - 输出里既能看到边框变色，也能看到射线穿过杯体后继续在后方出现，因此可和“撞到就停止”或“路径完全无障碍”区分开。
    - 每个样例都至少有一次新激活和一次穿越后续射，所以 2-4 组样例足以归纳规则。
  f_visual_cleanliness:
    - 只让相邻链路发生交互，避免多条旧射线同时穿过同一新杯。
    - 只允许边框单元承担“被撞改色”职责；口内内容仍只在杯口区域，starter 只出现一次，视觉角色清晰。
  g_color_inference:
    - 颜色规则不做数值映射；被撞边框直接变成撞它的那条射线颜色。
    - 生成器保证每条链路上 shooter_content != target_border，因此“边框变色”肉眼一定可见。
  new_solver_assumptions_and_generator_guarantees:
    - solver 假设不同杯边框在输入中可独立检测，generator 用唯一边框色和 bbox gap 保证。
    - solver 假设 starter 唯一，generator 显式校验。
    - solver 假设需要传播的碰撞都来自真实 cup border，generator 只通过放置杯边框来制造命中，不用隐藏标记。
"""


# ==== Generator / solver configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7

GEN_RETRIES_PRIMARY = 500
GEN_RETRIES_FALLBACK = 500
CUP_BUILD_RETRIES = 100
MIN_CHAIN_RAY_LEN = 4
# ==========================================


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> set[tuple[int, int]]:
    """4-connected flood inside bbox from one bbox edge, blocked by border cells."""
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


def _detect_cups(grid: np.ndarray) -> tuple[int, list[dict]]:
    """Recover all cups from immutable input using the original tail-disambiguation logic."""
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
                content_cells = [(int(r), int(c)) for r, c in mouth if int(g[r, c]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue

            tail_dir, spike, content_col, content_cells = passing[0]
            dr, dc = DIR4[tail_dir]
            starter_r = int(spike[0] + dr)
            starter_c = int(spike[1] + dc)
            starter = (
                0 <= starter_r < g.shape[0]
                and 0 <= starter_c < g.shape[1]
                and int(g[starter_r, starter_c]) == content_col
            )
            cups.append(
                {
                    "border_color": int(border_col),
                    "bbox": bbox,
                    "border_set": border_set,
                    "tail_dir": tail_dir,
                    "spike": (int(spike[0]), int(spike[1])),
                    "content_col": content_col,
                    "content_cells": content_cells,
                    "starter": starter,
                }
            )

    return bg, cups


def _simulate_combo(grid: np.ndarray, bg: int, cups: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    """Simulate the combo rule from already-detected cups and collect observability metrics."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    border_lookup: dict[tuple[int, int], int] = {}
    for idx, cup in enumerate(cups):
        for cell in cup["border_set"]:
            border_lookup[cell] = idx

    starters = [idx for idx, cup in enumerate(cups) if cup["starter"]]
    activated = set(starters)
    queue: deque[int] = deque(starters)
    metrics = {
        "starter_count": len(starters),
        "activated_count": 0,
        "chain_triggers": 0,
        "recolored_border_cells": 0,
        "post_hit_paint_cells": 0,
    }

    while queue:
        idx = queue.popleft()
        cup = cups[idx]

        for r, c in cup["content_cells"]:
            out[r, c] = bg

        dr, dc = DIR4[cup["tail_dir"]]
        rr = cup["spike"][0] + dr
        cc = cup["spike"][1] + dc
        placed = 0
        remaining = len(cup["content_cells"])
        saw_foreign_hit = False
        while placed < remaining and 0 <= rr < H and 0 <= cc < W:
            hit_idx = border_lookup.get((rr, cc), -1)
            if hit_idx >= 0:
                if hit_idx != idx:
                    out[rr, cc] = cup["content_col"]
                    metrics["recolored_border_cells"] += 1
                    saw_foreign_hit = True
                    if hit_idx not in activated:
                        activated.add(hit_idx)
                        queue.append(hit_idx)
                        metrics["chain_triggers"] += 1
                rr += dr
                cc += dc
                continue

            out[rr, cc] = cup["content_col"]
            placed += 1
            if saw_foreign_hit:
                metrics["post_hit_paint_cells"] += 1
            rr += dr
            cc += dc

    metrics["activated_count"] = len(activated)
    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Only cups with a same-colored starter pixel one step beyond the spike are initially active.
    - An active cup erases its mouth content and emits outward.
    - Non-border cells consume one content unit and are painted with the content color.
    - Foreign border cells are recolored, do not consume a unit, and activate the struck cup.
    - The ray continues through/after those cups, producing a chain reaction with path traversal.
    """
    g = np.array(grid, dtype=int)
    bg, cups = _detect_cups(g)
    out, _ = _simulate_combo(g, bg, cups)
    return out


def combo_behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Public diagnostic helper used for non-degeneracy checks during development/testing."""
    g = np.array(grid, dtype=int)
    bg, cups = _detect_cups(g)
    _, metrics = _simulate_combo(g, bg, cups)
    return metrics


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical Y/funnel cup with tail pointing UP."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1 : L + hL + 1, 0] = color
    if hR:
        patch[R + 1 : R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(
    patch: np.ndarray, border_color: int, opposite_side: str
) -> list[tuple[int, int]]:
    """Flood-fill the interior of a cup patch from its mouth side."""
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
    return sorted((int(r), int(c)) for r, c in visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical tail-UP cup to the requested tail direction."""
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
    """Sample canonical cup parameters within bbox limits after rotation."""
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


def _build_blueprint(rng: np.random.Generator, border_color: int, content_color: int) -> dict | None:
    """Sample one cup geometry before placement."""
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
    border_rel = {(int(r), int(c)) for r, c in np.argwhere(patch == border_color)}
    edge_border_rel = {
        "U": [(r, c) for (r, c) in border_rel if r == 0],
        "D": [(r, c) for (r, c) in border_rel if r == ph - 1],
        "L": [(r, c) for (r, c) in border_rel if c == 0],
        "R": [(r, c) for (r, c) in border_rel if c == pw - 1],
    }
    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "patch": patch,
        "shape": (ph, pw),
        "spike_rel": (int(spike[0]), int(spike[1])),
        "mouth_rel": [(int(r), int(c)) for (r, c) in mouth],
        "border_rel": border_rel,
        "edge_border_rel": edge_border_rel,
    }


def _tail_cells_to_edge(H: int, W: int, start: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    """All cells strictly beyond a spike until the grid edge."""
    dr, dc = DIR4[direction]
    rr = start[0] + dr
    cc = start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def _bbox_clear(r0: int, c0: int, ph: int, pw: int, occ_bbox: np.ndarray) -> bool:
    """Require a 1-cell gap between the candidate bbox and already placed bboxes."""
    occ_dilated = ndimage.binary_dilation(occ_bbox, structure=_STRUCT8)
    return not occ_dilated[r0 : r0 + ph, c0 : c0 + pw].any()


def _instantiate_blueprint(blueprint: dict, r0: int, c0: int, H: int, W: int) -> dict:
    """Materialize a blueprint at one absolute location."""
    ph, pw = blueprint["shape"]
    border_abs = {(r0 + r, c0 + c) for (r, c) in blueprint["border_rel"]}
    mouth_abs = [(r0 + r, c0 + c) for (r, c) in blueprint["mouth_rel"]]
    spike_abs = (r0 + blueprint["spike_rel"][0], c0 + blueprint["spike_rel"][1])
    bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
    bbox_cells = {(r, c) for r in range(r0, r0 + ph) for c in range(c0, c0 + pw)}
    return {
        **blueprint,
        "bbox": bbox,
        "bbox_cells": bbox_cells,
        "border_abs": border_abs,
        "mouth_abs": mouth_abs,
        "spike_abs": spike_abs,
        "full_ray": _tail_cells_to_edge(H, W, spike_abs, blueprint["direction"]),
        "available_mouth_so_far": list(mouth_abs),
    }


def _trace_shot(
    full_ray: list[tuple[int, int]], foreign_border: set[tuple[int, int]], content_count: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return painted non-border cells and struck foreign-border cells for one shot."""
    paint_cells: list[tuple[int, int]] = []
    hit_cells: list[tuple[int, int]] = []
    for cell in full_ray:
        if len(paint_cells) >= content_count:
            break
        if cell in foreign_border:
            hit_cells.append(cell)
            continue
        paint_cells.append(cell)
    return paint_cells, hit_cells


def _min_count_to_cross(full_ray: list[tuple[int, int]], foreign_border: set[tuple[int, int]]) -> int | None:
    """Minimum content count that guarantees a border hit and at least one painted cell after it."""
    before_hit = 0
    seen_hit = False
    for cell in full_ray:
        if cell in foreign_border:
            seen_hit = True
            continue
        if seen_hit:
            return before_hit + 1
        before_hit += 1
    return None


def _sample_color_pairs(rng: np.random.Generator, num_cups: int) -> list[tuple[int, int]]:
    """Choose unique border colors and content colors that keep recolors visible."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_colors = colors[:num_cups]
    pairs: list[tuple[int, int]] = []
    for i in range(num_cups):
        forbidden = {border_colors[i]}
        if i < num_cups - 1:
            forbidden.add(border_colors[i + 1])
        content_choices = [c for c in range(1, 10) if c not in forbidden]
        pairs.append((border_colors[i], int(rng.choice(content_choices))))
    return pairs


def _try_place_first_cup(
    rng: np.random.Generator, H: int, W: int, blueprint: dict, occ_bbox: np.ndarray, needs_successor: bool
) -> dict | None:
    """Place the first cup; it only needs space plus a sufficiently long ray to edge."""
    ph, pw = blueprint["shape"]
    coords = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords)
    min_len = MIN_CHAIN_RAY_LEN if needs_successor else 1
    for r0, c0 in coords:
        if not _bbox_clear(r0, c0, ph, pw, occ_bbox):
            continue
        cup = _instantiate_blueprint(blueprint, r0, c0, H, W)
        if len(cup["full_ray"]) < min_len:
            continue
        return cup
    return None


def _try_place_chain_target(
    rng: np.random.Generator,
    H: int,
    W: int,
    blueprint: dict,
    prev_cup: dict,
    placed: list[dict],
    reserved_paint_cells: set[tuple[int, int]],
    needs_successor: bool,
) -> tuple[dict, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place the next cup so the previous cup crosses it, recolors it, and activates it."""
    ph, pw = blueprint["shape"]
    incoming_side = OPPOSITE[prev_cup["direction"]]
    older_border_union = set().union(*(cup["border_abs"] for cup in placed)) if placed else set()
    older_full_ray_union = set().union(*(cup["full_ray"] for cup in placed)) if placed else set()
    older_full_rays_except_prev = set().union(*(cup["full_ray"] for cup in placed[:-1])) if len(placed) > 1 else set()
    prev_cap = min(CONTENT_COUNT_MAX, len(prev_cup["available_mouth_so_far"]))
    if prev_cap < 1:
        return None

    coords = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    occ_bbox = np.zeros((H, W), dtype=bool)
    for cup in placed:
        br0, bc0, br1, bc1 = cup["bbox"]
        occ_bbox[br0 : br1 + 1, bc0 : bc1 + 1] = True

    rng.shuffle(coords)
    for r0, c0 in coords:
        if not _bbox_clear(r0, c0, ph, pw, occ_bbox):
            continue

        cup = _instantiate_blueprint(blueprint, r0, c0, H, W)
        if cup["border_abs"] & reserved_paint_cells:
            continue
        if cup["bbox_cells"] & older_full_rays_except_prev:
            continue
        if set(cup["full_ray"]) & (older_border_union | older_full_ray_union | reserved_paint_cells):
            continue
        if len(cup["full_ray"]) < (MIN_CHAIN_RAY_LEN if needs_successor else 1):
            continue

        first_hit_idx = None
        first_hit_cell = None
        for idx, cell in enumerate(prev_cup["full_ray"], start=1):
            if cell in cup["border_abs"]:
                first_hit_idx = idx
                first_hit_cell = cell
                break
        if first_hit_idx is None or first_hit_idx < 2:
            continue

        incoming_abs = {(r0 + r, c0 + c) for (r, c) in blueprint["edge_border_rel"][incoming_side]}
        if first_hit_cell not in incoming_abs:
            continue

        min_cross = _min_count_to_cross(prev_cup["full_ray"], cup["border_abs"])
        if min_cross is None or min_cross > prev_cap:
            continue

        prev_count = int(min_cross)
        prev_paint_cells, prev_hit_cells = _trace_shot(prev_cup["full_ray"], cup["border_abs"], prev_count)
        prev_paint_set = set(prev_paint_cells)
        if not prev_hit_cells:
            continue
        if prev_paint_set & reserved_paint_cells:
            continue
        if prev_paint_set & older_border_union:
            continue
        if prev_paint_set & older_full_rays_except_prev:
            continue
        if prev_paint_set & set(cup["full_ray"]):
            continue

        available_now = [cell for cell in cup["mouth_abs"] if cell not in reserved_paint_cells and cell not in prev_paint_set]
        if len(available_now) < (2 if needs_successor else 1):
            continue
        cup["available_mouth_so_far"] = available_now
        return cup, prev_count, prev_paint_cells, prev_hit_cells

    return None


def _finalize_last_cup(
    rng: np.random.Generator,
    last_cup: dict,
    placed: list[dict],
    reserved_paint_cells: set[tuple[int, int]],
) -> bool:
    """Choose the last cup's own ejection path; it has no successor to hit."""
    older_border_union = set().union(*(cup["border_abs"] for cup in placed[:-1])) if len(placed) > 1 else set()
    older_full_ray_union = set().union(*(cup["full_ray"] for cup in placed[:-1])) if len(placed) > 1 else set()
    if set(last_cup["full_ray"]) & (older_border_union | older_full_ray_union | reserved_paint_cells):
        return False

    cap = min(CONTENT_COUNT_MAX, len(last_cup["available_mouth_so_far"]), len(last_cup["full_ray"]))
    if cap < 1:
        return False
    last_cup["content_count"] = int(rng.integers(1, cap + 1))
    last_cup["shot_paint_cells"] = list(last_cup["full_ray"][: last_cup["content_count"]])
    last_cup["shot_border_hits"] = []
    if set(last_cup["shot_paint_cells"]) & reserved_paint_cells:
        return False
    return True


def _render_expected(canvas_in: np.ndarray, cups: list[dict]) -> np.ndarray:
    """Render the expected output directly from the planned chain metadata."""
    out = canvas_in.copy()
    for cup in cups:
        for r, c in cup["content_cells_abs"]:
            out[r, c] = 0
    for cup in cups:
        for r, c in cup["shot_border_hits"]:
            out[r, c] = cup["content_color"]
        for r, c in cup["shot_paint_cells"]:
            out[r, c] = cup["content_color"]
    return out


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Build a linear chain where every predecessor crosses the next cup and keeps going."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    color_pairs = _sample_color_pairs(rng, num_cups)

    placed: list[dict] = []
    reserved_paint_cells: set[tuple[int, int]] = set()

    for idx in range(num_cups):
        placed_cup = None
        for _ in range(CUP_BUILD_RETRIES):
            border_color, content_color = color_pairs[idx]
            blueprint = _build_blueprint(rng, border_color, content_color)
            if blueprint is None:
                continue

            occ_bbox = np.zeros((H, W), dtype=bool)
            for cup in placed:
                br0, bc0, br1, bc1 = cup["bbox"]
                occ_bbox[br0 : br1 + 1, bc0 : bc1 + 1] = True

            if idx == 0:
                cup = _try_place_first_cup(
                    rng, H, W, blueprint, occ_bbox, needs_successor=(num_cups > 1)
                )
                if cup is None:
                    continue
                cup["available_mouth_so_far"] = list(cup["mouth_abs"])
                placed_cup = cup
            else:
                prev_cup = placed[-1]
                result = _try_place_chain_target(
                    rng,
                    H,
                    W,
                    blueprint,
                    prev_cup,
                    placed,
                    reserved_paint_cells,
                    needs_successor=(idx < num_cups - 1),
                )
                if result is None:
                    continue
                cup, prev_count, prev_paint_cells, prev_hit_cells = result
                prev_cup["content_count"] = prev_count
                prev_cup["shot_paint_cells"] = prev_paint_cells
                prev_cup["shot_border_hits"] = prev_hit_cells
                reserved_paint_cells |= set(prev_paint_cells)
                placed_cup = cup

            placed.append(placed_cup)
            break

        if placed_cup is None:
            return None

    if not _finalize_last_cup(rng, placed[-1], placed, reserved_paint_cells):
        return None
    reserved_paint_cells |= set(placed[-1]["shot_paint_cells"])

    for cup in placed:
        available_final = [cell for cell in cup["available_mouth_so_far"] if cell not in reserved_paint_cells]
        # A cup's own future shot is never inside its mouth, so removing global painted cells is safe.
        if "content_count" not in cup:
            return None
        if len(available_final) < cup["content_count"]:
            return None
        cup["available_mouth_final"] = available_final

    canvas_in = np.zeros((H, W), dtype=int)
    for cup in placed:
        for r, c in cup["border_abs"]:
            canvas_in[r, c] = cup["border_color"]

    for cup in placed:
        count = cup["content_count"]
        idxs = rng.choice(len(cup["available_mouth_final"]), size=count, replace=False)
        cup["content_cells_abs"] = [cup["available_mouth_final"][int(i)] for i in idxs]
        for r, c in cup["content_cells_abs"]:
            canvas_in[r, c] = cup["content_color"]

    if not placed[0]["shot_paint_cells"]:
        return None
    starter_r, starter_c = placed[0]["shot_paint_cells"][0]
    if int(canvas_in[starter_r, starter_c]) != 0:
        return None
    canvas_in[starter_r, starter_c] = placed[0]["content_color"]

    expected = _render_expected(canvas_in, placed)
    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None

    metrics = combo_behavior_metrics(canvas_in)
    if metrics["starter_count"] != 1:
        return None
    if metrics["activated_count"] != num_cups:
        return None
    if metrics["chain_triggers"] != num_cups - 1:
        return None
    if metrics["recolored_border_cells"] < num_cups - 1:
        return None
    if metrics["post_hit_paint_cells"] < num_cups - 1:
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
