# Variant of 3dc255db
# Dimension: D8 — 多步处理顺序 (multi_step)
# Change: ordered_processing — 内容像素逐个移动到尾部堆叠(实际效果等同于一次性放置)... → 链式反应

from __future__ import annotations

from collections import Counter, deque

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


"""
design_analysis:
  a_specific_behavior:
    original:
      - 每个杯只要口内有内容，就独立弹射；处理顺序不影响最终结果，因为原生成器保证尾刺前方全是背景。
      - 输出只是在每个尾刺外侧画出一段等长直线，并清空各自口内内容。
    variant:
      - 现在只有带"启动像素"的杯先弹射。启动像素定义为：位于尾刺外侧第 1 格、颜色等于该杯内容色的单个像素。
      - 该杯的射线沿尾刺方向前进；若在内容数量耗尽前先撞到另一杯的边框，则停止，不覆盖边框，并激活被撞杯。
      - 被激活的杯再用自己的内容色、沿自己的尾刺方向重复同样过程，形成队列式链式反应。
      - 未被激活的杯保持原状，口内内容不会被擦除。
    concrete_trace:
      - 例：A 杯尾向右，内容数为 4，尾刺外已有 1 个同色启动像素；A 的右侧第 4 步是 B 杯左边框。
      - 原规则：A 会把 4 个像素都画出去；若生成器没禁掉碰撞，甚至会直接覆盖 B 的边框，B 仍按自己的内容独立弹射。
      - 新规则：A 只会在第 1-3 步画线，第 4 步撞到 B 边框并停止，同时激活 B；B 此时才擦除自己的口内内容并从自己的尾刺继续弹射。
      - 差异是"是否触发下一个杯"以及"撞击后是否停止"，两者都体现在输出结构上。
  b_complexity_calibration:
    - 为了让 D8 的差异稳定可见，生成器不再出 1 杯样本，固定生成 2-3 杯。
    - 每个样本强制形成一条长度为 num_cups 的连锁；这样 2-4 个训练样例里人眼一定能看到"先撞杯，再从别处继续发射"。
    - 只引入一个新显式符号：启动像素。其余对象仍是原题的杯框、口内内容、直线弹射。
    - 不增加多种触发分支；每杯仍只有一条射线，因此链是线性的，复杂度够看见但不至于失控。
  c_hidden_assumptions_audit:
    main_old_lines:
      - "for each cup independent": 依赖"所有杯都应自行弹射"。变体中失效，必须改成先找启动杯，再 BFS/queue 传播。
      - "erase content before eject"：原本对所有杯都安全；现在只能对已激活杯执行，否则会错误擦除未触发杯的内容。
      - "always draw len(content) cells": 原本路径无障碍；现在若途中撞杯，必须提前停止，不能继续无条件写色。
      - "out[rr,cc] = content_col"：原本不会写到别的结构；现在若 rr,cc 是另一杯边框，不能覆盖，只能把它当作触发事件。
      - "tail detection via mouth content": 仍成立，但新增了一个假设：每个杯在输入里都保留口内内容，否则无法像原题那样从内容侧判定唯一尾刺。生成器会保证这一点。
    main_new_assumptions:
      - 启动像素只出现在一个杯的尾刺外侧第 1 格；生成器保证唯一。
      - 连锁碰撞只通过"射线撞到别的杯边框坐标"触发；生成器避免射线先穿过别的杯口或别的杂散结构。
      - 内容颜色在口内是该杯的主导非边框色；生成器仍只在口内放单一内容色。
  d_placement_legality:
    - 杯 bbox 仍保持至少 1 格间隔，避免视觉粘连。
    - 生成器按链顺序放置：后一杯必须落在前一杯射线的入射侧上，使首次接触就是边框命中。
    - 已确定的射线可见段会被保留为占位，后续杯不能覆盖这些路径。
    - 最后一杯的射线必须有至少 1 格可画空间；中间杯必须为将来的下一跳保留至少 2 格潜在通路。
  e_visual_observability:
    - 启动像素是唯一的局部提示，告诉解题者"不是所有杯都同时发射"。
    - 输出中射线会在目标杯前停止，随后目标杯从自身尾刺方向继续发射，肉眼可见是顺序传播而非独立平铺。
    - 不使用颜色值大小、扫描顺序等不可见信息决定触发顺序；顺序完全由可见的启动像素和几何碰撞决定。
  f_visual_cleanliness:
    - 去掉原题的 stray pixel，避免与启动像素混淆。
    - 射线不会穿过杯内部，也不会覆盖边框；各层含义清晰。
    - 杯框颜色仍唯一，路径只用各自内容色，视觉上可分离。
  g_color_inference:
    - 不引入新的颜色映射；每杯仍是"边框色 + 内容色"。
    - 启动像素直接复用该杯内容色，因此人类能把它看作"这条射线已经开始了"的同色前导。
    - 为减少歧义，生成器避免再添加无关同色孤立点。
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

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400
CUP_BUILD_RETRIES = 80
# ==========================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood inside bbox from one bbox edge, blocked by the border cells."""
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
    """Recover cup metadata from the input grid using the original tail-disambiguation logic."""
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
    Chain-reaction version of 3dc255db.

    Exactly one cup is visibly "armed" by a same-colored starter pixel one step beyond its spike.
    Only armed / hit cups eject. A ray stops when it hits another cup border and activates that cup.
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
                if hit_idx not in activated:
                    activated.add(hit_idx)
                    queue.append(hit_idx)
                break
            out[rr, cc] = cup["content_col"]

    return out


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup whose tail points UP in canonical orientation."""
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
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted((int(r), int(c)) for r, c in visited)


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


def _build_blueprint(rng: np.random.Generator, border_color: int, content_color: int) -> dict | None:
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
        "spike_rel": (int(spike[0]), int(spike[1])),
        "mouth_rel": [(int(r), int(c)) for r, c in mouth],
        "border_rel": border_rel,
        "edge_border_rel": edge_border_rel,
        "shape": (ph, pw),
        "content_cap": min(CONTENT_COUNT_MAX, len(mouth)),
    }


def _tail_cells_to_edge(H: int, W: int, start: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR4[direction]
    rr, cc = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def _clear_run(
    H: int, W: int, start: tuple[int, int], direction: str, occ_cup: np.ndarray, occ_path: np.ndarray
) -> list[tuple[int, int]]:
    cells = []
    for rr, cc in _tail_cells_to_edge(H, W, start, direction):
        if occ_cup[rr, cc] or occ_path[rr, cc]:
            break
        cells.append((rr, cc))
    return cells


def _bbox_clear(r0: int, c0: int, ph: int, pw: int, occ_cup: np.ndarray, occ_path: np.ndarray) -> bool:
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    return not occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any() and not occ_path[r0 : r0 + ph, c0 : c0 + pw].any()


def _instantiate_blueprint(blueprint: dict, r0: int, c0: int) -> dict:
    ph, pw = blueprint["shape"]
    spike_abs = (r0 + blueprint["spike_rel"][0], c0 + blueprint["spike_rel"][1])
    return {
        **blueprint,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "spike_abs": spike_abs,
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in blueprint["mouth_rel"]],
        "border_abs": {(r0 + r, c0 + c) for (r, c) in blueprint["border_rel"]},
    }


def _try_place_first_cup(
    rng: np.random.Generator, H: int, W: int, blueprint: dict, occ_cup: np.ndarray, occ_path: np.ndarray
) -> dict | None:
    ph, pw = blueprint["shape"]
    coords = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords)
    for r0, c0 in coords:
        if not _bbox_clear(r0, c0, ph, pw, occ_cup, occ_path):
            continue
        cup = _instantiate_blueprint(blueprint, r0, c0)
        run = _clear_run(H, W, cup["spike_abs"], cup["direction"], occ_cup, occ_path)
        if len(run) < 2:
            continue
        return cup
    return None


def _try_place_hit_target(
    rng: np.random.Generator,
    H: int,
    W: int,
    blueprint: dict,
    prev_cup: dict,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
    needs_successor: bool,
) -> tuple[dict, int, list[tuple[int, int]]] | None:
    ph, pw = blueprint["shape"]
    coords = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords)

    prev_sr, prev_sc = prev_cup["spike_abs"]
    prev_dir = prev_cup["direction"]
    prev_dr, prev_dc = DIR4[prev_dir]
    incoming_side = OPPOSITE[prev_dir]

    for r0, c0 in coords:
        if not _bbox_clear(r0, c0, ph, pw, occ_cup, occ_path):
            continue
        cup = _instantiate_blueprint(blueprint, r0, c0)
        run = _clear_run(H, W, cup["spike_abs"], cup["direction"], occ_cup, occ_path)
        if needs_successor and len(run) < 2:
            continue
        if (not needs_successor) and len(run) < 1:
            continue

        hit_options = list(blueprint["edge_border_rel"][incoming_side])
        rng.shuffle(hit_options)
        for hr_rel, hc_rel in hit_options:
            hit_r = r0 + hr_rel
            hit_c = c0 + hc_rel
            if prev_dir == "R":
                if hit_r != prev_sr or hit_c <= prev_sc:
                    continue
                dist = hit_c - prev_sc
            elif prev_dir == "L":
                if hit_r != prev_sr or hit_c >= prev_sc:
                    continue
                dist = prev_sc - hit_c
            elif prev_dir == "D":
                if hit_c != prev_sc or hit_r <= prev_sr:
                    continue
                dist = hit_r - prev_sr
            else:
                if hit_c != prev_sc or hit_r >= prev_sr:
                    continue
                dist = prev_sr - hit_r

            if not (2 <= dist <= prev_cup["content_cap"]):
                continue

            pre_hit_cells = []
            ok = True
            for step in range(1, dist):
                rr = prev_sr + prev_dr * step
                cc = prev_sc + prev_dc * step
                if occ_cup[rr, cc] or occ_path[rr, cc]:
                    ok = False
                    break
                if r0 <= rr < r0 + ph and c0 <= cc < c0 + pw:
                    ok = False
                    break
                pre_hit_cells.append((int(rr), int(cc)))
            if not ok:
                continue

            return cup, int(dist), pre_hit_cells

    return None


def _mark_bbox(occ_cup: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occ_cup[r0 : r1 + 1, c0 : c1 + 1] = True


def _render_expected(H: int, W: int, bg: int, canvas_in: np.ndarray, cups: list[dict]) -> np.ndarray:
    out = canvas_in.copy()
    border_lookup: dict[tuple[int, int], int] = {}
    for idx, cup in enumerate(cups):
        for cell in cup["border_abs"]:
            border_lookup[cell] = idx

    activated = {0}
    queue: deque[int] = deque([0])
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
                if hit_idx not in activated:
                    activated.add(hit_idx)
                    queue.append(hit_idx)
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


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups: list[dict] = []

    for idx in range(num_cups):
        placed = None
        for _ in range(CUP_BUILD_RETRIES):
            used_borders = {cup["border_color"] for cup in cups}
            border_choices = [c for c in range(1, 10) if c not in used_borders]
            if not border_choices:
                return None
            if idx > 0:
                prev_content = int(cups[-1]["content_color"])
                border_choices = [c for c in border_choices if c != prev_content]
            if not border_choices:
                return None
            border_color = int(rng.choice(border_choices))
            content_choices = [c for c in range(1, 10) if c != border_color]
            content_color = int(rng.choice(content_choices))
            blueprint = _build_blueprint(rng, border_color, content_color)
            if blueprint is None:
                continue

            if idx == 0:
                cup = _try_place_first_cup(rng, H, W, blueprint, occ_cup, occ_path)
                if cup is None:
                    continue
                placed = cup
            else:
                prev_cup = cups[-1]
                hit_result = _try_place_hit_target(
                    rng,
                    H,
                    W,
                    blueprint,
                    prev_cup,
                    occ_cup,
                    occ_path,
                    needs_successor=(idx < num_cups - 1),
                )
                if hit_result is None:
                    continue
                cup, dist, pre_hit_cells = hit_result
                prev_cup["content_count"] = int(dist)
                prev_cup["path_cells"] = list(pre_hit_cells)
                for r, c in pre_hit_cells:
                    occ_path[r, c] = True
                placed = cup

            _mark_bbox(occ_cup, placed["bbox"])
            cups.append(placed)
            break
        if placed is None:
            return None

    last = cups[-1]
    tail_run = _clear_run(H, W, last["spike_abs"], last["direction"], occ_cup, occ_path)
    max_last = min(last["content_cap"], len(tail_run))
    if max_last < 1:
        return None
    last["content_count"] = int(rng.integers(1, max_last + 1))
    last["path_cells"] = tail_run[: last["content_count"]]
    for r, c in last["path_cells"]:
        if occ_path[r, c]:
            return None
        occ_path[r, c] = True

    canvas_in = np.zeros((H, W), dtype=int)
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        patch_mask = cup["patch"] != 0
        canvas_in[r0 : r1 + 1, c0 : c1 + 1][patch_mask] = cup["border_color"]

    for cup in cups:
        if len(cup["mouth_abs"]) < cup["content_count"]:
            return None
        idxs = rng.choice(len(cup["mouth_abs"]), size=cup["content_count"], replace=False)
        cup["content_cells_abs"] = [cup["mouth_abs"][int(i)] for i in idxs]
        for r, c in cup["content_cells_abs"]:
            canvas_in[r, c] = cup["content_color"]

    starter_r, starter_c = cups[0]["path_cells"][0]
    if canvas_in[starter_r, starter_c] != 0:
        return None
    canvas_in[starter_r, starter_c] = cups[0]["content_color"]

    if _starter_count(cups, canvas_in) != 1:
        return None

    expected = _render_expected(H, W, 0, canvas_in, cups)
    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
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
