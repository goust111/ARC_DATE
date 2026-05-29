# Combo variant of abc82100 (index 4)
# combo_spec: D1/3+D16/2
# Name: 方向编码匹配+颜色对方向编码旋转

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


# Design analysis
# yaml:
#   variant_goal:
#     core_change: >
#       legend 里的 source->target 方向不再只是找到 anchor 的几何辅助信息，
#       而是同一时间承担两项语义：
#       1) 该方向就是 scattered source 旁 teal cue 需要指向的“规则编号”；
#       2) 该方向还决定 teal stencil 在输出里要旋转多少个 90 度。
#       因此方向线索必须被解码两次，而且两次结果必须一致地落到同一条规则上。
#     small_example: >
#       设有两条 legend，二者 source 颜色都为红色。
#       向右 legend 表示“右侧 cue 选中我”，target 为黄色，旋转 0 度；
#       向下 legend 表示“下侧 cue 选中我”，target 为绿色，旋转 90 度。
#       输入里两个红色 scattered source 即便颜色相同，只要一个最近 teal 在右边、
#       一个最近 teal 在下边，它们就会分别生成黄色原向模板和绿色旋转模板。
#   behavior_trace:
#     original:
#       - 发现 legend 后，任一 scattered source 只靠 source 色匹配规则。
#       - 复制模板时不旋转。
#     combo:
#       - 先发现每条 legend，并记录 source->target 方向。
#       - 对每个 scattered source，沿四个 cardinal 射线找唯一最近 teal，
#         其方向决定选哪条 legend。
#       - 选中的同一方向再被映射成 quarter-turn 数，先旋转该 legend 的 canonical stencil，
#         再平移到 scattered source 上并改成该 legend 的 target 色。
#     difference:
#       - 同色 scattered source 会因 cue 方向不同而走向不同 target 色和不同朝向的模板。
#       - 如果忽略方向，只按颜色或只按不旋转复制，都会得到错误结果。
#   complexity_calibration:
#     controls:
#       - 每个样例固定 2-4 条规则，避免单规则时方向匹配退化不可见。
#       - 所有规则共享同一种 source 色，强制求解者不能用颜色直接选规则。
#       - 每个方向至多一条规则，避免 “同一 cue 方向对应多条 legend” 的歧义。
#       - stencil 必须对 90/180/270 旋转都不对称，保证旋转编码肉眼可见。
#     restraint:
#       - cue 方向只取四个 cardinal 档位，信息量有限且可重复观察。
#       - 输出中的 stamp 禁止互相重叠，避免多重覆盖掩盖旋转差异。
#   implicit_assumptions_review:
#     solver:
#       - `_discover_rules` 不能只记颜色，必须记 `(dr, dc)` 并转成 rotation code。
#       - 原版 `for sr, sc in argwhere(g == source_color)` 默认“同色对应唯一规则”，
#         融合后改为“同色给出候选，再按 cue 方向唯一选 rule”。
#       - 原版直接平移 `pattern` 坐标；融合后必须先转成 anchor-relative offsets，
#         再按 rule 方向旋转后平移。
#       - 保留 bystander 时不能简单把所有 source 色删掉；若某个 source 色像素没有合法 cue，
#         solver 应保守地把它当普通像素保留。
#     generator:
#       - 必须保证每个 scattered source 的最近 teal 方向唯一，且正好等于它所属规则方向。
#       - 必须避免 cue 靠近 legend 背面，否则 `_discover_rules` 可能学出反向假规则。
#       - 必须避免 bystander 落在能看到有效 cue 的位置，否则会被误当可变换 source。
#   placement_legality:
#     legend_spacing: >
#       legend footprint 及其 Chebyshev 半径 2 区域都会被保护，
#       防止 cue 像素或其他 legend teal 进入 anchor 周边，造成误发现。
#     source_spacing: >
#       scattered source 仍保持 4-隔离，避免与其他颜色组成新的 source-target 对。
#     cue_validation: >
#       每放一个新 cue，都会重新检查所有已放置 scattered source 的最近 teal 方向，
#       从而防止后放的 cue 破坏先放 source 的匹配。
#   visual_observability:
#     inference_story: >
#       观察多个样例时，人类会看到：legend 箭头方向与 scattered source 旁 cue 的方向一一对应，
#       同时该 legend 展示的 teal 模板在输出中会以不同朝向出现。
#     non_degenerate_signal: >
#       因为所有 legend 共用同一 source 色，若回退到原题逻辑，
#       每个 scattered source 会被全部规则同时处理；和融合输出通常显著不同。
#   visual_cleanliness:
#     choices:
#       - 输出 stamp 全部完整落在网格内，不使用边界裁剪，便于比较旋转后形状。
#       - 输入中的 cue 只是一粒独立 teal 像素，和大 teal stencil 在尺度上可区分。
#       - bystander 使用未参与规则的颜色，并避开任何输出 stamp 区域。


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
ROT_BY_STEP = {
    (0, 1): 0,
    (1, 0): 1,
    (0, -1): 2,
    (-1, 0): 3,
}


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _rotate_offset(offset: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = map(int, offset)
    for _ in range(int(turns) % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_offsets(offsets: list[tuple[int, int]] | np.ndarray, turns: int) -> np.ndarray:
    return np.asarray([_rotate_offset((int(r), int(c)), turns) for r, c in offsets], dtype=int)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "dir": (int(dr), int(dc)),
                        "rotation": int(ROT_BY_STEP[(int(dr), int(dc))]),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _nearest_teal_direction_from_set(
    teal_cells: set[tuple[int, int]], pos: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int] | None:
    h, w = shape
    r, c = pos
    best_dist: int | None = None
    best_dirs: list[tuple[int, int]] = []

    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        dist = 1
        while 0 <= rr < h and 0 <= cc < w:
            if (rr, cc) in teal_cells:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_dirs = [(dr, dc)]
                elif dist == best_dist:
                    best_dirs.append((dr, dc))
                break
            rr += dr
            cc += dc
            dist += 1

    if best_dist is None or len(best_dirs) != 1:
        return None
    return best_dirs[0]


def _nearest_teal_direction(grid: np.ndarray, pos: tuple[int, int], pattern_color: int = 8) -> tuple[int, int] | None:
    teal_cells = {(int(r), int(c)) for r, c in map(tuple, np.argwhere(grid == pattern_color))}
    return _nearest_teal_direction_from_set(teal_cells, pos, grid.shape)


def _stamp_rel_cells(rel: list[tuple[int, int]] | np.ndarray, src: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    sr, sc = src
    cells: set[tuple[int, int]] = set()
    for rr, cc in rel:
        nr = int(sr) + int(rr)
        nc = int(sc) + int(cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use legend directions both to choose a rule and to rotate its teal stencil.

    Input structure:
    - Black background with colored pixels, multi-pixel teal stencil components, and isolated teal cue pixels.
    - Each legend is still a 3-cell line source->target->anchor, and the anchor touches a teal stencil component.
    - Different legends may share the same source color, so scattered source pixels cannot be matched by color alone;
      each scattered source instead has a nearest teal cue on one cardinal ray.

    Rule:
    - Discover every legend, recording its source->target direction, target color, and canonical teal stencil.
    - For each non-legend source-colored pixel, find the unique nearest teal direction; that direction selects the legend.
    - Rotate that legend's stencil according to the same direction code, translate the rotated stencil so its anchor lands
      on the scattered source, recolor it to the legend's target color, and erase all legend/cue teal from the output.

    Key insight: The same visible direction signal is overloaded twice: it identifies which legend a scattered source should
    follow, and it also tells how the legend's teal template must be quarter-turned before stamping.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    rules_by_dir: dict[tuple[int, int], dict] = {}
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])
        ar, ac = rule["anchor"]
        rel = [(int(pr) - int(ar), int(pc) - int(ac)) for pr, pc in rule["pattern"]]
        rule["rot_rel"] = _rotate_offsets(rel, rule["rotation"])
        if rule["dir"] not in rules_by_dir:
            rules_by_dir[rule["dir"]] = rule

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels:
            continue
        if color not in source_colors:
            out[r, c] = color
            continue
        cue_dir = _nearest_teal_direction(g, (r, c), pattern_color=pattern_color)
        if cue_dir not in rules_by_dir:
            out[r, c] = color

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color not in source_colors:
            continue
        cue_dir = _nearest_teal_direction(g, (r, c), pattern_color=pattern_color)
        rule = rules_by_dir.get(cue_dir)
        if rule is None:
            continue
        for nr, nc in _stamp_rel_cells(rule["rot_rel"], (r, c), (h, w)):
            out[nr, nc] = rule["target_color"]

    return out


# --------------- generator (combo) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 28
ASYMMETRIC_STENCIL_RETRIES = 200

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

CUE_DIST_MIN = 1
CUE_DIST_MAX = 3
BYSTANDER_MAX = 5

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 300


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]] | list[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in DIRS4:
            reserved.add((int(r) + int(dr), int(c) + int(dc)))


def _cheb_expand(cells: set[tuple[int, int]] | list[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((int(r) + int(dr), int(c) + int(dc)))
    return out


def _is_asymmetric(rel: list[tuple[int, int]]) -> bool:
    if len({int(r) for r, _ in rel}) < 2 or len({int(c) for _, c in rel}) < 2:
        return False
    base = tuple(sorted((int(r), int(c)) for r, c in rel))
    for turns in (1, 2, 3):
        rotated = tuple(sorted(_rotate_offset((int(r), int(c)), turns) for r, c in rel))
        if rotated == base:
            return False
    return True


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    lo = max(3, int(min_pixels))
    hi = max(lo, int(max_pixels))
    for _ in range(ASYMMETRIC_STENCIL_RETRIES):
        n = random.randint(lo, hi)
        coords = {(0, 0)}
        for _ in range(n - 1):
            frontier = set()
            for r, c in coords:
                for dr, dc in DIRS8:
                    pos = (int(r) + int(dr), int(c) + int(dc))
                    if pos not in coords:
                        frontier.add(pos)
            coords.add(random.choice(list(frontier)))
        coords_list = sorted(coords)
        anchor = random.choice(coords_list)
        rel = [(int(r) - int(anchor[0]), int(c) - int(anchor[1])) for r, c in coords_list]
        if _is_asymmetric(rel):
            return coords_list, anchor

    coords_list = [(0, 0), (1, 0), (1, 1)]
    anchor = (1, 0)
    return coords_list, anchor


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    common_src = int(colors[0])
    targets = [int(c) for c in colors[1 : 1 + num_rules]]
    dirs = random.sample(list(ROT_BY_STEP.keys()), num_rules)

    max_s = min(STENCIL_MAX_CAP, max(3, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    for tgt, direction in zip(targets, dirs):
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r) - int(anchor[0]), int(c) - int(anchor[1])) for r, c in stencil]
        rot = int(ROT_BY_STEP[direction])
        rules.append(
            {
                "src": int(common_src),
                "tgt": int(tgt),
                "dir": (int(direction[0]), int(direction[1])),
                "rot": rot,
                "rel": rel,
                "rot_rel": _rotate_offsets(rel, rot),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, {common_src, *targets}


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    blocked: set[tuple[int, int]],
    color_reserved: set[tuple[int, int]],
    legend_guard: set[tuple[int, int]],
    teal_cells: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        rel = rule["rel"]
        dr, dc = rule["dir"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal = set()
            ok = True
            for rr, cc in rel:
                nr = int(ar) + int(rr)
                nc = int(ac) + int(cc)
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal.add((nr, nc))
            if not ok:
                continue

            footprint = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal
            if footprint & blocked:
                continue
            if footprint & legend_guard:
                continue
            if {(int(sr), int(sc)), (int(tr), int(tc))} & color_reserved:
                continue
            reverse_anchor = (int(sr) - int(dr), int(sc) - int(dc))
            if any(_chebyshev(cell, reverse_anchor) <= 1 for cell in teal):
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for nr, nc in teal:
                inp[nr, nc] = 8

            rule["source_pos"] = (int(sr), int(sc))
            rule["target_pos"] = (int(tr), int(tc))
            rule["anchor"] = (int(ar), int(ac))

            blocked.update(footprint)
            color_reserved.update({(int(sr), int(sc)), (int(tr), int(tc))})
            _reserve_4neighbors(color_reserved, [(int(sr), int(sc)), (int(tr), int(tc))])
            teal_cells.update(teal)
            legend_guard.update(_cheb_expand(list(footprint), 2))
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_sources(
    teal_cells: set[tuple[int, int]],
    shape: tuple[int, int],
    scattered_refs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    for pos, expected_dir in scattered_refs:
        if _nearest_teal_direction_from_set(teal_cells, pos, shape) != expected_dir:
            return False
    return True


def _place_scattered(
    rules: list[dict],
    inp: np.ndarray,
    blocked: set[tuple[int, int]],
    color_reserved: set[tuple[int, int]],
    legend_guard: set[tuple[int, int]],
    teal_cells: set[tuple[int, int]],
) -> tuple[bool, set[tuple[int, int]]]:
    h, w = inp.shape
    output_used: set[tuple[int, int]] = set()
    scattered_refs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for rule in rules:
        positions: list[tuple[int, int]] = []
        cues: list[tuple[int, int]] = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            if (sr, sc) in blocked or (sr, sc) in color_reserved or (sr, sc) in legend_guard:
                continue

            stamp_cells = _stamp_rel_cells(rule["rot_rel"], (int(sr), int(sc)), (h, w))
            if len(stamp_cells) != len(rule["rot_rel"]):
                continue
            if stamp_cells & output_used:
                continue

            distances = list(range(CUE_DIST_MIN, CUE_DIST_MAX + 1))
            random.shuffle(distances)
            placed_here = False
            for dist in distances:
                cr = int(sr) + int(rule["dir"][0]) * int(dist)
                cc = int(sc) + int(rule["dir"][1]) * int(dist)
                cue = (cr, cc)
                if not (0 <= cr < h and 0 <= cc < w):
                    continue
                if cue in blocked or cue in legend_guard:
                    continue

                test_teal = set(teal_cells)
                test_teal.add(cue)
                test_refs = scattered_refs + [((int(sr), int(sc)), rule["dir"])]
                if not _validate_sources(test_teal, (h, w), test_refs):
                    continue

                inp[sr, sc] = int(rule["src"])
                inp[cr, cc] = 8
                blocked.update({(int(sr), int(sc)), cue})
                color_reserved.update({(int(sr), int(sc))})
                _reserve_4neighbors(color_reserved, [(int(sr), int(sc))])
                teal_cells.add(cue)
                output_used.update(stamp_cells)
                positions.append((int(sr), int(sc)))
                cues.append(cue)
                scattered_refs.append(((int(sr), int(sc)), rule["dir"]))
                placed_here = True
                break

            if not placed_here:
                continue

        if len(positions) < SCATTERED_MIN:
            return False, set()
        rule["scattered"] = positions
        rule["cues"] = cues

    return True, output_used


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    blocked: set[tuple[int, int]],
    color_reserved: set[tuple[int, int]],
    legend_guard: set[tuple[int, int]],
    teal_cells: set[tuple[int, int]],
    output_used: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [int(c) for c in NON_BG_NON_TEAL_COLORS if int(c) not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in blocked or (r, c) in color_reserved or (r, c) in legend_guard:
                continue
            if (r, c) in output_used:
                continue
            if _nearest_teal_direction_from_set(teal_cells, (int(r), int(c)), (h, w)) is not None:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = int(color)
            placed.append((int(r), int(c), int(color)))
            blocked.add((int(r), int(c)))
            color_reserved.add((int(r), int(c)))
            _reserve_4neighbors(color_reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        output[int(r), int(c)] = int(color)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_rel_cells(rule["rot_rel"], (int(sr), int(sc)), (h, w)):
                output[nr, nc] = int(rule["tgt"])
    return output


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    inp = np.zeros((h, w), dtype=int)
    blocked: set[tuple[int, int]] = set()
    color_reserved: set[tuple[int, int]] = set()
    legend_guard: set[tuple[int, int]] = set()
    teal_cells: set[tuple[int, int]] = set()

    if not _place_legends(rules, inp, blocked, color_reserved, legend_guard, teal_cells):
        return None
    ok, output_used = _place_scattered(rules, inp, blocked, color_reserved, legend_guard, teal_cells)
    if not ok:
        return None
    bystanders = _place_bystanders(
        used_colors, inp, blocked, color_reserved, legend_guard, teal_cells, output_used
    )

    output = _build_output(rules, bystanders, h, w)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
