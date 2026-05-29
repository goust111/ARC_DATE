# Variant of abc82100
# Dimension: D1 — 源像素与规则的匹配条件 (match_cond)
# Change: s.color == d.source_color：源像素的颜色必须等于规则示例中源色像素的颜色... → 规则示例的方向编码匹配

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


# Design analysis
# yaml:
#   variant_goal:
#     core_change: >
#       原题用 source 颜色直接选规则；本变体改成“方向选规则”。
#       legend 中 source->target 的朝向（上/下/左/右）是规则编码，
#       scattered source 本身不再靠颜色区分，而是靠它最近的 cardinal teal cue
#       位于哪个方向来匹配对应规则。
#     visibility_strategy: >
#       每个样例至少 2 条规则，且所有规则共享同一种 source 色；
#       因此若仍按原题“同色即同规则”解释会失败，方向信息必须参与推理。
#   concrete_objects:
#     legend:
#       - 仍是 3 格共线结构：source -> target -> anchor。
#       - anchor 接触一个 teal stencil 组件；该 stencil 是之后要复制的模板。
#       - 每条 legend 绑定一个唯一方向，方向由 source 指向 target。
#     scattered_source:
#       - 仍是单个 source 色像素。
#       - 旁边额外有一个单像素 teal cue，位于上下左右某一方向，距离 1-3。
#       - 该 cue 不参与复制，只在输入里提供“该 source 应选哪条 legend”的方向信息。
#   behavior_trace:
#     original:
#       - 若某个 scattered 像素颜色等于某条 legend 的 source_color，就复制该 legend 的 stencil。
#     variant:
#       - 先发现所有 legend 及其方向编码。
#       - 对每个 scattered source，沿四个 cardinal 射线找最近 teal；
#         唯一最近方向即 cue 方向。
#       - 用这个方向去选 legend，再把该 legend 的 stencil 平移到 source 位置并重着色。
#     difference:
#       - 同色 source 可以分流到不同规则；
#       - rule selection 依赖局部 teal 几何关系，而不是颜色 id。
#   complexity_calibration:
#     kept:
#       - legend 发现方式、stencil 平移方式、背景/旁观者保留逻辑都保留。
#     adapted:
#       - 规则数改为 2-4，避免单规则样例让方向机制不可观察。
#       - 所有规则共用一个 source 色，target 色仍不同，保证颜色不足以判定规则。
#       - cue 只允许四个离散方向，方向自由度可见且有限。
#   implicit_assumptions_review:
#     solver_changes:
#       - 原 main() 中 `for sr, sc in argwhere(g == rule["source_color"])` 默认同色只对应一条规则；
#         变体下不成立，所以改为“同色给出候选集合，再按 cue 方向挑具体规则”。
#       - 原 `source_colors` 仅用于删除所有 source 色像素；变体里仍可用于筛出可变换像素，
#         但不能用于决定目标颜色。
#       - 原 discover_rules 不记录方向；变体必须记录 `(dr, dc)` 供后续匹配。
#     generator_guarantees:
#       - 每个方向至多一条 legend，避免方向到规则的映射歧义。
#       - 每个 scattered source 的最近 cardinal teal 唯一且就是自己的 cue。
#       - bystander 不使用 source 色，且其位置上不存在可匹配 cue，避免误变换。
#   placement_legality:
#     legend_separation: >
#       legend 区域及其 Chebyshev-1 邻域被保护，防止 cue 单像素错误地被当作某条 legend 的 stencil。
#     source_cue_checks: >
#       放置新 cue 后，会重新验证所有已放置 scattered source 的最近 teal 方向没有被破坏。
#     output_cleanliness: >
#       各 scattered stamp 在输出中禁止重叠，减少视觉杂乱，也让 solver/generator 顺序无关。
#   visual_observability:
#     inference_story: >
#       人类可从 2-4 个样例看到：legend 的箭头方向决定某个 target 色与 stencil；
#       scattered 区域里，同样颜色的 source 因为 cue 朝向不同而产生不同颜色/图案。
#     non_degenerate_signal: >
#       若把输入交给原题 solver，同色 source 会被同一规则统一处理，通常与新输出不同；
#       这可作为后续量化验证指标。


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS4)}


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


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
    teal_cells = set(map(tuple, np.argwhere(grid == pattern_color)))
    teal_cells = {(int(r), int(c)) for r, c in teal_cells}
    return _nearest_teal_direction_from_set(teal_cells, pos, grid.shape)


def _stamp_cells(pattern: np.ndarray, anchor: tuple[int, int], src: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    ar, ac = anchor
    sr, sc = src
    dr = sr - ar
    dc = sc - ac
    cells: set[tuple[int, int]] = set()
    for pr, pc in pattern:
        nr = int(pr) + dr
        nc = int(pc) + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-target-pattern examples and nearby teal cues to choose a stencil.

    Input structure:
    - Black background with colored pixels, teal stencil components, and extra single-pixel teal cues.
    - Each rule example is still a 4-neighbor colored pair source->target whose continued
      direction reaches an anchor touching a teal stencil component.
    - Outside the legends, scattered source pixels share source colors, so color alone no longer
      identifies which legend applies; a nearby cardinal teal cue supplies that direction code.

    Rule:
    - Discover every legend rule and record both its target color and its source->target direction.
    - For each non-legend source-colored pixel, look along the four cardinal rays and find the
      unique nearest teal pixel; its direction selects the matching legend.
    - Copy that legend's teal stencil onto the source pixel by translating the legend anchor there,
      recolor the copied stencil to the legend's target color, and remove all legend/cue teal.

    Key insight: The legend's arrow direction is the real rule identifier; the scattered source
    pixel uses its nearest teal cue to point at which legend orientation should be invoked.
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
        rules_by_dir.setdefault(rule["dir"], rule)

    out = np.full_like(g, int(bg))

    # Keep non-source bystanders, and also source-colored pixels that do not expose a valid cue.
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
        for nr, nc in _stamp_cells(rule["pattern"], rule["anchor"], (r, c), (h, w)):
            out[nr, nc] = rule["target_color"]

    return out


# --------------- generator (direction-matching variant) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 9, 3: 13, 4: 17}

RULE_COUNT_WEIGHTS = [2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 28

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


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]] | list[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _cheb_expand(cells: set[tuple[int, int]] | list[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((int(r) + dr, int(c) + dc))
    return out


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in DIRS8:
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    common_src = colors[0]
    targets = colors[1 : 1 + num_rules]
    dirs = random.sample(DIRS4, num_rules)

    max_s = min(STENCIL_MAX_CAP, max(3, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    for tgt_c, direction in zip(targets, dirs):
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": common_src,
                "tgt": tgt_c,
                "dir": direction,
                "rel": rel,
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
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal.add((nr, nc))
            if not ok:
                continue

            footprint = {(sr, sc), (tr, tc), (ar, ac)} | teal
            if footprint & blocked:
                continue
            if footprint & legend_guard:
                continue
            if {(sr, sc), (tr, tc)} & color_reserved:
                continue
            reverse_anchor = (sr - dr, sc - dc)
            if any(_chebyshev(cell, reverse_anchor) <= 1 for cell in teal):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal:
                inp[nr, nc] = 8

            rule["source_pos"] = (sr, sc)
            rule["target_pos"] = (tr, tc)
            rule["anchor"] = (ar, ac)

            blocked.update(footprint)
            color_reserved.update({(sr, sc), (tr, tc)})
            _reserve_4neighbors(color_reserved, {(sr, sc), (tr, tc)})
            teal_cells.update(teal)
            # Radius 2 blocks cue pixels from sitting behind a legend pair and
            # accidentally creating a reversed fake rule.
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


def _stamp_rel_cells(rel: list[tuple[int, int]], src: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    sr, sc = src
    cells: set[tuple[int, int]] = set()
    for rr, cc in rel:
        nr = sr + rr
        nc = sc + cc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


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

            stamp_cells = _stamp_rel_cells(rule["rel"], (sr, sc), (h, w))
            if stamp_cells & output_used:
                continue

            distances = list(range(CUE_DIST_MIN, CUE_DIST_MAX + 1))
            random.shuffle(distances)
            placed_here = False
            for dist in distances:
                cr = sr + rule["dir"][0] * dist
                cc = sc + rule["dir"][1] * dist
                cue = (cr, cc)
                if not (0 <= cr < h and 0 <= cc < w):
                    continue
                if cue in blocked or cue in legend_guard:
                    continue

                test_teal = set(teal_cells)
                test_teal.add(cue)
                test_refs = scattered_refs + [((sr, sc), rule["dir"])]
                if not _validate_sources(test_teal, (h, w), test_refs):
                    continue

                inp[sr, sc] = rule["src"]
                inp[cr, cc] = 8
                blocked.update({(sr, sc), cue})
                color_reserved.update({(sr, sc)})
                _reserve_4neighbors(color_reserved, {(sr, sc)})
                teal_cells.add(cue)
                output_used.update(stamp_cells)
                positions.append((sr, sc))
                cues.append(cue)
                scattered_refs.append(((sr, sc), rule["dir"]))
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
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
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
            if _nearest_teal_direction_from_set(teal_cells, (r, c), (h, w)) is not None:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((r, c, bc))
            blocked.add((r, c))
            color_reserved.add((r, c))
            _reserve_4neighbors(color_reserved, {(r, c)})
            break
    return placed


def _build_output(rules: list[dict], bystander_pos: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr = sr + rr
                nc = sc + cc
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = rule["tgt"]
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
