# Variant of abc82100
# Dimension: D7 — 规则示例和teal模板的清除方式 (transform)
# Change: 规则示例像素和teal模板像素在输出中被移除（变为背景色）... → teal模板像素变为目标色而非清除
#
# Design analysis
# ===============
# spec_variant:
#   unchanged_core:
#     - 输入仍由若干局部规则示例组成：source 像素、与其4邻接的 target 像素、以及沿同方向延伸后触到的 teal(8) 模板。
#     - 每条规则仍控制同色散落 source 像素的“盖章”：把该 teal 模板平移到散落 source 上，并改染成 target 色。
#     - 旁观像素仍是未参与任何规则的其他颜色孤点，输出中保持不变。
#   D7_change:
#     - 原题在输出中会抹掉规则示例区里的 colored pair 和 teal 模板。
#     - 本变体只改 teal 模板这部分：示例区原位的 teal 模板不再消失，而是直接改染成该规则的 target 色。
#     - 示例 pair 的 source/target 像素仍按原题逻辑被清除；因此“规则示例区域有着色输出”来自模板本体被保留，而不是把整条 legend 都保留。
#   concrete_trace:
#     - 假设某条规则是 `red -> yellow`，其 teal 模板是一个 L 形三像素。
#     - 原题：输出里，这个 legend 的红黄示例和 teal L 形都变黑；另一个散落 red 像素处会出现一个 yellow L 形。
#     - 变体：散落 red 像素处仍出现 yellow L 形；同时 legend 旁边原本的 teal L 形也要在原位变成 yellow。
#     - 差异只发生在“原位 teal 模板”的处理阶段，规则发现、散落源的匹配、平移盖章都不变。
#   complexity_calibration:
#     - 若只改 solver 而不约束 generator，散落 source 的盖章可能覆盖到 legend 模板区域，使“保留模板”在视觉上不可见。
#     - 因此 generator 额外禁止任何散落盖章触碰 legend 区域，保证 D7 差异在样例中稳定可观察。
#     - 这属于可观察性适配，不改变原题的规则学习难度，也不新增颜色逻辑。
#   implicit_assumptions_review:
#     - 原 `out = full(bg)` 合法，因为原题所有 legend/template 都被清空；本变体下仍可这样初始化，但随后必须显式补回 legend teal 模板。
#     - 原“保留普通彩色像素”循环会跳过所有 source 色和 legend pair；这个前提仍成立，因为散落 source 仍应被替换，legend pair 仍应消失。
#     - 原盖章循环默认只处理散落 source，不处理 legend source；这个过滤仍成立，否则会把示例 source 也当作待替换目标。
#     - 新增的“原位保留模板”写入必须在散落盖章之前完成或至少与其顺序确定；generator 会避免两者重叠，因此顺序不会引入歧义。
#   placement_legality:
#     - legend 仍按 source -> target -> anchor 的共线三点放置，teal 模板完整落在网格内。
#     - 散落 source 仍必须与其他对象保持足够间隔，避免伪规则；并新增限制：其未来盖章的所有落点都不得接触任何 legend 区域。
#     - 这样 solver 学到的规则唯一，输出中的“示例模板保留”也不会被其他盖章覆盖。
#   visual_observability:
#     - 解题者可直接看到：每个示例旁边的 teal 模板，在输出里变成了与 target 像素同色的形状，而不是消失。
#     - 同一条规则既在 legend 区出现一次，又在多个散落 source 处重复出现，颜色对应关系有重复支撑。
#   cleanliness:
#     - 保护 legend 区不被散落盖章命中，可避免输出中规则示例区出现难以解释的混色覆盖。
#   color_inference:
#     - 仍只依赖局部 source->target 示例，不依赖颜色 id 排序或隐藏数值性质。
#   main_new_assumptions:
#     - 每条被发现的规则都对应唯一的 teal 组件。
#     - 不同 legend teal 组件互不重叠。
#     - generator 通过原有占位约束与新的 legend 保护约束保证这些条件。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


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

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
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
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-pattern examples and stamp their shapes elsewhere.

    Input structure:
    - A mostly black grid contains isolated colored pixels and several teal (8)
      pattern components.
    - Each rule example is a 4-neighbor source->target colored pair whose
      continued direction points to an anchor that touches a teal component.

    Rule:
    - For every rule, recolor the nearby teal component to the rule's target
      color in its original legend location.
    - Also copy that same component onto every non-legend pixel of the matching
      source color, translating the rule anchor onto the source pixel.
    - Remove the colored source/target legend pair itself, while uninvolved
      bystander pixels stay unchanged.

    Key insight: The adjacent color pair tells you which source color triggers
    which target color, and the attached teal component supplies the reusable
    stencil that appears both in the legend area and at scattered source pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    # Keep unrelated colored pixels; learned source colors are consumed by stamping.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    # D7 change: the legend's own teal template remains and is recolored in place.
    for rule in rules:
        pattern = rule["pattern"]
        out[pattern[:, 0], pattern[:, 1]] = int(rule["target_color"])

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

BYSTANDER_MAX = 5

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 50
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (int(r + dr), int(c + dc))
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _translated_cells(base_r: int, base_c: int, rel) -> set[tuple[int, int]]:
    return {(int(base_r + rr), int(base_c + cc)) for rr, cc in rel}


def _place_legends(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue
            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((int(nr), int(nc)))
            if not ok:
                continue
            legend_pair = {(int(sr), int(sc)), (int(tr), int(tc))}
            all_cells = legend_pair | teal_cells
            if all_cells & (occupied | reserved):
                continue
            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            rule["legend_pair"] = legend_pair
            rule["legend_teal"] = teal_cells
            rule["legend_region"] = all_cells
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    H, W = inp.shape
    protected_legend_cells = set()
    for rule in rules:
        protected_legend_cells.update(rule["legend_region"])

    for rule in rules:
        positions = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            stamp_cells = _translated_cells(r, c, rule["rel"])
            hits_legend = False
            for nr, nc in stamp_cells:
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) in protected_legend_cells:
                    hits_legend = True
                    break
            if hits_legend:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = int(rule["src"])
    return True


def _place_bystanders(used_colors, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    bystander_colors = [int(c) for c in NON_BG_NON_TEAL_COLORS if int(c) not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) not in occupied and (r, c) not in reserved:
                bc = random.choice(bystander_colors)
                inp[r, c] = int(bc)
                placed.append((int(r), int(c), int(bc)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(int(r), int(c))])
                break
    return placed


def _build_output(rules, bystander_pos, H: int, W: int):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        for r, c in rule["legend_teal"]:
            output[int(r), int(c)] = int(rule["tgt"])
    for rule in rules:
        rel = rule["rel"]
        for sr, sc in rule["scattered"]:
            for rr, cc in rel:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[int(nr), int(nc)] = int(rule["tgt"])
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, H, W)
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
