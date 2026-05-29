# Combo variant of abc82100 (index 17)
# combo_spec: D7/2+D11/0
# Name: teal模板原位重着色+全局共享模板
#
# Design analysis
# ===============
# concrete_trace:
#   - 原题里每条规则各自带一个 teal 组件；输出时这些 legend teal 都被清空，只在散落
#     source 处复制对应模板。
#   - 本融合版改成“整题只有一个全局 teal 连通块”，多条 source->target 规则都指向这同一块，
#     但每条规则触到的是共享模板上的不同 anchor 像素。
#   - 例子：若共享模板是 7 像素折线，蓝->黄 规则指向左端 anchor，红->绿 规则指向右端
#     anchor。输出时，每个蓝色散点仍盖出整块黄色模板，每个红色散点仍盖出整块绿色模板；
#     同时原位那块 teal 模板不再消失，而是按“离哪个 anchor 最近”被分成黄色区和绿色区。
#   - 差异因此不是简单把模板保留下来，而是把“全局共享”与“原位重着色”联动：共享模板会
#     混合多条规则的目标色，散落 source 则各自复制同一形状但使用自己的 anchor 对齐。
# complexity_calibration:
#   - 若只做 D7/2 而不做 D11/0，原位模板最多单色，无法体现共享冲突。
#   - 若只做 D11/0 而仍清空 legend 模板，解题者看不到“共享模板在原位如何被分色”。
#   - 因此 generator 强制至少 2 条规则、且输出中共享模板必须出现至少 2 种目标色；这样
#     视觉上能直接看出它不是原题，也不是单纯的 keep-source 版本。
# implicit_assumptions_review:
#   - 原 solver 默认“每条规则对应一个独立 teal 组件”；这里失效，所以规则发现仍允许多条
#     规则命中同一组件，但后续逻辑必须把该组件当作一个全局模板统一处理。
#   - 原 `out = full(bg)` 仍成立，但之后不能把 teal 全部抹掉，而要对共享模板逐像素分配目标色。
#   - 原“source 颜色全部消费掉”仍成立；generator 继续保证旁观者颜色不属于任何 source 色。
#   - 原每条规则的相对模板 `pattern-anchor` 仍可复用；共享的是绝对 teal 组件，不共享 anchor。
# placement_legality:
#   - generator 只放置一个 teal 8 连通块，并从其中挑出多个可朝外延伸两格的 boundary anchor。
#   - 每个 legend 仍满足 source->target->anchor 共线；anchor 直接落在 teal 模板上，便于观察。
#   - 散落 source 的未来盖章禁止碰到共享 legend 区或其它已生成盖章，避免覆盖造成混乱。
# visual_observability:
#   - 输入里能看到多个不同颜色对都“指向”同一个 teal 大形状；输出里该形状留下并被切成多色区。
#   - 每种 source 颜色又会在其它位置重复生成整块单色模板，给出充足的颜色映射支撑。
# cleanliness_and_inference:
#   - 共享模板分色只依赖几何关系（离哪个 anchor 最近），不依赖颜色编号或放置顺序。
#   - generator 拒绝共享模板上的距离平局，避免出现需要隐藏 tie-break 的样本。
#   - 旁观像素只放在不会被任何盖章覆盖的位置，输出读图时不会和规则产物混在一起。
# main_new_assumptions:
#   - 所有有效规则都命中同一个 teal 连通块。
#   - 共享模板上每个像素对所选 anchors 都有唯一最近者。
#   - 每个 source 颜色恰有一条 legend 规则；generator 用唯一颜色对和间距约束保证这一点.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


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
                        "pattern_index": int(pattern_index),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _select_shared_rules(rules: list[dict]) -> list[dict]:
    if not rules:
        return []
    pattern_index = Counter(int(rule["pattern_index"]) for rule in rules).most_common(1)[0][0]
    shared_rules = [rule for rule in rules if int(rule["pattern_index"]) == pattern_index]
    by_source: dict[int, dict] = {}
    for rule in shared_rules:
        by_source.setdefault(int(rule["source_color"]), rule)
    return list(by_source.values())


def _shared_template_colors(pattern: np.ndarray, rules: list[dict]) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    anchors = [tuple(map(int, rule["anchor"])) for rule in rules]
    targets = [int(rule["target_color"]) for rule in rules]
    for rr, cc in map(tuple, pattern):
        cell = (int(rr), int(cc))
        best_i = 0
        best_d = _manhattan(cell, anchors[0])
        for i in range(1, len(anchors)):
            dist = _manhattan(cell, anchors[i])
            if dist < best_d:
                best_d = dist
                best_i = i
        colors[cell] = targets[best_i]
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer several color rules that all reuse one shared teal stencil.

    Input structure:
    - The grid has a black background, scattered colored source pixels, and one
      global teal (8) connected component used as a shared template.
    - Each rule example is a 4-neighbor source->target pair whose continued
      direction lands on an anchor that touches or belongs to that same teal
      component.

    Rule:
    - Every non-legend pixel of a learned source color receives a translated copy
      of the whole shared template, recolored to that rule's target color, with
      the rule's anchor mapped onto the source pixel.
    - The colored legend pairs themselves disappear, but the original shared teal
      component stays in place and is recolored cell-by-cell by the target color
      of the nearest rule anchor.
    - Uninvolved bystander pixels remain unchanged.

    Key insight: The local color pairs teach both the source->target mapping and
    which anchor point inside the shared teal shape belongs to that rule; those
    anchors partition the original template in place while also controlling how
    the same whole shape is aligned when copied onto scattered source pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _select_shared_rules(_discover_rules(g, bg=bg, pattern_color=pattern_color))
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(tuple(map(int, rule["source_pos"])))
        legend_pixels.add(tuple(map(int, rule["target_pos"])))
        source_colors.add(int(rule["source_color"]))

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        cell = (int(r), int(c))
        color = int(g[cell])
        if cell in legend_pixels or color in source_colors:
            continue
        out[cell] = color

    if not rules:
        return out

    shared_pattern = np.asarray(rules[0]["pattern"], dtype=int)
    for (r, c), color in _shared_template_colors(shared_pattern, rules).items():
        out[int(r), int(c)] = int(color)

    for rule in rules:
        ar, ac = map(int, rule["anchor"])
        pattern = np.asarray(rule["pattern"], dtype=int)
        for sr, sc in map(tuple, np.argwhere(g == int(rule["source_color"]))):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            rows = pattern[:, 0] + (sr - ar)
            cols = pattern[:, 1] + (sc - ac)
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (c) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25
SHARED_STENCIL_RETRIES = 120
TEMPLATE_LAYOUT_RETRIES = 200
ANCHOR_SAMPLE_RETRIES = 200

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

BYSTANDER_MAX = 5

SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _normalize_coords(coords) -> list[tuple[int, int]]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    min_r = min(rows)
    min_c = min(cols)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in coords)


def _make_shared_stencil(min_pixels: int, max_pixels: int, min_anchorable: int):
    for _ in range(SHARED_STENCIL_RETRIES):
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
        norm = _normalize_coords(coords)
        if len(norm) >= min_anchorable:
            return norm
    return None


def _build_rule_specs(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    used = set()
    for i in range(num_rules):
        src_c = int(colors[2 * i])
        tgt_c = int(colors[2 * i + 1])
        used.update((src_c, tgt_c))
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _translate_coords(coords, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(r + top), int(c + left)) for r, c in coords]


def _candidate_slots(template_cells: set[tuple[int, int]], H: int, W: int):
    slots: dict[tuple[int, int], list[dict]] = {}
    for ar, ac in template_cells:
        for dr, dc in _DIRS4:
            tr = ar - dr
            tc = ac - dc
            sr = ar - 2 * dr
            sc = ac - 2 * dc
            rev_anchor = ar - 3 * dr
            rev_anchor_c = ac - 3 * dc
            if not (0 <= sr < H and 0 <= sc < W and 0 <= tr < H and 0 <= tc < W):
                continue
            if (tr, tc) in template_cells or (sr, sc) in template_cells:
                continue
            if any(_chebyshev((rev_anchor, rev_anchor_c), cell) <= 1 for cell in template_cells):
                continue
            slots.setdefault((ar, ac), []).append(
                {
                    "anchor": (int(ar), int(ac)),
                    "source": (int(sr), int(sc)),
                    "target": (int(tr), int(tc)),
                    "dir": (int(dr), int(dc)),
                }
            )
    return slots


def _owner_map(template_cells: set[tuple[int, int]], anchors: list[tuple[int, int]]):
    owner: dict[tuple[int, int], tuple[int, int]] = {}
    counts = {tuple(anchor): 0 for anchor in anchors}
    for cell in template_cells:
        dists = [_manhattan(cell, anchor) for anchor in anchors]
        best = min(dists)
        if dists.count(best) != 1:
            return None
        anchor = tuple(anchors[dists.index(best)])
        owner[cell] = anchor
        counts[anchor] += 1
    if min(counts.values()) < 1:
        return None
    return owner


def _slot_choices_for_anchors(slots_by_anchor, anchors: list[tuple[int, int]]):
    anchors = sorted(anchors, key=lambda anchor: len(slots_by_anchor[anchor]))
    used_pair_cells: set[tuple[int, int]] = set()
    pair_reserved: set[tuple[int, int]] = set()
    chosen: list[dict] = []

    def backtrack(i: int) -> bool:
        if i == len(anchors):
            return True
        anchor = anchors[i]
        options = slots_by_anchor[anchor][:]
        random.shuffle(options)
        for slot in options:
            pair_cells = {tuple(slot["source"]), tuple(slot["target"])}
            if pair_cells & used_pair_cells:
                continue
            if pair_cells & pair_reserved:
                continue
            chosen.append(slot)
            used_pair_cells.update(pair_cells)
            added_reserved = set()
            for rr, cc in pair_cells:
                for dr, dc in _DIRS4:
                    neigh = (int(rr + dr), int(cc + dc))
                    if neigh not in pair_reserved:
                        added_reserved.add(neigh)
                    pair_reserved.add(neigh)
            if backtrack(i + 1):
                return True
            chosen.pop()
            used_pair_cells.difference_update(pair_cells)
            for neigh in added_reserved:
                pair_reserved.remove(neigh)
        return False

    if backtrack(0):
        return chosen
    return None


def _place_shared_template_and_legends(num_rules: int, rules, inp: np.ndarray):
    H, W = inp.shape
    area = H * W
    max_s = min(STENCIL_MAX_CAP, max(num_rules + 2, area // STENCIL_MAX_AREA_DIV))
    min_s = max(num_rules + 2, 4, area // STENCIL_MIN_AREA_DIV)
    min_s = min(min_s, STENCIL_MAX_CAP)
    max_s = max(min_s, max_s)

    for _ in range(TEMPLATE_LAYOUT_RETRIES):
        local_template = _make_shared_stencil(min_s, max_s, min_anchorable=num_rules)
        if local_template is None:
            continue
        box_h = max(r for r, _ in local_template) + 1
        box_w = max(c for _, c in local_template) + 1
        if box_h > H or box_w > W:
            continue
        top = random.randint(0, H - box_h)
        left = random.randint(0, W - box_w)
        template_list = _translate_coords(local_template, top, left)
        template_cells = {tuple(cell) for cell in template_list}
        slots_by_anchor = _candidate_slots(template_cells, H, W)
        candidate_anchors = list(slots_by_anchor)
        if len(candidate_anchors) < num_rules:
            continue
        for _ in range(ANCHOR_SAMPLE_RETRIES):
            anchors = random.sample(candidate_anchors, num_rules)
            owner = _owner_map(template_cells, anchors)
            if owner is None:
                continue
            slots = _slot_choices_for_anchors(slots_by_anchor, anchors)
            if slots is None:
                continue
            random.shuffle(slots)
            random.shuffle(rules)
            for rule, slot in zip(rules, slots):
                anchor = tuple(map(int, slot["anchor"]))
                rule["anchor"] = anchor
                rule["source_pos"] = tuple(map(int, slot["source"]))
                rule["target_pos"] = tuple(map(int, slot["target"]))
                rule["rel"] = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in template_list]
                rule["legend_pair"] = {rule["source_pos"], rule["target_pos"]}
            for r, c in template_list:
                inp[int(r), int(c)] = 8
            for rule in rules:
                sr, sc = rule["source_pos"]
                tr, tc = rule["target_pos"]
                inp[int(sr), int(sc)] = int(rule["src"])
                inp[int(tr), int(tc)] = int(rule["tgt"])
            legend_pairs = set()
            for rule in rules:
                legend_pairs.update(rule["legend_pair"])
            return {
                "template_list": template_list,
                "template_cells": template_cells,
                "owner": owner,
                "legend_pairs": legend_pairs,
                "legend_region": template_cells | legend_pairs,
            }
    return None


def _stamp_cells(base_r: int, base_c: int, rel, H: int, W: int) -> set[tuple[int, int]]:
    cells = set()
    for rr, cc in rel:
        nr = int(base_r + rr)
        nc = int(base_c + cc)
        if 0 <= nr < H and 0 <= nc < W:
            cells.add((nr, nc))
    return cells


def _place_scattered(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]], protected_output: set[tuple[int, int]]):
    H, W = inp.shape
    for rule in rules:
        positions = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            stamp = _stamp_cells(r, c, rule["rel"], H, W)
            if stamp & protected_output:
                continue
            positions.append((int(r), int(c)))
            inp[int(r), int(c)] = int(rule["src"])
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            protected_output.update(stamp)
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
    return True


def _place_bystanders(used_colors, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]], protected_output: set[tuple[int, int]]):
    bystander_colors = [int(c) for c in NON_BG_NON_TEAL_COLORS if int(c) not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in protected_output:
                continue
            color = int(random.choice(bystander_colors))
            inp[int(r), int(c)] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, template_owner, template_cells: set[tuple[int, int]], bystanders, H: int, W: int):
    out = np.zeros((H, W), dtype=int)
    anchor_to_tgt = {tuple(map(int, rule["anchor"])): int(rule["tgt"]) for rule in rules}
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for cell in template_cells:
        out[int(cell[0]), int(cell[1])] = int(anchor_to_tgt[template_owner[cell]])
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr, nc] = int(rule["tgt"])
    return out


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    H = random.randint(MIN_DIM_BY_RULES[num_rules], GRID_DIM_MAX)
    W = random.randint(MIN_DIM_BY_RULES[num_rules], GRID_DIM_MAX)

    rules, used_colors = _build_rule_specs(num_rules, H * W)
    inp = np.zeros((H, W), dtype=int)
    layout = _place_shared_template_and_legends(num_rules, rules, inp)
    if layout is None:
        return None

    occupied = set(layout["legend_region"])
    reserved: set[tuple[int, int]] = set()
    _reserve_4neighbors(reserved, layout["legend_region"])
    protected_output = set(layout["legend_region"])

    if not _place_scattered(rules, inp, occupied, reserved, protected_output):
        return None
    bystanders = _place_bystanders(used_colors, inp, occupied, reserved, protected_output)

    output = _build_output(
        rules,
        layout["owner"],
        layout["template_cells"],
        bystanders,
        H,
        W,
    )
    template_colors = {int(output[r, c]) for r, c in layout["template_cells"]}
    if len(template_colors) < 2:
        return None
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
