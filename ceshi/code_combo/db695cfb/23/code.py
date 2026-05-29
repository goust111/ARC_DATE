# Combo variant of db695cfb (index 23)
# combo_spec: D6/1+D12/2
# Name: 线段源+第三标记定方向

import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# design:
#   fusion_rule:
#     - 原题里每个对象由两个蓝点直接给出唯一一条 45 度对角线。
#     - 本融合版把这两个蓝点替换成两条 2-3 格的蓝色短棒，真正有效的是每条短棒的左上端锚点。
#     - 这两个锚点不再直接定义要画的那条线，而是定义一个左上-右下角对齐、跨度为偶数的正方形。
#     - 正方形中心四邻中的第三标记像素决定画哪条对角线：上/下表示 "\"，左/右表示 "/"。
#     - 被选中的对角线上的品红像素仍像原题一样沿垂直对角方向发射品红射线。
#   behavior_trace:
#     - 例：两条蓝棒锚点在 (1,1) 和 (5,5)。原题/D6 单变体会直接补成 "\"：[(1,1),(2,2),(3,3),(4,4),(5,5)]。
#     - 融合后，这两个锚点只说明存在一个 5x5 正方形；若红色标记在中心左侧 (3,2)，真正要画的是 "/"：
#       [(1,5),(2,4),(3,3),(4,2),(5,1)]，它甚至不经过那两个蓝锚点。
#     - 因而 solver 需要先从蓝棒里提锚点，再把锚点解释成正方形角，再读第三标记决定方向；两项改动是串联的。
#   complexity_calibration:
#     - 蓝棒只允许水平或垂直、长度只允许 2 或 3，锚点固定为左上端，保证“怎么看蓝棒”这件事有重复支撑。
#     - 方向指示只分两类：上下对应 "\"，左右对应 "/"，避免过多离散映射。
#     - 正方形跨度限制为 4-16 且必须为偶数，保证中心格存在，第三标记的编码稳定且易观察。
#     - 若一个样例中有多条对象，生成器尽量混合两种方向，提升从 2-4 个样例中归纳方向规则的机会。
#   implicit_assumption_audit:
#     - 原 solver 的“同色像素本身就是端点”失效了；蓝色像素必须先按 4 邻域分成短棒组件，否则棒身会被错当端点。
#     - 原 solver 的“蓝点共对角线就能直接画线”也失效；这里蓝锚点只定义一个正方形，不定义最终所画对角线。
#     - 原 solver 的“最长同色对角对就是目标线色”需要改成“拥有合法短棒组件且能和第三标记形成候选正方形的颜色”。
#     - 新 solver 额外依赖：有效蓝棒彼此不能 4 邻接；每个真实对象在中心四邻恰有一个 selector；对象正方形彼此分离。
#     - 生成器会显式保证这些前提，否则组件会合并、selector 归属会模糊，solver 识别也会退化。
#   placement_legality:
#     - 生成器先放一个不与既有对象重叠的正方形 bbox，再为其左上和右下角锚点挑选向右或向下延伸的蓝棒。
#     - 蓝棒额外像素不能撞到已有输入像素、已有输出路径，也不能与其他蓝像素 4 邻接。
#     - 触发器只放在被选中的那条对角线内部；射线路径不得碰到任何输入像素，也不得穿过其他对象的保留区。
#   visual_observability:
#     - 输入中蓝色对象明显从“单点”变成“短棒”，这是第一层可见变化。
#     - 同样一对蓝棒锚点在不同样例中，若 selector 落在上下或左右，输出会分别画 "\" 或 "/"，这是第二层可见变化。
#     - "/" 情况下蓝线不经过两条蓝棒的锚点，能明确提示“第三标记真的在控制方向”，不会退化成只看蓝棒连线。
#   visual_cleanliness:
#     - 每个对象占一个独立正方形区域，selector 固定在中心四邻，视觉归属清楚。
#     - 品红杂点不进入这些正方形；射线也不会撞入其他对象区域，避免多层覆盖导致对象边界难辨。
#   quantitative_non_degeneracy:
#     - 退化风险主要是生成样本几乎总画 "\"，那样 selector 虽存在但不产生可观测分歧。
#     - 因此后续验证会统计 slash 对象占比，以及“至少含一个 slash 对象”的样本比例；这直接衡量第三标记是否在起作用。


def _diag_path_from_square(
    tl: tuple[int, int], br: tuple[int, int], slope: str
) -> list[tuple[int, int]]:
    r0, c0 = tl
    r1, c1 = br
    steps = r1 - r0
    if slope == "backslash":
        return [(r0 + k, c0 + k) for k in range(steps + 1)]
    return [(r0 + k, c1 - k) for k in range(steps + 1)]


def _connected_components(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        components.append(comp)
    return components


def _is_source_bar(comp: list[tuple[int, int]]) -> bool:
    if len(comp) not in (2, 3):
        return False
    rows = sorted(r for r, _ in comp)
    cols = sorted(c for _, c in comp)
    if len(set(rows)) == 1:
        return cols == list(range(cols[0], cols[0] + len(comp)))
    if len(set(cols)) == 1:
        return rows == list(range(rows[0], rows[0] + len(comp)))
    return False


def _selector_neighbors(center: tuple[int, int], h: int, w: int) -> list[tuple[tuple[int, int], str]]:
    cr, cc = center
    neighbors: list[tuple[tuple[int, int], str]] = []
    for pos in ((cr - 1, cc), (cr + 1, cc)):
        if 0 <= pos[0] < h and 0 <= pos[1] < w:
            neighbors.append((pos, "backslash"))
    for pos in ((cr, cc - 1), (cr, cc + 1)):
        if 0 <= pos[0] < h and 0 <= pos[1] < w:
            neighbors.append((pos, "slash"))
    return neighbors


def _extract_scene(grid: np.ndarray) -> dict:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    bg = get_bg(grid)

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if col != bg:
                by_color.setdefault(col, []).append((r, c))

    bars_by_color: dict[int, list[dict]] = {}
    for col, pts in by_color.items():
        bars = []
        for comp in _connected_components(pts):
            if not _is_source_bar(comp):
                continue
            bars.append(
                {
                    "cells": sorted(comp),
                    "anchor": min(comp),
                }
            )
        if bars:
            bars_by_color[col] = sorted(bars, key=lambda bar: bar["anchor"])

    candidate_cache: dict[int, list[dict]] = {}
    best = None  # (candidate_count, max_dist, bar_count, color)
    for col, bars in bars_by_color.items():
        anchors = [bar["anchor"] for bar in bars]
        candidates = []
        for i in range(len(anchors)):
            r0, c0 = anchors[i]
            for j in range(i + 1, len(anchors)):
                r1, c1 = anchors[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr < 4 or dr != dc or dr % 2 != 0:
                    continue
                center = ((r0 + r1) // 2, (c0 + c1) // 2)
                nearby = []
                for pos, slope in _selector_neighbors(center, h, w):
                    val = int(grid[pos])
                    if val != bg and val != col:
                        nearby.append((pos, val, slope))
                if nearby:
                    candidates.append(
                        {
                            "tl": (r0, c0),
                            "br": (r1, c1),
                            "dist": dr,
                            "center": center,
                            "nearby": nearby,
                        }
                    )
        candidate_cache[col] = candidates
        if not candidates:
            continue
        score = (len(candidates), max(cand["dist"] for cand in candidates), len(bars), col)
        if best is None or score > best:
            best = score

    assert best is not None, "at least one selector-supported pair of blue bars must exist"

    _, _, _, line_color = best
    line_candidates = candidate_cache[line_color]

    selector_counts: dict[int, int] = {}
    for cand in line_candidates:
        for _, col, _ in cand["nearby"]:
            selector_counts[col] = selector_counts.get(col, 0) + 1
    selector_color = max(selector_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    segments = []
    used_anchors = set()
    for cand in sorted(line_candidates, key=lambda item: item["dist"], reverse=True):
        tl = cand["tl"]
        br = cand["br"]
        if tl in used_anchors or br in used_anchors:
            continue

        selector_hits = []
        for pos, slope in _selector_neighbors(cand["center"], h, w):
            if int(grid[pos]) == selector_color:
                selector_hits.append((pos, slope))
        if len(selector_hits) != 1:
            continue

        selector_pos, slope = selector_hits[0]
        segments.append(
            {
                "tl": tl,
                "br": br,
                "center": cand["center"],
                "selector": selector_pos,
                "slope": slope,
                "path": _diag_path_from_square(tl, br, slope),
            }
        )
        used_anchors.add(tl)
        used_anchors.add(br)

    assert segments, "at least one fully specified square object must be recoverable"

    return {
        "bg": bg,
        "line_color": line_color,
        "selector_color": selector_color,
        "segments": segments,
        "bars": bars_by_color[line_color],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover square corners from blue bars, let a third marker choose the active diagonal, then emit perpendicular rays from trigger pixels on that diagonal.

    Input structure:
    - The background is the dominant color. Real source objects are short blue bars of length 2-3, each using its northwest-most cell as a hidden anchor. Two such anchors form the top-left and bottom-right corners of an even-sized square, and a third colored marker sits in one of the four neighbors of that square's center.

    Rule:
    - Detect each valid blue bar, take its northwest-most anchor, pair anchors into selector-supported squares, and draw the chosen blue diagonal: marker above/below center means "\"; marker left/right means "/".
    - Whenever that chosen diagonal passes through a non-blue, non-selector pixel, treat it as a trigger and extend rays of that trigger's color along the perpendicular diagonal directions until the grid boundary.

    Key insight:
    - The blue bars no longer tell you a line directly. They only encode which square is relevant, while the third marker resolves which of the square's two diagonals is active. Only after both deductions do the magenta trigger rays make sense.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    scene = _extract_scene(grid)
    bg = scene["bg"]
    line_color = scene["line_color"]
    selector_color = scene["selector_color"]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for seg in scene["segments"]:
        dc = 1 if seg["slope"] == "backslash" else -1
        perp_dirs = [(1, -dc), (-1, dc)]
        for r, c in seg["path"]:
            val = int(out[r, c])
            if val == bg:
                out[r, c] = line_color
            elif val != line_color and val != selector_color:
                triggers.setdefault((r, c, val), set()).update(perp_dirs)

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_SELECTOR_COLOR = 2
_TRIGGER_COLOR = 6
_OBJECT_COUNT = (1, 4)
_SQUARE_DIAG_RANGE = (4, 16)   # even square side/diagonal-step count
_TRIGGERS_PER_OBJECT = (0, 2)
_SOURCE_BAR_LEN_RANGE = (2, 3)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_OBJECT_PLACE_RETRIES = 160
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _selector_choices(center: tuple[int, int], slope: str, h: int, w: int) -> list[tuple[int, int]]:
    cr, cc = center
    if slope == "backslash":
        choices = [(cr - 1, cc), (cr + 1, cc)]
    else:
        choices = [(cr, cc - 1), (cr, cc + 1)]
    return [(r, c) for r, c in choices if 0 <= r < h and 0 <= c < w]


def _compute_ray_cells(
    triggers: list[tuple[int, int]], perp_dirs: list[tuple[int, int]], h: int, w: int
) -> set[tuple[int, int]]:
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _bar_cells(anchor: tuple[int, int], orient: str, length: int) -> list[tuple[int, int]]:
    r, c = anchor
    if orient == "h":
        return [(r, c + k) for k in range(length)]
    return [(r + k, c) for k in range(length)]


def _bar_options(anchor: tuple[int, int], h: int, w: int) -> list[list[tuple[int, int]]]:
    options = []
    for length in range(_SOURCE_BAR_LEN_RANGE[0], _SOURCE_BAR_LEN_RANGE[1] + 1):
        if anchor[1] + length - 1 < w:
            options.append(_bar_cells(anchor, "h", length))
        if anchor[0] + length - 1 < h:
            options.append(_bar_cells(anchor, "v", length))
    random.shuffle(options)
    return options


def _touches_blue(candidate_cells: set[tuple[int, int]], blue_cells: set[tuple[int, int]]) -> bool:
    for r, c in candidate_cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in blue_cells and (nr, nc) not in candidate_cells:
                return True
    return False


def _build_expected(inp: np.ndarray, bg: int, objects: list[dict]) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()
    for obj in objects:
        for r, c in obj["path"]:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in obj["trigger_positions"]:
            for ddr, ddc in obj["perp_dirs"]:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < h and 0 <= c < w:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_object(
    desired_slope: str,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
) -> dict | None:
    even_lengths = [
        dist
        for dist in range(_SQUARE_DIAG_RANGE[0], min(_SQUARE_DIAG_RANGE[1], min(h - 1, w - 1)) + 1)
        if dist % 2 == 0
    ]
    if not even_lengths:
        return None

    for _ in range(_OBJECT_PLACE_RETRIES):
        dist = random.choice(even_lengths)
        r0 = random.randint(0, h - 1 - dist)
        c0 = random.randint(0, w - 1 - dist)
        tl = (r0, c0)
        br = (r0 + dist, c0 + dist)

        bbox_cells = {(r, c) for r in range(r0, r0 + dist + 1) for c in range(c0, c0 + dist + 1)}
        if bbox_cells & reserved:
            continue

        selector_options = _selector_choices(((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2), desired_slope, h, w)
        if not selector_options:
            continue
        selector = random.choice(selector_options)

        path = _diag_path_from_square(tl, br, desired_slope)
        path_set = set(path)
        path_interior = path[1:-1]

        bar0_options = _bar_options(tl, h, w)
        bar1_options = _bar_options(br, h, w)
        if not bar0_options or not bar1_options:
            continue

        chosen_bars = None
        for bar0 in bar0_options:
            for bar1 in bar1_options:
                source_cells = set(bar0) | set(bar1)
                if len(source_cells) != len(bar0) + len(bar1):
                    continue
                if source_cells & occupied:
                    continue
                if source_cells & drawn:
                    continue
                if source_cells & reserved:
                    continue
                if _touches_blue(source_cells, blue_cells):
                    continue
                if (source_cells - {tl, br}) & path_set:
                    continue
                chosen_bars = [bar0, bar1]
                break
            if chosen_bars is not None:
                break
        if chosen_bars is None:
            continue

        max_triggers = min(_TRIGGERS_PER_OBJECT[1], len(path_interior))
        n_triggers = random.randint(_TRIGGERS_PER_OBJECT[0], max_triggers)
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 else []

        dc = 1 if desired_slope == "backslash" else -1
        perp_dirs = [(1, -dc), (-1, dc)]
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, h, w)

        source_cells = set().union(*map(set, chosen_bars))
        input_cells = source_cells | {selector, *trigger_positions}
        output_only = path_set - set(trigger_positions) - source_cells
        if input_cells & occupied:
            continue
        if (input_cells | output_only) & drawn:
            continue
        if ray_cells & occupied:
            continue
        if ray_cells & input_cells:
            continue
        if ray_cells & drawn:
            continue
        if ray_cells & reserved:
            continue

        return {
            "tl": tl,
            "br": br,
            "selector": selector,
            "bars": chosen_bars,
            "slope": desired_slope,
            "path": path,
            "trigger_positions": trigger_positions,
            "perp_dirs": perp_dirs,
            "bbox_cells": bbox_cells,
            "source_cells": source_cells,
            "drawn_cells": output_only | ray_cells,
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _SELECTOR_COLOR, _TRIGGER_COLOR)])

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    blue_cells: set[tuple[int, int]] = set()
    objects: list[dict] = []

    target_objects = random.randint(*_OBJECT_COUNT)
    desired_slopes = []
    if target_objects == 1:
        desired_slopes = [random.choice(["backslash", "slash"])]
    else:
        desired_slopes = ["backslash", "slash"]
        desired_slopes.extend(random.choice(["backslash", "slash"]) for _ in range(target_objects - 2))
        random.shuffle(desired_slopes)

    for desired_slope in desired_slopes:
        obj = _try_place_object(desired_slope, h, w, occupied, drawn, reserved, blue_cells)
        if obj is None:
            continue
        objects.append(obj)
        occupied.update(obj["source_cells"])
        occupied.add(obj["selector"])
        occupied.update(obj["trigger_positions"])
        drawn.update(obj["drawn_cells"])
        reserved.update(obj["bbox_cells"])
        reserved.update(obj["source_cells"])
        blue_cells.update(obj["source_cells"])

    if not objects:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for obj in objects:
        for bar in obj["bars"]:
            for r, c in bar:
                inp[r, c] = _LINE_COLOR
        inp[obj["selector"]] = _SELECTOR_COLOR
        for tp in obj["trigger_positions"]:
            inp[tp] = _TRIGGER_COLOR

    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn or (r, c) in reserved:
                continue
            if _touches_blue({(r, c)}, blue_cells):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            blue_cells.add((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn or (r, c) in reserved:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, objects)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
