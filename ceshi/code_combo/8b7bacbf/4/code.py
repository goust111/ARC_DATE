# Combo variant of 8b7bacbf (index 4)
# combo_spec: D2/1+D11/2
# Name: 分支路径+干扰假路径
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    原题里，每个边缘单像素 seed 只连到一条 8 连通细线；线旁挂着若干闭合框架环，solver 只需把
#    与该 seed 所在线路网络相连的腔体填成 seed 色。
#    这个融合版同时改两处：
#    - 主线路不再是单链，而是 1px 宽的 8 连通树。seed 的 trunk 会在弯折点处分叉，分支末端也会挂环。
#    - 画面里还会出现一条与某个 seed 使用同一 line_color、但与该 seed 完全断开的假路径网络；假路径
#      也会挂自己的框架环。
#    小例子：左上角 seed=6，经 color=2 的 trunk 向内走，在某个 bend 处分出一枝，trunk 旁一个环、
#    branch 末端再一个环，两者都应填成 6。同时右下角再放一条断开的 color=2 干扰路径，也挂一个环，
#    这个环不能填。原始逻辑若只做“找到同色线附近的环”，会误把右下角也填掉；融合后必须从 seed 出发
#    做 8 连通追踪，得到 seed 自己拥有的同色 line component，再只填该 component 可达的环。
#
# b) 复杂度校准
#    分支数控制在每个样例总计 1-3 个，避免树太碎；每个真实分支至少挂 1 个环，保证“分支会改变输出”
#    在肉眼上可见。干扰网络通常只有 1 条短路径和 1-3 个环，数量足以反制纯颜色匹配，又不会淹没主对象。
#
# c) 隐式假设审查
#    - seed 仍是唯一色、且只出现一次的边缘像素；生成器保持这一点。
#    - seed 的首个 line 像素仍与 seed 4 邻接；因此可从 seed 邻域稳定读出 line_color。
#    - 原题里 “traced 非背景分量” 大致等于“这条线和它挂着的环”；这里仍成立，但因同色假路径出现，
#      不能把全局所有 line_color 像素都视为 seed 所有物，必须额外求 seed-owned 同色分量。
#    - cavity 边界仍要求单一 frame 色；否则不是有效框架环。
#    - 真实环必须同时满足：边界有像素属于 seed 可达非背景分量，且边界 8 邻接 seed-owned line pixels。
#      这样既支持分支上的环，也排除仅因“全局同色”而误匹配的干扰环。
#
# d) 放置合法性
#    树形线路的新增像素只能落在空白处，且除父节点外不能 8 邻接任何已占用像素；这样能维持 1px 宽树且
#    不会自碰。分支只能从 bend node 发出，避免在 8 连通下从直线侧面长枝时意外形成环。框架放置时要求
#    ring 和 cavity 为空、cavity 不贴线、ring 至少 8 邻接目标 line_color，且 ring 不贴异物。干扰网络
#    从内部空白区起步，也用同样的细线约束，因此能保证与任何 seed 网络完全断开。
#
# e) 视觉可观察性
#    解题者会在输入里直接看到两种证据同时出现：
#    - 同一个 seed 的线路在内部明显分叉，且多处相隔较远的环都在输出里被同一 seed 色填充。
#    - 另一处出现同色但断开的细线和环，输出却不填那里。由此可归纳“要沿 seed 连通关系追踪整棵树”。
#
# f) 可视化整洁度
#    所有对象仍是原题语法中的细线和小矩形环，没有引入新图元。主树、分支、假路径之间都保留 1-cell
#    以上隔离，避免不同网络糊成一团；环尺寸维持 2x2 或 3x3 量级，画面信息量仍接近原题。
#
# g) 变化的可推断性
#    颜色规则没有新增映射：腔体始终填对应 seed 的唯一颜色。难点只在可见拓扑，不在颜色编号。干扰网络
#    故意复用某个真实 seed 的 line_color，让“同色不等于同属一条路径”在少量样例中反复出现。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _seed_line_color(g: np.ndarray, seed: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    sr, sc = seed
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    return int(Counter(neigh_cols).most_common(1)[0][0])


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _color_component8_from(
    g: np.ndarray,
    starts: list[tuple[int, int]] | set[tuple[int, int]],
    color: int,
) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [p for p in starts if int(g[p]) == color]
    vis = set(st)
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == color:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _background_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities = []
    for pts in bg_comps:
        region = set(pts)
        if any(p in region for p in corners if int(g[p]) == bg):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if boundary:
            cavities.append((region, boundary, {int(g[p]) for p in boundary}))
    return cavities


def _boundary_adjacent_to_pixels(
    boundary: set[tuple[int, int]],
    pixels: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    if not pixels:
        return False
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pixels:
                return True
    return False


def _seed_infos(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    infos = []
    for seed in _seed_pixels(g, bg):
        line_col = _seed_line_color(g, seed, bg)
        line_starts = []
        for nr, nc in neighbors4(*seed):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                line_starts.append((nr, nc))
        assert line_starts, "seed pixel must touch its line color orthogonally"
        infos.append(
            {
                "seed_rc": seed,
                "seed_col": int(g[seed]),
                "line_col": line_col,
                "owned_nonbg": _nonbg_component8_from(g, seed, bg),
                "owned_line": _color_component8_from(g, line_starts, line_col),
                "all_line": {(int(r), int(c)) for r, c in np.argwhere(g == line_col)},
            }
        )
    return infos


def _solve_with_connectivity_mode(g: np.ndarray, require_connected_line: bool) -> np.ndarray:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    cavities = _background_cavities(g, bg)

    for info in _seed_infos(g, bg):
        line_pixels = info["owned_line"] if require_connected_line else info["all_line"]
        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            if require_connected_line and not (boundary & info["owned_nonbg"]):
                continue
            if not _boundary_adjacent_to_pixels(boundary, line_pixels, h, w):
                continue
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = info["seed_col"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the cavities that belong to a border-seeded branching line network.

    Input structure:
    - The background is the most common color.
    - Each seed is a unique single pixel on the border and touches its own line color.
    - A seed's line is now an 8-connected tree, so one seed can reach several branches and loops.
    - Elsewhere there may be disconnected fake paths that reuse the same line color and may also
      have attached frame loops.

    Rule:
    - For each seed, trace the full non-background component reachable from that seed.
    - Inside that reachable drawing, isolate the seed-owned 8-connected component of its line color.
    - Flood-fill a background cavity with the seed color only if the cavity has a single-color
      frame, that frame lies in the seed-reachable component, and the frame is touched by the
      seed-owned line component.

    Key insight:
    - Color matching alone is insufficient because fake paths can copy the real line color.
    - The solver must follow connectivity from the seed across the whole branching tree, then only
      trust line pixels that remain in the seed's own connected line component.
    """
    return _solve_with_connectivity_mode(np.asarray(grid, dtype=int), require_connected_line=True)


# ========================== CFG ==========================
GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_TRUNK_LEN = 7
TRUNK_LEN_RANGE = (9, 22)
MIN_BRANCH_LEN = 4
BRANCH_LEN_RANGE = (4, 10)
MAX_BRANCHES_PER_SEED = 2
TOTAL_BRANCHES_ONE_SEED = (1, 3)
TOTAL_BRANCHES_TWO_SEEDS = (1, 3)
BRANCH_ATTACH_TRIES = 40

FRAME_COLORS_MAX = 4
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 60
DISTRACTOR_FRAME_TRIES = 50
EXTRA_LOOPS_PER_SEED = (0, 2)

DIAG_WEIGHT_ALIGNED = 4
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
TURN_BONUS = 2
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.35
PATH_STALL_LIMIT = 5

DISTRACTOR_NETWORK_TRIES = 80
DISTRACTOR_LINE_LEN = (5, 14)
DISTRACTOR_NETWORK_FRAMES = (1, 3)
DISTRACTOR_SOLO_MAX = 1

GEN_RETRIES = 500
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    fh, fw = ch + 2, cw + 2
    ring = []
    for dr in range(fh):
        for dc in range(fw):
            on_bdr = dr == 0 or dr == fh - 1 or dc == 0 or dc == fw - 1
            corner = dr in (0, fh - 1) and dc in (0, fw - 1)
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if int(canvas[rr, cc]) != -1:
            return False
    for rr, cc in cavity:
        if int(canvas[rr, cc]) != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return True
    return False


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(-1, fh + 1)
        for dc in range(-1, fw + 1)
        if not (0 <= dr < fh and 0 <= dc < fw)
    ]


def _try_attach_frame(
    canvas: np.ndarray,
    anchor_pixels: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    if not anchor_pixels:
        return None
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        lp = rng.choice(anchor_pixels)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
            return False
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _bias_for(pt: tuple[int, int], h: int, w: int, rng: random.Random) -> tuple[int, int]:
    dr_bias = 1 if pt[0] < h // 2 else -1
    dc_bias = 1 if pt[1] < w // 2 else -1
    if rng.random() < 0.25:
        dr_bias *= -1
    if rng.random() < 0.25:
        dc_bias *= -1
    return dr_bias, dc_bias


def _weighted_dirs(
    dr_bias: int,
    dc_bias: int,
    prev_dir: tuple[int, int] | None,
    rng: random.Random,
) -> list[tuple[int, int]]:
    dirs = []
    for ddr, ddc in DIAG_DIRS:
        aligned = ddr == dr_bias or ddc == dc_bias
        weight = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
        if prev_dir is not None and (ddr, ddc) != prev_dir:
            weight += TURN_BONUS
        dirs.extend([(ddr, ddc)] * max(1, weight))
    for ddr, ddc in ORTHO_DIRS:
        aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
        weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
        if prev_dir is not None and (ddr, ddc) != prev_dir:
            weight += TURN_BONUS
        dirs.extend([(ddr, ddc)] * max(1, weight))
    rng.shuffle(dirs)
    return dirs


def _can_place_line_pixel(canvas: np.ndarray, parent: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = canvas.shape
    r, c = cell
    if not _in_bounds(h, w, r, c) or int(canvas[r, c]) != -1:
        return False
    for nr, nc in neighbors8(r, c):
        if not _in_bounds(h, w, nr, nc) or (nr, nc) == parent:
            continue
        if int(canvas[nr, nc]) != -1:
            return False
    return True


def _add_line_edge(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    parent: tuple[int, int],
    child: tuple[int, int],
    line_col: int,
) -> None:
    canvas[child] = line_col
    tree_set.add(child)
    degrees.setdefault(parent, 0)
    degrees.setdefault(child, 0)
    degrees[parent] += 1
    degrees[child] += 1


def _extend_tip_path_inplace(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    tip: tuple[int, int],
    target_extra: int,
    line_col: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    cur = tip
    prev_dir = None
    dr_bias, dc_bias = _bias_for(tip, h, w, rng)
    added: list[tuple[int, int]] = []
    stalls = 0
    while len(added) < target_extra and stalls < PATH_STALL_LIMIT:
        dirs = _weighted_dirs(dr_bias, dc_bias, prev_dir, rng)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            if not _can_place_line_pixel(canvas, cur, (nr, nc)):
                continue
            _add_line_edge(canvas, tree_set, degrees, cur, (nr, nc), line_col)
            cur = (nr, nc)
            added.append(cur)
            prev_dir = (ddr, ddc)
            placed = True
            break
        if placed:
            stalls = 0
            if rng.random() < 0.15:
                dr_bias, dc_bias = _bias_for(cur, h, w, rng)
        else:
            stalls += 1
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
    return added


def _line_neighbors_of(tree_set: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    return [(nr, nc) for nr, nc in neighbors8(*pt) if (nr, nc) in tree_set]


def _is_bend_node(tree_set: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    neighs = _line_neighbors_of(tree_set, pt)
    if len(neighs) != 2:
        return False
    (r1, c1), (r2, c2) = neighs
    v1 = (r1 - pt[0], c1 - pt[1])
    v2 = (r2 - pt[0], c2 - pt[1])
    return not (v1[0] == -v2[0] and v1[1] == -v2[1])


def _spawn_branch_trial(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    base: tuple[int, int],
    target_len: int,
    line_col: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    trial_canvas = canvas.copy()
    trial_tree = set(tree_set)
    trial_degrees = dict(degrees)
    h, w = canvas.shape
    dr_bias, dc_bias = _bias_for(base, h, w, rng)
    for ddr, ddc in _weighted_dirs(dr_bias, dc_bias, None, rng):
        first = (base[0] + ddr, base[1] + ddc)
        if not _can_place_line_pixel(trial_canvas, base, first):
            continue
        _add_line_edge(trial_canvas, trial_tree, trial_degrees, base, first, line_col)
        path = [first]
        if target_len > 1:
            path.extend(
                _extend_tip_path_inplace(
                    trial_canvas,
                    trial_tree,
                    trial_degrees,
                    first,
                    target_len - 1,
                    line_col,
                    rng,
                )
            )
        if len(path) < MIN_BRANCH_LEN:
            continue
        canvas[:, :] = trial_canvas
        tree_set.clear()
        tree_set.update(trial_tree)
        degrees.clear()
        degrees.update(trial_degrees)
        return path
    return []


def _grow_branching_tree(
    canvas: np.ndarray,
    seed_rc: tuple[int, int],
    first_rc: tuple[int, int],
    line_col: int,
    branch_count: int,
    rng: random.Random,
) -> dict | None:
    h, w = canvas.shape
    tree_set = {first_rc}
    degrees = {first_rc: 0}

    trunk_hi = min(TRUNK_LEN_RANGE[1], max(TRUNK_LEN_RANGE[0], h + w - 6))
    trunk_target = rng.randint(TRUNK_LEN_RANGE[0], trunk_hi)
    trunk_path = [first_rc]
    trunk_path.extend(_extend_tip_path_inplace(canvas, tree_set, degrees, first_rc, trunk_target - 1, line_col, rng))
    if len(trunk_path) < MIN_TRUNK_LEN:
        return None

    segments = [trunk_path]
    for _ in range(branch_count):
        branch_hi = min(BRANCH_LEN_RANGE[1], max(BRANCH_LEN_RANGE[0], (h + w) // 2))
        branch_target = rng.randint(BRANCH_LEN_RANGE[0], branch_hi)
        placed = []
        bend_bases = [pt for pt in trunk_path if pt != first_rc and _is_bend_node(tree_set, pt)]
        bend_bases.extend(pt for seg in segments[1:] for pt in seg if _is_bend_node(tree_set, pt))
        rng.shuffle(bend_bases)
        for base in bend_bases[:BRANCH_ATTACH_TRIES]:
            placed = _spawn_branch_trial(canvas, tree_set, degrees, base, branch_target, line_col, rng)
            if placed:
                break
        if not placed:
            return None
        segments.append(placed)

    if len(tree_set) < MIN_TRUNK_LEN + branch_count * MIN_BRANCH_LEN:
        return None
    if branch_count > 0 and not any(deg >= 3 for deg in degrees.values()):
        return None

    return {
        "seed": seed_rc,
        "tree_set": tree_set,
        "degrees": degrees,
        "segments": segments,
        "branch_count": len(segments) - 1,
    }


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        seed = (0, 0)
        first = rng.choice([(0, 1), (1, 0)])
    elif corner_kind == "TR":
        seed = (0, w - 1)
        first = rng.choice([(0, w - 2), (1, w - 1)])
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        first = rng.choice([(h - 1, 1), (h - 2, 0)])
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        first = rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    else:
        raise ValueError(corner_kind)
    return seed, first


def _distribute_branch_budget(num_seeds: int, rng: random.Random) -> list[int]:
    total = rng.randint(*(TOTAL_BRANCHES_ONE_SEED if num_seeds == 1 else TOTAL_BRANCHES_TWO_SEEDS))
    out = [0] * num_seeds
    for _ in range(total):
        choices = [i for i in range(num_seeds) if out[i] < MAX_BRANCHES_PER_SEED]
        if not choices:
            break
        out[rng.choice(choices)] += 1
    if sum(out) == 0:
        out[rng.randrange(num_seeds)] = 1
    return out


def _attach_seed_loops(
    canvas: np.ndarray,
    tree_info: dict,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
) -> dict | None:
    segments = tree_info["segments"]
    trunk = segments[0]
    branches = segments[1:]

    branch_loops = 0
    for group in branches + [trunk]:
        frame_col = rng.choice(frame_colors_pool)
        placed = _try_attach_frame(canvas, group, line_col, frame_col, rng)
        if placed is None:
            return None
        if group is not trunk:
            branch_loops += 1

    extras = rng.randint(*EXTRA_LOOPS_PER_SEED)
    for _ in range(extras):
        frame_col = rng.choice(frame_colors_pool)
        _try_attach_frame(canvas, rng.choice(segments), line_col, frame_col, rng)

    return {"branch_loops": branch_loops, "segment_count": len(segments)}


def _count_filled_pixels(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.sum(inp != out))


def _count_branch_nodes(line_pts: set[tuple[int, int]]) -> int:
    count = 0
    for pt in line_pts:
        deg = sum((nr, nc) in line_pts for nr, nc in neighbors8(*pt))
        if deg >= 3:
            count += 1
    return count


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
    frame_goal: int,
) -> int:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        if h < 8 or w < 8:
            return 0
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(int(canvas[r + dr, c + dc]) != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue

        canvas[r, c] = line_col
        tree_set = {(r, c)}
        degrees = {(r, c): 0}
        path = [(r, c)]
        target_len = rng.randint(*DISTRACTOR_LINE_LEN)
        path.extend(_extend_tip_path_inplace(canvas, tree_set, degrees, (r, c), target_len - 1, line_col, rng))
        if len(path) < 3:
            for pr, pc in tree_set:
                canvas[pr, pc] = -1
            continue

        placed_cells = set(tree_set)
        attached = 0
        attempts = max(frame_goal * 2, frame_goal + 1)
        for _ in range(attempts):
            if attached >= frame_goal:
                break
            frame_col = rng.choice(frame_colors_pool)
            placed = _try_attach_frame(canvas, path, line_col, frame_col, rng)
            if placed is None:
                continue
            ring, _ = placed
            placed_cells.update(ring)
            attached += 1

        if attached >= frame_goal:
            return attached

        for pr, pc in placed_cells:
            canvas[pr, pc] = -1
    return 0


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if num_seeds * 2 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    per_seed_branches = _distribute_branch_budget(num_seeds, rng)
    lines_info = []
    total_actual_branches = 0
    total_branch_loops = 0

    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]

        tree_info = _grow_branching_tree(canvas, seed_rc, first_rc, line_colors[i], per_seed_branches[i], rng)
        if tree_info is None:
            return None

        loop_info = _attach_seed_loops(canvas, tree_info, line_colors[i], frame_colors_pool, rng)
        if loop_info is None:
            return None

        total_actual_branches += int(tree_info["branch_count"])
        total_branch_loops += int(loop_info["branch_loops"])
        lines_info.append(
            {
                "seed_col": seed_colors[i],
                "seed_rc": seed_rc,
                "line_col": line_colors[i],
                "tree_info": tree_info,
            }
        )

    if total_actual_branches == 0 or total_branch_loops == 0:
        return None

    distractor_owner = rng.choice(lines_info)
    distractor_goal = rng.randint(*DISTRACTOR_NETWORK_FRAMES)
    placed_distractor_loops = _try_place_distractor_network(
        canvas,
        distractor_owner["line_col"],
        frame_colors_pool,
        rng,
        distractor_goal,
    )
    if placed_distractor_loops == 0:
        return None

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for info in lines_info:
        if int(np.sum(grid == info["seed_col"])) != 1:
            return None

    branchy_seed_count = 0
    for info in lines_info:
        seed_rc = info["seed_rc"]
        line_col = info["line_col"]
        owned_nonbg = _nonbg_component8_from(grid, seed_rc, bg)
        line_pts = {pt for pt in owned_nonbg if int(grid[pt]) == line_col}
        if _count_branch_nodes(line_pts) > 0:
            branchy_seed_count += 1
    if branchy_seed_count == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    naive = _solve_with_connectivity_mode(grid, require_connected_line=False)
    if np.array_equal(naive, out):
        return None

    if _count_filled_pixels(grid, out) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
