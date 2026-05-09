# Combo variant of 332f06d7 (index 16)
# combo_spec: D7/0+D14/0
# Name: 紧凑裁剪+稀疏窄通道

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


"""
design_analysis:
  a_concretization_and_trace:
    original_pipeline: |
      原题输入是整张方形背景图，蓝色通道占比较大，黑色 n×n 方块沿可容纳整个 n×n
      窗口的蓝道滑动；若一路都有宽度 >= n 的可行放置，则黑块到红块位置，否则在第一个
      几何卡口前的最远可达放置处停下。输出保留原图尺寸。
    fused_pipeline: |
      融合后，滑动物理保持不变，但有两个联动改动：
      1. 输入端：通道改成稀疏细线式结构，整张输入里只有 10%-20% 的非背景像素，绝大多数
         位置都是背景，蓝道大多只“刚好够 n×n 方块通过”。
      2. 输出端：先按原滑动规则得到整张结果，再把结果裁成“包含整条通道区域的最小矩形”。
      小例子：设输入是 18×18，内部只有一条折线状 2 像素宽蓝道，黑色 2×2 方块在左下，
      红色 2×2 方块在右上。若中段没有被挖成 1 像素卡口，黑块滑到红块；若中段被挖出
      1 像素卡口，黑块停在卡口前。最后无论到达与否，输出都不再保留 18×18 外框，而是
      只保留围住这条折线河道的紧凑子矩形。
    visible_difference: |
      与原题相比，解题者现在要同时看出“黑块在窄通道里的移动/受阻规则”和“输出把整条
      通道外的背景边框全部裁掉”。
  b_complexity_calibration:
    notes:
      - 如果只做裁剪，不约束通道稀疏度，输入和输出差异会弱化成“去掉一点边框”。
      - 如果只做稀疏化，不做裁剪，则仍是原题同尺寸输出。
      - 因此生成器同时强制：输入稀疏、输出必然变小、且通道仍要有 >=3 次转折，避免退化成
        简单直线滑块题。
      - 方块尺寸限制在 n∈{2,3}，避免 n=1 时“窄于 n 的卡口”不可见。
  c_implicit_assumption_audit:
    main_line_by_line:
      - 背景仍用四角多数色识别；生成器保证外层 padding 纯背景，所以该前提成立。
      - 不再依赖“按颜色数值排序决定 mover/dest”，而是直接使用可见且稳定的颜色语义：
        黑色(0)=mover，红色(2)=destination。
      - 仍假设黑块和红块都是实心同尺寸正方形；生成器显式构造并验证这一点。
      - 仍假设除背景与黑/红外，像素数最多的剩余颜色是河道色；融合版只保留一条蓝河，
        没有额外语义色，因此该假设成立。
      - BFS 仍然在 n×n 顶点放置网格上进行；D14/0 没有改变物理规则，只是把大部分通道做成
        恰好 n 宽，并在 stuck 样例里把某处挖成 n-1。
      - D7/0 破坏了“输出尺寸等于输入尺寸”的旧假设，所以 solver 最后必须改为返回 tight crop。
  d_placement_legality:
    generator_guarantees:
      - 路径是在 top-left 放置坐标网格上生成的非自交、非自接触折线，画成像素河道后不会越界。
      - mover 和 dest 只覆盖河道中的两个 n×n 放置位，不重叠。
      - stuck 样例只在路径较后段的直线河段上挖卡口，避免在起点处直接堵死。
      - 每个候选样例都回算 `_analyze_motion()` 与 `main()`，不满足 reachability / crop /
        稀疏度约束的一律丢弃。
  e_visual_observability:
    human_view: |
      解题者看到 2-4 组样例时，可以直接观察到：
      1. 黑块沿细蓝道移动；
      2. 有些样例在更窄的部位前停下；
      3. 输出只保留通道所在的紧凑矩形，而不是原始大画布。
      这些都是肉眼可见的空间结构，不依赖颜色编号排序或隐藏随机参数。
  f_cleanliness:
    notes:
      - 只使用四种原色：背景绿、河道蓝、 mover 黑、 target 红。
      - 路径被限制为非自接触折线，避免生成大块蓝团导致“哪里是通道、哪里是宽水面”不清。
      - 裁剪发生在外层 padding 上，不会在逻辑区域内部叠加额外标记。
  g_change_inferability:
    notes:
      - 没有新颜色映射；唯一的颜色变化仍是黑块移动与红块被覆盖/保留。
      - 新增可推断结构来自“外框被裁掉”和“背景远多于河道”，而不是无提示的颜色重编码。
  h_quantified_behavior_target:
    generator_filters:
      - 输入非背景占比强制在 10%-20%。
      - 输出 tight crop 后必须在两个维度上都小于输入。
      - 裁剪后的包围盒面积相对输入面积也必须显著缩小，防止退化成几乎同尺寸的假裁剪。
  i_new_main_assumptions_and_generator_match:
    assumptions:
      - 恰有一个黑色实心方块和一个红色实心方块，且边长相同。
      - 蓝色是唯一河道色，其他颜色不会伪装成河道。
      - 至少存在一条河道，因此裁剪后不会空。
    matched_by_generator: |
      生成器逐项满足这些条件，并在返回前调用 `main()` 自验。
"""


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _detect_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _find_named_squares(grid: np.ndarray, bg: int) -> tuple[dict, dict]:
    squares: dict[int, dict] = {}
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col not in (_MOVER, _DEST):
            continue
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = grid[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares[col] = {"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))}

    assert _MOVER in squares and _DEST in squares
    assert int(squares[_MOVER]["n"]) == int(squares[_DEST]["n"])
    return squares[_MOVER], squares[_DEST]


def _remaining_colors(grid: np.ndarray, excluded: set[int]) -> list[tuple[int, int]]:
    counts = Counter(map(int, grid.flatten()))
    items = [(int(col), int(ct)) for col, ct in counts.items() if int(col) not in excluded]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items


def _feasible_mask(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - n + 1
    w2 = W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _pick_terminal(dist: np.ndarray, feasible: np.ndarray, sr: int, sc: int, dr: int, dc: int) -> tuple[int, int, bool]:
    if 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and feasible[dr, dc] and dist[dr, dc] >= 0:
        return dr, dc, True

    best = -1
    tr, tc = sr, sc
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc, False


def _analyze_motion(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = _detect_bg(g)
    mover, dest = _find_named_squares(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    assert 0 <= sr <= H - n and 0 <= sc <= W - n

    rem = _remaining_colors(g, {bg, _MOVER, _DEST})
    assert rem
    river = int(rem[0][0])

    trav = (g == river) | (g == _MOVER) | (g == _DEST)
    feasible = _feasible_mask(trav, n)
    assert bool(feasible[sr, sc])
    dist = _bfs_placement(feasible, sr, sc)
    tr, tc, dest_ok = _pick_terminal(dist, feasible, sr, sc, dr, dc)
    return {
        "bg": bg,
        "river": river,
        "n": n,
        "sr": sr,
        "sc": sc,
        "dr": dr,
        "dc": dc,
        "dest_ok": bool(dest_ok),
        "tr": int(tr),
        "tc": int(tc),
        "feasible": feasible,
        "dist": dist,
    }


def _solve_full(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    info = _analyze_motion(g)
    n = int(info["n"])
    sr, sc = int(info["sr"]), int(info["sc"])
    tr, tc = int(info["tr"]), int(info["tc"])
    river = int(info["river"])

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Original sliding-square physics on a very sparse narrow river, followed by a tight crop
    to the smallest rectangle containing the whole final river region.
    """
    g = np.array(grid, dtype=int)
    bg = _detect_bg(g)
    full = _solve_full(g)
    return _tight_crop_non_bg(full, bg)


def _admissible_step(prev: tuple[int, int], nxt: tuple[int, int], visited: set[tuple[int, int]], h2: int, w2: int) -> bool:
    r, c = nxt
    if not (0 <= r < h2 and 0 <= c < w2):
        return False
    if nxt in visited:
        return False
    for dr, dc in _DIR4:
        nb = (r + dr, c + dc)
        if nb in visited and nb != prev:
            return False
    return True


def _future_degree(prev: tuple[int, int], cur: tuple[int, int], visited: set[tuple[int, int]], h2: int, w2: int) -> int:
    seen = set(visited)
    seen.add(cur)
    deg = 0
    for dr, dc in _DIR4:
        nxt = (cur[0] + dr, cur[1] + dc)
        if _admissible_step(cur, nxt, seen, h2, w2):
            deg += 1
    return deg


def _walk_turns(path: list[tuple[int, int]]) -> int:
    dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    return sum(int(dirs[i] != dirs[i - 1]) for i in range(1, len(dirs)))


def _sample_non_touching_walk(h2: int, w2: int, target_len: int, min_turns: int = 3) -> list[tuple[int, int]] | None:
    target_len = int(target_len)
    if target_len < 5 or target_len > h2 * w2:
        return None

    for _attempt in range(500):
        start = (random.randrange(h2), random.randrange(w2))
        path = [start]
        visited = {start}

        while len(path) < target_len:
            prev_dir = None
            if len(path) >= 2:
                prev_dir = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])

            remaining = target_len - len(path)
            candidates: list[tuple[int, int, bool, tuple[int, int]]] = []
            dirs = list(_DIR4)
            random.shuffle(dirs)
            for dr, dc in dirs:
                nxt = (path[-1][0] + dr, path[-1][1] + dc)
                if not _admissible_step(path[-1], nxt, visited, h2, w2):
                    continue
                turn = prev_dir is not None and (dr, dc) != prev_dir
                future = _future_degree(path[-1], nxt, visited, h2, w2)
                if remaining > 1 and future == 0:
                    continue
                candidates.append((future, random.randint(0, 1), bool(turn), nxt))

            if not candidates:
                break

            candidates.sort(key=lambda item: (item[2], item[0], item[1]), reverse=True)
            nxt = random.choice(candidates[: min(3, len(candidates))])[3]
            visited.add(nxt)
            path.append(nxt)

        if len(path) != target_len:
            continue
        if _walk_turns(path) < min_turns:
            continue
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 2:
            continue
        if abs(path[0][0] - path[-1][0]) + abs(path[0][1] - path[-1][1]) < 4:
            continue
        return path

    return None


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], n: int) -> None:
    for r, c in path:
        grid[r : r + n, c : c + n] = _RIVER


def _add_chokepoint(
    grid: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    n: int,
    direction: str,
) -> None:
    H, W = grid.shape
    pr1, pc1 = p0
    pr2, pc2 = p1
    narrow_w = random.randint(1, n - 1)
    remove = n - narrow_w
    side = random.randint(0, 1)

    if direction == "h":
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        hold_len = random.randint(1, min(4, max(1, c_hi - c_lo - 2 * remove)))
        total_len = 2 * remove + max(1, hold_len)
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if not (0 <= c < W):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < H:
                    grid[rr, c] = _BG
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        hold_len = random.randint(1, min(4, max(1, r_hi - r_lo - 2 * remove)))
        total_len = 2 * remove + max(1, hold_len)
        r_mid = (r_lo + r_hi) // 2
        r_start = max(r_lo, r_mid - total_len // 2)
        for step in range(total_len):
            r = r_start + step
            if not (0 <= r < H):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= cc < W:
                    grid[r, cc] = _BG


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = (int(pts[0, 0]), int(pts[0, 1]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _embed_with_padding(core: np.ndarray, G: int, pad_min: int = 2) -> np.ndarray | None:
    ch, cw = core.shape
    extra_r = G - ch
    extra_c = G - cw
    if extra_r < 2 * pad_min or extra_c < 2 * pad_min:
        return None

    top = random.randint(pad_min, extra_r - pad_min)
    left = random.randint(pad_min, extra_c - pad_min)
    bottom = extra_r - top
    right = extra_c - left
    if min(top, bottom, left, right) < pad_min:
        return None

    grid = np.full((G, G), _BG, dtype=int)
    grid[top : top + ch, left : left + cw] = core
    return grid


def _input_metrics(grid: np.ndarray) -> dict:
    bg = _detect_bg(grid)
    crop = _tight_crop_non_bg(grid, bg)
    non_bg_ratio = float((grid != bg).sum()) / float(grid.size)
    row_ratio = float(np.any(grid != bg, axis=1).sum()) / float(grid.shape[0])
    col_ratio = float(np.any(grid != bg, axis=0).sum()) / float(grid.shape[1])
    crop_area_ratio = float(crop.size) / float(grid.size)
    crop_fill_ratio = float((crop != bg).sum()) / float(crop.size)
    return {
        "non_bg_ratio": non_bg_ratio,
        "row_ratio": row_ratio,
        "col_ratio": col_ratio,
        "crop_area_ratio": crop_area_ratio,
        "crop_fill_ratio": crop_fill_ratio,
    }


def _acceptable_sparse_metrics(grid: np.ndarray) -> bool:
    m = _input_metrics(grid)
    return (
        0.10 <= m["non_bg_ratio"] <= 0.20
        and m["row_ratio"] <= 0.70
        and m["col_ratio"] <= 0.70
        and m["crop_area_ratio"] <= 0.72
        and 0.18 <= m["crop_fill_ratio"] <= 0.55
    )


def _straight_segment_indices(path: list[tuple[int, int]]) -> list[int]:
    dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    idxs = [i for i in range(1, len(dirs)) if dirs[i] == dirs[i - 1]]
    if not idxs:
        idxs = list(range(1, len(path) - 2))
    return idxs


def _gen_attempt() -> np.ndarray | None:
    G = random.randint(16, 20)
    n = random.choice([2, 3])
    reached = random.random() < 0.55
    work = G - 4
    h2 = work - n + 1
    if h2 < 6:
        return None

    target_ratio = random.uniform(0.11, 0.18)
    target_area = target_ratio * G * G
    target_len = max(8, int(round((target_area - n * n) / n)) + 1)
    target_len = min(target_len, max(8, (h2 * h2) // 2))

    path = None
    for length in [target_len, target_len - 1, target_len + 1, target_len - 2, target_len + 2]:
        if length < 8:
            continue
        path = _sample_non_touching_walk(h2, h2, length)
        if path is not None:
            break
    if path is None:
        return None

    river_grid = np.full((work, work), _BG, dtype=int)
    _paint_path(river_grid, path, n)

    sr, sc = path[0]
    dr, dc = path[-1]

    if reached:
        puzzle = river_grid.copy()
        puzzle[sr : sr + n, sc : sc + n] = _MOVER
        puzzle[dr : dr + n, dc : dc + n] = _DEST
        try:
            info = _analyze_motion(puzzle)
        except Exception:
            return None
        if not info["dest_ok"]:
            return None
    else:
        puzzle = None
        candidates = _straight_segment_indices(path)
        candidates = [idx for idx in candidates if idx >= max(2, len(path) // 2)]
        if not candidates:
            candidates = _straight_segment_indices(path)
        random.shuffle(candidates)
        for idx in candidates:
            test = river_grid.copy()
            p0 = path[idx]
            p1 = path[idx + 1]
            direction = "h" if p0[0] == p1[0] else "v"
            _add_chokepoint(test, p0, p1, n, direction)
            if not _mask_connected(test == _RIVER):
                continue
            trial = test.copy()
            trial[sr : sr + n, sc : sc + n] = _MOVER
            trial[dr : dr + n, dc : dc + n] = _DEST
            try:
                info = _analyze_motion(trial)
            except Exception:
                continue
            if info["dest_ok"]:
                continue
            if (int(info["tr"]), int(info["tc"])) == (sr, sc):
                continue
            puzzle = trial
            break
        if puzzle is None:
            return None

    core = _tight_crop_non_bg(puzzle, _BG)
    embedded = _embed_with_padding(core, G, pad_min=2)
    if embedded is None or not _acceptable_sparse_metrics(embedded):
        return None

    try:
        info2 = _analyze_motion(embedded)
        actual = main(embedded)
        expected = _tight_crop_non_bg(_solve_full(embedded), _detect_bg(embedded))
    except Exception:
        return None

    if bool(info2["dest_ok"]) != bool(reached):
        return None
    if not np.array_equal(actual, expected):
        return None
    if actual.shape[0] >= embedded.shape[0] or actual.shape[1] >= embedded.shape[1]:
        return None
    return embedded


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid fused variant input after 600 attempts")
