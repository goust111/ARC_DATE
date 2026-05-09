# Combo variant of 5961cc34 (index 23)
# combo_spec: D1/0+D8/1
# Name: 对角线射线+颜色循环

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis
# a) 具体化与行为追踪
#    - 原题里，底部红杆+黄色帽只会向正上方发一条纯红细射线；命中蓝块后，
#      蓝块再按绿色端口的正交方向发纯红粗射线。
#    - 这个融合版先采用 D1/0：黄色帽改为相对红杆顶部的左上或右上单格，
#      因而起始射线必定沿 45 度对角线前进；blob 端口几何仍沿 8 方向模板
#      建模，但生成器只使用其中 4 个对角方向，让所有真正的可见射线都保持
#      45 度斜向传播。
#    - 再加入 D8/1：所有“真正的射线路径格”不再统一涂红，而是按
#      [红, 橙, 红, 橙, ...] 交替；但每种颜色连续维持的步数 = 发射源的
#      端口像素数。于是：
#        * 起始射线等价于 1 像素端口，颜色序列为 2,7,2,7,...
#        * 1 像素端口发射时，同样逐格红橙交替
#        * 2 像素端口发射时，沿每条 lane 呈现 2 格红、2 格橙、2 格红...
#      小例子：若某个宽度 2 的对角端口向右上发射，则两条平行斜线都会出现
#      RR OO RR OO 的分段，从视觉上就是彩色斜条纹。
#    - 被命中的蓝 blob 仍整体转成红色；只有“路径轨迹”做红/橙循环。这让
#      人眼能区分“被激活的对象”和“对象之间的传播轨迹”。
# b) 复杂度校准
#    - 颜色只新增橙色一种，周期也固定为双色，避免把任务变成任意颜色映射。
#    - 端口宽度仍限制在 1 或 2，正好给出两档可见的周期块长，既明显又可学。
#    - 生成器强制链长至少为 1，并额外要求至少存在一个被激活 blob 的 2 像素
#      端口，这样样例不会退化成“只是对角线 + 固定单格交替”。
# c) 隐式假设审查
#    - 原 solver 通过 4 邻接绿像素推端口方向；对角端口会破坏这个假设，所以
#      必须像 D1 单变体那样，按已知 diamond/octagon 模板恢复端口几何。
#    - 原 solver 在“先涂色再检查命中”时是安全的，因为所有目标都同色。这里若
#      先涂色就会把命中 blob 的首格误涂成橙色，因此新 solver 必须先判定
#      `g[r,c] == blue_col`，命中则激活 blob，否则才按周期上色。
#    - 原 generator 的放置逻辑只需要轴对齐 entry window；对角线后必须改成
#      “终点是否等于起点 + k*(dr,dc)” 的统一步进判断。
#    - 新增假设：周期块长由发射宽度决定。solver 对此的依赖是“起始源宽度=1，
#      blob 端口宽度 = 可见蓝 port pixels 个数”；generator 必须确保端口模板
#      唯一、无遮挡，并且 2 像素端口在输出里实际发射出足够长的可见轨迹。
# d) 放置合法性
#    - 仍沿用 output-first：先规划链，再反推输入。
#    - 候选 blob 必须整体在界内，且与已有对象做 8 邻域间隔；其入射路径只能
#      穿过背景，唯一允许的“已占用首步”是从蓝 port pixel 走到本 blob 自己的
#      绿色端口格。
#    - 最终边界射线同样禁止穿过旧 ray、旧 blob 或 distractor。
# e) 视觉可观察性
#    - 每个样例都出现对角起始帽，因此 D1/0 的变化始终可见。
#    - 输出只用红/橙两种轨迹色，且橙色只会出现在射线上，学习者容易把颜色变化
#      归因到“轨迹规则”，而非 blob 身份。
#    - 起始射线提供 1 像素周期样本；生成器再强制至少一个 2 像素端口发射，故
#      两档周期块长在少量样例中就能重复出现。
# f) 可视化整洁度
#    - ray 与 blob 仍采用原任务的稀疏链式布局；distractor 永远不落在任何 ray 上。
#    - 输出中每个格子只承担一种语义：背景、红 blob、或红/橙路径，不引入多层覆盖。
# g) 变化的可推断性
#    - 颜色变化并非依赖颜色编号，而是绑定到一个可见结构量：发射宽度。
#    - 规则是“固定双色循环 + 宽度决定每色连续步数”，而不是为不同 blob 任意
#      指派颜色，因此 2-4 个样例足以归纳。


BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
UP_LEFT = (-1, -1)
UP_RIGHT = (-1, 1)
DOWN_LEFT = (1, -1)
DOWN_RIGHT = (1, 1)
DIRS = (UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
OPP = {d: (-d[0], -d[1]) for d in DIRS}
DIAGONALS = {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}
RAY_DIRS = (UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)

SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 3
START_H_MIN, START_H_MAX = 2, 3
START_DIRS = (UP_LEFT, UP_RIGHT)

CHAIN_LEN_WEIGHTS = [0, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3
MIN_WIDE_RAY_CELLS = 4
GEN_RETRIES = 400


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Build a boolean diamond/octagon mask for the given (h, w) shape."""
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _lane_key(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Invariant shared by all points on the same ray lane in direction d."""
    r, c = rc
    dr, dc = d
    return int(r * dc - c * dr)


def _along_key(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Monotone key along the travel direction."""
    r, c = rc
    dr, dc = d
    return int(r * dr + c * dc)


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort parallel lanes consistently for a given direction."""
    return sorted(coords, key=lambda p: (_lane_key(p, d), _along_key(p, d), p[0], p[1]))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return mask-local coordinates on the cardinal edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _diag_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return 1-2 mask-local cells on the diagonal frontier facing direction d."""
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    frontier_score = max(int(r * d[0] + c * d[1]) for r, c in pts)
    frontier = [(r, c) for r, c in pts if int(r * d[0] + c * d[1]) == frontier_score]
    frontier.sort(key=lambda p: (p[0], p[1]), reverse=(d[0] > 0))
    return frontier[: min(2, len(frontier))]


def _port_green_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the generated green port cells for a given direction."""
    return _diag_offsets(mask, d) if d in DIAGONALS else _edge_offsets(mask, d)


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in coords)


def _steps_along(
    start: tuple[int, int], end: tuple[int, int], d: tuple[int, int]
) -> int | None:
    """Return k if end == start + k*d for some k>=0, else None."""
    sr, sc = start
    er, ec = end
    dr, dc = d
    rr = int(er - sr)
    cc = int(ec - sc)

    if dr == 0:
        if rr != 0 or dc == 0 or cc % dc != 0:
            return None
        k = cc // dc
        return int(k) if k >= 0 else None
    if dc == 0:
        if cc != 0 or rr % dr != 0:
            return None
        k = rr // dr
        return int(k) if k >= 0 else None
    if rr % dr != 0 or cc % dc != 0:
        return None
    kr = rr // dr
    kc = cc // dc
    if kr != kc or kr < 0:
        return None
    return int(kr)


def _flatten_lanes(lanes: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Flatten lane-wise traced cells."""
    return [cell for lane in lanes for cell in lane]


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    H: int,
    W: int,
) -> tuple[list[list[tuple[int, int]]], list[int]] | None:
    """Trace ray lanes from starts to entry cells; returned lanes exclude the source cell."""
    lanes: list[list[tuple[int, int]]] = []
    lengths: list[int] = []
    dr, dc = d
    for (r, c), (er, ec) in zip(starts, entries):
        k = _steps_along((r, c), (er, ec), d)
        if k is None:
            return None
        lane: list[tuple[int, int]] = []
        for _ in range(int(k)):
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                return None
            lane.append((int(r), int(c)))
        lanes.append(lane)
        lengths.append(int(k))
    return lanes, lengths


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    """Trace ray lanes from starts to the grid boundary; lanes exclude the source cell."""
    dr, dc = d
    lanes: list[list[tuple[int, int]]] = []
    for r, c in starts:
        lane: list[tuple[int, int]] = []
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            lane.append((int(r), int(c)))
        lanes.append(lane)
    return lanes


def _ray_color(distance: int, period: int) -> int:
    """Visible ray color at 1-based distance from the source."""
    assert distance >= 1
    assert period >= 1
    return int(RED if ((distance - 1) // period) % 2 == 0 else ORANGE)


def _paint_cyclic_lanes(
    out: np.ndarray,
    ray_mask: np.ndarray,
    lanes: list[list[tuple[int, int]]],
    period: int,
) -> None:
    """Paint traced lane cells using the fixed red/orange cycle."""
    for lane in lanes:
        for step_idx, (r, c) in enumerate(lane, start=1):
            out[int(r), int(c)] = _ray_color(step_idx, int(period))
            ray_mask[int(r), int(c)] = True


def _count_cycle_color(lengths: list[int], period: int, color: int) -> int:
    """Count how many visible cells of a lane bundle take a specific cycle color."""
    total = 0
    for length in lengths:
        for step in range(1, int(length) + 1):
            if _ray_color(step, period) == int(color):
                total += 1
    return int(total)


def _count_same_color_pairs(lengths: list[int], period: int) -> int:
    """Count adjacent same-color pairs along lane-local ray order."""
    total = 0
    for length in lengths:
        prev = None
        for step in range(1, int(length) + 1):
            cur = _ray_color(step, period)
            if prev == cur:
                total += 1
            prev = cur
    return int(total)


def _differs_from_period1(lengths: list[int], period: int) -> bool:
    """Whether any visible cell would differ from the width-1 alternating baseline."""
    if period == 1:
        return False
    for length in lengths:
        for step in range(1, int(length) + 1):
            if _ray_color(step, period) != _ray_color(step, 1):
                return True
    return False


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask by 1 pixel in all 8 directions."""
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given shape and port direction."""
    h, w = size
    mask = _shape_mask(h, w)
    green = _port_green_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    full_set = set(full)
    green_set = set(green)

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[int(r), int(c)] = GREEN

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br, bc = int(gr - port_dir[0]), int(gc - port_dir[1])
        if (br, bc) not in full_set or (br, bc) in green_set:
            return None
        port_pixels.append((int(br), int(bc)))

    blue = [p for p in full if p not in green_set]
    entry_by_move: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for d in DIRS:
        dr, dc = d
        entries = [
            (int(r), int(c))
            for r, c in blue
            if (int(r - dr), int(c - dc)) not in full_set
        ]
        entry_by_move[d] = _sort_on_ray(entries, d)

    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "full": full,
        "full_set": full_set,
        "blue": blue,
        "green": list(green),
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
        "entry_by_move": entry_by_move,
    }


ALL_SPECS = [s for size in SHAPES for d in RAY_DIRS if (s := _blob_spec(size, d)) is not None]
SPEC_LOOKUP: dict[tuple[tuple[int, int], frozenset[tuple[int, int]]], dict] = {}
for _spec in ALL_SPECS:
    key = (_spec["size"], frozenset(_spec["green"]))
    assert key not in SPEC_LOOKUP
    SPEC_LOOKUP[key] = _spec


def _all_specs() -> list[dict]:
    """Return a shuffled copy of all valid blob specs."""
    specs = list(ALL_SPECS)
    random.shuffle(specs)
    return specs


def _find_start(
    g: np.ndarray, bg: int
) -> tuple[int, int, tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    """Detect the unique red stem + diagonal cap start marker."""
    H, W = g.shape
    start_candidates: list[
        tuple[int, int, tuple[int, int], list[tuple[int, int]], tuple[int, int]]
    ] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

            if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != int(H - 1):
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            top_r = int(r0)
            cap_options: list[tuple[int, int, tuple[int, int]]] = []
            for d in START_DIRS:
                rr = int(top_r + d[0])
                cc = int(c0 + d[1])
                if 0 <= rr < H and 0 <= cc < W:
                    cap = int(g[rr, cc])
                    if cap != int(bg) and cap != int(col):
                        cap_options.append((cap, cc, d))
            if len(cap_options) != 1:
                continue
            cap_col, cap_c, start_dir = cap_options[0]
            cap_pos = (int(top_r + start_dir[0]), int(cap_c))
            start_candidates.append(
                (
                    int(col),
                    int(cap_col),
                    (int(start_dir[0]), int(start_dir[1])),
                    [(int(rr), int(c0)) for rr in range(r0, r1 + 1)],
                    cap_pos,
                )
            )

    assert len(start_candidates) == 1
    return start_candidates[0]


def _match_blob_port(
    g: np.ndarray, blue_pts: list[tuple[int, int]], green_col: int
) -> tuple[dict, tuple[int, int], list[tuple[int, int]]]:
    """Recover a blob's spec and its global blue port pixels from blue+green geometry."""
    H, W = g.shape
    green_pts: set[tuple[int, int]] = set()
    for r, c in blue_pts:
        for dr, dc in DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                green_pts.add((int(rr), int(cc)))

    assert green_pts
    full_pts = set(blue_pts) | green_pts
    rs = [r for r, _ in full_pts]
    cs = [c for _, c in full_pts]
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    size = (int(r1 - r0 + 1), int(c1 - c0 + 1))
    key = (size, frozenset((int(r - r0), int(c - c0)) for r, c in green_pts))
    spec = SPEC_LOOKUP[key]
    assert spec["port_dir"] in DIAGONALS

    local_full = {(int(r - r0), int(c - c0)) for r, c in full_pts}
    assert local_full == spec["full_set"]

    ports = _offset(spec["port_pixels"], r0, c0)
    return spec, (int(r0), int(c0)), ports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a diagonal ray network whose paths alternate red and orange by source width.

    Input structure:
    - The background is the dominant color. Blue diamond/octagon blobs carry embedded green
      port cells on one side or corner, and the bottom start marker is a red vertical stem
      with a single yellow cap placed diagonally left or right of the stem top.

    Rule:
    - The cap offset sets the first 45-degree ray direction. Trace that ray until it hits a
      blue blob or exits the grid; its path colors alternate red/orange with 1-cell blocks.
    - A hit blob turns red and emits through its green port in the port's encoded diagonal
      heading. Each emitted ray uses the same fixed [red, orange] cycle, but each color is
      repeated for as many consecutive steps as there are blue port pixels (here always 2
      for blob ports, versus 1 for the start ray).
    - Propagate breadth-first until no new blue blob is reached; untouched blobs and markers
      disappear back to the background.

    Key insight: The same geometry as the diagonal-ray variant still applies, but now the
    visible stripe length on each ray is determined by the emitter's width, so both direction
    and color rhythm must be read from the start cap and the green port template.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    red_col, yellow_col, start_dir, start_seg, cap_pos = _find_start(g, bg)

    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (bg, red_col, yellow_col)
    ]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            for dr, dc in DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(blue_col):
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, int(bg))

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        spec, _origin, port_pixels = _match_blob_port(g, pts, green_col)
        blobs.append(
            {
                "pts": pts,
                "ports": [{"dir": spec["port_dir"], "pixels": port_pixels}],
            }
        )

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit by a ray; enqueue it to emit its ports."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    def extend_ray(starts: list[tuple[int, int]], d: tuple[int, int], period: int) -> None:
        """Extend a colored ray from each source pixel until boundary or blob hit."""
        ddr, ddc = map(int, d)
        for sr, sc in starts:
            r, c = int(sr), int(sc)
            step = 0
            while True:
                r += int(ddr)
                c += int(ddc)
                step += 1
                if not (0 <= r < H and 0 <= c < W):
                    break
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break
                out[int(r), int(c)] = _ray_color(int(step), int(period))

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    out[int(cap_pos[0]), int(cap_pos[1])] = int(red_col)
    extend_ray([cap_pos], start_dir, period=1)

    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            pixels = port["pixels"]
            extend_ray(pixels, port["dir"], period=len(pixels))

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the width-controlled color-cycle behavior is visibly present."""
    g = np.array(grid, dtype=int)
    out = main(g)
    H, W = g.shape
    bg = int(get_bg(g))
    red_col, yellow_col, start_dir, _start_seg, cap_pos = _find_start(g, bg)

    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (bg, red_col, yellow_col)
    ]
    max_count = max(cnt for cnt, _col in other_cols)
    blue_col = [int(col) for cnt, col in other_cols if cnt == max_count][0]

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            for dr, dc in DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(blue_col):
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))
    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        spec, _origin, port_pixels = _match_blob_port(g, pts, green_col)
        blobs.append({"dir": spec["port_dir"], "pixels": port_pixels})

    activated: set[int] = set()
    queue: list[int] = []

    def activate(r: int, c: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))

    def trace(starts: list[tuple[int, int]], d: tuple[int, int]) -> list[int]:
        dr, dc = d
        lengths: list[int] = []
        for sr, sc in starts:
            r, c = int(sr), int(sc)
            length = 0
            while True:
                r += int(dr)
                c += int(dc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate(int(r), int(c))
                    break
                length += 1
            lengths.append(int(length))
        return lengths

    trace([cap_pos], start_dir)
    wide_emissions = 0
    diff_cells = 0
    same_color_pairs = 0
    while queue:
        bi = int(queue.pop(0))
        starts = blobs[bi]["pixels"]
        lengths = trace(starts, blobs[bi]["dir"])
        if len(starts) != 2:
            continue
        wide_emissions += 1
        for length in lengths:
            for step in range(1, int(length) + 1):
                if _ray_color(step, 2) != _ray_color(step, 1):
                    diff_cells += 1
            prev = None
            for step in range(1, int(length) + 1):
                cur = _ray_color(step, 2)
                if prev == cur:
                    same_color_pairs += 1
                prev = cur

    return {
        "orange_cells": int(np.sum(out == ORANGE)),
        "wide_emissions": int(wide_emissions),
        "diff_cells_from_period1": int(diff_cells),
        "same_color_pairs": int(same_color_pairs),
    }


def _candidate_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    min_gap: int,
    allow_emit_step: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Find a random valid placement for the next blob in the cascade chain."""
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    emit_allow = {
        (int(r + d[0]), int(c + d[1]))
        for r, c in starts
        if 0 <= int(r + d[0]) < H and 0 <= int(c + d[1]) < W
    } if allow_emit_step else set()

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        entry = spec["entry_by_move"][d]
        if len(entry) < thickness:
            continue
        windows = [entry[k : k + thickness] for k in range(len(entry) - thickness + 1)]
        random.shuffle(windows)
        placements = [
            (r0, c0)
            for r0 in range(H - spec["size"][0] + 1)
            for c0 in range(W - spec["size"][1] + 1)
        ]
        random.shuffle(placements)

        for r0, c0 in placements:
            full = _offset(spec["full"], r0, c0)
            if not _in_bounds(full, H, W):
                continue
            if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                continue

            for window in windows:
                entry_cells = _offset(window, r0, c0)
                traced = _path_to_entries(starts, d, entry_cells, H, W)
                if traced is None:
                    continue
                lanes, lengths = traced
                if any(int(k) < min_gap for k in lengths):
                    continue

                path = _flatten_lanes(lanes)
                entry_set = set(entry_cells)
                blocked = False
                for cell in path:
                    if cell in entry_set or cell in emit_allow:
                        continue
                    r, c = cell
                    if ray_mask[r, c] or occupied[r, c] or blue_mask[r, c]:
                        blocked = True
                        break
                if blocked:
                    continue

                return {
                    "spec": spec,
                    "r0": int(r0),
                    "c0": int(c0),
                    "entry": entry_cells,
                    "ray_lanes": lanes,
                    "ray_path": path,
                    "blue": _offset(spec["blue"], r0, c0),
                    "full": full,
                    "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
                }

    return None


def _place_blob(
    inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict
) -> None:
    """Paste a blob's patch onto the input grid and update tracking masks."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


def _place_distractor(
    inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, ray_mask: np.ndarray
) -> bool:
    """Try to place a single distractor blob in free space."""
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["size"]
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": int(r0),
                        "c0": int(c0),
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    top_r = H - start_h
    start_dir = random.choice(START_DIRS)
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    cap_r = int(top_r + start_dir[0])
    cap_c = int(start_col + start_dir[1])
    if not (0 <= cap_r < H and 0 <= cap_c < W):
        return None
    inp[cap_r, cap_c] = YELLOW
    out[cap_r, cap_c] = RED
    occupied[cap_r, cap_c] = True
    ray_mask[cap_r, cap_c] = True
    for r in range(top_r, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    if chain_len == 0:
        return None
    min_dist = 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(cap_r), int(cap_c))]
    direction = start_dir
    min_gap = MIN_GAP_FIRST
    allow_emit_step = False
    has_wide_port = False
    differs_from_alt1 = False
    visible_orange = 0

    for _ in range(chain_len):
        blob = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            min_gap,
            allow_emit_step,
            occupied,
            blue_mask,
            ray_mask,
        )
        if blob is None:
            return None
        _paint_cyclic_lanes(out, ray_mask, blob["ray_lanes"], period=len(starts))
        visible_lengths = [max(0, len(lane) - 1) for lane in blob["ray_lanes"]]
        visible_orange += _count_cycle_color(visible_lengths, len(starts), ORANGE)
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        min_gap = MIN_GAP_CHAIN
        allow_emit_step = True
        if len(starts) == 2:
            has_wide_port = True
            differs_from_alt1 |= _differs_from_period1(visible_lengths, period=2)

    final_lanes = _path_to_boundary(starts, direction, H, W)
    final_path = _flatten_lanes(final_lanes)
    emit_allow = {
        (int(r + direction[0]), int(c + direction[1]))
        for r, c in starts
        if 0 <= int(r + direction[0]) < H and 0 <= int(c + direction[1]) < W
    }
    for r, c in final_path:
        if (r, c) in emit_allow:
            continue
        if blue_mask[r, c] or ray_mask[r, c] or occupied[r, c]:
            return None
    _paint_cyclic_lanes(out, ray_mask, final_lanes, period=len(starts))
    final_visible_lengths = [len(lane) for lane in final_lanes]
    visible_orange += _count_cycle_color(final_visible_lengths, len(starts), ORANGE)
    if len(starts) == 2:
        has_wide_port = True
        differs_from_alt1 |= _differs_from_period1(final_visible_lengths, period=2)

    if not has_wide_port or not differs_from_alt1:
        return None
    if visible_orange == 0:
        return None

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    metrics = _behavior_metrics(inp)
    if metrics["diff_cells_from_period1"] == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
