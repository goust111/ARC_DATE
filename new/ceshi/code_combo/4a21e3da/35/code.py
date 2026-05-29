# Combo variant of 4a21e3da (index 35)
# combo_spec: D8/0+D13/0+D16/1
# Name: 三控制像素+连通归属+消失到中心

from __future__ import annotations

"""
design_analysis:
  concretization:
    input:
      - 仍然只有 bg/fg/ctrl 三色, 但固定为 3 个边界 ctrl。
      - 采用 2+1 布局: top+bottom 两个竖切列, 再加 left 或 right 一个横切行。
      - 三条切割线把精灵分成 6 个几何区: TL/TM/TR 和 BL/BM/BR。
      - 输入精灵不再要求整体 4 连通; generator 刻意做成整体 8 连通但非 4 连通:
        主体通过 split-line fg 串起来, 另外再挂 1-2 个只靠对角接触的孤岛。
    output:
      - ctrl 射线画法沿用原题: 从边界向内, 只在 bg 上涂 ctrl, 直到该行/列最远的 fg。
      - split-line fg 原地保留。
      - 删去 split-line fg 后, 对剩余 fg 做 4 连通分量。
      - 若分量有任一像素与 split-line fg 4 邻接, 则该分量保留, 并按所在几何区滑向角落。
      - 若分量不与 split-line fg 4 邻接, 则不再消失, 而是滑到中心区:
        * 左/右区去对应交叉点的内角;
        * 中条区去两条竖切线之间、split row 上下的中心条带。
    trace_example: |
      设 ctrl 在 top@c0, bottom@c1, left@r, 且 c0<c1。
      原题思路会把非 split-line 像素按矩形区整块处理; 在三控制扩展里, 同一区里通常整块一起去角落或消失。
      本融合里先删掉 split-line fg:
        1) TL 主块仍与 (2,c0)/(r,1..7) 上的 split fg 正交接壤, 所以保留并去左上角。
        2) TR 的一个 3 像素小岛只与 TR 主块对角接触, 不与任何 split fg 正交接壤。
           因此它不会跟 TR 主块一起去右上角, 而是单独滑到中心右上内角。
      差异点是: 同一几何区内的 fg 可以因 4 连通归属不同而落到不同目标区。
  complexity_calibration:
    - 仍只做平移, 不做旋转、镜像或颜色映射。
    - 几何类别只有 6 个固定区, 中心目标只有 3 类稳定位置:
      左内角 / 中心条带 / 右内角。自由度有限, 但视觉差异足够明显。
    - generator 强制 6 个区都有主块, 且至少 1 个孤岛去中心, 因而三控制、连通归属、中心保留三件事都会同时可见。
  implicit_assumptions_audit:
    - 原 main() 假设至多一条竖线和一条横线。这里失效, 必须改成 `v_splits` 列表 + `h_splits` 列表。
    - 原 main() 假设几何区域就是行为单位。这里失效, 必须先去掉 split-line fg 再做 4 连通分量。
    - 若输入仍强制整体 4 连通, 那么被 split-line fg 切开的每个分量都会与 split-line fg 接壤,
      D13/0 会退化。故 generator 必须改成整体 8 连通但非 4 连通。
    - 多个分量可能被送往同一角落(例如 TM 和 TL 在 singleton edge 为 left 时都去左上角),
      所以 solver 不能逐分量各自贴角, 必须按目标桶并集后整体平移, 否则会重叠。
  placement_legality:
    - 本生成器固定本体 bbox 为 8x9, 且整体 margin >= 4。
    - 6 个主块都放在各自几何区内, 不越 split line。
    - 孤岛只通过对角接触挂在主块上, 不与 split-line fg 4 邻接。
    - 中心目标按区分类投到内角或中心条带, 与 split-line fg 至少隔 1 格, 不会覆盖 ctrl 射线或静止 fg。
  visual_observability:
    - 每个样例都固定出现 3 个 ctrl 与 6 区切分。
    - 输出里既有四角/两角上的保留块, 又有中心附近的 1-2 个小岛。
    - 同一区里“主块去角落, 孤岛去中心”的对比, 直接暴露 D13/0 的拓扑归属逻辑。
  cleanliness:
    - 仍只有三色, 没有额外标记。
    - 孤岛数量限制在 1-2 个, 且都很小, 不会把中心区堆得太乱。
    - 中心区槽位按左/中/右分开, 输出层次清楚。
  color_inference:
    - 颜色不参与隐藏排序; 只有可见的边界位置、split 几何和 4 连通关系参与规则。
  main_new_assumptions:
    - solver 假设恰有 3 个 ctrl, 且是 top+bottom+left/right 这种 2+1 布局。
    - solver 假设每个删线后的 4 连通分量只落在单一几何区; generator 通过满行/满列 split 与区内模板保证这一点。
    - solver 假设去中心的分量按“所在区对应的中心槽位”放置即可; generator 保证去中心分量很小且每区至多一组。
"""

import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, get_bg, is_contiguous


Coord = tuple[int, int]
Region = tuple[str, str]

BG = 1
CTRL = 2
FG = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

LOCAL_H = 8
LOCAL_W = 9
BASE_R_SPLIT = 4
BASE_V_SPLITS = (2, 6)
BASE_SPLIT_FG = frozenset(
    {
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (2, 2),
        (3, 2),
        (5, 2),
        (2, 6),
        (3, 6),
        (5, 6),
    }
)
BASE_MAINS: dict[Region, frozenset[Coord]] = {
    ("T", "L"): frozenset({(2, 1), (3, 0), (3, 1)}),
    ("T", "M"): frozenset({(2, 3), (2, 4), (3, 4)}),
    ("T", "R"): frozenset({(2, 7), (3, 7), (3, 8)}),
    ("B", "L"): frozenset({(5, 1), (6, 1), (7, 0), (7, 1)}),
    ("B", "M"): frozenset({(5, 4), (6, 3), (6, 4), (7, 4)}),
    ("B", "R"): frozenset({(5, 7), (6, 7), (6, 8), (7, 8)}),
}
BASE_ROGUES: dict[Region, frozenset[Coord]] = {
    ("T", "L"): frozenset({(0, 0), (0, 1), (1, 0)}),
    ("T", "R"): frozenset({(0, 7), (0, 8), (1, 8)}),
}


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _parse_roles(grid: np.ndarray) -> tuple[int, int, int, list[Coord]]:
    bg = int(get_bg(grid))
    nonbg = colors_present(grid, exclude={bg})
    border_cols = set(map(int, grid[0, :])) | set(map(int, grid[-1, :])) | set(map(int, grid[:, 0])) | set(map(int, grid[:, -1]))
    border_cols.discard(int(bg))
    if not border_cols:
        raise ValueError("missing border control color")
    ctrl = int(sorted(border_cols)[0])
    fg_candidates = [int(c) for c in nonbg if int(c) != int(ctrl)]
    if len(fg_candidates) != 1:
        raise ValueError("expected exactly one foreground color")
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(ctrl))]
    return int(bg), int(ctrl), int(fg_candidates[0]), ctrl_pts


def _parse_layout(ctrl_pts: list[Coord], H: int, W: int) -> dict[str, object]:
    if len(ctrl_pts) != 3:
        raise ValueError("expected exactly 3 control pixels")

    v_splits: list[int] = []
    h_splits: list[int] = []
    edges: set[str] = set()
    for r, c in ctrl_pts:
        if r == 0:
            edges.add("top")
            v_splits.append(int(c))
        elif r == H - 1:
            edges.add("bottom")
            v_splits.append(int(c))
        elif c == 0:
            edges.add("left")
            h_splits.append(int(r))
        elif c == W - 1:
            edges.add("right")
            h_splits.append(int(r))
        else:
            raise ValueError("control pixel must lie on border")

    v_splits = sorted(v_splits)
    h_splits = sorted(h_splits)
    if len(v_splits) == 2 and len(h_splits) == 1 and {"top", "bottom"} <= edges and ("left" in edges or "right" in edges):
        return {
            "mode": "vertical_dup",
            "v_splits": v_splits,
            "h_splits": h_splits,
            "edges": set(edges),
            "singleton_edge": "left" if "left" in edges else "right",
        }
    if len(v_splits) == 1 and len(h_splits) == 2 and {"left", "right"} <= edges and ("top" in edges or "bottom" in edges):
        return {
            "mode": "horizontal_dup",
            "v_splits": v_splits,
            "h_splits": h_splits,
            "edges": set(edges),
            "singleton_edge": "top" if "top" in edges else "bottom",
        }
    raise ValueError("controls must form a 2+1 three-control layout")


def _band(index: int, splits: list[int], low: str, high: str) -> str:
    if not splits:
        return "A"
    if len(splits) == 1:
        return low if int(index) < int(splits[0]) else high
    s0, s1 = [int(x) for x in splits]
    if int(index) < s0:
        return low
    if int(index) > s1:
        return high
    return "M"


def _region_label(r: int, c: int, layout: dict[str, object]) -> Region:
    row_band = _band(int(r), list(layout["h_splits"]), "T", "B")
    col_band = _band(int(c), list(layout["v_splits"]), "L", "R")
    return str(row_band), str(col_band)


def _split_fg(grid: np.ndarray, fg: int, layout: dict[str, object]) -> set[Coord]:
    v_splits = set(int(x) for x in layout["v_splits"])
    h_splits = set(int(x) for x in layout["h_splits"])
    pts: set[Coord] = set()
    for r, c in np.argwhere(grid == int(fg)):
        r = int(r)
        c = int(c)
        if r in h_splits or c in v_splits:
            pts.add((r, c))
    return pts


def _components(cells: set[Coord]) -> list[set[Coord]]:
    remaining = set((int(r), int(c)) for r, c in cells)
    comps: list[set[Coord]] = []
    while remaining:
        start = next(iter(remaining))
        q = deque([start])
        remaining.remove(start)
        comp = {start}
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _split_adjacent(comp: set[Coord], split_fg: set[Coord]) -> bool:
    for r, c in comp:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in split_fg:
                return True
    return False


def _pick_row_dest(row_band: str, layout: dict[str, object]) -> str:
    edges = set(layout["edges"])
    h_splits = list(layout["h_splits"])
    if row_band in ("T", "B"):
        return str(row_band)
    if h_splits:
        return "T" if "top" in edges else "B"
    return "T"


def _pick_col_dest(col_band: str, layout: dict[str, object]) -> str:
    edges = set(layout["edges"])
    v_splits = list(layout["v_splits"])
    if col_band in ("L", "R"):
        return str(col_band)
    if v_splits:
        return "L" if "left" in edges else "R"
    return "L"


def _translate_to_outer_corner(pts: set[Coord], row_dest: str, col_dest: str, H: int, W: int) -> set[Coord]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    dr = -rmin if row_dest == "T" else int(H - 1 - rmax)
    dc = -cmin if col_dest == "L" else int(W - 1 - cmax)
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _translate_to_center_zone(pts: set[Coord], row_band: str, col_band: str, layout: dict[str, object]) -> set[Coord]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    ph = int(rmax - rmin + 1)
    pw = int(cmax - cmin + 1)

    if str(layout["mode"]) == "vertical_dup":
        c0, c1 = [int(x) for x in layout["v_splits"]]
        r_split = int(layout["h_splits"][0])
        if col_band == "L":
            dr = (int(r_split - 1) - rmax) if row_band == "T" else (int(r_split + 1) - rmin)
            dc = int(c0 - 1 - cmax)
        elif col_band == "R":
            dr = (int(r_split - 1) - rmax) if row_band == "T" else (int(r_split + 1) - rmin)
            dc = int(c1 + 1 - cmin)
        else:
            inner_c0 = int(c0 + 1)
            inner_c1 = int(c1 - 1)
            dc = int(inner_c0 + (inner_c1 - inner_c0 + 1 - pw) // 2 - cmin)
            dr = (int(r_split - 1) - rmax) if row_band == "T" else (int(r_split + 1) - rmin)
        return {(int(r + dr), int(c + dc)) for r, c in pts}

    r0, r1 = [int(x) for x in layout["h_splits"]]
    c_split = int(layout["v_splits"][0])
    if row_band == "T":
        dr = int(r0 - 1 - rmax)
        dc = int(c_split - 1 - cmax) if col_band == "L" else int(c_split + 1 - cmin)
    elif row_band == "B":
        dr = int(r1 + 1 - rmin)
        dc = int(c_split - 1 - cmax) if col_band == "L" else int(c_split + 1 - cmin)
    else:
        inner_r0 = int(r0 + 1)
        inner_r1 = int(r1 - 1)
        dr = int(inner_r0 + (inner_r1 - inner_r0 + 1 - ph) // 2 - rmin)
        dc = int(c_split - 1 - cmax) if col_band == "L" else int(c_split + 1 - cmin)
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _draw_rays(grid: np.ndarray, out: np.ndarray, ctrl_pts: list[Coord], bg: int, ctrl: int, fg: int) -> None:
    H, W = grid.shape
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(grid[:, c] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if int(grid[rr, c]) == int(bg):
                        out[rr, c] = int(ctrl)
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(grid[:, c] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if int(grid[rr, c]) == int(bg):
                        out[rr, c] = int(ctrl)
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(grid[r, :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if int(grid[r, cc]) == int(bg):
                        out[r, cc] = int(ctrl)
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(grid[r, :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if int(grid[r, cc]) == int(bg):
                        out[r, cc] = int(ctrl)
    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)


def _solve_core(grid: np.ndarray, *, center_for_nonowned: bool, component_ownership: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, ctrl, fg, ctrl_pts = _parse_roles(g)
    layout = _parse_layout(ctrl_pts, H, W)
    split_fg = _split_fg(g, fg, layout)

    out = np.full((H, W), int(bg), dtype=int)
    _draw_rays(g, out, ctrl_pts, bg, ctrl, fg)

    for r, c in split_fg:
        out[int(r), int(c)] = int(fg)

    if component_ownership:
        moving = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg)) if (int(r), int(c)) not in split_fg}
        pieces = _components(moving)
    else:
        by_region: dict[Region, set[Coord]] = defaultdict(set)
        for r, c in np.argwhere(g == int(fg)):
            pt = (int(r), int(c))
            if pt in split_fg:
                continue
            by_region[_region_label(int(r), int(c), layout)].add(pt)
        pieces = list(by_region.values())

    corner_buckets: dict[tuple[str, str], set[Coord]] = defaultdict(set)
    center_buckets: dict[Region, set[Coord]] = defaultdict(set)

    for comp in pieces:
        labels = {_region_label(r, c, layout) for r, c in comp}
        if len(labels) != 1:
            raise ValueError("component crosses region boundary")
        label = next(iter(labels))
        row_band, col_band = label
        owned = _split_adjacent(comp, split_fg) if component_ownership else True
        if owned:
            dest = (_pick_row_dest(row_band, layout), _pick_col_dest(col_band, layout))
            corner_buckets[dest] |= set(comp)
        elif center_for_nonowned:
            center_buckets[label] |= set(comp)

    for (row_dest, col_dest), pts in corner_buckets.items():
        for rr, cc in _translate_to_outer_corner(pts, row_dest, col_dest, H, W):
            out[int(rr), int(cc)] = int(fg)

    for (row_band, col_band), pts in center_buckets.items():
        for rr, cc in _translate_to_center_zone(pts, row_band, col_band, layout):
            out[int(rr), int(cc)] = int(fg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(np.array(grid, dtype=int), center_for_nonowned=True, component_ownership=True)


def _vanish_baseline(grid: np.ndarray) -> np.ndarray:
    return _solve_core(np.array(grid, dtype=int), center_for_nonowned=False, component_ownership=True)


def _region_baseline(grid: np.ndarray) -> np.ndarray:
    return _solve_core(np.array(grid, dtype=int), center_for_nonowned=False, component_ownership=False)


def _apply_hflip(points: set[Coord] | frozenset[Coord], width: int) -> set[Coord]:
    return {(int(r), int(width - 1 - c)) for r, c in points}


def _build_local_pattern() -> tuple[set[Coord], int, list[int]]:
    use_flip = bool(random.getrandbits(1))
    split_fg = set(BASE_SPLIT_FG)
    mains = {reg: set(pts) for reg, pts in BASE_MAINS.items()}
    rogue_regions = random.sample(list(BASE_ROGUES.keys()), k=random.choice([1, 2]))
    rogues = {reg: set(BASE_ROGUES[reg]) for reg in rogue_regions}

    all_pts = set(split_fg)
    for pts in mains.values():
        all_pts |= set(pts)
    for pts in rogues.values():
        all_pts |= set(pts)

    if use_flip:
        all_pts = _apply_hflip(all_pts, LOCAL_W)
        v_splits = sorted([LOCAL_W - 1 - BASE_V_SPLITS[1], LOCAL_W - 1 - BASE_V_SPLITS[0]])
    else:
        v_splits = [int(BASE_V_SPLITS[0]), int(BASE_V_SPLITS[1])]

    return set(all_pts), int(BASE_R_SPLIT), list(v_splits)


def _make_input() -> np.ndarray | None:
    H = W = random.randint(19, 20)
    sprite_r0 = random.randint(4, H - 5 - LOCAL_H)
    sprite_c0 = random.randint(4, W - 5 - LOCAL_W)
    singleton_edge = random.choice(["left", "right"])

    local_fg, local_r_split, local_v_splits = _build_local_pattern()
    global_fg = {(int(sprite_r0 + r), int(sprite_c0 + c)) for r, c in local_fg}
    r_split = int(sprite_r0 + local_r_split)
    v_splits = [int(sprite_c0 + c) for c in local_v_splits]
    ctrl_pts = [(0, int(v_splits[0])), (int(H - 1), int(v_splits[1])), (int(r_split), 0 if singleton_edge == "left" else int(W - 1))]

    if global_fg & set(ctrl_pts):
        return None

    fg_arr = np.array(list(global_fg), dtype=int)
    bbox_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    bbox_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if bbox_h != LOCAL_H or bbox_w != LOCAL_W:
        return None
    density = len(global_fg) / float(bbox_h * bbox_w)
    if not (0.39 <= density <= 0.60):
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in global_fg:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask, connectivity=8):
        return None
    if is_contiguous(mask, connectivity=4):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for r, c in global_fg:
        inp[int(r), int(c)] = FG
    for r, c in ctrl_pts:
        inp[int(r), int(c)] = CTRL
    return inp


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    _bg, _ctrl, fg, ctrl_pts = _parse_roles(g)
    layout = _parse_layout(ctrl_pts, g.shape[0], g.shape[1])
    split_fg = _split_fg(g, fg, layout)
    moving = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg)) if (int(r), int(c)) not in split_fg}
    comps = _components(moving)

    regions_present = len({_region_label(r, c, layout) for r, c in moving})
    nonowned = 0
    center_pixels = 0
    for comp in comps:
        if not _split_adjacent(comp, split_fg):
            nonowned += 1
            center_pixels += len(comp)

    out = main(g)
    vanish = _vanish_baseline(g)
    region = _region_baseline(g)
    return {
        "ctrl_count": int(len(ctrl_pts)),
        "regions_present": int(regions_present),
        "nonowned_components": int(nonowned),
        "center_pixels": int(center_pixels),
        "diff_vs_vanish": int(np.count_nonzero(out != vanish)),
        "diff_vs_region": int(np.count_nonzero(out != region)),
    }


def _gen_attempt() -> np.ndarray | None:
    inp = _make_input()
    if inp is None:
        return None

    try:
        _ = main(inp)
    except Exception:
        return None

    metrics = _behavior_metrics(inp)
    if metrics["ctrl_count"] != 3:
        return None
    if metrics["regions_present"] != 6:
        return None
    if metrics["nonowned_components"] < 1:
        return None
    if metrics["center_pixels"] < 3:
        return None
    if metrics["diff_vs_vanish"] < 3:
        return None
    if metrics["diff_vs_region"] < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo input after 400 attempts")
