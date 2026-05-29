# Variant of 7ed72f31
# Dimension: D3 — payload精灵的连通性和结构 (src)
# Change: payload为与红色标记4-连通的单色精灵，3-11像素，单一非红非背景颜色... → 多色payload精灵
#
# Design analysis
#   concrete_behavior:
#     - 保留原题三类红标记几何: 横线、竖线、单点中心。
#     - 变化点只放在 payload 结构: 原来是单色块，现在每个被反射的 payload 都使用 2-3 种非红非背景色。
#     - 反射规则不变: 横线做上下镜像，竖线做左右镜像，单点做中心点反射；每个像素反射后保持原颜色。
#     - 为了让多色差异肉眼可见，颜色不是噪声式随机散点，而是按 2-3 个局部色团分区，这样 output 中会出现“整块多色图案”被完整复制。
#   behavior_trace:
#     - 原始横轴案例中，若 payload 是 5 个蓝色像素，solver 只需把这 5 个蓝点镜到另一侧。
#     - 变体中，同一 payload 可能是 {(0,0):蓝, (0,1):黄, (1,1):黄, (2,1):绿, (2,2):蓝}。
#     - 正确行为是把这 5 个具体坐标逐点反射，并在目标位置保留蓝/黄/绿原色。
#     - 与原代码的关键差异: 原代码按“payload 颜色集合 + bbox”扫全图，单色时通常无害；多色后若附近碰巧出现同色噪声，旧写法会把不属于该 sprite 的像素一起反射。变体 solver 必须改成“只反射该组件内的 payload 像素”。
#   complexity_calibration:
#     - 每个 payload 固定使用 2 或 3 色，不开放更多颜色档位，避免样例太吵。
#     - payload 仍保持小尺寸 (4-11 px)，但至少 4 像素，保证 2-3 色都能真实出现。
#     - 每张图至少包含横线组和竖线组，再随机补充点反射组，保证规则重复出现而不是一次性特例。
#   implicit_assumptions_in_old_main:
#     - `payload = {v for v in vals if v != MARK}` 假设 payload 可以用颜色集合代表；多色后不够精确。
#     - `for r in range(H): for c in range(W):` 全图扫描再用 `eligible()` 过滤，依赖“同色杂点不会被卷入”；变体下改为直接遍历组件 payload 像素。
#     - isolated 分支用 `label_mask(g == v, connectivity=8)` 分颜色逐块处理，默认相邻像素若颜色不同就属于不同对象；多色 payload 下应改为对整个非红前景 8 连通组件逐像素反射。
#   placement_legality:
#     - generator 先构造完整 output，再删去镜像半边得到 input。
#     - 放置检测以 output 为准，并给所有已放置像素加 1 像素 halo，防止组间意外 4/8 连通、越界、或 isolated 红点粘连到别的对象。
#     - 横线组把 payload 宽度限制在线段长度内；竖线组把 payload 高度限制在线段长度内。这样红轴与 payload 的接触关系清晰且稳定。
#   visual_observability:
#     - 所有组默认多色；解题者在 2-4 个样例里能直接看到“同一块多色图案原样翻到另一边”，而不是只看到纯色面积复制。
#     - 点反射组允许多色，这样单点中心也遵循同一“保留内部配色”的规则，不会出现一部分组多色、一部分组单色的例外噪声。
#   visual_cleanliness:
#     - 组间至少保持 1 像素空隙，避免不同对象颜色交错成一团。
#     - 颜色数控制在每组 2-3 种，且每色形成小色团，避免花屏。
#   color_inference:
#     - 没有额外颜色映射，颜色只被搬运。人类需要推断的是“几何反射时每个像素保持自己的颜色”，不是学习某种颜色置换表。
#   new_main_assumptions_and_generator_guarantees:
#     - mixed 组件中红标记几何必须是单横线、单竖线或单红点；generator 只生成这三种。
#     - row/col payload 位于红线单侧；generator 从完整对称 output 中只保留一侧作为 input。
#     - isolated 红点应只邻接一个非红 8 连通组件；halo 放置保证这一点。
#     - 处理顺序不应影响结果；generator 保证不同组的镜像目标彼此不重叠。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS) not in sys.path:
    sys.path.append(str(ARC_UTILS))

from arc_utils import get_bg, label_mask


MARK = 2
PALETTE = [c for c in range(10) if c != MARK]
MAX_GEN_ATTEMPTS = 240


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _marker_axis(reds: list[tuple[int, int]]) -> tuple[str, int | tuple[int, int]] | None:
    rs = {int(r) for r, _c in reds}
    cs = {int(c) for _r, c in reds}
    if len(reds) == 1:
        return ("point", (int(reds[0][0]), int(reds[0][1])))
    if len(rs) == 1:
        return ("row", int(next(iter(rs))))
    if len(cs) == 1:
        return ("col", int(next(iter(cs))))
    return None


def _reflect(axis: tuple[str, int | tuple[int, int]], r: int, c: int) -> tuple[int, int]:
    kind, data = axis
    if kind == "row":
        return int(2 * int(data) - int(r)), int(c)
    if kind == "col":
        return int(r), int(2 * int(data) - int(c))
    pr, pc = data
    return int(2 * int(pr) - int(r)), int(2 * int(pc) - int(c))


def _extract_groups(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    lab4, n4 = label_mask(g != int(bg), connectivity=4)
    nonred8, _n8 = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=8)
    groups: list[dict[str, object]] = []

    for k in range(1, int(n4) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        if len(coords) == 1 and len(reds) == 1:
            pr, pc = reds[0]
            labels = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = int(pr + dr), int(pc + dc)
                    if not _inb(g.shape, rr, cc):
                        continue
                    lbl = int(nonred8[rr, cc])
                    if lbl != 0:
                        labels.add(lbl)
            for lbl in sorted(labels):
                payload = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(nonred8 == int(lbl))]
                if payload:
                    groups.append(
                        {
                            "kind": "isolated",
                            "axis": ("point", (int(pr), int(pc))),
                            "payload": payload,
                            "marker": (int(pr), int(pc)),
                        }
                    )
            continue

        axis = _marker_axis(reds)
        if axis is None:
            continue
        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if payload:
            groups.append({"kind": "mixed", "axis": axis, "payload": payload, "marker": reds})

    return bg, groups


def analyze_input(grid: np.ndarray) -> list[dict[str, object]]:
    _bg, groups = _extract_groups(np.array(grid, dtype=int))
    stats: list[dict[str, object]] = []
    for group in groups:
        payload = group["payload"]
        colors = sorted({int(v) for _r, _c, v in payload})
        stats.append(
            {
                "kind": str(group["kind"]),
                "axis": str(group["axis"][0]),
                "payload_pixels": len(payload),
                "payload_colors": len(colors),
                "colors": colors,
            }
        )
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror marker-associated sprites while preserving each payload pixel's original color.

    Variant change:
    - Payloads may now contain 2-3 colors.
    - Reflection is still purely geometric, but it operates on exact component pixels rather than
      on a color-set + bbox heuristic.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    _bg, groups = _extract_groups(g)

    for group in groups:
        axis = group["axis"]
        for r, c, v in group["payload"]:
            rr, cc = _reflect(axis, int(r), int(c))
            if _inb(g.shape, rr, cc):
                out[int(rr), int(cc)] = int(v)

    return out


def _neighbors(coord: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    r, c = coord
    if int(connectivity) == 4:
        return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((int(r + dr), int(c + dc)))
    return out


def _grow_connected_subset(
    rng: random.Random,
    allowed: set[tuple[int, int]],
    n: int,
    connectivity: int,
    seed: tuple[int, int],
) -> list[tuple[int, int]]:
    if seed not in allowed or int(n) > len(allowed):
        raise ValueError("invalid growth request")
    shape = {seed}
    while len(shape) < int(n):
        frontier = [
            cell
            for cell in allowed
            if cell not in shape and any(nb in shape for nb in _neighbors(cell, connectivity))
        ]
        if not frontier:
            raise ValueError("disconnected allowed region")
        shape.add(rng.choice(frontier))
    return sorted(shape)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _c in coords]
    cs = [int(c) for _r, c in coords]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _assign_cluster_colors(
    rng: random.Random, coords: list[tuple[int, int]], colors: list[int]
) -> dict[tuple[int, int], int]:
    seeds = rng.sample(coords, len(colors))
    mapping: dict[tuple[int, int], int] = {}
    for coord in coords:
        idx = min(
            range(len(colors)),
            key=lambda i: (abs(int(coord[0]) - int(seeds[i][0])) + abs(int(coord[1]) - int(seeds[i][1])), int(i)),
        )
        mapping[(int(coord[0]), int(coord[1]))] = int(colors[idx])
    return mapping


def _pick_group_colors(rng: random.Random, bg: int) -> list[int]:
    pool = [c for c in PALETTE if c != int(bg)]
    k = 2 if rng.random() < 0.65 else 3
    return rng.sample(pool, k)


def _build_row_group(rng: random.Random, bg: int) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    colors = _pick_group_colors(rng, bg)
    for _ in range(80):
        line_len = rng.randint(3, 6)
        width = rng.randint(2, line_len)
        height = rng.randint(2, 4)
        max_cells = min(11, height * width)
        min_cells = max(4, len(colors))
        if max_cells < min_cells:
            continue
        n = rng.randint(min_cells, max_cells)
        seed = (height - 1, rng.randrange(width))
        allowed = {(r, c) for r in range(height) for c in range(width)}
        shape = _grow_connected_subset(rng, allowed, n, 4, seed)
        r0, c0, r1, c1 = _bbox(shape)
        if r1 == r0 or c1 == c0:
            continue
        xoff = rng.randint(0, line_len - width)
        cmap = _assign_cluster_colors(rng, shape, colors)
        top = {(int(r), int(c + xoff)): int(cmap[(r, c)]) for r, c in shape}
        bottom = {(int(2 * height - r), int(c + xoff)): int(cmap[(r, c)]) for r, c in shape}
        red = {(int(height), int(c)): MARK for c in range(line_len)}
        inp = dict(red)
        inp.update(top if rng.random() < 0.5 else bottom)
        out = dict(red)
        out.update(top)
        out.update(bottom)
        return inp, out
    raise RuntimeError("row group generation failed")


def _build_col_group(rng: random.Random, bg: int) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    colors = _pick_group_colors(rng, bg)
    for _ in range(80):
        line_len = rng.randint(3, 6)
        height = rng.randint(2, line_len)
        width = rng.randint(2, 4)
        max_cells = min(11, height * width)
        min_cells = max(4, len(colors))
        if max_cells < min_cells:
            continue
        n = rng.randint(min_cells, max_cells)
        seed = (rng.randrange(height), width - 1)
        allowed = {(r, c) for r in range(height) for c in range(width)}
        shape = _grow_connected_subset(rng, allowed, n, 4, seed)
        r0, c0, r1, c1 = _bbox(shape)
        if r1 == r0 or c1 == c0:
            continue
        yoff = rng.randint(0, line_len - height)
        cmap = _assign_cluster_colors(rng, shape, colors)
        left = {(int(r + yoff), int(c)): int(cmap[(r, c)]) for r, c in shape}
        right = {(int(r + yoff), int(2 * width - c)): int(cmap[(r, c)]) for r, c in shape}
        red = {(int(r), int(width)): MARK for r in range(line_len)}
        inp = dict(red)
        inp.update(left if rng.random() < 0.5 else right)
        out = dict(red)
        out.update(left)
        out.update(right)
        return inp, out
    raise RuntimeError("col group generation failed")


def _rot90(dr: int, dc: int) -> tuple[int, int]:
    return int(-dc), int(dr)


def _rotate_rel(coords: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = [(int(r), int(c)) for r, c in coords]
    for _ in range(int(turns) % 4):
        out = [_rot90(r, c) for r, c in out]
    return out


def _build_point_group(
    rng: random.Random, bg: int, *, isolated: bool
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    colors = _pick_group_colors(rng, bg)
    for _ in range(100):
        if isolated:
            h = rng.randint(2, 4)
            w = rng.randint(2, 4)
            max_cells = min(8, h * w)
            allowed = {(int(dr), int(dc)) for dr in range(-h, 0) for dc in range(-w, 0)}
            seed = (-1, -1)
            connectivity = 8
        else:
            h = rng.randint(2, 4)
            w = rng.randint(2, 4)
            max_cells = min(11, (h + 1) * (w + 1) - 1)
            allowed = {
                (int(dr), int(dc))
                for dr in range(-h, 1)
                for dc in range(-w, 1)
                if not (dr == 0 and dc == 0)
            }
            seed = rng.choice([(-1, 0), (0, -1)])
            connectivity = 4
        min_cells = max(4, len(colors))
        if max_cells < min_cells:
            continue
        n = rng.randint(min_cells, max_cells)
        rel = _grow_connected_subset(rng, allowed, n, connectivity, seed)
        rows = {r for r, _c in rel}
        cols = {c for _r, c in rel}
        if len(rows) < 2 or len(cols) < 2:
            continue
        rel = _rotate_rel(rel, rng.randrange(4))
        cmap = _assign_cluster_colors(rng, rel, colors)
        mirror = [(-int(r), -int(c)) for r, c in rel]
        all_coords = rel + mirror + [(0, 0)]
        min_r = min(int(r) for r, _c in all_coords)
        min_c = min(int(c) for _r, c in all_coords)
        shift_r = -int(min_r)
        shift_c = -int(min_c)
        orig = {(int(r + shift_r), int(c + shift_c)): int(cmap[(r, c)]) for r, c in rel}
        twin = {(int(-r + shift_r), int(-c + shift_c)): int(cmap[(r, c)]) for r, c in rel}
        center = {(int(shift_r), int(shift_c)): MARK}
        inp = dict(center)
        inp.update(orig if rng.random() < 0.5 else twin)
        out = dict(center)
        out.update(orig)
        out.update(twin)
        return inp, out
    raise RuntimeError("point group generation failed")


def _group_builder(name: str):
    if name == "row":
        return _build_row_group
    if name == "col":
        return _build_col_group
    if name == "point":
        return lambda rng, bg: _build_point_group(rng, bg, isolated=False)
    if name == "isolated":
        return lambda rng, bg: _build_point_group(rng, bg, isolated=True)
    raise ValueError(f"unknown group type: {name}")


def _place_group(
    rng: random.Random,
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    forbidden: set[tuple[int, int]],
    local_in: dict[tuple[int, int], int],
    local_out: dict[tuple[int, int], int],
) -> bool:
    h, w = grid_in.shape
    max_r = max(int(r) for r, _c in local_out)
    max_c = max(int(c) for _r, c in local_out)
    for _ in range(200):
        top = rng.randint(0, int(h - max_r - 1))
        left = rng.randint(0, int(w - max_c - 1))
        placed_out = [(int(top + r), int(left + c), int(v)) for (r, c), v in local_out.items()]
        if any((int(r), int(c)) in forbidden for r, c, _v in placed_out):
            continue
        placed_in = [(int(top + r), int(left + c), int(v)) for (r, c), v in local_in.items()]
        for r, c, v in placed_out:
            grid_out[r, c] = int(v)
        for r, c, v in placed_in:
            grid_in[r, c] = int(v)
        for r, c, _v in placed_out:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = int(r + dr), int(c + dc)
                    if _inb(grid_out.shape, rr, cc):
                        forbidden.add((int(rr), int(cc)))
        return True
    return False


def _generate_pair(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    bg = rng.choice(PALETTE)
    grid_in = np.full((h, w), int(bg), dtype=int)
    grid_out = np.full((h, w), int(bg), dtype=int)
    forbidden: set[tuple[int, int]] = set()

    group_count = rng.randint(4, 6)
    group_types = ["row", "col"]
    group_types.extend(rng.choice(["row", "col", "point", "isolated"]) for _ in range(group_count - 2))
    rng.shuffle(group_types)

    for name in group_types:
        builder = _group_builder(name)
        placed = False
        for _ in range(80):
            local_in, local_out = builder(rng, bg)
            if _place_group(rng, grid_in, grid_out, forbidden, local_in, local_out):
                placed = True
                break
        if not placed:
            raise RuntimeError(f"unable to place {name} group")

    return grid_in, grid_out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        try:
            inp, out = _generate_pair(rng)
        except RuntimeError:
            continue
        if not np.array_equal(main(inp), out):
            continue
        stats = analyze_input(inp)
        if len(stats) < 4:
            continue
        if any(int(s["payload_colors"]) < 2 for s in stats):
            continue
        return inp
    raise RuntimeError("failed to generate a valid multicolor 7ed72f31 variant input")
