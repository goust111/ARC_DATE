# Combo variant of 7ed72f31 (index 27)
# combo_spec: D3/1+D7/0
# Name: 连通性不匹配检测
#
# Design analysis
# ===============
# concrete_behavior:
#   - 保留原题的三类几何直觉: 横向红线、纵向红线、单红点。
#   - 融合点 1: 红线组的 payload 按 8 连通理解。也就是说，和红线 4 邻接的核心像素
#     可以再通过对角线把更多同色像素串起来；这些对角链上的像素也属于同一个被镜像的 payload。
#   - 融合点 2: 单红点组不再按 8 连通“吃掉整块精灵”，而是只从红点的 4 邻域
#     (上/下/左/右) 触发，再沿 4 连通同色组件扩张。于是对角接上的枝杈不会被反射，
#     输出里只出现“部分点反射”。
# behavior_trace:
#   - 例 1, 横线组:
#       红线下一格有一个绿色像素 A，A 的左下角再斜接一个绿色像素 B。
#       原 solver 的 4 连通混合分块只会抓到 A；B 会被当成别的块，通常不镜像。
#       本变体把整块 payload 按 8 连通看待，因此 A 和 B 都会被镜到红线另一侧。
#   - 例 2, 单红点组:
#       红点上方一格有蓝色像素 A，A 的左上角再斜接蓝色像素 B。
#       8 连通视角下 A+B 是一整块 sprite；但本变体的单红点检测先看 4 邻域，只能碰到 A，
#       再按 4 连通扩张时也到不了 B，因此只会把 A 点反射到红点下方，B 留在原位。
#   - 差异总结:
#       红线组是“8 连通整块镜像”，单红点组是“4 连通局部镜像”。
# complexity_calibration:
#   - 每张图强制同时出现至少一个红线组和至少一个单红点组。
#   - 红线组的 payload 强制满足“8 连通但不是红线可达的 4 连通整块”，否则 D3/1 不可见。
#   - 单红点组的 payload 强制满足“4 连通可检测子集 < 完整 8 连通 sprite”，否则 D7/0 不可见。
#   - 颜色仍然一组一种，不再叠加额外颜色映射，保证观察者只需要推断连通性差异。
# implicit_assumption_review_of_old_main:
#   - `label_mask(g != bg, connectivity=4)` 默认把“红 + payload”整体按 4 连通分块。
#     该前提在 D3/1 下失效: 对角接上的 payload 像素会被漏掉。
#   - 原单红点分支先看 8 邻域，再按颜色 8 连通整块反射。
#     该前提在 D7/0 下失效: 变体要故意把检测收窄为 4 邻域 + 4 连通扩张，制造部分反射。
#   - 原 solver 的“单红点若与 payload 4 邻接则属于混合组件”在本融合里不再适合作为主分流条件，
#     因为我们希望单红点统一走局部检测逻辑，显式体现与红线组不同的连通性规则。
# placement_legality:
#   - generator 先构造每个局部模板的 input / output，再把它们整体放进大网格。
#   - 放置时按 input∪output 的所有非背景像素做 1 像素 Chebyshev halo，防止不同组在 8 连通意义下粘连。
#   - 所有镜像目标都落在背景上，不覆盖别的组，也不越界。
# visual_observability:
#   - 红线组输出会补出完整斜链。
#   - 单红点组输出只补出靠近红点的 4 连通核心，斜接尾巴仍只在原侧出现。
#   - 2-3 个样例里反复出现这两种对比，人眼可以直接归纳“同样是红标记，线组和点组的连通性判定不同”。
# cleanliness_and_inference:
#   - 不生成自由散点或同色干扰组，避免把注意力从连通性规则转移到颜色噪声。
#   - 新 solver 的核心新假设是“单红点附近的有效 payload 必须能在 4 邻域里触发到至少一个像素”；
#     generator 用构造保证这一点，并额外保证该 4 连通子集严格小于整块 8 连通 sprite。

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
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GEN_ATTEMPTS = 320


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = center
    return int(2 * int(pr) - int(r)), int(2 * int(pc) - int(c))


def _red_axis(reds: list[tuple[int, int]]) -> tuple[str, int | tuple[int, int]] | None:
    if not reds:
        return None
    rows = {int(r) for r, _c in reds}
    cols = {int(c) for _r, c in reds}
    if len(reds) == 1:
        return ("point", (int(reds[0][0]), int(reds[0][1])))
    if len(rows) == 1:
        return ("row", int(next(iter(rows))))
    if len(cols) == 1:
        return ("col", int(next(iter(cols))))
    return None


def _component_cells(lab: np.ndarray, labels: set[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lbl in sorted(int(x) for x in labels if int(x) != 0):
        for r, c in np.argwhere(lab == int(lbl)):
            out.append((int(r), int(c)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror with a deliberate connectivity mismatch.

    - Red lines use 8-connected payload grouping, so diagonal appendages are mirrored.
    - Single red points only detect payload through 4-neighborhood seeds and 4-connected expansion,
      so diagonal-only appendages are ignored and only part of the sprite is point-reflected.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    h, w = g.shape

    red_lab, red_n = label_mask(g == int(MARK), connectivity=4)
    fg8_lab, _ = label_mask(g != int(bg), connectivity=8)

    for k in range(1, int(red_n) + 1):
        reds = [(int(r), int(c)) for r, c in np.argwhere(red_lab == int(k))]
        axis = _red_axis(reds)
        if axis is None:
            continue

        if axis[0] in {"row", "col"}:
            labels = {int(fg8_lab[int(r), int(c)]) for r, c in reds if int(fg8_lab[int(r), int(c)]) != 0}
            for r, c in _component_cells(fg8_lab, labels):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if axis[0] == "row":
                    rr, cc = _refl_row(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = _refl_col(int(axis[1]), int(r), int(c))
                if _inb((h, w), rr, cc):
                    out[int(rr), int(cc)] = int(v)
            continue

        pr, pc = map(int, axis[1])
        seeds_by_color: dict[int, list[tuple[int, int]]] = {}
        for dr, dc in N4:
            rr, cc = int(pr + dr), int(pc + dc)
            if not _inb((h, w), rr, cc):
                continue
            v = int(g[int(rr), int(cc)])
            if v == int(bg) or v == int(MARK):
                continue
            seeds_by_color.setdefault(int(v), []).append((int(rr), int(cc)))

        for color, seeds in seeds_by_color.items():
            same4_lab, _ = label_mask(g == int(color), connectivity=4)
            labels = {int(same4_lab[int(r), int(c)]) for r, c in seeds if int(same4_lab[int(r), int(c)]) != 0}
            for r, c in _component_cells(same4_lab, labels):
                rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                if _inb((h, w), rr, cc):
                    out[int(rr), int(cc)] = int(color)

    return out


def _baseline_original_main(grid: np.ndarray) -> np.ndarray:
    """
    Original 7ed72f31 behavior, kept here only as a generator-side degeneration filter.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        rs = [int(r) for r, _c in coords]
        cs = [int(c) for _r, c in coords]
        comps.append({"coords": coords, "vals": vals, "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs)))})

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(vals):
            continue
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue
        rows = {int(r) for r, _c in reds}
        cols = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rows) == 1:
            axis = ("row", int(next(iter(rows))))
        elif len(cols) == 1:
            axis = ("col", int(next(iter(cols))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload_colors = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload_colors:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload_colors and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload_colors and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(h - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(w - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload_colors and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(h)):
            for c in range(int(w)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue
                if axis[0] == "row":
                    rr, cc = _refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = _refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = _refl_point(axis[1], int(r), int(c))
                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[int(rr), int(cc)])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        for color in neigh_colors:
            same8_lab, _ = label_mask(g == int(color), connectivity=8)
            labels = {
                int(same8_lab[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(color) and int(same8_lab[int(rr), int(cc)]) != 0
            }
            for lbl in labels:
                for r, c in np.argwhere(same8_lab == int(lbl)):
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(int(rr), int(cc)):
                        out[int(rr), int(cc)] = int(color)

    return out


def _neighbor_list(coord: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    r, c = coord
    if int(connectivity) == 4:
        return [(int(r + dr), int(c + dc)) for dr, dc in N4]
    return [(int(r + dr), int(c + dc)) for dr, dc in N8]


def _grow_connected_subset(
    rng: random.Random,
    allowed: set[tuple[int, int]],
    size: int,
    connectivity: int,
    seed: tuple[int, int],
) -> list[tuple[int, int]]:
    if seed not in allowed:
        raise ValueError("seed outside allowed cells")
    if int(size) < 1 or int(size) > len(allowed):
        raise ValueError("invalid shape size")
    chosen = {seed}
    while len(chosen) < int(size):
        frontier = [
            cell
            for cell in allowed
            if cell not in chosen and any(nb in chosen for nb in _neighbor_list(cell, connectivity))
        ]
        if not frontier:
            raise ValueError("growth got stuck")
        chosen.add(rng.choice(frontier))
    return sorted((int(r), int(c)) for r, c in chosen)


def _seeded_four_subset(coords: list[tuple[int, int]], seeds: list[tuple[int, int]]) -> set[tuple[int, int]]:
    if not coords or not seeds:
        return set()
    h = max(int(r) for r, _c in coords) + 1
    w = max(int(c) for _r, c in coords) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    lab4, _ = label_mask(mask, connectivity=4)
    labels = {int(lab4[int(r), int(c)]) for r, c in seeds if int(lab4[int(r), int(c)]) != 0}
    out: set[tuple[int, int]] = set()
    for lbl in labels:
        for r, c in np.argwhere(lab4 == int(lbl)):
            out.add((int(r), int(c)))
    return out


def _shape_is_diagonal_from_seed(coords: list[tuple[int, int]], seeds: list[tuple[int, int]]) -> bool:
    seeded = _seeded_four_subset(coords, seeds)
    return 0 < len(seeded) < len(coords)


def _shift_cells(cells: dict[tuple[int, int], int], dr: int, dc: int) -> dict[tuple[int, int], int]:
    return {(int(r + dr), int(c + dc)): int(v) for (r, c), v in cells.items()}


def _bounds(cells: dict[tuple[int, int], int]) -> tuple[int, int]:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    return int(max(rs) + 1), int(max(cs) + 1)


def _normalize_cells(cells: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    r0 = min(int(r) for r, _c in cells)
    c0 = min(int(c) for _r, c in cells)
    return {(int(r - r0), int(c - c0)): int(v) for (r, c), v in cells.items()}


def _nonbg_union(inp: dict[tuple[int, int], int], out: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    return set(inp) | set(out)


def _build_row_group(color: int, rng: random.Random) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[str, int]]:
    for _ in range(120):
        line_len = rng.randint(4, 6)
        depth = rng.randint(2, 4)
        span = rng.randint(2, line_len)
        max_cells = min(10, int(depth * span))
        size = rng.randint(3, max_cells)
        side = rng.choice(["above", "below"])
        seed = (int(depth - 1), rng.randrange(span)) if side == "above" else (0, rng.randrange(span))
        allowed = {(int(r), int(c)) for r in range(depth) for c in range(span)}
        shape = _grow_connected_subset(rng, allowed, size, connectivity=8, seed=seed)
        if not _shape_is_diagonal_from_seed(shape, [seed]):
            continue

        xoff = rng.randint(0, line_len - span)
        axis_r = int(depth)
        red = {(int(axis_r), int(c)): MARK for c in range(line_len)}
        if side == "above":
            src = {(int(r), int(c + xoff)): int(color) for r, c in shape}
        else:
            src = {(int(r + depth + 1), int(c + xoff)): int(color) for r, c in shape}

        out = dict(red)
        out.update(src)
        for r, c in src:
            rr, cc = _refl_row(int(axis_r), int(r), int(c))
            out[(int(rr), int(cc))] = int(color)

        inp = dict(red)
        inp.update(src)
        metric = {"line_8_extra": int(len(shape) - len(_seeded_four_subset(shape, [seed])))}
        return _normalize_cells(inp), _normalize_cells(out), metric
    raise RuntimeError("failed to build row group")


def _build_col_group(color: int, rng: random.Random) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[str, int]]:
    for _ in range(120):
        line_len = rng.randint(4, 6)
        depth = rng.randint(2, 4)
        span = rng.randint(2, line_len)
        max_cells = min(10, int(depth * span))
        size = rng.randint(3, max_cells)
        side = rng.choice(["left", "right"])
        seed = (rng.randrange(span), int(depth - 1)) if side == "left" else (rng.randrange(span), 0)
        allowed = {(int(r), int(c)) for r in range(span) for c in range(depth)}
        shape = _grow_connected_subset(rng, allowed, size, connectivity=8, seed=seed)
        if not _shape_is_diagonal_from_seed(shape, [seed]):
            continue

        yoff = rng.randint(0, line_len - span)
        axis_c = int(depth)
        red = {(int(r), int(axis_c)): MARK for r in range(line_len)}
        if side == "left":
            src = {(int(r + yoff), int(c)): int(color) for r, c in shape}
        else:
            src = {(int(r + yoff), int(c + depth + 1)): int(color) for r, c in shape}

        out = dict(red)
        out.update(src)
        for r, c in src:
            rr, cc = _refl_col(int(axis_c), int(r), int(c))
            out[(int(rr), int(cc))] = int(color)

        inp = dict(red)
        inp.update(src)
        metric = {"line_8_extra": int(len(shape) - len(_seeded_four_subset(shape, [seed])))}
        return _normalize_cells(inp), _normalize_cells(out), metric
    raise RuntimeError("failed to build col group")


def _build_point_group(color: int, rng: random.Random) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[str, int]]:
    for _ in range(160):
        orient = rng.choice(["north", "south", "west", "east"])
        depth = rng.randint(3, 4)
        span = rng.randint(3, 5)
        max_cells = min(8, int(depth * span))
        size = rng.randint(3, max_cells)

        if orient in {"north", "south"}:
            seed = (int(depth - 1), span // 2) if orient == "north" else (0, span // 2)
            allowed = {(int(r), int(c)) for r in range(depth) for c in range(span)}
            shape = _grow_connected_subset(rng, allowed, size, connectivity=8, seed=seed)
            if not _shape_is_diagonal_from_seed(shape, [seed]):
                continue
            marker = (int(depth), span // 2)
            if orient == "north":
                src = {(int(r), int(c)): int(color) for r, c in shape}
            else:
                src = {(int(r + depth + 1), int(c)): int(color) for r, c in shape}
        else:
            seed = (span // 2, int(depth - 1)) if orient == "west" else (span // 2, 0)
            allowed = {(int(r), int(c)) for r in range(span) for c in range(depth)}
            shape = _grow_connected_subset(rng, allowed, size, connectivity=8, seed=seed)
            if not _shape_is_diagonal_from_seed(shape, [seed]):
                continue
            marker = (span // 2, int(depth))
            if orient == "west":
                src = {(int(r), int(c)): int(color) for r, c in shape}
            else:
                src = {(int(r), int(c + depth + 1)): int(color) for r, c in shape}

        inp = {marker: MARK}
        inp.update(src)

        actual_seed = None
        for pos in src:
            mr, mc = marker
            if abs(int(pos[0]) - int(mr)) + abs(int(pos[1]) - int(mc)) == 1:
                actual_seed = pos
                break
        if actual_seed is None:
            continue
        detected = _seeded_four_subset(sorted(src), [actual_seed])
        if not (0 < len(detected) < len(src)):
            continue

        out = dict(inp)
        for r, c in detected:
            rr, cc = _refl_point(marker, int(r), int(c))
            out[(int(rr), int(cc))] = int(color)

        metric = {
            "point_full8": int(len(src)),
            "point_detected4": int(len(detected)),
            "point_partial": int(len(src) - len(detected)),
        }
        return _normalize_cells(inp), _normalize_cells(out), metric
    raise RuntimeError("failed to build point group")


def _halo_clear(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (int(r + dr), int(c + dc)) in occupied:
                    return False
    return True


def _place_group(
    rng: random.Random,
    h: int,
    w: int,
    inp: dict[tuple[int, int], int],
    out: dict[tuple[int, int], int],
    occupied: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    rh, rw = _bounds(_nonbg_union(inp, out))
    for _ in range(240):
        dr = rng.randint(0, int(h - rh))
        dc = rng.randint(0, int(w - rw))
        shifted_inp = _shift_cells(inp, dr, dc)
        shifted_out = _shift_cells(out, dr, dc)
        union = _nonbg_union(shifted_inp, shifted_out)
        if not _halo_clear(union, occupied):
            continue
        occupied.update(union)
        return shifted_inp, shifted_out
    raise RuntimeError("failed to place group")


def analyze_variant_features(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_lab, red_n = label_mask(g == int(MARK), connectivity=4)
    fg8_lab, _ = label_mask(g != int(bg), connectivity=8)
    fg4_lab, _ = label_mask(g != int(bg), connectivity=4)

    line_groups = 0
    line_with_extra8 = 0
    point_groups = 0
    point_partial = 0
    point_zero = 0

    for k in range(1, int(red_n) + 1):
        reds = [(int(r), int(c)) for r, c in np.argwhere(red_lab == int(k))]
        axis = _red_axis(reds)
        if axis is None:
            continue
        if axis[0] in {"row", "col"}:
            line_groups += 1
            labels8 = {int(fg8_lab[int(r), int(c)]) for r, c in reds if int(fg8_lab[int(r), int(c)]) != 0}
            labels4 = {int(fg4_lab[int(r), int(c)]) for r, c in reds if int(fg4_lab[int(r), int(c)]) != 0}
            payload8 = {(int(r), int(c)) for r, c in _component_cells(fg8_lab, labels8) if int(g[int(r), int(c)]) not in {int(bg), int(MARK)}}
            payload4 = {(int(r), int(c)) for r, c in _component_cells(fg4_lab, labels4) if int(g[int(r), int(c)]) not in {int(bg), int(MARK)}}
            if len(payload8) > len(payload4):
                line_with_extra8 += 1
            continue

        point_groups += 1
        pr, pc = map(int, axis[1])
        seed_colors = {
            int(g[int(pr + dr), int(pc + dc)])
            for dr, dc in N4
            if _inb(g.shape, int(pr + dr), int(pc + dc)) and int(g[int(pr + dr), int(pc + dc)]) not in {int(bg), int(MARK)}
        }
        detected_total = 0
        full_total = 0
        for color in seed_colors:
            same4_lab, _ = label_mask(g == int(color), connectivity=4)
            same8_lab, _ = label_mask(g == int(color), connectivity=8)
            labels4 = set()
            for dr, dc in N4:
                rr, cc = int(pr + dr), int(pc + dc)
                if _inb(g.shape, rr, cc) and int(g[int(rr), int(cc)]) == int(color):
                    lbl = int(same4_lab[int(rr), int(cc)])
                    if lbl != 0:
                        labels4.add(lbl)
            labels8 = set()
            for dr, dc in N8:
                rr, cc = int(pr + dr), int(pc + dc)
                if _inb(g.shape, rr, cc) and int(g[int(rr), int(cc)]) == int(color):
                    lbl = int(same8_lab[int(rr), int(cc)])
                    if lbl != 0:
                        labels8.add(lbl)
            detected = {(int(r), int(c)) for r, c in _component_cells(same4_lab, labels4)}
            full = {(int(r), int(c)) for r, c in _component_cells(same8_lab, labels8)}
            detected_total += len(detected)
            full_total += len(full)

        if detected_total == 0:
            point_zero += 1
        elif detected_total < full_total:
            point_partial += 1

    return {
        "line_groups": int(line_groups),
        "line_with_extra8": int(line_with_extra8),
        "point_groups": int(point_groups),
        "point_partial": int(point_partial),
        "point_zero": int(point_zero),
    }


def _generate_pair(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    bg = rng.choice([c for c in range(10) if c != int(MARK)])
    colors = [c for c in range(10) if c not in {int(bg), int(MARK)}]
    rng.shuffle(colors)

    group_builders = [
        ("row", _build_row_group),
        ("col", _build_col_group),
        ("point", _build_point_group),
    ]

    plan = ["row", "col", "point"]
    if rng.random() < 0.7:
        plan.append(rng.choice(["row", "col", "point"]))
    if rng.random() < 0.35:
        plan.append(rng.choice(["row", "point"]))

    inp_canvas = np.full((int(h), int(w)), int(bg), dtype=int)
    out_canvas = np.full((int(h), int(w)), int(bg), dtype=int)
    occupied: set[tuple[int, int]] = set()
    metric_totals = {"line_extra_groups": 0, "point_partial_groups": 0}

    for idx, kind in enumerate(plan):
        color = int(colors[idx])
        builder = dict(group_builders)[kind]
        local_inp, local_out, metric = builder(color, rng)
        placed_inp, placed_out = _place_group(rng, h, w, local_inp, local_out, occupied)
        for (r, c), v in placed_inp.items():
            inp_canvas[int(r), int(c)] = int(v)
        for (r, c), v in placed_out.items():
            out_canvas[int(r), int(c)] = int(v)
        if metric.get("line_8_extra", 0) > 0:
            metric_totals["line_extra_groups"] += 1
        if metric.get("point_partial", 0) > 0:
            metric_totals["point_partial_groups"] += 1

    return inp_canvas, out_canvas, metric_totals


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = random.Random() if rng is None else rng
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, out, meta = _generate_pair(rng)
        except RuntimeError:
            continue
        if meta["line_extra_groups"] < 1 or meta["point_partial_groups"] < 1:
            continue
        if not np.array_equal(main(inp), out):
            continue
        if np.array_equal(_baseline_original_main(inp), out):
            continue
        features = analyze_variant_features(inp)
        if features["line_with_extra8"] < 1:
            continue
        if features["point_partial"] < 1:
            continue
        return inp
    raise RuntimeError("failed to generate a valid D3/1+D7/0 combo input")
