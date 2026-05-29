# Combo variant of 7ed72f31 (index 9)
# combo_spec: D3/0+D5/1
# Name: 多色精灵颜色混合
#
# Design analysis
# - concrete_behavior:
#   - 保留原题的红标记几何语义：横红线=行镜像，竖红线=列镜像，单红点=中心点反射。
#   - D3/0: 每个 payload 不再是单色，而是 2-3 色的小连通精灵；反射时逐像素保留原色。
#   - D5/1: 若某个镜像像素写入时，目标格已经有别的非背景像素，则不直接覆盖，而是把
#     “已有颜色 + 镜像颜色”映射为第三种颜色。这里使用“非红非背景调色板中的索引相加取模，
#     若结果撞回任一输入色就顺延到下一个颜色”的规则，得到稳定且对称的第三色。
#   - 具体小例子：背景 0，活跃调色板 [1,3,4,5,6,7,8,9]。若蓝(1)镜到黄(4)单元上，
#     则先取索引和，再顺延避开 {1,4}，得到第三色；同一多色 payload 内不同源像素与同一
#     挡板颜色重叠时，会产生不同混色，输出能直接看见“颜色逐像素参与反射”。
# - behavior_trace:
#   - 原 solver 对混合组件采取“颜色集合 + bbox 全图扫描”策略；单色时通常没问题，但多色时，
#     附近同色杂点或自由精灵可能被误卷入反射。
#   - 融合后必须先精确抽取“属于该带标记组件的 payload 像素集合”，只反射这些像素。
#   - 原任务默认镜像目标为空白，直接覆盖即可；融合后生成器会故意把一些小自由精灵放在镜像
#     目标里，于是 solver 需要显式处理 overlap -> color-mix，而不是简单赋值。
# - complexity_calibration:
#   - 每个可反射 payload 固定 2 或 3 色，像素数 4-11。颜色数不再扩张到 4+，避免视觉噪声。
#   - 每个样本至少包含 row 组和 col 组，并额外加入 1-3 个 row/col/point 组，保证几何反射
#     规则重复出现。
#   - 每个样本再强制放置 3-6 个“镜像落点挡板”像素；它们位于未来镜像区域内，但在输入里与
#     红标记组断开，因此人类能直接看到“原地已有像素被混色，而不是被普通复制覆盖”。
# - implicit_assumption_audit:
#   - `payload = {v for v in vals if v != MARK}` 这种“颜色集合代表 payload”在多色变体下失效，
#     因此 solver 改为逐组保存精确 payload 坐标。
#   - 原 isolated 分支按“单色 8 连通组件”反射；多色下应对整个非红 8 连通组件逐像素反射。
#   - 原写入 `out[rr, cc] = v` 假设目标是背景；融合后不成立，必须改成 `_paint_reflection(...)`。
#   - 新 solver 的处理顺序若影响结果会有风险，因此生成器保证：不同带标记组之间的镜像目标
#     互不覆盖；唯一的重叠来自“镜像像素 vs 输入中预先存在的自由挡板像素”。
# - placement_legality:
#   - 活跃反射组按完整 output footprint + 1 像素 halo 放置，防止组间意外 4/8 连通或镜像互撞。
#   - 自由挡板只放在当前背景格，且与任何现有输入前景都不 4 邻接，这样不会并入红标记组件。
#   - 生成器还避免把挡板放到红轴/红点上；solver 也保守地把红色当作不可混的控制标记。
# - visual_observability:
#   - 多色 payload 本身在输入里就可见；输出里大部分镜像像素保持原色，小部分撞上挡板后变成
#     第三色。观察者可以从 2-4 个样例归纳出：几何规则不变，但 overlap 时颜色按固定混色表变。
#   - 挡板像素数量有限、分散到至少两个组附近，既能重复支撑规则，又不会让整张图变成花屏。
# - visual_cleanliness:
#   - 反射组之间保持空隙；混色只发生在少量未来镜像落点，且不叠加三层以上。
#   - 自由挡板使用单像素，避免它们自己形成新的大结构干扰“谁在镜谁”的辨识。
# - fusion-specific verification:
#   - `behavior_metrics()` 量化三个核心量：多色组数量、输入中被镜像击中的非背景像素数、
#     以及“混色版输出”和“无混色覆盖版输出”的差异格数。
#   - 生成器拒绝任何 `changed_vs_nomix == 0` 或 `overlap_cells < 3` 的样本，确保不会退化回
#     “只是多色反射”或“只是原题无重叠覆盖”。

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
MAX_GEN_ATTEMPTS = 320


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


def _mix_palette(bg: int) -> list[int]:
    return [int(c) for c in PALETTE if int(c) != int(bg)]


def _mix_colors(existing: int, incoming: int, bg: int, mix_palette: list[int]) -> int:
    existing = int(existing)
    incoming = int(incoming)
    bg = int(bg)
    if existing == int(bg):
        return int(incoming)
    if existing == int(MARK):
        return int(existing)
    if existing == int(incoming):
        return int(incoming)
    if existing not in mix_palette or incoming not in mix_palette:
        return int(incoming)
    idx_old = mix_palette.index(int(existing))
    idx_new = mix_palette.index(int(incoming))
    idx = int((idx_old + idx_new) % len(mix_palette))
    for _ in range(len(mix_palette)):
        candidate = int(mix_palette[idx])
        if candidate not in {int(existing), int(incoming)}:
            return int(candidate)
        idx = int((idx + 1) % len(mix_palette))
    return int(incoming)


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
                        labels.add(int(lbl))
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
        payload = list(group["payload"])
        colors = sorted({int(v) for _r, _c, v in payload})
        stats.append(
            {
                "kind": str(group["kind"]),
                "axis": str(group["axis"][0]),
                "payload_pixels": int(len(payload)),
                "payload_colors": int(len(colors)),
                "colors": colors,
            }
        )
    return stats


def _solve(grid: np.ndarray, *, mix_overlap: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg, groups = _extract_groups(g)
    mix_palette = _mix_palette(int(bg))

    for group in groups:
        axis = group["axis"]
        for r, c, v in group["payload"]:
            rr, cc = _reflect(axis, int(r), int(c))
            if not _inb(g.shape, rr, cc):
                continue
            cur = int(out[rr, cc])
            if cur == int(MARK):
                continue
            if not mix_overlap or cur == int(bg):
                out[rr, cc] = int(v)
            else:
                out[rr, cc] = _mix_colors(int(cur), int(v), int(bg), mix_palette)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect multicolor payload sprites around red markers.

    Combo behavior:
    - payload pixels keep their individual colors under reflection;
    - if a reflected pixel lands on an existing non-background pixel, the cell is recolored by
      a symmetric palette-index mixing rule instead of direct overwrite.
    """
    return _solve(np.array(grid, dtype=int), mix_overlap=True)


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg, groups = _extract_groups(g)
    mix_palette = _mix_palette(int(bg))
    combo = _solve(g, mix_overlap=True)
    nomix = _solve(g, mix_overlap=False)

    overlap_cells: dict[tuple[int, int], tuple[int, int, int]] = {}
    for group in groups:
        axis = group["axis"]
        for r, c, v in group["payload"]:
            rr, cc = _reflect(axis, int(r), int(c))
            if not _inb(g.shape, rr, cc):
                continue
            existing = int(g[rr, cc])
            if existing in {int(bg), int(MARK)} or existing == int(v):
                continue
            mixed = _mix_colors(int(existing), int(v), int(bg), mix_palette)
            overlap_cells[(int(rr), int(cc))] = (int(existing), int(v), int(mixed))

    groups_stats = analyze_input(g)
    multicolor_groups = sum(int(s["payload_colors"]) >= 2 for s in groups_stats)
    mixed_output_colors = len({int(mixed) for _old, _new, mixed in overlap_cells.values()})
    third_color_cells = sum(
        int(mixed not in {int(old), int(new)}) for old, new, mixed in overlap_cells.values()
    )
    changed_vs_nomix = int(np.sum(combo != nomix))
    extra_nonbg = int(np.sum((g != int(bg)) & (g != int(MARK)))) - sum(
        int(s["payload_pixels"]) for s in groups_stats
    )
    return {
        "group_count": int(len(groups_stats)),
        "multicolor_groups": int(multicolor_groups),
        "overlap_cells": int(len(overlap_cells)),
        "third_color_cells": int(third_color_cells),
        "mixed_output_colors": int(mixed_output_colors),
        "changed_vs_nomix": int(changed_vs_nomix),
        "free_nonred_pixels": int(max(0, extra_nonbg)),
    }


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
    pool = [int(c) for c in PALETTE if c != int(bg)]
    k = 2 if rng.random() < 0.68 else 3
    return rng.sample(pool, k)


def _make_group_record(
    kind: str,
    axis: tuple[str, int | tuple[int, int]],
    payload: list[tuple[int, int, int]],
    reflected: list[tuple[int, int, int]],
    marker_cells: dict[tuple[int, int], int],
) -> dict[str, object]:
    local_in = dict(marker_cells)
    for r, c, v in payload:
        local_in[(int(r), int(c))] = int(v)
    local_out = dict(local_in)
    for r, c, v in reflected:
        local_out[(int(r), int(c))] = int(v)
    return {
        "kind": str(kind),
        "axis": axis,
        "payload": [(int(r), int(c), int(v)) for r, c, v in payload],
        "reflected": [(int(r), int(c), int(v)) for r, c, v in reflected],
        "input": local_in,
        "output": local_out,
    }


def _build_row_group(rng: random.Random, bg: int) -> dict[str, object]:
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
        allowed = {(int(r), int(c)) for r in range(height) for c in range(width)}
        shape = _grow_connected_subset(rng, allowed, n, 4, seed)
        r0, c0, r1, c1 = _bbox(shape)
        if r1 == r0 or c1 == c0:
            continue
        xoff = rng.randint(0, line_len - width)
        cmap = _assign_cluster_colors(rng, shape, colors)
        top = [(int(r), int(c + xoff), int(cmap[(r, c)])) for r, c in shape]
        bottom = [(int(2 * height - r), int(c + xoff), int(cmap[(r, c)])) for r, c in shape]
        marker = {(int(height), int(c)): MARK for c in range(line_len)}
        if rng.random() < 0.5:
            payload, reflected = top, bottom
        else:
            payload, reflected = bottom, top
        return _make_group_record("row", ("row", int(height)), payload, reflected, marker)
    raise RuntimeError("row group generation failed")


def _build_col_group(rng: random.Random, bg: int) -> dict[str, object]:
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
        allowed = {(int(r), int(c)) for r in range(height) for c in range(width)}
        shape = _grow_connected_subset(rng, allowed, n, 4, seed)
        r0, c0, r1, c1 = _bbox(shape)
        if r1 == r0 or c1 == c0:
            continue
        yoff = rng.randint(0, line_len - height)
        cmap = _assign_cluster_colors(rng, shape, colors)
        left = [(int(r + yoff), int(c), int(cmap[(r, c)])) for r, c in shape]
        right = [(int(r + yoff), int(2 * width - c), int(cmap[(r, c)])) for r, c in shape]
        marker = {(int(r), int(width)): MARK for r in range(line_len)}
        if rng.random() < 0.5:
            payload, reflected = left, right
        else:
            payload, reflected = right, left
        return _make_group_record("col", ("col", int(width)), payload, reflected, marker)
    raise RuntimeError("col group generation failed")


def _rot90(dr: int, dc: int) -> tuple[int, int]:
    return int(-dc), int(dr)


def _rotate_rel(coords: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = [(int(r), int(c)) for r, c in coords]
    for _ in range(int(turns) % 4):
        out = [_rot90(r, c) for r, c in out]
    return out


def _build_point_group(rng: random.Random, bg: int) -> dict[str, object]:
    colors = _pick_group_colors(rng, bg)
    for _ in range(100):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        max_cells = min(11, (h + 1) * (w + 1) - 1)
        min_cells = max(4, len(colors))
        if max_cells < min_cells:
            continue
        allowed = {
            (int(dr), int(dc))
            for dr in range(-h, 1)
            for dc in range(-w, 1)
            if not (dr == 0 and dc == 0)
        }
        seed = rng.choice([(-1, 0), (0, -1)])
        rel = _grow_connected_subset(rng, allowed, rng.randint(min_cells, max_cells), 4, seed)
        rows = {int(r) for r, _c in rel}
        cols = {int(c) for _r, c in rel}
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
        orig = [(int(r + shift_r), int(c + shift_c), int(cmap[(r, c)])) for r, c in rel]
        twin = [(int(-r + shift_r), int(-c + shift_c), int(cmap[(r, c)])) for r, c in rel]
        marker = {(int(shift_r), int(shift_c)): MARK}
        if rng.random() < 0.5:
            payload, reflected = orig, twin
        else:
            payload, reflected = twin, orig
        return _make_group_record("point", ("point", (int(shift_r), int(shift_c))), payload, reflected, marker)
    raise RuntimeError("point group generation failed")


def _group_builder(name: str):
    if name == "row":
        return _build_row_group
    if name == "col":
        return _build_col_group
    if name == "point":
        return _build_point_group
    raise ValueError(f"unknown group type: {name}")


def _place_group(
    rng: random.Random,
    grid_in: np.ndarray,
    forbidden: set[tuple[int, int]],
    group: dict[str, object],
) -> dict[str, object] | None:
    h, w = grid_in.shape
    max_r = max(int(r) for r, _c in group["output"])
    max_c = max(int(c) for _r, c in group["output"])
    for _ in range(200):
        top = rng.randint(0, int(h - max_r - 1))
        left = rng.randint(0, int(w - max_c - 1))
        placed_out = [(int(top + r), int(left + c), int(v)) for (r, c), v in group["output"].items()]
        if any((int(r), int(c)) in forbidden for r, c, _v in placed_out):
            continue
        placed_in = [(int(top + r), int(left + c), int(v)) for (r, c), v in group["input"].items()]
        for r, c, v in placed_in:
            grid_in[r, c] = int(v)
        abs_group = {
            "kind": str(group["kind"]),
            "axis": group["axis"],
            "payload": [(int(top + r), int(left + c), int(v)) for r, c, v in group["payload"]],
            "reflected": [(int(top + r), int(left + c), int(v)) for r, c, v in group["reflected"]],
            "input_cells": [(int(top + r), int(left + c), int(v)) for (r, c), v in group["input"].items()],
            "output_cells": placed_out,
        }
        for r, c, _v in placed_out:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = int(r + dr), int(c + dc)
                    if _inb(grid_in.shape, rr, cc):
                        forbidden.add((int(rr), int(cc)))
        return abs_group
    return None


def _can_place_blocker(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    if int(grid[r, c]) != int(bg):
        return False
    for nr, nc in _neighbors((int(r), int(c)), 4):
        if _inb(grid.shape, nr, nc) and int(grid[nr, nc]) != int(bg):
            return False
    return True


def _place_blockers(
    rng: random.Random, grid_in: np.ndarray, placed_groups: list[dict[str, object]], bg: int
) -> list[tuple[int, int, int, int]]:
    blockers: list[tuple[int, int, int, int]] = []
    pool = [int(c) for c in PALETTE if c != int(bg)]
    group_hits: set[int] = set()
    target = rng.randint(3, 6)

    for _ in range(200):
        if len(blockers) >= int(target) and len(group_hits) >= 2:
            break
        gidx = rng.randrange(len(placed_groups))
        reflected = list(placed_groups[gidx]["reflected"])
        rng.shuffle(reflected)
        placed_here = False
        for r, c, src_col in reflected:
            if not _can_place_blocker(grid_in, int(r), int(c), int(bg)):
                continue
            choices = [int(col) for col in pool if int(col) not in {int(src_col), int(MARK)}]
            if not choices:
                continue
            blocker_col = int(rng.choice(choices))
            grid_in[int(r), int(c)] = int(blocker_col)
            blockers.append((int(r), int(c), int(blocker_col), int(src_col)))
            group_hits.add(int(gidx))
            placed_here = True
            break
        if len(blockers) >= int(target) and len(group_hits) >= 2:
            break
        if not placed_here and len(blockers) >= 3 and len(group_hits) >= 2:
            break

    if len(blockers) < 3 or len(group_hits) < 2:
        return []
    return blockers


def _generate_input_once(rng: random.Random) -> np.ndarray:
    h = rng.randint(22, 28)
    w = rng.randint(22, 28)
    bg = rng.choice(PALETTE)
    grid_in = np.full((h, w), int(bg), dtype=int)
    forbidden: set[tuple[int, int]] = set()

    group_count = rng.randint(4, 5)
    group_types = ["row", "col"]
    while len(group_types) < int(group_count):
        group_types.append(rng.choice(["row", "col", "point"]))
    rng.shuffle(group_types)

    placed_groups: list[dict[str, object]] = []
    for name in group_types:
        builder = _group_builder(name)
        placed = None
        for _ in range(80):
            local_group = builder(rng, int(bg))
            placed = _place_group(rng, grid_in, forbidden, local_group)
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError(f"unable to place {name} group")
        placed_groups.append(placed)

    blockers = _place_blockers(rng, grid_in, placed_groups, int(bg))
    if not blockers:
        raise RuntimeError("unable to place blockers")

    return grid_in


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        try:
            inp = _generate_input_once(rng)
        except RuntimeError:
            continue
        out = main(inp)
        stats = analyze_input(inp)
        if len(stats) < 4:
            continue
        if any(int(s["payload_colors"]) < 2 for s in stats):
            continue
        metrics = behavior_metrics(inp)
        if metrics["multicolor_groups"] < 4:
            continue
        if metrics["overlap_cells"] < 3:
            continue
        if metrics["third_color_cells"] < 3:
            continue
        if metrics["changed_vs_nomix"] < 3:
            continue
        if metrics["free_nonred_pixels"] < 3:
            continue
        if np.array_equal(out, _solve(inp, mix_overlap=False)):
            continue
        return inp
    raise RuntimeError("failed to generate a valid multicolor-mixing 7ed72f31 combo input")
