from __future__ import annotations

# Combo variant of 7ed72f31 (index 11)
# combo_spec: D3/0+D13/1
# Name: 多色对称补全
#
# Design analysis
# a) Concrete behavior and trace
#    - 保留原题的红色几何语义: 红横线表示上下镜像轴，红竖线表示左右镜像轴，单个红点表示中心点反射。
#    - 与原题不同，payload 不再是单色；同一个 payload 使用 2-3 种非红非背景色，而且输入已经“部分对称”。
#    - 小例子: 若红横线在第 5 行，输入上方已有 {(4,7):蓝, (3,8):黄}，下方已有 {(6,7):蓝}，则输出只补
#      (7,8):黄；不会重画已有的 (6,7)，也不会把整块重新生成。
#    - 与 D3/0 的协同点在于“缺失像素的颜色必须逐像素继承其镜像对应点”，因此 solver 必须跟踪 payload
#      的精确坐标与颜色，而不能只按颜色集合推断。
# b) Complexity calibration
#    - 每组固定 2 或 3 色，避免颜色过多导致视觉噪声，同时保证“多色”在少量样例中可见。
#    - 每组都强制同时出现:
#      1. 至少一对已经成对存在的对称像素；
#      2. 至少一处仅单侧存在、需要补全的像素；
#      3. 至少两种颜色。
#    - 这样样例不会退化成“单侧整块镜像”或“其实已经完全对称”的弱版本。
# c) Implicit-assumption audit of original main()
#    - 原版 `payload = {v for v in vals if v != MARK}` 只记录颜色集合；多色补全时不够，因为同色散点和同组像素
#      必须区分。这里改成直接提取组件中的精确 payload 像素列表。
#    - 原版 row/col 分支按 bbox + 颜色扫全图；这依赖“同色旁观像素不会被卷入”。融合后该前提不稳，因此
#      solver 只反射当前红标记组件自身的 payload。
#    - 原版直接写 `out[rr, cc] = v`，默认目标一定是背景。D13/1 打破了这一点；这里统一改成“只在背景格补色”。
#    - 原版 isolated 分支按“单色 8 连通分量”处理；多色后要改成“红点旁的整块非红 8 连通组件”，并逐像素保色。
# d) Placement legality
#    - generator 采用 output-first: 先构造完整对称输出，再删除部分对称对应像素得到输入。
#    - 放置时按完整输出 bbox 留 2 格 moat，避免组间相碰、越界、或意外连成同一组件。
#    - row/col 组把 payload 约束在红线投影宽/高范围内；point 组使用单一中心，保证人类能局部识别对称来源。
# e) Visual observability
#    - 在 2-4 个样例里，人类会同时看到:
#      1. 多色图案的配色被逐点保留；
#      2. 两侧已经有一部分是对称的；
#      3. 输出只把缺口补齐，而不是整块复制。
#    - 这三点在每组上重复出现，足以支持归纳。
# f) Visual cleanliness
#    - 不生成自由噪声散点，不让不同组贴边；每个红标记附近只对应一个清晰局部结构。
#    - 每组的多色采用局部色团分配，不做随机彩噪，避免花屏。
# g) Color inference
#    - 没有颜色映射表。颜色只随几何对应搬运，解题者推断的是“镜像位置继承原像素颜色”。
# h) New solver assumptions after fusion
#    - 红标记组件几何只能是单横线、单竖线或单红点；generator 只生成这三类。
#    - 需要补全的目标格在输入中为背景，而不是错误颜色；generator 用 output-first 与 oracle 校验保证。
#    - 对 mixed 组件，输入中保留下来的每一侧像素子集都与靠轴/靠中心的锚点连通，这样 `_extract_groups()`
#      看到的就是完整参与补全的组件，而不是碎裂成无关旁观像素。
# i) Generator-side guarantees and anti-degeneracy checks
#    - 每个样例至少含一个 row 组和一个 col 组；其余组在 row/col/point 中采样。
#    - 每组都记录 `payload_colors`, `kept_pairs`, `missing_pairs`；`generate_input()` 只接受所有组都满足
#      `payload_colors >= 2`, `kept_pairs >= 1`, `missing_pairs >= 1` 的样本。
#    - 这一定量指标正是融合行为本身，而不是单纯 `main(input)` 不报错。

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask


MARK = 2
ALL_COLORS = list(range(10))
PALETTE = [c for c in ALL_COLORS if c != MARK]
MAX_GEN_ATTEMPTS = 300


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = list(coords)
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


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
        payload = list(group["payload"])
        colors = sorted({int(v) for _r, _c, v in payload})
        mirrored_hits = 0
        mirrored_gaps = 0
        seen_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        axis = group["axis"]
        pos_to_color = {(int(r), int(c)): int(v) for r, c, v in payload}
        for r, c, v in payload:
            rr, cc = _reflect(axis, int(r), int(c))
            pair_key = tuple(sorted(((int(r), int(c)), (int(rr), int(cc)))))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            rv = pos_to_color.get((int(rr), int(cc)))
            if rv == int(v):
                mirrored_hits += 1
            else:
                mirrored_gaps += 1
        stats.append(
            {
                "kind": str(group["kind"]),
                "axis": str(group["axis"][0]),
                "payload_pixels": len(payload),
                "payload_colors": len(colors),
                "colors": colors,
                "kept_pairs": int(mirrored_hits),
                "missing_pairs": int(mirrored_gaps),
            }
        )
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete partial symmetry around red markers while preserving per-pixel colors.

    Fusion behavior:
    - payloads are multicolor (2-3 colors)
    - the input may already contain some symmetric counterparts
    - only missing background cells are filled
    """
    g = np.array(grid, dtype=int)
    bg, groups = _extract_groups(g)
    out = g.copy()

    for group in groups:
        axis = group["axis"]
        for r, c, v in group["payload"]:
            rr, cc = _reflect(axis, int(r), int(c))
            if _inb(g.shape, rr, cc) and int(out[int(rr), int(cc)]) == int(bg):
                out[int(rr), int(cc)] = int(v)

    return out


def _grow_connected_tree(
    rng: random.Random,
    allowed: set[tuple[int, int]],
    n: int,
    connectivity: int,
    seed: tuple[int, int],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int] | None]]:
    if seed not in allowed or int(n) > len(allowed):
        raise ValueError("invalid growth request")
    shape = {seed}
    order = [seed]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {seed: None}
    while len(shape) < int(n):
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for cell in allowed:
            if cell in shape:
                continue
            parents = [nb for nb in _neighbors(cell, connectivity) if nb in shape]
            if parents:
                frontier.append((cell, rng.choice(parents)))
        if not frontier:
            raise ValueError("disconnected allowed region")
        cell, par = rng.choice(frontier)
        shape.add(cell)
        order.append(cell)
        parent[cell] = par
    return order, parent


def _assign_tree_states(
    rng: random.Random,
    order: list[tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int] | None],
) -> dict[tuple[int, int], str]:
    if not order:
        raise ValueError("empty order")
    for _ in range(40):
        states: dict[tuple[int, int], str] = {order[0]: "both"}
        nonboth = 0
        for cell in order[1:]:
            par = parent[cell]
            pstate = states[par]
            if pstate == "both":
                state = rng.choices(["both", "a", "b"], weights=[5, 2, 2], k=1)[0]
            elif pstate == "a":
                state = "a"
            else:
                state = "b"
            states[cell] = state
            if state != "both":
                nonboth += 1
        if nonboth > 0:
            return states
    raise ValueError("failed to generate partial-symmetry state tree")


def _assign_cluster_colors(
    rng: random.Random,
    coords: list[tuple[int, int]],
    colors: list[int],
) -> dict[tuple[int, int], int]:
    seeds = rng.sample(coords, len(colors))
    mapping: dict[tuple[int, int], int] = {}
    for coord in coords:
        idx = min(
            range(len(colors)),
            key=lambda i: (
                abs(int(coord[0]) - int(seeds[i][0])) + abs(int(coord[1]) - int(seeds[i][1])),
                int(i),
            ),
        )
        mapping[(int(coord[0]), int(coord[1]))] = int(colors[idx])
    return mapping


def _pick_group_colors(rng: random.Random, bg: int) -> list[int]:
    pool = [c for c in PALETTE if c != int(bg)]
    k = 2 if rng.random() < 0.65 else 3
    return rng.sample(pool, k)


def _build_multicolor_pairs(
    rng: random.Random,
    half_coords: list[tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int] | None],
    axis: tuple[str, int | tuple[int, int]],
    colors: list[int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], dict[str, int]]:
    states = _assign_tree_states(rng, half_coords, parent)
    cmap = _assign_cluster_colors(rng, half_coords, colors)
    out_cells: list[tuple[int, int, int]] = []
    inp_cells: list[tuple[int, int, int]] = []
    kept_pairs = 0
    missing_pairs = 0

    for cell in half_coords:
        r, c = cell
        v = int(cmap[cell])
        rr, cc = _reflect(axis, int(r), int(c))
        out_cells.append((int(r), int(c), int(v)))
        out_cells.append((int(rr), int(cc), int(v)))
        state = states[cell]
        if state in {"both", "a"}:
            inp_cells.append((int(r), int(c), int(v)))
        if state in {"both", "b"}:
            inp_cells.append((int(rr), int(cc), int(v)))
        if state == "both":
            kept_pairs += 1
        else:
            missing_pairs += 1

    return inp_cells, out_cells, {
        "payload_colors": len(colors),
        "pair_count": len(half_coords),
        "kept_pairs": int(kept_pairs),
        "missing_pairs": int(missing_pairs),
    }


def _build_row_group(rng: random.Random, bg: int) -> dict[str, object]:
    colors = _pick_group_colors(rng, bg)
    for _ in range(100):
        line_len = rng.randint(3, 6)
        side_height = rng.randint(2, 4)
        max_cells = min(9, line_len * side_height)
        min_cells = max(4, len(colors))
        if max_cells < min_cells:
            continue
        n = rng.randint(min_cells, max_cells)
        seed = (1, rng.randrange(line_len))
        allowed = {(int(r), int(c)) for r in range(1, side_height + 1) for c in range(line_len)}
        order, parent = _grow_connected_tree(rng, allowed, n, 4, seed)
        rows = {r for r, _c in order}
        cols = {c for _r, c in order}
        if len(rows) < 2 or len(cols) < 2:
            continue
        inp_payload, out_payload, meta = _build_multicolor_pairs(rng, order, parent, ("row", 0), colors)
        reds = [(0, c, MARK) for c in range(line_len)]
        out_cells = out_payload + reds
        inp_cells = inp_payload + reds
        bbox = _bbox((r, c) for r, c, _v in out_cells)
        return {
            "type": "row",
            "inp_cells": inp_cells,
            "out_cells": out_cells,
            "bbox": bbox,
            **meta,
        }
    raise RuntimeError("row group generation failed")


def _build_col_group(rng: random.Random, bg: int) -> dict[str, object]:
    colors = _pick_group_colors(rng, bg)
    for _ in range(100):
        line_len = rng.randint(3, 6)
        side_width = rng.randint(2, 4)
        max_cells = min(9, line_len * side_width)
        min_cells = max(4, len(colors))
        if max_cells < min_cells:
            continue
        n = rng.randint(min_cells, max_cells)
        seed = (rng.randrange(line_len), 1)
        allowed = {(int(r), int(c)) for r in range(line_len) for c in range(1, side_width + 1)}
        order, parent = _grow_connected_tree(rng, allowed, n, 4, seed)
        rows = {r for r, _c in order}
        cols = {c for _r, c in order}
        if len(rows) < 2 or len(cols) < 2:
            continue
        inp_payload, out_payload, meta = _build_multicolor_pairs(rng, order, parent, ("col", 0), colors)
        reds = [(r, 0, MARK) for r in range(line_len)]
        out_cells = out_payload + reds
        inp_cells = inp_payload + reds
        bbox = _bbox((r, c) for r, c, _v in out_cells)
        return {
            "type": "col",
            "inp_cells": inp_cells,
            "out_cells": out_cells,
            "bbox": bbox,
            **meta,
        }
    raise RuntimeError("col group generation failed")


def _build_point_group(rng: random.Random, bg: int) -> dict[str, object]:
    colors = _pick_group_colors(rng, bg)
    for _ in range(120):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        allowed = {
            (int(r), int(c))
            for r in range(-h, 1)
            for c in range(-w, w + 1)
            if not (r == 0 and c == 0) and (r < 0 or (r == 0 and c < 0))
        }
        seed = rng.choice([(-1, 0), (0, -1)])
        max_cells = min(8, len(allowed))
        min_cells = max(4, len(colors))
        if max_cells < min_cells:
            continue
        n = rng.randint(min_cells, max_cells)
        order, parent = _grow_connected_tree(rng, allowed, n, 4, seed)
        rows = {r for r, _c in order}
        cols = {c for _r, c in order}
        if len(rows) < 2 and len(cols) < 2:
            continue
        inp_payload, out_payload, meta = _build_multicolor_pairs(rng, order, parent, ("point", (0, 0)), colors)
        out_cells = out_payload + [(0, 0, MARK)]
        inp_cells = inp_payload + [(0, 0, MARK)]
        bbox = _bbox((r, c) for r, c, _v in out_cells)
        return {
            "type": "point",
            "inp_cells": inp_cells,
            "out_cells": out_cells,
            "bbox": bbox,
            **meta,
        }
    raise RuntimeError("point group generation failed")


def _build_group(rng: random.Random, kind: str, bg: int) -> dict[str, object]:
    if kind == "row":
        return _build_row_group(rng, bg)
    if kind == "col":
        return _build_col_group(rng, bg)
    if kind == "point":
        return _build_point_group(rng, bg)
    raise ValueError(f"unknown kind: {kind}")


def _translate_cells(
    cells: list[tuple[int, int, int]],
    top: int,
    left: int,
    r0: int,
    c0: int,
) -> list[tuple[int, int, int]]:
    return [(int(top + r - r0), int(left + c - c0), int(v)) for r, c, v in cells]


def _place_group(
    rng: random.Random,
    inp: np.ndarray,
    out: np.ndarray,
    reserved: np.ndarray,
    group: dict[str, object],
    bg: int,
) -> bool:
    h, w = inp.shape
    r0, c0, r1, c1 = map(int, group["bbox"])
    gh = int(r1 - r0 + 1)
    gw = int(c1 - c0 + 1)
    moat = 2
    if gh + 2 * moat > h or gw + 2 * moat > w:
        return False

    for _ in range(240):
        top = rng.randint(moat, h - gh - moat)
        left = rng.randint(moat, w - gw - moat)
        rr0 = int(top - moat)
        cc0 = int(left - moat)
        rr1 = int(top + gh + moat - 1)
        cc1 = int(left + gw + moat - 1)
        if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        out_cells = _translate_cells(group["out_cells"], top, left, r0, c0)
        inp_cells = _translate_cells(group["inp_cells"], top, left, r0, c0)
        if any(int(out[r, c]) != int(bg) for r, c, _v in out_cells):
            continue
        if any(int(inp[r, c]) != int(bg) for r, c, _v in inp_cells):
            continue

        for r, c, v in out_cells:
            out[int(r), int(c)] = int(v)
        for r, c, v in inp_cells:
            inp[int(r), int(c)] = int(v)
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    bg = rng.choice([c for c in ALL_COLORS if c != MARK])
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    n_groups = rng.randint(4, 6)
    kinds = ["row", "col"]
    while len(kinds) < n_groups:
        kinds.append(rng.choice(["row", "col", "point"]))
    rng.shuffle(kinds)

    placed_groups: list[dict[str, object]] = []
    for kind in kinds:
        placed = False
        for _ in range(100):
            group = _build_group(rng, kind, bg)
            if _place_group(rng, inp, out, reserved, group, bg):
                placed_groups.append(group)
                placed = True
                break
        if not placed:
            raise RuntimeError(f"unable to place {kind} group")

    return inp, out, placed_groups


def _generate_valid_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    last_err: Exception | None = None
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, intended_out, groups = _generate_case(rng)
            actual_out = main(inp)
            if not np.array_equal(actual_out, intended_out):
                raise AssertionError("oracle mismatch")
            stats = analyze_input(inp)
            if len(stats) != len(groups):
                raise AssertionError("group count mismatch")
            if any(int(s["payload_colors"]) < 2 for s in stats):
                raise AssertionError("missing multicolor group")
            if any(int(s["kept_pairs"]) < 1 for s in stats):
                raise AssertionError("missing already-symmetric evidence")
            if any(int(s["missing_pairs"]) < 1 for s in stats):
                raise AssertionError("missing completion evidence")
            return inp, intended_out, groups, stats
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"failed to generate valid combo case: {last_err}")


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    try:
        inp, _out, _groups, _stats = _generate_valid_case(rng)
        return inp
    except Exception as exc:
        raise RuntimeError(f"failed to generate valid combo sample (seed={seed}): {exc}") from exc
