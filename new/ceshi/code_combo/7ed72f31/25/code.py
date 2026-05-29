from __future__ import annotations

# Combo variant of 7ed72f31 (index 25)
# combo_spec: D11/2+D13/1
# Name: 碎片对称聚合补全
#
# Design analysis
# a) Concrete behavior and trace
#    - 原题里，每个红标记组各自把一侧 payload 镜像到另一侧，组与组之间互不相干。
#    - 这个融合版改成“局部补全 + 全局拼合”：
#      1. 每个红竖线/红横线/红点仍然定义各自的局部对称规则；
#      2. 但输入中的 payload 已经是“部分对称”的，输出只补背景缺口；
#      3. 多个局部组在输入中彼此分离，补全后它们会在边界处刚好接上，拼成一个更大的整体图案。
#    - 一个小追踪：假设左、中、右三个红竖线组横向排开。左组右边界少了两格，中组左右都几乎完整，
#      右组左边界少了两格。原题逻辑会把每组当成独立镜像复制；这里则只把左组和右组缺失的内侧边界补上，
#      于是三个块在输出中连成一整条带状图案。输入里还能看到两侧已存在的对称像素，因此更像“查缺补漏”。
# b) Complexity calibration
#    - 为了让 D11 和 D13 都明显可见，generator 固定使用 3 个局部组：左/中/右三个竖轴对称碎片。
#      左右组在输出里与中组接边，但输入里因为内侧缺口而断开，肉眼能看到“补全后拼合”的效果。
#    - 只保留一种局部标记几何（红竖线）作为主训练分布，避免同时混入太多轴类型导致样例难归纳；
#      solver 仍保留横线/点对称支持，但生成器主要产出竖线组，使核心规律稳定重复。
#    - 每组颜色唯一，防止“哪个像素属于哪个标记组”发生歧义。
# c) Implicit-assumption audit of the original main()
#    - 原 solver 对 row/col 组的关键假设是：只要颜色相同且落在组 bbox 的投影带内，就都属于该组。
#      这个假设在 D13/1 下会失效，因为输入里允许出现同色旁观碎片；它们不该被补全。
#      因此这里把 row/col/embedded-point 分支改成“只反射当前 marker component 内部已有的 payload 像素”。
#    - 原 solver 直接写 `out[rr, cc] = v`，隐含目标格原本是背景；D13/1 不再成立，因为输入里可已有正确对称像素。
#      因此新版本统一改成“仅当目标仍为背景时补像素”。
#    - 原 solver 的 isolated red 分支会反射整块 8 连通同色组件；本融合版仍保留这一分支，
#      但 generator 不依赖它，避免和“多碎片聚合”的主视觉冲突。
# d) Placement legality
#    - generator 先构造完整输出，再删去内侧对称边界得到输入，属于 output-first。
#    - 左/中/右三个组共享同一高度，但按公式分开设置轴列，使输出正好边贴边、输入至少隔 2 列，不重叠也不越界。
#    - 少量同色 decoy 只放在最外侧，距离本组至少 2 列，保证它们不会并入 marker component。
# e) Visual observability
#    - 解题者能从样例直接看出两件事：
#      1. 每个红竖线附近都不是“整块新复制”，而是已有双侧像素 + 少量缺口；
#      2. 缺口一补上，原本分散的三块会连成一个更大的整体。
#    - 这种“先局部对称补全，再全局拼接”的差异在 2-4 个样例里是稳定可见的。
# f) Visual cleanliness
#    - 主体只用 3 个局部组，且都排成一条清晰的横向结构；没有把多种标记类型堆在同一区域。
#    - decoy 只放两侧外缘、数量极少，既能阻止任务退化回原题，又不会盖过主体结构。
# g) Color inference
#    - 没有颜色映射。颜色只承担“把某个局部碎片与它自己的红轴对应起来”的作用。
#    - 左右组颜色不同，但空间布局一致，解题者仍会依赖几何关系而不是颜色数值本身。
# h) New assumptions introduced by the revised main()
#    - 新 main() 假设 row/col 组的有效 payload 全都包含在与红标记 4 连通的同一组件里；
#      generator 通过让 c=1 一列始终保留、所有缺口只发生在外边界来保证这一点。
#    - 新 main() 假设正确输出永远不需要覆盖已有非背景像素；generator 用唯一颜色、无重叠放置和 oracle 校验保证。
#    - 新 main() 假设 decoy 与 marker 组断开；generator 明确留出 2 列空隙。
# i) Generator-side guarantees
#    - 每个样例都至少有 2 个“输出中会变成拼合桥”的边界缺口，所以 D11 的聚合行为不会退化成互不相干的小组。
#    - 每个样例都含至少 2 个同色 decoy（左右各一个），使“只看当前 component 补全”与原题旧逻辑可区分。
#    - `_generate_valid_case()` 内部执行 `main(input) == intended_output` 的 oracle 校验，失败即重采样。

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask

MARK = 2
ALL_COLORS = list(range(10))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete local symmetries defined by red markers, but only fill missing pixels.

    Difference from the original 7ed72f31:
    - row/col/embedded-point groups reflect only the payload already inside that marker component
    - existing non-background pixels stay unchanged
    - multiple completed groups may aggregate into one larger figure
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != bg, connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def paint_if_missing(r: int, c: int, v: int) -> None:
        if inb(int(r), int(c)) and int(out[int(r), int(c)]) == bg:
            out[int(r), int(c)] = int(v)

    def refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, center)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    marker_pts: list[tuple[int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]

        if len(coords) == 1 and int(vals[0]) == MARK:
            marker_pts.append(coords[0])
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == MARK]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        payload_pixels = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != MARK]
        for r, c, v in payload_pixels:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))
            paint_if_missing(int(rr), int(cc), int(v))

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
                v = int(g[rr, cc])
                if v == bg or v == MARK:
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    paint_if_missing(int(rr), int(cc), int(v))

    return out


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _translate_cells(
    cells: list[tuple[int, int, int]],
    top: int,
    left: int,
    r0: int,
    c0: int,
) -> list[tuple[int, int, int]]:
    return [(int(top + r - r0), int(left + c - c0), int(v)) for r, c, v in cells]


def _rand_choice(rng: random.Random, seq: list[tuple[int, int]]) -> tuple[int, int]:
    return seq[rng.randrange(len(seq))]


def _grow_half_shape(
    rng: random.Random,
    line_len: int,
    side_width: int,
    boundary_rows: list[int],
) -> set[tuple[int, int]]:
    domain_rows = range(line_len)
    mandatory = {(int(r), int(c)) for r in boundary_rows for c in range(1, side_width + 1)}
    allowed = {
        (int(r), int(c))
        for r in domain_rows
        for c in range(1, side_width + 1)
        if c < side_width or r in boundary_rows
    }
    extra_capacity = len(allowed) - len(mandatory)
    if extra_capacity <= 0:
        return set(mandatory)
    max_extra = min(5, extra_capacity)
    target = len(mandatory) + rng.randint(1, max_extra)
    cells = set(mandatory)

    for _ in range(300):
        if len(cells) >= int(target):
            break
        base = _rand_choice(rng, list(cells))
        cand: list[tuple[int, int]] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = int(base[0] + dr)
            nc = int(base[1] + dc)
            if nr not in domain_rows or not (1 <= nc <= side_width):
                continue
            if nc == side_width and nr not in boundary_rows:
                continue
            if (nr, nc) not in cells:
                cand.append((nr, nc))
        if cand:
            cells.add(_rand_choice(rng, cand))

    if len(cells) < int(target):
        raise ValueError("failed to grow half-shape")
    return cells


def _build_col_group_from_half(
    rng: random.Random,
    half: set[tuple[int, int]],
    color: int,
    line_len: int,
    side_width: int,
    boundary_rows: list[int],
    role: str,
    center_singleton_row: int | None = None,
) -> dict[str, object]:
    output_payload: set[tuple[int, int]] = set()
    input_payload: set[tuple[int, int]] = set()

    for r, c in sorted(half):
        left = (int(r), -int(c))
        right = (int(r), int(c))
        output_payload.add(left)
        output_payload.add(right)

        if c < side_width:
            state = "both"
        elif role == "left":
            state = "left"
        elif role == "right":
            state = "right"
        else:
            if center_singleton_row is not None and int(r) == int(center_singleton_row):
                state = rng.choice(["left", "right"])
            else:
                state = "both"

        if state in {"both", "left"}:
            input_payload.add(left)
        if state in {"both", "right"}:
            input_payload.add(right)

    reds = {(int(r), 0) for r in range(line_len)}
    out_cells = [(r, c, color) for r, c in sorted(output_payload)] + [(r, c, MARK) for r, c in sorted(reds)]
    inp_cells = [(r, c, color) for r, c in sorted(input_payload)] + [(r, c, MARK) for r, c in sorted(reds)]
    bbox = _bbox((r, c) for r, c, _v in out_cells)
    return {
        "type": "col",
        "color": int(color),
        "side_width": int(side_width),
        "boundary_rows": [int(r) for r in boundary_rows],
        "out_cells": out_cells,
        "inp_cells": inp_cells,
        "out_payload": [(r, c, color) for r, c in sorted(output_payload)],
        "inp_payload": [(r, c, color) for r, c in sorted(input_payload)],
        "bbox": bbox,
    }


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int, int]], bg: int) -> None:
    for r, c, v in cells:
        if int(grid[r, c]) != int(bg):
            raise ValueError("overlap while painting")
        grid[r, c] = int(v)


def _count_components(mask: np.ndarray) -> int:
    _lab, n = label_mask(mask, connectivity=4)
    return int(n)


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    line_len = rng.randint(4, 6)
    left_width = rng.randint(2, 3)
    center_width = rng.randint(2, 3)
    right_width = int(left_width)

    mid = line_len // 2
    contact_rows = sorted({max(0, mid - 1), min(line_len - 1, mid)})
    extra_candidates = [r for r in range(line_len) if r not in contact_rows]
    if not extra_candidates:
        raise ValueError("no extra boundary row")
    extra_row = rng.choice(extra_candidates)
    boundary_rows = sorted(contact_rows + [int(extra_row)])

    H = rng.randint(max(18, line_len + 10), 24)
    bg = rng.choice([c for c in ALL_COLORS if c != MARK])

    colors = [c for c in ALL_COLORS if c not in (bg, MARK)]
    rng.shuffle(colors)
    left_color, center_color, right_color = colors[:3]

    left_half = _grow_half_shape(rng, line_len, left_width, boundary_rows)
    center_half = _grow_half_shape(rng, line_len, center_width, boundary_rows)
    right_half = set(left_half)

    left_group = _build_col_group_from_half(
        rng, left_half, int(left_color), line_len, left_width, boundary_rows, role="left"
    )
    center_group = _build_col_group_from_half(
        rng,
        center_half,
        int(center_color),
        line_len,
        center_width,
        boundary_rows,
        role="center",
        center_singleton_row=int(extra_row),
    )
    right_group = _build_col_group_from_half(
        rng, right_half, int(right_color), line_len, right_width, boundary_rows, role="right"
    )

    left_gap = rng.randint(2, 3)
    right_gap = int(left_gap)

    core_width = int(2 * center_width + 4 * left_width + 3)
    total_span = int(core_width + left_gap + right_gap)
    if total_span > 28:
        raise ValueError("layout too wide")
    W = rng.randint(max(24, total_span), 28)

    row_top = rng.randint(3, H - line_len - 3)
    left_decoy_col = rng.randint(0, W - total_span)
    left_min_col = int(left_decoy_col + left_gap)
    center_axis = int(left_min_col + 2 * left_width + center_width + 1)
    left_axis = int(center_axis - center_width - left_width - 1)
    right_axis = int(center_axis + center_width + right_width + 1)

    left_r0, left_c0, _left_r1, _left_c1 = map(int, left_group["bbox"])
    center_r0, center_c0, _center_r1, _center_c1 = map(int, center_group["bbox"])
    right_r0, right_c0, _right_r1, _right_c1 = map(int, right_group["bbox"])

    left_top = int(row_top - left_r0)
    center_top = int(row_top - center_r0)
    right_top = int(row_top - right_r0)

    left_left = int(left_axis + left_c0)
    center_left = int(center_axis + center_c0)
    right_left = int(right_axis + right_c0)

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)

    translated: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    for name, group, top, left in (
        ("left", left_group, left_top, left_left),
        ("center", center_group, center_top, center_left),
        ("right", right_group, right_top, right_left),
    ):
        r0, c0, _r1, _c1 = map(int, group["bbox"])
        out_cells = _translate_cells(group["out_cells"], top, left, r0, c0)
        inp_cells = _translate_cells(group["inp_cells"], top, left, r0, c0)
        out_payload = _translate_cells(group["out_payload"], top, left, r0, c0)
        inp_payload = _translate_cells(group["inp_payload"], top, left, r0, c0)
        translated[name] = {
            "out_cells": out_cells,
            "inp_cells": inp_cells,
            "out_payload": out_payload,
            "inp_payload": inp_payload,
        }
        _paint_cells(out, out_cells, bg)
        _paint_cells(inp, inp_cells, bg)

    left_min_col = min(c for _r, c, _v in translated["left"]["out_cells"])
    right_max_col = max(c for _r, c, _v in translated["right"]["out_cells"])
    decoy_rows = boundary_rows[:2]
    left_decoy_col = int(left_min_col - left_gap)
    right_decoy_col = int(right_max_col + right_gap)
    if left_decoy_col < 0 or right_decoy_col >= W:
        raise ValueError("decoy out of bounds")

    decoys = [
        (int(row_top + r), int(left_decoy_col), int(left_color)) for r in decoy_rows
    ] + [
        (int(row_top + r), int(right_decoy_col), int(right_color)) for r in decoy_rows
    ]
    _paint_cells(inp, decoys, bg)
    _paint_cells(out, decoys, bg)

    core_in_mask = np.zeros((H, W), dtype=bool)
    core_out_mask = np.zeros((H, W), dtype=bool)
    for name in ("left", "center", "right"):
        for r, c, _v in translated[name]["inp_cells"]:
            core_in_mask[int(r), int(c)] = True
        for r, c, _v in translated[name]["out_cells"]:
            core_out_mask[int(r), int(c)] = True

    meta = {
        "groups": 3,
        "core_input_components": int(_count_components(core_in_mask)),
        "core_output_components": int(_count_components(core_out_mask)),
        "missing_pixels": int(np.count_nonzero((inp == bg) & (out != bg))),
        "decoy_pixels": len(decoys),
        "line_len": int(line_len),
        "left_width": int(left_width),
        "center_width": int(center_width),
        "bridge_rows": len(boundary_rows),
    }
    if meta["core_input_components"] < 2:
        raise ValueError("core fragments already connected in input")
    if meta["core_output_components"] != 1:
        raise ValueError("completed fragments did not aggregate")
    if meta["missing_pixels"] <= 0:
        raise ValueError("no pixels to repair")
    return inp, out, meta


def _generate_valid_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    last_err: Exception | None = None
    for _ in range(300):
        try:
            inp, intended_out, meta = _generate_case(rng)
            actual_out = main(inp)
            if not np.array_equal(actual_out, intended_out):
                raise AssertionError("oracle mismatch")
            return inp, intended_out, meta
        except Exception as exc:  # pragma: no cover - stochastic retry path
            last_err = exc
    raise RuntimeError(f"failed to generate valid combo case: {last_err}")


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D11/2 + D13/1 combo variant.

    The returned input contains three locally symmetric fragments with missing inner-edge pixels.
    After local completion, they join into one larger aggregate figure.
    """
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    try:
        inp, _out, _meta = _generate_valid_case(rng)
        return inp
    except Exception as exc:
        raise RuntimeError(f"failed to generate combo sample (seed={seed}): {exc}") from exc
