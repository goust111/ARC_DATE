# Combo variant of 7ed72f31 (index 1)
# combo_spec: D1/0+D4/1
# Name: 旋转角度编码

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location


MARK = 2
ROT_BY_RED_COUNT = {1: 1, 2: 2, 3: 3}  # clockwise quarter-turns


"""
design_analysis:
  variant_core:
    original:
      - 红标记几何形状直接决定反射类型：单红点做点反射，水平/竖直红线做轴反射。
    fused:
      - 红像素数量 1/2/3 不再决定“点/轴”类别，而是直接编码旋转角度 90/180/270 度。
      - 红标记的几何形状只负责给出旋转中心：
        * 1 个红像素：中心就是该红点。
        * 2 或 3 个红像素：要求它们构成连续的水平或竖直短线，中心是该线段中点。
      - payload 仍保留在原位，输出是在背景上叠加旋转后的同色副本。
  concrete_trace:
    example_3_reds:
      input_local:
        reds: [(0,0), (0,1), (0,2)]
        payload: [(-1,1), (-2,1), (-2,2)]
      original_solver:
        - 因为红点共行，会把 payload 对 row=0 做镜像，得到 [(1,1), (2,1), (2,2)]。
      D1_rotation_only:
        - 若只做 D1/0，则同样的 3 红线会绕中点 (0,1) 顺时针 90 度，副本跑到右侧。
      fused_solver:
        - 3 个红像素编码 270 度，仍绕同一中点旋转，但副本跑到左侧。
      difference:
        - 解题者必须先从“红点数=3”推断 270 度，再执行旋转；几何形状不再提供变换类型。
    example_2_reds:
      input_local:
        reds: [(0,0), (0,1)]
        payload: [(-1,0), (-2,0), (-2,1)]
      original_solver:
        - 视作水平反射轴，输出在下方。
      fused_solver:
        - 2 个红像素编码 180 度，绕线段中点 (0,0.5) 旋转，输出会落到下方但带有水平换侧，
          与简单镜像不同。
  complexity_calibration:
    - 角度档位仅 1/2/3 三种，足够少，能在 2-4 个样例中建立“数量 -> 角度”映射。
    - 每个样例至少包含一次 1 红、2 红、3 红，并随机重复其中若干档位，提供重复支撑。
    - 2 红与 3 红 group 都会同时出现横线和竖线版本，避免“方向本身决定角度”的误解。
    - payload 强制非对称、bbox 至少 2x2，并额外过滤掉“按新规则旋转”和“按原规则反射”
      输出相同的局部形状，防止视觉退化。
  implicit_assumption_audit:
    original_main_assumptions_broken:
      - `len(reds)==1 / len(rs)==1 / len(cs)==1` 原本直接选择点反射/行反射/列反射；
        融合后它们不能再决定变换类型，只能用来恢复旋转中心。
      - `refl_row/refl_col/refl_point` 不再适用，必须改成通用的“绕整数或半整数中心旋转”。
      - 原 solver 通过 bbox 和同色扫描整张图决定哪些像素参与；对旋转版来说这会把附近无关同色像素拖进去，
        因此新 solver 只旋转 marker 所在 4 连通组件内部的非红 payload。
      - 原任务的“孤立红点 + 相邻 8 连通精灵”分支与本融合不兼容：一旦允许 2/3 个红像素通过数量编码角度，
        就还得额外定义多红孤立标记如何选中心和关联对象，会把主要线索拆散。
        因此本变体只生成 mixed component，并让 solver 忽略 red-only component。
    new_main_assumptions:
      - 可操作组件必须含有 1/2/3 个红像素和至少 1 个非红 payload 像素。
      - 若红像素数大于 1，则红像素必须构成连续的单行或单列线段；generator 严格保证这一点。
      - 2 红只编码 180 度，所以即使中心是半整数也不会产生半格输出；1 红和 3 红允许 90/270，
        此时中心为整数点，也由 generator 保证。
  placement_legality:
    - 仍采用 output-first：先构造完整 output patch = 原输入局部 + 旋转副本，再把整个 patch 放到大画布。
    - 放置使用 `random_free_location(..., border=1, padding=1)`，确保不同 group 之间不接触、不重叠。
    - 局部构造时先检查 rotated copy 与原 payload、红标记完全不重叠，且所有格点都保持整数。
  observability:
    - 1 红 group 是点旋转，2/3 红 group 是短线中点旋转；肉眼能同时看到“红点数量差异”和“复制方向差异”。
    - 同一数量会在不同朝向下重复出现，解题者能排除“横竖决定变换”的旧规则。
  cleanliness:
    - 不生成自由散落的同色干扰点。
    - 每个 group 用不同 payload 颜色，局部结构单一，便于把颜色和红码绑定。
  color_inferability:
    - 颜色不参与规则；颜色只区分 group。真正控制变换的是红色像素数量。
"""


def _rot_origin(coord: tuple[int, int], quarters: int) -> tuple[int, int]:
    r, c = map(int, coord)
    for _ in range(int(quarters) % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_about_center2(
    center2: tuple[int, int],
    coord: tuple[int, int],
    quarters: int,
) -> tuple[int, int] | None:
    cr2, cc2 = map(int, center2)
    r, c = map(int, coord)
    dr2, dc2 = 2 * int(r) - cr2, 2 * int(c) - cc2
    for _ in range(int(quarters) % 4):
        dr2, dc2 = dc2, -dr2
    rr2, cc2_new = cr2 + dr2, cc2 + dc2
    if rr2 % 2 != 0 or cc2_new % 2 != 0:
        return None
    return rr2 // 2, cc2_new // 2


def _center2_from_reds(reds: set[tuple[int, int]]) -> tuple[int, int] | None:
    if len(reds) == 1:
        r, c = next(iter(reds))
        return 2 * int(r), 2 * int(c)

    rs = {int(r) for r, _c in reds}
    cs = {int(c) for _r, c in reds}
    if len(rs) == 1:
        row = next(iter(rs))
        cols = sorted(int(c) for _r, c in reds)
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        return 2 * int(row), int(cols[0] + cols[-1])
    if len(cs) == 1:
        col = next(iter(cs))
        rows = sorted(int(r) for r, _c in reds)
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return int(rows[0] + rows[-1]), 2 * int(col)
    return None


def _original_transform_cells(
    reds: set[tuple[int, int]],
    payload: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if len(reds) == 1:
        pr, pc = next(iter(reds))
        return {(2 * int(pr) - int(r), 2 * int(pc) - int(c)) for r, c in payload}

    rs = {int(r) for r, _c in reds}
    cs = {int(c) for _r, c in reds}
    if len(rs) == 1:
        row = next(iter(rs))
        return {(2 * int(row) - int(r), int(c)) for r, c in payload}
    if len(cs) == 1:
        col = next(iter(cs))
        return {(int(r), 2 * int(col) - int(c)) for r, c in payload}
    return None


def _shape_array(cells: set[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    h = max(rs) - r0 + 1
    w = max(cs) - c0 + 1
    arr = np.zeros((h, w), dtype=int)
    for r, c in cells:
        arr[int(r - r0), int(c - c0)] = 1
    return arr


def _is_interesting_shape(cells: set[tuple[int, int]]) -> bool:
    if len(cells) < 4:
        return False
    arr = _shape_array(cells)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return False
    variants = [
        np.flipud(arr),
        np.fliplr(arr),
        np.rot90(arr, 1),
        np.rot90(arr, 2),
        np.rot90(arr, 3),
    ]
    for var in variants:
        if arr.shape == var.shape and np.array_equal(arr, var):
            return False
    return True


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _grow_connected(
    rng: random.Random,
    allowed: set[tuple[int, int]],
    seed: tuple[int, int],
    target_size: int,
) -> set[tuple[int, int]] | None:
    cells = {tuple(seed)}
    while len(cells) < int(target_size):
        frontier: set[tuple[int, int]] = set()
        for cell in cells:
            for nxt in _neighbors4(cell):
                if nxt in allowed and nxt not in cells:
                    frontier.add(tuple(nxt))
        if not frontier:
            return None
        cells.add(rng.choice(tuple(frontier)))
    return cells


def _marker_shape(reds: set[tuple[int, int]]) -> str:
    if len(reds) == 1:
        return "point"
    rs = {int(r) for r, _c in reds}
    return "row" if len(rs) == 1 else "col"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    lab4, n4 = label_mask(g != bg, connectivity=4)
    for lbl in range(1, int(n4) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == int(lbl))]
        if not coords:
            continue

        reds = {(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == MARK}
        if len(reds) not in ROT_BY_RED_COUNT:
            continue

        center2 = _center2_from_reds(reds)
        if center2 is None:
            continue

        payload = [
            (int(r), int(c), int(g[int(r), int(c)]))
            for r, c in coords
            if int(g[int(r), int(c)]) != MARK
        ]
        if not payload:
            continue

        quarters = ROT_BY_RED_COUNT[len(reds)]
        for r, c, v in payload:
            dst = _rotate_about_center2(center2, (int(r), int(c)), quarters)
            if dst is None:
                continue
            rr, cc = dst
            if inb(rr, cc):
                out[int(rr), int(cc)] = int(v)

    return out


def _render_cells(
    cells: dict[tuple[int, int], int],
    bg: int,
    bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    coords = list(cells)
    if bounds is None:
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        r0, c0 = min(rs), min(cs)
        r1, c1 = max(rs), max(cs)
    else:
        r0, c0, r1, c1 = map(int, bounds)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for (r, c), v in cells.items():
        out[int(r - r0), int(c - c0)] = int(v)
    return out


def _build_point_payload(rng: random.Random) -> set[tuple[int, int]] | None:
    allowed = {(r, c) for r in range(-5, 0) for c in range(-2, 3)}
    for _ in range(200):
        size = rng.randint(4, 7)
        payload = _grow_connected(rng, allowed, (-1, 0), size)
        if payload is None or not _is_interesting_shape(payload):
            continue
        return payload
    return None


def _build_line_payload(rng: random.Random, red_count: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    line = {(0, c) for c in range(int(red_count))}
    for _ in range(300):
        size = rng.randint(4, 7)
        side = rng.choice(["above", "below"])
        if side == "above":
            allowed = {(r, c) for r in range(-5, 0) for c in range(-2, int(red_count) + 2)}
            seed = (-1, rng.randrange(int(red_count)))
        else:
            allowed = {(r, c) for r in range(1, 6) for c in range(-2, int(red_count) + 2)}
            seed = (1, rng.randrange(int(red_count)))
        payload = _grow_connected(rng, allowed, seed, size)
        if payload is None or not _is_interesting_shape(payload):
            continue
        return payload, line
    return None


def _build_group(
    rng: random.Random,
    bg: int,
    color: int,
    red_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    for _ in range(500):
        if int(red_count) == 1:
            payload = _build_point_payload(rng)
            if payload is None:
                continue
            reds = {(0, 0)}
        else:
            built = _build_line_payload(rng, int(red_count))
            if built is None:
                continue
            payload, reds = built

        base_rot = rng.randrange(4)
        payload = {_rot_origin(cell, base_rot) for cell in payload}
        reds = {_rot_origin(cell, base_rot) for cell in reds}

        center2 = _center2_from_reds(reds)
        if center2 is None:
            continue

        quarters = ROT_BY_RED_COUNT[int(red_count)]
        rotated: set[tuple[int, int]] = set()
        valid = True
        for cell in payload:
            dst = _rotate_about_center2(center2, cell, quarters)
            if dst is None:
                valid = False
                break
            rotated.add(dst)
        if not valid:
            continue
        if rotated & payload or rotated & reds:
            continue

        original_extra = _original_transform_cells(reds, payload)
        if original_extra is None or original_extra == rotated:
            continue
        diff_from_original = len(rotated.symmetric_difference(original_extra))
        if diff_from_original < 2:
            continue

        input_cells: dict[tuple[int, int], int] = {tuple(cell): int(color) for cell in payload}
        for cell in reds:
            input_cells[tuple(cell)] = MARK

        output_cells = dict(input_cells)
        for cell in rotated:
            output_cells[tuple(cell)] = int(color)

        all_coords = list(output_cells)
        rs = [r for r, _c in all_coords]
        cs = [c for _r, c in all_coords]
        bounds = (min(rs), min(cs), max(rs), max(cs))
        input_patch = _render_cells(input_cells, bg, bounds=bounds)
        output_patch = _render_cells(output_cells, bg, bounds=bounds)

        meta: dict[str, int | str] = {
            "red_count": int(red_count),
            "angle_deg": int(90 * quarters),
            "base_rot": int(base_rot),
            "marker_shape": _marker_shape(reds),
            "original_diff": int(diff_from_original),
        }
        return input_patch, output_patch, meta

    raise RuntimeError(f"failed to build group for red_count={int(red_count)}")


def generate_case(
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    rng = rng or random.Random()
    palette = list(range(10))

    for _attempt in range(300):
        H = rng.randint(18, 28)
        W = rng.randint(18, 28)
        bg = rng.choice([c for c in palette if c != MARK])
        input_grid = np.full((H, W), int(bg), dtype=int)
        output_grid = np.full((H, W), int(bg), dtype=int)

        n_groups = rng.randint(4, 6)
        counts = [1, 2, 3]
        while len(counts) < n_groups:
            counts.append(rng.choice([1, 2, 3]))
        rng.shuffle(counts)

        usable_colors = [c for c in palette if c not in (bg, MARK)]
        if len(usable_colors) < n_groups:
            continue
        colors = rng.sample(usable_colors, n_groups)

        meta: list[dict[str, int | str]] = []
        ok = True
        for color, red_count in zip(colors, counts):
            try:
                input_patch, output_patch, group_meta = _build_group(rng, bg, int(color), int(red_count))
                r0, c0 = random_free_location(
                    output_grid,
                    output_patch,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except (RuntimeError, ValueError):
                ok = False
                break

            out_mask = output_patch != int(bg)
            in_mask = input_patch != int(bg)
            oh, ow = output_patch.shape
            ih, iw = input_patch.shape
            output_grid[r0 : r0 + oh, c0 : c0 + ow][out_mask] = output_patch[out_mask]
            input_grid[r0 : r0 + ih, c0 : c0 + iw][in_mask] = input_patch[in_mask]
            meta.append(group_meta)

        if not ok:
            continue

        solved = main(input_grid)
        if not np.array_equal(solved, output_grid):
            continue

        counts_seen = {int(m["red_count"]) for m in meta}
        shapes_seen = {str(m["marker_shape"]) for m in meta if str(m["marker_shape"]) != "point"}
        if counts_seen != {1, 2, 3}:
            continue
        if shapes_seen != {"row", "col"}:
            continue
        return input_grid, output_grid, meta

    raise RuntimeError("failed to generate a valid combo-variant instance")


def generate_input() -> np.ndarray:
    grid, expected, _meta = generate_case()
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise AssertionError("generator self-check failed")
    return grid
