# Combo variant of 7ed72f31 (index 26)
# combo_spec: D1/0+D5/2+D9/0
# Name: 旋转碰撞清除

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


"""
design_analysis:
  fused_rule:
    original_task: 红色标记决定反射方式，原 payload 保留，镜像像素直接覆盖写入。
    combo_rule:
      - 水平/竖直红线不再做轴反射，而是把其连通 payload 绕红线中点顺时针旋转 90 度。
      - 单红点仍做 180 度旋转；孤立红点仍驱动其唯一相邻 8 连通组件做 180 度旋转。
      - 所有被移动的原 payload 都先被擦除，只保留红色标记。
      - 旋转后的目标像素若落到“擦源之后仍存在的非背景像素”上，则双方都变为背景。
      - 若多个旋转像素同时打到同一空格，也统一湮灭为背景；generator 避免这种歧义，但 solver 保持稳健。
  concrete_trace:
    horizontal_example:
      input:
        - 红线长度 3，中心在 (6,6)，payload 颜色 4，像素在 (5,5),(5,6),(4,6)
        - 另外在旋转目标 (6,7) 预放一个孤立 blocker 像素颜色 8
      original:
        - 反射到红线另一侧，源 payload 仍保留，blocker 不参与，输出里两侧都能看到颜色 4
      fused:
        - 先把源 payload 擦掉
        - 再绕红线中点顺时针旋转，目标本应是 (5,7),(6,7),(6,8)
        - 其中 (6,7) 与 blocker 碰撞，二者都删除；其余两个目标保留
      visible_difference:
        - 输出只剩旋转后的残缺副本；原侧消失；碰撞位置形成空洞
    vertical_example:
      input: payload 在红线左侧
      fused: 90 度顺时针后转到红线上方，而不是镜像到右侧
    isolated_point_example:
      input: 单红点与一个对角相邻的 8 连通小精灵
      fused: 该小精灵绕红点 180 度移动到对侧，原位置被清空；若目标已有 blocker，则该 blocker 与目标同时消失
  complexity_calibration:
    - 旋转方向固定为顺时针 90 度，避免同题里再引入多种角度。
    - 红线只生成奇数长度 3 或 5；否则 90 度格点旋转会落在半格。
    - 每个样例至少包含 row / col / isolated 三类 group，其中 row+col 至少两组负责承载碰撞。
    - payload 强制是不规则连通块，避免反射与旋转在视觉上近似。
    - collision blocker 使用单独颜色，且只放在旋转目标位置上，保证“湮灭”是主现象而不是杂乱噪声。
  implicit_assumption_audit:
    original_main_risk_1:
      code_pattern: 按整张网格扫描同色像素再用 bbox 判 eligible
      broken_by_combo: D9 删除语义下，这会把无关同色像素错误擦掉
      fix: 线类/嵌入式单点只移动 marker 所在 4 连通组件内部的 payload
    original_main_risk_2:
      code_pattern: red row / col 只需要一条轴坐标
      broken_by_combo: D1 的 90 度旋转需要真实中心
      fix: 只接受连续奇数长度红线，并用 doubled-center 表示旋转中心
    original_main_risk_3:
      code_pattern: out = g.copy(); 边遍历边直接写镜像
      broken_by_combo: D5+D9 需要“先擦源，再结算碰撞”，顺序错了会把本应消失的碰撞误判为保留
      fix: 分成 collect_sources -> erase -> resolve_hits 三阶段
    original_main_risk_4:
      code_pattern: 目标位已有非背景像素时直接覆盖
      broken_by_combo: D5 明确要求互相湮灭
      fix: resolve 时若目标已有非背景，或有多个 incoming，则统一置背景
    isolated_case_risk:
      issue: 孤立红点分支会把 8 邻域触发到的整色组件一起移动
      generator_support: blocker 绝不放到孤立红点 8 邻域，且 blocker 颜色不复用为 payload 颜色
  placement_legality:
    - generator 先生成局部模板：input_cells = 红标记 + 源 payload，moved_cells = 旋转后的目标 footprint。
    - 大网格放置时按 input_cells ∪ moved_cells 的并集占位，并额外留 1 格 moat，保证不同 group 在输入和输出里都不相撞。
    - blocker 只从 moved_cells 中选择，同时要求该格与所有输入非背景像素 4 不相邻，从而不会在输入里并入 marker 组件。
  visual_observability:
    - row/col 组保证“跨轴反射”与“正交旋转”差异肉眼可见。
    - 至少 3 个 collision 像素，且覆盖至少两个线类 group；输出里会看到明显缺口。
    - 输出中原侧被清空，因此整体观感是 sprite 被旋转搬走，而不是复制。
  cleanliness:
    - 不放原题 spec 中那种近 bbox 同色散点，避免 D9 下误导“被删除的是谁”。
    - group 使用唯一 payload 颜色；blocker 使用单独颜色；组间有 moat，局部结构清楚。
  behavior_metrics_used_for_validation:
    - moved_source_pixels: 输入中 payload 非背景而输出变背景的像素数，必须显著大于 0
    - collision_pixels: blocker 像素在输出中变背景的数量，必须每样例至少 3
    - rotation_delta: 本融合 solver 与“同样删除+碰撞但线类仍做反射”的对比差异，必须在绝大多数样例中 > 0
  new_solver_assumptions_checked_against_generator:
    - 红线连续且奇数长度；generator 只生成 3/5 长度连续线段。
    - marker-bearing mixed component 的非红像素就是唯一要移动的 payload；generator 通过 moat 与唯一颜色保证。
    - 孤立红点只触发一个相邻 8 连通 payload 组件；generator 把它放在单一对角象限并远离 blocker。
    - 多个旋转副本不会打到同一空格；generator 通过 full-footprint placement 保证，solver 仍做防御式湮灭。
"""


MARK = 2
Coord = tuple[int, int]
Axis = tuple[str, tuple[int, int]]


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _rot180(center: Coord, r: int, c: int) -> Coord:
    cr, cc = map(int, center)
    return int(2 * cr - r), int(2 * cc - c)


def _refl_row(r0: int, r: int, c: int) -> Coord:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> Coord:
    return int(r), int(2 * int(c0) - int(c))


def _rot90_cw_center2(center2: tuple[int, int], r: int, c: int) -> Coord | None:
    cr2, cc2 = map(int, center2)
    rr2 = int(cr2 + 2 * int(c) - cc2)
    cc2_new = int(cc2 - 2 * int(r) + cr2)
    if rr2 % 2 != 0 or cc2_new % 2 != 0:
        return None
    return int(rr2 // 2), int(cc2_new // 2)


def _build_components(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})
    return comps


def _classify_marker(reds: list[Coord]) -> Axis | None:
    if len(reds) == 1:
        return ("point", (int(reds[0][0]), int(reds[0][1])))

    rows = {int(r) for r, _c in reds}
    cols = {int(c) for _r, c in reds}
    if len(rows) == 1:
        row = int(next(iter(rows)))
        red_cols = sorted(int(c) for _r, c in reds)
        if red_cols == list(range(red_cols[0], red_cols[-1] + 1)) and len(red_cols) % 2 == 1:
            return ("row", (2 * int(row), int(red_cols[0] + red_cols[-1])))
    if len(cols) == 1:
        col = int(next(iter(cols)))
        red_rows = sorted(int(r) for r, _c in reds)
        if red_rows == list(range(red_rows[0], red_rows[-1] + 1)) and len(red_rows) % 2 == 1:
            return ("col", (int(red_rows[0] + red_rows[-1]), 2 * int(col)))
    return None


def _transform_src(axis: Axis, r: int, c: int, *, rotate_lines: bool) -> Coord | None:
    kind, data = axis
    if kind == "point":
        return _rot180(data, int(r), int(c))
    if rotate_lines:
        return _rot90_cw_center2(data, int(r), int(c))
    if kind == "row":
        return _refl_row(int(data[0] // 2), int(r), int(c))
    return _refl_col(int(data[1] // 2), int(r), int(c))


def _collect_moves(g: np.ndarray, bg: int, *, rotate_lines: bool) -> tuple[set[Coord], dict[Coord, list[int]]]:
    h, w = g.shape
    comps = _build_components(g, int(bg))
    erase: set[Coord] = set()
    hits: dict[Coord, list[int]] = {}

    def emit(rr: int, cc: int, v: int) -> None:
        if _inb((h, w), int(rr), int(cc)):
            hits.setdefault((int(rr), int(cc)), []).append(int(v))

    # Mixed components: only move payload inside the same marker-bearing 4-cc.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue
        if len(reds) == 1 and len(coords) == 1:
            continue  # isolated red marker is handled below

        axis = _classify_marker(reds)
        if axis is None:
            continue

        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not payload:
            continue
        for r, c, v in payload:
            erase.add((int(r), int(c)))
            dst = _transform_src(axis, int(r), int(c), rotate_lines=rotate_lines)
            if dst is not None:
                emit(int(dst[0]), int(dst[1]), int(v))

    # Isolated red points: move the unique adjacent 8-connected color component.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if len(coords) != 1 or int(vals[0]) != int(MARK):
            continue
        pr, pc = map(int, coords[0])

        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb((h, w), int(rr), int(cc)):
                    continue
                v = int(g[int(rr), int(cc)])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for color in neigh_colors:
            labv, _ = label_mask(g == int(color), connectivity=8)
            labels = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(color) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labels:
                for r, c in np.argwhere(labv == int(lbl)):
                    erase.add((int(r), int(c)))
                    dst = _rot180((int(pr), int(pc)), int(r), int(c))
                    emit(int(dst[0]), int(dst[1]), int(color))
    return erase, hits


def _resolve_moves(g: np.ndarray, bg: int, erase: set[Coord], hits: dict[Coord, list[int]]) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    for r, c in erase:
        if int(out[int(r), int(c)]) != int(MARK):
            out[int(r), int(c)] = int(bg)

    for (r, c), vals in hits.items():
        if not vals:
            continue
        if int(out[int(r), int(c)]) != int(bg) or len(vals) != 1:
            out[int(r), int(c)] = int(bg)
        else:
            out[int(r), int(c)] = int(vals[0])
    return out


def _simulate(grid: np.ndarray, *, rotate_lines: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    erase, hits = _collect_moves(g, int(bg), rotate_lines=bool(rotate_lines))
    return _resolve_moves(g, int(bg), erase, hits)


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, rotate_lines=True)


def simulate_reflect_move(grid: np.ndarray) -> np.ndarray:
    """
    Validation helper: same move+collision semantics as the combo variant,
    but line markers still use reflection instead of rotation.
    """
    return _simulate(grid, rotate_lines=False)


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = map(int, cell)
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _neighbors8(cell: Coord) -> list[Coord]:
    r, c = map(int, cell)
    return [
        (int(r + dr), int(c + dc))
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _shape_array(cells: set[Coord]) -> np.ndarray:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    r0, c0 = int(min(rs)), int(min(cs))
    h = int(max(rs) - r0 + 1)
    w = int(max(cs) - c0 + 1)
    arr = np.zeros((h, w), dtype=int)
    for r, c in cells:
        arr[int(r - r0), int(c - c0)] = 1
    return arr


def _is_interesting_line_shape(cells: set[Coord]) -> bool:
    if len(cells) < 4:
        return False
    arr = _shape_array(cells)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return False
    variants = [np.flipud(arr), np.fliplr(arr), np.rot90(arr, 1), np.rot90(arr, 2), np.rot90(arr, 3)]
    for var in variants:
        if arr.shape == var.shape and np.array_equal(arr, var):
            return False
    return True


def _is_interesting_iso_shape(cells: set[Coord]) -> bool:
    if len(cells) < 3:
        return False
    arr = _shape_array(cells)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return False
    return not np.array_equal(arr, np.rot90(arr, 2))


def _grow_connected(
    allowed: set[Coord],
    seed: Coord,
    target_size: int,
    *,
    connectivity: int,
    rng: random.Random,
    tries: int = 240,
) -> set[Coord] | None:
    allowed = {(int(r), int(c)) for r, c in allowed}
    seed = (int(seed[0]), int(seed[1]))
    if seed not in allowed:
        return None
    neighbor_fn = _neighbors4 if int(connectivity) == 4 else _neighbors8
    for _ in range(int(tries)):
        cells = {seed}
        while len(cells) < int(target_size):
            frontier = set()
            for cell in cells:
                for nxt in neighbor_fn(cell):
                    if nxt in allowed and nxt not in cells:
                        frontier.add((int(nxt[0]), int(nxt[1])))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) == int(target_size):
            return cells
    return None


def _rot90_cw_rel(cells: set[Coord]) -> set[Coord]:
    return {(int(c), int(-r)) for r, c in cells}


def _rot180_rel(cells: set[Coord]) -> set[Coord]:
    return {(int(-r), int(-c)) for r, c in cells}


def _safe_blocker_candidates(moved_cells: dict[Coord, int], input_cells: dict[Coord, int]) -> list[Coord]:
    input_pts = {(int(r), int(c)) for r, c in input_cells}
    safe: list[Coord] = []
    for r, c in moved_cells:
        if (int(r), int(c)) in input_pts:
            continue
        if all((int(rr), int(cc)) not in input_pts for rr, cc in _neighbors4((int(r), int(c)))):
            safe.append((int(r), int(c)))
    return safe


def _normalize_group(
    input_cells: dict[Coord, int],
    moved_cells: dict[Coord, int],
    *,
    kind: str,
    color: int,
) -> dict[str, object]:
    full = dict(input_cells)
    full.update(moved_cells)
    pts = list(full)
    min_r = min(int(r) for r, _c in pts)
    min_c = min(int(c) for _r, c in pts)

    norm_in = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in input_cells.items()}
    norm_moved = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in moved_cells.items()}
    norm_full = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in full.items()}
    h = 1 + max(int(r) for r, _c in norm_full)
    w = 1 + max(int(c) for _r, c in norm_full)
    return {
        "kind": str(kind),
        "color": int(color),
        "input": norm_in,
        "moved": norm_moved,
        "full": norm_full,
        "safe_blockers": _safe_blocker_candidates(norm_moved, norm_in),
        "h": int(h),
        "w": int(w),
    }


def _build_line_group(kind: str, color: int, rng: random.Random) -> dict[str, object] | None:
    for _ in range(240):
        line_len = rng.choice([3, 5])
        half = int(line_len // 2)
        size = rng.randint(4, 7)
        if kind == "row":
            side = rng.choice(["above", "below"])
            if side == "above":
                allowed = {(int(r), int(c)) for r in range(-4, 0) for c in range(-half - 2, half + 3)}
                seed = (-1, rng.randint(-half, half))
            else:
                allowed = {(int(r), int(c)) for r in range(1, 5) for c in range(-half - 2, half + 3)}
                seed = (1, rng.randint(-half, half))
            reds = {(0, int(c)) for c in range(-half, half + 1)}
        else:
            side = rng.choice(["left", "right"])
            if side == "left":
                allowed = {(int(r), int(c)) for r in range(-half - 2, half + 3) for c in range(-4, 0)}
                seed = (rng.randint(-half, half), -1)
            else:
                allowed = {(int(r), int(c)) for r in range(-half - 2, half + 3) for c in range(1, 5)}
                seed = (rng.randint(-half, half), 1)
            reds = {(int(r), 0) for r in range(-half, half + 1)}

        payload = _grow_connected(allowed, seed, size, connectivity=4, rng=rng)
        if payload is None or not _is_interesting_line_shape(payload):
            continue

        moved = _rot90_cw_rel(payload)
        if payload & reds or moved & reds or moved & payload:
            continue

        input_cells = {(int(r), int(c)): int(MARK) for r, c in reds}
        for r, c in payload:
            input_cells[(int(r), int(c))] = int(color)
        moved_cells = {(int(r), int(c)): int(color) for r, c in moved}
        group = _normalize_group(input_cells, moved_cells, kind=str(kind), color=int(color))
        if len(group["safe_blockers"]) >= 2:
            return group
    return None


def _build_isolated_group(color: int, rng: random.Random) -> dict[str, object] | None:
    quadrants = {
        "tl": ({(int(r), int(c)) for r in range(-4, 0) for c in range(-4, 0)}, (-1, -1)),
        "tr": ({(int(r), int(c)) for r in range(-4, 0) for c in range(1, 5)}, (-1, 1)),
        "bl": ({(int(r), int(c)) for r in range(1, 5) for c in range(-4, 0)}, (1, -1)),
        "br": ({(int(r), int(c)) for r in range(1, 5) for c in range(1, 5)}, (1, 1)),
    }
    for _ in range(240):
        allowed, seed = rng.choice(list(quadrants.values()))
        size = rng.randint(3, 6)
        payload = _grow_connected(allowed, seed, size, connectivity=8, rng=rng)
        if payload is None or not _is_interesting_iso_shape(payload):
            continue
        moved = _rot180_rel(payload)
        if (0, 0) in payload or (0, 0) in moved or payload & moved:
            continue

        input_cells = {(0, 0): int(MARK)}
        for r, c in payload:
            input_cells[(int(r), int(c))] = int(color)
        moved_cells = {(int(r), int(c)): int(color) for r, c in moved}
        return _normalize_group(input_cells, moved_cells, kind="isolated", color=int(color))
    return None


def _build_group(kind: str, color: int, rng: random.Random) -> dict[str, object]:
    if kind in {"row", "col"}:
        group = _build_line_group(str(kind), int(color), rng)
    else:
        group = _build_isolated_group(int(color), rng)
    if group is None:
        raise RuntimeError(f"failed to build group: {kind}")
    return group


def _can_place(group: dict[str, object], top: int, left: int, h: int, w: int, forbidden: set[Coord]) -> bool:
    for r, c in group["full"]:
        rr = int(top + r)
        cc = int(left + c)
        if not _inb((h, w), int(rr), int(cc)):
            return False
        if (int(rr), int(cc)) in forbidden:
            return False
    return True


def _stamp_moat(forbidden: set[Coord], cells: list[Coord]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((int(r + dr), int(c + dc)))


def _is_blocker_safe(inp: np.ndarray, pos: Coord, chosen: set[Coord], bg: int) -> bool:
    r, c = map(int, pos)
    if int(inp[int(r), int(c)]) != int(bg):
        return False
    for rr, cc in _neighbors4((int(r), int(c))):
        if _inb(inp.shape, int(rr), int(cc)):
            if int(inp[int(rr), int(cc)]) != int(bg):
                return False
            if (int(rr), int(cc)) in chosen:
                return False
    return True


def _choose_blockers(
    inp: np.ndarray,
    placed: list[dict[str, object]],
    blocker_color: int,
    bg: int,
    rng: random.Random,
) -> list[Coord]:
    carriers = [p for p in placed if p["kind"] in {"row", "col"} and p["safe_abs"]]
    if len(carriers) < 2:
        return []

    rng.shuffle(carriers)
    chosen_groups = carriers[: rng.randint(2, min(3, len(carriers)))]
    chosen: set[Coord] = set()

    for group in chosen_groups:
        cands = [tuple(map(int, pos)) for pos in group["safe_abs"] if _is_blocker_safe(inp, tuple(map(int, pos)), chosen, int(bg))]
        if not cands:
            return []
        chosen.add(tuple(map(int, rng.choice(cands))))

    target_total = rng.randint(max(3, len(chosen)), min(6, sum(len(p["safe_abs"]) for p in carriers)))
    extra_cands = []
    for group in carriers:
        for pos in group["safe_abs"]:
            extra_cands.append(tuple(map(int, pos)))
    rng.shuffle(extra_cands)
    for pos in extra_cands:
        if len(chosen) >= int(target_total):
            break
        if pos in chosen:
            continue
        if _is_blocker_safe(inp, pos, chosen, int(bg)):
            chosen.add((int(pos[0]), int(pos[1])))

    for r, c in chosen:
        inp[int(r), int(c)] = int(blocker_color)
    return sorted(chosen)


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = rng or random.Random()

    for _ in range(600):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        bg = rng.choice([c for c in range(10) if c != int(MARK)])

        usable = [c for c in range(10) if c not in (int(bg), int(MARK))]
        rng.shuffle(usable)
        n_groups = rng.randint(3, min(5, len(usable) - 1))
        group_colors = usable[:n_groups]
        blocker_color = int(usable[n_groups])

        kinds = ["row", "col", "isolated"]
        extras = ["row", "col", "isolated", "row", "col"]
        while len(kinds) < n_groups:
            kinds.append(rng.choice(extras))
        rng.shuffle(kinds)

        try:
            groups = [_build_group(str(kind), int(color), rng) for kind, color in zip(kinds, group_colors)]
        except RuntimeError:
            continue

        groups.sort(key=lambda g: len(g["full"]), reverse=True)
        inp = np.full((h, w), int(bg), dtype=int)
        expected = np.full((h, w), int(bg), dtype=int)
        forbidden: set[Coord] = set()
        placed: list[dict[str, object]] = []
        ok = True

        for group in groups:
            gh = int(group["h"])
            gw = int(group["w"])
            placed_here = False
            for _attempt in range(700):
                top = rng.randint(0, h - gh)
                left = rng.randint(0, w - gw)
                if not _can_place(group, int(top), int(left), int(h), int(w), forbidden):
                    continue

                full_abs: list[Coord] = []
                input_abs: dict[Coord, int] = {}
                moved_abs: dict[Coord, int] = {}
                red_abs: list[Coord] = []

                for (r, c), v in group["input"].items():
                    rr = int(top + r)
                    cc = int(left + c)
                    inp[int(rr), int(cc)] = int(v)
                    input_abs[(int(rr), int(cc))] = int(v)
                    full_abs.append((int(rr), int(cc)))
                    if int(v) == int(MARK):
                        expected[int(rr), int(cc)] = int(MARK)
                        red_abs.append((int(rr), int(cc)))
                for (r, c), v in group["moved"].items():
                    rr = int(top + r)
                    cc = int(left + c)
                    expected[int(rr), int(cc)] = int(v)
                    moved_abs[(int(rr), int(cc))] = int(v)
                    full_abs.append((int(rr), int(cc)))

                _stamp_moat(forbidden, full_abs)
                placed.append(
                    {
                        "kind": str(group["kind"]),
                        "color": int(group["color"]),
                        "input_abs": input_abs,
                        "moved_abs": moved_abs,
                        "red_abs": red_abs,
                        "safe_abs": [(int(top + r), int(left + c)) for r, c in group["safe_blockers"]],
                    }
                )
                placed_here = True
                break

            if not placed_here:
                ok = False
                break

        if not ok:
            continue

        blocker_cells = _choose_blockers(inp, placed, int(blocker_color), int(bg), rng)
        if len(blocker_cells) < 3:
            continue

        for r, c in blocker_cells:
            expected[int(r), int(c)] = int(bg)

        solved = main(inp)
        if not np.array_equal(solved, expected):
            continue

        reflected = simulate_reflect_move(inp)
        rotation_delta = int(np.sum(solved != reflected))
        if rotation_delta <= 0:
            continue

        collision_pixels = int(sum(1 for r, c in blocker_cells if int(solved[int(r), int(c)]) == int(bg)))
        moved_source_pixels = int(np.sum((inp != int(bg)) & (inp != int(MARK)) & (solved == int(bg))))
        if collision_pixels < 3 or moved_source_pixels < 8:
            continue

        meta = {
            "bg": int(bg),
            "shape": (int(h), int(w)),
            "group_kinds": [str(p["kind"]) for p in placed],
            "group_colors": [int(p["color"]) for p in placed],
            "blocker_color": int(blocker_color),
            "blocker_cells": blocker_cells,
            "collision_pixels": int(collision_pixels),
            "moved_source_pixels": int(moved_source_pixels),
            "rotation_delta": int(rotation_delta),
        }
        return inp, expected, meta

    raise RuntimeError("failed to generate a valid combo sample")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
