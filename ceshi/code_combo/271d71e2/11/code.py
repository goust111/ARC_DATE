# Combo variant of 271d71e2 (index 11)
# combo_spec: D2/1+D13/3
# Name: 灰色驱动行整除步数
#
# Design analysis
# a) 具体化与行为追踪
# - 视觉对象延续原题：6 背景，0 黑框容器，内部由 5 灰 / 7 橙组成，9 既画紧贴容器某一边的头线，
#   也画同轴更远处的目标线。
# - 融合后的唯一行为链变化有两处，而且两者必须同时生效：
#   1) 目标线只负责给出方向，不再限制位移远近。
#   2) 位移步数不是灰格总数，而是 `grey_count // inner_width`，只计算“完整灰行”的数量。
# - 具体小例子：内框 3x4，头在上侧，按头侧顺序已有前 2 格橙、后 10 格灰，目标线距头 2 格。
#   原题步数 = min(2, 10) = 2，只上移 2 格并把 2 个灰格变橙。
#   本融合题步数 = 10 // 4 = 2，仍上移 2 格，但会把 8 个灰格整体转橙，只留下 2 个残灰。
#   若目标线距头 1 格而灰格仍是 10，则原题上移 1，本题仍上移 2，并越过目标线。
# - 因此输出里应该同时能看到两种证据：
#   1) 有容器越过目标线或停在线前，说明忽略了目标线距离。
#   2) 仍残留 `grey_count % inner_width` 个灰格，说明步数来自“满行数”而不是像素数。
#
# b) 复杂度校准
# - 为了让“满行数”可被肉眼稳定识别，生成器把所有头侧限制为上/下两种竖直运动情形。
#   这样 `inner_width` 就直接对应水平一整行的长度，避免左/右运动时“用宽度除但按列充能”造成的
#   视觉歧义。
# - 仍保留 2-4 个容器、上下两个方向、不同宽高、0-1 个不动 decoy，以及 overrun / underrun /
#   equal 三类目标线关系，避免样例退化成单一模板。
# - 每个样例至少有一个 overrun mover，且至少有一个 remainder mover，让 D2 与 D13 的证据都出现。
#
# c) 隐式假设审查（main 逐段相关前提）
# - 容器检测仍假设灰橙区域外包一圈完整黑框；生成器保持矩形边框与纯灰/橙内部。
# - `pick_head_for_container()` 仍假设只有一条贴边头线最靠近容器；生成器给每个容器只放一条同色头线，
#   并保证对象之间留 1 格背景缓冲，避免别的 9 色结构误吸附。
# - 目标匹配仍从同色同形且位于头前方的候选里选最近者。虽然目标不再限制步数，但它仍必须存在且位于
#   正前方；生成器继续构造这一关系。
# - 旧代码把 `steps` 同时用于位移和转橙格数。融合后不成立：位移单位是“满行数”，转橙数量应是
#   `steps * inner_width`。solver 已显式拆开。
# - 旧代码的距离截断曾隐含防越界；现在可能越过目标线，因此生成器按“输入容器 + 输入头线 +
#   目标线 + 输出容器 + 输出头线”的并集包围盒摆放，保证 clip 实际不会裁剪。
#
# d) 放置合法性
# - 先按融合规则求出每个容器的真实 `steps`，再计算输入/输出并集包围盒，随机放置该包围盒并保留
#   1 格背景 moat。这样输入和输出都不会越界，也不会和别的组件互相覆盖。
# - 输出允许覆盖目标线；目标线只在输入中承担方向提示，不要求在输出仍完整可见。
#
# e) 视觉可观察性
# - 至少一个 mover 满足 `steps > head_dest_distance`，人类能看到“越线继续走”。
# - 至少一个 mover 满足 `grey_count % inner_width > 0`，人类能看到“剩下一点灰，但因为不满一整行，
#   不能再走一步”。
# - 通过同时混入 overrun / underrun / equal，解题者可以排除“目标线固定终点”与“灰格数逐像素驱动”
#   两种旧解释。
#
# f) 可视化整洁度
# - 不引入新颜色或额外标记；对象形态继续保持“一条线 + 一个矩形容器”的低复杂度结构。
# - 统一使用上下朝向，使满行残灰的证据更整齐，避免在同一张图里叠加过多不同几何语义。
#
# g) 变化的可推断性
# - 颜色语义不变：灰=待消耗预算，橙=已消耗预算，9=方向线，0=边框，6=背景。
# - 唯一需要学习的新关系是：步数由“完整灰行数”决定，且目标线只给方向。样例里强制保留余数残灰和
#   越线案例，为这两个推断提供重复支撑。

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
VERTICAL_HEAD_SIDES = (0, 1)
MAX_GAP = 7

LAST_GENERATED_META: dict | None = None


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _ordered_inner_cells(inner_h: int, inner_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if direction == (0, 1):
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif direction == (0, -1):
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif direction == (1, 0):
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    return _ordered_inner_cells(inner_h, inner_w, SIDE_TO_DIR[head_side])


def _grey_ordered_coords(r0: int, c0: int, inner: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    order = _ordered_inner_cells(inner.shape[0], inner.shape[1], direction)
    return [
        (int(r0 + 1 + rr), int(c0 + 1 + cc))
        for rr, cc in order
        if int(inner[rr, cc]) == GREY
    ]


def _row_span(inner_w: int) -> int:
    return int(inner_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Combo rule:
    - Destination markers still define the movement axis and forward direction.
    - Distance ignores the head-to-destination gap.
    - steps = grey_count // inner_width, i.e. only complete grey rows count.
    - Recoloring consumes `steps * inner_width` grey cells from the head-facing side inward.
    - The generator restricts training samples to top/bottom heads so "inner_width" is also the
      visible full-row length humans can count directly.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_head_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == int(hc)}
        comp0 = None
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

        border_pts = list(_rect_border_coords(r0, c0, r1, c1))
        best_d = None
        dist = {}
        for pr, pc in comp0["pts"]:
            d = min(abs(int(pr - br)) + abs(int(pc - bc)) for br, bc in border_pts)
            dist[(int(pr), int(pc))] = int(d)
            if best_d is None or d < best_d:
                best_d = int(d)
        if best_d is None:
            return None
        head_pts = {p for p, d in dist.items() if int(d) == int(best_d)}

        coords = np.array(list(head_pts), dtype=int)
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        if hc1 < c0:
            direction = (0, -1)
        elif hc0 > c1:
            direction = (0, 1)
        elif hr1 < r0:
            direction = (-1, 0)
        elif hr0 > r1:
            direction = (1, 0)
        else:
            direction = (0, 0)

        r00, c00 = int(hr0), int(hc0)
        head_shape = {(int(r - r00), int(c - c00)) for r, c in head_pts}
        return {
            "color": hc,
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": (r00, c00),
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]

        head_info = pick_head_for_container(cont["box"])
        if head_info is None:
            continue
        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape:
                continue
            if d["pts"] == comp_pts:
                continue
            dests.append(d)

        rem = comp_pts - head_pts
        if rem:
            mask_rem = np.zeros((H, W), dtype=bool)
            for rr, cc in rem:
                mask_rem[int(rr), int(cc)] = True
            labr, nr = label(mask_rem, structure=STRUCT4)
            for k in range(1, int(nr) + 1):
                pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
                if not pts:
                    continue
                adj = False
                for r, c in pts:
                    for dr, dc in DIR4:
                        rr, cc = int(r + dr), int(c + dc)
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(BORDER):
                            adj = True
                            break
                    if adj:
                        break
                if adj:
                    continue
                rs = [r for r, _c in pts]
                cs = [c for _r, c in pts]
                r00, c00 = int(min(rs)), int(min(cs))
                shp = {(int(r - r00), int(c - c00)) for r, c in pts}
                if shp != head_shape:
                    continue
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00)})

        if not dests:
            continue

        best = None
        best_dest = None
        for d in dests:
            tr0 = int(d["tl"][0] - head_tl[0])
            tc0 = int(d["tl"][1] - head_tl[1])
            if direction[0] != 0 and tc0 != 0:
                continue
            if direction[1] != 0 and tr0 != 0:
                continue
            if direction[0] == 1 and tr0 <= 0:
                continue
            if direction[0] == -1 and tr0 >= 0:
                continue
            if direction[1] == 1 and tc0 <= 0:
                continue
            if direction[1] == -1 and tc0 >= 0:
                continue
            dist = abs(tr0) + abs(tc0)
            if best is None or dist < best:
                best = dist
                best_dest = d
        if best_dest is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = _grey_ordered_coords(r0, c0, inner, direction)
        row_span = _row_span(inner.shape[1])
        steps = int(len(grey_coords) // row_span)

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)
        to_fill = set(grey_coords[: steps * row_span])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _relative_union_bbox(outer_h: int, outer_w: int, head_side: int, gap: int, steps: int) -> tuple[int, int, int, int]:
    dr, dc = SIDE_TO_DIR[head_side]
    boxes = [
        (0, 0, outer_h - 1, outer_w - 1),
        (dr * steps, dc * steps, dr * steps + outer_h - 1, dc * steps + outer_w - 1),
    ]

    if head_side == 0:
        boxes.extend([(-1, 0, -1, outer_w - 1), (-2 - gap, 0, -2 - gap, outer_w - 1), (-1 - steps, 0, -1 - steps, outer_w - 1)])
    elif head_side == 1:
        boxes.extend([(outer_h, 0, outer_h, outer_w - 1), (outer_h + 1 + gap, 0, outer_h + 1 + gap, outer_w - 1), (outer_h + steps, 0, outer_h + steps, outer_w - 1)])
    elif head_side == 2:
        boxes.extend([(0, -1, outer_h - 1, -1), (0, -2 - gap, outer_h - 1, -2 - gap), (0, -1 - steps, outer_h - 1, -1 - steps)])
    else:
        boxes.extend([(0, outer_w, outer_h - 1, outer_w), (0, outer_w + 1 + gap, outer_h - 1, outer_w + 1 + gap), (0, outer_w + steps, outer_h - 1, outer_w + steps)])

    r0 = min(b[0] for b in boxes)
    c0 = min(b[1] for b in boxes)
    r1 = max(b[2] for b in boxes)
    c1 = max(b[3] for b in boxes)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_head_or_dest(
    grid: np.ndarray,
    cr: int,
    cc: int,
    outer_h: int,
    outer_w: int,
    head_side: int,
    *,
    dest_gap: int | None = None,
) -> None:
    if dest_gap is None:
        if head_side == 0:
            grid[cr - 1, cc : cc + outer_w] = HEAD_COLOR
        elif head_side == 1:
            grid[cr + outer_h, cc : cc + outer_w] = HEAD_COLOR
        elif head_side == 2:
            grid[cr : cr + outer_h, cc - 1] = HEAD_COLOR
        else:
            grid[cr : cr + outer_h, cc + outer_w] = HEAD_COLOR
    else:
        if head_side == 0:
            grid[cr - 2 - dest_gap, cc : cc + outer_w] = HEAD_COLOR
        elif head_side == 1:
            grid[cr + outer_h + 1 + dest_gap, cc : cc + outer_w] = HEAD_COLOR
        elif head_side == 2:
            grid[cr : cr + outer_h, cc - 2 - dest_gap] = HEAD_COLOR
        else:
            grid[cr : cr + outer_h, cc + outer_w + 1 + dest_gap] = HEAD_COLOR


def _choose_relation_roles(n_movers: int) -> list[str]:
    roles = ["overrun"]
    if n_movers >= 2:
        roles.append(random.choice(["underrun", "equal"]))
    while len(roles) < n_movers:
        roles.append(random.choice(["overrun", "underrun", "equal"]))
    random.shuffle(roles)
    return roles


def _sample_mover_spec(relation: str, force_remainder: bool) -> dict | None:
    for _ in range(80):
        head_side = random.choice(VERTICAL_HEAD_SIDES)
        inner_h = random.randint(2, 8)
        inner_w = random.randint(2, 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        row_span = _row_span(inner_w)
        inner_area = inner_h * inner_w

        grey_choices = [g for g in range(row_span, inner_area + 1) if g // row_span >= 1]
        if force_remainder:
            grey_choices = [g for g in grey_choices if g % row_span != 0]
        if relation == "overrun":
            grey_choices = [g for g in grey_choices if g // row_span >= 2]
        if not grey_choices:
            continue

        grey = random.choice(grey_choices)
        steps = grey // row_span
        if steps < 1:
            continue

        if relation == "overrun":
            h2d = random.randint(1, steps - 1)
        elif relation == "equal":
            h2d = steps
        else:
            h2d = steps + random.randint(1, min(MAX_GAP, 4))
        gap = h2d - 1

        initial_orange = inner_area - grey
        bbox = _relative_union_bbox(outer_h, outer_w, head_side, gap, steps)
        total_h = bbox[2] - bbox[0] + 1
        total_w = bbox[3] - bbox[1] + 1
        if total_h > 26 or total_w > 26:
            continue

        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "head_dest_distance": h2d,
            "steps": steps,
            "initial_orange": initial_orange,
            "grey": grey,
            "row_span": row_span,
            "remainder": grey % row_span,
            "is_decoy": False,
            "relation": relation,
            "bbox": bbox,
            "total_h": total_h,
            "total_w": total_w,
            "cont_r": -bbox[0],
            "cont_c": -bbox[1],
        }
    return None


def _sample_decoy_spec() -> dict:
    head_side = random.choice(VERTICAL_HEAD_SIDES)
    inner_h = random.randint(1, 8)
    inner_w = random.randint(2, 5)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    gap = random.randint(0, MAX_GAP)
    bbox = _relative_union_bbox(outer_h, outer_w, head_side, gap, 0)
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": gap,
        "head_dest_distance": gap + 1,
        "steps": 0,
        "initial_orange": inner_h * inner_w,
        "grey": 0,
        "row_span": _row_span(inner_w),
        "remainder": 0,
        "is_decoy": True,
        "relation": "decoy",
        "bbox": bbox,
        "total_h": bbox[2] - bbox[0] + 1,
        "total_w": bbox[3] - bbox[1] + 1,
        "cont_r": -bbox[0],
        "cont_c": -bbox[1],
    }


def _build_expected(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        dr, dc = SIDE_TO_DIR[hs]
        row_span = a["row_span"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        if steps == 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = BG
        _draw_head_or_dest(expected, cr, cc, oh, ow, hs)

        nr = cr + dr * steps
        nc = cc + dc * steps

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        filled = initial_orange + steps * row_span
        for idx_cell in range(filled):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch
        _draw_head_or_dest(expected, nr, nc, oh, ow, hs)

    return expected


def _gen_attempt() -> np.ndarray | None:
    global LAST_GENERATED_META

    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    roles = _choose_relation_roles(n_movers)
    assemblies = []
    remainder_used = False
    for relation in roles:
        spec = _sample_mover_spec(relation, force_remainder=not remainder_used)
        if spec is None:
            return None
        remainder_used = remainder_used or (spec["remainder"] > 0)
        assemblies.append(spec)

    if not remainder_used:
        return None

    for _ in range(n_decoys):
        assemblies.append(_sample_decoy_spec())
    random.shuffle(assemblies)

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for a in assemblies:
        th, tw = a["total_h"], a["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                occupied[r0 : r0 + th, c0 : c0 + tw] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        gap = a["gap"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        _draw_head_or_dest(grid, cr, cc, oh, ow, hs)
        _draw_head_or_dest(grid, cr, cc, oh, ow, hs, dest_gap=gap)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _build_expected(grid, assemblies, placements)
    if not np.array_equal(output, expected):
        return None

    mover_specs = [a for a in assemblies if not a["is_decoy"]]
    LAST_GENERATED_META = {
        "containers": len(assemblies),
        "movers": len(mover_specs),
        "decoys": len(assemblies) - len(mover_specs),
        "vertical_only": True,
        "steps": [a["steps"] for a in mover_specs],
        "head_dest_distance": [a["head_dest_distance"] for a in mover_specs],
        "grey": [a["grey"] for a in mover_specs],
        "inner_width": [a["inner_w"] for a in mover_specs],
        "remainders": [a["remainder"] for a in mover_specs],
        "relations": [a["relation"] for a in mover_specs],
        "overrun_movers": sum(1 for a in mover_specs if a["steps"] > a["head_dest_distance"]),
        "remainder_movers": sum(1 for a in mover_specs if a["remainder"] > 0),
        "old_diff_movers": sum(
            1
            for a in mover_specs
            if min(a["head_dest_distance"], a["grey"]) != a["steps"]
        ),
    }
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
