# Combo variant of 247ef758 (index 15)
# combo_spec: D1/1+D2/0+D7/1
# Name: 多色精灵+尺寸匹配+面积排序
#
# Design analysis
# - concrete_visual:
#   保留原题大结构: 左侧精灵条带 | 单列 divider | 右侧整高矩形框。变化集中在三处并且互相耦合：
#   1) 左条带精灵改成双色 8 连通组件，求匹配时先数像素找主色，印刻时保留完整双色图案。
#   2) 框边标记不再是单格点，而是对边配对的连续段。长度仍编码 sprite bbox 尺寸，
#      但编码被拆成“前半段 + 后半段”分别放在对边上，二者共享的锚点给出目标行/列。
#      这样既保留尺寸匹配，又不会像整段镜像那样把不同 mover 的目标天然拉得过远。
#   3) 多个 mover 的覆盖顺序不看左条带上下位置，而看总像素面积，小者先印，大者后盖。
# - behavior_trace:
#   设某 mover 是 6 像素双色精灵，主色为 4，bbox 为 3x2，面积为 6。
#   原题会把“颜色 4 的单色 blob”匹配到边框上所有颜色 4 的单格行列标记，再按位置序覆盖。
#   本融合变体里要经历四步：
#   1) 先把左条带的整个双色组件提出来，统计得到主色 4；
#   2) 再到左右边找一对同色竖段：左边长度 + 右边长度 - 1 = 3，且二者在同一锚点行衔接；
#      上下边同理满足宽度 2。若同色段能对上锚点但总长度不对，仍必须忽略；
#   3) 最后把完整双色图案印到这些中心的笛卡尔积位置；
#   4) 若与别的 mover 重叠，则面积更大的 mover 最后覆盖。
# - complexity_calibration:
#   尺寸档位压在 2..4，便于人眼比较；mover 保持 2-3 个，distractor 保持 0-1 个。
#   每个样例强制至少出现一个“同色但错尺寸”的镜像边框段，否则 D1 会退化成纯颜色匹配。
#   mover 面积强制互异，并把小面积精灵摆在上方、大面积精灵摆在下方，再把目标中心聚成一团，
#   这样旧位置序与新面积序更容易产生可见差异。
# - implicit_assumptions_review:
#   原 solver 逐色提取 sprite，默认 sprite 单色；这里必须改成对左条带所有非背景像素整体连通域标记。
#   原 solver 逐格读取边框颜色，默认标记是单格点；这里必须在四条边上分别扫描最大同色段，
#   再按“左半长度/右半长度”或“上半长度/下半长度”配对回总尺寸。
#   原 solver 用 sprite 颜色同时做匹配与印刻；这里匹配键改成主色，印刻仍写入完整双色 pattern。
#   原 solver 的覆盖顺序依赖 r0/r1 的上下位置；这里主排序键改成 area，并用稳定 tie-break 防御外部输入。
#   新 solver 还隐含三条前提：每个左条带组件彼此 8 连通隔离、主色是严格多数、同色段之间不相接。
#   generate_input() 会逐条构造保证这些前提成立。
# - placement_legality:
#   同一侧的段不能重叠；同色段还必须留 base-color 间隔，避免 solver 把它们并成一个更长段。
#   有效锚点只从对应精灵能够完整落入框内的中心范围中选取；错误尺寸段只要求在边框内合法存在。
#   左条带精灵之间至少留 1 行空白，防止双色组件意外粘连成一个大连通域。
# - visual_observability:
#   人能直接看到：边框只使用主色，但输出框内出现了第二种 accent 色，所以必须先识别主色再搬运完整图案；
#   同时边框上同色段有长有短，且需要把对边两段的总长度拼起来才是有效尺寸；
#   重叠区的最终可见颜色则由可见面积大小决定，而不是由左条带上下位置决定。
# - cleanliness:
#   边框只叠加简单镜像段，框内背景保持空；accent 色只作为 sprite 内部结构出现，不会和 frame/divider 混用。

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the grid border."""
    h, w = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))
    return best_bbox


def _dominant_color(vals: list[int]) -> int:
    counts = Counter(map(int, vals))
    return sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0]


def _segment_extents(length: int) -> tuple[int, int]:
    upper = int((int(length) - 1) // 2)
    lower = int(length - 1 - upper)
    return upper, lower


def _segment_bounds(center: int, length: int) -> tuple[int, int]:
    upper, lower = _segment_extents(int(length))
    return int(center - upper), int(center + lower)


def _split_marker_lengths(total_length: int) -> tuple[int, int]:
    upper, lower = _segment_extents(int(total_length))
    return int(upper + 1), int(lower + 1)


def _paired_marker_bounds(anchor: int, total_length: int) -> tuple[tuple[int, int], tuple[int, int]]:
    upper, lower = _segment_extents(int(total_length))
    return (int(anchor - upper), int(anchor)), (int(anchor), int(anchor + lower))


def _find_divider_mask(g: np.ndarray, bg: int, fc0: int) -> np.ndarray:
    h, w = g.shape
    divider_mask = np.zeros((h, w), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True
    return divider_mask


def _extract_sprites(g: np.ndarray, bg: int, fc0: int, divider_mask: np.ndarray) -> list[dict]:
    left_mask = np.arange(g.shape[1])[None, :] < int(fc0)
    sprite_mask = (g != int(bg)) & left_mask & (~divider_mask)
    slab, sn = label_mask(sprite_mask, connectivity=8)

    sprites: list[dict] = []
    for sid in range(1, int(sn) + 1):
        pts = np.argwhere(slab == int(sid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        comp = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
        patch = np.where(comp, g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], int(bg))
        colors = [int(v) for v in patch[comp]]
        if not colors:
            continue
        sprites.append(
            {
                "main_color": int(_dominant_color(colors)),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
                "area": int(comp.sum()),
                "pattern": patch.astype(int),
            }
        )
    return sprites


def _scan_left_row_segments(g: np.ndarray, fc0: int, ir0: int, ir1: int, allowed_colors: set[int]) -> dict[tuple[int, int], list[int]]:
    marks: dict[tuple[int, int], list[int]] = defaultdict(list)
    r = int(ir0)
    while r <= int(ir1):
        col = int(g[int(r), int(fc0)])
        if col in allowed_colors:
            r0 = int(r)
            while r + 1 <= int(ir1) and int(g[int(r + 1), int(fc0)]) == col:
                r += 1
            r1 = int(r)
            marks[(int(col), int(r1 - r0 + 1))].append(int(r1))
        r += 1
    return marks


def _scan_right_row_segments(g: np.ndarray, fc1: int, ir0: int, ir1: int, allowed_colors: set[int]) -> dict[tuple[int, int], list[int]]:
    marks: dict[tuple[int, int], list[int]] = defaultdict(list)
    r = int(ir0)
    while r <= int(ir1):
        col = int(g[int(r), int(fc1)])
        if col in allowed_colors:
            r0 = int(r)
            while r + 1 <= int(ir1) and int(g[int(r + 1), int(fc1)]) == col:
                r += 1
            r1 = int(r)
            marks[(int(col), int(r1 - r0 + 1))].append(int(r0))
        r += 1
    return marks


def _scan_top_col_segments(g: np.ndarray, fr0: int, ic0: int, ic1: int, allowed_colors: set[int]) -> dict[tuple[int, int], list[int]]:
    marks: dict[tuple[int, int], list[int]] = defaultdict(list)
    c = int(ic0)
    while c <= int(ic1):
        col = int(g[int(fr0), int(c)])
        if col in allowed_colors:
            c0 = int(c)
            while c + 1 <= int(ic1) and int(g[int(fr0), int(c + 1)]) == col:
                c += 1
            c1 = int(c)
            marks[(int(col), int(c1 - c0 + 1))].append(int(c1))
        c += 1
    return marks


def _scan_bottom_col_segments(g: np.ndarray, fr1: int, ic0: int, ic1: int, allowed_colors: set[int]) -> dict[tuple[int, int], list[int]]:
    marks: dict[tuple[int, int], list[int]] = defaultdict(list)
    c = int(ic0)
    while c <= int(ic1):
        col = int(g[int(fr1), int(c)])
        if col in allowed_colors:
            c0 = int(c)
            while c + 1 <= int(ic1) and int(g[int(fr1), int(c + 1)]) == col:
                c += 1
            c1 = int(c)
            marks[(int(col), int(c1 - c0 + 1))].append(int(c0))
        c += 1
    return marks


def _intersect_anchor_maps(
    first: dict[tuple[int, int], list[int]],
    second: dict[tuple[int, int], list[int]],
    *,
    color: int,
    first_len: int,
    second_len: int,
) -> list[int]:
    return sorted(set(first.get((int(color), int(first_len)), [])) & set(second.get((int(color), int(second_len)), [])))


def _anchors_by_color(side_marks: dict[tuple[int, int], list[int]]) -> dict[int, list[int]]:
    merged: dict[int, set[int]] = defaultdict(set)
    for (color, _length), anchors in side_marks.items():
        merged[int(color)].update(map(int, anchors))
    return {int(color): sorted(vals) for color, vals in merged.items()}


def _extract_scene(grid: np.ndarray, *, match_mode: str) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    divider_mask = _find_divider_mask(g, int(bg), int(fc0))
    sprites = _extract_sprites(g, int(bg), int(fc0), divider_mask)
    move_colors = {int(s["main_color"]) for s in sprites}
    left_rows = _scan_left_row_segments(g, int(fc0), int(ir0), int(ir1), move_colors)
    right_rows = _scan_right_row_segments(g, int(fc1), int(ir0), int(ir1), move_colors)
    top_cols = _scan_top_col_segments(g, int(fr0), int(ic0), int(ic1), move_colors)
    bottom_cols = _scan_bottom_col_segments(g, int(fr1), int(ic0), int(ic1), move_colors)

    movers: list[dict] = []
    if match_mode == "color_size":
        for s in sprites:
            left_len, right_len = _split_marker_lengths(int(s["height"]))
            top_len, bottom_len = _split_marker_lengths(int(s["width"]))
            rows = _intersect_anchor_maps(
                left_rows,
                right_rows,
                color=int(s["main_color"]),
                first_len=int(left_len),
                second_len=int(right_len),
            )
            cols = _intersect_anchor_maps(
                top_cols,
                bottom_cols,
                color=int(s["main_color"]),
                first_len=int(top_len),
                second_len=int(bottom_len),
            )
            if rows and cols:
                item = dict(s)
                item["targets"] = [(int(r), int(c)) for r in rows for c in cols]
                movers.append(item)
    elif match_mode == "color_only":
        left_by_color = _anchors_by_color(left_rows)
        right_by_color = _anchors_by_color(right_rows)
        top_by_color = _anchors_by_color(top_cols)
        bottom_by_color = _anchors_by_color(bottom_cols)
        for s in sprites:
            rows = sorted(set(left_by_color.get(int(s["main_color"]), [])) & set(right_by_color.get(int(s["main_color"]), [])))
            cols = sorted(set(top_by_color.get(int(s["main_color"]), [])) & set(bottom_by_color.get(int(s["main_color"]), [])))
            if rows and cols:
                item = dict(s)
                item["targets"] = [(int(r), int(c)) for r in rows for c in cols]
                movers.append(item)
    else:
        raise ValueError(f"unknown match_mode: {match_mode}")

    return {
        "grid": g,
        "bg": int(bg),
        "inner_bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
        "movers": movers,
        "sprites": sprites,
    }


def _solve_with_modes(grid: np.ndarray, *, match_mode: str, order_mode: str) -> np.ndarray:
    scene = _extract_scene(grid, match_mode=match_mode)
    g = scene["grid"]
    bg = int(scene["bg"])
    ir0, ic0, ir1, ic1 = map(int, scene["inner_bbox"])
    movers = [dict(s) for s in scene["movers"]]

    out = g.copy()
    for s in movers:
        pat = np.array(s["pattern"], dtype=int)
        for rr, cc in np.argwhere(pat != int(bg)):
            out[int(s["r0"]) + int(rr), int(s["c0"]) + int(cc)] = int(bg)

    if order_mode == "area_asc":
        movers.sort(
            key=lambda s: (
                int(s["area"]),
                ((int(s["r0"]) + int(s["r1"])) / 2.0),
                int(s["c0"]),
                int(s["main_color"]),
            )
        )
    elif order_mode == "position":
        movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"]), int(s["main_color"])))
    else:
        raise ValueError(f"unknown order_mode: {order_mode}")

    for s in movers:
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pat = np.array(s["pattern"], dtype=int)
        pix = [(int(rr), int(cc2), int(pat[int(rr), int(cc2)])) for rr, cc2 in np.argwhere(pat != int(bg))]
        for tr, tc in s["targets"]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for rr0, cc0, color in pix:
                rr = int(s["r0"]) + int(rr0) + int(dr)
                cc3 = int(s["c0"]) + int(cc0) + int(dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule:
    - detect each left-strip sprite as a full two-color connected component
    - infer its dominant color as the matching key
    - recover row anchors from left/right edge segment pairs whose combined length matches
      the sprite bbox height, and col anchors from top/bottom pairs matching bbox width
    - stamp the full two-color pattern at every anchor-product target
    - apply overlap in ascending area order so the largest mover covers last
    """
    return _solve_with_modes(grid, match_mode="color_size", order_mode="area_asc")


def _center_bounds(ir0: int, ic0: int, ir1: int, ic1: int, sprite: np.ndarray) -> tuple[int, int, int, int]:
    sh, sw = sprite.shape
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    return min_row, max_row, min_col, max_col


def _make_shape_mask_exact(max_h: int, max_w: int, n_pixels: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2 or n_pixels < 3 or n_pixels > int(max_h * max_w):
        return None
    candidates = []
    for sh in range(2, int(max_h) + 1):
        for sw in range(2, int(max_w) + 1):
            base_area = int(sh + sw - 1)
            if int(base_area) <= int(n_pixels) <= int(sh * sw):
                candidates.append((int(sh), int(sw)))
    if not candidates:
        return None

    candidates.sort(key=lambda dims: (-(int(dims[0]) + int(dims[1])), -int(dims[0] * dims[1])))
    top = candidates[: max(1, min(len(candidates), 4))]
    for _ in range(120):
        sh, sw = random.choice(top)
        pivot_r = random.randrange(int(sh))
        pivot_c = random.randrange(int(sw))
        pixels = {(int(pivot_r), c) for c in range(int(sw))}
        pixels.update({(r, int(pivot_c)) for r in range(int(sh))})

        while len(pixels) < int(n_pixels):
            frontier = []
            for r in range(int(sh)):
                for c in range(int(sw)):
                    if (int(r), int(c)) in pixels:
                        continue
                    if any(
                        0 <= int(r + dr) < int(sh)
                        and 0 <= int(c + dc) < int(sw)
                        and (int(r + dr), int(c + dc)) in pixels
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if not (dr == 0 and dc == 0)
                    ):
                        frontier.append((int(r), int(c)))
            if not frontier:
                break
            pixels.add(random.choice(frontier))

        if len(pixels) != int(n_pixels):
            continue
        mask = np.zeros((int(sh), int(sw)), dtype=bool)
        for pr, pc in pixels:
            mask[int(pr), int(pc)] = True
        return mask
    return None


def _sample_connected_subset(coords: list[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    start = random.choice(coords)
    chosen = {start}
    frontier = [start]
    while len(chosen) < int(k):
        base = random.choice(frontier)
        nbrs = [
            (nr, nc)
            for nr in range(base[0] - 1, base[0] + 2)
            for nc in range(base[1] - 1, base[1] + 2)
            if (nr, nc) in coords and (nr, nc) not in chosen and not (nr == base[0] and nc == base[1])
        ]
        if nbrs:
            nxt = random.choice(nbrs)
            chosen.add(nxt)
            frontier.append(nxt)
            continue
        frontier.append(random.choice(list(chosen)))
        if len(frontier) > len(coords) * 4:
            break
    if len(chosen) < int(k):
        remaining = [p for p in coords if p not in chosen]
        random.shuffle(remaining)
        for p in remaining:
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in chosen):
                chosen.add(p)
                if len(chosen) >= int(k):
                    break
    return chosen


def _make_multicolor_sprite(primary: int, accent: int, max_h: int, max_w: int, area: int) -> np.ndarray | None:
    mask = _make_shape_mask_exact(int(max_h), int(max_w), int(area))
    if mask is None:
        return None
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    count = len(coords)
    max_accent = min(3, (int(count) - 1) // 2)
    if max_accent < 1:
        return None
    min_accent = 2 if int(count) >= 5 and max_accent >= 2 else 1
    n_accent = random.randint(int(min_accent), int(max_accent))
    accent_cells = _sample_connected_subset(coords, int(n_accent))

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        sprite[int(r), int(c)] = int(accent if (r, c) in accent_cells else primary)
    return sprite


def _make_block_multicolor_sprite(primary: int, accent: int, height: int, width: int) -> np.ndarray | None:
    if int(height) < 2 or int(width) < 2:
        return None
    sprite = np.full((int(height), int(width)), int(primary), dtype=int)
    accent_cells = [(0, 0)]
    if int(width) >= 3:
        accent_cells.append((0, 1))
    elif int(height) >= 3:
        accent_cells.append((1, 0))
    for r, c in accent_cells:
        sprite[int(r), int(c)] = int(accent)
    if int(np.count_nonzero(sprite == int(primary))) <= int(np.count_nonzero(sprite == int(accent))):
        return None
    return sprite


def _can_place_interval(start: int, end: int, occupied: np.ndarray, lo: int, hi: int) -> bool:
    if int(start) < int(lo) or int(end) > int(hi):
        return False
    for pos in range(int(start), int(end) + 1):
        if bool(occupied[int(pos - lo)]):
            return False
    return True


def _occupy_interval(start: int, end: int, occupied: np.ndarray, lo: int) -> None:
    occupied[int(start - lo) : int(end - lo) + 1] = True


def _reserve_interval(center: int, length: int, occupied: np.ndarray, lo: int, hi: int) -> bool:
    start, end = _segment_bounds(int(center), int(length))
    if not _can_place_interval(int(start), int(end), occupied, int(lo), int(hi)):
        return False
    _occupy_interval(int(start), int(end), occupied, int(lo))
    return True


def _place_interval_center(
    length: int,
    occupied: np.ndarray,
    lo: int,
    hi: int,
    *,
    center_lo: int | None = None,
    center_hi: int | None = None,
    forbidden_centers: set[int] | None = None,
    same_color_specs: list[tuple[int, int]] | None = None,
) -> int | None:
    upper, lower = _segment_extents(int(length))
    low = int(lo + upper if center_lo is None else max(int(center_lo), int(lo + upper)))
    high = int(hi - lower if center_hi is None else min(int(center_hi), int(hi - lower)))
    if low > high:
        return None
    candidates = list(range(int(low), int(high) + 1))
    random.shuffle(candidates)
    forbidden = set() if forbidden_centers is None else {int(v) for v in forbidden_centers}
    existing_specs = [] if same_color_specs is None else [(int(center), int(length)) for center, length in same_color_specs]
    for center in candidates:
        if int(center) in forbidden:
            continue
        start, end = _segment_bounds(int(center), int(length))
        if existing_specs:
            ok_gap = True
            for old_center, old_length in existing_specs:
                old_start, old_end = _segment_bounds(int(old_center), int(old_length))
                if not (int(end) < int(old_start - 1) or int(start) > int(old_end + 1)):
                    ok_gap = False
                    break
            if not ok_gap:
                continue
        if _can_place_interval(int(start), int(end), occupied, int(lo), int(hi)):
            _occupy_interval(int(start), int(end), occupied, int(lo))
            return int(center)
    return None


def _place_one_of_lengths(
    lengths: list[int],
    occupied: np.ndarray,
    lo: int,
    hi: int,
    *,
    center_lo: int | None = None,
    center_hi: int | None = None,
    forbidden_centers: set[int] | None = None,
    same_color_specs: list[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    choices = [int(v) for v in lengths]
    random.shuffle(choices)
    for length in choices:
        center = _place_interval_center(
            int(length),
            occupied,
            int(lo),
            int(hi),
            center_lo=center_lo,
            center_hi=center_hi,
            forbidden_centers=forbidden_centers,
            same_color_specs=same_color_specs,
        )
        if center is not None:
            return int(center), int(length)
    return None


def _pair_gap_ok(anchor: int, total_length: int, existing_specs: list[tuple[int, int]] | None) -> bool:
    if not existing_specs:
        return True
    (start_a, end_a), (start_b, end_b) = _paired_marker_bounds(int(anchor), int(total_length))
    for old_anchor, old_total in existing_specs:
        (old_start_a, old_end_a), (old_start_b, old_end_b) = _paired_marker_bounds(int(old_anchor), int(old_total))
        if not (int(end_a) < int(old_start_a - 1) or int(start_a) > int(old_end_a + 1)):
            return False
        if not (int(end_b) < int(old_start_b - 1) or int(start_b) > int(old_end_b + 1)):
            return False
    return True


def _can_place_pair_anchor(
    anchor: int,
    total_length: int,
    first_occ: np.ndarray,
    second_occ: np.ndarray,
    lo: int,
    hi: int,
    *,
    same_color_specs: list[tuple[int, int]] | None = None,
) -> bool:
    (start_a, end_a), (start_b, end_b) = _paired_marker_bounds(int(anchor), int(total_length))
    if not _pair_gap_ok(int(anchor), int(total_length), same_color_specs):
        return False
    return _can_place_interval(int(start_a), int(end_a), first_occ, int(lo), int(hi)) and _can_place_interval(
        int(start_b), int(end_b), second_occ, int(lo), int(hi)
    )


def _reserve_pair_anchor(
    anchor: int,
    total_length: int,
    first_occ: np.ndarray,
    second_occ: np.ndarray,
    lo: int,
    hi: int,
    *,
    same_color_specs: list[tuple[int, int]] | None = None,
) -> bool:
    if not _can_place_pair_anchor(
        int(anchor),
        int(total_length),
        first_occ,
        second_occ,
        int(lo),
        int(hi),
        same_color_specs=same_color_specs,
    ):
        return False
    (start_a, end_a), (start_b, end_b) = _paired_marker_bounds(int(anchor), int(total_length))
    _occupy_interval(int(start_a), int(end_a), first_occ, int(lo))
    _occupy_interval(int(start_b), int(end_b), second_occ, int(lo))
    return True


def _place_pair_anchor(
    total_length: int,
    first_occ: np.ndarray,
    second_occ: np.ndarray,
    lo: int,
    hi: int,
    *,
    anchor_lo: int | None = None,
    anchor_hi: int | None = None,
    forbidden_anchors: set[int] | None = None,
    same_color_specs: list[tuple[int, int]] | None = None,
) -> int | None:
    upper, lower = _segment_extents(int(total_length))
    low = int(lo + upper if anchor_lo is None else max(int(anchor_lo), int(lo + upper)))
    high = int(hi - lower if anchor_hi is None else min(int(anchor_hi), int(hi - lower)))
    if low > high:
        return None
    candidates = list(range(int(low), int(high) + 1))
    random.shuffle(candidates)
    forbidden = set() if forbidden_anchors is None else {int(v) for v in forbidden_anchors}
    for anchor in candidates:
        if int(anchor) in forbidden:
            continue
        if _reserve_pair_anchor(
            int(anchor),
            int(total_length),
            first_occ,
            second_occ,
            int(lo),
            int(hi),
            same_color_specs=same_color_specs,
        ):
            return int(anchor)
    return None


def _place_one_of_total_lengths(
    lengths: list[int],
    first_occ: np.ndarray,
    second_occ: np.ndarray,
    lo: int,
    hi: int,
    *,
    anchor_lo: int | None = None,
    anchor_hi: int | None = None,
    forbidden_anchors: set[int] | None = None,
    same_color_specs: list[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    choices = [int(v) for v in lengths]
    random.shuffle(choices)
    for length in choices:
        anchor = _place_pair_anchor(
            int(length),
            first_occ,
            second_occ,
            int(lo),
            int(hi),
            anchor_lo=anchor_lo,
            anchor_hi=anchor_hi,
            forbidden_anchors=forbidden_anchors,
            same_color_specs=same_color_specs,
        )
        if anchor is not None:
            return int(anchor), int(length)
    return None


def _choose_clustered_pair_anchors_1d(
    total_lengths: list[int],
    bounds: list[tuple[int, int]],
    lo: int,
    hi: int,
) -> list[int] | None:
    n = len(total_lengths)
    place_order = sorted(range(n), key=lambda idx: (-int(total_lengths[idx]), int(idx)))
    for _ in range(200):
        first_occ = np.zeros(int(hi - lo + 1), dtype=bool)
        second_occ = np.zeros(int(hi - lo + 1), dtype=bool)
        used: set[int] = set()
        anchors: list[int | None] = [None] * n
        anchor0 = random.randint(int(lo), int(hi))
        success = True
        for idx in place_order:
            min_anchor, max_anchor = bounds[int(idx)]
            desired = int(anchor0 + idx)
            candidates = list(range(int(min_anchor), int(max_anchor) + 1))
            random.shuffle(candidates)
            candidates.sort(key=lambda anchor: (abs(int(anchor) - int(desired)), abs(int(anchor) - int(anchor0))))
            placed = False
            for anchor in candidates:
                if int(anchor) in used:
                    continue
                if _reserve_pair_anchor(int(anchor), int(total_lengths[int(idx)]), first_occ, second_occ, int(lo), int(hi)):
                    used.add(int(anchor))
                    anchors[int(idx)] = int(anchor)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and all(anchor is not None for anchor in anchors):
            return [int(anchor) for anchor in anchors if anchor is not None]
    return None


def _choose_clustered_centers_1d(
    lengths: list[int],
    bounds: list[tuple[int, int]],
    lo: int,
    hi: int,
) -> list[int] | None:
    n = len(lengths)
    place_order = sorted(range(n), key=lambda idx: (-int(lengths[idx]), int(idx)))
    for _ in range(200):
        occupied = np.zeros(int(hi - lo + 1), dtype=bool)
        used: set[int] = set()
        centers: list[int | None] = [None] * n
        anchor = random.randint(int(lo), int(hi))
        success = True
        for idx in place_order:
            min_center, max_center = bounds[int(idx)]
            desired = int(anchor + idx)
            candidates = list(range(int(min_center), int(max_center) + 1))
            random.shuffle(candidates)
            candidates.sort(key=lambda center: (abs(int(center) - int(desired)), abs(int(center) - int(anchor))))
            placed = False
            for center in candidates:
                if int(center) in used:
                    continue
                if _reserve_interval(int(center), int(lengths[int(idx)]), occupied, int(lo), int(hi)):
                    used.add(int(center))
                    centers[int(idx)] = int(center)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and all(center is not None for center in centers):
            return [int(center) for center in centers if center is not None]
    return None


def _sample_gaps(total_extra: int, n_slots: int) -> list[int]:
    gaps = [0] * int(n_slots)
    for _ in range(int(total_extra)):
        gaps[random.randrange(int(n_slots))] += 1
    return gaps


def _new_accent_footprint(inp: np.ndarray, out: np.ndarray, accent_color: int) -> int:
    return int(np.count_nonzero((out == int(accent_color)) & (inp != int(accent_color))))


def _frame_new_accent_footprint(inp: np.ndarray, out: np.ndarray, accent_color: int) -> int:
    scene = _extract_scene(inp, match_mode="color_size")
    ir0, ic0, ir1, ic1 = map(int, scene["inner_bbox"])
    return int(np.count_nonzero((out[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1] == int(accent_color))))


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    h = interior_h + 2
    if not 10 <= h <= 15:
        return None
    sprite_area_w = random.randint(4, 5)
    frame_w = interior_w + 2
    w = sprite_area_w + 1 + frame_w
    if not 14 <= w <= 20:
        return None

    div_col = int(sprite_area_w)
    fc0, fc1 = int(sprite_area_w + 1), int(w - 1)
    ir0, ic0 = 1, int(fc0 + 1)
    ir1, ic1 = int(h - 2), int(fc1 - 1)

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 1])
    n_total = int(n_movers + n_distractors)

    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    accent_color = int(all_colors[idx])
    idx += 1
    primary_colors = [int(v) for v in all_colors[idx : idx + n_total]]
    mover_colors = primary_colors[:n_movers]
    distractor_colors = primary_colors[n_movers:]

    mover_info: list[dict] = []
    max_sprite_h = 4
    max_sprite_w = min(4, int(sprite_area_w))
    rect_choices = [
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
    ]
    rect_choices = [
        (int(sh), int(sw))
        for sh, sw in rect_choices
        if int(sh) <= int(max_sprite_h) and int(sw) <= int(max_sprite_w)
    ]
    random.shuffle(rect_choices)
    chosen_rects: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for sh, sw in rect_choices:
        area = int(sh * sw)
        if area in used_areas:
            continue
        chosen_rects.append((int(sh), int(sw)))
        used_areas.add(area)
        if len(chosen_rects) == int(n_movers):
            break
    if len(chosen_rects) < int(n_movers):
        return None
    chosen_rects.sort(key=lambda hw: (int(hw[0] * hw[1]), int(hw[0]), int(hw[1])))
    for mc, (sh, sw) in zip(mover_colors, chosen_rects):
        sp = _make_block_multicolor_sprite(int(mc), int(accent_color), int(sh), int(sw))
        if sp is None:
            return None
        mover_info.append({"main_color": int(mc), "sprite": sp, "area": int(sh * sw)})

    mover_info.sort(key=lambda item: (int(item["area"]), int(item["main_color"])))
    row_lengths = [int(np.array(item["sprite"]).shape[0]) for item in mover_info]
    col_lengths = [int(np.array(item["sprite"]).shape[1]) for item in mover_info]
    row_bounds = []
    col_bounds = []
    for item in mover_info:
        sp = np.array(item["sprite"], dtype=int)
        min_row, max_row, min_col, max_col = _center_bounds(int(ir0), int(ic0), int(ir1), int(ic1), sp)
        row_bounds.append((int(min_row), int(max_row)))
        col_bounds.append((int(min_col), int(max_col)))
    row_anchors = _choose_clustered_pair_anchors_1d(row_lengths, row_bounds, int(ir0), int(ir1))
    col_anchors = _choose_clustered_pair_anchors_1d(col_lengths, col_bounds, int(ic0), int(ic1))
    if row_anchors is None or col_anchors is None:
        return None

    left_row_occ = np.zeros(int(interior_h), dtype=bool)
    right_row_occ = np.zeros(int(interior_h), dtype=bool)
    top_col_occ = np.zeros(int(interior_w), dtype=bool)
    bottom_col_occ = np.zeros(int(interior_w), dtype=bool)
    used_row_anchors: set[int] = set()
    used_col_anchors: set[int] = set()
    mover_targets: dict[int, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {}
    for item, row_anchor, col_anchor in zip(mover_info, row_anchors, col_anchors):
        mc = int(item["main_color"])
        sp = np.array(item["sprite"], dtype=int)
        sh, sw = map(int, sp.shape)
        min_row, max_row, min_col, max_col = _center_bounds(int(ir0), int(ic0), int(ir1), int(ic1), sp)
        if not (int(min_row) <= int(row_anchor) <= int(max_row) and int(min_col) <= int(col_anchor) <= int(max_col)):
            return None
        if int(row_anchor) in used_row_anchors or int(col_anchor) in used_col_anchors:
            return None
        if not _reserve_pair_anchor(int(row_anchor), int(sh), left_row_occ, right_row_occ, int(ir0), int(ir1)):
            return None
        if not _reserve_pair_anchor(int(col_anchor), int(sw), top_col_occ, bottom_col_occ, int(ic0), int(ic1)):
            return None
        used_row_anchors.add(int(row_anchor))
        used_col_anchors.add(int(col_anchor))
        mover_targets[mc] = ([(int(row_anchor), int(sh))], [(int(col_anchor), int(sw))])

    if random.random() < 0.45:
        chosen = random.choice(mover_info)
        mc = int(chosen["main_color"])
        sp = np.array(chosen["sprite"], dtype=int)
        sh, sw = map(int, sp.shape)
        min_row, max_row, min_col, max_col = _center_bounds(int(ir0), int(ic0), int(ir1), int(ic1), sp)
        extra_row = _place_pair_anchor(
            int(sh),
            left_row_occ,
            right_row_occ,
            int(ir0),
            int(ir1),
            anchor_lo=int(min_row),
            anchor_hi=int(max_row),
            forbidden_anchors=used_row_anchors,
            same_color_specs=mover_targets[mc][0],
        )
        if extra_row is not None:
            used_row_anchors.add(int(extra_row))
            mover_targets[mc][0].append((int(extra_row), int(sh)))
        extra_col = _place_pair_anchor(
            int(sw),
            top_col_occ,
            bottom_col_occ,
            int(ic0),
            int(ic1),
            anchor_lo=int(min_col),
            anchor_hi=int(max_col),
            forbidden_anchors=used_col_anchors,
            same_color_specs=mover_targets[mc][1],
        )
        if extra_col is not None:
            used_col_anchors.add(int(extra_col))
            mover_targets[mc][1].append((int(extra_col), int(sw)))

    ambiguous = random.choice(mover_info)
    amb_color = int(ambiguous["main_color"])
    amb_sprite = np.array(ambiguous["sprite"], dtype=int)
    good_h, good_w = map(int, amb_sprite.shape)
    bad_row_lengths = [v for v in range(2, min(4, interior_h) + 1) if int(v) != int(good_h)]
    bad_col_lengths = [v for v in range(2, min(4, interior_w, sprite_area_w) + 1) if int(v) != int(good_w)]
    if not bad_row_lengths or not bad_col_lengths:
        return None
    bad_row_spec = _place_one_of_total_lengths(
        bad_row_lengths,
        left_row_occ,
        right_row_occ,
        int(ir0),
        int(ir1),
        forbidden_anchors=used_row_anchors,
        same_color_specs=mover_targets[amb_color][0],
    )
    if bad_row_spec is None:
        return None
    bad_row_center, bad_row_length = bad_row_spec
    mover_targets[amb_color][0].append((int(bad_row_center), int(bad_row_length)))
    used_row_anchors.add(int(bad_row_center))

    bad_col_spec = _place_one_of_total_lengths(
        bad_col_lengths,
        top_col_occ,
        bottom_col_occ,
        int(ic0),
        int(ic1),
        forbidden_anchors=used_col_anchors,
        same_color_specs=mover_targets[amb_color][1],
    )
    if bad_col_spec is None:
        return None
    bad_col_center, bad_col_length = bad_col_spec
    mover_targets[amb_color][1].append((int(bad_col_center), int(bad_col_length)))
    used_col_anchors.add(int(bad_col_center))

    all_items: list[tuple[int, np.ndarray, bool]] = [(int(item["main_color"]), np.array(item["sprite"], dtype=int), False) for item in mover_info]
    for dc in distractor_colors:
        d_choices = [v for v in range(4, 10) if int(v) <= int(max_sprite_h * max_sprite_w)]
        if not d_choices:
            return None
        d_area = random.choice(d_choices)
        ds = _make_multicolor_sprite(int(dc), int(accent_color), int(max_sprite_h), int(max_sprite_w), int(d_area))
        if ds is None:
            return None
        all_items.append((int(dc), np.array(ds, dtype=int), True))
    min_needed_h = sum(int(sp.shape[0]) for _, sp, _ in all_items) + max(0, len(all_items) - 1)
    if int(min_needed_h) > int(h):
        return None

    inp = np.full((h, w), int(bg), dtype=int)
    inp[:, int(div_col)] = int(divider_color)
    inp[0, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[int(h - 1), int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[1 : int(h - 1), int(fc0)] = int(frame_base)
    inp[1 : int(h - 1), int(fc1)] = int(frame_base)

    for mc, (row_specs, col_specs) in mover_targets.items():
        for anchor, length in row_specs:
            (lr0, lr1), (rr0, rr1) = _paired_marker_bounds(int(anchor), int(length))
            inp[int(lr0) : int(lr1) + 1, int(fc0)] = int(mc)
            inp[int(rr0) : int(rr1) + 1, int(fc1)] = int(mc)
        for anchor, length in col_specs:
            (tc0, tc1), (bc0, bc1) = _paired_marker_bounds(int(anchor), int(length))
            inp[0, int(tc0) : int(tc1) + 1] = int(mc)
            inp[int(h - 1), int(bc0) : int(bc1) + 1] = int(mc)

    extra_rows = int(h - min_needed_h)
    gap_extras = _sample_gaps(int(extra_rows), int(len(all_items) + 1))
    cur_row = int(gap_extras[0])
    for idx_item, (_primary, sp, _is_distractor) in enumerate(all_items):
        sh, sw = map(int, sp.shape)
        if int(sw) > int(sprite_area_w) or int(cur_row + sh) > int(h):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        inp[int(cur_row) : int(cur_row + sh), int(c_off) : int(c_off + sw)] = np.where(
            sp != int(bg),
            sp,
            inp[int(cur_row) : int(cur_row + sh), int(c_off) : int(c_off + sw)],
        )
        cur_row += int(sh)
        if idx_item < len(all_items) - 1:
            cur_row += 1 + int(gap_extras[idx_item + 1])

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        fused_out = main(inp)
        color_only_out = _solve_with_modes(inp, match_mode="color_only", order_mode="area_asc")
        position_out = _solve_with_modes(inp, match_mode="color_size", order_mode="position")
    except Exception:
        return None

    if np.array_equal(inp, fused_out):
        return None
    if np.array_equal(fused_out, color_only_out):
        return None
    if np.array_equal(fused_out, position_out):
        return None
    if _new_accent_footprint(inp, fused_out, int(accent_color)) <= 0:
        return None
    if _frame_new_accent_footprint(inp, fused_out, int(accent_color)) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 800 attempts")
