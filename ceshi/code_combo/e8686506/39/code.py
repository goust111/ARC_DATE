# Combo variant of e8686506 (index 39)
# combo_spec: D3/1+D13/0
# Name: 填入框架洞+双色背景

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, paste, random_free_location


# Design analysis
# a) 具体化与行为追踪:
#    - 原题: 单一背景上散落若干碎片，solver 统计全部前景面积，反推出一个左右对称的稠密矩形，再把碎片拼回去。
#    - 融合后: 画布先被左右双色背景切成两个大区；中间放着一个已经定位好的大框架，框架内部保留洞，框架外散落碎片。
#      这些碎片不是再去拼成整块矩形，而是分别填进框架洞中；左背景区的碎片对应左半洞，右背景区对应右半洞。
#    - 具体小例子: 假设框架 bbox 宽 9，高 8，中心列是实心框架。左腔里有 4 个洞，右腔是它们的镜像。
#      输入顶/底边先暴露出双色分界；solver 先从顶/底边读出 split，再找唯一最大前景组件作为框架，
#      然后在框架 bbox 内把“左区像素=左背景色”的位置当左洞，把“右区像素=右背景色”的位置当右洞；
#      最后只拿对应区的外部碎片去做同形匹配。与原题相比，`总面积->矩形尺寸` 这一步完全消失，取而代之的是
#      `背景分区检测 + 固定框架内洞检测 + 分区约束下的形状匹配`。
# b) 复杂度校准:
#    - 若只给旧题外面套一个框而不让背景分区参与推理，双色背景会退化成装饰；因此这里让框架跨越分界，洞也分布在左右两腔，
#      并强制外部碎片按所属背景区散布，保证人眼能看到“左边碎片填左洞、右边碎片填右洞”。
#    - 若洞跨过中心线，左右归属会变含混；因此生成器固定中心列为框架，所有洞都严格位于左腔或右腔，减少自由度但保留足够形状变化。
#    - 洞形状来自一个小型 4 连通形状库并允许旋转，洞数控制在 6-10 个，颜色数控制在输出 4-6 色，视觉差异明显但不会杂乱。
# c) 隐式假设审查:
#    - 原 `bg = get_bg(g)` 假设背景唯一；融合后无效，必须改成从顶/底边显式读出左右背景色与 split。
#    - 原 `for col in unique(g) if col != bg` 假设“非 bg 全是前景”；现在必须只把“不等于其所在分区背景色”的像素视为前景。
#    - 原 `total_area -> 因子分解求 HxW` 假设输出是密实矩形；融合后不成立，因为目标尺寸由框架 bbox 直接给定。
#    - 原角点/边条/竖线锚点服务于矩形边框；现在框架洞任务不再需要这些锚点，保留会引入误判，所以整体移除。
#    - 新增假设: 框架是唯一最大单色 8 连通前景组件；生成器用单色大框架且面积远大于任何外部碎片来保证。
#    - 新增假设: 框架 bbox 内除框架色外，只会出现“该列所属背景区的背景色”；生成器恢复 bbox 时按左右背景分别回填来保证。
# d) 放置合法性:
#    - 散布阶段先用 `-1` 阻塞整个框架 bbox，再把外部碎片放到各自背景区，最后恢复 bbox 内的双色背景并覆写框架像素，
#      因此碎片不会越界、不会提前落入洞中，也不会与框架重叠。
#    - 所有洞都由左腔中的 4 连通小形状镜像得到，天然不接触外边界，也不会跨越中心列。
#    - `random_free_location(..., border=1, padding=1)` 保证外部碎片彼此分离，顶/底边保留纯背景供 solver 检测 split。
# e) 视觉可观察性:
#    - 输入中最显眼的是一块横跨双色背景分界的大框架，内部左右各有空洞；框架外两侧分别散落与洞同形的碎片。
#    - 输出直接是同一框架被填满后的结果。对人类来说，规则可归纳为“先忽略双色背景本身，再用左右两侧碎片分别补上左右洞”。
#    - 这里不依赖颜色编号大小、组件枚举顺序等不可视因素；洞和碎片的对应完全由可见的形状与所在背景区决定。
# f) 可视化整洁度:
#    - 框架固定为单色，双色背景只在空白区域与洞位出现，外部碎片与框架隔离，图层关系清楚。
#    - 中心列始终是实心框架，使左右洞区视觉分离，避免“一个大洞跨中线”造成读图混乱。
# g) 变化的可推断性:
#    - 颜色映射只做随机重映射，不承载隐藏逻辑；真正的线索是“形状 + 左右背景区”。
#    - 同一洞形在镜像两侧会重复出现，给人类提供重复支撑；不同样例中双色背景颜色会变，但“按分区归属匹配”保持稳定。
# h) 新的 solver 假设与 generator 对应保证:
#    - solver 假设顶行和底行完整暴露同一个竖直 split；generator 对框架和碎片都保留 1 像素外边框。
#    - solver 假设每个洞形在对应背景区外部有同形碎片；generator 直接从每个洞组件裁出一个同色 patch 并散到同侧区域。
#    - solver 假设洞不跨 split，且外部碎片也不跨 split；generator 通过“中心列为框架 + 分区内散布”强制满足。
# i) 量化的变体判据:
#    - 统计框架 bbox 内左区洞组件数 `L_holes` 与右区洞组件数 `R_holes`，以及框架外左/右区碎片数 `L_pieces`、`R_pieces`。
#      本融合变体要求每个样本都满足 `L_holes > 0`, `R_holes > 0`, `L_holes = L_pieces`, `R_holes = R_pieces`。
#      原题与仅 D3 变体都不会同时出现“框架内双色洞位 + 外部碎片按双色分区归属”的统计特征。


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1), coords)


def _mirror_sig(sig: tuple[int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    h, w, coords = sig
    mirrored = tuple(sorted((int(r), int(w) - 1 - int(c)) for r, c in coords))
    return int(h), int(w), mirrored


def _detect_partition_background(g: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    H, W = g.shape
    assert H >= 3 and W >= 3, "grid must be large enough for a visible 2-zone background"

    top = list(map(int, g[0]))
    bottom = list(map(int, g[H - 1]))
    left_bg = int(top[0])
    right_bg = int(top[W - 1])
    assert left_bg != right_bg, "dual-background variant requires two distinct background colors"

    changes = [c for c in range(1, W) if int(top[c - 1]) != int(top[c])]
    assert len(changes) == 1, "top border must reveal a single vertical split"
    split = int(changes[0])
    assert 1 <= split <= W - 1, "background split must create two non-empty zones"

    assert all(int(v) == left_bg for v in top[:split]), "top-left border must be pure left background"
    assert all(int(v) == right_bg for v in top[split:]), "top-right border must be pure right background"
    assert all(int(v) == left_bg for v in bottom[:split]), "bottom-left border must preserve left background"
    assert all(int(v) == right_bg for v in bottom[split:]), "bottom-right border must preserve right background"

    bg_mask = np.zeros_like(g, dtype=bool)
    bg_mask[:, :split] = (g[:, :split] == left_bg)
    bg_mask[:, split:] = (g[:, split:] == right_bg)
    return split, left_bg, right_bg, bg_mask


def _classify_zone(pts: np.ndarray, split_col: int) -> str:
    cols = [int(c) for _, c in pts]
    if max(cols) < int(split_col):
        return "left"
    if min(cols) >= int(split_col):
        return "right"
    raise AssertionError("component must stay inside one background zone")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the holes of a symmetric frame by matching scattered fragments from the corresponding background zone.

    Input structure:
    - The canvas is split into left/right background regions, each with its own background color.
    - A single large monochrome frame already sits in its final position across the split; inside its bbox, empty cells show the background color of their zone.
    - Outside that bbox, monochrome fragments are scattered separately in the left and right background regions.

    Rule:
    - Read the vertical background split first and ignore only pixels that match their zone's background color.
    - Detect the unique largest foreground component as the frame, then treat the zone-colored cells inside its bbox as hole components.
    - For each hole, find an outside fragment with the same shape from the same background zone and fill the hole with that fragment's color.

    Key insight:
    - The frame fixes the destination geometry, while the dual background tells you which scattered pieces belong to the left half versus the right half; shape matching only works after both cues are combined.
    """
    g = np.asarray(grid, dtype=int)
    split, left_bg, right_bg, bg_mask = _detect_partition_background(g)

    fg_mask = ~bg_mask
    piece_colors = sorted(set(map(int, np.unique(g[fg_mask]))))
    assert piece_colors, "input must contain non-background foreground"

    frame_candidates: list[tuple[int, int, np.ndarray]] = []
    for col in piece_colors:
        for pts in components_mask((g == int(col)) & fg_mask, connectivity=8):
            frame_candidates.append((len(pts), int(col), np.array(pts, dtype=int)))

    assert frame_candidates, "input must contain a frame component"
    max_area = max(area for area, _, _ in frame_candidates)
    frame_choices = [(col, pts) for area, col, pts in frame_candidates if area == max_area]
    assert len(frame_choices) == 1, "frame must be the unique largest monochrome component"
    frame_col, frame_pts = frame_choices[0]

    r0, c0, r1, c1 = bbox_coords(frame_pts)
    sub = crop_bbox(g, (r0, c0, r1, c1))
    local_split = int(split - c0)
    assert 1 <= local_split <= sub.shape[1] - 1, "frame bbox must straddle the background split"

    frame_mask = sub == int(frame_col)
    hole_mask = np.zeros_like(sub, dtype=bool)
    hole_mask[:, :local_split] = (sub[:, :local_split] == int(left_bg))
    hole_mask[:, local_split:] = (sub[:, local_split:] == int(right_bg))

    assert hole_mask.any(), "frame must contain at least one hole"
    assert np.all(frame_mask | hole_mask), "frame bbox may only contain frame pixels or zone background holes"
    assert not hole_mask[0].any() and not hole_mask[-1].any(), "holes must stay away from bbox top/bottom edges"
    assert not hole_mask[:, 0].any() and not hole_mask[:, -1].any(), "holes must stay away from bbox side edges"
    assert np.array_equal(frame_mask, frame_mask[:, ::-1]), "frame mask must be LR-symmetric"
    assert np.array_equal(hole_mask, hole_mask[:, ::-1]), "hole mask must be LR-symmetric"

    holes: list[tuple[str, tuple[int, int, tuple[tuple[int, int], ...]], np.ndarray]] = []
    for pts in components_mask(hole_mask, connectivity=4):
        arr = np.array(pts, dtype=int)
        side = _classify_zone(arr, local_split)
        holes.append((side, _piece_sig(arr), arr))

    assert any(side == "left" for side, _, _ in holes), "there must be at least one left-side hole"
    assert any(side == "right" for side, _, _ in holes), "there must be at least one right-side hole"

    outside_fg = fg_mask.copy()
    outside_fg[r0 : r1 + 1, c0 : c1 + 1] = False

    pieces_by_bucket: dict[tuple[str, tuple[int, int, tuple[tuple[int, int], ...]]], Counter] = {}
    for col in sorted(set(map(int, np.unique(g[outside_fg])))):
        comps = components_mask(((g == int(col)) & outside_fg), connectivity=4)
        for pts in comps:
            arr = np.array(pts, dtype=int)
            side = _classify_zone(arr, split)
            key = _piece_sig(arr)
            pieces_by_bucket.setdefault((side, key), Counter())[int(col)] += 1

    assert pieces_by_bucket, "input must contain scattered fill pieces outside the frame"

    out = np.full(sub.shape, int(frame_col), dtype=int)
    for side, key, pts in holes:
        bucket_key = (side, key)
        cmap = pieces_by_bucket.get(bucket_key)
        assert cmap is not None, "every hole must have a same-side shape-matched outside piece"
        assert len(cmap) == 1, "same-side identical shapes must agree on color"
        fill_col = next(iter(cmap))
        assert cmap[fill_col] > 0, "piece count for a hole shape must be positive"
        cmap[fill_col] -= 1
        out[pts[:, 0], pts[:, 1]] = int(fill_col)

    for cmap in pieces_by_bucket.values():
        assert all(v == 0 for v in cmap.values()), "every outside piece must be consumed by a hole"

    assert np.array_equal(out, out[:, ::-1]), "filled output must remain LR-symmetric"
    return out.astype(int)


# --------------- generator (combo D3/1 + D13/0) ---------------

_OUT_H_CHOICES = [8, 9, 10]
_OUT_W = 9
_CHAMBER_W = 3

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_FILL_COLOR_MIN = 3
_FILL_COLOR_MAX = 5

_HOLE_PAIR_MIN = 3
_HOLE_PAIR_MAX = 5
_MAX_PIECE_AREA = 5

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_OUTER_BORDER = 1
_SCATTER_PADDING = 1

_LAYOUT_RETRIES = 160
_GEN_MAX_RETRIES = 600


_SHAPE_LIBRARY = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.astype(bool).tobytes())
        if key not in seen:
            seen.add(key)
            out.append(rot.astype(bool))
    return out


def _can_place_shape(chamber: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, shape.shape)
    new_cells = {
        (int(r0 + dr), int(c0 + dc))
        for dr in range(h)
        for dc in range(w)
        if bool(shape[dr, dc])
    }
    for rr, cc in new_cells:
        if chamber[rr, cc]:
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < chamber.shape[0] and 0 <= nc < chamber.shape[1]:
                if chamber[nr, nc] and (nr, nc) not in new_cells:
                    return False
    return True


def _build_hole_mask(H: int, W: int) -> np.ndarray | None:
    chamber_h = H - 2
    center_col = W // 2

    oriented_shapes = []
    for base in _SHAPE_LIBRARY:
        for shape in _all_orientations(base):
            if shape.shape[1] <= _CHAMBER_W and shape.shape[0] <= chamber_h and int(shape.sum()) <= _MAX_PIECE_AREA:
                oriented_shapes.append(shape)

    for _ in range(_LAYOUT_RETRIES):
        left_chamber = np.zeros((chamber_h, _CHAMBER_W), dtype=bool)
        target_pairs = random.randint(_HOLE_PAIR_MIN, _HOLE_PAIR_MAX)
        placed = 0

        for _inner in range(300):
            if placed >= target_pairs:
                break
            shape = random.choice(oriented_shapes)
            sh, sw = map(int, shape.shape)
            r0 = random.randint(0, chamber_h - sh)
            c0 = random.randint(0, _CHAMBER_W - sw)
            if not _can_place_shape(left_chamber, shape, r0, c0):
                continue
            for dr in range(sh):
                for dc in range(sw):
                    if bool(shape[dr, dc]):
                        left_chamber[r0 + dr, c0 + dc] = True
            placed += 1

        if placed != target_pairs:
            continue

        hole_mask = np.zeros((H, W), dtype=bool)
        hole_mask[1 : H - 1, 1 : 1 + _CHAMBER_W] = left_chamber
        right = left_chamber[:, ::-1]
        hole_mask[1 : H - 1, center_col + 1 : center_col + 1 + _CHAMBER_W] = right

        comps = [np.array(pts, dtype=int) for pts in components_mask(hole_mask, connectivity=4)]
        if len(comps) != 2 * target_pairs:
            continue
        if any(len(pts) > _MAX_PIECE_AREA for pts in comps):
            continue

        class_keys = {min(_piece_sig(pts), _mirror_sig(_piece_sig(pts))) for pts in comps}
        if len(class_keys) < _FILL_COLOR_MIN:
            continue
        if not np.array_equal(hole_mask, hole_mask[:, ::-1]):
            continue
        if not hole_mask[:, center_col].sum() == 0:
            continue
        return hole_mask
    return None


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, int]:
    hole_mask = _build_hole_mask(H, W)
    if hole_mask is None:
        return None, 0

    holes = [np.array(pts, dtype=int) for pts in components_mask(hole_mask, connectivity=4)]
    class_keys: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[np.ndarray]] = {}
    for pts in holes:
        sig = _piece_sig(pts)
        canon = min(sig, _mirror_sig(sig))
        class_keys.setdefault(canon, []).append(pts)

    if len(class_keys) < _FILL_COLOR_MIN:
        return None, 0

    n_fill_colors = random.randint(_FILL_COLOR_MIN, min(_FILL_COLOR_MAX, len(class_keys)))
    palette = list(range(1, n_fill_colors + 1))
    random.shuffle(palette)

    class_color: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    keys = list(class_keys.keys())
    random.shuffle(keys)
    for i, key in enumerate(keys):
        class_color[key] = int(palette[i % n_fill_colors])

    frame_color = 0
    out = np.full((H, W), int(frame_color), dtype=int)
    for key, comps in class_keys.items():
        fill_col = int(class_color[key])
        for pts in comps:
            out[pts[:, 0], pts[:, 1]] = fill_col

    assert np.array_equal(out, out[:, ::-1])
    n_colors = len(np.unique(out))
    if not (_N_COLORS_MIN <= n_colors <= _N_COLORS_MAX):
        return None, 0
    return out, frame_color


def _remap_colors(out: np.ndarray, frame_color: int) -> tuple[np.ndarray | None, int, int, int]:
    bg_left, bg_right = random.sample(list(range(10)), 2)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c not in {bg_left, bg_right}]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, 0, 0
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out).astype(int)
    return remapped, int(bg_left), int(bg_right), int(cmap[frame_color])


def _extract_patches(out: np.ndarray, frame_col: int, bg_left: int, bg_right: int) -> list[dict]:
    center_col = out.shape[1] // 2
    items: list[dict] = []
    for col in sorted(map(int, np.unique(out))):
        if int(col) == int(frame_col):
            continue
        for pts in components_mask(out == int(col), connectivity=4):
            arr = np.array(pts, dtype=int)
            side = _classify_zone(arr, center_col)
            bg = int(bg_left if side == "left" else bg_right)
            r0, c0, r1, c1 = bbox_coords(arr)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in arr:
                patch[int(r) - r0, int(c) - c0] = int(col)
            items.append({"patch": patch, "side": side, "area": int((patch != bg).sum())})
    random.shuffle(items)
    items.sort(key=lambda item: (-int(item["area"]), random.random()))
    return items


def _place_patch_in_zone(canvas: np.ndarray, patch: np.ndarray, bg: int, c_start: int, c_end: int) -> None:
    sub = canvas[:, c_start:c_end]
    r, c = random_free_location(
        sub,
        patch,
        transparent=int(bg),
        border=_OUTER_BORDER,
        padding=_SCATTER_PADDING,
        padding_connectivity=8,
    )
    paste(canvas, patch, int(r), int(c_start + c), transparent=int(bg))


def _scatter_input(out: np.ndarray, bg_left: int, bg_right: int, frame_col: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    split = cols // 2
    assert 2 <= split <= cols - 2, "canvas must expose both background zones"

    canvas = np.full((rows, cols), int(bg_right), dtype=int)
    canvas[:, :split] = int(bg_left)

    oh, ow = out.shape
    off_r = random.randint(_OUTER_BORDER, rows - oh - _OUTER_BORDER)
    off_c = split - (ow // 2)
    if off_c < _OUTER_BORDER or off_c + ow > cols - _OUTER_BORDER:
        raise ValueError("frame bbox cannot fit while straddling the split")

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    left_piece_count = 0
    right_piece_count = 0
    for item in _extract_patches(out, frame_col, bg_left, bg_right):
        patch = item["patch"]
        side = item["side"]
        if side == "left":
            _place_patch_in_zone(canvas, patch, bg_left, 0, split)
            left_piece_count += 1
        else:
            _place_patch_in_zone(canvas, patch, bg_right, split, cols)
            right_piece_count += 1

    assert left_piece_count > 0 and right_piece_count > 0

    region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    local_split = split - off_c
    region[:, :local_split] = int(bg_left)
    region[:, local_split:] = int(bg_right)
    region[out == int(frame_col)] = int(frame_col)
    return canvas


def _hole_piece_zone_metric(inp: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(inp, dtype=int)
    split, left_bg, right_bg, bg_mask = _detect_partition_background(g)
    fg_mask = ~bg_mask

    frame_candidates: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g[fg_mask])))):
        for pts in components_mask((g == int(col)) & fg_mask, connectivity=8):
            frame_candidates.append((len(pts), np.array(pts, dtype=int)))
    assert frame_candidates
    max_area = max(area for area, _ in frame_candidates)
    frame_pts = [pts for area, pts in frame_candidates if area == max_area][0]
    r0, c0, r1, c1 = bbox_coords(frame_pts)
    sub = crop_bbox(g, (r0, c0, r1, c1))
    local_split = int(split - c0)

    hole_mask = np.zeros_like(sub, dtype=bool)
    hole_mask[:, :local_split] = (sub[:, :local_split] == int(left_bg))
    hole_mask[:, local_split:] = (sub[:, local_split:] == int(right_bg))
    holes = [np.array(pts, dtype=int) for pts in components_mask(hole_mask, connectivity=4)]
    left_holes = sum(1 for pts in holes if _classify_zone(pts, local_split) == "left")
    right_holes = len(holes) - left_holes

    outside_fg = fg_mask.copy()
    outside_fg[r0 : r1 + 1, c0 : c1 + 1] = False
    pieces = []
    for col in sorted(set(map(int, np.unique(g[outside_fg])))):
        for pts in components_mask((g == int(col)) & outside_fg, connectivity=4):
            pieces.append(np.array(pts, dtype=int))
    left_pieces = sum(1 for pts in pieces if _classify_zone(pts, split) == "left")
    right_pieces = len(pieces) - left_pieces
    return int(left_holes), int(right_holes), int(left_pieces), int(right_pieces)


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_OUT_H_CHOICES)
    W = _OUT_W

    abstract_out, frame_abstract = _build_output(H, W)
    if abstract_out is None:
        return None

    out, bg_left, bg_right, frame_col = _remap_colors(abstract_out, frame_abstract)
    if out is None:
        return None

    try:
        inp = _scatter_input(out, bg_left, bg_right, frame_col)
        got = main(inp)
        lh, rh, lp, rp = _hole_piece_zone_metric(inp)
    except Exception:
        return None

    if not (lh > 0 and rh > 0 and lh == lp and rh == rp):
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
