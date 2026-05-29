# Variant of 7ed72f31
# Dimension: D14 — 反射结果的边界与碰撞处理 (transform)
# Change: 所有反射必须完全落在网格边界内，且仅落在背景格上（无重叠）... → 反射超出边界时环绕到对侧

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, label_mask

# Design analysis
# ---
# variant_spec:
#   preserved_structure:
#     - 背景色仍是出现次数最多的颜色，红色 2 仍是反射标记。
#     - 输入仍只包含原始 payload + 红标记；输出在原图基础上补出反射得到的像素。
#     - 仍保留两类标记：嵌入式红线（水平/垂直）与孤立红点。
#     - 仍要求生成器避免不同对象之间的覆盖与混淆。
#   d14_change:
#     - 原题中反射坐标若越界则直接丢弃；这里改为对行列分别取模，网格按环面处理。
#     - 例：宽度 W=20，列轴在 c=1，payload 像素在 c=4，则镜像列为 2*1-4=-2，输出落在 18 列。
#     - 视觉上，一个本来应落在轴左侧的镜像块会“分裂”到最右侧边缘。
#   behavior_trace:
#     - 原逻辑：水平红线在第 1 行，payload 在第 2..4 行；镜像行是 0,-1,-2，其中负数部分被 inb 过滤掉。
#     - 新逻辑：同一例子中，0 保持 0，-1→H-1，-2→H-2；因此镜像会同时出现在顶部和底部。
#     - 差异仅发生在“计算目标坐标后的边界处理”这一步，输入解析规则不变。
#   complexity_calibration:
#     - 不引入新的颜色语义、排序规则或多阶段变换，避免 D14 之外的额外难度。
#     - 为了让环绕可观察，生成器强制每个样本至少有一个 sprite 的镜像像素发生越界回绕。
#     - 同时保留未回绕 sprite，避免任务退化成“所有对象都贴边”的单一模板。
#   implicit_assumptions_review:
#     - 原 main() 的 inb(rr, cc) 依赖“越界目标应被忽略”；变体下该前提失效，必须改成取模。
#     - 原 main() 对嵌入式 marker 通过扫描整张网格寻找同色 payload；这依赖“同色干扰不会出现在相关投影范围”。
#       在本变体里，镜像可能远距落到另一侧，更需要避免这种隐式假设，因此改成只反射当前 component 内的 payload。
#     - 孤立红点分支仍基于输入中的普通 8 邻域寻找相邻组件，因为环绕只作用于输出坐标，不改变输入对象识别方式。
#   placement_legality:
#     - 生成器按输入局部块放置对象，保证输入对象始终在平面网格中连通、可识别，不跨边界拆开。
#     - 同时预先计算其“环绕后的输出足迹”，并在放置时检查：不与已有输出占用重叠、不与已有对象过近。
#     - 因此 solver 看到的输入总是合法，而输出会稳定出现跨边界镜像。
#   visual_observability:
#     - 人类可见线索是：某些红轴附近缺少一半，但补出的另一半并不只在轴另一侧，而会出现在网格对边。
#     - 这种跨边界补全会在 2-4 个样例中重复出现，足以归纳“越界后从另一边回来”。
#   cleanliness:
#     - 对象之间使用 1 格留白，避免输入组件误合并，也减少输出端的拥挤感。
#     - 不允许镜像与原图、自身红标记或其他 sprite 重叠。

MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

# Embedded payload
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

# Colors
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88
WRAPPED_SPRITES_MIN = 1
WRAPPED_SPRITES_MAX = 3

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 400

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red (2) markers as reflection axes, with toroidal wrapping.

    Input structure:
    - bg = most common color. Red (2) = marker color.
    - Colored sprites appear near red markers.

    Rule:
    - Red line within a sprite (vertical or horizontal) → mirror payload across that axis.
    - Single red pixel in a mixed sprite → point-reflect payload around it.
    - Single isolated red pixel (alone, no payload) → point-reflect each adjacent 8-connected color component.
    - Originals remain; mirrored pixels overlay.
    - New D14 rule: reflected destinations wrap around grid boundaries instead of being discarded.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    def wrap(r: int, c: int) -> tuple[int, int]:
        return int(r) % int(h), int(c) % int(w)

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    # 1) Marker-containing mixed components.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(vals):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload_pixels = [((int(r), int(c)), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not reds or not payload_pixels:
            continue

        rs = {int(r) for r, _ in reds}
        cs = {int(c) for _, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        for (r, c), v in payload_pixels:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))
            rr, cc = wrap(int(rr), int(cc))
            out[int(rr), int(cc)] = int(v)

    # 2) Isolated red pixels reflect adjacent 8-connected components by point reflection.
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
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    rr, cc = wrap(int(rr), int(cc))
                    out[int(rr), int(cc)] = int(v)

    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    hh, ww = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < hh and 0 <= nc < ww and mask[nr, nc]:
            return True
    return False


def _build_embedded_sprite(rng: random.Random) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    payload_coords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
    if rng.random() < 0.5:
        red_at_top = rng.random() < 0.5
        if red_at_top:
            red_coords = [(0, c) for c in range(pw)]
            payload_coords = [(r + 1, c) for r, c in payload_coords]
            local_shape = (ph + 1, pw)
            axis_value = 0
        else:
            red_coords = [(ph, c) for c in range(pw)]
            local_shape = (ph + 1, pw)
            axis_value = ph
        axis = ("row", axis_value)
    else:
        red_at_left = rng.random() < 0.5
        if red_at_left:
            red_coords = [(r, 0) for r in range(ph)]
            payload_coords = [(r, c + 1) for r, c in payload_coords]
            local_shape = (ph, pw + 1)
            axis_value = 0
        else:
            red_coords = [(r, pw) for r in range(ph)]
            local_shape = (ph, pw + 1)
            axis_value = pw
        axis = ("col", axis_value)

    return {
        "kind": "embedded",
        "payload_coords": payload_coords,
        "red_coords": red_coords,
        "axis": axis,
        "local_shape": local_shape,
    }


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = np.ones((ph, pw), dtype=bool) if n_px == ph * pw else _random_connected_mask(ph, pw, n_px, rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
    rng.shuffle(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            all_pts = pcoords + [(red_r, red_c)]
            minr = min(r for r, _ in all_pts)
            minc = min(c for _, c in all_pts)
            maxr = max(r for r, _ in all_pts)
            maxc = max(c for _, c in all_pts)
            hh = maxr - minr + 1
            ww = maxc - minc + 1

            payload_coords = [(r - minr, c - minc) for r, c in pcoords]
            red_pos = (red_r - minr, red_c - minc)
            payload_mask = np.zeros((hh, ww), dtype=bool)
            for r, c in payload_coords:
                payload_mask[r, c] = True

            if payload_mask[red_pos]:
                continue
            if _is_4connected_to(payload_mask, int(red_pos[0]), int(red_pos[1])):
                continue
            return {
                "kind": "isolated",
                "payload_coords": payload_coords,
                "red_coords": [red_pos],
                "axis": ("point", red_pos),
                "local_shape": (hh, ww),
            }
    return None


def _pad_cells(cells: set[tuple[int, int]], n: int, padding: int = PLACEMENT_PADDING) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    padded.add((int(rr), int(cc)))
    return padded


def _wrap_coord(r: int, c: int, n: int) -> tuple[int, int]:
    return int(r) % int(n), int(c) % int(n)


def _placed_input_cells(sprite: dict, r0: int, c0: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    payload = [(int(r0 + r), int(c0 + c)) for r, c in sprite["payload_coords"]]
    reds = [(int(r0 + r), int(c0 + c)) for r, c in sprite["red_coords"]]
    return payload, reds


def _reflected_cells(sprite: dict, payload: list[tuple[int, int]], reds: list[tuple[int, int]], n: int) -> tuple[list[tuple[int, int]], int]:
    reflected = []
    wrapped = 0
    if sprite["axis"][0] == "row":
        axis_r = int(reds[0][0])
        for r, c in payload:
            raw_r = int(2 * axis_r - r)
            raw_c = int(c)
            if not (0 <= raw_r < n and 0 <= raw_c < n):
                wrapped += 1
            reflected.append(_wrap_coord(raw_r, raw_c, n))
    elif sprite["axis"][0] == "col":
        axis_c = int(reds[0][1])
        for r, c in payload:
            raw_r = int(r)
            raw_c = int(2 * axis_c - c)
            if not (0 <= raw_r < n and 0 <= raw_c < n):
                wrapped += 1
            reflected.append(_wrap_coord(raw_r, raw_c, n))
    else:
        pr, pc = reds[0]
        for r, c in payload:
            raw_r = int(2 * pr - r)
            raw_c = int(2 * pc - c)
            if not (0 <= raw_r < n and 0 <= raw_c < n):
                wrapped += 1
            reflected.append(_wrap_coord(raw_r, raw_c, n))
    return reflected, int(wrapped)


def _find_placement(
    sprite: dict,
    n: int,
    reserved_in: set[tuple[int, int]],
    reserved_out: set[tuple[int, int]],
    rng: random.Random,
    *,
    must_wrap: bool,
) -> dict | None:
    hh, ww = sprite["local_shape"]
    if hh > n or ww > n:
        return None

    candidates = []
    for r0 in range(n - hh + 1):
        for c0 in range(n - ww + 1):
            payload, reds = _placed_input_cells(sprite, r0, c0)
            input_cells = set(payload) | set(reds)
            if any(cell in reserved_in for cell in input_cells):
                continue

            reflected, wrapped = _reflected_cells(sprite, payload, reds, n)
            if must_wrap and wrapped == 0:
                continue

            all_out = payload + reds + reflected
            if len(set(all_out)) != len(all_out):
                continue
            if any(cell in reserved_out for cell in set(all_out)):
                continue

            candidates.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "payload": payload,
                    "reds": reds,
                    "reflected": reflected,
                    "wrapped_pixels": int(wrapped),
                }
            )
    if not candidates:
        return None
    return rng.choice(candidates)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    rng = random.Random()
    n = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    sprites: list[dict] = []
    for builder, count in [(_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)]:
        for _ in range(count):
            spr = builder(rng)
            if spr is None:
                return None
            sprites.append(spr)

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)
    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, color in zip(sprites, assignment):
        spr["color"] = int(color)

    wrap_target = min(len(sprites), rng.randint(WRAPPED_SPRITES_MIN, WRAPPED_SPRITES_MAX))
    wrap_indices = set(rng.sample(range(len(sprites)), wrap_target))
    order = list(range(len(sprites)))
    rng.shuffle(order)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved_in: set[tuple[int, int]] = set()
    reserved_out: set[tuple[int, int]] = set()

    placed = []
    total_wrapped_pixels = 0
    wrapped_sprites = 0

    for idx in order:
        spr = sprites[idx]
        placed_info = _find_placement(
            spr,
            n,
            reserved_in,
            reserved_out,
            rng,
            must_wrap=idx in wrap_indices,
        )
        if placed_info is None:
            return None

        payload = placed_info["payload"]
        reds = placed_info["reds"]
        reflected = placed_info["reflected"]

        _paint_cells(inp, payload, spr["color"])
        _paint_cells(out, payload, spr["color"])
        _paint_cells(out, reflected, spr["color"])
        _paint_cells(inp, reds, MARK)
        _paint_cells(out, reds, MARK)

        reserved_in |= _pad_cells(set(payload) | set(reds), n)
        reserved_out |= _pad_cells(set(payload) | set(reds) | set(reflected), n)

        total_wrapped_pixels += int(placed_info["wrapped_pixels"])
        if placed_info["wrapped_pixels"] > 0:
            wrapped_sprites += 1

        placed.append(
            {
                "kind": spr["kind"],
                "color": int(spr["color"]),
                "payload": payload,
                "reds": reds,
                "reflected": reflected,
                "wrapped_pixels": int(placed_info["wrapped_pixels"]),
            }
        )

    if wrapped_sprites == 0 or total_wrapped_pixels == 0:
        return None
    if (inp == bg).sum() / float(n * n) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    stats = {
        "grid_size": int(n),
        "wrapped_sprites": int(wrapped_sprites),
        "total_wrapped_pixels": int(total_wrapped_pixels),
        "embedded_count": int(n_embedded),
        "isolated_count": int(n_isolated),
        "placed": placed,
    }
    return inp, stats


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        item = _gen_attempt()
        if item is not None:
            return item[0]
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")


def _sample_stats(trials: int = 32) -> dict:
    wrapped_sprites = []
    wrapped_pixels = []
    isolated_present = 0
    embedded_present = 0
    attempts = 0
    while len(wrapped_sprites) < trials and attempts < trials * GEN_RETRIES:
        attempts += 1
        item = _gen_attempt()
        if item is None:
            continue
        _, stats = item
        wrapped_sprites.append(int(stats["wrapped_sprites"]))
        wrapped_pixels.append(int(stats["total_wrapped_pixels"]))
        if int(stats["isolated_count"]) > 0:
            isolated_present += 1
        if int(stats["embedded_count"]) > 0:
            embedded_present += 1
    return {
        "samples": len(wrapped_sprites),
        "attempts": int(attempts),
        "wrapped_sprites": wrapped_sprites,
        "wrapped_pixels": wrapped_pixels,
        "isolated_present": int(isolated_present),
        "embedded_present": int(embedded_present),
    }
