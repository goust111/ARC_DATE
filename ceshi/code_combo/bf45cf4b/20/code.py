# Combo variant of bf45cf4b (index 20)
# combo_spec: D3/0+D15/1
# Name: 不对称瓦片砖墙铺贴

from __future__ import annotations

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ImportError:  # pragma: no cover - fallback for alternate PYTHONPATH layouts
    from solvers.arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


# design_analysis:
#   a_concretization_and_trace:
#     - 输入仍只有两个 8 连通前景分量：一个单色 control sprite，一个多色 square tile。
#     - 原题输出把 ctrl bbox 中每个实心像素扩成一个 tile，tile 左上角落在 `(r*th, c*tw)`。
#     - 融合后仍按 ctrl 的 bbox 逐格展开，但行布局改成“砖墙”：
#       偶数逻辑行用原位置，奇数逻辑行整体右移 `tw/2`。
#     - 具体例子：若 `tile` 是 4x4，`ctrl` 是 2x3，原题输出宽度是 12。
#       第 0 行的第 2 个 tile 仍放在列 `8..11`；第 1 行的第 2 个 tile 改放在列 `10..13`，
#       但输出窗口仍只有 12 列，所以 `12..13` 被裁掉，只保留右边界内的半块。
#     - D3/0 的作用不是改变 solver 判别逻辑，而是移除原先“对称 tile”这个弱先验。
#       若 tile 继续强制 LR/UD 对称，右边界被裁掉的半块视觉信息会变弱；因此 generator 要显式
#       造出左右、上下都不对称的多色 tile。
#
#   b_complexity_calibration:
#     - 不引入第三对象、额外标记或颜色映射，变化只集中在两个地方：
#       1) tile 变成无对称约束的多色方块；
#       2) 输出改成奇偶行半块错位。
#     - 为避免“看起来只是普通平移”的弱样例，generator 强制：
#       1) tile 边长为偶数；
#       2) 至少一个奇数行最右列 ctrl 像素为实心，从而必然发生右边界裁剪；
#       3) 至少一个奇数行有别的实心像素，能看出整行错位而非单点偶然。
#
#   c_implicit_assumption_audit_of_original_main:
#     - `find_components(...)=2`：仍成立，generator 继续只放 tile 和 ctrl。
#     - “多色者是 tile、单色者是 ctrl”：仍成立，是 solver 的主判别线索。
#     - 原排序里“偏好 square”仍适用，因为 tile 仍是 square；ctrl 仍是单色矩形带洞。
#     - 原 `out = np.full((ch*th, cw*tw), ...)` 的尺寸前提仍可保留，但语义变了：
#       它现在是固定的“墙面窗口”，不是所有 tile 都能完整落入的无限平铺画布。
#     - 原 `paste(out, block, r*th, c*tw)` 依赖“零水平偏移、无裁剪”，在本变体下失效；
#       必须改成 `c*tw + (r%2)*(tw//2)` 并启用 `clip=True`。
#     - 原代码对 tile 对称性没有显式依赖；融合后 solver 也不应引入这种依赖，避免把 D3/0
#       错做成识别规则的一部分。
#
#   d_placement_legality:
#     - 输入生成仍把 tile 与 ctrl 放进 12..18 的背景网格，保证不重叠、不越界、彼此至少隔 1 个
#       背景像素，因此不会在 8 连通下合并。
#     - 输出装配时允许 tile 在右边界裁剪，但不会访问越界，因为统一用 `paste(..., clip=True)`。
#
#   e_visual_observability:
#     - 人类从 2-4 组样例能直接看到：同一 tile 在奇数行横向错半块，且最右端会出现“被截断的砖”。
#     - tile 的非对称内部花纹让“半块截断”更明显，不会被左右镜像外观掩盖。
#
#   f_visual_cleanliness:
#     - 输入仍是两个清晰分离的对象，没有叠加。
#     - 输出只包含 tile 拷贝和背景；半块裁剪只发生在右边界，不会产生多层覆盖或交错歧义。
#
#   g_change_inferability:
#     - 没有颜色重映射。颜色只承担“记住 tile 内部像素布局”的作用。
#     - 关键可推断线索是：奇数逻辑行整体偏移固定为半个 tile 宽，而不是按颜色、大小或位置随机变化。
#
#   h_new_main_assumptions_and_generator_guarantees:
#     - solver 新假设 1：tile 是偶数边长 square。generator 只生成 `2x2` 或 `4x4`。
#     - solver 新假设 2：ctrl 仍是单色。generator 保持单色并保留洞。
#     - solver 新假设 3：至少存在奇数逻辑行，否则砖墙行为可能退化。generator 令 ctrl 高度 >= 4。
#     - solver 新假设 4：边界裁剪应可观察。generator 强制奇数行最后一列至少有一个实心像素，并审计
#       变体输出必须不同于原题紧密网格输出。


def _split_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        raise ValueError(f"expected exactly 2 foreground components, got {len(comps)}")

    infos: list[dict[str, object]] = []
    for comp in comps:
        patch = np.array(crop_bbox(grid, comp["bbox"]), dtype=int)
        cols = colors_present(patch, exclude={bg})
        infos.append(
            {
                "patch": patch,
                "n_cols": len(cols),
                "is_square": int(patch.shape[0] == patch.shape[1]),
            }
        )

    infos.sort(key=lambda d: (int(d["n_cols"] >= 2), int(d["n_cols"]), int(d["is_square"])), reverse=True)
    tile = np.array(infos[0]["patch"], dtype=int)
    ctrl = np.array(infos[1]["patch"], dtype=int)
    return tile, ctrl


def _assemble_tight(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _assemble_brick(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    if th != tw:
        raise ValueError("tile must be square")
    if tw % 2 != 0:
        raise ValueError("tile width must be even for half-tile offset")

    shift = tw // 2
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        row_shift = shift if (r % 2 == 1) else 0
        for c in range(cw):
            if int(ctrl[r, c]) == int(bg):
                continue
            paste(out, tile, r * th, c * tw + row_shift, clip=True)
    return out


def _scene_metrics(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> dict[str, int]:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    shift = tw // 2
    odd_rows = [r for r in range(1, ch, 2)]
    shifted_row_tiles = sum(int(np.count_nonzero(ctrl[r] != int(bg))) for r in odd_rows)
    clipped_tiles = sum(int(ctrl[r, cw - 1] != int(bg)) for r in odd_rows)
    return {
        "tile_size": int(th),
        "half_shift": int(shift),
        "ctrl_h": int(ch),
        "ctrl_w": int(cw),
        "shifted_row_tiles": int(shifted_row_tiles),
        "clipped_tiles": int(clipped_tiles),
        "tight_equal": int(np.array_equal(_assemble_brick(tile, ctrl, bg), _assemble_tight(tile, ctrl, bg))),
    }


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tile, ctrl = _split_objects(g, bg)
    return _scene_metrics(tile, ctrl, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the monochrome control sprite using the multicolor square tile.

    Even logical rows place tiles on the original grid.
    Odd logical rows shift every tile right by half the tile width.
    The output window keeps the original width `ctrl_w * tile_w`, so shifted
    tiles may be clipped at the right boundary.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tile, ctrl = _split_objects(g, bg)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if len(ctrl_cols) != 1:
        raise ValueError("control sprite must be monochrome")

    return _assemble_brick(tile, ctrl, bg)


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _make_asymmetric_even_tile(tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    for _ in range(800):
        size = int(rng.choice(np.array([2, 4], dtype=int)))
        n_colors = int(rng.integers(2, min(4, len(tile_colors)) + 1))
        colors = [int(x) for x in rng.choice(tile_colors, size=n_colors, replace=False)]
        tile = rng.choice(colors, size=(size, size), replace=True).astype(int)

        flat = tile.reshape(-1)
        for idx, color in enumerate(colors):
            flat[idx] = int(color)
        rng.shuffle(flat)
        tile = flat.reshape(size, size)

        if len(colors_present(tile)) < 2:
            continue
        if np.array_equal(tile, np.fliplr(tile)):
            continue
        if np.array_equal(tile, np.flipud(tile)):
            continue
        if np.array_equal(tile[:, : size // 2], tile[:, size // 2 :]):
            continue
        if len(set(int(x) for x in tile[:, -1])) < 2 and np.array_equal(tile[:, -1], tile[:, 0]):
            continue
        return tile
    raise RuntimeError("failed to generate asymmetric even tile")


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(1500):
        h = int(rng.integers(4, 7))
        w = int(rng.integers(4, 7))
        area = h * w
        min_fill = max(h + w - 1, int(np.ceil(0.45 * area)))
        max_fill = min(area - 1, int(np.floor(0.72 * area)))
        if min_fill > max_fill:
            continue
        target = int(rng.integers(min_fill, max_fill + 1))

        anchor_r = int(rng.integers(0, h))
        anchor_c = int(rng.integers(0, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, anchor_c] = True
        mask[anchor_r, :] = True

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            pick = frontier[int(rng.integers(0, len(frontier)))]
            mask[pick] = True

        if int(mask.sum()) != target:
            continue
        if int(mask.sum()) == area:
            continue
        if not mask[0].any():
            continue
        if not mask[-1].any():
            continue
        if not mask[:, 0].any():
            continue
        if not mask[:, -1].any():
            continue

        odd_rows = [r for r in range(1, h, 2)]
        if not odd_rows:
            continue
        if not any(mask[r].any() for r in odd_rows):
            continue
        if not any(mask[r, -1] for r in odd_rows):
            continue

        fill_ratio = float(mask.mean())
        if not (0.45 <= fill_ratio <= 0.72):
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)
        return ctrl
    raise RuntimeError("failed to generate control sprite with clipped odd-row endpoint")


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = map(int, left_obj.shape)
        rh, rw = map(int, right_obj.shape)
        H = int(rng.integers(max(12, max(lh, rh)), 19))
        W = int(rng.integers(max(12, lw + rw + gap), 19))
        slack = W - (lw + rw + gap)
        c_left = int(rng.integers(0, slack + 1))
        c_right = int(rng.integers(c_left + lw + gap, W - rw + 1))
        r_left = int(rng.integers(0, H - lh + 1))
        r_right = int(rng.integers(0, H - rh + 1))
        if left_obj is tile:
            tile_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            tile_pos = (r_right, c_right)
    else:
        if rng.random() < 0.5:
            top_obj, bot_obj = tile, ctrl
        else:
            top_obj, bot_obj = ctrl, tile
        th, tw = map(int, top_obj.shape)
        bh, bw = map(int, bot_obj.shape)
        H = int(rng.integers(max(12, th + bh + gap), 19))
        W = int(rng.integers(max(12, max(tw, bw)), 19))
        slack = H - (th + bh + gap)
        r_top = int(rng.integers(0, slack + 1))
        r_bot = int(rng.integers(r_top + th + gap, H - bh + 1))
        c_top = int(rng.integers(0, W - tw + 1))
        c_bot = int(rng.integers(0, W - bw + 1))
        if top_obj is tile:
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_colors = [int(x) for x in palette[2:]]

    tile = _make_asymmetric_even_tile(tile_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    metrics = _scene_metrics(tile, ctrl, bg)
    if metrics["clipped_tiles"] <= 0 or metrics["shifted_row_tiles"] <= 0:
        return None

    expected = _assemble_brick(tile, ctrl, bg)
    if metrics["tight_equal"] != 0:
        return None

    inp = _make_input(tile, ctrl, bg, rng)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    color_counts: list[int] = []
    for comp in comps:
        patch = np.array(crop_bbox(inp, comp["bbox"]), dtype=int)
        color_counts.append(len(colors_present(patch, exclude={bg})))
    if sorted(color_counts) != [1, len(colors_present(tile))]:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate input for bf45cf4b combo D3/0+D15/1")
