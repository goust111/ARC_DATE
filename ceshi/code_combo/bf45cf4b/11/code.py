# Combo variant of bf45cf4b (index 11)
# combo_spec: D1/3+D3/0
# Name: 不对称瓦片位置渐变旋转

from __future__ import annotations

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ImportError:  # pragma: no cover - fallback for alternate PYTHONPATH layouts
    from solvers.arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


# design_analysis:
#   a_concretization_and_trace:
#     - 原题输入仍是两个 8 连通前景分量：一个单色控制精灵 `ctrl`，一个多色正方形瓦片 `tile`。
#     - 原题输出对 ctrl 的每个填充像素都贴同一个 tile；融合后仍逐像素贴 tile，但旋转角度由
#       ctrl bbox 内的离散坐标决定：phase = (row + col) % 4，贴 `rot90(tile, phase)`。
#     - 例子：若 ctrl 的左上角填充像素在 (0,0)，原题与新题都贴原 tile；若右邻像素在 (0,1)，
#       原题仍贴原 tile，新题贴 90 度旋转版本；若再往右到 (0,2)，则贴 180 度版本。
#     - D3/0 的配合作用是关键：如果 tile 仍是 LR/UD 对称，很多旋转会肉眼不可见，D1/3 会退化。
#       因此生成器必须显式制造“不具 LR/UD 对称、四个旋转互异”的方形瓦片。
#
#   b_complexity_calibration:
#     - 不能把 tile 变成任意噪声大矩阵，否则输入太乱；这里仍限制为 3..5 的小方块、3..4 色。
#     - 不能让 ctrl 只覆盖 1-2 个 phase，否则看不出 `(r+c)%4` 渐变；生成器强制至少 3 个 phase
#       类别出现，且至少两个非零 phase 出现多次。
#     - 也不把自由度做得过多：只有一个位置公式 `(r+c)%4`，不存在按颜色或按对象再分支。
#
#   c_implicit_assumption_audit_of_original_main:
#     - `find_components(...)=2`：仍成立，生成器继续保证仅有 tile 和 ctrl 两个前景分量。
#     - “多色者是 tile、单色者是 ctrl”：仍成立，且比原题更重要，因为 D3/0 去掉了对称性线索。
#     - “tile 只需方形即可”：仍成立；但“tile 可能对称所以旋转无所谓”不再成立，因此 solver
#       不能依赖对称性做任何简化，generator 也必须禁止旋转退化。
#     - 原 `paste(out, block, ...)` 隐含“所有填充像素贴同一块”前提；融合后失效，必须在循环中按
#       `(r+c)%4` 选不同旋转版本。
#
#   d_placement_legality:
#     - 生成器仍先造 tile / ctrl，再放入 12..18 的背景网格。
#     - 放置使用横向或纵向分离，最少 1 格背景间隔，避免 8 连通误合并。
#     - tile 采用满方形前景，不会因边缘背景导致 crop 后不再是正方形。
#
#   e_visual_observability:
#     - 解题者能在输出里看到同一 tile 沿 ctrl 的行列位置系统性转向，且相邻位置常常旋转 90 度。
#     - 由于 tile 四个旋转互异，2-4 组样例足以看出是“位置控制旋转”，不是随机方向。
#
#   f_visual_cleanliness:
#     - 输入仍保持原题的干净布局：一个 control，一个 tile，没有额外标记或第三物体。
#     - 输出只是紧凑拼贴，不引入重叠层；每个 tile 块边界天然清晰。
#
#   g_change_inferability:
#     - 没有颜色映射，颜色只随几何旋转移动，人类可直接从局部图案方向判断规则。
#     - 为避免样例退化回原题，生成器还会审计 `main(inp)` 与“原题不旋转版本输出”必须不同。
#
#   h_new_main_assumptions_and_generator_guarantees:
#     - solver 新假设 1：tile 是正方形。generator 保证 tile 尺寸为 3..5 且高宽相等。
#     - solver 新假设 2：ctrl 是单色。generator 保持单色 control。
#     - solver 新假设 3：tile 的旋转可见。generator 保证四个旋转版本两两不同，且非 LR/UD 对称。
#     - solver 新假设 4：ctrl 的填充像素覆盖多个 phase。generator 检查 phase 分布并拒绝退化样本。


def _split_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        raise ValueError(f"expected exactly 2 foreground components, got {len(comps)}")

    infos: list[dict[str, object]] = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
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


def _rotate_tile(tile: np.ndarray, phase: int) -> np.ndarray:
    return np.rot90(tile, k=int(phase)).copy()


def _phase_hist_from_ctrl(ctrl: np.ndarray, bg: int) -> dict[int, int]:
    hist = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in range(ctrl.shape[0]):
        for c in range(ctrl.shape[1]):
            if int(ctrl[r, c]) != int(bg):
                hist[(int(r) + int(c)) % 4] += 1
    return hist


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, *, rotate_by_phase: bool) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == int(bg):
                continue
            patch = _rotate_tile(tile, (r + c) % 4) if rotate_by_phase else tile
            paste(out, patch, r * th, c * tw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the monochrome control sprite by stamping a square multicolor tile.
    Filled cell (r,c) in the control bbox receives the tile rotated by
    `((r + c) % 4) * 90` degrees; hole cells expand to background.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tile, ctrl = _split_objects(g, bg)

    if tile.shape[0] != tile.shape[1]:
        raise ValueError("tile must be square in this combo variant")

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if len(ctrl_cols) != 1:
        raise ValueError("control sprite must be monochrome")

    return _compose_output(tile, ctrl, bg, rotate_by_phase=True)


def _make_asymmetric_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    for _ in range(800):
        size = int(rng.integers(3, 6))
        n_colors = int(rng.integers(3, min(4, len(tile_colors)) + 1))
        colors = [int(x) for x in rng.choice(tile_colors, size=n_colors, replace=False)]
        tile = rng.choice(colors, size=(size, size), replace=True).astype(int)

        if len(colors_present(tile, exclude={bg})) < 3:
            continue
        rotations = [_rotate_tile(tile, k) for k in range(4)]
        if len({arr.tobytes() for arr in rotations}) != 4:
            continue
        if np.array_equal(tile, np.fliplr(tile)):
            continue
        if np.array_equal(tile, np.flipud(tile)):
            continue
        return tile
    raise RuntimeError("failed to generate a visibly asymmetric tile")


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


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(1500):
        h = int(rng.integers(4, 7))
        w = int(rng.integers(4, 7))
        area = h * w
        min_fill = max(6, int(np.ceil(0.42 * area)))
        max_fill = min(area - 1, int(np.floor(0.72 * area)))
        if min_fill > max_fill:
            continue
        target = int(rng.integers(min_fill, max_fill + 1))

        ar = int(rng.integers(0, h))
        ac = int(rng.integers(0, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, ac] = True
        mask[ar, :] = True

        while int(mask.sum()) < target:
            frontier = []
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

        fill = int(mask.sum())
        if fill != target:
            continue
        if fill == area:
            continue

        hist = {k: v for k, v in _phase_hist_from_ctrl(mask.astype(int), 0).items() if v > 0}
        nonzero_phases = sum(1 for k, v in hist.items() if k != 0 and v > 0)
        repeated_nonzero = sum(1 for k, v in hist.items() if k != 0 and v >= 2)
        fill_ratio = fill / area

        if len(hist) < 3:
            continue
        if nonzero_phases < 2:
            continue
        if repeated_nonzero < 1:
            continue
        if not (0.42 <= fill_ratio <= 0.72):
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)
        return ctrl
    raise RuntimeError("failed to generate control sprite with enough phase variety")


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
    tile_colors = palette[2:]

    tile = _make_asymmetric_tile(bg, tile_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    expected = _compose_output(tile, ctrl, bg, rotate_by_phase=True)
    baseline = _compose_output(tile, ctrl, bg, rotate_by_phase=False)
    if np.array_equal(expected, baseline):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    recovered_tile, recovered_ctrl = _split_objects(inp, bg)
    if recovered_tile.shape[0] != recovered_tile.shape[1]:
        return None
    if len(colors_present(recovered_ctrl, exclude={bg})) != 1:
        return None
    if len(colors_present(recovered_tile, exclude={bg})) < 2:
        return None

    hist = _phase_hist_from_ctrl(recovered_ctrl, bg)
    if len([k for k, v in hist.items() if v > 0]) < 3:
        return None
    if sum(hist[k] for k in (1, 2, 3)) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid combo input")


def audit_variant_behavior(trials: int = 64) -> dict[str, object]:
    """
    Quantify that generated samples really use the combo behavior instead of
    collapsing to the original task.
    """
    phase_class_counts: list[int] = []
    changed_tile_counts: list[int] = []
    same_as_baseline = 0
    lr_symmetric_tiles = 0
    ud_symmetric_tiles = 0

    for _ in range(int(trials)):
        inp = generate_input()
        bg = int(get_bg(inp))
        tile, ctrl = _split_objects(inp, bg)
        hist = _phase_hist_from_ctrl(ctrl, bg)
        phase_class_counts.append(sum(1 for v in hist.values() if v > 0))
        changed_tile_counts.append(hist[1] + hist[2] + hist[3])
        same_as_baseline += int(
            np.array_equal(
                _compose_output(tile, ctrl, bg, rotate_by_phase=True),
                _compose_output(tile, ctrl, bg, rotate_by_phase=False),
            )
        )
        lr_symmetric_tiles += int(np.array_equal(tile, np.fliplr(tile)))
        ud_symmetric_tiles += int(np.array_equal(tile, np.flipud(tile)))

    return {
        "trials": int(trials),
        "min_phase_classes": int(min(phase_class_counts)),
        "max_phase_classes": int(max(phase_class_counts)),
        "avg_phase_classes": float(np.mean(phase_class_counts)),
        "min_changed_tiles": int(min(changed_tile_counts)),
        "avg_changed_tiles": float(np.mean(changed_tile_counts)),
        "same_as_baseline": int(same_as_baseline),
        "lr_symmetric_tiles": int(lr_symmetric_tiles),
        "ud_symmetric_tiles": int(ud_symmetric_tiles),
    }
