# Combo variant of e8686506 (index 24)
# combo_spec: D4/0+D18/2
# Name: 上下对称+横向条带排列
#
# Design analysis
# - concretization_and_trace:
#   - 原题输入里有保持相对位置的 frame，solver 先靠 frame 锁定一个二维矩形，再在矩形内部做
#     左右镜像约束下的二维拼装。
#   - D18/2 把“框架+孔洞”改成“每个碎片都是一条整宽横条”，二维装配退化为按 y 顺序堆叠。
#   - D4/0 再把镜像轴改成水平轴，所以输出不再是“左半镜像右半”，而是“上半条带序列镜像下半”。
#   - 具体例子：若输入横条按从上到下依次是
#       [A, h=1], [B, h=2], [C, h=1], [D, h=1], [C, h=1], [B, h=2], [A, h=1]
#     且它们都宽 5，则输出就是把这些条带压紧后堆成一个 9x5 的稠密矩形。
#     原题会去寻找 frame 和孔洞形状；融合后只需要识别“共同宽度 + 垂直顺序 + 上下回文序列”。
# - complexity_calibration:
#   - 若允许偶数高度，则上下镜像的中轴两侧会出现同色镜像条带首尾相接，直接并成一个更高组件，
#     破坏“每个条带对应一个碎片”。因此 generator 固定使用奇数高度，并让中心始终是一条独立中轴条带。
#   - 若使用 5x5 或 7x5，高度太小，能容纳的镜像条带对太少，颜色角色上限只有 3-4，视觉上过于单薄。
#     这里固定使用 9x5：既保留原任务的 portrait 因子推断，又能容纳 3-4 对镜像条带加 1 条中心条带。
#   - 条带高度只取 1 或 2；多数样本至少含一对高度 2 的镜像条带，避免退化为“单纯压缩 9 行”。
# - hidden_assumptions_audit:
#   - 原 solver 的 `_build_requirements()`、frame 搜索、2D 放置搜索都依赖“存在轮廓碎片”和“局部几何咬合”；
#     这些前提在条带变体里全部失效，必须整体移除，而不是局部打补丁。
#   - 新 `main()` 依赖的前提改为：
#     1. 每个非背景组件都是实心矩形横条；
#     2. 所有横条共享同一宽度；
#     3. 输入中的垂直顺序就是输出的条带顺序；
#     4. 压紧后的条带序列关于水平轴回文对称。
#     generator 会逐条保证这些前提。
#   - 保留原题“总面积决定输出尺寸、且最近方形 portrait 因子唯一”的精神：solver 仍会校验
#     `total_area -> (H, W)` 的最近方形分解唯一，只是 `W` 现在由共同条带宽度直接读出。
# - placement_legality:
#   - 输出先生成，再散布到 15x13-15 画布；所有条带都放在背景上，不重叠、不越界。
#   - 通过严格递增的起始行保留可见的 y 顺序；当两条带之间没有空白行时，尽量让它们在 x 方向不重叠，
#     减少视觉粘连。
# - observability:
#   - 人类看到 2-4 组样例时，会直接发现输入全是同宽横条，输出则把这些横条压紧成一个矩形。
#   - 同时，上半条带序列与下半条带序列颜色和高度成镜像回文，中间还有一条独立中心条带；
#     这正是“横向条带排列”和“上下对称”融合后的可见规律。
#   - 规则完全基于可见几何与顺序，不使用颜色编号大小、扫描顺序等人类不可见信号。
# - visual_cleanliness:
#   - 不再混入 frame、角点或侧边锚，避免不同机制叠加。
#   - 条带宽度固定、间距受控、颜色角色数量仅 4 或 5，输出是清晰的分层带状结构。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


_WIDTH_REQUIRED = 5
_HEIGHT_REQUIRED = 9
_VALID_DIMS = [(_HEIGHT_REQUIRED, _WIDTH_REQUIRED)]
_DIM_WEIGHTS = [_HEIGHT_REQUIRED * _WIDTH_REQUIRED]

_N_COLORS_MIN = 4
_N_COLORS_MAX = 5
_STRIPE_COUNT_MIN = 7
_STRIPE_COUNT_MAX = 9
_CANVAS_MIN = 15
_CANVAS_MAX = 15
_GEN_MAX_RETRIES = 300

_TOP_HEIGHT_PATTERNS = [
    (1, 1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (2, 1, 1),
]
_TOP_HEIGHT_WEIGHTS = [0.4, 0.2, 0.2, 0.2]


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _read_bands(grid: np.ndarray, bg: int) -> list[dict]:
    bands = []
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in components_mask(grid == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            area = int(len(pts))
            bands.append(
                {
                    "col": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": h,
                    "w": w,
                    "area": area,
                }
            )
    return bands


def _infer_dims(bands: list[dict]) -> tuple[int, int]:
    widths = {b["w"] for b in bands}
    assert len(widths) == 1, "all bars must share one common width"
    width = next(iter(widths))

    total_area = sum(b["area"] for b in bands)
    assert total_area % width == 0, "total stripe area must be divisible by the common width"
    height = total_area // width

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    assert best_dims[0] == (height, width), "common stripe width must match the unique output factorization"
    return int(height), int(width)


def _stack_bands(bands: list[dict], width: int) -> np.ndarray:
    total_area = sum(b["area"] for b in bands)
    height = total_area // width
    out = np.empty((height, width), dtype=int)
    cursor = 0
    for band in sorted(bands, key=lambda b: (b["r0"], b["c0"])):
        out[cursor : cursor + band["h"], :] = band["col"]
        cursor += band["h"]
    assert cursor == height, "stacked stripe heights must fill the output exactly"
    return out


def _output_component_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack scattered horizontal bars into a dense rectangle whose stripe order is mirror-symmetric top-to-bottom.

    Input structure:
    - The dominant color is background; every remaining component is a solid horizontal bar.
    - All bars share one hidden common width, while their heights are 1 or 2 and their colors form a symmetric role sequence.
    - The bars are scattered apart, but their vertical order in the input is the order they should occupy after compression.

    Rule:
    - Extract every non-background component, verify that each one is a filled horizontal rectangle, and infer the common stripe width.
    - Sort the bars by their vertical position and stack them tightly in that order, preserving each bar's height and color.
    - The resulting dense rectangle must satisfy exact up-down mirror symmetry.

    Key insight:
    - The original puzzle used 2D contour matching around a frame; this fused variant collapses the geometry to a 1D stripe sequence.
    - Once the shared width is known, total area determines the rectangle height, and the visible y-order of the bars reveals a palindromic top/center/bottom stripe layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bands = _read_bands(g, bg)
    assert bands, "grid must contain at least one non-background bar"

    H, W = _infer_dims(bands)
    for band in bands:
        assert band["area"] == band["h"] * band["w"], "each piece must be a solid rectangle"
        assert band["w"] == W, "every bar must span the shared stripe width"
        assert band["h"] in (1, 2), "stripe heights must stay within the supported 1-2 range"
        assert band["w"] >= band["h"], "pieces must be horizontal bars, not tall columns"

    out = _stack_bands(bands, W)
    assert out.shape == (H, W), "stacked bars must realize the inferred output dimensions"
    assert np.array_equal(out, out[::-1, :]), "stacked stripe rectangle must be UD-symmetric"
    return out.astype(int)


def _choose_top_heights() -> list[int]:
    return list(random.choices(_TOP_HEIGHT_PATTERNS, weights=_TOP_HEIGHT_WEIGHTS, k=1)[0])


def _build_output() -> tuple[np.ndarray, int, list[dict]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    top_heights = _choose_top_heights()
    heights = top_heights + [1] + list(reversed(top_heights))

    n_roles = len(top_heights) + 1
    assert _N_COLORS_MIN <= n_roles <= _N_COLORS_MAX

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    role_colors = palette[:n_roles]
    top_colors = role_colors[:-1]
    center_color = role_colors[-1]
    colors = top_colors + [center_color] + list(reversed(top_colors))

    out = np.empty((H, W), dtype=int)
    stripes = []
    cursor = 0
    for h, col in zip(heights, colors):
        out[cursor : cursor + h, :] = col
        stripes.append({"col": int(col), "h": int(h), "w": int(W)})
        cursor += h
    assert cursor == H
    return out, bg, stripes


def _choose_column(cols: int, width: int, prev_interval: tuple[int, int] | None) -> int:
    choices = list(range(cols - width + 1))
    random.shuffle(choices)
    if prev_interval is not None:
        p0, p1 = prev_interval
        separated = [c for c in choices if (c + width - 1) < p0 or c > p1]
        if separated:
            return separated[0]
    return choices[0]


def _scatter(stripes: list[dict], bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(13, 15)
    total_h = sum(s["h"] for s in stripes)
    base_between = 1 if len(stripes) <= 7 else 0
    slack = rows - total_h - base_between * (len(stripes) - 1)
    if slack < 0:
        raise ValueError("canvas is too short to preserve stripe order")

    gaps = [0] * (len(stripes) + 1)
    for i in range(1, len(stripes)):
        gaps[i] = base_between
    for _ in range(slack):
        gaps[random.randint(0, len(stripes))] += 1

    canvas = np.full((rows, cols), bg, dtype=int)
    r = gaps[0]
    prev_r1 = None
    prev_interval = None
    for idx, stripe in enumerate(stripes):
        adjacent_to_prev = prev_r1 is not None and r == prev_r1 + 1
        c = _choose_column(cols, stripe["w"], prev_interval if adjacent_to_prev else None)
        canvas[r : r + stripe["h"], c : c + stripe["w"]] = stripe["col"]
        prev_r1 = r + stripe["h"] - 1
        prev_interval = (c, c + stripe["w"] - 1)
        r += stripe["h"] + gaps[idx + 1]
    return canvas


def _gen_attempt() -> np.ndarray | None:
    out, bg, stripes = _build_output()
    if not np.array_equal(out, out[::-1, :]):
        return None
    n_colors = len(np.unique(out))
    if not (_N_COLORS_MIN <= n_colors <= _N_COLORS_MAX):
        return None
    stripe_count = _output_component_count(out)
    if not (_STRIPE_COUNT_MIN <= stripe_count <= _STRIPE_COUNT_MAX):
        return None

    try:
        inp = _scatter(stripes, bg)
        got = main(inp)
    except Exception:
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
