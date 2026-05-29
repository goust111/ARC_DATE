# Variant of 7ed72f31
# Dimension: D2 — 反射轴的定义方式 (src)
# Change: 红色(2)像素作为标记：嵌入式红线（构成精灵4连通分量的一条边）定义轴对称轴，孤立红像素（4连通无其他像素）定义点反射中心
#         → 双色标记系统：嵌入式单像素标记中，红色(2)表示水平反射轴，蓝色(1)表示垂直反射轴；不再使用点反射。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import STRUCT8, get_bg, label_mask


"""
design_analysis:
  variant_summary:
    kept:
      - 仍然是 output-first 生成：先构造完整输出，再擦掉镜像半边得到输入。
      - 仍然使用 4 连通单色 payload 小精灵，镜像结果覆盖到背景上。
      - 仍然保持 18-28 的方形画布、较高背景占比、对象之间留空防止组件合并。
    changed:
      - 删除原题中 "孤立红点 -> 点反射" 的整条分支。
      - 删除原题中 "红线几何方向决定横/竖轴" 的判别。
      - 新规则只保留嵌入式单像素标记：红(2)表示经过该像素所在行的水平轴；蓝(1)表示经过该像素所在列的垂直轴。

  a_concretization_and_trace:
    input_appearance:
      - 每个对象是一个 4 连通单色 payload，加上紧贴 payload 的一个单像素标记。
      - 红标记嵌在 payload 的上边或下边，只和 payload 的一个边界像素 4 邻接；输出时 payload 会沿该红像素所在行上下镜像。
      - 蓝标记嵌在 payload 的左边或右边，只和 payload 的一个边界像素 4 邻接；输出时 payload 会沿该蓝像素所在列左右镜像。
      - 标记像素自己保留不动；原始 payload 也保留，只有镜像拷贝会被补画出来。
    flow_impact:
      - 原题在 solver 中先收集红像素，再靠 "单点 / 同行 / 同列" 三种几何形状选择 point / row / col。
      - 变体中 solver 不再看标记几何形状，而是要求组件里恰有一个标记像素，然后按颜色直接选 row 或 col。
      - 原题 generator 会混合两类对象：嵌入式红线和孤立红点。变体 generator 只生成嵌入式单点标记对象，且同一网格里同时出现红标记和蓝标记。
    trace_example:
      - 例：某个组件里唯一标记是红色，坐标 (6, 9)，payload 在其上方。
      - 原题逻辑：单个红点会被判成 point reflection，像素 (5, 8) 会去到 (7, 10)。
      - 变体逻辑：红色定义水平轴 row=6，像素 (5, 8) 会去到 (7, 8)。
      - 差异：旧逻辑同时翻转行和列；新逻辑只翻转行，且方向完全由颜色决定。

  b_complexity_calibration:
    choices:
      - 只保留两种轴类型：水平、垂直。没有再引入第三种颜色或多种子规则，避免样例不足时难以归纳。
      - 每个网格至少 3 个标记，且至少 1 个红标记和 1 个蓝标记，让颜色-方向对应在单个样例里就有重复支撑。
      - 标记全部是单像素，避免求解者继续依赖 "线 vs 点" 的旧启发式。
    why_not_too_simple:
      - 虽然只剩两类反射，但颜色是必须观察的新控制变量；旧题的几何判别在这里会系统性出错。
      - payload 形状保持不对称随机连接块，保证水平镜像和垂直镜像在视觉上都可见。

  c_hidden_assumption_audit:
    original_code_dependencies:
      - `MARK = 2`: 原题默认只有一种标记色。变体不成立，必须改成 `HMARK=2` 和 `VMARK=1`，并把两者都从 payload 颜色集中排除。
      - `if len(reds) == 1: axis = ("point", ...)`: 原题把单个红点直接解释为点反射中心。变体不成立；单点现在是常态，而且要继续区分横/竖轴。
      - `elif len(rs) == 1` / `elif len(cs) == 1`: 原题靠红线形状区分横轴或竖轴。变体中标记几何不再携带这个语义，所以不能再依赖。
      - `if v == bg or v == MARK`: 原题只排除红色。变体里蓝色也可能是标记，如果不排除会把蓝标记错误当作 payload 去镜像。
      - `marker_pts = isolated red singleton`: 原题第二大分支专门处理孤立红点邻域。变体不再生成这种结构，这段逻辑必须删除，否则会误读单蓝点或单红点。
    new_solver_assumptions:
      - 每个可变换组件恰好含有一个标记像素，且该像素颜色属于 {1,2}。
      - 红标记组件的 payload 完全在该标记行的一侧；蓝标记组件的 payload 完全在该标记列的一侧。
      - payload 与其镜像不会重叠，也不会与其他对象重叠。
      - generator 会保证这些条件，因此 solver 可以用直接镜像而无需复杂冲突消解。

  d_placement_legality:
    sprite_construction:
      - 水平轴对象使用高度 `2*ph+1`、宽度 `pw` 的 patch；红标记位于中间行，只占一个像素，payload 全在上半或下半。
      - 垂直轴对象使用高度 `ph`、宽度 `2*pw+1` 的 patch；蓝标记位于中间列，只占一个像素，payload 全在左半或右半。
      - 标记的位置必须选在贴近轴的一条 payload 边界上已有像素的对应行/列，保证输入里标记与 payload 4 连通。
    placement:
      - 输出阶段按完整 footprint（payload + mirror + marker）放置，使用带 8 邻域膨胀的 padding 检查，防止不同对象互相接触并被 solver 合并。
      - 擦除 mirror 得到输入后，组件仍保持原本的单对象结构，solver 的 4 连通分量分解是稳定的。

  e_visual_observability:
    human_view:
      - 解题者在样例里会看到：红单点总是触发上下镜像，蓝单点总是触发左右镜像。
      - 因为两种标记几何完全相同，不能再靠 "点=点反射" 或 "线方向=镜像方向" 解决，必须归纳颜色含义。
    anti_degeneracy:
      - 每个样本强制同时包含红标记和蓝标记，避免训练集恰好只出现一种颜色而让规则不可见。

  f_visual_cleanliness:
    choices:
      - 沿用较高背景占比和对象间 padding，减少局部拥挤。
      - 每个对象仍是单色 payload + 单像素标记，不做多层叠加，不引入遮挡。
      - 标记颜色 1/2 不作为 payload 颜色使用，避免视觉角色混淆。

  g_color_change_inferability:
    mapping:
      - 只有两种专用标记色：2 -> 水平轴，1 -> 垂直轴。
      - payload 颜色从其余非背景颜色中选择，保证颜色语义分层清楚。
    support:
      - 每个网格至少一红一蓝，多数网格会有多个重复实例，足以让人从 2-4 组样例中推断颜色-方向对应。
"""


HMARK = 2  # horizontal reflection axis marker
VMARK = 1  # vertical reflection axis marker
MARKERS = {HMARK, VMARK}

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_HORIZONTAL, MAX_VERTICAL = 4, 4

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using a two-color embedded marker system.

    Input structure:
    - bg = most common color.
    - Red (2) singleton embedded in a component means: reflect payload across that pixel's row.
    - Blue (1) singleton embedded in a component means: reflect payload across that pixel's column.
    - No isolated point-reflection markers exist in this variant.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != bg, connectivity=4)
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue

        vals = [int(g[r, c]) for r, c in coords]
        marker_cells = [((int(r), int(c)), int(v)) for (r, c), v in zip(coords, vals) if int(v) in MARKERS]
        if len(marker_cells) != 1:
            continue

        (mr, mc), marker = marker_cells[0]
        payload_cells = [((int(r), int(c)), int(v)) for (r, c), v in zip(coords, vals) if int(v) not in MARKERS]
        if not payload_cells:
            continue

        if marker == HMARK:
            for (r, c), v in payload_cells:
                rr, cc = refl_row(mr, r, c)
                if inb(rr, cc):
                    out[rr, cc] = int(v)
        else:
            for (r, c), v in payload_cells:
                rr, cc = refl_col(mc, r, c)
                if inb(rr, cc):
                    out[rr, cc] = int(v)

    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    """Grow a random 4-connected mask of exactly *target_pixels* inside an h x w bbox."""
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


def _build_axis_sprite(rng: random.Random, axis: str) -> dict | None:
    """Build an output sprite with a single embedded marker pixel controlling row/col reflection."""
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None

    pmask = None
    for _ in range(MASK_RETRIES):
        pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None

    if axis == "row":
        H, W = 2 * ph + 1, pw
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        payload_on_top = rng.random() < 0.5
        if payload_on_top:
            boundary_cols = [int(c) for c in np.where(pmask[ph - 1])[0]]
            if not boundary_cols:
                return None
            marker_col = rng.choice(boundary_cols)
            payload[:ph, :] = pmask
            mirror[ph + 1 :, :] = pmask[::-1]
        else:
            boundary_cols = [int(c) for c in np.where(pmask[0])[0]]
            if not boundary_cols:
                return None
            marker_col = rng.choice(boundary_cols)
            payload[ph + 1 :, :] = pmask
            mirror[:ph, :] = pmask[::-1]
        marker_pos = (ph, int(marker_col))
        marker_color = HMARK
    else:
        H, W = ph, 2 * pw + 1
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        payload_on_left = rng.random() < 0.5
        if payload_on_left:
            boundary_rows = [int(r) for r in np.where(pmask[:, pw - 1])[0]]
            if not boundary_rows:
                return None
            marker_row = rng.choice(boundary_rows)
            payload[:, :pw] = pmask
            mirror[:, pw + 1 :] = pmask[:, ::-1]
        else:
            boundary_rows = [int(r) for r in np.where(pmask[:, 0])[0]]
            if not boundary_rows:
                return None
            marker_row = rng.choice(boundary_rows)
            payload[:, pw + 1 :] = pmask
            mirror[:, :pw] = pmask[:, ::-1]
        marker_pos = (int(marker_row), pw)
        marker_color = VMARK

    if payload[marker_pos] or mirror[marker_pos]:
        return None
    if (payload & mirror).any():
        return None

    return {
        "axis": axis,
        "marker_color": marker_color,
        "payload": payload,
        "mirror": mirror,
        "marker_pos": marker_pos,
    }


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
):
    """Find a random position where *patch_mask* fits on *canvas* without touching other objects."""
    H, W = canvas.shape
    h, w = patch_mask.shape
    if h > H or w > W:
        return None
    occ = canvas != bg
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    """Paint a sprite's payload, mirror, and marker onto *canvas* at (r0, c0)."""
    h, w = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0 : r0 + h, c0 : c0 + w][colored] = int(spr["color"])
    mr, mc = spr["marker_pos"]
    canvas[r0 + mr, c0 + mc] = int(spr["marker_color"])


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    """Erase the mirrored half of a sprite back to background."""
    r0, c0 = spr["r0"], spr["c0"]
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = int(bg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg_choices = [c for c in range(10) if c not in MARKERS]
    bg = rng.choice(bg_choices)
    palette = [c for c in range(10) if c not in MARKERS and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    lo_h = max(1, total_markers - MAX_VERTICAL)
    hi_h = min(MAX_HORIZONTAL, total_markers - 1)
    if lo_h > hi_h:
        return None
    n_horizontal = rng.randint(lo_h, hi_h)
    n_vertical = total_markers - n_horizontal
    if n_vertical < 1 or n_vertical > MAX_VERTICAL:
        return None

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)

    sprites: list[dict] = []
    for _ in range(n_horizontal):
        s = _build_axis_sprite(rng, "row")
        if s is None:
            return None
        sprites.append(s)
    for _ in range(n_vertical):
        s = _build_axis_sprite(rng, "col")
        if s is None:
            return None
        sprites.append(s)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, c in zip(sprites, assignment):
        spr["color"] = int(c)

    output = np.full((N, N), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr["payload"] | spr["mirror"]
        mr, mc = spr["marker_pos"]
        footprint = footprint.copy()
        footprint[mr, mc] = True
        loc = _random_free_location(output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = int(r0), int(c0)
        _stamp_sprite(output, spr, int(r0), int(c0))

    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / (N * N) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
