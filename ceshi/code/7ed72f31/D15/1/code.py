# Variant of 7ed72f31
# Dimension: D15 — 符号编码：图例与辅助信息 (symbol_instructions)
# Change: 无显式图例或符号编码。变换规则完全由红色标记的空间几何隐式定义... → 调色板块编码反射方向

from __future__ import annotations

"""
Design analysis for the D15 variant
===================================

spec_variant:
  preserved_core:
    - 仍然是“红标记 + 彩色精灵 -> 在背景上补出镜像”。
    - 仍然保留两类对象：嵌入式红标记（红点与负载同一 4 连通块）和孤立红标记（红点单独存在，8 邻接一个彩色块）。
    - 仍然使用 output-first 生成：先构造完整输出，再抹掉镜像半边得到输入，并用 solver 回验。
  d15_change:
    - 原题依赖红标记几何：红横线=按行镜像，红竖线=按列镜像，单红点=点反射。
    - 新题把“方向说明”搬到左上角调色板块：
      - 左上角 1x3 彩条 -> 按水平轴镜像，也就是围绕每个红点所在的“行”做反射。
      - 左上角 3x1 彩条 -> 按垂直轴镜像，也就是围绕每个红点所在的“列”做反射。
    - 因而所有红标记统一为“单个红像素”。红点只负责给出轴的位置，不再负责给出轴的方向。
  concrete_pixel_behavior:
    - 嵌入式对象：一个单色负载贴着某个红点，但只在轴的一侧出现；输出补出另一侧的镜像。
      例：调色板是横向 1x3，则某负载在红点上方，输出在其下方补出按同一行反射的副本。
    - 孤立对象：红点与彩色负载只做 8 邻接，不做 4 邻接；输出同样按调色板指定的全局方向补镜像。
  trace_example:
    - 原逻辑：若某个混合组件里只有一个红点，solver 会把它视为“点反射中心”。
    - 新逻辑：同样的单红点混合组件，先读左上角调色板；若彩条横放，则这个红点表示“水平轴上的锚点”，只翻上下，不翻左右。
    - 差异：旧题看红标记自身几何，新题看单独的角落符号；因此同一个输入在新旧 solver 下通常会得到不同输出。
  complexity_calibration:
    - 方向只保留两档：horizontal-axis / vertical-axis。自由度足够低，2-4 个样例即可覆盖。
    - 每张图仍有 3-7 个红标记，让同一张图里出现多次重复证据，避免调色板只是孤例噪声。
    - 调色板固定在左上角，避免再引入“先找哪个角”这一额外维度，保证 D15 是唯一新增难点。
  implicit_assumption_audit:
    - 原 main 先把全部非背景做 4 连通分量；若直接沿用，会把新加入的调色板当成普通组件处理，所以必须先剔除调色板区域。
    - 原 main 用红点数量/排布推断 axis=(row/col/point)。这个前提在变体中失效，因为所有红标记都退化成单点。
    - 原 main 的 embedded 分支会扫描 bbox 内所有同色像素并做点反射；对本变体会把应做“行/列反射”的对象错误地做成“点反射”。
    - 原 main 默认可以覆盖任意目标格；本变体中调色板是辅助信息，不能被镜像覆盖，所以写出时需保护调色板格。
  placement_legality:
    - 角落调色板先画到画布上，后续放置统一避开它。
    - 每个 sprite 的完整输出 footprint = 原负载 + 镜像 + 红点，放置时要求 footprint 与现有非背景至少间隔 1 格，避免组件串连。
    - 生成器只构造“镜像落在背景上”的对象，因此 solver 只需把镜像写到输入中的背景格。
  visual_observability:
    - 输入里所有红标记都长成单红点，人类无法再从红标记几何直接读出方向，只能利用左上角彩条。
    - 横条样例与竖条样例会在训练中都出现；同一规则在一张图里反复作用于多个对象，规律可见。
  cleanliness_and_color_reasoning:
    - 调色板只占 3 格，且不含红色，和其他“红点+负载”对象视觉上分层明确。
    - 调色板颜色只作为“这是图例”的外观材料，不参与颜色映射；颜色数仍控制在 3-4 种负载色，避免无提示的颜色爆炸。
"""

import numpy as np
import random
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask


MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

# Palette / colors
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4
PALETTE_LEN = 3

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 300

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _palette_mask(shape: tuple[int, int], axis_kind: str) -> np.ndarray:
    """Mask for the fixed top-left palette strip."""
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    if axis_kind == "row":
        m[0, :PALETTE_LEN] = True
    else:
        m[:PALETTE_LEN, 0] = True
    return m


def _detect_palette(g: np.ndarray, bg: int) -> tuple[str, np.ndarray]:
    """Read the top-left palette strip and return ('row'|'col', palette_mask)."""
    h, w = g.shape
    if w >= PALETTE_LEN:
        vals = [int(g[0, c]) for c in range(PALETTE_LEN)]
        if all(v != bg and v != MARK for v in vals):
            return "row", _palette_mask((h, w), "row")
    if h >= PALETTE_LEN:
        vals = [int(g[r, 0]) for r in range(PALETTE_LEN)]
        if all(v != bg and v != MARK for v in vals):
            return "col", _palette_mask((h, w), "col")
    raise ValueError("palette strip missing or malformed")


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using single red markers plus a palette-encoded direction.

    Input structure:
    - bg = most common color. Red (2) = marker color.
    - Left-top palette strip encodes the global reflection direction:
      horizontal 1x3 strip -> reflect across each marker's row
      vertical 3x1 strip -> reflect across each marker's column
    - All red markers are single pixels.
    - A red pixel is either embedded in a payload component or isolated but 8-adjacent to one.

    Rule:
    - Read the palette strip first to determine whether all reflections are row-based or col-based.
    - For an embedded single-red component, reflect that component's non-red payload across the row/col through the red pixel.
    - For an isolated red pixel, reflect each adjacent 8-connected monochrome component across the same global row/col rule.
    - Keep originals, red markers, and palette unchanged.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    axis_kind, palette = _detect_palette(g, bg)
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < h and 0 <= int(c) < w

    def reflect(anchor: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        ar, ac = anchor
        if axis_kind == "row":
            return _refl_row(ar, r, c)
        return _refl_col(ac, r, c)

    def maybe_write(r: int, c: int, v: int) -> None:
        if inb(r, c) and not palette[r, c] and int(g[r, c]) == int(bg):
            out[r, c] = int(v)

    # Components exclude the palette so the legend never masquerades as a sprite.
    lab, n = label_mask((g != int(bg)) & (~palette), connectivity=4)
    comps: list[dict] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    # 1) Embedded single-red component: reflect that component's payload.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if len(reds) != 1 or len(coords) <= 1:
            continue
        anchor = reds[0]
        for (r, c), v in zip(coords, vals):
            if int(v) == int(MARK):
                continue
            rr, cc = reflect(anchor, int(r), int(c))
            maybe_write(int(rr), int(cc), int(v))

    # 2) Isolated red point: reflect each adjacent 8-connected monochrome component.
    marker_pts = [comp["coords"][0] for comp in comps if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK)]
    color_labels: dict[int, np.ndarray] = {}
    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_by_color: dict[int, set[int]] = {}
        for dr, dc in _DIRS8:
            rr, cc = int(pr + dr), int(pc + dc)
            if not inb(rr, cc) or palette[rr, cc]:
                continue
            v = int(g[rr, cc])
            if v == int(bg) or v == int(MARK):
                continue
            if v not in color_labels:
                color_labels[v], _ = label_mask((g == int(v)) & (~palette), connectivity=8)
            lbl = int(color_labels[v][rr, cc])
            if lbl == 0:
                continue
            neigh_by_color.setdefault(int(v), set()).add(lbl)

        for v, labels in neigh_by_color.items():
            labv = color_labels[int(v)]
            for lbl in labels:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = reflect((pr, pc), int(r), int(c))
                    maybe_write(int(rr), int(cc), int(v))

    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    """Grow a random 4-connected mask of exactly target_pixels inside an h x w bbox."""
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


def _build_embedded_sprite(axis_kind: str, rng: random.Random) -> dict | None:
    """Build an output sprite whose only instruction pixel is a single embedded red cell."""
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    side_flip = rng.random() < 0.5
    if axis_kind == "row":
        h, w = 2 * ph + 1, pw
        payload = np.zeros((h, w), dtype=bool)
        mirror = np.zeros((h, w), dtype=bool)
        if not side_flip:
            payload[:ph] = pmask
            mirror[ph + 1 :] = pmask[::-1]
            edge_cols = [int(c) for c in np.flatnonzero(pmask[-1])]
        else:
            payload[ph + 1 :] = pmask
            mirror[:ph] = pmask[::-1]
            edge_cols = [int(c) for c in np.flatnonzero(pmask[0])]
        red_pixels = [(ph, rng.choice(edge_cols))]
    else:
        h, w = ph, 2 * pw + 1
        payload = np.zeros((h, w), dtype=bool)
        mirror = np.zeros((h, w), dtype=bool)
        if not side_flip:
            payload[:, :pw] = pmask
            mirror[:, pw + 1 :] = pmask[:, ::-1]
            edge_rows = [int(r) for r in np.flatnonzero(pmask[:, -1])]
        else:
            payload[:, pw + 1 :] = pmask
            mirror[:, :pw] = pmask[:, ::-1]
            edge_rows = [int(r) for r in np.flatnonzero(pmask[:, 0])]
        red_pixels = [(rng.choice(edge_rows), pw)]

    return {"payload": payload, "mirror": mirror, "red_pixels": red_pixels}


def _build_isolated_sprite(axis_kind: str, rng: random.Random) -> dict | None:
    """Build an isolated red marker with a line-reflected partner chosen by the palette."""
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
    pset = set(pcoords)
    candidates: list[tuple[int, int]] = []
    if axis_kind == "row":
        top = rng.random() < 0.5
        edge_r = 0 if top else ph - 1
        red_r = -1 if top else ph
        edge_cells = [(r, c) for r, c in pcoords if int(r) == int(edge_r)]
        rng.shuffle(edge_cells)
        for _r, c in edge_cells:
            for dc in rng.sample([-1, 1], 2):
                red = (int(red_r), int(c + dc))
                candidates.append(red)
    else:
        left = rng.random() < 0.5
        edge_c = 0 if left else pw - 1
        red_c = -1 if left else pw
        edge_cells = [(r, c) for r, c in pcoords if int(c) == int(edge_c)]
        rng.shuffle(edge_cells)
        for r, _c in edge_cells:
            for dr in rng.sample([-1, 1], 2):
                red = (int(r + dr), int(red_c))
                candidates.append(red)

    seen: set[tuple[int, int]] = set()
    for red_r, red_c in candidates:
        if (red_r, red_c) in seen:
            continue
        seen.add((red_r, red_c))
        if (red_r, red_c) in pset:
            continue
        if any(abs(pr - red_r) + abs(pc - red_c) == 1 for pr, pc in pcoords):
            continue
        if not any(max(abs(pr - red_r), abs(pc - red_c)) == 1 for pr, pc in pcoords):
            continue

        if axis_kind == "row":
            refl = [(int(2 * red_r - r), int(c)) for r, c in pcoords]
        else:
            refl = [(int(r), int(2 * red_c - c)) for r, c in pcoords]
        rset = set(refl)
        if (red_r, red_c) in rset or pset & rset:
            continue
        if any(abs(rr - red_r) + abs(cc - red_c) == 1 for rr, cc in refl):
            continue

        all_pts = pcoords + refl + [(red_r, red_c)]
        minr = min(r for r, _ in all_pts)
        minc = min(c for _, c in all_pts)
        h = max(r for r, _ in all_pts) - minr + 1
        w = max(c for _, c in all_pts) - minc + 1
        payload = np.zeros((h, w), dtype=bool)
        mirror = np.zeros((h, w), dtype=bool)
        for r, c in pcoords:
            payload[r - minr, c - minc] = True
        for r, c in refl:
            mirror[r - minr, c - minc] = True
        red_pos = (red_r - minr, red_c - minc)
        return {"payload": payload, "mirror": mirror, "red_pixels": [red_pos]}
    return None


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
):
    """Find a random position where patch_mask fits on canvas without touching non-bg cells."""
    h, w = canvas.shape
    ph, pw = patch_mask.shape
    if ph > h or pw > w:
        return None
    occ = canvas != bg
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if not (occ[r0 : r0 + ph, c0 : c0 + pw] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0 : r0 + h, c0 : c0 + w][colored] = spr["color"]
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + rr, c0 + cc] = MARK


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = spr["r0"], spr["c0"]
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = bg


def _stamp_palette(canvas: np.ndarray, axis_kind: str, colors: list[int]) -> None:
    if axis_kind == "row":
        for c, v in enumerate(colors[:PALETTE_LEN]):
            canvas[0, c] = int(v)
    else:
        for r, v in enumerate(colors[:PALETTE_LEN]):
            canvas[r, 0] = int(v)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)
    axis_kind = rng.choice(["row", "col"])

    sprites: list[dict] = []
    for builder, count in [(_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)]:
        for _ in range(count):
            s = builder(axis_kind, rng)
            if s is None:
                return None
            sprites.append(s)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, c in zip(sprites, assignment):
        spr["color"] = int(c)

    output = np.full((n, n), bg, dtype=int)
    palette_colors = rng.sample(chosen_colors, PALETTE_LEN)
    _stamp_palette(output, axis_kind, palette_colors)

    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr["payload"] | spr["mirror"]
        for r, c in spr["red_pixels"]:
            footprint[r, c] = True
        loc = _random_free_location(output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = int(r0), int(c0)
        _stamp_sprite(output, spr, int(r0), int(c0))

    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / (n * n) < BG_COVERAGE_MIN:
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
