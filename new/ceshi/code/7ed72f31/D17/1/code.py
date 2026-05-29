# Variant of 7ed72f31
# Dimension: D17 — 背景与干扰元素 (meta)
# Change: 背景为单色，覆盖率>=88%，无干扰元素... → 添加干扰精灵（不参与反射）

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[5]
SOLVERS_DIR = ROOT / "solvers"
if str(SOLVERS_DIR) not in sys.path:
    sys.path.append(str(SOLVERS_DIR))

from arc_utils import STRUCT8, get_bg, label_mask

"""
Design analysis for D17
======================

source_task:
  none

variant_spec:
  unchanged_core_rule:
    - 红线/红点仍然定义镜像轴；活跃精灵仍按原题方式被复制到背景上。
  new_meta_change:
    - 输入里额外出现 1-2 个彩色干扰精灵。
    - 它们是没有任何红色标记参与的单色连通分量。
    - 输出中这些干扰精灵完全不变。
  observability_strategy:
    - 至少一个干扰精灵与某个 embedded 红线精灵同色。
    - 该干扰精灵被放到该红线精灵的“投影带”里：
      row-axis 时列范围落在红线组件 bbox 内；col-axis 时行范围落在 bbox 内。
    - 这样旧 solver 中“按颜色 + 投影带全图扫描”的写法会误反射它，
      而新 solver 只反射 marker 所在组件的 payload，因此差异真实可见。

behavior_trace:
  small_example:
    input:
      - 一个蓝色 payload 紧贴水平红线，列 10-12。
      - 另一个蓝色 L 形干扰精灵放在更高的位置，但也占据列 10-12。
    original_logic:
      - embedded 分支会遍历整张图，只要颜色属于 payload 且列在 10-12 内就反射。
      - 因此远处的蓝色干扰精灵也会被镜像，产生错误副本。
    variant_logic:
      - embedded 分支只遍历“含该红线的 4 连通组件”里的非红像素。
      - 远处蓝色干扰精灵不在该组件内，因此保持原样。
    difference:
      - 变体新增了“先判断是否与红标记关联”的过滤步骤。

complexity_calibration:
  - 干扰精灵数量限制在 1-2，避免画面过乱，也足以让规则在 2-4 个样例内可见。
  - 干扰精灵仍为单色连通分量，避免把复杂度推到多色配对或部件级匹配。
  - 至少一个干扰精灵与活跃 embedded 精灵同色，防止解题者用“是否同色”偷简化。

implicit_assumption_audit:
  original_embedded_branch:
    - `payload = {v for v in vals if v != MARK}` 默认“同色像素只出现在当前红线对象附近”。
    - 后续双层 for 全图扫描默认“符合颜色且落在 bbox 投影带里的像素都属于当前对象”。
    - D17 下这个前提失效，因为远处可有同色干扰精灵。
    - 修复：只对当前 marker-containing component 内的 payload 坐标做反射。
  original_isolated_branch:
    - 它先找红点 8 邻域，再只反射包含这些邻接像素的同色 8 连通分量。
    - 这在 D17 下仍然成立，只需维持“干扰精灵不与红点 8 邻接，也不 8 连到活跃 payload”。
  output_writes:
    - 反射仍然直接覆盖到 `out` 上，但 generator 保证所有反射目标是背景，不会发生新冲突。

placement_legality:
  - 所有对象仍使用 output-first 放置。
  - active 精灵先放，保持原题无重叠、无越界。
  - aligned distractor 再放：既要与目标 embedded 精灵共享投影带，又要与任何非背景保持至少 1 格 8 邻域间距。
  - 同时检查“如果按旧逻辑误反射，其反射落点必须在当前 output 的背景上”，从而保证新旧逻辑输出确实不同。

visual_cleanliness:
  - 干扰精灵不接触红标记，也不接触其他组件。
  - 同色但远离红标记，肉眼上能看到“有些同色块会扩张，有些不会”。

color_inference:
  - 不引入新的颜色规则；颜色只是对象身份的一部分。
  - 干扰精灵复用 active 调色板，避免“新颜色 = 不动”的低级捷径。

new_main_assumptions_to_check:
  - embedded 组件里的反射轴仍然是单行、单列或单点红标记；generator 延续原题约束。
  - isolated 红点的活跃 payload 必须由至少一个 8 邻接像素触发；generator 延续原题构造。
  - 干扰精灵必须与所有红标记 8 分离；generator 通过 dilation padding 保证。
"""

MARK = 2


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror only marker-linked sprites; inert distractor components remain unchanged."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != bg, connectivity=4)
    comps: list[dict] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == k)
        if pts.size == 0:
            continue
        coords = [(int(r), int(c)) for r, c in pts]
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
            }
        )

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = center
        return 2 * int(pr) - int(r), 2 * int(pc) - int(c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return 2 * int(r0) - int(r), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), 2 * int(c0) - int(c)

    # Embedded / mixed components: reflect only the payload pixels that belong to that
    # exact 4-connected component. This is the key D17 fix.
    for comp in comps:
        vals = comp["vals"]
        if MARK not in vals:
            continue

        reds = [(r, c) for (r, c), v in zip(comp["coords"], vals) if v == MARK]
        if not reds:
            continue

        red_rows = {r for r, _ in reds}
        red_cols = {c for _, c in reds}
        if len(reds) == 1:
            axis_kind = "point"
            axis_value: int | tuple[int, int] = reds[0]
        elif len(red_rows) == 1:
            axis_kind = "row"
            axis_value = next(iter(red_rows))
        elif len(red_cols) == 1:
            axis_kind = "col"
            axis_value = next(iter(red_cols))
        else:
            continue

        payload_pixels = [(r, c, v) for (r, c), v in zip(comp["coords"], vals) if v != MARK]
        if not payload_pixels:
            continue

        for r, c, v in payload_pixels:
            if axis_kind == "row":
                rr, cc = refl_row(int(axis_value), r, c)
            elif axis_kind == "col":
                rr, cc = refl_col(int(axis_value), r, c)
            else:
                rr, cc = refl_point(axis_value, r, c)  # type: ignore[arg-type]
            if inb(rr, cc):
                out[rr, cc] = int(v)

    # Isolated red points: reflect only the color-components that actually touch the red
    # point in the 8-neighborhood. Detached distractors of the same color are ignored.
    red_singletons = [
        comp["coords"][0]
        for comp in comps
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == MARK
    ]

    color_cache: dict[int, np.ndarray] = {}
    for pr, pc in red_singletons:
        neighbor_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = pr + dr, pc + dc
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == bg or v == MARK:
                    continue
                neighbor_pixels.append((rr, cc, v))
        if not neighbor_pixels:
            continue

        seen_labels: set[tuple[int, int]] = set()
        for rr, cc, v in neighbor_pixels:
            if v not in color_cache:
                color_cache[v], _ = label_mask(g == v, connectivity=8)
            labv = color_cache[v]
            lbl = int(labv[rr, cc])
            if lbl == 0 or (v, lbl) in seen_labels:
                continue
            seen_labels.add((v, lbl))
            for r, c in np.argwhere(labv == lbl):
                mr, mc = refl_point((pr, pc), int(r), int(c))
                if inb(mr, mc):
                    out[mr, mc] = int(v)

    return out


# Generator configuration
GRID_MIN, GRID_MAX = 18, 28
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

DISTRACTOR_COUNT_MIN, DISTRACTOR_COUNT_MAX = 1, 2
DISTRACTOR_SIDE_MIN, DISTRACTOR_SIDE_MAX = 2, 4
DISTRACTOR_PX_MIN, DISTRACTOR_PX_MAX = 2, 6
DISTRACTOR_DENSITY_LO, DISTRACTOR_DENSITY_HI = 0.35, 0.75

PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

MASK_RETRIES = 40
PLACEMENT_RETRIES = 80
GEN_RETRIES = 400

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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
    if not mask.any(axis=0).all() or not mask.any(axis=1).all():
        return None
    return mask


def _build_embedded_sprite(rng: random.Random) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(np.floor(PAYLOAD_DENSITY_HI * ph * pw)))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    is_row = rng.choice([True, False])
    side_flip = rng.random() < 0.5

    if is_row:
        h, w = 2 * ph + 1, pw
        payload = np.zeros((h, w), dtype=bool)
        mirror = np.zeros((h, w), dtype=bool)
        red_pixels = [(ph, c) for c in range(pw)]
        src_slice, dst_slice = (np.s_[:ph], np.s_[ph + 1 :]) if not side_flip else (np.s_[ph + 1 :], np.s_[:ph])
        payload[src_slice] = pmask
        mirror[dst_slice] = pmask[::-1]
        axis_kind = "row"
        axis_value_local = ph
    else:
        h, w = pw, 2 * ph + 1
        payload = np.zeros((h, w), dtype=bool)
        mirror = np.zeros((h, w), dtype=bool)
        red_pixels = [(r, ph) for r in range(pw)]
        src_slice, dst_slice = (np.s_[:, :ph], np.s_[:, ph + 1 :]) if not side_flip else (np.s_[:, ph + 1 :], np.s_[:, :ph])
        payload[src_slice] = pmask.T
        mirror[dst_slice] = pmask.T[:, ::-1]
        axis_kind = "col"
        axis_value_local = ph

    return {
        "kind": "embedded",
        "axis_kind": axis_kind,
        "axis_value_local": axis_value_local,
        "payload": payload,
        "mirror": mirror,
        "red_pixels": red_pixels,
    }


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            return True
    return False


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

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)
    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            h = max(all_rs) - minr + 1
            w = max(all_cs) - minc + 1

            payload = np.zeros((h, w), dtype=bool)
            mirror = np.zeros((h, w), dtype=bool)
            for r, c in pcoords:
                payload[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red_pos = (red_r - minr, red_c - minc)

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {
                "kind": "isolated",
                "axis_kind": "point",
                "payload": payload,
                "mirror": mirror,
                "red_pixels": [red_pos],
            }
    return None


def _build_distractor_sprite(
    rng: random.Random,
    *,
    max_h: int | None = None,
    max_w: int | None = None,
) -> dict | None:
    h_hi = min(DISTRACTOR_SIDE_MAX, int(max_h) if max_h is not None else DISTRACTOR_SIDE_MAX)
    w_hi = min(DISTRACTOR_SIDE_MAX, int(max_w) if max_w is not None else DISTRACTOR_SIDE_MAX)
    if h_hi < DISTRACTOR_SIDE_MIN or w_hi < DISTRACTOR_SIDE_MIN:
        return None
    for _ in range(MASK_RETRIES):
        h = rng.randint(DISTRACTOR_SIDE_MIN, h_hi)
        w = rng.randint(DISTRACTOR_SIDE_MIN, w_hi)
        max_px = min(DISTRACTOR_PX_MAX, int(np.floor(DISTRACTOR_DENSITY_HI * h * w)))
        min_px = max(DISTRACTOR_PX_MIN, int(np.ceil(DISTRACTOR_DENSITY_LO * h * w)))
        if min_px > max_px:
            continue
        mask = _random_connected_mask(h, w, rng.randint(min_px, max_px), rng=rng)
        if mask is None:
            continue
        return {
            "kind": "distractor",
            "axis_kind": "none",
            "payload": mask,
            "mirror": np.zeros((h, w), dtype=bool),
            "red_pixels": [],
        }
    return None


def _sprite_footprint(spr: dict) -> np.ndarray:
    footprint = spr["payload"] | spr["mirror"]
    if spr["red_pixels"]:
        footprint = footprint.copy()
        for r, c in spr["red_pixels"]:
            footprint[r, c] = True
    return footprint


def _candidate_locations(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    *,
    padding: int = PLACEMENT_PADDING,
    accept=None,
) -> list[tuple[int, int]]:
    h, w = patch_mask.shape
    H, W = canvas.shape
    if h > H or w > W:
        return []
    occ = canvas != bg
    if padding > 0 and occ.any():
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                continue
            if accept is not None and not accept(r0, c0):
                continue
            cands.append((r0, c0))
    return cands


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
    accept=None,
) -> tuple[int, int] | None:
    cands = _candidate_locations(canvas, patch_mask, bg, padding=padding, accept=accept)
    return rng.choice(cands) if cands else None


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


def _global_bbox(spr: dict) -> tuple[int, int, int, int]:
    h, w = spr["payload"].shape
    return int(spr["r0"]), int(spr["c0"]), int(spr["r0"] + h - 1), int(spr["c0"] + w - 1)


def _place_aligned_distractor(canvas: np.ndarray, target: dict, spr: dict, bg: int, rng: random.Random) -> bool:
    tr0, tc0, tr1, tc1 = _global_bbox(target)
    h, w = spr["payload"].shape
    H, W = canvas.shape
    axis_kind = target["axis_kind"]
    axis_value = int(target["r0"] + target["axis_value_local"]) if axis_kind == "row" else int(target["c0"] + target["axis_value_local"])
    payload_pts = [(int(r), int(c)) for r, c in np.argwhere(spr["payload"])]

    def reflects_to_empty(r0: int, c0: int) -> bool:
        for dr, dc in payload_pts:
            gr, gc = r0 + dr, c0 + dc
            if axis_kind == "row":
                rr, cc = 2 * axis_value - gr, gc
            else:
                rr, cc = gr, 2 * axis_value - gc
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if canvas[rr, cc] != bg:
                return False
        return True

    def accept(r0: int, c0: int) -> bool:
        if axis_kind == "row":
            if not (tc0 <= c0 and c0 + w - 1 <= tc1):
                return False
            if not (r0 + h - 1 < tr0 or r0 > tr1):
                return False
        else:
            if not (tr0 <= r0 and r0 + h - 1 <= tr1):
                return False
            if not (c0 + w - 1 < tc0 or c0 > tc1):
                return False
        return reflects_to_empty(r0, c0)

    loc = _random_free_location(canvas, _sprite_footprint(spr), bg, rng, accept=accept)
    if loc is None:
        return False
    spr["r0"], spr["c0"] = loc
    _stamp_sprite(canvas, spr, *loc)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c not in (bg, MARK)]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded
    n_distractors = rng.randint(DISTRACTOR_COUNT_MIN, DISTRACTOR_COUNT_MAX)

    active_sprites: list[dict] = []
    embedded_sprites: list[dict] = []
    for _ in range(n_embedded):
        spr = _build_embedded_sprite(rng)
        if spr is None:
            return None
        active_sprites.append(spr)
        embedded_sprites.append(spr)
    for _ in range(n_isolated):
        spr = _build_isolated_sprite(rng)
        if spr is None:
            return None
        active_sprites.append(spr)

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), len(active_sprites)))
    chosen_colors = rng.sample(palette, n_colors)
    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(active_sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, color in zip(active_sprites, assignment):
        spr["color"] = int(color)

    output = np.full((N, N), bg, dtype=int)
    rng.shuffle(active_sprites)
    for spr in active_sprites:
        loc = _random_free_location(output, _sprite_footprint(spr), bg, rng)
        if loc is None:
            return None
        spr["r0"], spr["c0"] = loc
        _stamp_sprite(output, spr, *loc)

    # Force at least one distractor to be a genuine D17 case: same color as an embedded
    # active sprite and aligned with its projection band so the legacy solver would
    # incorrectly mirror it.
    aligned_ok = False
    targets = embedded_sprites[:]
    rng.shuffle(targets)
    for target in targets:
        tr0, tc0, tr1, tc1 = _global_bbox(target)
        max_h = (tr1 - tr0 + 1) if target["axis_kind"] == "col" else None
        max_w = (tc1 - tc0 + 1) if target["axis_kind"] == "row" else None
        for _ in range(PLACEMENT_RETRIES):
            ds = _build_distractor_sprite(rng, max_h=max_h, max_w=max_w)
            if ds is None:
                continue
            ds["color"] = int(target["color"])
            if _place_aligned_distractor(output, target, ds, bg, rng):
                aligned_ok = True
                break
        if aligned_ok:
            break
    if not aligned_ok:
        return None

    for _ in range(n_distractors - 1):
        ds = _build_distractor_sprite(rng)
        if ds is None:
            return None
        ds["color"] = int(rng.choice(chosen_colors))
        loc = _random_free_location(output, _sprite_footprint(ds), bg, rng)
        if loc is None:
            return None
        ds["r0"], ds["c0"] = loc
        _stamp_sprite(output, ds, *loc)

    inp = output.copy()
    for spr in active_sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / float(N * N) < BG_COVERAGE_MIN:
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
