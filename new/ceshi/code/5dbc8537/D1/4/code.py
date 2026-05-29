# Variant of 5dbc8537
# Dimension: D1 — 匹配条件：精灵与洞的对应方式 (match_cond)
# Change: s.bbox == d.rect — 精灵外接矩形尺寸与洞的矩形尺寸完全相等（shape_match）... → 洞内计数匹配

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis for the D1 variant
# ----------------------------------
# variant_spec:
#   kept_from_original:
#     - 输入仍是“贴边条带模板 + 外部散落矩形精灵”。
#     - 输出仍是模板 bbox 的紧裁结果，洞内填入对应精灵。
#     - base 颜色仍由最大非背景连通块确定，外部精灵仍是分离的多色矩形组件。
#   concrete_change:
#     - 每个洞不再只是纯背景矩形，而是“背景矩形 + 1..3 个黑色标记像素”。
#     - 每个外部精灵也带有同样数量的黑色标记像素。
#     - 同尺寸矩形会重复出现，旧题里依赖 bbox 尺寸即可配对；现在尺寸只决定“能不能放进去”，
#       真正决定同尺寸对象对应关系的是黑点数量。
#   behavior_trace:
#     - 原逻辑示例：两个 2x4 洞、两个 2x4 精灵时，旧 solver 只按 2x4 搜索候选窗口，谁放左谁放右都合法。
#     - 变体逻辑：左洞有 1 个黑点、右洞有 2 个黑点；两个 2x4 精灵也分别有 1/2 个黑点。
#       因此 2x4 只给出物理兼容性，计数才给出真正配对。
#     - 差异：原 DFS 会把同尺寸精灵互换；新 solver 必须先数洞内黑点，再按 (尺寸, 黑点数) 配对。
#   complexity_calibration:
#     - 每个样例至少 3 个重复尺寸组，每组 2-3 个矩形，避免“所有尺寸都唯一”退化回原题。
#     - 黑点数限制在 1..3，便于人眼计数，也避免小矩形塞不下太多标记。
#     - 洞之间显式留 1 格 base 间隔，使“这个洞里有几个黑点”视觉上清楚，不需要在连通大走廊中分段猜测。
#   implicit_assumptions_review:
#     - 原 main 中 `sum(sprite bbox area) == hole_mask.sum()` 依赖“所有洞像素组成一个需要 exact-cover 的整体”。
#       这里不再需要 exact cover，因为每个洞被 base 隔开，可直接做组件级匹配。
#     - 原 main 中“任意全洞矩形窗口都是候选位置”依赖尺寸唯一或 DFS 可唯一收敛；当同尺寸重复时会把精灵交换。
#     - 原 `paste(..., mask=component_mask)` 允许非满矩形精灵；新生成器改成满矩形多色 patch，solver 仍能兼容。
#   placement_legality:
#     - 生成器沿长轴顺序放置洞，并在洞与洞、洞与条带端点之间保留至少 1 格 base，避免重叠和粘连。
#     - 洞在横向也保留至少 1 格 base 边框，保证 base 仍构成明显框架。
#     - 外部精灵继续用 `random_free_location` 散放，确保彼此与模板都至少隔 1 格背景。
#   observability:
#     - 解题者能直接看到：多个相同尺寸洞的黑点数不同，外部也有多个相同尺寸精灵的黑点数不同。
#     - 颜色不参与排序或映射，只作为装饰纹理；真正可推断规则来自可见计数。
#   new_solver_assumptions_checked:
#     - 黑色 0 只作为标记色，不会被选作 bg/base。
#     - 每个洞组件都是满矩形，且 `(h, w, marker_count)` 在洞集合中唯一。
#     - 每个外部精灵也是满矩形，且带相同唯一 key；生成器显式保证这些条件。


MARKER = 0

# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill marked rectangular cavities in a template strip by pairing them with marked sprites.

    Input structure:
    - bg is the most common color. One large base-colored strip touches three grid edges and contains several rectangular cavities.
    - Each cavity holds a few black marker cells, and separate full-rectangle sprites sit outside the strip.

    Rule:
    - The largest non-bg connected component color defines the template bbox.
    - Cavities are the non-base rectangles inside that bbox; several cavities can share the same size.
    - Match each outside sprite to the cavity with the same black-marker count among size-compatible rectangles, then paste it into that cavity.

    Key insight: Rectangle size only tells you which slots a sprite can physically fit; the decisive matching cue is how many black marker pixels appear inside the hole and inside the sprite.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(max_area, int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert base_coords.size > 0
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    dh, dw = map(int, dst.shape)

    out = np.full((dh, dw), int(base_col), dtype=int)

    cavity_mask = dst != int(base_col)
    cavity_comps = components_mask(cavity_mask, connectivity=4)
    cavities: dict[tuple[int, int, int], tuple[int, int]] = {}
    for coords in cavity_comps:
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        assert int(coords.shape[0]) == h * w
        marker_count = int(np.sum(dst[rr0 : rr1 + 1, cc0 : cc1 + 1] == int(MARKER)))
        key = (h, w, marker_count)
        assert key not in cavities
        cavities[key] = (int(rr0), int(cc0))

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: dict[tuple[int, int, int], np.ndarray] = {}
    for comp in sprite_comps:
        b = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, b)
        h, w = map(int, patch.shape)
        assert int(comp["area"]) == h * w
        marker_count = int(np.sum(patch == int(MARKER)))
        key = (int(h), int(w), int(marker_count))
        assert key not in sprites
        sprites[key] = patch

    assert set(cavities) == set(sprites)
    for key, (rr0, cc0) in cavities.items():
        paste(out, sprites[key], rr0, cc0, overwrite=True)

    return out


# --------------- generator ---------------

# ── Configuration ────────────────────────────────────────────────────────
GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.24, 0.40
MAX_RECT_LEN = 4
MAX_RECT_WIDTH = 6
MIN_PALETTE_SIZE = 4
BICHROMATIC_FRAC = 0.35
MAX_RETRIES = 400
LAYOUT_RETRIES = 160
SHAPE_RETRIES = 60


def _pick_marker_cells(h: int, w: int, count: int) -> list[tuple[int, int]]:
    """Choose marker cells, preferring separated inner positions when possible."""
    inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    pool = inner if len(inner) >= count else all_cells
    shuffled = pool[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    for cell in shuffled:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return random.sample(pool, count)


def _make_fill(h: int, w: int, cols: list[int]) -> np.ndarray:
    """Fill an (h, w) rectangle with one or two colors."""
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1:
        return out

    rr, cc = np.indices((h, w))
    alt = int(cols[1])
    pattern = random.choice(["checker", "dots", "split"])
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = alt
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any() or mask.all():
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = alt
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1) :, :] = alt
    else:
        out[:, random.randint(1, w - 1) :] = alt
    return out


def _make_piece_patch(h: int, w: int, cols: list[int], marker_count: int) -> np.ndarray:
    """Build a full rectangle sprite patch with visible black markers."""
    patch = _make_fill(h, w, cols)
    for rr, cc in _pick_marker_cells(h, w, marker_count):
        patch[rr, cc] = int(MARKER)
    return patch


def _make_hole_patch(h: int, w: int, bg: int, marker_count: int) -> np.ndarray:
    """Build an input cavity patch: bg rectangle with black markers."""
    patch = np.full((h, w), int(bg), dtype=int)
    for rr, cc in _pick_marker_cells(h, w, marker_count):
        patch[rr, cc] = int(MARKER)
    return patch


def _layout_patterns(long: int) -> list[tuple[int, ...]]:
    """Repeated-size multiplicities; every group size is 2 or 3."""
    out: list[tuple[int, ...]] = []
    for groups in range(3, 5):
        for mults in itertools.product([2, 3], repeat=groups):
            total = sum(mults)
            if 6 <= total <= min(10, long - 4):
                out.append(tuple(int(v) for v in mults))
    return out


def _sample_rects(long: int, thick: int) -> list[tuple[int, int, int, int, int]] | None:
    """Sample disjoint cavity rectangles with repeated shapes and unique counts per shape."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 2:
        return None

    patterns = _layout_patterns(long)
    if not patterns:
        return None

    for _ in range(LAYOUT_RETRIES):
        mults = list(random.choice(patterns))
        shapes: list[tuple[int, int, int]] = []
        used_shapes: set[tuple[int, int]] = set()
        ok = True
        for mult in mults:
            chosen = None
            for _ in range(SHAPE_RETRIES):
                h = random.randint(1, MAX_RECT_LEN)
                w = random.randint(2, max_w)
                if (h, w) in used_shapes or h * w < mult + 1:
                    continue
                chosen = (int(h), int(w), int(mult))
                break
            if chosen is None:
                ok = False
                break
            used_shapes.add((chosen[0], chosen[1]))
            shapes.append(chosen)
        if not ok:
            continue

        holes: list[tuple[int, int, int]] = []
        for h, w, mult in shapes:
            counts = list(range(1, mult + 1))
            random.shuffle(counts)
            for marker_count in counts:
                holes.append((int(h), int(w), int(marker_count)))
        random.shuffle(holes)

        total_len = sum(h for h, _w, _k in holes)
        min_used = total_len + len(holes) + 1
        if min_used > long:
            continue

        total_area = sum(h * w for h, w, _k in holes)
        frac = total_area / float(long * thick)
        if not (HOLE_FRAC_LO <= frac <= HOLE_FRAC_HI):
            continue

        gaps = [1] * (len(holes) + 1)
        extra = long - min_used
        for _ in range(extra):
            gaps[random.randrange(len(gaps))] += 1

        rects: list[tuple[int, int, int, int, int]] = []
        pos = gaps[0]
        for i, (h, w, marker_count) in enumerate(holes):
            c_hi = thick - w - 1
            if c_hi < 1:
                ok = False
                break
            c = random.randint(1, c_hi)
            rects.append((int(pos), int(c), int(h), int(w), int(marker_count)))
            pos += h + gaps[i + 1]
        if ok:
            return rects

    return None


def _make_assets(
    rects: list[tuple[int, int, int, int, int]], bg: int, base: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    """Create output pieces and input cavity marker patches."""
    available = [c for c in range(1, 10) if c not in (bg, base, MARKER)]
    if len(available) < MIN_PALETTE_SIZE:
        return None

    pieces: dict[int, np.ndarray] = {}
    holes: dict[int, np.ndarray] = {}
    used_fill_cols: set[int] = set()
    for i, (_r, _c, h, w, marker_count) in enumerate(rects):
        n_cols = 2 if (h * w >= 4 and random.random() < BICHROMATIC_FRAC) else 1
        cols = random.sample(available, n_cols)
        used_fill_cols.update(cols)
        pieces[i] = _make_piece_patch(h, w, cols, marker_count)
        holes[i] = _make_hole_patch(h, w, bg, marker_count)

    if len(used_fill_cols) < MIN_PALETTE_SIZE:
        return None
    return pieces, holes


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    """Expected output: tight template crop with all matched pieces pasted in."""
    out = np.full((dh, dw), int(base), dtype=int)
    for i, (r, c, _h, _w, _k) in enumerate(rects):
        paste(out, pieces[i], r, c, overwrite=True)
    return out


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int, int]],
    pieces: dict[int, np.ndarray],
    hole_patches: dict[int, np.ndarray],
) -> np.ndarray:
    """Build the input by carving marked cavities, then scattering full-rectangle sprites outside."""
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for i, (r, c, _h, _w, _k) in enumerate(rects):
        paste(inp, hole_patches[i], tr0 + r, tc0 + c, overwrite=True)

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c, overwrite=True)
        paste(placement, patch, r, c, overwrite=True)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single sample attempt: build a marked template, invert to input, oracle-check with main()."""
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0

    long = dh if vertical else dw
    canon = _sample_rects(long, thick)
    if canon is None:
        return None

    rects = canon if vertical else [(c, r, w, h, k) for r, c, h, w, k in canon]
    bg, base = random.sample(list(range(1, 10)), 2)
    assets = _make_assets(rects, bg, base)
    if assets is None:
        return None
    pieces, hole_patches = assets

    out = _render_output(dh, dw, base, rects, pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces, hole_patches)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 D1 input")
