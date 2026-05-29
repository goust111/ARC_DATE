# Combo variant of 5dbc8537 (index 12)
# combo_spec: D1/4+D12/2
# Name: 计数匹配多步消歧

from __future__ import annotations

import itertools
import random
from collections import Counter

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

# Design analysis for the combo variant
# -------------------------------------
# combo_design:
#   visual_concretization:
#     - 仍保留原题的“贴边模板条带 + 外部散落矩形精灵”大框架，输出仍是模板 bbox 的紧裁结果。
#     - 但洞区不再是一个连通走廊，而是被 base 像素隔开的多个独立矩形洞；每个洞里是 bg 底色加 1..3 个黑色标记点。
#     - 外部精灵仍是完整矩形 patch，内部也嵌入同样数量、同样相对坐标的黑色标记点。
#     - 关键融合点：同尺寸组内先看“黑点个数”做第一轮筛选；若同尺寸里仍有多个洞拥有相同个数，再看黑点在矩形内的分布坐标做第二轮消歧。
#   behavior_trace:
#     - 原题：两个 2x4 洞、两个 2x4 精灵时，只要 bbox 一样就可交换，DFS 负责找一个整体可行覆盖。
#     - D1/4 单变体：若两个 2x4 洞分别有 1/2 个黑点，那么先数点即可直接配对。
#     - 本融合版：例如三个 2x4 洞的黑点模式分别为 {(0,0)}, {(1,3)}, {(0,0),(1,3)}；
#       对应三个 2x4 精灵也带这些标记。第一轮先把“1 点精灵”缩小到前两个 2x4 洞的候选集合，
#       第二轮再靠单点位于左上还是右下的空间布局决定具体归属；而“2 点精灵”则在第一轮就唯一。
#     - 差异：尺寸只负责物理兼容；真正的推理链是“同尺寸 -> 数标记 -> 看标记布局”，不再靠原题的 exact-cover 搜索来偷偷确定答案。
#   complexity_calibration:
#     - 每个样例保持 3-4 个重复尺寸组，总矩形数 7-11，避免退化成“所有尺寸都唯一”的简单题。
#     - 标记数只取 1..3，便于人眼计数；但至少一个尺寸组会出现“同计数重复”的 bucket，迫使进入第二轮布局判别。
#     - 洞之间沿长轴显式留 1 格以上 base 间隙，横向也保留 base 边框，保证每个洞是独立矩形组件，标记视觉上干净。
#   implicit_assumptions_review:
#     - 原 main 的 `hole_mask` + 滑窗候选依赖“洞区是连通走廊且按尺寸搜索”；融合后改成组件级读取矩形洞，不再依赖 exact cover。
#     - 原逻辑默认尺寸足以唯一决定位置；这里该假设失效，所以 solver 必须显式两轮分桶：先按 marker_count，再按 marker_signature。
#     - 原 `paste(... mask=component_mask)` 支持非满矩形精灵；新生成器构造的是满矩形 patch，solver 也对 `area == h*w` 做显式检查。
#     - 新增假设是：同一 `(h,w,count)` bucket 内，每个洞和精灵的标记布局签名唯一；生成器通过模式池采样保证这一点。
#   placement_legality:
#     - 每个洞都是 axis-aligned 满矩形，放在模板内部且不重叠；洞与洞、洞与模板端点之间至少隔 1 格 base。
#     - 外部精灵继续使用 `random_free_location` 散放，确保彼此与模板均至少隔 1 格背景。
#     - 采用 output-first：先决定洞布局与目标精灵，再反推输入，并再次调用 `main()` 做 oracle 校验。
#   observability_and_cleanliness:
#     - 人类能直接看到：有多块同尺寸精灵；模板里也有多块同尺寸洞，而且洞内黑点数与黑点位置都可见。
#     - 第一轮“数几个点”在样例中很明显；第二轮“点在左上/右下/中间等哪里”同样是纯视觉线索，不依赖颜色编号或隐藏排序。
#     - 洞内除 bg 与黑点外没有额外彩色预填，因此画面层次简单，不会与 base 边框或外部精灵纹理混淆。
#   new_main_assumptions_checked:
#     - bg 与 base 都不会使用黑色 0，因此黑点是稳定可见的独立语义。
#     - 每个洞组件的像素数等于其 bbox 面积，保证它确实是完整矩形而不是锯齿形区域。
#     - 同一尺寸组中，count bucket 的多重性在洞集合与精灵集合中完全一致。
#     - 至少有一个尺寸组满足“按 count 后仍有 2 个以上候选，但按 signature 后唯一”，生成器会量化验证这一点。


MARKER = 0

# --------------- solver ---------------


def _find_base_color(g: np.ndarray, bg: int) -> int:
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
    return int(base_candidates[0][1])


def _marker_signature(patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(patch == int(MARKER))
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _extract_sprite_groups(
    g: np.ndarray, bg: int, dst_bbox: tuple[int, int, int, int]
) -> dict[tuple[int, int], list[dict]]:
    r0, c0, r1, c1 = map(int, dst_bbox)
    masked = g.copy()
    masked[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    groups: dict[tuple[int, int], list[dict]] = {}
    for comp in comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        h, w = map(int, patch.shape)
        assert int(comp["area"]) == h * w
        sig = _marker_signature(patch)
        groups.setdefault((int(h), int(w)), []).append(
            {
                "patch": patch,
                "count": int(len(sig)),
                "signature": sig,
            }
        )
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill marked rectangular cavities in a template strip by matching outside sprites in two rounds.

    Input structure:
    - The most common color is the background, while one dominant non-background color forms a long base-colored strip touching three grid edges.
    - Inside that strip are several isolated rectangular cavities; each cavity is mostly background-colored but contains a few black marker pixels.
    - Outside the strip sit separate full-rectangle sprites, and those sprites carry the same kind of black markers.

    Rule:
    - Use the largest non-bg connected component color to locate the template bbox, then read every non-base cavity inside it as a full rectangle.
    - Group cavities and sprites by rectangle size for physical compatibility.
    - Within each size group, first match by the number of black marker pixels; whenever that still leaves multiple candidates, resolve the tie by comparing the marker-coordinate pattern inside the rectangle.

    Key insight: The puzzle is solved by a genuine two-step cue hierarchy: size only says which pieces can fit, marker count narrows the candidate slot set, and the marker layout inside equally counted rectangles determines the final correspondence.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base = _find_base_color(g, bg)

    base_coords = np.argwhere(g == int(base))
    assert base_coords.size > 0
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))
    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)

    out = np.full((dh, dw), int(base), dtype=int)

    cavity_groups: dict[tuple[int, int], list[dict]] = {}
    cavity_mask = dst != int(base)
    for coords in components_mask(cavity_mask, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        assert int(coords.shape[0]) == h * w
        patch = dst[rr0 : rr1 + 1, cc0 : cc1 + 1]
        sig = _marker_signature(patch)
        cavity_groups.setdefault((h, w), []).append(
            {
                "pos": (int(rr0), int(cc0)),
                "count": int(len(sig)),
                "signature": sig,
            }
        )

    sprite_groups = _extract_sprite_groups(g, bg, dst_bbox)
    assert set(cavity_groups) == set(sprite_groups)

    for size, cavities in cavity_groups.items():
        sprites = sprite_groups[size]
        cavity_counts: dict[int, list[dict]] = {}
        sprite_counts: dict[int, list[dict]] = {}
        for cav in cavities:
            cavity_counts.setdefault(int(cav["count"]), []).append(cav)
        for sp in sprites:
            sprite_counts.setdefault(int(sp["count"]), []).append(sp)

        assert Counter({k: len(v) for k, v in cavity_counts.items()}) == Counter(
            {k: len(v) for k, v in sprite_counts.items()}
        )

        for count, bucket in sprite_counts.items():
            candidates = cavity_counts[int(count)]
            assert len(bucket) == len(candidates)

            if len(bucket) == 1:
                rr0, cc0 = map(int, candidates[0]["pos"])
                paste(out, bucket[0]["patch"], rr0, cc0, overwrite=True)
                continue

            by_signature = {}
            for cav in candidates:
                sig = cav["signature"]
                assert sig not in by_signature
                by_signature[sig] = cav

            for sp in bucket:
                sig = sp["signature"]
                assert sig in by_signature
                rr0, cc0 = map(int, by_signature[sig]["pos"])
                paste(out, sp["patch"], rr0, cc0, overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.24, 0.40
MAX_RECT_LEN = 4
MAX_RECT_WIDTH = 6
MIN_PALETTE_SIZE = 4
BICHROMATIC_FRAC = 0.35
MAX_RETRIES = 500
LAYOUT_RETRIES = 240
SHAPE_RETRIES = 80

MIN_GROUPS, MAX_GROUPS = 3, 4
MIN_TOTAL_RECTS, MAX_TOTAL_RECTS = 7, 11
MAX_MARKERS = 3
MIN_AMBIGUOUS_BUCKETS = 1
MIN_AMBIGUOUS_PIECES = 2


def _ordered_cells(h: int, w: int) -> list[tuple[int, int]]:
    raw = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (h // 2, w // 2),
        (0, w // 2),
        (h // 2, 0),
        (h - 1, w // 2),
        (h // 2, w - 1),
    ]
    ordered: list[tuple[int, int]] = []
    for pos in raw:
        if pos not in ordered:
            ordered.append(pos)
    for r in range(h):
        for c in range(w):
            if (r, c) not in ordered:
                ordered.append((r, c))
    return ordered


def _well_spaced(cells: tuple[tuple[int, int], ...]) -> bool:
    for i, (r0, c0) in enumerate(cells):
        for r1, c1 in cells[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _marker_pattern_pool(h: int, w: int, count: int) -> list[tuple[tuple[int, int], ...]]:
    ordered = _ordered_cells(h, w)
    combos = [tuple(sorted(cells)) for cells in itertools.combinations(ordered, count)]
    spacious = [cells for cells in combos if _well_spaced(cells)]
    pool = spacious if len(spacious) >= 6 else combos
    random.shuffle(pool)
    return pool


def _count_templates(mult: int) -> list[tuple[int, ...]]:
    if mult == 2:
        return [(1, 1), (1, 2), (2, 2)]
    if mult == 3:
        return [(1, 1, 2), (1, 2, 2), (1, 1, 1), (2, 2, 2)]
    if mult == 4:
        return [(1, 1, 2, 2), (1, 1, 1, 2), (1, 2, 2, 2), (1, 2, 3, 3)]
    return []


def _layout_patterns(long: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for groups in range(MIN_GROUPS, MAX_GROUPS + 1):
        for mults in itertools.product([2, 3, 4], repeat=groups):
            total = sum(mults)
            if max(mults) >= 3 and MIN_TOTAL_RECTS <= total <= min(MAX_TOTAL_RECTS, long - 4):
                out.append(tuple(int(v) for v in mults))
    return out


def _sample_group_specs(
    long: int, thick: int
) -> list[tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]] | None:
    patterns = _layout_patterns(long)
    if not patterns:
        return None

    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 2:
        return None

    for _ in range(LAYOUT_RETRIES):
        mults = list(random.choice(patterns))
        used_shapes: set[tuple[int, int]] = set()
        groups: list[tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]] = []
        ambiguous_buckets = 0
        ambiguous_pieces = 0
        stepwise_groups = 0
        ok = True

        for mult in mults:
            chosen = None
            for _ in range(SHAPE_RETRIES):
                h = random.randint(1, MAX_RECT_LEN)
                w = random.randint(2, max_w)
                if (h, w) in used_shapes or h * w < 4:
                    continue

                template_opts: list[
                    tuple[int, tuple[tuple[tuple[int, int], ...], ...], int, int, int, int]
                ] = []
                for counts in _count_templates(mult):
                    if max(counts) > min(MAX_MARKERS, h * w):
                        continue
                    freq = Counter(map(int, counts))
                    picked_patterns: list[tuple[tuple[int, int], ...]] = []
                    local_ambiguous_buckets = 0
                    local_ambiguous_pieces = 0
                    valid = True
                    for count, need in freq.items():
                        pool = _marker_pattern_pool(h, w, int(count))
                        if len(pool) < int(need):
                            valid = False
                            break
                        sample = pool[: int(need)]
                        picked_patterns.extend(sample)
                        if int(need) >= 2:
                            local_ambiguous_buckets += 1
                            local_ambiguous_pieces += int(need)
                    if not valid:
                        continue

                    mixed_counts = int(len(freq) >= 2)
                    stepwise = int(mixed_counts and local_ambiguous_buckets > 0)
                    patterns_tuple = tuple(picked_patterns)
                    template_opts.append(
                        (
                            int(sum(counts)),
                            patterns_tuple,
                            int(local_ambiguous_buckets),
                            int(local_ambiguous_pieces),
                            mixed_counts,
                            stepwise,
                        )
                    )

                if not template_opts:
                    continue

                random.shuffle(template_opts)
                _score, patterns_tuple, local_buckets, local_pieces, _mixed_counts, local_stepwise = max(
                    template_opts,
                    key=lambda item: (int(item[5]), int(item[4]), int(item[2]), int(item[3])),
                )
                chosen = (int(h), int(w), patterns_tuple, int(local_buckets), int(local_pieces), int(local_stepwise))
                break

            if chosen is None:
                ok = False
                break

            h, w, patterns_tuple, local_buckets, local_pieces, local_stepwise = chosen
            used_shapes.add((h, w))
            groups.append((h, w, patterns_tuple))
            ambiguous_buckets += int(local_buckets)
            ambiguous_pieces += int(local_pieces)
            stepwise_groups += int(local_stepwise)

        if not ok:
            continue

        rect_count = sum(len(patterns) for _h, _w, patterns in groups)
        total_len = sum(len(patterns) * h for h, _w, patterns in groups)
        min_used = total_len + rect_count + 1
        if min_used > long:
            continue

        total_area = sum(len(patterns) * h * w for h, w, patterns in groups)
        frac = total_area / float(long * thick)
        if not (HOLE_FRAC_LO <= frac <= HOLE_FRAC_HI):
            continue
        if ambiguous_buckets < MIN_AMBIGUOUS_BUCKETS or ambiguous_pieces < MIN_AMBIGUOUS_PIECES:
            continue
        if stepwise_groups < 1:
            continue
        return groups

    return None


def _sample_rects(
    long: int, thick: int
) -> list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]] | None:
    groups = _sample_group_specs(long, thick)
    if groups is None:
        return None

    holes: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
    for h, w, patterns in groups:
        for marker_cells in patterns:
            holes.append((int(h), int(w), marker_cells))
    random.shuffle(holes)

    total_len = sum(h for h, _w, _cells in holes)
    min_used = total_len + len(holes) + 1
    assert min_used <= long
    gaps = [1] * (len(holes) + 1)
    extra = long - min_used
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    rects: list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]] = []
    pos = gaps[0]
    for i, (h, w, marker_cells) in enumerate(holes):
        c_hi = thick - w - 1
        if c_hi < 1:
            return None
        c = random.randint(1, c_hi)
        rects.append((int(pos), int(c), int(h), int(w), marker_cells))
        pos += h + gaps[i + 1]
    return rects


def _make_fill(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), int(cols[0]), dtype=int)
    if len(cols) == 1:
        return out

    alt = int(cols[1])
    rr, cc = np.indices((h, w))
    pattern = random.choice(["checker", "dots", "split"])
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = alt
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not bool(mask.any()) or bool(mask.all()):
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = alt
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1) :, :] = alt
    else:
        out[:, random.randint(1, w - 1) :] = alt
    return out


def _make_piece_patch(h: int, w: int, cols: list[int], marker_cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = _make_fill(h, w, cols)
    for rr, cc in marker_cells:
        patch[int(rr), int(cc)] = int(MARKER)
    return patch


def _make_hole_patch(bg: int, h: int, w: int, marker_cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.full((h, w), int(bg), dtype=int)
    for rr, cc in marker_cells:
        patch[int(rr), int(cc)] = int(MARKER)
    return patch


def _make_assets(
    rects: list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]],
    bg: int,
    base: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    available = [c for c in range(1, 10) if c not in (bg, base, MARKER)]
    if len(available) < MIN_PALETTE_SIZE:
        return None

    pieces: dict[int, np.ndarray] = {}
    holes: dict[int, np.ndarray] = {}
    used_fill_cols: set[int] = set()
    for i, (_r, _c, h, w, marker_cells) in enumerate(rects):
        n_cols = 2 if (h * w >= 4 and random.random() < BICHROMATIC_FRAC) else 1
        cols = random.sample(available, n_cols)
        used_fill_cols.update(cols)
        pieces[i] = _make_piece_patch(h, w, cols, marker_cells)
        holes[i] = _make_hole_patch(bg, h, w, marker_cells)

    if len(used_fill_cols) < MIN_PALETTE_SIZE:
        return None
    return pieces, holes


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for i, (r, c, _h, _w, _cells) in enumerate(rects):
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
    rects: list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]],
    pieces: dict[int, np.ndarray],
    hole_patches: dict[int, np.ndarray],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for i, (r, c, _h, _w, _cells) in enumerate(rects):
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


def _variant_metrics(
    rects: list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]]
) -> dict[str, int]:
    groups: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {}
    for _r, _c, h, w, marker_cells in rects:
        groups.setdefault((int(h), int(w)), []).append(marker_cells)

    ambiguous_buckets = 0
    ambiguous_pieces = 0
    mixed_count_groups = 0
    stepwise_groups = 0
    for patterns in groups.values():
        by_count: dict[int, list[tuple[tuple[int, int], ...]]] = {}
        for marker_cells in patterns:
            by_count.setdefault(len(marker_cells), []).append(marker_cells)
        has_repeat = False
        for bucket in by_count.values():
            if len(bucket) >= 2:
                ambiguous_buckets += 1
                ambiguous_pieces += len(bucket)
                has_repeat = True
                assert len(set(bucket)) == len(bucket)
        if len(by_count) >= 2:
            mixed_count_groups += 1
            if has_repeat:
                stepwise_groups += 1
    return {
        "size_groups": int(len(groups)),
        "ambiguous_buckets": int(ambiguous_buckets),
        "ambiguous_pieces": int(ambiguous_pieces),
        "mixed_count_groups": int(mixed_count_groups),
        "stepwise_groups": int(stepwise_groups),
    }


def _transpose_rects(
    rects: list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]]
) -> list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]]:
    out: list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]] = []
    for r, c, h, w, marker_cells in rects:
        transposed = tuple(sorted((int(cc), int(rr)) for rr, cc in marker_cells))
        out.append((int(c), int(r), int(w), int(h), transposed))
    return out


def _gen_attempt() -> np.ndarray | None:
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

    rects = canon if vertical else _transpose_rects(canon)
    metrics = _variant_metrics(rects)
    if (
        metrics["ambiguous_buckets"] < MIN_AMBIGUOUS_BUCKETS
        or metrics["ambiguous_pieces"] < MIN_AMBIGUOUS_PIECES
        or metrics["stepwise_groups"] < 1
    ):
        return None

    bg, base = random.sample(range(1, 10), 2)
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
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
