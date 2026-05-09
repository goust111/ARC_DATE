# Combo variant of 5dbc8537 (index 24)
# combo_spec: D2/0+D14/1
# Name: 不规则精灵先组装后填入

from __future__ import annotations

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

# Design analysis for the fused variant
# a) Concrete behavior:
#    原题把“完整矩形外部精灵”直接塞进“同 bbox 的矩形洞”。这里改成 3 个彼此分离的不规则洞，
#    外部精灵也都改成不规则 polyomino。一个具体例子是：模板内出现一个 6x6 的阶梯形洞，
#    外部却没有同形状单片，而是两块颜色一致、都带缺角的 L 形片段；解题时要先看出两片可在
#    保持朝向不变的前提下拼成完整洞形，再一起放回。与原题的差异就在于“洞的 bbox 不再足够，
#    必须检查精确 mask，而且部分洞要由 2 片组合覆盖”。
# b) Complexity calibration:
#    如果允许任意数量的片段组装，搜索会过大且样例难以观察；如果所有洞都只需单片，又会退化。
#    因此这里固定每题 3 个洞，其中恰好 2 个是双片组装洞，1 个是单片洞，且洞形来自小型形状库。
#    这样每个样例都会重复展示“有的洞直接填，有的洞先拼再填”。
# c) Original assumptions broken:
#    原 solver 依赖“洞区域能由若干矩形 bbox 精确铺满”“sprite area = bbox area”“每个 sprite
#    对应一个洞”。这些前提在融合后都不成立，所以新 solver 改为：先提取每个洞的精确布尔 mask，
#    再回溯尝试由 1 片或 2 片 source sprite 平移覆盖该洞。
# d) Placement legality:
#    生成器先构造输出，再把每个组内片段拆出来散布到模板外；洞之间保留 base 间隔，外部 sprite
#    之间用 `random_free_location(..., padding=1)` 保证分离，不重叠不越界。
# e) Visual observability:
#    同一复合洞的两片 source sprite 共享同一组颜色与纹理，拼回后会形成连续图案；人眼可从
#    多个样例中看到“同色同纹的两片要组装成一个更大洞形”，而不是仅靠颜色编号做隐式排序。
# f) Visual cleanliness:
#    洞固定为 3 个，彼此分离；外部 sprite 固定为 5 片（2+2+1），避免过度拥挤。所有片段都保留
#    1 格以上背景间距，模板仍是单色长条框架，整体读图清晰。
# g) Inferability of variation:
#    颜色只作为组内连续纹理的辅助线索，真正决定匹配的是不规则形状的精确覆盖。规则在 2-3 组
#    样例中可见：单片洞直接回填，双片洞必须由两片在洞内拼接后回填，且不涉及旋转/镜像分支。
#
# New implicit assumptions audited:
# - base 色仍是最大的非背景连通块；生成器通过让模板面积远大于任何 sprite 保证这一点。
# - 每个洞最多由 2 片覆盖；生成器固定使用 1 片洞或 2 片洞，不产生更高组装度。
# - 所有 source sprite 的目标朝向与输入朝向一致；生成器只做平移散布，不做旋转镜像。
# - 每个洞都有至少一个、且在整题 exact-cover 下唯一一致的 1/2 片覆盖；生成器从受控形状库采样，
#   并用 oracle `main()` 反验证，不满足则丢弃重采。

# --------------- shared constants ---------------

GRID_MIN, GRID_MAX = 15, 25
OUTER_GAP_MIN, OUTER_GAP_MAX = 8, 11
THICK_PAD_MIN, THICK_PAD_MAX = 2, 4
LONG_SLACK_MAX = 3
MAX_RETRIES = 400
GLOBAL_SCALE = 2
HOLE_COUNT = 3
PAIR_HOLES = 2


def _label_array(rows: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


SHAPE_LIBRARY = {
    "single": [
        _label_array(["110", "011"]),
        _label_array(["111", "010"]),
        _label_array(["100", "111"]),
        _label_array(["10", "11", "01"]),
        _label_array(["111", "101"]),
    ],
    "pair": [
        _label_array(["110", "120", "022"]),
        _label_array(["110", "012", "012", "022"]),
        _label_array(["011", "210", "210", "220"]),
        _label_array(["001", "011", "210", "222"]),
    ],
}


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)


def _group_mask_from_labels(labels: np.ndarray, part_id: int | None = None) -> np.ndarray:
    mask = labels > 0 if part_id is None else labels == int(part_id)
    return _scale_mask(mask, GLOBAL_SCALE)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _pattern_patch(mask: np.ndarray, colors: list[int], bg: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    rr, cc = np.indices((h, w))
    patch = np.full((h, w), int(bg), dtype=int)
    if len(colors) == 1:
        patch[mask] = int(colors[0])
        return patch

    a, b = map(int, colors[:2])
    mode = random.choice(["checker", "diag", "split_r", "split_c"])
    if mode == "checker":
        pick_b = ((rr + cc + random.randint(0, 1)) % 2 == 0)
    elif mode == "diag":
        mod = random.choice([2, 3])
        pick_b = ((rr + 2 * cc + random.randint(0, mod - 1)) % mod == 0)
    elif mode == "split_r":
        cut = random.randint(1, h - 1)
        pick_b = rr >= cut
    else:
        cut = random.randint(1, w - 1)
        pick_b = cc >= cut

    pick_b &= mask
    if not pick_b.any() or np.array_equal(pick_b, mask):
        coords = np.argwhere(mask)
        pick_b = np.zeros_like(mask, dtype=bool)
        chosen = coords[::2] if len(coords) >= 2 else coords
        pick_b[chosen[:, 0], chosen[:, 1]] = True

    patch[mask] = a
    patch[pick_b] = b
    return patch


def _build_group(labels: np.ndarray, colors: list[int], bg: int) -> dict:
    full_mask = _group_mask_from_labels(labels)
    group_patch = _pattern_patch(full_mask, colors, bg)
    parts: list[dict] = []
    for part_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        full_part_mask = _group_mask_from_labels(labels, part_id=part_id)
        part_mask, (r0, c0) = _crop_mask(full_part_mask)
        part_patch = crop_bbox(group_patch, (r0, c0, r0 + part_mask.shape[0] - 1, c0 + part_mask.shape[1] - 1))
        part_patch[~part_mask] = int(bg)
        parts.append(
            {
                "mask": part_mask,
                "patch": part_patch,
                "offset": (int(r0), int(c0)),
                "area": int(part_mask.sum()),
            }
        )

    return {
        "hole_mask": full_mask,
        "patch": group_patch,
        "parts": parts,
        "area": int(full_mask.sum()),
        "bbox": tuple(map(int, full_mask.shape)),
        "piece_count": len(parts),
    }


def _pick_group_colors(bg: int, base: int) -> list[list[int]]:
    available = [c for c in range(1, 10) if c not in (bg, base)]
    random.shuffle(available)
    palettes: list[list[int]] = []
    bichromatic_slots = set(random.sample(range(HOLE_COUNT), random.randint(1, 2)))
    idx = 0
    for hole_idx in range(HOLE_COUNT):
        if hole_idx in bichromatic_slots:
            if idx + 2 > len(available):
                random.shuffle(available)
                idx = 0
            palettes.append([int(available[idx]), int(available[idx + 1])])
            idx += 2
        else:
            if idx >= len(available):
                random.shuffle(available)
                idx = 0
            palettes.append([int(available[idx])])
            idx += 1
    return palettes


def _largest_non_bg_color(g: np.ndarray, bg: int) -> int:
    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    return int(base_candidates[0][1])


def _extract_holes(g: np.ndarray, base_col: int) -> tuple[tuple[int, int, int, int], list[dict]]:
    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))
    dst = crop_bbox(g, dst_bbox)
    holes: list[dict] = []
    for coords in components_mask(dst != int(base_col), connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        mask[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        holes.append(
            {
                "bbox": (int(hr0), int(hc0), int(hr1), int(hc1)),
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    assert holes
    return dst_bbox, holes


def _extract_sprites(g: np.ndarray, bg: int, dst_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = map(int, dst_bbox)
    masked = g.copy()
    masked[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros_like(patch, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append(
            {
                "bbox": (pr0, pc0, pr1, pc1),
                "patch": patch,
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    assert sprites
    return sprites


def _placed_mask(shape: tuple[int, int], mask: np.ndarray, top: int, left: int) -> np.ndarray:
    hh, hw = map(int, shape)
    sh, sw = map(int, mask.shape)
    out = np.zeros((hh, hw), dtype=bool)
    out[top : top + sh, left : left + sw] = mask
    return out


def _placements_for_sprite(hole_mask: np.ndarray, sprite: dict) -> list[tuple[int, int, np.ndarray]]:
    hh, hw = map(int, hole_mask.shape)
    sh, sw = map(int, sprite["mask"].shape)
    placements: list[tuple[int, int, np.ndarray]] = []
    if sh > hh or sw > hw:
        return placements
    for rr in range(hh - sh + 1):
        for cc in range(hw - sw + 1):
            sub = hole_mask[rr : rr + sh, cc : cc + sw]
            if bool((sprite["mask"] & ~sub).any()):
                continue
            occ = _placed_mask((hh, hw), sprite["mask"], rr, cc)
            placements.append((int(rr), int(cc), occ))
    return placements


def _hole_cover_candidates(hole: dict, sprites: list[dict]) -> list[list[tuple[int, int, int]]]:
    hole_mask = hole["mask"]
    hole_area = int(hole["area"])
    placements = [_placements_for_sprite(hole_mask, sp) for sp in sprites]
    candidates: list[list[tuple[int, int, int]]] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()

    for i, sp in enumerate(sprites):
        if int(sp["area"]) != hole_area:
            continue
        for rr, cc, occ in placements[i]:
            if np.array_equal(occ, hole_mask):
                key = ((int(i), int(rr), int(cc)),)
                if key not in seen:
                    seen.add(key)
                    candidates.append(list(key))

    for i in range(len(sprites)):
        if not placements[i]:
            continue
        for j in range(i + 1, len(sprites)):
            if int(sprites[i]["area"]) + int(sprites[j]["area"]) != hole_area:
                continue
            if not placements[j]:
                continue
            for rr1, cc1, occ1 in placements[i]:
                for rr2, cc2, occ2 in placements[j]:
                    if bool((occ1 & occ2).any()):
                        continue
                    if np.array_equal(occ1 | occ2, hole_mask):
                        key = tuple(sorted(((int(i), int(rr1), int(cc1)), (int(j), int(rr2), int(cc2)))))
                        if key not in seen:
                            seen.add(key)
                            candidates.append(list(key))

    return candidates


def _solve_cover(holes: list[dict], sprites: list[dict]) -> dict[int, list[tuple[int, int, int]]]:
    candidates = [_hole_cover_candidates(hole, sprites) for hole in holes]
    assert all(candidates)
    order = sorted(range(len(holes)), key=lambda idx: (len(candidates[idx]), -holes[idx]["area"]))

    used = [False] * len(sprites)
    chosen: dict[int, list[tuple[int, int, int]]] = {}

    def dfs(k: int) -> bool:
        if k == len(order):
            return all(used)

        hole_idx = order[k]
        for cand in candidates[hole_idx]:
            ids = [part[0] for part in cand]
            if any(used[idx] for idx in ids):
                continue
            for idx in ids:
                used[idx] = True
            chosen[hole_idx] = cand
            if dfs(k + 1):
                return True
            del chosen[hole_idx]
            for idx in ids:
                used[idx] = False
        return False

    assert dfs(0)
    return chosen


def assembly_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base = _largest_non_bg_color(g, bg)
    dst_bbox, holes = _extract_holes(g, base)
    sprites = _extract_sprites(g, bg, dst_bbox)
    pair_holes = 0
    for hole in holes:
        cands = _hole_cover_candidates(hole, sprites)
        single = sum(1 for cand in cands if len(cand) == 1)
        pair = sum(1 for cand in cands if len(cand) == 2)
        if single == 0 and pair > 0:
            pair_holes += 1
    return {
        "holes": int(len(holes)),
        "sprites": int(len(sprites)),
        "extra_sprites": int(len(sprites) - len(holes)),
        "pair_holes": int(pair_holes),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a strip-shaped template by fitting irregular source sprites into irregular cavities, with some cavities requiring two sprites to be assembled first.

    Input structure:
    - The background is the most common color. One dominant non-background color forms a tall or wide template strip.
    - Inside that strip are several disconnected irregular holes. Outside the strip are irregular multicolor sprites, all separated by background.
    - Some holes correspond to one sprite, while others correspond to two separate sprites that jointly form a larger composite shape.

    Rule:
    - Identify the template by taking the largest non-background connected component and crop to its bounding box.
    - Extract every irregular hole mask inside the template crop and every irregular source sprite outside the crop.
    - For each hole, search for an exact cover by either one translated sprite or two translated sprites, without rotating or mirroring them.
    - Paste the matched sprite patch(es) into the hole positions and return the tight template crop.

    Key insight:
    - Bounding boxes are no longer sufficient: the puzzle is solved by exact mask coverage. The distinctive extra step is to detect when two smaller outside fragments are complementary parts of one larger cavity and must be assembled before filling it.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_col = _largest_non_bg_color(g, bg)
    dst_bbox, holes = _extract_holes(g, base_col)
    sprites = _extract_sprites(g, bg, dst_bbox)
    placement = _solve_cover(holes, sprites)

    r0, c0, r1, c1 = map(int, dst_bbox)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(base_col), dtype=int)
    for hole_idx, parts in placement.items():
        hr0, hc0, _hr1, _hc1 = map(int, holes[hole_idx]["bbox"])
        for sprite_idx, dr, dc in parts:
            sprite = sprites[int(sprite_idx)]
            paste(out, sprite["patch"], hr0 + int(dr), hc0 + int(dc), mask=sprite["mask"], overwrite=True)
    return out


def _sample_groups(bg: int, base: int) -> list[dict] | None:
    singles = random.sample(SHAPE_LIBRARY["single"], 1)
    pairs = random.sample(SHAPE_LIBRARY["pair"], PAIR_HOLES)
    labels_list = pairs + singles
    random.shuffle(labels_list)
    colors = _pick_group_colors(bg, base)
    groups = [_build_group(labels, cols, bg) for labels, cols in zip(labels_list, colors)]
    assert sum(group["piece_count"] for group in groups) == HOLE_COUNT + PAIR_HOLES
    return groups


def _layout_groups(groups: list[dict], vertical: bool) -> tuple[int, int, list[tuple[int, int]]] | None:
    dims = [(int(g["hole_mask"].shape[0]), int(g["hole_mask"].shape[1])) for g in groups]
    widths = [w if vertical else h for h, w in dims]
    heights = [h if vertical else w for h, w in dims]
    max_cross = max(widths)
    thick = max_cross + random.randint(THICK_PAD_MIN, THICK_PAD_MAX)
    if thick > GRID_MAX - OUTER_GAP_MIN:
        return None

    long_needed = sum(heights) + (len(groups) - 1) + 2
    long = long_needed + random.randint(0, LONG_SLACK_MAX)
    long = max(long, GRID_MIN)
    if long > GRID_MAX:
        return None

    positions: list[tuple[int, int]] = []
    cursor = 1 + random.randint(0, max(0, long - long_needed))
    for group, major, cross in zip(groups, heights, widths):
        cross_max = thick - cross - 1
        if cross_max < 1:
            return None
        offset = random.randint(1, cross_max)
        pos = (cursor, offset) if vertical else (offset, cursor)
        positions.append((int(pos[0]), int(pos[1])))
        cursor += major + 1
    return int(long if vertical else thick), int(thick if vertical else long), positions


def _render_output(dh: int, dw: int, base: int, groups: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for group, (r0, c0) in zip(groups, positions):
        for part in group["parts"]:
            pr, pc = map(int, part["offset"])
            paste(out, part["patch"], int(r0 + pr), int(c0 + pc), mask=part["mask"], overwrite=True)
    return out


def _render_input(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    groups: list[dict],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    placement = inp.copy()

    pieces: list[np.ndarray] = []
    for group, (r0, c0) in zip(groups, positions):
        hole_mask = group["hole_mask"]
        inp[tr0 + r0 : tr0 + r0 + hole_mask.shape[0], tc0 + c0 : tc0 + c0 + hole_mask.shape[1]][hole_mask] = int(bg)
        for part in group["parts"]:
            pieces.append(part["patch"])

    order = list(range(len(pieces)))
    random.shuffle(order)
    order.sort(key=lambda idx: int((pieces[idx] != bg).sum()), reverse=True)
    for idx in order:
        patch = pieces[idx]
        rr, cc = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, rr, cc, transparent=bg)
        paste(placement, patch, rr, cc, transparent=bg)
    return inp


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    bg, base = random.sample(range(1, 10), 2)
    groups = _sample_groups(bg, base)
    if groups is None:
        return None

    layout = _layout_groups(groups, vertical)
    if layout is None:
        return None
    dh, dw, positions = layout

    if vertical:
        outer = random.randint(OUTER_GAP_MIN, OUTER_GAP_MAX)
        H, W = dh, dw + outer
        tr0, tc0 = 0, 0 if random.random() < 0.5 else W - dw
    else:
        outer = random.randint(OUTER_GAP_MIN, OUTER_GAP_MAX)
        H, W = dh + outer, dw
        tr0, tc0 = 0 if random.random() < 0.5 else H - dh, 0

    if not (GRID_MIN <= H <= GRID_MAX and GRID_MIN <= W <= GRID_MAX):
        return None

    out = _render_output(dh, dw, base, groups, positions)
    try:
        inp = _render_input(H, W, bg, base, tr0, tc0, dh, dw, groups, positions)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    metric = assembly_metrics(inp)
    if metric["pair_holes"] < PAIR_HOLES or metric["extra_sprites"] < PAIR_HOLES:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo 5dbc8537 input")
