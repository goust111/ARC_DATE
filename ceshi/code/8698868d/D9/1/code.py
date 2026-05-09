# Variant of 8698868d
# Dimension: D9 — 洞计数范围与分配 (meta)
# Change: 洞计数为连续整数 0..N-1，每个值恰好出现一次（双射配对）... → 洞计数允许重复（非双射）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the D9 variant
# ----------------------------------
# spec:
#   preserved_structure:
#     - bg 固定为 8，输入仍由一组紧密框架网格 + 散落在外部的等尺寸 sprite 构成。
#     - frame_size 仍为 6..8，sprite_size = frame_size - 2，输出仍是紧凑框架网格。
#     - 主匹配特征仍是“洞的 4 连通分量个数”，颜色和洞面积都不是主键。
#   variant_core:
#     - 原题每个洞数只出现一次，因此 `洞数 -> sprite` 是双射。
#     - 现在 frame 和 sprite 的洞数只要求“多重集相同”，允许同一洞数出现多次。
#     - 当某个洞数 k 出现多次时，次级规则是：
#       frame 组按网格阅读顺序（上到下、左到右）排序；
#       sprite 组按输入中左上角坐标的阅读顺序排序；
#       第 i 个 frame 配第 i 个 sprite。
#   behavior_trace:
#     - 例：两个 frame 都有 1 个洞，位置分别在网格左、右；两个 sprite 也都 1 个洞，
#       但一个在输入更靠上、另一个更靠下。原 solver 会按组件枚举顺序“先遇到谁就配谁”，
#       在原题里这是无害的；现在必须显式按位置顺序配对，否则同洞数组会错配。
#     - 这会直接影响输出中“哪一种 sprite 图案/颜色”进入同洞数的哪个 frame。
#   complexity_calibration:
#     - 只增加一层位置顺序消歧，不引入新的几何变换、颜色规则或额外对象。
#     - 为了让规则肉眼可见，generator 强制至少有一个重复洞数组，
#       并避免所有重复组都退化成“颜色顺序刚好等于位置顺序”。
#   hidden_assumption_audit:
#     - 原 `main()` 的匹配循环：
#         for f in frame_grid:
#             for s in sprites:
#                 if same_holes and unused:
#                     match
#       依赖“每个洞数唯一”，否则“第一个未使用 sprite”会变成隐式二级规则。
#     - 原 generator 的 `_build_expected_output()` 用 `{k: sprite}` 字典，
#       在重复洞数时会覆盖同 key 项，必须改成“按洞数分组 + 组内排序配对”。
#     - `find_components(... monochromatic=True)` 对 sprite 的返回顺序受颜色影响；
#       这在原题无害，在本变体里不能再拿它当配对顺序。
#   placement_legality:
#     - frame 网格仍然整齐贴合放置。
#     - sprite 仍通过 `random_free_location` 放到网格外背景区，互不重叠、不越界。
#     - 生成后继续用 `main(input) == expected` 做 oracle 校验。
#   visual_observability:
#     - 重复洞数组中的不同 sprite 至少会有不同颜色，通常也有不同内部洞形。
#     - 输出保留 sprite 实色像素，因此解题者能观察“同洞数时按外部 sprite 的位置顺序入框”。


def _pair_by_hole_count(
    frames: list[dict],
    sprites: list[dict],
    *,
    frame_pos_key,
    sprite_pos_key,
) -> dict[int, dict]:
    """Pair equal-hole items, breaking ties by reading order within each hole-count bucket."""
    frames_by_holes: dict[int, list[dict]] = {}
    sprites_by_holes: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_holes.setdefault(int(frame["holes"]), []).append(frame)
    for sprite in sprites:
        sprites_by_holes.setdefault(int(sprite["holes"]), []).append(sprite)

    matches: dict[int, dict] = {}
    for hole_count in sorted(frames_by_holes):
        fgroup = sorted(frames_by_holes[hole_count], key=frame_pos_key)
        sgroup = sorted(sprites_by_holes.get(hole_count, []), key=sprite_pos_key)
        for frame, sprite in zip(fgroup, sgroup):
            matches[int(frame["id"])] = sprite
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count.
    If a hole count repeats, pair equal-count frames/sprites by reading order.
    """
    g = np.array(grid, dtype=int)
    bg = 8

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames_raw = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames_raw

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames_raw]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            inner = region[1:-1, 1:-1]
            hole_mask = inner != int(frame_color)
            _lab, num_holes = label_mask(hole_mask, connectivity=4)
            frame_grid.append(
                {
                    "id": len(frame_grid),
                    "color": frame_color,
                    "holes": int(num_holes),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "id": len(sprites),
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
                "in_r": int(b[0]),
                "in_c": int(b[1]),
            }
        )

    matches = _pair_by_hole_count(
        frame_grid,
        sprites,
        frame_pos_key=lambda item: (int(item["out_r"]), int(item["out_c"])),
        sprite_pos_key=lambda item: (int(item["in_r"]), int(item["in_c"])),
    )

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite = matches.get(int(frame["id"]))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


# --------------- generator ---------------

BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
MAX_DISTINCT_HOLE_VALUES = 4
SEED_RETRIES = 50
GEN_RETRIES = 600

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    inner_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_hole:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(comps)
                        if j != ci
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    _, num = label_mask(sprite != sprite_color, connectivity=4)
    if int(num) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _, num = label_mask(frame != frame_color, connectivity=4)
    if int(num) != int(hole_count):
        return None
    return frame


def _generate_hole_multiset(n: int, max_hole: int) -> list[int] | None:
    """Return a repeated-count multiset drawn from a short consecutive range."""
    if n < 2:
        return None

    max_distinct = min(MAX_DISTINCT_HOLE_VALUES, max_hole + 1, n - 1)
    if max_distinct < 1:
        return None

    distinct_n = 1 if n == 2 else random.randint(1, max_distinct)
    max_start = max_hole - (distinct_n - 1)
    if max_start < 0:
        return None

    start_candidates = list(range(0, min(HOLE_START_MAX, max_start) + 1))
    if not start_candidates:
        return None

    if distinct_n == 1 and 1 in start_candidates:
        start = 1 if random.random() < 0.75 else random.choice(start_candidates)
    else:
        start = random.choice(start_candidates)

    base = list(range(start, start + distinct_n))
    counts = base[:]

    extra_pool = base[:]
    if len(extra_pool) > 1:
        positive = [v for v in extra_pool if v > 0]
        if positive:
            extra_pool = positive + [v for v in extra_pool if v == 0]

    while len(counts) < n:
        counts.append(random.choice(extra_pool))

    if len(set(counts)) == n:
        return None
    return counts


def _has_active_tiebreak(sprite_records: list[dict]) -> bool:
    """Reject degenerate samples where repeated groups still match old color-order behavior."""
    by_holes: dict[int, list[dict]] = {}
    for sprite in sprite_records:
        by_holes.setdefault(int(sprite["holes"]), []).append(sprite)

    found_repeat = False
    for hole_count, group in by_holes.items():
        if len(group) < 2:
            continue
        found_repeat = True
        by_pos = [int(item["id"]) for item in sorted(group, key=lambda item: (int(item["in_r"]), int(item["in_c"])))]
        by_color = [
            int(item["id"])
            for item in sorted(group, key=lambda item: (int(item["color"]), int(item["in_r"]), int(item["in_c"])))
        ]
        if by_pos != by_color:
            return True
    return False if found_repeat else False


def _build_expected_output(
    frames: list[dict],
    sprites: list[dict],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    matches = _pair_by_hole_count(
        frames,
        sprites,
        frame_pos_key=lambda item: (int(item["out_r"]), int(item["out_c"])),
        sprite_pos_key=lambda item: (int(item["in_r"]), int(item["in_c"])),
    )

    for frame in frames:
        rr = int(frame["out_r"])
        cc = int(frame["out_c"])
        fc = int(frame["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(fc)

        sprite = matches.get(int(frame["id"]))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        content[content == BG] = int(fc)
        sh, sw = content.shape
        expected[rr + 1 : rr + 1 + sh, cc + 1 : cc + 1 + sw] = content

    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n)
    sprite_colors = random.sample(available, n)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    hole_counts = _generate_hole_multiset(n, max_hole)
    if hole_counts is None:
        return None

    frame_holes = hole_counts[:]
    sprite_holes = hole_counts[:]
    random.shuffle(frame_holes)
    random.shuffle(sprite_holes)

    if len(set(frame_holes)) == n:
        return None

    frames: list[dict] = []
    for idx, hole_count in enumerate(frame_holes):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[idx]))
        if patch is None:
            return None
        ri, ci = divmod(idx, cols_grid)
        frames.append(
            {
                "id": idx,
                "holes": int(hole_count),
                "color": int(frame_colors[idx]),
                "patch": patch,
                "out_r": int(ri * frame_size),
                "out_c": int(ci * frame_size),
            }
        )

    sprite_templates: list[dict] = []
    for idx, hole_count in enumerate(sprite_holes):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(sprite_colors[idx]))
        if patch is None:
            return None
        sprite_templates.append(
            {
                "id": idx,
                "holes": int(hole_count),
                "color": int(sprite_colors[idx]),
                "content": patch,
            }
        )

    canvas = np.full((h, w), BG, dtype=int)
    r0 = random.randint(0, h - grid_pixel_h)
    c0 = random.randint(0, w - grid_pixel_w)
    for frame in frames:
        rr = int(r0 + frame["out_r"])
        cc = int(c0 + frame["out_c"])
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    sprites: list[dict] = []
    for template in sprite_templates:
        patch = template["content"]
        try:
            rr, cc = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, rr, cc, transparent=BG)
        sprites.append(
            {
                **template,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "in_r": int(rr),
                "in_c": int(cc),
            }
        )

    if not _has_active_tiebreak(sprites):
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
