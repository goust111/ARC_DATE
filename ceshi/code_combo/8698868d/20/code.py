# Combo variant of 8698868d (index 20)
# combo_spec: D9/1+D15/0
# Name: 重复洞数且精灵过剩

from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   preserved_structure:
#     - 输入仍由一块紧密排列的 frame 网格和若干散落在外部的等尺寸 sprite 组成。
#     - frame_size 仍在 6..8，sprite_size = frame_size - 2，输出仍只保留紧凑 frame 网格。
#     - 基础信号仍是“洞的 4 连通分量数”；sprite 的洞像素在输出里仍要重着色为对应 frame 色。
#   combo_core:
#     - D9/1: frame 的洞数允许重复，因此不能再假设 `洞数 -> frame` 或 `洞数 -> sprite` 是双射。
#     - D15/0: sprite 总数多于 frame，总有干扰项；而且这里的干扰项不只可以是“无关洞数”，
#       也可以是“洞数相同但排位靠后”的同类 sprite。
#     - 融合后的统一规则是：
#       1) 先按洞数把 frame 和 sprite 分桶；
#       2) 每个 frame 桶按网格阅读顺序排序；
#       3) 每个 sprite 桶按输入中的左上角阅读顺序排序；
#       4) 对每个洞数桶，只取前 `len(frame_bucket)` 个 sprite 参与配对；
#       5) 同桶剩余 sprite 与所有不在 frame 洞数集合中的 sprite 都被丢弃。
#   behavior_trace:
#     - 例：frame 网格洞数为 [1, 1, 2]；外部 sprite 洞数为 [1, 2, 1, 1, 4]。
#     - 原题会失败在两处：
#       a) `1` 出现重复，旧的“遇到第一个未使用 sprite 就配”不再是无害实现细节；
#       b) sprite 比 frame 多，且多出来的一个 `1` 不是“异洞数干扰”，必须显式丢弃。
#     - 融合后：
#       - 洞数 1 的两个 frame 按输出左到右排序；
#       - 所有洞数 1 的三个 sprite 按输入位置从上到下、从左到右排序；
#       - 只取前两个 sprite 填入两个 frame，第三个洞数 1 sprite 被舍弃；
#       - 洞数 4 sprite 因为 frame 没有这个需求，也被舍弃。
#   complexity_calibration:
#     - generator 强制至少有一个重复 frame 洞数桶，并强制至少一个 sprite 桶数量严格大于对应 frame 需求，
#       避免样本退化成“唯一洞数 + 只需过滤异洞数”的弱版本。
#     - 同洞数 sprite 颜色互不相同，通常洞形也不同，因此“按位置取前几个”的后果在输出中肉眼可见。
#     - 洞数范围仍限制在一个很短的连续区间附近，避免自由度过大导致人类无法在 2-4 个样例中归纳。
#   hidden_assumption_audit:
#     - 原 solver 的双重循环默认每个洞数只会命中一次；现在必须改成“按桶 + 组内排序”。
#     - 原 generator 的 `{hole_count: sprite}` 字典会在重复洞数时覆盖条目；现在必须保留完整列表。
#     - `find_components(... monochromatic=True)` 的返回顺序不稳定且可能受颜色影响，不能拿它当消歧规则。
#     - 新 solver 假设：每个 frame 桶至少有足够多的 sprite 候选；generator 通过构造和 oracle 校验保证这一点。
#   placement_legality:
#     - frame 网格仍作为一个紧密矩形块粘贴到画布。
#     - 所有 sprite 都用 `random_free_location(..., padding=1)` 放到 frame 网格外的背景区，彼此不重叠、不接触。
#     - 画布大小按 frame 区域 + 全部 sprite 的有效占地面积估下界，降低放置失败率。
#   visual_observability:
#     - 解题者能直接看到：外部 sprite 数量多于 frame，且某些洞数会出现三选二、二选一等“桶内筛选”。
#     - 规则只依赖洞结构和可见位置顺序，不依赖颜色编号、组件枚举顺序等不可视因素。
#   visual_tidiness:
#     - 不额外加入标记层；输入仍是清晰的 frame 块 + 散落 sprite。
#     - 统一尺寸和 padding 让每个候选 sprite 都能被独立辨认。


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
MAX_DISTINCT_HOLE_VALUES = 4
EXTRA_SAME_COUNT_SPRITES = 1
EXTRA_OUTSIDE_COUNT_SPRITES = 1
SEED_RETRIES = 50
GEN_RETRIES = 700


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        _n = _r * _c
        if N_RANGE[0] <= _n <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_n, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _count_holes(region: np.ndarray, solid_color: int) -> int:
    hole_mask = np.array(region, dtype=int) != int(solid_color)
    _lab, num_holes = label_mask(hole_mask, connectivity=4)
    return int(num_holes)


def _max_isolated_count(size: int) -> int:
    inner_side = int(size) - 2
    if inner_side <= 0:
        return 0
    return int((inner_side * inner_side + 1) // 2)


def _find_square_regions(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        bbox = comp["bbox"]
        h = int(bbox_h(bbox))
        w = int(bbox_w(bbox))
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
    return frame_size, sprite_size, frames_raw, sprites_raw


def _extract_scene_signature(grid: np.ndarray) -> dict:
    frame_size, sprite_size, frames_raw, sprites_raw = _find_square_regions(grid)

    frame_positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames_raw]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frames: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = grid[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            holes = _count_holes(region[1:-1, 1:-1], frame_color)
            frames.append(
                {
                    "id": len(frames),
                    "color": frame_color,
                    "holes": int(holes),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        bbox = comp["bbox"]
        sprite_region = crop_bbox(grid, bbox)
        sprite_color = int(comp["color"])
        sprites.append(
            {
                "id": len(sprites),
                "color": sprite_color,
                "holes": _count_holes(sprite_region, sprite_color),
                "content": np.array(sprite_region, dtype=int).copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
                "in_r": int(bbox[0]),
                "in_c": int(bbox[1]),
            }
        )

    frame_need = Counter(int(frame["holes"]) for frame in frames)
    return {
        "frame_size": int(frame_size),
        "sprite_size": int(sprite_size),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "frames": frames,
        "sprites": sprites,
        "frame_need": frame_need,
    }


def _pair_frames_and_sprites(frames: list[dict], sprites: list[dict]) -> dict[int, dict]:
    frame_buckets: dict[int, list[dict]] = {}
    sprite_buckets: dict[int, list[dict]] = {}
    for frame in frames:
        frame_buckets.setdefault(int(frame["holes"]), []).append(frame)
    for sprite in sprites:
        sprite_buckets.setdefault(int(sprite["holes"]), []).append(sprite)

    matches: dict[int, dict] = {}
    for hole_count, frame_group in frame_buckets.items():
        ordered_frames = sorted(frame_group, key=lambda item: (int(item["out_r"]), int(item["out_c"])))
        ordered_sprites = sorted(
            sprite_buckets.get(int(hole_count), []),
            key=lambda item: (int(item["in_r"]), int(item["in_c"])),
        )
        assert len(ordered_sprites) >= len(ordered_frames)
        for frame, sprite in zip(ordered_frames, ordered_sprites[: len(ordered_frames)]):
            matches[int(frame["id"])] = sprite
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, while handling repeated counts
    and extra sprite candidates.

    Input structure:
    - bg is azure (8). Large square frames form a tight grid; smaller square sprites are scattered elsewhere.
    - Several frames may share the same hole count, and the sprite pool can be larger than the frame pool.
    - Extra sprites may either have irrelevant hole counts or duplicate a relevant hole count.

    Rule:
    - Count 4-connected hole regions inside each frame interior and each sprite.
    - Group objects by hole count.
    - Within one hole-count bucket, sort frames by grid reading order and sprites by their input positions.
    - Pair the first needed sprites to the frames in that bucket; discard every leftover sprite.
    - Nest each chosen sprite into its frame and recolor the sprite's hole pixels to the frame color.

    Key insight:
    - Hole count is still the primary key, but it is no longer enough by itself.
    - Repeated counts and surplus sprites are resolved by a visible secondary rule: position order inside each equal-count bucket.
    """
    g = np.array(grid, dtype=int)
    scene = _extract_scene_signature(g)
    frame_size = int(scene["frame_size"])
    frame_grid = scene["frames"]
    matches = _pair_frames_and_sprites(frame_grid, scene["sprites"])

    out_h = int(scene["n_rows"] * frame_size)
    out_w = int(scene["n_cols"] * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

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

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
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

    if _count_holes(sprite, sprite_color) != int(hole_count):
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

    if _count_holes(frame, frame_color) != int(hole_count):
        return None
    return frame


def _generate_frame_hole_multiset(n_frames: int, max_hole: int) -> list[int] | None:
    if n_frames < 3:
        return None

    max_distinct = min(MAX_DISTINCT_HOLE_VALUES, max_hole + 1, n_frames - 1)
    if max_distinct < 2:
        return None

    distinct_n = random.randint(2, max_distinct)
    max_start = max_hole - (distinct_n - 1)
    if max_start < 0:
        return None

    start_candidates = list(range(0, min(HOLE_START_MAX, max_start) + 1))
    if not start_candidates:
        return None
    start = random.choice(start_candidates)

    base = list(range(start, start + distinct_n))
    counts = base[:]
    extra_pool = [value for value in base if value > 0] or base[:]
    while len(counts) < n_frames:
        counts.append(random.choice(extra_pool))

    if len(set(counts)) == n_frames:
        return None
    return counts


def _build_expected_output(scene: dict) -> np.ndarray:
    frame_size = int(scene["frame_size"])
    expected = np.full((int(scene["n_rows"]) * frame_size, int(scene["n_cols"]) * frame_size), BG, dtype=int)
    matches = _pair_frames_and_sprites(scene["frames"], scene["sprites"])

    for frame in scene["frames"]:
        rr = int(frame["out_r"])
        cc = int(frame["out_c"])
        frame_color = int(frame["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)

        sprite = matches[int(frame["id"])]
        content = np.array(sprite["content"], dtype=int).copy()
        content[content == BG] = int(frame_color)
        sh, sw = content.shape
        expected[rr + 1 : rr + 1 + sh, cc + 1 : cc + 1 + sw] = content

    return expected


def _has_active_combo(scene: dict) -> bool:
    frame_need: Counter = scene["frame_need"]
    if len(scene["sprites"]) <= len(scene["frames"]):
        return False
    if len(set(int(frame["holes"]) for frame in scene["frames"])) == len(scene["frames"]):
        return False

    sprite_count_by_hole = Counter(int(sprite["holes"]) for sprite in scene["sprites"])
    if not any(int(sprite_count_by_hole[hole]) > int(need) for hole, need in frame_need.items()):
        return False
    if not any(int(hole) not in frame_need for hole in sprite_count_by_hole):
        return False

    overflowing_holes = [int(hole) for hole, need in frame_need.items() if int(sprite_count_by_hole[hole]) > int(need)]
    for hole in overflowing_holes:
        ordered = sorted(
            [sprite for sprite in scene["sprites"] if int(sprite["holes"]) == int(hole)],
            key=lambda item: (int(item["in_r"]), int(item["in_c"])),
        )
        if len(ordered) <= int(frame_need[hole]):
            continue
        chosen = ordered[: int(frame_need[hole])]
        discarded = ordered[int(frame_need[hole]) :]
        chosen_signatures = {(int(item["color"]), item["content"].tobytes()) for item in chosen}
        if any((int(item["color"]), item["content"].tobytes()) not in chosen_signatures for item in discarded):
            return True
    return False


def _choose_instance_params() -> tuple[int, int, int, int]:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    max_sprite_holes = _max_isolated_count(sprite_size)
    n_choices = [n for n in _VALID_NS if n >= 3 and n + EXTRA_SAME_COUNT_SPRITES + EXTRA_OUTSIDE_COUNT_SPRITES <= 9]
    assert n_choices
    n_frames = random.choice(n_choices)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    return frame_size, sprite_size, rows_grid, cols_grid


def _gen_attempt() -> np.ndarray | None:
    frame_size, sprite_size, rows_grid, cols_grid = _choose_instance_params()
    n_frames = rows_grid * cols_grid
    max_sprite_holes = _max_isolated_count(sprite_size)

    frame_holes = _generate_frame_hole_multiset(n_frames, max_sprite_holes)
    if frame_holes is None:
        return None

    repeated_holes = [hole for hole, count in Counter(frame_holes).items() if int(count) >= 2]
    if not repeated_holes:
        return None

    outsider_pool = [hole for hole in range(max_sprite_holes + 1) if hole not in set(frame_holes)]
    if not outsider_pool:
        return None

    total_sprites = n_frames + EXTRA_SAME_COUNT_SPRITES + EXTRA_OUTSIDE_COUNT_SPRITES
    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    effective_sprite_span = sprite_size + 2 * SPRITE_PLACEMENT_PAD
    needed_area = grid_pixel_h * grid_pixel_w + total_sprites * effective_sprite_span * effective_sprite_span + 12
    side_lb = max(CANVAS_MIN, grid_pixel_h, grid_pixel_w, int(math.ceil(math.sqrt(needed_area))))
    if side_lb > CANVAS_MAX:
        return None

    roomy_lb = max(side_lb, CANVAS_MAX - 5)
    height = random.randint(roomy_lb, CANVAS_MAX)
    width = random.randint(roomy_lb, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, total_sprites)

    random.shuffle(frame_holes)
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

    sprite_holes = frame_holes[:] + [random.choice(repeated_holes)] + [random.choice(outsider_pool)]
    random.shuffle(sprite_holes)
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

    canvas = np.full((height, width), BG, dtype=int)
    r0 = random.randint(0, height - grid_pixel_h)
    c0 = random.randint(0, width - grid_pixel_w)
    for frame in frames:
        rr = int(r0 + frame["out_r"])
        cc = int(c0 + frame["out_c"])
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    random.shuffle(sprite_templates)
    for template in sprite_templates:
        patch = template["content"]
        # Reserve the full sprite box during placement so hole pixels never reveal
        # a pre-existing object underneath.
        footprint = np.full_like(patch, int(template["color"]))
        try:
            rr, cc = random_free_location(canvas, footprint, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, rr, cc, transparent=BG)

    try:
        scene = _extract_scene_signature(canvas)
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(scene)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    if not _has_active_combo(scene):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
