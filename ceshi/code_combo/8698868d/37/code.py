# Combo variant of 8698868d (index 37)
# combo_spec: D9/1+D17/0
# Name: 按序处理消歧重复洞数

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for extra in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.insert(0, extra_s)

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    - 原题是“按洞数一次性并行配对”，且每个洞数只出现一次；因此 frame 与 sprite 之间没有歧义。
#    - 现在洞数允许重复，但 generator 强制样本呈现“较小洞数先唯一、较大洞数后重复”的结构。
#      例如 frame 洞数多重集可为 [0, 1, 2, 2]，sprite 也为 [0, 1, 2, 2]。
#    - 解题时先看 0 洞对象，唯一可配；再看 1 洞对象，也唯一可配；最后才轮到重复的 2 洞组。
#      此时可用候选只剩那两个 2 洞 frame 与两个 2 洞 sprite，再按可见位置顺序一一对应。
#    - 和 D9 单变体不同，这里实现层面不做“全局分桶后一次性配完”，而是显式按洞数升序消元。
#      和原题不同，这里的输入确实包含重复洞数组，不能再依赖“一洞数一对象”的隐式双射。
# b) Complexity calibration:
#    - 为了让“先唯一、后重复”的顺序肉眼可见，N 收紧到 3..6，至少需要一个唯一前缀和一个重复后缀。
#    - 洞数仍来自短连续区间，避免自由度过多；重复组只出现在后缀值，样本里通常能看到 1-2 个重复组。
# c) Hidden assumption audit:
#    - 原 solver 的双重循环默认“同洞数至多一个未使用 sprite”；这里不成立，必须显式维护剩余候选。
#    - 原 generator 用 `{hole_count: sprite}` 建 expected；重复 key 会覆盖，必须改成顺序式多重集配对。
#    - `find_components(... monochromatic=True)` 的返回顺序不能当做人类可见规则，因此重复组只用坐标阅读顺序消歧。
# d) Placement legality:
#    - frame 仍是紧贴的规则网格，sprite 仍放在网格外的背景区，使用 `random_free_location` 保证不重叠、不越界。
# e) Visual observability:
#    - 重复洞数组中的 sprite 保持不同颜色，通常也有不同洞形；输出里能直接看到“同洞数的多个 sprite 进入多个同洞 frame”。
#    - 唯一前缀保证解题者能从样例中看到一个升序处理链，而不是纯粹的同值分桶。
# f) Visual cleanliness:
#    - 输出仍是无重叠的紧凑 frame 网格；每格只有 frame 边框和一个 sprite，不引入额外标记层。
# g) Inferability:
#    - 不使用颜色编号、组件枚举顺序等不可见因素；重复时只依赖输入中 sprite 的空间阅读顺序。


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (3, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
MAX_DISTINCT_HOLE_VALUES = 4
SEED_RETRIES = 50
GEN_RETRIES = 700

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _frame_pos_key(item: dict) -> tuple[int, int]:
    return int(item["out_r"]), int(item["out_c"])


def _sprite_pos_key(item: dict) -> tuple[int, int]:
    return int(item["in_r"]), int(item["in_c"])


def _find_square_regions(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
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
    return frame_size, sizes.get(frame_size, []), sizes.get(sprite_size, [])


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    frame_size, frames_raw, sprites_raw = _find_square_regions(g)
    assert frames_raw

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames_raw]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frames: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            hole_mask = region[1:-1, 1:-1] != int(frame_color)
            _labels, num_holes = label_mask(hole_mask, connectivity=4)
            frames.append(
                {
                    "id": len(frames),
                    "color": frame_color,
                    "holes": int(num_holes),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        bbox = comp["bbox"]
        sprite_region = crop_bbox(g, bbox)
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _labels, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "id": len(sprites),
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
                "in_r": int(bbox[0]),
                "in_c": int(bbox[1]),
            }
        )

    return frame_size, n_rows, n_cols, frames, sprites


def _sequential_pair_by_holes(frames: list[dict], sprites: list[dict]) -> dict[int, dict]:
    matches: dict[int, dict] = {}
    remaining_frames = {int(frame["id"]): frame for frame in frames}
    remaining_sprites = {int(sprite["id"]): sprite for sprite in sprites}

    frame_counts = Counter(int(frame["holes"]) for frame in frames)
    sprite_counts = Counter(int(sprite["holes"]) for sprite in sprites)
    assert frame_counts == sprite_counts

    for hole_count in sorted(frame_counts):
        frame_bucket = sorted(
            [frame for frame in remaining_frames.values() if int(frame["holes"]) == int(hole_count)],
            key=_frame_pos_key,
        )
        sprite_bucket = sorted(
            [sprite for sprite in remaining_sprites.values() if int(sprite["holes"]) == int(hole_count)],
            key=_sprite_pos_key,
        )
        assert len(frame_bucket) == len(sprite_bucket)

        # Unique counts collapse immediately; repeated counts are resolved only after
        # the smaller-count buckets have already been removed from the candidate pool.
        for frame, sprite in zip(frame_bucket, sprite_bucket):
            matches[int(frame["id"])] = sprite
            remaining_frames.pop(int(frame["id"]))
            remaining_sprites.pop(int(sprite["id"]))

    assert not remaining_frames and not remaining_sprites
    return matches


def _legacy_parallel_match(frames: list[dict], sprites: list[dict]) -> dict[int, dict]:
    matches: dict[int, dict] = {}
    used_sprites: set[int] = set()
    for frame in frames:
        for sprite in sprites:
            sprite_id = int(sprite["id"])
            if sprite_id in used_sprites:
                continue
            if int(frame["holes"]) != int(sprite["holes"]):
                continue
            matches[int(frame["id"])] = sprite
            used_sprites.add(sprite_id)
            break
    return matches


def _render_output(
    frame_size: int,
    n_rows: int,
    n_cols: int,
    frames: list[dict],
    matches: dict[int, dict],
) -> np.ndarray:
    out = np.full((n_rows * frame_size, n_cols * frame_size), BG, dtype=int)
    for frame in frames:
        rr = int(frame["out_r"])
        cc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)

        sprite = matches.get(int(frame["id"]))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)
        sh, sw = content.shape
        out[rr + 1 : rr + 1 + sh, cc + 1 : cc + 1 + sw] = content
    return out


def _hole_profile(counts: list[int]) -> dict[str, int]:
    freq = Counter(map(int, counts))
    repeated_values = sorted(k for k, v in freq.items() if int(v) > 1)
    smallest_repeated = repeated_values[0] if repeated_values else -1
    unique_before_repeat = sum(1 for k, v in freq.items() if int(v) == 1 and (smallest_repeated < 0 or int(k) < smallest_repeated))
    return {
        "distinct": int(len(freq)),
        "repeated_groups": int(len(repeated_values)),
        "largest_bucket": int(max(freq.values())),
        "smallest_repeated": int(smallest_repeated),
        "unique_before_repeat": int(unique_before_repeat),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, handling repeats by an explicit ascending sequence.

    Input structure:
    - Background is azure (8). Large same-size square frames form a tight grid, while smaller same-size sprites sit elsewhere on the background.
    - Frame interiors contain isolated bg holes; sprite interiors contain bg hole-components that may span multiple cells.

    Rule:
    - Count 4-connected hole-components for every frame interior and every sprite.
    - Traverse hole counts from small to large, removing matched pairs as you go. Counts that occur once pair immediately; repeated counts are resolved after earlier counts are gone, using reading order inside the remaining equal-count items.
    - Place each matched sprite inside its frame border and repaint the sprite's hole pixels to the frame color.

    Key insight:
    - The governing feature is still the number of hole-components, but repeated counts force a staged elimination process rather than a one-shot bijection.
    """
    g = np.array(grid, dtype=int)
    frame_size, n_rows, n_cols, frames, sprites = _parse_scene(g)
    matches = _sequential_pair_by_holes(frames, sprites)
    return _render_output(frame_size, n_rows, n_cols, frames, matches)


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

    _labels, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if int(num_holes) != int(hole_count):
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

    _labels, num_holes = label_mask(frame != frame_color, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _generate_sequential_hole_multiset(n: int, max_hole: int) -> list[int] | None:
    max_distinct = min(MAX_DISTINCT_HOLE_VALUES, max_hole + 1, n)
    if max_distinct < 2:
        return None

    distinct_n = random.randint(2, max_distinct)
    max_start = max_hole - (distinct_n - 1)
    if max_start < 0:
        return None

    start = random.choice(list(range(0, min(HOLE_START_MAX, max_start) + 1)))
    base = list(range(start, start + distinct_n))
    unique_prefix_len = random.randint(1, distinct_n - 1)
    repeated_pool = base[unique_prefix_len:]
    if not repeated_pool:
        return None

    counts = base[:]
    while len(counts) < n:
        counts.append(random.choice(repeated_pool))

    profile = _hole_profile(counts)
    if profile["repeated_groups"] < 1:
        return None
    if profile["unique_before_repeat"] < 1:
        return None
    return counts


def _has_visible_repeat(sprites: list[dict]) -> bool:
    by_holes: dict[int, list[dict]] = {}
    for sprite in sprites:
        by_holes.setdefault(int(sprite["holes"]), []).append(sprite)

    for group in by_holes.values():
        if len(group) < 2:
            continue
        colors = {int(item["color"]) for item in group}
        if len(colors) >= 2:
            return True
    return False


def _build_expected_output(
    frames: list[dict],
    sprites: list[dict],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
) -> np.ndarray:
    matches = _sequential_pair_by_holes(frames, sprites)
    return _render_output(frame_size, rows_grid, cols_grid, frames, matches)


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
    hole_counts = _generate_sequential_hole_multiset(n, max_hole)
    if hole_counts is None:
        return None

    frame_holes = hole_counts[:]
    sprite_holes = hole_counts[:]
    random.shuffle(frame_holes)
    random.shuffle(sprite_holes)

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
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
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
        sprites.append({**template, "in_r": int(rr), "in_c": int(cc)})

    if not _has_visible_repeat(sprites):
        return None

    profile = _hole_profile(hole_counts)
    if profile["unique_before_repeat"] < 1 or profile["repeated_groups"] < 1:
        return None

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size)

    try:
        out = main(canvas)
    except Exception:
        return None
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    try:
        parsed_frame_size, parsed_rows, parsed_cols, parsed_frames, parsed_sprites = _parse_scene(canvas)
    except Exception:
        return None

    legacy_out = _render_output(
        parsed_frame_size,
        parsed_rows,
        parsed_cols,
        parsed_frames,
        _legacy_parallel_match(parsed_frames, parsed_sprites),
    )
    # Reject any seed where the older repeated-count matcher still works by accident.
    # This keeps the combo behavior observable instead of silently collapsing back to
    # a weaker "just match repeated counts somehow" variant.
    if np.array_equal(legacy_out, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
