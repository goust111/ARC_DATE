from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


RNG = np.random.default_rng()
BG_COLOR = int(Color.BLACK)
MAX_GENERATION_ATTEMPTS = 500


def _bbox_hw(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_solid_rectangle(comp: dict) -> bool:
    h, w = _bbox_hw(comp["bbox"])
    return int(comp["area"]) == h * w


def _scene_parts(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid is too small"

    bg = int(get_bg(g))
    ground_color = int(g[-1, 0])
    assert np.all(g[-1, :] == ground_color), "expected a solid ground row"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    ground_count = 0
    teeth: list[dict] = []
    keys: list[dict] = []

    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == h - 1 and r1 == h - 1 and c0 == 0 and c1 == w - 1 and int(comp["color"]) == ground_color:
            ground_count += 1
        elif r1 == h - 2 and _is_solid_rectangle(comp):
            teeth.append(comp)
        else:
            keys.append(comp)

    assert ground_count == 1, "expected exactly one ground row component"
    assert len(teeth) >= 2, "expected matched teeth plus one distractor"
    assert len(keys) >= 1, "expected at least one floating key block"
    return bg, ground_color, teeth, keys


def _down_notch_info(sprite: np.ndarray, bg: int) -> dict | None:
    h, w = map(int, sprite.shape)
    fg = sprite != int(bg)
    if not fg.any():
        return None
    hole = sprite == int(bg)
    if not hole.any():
        return None

    coords = np.argwhere(hole)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    notch_h = int(r1 - r0 + 1)
    notch_w = int(c1 - c0 + 1)
    if int(coords.shape[0]) != notch_h * notch_w:
        return None
    if r1 != h - 1:
        return None
    if r0 == 0 or c0 == 0 or c1 == w - 1:
        return None
    if not np.all(sprite[r0 : r1 + 1, c0 : c1 + 1] == int(bg)):
        return None

    mask = np.ones_like(sprite, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    if not np.all(sprite[mask] != int(bg)):
        return None

    return {
        "width": notch_w,
        "depth": notch_h,
        "offset": int(c0),
    }


def _orient_key(sprite: np.ndarray, bg: int) -> tuple[np.ndarray, dict, int]:
    candidates: list[tuple[np.ndarray, dict, int]] = []
    for k in range(4):
        rotated = np.rot90(sprite, k)
        info = _down_notch_info(rotated, bg)
        if info is not None:
            candidates.append((rotated, info, k))
    assert len(candidates) == 1, "expected exactly one rotation with a downward notch"
    return candidates[0]


def _component_sprite(grid: np.ndarray, comp: dict) -> np.ndarray:
    return crop_bbox(np.asarray(grid, dtype=int), comp["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each notched block so its notch faces down, then drop it onto the uniquely matching tooth.

    Input structure:
    - The bottom row is a solid ground strip on a plain background.
    - Several solid rectangular teeth stand on the ground, and one extra tooth is a distractor.
    - Floating monochrome blocks are rectangular except for one rectangular notch cut from exactly one side.

    Rule:
    - Identify the solid ground-touching rectangles as teeth and the non-rectangular floating components as keys.
    - For each key, try the four quarter-turn rotations and keep the unique one whose notch faces downward.
    - Match that notch by exact geometry `(width, depth)` to the unique tooth with the same `(width, height)`, then place the key so the tooth sits inside the notch and the key's bottom edge rests on the ground.
    - Remove the original floating keys, leave the distractor tooth unused, and preserve the ground and teeth.

    Key insight:
    - Color is unreliable; the only stable pairing signal is the notch geometry after first normalizing every key by rotation so the notch points downward.
    """
    g = np.asarray(grid, dtype=int)
    bg, _ground_color, teeth, keys = _scene_parts(g)
    h, w = g.shape

    tooth_by_shape: dict[tuple[int, int], dict] = {}
    for tooth in teeth:
        tooth_h, tooth_w = _bbox_hw(tooth["bbox"])
        shape = (int(tooth_w), int(tooth_h))
        assert shape not in tooth_by_shape, "teeth shapes must be unique"
        tooth_by_shape[shape] = tooth

    out = g.copy()
    for key in keys:
        for r, c in map(tuple, key["coords"]):
            out[int(r), int(c)] = bg

    placements: list[tuple[int, int, np.ndarray]] = []
    matched_shapes: set[tuple[int, int]] = set()
    for key in keys:
        sprite = _component_sprite(g, key)
        down_sprite, notch, _turns = _orient_key(sprite, bg)
        shape = (int(notch["width"]), int(notch["depth"]))
        assert shape in tooth_by_shape, "each notch must match one tooth"
        assert shape not in matched_shapes, "matching must be one-to-one"
        matched_shapes.add(shape)

        tooth = tooth_by_shape[shape]
        _tooth_r0, tooth_c0, tooth_r1, _tooth_c1 = map(int, tooth["bbox"])
        assert tooth_r1 == h - 2
        top = h - 1 - down_sprite.shape[0]
        left = tooth_c0 - int(notch["offset"])
        assert 0 <= top and 0 <= left and left + down_sprite.shape[1] <= w
        placements.append((top, left, down_sprite))

    placements.sort(key=lambda item: item[1])
    for top, left, sprite in placements:
        patch_mask = sprite != bg
        dst = out[top : top + sprite.shape[0], left : left + sprite.shape[1]]
        assert not np.any(patch_mask & (dst != bg)), "placed keys must not overlap teeth or each other"
        paste(out, sprite, top, left, transparent=bg)

    return out


def _build_key_sprite(
    tooth_w: int,
    tooth_h: int,
    left_shoulder: int,
    right_shoulder: int,
    top_cap: int,
    color: int,
    bg: int,
) -> np.ndarray:
    outer_h = int(top_cap + tooth_h)
    outer_w = int(left_shoulder + tooth_w + right_shoulder)
    sprite = np.full((outer_h, outer_w), int(color), dtype=int)
    sprite[outer_h - tooth_h : outer_h, left_shoulder : left_shoulder + tooth_w] = int(bg)
    return sprite


def _choose_pair_shapes(pair_count: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    pool = [(w, h) for w in range(1, 4) for h in range(2, 6)]
    RNG.shuffle(pool)
    matched = pool[:pair_count]

    for base_w, base_h in matched:
        choices = [(w, base_h) for w in range(1, 4) if w != base_w and (w, base_h) not in matched]
        if choices:
            return matched, choices[int(RNG.integers(len(choices)))]

    for base_w, base_h in matched:
        choices = [(base_w, h) for h in range(2, 6) if h != base_h and (base_w, h) not in matched]
        if choices:
            return matched, choices[int(RNG.integers(len(choices)))]

    raise RuntimeError("failed to find a distractor tooth shape")


def _choose_colors(pair_count: int) -> tuple[int, list[int], list[int], int]:
    colors = list(range(1, 10))
    RNG.shuffle(colors)
    ground = int(colors.pop())
    tooth_colors = [int(colors.pop()) for _ in range(pair_count + 1)]
    distractor_color = int(tooth_colors[-1])
    key_colors = [0] * pair_count

    same_idx = int(RNG.integers(pair_count))
    diff_idx = (same_idx + 1) % pair_count
    key_colors[same_idx] = int(tooth_colors[same_idx])

    misleading = [distractor_color] + [int(tooth_colors[i]) for i in range(pair_count) if i != diff_idx]
    misleading = [color for color in misleading if color != int(tooth_colors[diff_idx])]
    key_colors[diff_idx] = int(misleading[int(RNG.integers(len(misleading)))])

    remaining_choices = [c for c in range(1, 10) if c != ground]
    for idx in range(pair_count):
        if key_colors[idx] != 0:
            continue
        if RNG.random() < 0.5:
            key_colors[idx] = int(tooth_colors[idx])
        else:
            choices = [c for c in remaining_choices if c != int(tooth_colors[idx])]
            key_colors[idx] = int(choices[int(RNG.integers(len(choices)))])

    assert any(key_colors[i] == tooth_colors[i] for i in range(pair_count))
    assert any(key_colors[i] != tooth_colors[i] for i in range(pair_count))
    assert any(key_colors[i] == distractor_color for i in range(pair_count)) or any(
        key_colors[i] == tooth_colors[j]
        for i in range(pair_count)
        for j in range(pair_count)
        if i != j
    )
    return ground, tooth_colors[:-1], key_colors, distractor_color


def _layout_ground_items(
    key_specs: list[dict],
    distractor_shape: tuple[int, int],
) -> tuple[int, int, dict[int, int], int]:
    items: list[tuple[str, int, int]] = []
    for idx, spec in enumerate(key_specs):
        items.append(("key", idx, int(spec["final_w"])))
    items.append(("distractor", -1, int(distractor_shape[0])))
    RNG.shuffle(items)

    widths = [width for _, _, width in items]
    min_total = sum(widths) + len(items) + 1
    if min_total > 30:
        raise RuntimeError("layout is too wide")

    gaps = np.ones(len(items) + 1, dtype=int)
    extra = int(RNG.integers(0, min(30 - min_total, 6) + 1))
    for _ in range(extra):
        gaps[int(RNG.integers(len(gaps)))] += 1

    item_starts: dict[int, int] = {}
    distractor_start = -1
    cursor = int(gaps[0])
    for item_index, (kind, idx, width) in enumerate(items):
        if kind == "key":
            item_starts[idx] = cursor
        else:
            distractor_start = cursor
        cursor += int(width) + int(gaps[item_index + 1])

    assert distractor_start >= 0
    width = int(cursor)
    assert width <= 30

    max_tooth_h = max([int(spec["tooth_h"]) for spec in key_specs] + [int(distractor_shape[1])])
    max_key_h = max(int(spec["canonical"].shape[0]) for spec in key_specs)
    min_height = max(max_key_h + max_tooth_h + 7, 15)
    max_height = min(30, min_height + 6)
    if min_height > max_height:
        raise RuntimeError("layout is too tall")
    height = int(RNG.integers(min_height, max_height + 1))
    return height, width, item_starts, distractor_start


def _paint_ground_scene(
    height: int,
    width: int,
    ground_color: int,
    key_specs: list[dict],
    tooth_colors: list[int],
    distractor_shape: tuple[int, int],
    distractor_color: int,
    item_starts: dict[int, int],
    distractor_start: int,
) -> tuple[np.ndarray, list[dict], dict]:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[-1, :] = int(ground_color)

    tooth_meta: list[dict] = []
    for idx, spec in enumerate(key_specs):
        start = int(item_starts[idx])
        tooth_c = start + int(spec["left_shoulder"])
        tooth_w = int(spec["tooth_w"])
        tooth_h = int(spec["tooth_h"])
        grid[height - 1 - tooth_h : height - 1, tooth_c : tooth_c + tooth_w] = int(tooth_colors[idx])
        tooth_meta.append(
            {
                "shape": (tooth_w, tooth_h),
                "color": int(tooth_colors[idx]),
                "start": tooth_c,
                "height": tooth_h,
                "key_start": start,
            }
        )

    dist_w, dist_h = map(int, distractor_shape)
    grid[height - 1 - dist_h : height - 1, distractor_start : distractor_start + dist_w] = int(distractor_color)
    distractor_meta = {"shape": (dist_w, dist_h), "color": int(distractor_color), "start": int(distractor_start)}
    return grid, tooth_meta, distractor_meta


def _find_bbox_positions(
    occupied: np.ndarray,
    sprite_h: int,
    sprite_w: int,
    max_bottom: int,
    pad: int,
) -> list[tuple[int, int]]:
    height, width = occupied.shape
    last_top = int(max_bottom - sprite_h + 1)
    if last_top < 1:
        return []

    positions: list[tuple[int, int]] = []
    for r in range(1, last_top + 1):
        for c in range(1, width - sprite_w):
            r0 = max(0, r - pad)
            c0 = max(0, c - pad)
            r1 = min(height, r + sprite_h + pad)
            c1 = min(width, c + sprite_w + pad)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _place_floating_keys(
    grid: np.ndarray,
    key_specs: list[dict],
    max_tooth_h: int,
) -> list[dict]:
    occupied = grid != BG_COLOR
    placements: list[dict] = []
    air_bottom = grid.shape[0] - max_tooth_h - 5
    order = sorted(range(len(key_specs)), key=lambda idx: int(key_specs[idx]["input_sprite"].size), reverse=True)

    for idx in order:
        sprite = key_specs[idx]["input_sprite"]
        h, w = map(int, sprite.shape)
        positions = _find_bbox_positions(occupied, h, w, air_bottom, pad=1)
        if not positions:
            raise RuntimeError("failed to place floating keys")
        r, c = positions[int(RNG.integers(len(positions)))]
        paste(grid, sprite, r, c, transparent=BG_COLOR)
        occupied[r : r + h, c : c + w] = True
        placements.append({"index": idx, "row": r, "col": c})

    placements.sort(key=lambda item: item["index"])
    return placements


def _expected_output(
    grid: np.ndarray,
    key_specs: list[dict],
    floating_placements: list[dict],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for placement, spec in zip(floating_placements, key_specs):
        sprite = spec["input_sprite"]
        r = int(placement["row"])
        c = int(placement["col"])
        mask = sprite != BG_COLOR
        out[r : r + sprite.shape[0], c : c + sprite.shape[1]][mask] = BG_COLOR

    for spec in sorted(key_specs, key=lambda item: int(item["final_col"])):
        final_row = int(spec["final_row"])
        final_col = int(spec["final_col"])
        sprite = spec["canonical"]
        dst = out[final_row : final_row + sprite.shape[0], final_col : final_col + sprite.shape[1]]
        assert not np.any((sprite != BG_COLOR) & (dst != BG_COLOR))
        paste(out, sprite, final_row, final_col, transparent=BG_COLOR)
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    pair_count = int(RNG.integers(2, 4))
    matched_shapes, distractor_shape = _choose_pair_shapes(pair_count)
    ground_color, tooth_colors, key_colors, distractor_color = _choose_colors(pair_count)

    key_specs: list[dict] = []
    rotations: list[int] = []
    for idx, (tooth_w, tooth_h) in enumerate(matched_shapes):
        left_shoulder = int(RNG.integers(1, 3))
        right_shoulder = int(RNG.integers(1, 3))
        top_cap = int(RNG.integers(1, 4))
        canonical = _build_key_sprite(
            tooth_w,
            tooth_h,
            left_shoulder,
            right_shoulder,
            top_cap,
            key_colors[idx],
            BG_COLOR,
        )
        turns = int(RNG.integers(0, 4))
        rotations.append(turns)
        key_specs.append(
            {
                "tooth_w": int(tooth_w),
                "tooth_h": int(tooth_h),
                "left_shoulder": left_shoulder,
                "right_shoulder": right_shoulder,
                "canonical": canonical,
                "input_sprite": np.rot90(canonical, turns),
                "final_w": int(canonical.shape[1]),
                "color": int(key_colors[idx]),
                "tooth_color": int(tooth_colors[idx]),
            }
        )

    if all(turn == 0 for turn in rotations):
        idx = int(RNG.integers(pair_count))
        key_specs[idx]["input_sprite"] = np.rot90(key_specs[idx]["canonical"], 1)
        rotations[idx] = 1

    height, width, item_starts, distractor_start = _layout_ground_items(key_specs, distractor_shape)
    grid, tooth_meta, distractor_meta = _paint_ground_scene(
        height,
        width,
        ground_color,
        key_specs,
        tooth_colors,
        distractor_shape,
        distractor_color,
        item_starts,
        distractor_start,
    )

    for idx, spec in enumerate(key_specs):
        spec["final_row"] = int(height - 1 - spec["canonical"].shape[0])
        spec["final_col"] = int(item_starts[idx])
        spec["tooth_start"] = int(tooth_meta[idx]["start"])
    max_tooth_h = max([shape[1] for shape in matched_shapes] + [distractor_shape[1]])
    floating_placements = _place_floating_keys(grid, key_specs, max_tooth_h)
    expected = _expected_output(grid, key_specs, floating_placements)

    meta = {
        "pair_count": pair_count,
        "matched_shapes": matched_shapes,
        "distractor_shape": distractor_shape,
        "ground_color": int(ground_color),
        "key_specs": key_specs,
        "rotations": rotations,
        "tooth_meta": tooth_meta,
        "distractor_meta": distractor_meta,
    }
    return grid, expected, meta


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(grid, expected):
        return False
    if int(max(grid.shape)) > 30 or int(max(expected.shape)) > 30:
        return False

    bg, _ground_color, teeth, keys = _scene_parts(grid)
    pair_count = int(meta["pair_count"])
    if len(keys) != pair_count or len(teeth) != pair_count + 1:
        return False

    tooth_shapes = []
    for tooth in teeth:
        tooth_h, tooth_w = _bbox_hw(tooth["bbox"])
        tooth_shapes.append((int(tooth_w), int(tooth_h)))
    if len(set(tooth_shapes)) != len(tooth_shapes):
        return False

    notch_shapes = []
    for key in keys:
        sprite = _component_sprite(grid, key)
        try:
            _, notch, turns = _orient_key(sprite, bg)
        except Exception:
            return False
        notch_shapes.append((int(notch["width"]), int(notch["depth"])))
        if turns not in {0, 1, 2, 3}:
            return False

    if len(set(notch_shapes)) != len(notch_shapes):
        return False
    if sum(shape in tooth_shapes for shape in notch_shapes) != pair_count:
        return False

    distractors = [shape for shape in tooth_shapes if shape not in set(notch_shapes)]
    if len(distractors) != 1:
        return False
    dist_w, dist_h = distractors[0]
    if not any(h == dist_h and w != dist_w for w, h in notch_shapes) and not any(
        w == dist_w and h != dist_h for w, h in notch_shapes
    ):
        return False

    key_specs = meta["key_specs"]
    if not any(int(spec["color"]) == int(spec["tooth_color"]) for spec in key_specs):
        return False
    if not any(int(spec["color"]) != int(spec["tooth_color"]) for spec in key_specs):
        return False
    if not any(int(turn) != 0 for turn in meta["rotations"]):
        return False
    if not any(
        int(spec["color"]) == int(meta["distractor_meta"]["color"])
        or any(
            int(spec["color"]) == int(other["tooth_color"])
            for other in key_specs
            if other is not spec
        )
        for spec in key_specs
    ):
        return False

    if int(np.count_nonzero(grid == BG_COLOR)) <= int(np.count_nonzero(grid != BG_COLOR)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected, meta = _build_case()
        except RuntimeError:
            continue
        if _validate_case(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid key-lock ARC input")
