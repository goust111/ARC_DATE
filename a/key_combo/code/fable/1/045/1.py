from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import Color, find_components, get_bg, paste
except ModuleNotFoundError:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import Color, find_components, get_bg, paste


@dataclass(frozen=True)
class Sprite:
    sprite_id: int
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    coords: np.ndarray


@dataclass(frozen=True)
class SettledSprite:
    sprite_id: int
    original_color: int
    landed_color: int
    area: int
    bbox: tuple[int, int, int, int]
    landed_coords: np.ndarray
    shift: int
    touches_ground: bool
    support_id: int | None


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[Sprite]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one foreground sprite"

    colors = [int(comp["color"]) for comp in comps]
    areas = [int(comp["area"]) for comp in comps]
    assert len(colors) == len(set(colors)), "sprite colors must be unique"
    assert len(areas) == len(set(areas)), "sprite areas must be unique"

    sprites: list[Sprite] = []
    for sprite_id, comp in enumerate(comps):
        sprites.append(
            Sprite(
                sprite_id=int(sprite_id),
                color=int(comp["color"]),
                area=int(comp["area"]),
                bbox=tuple(map(int, comp["bbox"])),
                coords=np.asarray(comp["coords"], dtype=int),
            )
        )
    return bg, sprites


def _ordered_sprites(sprites: list[Sprite], mode: str) -> list[Sprite]:
    if mode == "area":
        key = lambda sprite: (sprite.area, sprite.bbox[0], sprite.bbox[1], sprite.color)
    elif mode == "top":
        key = lambda sprite: (sprite.bbox[0], sprite.bbox[1], sprite.area, sprite.color)
    elif mode == "bottom":
        key = lambda sprite: (sprite.bbox[2], sprite.bbox[1], sprite.area, sprite.color)
    else:
        raise ValueError(f"unknown ordering mode: {mode}")
    return sorted(sprites, key=key)


def _simulate(grid: np.ndarray, order_mode: str = "area") -> tuple[np.ndarray, list[SettledSprite]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg, sprites = _extract_sprites(g)
    order = _ordered_sprites(sprites, order_mode)

    out = np.full((H, W), bg, dtype=int)
    owner = np.full((H, W), -1, dtype=int)
    landed_color_by_id: dict[int, int] = {}
    records: list[SettledSprite] = []

    for sprite in order:
        coords = sprite.coords
        assert not np.any(owner[coords[:, 0], coords[:, 1]] != -1), "input sprites must start on empty cells"

        shift = 0
        while True:
            next_rows = coords[:, 0] + shift + 1
            next_cols = coords[:, 1]
            if int(next_rows.max()) >= H:
                break
            if np.any(owner[next_rows, next_cols] != -1):
                break
            shift += 1

        landed = coords.copy()
        landed[:, 0] += shift
        touches_ground = bool(np.any(landed[:, 0] == H - 1))

        support_id: int | None = None
        if touches_ground:
            landed_color = int(sprite.color)
        else:
            below_ids = {
                int(owner[int(r) + 1, int(c)])
                for r, c in landed
                if int(r) + 1 < H and int(owner[int(r) + 1, int(c)]) != -1
            }
            assert below_ids, "supported sprite must touch something below it"
            assert len(below_ids) == 1, "support relation must be unique"
            support_id = next(iter(below_ids))
            landed_color = int(landed_color_by_id[support_id])

        out[landed[:, 0], landed[:, 1]] = landed_color
        owner[landed[:, 0], landed[:, 1]] = int(sprite.sprite_id)
        landed_color_by_id[int(sprite.sprite_id)] = landed_color
        records.append(
            SettledSprite(
                sprite_id=int(sprite.sprite_id),
                original_color=int(sprite.color),
                landed_color=int(landed_color),
                area=int(sprite.area),
                bbox=tuple(map(int, sprite.bbox)),
                landed_coords=landed.copy(),
                shift=int(shift),
                touches_ground=touches_ground,
                support_id=support_id,
            )
        )

    return out, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the sprites in increasing-area order so they stack and inherit the support color.

    Input structure:
    - A black background contains several disjoint monochrome 4-connected sprites in the upper or middle part of the grid.
    - Each sprite has a unique color and a unique pixel area.
    - There are no other structures; the bottom border acts as the only ground.

    Rule:
    - Sort the sprites by area from smallest to largest.
    - Process them one by one: keep each sprite's shape and columns fixed, then let it fall straight down until the next step would hit the ground or an already settled sprite.
    - If the landed sprite touches the ground, keep its original color; otherwise recolor the whole sprite to the landed color of its unique supporter.
    - Clear the original locations and return only the settled arrangement.

    Key insight:
    - The hidden order is encoded by area, not by the sprites' initial heights, and color spreads upward through the support chain after each landing.
    """
    return _simulate(np.asarray(grid, dtype=int), order_mode="area")[0]


GEN_ATTEMPTS = 400
BG = Color.BLACK
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MIN_H = 14
MAX_H = 24
MIN_W = 14
MAX_W = 28
MIN_INTERVAL_GAP = 2
INPUT_ROW_CAP_MARGIN = 4


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    if lo > hi:
        raise ValueError(f"invalid randint bounds: {lo}..{hi}")
    if lo == hi:
        return int(lo)
    return int(rng.integers(lo, hi + 1))


def _split_slack(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[int(rng.integers(parts))] += 1
    return shares


def _layout_intervals(
    rng: np.random.Generator, widths: list[int], canvas_w: int, min_gap: int
) -> list[int]:
    base = sum(widths) + min_gap * max(0, len(widths) - 1)
    slack = canvas_w - base
    assert slack >= 0
    extras = _split_slack(rng, slack, len(widths) + 1)

    lefts: list[int] = []
    cur = extras[0]
    for i, width in enumerate(widths):
        lefts.append(int(cur))
        cur += width
        if i < len(widths) - 1:
            cur += min_gap + extras[i + 1]
    return lefts


def _gen_histogram_mask(
    rng: np.random.Generator, area: int, width: int, max_height: int
) -> np.ndarray | None:
    if area < width or area > width * max_height:
        return None

    for _ in range(80):
        heights = np.ones(width, dtype=int)
        extra = area - width
        for _ in range(extra):
            choices = np.flatnonzero(heights < max_height)
            if choices.size == 0:
                break
            heights[int(rng.choice(choices))] += 1
        if int(heights.sum()) != area:
            continue
        if width > 1 and area > width and len(set(map(int, heights))) == 1 and rng.random() < 0.8:
            continue

        h = int(heights.max())
        mask = np.zeros((h, width), dtype=bool)
        for c, column_h in enumerate(heights):
            mask[h - int(column_h) :, c] = True
        return mask
    return None


def _pick_unique_area(
    rng: np.random.Generator,
    used: set[int],
    width: int,
    min_area: int,
    max_area: int,
) -> int | None:
    lo = max(width, min_area)
    hi = max_area
    candidates = [area for area in range(lo, hi + 1) if area not in used]
    if not candidates:
        return None
    return int(rng.choice(candidates))


def _gen_main_stack(rng: np.random.Generator) -> list[dict] | None:
    w_bottom = _randint(rng, 4, 6)
    w_middle = _randint(rng, max(2, w_bottom - 2), w_bottom)
    w_top = _randint(rng, 2, min(w_middle, 4))

    used: set[int] = set()
    a_bottom = _pick_unique_area(rng, used, w_bottom, w_bottom, min(9, w_bottom * 3))
    if a_bottom is None:
        return None
    used.add(a_bottom)

    a_middle = _pick_unique_area(
        rng, used, w_middle, a_bottom + 2, min(13, w_middle * 4)
    )
    if a_middle is None:
        return None
    used.add(a_middle)

    a_top = _pick_unique_area(rng, used, w_top, a_middle + 2, min(18, w_top * 6))
    if a_top is None:
        return None

    masks = [
        _gen_histogram_mask(rng, a_bottom, w_bottom, 3),
        _gen_histogram_mask(rng, a_middle, w_middle, 4),
        _gen_histogram_mask(rng, a_top, w_top, 6),
    ]
    if any(mask is None for mask in masks):
        return None

    stack = []
    for level, (area, mask) in enumerate(zip((a_bottom, a_middle, a_top), masks), start=1):
        stack.append({"level": level, "area": int(area), "mask": mask})
    return stack


def _gen_extra_sprites(
    rng: np.random.Generator, count: int, used_areas: set[int]
) -> list[dict] | None:
    extras: list[dict] = []
    for _ in range(count):
        width = _randint(rng, 2, 5)
        max_height = _randint(rng, 2, 5)
        area = _pick_unique_area(
            rng,
            used_areas,
            width,
            width + 1,
            min(width * max_height, 18),
        )
        if area is None:
            return None
        mask = _gen_histogram_mask(rng, area, width, max_height)
        if mask is None:
            return None
        used_areas.add(int(area))
        extras.append({"area": int(area), "mask": mask})
    return extras


def _render_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _validate_generated(
    inp: np.ndarray, expected_objects: int, main_colors: tuple[int, int, int]
) -> bool:
    comps = find_components(inp, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != expected_objects:
        return False
    if len({int(comp["color"]) for comp in comps}) != expected_objects:
        return False
    if len({int(comp["area"]) for comp in comps}) != expected_objects:
        return False

    try:
        out, records = _simulate(inp, order_mode="area")
        wrong_top, _ = _simulate(inp, order_mode="top")
    except AssertionError:
        return False

    if np.array_equal(out, wrong_top):
        return False

    grounded = [record for record in records if record.touches_ground]
    supported = [record for record in records if not record.touches_ground]
    if not grounded or len(supported) < 2:
        return False

    by_color = {record.original_color: record for record in records}
    bottom_color, middle_color, top_color = map(int, main_colors)
    if by_color[middle_color].landed_color != bottom_color:
        return False
    if by_color[top_color].landed_color != bottom_color:
        return False
    if by_color[middle_color].support_id is None or by_color[top_color].support_id is None:
        return False
    if by_color[bottom_color].touches_ground is not True:
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    main_stack = _gen_main_stack(rng)
    if main_stack is None:
        return None

    used_areas = {int(obj["area"]) for obj in main_stack}
    extra_count = _randint(rng, 0, 2)
    extras = _gen_extra_sprites(rng, extra_count, used_areas)
    if extras is None:
        return None

    all_objects = [{"kind": "main", **obj} for obj in main_stack] + [
        {"kind": "extra", **obj} for obj in extras
    ]
    if len(all_objects) > len(PALETTE):
        return None

    colors = list(map(int, rng.choice(PALETTE, size=len(all_objects), replace=False)))
    rng.shuffle(colors)
    for obj, color in zip(all_objects, colors):
        obj["color"] = int(color)

    main_bottom, main_middle, main_top = [obj for obj in all_objects if obj["kind"] == "main"]
    extra_objects = [obj for obj in all_objects if obj["kind"] == "extra"]
    main_colors = (
        int(main_bottom["color"]),
        int(main_middle["color"]),
        int(main_top["color"]),
    )

    interval_items = [("main", int(main_bottom["mask"].shape[1]))]
    for idx, extra in enumerate(extra_objects):
        interval_items.append((f"extra_{idx}", int(extra["mask"].shape[1])))
    rng.shuffle(interval_items)

    min_width = sum(width for _name, width in interval_items) + MIN_INTERVAL_GAP * (
        len(interval_items) - 1
    )
    if min_width > MAX_W:
        return None
    grid_w_lo = max(MIN_W, min_width)
    grid_w_hi = min(MAX_W, grid_w_lo + 8)
    grid_w = _randint(rng, grid_w_lo, grid_w_hi)
    interval_lefts = _layout_intervals(
        rng, [width for _name, width in interval_items], grid_w, MIN_INTERVAL_GAP
    )
    interval_map = {
        name: int(left) for (name, _width), left in zip(interval_items, interval_lefts)
    }

    main_bottom["left"] = int(interval_map["main"])
    main_middle["left"] = int(
        main_bottom["left"]
        + _randint(rng, 0, int(main_bottom["mask"].shape[1] - main_middle["mask"].shape[1]))
    )
    main_top["left"] = int(
        main_middle["left"]
        + _randint(rng, 0, int(main_middle["mask"].shape[1] - main_top["mask"].shape[1]))
    )

    for idx, extra in enumerate(extra_objects):
        extra["left"] = int(interval_map[f"extra_{idx}"])

    gap1 = _randint(rng, 1, 2)
    gap2 = _randint(rng, 1, 2)
    main_input_order = [main_top, main_middle, main_bottom]
    main_block_height = (
        int(main_top["mask"].shape[0])
        + gap1
        + int(main_middle["mask"].shape[0])
        + gap2
        + int(main_bottom["mask"].shape[0])
    )

    max_sprite_h = max(int(obj["mask"].shape[0]) for obj in all_objects)
    min_height = max(MIN_H, main_block_height + INPUT_ROW_CAP_MARGIN, max_sprite_h + 3)
    if min_height > MAX_H:
        return None
    grid_h = _randint(rng, min_height, min(MAX_H, min_height + 6))

    max_start = min(grid_h // 2, grid_h - main_block_height - 2)
    if max_start < 0:
        return None
    start_row = _randint(rng, 0, max_start)

    row = start_row
    for obj in main_input_order:
        obj["top"] = int(row)
        row += int(obj["mask"].shape[0])
        if obj is main_top:
            row += gap1
        elif obj is main_middle:
            row += gap2

    row_cap = max(0, grid_h // 2)
    for extra in extra_objects:
        max_top = min(row_cap, grid_h - int(extra["mask"].shape[0]) - 2)
        if max_top < 0:
            return None
        extra["top"] = _randint(rng, 0, max_top)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for obj in all_objects:
        patch = _render_sprite(obj["mask"], int(obj["color"]))
        paste(grid, patch, int(obj["top"]), int(obj["left"]), transparent=BG)

    if not _validate_generated(grid, len(all_objects), main_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacking-sedimentation input")
