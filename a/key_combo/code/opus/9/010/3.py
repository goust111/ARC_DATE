from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


SPRITE_TEMPLATE = np.array(
    [
        [Color.GREY, Color.BLACK, Color.BLACK],
        [Color.GREY, Color.GREY, Color.GREY],
        [Color.BLACK, Color.BLACK, Color.GREY],
    ],
    dtype=int,
)

HOLE_LAYOUTS = {
    1: [(1, 1)],
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2), (1, 4), (4, 1)],
}
CONTOUR_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
GEN_RETRIES = 400


def _hole_count_from_coords(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = bbox_coords(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    padded = np.pad(local, 1, constant_values=False)
    holes = 0
    for bg_coords in components_mask(~padded, connectivity=4):
        rr = bg_coords[:, 0]
        cc = bg_coords[:, 1]
        touches_outer = np.any(
            (rr == 0)
            | (rr == padded.shape[0] - 1)
            | (cc == 0)
            | (cc == padded.shape[1] - 1)
        )
        if not touches_outer:
            holes += 1
    return holes


def _extract_sprite_template(grid: np.ndarray) -> np.ndarray:
    sprite_components = components_mask(grid == Color.GREY, connectivity=4)
    assert len(sprite_components) == 1, "expected exactly one grey sprite"
    sprite_bbox = bbox_coords(sprite_components[0])
    sprite_crop = crop_bbox(grid, sprite_bbox)
    return np.where(sprite_crop == Color.GREY, Color.GREY, Color.BLACK).astype(int)


def _contours_with_holes(grid: np.ndarray) -> list[tuple[int, int]]:
    contours: list[tuple[int, int]] = []
    for comp in find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color in (Color.BLACK, Color.GREY):
            continue
        holes = _hole_count_from_coords(comp["coords"])
        assert holes >= 1
        contours.append((color, holes))
    assert len(contours) > 0, "expected non-grey contour objects"
    assert len({holes for _color, holes in contours}) == len(contours), "hole counts must be unique"
    return contours


def _render_output(template: np.ndarray, contours: list[tuple[int, int]]) -> np.ndarray:
    sorted_contours = sorted(contours, key=lambda item: item[1])
    sprite_h, sprite_w = template.shape
    max_holes = max(holes for _color, holes in sorted_contours)

    out = np.full((len(sorted_contours) * sprite_h, max_holes * sprite_w), Color.BLACK, dtype=int)
    for row_idx, (color, holes) in enumerate(sorted_contours):
        recolored = np.where(template == Color.GREY, int(color), Color.BLACK).astype(int)
        row_top = row_idx * sprite_h
        for k in range(holes):
            paste(out, recolored, row_top, k * sprite_w, transparent=Color.BLACK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the grey sprite in rows ordered by contour hole count, recoloring each row from its source contour.

    Input structure:
    - The grid has a black background.
    - One 4-connected grey object is a small sprite template whose bbox is at most 3x3.
    - Three or four other foreground objects are monochromatic 4-connected closed contours in distinct non-grey colors.
    - Each contour encloses a unique number of 4-connected black holes inside its own bbox.

    Rule:
    - Extract the grey sprite's minimal bbox, treating non-grey cells inside that bbox as black.
    - For each non-grey contour, count the number of enclosed black hole regions inside the contour's bbox.
    - Sort the contours by hole count in ascending order.
    - Build an output grid with one sprite-height row per contour and enough width for the largest hole count.
    - In the i-th output row, place as many left-aligned copies of the sprite as that contour's hole count, replacing grey by the contour color and leaving all other cells black.

    Key insight:
    - The contour shapes themselves are distractions: each contour's topological hole count simultaneously determines both the row order and the horizontal repetition count for the separate grey sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == Color.BLACK

    template = _extract_sprite_template(g)
    contours = _contours_with_holes(g)
    return _render_output(template, contours)


def _dihedral_variant(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    k = random.randrange(4)
    out = np.rot90(out, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _add_tabs(mask: np.ndarray) -> np.ndarray:
    pad = 2
    canvas = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=int)
    canvas[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask

    r0 = pad
    c0 = pad
    r1 = pad + mask.shape[0] - 1
    c1 = pad + mask.shape[1] - 1

    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides[: random.randint(1, 3)]:
        if side in ("top", "bottom"):
            width = random.randint(1, max(1, min(3, mask.shape[1])))
            start = random.randint(c0, c1 - width + 1)
            depth = random.randint(1, pad)
            if side == "top":
                canvas[r0 - depth : r0, start : start + width] = 1
            else:
                canvas[r1 + 1 : r1 + 1 + depth, start : start + width] = 1
        else:
            height = random.randint(1, max(1, min(3, mask.shape[0])))
            start = random.randint(r0, r1 - height + 1)
            depth = random.randint(1, pad)
            if side == "left":
                canvas[start : start + height, c0 - depth : c0] = 1
            else:
                canvas[start : start + height, c1 + 1 : c1 + 1 + depth] = 1

    return crop_bbox(canvas, bbox_coords(np.argwhere(canvas == 1)))


def _contour_mask_with_holes(hole_count: int) -> np.ndarray:
    rows, cols = random.choice(HOLE_LAYOUTS[int(hole_count)])
    row_heights = [random.randint(1, 3) for _ in range(rows)]
    col_widths = [random.randint(1, 3) for _ in range(cols)]

    h = sum(row_heights) + rows + 1
    w = sum(col_widths) + cols + 1
    mask = np.ones((h, w), dtype=int)

    rr = 1
    for cell_h in row_heights:
        cc = 1
        for cell_w in col_widths:
            mask[rr : rr + cell_h, cc : cc + cell_w] = 0
            cc += cell_w + 1
        rr += cell_h + 1

    mask = _add_tabs(mask)
    mask = _dihedral_variant(mask)

    coords = np.argwhere(mask == 1)
    assert _hole_count_from_coords(coords) == hole_count
    assert len(components_mask(mask == 1, connectivity=4)) == 1
    return mask


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask == 1, int(color), Color.BLACK).astype(int)


def _rank_order(values, reverse: bool = False) -> tuple[int, ...]:
    return tuple(
        idx
        for idx, _value in sorted(
            enumerate(values),
            key=lambda item: item[1],
            reverse=reverse,
        )
    )


def _expected_from_scene(template: np.ndarray, objects: list[dict]) -> np.ndarray:
    contours = [(int(obj["color"]), int(obj["holes"])) for obj in objects]
    return _render_output(template, contours)


def _scene_is_non_degenerate(objects: list[dict]) -> bool:
    hole_order = _rank_order([obj["holes"] for obj in objects])
    forbidden = {hole_order, hole_order[::-1]}

    top_order = _rank_order([(obj["bbox"][0], obj["bbox"][1]) for obj in objects])
    left_order = _rank_order([(obj["bbox"][1], obj["bbox"][0]) for obj in objects])
    area_order = _rank_order([obj["area"] for obj in objects])
    color_order = _rank_order([obj["color"] for obj in objects])

    if top_order in forbidden or left_order in forbidden:
        return False
    if area_order in forbidden or color_order in forbidden:
        return False

    holes_sorted = [obj["holes"] for obj in sorted(objects, key=lambda obj: obj["holes"])]
    widths_sorted = [obj["bbox"][3] - obj["bbox"][1] + 1 for obj in sorted(objects, key=lambda obj: obj["holes"])]
    heights_sorted = [obj["bbox"][2] - obj["bbox"][0] + 1 for obj in sorted(objects, key=lambda obj: obj["holes"])]
    if widths_sorted == sorted(widths_sorted) or widths_sorted == sorted(widths_sorted, reverse=True):
        return False
    if heights_sorted == sorted(heights_sorted) or heights_sorted == sorted(heights_sorted, reverse=True):
        return False

    return holes_sorted != list(range(1, len(holes_sorted) + 1)) or len(objects) == 4


def _place_scene(items: list[dict], grid_shape: tuple[int, int]) -> np.ndarray | None:
    grid = np.zeros(grid_shape, dtype=int)
    placed: list[dict] = []

    for item in sorted(items, key=lambda obj: obj["patch"].shape[0] * obj["patch"].shape[1], reverse=True):
        try:
            r0, c0 = random_free_location(grid, item["patch"], transparent=Color.BLACK, border=1, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(grid, item["patch"], r0, c0, transparent=Color.BLACK)
        bbox = (r0, c0, r0 + item["patch"].shape[0] - 1, c0 + item["patch"].shape[1] - 1)
        placed.append(
            {
                "color": item["color"],
                "holes": item["holes"],
                "bbox": bbox,
                "area": int(np.count_nonzero(item["patch"])),
            }
        )

    try:
        sprite_r, sprite_c = random_free_location(grid, SPRITE_TEMPLATE, transparent=Color.BLACK, border=1, padding=2, padding_connectivity=8)
    except ValueError:
        return None
    paste(grid, SPRITE_TEMPLATE, sprite_r, sprite_c, transparent=Color.BLACK)

    if not _scene_is_non_degenerate(placed):
        return None
    return grid


def _sample_items() -> list[dict]:
    object_count = random.choice((3, 4))
    hole_counts = random.sample((1, 2, 3, 4), object_count)
    colors = random.sample(CONTOUR_COLORS, object_count)

    items = []
    for holes, color in zip(hole_counts, colors):
        mask = _contour_mask_with_holes(holes)
        patch = _colorize(mask, color)
        items.append({"holes": int(holes), "color": int(color), "patch": patch})
    random.shuffle(items)
    return items


def _gen_attempt() -> np.ndarray | None:
    items = _sample_items()
    max_patch_h = max(item["patch"].shape[0] for item in items)
    max_patch_w = max(item["patch"].shape[1] for item in items)
    min_h = max(18, max_patch_h + 4)
    min_w = max(15, max_patch_w + 4)

    for _ in range(8):
        grid_h = random.randint(min_h, 28)
        grid_w = random.randint(min_w, 28)
        grid = _place_scene(items, (grid_h, grid_w))
        if grid is None:
            continue

        expected = _expected_from_scene(SPRITE_TEMPLATE, items)
        if max(expected.shape) > 30:
            continue
        if not np.array_equal(main(grid), expected):
            continue

        parsed_contours = sorted(_contours_with_holes(grid), key=lambda item: item[1])
        expected_contours = sorted([(item["color"], item["holes"]) for item in items], key=lambda item: item[1])
        if parsed_contours != expected_contours:
            continue

        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topological-hole repetition input")
