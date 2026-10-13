from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, crop_bbox, find_components, paste


FRAME_COLOR = Color.GREY
SPRITE_COLOR = Color.MAROON
BG_COLOR = Color.BLACK
DEPTH_TO_COLOR = {
    1: Color.TEAL,
    2: Color.YELLOW,
    3: Color.GREEN,
}

CANVAS_SIDE = 30
GEN_ATTEMPTS = 500
GROUP_PATTERNS = (
    (1, 2, 3),
    (1, 1, 2),
    (1, 1, 3),
    (1, 2, 2),
    (2, 2, 3),
    (1, 1, 2, 2),
    (1, 2, 3, 1),
    (1, 1, 2, 3),
)
OUTER_SIZE_RANGES = {
    1: ((7, 16), (7, 16)),
    2: ((10, 16), (10, 16)),
    3: ((13, 16), (13, 16)),
}


def _is_hollow_rect_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(component["area"]) != 2 * h + 2 * w - 4:
        return False

    mask = np.zeros((h, w), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _strictly_contains(outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _inside_interior(component: dict, frame_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = frame_bbox
    coords = component["coords"]
    return bool(
        np.all(coords[:, 0] > r0)
        and np.all(coords[:, 0] < r1)
        and np.all(coords[:, 1] > c0)
        and np.all(coords[:, 1] < c1)
    )


def _analyze_groups(grid: np.ndarray) -> list[dict]:
    frame_components = [
        comp
        for comp in find_components(
            grid,
            bg=BG_COLOR,
            connectivity=4,
            monochromatic=True,
            colors=[FRAME_COLOR],
        )
        if _is_hollow_rect_frame(comp)
    ]
    assert frame_components, "expected grey hollow rectangle frames"

    sprite_components = find_components(
        grid,
        bg=BG_COLOR,
        connectivity=4,
        monochromatic=True,
        colors=[SPRITE_COLOR],
    )
    assert sprite_components, "expected maroon sprites"

    parents: list[int | None] = [None] * len(frame_components)
    for child_index, child in enumerate(frame_components):
        candidates = []
        for parent_index, parent in enumerate(frame_components):
            if parent_index == child_index:
                continue
            if _strictly_contains(parent["bbox"], child["bbox"]):
                pr0, pc0, pr1, pc1 = parent["bbox"]
                candidates.append(((pr1 - pr0 + 1) * (pc1 - pc0 + 1), parent_index))
        if candidates:
            parents[child_index] = min(candidates)[1]

    roots = [idx for idx, parent in enumerate(parents) if parent is None]
    assert 3 <= len(roots) <= 4

    groups = []
    for root_index in roots:
        chain_indices = []
        for idx in range(len(frame_components)):
            cur = idx
            while cur is not None:
                if cur == root_index:
                    chain_indices.append(idx)
                    break
                cur = parents[cur]
        chain = [frame_components[idx] for idx in chain_indices]
        chain.sort(key=lambda comp: (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1))
        depth = len(chain)
        assert depth in DEPTH_TO_COLOR
        innermost = chain[0]
        outermost = chain[-1]

        matching_sprites = [comp for comp in sprite_components if _inside_interior(comp, innermost["bbox"])]
        assert len(matching_sprites) == 1, "every frame chain must contain exactly one sprite"
        sprite = matching_sprites[0]
        groups.append(
            {
                "depth": depth,
                "frames": chain,
                "innermost_bbox": innermost["bbox"],
                "outer_bbox": outermost["bbox"],
                "outer_area": (outermost["bbox"][2] - outermost["bbox"][0] + 1)
                * (outermost["bbox"][3] - outermost["bbox"][1] + 1),
                "sprite": sprite,
            }
        )

    used_sprite_ids = {id(group["sprite"]) for group in groups}
    assert len(used_sprite_ids) == len(sprite_components)
    return groups


def _recolored_sprite_crop(grid: np.ndarray, sprite: dict, depth: int) -> np.ndarray:
    patch = crop_bbox(grid, sprite["bbox"])
    out = np.zeros_like(patch)
    out[patch == int(sprite["color"])] = DEPTH_TO_COLOR[depth]
    return out


def _compose_output_from_groups(grid: np.ndarray, groups: list[dict]) -> np.ndarray:
    ordered = sorted(groups, key=lambda group: group["outer_area"])
    sprites = [_recolored_sprite_crop(grid, group["sprite"], group["depth"]) for group in ordered]
    out_h = max(sprite.shape[0] for sprite in sprites)
    out_w = sum(sprite.shape[1] for sprite in sprites) + len(sprites) - 1
    out = np.full((out_h, out_w), BG_COLOR, dtype=int)

    cursor = 0
    for sprite in sprites:
        rr = out_h - sprite.shape[0]
        paste(out, sprite, rr, cursor, transparent=BG_COLOR)
        cursor += sprite.shape[1] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each innermost sprite by nesting depth, then reorder the cropped sprites by outer-frame area.

    Input structure:
    - The background is black.
    - The foreground contains 3-4 disconnected groups of grey one-pixel hollow rectangular frames.
    - Each group is a strict nesting chain of 1-3 frames, and one maroon 4-connected sprite lies inside the innermost frame.

    Rule:
    - Detect all grey hollow rectangular frames and group them by strict containment under the same outermost frame.
    - The depth of a group is its number of nested frames; recolor its inner sprite by depth: 1 -> teal, 2 -> yellow, 3 -> green.
    - Crop each recolored sprite to its own bounding box, ignoring the grey frames.
    - Measure each group's outermost frame area and sort the cropped sprites by that area in ascending order.
    - Place the sorted crops left to right on the smallest black canvas, with one blank column between neighbors and bottoms aligned.

    Key insight:
    - The same nesting structure yields two independent signals: transitive frame depth determines the sprite color, while outer-frame area determines the final ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    groups = _analyze_groups(g)
    return _compose_output_from_groups(g, groups)


def _random_sprite_mask() -> np.ndarray:
    for _ in range(200):
        target_area = random.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if abs(nr) <= 2 and abs(nc) <= 2 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h > 3 or w > 3:
            continue
        mask = np.zeros((h, w), dtype=int)
        mask[coords[:, 0], coords[:, 1]] = SPRITE_COLOR
        return mask
    raise RuntimeError("failed to sample sprite mask")


def _shift_boxes(boxes: list[tuple[int, int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _split_total(total: int, parts: int, min_piece: int) -> list[int]:
    assert total >= parts * min_piece
    pieces = [min_piece] * parts
    for _ in range(total - parts * min_piece):
        pieces[random.randrange(parts)] += 1
    random.shuffle(pieces)
    return pieces


def _make_group_patch(depth: int) -> dict:
    sprite = _random_sprite_mask()
    sh, sw = sprite.shape
    outer_h = random.randint(*OUTER_SIZE_RANGES[depth][0])
    outer_w = random.randint(*OUTER_SIZE_RANGES[depth][1])
    min_inner_h = max(4, sh + 2)
    min_inner_w = max(4, sw + 2)
    inner_h_max = outer_h - 4 * (depth - 1)
    inner_w_max = outer_w - 4 * (depth - 1)
    assert inner_h_max >= min_inner_h
    assert inner_w_max >= min_inner_w

    inner_h = random.randint(min_inner_h, inner_h_max)
    inner_w = random.randint(min_inner_w, inner_w_max)
    cur_h = inner_h
    cur_w = inner_w
    sprite_r = 1 + random.randint(0, inner_h - 2 - sh)
    sprite_c = 1 + random.randint(0, inner_w - 2 - sw)
    frame_boxes = [(0, 0, cur_h - 1, cur_w - 1)]

    if depth > 1:
        row_sums = _split_total(outer_h - inner_h, depth - 1, 4)
        col_sums = _split_total(outer_w - inner_w, depth - 1, 4)
        for row_sum, col_sum in zip(row_sums, col_sums):
            top = random.randint(2, row_sum - 2)
            bottom = row_sum - top
            left = random.randint(2, col_sum - 2)
            right = col_sum - left
            frame_boxes = _shift_boxes(frame_boxes, top, left)
            sprite_r += top
            sprite_c += left
            cur_h += row_sum
            cur_w += col_sum
            frame_boxes.append((0, 0, cur_h - 1, cur_w - 1))

    patch = np.full((cur_h, cur_w), BG_COLOR, dtype=int)
    for r0, c0, r1, c1 in frame_boxes:
        patch[r0, c0 : c1 + 1] = FRAME_COLOR
        patch[r1, c0 : c1 + 1] = FRAME_COLOR
        patch[r0 : r1 + 1, c0] = FRAME_COLOR
        patch[r0 : r1 + 1, c1] = FRAME_COLOR
    paste(patch, sprite, sprite_r, sprite_c, transparent=BG_COLOR)

    outer_h = cur_h
    outer_w = cur_w
    return {
        "depth": depth,
        "patch": patch,
        "outer_area": outer_h * outer_w,
        "sprite_crop": sprite,
        "shape_signature": tuple(map(tuple, (sprite == SPRITE_COLOR).astype(int))),
    }


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_groups_on_canvas(groups: list[dict]) -> np.ndarray | None:
    canvas = np.full((CANVAS_SIDE, CANVAS_SIDE), BG_COLOR, dtype=int)
    placements: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(groups)), key=lambda idx: groups[idx]["patch"].size, reverse=True)

    for index in order:
        patch = groups[index]["patch"]
        ph, pw = patch.shape
        candidates = [
            (r, c)
            for r in range(1, CANVAS_SIDE - ph)
            for c in range(1, CANVAS_SIDE - pw)
            if all(_boxes_separated((r, c, r + ph - 1, c + pw - 1), box, gap=1) for box in placements)
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placements.append((r, c, r + ph - 1, c + pw - 1))
        paste(canvas, patch, r, c, transparent=BG_COLOR)

    coords = np.argwhere(canvas != BG_COLOR)
    r0, c0, r1, c1 = bbox_coords(coords)
    pad_top = random.randint(0, min(2, r0))
    pad_left = random.randint(0, min(2, c0))
    pad_bottom = random.randint(0, min(2, CANVAS_SIDE - 1 - r1))
    pad_right = random.randint(0, min(2, CANVAS_SIDE - 1 - c1))
    return canvas[r0 - pad_top : r1 + pad_bottom + 1, c0 - pad_left : c1 + pad_right + 1].copy()


def _expected_output(groups: list[dict]) -> np.ndarray:
    ordered = sorted(groups, key=lambda group: group["outer_area"])
    crops = []
    for group in ordered:
        sprite = np.where(group["sprite_crop"] == SPRITE_COLOR, DEPTH_TO_COLOR[group["depth"]], BG_COLOR)
        crops.append(sprite.astype(int))

    out_h = max(crop.shape[0] for crop in crops)
    out_w = sum(crop.shape[1] for crop in crops) + len(crops) - 1
    out = np.full((out_h, out_w), BG_COLOR, dtype=int)
    cursor = 0
    for crop in crops:
        paste(out, crop, out_h - crop.shape[0], cursor, transparent=BG_COLOR)
        cursor += crop.shape[1] + 1
    return out


def _depth_area_inversion(groups: list[dict]) -> bool:
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if (groups[i]["depth"] - groups[j]["depth"]) * (groups[i]["outer_area"] - groups[j]["outer_area"]) < 0:
                return True
    return False


def _semantic_checks(grid: np.ndarray, groups: list[dict]) -> None:
    assert 3 <= len(groups) <= 4
    depths = [group["depth"] for group in groups]
    areas = [group["outer_area"] for group in groups]
    assert len(set(depths)) >= 2
    assert len(set(areas)) == len(areas)
    assert set(depths).issubset(set(DEPTH_TO_COLOR))
    assert _depth_area_inversion(groups)
    assert len({(group["depth"], group["shape_signature"]) for group in groups}) >= 2

    analyzed = _analyze_groups(grid)
    expected_signature = sorted(
        (group["depth"], group["outer_area"], group["sprite_crop"].shape, int(np.count_nonzero(group["sprite_crop"])))
        for group in groups
    )
    actual_signature = sorted(
        (
            group["depth"],
            group["outer_area"],
            crop_bbox(grid, group["sprite"]["bbox"]).shape,
            int(group["sprite"]["area"]),
        )
        for group in analyzed
    )
    assert expected_signature == actual_signature

    expected_out = _expected_output(groups)
    actual_out = main(grid)
    assert np.array_equal(actual_out, expected_out)
    assert max(grid.shape) <= 30
    assert max(actual_out.shape) <= 30


def _generate_candidate() -> np.ndarray | None:
    depths = list(random.choice(GROUP_PATTERNS))
    random.shuffle(depths)
    groups = [_make_group_patch(depth) for depth in depths]
    if len({group["outer_area"] for group in groups}) != len(groups):
        return None
    if not _depth_area_inversion(groups):
        return None

    grid = _place_groups_on_canvas(groups)
    if grid is None:
        return None

    try:
        _semantic_checks(grid, groups)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
