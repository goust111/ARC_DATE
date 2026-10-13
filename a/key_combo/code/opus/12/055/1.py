from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
DIVIDER_COLOR = 5
SPRITE_COLORS = (2, 3, 1)
GUIDE_COLORS = (4, 7, 9)
GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _nontrivial_rotational_shape(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 1))
        or np.array_equal(mask, np.rot90(mask, 2))
        or np.array_equal(mask, np.rot90(mask, 3))
    )


def _shape_catalog() -> list[np.ndarray]:
    catalog: list[np.ndarray] = []
    for height in range(2, 5):
        for width in range(2, 5):
            area = height * width
            for pixels in range(4, min(7, area) + 1):
                for combo in itertools.combinations(range(area), pixels):
                    mask = np.zeros((height, width), dtype=bool)
                    mask.ravel()[list(combo)] = True
                    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    if not _is_connected4(mask):
                        continue
                    if not _nontrivial_rotational_shape(mask):
                        continue
                    catalog.append(mask)
    return catalog


SHAPE_CATALOG = _shape_catalog()


def _one_component(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    assert _is_connected4(mask)
    return coords


def _sprite_array(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _perimeter_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    cells.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    cells.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return cells


def _guide_pixels_for_box(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    top = [(r0, c) for c in range(c0 + 1, c1)]
    bottom = [(r1, c) for c in range(c0 + 1, c1)]
    left = [(r, c0) for r in range(r0 + 1, r1)]
    right = [(r, c1) for r in range(r0 + 1, r1)]
    chosen = {
        random.choice(top),
        random.choice(bottom),
        random.choice(left),
        random.choice(right),
    }
    extras = [cell for cell in _perimeter_cells(box) if cell not in chosen]
    extra_count = random.randint(0, 2)
    if extra_count:
        chosen.update(random.sample(extras, extra_count))
    return sorted(chosen)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    divider_cols = [c for c in range(width) if np.all(g[:, c] == DIVIDER_COLOR)]
    assert len(divider_cols) == 1
    divider = divider_cols[0]
    assert 0 < divider < width - 1

    left = g[:, :divider]
    right = g[:, divider + 1 :]
    left_colors = sorted(int(x) for x in np.unique(left) if int(x) != BG)
    right_colors = sorted(int(x) for x in np.unique(right) if int(x) != BG)
    assert len(left_colors) == 3
    assert len(right_colors) == 3

    sprites = []
    for color in left_colors:
        mask = left == color
        coords = _one_component(mask)
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert 4 <= int(cropped.sum()) <= 7
        sprite = _sprite_array(cropped, color)
        sprites.append(
            {
                "color": color,
                "bbox": bbox,
                "row_sum": r0 + r1,
                "col_sum": c0 + c1,
                "sprite": sprite,
            }
        )
    sprites.sort(key=lambda item: item["bbox"][0])

    guides = []
    for color in right_colors:
        coords = np.argwhere(right == color)
        assert 4 <= len(coords) <= 6
        bbox_local = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox_local
        for rr, cc in coords:
            assert rr in (r0, r1) or cc in (c0, c1)
        guides.append(
            {
                "color": color,
                "coords_global": [(int(r), int(c + divider + 1)) for r, c in coords],
                "bbox_local": bbox_local,
                "bbox_global": (r0, c0 + divider + 1, r1, c1 + divider + 1),
                "row_sum": r0 + r1,
                "col_sum": c0 + c1,
            }
        )
    guides.sort(key=lambda item: item["bbox_global"][0])

    sprite_row_sums = [item["row_sum"] for item in sprites]
    sprite_col_sums = [item["col_sum"] for item in sprites]
    matches = []
    used_sprites = set()
    for guide in guides:
        row_hits = [i for i, value in enumerate(sprite_row_sums) if value == guide["row_sum"]]
        col_hits = [i for i, value in enumerate(sprite_col_sums) if value == guide["col_sum"]]
        assert (len(row_hits), len(col_hits)) in ((1, 0), (0, 1))
        if row_hits:
            sprite_index = row_hits[0]
            mode = "row"
        else:
            sprite_index = col_hits[0]
            mode = "col"
        assert sprite_index not in used_sprites
        used_sprites.add(sprite_index)
        matches.append({"guide": guide, "sprite_index": sprite_index, "mode": mode})

    assert len(matches) == 3 and len(used_sprites) == 3
    return {"divider": divider, "sprites": sprites, "guides": guides, "matches": matches}


def _stamp_into_box(out: np.ndarray, box: tuple[int, int, int, int], sprite: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = box
    stamp = np.where(sprite != BG, int(color), BG)
    height, width = stamp.shape
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    assert height <= box_h and width <= box_w
    top = r0 + (box_h - height) // 2
    left = c0 + (box_w - width) // 2
    region = out[top : top + height, left : left + width]
    region[stamp != BG] = color


def _solve(grid: np.ndarray) -> np.ndarray:
    scene = _extract_scene(grid)
    out = np.asarray(grid, dtype=int).copy()
    for match in scene["matches"]:
        guide = match["guide"]
        sprite = scene["sprites"][match["sprite_index"]]["sprite"]
        if match["mode"] == "col":
            sprite = np.rot90(sprite, -1)
        for r, c in guide["coords_global"]:
            out[r, c] = BG
        _stamp_into_box(out, guide["bbox_global"], sprite, guide["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp each left-panel sprite into the uniquely aligned right-panel guide box, rotating only the column-aligned matches.

    Input structure:
    - A black background is split by one full-height grey divider column into a left sprite panel and a right guide panel.
    - The left panel contains three distinct monochrome 4-connected sprites, while the right panel contains three sparse guide outlines of distinct colors.
    - Each guide outline is interpreted by its tight bounding box, and alignment is measured in the panel coordinate system induced by the divider.

    Rule:
    - Match every guide to the unique sprite whose bbox center shares either the same row-center or the same panel-local column-center, but not both.
    - If the match is row-aligned, center the sprite in the guide bbox unchanged; if it is column-aligned, center its 90-degree clockwise rotation instead.
    - Recolor the stamped sprite to the guide color, erase the original guide pixels, and leave the divider plus the left-panel sprites unchanged.

    Key insight:
    - The same alignment test determines both which sprite belongs to a guide and whether that sprite must be rotated before stamping.
    """
    return _solve(grid)


def _choose_sprites() -> list[np.ndarray]:
    shapes = random.sample(SHAPE_CATALOG, 3)
    return [shape.copy() for shape in shapes]


def _place_sprites(height: int, left_w: int, shapes: list[np.ndarray], colors: tuple[int, ...]) -> list[dict] | None:
    dims = [shape.shape for shape in shapes]
    min_total = sum(h for h, _ in dims) + 4
    if min_total > height:
        return None

    gap1 = 2 + random.randint(0, 2)
    gap2 = 2 + random.randint(0, 2)
    used = sum(h for h, _ in dims) + gap1 + gap2
    if used > height:
        return None
    slack = height - used
    top_margin = random.randint(0, slack)
    bottom_margin = slack - top_margin

    rows = []
    r = top_margin
    for i, (shape, color) in enumerate(zip(shapes, colors)):
        rows.append((r, shape, color))
        r += shape.shape[0]
        if i == 0:
            r += gap1
        elif i == 1:
            r += gap2
    assert r + bottom_margin == height

    col_positions = []
    for shape in shapes:
        width = shape.shape[1]
        choices = list(range(1, left_w - width))
        if not choices:
            return None
        col_positions.append(choices)

    for _ in range(120):
        cols = [random.choice(choices) for choices in col_positions]
        col_sums = [2 * c + shape.shape[1] - 1 for c, shape in zip(cols, shapes)]
        if len(set(col_sums)) == 3:
            sprites = []
            for (r0, shape, color), c0 in zip(rows, cols):
                h, w = shape.shape
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                sprites.append(
                    {
                        "color": int(color),
                        "mask": shape,
                        "bbox": bbox,
                        "row_sum": bbox[0] + bbox[2],
                        "col_sum": bbox[1] + bbox[3],
                    }
                )
            return sprites
    return None


def _guide_candidate_boxes(
    spec: dict,
    sprite: dict,
    height: int,
    right_w: int,
    row_sums: list[int],
    col_sums: list[int],
    min_dim: int,
    max_dim: int,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    heights = list(range(min_dim, max_dim + 1))
    widths = list(range(min_dim, max_dim + 1))
    if spec["mode"] == "row":
        for box_h in heights:
            if (sprite["row_sum"] - (box_h - 1)) % 2 != 0:
                continue
            r0 = (sprite["row_sum"] - box_h + 1) // 2
            if not (0 <= r0 <= height - box_h):
                continue
            for box_w in widths:
                for c0 in range(0, right_w - box_w + 1):
                    col_sum = 2 * c0 + box_w - 1
                    if col_sum in col_sums:
                        continue
                    boxes.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1))
    else:
        for box_w in widths:
            if (sprite["col_sum"] - (box_w - 1)) % 2 != 0:
                continue
            c0 = (sprite["col_sum"] - box_w + 1) // 2
            if not (0 <= c0 <= right_w - box_w):
                continue
            for box_h in heights:
                for r0 in range(0, height - box_h + 1):
                    row_sum = 2 * r0 + box_h - 1
                    if row_sum in row_sums:
                        continue
                    boxes.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1))
    random.shuffle(boxes)
    return boxes


def _select_guide_boxes(
    specs: list[dict],
    sprites: list[dict],
    height: int,
    right_w: int,
    min_dim: int,
    max_dim: int,
) -> list[dict] | None:
    row_sums = [sprite["row_sum"] for sprite in sprites]
    col_sums = [sprite["col_sum"] for sprite in sprites]
    candidates = []
    for spec in specs:
        sprite = sprites[spec["sprite_index"]]
        boxes = _guide_candidate_boxes(spec, sprite, height, right_w, row_sums, col_sums, min_dim, max_dim)
        if not boxes:
            return None
        candidates.append(boxes)

    order = sorted(range(len(specs)), key=lambda idx: len(candidates[idx]))
    placed: dict[int, tuple[int, int, int, int]] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for box in candidates[idx]:
            if any(_boxes_too_close(box, placed[other]) for other in placed):
                continue
            placed[idx] = box
            if backtrack(pos + 1):
                return True
            del placed[idx]
        return False

    if not backtrack(0):
        return None

    result = []
    for idx, spec in enumerate(specs):
        box = placed[idx]
        result.append({**spec, "bbox_local": box})
    return result


def _render_input(height: int, left_w: int, right_w: int, sprites: list[dict], guides: list[dict]) -> np.ndarray:
    width = left_w + 1 + right_w
    grid = np.zeros((height, width), dtype=int)
    grid[:, left_w] = DIVIDER_COLOR

    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = _sprite_array(sprite["mask"], sprite["color"])

    for guide in guides:
        r0, c0, r1, c1 = guide["bbox_local"]
        global_box = (r0, c0 + left_w + 1, r1, c1 + left_w + 1)
        for r, c in _guide_pixels_for_box(global_box):
            grid[r, c] = guide["color"]

    return grid


def _expected_from_specs(grid: np.ndarray, guides: list[dict], sprites: list[dict], divider: int) -> np.ndarray:
    out = grid.copy()
    for guide in guides:
        r0, c0, r1, c1 = guide["bbox_local"]
        global_box = (r0, c0 + divider + 1, r1, c1 + divider + 1)
        color = guide["color"]
        coords = np.argwhere(out[:, divider + 1 :] == color)
        for rr, cc in coords:
            out[int(rr), int(cc) + divider + 1] = BG
        sprite = _sprite_array(sprites[guide["sprite_index"]]["mask"], sprites[guide["sprite_index"]]["color"])
        if guide["mode"] == "col":
            sprite = np.rot90(sprite, -1)
        _stamp_into_box(out, global_box, sprite, color)
    return out


def _validate_example(grid: np.ndarray, intended_guides: list[dict], sprites: list[dict], left_w: int) -> bool:
    try:
        scene = _extract_scene(grid)
    except AssertionError:
        return False

    mode_by_color = {guide["color"]: guide["mode"] for guide in intended_guides}
    sprite_by_color = {
        guide["color"]: sprites[guide["sprite_index"]]["color"] for guide in intended_guides
    }

    if {match["guide"]["color"] for match in scene["matches"]} != set(GUIDE_COLORS):
        return False
    for match in scene["matches"]:
        color = match["guide"]["color"]
        if match["mode"] != mode_by_color[color]:
            return False
        if scene["sprites"][match["sprite_index"]]["color"] != sprite_by_color[color]:
            return False

    modes = [match["mode"] for match in scene["matches"]]
    if not ("row" in modes and "col" in modes):
        return False

    any_half_offset = False
    for match in scene["matches"]:
        guide = match["guide"]
        sprite = scene["sprites"][match["sprite_index"]]["sprite"]
        if match["mode"] == "col":
            sprite = np.rot90(sprite, -1)
        box_h = guide["bbox_global"][2] - guide["bbox_global"][0] + 1
        box_w = guide["bbox_global"][3] - guide["bbox_global"][1] + 1
        spr_h, spr_w = sprite.shape
        if spr_h > box_h or spr_w > box_w:
            return False
        if (box_h - spr_h) % 2 == 1 or (box_w - spr_w) % 2 == 1:
            any_half_offset = True
    if not any_half_offset:
        return False

    solved = _solve(grid)
    never_rotate = grid.copy()
    always_rotate = grid.copy()
    for match in scene["matches"]:
        guide = match["guide"]
        sprite = scene["sprites"][match["sprite_index"]]["sprite"]
        for r, c in guide["coords_global"]:
            never_rotate[r, c] = BG
            always_rotate[r, c] = BG
        _stamp_into_box(never_rotate, guide["bbox_global"], sprite, guide["color"])
        _stamp_into_box(always_rotate, guide["bbox_global"], np.rot90(sprite, -1), guide["color"])
    if np.array_equal(never_rotate, solved) or np.array_equal(always_rotate, solved):
        return False

    sprite_order = {
        sprite["color"]: rank for rank, sprite in enumerate(sorted(sprites, key=lambda item: item["bbox"][0]))
    }
    guide_order = {
        guide["color"]: rank for rank, guide in enumerate(sorted(intended_guides, key=lambda item: item["bbox_local"][0]))
    }
    if all(guide_order[color] == sprite_order[sprite_by_color[color]] for color in GUIDE_COLORS):
        return False

    divider = left_w
    expected = _expected_from_specs(grid, intended_guides, sprites, divider)
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 19)
    left_w = random.randint(8, 10)
    right_w = random.randint(11, 14)
    if left_w + 1 + right_w > 30:
        return None

    sprite_colors = tuple(random.sample(list(SPRITE_COLORS), len(SPRITE_COLORS)))
    sprites = _place_sprites(height, left_w, _choose_sprites(), sprite_colors)
    if sprites is None:
        return None

    max_dim = max(max(sprite["mask"].shape) for sprite in sprites)
    min_dim = max(4, max_dim)
    max_dim = min(7, min_dim + 2)
    if min_dim > max_dim:
        return None

    col_count = random.choice((1, 2))
    col_indices = set(random.sample(range(3), col_count))
    pairings = []
    for sprite_index in range(3):
        pairings.append({"sprite_index": sprite_index, "mode": "col" if sprite_index in col_indices else "row"})
    random.shuffle(pairings)
    intended_guides = []
    for color, pairing in zip(GUIDE_COLORS, pairings):
        intended_guides.append({**pairing, "color": color})

    guides = _select_guide_boxes(intended_guides, sprites, height, right_w, min_dim, max_dim)
    if guides is None:
        return None

    grid = _render_input(height, left_w, right_w, sprites, guides)
    if not _validate_example(grid, guides, sprites, left_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid axis-aligned dual-stamp puzzle")
