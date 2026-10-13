from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


MAX_GENERATION_ATTEMPTS = 500
OTHER_SPRITE_COLORS = [Color.BLUE, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
DIR_BY_SIDE = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _component_colors(grid: np.ndarray, comp: dict) -> set[int]:
    return {int(grid[int(r), int(c)]) for r, c in comp["coords"]}


def _side_for_local(local_rc: tuple[int, int], h: int, w: int) -> str:
    r, c = map(int, local_rc)
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    side_count = sum([on_top, on_bottom, on_left, on_right])
    assert side_count == 1, "indicator must lie on exactly one non-corner side"
    if on_top:
        return "up"
    if on_bottom:
        return "down"
    if on_left:
        return "left"
    return "right"


def _extract_problem(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert comps, "expected at least one foreground component"

    sprite_candidates: list[dict] = []
    placeholders: list[dict] = []
    for comp in comps:
        colors = _component_colors(g, comp)
        if colors == {Color.YELLOW}:
            placeholders.append(comp)
        elif len(colors) >= 2:
            sprite_candidates.append(comp)
        else:
            raise AssertionError("unexpected extra non-yellow monochrome object")

    assert len(sprite_candidates) == 1, "expected exactly one multicolor sprite source"
    sprite_comp = sprite_candidates[0]
    sprite_patch = crop_bbox(g, sprite_comp["bbox"])
    sprite_mask = sprite_patch != bg
    h, w = sprite_patch.shape
    assert 3 <= h <= 5 and 3 <= w <= 5

    red_local = [tuple(map(int, rc)) for rc in np.argwhere(sprite_patch == Color.RED)]
    assert 1 <= len(red_local) <= 2, "expected one or two red indicator pixels"
    sides = {_side_for_local(rc, h, w) for rc in red_local}
    assert len(sides) == 1, "all indicators must agree on one side"
    indicator_side = next(iter(sides))

    matched: list[dict] = []
    for comp in placeholders:
        patch = crop_bbox(g, comp["bbox"])
        if patch.shape == sprite_patch.shape and np.array_equal(patch == Color.YELLOW, sprite_mask):
            matched.append(comp)
        else:
            raise AssertionError("every yellow component must match the sprite mask")

    assert 2 <= len(matched) <= 3, "expected two or three placeholder objects"
    assert len(comps) == 1 + len(matched), "unexpected extra foreground components"

    return {
        "grid": g,
        "bg": int(bg),
        "sprite_comp": sprite_comp,
        "sprite_patch": sprite_patch,
        "sprite_mask": sprite_mask,
        "red_local": red_local,
        "indicator_side": indicator_side,
        "placeholders": matched,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a multicolor sprite onto matching yellow masks, erase the source, then extend each copied red indicator to the canvas edge.
    
    Input structure:
    - The background is black.
    - One connected multicolor sprite appears once; it contains non-red body colors plus one or two red edge indicators.
    - Two or three disconnected yellow placeholder objects have the same cropped occupancy mask as the sprite.
    - Each red indicator lies on exactly one non-corner side of the sprite bbox, so it encodes one global extension direction.
    
    Rule:
    - Find the unique multicolor sprite and every yellow placeholder whose normalized mask matches it.
    - Replace each placeholder by stamping the full sprite colors into that bbox position.
    - Erase the original sprite entirely.
    - For every stamped copy, extend red from each copied red indicator outward in its indicated direction until the next step would leave the grid or hit another stamped copy pixel.
    - Compute all line stops against the post-stamp, pre-line canvas, so red lines do not block one another.
    
    Key insight:
    - The indicator direction is local to the sprite template, but the line length is determined only after copying because each stamped instance puts the same red pixel at a different absolute coordinate.
    """
    info = _extract_problem(np.asarray(grid, dtype=int))
    g = info["grid"]
    bg = info["bg"]
    sprite_patch = info["sprite_patch"]
    red_local = info["red_local"]
    side = info["indicator_side"]
    sprite_comp = info["sprite_comp"]
    placeholders = info["placeholders"]

    out = g.copy()
    for comp in placeholders:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        paste(out, sprite_patch, r0, c0, transparent=bg)

    for r, c in sprite_comp["coords"]:
        out[int(r), int(c)] = bg

    base = out.copy()
    dr, dc = DIR_BY_SIDE[side]
    H, W = out.shape
    for comp in placeholders:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        for lr, lc in red_local:
            rr = r0 + lr
            cc = c0 + lc
            nr = rr + dr
            nc = cc + dc
            while 0 <= nr < H and 0 <= nc < W:
                if base[nr, nc] != bg:
                    break
                out[nr, nc] = Color.RED
                nr += dr
                nc += dc
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_irregular_mask(h: int, w: int, locked: set[tuple[int, int]]) -> np.ndarray:
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in locked]
    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        remove_count = random.randint(1, min(3, len(candidates)))
        for rc in random.sample(candidates, remove_count):
            mask[rc] = False
        if mask.sum() < max(5, len(locked) + 2):
            continue
        if not all(mask[r, :].any() for r in range(h)):
            continue
        if not all(mask[:, c].any() for c in range(w)):
            continue
        if _mask_connected(mask):
            return mask

    fallback = np.ones((h, w), dtype=bool)
    for rc in [(h // 2, w // 2), (h - 1, w // 2), (h // 2, w - 1), (1, 1)]:
        if rc not in locked and 0 <= rc[0] < h and 0 <= rc[1] < w:
            fallback[rc] = False
            if _mask_connected(fallback):
                return fallback
            fallback[rc] = True
    raise RuntimeError("failed to build connected irregular mask")


def _choose_indicator_positions(h: int, w: int, side: str, count: int) -> list[tuple[int, int]]:
    if side in {"up", "down"}:
        inner = [(0 if side == "up" else h - 1, c) for c in range(1, w - 1)]
    else:
        inner = [(r, 0 if side == "left" else w - 1) for r in range(1, h - 1)]
    assert len(inner) >= count
    return sorted(random.sample(inner, count))


def _build_sprite(mask: np.ndarray, indicators: list[tuple[int, int]]) -> np.ndarray:
    sprite = np.full(mask.shape, Color.BLACK, dtype=int)
    non_red_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask) if tuple(map(int, rc)) not in set(indicators)]
    n_other = min(random.choice([2, 2, 3]), len(non_red_cells))
    palette = random.sample(OTHER_SPRITE_COLORS, n_other)
    random.shuffle(non_red_cells)
    for idx, rc in enumerate(non_red_cells):
        if idx < len(palette):
            color = palette[idx]
        else:
            color = random.choice(palette)
        sprite[rc] = color
    for rc in indicators:
        sprite[rc] = Color.RED
    return sprite


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _in_box(box: tuple[int, int, int, int], rc: tuple[int, int]) -> bool:
    r, c = rc
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _path_cells(
    top_left: tuple[int, int],
    local_rc: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    H, W = shape
    rr = top_left[0] + local_rc[0] + direction[0]
    cc = top_left[1] + local_rc[1] + direction[1]
    cells: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        cells.append((rr, cc))
        rr += direction[0]
        cc += direction[1]
    return cells


def _validate_layout(
    shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    indicators: list[tuple[int, int]],
    direction: tuple[int, int],
    copy_tops: list[tuple[int, int]],
    original_top: tuple[int, int],
) -> bool:
    h, w = sprite_shape
    all_boxes = [(r, c, r + h - 1, c + w - 1) for r, c in [*copy_tops, original_top]]
    for i, box_a in enumerate(all_boxes):
        for box_b in all_boxes[i + 1 :]:
            if _boxes_overlap(box_a, box_b, padding=1):
                return False

    line_lengths: list[int] = []
    occupied_paths: set[tuple[int, int]] = set()
    copy_boxes = all_boxes[:-1]
    original_box = all_boxes[-1]

    for idx, top_left in enumerate(copy_tops):
        sample_len = None
        own_box = copy_boxes[idx]
        for local_rc in indicators:
            path = _path_cells(top_left, local_rc, direction, shape)
            if not path:
                return False
            for cell in path:
                if cell in occupied_paths:
                    return False
                if _in_box(original_box, cell):
                    return False
                for j, other_box in enumerate(copy_boxes):
                    if j != idx and _in_box(other_box, cell):
                        return False
            occupied_paths.update(path)
            if sample_len is None:
                sample_len = len(path)
        assert sample_len is not None
        line_lengths.append(sample_len)

    if len(set(line_lengths)) < 2:
        return False

    for local_rc in indicators:
        original_path = _path_cells(original_top, local_rc, direction, shape)
        if not original_path:
            return False
        for cell in original_path:
            for box in copy_boxes:
                if _in_box(box, cell):
                    return False
    return True


def _random_top_left(shape: tuple[int, int], sprite_shape: tuple[int, int]) -> tuple[int, int]:
    H, W = shape
    h, w = sprite_shape
    return random.randint(0, H - h), random.randint(0, W - w)


def _generate_layout(
    shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    indicators: list[tuple[int, int]],
    direction: tuple[int, int],
    copy_count: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(400):
        copy_tops: list[tuple[int, int]] = []
        for _copy in range(copy_count):
            placed = False
            for _ in range(250):
                top_left = _random_top_left(shape, sprite_shape)
                box = (
                    top_left[0],
                    top_left[1],
                    top_left[0] + sprite_shape[0] - 1,
                    top_left[1] + sprite_shape[1] - 1,
                )
                if any(
                    _boxes_overlap(
                        box,
                        (r, c, r + sprite_shape[0] - 1, c + sprite_shape[1] - 1),
                        padding=1,
                    )
                    for r, c in copy_tops
                ):
                    continue
                copy_tops.append(top_left)
                placed = True
                break
            if not placed:
                break
        if len(copy_tops) != copy_count:
            continue

        for _ in range(250):
            original_top = _random_top_left(shape, sprite_shape)
            if _validate_layout(shape, sprite_shape, indicators, direction, copy_tops, original_top):
                return copy_tops, original_top
    return None


def _validate_generated(
    grid: np.ndarray,
    sprite: np.ndarray,
    mask: np.ndarray,
    copy_tops: list[tuple[int, int]],
    original_top: tuple[int, int],
    indicators: list[tuple[int, int]],
    direction: tuple[int, int],
) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    H, W = grid.shape
    h, w = sprite.shape

    or0, oc0 = original_top
    original_crop = out[or0 : or0 + h, oc0 : oc0 + w]
    if np.any(original_crop[mask] != Color.BLACK):
        return False

    copy_lengths: list[int] = []
    for r0, c0 in copy_tops:
        crop = out[r0 : r0 + h, c0 : c0 + w]
        if not np.array_equal(crop, sprite):
            return False

        sample_len = None
        for lr, lc in indicators:
            rr = r0 + lr + direction[0]
            cc = c0 + lc + direction[1]
            length = 0
            while 0 <= rr < H and 0 <= cc < W and out[rr, cc] == Color.RED:
                length += 1
                rr += direction[0]
                cc += direction[1]
            if length == 0:
                return False
            if 0 <= rr < H and 0 <= cc < W:
                return False
            if sample_len is None:
                sample_len = length
        assert sample_len is not None
        copy_lengths.append(sample_len)

    if len(set(copy_lengths)) < 2:
        return False

    comps = find_components(out, bg=Color.BLACK, connectivity=4, monochromatic=False)
    if len(comps) != len(copy_tops):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 24)
    W = random.randint(18, 24)
    side = random.choice(["up", "down", "left", "right"])
    indicator_count = random.choice([1, 2])

    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        side_capacity = (w - 2) if side in {"up", "down"} else (h - 2)
        if side_capacity >= indicator_count:
            break
    else:
        return None

    indicators = _choose_indicator_positions(h, w, side, indicator_count)
    mask = _build_irregular_mask(h, w, set(indicators))
    sprite = _build_sprite(mask, indicators)
    direction = DIR_BY_SIDE[side]
    copy_count = random.choice([2, 3])

    layout = _generate_layout((H, W), (h, w), indicators, direction, copy_count)
    if layout is None:
        return None
    copy_tops, original_top = layout

    grid = np.full((H, W), Color.BLACK, dtype=int)
    placeholder = np.where(mask, Color.YELLOW, Color.BLACK)

    paste(grid, sprite, original_top[0], original_top[1], transparent=Color.BLACK)
    for r0, c0 in copy_tops:
        paste(grid, placeholder, r0, c0, transparent=Color.BLACK)

    if not _validate_generated(grid, sprite, mask, copy_tops, original_top, indicators, direction):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
