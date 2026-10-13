from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
BARRIER = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
BOX_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )
    return components


def _find_barrier(grid: np.ndarray) -> tuple[dict, str]:
    h, w = grid.shape
    comps = _find_components(grid, {BARRIER})
    assert comps, "expected a grey barrier"
    barrier = max(comps, key=lambda comp: comp["area"])
    r0, c0, r1, c1 = barrier["bbox"]
    spans_width = c0 == 0 and c1 == w - 1
    spans_height = r0 == 0 and r1 == h - 1
    assert spans_width ^ spans_height, "barrier must span exactly one full axis"
    return barrier, ("horizontal" if spans_width else "vertical")


def _other_color(colors: tuple[int, int], color: int) -> int:
    return colors[1] if color == colors[0] else colors[0]


def _extract_sprites(grid: np.ndarray, orientation: str, barrier_bbox: tuple[int, int, int, int]) -> list[dict]:
    sprite_colors = {int(v) for v in np.unique(grid)} - {BG, BARRIER}
    assert len(sprite_colors) == 2, "expected exactly two sprite colors"
    sprites = _find_components(grid, sprite_colors)

    out: list[dict] = []
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        if orientation == "vertical":
            barrier_col = barrier_bbox[1]
            if c1 < barrier_col:
                side = "left"
            else:
                assert c0 > barrier_col, "sprites must stay on one side of the barrier"
                side = "right"
            anchor = r0
        else:
            barrier_row = barrier_bbox[0]
            if r1 < barrier_row:
                side = "top"
            else:
                assert r0 > barrier_row, "sprites must stay on one side of the barrier"
                side = "bottom"
            anchor = c0

        sprite = dict(sprite)
        sprite["side"] = side
        sprite["anchor"] = anchor
        sprite["size"] = sprite["mask"].shape
        out.append(sprite)

    side_order = {"left": 0, "right": 1, "top": 0, "bottom": 1}
    out.sort(key=lambda s: (side_order[str(s["side"])], int(s["anchor"]), s["bbox"][0], s["bbox"][1], int(s["color"])))
    return out


def _pair_sprites(sprites: list[dict], orientation: str, colors: tuple[int, int]) -> list[tuple[int, int]]:
    if orientation == "vertical":
        side0_name, side1_name = "left", "right"
    else:
        side0_name, side1_name = "top", "bottom"

    side0 = [(idx, sprite) for idx, sprite in enumerate(sprites) if sprite["side"] == side0_name]
    side1 = [(idx, sprite) for idx, sprite in enumerate(sprites) if sprite["side"] == side1_name]

    lookup0: dict[tuple[int, int, int, int], list[int]] = {}
    lookup1: dict[tuple[int, int, int, int], list[int]] = {}

    for idx, sprite in side0:
        sh, sw = sprite["size"]
        key = (int(sprite["anchor"]), int(sh), int(sw), int(sprite["color"]))
        lookup0.setdefault(key, []).append(idx)
    for idx, sprite in side1:
        sh, sw = sprite["size"]
        key = (int(sprite["anchor"]), int(sh), int(sw), int(sprite["color"]))
        lookup1.setdefault(key, []).append(idx)

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for idx0, sprite0 in side0:
        other = _other_color(colors, int(sprite0["color"]))
        sh, sw = sprite0["size"]
        key = (int(sprite0["anchor"]), int(sh), int(sw), other)
        candidates = lookup1.get(key, [])
        if len(candidates) != 1:
            continue
        idx1 = candidates[0]
        sprite1 = sprites[idx1]
        reverse_key = (
            int(sprite1["anchor"]),
            int(sprite1["size"][0]),
            int(sprite1["size"][1]),
            int(sprite0["color"]),
        )
        if len(lookup0.get(reverse_key, [])) != 1:
            continue
        if idx0 in used or idx1 in used:
            continue
        used.add(idx0)
        used.add(idx1)
        pairs.append((idx0, idx1))
    return pairs


def _draw_mask(out: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray, color: int) -> None:
    r0, c0, _r1, _c1 = bbox
    rows, cols = np.nonzero(mask)
    out[r0 + rows, c0 + cols] = int(color)


def _expected_from_analysis(grid: np.ndarray, sprites: list[dict], pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.full_like(grid, BG)
    out[grid == BARRIER] = BARRIER
    for idx0, idx1 in pairs:
        sprite0 = sprites[idx0]
        sprite1 = sprites[idx1]
        _draw_mask(out, sprite0["bbox"], sprite1["mask"], int(sprite0["color"]))
        _draw_mask(out, sprite1["bbox"], sprite0["mask"], int(sprite1["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-anchor sprites across a full grey barrier, swap their shape templates, and erase the rest.

    Input structure:
    - The background is black, and one grey component spans the full width or full height as a 1-cell barrier.
    - All remaining foreground pixels form 4-connected single-color sprites on the two sides of that barrier.
    - Exactly two non-grey sprite colors appear, each color reusing one template mask, and the two templates share a bounding-box size.

    Rule:
    - Detect whether the grey barrier is vertical or horizontal.
    - For a vertical barrier, pair a left sprite with a right sprite only when their bounding boxes have the same top row, the same size, and opposite colors; for a horizontal barrier, use the left column instead of the top row.
    - Clear every sprite, then redraw only the paired ones: each sprite keeps its original position and color, but receives its partner's mask shape; unpaired sprites stay erased and the barrier is unchanged.

    Key insight:
    - Crossing the barrier transmits geometry rather than location: the shape comes from the opposite side, while the color remains anchored in its original bounding box.
    """
    g = np.asarray(grid, dtype=int)
    barrier, orientation = _find_barrier(g)
    colors = tuple(sorted({int(v) for v in np.unique(g)} - {BG, BARRIER}))
    assert len(colors) == 2, "expected exactly two sprite colors"
    sprites = _extract_sprites(g, orientation, barrier["bbox"])
    pairs = _pair_sprites(sprites, orientation, colors)
    return _expected_from_analysis(g, sprites, pairs)


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(map(int, cells[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


_MASK_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _valid_masks(shape: tuple[int, int]) -> list[np.ndarray]:
    if shape in _MASK_CACHE:
        return _MASK_CACHE[shape]

    h, w = shape
    total = h * w
    masks: list[np.ndarray] = []
    min_area = max(3, (total + 1) // 2)
    max_area = min(total - 1, total - 1 if total <= 4 else total - 2)
    for area in range(min_area, max_area + 1):
        for combo in itertools.combinations(range(total), area):
            mask = np.zeros((h, w), dtype=bool)
            for idx in combo:
                mask[idx // w, idx % w] = True
            if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                continue
            if _is_connected(mask):
                masks.append(mask)
    _MASK_CACHE[shape] = masks
    return masks


def _choose_masks() -> tuple[tuple[int, int], dict[int, np.ndarray], tuple[int, int]]:
    shape = random.choice(BOX_SHAPES)
    color_a, color_b = random.sample(COLOR_POOL, 2)
    masks = _valid_masks(shape)
    pairs = [
        (mask_a, mask_b)
        for mask_a in masks
        for mask_b in masks
        if not np.array_equal(mask_a, mask_b)
        and int((mask_a ^ mask_b).sum()) >= 2
        and (int(mask_a.sum()) != int(mask_b.sum()) or int((mask_a != mask_b).sum()) >= 2)
    ]
    mask_a, mask_b = random.choice(pairs)
    return shape, {color_a: mask_a, color_b: mask_b}, (color_a, color_b)


def _sample_non_overlapping_starts(length: int, size: int, count: int) -> list[int] | None:
    candidates = list(range(1, length - size))
    random.shuffle(candidates)
    chosen: list[int] = []

    def search(pos: int) -> bool:
        if len(chosen) == count:
            return True
        if pos >= len(candidates):
            return False
        remaining = len(candidates) - pos
        if len(chosen) + remaining < count:
            return False

        start = candidates[pos]
        if all(abs(start - other) >= size + 1 for other in chosen):
            chosen.append(start)
            if search(pos + 1):
                return True
            chosen.pop()
        return search(pos + 1)

    if search(0):
        random.shuffle(chosen)
        return chosen
    return None


def _render_state(state: dict) -> np.ndarray:
    rows, cols = state["shape"]
    grid = np.zeros((rows, cols), dtype=int)
    if state["orientation"] == "vertical":
        grid[:, state["barrier_index"]] = BARRIER
    else:
        grid[state["barrier_index"], :] = BARRIER

    for sprite in state["sprites"]:
        _draw_mask(grid, sprite["bbox"], sprite["mask"], int(sprite["color"]))
    return grid


def _expected_from_state(state: dict) -> np.ndarray:
    rows, cols = state["shape"]
    out = np.zeros((rows, cols), dtype=int)
    if state["orientation"] == "vertical":
        out[:, state["barrier_index"]] = BARRIER
    else:
        out[state["barrier_index"], :] = BARRIER

    for idx0, idx1 in state["pairs"]:
        sprite0 = state["sprites"][idx0]
        sprite1 = state["sprites"][idx1]
        _draw_mask(out, sprite0["bbox"], sprite1["mask"], int(sprite0["color"]))
        _draw_mask(out, sprite1["bbox"], sprite0["mask"], int(sprite1["color"]))
    return out


def _wrong_global_swap(state: dict) -> np.ndarray:
    rows, cols = state["shape"]
    out = np.zeros((rows, cols), dtype=int)
    if state["orientation"] == "vertical":
        out[:, state["barrier_index"]] = BARRIER
    else:
        out[state["barrier_index"], :] = BARRIER
    colors = state["colors"]
    masks = state["masks_by_color"]
    for sprite in state["sprites"]:
        color = int(sprite["color"])
        _draw_mask(out, sprite["bbox"], masks[_other_color(colors, color)], color)
    return out


def _wrong_row_only_swap(state: dict) -> np.ndarray:
    rows, cols = state["shape"]
    out = np.zeros((rows, cols), dtype=int)
    if state["orientation"] == "vertical":
        out[:, state["barrier_index"]] = BARRIER
    else:
        out[state["barrier_index"], :] = BARRIER

    if state["orientation"] == "vertical":
        side0_name, side1_name = "left", "right"
    else:
        side0_name, side1_name = "top", "bottom"

    side0 = {int(sprite["anchor"]): idx for idx, sprite in enumerate(state["sprites"]) if sprite["side"] == side0_name}
    side1 = {int(sprite["anchor"]): idx for idx, sprite in enumerate(state["sprites"]) if sprite["side"] == side1_name}
    for anchor, idx0 in side0.items():
        if anchor not in side1:
            continue
        idx1 = side1[anchor]
        sprite0 = state["sprites"][idx0]
        sprite1 = state["sprites"][idx1]
        _draw_mask(out, sprite0["bbox"], sprite1["mask"], int(sprite0["color"]))
        _draw_mask(out, sprite1["bbox"], sprite0["mask"], int(sprite1["color"]))
    return out


def _candidate_count(sprite: dict, sprites: list[dict], colors: tuple[int, int], orientation: str) -> int:
    other = _other_color(colors, int(sprite["color"]))
    count = 0
    for candidate in sprites:
        if candidate["side"] == sprite["side"]:
            continue
        if int(candidate["color"]) != other:
            continue
        if int(candidate["anchor"]) != int(sprite["anchor"]):
            continue
        if candidate["size"] != sprite["size"]:
            continue
        count += 1
    return count


def _validate_state(state: dict) -> bool:
    inp = _render_state(state)
    expected = _expected_from_state(state)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    barrier, orientation = _find_barrier(inp)
    if orientation != state["orientation"]:
        return False
    sprites = _extract_sprites(inp, orientation, barrier["bbox"])
    if len(sprites) != len(state["sprites"]):
        return False
    colors = tuple(sorted({int(v) for v in np.unique(inp)} - {BG, BARRIER}))
    if colors != tuple(sorted(state["colors"])):
        return False

    pairs = _pair_sprites(sprites, orientation, colors)
    if len(pairs) != len(state["pairs"]):
        return False

    counts_by_side_color: dict[tuple[str, int], int] = {}
    for sprite in sprites:
        key = (str(sprite["side"]), int(sprite["color"]))
        counts_by_side_color[key] = counts_by_side_color.get(key, 0) + 1
    if orientation == "vertical":
        sides = ("left", "right")
    else:
        sides = ("top", "bottom")
    for side in sides:
        for color in colors:
            if counts_by_side_color.get((side, color), 0) == 0:
                return False

    unpaired = len(sprites) - 2 * len(pairs)
    if unpaired < 2:
        return False
    if any(_candidate_count(sprite, sprites, colors, orientation) > 1 for sprite in sprites):
        return False

    pair_color_signatures = {
        (int(sprites[idx0]["color"]), int(sprites[idx1]["color"]))
        for idx0, idx1 in pairs
    }
    needed = {(colors[0], colors[1]), (colors[1], colors[0])}
    if not needed.issubset(pair_color_signatures):
        return False

    if np.array_equal(_wrong_global_swap(state), expected):
        return False
    if state["decoy_count"] > 0 and np.array_equal(_wrong_row_only_swap(state), expected):
        return False
    return True


def _make_sprite(color: int, mask: np.ndarray, side: str, anchor: int, cross: int, orientation: str) -> dict:
    h, w = mask.shape
    if orientation == "vertical":
        r0, c0 = anchor, cross
    else:
        r0, c0 = cross, anchor
    return {
        "color": int(color),
        "mask": mask.copy(),
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "anchor": int(anchor),
        "side": side,
        "size": mask.shape,
    }


def _build_vertical(shape: tuple[int, int], colors: tuple[int, int], masks_by_color: dict[int, np.ndarray]) -> dict | None:
    h, w = shape
    pair_count = random.randint(2, 3)
    decoy_count = 1 if random.random() < 0.65 else 0
    solo_count = 2
    total_rows = pair_count + decoy_count + solo_count

    min_rows = total_rows * (h + 1) + 1
    min_cols = 2 * (w + 3) + 1
    if min_rows > 28 or min_cols > 28:
        return None
    rows = random.randint(min_rows, min(28, min_rows + 4))
    cols = random.randint(min_cols, min(28, min_cols + 6))
    barrier_col = random.randint(w + 3, cols - w - 4)

    anchors = _sample_non_overlapping_starts(rows, h, total_rows)
    if anchors is None:
        return None

    left_min, left_max = 1, barrier_col - w - 1
    right_min, right_max = barrier_col + 2, cols - w - 1
    if left_min > left_max or right_min > right_max:
        return None

    pair_rows = anchors[:pair_count]
    decoy_rows = anchors[pair_count : pair_count + decoy_count]
    solo_rows = anchors[pair_count + decoy_count :]

    pair_specs = [(colors[0], colors[1]), (colors[1], colors[0])]
    while len(pair_specs) < pair_count:
        if random.random() < 0.5:
            pair_specs.append((colors[0], colors[1]))
        else:
            pair_specs.append((colors[1], colors[0]))
    random.shuffle(pair_specs)

    sprites: list[dict] = []
    pairs: list[tuple[int, int]] = []

    for anchor, (left_color, right_color) in zip(pair_rows, pair_specs):
        left_sprite = _make_sprite(
            left_color,
            masks_by_color[left_color],
            "left",
            anchor,
            random.randint(left_min, left_max),
            "vertical",
        )
        right_sprite = _make_sprite(
            right_color,
            masks_by_color[right_color],
            "right",
            anchor,
            random.randint(right_min, right_max),
            "vertical",
        )
        idx0 = len(sprites)
        sprites.append(left_sprite)
        idx1 = len(sprites)
        sprites.append(right_sprite)
        pairs.append((idx0, idx1))

    for anchor in decoy_rows:
        color = random.choice(colors)
        sprites.append(
            _make_sprite(
                color,
                masks_by_color[color],
                "left",
                anchor,
                random.randint(left_min, left_max),
                "vertical",
            )
        )
        sprites.append(
            _make_sprite(
                color,
                masks_by_color[color],
                "right",
                anchor,
                random.randint(right_min, right_max),
                "vertical",
            )
        )

    solo_sides = ["left", "right"]
    random.shuffle(solo_sides)
    for anchor, side in zip(solo_rows, solo_sides):
        color = random.choice(colors)
        cross = random.randint(left_min, left_max) if side == "left" else random.randint(right_min, right_max)
        sprites.append(_make_sprite(color, masks_by_color[color], side, anchor, cross, "vertical"))

    return {
        "shape": (rows, cols),
        "orientation": "vertical",
        "barrier_index": barrier_col,
        "sprites": sprites,
        "pairs": pairs,
        "masks_by_color": masks_by_color,
        "colors": colors,
        "decoy_count": decoy_count,
    }


def _build_horizontal(shape: tuple[int, int], colors: tuple[int, int], masks_by_color: dict[int, np.ndarray]) -> dict | None:
    h, w = shape
    pair_count = random.randint(2, 3)
    decoy_count = 1 if random.random() < 0.65 else 0
    solo_count = 2
    total_cols = pair_count + decoy_count + solo_count

    min_cols = total_cols * (w + 1) + 1
    min_rows = 2 * (h + 3) + 1
    if min_cols > 28 or min_rows > 28:
        return None
    cols = random.randint(min_cols, min(28, min_cols + 4))
    rows = random.randint(min_rows, min(28, min_rows + 6))
    barrier_row = random.randint(h + 3, rows - h - 4)

    anchors = _sample_non_overlapping_starts(cols, w, total_cols)
    if anchors is None:
        return None

    top_min, top_max = 1, barrier_row - h - 1
    bottom_min, bottom_max = barrier_row + 2, rows - h - 1
    if top_min > top_max or bottom_min > bottom_max:
        return None

    pair_cols = anchors[:pair_count]
    decoy_cols = anchors[pair_count : pair_count + decoy_count]
    solo_cols = anchors[pair_count + decoy_count :]

    pair_specs = [(colors[0], colors[1]), (colors[1], colors[0])]
    while len(pair_specs) < pair_count:
        if random.random() < 0.5:
            pair_specs.append((colors[0], colors[1]))
        else:
            pair_specs.append((colors[1], colors[0]))
    random.shuffle(pair_specs)

    sprites: list[dict] = []
    pairs: list[tuple[int, int]] = []

    for anchor, (top_color, bottom_color) in zip(pair_cols, pair_specs):
        top_sprite = _make_sprite(
            top_color,
            masks_by_color[top_color],
            "top",
            anchor,
            random.randint(top_min, top_max),
            "horizontal",
        )
        bottom_sprite = _make_sprite(
            bottom_color,
            masks_by_color[bottom_color],
            "bottom",
            anchor,
            random.randint(bottom_min, bottom_max),
            "horizontal",
        )
        idx0 = len(sprites)
        sprites.append(top_sprite)
        idx1 = len(sprites)
        sprites.append(bottom_sprite)
        pairs.append((idx0, idx1))

    for anchor in decoy_cols:
        color = random.choice(colors)
        sprites.append(
            _make_sprite(
                color,
                masks_by_color[color],
                "top",
                anchor,
                random.randint(top_min, top_max),
                "horizontal",
            )
        )
        sprites.append(
            _make_sprite(
                color,
                masks_by_color[color],
                "bottom",
                anchor,
                random.randint(bottom_min, bottom_max),
                "horizontal",
            )
        )

    solo_sides = ["top", "bottom"]
    random.shuffle(solo_sides)
    for anchor, side in zip(solo_cols, solo_sides):
        color = random.choice(colors)
        cross = random.randint(top_min, top_max) if side == "top" else random.randint(bottom_min, bottom_max)
        sprites.append(_make_sprite(color, masks_by_color[color], side, anchor, cross, "horizontal"))

    return {
        "shape": (rows, cols),
        "orientation": "horizontal",
        "barrier_index": barrier_row,
        "sprites": sprites,
        "pairs": pairs,
        "masks_by_color": masks_by_color,
        "colors": colors,
        "decoy_count": decoy_count,
    }


def _generate_state() -> dict | None:
    shape, masks_by_color, colors = _choose_masks()
    if random.random() < 0.5:
        state = _build_vertical(shape, colors, masks_by_color)
    else:
        state = _build_horizontal(shape, colors, masks_by_color)
    if state is None:
        return None
    return state if _validate_state(state) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        state = _generate_state()
        if state is not None:
            return _render_state(state)
    raise RuntimeError("failed to generate a valid puzzle instance")
