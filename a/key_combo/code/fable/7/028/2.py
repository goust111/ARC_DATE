from __future__ import annotations

import random
from typing import Any

import numpy as np


ColorPair = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _inside(box: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _apply_swap_value(value: int, pair: ColorPair) -> int:
    a, b = pair
    if value == a:
        return b
    if value == b:
        return a
    return value


def _apply_swap_array(grid: np.ndarray, pair: ColorPair) -> np.ndarray:
    a, b = pair
    out = np.array(grid, copy=True)
    a_mask = out == a
    b_mask = out == b
    out[a_mask] = b
    out[b_mask] = a
    return out


def _partial_vertical_overlap(box: BBox, width: int) -> bool:
    _r0, c0, _r1, c1 = box
    mc0 = width - 1 - c1
    mc1 = width - 1 - c0
    overlap = max(0, min(c1, mc1) - max(c0, mc0) + 1)
    side = c1 - c0 + 1
    return 0 < overlap < side


def _find_occluder(grid: np.ndarray) -> tuple[int, BBox]:
    h, w = grid.shape
    candidates: list[tuple[int, BBox]] = []
    for color in map(int, np.unique(grid)):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        if side_r != side_c or side_r < 2:
            continue
        if int(coords.shape[0]) != side_r * side_c:
            continue
        if not _partial_vertical_overlap(box, w):
            continue
        candidates.append((color, box))
    assert len(candidates) == 1, "expected one unique partially self-overlapping solid-square occluder"
    return candidates[0]


def _infer_swap_pair(grid: np.ndarray, occluder_color: int, axis: str) -> ColorPair:
    h, w = grid.shape
    diff_pairs: set[ColorPair] = set()
    unequal_count = 0

    if axis == "vertical":
        coords = ((r, c, r, w - 1 - c) for r in range(h) for c in range(w // 2))
    elif axis == "horizontal":
        coords = ((r, c, h - 1 - r, c) for r in range(h // 2) for c in range(w))
    else:
        raise ValueError("axis must be 'vertical' or 'horizontal'")

    for r0, c0, r1, c1 in coords:
        a = int(grid[r0, c0])
        b = int(grid[r1, c1])
        if a == occluder_color or b == occluder_color:
            continue
        if a != b:
            unequal_count += 1
            diff_pairs.add(tuple(sorted((a, b))))

    assert unequal_count > 0, f"expected visible unequal mirrored pairs for {axis} swap inference"
    assert len(diff_pairs) == 1, f"{axis} swap must be uniquely inferable"
    pair = next(iter(diff_pairs))

    for r0, c0, r1, c1 in coords:
        a = int(grid[r0, c0])
        b = int(grid[r1, c1])
        if a == occluder_color or b == occluder_color:
            continue
        if a == b:
            assert a not in pair, f"{axis} swap color cannot mirror to itself"
        else:
            assert tuple(sorted((a, b))) == pair, f"inconsistent {axis} swap evidence"

    return pair


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the occluded square by splitting it between two color-swapping mirror axes.

    Input structure:
    - The input is an even-sized square scene with one unique solid-color square occluder.
    - The occluder color appears nowhere else, and its box partially overlaps its own vertical mirror.
    - Outside the occluder, the scene is symmetric across the vertical axis up to one color swap, and across the horizontal axis up to a different color swap.

    Rule:
    - Identify the unique solid square occluder and infer the vertical and horizontal color-swap pairs from visible mirrored cell pairs.
    - For each hidden cell, first look at its vertical mirror source and apply the vertical swap.
    - If that vertical source still lies inside the occluder, instead use the horizontal mirror source and apply the horizontal swap.
    - Output only the recovered occluder patch.

    Key insight:
    - The visible scene certifies two distinct mirror symmetries with different recolorings; the occluder is recovered by routing each hidden cell to whichever axis still has an uncontaminated mirror source.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    h, w = grid.shape
    assert h == w and h % 2 == 0, "expected an even square grid"

    occluder_color, box = _find_occluder(grid)
    v_pair = _infer_swap_pair(grid, occluder_color, "vertical")
    h_pair = _infer_swap_pair(grid, occluder_color, "horizontal")
    assert v_pair != h_pair, "the two swap pairs must differ"

    r0, c0, r1, c1 = box
    side = r1 - r0 + 1
    out = np.zeros((side, side), dtype=int)

    for dr in range(side):
        for dc in range(side):
            rr = r0 + dr
            cc = c0 + dc

            v_src = (rr, w - 1 - cc)
            h_src = (h - 1 - rr, cc)

            candidates: list[int] = []
            chosen: int | None = None

            if not _inside(box, *v_src):
                value = _apply_swap_value(int(grid[v_src]), v_pair)
                candidates.append(value)
                chosen = value

            if not _inside(box, *h_src):
                value = _apply_swap_value(int(grid[h_src]), h_pair)
                candidates.append(value)
                if chosen is None:
                    chosen = value

            assert chosen is not None, "every hidden cell needs at least one visible mirror source"
            assert all(value == chosen for value in candidates), "mirror reconstructions must agree"
            out[dr, dc] = chosen

    return out


def _compose_scene(quadrant: np.ndarray, v_pair: ColorPair, h_pair: ColorPair) -> np.ndarray:
    top_left = quadrant
    top_right = _apply_swap_array(np.fliplr(top_left), v_pair)
    bottom_left = _apply_swap_array(np.flipud(top_left), h_pair)
    bottom_right = _apply_swap_array(np.flipud(top_right), h_pair)
    return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def _paint_random_rectangles(quadrant: np.ndarray, palette: list[int], bg: int) -> None:
    size = quadrant.shape[0]
    max_span = max(2, size // 3)

    for _ in range(random.randint(size + 3, size + 10)):
        rh = random.randint(1, max_span)
        cw = random.randint(1, max_span)
        r0 = random.randint(0, size - rh)
        c0 = random.randint(0, size - cw)
        color = random.choice(palette)
        quadrant[r0 : r0 + rh, c0 : c0 + cw] = color

    for _ in range(random.randint(size // 2, size)):
        r = random.randrange(size)
        c = random.randrange(size)
        color = random.choice(palette)
        quadrant[r, c] = color

    spots = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(spots)
    forced_colors = palette[:]
    random.shuffle(forced_colors)
    for color in forced_colors:
        r, c = spots.pop()
        quadrant[r, c] = color

    if np.all(quadrant == bg):
        quadrant[random.randrange(size), random.randrange(size)] = random.choice(palette[1:])


def _choose_occluder_box(size: int, side: int) -> BBox:
    half = size // 2
    left_cols = random.randint(1, side - 1)
    while left_cols * 2 == side:
        left_cols = random.randint(1, side - 1)
    right_cols = side - left_cols
    c0 = half - left_cols
    c1 = half + right_cols - 1

    top_range = list(range(0, half - side + 1))
    bottom_range = list(range(half, size - side + 1))
    starts = []
    if top_range:
        starts.extend(("top", r) for r in top_range)
    if bottom_range:
        starts.extend(("bottom", r) for r in bottom_range)
    band, r0 = random.choice(starts)
    _ = band
    return int(r0), int(c0), int(r0 + side - 1), int(c1)


def _visible_unequal_pairs(grid: np.ndarray, occluder_color: int, axis: str) -> int:
    h, w = grid.shape
    count = 0
    if axis == "vertical":
        for r in range(h):
            for c in range(w // 2):
                a = int(grid[r, c])
                b = int(grid[r, w - 1 - c])
                if occluder_color not in (a, b) and a != b:
                    count += 1
    else:
        for r in range(h // 2):
            for c in range(w):
                a = int(grid[r, c])
                b = int(grid[h - 1 - r, c])
                if occluder_color not in (a, b) and a != b:
                    count += 1
    return count


def _branch_masks(box: BBox, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    r0, c0, r1, c1 = box
    side = r1 - r0 + 1
    use_v = np.zeros((side, side), dtype=bool)
    use_h = np.zeros((side, side), dtype=bool)
    for dr in range(side):
        for dc in range(side):
            rr = r0 + dr
            cc = c0 + dc
            v_src = (rr, w - 1 - cc)
            if _inside(box, *v_src):
                use_h[dr, dc] = True
            else:
                use_v[dr, dc] = True
    return use_v, use_h


def _generate_case(*, return_meta: bool = False) -> Any:
    size = random.choice(list(range(18, 31, 2)))
    half = size // 2
    side = random.randint(4, min(8, half))

    all_colors = list(range(10))
    random.shuffle(all_colors)
    occluder_color = all_colors.pop()
    bg = all_colors.pop()
    v_pair = tuple(sorted((all_colors.pop(), all_colors.pop())))
    h_pair = tuple(sorted((all_colors.pop(), all_colors.pop())))
    while set(v_pair) & set(h_pair):
        all_colors = [c for c in range(10) if c not in {occluder_color, bg}]
        random.shuffle(all_colors)
        v_pair = tuple(sorted((all_colors.pop(), all_colors.pop())))
        h_pair = tuple(sorted((all_colors.pop(), all_colors.pop())))
    fixed_count = random.randint(1, min(2, len(all_colors)))
    fixed_colors = random.sample(all_colors, fixed_count)
    palette = [bg, *v_pair, *h_pair, *fixed_colors]

    quadrant = np.full((half, half), bg, dtype=int)
    _paint_random_rectangles(quadrant, palette, bg)
    scene = _compose_scene(quadrant, v_pair, h_pair)
    box = _choose_occluder_box(size, side)
    expected = _crop(scene, box)

    inp = scene.copy()
    r0, c0, r1, c1 = box
    inp[r0 : r1 + 1, c0 : c1 + 1] = occluder_color

    if return_meta:
        return {
            "input": inp,
            "scene": scene,
            "box": box,
            "expected": expected,
            "occluder_color": occluder_color,
            "v_pair": v_pair,
            "h_pair": h_pair,
        }
    return inp


def _semantic_checks(case: dict[str, Any]) -> bool:
    inp = case["input"]
    scene = case["scene"]
    box = case["box"]
    expected = case["expected"]
    occluder_color = case["occluder_color"]
    v_pair = case["v_pair"]
    h_pair = case["h_pair"]

    if not _partial_vertical_overlap(box, inp.shape[1]):
        return False
    if _infer_swap_pair(inp, occluder_color, "vertical") != v_pair:
        return False
    if _infer_swap_pair(inp, occluder_color, "horizontal") != h_pair:
        return False
    if _visible_unequal_pairs(inp, occluder_color, "vertical") < 2:
        return False
    if _visible_unequal_pairs(inp, occluder_color, "horizontal") < 2:
        return False

    use_v, use_h = _branch_masks(box, inp.shape)
    if not use_v.any() or not use_h.any():
        return False

    out = main(inp)
    if not np.array_equal(out, expected):
        return False
    if len(np.unique(expected)) < 3:
        return False

    v_needed = False
    h_needed = False
    r0, c0, r1, c1 = box
    side = r1 - r0 + 1
    h, w = inp.shape
    for dr in range(side):
        for dc in range(side):
            rr = r0 + dr
            cc = c0 + dc
            if use_v[dr, dc]:
                src = int(scene[rr, w - 1 - cc])
                if _apply_swap_value(src, v_pair) != src:
                    v_needed = True
            if use_h[dr, dc]:
                src = int(scene[h - 1 - rr, cc])
                if _apply_swap_value(src, h_pair) != src:
                    h_needed = True

    if not v_needed or not h_needed:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        case = _generate_case(return_meta=True)
        try:
            if _semantic_checks(case):
                return np.array(case["input"], copy=True)
        except AssertionError:
            continue
    raise RuntimeError("generate_input: failed to build a non-degenerate dual-axis recolor case")
