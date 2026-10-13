from __future__ import annotations

import random
from collections import Counter

import numpy as np


BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_side(box: BBox, shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = box
    h, w = shape
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, "each rectangle must touch exactly one edge"
    return sides[0]


def _swap_patch(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    out[patch == a] = b
    out[patch == b] = a
    return out


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components(g != 0)
    assert comps, "expected at least one foreground rectangle"

    objects = []
    for coords in comps:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert int(coords.shape[0]) == h * w, "foreground object must be a solid rectangle"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, patch.ravel())))
        assert 0 not in colors
        assert len(colors) == 2, "each rectangle must contain exactly two colors"
        counts = Counter(map(int, patch.ravel()))
        a, b = colors
        assert counts[a] != counts[b], "rectangle colors must have a strict minority"
        minority, majority = (a, b) if counts[a] < counts[b] else (b, a)
        objects.append(
            {
                "bbox": box,
                "side": _edge_side(box, g.shape),
                "h": h,
                "w": w,
                "patch": patch.copy(),
                "minority": int(minority),
                "majority": int(majority),
                "colors": (a, b),
            }
        )

    used_sides = [obj["side"] for obj in objects]
    assert len(set(used_sides)) == len(used_sides), "at most one rectangle may touch each edge"
    color_pairs = [tuple(sorted(obj["colors"])) for obj in objects]
    assert len(set(color_pairs)) == len(color_pairs), "each rectangle must use a distinct color pair"
    return objects


def _solve_from_objects(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        rect_h = obj["h"]
        rect_w = obj["w"]
        trail = obj["minority"]
        if obj["side"] == "top":
            out[: h - rect_h, c0 : c1 + 1] = trail
            dr0, dc0 = h - rect_h, c0
        elif obj["side"] == "bottom":
            out[rect_h:, c0 : c1 + 1] = trail
            dr0, dc0 = 0, c0
        elif obj["side"] == "left":
            out[r0 : r1 + 1, : w - rect_w] = trail
            dr0, dc0 = r0, w - rect_w
        else:
            out[r0 : r1 + 1, rect_w:] = trail
            dr0, dc0 = r0, 0

        swapped = _swap_patch(obj["patch"], obj["minority"], obj["majority"])
        out[dr0 : dr0 + rect_h, dc0 : dc0 + rect_w] = swapped

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each edge-touching bicolor rectangle to the opposite edge, paint its trail with the rectangle's minority color, and swap the two colors at the destination.

    Input structure:
    - The grid has a black background and 2-4 separated solid rectangles.
    - Each rectangle touches exactly one outer edge and fills its own bounding box.
    - Inside each rectangle there are exactly two non-black colors, with unequal pixel counts.

    Rule:
    - For each rectangle, use the touched edge to determine the slide direction toward the opposite edge while keeping the same row band or column band.
    - Count the two colors inside that rectangle; the less frequent color becomes the trail color.
    - Paint every swept cell from the source position up to but not including the destination rectangle with that minority color.
    - Stamp the rectangle at the opposite edge and swap its two internal colors there, preserving the geometry of the pattern.

    Key insight:
    - The trail color is not fixed globally: it is determined locally by frequency analysis inside each moving rectangle, and the destination rectangle then flips those two roles.
    """
    g = np.asarray(grid, dtype=int)
    objects = _parse_objects(g)
    return _solve_from_objects(g.shape, objects)


def _add_mask(options: dict[tuple[int, ...], np.ndarray], h: int, w: int, cells: list[tuple[int, int]]) -> None:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True
    count = int(mask.sum())
    area = h * w
    if 0 < count < area - count:
        options[tuple(mask.ravel().tolist())] = mask


def _minority_mask_options(h: int, w: int) -> list[np.ndarray]:
    opts: dict[tuple[int, ...], np.ndarray] = {}
    cr = h // 2
    cc = w // 2

    for cell in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (cr, cc)]:
        _add_mask(opts, h, w, [cell])

    if h >= 3:
        _add_mask(opts, h, w, [(0, c) for c in range(w)])
        _add_mask(opts, h, w, [(h - 1, c) for c in range(w)])
    if w >= 3:
        _add_mask(opts, h, w, [(r, 0) for r in range(h)])
        _add_mask(opts, h, w, [(r, w - 1) for r in range(h)])
    if w >= 2:
        _add_mask(opts, h, w, [(0, 0), (0, w - 1)])
        _add_mask(opts, h, w, [(h - 1, 0), (h - 1, w - 1)])
    if h >= 2:
        _add_mask(opts, h, w, [(0, 0), (h - 1, 0)])
        _add_mask(opts, h, w, [(0, w - 1), (h - 1, w - 1)])
    if h >= 3 and w >= 3:
        _add_mask(opts, h, w, [(cr, max(0, cc - 1)), (cr, cc)])
        _add_mask(opts, h, w, [(max(0, cr - 1), cc), (cr, cc)])
        _add_mask(opts, h, w, [(0, 0), (cr, cc)])
        _add_mask(opts, h, w, [(0, w - 1), (cr, cc)])
        _add_mask(opts, h, w, [(h - 1, 0), (cr, cc)])
        _add_mask(opts, h, w, [(h - 1, w - 1), (cr, cc)])
        _add_mask(opts, h, w, [(0, 0), (h - 1, w - 1)])
        _add_mask(opts, h, w, [(0, w - 1), (h - 1, 0)])
    if h >= 4 and w >= 4:
        _add_mask(opts, h, w, [(cr - 1, cc - 1), (cr - 1, cc), (cr, cc - 1), (cr, cc)])

    return list(opts.values())


def _pattern_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(mask.astype(int).ravel().tolist())


def _choose_patch(h: int, w: int, minority: int, majority: int, used: set[tuple[int, ...]]) -> np.ndarray | None:
    masks = _minority_mask_options(h, w)
    random.shuffle(masks)
    for mask in masks:
        sig = _pattern_signature(mask)
        if sig in used:
            continue
        patch = np.full((h, w), majority, dtype=int)
        patch[mask] = minority
        used.add(sig)
        return patch
    return None


def _touches_rect_boundary(mask: np.ndarray) -> bool:
    h, w = mask.shape
    rr, cc = np.argwhere(mask).T
    return bool(np.any(rr == 0) or np.any(rr == h - 1) or np.any(cc == 0) or np.any(cc == w - 1))


def _make_object(side: str, r0: int, c0: int, patch: np.ndarray) -> dict:
    h, w = patch.shape
    return {"side": side, "bbox": (r0, c0, r0 + h - 1, c0 + w - 1), "patch": patch}


def _distinct_colors(n_rects: int) -> list[tuple[int, int]]:
    colors = random.sample(list(range(1, 10)), 2 * n_rects)
    return [(colors[2 * i], colors[2 * i + 1]) for i in range(n_rects)]


def _random_interval(total: int, length: int, forbidden: list[tuple[int, int]], margin: int = 1) -> int | None:
    starts = list(range(margin, total - margin - length + 1))
    random.shuffle(starts)
    for s in starts:
        e = s + length - 1
        ok = True
        for a, b in forbidden:
            if not (e < a or b < s):
                ok = False
                break
        if ok:
            return s
    return None


def _generate_vertical_only() -> list[dict] | None:
    h = random.randint(14, 22)
    w = random.randint(12, 22)
    top_h, bot_h = random.randint(2, 4), random.randint(2, 4)
    top_w, bot_w = random.randint(2, 4), random.randint(2, 4)

    top_c = _random_interval(w, top_w, [])
    if top_c is None:
        return None
    bot_c = _random_interval(w, bot_w, [(top_c - 1, top_c + top_w)], margin=1)
    if bot_c is None:
        return None

    used_patterns: set[tuple[int, ...]] = set()
    pairs = _distinct_colors(2)
    patch_top = _choose_patch(top_h, top_w, pairs[0][0], pairs[0][1], used_patterns)
    patch_bot = _choose_patch(bot_h, bot_w, pairs[1][0], pairs[1][1], used_patterns)
    if patch_top is None or patch_bot is None:
        return None

    return [
        {"shape": (h, w), "objects": [_make_object("top", 0, top_c, patch_top), _make_object("bottom", h - bot_h, bot_c, patch_bot)]}
    ]


def _generate_horizontal_only() -> list[dict] | None:
    h = random.randint(12, 22)
    w = random.randint(14, 22)
    left_h, right_h = random.randint(2, 4), random.randint(2, 4)
    left_w, right_w = random.randint(2, 4), random.randint(2, 4)

    left_r = _random_interval(h, left_h, [])
    if left_r is None:
        return None
    right_r = _random_interval(h, right_h, [(left_r - 1, left_r + left_h)], margin=1)
    if right_r is None:
        return None

    used_patterns: set[tuple[int, ...]] = set()
    pairs = _distinct_colors(2)
    patch_left = _choose_patch(left_h, left_w, pairs[0][0], pairs[0][1], used_patterns)
    patch_right = _choose_patch(right_h, right_w, pairs[1][0], pairs[1][1], used_patterns)
    if patch_left is None or patch_right is None:
        return None

    return [
        {"shape": (h, w), "objects": [_make_object("left", left_r, 0, patch_left), _make_object("right", right_r, w - right_w, patch_right)]}
    ]


def _build_grid(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = obj["patch"]
    return grid


def _expected_from_generated(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    parsed = []
    for obj in objects:
        patch = obj["patch"]
        counts = Counter(map(int, patch.ravel()))
        colors = sorted(counts)
        a, b = colors
        minority, majority = (a, b) if counts[a] < counts[b] else (b, a)
        r0, c0, r1, c1 = obj["bbox"]
        parsed.append(
            {
                "bbox": obj["bbox"],
                "side": obj["side"],
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "patch": patch,
                "minority": minority,
                "majority": majority,
            }
        )
    return _solve_from_objects(shape, parsed)


def _validate_generated_case(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        objects = _parse_objects(grid)
    except AssertionError:
        return False

    if len(objects) != 2:
        return False

    boundary_flags = []
    interior_flags = []
    for obj in objects:
        patch = obj["patch"]
        mask = patch == obj["minority"]
        boundary_flags.append(_touches_rect_boundary(mask))
        interior_flags.append(bool(np.any(mask[1:-1, 1:-1])) if min(mask.shape) >= 3 else False)

    if not any(boundary_flags):
        return False
    if not any(interior_flags):
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        patch = obj["patch"]
        counts = Counter(map(int, patch.ravel()))
        colors = sorted(counts)
        a, b = colors
        minority, majority = (a, b) if counts[a] < counts[b] else (b, a)
        swapped = _swap_patch(patch, minority, majority)

        if obj["side"] == "top":
            dr0, dc0 = grid.shape[0] - h, c0
            trail_ok = np.all(actual[: grid.shape[0] - h, c0 : c0 + w] == minority)
        elif obj["side"] == "bottom":
            dr0, dc0 = 0, c0
            trail_ok = np.all(actual[h:, c0 : c0 + w] == minority)
        elif obj["side"] == "left":
            dr0, dc0 = r0, grid.shape[1] - w
            trail_ok = np.all(actual[r0 : r0 + h, : grid.shape[1] - w] == minority)
        else:
            dr0, dc0 = r0, 0
            trail_ok = np.all(actual[r0 : r0 + h, w:] == minority)

        if not trail_ok:
            return False
        if not np.array_equal(actual[dr0 : dr0 + h, dc0 : dc0 + w], swapped):
            return False

    return True


def generate_input() -> np.ndarray:
    builders = [_generate_vertical_only, _generate_horizontal_only]
    weights = [0.5, 0.5]
    for _ in range(300):
        builder = random.choices(builders, weights=weights, k=1)[0]
        cases = builder()
        if not cases:
            continue
        case = cases[0]
        grid = _build_grid(case["shape"], case["objects"])
        expected = _expected_from_generated(case["shape"], case["objects"])
        if _validate_generated_case(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
