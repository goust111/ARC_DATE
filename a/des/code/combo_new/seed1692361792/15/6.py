import random
from collections import deque

import numpy as np


BG = 0
SHAPE_COLOR = 5
GEN_ATTEMPTS = 500
SIDE_NAMES = ("top", "bottom", "left", "right")
SIZE_FAMILIES = (
    ((1, 2), (2, 3), (3, 1)),
    ((1, 3), (2, 1), (3, 2)),
)
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(grid: np.ndarray, exclude: set[int] | None = None) -> list[dict]:
    exclude = exclude or set()
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in exclude:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == int(comp["color"])))


def _largest_non_bg_component(grid: np.ndarray, bg: int) -> dict:
    comps = _components(grid, exclude={bg})
    assert comps, "expected at least one foreground component"
    best_area = max(int(comp["area"]) for comp in comps)
    best = [comp for comp in comps if int(comp["area"]) == best_area]
    assert len(best) == 1, "expected a unique largest foreground component"
    return best[0]


def _shape_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r), int(c)] = True
    return mask


def _all_true(mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w or r0 > r1 or c0 > c1:
        return False
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _all_false_or_oob(mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    h, w = mask.shape
    if r0 > r1 or c0 > c1:
        return False
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return True
    return bool(np.all(~mask[r0 : r1 + 1, c0 : c1 + 1]))


def _find_matches(
    grid: np.ndarray,
    shape_mask: np.ndarray,
    bg: int,
    size: tuple[int, int],
    kind: str,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    h, w = grid.shape
    fh, fw = size
    matches = []
    for r0 in range(h - fh + 1):
        r1 = r0 + fh - 1
        for c0 in range(w - fw + 1):
            c1 = c0 + fw - 1
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if kind == "notch":
                if not np.all(patch == bg):
                    continue
                if (
                    _all_true(shape_mask, r1 + 1, r1 + 1, c0, c1)
                    and _all_true(shape_mask, r0, r1, c0 - 1, c0 - 1)
                    and _all_true(shape_mask, r0, r1, c1 + 1, c1 + 1)
                    and _all_false_or_oob(shape_mask, r0 - 1, r0 - 1, c0, c1)
                ):
                    matches.append(("top", (r0, c0, r1, c1)))
                if (
                    _all_true(shape_mask, r0 - 1, r0 - 1, c0, c1)
                    and _all_true(shape_mask, r0, r1, c0 - 1, c0 - 1)
                    and _all_true(shape_mask, r0, r1, c1 + 1, c1 + 1)
                    and _all_false_or_oob(shape_mask, r1 + 1, r1 + 1, c0, c1)
                ):
                    matches.append(("bottom", (r0, c0, r1, c1)))
                if (
                    _all_true(shape_mask, r0, r1, c1 + 1, c1 + 1)
                    and _all_true(shape_mask, r0 - 1, r0 - 1, c0, c1)
                    and _all_true(shape_mask, r1 + 1, r1 + 1, c0, c1)
                    and _all_false_or_oob(shape_mask, r0, r1, c0 - 1, c0 - 1)
                ):
                    matches.append(("left", (r0, c0, r1, c1)))
                if (
                    _all_true(shape_mask, r0, r1, c0 - 1, c0 - 1)
                    and _all_true(shape_mask, r0 - 1, r0 - 1, c0, c1)
                    and _all_true(shape_mask, r1 + 1, r1 + 1, c0, c1)
                    and _all_false_or_oob(shape_mask, r0, r1, c1 + 1, c1 + 1)
                ):
                    matches.append(("right", (r0, c0, r1, c1)))
            else:
                if not np.all(shape_mask[r0 : r1 + 1, c0 : c1 + 1]):
                    continue
                if (
                    _all_true(shape_mask, r1 + 1, r1 + 1, c0, c1)
                    and _all_false_or_oob(shape_mask, r0 - 1, r0 - 1, c0, c1)
                    and _all_false_or_oob(shape_mask, r0, r1, c0 - 1, c0 - 1)
                    and _all_false_or_oob(shape_mask, r0, r1, c1 + 1, c1 + 1)
                ):
                    matches.append(("top", (r0, c0, r1, c1)))
                if (
                    _all_true(shape_mask, r0 - 1, r0 - 1, c0, c1)
                    and _all_false_or_oob(shape_mask, r1 + 1, r1 + 1, c0, c1)
                    and _all_false_or_oob(shape_mask, r0, r1, c0 - 1, c0 - 1)
                    and _all_false_or_oob(shape_mask, r0, r1, c1 + 1, c1 + 1)
                ):
                    matches.append(("bottom", (r0, c0, r1, c1)))
                if (
                    _all_true(shape_mask, r0, r1, c1 + 1, c1 + 1)
                    and _all_false_or_oob(shape_mask, r0, r1, c0 - 1, c0 - 1)
                    and _all_false_or_oob(shape_mask, r0 - 1, r0 - 1, c0, c1)
                    and _all_false_or_oob(shape_mask, r1 + 1, r1 + 1, c0, c1)
                ):
                    matches.append(("left", (r0, c0, r1, c1)))
                if (
                    _all_true(shape_mask, r0, r1, c0 - 1, c0 - 1)
                    and _all_false_or_oob(shape_mask, r0, r1, c1 + 1, c1 + 1)
                    and _all_false_or_oob(shape_mask, r0 - 1, r0 - 1, c0, c1)
                    and _all_false_or_oob(shape_mask, r1 + 1, r1 + 1, c0, c1)
                ):
                    matches.append(("right", (r0, c0, r1, c1)))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each colored stamp to fill a matching boundary notch and erase the matching boundary tab.

    Input structure:
    - The background is black.
    - One large grey 4-connected foreground object forms an irregular orthogonal shape.
    - Its boundary contains rectangular black notches and rectangular grey protrusions.
    - Several disconnected colored solid-rectangle stamps are scattered elsewhere in the grid.

    Rule:
    - Find the largest non-background component as the main grey shape.
    - Treat every other monochrome solid rectangular component as a stamp; each stamp size appears exactly once as a notch and once as a protrusion on the main shape boundary.
    - For each stamp, clear the stamp's original cells to black, fill the matching black notch with the stamp color, and erase the matching grey protrusion to black.

    Key insight:
    - The stamp size, not its position, links three objects at once: the stamp itself, an inward rectangular notch to color, and an outward rectangular tab to remove.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _components(g, exclude={bg})
    assert comps, "expected at least one foreground component"
    best_area = max(int(comp["area"]) for comp in comps)
    main_candidates = [comp for comp in comps if int(comp["area"]) == best_area]
    assert len(main_candidates) == 1, "expected a unique largest foreground component"
    main_comp = main_candidates[0]
    shape = _shape_mask(g, main_comp)
    main_key = (int(main_comp["color"]), tuple(main_comp["bbox"]), int(main_comp["area"]))

    stamps = []
    for comp in comps:
        comp_key = (int(comp["color"]), tuple(comp["bbox"]), int(comp["area"]))
        if comp_key == main_key:
            continue
        assert _is_solid_rectangle(g, comp), "stamp components must be solid rectangles"
        stamps.append(comp)

    assert 2 <= len(stamps) <= 3, "expected two or three stamps"
    sizes = [_bbox_size(comp["bbox"]) for comp in stamps]
    assert len(set(sizes)) == len(sizes), "stamp sizes must be unique"

    out = g.copy()
    for comp in stamps:
        r0, c0, r1, c1 = comp["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for comp in stamps:
        color = int(comp["color"])
        size = _bbox_size(comp["bbox"])
        notch_matches = _find_matches(g, shape, bg, size, kind="notch")
        protrusion_matches = _find_matches(g, shape, bg, size, kind="protrusion")
        assert len(notch_matches) == 1, f"expected one notch for size {size}"
        assert len(protrusion_matches) == 1, f"expected one protrusion for size {size}"

        _, (nr0, nc0, nr1, nc1) = notch_matches[0]
        _, (pr0, pc0, pr1, pc1) = protrusion_matches[0]
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = color
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = bg

    return out


def _pick_sizes(pair_count: int) -> list[tuple[int, int]]:
    family = list(random.choice(SIZE_FAMILIES))
    random.shuffle(family)
    return family[:pair_count]


def _span_on_side(size: tuple[int, int], side: str) -> int:
    h, w = size
    return w if side in {"top", "bottom"} else h


def _required_side_length(sizes: list[tuple[int, int]], sides: list[str], target_side: str) -> int:
    spans = [_span_on_side(size, side) for size, side in zip(sizes, sides) if side == target_side]
    if not spans:
        return 0
    return sum(spans) + len(spans) + 1


def _layout_side(length: int, items: list[tuple[tuple[int, str], int]]) -> dict[tuple[int, str], int] | None:
    if not items:
        return {}
    order = items[:]
    random.shuffle(order)
    needed = sum(span for _, span in order) + len(order) + 1
    if needed > length:
        return None
    slack = length - needed
    extras = [0] * (len(order) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1

    positions = {}
    pos = 1 + extras[0]
    for idx, (key, span) in enumerate(order):
        positions[key] = pos
        pos += span + 1 + extras[idx + 1]
    return positions


def _feature_box(
    base_r0: int,
    base_c0: int,
    base_h: int,
    base_w: int,
    size: tuple[int, int],
    side: str,
    start: int,
    kind: str,
) -> tuple[int, int, int, int]:
    h, w = size
    if kind == "notch":
        if side == "top":
            return base_r0, base_c0 + start, base_r0 + h, base_c0 + start + w
        if side == "bottom":
            return base_r0 + base_h - h, base_c0 + start, base_r0 + base_h, base_c0 + start + w
        if side == "left":
            return base_r0 + start, base_c0, base_r0 + start + h, base_c0 + w
        return base_r0 + start, base_c0 + base_w - w, base_r0 + start + h, base_c0 + base_w

    if side == "top":
        return base_r0 - h, base_c0 + start, base_r0, base_c0 + start + w
    if side == "bottom":
        return base_r0 + base_h, base_c0 + start, base_r0 + base_h + h, base_c0 + start + w
    if side == "left":
        return base_r0 + start, base_c0 - w, base_r0 + start + h, base_c0
    return base_r0 + start, base_c0 + base_w, base_r0 + start + h, base_c0 + base_w + w


def _place_stamp(grid: np.ndarray, size: tuple[int, int], color: int) -> bool:
    h, w = size
    gh, gw = grid.shape
    candidates = []
    for r in range(gh - h + 1):
        for c in range(gw - w + 1):
            rr0 = max(0, r - 1)
            rr1 = min(gh, r + h + 1)
            cc0 = max(0, c - 1)
            cc1 = min(gw, c + w + 1)
            if np.any(grid[rr0:rr1, cc0:cc1] != BG):
                continue
            candidates.append((r, c))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    grid[r0 : r0 + h, c0 : c0 + w] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3))
    sizes = _pick_sizes(pair_count)

    notch_sides = [random.choice(SIDE_NAMES) for _ in range(pair_count)]
    protrusion_sides = [random.choice(SIDE_NAMES) for _ in range(pair_count)]

    base_h_min = max(
        10,
        _required_side_length(sizes, notch_sides, "left"),
        _required_side_length(sizes, notch_sides, "right"),
        _required_side_length(sizes, protrusion_sides, "left"),
        _required_side_length(sizes, protrusion_sides, "right"),
    )
    base_w_min = max(
        10,
        _required_side_length(sizes, notch_sides, "top"),
        _required_side_length(sizes, notch_sides, "bottom"),
        _required_side_length(sizes, protrusion_sides, "top"),
        _required_side_length(sizes, protrusion_sides, "bottom"),
    )
    if base_h_min > 14 or base_w_min > 14:
        return None

    base_h = random.randint(base_h_min, 14)
    base_w = random.randint(base_w_min, 14)

    side_layouts = {}
    for side in SIDE_NAMES:
        items = []
        for idx, size in enumerate(sizes):
            if notch_sides[idx] == side:
                items.append(((idx, "notch"), _span_on_side(size, side)))
            if protrusion_sides[idx] == side:
                items.append(((idx, "protrusion"), _span_on_side(size, side)))
        length = base_w if side in {"top", "bottom"} else base_h
        layout = _layout_side(length, items)
        if layout is None:
            return None
        side_layouts[side] = layout

    top_need = max((size[0] for size, side in zip(sizes, protrusion_sides) if side == "top"), default=0)
    bottom_need = max((size[0] for size, side in zip(sizes, protrusion_sides) if side == "bottom"), default=0)
    left_need = max((size[1] for size, side in zip(sizes, protrusion_sides) if side == "left"), default=0)
    right_need = max((size[1] for size, side in zip(sizes, protrusion_sides) if side == "right"), default=0)

    extra_top = random.randint(4, 5)
    extra_bottom = random.randint(4, 5)
    extra_left = random.randint(4, 5)
    extra_right = random.randint(4, 5)
    h = base_h + top_need + bottom_need + extra_top + extra_bottom
    w = base_w + left_need + right_need + extra_left + extra_right
    if h > 30 or w > 30:
        return None

    base_r0 = top_need + extra_top
    base_c0 = left_need + extra_left
    grid = np.full((h, w), BG, dtype=int)
    grid[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = SHAPE_COLOR

    for idx, size in enumerate(sizes):
        side = notch_sides[idx]
        start = side_layouts[side][(idx, "notch")]
        r0, c0, r1, c1 = _feature_box(base_r0, base_c0, base_h, base_w, size, side, start, "notch")
        grid[r0:r1, c0:c1] = BG

    for idx, size in enumerate(sizes):
        side = protrusion_sides[idx]
        start = side_layouts[side][(idx, "protrusion")]
        r0, c0, r1, c1 = _feature_box(base_r0, base_c0, base_h, base_w, size, side, start, "protrusion")
        grid[r0:r1, c0:c1] = SHAPE_COLOR

    colors = random.sample(STAMP_COLORS, pair_count)
    stamp_order = list(range(pair_count))
    random.shuffle(stamp_order)
    for idx in stamp_order:
        if not _place_stamp(grid, sizes[idx], colors[idx]):
            return None

    if np.count_nonzero(grid == BG) <= np.count_nonzero(grid == SHAPE_COLOR):
        return None

    try:
        comps = _components(grid, exclude={BG})
        main_comp = _largest_non_bg_component(grid, BG)
        if int(main_comp["color"]) != SHAPE_COLOR:
            return None
        main_key = (int(main_comp["color"]), tuple(main_comp["bbox"]), int(main_comp["area"]))
        if any(
            int(comp["area"]) >= int(main_comp["area"])
            for comp in comps
            if (int(comp["color"]), tuple(comp["bbox"]), int(comp["area"])) != main_key
        ):
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
