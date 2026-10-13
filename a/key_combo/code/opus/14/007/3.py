from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 7
YELLOW = 4
DIAMOND_COLORS = (1, 2, 3, 6)
RADII = (2, 3, 4, 5)
GRID_MIN = 17
GRID_MAX = 20
GEN_ATTEMPTS = 500
EXTRA_CENTER_TRIES = 160


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _outline_mask(shape: tuple[int, int], cr: int, cc: int, radius: int) -> np.ndarray:
    rr = np.arange(shape[0])[:, None]
    cc_grid = np.arange(shape[1])[None, :]
    return np.abs(rr - cr) + np.abs(cc_grid - cc) == radius


def _interior_mask(shape: tuple[int, int], cr: int, cc: int, radius: int) -> np.ndarray:
    rr = np.arange(shape[0])[:, None]
    cc_grid = np.arange(shape[1])[None, :]
    return np.abs(rr - cr) + np.abs(cc_grid - cc) < radius


def _mask_is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c):
            if (
                0 <= nr < mask.shape[0]
                and 0 <= nc < mask.shape[1]
                and bool(mask[nr, nc])
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _detect_diamond(mask: np.ndarray, color: int) -> dict:
    coords = np.argwhere(mask)
    assert coords.size > 0, "each foreground color must appear"
    assert _mask_is_single_component(mask), "each diamond color must form one outline component"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height == width and height % 2 == 1, "diamond bbox must be an odd square"

    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    radius = (height - 1) // 2
    assert radius >= 2, "expected input radii in [2, 5]"

    expected = _outline_mask(mask.shape, cr, cc, radius)
    assert np.array_equal(mask, expected), "foreground component must be a 1px Manhattan-diamond outline"
    return {"color": int(color), "cr": cr, "cc": cc, "radius": radius}


def _render_output_from_diamonds(shape: tuple[int, int], diamonds: list[dict]) -> np.ndarray:
    outline_mask = np.zeros(shape, dtype=bool)
    outline_colors = np.full(shape, BG, dtype=int)

    for diamond in sorted(diamonds, key=lambda item: item["radius"]):
        claim = _outline_mask(shape, diamond["cr"], diamond["cc"], diamond["radius"])
        claim |= _outline_mask(shape, diamond["cr"], diamond["cc"], diamond["radius"] + 1)
        claim &= ~outline_mask
        outline_colors[claim] = diamond["color"]
        outline_mask |= claim

    out = np.full(shape, BG, dtype=int)
    out[outline_mask] = outline_colors[outline_mask]

    for diamond in sorted(diamonds, key=lambda item: item["radius"], reverse=True):
        fill_region = _interior_mask(shape, diamond["cr"], diamond["cc"], diamond["radius"])
        fill_region &= ~outline_mask
        fill_color = YELLOW if int(fill_region.sum()) % 2 == 0 else diamond["color"]
        out[fill_region] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Dilate every diamond outline by one layer, resolve overlap by smaller original radius, then parity-fill the preserved inner regions.

    Input structure:
    - An orange background contains 3-4 one-cell-thick Manhattan-diamond outlines.
    - Each outline uses a distinct color from blue, red, green, and magenta.
    - Diamonds may be nested, but their original outline pixels do not overlap.

    Rule:
    - Recover each diamond's center and original radius from its outline.
    - Keep the original outline and also draw the radius+1 outline in the same color.
    - When multiple post-dilation outline claims hit one cell, keep the color of the diamond with the smaller original radius.
    - For each diamond, inspect the cells strictly inside its original outline, exclude all final outline pixels, and fill the remaining cells yellow when that usable interior area is even, otherwise with the diamond's own color.
    - Apply fills from larger diamonds to smaller diamonds so contained small diamonds overwrite enclosing fills.

    Key insight:
    - The final outline depends on original-radius priority, while the fill parity depends on the usable interior left after all dilated outlines are resolved, so both the pre-dilation radius and the post-dilation geometry must be tracked together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert 3 <= len(colors) <= 4, "expected 3-4 colored diamonds"
    assert len(colors) == len(set(colors)), "diamond colors must be distinct"

    diamonds = [_detect_diamond(g == color, color) for color in colors]
    radii = [diamond["radius"] for diamond in diamonds]
    assert len(set(radii)) == len(radii), "original radii must be unique"

    return _render_output_from_diamonds(g.shape, diamonds)


def _fits_with_dilation(shape: tuple[int, int], cr: int, cc: int, radius: int) -> bool:
    return (
        radius + 1 <= cr < shape[0] - (radius + 1)
        and radius + 1 <= cc < shape[1] - (radius + 1)
    )


def _outline_points(shape: tuple[int, int], cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    return set(map(tuple, np.argwhere(_outline_mask(shape, cr, cc, radius))))


def _expanded_points(shape: tuple[int, int], cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    pts = _outline_points(shape, cr, cc, radius)
    pts |= _outline_points(shape, cr, cc, radius + 1)
    return pts


def _nested(inner: dict, outer: dict) -> bool:
    center_dist = abs(inner["cr"] - outer["cr"]) + abs(inner["cc"] - outer["cc"])
    return center_dist + inner["radius"] < outer["radius"]


def _render_input(shape: tuple[int, int], diamonds: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for diamond in diamonds:
        grid[_outline_mask(shape, diamond["cr"], diamond["cc"], diamond["radius"])] = diamond["color"]
    return grid


def _try_place_extra(
    shape: tuple[int, int],
    radius: int,
    placed: list[dict],
    base_center: tuple[int, int],
) -> dict | None:
    modes = ["same", "near", "free", "near"]
    for _ in range(EXTRA_CENTER_TRIES):
        mode = random.choice(modes)
        if mode == "same":
            cr, cc = base_center
        elif mode == "near":
            max_offset = max(2, radius + 1)
            cr = base_center[0] + random.randint(-max_offset, max_offset)
            cc = base_center[1] + random.randint(-max_offset, max_offset)
        else:
            cr = random.randint(radius + 1, shape[0] - radius - 2)
            cc = random.randint(radius + 1, shape[1] - radius - 2)

        if not _fits_with_dilation(shape, cr, cc, radius):
            continue

        candidate = {"cr": cr, "cc": cc, "radius": radius}
        candidate_outline = _outline_points(shape, cr, cc, radius)
        if any(candidate_outline & _outline_points(shape, item["cr"], item["cc"], item["radius"]) for item in placed):
            continue

        if mode != "free":
            interacts = False
            cand_expanded = _expanded_points(shape, cr, cc, radius)
            for item in placed:
                if cand_expanded & _expanded_points(shape, item["cr"], item["cc"], item["radius"]):
                    interacts = True
                    break
                inner, outer = (candidate, item) if radius < item["radius"] else (item, candidate)
                if _nested(inner, outer):
                    interacts = True
                    break
            if not interacts:
                continue

        return candidate
    return None


def _gen_layout() -> tuple[tuple[int, int], list[dict]] | None:
    shape = (
        random.randint(GRID_MIN, GRID_MAX),
        random.randint(GRID_MIN, GRID_MAX),
    )
    count = random.choice((3, 4))
    radii = sorted(random.sample(RADII, count))
    colors = random.sample(DIAMOND_COLORS, count)

    adjacent_pairs = [(a, b) for a, b in zip(radii, radii[1:]) if b == a + 1]
    if not adjacent_pairs:
        return None

    inner_radius, outer_radius = random.choice(adjacent_pairs)
    center_margin = outer_radius + 1
    base_center = (
        random.randint(center_margin, shape[0] - center_margin - 1),
        random.randint(center_margin, shape[1] - center_margin - 1),
    )
    diamonds = [
        {"color": colors[0], "cr": base_center[0], "cc": base_center[1], "radius": inner_radius},
        {"color": colors[1], "cr": base_center[0], "cc": base_center[1], "radius": outer_radius},
    ]

    used_radii = {inner_radius, outer_radius}
    color_index = 2
    for radius in radii:
        if radius in used_radii:
            continue
        candidate = _try_place_extra(shape, radius, diamonds, base_center)
        if candidate is None:
            return None
        candidate["color"] = colors[color_index]
        diamonds.append(candidate)
        color_index += 1

    return shape, diamonds


def _valid_layout(shape: tuple[int, int], diamonds: list[dict]) -> bool:
    outlines = [
        _outline_points(shape, diamond["cr"], diamond["cc"], diamond["radius"])
        for diamond in diamonds
    ]
    for i in range(len(outlines)):
        for j in range(i + 1, len(outlines)):
            if outlines[i] & outlines[j]:
                return False

    has_nested = False
    has_expanded_overlap = False
    for i, a in enumerate(diamonds):
        expanded_a = _expanded_points(shape, a["cr"], a["cc"], a["radius"])
        for j, b in enumerate(diamonds):
            if i >= j:
                continue
            expanded_b = _expanded_points(shape, b["cr"], b["cc"], b["radius"])
            if expanded_a & expanded_b:
                has_expanded_overlap = True
            small, large = (a, b) if a["radius"] < b["radius"] else (b, a)
            if _nested(small, large):
                has_nested = True
    if not (has_nested and has_expanded_overlap):
        return False

    inp = _render_input(shape, diamonds)
    out = main(inp)
    if np.array_equal(inp, out):
        return False

    non_outline = out[(inp == BG)]
    if YELLOW not in non_outline:
        return False
    if not any(int(color) in DIAMOND_COLORS for color in non_outline):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        layout = _gen_layout()
        if layout is None:
            continue
        shape, diamonds = layout
        if max(shape) > 30:
            continue
        if not _valid_layout(shape, diamonds):
            continue
        return _render_input(shape, diamonds)
    raise RuntimeError("failed to generate a valid input for puzzle 3")
