import random
from typing import Iterable

import numpy as np


DEPTH_TO_COLOR = {
    0: 0,
    1: 5,
    2: 7,
    3: 6,
    4: 8,
}

OUTLINE_COLORS = (1, 2, 3, 4)
MAX_GEN_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the full hollow rectangles hidden by outline-over-outline occlusion, then fill each black region by nesting depth.

    Input structure:
    - The grid is a black background containing 3-4 one-pixel-thick hollow axis-aligned rectangles.
    - Each rectangle uses a distinct outline color, and some of its outline cells may be hidden by another rectangle's outline color.
    - Hidden cells are still foreground pixels, so the union of outlines stays non-black even when an individual rectangle is partially occluded.

    Rule:
    - For each outline color, recover the unique full rectangle whose perimeter contains every visible pixel of that color and lies entirely on non-black cells.
    - Treat the recovered rectangle perimeters as the completed set of blockers, find every maximal 4-connected non-perimeter region, and count how many recovered rectangles strictly enclose that region.
    - Keep the input outline colors unchanged and recolor each non-outline region by depth: 0 stays black, 1 -> gray, 2 -> orange, 3 -> pink, 4 -> teal.

    Key insight:
    - Occlusion order only changes which outline color is visible at shared perimeter cells; the target fill depends instead on reconstructing each rectangle's geometry and counting how many recovered interiors contain each region.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    rectangles = _recover_rectangles(g)
    completed_mask = np.zeros_like(g, dtype=bool)
    for rect in rectangles.values():
        for r, c in _perimeter_cells(rect):
            completed_mask[r, c] = True

    out = g.copy()
    for coords in _components_from_mask(~completed_mask):
        depth = _region_depth(coords, rectangles.values())
        fill_color = DEPTH_TO_COLOR.get(depth)
        if fill_color is None:
            raise ValueError(f"unsupported nesting depth: {depth}")
        if fill_color != 0:
            out[coords[:, 0], coords[:, 1]] = fill_color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        spec = _sample_instance(rng)
        if spec is None:
            continue
        grid, rectangles = spec
        try:
            recovered = _recover_rectangles(grid)
        except ValueError:
            continue
        if recovered != rectangles:
            continue
        if not _has_occlusion(grid, rectangles):
            continue
        if not _has_required_depths(rectangles, grid.shape):
            continue
        expected = _render_output(grid, rectangles)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")


def _sample_instance(rng: random.Random) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]] | None:
    has_third_nested = rng.random() < 0.82
    min_size = 20 if has_third_nested else 17
    h = rng.randint(min_size, 21)
    w = rng.randint(min_size, 21)
    variant = rng.choice(
        (
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
            "left_top",
            "left_bottom",
            "right_top",
            "right_bottom",
        )
    )

    outer = _sample_outer_rect(rng, h, w)
    middle = _sample_middle_rect(rng, outer, variant, has_third_nested)
    if middle is None:
        return None

    rectangles: list[tuple[int, int, int, int]] = [outer, middle]
    if has_third_nested:
        inner = _sample_inner_rect(rng, middle)
        if inner is None:
            return None
        rectangles.append(inner)

    attached = _sample_attached_rect(rng, outer, middle, h, w, variant)
    if attached is None:
        return None
    rectangles.append(attached)

    if len(rectangles) not in (3, 4):
        return None

    if not _rectangles_are_distinct(rectangles):
        return None

    colors = list(rng.sample(OUTLINE_COLORS, len(rectangles)))
    color_map = {colors[i]: rectangles[i] for i in range(len(rectangles))}

    non_attached = colors[:-1]
    rng.shuffle(non_attached)
    order = non_attached + [colors[-1]]

    grid = np.zeros((h, w), dtype=int)
    for color in order:
        _paint_rect(grid, color_map[color], color)

    if np.any((grid != 0) & ~_union_perimeter_mask(color_map.values(), grid.shape)):
        return None

    if not _visible_side_support(grid, color_map):
        return None

    return grid, color_map


def _sample_outer_rect(rng: random.Random, h: int, w: int) -> tuple[int, int, int, int]:
    r0 = rng.randint(2, 3)
    c0 = rng.randint(2, 3)
    r1 = h - 1 - rng.randint(2, 3)
    c1 = w - 1 - rng.randint(2, 3)
    return (r0, c0, r1, c1)


def _sample_middle_rect(
    rng: random.Random,
    outer: tuple[int, int, int, int],
    variant: str,
    need_large_inner_room: bool,
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = outer
    top_gap = 3 + rng.randint(0, 1)
    bottom_gap = 3 + rng.randint(0, 1)
    left_gap = 3 + rng.randint(0, 1)
    right_gap = 3 + rng.randint(0, 1)
    primary, secondary = variant.split("_")
    if primary == "top":
        top_gap += 2
    elif primary == "bottom":
        bottom_gap += 2
    elif primary == "left":
        left_gap += 2
    else:
        right_gap += 2
    if secondary == "top":
        top_gap += 2
    elif secondary == "bottom":
        bottom_gap += 2
    elif secondary == "left":
        left_gap += 2
    else:
        right_gap += 2

    mid = (r0 + top_gap, c0 + left_gap, r1 - bottom_gap, c1 - right_gap)
    min_span = 7 if need_large_inner_room else 5
    if _rect_height(mid) < min_span or _rect_width(mid) < min_span:
        return None
    return mid


def _sample_inner_rect(
    rng: random.Random,
    middle: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = middle
    top_gap = 2 + rng.randint(0, 1)
    bottom_gap = 2 + rng.randint(0, 1)
    left_gap = 2 + rng.randint(0, 1)
    right_gap = 2 + rng.randint(0, 1)
    inner = (r0 + top_gap, c0 + left_gap, r1 - bottom_gap, c1 - right_gap)
    if _rect_height(inner) < 3 or _rect_width(inner) < 3:
        return None
    return inner


def _sample_attached_rect(
    rng: random.Random,
    outer: tuple[int, int, int, int],
    middle: tuple[int, int, int, int],
    h: int,
    w: int,
    variant: str,
) -> tuple[int, int, int, int] | None:
    or0, oc0, or1, oc1 = outer
    mr0, mc0, mr1, mc1 = middle

    if variant == "top_left":
        r0 = or0
        if mr0 - 2 < or0 + 2 or oc0 - 2 < 0 or mc0 - 2 < oc0 + 2:
            return None
        r1 = rng.randint(or0 + 2, mr0 - 2)
        c0 = rng.randint(0, oc0 - 2)
        c1 = rng.randint(oc0 + 2, mc0 - 2)
    elif variant == "top_right":
        r0 = or0
        if mr0 - 2 < or0 + 2 or mc1 + 2 > oc1 - 2 or oc1 + 2 > w - 1:
            return None
        r1 = rng.randint(or0 + 2, mr0 - 2)
        c0 = rng.randint(mc1 + 2, oc1 - 2)
        c1 = rng.randint(oc1 + 2, w - 1)
    elif variant == "bottom_left":
        r1 = or1
        if mr1 + 2 > or1 - 2 or oc0 - 2 < 0 or mc0 - 2 < oc0 + 2:
            return None
        r0 = rng.randint(mr1 + 2, or1 - 2)
        c0 = rng.randint(0, oc0 - 2)
        c1 = rng.randint(oc0 + 2, mc0 - 2)
    elif variant == "bottom_right":
        r1 = or1
        if mr1 + 2 > or1 - 2 or mc1 + 2 > oc1 - 2 or oc1 + 2 > w - 1:
            return None
        r0 = rng.randint(mr1 + 2, or1 - 2)
        c0 = rng.randint(mc1 + 2, oc1 - 2)
        c1 = rng.randint(oc1 + 2, w - 1)
    elif variant == "left_top":
        c0 = oc0
        if mc0 - 2 < oc0 + 2 or or0 - 2 < 0 or mr0 - 2 < or0 + 2:
            return None
        c1 = rng.randint(oc0 + 2, mc0 - 2)
        r0 = rng.randint(0, or0 - 2)
        r1 = rng.randint(or0 + 2, mr0 - 2)
    elif variant == "left_bottom":
        c0 = oc0
        if mc0 - 2 < oc0 + 2 or mr1 + 2 > or1 - 2 or or1 + 2 > h - 1:
            return None
        c1 = rng.randint(oc0 + 2, mc0 - 2)
        r0 = rng.randint(mr1 + 2, or1 - 2)
        r1 = rng.randint(or1 + 2, h - 1)
    else:
        c1 = oc1
        if variant == "right_top":
            if mc1 + 2 > oc1 - 2 or or0 - 2 < 0 or mr0 - 2 < or0 + 2:
                return None
            c0 = rng.randint(mc1 + 2, oc1 - 2)
            r0 = rng.randint(0, or0 - 2)
            r1 = rng.randint(or0 + 2, mr0 - 2)
        else:
            if mc1 + 2 > oc1 - 2 or mr1 + 2 > or1 - 2 or or1 + 2 > h - 1:
                return None
            c0 = rng.randint(mc1 + 2, oc1 - 2)
            r0 = rng.randint(mr1 + 2, or1 - 2)
            r1 = rng.randint(or1 + 2, h - 1)
        c1 = oc1

    rect = (r0, c0, r1, c1)
    if _rect_height(rect) < 3 or _rect_width(rect) < 3:
        return None
    return rect


def _recover_rectangles(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != 0)
    if len(colors) < 3 or len(colors) > 4:
        raise ValueError("expected 3-4 foreground rectangle colors")

    rectangles: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        if coords.size == 0:
            raise ValueError(f"missing visible pixels for color {color}")
        candidates: list[tuple[int, int, int, int]] = []
        min_r = int(coords[:, 0].min())
        max_r = int(coords[:, 0].max())
        min_c = int(coords[:, 1].min())
        max_c = int(coords[:, 1].max())
        for r0 in range(0, min_r + 1):
            for r1 in range(max_r, g.shape[0]):
                if r1 - r0 < 2:
                    continue
                for c0 in range(0, min_c + 1):
                    for c1 in range(max_c, g.shape[1]):
                        if c1 - c0 < 2:
                            continue
                        rect = (r0, c0, r1, c1)
                        if not _coords_lie_on_perimeter(coords, rect):
                            continue
                        if not _perimeter_is_nonblack(g, rect):
                            continue
                        candidates.append(rect)
        unique_candidates = sorted(set(candidates))
        if len(unique_candidates) != 1:
            raise ValueError(f"color {color} does not imply a unique rectangle")
        rectangles[color] = unique_candidates[0]

    mask = _union_perimeter_mask(rectangles.values(), g.shape)
    if np.any((g != 0) & ~mask):
        raise ValueError("found foreground pixels outside recovered rectangles")
    return rectangles


def _render_output(
    grid: np.ndarray,
    rectangles: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    mask = _union_perimeter_mask(rectangles.values(), out.shape)
    for coords in _components_from_mask(~mask):
        depth = _region_depth(coords, rectangles.values())
        color = DEPTH_TO_COLOR[depth]
        if color != 0:
            out[coords[:, 0], coords[:, 1]] = color
    return out


def _region_depth(
    coords: np.ndarray,
    rectangles: Iterable[tuple[int, int, int, int]],
) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    depth = 0
    for rr0, cc0, rr1, cc1 in rectangles:
        if rr0 < r0 and cc0 < c0 and rr1 > r1 and cc1 > c1:
            depth += 1
    return depth


def _visible_side_support(
    grid: np.ndarray,
    rectangles: dict[int, tuple[int, int, int, int]],
) -> bool:
    for color, rect in rectangles.items():
        coords = np.argwhere(grid == color)
        hits = [False, False, False, False]
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr == rect[0] and rect[1] <= cc <= rect[3]:
                hits[0] = True
            if rr == rect[2] and rect[1] <= cc <= rect[3]:
                hits[1] = True
            if cc == rect[1] and rect[0] <= rr <= rect[2]:
                hits[2] = True
            if cc == rect[3] and rect[0] <= rr <= rect[2]:
                hits[3] = True
        if not all(hits):
            return False
    return True


def _has_occlusion(
    grid: np.ndarray,
    rectangles: dict[int, tuple[int, int, int, int]],
) -> bool:
    for color, rect in rectangles.items():
        for r, c in _perimeter_cells(rect):
            if int(grid[r, c]) != int(color):
                return True
    return False


def _has_required_depths(
    rectangles: dict[int, tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> bool:
    mask = _union_perimeter_mask(rectangles.values(), shape)
    present = {_region_depth(coords, rectangles.values()) for coords in _components_from_mask(~mask)}
    return 0 in present and 1 in present and 2 in present


def _rectangles_are_distinct(rectangles: list[tuple[int, int, int, int]]) -> bool:
    return len(set(rectangles)) == len(rectangles)


def _union_perimeter_mask(
    rectangles: Iterable[tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rectangles:
        for r, c in _perimeter_cells(rect):
            mask[r, c] = True
    return mask


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    for r, c in _perimeter_cells(rect):
        grid[r, c] = int(color)


def _perimeter_is_nonblack(grid: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    for r, c in _perimeter_cells(rect):
        if int(grid[r, c]) == 0:
            return False
    return True


def _coords_lie_on_perimeter(coords: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        on_h = rr in (r0, r1) and c0 <= cc <= c1
        on_v = cc in (c0, c1) and r0 <= rr <= r1
        if not (on_h or on_v):
            return False
    return True


def _perimeter_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_height(rect: tuple[int, int, int, int]) -> int:
    return int(rect[2] - rect[0] + 1)


def _rect_width(rect: tuple[int, int, int, int]) -> int:
    return int(rect[3] - rect[1] + 1)
