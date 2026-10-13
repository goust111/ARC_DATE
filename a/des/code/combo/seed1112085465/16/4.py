from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = tuple(color for color in range(10) if color not in {BG, GREY})


TOP_TEMPLATES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (-1, 1)),
)
BOTTOM_TEMPLATES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0)),
)
LEFT_TEMPLATES = (
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, -1)),
)
RIGHT_TEMPLATES = (
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (0, 1)),
)
CONTROL_TEMPLATES = {
    "top": TOP_TEMPLATES,
    "bottom": BOTTOM_TEMPLATES,
    "left": LEFT_TEMPLATES,
    "right": RIGHT_TEMPLATES,
}

BRANCHES = ("vertical_flip", "horizontal_flip", "rotate180")
GEN_ATTEMPTS = 400


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _components_from_cells(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    component.append(nxt)
        components.append(component)
    return components


def _touched_sides(
    component: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int
) -> set[str]:
    sides = set()
    for r, c in component:
        if r + 1 == r0 and c0 <= c <= c1:
            sides.add("top")
        if r - 1 == r1 and c0 <= c <= c1:
            sides.add("bottom")
        if c + 1 == c0 and r0 <= r <= r1:
            sides.add("left")
        if c - 1 == c1 and r0 <= r <= r1:
            sides.add("right")
    return sides


def _find_target(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[str, ...]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    nonblack = g != BG
    total_nonblack = int(nonblack.sum())
    assert total_nonblack >= 6, "expected a target rectangle plus control objects"

    black_prefix = _prefix_sum(g == BG)
    grey_prefix = _prefix_sum(g == GREY)
    nonblack_cells = [tuple(map(int, rc)) for rc in np.argwhere(nonblack)]

    min_area = total_nonblack - 6
    max_area = total_nonblack - 2
    candidates: list[tuple[int, int, int, int, int, tuple[str, ...]]] = []

    for r0 in range(h):
        for r1 in range(r0, h):
            rect_h = r1 - r0 + 1
            min_w = max(1, (min_area + rect_h - 1) // rect_h)
            max_w = min(w, max_area // rect_h)
            if min_w > max_w:
                continue

            for rect_w in range(min_w, max_w + 1):
                area = rect_h * rect_w
                outside = total_nonblack - area
                if outside < 2 or outside > 6:
                    continue

                for c0 in range(0, w - rect_w + 1):
                    c1 = c0 + rect_w - 1
                    if _rect_sum(black_prefix, r0, c0, r1, c1) != 0:
                        continue
                    if _rect_sum(grey_prefix, r0, c0, r1, c1) == 0:
                        continue

                    outside_cells = [
                        (rr, cc)
                        for rr, cc in nonblack_cells
                        if not (r0 <= rr <= r1 and c0 <= cc <= c1)
                    ]
                    if len(outside_cells) != outside:
                        continue

                    components = _components_from_cells(outside_cells)
                    if not (1 <= len(components) <= 2):
                        continue

                    sides: list[str] = []
                    ok = True
                    for comp in components:
                        if len(comp) not in (2, 3):
                            ok = False
                            break
                        if len({int(g[r, c]) for r, c in comp}) != 1:
                            ok = False
                            break
                        touched = _touched_sides(comp, r0, c0, r1, c1)
                        if len(touched) != 1:
                            ok = False
                            break
                        sides.append(next(iter(touched)))

                    if ok:
                        candidates.append((area, r0, c0, r1, c1, tuple(sorted(sides))))

    assert candidates, "failed to locate a valid target rectangle"
    max_area_found = max(area for area, *_ in candidates)
    best = [cand for cand in candidates if cand[0] == max_area_found]
    assert len(best) == 1, "target rectangle must be unique"
    _, r0, c0, r1, c1, sides = best[0]
    return (r0, c0, r1, c1), sides


def _apply_rule(region: np.ndarray, sides: tuple[str, ...]) -> np.ndarray:
    has_horizontal_edge_control = any(side in {"top", "bottom"} for side in sides)
    has_vertical_edge_control = any(side in {"left", "right"} for side in sides)
    assert has_horizontal_edge_control or has_vertical_edge_control

    if has_horizontal_edge_control and has_vertical_edge_control:
        return np.rot90(region, 2)
    if has_horizontal_edge_control:
        return np.flipud(region)
    return np.fliplr(region)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the side-adjacent control objects to choose a symmetry transform for the target rectangle.

    - control on top/bottom -> vertical flip
    - control on left/right -> horizontal flip
    - one horizontal-side control and one vertical-side control -> 180-degree rotation
    - controls are erased in the output
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (r0, c0, r1, c1), sides = _find_target(g)
    region = g[r0 : r1 + 1, c0 : c1 + 1]

    out = np.zeros_like(g)
    out[r0 : r1 + 1, c0 : c1 + 1] = _apply_rule(region, sides)
    return out


def _branch_sides(branch: str) -> list[str]:
    if branch == "vertical_flip":
        return [random.choice(["top", "bottom"])]
    if branch == "horizontal_flip":
        return [random.choice(["left", "right"])]
    return [random.choice(["top", "bottom"]), random.choice(["left", "right"])]


def _random_patch(region: np.ndarray, color: int) -> None:
    h, w = region.shape
    mode = random.choice(("pixel", "hseg", "vseg", "block"))

    if mode == "pixel":
        r = random.randint(1, h - 2)
        c = random.randint(1, w - 2)
        region[r, c] = color
        return

    if mode == "hseg":
        seg_w = random.randint(2, w - 2)
        r = random.randint(1, h - 2)
        c0 = random.randint(1, w - 1 - seg_w)
        region[r, c0 : c0 + seg_w] = color
        return

    if mode == "vseg":
        seg_h = random.randint(2, h - 2)
        r0 = random.randint(1, h - 1 - seg_h)
        c = random.randint(1, w - 2)
        region[r0 : r0 + seg_h, c] = color
        return

    block_h = random.randint(1, min(3, h - 2))
    block_w = random.randint(1, min(3, w - 2))
    r0 = random.randint(1, h - 1 - block_h)
    c0 = random.randint(1, w - 1 - block_w)
    region[r0 : r0 + block_h, c0 : c0 + block_w] = color


def _make_region(rect_h: int, rect_w: int) -> np.ndarray:
    for _ in range(200):
        region = np.full((rect_h, rect_w), GREY, dtype=int)
        color_count = random.randint(2, min(4, len(PALETTE)))
        colors = random.sample(PALETTE, color_count)
        patch_count = random.randint(color_count + 1, color_count + 4)

        for index in range(patch_count):
            color = colors[index] if index < color_count else random.choice(colors)
            _random_patch(region, color)

        non_grey = region != GREY
        if int(non_grey.sum()) < 4:
            continue
        if len(set(map(int, region[non_grey]))) < 2:
            continue
        if np.array_equal(region, np.flipud(region)):
            continue
        if np.array_equal(region, np.fliplr(region)):
            continue
        if np.array_equal(region, np.rot90(region, 2)):
            continue
        return region

    raise RuntimeError("failed to generate an asymmetric target region")


def _margin_requirements(side: str, template: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    drs = [dr for dr, _ in template]
    dcs = [dc for _, dc in template]

    top = bottom = left = right = 1
    if side == "top":
        top = max(top, 1 - min(drs))
    elif side == "bottom":
        bottom = max(bottom, 1 + max(drs))
    elif side == "left":
        left = max(left, 1 - min(dcs))
    else:
        right = max(right, 1 + max(dcs))
    return top, bottom, left, right


def _place_control(
    grid: np.ndarray,
    side: str,
    template: tuple[tuple[int, int], ...],
    bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    drs = [dr for dr, _ in template]
    dcs = [dc for _, dc in template]

    if side == "top":
        anchor_r = r0 - 1
        lo = c0 + 1 - min(dcs)
        hi = c1 - 1 - max(dcs)
        anchor_c = random.randint(lo, hi)
    elif side == "bottom":
        anchor_r = r1 + 1
        lo = c0 + 1 - min(dcs)
        hi = c1 - 1 - max(dcs)
        anchor_c = random.randint(lo, hi)
    elif side == "left":
        anchor_c = c0 - 1
        lo = r0 + 1 - min(drs)
        hi = r1 - 1 - max(drs)
        anchor_r = random.randint(lo, hi)
    else:
        anchor_c = c1 + 1
        lo = r0 + 1 - min(drs)
        hi = r1 - 1 - max(drs)
        anchor_r = random.randint(lo, hi)

    coords = [(anchor_r + dr, anchor_c + dc) for dr, dc in template]
    for rr, cc in coords:
        assert 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
        assert grid[rr, cc] == BG
    for rr, cc in coords:
        grid[rr, cc] = color


def _apply_branch(region: np.ndarray, branch: str) -> np.ndarray:
    if branch == "vertical_flip":
        return np.flipud(region)
    if branch == "horizontal_flip":
        return np.fliplr(region)
    return np.rot90(region, 2)


def _gen_attempt() -> np.ndarray | None:
    branch = random.choice(BRANCHES)
    sides = _branch_sides(branch)
    templates = [random.choice(CONTROL_TEMPLATES[side]) for side in sides]

    rect_h = random.randint(5, 9)
    rect_w = random.randint(5, 9)
    region = _make_region(rect_h, rect_w)

    top_req = bottom_req = left_req = right_req = 1
    for side, template in zip(sides, templates):
        add_top, add_bottom, add_left, add_right = _margin_requirements(side, template)
        top_req = max(top_req, add_top)
        bottom_req = max(bottom_req, add_bottom)
        left_req = max(left_req, add_left)
        right_req = max(right_req, add_right)

    top_margin = top_req + random.randint(0, 3)
    bottom_margin = bottom_req + random.randint(0, 3)
    left_margin = left_req + random.randint(0, 3)
    right_margin = right_req + random.randint(0, 3)

    height = top_margin + rect_h + bottom_margin
    width = left_margin + rect_w + right_margin
    grid = np.zeros((height, width), dtype=int)

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = region

    region_colors = {int(color) for color in np.unique(region) if int(color) != GREY}
    available = [color for color in PALETTE if color not in region_colors]
    if len(available) < len(sides):
        available = list(PALETTE)
    control_colors = random.sample(available, len(sides))

    for side, template, color in zip(sides, templates, control_colors):
        _place_control(grid, side, template, (r0, c0, r1, c1), color)

    expected = np.zeros_like(grid)
    expected[r0 : r1 + 1, c0 : c1 + 1] = _apply_branch(region, branch)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
