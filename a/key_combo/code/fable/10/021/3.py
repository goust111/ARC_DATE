from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BACKGROUND = 0
PALETTE = list(range(1, 10))
MAX_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_contains(bbox: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int, int, int]] = []

    for r in range(h):
        holes = np.flatnonzero(grid[r] == bg)
        if len(holes) != 1:
            continue
        vals = grid[r][grid[r] != bg]
        if len(vals) == w - 1 and len(set(map(int, vals))) == 1:
            candidates.append(("h", r, r, int(holes[0]), int(vals[0])))

    for c in range(w):
        holes = np.flatnonzero(grid[:, c] == bg)
        if len(holes) != 1:
            continue
        vals = grid[:, c][grid[:, c] != bg]
        if len(vals) == h - 1 and len(set(map(int, vals))) == 1:
            candidates.append(("v", c, int(holes[0]), c, int(vals[0])))

    assert len(candidates) == 1, "expected exactly one separator line with one pinhole"
    return candidates[0]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_screen(
    grid: np.ndarray, bg: int, sep_color: int, orient: str, axis: int
) -> tuple[int, BBox, np.ndarray]:
    foreground = (grid != bg) & (grid != sep_color)
    candidates: list[tuple[int, int, BBox, np.ndarray]] = []

    for color in _colors_present(grid, exclude={bg, sep_color}):
        for coords in _components_of_color(grid, color):
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area != len(coords) or r0 == r1 or c0 == c1:
                continue

            component_mask = np.zeros_like(grid, dtype=bool)
            component_mask[coords[:, 0], coords[:, 1]] = True
            others = foreground & ~component_mask
            other_coords = np.argwhere(others)
            if len(other_coords) < 2:
                continue

            if orient == "h":
                if not (r1 <= axis - 2 or r0 >= axis + 2):
                    continue
                if r1 < axis:
                    if not np.all(other_coords[:, 0] > axis):
                        continue
                else:
                    if not np.all(other_coords[:, 0] < axis):
                        continue
            else:
                if not (c1 <= axis - 2 or c0 >= axis + 2):
                    continue
                if c1 < axis:
                    if not np.all(other_coords[:, 1] > axis):
                        continue
                else:
                    if not np.all(other_coords[:, 1] < axis):
                        continue

            candidates.append((area, int(color), bbox, component_mask))

    assert len(candidates) == 1, "expected exactly one monochrome screen rectangle"
    _area, color, bbox, component_mask = candidates[0]
    return color, bbox, component_mask


def _project_through_hole(hole_r: int, hole_c: int, r: int, c: int) -> Coord:
    return 2 * hole_r - r, 2 * hole_c - c


def _layout_from_input(grid: np.ndarray) -> tuple[str, int, Coord, int, int, BBox, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    orient, axis, hole_r, hole_c, sep_color = _find_separator(g, bg)
    screen_color, screen_bbox, screen_mask = _find_screen(g, bg, sep_color, orient, axis)

    source_mask = (g != bg) & (g != sep_color) & ~screen_mask
    source_coords = np.argwhere(source_mask)
    assert len(source_coords) >= 2, "expected a non-empty multicolor source pattern"
    source_colors = set(map(int, g[source_mask]))
    assert len(source_colors) >= 2, "source pattern must be multicolor"

    r0, c0, r1, c1 = screen_bbox
    if orient == "h":
        assert r1 < axis or r0 > axis, "screen must lie on one side of the separator"
        assert r1 <= axis - 2 or r0 >= axis + 2, "screen must not touch the separator"
        if r1 < axis:
            assert np.all(source_coords[:, 0] > axis), "source must lie on the opposite side"
        else:
            assert np.all(source_coords[:, 0] < axis), "source must lie on the opposite side"
    else:
        assert c1 < axis or c0 > axis, "screen must lie on one side of the separator"
        assert c1 <= axis - 2 or c0 >= axis + 2, "screen must not touch the separator"
        if c1 < axis:
            assert np.all(source_coords[:, 1] > axis), "source must lie on the opposite side"
        else:
            assert np.all(source_coords[:, 1] < axis), "source must lie on the opposite side"

    return orient, axis, (hole_r, hole_c), sep_color, screen_color, screen_bbox, source_coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the source pattern through the separator's single pinhole and keep only the inverted image inside the screen.

    Input structure:
    - One full row or column separator contains exactly one background-cell gap, the pinhole.
    - A multicolor source pattern lies entirely on one side of that separator.
    - A solid monochrome rectangular screen lies on the opposite side, with at least one-cell gap from the separator.

    Rule:
    - Use the pinhole cell as the center of point symmetry.
    - For each source pixel, map its position to the point-symmetric cell on the other side while keeping the color unchanged.
    - Paint only mapped pixels that land inside the screen rectangle; discard mappings outside the screen or outside the grid, and leave the source, separator, pinhole, and remaining screen cells unchanged.

    Key insight:
    - This is not reflection across the separator line: the image is a 180-degree turn around the pinhole, and the screen acts as the clipping window.
    """
    g = np.asarray(grid, dtype=int)
    orient, _axis, hole, _sep_color, _screen_color, screen_bbox, source_coords = _layout_from_input(g)
    hole_r, hole_c = hole

    out = g.copy()
    for r, c in source_coords:
        rr, cc = _project_through_hole(hole_r, hole_c, int(r), int(c))
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and _bbox_contains(screen_bbox, rr, cc):
            out[rr, cc] = int(g[r, c])

    if orient == "h":
        assert hole_r == _axis
    else:
        assert hole_c == _axis
    return out


def _frontier(mask: np.ndarray) -> list[Coord]:
    h, w = mask.shape
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r + 1 < h and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c + 1 < w and mask[r, c + 1])
            ):
                cells.append((r, c))
    return cells


def _grow_connected_mask(h: int, w: int, count: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < count:
        mask[random.choice(_frontier(mask))] = True
    return mask


def _sample_source_mask(h: int, w: int, count: int, orient: str) -> np.ndarray:
    for _ in range(200):
        if random.random() < 0.7 or h < 4 or w < 4:
            mask = _grow_connected_mask(h, w, count)
        else:
            if random.random() < 0.5 and w >= 5:
                split = random.randint(2, w - 2)
                left_w = split - 1
                right_w = w - split
                if left_w < 2 or right_w < 2:
                    continue
                left_count = random.randint(2, count - 2)
                right_count = count - left_count
                if left_count > h * left_w or right_count > h * right_w:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                mask[:, :left_w] = _grow_connected_mask(h, left_w, left_count)
                mask[:, split:] = _grow_connected_mask(h, right_w, right_count)
            elif h >= 5:
                split = random.randint(2, h - 2)
                top_h = split - 1
                bottom_h = h - split
                if top_h < 2 or bottom_h < 2:
                    continue
                top_count = random.randint(2, count - 2)
                bottom_count = count - top_count
                if top_count > top_h * w or bottom_count > bottom_h * w:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                mask[:top_h, :] = _grow_connected_mask(top_h, w, top_count)
                mask[split:, :] = _grow_connected_mask(bottom_h, w, bottom_count)
            else:
                continue

        coords = np.argwhere(mask)
        bbox = _bbox_from_coords(coords)
        if bbox == (0, 0, h - 1, w - 1) and count == h * w:
            continue
        if orient == "h" and len(set(map(int, coords[:, 1]))) < 2:
            continue
        if orient == "v" and len(set(map(int, coords[:, 0]))) < 2:
            continue
        return mask
    raise RuntimeError("failed to sample a valid source mask")


def _inside_rect(coords: list[Coord], bbox: BBox) -> list[Coord]:
    return [(r, c) for r, c in coords if _bbox_contains(bbox, r, c)]


def _axis_mirror_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    orient, axis, hole, _sep_color, _screen_color, screen_bbox, source_coords = _layout_from_input(g)
    hole_r, hole_c = hole
    out = g.copy()
    for r, c in source_coords:
        if orient == "h":
            rr, cc = 2 * axis - int(r), int(c)
        else:
            rr, cc = int(r), 2 * axis - int(c)
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and _bbox_contains(screen_bbox, rr, cc):
            out[rr, cc] = int(g[r, c])
    assert (hole_r == axis) if orient == "h" else (hole_c == axis)
    return out


def _candidate_screen_boxes(
    shape: tuple[int, int],
    orient: str,
    axis: int,
    screen_side: int,
    seeds: list[Coord],
) -> list[BBox]:
    h, w = shape
    boxes: list[BBox] = []
    for _ in range(180):
        sample_size = random.randint(3, len(seeds))
        subset = random.sample(seeds, sample_size)
        rr = [r for r, _c in subset]
        cc = [c for _r, c in subset]
        r0, r1 = min(rr), max(rr)
        c0, c1 = min(cc), max(cc)

        top_pad = random.randint(0, 1)
        bottom_pad = random.randint(0, 1)
        left_pad = random.randint(0, 2)
        right_pad = random.randint(0, 2)

        r0 = max(0, r0 - top_pad)
        c0 = max(0, c0 - left_pad)
        r1 = min(h - 1, r1 + bottom_pad)
        c1 = min(w - 1, c1 + right_pad)

        if orient == "h":
            if screen_side < 0:
                if r1 >= axis - 1:
                    continue
            else:
                if r0 <= axis + 1:
                    continue
        else:
            if screen_side < 0:
                if c1 >= axis - 1:
                    continue
            else:
                if c0 <= axis + 1:
                    continue

        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        boxes.append((r0, c0, r1, c1))
    return boxes


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    orient = random.choice(["h", "v"])
    sep_color, screen_color = random.sample(PALETTE, 2)
    source_palette = random.sample([c for c in PALETTE if c not in {sep_color, screen_color}], random.randint(2, 4))

    if orient == "h":
        axis = random.randint(4, h - 5)
        hole = (axis, random.randint(2, w - 3))
        source_side = random.choice([-1, 1])
        screen_side = -source_side
        row_lo = 0 if source_side < 0 else axis + 2
        row_hi = axis - 2 if source_side < 0 else h - 1
        if row_hi - row_lo + 1 < 4:
            return None
        src_h = random.randint(3, min(6, row_hi - row_lo + 1))
        src_w = random.randint(3, min(6, w - 2))
        src_r0 = random.randint(row_lo, row_hi - src_h + 1)
        src_c0 = random.randint(1, w - src_w - 1)
    else:
        axis = random.randint(4, w - 5)
        hole = (random.randint(2, h - 3), axis)
        source_side = random.choice([-1, 1])
        screen_side = -source_side
        col_lo = 0 if source_side < 0 else axis + 2
        col_hi = axis - 2 if source_side < 0 else w - 1
        if col_hi - col_lo + 1 < 4:
            return None
        src_h = random.randint(3, min(6, h - 2))
        src_w = random.randint(3, min(6, col_hi - col_lo + 1))
        src_r0 = random.randint(1, h - src_h - 1)
        src_c0 = random.randint(col_lo, col_hi - src_w + 1)

    count = random.randint(5, min(10, src_h * src_w - 1))
    source_mask = _sample_source_mask(src_h, src_w, count, orient)
    local_coords = [tuple(map(int, rc)) for rc in np.argwhere(source_mask)]
    source_coords = [(src_r0 + r, src_c0 + c) for r, c in local_coords]

    if orient == "h" and not any(c != hole[1] for _r, c in source_coords):
        return None
    if orient == "v" and not any(r != hole[0] for r, _c in source_coords):
        return None

    projections = [_project_through_hole(hole[0], hole[1], r, c) for r, c in source_coords]
    in_bounds = [(r, c) for r, c in projections if 0 <= r < h and 0 <= c < w]
    if len(in_bounds) < 4:
        return None

    boxes = _candidate_screen_boxes((h, w), orient, axis, screen_side, in_bounds)
    random.shuffle(boxes)
    screen_bbox = None
    retained_targets: list[Coord] = []
    for candidate in boxes:
        retained = _inside_rect(in_bounds, candidate)
        if len(retained) < 3 or len(retained) >= len(source_coords):
            continue
        if len(retained) == len(in_bounds):
            continue
        area = (candidate[2] - candidate[0] + 1) * (candidate[3] - candidate[1] + 1)
        if area <= len(retained):
            continue
        screen_bbox = candidate
        retained_targets = retained
        break
    if screen_bbox is None:
        return None

    coords_by_target = {target: src for src, target in zip(source_coords, projections)}
    source_colors: dict[Coord, int] = {}
    shuffled = source_coords[:]
    random.shuffle(shuffled)
    for color, coord in zip(source_palette, shuffled):
        source_colors[coord] = color
    for coord in shuffled[len(source_palette) :]:
        source_colors[coord] = random.choice(source_palette)

    inp = np.full((h, w), BACKGROUND, dtype=int)
    if orient == "h":
        inp[axis, :] = sep_color
    else:
        inp[:, axis] = sep_color
    inp[hole] = BACKGROUND

    r0, c0, r1, c1 = screen_bbox
    inp[r0 : r1 + 1, c0 : c1 + 1] = screen_color
    for coord, color in source_colors.items():
        inp[coord] = color

    out = inp.copy()
    for target in retained_targets:
        out[target] = source_colors[coords_by_target[target]]

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_axis_mirror_output(inp), out):
        return None

    uncropped = inp.copy()
    for coord, color in source_colors.items():
        rr, cc = _project_through_hole(hole[0], hole[1], coord[0], coord[1])
        if 0 <= rr < h and 0 <= cc < w:
            uncropped[rr, cc] = color
    if np.array_equal(uncropped, out):
        return None

    source_color_counts = Counter(source_colors.values())
    if len(source_color_counts) < 2:
        return None
    for color in source_palette:
        coords = np.argwhere(inp == color)
        if len(coords) == 0:
            continue
        bbox = _bbox_from_coords(coords)
        area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        if area == len(coords) and len(coords) >= 4:
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, _out = built
        return inp
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
