from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GRAY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
WALL_GAP = 2
LEFT_STRIP = 2
TOP_MARGIN = 1
RIGHT_MARGIN = 1
BOTTOM_MARGIN = 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                out[rr, cc] = True
                break
    return out


def _enclosed_cells(wall: np.ndarray) -> np.ndarray:
    h, w = wall.shape
    passable = ~wall
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return passable & (~outside)


def _scaled_outline(mask: np.ndarray, scale: int) -> np.ndarray:
    scaled = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
    return _boundary(scaled)


def _center_offset(length: int, scale: int) -> int:
    return -((length * (scale - 1)) // 2)


def _legend_scales(grid: np.ndarray) -> dict[int, int]:
    colors = [int(grid[r, 0]) for r in range(grid.shape[0]) if int(grid[r, 0]) not in (BLACK, GRAY)]
    assert colors, "expected a non-empty left-column legend"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return {color: idx + 2 for idx, color in enumerate(colors)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each grey fenced region with a centered enlarged outline keyed by the seed color legend.

    Input structure:
    - The background is black.
    - The leftmost column contains a contiguous-looking color legend whose row order encodes scale factors 2, 3, 4, ...; these colors never equal black or grey.
    - Away from the legend, grey single-pixel walls form separate closed loops, and each loop encloses exactly one non-grey seed pixel.

    Rule:
    - Read the left-column legend from top to bottom to build a color -> scale mapping.
    - For each grey loop, find its enclosed seed color, recover the full fenced region shape (wall plus interior), and scale that solid shape by the mapped factor.
    - Keep only the outer boundary of the scaled solid shape, color it with the seed color, and place it centered on the original fenced region.
    - Clear the original grey walls, the seeds, and the left-column legend to black.

    Key insight:
    The seed color has two roles at once: it is both the drawing color and the lookup key that selects how much the enclosing region's silhouette should be enlarged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    scale_by_color = _legend_scales(g)

    out = np.full((h, w), BLACK, dtype=int)
    for comp in _components4(g == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        wall = sub == GRAY
        interior = _enclosed_cells(wall)
        seed_pts = np.argwhere(interior & (sub != BLACK) & (sub != GRAY))
        assert seed_pts.shape[0] == 1, "each fenced region must contain exactly one seed"
        seed_r, seed_c = map(int, seed_pts[0])
        seed_color = int(sub[seed_r, seed_c])
        assert seed_color in scale_by_color, "seed color must appear in the legend"

        solid = wall | interior
        outline = _scaled_outline(solid, scale_by_color[seed_color])
        dst_r0 = r0 + _center_offset(solid.shape[0], scale_by_color[seed_color])
        dst_c0 = c0 + _center_offset(solid.shape[1], scale_by_color[seed_color])
        for rr, cc in np.argwhere(outline):
            gr = dst_r0 + int(rr)
            gc = dst_c0 + int(cc)
            if 0 <= gr < h and 0 <= gc < w:
                out[gr, gc] = seed_color

    return out


def _region_from_rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _region_from_l(h: int, w: int, corner: str, cut_h: int, cut_w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if corner == "TL":
        mask[:cut_h, :cut_w] = False
    elif corner == "TR":
        mask[:cut_h, w - cut_w :] = False
    elif corner == "BL":
        mask[h - cut_h :, :cut_w] = False
    else:
        mask[h - cut_h :, w - cut_w :] = False
    return mask


def _region_is_valid(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or not mask.any():
        return False
    if len(_components4(mask)) != 1:
        return False
    wall = _boundary(mask)
    if len(_components4(wall)) != 1:
        return False
    interior = mask & (~wall)
    if not interior.any():
        return False
    return np.array_equal(wall | _enclosed_cells(wall), mask)


def _sample_region(scale: int) -> tuple[str, np.ndarray]:
    shape_kind = random.choice(("rect", "l"))
    max_side = {2: 6, 3: 5, 4: 4, 5: 4}.get(scale, 4)
    dims = [n for n in range(3, max_side + 1) if scale % 2 == 1 or n % 2 == 0]

    for _ in range(200):
        if shape_kind == "rect" or len(dims) < 2 or max(dims) < 4:
            h = random.choice(dims)
            w = random.choice(dims)
            mask = _region_from_rect(h, w)
            if _region_is_valid(mask):
                return "rect", mask
            continue

        h = random.choice([n for n in dims if n >= 4])
        w = random.choice([n for n in dims if n >= 4])
        cut_h = random.randint(1, h - 2)
        cut_w = random.randint(1, w - 2)
        arm_h = h - cut_h
        arm_w = w - cut_w
        if arm_h < 2 or arm_w < 2:
            continue
        corner = random.choice(("TL", "TR", "BL", "BR"))
        mask = _region_from_l(h, w, corner, cut_h, cut_w)
        if _region_is_valid(mask):
            return "l", mask

    h = random.choice(dims)
    w = random.choice(dims)
    return "rect", _region_from_rect(h, w)


def _render_expected(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        outline = spec["outline"]
        r0 = int(spec["out_r0"])
        c0 = int(spec["out_c0"])
        color = int(spec["color"])
        hh, ww = outline.shape
        patch = out[r0 : r0 + hh, c0 : c0 + ww]
        patch[outline] = color
    return out


def _pick_layout(boxes: list[tuple[int, int]]) -> tuple[list[list[int]], int, int] | None:
    groups_by_count = {
        1: [[[0]]],
        2: [[[0, 1]], [[0], [1]]],
        3: [[[0, 1, 2]], [[0, 1], [2]], [[0], [1, 2]], [[0], [1], [2]]],
    }
    layouts = groups_by_count[len(boxes)]
    random.shuffle(layouts)

    feasible: list[tuple[list[list[int]], int, int]] = []
    for groups in layouts:
        row_widths = []
        row_heights = []
        for row in groups:
            widths = [boxes[idx][1] for idx in row]
            heights = [boxes[idx][0] for idx in row]
            row_widths.append(sum(widths) + WALL_GAP * (len(row) - 1))
            row_heights.append(max(heights))
        layout_w = max(row_widths)
        layout_h = sum(row_heights) + WALL_GAP * (len(groups) - 1)
        need_w = LEFT_STRIP + layout_w + RIGHT_MARGIN
        need_h = TOP_MARGIN + layout_h + BOTTOM_MARGIN
        if need_w <= 30 and need_h <= 30:
            feasible.append((groups, layout_h, layout_w))

    return random.choice(feasible) if feasible else None


def _place_specs(
    h: int,
    w: int,
    specs: list[dict[str, object]],
    layout: tuple[list[list[int]], int, int],
) -> None:
    groups, layout_h, _layout_w = layout
    row_heights = [max(specs[idx]["out_h"] for idx in row) for row in groups]
    row_widths = [sum(specs[idx]["out_w"] for idx in row) + WALL_GAP * (len(row) - 1) for row in groups]

    total_slack_h = h - (TOP_MARGIN + layout_h + BOTTOM_MARGIN)
    row_cursor = TOP_MARGIN + random.randint(0, total_slack_h)
    for row_idx, row in enumerate(groups):
        row_w = int(row_widths[row_idx])
        row_h = int(row_heights[row_idx])
        slack_w = w - (LEFT_STRIP + row_w + RIGHT_MARGIN)
        col_cursor = LEFT_STRIP + random.randint(0, slack_w)
        for spec_idx in row:
            spec = specs[spec_idx]
            out_r0 = row_cursor + random.randint(0, row_h - int(spec["out_h"]))
            out_c0 = col_cursor
            spec["out_r0"] = out_r0
            spec["out_c0"] = out_c0
            spec["in_r0"] = out_r0 - int(spec["offset_r"])
            spec["in_c0"] = out_c0 - int(spec["offset_c"])
            col_cursor += int(spec["out_w"]) + WALL_GAP
        row_cursor += row_h + WALL_GAP


def _build_specs(legend_colors: list[int], chosen_colors: list[int]) -> list[dict[str, object]]:
    scale_by_color = {color: idx + 2 for idx, color in enumerate(legend_colors)}
    specs: list[dict[str, object]] = []
    for color in chosen_colors:
        scale = scale_by_color[color]
        shape_kind, region = _sample_region(scale)
        wall = _boundary(region)
        interior = region & (~wall)
        seed_choices = [tuple(map(int, pt)) for pt in np.argwhere(interior)]
        seed_local = random.choice(seed_choices)
        outline = _scaled_outline(region, scale)
        out_h, out_w = map(int, outline.shape)
        specs.append(
            {
                "color": color,
                "scale": scale,
                "shape_kind": shape_kind,
                "region": region,
                "wall": wall,
                "seed_local": seed_local,
                "outline": outline,
                "out_h": out_h,
                "out_w": out_w,
                "offset_r": _center_offset(region.shape[0], scale),
                "offset_c": _center_offset(region.shape[1], scale),
            }
        )
    random.shuffle(specs)
    return specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        legend_len = random.randint(2, 4)
        legend_colors = random.sample(COLOR_POOL, legend_len)

        max_objects = min(3, legend_len)
        if legend_len == 4:
            max_objects = 2
        n_objects = random.randint(1, max_objects)
        chosen_colors = random.sample(legend_colors, n_objects)

        specs = _build_specs(legend_colors, chosen_colors)
        boxes = [(int(spec["out_h"]), int(spec["out_w"])) for spec in specs]
        layout = _pick_layout(boxes)
        if layout is None:
            continue

        _, layout_h, layout_w = layout
        min_h = max(legend_len + 2, TOP_MARGIN + layout_h + BOTTOM_MARGIN)
        min_w = LEFT_STRIP + layout_w + RIGHT_MARGIN
        if min_h > 30 or min_w > 30:
            continue

        h = random.randint(max(min_h, 18), 30)
        w = random.randint(max(min_w, 18), 30)
        _place_specs(h, w, specs, layout)

        grid = np.full((h, w), BLACK, dtype=int)
        for row, color in enumerate(legend_colors):
            grid[row, 0] = color

        valid = True
        for spec in specs:
            region = spec["region"]
            wall = spec["wall"]
            in_r0 = int(spec["in_r0"])
            in_c0 = int(spec["in_c0"])
            rh, rw = region.shape
            if in_r0 < 0 or in_c0 < LEFT_STRIP or in_r0 + rh > h or in_c0 + rw > w:
                valid = False
                break
            patch = grid[in_r0 : in_r0 + rh, in_c0 : in_c0 + rw]
            if np.any(patch[wall] != BLACK):
                valid = False
                break
            patch[wall] = GRAY
            seed_r, seed_c = spec["seed_local"]
            if patch[seed_r, seed_c] != BLACK:
                valid = False
                break
            patch[seed_r, seed_c] = int(spec["color"])
        if not valid:
            continue

        expected = _render_expected((h, w), specs)
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
