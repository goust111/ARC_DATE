from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

LAYER_MIN = 2
LAYER_MAX = 4
GROUP_MIN = 1
GROUP_MAX = 3

GRID_MIN = 20
GRID_MAX = 30
CORE_MAX_BY_LAYERS = {2: 6, 3: 5, 4: 4}

PLACEMENT_ATTEMPTS = 120
GEN_ATTEMPTS = 400
GROUP_PAD = 2

BBox = tuple[int, int, int, int]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _bbox(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(pts)),
                }
            )

    return comps


def _is_solid_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_rect_ring(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return bool(np.array_equal(patch, mask))


def _expand_bbox(bbox: BBox, pad: int, shape: tuple[int, int]) -> BBox | None:
    r0, c0, r1, c1 = bbox
    nr0, nc0 = r0 - pad, c0 - pad
    nr1, nc1 = r1 + pad, c1 + pad
    h, w = shape
    if nr0 < 0 or nc0 < 0 or nr1 >= h or nc1 >= w:
        return None
    return nr0, nc0, nr1, nc1


def _find_groups(grid: np.ndarray, bg: int) -> tuple[list[dict], list[tuple[int, list[int]]]]:
    comps = _components_by_color(grid, bg)
    rings_by_bbox: dict[BBox, int] = {}
    solid_ids: list[int] = []

    for idx, comp in enumerate(comps):
        if _is_rect_ring(grid, comp):
            rings_by_bbox[comp["bbox"]] = idx
        elif _is_solid_rectangle(comp):
            solid_ids.append(idx)

    used_ids: set[int] = set()
    groups: list[tuple[int, list[int]]] = []

    solid_ids.sort(key=lambda idx: (comps[idx]["area"], comps[idx]["bbox"]))
    for core_idx in solid_ids:
        if core_idx in used_ids:
            continue

        core_bbox = comps[core_idx]["bbox"]
        ring_ids: list[int] = []
        pad = 1
        while True:
            ring_bbox = _expand_bbox(core_bbox, pad, grid.shape)
            if ring_bbox is None:
                break
            ring_idx = rings_by_bbox.get(ring_bbox)
            if ring_idx is None or ring_idx in used_ids:
                break
            ring_ids.append(ring_idx)
            pad += 2

        if LAYER_MIN <= len(ring_ids) <= LAYER_MAX:
            groups.append((core_idx, ring_ids))
            used_ids.add(core_idx)
            used_ids.update(ring_ids)

    return comps, groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift nested frame colors outward by one layer and remove the edge strip.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps, groups = _find_groups(g, bg)

    out = g.copy()
    keep = np.zeros_like(g, dtype=bool)

    for core_idx, ring_ids in groups:
        core = comps[core_idx]
        core_coords = core["coords"]
        keep[core_coords[:, 0], core_coords[:, 1]] = True

        ring_colors_inner_to_outer = [int(comps[idx]["color"]) for idx in ring_ids]
        shifted_inner_to_outer = [int(core["color"])] + ring_colors_inner_to_outer[:-1]
        for ring_idx, new_color in zip(ring_ids, shifted_inner_to_outer, strict=False):
            coords = comps[ring_idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = new_color
            keep[coords[:, 0], coords[:, 1]] = True

    out[(g != bg) & (~keep)] = bg
    return out


def _draw_ring(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_group(
    grid: np.ndarray,
    top: int,
    left: int,
    core_h: int,
    core_w: int,
    frame_colors_outer_to_inner: list[int],
    core_color: int,
) -> None:
    layers = len(frame_colors_outer_to_inner)
    outer_h = core_h + 4 * layers - 2
    outer_w = core_w + 4 * layers - 2

    for depth, color in enumerate(frame_colors_outer_to_inner):
        bbox = (
            top + 2 * depth,
            left + 2 * depth,
            top + outer_h - 1 - 2 * depth,
            left + outer_w - 1 - 2 * depth,
        )
        _draw_ring(grid, bbox, color)

    core_top = top + 2 * layers - 1
    core_left = left + 2 * layers - 1
    grid[core_top : core_top + core_h, core_left : core_left + core_w] = core_color


def _boxes_separated(a: BBox, b: BBox, pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _place_boxes(
    rng: random.Random,
    box_sizes: list[tuple[int, int]],
    height: int,
    width: int,
    strip_side: str,
) -> list[tuple[int, int]] | None:
    top_margin = 2 if strip_side == "top" else 1
    bottom_limit = height - 3 if strip_side == "bottom" else height - 2
    left_margin = 1
    right_limit = width - 2

    order = sorted(range(len(box_sizes)), key=lambda idx: box_sizes[idx][0] * box_sizes[idx][1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(box_sizes)
    placed_boxes: list[BBox] = []

    for idx in order:
        box_h, box_w = box_sizes[idx]
        max_r0 = bottom_limit - box_h + 1
        max_c0 = right_limit - box_w + 1
        if max_r0 < top_margin or max_c0 < left_margin:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(top_margin, max_r0)
            c0 = rng.randint(left_margin, max_c0)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(_boxes_separated(bbox, other, GROUP_PAD) for other in placed_boxes):
                placements[idx] = (r0, c0)
                placed_boxes.append(bbox)
                break
        else:
            return None

    if any(pos is None for pos in placements):
        return None
    return [(int(pos[0]), int(pos[1])) for pos in placements if pos is not None]


def _build_expected_output(
    height: int,
    width: int,
    placements: list[tuple[int, int]],
    core_sizes: list[tuple[int, int]],
    frame_colors_outer_to_inner: list[int],
    core_color: int,
) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    shifted = frame_colors_outer_to_inner[1:] + [core_color]
    for (top, left), (core_h, core_w) in zip(placements, core_sizes, strict=False):
        _draw_group(out, top, left, core_h, core_w, shifted, core_color)
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    layers = rng.randint(LAYER_MIN, LAYER_MAX)
    n_groups = rng.randint(GROUP_MIN, GROUP_MAX)

    core_max = max(1, CORE_MAX_BY_LAYERS[layers] - max(0, n_groups - 1))
    frame_colors = rng.sample(list(PALETTE), layers)
    remaining = [c for c in PALETTE if c not in frame_colors]
    core_color = rng.choice(remaining)

    core_sizes: list[tuple[int, int]] = []
    box_sizes: list[tuple[int, int]] = []
    for _ in range(n_groups):
        core_h = rng.randint(1, core_max)
        core_w = rng.randint(1, core_max)
        if rng.random() < 0.25:
            core_h = 1
        if rng.random() < 0.25:
            core_w = 1
        core_sizes.append((core_h, core_w))
        box_sizes.append((core_h + 4 * layers - 2, core_w + 4 * layers - 2))

    min_h = max(GRID_MIN, max(box_h for box_h, _box_w in box_sizes) + 4)
    min_w = max(GRID_MIN, max(box_w for _box_h, box_w in box_sizes) + 4)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = rng.randint(min_h, GRID_MAX)
    width = rng.randint(min_w, GRID_MAX)
    strip_side = rng.choice(["top", "bottom"])
    placements = _place_boxes(rng, box_sizes, height, width, strip_side)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for (top, left), (core_h, core_w) in zip(placements, core_sizes, strict=False):
        _draw_group(grid, top, left, core_h, core_w, frame_colors, core_color)

    strip = frame_colors + [core_color]
    if width - len(strip) < 2:
        return None
    strip_col = rng.randint(1, width - len(strip) - 1)
    strip_row = 0 if strip_side == "top" else height - 1
    for offset, color in enumerate(strip):
        grid[strip_row, strip_col + offset] = color

    expected = _build_expected_output(height, width, placements, core_sizes, frame_colors, core_color)

    _comps, groups = _find_groups(grid, BG)
    if len(groups) != n_groups:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
