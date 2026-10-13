from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, paste


BG = 0
OUTLINE = 5
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
AREAS = (3, 4, 5, 6, 7, 8)
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (22, 30)
GEN_ATTEMPTS = 400


def _row_interior_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    rows: dict[int, list[int]] = {}
    for r, c in coords:
        rows.setdefault(int(r), []).append(int(c))

    interior: list[tuple[int, int]] = []
    for r, cols in rows.items():
        left = min(cols)
        right = max(cols)
        for c in range(left + 1, right):
            if int(grid[r, c]) == BG:
                interior.append((int(r), int(c)))

    assert interior, "outline must enclose a non-empty cavity"
    interior_coords = np.array(interior, dtype=int)
    return {
        "area": int(len(interior)),
        "coords": interior_coords,
        "bbox": bbox_coords(interior_coords),
    }


def _template_patch(grid: np.ndarray, comp: dict) -> np.ndarray:
    return crop_bbox(grid, comp["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match monochrome template tiles to gray outline cavities by equal row-interior area, then stamp them in place.

    Input structure:
    - A black background contains several small 4-connected monochrome template components in distinct non-gray colors.
    - Several separate gray 8-connected one-pixel-wide closed outline objects enclose black cavities.
    - Templates and outlines are spatially disjoint, and each outline's row-wise interior area matches exactly one template's pixel count.

    Rule:
    - Find every gray outline component and compute its cavity area by summing, for each occupied row, the black cells strictly between that row's leftmost and rightmost gray pixels.
    - Find every non-black, non-gray monochrome template component and count its pixels.
    - Pair templates to outlines by equality between template pixel count and outline cavity area.
    - Copy each template's cropped shape into the matched cavity, aligning the template bbox top-left to the cavity bbox top-left, then erase the template from its source location.
    - Keep the gray outline pixels unchanged.

    Key insight:
    - Neither color identity nor spatial proximity determines the destination; the only reliable pairing key is the equality between template area and outline row-interior area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    outlines = find_components(
        g, bg=BG, colors=[OUTLINE], connectivity=8, monochromatic=True, return_format="dict"
    )
    templates = [
        comp
        for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
        if int(comp["color"]) not in (BG, OUTLINE)
    ]

    outline_by_area: dict[int, dict] = {}
    for comp in outlines:
        info = _row_interior_info(g, comp["coords"])
        assert info["area"] not in outline_by_area, "outline cavity areas must be unique"
        outline_by_area[info["area"]] = info

    template_by_area: dict[int, dict] = {}
    for comp in templates:
        area = int(comp["area"])
        assert area not in template_by_area, "template areas must be unique"
        template_by_area[area] = comp

    assert set(template_by_area) == set(outline_by_area), "template areas must bijectively match outline areas"

    out = g.copy()
    for comp in templates:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = BG

    for area, comp in template_by_area.items():
        patch = _template_patch(g, comp)
        cavity_bbox = outline_by_area[area]["bbox"]
        paste(out, patch, cavity_bbox[0], cavity_bbox[1], transparent=BG)

    return out


def _random_composition(total: int, parts: int, max_part: int) -> list[int]:
    for _ in range(200):
        values = [1] * parts
        for _ in range(total - parts):
            values[random.randrange(parts)] += 1
        if max(values) <= max_part:
            random.shuffle(values)
            return values
    raise RuntimeError("failed to sample a bounded composition")


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _flip_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    if random.random() < 0.5:
        max_c = max(c for _, c in out)
        out = {(r, max_c - c) for r, c in out}
    if random.random() < 0.5:
        max_r = max(r for r, _ in out)
        out = {(max_r - r, c) for r, c in out}
    return _normalize(out)


def _shape_to_patch(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _make_shape(area: int) -> set[tuple[int, int]]:
    max_rows = min(4, area)
    for _ in range(400):
        rows = random.randint(2, max_rows)
        widths = _random_composition(area, rows, max_part=5)
        starts = [0]
        for idx in range(1, rows):
            prev_start = starts[-1]
            prev_width = widths[idx - 1]
            cur_width = widths[idx]
            low = prev_start - cur_width + 1
            high = prev_start + prev_width - 1
            starts.append(random.randint(low, high))

        cells = {
            (r, c)
            for r, (start, width) in enumerate(zip(starts, widths))
            for c in range(start, start + width)
        }
        cells = _flip_cells(cells)
        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w < 2:
            continue
        if h * w == area:
            continue
        return cells
    raise RuntimeError(f"failed to generate an irregular row-convex shape for area {area}")


def _outline_patch_from_template(template_patch: np.ndarray) -> np.ndarray:
    mask = template_patch != BG
    h, w = mask.shape
    inner = np.zeros((h + 2, w + 2), dtype=bool)
    inner[1 : h + 1, 1 : w + 1] = mask
    shell = np.zeros_like(inner)
    for r, c in np.argwhere(inner):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < inner.shape[0] and 0 <= cc < inner.shape[1] and not inner[rr, cc]:
                    shell[rr, cc] = True
    patch = np.zeros_like(inner, dtype=int)
    patch[shell] = OUTLINE
    return patch


def _place_rectangles(
    grid_shape: tuple[int, int], sizes: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    h, w = map(int, grid_shape)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[tuple[int, int] | None] = [None] * len(sizes)

    for idx in order:
        ph, pw = sizes[idx]
        for _ in range(250):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            placed[idx] = (r0, c0)
            break
        else:
            return None

    return [pos for pos in placed if pos is not None]


def _manual_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()

    outlines = find_components(
        g, bg=BG, colors=[OUTLINE], connectivity=8, monochromatic=True, return_format="dict"
    )
    templates = [
        comp
        for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
        if int(comp["color"]) not in (BG, OUTLINE)
    ]

    outline_by_area = {info["area"]: info for info in (_row_interior_info(g, comp["coords"]) for comp in outlines)}
    template_by_area = {int(comp["area"]): comp for comp in templates}

    for comp in templates:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = BG

    for area, comp in template_by_area.items():
        patch = _template_patch(g, comp)
        cavity_bbox = outline_by_area[area]["bbox"]
        paste(out, patch, cavity_bbox[0], cavity_bbox[1], transparent=BG)
    return out


def _semantic_signature(grid: np.ndarray) -> tuple[list[tuple[int, tuple[int, int], int]], list[tuple[int, tuple[int, int], int]]]:
    g = np.asarray(grid, dtype=int)
    templates = [
        comp
        for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
        if int(comp["color"]) not in (BG, OUTLINE)
    ]
    outlines = find_components(
        g, bg=BG, colors=[OUTLINE], connectivity=8, monochromatic=True, return_format="dict"
    )

    template_items = sorted(
        [
            (
                int(comp["area"]),
                (int(comp["bbox"][0]), int(comp["bbox"][1])),
                int(comp["color"]),
            )
            for comp in templates
        ],
        key=lambda item: item[1],
    )
    outline_items = sorted(
        [
            (
                info["area"],
                (int(info["bbox"][0]), int(info["bbox"][1])),
                idx,
            )
            for idx, info in enumerate(_row_interior_info(g, comp["coords"]) for comp in outlines)
        ],
        key=lambda item: item[1],
    )
    return template_items, outline_items


def _nearest_outline_ok(template_items, outline_items) -> bool:
    centers = [
        (area, (pos[0], pos[1]))
        for area, pos, _ in outline_items
    ]
    mismatched_nearest = 0
    for area, pos, _ in template_items:
        best_area = min(centers, key=lambda item: abs(item[1][0] - pos[0]) + abs(item[1][1] - pos[1]))[0]
        if best_area != area:
            mismatched_nearest += 1
    return mismatched_nearest >= 1


def _row_major_order_ok(template_items, outline_items) -> bool:
    return [area for area, _pos, _color in template_items] != [area for area, _pos, _idx in outline_items]


def _color_area_order_ok(template_items) -> bool:
    by_area = [color for _area, _pos, color in sorted(template_items, key=lambda item: item[0])]
    return by_area != sorted(by_area) and by_area != sorted(by_area, reverse=True)


def _valid_outline_patch(patch: np.ndarray, expected_area: int) -> bool:
    probe = patch.copy()
    gray_components = find_components(
        probe, bg=BG, colors=[OUTLINE], connectivity=8, monochromatic=True, return_format="dict"
    )
    if len(gray_components) != 1:
        return False
    info = _row_interior_info(probe, gray_components[0]["coords"])
    return info["area"] == int(expected_area)


def _gen_attempt() -> np.ndarray | None:
    areas = random.sample(AREAS, 3)
    colors = random.sample(TEMPLATE_COLORS, 3)

    templates: list[dict] = []
    outlines: list[dict] = []
    for area, color in zip(areas, colors):
        cells = _make_shape(area)
        template_patch = _shape_to_patch(cells, color)
        outline_patch = _outline_patch_from_template(template_patch)
        if not _valid_outline_patch(outline_patch, area):
            return None
        templates.append({"area": area, "color": color, "patch": template_patch})
        outlines.append({"area": area, "patch": outline_patch})

    random.shuffle(templates)
    random.shuffle(outlines)
    sizes = [tuple(item["patch"].shape) for item in templates] + [tuple(item["patch"].shape) for item in outlines]
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    placements = _place_rectangles((grid_h, grid_w), sizes)
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros_like(grid)
    expected[:] = BG
    template_positions = placements[: len(templates)]
    outline_positions = placements[len(templates) :]

    for item, (r0, c0) in zip(templates, template_positions):
        paste(grid, item["patch"], r0, c0, transparent=BG)

    for item, (r0, c0) in zip(outlines, outline_positions):
        paste(grid, item["patch"], r0, c0, transparent=BG)

    expected = _manual_output(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    template_items, outline_items = _semantic_signature(grid)
    if len(template_items) != 3 or len(outline_items) != 3:
        return None
    if not _row_major_order_ok(template_items, outline_items):
        return None
    if not _nearest_outline_ok(template_items, outline_items):
        return None
    if not _color_area_order_ok(template_items):
        return None
    if {area for area, _pos, _color in template_items} != {area for area, _pos, _idx in outline_items}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-matched stamp puzzle")
