from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, tile_fill


BG = Color.BLACK
DIVIDER = Color.GREY
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _region_slices(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = map(int, grid.shape)
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == DIVIDER)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == DIVIDER)]
    row_bounds = [-1] + sorted(set(map(int, sep_rows))) + [h]
    col_bounds = [-1] + sorted(set(map(int, sep_cols))) + [w]

    regions: list[tuple[int, int, int, int]] = []
    for i in range(len(row_bounds) - 1):
        r0 = row_bounds[i] + 1
        r1 = row_bounds[i + 1]
        if r0 >= r1:
            continue
        for j in range(len(col_bounds) - 1):
            c0 = col_bounds[j] + 1
            c1 = col_bounds[j + 1]
            if c0 >= c1:
                continue
            regions.append((int(r0), int(r1), int(c0), int(c1)))
    return regions


def _corner_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _fg_mask(region: np.ndarray) -> np.ndarray:
    return (region != BG) & (region != DIVIDER)


def _is_corner_marker_region(region: np.ndarray) -> bool:
    mask = _fg_mask(region)
    coords = np.argwhere(mask)
    if int(coords.shape[0]) != 4:
        return False
    return {tuple(map(int, coord)) for coord in coords} == set(_corner_positions(region.shape))


def _corner_colors(region: np.ndarray) -> list[int]:
    return [
        int(region[0, 0]),
        int(region[0, -1]),
        int(region[-1, 0]),
        int(region[-1, -1]),
    ]


def _extract_template(region: np.ndarray) -> np.ndarray:
    mask = _fg_mask(region)
    box = bbox(mask)
    assert box is not None, "template region must contain foreground"
    template = crop_bbox(region, box)

    template_only = np.where(template != BG, template, BG)
    comps = find_components(
        template_only,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(comps) == 1, "template must be a single 4-connected figure"
    return template


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    regions = _region_slices(g)
    assert len(regions) >= 2, "expected a divider grid with multiple regions"

    template_index = None
    template = None
    corner_info: dict[int, list[int]] = {}

    for idx, (r0, r1, c0, c1) in enumerate(regions):
        region = g[r0:r1, c0:c1]
        mask = _fg_mask(region)
        assert mask.any(), "every region must contain either markers or the template"
        if _is_corner_marker_region(region):
            corner_info[idx] = _corner_colors(region)
            continue

        assert template_index is None, "expected a unique template region"
        template_index = idx
        template = _extract_template(region)

    assert template_index is not None, "template region not found"
    assert template is not None
    assert len(corner_info) == len(regions) - 1, "every non-template region must be a corner-marker region"
    return {
        "regions": regions,
        "template_index": int(template_index),
        "template": template,
        "corner_info": corner_info,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the cropped template into only those divider-grid regions whose four corner markers agree, recoloring the copies to the consensus color.

    Input structure:
    - Grey full rows and columns form a rectangular divider grid, splitting the canvas into rectangular interior regions.
    - Exactly one region contains a 4-connected non-black, non-grey template figure on black background and has no corner markers.
    - Every other region contains exactly four singleton marker pixels placed at its four interior corners, with no other foreground inside the region.

    Rule:
    - Identify the unique template region, crop its foreground to the minimal bounding box, and keep that region unchanged.
    - For each non-template region, inspect the four interior-corner marker colors; only regions whose four markers are all the same color pass the gate.
    - Replace each passing region's interior by a top-left-anchored tiling of the cropped template, clipping partial tiles at the region boundary, and recolor every non-black tiled cell to the shared marker color.
    - Leave failing marker regions, divider lines, and the original template region unchanged.

    Key insight:
    - The corner markers do not directly paint their own region; they only decide whether the independently extracted template geometry is allowed to appear there, while their shared color determines the recoloring of that pasted geometry.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)
    template = analysis["template"]

    out = g.copy()
    for idx, colors in analysis["corner_info"].items():
        if len(set(colors)) != 1:
            continue
        color = int(colors[0])
        r0, r1, c0, c1 = analysis["regions"][idx]
        shape = (r1 - r0, c1 - c0)
        tiled = tile_fill(shape, template)
        filled = np.full(shape, BG, dtype=int)
        filled[tiled != BG] = color
        out[r0:r1, c0:c1] = filled
    return out


def _pick_distinct_colors(rng: np.random.Generator, count: int, *, exclude: set[int] | None = None) -> list[int]:
    banned = set() if exclude is None else set(map(int, exclude))
    pool = [color for color in PALETTE if color not in banned]
    assert count <= len(pool)
    return [int(color) for color in rng.choice(pool, size=count, replace=False)]


def _shape_bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _random_connected_shape(
    rng: np.random.Generator,
    box_h: int,
    box_w: int,
    area: int,
) -> set[tuple[int, int]] | None:
    for _ in range(300):
        start = (int(rng.integers(box_h)), int(rng.integers(box_w)))
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != area:
            continue
        if (0, 0) in cells:
            continue
        if _shape_bbox(cells) != (0, 0, box_h - 1, box_w - 1):
            continue

        mask = np.zeros((box_h, box_w), dtype=int)
        for r, c in cells:
            mask[r, c] = 1
        comps = find_components(
            mask,
            bg=0,
            connectivity=4,
            monochromatic=False,
            return_format="dict",
        )
        if len(comps) != 1:
            continue
        return cells
    return None


def _has_proper_period(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                continue
            if h % ph != 0 or w % pw != 0:
                continue
            tile = mask[:ph, :pw]
            if np.array_equal(np.tile(tile, (h // ph, w // pw)), mask):
                return True
    return False


def _build_template_region(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if region_h < 4 or region_w < 4:
        return None

    for _ in range(300):
        box_h = int(rng.integers(2, region_h - 1))
        box_w = int(rng.integers(2, region_w - 1))
        if box_h > region_h - 2 or box_w > region_w - 2:
            continue

        max_area = min(8, box_h * box_w - 1)
        min_area = 3 if max_area == 3 else 4
        if min_area > max_area:
            continue
        area = int(rng.integers(min_area, max_area + 1))

        shape = _random_connected_shape(rng, box_h, box_w, area)
        if shape is None:
            continue

        template = np.zeros((box_h, box_w), dtype=int)
        palette_size = 2 if area < 5 else int(rng.integers(2, 4))
        colors = _pick_distinct_colors(rng, palette_size)
        color_assignments = colors.copy()
        while len(color_assignments) < area:
            color_assignments.append(int(colors[int(rng.integers(len(colors)))]))
        rng.shuffle(color_assignments)
        for (r, c), color in zip(sorted(shape), color_assignments):
            template[r, c] = int(color)

        if template[0, 0] != BG:
            continue
        if len(set(map(int, template[template != BG].tolist()))) < 2:
            continue
        if _has_proper_period(template != BG):
            continue

        region = np.zeros((region_h, region_w), dtype=int)
        top = int(rng.integers(1, region_h - box_h))
        left = int(rng.integers(1, region_w - box_w))
        region[top : top + box_h, left : left + box_w] = template
        cropped = _extract_template(region)
        if not np.array_equal(cropped, template):
            continue
        return region, template
    return None


def _marker_colors_for_status(
    rng: np.random.Generator,
    status: str,
    pass_pool: list[int],
    all_pool: list[int],
) -> tuple[list[int], int | None]:
    if status == "pass":
        color = int(pass_pool[int(rng.integers(len(pass_pool)))])
        return [color, color, color, color], color

    if status == "fail_majority":
        a, b = _pick_distinct_colors(rng, 2)
        colors = [a, a, a, b]
        rng.shuffle(colors)
        return colors, None

    if status == "fail_pair":
        a, b = _pick_distinct_colors(rng, 2)
        colors = [a, a, b, b]
        rng.shuffle(colors)
        return colors, None

    if status == "fail_all_diff":
        return _pick_distinct_colors(rng, 4), None

    raise ValueError(f"unknown status: {status}")


def _passes_semantic_checks(grid: np.ndarray, meta: dict) -> bool:
    analysis = _analyze_grid(grid)
    if analysis["template_index"] != meta["template_index"]:
        return False
    if not np.array_equal(analysis["template"], meta["template"]):
        return False

    template = analysis["template"]
    template_mask = template != BG
    if template[0, 0] != BG or _has_proper_period(template_mask):
        return False
    if len(set(map(int, template[template != BG].tolist()))) < 2:
        return False

    template_region = analysis["regions"][analysis["template_index"]]
    tr0, tr1, tc0, tc1 = template_region
    if (tr1 - tr0) <= template.shape[0] or (tc1 - tc0) <= template.shape[1]:
        return False

    pass_indices = [idx for idx, info in meta["regions"].items() if info["status"] == "pass"]
    fail_indices = [idx for idx, info in meta["regions"].items() if info["status"] != "pass"]
    if len(pass_indices) < 2 or len(fail_indices) < 2:
        return False
    if not any(info["status"] == "fail_majority" for info in meta["regions"].values()):
        return False

    has_truncation = False
    has_repeat = False
    out = main(grid)
    for idx, (r0, r1, c0, c1) in enumerate(analysis["regions"]):
        region_in = grid[r0:r1, c0:c1]
        region_out = out[r0:r1, c0:c1]

        if idx == analysis["template_index"]:
            if not np.array_equal(region_in, region_out):
                return False
            continue

        info = meta["regions"][idx]
        if not _is_corner_marker_region(region_in):
            return False

        if info["status"] == "pass":
            expected = np.full(region_out.shape, BG, dtype=int)
            tiled = tile_fill(region_out.shape, template)
            expected[tiled != BG] = info["color"]
            if not np.array_equal(region_out, expected):
                return False
            if region_out.shape[0] % template.shape[0] != 0 or region_out.shape[1] % template.shape[1] != 0:
                has_truncation = True
            if region_out.shape[0] > template.shape[0] or region_out.shape[1] > template.shape[1]:
                has_repeat = True
            if int(region_in[0, 0]) != BG and int(region_out[0, 0]) == BG:
                meta["marker_erasure_visible"] = True
        else:
            if not np.array_equal(region_out, region_in):
                return False

    if not has_truncation or not has_repeat:
        return False
    if not meta.get("marker_erasure_visible", False):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    region_rows = int(rng.integers(2, 4))
    region_cols = int(rng.integers(3, 5))
    heights = [int(rng.integers(4, 7)) for _ in range(region_rows)]
    widths = [int(rng.integers(4, 7)) for _ in range(region_cols)]
    grid_h = sum(heights) + region_rows + 1
    grid_w = sum(widths) + region_cols + 1
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), DIVIDER, dtype=int)
    regions: list[tuple[int, int, int, int]] = []
    cursor_r = 1
    for h in heights:
        cursor_c = 1
        for w in widths:
            grid[cursor_r : cursor_r + h, cursor_c : cursor_c + w] = BG
            regions.append((cursor_r, cursor_r + h, cursor_c, cursor_c + w))
            cursor_c += w + 1
        cursor_r += h + 1

    template_index = int(rng.integers(len(regions)))
    tr0, tr1, tc0, tc1 = regions[template_index]
    built = _build_template_region(rng, tr1 - tr0, tc1 - tc0)
    if built is None:
        return None
    template_region, template = built
    grid[tr0:tr1, tc0:tc1] = template_region

    other_indices = [idx for idx in range(len(regions)) if idx != template_index]
    if len(other_indices) < 4:
        return None

    statuses = ["pass", "pass", "fail_majority", "fail_pair"]
    extra_choices = ("pass", "fail_majority", "fail_pair", "fail_all_diff")
    while len(statuses) < len(other_indices):
        statuses.append(str(extra_choices[int(rng.integers(len(extra_choices)))]))
    rng.shuffle(statuses)

    template_colors = set(map(int, template[template != BG].tolist()))
    pass_pool = [color for color in PALETTE if color not in template_colors]
    if len(pass_pool) < 2:
        return None

    meta = {
        "template_index": template_index,
        "template": template.copy(),
        "regions": {},
    }

    for idx, status in zip(other_indices, statuses):
        r0, r1, c0, c1 = regions[idx]
        region = np.zeros((r1 - r0, c1 - c0), dtype=int)
        colors, pass_color = _marker_colors_for_status(rng, status, pass_pool, list(PALETTE))
        for (rr, cc), color in zip(_corner_positions(region.shape), colors):
            region[rr, cc] = int(color)
        grid[r0:r1, c0:c1] = region
        meta["regions"][idx] = {"status": status, "color": pass_color}

    if not _passes_semantic_checks(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
