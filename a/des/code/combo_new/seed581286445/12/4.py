from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _crop_true_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _symmetry_branch(mask: np.ndarray) -> str:
    if np.array_equal(mask, np.fliplr(mask)):
        return "vertical"
    if np.array_equal(mask, np.flipud(mask)):
        return "horizontal"
    return "asymmetric"


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GRAY)
    assert coords.size > 0, "legend frame not found"
    return _bbox(coords)


def _extract_templates(grid: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _legend_bbox(grid)
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]
    full_gray_rows = [r for r in range(legend.shape[0]) if np.all(legend[r] == GRAY)]
    templates: list[np.ndarray] = []
    for top, bottom in zip(full_gray_rows, full_gray_rows[1:]):
        if bottom - top <= 1:
            continue
        slot = legend[top + 1 : bottom, 1:-1].copy()
        templates.append(slot)
    assert templates, "legend templates not found"
    return templates


def _find_target_regions(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    fillable = (grid != BLUE) & (grid != GRAY)
    regions: list[np.ndarray] = []
    for comp in _components(fillable):
        rows = comp[:, 0]
        cols = comp[:, 1]
        if rows.min() == 0 or cols.min() == 0 or rows.max() == h - 1 or cols.max() == w - 1:
            continue
        comp_set = {tuple(x) for x in comp.tolist()}
        border_colors = set()
        for r, c in comp_set:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in comp_set:
                    border_colors.add(int(grid[nr, nc]))
        if border_colors == {BLUE}:
            regions.append(comp)
    return regions


def _apply_region_fill(
    out: np.ndarray,
    coords: np.ndarray,
    template: np.ndarray,
    branch: str,
) -> None:
    if branch == "horizontal":
        tile = np.rot90(template, 3)
    elif branch == "asymmetric":
        tile = np.rot90(template, 2)
    else:
        tile = template
    r0, c0, _r1, _c1 = _bbox(coords)
    th, tw = tile.shape
    for r, c in coords:
        out[r, c] = int(tile[(r - r0) % th, (c - c0) % tw])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue-enclosed region with a legend-selected tiled pattern whose rotation depends on the region's symmetry.

    Input structure:
    - The background is black.
    - Several one-pixel-thick blue boundaries form disjoint closed regions whose interiors contain black cells and scattered red seed pixels.
    - A gray-framed legend sits on the right or bottom; inside it, green-on-black templates are stacked from top to bottom and separated by gray rows.

    Rule:
    - Count the red seed pixels inside each blue-enclosed region and select the legend's N-th template for seed count N.
    - Inspect the region's interior shape: vertical symmetry keeps the template unchanged, horizontal-only symmetry rotates it 90 degrees clockwise, and no horizontal/vertical symmetry rotates it 180 degrees.
    - Tile the chosen template from the region bbox's top-left corner across the region interior, clipping at the boundary and replacing both black interior cells and red seeds.

    Key insight:
    - The seed count decides which template to use, while the region geometry independently decides how that template must be rotated before tiling.
    """
    g = np.asarray(grid, dtype=int)
    templates = _extract_templates(g)
    out = g.copy()
    for comp in _find_target_regions(g):
        seed_count = int(np.sum(g[comp[:, 0], comp[:, 1]] == RED))
        assert 1 <= seed_count <= len(templates), "seed count exceeds legend size"
        mask, _origin = _coords_to_mask(comp)
        branch = _symmetry_branch(mask)
        _apply_region_fill(out, comp, templates[seed_count - 1], branch)
    return out


VERTICAL_BASES = (
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
)

HORIZONTAL_BASES = tuple(base.T.copy() for base in VERTICAL_BASES)

ASYMMETRIC_BASES = (
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
)

GEN_ATTEMPTS = 400
TEMPLATE_COUNT = 3


def _sample_template(width: int) -> np.ndarray:
    height = random.randint(2, 3)
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(80):
        arr = np.zeros((height, width), dtype=int)
        green_count = random.randint(2, max(2, height * width - 1))
        for r, c in random.sample(cells, green_count):
            arr[r, c] = GREEN
        if np.all(arr == GREEN):
            continue
        if np.array_equal(arr, np.rot90(arr, 2)):
            continue
        if height == width and np.array_equal(arr, np.rot90(arr, 1)):
            continue
        return arr
    raise RuntimeError("failed to sample non-symmetric template")


def _build_legend(templates: list[np.ndarray]) -> np.ndarray:
    inner_w = templates[0].shape[1]
    inner_h = sum(t.shape[0] for t in templates) + len(templates) - 1
    legend = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    legend[0, :] = legend[-1, :] = GRAY
    legend[:, 0] = legend[:, -1] = GRAY
    row = 1
    for i, template in enumerate(templates):
        th, tw = template.shape
        legend[row : row + th, 1 : 1 + tw] = template
        row += th
        if i < len(templates) - 1:
            legend[row, :] = GRAY
            row += 1
    return legend


def _sample_category_mask(category: str) -> np.ndarray:
    if category == "vertical":
        bases = VERTICAL_BASES
    elif category == "horizontal":
        bases = HORIZONTAL_BASES
    else:
        bases = ASYMMETRIC_BASES
    for _ in range(40):
        mask = random.choice(bases).copy()
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if category == "asymmetric" and random.random() < 0.5:
            mask = np.rot90(mask, random.randint(0, 3))
        if _symmetry_branch(mask) == category:
            return mask
    raise RuntimeError("failed to sample region mask")


def _make_region_record(category: str, seed_count: int) -> dict:
    coarse = _sample_category_mask(category)
    scale = random.choice((1, 2, 2))
    interior = np.kron(coarse, np.ones((scale, scale), dtype=bool))
    padded = np.pad(interior, 1)
    boundary = _dilate8(padded) & ~padded
    sprite = np.full(padded.shape, BLACK, dtype=int)
    sprite[boundary] = BLUE
    coords = np.argwhere(padded)
    sprite = sprite.copy()
    for r, c in random.sample(coords.tolist(), seed_count):
        sprite[r, c] = RED
    bbox_mask, bbox_offset = _crop_true_mask(padded)
    return {
        "sprite": sprite,
        "interior_mask": padded,
        "bbox_mask": bbox_mask,
        "bbox_offset": bbox_offset,
        "seed_count": seed_count,
        "branch": _symmetry_branch(bbox_mask),
    }


def _layout_region_canvas(records: list[dict]) -> tuple[np.ndarray, list[dict]]:
    slot_ids = random.sample([0, 1, 2, 3], len(records))
    assigned: dict[int, dict] = {}
    for slot, record in zip(slot_ids, records):
        assigned[slot] = record

    cell_heights = [random.randint(2, 4) for _ in range(4)]
    cell_widths = [random.randint(2, 4) for _ in range(4)]
    for slot, record in assigned.items():
        sh, sw = record["sprite"].shape
        cell_heights[slot] = sh + random.randint(0, 1)
        cell_widths[slot] = sw + random.randint(0, 1)

    row0_h = max(cell_heights[0], cell_heights[1])
    row1_h = max(cell_heights[2], cell_heights[3])
    col0_w = max(cell_widths[0], cell_widths[2])
    col1_w = max(cell_widths[1], cell_widths[3])
    row_gap = 1
    col_gap = 1
    area_h = row0_h + row_gap + row1_h
    area_w = col0_w + col_gap + col1_w
    canvas = np.full((area_h, area_w), BLACK, dtype=int)

    placed_records: list[dict] = []
    for slot, record in assigned.items():
        sh, sw = record["sprite"].shape
        row_base = 0 if slot < 2 else row0_h + row_gap
        col_base = 0 if slot % 2 == 0 else col0_w + col_gap
        slot_h = row0_h if slot < 2 else row1_h
        slot_w = col0_w if slot % 2 == 0 else col1_w
        top = row_base + random.randint(0, slot_h - sh)
        left = col_base + random.randint(0, slot_w - sw)
        canvas[top : top + sh, left : left + sw] = record["sprite"]
        placed = dict(record)
        placed["local_top"] = top
        placed["local_left"] = left
        placed_records.append(placed)
    return canvas, placed_records


def _split_slack(total: int) -> tuple[int, int]:
    left = random.randint(0, total)
    return left, total - left


def _compose_input(
    region_canvas: np.ndarray,
    legend: np.ndarray,
    records: list[dict],
    side: str,
) -> tuple[np.ndarray, list[dict]] | None:
    rh, rw = region_canvas.shape
    lh, lw = legend.shape
    gap = 1
    if side == "right":
        combined_h = max(rh, lh)
        combined_w = rw + gap + lw
        if combined_h > 30 or combined_w > 30:
            return None
        final_h = combined_h + random.randint(0, 30 - combined_h)
        final_w = combined_w + random.randint(0, 30 - combined_w)
        top_margin = random.randint(0, final_h - combined_h)
        left_margin = random.randint(0, final_w - combined_w)
        region_y = top_margin + random.randint(0, combined_h - rh)
        region_x = left_margin
        legend_y = top_margin + random.randint(0, combined_h - lh)
        legend_x = left_margin + rw + gap
    else:
        combined_h = rh + gap + lh
        combined_w = max(rw, lw)
        if combined_h > 30 or combined_w > 30:
            return None
        final_h = combined_h + random.randint(0, 30 - combined_h)
        final_w = combined_w + random.randint(0, 30 - combined_w)
        top_margin = random.randint(0, final_h - combined_h)
        left_margin = random.randint(0, final_w - combined_w)
        region_y = top_margin
        region_x = left_margin + random.randint(0, combined_w - rw)
        legend_y = top_margin + rh + gap
        legend_x = left_margin + random.randint(0, combined_w - lw)

    grid = np.full((final_h, final_w), BLACK, dtype=int)
    grid[region_y : region_y + rh, region_x : region_x + rw] = region_canvas
    grid[legend_y : legend_y + lh, legend_x : legend_x + lw] = legend

    global_records: list[dict] = []
    for record in records:
        placed = dict(record)
        placed["global_top"] = region_y + record["local_top"]
        placed["global_left"] = region_x + record["local_left"]
        global_records.append(placed)
    return grid, global_records


def _expected_output(inp: np.ndarray, templates: list[np.ndarray], records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for record in records:
        coords = np.argwhere(record["interior_mask"])
        coords = coords + np.array([[record["global_top"], record["global_left"]]])
        _apply_region_fill(out, coords, templates[record["seed_count"] - 1], record["branch"])
    return out


def _gen_attempt() -> np.ndarray | None:
    template_width = random.randint(2, 4)
    templates: list[np.ndarray] = []
    while len(templates) < TEMPLATE_COUNT:
        template = _sample_template(template_width)
        if any(np.array_equal(template, other) for other in templates):
            continue
        templates.append(template)
    legend = _build_legend(templates)

    n_regions = 3 + int(random.random() < 0.35)
    categories = ["vertical", "horizontal", "asymmetric"]
    if n_regions == 4:
        categories.append(random.choice(categories))
        side = "right"
    else:
        side = random.choice(("right", "bottom"))

    seed_counts = [1, 2, 3]
    if n_regions == 4:
        seed_counts.append(random.randint(1, 3))
    random.shuffle(seed_counts)

    try:
        region_records = [
            _make_region_record(category, seed_count)
            for category, seed_count in zip(categories, seed_counts)
        ]
        region_canvas, placed_records = _layout_region_canvas(region_records)
    except RuntimeError:
        return None

    composed = _compose_input(region_canvas, legend, placed_records, side)
    if composed is None:
        return None
    inp, global_records = composed
    expected = _expected_output(inp, templates, global_records)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
