import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    GREY = 5


BG = Color.BLACK
MASK_A_COLOR = Color.RED
MASK_B_COLOR = Color.BLUE
TEMPLATE_COLOR = Color.GREEN
DIVIDER_COLOR = Color.GREY

CELL_W = 3
GEN_RETRIES = 400
PALETTE_CHOICES = [1, 2, 4, 6, 7, 8, 9]
TEMPLATE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
)


def _regions_between(lines: list[int], length: int) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for line in sorted(int(x) for x in lines):
        if start <= line - 1:
            regions.append((start, line - 1))
        start = line + 1
    if start <= length - 1:
        regions.append((start, length - 1))
    return regions


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == DIVIDER_COLOR)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == DIVIDER_COLOR)]
    assert len(full_rows) == 2, "expected exactly two horizontal divider rows"
    assert len(full_cols) >= 4, "expected several vertical divider columns"

    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    assert len(row_regions) == 3, "expected three horizontal bands"
    assert len(col_regions) >= 5, "expected at least five cell columns"

    band_h = row_regions[0][1] - row_regions[0][0] + 1
    assert all((r1 - r0 + 1) == band_h for r0, r1 in row_regions), "bands must be equal height"
    assert all((c1 - c0 + 1) == CELL_W for c0, c1 in col_regions), "cells must share the same width"

    def cell_patch(row_band: tuple[int, int], col_band: tuple[int, int]) -> np.ndarray:
        r0, r1 = row_band
        c0, c1 = col_band
        return g[r0 : r1 + 1, c0 : c1 + 1]

    mask_a: list[bool] = []
    mask_b: list[bool] = []
    for col_band in col_regions:
        top_patch = cell_patch(row_regions[0], col_band)
        mid_patch = cell_patch(row_regions[1], col_band)
        assert np.all(np.isin(top_patch, [BG, MASK_A_COLOR]))
        assert np.all(np.isin(mid_patch, [BG, MASK_B_COLOR]))
        mask_a.append(bool(np.any(top_patch == MASK_A_COLOR)))
        mask_b.append(bool(np.any(mid_patch == MASK_B_COLOR)))

    active = [a and b for a, b in zip(mask_a, mask_b)]
    assert any(active), "mask intersection must be non-empty"
    assert active != mask_a and active != mask_b, "intersection must be a strict subset of each mask"

    lower_band = row_regions[2]
    first_cell = col_regions[0]
    lr0, lr1 = lower_band
    fc0, fc1 = first_cell
    palette: list[int] = []
    for rr in range(lr0, lr1 + 1):
        row_vals = [int(v) for v in g[rr, fc0 : fc1 + 1] if int(v) not in (BG, DIVIDER_COLOR)]
        assert len(row_vals) == 1, "expected exactly one palette marker per lower-band row"
        palette.append(row_vals[0])

    template_cells: list[tuple[int, np.ndarray]] = []
    for idx, col_band in enumerate(col_regions[1:], start=1):
        patch = cell_patch(lower_band, col_band)
        if np.any(patch != BG):
            template_cells.append((idx, patch))
    assert len(template_cells) == 1, "expected exactly one non-empty non-palette lower cell"

    template_idx, template_patch = template_cells[0]
    assert np.all(np.isin(template_patch, [BG, TEMPLATE_COLOR]))
    template_coords = np.argwhere(template_patch == TEMPLATE_COLOR)
    assert 3 <= len(template_coords) <= 4, "template should have 3-4 pixels"
    assert _component_count(template_patch == TEMPLATE_COLOR) == 1, "template must be 4-connected"
    used_rows = sorted({int(r) for r, _ in template_coords})
    assert len(used_rows) >= 2, "template should span multiple rows"

    return {
        "grid": g,
        "row_regions": row_regions,
        "col_regions": col_regions,
        "band_h": band_h,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "active": active,
        "palette": palette,
        "template_idx": template_idx,
        "template_coords": [(int(r), int(c)) for r, c in template_coords],
        "used_rows": used_rows,
    }


def _render_output(
    cell_count: int,
    cell_w: int,
    band_h: int,
    active: list[bool],
    template_coords: list[tuple[int, int]],
    palette: list[int],
) -> np.ndarray:
    out = np.zeros((band_h, cell_count * cell_w), dtype=int)
    for idx in range(cell_count):
        if not active[idx]:
            continue
        c_shift = idx * cell_w
        for r, c in template_coords:
            out[r, c_shift + c] = int(palette[r])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the lower-band template into the cellwise intersection of the two upper masks, using row palette colors.

    Input structure:
    - The grid has a black background with two full-width grey divider rows, creating top, middle, and lower bands of equal height.
    - Grey full-height vertical divider columns split every band into aligned 3-column cells.
    - The top band marks mask A with red-filled cells, the middle band marks mask B with blue-filled cells, and the lower band contains one row-palette marker pixel per row in its first cell plus one green template shape in exactly one other cell.

    Rule:
    - Align the top and middle bands by cell column and mark a cell active exactly when its top cell contains red and its middle cell contains blue.
    - Read the lower-band template as the green pixels in the unique non-first non-empty cell, preserving each pixel's in-cell coordinates.
    - Read the lower band's first cell as a per-row palette: each row contributes the color of its lone non-black non-grey marker pixel.
    - Clear the whole lower band, remove all divider columns, and in every active cell stamp the template at the same in-cell coordinates, recoloring each stamped pixel by the palette color of that pixel's row.

    Key insight:
    - The output location comes from the boolean intersection of two aligned masks, while the output colors come from a separate row palette; neither source alone is sufficient.
    """
    scene = _parse_scene(grid)
    return _render_output(
        cell_count=len(scene["col_regions"]),
        cell_w=CELL_W,
        band_h=scene["band_h"],
        active=scene["active"],
        template_coords=scene["template_coords"],
        palette=scene["palette"],
    )


def _choose_masks(cell_count: int) -> tuple[list[bool], list[bool], list[bool]]:
    cells = list(range(cell_count))
    max_intersection = min(3, cell_count - 2)
    inter_size = random.randint(1, max_intersection)
    intersection = set(random.sample(cells, inter_size))

    remaining = [c for c in cells if c not in intersection]
    random.shuffle(remaining)
    split_a = random.randint(1, len(remaining) - 1)
    split_b = random.randint(split_a + 1, len(remaining))
    extra_a = set(remaining[:split_a])
    extra_b = set(remaining[split_a:split_b])

    mask_a = [idx in intersection or idx in extra_a for idx in cells]
    mask_b = [idx in intersection or idx in extra_b for idx in cells]
    active = [a and b for a, b in zip(mask_a, mask_b)]
    assert any(active) and active != mask_a and active != mask_b
    return mask_a, mask_b, active


def _shift_template(shape: tuple[tuple[int, int], ...], band_h: int) -> list[tuple[int, int]]:
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    top_choices = list(range(band_h - shape_h + 1))
    left_choices = list(range(CELL_W - shape_w + 1))
    shifts = [(tr, tc) for tr in top_choices for tc in left_choices]
    non_zero_shifts = [(tr, tc) for tr, tc in shifts if (tr, tc) != (0, 0)]
    shift_r, shift_c = random.choice(non_zero_shifts or shifts)
    return [(shift_r + r, shift_c + c) for r, c in shape]


def _build_input() -> tuple[np.ndarray, dict]:
    band_h = random.randint(4, 5)
    cell_count = random.randint(5, 7)
    height = band_h * 3 + 2
    width = cell_count * CELL_W + (cell_count - 1)

    grid = np.zeros((height, width), dtype=int)
    divider_rows = [band_h, 2 * band_h + 1]
    divider_cols = [k * (CELL_W + 1) - 1 for k in range(1, cell_count)]
    grid[divider_rows, :] = DIVIDER_COLOR
    grid[:, divider_cols] = DIVIDER_COLOR

    row_regions = _regions_between(divider_rows, height)
    col_regions = _regions_between(divider_cols, width)

    mask_a, mask_b, active = _choose_masks(cell_count)
    for idx, (c0, c1) in enumerate(col_regions):
        if mask_a[idx]:
            r0, r1 = row_regions[0]
            grid[r0 : r1 + 1, c0 : c1 + 1] = MASK_A_COLOR
        if mask_b[idx]:
            r0, r1 = row_regions[1]
            grid[r0 : r1 + 1, c0 : c1 + 1] = MASK_B_COLOR

    palette = random.sample(PALETTE_CHOICES, band_h)
    first_c0, first_c1 = col_regions[0]
    low_r0, low_r1 = row_regions[2]
    marker_cols = [random.randint(first_c0, first_c1) for _ in range(band_h)]
    for local_r, color in enumerate(palette):
        grid[low_r0 + local_r, marker_cols[local_r]] = color

    shape = random.choice(TEMPLATE_SHAPES)
    template_coords = _shift_template(shape, band_h)
    template_idx = random.randint(1, cell_count - 1)
    tc0, tc1 = col_regions[template_idx]
    for r, c in template_coords:
        grid[low_r0 + r, tc0 + c] = TEMPLATE_COLOR

    spec = {
        "band_h": band_h,
        "cell_count": cell_count,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "active": active,
        "palette": palette,
        "template_coords": template_coords,
        "template_idx": template_idx,
        "template_shifted": any((r, c) != (0, 0) for r, c in template_coords),
    }
    return grid, spec


def _validate_semantics(grid: np.ndarray, spec: dict) -> bool:
    scene = _parse_scene(grid)
    if scene["mask_a"] != spec["mask_a"] or scene["mask_b"] != spec["mask_b"]:
        return False
    if scene["active"] != spec["active"]:
        return False
    if scene["palette"] != spec["palette"]:
        return False
    if sorted(scene["template_coords"]) != sorted(spec["template_coords"]):
        return False
    if scene["template_idx"] != spec["template_idx"]:
        return False

    out = main(grid)
    expected = _render_output(
        cell_count=spec["cell_count"],
        cell_w=CELL_W,
        band_h=spec["band_h"],
        active=spec["active"],
        template_coords=spec["template_coords"],
        palette=spec["palette"],
    )
    if not np.array_equal(out, expected):
        return False

    if int(np.count_nonzero(out)) != sum(spec["active"]) * len(spec["template_coords"]):
        return False

    used_palette = {spec["palette"][r] for r, _ in spec["template_coords"]}
    if len(used_palette) < 2:
        return False

    active_a_only = _render_output(
        spec["cell_count"], CELL_W, spec["band_h"], spec["mask_a"], spec["template_coords"], spec["palette"]
    )
    active_b_only = _render_output(
        spec["cell_count"], CELL_W, spec["band_h"], spec["mask_b"], spec["template_coords"], spec["palette"]
    )
    flat_palette = [spec["palette"][scene["used_rows"][0]]] * spec["band_h"]
    flat_color = _render_output(
        spec["cell_count"], CELL_W, spec["band_h"], spec["active"], spec["template_coords"], flat_palette
    )
    cropped_coords = [(r - min(rr for rr, _ in spec["template_coords"]), c - min(cc for _, cc in spec["template_coords"])) for r, c in spec["template_coords"]]
    cropped = _render_output(
        spec["cell_count"], CELL_W, spec["band_h"], spec["active"], cropped_coords, spec["palette"]
    )

    if np.array_equal(active_a_only, out):
        return False
    if np.array_equal(active_b_only, out):
        return False
    if np.array_equal(flat_color, out):
        return False
    if np.array_equal(cropped, out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid, spec = _build_input()
        if _validate_semantics(grid, spec):
            return grid
    raise RuntimeError("failed to generate a valid dual-mask stamp scene")
