from __future__ import annotations

import random
import numpy as np


BG = 0
BORDER = 5
SEED_COLORS = (1, 2, 3, 4)  # blue, red, green, yellow in ARC
LEGEND_MARK = 8

TEMPLATE_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((-1, 0), (1, 0), (0, -1), (0, 1)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    ((0, -1), (0, 1)),
    ((-1, 0), (1, 0)),
    ((-1, 0), (0, 1)),
    ((1, 0), (0, -1)),
    ((-1, 0), (0, -1), (0, 1)),
    ((1, 0), (0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 1), (1, -1)),
)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_templates(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], ...]]:
    h, w = grid.shape
    top_centers: list[tuple[int, int]] = []
    left_centers: list[tuple[int, int]] = []

    if h >= 3:
        for c in range(1, w - 1):
            if int(grid[1, c]) in SEED_COLORS and BORDER not in grid[0:3, c - 1 : c + 2]:
                top_centers.append((1, c))

    if w >= 3:
        for r in range(1, h - 1):
            if int(grid[r, 1]) in SEED_COLORS and BORDER not in grid[r - 1 : r + 2, 0:3]:
                left_centers.append((r, 1))

    centers = top_centers if len(top_centers) >= len(left_centers) else left_centers
    assert centers, "expected at least one 3x3 legend pattern"

    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    for r, c in centers:
        color = int(grid[r, c])
        window = grid[r - 1 : r + 2, c - 1 : c + 2]
        offsets = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if int(window[dr + 1, dc + 1]) != BG:
                    offsets.append((dr, dc))
        templates[color] = tuple(sorted(set(offsets)))

    return templates


def _find_soil_regions(grid: np.ndarray) -> list[tuple[int, int, int, int, tuple[int, int], int]]:
    regions: list[tuple[int, int, int, int, tuple[int, int], int]] = []
    for comp in _components(grid == BORDER, connectivity=4):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border_cells = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) != border_cells:
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((inner != BG) & (inner != BORDER))
        if seeds.shape[0] != 1:
            continue

        sr, sc = map(int, seeds[0])
        color = int(inner[sr, sc])
        regions.append((r0, c0, r1, c1, (sr, sc), color))

    return regions


def _grow_mask(shape: tuple[int, int], seed: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = shape
    reached = np.zeros((h, w), dtype=bool)
    queue = [seed]
    reached[seed] = True
    head = 0

    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reached[nr, nc]:
                reached[nr, nc] = True
                queue.append((nr, nc))

    return reached


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read 3x3 growth templates from the grid edge, then grow each seed inside its grey-bounded soil box.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    templates = _extract_templates(g)

    for r0, c0, r1, c1, seed, color in _find_soil_regions(g):
        assert color in templates, "every soil seed color must appear in the legend"
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        mask = _grow_mask((inner_h, inner_w), seed, templates[color])
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[mask] = color

    return out


def _legend_block(color: int, offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    block = np.zeros((3, 3), dtype=int)
    block[1, 1] = color
    for dr, dc in offsets:
        block[1 + dr, 1 + dc] = LEGEND_MARK
    return block


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] < b[0] - gap
        or b[2] < a[0] - gap
        or a[3] < b[1] - gap
        or b[3] < a[1] - gap
    )


def _render_expected(
    inp: np.ndarray,
    region_specs: list[tuple[int, int, int, int, tuple[int, int], int]],
    templates: dict[int, tuple[tuple[int, int], ...]],
) -> np.ndarray:
    out = inp.copy()
    for r0, c0, inner_h, inner_w, seed, color in region_specs:
        mask = _grow_mask((inner_h, inner_w), seed, templates[color])
        patch = out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w]
        patch[mask] = color
    return out


def _generate_once() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    scene_r0 = 4
    scene_c0 = 4
    scene_h = height - scene_r0
    scene_w = width - scene_c0
    if scene_h < 10 or scene_w < 10:
        return None

    orientation = random.choice(["top", "left"])
    legend_span_limit = width if orientation == "top" else height
    max_by_legend = min(4, (legend_span_limit - 4 + 1) // 4)
    max_by_area = min(4, (scene_h * scene_w) // 36)
    max_colors = min(max_by_legend, max_by_area)
    if max_colors < 2:
        return None

    n_colors = random.randint(2, max_colors)
    colors = random.sample(list(SEED_COLORS), n_colors)
    templates = {
        color: offsets
        for color, offsets in zip(colors, random.sample(list(TEMPLATE_LIBRARY), n_colors), strict=False)
    }

    inp = np.zeros((height, width), dtype=int)

    legend_order = colors[:]
    random.shuffle(legend_order)
    span = 4 * n_colors - 1
    if orientation == "top":
        start = random.randint(4, width - span)
        for i, color in enumerate(legend_order):
            c0 = start + 4 * i
            inp[0:3, c0 : c0 + 3] = _legend_block(color, templates[color])
    else:
        start = random.randint(4, height - span)
        for i, color in enumerate(legend_order):
            r0 = start + 4 * i
            inp[r0 : r0 + 3, 0:3] = _legend_block(color, templates[color])

    n_regions = random.randint(n_colors, min(4, n_colors + 1))
    assigned_colors = colors[:]
    while len(assigned_colors) < n_regions:
        assigned_colors.append(random.choice(colors))
    random.shuffle(assigned_colors)

    region_specs: list[tuple[int, int, int, int, tuple[int, int], int]] = []
    used_boxes: list[tuple[int, int, int, int]] = []
    size_min, size_max = {2: (4, 8), 3: (3, 7), 4: (3, 6)}[n_regions]

    for color in assigned_colors:
        placed = False
        for _ in range(250):
            inner_h = random.randint(size_min, min(size_max, scene_h - 2))
            inner_w = random.randint(size_min, min(size_max, scene_w - 2))
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h > scene_h or outer_w > scene_w:
                continue

            r0 = random.randint(scene_r0, height - outer_h)
            c0 = random.randint(scene_c0, width - outer_w)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_overlap(box, other, gap=1) for other in used_boxes):
                continue

            seed_ok = False
            for _ in range(40):
                sr = random.randint(0, inner_h - 1)
                sc = random.randint(0, inner_w - 1)
                fill = _grow_mask((inner_h, inner_w), (sr, sc), templates[color])
                if int(fill.sum()) > 1:
                    seed_ok = True
                    break
            if not seed_ok:
                continue

            inp[r0 : r0 + outer_h, c0 : c0 + outer_w] = BG
            inp[r0, c0 : c0 + outer_w] = BORDER
            inp[r0 + outer_h - 1, c0 : c0 + outer_w] = BORDER
            inp[r0 : r0 + outer_h, c0] = BORDER
            inp[r0 : r0 + outer_h, c0 + outer_w - 1] = BORDER
            inp[r0 + 1 + sr, c0 + 1 + sc] = color

            used_boxes.append(box)
            region_specs.append((r0, c0, inner_h, inner_w, (sr, sc), color))
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(inp, region_specs, templates)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    if len(_find_soil_regions(inp)) != len(region_specs):
        return None

    if len(_extract_templates(inp)) != len(templates):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
