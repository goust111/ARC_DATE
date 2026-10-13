import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
SEED_COLORS = (BLUE, RED, GREEN)

MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 300
MAX_REGION_ATTEMPTS = 120
MAX_LAYOUT_ATTEMPTS = 80
MIN_REGION_COUNT = 2
MAX_REGION_COUNT = 4
MIN_GAP = 2
MAX_GAP = 4
MIN_MARGIN = 1
MAX_MARGIN = 3


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _is_connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components(mask)) == 1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _step_from_seed_count(seed_count: int) -> int:
    if seed_count == 1:
        return 1
    if seed_count == 2:
        return 2
    return 3


def _should_fill(color: int, step: int, row: int, col: int, top: int, left: int) -> bool:
    rr = row - top
    cc = col - left
    if color == RED:
        return (rr // step) % 2 == 0
    if color == BLUE:
        return (cc // step) % 2 == 0
    return ((rr // step) + (cc // step)) % 2 == 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray-enclosed region with a seed-controlled stripe or checker pattern.

    Input structure:
    - The background is black.
    - One-pixel-wide gray closed curves surround one or more irregular enclosed regions.
    - Each enclosed region contains one or more same-color single-pixel seeds, using only red, blue, or green.

    Rule:
    - Treat every 4-connected non-gray component that does not touch the grid border as one enclosed region.
    - Read the region's seed color and seed count.
    - Red seeds fill the region with horizontal stripes, blue seeds with vertical stripes, and green seeds with a checkerboard.
    - One seed uses spacing 1, two seeds spacing 2, and three or more seeds spacing 3; align the pattern to the region bbox top-left.
    - Overwrite seed pixels with the pattern and keep every gray boundary pixel unchanged.

    Key insight:
    - The gray outlines segment the grid into closed interior components, while seed color selects the pattern family and seed multiplicity selects the pattern period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for comp in _components(g != GRAY):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in comp):
            continue

        colored = [(r, c) for r, c in comp if int(g[r, c]) != BG]
        if not colored:
            continue

        colors = {int(g[r, c]) for r, c in colored}
        assert colors <= set(SEED_COLORS), "region markers must be red/blue/green seeds"
        assert len(colors) == 1, "all seeds inside one region must share a color"

        seed_color = next(iter(colors))
        step = _step_from_seed_count(len(colored))
        top = min(r for r, _c in comp)
        left = min(c for _r, c in comp)

        for r, c in comp:
            out[r, c] = seed_color if _should_fill(seed_color, step, r, c, top, left) else BG

    return out


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if any(not mask[nr, nc] for nr, nc in _neighbors4(rr, cc, height, width)):
            cells.append((rr, cc))
            continue
        if rr in (0, height - 1) or cc in (0, width - 1):
            cells.append((rr, cc))
    return cells


def _sample_seed_positions(mask: np.ndarray, seed_count: int) -> list[tuple[int, int]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    for _ in range(60):
        random.shuffle(coords)
        chosen: list[tuple[int, int]] = []
        for r, c in coords:
            if all(max(abs(r - sr), abs(c - sc)) > 1 for sr, sc in chosen):
                chosen.append((r, c))
                if len(chosen) == seed_count:
                    return chosen
    return None


def _pattern_has_both_states(mask: np.ndarray, color: int, step: int) -> bool:
    values = {
        _should_fill(int(color), int(step), int(r), int(c), 0, 0)
        for r, c in np.argwhere(mask)
    }
    return values == {False, True}


def _generate_region_mask(color: int, seed_count: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    step = _step_from_seed_count(seed_count)
    min_h = max(4, 2 * step) if color in (RED, GREEN) else 4
    min_w = max(4, 2 * step) if color in (BLUE, GREEN) else 4

    for _ in range(MAX_REGION_ATTEMPTS):
        height = random.randint(min_h, 8)
        width = random.randint(min_w, 8)
        mask = np.ones((height, width), dtype=bool)
        min_area = max(8, seed_count * 3)
        remove_budget = random.randint(0, max(1, (height * width) // 4))

        for _ in range(remove_budget):
            if int(mask.sum()) <= min_area:
                break
            candidates: list[tuple[int, int]] = []
            for r, c in _boundary_cells(mask):
                mask[r, c] = False
                if _is_connected(mask):
                    candidates.append((r, c))
                mask[r, c] = True
            if not candidates:
                break
            rr, cc = random.choice(candidates)
            mask[rr, cc] = False

        mask = _crop_mask(mask)
        if mask.shape[0] < min_h or mask.shape[1] < min_w:
            continue
        if not _pattern_has_both_states(mask, color, step):
            continue

        seeds = _sample_seed_positions(mask, seed_count)
        if seeds is None:
            continue
        return mask, seeds

    return None


def _build_region(color: int, seed_count: int) -> dict | None:
    generated = _generate_region_mask(color, seed_count)
    if generated is None:
        return None
    mask, seeds = generated

    inner_h, inner_w = mask.shape
    bubble = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    inner = np.zeros_like(bubble, dtype=bool)
    inner[1 : inner_h + 1, 1 : inner_w + 1] = mask

    shell = np.zeros_like(inner, dtype=bool)
    for r, c in np.argwhere(inner):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < inner.shape[0] and 0 <= nc < inner.shape[1] and not inner[nr, nc]:
                    shell[nr, nc] = True

    bubble[shell] = GRAY
    for r, c in seeds:
        bubble[r + 1, c + 1] = color

    local_cells = [(int(r) + 1, int(c) + 1) for r, c in np.argwhere(mask)]
    return {
        "bubble": bubble,
        "color": int(color),
        "seed_count": int(seed_count),
        "step": int(_step_from_seed_count(seed_count)),
        "local_cells": local_cells,
    }


def _sample_layout(regions: list[dict]) -> tuple[int, int, list[tuple[int, int, int]]] | None:
    count = len(regions)
    indices = list(range(count))
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        row_count = 2 if count == 4 else random.choice([1, 2])
        if row_count == 1:
            rows = [random.sample(indices, len(indices))]
        else:
            order = random.sample(indices, len(indices))
            split = random.randint(1, count - 1)
            rows = [order[:split], order[split:]]
            if not rows[0] or not rows[1]:
                continue

        row_specs = []
        for row in rows:
            widths = [regions[idx]["bubble"].shape[1] for idx in row]
            heights = [regions[idx]["bubble"].shape[0] for idx in row]
            gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(max(0, len(row) - 1))]
            left_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
            right_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
            total_width = left_margin + sum(widths) + sum(gaps) + right_margin
            row_specs.append(
                {
                    "row": row,
                    "gaps": gaps,
                    "left_margin": left_margin,
                    "width": total_width,
                    "height": max(heights),
                }
            )

        top_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
        bottom_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
        row_gap = random.randint(MIN_GAP, MAX_GAP) if row_count == 2 else 0
        grid_h = top_margin + sum(spec["height"] for spec in row_specs) + row_gap + bottom_margin
        grid_w = max(spec["width"] for spec in row_specs)
        if grid_h > MAX_GRID_SIZE or grid_w > MAX_GRID_SIZE:
            continue

        placements: list[tuple[int, int, int]] = []
        current_top = top_margin
        for spec_index, spec in enumerate(row_specs):
            row_offset = random.randint(0, grid_w - spec["width"])
            current_left = row_offset + spec["left_margin"]
            for j, region_idx in enumerate(spec["row"]):
                bubble_h, bubble_w = regions[region_idx]["bubble"].shape
                vertical_slack = spec["height"] - bubble_h
                bubble_top = current_top + random.randint(0, vertical_slack)
                placements.append((region_idx, bubble_top, current_left))
                current_left += bubble_w
                if j + 1 < len(spec["row"]):
                    current_left += spec["gaps"][j]
            current_top += spec["height"]
            if spec_index + 1 < len(row_specs):
                current_top += row_gap

        return grid_h, grid_w, placements

    return None


def _render_input_and_expected(regions: list[dict], layout: tuple[int, int, list[tuple[int, int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    grid_h, grid_w, placements = layout
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full_like(inp, BG)

    for region_idx, top, left in placements:
        bubble = regions[region_idx]["bubble"]
        bubble_h, bubble_w = bubble.shape
        inp[top : top + bubble_h, left : left + bubble_w] = np.where(
            bubble != BG,
            bubble,
            inp[top : top + bubble_h, left : left + bubble_w],
        )
        expected[top : top + bubble_h, left : left + bubble_w] = np.where(
            bubble == GRAY,
            GRAY,
            expected[top : top + bubble_h, left : left + bubble_w],
        )

    for region_idx, top, left in placements:
        region = regions[region_idx]
        bbox_top = top + 1
        bbox_left = left + 1
        for lr, lc in region["local_cells"]:
            gr = top + lr
            gc = left + lc
            expected[gr, gc] = (
                region["color"]
                if _should_fill(region["color"], region["step"], gr, gc, bbox_top, bbox_left)
                else BG
            )

    return inp, expected


def _sample_region_specs(region_count: int) -> list[tuple[int, int]]:
    combos = [(color, bucket) for color in (RED, BLUE, GREEN) for bucket in (1, 2, 3)]
    return random.sample(combos, region_count)


def _bucket_to_seed_count(bucket: int) -> int:
    if bucket < 3:
        return bucket
    return random.choice([3, 4])


def _generate_attempt() -> np.ndarray | None:
    region_count = random.randint(MIN_REGION_COUNT, MAX_REGION_COUNT)
    specs = _sample_region_specs(region_count)
    regions: list[dict] = []
    for color, bucket in specs:
        region = _build_region(color, _bucket_to_seed_count(bucket))
        if region is None:
            return None
        regions.append(region)

    layout = _sample_layout(regions)
    if layout is None:
        return None

    inp, expected = _render_input_and_expected(regions, layout)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    enclosed = [
        comp
        for comp in _components(inp != GRAY)
        if not any(r == 0 or c == 0 or r == inp.shape[0] - 1 or c == inp.shape[1] - 1 for r, c in comp)
    ]
    if len(enclosed) != region_count:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ARC input for puzzle 7")
