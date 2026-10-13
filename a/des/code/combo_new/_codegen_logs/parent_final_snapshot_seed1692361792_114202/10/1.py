import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4, 7)  # blue, red, green, yellow, orange
LEGEND_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
LEGEND_LEVELS = 5
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    candidate_mask = edge & (grid != BLACK) & (grid != GREY)

    candidates: list[list[tuple[int, int]]] = []
    for comp in _components(candidate_mask):
        rows = sorted(r for r, _ in comp)
        cols = {c for _, c in comp}
        if len(cols) != 1:
            continue
        col = next(iter(cols))
        if col not in (0, w - 1):
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        if not (rows[0] == 0 or rows[-1] == h - 1):
            continue
        candidates.append(comp)

    assert len(candidates) == 1, "expected exactly one corner legend strip"
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, c in candidates[0]:
        legend_mask[r, c] = True

    legend_rows = sorted(r for r, _ in candidates[0])
    legend_col = candidates[0][0][1]
    legend_colors = [int(grid[r, legend_col]) for r in legend_rows]
    return legend_mask, legend_colors


def _enclosed_regions(grid: np.ndarray, legend_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    passable = (grid != GREY) & (~legend_mask)
    h, w = grid.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed region's foreground pixels by looking up the region's color-count in the corner legend.

    Input structure:
    - The background is black.
    - Several irregular closed grey borders each surround one enclosed region containing black cells and scattered foreground pixels.
    - A vertical legend strip sits in one corner on the outer edge; its k-th row color encodes the repaint color for regions that contain exactly k distinct foreground colors.

    Rule:
    - Detect the corner legend and read its colors from top to bottom.
    - Treat grey pixels as walls, then find every non-grey connected component that does not touch the grid boundary; these are the enclosed regions.
    - For each enclosed region, count how many distinct non-black colors appear inside it and replace every non-black pixel in that region with the legend color from row k.
    - Keep black interior cells, grey borders, and the legend unchanged.

    Key insight:
    - The output color is not tied to any specific interior color; it is determined only by the number of different foreground colors inside each closed region, using the corner strip as a lookup table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_mask, legend_colors = _find_legend(g)
    out = g.copy()

    for region in _enclosed_regions(g, legend_mask):
        colors = {int(g[r, c]) for r, c in region if int(g[r, c]) != BLACK}
        if not colors:
            continue
        level = len(colors)
        assert 1 <= level <= len(legend_colors), "region color count exceeds legend length"
        repaint = legend_colors[level - 1]
        for r, c in region:
            if int(g[r, c]) != BLACK:
                out[r, c] = repaint

    return out


def _pick_spaced_cells(rng: random.Random, cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    order = cells[:]
    rng.shuffle(order)
    picked: list[tuple[int, int]] = []
    for cell in order:
        if all(abs(cell[0] - pr) + abs(cell[1] - pc) > 1 for pr, pc in picked):
            picked.append(cell)
            if len(picked) == count:
                return picked
    if len(cells) < count:
        return None
    return rng.sample(cells, count)


def _build_region_patch(rng: random.Random, distinct_count: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    for _ in range(200):
        inner_h = rng.randint(4, 6)
        inner_w = rng.randint(4, 6)
        if (inner_h - 2) * (inner_w - 2) < distinct_count:
            continue

        patch_h, patch_w = inner_h + 4, inner_w + 4
        top, left = 2, 2
        interior = np.zeros((patch_h, patch_w), dtype=bool)
        interior[top:top + inner_h, left:left + inner_w] = True

        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        for side in sides[:rng.randint(0, 3)]:
            if side in ("top", "bottom"):
                row = top - 1 if side == "top" else top + inner_h
                seg_w = rng.randint(1, inner_w)
                seg_c = rng.randint(left, left + inner_w - seg_w)
                interior[row, seg_c:seg_c + seg_w] = True
            else:
                col = left - 1 if side == "left" else left + inner_w
                seg_h = rng.randint(1, inner_h)
                seg_r = rng.randint(top, top + inner_h - seg_h)
                interior[seg_r:seg_r + seg_h, col] = True

        core: list[tuple[int, int]] = []
        for r, c in map(tuple, np.argwhere(interior)):
            if all(0 <= nr < patch_h and 0 <= nc < patch_w and interior[nr, nc] for nr, nc in _neighbors4(r, c)):
                core.append((int(r), int(c)))
        if len(core) < distinct_count:
            continue

        border = np.zeros((patch_h, patch_w), dtype=bool)
        for r, c in map(tuple, np.argwhere(interior)):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < patch_h and 0 <= nc < patch_w and not interior[nr, nc]:
                    border[nr, nc] = True

        patch = np.zeros((patch_h, patch_w), dtype=int)
        patch[border] = GREY

        colors = rng.sample(SOURCE_COLORS, distinct_count)
        chosen = _pick_spaced_cells(rng, core, distinct_count)
        if chosen is None:
            continue
        used = set(chosen)
        for (r, c), color in zip(chosen, colors):
            patch[r, c] = color

        extra_cells = [cell for cell in core if cell not in used]
        rng.shuffle(extra_cells)
        extra_count = rng.randint(0, min(3, len(extra_cells)))
        for r, c in extra_cells[:extra_count]:
            patch[r, c] = rng.choice(colors)

        recolor_coords = [tuple(map(int, rc)) for rc in np.argwhere((patch != BLACK) & (patch != GREY))]
        return patch, recolor_coords

    raise RuntimeError("failed to build region patch")


def _legend_cells(height: int, width: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(r, 0) for r in range(LEGEND_LEVELS)]
    if corner == "tr":
        return [(r, width - 1) for r in range(LEGEND_LEVELS)]
    if corner == "bl":
        return [(height - LEGEND_LEVELS + r, 0) for r in range(LEGEND_LEVELS)]
    if corner == "br":
        return [(height - LEGEND_LEVELS + r, width - 1) for r in range(LEGEND_LEVELS)]
    raise ValueError(f"unknown corner {corner}")


def _arrange_patches(
    rng: random.Random,
    patches: list[np.ndarray],
) -> tuple[int, int, dict[int, tuple[int, int]]]:
    count = len(patches)
    for _ in range(200):
        cols = rng.choice((2, 3))
        rows = (count + cols - 1) // cols
        order = list(range(count))
        rng.shuffle(order)

        slots: list[list[int | None]] = [[None for _ in range(cols)] for _ in range(rows)]
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < count:
                    slots[r][c] = order[idx]
                    idx += 1

        row_heights = [
            max(patches[p].shape[0] for p in row if p is not None)
            for row in slots
        ]
        col_widths = []
        for c in range(cols):
            present = [patches[slots[r][c]].shape[1] for r in range(rows) if slots[r][c] is not None]
            col_widths.append(max(present))

        gap = rng.choice((1, 2))
        used_h = sum(row_heights) + gap * (rows - 1)
        used_w = sum(col_widths) + gap * (cols - 1)
        min_h = max(LEGEND_LEVELS, used_h + 2)
        min_w = used_w + 2
        if min_h > 30 or min_w > 30:
            continue

        height = rng.randint(min_h, min(30, min_h + 4))
        width = rng.randint(min_w, min(30, min_w + 4))
        top_margin = rng.randint(1, height - used_h - 1)
        left_margin = rng.randint(1, width - used_w - 1)

        positions: dict[int, tuple[int, int]] = {}
        base_r = top_margin
        for r in range(rows):
            base_c = left_margin
            for c in range(cols):
                patch_idx = slots[r][c]
                cell_h = row_heights[r]
                cell_w = col_widths[c]
                if patch_idx is not None:
                    patch_h, patch_w = patches[patch_idx].shape
                    dr = rng.randint(0, cell_h - patch_h)
                    dc = rng.randint(0, cell_w - patch_w)
                    positions[patch_idx] = (base_r + dr, base_c + dc)
                base_c += cell_w + gap
            base_r += row_heights[r] + gap

        if all(
            1 <= r0 and 1 <= c0 and r0 + patches[i].shape[0] < height and c0 + patches[i].shape[1] < width
            for i, (r0, c0) in positions.items()
        ):
            return height, width, positions

    raise RuntimeError("failed to arrange patches")


def _expected_output(
    grid: np.ndarray,
    legend_colors: list[int],
    records: list[tuple[int, list[tuple[int, int]]]],
) -> np.ndarray:
    out = grid.copy()
    for distinct_count, coords in records:
        repaint = legend_colors[distinct_count - 1]
        for r, c in coords:
            out[r, c] = repaint
    return out


def _validate_instance(
    grid: np.ndarray,
    legend_colors: list[int],
    expected: np.ndarray,
    required_count: int,
) -> bool:
    legend_mask, found_legend = _find_legend(grid)
    if found_legend != legend_colors:
        return False

    counts: list[int] = []
    for region in _enclosed_regions(grid, legend_mask):
        colors = {int(grid[r, c]) for r, c in region if int(grid[r, c]) != BLACK}
        if not colors:
            return False
        counts.append(len(colors))
        for r, c in region:
            if int(grid[r, c]) == BLACK:
                continue
            if any(
                0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == GREY
                for nr, nc in _neighbors4(r, c)
            ):
                return False

    if len(counts) != required_count:
        return False
    for level in range(1, LEGEND_LEVELS + 1):
        if level not in counts:
            return False
    return np.array_equal(main(grid), expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    extra_count = rng.randint(0, 2)
    region_counts = list(range(1, LEGEND_LEVELS + 1))
    for _ in range(extra_count):
        region_counts.append(rng.randint(1, LEGEND_LEVELS))
    rng.shuffle(region_counts)

    patches: list[np.ndarray] = []
    patch_records: list[tuple[int, list[tuple[int, int]]]] = []
    for distinct_count in region_counts:
        patch, coords = _build_region_patch(rng, distinct_count)
        patches.append(patch)
        patch_records.append((distinct_count, coords))

    try:
        height, width, positions = _arrange_patches(rng, patches)
    except RuntimeError:
        return None

    grid = np.zeros((height, width), dtype=int)
    records: list[tuple[int, list[tuple[int, int]]]] = []
    for idx, patch in enumerate(patches):
        r0, c0 = positions[idx]
        patch_h, patch_w = patch.shape
        target = grid[r0:r0 + patch_h, c0:c0 + patch_w]
        if np.any(((target != BLACK) & (patch != BLACK))):
            return None
        target[:] = np.where(patch != BLACK, patch, target)
        colored_abs = [(r0 + r, c0 + c) for r, c in patch_records[idx][1]]
        records.append((patch_records[idx][0], colored_abs))

    legend_colors = rng.sample(LEGEND_COLOR_POOL, LEGEND_LEVELS)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    for (r, c), color in zip(_legend_cells(height, width, corner), legend_colors):
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = color

    expected = _expected_output(grid, legend_colors, records)
    if not _validate_instance(grid, legend_colors, expected, len(region_counts)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
