import random

import numpy as np


BG = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            stack = [(int(r0), int(c0))]
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_colors(grid: np.ndarray, bg: int) -> list[int]:
    col = grid[:, -1]
    colors: list[int] = []
    r = 0
    while r < grid.shape[0]:
        if int(col[r]) == bg:
            r += 1
            continue
        color = int(col[r])
        start = r
        while r < grid.shape[0] and int(col[r]) == color:
            r += 1
        assert r - start == 2, "legend blocks must be 2x1 runs on the right edge"
        colors.append(color)
    assert colors, "right-edge legend must be present"
    return colors


def _grey_color(grid: np.ndarray, bg: int, legend_colors: list[int]) -> int:
    forbidden = {int(bg), *map(int, legend_colors)}
    vals, cnts = np.unique(grid, return_counts=True)
    candidates = [(int(cnt), int(val)) for val, cnt in zip(vals, cnts) if int(val) not in forbidden]
    assert candidates, "a non-legend enclosure color must be present"
    return max(candidates)[1]


def _enclosed_black_regions(subgrid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = subgrid.shape
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(subgrid == int(bg)):
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp):
            holes.append(comp)
    return holes


def _legend_shift(legend_len: int, enclosed_counts: list[int]) -> int:
    best = None
    for shift in (0, -1):
        idxs = [count + shift for count in enclosed_counts]
        if min(idxs) < 0 or max(idxs) >= legend_len:
            continue
        unused = legend_len - (max(idxs) + 1)
        # Prefer the convention that uses the legend most tightly; ties keep the direct count.
        cand = (unused, 0 if shift == 0 else 1, shift)
        if best is None or cand < best:
            best = cand
    assert best is not None, "legend does not cover the observed enclosure counts"
    return int(best[2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey enclosure with the legend color selected by its enclosed black-hole count.

    Input structure:
    - Black background.
    - Several 4-connected grey closed enclosures whose interiors contain one or more enclosed black regions.
    - A right-edge vertical legend made of 2x1 color blocks, stacked top-to-bottom with black gaps; the kth block gives the fill color for hole count k.

    Rule:
    - Keep all grey enclosure pixels unchanged.
    - For each grey closed region, count the enclosed black 4-connected holes inside its bounding box.
    - Use that count as an index into the right-edge legend and select the corresponding fill color.
    - Fill every enclosed black pixel of that grey region with the selected color.
    - Leave the legend unchanged.

    Key insight:
    - The objects are distinguished by topology rather than shape, and the right-edge column is an ordered lookup table keyed by the number of enclosed black regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG
    legend = _legend_colors(g, bg)
    grey = _grey_color(g, bg, legend)

    comps = _components(g == int(grey))
    assert comps, "at least one grey enclosure must be present"

    comp_origins: list[tuple[int, int]] = []
    enclosed_per_comp: list[list[list[tuple[int, int]]]] = []
    enclosed_counts: list[int] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        subgrid = g[r0 : r1 + 1, c0 : c1 + 1]
        holes = _enclosed_black_regions(subgrid, bg)
        comp_origins.append((r0, c0))
        enclosed_per_comp.append(holes)
        enclosed_counts.append(len(holes))

    shift = _legend_shift(len(legend), enclosed_counts)
    out = g.copy()
    for (r0, c0), holes, count in zip(comp_origins, enclosed_per_comp, enclosed_counts):
        fill_color = int(legend[count + shift])
        for hole in holes:
            for rr, cc in hole:
                out[int(r0 + rr), int(c0 + cc)] = fill_color

    return out


def _make_patch(extra_holes: int) -> np.ndarray:
    orientation = random.choice(["vertical", "horizontal"])
    chambers = [1 + int(random.random() < 0.35) for _ in range(extra_holes + 1)]
    cross = random.randint(2, 4)

    if orientation == "vertical":
        inner_h = cross
        inner_w = sum(chambers) + extra_holes
        patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        col = 1 + chambers[0]
        for width in chambers[1:]:
            patch[1:-1, col] = GREY
            col += 1 + width
        return patch

    inner_h = sum(chambers) + extra_holes
    inner_w = cross
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    row = 1 + chambers[0]
    for height in chambers[1:]:
        patch[row, 1:-1] = GREY
        row += 1 + height
    return patch


def _paint(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    region[patch != BG] = patch[patch != BG]


def _can_place(occupied: np.ndarray, h: int, w: int, top: int, left: int) -> bool:
    H, W = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_patches(patches: list[np.ndarray], height: int, width: int) -> list[tuple[np.ndarray, int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[tuple[np.ndarray, int, int]] = []
    order = sorted(patches, key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    for patch in order:
        h, w = patch.shape
        if h + 2 > height or w + 2 > width:
            return None
        positions = [
            (top, left)
            for top in range(1, height - h)
            for left in range(1, width - w)
        ]
        random.shuffle(positions)
        spot = None
        for top, left in positions:
            if _can_place(occupied, h, w, top, left):
                spot = (top, left)
                break
        if spot is None:
            return None
        top, left = spot
        occupied[top : top + h, left : left + w] = True
        placed.append((patch, top, left))
    return placed


def _generate_once() -> np.ndarray | None:
    max_count = random.randint(1, 4)
    counts = list(range(max_count + 1))
    random.shuffle(counts)
    patches = [_make_patch(count) for count in counts]

    legend_colors = random.sample([c for c in range(1, 10) if c != GREY], len(counts))
    legend_height = 3 * len(legend_colors) - 1

    max_patch_h = max(int(patch.shape[0]) for patch in patches)
    max_patch_w = max(int(patch.shape[1]) for patch in patches)
    min_h = max(legend_height, max_patch_h + 2, 12)
    min_w = max(max_patch_w + 4, 14)
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    object_width = width - 2
    placed = _place_patches(patches, height, object_width)
    if placed is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for patch, top, left in placed:
        _paint(grid, patch, top, left)

    legend_top = random.randint(0, height - legend_height)
    row = legend_top
    for color in legend_colors:
        grid[row : row + 2, -1] = int(color)
        row += 3

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise AssertionError("generate_input: failed to build a valid grid")
