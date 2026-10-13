import random

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _background_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _legend_info(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    best = None  # (score, side, start, end)

    for side, col in (("left", 0), ("right", w - 1)):
        start = None
        for r in range(h + 1):
            filled = r < h and int(grid[r, col]) != bg
            if filled and start is None:
                start = r
            elif not filled and start is not None:
                end = r
                colors = [int(v) for v in grid[start:end, col]]
                score = (end - start, len(set(colors)) == len(colors), -start)
                cand = (score, side, start, end)
                if best is None or cand[0] > best[0]:
                    best = cand
                start = None

    assert best is not None, "expected a vertical legend on the left or right edge"
    _, side, start, end = best
    col = 0 if side == "left" else w - 1

    mapping: dict[int, int] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for idx, r in enumerate(range(start, end), start=1):
        mapping[idx] = int(grid[r, col])
        legend_mask[r, col] = True
    return mapping, legend_mask


def _foreground_components(grid: np.ndarray, bg: int, legend_mask: np.ndarray):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if legend_mask[r, c] or int(grid[r, c]) == bg or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not legend_mask[nr, nc]
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return comps


def _inside_cells(comp: dict[str, object]) -> set[tuple[int, int]]:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    if bh < 3 or bw < 3:
        return set()

    wall = np.zeros((bh, bw), dtype=bool)
    for r, c in coords:
        wall[r - r0, c - c0] = True

    seen = np.zeros((bh, bw), dtype=bool)
    stack = []
    for rr in range(bh):
        for cc in (0, bw - 1):
            if not wall[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))
    for cc in range(bw):
        for rr in (0, bh - 1):
            if not wall[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))

    while stack:
        cr, cc = stack.pop()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < bh and 0 <= nc < bw and not wall[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    inside = set()
    for rr in range(bh):
        for cc in range(bw):
            if not wall[rr, cc] and not seen[rr, cc]:
                inside.add((r0 + rr, c0 + cc))
    return inside


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend, legend_mask = _legend_info(g, bg)
    fg_comps = _foreground_components(g, bg, legend_mask)

    out = g.copy()
    for idx, comp in enumerate(fg_comps):
        inside = _inside_cells(comp)
        bg_inside = [cell for cell in inside if int(g[cell]) == bg]
        if not bg_inside:
            continue

        island_count = 0
        for other_idx, other in enumerate(fg_comps):
            if other_idx == idx or other["color"] != comp["color"]:
                continue
            if all(cell in inside for cell in other["coords"]):
                island_count += 1
        if island_count not in legend:
            continue
        fill_color = legend[island_count]
        for r, c in bg_inside:
            out[r, c] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each closed ring interior by looking up a border legend row indexed by the number of interior islands.

    Input structure:
    - The background is the dominant color.
    - Several same-colored closed ring borders appear away from the outer edge, and each ring encloses background cells plus some same-colored interior island components.
    - A vertical color legend sits on the leftmost or rightmost grid column, where the first legend row means "1 island", the second means "2 islands", and so on.

    Rule:
    - Detect the legend on the grid edge and read its colors from top to bottom as a count-to-color lookup table.
    - Find every background region that is fully enclosed by a ring and does not touch the grid boundary.
    - For each enclosed region, count how many connected components of the ring color touch that region; one of them is the enclosing border, so the remaining components are the interior islands.
    - Fill all background cells of that enclosed region with the legend color indexed by that island count, while leaving the border, islands, and legend unchanged.

    Key insight:
    - The interior same-colored pixels are not decoration: together with the enclosing border they form adjacent components, so "adjacent component count minus one" reveals the legend row used to color the cavity.
    """
    return _solve(grid)


COLORS = list(range(10))
MAX_ATTEMPTS = 400


def _choose_palette():
    bg = random.choice(COLORS)
    remaining = [c for c in COLORS if c != bg]
    border = random.choice(remaining)
    remaining = [c for c in remaining if c != border]
    legend_len = random.randint(2, 5)
    legend_colors = random.sample(remaining, legend_len)
    return bg, border, legend_colors


def _inner_size_options(count: int) -> list[tuple[int, int]]:
    options = []
    for ih in range(5, 10):
        for iw in range(5, 10):
            core_area = (ih - 2) * (iw - 2)
            if core_area <= count:
                continue
            if (core_area + 1) // 2 < count:
                continue
            options.append((ih, iw))
    return options


def _sample_islands(top: int, left: int, ih: int, iw: int, count: int) -> list[tuple[int, int]]:
    parity_first = random.choice([0, 1])
    for parity in (parity_first, 1 - parity_first):
        cells = [
            (top + 2 + r, left + 2 + c)
            for r in range(ih - 2)
            for c in range(iw - 2)
            if (r + c) % 2 == parity
        ]
        if len(cells) >= count:
            return random.sample(cells, count)
    raise AssertionError("checkerboard sampling must fit the requested island count")


def _paint_ring(grid: np.ndarray, border: int, top: int, left: int, ih: int, iw: int, islands: list[tuple[int, int]]):
    bottom = top + ih + 1
    right = left + iw + 1
    grid[top, left:right + 1] = border
    grid[bottom, left:right + 1] = border
    grid[top:bottom + 1, left] = border
    grid[top:bottom + 1, right] = border
    for r, c in islands:
        grid[r, c] = border


def _place_rings(grid: np.ndarray, expected: np.ndarray, bg: int, border: int, legend_colors: list[int]) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[h - 1, :] = True
    blocked[:, 0] = True
    blocked[:, w - 1] = True

    legend_side = random.choice(["left", "right"])
    legend_col = 0 if legend_side == "left" else w - 1
    pad_col = 1 if legend_side == "left" else w - 2
    blocked[:, legend_col] = True
    blocked[:, pad_col] = True

    legend_len = len(legend_colors)
    start_row = random.randint(0, h - legend_len)
    for idx, color in enumerate(legend_colors):
        grid[start_row + idx, legend_col] = color
        expected[start_row + idx, legend_col] = color

    max_rings = 3 if h * w >= 220 else 2
    ring_count = random.randint(1, max_rings)
    counts = [random.randint(1, legend_len) for _ in range(ring_count)]
    if ring_count > 1 and legend_len > 1 and len(set(counts)) == 1:
        counts[random.randrange(ring_count)] = 1 + (counts[0] % legend_len)
    random.shuffle(counts)

    for count in counts:
        dims = _inner_size_options(count)
        random.shuffle(dims)
        placed = False
        for ih, iw in dims:
            oh, ow = ih + 2, iw + 2
            row_starts = list(range(1, h - oh))
            col_starts = list(range(1, w - ow))
            random.shuffle(row_starts)
            random.shuffle(col_starts)
            for top in row_starts:
                for left in col_starts:
                    if blocked[top - 1 : top + oh + 1, left - 1 : left + ow + 1].any():
                        continue
                    islands = _sample_islands(top, left, ih, iw, count)
                    _paint_ring(grid, border, top, left, ih, iw, islands)
                    _paint_ring(expected, border, top, left, ih, iw, islands)
                    fill = legend_colors[count - 1]
                    for r in range(top + 1, top + ih + 1):
                        for c in range(left + 1, left + iw + 1):
                            if grid[r, c] == bg:
                                expected[r, c] = fill
                    blocked[top - 1 : top + oh + 1, left - 1 : left + ow + 1] = True
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return False
    return True


def _generate_case() -> np.ndarray | None:
    bg, border, legend_colors = _choose_palette()
    min_h = max(14, len(legend_colors) + 2)
    h = random.randint(min_h, 30)
    w = random.randint(14, 30)

    grid = np.full((h, w), bg, dtype=int)
    expected = grid.copy()
    if not _place_rings(grid, expected, bg, border, legend_colors):
        return None

    if _background_color(grid) != bg:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid seed287049989/1 input")
