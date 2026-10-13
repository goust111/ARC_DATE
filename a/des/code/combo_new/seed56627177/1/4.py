from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GRID = 30
GEN_ATTEMPTS = 400
PAIR_COUNTS = (2, 4)
RECT_SIDE_RANGE = (2, 6)
PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    return _bbox(coords)


def _parse_legend(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, _ = grid.shape
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend: list[tuple[int, int]] = []
    r = 0
    while r < h:
        if int(grid[r, 0]) == BG:
            r += 1
            continue
        assert r + 1 < h and int(grid[r + 1, 0]) != BG, "legend entry must be a vertical 2-pixel pair"
        if r > 0:
            assert int(grid[r - 1, 0]) == BG, "legend entries must be separated"
        top = int(grid[r, 0])
        bottom = int(grid[r + 1, 0])
        assert top != bottom, "legend colors inside one pair must differ"
        legend.append((top, bottom))
        legend_mask[r, 0] = True
        legend_mask[r + 1, 0] = True
        r += 2
        if r < h and int(grid[r, 0]) != BG:
            raise AssertionError("legend entries must be separated by a blank row")
    assert legend, "missing legend"
    return legend, legend_mask


def _extract_fragments(grid: np.ndarray, legend_mask: np.ndarray) -> list[dict]:
    comps = _components((grid != BG) & (~legend_mask))
    fragments: list[dict] = []
    for coords in comps:
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each fragment must be monochrome"
        mask = _coords_to_mask(coords)
        fragments.append(
            {
                "coords": coords,
                "color": int(colors[0]),
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    return fragments


def _row_or_col_order(first: np.ndarray, second: np.ndarray, *, by_rows: bool) -> int | None:
    line_count = first.shape[0] if by_rows else first.shape[1]
    relation: int | None = None
    evidence = 0
    for idx in range(line_count):
        a = np.flatnonzero(first[idx] if by_rows else first[:, idx])
        b = np.flatnonzero(second[idx] if by_rows else second[:, idx])
        if a.size == 0 or b.size == 0:
            continue
        if int(a[-1]) < int(b[0]):
            cur = -1
        elif int(b[-1]) < int(a[0]):
            cur = 1
        else:
            return None
        evidence += 1
        if relation is None:
            relation = cur
        elif relation != cur:
            return None
    return relation if evidence > 0 else None


def _classify_relation(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[str, int]:
    lr = _row_or_col_order(mask_a, mask_b, by_rows=True)
    if lr is not None:
        return "lr", lr
    tb = _row_or_col_order(mask_a, mask_b, by_rows=False)
    if tb is not None:
        return "tb", tb

    a_center = np.argwhere(mask_a).mean(axis=0)
    b_center = np.argwhere(mask_b).mean(axis=0)
    row_gap = float(a_center[0] - b_center[0])
    col_gap = float(a_center[1] - b_center[1])
    if abs(col_gap) >= abs(row_gap):
        return "lr", -1 if col_gap < 0 else 1
    return "tb", -1 if row_gap < 0 else 1


def _fit_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> dict | None:
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    total = area_a + area_b
    ha, wa = mask_a.shape

    for rotate_b in (False, True):
        placed_b_mask = np.rot90(mask_b, 2) if rotate_b else mask_b
        hb, wb = placed_b_mask.shape
        for rect_h in range(1, total + 1):
            if total % rect_h != 0:
                continue
            rect_w = total // rect_h
            if rect_h < max(ha, hb) or rect_w < max(wa, wb):
                continue
            for ra in range(rect_h - ha + 1):
                for ca in range(rect_w - wa + 1):
                    placed_a = np.zeros((rect_h, rect_w), dtype=bool)
                    placed_a[ra : ra + ha, ca : ca + wa] = mask_a
                    complement = ~placed_a
                    if int(complement.sum()) != area_b:
                        continue
                    r0, c0, r1, c1 = _mask_bbox(complement)
                    if (r1 - r0 + 1, c1 - c0 + 1) != (hb, wb):
                        continue
                    if not np.array_equal(complement[r0 : r1 + 1, c0 : c1 + 1], placed_b_mask):
                        continue
                    placed_b = np.zeros_like(placed_a)
                    placed_b[r0 : r1 + 1, c0 : c1 + 1] = placed_b_mask
                    relation, order = _classify_relation(placed_a, placed_b)
                    return {
                        "rect_shape": (rect_h, rect_w),
                        "placed_masks": (placed_a, placed_b),
                        "relation": relation,
                        "order": order,
                        "area": total,
                    }
    return None


def _match_graph(fragments: list[dict]) -> dict[tuple[int, int], dict]:
    graph: dict[tuple[int, int], dict] = {}
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            info = _fit_pair(fragments[i]["mask"], fragments[j]["mask"])
            if info is not None:
                graph[(i, j)] = info
    return graph


def _solve_matching(fragments: list[dict]) -> list[tuple[int, int, dict]]:
    graph = _match_graph(fragments)
    n = len(fragments)
    used = [False] * n
    solution: list[tuple[int, int, dict]] = []

    def dfs() -> bool:
        try:
            i = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            return True
        used[i] = True
        for j in range(i + 1, n):
            key = (i, j)
            if used[j] or key not in graph:
                continue
            used[j] = True
            solution.append((i, j, graph[key]))
            if dfs():
                return True
            solution.pop()
            used[j] = False
        used[i] = False
        return False

    if not dfs():
        raise AssertionError("could not pair fragments into rectangles")
    return solution


def _pair_to_patch(info: dict, top_color: int, bottom_color: int) -> np.ndarray:
    rect_h, rect_w = info["rect_shape"]
    placed_a, placed_b = info["placed_masks"]
    patch = np.zeros((rect_h, rect_w), dtype=int)
    if info["order"] == -1:
        first_mask, second_mask = placed_a, placed_b
    else:
        first_mask, second_mask = placed_b, placed_a
    patch[first_mask] = int(top_color)
    patch[second_mask] = int(bottom_color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary fragments into rectangles, recolor them from the left-edge legend, and pack the finished rectangles by area.

    Input structure:
    - The background is black.
    - Along the left edge, one column contains vertically stacked legend pairs:
      each entry is two adjacent single pixels with different colors, and entries
      are separated by one black row.
    - Away from the legend, the non-black cells form several monochrome 4-connected
      irregular fragments.

    Rule:
    - Pair the scattered fragments so that each pair exactly tiles a filled rectangle;
      one fragment may need a 180-degree rotation.
    - Sort the reconstructed rectangles by area from small to large, and match them
      to the legend entries from top to bottom.
    - If a pair forms a left/right split, recolor the left fragment with the legend's
      upper color and the right fragment with the lower color. If it forms an
      upper/lower split, use the upper color for the top fragment and the lower
      color for the bottom fragment.
    - Output only the recolored rectangles, packed left to right in that area order,
      with one black column between them and their bottoms aligned.

    Key insight:
    - The solution requires three linked deductions: complementary shape pairing,
      area-based alignment with the edge legend, and position-based assignment of
      the legend's two colors within each assembled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    legend, legend_mask = _parse_legend(g)
    fragments = _extract_fragments(g, legend_mask)
    assert len(fragments) == 2 * len(legend), "fragment count must match legend count"

    matches = _solve_matching(fragments)
    matches.sort(key=lambda item: (item[2]["area"], item[2]["rect_shape"][0], item[2]["rect_shape"][1]))
    assert len(matches) == len(legend), "matching count must equal legend count"

    patches: list[np.ndarray] = []
    for idx, (_, _, info) in enumerate(matches):
        top_color, bottom_color = legend[idx]
        patches.append(_pair_to_patch(info, top_color, bottom_color))

    out_h = max(patch.shape[0] for patch in patches)
    out_w = sum(patch.shape[1] for patch in patches) + len(patches) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for patch in patches:
        h, w = patch.shape
        out[out_h - h :, col : col + w] = patch
        col += w + 1
    return out


def _area_to_dims() -> dict[int, list[tuple[int, int]]]:
    mapping: dict[int, list[tuple[int, int]]] = {}
    lo, hi = RECT_SIDE_RANGE
    for h in range(lo, hi + 1):
        for w in range(lo, hi + 1):
            mapping.setdefault(h * w, []).append((h, w))
    return mapping


AREA_TO_DIMS = _area_to_dims()


def _sample_legend_pairs(count: int) -> list[tuple[int, int]]:
    options = [(a, b) for a in COLORS for b in COLORS if a != b]
    return random.sample(options, count)


def _sample_rectangles(count: int) -> list[tuple[int, int]]:
    areas = random.sample(sorted(AREA_TO_DIMS), count)
    rects = [(area, random.choice(AREA_TO_DIMS[area])) for area in areas]
    rects.sort(key=lambda item: item[0])
    dims = [dim for _, dim in rects]
    out_width = sum(w for _, w in dims) + count - 1
    if out_width > MAX_GRID:
        return []
    return dims


def _build_split_masks(rect_h: int, rect_w: int, relation: str) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        if relation == "lr":
            cuts = [random.randint(1, rect_w - 1) for _ in range(rect_h)]
            if rect_h > 1 and rect_w > 2 and len(set(cuts)) == 1 and random.random() < 0.75:
                continue
            first = np.zeros((rect_h, rect_w), dtype=bool)
            for r, cut in enumerate(cuts):
                first[r, :cut] = True
        else:
            cuts = [random.randint(1, rect_h - 1) for _ in range(rect_w)]
            if rect_w > 1 and rect_h > 2 and len(set(cuts)) == 1 and random.random() < 0.75:
                continue
            first = np.zeros((rect_h, rect_w), dtype=bool)
            for c, cut in enumerate(cuts):
                first[:cut, c] = True
        second = ~first
        if min(int(first.sum()), int(second.sum())) < 2:
            continue
        return first, second
    raise RuntimeError("failed to build split masks")


def _can_place(mask: np.ndarray, occupied: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    coords = np.argwhere(mask)
    for rr, cc in coords:
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        if occupied[gr, gc]:
            return False
        for nr in range(max(0, gr - 1), min(occupied.shape[0], gr + 2)):
            for nc in range(max(0, gc - 1), min(occupied.shape[1], gc + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _paint_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    coords = np.argwhere(mask)
    for rr, cc in coords:
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        grid[gr, gc] = int(color)
        occupied[gr, gc] = True


def _scatter_fragments(height: int, width: int, legend_pairs: list[tuple[int, int]], pieces: list[dict]) -> np.ndarray | None:
    legend_height = 3 * len(legend_pairs) - 1
    if height < legend_height:
        return None
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    start_row = random.randint(0, height - legend_height)
    for idx, (top, bottom) in enumerate(legend_pairs):
        r = start_row + 3 * idx
        grid[r, 0] = int(top)
        grid[r + 1, 0] = int(bottom)
        occupied[r, 0] = True
        occupied[r + 1, 0] = True
    occupied[:, 1] = True

    order = list(range(len(pieces)))
    order.sort(key=lambda idx: pieces[idx]["area"], reverse=True)
    random.shuffle(order)
    order.sort(key=lambda idx: pieces[idx]["area"], reverse=True)

    for idx in order:
        mask = pieces[idx]["mask"]
        color = int(pieces[idx]["color"])
        h, w = mask.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - h)
            c0 = random.randint(2, width - w)
            if not _can_place(mask, occupied, r0, c0):
                continue
            _paint_mask(grid, occupied, mask, r0, c0, color)
            placed = True
            break
        if not placed:
            return None
    occupied[:, 1] = False
    return grid


def _build_expected_output(pair_specs: list[dict], legend_pairs: list[tuple[int, int]]) -> np.ndarray:
    patches = []
    for spec, (top, bottom) in zip(pair_specs, legend_pairs):
        patch = np.zeros(spec["assembled"][0].shape, dtype=int)
        patch[spec["assembled"][0]] = int(top)
        patch[spec["assembled"][1]] = int(bottom)
        patches.append(patch)
    out_h = max(patch.shape[0] for patch in patches)
    out_w = sum(patch.shape[1] for patch in patches) + len(patches) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for patch in patches:
        h, w = patch.shape
        out[out_h - h :, col : col + w] = patch
        col += w + 1
    return out


def _count_matchings(fragments: list[dict], limit: int = 2) -> int:
    graph = _match_graph(fragments)
    n = len(fragments)
    used = [False] * n
    count = 0

    def dfs() -> None:
        nonlocal count
        if count >= limit:
            return
        try:
            i = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            count += 1
            return
        used[i] = True
        for j in range(i + 1, n):
            if used[j] or (i, j) not in graph:
                continue
            used[j] = True
            dfs()
            used[j] = False
        used[i] = False

    dfs()
    return count


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNTS)
    dims = _sample_rectangles(pair_count)
    if not dims:
        return None

    legend_pairs = _sample_legend_pairs(pair_count)
    pair_specs: list[dict] = []
    scattered_pieces: list[dict] = []

    for idx, (rect_h, rect_w) in enumerate(dims):
        relation = "lr" if random.random() < 0.5 else "tb"
        first_mask, second_mask = _build_split_masks(rect_h, rect_w, relation)

        rotate_idx = random.randint(0, 1) if random.random() < 0.5 else None
        observed = [first_mask.copy(), second_mask.copy()]
        if rotate_idx is not None:
            observed[rotate_idx] = np.rot90(observed[rotate_idx], 2)

        pair_specs.append(
            {
                "area": rect_h * rect_w,
                "assembled": (first_mask, second_mask),
            }
        )

        for piece_mask in observed:
            scattered_pieces.append(
                {
                    "mask": piece_mask,
                    "color": random.choice(COLORS),
                    "area": int(piece_mask.sum()),
                }
            )

    pair_specs.sort(key=lambda spec: spec["area"])
    expected = _build_expected_output(pair_specs, legend_pairs)
    if max(expected.shape) > MAX_GRID:
        return None

    legend_height = 3 * pair_count - 1
    max_piece_h = max(piece["mask"].shape[0] for piece in scattered_pieces)
    max_piece_w = max(piece["mask"].shape[1] for piece in scattered_pieces)
    min_h = max(legend_height + 1, max_piece_h + 2, 16)
    min_w = max(max_piece_w + 3, 16)
    if min_h > MAX_GRID or min_w > MAX_GRID:
        return None

    for _ in range(40):
        height = random.randint(min_h, MAX_GRID)
        width = random.randint(min_w, MAX_GRID)
        grid = _scatter_fragments(height, width, legend_pairs, scattered_pieces)
        if grid is None:
            continue

        legend, legend_mask = _parse_legend(grid)
        fragments = _extract_fragments(grid, legend_mask)
        if len(fragments) != 2 * pair_count:
            continue
        if _count_matchings(fragments) != 1:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
