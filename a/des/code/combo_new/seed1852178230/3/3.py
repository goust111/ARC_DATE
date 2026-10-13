import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = end = indices[0]
    for value in indices[1:]:
        if value == end + 1:
            end = value
        else:
            out.append((start, end))
            start = end = value
    out.append((start, end))
    return out


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for pr, pc in pts:
                mask[pr - r0, pc - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(pts),
                }
            )
    return objects


def _find_legend_top(grid: np.ndarray) -> int:
    h = grid.shape[0]
    r = h - 1
    while r >= 0 and not np.any(grid[r] != 0):
        r -= 1
    assert r == h - 1 and r >= 0
    while r >= 0 and np.any(grid[r] != 0):
        r -= 1
    legend_top = r + 1
    assert legend_top < h
    return legend_top


def _unique_nonzero(block: np.ndarray) -> int:
    values = sorted({int(v) for v in block.reshape(-1) if int(v) != 0})
    assert len(values) == 1
    return values[0]


def _parse_legend(legend: np.ndarray) -> dict[frozenset[int], int]:
    active_cols = [c for c in range(legend.shape[1]) if np.any(legend[:, c] != 0)]
    runs = _runs(active_cols)
    assert runs and len(runs) % 2 == 0
    mapping: dict[frozenset[int], int] = {}
    for idx in range(0, len(runs), 2):
        left0, left1 = runs[idx]
        right0, right1 = runs[idx + 1]
        assert right0 - left1 == 2
        if idx + 2 < len(runs):
            next_gap = runs[idx + 2][0] - right1 - 1
            assert next_gap >= 2
        left_color = _unique_nonzero(legend[:, left0 : left1 + 1])
        right_color = _unique_nonzero(legend[:, right0 : right1 + 1])
        key = frozenset((left_color, right_color))
        assert key not in mapping
        mapping[key] = left_color
    return mapping


def _pad_tl(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((out_h, out_w), dtype=bool)
    out[:h, :w] = mask
    return out


def _pad_br(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((out_h, out_w), dtype=bool)
    out[out_h - h :, out_w - w :] = mask
    return out


def _support_matches(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int, int]]:
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    matches: set[tuple[int, int, int]] = set()
    for out_h in range(max(ha, hb), ha + hb + 1):
        for out_w in range(max(wa, wb), wa + wb + 1):
            a_tl = _pad_tl(mask_a, out_h, out_w)
            b_br = _pad_br(mask_b, out_h, out_w)
            if np.array_equal(a_tl, ~b_br):
                matches.add((out_h, out_w, 0))
            a_br = _pad_br(mask_a, out_h, out_w)
            b_tl = _pad_tl(mask_b, out_h, out_w)
            if np.array_equal(a_br, ~b_tl):
                matches.add((out_h, out_w, 1))
    return sorted(matches, key=lambda item: (item[0] * item[1], item[0], item[1], item[2]))


def _solve_pairs(objects: list[dict]) -> list[dict]:
    n = len(objects)
    assert n % 2 == 0 and n > 0
    by_object: list[list[dict]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matches = _support_matches(objects[i]["mask"], objects[j]["mask"])
            if len(matches) != 1:
                continue
            out_h, out_w, tl_index = matches[0]
            record = {
                "a": i,
                "b": j,
                "out_h": out_h,
                "out_w": out_w,
                "tl_index": i if tl_index == 0 else j,
            }
            by_object[i].append(record)
            by_object[j].append(record)
    for records in by_object:
        assert len(records) == 1
    pairs: list[dict] = []
    used: set[int] = set()
    for idx in range(n):
        if idx in used:
            continue
        record = by_object[idx][0]
        used.add(record["a"])
        used.add(record["b"])
        pairs.append(record)
    assert len(pairs) * 2 == n
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary color fragments, rebuild their hidden rectangles, and recolor them from the legend order.

    Input structure:
    - The background is black.
    - The main field contains an even number of single-color 4-connected fragments, with exactly one matching partner for each fragment.
    - Each partner pair represents opposite occupancy inside one shared rectangle, so together they cover every cell of that rectangle.
    - A bottom legend strip contains two-color blocks; each block has a left color, a black divider column, and a right color.

    Rule:
    - Isolate the bottom legend strip and read each block as an unordered color pair whose output fill is the block's left color.
    - Find fragment pairs whose shapes are complementary inside one common rectangle.
    - For every pair, locate the legend block containing the two fragment colors, ignoring their order inside the pair.
    - Draw the full rectangle at the left fragment's position, filled entirely with the legend's left-side color, and remove the original fragments and the legend.

    Key insight:
    - The fragments are not matched by color or proximity but by shape complementarity, and the final color comes from the legend's left-right ordering rather than from either fragment alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend_top = _find_legend_top(g)
    legend = g[legend_top:]
    color_map = _parse_legend(legend)
    objects = _find_components(g[:legend_top])
    pairs = _solve_pairs(objects)

    out = np.zeros_like(g)
    occupied = np.zeros_like(g, dtype=bool)
    for pair in pairs:
        obj_a = objects[pair["a"]]
        obj_b = objects[pair["b"]]
        fill = color_map[frozenset((obj_a["color"], obj_b["color"]))]
        left_idx = min((pair["a"], pair["b"]), key=lambda idx: (objects[idx]["bbox"][1], objects[idx]["bbox"][0]))
        r0, c0, _r1, _c1 = objects[left_idx]["bbox"]
        out_h = pair["out_h"]
        out_w = pair["out_w"]
        assert r0 + out_h <= out.shape[0]
        assert c0 + out_w <= out.shape[1]
        assert not np.any(occupied[r0 : r0 + out_h, c0 : c0 + out_w])
        occupied[r0 : r0 + out_h, c0 : c0 + out_w] = True
        out[r0 : r0 + out_h, c0 : c0 + out_w] = fill
    return out


def _horizontal_snake(h: int, w: int) -> list[tuple[int, int]]:
    assert h % 2 == 1
    coords: list[tuple[int, int]] = []
    for r in range(h):
        cols = range(w) if r % 2 == 0 else range(w - 1, -1, -1)
        for c in cols:
            coords.append((r, c))
    assert coords[0] == (0, 0)
    assert coords[-1] == (h - 1, w - 1)
    return coords


def _vertical_snake(h: int, w: int) -> list[tuple[int, int]]:
    assert w % 2 == 1
    coords: list[tuple[int, int]] = []
    for c in range(w):
        rows = range(h) if c % 2 == 0 else range(h - 1, -1, -1)
        for r in rows:
            coords.append((r, c))
    assert coords[0] == (0, 0)
    assert coords[-1] == (h - 1, w - 1)
    return coords


def _sample_pair_masks() -> dict | None:
    support_options = [
        ("h", 5, 4),
        ("h", 5, 5),
        ("h", 5, 6),
        ("v", 4, 5),
        ("v", 5, 5),
        ("v", 6, 5),
    ]
    orientation, h, w = random.choice(support_options)
    path = _horizontal_snake(h, w) if orientation == "h" else _vertical_snake(h, w)
    area = h * w
    min_cells = max(4, area // 4)
    max_cells = min(area - 4, (3 * area) // 4)
    cuts = [cut for cut in range(min_cells, max_cells + 1)]
    cuts = [cut for cut in cuts if cut % (w if orientation == "h" else h) != 0]
    if not cuts:
        cuts = list(range(min_cells, max_cells + 1))
    cut = random.choice(cuts)
    left_support = np.zeros((h, w), dtype=bool)
    for r, c in path[:cut]:
        left_support[r, c] = True
    right_support = ~left_support
    left_mask, _ = _crop_mask(left_support)
    right_mask, _ = _crop_mask(right_support)
    if left_mask.size == 0 or right_mask.size == 0:
        return None
    return {
        "support_h": h,
        "support_w": w,
        "left_support": left_support,
        "right_support": right_support,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "signature": (h, w, cut, orientation),
    }


def _render_support(grid: np.ndarray, top: int, left: int, support_mask: np.ndarray, color: int) -> None:
    h, w = support_mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[support_mask] = color


def _legend_width(pair_count: int, half_w: int, gap_w: int) -> int:
    return pair_count * (2 * half_w + 1) + (pair_count - 1) * gap_w


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([2, 3])
    pair_shapes: list[dict] = []
    seen_signatures: set[tuple] = set()
    for _ in range(pair_count):
        for _retry in range(100):
            shape = _sample_pair_masks()
            if shape is None:
                continue
            if shape["signature"] in seen_signatures:
                continue
            seen_signatures.add(shape["signature"])
            pair_shapes.append(shape)
            break
        else:
            return None

    object_colors = random.sample(COLORS, 2 * pair_count)
    left_colors = object_colors[:pair_count]
    right_colors = object_colors[pair_count:]

    legend_h = random.choice([1, 2])
    legend_half_w = random.choice([1, 2])
    legend_gap_w = random.choice([2, 3])
    spacer_h = 1
    band_gap = random.choice([1, 2])
    top_margin = random.choice([0, 1])

    max_support_w = max(shape["support_w"] for shape in pair_shapes)
    max_support_h = max(shape["support_h"] for shape in pair_shapes)
    band_heights = [shape["support_h"] + random.choice([0, 1]) for shape in pair_shapes]
    object_h = top_margin + sum(band_heights) + band_gap * (pair_count - 1)
    total_h = object_h + spacer_h + legend_h
    if total_h > 30:
        return None

    left_margin = 1
    right_margin = 1
    mid_gap = random.choice([2, 3, 4])
    extra_left = random.randint(0, 2)
    extra_right = random.randint(0, 2)
    left_zone_w = max_support_w + extra_left
    right_zone_w = max_support_w + extra_right
    min_w = left_margin + left_zone_w + mid_gap + right_zone_w + right_margin
    min_w = max(min_w, _legend_width(pair_count, legend_half_w, legend_gap_w) + 2)
    if min_w > 30:
        return None
    extra_w = random.randint(0, min(30 - min_w, 4))
    left_zone_w += random.randint(0, extra_w)
    right_zone_w += extra_w - (left_zone_w - (max_support_w + extra_left))
    total_w = left_margin + left_zone_w + mid_gap + right_zone_w + right_margin

    grid = np.zeros((total_h, total_w), dtype=int)
    expected = np.zeros_like(grid)

    order = list(range(pair_count))
    random.shuffle(order)
    cur_row = top_margin
    placed_pairs: list[dict] = []
    for band_index, shape_index in enumerate(order):
        shape = pair_shapes[shape_index]
        support_h = shape["support_h"]
        support_w = shape["support_w"]
        band_h = band_heights[band_index]
        row_low = cur_row
        row_high = cur_row + band_h - support_h
        if row_low > row_high:
            return None
        left_col_low = left_margin
        left_col_high = left_margin + left_zone_w - support_w
        right_col_low = total_w - right_margin - right_zone_w
        right_col_high = total_w - right_margin - support_w
        if left_col_low > left_col_high or right_col_low > right_col_high:
            return None
        left_top = random.randint(row_low, row_high)
        right_top = random.randint(row_low, row_high)
        left_left = random.randint(left_col_low, left_col_high)
        right_left = random.randint(right_col_low, right_col_high)

        left_color = left_colors[shape_index]
        right_color = right_colors[shape_index]
        fill_color = random.choice([left_color, right_color])
        other_color = right_color if fill_color == left_color else left_color

        _render_support(grid, left_top, left_left, shape["left_support"], left_color)
        _render_support(grid, right_top, right_left, shape["right_support"], right_color)
        expected[left_top : left_top + support_h, left_left : left_left + support_w] = fill_color
        placed_pairs.append(
            {
                "shape_index": shape_index,
                "left_color": left_color,
                "right_color": right_color,
                "fill_color": fill_color,
                "other_color": other_color,
            }
        )
        cur_row += band_h
        if band_index + 1 < pair_count:
            cur_row += band_gap

    legend_top = total_h - legend_h
    legend_w = _legend_width(pair_count, legend_half_w, legend_gap_w)
    legend_left = random.randint(1, total_w - legend_w - 1) if total_w - legend_w - 1 >= 1 else 0
    legend_order = placed_pairs.copy()
    random.shuffle(legend_order)
    block_w = 2 * legend_half_w + 1
    for idx, pair in enumerate(legend_order):
        col0 = legend_left + idx * (block_w + legend_gap_w)
        grid[legend_top:, col0 : col0 + legend_half_w] = pair["fill_color"]
        grid[legend_top:, col0 + legend_half_w + 1 : col0 + block_w] = pair["other_color"]

    if np.any(grid[legend_top - 1] != 0):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
