import random
from collections import Counter, defaultdict, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return components


def shape_key(component: dict) -> tuple[tuple[int, int], ...]:
    coords = component["coords"]
    rel = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in rel))


def bbox_center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def paint_axis_segment(out: np.ndarray, bg: int, a: dict, b: dict, color: int) -> None:
    ar2, ac2 = bbox_center2(a["bbox"])
    br2, bc2 = bbox_center2(b["bbox"])
    if ar2 == br2:
        row = ar2 // 2
        start = min(ac2, bc2) // 2
        stop = max(ac2, bc2) // 2
        for c in range(start, stop + 1):
            if int(out[row, c]) == bg:
                out[row, c] = color
        return
    if ac2 == bc2:
        col = ac2 // 2
        start = min(ar2, br2) // 2
        stop = max(ar2, br2) // 2
        for r in range(start, stop + 1):
            if int(out[r, col]) == bg:
                out[r, col] = color
        return
    raise AssertionError("paired objects must be horizontally or vertically aligned")


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-shape monochrome objects and connect each pair with a source-colored axis-aligned segment."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg)
    assert components and len(components) % 2 == 0

    grouped: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for component in components:
        grouped[shape_key(component)].append(component)

    out = g.copy()
    for group in grouped.values():
        assert len(group) == 2
        a, b = group
        assert a["color"] != b["color"]
        source, target = sorted(group, key=lambda obj: (bbox_center2(obj["bbox"])[0], bbox_center2(obj["bbox"])[1]))
        paint_axis_segment(out, bg, source, target, source["color"])
    return out


PALETTE = tuple(range(10))
GRID_RETRIES = 400
PAIR_COUNT_CHOICES = (2, 3, 4)
ZONE_SIZE_RANGE = (7, 11)
ZONE_GAP_RANGE = (2, 5)
OUTER_MARGIN_RANGE = (1, 3)
GAP_BETWEEN_OBJECTS = (2, 6)
MAX_SHAPE_SIDE = 5
_GEN_COUNTER = 0


def _odd_sizes(limit: int) -> list[int]:
    return [n for n in (1, 3, 5) if n <= min(limit, MAX_SHAPE_SIDE)]


def _make_shape(max_h: int, max_w: int) -> np.ndarray | None:
    hs = _odd_sizes(max_h)
    ws = _odd_sizes(max_w)
    if not hs or not ws:
        return None
    h = random.choice(hs)
    w = random.choice(ws)
    mask = np.zeros((h, w), dtype=bool)
    cr, cc = h // 2, w // 2

    mask[cr, :] = True
    mask[:, cc] = True

    extras = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
    random.shuffle(extras)
    add_count = random.randint(0, max(0, min(len(extras), (h * w) // 3)))
    for r, c in extras:
        if add_count == 0:
            break
        if any(mask[nr, nc] for nr, nc in neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w):
            mask[r, c] = True
            add_count -= 1
    return mask


def _paint_shape(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[top + coords[:, 0], left + coords[:, 1]] = color


def _zone_boxes(num_pairs: int) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    row_heights = [random.randint(*ZONE_SIZE_RANGE) for _ in range(2)]
    col_widths = [random.randint(*ZONE_SIZE_RANGE) for _ in range(2)]
    margin = random.randint(*OUTER_MARGIN_RANGE)
    gap_r = random.randint(*ZONE_GAP_RANGE)
    gap_c = random.randint(*ZONE_GAP_RANGE)

    h = margin * 2 + row_heights[0] + row_heights[1] + gap_r
    w = margin * 2 + col_widths[0] + col_widths[1] + gap_c

    r0 = margin
    r1 = margin + row_heights[0] + gap_r
    c0 = margin
    c1 = margin + col_widths[0] + gap_c
    zones = [
        (r0, c0, row_heights[0], col_widths[0]),
        (r0, c1, row_heights[0], col_widths[1]),
        (r1, c0, row_heights[1], col_widths[0]),
        (r1, c1, row_heights[1], col_widths[1]),
    ]
    random.shuffle(zones)
    return h, w, zones[:num_pairs]


def _orientation_plan(num_pairs: int, counter: int) -> list[str]:
    base = ["H", "V"]
    extras = ["H" if (counter + i) % 2 == 0 else "V" for i in range(max(0, num_pairs - 2))]
    plan = base + extras
    if counter % 2:
        plan.reverse()
    random.shuffle(plan)
    return plan


def _place_pair(grid: np.ndarray, zone: tuple[int, int, int, int], orientation: str, mask: np.ndarray, colors: tuple[int, int]) -> bool:
    zr, zc, zh, zw = zone
    h, w = mask.shape

    if orientation == "H":
        free_cols = zw - 2 * w
        free_rows = zh - h
        if free_cols < GAP_BETWEEN_OBJECTS[0] or free_rows < 0:
            return False
        gap_hi = min(GAP_BETWEEN_OBJECTS[1], free_cols)
        gap = random.randint(GAP_BETWEEN_OBJECTS[0], gap_hi)
        leftover = free_cols - gap
        left_pad = random.randint(0, leftover)
        row_pad = random.randint(0, free_rows)
        top = zr + row_pad
        left0 = zc + left_pad
        left1 = left0 + w + gap
        _paint_shape(grid, mask, top, left0, colors[0])
        _paint_shape(grid, mask, top, left1, colors[1])
        return True

    free_rows = zh - 2 * h
    free_cols = zw - w
    if free_rows < GAP_BETWEEN_OBJECTS[0] or free_cols < 0:
        return False
    gap_hi = min(GAP_BETWEEN_OBJECTS[1], free_rows)
    gap = random.randint(GAP_BETWEEN_OBJECTS[0], gap_hi)
    leftover = free_rows - gap
    top_pad = random.randint(0, leftover)
    col_pad = random.randint(0, free_cols)
    top0 = zr + top_pad
    top1 = top0 + h + gap
    left = zc + col_pad
    _paint_shape(grid, mask, top0, left, colors[0])
    _paint_shape(grid, mask, top1, left, colors[1])
    return True


def _valid_input(grid: np.ndarray, expected_pairs: int, expect_both_orientations: bool) -> bool:
    bg = get_bg(grid)
    comps = find_components(grid, bg)
    if len(comps) != expected_pairs * 2:
        return False

    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in comps:
        groups[shape_key(comp)].append(comp)
    if len(groups) != expected_pairs or any(len(group) != 2 for group in groups.values()):
        return False

    orientations = set()
    for group in groups.values():
        a, b = group
        ar2, ac2 = bbox_center2(a["bbox"])
        br2, bc2 = bbox_center2(b["bbox"])
        if ar2 == br2:
            orientations.add("H")
        elif ac2 == bc2:
            orientations.add("V")
        else:
            return False
    if expect_both_orientations and orientations != {"H", "V"}:
        return False

    out = main(grid)
    return not np.array_equal(out, grid)


def _gen_attempt(counter: int) -> np.ndarray | None:
    num_pairs = random.choice(PAIR_COUNT_CHOICES)
    h, w, zones = _zone_boxes(num_pairs)
    bg = random.choice(PALETTE)
    colors = random.sample([c for c in PALETTE if c != bg], 2 * num_pairs)
    grid = np.full((h, w), bg, dtype=int)

    used_shapes = set()
    orientations = _orientation_plan(num_pairs, counter)
    for i, zone in enumerate(zones):
        zr, zc, zh, zw = zone
        shape = _make_shape(min(zh, MAX_SHAPE_SIDE), min(zw, MAX_SHAPE_SIDE))
        if shape is None:
            return None

        key = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(shape)))
        if key in used_shapes:
            return None
        used_shapes.add(key)

        if not _place_pair(grid, zone, orientations[i], shape, (colors[2 * i], colors[2 * i + 1])):
            return None

    if not _valid_input(grid, num_pairs, expect_both_orientations=True):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(GRID_RETRIES):
        grid = _gen_attempt(_GEN_COUNTER)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid sample")
