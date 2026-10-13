import random
from collections import Counter, deque, defaultdict

import numpy as np


ARC_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
PAIR_HEIGHT_RANGE = (4, 7)
PAIR_WIDTH_RANGE = (4, 7)
MAX_GENERATION_ATTEMPTS = 300
PAIR_MASK_ATTEMPTS = 120
LAYOUT_ATTEMPTS = 80


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _neighbors8(r: int, c: int, h: int, w: int):
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if nr == r and nc == c:
                continue
            yield nr, nc


def _is_connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    h, w = mask.shape
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    stepper = _neighbors4 if connectivity == 4 else _neighbors8
    while q:
        r, c = q.popleft()
        for nr, nc in stepper(r, c, h, w):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_borders(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    components: list[dict] = []
    stepper = _neighbors8
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        seen = np.zeros((h, w), dtype=bool)
        for r, c in np.argwhere(grid == color):
            r = int(r)
            c = int(c)
            if seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in stepper(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [pt[0] for pt in cells]
            cols = [pt[1] for pt in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                sprite[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "area": len(cells),
                    "bbox": (r0, c0, r1, c1),
                    "mask": sprite,
                }
            )
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble complementary sprite pairs into solid rectangles and stack them by area.

    Input structure:
    - A dominant background surrounds several monochrome sprites, each in a unique color.
    - The sprites come in pairs with identical bounding-box sizes.
    - Within a pair, horizontally flipping one sprite makes its occupied cells exactly fill the other sprite's empty cells inside that shared bbox.

    Rule:
    - Find every monochrome sprite, record its bounding box, and group sprites by bbox size.
    - Inside each size group, pair the two sprites whose masks become perfect complements after horizontally flipping one of them.
    - Turn each recovered pair into a solid rectangle of that bbox size, colored by the smaller sprite in the pair.
    - Sort the recovered rectangles by area from smallest to largest and stack them top-to-bottom with no gaps, left-aligned on a background canvas as wide as the widest rectangle.

    Key insight:
    - The scattered fragments are not output objects by themselves; each same-sized complementary pair encodes one full rectangle, and the smaller fragment decides that rectangle's final color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = _find_components(g, bg)
    assert components and len(components) % 2 == 0

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for comp in components:
        groups[tuple(map(int, comp["mask"].shape))].append(comp)

    rectangles: list[tuple[int, int, int, int]] = []
    for dims, group in groups.items():
        pending = list(group)
        while pending:
            comp = pending.pop()
            expected = np.logical_not(comp["mask"])
            matches = [other for other in pending if np.array_equal(np.fliplr(other["mask"]), expected)]
            assert len(matches) == 1
            other = matches[0]
            pending.remove(other)

            h, w = dims
            rect_area = h * w
            assert comp["area"] + other["area"] == rect_area
            assert comp["area"] != other["area"]
            fill = int(comp["color"] if comp["area"] < other["area"] else other["color"])
            rectangles.append((rect_area, h, w, fill))

    rectangles.sort(key=lambda item: (item[0], item[1], item[2]))
    out_h = sum(h for _, h, _, _ in rectangles)
    out_w = max(w for _, _, w, _ in rectangles)
    out = np.full((out_h, out_w), bg, dtype=int)

    r = 0
    for _, h, w, color in rectangles:
        out[r : r + h, :w] = color
        r += h
    return out


def _pair_mask_valid(mask: np.ndarray) -> bool:
    comp = np.logical_not(mask)
    return bool(
        mask.any()
        and comp.any()
        and _touches_all_borders(mask)
        and _touches_all_borders(comp)
        and _is_connected(mask, connectivity=8)
        and _is_connected(comp, connectivity=8)
    )


def _stair_band_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(PAIR_MASK_ATTEMPTS):
        band_w = random.randint(1, min(3, w - 2))
        max_start = w - band_w
        if max_start < 1:
            continue

        pivot = random.randint(1, h - 2)
        starts = [0]
        cur = 0
        ok = True
        for r in range(1, pivot + 1):
            remain = pivot - r + 1
            need = max_start - cur
            lo = max(0, need - remain * band_w)
            hi = min(band_w, need)
            if lo > hi:
                ok = False
                break
            cur += random.randint(lo, hi)
            starts.append(cur)
        if not ok:
            continue

        starts[pivot] = max_start
        cur = max_start
        for _ in range(pivot + 1, h):
            cur = max(0, cur - random.randint(0, min(band_w, cur)))
            starts.append(cur)

        if len(starts) != h:
            continue
        if 0 not in starts or max_start not in starts or all(v in (0, max_start) for v in starts):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, start in enumerate(starts):
            mask[r, start : start + band_w] = True

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        if _pair_mask_valid(mask):
            return mask
    return None


def _make_region_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    if min(h, w) < 4 or (h == 4 and w == 4):
        return None

    for _ in range(PAIR_MASK_ATTEMPTS):
        if w >= 5:
            mask = _stair_band_mask(h, w)
        elif h >= 5:
            base = _stair_band_mask(w, h)
            mask = None if base is None else base.T
        else:
            mask = None
        if mask is None:
            continue
        if int(mask.sum()) * 2 == h * w:
            continue
        return mask, np.logical_not(mask)
    return None


def _sample_pair_shapes(pair_count: int) -> list[tuple[int, int]] | None:
    candidates = [(h, w) for h in range(PAIR_HEIGHT_RANGE[0], PAIR_HEIGHT_RANGE[1] + 1)
                  for w in range(PAIR_WIDTH_RANGE[0], PAIR_WIDTH_RANGE[1] + 1)
                  if not (h == 4 and w == 4)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    height_sum = 0
    for h, w in candidates:
        area = h * w
        if area in used_areas or height_sum + h > ARC_MAX:
            continue
        chosen.append((h, w))
        used_areas.add(area)
        height_sum += h
        if len(chosen) == pair_count:
            return chosen
    return None


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    bins: list[int] = []
    prev = -1
    limit = total + parts - 1
    for cut in cuts + [limit]:
        bins.append(cut - prev - 1)
        prev = cut
    return bins


def _split_rows(objects: list[dict], limit: int) -> list[list[dict]]:
    rows: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for obj in objects:
        h, w = obj["mask"].shape
        need = w if not current else w + 1
        if current and (used + need > limit or random.random() < 0.18):
            rows.append(current)
            current = [obj]
            used = w
        else:
            current.append(obj)
            used += need
    if current:
        rows.append(current)
    return rows


def _layout_objects(objects: list[dict]) -> tuple[int, int, list[tuple[dict, int, int]]] | None:
    max_w = max(obj["mask"].shape[1] for obj in objects)
    total_w = sum(obj["mask"].shape[1] for obj in objects) + len(objects) - 1
    for _ in range(LAYOUT_ATTEMPTS):
        order = objects[:]
        random.shuffle(order)
        low = max(max_w + 2, min(total_w, max_w + 2))
        high = min(28, max(total_w, low))
        if low > high:
            return None
        row_limit = random.randint(low, high)
        rows = _split_rows(order, row_limit)

        row_heights = [max(obj["mask"].shape[0] for obj in row) for row in rows]
        row_gaps = [random.randint(1, 2) for _ in range(max(0, len(rows) - 1))]
        pack_h = sum(row_heights) + sum(row_gaps)
        if pack_h > 26:
            continue

        extra_h = random.randint(2, ARC_MAX - pack_h)
        extra_w = random.randint(2, ARC_MAX - row_limit)
        top_margin = random.randint(1, extra_h - 1)
        left_margin = random.randint(1, extra_w - 1)
        h = pack_h + extra_h
        w = row_limit + extra_w

        placements: list[tuple[dict, int, int]] = []
        y = top_margin
        ok = True
        for row_idx, row in enumerate(rows):
            row_h = row_heights[row_idx]
            base_w = sum(obj["mask"].shape[1] for obj in row) + len(row) - 1
            slack = row_limit - base_w
            extras = _random_composition(slack, len(row) + 1)
            x = left_margin + extras[0]
            for item_idx, obj in enumerate(row):
                obj_h, obj_w = obj["mask"].shape
                y0 = y + random.randint(0, row_h - obj_h)
                placements.append((obj, y0, x))
                x += obj_w + 1 + extras[item_idx + 1]
            y += row_h
            if row_idx < len(row_gaps):
                y += row_gaps[row_idx]

        occupied = np.zeros((h, w), dtype=bool)
        for obj, r0, c0 in placements:
            obj_h, obj_w = obj["mask"].shape
            rs = slice(max(0, r0 - 1), min(h, r0 + obj_h + 1))
            cs = slice(max(0, c0 - 1), min(w, c0 + obj_w + 1))
            if occupied[rs, cs].any():
                ok = False
                break
            occupied[rs, cs] = True
        if ok:
            return h, w, placements
    return None


def _build_expected_output(bg: int, pair_meta: list[dict]) -> np.ndarray:
    ordered = sorted(pair_meta, key=lambda item: (item["area"], item["h"], item["w"]))
    out_h = sum(item["h"] for item in ordered)
    out_w = max(item["w"] for item in ordered)
    out = np.full((out_h, out_w), bg, dtype=int)
    r = 0
    for item in ordered:
        out[r : r + item["h"], : item["w"]] = int(item["fill"])
        r += item["h"]
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    shapes = _sample_pair_shapes(pair_count)
    if shapes is None:
        return None

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 2 * pair_count:
        return None

    objects: list[dict] = []
    pair_meta: list[dict] = []
    color_idx = 0
    for h, w in shapes:
        pair = _make_region_pair(h, w)
        if pair is None:
            return None
        region_a, region_b = pair
        variants = [
            (region_a, np.fliplr(region_b)),
            (region_b, np.fliplr(region_a)),
            (np.fliplr(region_a), region_b),
            (np.fliplr(region_b), region_a),
        ]
        mask1, mask2 = random.choice(variants)
        color1 = colors[color_idx]
        color2 = colors[color_idx + 1]
        color_idx += 2

        area1 = int(mask1.sum())
        area2 = int(mask2.sum())
        fill = color1 if area1 < area2 else color2
        pair_meta.append({"area": h * w, "h": h, "w": w, "fill": fill})
        objects.append({"mask": mask1, "color": color1})
        objects.append({"mask": mask2, "color": color2})

    layout = _layout_objects(objects)
    if layout is None:
        return None
    h, w, placements = layout
    grid = np.full((h, w), bg, dtype=int)
    for obj, r0, c0 in placements:
        mask = obj["mask"]
        obj_h, obj_w = mask.shape
        patch = grid[r0 : r0 + obj_h, c0 : c0 + obj_w]
        patch[mask] = int(obj["color"])

    counts = Counter(map(int, grid.ravel()))
    if counts.most_common(1)[0][0] != bg:
        return None

    expected = _build_expected_output(bg, pair_meta)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid complementary-pair input")
