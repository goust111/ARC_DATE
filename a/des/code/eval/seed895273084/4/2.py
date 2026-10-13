import math
import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 300
PAIR_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (4, 8)
BLOCK_COUNT_MAX = 4
GRID_SIDE_MAX = 30
PLACEMENT_RETRIES = 250
SHAPE_RETRIES = 200


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dr, dc in directions:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _get_bg(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    if flat.size == 0:
        return 0
    counts = np.bincount(flat, minlength=max(10, int(flat.max()) + 1))
    return int(np.argmax(counts))


def _coords_connected(coords, shape, connectivity: int) -> bool:
    cells = {(int(r), int(c)) for r, c in coords}
    if not cells:
        return False
    h, w = shape
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors(r, c, h, w, connectivity):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _mask_connected(mask: np.ndarray, connectivity: int) -> bool:
    coords = np.argwhere(mask)
    return _coords_connected(coords, mask.shape, connectivity)


def _touches_all_sides(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return False
    h, w = mask.shape
    return (
        int(rows.min()) == 0
        and int(rows.max()) == h - 1
        and int(cols.min()) == 0
        and int(cols.max()) == w - 1
    )


def _find_components(grid: np.ndarray, bg: int, connectivity: int):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            rel = arr - np.array([r0, c0], dtype=int)
            mask[rel[:, 0], rel[:, 1]] = True
            components.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _pair_components(components, bg: int):
    if not components or len(components) % 2 != 0:
        raise ValueError("need a non-empty even number of fragments")

    groups = defaultdict(list)
    for idx, comp in enumerate(components):
        h, w = comp["mask"].shape
        item = dict(comp)
        item["id"] = idx
        item["shape"] = (int(h), int(w))
        groups[item["shape"]].append(item)

    rectangles = []
    for shape, items in groups.items():
        if len(items) % 2 != 0:
            raise ValueError("same-size fragment count must be even")

        by_key = defaultdict(list)
        for item in items:
            by_key[item["mask"].tobytes()].append(item)

        used = set()
        ordered = sorted(items, key=lambda item: (item["bbox"][0], item["bbox"][1], item["id"]))
        for item in ordered:
            if item["id"] in used:
                continue

            complement_key = np.logical_not(item["mask"]).tobytes()
            candidates = [
                candidate
                for candidate in by_key.get(complement_key, [])
                if candidate["id"] not in used and candidate["id"] != item["id"]
            ]
            if not candidates:
                raise ValueError("missing complementary partner")

            mate = min(candidates, key=lambda cand: (cand["bbox"][0], cand["bbox"][1], cand["id"]))
            used.add(item["id"])
            used.add(mate["id"])

            merged = np.full(shape, bg, dtype=int)
            merged[item["mask"]] = int(item["color"])
            merged[mate["mask"]] = int(mate["color"])

            leftmost = min(item["bbox"][1], mate["bbox"][1])
            topmost = min(item["bbox"][0], mate["bbox"][0])
            rectangles.append(
                {
                    "area": int(shape[0] * shape[1]),
                    "shape": shape,
                    "merged": merged,
                    "anchor": (int(leftmost), int(topmost)),
                }
            )

        if len(used) != len(items):
            raise ValueError("failed to pair all fragments")

    return rectangles


def _pack_rectangles(rectangles, bg: int) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda rect: (rect["area"], rect["anchor"][0], rect["anchor"][1]))
    out_h = max(rect["shape"][0] for rect in ordered)
    out_w = sum(rect["shape"][1] for rect in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c0 = 0
    for rect in ordered:
        merged = rect["merged"]
        h, w = merged.shape
        out[:h, c0 : c0 + w] = merged
        c0 += w + 1

    return out


def _solve_with_connectivity(grid: np.ndarray, bg: int, connectivity: int) -> np.ndarray:
    components = _find_components(grid, bg=bg, connectivity=connectivity)
    rectangles = _pair_components(components, bg=bg)
    return _pack_rectangles(rectangles, bg=bg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    last_error = None
    for connectivity in (4, 8):
        try:
            return _solve_with_connectivity(g, bg=bg, connectivity=connectivity)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"could not reconstruct complementary rectangles: {last_error}")


def _partition_sizes(total: int, parts: int, rng: random.Random):
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    sizes = []
    prev = 0
    for cut in cuts + [total]:
        sizes.append(cut - prev)
        prev = cut
    return sizes


def _sample_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        row_blocks = rng.randint(2, min(BLOCK_COUNT_MAX, h))
        col_blocks = rng.randint(2, min(BLOCK_COUNT_MAX, w))
        row_sizes = _partition_sizes(h, row_blocks, rng)
        col_sizes = _partition_sizes(w, col_blocks, rng)
        phase = rng.randrange(2)

        mask = np.zeros((h, w), dtype=bool)
        r0 = 0
        for i, rlen in enumerate(row_sizes):
            c0 = 0
            for j, clen in enumerate(col_sizes):
                if (i + j + phase) % 2 == 0:
                    mask[r0 : r0 + rlen, c0 : c0 + clen] = True
                c0 += clen
            r0 += rlen

        if rng.random() < 0.5:
            mask = np.flip(mask, axis=0)
        if rng.random() < 0.5:
            mask = np.flip(mask, axis=1)

        area = int(mask.sum())
        total = h * w
        if area < max(4, total // 5) or total - area < max(4, total // 5):
            continue
        if not (_touches_all_sides(mask) and _touches_all_sides(~mask)):
            continue
        if not (_mask_connected(mask, 8) and _mask_connected(~mask, 8)):
            continue
        return mask

    raise RuntimeError("failed to sample a valid complementary mask")


def _choose_specs(pair_count: int, rng: random.Random):
    specs = []
    duplicate_mode = rng.random() < 0.65

    if duplicate_mode:
        shared = (
            rng.randint(*RECT_SIDE_RANGE),
            rng.randint(*RECT_SIDE_RANGE),
        )
        specs.extend([shared, shared])

    while len(specs) < pair_count:
        candidate = (
            rng.randint(*RECT_SIDE_RANGE),
            rng.randint(*RECT_SIDE_RANGE),
        )
        if candidate in specs:
            continue
        specs.append(candidate)

    rng.shuffle(specs)
    return specs[:pair_count]


def _place_fragments(fragments, h: int, w: int, rng: random.Random):
    placements = []
    blocked = np.zeros((h, w), dtype=bool)

    order = list(range(len(fragments)))
    rng.shuffle(order)

    for idx in order:
        frag = fragments[idx]
        fh, fw = frag["mask"].shape
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randrange(h - fh + 1)
            c0 = rng.randrange(w - fw + 1)
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + fh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + fw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue

            blocked[rr0:rr1, cc0:cc1] = True
            placements.append((idx, r0, c0))
            break
        else:
            return None

    return placements


def _build_expected(pairs, placements):
    pair_anchors = defaultdict(list)
    rectangles = []
    by_pair = {pair["pair_id"]: pair for pair in pairs}

    for frag in placements:
        pair_anchors[frag["pair_id"]].append((frag["row"], frag["col"]))

    for pair_id, pair in by_pair.items():
        h, w = pair["shape"]
        merged = np.full((h, w), BG, dtype=int)
        merged[pair["mask_a"]] = pair["color_a"]
        merged[pair["mask_b"]] = pair["color_b"]
        anchors = pair_anchors[pair_id]
        leftmost = min(col for _row, col in anchors)
        topmost = min(row for row, _col in anchors)
        rectangles.append(
            {
                "area": h * w,
                "shape": (h, w),
                "merged": merged,
                "anchor": (leftmost, topmost),
            }
        )

    return _pack_rectangles(rectangles, bg=BG)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    specs = _choose_specs(pair_count, rng)
    total_footprint = sum(2 * (h + 2) * (w + 2) for h, w in specs)
    min_side = max(
        max(h for h, _w in specs) + 3,
        max(w for _h, w in specs) + 3,
        int(math.ceil(math.sqrt(total_footprint * 1.15))),
    )
    if min_side > GRID_SIDE_MAX:
        return None

    grid_h = rng.randint(min_side, GRID_SIDE_MAX)
    grid_w = rng.randint(min_side, GRID_SIDE_MAX)

    colors = rng.sample(PALETTE, 2 * pair_count)
    seen_masks = defaultdict(set)
    pairs = []
    fragments = []

    for pair_id, (h, w) in enumerate(specs):
        for _ in range(SHAPE_RETRIES):
            mask_a = _sample_mask(h, w, rng)
            key_a = mask_a.tobytes()
            key_b = np.logical_not(mask_a).tobytes()
            if key_a in seen_masks[(h, w)] or key_b in seen_masks[(h, w)]:
                continue
            seen_masks[(h, w)].update({key_a, key_b})
            break
        else:
            return None

        color_a = colors[2 * pair_id]
        color_b = colors[2 * pair_id + 1]
        if rng.random() < 0.5:
            color_a, color_b = color_b, color_a

        pair = {
            "pair_id": pair_id,
            "shape": (h, w),
            "mask_a": mask_a,
            "mask_b": np.logical_not(mask_a),
            "color_a": color_a,
            "color_b": color_b,
        }
        pairs.append(pair)
        fragments.append({"pair_id": pair_id, "mask": pair["mask_a"], "color": color_a})
        fragments.append({"pair_id": pair_id, "mask": pair["mask_b"], "color": color_b})

    placements_raw = _place_fragments(fragments, grid_h, grid_w, rng)
    if placements_raw is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    placed = []
    for idx, r0, c0 in placements_raw:
        frag = fragments[idx]
        mask = frag["mask"]
        color = frag["color"]
        h, w = mask.shape
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((patch != BG) & mask):
            return None
        patch[mask] = color
        placed.append(
            {
                "pair_id": frag["pair_id"],
                "row": r0,
                "col": c0,
            }
        )

    expected = _build_expected(pairs, placed)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-rectangle puzzle")
