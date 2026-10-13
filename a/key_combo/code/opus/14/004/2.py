import random
from collections import defaultdict, deque

import numpy as np


BG = 0
STRIPE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
AREA_CHOICES = [3, 4, 5, 6, 7]
COUNT_CHOICES = [(2, 2), (2, 3), (3, 2)]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask(area: int) -> np.ndarray:
    for _ in range(200):
        candidates = []
        max_h = min(4, area)
        for h in range(1, max_h + 1):
            min_w = max(1, (area + h - 1) // h)
            for w in range(min_w, min(5, area) + 1):
                if h * w >= area:
                    candidates.append((h, w))
        h, w = random.choice(candidates)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return _crop_mask(mask)
    raise RuntimeError("failed to sample a connected mask")


def _placed_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [(top + int(r), left + int(c)) for r, c in coords]


def _translated_cells(
    mask: np.ndarray,
    top: int,
    left: int,
    side: str,
    area: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    shift = area if side == "up" else -area
    out = []
    for r, c in _placed_cells(mask, top, left):
        rr = r + shift
        if 0 <= rr < h and 0 <= c < w:
            out.append((rr, c))
    return out


def _reserve_with_halo(forbidden: set[tuple[int, int]], cells: list[tuple[int, int]], shape: tuple[int, int]) -> None:
    h, w = shape
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                forbidden.add((nr, nc))
        forbidden.add((r, c))


def _extract_objects(grid: np.ndarray):
    h, w = grid.shape
    stripe_rows = [r for r in range(h) if np.all(grid[r, :] == STRIPE)]
    assert len(stripe_rows) == 1
    stripe = stripe_rows[0]

    objects = []
    for side, rows in (("up", range(0, stripe)), ("down", range(stripe + 1, h))):
        seen = np.zeros((h, w), dtype=bool)
        for r in rows:
            for c in range(w):
                color = int(grid[r, c])
                if color in (BG, STRIPE) or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords = []
                while q:
                    cr, cc = q.popleft()
                    coords.append((cr, cc))
                    for nr, nc in _neighbors4(cr, cc):
                        if nr not in rows or not (0 <= nc < w):
                            continue
                        if seen[nr, nc] or int(grid[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        q.append((nr, nc))
                objects.append(
                    {
                        "side": side,
                        "color": color,
                        "coords": coords,
                        "area": len(coords),
                    }
                )
    return stripe, objects


def main(grid: np.ndarray) -> np.ndarray:
    """Move upper objects downward and lower objects upward by their own areas, with smaller-area overlaps taking priority.

    Input structure:
    - The grid is black background with exactly one full-width gray horizontal stripe.
    - The stripe splits the canvas into an upper half and a lower half.
    - Each half contains several separated 4-connected monochrome objects.

    Rule:
    - Erase the gray stripe to black.
    - Move every upper object straight downward by its pixel area, and every lower object straight upward by its pixel area.
    - Clip any translated pixels that leave the grid.
    - If translated objects overlap, the cell takes the color of the object with smaller area.

    Key insight:
    - Each object's area controls both how far it moves and which color survives a collision.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    stripe, objects = _extract_objects(g)

    out = np.full((h, w), BG, dtype=int)
    best_area = np.full((h, w), 10**9, dtype=int)
    best_order = np.full((h, w), -1, dtype=int)

    for order, obj in enumerate(objects):
        shift = obj["area"] if obj["side"] == "up" else -obj["area"]
        for r, c in obj["coords"]:
            rr = r + shift
            if not (0 <= rr < h and 0 <= c < w):
                continue
            area = int(obj["area"])
            if area < best_area[rr, c] or (area == best_area[rr, c] and order > best_order[rr, c]):
                best_area[rr, c] = area
                best_order[rr, c] = order
                out[rr, c] = int(obj["color"])

    return out


def _choose_overlap_positions(up_obj: dict, down_obj: dict, h: int, w: int, stripe: int):
    up_mask = up_obj["mask"]
    down_mask = down_obj["mask"]
    hu, wu = up_mask.shape
    hd, wd = down_mask.shape
    up_cells = [(int(r), int(c)) for r, c in np.argwhere(up_mask)]
    down_cells = [(int(r), int(c)) for r, c in np.argwhere(down_mask)]
    random.shuffle(up_cells)
    random.shuffle(down_cells)

    for ur, uc in up_cells:
        for dr, dc in down_cells:
            row_lo = max(up_obj["area"] + ur, stripe + 1 - down_obj["area"] + dr)
            row_hi = min(up_obj["area"] + ur + (stripe - hu), dr - down_obj["area"] + (h - hd))
            col_lo = max(uc, dc)
            col_hi = min(w - wu + uc, w - wd + dc)
            if row_lo > row_hi or col_lo > col_hi:
                continue

            if random.random() < 0.35:
                overlap_r = random.choice([row_lo, row_hi])
                overlap_c = random.choice([col_lo, col_hi])
            else:
                overlap_r = random.randint(row_lo, row_hi)
                overlap_c = random.randint(col_lo, col_hi)

            up_top = overlap_r - ur - up_obj["area"]
            up_left = overlap_c - uc
            down_top = overlap_r - dr + down_obj["area"]
            down_left = overlap_c - dc
            return (up_top, up_left), (down_top, down_left)
    return None


def _place_random_object(
    obj: dict,
    h: int,
    w: int,
    stripe: int,
    forbidden: set[tuple[int, int]],
    dest_union: set[tuple[int, int]],
):
    mask = obj["mask"]
    mh, mw = mask.shape
    if obj["side"] == "up":
        top_min, top_max = 0, stripe - mh
    else:
        top_min, top_max = stripe + 1, h - mh
    if top_min > top_max:
        return None

    for _ in range(300):
        if obj["side"] == "down" and obj["area"] > stripe and random.random() < 0.5:
            top = random.randint(top_min, min(top_max, top_min + 1))
        else:
            top = random.randint(top_min, top_max)
        left = random.randint(0, w - mw)
        cells = _placed_cells(mask, top, left)
        if any(cell in forbidden for cell in cells):
            continue
        dest_cells = _translated_cells(mask, top, left, obj["side"], obj["area"], (h, w))
        if not dest_cells or any(cell in dest_union for cell in dest_cells):
            continue
        obj["top"] = top
        obj["left"] = left
        return cells, dest_cells
    return None


def _render_input(h: int, w: int, stripe: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[stripe, :] = STRIPE
    for obj in objects:
        for r, c in _placed_cells(obj["mask"], obj["top"], obj["left"]):
            grid[r, c] = obj["color"]
    return grid


def _overlap_stats(objects: list[dict], shape: tuple[int, int]):
    owners = defaultdict(list)
    for idx, obj in enumerate(objects):
        for cell in _translated_cells(obj["mask"], obj["top"], obj["left"], obj["side"], obj["area"], shape):
            owners[cell].append(idx)
    cross = 0
    total = 0
    for idxs in owners.values():
        if len(idxs) < 2:
            continue
        total += 1
        sides = {objects[idx]["side"] for idx in idxs}
        if len(sides) > 1:
            cross += 1
    return cross, total


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(16, 24)
    stripe = random.randint(5, h - 7)
    up_count, down_count = random.choice(COUNT_CHOICES)
    total = up_count + down_count

    areas = random.sample(AREA_CHOICES, total)
    random.shuffle(areas)
    colors = random.sample(PALETTE, total)

    objects = []
    for idx in range(total):
        side = "up" if idx < up_count else "down"
        objects.append(
            {
                "side": side,
                "area": areas[idx],
                "color": colors[idx],
                "mask": _random_connected_mask(areas[idx]),
            }
        )

    upper = [obj for obj in objects if obj["side"] == "up"]
    lower = [obj for obj in objects if obj["side"] == "down"]
    up_obj = random.choice(upper)
    down_obj = random.choice(lower)
    forced = _choose_overlap_positions(up_obj, down_obj, h, w, stripe)
    if forced is None:
        return None

    forbidden: set[tuple[int, int]] = set()
    dest_union: set[tuple[int, int]] = set()
    for obj, (top, left) in zip((up_obj, down_obj), forced):
        cells = _placed_cells(obj["mask"], top, left)
        if any(cell in forbidden for cell in cells):
            return None
        obj["top"] = top
        obj["left"] = left
        _reserve_with_halo(forbidden, cells, (h, w))
        for cell in _translated_cells(obj["mask"], top, left, obj["side"], obj["area"], (h, w)):
            dest_union.add(cell)

    remaining = [obj for obj in objects if "top" not in obj]
    random.shuffle(remaining)
    for obj in remaining:
        placed = _place_random_object(obj, h, w, stripe, forbidden, dest_union)
        if placed is None:
            return None
        cells, dest_cells = placed
        _reserve_with_halo(forbidden, cells, (h, w))
        dest_union.update(dest_cells)

    grid = _render_input(h, w, stripe, objects)
    stripe2, extracted = _extract_objects(grid)
    if stripe2 != stripe or len(extracted) != total:
        return None
    if sorted(obj["area"] for obj in extracted) != sorted(obj["area"] for obj in objects):
        return None
    if sorted(obj["color"] for obj in extracted) != sorted(obj["color"] for obj in objects):
        return None

    cross_overlap, total_overlap = _overlap_stats(objects, (h, w))
    if cross_overlap < 1 or total_overlap > 4:
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if np.any(out[stripe, :] != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
