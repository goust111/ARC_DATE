from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 8
OBJECT_COLORS = (1, 2, 3, 4, 6, 7)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PAIR_COUNT_CHOICES = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = [int(x) for x in grid[0]]
    if h > 1:
        border.extend(int(x) for x in grid[-1])
    if w > 1:
        border.extend(int(x) for x in grid[1:-1, 0])
        border.extend(int(x) for x in grid[1:-1, -1])
    if border:
        return int(Counter(border).most_common(1)[0][0])
    return int(Counter(int(x) for x in grid.flat).most_common(1)[0][0])


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            min_r = max_r = r
            min_c = max_c = c

            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
                }
            )

    return objects


def _hole_key(grid: np.ndarray, bg: int, obj: dict) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = obj["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(obj["color"])
    assert np.all((patch == bg) | (patch == color)), "each object must be a filled rectangle with an azure cutout"

    hole = tuple(
        (int(rr), int(cc))
        for rr in range(patch.shape[0])
        for cc in range(patch.shape[1])
        if int(patch[rr, cc]) == bg
    )
    assert hole, "each object must contain an internal azure cutout"
    assert all(0 < rr < patch.shape[0] - 1 and 0 < cc < patch.shape[1] - 1 for rr, cc in hole)
    return hole


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair rectangles by their internal azure cutout and swap each pair's fill colors.

    Input structure:
    - The background is azure.
    - Several non-overlapping monochrome rectangles sit on the background, each using a unique non-azure color.
    - Inside each rectangle, some azure cells form one connected cutout that stays strictly inside the rectangle border.

    Rule:
    - Extract each rectangle as a 4-connected monochrome component and inspect the azure cells inside its bounding box.
    - Represent that cutout as coordinates relative to the rectangle's top-left corner, ignoring the rectangle's absolute position and fill color.
    - Pair rectangles whose relative cutout coordinates are identical, then repaint each rectangle with its partner's color while leaving all azure cutout cells unchanged.

    Key insight: The visible colors and outer sizes are distractions; the true matching key is the local negative-space pattern carved out by azure cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects = _find_objects(g, bg)
    assert objects and len(objects) % 2 == 0, "expected an even number of colored objects"

    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for obj in objects:
        groups[_hole_key(g, bg, obj)].append(obj)

    assert all(len(group) == 2 for group in groups.values()), "each cutout pattern must appear exactly twice"

    out = g.copy()
    for group in groups.values():
        a, b = group
        for r, c in a["coords"]:
            out[r, c] = int(b["color"])
        for r, c in b["coords"]:
            out[r, c] = int(a["color"])
    return out


def _random_connected_shape(max_h: int, max_w: int, max_area: int) -> tuple[tuple[int, int], ...]:
    box_h = random.randint(1, max_h)
    box_w = random.randint(1, max_w)
    target = random.randint(1, min(max_area, box_h * box_w))

    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((int(r - min_r), int(c - min_c)) for r, c in cells)
    return tuple(norm)


def _random_hole_key(pair_count: int) -> tuple[tuple[int, int], ...]:
    max_shape_h = 4 if pair_count == 1 else 3
    max_shape_w = 4 if pair_count == 1 else 3
    max_area = 6 if pair_count == 1 else 5
    top_margin = random.randint(1, 3 if pair_count == 1 else 2)
    left_margin = random.randint(1, 3 if pair_count == 1 else 2)
    base_shape = _random_connected_shape(max_shape_h, max_shape_w, max_area)
    return tuple(sorted((int(top_margin + r), int(left_margin + c)) for r, c in base_shape))


def _random_rect_size(hole_key: tuple[tuple[int, int], ...], pair_count: int) -> tuple[int, int]:
    max_r = max(r for r, _ in hole_key)
    max_c = max(c for _, c in hole_key)
    extra_cap = {1: 4, 2: 3, 3: 2}[pair_count]
    height = int(max_r + 2 + random.randint(0, extra_cap))
    width = int(max_c + 2 + random.randint(0, extra_cap))
    return height, width


def _build_objects(pair_count: int) -> list[dict] | None:
    colors = random.sample(list(OBJECT_COLORS), 2 * pair_count)
    objects: list[dict] = []
    used_holes: set[tuple[tuple[int, int], ...]] = set()

    for pair_id in range(pair_count):
        hole_key = None
        for _ in range(80):
            candidate = _random_hole_key(pair_count)
            if candidate not in used_holes:
                hole_key = candidate
                used_holes.add(candidate)
                break
        if hole_key is None:
            return None

        same_size = random.choice((False, True))
        size_a = _random_rect_size(hole_key, pair_count)
        size_b = size_a if same_size else _random_rect_size(hole_key, pair_count)

        objects.append(
            {
                "id": len(objects),
                "pair_id": int(pair_id),
                "color": int(colors[2 * pair_id]),
                "swap_color": int(colors[2 * pair_id + 1]),
                "height": int(size_a[0]),
                "width": int(size_a[1]),
                "hole_key": hole_key,
            }
        )
        objects.append(
            {
                "id": len(objects),
                "pair_id": int(pair_id),
                "color": int(colors[2 * pair_id + 1]),
                "swap_color": int(colors[2 * pair_id]),
                "height": int(size_b[0]),
                "width": int(size_b[1]),
                "hole_key": hole_key,
            }
        )

    return objects


def _random_grid_shape(objects: list[dict]) -> tuple[int, int] | None:
    max_h = max(int(obj["height"]) for obj in objects) + 2
    max_w = max(int(obj["width"]) for obj in objects) + 2
    total_area = sum(int(obj["height"]) * int(obj["width"]) for obj in objects)
    target_area = total_area + 24 * len(objects)
    base = int(np.ceil(np.sqrt(target_area)))
    if max(max_h, max_w) > 30:
        return None

    h_low = max(10, max_h, base - 2)
    w_low = max(10, max_w, base - 2)
    if h_low > 30 or w_low > 30:
        return None

    aspect = random.choice(("balanced", "tall", "wide"))
    if aspect == "balanced":
        h = random.randint(h_low, min(30, h_low + 8))
        w = random.randint(w_low, min(30, w_low + 8))
    elif aspect == "tall":
        h = random.randint(h_low, min(30, h_low + 10))
        w = random.randint(w_low, min(30, w_low + 5))
    else:
        h = random.randint(h_low, min(30, h_low + 5))
        w = random.randint(w_low, min(30, w_low + 10))

    while h * w < target_area and (h < 30 or w < 30):
        if h <= w and h < 30:
            h += 1
        elif w < 30:
            w += 1
        elif h < 30:
            h += 1
        else:
            break

    if h > 30 or w > 30:
        return None
    return int(h), int(w)


def _place_objects(objects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    placed: dict[int, tuple[int, int]] = {}
    order = sorted(objects, key=lambda obj: int(obj["height"]) * int(obj["width"]), reverse=True)

    for obj in order:
        oh = int(obj["height"])
        ow = int(obj["width"])
        if oh + 2 > h or ow + 2 > w:
            return None

        candidates = []
        for top in range(1, h - oh):
            for left in range(1, w - ow):
                r0 = top - 1
                r1 = top + oh + 1
                c0 = left - 1
                c1 = left + ow + 1
                if not occupied[r0:r1, c0:c1].any():
                    candidates.append((int(top), int(left)))

        if not candidates:
            return None

        top, left = random.choice(candidates)
        occupied[top : top + oh, left : left + ow] = True
        placed[int(obj["id"])] = (int(top), int(left))

    out = []
    for obj in objects:
        top, left = placed[int(obj["id"])]
        enriched = dict(obj)
        enriched["top"] = int(top)
        enriched["left"] = int(left)
        out.append(enriched)
    return out


def _render_scene(shape: tuple[int, int], objects: list[dict], swapped: bool) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)

    for obj in objects:
        top = int(obj["top"])
        left = int(obj["left"])
        oh = int(obj["height"])
        ow = int(obj["width"])
        color = int(obj["swap_color"] if swapped else obj["color"])
        grid[top : top + oh, left : left + ow] = color
        for hr, hc in obj["hole_key"]:
            grid[top + int(hr), left + int(hc)] = BG

    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    objects = _build_objects(pair_count)
    if objects is None:
        return None

    shape = _random_grid_shape(objects)
    if shape is None:
        return None

    placed = _place_objects(objects, shape)
    if placed is None:
        return None

    inp = _render_scene(shape, placed, swapped=False)
    expected = _render_scene(shape, placed, swapped=True)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
