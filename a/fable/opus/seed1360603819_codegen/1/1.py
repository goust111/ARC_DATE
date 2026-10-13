import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
AZURE = 8
MAROON = 9

LABEL_COLORS = [BLUE, RED, GREEN, YELLOW, AZURE, MAGENTA]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 300
SHAPE_RETRIES = 80


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((int(r), int(c)))
                for dr, dc in NEIGHBORS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for dr, dc in NEIGHBORS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_border:
                holes += 1
    return holes


def _divider_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert rows, "expected a full-width grey divider row"
    for r in rows:
        if np.any(grid[:r] == MAROON) and np.any(grid[r + 1 :] == GREY):
            return int(r)
    return int(rows[0])


def _legend_mapping(top: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for coords in _components(top == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        mask = top[r0 : r1 + 1, c0 : c1 + 1] == MAROON
        hole_count = _hole_count(mask)
        label_colors = set()
        for r, c in coords:
            for dr, dc in NEIGHBORS4:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < top.shape[0] and 0 <= cc < top.shape[1]:
                    value = int(top[rr, cc])
                    if value not in (BLACK, GREY, MAROON):
                        label_colors.add(value)
        assert len(label_colors) == 1, "each legend block should have exactly one adjacent label color"
        mapping[int(hole_count)] = int(next(iter(label_colors)))
    assert mapping, "legend mapping must be non-empty"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a hole-count -> color legend from the top panel, then recolor the lower
    grey objects by descending area rank using counts 1, 2, 3, ...
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _divider_row(g)
    mapping = _legend_mapping(g[:divider, :])

    out = g.copy()
    lower = g[divider + 1 :, :]
    grey_objects = []
    for coords in _components(lower == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        grey_objects.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": (r0, c0, r1, c1),
            }
        )

    grey_objects.sort(key=lambda obj: (-obj["area"], obj["bbox"][0], obj["bbox"][1]))
    for rank, obj in enumerate(grey_objects, start=1):
        assert rank in mapping, "missing legend entry for object rank"
        color = int(mapping[rank])
        coords = obj["coords"]
        out[coords[:, 0] + divider + 1, coords[:, 1]] = color

    return out


def _legend_block(hole_count: int, orientation: str) -> np.ndarray:
    if orientation == "horizontal":
        block = np.full((3, 2 * hole_count + 1), MAROON, dtype=int)
        block[1, 1::2] = BLACK
    else:
        block = np.full((2 * hole_count + 1, 3), MAROON, dtype=int)
        block[1::2, 1] = BLACK
    return block


def _region_is_free(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(occupied.shape[0], r + h + pad)
    c1 = min(occupied.shape[1], c + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_region(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(occupied.shape[0], r + h + pad)
    c1 = min(occupied.shape[1], c + w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_legends(top: np.ndarray, specs: list[dict]) -> bool:
    occupied = np.zeros_like(top, dtype=bool)
    ordered = sorted(specs, key=lambda spec: -(spec["block"].shape[0] * spec["block"].shape[1]))
    for spec in ordered:
        block = spec["block"]
        bh, bw = block.shape
        tag_side = random.choice(["above", "below"])
        tag_offset = random.randrange(bw)
        item_h = bh + 1
        item_w = bw
        positions = []
        for r in range(top.shape[0] - item_h + 1):
            for c in range(top.shape[1] - item_w + 1):
                if _region_is_free(occupied, r, c, item_h, item_w, pad=1):
                    positions.append((r, c))
        if not positions:
            return False

        r, c = random.choice(positions)
        block_r = r + 1 if tag_side == "above" else r
        tag_r = r if tag_side == "above" else r + bh
        tag_c = c + tag_offset

        top[block_r : block_r + bh, c : c + bw] = block
        top[tag_r, tag_c] = int(spec["color"])
        _mark_region(occupied, r, c, item_h, item_w, pad=1)
    return True


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_shape(area: int) -> np.ndarray | None:
    seed_options = [
        {(0, 0)},
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0)},
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (1, 0), (1, 1)},
    ]
    valid_seeds = [seed for seed in seed_options if len(seed) <= area]
    for _ in range(SHAPE_RETRIES):
        cells = set(random.choice(valid_seeds))
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    frontier.add((r + dr, c + dc))
            frontier -= cells

            candidates = []
            for candidate in frontier:
                new_cells = cells | {candidate}
                rows = [r for r, _ in new_cells]
                cols = [c for _, c in new_cells]
                h = max(rows) - min(rows) + 1
                w = max(cols) - min(cols) + 1
                if h > 7 or w > 9:
                    continue
                if max(h, w) > 3 * min(h, w) + 2:
                    continue
                contacts = sum((candidate[0] + dr, candidate[1] + dc) in cells for dr, dc in NEIGHBORS4)
                score = contacts * 3.0 - abs(h - w) * 0.2 + random.random()
                candidates.append((score, candidate))
            if not candidates:
                break

            candidates.sort(reverse=True)
            top = [candidate for _score, candidate in candidates[: min(6, len(candidates))]]
            cells.add(random.choice(top))

        if len(cells) != area:
            continue

        mask = _mask_from_cells(cells)
        if random.choice([True, False]):
            mask = mask[:, ::-1]
        if random.choice([True, False]):
            mask = mask[::-1, :]
        if random.choice([True, False]):
            mask = mask.T

        h, w = mask.shape
        if area >= 6 and (h == 1 or w == 1):
            continue
        if area / float(h * w) < 0.35:
            continue
        if _hole_count(mask):
            continue
        return mask
    return None


def _mark_forbidden(forbidden: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = r0 + int(r)
        cc = c0 + int(c)
        rr0 = max(0, rr - 1)
        cc0 = max(0, cc - 1)
        rr1 = min(forbidden.shape[0], rr + 2)
        cc1 = min(forbidden.shape[1], cc + 2)
        forbidden[rr0:rr1, cc0:cc1] = True


def _place_objects(bottom: np.ndarray, objects: list[dict]) -> list[dict] | None:
    forbidden = np.zeros_like(bottom, dtype=bool)
    placed: list[dict] = []
    ordered = sorted(objects, key=lambda obj: -obj["area"])
    for obj in ordered:
        mask = obj["mask"]
        h, w = mask.shape
        positions = []
        row_choices = range(bottom.shape[0] - h + 1)
        col_choices = range(bottom.shape[1] - w + 1)
        for r in row_choices:
            for c in col_choices:
                if np.any(forbidden[r : r + h, c : c + w] & mask):
                    continue
                positions.append((r, c))
        if not positions:
            return None

        r, c = random.choice(positions)
        bottom[r : r + h, c : c + w][mask] = GREY
        _mark_forbidden(forbidden, mask, r, c)
        placed.append({"area": obj["area"], "mask": mask, "r": r, "c": c})
    return placed


def _expected_output(grid: np.ndarray, divider: int, placed: list[dict], count_to_color: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for rank, obj in enumerate(sorted(placed, key=lambda item: -item["area"]), start=1):
        mask = obj["mask"]
        r = obj["r"] + divider + 1
        c = obj["c"]
        out[r : r + mask.shape[0], c : c + mask.shape[1]][mask] = int(count_to_color[rank])
    return out


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(2, 6)
    count_to_color = {i + 1: color for i, color in enumerate(random.sample(LABEL_COLORS, object_count))}

    specs = []
    for hole_count in random.sample(list(range(1, object_count + 1)), object_count):
        block = _legend_block(hole_count, random.choice(["horizontal", "vertical"]))
        specs.append({"hole_count": hole_count, "color": count_to_color[hole_count], "block": block})

    max_item_h = max(spec["block"].shape[0] + 1 for spec in specs)
    max_item_w = max(spec["block"].shape[1] for spec in specs)
    top_min = max(6, max_item_h)
    top_max = min(16, top_min + 3)
    if top_min > top_max:
        return None
    top_h = random.randint(top_min, top_max)

    bottom_min = max(8, object_count + 6)
    bottom_max = min(15, 30 - top_h - 1)
    if bottom_min > bottom_max:
        return None
    bottom_h = random.randint(bottom_min, bottom_max)

    width_min = max(20, max_item_w + 2, 4 * object_count + 6)
    if width_min > 30:
        return None
    width = random.randint(width_min, 30)

    top = np.full((top_h, width), BLACK, dtype=int)
    if not _place_legends(top, specs):
        return None

    area_choices = list(range(5, min(19, bottom_h + 7)))
    if len(area_choices) < object_count:
        return None
    areas = sorted(random.sample(area_choices, object_count), reverse=True)

    objects = []
    for area in areas:
        mask = _random_shape(area)
        if mask is None:
            return None
        objects.append({"area": area, "mask": mask})

    bottom = np.full((bottom_h, width), BLACK, dtype=int)
    placed = _place_objects(bottom, objects)
    if placed is None:
        return None

    grid = np.full((top_h + 1 + bottom_h, width), BLACK, dtype=int)
    grid[:top_h, :] = top
    grid[top_h, :] = GREY
    grid[top_h + 1 :, :] = bottom

    expected = _expected_output(grid, top_h, placed, count_to_color)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
