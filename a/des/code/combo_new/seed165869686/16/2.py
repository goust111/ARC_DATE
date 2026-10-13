from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 8
ARC_MAX_SIDE = 30
OBJECT_COLOR_POOL = tuple(color for color in range(10) if color != BACKGROUND)
GROUP_SUBSETS = (
    (2,),
    (3,),
    (4,),
    (2, 3),
    (2, 4),
    (3, 4),
    (2, 3, 4),
)
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_COUNT = 5


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < height:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < width:
        out.append((r, c + 1))
    return out


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects: list[dict[str, object]] = []

    for coords in _find_components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        patch[rows - r0, cols - c0] = g[rows, cols]
        colors = sorted(int(color) for color in np.unique(patch[patch != bg]))
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "area": int(coords.shape[0]),
                "n_colors": len(colors),
            }
        )

    return bg, objects


def _pack_objects(bg: int, objects: list[dict[str, object]]) -> np.ndarray:
    assert objects, "expected at least one object"
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for obj in objects:
        grouped[int(obj["n_colors"])].append(obj)

    ordered_groups: list[list[dict[str, object]]] = []
    for n_colors in sorted(grouped):
        group = sorted(
            grouped[n_colors],
            key=lambda obj: (
                -int(obj["area"]),
                int(obj["bbox"][0]),  # type: ignore[index]
                int(obj["bbox"][1]),  # type: ignore[index]
            ),
        )
        ordered_groups.append(group)

    out_h = max(int(obj["patch"].shape[0]) for obj in objects)  # type: ignore[index]
    group_widths = [
        sum(int(obj["patch"].shape[1]) for obj in group) + max(0, len(group) - 1)  # type: ignore[index]
        for group in ordered_groups
    ]
    out_w = sum(group_widths) + max(0, len(ordered_groups) - 1) * 2
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for gi, group in enumerate(ordered_groups):
        if gi > 0:
            col += 2
        cursor = col
        for oi, obj in enumerate(group):
            patch = obj["patch"]  # type: ignore[assignment]
            ph, pw = patch.shape
            out[out_h - ph : out_h, cursor : cursor + pw] = patch
            cursor += pw
            if oi + 1 < len(group):
                cursor += 1
        col = cursor

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered multicolor objects by their internal color count and pack each group into a bottom-aligned horizontal composite.

    Input structure:
    - The grid has one background color.
    - Several disjoint 4-connected non-background objects are scattered across the canvas.
    - Each object is irregular and uses between 2 and 4 distinct colors internally.

    Rule:
    - Treat each 4-connected non-background component as one object and count how many distinct non-background colors it contains.
    - Group objects by that count N.
    - Within each group, sort objects by area from large to small, crop each object to its tight bounding box, and place those crops left-to-right with one background column between consecutive boxes and all crops bottom-aligned.
    - Place the groups left-to-right in increasing N order with two background columns between neighboring groups.
    - The output height is the tallest object bounding box and the output width is the total packed group width plus the group separators.

    Key insight:
    - The organizing signal is the number of different colors inside each object, not the object's shape, location, or any particular color identity; packing only happens after those color-count classes are discovered.
    """
    bg, objects = _extract_objects(np.asarray(grid, dtype=int))
    assert objects, "expected at least one non-background object"
    for obj in objects:
        assert 2 <= int(obj["n_colors"]) <= 4, "objects must contain between 2 and 4 colors"
    return _pack_objects(bg, objects)


def _partition(total: int, parts: int) -> list[int]:
    counts = [1] * parts
    for _ in range(total - parts):
        counts[random.randrange(parts)] += 1
    random.shuffle(counts)
    return counts


def _random_connected_mask(area: int) -> np.ndarray | None:
    for _ in range(120):
        height = random.randint(3, 5)
        width = random.randint(3, 5)
        if height * width <= area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True
        frontier = set(_neighbors4(start[0], start[1], height, width))

        while int(mask.sum()) < area and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            if mask[rr, cc]:
                continue
            mask[rr, cc] = True
            for nr, nc in _neighbors4(rr, cc, height, width):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != area:
            continue

        cropped = _crop_mask(mask)
        if min(cropped.shape) < 2:
            continue
        if bool(np.all(cropped)):
            continue
        return cropped
    return None


def _colorize_mask(mask: np.ndarray, n_colors: int) -> np.ndarray:
    area = int(mask.sum())
    counts = _partition(area, n_colors)
    palette = random.sample(OBJECT_COLOR_POOL, n_colors)
    values: list[int] = []
    for color, count in zip(palette, counts):
        values.extend([color] * count)
    random.shuffle(values)

    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    coords = np.argwhere(mask)
    for (r, c), value in zip(coords, values):
        patch[int(r), int(c)] = int(value)
    return patch


def _random_transform(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, random.randrange(4)).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    if random.random() < 0.5:
        out = np.flipud(out).copy()
    return out


def _make_object_patch(n_colors: int, area: int) -> np.ndarray | None:
    mask = _random_connected_mask(area)
    if mask is None:
        return None
    return _random_transform(_colorize_mask(mask, n_colors))


def _sample_group_counts() -> dict[int, int]:
    groups = list(random.choice(GROUP_SUBSETS))
    min_total = max(2 if len(groups) == 1 else 3, len(groups))
    max_total = min(MAX_OBJECT_COUNT, 3 * len(groups))
    target_total = random.randint(min_total, max_total)

    counts = {group: 1 for group in groups}
    remaining = target_total - len(groups)
    while remaining > 0:
        choices = [group for group in groups if counts[group] < 3]
        chosen = random.choice(choices)
        counts[chosen] += 1
        remaining -= 1
    return counts


def _sample_area_choices(n_colors: int, count: int) -> list[int]:
    low = max(4, n_colors + 1)
    high = 10
    pool = list(range(low, high + 1))
    return random.sample(pool, count)


def _build_object_specs() -> list[dict[str, object]] | None:
    specs: list[dict[str, object]] = []
    for n_colors, count in _sample_group_counts().items():
        for area in _sample_area_choices(n_colors, count):
            patch = _make_object_patch(n_colors, area)
            if patch is None:
                return None
            specs.append({"n_colors": n_colors, "area": area, "patch": patch})
    random.shuffle(specs)
    return specs


def _canvas_candidates(patches: list[np.ndarray]) -> list[tuple[int, int]]:
    dims = [patch.shape for patch in patches]
    total_area = sum(int(np.count_nonzero(patch != BACKGROUND)) for patch in patches)
    padded_area = sum((h + 2) * (w + 2) for h, w in dims)
    min_h = max(12, max(h for h, _ in dims) + 2)
    min_w = max(12, max(w for _, w in dims) + 2)

    candidates = []
    for height in range(min_h, ARC_MAX_SIDE + 1):
        for width in range(min_w, ARC_MAX_SIDE + 1):
            if height * width < padded_area:
                continue
            if height * width <= 2 * total_area:
                continue
            candidates.append((height, width))
    return candidates


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    candidates = _canvas_candidates(patches)
    if not candidates:
        return None
    height, width = random.choice(candidates)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        positions = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + pw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if max(grid.shape) > ARC_MAX_SIDE or max(out.shape) > ARC_MAX_SIDE:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _build_object_specs()
    if specs is None:
        return None

    patches = [spec["patch"] for spec in specs]  # type: ignore[misc]
    grid = _place_patches(patches)
    if grid is None:
        return None
    if not _is_valid_input(grid):
        return None

    bg, objects = _extract_objects(grid)
    groups_present = {int(obj["n_colors"]) for obj in objects}
    if len(objects) != len(specs):
        return None
    if not groups_present:
        return None

    out = _pack_objects(bg, objects)
    if max(out.shape) > ARC_MAX_SIDE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
