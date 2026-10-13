from __future__ import annotations

import random

import numpy as np


Container = dict[str, object]
Seed = dict[str, object]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(order) > 0, "expected a nonempty grid"
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]]), "background must be uniquely dominant"
    return int(values[order[0]])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _analyze_container(grid: np.ndarray, color: int, coords: np.ndarray, bg: int) -> Container | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not (local[0, :].any() and local[-1, :].any() and local[:, 0].any() and local[:, -1].any()):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None
    stripe_mask = (inner != bg) & (inner != color)
    if not stripe_mask.any():
        return None
    if not np.all((inner == bg) | stripe_mask):
        return None

    stripe_components = _components(stripe_mask)
    if len(stripe_components) != 1:
        return None
    stripe = stripe_components[0]
    stripe_colors = np.unique(inner[stripe_mask])
    if stripe_colors.size != 1:
        return None

    stripe_rows = np.unique(stripe[:, 0])
    stripe_cols = np.unique(stripe[:, 1])
    inner_h, inner_w = inner.shape
    orient: str | None = None
    if stripe_rows.size == 1 and stripe.shape[0] == inner_w and stripe_cols.size == inner_w:
        orient = "h"
    elif stripe_cols.size == 1 and stripe.shape[0] == inner_h and stripe_rows.size == inner_h:
        orient = "v"
    if orient is None:
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "size": (h, w),
        "orient": orient,
        "stripe_color": int(stripe_colors[0]),
    }


def _find_containers(grid: np.ndarray, bg: int) -> list[Container]:
    candidates: dict[int, list[Container]] = {}
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        parts = _components(grid == color)
        found: list[Container] = []
        for coords in parts:
            info = _analyze_container(grid, color, coords, bg)
            if info is not None:
                found.append(info)
        if found:
            candidates[color] = found

    assert candidates, "expected at least one grey container color"
    ranked = sorted(candidates.items(), key=lambda kv: len(kv[1]), reverse=True)
    if len(ranked) > 1:
        assert len(ranked[0][1]) > len(ranked[1][1]), "container color must be uniquely identifiable"
    return ranked[0][1]


def _seed_fill_mask(seed_mask: np.ndarray, orient: str) -> np.ndarray:
    filled = np.zeros_like(seed_mask, dtype=bool)
    if orient == "h":
        rows = np.any(seed_mask, axis=1)
        filled[rows, :] = True
    else:
        cols = np.any(seed_mask, axis=0)
        filled[:, cols] = True
    return filled


def _extract_seeds(grid: np.ndarray, bg: int, container_mask: np.ndarray, container_color: int) -> list[Seed]:
    seeds: list[Seed] = []
    for color in map(int, np.unique(grid)):
        if color in {bg, container_color}:
            continue
        mask = (grid == color) & ~container_mask
        if not mask.any():
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        seeds.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "size": local.shape,
                "mask": local,
            }
        )
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed blob to the grey container with the same bounding-box size,
    then expand the seed into horizontal or vertical stripes according to the
    matched container's internal band orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _dominant_color(g)
    containers = _find_containers(g, bg)
    container_color = int(containers[0]["color"])
    assert all(int(container["color"]) == container_color for container in containers), "containers must share one color"

    size_to_container: dict[tuple[int, int], Container] = {}
    container_mask = np.zeros(g.shape, dtype=bool)
    for container in containers:
        size = tuple(map(int, container["size"]))
        assert size not in size_to_container, "container sizes must be unique"
        size_to_container[size] = container
        r0, c0, r1, c1 = map(int, container["bbox"])
        container_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    seeds = _extract_seeds(g, bg, container_mask, container_color)
    assert len(seeds) == len(containers), "expected a one-to-one seed/container correspondence"

    out = g.copy()
    for seed in seeds:
        size = tuple(map(int, seed["size"]))
        assert size in size_to_container, "seed bbox must match a container bbox"
        container = size_to_container[size]
        orient = str(container["orient"])
        r0, c0, r1, c1 = map(int, seed["bbox"])
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        fill = _seed_fill_mask(np.asarray(seed["mask"], dtype=bool), orient)
        region[fill] = int(seed["color"])
    return out


# --------------- generator ---------------

GREY = 5
PAIR_COUNT_RANGE = (2, 4)
SIZE_POOL = ((5, 5), (5, 7), (7, 5), (7, 7))
GRID_RANGES = {
    2: (18, 24),
    3: (23, 28),
    4: (30, 32),
}
MAX_GENERATION_ATTEMPTS = 400


def _sample_orientations(count: int) -> list[str]:
    orientations = ["h", "v"] * (count // 2)
    if count % 2:
        orientations.append(random.choice(["h", "v"]))
    random.shuffle(orientations)
    return orientations


def _seed_template(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[::2, ::2] = True
    return mask


def _draw_container(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int, orient: str) -> None:
    grid[top : top + height, left] = GREY
    grid[top : top + height, left + width - 1] = GREY
    grid[top, left : left + width] = GREY
    grid[top + height - 1, left : left + width] = GREY
    if orient == "h":
        stripe_r = top + 1 + random.randrange(height - 2)
        grid[stripe_r, left + 1 : left + width - 1] = color
    else:
        stripe_c = left + 1 + random.randrange(width - 2)
        grid[top + 1 : top + height - 1, stripe_c] = color


def _draw_seed(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    region = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    region[mask] = color


def _paint_seed_output(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int, orient: str) -> None:
    region = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    region[_seed_fill_mask(mask, orient)] = color


def _place_rectangles(zone_h: int, zone_w: int, sizes: list[tuple[int, int]], gap: int = 1) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    occupied = np.zeros((zone_h, zone_w), dtype=bool)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)

    for idx in order:
        height, width = sizes[idx]
        valid: list[tuple[int, int]] = []
        for top in range(zone_h - height + 1):
            for left in range(zone_w - width + 1):
                r0 = max(0, top - gap)
                c0 = max(0, left - gap)
                r1 = min(zone_h, top + height + gap)
                c1 = min(zone_w, left + width + gap)
                if not occupied[r0:r1, c0:c1].any():
                    valid.append((top, left))
        if not valid:
            return None
        top, left = random.choice(valid)
        placements[idx] = (top, left)
        r0 = max(0, top - gap)
        c0 = max(0, left - gap)
        r1 = min(zone_h, top + height + gap)
        c1 = min(zone_w, left + width + gap)
        occupied[r0:r1, c0:c1] = True

    return [tuple(p) for p in placements if p is not None]


def _build_zones(height: int, width: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if random.random() < 0.5:
        split = width // 2
        left_zone = (0, 0, height, split)
        right_zone = (0, split, height, width - split)
        return (left_zone, right_zone) if random.random() < 0.5 else (right_zone, left_zone)

    split = height // 2
    top_zone = (0, 0, split, width)
    bottom_zone = (split, 0, height - split, width)
    return (top_zone, bottom_zone) if random.random() < 0.5 else (bottom_zone, top_zone)


def _stamp_layout(
    grid: np.ndarray,
    output: np.ndarray,
    sizes: list[tuple[int, int]],
    colors: list[int],
    orientations: list[str],
    container_zone: tuple[int, int, int, int],
    seed_zone: tuple[int, int, int, int],
) -> bool:
    cz_r0, cz_c0, cz_h, cz_w = container_zone
    sz_r0, sz_c0, sz_h, sz_w = seed_zone
    container_pos = _place_rectangles(cz_h, cz_w, sizes)
    seed_pos = _place_rectangles(sz_h, sz_w, sizes)
    if container_pos is None or seed_pos is None:
        return False

    for (h, w), color, orient, (ctop, cleft), (stop, sleft) in zip(
        sizes, colors, orientations, container_pos, seed_pos, strict=True
    ):
        _draw_container(grid, cz_r0 + ctop, cz_c0 + cleft, h, w, color, orient)
        _draw_container(output, cz_r0 + ctop, cz_c0 + cleft, h, w, color, orient)

        mask = _seed_template(h, w)
        _draw_seed(grid, sz_r0 + stop, sz_c0 + sleft, mask, color)
        _draw_seed(output, sz_r0 + stop, sz_c0 + sleft, mask, color)
        _paint_seed_output(output, sz_r0 + stop, sz_c0 + sleft, mask, color, orient)

    return True


def _valid_sample(grid: np.ndarray, expected: np.ndarray, pair_count: int) -> bool:
    try:
        bg = _dominant_color(grid)
        containers = _find_containers(grid, bg)
        if len(containers) != pair_count:
            return False
        sizes = [tuple(map(int, container["size"])) for container in containers]
        if len(set(sizes)) != pair_count:
            return False
        container_mask = np.zeros(grid.shape, dtype=bool)
        for container in containers:
            r0, c0, r1, c1 = map(int, container["bbox"])
            container_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        seeds = _extract_seeds(grid, bg, container_mask, GREY)
        if len(seeds) != pair_count:
            return False
        if len({tuple(map(int, seed["size"])) for seed in seeds}) != pair_count:
            return False
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    height = random.randint(*GRID_RANGES[pair_count])
    width = random.randint(*GRID_RANGES[pair_count])

    bg_choices = [c for c in range(10) if c != GREY]
    bg = random.choice(bg_choices)
    pair_colors = random.sample([c for c in range(10) if c not in {bg, GREY}], pair_count)
    sizes = random.sample(list(SIZE_POOL), pair_count)
    orientations = _sample_orientations(pair_count)

    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()
    container_zone, seed_zone = _build_zones(height, width)
    if not _stamp_layout(grid, expected, sizes, pair_colors, orientations, container_zone, seed_zone):
        return None

    if not _valid_sample(grid, expected, pair_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _generate_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid sample")
