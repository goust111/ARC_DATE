import math
from collections import Counter

import numpy as np


MARKER_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
ANGLE_SLOTS = [i * (math.pi / 4.0) for i in range(8)]
QUADRANT_SLOTS = ((7, 0), (1, 2), (3, 4), (5, 6))
REGION_PATCHES = (
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: list[int] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)
    if colors is None:
        colors = sorted(set(map(int, grid.ravel())) - {bg})

    out: list[dict] = []
    for color in colors:
        if color == bg:
            continue
        for coords in components_mask(grid == int(color), connectivity=connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                }
            )
    return out


def centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def coords_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def clockwise_angle(center: np.ndarray, coords: np.ndarray) -> float:
    pt = centroid(coords)
    return float(math.atan2(pt[0] - center[0], pt[1] - center[1]) % (2.0 * math.pi))


def isolated_marker_cells(grid: np.ndarray, bg: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for comp in find_components(grid, bg=bg, colors=[MARKER_COLOR], connectivity=4):
        if comp["area"] != 1:
            continue
        r, c = map(int, comp["coords"][0])
        if all(
            grid[nr, nc] == bg
            for nr, nc in neighbors(r, c, 8)
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
        ):
            cells.add((r, c))
    return cells


def analyze_scene(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    marker_cells = isolated_marker_cells(g, bg)

    objects: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4):
        if comp["color"] == MARKER_COLOR and comp["area"] == 1:
            cell = tuple(map(int, comp["coords"][0]))
            if cell in marker_cells:
                continue
        objects.append(comp)

    assert len(objects) >= 3, "expected divider plus multiple regions"
    centroids = [centroid(comp["coords"]) for comp in objects]
    mean_dists = []
    for i, ctr in enumerate(centroids):
        dists = [float(np.linalg.norm(ctr - other)) for j, other in enumerate(centroids) if j != i]
        mean_dists.append(sum(dists) / len(dists))

    divider_idx = min(range(len(objects)), key=lambda i: (mean_dists[i], -objects[i]["area"]))
    divider = objects[divider_idx]
    center = centroid(divider["coords"])

    regions = [objects[i] for i in range(len(objects)) if i != divider_idx]
    regions.sort(
        key=lambda comp: (
            clockwise_angle(center, comp["coords"]),
            float(np.linalg.norm(centroid(comp["coords"]) - center)),
            coords_key(comp["coords"]),
        )
    )
    return bg, sorted(marker_cells), divider, regions


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate region colors clockwise around the central divider by a marker-count step.

    Input structure:
    - A dominant background surrounds one fixed-color divider structure near the center.
    - Several other non-background colors form separated monochrome 4-connected regions around that divider.
    - Additional gray singleton marker pixels sit on background cells and do not touch any region.

    Rule:
    - Keep the divider structure and every region shape in the same place.
    - Order the non-divider regions by the clockwise angle of their centroids around the divider centroid.
    - Count the gray marker pixels as k, erase them, and rotate the region colors forward by k positions along that clockwise cycle.

    Key insight:
    - The gray dots are not objects to preserve; they are a discarded count knob that determines how far the surrounding region colors rotate.
    """
    g = np.asarray(grid, dtype=int)
    bg, marker_cells, _divider, regions = analyze_scene(g)

    out = g.copy()
    for r, c in marker_cells:
        out[r, c] = bg

    region_count = len(regions)
    if region_count == 0:
        return out
    shift = len(marker_cells) % region_count
    colors = [int(comp["color"]) for comp in regions]
    for i, comp in enumerate(regions):
        new_color = colors[(i - shift) % region_count]
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)
    return out


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    out = np.array(coords, dtype=int)
    out[:, 0] -= out[:, 0].min()
    out[:, 1] -= out[:, 1].min()
    return out


def _patch_coords(patch: np.ndarray) -> np.ndarray:
    return np.argwhere(patch > 0).astype(int)


def _random_region_shape(rng: np.random.Generator) -> np.ndarray:
    patch = np.array(REGION_PATCHES[int(rng.integers(len(REGION_PATCHES)))], dtype=int)
    if int(rng.integers(2)):
        patch = np.fliplr(patch)
    patch = np.rot90(patch, int(rng.integers(4)))
    return _normalize_coords(_patch_coords(patch))


def _divider_shape(kind: str, rng: np.random.Generator) -> np.ndarray:
    cells: set[tuple[int, int]] = set()
    if kind == "plus":
        arm_r = int(rng.integers(2, 4))
        arm_c = int(rng.integers(2, 5))
        for dr in range(-arm_r, arm_r + 1):
            cells.add((dr, 0))
        for dc in range(-arm_c, arm_c + 1):
            cells.add((0, dc))
    elif kind == "diamond":
        radius = int(rng.integers(2, 4))
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) <= radius:
                    cells.add((dr, dc))
    else:
        half_h = int(rng.integers(2, 4))
        half_w = int(rng.integers(2, 4))
        for dc in range(-half_w, half_w + 1):
            cells.add((-half_h, dc))
            cells.add((half_h, dc))
        for dr in range(-half_h, half_h + 1):
            cells.add((dr, -half_w))
            cells.add((dr, half_w))
    coords = np.array(sorted(cells), dtype=int)
    return coords


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray, pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _sample_slots(count: int, rng: np.random.Generator) -> list[int]:
    chosen = {int(rng.choice(group)) for group in QUADRANT_SLOTS}
    remaining = [slot for slot in range(8) if slot not in chosen]
    rng.shuffle(remaining)
    while len(chosen) < count:
        chosen.add(int(remaining.pop()))
    return sorted(chosen, key=lambda idx: ANGLE_SLOTS[idx])


def _place_region(
    blocked: np.ndarray,
    center: np.ndarray,
    angle: float,
    shape: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    h, w = blocked.shape
    shape_center = centroid(shape)
    max_radius = int(
        min(center[0] - 2, center[1] - 2, (h - 1) - center[0] - 2, (w - 1) - center[1] - 2)
    )
    if max_radius < 6:
        return None
    low = 6
    high = min(9, max_radius)
    for _ in range(80):
        radius = int(rng.integers(low, high + 1))
        target = np.array(
            [
                center[0] + math.sin(angle) * radius + int(rng.integers(-1, 2)),
                center[1] + math.cos(angle) * radius + int(rng.integers(-1, 2)),
            ],
            dtype=float,
        )
        offset = np.rint(target - shape_center).astype(int)
        placed = shape + offset
        if (
            placed[:, 0].min() < 0
            or placed[:, 1].min() < 0
            or placed[:, 0].max() >= h
            or placed[:, 1].max() >= w
        ):
            continue
        if any(blocked[int(r), int(c)] for r, c in placed):
            continue
        return placed
    return None


def _candidate_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != bg:
                continue
            if all(
                grid[nr, nc] == bg
                for nr, nc in neighbors(r, c, 8)
                if 0 <= nr < h and 0 <= nc < w
            ):
                out.append((r, c))
    return out


def _choose_marker_cells(
    candidates: list[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    cand = list(candidates)
    rng.shuffle(cand)
    chosen: list[tuple[int, int]] = []
    for cell in cand:
        if all(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) > 1 for other in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _expected_output(
    grid: np.ndarray,
    bg: int,
    markers: list[tuple[int, int]],
    ordered_regions: list[dict],
    shift: int,
) -> np.ndarray:
    out = grid.copy()
    for r, c in markers:
        out[r, c] = bg
    colors = [int(region["color"]) for region in ordered_regions]
    n = len(ordered_regions)
    for i, region in enumerate(ordered_regions):
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = colors[(i - shift) % n]
    return out


def _scene_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    h = int(rng.choice(np.array([19, 21, 23, 25, 27, 29], dtype=int)))
    w = int(rng.choice(np.array([19, 21, 23, 25, 27, 29], dtype=int)))
    bg = 0
    grid = np.full((h, w), bg, dtype=int)

    divider_color = int(rng.choice(PALETTE))
    divider_rel = _divider_shape(str(rng.choice(np.array(["plus", "diamond", "frame"], dtype=object))), rng)
    anchor = np.array(
        [h // 2 + int(rng.integers(-1, 2)), w // 2 + int(rng.integers(-1, 2))],
        dtype=int,
    )
    divider = divider_rel + anchor
    if (
        divider[:, 0].min() < 0
        or divider[:, 1].min() < 0
        or divider[:, 0].max() >= h
        or divider[:, 1].max() >= w
    ):
        return None
    _paint(grid, divider, divider_color)

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, divider, pad=1)
    true_center = centroid(divider)

    region_count = int(rng.integers(4, 8))
    available_colors = [color for color in PALETTE if color != divider_color]
    region_colors = [int(color) for color in rng.choice(available_colors, size=region_count, replace=False)]
    slot_ids = _sample_slots(region_count, rng)

    regions_meta: list[dict] = []
    for color, slot_id in zip(region_colors, slot_ids):
        shape = _random_region_shape(rng)
        placed = _place_region(blocked, true_center, ANGLE_SLOTS[slot_id], shape, rng)
        if placed is None:
            return None
        _paint(grid, placed, color)
        _mark_blocked(blocked, placed, pad=1)
        regions_meta.append(
            {
                "color": int(color),
                "coords": placed,
                "angle": clockwise_angle(true_center, placed),
            }
        )

    candidates = _candidate_markers(grid, bg)
    if len(candidates) < region_count + 2:
        return None

    wrap_counts = [k for k in range(region_count + 1, min(12, 2 * region_count + 4) + 1) if k % region_count != 0]
    small_counts = [k for k in range(1, region_count) if k % region_count != 0]
    if int(rng.integers(2)) and wrap_counts:
        marker_count = int(rng.choice(np.array(wrap_counts, dtype=int)))
    else:
        marker_count = int(rng.choice(np.array(small_counts, dtype=int)))

    markers = _choose_marker_cells(candidates, marker_count, rng)
    if markers is None:
        return None
    for r, c in markers:
        grid[r, c] = MARKER_COLOR

    ordered_regions = sorted(regions_meta, key=lambda item: (item["angle"], coords_key(item["coords"])))
    expected = _expected_output(grid, bg, markers, ordered_regions, marker_count % region_count)

    det_bg, det_markers, det_divider, det_regions = analyze_scene(grid)
    if det_bg != bg:
        return None
    if det_markers != sorted(markers):
        return None
    if coords_key(det_divider["coords"]) != coords_key(divider):
        return None
    if len(det_regions) != region_count:
        return None
    if [coords_key(comp["coords"]) for comp in det_regions] != [coords_key(item["coords"]) for item in ordered_regions]:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _scene_attempt(RNG)
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid rotating-color-wheel input")
