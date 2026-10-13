from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RAY_COLORS = (2, 3)
OBJECT_COLORS = (1, 4, 5, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GENERATION_PLANS = [
    {"object_sizes": [4, 4, 5], "winner_count": 2},
    {"object_sizes": [4, 4, 6, 3], "winner_count": 2},
    {"object_sizes": [4, 4, 5, 6, 3], "winner_count": 2},
    {"object_sizes": [3, 4, 4, 6, 5], "winner_count": 3},
]


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    labels, components = _label_components(mask)
    del labels
    return components


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []
    next_label = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = next_label
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))

            components.append(np.array(coords, dtype=int))
            next_label += 1

    return labels, components


def _cross_mask(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    h, w = shape
    cr, cc = center
    mask = np.zeros((h, w), dtype=bool)
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        r, c = cr, cc
        while 0 <= r < h and 0 <= c < w:
            mask[r, c] = True
            r += dr
            c += dc
    return mask


def _mode(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _detect_ray_center(grid: np.ndarray, color: int) -> tuple[tuple[int, int], np.ndarray]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing ray color {color}"

    diag_diff = _mode(coords[:, 0] - coords[:, 1])
    diag_sum = _mode(coords[:, 0] + coords[:, 1])
    assert (diag_diff + diag_sum) % 2 == 0
    assert (diag_sum - diag_diff) % 2 == 0

    center = ((diag_diff + diag_sum) // 2, (diag_sum - diag_diff) // 2)
    h, w = grid.shape
    cr, cc = center
    assert 0 <= cr < h and 0 <= cc < w
    assert int(grid[cr, cc]) == int(color)

    cross = _cross_mask((h, w), center)
    assert np.all(cross[coords[:, 0], coords[:, 1]])
    return (int(cr), int(cc)), cross


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    centers: dict[int, tuple[int, int]] = {}
    per_color_masks: dict[int, np.ndarray] = {}
    boundary_mask = np.zeros((h, w), dtype=bool)

    for color in RAY_COLORS:
        center, cross = _detect_ray_center(g, color)
        centers[color] = center
        per_color_masks[color] = cross
        boundary_mask |= cross

    center_values = [centers[color] for color in RAY_COLORS]
    assert center_values[0][0] != center_values[1][0]
    assert center_values[0][1] != center_values[1][1]
    assert center_values[0][0] - center_values[0][1] != center_values[1][0] - center_values[1][1]
    assert center_values[0][0] + center_values[0][1] != center_values[1][0] + center_values[1][1]
    assert np.all(np.isin(g[boundary_mask], RAY_COLORS))

    region_labels, region_components = _label_components(~boundary_mask)
    assert region_components, "expected at least one non-boundary region"

    object_mask = (g != BG) & (~np.isin(g, RAY_COLORS))
    object_components = _connected_components(object_mask)
    assert object_components, "expected foreground objects"

    area_by_region: dict[int, int] = {}
    object_info: list[dict] = []
    for coords in object_components:
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        anchor = coords[int(order[0])]
        label = int(region_labels[int(anchor[0]), int(anchor[1])])
        assert label >= 0
        assert np.all(region_labels[coords[:, 0], coords[:, 1]] == label)
        area_by_region[label] = area_by_region.get(label, 0) + int(coords.shape[0])
        object_info.append(
            {
                "coords": coords,
                "region": label,
                "size": int(coords.shape[0]),
                "anchor": (int(anchor[0]), int(anchor[1])),
            }
        )

    best_area = max(area_by_region.values())
    winner_regions = [label for label, area in area_by_region.items() if area == best_area]
    assert len(winner_regions) == 1, "largest object area sum must be unique"
    winner_region = int(winner_regions[0])

    winner_coords = region_components[winner_region]
    bbox = _bbox_from_coords(winner_coords)
    centroid = (
        _round_half_up(float(winner_coords[:, 0].mean())),
        _round_half_up(float(winner_coords[:, 1].mean())),
    )

    dist2 = {}
    for color, (cr, cc) in centers.items():
        dist2[color] = (centroid[0] - cr) ** 2 + (centroid[1] - cc) ** 2
    best_dist = min(dist2.values())
    nearest_colors = [color for color, value in dist2.items() if value == best_dist]
    assert len(nearest_colors) == 1, "region centroid must have a unique nearest ray center"
    repaint_color = int(nearest_colors[0])

    crop = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
    output = np.zeros_like(crop)
    crop_object_mask = (crop != BG) & (~np.isin(crop, RAY_COLORS))
    output[crop_object_mask] = repaint_color

    return {
        "grid": g,
        "centers": centers,
        "boundary_mask": boundary_mask,
        "region_labels": region_labels,
        "region_components": region_components,
        "object_info": object_info,
        "area_by_region": area_by_region,
        "winner_region": winner_region,
        "winner_bbox": bbox,
        "winner_centroid": centroid,
        "repaint_color": repaint_color,
        "output": output,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the diagonal partition whose foreground area is uniquely largest, then recolor its contents by the nearer ray center.

    Input structure:
    - The grid has black background, two interior colored ray centers, and each center emits a full X of diagonal rays to the border.
    - Additional small 4-connected monochrome objects use colors different from the two ray colors and lie entirely inside single ray-bounded regions.

    Rule:
    - Treat every diagonal-ray pixel as a boundary and split the remaining cells into 4-connected regions.
    - Assign each foreground object to the region containing its top-left pixel, sum object areas by region, and choose the unique largest total.
    - Take the axis-aligned bounding box of that winning region, crop it, repaint every non-ray foreground pixel inside the crop with the color of the ray center nearest to the winning region centroid, and leave background/rays black.

    Key insight:
    - The diagonal X marks are not output decorations; together they define a non-rectangular partition whose winning region determines both the crop window and the recolor source.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _count_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(np.count_nonzero(mask_a & mask_b))


def _neighbor_cells(r: int, c: int, h: int, w: int, directions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr, dc in directions:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.append((nr, nc))
    return cells


def _cell_usable(
    r: int,
    c: int,
    region_labels: np.ndarray,
    target_region: int,
    boundary_mask: np.ndarray,
    occupied: np.ndarray,
    forbidden_mask: np.ndarray | None,
) -> bool:
    h, w = region_labels.shape
    if int(region_labels[r, c]) != int(target_region):
        return False
    if occupied[r, c]:
        return False
    if forbidden_mask is not None and forbidden_mask[r, c]:
        return False
    if boundary_mask[r, c]:
        return False

    for rr, cc in [(r, c)] + _neighbor_cells(r, c, h, w, DIR8):
        if boundary_mask[rr, cc]:
            return False
        if occupied[rr, cc]:
            return False
        if forbidden_mask is not None and forbidden_mask[rr, cc]:
            return False
    return True


def _grow_object(
    region_labels: np.ndarray,
    target_region: int,
    boundary_mask: np.ndarray,
    occupied: np.ndarray,
    size: int,
    forbidden_mask: np.ndarray | None = None,
    attempts: int = 140,
) -> np.ndarray | None:
    h, w = region_labels.shape
    usable = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if _cell_usable(r, c, region_labels, target_region, boundary_mask, occupied, forbidden_mask)
    ]
    if len(usable) < size:
        return None

    for _ in range(attempts):
        seed = random.choice(usable)
        shape = {seed}

        while len(shape) < size:
            frontier: set[tuple[int, int]] = set()
            for r, c in shape:
                for nr, nc in _neighbor_cells(r, c, h, w, DIR4):
                    if (nr, nc) in shape:
                        continue
                    if _cell_usable(
                        nr,
                        nc,
                        region_labels,
                        target_region,
                        boundary_mask,
                        occupied,
                        forbidden_mask,
                    ):
                        frontier.add((nr, nc))

            if not frontier:
                break
            shape.add(random.choice(tuple(frontier)))

        if len(shape) == size:
            coords = np.array(sorted(shape), dtype=int)
            if len(_connected_components(_coords_to_mask(coords, (h, w)))) == 1:
                return coords

    return None


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _make_forbidden_box(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _region_centroid(coords: np.ndarray) -> tuple[int, int]:
    return (
        _round_half_up(float(coords[:, 0].mean())),
        _round_half_up(float(coords[:, 1].mean())),
    )


def _nearest_center_color(centroid: tuple[int, int], centers: dict[int, tuple[int, int]]) -> int | None:
    dists = {
        color: (centroid[0] - center[0]) ** 2 + (centroid[1] - center[1]) ** 2
        for color, center in centers.items()
    }
    best = min(dists.values())
    winners = [color for color, dist in dists.items() if dist == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _render_expected(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    repaint_color: int,
) -> np.ndarray:
    crop = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
    out = np.zeros_like(crop)
    object_mask = (crop != BG) & (~np.isin(crop, RAY_COLORS))
    out[object_mask] = int(repaint_color)
    return out


def _choose_layout() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(300):
        h = random.randint(16, 22)
        w = random.randint(16, 22)
        center_a = (random.randint(3, h - 4), random.randint(3, w - 4))
        center_b = (random.randint(3, h - 4), random.randint(3, w - 4))

        if center_a == center_b:
            continue
        if center_a[0] == center_b[0] or center_a[1] == center_b[1]:
            continue
        if center_a[0] - center_a[1] == center_b[0] - center_b[1]:
            continue
        if center_a[0] + center_a[1] == center_b[0] + center_b[1]:
            continue

        mask_a = _cross_mask((h, w), center_a)
        mask_b = _cross_mask((h, w), center_b)
        if _count_overlap(mask_a, mask_b) != 0:
            continue

        boundary_mask = mask_a | mask_b
        region_labels, region_components = _label_components(~boundary_mask)
        if len(region_components) < 5:
            continue

        centers = {RAY_COLORS[0]: center_a, RAY_COLORS[1]: center_b}
        region_meta: dict[int, dict] = {}
        for label, coords in enumerate(region_components):
            bbox = _bbox_from_coords(coords)
            centroid = _region_centroid(coords)
            repaint_color = _nearest_center_color(centroid, centers)
            if repaint_color is None:
                continue
            box_h = bbox[2] - bbox[0] + 1
            box_w = bbox[3] - bbox[1] + 1
            if box_h < 5 or box_w < 5:
                continue
            if box_h * box_w >= h * w - 8:
                continue
            region_meta[label] = {
                "coords": coords,
                "bbox": bbox,
                "centroid": centroid,
                "repaint_color": repaint_color,
            }

        if len(region_meta) < 4:
            continue

        return centers, mask_a, mask_b, region_labels

    return None


def _gen_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None

    centers, mask_a, mask_b, region_labels = layout
    h, w = region_labels.shape
    boundary_mask = mask_a | mask_b
    region_components = _label_components(~boundary_mask)[1]

    region_meta: dict[int, dict] = {}
    for label, coords in enumerate(region_components):
        bbox = _bbox_from_coords(coords)
        centroid = _region_centroid(coords)
        repaint_color = _nearest_center_color(centroid, centers)
        if repaint_color is None:
            continue
        region_meta[label] = {
            "coords": coords,
            "bbox": bbox,
            "centroid": centroid,
            "repaint_color": repaint_color,
        }

    plan = random.choice(GENERATION_PLANS)
    winner_count = int(plan["winner_count"])
    object_sizes = list(plan["object_sizes"])
    loser_count = len(object_sizes) - winner_count

    desired_color = random.choice(RAY_COLORS)
    winner_candidates = [
        label
        for label, meta in region_meta.items()
        if int(meta["repaint_color"]) == int(desired_color)
    ]
    random.shuffle(winner_candidates)
    if not winner_candidates:
        return None

    for winner_region in winner_candidates:
        winner_bbox = region_meta[winner_region]["bbox"]
        winner_forbidden = None
        loser_forbidden = _make_forbidden_box((h, w), winner_bbox)
        candidate_losers = [label for label in region_meta if label != winner_region]
        random.shuffle(candidate_losers)
        if len(candidate_losers) < loser_count:
            continue

        chosen_losers = candidate_losers[:loser_count]
        occupied = np.zeros((h, w), dtype=bool)
        objects: list[dict] = []

        sizes = object_sizes[:]
        winner_sizes = sizes[:winner_count]
        loser_sizes = sizes[winner_count:]

        failed = False
        for size in winner_sizes:
            coords = _grow_object(
                region_labels,
                winner_region,
                boundary_mask,
                occupied,
                int(size),
                forbidden_mask=winner_forbidden,
            )
            if coords is None:
                failed = True
                break
            occupied[coords[:, 0], coords[:, 1]] = True
            objects.append({"region": winner_region, "size": int(size), "coords": coords})
        if failed:
            continue

        for loser_region, size in zip(chosen_losers, loser_sizes):
            coords = _grow_object(
                region_labels,
                loser_region,
                boundary_mask,
                occupied,
                int(size),
                forbidden_mask=loser_forbidden,
            )
            if coords is None:
                failed = True
                break
            occupied[coords[:, 0], coords[:, 1]] = True
            objects.append({"region": loser_region, "size": int(size), "coords": coords})
        if failed:
            continue

        winner_object_sizes = [obj["size"] for obj in objects if int(obj["region"]) == int(winner_region)]
        loser_object_sizes = [obj["size"] for obj in objects if int(obj["region"]) != int(winner_region)]
        assert winner_object_sizes
        assert loser_object_sizes
        if max(winner_object_sizes) >= max(loser_object_sizes):
            continue

        area_by_region: dict[int, int] = {}
        for obj in objects:
            area_by_region[obj["region"]] = area_by_region.get(obj["region"], 0) + int(obj["size"])
        best_total = max(area_by_region.values())
        winners = [label for label, total in area_by_region.items() if total == best_total]
        if winners != [winner_region]:
            continue

        colors = random.sample(OBJECT_COLORS, len(objects))
        grid = np.zeros((h, w), dtype=int)
        grid[mask_a] = RAY_COLORS[0]
        grid[mask_b] = RAY_COLORS[1]

        for color, obj in zip(colors, objects):
            coords = obj["coords"]
            grid[coords[:, 0], coords[:, 1]] = int(color)

        any_loser_inside = False
        r0, c0, r1, c1 = winner_bbox
        for obj in objects:
            if int(obj["region"]) == int(winner_region):
                continue
            coords = obj["coords"]
            inside = (
                (coords[:, 0] >= r0)
                & (coords[:, 0] <= r1)
                & (coords[:, 1] >= c0)
                & (coords[:, 1] <= c1)
            )
            if bool(np.any(inside)):
                any_loser_inside = True
                break
        if any_loser_inside:
            continue

        expected = _render_expected(
            grid,
            winner_bbox,
            int(region_meta[winner_region]["repaint_color"]),
        )
        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue

        analysis = _analyze(grid)
        if int(analysis["winner_region"]) != int(winner_region):
            continue
        if int(analysis["repaint_color"]) != int(region_meta[winner_region]["repaint_color"]):
            continue
        if analysis["winner_bbox"] != winner_bbox:
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-partition crop instance")
