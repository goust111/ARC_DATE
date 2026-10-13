from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_SIZE_RANGE = (15, 22)
CLUSTER_COUNT_CHOICES = (2, 3)
PADDING_CHOICES = ((1, 1), (1, 2), (2, 1), (2, 2))
MAX_GENERATION_ATTEMPTS = 500
_VALID_MASKS: dict[tuple[int, int], list[np.ndarray]] = {}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    row_idx = np.arange(shape[0]) % h
    col_idx = np.arange(shape[1]) % w
    return tile[row_idx[:, None], col_idx[None, :]]


def _find_monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def _component_strictly_inside(
    comp: dict,
    rect: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = rect
    return all(r0 < r < r1 and c0 < c < c1 for r, c in comp["coords"])


def _find_clusters(grid: np.ndarray, bg: int = BG) -> list[dict]:
    components = _find_monochrome_components(grid, bg=bg)
    markers_by_color: dict[int, list[tuple[int, int]]] = {}
    centers = [comp for comp in components if comp["area"] >= 2]
    for comp in components:
        if comp["area"] == 1:
            markers_by_color.setdefault(comp["color"], []).append(comp["coords"][0])

    clusters: list[dict] = []
    seen: set[tuple[int, int, int, int, tuple[int, int, int, int]]] = set()
    for color, markers in markers_by_color.items():
        marker_set = set(markers)
        rows = sorted({r for r, _ in markers})
        cols = sorted({c for _, c in markers})
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                for j, c0 in enumerate(cols):
                    for c1 in cols[j + 1 :]:
                        corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
                        if not all(corner in marker_set for corner in corners):
                            continue
                        if r1 - r0 < 2 or c1 - c0 < 2:
                            continue
                        rect = (r0, c0, r1, c1)
                        inside = [comp for comp in centers if _component_strictly_inside(comp, rect)]
                        if len(inside) != 1:
                            continue
                        center = inside[0]
                        region = grid[r0 + 1 : r1, c0 + 1 : c1]
                        region_coords = {
                            (r0 + 1 + rr, c0 + 1 + cc)
                            for rr, cc in np.argwhere(region != bg)
                        }
                        if region_coords != set(center["coords"]):
                            continue
                        key = (*rect, center["bbox"])
                        if key in seen:
                            continue
                        seen.add(key)
                        clusters.append(
                            {
                                "marker_color": color,
                                "marker_rect": rect,
                                "center": center,
                            }
                        )
    return clusters


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the self-colored enclosed object and tile that crop across the full canvas.

    Input structure:
    - The grid has a black background and 2-3 separated clusters.
    - Each cluster is defined by four same-color single-pixel corner markers that
      form a rectangle, with one 4-connected monochromatic object strictly inside.
    - Exactly one cluster is self-colored: its corner-marker color matches its
      center object's color, while every other cluster uses a different center color.

    Rule:
    - Find every valid corner-marker rectangle and its unique enclosed center object.
    - Select the unique self-colored cluster whose marker color equals the center
      object's color.
    - Crop the selected center object to its bounding box, keeping any black cells
      that lie inside that bounding box.
    - Repeat that cropped tile from the output's top-left corner with its own
      height and width as the period, clipping partial tiles at the right/bottom edges.

    Key insight:
    - The target is chosen by the marker/object color equality, and the chosen
      object's cropped bounding box is not the final answer by itself: it becomes
      the periodic tile that generates the whole output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    clusters = _find_clusters(g, bg=BG)
    assert clusters, "expected at least one valid marker cluster"
    self_clusters = [cluster for cluster in clusters if cluster["marker_color"] == cluster["center"]["color"]]
    assert len(self_clusters) == 1, "expected exactly one self-colored cluster"

    target = self_clusters[0]["center"]
    tile = _crop_bbox(g, target["bbox"])
    assert tile.shape[0] >= 2 and tile.shape[1] >= 2
    return _tile_fill(g.shape, tile)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start_r, start_c = map(int, coords[0])
    seen = {(start_r, start_c)}
    stack = [(start_r, start_c)]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _valid_object_masks(height: int, width: int) -> list[np.ndarray]:
    key = (height, width)
    if key in _VALID_MASKS:
        return _VALID_MASKS[key]

    masks: list[np.ndarray] = []
    total = height * width
    max_area = min(8, total - 1)
    for bits in range(1 << total):
        area = bits.bit_count()
        if area < 3 or area > max_area:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for idx in range(total):
            if bits & (1 << idx):
                r = idx // width
                c = idx % width
                mask[r, c] = True
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, height - 1, width - 1):
            continue
        if not _is_connected(mask):
            continue
        masks.append(mask)

    _VALID_MASKS[key] = masks
    return masks


def _sample_object(
    dims: tuple[int, int] | None = None,
    area_predicate=None,
) -> dict | None:
    dim_options = [(h, w) for h in range(2, 5) for w in range(2, 5)]
    random.shuffle(dim_options)
    if dims is not None:
        dim_options = [dims]

    options: list[dict] = []
    for height, width in dim_options:
        for mask in _valid_object_masks(height, width):
            area = int(mask.sum())
            if area_predicate is not None and not area_predicate(area):
                continue
            options.append(
                {
                    "height": height,
                    "width": width,
                    "mask": mask.copy(),
                    "area": area,
                }
            )
    if not options:
        return None
    return random.choice(options)


def _area_mode_ok(target_area: int, distractor_areas: list[int], mode: str) -> bool:
    if len(distractor_areas) == 1:
        if mode == "larger":
            return target_area > distractor_areas[0]
        return target_area < distractor_areas[0]
    if mode == "min":
        return target_area < min(distractor_areas)
    if mode == "max":
        return target_area > max(distractor_areas)
    return min(distractor_areas) < target_area < max(distractor_areas)


def _sample_cluster_specs(n_clusters: int, target_idx: int) -> list[dict] | None:
    target_object = _sample_object()
    if target_object is None:
        return None

    if n_clusters == 2:
        area_mode = random.choice(("larger", "smaller"))
    else:
        area_mode = random.choice(("min", "middle", "max"))

    shared_dims_idx = None
    distractor_indices = [idx for idx in range(n_clusters) if idx != target_idx]
    if distractor_indices and random.random() < 0.55:
        shared_dims_idx = random.choice(distractor_indices)

    specs: list[dict] = [{} for _ in range(n_clusters)]
    specs[target_idx] = target_object

    for _ in range(200):
        distractor_specs: dict[int, dict] = {}
        distractor_areas: list[int] = []
        for idx in distractor_indices:
            dims = None
            if idx == shared_dims_idx:
                dims = (target_object["height"], target_object["width"])
            obj = _sample_object(dims=dims)
            if obj is None:
                break
            distractor_specs[idx] = obj
            distractor_areas.append(obj["area"])
        else:
            if _area_mode_ok(target_object["area"], distractor_areas, area_mode):
                for idx, obj in distractor_specs.items():
                    specs[idx] = obj
                return specs
    return None


def _render_cluster_patch(
    marker_color: int,
    center_color: int,
    obj: dict,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    top_pad, bottom_pad = random.choice(PADDING_CHOICES)
    left_pad, right_pad = random.choice(PADDING_CHOICES)
    patch = np.zeros(
        (top_pad + obj["height"] + bottom_pad, left_pad + obj["width"] + right_pad),
        dtype=int,
    )
    patch[0, 0] = marker_color
    patch[0, -1] = marker_color
    patch[-1, 0] = marker_color
    patch[-1, -1] = marker_color

    r0 = top_pad
    c0 = left_pad
    window = patch[r0 : r0 + obj["height"], c0 : c0 + obj["width"]]
    window[obj["mask"]] = center_color
    return patch, (r0, c0, r0 + obj["height"] - 1, c0 + obj["width"] - 1)


def _rects_separated(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_grid_shape(tile_shape: tuple[int, int], require_clip: bool) -> tuple[int, int]:
    tile_h, tile_w = tile_shape
    for _ in range(200):
        height = random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
        width = random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
        clipped = (height % tile_h != 0) or (width % tile_w != 0)
        if require_clip and not clipped:
            continue
        if not require_clip and clipped and random.random() < 0.65:
            continue
        return height, width
    raise RuntimeError("failed to choose a grid shape")


def _place_patches(grid_shape: tuple[int, int], patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    height, width = grid_shape
    order = list(range(len(patches)))
    random.shuffle(order)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    occupied_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch_h, patch_w = patches[idx].shape
        if patch_h > height or patch_w > width:
            return None
        for _ in range(300):
            r0 = random.randint(0, height - patch_h)
            c0 = random.randint(0, width - patch_w)
            rect = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            if all(_rects_separated(rect, other, gap=1) for other in occupied_rects):
                placements[idx] = (r0, c0)
                occupied_rects.append(rect)
                break
        else:
            return None

    return [(r, c) for r, c in placements if r is not None and c is not None]


def _build_expected_output(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clusters = _find_clusters(grid, bg=BG)
    self_clusters = [cluster for cluster in clusters if cluster["marker_color"] == cluster["center"]["color"]]
    assert len(self_clusters) == 1
    tile = _crop_bbox(grid, self_clusters[0]["center"]["bbox"])
    return tile, _tile_fill(grid.shape, tile)


def _validate_generated(
    grid: np.ndarray,
    expected_cluster_count: int,
    require_clip: bool,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    clusters = _find_clusters(grid, bg=BG)
    if len(clusters) != expected_cluster_count:
        return False

    self_clusters = [cluster for cluster in clusters if cluster["marker_color"] == cluster["center"]["color"]]
    if len(self_clusters) != 1:
        return False

    marker_colors = [cluster["marker_color"] for cluster in clusters]
    target_tile, expected_output = _build_expected_output(grid)
    if target_tile.shape[0] not in range(2, 5) or target_tile.shape[1] not in range(2, 5):
        return False
    if not (target_tile == BG).any():
        return False
    if require_clip and grid.shape[0] % target_tile.shape[0] == 0 and grid.shape[1] % target_tile.shape[1] == 0:
        return False

    for cluster in clusters:
        center = cluster["center"]
        if center["area"] < 3 or center["area"] > 8:
            return False
        r0, c0, r1, c1 = center["bbox"]
        if r1 - r0 + 1 not in range(2, 5) or c1 - c0 + 1 not in range(2, 5):
            return False
        if cluster["marker_color"] != center["color"] and center["color"] in marker_colors:
            return False
        tiled = _tile_fill(grid.shape, _crop_bbox(grid, center["bbox"]))
        if cluster["marker_color"] != center["color"] and np.array_equal(tiled, expected_output):
            return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected_output)


def _generate_attempt() -> np.ndarray | None:
    n_clusters = random.choice(CLUSTER_COUNT_CHOICES)
    target_idx = random.randrange(n_clusters)
    cluster_specs = _sample_cluster_specs(n_clusters, target_idx)
    if cluster_specs is None:
        return None

    marker_colors = random.sample(COLOR_POOL, n_clusters)
    other_colors = [color for color in COLOR_POOL if color not in marker_colors]
    random.shuffle(other_colors)
    center_colors: list[int] = [0] * n_clusters
    for idx in range(n_clusters):
        if idx == target_idx:
            center_colors[idx] = marker_colors[idx]
        else:
            center_colors[idx] = other_colors.pop()

    patches: list[np.ndarray] = []
    for idx in range(n_clusters):
        patch, _ = _render_cluster_patch(marker_colors[idx], center_colors[idx], cluster_specs[idx])
        patches.append(patch)

    target_shape = (cluster_specs[target_idx]["height"], cluster_specs[target_idx]["width"])
    require_clip = random.random() < 0.7
    grid_shape = _choose_grid_shape(target_shape, require_clip=require_clip)
    placements = _place_patches(grid_shape, patches)
    if placements is None:
        return None

    grid = np.zeros(grid_shape, dtype=int)
    for patch, (r0, c0) in zip(patches, placements, strict=False):
        patch_h, patch_w = patch.shape
        grid[r0 : r0 + patch_h, c0 : c0 + patch_w] = patch

    if not _validate_generated(grid, expected_cluster_count=n_clusters, require_clip=require_clip):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
