from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
MARKER_COLORS = (2, 3, 4, 6)
CENTER_COLORS = (1, 5, 7, 8)
MARKER_AREA_VALUES = (2, 3)
MARKER_PIXELS_PER_CLUSTER = 12
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


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


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(
    grid: np.ndarray,
    color: int,
    *,
    connectivity: int,
) -> list[dict]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    components = []
    neighbors = _neighbors4 if connectivity == 4 else _neighbors8

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != int(color) or (r, c) in seen:
                continue

            q = deque([(r, c)])
            seen.add((r, c))
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and (nr, nc) not in seen
                        and int(grid[nr, nc]) == int(color)
                    ):
                        seen.add((nr, nc))
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox_from_coords(coords)
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                    "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                }
            )
    return components


def _find_monochrome_components(grid: np.ndarray, *, bg: int, connectivity: int) -> list[dict]:
    components = []
    for color in sorted({int(v) for v in grid.ravel()} - {int(bg)}):
        components.extend(_find_color_components(grid, color, connectivity=connectivity))
    return components


def _extract_center_component(
    grid: np.ndarray,
    inner_bbox: tuple[int, int, int, int],
    marker_color: int,
    bg: int,
) -> dict | None:
    r0, c0, r1, c1 = inner_bbox
    if r0 > r1 or c0 > c1:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    center_colors = {
        int(v)
        for v in patch.ravel()
        if int(v) not in (int(bg), int(marker_color))
    }
    if len(center_colors) != 1:
        return None

    center_color = next(iter(center_colors))
    comps = _find_color_components(patch, center_color, connectivity=4)
    if len(comps) != 1:
        return None

    comp = comps[0]
    cr0, cc0, cr1, cc1 = comp["bbox"]
    return {
        "color": center_color,
        "area": comp["area"],
        "bbox": (r0 + cr0, c0 + cc0, r0 + cr1, c0 + cc1),
    }


def _cluster_from_marker_combo(
    grid: np.ndarray,
    marker_color: int,
    combo: tuple[dict, dict, dict, dict],
    bg: int,
) -> dict | None:
    mean_r = sum(comp["center"][0] for comp in combo) / 4.0
    mean_c = sum(comp["center"][1] for comp in combo) / 4.0

    by_quad: dict[tuple[int, int], dict] = {}
    for comp in combo:
        quad = (0 if comp["center"][0] < mean_r else 1, 0 if comp["center"][1] < mean_c else 1)
        if quad in by_quad:
            return None
        by_quad[quad] = comp

    if set(by_quad) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return None

    nw = by_quad[(0, 0)]
    ne = by_quad[(0, 1)]
    sw = by_quad[(1, 0)]
    se = by_quad[(1, 1)]
    outer_r0 = min(comp["bbox"][0] for comp in combo)
    outer_c0 = min(comp["bbox"][1] for comp in combo)
    outer_r1 = max(comp["bbox"][2] for comp in combo)
    outer_c1 = max(comp["bbox"][3] for comp in combo)

    if nw["bbox"][0] != outer_r0 or nw["bbox"][1] != outer_c0:
        return None
    if ne["bbox"][0] != outer_r0 or ne["bbox"][3] != outer_c1:
        return None
    if sw["bbox"][2] != outer_r1 or sw["bbox"][1] != outer_c0:
        return None
    if se["bbox"][2] != outer_r1 or se["bbox"][3] != outer_c1:
        return None

    inner_r0 = max(nw["bbox"][2], ne["bbox"][2]) + 1
    inner_r1 = min(sw["bbox"][0], se["bbox"][0]) - 1
    inner_c0 = max(nw["bbox"][3], sw["bbox"][3]) + 1
    inner_c1 = min(ne["bbox"][1], se["bbox"][1]) - 1
    center = _extract_center_component(grid, (inner_r0, inner_c0, inner_r1, inner_c1), marker_color, bg)
    if center is None:
        return None

    crop = grid[center["bbox"][0] : center["bbox"][2] + 1, center["bbox"][1] : center["bbox"][3] + 1].copy()
    if not np.any(crop == center["color"]):
        return None

    return {
        "marker_color": int(marker_color),
        "marker_bbox": (outer_r0, outer_c0, outer_r1, outer_c1),
        "center_bbox": center["bbox"],
        "center_crop": crop,
    }


def _find_clusters(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    comps = _find_monochrome_components(grid, bg=bg, connectivity=8)
    marker_candidates = [comp for comp in comps if comp["area"] in MARKER_AREA_VALUES]
    by_color: dict[int, list[dict]] = {}
    for comp in marker_candidates:
        by_color.setdefault(comp["color"], []).append(comp)

    clusters = []
    for color, color_comps in by_color.items():
        matches = []
        for combo in combinations(color_comps, 4):
            cluster = _cluster_from_marker_combo(grid, color, combo, bg)
            if cluster is not None:
                matches.append(cluster)
        if len(matches) == 1:
            clusters.append(matches[0])

    assert len(clusters) == 4, f"expected 4 clusters, found {len(clusters)}"
    return clusters


def _pad_to_square_upper_left(patch: np.ndarray, side: int, bg: int) -> np.ndarray:
    out = np.full((side, side), bg, dtype=int)
    h, w = patch.shape
    out[:h, :w] = patch
    return out


def _assemble_output(clusters: list[dict], grid: np.ndarray) -> np.ndarray:
    bg = _get_bg(grid)
    decorated = []
    for cluster in clusters:
        freq = int(np.sum(grid == cluster["marker_color"]))
        decorated.append((freq, cluster))

    freqs = [freq for freq, _ in decorated]
    assert len(set(freqs)) == 4, "marker frequencies must be distinct"
    ordered = [cluster for _, cluster in sorted(decorated, key=lambda item: item[0])]

    side = max(max(cluster["center_crop"].shape) for cluster in ordered)
    tiles = [
        np.rot90(_pad_to_square_upper_left(cluster["center_crop"], side, bg), k)
        for k, cluster in enumerate(ordered)
    ]

    out = np.full((2 * side, 2 * side), bg, dtype=int)
    out[:side, :side] = tiles[0]
    out[:side, side:] = tiles[1]
    out[side:, :side] = tiles[2]
    out[side:, side:] = tiles[3]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble the four enclosed center patterns by the global frequency rank of their corner-marker colors.

    Input structure:
    - The background is black, and there are four separated clusters built from four same-color corner markers.
    - Each cluster encloses one monochromatic 4-connected center pattern of a different color.
    - Extra isolated noise pixels reuse the four marker colors elsewhere in the grid, changing each marker color's global frequency.

    Rule:
    - Detect the four corner-marker clusters and crop each enclosed center pattern to its tight bounding box.
    - Count how many times each cluster's marker color appears in the full input, including the outside noise pixels.
    - Rank the clusters from lowest to highest marker-color frequency, pad each crop to the largest square size, then place rank 1/2/3/4 in the output quadrants with rotations 0/90CCW/180/270CCW respectively.

    Key insight:
    - The outside same-color noise is part of the code: it changes the global frequency ranking, and that single ranking determines both which quadrant a pattern uses and how much it is rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    clusters = _find_clusters(g)
    return _assemble_output(clusters, g)


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    for _ in range(120):
        area = random.randint(5, min(9, height * width))
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True

        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            mask[random.choice(tuple(frontier))] = True

        if int(mask.sum()) != area:
            continue
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != height - 1 or cols.min() != 0 or cols.max() != width - 1:
            continue
        return mask

    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    mask[0, :] = True
    return mask


def _make_center_patch(color: int) -> np.ndarray:
    for _ in range(120):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        if height * width < 5:
            continue
        mask = _random_connected_mask(height, width)
        if not (5 <= int(mask.sum()) <= 9):
            continue
        patch = np.zeros((height, width), dtype=int)
        patch[mask] = int(color)
        return patch
    raise RuntimeError("failed to generate center patch")


def _make_cluster_patch(marker_color: int, center_patch: np.ndarray) -> np.ndarray:
    h, w = center_patch.shape
    patch = np.zeros((h + 4, w + 4), dtype=int)
    coords = {
        (0, 0), (0, 1), (1, 0),
        (0, w + 3), (0, w + 2), (1, w + 3),
        (h + 3, 0), (h + 2, 0), (h + 3, 1),
        (h + 3, w + 3), (h + 3, w + 2), (h + 2, w + 3),
    }
    for r, c in coords:
        patch[r, c] = int(marker_color)
    patch[2 : 2 + h, 2 : 2 + w] = center_patch
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = np.where(patch != 0, patch, grid[r0 : r0 + h, c0 : c0 + w])


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, gap: int = 1) -> bool:
    h, w = patch.shape
    gh, gw = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(gh, r0 + h + gap)
    cc1 = min(gw, c0 + w + gap)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_clusters(
    grid_shape: tuple[int, int],
    cluster_patches: list[np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    grid = np.zeros(grid_shape, dtype=int)
    occupied = np.zeros(grid_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(cluster_patches)

    order = list(range(len(cluster_patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: cluster_patches[idx].shape[0] * cluster_patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = cluster_patches[idx]
        ph, pw = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(grid_shape[0] - ph + 1)
            for c0 in range(grid_shape[1] - pw + 1)
            if _can_place_patch(occupied, patch, r0, c0, gap=1)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _place_patch(grid, patch, r0, c0)
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch != 0
        placements[idx] = (r0, c0)

    return grid, [pos for pos in placements if pos is not None]


def _can_place_noise(grid: np.ndarray, occupied: np.ndarray, r: int, c: int) -> bool:
    if occupied[r, c] or int(grid[r, c]) != 0:
        return False
    h, w = grid.shape
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
            return False
    return True


def _add_noise_pixels(
    grid: np.ndarray,
    occupied: np.ndarray,
    color_to_extra: dict[int, int],
) -> bool:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)

    for color, count in color_to_extra.items():
        placed = 0
        for r, c in cells:
            if _can_place_noise(grid, occupied, r, c):
                grid[r, c] = int(color)
                occupied[r, c] = True
                placed += 1
                if placed == count:
                    break
        if placed != count:
            return False
    return True


def _crop_center_patch(patch: np.ndarray) -> np.ndarray:
    rows, cols = np.where(patch != 0)
    return patch[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _oracle_expected(center_patches: dict[int, np.ndarray], color_to_extra: dict[int, int]) -> np.ndarray:
    ranking = sorted(MARKER_COLORS, key=lambda color: color_to_extra[color])
    side = max(max(center_patches[color].shape) for color in MARKER_COLORS)
    tiles = [
        np.rot90(_pad_to_square_upper_left(center_patches[color], side, BG), k)
        for k, color in enumerate(ranking)
    ]
    out = np.zeros((2 * side, 2 * side), dtype=int)
    out[:side, :side] = tiles[0]
    out[:side, side:] = tiles[1]
    out[side:, :side] = tiles[2]
    out[side:, side:] = tiles[3]
    return out


def _generator_attempt() -> np.ndarray | None:
    center_color_order = list(CENTER_COLORS)
    random.shuffle(center_color_order)
    center_patches = {
        marker_color: _make_center_patch(center_color)
        for marker_color, center_color in zip(MARKER_COLORS, center_color_order)
    }
    cluster_patches = [
        _make_cluster_patch(marker_color, center_patches[marker_color])
        for marker_color in MARKER_COLORS
    ]

    grid_shape = (random.randint(20, 26), random.randint(20, 26))
    placed = _place_clusters(grid_shape, cluster_patches)
    if placed is None:
        return None
    grid, _placements = placed
    occupied = grid != 0

    extras = random.sample(range(1, 10), 4)
    shuffled_colors = list(MARKER_COLORS)
    random.shuffle(shuffled_colors)
    color_to_extra = {color: extra for color, extra in zip(shuffled_colors, extras)}

    if not _add_noise_pixels(grid, occupied, color_to_extra):
        return None

    if len({int(np.sum(grid == color)) for color in MARKER_COLORS}) != 4:
        return None

    expected = _oracle_expected(
        {color: _crop_center_patch(center_patches[color]) for color in MARKER_COLORS},
        color_to_extra,
    )

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    clusters = _find_clusters(grid)
    solved_colors = sorted(cluster["marker_color"] for cluster in clusters)
    if solved_colors != sorted(MARKER_COLORS):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generator_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
