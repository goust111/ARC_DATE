from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GRAY = 5
PALETTE = (1, 2, 3, 4)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GENERATION_TRIES = 400


OBJECT_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool),
]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _random_object_mask() -> np.ndarray:
    mask = OBJECT_SHAPES[random.randrange(len(OBJECT_SHAPES))]
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return _trim_mask(mask.copy())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, h, w = rect
    return (r0 + (h - 1) / 2.0, c0 + (w - 1) / 2.0)


def _render_output(crop: np.ndarray, cavities: list[dict], colors: list[int]) -> np.ndarray:
    out = crop.copy()
    for cavity, color in zip(cavities, colors):
        rr, cc = cavity["coords"].T
        out[rr, cc] = int(color)
    return out


def _analyze_grid(grid: np.ndarray) -> dict:
    gray_mask = grid == GRAY
    if not np.any(gray_mask):
        raise AssertionError("missing gray frame")
    r0, c0, r1, c1 = _bbox_from_mask(gray_mask)
    crop = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if not np.all(crop[0] == GRAY) or not np.all(crop[-1] == GRAY):
        raise AssertionError("frame border broken")
    if not np.all(crop[:, 0] == GRAY) or not np.all(crop[:, -1] == GRAY):
        raise AssertionError("frame border broken")

    cavities = []
    for idx, coords in enumerate(_components(crop != GRAY)):
        rr, cc = coords.T
        cavity_bbox = (
            int(rr.min()),
            int(cc.min()),
            int(rr.max()) - int(rr.min()) + 1,
            int(cc.max()) - int(cc.min()) + 1,
        )
        cavities.append(
            {
                "id": idx,
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": cavity_bbox,
                "center": _rect_center(cavity_bbox),
                "center_abs": (r0 + _rect_center(cavity_bbox)[0], c0 + _rect_center(cavity_bbox)[1]),
            }
        )

    outside = np.array(grid, dtype=int)
    outside[r0 : r1 + 1, c0 : c1 + 1] = BG
    colors = [int(v) for v in np.unique(outside) if v not in (BG, GRAY)]
    color_stats = []
    for color in colors:
        comps = _components(outside == color)
        coords = np.argwhere(outside == color)
        color_stats.append(
            {
                "color": int(color),
                "count": len(comps),
                "total_pixels": int(coords.shape[0]),
                "centroid": (float(coords[:, 0].mean()), float(coords[:, 1].mean())),
            }
        )
    return {
        "frame_bbox": (r0, c0, r1, c1),
        "crop": crop,
        "cavities": cavities,
        "color_stats": color_stats,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the frame's rectangular cavities by matching cavity-area rank with outside object-count rank.

    Input structure:
    - The grid has black background, one gray one-pixel-thick rectangular frame, and gray internal separators that split the frame interior into 3 or 4 rectangular cavities.
    - Outside the frame are several small monochrome 4-connected objects using colors from {blue, red, green, yellow}; each used color appears in one or more separate objects.

    Rule:
    - Crop to the gray frame bbox, and detect each interior cavity as a connected non-gray region inside that crop.
    - Count, for every non-background non-gray outside color, how many 4-connected objects of that color exist; sort colors by decreasing object count.
    - Sort the frame cavities by decreasing area.
    - Fill the k-th largest cavity with the color whose outside object count ranks k-th.

    Key insight:
    - The color-to-cavity mapping is not based on position or shape matching: it is a zipper pairing between two independent rankings, outside object counts and inside cavity areas.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    cavities = sorted(info["cavities"], key=lambda item: item["area"], reverse=True)
    color_stats = sorted(info["color_stats"], key=lambda item: item["count"], reverse=True)

    counts = [item["count"] for item in color_stats]
    areas = [item["area"] for item in cavities]
    if len(cavities) not in (3, 4):
        raise AssertionError("expected 3 or 4 cavities")
    if len(color_stats) != len(cavities):
        raise AssertionError("number of colors must match number of cavities")
    if len(set(counts)) != len(counts):
        raise AssertionError("outside object counts must be unique")
    if len(set(areas)) != len(areas):
        raise AssertionError("cavity areas must be unique")

    ranked_colors = [item["color"] for item in color_stats]
    return _render_output(info["crop"], cavities, ranked_colors)


def _build_four_cavity_crop() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    for _ in range(200):
        h1, h2 = random.randint(2, 5), random.randint(2, 5)
        w1, w2 = random.randint(2, 5), random.randint(2, 5)
        areas = [h1 * w1, h1 * w2, h2 * w1, h2 * w2]
        if max(areas) > 20 or min(areas) < 4:
            continue
        if len(set(areas)) != 4:
            continue
        widths = [w1, w2, w1, w2]
        heights = [h1, h1, h2, h2]
        order_area = tuple(np.argsort([-a for a in areas]).tolist())
        order_width = tuple(np.argsort([-x for x in widths]).tolist())
        order_height = tuple(np.argsort([-x for x in heights]).tolist())
        if order_area == order_width or order_area == order_height:
            continue
        crop = np.full((h1 + h2 + 3, w1 + w2 + 3), GRAY, dtype=int)
        rects = [
            (1, 1, h1, w1),
            (1, w1 + 2, h1, w2),
            (h1 + 2, 1, h2, w1),
            (h1 + 2, w1 + 2, h2, w2),
        ]
        for r0, c0, h, w in rects:
            crop[r0 : r0 + h, c0 : c0 + w] = BG
        return crop, rects
    raise RuntimeError("failed to build 4-cavity crop")


def _build_three_cavity_crop() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    variants = ["left_split", "right_split", "top_split", "bottom_split"]
    for _ in range(300):
        variant = random.choice(variants)
        h1, h2 = random.randint(2, 4), random.randint(2, 4)
        w1, w2 = random.randint(2, 4), random.randint(2, 4)

        if variant in ("left_split", "right_split"):
            tall_h = h1 + h2 + 1
            areas = [h1 * w1, h2 * w1, tall_h * w2]
            widths = [w1, w1, w2]
            heights = [h1, h2, tall_h]
            if max(areas) > 20 or min(areas) < 4 or len(set(areas)) != 3:
                continue
            order_area = tuple(np.argsort([-a for a in areas]).tolist())
            if order_area in (
                tuple(np.argsort([-x for x in widths]).tolist()),
                tuple(np.argsort([-x for x in heights]).tolist()),
            ):
                continue
            crop = np.full((h1 + h2 + 3, w1 + w2 + 3), GRAY, dtype=int)
            if variant == "left_split":
                rects = [
                    (1, 1, h1, w1),
                    (h1 + 2, 1, h2, w1),
                    (1, w1 + 2, tall_h, w2),
                ]
            else:
                rects = [
                    (1, 1, tall_h, w1),
                    (1, w1 + 2, h1, w2),
                    (h1 + 2, w1 + 2, h2, w2),
                ]
            for r0, c0, h, w in rects:
                crop[r0 : r0 + h, c0 : c0 + w] = BG
            return crop, rects

        wide_w = w1 + w2 + 1
        areas = [h1 * w1, h1 * w2, h2 * wide_w]
        widths = [w1, w2, wide_w]
        heights = [h1, h1, h2]
        if max(areas) > 20 or min(areas) < 4 or len(set(areas)) != 3:
            continue
        order_area = tuple(np.argsort([-a for a in areas]).tolist())
        if order_area in (
            tuple(np.argsort([-x for x in widths]).tolist()),
            tuple(np.argsort([-x for x in heights]).tolist()),
        ):
            continue
        crop = np.full((h1 + h2 + 3, w1 + w2 + 3), GRAY, dtype=int)
        if variant == "top_split":
            rects = [
                (1, 1, h1, w1),
                (1, w1 + 2, h1, w2),
                (h1 + 2, 1, h2, wide_w),
            ]
        else:
            rects = [
                (1, 1, h1, wide_w),
                (h1 + 2, 1, h2, w1),
                (h1 + 2, w1 + 2, h2, w2),
            ]
        for r0, c0, h, w in rects:
            crop[r0 : r0 + h, c0 : c0 + w] = BG
        return crop, rects
    raise RuntimeError("failed to build 3-cavity crop")


def _build_frame_crop() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    if random.random() < 0.5:
        return _build_three_cavity_crop()
    return _build_four_cavity_crop()


def _sample_color_counts(k: int) -> dict[int, int]:
    colors = random.sample(PALETTE, k)
    counts = random.sample([1, 2, 3, 4, 5], k)
    random.shuffle(counts)
    return {color: count for color, count in zip(colors, counts)}


def _masked_centroid(mask: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(mask)
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _can_place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    for rr, cc in np.argwhere(mask):
        gr = r0 + int(rr)
        gc = c0 + int(cc)
        if grid[gr, gc] != BG:
            return False
        r1 = max(0, gr - 1)
        r2 = min(gh, gr + 2)
        c1 = max(0, gc - 1)
        c2 = min(gw, gc + 2)
        if np.any(grid[r1:r2, c1:c2] != BG):
            return False
    return True


def _place_objects(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    count_by_color: dict[int, int],
) -> dict[int, list[np.ndarray]] | None:
    r0, c0, r1, c1 = frame_bbox
    placed_masks: dict[int, list[np.ndarray]] = {color: [] for color in count_by_color}
    ranked_colors = sorted(count_by_color, key=lambda color: count_by_color[color], reverse=True)
    size_bias = {color: idx for idx, color in enumerate(ranked_colors)}

    colors_expanded = []
    for color, count in count_by_color.items():
        colors_expanded.extend([color] * count)
    random.shuffle(colors_expanded)
    colors_expanded.sort(key=lambda color: (-size_bias[color], random.random()))

    for color in colors_expanded:
        placed = False
        for _ in range(60):
            mask = _random_object_mask()
            area = int(mask.sum())
            if size_bias[color] == 0 and area > 3:
                continue
            if size_bias[color] >= len(ranked_colors) - 1 and area < 4:
                continue
            h, w = mask.shape
            candidates = [
                (rr, cc)
                for rr in range(grid.shape[0] - h + 1)
                for cc in range(grid.shape[1] - w + 1)
                if not (
                    rr + h > r0 - 1
                    and rr < r1 + 2
                    and cc + w > c0 - 1
                    and cc < c1 + 2
                )
            ]
            random.shuffle(candidates)
            for rr, cc in candidates:
                if _can_place_mask(grid, mask, rr, cc):
                    grid[rr : rr + h, cc : cc + w][mask] = int(color)
                    abs_mask = np.zeros_like(grid, dtype=bool)
                    abs_mask[rr : rr + h, cc : cc + w] = mask
                    placed_masks[color].append(abs_mask)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    return placed_masks


def _alternative_output_checks(info: dict, out: np.ndarray) -> bool:
    cavities = sorted(info["cavities"], key=lambda item: item["area"], reverse=True)
    color_stats = sorted(info["color_stats"], key=lambda item: item["count"], reverse=True)
    correct_colors = [item["color"] for item in color_stats]

    alt_color_orders = []
    area_order = [item["color"] for item in sorted(info["color_stats"], key=lambda item: item["total_pixels"], reverse=True)]
    if area_order != correct_colors:
        alt_color_orders.append(area_order)

    cavity_width_order = sorted(info["cavities"], key=lambda item: item["bbox"][3], reverse=True)
    cavity_height_order = sorted(info["cavities"], key=lambda item: item["bbox"][2], reverse=True)
    alt_cavity_orders = []
    if [item["id"] for item in cavity_width_order] != [item["id"] for item in cavities]:
        alt_cavity_orders.append(cavity_width_order)
    if [item["id"] for item in cavity_height_order] != [item["id"] for item in cavities]:
        alt_cavity_orders.append(cavity_height_order)

    if not alt_color_orders or not alt_cavity_orders:
        return False

    for color_order in alt_color_orders:
        if np.array_equal(_render_output(info["crop"], cavities, color_order), out):
            return False
    for cavity_order in alt_cavity_orders:
        if np.array_equal(_render_output(info["crop"], cavity_order, correct_colors), out):
            return False
    return True


def _distance_mismatch_exists(info: dict) -> bool:
    cavities = sorted(info["cavities"], key=lambda item: item["area"], reverse=True)
    color_stats = sorted(info["color_stats"], key=lambda item: item["count"], reverse=True)
    mapping = {item["color"]: cavities[idx]["id"] for idx, item in enumerate(color_stats)}
    for item in color_stats:
        dists = [
            abs(item["centroid"][0] - cavity["center_abs"][0]) + abs(item["centroid"][1] - cavity["center_abs"][1])
            for cavity in info["cavities"]
        ]
        nearest = info["cavities"][int(np.argmin(dists))]["id"]
        if nearest != mapping[item["color"]]:
            return True
    return False


def _validate_candidate(grid: np.ndarray) -> bool:
    info = _analyze_grid(grid)
    cavities = info["cavities"]
    color_stats = info["color_stats"]
    if len(cavities) not in (3, 4) or len(color_stats) != len(cavities):
        return False

    cavity_areas = [item["area"] for item in cavities]
    counts = [item["count"] for item in color_stats]
    if len(set(cavity_areas)) != len(cavity_areas):
        return False
    if len(set(counts)) != len(counts):
        return False
    if any(color not in PALETTE for color in [item["color"] for item in color_stats]):
        return False

    ranked_counts = sorted(color_stats, key=lambda item: item["count"], reverse=True)
    totals = [item["total_pixels"] for item in ranked_counts]
    if all(totals[i] > totals[i + 1] for i in range(len(totals) - 1)):
        return False

    out = main(grid)
    if not _alternative_output_checks(info, out):
        return False
    if not _distance_mismatch_exists(info):
        return False

    frame_r0, frame_c0, frame_r1, frame_c1 = info["frame_bbox"]
    expected_shape = (frame_r1 - frame_r0 + 1, frame_c1 - frame_c0 + 1)
    if out.shape != expected_shape:
        return False
    if np.any(out == BG):
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    try:
        crop, rects = _build_frame_crop()
    except RuntimeError:
        return None
    fh, fw = crop.shape
    min_h = max(16, fh + 6)
    min_w = max(16, fw + 6)
    max_h = min(24, fh + 12)
    max_w = min(24, fw + 12)
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = random.randint(min_h, max_h)
    grid_w = random.randint(min_w, max_w)
    top = random.randint(2, grid_h - fh - 2)
    left = random.randint(2, grid_w - fw - 2)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[top : top + fh, left : left + fw] = crop

    count_by_color = _sample_color_counts(len(rects))
    placed = _place_objects(grid, (top, left, top + fh - 1, left + fw - 1), count_by_color)
    if placed is None:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        candidate = _generate_candidate()
        if candidate is not None and _validate_candidate(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
