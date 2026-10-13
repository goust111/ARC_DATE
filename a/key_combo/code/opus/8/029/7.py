from __future__ import annotations

import math
import random
from collections import deque
from itertools import permutations

import numpy as np


BG = 0
DIVIDER_COLORS = (4, 7)
PATTERN_COLORS = (1, 2, 3, 5, 6)
BOX_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
MIN_ANGLE_GAP = 0.30


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def stretch_patch(patch: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = patch.shape
    row_idx = np.floor(np.arange(out_h) * src_h / out_h).astype(int)
    col_idx = np.floor(np.arange(out_w) * src_w / out_w).astype(int)
    return patch[np.ix_(row_idx, col_idx)]


def _divider_mask(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, DIVIDER_COLORS)


def _region_angle(coords: np.ndarray, center_r: float, center_c: float) -> float:
    cr, cc = coords.mean(axis=0)
    return math.atan2(float(cr - center_r), float(cc - center_c)) % (2 * math.pi)


def _extract_regions_and_patterns(grid: np.ndarray) -> tuple[list[dict], tuple[float, float]]:
    divider = _divider_mask(grid)
    divider_coords = np.argwhere(divider)
    assert divider_coords.size > 0, "expected divider pixels"
    center_r, center_c = divider_coords.mean(axis=0)

    regions: list[dict] = []
    for coords in components_mask(~divider):
        region_mask = np.zeros_like(divider, dtype=bool)
        region_mask[coords[:, 0], coords[:, 1]] = True
        pattern_coords = np.argwhere(region_mask & (grid != BG) & ~divider)
        assert pattern_coords.size > 0, "each region must contain one pattern"
        box = bbox_coords(pattern_coords)
        patch = crop_bbox(grid, box)
        angle = _region_angle(coords, center_r, center_c)
        regions.append(
            {
                "coords": coords,
                "bbox": box,
                "patch": patch,
                "angle": angle,
            }
        )

    regions.sort(key=lambda item: item["angle"])
    assert 3 <= len(regions) <= 4, "expected 3 or 4 surrounding regions"
    gaps = [((regions[(i + 1) % len(regions)]["angle"] - regions[i]["angle"]) % (2 * math.pi)) for i in range(len(regions))]
    assert min(gaps) > 0.0, "angles must define a unique cyclic order"
    return regions, (float(center_r), float(center_c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle each separated region's internal multicolor pattern to the next region and stretch it to the receiver's box.

    Input structure:
    - The grid has black background with a fixed yellow/orange divider shaped like a cross or T.
    - The divider splits the non-divider cells into 3 or 4 disconnected regions.
    - Each region contains exactly one small multicolor pattern away from the divider and outer border.

    Rule:
    - Leave all yellow/orange divider pixels unchanged.
    - Sort the regions by centroid angle around the divider centroid, using 0 degrees at the right and increasing clockwise.
    - In each region, crop the tight bounding box of its internal pattern.
    - Region R_i receives the previous region's cropped pattern, stretched independently in rows and columns to R_i's original pattern-box size with nearest-neighbor sampling.
    - Replace only each original pattern box with the stretched incoming pattern; all other cells stay unchanged.

    Key insight:
    - The spatial cycle is determined by angular order of the host regions, while the differing host box sizes force an additional geometric stretch of the transferred 2D pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    regions, _center = _extract_regions_and_patterns(g)
    out = g.copy()
    for i, region in enumerate(regions):
        src = regions[(i - 1) % len(regions)]["patch"]
        r0, c0, r1, c1 = region["bbox"]
        out_h = r1 - r0 + 1
        out_w = c1 - c0 + 1
        stretched = stretch_patch(src, out_h, out_w)
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        paste(out, stretched, r0, c0)
    return out


def _paint_divider(grid: np.ndarray, kind: str, mid_r: int, mid_c: int) -> int:
    h, w = grid.shape
    if kind == "cross":
        grid[:, mid_c] = 4
        grid[mid_r, :] = 7
        return 4
    if kind == "t_right":
        grid[:, mid_c] = 4
        grid[mid_r, mid_c:] = 7
        return 3
    if kind == "t_left":
        grid[:, mid_c] = 4
        grid[mid_r, : mid_c + 1] = 7
        return 3
    if kind == "t_down":
        grid[mid_r, :] = 4
        grid[mid_r:, mid_c] = 7
        return 3
    if kind == "t_up":
        grid[mid_r, :] = 4
        grid[: mid_r + 1, mid_c] = 7
        return 3
    raise ValueError(kind)


def _candidate_boxes(region_mask: np.ndarray, box_h: int, box_w: int) -> list[tuple[int, int]]:
    h, w = region_mask.shape
    spots: list[tuple[int, int]] = []
    for r0 in range(1, h - box_h):
        for c0 in range(1, w - box_w):
            if not region_mask[r0 : r0 + box_h, c0 : c0 + box_w].all():
                continue
            if not region_mask[r0 - 1 : r0 + box_h + 1, c0 - 1 : c0 + box_w + 1].all():
                continue
            spots.append((r0, c0))
    return spots


def _random_pattern(shape: tuple[int, int], used: set[tuple[int, ...]]) -> np.ndarray:
    h, w = shape
    for _ in range(200):
        n_colors = 2 if h * w <= 4 else random.choice((2, 3))
        palette = random.sample(PATTERN_COLORS, n_colors)
        patch = np.array([[random.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(map(int, patch.flatten()))) < 2:
            continue
        if h > 1 and len({tuple(row.tolist()) for row in patch}) == 1:
            continue
        if w > 1 and len({tuple(col.tolist()) for col in patch.T}) == 1:
            continue
        sig = tuple(map(int, patch.flatten()))
        if sig in used:
            continue
        used.add(sig)
        return patch
    raise RuntimeError("failed to sample a distinct pattern")


def _expected_output_from_metadata(
    inp: np.ndarray,
    region_infos: list[dict],
    order: list[int],
) -> np.ndarray:
    out = inp.copy()
    for pos, dst_index in enumerate(order):
        src_index = order[(pos - 1) % len(order)]
        src_patch = region_infos[src_index]["pattern"]
        r0, c0, r1, c1 = region_infos[dst_index]["box"]
        stretched = stretch_patch(src_patch, r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        paste(out, stretched, r0, c0)
    return out


def _choose_layout() -> tuple[np.ndarray, int]:
    h = random.randint(12, 15)
    w = random.randint(12, 15)
    mid_r = random.randint(5, h - 6)
    mid_c = random.randint(5, w - 6)
    kind = random.choice(["cross", "cross", "t_right", "t_left", "t_down", "t_up"])
    grid = np.zeros((h, w), dtype=int)
    expected_regions = _paint_divider(grid, kind, mid_r, mid_c)
    return grid, expected_regions


def _choose_boxes(region_masks: list[np.ndarray]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    perms = list(permutations(BOX_SHAPES, len(region_masks)))
    random.shuffle(perms)
    for shapes in perms:
        candidate_lists: list[list[tuple[int, int]]] = []
        ok = True
        for region_mask, shape in zip(region_masks, shapes):
            spots = _candidate_boxes(region_mask, shape[0], shape[1])
            if not spots:
                ok = False
                break
            candidate_lists.append(spots)
        if ok:
            picks = [random.choice(spots) for spots in candidate_lists]
            return list(zip(shapes, picks))
    raise RuntimeError("failed to fit distinct boxes into the regions")


def _semantic_checks(inp: np.ndarray, region_infos: list[dict], order: list[int], expected: np.ndarray) -> bool:
    sizes = [tuple(info["pattern"].shape) for info in region_infos]
    if len(set(sizes)) != len(sizes):
        return False

    in_patches = [tuple(map(int, info["pattern"].flatten())) for info in region_infos]
    if len(set(in_patches)) != len(in_patches):
        return False

    for pos, dst_index in enumerate(order):
        src_index = order[(pos - 1) % len(order)]
        dst_patch = region_infos[dst_index]["pattern"]
        src_patch = region_infos[src_index]["pattern"]
        stretched = stretch_patch(src_patch, *dst_patch.shape)
        if np.array_equal(stretched, dst_patch):
            return False

    if np.array_equal(inp, expected):
        return False

    divider = _divider_mask(inp)
    if not np.array_equal(inp[divider], expected[divider]):
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    solved_regions, center = _extract_regions_and_patterns(inp)
    angles = [region["angle"] for region in solved_regions]
    gaps = [((angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi)) for i in range(len(angles))]
    if min(gaps) < MIN_ANGLE_GAP:
        return False
    _ = center
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        grid, expected_regions = _choose_layout()
        divider = _divider_mask(grid)
        region_coords = components_mask(~divider)
        if len(region_coords) != expected_regions:
            return None

        divider_center = np.argwhere(divider).mean(axis=0)
        ordered_pairs = []
        region_masks: list[np.ndarray] = []
        for idx, coords in enumerate(region_coords):
            mask = np.zeros_like(divider, dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            region_masks.append(mask)
            angle = _region_angle(coords, float(divider_center[0]), float(divider_center[1]))
            ordered_pairs.append((angle, idx))
        ordered_pairs.sort()
        order = [idx for _angle, idx in ordered_pairs]
        gaps = [((ordered_pairs[(i + 1) % len(order)][0] - ordered_pairs[i][0]) % (2 * math.pi)) for i in range(len(order))]
        if min(gaps) < MIN_ANGLE_GAP:
            return None

        assignments = _choose_boxes(region_masks)
        used_patterns: set[tuple[int, ...]] = set()
        region_infos: list[dict] = []
        for region_index, ((box_h, box_w), (r0, c0)) in enumerate(assignments):
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            pattern = _random_pattern((box_h, box_w), used_patterns)
            paste(grid, pattern, r0, c0)
            region_infos.append(
                {
                    "region_index": region_index,
                    "box": box,
                    "pattern": pattern,
                }
            )

        expected = _expected_output_from_metadata(grid, region_infos, order)
        if not _semantic_checks(grid, region_infos, order, expected):
            return None
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 7")
