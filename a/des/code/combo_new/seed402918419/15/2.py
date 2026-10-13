from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 8
COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
MAX_GRID_SIZE = 30
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, points[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_patch(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _aligned_mask(patch: np.ndarray, out_h: int, out_w: int, corner: str) -> np.ndarray:
    mask = patch != BG
    h, w = patch.shape
    placed = np.zeros((out_h, out_w), dtype=bool)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, out_w - w
    elif corner == "bl":
        r0, c0 = out_h - h, 0
    else:
        r0, c0 = out_h - h, out_w - w
    placed[r0 : r0 + h, c0 : c0 + w] = mask
    return placed


def _recover_rectangle_dims(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    out_h = max(int(patch_a.shape[0]), int(patch_b.shape[0]))
    out_w = max(int(patch_a.shape[1]), int(patch_b.shape[1]))
    for corner in ("tl", "tr", "bl", "br"):
        mask_a = _aligned_mask(patch_a, out_h, out_w, corner)
        mask_b = _aligned_mask(patch_b, out_h, out_w, corner)
        if np.any(mask_a & mask_b):
            continue
        if bool(np.all(mask_a | mask_b)):
            candidates.append((out_h, out_w))
    assert candidates, "same-color fragments must reconstruct a solid rectangle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, reconstruct their rectangles, then stack the rectangles by area.

    Input structure:
    - The background is azure.
    - The grid contains an even number of 4-connected monochrome fragments.
    - Each color appears on exactly two fragments, and different pairs use different colors.
    - When the two matching fragments are corner-aligned by their bounding boxes, their union forms a solid rectangle.

    Rule:
    - Group the fragments by color and pair the two fragments of each color.
    - For each pair, align the two cropped fragments at a common bounding-box corner and recover the filled rectangle implied by their complementary union.
    - Compute each recovered rectangle's area.
    - Build the output by stacking the rectangles from smallest area to largest area, top to bottom, left-aligned, with azure filling unused cells.

    Key insight:
    - The scattered fragments are complementary puzzle pieces: color identifies the pairing, while the aligned union reveals the rectangle that must be sorted and repacked.
    """
    g = np.asarray(grid, dtype=int)
    by_color: dict[int, list[np.ndarray]] = defaultdict(list)
    for comp in _find_components(g):
        color = int(comp["color"])
        by_color[color].append(_component_patch(g, comp["coords"], color))

    rects: list[tuple[int, int, int, int]] = []
    for color, patches in by_color.items():
        assert len(patches) == 2, "each foreground color must appear on exactly two fragments"
        rect_h, rect_w = _recover_rectangle_dims(patches[0], patches[1])
        rects.append((rect_h * rect_w, rect_h, rect_w, color))

    rects.sort(key=lambda item: (item[0], item[1], item[2]))
    out_h = sum(h for _area, h, _w, _color in rects)
    out_w = max(w for _area, _h, w, _color in rects)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for _area, h, w, color in rects:
        out[row : row + h, :w] = color
        row += h
    return out


def _transform_corner(mask: np.ndarray, corner: str) -> np.ndarray:
    out = mask
    if corner in ("tr", "br"):
        out = np.fliplr(out)
    if corner in ("bl", "br"):
        out = np.flipud(out)
    return out


def _build_row_staircase(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        positive_rows = random.randint(2, h - 1)
        lengths = [random.randint(2, w - 1)]
        for _row in range(1, positive_rows):
            lengths.append(random.randint(1, lengths[-1]))
        lengths.extend([0] * (h - positive_rows))
        area = sum(lengths)
        if not (h * w * 0.25 <= area <= h * w * 0.75):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, length in enumerate(lengths):
            if length > 0:
                mask[r, :length] = True
        comp = ~mask
        if _connected(mask) and _connected(comp) and mask[0].any() and mask[:, 0].any() and comp[0].any() and comp[:, 0].any():
            return mask
    return None


def _build_col_staircase(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        positive_cols = random.randint(2, w - 1)
        heights = [random.randint(2, h - 1)]
        for _col in range(1, positive_cols):
            heights.append(random.randint(1, heights[-1]))
        heights.extend([0] * (w - positive_cols))
        area = sum(heights)
        if not (h * w * 0.25 <= area <= h * w * 0.75):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for c, height in enumerate(heights):
            if height > 0:
                mask[:height, c] = True
        comp = ~mask
        if _connected(mask) and _connected(comp) and mask[0].any() and mask[:, 0].any() and comp[0].any() and comp[:, 0].any():
            return mask
    return None


def _sample_full_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, str] | None:
    builder = random.choice((_build_row_staircase, _build_col_staircase))
    mask_a = builder(h, w)
    if mask_a is None:
        return None
    mask_b = ~mask_a
    if random.random() < 0.5:
        mask_a, mask_b = mask_b, mask_a
    corner = random.choice(("tl", "tr", "bl", "br"))
    return _transform_corner(mask_a, corner), _transform_corner(mask_b, corner), corner


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    cropped = _crop_mask(mask)
    patch = np.full(cropped.shape, BG, dtype=int)
    patch[cropped] = color
    return patch


def _sample_rectangles() -> list[tuple[int, int, int]]:
    pair_count = random.randint(2, 4)
    candidates = [(h, w) for h in range(4, 8) for w in range(4, 9)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int, int]] = []
    used_areas: set[int] = set()
    for h, w in candidates:
        area = h * w
        if area in used_areas:
            continue
        if sum(item[0] for item in chosen) + h > MAX_GRID_SIZE:
            continue
        chosen.append((h, w, area))
        used_areas.add(area)
        if len(chosen) == pair_count:
            return chosen
    raise RuntimeError("failed to sample unique rectangle dimensions")


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    for _ in range(120):
        grid_h = random.randint(max(16, max_h + 2), MAX_GRID_SIZE)
        grid_w = random.randint(max(16, max_w + 2), MAX_GRID_SIZE)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        bbox_occ = np.zeros((grid_h, grid_w), dtype=bool)
        placements: list[tuple[np.ndarray, int, int]] = []
        ok = True
        for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            ph, pw = patch.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(grid_h - ph + 1):
                for c0 in range(grid_w - pw + 1):
                    rr0 = max(0, r0 - 1)
                    cc0 = max(0, c0 - 1)
                    rr1 = min(grid_h, r0 + ph + 1)
                    cc1 = min(grid_w, c0 + pw + 1)
                    if not bbox_occ[rr0:rr1, cc0:cc1].any():
                        candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            placements.append((patch, r0, c0))
            bbox_occ[r0 : r0 + ph, c0 : c0 + pw] = True
        if not ok:
            continue
        for patch, r0, c0 in placements:
            mask = patch != BG
            grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
        return grid
    return None


def _expected_output(rect_specs: list[tuple[int, int, int]]) -> np.ndarray:
    ordered = sorted(rect_specs, key=lambda item: (item[0] * item[1], item[0], item[1]))
    out_h = sum(h for h, _w, _color in ordered)
    out_w = max(w for _h, w, _color in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    for h, w, color in ordered:
        out[row : row + h, :w] = color
        row += h
    return out


def _gen_attempt() -> np.ndarray | None:
    dims = _sample_rectangles()
    colors = random.sample(COLORS, len(dims))
    patches: list[np.ndarray] = []
    rect_specs: list[tuple[int, int, int]] = []
    for (h, w, _area), color in zip(dims, colors, strict=False):
        sample = _sample_full_masks(h, w)
        if sample is None:
            return None
        mask_a, mask_b, _corner = sample
        patches.append(_render_patch(mask_a, color))
        patches.append(_render_patch(mask_b, color))
        rect_specs.append((h, w, color))

    grid = _place_patches(patches)
    if grid is None:
        return None

    expected = _expected_output(rect_specs)
    if expected.ndim != 2 or max(expected.shape) > MAX_GRID_SIZE:
        return None
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
