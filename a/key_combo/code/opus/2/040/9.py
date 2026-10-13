from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 5
RECT_COLOR = 9
SHAPE_COLOR = 4
RAY_COLORS = (1, 2, 3)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _bbox_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _in_bbox(cell: Coord, bbox: BBox) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = (grid == int(color))
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append({"color": int(color), "coords": arr, "bbox": _bbox_from_coords(arr), "area": len(coords)})
    return out


def _solid_rect_component(grid: np.ndarray, comp: dict) -> bool:
    bbox = comp["bbox"]
    color = int(comp["color"])
    if comp["area"] != _bbox_area(bbox):
        return False
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == color))


def _find_rectangle(grid: np.ndarray) -> BBox:
    rects = []
    for comp in _components_of_color(grid, RECT_COLOR):
        bbox = comp["bbox"]
        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        if h >= 4 and w >= 4 and _solid_rect_component(grid, comp):
            rects.append(comp)
    assert rects, "expected a solid maroon rectangle"
    rects.sort(key=lambda comp: (-comp["area"], comp["bbox"]))
    return rects[0]["bbox"]


def _find_shapes(grid: np.ndarray) -> list[dict]:
    shapes = []
    for comp in _components_of_color(grid, SHAPE_COLOR):
        bbox = comp["bbox"]
        if bbox[2] - bbox[0] == 2 and bbox[3] - bbox[1] == 2:
            shapes.append(comp)
    assert shapes, "expected at least one yellow 3x3-bbox shape"
    shapes.sort(key=lambda comp: comp["bbox"])
    return shapes


def _find_markers(grid: np.ndarray, bg: int) -> list[dict]:
    markers = []
    for color in sorted(set(map(int, grid.ravel())) - {bg, RECT_COLOR, SHAPE_COLOR}):
        for comp in _components_of_color(grid, color):
            if comp["area"] == 1:
                r, c = map(int, comp["coords"][0])
                markers.append({"color": int(color), "coord": (r, c)})
    markers.sort(key=lambda item: (item["coord"][0], item["coord"][1], item["color"]))
    return markers


def _ray_from_marker(coord: Coord, rect_bbox: BBox) -> tuple[str, Coord, Coord] | None:
    r, c = coord
    r0, c0, r1, c1 = rect_bbox
    if r0 <= r <= r1 and c < c0:
        return "h", (0, 1), (r, c0 - 1)
    if r0 <= r <= r1 and c > c1:
        return "h", (0, -1), (r, c1 + 1)
    if c0 <= c <= c1 and r < r0:
        return "v", (1, 0), (r0 - 1, c)
    if c0 <= c <= c1 and r > r1:
        return "v", (-1, 0), (r1 + 1, c)
    return None


def _ray_cells(start: Coord, step: Coord, stop: Coord) -> list[Coord]:
    r, c = start
    dr, dc = step
    out = []
    while True:
        r += dr
        c += dc
        out.append((r, c))
        if (r, c) == stop:
            return out


def _first_hit_bbox(path: list[Coord], shape_bboxes: list[BBox]) -> BBox | None:
    best_idx = None
    best_bbox = None
    for bbox in shape_bboxes:
        for idx, cell in enumerate(path):
            if _in_bbox(cell, bbox):
                if best_idx is None or idx < best_idx:
                    best_idx = idx
                    best_bbox = bbox
                break
    return best_bbox


def _apply_bbox_transform(out: np.ndarray, bbox: BBox, fill_color: int, orient: str, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
    patch[patch == bg] = int(fill_color)
    if orient == "h":
        patch = np.fliplr(patch)
    else:
        patch = np.flipud(patch)
    out[r0 : r1 + 1, c0 : c1 + 1] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast aligned rays toward the large rectangle, then fill and flip each hit 3x3 motif according to ray direction.

    Input structure
    - The background is grey, with one solid maroon rectangle near the upper-middle of the grid.
    - Several yellow 4-connected motifs each occupy part of a 3x3 bounding box around that rectangle.
    - Other non-background pixels are isolated colored markers; some share a row or column with the rectangle.

    Rule
    - Find the solid maroon rectangle, every yellow 3x3-bbox motif, and every isolated colored marker.
    - For each marker aligned with the rectangle, cast a same-colored horizontal or vertical ray toward the rectangle and stop one cell before the rectangle would be entered.
    - When that ray crosses a yellow motif's 3x3 box, fill the grey cells inside that box with the ray color, then flip the full 3x3 patch left-right for a horizontal ray or top-bottom for a vertical ray.
    - Draw the ray itself everywhere outside the hit 3x3 box, keep unhit yellow motifs unchanged, and process all horizontal motif flips before all vertical ones.

    Key insight
    - A ray carries two pieces of information at once: its color completes the missing pixels, and its orientation selects the mirror axis for the completed motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    rect_bbox = _find_rectangle(g)
    shape_bboxes = [shape["bbox"] for shape in _find_shapes(g)]
    markers = _find_markers(g, bg)

    rays = []
    for marker in markers:
        info = _ray_from_marker(marker["coord"], rect_bbox)
        if info is None:
            continue
        orient, step, stop = info
        path = _ray_cells(marker["coord"], step, stop)
        rays.append(
            {
                "orient": orient,
                "color": int(marker["color"]),
                "coord": marker["coord"],
                "path": path,
                "bbox": _first_hit_bbox(path, shape_bboxes),
            }
        )

    out = g.copy()
    for orient in ("h", "v"):
        seen_pairs: set[tuple[BBox, str]] = set()
        group = [ray for ray in rays if ray["orient"] == orient]
        group.sort(key=lambda ray: (ray["coord"][0], ray["coord"][1], ray["color"]))
        for ray in group:
            bbox = ray["bbox"]
            if bbox is not None and (bbox, orient) not in seen_pairs:
                _apply_bbox_transform(out, bbox, int(ray["color"]), orient, bg)
                seen_pairs.add((bbox, orient))
            for cell in ray["path"]:
                if bbox is None or not _in_bbox(cell, bbox):
                    out[cell] = int(ray["color"])
    return out


def _connected_shape_masks() -> list[np.ndarray]:
    masks = []
    for bits in range(1 << 9):
        coords = [(idx // 3, idx % 3) for idx in range(9) if (bits >> idx) & 1]
        if not 2 <= len(coords) <= 6:
            continue
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        if min(rows) != 0 or max(rows) != 2 or min(cols) != 0 or max(cols) != 2:
            continue
        q = deque([coords[0]])
        seen = {coords[0]}
        coord_set = set(coords)
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in coord_set and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if len(seen) != len(coords):
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        masks.append(mask)
    return masks


ALL_MASKS = _connected_shape_masks()
H_FLIP_VISIBLE_MASKS = [mask for mask in ALL_MASKS if not np.array_equal(mask, np.fliplr(mask))]
V_FLIP_VISIBLE_MASKS = [mask for mask in ALL_MASKS if not np.array_equal(mask, np.flipud(mask))]


def _bbox_clear(grid: np.ndarray, bbox: BBox, *, pad: int = 0) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return bool(np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == BACKGROUND))


def _paint_mask(grid: np.ndarray, bbox: BBox, mask: np.ndarray, color: int) -> None:
    r0, c0, _r1, _c1 = bbox
    patch = grid[r0 : r0 + 3, c0 : c0 + 3]
    patch[mask] = int(color)


def _sample_rect() -> tuple[int, int, int, int, int, int]:
    h = random.randint(20, 24)
    w = random.randint(20, 24)
    rect_h = random.randint(4, 6)
    rect_w = random.randint(4, 6)
    top = random.randint(6, min(8, h - rect_h - 6))
    left_low = max(5, w // 2 - rect_w - 1)
    left_high = min(w - rect_w - 6, w // 2 + 1)
    left = random.randint(left_low, left_high)
    return h, w, top, left, rect_h, rect_w


def _active_orientations() -> list[str]:
    if random.random() < 0.5:
        result = ["h", "h", "v"]
    else:
        result = ["h", "v", "v"]
    random.shuffle(result)
    return result


def _place_active_rays(grid: np.ndarray, rect_bbox: BBox) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = rect_bbox
    orientations = _active_orientations()
    colors = list(RAY_COLORS)
    random.shuffle(colors)

    h_needed = orientations.count("h")
    v_needed = orientations.count("v")
    rows = random.sample(list(range(r0, r1 + 1)), h_needed)
    cols = random.sample(list(range(c0, c1 + 1)), v_needed)
    h_sides = ["left", "right"][:h_needed]
    v_sides = ["top", "bottom"][:v_needed]
    if h_needed == 1:
        h_sides = [random.choice(["left", "right"])]
    elif h_needed == 2:
        random.shuffle(h_sides)
    if v_needed == 1:
        v_sides = [random.choice(["top", "bottom"])]
    elif v_needed == 2:
        random.shuffle(v_sides)

    active_h_rows = []
    active_v_cols = []
    h_iter = iter(zip(rows, h_sides))
    v_iter = iter(zip(cols, v_sides))

    for color, orient in zip(colors, orientations):
        if orient == "h":
            row, side = next(h_iter)
            gap = random.randint(1, 3)
            if side == "left":
                bbox_c1 = c0 - gap - 1
                bbox = (row - 1, bbox_c1 - 2, row + 1, bbox_c1)
                marker_c = random.randint(0, bbox[1] - 2)
            else:
                bbox_c0 = c1 + gap + 1
                bbox = (row - 1, bbox_c0, row + 1, bbox_c0 + 2)
                marker_c = random.randint(bbox[3] + 2, grid.shape[1] - 1)
            assert _bbox_clear(grid, bbox, pad=1)
            mask = random.choice(H_FLIP_VISIBLE_MASKS)
            _paint_mask(grid, bbox, mask, SHAPE_COLOR)
            grid[row, marker_c] = int(color)
            active_h_rows.append(row)
            continue

        col, side = next(v_iter)
        gap = random.randint(1, 3)
        if side == "top":
            bbox_r1 = r0 - gap - 1
            bbox = (bbox_r1 - 2, col - 1, bbox_r1, col + 1)
            marker_r = random.randint(0, bbox[0] - 2)
        else:
            bbox_r0 = r1 + gap + 1
            bbox = (bbox_r0, col - 1, bbox_r0 + 2, col + 1)
            marker_r = random.randint(bbox[2] + 2, grid.shape[0] - 1)
        assert _bbox_clear(grid, bbox, pad=1)
        mask = random.choice(V_FLIP_VISIBLE_MASKS)
        _paint_mask(grid, bbox, mask, SHAPE_COLOR)
        grid[marker_r, col] = int(color)
        active_v_cols.append(col)

    return active_h_rows, active_v_cols


def _extra_shape_candidates(grid: np.ndarray, rect_bbox: BBox, active_rows: list[int], active_cols: list[int]) -> list[BBox]:
    h, w = grid.shape
    r0, c0, r1, c1 = rect_bbox
    candidates = []
    for top in range(1, h - 3):
        for left in range(1, w - 3):
            bbox = (top, left, top + 2, left + 2)
            rows = {top, top + 1, top + 2}
            cols = {left, left + 1, left + 2}
            if rows & set(active_rows):
                continue
            if cols & set(active_cols):
                continue
            if not _bbox_clear(grid, bbox, pad=1):
                continue
            if not (bbox[2] < r0 - 1 or bbox[0] > r1 + 1 or bbox[3] < c0 - 1 or bbox[1] > c1 + 1):
                continue
            candidates.append(bbox)
    random.shuffle(candidates)
    return candidates


def _place_extra_shapes(grid: np.ndarray, rect_bbox: BBox, active_rows: list[int], active_cols: list[int]) -> None:
    candidates = _extra_shape_candidates(grid, rect_bbox, active_rows, active_cols)
    wanted = random.randint(0, 2)
    for bbox in candidates[:wanted]:
        mask = random.choice(ALL_MASKS)
        _paint_mask(grid, bbox, mask, SHAPE_COLOR)


def _inactive_marker_candidates(grid: np.ndarray, rect_bbox: BBox) -> list[Coord]:
    h, w = grid.shape
    r0, c0, r1, c1 = rect_bbox
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BACKGROUND:
                continue
            if r0 <= r <= r1 or c0 <= c <= c1:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if grid[r + dr, c + dc] != BACKGROUND:
                        ok = False
            if ok:
                out.append((r, c))
    random.shuffle(out)
    return out


def _place_inactive_markers(grid: np.ndarray, rect_bbox: BBox) -> None:
    candidates = _inactive_marker_candidates(grid, rect_bbox)
    wanted = random.randint(0, 2)
    for (r, c), color in zip(candidates, (random.choice(RAY_COLORS) for _ in range(wanted))):
        grid[r, c] = int(color)


def _build_input() -> np.ndarray:
    h, w, top, left, rect_h, rect_w = _sample_rect()
    grid = np.full((h, w), BACKGROUND, dtype=int)
    rect_bbox = (top, left, top + rect_h - 1, left + rect_w - 1)
    r0, c0, r1, c1 = rect_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR
    active_rows, active_cols = _place_active_rays(grid, rect_bbox)
    _place_extra_shapes(grid, rect_bbox, active_rows, active_cols)
    _place_inactive_markers(grid, rect_bbox)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid = _build_input()
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(grid, out):
            continue
        if int(np.sum(grid == SHAPE_COLOR)) < 6:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
