from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (16, 20)
GRID_W_RANGE = (16, 20)
REGION_COUNT_RANGE = (3, 5)
AREA_CHOICES = [4, 6, 8, 10, 12, 14]
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_components(mask)) == 1


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool]:
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    return horizontal, vertical


def _classify_boundary(mask: np.ndarray) -> int:
    horizontal, vertical = _symmetry_flags(mask)
    if horizontal and vertical:
        return GREEN
    if horizontal:
        return BLUE
    if vertical:
        return RED
    raise ValueError("boundary does not match a supported symmetry class")


def _row_interval_mask(intervals: list[tuple[int, int]], width: int) -> np.ndarray:
    mask = np.zeros((len(intervals), width), dtype=bool)
    for r, (c0, c1) in enumerate(intervals):
        mask[r, c0 : c1 + 1] = True
    return mask


def _choose_overlapping_interval(prev: tuple[int, int], width: int, centered: bool) -> tuple[int, int] | None:
    prev_l, prev_r = prev
    candidates: list[tuple[int, int]] = []
    if centered:
        for left in range(width):
            right = width - 1 - left
            if left > right:
                continue
            if max(left, prev_l) <= min(right, prev_r):
                candidates.append((left, right))
    else:
        for left in range(width):
            for right in range(left, width):
                if max(left, prev_l) <= min(right, prev_r):
                    candidates.append((left, right))
    if not candidates:
        return None
    return random.choice(candidates)


def _make_h_blob(height: int, width: int) -> np.ndarray | None:
    if height < 3 or width < 3:
        return None
    core_choices = [(l, r) for l in range(width) for r in range(l, width) if r - l + 1 >= 2]
    if not core_choices:
        return None

    if height % 2:
        center = random.choice(core_choices)
        above: list[tuple[int, int]] = []
        prev = center
        for _ in range(height // 2):
            nxt = _choose_overlapping_interval(prev, width, centered=False)
            if nxt is None:
                return None
            above.append(nxt)
            prev = nxt
        intervals = above[::-1] + [center] + above
    else:
        upper_mid = random.choice(core_choices)
        above = []
        prev = upper_mid
        for _ in range(height // 2 - 1):
            nxt = _choose_overlapping_interval(prev, width, centered=False)
            if nxt is None:
                return None
            above.append(nxt)
            prev = nxt
        top = above[::-1] + [upper_mid]
        intervals = top + top[::-1]

    mask = _row_interval_mask(intervals, width)
    horizontal, vertical = _symmetry_flags(mask)
    if horizontal and not vertical and _is_connected(mask):
        return mask
    return None


def _make_both_blob(height: int, width: int) -> np.ndarray | None:
    if height < 3 or width < 3:
        return None
    centered_choices = [(l, width - 1 - l) for l in range((width + 1) // 2) if width - 2 * l >= 2]
    if not centered_choices:
        return None

    if height % 2:
        center = random.choice(centered_choices)
        above: list[tuple[int, int]] = []
        prev = center
        for _ in range(height // 2):
            nxt = _choose_overlapping_interval(prev, width, centered=True)
            if nxt is None:
                return None
            above.append(nxt)
            prev = nxt
        intervals = above[::-1] + [center] + above
    else:
        upper_mid = random.choice(centered_choices)
        above = []
        prev = upper_mid
        for _ in range(height // 2 - 1):
            nxt = _choose_overlapping_interval(prev, width, centered=True)
            if nxt is None:
                return None
            above.append(nxt)
            prev = nxt
        top = above[::-1] + [upper_mid]
        intervals = top + top[::-1]

    mask = _row_interval_mask(intervals, width)
    horizontal, vertical = _symmetry_flags(mask)
    if horizontal and vertical and _is_connected(mask):
        return mask
    return None


def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr in (0, h - 1) or cc in (0, w - 1):
            boundary[rr, cc] = True
            continue
        if any(not mask[nr, nc] for nr, nc in _neighbors4(rr, cc)):
            boundary[rr, cc] = True
    return boundary


def _is_cycle(mask: np.ndarray) -> bool:
    if mask.sum() == 0 or len(_components(mask)) != 1:
        return False
    pts = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if degree != 2:
            return False
    return True


def _ring_from_notches(
    height: int,
    width: int,
    top_notches: list[tuple[int, int, int]],
    bottom_notches: list[tuple[int, int, int]],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[height - 1, :] = True
    mask[:, 0] = True
    mask[:, width - 1] = True

    for left, right, depth in top_notches:
        mask[0, left + 1 : right] = False
        mask[: depth + 1, left] = True
        mask[: depth + 1, right] = True
        mask[depth, left : right + 1] = True

    for left, right, depth in bottom_notches:
        floor = height - 1 - depth
        mask[height - 1, left + 1 : right] = False
        mask[floor:height, left] = True
        mask[floor:height, right] = True
        mask[floor, left : right + 1] = True

    return mask


def _build_cycle_patch(kind: str) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(4, 8)
        width = random.randint(4, 8)
        if height < 4 or width < 4:
            continue

        top_notches: list[tuple[int, int, int]] = []
        bottom_notches: list[tuple[int, int, int]] = []

        if kind == "H":
            if height < 5 or width < 5:
                continue
            left = random.randint(1, width - 3)
            right = random.randint(left + 1, width - 2)
            if left == width - 1 - right:
                continue
            max_depth = (height - 3) // 2
            if max_depth < 1:
                continue
            depth = random.randint(1, max_depth)
            top_notches = [(left, right, depth)]
            bottom_notches = [(left, right, depth)]
        elif kind == "V":
            if height < 5 or width < 5:
                continue
            left = random.randint(1, max(1, width // 2 - 1))
            right = width - 1 - left
            if left >= right:
                left, right = 1, width - 2
            depth = random.randint(1, height - 3)
            top_notches = [(left, right, depth)]
        else:
            if random.random() < 0.35:
                pass
            else:
                if height < 5 or width < 5:
                    continue
                left = random.randint(1, max(1, width // 2 - 1))
                right = width - 1 - left
                if left >= right:
                    left, right = 1, width - 2
                max_depth = (height - 3) // 2
                if max_depth < 1:
                    continue
                depth = random.randint(1, max_depth)
                top_notches = [(left, right, depth)]
                bottom_notches = [(left, right, depth)]

        boundary = _ring_from_notches(height, width, top_notches, bottom_notches)
        if not _is_cycle(boundary):
            continue

        horizontal, vertical = _symmetry_flags(boundary)
        if kind == "H" and not (horizontal and not vertical):
            continue
        if kind == "V" and not (vertical and not horizontal):
            continue
        if kind == "B" and not (horizontal and vertical):
            continue

        patch = np.zeros(boundary.shape, dtype=int)
        patch[boundary] = GREY
        enclosed = _extract_enclosed_regions(patch)
        if len(enclosed) != 1:
            continue
        if not (4 <= int(enclosed[0].shape[0]) <= 20):
            continue
        if not (10 <= int(boundary.sum()) <= 30):
            continue
        return patch
    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    nonzero = patch != BLACK
    window[nonzero] = patch[nonzero]


def _extract_enclosed_regions(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    black = grid == BLACK
    seen = np.zeros((h, w), dtype=bool)
    enclosed: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                enclosed.append(np.array(coords, dtype=int))
    return enclosed


def _boundary_coords_for_region(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    region = {tuple(map(int, rc)) for rc in coords.tolist()}
    boundary: set[tuple[int, int]] = set()
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                boundary.add((nr, nc))
    if not boundary:
        raise ValueError("enclosed region without a grey boundary")
    return np.array(sorted(boundary), dtype=int)


def _boundary_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox([tuple(map(int, rc)) for rc in coords.tolist()])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black region according to the reflection symmetry of its grey boundary.

    Input structure:
    - Black background cells and one-pixel-thick grey closed boundaries appear on a 2D grid.
    - Some black 4-connected regions are fully enclosed by grey pixels, while the outer black background touches the grid edge.
    - Each enclosed region is surrounded by a grey boundary whose mask is either only horizontally symmetric, only vertically symmetric, or symmetric in both directions.

    Rule:
    - Find every black 4-connected component that does not touch the grid edge.
    - For each such enclosed region, collect all grey pixels that are 4-neighbors of the region and treat that set as the region's boundary mask.
    - Fill the whole region blue when the boundary is only horizontally symmetric, red when it is only vertically symmetric, and green when it is symmetric both horizontally and vertically.
    - Leave all grey boundary pixels and all non-enclosed black background pixels unchanged.

    Key insight:
    - The fill color is routed by the geometry of the enclosing boundary, not by the size or position of the empty region inside it.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for region in _extract_enclosed_regions(g):
        boundary_coords = _boundary_coords_for_region(g, region)
        fill_color = _classify_boundary(_boundary_mask(boundary_coords))
        out[region[:, 0], region[:, 1]] = fill_color
    return out


def _try_place_regions(grid_h: int, grid_w: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        positions = [
            (r0, c0)
            for r0 in range(PLACEMENT_MARGIN, grid_h - ph - PLACEMENT_MARGIN + 1)
            for c0 in range(PLACEMENT_MARGIN, grid_w - pw - PLACEMENT_MARGIN + 1)
        ]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            rr0 = max(0, r0 - PLACEMENT_GAP)
            cc0 = max(0, c0 - PLACEMENT_GAP)
            rr1 = min(grid_h, r0 + ph + PLACEMENT_GAP)
            cc1 = min(grid_w, c0 + pw + PLACEMENT_GAP)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            _paste_patch(grid, patch, r0, c0)
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch != BLACK
            placed = True
            break
        if not placed:
            return None
    return grid


def _analyze_input(grid: np.ndarray) -> list[dict]:
    regions = _extract_enclosed_regions(grid)
    info = []
    for region in regions:
        boundary_coords = _boundary_coords_for_region(grid, region)
        boundary_mask = _boundary_mask(boundary_coords)
        info.append(
            {
                "area": int(region.shape[0]),
                "boundary_area": int(boundary_mask.sum()),
                "color": int(_classify_boundary(boundary_mask)),
            }
        )
    return info


def _is_valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, GREY}):
        return False

    info = _analyze_input(grid)
    if not (REGION_COUNT_RANGE[0] <= len(info) <= REGION_COUNT_RANGE[1]):
        return False

    colors = {item["color"] for item in info}
    if colors != {BLUE, RED, GREEN}:
        return False

    if any(not (4 <= item["area"] <= 20) for item in info):
        return False
    if any(not (10 <= item["boundary_area"] <= 30) for item in info):
        return False

    # Block area-only shortcuts: at least two symmetry classes must share one enclosed area.
    area_to_colors: dict[int, set[int]] = {}
    for item in info:
        area_to_colors.setdefault(item["area"], set()).add(item["color"])
    if not any(len(cols) >= 2 for cols in area_to_colors.values()):
        return False

    out = main(grid)
    if not np.array_equal(grid == GREY, out == GREY):
        return False
    if not np.all(out[(grid == BLACK) & (out != BLACK)] != GREY):
        return False
    if not {BLUE, RED, GREEN}.issubset(set(map(int, np.unique(out)))):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid_h = random.randint(*GRID_H_RANGE)
        grid_w = random.randint(*GRID_W_RANGE)
        region_count = random.randint(*REGION_COUNT_RANGE)

        categories = ["H", "V", "B"]
        while len(categories) < region_count:
            categories.append(random.choice(["H", "V", "B"]))
        random.shuffle(categories)

        patches: list[np.ndarray] = []
        ok = True
        for kind in categories:
            patch = _build_cycle_patch(kind)
            if patch is None:
                ok = False
                break
            patches.append(patch)
        if not ok:
            continue

        grid = _try_place_regions(grid_h, grid_w, patches)
        if grid is None:
            continue
        if _is_valid_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 2")
