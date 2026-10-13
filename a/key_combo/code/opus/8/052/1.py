from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _erase_occluder(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    grey_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert grey_coords, "expected a grey occluder"
    r0, c0, r1, c1 = _bbox(grey_coords)
    occ_h = r1 - r0 + 1
    occ_w = c1 - c0 + 1
    assert len(grey_coords) == occ_h * occ_w, "grey pixels must form one solid rectangle"
    cleared = grid.copy()
    cleared[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return cleared, (r0, c0, r1, c1)


def _mirror_blue(grid: np.ndarray) -> np.ndarray:
    blue_coords = np.argwhere(grid == BLUE)
    assert len(blue_coords) > 0, "expected blue pixels"
    min_c = int(blue_coords[:, 1].min())
    max_c = int(blue_coords[:, 1].max())
    axis_sum = min_c + max_c
    out = grid.copy()
    for r, c in blue_coords:
        mc = axis_sum - int(c)
        if 0 <= mc < grid.shape[1] and out[int(r), mc] == BLACK:
            out[int(r), mc] = BLUE
    return out


def _hole_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    holes: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for coords in _components(grid == BLACK):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords):
            continue
        holes.append(coords)
    return holes


def _fill_holes(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for coords in _hole_components(grid):
        color = RED if len(coords) % 2 == 0 else ORANGE
        for r, c in coords:
            out[r, c] = color
    return out


def _blue_component_count(grid: np.ndarray) -> int:
    return len(_components(grid == BLUE))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the grey occluder, restore the blue object's left-right symmetry, then fill the newly enclosed holes by parity.

    Input structure:
    - A black background contains one blue connected object and one grey solid rectangular occluder.
    - The complete blue object is left-right mirror symmetric around the midpoint of its blue horizontal span.
    - The grey rectangle overlaps part of the blue object so that potential internal black holes are open before restoration.

    Rule:
    - Detect the unique grey filled rectangle and erase it to black.
    - Use the remaining blue pixels to infer the vertical mirror axis from the blue horizontal range midpoint, then reflect every blue pixel across that axis to restore missing blue cells.
    - After restoration, find every black 4-connected region that does not touch the grid border; fill each such hole red when its area is even and orange when its area is odd.
    - Preserve the blue object and the exterior black background.

    Key insight:
    - The holes are not visible in the occluded input: they only become closed regions after the occluder is removed and the symmetric side wall is reconstructed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    cleared, _ = _erase_occluder(g)
    restored = _mirror_blue(cleared)
    return _fill_holes(restored)


def _build_blue_patch(rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int, int, int]]:
    hole_count = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
    outer_w = int(rng.choice([7, 8, 9, 10, 11, 12, 13]))
    heights = [int(rng.integers(5, 9)) for _ in range(hole_count)]
    if outer_w % 2 == 1 and hole_count >= 2 and len({h % 2 for h in heights}) == 1:
        idx = int(rng.integers(hole_count))
        if heights[idx] == 8:
            heights[idx] -= 1
        elif heights[idx] == 5:
            heights[idx] += 1
        else:
            heights[idx] += 1 if rng.random() < 0.5 else -1
    gaps = [int(rng.integers(1, 3)) for _ in range(hole_count - 1)]
    patch_h = sum(heights) + sum(gaps)
    patch = np.zeros((patch_h, outer_w), dtype=int)

    frames: list[tuple[int, int]] = []
    top = 0
    for i, height in enumerate(heights):
        bottom = top + height - 1
        patch[top, :] = BLUE
        patch[bottom, :] = BLUE
        patch[top : bottom + 1, 0] = BLUE
        patch[top : bottom + 1, outer_w - 1] = BLUE
        frames.append((top, bottom))
        if i < hole_count - 1:
            top = bottom + 1 + gaps[i]

    if outer_w % 2 == 1:
        center_cols = [outer_w // 2]
    else:
        center_cols = [outer_w // 2 - 1, outer_w // 2]
    for i in range(hole_count - 1):
        bottom = frames[i][1]
        next_top = frames[i + 1][0]
        for cc in center_cols:
            patch[bottom : next_top + 1, cc] = BLUE

    occ_w = int(rng.integers(2, min(4, outer_w // 2 + 1)))
    occ_top = frames[0][0] + 1
    occ_bottom = frames[-1][1] - 1
    if occ_bottom <= occ_top:
        occ_top = frames[0][0]
        occ_bottom = frames[-1][1]
    if rng.random() < 0.5:
        occ_bbox = (occ_top, 0, occ_bottom, occ_w - 1)
    else:
        occ_bbox = (occ_top, outer_w - occ_w, occ_bottom, outer_w - 1)
    return patch, frames, occ_bbox


def _place_patch(
    rng: np.random.Generator, patch: np.ndarray, occ_bbox: tuple[int, int, int, int]
) -> np.ndarray:
    ph, pw = patch.shape
    extra_rows_min = max(2, 15 - ph)
    extra_rows_max = 30 - ph
    extra_cols_min = max(2, 15 - pw)
    extra_cols_max = 30 - pw
    h = ph + int(rng.integers(extra_rows_min, extra_rows_max + 1))
    w = pw + int(rng.integers(extra_cols_min, extra_cols_max + 1))
    grid = np.zeros((h, w), dtype=int)
    top = int(rng.integers(1, h - ph))
    left = int(rng.integers(1, w - pw))
    blue_mask = patch == BLUE
    grid[top : top + ph, left : left + pw][blue_mask] = BLUE
    r0, c0, r1, c1 = occ_bbox
    grid[top + r0 : top + r1 + 1, left + c0 : left + c1 + 1] = GREY
    return grid


def _validate_sample(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    colors = set(map(int, np.unique(grid)))
    if not colors <= {BLACK, BLUE, GREY} or BLUE not in colors or GREY not in colors:
        return False
    if _blue_component_count(grid) != 1:
        return False

    cleared, occ_bbox = _erase_occluder(grid)
    if _blue_component_count(cleared) != 1:
        return False
    if _hole_components(cleared):
        return False

    blue_before = np.argwhere(cleared == BLUE)
    restored = _mirror_blue(cleared)
    blue_after = np.argwhere(restored == BLUE)
    if len(blue_after) <= len(blue_before):
        return False

    holes = _hole_components(restored)
    if not 1 <= len(holes) <= 3:
        return False

    added = (restored == BLUE) & (cleared != BLUE)
    r0, c0, r1, c1 = occ_bbox
    occ_mask = np.zeros_like(grid, dtype=bool)
    occ_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    if not np.all(~added | occ_mask):
        return False

    if _blue_component_count(restored) != 1:
        return False

    restored_bbox = _bbox([tuple(map(int, rc)) for rc in np.argwhere(restored == BLUE)])
    cleared_bbox = _bbox([tuple(map(int, rc)) for rc in blue_before])
    if restored_bbox != cleared_bbox:
        return False

    out = main(grid)
    if np.array_equal(out, restored):
        return False
    if not np.any(np.isin(out, [RED, ORANGE])):
        return False

    for coords in holes:
        expected = RED if len(coords) % 2 == 0 else ORANGE
        for r, c in coords:
            if out[r, c] != expected:
                return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        patch, _frames, occ_bbox = _build_blue_patch(rng)
        grid = _place_patch(rng, patch, occ_bbox)
        if _validate_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid sample")
