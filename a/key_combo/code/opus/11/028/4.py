import random
from collections import deque

import numpy as np


BG = 0
ORANGE = 7
MAIN_COLORS = (1, 2, 3)
ODD_SIZES = (9, 11, 13)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bbox")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    dirs = DIRS4 if connectivity == 4 else DIRS8
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        q = deque([(r, c)])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            for dr, dc in dirs:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _complete_horizontal(patch: np.ndarray) -> np.ndarray:
    src = np.asarray(patch, dtype=int)
    h, w = src.shape
    out = src.copy()
    for r, c in np.argwhere(src == BG):
        rr = h - 1 - int(r)
        cc = int(c)
        color = int(src[rr, cc])
        if color != BG:
            out[int(r), int(c)] = color
    return out


def _complete_vertical(patch: np.ndarray) -> np.ndarray:
    src = np.asarray(patch, dtype=int)
    h, w = src.shape
    out = src.copy()
    for r, c in np.argwhere(src == BG):
        rr = int(r)
        cc = w - 1 - int(c)
        color = int(src[rr, cc])
        if color != BG:
            out[int(r), int(c)] = color
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orange_mask = g == ORANGE
    orange_count = int(orange_mask.sum())
    assert 2 <= orange_count <= 4, "orange indicator must contain 2-4 pixels"
    assert _component_count(orange_mask, 4) == 1, "orange indicator must be a single 4-connected cluster"

    main_mask = (g != BG) & (g != ORANGE)
    assert main_mask.any(), "main pattern is missing"
    box = _bbox(main_mask)
    patch = _crop_bbox(g, box)
    assert not np.any(patch == ORANGE), "orange pixels must be outside the main bbox"
    assert np.any(patch == BG), "main pattern must contain erased cells"

    h_patch = _complete_horizontal(patch)
    v_patch = _complete_vertical(patch)
    h_count = int(np.count_nonzero(h_patch))
    v_count = int(np.count_nonzero(v_patch))
    assert h_count != v_count, "horizontal and vertical candidate counts must differ"
    assert not np.array_equal(h_patch, v_patch), "horizontal and vertical candidates must differ"

    matches = []
    if h_count // 10 == orange_count:
        matches.append("H")
    if v_count // 10 == orange_count:
        matches.append("V")
    assert len(matches) == 1, "orange count must uniquely select one axis"

    chosen_axis = matches[0]
    chosen = h_patch if chosen_axis == "H" else v_patch
    assert not np.array_equal(chosen, patch), "chosen completion must change the input crop"

    return {
        "bbox": box,
        "patch": patch,
        "horizontal": h_patch,
        "vertical": v_patch,
        "horizontal_count": h_count,
        "vertical_count": v_count,
        "axis": chosen_axis,
        "output": chosen,
        "orange_count": orange_count,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the orange pixel count to choose the correct reflection axis and complete the erased pattern.

    Input structure:
    - The background is black (0).
    - One multicolor main pattern uses only blue/red/green cells and contains erased black holes.
    - A separate orange (7) 4-connected indicator cluster sits away from the main pattern.

    Rule:
    - Crop the tight bbox of all non-black, non-orange cells to isolate the main pattern.
    - Build two candidate completions of that crop: one by mirroring each black cell across the horizontal midline, and one by mirroring across the vertical midline.
    - Count the non-black cells in both completed candidates; let those totals be H and V.
    - Count the orange indicator pixels N, choose the unique axis whose completed total satisfies floor(total / 10) == N, and return only that completed crop.

    Key insight:
    - The visible pattern supports two plausible one-axis completions, so the correct axis is not determined internally; the orange cluster encodes the axis indirectly through the completed area count.
    """
    return _analyze(grid)["output"]


def _orbit_cells(h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r, c),
        (r, w - 1 - c),
        (h - 1 - r, c),
        (h - 1 - r, w - 1 - c),
    )


def _random_quarter_shape(cr: int, cc: int, count: int) -> list[tuple[int, int]] | None:
    boundary = {(cr - 1, c) for c in range(cc)} | {(r, cc - 1) for r in range(cr)}
    boundary = list(boundary)
    for _ in range(120):
        cells = {random.choice(boundary)}
        while len(cells) < count:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < cr and 0 <= nc < cc and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) == count:
            return sorted(cells)
    return None


def _paint_full_pattern(h: int, w: int, quarter_cells: list[tuple[int, int]], palette: tuple[int, ...]) -> np.ndarray:
    cr = h // 2
    cc = w // 2
    groups: list[list[tuple[int, int]]] = [[(cr, cc)]]
    for c in range(cc):
        groups.append([(cr, c), (cr, w - 1 - c)])
    for r in range(cr):
        groups.append([(r, cc), (h - 1 - r, cc)])
    for r, c in quarter_cells:
        groups.append(list(_orbit_cells(h, w, r, c)))

    order = list(range(len(groups)))
    random.shuffle(order)
    colors = [palette[0]] * len(groups)
    for idx, color in zip(order, random.sample(palette, len(palette))):
        colors[idx] = int(color)
    for idx in order[len(palette) :]:
        colors[idx] = int(random.choice(palette))

    patch = np.zeros((h, w), dtype=int)
    for color, group in zip(colors, groups):
        for r, c in group:
            patch[r, c] = int(color)
    return patch


def _valid_profiles(full_count: int, orbit_count: int) -> list[tuple[int, int, str]]:
    profiles = []
    for h_only in range(orbit_count + 1):
        for v_only in range(orbit_count - h_only + 1):
            if h_only + v_only == 0:
                continue
            h_count = full_count - 2 * v_only
            v_count = full_count - 2 * h_only
            if not (20 <= h_count < 50 and 20 <= v_count < 50):
                continue
            if not (2 <= h_count // 10 <= 4 and 2 <= v_count // 10 <= 4):
                continue
            if h_count // 10 == v_count // 10:
                continue
            if h_only > 0:
                profiles.append((h_only, v_only, "H"))
            if v_only > 0:
                profiles.append((h_only, v_only, "V"))
    return profiles


def _erase_half_orbits(
    full_patch: np.ndarray,
    quarter_cells: list[tuple[int, int]],
    h_only: int,
    v_only: int,
) -> np.ndarray:
    h, w = full_patch.shape
    partial = full_patch.copy()
    chosen = random.sample(quarter_cells, h_only + v_only)
    h_cells = chosen[:h_only]
    v_cells = chosen[h_only:]

    for r, c in h_cells:
        if random.random() < 0.5:
            coords = ((r, c), (r, w - 1 - c))
        else:
            coords = ((h - 1 - r, c), (h - 1 - r, w - 1 - c))
        for rr, cc in coords:
            partial[rr, cc] = BG

    for r, c in v_cells:
        if random.random() < 0.5:
            coords = ((r, c), (h - 1 - r, c))
        else:
            coords = ((r, w - 1 - c), (h - 1 - r, w - 1 - c))
        for rr, cc in coords:
            partial[rr, cc] = BG

    remaining = [cell for cell in quarter_cells if cell not in chosen]
    random.shuffle(remaining)
    common_count = random.randint(0, min(2, len(remaining)))
    for r, c in remaining[:common_count]:
        rr, cc = random.choice(_orbit_cells(h, w, r, c))
        partial[rr, cc] = BG

    return partial


def _orange_shapes(size: int) -> list[list[tuple[int, int]]]:
    if size == 2:
        return [[(0, 0), (0, 1)], [(0, 0), (1, 0)]]
    if size == 3:
        return [
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (1, 0), (1, 1)],
        ]
    if size == 4:
        return [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (1, 0), (2, 0), (2, 1)],
            [(0, 1), (1, 0), (1, 1), (1, 2)],
        ]
    raise ValueError(f"unsupported orange cluster size: {size}")


def _place_orange_cluster(grid: np.ndarray, patch_box: tuple[int, int, int, int], size: int) -> None:
    r0, c0, r1, c1 = patch_box
    for _ in range(80):
        shape = random.choice(_orange_shapes(size))
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        min_r = r1 + 2
        min_c = c1 + 2
        max_r = grid.shape[0] - sh
        max_c = grid.shape[1] - sw
        if min_r > max_r or min_c > max_c:
            continue
        base_r = random.randint(min_r, max_r)
        base_c = random.randint(min_c, max_c)
        coords = [(base_r + dr, base_c + dc) for dr, dc in shape]
        if any(grid[rr, cc] != BG for rr, cc in coords):
            continue
        for rr, cc in coords:
            grid[rr, cc] = ORANGE
        return
    raise RuntimeError("failed to place orange indicator")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.choice(ODD_SIZES)
        w = random.choice(ODD_SIZES)
        cr = h // 2
        cc = w // 2
        max_orbits = min(6, cr * cc)
        orbit_count = random.randint(3, max_orbits)
        profiles = _valid_profiles((h + w - 1) + 4 * orbit_count, orbit_count)
        if not profiles:
            continue

        quarter_cells = _random_quarter_shape(cr, cc, orbit_count)
        if quarter_cells is None:
            continue

        palette_size = random.choice((2, 3))
        palette = tuple(sorted(random.sample(MAIN_COLORS, palette_size)))
        full_patch = _paint_full_pattern(h, w, quarter_cells, palette)
        h_only, v_only, target_axis = random.choice(profiles)
        partial_patch = _erase_half_orbits(full_patch, quarter_cells, h_only, v_only)

        if _component_count(partial_patch != BG, 8) != 1:
            continue

        h_candidate = _complete_horizontal(partial_patch)
        v_candidate = _complete_vertical(partial_patch)
        if np.array_equal(h_candidate, v_candidate):
            continue

        h_count = int(np.count_nonzero(h_candidate))
        v_count = int(np.count_nonzero(v_candidate))
        chosen_count = h_count if target_axis == "H" else v_count
        other_count = v_count if target_axis == "H" else h_count
        indicator_size = chosen_count // 10
        if not (2 <= indicator_size <= 4):
            continue
        if other_count // 10 == indicator_size:
            continue
        if np.array_equal(h_candidate if target_axis == "H" else v_candidate, partial_patch):
            continue

        top = random.randint(0, 2)
        left = random.randint(0, 2)
        extra_bottom = random.randint(4, 8)
        extra_right = random.randint(4, 8)
        grid_h = h + top + extra_bottom
        grid_w = w + left + extra_right
        if max(grid_h, grid_w) > 30:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        grid[top : top + h, left : left + w] = partial_patch
        patch_box = (top, left, top + h - 1, left + w - 1)

        try:
            _place_orange_cluster(grid, patch_box, indicator_size)
        except RuntimeError:
            continue

        analysis = _analyze(grid)
        if analysis["axis"] != target_axis:
            continue
        if not np.array_equal(analysis["output"], h_candidate if target_axis == "H" else v_candidate):
            continue
        if _component_count((grid != BG) & (grid != ORANGE), 8) != 1:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
