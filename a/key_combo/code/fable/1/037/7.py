from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
SECTOR_COLOR_CYCLE = (1, 2, 3, 4)
FRAGMENT_COLORS = (5, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GEN_ATTEMPTS = 300
GEN_MARGIN = 1
GEN_GAP = 2
_GEN_COUNTER = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.astype(np.uint8).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _shape_equal_under_rotation(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return _mask_signature(mask_a) == _mask_signature(mask_b)


def _component_sets(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                    "sum_r": int(coords[:, 0].sum()),
                    "sum_c": int(coords[:, 1].sum()),
                    "mask": _coords_to_mask(coords),
                }
            )
    return components


def _rotate_coords(coords: np.ndarray, center2: tuple[int, int], steps: int, shape: tuple[int, int]) -> np.ndarray | None:
    rr2 = 2 * coords[:, 0] - int(center2[0])
    cc2 = 2 * coords[:, 1] - int(center2[1])
    for _ in range(int(steps) % 4):
        rr2, cc2 = cc2.copy(), -rr2.copy()
    nr2 = int(center2[0]) + rr2
    nc2 = int(center2[1]) + cc2
    if np.any(nr2 % 2) or np.any(nc2 % 2):
        return None
    nr = nr2 // 2
    nc = nc2 // 2
    h, w = map(int, shape)
    if np.any(nr < 0) or np.any(nr >= h) or np.any(nc < 0) or np.any(nc >= w):
        return None
    return np.stack([nr, nc], axis=1)


def _coords_key(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset(map(tuple, coords.tolist()))


def _unique_fragment_rotation(fragment_mask: np.ndarray, hole_mask: np.ndarray) -> bool:
    matches = 0
    for k in range(4):
        if np.array_equal(np.rot90(fragment_mask, k), hole_mask):
            matches += 1
    return matches == 1


def _candidate_centers2(components: list[dict]) -> list[tuple[int, int]]:
    centers: set[tuple[int, int]] = set()
    for src in components:
        area = int(src["area"])
        for dst in components:
            if src is dst:
                continue
            sr_a = int(src["sum_r"])
            sc_a = int(src["sum_c"])
            sr_b = int(dst["sum_r"])
            sc_b = int(dst["sum_c"])

            numerators = (
                (sr_a + sr_b + sc_b - sc_a, sr_a - sr_b + sc_a + sc_b),
                (sr_a + sr_b, sc_a + sc_b),
                (sr_a + sr_b + sc_a - sc_b, -sr_a + sr_b + sc_a + sc_b),
            )
            for num_r, num_c in numerators:
                if num_r % area == 0 and num_c % area == 0:
                    center2 = (num_r // area, num_c // area)
                    if center2[0] % 2 == center2[1] % 2:
                        centers.add(center2)
    return sorted(centers)


def _discover_installation(grid: np.ndarray) -> tuple[int, np.ndarray, int]:
    bg = _get_bg(grid)
    components = _component_sets(grid, bg)
    assert len(components) == 4, "expected exactly three sectors and one detached fragment"

    areas = {component["area"] for component in components}
    assert len(areas) == 1, "all foreground components should have the same area"

    signatures = {_mask_signature(component["mask"]) for component in components}
    assert len(signatures) == 1, "all components should share one rotation-equivalent shape"

    comp_keys = [_coords_key(component["coords"]) for component in components]
    candidates: dict[tuple[frozenset[tuple[int, int]], int, int], tuple[int, np.ndarray, int]] = {}
    cycle_index = {color: idx for idx, color in enumerate(SECTOR_COLOR_CYCLE)}

    for center2 in _candidate_centers2(components):
        for ref_idx, ref in enumerate(components):
            if ref["color"] not in cycle_index:
                continue

            used: dict[int, int] = {}
            hole_coords: np.ndarray | None = None
            hole_step: int | None = None
            for step in range(4):
                rotated = _rotate_coords(ref["coords"], center2, step, grid.shape)
                if rotated is None:
                    break
                key = _coords_key(rotated)
                matches = [idx for idx, comp_key in enumerate(comp_keys) if comp_key == key]
                if len(matches) == 1:
                    comp_idx = matches[0]
                    if comp_idx in used:
                        break
                    used[comp_idx] = step
                elif len(matches) == 0:
                    if hole_coords is not None:
                        break
                    hole_coords = rotated
                    hole_step = step
                else:
                    break
            else:
                if hole_coords is None or hole_step is None or len(used) != 3:
                    continue
                fragment_idxs = [idx for idx in range(4) if idx not in used]
                if len(fragment_idxs) != 1:
                    continue
                fragment_idx = fragment_idxs[0]
                if ref_idx == fragment_idx:
                    continue

                start = cycle_index[ref["color"]]
                if any(
                    components[idx]["color"] != SECTOR_COLOR_CYCLE[(start + step) % len(SECTOR_COLOR_CYCLE)]
                    for idx, step in used.items()
                ):
                    continue

                hole_color = SECTOR_COLOR_CYCLE[(start + hole_step) % len(SECTOR_COLOR_CYCLE)]
                if components[fragment_idx]["color"] == hole_color:
                    continue
                if not np.all(grid[hole_coords[:, 0], hole_coords[:, 1]] == bg):
                    continue

                hole_mask = _coords_to_mask(hole_coords)
                if not _unique_fragment_rotation(components[fragment_idx]["mask"], hole_mask):
                    continue

                key = (_coords_key(hole_coords), fragment_idx, hole_color)
                candidates[key] = (fragment_idx, hole_coords, hole_color)

    assert len(candidates) == 1, "expected one unique rotational completion"
    return next(iter(candidates.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinstall the detached quarter-turn fragment into the missing rotational sector and repaint it with the cycle-implied color.

    Input structure:
    - The background contains four same-shape monochrome components: three are installed sectors of one large figure and one is a detached fragment elsewhere.
    - The installed sectors are quarter-turn rotations of the same sprite around one common center, with exactly one sector position missing as background.
    - Their colors follow a fixed visible cycle along clockwise rotation, while the detached fragment uses a distractor color that is not the hole color.

    Rule:
    - Find the unique rotation center and sector layout for which one rotated sector position is missing and the other three positions are occupied by matching components.
    - Use the fixed color cycle to infer the missing sector color from the three installed sectors.
    - Clear the detached fragment, then fill the missing sector cells with that inferred color.

    Key insight:
    - The detached piece supplies only the missing shape; the correct output color comes from the rotational color cycle carried by the three sectors already in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    fragment_idx, hole_coords, hole_color = _discover_installation(g)
    components = _component_sets(g, bg)

    out = g.copy()
    fragment_coords = components[fragment_idx]["coords"]
    out[fragment_coords[:, 0], fragment_coords[:, 1]] = bg
    out[hole_coords[:, 0], hole_coords[:, 1]] = int(hole_color)
    return out


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = sorted(
            {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in NEIGHBORS4
                if 0 <= r + dr < height and 0 <= c + dc < width and (r + dr, c + dc) not in cells
            }
        )
        cells.add(random.choice(frontier))
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _asymmetric_mask() -> np.ndarray:
    for _ in range(200):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        area_lo = max(4, height + width - 1)
        area_hi = min(height * width - 1, 7)
        if area_lo > area_hi:
            continue
        area = random.randint(area_lo, area_hi)
        mask = _grow_connected_mask(height, width, area)
        if all(not np.array_equal(np.rot90(mask, k), mask) for k in range(1, 4)):
            return mask
    raise RuntimeError("failed to generate a rotation-asymmetric sector mask")


def _base_offsets2(mask: np.ndarray, parity: int) -> np.ndarray:
    coords = np.argwhere(mask)
    r_steps = 2 * (coords[:, 0] + 1) + int(parity)
    c_steps = 2 * (coords[:, 1] + 1) + int(parity)
    return np.stack([-r_steps, c_steps], axis=1)


def _rotate_offsets2(offsets2: np.ndarray, steps: int) -> np.ndarray:
    rel = offsets2.copy()
    for _ in range(int(steps) % 4):
        rel = np.stack([rel[:, 1], -rel[:, 0]], axis=1)
    return rel


def _offsets2_to_coords(offsets2: np.ndarray, center2: tuple[int, int]) -> np.ndarray:
    rr2 = int(center2[0]) + offsets2[:, 0]
    cc2 = int(center2[1]) + offsets2[:, 1]
    assert not np.any(rr2 % 2) and not np.any(cc2 % 2)
    return np.stack([rr2 // 2, cc2 // 2], axis=1)


def _placement_ranges(side: str, grid_h: int, grid_w: int, main_h: int, main_w: int, frag_h: int, frag_w: int) -> tuple[range, range, range, range]:
    if side == "above":
        main_top = range(frag_h + GEN_GAP + GEN_MARGIN, grid_h - main_h - GEN_MARGIN + 1)
        main_left = range(GEN_MARGIN, grid_w - main_w - GEN_MARGIN + 1)
        return main_top, main_left, range(GEN_MARGIN, grid_h), range(GEN_MARGIN, grid_w)
    if side == "below":
        main_top = range(GEN_MARGIN, grid_h - main_h - frag_h - GEN_GAP - GEN_MARGIN + 1)
        main_left = range(GEN_MARGIN, grid_w - main_w - GEN_MARGIN + 1)
        return main_top, main_left, range(GEN_MARGIN, grid_h), range(GEN_MARGIN, grid_w)
    if side == "left":
        main_top = range(GEN_MARGIN, grid_h - main_h - GEN_MARGIN + 1)
        main_left = range(frag_w + GEN_GAP + GEN_MARGIN, grid_w - main_w - GEN_MARGIN + 1)
        return main_top, main_left, range(GEN_MARGIN, grid_h), range(GEN_MARGIN, grid_w)
    main_top = range(GEN_MARGIN, grid_h - main_h - GEN_MARGIN + 1)
    main_left = range(GEN_MARGIN, grid_w - main_w - frag_w - GEN_GAP - GEN_MARGIN + 1)
    return main_top, main_left, range(GEN_MARGIN, grid_h), range(GEN_MARGIN, grid_w)


def _valid_fragment_position(side: str, frag_top: int, frag_left: int, frag_h: int, frag_w: int, main_top: int, main_left: int, main_h: int, main_w: int) -> bool:
    if side == "above":
        return frag_top + frag_h + GEN_GAP <= main_top
    if side == "below":
        return main_top + main_h + GEN_GAP <= frag_top
    if side == "left":
        return frag_left + frag_w + GEN_GAP <= main_left
    return main_left + main_w + GEN_GAP <= frag_left


def _build_expected_output(inp: np.ndarray, fragment_mask: np.ndarray, fragment_top: int, fragment_left: int, hole_coords: np.ndarray, hole_color: int) -> np.ndarray:
    out = inp.copy()
    frag_coords = np.argwhere(fragment_mask) + np.array([fragment_top, fragment_left], dtype=int)
    out[frag_coords[:, 0], frag_coords[:, 1]] = BACKGROUND
    out[hole_coords[:, 0], hole_coords[:, 1]] = int(hole_color)
    return out


def _gen_attempt(branch: int) -> np.ndarray | None:
    parity = (branch // 16) % 2
    missing_step = branch % 4
    fragment_rot = (branch // 4) % 4
    side = ("above", "below", "left", "right")[(branch // 8) % 4]
    cycle_start = (branch // 32) % len(SECTOR_COLOR_CYCLE)

    sector_mask = _asymmetric_mask()
    offsets2 = _base_offsets2(sector_mask, parity)
    sample_center2 = (parity, parity)

    sector_coords = [
        _offsets2_to_coords(_rotate_offsets2(offsets2, step), sample_center2)
        for step in range(4)
    ]
    all_coords = np.concatenate(sector_coords, axis=0)
    sample_r0, sample_c0, sample_r1, sample_c1 = _bbox_coords(all_coords)
    main_h = sample_r1 - sample_r0 + 1
    main_w = sample_c1 - sample_c0 + 1

    hole_mask = _coords_to_mask(sector_coords[missing_step])
    fragment_mask = np.rot90(hole_mask, fragment_rot)
    frag_h, frag_w = map(int, fragment_mask.shape)

    min_h = max(main_h, frag_h) + 2 * GEN_MARGIN
    min_w = max(main_w, frag_w) + 2 * GEN_MARGIN
    if side in {"above", "below"}:
        min_h = main_h + frag_h + GEN_GAP + 2 * GEN_MARGIN
    else:
        min_w = main_w + frag_w + GEN_GAP + 2 * GEN_MARGIN

    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, min(30, min_h + 6))
    grid_w = random.randint(min_w, min(30, min_w + 6))

    main_top_choices, main_left_choices, frag_top_all, frag_left_all = _placement_ranges(
        side, grid_h, grid_w, main_h, main_w, frag_h, frag_w
    )
    if not main_top_choices or not main_left_choices:
        return None

    main_top = random.choice(list(main_top_choices))
    main_left = random.choice(list(main_left_choices))

    frag_positions: list[tuple[int, int]] = []
    for frag_top in range(GEN_MARGIN, grid_h - frag_h - GEN_MARGIN + 1):
        for frag_left in range(GEN_MARGIN, grid_w - frag_w - GEN_MARGIN + 1):
            if _valid_fragment_position(side, frag_top, frag_left, frag_h, frag_w, main_top, main_left, main_h, main_w):
                frag_positions.append((frag_top, frag_left))
    if not frag_positions:
        return None
    fragment_top, fragment_left = random.choice(frag_positions)

    shift_r = main_top - sample_r0
    shift_c = main_left - sample_c0
    center2 = (sample_center2[0] + 2 * shift_r, sample_center2[1] + 2 * shift_c)

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    sector_colors = [SECTOR_COLOR_CYCLE[(cycle_start + step) % len(SECTOR_COLOR_CYCLE)] for step in range(4)]
    hole_coords = _offsets2_to_coords(_rotate_offsets2(offsets2, missing_step), center2)

    for step in range(4):
        if step == missing_step:
            continue
        coords = _offsets2_to_coords(_rotate_offsets2(offsets2, step), center2)
        grid[coords[:, 0], coords[:, 1]] = int(sector_colors[step])

    fragment_color = random.choice([color for color in FRAGMENT_COLORS if color != sector_colors[missing_step]])
    frag_coords = np.argwhere(fragment_mask) + np.array([fragment_top, fragment_left], dtype=int)
    grid[frag_coords[:, 0], frag_coords[:, 1]] = int(fragment_color)

    expected = _build_expected_output(
        grid,
        fragment_mask,
        fragment_top,
        fragment_left,
        hole_coords,
        sector_colors[missing_step],
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    branch = _GEN_COUNTER
    _GEN_COUNTER += 1
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-sector completion input")
