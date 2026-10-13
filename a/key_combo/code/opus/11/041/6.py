from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
try:
    from arc_utils import find_components, get_bg, in_bounds, neighbors4, neighbors8
except ModuleNotFoundError:
    _arc_utils = SourceFileLoader("_arc_utils_ref", str(_ARC_UTILS_PATH)).load_module()
    find_components = _arc_utils.find_components
    get_bg = _arc_utils.get_bg
    in_bounds = _arc_utils.in_bounds
    neighbors4 = _arc_utils.neighbors4
    neighbors8 = _arc_utils.neighbors8


BG = 0
MARKER = 2
RAY = 4
SKELETON = 5
COLLISION = 8
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 22
GRID_MAX = 28
COPY_COUNT_CHOICES = (3, 4)
SKELETON_SIZE_RANGE = (5, 8)
SKELETON_BBOX_MAX = 5
MAX_ATTEMPTS = 600


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rotate_once(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(c, -r) for r, c in cells}


def _shape_rotations(shape: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    out = []
    current = set(shape)
    for turns in range(4):
        if turns:
            current = _rotate_once(current)
        out.append(_normalize(current))
    return out


def _shape_with_point_rotations(
    shape: frozenset[tuple[int, int]],
    point: tuple[int, int],
) -> list[tuple[frozenset[tuple[int, int]], tuple[int, int]]]:
    out = []
    current_shape = set(shape)
    current_point = tuple(point)
    for turns in range(4):
        if turns == 0:
            rotated_shape = current_shape
            rotated_point = current_point
        else:
            rotated_shape = _rotate_once(rotated_shape)
            rotated_point = next(iter(_rotate_once({rotated_point})))
        min_r = min(r for r, _ in rotated_shape)
        min_c = min(c for _, c in rotated_shape)
        out.append(
            (
                frozenset((r - min_r, c - min_c) for r, c in rotated_shape),
                (rotated_point[0] - min_r, rotated_point[1] - min_c),
            )
        )
    return out


def _direction_from_centroid(
    skeleton: frozenset[tuple[int, int]],
    marker: tuple[int, int],
) -> tuple[int, int]:
    rows = [r for r, _ in skeleton]
    cols = [c for _, c in skeleton]
    dr = float(marker[0]) - float(sum(rows) / len(rows))
    dc = float(marker[1]) - float(sum(cols) / len(cols))
    assert not np.isclose(abs(dr), abs(dc)), "marker direction must not tie between axes"
    if abs(dr) > abs(dc):
        return (1, 0) if dr > 0 else (-1, 0)
    return (0, 1) if dc > 0 else (0, -1)


def _full_extents(
    skeleton: frozenset[tuple[int, int]],
    marker: tuple[int, int],
) -> tuple[int, int, int, int]:
    pts = set(skeleton) | {marker}
    return (
        min(r - marker[0] for r, _ in pts),
        max(r - marker[0] for r, _ in pts),
        min(c - marker[1] for _, c in pts),
        max(c - marker[1] for _, c in pts),
    )


def _dilate8(cells: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _render_from_rays(grid: np.ndarray, ray_cells_list: list[tuple[tuple[int, int], ...]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    cover = np.zeros(out.shape, dtype=int)
    for ray_cells in ray_cells_list:
        for r, c in ray_cells:
            out[r, c] = RAY
        for r, c in _dilate8(ray_cells, out.shape):
            cover[r, c] += 1
    for r, c in np.argwhere(cover >= 2):
        rr = int(r)
        cc = int(c)
        if int(grid[rr, cc]) == BG or int(out[rr, cc]) == RAY:
            out[rr, cc] = COLLISION
    return out


def _render_from_mask(
    grid: np.ndarray,
    ray_cells_list: list[tuple[tuple[int, int], ...]],
    mask: np.ndarray,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for ray_cells in ray_cells_list:
        for r, c in ray_cells:
            out[r, c] = RAY
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if int(grid[rr, cc]) == BG or int(out[rr, cc]) == RAY:
            out[rr, cc] = COLLISION
    return out


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BG, MARKER, SKELETON}
    bg = int(get_bg(g))
    assert bg == BG

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert 4 <= len(comps) <= 5, "expected one template and 3-4 copies"

    template_ids = [
        idx
        for idx, comp in enumerate(comps)
        if any(int(g[int(r), int(c)]) == MARKER for r, c in comp["coords"])
    ]
    assert len(template_ids) == 1, "expected exactly one template"
    template_id = template_ids[0]

    template = comps[template_id]
    template_skeleton_abs = {
        (int(r), int(c))
        for r, c in template["coords"]
        if int(g[int(r), int(c)]) == SKELETON
    }
    template_marker_abs = [
        (int(r), int(c))
        for r, c in template["coords"]
        if int(g[int(r), int(c)]) == MARKER
    ]
    assert len(template_marker_abs) == 1
    template_marker_abs = template_marker_abs[0]

    for r, c in template_skeleton_abs:
        assert any((nr, nc) == template_marker_abs for nr, nc in neighbors8(r, c)) or any(
            (nr, nc) in template_skeleton_abs for nr, nc in neighbors4(r, c)
        )

    tr0 = min(r for r, _ in template_skeleton_abs)
    tc0 = min(c for _, c in template_skeleton_abs)
    template_shape = frozenset((r - tr0, c - tc0) for r, c in template_skeleton_abs)
    template_marker = (template_marker_abs[0] - tr0, template_marker_abs[1] - tc0)
    variants = []
    seen_shapes = set()
    for turns, (shape_rot, marker_rot) in enumerate(_shape_with_point_rotations(template_shape, template_marker)):
        assert marker_rot not in shape_rot
        direction = _direction_from_centroid(shape_rot, marker_rot)
        seen_shapes.add(shape_rot)
        variants.append(
            {
                "turns": turns,
                "shape": shape_rot,
                "marker": marker_rot,
                "direction": direction,
                "extents": _full_extents(shape_rot, marker_rot),
            }
        )
    assert len(seen_shapes) == 4, "template skeleton must have four distinct rotations"

    motifs = []
    ray_cells_list: list[tuple[tuple[int, int], ...]] = []
    copy_rotations = []
    all_virtual_markers = set()

    for idx, comp in enumerate(comps):
        skeleton_abs = {
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(g[int(r), int(c)]) == SKELETON
        }
        assert len(skeleton_abs) == len(template_skeleton_abs), "all skeletons must share one shape"
        r0 = min(r for r, _ in skeleton_abs)
        c0 = min(c for _, c in skeleton_abs)
        shape_norm = frozenset((r - r0, c - c0) for r, c in skeleton_abs)
        matches = [variant for variant in variants if variant["shape"] == shape_norm]
        assert len(matches) == 1, "each copy must have a unique rotation match"
        variant = matches[0]

        marker_abs = (r0 + variant["marker"][0], c0 + variant["marker"][1])
        direction = variant["direction"]
        if idx == template_id:
            assert marker_abs == template_marker_abs
            copy_kind = "template"
        else:
            assert int(g[marker_abs]) == BG, "copy virtual marker must stay empty in the input"
            copy_kind = "copy"
            copy_rotations.append(int(variant["turns"]))
            all_virtual_markers.add(marker_abs)

        cells: list[tuple[int, int]] = []
        rr, cc = marker_abs
        first = True
        while in_bounds(g.shape, rr, cc):
            val = int(g[rr, cc])
            if val == BG:
                cells.append((rr, cc))
            else:
                assert first and val == MARKER, "only the template marker may occupy a ray start"
            rr += direction[0]
            cc += direction[1]
            first = False
        assert cells, "every ray must paint at least one background cell"

        ray_tuple = tuple(cells)
        motifs.append(
            {
                "kind": copy_kind,
                "turns": int(variant["turns"]),
                "direction": direction,
                "marker_abs": marker_abs,
                "ray_cells": ray_tuple,
                "skeleton_abs": skeleton_abs,
            }
        )
        ray_cells_list.append(ray_tuple)

    out = _render_from_rays(g, ray_cells_list)
    cover = np.zeros(g.shape, dtype=int)
    exact_overlap = np.zeros(g.shape, dtype=int)
    for ray_cells in ray_cells_list:
        for r, c in ray_cells:
            exact_overlap[r, c] += 1
        for r, c in _dilate8(ray_cells, g.shape):
            cover[r, c] += 1

    return {
        "output": out,
        "motifs": motifs,
        "ray_cells": ray_cells_list,
        "collision_mask": cover >= 2,
        "exact_overlap_mask": exact_overlap >= 2,
        "copy_rotations": tuple(copy_rotations),
        "template_id": template_id,
        "virtual_markers": frozenset(all_virtual_markers),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recover each copy's missing marker by rotation-aligning its grey skeleton to the red-marked template, then cast rays and paint their collision halos.

    Input structure:
    - The background is black.
    - One 8-connected template sprite contains a grey skeleton plus one red marker pixel that is 8-neighbor-adjacent to the skeleton.
    - Three or four other 8-connected sprites contain only the same grey skeleton, each rotated by 0/90/180/270 degrees.

    Rule:
    - Identify the template as the only component containing the red marker.
    - For every grey-only copy, find the unique rotation that makes the template skeleton coincide with that copy's skeleton, and transfer the template marker to the correspondingly rotated empty position.
    - Convert each real or virtual marker into a cardinal ray direction by comparing the marker position to its skeleton centroid and taking the nearest axis.
    - Paint yellow along every ray from the marker position to the grid edge, skipping already non-black start cells, then 8-dilate each yellow ray by one pixel and recolor every background-or-yellow cell covered by at least two dilated rays to cyan.

    Key insight:
    - The copies do not show their markers explicitly, so the ray starts and directions are hidden in the skeleton rotation; the cyan result appears only after all inferred rays are dilated and intersected.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _random_skeleton(rng: random.Random) -> frozenset[tuple[int, int]] | None:
    size = rng.randint(*SKELETON_SIZE_RANGE)
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(tuple(frontier)))
    shape = _normalize(cells)
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    if max(rows) + 1 > SKELETON_BBOX_MAX or max(cols) + 1 > SKELETON_BBOX_MAX:
        return None
    if len(set(_shape_rotations(shape))) != 4:
        return None
    return shape


def _pick_marker(
    shape: frozenset[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], dict[tuple[int, int], dict]] | None:
    candidates = set()
    for r, c in shape:
        for nr, nc in neighbors8(r, c):
            if (nr, nc) not in shape:
                candidates.add((nr, nc))
    valid = []
    shape_rows = [r for r, _ in shape]
    shape_cols = [c for _, c in shape]
    for marker in candidates:
        try:
            direction = _direction_from_centroid(shape, marker)
        except AssertionError:
            continue
        if direction == (0, 1) and marker[1] <= max(shape_cols):
            continue
        if direction == (0, -1) and marker[1] >= min(shape_cols):
            continue
        if direction == (1, 0) and marker[0] <= max(shape_rows):
            continue
        if direction == (-1, 0) and marker[0] >= min(shape_rows):
            continue

        by_dir: dict[tuple[int, int], dict] = {}
        ok = True
        for turns, (shape_rot, marker_rot) in enumerate(_shape_with_point_rotations(shape, marker)):
            if marker_rot in shape_rot:
                ok = False
                break
            try:
                dir_rot = _direction_from_centroid(shape_rot, marker_rot)
            except AssertionError:
                ok = False
                break
            by_dir[dir_rot] = {
                "turns": turns,
                "shape": shape_rot,
                "marker": marker_rot,
                "extents": _full_extents(shape_rot, marker_rot),
            }
        if ok and len(by_dir) == 4:
            valid.append((marker, by_dir))
    if not valid:
        return None
    return rng.choice(valid)


def _lane_offsets(count: int) -> tuple[int, ...]:
    if count == 2:
        return (-4, 4)
    assert count == 3
    return (-6, 0, 6)


def _min_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(r0 - r1) + abs(c0 - c1) for r0, c0 in a for r1, c1 in b)


def _place_motifs(
    height: int,
    width: int,
    variants: dict[tuple[int, int], dict],
    motif_dirs: list[tuple[int, int]],
    rng: random.Random,
) -> list[dict] | None:
    horiz_count = sum(direction in {(0, -1), (0, 1)} for direction in motif_dirs)
    vert_count = len(motif_dirs) - horiz_count
    row_offsets = _lane_offsets(horiz_count)
    col_offsets = _lane_offsets(vert_count)
    for _ in range(80):
        center_r = rng.randint(8, height - 9)
        center_c = rng.randint(8, width - 9)
        rows = [center_r + off for off in row_offsets]
        cols = [center_c + off for off in col_offsets]
        rng.shuffle(rows)
        rng.shuffle(cols)

        row_iter = iter(rows)
        col_iter = iter(cols)
        placed: list[dict] = []
        occupied_full: list[set[tuple[int, int]]] = []
        failed = False

        for direction in motif_dirs:
            variant = variants[direction]
            min_dr, max_dr, min_dc, max_dc = variant["extents"]
            if direction == (0, 1):
                marker_r = next(row_iter)
                lo = max(-min_dc, 2)
                hi = min(width - 1 - max_dc, 7)
                if not (0 <= marker_r - min_dr and marker_r + max_dr < height and lo <= hi):
                    failed = True
                    break
                marker_c = rng.randint(lo, hi)
            elif direction == (0, -1):
                marker_r = next(row_iter)
                lo = max(-min_dc, width - 8)
                hi = min(width - 1 - max_dc, width - 3)
                if not (0 <= marker_r - min_dr and marker_r + max_dr < height and lo <= hi):
                    failed = True
                    break
                marker_c = rng.randint(lo, hi)
            elif direction == (1, 0):
                marker_c = next(col_iter)
                lo = max(-min_dr, 2)
                hi = min(height - 1 - max_dr, 7)
                if not (0 <= marker_c - min_dc and marker_c + max_dc < width and lo <= hi):
                    failed = True
                    break
                marker_r = rng.randint(lo, hi)
            else:
                marker_c = next(col_iter)
                lo = max(-min_dr, height - 8)
                hi = min(height - 1 - max_dr, height - 3)
                if not (0 <= marker_c - min_dc and marker_c + max_dc < width and lo <= hi):
                    failed = True
                    break
                marker_r = rng.randint(lo, hi)

            r0 = marker_r - variant["marker"][0]
            c0 = marker_c - variant["marker"][1]
            skeleton_abs = {(r0 + r, c0 + c) for r, c in variant["shape"]}
            marker_abs = (marker_r, marker_c)
            full_abs = set(skeleton_abs) | {marker_abs}

            if any(not in_bounds((height, width), r, c) for r, c in full_abs):
                failed = True
                break
            for prev in occupied_full:
                if _min_manhattan(full_abs, prev) < 5:
                    failed = True
                    break
            if failed:
                break

            placed.append(
                {
                    "direction": direction,
                    "turns": variant["turns"],
                    "skeleton_abs": skeleton_abs,
                    "marker_abs": marker_abs,
                    "full_abs": full_abs,
                }
            )
            occupied_full.append(full_abs)

        if failed:
            continue

        all_full = set().union(*occupied_full)
        clear = True
        for motif in placed:
            rr, cc = motif["marker_abs"]
            dr, dc = motif["direction"]
            while in_bounds((height, width), rr, cc):
                if (rr, cc) in all_full and (rr, cc) != motif["marker_abs"]:
                    clear = False
                    break
                rr += dr
                cc += dc
            if not clear:
                break
        if clear:
            return placed
    return None


def _validate_instance(grid: np.ndarray) -> None:
    details = _solve_details(grid)
    out = details["output"]
    exact_out = _render_from_mask(grid, details["ray_cells"], details["exact_overlap_mask"])

    assert np.any(out == COLLISION), "dilated ray intersections must exist"
    assert not np.array_equal(exact_out, out), "dilation must enlarge the direct ray collisions"
    assert details["copy_rotations"], "expected at least one copy"
    assert any(turns != 0 for turns in details["copy_rotations"]), "at least one copy must be rotated"
    assert len({motif["direction"] for motif in details["motifs"]}) >= 4, "the sample must expose all four directions"

    collision_count = int(np.count_nonzero(details["collision_mask"]))
    assert collision_count <= 70, "collision area should stay visually local"

    for idx in range(len(details["ray_cells"])):
        reduced = [ray for j, ray in enumerate(details["ray_cells"]) if j != idx]
        alt = _render_from_rays(grid, reduced)
        assert not np.array_equal(alt, out), "every ray must affect the final result"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)

    shape = _random_skeleton(rng)
    if shape is None:
        return None
    marker_pick = _pick_marker(shape, rng)
    if marker_pick is None:
        return None
    _marker, variants = marker_pick

    copy_count = rng.choice(COPY_COUNT_CHOICES)
    motif_dirs = list(DIRS)
    if copy_count == 4:
        motif_dirs.append(rng.choice(DIRS))
    rng.shuffle(motif_dirs)

    placed = _place_motifs(height, width, variants, motif_dirs, rng)
    if placed is None:
        return None

    template_idx = rng.randrange(len(placed))
    grid = np.full((height, width), BG, dtype=int)
    for idx, motif in enumerate(placed):
        for r, c in motif["skeleton_abs"]:
            grid[r, c] = SKELETON
        if idx == template_idx:
            mr, mc = motif["marker_abs"]
            assert int(grid[mr, mc]) == BG
            grid[mr, mc] = MARKER

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
