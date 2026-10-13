from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
OCCLUDER_COLORS = (Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
HALO_OFFSETS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _zone_around_center(shape: tuple[int, int], center: tuple[int, int], radius: int = 2) -> set[tuple[int, int]]:
    r, c = map(int, center)
    zone = set()
    for rr in range(r - radius, r + radius + 1):
        for cc in range(c - radius, c + radius + 1):
            if _in_bounds(shape, rr, cc):
                zone.add((rr, cc))
    return zone


def _zone_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]], margin: int = 1) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    zone = set()
    for rr in range(r0 - margin, r1 + margin + 1):
        for cc in range(c0 - margin, c1 + margin + 1):
            if _in_bounds(shape, rr, cc):
                zone.add((rr, cc))
    return zone


def _connected_offsets(offsets: set[tuple[int, int]]) -> bool:
    if not offsets:
        return False
    patch = np.zeros((3, 3), dtype=int)
    for dr, dc in offsets:
        patch[dr + 1, dc + 1] = BLUE
    comps = find_components(
        patch,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )
    return len(comps) == 1


def _component_candidates(grid: np.ndarray, comp_coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG:
                continue
            if _is_valid_halo_center(grid, comp_coords, r, c):
                candidates.append((r, c))
    return candidates


def _is_valid_halo_center(grid: np.ndarray, comp_coords: set[tuple[int, int]], center_r: int, center_c: int) -> bool:
    for rr, cc in comp_coords:
        if max(abs(int(rr) - int(center_r)), abs(int(cc) - int(center_c))) != 1:
            return False

    for dr, dc in HALO_OFFSETS:
        rr = int(center_r) + dr
        cc = int(center_c) + dc
        if not _in_bounds(grid.shape, rr, cc):
            continue
        value = int(grid[rr, cc])
        if value == BLUE:
            if (rr, cc) not in comp_coords:
                return False
        elif value == BG:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewind each explainable blue halo back into its hidden red center while leaving fake halos unchanged.

    Input structure:
    - A black-background grid contains blue connected shapes plus other non-blue foreground pixels.
    - Some blue shapes are remnants of a clipped 3x3 halo around a black center cell.
    - Missing halo cells are valid only when they fall outside the grid or are occupied by non-blue foreground occluders.

    Rule:
    - For each blue 4-connected component, search for a black center cell whose 3x3 neighborhood explains that component exactly.
    - A candidate center is valid only when every in-bounds halo position is either one of that component's blue pixels or a non-blue foreground occluder.
    - If a component has exactly one valid center, erase the blue component and paint that center red; otherwise keep the blue component unchanged.

    Key insight:
    - The missing blue cells are not arbitrary damage: every gap must be justified locally by an occluder or by boundary clipping, which uniquely identifies true halos and rejects fake ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    blue_components = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )

    for comp in blue_components:
        coords = {tuple(map(int, rc)) for rc in np.asarray(comp["coords"], dtype=int)}
        candidates = _component_candidates(g, coords)
        if len(candidates) != 1:
            continue
        center_r, center_c = candidates[0]
        for rr, cc in coords:
            out[rr, cc] = BG
        out[center_r, center_c] = RED

    return out


def _sample_connected_subset(
    offsets: list[tuple[int, int]],
    *,
    min_size: int,
    max_size: int,
    attempts: int = 80,
) -> set[tuple[int, int]] | None:
    for _ in range(attempts):
        size = random.randint(min_size, max_size)
        subset = set(random.sample(offsets, size))
        if _connected_offsets(subset):
            return subset
    return None


def _candidate_true_centers(shape: tuple[int, int], mode: str) -> list[tuple[int, int]]:
    h, w = shape
    if mode == "occlusion":
        return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]

    border_cells = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                border_cells.append((r, c))
    random.shuffle(border_cells)
    return border_cells


def _build_true_object(shape: tuple[int, int], mode: str) -> dict | None:
    centers = _candidate_true_centers(shape, mode)
    for center in centers:
        r, c = center
        in_bounds_offsets = [(dr, dc) for dr, dc in HALO_OFFSETS if _in_bounds(shape, r + dr, c + dc)]
        out_of_bounds_count = len(HALO_OFFSETS) - len(in_bounds_offsets)

        if mode == "occlusion":
            if out_of_bounds_count != 0 or len(in_bounds_offsets) < 4:
                continue
            visible = _sample_connected_subset(
                in_bounds_offsets,
                min_size=3,
                max_size=len(in_bounds_offsets) - 1,
            )
            if visible is None:
                continue
            occluded = set(in_bounds_offsets) - visible
            if not occluded:
                continue
        elif mode == "edge":
            if out_of_bounds_count == 0:
                continue
            visible = set(in_bounds_offsets)
            occluded = set()
            if len(visible) < 2 or not _connected_offsets(visible):
                continue
        elif mode == "mixed":
            if out_of_bounds_count == 0 or len(in_bounds_offsets) < 3:
                continue
            visible = _sample_connected_subset(
                in_bounds_offsets,
                min_size=2,
                max_size=len(in_bounds_offsets) - 1,
            )
            if visible is None:
                continue
            occluded = set(in_bounds_offsets) - visible
            if not occluded:
                continue
        else:
            raise ValueError(f"unknown mode: {mode}")

        blue = {(r + dr, c + dc) for dr, dc in visible}
        occluders = {
            (r + dr, c + dc): random.choice(OCCLUDER_COLORS)
            for dr, dc in sorted(occluded)
        }
        return {
            "center": center,
            "blue": blue,
            "occluders": occluders,
            "reserve": _zone_around_center(shape, center, radius=2),
            "mode": mode,
            "out_of_bounds_count": out_of_bounds_count,
        }
    return None


def _candidate_fake_center(shape: tuple[int, int], margin: int) -> tuple[int, int] | None:
    h, w = shape
    rows = range(margin, h - margin)
    cols = range(margin, w - margin)
    cells = [(r, c) for r in rows for c in cols]
    if not cells:
        return None
    return random.choice(cells)


def _validate_fake_local(blue: set[tuple[int, int]], occluders: dict[tuple[int, int], int]) -> bool:
    grid = np.zeros((9, 9), dtype=int)
    for rr, cc in blue:
        grid[rr, cc] = BLUE
    for (rr, cc), color in occluders.items():
        grid[rr, cc] = int(color)
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )
    if len(comps) != 1:
        return False
    coords = {tuple(map(int, rc)) for rc in np.asarray(comps[0]["coords"], dtype=int)}
    return len(_component_candidates(grid, coords)) == 0


def _build_gap_fake(shape: tuple[int, int]) -> dict | None:
    for _ in range(120):
        center = _candidate_fake_center(shape, margin=2)
        if center is None:
            return None
        r, c = center
        ring = list(HALO_OFFSETS)
        visible = _sample_connected_subset(ring, min_size=4, max_size=6)
        if visible is None:
            continue
        remaining = list(set(ring) - visible)
        if len(remaining) < 2:
            continue
        gap_count = random.randint(1, min(2, len(remaining) - 1))
        gap_offsets = set(random.sample(remaining, gap_count))
        occluded_offsets = set(remaining) - gap_offsets
        if not occluded_offsets:
            continue

        blue = {(r + dr, c + dc) for dr, dc in visible}
        occluders = {
            (r + dr, c + dc): random.choice(OCCLUDER_COLORS)
            for dr, dc in sorted(occluded_offsets)
        }

        local_blue = {(rr - r + 4, cc - c + 4) for rr, cc in blue}
        local_occ = {(rr - r + 4, cc - c + 4): color for (rr, cc), color in occluders.items()}
        if not _validate_fake_local(local_blue, local_occ):
            continue

        all_cells = set(blue) | set(occluders)
        return {
            "blue": blue,
            "occluders": occluders,
            "reserve": _zone_from_cells(shape, all_cells, margin=1),
            "kind": "gap",
            "center_hint": center,
        }
    return None


def _spur_options(visible: set[tuple[int, int]]) -> list[tuple[int, int]]:
    options = set()
    for dr, dc in visible:
        for sdr, sdc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cand = (dr + sdr, dc + sdc)
            if max(abs(cand[0]), abs(cand[1])) <= 1:
                continue
            if max(abs(cand[0]), abs(cand[1])) > 2:
                continue
            if cand in visible:
                continue
            options.add(cand)
    return sorted(options)


def _build_spur_fake(shape: tuple[int, int]) -> dict | None:
    for _ in range(160):
        center = _candidate_fake_center(shape, margin=3)
        if center is None:
            return None
        r, c = center
        ring = list(HALO_OFFSETS)
        visible = _sample_connected_subset(ring, min_size=4, max_size=6)
        if visible is None:
            continue
        occluded_offsets = set(ring) - visible
        if not occluded_offsets:
            continue
        spur_rel_options = _spur_options(visible)
        if not spur_rel_options:
            continue
        spur_dr, spur_dc = random.choice(spur_rel_options)

        blue = {(r + dr, c + dc) for dr, dc in visible}
        spur = (r + spur_dr, c + spur_dc)
        if not _in_bounds(shape, *spur):
            continue
        blue.add(spur)
        occluders = {
            (r + dr, c + dc): random.choice(OCCLUDER_COLORS)
            for dr, dc in sorted(occluded_offsets)
        }

        local_blue = {(rr - r + 4, cc - c + 4) for rr, cc in blue}
        local_occ = {(rr - r + 4, cc - c + 4): color for (rr, cc), color in occluders.items()}
        if not _validate_fake_local(local_blue, local_occ):
            continue

        all_cells = set(blue) | set(occluders)
        return {
            "blue": blue,
            "occluders": occluders,
            "reserve": _zone_from_cells(shape, all_cells, margin=1),
            "kind": "spur",
            "center_hint": center,
            "spur": spur,
        }
    return None


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    blue: set[tuple[int, int]],
    occluders: dict[tuple[int, int], int],
    reserve_zone: set[tuple[int, int]],
) -> bool:
    if any(reserved[r, c] for r, c in reserve_zone):
        return False
    for rr, cc in blue:
        if grid[rr, cc] != BG:
            return False
    for (rr, cc), _color in occluders.items():
        if grid[rr, cc] != BG:
            return False
    for rr, cc in blue:
        grid[rr, cc] = BLUE
    for (rr, cc), color in occluders.items():
        grid[rr, cc] = int(color)
    for rr, cc in reserve_zone:
        reserved[rr, cc] = True
    return True


def _expected_from_metadata(grid: np.ndarray, true_objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for obj in true_objects:
        for rr, cc in obj["blue"]:
            out[rr, cc] = BG
        cr, cc = obj["center"]
        out[cr, cc] = RED
    return out


def _semantic_validate(grid: np.ndarray, true_objects: list[dict], fake_objects: list[dict]) -> bool:
    if not true_objects or not fake_objects:
        return False
    if not any(obj["mode"] in {"edge", "mixed"} for obj in true_objects):
        return False
    if not any(obj["mode"] in {"occlusion", "mixed"} for obj in true_objects):
        return False

    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )
    if len(comps) != len(true_objects) + len(fake_objects):
        return False

    discovered_true_centers = set()
    zero_candidate_fakes = 0
    for comp in comps:
        coords = {tuple(map(int, rc)) for rc in np.asarray(comp["coords"], dtype=int)}
        candidates = _component_candidates(grid, coords)
        if len(candidates) == 1:
            discovered_true_centers.add(candidates[0])
        elif len(candidates) == 0:
            zero_candidate_fakes += 1
        else:
            return False

    expected_true_centers = {tuple(obj["center"]) for obj in true_objects}
    if discovered_true_centers != expected_true_centers:
        return False
    if zero_candidate_fakes != len(fake_objects):
        return False

    expected = _expected_from_metadata(grid, true_objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    for obj in true_objects:
        blue = obj["blue"]
        center = obj["center"]
        local_candidates = _component_candidates(grid, blue)
        if local_candidates != [center]:
            return False
        if obj["mode"] == "edge" and obj["out_of_bounds_count"] == 0:
            return False
        if obj["mode"] == "occlusion" and obj["out_of_bounds_count"] != 0:
            return False
    return True


def _scatter_extra_occluders(grid: np.ndarray, reserved: np.ndarray) -> None:
    h, w = grid.shape
    extras = random.randint(0, 4)
    free_cells = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BG and not reserved[r, c]]
    random.shuffle(free_cells)
    for r, c in free_cells[:extras]:
        grid[r, c] = random.choice(OCCLUDER_COLORS)


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(11, 18), random.randint(11, 18))
    grid = np.zeros(shape, dtype=int)
    reserved = np.zeros(shape, dtype=bool)

    true_modes = ["occlusion", "edge"]
    if random.random() < 0.55:
        true_modes.append("mixed")
    random.shuffle(true_modes)

    true_objects: list[dict] = []
    for mode in true_modes:
        placed = False
        for _ in range(80):
            obj = _build_true_object(shape, mode)
            if obj is None:
                break
            if _place_object(grid, reserved, obj["blue"], obj["occluders"], obj["reserve"]):
                true_objects.append(obj)
                placed = True
                break
        if not placed:
            return None

    fake_kinds = ["gap"]
    if random.random() < 0.6:
        fake_kinds.append("spur")
    random.shuffle(fake_kinds)

    fake_objects: list[dict] = []
    for kind in fake_kinds:
        placed = False
        for _ in range(100):
            obj = _build_gap_fake(shape) if kind == "gap" else _build_spur_fake(shape)
            if obj is None:
                break
            if _place_object(grid, reserved, obj["blue"], obj["occluders"], obj["reserve"]):
                fake_objects.append(obj)
                placed = True
                break
        if not placed:
            return None

    _scatter_extra_occluders(grid, reserved)
    if not _semantic_validate(grid, true_objects, fake_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid halo rewind input")
