from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

BG = BLACK
OBJECT_COLOR = GREY
REPAIR_COLOR = YELLOW
LEGEND_MARKER = CYAN
LEGEND_COLOR_POOL = (BLUE, RED, GREEN, MAGENTA, ORANGE, BROWN)

KIND_H = "H"
KIND_V = "V"
KIND_HV = "HV"
KINDS = (KIND_H, KIND_V, KIND_HV)

LEGEND_ENTRY_H = 3
LEGEND_ENTRY_W = 5
LEGEND_GAP = 1
LEGEND_PATTERNS = {
    KIND_H: ((1, 2), (1, 3), (1, 4)),
    KIND_V: ((0, 2), (1, 2), (2, 2)),
    KIND_HV: ((0, 3), (1, 2), (1, 3), (1, 4), (2, 3)),
}
LEGEND_PATTERN_TO_KIND = {tuple(sorted(v)): k for k, v in LEGEND_PATTERNS.items()}

GRID_SIDE_RANGE = (20, 26)
OBJECT_COUNT_RANGE = (3, 4)
BASE_SIZE_RANGE = (3, 5)
BASE_AREA_RANGE = (5, 12)
MASK_ATTEMPTS = 50
PERTURB_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 60
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def _has_horizontal_symmetry(mask: np.ndarray) -> bool:
    h, _w = mask.shape
    for axis2 in range(2 * h - 1):
        ok = True
        for r in range(h):
            rr = axis2 - r
            if 0 <= rr < h:
                if not np.array_equal(mask[r], mask[rr]):
                    ok = False
                    break
            elif mask[r].any():
                ok = False
                break
        if ok:
            return True
    return False


def _has_vertical_symmetry(mask: np.ndarray) -> bool:
    _h, w = mask.shape
    for axis2 in range(2 * w - 1):
        ok = True
        for c in range(w):
            cc = axis2 - c
            if 0 <= cc < w:
                if not np.array_equal(mask[:, c], mask[:, cc]):
                    ok = False
                    break
            elif mask[:, c].any():
                ok = False
                break
        if ok:
            return True
    return False


def _symmetry_kind(mask: np.ndarray) -> str | None:
    h_sym = _has_horizontal_symmetry(mask)
    v_sym = _has_vertical_symmetry(mask)
    if h_sym and v_sym:
        return KIND_HV
    if h_sym:
        return KIND_H
    if v_sym:
        return KIND_V
    return None


def _mask_signature(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8, copy=False).tobytes()


def _repair_candidates(observed: np.ndarray) -> dict[str, tuple[int, bytes, np.ndarray] | None]:
    h, w = observed.shape
    all_positions = list(range(h * w))
    best: dict[str, tuple[int, bytes, np.ndarray] | None] = {k: None for k in KINDS}

    def consider(candidate: np.ndarray, dist: int) -> None:
        if candidate.sum() == 0 or not _is_connected(candidate):
            return
        h_sym = _has_horizontal_symmetry(candidate)
        v_sym = _has_vertical_symmetry(candidate)
        if not (h_sym or v_sym):
            return
        sig = _mask_signature(candidate)
        for kind, enabled in (
            (KIND_H, h_sym),
            (KIND_V, v_sym),
            (KIND_HV, h_sym and v_sym),
        ):
            if not enabled:
                continue
            current = best[kind]
            if current is None or dist < current[0] or (dist == current[0] and sig < current[1]):
                best[kind] = (dist, sig, candidate.copy())

    consider(observed, 0)
    for dist in (1, 2):
        for combo in combinations(all_positions, dist):
            candidate = observed.copy()
            for idx in combo:
                rr, cc = divmod(idx, w)
                candidate[rr, cc] = not candidate[rr, cc]
            consider(candidate, dist)
    return best


def _solve_local_object(observed: np.ndarray) -> tuple[str, np.ndarray, int]:
    assert observed.ndim == 2 and observed.dtype == bool
    assert observed.any()
    best = _repair_candidates(observed)
    dist_h = 3 if best[KIND_H] is None else best[KIND_H][0]
    dist_v = 3 if best[KIND_V] is None else best[KIND_V][0]
    assert min(dist_h, dist_v) <= 2

    if dist_h < dist_v:
        return KIND_H, best[KIND_H][2], dist_h
    if dist_v < dist_h:
        return KIND_V, best[KIND_V][2], dist_v

    best_hv = best[KIND_HV]
    assert best_hv is not None and best_hv[0] == dist_h
    return KIND_HV, best_hv[2], dist_h


def _legend_entry_patch(color: int, kind: str) -> np.ndarray:
    patch = np.zeros((LEGEND_ENTRY_H, LEGEND_ENTRY_W), dtype=int)
    patch[:, :2] = color
    for r, c in LEGEND_PATTERNS[kind]:
        patch[r, c] = LEGEND_MARKER
    return patch


def _legend_patch(mapping: dict[str, int], order: list[str]) -> np.ndarray:
    height = len(order) * LEGEND_ENTRY_H + (len(order) - 1) * LEGEND_GAP
    patch = np.zeros((height, LEGEND_ENTRY_W), dtype=int)
    for idx, kind in enumerate(order):
        r0 = idx * (LEGEND_ENTRY_H + LEGEND_GAP)
        patch[r0 : r0 + LEGEND_ENTRY_H, :] = _legend_entry_patch(mapping[kind], kind)
    return patch


def _extract_component_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    patch = np.zeros((int(r1 - r0), int(c1 - c0)), dtype=int)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return patch


def _parse_legend(grid: np.ndarray) -> tuple[dict[str, int], list[np.ndarray]]:
    components = _component_coords(grid != BG)
    legend: dict[str, int] = {}
    objects: list[np.ndarray] = []
    for coords in components:
        colors = set(int(grid[r, c]) for r, c in coords)
        if colors == {OBJECT_COLOR}:
            objects.append(coords)
            continue

        patch = _extract_component_patch(grid, coords)
        present = [int(c) for c in np.unique(patch) if int(c) != BG]
        assert len(present) == 2
        counts = {color: int((patch == color).sum()) for color in present}
        block_color = next(color for color, count in counts.items() if count == 6)
        marker_color = next(color for color in present if color != block_color)
        marker_coords = tuple(sorted(map(tuple, np.argwhere(patch == marker_color))))
        assert marker_coords in LEGEND_PATTERN_TO_KIND
        legend[LEGEND_PATTERN_TO_KIND[marker_coords]] = block_color

    assert set(legend) == set(KINDS)
    assert objects
    return legend, objects


def _apply_object_repair(
    out: np.ndarray,
    coords: np.ndarray,
    repaired_local: np.ndarray,
    observed_local: np.ndarray,
    color: int,
) -> None:
    rr0, cc0 = coords.min(axis=0)
    rr1, cc1 = coords.max(axis=0) + 1
    r0 = max(0, int(rr0) - 1)
    c0 = max(0, int(cc0) - 1)
    r1 = min(out.shape[0], int(rr1) + 1)
    c1 = min(out.shape[1], int(cc1) + 1)

    window = out[r0:r1, c0:c1]
    changed = observed_local | repaired_local
    window[changed] = BG
    window[repaired_local & observed_local] = color
    window[repaired_local & ~observed_local] = REPAIR_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair near-symmetric grey objects, recolor them from the legend, and mark added pixels yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend, objects = _parse_legend(g)
    out = g.copy()

    for coords in objects:
        rr0, cc0 = coords.min(axis=0)
        rr1, cc1 = coords.max(axis=0) + 1
        r0 = max(0, int(rr0) - 1)
        c0 = max(0, int(cc0) - 1)
        r1 = min(g.shape[0], int(rr1) + 1)
        c1 = min(g.shape[1], int(cc1) + 1)

        observed_local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        for r, c in coords:
            observed_local[int(r) - r0, int(c) - c0] = True

        kind, repaired_local, _dist = _solve_local_object(observed_local)
        _apply_object_repair(out, coords, repaired_local, observed_local, legend[kind])

    return out


def _orbit_of(kind: str, h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    cells = {(r, c)}
    if kind in (KIND_H, KIND_HV):
        cells.add((h - 1 - r, c))
    if kind in (KIND_V, KIND_HV):
        cells.add((r, w - 1 - c))
    if kind == KIND_HV:
        cells.add((h - 1 - r, w - 1 - c))
    return tuple(sorted(cells))


def _all_orbits(kind: str, h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = _orbit_of(kind, h, w, r, c)
            orbits.append(orbit)
            seen.update(orbit)
    return orbits


def _orbit_adjacent(orbit: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    for r, c in orbit:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _generate_base_mask(kind: str, rng: random.Random) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        h = rng.randint(*BASE_SIZE_RANGE)
        w = rng.randint(*BASE_SIZE_RANGE)
        orbits = _all_orbits(kind, h, w)
        start_idx = rng.randrange(len(orbits))
        chosen = {start_idx}
        occupied = set(orbits[start_idx])
        target = rng.randint(*BASE_AREA_RANGE)

        for _step in range(len(orbits) * 2):
            frontier = [idx for idx, orbit in enumerate(orbits) if idx not in chosen and _orbit_adjacent(orbit, occupied)]
            area = len(occupied)
            if area >= target and frontier and rng.random() < 0.55:
                break
            if not frontier:
                break
            idx = rng.choice(frontier)
            chosen.add(idx)
            occupied.update(orbits[idx])
            if len(occupied) > BASE_AREA_RANGE[1]:
                break

        if not (BASE_AREA_RANGE[0] <= len(occupied) <= BASE_AREA_RANGE[1]):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in occupied:
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        if _symmetry_kind(mask) != kind:
            continue
        if mask[0].sum() == 0 and mask[-1].sum() == 0:
            continue
        if mask[:, 0].sum() == 0 and mask[:, -1].sum() == 0:
            continue
        return mask
    return None


def _adjacent_empty_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                cells.append((r, c))
    return cells


def _removable_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    coords = np.argwhere(mask)
    if len(coords) <= 1:
        return cells
    for r, c in coords:
        candidate = mask.copy()
        candidate[int(r), int(c)] = False
        if candidate.any() and _is_connected(candidate):
            cells.append((int(r), int(c)))
    return cells


def _perturb_base(base_tight: np.ndarray, kind: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = base_tight.shape
    base = np.zeros((h + 2, w + 2), dtype=bool)
    base[1:-1, 1:-1] = base_tight
    operation_patterns = [
        ("remove",),
        ("add",),
        ("remove", "remove"),
        ("add", "add"),
        ("remove", "add"),
        ("add", "remove"),
    ]

    for _ in range(PERTURB_ATTEMPTS):
        observed = base.copy()
        used: set[tuple[int, int]] = set()
        pattern = rng.choice(operation_patterns)
        valid = True

        for op in pattern:
            if op == "remove":
                choices = [cell for cell in _removable_cells(observed) if cell not in used]
            else:
                choices = [cell for cell in _adjacent_empty_cells(observed) if cell not in used]
            if not choices:
                valid = False
                break
            r, c = rng.choice(choices)
            observed[r, c] = not observed[r, c]
            used.add((r, c))

        if not valid or not _is_connected(observed):
            continue

        try:
            kind_found, repaired, dist = _solve_local_object(observed)
        except AssertionError:
            continue
        if dist != int(np.sum(base ^ observed)):
            continue
        if kind_found != kind:
            continue
        if not np.array_equal(repaired, base):
            continue
        if np.array_equal(observed, base):
            continue
        return base, observed
    return None


def _build_object(kind: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(MASK_ATTEMPTS):
        base = _generate_base_mask(kind, rng)
        if base is None:
            continue
        perturbed = _perturb_base(base, kind, rng)
        if perturbed is not None:
            return perturbed
    return None


def _reserve(mask: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(mask.shape[0], top + h + 1)
    c1 = min(mask.shape[1], left + w + 1)
    mask[r0:r1, c0:c1] = True


def _can_place(mask: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(mask.shape[0], top + h + 1)
    c1 = min(mask.shape[1], left + w + 1)
    return not mask[r0:r1, c0:c1].any()


def _legend_top_left(shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    ph, pw = patch_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - pw
    if corner == "bl":
        return h - ph, 0
    assert corner == "br"
    return h - ph, w - pw


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    kinds = [KIND_H, KIND_V, KIND_HV]
    while len(kinds) < object_count:
        kinds.append(rng.choice(KINDS))
    rng.shuffle(kinds)

    colors = rng.sample(list(LEGEND_COLOR_POOL), 3)
    mapping = {kind: color for kind, color in zip(rng.sample(list(KINDS), 3), colors)}
    order = rng.sample(list(KINDS), 3)
    legend = _legend_patch(mapping, order)

    objects: list[tuple[np.ndarray, np.ndarray]] = []
    for kind in kinds:
        built = _build_object(kind, rng)
        if built is None:
            return None
        objects.append(built)

    grid_h = rng.randint(max(GRID_SIDE_RANGE[0], legend.shape[0] + 5), GRID_SIDE_RANGE[1])
    grid_w = rng.randint(max(GRID_SIDE_RANGE[0], legend.shape[1] + 8), GRID_SIDE_RANGE[1] + 2)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    lr, lc = _legend_top_left(grid.shape, legend.shape, corner)
    grid[lr : lr + legend.shape[0], lc : lc + legend.shape[1]] = legend
    _reserve(reserved, lr, lc, legend.shape[0], legend.shape[1])

    for _base, observed in objects:
        oh, ow = observed.shape
        placed = False
        row_min = 1
        col_min = 1
        row_max = grid_h - oh - 1
        col_max = grid_w - ow - 1
        if row_max < row_min or col_max < col_min:
            return None
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(row_min, row_max)
            left = rng.randint(col_min, col_max)
            if not _can_place(reserved, top, left, oh, ow):
                continue
            block = grid[top : top + oh, left : left + ow]
            if np.any(block[observed] != BG):
                continue
            block[observed] = OBJECT_COLOR
            _reserve(reserved, top, left, oh, ow)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
