from __future__ import annotations

from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ORIENTATIONS = ("H", "V", "D1", "D2")
AXIAL_SIZES = (3, 5, 7, 9)
DIAGONAL_SIZES = (3, 5, 7)
AREA_MIN = 5
AREA_MAX = 11
GEN_RETRIES = 400
MASK_RETRIES = 120
PLACE_RETRIES = 180
BG_COLOR = 5
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]
SAMPLE_MODES = ("free", "boundary", "blocker", "both")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _anti_diagonal_reflection(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2).T


def _classify_mask(mask: np.ndarray) -> str | None:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    if mask.shape[0] == mask.shape[1]:
        main_diag = bool(np.array_equal(mask, mask.T))
        anti_diag = bool(np.array_equal(mask, _anti_diagonal_reflection(mask)))
    else:
        main_diag = False
        anti_diag = False

    hits = [name for name, ok in (("H", horizontal), ("V", vertical), ("D1", main_diag), ("D2", anti_diag)) if ok]
    if len(hits) != 1:
        return None
    return hits[0]


def _axis_step(kind: str) -> tuple[int, int]:
    if kind == "H":
        return (0, 1)
    if kind == "V":
        return (1, 0)
    if kind == "D1":
        return (1, 1)
    if kind == "D2":
        return (1, -1)
    raise ValueError(f"unknown orientation {kind}")


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = map(int, bbox)
    if (r0 + r1) % 2 != 0 or (c0 + c1) % 2 != 0:
        return None
    return (r0 + r1) // 2, (c0 + c1) // 2


def _cast_ray(
    out: np.ndarray,
    occupied: np.ndarray,
    color: int,
    start: tuple[int, int],
    step: tuple[int, int],
    quota: int,
    source_cells: set[tuple[int, int]] | None = None,
) -> dict:
    h, w = out.shape
    r, c = map(int, start)
    dr, dc = map(int, step)
    painted = 0
    skipped_other = False
    added: list[tuple[int, int]] = []

    while 0 <= r < h and 0 <= c < w and painted < quota:
        if bool(occupied[r, c]):
            if source_cells is not None and (r, c) not in source_cells:
                skipped_other = True
        else:
            out[r, c] = int(color)
            painted += 1
            added.append((r, c))
        r += dr
        c += dc

    return {
        "painted": int(painted),
        "truncated": bool(painted < quota),
        "skipped_other": skipped_other,
        "cells": added,
    }


def _solve(grid: np.ndarray, with_details: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    occupied = g != int(bg)
    details: list[dict] = []

    comps = sorted(_find_components(g, bg), key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    for comp in comps:
        mask = _component_mask(comp)
        kind = _classify_mask(mask)
        assert kind is not None, "each object must have exactly one supported reflection axis"
        center = _center_from_bbox(comp["bbox"])
        assert center is not None, "object center must land on a cell"

        color = int(comp["color"])
        quota = int(comp["area"])
        base_step = _axis_step(kind)
        source_cells = {tuple(map(int, cell)) for cell in comp["coords"]}
        sides = []
        added_cells: list[tuple[int, int]] = []

        for sign in (-1, 1):
            side = _cast_ray(
                out,
                occupied,
                color,
                center,
                (sign * base_step[0], sign * base_step[1]),
                quota,
                source_cells=source_cells,
            )
            sides.append(side)
            added_cells.extend(side["cells"])

        details.append(
            {
                "color": color,
                "bbox": comp["bbox"],
                "kind": kind,
                "area": quota,
                "center": center,
                "cells": source_cells,
                "added_cells": added_cells,
                "sides": sides,
            }
        )

    if with_details:
        return out, details
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each symmetric object along its own reflection axis with same-color rays whose length matches the object's area.

    Input structure:
    - The background is the dominant color.
    - The foreground contains several monochrome 4-connected objects.
    - Every foreground object has exactly one reflection axis among horizontal,
      vertical, main-diagonal, or anti-diagonal.
    - Objects are separate and do not overlap.

    Rule:
    - For each object, identify its unique reflection-axis direction and count
      its pixels to get N.
    - Start from the axis center cell of that object and extend in both axis
      directions, painting up to N background cells per side in the object's
      color.
    - Any original object cell on the ray path is skipped rather than covered,
      so the ray keeps traveling until it has painted N cells or hits the grid
      boundary.
    - Keep every original object unchanged.

    Key insight:
    - The important hidden attribute is not object size alone but the pairing of
      a unique symmetry-axis direction with that object's pixel count; the count
      controls the ray length and the axis controls the ray orientation.
    """
    return _solve(grid, with_details=False)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return False
    start = (int(rows[0]), int(cols[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _orbit_cells(kind: str, h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    if kind == "H":
        cells = {(r, c), (h - 1 - r, c)}
    elif kind == "V":
        cells = {(r, c), (r, w - 1 - c)}
    elif kind == "D1":
        cells = {(r, c), (c, r)}
    elif kind == "D2":
        cells = {(r, c), (h - 1 - c, w - 1 - r)}
    else:
        raise ValueError(kind)
    return tuple(sorted((int(rr), int(cc)) for rr, cc in cells))


def _all_orbits(kind: str, h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            orbit = _orbit_cells(kind, h, w, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            out.append(orbit)
    return out


def _orbit_adjacent(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> bool:
    for r, c in orbit:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                return True
    return False


def _add_orbit(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> None:
    for r, c in orbit:
        mask[r, c] = True


def _generate_mask(kind: str, rng: np.random.Generator) -> np.ndarray:
    for _ in range(MASK_RETRIES):
        if kind in ("H", "V"):
            h = int(rng.choice(AXIAL_SIZES))
            w = int(rng.choice(AXIAL_SIZES))
        else:
            h = w = int(rng.choice(DIAGONAL_SIZES))

        center = (h // 2, w // 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[center] = True
        orbits = _all_orbits(kind, h, w)
        target = int(rng.integers(AREA_MIN, min(AREA_MAX, h * w - 1) + 1))

        frontier = [orbit for orbit in orbits if not any(mask[r, c] for r, c in orbit) and _orbit_adjacent(mask, orbit)]
        frontier = [orbit for orbit in frontier if any((r, c) != center for r, c in orbit)]
        if frontier:
            _add_orbit(mask, frontier[int(rng.integers(len(frontier)))])

        for _step in range(h * w):
            frontier = [orbit for orbit in orbits if not any(mask[r, c] for r, c in orbit) and _orbit_adjacent(mask, orbit)]
            if not frontier:
                break
            if int(mask.sum()) >= target and rng.random() < 0.45:
                break
            orbit = frontier[int(rng.integers(len(frontier)))]
            if int(mask.sum()) + len(orbit) > target + 2 and int(mask.sum()) >= AREA_MIN:
                smaller = [cand for cand in frontier if int(mask.sum()) + len(cand) <= target + 2]
                if smaller:
                    orbit = smaller[int(rng.integers(len(smaller)))]
            _add_orbit(mask, orbit)

        mask = _trim_mask(mask)
        if not _mask_connected(mask):
            continue
        if mask.shape[0] % 2 == 0 or mask.shape[1] % 2 == 0:
            continue
        if not bool(mask[mask.shape[0] // 2, mask.shape[1] // 2]):
            continue
        if not (AREA_MIN <= int(mask.sum()) <= AREA_MAX):
            continue
        if _classify_mask(mask) != kind:
            continue
        return mask

    raise RuntimeError(f"failed to build a unique {kind} mask")


def _sample_colors(count: int, rng: np.random.Generator) -> list[int]:
    colors = [int(c) for c in rng.choice(COLOR_POOL, size=count, replace=False)]
    if count >= 3 and rng.random() < 0.35:
        dst = int(rng.integers(count))
        src = int((dst + rng.integers(1, count)) % count)
        colors[dst] = colors[src]
    return colors


def _can_place(mask: np.ndarray, top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    rows, cols = np.where(mask)
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        r = int(top + rr)
        c = int(left + cc)
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _stamp(grid: np.ndarray, occupied: np.ndarray, obj: dict, top: int, left: int) -> dict:
    mask = obj["mask"]
    color = int(obj["color"])
    rows, cols = np.where(mask)
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        grid[top + rr, left + cc] = color
        occupied[top + rr, left + cc] = True

    center_r = int(top + obj["center_local"][0])
    center_c = int(left + obj["center_local"][1])
    return {
        **obj,
        "top": int(top),
        "left": int(left),
        "center": (center_r, center_c),
    }


def _try_place_with_bounds(
    obj: dict,
    occupied: np.ndarray,
    rng: np.random.Generator,
    row_bounds: tuple[int, int],
    col_bounds: tuple[int, int],
    tries: int = PLACE_RETRIES,
) -> tuple[int, int] | None:
    h, w = obj["mask"].shape
    h_lim, w_lim = occupied.shape
    r_min = max(0, int(row_bounds[0]))
    r_max = min(h_lim - h, int(row_bounds[1]))
    c_min = max(0, int(col_bounds[0]))
    c_max = min(w_lim - w, int(col_bounds[1]))
    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(tries):
        top = int(rng.integers(r_min, r_max + 1))
        left = int(rng.integers(c_min, c_max + 1))
        if _can_place(obj["mask"], top, left, occupied):
            return top, left
    return None


def _place_boundary_object(obj: dict, occupied: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    h_lim, w_lim = occupied.shape
    h, w = obj["mask"].shape
    kind = obj["kind"]
    edge = 2

    if kind == "H":
        bounds = [
            ((0, h_lim - h), (0, edge)),
            ((0, h_lim - h), (max(0, w_lim - w - edge), w_lim - w)),
        ]
    elif kind == "V":
        bounds = [
            ((0, edge), (0, w_lim - w)),
            ((max(0, h_lim - h - edge), h_lim - h), (0, w_lim - w)),
        ]
    elif kind == "D1":
        bounds = [
            ((0, edge), (0, edge)),
            ((max(0, h_lim - h - edge), h_lim - h), (max(0, w_lim - w - edge), w_lim - w)),
        ]
    else:
        bounds = [
            ((0, edge), (max(0, w_lim - w - edge), w_lim - w)),
            ((max(0, h_lim - h - edge), h_lim - h), (0, edge)),
        ]

    order = rng.permutation(len(bounds))
    for idx in order.tolist():
        pos = _try_place_with_bounds(obj, occupied, rng, bounds[idx][0], bounds[idx][1])
        if pos is not None:
            return pos
    return None


def _place_central_object(obj: dict, occupied: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    h_lim, w_lim = occupied.shape
    h, w = obj["mask"].shape
    row_bounds = (max(0, h_lim // 5), max(0, h_lim - h - h_lim // 5))
    col_bounds = (max(0, w_lim // 5), max(0, w_lim - w - w_lim // 5))
    pos = _try_place_with_bounds(obj, occupied, rng, row_bounds, col_bounds)
    if pos is not None:
        return pos
    return _try_place_with_bounds(obj, occupied, rng, (0, h_lim - h), (0, w_lim - w))


def _place_blocker(
    blocker: dict,
    occupied: np.ndarray,
    primary: dict,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    step = _axis_step(primary["kind"])
    center_r, center_c = map(int, primary["center"])
    local_r, local_c = map(int, blocker["center_local"])
    max_dist = int(primary["area"] + max(primary["mask"].shape) + 2)
    distances = list(range(2, max_dist + 1))
    signs = [-1, 1]
    rng.shuffle(distances)
    rng.shuffle(signs)

    for sign in signs:
        for dist in distances:
            target_r = center_r + sign * dist * step[0]
            target_c = center_c + sign * dist * step[1]
            top = int(target_r - local_r)
            left = int(target_c - local_c)
            if _can_place(blocker["mask"], top, left, occupied):
                return top, left
    return None


def _place_random_object(obj: dict, occupied: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    h_lim, w_lim = occupied.shape
    h, w = obj["mask"].shape
    return _try_place_with_bounds(obj, occupied, rng, (0, h_lim - h), (0, w_lim - w))


def _build_objects(rng: np.random.Generator, count: int) -> list[dict]:
    shuffled = list(ORIENTATIONS)
    rng.shuffle(shuffled)
    kinds = shuffled[: min(count, len(shuffled))]
    while len(kinds) < count:
        kinds.append(str(rng.choice(ORIENTATIONS)))
    colors = _sample_colors(count, rng)

    objects = []
    for idx, kind in enumerate(kinds):
        mask = _generate_mask(kind, rng)
        objects.append(
            {
                "kind": kind,
                "mask": mask,
                "area": int(mask.sum()),
                "center_local": (mask.shape[0] // 2, mask.shape[1] // 2),
                "color": int(colors[idx]),
            }
        )
    return objects


def _details_are_clear(details: list[dict]) -> bool:
    claimed: dict[tuple[int, int], int] = {}
    for detail in details:
        color = int(detail["color"])
        for cell in detail["added_cells"]:
            if cell in claimed:
                return False
            claimed[cell] = color
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    mode = str(rng.choice(SAMPLE_MODES))
    need_boundary = mode in {"boundary", "both"}
    need_blocker = mode in {"blocker", "both"}

    count = int(rng.choice((4, 5))) if mode == "both" else int(rng.choice((3, 4, 5)))
    objects = _build_objects(rng, count)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    free_objects = objects[:]

    if need_boundary:
        boundary_obj = free_objects.pop(0)
        pos = _place_boundary_object(boundary_obj, occupied, rng)
        if pos is None:
            return None
        placed.append(_stamp(grid, occupied, boundary_obj, *pos))

    if need_blocker:
        if len(free_objects) < 2:
            return None
        primary_obj = free_objects.pop(0)
        blocker_obj = free_objects.pop(0)
        pos = _place_central_object(primary_obj, occupied, rng)
        if pos is None:
            return None
        placed_primary = _stamp(grid, occupied, primary_obj, *pos)
        placed.append(placed_primary)

        pos = _place_blocker(blocker_obj, occupied, placed_primary, rng)
        if pos is None:
            return None
        placed.append(_stamp(grid, occupied, blocker_obj, *pos))

    for obj in free_objects:
        pos = _place_random_object(obj, occupied, rng)
        if pos is None:
            return None
        placed.append(_stamp(grid, occupied, obj, *pos))

    if len({item["kind"] for item in placed}) < 2:
        return None

    out, details = _solve(grid, with_details=True)
    if not _details_are_clear(details):
        return None

    has_boundary = any(side["truncated"] for detail in details for side in detail["sides"])
    has_blocker = any(side["skipped_other"] for detail in details for side in detail["sides"])
    if need_boundary and not has_boundary:
        return None
    if need_blocker and not has_blocker:
        return None

    total_added = sum(len(detail["added_cells"]) for detail in details)
    if total_added < max(8, 2 * len(details)):
        return None
    if int((out != BG_COLOR).sum()) > int(0.45 * h * w):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-ray sample")
