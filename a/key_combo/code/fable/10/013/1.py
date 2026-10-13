from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
SPIRAL = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 1),
)
GEN_ATTEMPTS = 400
PATCH_ATTEMPTS = 200
REMOVAL_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                coords.add((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _rotate_step(r2: int, c2: int, cr2: int, cc2: int) -> tuple[int, int]:
    return cr2 - (c2 - cc2), cc2 + (r2 - cr2)


def _analyze_component(grid: np.ndarray, comp: dict[str, object]) -> dict[str, object]:
    h, w = grid.shape
    color = int(comp["color"])
    visible = set(comp["coords"])
    best_missing: int | None = None
    best: list[dict[str, object]] = []

    for cr2 in range(0, 2 * h - 1):
        for cc2 in range(0, 2 * w - 1):
            if (cr2 - cc2) % 2 != 0:
                continue
            full_coords: set[tuple[int, int]] = set()
            valid = True
            for r, c in visible:
                r2, c2 = 2 * r, 2 * c
                for _ in range(4):
                    if (r2 | c2) & 1:
                        valid = False
                        break
                    rr, cc = r2 // 2, c2 // 2
                    if not (0 <= rr < h and 0 <= cc < w):
                        valid = False
                        break
                    cell = int(grid[rr, cc])
                    if cell not in (BG, color):
                        valid = False
                        break
                    full_coords.add((rr, cc))
                    r2, c2 = _rotate_step(r2, c2, cr2, cc2)
                if not valid:
                    break
            if not valid or not _is_connected(full_coords):
                continue
            missing = len(full_coords) - len(visible)
            candidate = {"center2": (cr2, cc2), "full_coords": full_coords, "missing": missing}
            if best_missing is None or missing < best_missing:
                best_missing = missing
                best = [candidate]
            elif missing == best_missing:
                best.append(candidate)

    assert best_missing is not None, "component must match some rotational completion"
    assert len(best) == 1, "rotational center must be unique"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely damaged rotationally symmetric object and encode its loss as a 3x3 spiral gauge.

    Input structure:
    - The background is black, and exactly two non-touching 4-connected monochrome objects appear in different colors.
    - One object is already fully 90-degree rotationally symmetric.
    - The other object is a visible subset of a 90-degree rotationally symmetric object, with one to eight black missing cells.

    Rule:
    - Analyze each object under 90-degree rotation, using the unique center that yields the smallest valid completion.
    - The intact object must need zero added cells; the damaged object must need between one and eight added cells.
    - Output a 3x3 grid filled with the intact object's color, then recolor the first N cells of the clockwise spiral from the top-left with the damaged object's color, where N is the damaged object's missing-cell count.

    Key insight:
    - The rotational completion is only an intermediate measurement: the answer is a compact gauge whose fill amount equals the symmetry loss, and whose two colors come from the broken and intact source objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert int(np.min(g)) == 0, "background must be black"

    comps = _find_components(g)
    assert len(comps) == 2, "expected exactly two objects"
    assert len({int(comp["color"]) for comp in comps}) == 2, "objects must have different colors"

    analyses = []
    for comp in comps:
        analyses.append({**comp, **_analyze_component(g, comp)})

    intact = [a for a in analyses if int(a["missing"]) == 0]
    damaged = [a for a in analyses if 1 <= int(a["missing"]) <= 8]
    assert len(intact) == 1, "exactly one object must already be rotationally symmetric"
    assert len(damaged) == 1, "exactly one object must be damaged"

    intact_color = int(intact[0]["color"])
    damaged_color = int(damaged[0]["color"])
    missing = int(damaged[0]["missing"])

    out = np.full((3, 3), intact_color, dtype=int)
    for r, c in SPIRAL[:missing]:
        out[r, c] = damaged_color
    return out


def _make_orbits(side: int) -> list[tuple[tuple[int, int], ...]]:
    center2 = side - 1
    remaining = {(r, c) for r in range(side) for c in range(side)}
    orbits: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        start = next(iter(remaining))
        orbit = set()
        r2, c2 = 2 * start[0], 2 * start[1]
        for _ in range(4):
            orbit.add((r2 // 2, c2 // 2))
            r2, c2 = _rotate_step(r2, c2, center2, center2)
        frozen = tuple(sorted(orbit))
        orbits.append(frozen)
        remaining.difference_update(orbit)
    return orbits


def _build_symmetric_patch(min_area: int, max_area: int) -> np.ndarray | None:
    side = random.randint(5, 8)
    orbits = _make_orbits(side)
    orbit_index = {cell: idx for idx, orbit in enumerate(orbits) for cell in orbit}
    adjacency = [set() for _ in orbits]
    for idx, orbit in enumerate(orbits):
        for r, c in orbit:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                other = orbit_index.get((nr, nc))
                if other is not None and other != idx:
                    adjacency[idx].add(other)

    center2 = side - 1
    ranked = sorted(
        range(len(orbits)),
        key=lambda idx: min(
            max(abs(2 * r - center2), abs(2 * c - center2)) for r, c in orbits[idx]
        ),
    )
    start_choices = ranked[: min(4, len(ranked))]
    chosen = {random.choice(start_choices)}
    target_area = random.randint(min_area, min(max_area, side * side - 1))

    while True:
        area = sum(len(orbits[idx]) for idx in chosen)
        frontier = {n for idx in chosen for n in adjacency[idx] if n not in chosen}
        need_more = area < min_area or len(chosen) < 3 or (area < target_area and frontier and random.random() < 0.8)
        if not need_more:
            break
        options = [idx for idx in frontier if area + len(orbits[idx]) <= max_area]
        if not options:
            break
        options.sort(
            key=lambda idx: (
                len(orbits[idx]),
                min(max(abs(2 * r - center2), abs(2 * c - center2)) for r, c in orbits[idx]),
            )
        )
        pool = options[: min(len(options), 6)]
        chosen.add(random.choice(pool))

    mask = np.zeros((side, side), dtype=int)
    for idx in chosen:
        for r, c in orbits[idx]:
            mask[r, c] = 1
    mask = _crop_mask(mask)
    coords = set(map(tuple, np.argwhere(mask == 1)))
    if len(coords) < min_area or len(coords) > max_area or not _is_connected(coords):
        return None

    comp = {"color": 1, "coords": coords, "bbox": _bbox(coords)}
    analysis = _analyze_component(mask, comp)
    if int(analysis["missing"]) != 0 or set(analysis["full_coords"]) != coords:
        return None
    return mask


def _remove_pixels(mask: np.ndarray, missing: int) -> np.ndarray | None:
    full_coords = [tuple(x) for x in np.argwhere(mask == 1)]
    full_set = set(full_coords)
    if len(full_coords) - missing < 4:
        return None
    for _ in range(REMOVAL_ATTEMPTS):
        removed = set(random.sample(full_coords, missing))
        visible = mask.copy()
        for r, c in removed:
            visible[r, c] = 0
        visible_coords = set(map(tuple, np.argwhere(visible == 1)))
        if not _is_connected(visible_coords):
            continue
        comp = {"color": 1, "coords": visible_coords, "bbox": _bbox(visible_coords)}
        try:
            analysis = _analyze_component(visible, comp)
        except AssertionError:
            continue
        if int(analysis["missing"]) != missing:
            continue
        if set(analysis["full_coords"]) != full_set:
            continue
        return visible
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> set[tuple[int, int]]:
    placed = set()
    for r, c in np.argwhere(mask > 0):
        rr, cc = top + int(r), left + int(c)
        grid[rr, cc] = color
        placed.add((rr, cc))
    return placed


def _can_place(existing: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    gh, gw = existing.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    for r, c in np.argwhere(mask > 0):
        rr, cc = top + int(r), left + int(c)
        if int(existing[rr, cc]) != BG:
            return False
        for dr, dc in DIR8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < gh and 0 <= nc < gw and int(existing[nr, nc]) != BG:
                return False
    return True


def _translate(coords: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(r + top, c + left) for r, c in coords}


def _expected_output(intact_color: int, damaged_color: int, missing: int) -> np.ndarray:
    out = np.full((3, 3), intact_color, dtype=int)
    for r, c in SPIRAL[:missing]:
        out[r, c] = damaged_color
    return out


def _gen_attempt() -> np.ndarray | None:
    damaged_missing = random.randint(1, 8)
    damaged_full = None
    for _ in range(PATCH_ATTEMPTS):
        damaged_full = _build_symmetric_patch(min_area=damaged_missing + 5, max_area=20)
        if damaged_full is not None:
            break
    if damaged_full is None:
        return None

    damaged_visible = _remove_pixels(damaged_full, damaged_missing)
    if damaged_visible is None:
        return None

    intact_full = None
    for _ in range(PATCH_ATTEMPTS):
        intact_full = _build_symmetric_patch(min_area=9, max_area=20)
        if intact_full is not None:
            break
    if intact_full is None:
        return None

    colors = random.sample(range(1, 10), 2)
    damaged_color, intact_color = colors[0], colors[1]

    shapes = [
        {"visible": damaged_visible, "full": damaged_full, "color": damaged_color, "missing": damaged_missing},
        {"visible": intact_full, "full": intact_full, "color": intact_color, "missing": 0},
    ]
    random.shuffle(shapes)

    max_h = max(shape["visible"].shape[0] for shape in shapes)
    max_w = max(shape["visible"].shape[1] for shape in shapes)
    grid_h = random.randint(max(14, max_h + 4), 26)
    grid_w = random.randint(max(14, max_w + 4), 26)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    placements = []
    for shape in shapes:
        placed = False
        sh, sw = shape["visible"].shape
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, grid_h - sh - 1)
            left = random.randint(1, grid_w - sw - 1)
            if not _can_place(grid, shape["visible"], top, left):
                continue
            visible_coords = _paint_mask(grid, shape["visible"], top, left, shape["color"])
            full_coords = _translate(set(map(tuple, np.argwhere(shape["full"] == 1))), top, left)
            placements.append(
                {
                    "color": shape["color"],
                    "missing": shape["missing"],
                    "visible_coords": visible_coords,
                    "full_coords": full_coords,
                }
            )
            placed = True
            break
        if not placed:
            return None

    try:
        comps = _find_components(grid)
        if len(comps) != 2:
            return None
        analyses = [{**comp, **_analyze_component(grid, comp)} for comp in comps]
    except AssertionError:
        return None

    for placement in placements:
        matches = [a for a in analyses if int(a["color"]) == int(placement["color"])]
        if len(matches) != 1:
            return None
        match = matches[0]
        if int(match["missing"]) != int(placement["missing"]):
            return None
        if set(match["coords"]) != placement["visible_coords"]:
            return None
        if set(match["full_coords"]) != placement["full_coords"]:
            return None

    damaged = [p for p in placements if int(p["missing"]) > 0]
    intact = [p for p in placements if int(p["missing"]) == 0]
    if len(damaged) != 1 or len(intact) != 1:
        return None

    expected = _expected_output(
        intact_color=int(intact[0]["color"]),
        damaged_color=int(damaged[0]["color"]),
        missing=int(damaged[0]["missing"]),
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-loss gauge input")
