"""ARC task: slide differently sized objects toward color-matched frame sides."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = tuple(range(10))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True components as (N, 2) arrays of (row, col)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        queue = deque([(r0, c0)])
        coords: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for dr, dc in _DIRS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _parse(grid: np.ndarray) -> tuple[int, dict[int, tuple[int, int]], list[dict]]:
    """Read the corner background, four colored sides, and interior objects."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    assert len(set(corners)) == 1, "the four frame corners must share the background color"
    bg = corners[0]

    side_specs = [
        (g[0, 1:-1], (-1, 0)),
        (g[-1, 1:-1], (1, 0)),
        (g[1:-1, 0], (0, -1)),
        (g[1:-1, -1], (0, 1)),
    ]
    directions: dict[int, tuple[int, int]] = {}
    for pixels, direction in side_specs:
        color = int(pixels.flat[0])
        assert color != bg and bool(np.all(pixels == color)), "each frame side must be solid"
        assert color not in directions, "the four frame-side colors must be distinct"
        directions[color] = direction

    interior = g[1:-1, 1:-1]
    assert set(map(int, np.unique(interior))) <= ({bg} | set(directions))
    objects: list[dict] = []
    for color, direction in directions.items():
        for local in _components(interior == color):
            coords = local + np.array([1, 1], dtype=int)
            objects.append(
                {
                    "area": int(len(coords)),
                    "color": int(color),
                    "direction": direction,
                    "coords": coords,
                }
            )
    assert objects, "at least one interior object is required"
    areas = [int(obj["area"]) for obj in objects]
    assert len(areas) == len(set(areas)), "interior object areas must be pairwise distinct"
    return bg, directions, objects


def _slide_numpy(grid: np.ndarray, bg: int, objects: list[dict]) -> np.ndarray:
    """Sequential whole-shape motion used by the solver."""
    out = np.asarray(grid, dtype=int).copy()
    occupied = out != int(bg)
    h, w = out.shape

    for obj in sorted(objects, key=lambda item: int(item["area"])):
        coords = np.asarray(obj["coords"], dtype=int).copy()
        color = int(obj["color"])
        dr, dc = map(int, obj["direction"])

        occupied[coords[:, 0], coords[:, 1]] = False
        out[coords[:, 0], coords[:, 1]] = int(bg)
        delta = np.array([dr, dc], dtype=int)
        while True:
            candidate = coords + delta
            inside = bool(
                np.all((0 <= candidate[:, 0]) & (candidate[:, 0] < h))
                and np.all((0 <= candidate[:, 1]) & (candidate[:, 1] < w))
            )
            if not inside or bool(np.any(occupied[candidate[:, 0], candidate[:, 1]])):
                break
            coords = candidate

        assert not bool(np.any(occupied[coords[:, 0], coords[:, 1]]))
        occupied[coords[:, 0], coords[:, 1]] = True
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide interior objects, smallest first, toward their color-matched frame sides.

    Input structure:
    - The one-cell frame has background-colored corners and four solid, pairwise
      different side colors. Interior monochrome 4-connected objects have unique
      pixel areas, and every object color matches exactly one frame side.

    Rule:
    - Sort objects by pixel area from smallest to largest. Move each whole object
      one cell at a time toward its same-colored side until its next step would
      overlap the frame, an unprocessed object, or an already settled object.
      Its current cells are then fixed as obstacles for all later objects.

    Key insight:
    - Area is a hidden time axis: the destination of a later large object can be
      changed by a smaller object that has already slid across its path.
    """
    g = np.asarray(grid, dtype=int)
    bg, _directions, objects = _parse(g)
    return _slide_numpy(g, bg, objects)


# ---------------------------------------------------------------------------
# Generator and a deliberately independent, set-based semantic oracle.
# ---------------------------------------------------------------------------


def _normalized_shape(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    shifted = coords - coords.min(axis=0)
    return frozenset(map(tuple, shifted.tolist()))


def _set_oracle(
    grid: np.ndarray,
    objects: list[dict],
    order: str = "ascending",
) -> tuple[np.ndarray, list[dict]]:
    """Independently simulate with coordinate sets and expose collision witnesses."""
    g = np.asarray(grid, dtype=int)
    bg = int(g[0, 0])
    h, w = g.shape
    occupancy: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if int(g[r, c]) != bg:
                    occupancy[(r, c)] = 0  # frame witness
    for obj in objects:
        area = int(obj["area"])
        for rc in map(tuple, np.asarray(obj["coords"], dtype=int).tolist()):
            occupancy[rc] = area

    reverse = order == "descending"
    ordered = sorted(objects, key=lambda item: int(item["area"]), reverse=reverse)
    final_records: list[dict] = []
    for obj in ordered:
        area = int(obj["area"])
        coords = [tuple(rc) for rc in np.asarray(obj["coords"], dtype=int).tolist()]
        for rc in coords:
            assert occupancy.pop(rc) == area
        dr, dc = map(int, obj["direction"])
        steps = 0
        blockers: set[int] = set()
        while True:
            candidate = [(r + dr, c + dc) for r, c in coords]
            if any(not (0 <= r < h and 0 <= c < w) for r, c in candidate):
                blockers = {0}
                break
            blockers = {occupancy[rc] for rc in candidate if rc in occupancy}
            if blockers:
                break
            coords = candidate
            steps += 1
        for rc in coords:
            assert rc not in occupancy
            occupancy[rc] = area
        final_records.append(
            {
                "area": area,
                "color": int(obj["color"]),
                "coords": np.asarray(coords, dtype=int),
                "steps": steps,
                "blockers": blockers,
            }
        )

    out = np.full_like(g, bg)
    out[0, :] = g[0, :]
    out[-1, :] = g[-1, :]
    out[:, 0] = g[:, 0]
    out[:, -1] = g[:, -1]
    for record in final_records:
        coords = record["coords"]
        out[coords[:, 0], coords[:, 1]] = int(record["color"])
    return out, final_records


def _grow_polyomino(area: int) -> np.ndarray:
    """Create a varied exact-area, 4-connected monochrome sprite mask."""
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in _DIRS4
            if (r + dr, c + dc) not in cells
        }
        cells.add(random.choice(tuple(frontier)))
    arr = np.asarray(list(cells), dtype=int)
    arr -= arr.min(axis=0)
    mask = np.zeros(tuple(arr.max(axis=0) + 1), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _can_place_with_gap(occupied: np.ndarray, coords: np.ndarray) -> bool:
    """Require an empty Chebyshev ring for clear object segmentation."""
    h, w = occupied.shape
    coord_set = set(map(tuple, coords.tolist()))
    for r, c in coord_set:
        if not (2 <= r < h - 2 and 2 <= c < w - 2) or occupied[r, c]:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (rr, cc) not in coord_set and occupied[rr, cc]:
                    return False
    return True


def _place_extra(
    grid: np.ndarray,
    occupied: np.ndarray,
    area: int,
    color: int,
) -> bool:
    """Place one random polyomino without overlap or visual crowding."""
    h, w = grid.shape
    for _ in range(120):
        mask = _grow_polyomino(area)
        mh, mw = mask.shape
        if mh > h - 4 or mw > w - 4:
            continue
        r0 = random.randint(2, h - 2 - mh)
        c0 = random.randint(2, w - 2 - mw)
        local = np.argwhere(mask) + np.array([r0, c0], dtype=int)
        if not _can_place_with_gap(occupied, local):
            continue
        grid[local[:, 0], local[:, 1]] = int(color)
        occupied[local[:, 0], local[:, 1]] = True
        return True
    return False


def _build_attempt() -> np.ndarray | None:
    """Construct a random scene containing a guaranteed order-sensitive crossing."""
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    bg = random.choice(_PALETTE)
    top, bottom, left, right = random.sample([c for c in _PALETTE if c != bg], 4)

    grid = np.full((h, w), bg, dtype=int)
    grid[0, 1:-1] = top
    grid[-1, 1:-1] = bottom
    grid[1:-1, 0] = left
    grid[1:-1, -1] = right

    # Canonical interaction chain, later rotated with the complete frame:
    # area 1 slides left and stops beside area 2; area 2 then slides down and
    # leaves; area 3 slides up and is stopped by the already-settled area 1.
    cross_r = random.randint(3, h - 9)
    stop_c = random.randint(3, w - 8)
    small_c = random.randint(stop_c + 4, w - 4)
    large_r = random.randint(cross_r + 4, h - 5)
    core = [
        (left, [(cross_r, small_c)]),
        (bottom, [(cross_r, stop_c), (cross_r + 1, stop_c)]),
        (top, [(large_r + i, stop_c + 1) for i in range(3)]),
    ]
    occupied = np.zeros((h, w), dtype=bool)
    for color, coords_list in core:
        coords = np.asarray(coords_list, dtype=int)
        if not _can_place_with_gap(occupied, coords):
            return None
        grid[coords[:, 0], coords[:, 1]] = int(color)
        occupied[coords[:, 0], coords[:, 1]] = True

    object_count = random.choice((4, 5, 6))
    extra_colors = [right]
    extra_colors.extend(
        random.choice((top, bottom, left, right)) for _ in range(object_count - 4)
    )
    for area, color in zip(range(4, object_count + 1), extra_colors):
        if not _place_extra(grid, occupied, area, color):
            return None

    # Rotation randomizes all four absolute directions while preserving the
    # visible color-to-side reference system.
    grid = np.rot90(grid, random.randrange(4)).copy()
    return grid if _validate_generated(grid) else None


def _validate_generated(grid: np.ndarray) -> bool:
    """Check stated semantics, witnesses, uniqueness, and non-degeneracy."""
    try:
        bg, _directions, objects = _parse(grid)
        areas = sorted(int(obj["area"]) for obj in objects)
        if areas != list(range(1, len(objects) + 1)) or not 4 <= len(objects) <= 6:
            return False

        solved = main(grid)
        oracle, trace = _set_oracle(grid, objects, order="ascending")
        if not np.array_equal(solved, oracle) or np.array_equal(solved, grid):
            return False

        # The frame is a persistent obstacle and is never repainted.
        if not (
            np.array_equal(solved[0, :], grid[0, :])
            and np.array_equal(solved[-1, :], grid[-1, :])
            and np.array_equal(solved[:, 0], grid[:, 0])
            and np.array_equal(solved[:, -1], grid[:, -1])
        ):
            return False

        # Translation preserves every object's shape, color, and pixel count.
        by_area = {int(obj["area"]): obj for obj in objects}
        for record in trace:
            area = int(record["area"])
            if int(record["steps"]) < 1:
                return False
            if _normalized_shape(record["coords"]) != _normalized_shape(by_area[area]["coords"]):
                return False
        if Counter(map(int, grid.ravel())) != Counter(map(int, solved.ravel())):
            return False

        witnesses = {int(record["area"]): set(record["blockers"]) for record in trace}
        # Area 2 is a necessary initial obstacle for area 1; after area 2 moves
        # away, the settled area 1 is a necessary obstacle for area 3.
        if 2 not in witnesses.get(1, set()) or 1 not in witnesses.get(3, set()):
            return False
        # At least one object supplies the tempting independent-to-frame branch.
        if not any(record["blockers"] == {0} for record in trace):
            return False

        descending, _ = _set_oracle(grid, objects, order="descending")
        if np.array_equal(descending, solved):
            return False
        return max(grid.shape) <= 30 and max(solved.shape) <= 30
    except (AssertionError, IndexError, ValueError):
        return False


def generate_input() -> np.ndarray:
    """Generate a varied, semantically validated input grid no larger than 22x22."""
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid order-sensitive sliding scene")
