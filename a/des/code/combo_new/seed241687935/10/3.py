import itertools
import random
from collections import deque

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GAP_REL = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
REL_TO_DIR = {rel: name for name, rel in GAP_REL.items()}
ALL_3X3 = {(r, c) for r in range(3) for c in range(3)}
DIR_NAMES = tuple(DIRS)
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

GEN_HEIGHT_RANGE = (19, 29)
GEN_WIDTH_RANGE = (19, 29)
GEN_OBJECT_COUNT_RANGE = (3, 5)
GEN_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_CENTER_START = 4
GEN_CENTER_STEP = 5
GEN_MAX_ATTEMPTS = 400


def _components4_same_color(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return out


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    comps = _components4_same_color(grid, bg=BG)
    assert comps, "expected at least one object"

    objects = []
    cell_to_obj = {}

    for idx, comp in enumerate(comps):
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each object must occupy a 3x3 bbox"
        rels = {(int(r - r0), int(c - c0)) for r, c in coords}
        assert len(rels) == 8, "each object must contain exactly eight colored cells"

        missing = ALL_3X3 - rels
        assert len(missing) == 1, "each object must have one missing cell"
        gap_rel = next(iter(missing))
        assert gap_rel in REL_TO_DIR, "the missing cell must be an edge-center pixel"

        direction = REL_TO_DIR[gap_rel]
        gap = (r0 + gap_rel[0], c0 + gap_rel[1])
        assert int(grid[gap]) == BG, "the gap cell must be background"

        obj = {
            "color": int(comp["color"]),
            "coords": coords,
            "bbox": (r0, c0, r1, c1),
            "direction": direction,
            "gap": (int(gap[0]), int(gap[1])),
        }
        objects.append(obj)

        for r, c in map(tuple, coords):
            assert (int(r), int(c)) not in cell_to_obj, "objects must not overlap"
            cell_to_obj[(int(r), int(c))] = idx

    return objects, cell_to_obj


def _cast_ray(
    grid: np.ndarray,
    cell_to_obj: dict[tuple[int, int], int],
    obj: dict,
) -> tuple[list[tuple[int, int]], int | None]:
    h, w = grid.shape
    dr, dc = DIRS[obj["direction"]]
    r, c = obj["gap"]
    path = []

    while 0 <= r < h and 0 <= c < w:
        assert int(grid[r, c]) == BG, "ray path must stay on background cells until impact"
        path.append((int(r), int(c)))

        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return path, None
        if int(grid[nr, nc]) != BG:
            return path, int(cell_to_obj[(int(nr), int(nc))])

        r, c = nr, nc

    raise AssertionError("unreachable")


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, list[int]]]:
    objects, cell_to_obj = _parse_objects(grid)
    rays = []
    incoming: dict[int, list[int]] = {}

    for src_idx, obj in enumerate(objects):
        path, hit = _cast_ray(grid, cell_to_obj, obj)
        rays.append({"path": path, "hit": hit})
        if hit is not None:
            incoming.setdefault(int(hit), []).append(int(src_idx))

    return objects, rays, incoming


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend rays from each hollow 3x3 object and recolor any object that those rays hit.

    Input structure:
    - The background is black.
    - The foreground consists of separated 4-connected single-color objects.
    - Each object exactly fills a 3x3 bbox except for one missing edge-center cell,
      so the shape has an opening pointing up, down, left, or right.

    Rule:
    - Detect each object's opening direction from the location of its missing edge-center cell.
    - Starting at that missing cell, draw a one-pixel-wide ray of the same color in the opening direction.
    - The ray continues across background cells, including the starting gap cell, and stops just before the
      first foreground pixel it would enter or at the grid boundary if no object is encountered.
    - If a ray reaches another object, recolor that entire target object to the shooter's color.
      All hits are decided from the original input, so recoloring never changes other rays.

    Key insight:
    The object's missing edge-center cell acts as a directional nozzle: first infer its orientation from the
    local 3x3 shape, then trace the emitted ray to determine which distant object, if any, inherits its color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects, rays, incoming = _analyze_scene(g)
    out = g.copy()

    for obj, ray in zip(objects, rays):
        path = ray["path"]
        if not path:
            continue
        arr = np.array(path, dtype=int)
        out[arr[:, 0], arr[:, 1]] = int(obj["color"])

    for target_idx, source_indices in incoming.items():
        source_colors = {int(objects[src]["color"]) for src in source_indices}
        assert len(source_colors) == 1, "a target object received conflicting colors"
        color = next(iter(source_colors))
        coords = objects[target_idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def _candidate_centers(size: int) -> list[int]:
    centers = list(range(GEN_CENTER_START, size - 4, GEN_CENTER_STEP))
    assert centers, "grid side is too small for any object center"
    return centers


def _object_cells(center: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r, c = center
    gap_r, gap_c = GAP_REL[direction]
    cells = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = r + dr
            cc = c + dc
            if (dr + 1, dc + 1) == (gap_r, gap_c):
                continue
            cells.append((int(rr), int(cc)))
    return cells


def _compose_grid(
    height: int,
    width: int,
    centers: list[tuple[int, int]],
    directions: tuple[str, ...],
    colors: tuple[int, ...],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for (center, direction, color) in zip(centers, directions, colors):
        for r, c in _object_cells(center, direction):
            assert 0 <= r < height and 0 <= c < width, "object left the grid"
            assert int(grid[r, c]) == BG, "objects must not overlap"
            grid[r, c] = int(color)
    return grid


def _ray_paths_overlap(rays: list[dict]) -> bool:
    seen = set()
    for ray in rays:
        for cell in ray["path"]:
            if cell in seen:
                return True
            seen.add(cell)
    return False


def _incoming_hits_solid_face(rays: list[dict], directions: tuple[str, ...]) -> bool:
    for src_idx, ray in enumerate(rays):
        hit = ray["hit"]
        if hit is None:
            continue
        if directions[hit] == OPPOSITE[directions[src_idx]]:
            return False
    return True


def _sample_centers(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    rows_pool = _candidate_centers(height)
    cols_pool = _candidate_centers(width)

    for _ in range(80):
        row_cap = min(3, len(rows_pool))
        col_cap = min(3, len(cols_pool))
        n_rows = random.randint(2, row_cap)
        n_cols = random.randint(2, col_cap)

        while n_rows * n_cols < count:
            if n_rows < row_cap and (n_rows <= n_cols or n_cols == col_cap):
                n_rows += 1
            elif n_cols < col_cap:
                n_cols += 1
            else:
                break
        if n_rows * n_cols < count:
            continue

        rows = sorted(random.sample(rows_pool, n_rows))
        cols = sorted(random.sample(cols_pool, n_cols))
        points = [(r, c) for r in rows for c in cols]
        centers = sorted(random.sample(points, count))
        if len({r for r, _ in centers}) < count and len({c for _, c in centers}) < count:
            return centers

    return None


def _valid_direction_sets(height: int, width: int, centers: list[tuple[int, int]]) -> list[tuple[str, ...]]:
    count = len(centers)
    colors = tuple(range(1, count + 1))
    valid = []

    for directions in itertools.product(DIR_NAMES, repeat=count):
        grid = _compose_grid(height, width, centers, directions, colors)
        objects, rays, incoming = _analyze_scene(grid)
        hits = [ray["hit"] is not None for ray in rays]

        if not any(hits) or all(hits):
            continue
        if any(len(srcs) > 1 for srcs in incoming.values()):
            continue
        if _ray_paths_overlap(rays):
            continue
        if not _incoming_hits_solid_face(rays, directions):
            continue

        assert len(objects) == count, "generator created an invalid object count"
        valid.append(tuple(directions))

    return valid


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    objects, rays, incoming = _analyze_scene(grid)
    assert GEN_OBJECT_COUNT_RANGE[0] <= len(objects) <= GEN_OBJECT_COUNT_RANGE[1]
    assert len({int(obj["color"]) for obj in objects}) == len(objects), "object colors must be distinct"
    assert any(ray["hit"] is not None for ray in rays), "expected at least one hit branch"
    assert any(ray["hit"] is None for ray in rays), "expected at least one boundary branch"
    assert all(len(srcs) <= 1 for srcs in incoming.values()), "targets must have unique incoming rays"
    assert not _ray_paths_overlap(rays), "rays should not overlap in generated inputs"

    out = main(grid)
    assert out.shape == grid.shape


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)
        count = random.randint(*GEN_OBJECT_COUNT_RANGE)

        centers = _sample_centers(height, width, count)
        if centers is None:
            continue

        direction_sets = _valid_direction_sets(height, width, centers)
        if not direction_sets:
            continue

        directions = random.choice(direction_sets)
        colors = tuple(random.sample(GEN_COLORS, count))
        grid = _compose_grid(height, width, centers, directions, colors)
        _validate_generated(grid)
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 3")
