from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

try:
    from arc_utils import Color, components_mask, get_bg, neighbors4
except Exception:
    class Color:
        BLACK = 0

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def neighbors4(r: int, c: int):
        yield (r - 1, c)
        yield (r + 1, c)
        yield (r, c - 1)
        yield (r, c + 1)

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        if connectivity != 4:
            raise ValueError("fallback components_mask only supports 4-connectivity")
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        out: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                out.append(np.asarray(coords, dtype=int))
        return out


BG = int(Color.BLACK)
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
LEFT_TURN = {(0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0), (1, 0): (0, 1)}
RIGHT_TURN = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}

GEN_RETRIES = 400
PATH_RETRIES = 300
SEGMENT_RETRIES = 25


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield (r + dr, c + dc)


def _normalize_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in path)
    min_c = min(c for _, c in path)
    return [(r - min_r, c - min_c) for r, c in path]


def _bbox_dims(path: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = [tuple(map(int, rc)) for rc in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _ordered_path(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in coords}
    if len(cells) == 1:
        return [next(iter(cells))]

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints = []
    for cell in cells:
        nbrs = [nbr for nbr in neighbors4(*cell) if nbr in cells]
        adjacency[cell] = nbrs
        if len(nbrs) == 1:
            endpoints.append(cell)
        else:
            assert len(nbrs) == 2, "every polyline pixel must have degree 1 or 2"

    assert len(endpoints) == 2, "expected a simple open polyline"
    start = min(endpoints)
    path = [start]
    prev = None
    current = start

    while True:
        nxt = [nbr for nbr in adjacency[current] if nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "polyline traversal became ambiguous"
        prev, current = current, nxt[0]
        path.append(current)

    assert len(path) == len(cells), "polyline traversal missed pixels"
    return path


def _count_bends(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0

    bends = 0
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(1, len(path) - 1):
        next_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if next_dir != prev_dir:
            bends += 1
        prev_dir = next_dir
    return bends


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects = []
    for coords in components_mask(g != bg, connectivity=4):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochromatic"
        path = _ordered_path(coords)
        objects.append(
            {
                "color": next(iter(colors)),
                "coords": coords,
                "path": path,
                "area": int(coords.shape[0]),
                "bends": _count_bends(path),
            }
        )
    assert objects, "expected at least one polyline object"
    return objects, bg


def _build_ring_output(color_by_bends: dict[int, int], bg: int = BG) -> np.ndarray:
    n = max(color_by_bends)
    size = 2 * n + 1
    center = n
    rr, cc = np.indices((size, size))
    dist = np.maximum(np.abs(rr - center), np.abs(cc - center))
    out = np.full((size, size), int(bg), dtype=int)
    for bends, color in color_by_bends.items():
        out[dist == int(bends)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Annihilate paired bend-count polylines, then encode the surviving bend counts as concentric square rings.

    Input structure:
    - A black background contains several separated, single-color, one-pixel-wide polyline objects.
    - Each object is a simple 4-connected path, so its internal right-angle turns can be counted from the path topology.
    - Bend counts that survive appear exactly once and cover 0..n, while some larger bend counts appear exactly twice.

    Rule:
    - Find every foreground object and trace its path from endpoint to endpoint.
    - Count how many times the path changes between horizontal and vertical motion.
    - Remove bend counts that occur twice; for each remaining unique bend count d, fill the full Chebyshev-distance ring d of a (2n+1)x(2n+1) square with that object's color.

    Key insight:
    - Object size, shape footprint, and position are distractions; only the topological bend count determines whether a shape survives and which output ring it paints.
    """
    objects, bg = _extract_objects(np.asarray(grid, dtype=int))
    groups: dict[int, list[dict]] = {}
    for obj in objects:
        groups.setdefault(int(obj["bends"]), []).append(obj)

    survivors = {bends: group[0] for bends, group in groups.items() if len(group) == 1}
    assert survivors, "expected at least one surviving bend count"

    n = max(survivors)
    assert set(survivors) == set(range(n + 1)), "surviving bend counts must cover 0..n exactly once"
    for bends, group in groups.items():
        if len(group) == 1:
            continue
        assert len(group) == 2, "non-surviving bend counts must occur in pairs"
        assert bends > n, "paired bend counts must lie outside the survivor range"

    return _build_ring_output({bends: obj["color"] for bends, obj in survivors.items()}, bg=bg)


def _turn(direction: tuple[int, int], sign: int) -> tuple[int, int]:
    return LEFT_TURN[direction] if sign < 0 else RIGHT_TURN[direction]


def _sample_segment_length(mode: str, bends: int, seg_index: int, rng: np.random.Generator) -> int:
    if mode == "compact":
        choices = (1, 1, 2)
    elif mode == "stretch":
        choices = (2, 2, 3)
    elif mode == "line":
        choices = (2, 3, 4, 5)
    else:
        choices = (1, 2, 2, 3)

    if bends >= 4 and mode == "stretch" and seg_index in (0, bends):
        choices = (2, 3)
    return int(rng.choice(choices))


def _extend_segment(
    occupied: set[tuple[int, int]],
    current: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    r, c = current
    new_cells = []
    for step in range(length):
        r += dr
        c += dc
        cell = (r, c)
        if cell in occupied:
            return None
        allowed_existing = {current} if step == 0 else set()
        for nbr in neighbors4(r, c):
            if nbr in occupied and nbr not in allowed_existing:
                return None
        new_cells.append(cell)
    return new_cells


def _generate_path(
    bends: int,
    rng: np.random.Generator,
    *,
    mode: str,
    max_dim: int,
    allow_single_pixel: bool = False,
) -> list[tuple[int, int]]:
    if bends == 0:
        if allow_single_pixel and float(rng.random()) < 0.5:
            return [(0, 0)]
        direction = DIRS[int(rng.integers(len(DIRS)))]
        length = _sample_segment_length("line", bends, 0, rng)
        path = [(0, 0)]
        r, c = 0, 0
        for _ in range(length):
            r += direction[0]
            c += direction[1]
            path.append((r, c))
        norm = _normalize_path(path)
        h, w = _bbox_dims(norm)
        if max(h, w) <= max_dim:
            return norm

    for _ in range(PATH_RETRIES):
        direction = DIRS[int(rng.integers(len(DIRS)))]
        occupied = {(0, 0)}
        path = [(0, 0)]
        current = (0, 0)
        ok = True

        for seg_index in range(bends + 1):
            if seg_index > 0:
                direction = _turn(direction, int(rng.choice((-1, 1))))

            segment = None
            for _ in range(SEGMENT_RETRIES):
                length = _sample_segment_length(mode, bends, seg_index, rng)
                candidate = _extend_segment(occupied, current, direction, length)
                if candidate is None:
                    continue
                trial = _normalize_path(path + candidate)
                h, w = _bbox_dims(trial)
                if max(h, w) > max_dim:
                    continue
                segment = candidate
                break

            if segment is None:
                ok = False
                break

            path.extend(segment)
            occupied.update(segment)
            current = path[-1]

        if not ok:
            continue

        norm = _normalize_path(path)
        if _count_bends(norm) != bends:
            continue
        h, w = _bbox_dims(norm)
        if max(h, w) <= max_dim:
            return norm

    raise RuntimeError(f"failed to generate a path with {bends} bends")


def _place_objects(
    objects: list[dict],
    rng: np.random.Generator,
) -> np.ndarray | None:
    count = len(objects)
    layout_options = []
    for rows, cols in ((2, 2), (2, 3), (3, 2), (3, 3)):
        if rows * cols >= count:
            layout_options.append((rows, cols))
    rng.shuffle(layout_options)

    max_dim = max(max(obj["shape"]) for obj in objects)
    for rows, cols in layout_options:
        tile_limit = min(30 // rows, 30 // cols)
        tile_min = max_dim + 2
        if tile_min > tile_limit:
            continue

        tile = int(rng.integers(tile_min, tile_limit + 1))
        canvas = np.full((rows * tile, cols * tile), BG, dtype=int)
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        rng.shuffle(slots)

        for obj, (slot_r, slot_c) in zip(objects, slots):
            h, w = obj["shape"]
            r_lo = slot_r * tile + 1
            c_lo = slot_c * tile + 1
            r_hi = (slot_r + 1) * tile - h - 1
            c_hi = (slot_c + 1) * tile - w - 1
            if r_hi < r_lo or c_hi < c_lo:
                break
            r0 = int(rng.integers(r_lo, r_hi + 1))
            c0 = int(rng.integers(c_lo, c_hi + 1))
            for rr, cc in obj["path"]:
                canvas[r0 + rr, c0 + cc] = int(obj["color"])
        else:
            coords = np.argwhere(canvas != BG)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            top = int(rng.integers(1, min(3, int(r0)) + 1))
            left = int(rng.integers(1, min(3, int(c0)) + 1))
            bottom = int(rng.integers(1, min(3, canvas.shape[0] - 1 - int(r1)) + 1))
            right = int(rng.integers(1, min(3, canvas.shape[1] - 1 - int(c1)) + 1))
            return canvas[int(r0) - top : int(r1) + bottom + 1, int(c0) - left : int(c1) + right + 1].copy()

    return None


def _validate_instance(grid: np.ndarray) -> None:
    objects, bg = _extract_objects(grid)
    assert bg == BG, "generated instances must use black background"
    assert len(set(int(obj["color"]) for obj in objects)) == len(objects), "object colors must be unique"

    h, w = grid.shape
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["coords"]:
            for nr, nc in _neighbors8(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (bg, color):
                    raise AssertionError("objects must be 8-separated from each other")

    groups: dict[int, list[dict]] = {}
    for obj in objects:
        groups.setdefault(int(obj["bends"]), []).append(obj)

    assert all(len(group) in (1, 2) for group in groups.values()), "bend counts must be unique or paired"
    survivor_counts = sorted(bends for bends, group in groups.items() if len(group) == 1)
    duplicate_counts = sorted(bends for bends, group in groups.items() if len(group) == 2)
    assert duplicate_counts, "need at least one annihilating pair"

    n = max(survivor_counts)
    assert survivor_counts == list(range(n + 1)), "survivors must cover 0..n exactly once"
    assert all(bends > n for bends in duplicate_counts), "duplicate bend counts must lie beyond the survivors"

    largest = max(objects, key=lambda obj: obj["area"])
    assert int(largest["bends"]) in duplicate_counts, "largest shape should be annihilated"

    survivor_areas = [int(groups[bends][0]["area"]) for bends in range(n + 1)]
    assert any(
        survivor_areas[i] >= survivor_areas[i + 1] for i in range(len(survivor_areas) - 1)
    ), "survivor areas must not increase strictly with bend count"

    for bends in duplicate_counts:
        group = groups[bends]
        signatures = {_shape_signature(obj["coords"]) for obj in group}
        assert len(signatures) == 2, "paired annihilated shapes should not be identical up to translation"

    expected = _build_ring_output({bends: groups[bends][0]["color"] for bends in survivor_counts}, bg=bg)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must realize the intended ring mapping"


def _make_object_specs(rng: np.random.Generator) -> list[dict]:
    n = int(rng.integers(2, 5))
    available_duplicate_counts = list(range(n + 1, 6))
    pair_count = 1 if len(available_duplicate_counts) == 1 else int(rng.choice((1, 2)))
    duplicate_counts = sorted(rng.choice(available_duplicate_counts, size=pair_count, replace=False).tolist())

    bend_counts = list(range(n + 1)) + [bends for bends in duplicate_counts for _ in range(2)]
    rng.shuffle(bend_counts)
    colors = rng.choice(PALETTE, size=len(bend_counts), replace=False)

    stretch_count = int(rng.integers(1, n + 1))
    compact_count = int(rng.integers(stretch_count + 1, n + 1)) if stretch_count < n else n

    duplicate_index_by_bends = Counter()
    objects = []
    for color, bends in zip(colors, bend_counts):
        duplicate_index_by_bends[int(bends)] += 1
        if bends in duplicate_counts:
            mode = "stretch" if duplicate_index_by_bends[int(bends)] == 1 else "mixed"
        elif bends == 0:
            mode = "line"
        elif bends == stretch_count:
            mode = "stretch"
        elif bends == compact_count:
            mode = "compact"
        else:
            mode = str(rng.choice(("compact", "mixed")))

        path = _generate_path(
            int(bends),
            rng,
            mode=mode,
            max_dim=8,
            allow_single_pixel=(int(bends) == 0),
        )
        objects.append(
            {
                "color": int(color),
                "bends": int(bends),
                "path": path,
                "shape": _bbox_dims(path),
                "area": len(path),
            }
        )

    return objects


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        try:
            objects = _make_object_specs(rng)
            grid = _place_objects(objects, rng)
            if grid is None:
                continue
            _validate_instance(grid)
            return grid
        except AssertionError:
            continue
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid bent-annihilation ARC instance")
