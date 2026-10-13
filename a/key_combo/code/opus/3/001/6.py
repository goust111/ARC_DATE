from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

import numpy as np


BACKGROUND = 0
MARKER = 3
GRAY = 5
OBJECT_COLORS = (1, 2, 4, 7)
SIZE_CANDIDATES = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
)
GEN_ATTEMPTS = 500


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Target:
    outer_bbox: BBox
    corners: tuple[tuple[int, int], ...]
    inner_top_left: tuple[int, int]
    size: tuple[int, int]


@dataclass(frozen=True)
class Obj:
    bbox: BBox
    patch: np.ndarray
    size: tuple[int, int]
    gray_count: int


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: BBox) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: BBox) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= dst.shape[0] and c0 + w <= dst.shape[1]
    mask = patch != BACKGROUND
    region = dst[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(region[mask] != BACKGROUND), "objects must not overlap in output"
    region[mask] = patch[mask]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _targets(grid: np.ndarray) -> list[Target]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    assert len(coords) == 16, "expected four rectangles with four green corners each"

    candidates = []
    by_first: dict[int, list[dict]] = {}
    for combo in combinations(range(len(coords)), 4):
        pts = [coords[i] for i in combo]
        rows = sorted({r for r, _ in pts})
        cols = sorted({c for _, c in pts})
        if len(rows) != 2 or len(cols) != 2:
            continue
        corners = {
            (rows[0], cols[0]),
            (rows[0], cols[1]),
            (rows[1], cols[0]),
            (rows[1], cols[1]),
        }
        if set(pts) != corners:
            continue
        if rows[1] - rows[0] < 2 or cols[1] - cols[0] < 2:
            continue
        cand = {
            "indices": frozenset(combo),
            "outer_bbox": (rows[0], cols[0], rows[1], cols[1]),
            "corners": tuple(sorted(corners)),
        }
        candidates.append(cand)
        by_first.setdefault(min(combo), []).append(cand)

    solutions: list[list[dict]] = []
    used: set[int] = set()
    chosen: list[dict] = []

    def rec() -> None:
        if len(used) == len(coords):
            if len(chosen) == 4:
                solutions.append(chosen.copy())
            return
        if len(chosen) >= 4 or len(solutions) > 1:
            return
        first = next(i for i in range(len(coords)) if i not in used)
        for cand in by_first.get(first, []):
            indices = cand["indices"]
            if indices & used:
                continue
            used.update(indices)
            chosen.append(cand)
            rec()
            chosen.pop()
            used.difference_update(indices)

    rec()
    assert len(solutions) == 1, "green corners must partition into a unique set of rectangles"

    parsed: list[Target] = []
    seen_sizes: set[tuple[int, int]] = set()
    for cand in solutions[0]:
        r0, c0, r1, c1 = cand["outer_bbox"]
        size = (int(r1 - r0 - 1), int(c1 - c0 - 1))
        assert size not in seen_sizes, "target interior sizes must be unique"
        seen_sizes.add(size)
        parsed.append(
            Target(
                outer_bbox=(r0, c0, r1, c1),
                corners=tuple(cand["corners"]),
                inner_top_left=(r0 + 1, c0 + 1),
                size=size,
            )
        )
    parsed.sort(key=lambda t: t.outer_bbox)
    return parsed


def _objects(grid: np.ndarray) -> list[Obj]:
    mask = (grid != BACKGROUND) & (grid != MARKER)
    comps = _components(mask)
    assert len(comps) == 4, "expected four non-green foreground objects"

    parsed: list[Obj] = []
    seen_sizes: set[tuple[int, int]] = set()
    for coords in comps:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        rel = coords - np.array([r0, c0])
        patch[rel[:, 0], rel[:, 1]] = grid[coords[:, 0], coords[:, 1]]
        size = patch.shape
        assert size not in seen_sizes, "object bbox sizes must be unique"
        seen_sizes.add((int(size[0]), int(size[1])))
        parsed.append(
            Obj(
                bbox=bbox,
                patch=patch,
                size=(int(size[0]), int(size[1])),
                gray_count=int(np.count_nonzero(patch == GRAY)),
            )
        )
    parsed.sort(key=lambda obj: obj.bbox)
    return parsed


def _render_solution(shape: tuple[int, int], targets: list[Target], objects: list[Obj]) -> np.ndarray:
    target_by_size = {target.size: target for target in targets}
    assert len(target_by_size) == 4

    gray_counts = [obj.gray_count for obj in objects]
    least = min(gray_counts)
    assert gray_counts.count(least) == 1 and least == 1
    removed_size = objects[gray_counts.index(least)].size

    out = np.zeros(shape, dtype=int)
    for target in targets:
        if target.size == removed_size:
            continue
        for r, c in target.corners:
            out[r, c] = MARKER

    for obj in objects:
        if obj.size == removed_size:
            continue
        target = target_by_size[obj.size]
        _paste_nonzero(out, obj.patch, target.inner_top_left[0], target.inner_top_left[1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Remove the uniquely least-marked object and move the other size-matched objects into their green corner rectangles.

    Input structure:
    - The background is black, four green corner-marker quartets each define one empty target rectangle,
      and four separate 4-connected foreground objects use distinct colors plus gray marker pixels.
    - Each object's bounding-box height and width matches exactly one target rectangle's interior size.
    - Exactly one object contains the strictly fewest gray pixels, and that minimum count is one.

    Rule:
    - Partition the green pixels into four corner-marker rectangles and convert each one to its interior window.
    - Find the four non-green foreground objects, count their gray pixels, and identify the unique object with the
      smallest gray count.
    - Delete that object and also delete the green target rectangle whose interior size matches its bounding box.
    - For the remaining three objects, erase their original locations and translate each unchanged patch so that its
      bounding-box upper-left corner aligns with the upper-left corner of its matched target interior.

    Key insight:
    - Gray-marker counting only decides which size-matched object-target pair is removed; the surviving objects still
      have to be matched by exact bounding-box dimensions in the puzzle's row-by-column coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    targets = _targets(g)
    objects = _objects(g)
    return _render_solution(g.shape, targets, objects)


def _choose_sizes() -> list[tuple[int, int]]:
    area_groups: dict[int, list[tuple[int, int]]] = {}
    for size in SIZE_CANDIDATES:
        area_groups.setdefault(size[0] * size[1], []).append(size)

    for _ in range(200):
        if random.random() < 0.7:
            repeated = [sizes for sizes in area_groups.values() if len(sizes) >= 2]
            pair = random.sample(random.choice(repeated), 2)
            pool = [size for size in SIZE_CANDIDATES if size not in pair]
            sizes = pair + random.sample(pool, 2)
        else:
            sizes = random.sample(list(SIZE_CANDIDATES), 4)
        if any(h > w for h, w in sizes) and any(h < w for h, w in sizes):
            return sizes
    raise RuntimeError("failed to choose sizes")


def _random_mask(size: tuple[int, int], gray_count: int) -> np.ndarray:
    h, w = size
    area = h * w
    low = max(5, h + w - 2, gray_count + 1)
    high = min(15, area - 1 if area > 5 else area)
    assert low <= high

    for _ in range(300):
        n_cells = random.randint(low, high)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < n_cells:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != n_cells:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != h - 1 or cols.min() != 0 or cols.max() != w - 1:
            continue
        if np.all(mask):
            continue
        return mask
    raise RuntimeError(f"failed to build connected mask for size {size}")


def _build_object(size: tuple[int, int], color: int, gray_count: int) -> Obj:
    mask = _random_mask(size, gray_count)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    gray_cells = set(random.sample(coords, gray_count))
    patch = np.zeros(size, dtype=int)
    for r, c in coords:
        patch[r, c] = GRAY if (r, c) in gray_cells else color
    bbox = (0, 0, size[0] - 1, size[1] - 1)
    return Obj(bbox=bbox, patch=patch, size=size, gray_count=gray_count)


def _candidate_positions(
    occupied: np.ndarray, height: int, width: int, *, border: int = 1, padding: int = 1
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions = []
    for r0 in range(border, h - border - height + 1):
        for c0 in range(border, w - border - width + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + height + padding)
            cc1 = min(w, c0 + width + padding)
            if not np.any(occupied[rr0:rr1, cc0:cc1]):
                positions.append((int(r0), int(c0)))
    random.shuffle(positions)
    return positions


def _place_layout(
    shape: tuple[int, int], sizes: list[tuple[int, int]]
) -> tuple[dict[tuple[int, int], Target], dict[tuple[int, int], tuple[int, int]]] | None:
    occupied = np.zeros(shape, dtype=bool)
    items = []
    for size in sizes:
        h, w = size
        items.append(("target", size, h + 2, w + 2))
        items.append(("object", size, h, w))
    random.shuffle(items)
    items.sort(key=lambda item: item[2] * item[3], reverse=True)

    targets: dict[tuple[int, int], Target] = {}
    objects: dict[tuple[int, int], tuple[int, int]] = {}
    for kind, size, h, w in items:
        candidates = _candidate_positions(occupied, h, w)
        if not candidates:
            return None
        r0, c0 = candidates[0]
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        if kind == "target":
            corners = (
                (r0, c0),
                (r0, c0 + w - 1),
                (r0 + h - 1, c0),
                (r0 + h - 1, c0 + w - 1),
            )
            targets[size] = Target(
                outer_bbox=(r0, c0, r0 + h - 1, c0 + w - 1),
                corners=corners,
                inner_top_left=(r0 + 1, c0 + 1),
                size=size,
            )
        else:
            objects[size] = (r0, c0)
    return targets, objects


def _build_input(
    shape: tuple[int, int],
    targets: dict[tuple[int, int], Target],
    object_origins: dict[tuple[int, int], tuple[int, int]],
    objects: dict[tuple[int, int], Obj],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for target in targets.values():
        for r, c in target.corners:
            grid[r, c] = MARKER
    for size, origin in object_origins.items():
        obj = objects[size]
        _paste_nonzero(grid, obj.patch, origin[0], origin[1])
    return grid


def _expected_from_metadata(
    shape: tuple[int, int], targets: dict[tuple[int, int], Target], objects: dict[tuple[int, int], Obj]
) -> np.ndarray:
    return _render_solution(shape, list(targets.values()), list(objects.values()))


def _is_valid_instance(
    grid: np.ndarray,
    targets: dict[tuple[int, int], Target],
    object_origins: dict[tuple[int, int], tuple[int, int]],
    objects: dict[tuple[int, int], Obj],
) -> bool:
    gray_counts = sorted(obj.gray_count for obj in objects.values())
    if gray_counts != [1, 2, 3, 4]:
        return False
    if len(targets) != 4 or len(object_origins) != 4 or len(objects) != 4:
        return False
    expected = _expected_from_metadata(grid.shape, targets, objects)
    try:
        parsed_targets = _targets(grid)
        parsed_objects = _objects(grid)
        solved = main(grid)
    except AssertionError:
        return False
    if {target.size for target in parsed_targets} != set(targets):
        return False
    if {obj.size for obj in parsed_objects} != set(objects):
        return False
    if not np.array_equal(solved, expected):
        return False

    removed_size = min(objects.values(), key=lambda obj: obj.gray_count).size
    target = targets[removed_size]
    if any(int(expected[r, c]) != BACKGROUND for r, c in target.corners):
        return False
    for size, origin in object_origins.items():
        if size == removed_size:
            continue
        if origin == targets[size].inner_top_left:
            return False
    return True


def _attempt() -> np.ndarray | None:
    sizes = _choose_sizes()
    gray_counts = [1, 2, 3, 4]
    random.shuffle(gray_counts)
    colors = list(OBJECT_COLORS)
    random.shuffle(colors)
    objects = {
        size: _build_object(size, color, gray_count)
        for size, color, gray_count in zip(sizes, colors, gray_counts)
    }

    for _ in range(50):
        shape = (random.randint(22, 29), random.randint(22, 29))
        layout = _place_layout(shape, sizes)
        if layout is None:
            continue
        targets, origins = layout
        grid = _build_input(shape, targets, origins, objects)
        if _is_valid_instance(grid, targets, origins, objects):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
