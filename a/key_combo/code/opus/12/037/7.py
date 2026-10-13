from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


BG = 0
FG = 5
HOLE_COLOR = 8
GROUP_COLORS = (1, 2, 3, 4, 7)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

SLOT_SIZE = 5
SLOT_GAP = 2
SLOT_STEP = SLOT_SIZE + SLOT_GAP
MAX_ATTEMPTS = 300


@dataclass(frozen=True)
class ObjectSpec:
    family: str
    has_hole: bool
    cells: tuple[tuple[int, int], ...]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(cells), dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_in_bbox(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    coords = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    return mask | (~mask & ~exterior)


def _has_hole(mask: np.ndarray) -> bool:
    filled = _fill_holes(mask)
    return bool(np.any(filled & ~mask))


def _top_left(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, _, _ = _bbox(coords)
    return r0, c0


def _symmetry(coords: tuple[tuple[int, int], ...], idx: int) -> tuple[tuple[int, int], ...]:
    pts = np.array(coords, dtype=int)
    r = pts[:, 0]
    c = pts[:, 1]
    if idx == 0:
        rr, cc = r, c
    elif idx == 1:
        rr, cc = c, -r
    elif idx == 2:
        rr, cc = -r, -c
    elif idx == 3:
        rr, cc = -c, r
    elif idx == 4:
        rr, cc = r, -c
    elif idx == 5:
        rr, cc = c, r
    elif idx == 6:
        rr, cc = -r, c
    else:
        rr, cc = -c, -r
    transformed = np.stack([rr, cc], axis=1)
    transformed -= transformed.min(axis=0)
    return tuple(sorted((int(a), int(b)) for a, b in transformed))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor hole-free objects by normalized shape, but short-circuit any object with an enclosed hole to cyan.

    Input structure:
    - A black background contains several separated gray 4-connected foreground objects.
    - Some objects enclose one black hole inside their own bbox; other objects are hole-free.
    - Repeated hole-free objects may share the same translation-normalized mask at different positions.

    Rule:
    - Find every 4-connected foreground object.
    - If an object's mask encloses at least one 4-connected background hole, recolor that object's foreground pixels to cyan and keep the hole black.
    - Otherwise group the object by its exact translation-normalized mask.
    - Order the hole-free groups by the row-major top-left corner of each group's earliest object, then recolor the groups with blue, red, green, yellow, and orange.

    Key insight:
    - Topology is checked before geometry: a hole-containing object never joins the later shape-color grouping, even if its outer silhouette resembles a hole-free object.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _component_coords(g != bg)

    out = g.copy()
    hole_free_groups: dict[tuple[tuple[int, int], ...], list[np.ndarray]] = defaultdict(list)

    for coords in comps:
        mask = _mask_in_bbox(coords)
        if _has_hole(mask):
            out[coords[:, 0], coords[:, 1]] = HOLE_COLOR
        else:
            hole_free_groups[_normalize_mask(mask)].append(coords)

    ordered = sorted(
        hole_free_groups.items(),
        key=lambda item: min(_top_left(coords) for coords in item[1]),
    )
    assert len(ordered) <= len(GROUP_COLORS)
    for color, (_sig, members) in zip(GROUP_COLORS, ordered):
        for coords in members:
            out[coords[:, 0], coords[:, 1]] = color
    return out


BASE_FAMILIES = {
    "square3": {
        "nohole": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        "hole": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
    },
    "rect34": {
        "nohole": (
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
        ),
        "hole": (
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
        ),
    },
    "plus5": {
        "nohole": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    },
    "hook6": {
        "nohole": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
    },
    "stair6": {
        "nohole": ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    },
}


def _family_objects(rng: np.random.Generator) -> tuple[list[ObjectSpec], int]:
    nohole_group_count = int(rng.integers(3, 5))
    extras = ["plus5", "hook6", "stair6"]
    chosen_extras = rng.choice(extras, size=nohole_group_count - 2, replace=False).tolist()
    active_families = ["square3", "rect34", *chosen_extras]

    min_total = max(8, 2 * nohole_group_count)
    max_total = min(12, 3 * nohole_group_count)
    total_objects = int(rng.integers(min_total, max_total + 1))
    counts = {family: 2 for family in active_families}
    for family in rng.choice(active_families, size=total_objects - 2 * nohole_group_count, replace=False):
        counts[str(family)] += 1

    rect34_has_hole = bool(rng.integers(0, 2))
    sym = {
        "square3": 0,
        "rect34": int(rng.choice([0, 1])),
    }
    for family in chosen_extras:
        sym[family] = int(rng.integers(0, 8))

    specs: list[ObjectSpec] = []
    for family in active_families:
        hole_count = 1 if family == "square3" else 0
        if family == "rect34" and rect34_has_hole:
            hole_count = 1
        nohole_count = counts[family] - hole_count
        assert nohole_count >= 1

        nohole_cells = _symmetry(BASE_FAMILIES[family]["nohole"], sym[family])
        for _ in range(nohole_count):
            specs.append(ObjectSpec(family=family, has_hole=False, cells=nohole_cells))
        if hole_count:
            hole_cells = _symmetry(BASE_FAMILIES[family]["hole"], sym[family])
            specs.append(ObjectSpec(family=family, has_hole=True, cells=hole_cells))

    rng.shuffle(specs)
    return specs, nohole_group_count


def _layout_shape(total_objects: int, rng: np.random.Generator) -> tuple[int, int]:
    candidates = [
        (rows, cols)
        for rows in (3, 4)
        for cols in (3, 4)
        if rows * cols >= total_objects
    ]
    return candidates[int(rng.integers(len(candidates)))]


def _place_objects(specs: list[ObjectSpec], rng: np.random.Generator):
    rowslots, colslots = _layout_shape(len(specs), rng)
    top = int(rng.integers(0, 3))
    left = int(rng.integers(0, 3))
    bottom = int(rng.integers(0, 3))
    right = int(rng.integers(0, 3))
    h = top + rowslots * SLOT_SIZE + (rowslots - 1) * SLOT_GAP + bottom
    w = left + colslots * SLOT_SIZE + (colslots - 1) * SLOT_GAP + right
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    slots = [(sr, sc) for sr in range(rowslots) for sc in range(colslots)]
    rng.shuffle(slots)

    placed = []
    for spec, (sr, sc) in zip(specs, slots):
        local = np.array(spec.cells, dtype=int)
        box_h = int(local[:, 0].max()) + 1
        box_w = int(local[:, 1].max()) + 1
        r0 = top + sr * SLOT_STEP + int(rng.integers(0, SLOT_SIZE - box_h + 1))
        c0 = left + sc * SLOT_STEP + int(rng.integers(0, SLOT_SIZE - box_w + 1))
        coords = local + np.array([r0, c0], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = FG
        placed.append(
            {
                "family": spec.family,
                "has_hole": spec.has_hole,
                "coords": coords,
            }
        )
    return grid, placed


def _expected_output(grid: np.ndarray, placed) -> np.ndarray:
    out = grid.copy()
    hole_free = defaultdict(list)
    for obj in placed:
        coords = obj["coords"]
        mask = _mask_in_bbox(coords)
        if obj["has_hole"]:
            out[coords[:, 0], coords[:, 1]] = HOLE_COLOR
        else:
            hole_free[_normalize_mask(mask)].append(coords)

    ordered = sorted(
        hole_free.items(),
        key=lambda item: min(_top_left(coords) for coords in item[1]),
    )
    for color, (_sig, members) in zip(GROUP_COLORS, ordered):
        for coords in members:
            out[coords[:, 0], coords[:, 1]] = color
    return out


def _filled_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_mask(_fill_holes(_mask_in_bbox(coords)))


def _outer_family_baseline(grid: np.ndarray) -> np.ndarray:
    comps = _component_coords(grid == FG)
    out = grid.copy()
    groups = defaultdict(list)
    for coords in comps:
        groups[_filled_signature(coords)].append(coords)
    ordered = sorted(
        groups.items(),
        key=lambda item: min(_top_left(coords) for coords in item[1]),
    )
    for color, (_sig, members) in zip(GROUP_COLORS, ordered):
        for coords in members:
            out[coords[:, 0], coords[:, 1]] = color
    return out


def _topology_only_baseline(grid: np.ndarray) -> np.ndarray:
    comps = _component_coords(grid == FG)
    out = grid.copy()
    for coords in comps:
        mask = _mask_in_bbox(coords)
        color = HOLE_COLOR if _has_hole(mask) else GROUP_COLORS[0]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _validate_scene(grid: np.ndarray, placed, expected: np.ndarray, nohole_group_count: int) -> bool:
    comps = _component_coords(grid == FG)
    if len(comps) != len(placed):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if nohole_group_count < 3 or nohole_group_count > 4:
        return False

    filled_groups = defaultdict(set)
    exact_holefree = set()
    hole_objects = 0
    for obj in placed:
        coords = obj["coords"]
        filled_groups[_filled_signature(coords)].add(bool(obj["has_hole"]))
        if obj["has_hole"]:
            hole_objects += 1
        else:
            exact_holefree.add(_normalize_mask(_mask_in_bbox(coords)))
    if hole_objects < 1:
        return False
    if len(exact_holefree) != nohole_group_count:
        return False
    if not any(flags == {False, True} for flags in filled_groups.values()):
        return False
    if np.array_equal(expected, _outer_family_baseline(grid)):
        return False
    if np.array_equal(expected, _topology_only_baseline(grid)):
        return False
    if HOLE_COLOR not in set(map(int, expected.ravel())):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs, nohole_group_count = _family_objects(rng)
    placed_scene = _place_objects(specs, rng)
    if placed_scene is None:
        return None
    grid, placed = placed_scene
    expected = _expected_output(grid, placed)
    if not _validate_scene(grid, placed, expected, nohole_group_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
