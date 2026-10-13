from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
INPUT_COLOR = 3
BEND_TO_COLOR = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
GEN_ATTEMPTS = 400


# Normalized connected templates grouped by (bend_count, area).
# Each tuple stores (r, c) coordinates with top-left normalization.
SHAPE_LIBRARY: dict[int, dict[int, tuple[tuple[int, int], ...]]] = {
    1: {
        4: ((0, 1), (1, 1), (2, 0), (2, 1)),
        5: ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
        6: ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (3, 1)),
        7: ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 2)),
        8: ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
        9: ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)),
        10: ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)),
    },
    2: {
        4: ((0, 0), (0, 1), (0, 2), (1, 1)),
        5: ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
        6: ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)),
        7: ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
        8: ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        9: ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)),
        10: ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)),
    },
    3: {
        5: ((0, 2), (1, 0), (1, 1), (1, 2), (2, 1)),
        6: ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)),
        7: ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
        8: ((0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2)),
        9: ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 1)),
        10: ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)),
    },
    4: {
        5: ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        6: ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)),
        7: ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)),
        8: ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
        9: ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 2)),
        10: ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)),
    },
    5: {
        7: ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (3, 1)),
        8: ((0, 1), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2)),
        9: ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)),
        10: ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3), (2, 1), (2, 2), (2, 3)),
    },
}


def _normalize_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _cells_to_mask(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _transform_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    mask = _cells_to_mask(cells)
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return _normalize_cells(tuple(map(tuple, np.argwhere(mask))))


def _component_masks(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [r0 for r0, _c0 in cells]
            cs = [c0 for _r0, c0 in cells]
            mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
            for rr, cc in cells:
                mask[int(rr - min(rs)), int(cc - min(cs))] = True
            out.append(mask)
    return out


def _count_bends(mask: np.ndarray) -> int:
    h, w = mask.shape
    count = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if int(mask[r : r + 2, c : c + 2].sum()) == 3:
                count += 1
    return count


def _ring_output(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor same-color objects by bend count, then encode their area order as nested square rings.

    Input structure:
    - The grid has a black background and several non-touching green 4-connected objects.
    - Every object is a small irregular connected mask, and all input objects share the same original color.
    - The objects have distinct bend counts and distinct areas, so topology and size must be read independently.

    Rule:
    - Find every green 4-connected component and count, inside its own mask, how many 2x2 windows contain exactly three foreground pixels.
    - Recolor each component by that bend count using 1->blue, 2->red, 3->yellow, 4->grey, 5->magenta.
    - Sort the components by decreasing area and create a square output of side 2N-1 whose outer-to-inner rings use the recolored components in that area order.

    Key insight:
    - Output color comes from a topological property of each object, while ring position comes from an independent size ranking; both analyses are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _component_masks(g, INPUT_COLOR)
    assert components, "expected at least one green component"

    annotated: list[tuple[int, int]] = []
    for mask in components:
        bends = _count_bends(mask)
        assert bends in BEND_TO_COLOR, f"unsupported bend count: {bends}"
        area = int(mask.sum())
        annotated.append((area, BEND_TO_COLOR[bends]))

    annotated.sort(key=lambda item: item[0], reverse=True)
    return _ring_output([color for _area, color in annotated])


def _choose_area_assignment(n_objects: int) -> dict[int, int] | None:
    bends = list(range(1, n_objects + 1))
    for _ in range(200):
        area_by_bend: dict[int, int] = {}
        used: set[int] = set()
        order = bends[:]
        random.shuffle(order)
        ok = True
        for bend in order:
            candidates = [area for area in SHAPE_LIBRARY[bend] if area not in used]
            if not candidates:
                ok = False
                break
            area = random.choice(candidates)
            area_by_bend[bend] = int(area)
            used.add(int(area))
        if not ok:
            continue

        areas = [area_by_bend[bend] for bend in bends]
        if all(areas[i] < areas[i + 1] for i in range(n_objects - 1)):
            continue
        if all(areas[i] > areas[i + 1] for i in range(n_objects - 1)):
            continue
        if area_by_bend[n_objects] == max(areas):
            continue
        if area_by_bend[1] == min(areas):
            continue
        return area_by_bend
    return None


def _dilate_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_objects(h: int, w: int, specs: list[tuple[int, tuple[tuple[int, int], ...]]]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    placed_specs = sorted(
        specs,
        key=lambda item: (
            max(r for r, _c in item[1]) + 1,
            max(c for _r, c in item[1]) + 1,
            len(item[1]),
        ),
        reverse=True,
    )

    for _bend, cells in placed_specs:
        patch_cells = list(cells)
        patch_h = max(r for r, _c in patch_cells) + 1
        patch_w = max(c for _r, c in patch_cells) + 1
        candidates = [(r0, c0) for r0 in range(h - patch_h + 1) for c0 in range(w - patch_w + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            absolute = [(r0 + r, c0 + c) for r, c in patch_cells]
            if any(blocked[r, c] for r, c in absolute):
                continue
            for r, c in absolute:
                grid[r, c] = INPUT_COLOR
            _dilate_blocked(blocked, absolute)
            placed = True
            break
        if not placed:
            return None
    return grid


def _expected_output(area_by_bend: dict[int, int]) -> np.ndarray:
    ordered = sorted(
        ((int(area), int(BEND_TO_COLOR[bend])) for bend, area in area_by_bend.items()),
        key=lambda item: item[0],
        reverse=True,
    )
    return _ring_output([color for _area, color in ordered])


def _validate_templates() -> None:
    for bend, area_map in SHAPE_LIBRARY.items():
        for area, cells in area_map.items():
            mask = _cells_to_mask(cells)
            assert int(mask.sum()) == int(area)
            assert _count_bends(mask) == int(bend)


def _validate_semantics(grid: np.ndarray, n_objects: int, expected: np.ndarray) -> bool:
    components = _component_masks(grid, INPUT_COLOR)
    if len(components) != int(n_objects):
        return False

    areas: list[int] = []
    bends: list[int] = []
    for mask in components:
        areas.append(int(mask.sum()))
        bends.append(int(_count_bends(mask)))
    if sorted(bends) != list(range(1, n_objects + 1)):
        return False
    if len(set(areas)) != int(n_objects):
        return False

    ordered_by_bend = [areas[bends.index(b)] for b in range(1, n_objects + 1)]
    if all(ordered_by_bend[i] < ordered_by_bend[i + 1] for i in range(n_objects - 1)):
        return False
    if all(ordered_by_bend[i] > ordered_by_bend[i + 1] for i in range(n_objects - 1)):
        return False
    if ordered_by_bend[-1] == max(ordered_by_bend):
        return False
    if ordered_by_bend[0] == min(ordered_by_bend):
        return False

    produced = main(grid)
    return np.array_equal(produced, expected)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(3, 5)
    area_by_bend = _choose_area_assignment(n_objects)
    if area_by_bend is None:
        return None

    specs: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for bend in range(1, n_objects + 1):
        area = area_by_bend[bend]
        cells = SHAPE_LIBRARY[bend][area]
        cells = _transform_cells(cells)
        specs.append((bend, cells))

    max_h = max(max(r for r, _c in cells) + 1 for _bend, cells in specs)
    max_w = max(max(c for _r, c in cells) + 1 for _bend, cells in specs)

    min_side = max(13, 2 * n_objects + max(max_h, max_w) + 2)
    if min_side > 18:
        return None
    h = random.randint(min_side, 18)
    w = random.randint(min_side, 18)

    grid = _place_objects(h, w, specs)
    if grid is None:
        return None

    expected = _expected_output(area_by_bend)
    if not _validate_semantics(grid, n_objects, expected):
        return None
    return grid


_validate_templates()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
