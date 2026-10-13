import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIZE_CHOICES = (3, 5)
GRID_H_RANGE = (28, 37)
GRID_W_RANGE = (28, 37)
LINE_LEN_RANGE = (2, 6)
EXTRA_OBJECT_RANGE = (1, 4)
MAX_GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_points(
    coords: Iterable[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    quarter_turns_cw: int,
) -> list[tuple[int, int]]:
    k = quarter_turns_cw % 4
    if k == 0:
        return [(int(r), int(c)) for r, c in coords]

    r0, c0, r1, c1 = bbox
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    rotated: list[tuple[int, int]] = []
    for r, c in coords:
        dr2 = 2 * int(r) - center_r2
        dc2 = 2 * int(c) - center_c2
        if k == 1:
            nr2 = center_r2 + dc2
            nc2 = center_c2 - dr2
        elif k == 2:
            nr2 = center_r2 - dr2
            nc2 = center_c2 - dc2
        else:
            nr2 = center_r2 - dc2
            nc2 = center_c2 + dr2
        assert nr2 % 2 == 0 and nc2 % 2 == 0, "rotation must land on integer pixels"
        rotated.append((nr2 // 2, nc2 // 2))
    return rotated


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert h >= 2 and w >= 2, "grid must fit a 2x2 legend"
    legend = grid[:2, w - 2 : w]
    colors = [int(legend[0, 0]), int(legend[0, 1]), int(legend[1, 1]), int(legend[1, 0])]
    assert len(set(colors)) == 4, "legend colors must be distinct"
    assert all(color not in (BG, GREY) for color in colors), "legend colors must avoid bg/grey"
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[:2, w - 2 : w] = True
    return {colors[0]: 0, colors[1]: 1, colors[2]: 2, colors[3]: 3}, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-right 2x2 color ring as a rotation legend, then rotate each
    pair of grey objects connected by a colored line around its own bbox center.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rot_map, legend_mask = _parse_legend(g)

    grey_components = _components(g == GREY)
    assert grey_components, "expected at least one grey object"
    objects = []
    object_id_grid = np.full((h, w), -1, dtype=int)
    for oid, comp in enumerate(grey_components):
        bb = _bbox(comp)
        objects.append({"coords": comp, "bbox": bb})
        for r, c in comp:
            object_id_grid[r, c] = oid

    line_mask = (g != BG) & (g != GREY) & ~legend_mask
    line_components = _components(line_mask)
    object_rotations: dict[int, int] = {}
    out = g.copy()

    for comp in line_components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each line must be monochrome"
        color = next(iter(colors))
        assert color in rot_map, "line color must come from the legend"

        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(rows) == 1 or len(cols) == 1, "lines must be horizontal or vertical"
        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == max(r1 - r0, c1 - c0) + 1, "line segments must be gap-free"

        adjacent_objects = set()
        for r, c in comp:
            out[r, c] = BG
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and object_id_grid[nr, nc] != -1:
                    adjacent_objects.add(int(object_id_grid[nr, nc]))
        assert len(adjacent_objects) == 2, "each line must connect exactly two grey objects"

        turns = rot_map[color]
        for oid in adjacent_objects:
            assert oid not in object_rotations, "each grey object may touch at most one line"
            object_rotations[oid] = turns

    for oid in object_rotations:
        for r, c in objects[oid]["coords"]:
            out[r, c] = BG

    for oid, turns in object_rotations.items():
        rotated = _rotate_points(objects[oid]["coords"], objects[oid]["bbox"], turns)
        for r, c in rotated:
            assert 0 <= r < h and 0 <= c < w, "rotated object must stay in bounds"
            assert out[r, c] == BG, "rotated objects must not overlap other content"
            out[r, c] = GREY

    return out


def _rot90_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -turns % 4)


def _shape_changes(mask: np.ndarray, turns: int) -> bool:
    if turns % 4 == 0:
        return True
    return not np.array_equal(mask, _rot90_cw(mask, turns))


def _make_shape(size: int, rng: random.Random, require_change_for: int | None) -> np.ndarray | None:
    center = size // 2
    base = np.zeros((size, size), dtype=bool)
    base[center, :] = True
    base[:, center] = True

    cells = int(base.sum())
    max_extra = 3 if size == 3 else 7
    target = rng.randint(cells + 1, min(size * size - 1, cells + max_extra))

    for _ in range(80):
        mask = base.copy()
        while int(mask.sum()) < target:
            candidates = []
            for r in range(size):
                for c in range(size):
                    if mask[r, c]:
                        continue
                    if any(
                        0 <= r + dr < size
                        and 0 <= c + dc < size
                        and mask[r + dr, c + dc]
                        for dr, dc in NEIGHBORS4
                    ):
                        candidates.append((r, c))
            if not candidates:
                break
            rr, cc = rng.choice(candidates)
            mask[rr, cc] = True

        mask = _rot90_cw(mask, rng.randrange(4))
        if require_change_for is not None and not _shape_changes(mask, require_change_for):
            continue
        return mask
    return None


def _draw_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _expanded_box(
    r0: int, c0: int, r1: int, c1: int, margin: int, h: int, w: int
) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - margin),
        max(0, c0 - margin),
        min(h - 1, r1 + margin),
        min(w - 1, c1 + margin),
    )


def _can_place(occupied: np.ndarray, box: tuple[int, int, int, int], margin: int = 1) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = _expanded_box(*box, margin, h, w)
    return not occupied[r0 : r1 + 1, c0 : c1 + 1].any()


def _mark_occupied(occupied: np.ndarray, box: tuple[int, int, int, int], margin: int = 1) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = _expanded_box(*box, margin, h, w)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    line_color: int,
    horizontal: bool,
    line_len: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    s1 = left_mask.shape[0]
    s2 = right_mask.shape[0]

    if horizontal:
        box_h = max(s1, s2)
        box_w = s1 + line_len + s2
    else:
        box_h = s1 + line_len + s2
        box_w = max(s1, s2)

    if box_h > h or box_w > w:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(0, h - box_h)
        left = rng.randint(0, w - box_w)
        box = (top, left, top + box_h - 1, left + box_w - 1)
        if not _can_place(occupied, box):
            continue

        if horizontal:
            row_mid = top + box_h // 2
            top1 = top + (box_h - s1) // 2
            top2 = top + (box_h - s2) // 2
            left1 = left
            left2 = left + s1 + line_len
            _draw_mask(grid, top1, left1, left_mask, GREY)
            _draw_mask(grid, top2, left2, right_mask, GREY)
            grid[row_mid, left + s1 : left + s1 + line_len] = int(line_color)
        else:
            col_mid = left + box_w // 2
            top1 = top
            top2 = top + s1 + line_len
            left1 = left + (box_w - s1) // 2
            left2 = left + (box_w - s2) // 2
            _draw_mask(grid, top1, left1, left_mask, GREY)
            _draw_mask(grid, top2, left2, right_mask, GREY)
            grid[top + s1 : top + s1 + line_len, col_mid] = int(line_color)

        _mark_occupied(occupied, box)
        return True
    return False


def _place_single(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    s = mask.shape[0]
    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(0, h - s)
        left = rng.randint(0, w - s)
        box = (top, left, top + s - 1, left + s - 1)
        if not _can_place(occupied, box):
            continue
        _draw_mask(grid, top, left, mask, GREY)
        _mark_occupied(occupied, box)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    palette = [color for color in range(1, 10) if color != GREY]
    rng.shuffle(palette)
    ring = palette[:4]
    grid[0, w - 2] = ring[0]
    grid[0, w - 1] = ring[1]
    grid[1, w - 1] = ring[2]
    grid[1, w - 2] = ring[3]
    _mark_occupied(occupied, (0, w - 2, 1, w - 1))

    pair_specs = [(0, ring[0]), (1, ring[1]), (2, ring[2]), (3, ring[3])]
    rng.shuffle(pair_specs)
    orientations = [True, False, rng.choice([True, False]), rng.choice([True, False])]
    rng.shuffle(orientations)

    for (turns, line_color), horizontal in zip(pair_specs, orientations):
        left_size = rng.choice(SIZE_CHOICES)
        right_size = rng.choice(SIZE_CHOICES)
        left_mask = _make_shape(left_size, rng, turns)
        right_mask = _make_shape(right_size, rng, turns)
        if left_mask is None or right_mask is None:
            return None
        line_len = rng.randint(*LINE_LEN_RANGE)
        if not _place_pair(grid, occupied, left_mask, right_mask, line_color, horizontal, line_len, rng):
            return None

    extra_count = rng.randint(*EXTRA_OBJECT_RANGE)
    for _ in range(extra_count):
        turns = rng.choice([0, 1, 2, 3])
        mask = _make_shape(rng.choice(SIZE_CHOICES), rng, turns if turns else None)
        if mask is None or not _place_single(grid, occupied, mask, rng):
            return None

    # Structural sanity checks for the generated sample.
    grey_components = _components(grid == GREY)
    if len(grey_components) < 9:
        return None
    line_mask = (grid != BG) & (grid != GREY)
    line_mask[:2, w - 2 : w] = False
    line_components = _components(line_mask)
    if len(line_components) != 4:
        return None
    label_grid = np.full((h, w), -1, dtype=int)
    for idx, obj in enumerate(grey_components):
        for r, c in obj:
            label_grid[r, c] = idx

    connected_turns = set()
    changed_nonzero = 0
    for comp in line_components:
        color = int(grid[comp[0][0], comp[0][1]])
        turns = {ring[0]: 0, ring[1]: 1, ring[2]: 2, ring[3]: 3}[color]
        connected_turns.add(turns)
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if not (len(rows) == 1 or len(cols) == 1):
            return None
        adjacent = set()
        for r, c in comp:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == GREY:
                    adjacent.add((nr, nc))
        object_ids = set()
        for r, c in adjacent:
            object_ids.add(int(label_grid[r, c]))
        if len(object_ids) != 2:
            return None
        if turns:
            for oid in object_ids:
                coords = grey_components[oid]
                bb = _bbox(coords)
                if set(_rotate_points(coords, bb, turns)) != set(coords):
                    changed_nonzero += 1
                    break
    if connected_turns != {0, 1, 2, 3}:
        return None
    if changed_nonzero < 3:
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
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-legend sample")
