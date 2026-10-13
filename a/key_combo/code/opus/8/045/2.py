from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


@dataclass(frozen=True)
class ShapeTemplate:
    name: str
    cells: tuple[tuple[int, int], ...]


SPECIAL_TEMPLATE = ShapeTemplate(
    "line4",
    ((0, 0), (0, 1), (0, 2), (0, 3)),
)

IRREGULAR_TEMPLATES = [
    ShapeTemplate("L3", ((0, 0), (1, 0), (1, 1))),
    ShapeTemplate("L4", ((0, 0), (1, 0), (2, 0), (2, 1))),
    ShapeTemplate("Z4", ((0, 0), (0, 1), (1, 1), (1, 2))),
    ShapeTemplate("T4", ((0, 1), (1, 0), (1, 1), (1, 2))),
    ShapeTemplate("P5", ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))),
    ShapeTemplate("U5", ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2))),
    ShapeTemplate("L5", ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))),
    ShapeTemplate("T5", ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))),
]


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _transform_cells(
    cells: tuple[tuple[int, int], ...],
    rotation: int = 0,
    flip: bool = False,
) -> tuple[tuple[int, int], ...]:
    pts = [(r, c) for r, c in cells]
    for _ in range(rotation % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    return _normalize(pts)


def _variants(template: ShapeTemplate) -> list[tuple[tuple[int, int], ...]]:
    out = []
    seen = set()
    for rot in range(4):
        for flip in (False, True):
            variant = _transform_cells(template.cells, rot, flip)
            if variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out


def _bbox(cells: np.ndarray | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    arr = np.asarray(cells, dtype=int)
    return int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 0].max()), int(arr[:, 1].max())


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(sorted(coords), dtype=int))
    return comps


def _find_blue_components(grid: np.ndarray) -> list[np.ndarray]:
    comps = _components_mask(grid == BLUE)
    if not comps:
        raise AssertionError("expected at least one blue shape")
    return comps


def _settle_sort_key(coords: np.ndarray) -> tuple[int, int, int]:
    r0, c0, r1, _ = _bbox(coords)
    return (-r1, c0, r0)


def _settle_components(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    if H < 4 or W < 8 or W % 2 != 0:
        raise AssertionError("expected an even-width ARC-sized grid")
    if not np.all(g[-1, :] == RED):
        raise AssertionError("expected a full red bottom wall")
    if np.any(g[:-1, :] == RED):
        raise AssertionError("red should only appear in the bottom wall")

    comps = _find_blue_components(g)
    occupied = np.zeros((H, W), dtype=bool)
    occupied[-1, :] = True
    settled = []

    for coords in sorted(comps, key=_settle_sort_key):
        cur = coords.copy()
        while True:
            nxt = cur + np.array([1, 0], dtype=int)
            if np.any(nxt[:, 0] >= H) or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            cur = nxt
        if np.any(occupied[cur[:, 0], cur[:, 1]]):
            raise AssertionError("settled shapes must not overlap")
        occupied[cur[:, 0], cur[:, 1]] = True
        settled.append(cur)
    return settled


def _symmetry_recolor(shape_mask: np.ndarray) -> np.ndarray:
    reflected = shape_mask[:, ::-1]
    symmetric = shape_mask & reflected
    out = np.zeros(shape_mask.shape, dtype=int)
    out[symmetric] = GREEN
    out[shape_mask & ~symmetric] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the blue shapes onto the red floor wall, then recolor the settled mask by left-right reflection symmetry.

    Input structure:
    - The grid has black background and a full red wall along the entire bottom row.
    - Above the wall are 3 to 5 separate blue 4-connected small shapes, each with 3 to 5 pixels.
    - The width is even, so the vertical reflection axis lies between the two middle columns.

    Rule:
    - Keep the red wall fixed and remove the blue shapes from their original positions.
    - Settle the blue shapes downward as rigid bodies, nearest to the wall first; each shape stops when its next step would hit the wall, the grid boundary, or an already settled shape.
    - After all settling is complete, take the union of settled blue pixels and compare it with its left-right mirror.
    - Recolor settled pixels whose mirrored counterpart is also settled to green, and recolor all other settled pixels to orange.
    - Leave the red wall red and the background black.

    Key insight:
    - The symmetry test is not applied to the input layout; it is applied only after gravity has changed the blue configuration.
    """
    g = np.asarray(grid, dtype=int)
    settled = _settle_components(g)
    out = np.zeros_like(g)
    out[-1, :] = RED
    mask = np.zeros(g.shape, dtype=bool)
    for coords in settled:
        mask[coords[:, 0], coords[:, 1]] = True
    recolored = _symmetry_recolor(mask)
    out[recolored != BLACK] = recolored[recolored != BLACK]
    return out


def _touches_8(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    cells_b = {tuple(map(int, p)) for p in coords_b}
    for r, c in coords_a:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR8 + [(0, 0)]:
            if (rr + dr, cc + dc) in cells_b:
                return True
    return False


def _render_input(shape: tuple[int, int], input_objects: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[-1, :] = RED
    for coords in input_objects:
        grid[coords[:, 0], coords[:, 1]] = BLUE
    return grid


def _render_settled(shape: tuple[int, int], settled_objects: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[-1, :] = RED
    for coords in settled_objects:
        grid[coords[:, 0], coords[:, 1]] = BLUE
    return grid


def _candidate_positions(
    shape_cells: tuple[tuple[int, int], ...],
    occupied: np.ndarray,
    require_shape_support: bool,
) -> list[np.ndarray]:
    H, W = occupied.shape
    sr0, sc0, sr1, sc1 = _bbox(shape_cells)
    h = sr1 - sr0 + 1
    w = sc1 - sc0 + 1
    candidates = []
    for r0 in range(0, H - 1 - h + 1):
        for c0 in range(0, W - w + 1):
            coords = np.asarray([(r0 + r, c0 + c) for r, c in shape_cells], dtype=int)
            if np.any(occupied[coords[:, 0], coords[:, 1]]):
                continue
            below = coords + np.array([1, 0], dtype=int)
            support_cells = occupied[below[:, 0], below[:, 1]]
            if not np.any(support_cells):
                continue
            if require_shape_support:
                if not np.any(support_cells & (below[:, 0] < H - 1)):
                    continue
            candidates.append(coords)
    random.shuffle(candidates)
    return candidates


def _canonical_shape_key(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    best = None
    for rot in range(4):
        for flip in (False, True):
            cand = _transform_cells(cells, rot, flip)
            if best is None or cand < best:
                best = cand
    assert best is not None
    return best


def _build_settled_scene() -> tuple[list[np.ndarray], list[str], tuple[int, int]] | None:
    H = random.randint(10, 13)
    W = random.choice([10, 12, 14])
    object_count = random.randint(3, 5)

    irregular_count = object_count - 1
    templates = random.sample(IRREGULAR_TEMPLATES, irregular_count)
    variants = []
    shape_names = [SPECIAL_TEMPLATE.name]
    total_width = 4

    for template in templates:
        choices = _variants(template)
        random.shuffle(choices)
        picked = choices[0]
        total_width += _bbox(picked)[3] - _bbox(picked)[1] + 1
        variants.append(picked)
        shape_names.append(template.name)

    if total_width > int(np.floor(0.8 * W)):
        return None

    occupied = np.zeros((H, W), dtype=bool)
    occupied[-1, :] = True
    settled_objects: list[np.ndarray] = []

    center_c = W // 2 - 2
    base = np.asarray([(H - 2, center_c + c) for _, c in SPECIAL_TEMPLATE.cells], dtype=int)
    occupied[base[:, 0], base[:, 1]] = True
    settled_objects.append(base)

    stacked_done = False
    side_orange_done = False

    planned = variants[:]
    random.shuffle(planned)
    for idx, shape_cells in enumerate(planned):
        require_shape_support = not stacked_done
        candidates = _candidate_positions(shape_cells, occupied, require_shape_support)
        if require_shape_support and not candidates:
            return None
        if not candidates:
            candidates = _candidate_positions(shape_cells, occupied, False)
        if not candidates:
            return None

        if not side_orange_done:
            side_candidates = []
            for cand in candidates:
                _, c0, _, c1 = _bbox(cand)
                mirrored_cols = {W - 1 - int(c) for c in cand[:, 1]}
                if max(cand[:, 0]) == H - 2 and (c1 < W // 2 - 2 or c0 > W // 2 + 1):
                    if not all(int(c) in mirrored_cols for c in cand[:, 1]):
                        side_candidates.append(cand)
            if side_candidates:
                candidates = side_candidates

        chosen = candidates[0]
        below = chosen + np.array([1, 0], dtype=int)
        if np.any(occupied[below[:, 0], below[:, 1]] & (below[:, 0] < H - 1)):
            stacked_done = True
        if max(chosen[:, 0]) == H - 2:
            cols = set(map(int, chosen[:, 1]))
            mirrored = {W - 1 - c for c in cols}
            if cols != mirrored:
                side_orange_done = True

        occupied[chosen[:, 0], chosen[:, 1]] = True
        settled_objects.append(chosen)

    if not stacked_done or not side_orange_done:
        return None
    return settled_objects, shape_names, (H, W)


def _assign_input_positions(settled_objects: list[np.ndarray], shape: tuple[int, int]) -> list[np.ndarray] | None:
    H, W = shape
    order = sorted(range(len(settled_objects)), key=lambda i: _settle_sort_key(settled_objects[i]))
    assigned: list[np.ndarray | None] = [None] * len(settled_objects)

    def rec(pos: int, prev_bottom: int | None, placed: list[np.ndarray]) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        final_coords = settled_objects[idx]
        r0, _, r1, _ = _bbox(final_coords)
        max_shift = r0
        choices = list(range(1, max_shift + 1))
        random.shuffle(choices)

        ranked = []
        for shift in choices:
            moved = final_coords - np.array([shift, 0], dtype=int)
            moved_bottom = int(moved[:, 0].max())
            if moved_bottom >= H - 2:
                continue
            if prev_bottom is not None and moved_bottom >= prev_bottom:
                continue
            if any(_touches_8(moved, other) for other in placed):
                continue
            ranked.append((shift, moved))

        ranked.sort(key=lambda item: (-item[0], -int(item[1][:, 0].max())))
        for _, moved in ranked:
            assigned[idx] = moved
            if rec(pos + 1, int(moved[:, 0].max()), placed + [moved]):
                return True
            assigned[idx] = None
        return False

    if not rec(0, None, []):
        return None
    return [arr for arr in assigned if arr is not None]


def _classify_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reflected = mask[:, ::-1]
    symmetric = mask & reflected
    asymmetric = mask & ~reflected
    return symmetric, asymmetric


def _validate_generated(
    inp: np.ndarray,
    settled_objects: list[np.ndarray],
    input_objects: list[np.ndarray],
    shape_names: list[str],
) -> bool:
    H, W = inp.shape
    if W % 2 != 0 or H > 30 or W > 30:
        return False
    if len(input_objects) != len(settled_objects) or not (3 <= len(input_objects) <= 5):
        return False
    if not np.all(inp[-1, :] == RED) or np.any(inp[:-1, :] == RED):
        return False
    if any(max(obj[:, 0]) >= H - 2 for obj in input_objects):
        return False
    if len(set(shape_names)) != len(shape_names):
        return False
    if SPECIAL_TEMPLATE.name not in shape_names:
        return False
    if all(name == SPECIAL_TEMPLATE.name for name in shape_names):
        return False
    if sum(_bbox(obj)[3] - _bbox(obj)[1] + 1 for obj in input_objects) > int(np.floor(0.8 * W)):
        return False

    for obj in input_objects:
        if len(obj) < 3 or len(obj) > 5:
            return False
    for i in range(len(input_objects)):
        for j in range(i + 1, len(input_objects)):
            if _touches_8(input_objects[i], input_objects[j]):
                return False

    components = _find_blue_components(inp)
    areas = sorted(len(comp) for comp in components)
    if len(components) != len(input_objects) or areas != sorted(len(obj) for obj in input_objects):
        return False

    settled_grid = _render_settled(inp.shape, settled_objects)
    expected = main(inp)
    if not np.array_equal(expected, _oracle_output(inp)):
        return False

    output_mask = settled_grid == BLUE
    sym, asym = _classify_mask(output_mask)
    if not np.any(sym) or not np.any(asym):
        return False
    if not np.any(expected == GREEN) or not np.any(expected == ORANGE):
        return False

    input_mask = inp == BLUE
    input_sym, input_asym = _classify_mask(input_mask)
    if np.array_equal(sym, input_sym) and np.array_equal(asym, input_asym):
        return False

    moved = [
        int(final[:, 0].min() - inp_obj[:, 0].min())
        for final, inp_obj in zip(settled_objects, input_objects)
    ]
    if not any(delta > 0 for delta in moved):
        return False

    stacked = 0
    occupied = np.zeros(inp.shape, dtype=bool)
    occupied[-1, :] = True
    for coords in sorted(settled_objects, key=_settle_sort_key):
        below = coords + np.array([1, 0], dtype=int)
        if np.any(occupied[below[:, 0], below[:, 1]] & (below[:, 0] < H - 1)):
            stacked += 1
        occupied[coords[:, 0], coords[:, 1]] = True
    if stacked == 0:
        return False
    return True


def _oracle_output(inp: np.ndarray) -> np.ndarray:
    settled = _settle_components(inp)
    out = np.zeros_like(inp)
    out[-1, :] = RED
    mask = np.zeros(inp.shape, dtype=bool)
    for coords in settled:
        mask[coords[:, 0], coords[:, 1]] = True
    recolored = _symmetry_recolor(mask)
    out[recolored != BLACK] = recolored[recolored != BLACK]
    return out


def _single_attempt() -> np.ndarray | None:
    built = _build_settled_scene()
    if built is None:
        return None
    settled_objects, shape_names, shape = built
    input_objects = _assign_input_positions(settled_objects, shape)
    if input_objects is None:
        return None
    inp = _render_input(shape, input_objects)
    if not _validate_generated(inp, settled_objects, input_objects, shape_names):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _single_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid gravity-reflection ARC input")
