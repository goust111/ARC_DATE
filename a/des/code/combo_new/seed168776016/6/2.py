from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 400
COLOR_POOL = tuple(range(1, 10))
ORTHOGONAL_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components from a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHOGONAL_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    shape = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return shape, max(rows) - r0 + 1, max(cols) - c0 + 1


def _placements(
    shape: tuple[tuple[int, int], ...], shape_h: int, shape_w: int, rect_h: int, rect_w: int
) -> list[frozenset[int]]:
    occs: list[frozenset[int]] = []
    for r0 in range(rect_h - shape_h + 1):
        for c0 in range(rect_w - shape_w + 1):
            occs.append(
                frozenset((r + r0) * rect_w + (c + c0) for r, c in shape)
            )
    return occs


def _tiles_rectangle(
    a_shape: tuple[tuple[int, int], ...],
    a_h: int,
    a_w: int,
    b_shape: tuple[tuple[int, int], ...],
    b_h: int,
    b_w: int,
    rect_h: int,
    rect_w: int,
) -> bool:
    if rect_h < max(a_h, b_h) or rect_w < max(a_w, b_w):
        return False

    a_places = _placements(a_shape, a_h, a_w, rect_h, rect_w)
    b_places = _placements(b_shape, b_h, b_w, rect_h, rect_w)
    if not a_places or not b_places:
        return False

    if len(a_places) > len(b_places):
        a_places, b_places = b_places, a_places

    full = set(range(rect_h * rect_w))
    b_set = set(b_places)
    for occ in a_places:
        if frozenset(full.difference(occ)) in b_set:
            return True
    return False


def _candidate_rectangles(
    comp_a: list[tuple[int, int]],
    comp_b: list[tuple[int, int]],
    max_h: int,
    max_w: int,
) -> list[tuple[int, int]]:
    a_shape, a_h, a_w = _normalize(comp_a)
    b_shape, b_h, b_w = _normalize(comp_b)
    total_area = len(comp_a) + len(comp_b)
    dims: set[tuple[int, int]] = set()
    for h in range(1, int(total_area**0.5) + 1):
        if total_area % h != 0:
            continue
        for rect_h, rect_w in ((h, total_area // h), (total_area // h, h)):
            if rect_h > max_h or rect_w > max_w:
                continue
            if rect_h < max(a_h, b_h) or rect_w < max(a_w, b_w):
                continue
            if _tiles_rectangle(a_shape, a_h, a_w, b_shape, b_h, b_w, rect_h, rect_w):
                dims.add((rect_h, rect_w))
    return sorted(dims)


def _infer_rectangle_shape(
    comp_a: list[tuple[int, int]],
    comp_b: list[tuple[int, int]],
    max_h: int,
    max_w: int,
) -> tuple[int, int]:
    dims = _candidate_rectangles(comp_a, comp_b, max_h, max_w)
    assert len(dims) == 1, f"expected one rectangle size, got {dims}"
    return dims[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each color's two complementary fragments, recover the full rectangle, and move it to the edge-marked destination.

    Input structure:
    - Background is black.
    - Each non-black color appears in exactly two interior fragments plus one marker on the top edge and one marker on the left edge.
    - The two same-color fragments are 4-connected shapes that do not touch the top row or left column.
    - A top-edge marker gives the destination column, and a left-edge marker gives the destination row.

    Rule:
    - For each color, ignore the markers and find the two interior fragments of that color.
    - Translate the two fragments without rotating or flipping them until they exactly tile one solid rectangle.
    - Clear the original scene and draw only the recovered solid rectangle at the row/column determined by the same-color edge markers.

    Key insight:
    - The fragments encode only the rectangle's size; the top-row and left-column singleton markers encode where that rectangle must be placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    interior = g.copy()
    interior[0, :] = BG
    interior[:, 0] = BG
    out = np.full((h, w), BG, dtype=int)

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors
    for color in colors:
        row_marks = np.flatnonzero(g[:, 0] == color)
        col_marks = np.flatnonzero(g[0, :] == color)
        assert len(row_marks) >= 1 and len(col_marks) >= 1
        target_r = int(row_marks[0])
        target_c = int(col_marks[0])

        comps = _components(interior == color)
        assert len(comps) == 2, f"color {color} must have exactly two fragments"
        rect_h, rect_w = _infer_rectangle_shape(comps[0], comps[1], h - target_r, w - target_c)

        dest = out[target_r : target_r + rect_h, target_c : target_c + rect_w]
        assert dest.shape == (rect_h, rect_w)
        assert np.all(dest == BG), "assembled rectangles must not overlap"
        dest[:, :] = color

    return out


def _crop_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    assert len(coords) > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _profile(length: int, low: int, high: int) -> list[int]:
    values = [random.randint(low, high)]
    for _ in range(1, length):
        prev = values[-1]
        options = {prev}
        if prev > low:
            options.add(prev - 1)
        if prev < high:
            options.add(prev + 1)
        values.append(random.choice(sorted(options)))
    if len(set(values)) == 1:
        idx = random.randrange(length)
        if values[idx] == low:
            values[idx] += 1
        elif values[idx] == high:
            values[idx] -= 1
        else:
            values[idx] += random.choice((-1, 1))
    return values


def _complementary_pieces(color: int, rect_h: int, rect_w: int, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    patch_a = np.zeros((rect_h, rect_w), dtype=int)
    patch_b = np.zeros((rect_h, rect_w), dtype=int)

    if orientation == "vertical":
        split = _profile(rect_h, 1, rect_w - 1)
        for r, width in enumerate(split):
            patch_a[r, :width] = color
            patch_b[r, width:] = color
    else:
        split = _profile(rect_w, 1, rect_h - 1)
        for c, height in enumerate(split):
            patch_a[:height, c] = color
            patch_b[height:, c] = color

    piece_a = _crop_patch(patch_a)
    piece_b = _crop_patch(patch_b)

    dims = _candidate_rectangles(
        [tuple(map(int, rc)) for rc in np.argwhere(piece_a == color)],
        [tuple(map(int, rc)) for rc in np.argwhere(piece_b == color)],
        30,
        30,
    )
    if dims != [(rect_h, rect_w)]:
        raise AssertionError(f"ambiguous complementary pair: expected {(rect_h, rect_w)}, got {dims}")

    return piece_a, piece_b


def _balanced_orientations(count: int) -> list[str]:
    result = ["vertical", "horizontal"] * (count // 2)
    if count % 2 == 1:
        result.append(random.choice(["vertical", "horizontal"]))
    random.shuffle(result)
    return result


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    blocked[max(0, r0 - 1) : min(blocked.shape[0], r1 + 2), max(0, c0 - 1) : min(blocked.shape[1], c1 + 2)] = True


def _sample_targets(specs: list[dict], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    targets: list[tuple[int, int] | None] = [None] * len(specs)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(range(len(specs)), key=lambda i: specs[i]["rect_h"] * specs[i]["rect_w"], reverse=True)
    for idx in order:
        rect_h = int(specs[idx]["rect_h"])
        rect_w = int(specs[idx]["rect_w"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, grid_h - rect_h + 1):
            if r0 in used_rows:
                continue
            for c0 in range(1, grid_w - rect_w + 1):
                if c0 in used_cols:
                    continue
                if blocked[r0 : r0 + rect_h, c0 : c0 + rect_w].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        target = random.choice(candidates)
        targets[idx] = target
        used_rows.add(target[0])
        used_cols.add(target[1])
        _mark_blocked(blocked, target[0], target[1], target[0] + rect_h - 1, target[1] + rect_w - 1)

    assert all(target is not None for target in targets)
    return [target for target in targets if target is not None]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    view[mask] = patch[mask]


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    patch_h, patch_w = patch.shape
    h, w = grid.shape
    row_stop = h - patch_h
    col_stop = w - patch_w
    if row_stop < 2 or col_stop < 2:
        return False

    candidates: list[tuple[int, int]] = []
    for r0 in range(2, row_stop + 1):
        for c0 in range(2, col_stop + 1):
            if blocked[r0 : r0 + patch_h, c0 : c0 + patch_w].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False

    top, left = random.choice(candidates)
    _paste_patch(grid, patch, top, left)
    _mark_blocked(blocked, top, left, top + patch_h - 1, left + patch_w - 1)
    return True


def _gen_specs(grid_h: int, grid_w: int) -> list[dict]:
    max_colors = 4 if min(grid_h, grid_w) >= 24 else 3 if min(grid_h, grid_w) >= 20 else 2
    count = random.randint(2, max_colors)
    colors = random.sample(COLOR_POOL, count)
    orientations = _balanced_orientations(count)
    specs: list[dict] = []

    for color, orientation in zip(colors, orientations):
        built = None
        for _ in range(60):
            if orientation == "vertical":
                rect_h = random.randint(3, 7)
                rect_w = random.randint(4, 8)
            else:
                rect_h = random.randint(4, 8)
                rect_w = random.randint(3, 7)
            try:
                piece_a, piece_b = _complementary_pieces(color, rect_h, rect_w, orientation)
            except AssertionError:
                continue
            built = {
                "color": color,
                "orientation": orientation,
                "rect_h": rect_h,
                "rect_w": rect_w,
                "pieces": (piece_a, piece_b),
            }
            break
        if built is None:
            raise AssertionError("failed to build an unambiguous fragment pair")
        specs.append(built)
    return specs


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(18, 30)
    grid_w = random.randint(18, 30)
    try:
        specs = _gen_specs(grid_h, grid_w)
    except AssertionError:
        return None
    targets = _sample_targets(specs, grid_h, grid_w)
    if targets is None:
        return None

    expected = np.zeros((grid_h, grid_w), dtype=int)
    for spec, (r0, c0) in zip(specs, targets):
        expected[r0 : r0 + spec["rect_h"], c0 : c0 + spec["rect_w"]] = spec["color"]

    inp = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    for spec, (r0, c0) in zip(specs, targets):
        color = int(spec["color"])
        inp[0, c0] = color
        inp[r0, 0] = color
        _mark_blocked(blocked, 0, c0, 0, c0)
        _mark_blocked(blocked, r0, 0, r0, 0)

    fragments = []
    for spec in specs:
        for piece in spec["pieces"]:
            fragments.append(piece)
    random.shuffle(fragments)

    for piece in fragments:
        if not _place_patch(inp, blocked, piece):
            return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment ARC input")
