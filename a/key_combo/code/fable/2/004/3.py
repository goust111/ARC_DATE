from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC = SourceFileLoader(
    "_shared_arc_utils_fable_2_004_3",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

colors_present = _ARC.colors_present
components_mask = _ARC.components_mask
find_components = _ARC.find_components
get_bg = _ARC.get_bg
neighbors4 = _ARC.neighbors4
paste = _ARC.paste


GRID_LIMIT = 30
HIDDEN_HEIGHT_RANGE = (4, 6)
HIDDEN_WIDTH_RANGE = (4, 6)
HIDDEN_AREA_MIN = 8
HIDDEN_AREA_MAX = 14
COPY_COUNT_RANGE = (3, 5)
EXTRA_REMOVALS_MAX = 2
GEN_RETRIES = 400
PALETTE = tuple(range(10))

LAYOUTS_BY_COPY_COUNT = {
    3: ((1, 3), (2, 2)),
    4: ((2, 2),),
    5: ((2, 3), (3, 2)),
}


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in cells)


def _cells_mask(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _is_connected(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    if not cells:
        return False
    return len(components_mask(_cells_mask(cells, h, w), connectivity=4)) == 1


def _spans_bbox(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows) == 0 and max(rows) == h - 1 and min(cols) == 0 and max(cols) == w - 1


def _boundary_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for r, c in cells:
        if any((nr, nc) not in cells for nr, nc in neighbors4(r, c)):
            out.append((int(r), int(c)))
    return out


def _stable_cells(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for cell in cells:
        remainder = set(cells)
        remainder.remove(cell)
        if remainder and _is_connected(remainder, h, w):
            out.append((int(cell[0]), int(cell[1])))
    return out


def _random_hidden_shape() -> tuple[int, int, set[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        h = random.randint(*HIDDEN_HEIGHT_RANGE)
        w = random.randint(*HIDDEN_WIDTH_RANGE)
        target_area = random.randint(max(HIDDEN_AREA_MIN, h + w - 1), min(HIDDEN_AREA_MAX, h * w - 1))

        cells = {(r, c) for r in range(h) for c in range(w)}
        while len(cells) > target_area:
            removable = []
            for cell in _boundary_cells(cells):
                remainder = set(cells)
                remainder.remove(cell)
                if _spans_bbox(remainder, h, w) and _is_connected(remainder, h, w):
                    removable.append(cell)
            if not removable:
                break
            cells.remove(random.choice(removable))

        if len(cells) != target_area:
            continue

        stable = _stable_cells(cells, h, w)
        anchor_candidates = [cell for cell in cells if sum(1 for s in stable if s != cell) >= 2]
        if not anchor_candidates:
            continue

        interior = [cell for cell in anchor_candidates if 0 < cell[0] < h - 1 and 0 < cell[1] < w - 1]
        if interior and random.random() < 0.65:
            anchor = random.choice(interior)
        else:
            anchor = random.choice(anchor_candidates)
        return h, w, cells, (int(anchor[0]), int(anchor[1]))

    raise RuntimeError("failed to create a hidden shape")


def _build_fragments(
    shape_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
    h: int,
    w: int,
    n_copies: int,
) -> list[set[tuple[int, int]]] | None:
    stable = [cell for cell in _stable_cells(shape_cells, h, w) if cell != anchor]
    if len(stable) < 2:
        return None

    max_unique = min(4, n_copies, len(stable), max(2, len(shape_cells) // 3))
    if max_unique < 2:
        return None

    for _ in range(120):
        unique_count = random.randint(2, max_unique)
        unique_cells = random.sample(stable, unique_count)
        owner_indices = random.sample(range(n_copies), unique_count)
        owner_of = {cell: owner for cell, owner in zip(unique_cells, owner_indices)}

        fragments: list[set[tuple[int, int]]] = []
        for frag_index in range(n_copies):
            fragment = set(shape_cells)
            for cell, owner in owner_of.items():
                if owner != frag_index:
                    fragment.remove(cell)

            if anchor not in fragment or len(fragment) < 3 or not _is_connected(fragment, h, w):
                break

            protected = {cell for cell, owner in owner_of.items() if owner == frag_index}
            removable_pool = list(fragment - {anchor} - protected)
            random.shuffle(removable_pool)
            extra_budget = random.randint(0, min(EXTRA_REMOVALS_MAX, max(0, len(removable_pool) - 2)))
            for cell in removable_pool:
                if extra_budget == 0:
                    break
                candidate = set(fragment)
                candidate.remove(cell)
                if len(candidate) >= 3 and _is_connected(candidate, h, w):
                    fragment = candidate
                    extra_budget -= 1

            fragments.append(fragment)
        else:
            union = set().union(*fragments)
            if union != shape_cells:
                continue
            if any(fragment == shape_cells for fragment in fragments):
                continue
            if len({_normalize(fragment) for fragment in fragments}) != len(fragments):
                continue
            if len({frozenset(shape_cells - fragment) for fragment in fragments}) != len(fragments):
                continue

            unique_contributors = 0
            for i, fragment in enumerate(fragments):
                others = set().union(*(fragments[j] for j in range(len(fragments)) if j != i))
                if (fragment - others) - {anchor}:
                    unique_contributors += 1
            if unique_contributors < 2:
                continue

            return fragments

    return None


def _layout_positions(h: int, w: int, n_copies: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    layouts = list(LAYOUTS_BY_COPY_COUNT[n_copies])
    random.shuffle(layouts)
    for rows, cols in layouts:
        slot_h = h + 4
        slot_w = w + 4
        height = rows * slot_h
        width = cols * slot_w
        if height > GRID_LIMIT or width > GRID_LIMIT or rows * cols < n_copies:
            continue

        slot_ids = [(r, c) for r in range(rows) for c in range(cols)]
        chosen_slots = random.sample(slot_ids, n_copies)
        positions = []
        for sr, sc in chosen_slots:
            top = sr * slot_h + random.randint(1, slot_h - h - 1)
            left = sc * slot_w + random.randint(1, slot_w - w - 1)
            positions.append((int(top), int(left)))
        return height, width, positions
    return None


def _render_input(
    h: int,
    w: int,
    shape_anchor: tuple[int, int],
    fragments: list[set[tuple[int, int]]],
    positions: list[tuple[int, int]],
    colors: tuple[int, int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    bg, body_color, anchor_color = colors
    grid = np.full(canvas_shape, bg, dtype=int)
    for fragment, (top, left) in zip(fragments, positions):
        patch = np.full((h, w), bg, dtype=int)
        for r, c in fragment:
            patch[r, c] = anchor_color if (r, c) == shape_anchor else body_color
        paste(grid, patch, top, left)
    return grid


def _render_output(
    h: int,
    w: int,
    shape_cells: set[tuple[int, int]],
    positions: list[tuple[int, int]],
    colors: tuple[int, int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    bg, body_color, _anchor_color = colors
    grid = np.full(canvas_shape, bg, dtype=int)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in shape_cells:
        patch[r, c] = body_color
    for top, left in positions:
        paste(grid, patch, top, left)
    return grid


def _extract_instances(grid: np.ndarray) -> tuple[int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected one body color and one anchor color"

    color_components = {
        int(color): find_components(g, bg=bg, colors=[int(color)], connectivity=8, monochromatic=True)
        for color in fg_colors
    }
    anchor_candidates = [
        color
        for color, comps in color_components.items()
        if comps and all(int(comp["area"]) == 1 for comp in comps)
    ]
    assert len(anchor_candidates) == 1, "anchor color must be the unique singleton-only foreground color"
    anchor_color = int(anchor_candidates[0])
    body_color = next(int(color) for color in fg_colors if int(color) != anchor_color)
    assert all(int(comp["area"]) >= 2 for comp in color_components[body_color]), "body fragments must be multi-pixel"

    instances = []
    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=False):
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        anchors = [(r, c) for r, c in coords if int(g[r, c]) == anchor_color]
        body = [(r, c) for r, c in coords if int(g[r, c]) == body_color]
        assert len(anchors) == 1 and len(body) >= 2, "each instance must have one anchor and body pixels"
        ar, ac = anchors[0]
        offsets = {(0, 0)}
        for r, c in body:
            offsets.add((int(r - ar), int(c - ac)))
        instances.append({"anchor": (ar, ac), "offsets": offsets})

    instances.sort(key=lambda item: item["anchor"])
    assert len(instances) == len(color_components[anchor_color]), "one anchor per instance"
    return bg, body_color, anchor_color, instances


def _completed_output(grid: np.ndarray) -> np.ndarray:
    bg, body_color, _anchor_color, instances = _extract_instances(grid)
    h, w = grid.shape
    full_offsets: set[tuple[int, int]] = set()
    for instance in instances:
        full_offsets |= instance["offsets"]

    out = np.full((h, w), bg, dtype=int)
    for instance in instances:
        ar, ac = instance["anchor"]
        for dr, dc in full_offsets:
            rr = ar + dr
            cc = ac + dc
            assert 0 <= rr < h and 0 <= cc < w, "reconstructed shape must stay on the canvas"
            out[rr, cc] = body_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Align all anchored fragments of one hidden sprite and fill every copy back to the full union shape.

    Input structure:
    - The background is dominant, and exactly two foreground colors appear.
    - One foreground color forms several incomplete copies of the same sprite; the
      other foreground color marks exactly one anchor cell inside each copy.
    - No copy is complete, but all copies share the same anchor position in the
      hidden sprite's coordinate system.

    Rule:
    - Identify each copy as one multicolor connected component with exactly one
      anchor cell.
    - Translate every copy into anchor-relative coordinates and take the union of
      all occupied offsets, counting the anchor cell itself as part of the sprite.
    - At every original anchor position, repaint that full union sprite in the
      body color and remove the anchor color from the output.

    Key insight:
    - The complete sprite never appears directly; only the anchor-aligned union of
      all incomplete traces reconstructs it, and that reconstructed union is then
      stamped back at every original location.
    """
    return _completed_output(np.asarray(grid, dtype=int))


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    shape_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
    fragments: list[set[tuple[int, int]]],
) -> bool:
    if np.array_equal(inp, expected):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    try:
        bg, body_color, anchor_color, instances = _extract_instances(inp)
    except Exception:
        return False

    if len(instances) != len(fragments):
        return False
    if any(len(fragment) >= len(shape_cells) for fragment in fragments):
        return False
    if set().union(*fragments) != shape_cells:
        return False

    recovered_offsets = set().union(*(instance["offsets"] for instance in instances))
    target_offsets = {(r - anchor[0], c - anchor[1]) for r, c in shape_cells}
    if recovered_offsets != target_offsets:
        return False

    if int(anchor_color) in set(map(int, np.unique(expected))):
        return False
    if set(map(int, np.unique(expected))) - {int(bg), int(body_color)}:
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]], tuple[int, int], list[set[tuple[int, int]]]]:
    n_copies = random.randint(*COPY_COUNT_RANGE)
    h, w, shape_cells, anchor = _random_hidden_shape()
    fragments = _build_fragments(shape_cells, anchor, h, w, n_copies)
    if fragments is None:
        raise RuntimeError("failed to build fragments")

    layout = _layout_positions(h, w, n_copies)
    if layout is None:
        raise RuntimeError("failed to place copies")
    height, width, positions = layout
    colors = tuple(random.sample(PALETTE, 3))

    inp = _render_input(h, w, anchor, fragments, positions, colors, (height, width))
    expected = _render_output(h, w, shape_cells, positions, colors, (height, width))
    return inp, expected, shape_cells, anchor, fragments


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected, shape_cells, anchor, fragments = _generate_case()
        except RuntimeError:
            continue
        if _validate_generated_case(inp, expected, shape_cells, anchor, fragments):
            return inp
    raise RuntimeError("failed to generate a valid residue-overlay input")
