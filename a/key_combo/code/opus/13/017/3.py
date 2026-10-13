from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if view.shape != patch.shape:
        raise ValueError("patch does not fit")
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
) -> list[dict]:
    if connectivity != 4:
        raise ValueError("only 4-connectivity is used in this task")
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def _component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[r - r0, c - c0] = int(comp["color"])
    return patch


def _is_lr_symmetric(patch: np.ndarray) -> bool:
    return bool(np.array_equal(patch, np.fliplr(patch)))


def _panel_layout(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _get_bg(g)

    candidates = []
    for c in range(W):
        col = g[:, c]
        if int(col[0]) == int(bg):
            continue
        if np.all(col == col[0]):
            candidates.append((c, int(col[0])))

    layouts = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, color1 = candidates[i]
            c2, color2 = candidates[j]
            if color1 != color2:
                continue
            left_w = c1
            mid_w = c2 - c1 - 1
            right_w = W - c2 - 1
            if left_w > 0 and left_w == mid_w == right_w:
                layouts.append((c1, c2, color1, left_w))

    assert len(layouts) == 1, "expected exactly two separator columns splitting three equal panels"
    sep1, sep2, sep_color, panel_w = layouts[0]
    assert sep_color != int(bg)
    panels = [g[:, :sep1], g[:, sep1 + 1 : sep2], g[:, sep2 + 1 :]]
    assert all(panel.shape == (H, panel_w) for panel in panels)
    return panels, panel_w


def _selected_panel_object(panel: np.ndarray, bg: int) -> dict:
    objects = _find_components(panel, bg=bg, connectivity=4)
    assert len(objects) == 2, "each panel must contain exactly two 4-connected objects"
    colors = {int(obj["color"]) for obj in objects}
    assert len(colors) == 2, "the two panel objects must use different colors"

    symmetric = []
    for obj in objects:
        assert 4 <= int(obj["area"]) <= 8, "panel objects must have area 4-8"
        patch = _component_patch(panel, obj, bg)
        if _is_lr_symmetric(patch):
            symmetric.append({**obj, "patch": patch})
    assert len(symmetric) == 1, "each panel must have a unique vertically symmetric object"
    return symmetric[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the vertically symmetric object inside each separator-defined panel and merge the three selections in panel order.

    Input structure:
    - The grid has a black background and three equal-width panels separated by two solid vertical separator columns.
    - Each panel contains exactly two non-touching 4-connected foreground objects of different colors.
    - In every panel, exactly one object's cropped bounding-box pattern is symmetric under left-right reflection.

    Rule:
    - Split the grid into the three panels using the solid separator columns.
    - Within each panel, crop each object to its own bounding box and test whether that crop equals its horizontal mirror image.
    - Keep the unique symmetric object from each panel.
    - Create a black output canvas with the panel height and at least the panel width; paste the three selected crops left-to-right in panel order, preserving each crop's original row position and leaving exactly one blank column between consecutive crops.

    Key insight:
    - The selection criterion is panel-local mirror symmetry, not position, color, or size; the separator bars define the scope in which symmetry must be evaluated before the chosen crops are compactly merged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, _W = g.shape
    bg = _get_bg(g)
    panels, panel_w = _panel_layout(g)

    selected = [_selected_panel_object(panel, bg) for panel in panels]
    needed_width = sum(obj["patch"].shape[1] for obj in selected) + (len(selected) - 1)
    out = np.full((H, max(panel_w, needed_width)), int(bg), dtype=int)

    cursor = 0
    for obj in selected:
        r0, _c0, _r1, _c1 = obj["bbox"]
        patch = obj["patch"]
        _paste(out, patch, int(r0), int(cursor), transparent=bg)
        cursor += patch.shape[1] + 1

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    H, W = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _grow_connected_mask(area: int, dims: tuple[int, int]) -> np.ndarray | None:
    h, w = dims
    if area > h * w:
        return None
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(cells) < area:
        options = list(frontier)
        if not options:
            return None
        weights = []
        for cell in options:
            score = sum((nbr in cells) for nbr in _neighbors4(*cell))
            weights.append(max(1, score))
        choice = random.choices(options, weights=weights, k=1)[0]
        frontier.remove(choice)
        cells.add(choice)
        add_frontier(choice)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _render_symmetric_units(h: int, w: int, units: set[tuple[int, int]]) -> np.ndarray:
    reps = (w + 1) // 2
    mask = np.zeros((h, w), dtype=bool)
    for r, c in units:
        assert 0 <= c < reps
        mirror = w - 1 - c
        mask[r, c] = True
        mask[r, mirror] = True
    return _crop_mask(mask)


def _unit_weight(w: int, c: int) -> int:
    return 1 if w % 2 == 1 and c == w // 2 else 2


def _generate_symmetric_mask(area: int, *, compact: bool) -> np.ndarray | None:
    width_choices = (2, 3) if compact else (3, 4, 5)
    for _ in range(150):
        w = random.choice(width_choices)
        h = random.randint(2, 4)
        reps = (w + 1) // 2
        bridge_c = reps - 1
        start = (random.randrange(h), bridge_c)
        units = {start}
        frontier: set[tuple[int, int]] = set()
        current = _unit_weight(w, bridge_c)

        def add_unit_frontier(cell: tuple[int, int]) -> None:
            r, c = cell
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < reps and (nr, nc) not in units:
                    frontier.add((nr, nc))

        add_unit_frontier(start)
        while current < area:
            options = [cell for cell in frontier if current + _unit_weight(w, cell[1]) <= area]
            if not options:
                break
            weights = []
            for cell in options:
                score = sum((nbr in units) for nbr in _neighbors4(*cell))
                weights.append(max(1, score))
            choice = random.choices(options, weights=weights, k=1)[0]
            frontier.remove(choice)
            units.add(choice)
            current += _unit_weight(w, choice[1])
            add_unit_frontier(choice)

        if current != area:
            continue
        mask = _render_symmetric_units(h, w, units)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not _mask_connected(mask):
            continue
        if not np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _generate_asymmetric_mask(area: int) -> np.ndarray | None:
    dims = [(h, w) for h in range(2, 5) for w in range(2, 6) if h * w >= area]
    random.shuffle(dims)
    for dims_choice in dims[:]:
        for _ in range(40):
            mask = _grow_connected_mask(area, dims_choice)
            if mask is None:
                continue
            if mask.shape[0] < 2 or mask.shape[1] < 2:
                continue
            if np.array_equal(mask, np.fliplr(mask)):
                continue
            return mask
    return None


def _color_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _rect_conflicts(
    candidate: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    *,
    gap: int = 1,
) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        separated = r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0
        if not separated:
            return True
    return False


def _place_two_patches(
    panel_size: int,
    first: np.ndarray,
    second: np.ndarray,
    *,
    symmetric_should_be_left: bool,
) -> tuple[np.ndarray, dict, dict] | None:
    panel = np.zeros((panel_size, panel_size), dtype=int)
    patches = [("sym", first), ("asym", second)]
    order = [0, 1]
    random.shuffle(order)
    placements: dict[str, dict] = {}
    placed_rects: list[tuple[int, int, int, int]] = []

    for index in order:
        key, patch = patches[index]
        h, w = patch.shape
        candidates = []
        for r0 in range(panel_size - h + 1):
            for c0 in range(panel_size - w + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if not _rect_conflicts(rect, placed_rects, gap=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _paste(panel, patch, r0, c0, transparent=0)
        placements[key] = {
            "patch": patch,
            "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        }
        placed_rects.append((r0, c0, r0 + h - 1, c0 + w - 1))

    sym_c0 = placements["sym"]["bbox"][1]
    asym_c0 = placements["asym"]["bbox"][1]
    if symmetric_should_be_left != (sym_c0 < asym_c0):
        return None
    return panel, placements["sym"], placements["asym"]


def _expected_output(selected: list[dict], panel_size: int) -> np.ndarray:
    height = panel_size
    needed_width = sum(obj["patch"].shape[1] for obj in selected) + 2
    out = np.zeros((height, max(panel_size, needed_width)), dtype=int)
    cursor = 0
    for obj in selected:
        r0 = obj["bbox"][0]
        _paste(out, obj["patch"], r0, cursor, transparent=0)
        cursor += obj["patch"].shape[1] + 1
    return out


def _panel_selected_from_analysis(panel: np.ndarray) -> dict:
    bg = 0
    obj = _selected_panel_object(panel, bg)
    return {"patch": obj["patch"], "bbox": obj["bbox"], "color": obj["color"]}


def _generate_panel(panel_size: int, *, compact_symmetry: bool, symmetric_should_be_left: bool) -> tuple[np.ndarray, dict] | None:
    equal_area = random.random() < 0.35
    for _ in range(120):
        sym_area = random.randint(4, 8)
        asym_area = sym_area if equal_area else random.randint(4, 8)
        sym_mask = _generate_symmetric_mask(sym_area, compact=compact_symmetry)
        asym_mask = _generate_asymmetric_mask(asym_area)
        if sym_mask is None or asym_mask is None:
            continue
        sym_color, asym_color = random.sample(OBJECT_COLORS, 2)
        sym_patch = _color_patch(sym_mask, sym_color)
        asym_patch = _color_patch(asym_mask, asym_color)
        placed = _place_two_patches(
            panel_size,
            sym_patch,
            asym_patch,
            symmetric_should_be_left=symmetric_should_be_left,
        )
        if placed is None:
            continue
        panel, _sym_meta, _asym_meta = placed
        selected = _panel_selected_from_analysis(panel)
        if not np.array_equal(selected["patch"], sym_patch):
            continue
        return panel, selected
    return None


def _assemble_input(panels: list[np.ndarray]) -> np.ndarray:
    H, panel_w = panels[0].shape
    grid = np.full((H, 3 * panel_w + 2), SEPARATOR, dtype=int)
    grid[:, :panel_w] = panels[0]
    grid[:, panel_w + 1 : 2 * panel_w + 1] = panels[1]
    grid[:, 2 * panel_w + 2 :] = panels[2]
    return grid


def _generate_one() -> np.ndarray | None:
    expand_output = random.random() < 0.5
    panel_size = random.choice((6, 7, 8) if expand_output else (9,))
    compact_symmetry = not expand_output
    side_pattern = [random.choice((True, False)) for _ in range(3)]

    panels = []
    selected = []
    for symmetric_should_be_left in side_pattern:
        generated = _generate_panel(
            panel_size,
            compact_symmetry=compact_symmetry,
            symmetric_should_be_left=symmetric_should_be_left,
        )
        if generated is None:
            return None
        panel, selected_obj = generated
        panels.append(panel)
        selected.append(selected_obj)

    expected = _expected_output(selected, panel_size)
    if expand_output != (expected.shape[1] > panel_size):
        return None

    grid = _assemble_input(panels)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
