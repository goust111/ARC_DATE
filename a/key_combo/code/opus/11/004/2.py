from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
FG = 2
DIVIDER = 5
GREEN = 3
BLUE = 1
YELLOW = 4

GEN_RETRIES = 400
ORTH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTH4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    divider_rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] == DIVIDER)
    ]
    assert len(divider_rows) == 2, "expected exactly two solid grey divider rows"
    d0, d1 = divider_rows
    panel_h = d0
    assert d1 == 2 * panel_h + 1, "divider rows must split the grid into three equal-height panels"
    top = grid[:d0]
    mid = grid[d0 + 1 : d1]
    bot = grid[d1 + 1 :]
    assert top.shape == mid.shape == bot.shape and top.size > 0
    return [top, mid, bot], divider_rows


def main(grid: np.ndarray) -> np.ndarray:
    """Delete the dividers, overlay the three panels by component-count order, and recolor merged components by area.

    Input structure:
    - The grid is black except for two full-width grey divider rows that split it into three equal-height panels.
    - Each panel contains one or more red 4-connected foreground objects on the shared local coordinate system.
    - The three panels have different red component counts.

    Rule:
    - Remove the two grey divider rows.
    - Count the red 4-connected components inside each panel.
    - Overlay the three panels onto one panel-sized canvas, from fewer components at the bottom to more components at the top.
    - Recompute 4-connected foreground components in the merged overlay and recolor each by size:
      size <= 2 -> green, size 3..5 -> blue, size >= 6 -> yellow.

    Key insight:
    - The first count determines how the equal-size panels are superimposed, while the final colors depend on the
      component topology only after that superposition has merged or occluded panel content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    panels, _ = _split_panels(g)

    counts = [len(_components(panel == FG)) for panel in panels]
    assert len(set(counts)) == 3, "panel component counts must be pairwise distinct"

    order = sorted(range(3), key=lambda i: counts[i])
    out_mask = np.zeros_like(panels[0], dtype=bool)
    for idx in order:
        out_mask |= panels[idx] == FG

    out = np.full(panels[0].shape, BG, dtype=int)
    for comp in _components(out_mask):
        size = int(comp.shape[0])
        color = GREEN if size <= 2 else BLUE if size <= 5 else YELLOW
        out[comp[:, 0], comp[:, 1]] = color
    return out


def _normalize(cells: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(cells, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _random_shape(rng: np.random.Generator, size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in ORTH4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(frontier[rng.integers(len(frontier))])
    return _normalize(list(cells))


def _try_place(
    rng: np.random.Generator,
    panel_mask: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray | None:
    h, w = panel_mask.shape
    sh, sw = int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1
    if sh > h or sw > w:
        return None
    tries = 60
    for _ in range(tries):
        r0 = int(rng.integers(0, h - sh + 1))
        c0 = int(rng.integers(0, w - sw + 1))
        coords = shape + np.array([[r0, c0]])
        if np.any(panel_mask[coords[:, 0], coords[:, 1]]):
            continue
        ok = True
        for r, c in coords:
            for dr, dc in ORTH4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and panel_mask[nr, nc] and not np.any(np.all(coords == (nr, nc), axis=1)):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return coords
    return None


def _special_pair(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    templates = [
        ([(0, 0), (0, 1)], [(0, 1), (0, 2)]),
        ([(0, 0), (1, 0)], [(1, 0), (2, 0)]),
        ([(0, 0), (0, 1), (1, 1)], [(1, 0), (1, 1)]),
        ([(0, 0), (1, 0)], [(1, 0), (1, 1), (2, 1)]),
    ]
    a_cells, b_cells = templates[int(rng.integers(len(templates)))]
    a = _normalize(a_cells)
    b = _normalize(b_cells)
    max_r = max(int(a[:, 0].max()), int(b[:, 0].max()))
    max_c = max(int(a[:, 1].max()), int(b[:, 1].max()))
    r0 = int(rng.integers(0, h - max_r))
    c0 = int(rng.integers(0, w - max_c))
    shift = np.array([[r0, c0]])
    return a + shift, b + shift


def _build_panel_objects(
    rng: np.random.Generator,
    panel_h: int,
    width: int,
    counts: list[int],
) -> list[list[np.ndarray]] | None:
    panels = [np.zeros((panel_h, width), dtype=bool) for _ in range(3)]
    objects: list[list[np.ndarray]] = [[] for _ in range(3)]

    merge_pair = rng.choice(3, size=2, replace=False)
    special_a, special_b = _special_pair(rng, panel_h, width)
    for panel_idx, coords in zip(merge_pair, [special_a, special_b]):
        panels[int(panel_idx)][coords[:, 0], coords[:, 1]] = True
        objects[int(panel_idx)].append(coords)

    must_place_six = int(rng.integers(3))
    if counts[must_place_six] <= len(objects[must_place_six]):
        must_place_six = int(max(range(3), key=lambda i: counts[i]))

    for panel_idx in range(3):
        while len(objects[panel_idx]) < counts[panel_idx]:
            target_size = 6 if panel_idx == must_place_six and len(objects[panel_idx]) == 0 else int(rng.integers(2, 7))
            shape = _random_shape(rng, target_size)
            placed = _try_place(rng, panels[panel_idx], shape)
            if placed is None:
                return None
            panels[panel_idx][placed[:, 0], placed[:, 1]] = True
            objects[panel_idx].append(placed)
    return objects


def _assemble_grid(panel_h: int, width: int, objects: list[list[np.ndarray]]) -> np.ndarray:
    grid = np.full((panel_h * 3 + 2, width), BG, dtype=int)
    divider_rows = [panel_h, 2 * panel_h + 1]
    grid[divider_rows[0], :] = DIVIDER
    grid[divider_rows[1], :] = DIVIDER
    for panel_idx, panel_objects in enumerate(objects):
        row0 = panel_idx * (panel_h + 1)
        for coords in panel_objects:
            rr = coords[:, 0] + row0
            cc = coords[:, 1]
            grid[rr, cc] = FG
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))) <= {BG, FG, DIVIDER}:
        return False

    try:
        panels, _ = _split_panels(grid)
    except AssertionError:
        return False

    counts = []
    all_panel_components: list[list[np.ndarray]] = []
    panel_masks = []
    for panel in panels:
        if np.any(panel == DIVIDER):
            return False
        comps = _components(panel == FG)
        if not comps:
            return False
        sizes = [int(comp.shape[0]) for comp in comps]
        if not all(2 <= size <= 6 for size in sizes):
            return False
        counts.append(len(comps))
        all_panel_components.append(comps)
        panel_masks.append(panel == FG)
    if len(set(counts)) != 3:
        return False

    overlap_exists = False
    for i in range(3):
        for j in range(i + 1, 3):
            if np.any(panel_masks[i] & panel_masks[j]):
                overlap_exists = True
    if not overlap_exists:
        return False

    overlay_mask = panel_masks[0] | panel_masks[1] | panel_masks[2]
    overlay_components = _components(overlay_mask)
    if len(overlay_components) >= sum(counts):
        return False

    merged_across_panels = False
    for comp in overlay_components:
        comp_set = {tuple(x) for x in comp.tolist()}
        touched = set()
        for panel_idx, panel_components in enumerate(all_panel_components):
            for obj_idx, obj in enumerate(panel_components):
                if comp_set & {tuple(x) for x in obj.tolist()}:
                    touched.add((panel_idx, obj_idx))
        if len({panel_idx for panel_idx, _ in touched}) >= 2 and len(touched) >= 2:
            merged_across_panels = True
            break
    if not merged_across_panels:
        return False

    out = main(grid)
    if out.shape != panels[0].shape:
        return False
    out_colors = set(map(int, np.unique(out))) - {BG}
    if not out_colors or not out_colors <= {GREEN, BLUE, YELLOW}:
        return False
    if len(out_colors) < 2:
        return False
    if np.mean(overlay_mask) > 0.45:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    count_patterns = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 3, 4],
        [2, 3, 4],
    ]
    for _ in range(GEN_RETRIES):
        panel_h = int(rng.integers(4, 7))
        width = int(rng.integers(8, 13))
        counts = count_patterns[int(rng.integers(len(count_patterns)))].copy()
        rng.shuffle(counts)
        objects = _build_panel_objects(rng, panel_h, width, counts)
        if objects is None:
            continue
        grid = _assemble_grid(panel_h, width, objects)
        if _validate_generated(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
