from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
CROSS = 5
PANEL_NAMES = ("TL", "TR", "BL", "BR")
PANEL_COLORS = (1, 2, 3, 4)
LAYER_COUNTS = (3, 5, 7, 9)
GEN_MAX_ATTEMPTS = 600
BASE_BOX = 7
BASE_SPECIALS = {
    "x35": (3, 2),
    "x57": (3, 3),
    "x79": (3, 4),
    "e3": (1, 2),
    "e5": (1, 4),
    "e7": (1, 5),
    "e9": (5, 5),
}
REQUIRED_KEYS = {
    3: ("e3", "x35"),
    5: ("e5", "x35", "x57"),
    7: ("e7", "x57", "x79"),
    9: ("e9", "x79"),
}
DISCOURAGED_ORDERS = {
    ("TL", "TR", "BL", "BR"),
    ("BL", "TR", "TL", "BR"),
    ("TL", "BL", "TR", "BR"),
}


def _neighbors4(shape: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _component_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for start in map(tuple, np.argwhere(mask)):
        if seen[start]:
            continue
        count += 1
        q = deque([start])
        seen[start] = True
        while q:
            cell = q.popleft()
            for nb in _neighbors4(mask.shape, cell):
                if mask[nb] and not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
    return count


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    rows = np.where(np.all(g == CROSS, axis=1))[0]
    cols = np.where(np.all(g == CROSS, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full grey cross"
    return int(rows[0]), int(cols[0])


def _split_panels(grid: np.ndarray) -> dict[str, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    cross_r, cross_c = _find_cross(g)
    assert cross_r > 0 and cross_c > 0
    panels = {
        "TL": g[:cross_r, :cross_c],
        "TR": g[:cross_r, cross_c + 1 :],
        "BL": g[cross_r + 1 :, :cross_c],
        "BR": g[cross_r + 1 :, cross_c + 1 :],
    }
    shapes = {panel.shape for panel in panels.values()}
    assert len(shapes) == 1, "expected equal-size panels"
    return panels


def _panel_count(panel: np.ndarray) -> int:
    return int(np.count_nonzero(panel != BG))


def _overlay_in_order(panels: dict[str, np.ndarray], order: tuple[str, ...] | list[str]) -> np.ndarray:
    shape = next(iter(panels.values())).shape
    out = np.full(shape, BG, dtype=int)
    for name in order:
        panel = panels[name]
        mask = panel != BG
        out[mask] = panel[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the four cross-split panels after sorting them by foreground-pixel count.

    Input structure:
    - A full grey row and full grey column form a cross that splits the odd-sized grid into four equal panels.
    - Each panel has black background plus exactly one distinct foreground color from the non-grey colors.
    - The four panel foreground shapes have different non-black pixel counts.

    Rule:
    - Remove the grey cross and crop the four panels into the same local coordinate frame.
    - Count the non-black pixels in each panel.
    - Sort the panels from fewest foreground pixels to most foreground pixels.
    - Transparently overlay them in that order, so later non-black pixels overwrite earlier ones.

    Key insight:
    - The collapsing step is the same local-coordinate overlay as a quadrant-composition task, but the layer order is determined by counted content, not by quadrant position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels = _split_panels(g)
    counts: dict[str, int] = {}
    for name, panel in panels.items():
        colors = set(map(int, np.unique(panel)))
        assert colors <= set(PANEL_COLORS) | {BG}, f"unexpected colors in {name}: {colors}"
        fg_colors = colors - {BG}
        assert len(fg_colors) == 1, f"expected one foreground color in {name}"
        assert _component_count(panel != BG) == 1, f"expected one 4-connected shape in {name}"
        counts[name] = _panel_count(panel)

    assert len(set(counts.values())) == 4, "foreground counts must be unique"
    order = tuple(sorted(PANEL_NAMES, key=lambda name: counts[name]))
    return _overlay_in_order(panels, order)


def _transform_point(pt: tuple[int, int], turns: int, mirror: bool) -> tuple[int, int]:
    r, c = pt
    if mirror:
        c = BASE_BOX - 1 - c
    for _ in range(turns % 4):
        r, c = c, BASE_BOX - 1 - r
    return int(r), int(c)


def _special_layout(panel_size: int) -> dict[str, tuple[int, int]]:
    turns = random.randrange(4)
    mirror = bool(random.randrange(2))
    transformed = {key: _transform_point(pt, turns, mirror) for key, pt in BASE_SPECIALS.items()}
    max_r = max(r for r, _ in transformed.values())
    max_c = max(c for _, c in transformed.values())
    row_shift = random.randint(0, panel_size - 1 - max_r)
    col_shift = random.randint(0, panel_size - 1 - max_c)
    return {key: (r + row_shift, c + col_shift) for key, (r, c) in transformed.items()}


def _bfs_path(
    shape: tuple[int, int],
    start_cells: set[tuple[int, int]],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if goal in start_cells:
        return [goal]
    q = deque()
    parents: dict[tuple[int, int], tuple[int, int] | None] = {}
    for cell in random.sample(list(start_cells), len(start_cells)):
        q.append(cell)
        parents[cell] = None
    while q:
        cell = q.popleft()
        neighbors = _neighbors4(shape, cell)
        random.shuffle(neighbors)
        for nb in neighbors:
            if nb in blocked or nb in parents:
                continue
            parents[nb] = cell
            if nb == goal:
                path = [goal]
                cur = goal
                while parents[cur] is not None:
                    cur = parents[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(nb)
    return None


def _build_connected_mask(
    shape: tuple[int, int],
    target_count: int,
    required: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    preferred: set[tuple[int, int]],
) -> np.ndarray | None:
    if len(required) > target_count:
        return None
    if required & blocked:
        return None

    required_list = random.sample(list(required), len(required))
    cells = {required_list[0]}
    for goal in required_list[1:]:
        path = _bfs_path(shape, cells, goal, blocked)
        if path is None:
            return None
        cells.update(path)
        if len(cells) > target_count:
            return None

    while len(cells) < target_count:
        frontier = []
        for cell in cells:
            for nb in _neighbors4(shape, cell):
                if nb not in cells and nb not in blocked:
                    frontier.append(nb)
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            return None

        def _weight(cell: tuple[int, int]) -> tuple[int, int, float]:
            overlap_bias = 3 if cell in preferred else 0
            center_bias = -abs(cell[0] - shape[0] / 2.0) - abs(cell[1] - shape[1] / 2.0)
            return (overlap_bias, random.randint(0, 2), center_bias)

        frontier.sort(key=_weight, reverse=True)
        top_band = frontier[: min(len(frontier), 5)]
        cells.add(random.choice(top_band))

    mask = np.zeros(shape, dtype=bool)
    for cell in cells:
        mask[cell] = True
    if int(mask.sum()) != target_count:
        return None
    if _component_count(mask) != 1:
        return None
    if not required <= set(cells):
        return None
    return mask


def _make_masks(panel_size: int) -> dict[int, np.ndarray] | None:
    specials = _special_layout(panel_size)
    all_special_cells = set(specials.values())
    masks: dict[int, np.ndarray] = {}
    preferred: set[tuple[int, int]] = set()

    for count in LAYER_COUNTS:
        required = {specials[key] for key in REQUIRED_KEYS[count]}
        blocked = all_special_cells - required
        mask = _build_connected_mask(
            shape=(panel_size, panel_size),
            target_count=count,
            required=required,
            blocked=blocked,
            preferred=preferred,
        )
        if mask is None:
            return None
        masks[count] = mask
        preferred |= set(map(tuple, np.argwhere(mask)))

    return masks


def _render_input(
    panel_size: int,
    count_by_panel: dict[str, int],
    color_by_panel: dict[str, int],
    masks_by_count: dict[int, np.ndarray],
) -> np.ndarray:
    full_size = 2 * panel_size + 1
    grid = np.full((full_size, full_size), BG, dtype=int)
    grid[panel_size, :] = CROSS
    grid[:, panel_size] = CROSS

    panel_slices = {
        "TL": (slice(0, panel_size), slice(0, panel_size)),
        "TR": (slice(0, panel_size), slice(panel_size + 1, full_size)),
        "BL": (slice(panel_size + 1, full_size), slice(0, panel_size)),
        "BR": (slice(panel_size + 1, full_size), slice(panel_size + 1, full_size)),
    }
    for name in PANEL_NAMES:
        count = count_by_panel[name]
        color = color_by_panel[name]
        mask = masks_by_count[count]
        panel = np.full((panel_size, panel_size), BG, dtype=int)
        panel[mask] = color
        rs, cs = panel_slices[name]
        grid[rs, cs] = panel
    return grid


def _parse_and_validate_structure(grid: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    panels = _split_panels(grid)
    counts = {}
    seen_colors = set()
    for name, panel in panels.items():
        colors = set(map(int, np.unique(panel)))
        fg = colors - {BG}
        if len(fg) != 1:
            raise AssertionError(f"{name} does not contain exactly one foreground color")
        color = next(iter(fg))
        if color in seen_colors:
            raise AssertionError("panel colors must be distinct")
        seen_colors.add(color)
        counts[name] = _panel_count(panel)
        if counts[name] not in set(LAYER_COUNTS):
            raise AssertionError("unexpected foreground count")
        if _component_count(panel != BG) != 1:
            raise AssertionError("panel shape must be a single 4-connected component")
    if sorted(counts.values()) != list(LAYER_COUNTS):
        raise AssertionError("panel counts must be exactly 3, 5, 7, 9")
    return panels, counts


def _nondegenerate(grid: np.ndarray) -> bool:
    try:
        panels, counts = _parse_and_validate_structure(grid)
    except Exception:
        return False

    correct_order = tuple(sorted(PANEL_NAMES, key=lambda name: counts[name]))
    if correct_order in DISCOURAGED_ORDERS:
        return False

    expected = _overlay_in_order(panels, correct_order)
    if not np.array_equal(main(grid), expected):
        return False

    visible_colors = set(map(int, np.unique(expected))) - {BG}
    if len(visible_colors) != 4:
        return False

    visible_panels = 0
    cover = np.zeros(expected.shape, dtype=int)
    for name, panel in panels.items():
        mask = panel != BG
        cover += mask.astype(int)
        color = int(next(iter(set(map(int, np.unique(panel))) - {BG})))
        if np.any(expected == color):
            visible_panels += 1
        else:
            return False
    if visible_panels != 4:
        return False
    if int(np.count_nonzero(cover >= 2)) < 3:
        return False

    matches = 0
    for order in itertools.permutations(PANEL_NAMES):
        out = _overlay_in_order(panels, order)
        if np.array_equal(out, expected):
            matches += 1
    if matches != 1:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    panel_size = random.randint(7, 10)
    masks_by_count = _make_masks(panel_size)
    if masks_by_count is None:
        return None

    counts_perm = random.sample(list(LAYER_COUNTS), len(LAYER_COUNTS))
    colors_perm = random.sample(list(PANEL_COLORS), len(PANEL_COLORS))
    count_by_panel = {name: count for name, count in zip(PANEL_NAMES, counts_perm)}
    color_by_panel = {name: color for name, color in zip(PANEL_NAMES, colors_perm)}

    grid = _render_input(panel_size, count_by_panel, color_by_panel, masks_by_count)
    if _nondegenerate(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
