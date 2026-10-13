from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BG = 0
CTRL = 6
FG_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            stack = [(int(r), int(c))]
            labels[r, c] = int(next_id)
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = int(next_id)
                        stack.append((int(nr), int(nc)))
    return labels, int(next_id)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    labels, count = _label_mask(grid == int(color))
    return [np.argwhere(labels == idx) for idx in range(1, int(count) + 1)]


def _draw_rays(grid: np.ndarray, ctrl: int, ctrl_pts: list[tuple[int, int]]) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    bg = _get_bg(g)
    for r, c in ctrl_pts:
        if r == 0 or r == h - 1:
            step = 1 if r == 0 else -1
            for rr in range(int(r), h if step > 0 else -1, step):
                if g[rr, c] == bg:
                    g[rr, c] = int(ctrl)
        else:
            step = 1 if c == 0 else -1
            for cc in range(int(c), w if step > 0 else -1, step):
                if g[r, cc] == bg:
                    g[r, cc] = int(ctrl)
    return g


def _region_edge_sets(ray_grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[int, set[str]]]:
    labels, count = _label_mask(ray_grid == int(bg))
    h, w = ray_grid.shape
    edges: dict[int, set[str]] = {}
    for idx in range(1, int(count) + 1):
        coords = np.argwhere(labels == idx)
        touched: set[str] = set()
        if np.any(coords[:, 0] == 0):
            touched.add("top")
        if np.any(coords[:, 0] == h - 1):
            touched.add("bottom")
        if np.any(coords[:, 1] == 0):
            touched.add("left")
        if np.any(coords[:, 1] == w - 1):
            touched.add("right")
        edges[int(idx)] = touched
    return labels, edges


def _component_region_ids(coords: np.ndarray, bg_labels: np.ndarray) -> set[int]:
    h, w = bg_labels.shape
    ids: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and bg_labels[nr, nc] > 0:
                ids.add(int(bg_labels[nr, nc]))
    return ids


def _render_nested(colors: list[int]) -> np.ndarray:
    n = len(colors)
    if n <= 0:
        raise AssertionError("expected at least one kept component")
    size = 2 * n if n % 2 == 0 else 2 * n - 1
    out = np.zeros((size, size), dtype=int)
    for layer, color in enumerate(colors):
        end = size - layer - 1
        out[layer : end + 1, layer] = int(color)
        out[layer : end + 1, end] = int(color)
        out[layer, layer : end + 1] = int(color)
        out[end, layer : end + 1] = int(color)
    return out


def _kept_components(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    border_values = set(map(int, g[0])) | set(map(int, g[h - 1])) | set(map(int, g[:, 0])) | set(map(int, g[:, w - 1]))
    border_values.discard(int(bg))
    if not border_values:
        raise AssertionError("expected at least one border control color")
    ctrl = int(sorted(border_values)[0])
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl)) if r in (0, h - 1) or c in (0, w - 1)]
    if not ctrl_pts:
        raise AssertionError("expected border control pixels")

    control_edges = {("top" if r == 0 else "bottom" if r == h - 1 else "left" if c == 0 else "right") for r, c in ctrl_pts}
    ray_grid = _draw_rays(g, ctrl, ctrl_pts)
    bg_labels, region_edges = _region_edge_sets(ray_grid, bg)

    kept: list[tuple[int, int]] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg), int(ctrl)}):
        comps = _components_of_color(g, int(color))
        if len(comps) != 1:
            raise AssertionError("each foreground color must form exactly one component")
        coords = comps[0]
        region_ids = _component_region_ids(coords, bg_labels)
        if len(region_ids) != 1:
            raise AssertionError("each component must border exactly one cut region")
        region_id = next(iter(region_ids))
        if region_edges[int(region_id)] & control_edges:
            kept.append((int(color), int(coords.shape[0])))

    if len(kept) < 2:
        raise AssertionError("expected at least two kept components")
    kept.sort(key=lambda item: (-int(item[1]), int(item[0])))
    areas = [area for _color, area in kept]
    if len(set(areas)) != len(areas):
        raise AssertionError("kept component areas must be distinct")
    return kept


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the central multi-color sprite by border-control rays, keep only border-adjacent fragments, and redraw their colors as area-ordered concentric square shells.

    Input structure:
    - The grid has a dominant black background, 3-4 colored foreground components near the center, and 1-2 magenta control pixels on distinct border edges.
    - Each non-magenta foreground color forms exactly one 4-connected component, and each component lies entirely inside one region induced by the control rays.
    - A control pixel on the top or bottom edge defines a vertical cut column; a control pixel on the left or right edge defines a horizontal cut row.

    Rule:
    - Extend a virtual magenta ray inward from every control pixel along its row or column, recoloring only background cells and skipping over foreground cells until the opposite border.
    - Treat those intermittent rays as cut lines on the background, and identify which background regions touch at least one border edge that contains a control pixel.
    - Keep exactly the foreground components adjacent to such control-edge regions and discard the rest.
    - Count the area of each kept component, sort the kept colors from largest area to smallest, and redraw them as tight concentric square shells from outside to inside.

    Key insight:
    - The cuts are only a filter: they decide which colored fragments survive, while the surviving fragment areas alone determine the final outer-to-inner color order.
    """
    kept = _kept_components(np.asarray(grid, dtype=int))
    return _render_nested([color for color, _area in kept])


def _region_boxes(
    sprite_bbox: tuple[int, int, int, int],
    row_split: int | None,
    col_split: int | None,
) -> dict[str, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, sprite_bbox)
    if row_split is None and col_split is None:
        raise AssertionError("at least one split is required")
    if row_split is None:
        return {
            "L": (int(r0), int(c0), int(r1), int(col_split - 1)),
            "R": (int(r0), int(col_split + 1), int(r1), int(c1)),
        }
    if col_split is None:
        return {
            "T": (int(r0), int(c0), int(row_split - 1), int(c1)),
            "B": (int(row_split + 1), int(c0), int(r1), int(c1)),
        }
    return {
        "TL": (int(r0), int(c0), int(row_split - 1), int(col_split - 1)),
        "TR": (int(r0), int(col_split + 1), int(row_split - 1), int(c1)),
        "BL": (int(row_split + 1), int(c0), int(r1), int(col_split - 1)),
        "BR": (int(row_split + 1), int(col_split + 1), int(r1), int(c1)),
    }


def _kept_region_names(
    control_edges: set[str],
    region_names: list[str],
    row_split: int | None,
    col_split: int | None,
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for name in region_names:
        if row_split is None:
            touches = {"top", "bottom"}
            touches.add("left" if name == "L" else "right")
        elif col_split is None:
            touches = {"left", "right"}
            touches.add("top" if name == "T" else "bottom")
        else:
            touches = set()
            if "T" in name:
                touches.add("top")
            if "B" in name:
                touches.add("bottom")
            if "L" in name:
                touches.add("left")
            if "R" in name:
                touches.add("right")
        (kept if touches & control_edges else dropped).append(name)
    return kept, dropped


def _split_region_for_count(
    region: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, region)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if count == 1:
        return [region]
    if count != 2:
        raise AssertionError("generator only supports up to two components per region")
    if h >= w:
        cut = r0 + h // 2
        if h >= 5:
            return [(int(r0), int(c0), int(cut - 1), int(c1)), (int(cut + 1), int(c0), int(r1), int(c1))]
        return [(int(r0), int(c0), int(cut), int(c1)), (int(cut + 1), int(c0), int(r1), int(c1))]
    cut = c0 + w // 2
    if w >= 5:
        return [(int(r0), int(c0), int(r1), int(cut - 1)), (int(r0), int(cut + 1), int(r1), int(c1))]
    return [(int(r0), int(c0), int(r1), int(cut)), (int(r0), int(cut + 1), int(r1), int(c1))]


def _grow_polyomino(area: int, box_h: int, box_w: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((int(box_h), int(box_w)), dtype=bool)
    seed = (int(rng.integers(0, box_h)), int(rng.integers(0, box_w)))
    mask[seed] = True
    cells = {seed}
    while len(cells) < int(area):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        nr, nc = list(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add((int(nr), int(nc)))
        mask[nr, nc] = True
    return mask


def _place_component_in_slot(
    grid: np.ndarray,
    slot: tuple[int, int, int, int],
    color: int,
    area: int,
    rng: np.random.Generator,
) -> bool:
    r0, c0, r1, c1 = map(int, slot)
    slot_h = int(r1 - r0 + 1)
    slot_w = int(c1 - c0 + 1)
    feasible: list[tuple[int, int]] = []
    for h in range(2, slot_h + 1):
        for w in range(2, slot_w + 1):
            if h * w >= int(area) and h * w <= int(area) + 4:
                feasible.append((int(h), int(w)))
    if not feasible:
        for h in range(2, slot_h + 1):
            for w in range(2, slot_w + 1):
                if h * w >= int(area):
                    feasible.append((int(h), int(w)))
    if not feasible:
        return False
    rng.shuffle(feasible)
    for box_h, box_w in feasible:
        for _ in range(30):
            top = int(rng.integers(r0, r1 - box_h + 2))
            left = int(rng.integers(c0, c1 - box_w + 2))
            shape = _grow_polyomino(int(area), int(box_h), int(box_w), rng)
            if shape is None:
                continue
            patch = grid[top : top + box_h, left : left + box_w]
            if np.any((patch != BG) & shape):
                continue
            patch[shape] = int(color)
            return True
    return False


def _allocate_regions(
    use_two_controls: bool,
    num_colors: int,
    kept_regions: list[str],
    dropped_regions: list[str],
    rng: np.random.Generator,
) -> dict[str, int]:
    counts = {name: 0 for name in kept_regions + dropped_regions}
    remaining = int(num_colors)
    if use_two_controls and dropped_regions and rng.random() < 0.55:
        counts[dropped_regions[0]] = 1
        remaining -= 1

    if use_two_controls:
        chosen = kept_regions.copy()
        rng.shuffle(chosen)
        for name in chosen[:remaining]:
            counts[name] += 1
        remaining = 0
    else:
        order = kept_regions.copy()
        rng.shuffle(order)
        for name in order:
            if remaining == 0:
                break
            counts[name] += 1
            remaining -= 1
        while remaining > 0:
            choices = [name for name in kept_regions if counts[name] < 2]
            if not choices:
                return {}
            name = choices[int(rng.integers(0, len(choices)))]
            counts[name] += 1
            remaining -= 1

    if sum(counts[name] for name in kept_regions) < 2:
        return {}
    if use_two_controls and any(counts[name] > 1 for name in kept_regions + dropped_regions):
        return {}
    return counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(17, 22))
    w = int(rng.integers(17, 22))
    grid = np.zeros((h, w), dtype=int)

    sprite_h = int(rng.integers(8, 11))
    sprite_w = int(rng.integers(8, 11))
    max_r0 = h - sprite_h - 3
    max_c0 = w - sprite_w - 3
    if max_r0 < 3 or max_c0 < 3:
        return None
    r0 = int(rng.integers(3, max_r0 + 1))
    c0 = int(rng.integers(3, max_c0 + 1))
    r1 = r0 + sprite_h - 1
    c1 = c0 + sprite_w - 1

    use_two_controls = bool(rng.random() < 0.55)
    row_split: int | None = None
    col_split: int | None = None
    control_edges: set[str] = set()
    ctrl_pts: list[tuple[int, int]] = []

    if use_two_controls:
        row_split = int(rng.integers(r0 + 3, r1 - 1))
        col_split = int(rng.integers(c0 + 3, c1 - 1))
        vert_edge = "top" if rng.random() < 0.5 else "bottom"
        horiz_edge = "left" if rng.random() < 0.5 else "right"
        ctrl_pts.append((0 if vert_edge == "top" else h - 1, int(col_split)))
        ctrl_pts.append((int(row_split), 0 if horiz_edge == "left" else w - 1))
        control_edges = {vert_edge, horiz_edge}
    else:
        edge = ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
        control_edges = {edge}
        if edge in {"top", "bottom"}:
            col_split = int(rng.integers(c0 + 3, c1 - 1))
            ctrl_pts.append((0 if edge == "top" else h - 1, int(col_split)))
        else:
            row_split = int(rng.integers(r0 + 3, r1 - 1))
            ctrl_pts.append((int(row_split), 0 if edge == "left" else w - 1))

    regions = _region_boxes((r0, c0, r1, c1), row_split, col_split)
    kept_regions, dropped_regions = _kept_region_names(control_edges, list(regions), row_split, col_split)

    num_colors = int(rng.integers(3, 5))
    counts = _allocate_regions(use_two_controls, num_colors, kept_regions, dropped_regions, rng)
    if not counts:
        return None

    colors = list(rng.choice(np.array(FG_COLORS, dtype=int), size=num_colors, replace=False))
    areas = list(rng.choice(np.array([3, 4, 5, 6, 7], dtype=int), size=num_colors, replace=False))
    rng.shuffle(colors)
    rng.shuffle(areas)
    specs = list(zip(colors, areas))

    slots: list[tuple[str, tuple[int, int, int, int]]] = []
    for name, count in counts.items():
        if count == 0:
            continue
        for slot in _split_region_for_count(regions[name], count):
            area = (slot[2] - slot[0] + 1) * (slot[3] - slot[1] + 1)
            if area < 4:
                return None
            slots.append((name, slot))
    if len(slots) != num_colors:
        return None

    for (name, slot), (color, area) in zip(slots, specs):
        if area >= (slot[2] - slot[0] + 1) * (slot[3] - slot[1] + 1):
            return None
        if not _place_component_in_slot(grid, slot, int(color), int(area), rng):
            return None

    for r, c in ctrl_pts:
        grid[int(r), int(c)] = CTRL

    try:
        kept = _kept_components(grid)
        predicted = main(grid)
    except Exception:
        return None

    if len(kept) != len({color for color, _area in kept}):
        return None
    if not np.any(grid[1:-1, 1:-1] != BG):
        return None
    if predicted.ndim != 2 or max(predicted.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
