from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SEP = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400

PANEL_COUNT_RANGE = (2, 4)
PANEL_HEIGHT_RANGE = (10, 16)
PANEL_WIDTH_RANGE = (9, 15)
SEPARATOR_WIDTH_CHOICES = (1, 2)
SHAPE_AREA_RANGE = (3, 8)
SHAPE_SPAN_MAX = 5
LAYOUT_PATTERNS = ((2,), (3,), (2, 2), (4,))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    variants: list[tuple[bytes, tuple[int, int]]] = []
    current = np.asarray(mask, dtype=bool)
    for _ in range(4):
        variants.append((current.tobytes(), tuple(map(int, current.shape))))
        flipped = np.fliplr(current)
        variants.append((flipped.tobytes(), tuple(map(int, flipped.shape))))
        current = np.rot90(current)
    return min(variants)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    transformed = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _find_separator_runs(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    cols = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != bg and bool(np.all(col == col[0])):
            cols.append(c)
    assert cols, "expected full-height separator columns"

    runs = []
    start = cols[0]
    prev = cols[0]
    sep_color = int(grid[0, cols[0]])
    for c in cols[1:]:
        assert int(grid[0, c]) == sep_color, "separator columns must share one color"
        if c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        start = prev = c
    runs.append((start, prev))
    return runs


def _split_panels(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    runs = _find_separator_runs(grid, bg)
    panels = []
    left = 0
    for start, end in runs:
        if left < start:
            panels.append(grid[:, left:start].copy())
        left = end + 1
    if left < grid.shape[1]:
        panels.append(grid[:, left:].copy())

    assert len(panels) >= 2, "expected multiple panels"
    widths = {panel.shape[1] for panel in panels}
    assert len(widths) == 1, "panels must have equal width"
    return panels


def _component_patch(panel: np.ndarray, coords: list[tuple[int, int]], color: int, bg: int) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    patch = np.full((h, w), bg, dtype=int)
    patch[mask] = color
    return {
        "bbox": (r0, c0, r1, c1),
        "color": color,
        "mask": mask,
        "patch": patch,
        "shape_key": _canonical_shape_key(mask),
    }


def _find_objects(panel: np.ndarray, bg: int) -> list[dict]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(panel[r, c]) == bg:
                continue
            color = int(panel[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(panel[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            objects.append(_component_patch(panel, coords, color, bg))
    assert objects, "each panel must contain objects"
    return objects


def _odd_object_patch(panel: np.ndarray, bg: int) -> np.ndarray:
    objects = _find_objects(panel, bg)
    counts = Counter(obj["shape_key"] for obj in objects)
    odd = [obj for obj in objects if counts[obj["shape_key"]] == 1]
    assert len(odd) == 1, "each panel must contain exactly one odd-shape object"
    return odd[0]["patch"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each panel, find the single object whose shape matches no other object
    under rotation/flip, then stack those odd objects' bounding boxes vertically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    panels = _split_panels(g, bg)
    odd_patches = [_odd_object_patch(panel, bg) for panel in panels]

    out_h = sum(patch.shape[0] for patch in odd_patches) + len(odd_patches) - 1
    out_w = max(patch.shape[1] for patch in odd_patches)
    out = np.full((out_h, out_w), bg, dtype=int)

    r = 0
    for patch in odd_patches:
        h, w = patch.shape
        out[r : r + h, :w] = patch
        r += h + 1
    return out


def _random_connected_shape() -> np.ndarray:
    for _ in range(200):
        target_area = random.randint(*SHAPE_AREA_RANGE)
        coords = {(0, 0)}
        while len(coords) < target_area:
            frontier = set()
            for r, c in coords:
                for dr, dc in NEIGHBORS4:
                    candidate = (r + dr, c + dc)
                    if candidate in coords:
                        continue
                    trial = coords | {candidate}
                    rows = [rr for rr, _ in trial]
                    cols = [cc for _, cc in trial]
                    if max(rows) - min(rows) + 1 > SHAPE_SPAN_MAX:
                        continue
                    if max(cols) - min(cols) + 1 > SHAPE_SPAN_MAX:
                        continue
                    frontier.add(candidate)
            if not frontier:
                break
            coords.add(random.choice(tuple(frontier)))
        if len(coords) != target_area:
            continue
        mask = _normalize_coords(coords)
        if max(mask.shape) < 2:
            continue
        return mask
    raise RuntimeError("failed to sample a connected shape")


def _shape_template_with_key(excluded: set[tuple[bytes, tuple[int, int]]]) -> tuple[np.ndarray, tuple[bytes, tuple[int, int]]]:
    for _ in range(200):
        mask = _random_connected_shape()
        key = _canonical_shape_key(mask)
        if key not in excluded:
            return mask, key
    raise RuntimeError("failed to find a distinct shape template")


def _paint_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _dilate(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _place_panel_objects(height: int, width: int, objects: list[dict]) -> np.ndarray | None:
    panel = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(objects)), key=lambda i: int(objects[i]["mask"].sum()), reverse=True)

    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        h, w = mask.shape
        valid = []
        dilated = _dilate(occupied)
        for r0 in range(height - h + 1):
            for c0 in range(width - w + 1):
                if bool((dilated[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                    continue
                valid.append((r0, c0))
        if not valid:
            return None
        r0, c0 = random.choice(valid)
        panel[r0 : r0 + h, c0 : c0 + w][mask] = obj["color"]
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask
        obj["patch"] = _paint_patch(mask, obj["color"])
    return panel


def _build_panel(pattern: tuple[int, ...], height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = 1 + sum(pattern)
    colors = random.sample(OBJECT_COLORS, n_objects)
    used_keys: set[tuple[bytes, tuple[int, int]]] = set()

    odd_mask, odd_key = _shape_template_with_key(used_keys)
    used_keys.add(odd_key)
    objects = [{"mask": _transform_mask(odd_mask), "color": colors[0], "odd": True}]

    color_index = 1
    for group_size in pattern:
        template, key = _shape_template_with_key(used_keys)
        used_keys.add(key)
        for _ in range(group_size):
            objects.append(
                {
                    "mask": _transform_mask(template),
                    "color": colors[color_index],
                    "odd": False,
                }
            )
            color_index += 1

    random.shuffle(objects)
    panel = _place_panel_objects(height, width, objects)
    if panel is None:
        return None

    odd_patch = None
    for obj in objects:
        if obj["odd"]:
            odd_patch = obj["patch"]
            break
    assert odd_patch is not None
    return panel, odd_patch


def _expected_output(odd_patches: list[np.ndarray]) -> np.ndarray:
    out_h = sum(patch.shape[0] for patch in odd_patches) + len(odd_patches) - 1
    out_w = max(patch.shape[1] for patch in odd_patches)
    out = np.full((out_h, out_w), BG, dtype=int)
    r = 0
    for patch in odd_patches:
        h, w = patch.shape
        out[r : r + h, :w] = patch
        r += h + 1
    return out


def _compose_input(panels: list[np.ndarray], sep_w: int) -> np.ndarray:
    height = panels[0].shape[0]
    width = len(panels) * panels[0].shape[1] + (len(panels) - 1) * sep_w
    grid = np.full((height, width), BG, dtype=int)

    c = 0
    for i, panel in enumerate(panels):
        w = panel.shape[1]
        grid[:, c : c + w] = panel
        c += w
        if i < len(panels) - 1:
            grid[:, c : c + sep_w] = SEP
            c += sep_w
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    height = random.randint(*PANEL_HEIGHT_RANGE)
    panel_width = random.randint(*PANEL_WIDTH_RANGE)
    sep_w = random.choice(SEPARATOR_WIDTH_CHOICES)

    panels = []
    odd_patches = []
    for _ in range(panel_count):
        pattern = random.choice(LAYOUT_PATTERNS)
        built = _build_panel(pattern, height, panel_width)
        if built is None:
            return None
        panel, odd_patch = built
        panels.append(panel)
        odd_patches.append(odd_patch)

    grid = _compose_input(panels, sep_w)
    expected = _expected_output(odd_patches)

    if int(np.count_nonzero(grid == BG)) <= max(int(np.count_nonzero(grid == SEP)), 0):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
