from __future__ import annotations

from collections import Counter

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
FORWARD_TRANSFORMS = ("rot90cw", "hflip", "vflip")
INVERSE_TRANSFORM = {
    "rot90cw": "rot90ccw",
    "hflip": "hflip",
    "vflip": "vflip",
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert np.any(rows) and np.any(cols), "mask must contain at least one foreground cell"
    return mask[np.ix_(rows, cols)]


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "rot90cw":
        return np.rot90(mask, -1)
    if name == "rot90ccw":
        return np.rot90(mask, 1)
    if name == "hflip":
        return np.fliplr(mask)
    if name == "vflip":
        return np.flipud(mask)
    raise ValueError(f"unknown transform: {name}")


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return (int(mask.shape[0]), int(mask.shape[1]), tuple(int(x) for x in mask.ravel()))


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        color = int(row[0])
        if color != bg and np.all(row == color):
            candidates.append((r, color))
    assert len(candidates) == 1, "expected exactly one full-width non-background separator row"
    return candidates[0]


def _find_components(panel: np.ndarray, bg: int, row_offset: int = 0) -> list[dict[str, object]]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(panel[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            abs_coords = []
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
                abs_coords.append((rr + row_offset, cc))

            comps.append(
                {
                    "color": color,
                    "coords": abs_coords,
                    "bbox": (r0 + row_offset, c0, r1 + row_offset, c1),
                    "mask": mask,
                }
            )

    return comps


def _components_by_color(comps: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    mapping: dict[int, dict[str, object]] = {}
    for comp in comps:
        color = int(comp["color"])
        assert color not in mapping, "object colors must be unique within each panel"
        mapping[color] = comp
    return mapping


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int, bg: int) -> None:
    h, w = mask.shape
    assert 0 <= top <= grid.shape[0] - h
    assert 0 <= left <= grid.shape[1] - w
    for dr, dc in np.argwhere(mask):
        rr = top + int(dr)
        cc = left + int(dc)
        assert int(grid[rr, cc]) == bg, "transformed objects must not overlap"
        grid[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each bottom-panel transform, then apply its inverse to the paired top-panel object.

    Input structure
    - Black background with one full-width gray separator row.
    - The separator splits the grid into top and bottom panels.
    - Each panel contains the same number of monochrome 4-connected objects.
    - Colors are unique inside each panel and pair objects across panels.

    Rule
    - Pair top and bottom objects by color.
    - Identify whether the bottom object is a clockwise 90-degree rotation,
      a horizontal flip, or a vertical flip of its top partner.
    - Replace each top object by the inverse of that transform, keeping the
      original top-left bbox corner fixed.
    - Leave the separator row and the entire bottom panel unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    sep_row, _sep_color = _find_separator_row(g, bg)
    assert 0 < sep_row < g.shape[0] - 1, "separator must split the grid into two non-empty panels"

    top_comps = _find_components(g[:sep_row], bg, row_offset=0)
    bottom_comps = _find_components(g[sep_row + 1 :], bg, row_offset=sep_row + 1)
    assert top_comps and len(top_comps) == len(bottom_comps), "panels must contain the same non-zero number of objects"

    top_by_color = _components_by_color(top_comps)
    bottom_by_color = _components_by_color(bottom_comps)
    assert set(top_by_color) == set(bottom_by_color), "top and bottom panels must contain the same colors"

    out = g.copy()
    for comp in top_comps:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = bg

    for color, top_obj in top_by_color.items():
        top_mask = np.asarray(top_obj["mask"], dtype=bool)
        bottom_mask = np.asarray(bottom_by_color[color]["mask"], dtype=bool)

        matches = [name for name in FORWARD_TRANSFORMS if np.array_equal(_apply_transform(top_mask, name), bottom_mask)]
        assert len(matches) == 1, "each color pair must imply exactly one transform"
        inverse_mask = _apply_transform(top_mask, INVERSE_TRANSFORM[matches[0]])

        r0, c0, _r1, _c1 = top_obj["bbox"]
        _paint_mask(out, inverse_mask, color, int(r0), int(c0), bg)

    return out


# --------------- generator ---------------

RNG = np.random.default_rng()

BG = 0
SEP_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

OBJECT_COUNT_RANGE = (3, 6)
SLOT_WIDTH_OPTIONS = (6, 7)
PANEL_HEIGHT_RANGE = (8, 12)
SHAPE_SIZE_RANGE = (2, 6)
MAX_SHAPE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _random_connected_mask() -> np.ndarray:
    h = int(RNG.integers(SHAPE_SIZE_RANGE[0], SHAPE_SIZE_RANGE[1]))
    w = int(RNG.integers(SHAPE_SIZE_RANGE[0], SHAPE_SIZE_RANGE[1]))
    max_area = h * w - 1
    min_area = 3
    area = int(RNG.integers(min_area, max_area + 1))

    start = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(RNG.integers(0, len(frontier)))]
        cells.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _trim_mask(mask)


def _shape_valid_for_transform(mask: np.ndarray, transform: str) -> bool:
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    if int(mask.sum()) < 3 or np.all(mask):
        return False
    if int(np.any(mask, axis=0).sum()) < 2 or int(np.any(mask, axis=1).sum()) < 2:
        return False

    forward = _apply_transform(mask, transform)
    inverse = _apply_transform(mask, INVERSE_TRANSFORM[transform])
    if np.array_equal(mask, forward) or np.array_equal(mask, inverse):
        return False

    matches = [name for name in FORWARD_TRANSFORMS if np.array_equal(_apply_transform(mask, name), forward)]
    return matches == [transform]


def _make_shape(transform: str, used_keys: set[tuple[int, int, tuple[int, ...]]]) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        mask = _random_connected_mask()
        key = _mask_key(mask)
        if key in used_keys:
            continue
        if not _shape_valid_for_transform(mask, transform):
            continue
        used_keys.add(key)
        return mask
    return None


def _build_objects(count: int) -> list[dict[str, object]] | None:
    colors = [int(x) for x in RNG.choice(np.array(COLOR_POOL), size=count, replace=False)]
    transforms = list(FORWARD_TRANSFORMS)
    while len(transforms) < count:
        transforms.append(str(RNG.choice(FORWARD_TRANSFORMS)))
    RNG.shuffle(transforms)

    used_keys: set[tuple[int, int, tuple[int, ...]]] = set()
    objects: list[dict[str, object]] = []
    for color, transform in zip(colors, transforms):
        top_in = _make_shape(transform, used_keys)
        if top_in is None:
            return None
        bottom_in = _apply_transform(top_in, transform)
        top_out = _apply_transform(top_in, INVERSE_TRANSFORM[transform])
        objects.append(
            {
                "color": color,
                "transform": transform,
                "top_in": top_in,
                "bottom_in": bottom_in,
                "top_out": top_out,
                "top_reservation": (
                    max(int(top_in.shape[0]), int(top_out.shape[0])),
                    max(int(top_in.shape[1]), int(top_out.shape[1])),
                ),
            }
        )
    return objects


def _place_in_slots(
    count: int,
    slot_width: int,
    panel_height: int,
    sizes: list[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]]:
    right_pad = int(RNG.integers(0, 3))
    width = 1 + count * slot_width + right_pad
    positions = [(-1, -1)] * count
    order = [int(x) for x in RNG.permutation(count)]

    for slot_idx, obj_idx in enumerate(order):
        h, w = sizes[obj_idx]
        assert h <= panel_height and w <= slot_width
        r = int(RNG.integers(0, panel_height - h + 1))
        c0 = 1 + slot_idx * slot_width
        c = c0 + int(RNG.integers(0, slot_width - w + 1))
        positions[obj_idx] = (r, c)

    return width, positions


def _draw_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    for dr, dc in np.argwhere(mask):
        grid[top + int(dr), left + int(dc)] = color


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1]))
    objects = _build_objects(count)
    if objects is None:
        return None

    slot_width = int(RNG.choice(SLOT_WIDTH_OPTIONS))
    max_top_h = max(int(obj["top_reservation"][0]) for obj in objects)
    max_bottom_h = max(int(np.asarray(obj["bottom_in"]).shape[0]) for obj in objects)
    top_h = int(RNG.integers(max(max_top_h + 2, PANEL_HEIGHT_RANGE[0]), PANEL_HEIGHT_RANGE[1]))
    bottom_h = int(RNG.integers(max(max_bottom_h + 2, PANEL_HEIGHT_RANGE[0]), PANEL_HEIGHT_RANGE[1]))

    top_sizes = [tuple(map(int, obj["top_reservation"])) for obj in objects]
    bottom_sizes = [tuple(map(int, np.asarray(obj["bottom_in"]).shape)) for obj in objects]
    width_top, top_pos = _place_in_slots(count, slot_width, top_h, top_sizes)
    width_bottom, bottom_pos = _place_in_slots(count, slot_width, bottom_h, bottom_sizes)
    width = max(width_top, width_bottom)

    grid = np.full((top_h + 1 + bottom_h, width), BG, dtype=int)
    grid[top_h, :] = SEP_COLOR

    for obj, (r, c) in zip(objects, top_pos):
        _draw_mask(grid, np.asarray(obj["top_in"], dtype=bool), int(obj["color"]), r, c)
    for obj, (r, c) in zip(objects, bottom_pos):
        _draw_mask(grid, np.asarray(obj["bottom_in"], dtype=bool), int(obj["color"]), top_h + 1 + r, c)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
