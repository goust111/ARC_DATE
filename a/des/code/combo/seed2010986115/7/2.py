from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    out: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components_from_mask(grid == color):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                }
            )
    return out


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    return m.tobytes(), tuple(map(int, m.shape))


def _touches_corner(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return any((int(r), int(c)) in corners for r, c in coords)


def _parse_palette_patch(patch: np.ndarray) -> dict | None:
    sep = int(Counter(map(int, patch.ravel())).most_common(1)[0][0])
    mapping: dict[frozenset[int], int] = {}
    value_cols: tuple[int, int, int] | None = None
    data_rows: list[int] = []

    for r in range(patch.shape[0]):
        cols = [c for c in range(patch.shape[1]) if int(patch[r, c]) != sep]
        if not cols:
            continue
        if len(cols) != 3:
            return None
        cols_t = tuple(cols)
        if value_cols is None:
            value_cols = cols_t
        elif cols_t != value_cols:
            return None

        a, b, out = (int(patch[r, c]) for c in cols_t)
        if a == sep or b == sep or out == sep or a == b:
            return None
        key = frozenset((a, b))
        if key in mapping:
            return None
        mapping[key] = out
        data_rows.append(r)

    if not mapping or value_cols is None:
        return None

    for r in range(patch.shape[0]):
        if r in data_rows:
            continue
        if np.any(patch[r] != sep):
            return None

    return {"sep": sep, "mapping": mapping, "data_rows": tuple(data_rows), "value_cols": value_cols}


def _find_palette(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], dict[frozenset[int], int]]:
    candidates: list[tuple[int, tuple[int, int, int, int], dict[frozenset[int], int]]] = []
    for coords in _components_from_mask(grid != bg):
        if not _touches_corner(coords, grid.shape):
            continue
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        parsed = _parse_palette_patch(patch)
        if parsed is None:
            continue
        candidates.append((len(coords), bbox, parsed["mapping"]))

    assert candidates, "expected a corner palette"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _area, bbox, mapping = candidates[0]
    return bbox, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shape fragments by exact translated shape and recolor them via a corner palette.

    Input structure
    - Background fills most of the grid.
    - Small 4-connected monochrome fragments are scattered in the main area.
    - A multicolor palette table sits in one corner; each non-separator row encodes A + B -> C.

    Rule
    - Find fragment pairs by normalized shape, ignoring translation only.
    - For each same-shape pair with two different colors, look up the palette row containing those colors.
    - Recolor both fragments to the palette result color.
    - Clear the palette region to background in the output.
    - Leave unpaired fragments unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    palette_bbox, mix_map = _find_palette(g, bg)
    r0, c0, r1, c1 = palette_bbox

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = bg

    scene = g.copy()
    scene[r0 : r1 + 1, c0 : c1 + 1] = bg
    comps = _find_mono_components(scene, bg)

    by_shape: defaultdict[tuple[bytes, tuple[int, int]], list[dict]] = defaultdict(list)
    for comp in comps:
        mask = _component_mask(comp["coords"], comp["bbox"])
        by_shape[_shape_key(mask)].append(comp)

    for group in by_shape.values():
        if len(group) != 2:
            continue
        colors = {int(comp["color"]) for comp in group}
        if len(colors) != 2:
            continue
        result = mix_map.get(frozenset(colors))
        if result is None:
            continue
        for comp in group:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = int(result)

    return out


RNG = np.random.default_rng()
BG = 0
PALETTE_COLORS = tuple(range(1, 10))
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
PAIR_COUNT_RANGE = (2, 4)
SINGLETON_COUNT_RANGE = (1, 3)
MAX_SHAPE_AREA = 6
MAX_SHAPE_SPAN = 4
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300
PALETTE_CORNERS = ("tl", "tr", "bl", "br")


def _random_connected_mask(area: int) -> np.ndarray:
    for _ in range(100):
        h = int(RNG.integers(1, MAX_SHAPE_SPAN + 1))
        w = int(RNG.integers(1, MAX_SHAPE_SPAN + 1))
        if h * w < area:
            continue
        seed = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
        cells = {seed}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = list(frontier)
            cells.add(frontier_list[int(RNG.integers(0, len(frontier_list)))])
        if len(cells) != area:
            continue
        coords = np.asarray(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        return mask
    raise RuntimeError("failed to create connected mask")


def _random_shape() -> np.ndarray:
    area = int(RNG.choice([1, 2, 3, 4, 5, 6], p=[0.08, 0.14, 0.22, 0.22, 0.2, 0.14]))
    return _random_connected_mask(area)


def _sample_unique_shapes(count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    while len(shapes) < count:
        mask = _random_shape()
        key = _shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    return shapes


def _stamp_palette(
    grid: np.ndarray, rows: list[tuple[int, int, int]], sep: int, corner: str
) -> tuple[int, int, int, int]:
    ph = 2 * len(rows) + 1
    pw = 7
    patch = np.full((ph, pw), sep, dtype=int)
    for i, (a, b, out) in enumerate(rows):
        rr = 2 * i + 1
        if int(RNG.integers(0, 2)):
            a, b = b, a
        patch[rr, 1] = a
        patch[rr, 3] = b
        patch[rr, 5] = out

    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - pw

    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return r0, c0, r0 + ph - 1, c0 + pw - 1


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_object(
    grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int
) -> np.ndarray | None:
    h, w = grid.shape
    ph, pw = mask.shape
    candidates = [
        (r, c)
        for r in range(1, h - ph)
        for c in range(1, w - pw)
    ]
    if not candidates:
        return None

    order = RNG.permutation(len(candidates))
    for idx in order[:PLACEMENT_RETRIES]:
        r0, c0 = candidates[int(idx)]
        if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw]):
            continue
        coords = np.argwhere(mask)
        abs_coords = coords + np.array([[r0, c0]])
        grid[abs_coords[:, 0], abs_coords[:, 1]] = int(color)
        _mark_blocked(blocked, (r0, c0, r0 + ph - 1, c0 + pw - 1), pad=1)
        return abs_coords
    return None


def _expected_output(
    grid: np.ndarray, palette_bbox: tuple[int, int, int, int], objects: list[dict]
) -> np.ndarray:
    out = grid.copy()
    bg = _get_bg(grid)
    r0, c0, r1, c1 = palette_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    for obj in objects:
        color = obj["result_color"] if obj["result_color"] is not None else obj["color"]
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(*GRID_H_RANGE))
    w = int(RNG.integers(*GRID_W_RANGE))
    pair_count = int(RNG.integers(*PAIR_COUNT_RANGE))
    has_singletons = bool(RNG.integers(0, 2))
    singleton_count = int(RNG.integers(*SINGLETON_COUNT_RANGE)) if has_singletons else 0
    total_scene_colors = 2 * pair_count + singleton_count

    corner = PALETTE_CORNERS[int(RNG.integers(0, len(PALETTE_CORNERS)))]
    sep = int(RNG.choice(PALETTE_COLORS))
    scene_palette = [c for c in PALETTE_COLORS if c != sep]
    if total_scene_colors > len(scene_palette):
        return None

    grid = np.full((h, w), BG, dtype=int)
    picked = list(map(int, RNG.choice(scene_palette, size=total_scene_colors, replace=False)))
    shapes = _sample_unique_shapes(pair_count + singleton_count)

    pair_rows: list[tuple[int, int, int]] = []
    objects: list[dict] = []
    color_idx = 0
    for i in range(pair_count):
        a = picked[color_idx]
        b = picked[color_idx + 1]
        color_idx += 2
        result_choices = [c for c in PALETTE_COLORS if c not in {sep, a, b}]
        result = int(RNG.choice(result_choices))
        pair_rows.append((a, b, result))
        mask = shapes[i]
        objects.append({"mask": mask, "color": a, "result_color": result})
        objects.append({"mask": mask, "color": b, "result_color": result})

    for i in range(singleton_count):
        color = picked[color_idx]
        color_idx += 1
        objects.append({"mask": shapes[pair_count + i], "color": color, "result_color": None})

    palette_bbox = _stamp_palette(grid, pair_rows, sep, corner)

    blocked = np.zeros_like(grid, dtype=bool)
    _mark_blocked(blocked, palette_bbox, pad=1)

    order = list(RNG.permutation(len(objects)))
    order.sort(key=lambda idx: int(objects[idx]["mask"].sum()), reverse=True)
    placed_objects: list[dict] = []
    for idx in order:
        obj = objects[idx]
        coords = _place_object(grid, blocked, obj["mask"], int(obj["color"]))
        if coords is None:
            return None
        placed = dict(obj)
        placed["coords"] = coords
        placed_objects.append(placed)

    expected = _expected_output(grid, palette_bbox, placed_objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    scene = grid.copy()
    r0, c0, r1, c1 = palette_bbox
    scene[r0 : r1 + 1, c0 : c1 + 1] = BG
    comps = _find_mono_components(scene, BG)
    if len(comps) != len(placed_objects):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
