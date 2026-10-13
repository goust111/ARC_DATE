import random
from collections import Counter

import numpy as np


BG = 0
NEUTRAL = 5
PALETTE = (1, 2, 3, 4)
ZONE_H = 9
ZONE_W = 7
GRAY_ORIGIN_R = 3
GRAY_ORIGIN_C = 2
MAX_GENERATION_ATTEMPTS = 800


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(shape, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    pts = sorted((int(r), int(c)) for r, c in arr)
    return tuple(pts)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.array(shape, dtype=int)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def _flip_h(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, w = _shape_dims(shape)
    _ = h
    return _normalize_shape(tuple((r, w - 1 - c) for r, c in shape))


def _flip_v(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, w = _shape_dims(shape)
    _ = w
    return _normalize_shape(tuple((h - 1 - r, c) for r, c in shape))


def _shape_variants(base_shapes: list[tuple[tuple[int, int], ...]]) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for shape in base_shapes:
        s0 = _normalize_shape(shape)
        s1 = _flip_h(s0)
        s2 = _flip_v(s0)
        s3 = _flip_h(s2)
        variants.update((s0, s1, s2, s3))
    return sorted(variants)


GRAY_BASES = [
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
]

ASYM_COLOR_BASES = [
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 1), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 2), (1, 2), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 1), (2, 1)),
]

STABLE_COLOR_BASES = [
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0)),
]

GRAY_SHAPES = _shape_variants(GRAY_BASES)
ASYM_COLOR_SHAPES = _shape_variants(ASYM_COLOR_BASES)
STABLE_COLOR_SHAPES = _shape_variants(STABLE_COLOR_BASES)


def _row_cols(shape: tuple[tuple[int, int], ...], row: int) -> set[int]:
    return {c for r, c in shape if r == row}


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    for dr, dc in shape:
        rr = r0 + dr
        cc = c0 + dc
        assert grid[rr, cc] == BG
        grid[rr, cc] = int(color)


def _mirror_foregrounds(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    mask = (g != BG) & (g != NEUTRAL)
    out[mask] = BG
    paint = np.zeros_like(g)
    used = np.zeros_like(g, dtype=bool)
    for color in colors_present(g, exclude={BG, NEUTRAL}):
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, _c0, r1, _c1 = bbox_coords(coords)
        rr = r0 + r1 - coords[:, 0]
        cc = coords[:, 1]
        assert not np.any(g[rr, cc] == NEUTRAL)
        assert not np.any(used[rr, cc])
        paint[rr, cc] = int(color)
        used[rr, cc] = True
    out[used] = paint[used]
    return out


def _touching_colors(grid: np.ndarray, comp: np.ndarray) -> set[int]:
    h, w = grid.shape
    colors: set[int] = set()
    for r, c in comp:
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                color = int(grid[nr, nc])
                if color not in {BG, NEUTRAL}:
                    colors.add(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each colored mask inside its bbox, then recolor each gray component from its post-mirror neighbor.

    Input structure:
    - A black background contains 2 to 3 gray 4-connected neutral regions.
    - Several non-gray foreground colors form single 4-connected masks whose color bboxes do not overlap.
    - Gray regions and colored masks are separated so that vertical mirroring can change which colors touch gray.

    Rule:
    - For every non-background, non-gray color independently, find its tight bounding box and reflect the color mask vertically inside that box.
    - Keep gray and background fixed while only the colored masks move to their mirrored positions.
    - For each gray 4-connected component, inspect the mirrored scene and collect foreground colors touching that component by 4-neighborhood.
    - If exactly one foreground color touches a gray component, repaint the whole component to that color; otherwise leave it gray.

    Key insight:
    - The final recoloring is determined by adjacency after the per-color mirror, so colors can be introduced or removed solely because the mirror changes which bbox edge cells face a gray region.
    """
    mirrored = _mirror_foregrounds(np.asarray(grid, dtype=int))
    out = mirrored.copy()
    for comp in _components(mirrored == NEUTRAL):
        touching = _touching_colors(mirrored, comp)
        if len(touching) == 1:
            color = next(iter(touching))
            out[comp[:, 0], comp[:, 1]] = int(color)
        elif len(touching) > 1:
            raise AssertionError("gray component has ambiguous post-mirror adjacency")
    return out


def _touch_status(
    gray_shape: tuple[tuple[int, int], ...],
    color_shape: tuple[tuple[int, int], ...],
    side: str,
    dx: int,
) -> tuple[bool, bool]:
    gray_h, _gray_w = _shape_dims(gray_shape)
    color_h, _color_w = _shape_dims(color_shape)
    gray_cols = _row_cols(gray_shape, 0 if side == "above" else gray_h - 1)
    top_cols = {c + dx for c in _row_cols(color_shape, 0)}
    bottom_cols = {c + dx for c in _row_cols(color_shape, color_h - 1)}
    if side == "above":
        before = bool(gray_cols & bottom_cols)
        after = bool(gray_cols & top_cols)
    else:
        before = bool(gray_cols & top_cols)
        after = bool(gray_cols & bottom_cols)
    return before, after


def _matches_mode(before: bool, after: bool, mode: str) -> bool:
    if mode == "gain":
        return (not before) and after
    if mode == "lose":
        return before and (not after)
    if mode == "stable":
        return before and after
    raise ValueError(mode)


def _find_object_spec(
    gray_shape: tuple[tuple[int, int], ...],
    side: str,
    mode: str,
    shape_pool: list[tuple[tuple[int, int], ...]],
) -> dict | None:
    candidates = random.sample(shape_pool, len(shape_pool))
    for shape in candidates:
        _h, w = _shape_dims(shape)
        dx_lo = -GRAY_ORIGIN_C
        dx_hi = ZONE_W - w - GRAY_ORIGIN_C
        dxs = list(range(dx_lo, dx_hi + 1))
        random.shuffle(dxs)
        for dx in dxs:
            before, after = _touch_status(gray_shape, shape, side, dx)
            if _matches_mode(before, after, mode):
                return {"shape": shape, "side": side, "mode": mode, "dx": dx}
    return None


def _zone_layout(gray_count: int) -> list[tuple[int, int]]:
    if gray_count == 2:
        options = [
            [(0, 0), (0, 1)],
            [(0, 0), (1, 0)],
            [(0, 0), (1, 1)],
        ]
    else:
        options = [
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0), (1, 1)],
        ]
    return random.choice(options)


def _render_scene(specs: list[dict], slots: list[tuple[int, int]]) -> np.ndarray:
    max_sr = max(sr for sr, _sc in slots)
    max_sc = max(sc for _sr, sc in slots)
    grid = np.zeros(((max_sr + 1) * ZONE_H, (max_sc + 1) * ZONE_W), dtype=int)
    for spec, (slot_r, slot_c) in zip(specs, slots):
        zone_r = slot_r * ZONE_H
        zone_c = slot_c * ZONE_W
        gray_origin = (zone_r + GRAY_ORIGIN_R, zone_c + GRAY_ORIGIN_C)
        _paint_shape(grid, spec["gray_shape"], gray_origin, NEUTRAL)

        gray_h, _gray_w = _shape_dims(spec["gray_shape"])
        gray_r0, gray_c0 = gray_origin
        gray_r1 = gray_r0 + gray_h - 1

        for item in spec["objects"]:
            shape = item["shape"]
            obj_h, _obj_w = _shape_dims(shape)
            if item["side"] == "above":
                obj_origin = (gray_r0 - obj_h, gray_c0 + item["dx"])
            else:
                obj_origin = (gray_r1 + 1, gray_c0 + item["dx"])
            _paint_shape(grid, shape, obj_origin, item["color"])
    return grid


def _validate_scene(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(grid.min()) < 0 or int(grid.max()) > 9:
        return False

    gray_comps = _components(grid == NEUTRAL)
    if len(gray_comps) not in {2, 3}:
        return False
    if not all(4 <= len(comp) <= 8 for comp in gray_comps):
        return False

    fg_colors = colors_present(grid, exclude={BG, NEUTRAL})
    if len(fg_colors) not in {3, 4}:
        return False

    bboxes: list[tuple[int, int, int, int]] = []
    for color in fg_colors:
        mask = grid == color
        comps = _components(mask)
        if len(comps) != 1:
            return False
        if not (3 <= len(comps[0]) <= 6):
            return False
        bboxes.append(bbox_coords(np.argwhere(mask)))

    for i, (r0, c0, r1, c1) in enumerate(bboxes):
        for j in range(i + 1, len(bboxes)):
            rr0, cc0, rr1, cc1 = bboxes[j]
            overlap = not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0)
            if overlap:
                return False

    mirrored = _mirror_foregrounds(grid)
    after_sets = [_touching_colors(mirrored, comp) for comp in gray_comps]
    if any(len(s) != 1 for s in after_sets):
        return False

    before_touching: set[int] = set()
    after_touching: set[int] = set()
    for comp in gray_comps:
        before_touching |= _touching_colors(grid, comp)
        after_touching |= _touching_colors(mirrored, comp)

    if not (before_touching - after_touching):
        return False
    if not (after_touching - before_touching):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.sum(out == NEUTRAL) != 0:
        return False
    expected_colors = Counter(map(int, out.ravel()))
    if expected_colors[BG] <= 0:
        return False
    return True


def _make_zone_spec(
    target_color: int,
    distractor_color: int | None,
    force_gain: bool,
) -> dict | None:
    gray_shape = random.choice(GRAY_SHAPES)
    target_side = random.choice(["above", "below"])
    target_mode = "gain" if force_gain or random.random() < 0.6 else "stable"
    target_pool = ASYM_COLOR_SHAPES if target_mode == "gain" else STABLE_COLOR_SHAPES
    target = _find_object_spec(gray_shape, target_side, target_mode, target_pool)
    if target is None:
        return None

    objects = [{**target, "color": target_color}]
    if distractor_color is not None:
        distractor_side = "below" if target_side == "above" else "above"
        distractor = _find_object_spec(gray_shape, distractor_side, "lose", ASYM_COLOR_SHAPES)
        if distractor is None:
            return None
        objects.append({**distractor, "color": distractor_color})

    return {"gray_shape": gray_shape, "objects": objects}


def _gen_attempt() -> np.ndarray | None:
    gray_count = random.choice([2, 3])
    distractor_count = random.choice([1, 2]) if gray_count == 2 else 1
    total_colors = gray_count + distractor_count
    colors = random.sample(PALETTE, total_colors)
    target_colors = colors[:gray_count]
    distractor_colors = colors[gray_count:]
    slots = _zone_layout(gray_count)
    force_gain_idx = random.randrange(gray_count)
    distractor_slots = random.sample(range(gray_count), distractor_count)
    distractor_map: dict[int, int] = {
        zone_idx: color for zone_idx, color in zip(distractor_slots, distractor_colors)
    }

    specs: list[dict] = []
    for idx in range(gray_count):
        spec = _make_zone_spec(
            target_color=target_colors[idx],
            distractor_color=distractor_map.get(idx),
            force_gain=idx == force_gain_idx,
        )
        if spec is None:
            return None
        specs.append(spec)

    grid = _render_scene(specs, slots)
    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-trigger adjacency-transfer input")
