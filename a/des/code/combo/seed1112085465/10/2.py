from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must be non-empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _spans_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = map(int, mask.shape)
    dst_h, dst_w = map(int, target_shape)
    assert src_h > 0 and src_w > 0 and dst_h > 0 and dst_w > 0

    out = np.zeros((dst_h, dst_w), dtype=bool)
    for r, c in np.argwhere(mask):
        r0 = int(r) * dst_h // src_h
        r1 = (int(r) + 1) * dst_h + src_h - 1
        r1 //= src_h
        c0 = int(c) * dst_w // src_w
        c1 = (int(c) + 1) * dst_w + src_w - 1
        c1 //= src_w
        out[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = True

    assert out.any(), "resized mask must stay non-empty"
    assert _bbox(out) == (0, 0, dst_h - 1, dst_w - 1), "resized mask must fill its target bbox"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each 4-connected two-color object into its two color layers.

    Keep the smaller layer unchanged. Remove the larger layer from its original
    cells, scale that binary layer to the smaller layer's bbox size, and place
    it directly below the original object bbox with horizontal centering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    in_h, in_w = map(int, g.shape)
    bg = int(get_bg(g))

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert components, "expected at least one foreground component"
    components = sorted(components, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))

    extracted: list[dict[str, object]] = []
    out_h = in_h

    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        coords = np.asarray(comp["coords"], dtype=int)
        comp_colors = sorted({int(g[int(r), int(c)]) for r, c in coords} - {bg})
        assert len(comp_colors) == 2, "each object must use exactly two foreground colors"

        color_masks = {color: patch == color for color in comp_colors}
        color_areas = {color: int(mask.sum()) for color, mask in color_masks.items()}
        assert color_areas[comp_colors[0]] != color_areas[comp_colors[1]], "the two color areas must differ"

        small_color, large_color = sorted(comp_colors, key=lambda color: color_areas[color])
        small_mask = color_masks[small_color]
        large_mask = color_masks[large_color]

        small_bbox = _bbox(small_mask)
        large_bbox = _bbox(large_mask)
        target_h = int(small_bbox[2] - small_bbox[0] + 1)
        target_w = int(small_bbox[3] - small_bbox[1] + 1)
        scaled_large = _resize_mask(_crop_mask(large_mask, large_bbox), (target_h, target_w))

        obj_h = int(r1 - r0 + 1)
        obj_w = int(c1 - c0 + 1)
        dest_r0 = int(r1 + 1)
        dest_c0 = int(c0 + (obj_w - target_w) // 2)
        assert 0 <= dest_c0 and dest_c0 + target_w <= in_w

        large_coords = coords[g[coords[:, 0], coords[:, 1]] == large_color]
        extracted.append(
            {
                "large_coords": large_coords,
                "large_color": int(large_color),
                "dest_r0": dest_r0,
                "dest_c0": dest_c0,
                "scaled_large": scaled_large,
            }
        )
        out_h = max(out_h, int(dest_r0 + target_h))

    out = np.full((out_h, in_w), bg, dtype=int)
    out[:in_h, :] = g

    for item in extracted:
        coords = np.asarray(item["large_coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg

    for item in extracted:
        dest_r0 = int(item["dest_r0"])
        dest_c0 = int(item["dest_c0"])
        color = int(item["large_color"])
        mask = np.asarray(item["scaled_large"], dtype=bool)
        target = out[dest_r0 : dest_r0 + mask.shape[0], dest_c0 : dest_c0 + mask.shape[1]]
        assert np.all(target[mask] == bg), "shifted larger layers must not overlap"
        target[mask] = color

    return out


_RNG = np.random.default_rng()

BG_COLOR = 7
FG_COLORS = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9], dtype=int)

GRID_H_MIN = 10
GRID_H_MAX = 18
GRID_W_MIN = 12
GRID_W_MAX = 26
NUM_OBJECTS_MIN = 1
NUM_OBJECTS_MAX = 3
OBJECT_H_MIN = 4
OBJECT_H_MAX = 8
OBJECT_W_MIN = 4
OBJECT_W_MAX = 8
MIN_COL_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _sample_small_mask(h: int, w: int, object_area: int) -> np.ndarray | None:
    min_area = int(h + w - 1)
    max_area = min(int(h * w), int((object_area - 1) // 2))
    if min_area > max_area:
        return None

    for _ in range(50):
        target_area = int(_RNG.integers(min_area, max_area + 1))
        mask = np.ones((h, w), dtype=bool)

        while int(mask.sum()) > target_area:
            removable: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                trial = mask.copy()
                trial[int(r), int(c)] = False
                if not _spans_full_bbox(trial):
                    continue
                if not _is_connected(trial):
                    continue
                removable.append((int(r), int(c)))

            if not removable:
                break
            rr, cc = removable[int(_RNG.integers(0, len(removable)))]
            mask[rr, cc] = False

        if int(mask.sum()) == target_area and _spans_full_bbox(mask) and _is_connected(mask):
            return mask

    return None


def _make_object_patch(obj_h: int, obj_w: int, small_color: int, large_color: int) -> dict[str, object] | None:
    if obj_h < 4 or obj_w < 4:
        return None

    object_area = int(obj_h * obj_w)
    for _ in range(80):
        small_h = int(_RNG.integers(2, obj_h - 1))
        small_w = int(_RNG.integers(2, obj_w - 1))
        if small_h > obj_h - 2 or small_w > obj_w - 2:
            continue

        small_mask = _sample_small_mask(small_h, small_w, object_area)
        if small_mask is None:
            continue

        inner_r0 = int(_RNG.integers(1, obj_h - small_h))
        inner_c0 = int(_RNG.integers(1, obj_w - small_w))

        full_small = np.zeros((obj_h, obj_w), dtype=bool)
        full_small[inner_r0 : inner_r0 + small_h, inner_c0 : inner_c0 + small_w] = small_mask
        large_mask = ~full_small

        small_area = int(full_small.sum())
        large_area = int(large_mask.sum())
        if large_area <= small_area:
            continue

        patch = np.full((obj_h, obj_w), large_color, dtype=int)
        patch[full_small] = small_color

        return {
            "patch": patch,
            "h": obj_h,
            "w": obj_w,
            "small_h": small_h,
            "small_w": small_w,
            "small_area": small_area,
            "large_area": large_area,
        }

    return None


def _assign_columns(widths: list[int], grid_w: int) -> list[int] | None:
    count = len(widths)
    required = sum(widths) + MIN_COL_GAP * (count - 1)
    if required > grid_w:
        return None

    slack = int(grid_w - required)
    extras = _RNG.multinomial(slack, np.ones(count + 1) / float(count + 1))
    cols: list[int] = []
    c = int(extras[0])
    for i, width in enumerate(widths):
        cols.append(c)
        c += int(width)
        if i < count - 1:
            c += MIN_COL_GAP + int(extras[i + 1])
    return cols


def _choose_rows(specs: list[dict[str, object]], grid_h: int, force_expand: bool) -> list[int] | None:
    rows: list[int] = []
    expand_index = int(_RNG.integers(0, len(specs))) if force_expand else -1

    for i, spec in enumerate(specs):
        obj_h = int(spec["h"])
        small_h = int(spec["small_h"])
        max_top = grid_h - obj_h
        safe_top_max = grid_h - obj_h - small_h
        if max_top < 0:
            return None

        if force_expand and i == expand_index:
            low = max(0, safe_top_max + 1)
            if low > max_top:
                return None
            top = int(_RNG.integers(low, max_top + 1))
        elif force_expand and safe_top_max >= 0 and bool(_RNG.integers(0, 2)):
            top = int(_RNG.integers(0, safe_top_max + 1))
        elif force_expand:
            low = max(0, safe_top_max + 1)
            top = int(_RNG.integers(low, max_top + 1))
        else:
            if safe_top_max < 0:
                return None
            top = int(_RNG.integers(0, safe_top_max + 1))

        rows.append(top)

    return rows


def _grid_height_bounds(specs: list[dict[str, object]], force_expand: bool) -> tuple[int, int] | None:
    low = max(int(spec["h"]) for spec in specs)
    if not force_expand:
        low = max(low, max(int(spec["h"]) + int(spec["small_h"]) for spec in specs))
    low = max(low, GRID_H_MIN)
    high = GRID_H_MAX
    if low > high:
        return None
    return low, high


def _build_input(specs: list[dict[str, object]], rows: list[int], cols: list[int], grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, grid_shape)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for spec, r0, c0 in zip(specs, rows, cols):
        patch = np.asarray(spec["patch"], dtype=int)
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return grid


def _valid_input(grid: np.ndarray, expected_objects: int) -> bool:
    if int(get_bg(grid)) != BG_COLOR:
        return False

    comps = find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=False, return_format="dict")
    if len(comps) != expected_objects:
        return False

    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        colors = sorted({int(grid[int(r), int(c)]) for r, c in coords} - {BG_COLOR})
        if len(colors) != 2:
            return False
        counts = [int(np.sum(grid[coords[:, 0], coords[:, 1]] == color)) for color in colors]
        if counts[0] == counts[1]:
            return False

    try:
        main(grid)
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = int(_RNG.integers(NUM_OBJECTS_MIN, NUM_OBJECTS_MAX + 1))
    sampled_colors = _RNG.choice(FG_COLORS, size=2 * object_count, replace=False)

    specs: list[dict[str, object]] = []
    for i in range(object_count):
        obj_h = int(_RNG.integers(OBJECT_H_MIN, OBJECT_H_MAX + 1))
        obj_w = int(_RNG.integers(OBJECT_W_MIN, OBJECT_W_MAX + 1))
        color_a = int(sampled_colors[2 * i])
        color_b = int(sampled_colors[2 * i + 1])
        if bool(_RNG.integers(0, 2)):
            small_color, large_color = color_a, color_b
        else:
            small_color, large_color = color_b, color_a
        spec = _make_object_patch(obj_h, obj_w, small_color, large_color)
        if spec is None:
            return None
        specs.append(spec)

    force_expand = bool(_RNG.integers(0, 2))
    height_bounds = _grid_height_bounds(specs, force_expand)
    if height_bounds is None:
        return None
    grid_h = int(_RNG.integers(height_bounds[0], height_bounds[1] + 1))

    min_w = max(GRID_W_MIN, sum(int(spec["w"]) for spec in specs) + MIN_COL_GAP * (object_count - 1))
    if min_w > GRID_W_MAX:
        return None
    grid_w = int(_RNG.integers(min_w, GRID_W_MAX + 1))

    cols = _assign_columns([int(spec["w"]) for spec in specs], grid_w)
    if cols is None:
        return None
    rows = _choose_rows(specs, grid_h, force_expand)
    if rows is None:
        return None

    grid = _build_input(specs, rows, cols, (grid_h, grid_w))
    if not _valid_input(grid, object_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
