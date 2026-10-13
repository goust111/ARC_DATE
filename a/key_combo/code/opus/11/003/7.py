from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


TOP_STOP_SHAPES = [
    np.array([[1], [1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
]

GENERAL_SHAPES = [
    np.array([[1], [1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
]

GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1])


def _color_crop_mask(grid: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = bbox(grid == color)
    assert box is not None
    return crop_bbox(grid == color, box), box


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == color, dtype=bool)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            shape[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": shape,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _visible_coords(shape: np.ndarray, x: int, y: int, canvas_shape: tuple[int, int]) -> tuple[list[tuple[int, int]], bool]:
    H, W = canvas_shape
    coords: list[tuple[int, int]] = []
    truncated = False
    for rr, cc in np.argwhere(shape):
        dr = y + int(rr)
        dc = x + int(cc)
        if 0 <= dr < H and 0 <= dc < W:
            coords.append((dr, dc))
        else:
            truncated = True
    return coords, truncated


def _simulate_drop(
    terrain_mask: np.ndarray,
    terrain_color: int,
    components: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    H, W = terrain_mask.shape
    out = np.zeros((H, W), dtype=int)
    out[terrain_mask] = int(terrain_color)
    terrain = np.asarray(terrain_mask, dtype=bool)
    object_solid = np.zeros((H, W), dtype=bool)
    solid = terrain.copy()
    placements: list[dict] = []

    for comp in components:
        shape = np.asarray(comp["shape"], dtype=bool)
        h, _w = shape.shape
        x = int(comp["rel_x"])
        row = 0
        support_kind = "floor"

        while row < H - h:
            coords, truncated = _visible_coords(shape, x, row, (H, W))
            hits_terrain = False
            hits_object = False
            for dr, dc in coords:
                if dr + 1 >= H:
                    continue
                if solid[dr + 1, dc]:
                    if terrain[dr + 1, dc]:
                        hits_terrain = True
                    elif object_solid[dr + 1, dc]:
                        hits_object = True
            if hits_terrain or hits_object:
                support_kind = "object" if hits_object else "terrain"
                break
            row += 1

        coords, truncated = _visible_coords(shape, x, row, (H, W))
        overwritten = 0
        for dr, dc in coords:
            if terrain[dr, dc]:
                continue
            if object_solid[dr, dc]:
                overwritten += 1
            out[dr, dc] = int(comp["color"])
            object_solid[dr, dc] = True
            solid[dr, dc] = True

        placements.append(
            {
                "color": int(comp["color"]),
                "orig_bbox": tuple(map(int, comp["bbox"])),
                "shape": shape,
                "rel_x": int(x),
                "final_row": int(row),
                "support": support_kind,
                "truncated": bool(truncated),
                "visible_area": len(coords),
                "overwritten": int(overwritten),
            }
        )

    return out, placements


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert fg_colors, "expected foreground colors"

    symmetric: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
    for color in fg_colors:
        mask_crop, box = _color_crop_mask(g, color)
        if np.array_equal(mask_crop, np.fliplr(mask_crop)):
            symmetric.append((int(color), mask_crop, box))
    assert len(symmetric) == 1, "expected exactly one vertically symmetric color mask"

    terrain_color, terrain_mask, terrain_bbox = symmetric[0]
    components: list[dict] = []
    for color in fg_colors:
        if color == terrain_color:
            continue
        components.extend(_components_of_color(g, color))

    components.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    terrain_left = int(terrain_bbox[1])
    for comp in components:
        comp["rel_x"] = int(comp["bbox"][1]) - terrain_left

    out, placements = _simulate_drop(terrain_mask, terrain_color, components)
    return {
        "bg": int(bg),
        "terrain_color": int(terrain_color),
        "terrain_mask": terrain_mask,
        "terrain_bbox": tuple(map(int, terrain_bbox)),
        "components": components,
        "placements": placements,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the unique symmetric color mask as a cropped terrain, then drop the other objects onto it.

    Input structure:
    - The background is black and the foreground contains 3-4 non-background colors.
    - Exactly one foreground color, always grey or maroon, becomes left-right symmetric after cropping all of its pixels to one bounding box.
    - The remaining colors form small 4-connected objects, with one or two components per color.

    Rule:
    - Crop each foreground color mask to its own bounding box and identify the unique left-right symmetric one.
    - Use that cropped symmetric mask as the output canvas: its foreground pixels stay as solid terrain and its background cells become air.
    - Crop every 4-connected component of every non-symmetric color to its own bounding box, sort those components by original left column then top row, align each component horizontally to the terrain coordinate frame, and let it fall from the top.
    - A falling component stops when any of its pixels would have terrain or an already settled object directly below on the next row; later objects may overwrite earlier objects, but never terrain.

    Key insight:
    - The symmetric mask is not the final picture by itself: it defines the cropped physical world in which the asymmetric components are re-simulated under gravity.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _mirror(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask)


def _build_terrain_mask(height: int, width: int) -> tuple[np.ndarray, dict]:
    assert height >= 6 and width >= 5
    mask = np.zeros((height, width), dtype=bool)

    heights: list[int] = []
    for c in range((width + 1) // 2):
        heights.append(random.randint(1, 2))
    for c, h in enumerate(heights):
        mirror_c = width - 1 - c
        mask[height - h :, c] = True
        mask[height - h :, mirror_c] = True
    if len(set(heights)) == 1:
        idx = random.randrange(len(heights))
        heights[idx] = 1 if heights[idx] == 2 else 2
        c = idx
        mirror_c = width - 1 - c
        mask[:, c] = False
        mask[:, mirror_c] = False
        h = heights[idx]
        mask[height - h :, c] = True
        mask[height - h :, mirror_c] = True

    mask[0, 0] = True
    mask[0, width - 1] = True

    top_stop_cols = [1, width - 2]
    mask[1, top_stop_cols[0]] = True
    mask[1, top_stop_cols[1]] = True

    platform_row = random.randint(2, height - 4)
    platform_half = random.randint(1, max(1, min(2, width // 2 - 1)))
    center_left = (width - 1) // 2 - platform_half + 1
    center_right = width - 1 - center_left
    platform_cols = list(range(center_left, center_right + 1))
    for c in platform_cols:
        mask[platform_row, c] = True

    if random.random() < 0.6 and platform_row + 2 < height - 1:
        inner = [c for c in platform_cols if c not in {0, width - 1}]
        if inner:
            c = random.choice(inner)
            mask[platform_row + 2, c] = True
            mask[platform_row + 2, width - 1 - c] = True

    assert np.array_equal(mask, _mirror(mask))
    assert mask[0].any() and mask[-1].any()
    assert mask[:, 0].any() and mask[:, -1].any()
    assert any(mask[r, c] and not mask[r + 1, c] for r in range(height - 1) for c in range(width))
    return mask, {
        "top_stop_cols": top_stop_cols,
        "platform_row": int(platform_row),
        "platform_cols": platform_cols,
    }


def _random_shape(kind: str) -> np.ndarray:
    base = random.choice(TOP_STOP_SHAPES if kind == "top" else GENERAL_SHAPES)
    shape = np.rot90(base, random.randrange(4))
    return np.asarray(shape, dtype=bool)


def _place_anchor(shape: np.ndarray, target_col: int, prefer_top: bool) -> int | None:
    coords = np.argwhere(shape)
    if prefer_top:
        row = int(coords[:, 0].min())
    else:
        row = int(coords[:, 0].max())
    cols = [int(c) for r, c in coords if int(r) == row]
    random.shuffle(cols)
    for local_c in cols:
        x = int(target_col) - int(local_c)
        if x >= 0:
            return int(x)
    return None


def _paint_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> bool:
    H, W = grid.shape
    coords = np.argwhere(shape)
    for rr, cc in coords:
        dr = top + int(rr)
        dc = left + int(cc)
        if not (0 <= dr < H and 0 <= dc < W) or int(grid[dr, dc]) != 0:
            return False
    for rr, cc in coords:
        grid[top + int(rr), left + int(cc)] = int(color)
    return True


def _assign_component_colors(component_count: int, terrain_color: int) -> list[int]:
    pool = [c for c in range(1, 10) if c not in {terrain_color}]
    random.shuffle(pool)
    if component_count == 3:
        if random.random() < 0.5:
            return [pool[0], pool[1], pool[0]]
        return [pool[0], pool[1], pool[2]]
    if random.random() < 0.5:
        return [pool[0], pool[1], pool[2], pool[0]]
    return [pool[0], pool[1], pool[2], pool[1]]


def _semantic_checks(grid: np.ndarray, analysis: dict, require_truncation: bool) -> bool:
    terrain_mask = np.asarray(analysis["terrain_mask"], dtype=bool)
    H, W = terrain_mask.shape
    terrain_color = int(analysis["terrain_color"])
    terrain_bbox = analysis["terrain_bbox"]
    components = analysis["components"]
    placements = analysis["placements"]
    output = analysis["output"]

    if terrain_color not in {5, 9}:
        return False
    if terrain_mask.shape[0] < 6 or terrain_mask.shape[1] < 5:
        return False
    if len(colors_present(grid, exclude={0})) not in {3, 4}:
        return False
    if len(components) < 3:
        return False
    if any(not (2 <= int(comp["area"]) <= 5) for comp in components):
        return False
    if any(comp["shape"].shape[1] > W for comp in components):
        return False
    if any(int(comp["rel_x"]) < 0 for comp in components):
        return False
    if not any(p["support"] == "object" for p in placements):
        return False
    if not any(
        p["support"] == "terrain" and int(p["final_row"]) < H - p["shape"].shape[0]
        for p in placements
    ):
        return False
    if not any(p["support"] == "terrain" and int(p["final_row"]) == 0 for p in placements):
        return False
    if require_truncation and not any(p["truncated"] for p in placements):
        return False
    if output.shape != terrain_mask.shape:
        return False
    if np.array_equal(output, np.where(terrain_mask, terrain_color, 0)):
        return False

    fg_colors = colors_present(grid, exclude={0})
    symmetric_count = 0
    for color in fg_colors:
        crop, _box = _color_crop_mask(grid, color)
        if np.array_equal(crop, np.fliplr(crop)):
            symmetric_count += 1
    if symmetric_count != 1:
        return False

    if len({tuple(p["orig_bbox"]) for p in placements}) != len(placements):
        return False

    terrain_cols = set(np.where(terrain_mask.any(axis=0))[0].tolist())
    if len(terrain_cols) != terrain_bbox[3] - terrain_bbox[1] + 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    terrain_h = random.randint(6, 10)
    terrain_w = random.randint(5, 9)
    terrain_mask, terrain_info = _build_terrain_mask(terrain_h, terrain_w)
    terrain_color = random.choice([5, 9])

    include_truncation = random.random() < 0.55
    component_count = 4 if include_truncation else 3
    colors = _assign_component_colors(component_count, terrain_color)

    top_shape = _random_shape("top")
    x_top = _place_anchor(top_shape, terrain_info["top_stop_cols"][0], prefer_top=True)
    if x_top is None or x_top + top_shape.shape[1] > terrain_w:
        return None

    stack_shape = _random_shape("general")
    bottom_cols = [int(c) for r, c in np.argwhere(stack_shape) if int(r) == stack_shape.shape[0] - 1]
    if not bottom_cols:
        return None
    support_col = random.choice(terrain_info["platform_cols"])
    x_stack = _place_anchor(stack_shape, support_col, prefer_top=False)
    if x_stack is None or x_stack + stack_shape.shape[1] > terrain_w:
        return None

    component_specs = [
        {"shape": top_shape, "rel_x": int(x_top), "color": int(colors[0]), "role": "top"},
        {"shape": stack_shape, "rel_x": int(x_stack), "color": int(colors[1]), "role": "stack_base"},
        {"shape": np.rot90(stack_shape, random.randrange(4)), "rel_x": int(x_stack), "color": int(colors[2]), "role": "stack_top"},
    ]

    if include_truncation:
        trunc_shape = _random_shape("general")
        if trunc_shape.shape[1] < 2:
            trunc_shape = np.array([[1, 1], [1, 0]], dtype=bool)
        x_trunc = terrain_w - trunc_shape.shape[1] + random.randint(1, trunc_shape.shape[1] - 1)
        component_specs.append(
            {"shape": trunc_shape, "rel_x": int(x_trunc), "color": int(colors[3]), "role": "trunc"}
        )

    max_right = max(int(spec["rel_x"]) + spec["shape"].shape[1] for spec in component_specs)
    min_full_w = max(13, max_right + 3, terrain_w + 2)
    max_full_w = max(min_full_w, min(20, min_full_w + 5))
    full_w = random.randint(min_full_w, max_full_w)
    min_full_h = max(13, terrain_h + 5)
    max_full_h = max(min_full_h, min(20, min_full_h + 5))
    full_h = random.randint(min_full_h, max_full_h)

    max_terrain_left = min(full_w - max_right - 1, full_w - terrain_w - 1)
    if max_terrain_left < 1:
        return None
    terrain_left = random.randint(1, max_terrain_left)
    terrain_top = random.randint(1, max(1, full_h - terrain_h - 1))

    grid = np.zeros((full_h, full_w), dtype=int)
    grid[terrain_top : terrain_top + terrain_h, terrain_left : terrain_left + terrain_w][terrain_mask] = terrain_color

    row_candidates = list(range(full_h))
    random.shuffle(row_candidates)
    same_x_rows: dict[int, list[int]] = {}

    ordered_specs = sorted(
        enumerate(component_specs),
        key=lambda item: (int(item[1]["rel_x"]), {"top": 0, "stack_base": 1, "stack_top": 2, "trunc": 3}[item[1]["role"]]),
    )

    for idx, spec in ordered_specs:
        shape = np.asarray(spec["shape"], dtype=bool)
        h, _w = shape.shape
        left = terrain_left + int(spec["rel_x"])
        placed = False

        if spec["role"] == "stack_base":
            candidate_rows = list(range(0, max(1, terrain_top))) + list(range(terrain_top + terrain_h, full_h - h + 1))
        elif spec["role"] == "stack_top":
            candidate_rows = list(range(terrain_top + terrain_h, full_h - h + 1))
            if x_stack in same_x_rows:
                candidate_rows = [r for r in candidate_rows if r > max(same_x_rows[x_stack])]
        else:
            candidate_rows = list(range(0, full_h - h + 1))

        random.shuffle(candidate_rows)
        for top in candidate_rows:
            if _paint_shape(grid, shape, top, left, int(spec["color"])):
                same_x_rows.setdefault(int(spec["rel_x"]), []).append(int(top))
                component_specs[idx]["input_top"] = int(top)
                placed = True
                break
        if not placed:
            return None

    try:
        analysis = _analyze(grid)
    except AssertionError:
        return None

    if not _semantic_checks(grid, analysis, require_truncation=include_truncation):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-terrain gravity sample")
