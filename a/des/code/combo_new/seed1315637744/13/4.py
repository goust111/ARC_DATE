from __future__ import annotations

import random

import numpy as np


FRAME_COLOR = 5
GEN_RETRIES = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        start_r = int(start_r)
        start_c = int(start_c)
        if seen[start_r, start_c]:
            continue
        stack = [(start_r, start_c)]
        seen[start_r, start_c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int | None = None) -> None:
    height, width = patch.shape
    target = canvas[top : top + height, left : left + width]
    if transparent is None:
        target[:, :] = patch
        return
    mask = patch != transparent
    target[mask] = patch[mask]


def _layer_kind(patch: np.ndarray, bg: int) -> str | None:
    filled = patch != bg
    height, width = patch.shape
    count = int(np.count_nonzero(filled))
    if count == height * width:
        return "solid"
    if count != 2 * (height + width) - 4 or height < 3 or width < 3:
        return None
    if not np.all(filled[0, :]) or not np.all(filled[-1, :]):
        return None
    if not np.all(filled[:, 0]) or not np.all(filled[:, -1]):
        return None
    if np.any(filled[1:-1, 1:-1]):
        return None
    return "frame"


def _find_empty_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    frames: list[dict] = []
    frame_mask = np.zeros_like(grid, dtype=bool)
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            if comp.shape[0] != 2 * (height + width) - 4:
                continue
            box = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(box[0, :] == color) or not np.all(box[-1, :] == color):
                continue
            if not np.all(box[:, 0] == color) or not np.all(box[:, -1] == color):
                continue
            if not np.all(box[1:-1, 1:-1] == bg):
                continue
            frames.append({"bbox": (r0, c0, r1, c1), "color": color})
            frame_mask[comp[:, 0], comp[:, 1]] = True
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames, frame_mask


def _extract_object(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    colors = np.array([int(grid[r, c]) for r, c in coords], dtype=int)
    unique_colors = list(np.unique(colors))
    if len(unique_colors) < 2:
        return None

    layers: list[dict] = []
    for color in unique_colors:
        pts = coords[colors == color]
        r0, c0, r1, c1 = _bbox(pts)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        patch[pts[:, 0] - r0, pts[:, 1] - c0] = int(color)
        kind = _layer_kind(patch, bg)
        if kind is None:
            return None
        layers.append({"bbox": (r0, c0, r1, c1), "patch": patch, "kind": kind})

    layers.sort(
        key=lambda item: (
            item["patch"].shape[0] * item["patch"].shape[1],
            item["bbox"][0],
            item["bbox"][1],
        )
    )
    if layers[0]["kind"] != "solid" or any(layer["kind"] != "frame" for layer in layers[1:]):
        return None

    for inner, outer in zip(layers, layers[1:]):
        ir0, ic0, ir1, ic1 = inner["bbox"]
        or0, oc0, or1, oc1 = outer["bbox"]
        if not (or0 == ir0 - 1 and oc0 == ic0 - 1 and or1 == ir1 + 1 and oc1 == ic1 + 1):
            return None

    return {"coords": coords, "bbox": _bbox(coords), "layers": [layer["patch"] for layer in layers]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel nested multicolor rectangles into separate single-color layers and restack them into the empty target frames.

    Input structure:
    - The dominant color is the background.
    - The left side contains several connected multicolor objects made of concentric rectangular layers: outer layers are one-cell frames and the innermost layer is a solid rectangle.
    - The right side contains vertically stacked empty rectangular frames with background interiors.

    Rule:
    - Detect each multicolor nested object, split it into its original single-color layers, and order those layers from inner to outer.
    - Process objects by descending layer count; break ties by their top-to-bottom position.
    - Erase the original objects, then place the peeled layers into the right-side empty frames from top to bottom, centering each layer inside its destination frame while keeping the frame border intact.

    Key insight:
    - The empty frames are the targets, while each left-side object can be reconstructed as a chain of nested rectangular shells whose order is determined by geometric containment rather than by color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)

    frames, frame_mask = _find_empty_frames(g, bg)
    assert frames, "expected at least one empty target frame"

    objects: list[dict] = []
    for comp in _components((g != bg) & (~frame_mask)):
        obj = _extract_object(g, comp, bg)
        if obj is not None:
            objects.append(obj)
    assert objects, "expected at least one nested multicolor object"

    objects.sort(key=lambda item: (-len(item["layers"]), item["bbox"][0], item["bbox"][1]))
    peeled_layers = [patch for obj in objects for patch in obj["layers"]]
    assert len(peeled_layers) == len(frames), "number of target frames must match total peeled layers"

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for frame, patch in zip(frames, peeled_layers):
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        patch_h, patch_w = patch.shape
        assert patch_h <= inner_h and patch_w <= inner_w, "peeled layer must fit inside target frame"
        top = r0 + 1 + (inner_h - patch_h) // 2
        left = c0 + 1 + (inner_w - patch_w) // 2
        _paste(out, patch, top, left, transparent=bg)

    return out


def _make_layer_patch(height: int, width: int, color: int, bg: int, *, solid: bool) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    if solid:
        patch[:, :] = color
        return patch
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _build_object(inner_h: int, inner_w: int, colors_inner_to_outer: list[int], bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    layers: list[np.ndarray] = []
    for depth, color in enumerate(colors_inner_to_outer):
        height = inner_h + 2 * depth
        width = inner_w + 2 * depth
        layers.append(_make_layer_patch(height, width, color, bg, solid=(depth == 0)))

    patch = np.full(layers[-1].shape, bg, dtype=int)
    for layer in reversed(layers):
        top = (patch.shape[0] - layer.shape[0]) // 2
        left = (patch.shape[1] - layer.shape[1]) // 2
        _paste(patch, layer, top, left, transparent=bg)
    return patch, layers


def _starts(limit: int, size: int) -> list[int]:
    slack = limit - size
    if slack < 0:
        return []
    if slack >= 2:
        return list(range(1, slack))
    return list(range(slack + 1))


def _separated(start_a: int, len_a: int, start_b: int, len_b: int, pad: int = 1) -> bool:
    return start_a + len_a + pad <= start_b or start_b + len_b + pad <= start_a


def _choose_specs(rng: random.Random) -> list[dict]:
    branch = rng.choice(("single2", "single3", "double2"))
    if branch == "single2":
        return [{"layers": 2, "inner_h": rng.choice((1, 3, 5)), "inner_w": rng.choice((1, 3, 5))}]
    if branch == "single3":
        return [{"layers": 3, "inner_h": rng.choice((1, 3)), "inner_w": rng.choice((1, 3, 5))}]
    return [
        {"layers": 2, "inner_h": 1, "inner_w": rng.choice((1, 3, 5))},
        {"layers": 2, "inner_h": 1, "inner_w": rng.choice((1, 3, 5))},
    ]


def _choose_object_positions(
    rng: random.Random,
    grid_h: int,
    left_zone_w: int,
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    if len(patches) == 1:
        patch = patches[0]
        row_choices = _starts(grid_h, patch.shape[0])
        col_choices = _starts(left_zone_w, patch.shape[1])
        if not row_choices or not col_choices:
            return None
        return [(rng.choice(row_choices), rng.choice(col_choices))]

    row_pairs: list[tuple[int, int]] = []
    first, second = patches
    first_rows = _starts(grid_h, first.shape[0])
    second_rows = _starts(grid_h, second.shape[0])
    for r0 in first_rows:
        for r1 in second_rows:
            if _separated(r0, first.shape[0], r1, second.shape[0], pad=1):
                row_pairs.append((r0, r1))
    if not row_pairs:
        return None

    cols: list[int] = []
    for patch in patches:
        col_choices = _starts(left_zone_w, patch.shape[1])
        if not col_choices:
            return None
        cols.append(rng.choice(col_choices))
    r0, r1 = rng.choice(row_pairs)
    return [(r0, cols[0]), (r1, cols[1])]


def _build_expected(
    grid: np.ndarray,
    objects: list[dict],
    frame_positions: list[tuple[int, int]],
    frame_inner_h: int,
    frame_inner_w: int,
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    ordered_objects = sorted(objects, key=lambda item: (-len(item["layers"]), item["row"], item["col"]))
    for obj in ordered_objects:
        patch = obj["patch"]
        row = obj["row"]
        col = obj["col"]
        out[row : row + patch.shape[0], col : col + patch.shape[1]] = bg

    peeled_layers = [layer for obj in ordered_objects for layer in obj["layers"]]
    for (row, col), patch in zip(frame_positions, peeled_layers):
        top = row + 1 + (frame_inner_h - patch.shape[0]) // 2
        left = col + 1 + (frame_inner_w - patch.shape[1]) // 2
        _paste(out, patch, top, left, transparent=bg)
    return out


def _valid_bg_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    if int(values[int(order[0])]) != bg:
        return False
    if len(order) == 1:
        return True
    return int(counts[int(order[0])]) > int(counts[int(order[1])])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _choose_specs(rng)
    total_layers = sum(spec["layers"] for spec in specs)

    bg = rng.choice([color for color in range(10) if color != FRAME_COLOR])
    palette = [color for color in range(10) if color not in (bg, FRAME_COLOR)]
    layer_colors = rng.sample(palette, total_layers)

    objects: list[dict] = []
    offset = 0
    for spec in specs:
        colors = layer_colors[offset : offset + spec["layers"]]
        offset += spec["layers"]
        rng.shuffle(colors)
        patch, layers = _build_object(spec["inner_h"], spec["inner_w"], colors, bg)
        objects.append({"patch": patch, "layers": layers})

    frame_inner_h = max(obj["patch"].shape[0] for obj in objects)
    frame_inner_w = max(obj["patch"].shape[1] for obj in objects)
    frame_patch = _make_layer_patch(frame_inner_h + 2, frame_inner_w + 2, FRAME_COLOR, bg, solid=False)
    frame_gap = 1
    stack_h = total_layers * frame_patch.shape[0] + (total_layers - 1) * frame_gap

    left_min_h = sum(obj["patch"].shape[0] for obj in objects) + max(0, len(objects) - 1) * 2
    min_h = max(stack_h, left_min_h)
    if min_h > 30:
        return None
    grid_h = rng.randint(min_h, min(30, min_h + 6))

    separator = rng.randint(2, 4)
    right_margin = rng.randint(1, 2)
    max_obj_w = max(obj["patch"].shape[1] for obj in objects)
    min_left_zone_w = max(max_obj_w + 2, frame_patch.shape[1] + right_margin - separator)
    max_left_zone_w = 30 - separator - frame_patch.shape[1] - right_margin
    if max_left_zone_w < min_left_zone_w:
        return None
    left_zone_w = rng.randint(min_left_zone_w, min(max_left_zone_w, min_left_zone_w + 6))
    grid_w = left_zone_w + separator + frame_patch.shape[1] + right_margin
    frame_left = left_zone_w + separator
    if frame_left < grid_w // 2:
        return None

    positions = _choose_object_positions(rng, grid_h, left_zone_w, [obj["patch"] for obj in objects])
    if positions is None:
        return None
    for obj, (row, col) in zip(objects, positions):
        obj["row"] = row
        obj["col"] = col

    frame_top = rng.randint(0, grid_h - stack_h)
    frame_positions = [
        (frame_top + idx * (frame_patch.shape[0] + frame_gap), frame_left)
        for idx in range(total_layers)
    ]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for obj in objects:
        _paste(grid, obj["patch"], obj["row"], obj["col"])
    for row, col in frame_positions:
        _paste(grid, frame_patch, row, col, transparent=bg)

    expected = _build_expected(grid, objects, frame_positions, frame_inner_h, frame_inner_w, bg)
    if np.array_equal(expected, grid):
        return None
    if not _valid_bg_mode(grid, bg):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
