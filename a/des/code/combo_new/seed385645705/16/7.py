from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

OBJECT_COUNT_RANGE = (2, 4)
TEMPLATE_LAYER_RANGE = (3, 5)
INNER_HEIGHT_RANGE = (1, 4)
INNER_WIDTH_RANGE = (1, 5)
ROW_GAP_RANGE = (2, 4)
COL_GAP_RANGE = (2, 4)
GEN_RETRIES = 400


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append({"coords": arr, "bbox": (r0, c0, r1, c1)})

    return components


def _extract_layers(rect: np.ndarray) -> list[int]:
    sub = np.asarray(rect, dtype=int)
    layers: list[int] = []

    while True:
        assert sub.size > 0
        first = int(sub[0, 0])
        if np.all(sub == first):
            layers.append(first)
            return layers

        h, w = sub.shape
        assert h >= 3 and w >= 3
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border_colors = np.unique(sub[border])
        assert len(border_colors) == 1
        layers.append(int(border_colors[0]))
        sub = sub[1:-1, 1:-1]


def _paint_layers(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    dist = np.minimum.reduce((rr, cc, h - 1 - rr, w - 1 - cc))
    idx = np.minimum(dist, len(colors) - 1)
    return np.array(colors, dtype=int)[idx]


def _parse_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for comp in _find_components(grid != BG):
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        assert np.all(sub != BG)
        assert comp["coords"].shape[0] == sub.size
        layers = _extract_layers(sub)
        objects.append({"bbox": (r0, c0, r1, c1), "layers": layers, "shape": sub.shape})
    objects.sort(key=lambda obj: obj["bbox"])
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every concentric rectangle object so its outer-to-inner colors follow the deepest object's layer order.

    Input structure:
    - The background is black.
    - Foreground objects are separated filled rectangles made of nested one-cell-thick rectangular layers.
    - Each object has a uniform color on every outer frame, and its innermost layer is a solid rectangle that may be larger than 1x1.

    Rule:
    - Split the non-black pixels into connected rectangular objects and read each object's color sequence from the outer border inward.
    - Identify the object with the greatest number of layers and use its outer-to-inner color sequence as the template.
    - Leave the template object unchanged; recolor every other object so its first N layers use the first N template colors, where N is that object's layer count.

    Key insight:
    - Colors are not matched by identity or position; the only selector is the structural property "has the most nested layers", and its layer order propagates to all smaller concentric rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _parse_objects(g)
    assert objects

    template = max(objects, key=lambda obj: len(obj["layers"]))
    template_colors = list(template["layers"])

    out = g.copy()
    for obj in objects:
        if obj is template:
            continue
        r0, c0, r1, c1 = obj["bbox"]
        new_patch = _paint_layers(obj["shape"], template_colors[: len(obj["layers"])])
        out[r0 : r1 + 1, c0 : c1 + 1] = new_patch

    return out


def _sample_sequence(length: int, forbidden: list[int] | None = None) -> list[int]:
    forbidden = forbidden or []
    for _ in range(100):
        seq = random.sample(PALETTE, length)
        if seq != forbidden:
            return seq
    raise RuntimeError("failed to sample a distinct color sequence")


def _build_patch(inner_h: int, inner_w: int, layers: int, colors: list[int]) -> np.ndarray:
    h = inner_h + 2 * (layers - 1)
    w = inner_w + 2 * (layers - 1)
    return _paint_layers((h, w), colors)


def _layout_objects(patches: list[np.ndarray]) -> tuple[dict[int, tuple[int, int]], int, int] | None:
    n = len(patches)
    row_options = [1] if n == 1 else [1, 2]
    random.shuffle(row_options)

    for row_count in row_options:
        for _ in range(60):
            order = list(range(n))
            random.shuffle(order)
            rows: list[list[int]] = [[] for _ in range(row_count)]
            row_widths = [0] * row_count

            for i, idx in enumerate(order[:row_count]):
                rows[i].append(idx)
                row_widths[i] = int(patches[idx].shape[1])

            for idx in order[row_count:]:
                target = min(range(row_count), key=lambda rid: (row_widths[rid], random.random()))
                rows[target].append(idx)
                row_widths[target] += int(patches[idx].shape[1])

            for row in rows:
                random.shuffle(row)

            placements: dict[int, tuple[int, int]] = {}
            y_cursor = 0
            packed_w = 0

            for rid, row in enumerate(rows):
                row_height = max(int(patches[idx].shape[0]) for idx in row)
                x_cursor = 0

                for j, idx in enumerate(row):
                    if j > 0:
                        x_cursor += random.randint(*COL_GAP_RANGE)
                    patch = patches[idx]
                    patch_h, patch_w = map(int, patch.shape)
                    y_offset = random.randint(0, row_height - patch_h)
                    placements[idx] = (y_cursor + y_offset, x_cursor)
                    x_cursor += patch_w

                packed_w = max(packed_w, x_cursor)
                y_cursor += row_height
                if rid + 1 < row_count:
                    y_cursor += random.randint(*ROW_GAP_RANGE)

            packed_h = y_cursor
            if packed_h <= 28 and packed_w <= 28:
                return placements, packed_h, packed_w

    return None


def _build_expected(grid: np.ndarray, objects: list[dict], template_colors: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for obj in objects:
        if obj["is_template"]:
            continue
        r0, c0 = obj["pos"]
        patch = _build_patch(obj["inner_h"], obj["inner_w"], obj["layers"], template_colors[: obj["layers"]])
        h, w = patch.shape
        out[r0 : r0 + h, c0 : c0 + w] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    inner_h = random.randint(*INNER_HEIGHT_RANGE)
    inner_w = random.randint(*INNER_WIDTH_RANGE)
    template_layers = random.randint(*TEMPLATE_LAYER_RANGE)

    layer_counts = [template_layers]
    for _ in range(n_objects - 1):
        layer_counts.append(random.randint(1, template_layers - 1))
    random.shuffle(layer_counts)

    template_index = layer_counts.index(template_layers)
    template_colors = _sample_sequence(template_layers)

    specs = []
    for idx, layers in enumerate(layer_counts):
        if idx == template_index:
            colors = list(template_colors)
            is_template = True
        else:
            colors = _sample_sequence(layers, forbidden=template_colors[:layers])
            is_template = False
        patch = _build_patch(inner_h, inner_w, layers, colors)
        specs.append(
            {
                "layers": layers,
                "colors": colors,
                "patch": patch,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "is_template": is_template,
            }
        )

    layout = _layout_objects([spec["patch"] for spec in specs])
    if layout is None:
        return None

    placements, packed_h, packed_w = layout
    extra_h = random.randint(2, 30 - packed_h)
    extra_w = random.randint(2, 30 - packed_w)
    top = random.randint(1, extra_h - 1)
    left = random.randint(1, extra_w - 1)
    bottom = extra_h - top
    right = extra_w - left

    grid_h = packed_h + top + bottom
    grid_w = packed_w + left + right
    grid = np.zeros((grid_h, grid_w), dtype=int)

    objects = []
    for idx, spec in enumerate(specs):
        patch = spec["patch"]
        r0 = top + placements[idx][0]
        c0 = left + placements[idx][1]
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        objects.append(
            {
                "layers": spec["layers"],
                "colors": list(spec["colors"]),
                "inner_h": spec["inner_h"],
                "inner_w": spec["inner_w"],
                "is_template": spec["is_template"],
                "pos": (r0, c0),
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
            }
        )

    expected = _build_expected(grid, objects, template_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-rectangle task input")
