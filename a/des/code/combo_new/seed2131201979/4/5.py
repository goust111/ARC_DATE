import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 500
BACKGROUND = 0


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).reshape(-1)))
    if BACKGROUND in counts:
        return BACKGROUND
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame(mask: np.ndarray) -> bool:
    height, width = map(int, mask.shape)
    if height < 3 or width < 3:
        return False
    frame = np.zeros((height, width), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return bool(np.array_equal(mask, frame))


def _extract_layers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    bg = _background_color(g)
    seen = np.zeros((height, width), dtype=bool)
    layers: list[dict] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), height, width):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            r0, c0, r1, c1 = _bbox_from_coords(coords)
            submask = g[r0 : r1 + 1, c0 : c1 + 1] == color
            if _is_frame(submask):
                kind = "frame"
            elif bool(submask.all()):
                kind = "solid"
            else:
                raise AssertionError("non-background component is neither a 1-pixel frame nor a solid rectangle")

            layers.append(
                {
                    "kind": kind,
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )

    assert layers, "expected at least one non-background layer"
    layers.sort(key=lambda layer: (-int(layer["area"]), -int(layer["height"]), -int(layer["width"])))

    solid_count = sum(layer["kind"] == "solid" for layer in layers)
    assert solid_count <= 1, "expected at most one solid core"
    if solid_count:
        assert layers[-1]["kind"] == "solid", "solid core must be innermost"

    for outer, inner in zip(layers, layers[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "layers must be strictly nested"

    return layers


def _compose_output(grid: np.ndarray, layers: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out_width = int(layers[0]["width"])
    out_height = sum(int(layer["height"]) for layer in layers) + max(0, len(layers) - 1)
    out = np.full((out_height, out_width), bg, dtype=int)

    row = 0
    for index, layer in enumerate(layers):
        r0, c0, r1, c1 = layer["bbox"]
        tile = g[r0 : r1 + 1, c0 : c1 + 1]
        col = (out_width - tile.shape[1]) // 2
        out[row : row + tile.shape[0], col : col + tile.shape[1]] = tile
        row += tile.shape[0]
        if index + 1 < len(layers):
            row += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unpack concentric rectangular layers into vertically stacked rows ordered from outermost to innermost.

    Input structure:
    - The background is black.
    - A single nested figure is present: several axis-aligned 1-pixel-thick rectangular frames with different colors.
    - The frames are strictly nested with background-filled gaps between them, and the innermost gap may contain one solid monochrome rectangle as a core.

    Rule:
    - Detect every rectangular layer, treating each colored ring as one frame layer and the optional solid inner rectangle as the final core layer.
    - Sort the layers by nesting depth from outermost to innermost.
    - Build an output whose width equals the outermost frame width and whose height is the sum of all layer heights plus one background row between consecutive layers.
    - Place each layer unchanged, horizontally centered in its own row block, stacking from top to bottom.

    Key insight:
    - The apparent multi-color target is a hierarchy of separate rectangular components, so solving requires first decomposing the nesting structure and then re-laying those layers in depth order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    layers = _extract_layers(g)
    return _compose_output(g, layers)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _draw_solid(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _sample_inner_object(rng: random.Random, with_core: bool) -> tuple[str, int, int]:
    if with_core:
        return "solid", rng.randint(1, 4), rng.randint(1, 6)
    return "frame", rng.randint(3, 6), rng.randint(3, 8)


def _sample_expansion(rng: random.Random, wide: bool) -> tuple[int, int, int, int]:
    top = rng.randint(2, 3)
    bottom = rng.randint(2, 3)
    if wide:
        left = rng.randint(2, 5)
        right = rng.randint(2, 5)
    else:
        left = rng.randint(2, 4)
        right = rng.randint(2, 4)
    return top, bottom, left, right


def _build_nested_specs(
    rng: random.Random,
) -> tuple[list[dict], list[tuple[int, int, int, int]], tuple[int, int]]:
    with_core = rng.random() < 0.5
    frame_count = 2 if with_core else rng.choice([2, 3])
    base_kind, base_h, base_w = _sample_inner_object(rng, with_core)
    current_h, current_w = base_h, base_w
    expansions: list[tuple[int, int, int, int]] = []
    expansion_count = frame_count if with_core else frame_count - 1

    for index in range(expansion_count):
        wide = index == expansion_count - 1 or rng.random() < 0.5
        exp = _sample_expansion(rng, wide=wide)
        current_h += exp[0] + exp[1]
        current_w += exp[2] + exp[3]
        expansions.append(exp)

    dims: list[tuple[int, int]] = [(current_h, current_w)]
    walk_h, walk_w = current_h, current_w
    for top, bottom, left, right in reversed(expansions):
        walk_h -= top + bottom
        walk_w -= left + right
        dims.append((walk_h, walk_w))

    total_layers = frame_count + (1 if with_core else 0)
    kinds = ["frame"] * total_layers
    if with_core:
        kinds[-1] = base_kind

    output_height = sum(h for h, _ in dims) + len(dims) - 1
    outer_h, outer_w = dims[0]
    if output_height > 30 or outer_h > 30 or outer_w > 30:
        raise ValueError("candidate exceeds ARC size limits")

    layer_specs = []
    for kind, (height, width) in zip(kinds, dims):
        layer_specs.append({"kind": kind, "height": height, "width": width})
    return layer_specs, expansions, (outer_h, outer_w)


def _place_layers(
    layer_specs: list[dict],
    expansions: list[tuple[int, int, int, int]],
    outer_origin: tuple[int, int],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    r0, c0 = map(int, outer_origin)
    outer_h = int(layer_specs[0]["height"])
    outer_w = int(layer_specs[0]["width"])
    objects: list[tuple[str, tuple[int, int, int, int]]] = []

    bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    objects.append(("frame", bbox))
    current = bbox
    for index, (top, bottom, left, right) in enumerate(reversed(expansions)):
        nr0 = int(current[0] + top)
        nc0 = int(current[1] + left)
        nr1 = int(current[2] - bottom)
        nc1 = int(current[3] - right)
        current = (nr0, nc0, nr1, nc1)
        if index + 1 < len(layer_specs):
            objects.append((str(layer_specs[index + 1]["kind"]), current))
    return objects


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    try:
        layer_specs, expansions, (outer_h, outer_w) = _build_nested_specs(rng)
    except ValueError:
        return None

    remaining_h = 30 - outer_h
    remaining_w = 30 - outer_w
    if remaining_h < 0 or remaining_w < 0:
        return None

    margin_top = rng.randint(0, min(5, remaining_h))
    margin_bottom = rng.randint(0, remaining_h - margin_top)
    margin_left = rng.randint(0, min(5, remaining_w))
    margin_right = rng.randint(0, remaining_w - margin_left)
    grid_h = outer_h + margin_top + margin_bottom
    grid_w = outer_w + margin_left + margin_right

    color_count = len(layer_specs)
    palette = rng.sample(range(1, 10), color_count)
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    objects = _place_layers(layer_specs, expansions, (margin_top, margin_left))
    for (kind, bbox), color in zip(objects, palette):
        if kind == "frame":
            _draw_frame(grid, bbox, color)
        else:
            _draw_solid(grid, bbox, color)

    extracted = _extract_layers(grid)
    if len(extracted) != len(objects):
        return None
    if [layer["kind"] for layer in extracted] != [kind for kind, _ in objects]:
        return None

    out = main(grid)
    if out.shape[1] != outer_w or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
