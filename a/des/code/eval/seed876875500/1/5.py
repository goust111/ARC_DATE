from __future__ import annotations

import numpy as np


def _perimeter_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height - 1):
        coords.append((r, width - 1))
    for c in range(width - 1, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _depth_map(height: int, width: int) -> np.ndarray:
    rows = np.minimum(np.arange(height), np.arange(height - 1, -1, -1))[:, None]
    cols = np.minimum(np.arange(width), np.arange(width - 1, -1, -1))[None, :]
    return np.minimum(rows, cols)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_length_by_color(grid: np.ndarray) -> dict[int, int]:
    values = [int(grid[r, c]) for r, c in _perimeter_coords(*grid.shape)]
    assert values[0] != values[-1], "border reading must start at a segment boundary"

    lengths: dict[int, int] = {}
    color = values[0]
    run = 1
    for value in values[1:]:
        if value == color:
            run += 1
            continue
        assert color not in lengths, "expected each border color to appear in one segment"
        lengths[color] = run
        color = value
        run = 1
    assert color not in lengths, "expected each border color to appear in one segment"
    lengths[color] = run
    return lengths


def _ring_thickness(region: np.ndarray, color: int) -> int:
    height, width = map(int, region.shape)
    depth = 0
    while depth < (min(height, width) + 1) // 2:
        r0, c0 = depth, depth
        r1, c1 = height - 1 - depth, width - 1 - depth
        if not np.all(region[r0, c0 : c1 + 1] == color):
            break
        if not np.all(region[r1, c0 : c1 + 1] == color):
            break
        if not np.all(region[r0 : r1 + 1, c0] == color):
            break
        if not np.all(region[r0 : r1 + 1, c1] == color):
            break
        depth += 1
    return depth


def _extract_rings(inner: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    rings: list[dict[str, int | tuple[int, int, int, int]]] = []
    bbox = _bbox(inner != bg)
    while bbox is not None:
        r0, c0, r1, c1 = bbox
        region = inner[r0 : r1 + 1, c0 : c1 + 1]
        color = int(region[0, 0])
        assert color != bg, "outer ring must start at the current bounding box corner"

        thickness = _ring_thickness(region, color)
        assert thickness > 0, "failed to detect a positive ring thickness"

        ring_mask = _depth_map(*region.shape) < thickness
        assert np.all(region[ring_mask] == color), "ring shell must be monochromatic"
        rings.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "color": color,
                "thickness": thickness,
            }
        )

        nr0, nc0 = r0 + thickness, c0 + thickness
        nr1, nc1 = r1 - thickness, c1 - thickness
        if nr0 > nr1 or nc0 > nc1:
            break

        sub = inner[nr0 : nr1 + 1, nc0 : nc1 + 1]
        if not np.any(sub != bg):
            break
        assert np.all(sub[0, :] != bg)
        assert np.all(sub[-1, :] != bg)
        assert np.all(sub[:, 0] != bg)
        assert np.all(sub[:, -1] != bg)
        bbox = (nr0, nc0, nr1, nc1)

    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """Peel each nested rectangular ring by the thickness encoded by its border color segment."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert height >= 7 and width >= 7

    border_lengths = _border_length_by_color(g)
    border_colors = set(border_lengths)
    bg_candidates = set(map(int, np.unique(g))) - border_colors
    assert len(bg_candidates) == 1, "expected exactly one non-border background color"
    bg = int(next(iter(bg_candidates)))

    inner = g[1:-1, 1:-1]
    rings = _extract_rings(inner, bg)
    assert rings, "expected at least one nested ring"

    out = g.copy()
    for ring in rings:
        color = int(ring["color"])
        thickness = int(ring["thickness"])
        assert color in border_lengths, "ring color must be encoded on the border"
        peel = min(thickness, int(border_lengths[color]))
        if peel == 0:
            continue

        r0, c0, r1, c1 = ring["bbox"]
        region = out[1 + r0 : 2 + r1, 1 + c0 : 2 + c1]
        region[_depth_map(*region.shape) < peel] = bg

    return out


# --------------- generator (o4) ---------------

_RNG = np.random.default_rng()

GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (14, 25)
RING_COUNT_RANGE = (4, 8)
HOLE_SIZE_RANGE = (1, 5)
TOTAL_EXTRA_THICKNESS = (1, 5)
GENERATION_RETRIES = 400


def _split_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = np.sort(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    values = np.diff(np.concatenate(([0], cuts, [total])))
    return [int(v) for v in values]


def _sample_border_lengths(thicknesses: list[int], perimeter: int) -> tuple[list[int], list[str]] | None:
    count = len(thicknesses)
    partial_candidates = [i for i, t in enumerate(thicknesses) if t >= 2]
    if not partial_candidates:
        return None

    relations = ["full"] * count
    partial_index = int(_RNG.choice(partial_candidates))
    relations[partial_index] = "partial"

    if count >= 4:
        extra_full = [i for i in range(count) if i != partial_index]
        second_full = int(_RNG.choice(extra_full))
        relations[second_full] = "full"

    remaining = [i for i in range(count) if i != partial_index]
    if remaining and _RNG.random() < 0.6:
        exact_index = int(_RNG.choice(remaining))
        relations[exact_index] = "exact"

    for i, thickness in enumerate(thicknesses):
        if i == partial_index or relations[i] == "exact":
            continue
        valid = ["full", "exact"]
        if thickness >= 2:
            valid.append("partial")
        relations[i] = str(_RNG.choice(valid))

    if "full" not in relations:
        replaceable = [i for i in range(count) if relations[i] != "partial"]
        if not replaceable:
            return None
        relations[int(_RNG.choice(replaceable))] = "full"

    base_lengths: list[int] = []
    full_indices: list[int] = []
    for i, (thickness, relation) in enumerate(zip(thicknesses, relations)):
        if relation == "partial":
            length = int(_RNG.integers(1, thickness))
        elif relation == "exact":
            length = thickness
        else:
            length = thickness + int(_RNG.integers(1, 5))
            full_indices.append(i)
        base_lengths.append(length)

    remainder = int(perimeter - sum(base_lengths))
    if remainder < 0 or not full_indices:
        return None

    extras = _RNG.multinomial(remainder, np.full(len(full_indices), 1 / len(full_indices)))
    lengths = base_lengths[:]
    for idx, extra in zip(full_indices, extras):
        lengths[idx] += int(extra)

    return lengths, relations


def _paint_shell(grid: np.ndarray, bbox: tuple[int, int, int, int], thickness: int, color: int) -> None:
    if thickness <= 0:
        return
    r0, c0, r1, c1 = bbox
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    region[_depth_map(*region.shape) < thickness] = color


def _render_pair(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    height = int(spec["height"])
    width = int(spec["width"])
    bg = int(spec["bg"])
    colors = [int(x) for x in spec["colors"]]
    border_lengths = [int(x) for x in spec["border_lengths"]]
    thicknesses = [int(x) for x in spec["thicknesses"]]
    top = int(spec["top"])
    left = int(spec["left"])
    structure_height = int(spec["structure_height"])
    structure_width = int(spec["structure_width"])

    inp = np.full((height, width), bg, dtype=int)
    perimeter = _perimeter_coords(height, width)
    offset = 0
    for color, length in zip(colors, border_lengths):
        for r, c in perimeter[offset : offset + length]:
            inp[r, c] = color
        offset += length
    assert offset == len(perimeter)

    bbox = (
        top,
        left,
        top + structure_height - 1,
        left + structure_width - 1,
    )
    current_bbox = bbox
    for color, thickness in zip(colors, thicknesses):
        _paint_shell(inp, current_bbox, thickness, color)
        r0, c0, r1, c1 = current_bbox
        current_bbox = (r0 + thickness, c0 + thickness, r1 - thickness, c1 - thickness)

    out = inp.copy()
    current_bbox = bbox
    for color, thickness, peel in zip(colors, thicknesses, border_lengths):
        _paint_shell(out, current_bbox, min(thickness, peel), bg)
        r0, c0, r1, c1 = current_bbox
        current_bbox = (r0 + thickness, c0 + thickness, r1 - thickness, c1 - thickness)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = int(_RNG.integers(0, 10))

    ring_count = int(_RNG.integers(*RING_COUNT_RANGE))
    total_thickness = ring_count + int(_RNG.integers(*TOTAL_EXTRA_THICKNESS))
    thicknesses = _split_positive(total_thickness, ring_count)
    if max(thicknesses) < 2:
        return None

    hole_h = int(_RNG.integers(*HOLE_SIZE_RANGE))
    hole_w = int(_RNG.integers(*HOLE_SIZE_RANGE))
    structure_height = 2 * total_thickness + hole_h
    structure_width = 2 * total_thickness + hole_w

    min_height = max(GRID_H_RANGE[0], structure_height + 2)
    min_width = max(GRID_W_RANGE[0], structure_width + 2)
    if min_height >= GRID_H_RANGE[1] or min_width >= GRID_W_RANGE[1]:
        return None

    height = int(_RNG.integers(min_height, GRID_H_RANGE[1]))
    width = int(_RNG.integers(min_width, GRID_W_RANGE[1]))

    row_slack = height - 2 - structure_height
    col_slack = width - 2 - structure_width
    top = 1 + int(_RNG.integers(0, row_slack + 1))
    left = 1 + int(_RNG.integers(0, col_slack + 1))

    color_pool = [color for color in range(10) if color != bg]
    colors = [int(x) for x in _RNG.choice(color_pool, size=ring_count, replace=False)]

    perimeter = 2 * height + 2 * width - 4
    border_sample = _sample_border_lengths(thicknesses, perimeter)
    if border_sample is None:
        return None
    border_lengths, relations = border_sample

    if "partial" not in relations or "full" not in relations:
        return None

    spec = {
        "height": height,
        "width": width,
        "bg": bg,
        "colors": colors,
        "border_lengths": border_lengths,
        "thicknesses": thicknesses,
        "top": top,
        "left": left,
        "structure_height": structure_height,
        "structure_width": structure_width,
    }
    inp, expected = _render_pair(spec)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid peeling-rings ARC input")
