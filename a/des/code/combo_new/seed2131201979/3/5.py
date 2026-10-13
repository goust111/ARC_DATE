from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_BG = 0
_DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
_RIGHT_ARROW_PATTERNS = (
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
)
_BLOCK_SPECS: list[tuple[int, int, str | None, int]] = []
for _h in range(3, 6):
    for _w in range(2, 6):
        _BLOCK_SPECS.append((_h, _w, None, _h * _w))
        for _notch in ("tl", "tr", "bl", "br"):
            _BLOCK_SPECS.append((_h, _w, _notch, _h * _w - 1))

_GEN_MAX_ATTEMPTS = 400
_GEN_BASE_SIZE = 30


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in _DIR_VECS.values():
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR_VECS.values():
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(coords)


def _infer_direction(
    component_coords: list[tuple[int, int]],
    arrow_coords: list[tuple[int, int]],
) -> str:
    assert len(arrow_coords) == 3
    comp_r0, comp_c0, comp_r1, comp_c1 = _bbox(component_coords)
    arr_r0, arr_c0, arr_r1, arr_c1 = _bbox(arrow_coords)
    assert arr_r1 - arr_r0 == 1 and arr_c1 - arr_c0 == 1

    occ = np.zeros((2, 2), dtype=int)
    for r, c in arrow_coords:
        occ[r - arr_r0, c - arr_c0] = 1
    assert int(occ.sum()) == 3

    row_counts = occ.sum(axis=1).tolist()
    col_counts = occ.sum(axis=0).tolist()
    assert sorted(row_counts) == [1, 2]
    assert sorted(col_counts) == [1, 2]

    candidates = [
        "up" if row_counts[0] == 1 else "down",
        "left" if col_counts[0] == 1 else "right",
    ]

    distances = {
        "up": arr_r0 - comp_r0,
        "down": comp_r1 - arr_r1,
        "left": arr_c0 - comp_c0,
        "right": comp_c1 - arr_c1,
    }
    best_dist = min(distances[d] for d in candidates)
    nearest = [d for d in candidates if distances[d] == best_dist]
    if len(nearest) == 1:
        return nearest[0]

    comp_set = set(component_coords)

    def exposure(direction: str) -> int:
        dr, dc = _DIR_VECS[direction]
        return sum((r + dr, c + dc) not in comp_set for r, c in arrow_coords)

    best_exp = max(exposure(d) for d in nearest)
    exposed = [d for d in nearest if exposure(d) == best_exp]
    return exposed[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each multicolor object into ranked color layers and fan them out along the arrow direction.

    Input structure:
    - The background is black.
    - Each foreground object is one 4-connected multicolor component with 2-4 colors.
    - In every object, the least frequent color is a 3-pixel L-shaped arrow marker.

    Rule:
    - For each object, find the least frequent color and read its L-marker direction.
    - Sort that object's color layers by pixel count from fewest to most.
    - Keep the smallest layer fixed; move the 2nd, 3rd, ... layers by 2, 4, ...
      cells along the arrow direction.
    - Paint moved layers in that same order so larger later layers overwrite earlier ones.
    - Any pixels vacated by a moved layer become background.

    Key insight:
    - The smallest color layer is both the arrow token and the first item of the
      count ordering, so one visual feature simultaneously encodes direction and
      the expansion sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full(g.shape, _BG, dtype=int)
    for component in _find_components(g != _BG):
        color_to_coords: dict[int, list[tuple[int, int]]] = {}
        for r, c in component:
            color = int(g[r, c])
            color_to_coords.setdefault(color, []).append((r, c))

        assert 2 <= len(color_to_coords) <= 4
        layer_sizes = sorted(len(coords) for coords in color_to_coords.values())
        assert layer_sizes[0] == 3
        assert len(set(layer_sizes)) == len(layer_sizes)

        arrow_color = min(color_to_coords, key=lambda color: len(color_to_coords[color]))
        direction = _infer_direction(component, color_to_coords[arrow_color])
        dr, dc = _DIR_VECS[direction]

        layers = sorted(color_to_coords.items(), key=lambda item: len(item[1]))
        for idx, (color, coords) in enumerate(layers):
            shift_r = dr * 2 * idx
            shift_c = dc * 2 * idx
            for r, c in coords:
                rr = r + shift_r
                cc = c + shift_c
                assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
                out[rr, cc] = int(color)

    return out


def _block_cells(spec: tuple[int, int, str | None, int], top: int, left: int) -> set[tuple[int, int]]:
    h, w, notch, _area = spec
    cells = {(top + r, left + c) for r in range(h) for c in range(w)}
    if notch == "tl":
        cells.remove((top, left))
    elif notch == "tr":
        cells.remove((top, left + w - 1))
    elif notch == "bl":
        cells.remove((top + h - 1, left))
    elif notch == "br":
        cells.remove((top + h - 1, left + w - 1))
    return cells


def _rotate_point(r: int, c: int, height: int, width: int, direction: str) -> tuple[int, int]:
    if direction == "right":
        return r, c
    if direction == "down":
        return c, height - 1 - r
    if direction == "left":
        return height - 1 - r, width - 1 - c
    assert direction == "up"
    return width - 1 - c, r


def _rotate_layers(
    layers: list[set[tuple[int, int]]],
    height: int,
    width: int,
    direction: str,
) -> tuple[int, int, list[set[tuple[int, int]]]]:
    if direction in ("right", "left"):
        new_h, new_w = height, width
    else:
        new_h, new_w = width, height

    rotated: list[set[tuple[int, int]]] = []
    for layer in layers:
        rotated.append({_rotate_point(r, c, height, width, direction) for r, c in layer})
    return new_h, new_w, rotated


def _normalize_layers(
    input_layers: list[set[tuple[int, int]]],
    output_layers: list[set[tuple[int, int]]],
) -> tuple[int, int, list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    all_coords = set().union(*input_layers, *output_layers)
    r0, c0, r1, c1 = _bbox(all_coords)

    def shifted(layers: list[set[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
        return [
            sorted((r - r0, c - c0) for r, c in layer)
            for layer in layers
        ]

    return r1 - r0 + 1, c1 - c0 + 1, shifted(input_layers), shifted(output_layers)


def _build_canonical_right_object(num_layers: int) -> tuple[int, int, list[set[tuple[int, int]]]] | None:
    for _ in range(120):
        specs: list[tuple[int, int, str | None, int]] = []
        prev_area = 3
        total_width = 2
        ok = True
        for idx in range(num_layers - 1):
            max_remaining_width = 5 * (num_layers - idx - 2)
            candidates = [
                spec
                for spec in _BLOCK_SPECS
                if spec[3] > prev_area
                and (idx > 0 or spec[0] >= 4)
                and total_width + spec[1] + max_remaining_width <= 17
            ]
            if not candidates:
                ok = False
                break
            spec = random.choice(candidates)
            specs.append(spec)
            prev_area = spec[3]
            total_width += spec[1]
        if not ok:
            continue

        height = max(max(spec[0] for spec in specs), 4) + random.randint(2, 3)
        arrow_top = height // 2 - 1
        row_ranges: list[tuple[int, int]] = []
        tops: list[int] = []

        for idx, spec in enumerate(specs):
            h = spec[0]
            base_top = (height - h) // 2
            options = sorted({
                top
                for top in (base_top - 1, base_top, base_top + 1)
                if 0 <= top <= height - h
            })
            if idx == 0:
                options = [top for top in options if top <= arrow_top - 1 and top + h - 1 >= arrow_top + 2]
            if row_ranges:
                prev_r0, prev_r1 = row_ranges[-1]
                options = [top for top in options if max(top, prev_r0) <= min(top + h - 1, prev_r1)]
            if not options:
                ok = False
                break
            top = random.choice(options)
            tops.append(top)
            row_ranges.append((top, top + h - 1))
        if not ok:
            continue

        layers: list[set[tuple[int, int]]] = []
        cursor = total_width - 2
        for spec, top in zip(specs, tops):
            left = cursor - spec[1]
            layers.append(_block_cells(spec, top, left))
            cursor = left

        arrow_left = total_width - 2
        arrow = {
            (arrow_top + dr, arrow_left + dc)
            for dr, dc in random.choice(_RIGHT_ARROW_PATTERNS)
        }
        all_layers = [arrow] + layers
        union = set().union(*all_layers)
        if not _is_connected(union):
            continue

        comp_r0, _comp_c0, comp_r1, comp_c1 = _bbox(union)
        arrow_r0, _arrow_c0, arrow_r1, arrow_c1 = _bbox(arrow)
        if arrow_c1 != comp_c1:
            continue
        if arrow_r0 == comp_r0 or arrow_r1 == comp_r1:
            continue

        return height, total_width, all_layers
    return None


def _build_object() -> dict | None:
    num_layers = random.randint(2, 4)
    built = _build_canonical_right_object(num_layers)
    if built is None:
        return None
    height, width, layers = built
    direction = random.choice(tuple(_DIR_VECS))
    _new_h, _new_w, input_layers = _rotate_layers(layers, height, width, direction)

    dr, dc = _DIR_VECS[direction]
    output_layers = [
        {(r + 2 * idx * dr, c + 2 * idx * dc) for r, c in layer}
        for idx, layer in enumerate(input_layers)
    ]

    box_h, box_w, norm_input, norm_output = _normalize_layers(input_layers, output_layers)
    colors = random.sample(range(1, 10), num_layers)

    local_in = np.zeros((box_h, box_w), dtype=int)
    local_out = np.zeros((box_h, box_w), dtype=int)
    for color, layer in zip(colors, norm_input):
        for r, c in layer:
            local_in[r, c] = int(color)
    for color, layer in zip(colors, norm_output):
        for r, c in layer:
            local_out[r, c] = int(color)

    try:
        if not np.array_equal(main(local_in), local_out):
            return None
    except Exception:
        return None

    return {
        "box_h": box_h,
        "box_w": box_w,
        "colors": colors,
        "input_layers": norm_input,
        "output_layers": norm_output,
    }


def _boxes_too_close(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_objects(objects: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    for obj in sorted(objects, key=lambda item: item["box_h"] * item["box_w"], reverse=True):
        candidates: list[tuple[int, int]] = []
        for top in range(_GEN_BASE_SIZE - obj["box_h"] + 1):
            for left in range(_GEN_BASE_SIZE - obj["box_w"] + 1):
                box = (top, left, top + obj["box_h"] - 1, left + obj["box_w"] - 1)
                if all(not _boxes_too_close(box, other["box"], gap=1) for other in placed):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placed.append(
            {
                "top": top,
                "left": left,
                "box": (top, left, top + obj["box_h"] - 1, left + obj["box_w"] - 1),
                "obj": obj,
            }
        )
    return placed


def _paint_scene(placed: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((_GEN_BASE_SIZE, _GEN_BASE_SIZE), dtype=int)
    expected = np.zeros((_GEN_BASE_SIZE, _GEN_BASE_SIZE), dtype=int)

    for item in placed:
        top = item["top"]
        left = item["left"]
        obj = item["obj"]
        for color, layer in zip(obj["colors"], obj["input_layers"]):
            for r, c in layer:
                grid[top + r, left + c] = int(color)
        for color, layer in zip(obj["colors"], obj["output_layers"]):
            for r, c in layer:
                expected[top + r, left + c] = int(color)
    return grid, expected


def _crop_scene(grid: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere((grid != _BG) | (expected != _BG))
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())

    top_margin = random.randint(0, min(3, r0))
    left_margin = random.randint(0, min(3, c0))
    bottom_margin = random.randint(0, min(3, grid.shape[0] - 1 - r1))
    right_margin = random.randint(0, min(3, grid.shape[1] - 1 - c1))

    rs = slice(r0 - top_margin, r1 + bottom_margin + 1)
    cs = slice(c0 - left_margin, c1 + right_margin + 1)
    return grid[rs, cs], expected[rs, cs]


def _gen_attempt() -> np.ndarray | None:
    num_objects = random.randint(1, 3)
    objects = []
    for _ in range(num_objects):
        obj = _build_object()
        if obj is None:
            return None
        objects.append(obj)

    placed = _place_objects(objects)
    if placed is None:
        return None

    grid, expected = _paint_scene(placed)
    grid, expected = _crop_scene(grid, expected)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
