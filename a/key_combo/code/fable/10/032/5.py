import hashlib
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
    return tuple((int(r), int(c)) for r, c in arr)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _parse_legends(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
    h, _w = grid.shape
    legends = []
    legend_cells: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    for r in range(h - 1):
        if r in used_rows or r + 1 in used_rows:
            continue
        block = grid[r : r + 2, 0:2]
        if block.shape != (2, 2):
            continue
        if np.any(block == bg):
            continue
        if not (np.all(block[:, 0] == block[0, 0]) and np.all(block[:, 1] == block[0, 1])):
            continue
        src = int(block[0, 0])
        dst = int(block[0, 1])
        if src == dst:
            continue
        legends.append((r, src, dst))
        used_rows.update({r, r + 1})
        legend_cells.update({(r, 0), (r, 1), (r + 1, 0), (r + 1, 1)})
    legends.sort()
    return legends, legend_cells


def _collect_objects(
    grid: np.ndarray,
    bg: int,
    legend_cells: set[tuple[int, int]],
    required_colors: set[int],
) -> dict[int, dict]:
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    objects: dict[int, dict] = {}
    for color in colors:
        comps = _components_of_color(grid, color)
        non_legend = []
        for coords in comps:
            cells = {tuple(map(int, coord)) for coord in coords}
            if cells <= legend_cells:
                continue
            non_legend.append(coords)
        if not non_legend:
            assert color not in required_colors, "legend color must have a matching object"
            continue
        assert len(non_legend) == 1, "each color must have exactly one non-legend object"
        coords = non_legend[0]
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
        objects[color] = {"bbox": bbox, "patch": patch, "coords": coords}
    return objects


def _stamp(out: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= out.shape[0] and c0 + w <= out.shape[1]
    mask = patch != bg
    target_occ = occupied[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(target_occ & mask), "objects must not overlap in the output"
    out_view = out[r0 : r0 + h, c0 : c0 + w]
    out_view[mask] = patch[mask]
    target_occ[mask] = True


def _compose_output_from_objects(
    grid_shape: tuple[int, int],
    bg: int,
    objects: dict[int, dict],
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    occupied = np.zeros(grid_shape, dtype=bool)
    for color, obj in sorted(objects.items()):
        if color in mapping:
            continue
        r0, c0, _r1, _c1 = obj["bbox"]
        _stamp(out, occupied, obj["patch"], r0, c0, bg)
    for color, target_color in mapping.items():
        target_r0, target_c0, _tr1, _tc1 = objects[target_color]["bbox"]
        _stamp(out, occupied, objects[color]["patch"], target_r0, target_c0, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move each legend-listed object to another object's original bbox anchor without recoloring.

    Input structure:
    - The left edge contains one or more 2x2 legend blocks; in each block the left column is a source color A and the right column is a destination-reference color B.
    - Away from the legend, every participating color has exactly one separate monochrome object, and the object shapes are pairwise different.
    - Additional object colors may appear that are not mentioned in the legend.

    Rule:
    - For each legend pair (A, B), translate the A-colored object so that its bbox top-left lands on the input bbox top-left of the B-colored object.
    - All such moves are resolved simultaneously from the input positions; chains and cycles therefore use the original target positions rather than any newly moved positions.
    - Objects whose colors never appear in a legend's left column stay in place, and all legend blocks are erased from the output.

    Key insight:
    - The legend encodes a permutation of object positions, not a recoloring rule: colors and shapes stay attached to their original objects while only the anchor locations change.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    legends, legend_cells = _parse_legends(g, bg)
    assert legends, "expected at least one left-edge legend block"

    mapping: dict[int, int] = {}
    required_colors = set()
    for _r, src, dst in legends:
        assert src not in mapping, "each source color must appear in at most one legend block"
        mapping[src] = dst
        required_colors.add(src)
        required_colors.add(dst)

    objects = _collect_objects(g, bg, legend_cells, required_colors)
    for color in required_colors:
        assert color in objects, "every legend color must correspond to an object"

    return _compose_output_from_objects(g.shape, bg, objects, mapping)


_BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0), (3, 0)),
]

_GEN_COUNTER = 0
_SEEN_OUTPUT_SIGS: set[str] = set()
_DISJOINT_PAIRS = [
    (((0, 0), (0, 1)), ((0, 2), (1, 0), (1, 1), (1, 2))),
    (((0, 0), (0, 1)), ((0, 2), (0, 3), (1, 0), (1, 1), (1, 2))),
    (((0, 0), (0, 1)), ((0, 2), (1, 1), (1, 2), (2, 0), (2, 1))),
]


def _transform_shape(shape: tuple[tuple[int, int], ...], code: int) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(shape, dtype=int)
    if code >= 4:
        arr = arr[:, [0, 1]]
        arr[:, 1] *= -1
        code -= 4
    for _ in range(code):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    return _normalize_coords(arr)


def _choose_shapes(
    rng: np.random.Generator,
    count: int,
    forbidden: set[tuple[tuple[int, int], ...]] | None = None,
) -> list[tuple[tuple[int, int], ...]]:
    seen = set() if forbidden is None else set(forbidden)
    shapes = []
    attempts = 0
    while len(shapes) < count and attempts < 800:
        attempts += 1
        base = _BASE_SHAPES[int(rng.integers(len(_BASE_SHAPES)))]
        shape = _transform_shape(base, int(rng.integers(8)))
        if shape in seen:
            continue
        h, w = _shape_dims(shape)
        if h > 4 or w > 4:
            continue
        seen.add(shape)
        shapes.append(shape)
    assert len(shapes) == count, "failed to sample enough unique shapes"
    return shapes


def _choose_mapping(colors: list[int], mode: int, rng: np.random.Generator) -> dict[int, int]:
    order = list(rng.permutation(colors))
    mapping: dict[int, int]
    if mode == 0:
        a, b, c, d = order[:4]
        mapping = {a: b, b: a, c: d}
    elif mode == 1:
        a, b, c = order[:3]
        mapping = {a: b, b: c, c: a}
    else:
        a, b, c = order[:3]
        mapping = {a: b, b: c}
    assert len(set(mapping)) == len(mapping)
    assert len(set(mapping.values())) == len(mapping)
    return mapping


def _legend_rows(rng: np.random.Generator, height: int, count: int) -> list[int]:
    gaps = [int(rng.integers(1, 4)) for _ in range(count - 1)]
    total = 2 * count + sum(gaps)
    max_start = height - total
    start = int(rng.integers(0, max_start + 1))
    rows = []
    cur = start
    for idx in range(count):
        rows.append(cur)
        cur += 2
        if idx < count - 1:
            cur += gaps[idx]
    return rows


def _candidate_slots(height: int, width: int) -> list[tuple[int, int]]:
    rows = list(range(1, height - 3, 5))
    cols = list(range(4, width - 3, 5))
    return [(r, c) for r in rows for c in cols]


def _build_input(
    height: int,
    width: int,
    bg: int,
    objects: dict[int, dict],
    mapping: dict[int, int],
    legend_top_rows: list[int],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    for top, (src, dst) in zip(legend_top_rows, mapping.items()):
        grid[top : top + 2, 0] = src
        grid[top : top + 2, 1] = dst
    for color, obj in objects.items():
        r0, c0 = obj["pos"]
        _stamp(grid, occupied, obj["patch"], r0, c0, bg)
    return grid


def _expected_output(shape: tuple[int, int], bg: int, objects: dict[int, dict], mapping: dict[int, int]) -> np.ndarray:
    converted = {
        color: {
            "bbox": (obj["pos"][0], obj["pos"][1], obj["pos"][0] + obj["patch"].shape[0] - 1, obj["pos"][1] + obj["patch"].shape[1] - 1),
            "patch": obj["patch"],
        }
        for color, obj in objects.items()
    }
    return _compose_output_from_objects(shape, bg, converted, mapping)


def _naive_recolor_output(inp: np.ndarray, bg: int, objects: dict[int, dict], mapping: dict[int, int]) -> np.ndarray:
    out = np.full_like(inp, bg)
    occupied = np.zeros_like(inp, dtype=bool)
    for color, obj in objects.items():
        patch = obj["patch"].copy()
        if color in mapping:
            patch[patch == color] = mapping[color]
        r0, c0 = obj["pos"]
        _stamp(out, occupied, patch, r0, c0, bg)
    return out


def _chasing_output(shape: tuple[int, int], bg: int, objects: dict[int, dict], mapping: dict[int, int]) -> np.ndarray:
    correct_pos = {}
    for color in objects:
        if color in mapping:
            correct_pos[color] = objects[mapping[color]]["pos"]
        else:
            correct_pos[color] = objects[color]["pos"]

    out = np.full(shape, bg, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for color, obj in objects.items():
        if color in mapping:
            r0, c0 = correct_pos[mapping[color]]
        else:
            r0, c0 = obj["pos"]
        _stamp(out, occupied, obj["patch"], r0, c0, bg)
    return out


def _disjoint_when_aligned(patch_a: np.ndarray, patch_b: np.ndarray, bg: int) -> bool:
    h = max(patch_a.shape[0], patch_b.shape[0])
    w = max(patch_a.shape[1], patch_b.shape[1])
    a = np.full((h, w), bg, dtype=int)
    b = np.full((h, w), bg, dtype=int)
    a[: patch_a.shape[0], : patch_a.shape[1]] = patch_a
    b[: patch_b.shape[0], : patch_b.shape[1]] = patch_b
    return not np.any((a != bg) & (b != bg))


def _validate_case(inp: np.ndarray, expected: np.ndarray, objects: dict[int, dict], mapping: dict[int, int], bg: int) -> None:
    legends, legend_cells = _parse_legends(inp, bg)
    assert len(legends) == len(mapping)
    parsed_map = {src: dst for _r, src, dst in legends}
    assert parsed_map == mapping

    required = set(mapping) | set(mapping.values())
    parsed_objects = _collect_objects(inp, bg, legend_cells, required)
    assert set(parsed_objects) == set(objects)
    shapes = [_normalize_coords(np.argwhere(obj["patch"] != bg)) for obj in objects.values()]
    assert len(set(shapes)) == len(shapes), "object shapes must all differ"

    got = main(inp)
    assert np.array_equal(got, expected), "main() does not match the intended transformation"

    recolor = _naive_recolor_output(inp, bg, objects, mapping)
    assert not np.array_equal(recolor, expected), "recolor shortcut accidentally matches the true output"

    try:
        chasing = _chasing_output(inp.shape, bg, objects, mapping)
    except AssertionError:
        chasing = None
    if chasing is not None:
        assert not np.array_equal(chasing, expected), "new-position chasing shortcut accidentally matches the true output"


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    mode = _GEN_COUNTER % 3
    _GEN_COUNTER += 1

    for _ in range(600):
        rng = np.random.default_rng()
        height = int(rng.integers(18, 25))
        width = int(rng.integers(20, 30))
        bg = int(rng.integers(10))

        slots = _candidate_slots(height, width)
        object_count = int(rng.integers(4, min(7, len(slots) + 1)))
        if len(slots) < object_count:
            continue

        palette = [c for c in range(10) if c != bg]
        if len(palette) < object_count:
            continue
        colors = list(rng.choice(palette, size=object_count, replace=False))
        mapping = _choose_mapping(colors, mode, rng)
        legend_rows = _legend_rows(rng, height, len(mapping))

        forced_shapes: dict[int, tuple[tuple[int, int], ...]] = {}
        terminal_edges = [(src, dst) for src, dst in mapping.items() if dst not in mapping]
        if terminal_edges:
            src, dst = terminal_edges[0]
            src_shape, dst_shape = _DISJOINT_PAIRS[int(rng.integers(len(_DISJOINT_PAIRS)))]
            forced_shapes[src] = src_shape
            forced_shapes[dst] = dst_shape

        remaining_colors = [color for color in colors if color not in forced_shapes]
        extra_shapes = _choose_shapes(rng, len(remaining_colors), forbidden=set(forced_shapes.values()))
        shape_by_color = dict(forced_shapes)
        for color, shape in zip(remaining_colors, extra_shapes):
            shape_by_color[color] = shape

        slot_order = list(rng.permutation(len(slots)))[:object_count]

        objects = {}
        ok = True
        for color, slot_idx in zip(colors, slot_order):
            shape = shape_by_color[color]
            r0, c0 = slots[slot_idx]
            patch = _shape_to_patch(shape, color, bg)
            if r0 + patch.shape[0] > height or c0 + patch.shape[1] > width:
                ok = False
                break
            objects[color] = {"shape": shape, "patch": patch, "pos": (r0, c0)}
        if not ok:
            continue

        terminal_overlap_ok = True
        for src, dst in mapping.items():
            if dst not in mapping and not _disjoint_when_aligned(objects[src]["patch"], objects[dst]["patch"], bg):
                terminal_overlap_ok = False
                break
        if not terminal_overlap_ok:
            continue

        try:
            inp = _build_input(height, width, bg, objects, mapping, legend_rows)
            expected = _expected_output(inp.shape, bg, objects, mapping)
            _validate_case(inp, expected, objects, mapping, bg)
        except AssertionError:
            continue

        output_sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
        if output_sig in _SEEN_OUTPUT_SIGS:
            continue

        _SEEN_OUTPUT_SIGS.add(output_sig)
        return inp

    raise RuntimeError("failed to generate a valid non-degenerate input")
