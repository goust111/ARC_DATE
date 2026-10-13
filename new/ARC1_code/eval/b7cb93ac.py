from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import itertools
import random


def _normalized(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    return pts - pts.min(axis=0)


def _orientations(coords: np.ndarray) -> list[np.ndarray]:
    pts = _normalized(coords)
    transforms = [
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    ]

    out = []
    seen = set()
    for transform in transforms:
        variant = np.array([transform(int(r), int(c)) for r, c in pts], dtype=int)
        variant = _normalized(variant)
        key = tuple(sorted(map(tuple, variant.tolist())))
        if key not in seen:
            seen.add(key)
            out.append(variant)
    return out


def _compact_shape(area: int) -> tuple[int, int]:
    factors = [(h, area // h) for h in range(1, int(area**0.5) + 1) if area % h == 0]
    assert factors, "foreground area must admit a rectangular packing"
    return max(factors, key=lambda hw: hw[0])


def _placements(coords: np.ndarray, shape: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    height, width = shape
    out = []
    for orient in _orientations(coords):
        orient_h = int(orient[:, 0].max()) + 1
        orient_w = int(orient[:, 1].max()) + 1
        if orient_h > height or orient_w > width:
            continue
        for row in range(height - orient_h + 1):
            for col in range(width - orient_w + 1):
                cells = tuple(sorted((int(r + row), int(c + col)) for r, c in orient))
                out.append(cells)
    assert out, "each component must fit in the compact rectangle"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack all foreground fragments into a solid compact rectangle while preserving their colors.

    Input structure
    - The background is the dominant color.
    - The foreground is split into monochromatic 4-connected polyomino fragments.
    - The total foreground area is exactly the area of the compact output rectangle.

    Rule
    - Sort fragments by decreasing area, with equal-area fragments ordered left-to-right in the input.
    - For each fragment, try its identity/reflected/rotated orientations in that order and scan placements from top-left to bottom-right.
    - Keep the first set of non-overlapping placements that fills the whole compact rectangle.

    Key insight
    The scattered objects are jigsaw fragments; the output is their canonical top-left packing into the smallest near-square rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(components) >= 2, "expected multiple foreground fragments"

    pieces = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        coords = np.asarray(comp["coords"], dtype=int) - np.array([r0, c0], dtype=int)
        pieces.append(
            {
                "color": int(comp["color"]),
                "area": int(comp["area"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": coords,
            }
        )

    total_area = sum(piece["area"] for piece in pieces)
    out_shape = _compact_shape(total_area)
    assert out_shape == (3, 4), "this task packs twelve cells into a 3x4 rectangle"

    order = sorted(
        range(len(pieces)),
        key=lambda idx: (
            -pieces[idx]["area"],
            pieces[idx]["bbox"][1],
            pieces[idx]["bbox"][0],
            pieces[idx]["bbox"][3],
            pieces[idx]["bbox"][2],
        ),
    )
    placement_options = [_placements(pieces[idx]["coords"], out_shape) for idx in order]

    occupied: set[tuple[int, int]] = set()
    chosen: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    def search(k: int) -> bool:
        if k == len(order):
            return len(occupied) == total_area
        piece_idx = order[k]
        for cells in placement_options[k]:
            cell_set = set(cells)
            if occupied & cell_set:
                continue
            occupied.update(cell_set)
            chosen.append((piece_idx, cells))
            if search(k + 1):
                return True
            chosen.pop()
            occupied.difference_update(cell_set)
        return False

    assert search(0), "expected the fragments to tile the compact rectangle"
    assert len(chosen) == len(pieces)

    out = np.full(out_shape, int(bg), dtype=int)
    for piece_idx, cells in chosen:
        for r, c in cells:
            out[r, c] = pieces[piece_idx]["color"]

    assert np.all(out != bg), "the compact rectangle must be completely filled"
    return out

# --------------- generator (o5x) ---------------

# Packed output geometry.
_GEN_OUT_SHAPE = (3, 4)
_GEN_TOTAL_AREA = _GEN_OUT_SHAPE[0] * _GEN_OUT_SHAPE[1]
_GEN_CELLS = tuple((r, c) for r in range(_GEN_OUT_SHAPE[0]) for c in range(_GEN_OUT_SHAPE[1]))
_GEN_NEIGHBOR_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Fragment count and area constraints.
_GEN_MIN_PIECES, _GEN_MAX_PIECES = 3, 5
_GEN_MIN_PIECE_AREA, _GEN_MAX_PIECE_AREA = 1, 7
_GEN_MIN_LARGEST_AREA = 5
_GEN_SMALL_AREA_MAX = 2
_GEN_MIN_SMALL_BY_PIECE_COUNT = {4: 2, 5: 4}

# Palette and input canvas constraints.
_GEN_COLOR_COUNT = 3
_GEN_MIN_COLOR, _GEN_MAX_COLOR = 1, 9
_GEN_REPEAT_PAIR_SIZE = 2
_GEN_BG = 0
_GEN_INPUT_HEIGHT = 10
_GEN_MIN_INPUT_WIDTH, _GEN_MAX_INPUT_WIDTH = 10, 13
_GEN_BORDER_PADDING, _GEN_FRAGMENT_PADDING = 1, 1
_GEN_PADDING_CONNECTIVITY = 4

# Rejection sampling bound.
_GEN_MAX_ATTEMPTS = 300


def _gen_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in _GEN_NEIGHBOR_STEPS:
        nb = (r + dr, c + dc)
        if nb in _GEN_CELLS:
            out.append(nb)
    return out


def _gen_is_connected(cells) -> bool:
    cells = set(cells)
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        cell = stack.pop()
        for nb in _gen_neighbors(cell):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _gen_components(cells) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    comps = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            cell = stack.pop()
            for nb in _gen_neighbors(cell):
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _gen_component_sizes_fit(cells, sizes: tuple[int, ...]) -> bool:
    caps = tuple(sorted((len(comp) for comp in _gen_components(cells)), reverse=True))
    sizes = tuple(sorted(sizes, reverse=True))

    def pack(i: int, capacities: tuple[int, ...]) -> bool:
        if i == len(sizes):
            return all(cap == 0 for cap in capacities)
        size = sizes[i]
        tried = set()
        for j, cap in enumerate(capacities):
            if cap < size or cap in tried:
                continue
            tried.add(cap)
            nxt = list(capacities)
            nxt[j] -= size
            if pack(i + 1, tuple(sorted(nxt, reverse=True))):
                return True
        return False

    return pack(0, caps)


def _gen_connected_subsets(cells, size: int) -> list[set[tuple[int, int]]]:
    opts = []
    for combo in itertools.combinations(sorted(cells), size):
        if _gen_is_connected(combo):
            opts.append(set(combo))
    random.shuffle(opts)
    return opts


def _gen_partition(sizes: tuple[int, ...]) -> list[set[tuple[int, int]]] | None:
    def rec(cells: set[tuple[int, int]], rest: tuple[int, ...]):
        if not rest:
            return [] if not cells else None
        if len(rest) == 1:
            return [set(cells)] if len(cells) == rest[0] and _gen_is_connected(cells) else None

        size = rest[0]
        for subset in _gen_connected_subsets(cells, size):
            remaining = cells - subset
            if not _gen_component_sizes_fit(remaining, rest[1:]):
                continue
            tail = rec(remaining, rest[1:])
            if tail is not None:
                return [subset] + tail
        return None

    return rec(set(_GEN_CELLS), sizes)


def _gen_sizes(piece_count: int) -> tuple[int, ...]:
    opts = []
    for combo in itertools.combinations_with_replacement(
        range(_GEN_MIN_PIECE_AREA, _GEN_MAX_PIECE_AREA + 1), piece_count
    ):
        if sum(combo) != _GEN_TOTAL_AREA or max(combo) < _GEN_MIN_LARGEST_AREA:
            continue
        small_count = sum(1 for size in combo if size <= _GEN_SMALL_AREA_MAX)
        if small_count < _GEN_MIN_SMALL_BY_PIECE_COUNT.get(piece_count, 0):
            continue
        opts.append(tuple(sorted(combo, reverse=True)))
    return random.choice(opts)


def _gen_normalized_coords(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    return coords - coords.min(axis=0)


def _gen_random_orientation(coords: np.ndarray) -> np.ndarray:
    return random.choice(_orientations(coords)).copy()


def _gen_patch(coords: np.ndarray, color: int) -> np.ndarray:
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    patch = np.full((h, w), _GEN_BG, dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


def _gen_colors(sizes: tuple[int, ...]) -> list[int]:
    piece_count = len(sizes)
    colors = random.sample(range(_GEN_MIN_COLOR, _GEN_MAX_COLOR + 1), _GEN_COLOR_COUNT)
    out = [None] * piece_count
    small = [i for i, size in enumerate(sizes) if size <= _GEN_SMALL_AREA_MAX]
    random.shuffle(small)

    if piece_count == _GEN_COLOR_COUNT:
        out = colors[:]
        random.shuffle(out)
        return out

    first_pair = small[:_GEN_REPEAT_PAIR_SIZE]
    for idx in first_pair:
        out[idx] = colors[0]
    used_small = _GEN_REPEAT_PAIR_SIZE

    if piece_count == _GEN_MAX_PIECES:
        second_pair = small[used_small : used_small + _GEN_REPEAT_PAIR_SIZE]
        for idx in second_pair:
            out[idx] = colors[1]
        used_colors = _GEN_REPEAT_PAIR_SIZE
    else:
        used_colors = 1

    remaining_indices = [i for i, color in enumerate(out) if color is None]
    remaining_colors = colors[used_colors:]
    random.shuffle(remaining_colors)
    for idx, color in zip(remaining_indices, remaining_colors):
        out[idx] = color
    return out


def _gen_fragment_order(areas: list[int], bboxes: list[tuple[int, int, int, int]]) -> list[int]:
    return sorted(
        range(len(areas)),
        key=lambda i: (-areas[i], bboxes[i][1], bboxes[i][0], bboxes[i][3], bboxes[i][2]),
    )


def _gen_pack_output(coords_list: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    placement_options = [_placements(coords, _GEN_OUT_SHAPE) for coords in coords_list]
    occupied: set[tuple[int, int]] = set()
    chosen: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    def search(k: int) -> bool:
        if k == len(coords_list):
            return len(occupied) == _GEN_TOTAL_AREA
        for cells in placement_options[k]:
            cell_set = set(cells)
            if occupied & cell_set:
                continue
            occupied.update(cell_set)
            chosen.append((k, cells))
            if search(k + 1):
                return True
            chosen.pop()
            occupied.difference_update(cell_set)
        return False

    if not search(0):
        return None

    out = np.full(_GEN_OUT_SHAPE, _GEN_BG, dtype=int)
    for idx, cells in chosen:
        for r, c in cells:
            out[r, c] = int(colors[idx])
    return out if np.all(out != _GEN_BG) else None


def _gen_place_fragment(inp: np.ndarray, patch: np.ndarray) -> tuple[int, int, int, int] | None:
    try:
        r, c = random_free_location(
            inp,
            patch,
            transparent=_GEN_BG,
            border=_GEN_BORDER_PADDING,
            padding=_GEN_FRAGMENT_PADDING,
            padding_connectivity=_GEN_PADDING_CONNECTIVITY,
        )
    except ValueError:
        return None
    paste(inp, patch, r, c, transparent=_GEN_BG)
    return (int(r), int(c), int(r + patch.shape[0] - 1), int(c + patch.shape[1] - 1))


def _gen_scatter(coords_list: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    width = random.randint(_GEN_MIN_INPUT_WIDTH, _GEN_MAX_INPUT_WIDTH)
    inp = np.full((_GEN_INPUT_HEIGHT, width), _GEN_BG, dtype=int)
    placements = [None] * len(coords_list)

    for idx, (coords, color) in enumerate(zip(coords_list, colors)):
        patch = _gen_patch(coords, color)
        placement = _gen_place_fragment(inp, patch)
        if placement is None:
            return None
        placements[idx] = placement

    areas = [int(coords.shape[0]) for coords in coords_list]
    actual_order = _gen_fragment_order(areas, placements)
    if actual_order != list(range(len(coords_list))):
        return None
    return inp


def _gen_design_fragments() -> tuple[tuple[int, ...], list[np.ndarray]] | None:
    piece_count = random.randint(_GEN_MIN_PIECES, _GEN_MAX_PIECES)
    sizes = _gen_sizes(piece_count)
    parts = _gen_partition(sizes)
    if parts is None:
        return None
    coords_list = [_gen_random_orientation(_gen_normalized_coords(part)) for part in parts]
    return sizes, coords_list


def _gen_oracle_accepts(inp: np.ndarray, output: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(inp), output))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    designed = _gen_design_fragments()
    if designed is None:
        return None
    sizes, coords_list = designed
    colors = _gen_colors(sizes)
    output = _gen_pack_output(coords_list, colors)
    if output is None:
        return None

    inp = _gen_scatter(coords_list, colors)
    if inp is None:
        return None

    return inp if _gen_oracle_accepts(inp, output) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid input")
