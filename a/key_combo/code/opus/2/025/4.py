import hashlib
import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


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
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _coords_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_signature(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in cells))


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _shape_bbox(cells)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_irregular_shape(cells: set[tuple[int, int]]) -> bool:
    h, w = _shape_dims(cells)
    return h > 1 and w > 1 and len(cells) < h * w


def _shape_frontier(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in cells:
                continue
            new_cells = set(cells)
            new_cells.add((nr, nc))
            h, w = _shape_dims(new_cells)
            if h <= 4 and w <= 4:
                frontier.add((nr, nc))
    return list(frontier)


def _random_core() -> set[tuple[int, int]]:
    if random.random() < 0.45:
        return {(0, 0), random.choice(DIRS4)}
    first = random.choice(DIRS4)
    opposite = (-first[0], -first[1])
    second_choices = [d for d in DIRS4 if d not in (first, opposite)]
    second = random.choice(second_choices)
    return {(0, 0), first, second}


def _grow_shape(
    base: set[tuple[int, int]],
    target_size: int,
    banned: set[tuple[tuple[int, int], ...]],
) -> set[tuple[int, int]] | None:
    for _ in range(120):
        cells = set(base)
        while len(cells) < target_size:
            frontier = _shape_frontier(cells)
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_size:
            continue
        sig = _shape_signature(cells)
        if sig in banned:
            continue
        if not _is_irregular_shape(cells):
            continue
        return cells
    return None


def _random_shape(banned: set[tuple[tuple[int, int], ...]]) -> set[tuple[int, int]] | None:
    for _ in range(200):
        size = random.randint(3, 7)
        shape = _grow_shape({(0, 0)}, size, banned)
        if shape is not None:
            return shape
    return None


def _generate_common_shapes() -> list[set[tuple[int, int]]]:
    for _ in range(200):
        core = _random_core()
        if len(core) == 3 and not _is_irregular_shape(core):
            continue
        banned: set[tuple[tuple[int, int], ...]] = set()
        shapes: list[set[tuple[int, int]]] = []
        for _ in range(3):
            min_size = max(3, len(core))
            size = random.randint(min_size, 7)
            shape = _grow_shape(core, size, banned)
            if shape is None:
                shapes = []
                break
            banned.add(_shape_signature(shape))
            shapes.append(shape)
        if len(shapes) != 3:
            continue
        counts = Counter()
        for shape in shapes:
            for cell in shape:
                counts[cell] += 1
        if any(cnt >= 2 for cell, cnt in counts.items() if cell != (0, 0)):
            return shapes
    raise RuntimeError("failed to generate common shapes")


def _relative_map_from_shape(shape: set[tuple[int, int]], fg_color: int) -> dict[tuple[int, int], int]:
    rel: dict[tuple[int, int], int] = {}
    for r, c in shape:
        rel[(int(r), int(c))] = GREY if (r, c) == (0, 0) else int(fg_color)
    return rel


def _relative_map_from_patch(patch: np.ndarray) -> dict[tuple[int, int], int]:
    anchor_coords = np.argwhere(patch == GREY)
    assert anchor_coords.shape == (1, 2)
    ar, ac = map(int, anchor_coords[0])
    rel: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(patch != BLACK):
        rel[(int(r) - ar, int(c) - ac)] = int(patch[int(r), int(c)])
    return rel


def _overlay_relative_maps(
    maps: list[dict[tuple[int, int], int]],
    overlap_color: int,
) -> np.ndarray:
    layers: dict[tuple[int, int], list[int]] = {}
    for rel_map in maps:
        for pos, color in rel_map.items():
            layers.setdefault(pos, []).append(int(color))
    assert layers
    rows = [r for r, _ in layers]
    cols = [c for _, c in layers]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BLACK, dtype=int)
    for (r, c), colors in layers.items():
        rr = r - min_r
        cc = c - min_c
        out[rr, cc] = int(overlap_color) if len(colors) >= 2 else int(colors[0])
    return out


def _separator_pair_is_valid(
    grid: np.ndarray,
    infos: list[dict],
    split_row: int,
    split_col: int,
) -> bool:
    h, w = grid.shape
    if not (0 <= split_row < h and 0 <= split_col < w):
        return False
    if not np.all(grid[split_row, :] == BLACK):
        return False
    if not np.all(grid[:, split_col] == BLACK):
        return False
    counts = [0, 0, 0, 0]
    for info in infos:
        r0, c0, r1, c1 = info["bbox"]
        if r1 < split_row:
            row_half = 0
        elif r0 > split_row:
            row_half = 1
        else:
            return False
        if c1 < split_col:
            col_half = 0
        elif c0 > split_col:
            col_half = 1
        else:
            return False
        counts[row_half * 2 + col_half] += 1
    return counts == [1, 1, 1, 1]


def _find_separator_pair(grid: np.ndarray, infos: list[dict]) -> tuple[int, int] | None:
    zero_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BLACK)]
    zero_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLACK)]
    candidates = [
        (r, c)
        for r in zero_rows
        for c in zero_cols
        if _separator_pair_is_valid(grid, infos, r, c)
    ]
    if not candidates:
        return None
    center_r2 = grid.shape[0] - 1
    center_c2 = grid.shape[1] - 1
    candidates.sort(key=lambda rc: (abs(2 * rc[0] - center_r2), abs(2 * rc[1] - center_c2)))
    return candidates[0]


def _extract_sprites(grid: np.ndarray) -> tuple[list[dict], tuple[int, int]]:
    comps = _components(grid != BLACK)
    infos: list[dict] = []
    for coords in comps:
        colors = sorted({int(grid[int(r), int(c)]) for r, c in coords if int(grid[int(r), int(c)]) != BLACK})
        assert GREY in colors and len(colors) == 2
        fg_color = next(color for color in colors if color != GREY)
        grey_cells = [(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == GREY]
        assert len(grey_cells) == 1
        box = _coords_bbox(coords)
        patch = _crop_bbox(grid, box)
        infos.append(
            {
                "coords": coords,
                "bbox": box,
                "fg": int(fg_color),
                "anchor": grey_cells[0],
                "patch": patch,
            }
        )
    assert len(infos) == 4
    split = _find_separator_pair(grid, infos)
    assert split is not None
    return infos, split


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the unique-color quadrant, align the other three anchor-marked sprites, and recolor overlap cells with the outlier color.

    Input structure:
    - The background is black, with one full black separator row and one full black separator column partitioning the grid into four quadrants.
    - Each quadrant contains exactly one 4-connected sprite made of one gray anchor pixel and one other foreground color.
    - Three quadrants share the same foreground color, while one quadrant has a unique foreground color.

    Rule:
    - Find the outlier quadrant by its unique non-gray foreground color and exclude that sprite from the overlay.
    - Crop the other three sprites to their tight bounding boxes, align them so their gray anchor pixels coincide, and overlay them in a shared coordinate system.
    - Any aligned cell covered by two or more sprites is recolored to the outlier foreground color; singly covered cells keep their original sprite color, then the result is tightly cropped.

    Key insight:
    - The outlier sprite contributes only its color, while the three majority-color sprites contribute only their shapes; solving requires separating those two roles before combining them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    infos, _split = _extract_sprites(g)
    color_counts = Counter(int(info["fg"]) for info in infos)
    assert sorted(color_counts.values()) == [1, 3]

    common_color = next(color for color, count in color_counts.items() if count == 3)
    outlier_color = next(color for color, count in color_counts.items() if count == 1)
    rel_maps = [_relative_map_from_patch(info["patch"]) for info in infos if int(info["fg"]) == int(common_color)]
    out = _overlay_relative_maps(rel_maps, outlier_color)
    box = _bbox(out != BLACK)
    assert box is not None
    return _crop_bbox(out, box)


def _offset2(bounds: tuple[int, int, int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bounds
    ar, ac = anchor
    return 2 * ar - (r0 + r1), 2 * ac - (c0 + c1)


def _feasible_anchor_positions(
    shape: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    margin: int = 1,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bounds
    sr0, sc0, sr1, sc1 = _shape_bbox(shape)
    low_r = r0 + margin - sr0
    high_r = r1 - margin - sr1
    low_c = c0 + margin - sc0
    high_c = c1 - margin - sc1
    if low_r > high_r or low_c > high_c:
        if margin > 0:
            return _feasible_anchor_positions(shape, bounds, margin=0)
        return []
    return [(int(r), int(c)) for r in range(low_r, high_r + 1) for c in range(low_c, high_c + 1)]


def _paint_shape(
    grid: np.ndarray,
    shape: set[tuple[int, int]],
    anchor: tuple[int, int],
    fg_color: int,
) -> None:
    ar, ac = anchor
    for dr, dc in shape:
        color = GREY if (dr, dc) == (0, 0) else int(fg_color)
        grid[ar + dr, ac + dc] = color


def _array_sig(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(tuple(map(int, arr.shape))).encode()).hexdigest()


def _choose_placements(
    quadrants: dict[str, tuple[int, int, int, int]],
    outlier_quad: str,
    common_assignment: dict[str, set[tuple[int, int]]],
    outlier_shape: set[tuple[int, int]],
) -> dict[str, tuple[int, int]] | None:
    for _ in range(120):
        placements: dict[str, tuple[int, int]] = {}
        used_offsets: set[tuple[int, int]] = set()
        pairs = list(common_assignment.items())
        random.shuffle(pairs)
        ok = True
        for quad_name, shape in pairs:
            options = _feasible_anchor_positions(shape, quadrants[quad_name])
            random.shuffle(options)
            options = [pos for pos in options if _offset2(quadrants[quad_name], pos) not in used_offsets]
            if not options:
                ok = False
                break
            chosen = options[0]
            placements[quad_name] = chosen
            used_offsets.add(_offset2(quadrants[quad_name], chosen))
        if not ok:
            continue
        outlier_options = _feasible_anchor_positions(outlier_shape, quadrants[outlier_quad])
        if not outlier_options:
            continue
        placements[outlier_quad] = random.choice(outlier_options)
        return placements
    return None


def _validate_generated_instance(
    grid: np.ndarray,
    split_row: int,
    split_col: int,
    quadrants: dict[str, tuple[int, int, int, int]],
    placements: dict[str, tuple[int, int]],
    outlier_quad: str,
    common_shapes: list[set[tuple[int, int]]],
    common_color: int,
    outlier_color: int,
) -> bool:
    infos, _ = _extract_sprites(grid)
    if not _separator_pair_is_valid(grid, infos, split_row, split_col):
        return False

    fg_counts = Counter(int(info["fg"]) for info in infos)
    if fg_counts.get(int(common_color), 0) != 3 or fg_counts.get(int(outlier_color), 0) != 1:
        return False

    shape_sigs = {_shape_signature(set(_relative_map_from_patch(info["patch"]).keys())) for info in infos}
    if len(shape_sigs) != 4:
        return False

    common_offsets = [
        _offset2(quadrants[qname], placements[qname])
        for qname in quadrants
        if qname != outlier_quad
    ]
    if len(set(common_offsets)) != 3:
        return False

    counts = Counter()
    for shape in common_shapes:
        for cell in shape:
            counts[cell] += 1
    if not any(cnt >= 2 for cell, cnt in counts.items() if cell != (0, 0)):
        return False

    expected = _overlay_relative_maps(
        [_relative_map_from_shape(shape, common_color) for shape in common_shapes],
        outlier_color,
    )
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    colors_out = set(map(int, np.unique(actual))) - {BLACK}
    if outlier_color not in colors_out or common_color not in colors_out:
        return False
    overlap_pixels = int(np.sum(actual == outlier_color))
    solid_pixels = int(np.sum(actual != BLACK))
    if not (1 <= overlap_pixels < solid_pixels):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    top = random.randint(6, 9)
    bottom = random.randint(6, 9)
    left = random.randint(6, 9)
    right = random.randint(6, 9)
    split_row = top
    split_col = left
    height = top + 1 + bottom
    width = left + 1 + right

    quadrants = {
        "tl": (0, 0, split_row - 1, split_col - 1),
        "tr": (0, split_col + 1, split_row - 1, width - 1),
        "bl": (split_row + 1, 0, height - 1, split_col - 1),
        "br": (split_row + 1, split_col + 1, height - 1, width - 1),
    }

    common_color, outlier_color = random.sample(PALETTE, 2)
    common_shapes = _generate_common_shapes()
    banned = {_shape_signature(shape) for shape in common_shapes}
    outlier_shape = _random_shape(banned)
    if outlier_shape is None:
        return None

    outlier_quad = random.choice(list(quadrants))
    random.shuffle(common_shapes)
    common_quads = [name for name in quadrants if name != outlier_quad]
    common_assignment = dict(zip(common_quads, common_shapes))
    placements = _choose_placements(quadrants, outlier_quad, common_assignment, outlier_shape)
    if placements is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for quad_name in quadrants:
        if quad_name == outlier_quad:
            _paint_shape(grid, outlier_shape, placements[quad_name], outlier_color)
        else:
            _paint_shape(grid, common_assignment[quad_name], placements[quad_name], common_color)

    if not _validate_generated_instance(
        grid,
        split_row,
        split_col,
        quadrants,
        placements,
        outlier_quad,
        common_shapes,
        common_color,
        outlier_color,
    ):
        return None

    out = main(grid)
    in_sig = _array_sig(grid)
    out_sig = _array_sig(out)
    if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_INPUT_SIGS.add(in_sig)
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
