import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(c, -r) for r, c in cells])


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(r, -c) for r, c in cells])


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _extract_objects(grid: np.ndarray) -> tuple[int, dict[int, int], dict[int, int], dict[int, tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 4

    bg = _background_color(g)

    top_marks: dict[int, int] = {}
    for c in range(1, w):
        color = int(g[0, c])
        if color == bg:
            continue
        assert color not in top_marks
        top_marks[color] = c

    left_marks: dict[int, int] = {}
    for r in range(1, h):
        color = int(g[r, 0])
        if color == bg:
            continue
        assert color not in left_marks
        left_marks[color] = r

    assert top_marks
    assert set(top_marks) == set(left_marks)

    canvas = g[1:, 1:]
    objects: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in top_marks:
        comps = _components_of_color(canvas, color)
        assert len(comps) == 1
        coords = tuple(sorted((r + 1, c + 1) for r, c in comps[0]))
        objects[color] = coords

    all_canvas_colors = set(map(int, np.unique(canvas))) - {bg}
    assert all_canvas_colors == set(top_marks)
    return bg, top_marks, left_marks, objects


def _scaled_relative_cells(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    rel = [(r - min_r, c - min_c) for r, c in coords]
    scale = len(coords)
    scaled: list[tuple[int, int]] = []
    for r, c in rel:
        base_r = r * scale
        base_c = c * scale
        for dr in range(scale):
            for dc in range(scale):
                scaled.append((base_r + dr, base_c + dc))
    height = (max(r for r, _ in rel) + 1) * scale
    width = (max(c for _, c in rel) + 1) * scale
    return height, width, tuple(sorted(scaled))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-row and left-column rulers to place enlarged copies of the canvas objects.

    Input structure:
    - The dominant background fills most cells.
    - Row 0 and column 0 are rulers with one marker per object color.
    - The interior canvas contains one 4-connected monochrome object per ruler color.

    Rule:
    - For each color C, read its target row from the left ruler and target column from the top ruler.
    - Let the object's area be k. Replace each source pixel by a solid k x k block, preserving the object's
      relative layout after multiplying offsets from its top-left corner by k.
    - Erase the original small objects, keep the rulers unchanged, and paste the enlarged object with top-left
      corner at the ruler intersection. Clip any overflow at the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg, top_marks, left_marks, objects = _extract_objects(g)
    h, w = g.shape

    out = np.full_like(g, bg)
    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]

    for color in top_marks:
        start_r = left_marks[color]
        start_c = top_marks[color]
        _height, _width, rel_cells = _scaled_relative_cells(objects[color])
        for dr, dc in rel_cells:
            rr = start_r + dr
            cc = start_c + dc
            if 0 <= rr < h and 0 <= cc < w and rr > 0 and cc > 0:
                assert out[rr, cc] in (bg, color)
                out[rr, cc] = color

    return out


_BASE_SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (((0, 0),),),
    2: (((0, 0), (0, 1)),),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ),
}

_RNG = np.random.default_rng()
_MAX_ATTEMPTS = 400


def _random_shape(area: int) -> tuple[tuple[int, int], ...]:
    shape = _BASE_SHAPES[area][int(_RNG.integers(len(_BASE_SHAPES[area])))]
    shape = _normalize(list(shape))
    if int(_RNG.integers(2)):
        shape = _flip_shape(shape)
    for _ in range(int(_RNG.integers(4))):
        shape = _rotate_shape(shape)
    return _normalize(list(shape))


def _scaled_object_data(shape: tuple[tuple[int, int], ...]) -> dict[str, object]:
    scale = len(shape)
    sh, sw = _shape_dims(shape)
    rel_cells: list[tuple[int, int]] = []
    for r, c in shape:
        for dr in range(scale):
            for dc in range(scale):
                rel_cells.append((r * scale + dr, c * scale + dc))
    return {
        "shape": shape,
        "scale": scale,
        "scaled_h": sh * scale,
        "scaled_w": sw * scale,
        "scaled_cells": tuple(sorted(rel_cells)),
    }


def _candidate_positions(
    height: int,
    width: int,
    obj: dict[str, object],
    mode: str,
) -> list[tuple[int, int]]:
    scaled_h = int(obj["scaled_h"])
    scaled_w = int(obj["scaled_w"])
    fit_r_max = height - scaled_h
    fit_c_max = width - scaled_w

    if mode == "fit":
        if fit_r_max < 1 or fit_c_max < 1:
            return []
        rows = range(1, fit_r_max + 1)
        cols = range(1, fit_c_max + 1)
    elif mode == "bottom":
        if fit_c_max < 1:
            return []
        rows = range(max(1, height - scaled_h + 1), height)
        cols = range(1, fit_c_max + 1)
    elif mode == "right":
        if fit_r_max < 1:
            return []
        rows = range(1, fit_r_max + 1)
        cols = range(max(1, width - scaled_w + 1), width)
    elif mode == "both":
        rows = range(max(1, height - scaled_h + 1), height)
        cols = range(max(1, width - scaled_w + 1), width)
    else:
        raise ValueError(mode)

    return [(int(r), int(c)) for r in rows for c in cols]


def _visible_cells(
    start_r: int,
    start_c: int,
    obj: dict[str, object],
    height: int,
    width: int,
) -> tuple[tuple[int, int], ...]:
    cells = []
    for dr, dc in obj["scaled_cells"]:
        rr = start_r + dr
        cc = start_c + dc
        if 0 <= rr < height and 0 <= cc < width:
            cells.append((int(rr), int(cc)))
    return tuple(cells)


def _pick_target_positions(
    height: int,
    width: int,
    objects: list[dict[str, object]],
    force_clip: bool,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    n = len(objects)

    if force_clip:
        clip_idx = int(_RNG.integers(n))
        clip_modes = [mode for mode in ("bottom", "right", "both") if _candidate_positions(height, width, objects[clip_idx], mode)]
        if not clip_modes:
            return None
        mode_by_index = ["fit"] * n
        mode_by_index[clip_idx] = str(_RNG.choice(clip_modes))
    else:
        mode_by_index = ["fit"] * n

    candidates = [_candidate_positions(height, width, obj, mode_by_index[i]) for i, obj in enumerate(objects)]
    if any(not opts for opts in candidates):
        return None

    order = sorted(range(n), key=lambda i: len(candidates[i]))
    for _ in range(250):
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        occupied: set[tuple[int, int]] = set()
        chosen: list[tuple[int, int] | None] = [None] * n
        success = True

        for idx in order:
            options = candidates[idx].copy()
            _RNG.shuffle(options)
            valid: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
            for r, c in options:
                if r in used_rows or c in used_cols:
                    continue
                vis = _visible_cells(r, c, objects[idx], height, width)
                if any(cell[0] == 0 or cell[1] == 0 for cell in vis):
                    continue
                if any(cell in occupied for cell in vis):
                    continue
                valid.append((r, c, vis))
                if len(valid) >= 24:
                    break
            if not valid:
                success = False
                break
            r, c, vis = valid[int(_RNG.integers(len(valid)))]
            chosen[idx] = (r, c)
            used_rows.add(r)
            used_cols.add(c)
            occupied.update(vis)

        if success:
            starts = [pos for pos in chosen if pos is not None]
            assert len(starts) == n
            return starts, occupied

    return None


def _moat(cells: tuple[tuple[int, int], ...], height: int, width: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < height and 0 <= nc < width:
                    blocked.add((int(nr), int(nc)))
    return blocked


def _place_small_objects(
    grid: np.ndarray,
    colors: list[int],
    shapes: list[tuple[tuple[int, int], ...]],
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    blocked = set(forbidden)
    order = list(range(len(colors)))
    _RNG.shuffle(order)

    for idx in order:
        shape = shapes[idx]
        sh, sw = _shape_dims(shape)
        candidates: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
        for r0 in range(2, h - sh + 1):
            for c0 in range(2, w - sw + 1):
                cells = tuple((r0 + dr, c0 + dc) for dr, dc in shape)
                if any(cell in blocked for cell in cells):
                    continue
                candidates.append(((r0, c0), cells))

        if not candidates:
            return False

        choice_order = _RNG.permutation(len(candidates))
        anchor, cells = candidates[int(choice_order[0])]
        _ = anchor
        for r, c in cells:
            grid[r, c] = colors[idx]
        blocked.update(_moat(cells, h, w))

    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(15, 24))
    width = int(_RNG.integers(15, 24))
    bg = int(_RNG.integers(10))

    palette = [color for color in range(10) if color != bg]
    _RNG.shuffle(palette)
    n_objects = int(_RNG.integers(2, 6))
    colors = [int(c) for c in palette[:n_objects]]

    force_clip = bool(int(_RNG.integers(2)))

    area_choices = np.array([1, 2, 3, 4], dtype=int)
    area_probs = np.array([0.15, 0.35, 0.35, 0.15], dtype=float)
    areas = [int(_RNG.choice(area_choices, p=area_probs)) for _ in range(n_objects)]
    if all(area == 1 for area in areas):
        areas[int(_RNG.integers(n_objects))] = int(_RNG.choice([2, 3, 4], p=[0.45, 0.4, 0.15]))

    if sum(area ** 3 for area in areas) > (height - 1) * (width - 1) * 0.48:
        return None

    shapes = [_random_shape(area) for area in areas]
    objects = [_scaled_object_data(shape) for shape in shapes]

    if any(int(obj["scaled_h"]) > height or int(obj["scaled_w"]) > width for obj in objects):
        return None

    target = _pick_target_positions(height, width, objects, force_clip=force_clip)
    if target is None:
        return None
    starts, output_cells = target

    clip_flags = [
        r + int(obj["scaled_h"]) > height or c + int(obj["scaled_w"]) > width
        for (r, c), obj in zip(starts, objects)
    ]
    if force_clip != any(clip_flags):
        return None

    grid = np.full((height, width), bg, dtype=int)
    for color, (_r, c) in zip(colors, starts):
        grid[0, c] = color
    for color, (r, _c) in zip(colors, starts):
        grid[r, 0] = color

    if not _place_small_objects(grid, colors, shapes, output_cells):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
