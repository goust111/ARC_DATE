from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
WALL = 5
BLOB = 7
SOURCE = 8
LEGEND_PALETTE = [1, 2, 3, 4, 6, 9]

GENERATE_RETRIES = 400
STRIPE_GAP = 4


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    labels = -np.ones((h, w), dtype=int)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            comp_id = len(comps)
            while stack:
                cr, cc = stack.pop()
                labels[cr, cc] = comp_id
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps, labels


def _legend_colors(row: np.ndarray, bg: int) -> list[int]:
    return [int(v) for v in row if int(v) != int(bg)]


def _first_touch_times(
    scene: np.ndarray,
    bg: int,
    source: tuple[int, int],
    blob_labels: np.ndarray,
    n_blobs: int,
) -> dict[int, int]:
    h, w = scene.shape
    dist = -np.ones((h, w), dtype=int)
    q = deque([source])
    dist[source] = 0
    touch: dict[int, int] = {}

    while q and len(touch) < n_blobs:
        r, c = q.popleft()
        cur_dist = int(dist[r, c])
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            cell = int(scene[nr, nc])
            if cell == BLOB:
                blob_id = int(blob_labels[nr, nc])
                if blob_id not in touch:
                    touch[blob_id] = cur_dist + 1
                continue
            if cell != bg or int(dist[nr, nc]) != -1:
                continue
            dist[nr, nc] = cur_dist + 1
            q.append((nr, nc))
    return touch


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] < 3:
        raise ValueError("expected a 2D grid with scene, separator row, and legend row")

    bg = int(np.bincount(g[-2]).argmax())
    scene = g[:-2]
    legend = _legend_colors(g[-1], bg)
    source_pts = np.argwhere(scene == SOURCE)
    if source_pts.shape[0] != 1:
        raise ValueError("expected exactly one source pixel")
    source = tuple(map(int, source_pts[0]))

    blob_comps, blob_labels = _components(scene == BLOB)
    if not blob_comps:
        raise ValueError("expected at least one blob")
    if len(blob_comps) != len(legend):
        raise ValueError("legend length must equal blob count")

    touch = _first_touch_times(scene, bg, source, blob_labels, len(blob_comps))
    if len(touch) != len(blob_comps):
        raise ValueError("every blob must be reachable from the source through background")
    touch_values = list(touch.values())
    if len(set(touch_values)) != len(touch_values):
        raise ValueError("blob touch distances must be unique")

    order = sorted(range(len(blob_comps)), key=lambda idx: int(touch[idx]))
    return g, bg, scene, legend, blob_comps, order, source, touch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each orange blob by the legend color whose position matches its first wave-contact order from the cyan source.

    Input structure:
    - The main scene contains grey walls, one cyan source pixel, and several orange 4-connected blobs on a background field.
    - The last row is a legend row with one colored pixel per blob, read left to right.
    - The row just above the legend is pure background and separates the legend from the scene.

    Rule:
    - Start a 4-connected wave at the source and let it travel only through background pixels.
    - Grey walls and orange blobs are barriers; a blob is considered reached when the wave first touches one of its edge cells.
    - Sort blobs by that first-touch time and recolor the whole i-th blob with the i-th legend color, while preserving walls, source, background, separator, and legend.

    Key insight:
    - The relevant order is geodesic distance through the open background around walls, not straight-line or raw coordinate distance.
    """
    g, _bg, _scene, legend, blob_comps, order, _source, _touch = _analyze(grid)
    out = g.copy()
    for i, blob_id in enumerate(order):
        color = int(legend[i])
        coords = blob_comps[blob_id]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _snake_path(scene_h: int, scene_w: int, start_left: bool) -> tuple[list[int], list[tuple[int, int]], dict[tuple[int, int], int]]:
    stripe_rows = [1 + STRIPE_GAP * i for i in range((scene_h + 1) // STRIPE_GAP)]
    path: list[tuple[int, int]] = []
    for i, row in enumerate(stripe_rows):
        go_right = start_left if i % 2 == 0 else not start_left
        cols = range(scene_w) if go_right else range(scene_w - 1, -1, -1)
        for col in cols:
            path.append((row, col))
        if i + 1 < len(stripe_rows):
            connector_col = scene_w - 1 if go_right else 0
            for rr in range(row + 1, stripe_rows[i + 1]):
                path.append((rr, connector_col))
    return stripe_rows, path, {coord: idx for idx, coord in enumerate(path)}


def _shape_cells(anchor: tuple[int, int], dr: int, style: int) -> list[tuple[int, int]]:
    r, c = anchor
    cells = [(r + dr, c)]
    if style == 0:
        return cells
    cells.append((r + 2 * dr, c))
    if style == 1:
        return cells
    if style == 2:
        cells.append((r + 2 * dr, c - 1))
        return cells
    if style == 3:
        cells.append((r + 2 * dr, c + 1))
        return cells
    cells.extend([(r + 2 * dr, c - 1), (r + 2 * dr, c + 1)])
    return cells


def _slot_valid(
    anchor: tuple[int, int],
    dr: int,
    scene_h: int,
    scene_w: int,
    path_set: set[tuple[int, int]],
) -> bool:
    r, c = anchor
    if not (2 <= c <= scene_w - 3):
        return False
    if not (0 <= r + 2 * dr < scene_h):
        return False
    for cell in _shape_cells(anchor, dr, 4):
        rr, cc = cell
        if not (0 <= rr < scene_h and 0 <= cc < scene_w):
            return False
        if cell in path_set:
            return False
    return True


def _conflicts(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _transform_scene(
    scene: np.ndarray,
    specs: list[dict[str, object]],
    flip_vertical: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if not flip_vertical:
        return scene, specs

    h = scene.shape[0]
    scene2 = np.flipud(scene)
    specs2: list[dict[str, object]] = []
    for spec in specs:
        coords = [(h - 1 - int(r), int(c)) for r, c in spec["cells"]]
        specs2.append({"path_index": int(spec["path_index"]), "cells": coords})
    return scene2, specs2


def _manual_output(inp: np.ndarray, specs: list[dict[str, object]], legend_colors: list[int]) -> np.ndarray:
    out = inp.copy()
    ordered = sorted(specs, key=lambda spec: int(spec["path_index"]))
    for i, spec in enumerate(ordered):
        color = int(legend_colors[i])
        for r, c in spec["cells"]:
            out[int(r), int(c)] = color
    return out


def _nearest_manhattan(coords: np.ndarray, source: tuple[int, int]) -> int:
    sr, sc = source
    return min(abs(int(r) - sr) + abs(int(c) - sc) for r, c in coords)


def _semantic_checks(inp: np.ndarray, specs: list[dict[str, object]], expected: np.ndarray) -> bool:
    try:
        _g, bg, scene, legend, blob_comps, order, source, touch = _analyze(inp)
    except Exception:
        return False

    if len(legend) != len(specs):
        return False
    if len(blob_comps) != len(specs):
        return False
    if len(set(touch.values())) != len(specs):
        return False

    comp_to_spec: dict[frozenset[tuple[int, int]], int] = {}
    spec_sets: list[frozenset[tuple[int, int]]] = []
    for i, spec in enumerate(specs):
        s = frozenset((int(r), int(c)) for r, c in spec["cells"])
        spec_sets.append(s)
        comp_to_spec[s] = i

    actual_sets = [frozenset((int(r), int(c)) for r, c in comp) for comp in blob_comps]
    if set(actual_sets) != set(spec_sets):
        return False

    expected_order = [idx for idx, _ in sorted(enumerate(specs), key=lambda item: int(item[1]["path_index"]))]
    actual_order = [comp_to_spec[actual_sets[blob_id]] for blob_id in order]
    if actual_order != expected_order:
        return False

    manhattan = {}
    for blob_id, comp in enumerate(blob_comps):
        manhattan[blob_id] = _nearest_manhattan(comp, source)
    has_inversion = False
    for earlier_pos, earlier_blob in enumerate(order):
        for later_blob in order[earlier_pos + 1 :]:
            if int(manhattan[later_blob]) < int(manhattan[earlier_blob]):
                has_inversion = True
                break
        if has_inversion:
            break
    if not has_inversion:
        return False

    if np.array_equal(inp, expected):
        return False
    if np.array_equal(main(inp), expected):
        return True
    return False


def _pick_special_anchor(
    stripe_row: int,
    candidate_cols: list[int],
    dr_choices: list[int],
    path_index: dict[tuple[int, int], int],
    scene_h: int,
    scene_w: int,
    path_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> dict[str, object] | None:
    options: list[dict[str, object]] = []
    for col in candidate_cols:
        anchor = (stripe_row, col)
        for dr in dr_choices:
            if not _slot_valid(anchor, dr, scene_h, scene_w, path_set):
                continue
            for style in [4, 3, 2, 1, 0]:
                cells = _shape_cells(anchor, dr, style)
                if not _conflicts(cells, occupied):
                    options.append(
                        {
                            "path_index": int(path_index[anchor]),
                            "cells": cells,
                        }
                    )
                    break
    if not options:
        return None
    return random.choice(options)


def _random_spec_pool(
    stripe_rows: list[int],
    scene_h: int,
    scene_w: int,
    path_set: set[tuple[int, int]],
    path_index: dict[tuple[int, int], int],
) -> list[dict[str, object]]:
    pool: list[dict[str, object]] = []
    for row in stripe_rows:
        for col in range(2, scene_w - 2):
            anchor = (row, col)
            for dr in (-1, 1):
                if not _slot_valid(anchor, dr, scene_h, scene_w, path_set):
                    continue
                styles = [0, 1, 2, 3, 4]
                random.shuffle(styles)
                for style in styles:
                    pool.append({"path_index": int(path_index[anchor]), "cells": _shape_cells(anchor, dr, style)})
    random.shuffle(pool)
    return pool


def _attempt_generate() -> np.ndarray | None:
    n_blobs = random.randint(3, 5)
    stripe_count = random.randint(max(3, n_blobs), 5)
    scene_h = 4 * stripe_count - 1
    scene_w = random.randint(max(11, n_blobs + 6), 22)
    start_left = random.choice([True, False])
    flip_vertical = random.choice([True, False])

    stripe_rows, path, path_index = _snake_path(scene_h, scene_w, start_left=start_left)
    path_set = set(path)
    scene = np.full((scene_h, scene_w), WALL, dtype=int)
    for r, c in path[1:]:
        scene[r, c] = BACKGROUND
    src_r, src_c = path[0]
    scene[src_r, src_c] = SOURCE

    occupied: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []

    source_side_cols = [2, 3] if start_left else [scene_w - 4, scene_w - 3]
    far_side_cols = [scene_w - 4, scene_w - 3] if start_left else [2, 3]

    special_early = _pick_special_anchor(
        stripe_rows[0],
        far_side_cols,
        [1],
        path_index,
        scene_h,
        scene_w,
        path_set,
        occupied,
    )
    if special_early is None:
        return None
    specs.append(special_early)
    occupied.update((int(r), int(c)) for r, c in special_early["cells"])

    special_late = _pick_special_anchor(
        stripe_rows[1],
        source_side_cols,
        [-1, 1],
        path_index,
        scene_h,
        scene_w,
        path_set,
        occupied,
    )
    if special_late is None:
        return None
    specs.append(special_late)
    occupied.update((int(r), int(c)) for r, c in special_late["cells"])

    pool = _random_spec_pool(stripe_rows, scene_h, scene_w, path_set, path_index)
    for candidate in pool:
        if len(specs) >= n_blobs:
            break
        cells = [(int(r), int(c)) for r, c in candidate["cells"]]
        if _conflicts(cells, occupied):
            continue
        if any(int(candidate["path_index"]) == int(existing["path_index"]) for existing in specs):
            continue
        specs.append({"path_index": int(candidate["path_index"]), "cells": cells})
        occupied.update(cells)
    if len(specs) != n_blobs:
        return None

    for spec in specs:
        for r, c in spec["cells"]:
            scene[int(r), int(c)] = BLOB

    scene, specs = _transform_scene(scene, specs, flip_vertical=flip_vertical)

    legend_colors = random.sample(LEGEND_PALETTE, n_blobs)
    total_h = scene_h + 2
    inp = np.full((total_h, scene_w), BACKGROUND, dtype=int)
    inp[:scene_h] = scene
    legend_start = random.randint(0, scene_w - n_blobs)
    inp[-1, legend_start : legend_start + n_blobs] = np.array(legend_colors, dtype=int)

    expected = _manual_output(inp, specs, legend_colors)
    if not _semantic_checks(inp, specs, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wavefront-ordering puzzle")
