from __future__ import annotations

import random
from collections import Counter

import numpy as np


GEN_RETRIES = 400
CHAIN_RETRIES = 220
PATH_RETRIES = 120
SCATTER_RETRIES = 220
PALETTE = tuple(range(10))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            comps.append(np.array(pts, dtype=int))

    return comps


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.asarray(coords, dtype=int)}


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    mins = arr.min(axis=0)
    return arr - mins


def _path_from_runs(runs: list[tuple[int, int, int]]) -> np.ndarray:
    current = (0, 0)
    pts = [current]
    seen = {current}

    for dr, dc, length in runs:
        for _ in range(int(length)):
            current = (int(current[0] + dr), int(current[1] + dc))
            if current in seen:
                raise AssertionError("template path self-intersects")
            seen.add(current)
            pts.append(current)

    arr = _normalize(np.array(pts, dtype=int))
    _endpoints(arr)
    return arr


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _union_bbox(all_coords: list[np.ndarray]) -> tuple[int, int, int, int]:
    mins = np.vstack([coords.min(axis=0) for coords in all_coords]).min(axis=0)
    maxs = np.vstack([coords.max(axis=0) for coords in all_coords]).max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _shift(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.asarray(coords, dtype=int) + np.array([int(dr), int(dc)], dtype=int)


def _endpoints(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = _coord_set(coords)
    ends: list[tuple[int, int]] = []
    for r, c in pts:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if deg == 1:
            ends.append((int(r), int(c)))
        elif deg not in (1, 2):
            raise AssertionError("path component must be a single non-branching polyline")
    if len(ends) != 2:
        raise AssertionError("path component must have exactly two endpoints")
    return ends


def _top_endpoint(coords: np.ndarray) -> tuple[int, int]:
    return min(_endpoints(coords), key=lambda p: (int(p[0]), int(p[1])))


def _interface_endpoint(coords: np.ndarray) -> tuple[int, int]:
    return max(_endpoints(coords), key=lambda p: (int(p[1]), -int(p[0])))


def _interface_has_free_below(coords: np.ndarray) -> bool:
    anchor = _interface_endpoint(coords)
    return (int(anchor[0] + 1), int(anchor[1])) not in _coord_set(coords)


def _has_pixels_below_interface(coords: np.ndarray) -> bool:
    anchor = _interface_endpoint(coords)
    return any(int(r) > int(anchor[0]) for r, _c in np.asarray(coords, dtype=int))


def _flip_horizontal(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    _r0, c0, _r1, c1 = _bbox(arr)
    arr[:, 1] = int(c0 + c1) - arr[:, 1]
    return arr


def _align_under_anchor(coords: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    er, ec = _top_endpoint(arr)
    dr = int(anchor[0] + 1 - er)
    dc = int(anchor[1] - ec)
    return _shift(arr, dr, dc)


def _in_bounds(coords: np.ndarray, h: int, w: int) -> bool:
    arr = np.asarray(coords, dtype=int)
    return bool(
        np.all((0 <= arr[:, 0]) & (arr[:, 0] < h))
        and np.all((0 <= arr[:, 1]) & (arr[:, 1] < w))
    )


def _overlaps(coords: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    return any((int(r), int(c)) in occupied for r, c in np.asarray(coords, dtype=int))


def _contact_count(coords: np.ndarray, occupied: set[tuple[int, int]]) -> int:
    pairs = set()
    for r, c in np.asarray(coords, dtype=int):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if (int(nr), int(nc)) in occupied:
                pairs.add(((rr, cc), (int(nr), int(nc))))
    return len(pairs)


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    for r, c in np.asarray(coords, dtype=int):
        grid[int(r), int(c)] = int(color)


def _extract_path_component(grid: np.ndarray, color: int, strip_row: int) -> np.ndarray:
    comps = _components(grid == int(color))
    markers = 0
    paths: list[np.ndarray] = []

    for comp in comps:
        if int(comp.shape[0]) == 1 and int(comp[0, 0]) == int(strip_row):
            markers += 1
        else:
            paths.append(comp)

    assert markers == 1, "each order color must appear exactly once in the bottom strip"
    assert len(paths) == 1, "each order color must have exactly one path fragment"
    return np.asarray(paths[0], dtype=int)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 3

    bg = _bg_color(g)
    strip_row = h - 1
    order = [int(g[strip_row, c]) for c in range(w) if int(g[strip_row, c]) != int(bg)]
    assert len(order) >= 2
    assert len(order) == len(set(order))

    pieces = [(int(color), _extract_path_component(g, int(color), strip_row)) for color in order]

    out = np.full_like(g, int(bg))
    occupied: set[tuple[int, int]] = set()

    first_color, first_coords = pieces[0]
    _paint(out, first_coords, first_color)
    occupied |= _coord_set(first_coords)
    previous = np.asarray(first_coords, dtype=int)

    for color, orig_coords in pieces[1:]:
        anchor = _interface_endpoint(previous)
        placed = _align_under_anchor(orig_coords, anchor)
        needs_flip = (not _in_bounds(placed, h, w)) or _overlaps(placed, occupied)
        if needs_flip:
            placed = _align_under_anchor(_flip_horizontal(orig_coords), anchor)
        assert _in_bounds(placed, h, w)
        assert not _overlaps(placed, occupied)
        _paint(out, placed, color)
        occupied |= _coord_set(placed)
        previous = placed

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Join scattered path fragments into one ordered chain by using the bottom color strip.

    Input structure:
    - The background is the dominant color.
    - The bottom row contains a left-to-right order strip: isolated single pixels of distinct colors.
    - For each strip color, the rest of the grid contains exactly one same-colored width-1 path fragment made of horizontal and vertical steps.
    - Each fragment is a single 4-connected non-branching polyline with exactly two endpoints.

    Rule:
    - Read the strip colors from left to right to get the fragment order.
    - Keep the first fragment fixed in place.
    - For every later fragment, find the previously placed fragment's endpoint with the larger column and use it as the interface endpoint.
    - Translate the current fragment so that its endpoint with the smaller row sits exactly one cell below that interface endpoint.
    - If that first placement would go out of bounds or overlap any already placed pixel, horizontally flip the current fragment and align it again instead.
    - Erase the entire bottom strip to background in the output.

    Key insight:
    The strip gives only the fragment order; the actual assembly requires inferring two endpoint-selection rules and then testing whether horizontal mirroring is needed to avoid overlap.
    """
    return _solve(np.asarray(grid, dtype=int))


def _random_path(
    rng: random.Random,
    first_horizontal_dir: int | None = None,
    require_upturn: bool = False,
) -> np.ndarray:
    for _ in range(PATH_RETRIES):
        run_count = rng.randint(3, 6)
        current = (0, 0)
        pts = [current]
        pt_set = {current}
        used_bias = False
        used_upturn = False
        valid = True

        for run_idx in range(run_count):
            if run_idx % 2 == 0:
                if run_idx == 0:
                    sign = 1
                else:
                    can_go_up = int(current[0]) >= 2
                    if require_upturn and (not used_upturn) and can_go_up:
                        sign = -1
                    else:
                        sign = rng.choices([1, -1], weights=[3, 2])[0] if can_go_up else 1

                if sign == -1:
                    max_len = min(2, int(current[0]) - 1)
                    if max_len < 1:
                        valid = False
                        break
                    length = rng.randint(1, max_len)
                    used_upturn = True
                else:
                    length = rng.randint(1, 3 if run_idx else 4)
                step = (sign, 0)
            else:
                if first_horizontal_dir in (-1, 1) and not used_bias and rng.random() < 0.8:
                    sign = int(first_horizontal_dir)
                else:
                    sign = rng.choice([-1, 1])
                used_bias = True
                length = rng.randint(1, 4)
                step = (0, sign)

            for _step_idx in range(length):
                current = (int(current[0] + step[0]), int(current[1] + step[1]))
                if current in pt_set:
                    valid = False
                    break
                pt_set.add(current)
                pts.append(current)
            if not valid:
                break

        if not valid:
            continue

        arr = _normalize(np.array(pts, dtype=int))
        if int(arr.shape[0]) < 4 or int(arr.shape[0]) > 10:
            continue
        if int(arr[:, 0].max() - arr[:, 0].min()) > 5:
            continue
        if int(arr[:, 1].max() - arr[:, 1].min()) > 6:
            continue
        if require_upturn and not used_upturn:
            continue

        try:
            ends = _endpoints(arr)
        except AssertionError:
            continue

        if len({int(r) for r, _c in ends}) != 2:
            continue
        if len({int(c) for _r, c in ends}) != 2:
            continue
        return arr

    raise RuntimeError("failed to sample a valid path shape")


def _balanced_flip_flags(rng: random.Random, transitions: int) -> list[bool]:
    if transitions <= 0:
        return []
    if transitions == 1:
        return [bool(rng.getrandbits(1))]

    tail = transitions - 1
    if tail % 2 == 0:
        flip_count = tail // 2
    else:
        flip_count = rng.choice([tail // 2, tail // 2 + 1])
    flip_count = max(1, int(flip_count))

    rest = [True] * flip_count + [False] * (tail - flip_count)
    rng.shuffle(rest)
    return [False] + rest


def _choose_colors(rng: random.Random, bg: int, n: int) -> list[int]:
    pool = [c for c in PALETTE if int(c) != int(bg)]
    rng.shuffle(pool)
    return [int(c) for c in pool[:n]]


def _make_simple_piece(rng: random.Random) -> tuple[np.ndarray, dict[str, int | str]]:
    down1 = rng.randint(2, 4)
    right = rng.randint(1, 3)
    down2 = rng.randint(1, 2)
    coords = _path_from_runs([(1, 0, down1), (0, 1, right), (1, 0, down2)])
    return coords, {"kind": "simple"}


def _make_scaffold_piece(rng: random.Random) -> tuple[np.ndarray, dict[str, int | str]]:
    down = rng.randint(4, 5)
    right1 = rng.randint(2, 3)
    right2 = rng.randint(1, 3)
    up = 2
    coords = _path_from_runs([(1, 0, down), (0, 1, right1), (-1, 0, up), (0, 1, right2)])
    return coords, {"kind": "scaffold", "tail": right2}


def _make_flip_raw_piece(
    rng: random.Random, overlap_left: int
) -> tuple[np.ndarray, dict[str, int | str]]:
    extra_left = rng.randint(1, 2)
    coords = _path_from_runs([(1, 0, 1), (0, -1, overlap_left), (1, 0, 1), (0, -1, extra_left)])
    return coords, {"kind": "flip_raw"}


def _grow8(points: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    grown: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    grown.add((nr, nc))
    return grown


def _build_chain(
    rng: random.Random, piece_count: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    orig_shapes: list[np.ndarray] = []
    finals: list[np.ndarray] = []
    trials: list[np.ndarray] = []

    layout: list[tuple[str, dict[str, int | str]]] = []
    first_builder = rng.choice([_make_simple_piece, _make_scaffold_piece])
    first_shape, first_meta = first_builder(rng)
    layout.append(("keep", first_meta))
    orig_shapes.append(first_shape)
    finals.append(first_shape.copy())
    trials.append(first_shape.copy())

    occupied = _coord_set(first_shape)
    previous = first_shape.copy()

    for idx in range(1, piece_count):
        is_last = idx == piece_count - 1
        if is_last:
            break

        if idx == piece_count - 2:
            shape, meta = _make_scaffold_piece(rng)
        else:
            builder = rng.choice([_make_simple_piece, _make_scaffold_piece])
            shape, meta = builder(rng)

        placed = _align_under_anchor(shape, _interface_endpoint(previous))
        if _overlaps(placed, occupied):
            return None
        if _contact_count(placed, occupied) != 1:
            return None
        if not _interface_has_free_below(placed):
            return None

        layout.append(("keep", meta))
        orig_shapes.append(shape)
        finals.append(placed)
        trials.append(placed.copy())
        occupied |= _coord_set(placed)
        previous = placed

    scaffold_meta = layout[-1][1]
    if str(scaffold_meta.get("kind")) != "scaffold":
        return None

    overlap_left = int(scaffold_meta["tail"])
    raw_flip, _meta = _make_flip_raw_piece(rng, overlap_left)
    anchor = _interface_endpoint(previous)
    raw_trial = _align_under_anchor(raw_flip, anchor)
    flipped_final = _align_under_anchor(_flip_horizontal(raw_flip), anchor)

    if not _overlaps(raw_trial, occupied):
        return None
    if _overlaps(flipped_final, occupied):
        return None
    if _contact_count(flipped_final, occupied) != 1:
        return None

    orig_shapes.append(raw_flip)
    finals.append(flipped_final)
    trials.append(raw_trial)

    bbox = _union_bbox(finals + trials)
    if int(bbox[2] - bbox[0] + 1) > 14 or int(bbox[3] - bbox[1] + 1) > 18:
        return None

    min_r, min_c, _max_r, _max_c = bbox
    if min_r != 0 or min_c != 0:
        finals = [_shift(coords, -min_r, -min_c) for coords in finals]
        trials = [_shift(coords, -min_r, -min_c) for coords in trials]

    return orig_shapes, finals, trials


def _scatter_piece(
    rng: random.Random,
    shape: np.ndarray,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    avoid_exact: set[tuple[int, int]] | None = None,
) -> np.ndarray | None:
    arr = np.asarray(shape, dtype=int)
    shape_h = int(arr[:, 0].max()) + 1
    shape_w = int(arr[:, 1].max()) + 1
    max_dr = int(h - 2 - shape_h)
    max_dc = int(w - shape_w)
    if max_dr < 0 or max_dc < 0:
        return None

    for _ in range(SCATTER_RETRIES):
        dr = rng.randint(0, max_dr)
        dc = rng.randint(0, max_dc)
        placed = _shift(arr, dr, dc)
        pts = _coord_set(placed)
        if pts & blocked:
            continue
        if avoid_exact is not None and pts == avoid_exact:
            continue
        return placed

    return None


def _sample_strip_columns(rng: random.Random, width: int, count: int) -> list[int]:
    assert count <= width
    cols = rng.sample(list(range(width)), count)
    cols.sort()
    return [int(c) for c in cols]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = 3 if rng.random() < 0.65 else 4
    bg = int(rng.choice(PALETTE))
    colors = _choose_colors(rng, bg, piece_count)
    built = _build_chain(rng, piece_count)
    if built is None:
        return None
    orig_shapes, final_local, trial_local = built

    union_r0, union_c0, union_r1, union_c1 = _union_bbox(final_local + trial_local)
    union_h = int(union_r1 - union_r0 + 1)
    union_w = int(union_c1 - union_c0 + 1)

    min_h = max(12, union_h + 2)
    min_w = max(12, union_w, piece_count + 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        h_hi = min(30, max(min_h, union_h + 8))
        w_hi = min(30, max(min_w, union_w + 10))
        h = rng.randint(min_h, h_hi)
        w = rng.randint(min_w, w_hi)

        row_room = int(h - union_h - 2)
        col_room = int(w - union_w)
        if row_room < 0 or col_room < 0:
            continue

        out_dr = rng.randint(0, row_room)
        out_dc = rng.randint(0, col_room)
        final_embedded = [_shift(coords, out_dr, out_dc) for coords in final_local]
        trial_embedded = [_shift(coords, out_dr, out_dc) for coords in trial_local]

        input_grid = np.full((h, w), bg, dtype=int)
        output_grid = np.full((h, w), bg, dtype=int)

        blocked: set[tuple[int, int]] = set()
        used_input_positions: list[np.ndarray] = []

        first_input = final_embedded[0]
        used_input_positions.append(first_input)
        first_pts = _coord_set(first_input)
        blocked |= _grow8(first_pts, h, w)
        _paint(input_grid, first_input, colors[0])

        all_scattered = True
        for idx in range(1, piece_count):
            avoid = _coord_set(final_embedded[idx])
            scattered = _scatter_piece(rng, orig_shapes[idx], h, w, blocked, avoid_exact=avoid)
            if scattered is None:
                all_scattered = False
                break
            used_input_positions.append(scattered)
            pts = _coord_set(scattered)
            blocked |= _grow8(pts, h, w)
            _paint(input_grid, scattered, colors[idx])

        if not all_scattered:
            continue

        strip_cols = _sample_strip_columns(rng, w, piece_count)
        for col, color in zip(strip_cols, colors):
            input_grid[h - 1, int(col)] = int(color)

        for color, coords in zip(colors, final_embedded):
            _paint(output_grid, coords, color)

        if _bg_color(input_grid) != bg:
            continue
        if _bg_color(output_grid) != bg:
            continue
        if not all(_in_bounds(coords, h, w) for coords in trial_embedded):
            continue

        try:
            solved = main(input_grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, output_grid):
            continue

        return input_grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
