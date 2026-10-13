from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
TEAL = 8
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 9)
PAIR_COUNT_OPTIONS = (3, 4)
MAX_ATTEMPTS = 400


BASE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (2, 3)),
    ((0, 1), (1, 1), (2, 1), (2, 0), (3, 0)),
)


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _shape_array(coords: list[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    arr = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        arr[r, c] = True
    return arr


def _is_asymmetric(coords: list[tuple[int, int]]) -> bool:
    arr = _shape_array(coords)
    return not np.array_equal(arr, np.fliplr(arr)) and not np.array_equal(arr, np.flipud(arr))


def _random_shape() -> list[tuple[int, int]]:
    coords = [tuple(cell) for cell in random.choice(BASE_SHAPES)]
    arr = _shape_array(coords)
    if random.random() < 0.5:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, random.randint(0, 3))
    coords2 = [tuple(map(int, rc)) for rc in np.argwhere(arr)]
    coords2 = _normalize(coords2)
    if not _is_asymmetric(coords2):
        return _random_shape()
    return coords2


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox_center(coords: list[tuple[int, int]]) -> tuple[float, float]:
    r0, c0, r1, c1 = _bbox(coords)
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, grid.flatten()))
    if exclude is not None:
        cols -= set(exclude)
    return sorted(cols)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": int(color),
                    "coords": sorted(coords),
                    "bbox": _bbox(coords),
                    "center": _bbox_center(coords),
                    "area": len(coords),
                }
            )
    return comps


def _pair_infos(grid: np.ndarray) -> list[dict]:
    colors = _colors_present(grid, exclude={BG, GREY, TEAL})
    infos = []
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(coords) == 2:
            a, b = sorted(coords)
            assert a[0] != b[0] and a[1] != b[1], "marker pair must be on different rows and columns"
            top, bottom = (a, b) if a[0] < b[0] else (b, a)
            elbow = (top[0], bottom[1])
            infos.append({"color": int(color), "markers": (top, bottom), "elbow": elbow})
        else:
            raise AssertionError("every non-grey non-teal foreground color must appear exactly twice")
    assert 3 <= len(infos) <= 4, "expected 3-4 marker pairs"
    return infos


def _path_cells(top: tuple[int, int], bottom: tuple[int, int]) -> list[tuple[int, int]]:
    tr, tc = top
    br, bc = bottom
    assert tr < br and tc != bc
    c_step = 1 if bc > tc else -1
    path = [(tr, c) for c in range(tc + c_step, bc + c_step, c_step)]
    path.extend((r, bc) for r in range(tr + 1, br))
    return path


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _associate_objects(elbows: list[tuple[int, int]], grey_objects: list[dict]) -> tuple[list[int], list[float]]:
    assert len(grey_objects) >= len(elbows)
    chosen = []
    margins = []
    used = set()
    for elbow in elbows:
        dists = sorted((_dist2((float(elbow[0]), float(elbow[1])), obj["center"]), idx) for idx, obj in enumerate(grey_objects))
        assert len(dists) >= 2 or len(grey_objects) == 1
        best_dist, best_idx = dists[0]
        if len(dists) >= 2:
            second_dist = dists[1][0]
            assert best_dist + 0.49 < second_dist, "grey-object ownership must be unambiguous"
            margins.append(second_dist - best_dist)
        else:
            margins.append(999.0)
        assert best_idx not in used, "each elbow must own a distinct nearest grey object"
        used.add(best_idx)
        chosen.append(best_idx)
    return chosen, margins


def _reflect(coords: list[tuple[int, int]], elbow: tuple[int, int], odd_rank: bool) -> list[tuple[int, int]]:
    er, ec = elbow
    if odd_rank:
        return sorted((r, 2 * ec - c) for r, c in coords)
    return sorted((2 * er - r, c) for r, c in coords)


def _render_output_from_parts(
    grid: np.ndarray,
    ordered_pairs: list[dict],
    grey_objects: list[dict],
    object_indices: list[int],
    *,
    branch_override: dict[int, bool] | None = None,
) -> np.ndarray:
    out = grid.copy()
    for pair in ordered_pairs:
        top, bottom = pair["markers"]
        for r, c in _path_cells(top, bottom):
            assert int(out[r, c]) == BG, "L path must traverse only background"
            out[r, c] = TEAL

    for rank, pair in enumerate(ordered_pairs, start=1):
        odd_rank = rank % 2 == 1
        if branch_override and rank in branch_override:
            odd_rank = branch_override[rank]
        obj = grey_objects[object_indices[rank - 1]]
        reflected = _reflect(obj["coords"], pair["elbow"], odd_rank)
        for r, c in reflected:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert int(out[r, c]) == BG, "reflected copy must land on empty background"
        for r, c in reflected:
            out[r, c] = pair["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw all marker-defined teal L paths, rank their elbows by row, then mirror the nearest grey object across the elbow-selected axis and recolor the copy with the marker-pair color.

    Input structure:
    - A black background contains 3-4 color pairs of singleton markers; each pair uses a unique color and its two markers are in different rows and different columns.
    - Each pair determines a horizontal-then-vertical L path whose elbow is at `(top_marker.row, bottom_marker.col)`.
    - The grid also contains the same number of small 4-connected grey objects, each clearly closest to one elbow by bbox-center Euclidean distance.

    Rule:
    - For every same-color marker pair, start from the marker in the smaller row and draw a teal L path: first horizontally to the other marker's column, then vertically down to the other marker, excluding the markers themselves.
    - Sort the elbows by row from top to bottom and assign ranks `1, 2, 3, ...`.
    - Associate each elbow with the grey object whose bbox center is nearest to that elbow.
    - Odd-ranked elbows use their column as a vertical mirror axis, so the associated grey object is reflected horizontally; even-ranked elbows use their row as a horizontal mirror axis, so the associated grey object is reflected vertically.
    - Keep the original markers and grey objects, and paint each reflected copy in that pair's marker color.

    Key insight:
    - The mirror axis direction is not local to one pair; it emerges only after all elbows are globally ordered by row, so the solver must combine path construction, elbow ranking, nearest-object matching, and reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    bg = _get_bg(g)
    assert bg == BG, "generator uses black background"

    pairs = _pair_infos(g)
    ordered_pairs = sorted(pairs, key=lambda info: (info["elbow"][0], info["elbow"][1]))
    grey_objects = _find_components_of_color(g, GREY)
    assert len(grey_objects) == len(ordered_pairs), "expected one grey object per marker pair"
    object_indices, _ = _associate_objects([pair["elbow"] for pair in ordered_pairs], grey_objects)
    return _render_output_from_parts(g, ordered_pairs, grey_objects, object_indices)


def _sample_elbow_rows(pair_count: int) -> list[int]:
    top_margin = random.randint(3, 4)
    return [top_margin + 5 * i for i in range(pair_count)]


def _marker_skeletons(height: int, width: int, colors: list[int], elbow_rows: list[int]) -> list[dict] | None:
    infos = []
    reserved_markers: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    for er, color in zip(elbow_rows, colors):
        br = er + random.choice((2, 3))
        if br >= height - 1:
            return None
        cols = list(range(7, width - 7))
        random.shuffle(cols)
        placed = None
        for ec in cols:
            side = random.choice((-1, 1))
            offset = random.randint(4, 6)
            tc = ec + side * offset
            if not (1 <= tc < width - 1):
                continue
            top = (er, tc)
            bottom = (br, ec)
            path = _path_cells(top, bottom)
            cells = {top, bottom, *path}
            if cells & reserved_markers:
                continue
            if set(path) & reserved_paths:
                continue
            if any(cell in reserved_markers for cell in path):
                continue
            placed = {"color": int(color), "markers": (top, bottom), "elbow": (er, ec), "path": path}
            break
        if placed is None:
            return None
        reserved_markers.update(placed["markers"])
        reserved_paths.update(placed["path"])
        infos.append(placed)
    return infos


def _place_shape(coords: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return sorted((top + r, left + c) for r, c in coords)


def _try_object_placement(
    pair: dict,
    rank: int,
    height: int,
    width: int,
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    er, ec = pair["elbow"]
    odd_rank = rank % 2 == 1
    for _ in range(60):
        shape = _random_shape()
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        gap_r = random.choice((1, 2))
        gap_c = random.choice((1, 2))
        above = random.choice((True, False))
        left_side = random.choice((True, False))
        top = er - gap_r - h + 1 if above else er + gap_r
        left = ec - gap_c - w + 1 if left_side else ec + gap_c
        placed = _place_shape(shape, top, left)
        if any(not (0 <= r < height and 0 <= c < width) for r, c in placed):
            continue
        if any((r, c) in input_occupied for r, c in placed):
            continue
        reflected = _reflect(placed, pair["elbow"], odd_rank)
        if any(not (0 <= r < height and 0 <= c < width) for r, c in reflected):
            continue
        if set(placed) & set(reflected):
            continue
        if any((r, c) in output_occupied for r, c in placed):
            continue
        if any((r, c) in output_occupied for r, c in reflected):
            continue
        return placed, reflected
    return None


def _build_input_grid(height: int, width: int, pairs: list[dict], grey_coords: list[list[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for pair in pairs:
        for r, c in pair["markers"]:
            grid[r, c] = pair["color"]
    for coords in grey_coords:
        for r, c in coords:
            grid[r, c] = GREY
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        pairs = _pair_infos(grid)
        ordered_pairs = sorted(pairs, key=lambda info: (info["elbow"][0], info["elbow"][1]))
        grey_objects = _find_components_of_color(grid, GREY)
        if len(grey_objects) != len(ordered_pairs):
            return False
        object_indices, margins = _associate_objects([pair["elbow"] for pair in ordered_pairs], grey_objects)
        if min(margins) < 1.0:
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        for rank in range(1, len(ordered_pairs) + 1):
            alt = _render_output_from_parts(
                grid,
                ordered_pairs,
                grey_objects,
                object_indices,
                branch_override={rank: rank % 2 == 0},
            )
            if np.array_equal(alt, out):
                return False
        wrong_all = _render_output_from_parts(
            grid,
            ordered_pairs,
            grey_objects,
            object_indices,
            branch_override={rank: False for rank in range(1, len(ordered_pairs) + 1)},
        )
        if np.array_equal(wrong_all, out):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    colors = random.sample(MARKER_COLORS, pair_count)
    elbow_rows = _sample_elbow_rows(pair_count)
    height = elbow_rows[-1] + random.randint(6, 8)
    width = random.randint(18, 24)
    if max(height, width) > 30:
        return None

    pairs = _marker_skeletons(height, width, colors, elbow_rows)
    if pairs is None:
        return None
    ordered_pairs = sorted(pairs, key=lambda info: (info["elbow"][0], info["elbow"][1]))

    input_occupied = set()
    output_occupied = set()
    for pair in pairs:
        input_occupied.update(pair["markers"])
        output_occupied.update(pair["markers"])
        output_occupied.update(pair["path"])

    grey_coords = []
    for rank, pair in enumerate(ordered_pairs, start=1):
        placed = _try_object_placement(pair, rank, height, width, input_occupied, output_occupied)
        if placed is None:
            return None
        grey, reflected = placed
        grey_coords.append(grey)
        input_occupied.update(grey)
        output_occupied.update(grey)
        output_occupied.update(reflected)

    grid = _build_input_grid(height, width, pairs, grey_coords)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
