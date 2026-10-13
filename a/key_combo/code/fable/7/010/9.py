from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


PATH_COLOR = Color.BLUE
MIRROR_COLOR = Color.RED
BG_COLOR = Color.BLACK
DECOY_COLORS = [Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

REFLECTION_COUNT_CHOICES = [1, 2, 2, 3, 3, 4, 4, 5]
SEGMENT_LEN_RANGE = (2, 6)
EXTRA_MARGIN_RANGE = (1, 6)
GRID_LIMIT = 30
DECOY_COUNT_RANGE = (0, 4)
GENERATION_RETRIES = 400


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _diag_neighbors(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]


def _orth_neighbors(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _adjacency(coords: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    return {rc: [nbr for nbr in _diag_neighbors(rc) if nbr in coords] for rc in coords}


def _is_simple_diagonal_path(coords: set[tuple[int, int]]) -> bool:
    if len(coords) < 3:
        return False
    if any(nbr in coords for rc in coords for nbr in _orth_neighbors(rc)):
        return False

    adj = _adjacency(coords)
    degrees = sorted(len(nbrs) for nbrs in adj.values())
    if degrees[:2] != [1, 1]:
        return False
    if any(deg not in (1, 2) for deg in degrees):
        return False
    if any(deg != 2 for deg in degrees[2:]):
        return False
    return True


def _ordered_path(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    adj = _adjacency(coords)
    endpoints = sorted(rc for rc, nbrs in adj.items() if len(nbrs) == 1)
    assert len(endpoints) == 2, "expected exactly two path endpoints"

    start = endpoints[0]
    ordered = [start]
    prev = None
    curr = start
    while True:
        nxts = [nbr for nbr in adj[curr] if nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, curr = curr, nxt

    assert len(ordered) == len(coords), "path traversal must cover the whole component"
    return ordered


def _choose_start_endpoint(coords: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int]:
    adj = _adjacency(coords)
    endpoints = [rc for rc, nbrs in adj.items() if len(nbrs) == 1]
    starts = []
    for r, c in endpoints:
        nr, nc = adj[(r, c)][0]
        dr, dc = nr - r, nc - c
        prev = (r - dr, c - dc)
        if not in_bounds(shape, prev[0], prev[1]):
            starts.append((r, c))
    assert len(starts) == 1, "expected one endpoint whose previous diagonal step leaves the grid"
    return starts[0]


def _oriented_path(coords: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    start = _choose_start_endpoint(coords, shape)
    adj = _adjacency(coords)
    ordered = [start]
    prev = None
    curr = start
    while True:
        nxts = [nbr for nbr in adj[curr] if nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, curr = curr, nxt
    assert len(ordered) == len(coords), "path traversal must cover the whole component"
    return ordered


def _find_path_component(g: np.ndarray) -> set[tuple[int, int]]:
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, colors=[PATH_COLOR])
    candidates = []
    for comp in comps:
        coords = _coord_set(comp["coords"])
        if _is_simple_diagonal_path(coords):
            candidates.append(coords)
    assert len(candidates) == 1, "expected one simple blue diagonal path"
    return candidates[0]


def _corner_records(path: list[tuple[int, int]]) -> list[dict]:
    corners = []
    for idx in range(1, len(path) - 1):
        prev_r, prev_c = path[idx - 1]
        cur_r, cur_c = path[idx]
        nxt_r, nxt_c = path[idx + 1]
        incoming = (cur_r - prev_r, cur_c - prev_c)
        outgoing = (nxt_r - cur_r, nxt_c - cur_c)
        if incoming == outgoing:
            continue

        row_flip = incoming[0] == -outgoing[0] and incoming[1] == outgoing[1]
        col_flip = incoming[0] == outgoing[0] and incoming[1] == -outgoing[1]
        assert row_flip ^ col_flip, "corner must flip exactly one diagonal component"

        corners.append(
            {
                "corner": (cur_r, cur_c),
                "incoming": incoming,
                "outgoing": outgoing,
                "axis": "H" if row_flip else "V",
                "mirror": (cur_r + incoming[0], cur_c + incoming[1]),
            }
        )
    assert corners, "puzzle requires at least one reflection corner"
    return corners


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the erased red mirrors that caused each turn of the blue diagonal light path.

    Input structure:
    - A black-background grid contains one complete blue 8-connected path made only of diagonal steps.
    - The path is a simple polyline with straight diagonal segments joined at corners.
    - The red single-cell mirrors are missing, and unrelated non-blue decoy pixels may also appear.

    Rule:
    - Isolate the unique simple blue diagonal path and orient it from the unique endpoint whose previous
      diagonal step would already be outside the grid.
    - At each corner, compare the incoming and outgoing diagonal directions: if the row sign flips then the
      erased mirror was horizontal, and if the column sign flips then the erased mirror was vertical.
    - Place one red cell at the square reached by taking one more step from the corner along the incoming
      direction. Keep the blue path and every unrelated pixel unchanged.

    Key insight:
    - Each bend is the visible effect of an erased reflector, and the missing mirror sits exactly in the
      square the light would have entered if that reflection had not happened.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    path_coords = _find_path_component(g)
    path = _oriented_path(path_coords, g.shape)
    corners = _corner_records(path)

    out = g.copy()
    for rec in corners:
        mr, mc = rec["mirror"]
        assert in_bounds(g.shape, mr, mc), "mirror must be in bounds"
        assert (mr, mc) not in path_coords, "mirror may not overlap the path"
        assert int(g[mr, mc]) == int(bg), "mirror location must be empty in the input"
        out[mr, mc] = MIRROR_COLOR

    return out


def _too_close(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    *,
    allow: set[tuple[int, int]] | None = None,
) -> bool:
    allow = set() if allow is None else allow
    r, c = cell
    for rr, cc in occupied:
        if (rr, cc) in allow:
            continue
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _sample_orientations(n_reflections: int) -> list[str]:
    if n_reflections == 1:
        return [random.choice(["H", "V"])]

    seq = [random.choice(["H", "V"])]
    for _ in range(1, n_reflections):
        if random.random() < 0.65:
            seq.append("V" if seq[-1] == "H" else "H")
        else:
            seq.append(random.choice(["H", "V"]))
    if len(set(seq)) == 1:
        flip_idx = random.randrange(n_reflections)
        seq[flip_idx] = "V" if seq[flip_idx] == "H" else "H"
    return seq


def _abstract_case() -> dict | None:
    n_reflections = random.choice(REFLECTION_COUNT_CHOICES)
    lengths = [random.randint(*SEGMENT_LEN_RANGE) for _ in range(n_reflections + 1)]
    direction = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    orientations = _sample_orientations(n_reflections)

    path = [(0, 0)]
    path_set = {(0, 0)}
    mirror_set: set[tuple[int, int]] = set()
    mirrors = []
    curr = (0, 0)

    for seg_idx, seg_len in enumerate(lengths):
        for _ in range(seg_len):
            nxt = (curr[0] + direction[0], curr[1] + direction[1])
            if nxt in mirror_set or _too_close(nxt, path_set, allow={curr}):
                return None
            path.append(nxt)
            path_set.add(nxt)
            curr = nxt

        if seg_idx == n_reflections:
            break

        axis = orientations[seg_idx]
        mirror = (curr[0] + direction[0], curr[1] + direction[1])
        new_direction = (-direction[0], direction[1]) if axis == "H" else (direction[0], -direction[1])
        first_after_turn = (curr[0] + new_direction[0], curr[1] + new_direction[1])

        if mirror in mirror_set or _too_close(mirror, path_set, allow={curr}):
            return None
        if first_after_turn in mirror_set or _too_close(first_after_turn, path_set, allow={curr}):
            return None

        mirror_set.add(mirror)
        mirrors.append(
            {
                "corner": curr,
                "incoming": direction,
                "outgoing": new_direction,
                "axis": axis,
                "mirror": mirror,
            }
        )
        direction = new_direction

    if not _is_simple_diagonal_path(path_set):
        return None
    if len(mirrors) != n_reflections:
        return None

    all_cells = path_set | mirror_set
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    inner_h = r_max - r_min + 1
    inner_w = c_max - c_min + 1
    if inner_h > GRID_LIMIT or inner_w > GRID_LIMIT:
        return None

    start = path[0]
    end = path[-1]
    first_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    last_dir = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
    after_end = (end[0] + last_dir[0], end[1] + last_dir[1])

    feasible_axes = []
    if first_dir[0] == 1 and start[0] == r_min:
        feasible_axes.append("row")
    if first_dir[0] == -1 and start[0] == r_max:
        feasible_axes.append("row")
    if first_dir[1] == 1 and start[1] == c_min:
        feasible_axes.append("col")
    if first_dir[1] == -1 and start[1] == c_max:
        feasible_axes.append("col")
    if not feasible_axes:
        return None

    axis = random.choice(feasible_axes)
    placed = _place_with_direction_clue(path, mirrors, axis, (r_min, r_max, c_min, c_max), after_end)
    if placed is None:
        return None
    shifted_path, shifted_mirrors, shifted_corner_records, h, w = placed

    inp = np.full((h, w), BG_COLOR, dtype=int)
    out = inp.copy()
    for r, c in shifted_path:
        inp[r, c] = PATH_COLOR
        out[r, c] = PATH_COLOR
    for r, c in shifted_mirrors:
        out[r, c] = MIRROR_COLOR

    occupied = set(shifted_path) | set(shifted_mirrors)
    n_decoys = random.randint(*DECOY_COUNT_RANGE)
    for _ in range(n_decoys):
        if not _place_decoy(inp, out, occupied):
            break

    return {
        "input": inp,
        "output": out,
        "path": shifted_path,
        "corners": shifted_corner_records,
        "mirrors": shifted_mirrors,
    }


def _free_axis_size_and_shift(
    lo: int,
    hi: int,
    extra_point: int,
) -> tuple[int, int] | None:
    min_size = max(hi, extra_point) - min(lo, extra_point) + 1
    if min_size > GRID_LIMIT:
        return None
    max_size = min(GRID_LIMIT, min_size + EXTRA_MARGIN_RANGE[1])
    size = random.randint(min_size, max_size)

    shift_lo = max(-lo, -extra_point)
    shift_hi = min(size - 1 - hi, size - 1 - extra_point)
    if shift_lo > shift_hi:
        return None
    shift = random.randint(shift_lo, shift_hi)
    return size, shift


def _place_with_direction_clue(
    path: list[tuple[int, int]],
    mirrors: list[dict],
    axis: str,
    bbox: tuple[int, int, int, int],
    after_end: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict], int, int] | None:
    r_min, r_max, c_min, c_max = bbox
    start = path[0]
    first_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    last_dir = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])

    if axis == "row":
        if first_dir[0] == 1:
            r_shift = -start[0]
        else:
            min_h = max(r_max - start[0] + 1, after_end[0] - start[0] + 1)
            if min_h > GRID_LIMIT:
                return None
            max_h = min(GRID_LIMIT, min_h + EXTRA_MARGIN_RANGE[1])
            h = random.randint(min_h, max_h)
            r_shift = h - 1 - start[0]
        if first_dir[0] == 1:
            min_h = max(r_max + r_shift + 1, after_end[0] + r_shift + 1)
            if after_end[0] + r_shift < 0 or min_h > GRID_LIMIT:
                return None
            max_h = min(GRID_LIMIT, min_h + EXTRA_MARGIN_RANGE[1])
            h = random.randint(min_h, max_h)
        free = _free_axis_size_and_shift(c_min, c_max, after_end[1])
        if free is None:
            return None
        w, c_shift = free
    else:
        if first_dir[1] == 1:
            c_shift = -start[1]
        else:
            min_w = max(c_max - start[1] + 1, after_end[1] - start[1] + 1)
            if min_w > GRID_LIMIT:
                return None
            max_w = min(GRID_LIMIT, min_w + EXTRA_MARGIN_RANGE[1])
            w = random.randint(min_w, max_w)
            c_shift = w - 1 - start[1]
        if first_dir[1] == 1:
            min_w = max(c_max + c_shift + 1, after_end[1] + c_shift + 1)
            if after_end[1] + c_shift < 0 or min_w > GRID_LIMIT:
                return None
            max_w = min(GRID_LIMIT, min_w + EXTRA_MARGIN_RANGE[1])
            w = random.randint(min_w, max_w)
        free = _free_axis_size_and_shift(r_min, r_max, after_end[0])
        if free is None:
            return None
        h, r_shift = free

    shifted_path = [(r + r_shift, c + c_shift) for r, c in path]
    shifted_mirrors = [(rec["mirror"][0] + r_shift, rec["mirror"][1] + c_shift) for rec in mirrors]
    shifted_corner_records = []
    for rec in mirrors:
        shifted_corner_records.append(
            {
                **rec,
                "corner": (rec["corner"][0] + r_shift, rec["corner"][1] + c_shift),
                "mirror": (rec["mirror"][0] + r_shift, rec["mirror"][1] + c_shift),
            }
        )

    if not all(in_bounds((h, w), r, c) for r, c in shifted_path):
        return None
    if not all(in_bounds((h, w), r, c) for r, c in shifted_mirrors):
        return None
    if not in_bounds((h, w), path[-1][0] + r_shift + last_dir[0], path[-1][1] + c_shift + last_dir[1]):
        return None
    return shifted_path, shifted_mirrors, shifted_corner_records, h, w


def _decoy_shapes() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
    ]


def _place_decoy(inp: np.ndarray, out: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    color = random.choice(DECOY_COLORS)
    shape = random.choice(_decoy_shapes())
    for _ in range(60):
        r0 = random.randint(0, h - 1)
        c0 = random.randint(0, w - 1)
        coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if not all(in_bounds(inp.shape, r, c) for r, c in coords):
            continue
        if any(_too_close((r, c), occupied) for r, c in coords):
            continue
        if len(set(coords)) != len(coords):
            continue
        if any(inp[r, c] != BG_COLOR for r, c in coords):
            continue
        for r, c in coords:
            inp[r, c] = color
            out[r, c] = color
            occupied.add((r, c))
        return True
    return False


def _generate_case() -> dict:
    for _ in range(GENERATION_RETRIES):
        case = _abstract_case()
        if case is not None and _validate_case(case):
            return case
    raise RuntimeError("failed to generate a valid inverted-light-path puzzle")


def generate_input() -> np.ndarray:
    return _generate_case()["input"]


def _validate_case(case: dict) -> bool:
    inp = case["input"]
    out = case["output"]
    if not (isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)):
        return False
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > GRID_LIMIT or max(out.shape) > GRID_LIMIT:
        return False

    pred = main(inp)
    if not np.array_equal(pred, out):
        return False

    path = _oriented_path(_find_path_component(inp), inp.shape)
    corners = _corner_records(path)
    added = set(map(tuple, np.argwhere((pred == MIRROR_COLOR) & (inp != MIRROR_COLOR))))
    expected = {tuple(rec["mirror"]) for rec in corners}
    if added != expected:
        return False

    if len(corners) != len(case["corners"]):
        return False
    if {tuple(rec["mirror"]) for rec in case["corners"]} != expected:
        return False
    if len({rec["axis"] for rec in corners}) > 2:
        return False
    return True
