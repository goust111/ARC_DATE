from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
import random

import numpy as np


try:
    from arc_utils import Color, find_components, get_bg, in_bounds
except Exception:
    try:
        _ARC_UTILS = SourceFileLoader(
            "_shared_arc_utils", "/home/lixinyu/snap/new/solvers/arc_utils.py"
        ).load_module()
    except Exception:
        _ARC_UTILS = None

    if _ARC_UTILS is not None:
        Color = _ARC_UTILS.Color
        find_components = _ARC_UTILS.find_components
        get_bg = _ARC_UTILS.get_bg
        in_bounds = _ARC_UTILS.in_bounds
    else:
        class Color:
            BLACK = 0
            BLUE = 1
            RED = 2
            GREEN = 3
            YELLOW = 4
            GREY = 5
            PINK = 6
            ORANGE = 7
            TEAL = 8
            MAROON = 9

        def get_bg(grid: np.ndarray) -> int:
            return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

        def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
            h, w = map(int, shape)
            return 0 <= int(r) < h and 0 <= int(c) < w

        def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            return int(r0), int(c0), int(r1), int(c1)

        def find_components(
            grid: np.ndarray,
            *,
            bg: int | None = None,
            connectivity: int = 4,
            monochromatic: bool = True,
            colors: list[int] | tuple[int, ...] | None = None,
            min_area: int = 1,
            return_format: str = "dict",
        ) -> list[dict]:
            g = np.asarray(grid, dtype=int)
            if bg is None:
                bg = get_bg(g)
            bg = int(bg)
            allow = None if colors is None else {int(c) for c in colors if int(c) != bg}
            h, w = g.shape
            seen = np.zeros((h, w), dtype=bool)
            if int(connectivity) == 4:
                deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
            else:
                deltas = (
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1),
                )

            out: list[dict] = []
            for r in range(h):
                for c in range(w):
                    if seen[r, c]:
                        continue
                    color = int(g[r, c])
                    if color == bg:
                        continue
                    if allow is not None and color not in allow:
                        continue
                    seen[r, c] = True
                    stack = [(r, c)]
                    coords: list[tuple[int, int]] = []
                    while stack:
                        rr, cc = stack.pop()
                        coords.append((rr, cc))
                        for dr, dc in deltas:
                            nr = rr + dr
                            nc = cc + dc
                            if not (0 <= nr < h and 0 <= nc < w):
                                continue
                            if seen[nr, nc]:
                                continue
                            nxt = int(g[nr, nc])
                            if nxt == bg:
                                continue
                            if allow is not None and nxt not in allow:
                                continue
                            if monochromatic and nxt != color:
                                continue
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                    coords_arr = np.array(coords, dtype=int)
                    if int(coords_arr.shape[0]) < int(min_area):
                        continue
                    out.append(
                        {
                            "color": color if monochromatic else None,
                            "coords": coords_arr,
                            "bbox": _bbox_coords(coords_arr),
                            "area": int(coords_arr.shape[0]),
                        }
                    )
            if return_format != "dict":
                raise ValueError("fallback find_components supports only return_format='dict'")
            return out


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Segment:
    color: int
    orientation: str


@dataclass(frozen=True)
class MirrorPlan:
    color: int
    orientation: str
    cells: tuple[Coord, ...]


@dataclass(frozen=True)
class DecoyPlan:
    color: int
    cells: tuple[Coord, ...]


GEN_MAX_ATTEMPTS = 500
GEN_HEIGHT_RANGE = (18, 27)
GEN_WIDTH_RANGE = (18, 27)
GEN_REFLECTIONS = (1, 2, 3)
GEN_REFLECTION_WEIGHTS = (2, 4, 3)
GEN_SEED_LENGTHS = (2, 3)
GEN_MIN_SEG = 3
GEN_MIN_FINAL_SEG = 3
GEN_MIN_BAR = 2
GEN_MAX_BAR = 8
GEN_ONPATH_DECOYS = (1, 2)
GEN_OFFPATH_DECOYS = (0, 1, 2)
GEN_CORNERS = ("tl", "tr", "bl", "br")
GEN_COLORS = tuple(range(1, 10))


def _bbox_size(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _legend_bbox(shape: tuple[int, int], rows: int, corner: str) -> BBox:
    h, w = map(int, shape)
    rows = int(rows)
    if corner == "tl":
        return 0, 0, rows - 1, 1
    if corner == "tr":
        return 0, w - 2, rows - 1, w - 1
    if corner == "bl":
        return h - rows, 0, h - 1, 1
    if corner == "br":
        return h - rows, w - 2, h - 1, w - 1
    raise ValueError("invalid legend corner")


def _expanded_cells(shape: tuple[int, int], box: BBox, pad: int = 1) -> set[Coord]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = box
    out: set[Coord] = set()
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            out.add((r, c))
    return out


def _coord_set(coords: np.ndarray) -> set[Coord]:
    return {tuple(map(int, rc)) for rc in coords}


def _is_contiguous_diagonal(coords: set[Coord]) -> bool:
    if len(coords) not in (2, 3):
        return False
    if len({r - c for r, c in coords}) == 1:
        rows = sorted(r for r, _c in coords)
        cols = sorted(c for _r, c in coords)
    elif len({r + c for r, c in coords}) == 1:
        rows = sorted(r for r, _c in coords)
        cols = sorted(c for _r, c in coords)
    else:
        return False
    return (
        rows == list(range(rows[0], rows[-1] + 1))
        and cols == list(range(cols[0], cols[-1] + 1))
        and len(coords) == rows[-1] - rows[0] + 1
    )


def _legend(grid: np.ndarray, bg: int) -> tuple[BBox, dict[int, int], np.ndarray]:
    h, w = grid.shape
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    candidates: list[tuple[BBox, dict[int, int], np.ndarray]] = []

    for comp in comps:
        box = tuple(map(int, comp["bbox"]))
        bh, bw = _bbox_size(box)
        if bw != 2 or bh < 2:
            continue
        r0, c0, r1, c1 = box
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if not touches_corner:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        mapping: dict[int, int] = {}
        valid = True
        for left, right in patch:
            left = int(left)
            right = int(right)
            if left in mapping:
                valid = False
                break
            mapping[left] = right
        if valid:
            candidates.append((box, mapping, patch.copy()))

    assert len(candidates) == 1, "expected one compact 2-column legend in a corner"
    return candidates[0]


def _ray_seed(grid: np.ndarray, bg: int, legend_box: BBox) -> tuple[int, set[Coord]]:
    work = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = legend_box
    work[r0 : r1 + 1, c0 : c1 + 1] = bg
    comps = find_components(work, bg=bg, connectivity=8, monochromatic=True)
    candidates: list[tuple[int, set[Coord]]] = []
    for comp in comps:
        coords = _coord_set(comp["coords"])
        if _is_contiguous_diagonal(coords):
            candidates.append((int(comp["color"]), coords))
    assert len(candidates) == 1, "expected one diagonal 2-3 cell seed"
    return candidates[0]


def _initial_state(shape: tuple[int, int], seed: set[Coord]) -> tuple[int, int, int, int]:
    if len({r - c for r, c in seed}) == 1:
        first = min(seed)
        last = max(seed)
        endpoints = ((first, (-1, -1)), (last, (1, 1)))
    else:
        top = min(seed)
        bottom = max(seed)
        endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

    candidates = []
    for (r, c), (dr, dc) in endpoints:
        if in_bounds(shape, r + dr, c + dc):
            candidates.append((r, c, dr, dc))
    assert len(candidates) == 1, "seed must indicate a unique forward diagonal direction"
    return candidates[0]


def _segments(grid: np.ndarray, bg: int, legend_box: BBox, seed: set[Coord]) -> dict[Coord, Segment]:
    work = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = legend_box
    work[r0 : r1 + 1, c0 : c1 + 1] = bg
    for r, c in seed:
        work[r, c] = bg

    segment_map: dict[Coord, Segment] = {}
    comps = find_components(work, bg=bg, connectivity=4, monochromatic=True)
    for comp in comps:
        coords = comp["coords"]
        color = int(comp["color"])
        area = int(comp["area"])
        assert area >= 2, "all non-seed foreground objects must be line segments"
        r0, c0, r1, c1 = map(int, comp["bbox"])
        bh, bw = _bbox_size((r0, c0, r1, c1))
        if bh == 1:
            assert area == bw, "horizontal segment must be contiguous"
            orientation = "horizontal"
        elif bw == 1:
            assert area == bh, "vertical segment must be contiguous"
            orientation = "vertical"
        else:
            raise AssertionError("line segments must be horizontal or vertical")
        for cell in _coord_set(coords):
            segment_map[cell] = Segment(color=color, orientation=orientation)
    assert segment_map, "expected at least one line segment"
    return segment_map


def _reflect_direction(orientation: str, dr: int, dc: int) -> tuple[int, int]:
    if orientation == "horizontal":
        return -dr, dc
    return dr, -dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a diagonal ray through state-matched mirrors and recolor each reflected leg from the corner legend.

    Input structure:
    - The grid has a black background, one 2-3 cell monochromatic diagonal seed, and several straight horizontal or vertical colored segments.
    - A compact full 2-column legend block sits in one corner; each row maps its left color to its right color.
    - The seed, legend, and line segments are separated so the seed is the only diagonal 8-connected object.

    Rule:
    - Start from the seed endpoint whose outward diagonal continuation stays inside the grid, and move cell by cell in that diagonal direction.
    - Paint only background cells with the ray's current color.
    - If the next cell belongs to a segment of the current color, do not paint that cell; reflect on a horizontal segment by flipping the row direction or on a vertical segment by flipping the column direction, then recolor the ray using the legend row whose left entry matches the current color.
    - If the next cell belongs to a segment of any other color, move through it transparently without repainting it.
    - Stop when the next diagonal step would leave the grid; preserve the seed, legend, and every segment cell.

    Key insight:
    - Mirror behavior depends on the ray's current color, not on a fixed mirror color, and the legend chains those color states together so the path is a sequence of conditional reflections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    legend_box, legend_map, _legend_patch = _legend(g, bg)
    ray_color, seed = _ray_seed(g, bg, legend_box)
    r, c, dr, dc = _initial_state(g.shape, seed)
    segment_map = _segments(g, bg, legend_box, seed)

    out = g.copy()
    seen: set[tuple[int, int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc, ray_color)
        assert state not in seen, "ray path must leave the grid instead of looping"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break

        segment = segment_map.get((nr, nc))
        if segment is None:
            assert int(g[nr, nc]) == bg, "only background cells may lie outside the seed, legend, and line segments"
            out[nr, nc] = ray_color
            r, c = nr, nc
            continue

        if segment.color == ray_color:
            assert ray_color in legend_map, "every reflecting color must appear in the legend left column"
            dr, dc = _reflect_direction(segment.orientation, dr, dc)
            ray_color = int(legend_map[ray_color])
            continue

        r, c = nr, nc

    return out


def _render_legend(grid: np.ndarray, rows: list[tuple[int, int]], corner: str) -> BBox:
    box = _legend_bbox(grid.shape, len(rows), corner)
    r0, c0, r1, c1 = box
    patch = np.array(rows, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return box


def _seed_guard(shape: tuple[int, int], seed_cells: list[Coord]) -> set[Coord]:
    h, w = map(int, shape)
    out: set[Coord] = set()
    for r, c in seed_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _candidate_start_states(shape: tuple[int, int], seed_len: int, blocked: set[Coord]) -> list[tuple[int, int, int, int, list[Coord]]]:
    h, w = map(int, shape)
    out: list[tuple[int, int, int, int, list[Coord]]] = []
    for r in range(h):
        for c in range(w):
            on_edge = r in (0, h - 1) or c in (0, w - 1)
            if not on_edge:
                continue
            for dr in (-1, 1):
                for dc in (-1, 1):
                    if in_bounds(shape, r - dr, c - dc):
                        continue
                    cells: list[Coord] = []
                    ok = True
                    for i in range(seed_len):
                        rr = r + i * dr
                        cc = c + i * dc
                        if not in_bounds(shape, rr, cc) or (rr, cc) in blocked:
                            ok = False
                            break
                        cells.append((rr, cc))
                    if not ok:
                        continue
                    nr = cells[-1][0] + dr
                    nc = cells[-1][1] + dc
                    if not in_bounds(shape, nr, nc) or (nr, nc) in blocked:
                        continue
                    out.append((r, c, dr, dc, cells))
    random.shuffle(out)
    return out


def _free_run(shape: tuple[int, int], start: Coord, dr: int, dc: int, blocked: set[Coord]) -> tuple[list[Coord], bool]:
    cells: list[Coord] = []
    r, c = start
    while True:
        nr = r + dr
        nc = c + dc
        if not in_bounds(shape, nr, nc):
            return cells, True
        if (nr, nc) in blocked:
            return cells, False
        cells.append((nr, nc))
        r, c = nr, nc


def _segment_start_range(anchor: int, length: int, limit: int) -> tuple[int, int] | None:
    start_min = max(0, anchor - length + 1)
    start_max = min(anchor, limit - length)
    if start_min > start_max:
        return None
    return start_min, start_max


def _bar_cells(anchor: Coord, orientation: str, start: int, length: int) -> list[Coord]:
    r, c = anchor
    if orientation == "horizontal":
        return [(r, start + j) for j in range(length)]
    return [(start + j, c) for j in range(length)]


def _sample_bar_through(
    shape: tuple[int, int],
    anchor: Coord,
    orientation: str,
    blocked: set[Coord],
    min_len: int = GEN_MIN_BAR,
    max_len: int = GEN_MAX_BAR,
) -> list[Coord] | None:
    h, w = map(int, shape)
    limit = w if orientation == "horizontal" else h
    anchor_axis = anchor[1] if orientation == "horizontal" else anchor[0]
    lengths = list(range(min_len, min(max_len, limit) + 1))
    random.shuffle(lengths)
    for length in lengths:
        bounds = _segment_start_range(anchor_axis, length, limit)
        if bounds is None:
            continue
        starts = list(range(bounds[0], bounds[1] + 1))
        random.shuffle(starts)
        for start in starts[:8]:
            cells = _bar_cells(anchor, orientation, start, length)
            if set(cells) & blocked:
                continue
            return cells
    return None


def _plan_route(
    shape: tuple[int, int],
    pos: Coord,
    dr: int,
    dc: int,
    chain: list[int],
    color_index: int,
    reflections_left: int,
    blocked: set[Coord],
    bar_forbidden: set[Coord],
) -> tuple[list[tuple[int, int, int, int]], list[MirrorPlan]] | None:
    run_cells, exits = _free_run(shape, pos, dr, dc, blocked)
    if reflections_left == 0:
        if not exits or len(run_cells) < GEN_MIN_FINAL_SEG:
            return None
        return ([(r, c, int(chain[color_index]), color_index) for r, c in run_cells], [])

    if len(run_cells) <= GEN_MIN_SEG:
        return None

    lengths = list(range(GEN_MIN_SEG, len(run_cells)))
    random.shuffle(lengths)
    for seg_len in lengths:
        path_cells = run_cells[:seg_len]
        endpoint = path_cells[-1]
        mirror_anchor = run_cells[seg_len]
        blocked_after_path = blocked | set(path_cells)

        orientations = ["horizontal", "vertical"]
        random.shuffle(orientations)
        for orientation in orientations:
            bounced_dr, bounced_dc = _reflect_direction(orientation, dr, dc)
            next_after_bounce = (endpoint[0] + bounced_dr, endpoint[1] + bounced_dc)
            if not in_bounds(shape, next_after_bounce[0], next_after_bounce[1]):
                continue
            if next_after_bounce in blocked_after_path:
                continue

            mirror_cells = _sample_bar_through(
                shape,
                mirror_anchor,
                orientation,
                blocked_after_path | bar_forbidden,
            )
            if mirror_cells is None:
                continue

            blocked_after_mirror = blocked_after_path | set(mirror_cells)
            rest = _plan_route(
                shape,
                endpoint,
                bounced_dr,
                bounced_dc,
                chain,
                color_index + 1,
                reflections_left - 1,
                blocked_after_mirror,
                bar_forbidden,
            )
            if rest is None:
                continue

            rest_path, rest_mirrors = rest
            path_here = [(r, c, int(chain[color_index]), color_index) for r, c in path_cells]
            mirror = MirrorPlan(
                color=int(chain[color_index]),
                orientation=orientation,
                cells=tuple(mirror_cells),
            )
            return path_here + rest_path, [mirror] + rest_mirrors

    return None


def _segment_groups(path_info: list[tuple[int, int, int, int]]) -> dict[int, list[Coord]]:
    groups: dict[int, list[Coord]] = {}
    for r, c, _color, idx in path_info:
        groups.setdefault(int(idx), []).append((int(r), int(c)))
    return groups


def _bar_intersections(cells: list[Coord], path_set: set[Coord]) -> set[Coord]:
    return set(cells) & path_set


def _place_decoy_through_path(
    shape: tuple[int, int],
    anchor: Coord,
    color: int,
    existing_objects: set[Coord],
    bar_forbidden: set[Coord],
    path_set: set[Coord],
) -> DecoyPlan | None:
    orientations = ["horizontal", "vertical"]
    random.shuffle(orientations)
    for orientation in orientations:
        h, w = map(int, shape)
        limit = w if orientation == "horizontal" else h
        anchor_axis = anchor[1] if orientation == "horizontal" else anchor[0]
        lengths = list(range(GEN_MIN_BAR, min(GEN_MAX_BAR, limit) + 1))
        random.shuffle(lengths)
        for length in lengths:
            bounds = _segment_start_range(anchor_axis, length, limit)
            if bounds is None:
                continue
            starts = list(range(bounds[0], bounds[1] + 1))
            random.shuffle(starts)
            for start in starts[:10]:
                cells = _bar_cells(anchor, orientation, start, length)
                cell_set = set(cells)
                if cell_set & (existing_objects | bar_forbidden):
                    continue
                if _bar_intersections(cells, path_set) != {anchor}:
                    continue
                return DecoyPlan(color=int(color), cells=tuple(cells))
    return None


def _place_offpath_decoy(
    shape: tuple[int, int],
    color: int,
    existing_objects: set[Coord],
    bar_forbidden: set[Coord],
    path_set: set[Coord],
) -> DecoyPlan | None:
    h, w = map(int, shape)
    for _ in range(40):
        orientation = random.choice(["horizontal", "vertical"])
        if orientation == "horizontal":
            anchor = (random.randrange(h), random.randrange(w))
        else:
            anchor = (random.randrange(h), random.randrange(w))
        decoy = _place_decoy_through_path(
            shape,
            anchor,
            color,
            existing_objects,
            bar_forbidden,
            path_set=set(),
        )
        if decoy is None:
            continue
        if set(decoy.cells) & path_set:
            continue
        return decoy
    return None


def _render_grid(
    shape: tuple[int, int],
    legend_rows: list[tuple[int, int]],
    legend_corner: str,
    seed_cells: list[Coord],
    seed_color: int,
    mirrors: list[MirrorPlan],
    decoys: list[DecoyPlan],
) -> np.ndarray:
    grid = np.full(shape, int(Color.BLACK), dtype=int)
    _render_legend(grid, legend_rows, legend_corner)
    for r, c in seed_cells:
        grid[r, c] = int(seed_color)
    for mirror in mirrors:
        for r, c in mirror.cells:
            grid[r, c] = int(mirror.color)
    for decoy in decoys:
        for r, c in decoy.cells:
            grid[r, c] = int(decoy.color)
    return grid


def _expected_output(grid: np.ndarray, path_info: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c, color, _idx in path_info:
        if int(out[r, c]) == int(Color.BLACK):
            out[r, c] = int(color)
    return out


def _legend_rows_for_chain(chain: list[int], reflections: int) -> list[tuple[int, int]]:
    legend_rows = random.randint(max(2, reflections), min(4, reflections + 1))
    rows = [(int(chain[i]), int(chain[i + 1])) for i in range(reflections)]
    extra = legend_rows - reflections
    remaining = [c for c in GEN_COLORS if c not in chain]
    for _ in range(extra):
        left, right = random.sample(remaining, 2)
        remaining = [c for c in remaining if c not in {left, right}]
        rows.append((int(left), int(right)))
    return rows


def _build_decoys(
    shape: tuple[int, int],
    chain: list[int],
    path_info: list[tuple[int, int, int, int]],
    seed_cells: list[Coord],
    legend_box: BBox,
    mirrors: list[MirrorPlan],
) -> list[DecoyPlan] | None:
    path_set = {(int(r), int(c)) for r, c, _color, _idx in path_info}
    groups = _segment_groups(path_info)
    seed_guard = _seed_guard(shape, seed_cells)
    bar_forbidden = seed_guard | _expanded_cells(shape, legend_box, pad=1)
    existing_objects = set(seed_cells)
    for mirror in mirrors:
        existing_objects |= set(mirror.cells)

    decoys: list[DecoyPlan] = []
    chosen_indices: list[int] = []
    if 0 in groups:
        chosen_indices.append(0)
    if len(groups) > 1 and (len(chosen_indices) == 0 or random.random() < 0.8):
        last_idx = max(groups)
        if last_idx not in chosen_indices:
            chosen_indices.append(last_idx)

    target_decoys = random.choice(GEN_ONPATH_DECOYS)
    random.shuffle(chosen_indices)
    chosen_indices = chosen_indices[:target_decoys]
    if not chosen_indices:
        chosen_indices = [min(groups)]

    for idx in chosen_indices:
        cells = groups[idx]
        candidates = cells[1:-1] if len(cells) >= 4 else cells[:]
        random.shuffle(candidates)
        wrong_choices = [c for c in chain if c != chain[idx]]
        if idx + 1 < len(chain):
            wrong_choices = [chain[idx + 1]] + [c for c in wrong_choices if c != chain[idx + 1]]
        elif idx > 0:
            wrong_choices = [chain[idx - 1]] + [c for c in wrong_choices if c != chain[idx - 1]]

        placed = None
        for anchor in candidates:
            for wrong_color in wrong_choices:
                placed = _place_decoy_through_path(
                    shape,
                    anchor,
                    wrong_color,
                    existing_objects,
                    bar_forbidden,
                    path_set,
                )
                if placed is not None:
                    break
            if placed is not None:
                break
        if placed is None:
            return None
        decoys.append(placed)
        existing_objects |= set(placed.cells)

    for _ in range(random.choice(GEN_OFFPATH_DECOYS)):
        color_choices = [c for c in GEN_COLORS if c != chain[0]]
        color = random.choice(color_choices)
        decoy = _place_offpath_decoy(
            shape,
            color,
            existing_objects,
            bar_forbidden,
            path_set,
        )
        if decoy is not None:
            decoys.append(decoy)
            existing_objects |= set(decoy.cells)

    return decoys


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GEN_HEIGHT_RANGE),
        random.randint(*GEN_WIDTH_RANGE),
    )
    reflections = random.choices(GEN_REFLECTIONS, weights=GEN_REFLECTION_WEIGHTS, k=1)[0]
    chain = random.sample(GEN_COLORS, reflections + 1)
    legend_rows = _legend_rows_for_chain(chain, reflections)
    legend_corner = random.choice(GEN_CORNERS)

    blank = np.full(shape, int(Color.BLACK), dtype=int)
    legend_box = _render_legend(blank, legend_rows, legend_corner)
    blocked = _expanded_cells(shape, legend_box, pad=1)

    seed_len = random.choice(GEN_SEED_LENGTHS)
    starts = _candidate_start_states(shape, seed_len, blocked)
    if not starts:
        return None

    for _r0, _c0, dr, dc, seed_cells in starts[:20]:
        seed_barrier = _seed_guard(shape, seed_cells)
        route = _plan_route(
            shape,
            seed_cells[-1],
            dr,
            dc,
            chain,
            0,
            reflections,
            blocked | set(seed_cells),
            blocked | seed_barrier,
        )
        if route is None:
            continue

        path_info, mirrors = route
        if not (8 <= len(path_info) <= 24):
            continue

        decoys = _build_decoys(shape, chain, path_info, seed_cells, legend_box, mirrors)
        if decoys is None:
            continue

        grid = _render_grid(shape, legend_rows, legend_corner, seed_cells, chain[0], mirrors, decoys)
        expected = _expected_output(grid, path_info)

        transparent_hits = sum(1 for r, c, _color, _idx in path_info if int(grid[r, c]) != int(Color.BLACK))
        if transparent_hits < 1:
            continue

        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recoloring mirror-ray puzzle")
