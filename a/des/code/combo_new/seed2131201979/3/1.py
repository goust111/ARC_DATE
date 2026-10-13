import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
REGION_COLORS = (1, 2, 3, 4, 6)
MARKER_COLORS = (7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_TO_DIR = {"top": 2, "right": 3, "bottom": 0, "left": 1}
GEN_RETRIES = 240
RAY_RETRIES = 48
SEGMENT_RETRIES = 40


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _step(pos: tuple[int, int], dir_idx: int, dist: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return pos[0] + dr * dist, pos[1] + dc * dist


def _cw(dir_idx: int) -> int:
    return (dir_idx + 1) % 4


def _left_of(dir_idx: int) -> int:
    return (dir_idx - 1) % 4


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _grey_components(grid: np.ndarray) -> tuple[list[set[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps = _components(grid == GREY)
    indexed: list[set[tuple[int, int]]] = []
    lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        cell_set = set(comp)
        indexed.append(cell_set)
        for cell in comp:
            lookup[cell] = idx
    return indexed, lookup


def _find_regions(
    grid: np.ndarray,
) -> tuple[list[int], dict[tuple[int, int], int]]:
    h, w = grid.shape
    grey_comps, grey_lookup = _grey_components(grid)
    fg_mask = (grid != BLACK) & (grid != GREY)
    seen = np.zeros((h, w), dtype=bool)
    region_colors: list[int] = []
    boundary_to_region: dict[tuple[int, int], int] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not fg_mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg_mask[nr, nc] and not seen[nr, nc]:
                        assert int(grid[nr, nc]) == color, "each region must be monochrome"
                        seen[nr, nc] = True
                        q.append((nr, nc))

            grey_ids = {
                grey_lookup[(nr, nc)]
                for rr, cc in cells
                for nr, nc in _neighbors4(rr, cc)
                if (nr, nc) in grey_lookup
            }
            assert len(grey_ids) == 1, "each region must have exactly one enclosing grey loop"
            grey_id = next(iter(grey_ids))
            grey_loop = grey_comps[grey_id]
            assert all(0 < rr < h - 1 and 0 < cc < w - 1 for rr, cc in grey_loop), "region loops must be internal"

            region_id = len(region_colors)
            region_colors.append(color)
            for cell in grey_loop:
                assert cell not in boundary_to_region, "grey loops must belong to one region only"
                boundary_to_region[cell] = region_id

    assert region_colors, "expected at least one internal region"
    return region_colors, boundary_to_region


def _marker_sort_key(marker: dict[str, object]) -> tuple[int, int]:
    r, c = marker["pos"]
    side = marker["side"]
    if side == "top":
        return 0, c
    if side == "right":
        return 1, r
    if side == "bottom":
        return 2, -c
    return 3, -r


def _find_markers(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    markers: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            on_border = r in {0, h - 1} or c in {0, w - 1}
            if not on_border:
                continue
            value = int(grid[r, c])
            assert value != BLACK, "the outer frame must contain no black cells"
            if value == GREY:
                continue
            assert (r in {0, h - 1}) ^ (c in {0, w - 1}), "markers must not be placed on corners"
            if r == 0:
                side = "top"
            elif c == w - 1:
                side = "right"
            elif r == h - 1:
                side = "bottom"
            else:
                side = "left"
            markers.append(
                {
                    "side": side,
                    "pos": (r, c),
                    "dir": SIDE_TO_DIR[side],
                    "color": value,
                }
            )

    markers.sort(key=_marker_sort_key)
    assert 1 <= len(markers) <= len(MARKER_COLORS), "expected one to three border markers"
    return markers


def _trace_ray(
    grid: np.ndarray,
    out: np.ndarray,
    marker: dict[str, object],
    boundary_to_region: dict[tuple[int, int], int],
    region_colors: list[int],
) -> None:
    h, w = grid.shape
    r, c = marker["pos"]
    dir_idx = int(marker["dir"])
    color = int(marker["color"])
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dir_idx, color)
        assert state not in seen_states, "ray routing must terminate"
        seen_states.add(state)

        nr, nc = _step((r, c), dir_idx)
        if not _in_bounds((h, w), nr, nc):
            return

        cell = int(grid[nr, nc])
        if cell == BLACK:
            out[nr, nc] = color
            r, c = nr, nc
            continue

        region_id = boundary_to_region.get((nr, nc))
        if cell == GREY and region_id is not None:
            dir_idx = _cw(dir_idx)
            while True:
                sr, sc = _step((r, c), dir_idx)
                if not _in_bounds((h, w), sr, sc) or int(grid[sr, sc]) != BLACK:
                    break
                lr, lc = _step((sr, sc), _left_of(dir_idx))
                if boundary_to_region.get((lr, lc)) != region_id:
                    break
                out[sr, sc] = color
                r, c = sr, sc
            color = region_colors[region_id]
            continue

        return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays from colored frame gaps, bend them clockwise along grey-bounded monochrome regions, and recolor the departing beam with each touched region.

    Input structure:
    - The background is black and the whole canvas has a one-pixel grey outer frame.
    - Some non-corner frame cells are replaced by single colored markers; each such border gap emits inward from its side.
    - Inside the frame, one or more monochrome colored regions are enclosed by separate one-pixel grey loops.

    Rule:
    - Keep the outer frame, grey loops, marker pixels, and region fills unchanged.
    - From each marker, extend a one-pixel ray through black cells in the inward direction using the marker color.
    - When the next step would hit a grey loop, turn clockwise and slide along the contacted face while the region stays on the ray's left side.
    - After the ray leaves that boundary face, recolor subsequent black cells with the enclosed region's fill color and continue until another region or the frame stops it.

    Key insight:
    - The grey loop is both obstacle and guide: each contact induces a clockwise edge-following bend, and the first free segment after that bend inherits the touched region's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    region_colors, boundary_to_region = _find_regions(g)
    markers = _find_markers(g)

    out = g.copy()
    for marker in markers:
        _trace_ray(g, out, marker, boundary_to_region, region_colors)
    return out


def _distance_to_frame(pos: tuple[int, int], dir_idx: int, h: int, w: int) -> int:
    rr, cc = pos
    dr, dc = DIRS[dir_idx]
    count = 0
    while True:
        rr += dr
        cc += dc
        if not (0 < rr < h - 1 and 0 < cc < w - 1):
            return count
        count += 1


def _marker_pos(side: str, offset: int, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, offset
    if side == "bottom":
        return h - 1, offset
    if side == "left":
        return offset, 0
    return offset, w - 1


def _rect_from_hit(
    hit: tuple[int, int], in_dir: int, face_len: int, depth: int
) -> tuple[int, int, int, int]:
    r, c = hit
    if in_dir == 1:
        return r, c, r + face_len - 1, c + depth - 1
    if in_dir == 2:
        return r, c - face_len + 1, r + depth - 1, c
    if in_dir == 3:
        return r - face_len + 1, c - depth + 1, r, c
    return r - depth + 1, c, r, c + face_len - 1


def _rect_border(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _rect_fill(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _expand(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h - 1, r + pad) + 1):
            for cc in range(max(0, c - pad), min(w - 1, c + pad) + 1):
                expanded.add((rr, cc))
    return expanded


def _safe_inner_rect(rect: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = rect
    return 2 <= r0 < r1 <= h - 3 and 2 <= c0 < c1 <= w - 3


def _build_ray_candidate(
    h: int,
    w: int,
    side: str,
    marker_color: int,
    region_colors: list[int],
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> dict[str, object] | None:
    offsets = list(range(3, (w - 3) if side in {"top", "bottom"} else (h - 3)))
    rng.shuffle(offsets)

    for offset in offsets:
        start = _marker_pos(side, offset, h, w)
        if start in blocked:
            continue

        pos = start
        dir_idx = SIDE_TO_DIR[side]
        path: list[tuple[int, int]] = []
        path_set: set[tuple[int, int]] = set()
        solid_set: set[tuple[int, int]] = set()
        regions: list[dict[str, object]] = []
        ok = True

        for region_color in region_colors:
            built = False
            for _ in range(SEGMENT_RETRIES):
                run_cap = _distance_to_frame(pos, dir_idx, h, w)
                if run_cap < 2:
                    break
                min_pre = 2 if pos == start else 1
                if run_cap - 1 < min_pre:
                    break
                pre_len = rng.randint(min_pre, min(6, run_cap - 1))
                pre_cells = [_step(pos, dir_idx, k) for k in range(1, pre_len + 1)]
                hit = _step(pre_cells[-1], dir_idx)

                face_len = rng.randint(3, 5)
                depth = rng.randint(3, 4)
                rect = _rect_from_hit(hit, dir_idx, face_len, depth)
                if not _safe_inner_rect(rect, h, w):
                    continue

                turn_dir = _cw(dir_idx)
                mandatory = [_step(pre_cells[-1], turn_dir, k) for k in range(1, face_len + 1)]
                if not all(0 < r < h - 1 and 0 < c < w - 1 for r, c in pre_cells + mandatory):
                    continue

                border_cells = _rect_border(rect)
                fill_cells = _rect_fill(rect)
                solid_cells = border_cells | fill_cells

                candidate_cells = set(pre_cells) | set(mandatory) | solid_cells
                if candidate_cells & blocked:
                    continue
                if candidate_cells & path_set:
                    continue
                if candidate_cells & solid_set:
                    continue
                if (set(pre_cells) | set(mandatory)) & solid_cells:
                    continue

                next_pos = mandatory[-1]
                remaining_regions = len(region_colors) - len(regions) - 1
                if remaining_regions > 0 and _distance_to_frame(next_pos, turn_dir, h, w) < 2:
                    continue

                path.extend(pre_cells)
                path.extend(mandatory)
                path_set.update(pre_cells)
                path_set.update(mandatory)
                solid_set.update(solid_cells)
                regions.append({"rect": rect, "color": region_color})
                pos = next_pos
                dir_idx = turn_dir
                built = True
                break

            if not built:
                ok = False
                break

        if not ok:
            continue

        tail_len = _distance_to_frame(pos, dir_idx, h, w)
        if tail_len < 1:
            continue
        tail = [_step(pos, dir_idx, k) for k in range(1, tail_len + 1)]
        tail_set = set(tail)
        if tail_set & blocked or tail_set & path_set or tail_set & solid_set:
            continue

        path.extend(tail)
        path_set.update(tail)
        footprint = set(path_set) | set(solid_set) | {start}
        return {
            "marker": {"pos": start, "color": marker_color},
            "regions": regions,
            "path": path,
            "path_cells": set(path_set),
            "solid_cells": set(solid_set),
            "footprint": footprint,
        }

    return None


def _render_scene(h: int, w: int, rays: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    for ray in rays:
        for region in ray["regions"]:
            rect = region["rect"]
            color = int(region["color"])
            for r, c in _rect_border(rect):
                grid[r, c] = GREY
            for r, c in _rect_fill(rect):
                grid[r, c] = color
        mr, mc = ray["marker"]["pos"]
        grid[mr, mc] = int(ray["marker"]["color"])

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "the transformation must draw at least one ray"

    marker_colors = [int(marker["color"]) for marker in _find_markers(grid)]
    assert len(marker_colors) >= 2
    changed = int(np.count_nonzero(out != grid))
    assert changed >= 8, "output should contain a visible ray trace"


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    ray_count = rng.choice((2, 3))
    sides = rng.sample(["top", "right", "bottom", "left"], ray_count)
    turn_counts = rng.sample([0, 1, 2], ray_count)
    marker_colors = rng.sample(MARKER_COLORS, ray_count)
    region_color_pool = iter(rng.sample(REGION_COLORS, sum(turn_counts)))

    rays: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    solid_halo: set[tuple[int, int]] = set()

    for side, marker_color, turn_count in zip(sides, marker_colors, turn_counts):
        ray_region_colors = [next(region_color_pool) for _ in range(turn_count)]
        candidate = None
        for _ in range(RAY_RETRIES):
            candidate = _build_ray_candidate(h, w, side, marker_color, ray_region_colors, occupied | solid_halo, rng)
            if candidate is not None:
                break
        if candidate is None:
            return None
        rays.append(candidate)
        occupied |= candidate["footprint"]
        solid_halo |= _expand(candidate["solid_cells"], h, w, pad=1)

    grid = _render_scene(h, w, rays)
    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-refraction ARC input")
