from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, in_bounds


GREY = int(Color.GREY)
PALETTE = [int(c) for c in Color.NOT_BLACK if int(c) != GREY]
VECTOR_CHOICES = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-3, -2, -1, 1, 2, 3)
]
MAX_TRACE_STEPS = 5
GEN_RETRIES = 400


def _find_divider(g: np.ndarray, bg: int) -> tuple[np.ndarray, int, bool]:
    row_hits = [
        r
        for r in range(g.shape[0])
        if int(g[r, 0]) != bg and np.all(g[r, :] == g[r, 0])
    ]
    col_hits = [
        c
        for c in range(g.shape[1])
        if int(g[0, c]) != bg and np.all(g[:, c] == g[0, c])
    ]

    assert len(row_hits) + len(col_hits) == 1, "expected exactly one full divider line"
    if row_hits:
        return g, int(row_hits[0]), False
    return g.T.copy(), int(col_hits[0]), True


def _extract_mapping(canonical: np.ndarray, divider_r: int, bg: int) -> dict[int, tuple[int, int]]:
    legend = canonical[:divider_r, :]
    color_cells = np.argwhere((legend != bg) & (legend != GREY))
    mapping: dict[int, tuple[int, int]] = {}
    used_greys: set[tuple[int, int]] = set()

    for r, c in color_cells:
        r = int(r)
        c = int(c)
        color = int(legend[r, c])
        assert color not in mapping, "legend colors must be unique"
        band_r0 = max(0, r - 1)
        band_r1 = min(divider_r, r + 2)
        band = legend[band_r0:band_r1, :]
        grey_local = np.argwhere(band == GREY)
        assert len(grey_local) == 1, "each legend color must have one grey partner in its local band"
        gr = int(grey_local[0, 0] + band_r0)
        gc = int(grey_local[0, 1])
        assert (gr, gc) not in used_greys, "grey legend markers must not be shared"
        used_greys.add((gr, gc))
        mapping[color] = (gr - r, gc - c)

    assert mapping, "legend must define at least one color displacement"
    return mapping


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    canonical, divider_r, transposed = _find_divider(g, bg)
    mapping = _extract_mapping(canonical, divider_r, bg)
    legend_h = int(divider_r)
    sources = []
    walls = []
    for r in range(legend_h + 1, canonical.shape[0]):
        for c in range(canonical.shape[1]):
            color = int(canonical[r, c])
            if color == bg:
                continue
            if color in mapping:
                sources.append((r, c, color))
            else:
                walls.append((r, c, color))
    return {
        "grid": g,
        "canonical": canonical,
        "bg": bg,
        "divider_r": divider_r,
        "transposed": transposed,
        "mapping": mapping,
        "sources": sorted(sources),
        "walls": sorted(walls),
    }


def _simulate_canonical(parsed: dict) -> np.ndarray:
    canonical = parsed["canonical"]
    bg = int(parsed["bg"])
    mapping = parsed["mapping"]
    out = canonical.copy()

    for r, c, color in parsed["sources"]:
        dr, dc = mapping[color]
        cur_r, cur_c = int(r), int(c)
        while True:
            nr = cur_r + dr
            nc = cur_c + dc
            if not in_bounds(canonical.shape, nr, nc):
                break
            if int(canonical[nr, nc]) != bg:
                break
            prev = int(out[nr, nc])
            assert prev in (bg, color), "trace collisions between colors are not allowed"
            out[nr, nc] = color
            cur_r, cur_c = nr, nc

    if parsed["transposed"]:
        return out.T.copy()
    return out


def _trace_info(
    canonical: np.ndarray,
    bg: int,
    mapping: dict[int, tuple[int, int]],
    source: tuple[int, int, int],
) -> dict:
    r, c, color = source
    dr, dc = mapping[color]
    path: list[tuple[int, int]] = []
    cur_r, cur_c = int(r), int(c)
    while True:
        nr = cur_r + dr
        nc = cur_c + dc
        if not in_bounds(canonical.shape, nr, nc):
            return {
                "source": source,
                "vector": (dr, dc),
                "path": path,
                "stop": "boundary",
                "stop_cell": None,
            }
        if int(canonical[nr, nc]) != bg:
            stop_kind = "wall"
            if int(canonical[nr, nc]) in mapping:
                stop_kind = "object"
            return {
                "source": source,
                "vector": (dr, dc),
                "path": path,
                "stop": stop_kind,
                "stop_cell": (nr, nc),
            }
        path.append((nr, nc))
        cur_r, cur_c = nr, nc


def _scene_diagnostics(grid: np.ndarray) -> dict:
    parsed = _parse_scene(grid)
    trace_infos = [
        _trace_info(parsed["canonical"], parsed["bg"], parsed["mapping"], source)
        for source in parsed["sources"]
    ]
    used_colors = {color for _r, _c, color in parsed["sources"]}
    return {
        "parsed": parsed,
        "traces": trace_infos,
        "used_colors": used_colors,
        "unused_legend_colors": sorted(set(parsed["mapping"]) - used_colors),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the legend-defined color displacements and leave dotted jump traces for each main-region colored pixel.

    Input structure:
    - One full single-color divider line splits the canvas into a small legend region and a larger main region.
    - Each legend entry consists of one colored pixel and one grey pixel; the grey pixel's row/column offset from
      the colored pixel is that color's jump vector.
    - The main region contains colored singleton pixels using some legend colors, plus obstacle wall cells.

    Rule:
    - Infer the color-to-vector mapping from the legend alone; the mapping is not fixed across tasks.
    - For each colored pixel in the main region, repeatedly add its color's vector and paint every empty landing
      cell with that same color.
    - Stop before the first landing that would go out of bounds or land on any original foreground cell; keep the
      divider, legend, source pixels, and wall cells unchanged.

    Key insight:
    - The grey point defines a per-instance displacement relative to the nearby colored legend point, and jumps
      with length greater than one create a dotted trajectory rather than a filled line.
    """
    parsed = _parse_scene(grid)
    return _simulate_canonical(parsed)


def _sample_vectors(count: int, used_count: int) -> list[tuple[int, int]]:
    while True:
        vectors = random.sample(VECTOR_CHOICES, count)
        used_vectors = vectors[:used_count]
        has_diagonal = any(dr != 0 and dc != 0 for dr, dc in used_vectors)
        has_long = any(abs(dc) >= 2 for _dr, dc in used_vectors)
        if count == 1:
            if has_diagonal or has_long:
                return vectors
        elif has_diagonal and has_long:
            return vectors


def _trace_cells(src: tuple[int, int], vec: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = src
    dr, dc = vec
    return [(r + dr * i, c + dc * i) for i in range(1, steps + 1)]


def _valid_path(
    shape: tuple[int, int],
    main_r0: int,
    src: tuple[int, int],
    vec: tuple[int, int],
    steps: int,
    occupied: set[tuple[int, int]],
    reserved_traces: set[tuple[int, int]],
) -> bool:
    if not in_bounds(shape, src[0], src[1]):
        return False
    if src in occupied or src in reserved_traces:
        return False
    if src[0] < main_r0:
        return False
    for cell in _trace_cells(src, vec, steps):
        r, c = cell
        if not in_bounds(shape, r, c):
            return False
        if r < main_r0:
            return False
        if cell in occupied or cell in reserved_traces:
            return False
    return True


def _wall_candidates(
    shape: tuple[int, int],
    main_r0: int,
    mapping: dict[int, tuple[int, int]],
    colors: list[int],
    occupied: set[tuple[int, int]],
    reserved_traces: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    candidates = []
    for color in colors:
        vec = mapping[color]
        for steps in range(0, MAX_TRACE_STEPS + 1):
            for r in range(main_r0, h):
                for c in range(w):
                    src = (r, c)
                    if not _valid_path(shape, main_r0, src, vec, steps, occupied, reserved_traces):
                        continue
                    stop_r = r + vec[0] * (steps + 1)
                    stop_c = c + vec[1] * (steps + 1)
                    stop = (stop_r, stop_c)
                    if not in_bounds(shape, stop_r, stop_c):
                        continue
                    if stop_r < main_r0:
                        continue
                    if stop in occupied or stop in reserved_traces:
                        continue
                    candidates.append(
                        {
                            "color": color,
                            "src": src,
                            "steps": steps,
                            "trail": _trace_cells(src, vec, steps),
                            "stop": stop,
                            "vector": vec,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _object_candidates(
    shape: tuple[int, int],
    main_r0: int,
    mapping: dict[int, tuple[int, int]],
    colors: list[int],
    occupied: set[tuple[int, int]],
    reserved_traces: set[tuple[int, int]],
    blocker: tuple[int, int],
) -> list[dict]:
    candidates = []
    for color in colors:
        vec = mapping[color]
        for steps in range(0, MAX_TRACE_STEPS + 1):
            src = (
                blocker[0] - vec[0] * (steps + 1),
                blocker[1] - vec[1] * (steps + 1),
            )
            if not _valid_path(shape, main_r0, src, vec, steps, occupied, reserved_traces):
                continue
            if src == blocker:
                continue
            candidates.append(
                {
                    "color": color,
                    "src": src,
                    "steps": steps,
                    "trail": _trace_cells(src, vec, steps),
                    "stop": blocker,
                    "vector": vec,
                }
            )
    random.shuffle(candidates)
    return candidates


def _boundary_candidates(
    shape: tuple[int, int],
    main_r0: int,
    mapping: dict[int, tuple[int, int]],
    colors: list[int],
    occupied: set[tuple[int, int]],
    reserved_traces: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    candidates = []
    for color in colors:
        vec = mapping[color]
        for steps in range(1, MAX_TRACE_STEPS + 1):
            for r in range(main_r0, h):
                for c in range(w):
                    src = (r, c)
                    if not _valid_path(shape, main_r0, src, vec, steps, occupied, reserved_traces):
                        continue
                    stop_r = r + vec[0] * (steps + 1)
                    stop_c = c + vec[1] * (steps + 1)
                    if in_bounds(shape, stop_r, stop_c):
                        continue
                    candidates.append(
                        {
                            "color": color,
                            "src": src,
                            "steps": steps,
                            "trail": _trace_cells(src, vec, steps),
                            "stop": None,
                            "vector": vec,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _add_singletons(
    grid: np.ndarray,
    colors: list[int],
    mapping: dict[int, tuple[int, int]],
    main_r0: int,
    occupied: set[tuple[int, int]],
    reserved_traces: set[tuple[int, int]],
    want: int,
) -> bool:
    colors = colors[:]
    random.shuffle(colors)
    added = 0
    shape = grid.shape
    for color in colors:
        all_boundary = _boundary_candidates(shape, main_r0, mapping, [color], occupied, reserved_traces)
        if not all_boundary:
            continue
        cand = all_boundary[0]
        sr, sc = cand["src"]
        grid[sr, sc] = color
        occupied.add((sr, sc))
        reserved_traces.update(cand["trail"])
        added += 1
        if added >= want:
            return True
    return added >= want


def _build_attempt() -> np.ndarray | None:
    used_count = random.randint(2, 4)
    extra_count = random.randint(0, min(2, 5 - used_count))
    legend_count = used_count + extra_count

    all_colors = random.sample(PALETTE, legend_count + 2)
    divider_color = all_colors[0]
    wall_color = all_colors[1]
    legend_colors = all_colors[2:]
    used_colors = legend_colors[:used_count]
    vectors = _sample_vectors(legend_count, used_count)
    mapping = {color: vec for color, vec in zip(legend_colors, vectors)}

    legend_h = 4 * legend_count - 1
    main_h_min = 8
    main_h_max = min(14, 30 - legend_h - 1)
    if main_h_max < main_h_min:
        return None
    main_h = random.randint(main_h_min, main_h_max)
    width = random.randint(14, 24)
    height = legend_h + 1 + main_h

    grid = np.zeros((height, width), dtype=int)
    divider_r = legend_h
    grid[divider_r, :] = divider_color

    anchor_cols = [random.randint(3, width - 4) for _ in range(legend_count)]
    for i, (color, vec) in enumerate(zip(legend_colors, vectors)):
        r = 4 * i + 1
        c = anchor_cols[i]
        dr, dc = vec
        grid[r, c] = color
        grid[r + dr, c + dc] = GREY

    main_r0 = divider_r + 1
    occupied = {tuple(map(int, rc)) for rc in np.argwhere(grid != 0)}
    reserved_traces: set[tuple[int, int]] = set()

    wall_plans = _wall_candidates(grid.shape, main_r0, mapping, used_colors, occupied, reserved_traces)
    for wall_plan in wall_plans[:120]:
        temp_occ = set(occupied)
        temp_traces = set(reserved_traces)
        blocker_src = wall_plan["src"]
        wall_stop = wall_plan["stop"]
        temp_occ.add(blocker_src)
        temp_occ.add(wall_stop)
        temp_traces.update(wall_plan["trail"])

        object_plans = _object_candidates(
            grid.shape,
            main_r0,
            mapping,
            used_colors,
            temp_occ,
            temp_traces,
            blocker_src,
        )
        if not object_plans:
            continue

        object_plan = object_plans[0]
        temp_occ2 = set(temp_occ)
        temp_traces2 = set(temp_traces)
        temp_occ2.add(object_plan["src"])
        temp_traces2.update(object_plan["trail"])

        boundary_plans = _boundary_candidates(
            grid.shape,
            main_r0,
            mapping,
            used_colors,
            temp_occ2,
            temp_traces2,
        )
        if not boundary_plans:
            continue

        boundary_plan = boundary_plans[0]
        plans = [wall_plan, object_plan, boundary_plan]
        break
    else:
        return None

    for plan in plans:
        sr, sc = plan["src"]
        grid[sr, sc] = plan["color"]
        occupied.add((sr, sc))
        reserved_traces.update(plan["trail"])
    wr, wc = wall_plan["stop"]
    grid[wr, wc] = wall_color
    occupied.add((wr, wc))

    if random.random() < 0.75:
        extras_wanted = random.randint(1, min(used_count, 2))
        _add_singletons(grid, used_colors, mapping, main_r0, occupied, reserved_traces, extras_wanted)

    if random.random() < 0.6:
        extra_wall_options = []
        for r in range(main_r0, height):
            for c in range(width):
                if grid[r, c] != 0:
                    continue
                if (r, c) in reserved_traces:
                    continue
                extra_wall_options.append((r, c))
        random.shuffle(extra_wall_options)
        for r, c in extra_wall_options[:2]:
            grid[r, c] = wall_color

    if random.random() < 0.5:
        grid = grid.T.copy()

    diagnostics = _scene_diagnostics(grid)
    traces = diagnostics["traces"]
    stop_kinds = {item["stop"] for item in traces}
    if not {"boundary", "wall", "object"} <= stop_kinds:
        return None
    if not any(item["path"] for item in traces):
        return None
    used_vectors = [diagnostics["parsed"]["mapping"][color] for color in diagnostics["used_colors"]]
    if not any(abs(dc) >= 2 for _dr, dc in used_vectors):
        return None
    if not any(dr != 0 and dc != 0 for dr, dc in used_vectors):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-jump input")
