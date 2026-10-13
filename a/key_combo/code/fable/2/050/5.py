from __future__ import annotations

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
import itertools
import random
from pathlib import Path

import numpy as np


def _fallback_get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


try:
    _ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    _ARC_UTILS_SPEC = spec_from_file_location("_arc_utils_fable_2_050_5", _ARC_UTILS_PATH)
    if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
        raise ImportError("cannot build arc_utils spec")
    _ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
    _ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)
    get_bg = _ARC_UTILS.get_bg
except Exception:
    get_bg = _fallback_get_bg


DIRS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}

BG_COLOR = 0
SEPARATOR_CHOICES = (5, 8, 9)
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CELL_SIZE_CHOICES = (3, 5)
SCENARIO_WEIGHTS = (("mixed", 0.45), ("single_bounce", 0.30), ("multi_bounce", 0.25))
GEN_RETRIES = 300
PLACEMENT_RETRIES = 80
MAX_ARROWS = 4


def _mode(values) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _separator_candidates(grid: np.ndarray) -> list[tuple[int, int, list[int], list[int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = []
    for color in map(int, np.unique(g)):
        row_lines = [r for r in range(h) if np.all(g[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue
        row_steps = {b - a for a, b in zip(row_lines, row_lines[1:])}
        col_steps = {b - a for a, b in zip(col_lines, col_lines[1:])}
        if len(row_steps) != 1 or row_steps != col_steps:
            continue
        pitch = next(iter(row_steps))
        if pitch < 3:
            continue
        if row_lines[0] != 0 or row_lines[-1] != h - 1:
            continue
        if col_lines[0] != 0 or col_lines[-1] != w - 1:
            continue
        if (h - 1) % pitch != 0 or (w - 1) % pitch != 0:
            continue
        out.append((color, pitch, row_lines, col_lines))
    return out


def _glyph_offsets(cell_size: int) -> dict[str, frozenset[tuple[int, int]]]:
    s = int(cell_size)
    return {
        "NW": frozenset({(0, 0), (0, 1), (1, 0)}),
        "NE": frozenset({(0, s - 2), (0, s - 1), (1, s - 1)}),
        "SW": frozenset({(s - 2, 0), (s - 1, 0), (s - 1, 1)}),
        "SE": frozenset({(s - 2, s - 1), (s - 1, s - 2), (s - 1, s - 1)}),
    }


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    candidates = _separator_candidates(g)
    assert len(candidates) == 1, f"expected one separator layout, found {len(candidates)}"
    sep_color, pitch, row_lines, col_lines = candidates[0]
    cell_size = pitch - 1
    assert cell_size >= 3 and cell_size % 2 == 1, "cell interiors must have an odd side length >= 3"

    interior_mask = np.ones_like(g, dtype=bool)
    interior_mask[row_lines, :] = False
    interior_mask[:, col_lines] = False
    bg = _mode(g[interior_mask].ravel())
    glyphs = _glyph_offsets(cell_size)

    arrows = {}
    for rr in range(len(row_lines) - 1):
        r0 = row_lines[rr] + 1
        r1 = row_lines[rr + 1]
        for cc in range(len(col_lines) - 1):
            c0 = col_lines[cc] + 1
            c1 = col_lines[cc + 1]
            patch = g[r0:r1, c0:c1]
            coords = np.argwhere(patch != bg)
            if coords.size == 0:
                continue
            colors = {int(v) for v in patch[patch != bg]}
            assert len(colors) == 1, f"cell {(rr, cc)} contains multiple arrow colors"
            offsets = frozenset((int(r), int(c)) for r, c in coords)
            matches = [name for name, pat in glyphs.items() if pat == offsets]
            assert len(matches) == 1, f"cell {(rr, cc)} does not contain a valid arrow glyph"
            arrows[(rr, cc)] = {"dir": matches[0], "color": next(iter(colors))}

    assert len(arrows) >= 2, "expected at least two arrows"
    return {
        "grid": g,
        "h": h,
        "w": w,
        "bg": bg,
        "sep_color": sep_color,
        "cell_size": cell_size,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "nrows": len(row_lines) - 1,
        "ncols": len(col_lines) - 1,
        "arrows": arrows,
    }


def _reflect_step(r: int, c: int, dr: int, dc: int, nrows: int, ncols: int) -> tuple[int, int, int, int, bool]:
    ndr, ndc = int(dr), int(dc)
    bounced = False
    if not (0 <= r + ndr < nrows):
        ndr = -ndr
        bounced = True
    if not (0 <= c + ndc < ncols):
        ndc = -ndc
        bounced = True
    return r + ndr, c + ndc, ndr, ndc, bounced


def _trace_ray(
    start: tuple[int, int],
    direction: str,
    arrow_cells: set[tuple[int, int]],
    nrows: int,
    ncols: int,
) -> dict | None:
    r, c = start
    dr, dc = DIRS[direction]
    seen = {(r, c, dr, dc)}
    path = []
    bounces = 0

    for _ in range(8 * nrows * ncols):
        nr, nc, dr, dc, bounced = _reflect_step(r, c, dr, dc, nrows, ncols)
        if bounced:
            bounces += 1
        if (nr, nc) in arrow_cells:
            if (nr, nc) == start:
                return None
            return {"target": (nr, nc), "path": path, "bounces": bounces}
        path.append((nr, nc))
        r, c = nr, nc
        state = (r, c, dr, dc)
        if state in seen:
            return None
        seen.add(state)
    return None


def _solve_cell_level(nrows: int, ncols: int, arrows: dict[tuple[int, int], dict], *, require_disjoint: bool = True) -> dict | None:
    arrow_cells = set(arrows)
    traces = {}
    used_path_cells = set()
    for cell, info in sorted(arrows.items()):
        trace = _trace_ray(cell, info["dir"], arrow_cells, nrows, ncols)
        if trace is None or not trace["path"]:
            return None
        if len(set(trace["path"])) != len(trace["path"]):
            return None
        overlap = used_path_cells.intersection(trace["path"])
        if require_disjoint and overlap:
            return None
        used_path_cells.update(trace["path"])
        traces[cell] = trace
    return traces


def _cell_center(layout: dict, cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    cell_size = layout["cell_size"]
    rr = layout["row_lines"][r] + 1 + cell_size // 2
    cc = layout["col_lines"][c] + 1 + cell_size // 2
    return rr, cc


def _paint_output(layout: dict, traces: dict) -> np.ndarray:
    out = layout["grid"].copy()
    bg = layout["bg"]
    for cell, info in layout["arrows"].items():
        color = int(info["color"])
        for ray_cell in traces[cell]["path"]:
            rr, cc = _cell_center(layout, ray_cell)
            assert int(out[rr, cc]) in {bg, color}, "ray lighting must only touch empty cell centers"
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each separator-bounded cell as one coarse pixel and extend each diagonal arrow ray with mirror bounces until it enters another arrow cell.

    Input structure:
    - The grid is partitioned by one-color separator lines into equal square cells.
    - Some cells contain a monochromatic corner-arrow glyph indicating one of the four diagonal directions; other cells are empty.
    - Empty cells share one background color distinct from the separator and the arrow colors.

    Rule:
    - Recover the cell lattice and read each arrow orientation from its corner glyph.
    - In cell coordinates, each arrow advances one cell per step along its diagonal direction; if the next step would leave the cell array, reflect the offending direction component(s) at the boundary and continue.
    - Stop a ray as soon as it would enter another arrow cell.
    - For every traversed empty cell, paint only that cell's center pixel with the emitting arrow's color, leaving the separator lines and arrow glyphs unchanged.

    Key insight:
    - The meaningful geometry lives on the coarse cell lattice rather than on raw pixels: the solver must infer the cell grid, decode the arrow directions, and trace reflected diagonal paths until the next arrow blocks the ray.
    """
    layout = _parse_grid(np.asarray(grid, dtype=int))
    traces = _solve_cell_level(layout["nrows"], layout["ncols"], layout["arrows"], require_disjoint=False)
    assert traces is not None, "invalid reflected-ray configuration"
    return _paint_output(layout, traces)


def _scenario_ok(traces: dict, scenario: str) -> bool:
    bounce_counts = [int(info["bounces"]) for info in traces.values()]
    if not any(b == 0 for b in bounce_counts):
        return False
    if not any(b > 0 for b in bounce_counts):
        return False
    if scenario == "single_bounce" and not any(b == 1 for b in bounce_counts):
        return False
    if scenario == "multi_bounce" and not any(b >= 2 for b in bounce_counts):
        return False
    return True


def _render_input(
    nrows: int,
    ncols: int,
    cell_size: int,
    bg: int,
    sep_color: int,
    arrows: dict[tuple[int, int], dict],
) -> np.ndarray:
    h = nrows * cell_size + (nrows + 1)
    w = ncols * cell_size + (ncols + 1)
    grid = np.full((h, w), sep_color, dtype=int)
    glyphs = _glyph_offsets(cell_size)
    pitch = cell_size + 1

    for r in range(nrows):
        for c in range(ncols):
            r0 = r * pitch + 1
            c0 = c * pitch + 1
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = bg

    for (r, c), info in arrows.items():
        r0 = r * pitch + 1
        c0 = c * pitch + 1
        for dr, dc in glyphs[info["dir"]]:
            grid[r0 + dr, c0 + dc] = int(info["color"])

    return grid


def _choose_scenario() -> str:
    labels = [label for label, _ in SCENARIO_WEIGHTS]
    weights = [weight for _, weight in SCENARIO_WEIGHTS]
    return random.choices(labels, weights=weights, k=1)[0]


def _valid_positions(nrows: int, ncols: int, parity: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(nrows) for c in range(ncols) if (r + c) % 2 == parity]


def _pick_cells(nrows: int, ncols: int, count: int) -> list[tuple[int, int]] | None:
    parity = random.randint(0, 1)
    positions = _valid_positions(nrows, ncols, parity)
    if len(positions) < count:
        return None

    for _ in range(PLACEMENT_RETRIES):
        cells = random.sample(positions, count)
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 2:
            continue
        return cells
    return None


def _candidate_counts(nrows: int, ncols: int, cell_size: int) -> list[int]:
    upper = 4 if cell_size == 5 else MAX_ARROWS
    upper = min(upper, max(3, (nrows * ncols) // 3))
    return [count for count in (4, 3) if count <= upper]


def _find_arrow_layout(nrows: int, ncols: int, cell_size: int, scenario: str) -> tuple[dict, dict] | None:
    counts = _candidate_counts(nrows, ncols, cell_size)
    for _ in range(PLACEMENT_RETRIES):
        count = random.choice(counts)
        cells = _pick_cells(nrows, ncols, count)
        if cells is None:
            continue

        solutions = []
        for dirs in itertools.product(DIRS.keys(), repeat=count):
            arrows = {cells[i]: {"dir": dirs[i]} for i in range(count)}
            if len({info["dir"] for info in arrows.values()}) < 2:
                continue
            traces = _solve_cell_level(nrows, ncols, arrows, require_disjoint=True)
            if traces is None:
                continue
            if not _scenario_ok(traces, scenario):
                continue
            total_path = sum(len(info["path"]) for info in traces.values())
            if total_path < count + 2:
                continue
            if total_path > max(6, int(0.45 * (nrows * ncols - count))):
                continue
            targets = {tuple(info["target"]) for info in traces.values()}
            if len(targets) < 2:
                continue
            solutions.append((arrows, traces))

        if not solutions:
            continue

        arrows, traces = random.choice(solutions)
        colors = random.sample(ARROW_COLORS, len(arrows))
        colored = {}
        for idx, cell in enumerate(sorted(arrows)):
            colored[cell] = {"dir": arrows[cell]["dir"], "color": colors[idx]}
        return colored, traces
    return None


def _validate_instance(
    grid: np.ndarray,
    nrows: int,
    ncols: int,
    cell_size: int,
    bg: int,
    sep_color: int,
    arrows: dict[tuple[int, int], dict],
    traces: dict,
) -> bool:
    parsed = _parse_grid(grid)
    if parsed["nrows"] != nrows or parsed["ncols"] != ncols or parsed["cell_size"] != cell_size:
        return False
    if parsed["bg"] != bg or parsed["sep_color"] != sep_color:
        return False
    if parsed["arrows"] != arrows:
        return False

    parsed_traces = _solve_cell_level(parsed["nrows"], parsed["ncols"], parsed["arrows"], require_disjoint=True)
    if parsed_traces is None or parsed_traces != traces:
        return False

    out = main(grid)
    expected = _paint_output(parsed, parsed_traces)
    if not np.array_equal(out, expected):
        return False

    changed = np.argwhere(out != grid)
    if changed.size == 0:
        return False
    if np.any(out[grid == sep_color] != sep_color):
        return False

    bounce_counts = [int(info["bounces"]) for info in traces.values()]
    if not any(b == 0 for b in bounce_counts) or not any(b > 0 for b in bounce_counts):
        return False

    direct_only = True
    for cell, info in arrows.items():
        dr, dc = DIRS[info["dir"]]
        r, c = cell
        nr, nc = r + dr, c + dc
        if not (0 <= nr < nrows and 0 <= nc < ncols and (nr, nc) == traces[cell]["target"]):
            direct_only = False
            break
    if direct_only:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZE_CHOICES)
    max_dim = 7 if cell_size == 3 else 4
    nrows = random.randint(4, max_dim)
    ncols = random.randint(4, max_dim)
    scenario = _choose_scenario()
    layout = _find_arrow_layout(nrows, ncols, cell_size, scenario)
    if layout is None:
        return None
    arrows, traces = layout

    sep_color = random.choice(SEPARATOR_CHOICES)
    bg = BG_COLOR
    if sep_color == bg:
        return None

    grid = _render_input(nrows, ncols, cell_size, bg, sep_color, arrows)
    if max(grid.shape) > 30:
        return None
    if not _validate_instance(grid, nrows, ncols, cell_size, bg, sep_color, arrows, traces):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid reflected-ray input after {GEN_RETRIES} attempts")
