from __future__ import annotations

from collections import Counter, deque
import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_shared_arc_utils():
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in [here.parent, *here.parents]:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            candidates.append(candidate.resolve())
    absolute = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if absolute.is_file():
        candidates.append(absolute.resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        spec = importlib.util.spec_from_file_location("_shared_arc_utils", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


_ARC_UTILS = _load_shared_arc_utils()

if _ARC_UTILS is not None:
    RED = int(_ARC_UTILS.Color.RED)
    ARC_COLORS = tuple(int(c) for c in _ARC_UTILS.Color.ALL_COLORS[:10])
    _shared_neighbors4 = _ARC_UTILS.neighbors4
else:
    RED = 2
    ARC_COLORS = tuple(range(10))
    _shared_neighbors4 = None


def _neighbors4(r: int, c: int):
    if _shared_neighbors4 is not None:
        yield from _shared_neighbors4(int(r), int(c))
        return
    rr = int(r)
    cc = int(c)
    yield rr - 1, cc
    yield rr + 1, cc
    yield rr, cc - 1
    yield rr, cc + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        coords: list[tuple[int, int]] = []
        while queue:
            cr, cc2 = queue.popleft()
            coords.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(coords)
    return components


def _touches_boundary(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _solid_monochrome(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    color = int(grid[coords[0]])
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return np.all(patch == color) and len(coords) == patch.size


def _most_common_non_red(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != RED)
    if not counts:
        raise ValueError("grid must contain a non-red background")
    return counts.most_common(1)[0][0]


def _find_divider_row(grid: np.ndarray) -> int:
    candidates = [
        r
        for r in range(grid.shape[0] - 1)
        if np.all(grid[r, :] == RED) and np.any(grid[r + 1 :, :] != RED)
    ]
    if not candidates:
        raise ValueError("could not locate the legend divider row")
    return max(candidates)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    divider_row = _find_divider_row(g)
    body = g[: divider_row + 1]
    legend = g[divider_row + 1 :]
    if legend.size == 0:
        raise ValueError("legend area must be non-empty")

    bg = _most_common_non_red(g)

    region_components = [
        coords
        for coords in _connected_components(body == bg)
        if not _touches_boundary(coords, body.shape)
    ]
    if not region_components:
        raise ValueError("no enclosed regions found")

    regions = []
    area_set: set[int] = set()
    for coords in region_components:
        area = len(coords)
        if area in area_set:
            raise ValueError("region areas must be unique")
        area_set.add(area)
        regions.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "area": area,
            }
        )

    bars: list[dict] = []
    used_colors: set[int] = set()
    used_heights: set[int] = set()
    for color in sorted(set(int(v) for v in legend.ravel()) - {bg, RED}):
        for coords in _connected_components(legend == color):
            if not coords:
                continue
            if not _solid_monochrome(legend, coords):
                raise ValueError("legend bars must be solid monochrome rectangles")
            r0, c0, r1, c1 = _bbox(coords)
            if r1 != legend.shape[0] - 1:
                raise ValueError("legend bars must stand on the bottom edge")
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if color in used_colors:
                raise ValueError("legend bar colors must be unique")
            if height in used_heights:
                raise ValueError("legend bar heights must be unique")
            used_colors.add(color)
            used_heights.add(height)
            bars.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "height": height,
                    "width": width,
                }
            )

    if len(bars) <= len(regions):
        raise ValueError("legend must contain more bars than enclosed regions")

    bar_by_height = {bar["height"]: bar for bar in bars}
    if len(bar_by_height) != len(bars):
        raise ValueError("bar heights must be unique")

    for region in regions:
        if region["area"] not in bar_by_height:
            raise ValueError("every region area must match a unique legend bar height")

    unused_bars = [bar for bar in bars if bar["height"] not in area_set]
    if not unused_bars:
        raise ValueError("legend must include at least one unused decoy bar")

    return {
        "grid": g,
        "bg": bg,
        "divider_row": divider_row,
        "regions": regions,
        "bars": bars,
        "bar_by_height": bar_by_height,
        "unused_bars": unused_bars,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill each enclosed background region with the color of the unique legend bar whose height equals that region's area.

    Input structure:
    - A red-walled upper chamber is partitioned into several enclosed background regions whose areas are all different.
    - A full red divider row separates that chamber from a lower legend area containing several vertical bars.
    - Each legend bar is a solid monochrome rectangle resting on the bottom edge; bar colors and bar heights are all distinct, and there are more bars than regions.

    Rule:
    - Compute the cell count of every enclosed upper-region bounded by the red walls.
    - Find the unique legend bar whose height equals that area.
    - Flood-fill the whole region with that bar's color, while leaving the red walls, divider row, legend bars, and unmatched decoy bars unchanged.

    Key insight:
    - Regions and bars are paired neither by position nor by rank, but only by the visible numeric code “region area = bar height”; the extra bars rule out any order-based shortcut.
    """
    analysis = _analyze_scene(np.asarray(input_grid, dtype=int))
    out = analysis["grid"].copy()
    for region in analysis["regions"]:
        color = analysis["bar_by_height"][region["area"]]["color"]
        for r, c in region["coords"]:
            out[r, c] = color
    return out


def _render_body(row_heights: list[int], col_widths: list[int], bg: int) -> tuple[np.ndarray, list[int]]:
    body_h = sum(row_heights) + len(row_heights) + 1
    body_w = sum(col_widths) + len(col_widths) + 1
    body = np.full((body_h, body_w), bg, dtype=int)

    wall_rows = [0]
    cursor = 0
    for height in row_heights:
        cursor += height + 1
        wall_rows.append(cursor)

    wall_cols = [0]
    cursor = 0
    for width in col_widths:
        cursor += width + 1
        wall_cols.append(cursor)

    body[wall_rows, :] = RED
    body[:, wall_cols] = RED

    areas = [rh * cw for rh in row_heights for cw in col_widths]
    return body, areas


def _partition_options() -> list[tuple[list[int], list[int]]]:
    options: list[tuple[list[int], list[int]]] = []
    row_choices = [1, 2]
    col_choices = [2, 3, 4]
    for n_rows in row_choices:
        for n_cols in col_choices:
            if n_rows * n_cols < 2 or n_rows * n_cols > 6:
                continue
            for row_heights in np.ndindex(*((4,) * n_rows)):
                rows = [int(v) + 1 for v in row_heights]
                for col_widths in np.ndindex(*((4,) * n_cols)):
                    cols = [int(v) + 2 for v in col_widths]
                    areas = [rh * cw for rh in rows for cw in cols]
                    if len(set(areas)) != len(areas):
                        continue
                    body_h = sum(rows) + len(rows) + 1
                    body_w = sum(cols) + len(cols) + 1
                    if body_h > 12 or body_w > 22:
                        continue
                    if max(areas) > 16:
                        continue
                    if body_w < 2 * (len(areas) + 1) + 1:
                        continue
                    options.append((rows, cols))
    return options


PARTITIONS = _partition_options()


def _sample_partition() -> tuple[list[int], list[int]]:
    single_row = [option for option in PARTITIONS if len(option[0]) == 1]
    two_row_distinct = [option for option in PARTITIONS if len(option[0]) == 2 and len(set(option[0])) == 2]
    two_row_equal = [option for option in PARTITIONS if len(option[0]) == 2 and len(set(option[0])) == 1]

    families = [
        (single_row, 2),
        (two_row_distinct, 4),
        (two_row_equal, 1),
    ]
    available = [(options, weight) for options, weight in families if options]
    choice_pool = [options for options, weight in available for _ in range(weight)]
    rows, cols = random.choice(random.choice(choice_pool))
    return list(rows), list(cols)


def _pick_palette(bar_count: int) -> tuple[int, list[int]]:
    bg = random.choice([c for c in ARC_COLORS if c != RED])
    choices = [c for c in ARC_COLORS if c not in {bg, RED}]
    return bg, random.sample(choices, bar_count)


def _choose_legend_heights(areas: list[int], body_h: int, body_w: int) -> tuple[int, list[int]] | None:
    region_count = len(areas)
    max_bars = min(8, (body_w - 1) // 2)
    if max_bars <= region_count:
        return None

    max_legend_h = 30 - body_h
    max_area = max(areas)
    if max_legend_h < max_area:
        return None

    legend_h = random.randint(max_area, min(max_area + 4, max_legend_h))
    available_extras = [h for h in range(1, legend_h + 1) if h not in set(areas)]
    if not available_extras:
        return None

    extra_cap = min(3, len(available_extras), max_bars - region_count)
    if extra_cap < 1:
        return None

    extra_count = random.randint(1, extra_cap)
    heights = list(areas) + random.sample(available_extras, extra_count)
    random.shuffle(heights)
    return legend_h, heights


def _plan_bar_widths(bar_count: int, body_w: int) -> tuple[list[int], list[int]] | None:
    for _ in range(80):
        widths = [random.randint(1, 2) for _ in range(bar_count)]
        gaps = [random.randint(1, 2) for _ in range(bar_count + 1)]
        required = sum(widths) + sum(gaps)
        if required > body_w:
            continue
        slack = body_w - required
        for _ in range(slack):
            gaps[random.randrange(len(gaps))] += 1
        return widths, gaps
    return None


def _render_legend(body_w: int, legend_h: int, bg: int, heights: list[int], colors: list[int]) -> np.ndarray | None:
    layout = _plan_bar_widths(len(heights), body_w)
    if layout is None:
        return None
    widths, gaps = layout
    legend = np.full((legend_h, body_w), bg, dtype=int)
    c = gaps[0]
    for i, (height, width, color) in enumerate(zip(heights, widths, colors, strict=True)):
        legend[legend_h - height :, c : c + width] = color
        c += width + gaps[i + 1]
    return legend


def _generate_attempt() -> np.ndarray | None:
    row_heights, col_widths = _sample_partition()
    bg, _ = _pick_palette(1)
    body, areas = _render_body(row_heights, col_widths, bg)

    legend_plan = _choose_legend_heights(areas, body.shape[0], body.shape[1])
    if legend_plan is None:
        return None
    legend_h, heights = legend_plan

    bg, colors = _pick_palette(len(heights))
    body, areas = _render_body(row_heights, col_widths, bg)
    legend = _render_legend(body.shape[1], legend_h, bg, heights, colors)
    if legend is None:
        return None

    grid = np.vstack([body, legend])
    if max(grid.shape) > 30:
        return None

    try:
        analysis = _analyze_scene(grid)
        if len(analysis["regions"]) != len(areas):
            return None
        if sorted(region["area"] for region in analysis["regions"]) != sorted(areas):
            return None
        if len(analysis["unused_bars"]) < 1:
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-height code puzzle")
