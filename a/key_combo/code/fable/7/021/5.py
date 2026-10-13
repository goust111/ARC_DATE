from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, neighbors4


_BASE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)

_PALETTE = tuple(Color.NOT_BLACK)
_MAX_ATTEMPTS = 400


def _orientations(shape: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    pts = np.array(shape, dtype=int)
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    )
    out: list[np.ndarray] = []
    seen = set()
    for fn in transforms:
        variant = np.array([fn(int(r), int(c)) for r, c in pts], dtype=int)
        variant -= variant.min(axis=0)
        key = tuple(sorted(map(tuple, variant.tolist())))
        if key not in seen:
            seen.add(key)
            out.append(variant)
    return out


_SHAPE_VARIANTS = [_orientations(shape) for shape in _BASE_SHAPES]


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[int], int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3, "expected a 2D ARC grid"
    bg = int(get_bg(g))

    legend: list[int] = []
    c = 0
    while c < g.shape[1] and int(g[0, c]) != bg:
        color = int(g[0, c])
        if legend:
            assert color != legend[-1], "legend colors must be distinct"
        legend.append(color)
        c += 1
    assert legend, "expected a non-empty legend strip"
    assert np.all(g[0, c:] == bg), "legend strip must occupy a left prefix of the first row"
    assert len(set(legend)) == len(legend), "legend colors must be unique"

    separator = int(g[1, 0])
    assert separator != bg and separator not in legend, "separator color must be unique"
    assert np.all(g[1, :] == separator), "second row must be a full-width separator"

    scatter = g[2:, :]
    return g, bg, legend, separator, scatter


def _components_by_color(scatter: np.ndarray, bg: int) -> dict[int, list[np.ndarray]]:
    groups: dict[int, list[np.ndarray]] = {}
    for comp in find_components(scatter, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        groups.setdefault(int(comp["color"]), []).append(np.asarray(comp["coords"], dtype=int))
    return groups


def _component_touches_color(coords: np.ndarray, scatter: np.ndarray, target_color: int) -> bool:
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(scatter.shape, nr, nc) and int(scatter[nr, nc]) == int(target_color):
                return True
    return False


def _chain_counts(legend: list[int], scatter: np.ndarray, bg: int) -> list[int]:
    by_color = _components_by_color(scatter, bg)
    counts: list[int] = []
    for i, color in enumerate(legend):
        comps = by_color.get(int(color), [])
        if i == 0:
            counts.append(len(comps))
        else:
            prev_color = legend[i - 1]
            counts.append(sum(_component_touches_color(comp, scatter, prev_color) for comp in comps))
    return counts


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Count legend colors through a contact-qualification chain and emit the surviving colors as a single column.

    Input structure:
    - The first row is a left-aligned legend strip of distinct colors followed by background.
    - The second row is a full-width separator in a unique color.
    - Below the separator, each legend color appears as one or more separated 4-connected objects, and extra distractor colors may also appear.

    Rule:
    - For the first legend color, count all of its 4-connected objects in the scatter area.
    - For each later legend color, count only those objects that 4-neighbor touch at least one object of the immediately previous legend color.
    - Return a width-1 output column that repeats each legend color by its counted object total, skipping colors whose count is zero.

    Key insight:
    - The top strip is an ordered filter chain, not a simple inventory: every later color is qualified by contact with the previous legend color, while distractor colors never contribute.
    """
    _grid, bg, legend, _separator, scatter = _extract_scene(input_grid)
    counts = _chain_counts(legend, scatter, bg)
    out_colors = [color for color, count in zip(legend, counts) for _ in range(int(count))]
    assert out_colors, "the first legend color should contribute at least one object"
    return np.array(out_colors, dtype=int).reshape(-1, 1)


def _neighbor_colors(scatter: np.ndarray, coords: np.ndarray) -> set[int]:
    out: set[int] = set()
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(scatter.shape, nr, nc):
                color = int(scatter[nr, nc])
                if color != Color.BLACK:
                    out.add(color)
    return out


def _can_place(
    scatter: np.ndarray,
    coords: np.ndarray,
    *,
    bg: int,
    allowed_touch_colors: set[int],
    require_touch_color: int | None,
) -> bool:
    for r, c in coords:
        if not in_bounds(scatter.shape, int(r), int(c)):
            return False
        if int(scatter[int(r), int(c)]) != bg:
            return False
    adjacent = _neighbor_colors(scatter, coords)
    if not adjacent.issubset(allowed_touch_colors):
        return False
    if require_touch_color is not None and require_touch_color not in adjacent:
        return False
    return True


def _placement_candidates(
    scatter: np.ndarray,
    variants: list[np.ndarray],
    *,
    bg: int,
    allowed_touch_colors: set[int],
    require_touch_color: int | None,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    height, width = scatter.shape
    variant_order = variants[:]
    random.shuffle(variant_order)
    for variant in variant_order:
        max_r = int(variant[:, 0].max()) + 1
        max_c = int(variant[:, 1].max()) + 1
        anchors = [(r0, c0) for r0 in range(height - max_r + 1) for c0 in range(width - max_c + 1)]
        random.shuffle(anchors)
        for r0, c0 in anchors:
            placed = variant + np.array([r0, c0], dtype=int)
            if _can_place(
                scatter,
                placed,
                bg=bg,
                allowed_touch_colors=allowed_touch_colors,
                require_touch_color=require_touch_color,
            ):
                candidates.append(placed)
    return candidates


def _paint_component(scatter: np.ndarray, coords: np.ndarray, color: int) -> None:
    scatter[coords[:, 0], coords[:, 1]] = int(color)


def _pick_variant() -> list[np.ndarray]:
    return random.choice(_SHAPE_VARIANTS)


def _count_plan(legend_len: int) -> tuple[list[int], list[int]]:
    totals = [random.randint(1, 4) for _ in range(legend_len)]
    mode = random.choices(
        ("mid_zero_later_positive", "mixed", "all_positive"),
        weights=(4, 5, 2),
        k=1,
    )[0]

    qualified = [totals[0]]
    for i in range(1, legend_len):
        low = 1 if mode == "all_positive" else 0
        qualified.append(random.randint(low, totals[i]))

    if mode == "mid_zero_later_positive" and legend_len >= 3:
        pivot = random.randint(1, legend_len - 2)
        qualified[pivot] = 0
        qualified[pivot + 1] = random.randint(1, totals[pivot + 1])
    elif mode == "mixed" and legend_len >= 2 and all(q > 0 for q in qualified[1:]):
        pivot = random.randint(1, legend_len - 1)
        qualified[pivot] = 0

    if legend_len >= 2 and all(t == q for t, q in zip(totals[1:], qualified[1:])):
        pivot = random.randint(1, legend_len - 1)
        if totals[pivot] > 1:
            qualified[pivot] = random.randint(0, totals[pivot] - 1)

    return totals, qualified


def _build_scatter(
    scatter_shape: tuple[int, int],
    legend: list[int],
    totals: list[int],
    qualified: list[int],
    distractor_colors: list[int],
    bg: int,
) -> np.ndarray | None:
    scatter = np.full(scatter_shape, bg, dtype=int)

    for i, color in enumerate(legend):
        need_touch = 0 if i == 0 else qualified[i]
        need_isolated = totals[i] - need_touch

        for _ in range(need_touch):
            candidates = _placement_candidates(
                scatter,
                _pick_variant(),
                bg=bg,
                allowed_touch_colors={legend[i - 1]},
                require_touch_color=legend[i - 1],
            )
            if not candidates:
                return None
            _paint_component(scatter, random.choice(candidates), color)

        for _ in range(need_isolated):
            candidates = _placement_candidates(
                scatter,
                _pick_variant(),
                bg=bg,
                allowed_touch_colors=set(),
                require_touch_color=None,
            )
            if not candidates:
                return None
            _paint_component(scatter, random.choice(candidates), color)

    distractor_component_total = random.randint(len(distractor_colors), len(distractor_colors) * 2)
    used_touching_distractor = False
    for idx in range(distractor_component_total):
        color = distractor_colors[idx % len(distractor_colors)]
        touch_existing = random.random() < 0.55
        if touch_existing:
            target = random.choice(legend)
            candidates = _placement_candidates(
                scatter,
                _pick_variant(),
                bg=bg,
                allowed_touch_colors={target},
                require_touch_color=target,
            )
            if candidates:
                _paint_component(scatter, random.choice(candidates), color)
                used_touching_distractor = True
                continue
        candidates = _placement_candidates(
            scatter,
            _pick_variant(),
            bg=bg,
            allowed_touch_colors=set(),
            require_touch_color=None,
        )
        if not candidates:
            return None
        _paint_component(scatter, random.choice(candidates), color)

    if distractor_component_total and not used_touching_distractor:
        return None
    return scatter


def _attempt_generate() -> np.ndarray | None:
    legend_len = random.randint(3, 5)
    color_pool = random.sample(_PALETTE, legend_len + 1 + random.randint(1, 2))
    legend = color_pool[:legend_len]
    separator = color_pool[legend_len]
    distractor_colors = color_pool[legend_len + 1 :]
    totals, qualified = _count_plan(legend_len)

    scatter_h = random.randint(12, 22)
    scatter_w = random.randint(max(12, legend_len + 6), 22)
    scatter = _build_scatter((scatter_h, scatter_w), legend, totals, qualified, distractor_colors, Color.BLACK)
    if scatter is None:
        return None

    grid = np.full((scatter_h + 2, scatter_w), Color.BLACK, dtype=int)
    grid[0, :legend_len] = np.array(legend, dtype=int)
    grid[1, :] = separator
    grid[2:, :] = scatter

    counts = _chain_counts(legend, scatter, Color.BLACK)
    if counts != qualified:
        return None
    if sum(counts) > 30:
        return None
    if max(grid.shape) > 30:
        return None
    if np.count_nonzero(grid == Color.BLACK) <= max(np.count_nonzero(grid == c) for c in legend + [separator]):
        return None

    later_positive_after_zero = any(counts[i] == 0 and counts[i + 1] > 0 for i in range(1, len(counts) - 1))
    if random.random() < 0.35 and not later_positive_after_zero:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or out.shape[1] != 1:
        return None
    if out.shape[0] != sum(counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-contact counting input")
