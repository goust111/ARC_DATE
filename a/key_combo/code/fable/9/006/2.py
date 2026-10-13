from __future__ import annotations

from collections import Counter, deque
import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int, *, connectivity: int = 4) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    components: list[dict] = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q: deque[Coord] = deque([(r, c)])
            visited[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": _bbox(arr),
                }
            )
    return components


def _crop_panel(grid: np.ndarray, axis: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    if axis == "row":
        return grid[:idx, :].copy(), grid[idx + 1 :, :].copy()
    return grid[:, :idx].copy(), grid[:, idx + 1 :].copy()


def _is_solid_bar(panel: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    return np.all(panel[r0 : r1 + 1, c0 : c1 + 1] == color)


def _extract_bars(panel: np.ndarray) -> list[dict] | None:
    bg = _get_bg(panel)
    comps = _find_components(panel, bg, connectivity=4)
    if len(comps) < 2:
        return None

    bars: list[dict] = []
    H, _W = panel.shape
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r1 != H - 1:
            return None
        if not _is_solid_bar(panel, comp):
            return None
        bars.append(
            {
                "color": int(comp["color"]),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "left": int(c0),
                "bbox": comp["bbox"],
            }
        )

    heights = [bar["height"] for bar in bars]
    if len(set(heights)) != len(heights):
        return None
    return bars


def _extract_objects(panel: np.ndarray, expected_count: int) -> tuple[int, list[dict]] | None:
    bg = _get_bg(panel)
    comps = _find_components(panel, bg, connectivity=4)
    if len(comps) != expected_count:
        return None
    areas = [comp["area"] for comp in comps]
    if len(set(areas)) != len(areas):
        return None
    return bg, comps


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    H, W = g.shape

    candidates: list[dict] = []
    for r in range(1, H - 1):
        if np.all(g[r, :] == g[r, 0]):
            candidates.append({"axis": "row", "idx": int(r), "color": int(g[r, 0])})
    for c in range(1, W - 1):
        if np.all(g[:, c] == g[0, c]):
            candidates.append({"axis": "col", "idx": int(c), "color": int(g[0, c])})

    solutions: list[dict] = []
    for candidate in candidates:
        first, second = _crop_panel(g, candidate["axis"], candidate["idx"])
        if 0 in first.shape or 0 in second.shape:
            continue
        for legend_panel, object_panel in ((first, second), (second, first)):
            bars = _extract_bars(legend_panel)
            if bars is None:
                continue
            object_info = _extract_objects(object_panel, len(bars))
            if object_info is None:
                continue
            object_bg, objects = object_info
            solutions.append(
                {
                    "separator": candidate,
                    "legend_panel": legend_panel,
                    "object_panel": object_panel,
                    "object_bg": object_bg,
                    "bars": bars,
                    "objects": objects,
                }
            )

    assert len(solutions) == 1, "expected one valid separator / panel interpretation"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the cropped object panel by matching object area rank to legend bar height rank.

    Input structure:
    - One full-length separator line splits the grid into a legend panel and an object panel.
    - The legend panel contains several solid colored vertical bars whose bottom edges align,
      with distinct heights and left-to-right order unrelated to height.
    - The object panel contains the same number of monochrome connected objects with distinct
      areas; some object colors overlap with bar colors as decoys.

    Rule:
    - Remove the separator and the legend panel; output only the object panel.
    - Rank legend bars by descending height and rank objects by descending area.
    - Recolor every cell of the i-th largest object to the color of the i-th tallest bar,
      preserving each object's shape and position.

    Key insight:
    - Pairing is determined by rank-to-rank correspondence (height rank to area rank), not by
      shared color and not by horizontal placement.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    out = info["object_panel"].copy()

    bars = sorted(info["bars"], key=lambda bar: bar["height"], reverse=True)
    objects = sorted(info["objects"], key=lambda comp: comp["area"], reverse=True)
    for bar, comp in zip(bars, objects):
        rr, cc = comp["coords"].T
        out[rr, cc] = int(bar["color"])
    return out


ALL_COLORS = tuple(range(10))
MIN_OBJECTS = 3
MAX_OBJECTS = 5
MAX_ATTEMPTS = 400


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize_coords(coords: set[Coord]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _transform_coords(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arr = coords.copy()
    if rng.integers(0, 2):
        arr[:, 0] *= -1
    if rng.integers(0, 2):
        arr[:, 1] *= -1
    if rng.integers(0, 2):
        arr = arr[:, ::-1]
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _rectangle_shape(area: int, rng: np.random.Generator) -> np.ndarray | None:
    options = []
    for h in range(1, 6):
        if area % h != 0:
            continue
        w = area // h
        if 1 <= w <= 6:
            options.append((h, w))
    if not options:
        return None
    h, w = options[int(rng.integers(0, len(options)))]
    coords = {(r, c) for r in range(h) for c in range(w)}
    return _normalize_coords(coords)


def _l_shape(area: int, rng: np.random.Generator) -> np.ndarray | None:
    options = []
    for arm_r in range(2, 7):
        for arm_c in range(2, 7):
            if arm_r + arm_c - 1 == area:
                options.append((arm_r, arm_c))
    if not options:
        return None
    arm_r, arm_c = options[int(rng.integers(0, len(options)))]
    coords = {(r, 0) for r in range(arm_r)} | {(0, c) for c in range(arm_c)}
    return _normalize_coords(coords)


def _grown_shape(area: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(80):
        coords: set[Coord] = {(0, 0)}
        while len(coords) < area:
            frontier = set()
            for cell in coords:
                for nxt in _neighbors4(cell):
                    if nxt not in coords:
                        frontier.add(nxt)
            candidates = []
            for cand in frontier:
                trial = coords | {cand}
                rs = [r for r, _c in trial]
                cs = [c for _r, c in trial]
                if max(rs) - min(rs) + 1 <= 6 and max(cs) - min(cs) + 1 <= 6:
                    candidates.append(cand)
            if not candidates:
                break
            coords.add(candidates[int(rng.integers(0, len(candidates)))])
        if len(coords) == area:
            return _normalize_coords(coords)
    raise RuntimeError("failed to grow a connected shape")


def _random_shape(area: int, rng: np.random.Generator) -> np.ndarray:
    builders = []
    rect = _rectangle_shape(area, rng)
    if rect is not None:
        builders.append(rect)
    ell = _l_shape(area, rng)
    if ell is not None:
        builders.append(ell)
    builders.append(_grown_shape(area, rng))
    shape = builders[int(rng.integers(0, len(builders)))]
    return _transform_coords(shape, rng)


def _mark_forbidden(mask: np.ndarray, coords: np.ndarray) -> None:
    H, W = mask.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < H and 0 <= nc < W:
                    mask[nr, nc] = True


def _place_objects(
    panel_shape: tuple[int, int],
    areas: list[int],
    colors_by_rank: list[int],
    rng: np.random.Generator,
    bg: int,
) -> tuple[np.ndarray, list[dict]] | None:
    H, W = panel_shape
    panel = np.full((H, W), bg, dtype=int)
    forbidden = np.zeros((H, W), dtype=bool)
    objects: list[dict] = []

    shuffled_indices = list(rng.permutation(len(areas)))
    for rank_idx in shuffled_indices:
        area = int(areas[rank_idx])
        color = int(colors_by_rank[rank_idx])
        placed = False
        for _ in range(120):
            shape = _random_shape(area, rng)
            h = int(shape[:, 0].max()) + 1
            w = int(shape[:, 1].max()) + 1
            if H - h - 1 <= 1 or W - w - 1 <= 1:
                return None
            r0 = int(rng.integers(1, H - h))
            c0 = int(rng.integers(1, W - w))
            coords = shape + np.array([r0, c0])
            rr, cc = coords.T
            if np.any(forbidden[rr, cc]):
                continue
            panel[rr, cc] = color
            _mark_forbidden(forbidden, coords)
            objects.append(
                {
                    "rank": rank_idx,
                    "color": color,
                    "coords": coords.astype(int),
                    "area": area,
                    "bbox": _bbox(coords.astype(int)),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return panel, objects


def _sample_bar_colors_and_object_colors(
    n: int, rng: np.random.Generator
) -> tuple[int, int, list[int], list[int]]:
    bg, separator = rng.choice(ALL_COLORS, size=2, replace=False).tolist()
    pool = [c for c in ALL_COLORS if c not in {bg, separator}]
    bar_colors = rng.choice(pool, size=n, replace=False).tolist()
    extras = [c for c in pool if c not in bar_colors]
    min_overlap = max(1, n - len(extras))
    overlap_count = int(rng.integers(min_overlap, n))
    overlap = rng.choice(bar_colors, size=overlap_count, replace=False).tolist()
    extra_pick = rng.choice(extras, size=n - overlap_count, replace=False).tolist()
    object_palette = overlap + extra_pick
    return int(bg), int(separator), list(map(int, bar_colors)), list(map(int, object_palette))


def _assign_object_colors(
    palette: list[int], target_colors: list[int], rng: np.random.Generator
) -> list[int] | None:
    bar_set = set(target_colors)
    for _ in range(80):
        perm = [int(x) for x in rng.permutation(palette)]
        if perm == target_colors:
            continue
        if not any(src != dst for src, dst in zip(perm, target_colors)):
            continue
        if any(src in bar_set and src != dst for src, dst in zip(perm, target_colors)):
            return perm
    return None


def _sample_bar_layout(
    n: int, panel_height: int, total_width: int, rng: np.random.Generator
) -> list[dict] | None:
    max_height = min(panel_height - 1, 8)
    if max_height < n + 1:
        return None

    heights_pool = list(range(2, max_height + 1))
    heights = rng.choice(heights_pool, size=n, replace=False).tolist()
    widths = rng.integers(1, 3, size=n).tolist()
    min_left = int(rng.integers(1, 4))
    min_right = int(rng.integers(1, 4))
    min_gaps = rng.integers(1, 3, size=n - 1).tolist()
    required = min_left + min_right + sum(widths) + sum(min_gaps)
    if required > total_width:
        return None

    extra = total_width - required
    slots = [0] * (n + 1)
    for _ in range(extra):
        slots[int(rng.integers(0, n + 1))] += 1

    x = min_left + slots[0]
    bars: list[dict] = []
    for idx in range(n):
        w = int(widths[idx])
        h = int(heights[idx])
        bars.append({"left": x, "width": w, "height": h})
        x += w
        if idx < n - 1:
            x += int(min_gaps[idx] + slots[idx + 1])

    height_sequence = [bar["height"] for bar in bars]
    if height_sequence == sorted(height_sequence) or height_sequence == sorted(height_sequence, reverse=True):
        return None
    return bars


def _choose_areas(
    panel_shape: tuple[int, int], n: int, rng: np.random.Generator
) -> list[int] | None:
    H, W = panel_shape
    max_area = min(12, max(7, (H * W) // max(10, 2 * n)))
    candidates = list(range(4, max_area + 1))
    if len(candidates) < n:
        return None
    areas = sorted(rng.choice(candidates, size=n, replace=False).tolist(), reverse=True)
    if sum(areas) > int(0.28 * H * W):
        return None
    return [int(a) for a in areas]


def _render_bars(
    panel_shape: tuple[int, int],
    bars: list[dict],
    colors_in_x_order: list[int],
    bg: int,
) -> np.ndarray:
    H, W = panel_shape
    panel = np.full((H, W), bg, dtype=int)
    for bar, color in zip(bars, colors_in_x_order):
        left = int(bar["left"])
        width = int(bar["width"])
        height = int(bar["height"])
        panel[H - height : H, left : left + width] = int(color)
    return panel


def _bars_rank_colors(bars: list[dict], colors_in_x_order: list[int]) -> list[int]:
    combined = []
    for bar, color in zip(bars, colors_in_x_order):
        combined.append({"height": int(bar["height"]), "color": int(color)})
    combined.sort(key=lambda item: item["height"], reverse=True)
    return [int(item["color"]) for item in combined]


def _object_x_order_matches_rank(objects: list[dict]) -> bool:
    by_x = sorted(objects, key=lambda obj: (obj["bbox"][1] + obj["bbox"][3], obj["bbox"][0]))
    return [obj["area"] for obj in by_x] == sorted((obj["area"] for obj in objects), reverse=True)


def _expected_output(panel: np.ndarray, objects: list[dict], rank_colors: list[int]) -> np.ndarray:
    out = panel.copy()
    ranked_objects = sorted(objects, key=lambda obj: obj["area"], reverse=True)
    for obj, color in zip(ranked_objects, rank_colors):
        rr, cc = obj["coords"].T
        out[rr, cc] = int(color)
    return out


def _generate_horizontal(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    bg, separator, bar_color_pool, object_palette = _sample_bar_colors_and_object_colors(n, rng)

    object_h = int(rng.integers(9, 18))
    total_w = int(rng.integers(14, 25))
    legend_h = int(rng.integers(5, 10))
    if object_h + legend_h + 1 > 30:
        return None

    bars = _sample_bar_layout(n, legend_h, total_w, rng)
    if bars is None:
        return None
    bar_colors_in_x_order = [int(x) for x in rng.permutation(bar_color_pool)]
    rank_colors = _bars_rank_colors(bars, bar_colors_in_x_order)

    object_colors_by_rank = _assign_object_colors(object_palette, rank_colors, rng)
    if object_colors_by_rank is None:
        return None

    areas = _choose_areas((object_h, total_w), n, rng)
    if areas is None:
        return None
    placed = _place_objects((object_h, total_w), areas, object_colors_by_rank, rng, bg)
    if placed is None:
        return None
    object_panel, objects = placed
    if _object_x_order_matches_rank(objects):
        return None

    legend_panel = _render_bars((legend_h, total_w), bars, bar_colors_in_x_order, bg)
    expected = _expected_output(object_panel, objects, rank_colors)

    if rng.integers(0, 2) == 0:
        grid = np.vstack(
            [
                legend_panel,
                np.full((1, total_w), separator, dtype=int),
                object_panel,
            ]
        )
    else:
        grid = np.vstack(
            [
                object_panel,
                np.full((1, total_w), separator, dtype=int),
                legend_panel,
            ]
        )
    return grid, expected


def _generate_vertical(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    bg, separator, bar_color_pool, object_palette = _sample_bar_colors_and_object_colors(n, rng)

    total_h = int(rng.integers(12, 21))
    object_w = int(rng.integers(9, 18))
    bars = _sample_bar_layout(n, total_h, int(rng.integers(9, 14)), rng)
    if bars is None:
        return None
    legend_w = max(bar["left"] + bar["width"] for bar in bars) + int(rng.integers(1, 4))
    if legend_w + 1 + object_w > 30:
        return None

    bar_colors_in_x_order = [int(x) for x in rng.permutation(bar_color_pool)]
    rank_colors = _bars_rank_colors(bars, bar_colors_in_x_order)
    object_colors_by_rank = _assign_object_colors(object_palette, rank_colors, rng)
    if object_colors_by_rank is None:
        return None

    areas = _choose_areas((total_h, object_w), n, rng)
    if areas is None:
        return None
    placed = _place_objects((total_h, object_w), areas, object_colors_by_rank, rng, bg)
    if placed is None:
        return None
    object_panel, objects = placed
    if _object_x_order_matches_rank(objects):
        return None

    legend_panel = _render_bars((total_h, legend_w), bars, bar_colors_in_x_order, bg)
    expected = _expected_output(object_panel, objects, rank_colors)

    if rng.integers(0, 2) == 0:
        grid = np.hstack(
            [
                legend_panel,
                np.full((total_h, 1), separator, dtype=int),
                object_panel,
            ]
        )
    else:
        grid = np.hstack(
            [
                object_panel,
                np.full((total_h, 1), separator, dtype=int),
                legend_panel,
            ]
        )
    return grid, expected


def _valid_decoy(panel: np.ndarray, expected: np.ndarray, bar_rank_colors: list[int]) -> bool:
    changed = int(np.count_nonzero(panel != expected))
    if changed == 0:
        return False

    bg = _get_bg(panel)
    objects = sorted(_find_components(panel, bg, connectivity=4), key=lambda comp: comp["area"], reverse=True)
    overlap_wrong = False
    for comp, target in zip(objects, bar_rank_colors):
        color = int(comp["color"])
        if color in set(bar_rank_colors) and color != int(target):
            overlap_wrong = True
            break
    return overlap_wrong


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        generated = _generate_horizontal(rng) if rng.integers(0, 2) == 0 else _generate_vertical(rng)
        if generated is None:
            continue
        grid, expected = generated
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue

        info = _analyze_grid(grid)
        rank_colors = [bar["color"] for bar in sorted(info["bars"], key=lambda bar: bar["height"], reverse=True)]
        if not _valid_decoy(info["object_panel"], expected, rank_colors):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
