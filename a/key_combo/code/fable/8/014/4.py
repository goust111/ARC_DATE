from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIHEDRAL_TRANSFORMS = (
    (1, 0, 0, 1),
    (1, 0, 0, -1),
    (-1, 0, 0, 1),
    (-1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, 1, -1, 0),
    (0, -1, 1, 0),
    (0, -1, -1, 0),
)
PALETTE = tuple(range(10))
GEN_RETRIES = 400
GEN_TEMPLATE_COUNT_RANGE = (2, 4)
GEN_TOTAL_QUERY_RANGE = (4, 14)
GEN_HEIGHT_RANGE = (14, 26)
GEN_RIGHT_WIDTH_RANGE = (10, 18)
GEN_LEGEND_GAP_CHOICES = (1, 2)
GEN_TEMPLATE_SIZE_RANGE = (3, 5)
GEN_TEMPLATE_MAX_SPAN = 4
GEN_QUERY_BBOX_GAP = 2
GEN_TOP_MARGIN_CHOICES = (0, 1, 2)
GEN_RIGHT_MARGIN = 1


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize(points: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in points))


def _shape_variants(points: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    base = [(int(r), int(c)) for r, c in points]
    seen: set[tuple[tuple[int, int], ...]] = set()
    out = []
    for rr_r, rr_c, cc_r, cc_c in DIHEDRAL_TRANSFORMS:
        transformed = [(rr_r * r + rr_c * c, cc_r * r + cc_c * c) for r, c in base]
        norm = _normalize(transformed)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return tuple(out)


def _canonical_shape(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    points = [(int(r), int(c)) for r, c in coords]
    return min(_shape_variants(points))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(
    grid: np.ndarray,
    *,
    colors: set[int] | None = None,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == -1:
                continue
            if colors is not None and color not in colors:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    counts = Counter(map(int, grid.ravel()))
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == bg:
            continue
        if counts[color] == h and np.all(grid[:, c] == color):
            candidates.append((c, color))
    assert len(candidates) == 1, "expected one full-height divider column"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Count query sprites by legend shape class and encode the counts as a vertical run-length bar.

    Input structure:
    - A grid has one background color and a single full-height monochrome divider column.
    - To the left of the divider, distinct-colored legend sprites are stacked from top to bottom.
    - To the right of the divider, all query sprites share one placeholder color and each matches
      exactly one legend shape up to rotation or reflection.

    Rule:
    - Detect the divider and read the left-side legend sprites in top-to-bottom order.
    - Canonicalize every legend and query sprite under all rotations/reflections, then classify each
      query by the matching legend shape.
    - Return a one-column output whose consecutive color runs follow the legend order, with each run
      length equal to that legend's query count.

    Key insight:
    - The query sprites are not recolored in place; their shape frequencies are summarized and then
      compressed into an ordered run-length histogram using the legend colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    divider_c, divider_color = _find_divider(g, bg)

    legend = g[:, :divider_c]
    query_zone = g[:, divider_c + 1 :]
    assert legend.size > 0 and query_zone.size > 0, "divider must split legend and query regions"

    legend_components = _find_components(legend, colors=set(int(v) for v in np.unique(legend) if int(v) != bg))
    assert legend_components, "expected at least one legend sprite"
    legend_components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    ordered_legend: list[tuple[tuple[tuple[int, int], ...], int]] = []
    shape_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    legend_colors: set[int] = set()
    for comp in legend_components:
        color = int(comp["color"])
        assert color != divider_color
        key = _canonical_shape(comp["coords"])
        assert key not in shape_to_color, "legend shapes must be pairwise inequivalent"
        shape_to_color[key] = color
        legend_colors.add(color)
        ordered_legend.append((key, color))

    right_colors = {int(v) for v in np.unique(query_zone) if int(v) != bg}
    assert len(right_colors) == 1, "expected one query color on the right side"
    query_color = next(iter(right_colors))
    assert query_color not in legend_colors and query_color != divider_color

    query_components = _find_components(query_zone, colors={query_color})
    assert query_components, "expected at least one query sprite"

    counts = {key: 0 for key, _ in ordered_legend}
    for comp in query_components:
        key = _canonical_shape(comp["coords"])
        assert key in shape_to_color, "query sprite must match a legend shape"
        counts[key] += 1

    total_queries = len(query_components)
    out = np.empty((total_queries, 1), dtype=int)
    start = 0
    for key, color in ordered_legend:
        count = counts[key]
        out[start : start + count, 0] = color
        start += count
    assert start == total_queries
    return out


def _build_shape_catalog() -> list[dict]:
    low, high = GEN_TEMPLATE_SIZE_RANGE
    fixed: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, high + 1):
        shapes = set()
        for shape in fixed[size - 1]:
            cells = set(shape)
            for r, c in shape:
                for dr, dc in NEIGHBORS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        shapes.add(_normalize(list(cells | {nxt})))
        fixed[size] = shapes

    catalog = []
    seen = set()
    for size in range(low, high + 1):
        for shape in fixed[size]:
            key = _canonical_shape(shape)
            if key in seen:
                continue
            seen.add(key)
            variants = _shape_variants(key)
            spans = [(max(r for r, _ in variant) + 1, max(c for _, c in variant) + 1) for variant in variants]
            if len(variants) < 2:
                continue
            if max(h for h, _ in spans) > GEN_TEMPLATE_MAX_SPAN or max(w for _, w in spans) > GEN_TEMPLATE_MAX_SPAN:
                continue
            catalog.append({"key": key, "variants": variants, "area": size})
    return catalog


SHAPE_CATALOG = _build_shape_catalog()


def _patch_from_coords(coords: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.full((h, w), bg, dtype=int)
    rr, cc = np.array(coords, dtype=int).T
    patch[rr, cc] = int(color)
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    mask = patch != int(transparent)
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _interval_sep(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _bbox_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    dr = _interval_sep(a[0], a[2], b[0], b[2])
    dc = _interval_sep(a[1], a[3], b[1], b[3])
    return max(dr, dc) >= gap


def _random_positive_counts(total: int, n_parts: int) -> list[int]:
    counts = [1] * n_parts
    for _ in range(total - n_parts):
        counts[random.randrange(n_parts)] += 1
    random.shuffle(counts)
    return counts


def _legend_layout(templates: list[dict]) -> tuple[list[dict], int, int] | None:
    legend_variants = []
    left_w = 0
    y = random.choice(GEN_TOP_MARGIN_CHOICES)
    for idx, item in enumerate(templates):
        if idx:
            y += random.choice(GEN_LEGEND_GAP_CHOICES)
        variant = random.choice(item["variants"])
        patch_h = max(r for r, _ in variant) + 1
        patch_w = max(c for _, c in variant) + 1
        legend_variants.append({"key": item["key"], "coords": variant, "h": patch_h, "w": patch_w, "y": y})
        left_w = max(left_w, patch_w)
        y += patch_h
    total_h = y + random.choice(GEN_TOP_MARGIN_CHOICES)
    if total_h > GEN_HEIGHT_RANGE[1] - 2:
        return None
    left_w += random.choice((1, 2, 3))
    return legend_variants, left_w, total_h


def _draw_legend(
    grid: np.ndarray,
    legend_variants: list[dict],
    left_w: int,
    colors: list[int],
    bg: int,
) -> list[dict]:
    infos = []
    for item, color in zip(legend_variants, colors):
        y = int(item["y"])
        x = random.randint(0, left_w - item["w"])
        patch = _patch_from_coords(item["coords"], color, bg)
        _paste(grid, patch, y, x, transparent=bg)
        infos.append({"key": item["key"], "color": color, "legend_coords": item["coords"]})
    return infos


def _query_candidates(
    grid: np.ndarray,
    patch: np.ndarray,
    divider_c: int,
    bg: int,
    existing: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    h, w = patch.shape
    grid_h, grid_w = grid.shape
    mask = patch != bg
    candidates = []
    for r0 in range(0, grid_h - h + 1):
        for c0 in range(divider_c + 1, grid_w - w + 1):
            if c0 + w + GEN_RIGHT_MARGIN > grid_w:
                continue
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(not _bbox_far_enough(bbox, other, GEN_QUERY_BBOX_GAP) for other in existing):
                continue
            view = grid[r0 : r0 + h, c0 : c0 + w]
            if np.any(view[mask] != bg):
                continue
            candidates.append((r0, c0, bbox))
    return candidates


def _place_queries(
    grid: np.ndarray,
    legend_infos: list[dict],
    counts: list[int],
    query_color: int,
    divider_c: int,
    bg: int,
) -> bool:
    assignments = [idx for idx, count in enumerate(counts) for _ in range(count)]
    random.shuffle(assignments)
    need_transformed = len(assignments) // 2 + 1
    transformed_slots = set(random.sample(range(len(assignments)), need_transformed))
    placed_bboxes: list[tuple[int, int, int, int]] = []

    for slot, template_idx in enumerate(assignments):
        info = legend_infos[template_idx]
        variants = list(_shape_variants(info["key"]))
        legend_coords = info["legend_coords"]
        if slot in transformed_slots:
            usable = [variant for variant in variants if variant != legend_coords]
        else:
            usable = variants
        random.shuffle(usable)
        placed = False
        for variant in usable:
            patch = _patch_from_coords(variant, query_color, bg)
            candidates = _query_candidates(grid, patch, divider_c, bg, placed_bboxes)
            if not candidates:
                continue
            r0, c0, bbox = random.choice(candidates)
            _paste(grid, patch, r0, c0, transparent=bg)
            placed_bboxes.append(bbox)
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_output(colors: list[int], counts: list[int]) -> np.ndarray:
    total = sum(counts)
    out = np.empty((total, 1), dtype=int)
    start = 0
    for color, count in zip(colors, counts):
        out[start : start + count, 0] = color
        start += count
    return out


def _attempt() -> np.ndarray | None:
    n_templates = random.randint(*GEN_TEMPLATE_COUNT_RANGE)
    templates = random.sample(SHAPE_CATALOG, n_templates)
    legend_layout = _legend_layout(templates)
    if legend_layout is None:
        return None
    legend_variants, left_w, min_h = legend_layout

    total_queries = random.randint(max(GEN_TOTAL_QUERY_RANGE[0], n_templates), GEN_TOTAL_QUERY_RANGE[1])
    counts = _random_positive_counts(total_queries, n_templates)
    height = random.randint(max(min_h, GEN_HEIGHT_RANGE[0]), GEN_HEIGHT_RANGE[1])
    right_w = random.randint(*GEN_RIGHT_WIDTH_RANGE)
    width = left_w + 1 + right_w
    if width > 30 or total_queries > 30:
        return None

    bg, divider_color, query_color, *template_colors = random.sample(PALETTE, n_templates + 3)
    grid = np.full((height, width), bg, dtype=int)
    divider_c = left_w
    grid[:, divider_c] = divider_color

    legend_infos = _draw_legend(grid, legend_variants, left_w, template_colors, bg)
    if not _place_queries(grid, legend_infos, counts, query_color, divider_c, bg):
        return None

    if _bg_color(grid) != bg:
        return None

    expected = _expected_output(template_colors, counts)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-histogram input")
