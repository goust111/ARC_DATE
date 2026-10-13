from __future__ import annotations

from collections import Counter, deque
from itertools import combinations, permutations
import random

import numpy as np


BACKGROUND = 0
TEMPLATE_COLORS = (2, 1, 4)
DIVIDER_COLOR = 3
SOURCE_COLORS = (7, 5, 6)
BARRIER_COLOR = 8  # visible replacement for otherwise invisible black-on-black barriers
GATE_MARK_COLOR = 9

MAX_GENERATION_ATTEMPTS = 500
MIN_TEMPLATE_SIZE = 5
MAX_TEMPLATE_SIZE = 7
MIN_SOURCE_GAP = 2
MIN_TEMPLATE_GAP = 2
MIN_CORRIDOR_LEN = 5


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
            out.append(np.asarray(comp, dtype=int))
    return out


def _build_shape_pool() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(4) for c in range(4)]
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for area in range(2, 5):
        for combo in combinations(cells, area):
            coords = np.asarray(combo, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if len(_components(mask)) == 1:
                shapes.add(_normalize(coords))
    return sorted(shapes, key=lambda shape: (len(shape), _shape_dims(shape), shape))


SHAPE_POOL = _build_shape_pool()
SHAPE_GROUPS: dict[tuple[int, int, int], list[tuple[tuple[int, int], ...]]] = {}
for shape in SHAPE_POOL:
    area = len(shape)
    sh, sw = _shape_dims(shape)
    SHAPE_GROUPS.setdefault((area, sh, sw), []).append(shape)


def _extract_templates(grid: np.ndarray, divider_c: int) -> list[dict]:
    left = grid[:, :divider_c]
    templates: list[dict] = []
    for color in TEMPLATE_COLORS:
        coords = np.argwhere(left == color)
        assert coords.size > 0, f"missing template color {color}"
        comps = _components(left == color)
        comps = [comp for comp in comps if int(left[tuple(comp[0])]) == color]
        assert len(comps) == 1, f"template color {color} must form one component"
        local = comps[0]
        r0, c0, r1, c1 = _bbox(local)
        region = left[r0 : r1 + 1, c0 : c1 + 1]
        assert set(map(int, np.unique(region))).issubset({BACKGROUND, color})
        hole_mask = region == BACKGROUND
        hole_components = _components(hole_mask)
        assert len(hole_components) == 1, "each template must contain exactly one connected hole"
        hole_local = hole_components[0]
        assert 2 <= len(hole_local) <= 4
        assert not hole_mask[0].any()
        assert not hole_mask[-1].any()
        assert not hole_mask[:, 0].any()
        assert not hole_mask[:, -1].any()
        abs_hole = hole_local + np.array([r0, c0])
        templates.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "hole_coords": abs_hole,
                "hole_shape": _normalize(hole_local),
            }
        )
    templates.sort(key=lambda item: item["bbox"])
    return templates


def _extract_sources(grid: np.ndarray, divider_c: int) -> list[dict]:
    right = grid[:, divider_c + 1 :]
    sources: list[dict] = []
    for color in SOURCE_COLORS:
        coords = np.argwhere(right == color)
        assert coords.size > 0, f"missing source color {color}"
        comps = _components(right == color)
        comps = [comp for comp in comps if int(right[tuple(comp[0])]) == color]
        assert len(comps) == 1, f"source color {color} must form one component"
        local = comps[0]
        abs_coords = local + np.array([0, divider_c + 1])
        sources.append(
            {
                "color": color,
                "bbox": _bbox(abs_coords),
                "coords": abs_coords,
                "shape": _normalize(local),
            }
        )
    sources.sort(key=lambda item: item["bbox"])
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each right-side source shape to the unique left template hole, fill that hole, and mark only gated corridor columns.

    Input structure:
    - The background is black, and one full green divider column splits the board into left templates and right source shapes.
    - The left side contains three solid colored rectangles; each rectangle encloses one connected black hole shape.
    - The right side contains three small single-color 4-connected source shapes plus visible barrier segments above and below each horizontal corridor.

    Rule:
    - Extract every left-hole shape and every right-side source shape, normalized to their own bounding boxes.
    - Pair each source with the unique equal hole shape, erase the source, and recolor that hole with the source color.
    - For each source, scan the corridor columns strictly between the divider and the source: if the source row span is flanked immediately above and below by barrier cells in the same column, paint that whole source row span maroon in that column.
    - Keep the divider, templates, and barrier segments unchanged otherwise.

    Key insight:
    - Shape identity chooses the destination hole, while the separate barrier layout chooses which swept corridor columns receive the path mark.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    divider_cols = np.flatnonzero(np.all(g == DIVIDER_COLOR, axis=0))
    assert len(divider_cols) == 1, "expected exactly one full divider column"
    divider_c = int(divider_cols[0])

    templates = _extract_templates(g, divider_c)
    sources = _extract_sources(g, divider_c)
    assert len(templates) == len(sources) == 3

    hole_to_template = {tpl["hole_shape"]: tpl for tpl in templates}
    assert len(hole_to_template) == 3, "template holes must all differ"
    assert len({src["shape"] for src in sources}) == 3, "source shapes must all differ"

    out = g.copy()
    for src in sources:
        out[src["coords"][:, 0], src["coords"][:, 1]] = BACKGROUND

    for src in sources:
        assert src["shape"] in hole_to_template, "every source must have a matching template hole"
        tpl = hole_to_template[src["shape"]]
        out[tpl["hole_coords"][:, 0], tpl["hole_coords"][:, 1]] = src["color"]

        r0, c0, r1, _c1 = src["bbox"]
        assert r0 > 0 and r1 + 1 < g.shape[0]
        for c in range(divider_c + 1, c0):
            if int(g[r0 - 1, c]) != BACKGROUND and int(g[r1 + 1, c]) != BACKGROUND:
                out[r0 : r1 + 1, c] = GATE_MARK_COLOR

    return out


def _sample_shapes() -> list[tuple[tuple[int, int], ...]]:
    candidate_groups = [group for group in SHAPE_GROUPS.values() if len(group) >= 2]
    area_h_w = random.choice(candidate_groups)
    first_two = random.sample(area_h_w, 2)
    remaining = [shape for shape in SHAPE_POOL if shape not in first_two]
    third = random.choice(remaining)
    shapes = first_two + [third]
    random.shuffle(shapes)
    return shapes


def _distribute(total: int, buckets: int) -> list[int]:
    cuts = sorted(random.sample(range(total + buckets - 1), buckets - 1))
    values = []
    prev = -1
    for cut in cuts + [total + buckets - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _place_vertical_spans(heights: list[int], min_gap: int, total_height: int) -> list[int] | None:
    required = sum(heights) + min_gap * (len(heights) - 1)
    if required > total_height:
        return None
    extra = total_height - required
    slack = _distribute(extra, len(heights) + 1)
    starts: list[int] = []
    r = slack[0]
    for i, h in enumerate(heights):
        starts.append(r)
        r += h
        if i < len(heights) - 1:
            r += min_gap + slack[i + 1]
    return starts


def _choose_template_layout(shapes: list[tuple[tuple[int, int], ...]]) -> tuple[int, list[dict]] | None:
    sizes = []
    for shape in shapes:
        sh, sw = _shape_dims(shape)
        th = random.randint(max(MIN_TEMPLATE_SIZE, sh + 2), MAX_TEMPLATE_SIZE)
        tw = random.randint(max(MIN_TEMPLATE_SIZE, sw + 2), MAX_TEMPLATE_SIZE)
        sizes.append((th, tw))

    min_height = sum(th for th, _ in sizes) + MIN_TEMPLATE_GAP * 2
    h = random.randint(max(22, min_height), min(29, min_height + 5))
    starts = _place_vertical_spans([th for th, _ in sizes], MIN_TEMPLATE_GAP, h)
    if starts is None:
        return None

    divider_min = max(tw + 2 for _, tw in sizes)
    if divider_min > 12:
        return None
    divider_c = random.randint(divider_min, 12)

    layout = []
    template_colors = list(TEMPLATE_COLORS)
    random.shuffle(template_colors)
    for idx, (shape, (th, tw), r0, color) in enumerate(zip(shapes, sizes, starts, template_colors)):
        sh, sw = _shape_dims(shape)
        c0 = random.randint(1, divider_c - tw - 1)
        hole_r = random.randint(1, th - sh - 1)
        hole_c = random.randint(1, tw - sw - 1)
        layout.append(
            {
                "index": idx,
                "shape": shape,
                "template_color": color,
                "bbox": (r0, c0, r0 + th - 1, c0 + tw - 1),
                "hole_origin": (r0 + hole_r, c0 + hole_c),
            }
        )
    layout.sort(key=lambda item: item["bbox"])
    for new_idx, item in enumerate(layout):
        item["index"] = new_idx
    return divider_c, layout


def _non_identity_permutation() -> tuple[int, int, int]:
    perms = [perm for perm in permutations(range(3)) if perm != (0, 1, 2)]
    return random.choice(perms)


def _choose_source_layout(
    h: int,
    w: int,
    divider_c: int,
    templates: list[dict],
) -> tuple[list[dict], tuple[int, int, int]] | None:
    perm = _non_identity_permutation()
    source_shapes = [templates[idx]["shape"] for idx in perm]
    heights = [_shape_dims(shape)[0] for shape in source_shapes]
    starts = _place_vertical_spans(heights, MIN_SOURCE_GAP, h - 2)
    if starts is None:
        return None

    source_colors = list(SOURCE_COLORS)
    random.shuffle(source_colors)
    sources: list[dict] = []
    for r0_inner, shape, color, target_idx in zip(starts, source_shapes, source_colors, perm):
        sh, sw = _shape_dims(shape)
        r0 = r0_inner + 1
        max_corridor = w - divider_c - sw - 2
        if max_corridor < MIN_CORRIDOR_LEN:
            return None
        corridor_len = random.randint(MIN_CORRIDOR_LEN, max_corridor)
        c0 = divider_c + 1 + corridor_len
        sources.append(
            {
                "color": color,
                "shape": shape,
                "target_index": target_idx,
                "origin": (r0, c0),
            }
        )
    sources.sort(key=lambda item: item["origin"])
    return sources, perm


def _interval_to_cols(start: int, length: int) -> set[int]:
    return set(range(start, start + length))


def _add_barriers(grid: np.ndarray, src: dict, divider_c: int) -> dict | None:
    shape = src["shape"]
    sh, _sw = _shape_dims(shape)
    r0, c0 = src["origin"]
    corridor_cols = list(range(divider_c + 1, c0))
    if len(corridor_cols) < MIN_CORRIDOR_LEN:
        return None

    upper_row = r0 - 1
    lower_row = r0 + sh
    gate_len = random.randint(2, min(3, len(corridor_cols) - 2))
    gate_start = random.randint(corridor_cols[0], corridor_cols[-gate_len])
    both = _interval_to_cols(gate_start, gate_len)

    side_len = 2
    side_positions = []
    for start in corridor_cols:
        run = _interval_to_cols(start, side_len)
        if run.issubset(corridor_cols) and run.isdisjoint(both):
            side_positions.append(start)
    if not side_positions:
        return None
    side_start = random.choice(side_positions)
    side = _interval_to_cols(side_start, side_len)

    upper_cols = set(both)
    lower_cols = set(both)
    if random.random() < 0.5:
        upper_cols |= side
    else:
        lower_cols |= side

    free_cols = [c for c in corridor_cols if c not in upper_cols and c not in lower_cols]
    if len(free_cols) >= 2 and random.random() < 0.6:
        extra_start_candidates = []
        for start in free_cols:
            run = _interval_to_cols(start, 2)
            if run.issubset(corridor_cols) and run.isdisjoint(upper_cols | lower_cols):
                extra_start_candidates.append(start)
        if extra_start_candidates:
            extra = _interval_to_cols(random.choice(extra_start_candidates), 2)
            upper_cols |= extra
            lower_cols |= extra

    grid[upper_row, sorted(upper_cols)] = BARRIER_COLOR
    grid[lower_row, sorted(lower_cols)] = BARRIER_COLOR
    return {
        "upper_row": upper_row,
        "lower_row": lower_row,
        "upper_cols": upper_cols,
        "lower_cols": lower_cols,
    }


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], origin: tuple[int, int], color: int) -> np.ndarray:
    r0, c0 = origin
    coords = np.asarray([(r0 + dr, c0 + dc) for dr, dc in shape], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = color
    return coords


def _build_expected_output(inp: np.ndarray, templates: list[dict], sources: list[dict], divider_c: int) -> np.ndarray:
    out = inp.copy()
    for src in sources:
        coords = src["coords"]
        out[coords[:, 0], coords[:, 1]] = BACKGROUND
    for src in sources:
        tpl = templates[src["target_index"]]
        hole_r0, hole_c0 = tpl["hole_origin"]
        hole_coords = np.asarray([(hole_r0 + dr, hole_c0 + dc) for dr, dc in tpl["shape"]], dtype=int)
        out[hole_coords[:, 0], hole_coords[:, 1]] = src["color"]
        r0, c0 = src["origin"]
        sh, _sw = _shape_dims(src["shape"])
        for c in range(divider_c + 1, c0):
            if inp[r0 - 1, c] == BARRIER_COLOR and inp[r0 + sh, c] == BARRIER_COLOR:
                out[r0 : r0 + sh, c] = GATE_MARK_COLOR
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray, templates: list[dict], sources: list[dict], perm: tuple[int, int, int]) -> bool:
    if perm == (0, 1, 2):
        return False
    if not any(
        len(shape_group) >= 2 for shape_group in SHAPE_GROUPS.values()
    ):
        return False
    if np.array_equal(inp, out):
        return False

    shapes = [tpl["shape"] for tpl in templates]
    counts = Counter((len(shape),) + _shape_dims(shape) for shape in shapes)
    if not any(v >= 2 for v in counts.values()):
        return False

    for src in sources:
        meta = src["barriers"]
        both = meta["upper_cols"] & meta["lower_cols"]
        one_sided = meta["upper_cols"] ^ meta["lower_cols"]
        if not both or not one_sided:
            return False

        tpl = templates[src["target_index"]]
        hole_r0, hole_c0 = tpl["hole_origin"]
        hole_coords = np.asarray([(hole_r0 + dr, hole_c0 + dc) for dr, dc in tpl["shape"]], dtype=int)
        if set(map(tuple, hole_coords)) == set(map(tuple, src["coords"])):
            return False
        if not np.all(out[hole_coords[:, 0], hole_coords[:, 1]] == src["color"]):
            return False

    if int(np.count_nonzero(out == GATE_MARK_COLOR)) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shapes = _sample_shapes()
    tpl_layout = _choose_template_layout(shapes)
    if tpl_layout is None:
        return None
    divider_c, templates = tpl_layout

    max_source_w = max(_shape_dims(tpl["shape"])[1] for tpl in templates)
    min_width = divider_c + 1 + MIN_CORRIDOR_LEN + max_source_w + 2
    if min_width > 29:
        return None
    width = random.randint(max(24, min_width), 29)
    height_floor = max(22, max(tpl["bbox"][2] for tpl in templates) + 2)
    if height_floor > 29:
        return None
    height = random.randint(height_floor, 29)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:, divider_c] = DIVIDER_COLOR

    for tpl in templates:
        r0, c0, r1, c1 = tpl["bbox"]
        if r1 >= height:
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1] = tpl["template_color"]
        hole_r0, hole_c0 = tpl["hole_origin"]
        hole_coords = np.asarray([(hole_r0 + dr, hole_c0 + dc) for dr, dc in tpl["shape"]], dtype=int)
        grid[hole_coords[:, 0], hole_coords[:, 1]] = BACKGROUND

    source_layout = _choose_source_layout(height, width, divider_c, templates)
    if source_layout is None:
        return None
    sources, perm = source_layout

    for src in sources:
        coords = _paint_shape(grid, src["shape"], src["origin"], src["color"])
        src["coords"] = coords
        barriers = _add_barriers(grid, src, divider_c)
        if barriers is None:
            return None
        src["barriers"] = barriers

    expected = _build_expected_output(grid, templates, sources, divider_c)
    if not _semantic_checks(grid, expected, templates, sources, perm):
        return None

    try:
        observed = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
