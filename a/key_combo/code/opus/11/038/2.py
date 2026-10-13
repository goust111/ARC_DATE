from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
EMITTER = 7
TEMPLATE_COLORS = (1, 5, 6, 8, 9)
SOURCE_COLORS = (2, 3, 4)
MAX_ATTEMPTS = 400


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class MonoComponent:
    color: int
    coords: tuple[Coord, ...]
    bbox: BBox


@dataclass(frozen=True)
class TemplateInfo:
    color: int
    coords: tuple[Coord, ...]
    bbox: BBox
    gap_coords: tuple[Coord, ...]
    gap_sig: frozenset[Coord]


@dataclass(frozen=True)
class SourceInfo:
    color: int
    coords: tuple[Coord, ...]
    bbox: BBox
    emitter: Coord
    shape_sig: frozenset[Coord]


@dataclass(frozen=True)
class PlannedTemplate:
    color: int
    bbox: BBox
    gap_coords: tuple[Coord, ...]
    gap_sig: frozenset[Coord]


@dataclass(frozen=True)
class PlannedSource:
    color: int
    coords: tuple[Coord, ...]
    emitter: Coord
    target_index: int
    should_match: bool


SHAPE_LIBRARY: tuple[frozenset[Coord], ...] = (
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (1, 1), (2, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    frozenset({(0, 0), (1, 0), (1, 1), (2, 1)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)}),
    frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 0)}),
)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[Coord] | tuple[Coord, ...]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[Coord] | tuple[Coord, ...]) -> frozenset[Coord]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _shape_dims(sig: frozenset[Coord]) -> tuple[int, int]:
    coords = list(sig)
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _most_common_value(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_monochrome_components(grid: np.ndarray, ignored: set[int]) -> tuple[list[MonoComponent], dict[Coord, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[MonoComponent] = []
    coord_to_id: dict[Coord, int] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignored:
                continue
            seen[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_t = tuple(sorted(coords))
            comp = MonoComponent(color=color, coords=coords_t, bbox=_bbox(coords_t))
            comp_id = len(components)
            components.append(comp)
            for coord in coords_t:
                coord_to_id[coord] = comp_id
    return components, coord_to_id


def _template_from_component(grid: np.ndarray, comp: MonoComponent) -> TemplateInfo | None:
    r0, c0, r1, c1 = comp.bbox
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    color = int(comp.color)
    if np.any((patch != BG) & (patch != color)):
        return None
    gap_mask = patch == BG
    if not np.any(gap_mask):
        return None
    if np.any(gap_mask[0, :]) or np.any(gap_mask[-1, :]) or np.any(gap_mask[:, 0]) or np.any(gap_mask[:, -1]):
        return None

    seen = np.zeros_like(gap_mask, dtype=bool)
    groups: list[list[Coord]] = []
    for rr in range(gap_mask.shape[0]):
        for cc in range(gap_mask.shape[1]):
            if not gap_mask[rr, cc] or seen[rr, cc]:
                continue
            seen[rr, cc] = True
            q: deque[Coord] = deque([(rr, cc)])
            part: list[Coord] = []
            while q:
                pr, pc = q.popleft()
                part.append((pr, pc))
                for nr, nc in _neighbors4(pr, pc):
                    if 0 <= nr < gap_mask.shape[0] and 0 <= nc < gap_mask.shape[1] and gap_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            groups.append(part)
    if len(groups) != 1:
        return None

    gap_coords = tuple(sorted((r0 + rr, c0 + cc) for rr, cc in groups[0]))
    if not (3 <= len(gap_coords) <= 5):
        return None
    return TemplateInfo(
        color=color,
        coords=comp.coords,
        bbox=comp.bbox,
        gap_coords=gap_coords,
        gap_sig=_normalize(gap_coords),
    )


def _parse_scene(grid: np.ndarray) -> tuple[list[TemplateInfo], list[SourceInfo]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _most_common_value(g) == BG, "background must be black"

    components, coord_to_id = _find_monochrome_components(g, ignored={BG, EMITTER})
    emitters = sorted((int(r), int(c)) for r, c in np.argwhere(g == EMITTER))
    assert emitters, "expected at least one orange emitter"

    source_ids: list[int] = []
    for er, ec in emitters:
        assert ec + 1 < g.shape[1], "emitter must have room for a source on its right"
        assert int(g[er, ec + 1]) not in {BG, EMITTER}, "emitter must touch its source on the right"
        comp_id = coord_to_id[(er, ec + 1)]
        source_ids.append(comp_id)
    assert len(set(source_ids)) == len(source_ids), "each emitter must point to a distinct source component"

    sources: list[SourceInfo] = []
    used_source_ids = set(source_ids)
    for emitter, comp_id in zip(emitters, source_ids):
        comp = components[comp_id]
        sources.append(
            SourceInfo(
                color=int(comp.color),
                coords=comp.coords,
                bbox=comp.bbox,
                emitter=emitter,
                shape_sig=_normalize(comp.coords),
            )
        )

    templates: list[TemplateInfo] = []
    for comp_id, comp in enumerate(components):
        if comp_id in used_source_ids:
            continue
        template = _template_from_component(g, comp)
        assert template is not None, "all non-source components must be templates"
        templates.append(template)

    assert 2 <= len(templates) <= 3, "expected 2 or 3 templates"
    assert len(sources) == len(templates), "source/template counts must match"
    assert len({t.color for t in templates}) == len(templates), "template colors must be distinct"
    assert len({s.color for s in sources}) == len(sources), "source colors must be distinct"
    assert len({t.gap_sig for t in templates}) == len(templates), "template gaps must be shape-distinct"
    return templates, sources


def _visible_templates(source: SourceInfo, templates: list[TemplateInfo]) -> list[TemplateInfo]:
    er, ec = source.emitter
    candidates = [
        tmpl
        for tmpl in templates
        if tmpl.bbox[0] <= er <= tmpl.bbox[2] and tmpl.bbox[1] < ec
    ]
    return sorted(candidates, key=lambda t: t.bbox[3], reverse=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot each emitter's leftward probe across template blocks and fill the first complementary gap it finds.

    Input structure:
    - The background is black.
    - Several colored rectangular template blocks lie to the left; each block is a single-color rectangle with one internal black gap.
    - An equal number of colored source shapes lie on the right, and each source has one orange emitter pixel immediately to its left on one source row.

    Rule:
    - For every emitter, trace an invisible ray leftward along the emitter row and inspect template blocks whose row span contains that ray.
    - At each encountered template, normalize the source shape and the template's black gap by their own bounding-box top-left corners and compare the coordinate sets.
    - If they match, repaint the gap with the source color, erase that source shape and its emitter, and stop that ray; otherwise continue left to the next template. If no visible template matches, leave that source and emitter unchanged.

    Key insight:
    - The orange pixel does not mark a destination directly; it fixes a search row, and only the first template on that row whose missing shape exactly matches the source gets filled.
    """
    g = np.asarray(grid, dtype=int)
    templates, sources = _parse_scene(g)

    global_match_counts = {source.emitter: sum(source.shape_sig == tmpl.gap_sig for tmpl in templates) for source in sources}
    assert set(global_match_counts.values()) == {1}, "each source shape must globally match exactly one template gap"

    out = g.copy()
    for source in sorted(sources, key=lambda s: (s.emitter[0], s.emitter[1])):
        for tmpl in _visible_templates(source, templates):
            if source.shape_sig != tmpl.gap_sig:
                continue
            for r, c in tmpl.gap_coords:
                out[r, c] = int(source.color)
            for r, c in source.coords:
                out[r, c] = BG
            er, ec = source.emitter
            out[er, ec] = BG
            break
    return out


def _choose_shapes(count: int) -> list[frozenset[Coord]]:
    return random.sample(list(SHAPE_LIBRARY), count)


def _paint_template(grid: np.ndarray, color: int, bbox: BBox, gap_coords: tuple[Coord, ...]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    for r, c in gap_coords:
        grid[r, c] = BG


def _paint_source(grid: np.ndarray, color: int, coords: tuple[Coord, ...], emitter: Coord) -> None:
    for r, c in coords:
        grid[r, c] = color
    er, ec = emitter
    grid[er, ec] = EMITTER


def _pick_distinct_rows(candidates: list[list[int]]) -> list[int] | None:
    chosen: list[int] = []
    used: set[int] = set()
    for options in candidates:
        avail = [row for row in options if row not in used]
        if not avail:
            return None
        row = random.choice(avail)
        chosen.append(row)
        used.add(row)
    return chosen


def _planned_layout(shapes: list[frozenset[Coord]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int] | None:
    count = len(shapes)
    widths = []
    heights = []
    for sig in shapes:
        sh_h, sh_w = _shape_dims(sig)
        widths.append(max(sh_w + 2, random.randint(4, 5)))
        heights.append(max(sh_h + 2, 5))

    col_starts = [random.randint(1, 2)]
    for i in range(1, count):
        col_starts.append(col_starts[-1] + widths[i - 1] + random.randint(1, 2))

    overlaps = [random.randint(1, 2) for _ in range(count - 1)]
    row_starts = [random.randint(1, 3)]
    for i in range(1, count):
        row_starts.append(row_starts[-1] + heights[i - 1] - overlaps[i - 1])

    last_right = col_starts[-1] + widths[-1] - 1
    corridor = random.randint(2, 4)
    width = last_right + 1 + corridor + 7 + random.randint(0, 1)
    height = max(r + h for r, h in zip(row_starts, heights)) + random.randint(2, 4)

    if width > 30 or height > 30:
        return None
    return list(zip(row_starts, heights)), list(zip(col_starts, widths)), height, width


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.choice((2, 3))
    shapes = _choose_shapes(count)
    row_layout, col_layout, height, width = _planned_layout(shapes) or (None, None, None, None)
    if row_layout is None:
        return None

    template_colors = random.sample(TEMPLATE_COLORS, count)
    source_colors = random.sample(SOURCE_COLORS, count)
    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()

    templates: list[PlannedTemplate] = []
    row_ranges: list[set[int]] = []
    for idx, sig in enumerate(shapes):
        sh_h, sh_w = _shape_dims(sig)
        r0, h = row_layout[idx]
        c0, w = col_layout[idx]
        max_off_r = h - sh_h - 1
        max_off_c = w - sh_w - 1
        if max_off_r < 1 or max_off_c < 1:
            return None
        off_r = random.randint(1, max_off_r)
        off_c = random.randint(1, max_off_c)
        gap_coords = tuple(sorted((r0 + off_r + dr, c0 + off_c + dc) for dr, dc in sig))
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        _paint_template(grid, template_colors[idx], bbox, gap_coords)
        _paint_template(expected, template_colors[idx], bbox, gap_coords)
        templates.append(
            PlannedTemplate(
                color=template_colors[idx],
                bbox=bbox,
                gap_coords=gap_coords,
                gap_sig=_normalize(gap_coords),
            )
        )
        row_ranges.append(set(range(r0, r0 + h)))

    rightmost_only = sorted(row_ranges[-1] - (row_ranges[-2] if count >= 2 else set()))
    if not rightmost_only:
        return None
    overlap_rows = [sorted(row_ranges[i] & row_ranges[i + 1]) for i in range(count - 1)]
    if any(not rows for rows in overlap_rows):
        return None

    no_match_left = False
    target_row_options: list[list[int]] = []
    for target_index in range(count):
        if target_index == count - 1:
            target_row_options.append(rightmost_only)
        else:
            target_row_options.append(overlap_rows[target_index])

    chosen_rows = _pick_distinct_rows(target_row_options)
    if chosen_rows is None:
        return None

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[grid != BG] = True
    sources: list[PlannedSource] = []
    for target_index in range(count):
        sig = shapes[target_index]
        rows = [r for r, _c in sig]
        cols = [c for _r, c in sig]
        shape_h = max(rows) + 1
        shape_w = max(cols) + 1
        anchor_rows = [r for r, c in sig if c == 0]
        anchor_r = random.choice(anchor_rows)
        if target_index == 0 and no_match_left:
            source_row = max(t.bbox[2] for t in templates) + 1 + anchor_r
        else:
            source_row = chosen_rows[target_index]
        top = source_row - anchor_r
        if top < 0:
            return None
        if top + shape_h > height:
            extra = top + shape_h - height
            grid = np.pad(grid, ((0, extra), (0, 0)), constant_values=BG)
            expected = np.pad(expected, ((0, extra), (0, 0)), constant_values=BG)
            occupied = np.pad(occupied, ((0, extra), (0, 0)), constant_values=False)
            height += extra
        lane_offsets = [0, 3]
        random.shuffle(lane_offsets)
        placed = False
        emitter = (-1, -1)
        coords: tuple[Coord, ...] = ()
        for lane_offset in lane_offsets:
            left = width - 4 - lane_offset
            emitter_try = (source_row, left - 1)
            coords_try = tuple(sorted((top + dr, left + dc) for dr, dc in sig))
            if emitter_try[1] < 0:
                continue
            if (source_row, left) not in coords_try:
                continue
            if any(occupied[r, c] for r, c in coords_try):
                continue
            if occupied[emitter_try]:
                continue
            emitter = emitter_try
            coords = coords_try
            placed = True
            break
        if not placed:
            return None

        should_match = not (target_index == 0 and no_match_left)
        _paint_source(grid, source_colors[target_index], coords, emitter)
        for r, c in coords:
            occupied[r, c] = True
        occupied[emitter] = True

        sources.append(
            PlannedSource(
                color=source_colors[target_index],
                coords=coords,
                emitter=emitter,
                target_index=target_index,
                should_match=should_match,
            )
        )
        if should_match:
            for r, c in coords:
                expected[r, c] = BG
            expected[emitter] = BG
            for r, c in templates[target_index].gap_coords:
                expected[r, c] = source_colors[target_index]

    if np.array_equal(grid, expected):
        return None

    parsed_templates, parsed_sources = _parse_scene(grid)
    if len(parsed_templates) != count or len(parsed_sources) != count:
        return None

    visible_match_counts: list[int] = []
    skip_then_match = 0
    first_match = 0
    no_match = 0
    for source in parsed_sources:
        global_matches = [tmpl for tmpl in parsed_templates if tmpl.gap_sig == source.shape_sig]
        if len(global_matches) != 1:
            return None
        visibles = _visible_templates(source, parsed_templates)
        visible_matches = [idx for idx, tmpl in enumerate(visibles) if tmpl.gap_sig == source.shape_sig]
        visible_match_counts.append(len(visible_matches))
        if not visible_matches:
            no_match += 1
            continue
        if visible_matches[0] == 0:
            first_match += 1
        else:
            skip_then_match += 1

    if skip_then_match < 1 or first_match < 1:
        return None
    if no_match_left:
        if no_match < 1:
            return None
    elif no_match != 0:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def _build_no_match_instance() -> tuple[np.ndarray, np.ndarray] | None:
    count = 3
    shapes = _choose_shapes(count)
    row_layout, col_layout, height, width = _planned_layout(shapes) or (None, None, None, None)
    if row_layout is None:
        return None

    template_colors = random.sample(TEMPLATE_COLORS, count)
    source_colors = random.sample(SOURCE_COLORS, count)
    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()

    templates: list[PlannedTemplate] = []
    row_ranges: list[set[int]] = []
    for idx, sig in enumerate(shapes):
        sh_h, sh_w = _shape_dims(sig)
        r0, h = row_layout[idx]
        c0, w = col_layout[idx]
        max_off_r = h - sh_h - 1
        max_off_c = w - sh_w - 1
        if max_off_r < 1 or max_off_c < 1:
            return None
        off_r = random.randint(1, max_off_r)
        off_c = random.randint(1, max_off_c)
        gap_coords = tuple(sorted((r0 + off_r + dr, c0 + off_c + dc) for dr, dc in sig))
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        _paint_template(grid, template_colors[idx], bbox, gap_coords)
        _paint_template(expected, template_colors[idx], bbox, gap_coords)
        templates.append(
            PlannedTemplate(
                color=template_colors[idx],
                bbox=bbox,
                gap_coords=gap_coords,
                gap_sig=_normalize(gap_coords),
            )
        )
        row_ranges.append(set(range(r0, r0 + h)))

    overlap12 = sorted(row_ranges[1] & row_ranges[2])
    rightmost_only = sorted(row_ranges[2] - row_ranges[1])
    if not overlap12 or not rightmost_only:
        return None

    skip_row = random.choice(overlap12)
    first_row = random.choice(rightmost_only)
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[grid != BG] = True

    lane_for_target = {0: 0, 1: 3, 2: 0}
    source_rows = {1: skip_row, 2: first_row}

    for target_index in range(count):
        sig = shapes[target_index]
        rows = [r for r, _c in sig]
        cols = [c for _r, c in sig]
        shape_h = max(rows) + 1
        shape_w = max(cols) + 1
        anchor_rows = [r for r, c in sig if c == 0]
        anchor_r = random.choice(anchor_rows)

        if target_index == 0:
            source_row = max(t.bbox[2] for t in templates) + 1 + anchor_r
        else:
            source_row = source_rows[target_index]
        top = source_row - anchor_r
        if top < 0:
            return None
        if top + shape_h > height:
            extra = top + shape_h - height
            grid = np.pad(grid, ((0, extra), (0, 0)), constant_values=BG)
            expected = np.pad(expected, ((0, extra), (0, 0)), constant_values=BG)
            occupied = np.pad(occupied, ((0, extra), (0, 0)), constant_values=False)
            height += extra

        left = width - 4 - lane_for_target[target_index]
        emitter = (source_row, left - 1)
        coords = tuple(sorted((top + dr, left + dc) for dr, dc in sig))
        if emitter[1] < 0 or (source_row, left) not in coords:
            return None
        if any(occupied[r, c] for r, c in coords) or occupied[emitter]:
            return None

        _paint_source(grid, source_colors[target_index], coords, emitter)
        for r, c in coords:
            occupied[r, c] = True
        occupied[emitter] = True

        if target_index == 0:
            _paint_source(expected, source_colors[target_index], coords, emitter)
        else:
            for r, c in coords:
                expected[r, c] = BG
            expected[emitter] = BG
            for r, c in templates[target_index].gap_coords:
                expected[r, c] = source_colors[target_index]

    parsed_templates, parsed_sources = _parse_scene(grid)
    if len(parsed_templates) != count or len(parsed_sources) != count:
        return None

    skip_then_match = 0
    first_match = 0
    no_match = 0
    for source in parsed_sources:
        global_matches = [tmpl for tmpl in parsed_templates if tmpl.gap_sig == source.shape_sig]
        if len(global_matches) != 1:
            return None
        visibles = _visible_templates(source, parsed_templates)
        visible_matches = [idx for idx, tmpl in enumerate(visibles) if tmpl.gap_sig == source.shape_sig]
        if not visible_matches:
            no_match += 1
            continue
        if visible_matches[0] == 0:
            first_match += 1
        else:
            skip_then_match += 1

    if skip_then_match < 1 or first_match < 1 or no_match < 1:
        return None
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        if random.random() < 0.35:
            built = _build_no_match_instance()
        else:
            built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ray/complement puzzle instance")
