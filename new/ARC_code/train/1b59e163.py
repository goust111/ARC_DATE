from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import components_mask, find_components, get_bg, in_bounds, paste, random_free_location
from functools import lru_cache
from itertools import product
import random


def _source_templates(g: np.ndarray, bg: int) -> dict[int, tuple[tuple[int, int], int, list[tuple[int, int]]]]:
    templates: dict[int, tuple[tuple[int, int], int, list[tuple[int, int]]]] = {}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    for comp in comps:
        coords = comp["coords"]
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        colors = sorted(set(vals))
        if len(colors) == 1:
            continue

        assert len(colors) == 2, "each source component must be one sprite color plus one marker color"
        counts = Counter(vals)
        marker_colors = [color for color in colors if counts[color] == 1]
        assert len(marker_colors) == 1, "the marker is the unique one-pixel color in its source component"

        marker_color = int(marker_colors[0])
        sprite_color = int(next(color for color in colors if color != marker_color))
        marker_coords = [
            (int(r), int(c))
            for r, c in coords
            if int(g[int(r), int(c)]) == marker_color
        ]
        assert len(marker_coords) == 1
        source_marker = marker_coords[0]

        offsets: list[tuple[int, int]] = []
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if int(g[rr, cc]) == sprite_color:
                offsets.append((rr - source_marker[0], cc - source_marker[1]))
        assert offsets, "source component must contain sprite pixels"
        assert marker_color not in templates, "each marker color must define one source sprite"
        templates[marker_color] = (source_marker, sprite_color, offsets)

    assert templates, "expected at least one marked source sprite"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each marked sprite to every standalone marker of the same color and erase the original sources.

    Input structure:
    - The background is the dominant color.
    - Each source is an 8-connected component containing a monochrome sprite and one marker pixel of another color.
    - Other isolated marker pixels may be destinations; marker colors with no source sprite are distractors.

    Rule:
    - Clear the grid to background.
    - For every source marker color, place the sprite at each other marker of that color using the source marker as the anchor.
    - Keep each destination marker pixel in place; do not keep source sprites, source markers, or distractor markers.

    Key insight:
    - A colored marker is not part of the sprite geometry; it is the anchor that says where same-colored copies of that sprite should be stamped.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    templates = _source_templates(g, bg)

    for marker_color, (source_marker, sprite_color, offsets) in templates.items():
        all_markers = [tuple(map(int, coord)) for coord in np.argwhere(g == int(marker_color))]
        targets = [coord for coord in all_markers if coord != source_marker]
        assert targets, "a source marker color must have at least one destination marker"

        for tr, tc in targets:
            assert int(out[tr, tc]) == bg, "destination markers must not overlap earlier stamps"
            out[tr, tc] = int(marker_color)
            for dr, dc in offsets:
                rr = int(tr + dr)
                cc = int(tc + dc)
                assert in_bounds(out.shape, rr, cc), "stamped sprite must stay inside the grid"
                assert int(out[rr, cc]) in (bg, int(sprite_color)), "stamped sprites must not collide"
                out[rr, cc] = int(sprite_color)

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
BG = 4
GRID_SHAPE = (18, 18)
DRAW_COLORS = (0, 1, 2, 3, 6, 7, 8, 9)
DISTRACTOR_COLOR = 5

# Source sprite geometry.
SOURCE_COUNTS = (2, 3)
SPRITE_HEIGHTS = (3, 4)
SPRITE_WIDTHS = (3, 5)
SPRITE_PIXEL_RANGE = range(3, 9)
SPRITE_CONNECTIVITY = 8
MARKER_ADJACENCY_DISTANCE = 1

# Destination and placement constraints.
DESTINATION_COUNT_OPTIONS = (1, 2, 3)
DESTINATION_TOTAL_RANGE = range(3, 7)
MARKER_SPACING_RADIUS = 1
PLACEMENT_PADDING = 1
PLACEMENT_PADDING_CONNECTIVITY = 8

# Distractor and retry controls.
DISTRACTOR_COUNTS = (2, 3)
GENERATION_RETRIES = 300


@lru_cache(maxsize=None)
def _sprite_options(h: int, w: int, marker_r: int) -> tuple[tuple[int, tuple[tuple[tuple[int, int], ...], ...]], ...]:
    marker_c = w // 2
    units: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        if r != marker_r:
            units.append(((r, marker_c),))
        for d in range(1, marker_c + 1):
            units.append(((r, marker_c - d), (r, marker_c + d)))

    grouped: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    marker = (marker_r, marker_c)
    for bits in range(1, 1 << len(units)):
        cells: list[tuple[int, int]] = []
        for i, unit in enumerate(units):
            if bits & (1 << i):
                cells.extend(unit)

        n = len(cells)
        if n not in SPRITE_PIXEL_RANGE:
            continue

        all_cells = cells + [marker]
        rows = [r for r, _ in all_cells]
        cols = [c for _, c in all_cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        if not any(max(abs(r - marker_r), abs(c - marker_c)) == MARKER_ADJACENCY_DISTANCE for r, c in cells):
            continue

        mask = np.zeros((h, w), dtype=bool)
        rr, cc = np.array(cells).T
        mask[rr, cc] = True
        if len(components_mask(mask, connectivity=SPRITE_CONNECTIVITY)) != 1:
            continue

        offsets = tuple(sorted((r - marker_r, c - marker_c) for r, c in cells))
        grouped.setdefault(n, []).append(offsets)

    return tuple((n, tuple(opts)) for n, opts in sorted(grouped.items()))


def _make_source(marker_color: int, sprite_color: int) -> dict:
    while True:
        h = random.choice(SPRITE_HEIGHTS)
        w = random.choice(SPRITE_WIDTHS)
        marker_r = random.randrange(h)
        groups = dict(_sprite_options(h, w, marker_r))
        if groups:
            break

    pixel_count = random.choice(list(groups))
    offsets = random.choice(groups[pixel_count])
    marker_c = w // 2
    patch = np.full((h, w), BG, dtype=int)
    patch[marker_r, marker_c] = int(marker_color)
    for dr, dc in offsets:
        patch[marker_r + dr, marker_c + dc] = int(sprite_color)

    return {
        "marker_color": int(marker_color),
        "sprite_color": int(sprite_color),
        "patch": patch,
        "anchor": (marker_r, marker_c),
        "destinations": [],
    }


def _make_sources() -> list[dict]:
    n_sources = random.choice(SOURCE_COUNTS)
    marker_colors = random.sample(DRAW_COLORS, n_sources)
    sprite_palette = [color for color in DRAW_COLORS if color not in marker_colors]
    return [_make_source(marker_color, random.choice(sprite_palette)) for marker_color in marker_colors]


def _destination_counts(n_sources: int) -> list[int]:
    choices = [counts for counts in product(DESTINATION_COUNT_OPTIONS, repeat=n_sources) if sum(counts) in DESTINATION_TOTAL_RANGE]
    return list(random.choice(choices))


def _marker_is_spaced(mask: np.ndarray, r: int, c: int) -> bool:
    rows = slice(max(0, r - MARKER_SPACING_RADIUS), min(mask.shape[0], r + MARKER_SPACING_RADIUS + 1))
    cols = slice(max(0, c - MARKER_SPACING_RADIUS), min(mask.shape[1], c + MARKER_SPACING_RADIUS + 1))
    return not bool(mask[rows, cols].any())


def _place_output_stamp(out: np.ndarray, dest_mask: np.ndarray, source: dict) -> tuple[int, int] | None:
    """Place one output stamp and return its marker position, or None."""
    patch = source["patch"]
    anchor_r, anchor_c = source["anchor"]
    h, w = patch.shape
    patch_mask = patch != BG
    candidates: list[tuple[int, int]] = []

    for r0 in range(out.shape[0] - h + 1):
        for c0 in range(out.shape[1] - w + 1):
            if bool(((out[r0 : r0 + h, c0 : c0 + w] != BG) & patch_mask).any()):
                continue
            marker_r = r0 + anchor_r
            marker_c = c0 + anchor_c
            if not _marker_is_spaced(dest_mask, marker_r, marker_c):
                continue
            candidates.append((r0, c0))

    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    paste(out, patch, r0, c0, transparent=BG)
    marker_pos = (r0 + anchor_r, c0 + anchor_c)
    dest_mask[marker_pos] = True
    return marker_pos


def _render_output_stamps(out: np.ndarray, sources: list[dict]) -> bool:
    """Mutate output and sources with all destination stamp placements."""
    dest_mask = np.zeros(GRID_SHAPE, dtype=bool)
    for source, count in zip(sources, _destination_counts(len(sources))):
        for _ in range(count):
            marker_pos = _place_output_stamp(out, dest_mask, source)
            if marker_pos is None:
                return False
            source["destinations"].append(marker_pos)
    return True


def _free_location(grid: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    try:
        return random_free_location(grid, patch, transparent=BG, padding=PLACEMENT_PADDING, padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY)
    except ValueError:
        return None


def _render_input_grid(sources: list[dict]) -> np.ndarray | None:
    """Build input from destination markers, source components, and distractors."""
    inp = np.full(GRID_SHAPE, BG, dtype=int)
    for source in sources:
        marker_color = source["marker_color"]
        for r, c in source["destinations"]:
            inp[r, c] = marker_color

    for source in sources:
        loc = _free_location(inp, source["patch"])
        if loc is None:
            return None
        r0, c0 = loc
        paste(inp, source["patch"], r0, c0, transparent=BG)

    point = np.array([[DISTRACTOR_COLOR]], dtype=int)
    for _ in range(random.choice(DISTRACTOR_COUNTS)):
        loc = _free_location(inp, point)
        if loc is None:
            return None
        r, c = loc
        inp[r, c] = DISTRACTOR_COLOR
    return inp


def _gen_attempt() -> np.ndarray | None:
    sources = _make_sources()
    out = np.full(GRID_SHAPE, BG, dtype=int)
    if not _render_output_stamps(out, sources):
        return None

    inp = _render_input_grid(sources)
    if inp is None:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 1b59e163 input")
