from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

KEY_COLORS = (BLUE, GREEN, YELLOW, PINK, ORANGE, MAROON)
MAX_ATTEMPTS = 400

# Normalized polyomino-like shapes, compared up to translation only.
SHAPE_POOL: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (
        ((0, 0),),
    ),
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
    5: (
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ),
}


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = list(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    r0, c0, _, _ = _bbox(pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    area = dst[top : top + h, left : left + w]
    assert np.all((area == BG) | (patch == BG))
    area[patch != BG] = patch[patch != BG]


def _analyze_grid(grid: np.ndarray) -> dict[str, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    comps = _component_coords(g != BG)
    legend: list[dict[str, object]] = []
    panels: list[dict[str, object]] = []
    for coords in comps:
        box = _bbox(coords)
        patch = _crop(g, box)
        colors = {int(g[r, c]) for r, c in coords}
        if TEAL in colors or RED in colors:
            assert np.all(patch != BG), "panel must be a solid rectangle with embedded colors"
            reds = [(r, c) for r, c in coords if int(g[r, c]) == RED]
            greys = [(r, c) for r, c in coords if int(g[r, c]) == GREY]
            teals = [(r, c) for r, c in coords if int(g[r, c]) == TEAL]
            assert reds and greys and teals, "every panel needs red, grey, and teal pixels"
            panels.append(
                {
                    "bbox": box,
                    "shape": _normalize(reds),
                    "red_count": len(reds),
                    "grey_count": len(greys),
                }
            )
        else:
            key_cells = [(r, c) for r, c in coords if int(g[r, c]) != GREY]
            grey_cells = [(r, c) for r, c in coords if int(g[r, c]) == GREY]
            assert len(key_cells) == 1 and len(grey_cells) >= 3
            kr, kc = key_cells[0]
            legend.append(
                {
                    "bbox": box,
                    "shape": _normalize(grey_cells),
                    "count": len(grey_cells),
                    "key_color": int(g[kr, kc]),
                }
            )
    return {"legend": legend, "panels": panels}


def _winner_and_target(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    info = _analyze_grid(grid)
    legend = info["legend"]
    panels = info["panels"]
    assert 2 <= len(legend) <= 3
    assert 3 <= len(panels) <= 5

    best = max(int(panel["red_count"]) for panel in panels)
    winners = [panel for panel in panels if int(panel["red_count"]) == best]
    assert len(winners) == 1, "expected a unique panel with the most red markers"
    winner = winners[0]

    matches = [entry for entry in legend if entry["shape"] == winner["shape"]]
    assert len(matches) == 1, "winner red shape must match exactly one legend glyph"
    return winner["bbox"], int(matches[0]["key_color"])


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the unique panel with the most red markers, decode its red-marker shape via the legend, and recolor the panel's grey pixels.

    Input structure
    - The background is black.
    - Several solid teal rectangular panels contain embedded red marker pixels and grey pixels.
    - Separate legend entries each contain one colored key pixel and one adjacent grey glyph.

    Rule
    - Count red pixels in every panel and select the unique panel with the largest red count.
    - Normalize the selected panel's red-marker coordinates by their own minimal bounding box.
    - Find the unique legend glyph with the same normalized grey shape and read its key color.
    - Crop the selected panel and replace only its grey pixels with that decoded color.

    Key insight
    - The same red pixels play two roles at once: their count chooses the panel, and their layout decodes the recoloring color through the legend.
    """
    g = np.asarray(grid, dtype=int)
    box, color = _winner_and_target(g)
    out = _crop(g, box)
    out[out == GREY] = color
    return out


def _choose_shape(
    rng: random.Random,
    count: int,
    *,
    exclude: set[tuple[tuple[int, int], ...]] | None = None,
) -> tuple[tuple[int, int], ...]:
    candidates = list(SHAPE_POOL[count])
    if exclude:
        filtered = [shape for shape in candidates if shape not in exclude]
        if filtered:
            candidates = filtered
    return rng.choice(candidates)


def _build_legend_patch(
    rng: random.Random, shape: tuple[tuple[int, int], ...], key_color: int
) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w + 1), dtype=int)
    left_touch_rows = sorted({r for r, c in shape if c == 0})
    patch[rng.choice(left_touch_rows), 0] = key_color
    for r, c in shape:
        patch[r, c + 1] = GREY
    return patch


def _build_panel_patch(
    rng: random.Random,
    shape: tuple[tuple[int, int], ...],
    panel_count: int,
) -> np.ndarray:
    sh, sw = _shape_dims(shape)
    max_h = 7 if panel_count <= 3 else 6
    max_w = 7 if panel_count == 3 else 6 if panel_count == 4 else 5
    min_h = max(4, sh + 1)
    min_w = max(4, sw + 1)
    if min_h > max_h or min_w > max_w:
        raise ValueError("shape does not fit panel bounds")
    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    area = h * w
    if area < len(shape) + 3:
        raise ValueError("panel too small")

    patch = np.full((h, w), TEAL, dtype=int)
    off_r = rng.randint(0, h - sh)
    off_c = rng.randint(0, w - sw)
    red_cells = {(off_r + r, off_c + c) for r, c in shape}
    for r, c in red_cells:
        patch[r, c] = RED

    remaining = [(r, c) for r in range(h) for c in range(w) if (r, c) not in red_cells]
    max_grey = min(len(remaining) - 1, max(2, area // 3))
    grey_count = rng.randint(1 if max_grey == 1 else 2, max_grey)
    for r, c in rng.sample(remaining, grey_count):
        patch[r, c] = GREY
    return patch


def _build_case(rng: random.Random) -> np.ndarray:
    panel_count = rng.randint(3, 5)
    winner_count = 5 if panel_count == 5 else rng.choice((4, 5))
    legend_count = 3 if rng.random() < 0.75 else 2

    winner_shape = _choose_shape(rng, winner_count)
    decoy_shape = _choose_shape(rng, winner_count, exclude={winner_shape})

    support_count = None
    support_shape = None
    if legend_count == 3:
        support_count = 3 if winner_count == 4 else rng.choice((3, 4))
        support_shape = _choose_shape(rng, support_count)

    legend_shapes = [winner_shape, decoy_shape]
    if support_shape is not None:
        legend_shapes.append(support_shape)

    legend_colors = rng.sample(KEY_COLORS, len(legend_shapes))
    legend_entries = list(zip(legend_shapes, legend_colors))
    rng.shuffle(legend_entries)

    other_counts: list[int] = []
    if support_count is not None:
        other_counts.append(support_count)
    remaining_pool = [count for count in range(1, winner_count) if count not in other_counts]
    other_counts.extend(rng.sample(remaining_pool, panel_count - 1 - len(other_counts)))
    rng.shuffle(other_counts)

    winner_index = rng.randrange(panel_count)
    support_index = None
    if support_count is not None:
        candidates = [idx for idx in range(panel_count) if idx != winner_index]
        support_index = rng.choice(candidates)

    panel_specs: list[dict[str, object] | None] = [None] * panel_count
    panel_specs[winner_index] = {"count": winner_count, "shape": winner_shape}
    if support_index is not None:
        panel_specs[support_index] = {"count": support_count, "shape": support_shape}

    for idx in range(panel_count):
        if panel_specs[idx] is not None:
            continue
        count = other_counts.pop()
        if support_index is not None and count == support_count:
            count = other_counts.pop()
        excludes = {shape for shape, _ in legend_entries if len(shape) == count}
        shape = _choose_shape(rng, count, exclude=excludes)
        panel_specs[idx] = {"count": count, "shape": shape}

    assert all(spec is not None for spec in panel_specs)
    panel_patches = [
        _build_panel_patch(rng, spec["shape"], panel_count) for spec in panel_specs if spec is not None
    ]
    legend_patches = [_build_legend_patch(rng, shape, color) for shape, color in legend_entries]

    panel_gaps = [rng.randint(1, 2) for _ in range(panel_count - 1)]
    core_width = sum(p.shape[1] for p in panel_patches) + sum(panel_gaps)
    left_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 1)
    width = core_width + left_margin + right_margin
    if width > 30:
        raise ValueError("canvas too wide")

    legend_gap_rows = [rng.randint(1, 2) for _ in range(len(legend_patches) - 1)]
    legend_height = sum(p.shape[0] for p in legend_patches) + sum(legend_gap_rows)
    panel_top_offsets = [rng.randint(0, 2) for _ in range(panel_count)]
    lower_height = max(offset + patch.shape[0] for offset, patch in zip(panel_top_offsets, panel_patches))
    sep_rows = rng.randint(2, 3)
    height = legend_height + sep_rows + lower_height + rng.randint(1, 2)
    if height > 30:
        raise ValueError("canvas too tall")

    legend_width = max(p.shape[1] for p in legend_patches)
    if legend_width > width:
        raise ValueError("legend wider than canvas")

    grid = np.zeros((height, width), dtype=int)

    row = 0
    for idx, patch in enumerate(legend_patches):
        left = rng.randint(0, width - patch.shape[1])
        _paste(grid, patch, row, left)
        row += patch.shape[0]
        if idx < len(legend_gap_rows):
            row += legend_gap_rows[idx]

    panel_base = legend_height + sep_rows
    col = left_margin
    for idx, patch in enumerate(panel_patches):
        top = panel_base + panel_top_offsets[idx]
        _paste(grid, patch, top, col)
        col += patch.shape[1]
        if idx < len(panel_gaps):
            col += panel_gaps[idx]

    return grid


def _validate_generated_grid(grid: np.ndarray) -> None:
    info = _analyze_grid(grid)
    legend = info["legend"]
    panels = info["panels"]
    assert 2 <= len(legend) <= 3
    assert 3 <= len(panels) <= 5
    assert len({entry["shape"] for entry in legend}) == len(legend)

    winner_count = max(int(panel["red_count"]) for panel in panels)
    winners = [panel for panel in panels if int(panel["red_count"]) == winner_count]
    assert len(winners) == 1
    winner = winners[0]

    shape_matches = [entry for entry in legend if entry["shape"] == winner["shape"]]
    assert len(shape_matches) == 1
    assert sum(1 for entry in legend if int(entry["count"]) == winner_count) >= 2

    non_winner_legend_matches = sum(
        1
        for panel in panels
        if panel is not winner and any(entry["shape"] == panel["shape"] for entry in legend)
    )
    if len(legend) == 3:
        assert non_winner_legend_matches >= 1

    out = main(grid)
    r0, c0, r1, c1 = winner["bbox"]
    expected = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    expected[expected == GREY] = int(shape_matches[0]["key_color"])
    assert np.array_equal(out, expected)
    assert out.ndim == 2 and max(out.shape) <= 30
    assert grid.ndim == 2 and max(grid.shape) <= 30


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        try:
            grid = _build_case(rng)
            _validate_generated_grid(grid)
            return grid
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid input")
