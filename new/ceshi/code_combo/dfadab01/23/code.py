# Combo variant of dfadab01 (index 23)
# combo_spec: D1/4+D7/0+D10/0
# Name: 叠加合成+随机散布+颜色混合

from __future__ import annotations

import random

import numpy as np


GLYPH_SIZE = 4

GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}

MARKER_ORDER = {marker_color: idx for idx, marker_color in enumerate(GLYPHS)}
ALLOWED_COLORS = {0, *GLYPHS.keys(), *(glyph_color for glyph_color, _ in GLYPHS.values())}
GEN_RETRIES = 400


# Design analysis for combo D1/4 + D7/0 + D10/0
# design:
#   a_concrete_behavior_and_trace:
#     original_task:
#       - Real content in the input is a set of single-pixel markers on a loose grid.
#       - Output replaces each marker by one fixed 4x4 glyph in a fixed output color.
#       - The old spacing guarantee meant glyphs never met, so simple painting order was harmless.
#     fused_variant:
#       - Markers are no longer arranged on a regular lattice; they are scattered at irregular coordinates.
#       - Every real marker still expands to the same 4x4 glyph dictionary, but all markers are interpreted
#         simultaneously and overlapping output cells use max(nonzero colors), which is equivalent here to
#         taking the larger glyph color because background is 0.
#       - The legend mechanism is kept: a fully drawn 4x4 sample glyph may appear in the input, with its label
#         marker at (+4,+4); that label marker must still be skipped.
#     concrete_trace:
#       - Suppose a red marker 2 is at (4,4) and a green marker 3 is at (5,5). The red frame paints yellow 4s
#         on the border of rows 4..7 / cols 4..7; the green ring paints blue 1s on rows 5..8 / cols 5..8.
#       - In the old solver, the 3-marker would be processed after the 2-marker and overwrite shared cells with 1.
#       - In this combo, shared cells stay 4 because 4 > 1. That visible disagreement is the point of D1/4+D10/0.
#   b_complexity_calibration:
#     - Keep the original four glyph families and color mapping so the only new reasoning burden is spatial:
#       find scattered markers, expand them, and handle overlaps by one consistent composition rule.
#     - Generator always forces at least one overlapping 2/3 pair because that pair makes max-composition visibly
#       different from naive overwrite. Extra markers add repetition without introducing new hidden conventions.
#     - Overlap density is bounded: enough to be common and obvious, but rejected if it dominates most painted cells.
#   c_hidden_assumption_review_of_main:
#     - Original _paint_clipped assumed a unique owner for each output cell; unconditional assignment no longer
#       matches the task because overlaps are intentional.
#     - Original regular-grid intuition is gone; main must treat every surviving marker independently instead of
#       relying on predictable spacing.
#     - Legend detection via a complete 4x4 sample at (-4,-4) from the label still survives unchanged because
#       legend colors {1,4,6,7} remain disjoint from marker colors {2,3,5,8}.
#     - Original seen_marker could be satisfied by legend labels alone. The new main instead requires at least one
#       non-legend marker after filtering, and generate_input guarantees that.
#   d_placement_legality:
#     - Input pixels never overlap: marker positions are unique, and legend cells are only placed in empty space.
#     - Output overlap is deliberate and legal. The generator controls it through marker proximity rather than by
#       drawing illegal multi-color input cells.
#     - Core overlap markers are kept fully in-bounds so the mixed-color region is clear; other markers may touch
#       edges so the original clipping behavior remains present.
#   e_visual_observability:
#     - The scattered marker locations are directly visible in the input; the fused effect is directly visible in
#       the output where some overlap bands keep the larger color instead of the later-processed one.
#     - Because the generator also leaves many non-overlapped glyph pixels, the solver can compare "plain glyph"
#       regions with "merged glyph" regions and infer one uniform overlap rule.
#   f_visual_tidiness:
#     - The generator limits local crowding so a cell is rarely influenced by more than two nearby glyphs.
#     - One forced 2/3 overlap cluster plus a few additional scattered markers keeps the board readable; fully
#       saturated tangles are rejected.
#   g_change_inferability:
#     - No invisible ordering tricks are used in the task rule itself. The decisive signal is a visible comparison
#       between overlapping colored glyph pixels, and the output color is determined by the larger visible color.
#       Humans do not need to know numeric ids in advance because repeated examples show the same "larger-color wins"
#       behavior on the same glyph families.
#   main_generator_contract:
#     - main assumes each legend label, if present, has a complete matching 4x4 sample glyph at (-4,-4); the
#       generator places legends exactly that way.
#     - main assumes at least one real marker remains after legend filtering; the generator always places several.
#     - main assumes all nonzero input colors belong to the original marker/sample palette; the generator enforces it.


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + GLYPH_SIZE > h or c0 + GLYPH_SIZE > w:
        return False
    patch = g[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped_overwrite(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(GLYPH_SIZE):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(GLYPH_SIZE):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _accumulate_clipped_max(
    out: np.ndarray,
    cover_count: np.ndarray,
    r0: int,
    c0: int,
    color: int,
    mask: np.ndarray,
) -> None:
    h, w = out.shape
    for dr in range(GLYPH_SIZE):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(GLYPH_SIZE):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                cover_count[rr, cc] += 1
                if color > out[rr, cc]:
                    out[rr, cc] = color


def _render_composite(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(shape, dtype=int)
    cover_count = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        glyph_color, mask = GLYPHS[marker_color]
        _accumulate_clipped_max(out, cover_count, r, c, glyph_color, mask)
    return out, cover_count


def _render_sequential(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(markers, key=lambda item: (MARKER_ORDER[item[2]], item[0], item[1]))
    for r, c, marker_color in ordered:
        glyph_color, mask = GLYPHS[marker_color]
        _paint_clipped_overwrite(out, r, c, glyph_color, mask)
    return out


def _mask_overlap_count(mask_a: np.ndarray, mask_b: np.ndarray, dr: int, dc: int) -> int:
    count = 0
    for r in range(GLYPH_SIZE):
        for c in range(GLYPH_SIZE):
            if not bool(mask_a[r, c]):
                continue
            rr = r - dr
            cc = c - dc
            if 0 <= rr < GLYPH_SIZE and 0 <= cc < GLYPH_SIZE and bool(mask_b[rr, cc]):
                count += 1
    return count


CORE_PAIR_DELTAS: list[tuple[int, int, int]] = []
for _dr in range(-(GLYPH_SIZE - 1), GLYPH_SIZE):
    for _dc in range(-(GLYPH_SIZE - 1), GLYPH_SIZE):
        if _dr == 0 and _dc == 0:
            continue
        overlap = _mask_overlap_count(GLYPHS[2][1], GLYPHS[3][1], _dr, _dc)
        if 2 <= overlap <= 6:
            CORE_PAIR_DELTAS.append((_dr, _dc, overlap))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand every real marker into its fixed 4x4 glyph and merge overlaps by color maximum.

    Input structure:
    - Black background with single-cell markers in colors 2, 3, 5, and/or 8 at irregular scattered positions.
    - Some inputs may also contain one legend entry: a full 4x4 glyph already drawn in its output color plus a
      matching label marker located one cell below-right of the legend's bounding box.

    Rule:
    - Ignore any marker that is only the label of such a legend sample.
    - Replace each remaining marker by its fixed 4x4 glyph anchored at the marker position, clipping at the border.
    - Treat all glyph placements as simultaneous: if multiple glyph pixels land on the same output cell, keep the
      non-background color with the largest value, which is equivalent here to taking the maximum painted color.

    Key insight:
    - The marker-to-glyph dictionary is unchanged; the real change is that spacing no longer prevents collisions, so
      the solver must detect all real markers first and then composite their glyphs with a collision rule instead of
      trusting paint order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_COLORS)

    markers: list[tuple[int, int, int]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - GLYPH_SIZE, cc - GLYPH_SIZE, glyph_color, mask):
                continue
            markers.append((rr, cc, marker_color))

    assert markers
    out, _ = _render_composite(g.shape, markers)
    return out


def _choose_core_pair(rng: random.Random, size: int) -> list[tuple[int, int, int]]:
    deltas = CORE_PAIR_DELTAS[:]
    rng.shuffle(deltas)
    for dr, dc, _overlap in deltas:
        r_min = max(0, -dr)
        r_max = min(size - GLYPH_SIZE, size - GLYPH_SIZE - dr)
        c_min = max(0, -dc)
        c_max = min(size - GLYPH_SIZE, size - GLYPH_SIZE - dc)
        if r_min > r_max or c_min > c_max:
            continue
        r0 = rng.randint(r_min, r_max)
        c0 = rng.randint(c_min, c_max)
        return [(r0, c0, 2), (r0 + dr, c0 + dc, 3)]
    raise RuntimeError("failed to place the forced 2/3 overlap pair")


def _sample_extra_position(
    rng: random.Random,
    size: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    positions = list(occupied)
    for _ in range(150):
        if positions and rng.random() < 0.7:
            br, bc = rng.choice(positions)
            r = br + rng.randint(-3, 3)
            c = bc + rng.randint(-3, 3)
        else:
            r = rng.randint(0, size - 1)
            c = rng.randint(0, size - 1)

        if not (0 <= r < size and 0 <= c < size):
            continue
        if (r, c) in occupied:
            continue

        close_neighbors = sum(abs(r - rr) < GLYPH_SIZE and abs(c - cc) < GLYPH_SIZE for rr, cc in occupied)
        if close_neighbors > 2:
            continue
        return r, c
    return None


def _choose_extra_colors(rng: random.Random, extra_count: int) -> list[int]:
    if extra_count <= 0:
        return []
    optional = [5, 8]
    used = [2, 3]
    used.extend(rng.sample(optional, rng.randint(0, min(len(optional), extra_count))))

    palette: list[int] = []
    if extra_count >= 1:
        palette.append(rng.choice([2, 3]))
    if extra_count >= 2 and optional and rng.random() < 0.7:
        palette.append(rng.choice(optional))
    while len(palette) < extra_count:
        palette.append(rng.choice(used))
    rng.shuffle(palette)
    return palette


def _choose_markers(rng: random.Random, size: int) -> list[tuple[int, int, int]]:
    markers = _choose_core_pair(rng, size)
    occupied = {(r, c) for r, c, _ in markers}
    target_count = rng.randint(6, 9)
    while len(markers) < target_count:
        pos = _sample_extra_position(rng, size, occupied)
        if pos is None:
            break
        occupied.add(pos)
        markers.append((pos[0], pos[1], 0))

    extra_colors = _choose_extra_colors(rng, len(markers) - 2)
    for idx, color in enumerate(extra_colors, start=2):
        r, c, _ = markers[idx]
        markers[idx] = (r, c, color)
    return markers


def _looks_like_regular_grid(markers: list[tuple[int, int, int]]) -> bool:
    rows = sorted({r for r, _, _ in markers})
    cols = sorted({c for _, c, _ in markers})
    if len(rows) < 3 or len(cols) < 3:
        return False
    row_steps = {rows[i + 1] - rows[i] for i in range(len(rows) - 1)}
    col_steps = {cols[i + 1] - cols[i] for i in range(len(cols) - 1)}
    return len(row_steps) == 1 and len(col_steps) == 1


def _find_legend_spots(shape: tuple[int, int], occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    spots: list[tuple[int, int]] = []
    for r0 in range(h - GLYPH_SIZE):
        for c0 in range(w - GLYPH_SIZE):
            patch_cells = {
                (r, c)
                for r in range(r0, r0 + GLYPH_SIZE)
                for c in range(c0, c0 + GLYPH_SIZE)
            }
            label = (r0 + GLYPH_SIZE, c0 + GLYPH_SIZE)
            if patch_cells & occupied:
                continue
            if label in occupied:
                continue
            spots.append((r0, c0))
    return spots


def _maybe_add_legend(
    rng: random.Random,
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
) -> None:
    if rng.random() >= 0.7:
        return
    occupied = {(r, c) for r, c, _ in markers}
    spots = _find_legend_spots(grid.shape, occupied)
    if not spots:
        return

    marker_color = rng.choice([marker_color for _, _, marker_color in markers])
    glyph_color, mask = GLYPHS[marker_color]
    r0, c0 = rng.choice(spots)
    for dr in range(GLYPH_SIZE):
        for dc in range(GLYPH_SIZE):
            if bool(mask[dr, dc]):
                grid[r0 + dr, c0 + dc] = glyph_color
    grid[r0 + GLYPH_SIZE, c0 + GLYPH_SIZE] = marker_color


def _score_behavior(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    composite, cover_count = _render_composite(shape, markers)
    sequential = _render_sequential(shape, markers)
    overlap_pixels = int(np.count_nonzero(cover_count >= 2))
    changed_pixels = int(np.count_nonzero(composite != sequential))
    painted_pixels = int(np.count_nonzero(composite))
    return overlap_pixels, changed_pixels, painted_pixels


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        size = rng.choice([12, 14, 16, 18, 20])
        markers = _choose_markers(rng, size)

        if len(markers) < 6:
            continue
        if _looks_like_regular_grid(markers):
            continue

        overlap_pixels, changed_pixels, painted_pixels = _score_behavior((size, size), markers)
        if overlap_pixels < 3:
            continue
        if changed_pixels < 2:
            continue
        if overlap_pixels * 2 > painted_pixels:
            continue

        grid = np.zeros((size, size), dtype=int)
        for r, c, marker_color in markers:
            grid[r, c] = marker_color
        _maybe_add_legend(rng, grid, markers)

        expected, _ = _render_composite(grid.shape, markers)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid scattered-overlap dfadab01 combo sample")
