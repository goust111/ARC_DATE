# Combo variant of dfadab01 (index 7)
# combo_spec: D7/0+D10/0
# Name: 随机散布+颜色混合重叠

from __future__ import annotations

import random

import numpy as np


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

MARKER_ORDER: tuple[int, ...] = (2, 3, 5, 8)
OVERLAP_OFFSETS_23: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
)

# Design analysis
# design:
#   a_concrete_behavior_and_trace:
#     original task: markers lived on a coarse row/column lattice, so glyph top-left positions were easy to infer
#       from repeated spacing and expanded glyphs never touched each other.
#     combo task: markers are scattered as individual points with no equal-step guarantee. Example:
#       red marker 2 at (2,2), green marker 3 at (3,4), grey marker 5 at (8,1).
#       original behavior on the first two markers was impossible because the lattice would keep them >=4 apart.
#       combo behavior expands all three glyphs at those exact top-left cells; the 2-glyph and 3-glyph overlap on
#       several pixels, and each shared cell keeps max(4,1)=4 rather than whichever glyph happened to be painted later.
#       The grey marker is isolated, so it shows the unchanged per-marker substitution rule beside the new collision rule.
#   b_complexity_calibration:
#     only two freedoms change together: placement becomes irregular, and overlap resolves by visible color mixing.
#     The glyph dictionary, anchor convention, clipping rule, and optional legend stay fixed, keeping the task legible.
#     Generator always includes marker colors 2 and 3 in at least one real overlap because that pair makes
#       max-color mixing visibly differ from the original overwrite-style implementation.
#   c_hidden_assumptions_in_original_main:
#     iterating markers directly into an output grid assumed each output pixel belonged to at most one glyph.
#       That is false now, so rendering must aggregate coverage and choose the maximum non-background glyph color.
#     the original solver also benefited from a regular marker lattice for human inference; solver code did not need it,
#       but the generator must now explicitly avoid falling back to lattice-like placements.
#     legend detection via a full 4x4 sample at (-4,-4) from the label marker still works, but only if the generator
#       reserves an empty 5x5 legend box so no real marker fakes a label context.
#   d_placement_legality:
#     input objects are still legal single pixels plus at most one legend entry. Real markers never share a cell.
#     overlap is allowed only in derived glyph footprints, not in the input markers themselves.
#     legend placement requires a clean 5x5 area, so legend pixels never erase markers and never create false labels.
#   e_visual_observability:
#     every accepted sample has overlap_pixels > 0, mixed_diff_pixels > 0 against the original overwrite rendering,
#       and also retains non-overlapped glyph area, so the new rule is visible without collapsing into a solid blob.
#     irregular scattering is supported by mostly unique rows and columns rather than a rectangular marker lattice.
#   f_visual_cleanliness:
#     generator centers one mandatory overlap cluster and keeps the remaining markers more dispersed.
#     Samples with excessive overlap density are rejected, preventing unreadable output carpets.
#   g_inference_of_color_change:
#     no hidden ordering or arithmetic on marker ids is used. The mixed color is simply the larger visible glyph color
#       among coincident non-black pixels, so repeated overlaps let a human infer the rule from structure alone.
#   solver_generator_contract:
#     main() assumes all real markers use colors 2/3/5/8 and any legend label exactly matches a 4x4 source glyph.
#     generate_input() guarantees at least one real marker, at least one 2-vs-3 overlap, and enough empty area for
#       the optional legend to remain unambiguous.


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                continue
            markers.append((marker_color, rr, cc))
    return markers


def _render_blended(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    out = np.zeros(shape, dtype=int)
    cover_count = np.zeros(shape, dtype=int)
    for marker_color, r0, c0 in markers:
        glyph_color, mask = GLYPHS[marker_color]
        for dr in range(4):
            rr = r0 + dr
            if rr < 0 or rr >= h:
                continue
            for dc in range(4):
                cc = c0 + dc
                if 0 <= cc < w and bool(mask[dr, dc]):
                    cover_count[rr, cc] += 1
                    if glyph_color > out[rr, cc]:
                        out[rr, cc] = glyph_color
    return out, cover_count


def _render_original_overwrite(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for marker_color in MARKER_ORDER:
        glyph_color, mask = GLYPHS[marker_color]
        for other_color, r0, c0 in markers:
            if other_color != marker_color:
                continue
            _paint_clipped(out, r0, c0, glyph_color, mask)
    return out


def _covered_cells(shape: tuple[int, int], marker_color: int, r0: int, c0: int) -> set[tuple[int, int]]:
    h, w = shape
    _, mask = GLYPHS[marker_color]
    cells: set[tuple[int, int]] = set()
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                cells.add((rr, cc))
    return cells


def _axis_gap_variety(values: list[int]) -> int:
    if len(values) < 3:
        return 0
    gaps = [b - a for a, b in zip(values, values[1:])]
    return len(set(gaps))


def _variant_metrics(g: np.ndarray) -> dict[str, float]:
    markers = _real_markers(g)
    blended, cover_count = _render_blended(g.shape, markers)
    overwrite = _render_original_overwrite(g.shape, markers)
    covered = [_covered_cells(g.shape, marker_color, r0, c0) for marker_color, r0, c0 in markers]
    overlapped_pairs = 0
    for i in range(len(covered)):
        for j in range(i + 1, len(covered)):
            if covered[i] & covered[j]:
                overlapped_pairs += 1

    rows = sorted({r for _, r, _ in markers})
    cols = sorted({c for _, _, c in markers})
    marker_count = len(markers)
    return {
        "real_markers": float(marker_count),
        "painted_pixels": float(np.count_nonzero(blended)),
        "overlap_pixels": float(np.count_nonzero(cover_count >= 2)),
        "overlapped_pairs": float(overlapped_pairs),
        "mixed_diff_pixels": float(np.count_nonzero(blended != overwrite)),
        "unique_row_ratio": float(len(rows) / marker_count) if marker_count else 0.0,
        "unique_col_ratio": float(len(cols) / marker_count) if marker_count else 0.0,
        "row_gap_variety": float(_axis_gap_variety(rows)),
        "col_gap_variety": float(_axis_gap_variety(cols)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each real marker into its fixed 4x4 glyph, then resolve collisions by visible color mixing.

    Input structure:
    - Black background with single-cell markers in colors 2, 3, 5, and/or 8, scattered irregularly rather than on a regular lattice.
    - Some inputs also contain one legend entry: a 4x4 sample glyph drawn in its output color, plus a matching marker one cell below-right of that sample.
    - Real markers may be close enough that their future 4x4 glyph footprints overlap after expansion.

    Rule:
    - Ignore any marker whose matching sample glyph sits exactly at (-4, -4); that marker is a legend label, not a scene marker.
    - Replace each remaining marker by the same fixed glyph dictionary as the original task, anchored at the marker cell and clipped by the grid boundary.
    - When several expanded glyphs cover the same output cell, keep the non-black glyph color with the largest value; otherwise keep the lone glyph color.

    Key insight:
    - The marker-to-glyph alphabet is unchanged. The new work is to locate scattered real markers one by one and render all overlapping glyph contributions before deciding each output pixel by max-color mixing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    markers = _real_markers(g)
    assert markers
    out, _ = _render_blended(g.shape, markers)
    return out


def _mandatory_overlap_pair(rng: random.Random, size: int) -> list[tuple[int, int, int]]:
    dr, dc = rng.choice(OVERLAP_OFFSETS_23)
    r0 = rng.randint(0, size - 1 - dr)
    c0 = rng.randint(0, size - 1 - dc)
    return [(2, r0, c0), (3, r0 + dr, c0 + dc)]


def _choose_marker_count(rng: random.Random, size: int) -> int:
    if size == 10:
        return rng.randint(4, 6)
    return rng.randint(7, 10)


def _sample_extra_positions(
    rng: random.Random,
    size: int,
    markers: list[tuple[int, int, int]],
    target_count: int,
) -> list[tuple[int, int]]:
    positions = {(r, c) for _, r, c in markers}
    used_rows = {r for _, r, _ in markers}
    used_cols = {c for _, _, c in markers}
    extras: list[tuple[int, int]] = []

    for _ in range(800):
        if len(positions) >= target_count:
            break

        if rng.random() < 0.35:
            _, base_r, base_c = rng.choice(markers)
            r = base_r + rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            c = base_c + rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        else:
            r = rng.randrange(size)
            c = rng.randrange(size)

        if not (0 <= r < size and 0 <= c < size):
            continue
        if (r, c) in positions:
            continue
        if r in used_rows and c in used_cols:
            continue

        close_neighbors = 0
        for rr, cc in positions:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                close_neighbors += 1
        if close_neighbors >= 2:
            continue

        positions.add((r, c))
        used_rows.add(r)
        used_cols.add(c)
        extras.append((r, c))

    if len(positions) < target_count:
        raise RuntimeError("could not sample enough irregular marker positions")
    return extras


def _assign_extra_colors(
    rng: random.Random,
    extra_count: int,
) -> list[int]:
    used_colors = [2, 3]
    optional = rng.sample([5, 8], rng.randint(0, 2))
    used_colors.extend(optional)

    colors = used_colors[:]
    while len(colors) < extra_count:
        colors.append(rng.choice(used_colors))
    rng.shuffle(colors)
    return colors[:extra_count]


def _build_markers(rng: random.Random, size: int) -> list[tuple[int, int, int]]:
    markers = _mandatory_overlap_pair(rng, size)
    target_count = _choose_marker_count(rng, size)
    extra_positions = _sample_extra_positions(rng, size, markers, target_count)
    extra_colors = _assign_extra_colors(rng, len(extra_positions))
    for (r, c), color in zip(extra_positions, extra_colors):
        markers.append((color, r, c))
    return markers


def _find_legend_spots(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    occupied = {(r, c) for _, r, c in markers}
    spots: list[tuple[int, int]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            blocked = False
            for rr in range(r0, r0 + 5):
                for cc in range(c0, c0 + 5):
                    if (rr, cc) in occupied:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                spots.append((r0, c0))
    return spots


def _add_legend(rng: random.Random, grid: np.ndarray, markers: list[tuple[int, int, int]]) -> None:
    if rng.random() >= 0.45:
        return
    spots = _find_legend_spots(grid.shape, markers)
    if not spots:
        return

    marker_color = rng.choice([marker_color for marker_color, _, _ in markers])
    glyph_color, mask = GLYPHS[marker_color]
    r0, c0 = rng.choice(spots)
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                grid[r0 + dr, c0 + dc] = glyph_color
    grid[r0 + 4, c0 + 4] = marker_color


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        size = rng.choice([10, 20])
        markers = _build_markers(rng, size)
        expected, _ = _render_blended((size, size), markers)
        blank = np.zeros((size, size), dtype=int)
        for marker_color, r, c in markers:
            blank[r, c] = marker_color
        cover_metrics = _variant_metrics(blank)

        if cover_metrics["overlap_pixels"] < 2:
            continue
        if cover_metrics["mixed_diff_pixels"] < 1:
            continue
        if cover_metrics["painted_pixels"] <= 0:
            continue
        if cover_metrics["overlap_pixels"] * 2 > cover_metrics["painted_pixels"]:
            continue
        if cover_metrics["unique_row_ratio"] < 0.6:
            continue
        if cover_metrics["unique_col_ratio"] < 0.6:
            continue
        if cover_metrics["row_gap_variety"] == 0 and cover_metrics["col_gap_variety"] == 0:
            continue

        grid = blank.copy()
        _add_legend(rng, grid, markers)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a non-degenerate scattered-overlap sample")
