# Combo variant of dfadab01 (index 22)
# combo_spec: D2/1+D5/0+D4/1
# Name: 动态映射+旋转变体+完整图例

from __future__ import annotations

import random

import numpy as np


MARKER_COLORS: tuple[int, ...] = (2, 3, 5, 8)
GLYPH_COLORS: tuple[int, ...] = (1, 4, 6, 7)
GRID_SIZE = 20
GLYPH_SIZE = 4
GLYPH_AREA = 8
LEGEND_COL = 0
LEGEND_LABEL_COL = 4
LEGEND_ROW_STARTS: tuple[int, ...] = (0, 5, 10, 15)
BODY_ROW_LAYOUTS: tuple[tuple[int, ...], ...] = (
    (0, 4, 8, 12, 16),
    (1, 5, 9, 13, 17),
    (2, 6, 10, 14, 18),
    (0, 5, 10, 15),
    (2, 7, 12, 17),
)
BODY_COL_LAYOUTS: tuple[tuple[int, ...], ...] = (
    (8, 12, 16),
    (9, 13, 17),
    (10, 14, 18),
    (8, 13, 18),
)
CANONICAL_COLOR_BY_MARKER: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
CANONICAL_ROTATION_BY_MARKER: dict[int, int] = {2: 0, 3: 1, 5: 2, 8: 3}
MAX_GEN_ATTEMPTS = 400


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    The fused task keeps the original "single-pixel marker expands into a 4x4
#    glyph" scaffold, but every lookup becomes local to the visible legend.
#    - Left legend band: exactly four entries are shown, one per marker color.
#      Each entry is a 4x4 sample glyph plus a marker-color label at (+4, +4).
#    - The four sample glyphs are not unrelated templates. They are the 0/90/180/
#      270-degree rotations of one random asymmetric 4x4 base mask.
#    - The sample color is also puzzle-local. Example: if the legend shows an
#      orange sample above-left of label `2`, and that orange sample is the 180°
#      rotation of the common base, then every real marker `2` in the body must
#      expand into that orange 180° variant.
#    In the original task, color 2 always meant "yellow frame". Here, color 2 is
#    meaningless until the legend is decoded, and the solver must also notice
#    that the four legend masks come from the same rotated prototype.
#
# b) Complexity calibration
#    The design exposes all three changes without becoming ambiguous:
#    - exactly four marker colors and four glyph colors are reused, not expanded;
#    - the body always contains all four marker colors, each at least twice, so
#      the decoded mapping is visibly reused instead of appearing only once;
#    - the base mask is constrained to be connected, non-symmetric under quarter
#      turns, and to have all four rotations visually distinct.
#    This keeps the hypothesis space small enough for ARC while making the
#    rotation relation and local color mapping both visible.
#
# c) Implicit assumption audit
#    - Original `GLYPHS[marker_color]` hard-codes both shape and color. That is
#      invalid now; `main()` must first parse the legend.
#    - Original legend skipping used a known `(glyph_color, mask)` pair. That is
#      also invalid now; `main()` must identify legend labels by reading the
#      4x4 patch immediately above-left of each marker candidate.
#    - Reading four legend samples independently is not sufficient for the fused
#      rule: the solver additionally assumes those four masks form one rotation
#      family. `main()` validates this explicitly.
#    - Because the legend is complete, `main()` assumes each marker color has
#      exactly one legal label/sample pair and that the legend samples use four
#      distinct glyph colors. `generate_input()` guarantees both.
#
# d) Placement legality
#    The generator fixes the legend in the left 5 columns at row starts
#    0/5/10/15, so each 4x4 sample and its label fit exactly inside the 20x20
#    canvas with no overlap. Real markers are sampled only from columns >= 8 and
#    from row/column spacings >= 4, so expanded 4x4 glyphs never overlap the
#    legend or each other. Rows/columns reaching 17 or 18 preserve boundary
#    clipping near the bottom/right edge.
#
# e) Visual observability
#    A human sees a clean decomposition: the left column is the legend, the right
#    side is the marker field. Because all four legend samples are visible at
#    once, the shared-rotation relation can be spotted directly, and because the
#    label colors are shuffled against both rotation and output color, the viewer
#    must actually decode the legend instead of recalling a fixed dictionary.
#
# f) Visual cleanliness
#    No object layers overlap. Legend samples never mix with body markers, and
#    body glyphs lie on a regular sparse lattice. The only non-black colors are
#    the four marker labels and the four sample/output colors, so the puzzle
#    remains readable despite the added freedom.
#
# g) Change inferability and anti-degeneracy
#    To avoid collapsing back to the original task, the generator forces:
#    - a full four-entry legend on every sample;
#    - a color permutation with at least two moved marker->glyph-color links;
#    - a rotation permutation with at least two moved marker->rotation links;
#    - a random base mask whose quarter turns are all different.
#    The validation step later measures these moved-counts directly so we can
#    confirm the generated distribution really exercises the fused behavior.
#
# h) New solver assumptions mirrored by generation
#    `main()` assumes every real marker color has exactly one legend label, the
#    patch above-left of that label is monochrome and non-empty, the four legend
#    masks share one rotation family, and real markers never sit in the legend
#    band. `_build_case()` constructs inputs to satisfy those assumptions and
#    rejects any sample that fails a full `main(input) == output` round-trip.


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(GLYPH_SIZE):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(GLYPH_SIZE):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _mask_key(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).tobytes()


def _rotation_index(base_mask: np.ndarray, mask: np.ndarray) -> int | None:
    for k in range(4):
        if np.array_equal(np.rot90(base_mask, k), mask):
            return k
    return None


def _infer_rotation_base(masks: list[np.ndarray]) -> np.ndarray:
    target_keys = {_mask_key(mask) for mask in masks}
    assert len(target_keys) == 4, "legend masks must all be distinct"
    for candidate in masks:
        rotated = [np.rot90(candidate, k).copy() for k in range(4)]
        if {_mask_key(mask) for mask in rotated} == target_keys:
            return candidate.copy()
    raise AssertionError("legend masks are not the four quarter-turns of one base glyph")


def _read_legend_entry(g: np.ndarray, r: int, c: int) -> tuple[int, np.ndarray] | None:
    if r < GLYPH_SIZE or c < GLYPH_SIZE:
        return None
    patch = g[r - GLYPH_SIZE : r, c - GLYPH_SIZE : c]
    nonzero = patch[patch != 0]
    if nonzero.size == 0:
        return None
    glyph_color = int(nonzero[0])
    if not np.all(nonzero == glyph_color):
        return None
    mask = patch == glyph_color
    if int(mask.sum()) != GLYPH_AREA:
        return None
    return glyph_color, mask


def _decode_legend_mapping(
    g: np.ndarray,
) -> tuple[dict[int, tuple[int, np.ndarray]], set[tuple[int, int]]]:
    mapping: dict[int, tuple[int, np.ndarray]] = {}
    legend_labels: set[tuple[int, int]] = set()

    for marker_color in MARKER_COLORS:
        matches: list[tuple[tuple[int, int], tuple[int, np.ndarray]]] = []
        for r, c in np.argwhere(g == marker_color):
            entry = _read_legend_entry(g, int(r), int(c))
            if entry is not None:
                matches.append(((int(r), int(c)), entry))
        assert len(matches) == 1, f"marker color {marker_color} must have exactly one legend label"
        (r, c), entry = matches[0]
        mapping[marker_color] = entry
        legend_labels.add((r, c))

    glyph_colors = {mapping[marker_color][0] for marker_color in MARKER_COLORS}
    assert len(glyph_colors) == 4, "legend glyph colors must be distinct"

    masks = [mapping[marker_color][1] for marker_color in MARKER_COLORS]
    base_mask = _infer_rotation_base(masks)
    used_rotations: set[int] = set()
    for marker_color in MARKER_COLORS:
        rotation = _rotation_index(base_mask, mapping[marker_color][1])
        assert rotation is not None, f"marker {marker_color} does not match the inferred base glyph"
        assert rotation not in used_rotations, f"rotation {rotation} appears more than once in the legend"
        used_rotations.add(rotation)

    return mapping, legend_labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a full local legend, verify it is one rotated glyph family, then expand the real markers.

    Input structure:
    - The grid is 20x20 with black background.
    - The left legend band contains four complete 4x4 sample glyphs, each with a marker-color label
      one cell below-right of the sample's bounding box.
    - The body contains single-pixel markers in colors 2, 3, 5, and 8; the same marker colors also
      appear as the four legend labels.

    Rule:
    - For each marker color, read the 4x4 monochrome sample immediately above-left of its legend label.
    - Treat that sample's color as the output color and its 4x4 footprint as the glyph mask for that
      marker color; also confirm that the four learned masks are exactly the four quarter-turn rotations
      of one shared base glyph.
    - Replace every non-legend marker by its learned colored glyph, anchored with the marker as the
      glyph's top-left corner and clipped at the grid boundary.
    - Remove the entire legend region and the label markers themselves from the output.

    Key insight:
    - Nothing is globally fixed anymore: the legend simultaneously defines which output color belongs to
      each marker color and which rotation of the puzzle-specific base glyph that marker should use.
      The shared rotation family is the structural clue tying the four legend entries together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape == (GRID_SIZE, GRID_SIZE)
    assert set(map(int, np.unique(g))).issubset({0, *MARKER_COLORS, *GLYPH_COLORS})

    mapping, legend_labels = _decode_legend_mapping(g)

    out = np.zeros_like(g)
    seen_real_marker = False
    for marker_color in MARKER_COLORS:
        glyph_color, mask = mapping[marker_color]
        for r, c in np.argwhere(g == marker_color):
            rc = (int(r), int(c))
            if rc in legend_labels:
                continue
            seen_real_marker = True
            _paint_clipped(out, rc[0], rc[1], glyph_color, mask)

    assert seen_real_marker, "expected at least one real body marker"
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    stack = [cells[0]]
    seen = {cells[0]}
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < GLYPH_SIZE and 0 <= cc < GLYPH_SIZE and bool(mask[rr, cc]) and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    stack.append((rr, cc))
    return len(seen) == len(cells)


def _random_base_mask() -> np.ndarray:
    while True:
        picks = random.sample(range(GLYPH_SIZE * GLYPH_SIZE), GLYPH_AREA)
        mask = np.zeros((GLYPH_SIZE, GLYPH_SIZE), dtype=bool)
        for idx in picks:
            mask[idx // GLYPH_SIZE, idx % GLYPH_SIZE] = True
        if not _mask_connected(mask):
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue

        rotations = [np.rot90(mask, k).copy() for k in range(4)]
        if len({_mask_key(rot) for rot in rotations}) != 4:
            continue
        min_distance = min(
            int(np.sum(rotations[i] != rotations[j]))
            for i in range(4)
            for j in range(i + 1, 4)
        )
        if min_distance < 6:
            continue
        return mask


def _sample_rotation_map() -> dict[int, int]:
    while True:
        rotations = random.sample(range(4), k=4)
        mapping = {marker_color: rotations[i] for i, marker_color in enumerate(MARKER_COLORS)}
        moved = sum(
            int(mapping[marker_color] != CANONICAL_ROTATION_BY_MARKER[marker_color])
            for marker_color in MARKER_COLORS
        )
        if moved >= 2:
            return mapping


def _sample_color_map() -> dict[int, int]:
    while True:
        colors = random.sample(list(GLYPH_COLORS), k=4)
        mapping = {marker_color: colors[i] for i, marker_color in enumerate(MARKER_COLORS)}
        moved = sum(
            int(mapping[marker_color] != CANONICAL_COLOR_BY_MARKER[marker_color])
            for marker_color in MARKER_COLORS
        )
        if moved >= 2:
            return mapping


def _body_positions() -> list[tuple[int, int]]:
    rows = list(random.choice(BODY_ROW_LAYOUTS))
    cols = list(random.choice(BODY_COL_LAYOUTS))
    positions = [(int(r), int(c)) for r in rows for c in cols]
    random.shuffle(positions)
    keep = random.randint(8, len(positions))
    return positions[:keep]


def _sample_real_markers() -> list[tuple[int, int, int]]:
    positions = _body_positions()
    marker_colors = [color for color in MARKER_COLORS for _ in range(2)]
    marker_colors.extend(random.choices(list(MARKER_COLORS), k=len(positions) - len(marker_colors)))
    random.shuffle(marker_colors)
    return [(r, c, marker_colors[i]) for i, (r, c) in enumerate(positions)]


def _build_output(
    base_mask: np.ndarray,
    rotation_map: dict[int, int],
    color_map: dict[int, int],
    real_markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c, marker_color in real_markers:
        mask = np.rot90(base_mask, rotation_map[marker_color]).copy()
        _paint_clipped(out, r, c, color_map[marker_color], mask)
    return out


def _build_input(
    base_mask: np.ndarray,
    rotation_map: dict[int, int],
    color_map: dict[int, int],
    real_markers: list[tuple[int, int, int]],
) -> tuple[np.ndarray, tuple[int, ...]]:
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legend_order = list(MARKER_COLORS)
    random.shuffle(legend_order)
    for r0, marker_color in zip(LEGEND_ROW_STARTS, legend_order):
        mask = np.rot90(base_mask, rotation_map[marker_color]).copy()
        _paint_clipped(g, r0, LEGEND_COL, color_map[marker_color], mask)
        g[r0 + GLYPH_SIZE, LEGEND_LABEL_COL] = marker_color

    for r, c, marker_color in real_markers:
        g[r, c] = marker_color
    return g, tuple(legend_order)


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    for _ in range(MAX_GEN_ATTEMPTS):
        base_mask = _random_base_mask()
        rotation_map = _sample_rotation_map()
        color_map = _sample_color_map()
        real_markers = _sample_real_markers()

        inp, legend_order = _build_input(base_mask, rotation_map, color_map, real_markers)
        out = _build_output(base_mask, rotation_map, color_map, real_markers)

        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue

        counts = {marker_color: 0 for marker_color in MARKER_COLORS}
        clipped_markers = 0
        rotations = [np.rot90(base_mask, k).copy() for k in range(4)]
        for r, c, marker_color in real_markers:
            counts[marker_color] += 1
            if r > GRID_SIZE - GLYPH_SIZE or c > GRID_SIZE - GLYPH_SIZE:
                clipped_markers += 1

        meta = {
            "rotation_map": dict(rotation_map),
            "color_map": dict(color_map),
            "rotation_moved_count": sum(
                int(rotation_map[marker_color] != CANONICAL_ROTATION_BY_MARKER[marker_color])
                for marker_color in MARKER_COLORS
            ),
            "color_moved_count": sum(
                int(color_map[marker_color] != CANONICAL_COLOR_BY_MARKER[marker_color])
                for marker_color in MARKER_COLORS
            ),
            "legend_order": legend_order,
            "real_marker_counts": counts,
            "real_marker_total": len(real_markers),
            "clipped_markers": clipped_markers,
            "rotation_min_distance": min(
                int(np.sum(rotations[i] != rotations[j]))
                for i in range(4)
                for j in range(i + 1, 4)
            ),
        }
        return inp, out, meta

    raise RuntimeError(f"failed to generate a valid combo instance after {MAX_GEN_ATTEMPTS} attempts")


def generate_input() -> np.ndarray:
    inp, _, _ = _build_case()
    return inp


def generate_pair() -> tuple[np.ndarray, np.ndarray]:
    inp, out, _ = _build_case()
    return inp, out
