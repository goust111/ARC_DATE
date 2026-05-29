# Combo variant of e8686506 (index 31)
# combo_spec: D9/2+D16/2
# Name: 按行排列+碎片按列分组

# Design analysis
# variant_analysis:
#   concrete_behavior:
#     output:
#       - Keep the original "assemble fragments into a dense LR-symmetric rectangle" theme, but freeze the target
#         to a 5x5 three-layer square so both row and layer signals stay visible.
#       - The 5x5 output is a concentric layout:
#         row 0 = one outer-layer bar of width 5
#         row 1 = two outer singletons + one inner bar of width 3
#         row 2 = two outer singletons + two inner singletons + one center singleton
#         row 3 = same pattern as row 1
#         row 4 = one outer-layer bar of width 5
#       - Every connected piece is a single horizontal fragment of height 1, so each piece belongs to exactly one
#         output row and exactly one output layer.
#     input:
#       - Fragments are no longer freely scattered. They are placed on five horizontal rows and three separated
#         column bands.
#       - Input row order encodes output row order from top to bottom.
#       - Input column band encodes layer: left band -> outer layer, middle band -> inner layer, right band -> center.
#     trace_example:
#       - Original logic: a 1x1 fragment could be placed by symmetry plus shape compatibility alone.
#       - Fused logic: a 1x1 fragment seen on the third input row and middle band is immediately known to belong to
#         output row 2, layer 1; it can only fill one of that row's two inner symmetric singleton slots.
#       - Difference: input position now supplies both coordinates of the abstract placement system:
#         y-position gives row, x-band gives depth/layer.
#   complexity_calibration:
#     - There are exactly 5 row groups and 3 layer bands in every sample, so the new encoding is obvious in 2-3 examples.
#     - Repeated widths are deliberate:
#         outer width-1 pieces repeat across three rows, so row information matters;
#         row 2 contains five width-1 pieces across three layers, so layer information matters.
#     - The remaining ambiguity is only the intended left/right mirror pairing inside a row-layer slot.
#   hidden_assumption_audit:
#     - Original frame locking assumed 1-2 8-connected outline components preserve output-relative positions in input.
#       This is invalid after row/band rearrangement, so the fused solver removes frame detection entirely.
#     - Original optional anchor heuristics (corners, horizontal bars, paired vertical lines) assumed input geometry
#       reflected output edges. In the fused task, those roles are replaced by explicit row/layer coordinates.
#     - Original grouping by (color, shape) was enough because input position had no semantic meaning. Here the solver
#       must first classify each piece by input row and input band before reconstructing the output.
#     - Original propagation searched over all rectangle placements. The fused variant instead uses a fixed slot catalog:
#       each input piece must be a 1-row horizontal fragment and generator guarantees the canonical 5x5 layout.
#   placement_legality:
#     - Generator uses a 15x15 canvas with row anchors spaced by blank rows and three separated column bands
#       (0-4, 6-10, 12-14), so pieces never overlap and pieces from different bands never become adjacent.
#     - Within a row-band cell, slot coordinates are fixed and fit the fragment widths exactly.
#   visual_observability:
#     - Humans can visibly read a 5x3 arrangement on the input side and map it to a 5x5 layered square on the output side.
#     - The rule does not depend on color ids. Colors are permuted each sample; only the row and band structure is stable.
#   cleanliness:
#     - Blank spacer rows and blank spacer columns keep the input uncluttered.
#     - The output is a tidy concentric square with no overdraw or mixed encodings.
#   color_inference:
#     - Only five non-background colors are used, and repeated colors are tied to repeated structural roles
#       (top/bottom outer bars, side-pair rows, inner bars, inner row-2 pair, center), making the palette easy to track.
#   solver_generator_contract:
#     - Solver assumes:
#         1. exactly five occupied input row groups,
#         2. exactly three occupied input column bands split by cuts at 5 and 11,
#         3. every piece is a filled horizontal bar of height 1,
#         4. multi-piece symmetric row-layer groups share one color.
#       Generator enforces all four conditions and validates them through main(input) == output.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


_INPUT_BAND_CUTS = (5, 11)
_CANVAS_ROWS = 15
_CANVAS_COLS = 15
_GEN_MAX_RETRIES = 300

_ROW_ANCHOR_OPTIONS = (
    (1, 4, 7, 10, 13),
    (0, 3, 6, 9, 12),
)

_GROUP_SLOTS: dict[tuple[int, int], list[tuple[int, int]]] = {
    (0, 0): [(0, 5)],
    (1, 0): [(0, 1), (4, 1)],
    (1, 1): [(1, 3)],
    (2, 0): [(0, 1), (4, 1)],
    (2, 1): [(1, 1), (3, 1)],
    (2, 2): [(2, 1)],
    (3, 0): [(0, 1), (4, 1)],
    (3, 1): [(1, 3)],
    (4, 0): [(0, 5)],
}

_SLOT_LIST = [
    {"row": 0, "layer": 0, "c0": 0, "w": 5},
    {"row": 1, "layer": 0, "c0": 0, "w": 1},
    {"row": 1, "layer": 0, "c0": 4, "w": 1},
    {"row": 1, "layer": 1, "c0": 1, "w": 3},
    {"row": 2, "layer": 0, "c0": 0, "w": 1},
    {"row": 2, "layer": 0, "c0": 4, "w": 1},
    {"row": 2, "layer": 1, "c0": 1, "w": 1},
    {"row": 2, "layer": 2, "c0": 2, "w": 1},
    {"row": 2, "layer": 1, "c0": 3, "w": 1},
    {"row": 3, "layer": 0, "c0": 0, "w": 1},
    {"row": 3, "layer": 0, "c0": 4, "w": 1},
    {"row": 3, "layer": 1, "c0": 1, "w": 3},
    {"row": 4, "layer": 0, "c0": 0, "w": 5},
]

_BAND_BASES = {
    0: (0, {1: [0, 4], 5: [0]}),
    1: (6, {1: [0, 4], 3: [1]}),
    2: (12, {1: [1]}),
}


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _extract_input_pieces(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    raw_pieces: list[dict] = []

    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        for pts in comps:
            arr = np.asarray(pts, dtype=int)
            r0, c0, r1, c1 = bbox_coords(arr)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            assert h == 1, "every fused-variant piece must occupy exactly one input row"
            assert len(arr) == w, "every fused-variant piece must be a filled horizontal fragment"
            raw_pieces.append(
                {
                    "col": int(col),
                    "row_pos": int(r0),
                    "mid_c": (int(c0) + int(c1)) / 2.0,
                    "w": int(w),
                }
            )
    assert raw_pieces, "grid must contain at least one non-background piece"
    return bg, raw_pieces


def _infer_rows(raw_pieces: list[dict]) -> list[int]:
    row_positions = sorted({int(piece["row_pos"]) for piece in raw_pieces})
    assert len(row_positions) == 5, "input must contain exactly five occupied fragment rows"
    row_to_idx = {row: idx for idx, row in enumerate(row_positions)}
    return [row_to_idx[int(piece["row_pos"])] for piece in raw_pieces]


def _infer_layers(raw_pieces: list[dict]) -> list[int]:
    layers = []
    counts = [0, 0, 0]
    for piece in raw_pieces:
        mid = float(piece["mid_c"])
        if mid < _INPUT_BAND_CUTS[0]:
            layer = 0
        elif mid < _INPUT_BAND_CUTS[1]:
            layer = 1
        else:
            layer = 2
        layers.append(layer)
        counts[layer] += 1
    assert all(n > 0 for n in counts), "all three input column bands must be populated"
    return layers


def _build_output_from_groups(groups: dict[tuple[int, int], list[dict]]) -> np.ndarray:
    out = np.full((5, 5), -1, dtype=int)
    seen_keys = set(groups)
    assert seen_keys == set(_GROUP_SLOTS), "row/layer occupancy must match the fused template"

    for key, slots in _GROUP_SLOTS.items():
        pieces = groups[key]
        want_widths = sorted(w for _, w in slots)
        got_widths = sorted(int(piece["w"]) for piece in pieces)
        assert got_widths == want_widths, f"width pattern mismatch for row/layer group {key}"
        colors = {int(piece["col"]) for piece in pieces}
        assert len(colors) == 1, f"symmetric row/layer group {key} must use one color"
        color = next(iter(colors))
        for c0, w in slots:
            out[key[0], c0 : c0 + w] = color

    assert not np.any(out < 0), "output rectangle must be fully covered"
    assert np.array_equal(out, out[:, ::-1]), "output must remain left-right symmetric"
    return out


def _slot_candidate_count(width: int, *, row: int | None = None, layer: int | None = None) -> int:
    count = 0
    for spec in _SLOT_LIST:
        if spec["w"] != int(width):
            continue
        if row is not None and spec["row"] != int(row):
            continue
        if layer is not None and spec["layer"] != int(layer):
            continue
        count += 1
    return count


def fusion_metrics(grid: np.ndarray) -> dict[str, float]:
    _, raw_pieces = _extract_input_pieces(grid)
    rows = _infer_rows(raw_pieces)
    layers = _infer_layers(raw_pieces)

    full_counts = []
    rowless_counts = []
    layerless_counts = []
    row_signal = 0
    layer_signal = 0

    for piece, row, layer in zip(raw_pieces, rows, layers):
        width = int(piece["w"])
        full = _slot_candidate_count(width, row=row, layer=layer)
        without_row = _slot_candidate_count(width, layer=layer)
        without_layer = _slot_candidate_count(width, row=row)
        full_counts.append(full)
        rowless_counts.append(without_row)
        layerless_counts.append(without_layer)
        row_signal += int(without_row > full)
        layer_signal += int(without_layer > full)

    return {
        "pieces": float(len(raw_pieces)),
        "row_groups": float(len(set(rows))),
        "layer_groups": float(len(set(layers))),
        "avg_candidates_full": float(sum(full_counts) / len(full_counts)),
        "avg_candidates_without_row": float(sum(rowless_counts) / len(rowless_counts)),
        "avg_candidates_without_layer": float(sum(layerless_counts) / len(layerless_counts)),
        "row_signal_pieces": float(row_signal),
        "layer_signal_pieces": float(layer_signal),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble row-aligned, column-banded fragments into a layered left-right symmetric square.

    Input structure:
    - The most common color is background; every other 4-connected component is a single-row horizontal fragment.
    - Fragments appear on five ordered input rows and inside three separated column bands.
    - Input row order encodes output row, while input band encodes output layer: outer, inner, then center.

    Rule:
    - Collect every non-background component and infer the output size from the total area.
    - Map each fragment to an output row by its input row rank and to an output layer by its input column band.
    - Place the fragment colors into the canonical 5x5 layered-square slots for that row/layer so the rectangle is dense and exactly left-right symmetric.

    Key insight:
    - The input layout now carries the full abstract address of each fragment: y-position gives which output row it belongs to, and x-band gives which concentric layer it belongs to.
    - Shape matching is reduced to validating the expected width pattern inside each row/layer bucket; only the intended left-right mirror pairing remains ambiguous.
    """
    raw_grid = np.asarray(grid, dtype=int)
    _, raw_pieces = _extract_input_pieces(raw_grid)

    total_area = sum(int(piece["w"]) for piece in raw_pieces)
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1 and best_dims[0] == (5, 5), "fused variant expects total area 25 -> output 5x5"

    rows = _infer_rows(raw_pieces)
    layers = _infer_layers(raw_pieces)

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for piece, row, layer in zip(raw_pieces, rows, layers):
        groups[(int(row), int(layer))].append(piece)

    out = _build_output_from_groups(groups)
    return out.astype(int)


def _build_output() -> tuple[np.ndarray, int]:
    abstract = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 2, 2, 2, 1],
            [0, 3, 4, 3, 0],
            [1, 2, 2, 2, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    cmap = {idx: palette[idx] for idx in range(5)}
    out = np.vectorize(lambda x: cmap[int(x)])(abstract).astype(int)
    return out, int(bg)


def _extract_patches_by_row_layer(out: np.ndarray, bg: int) -> dict[tuple[int, int], list[np.ndarray]]:
    patches: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for col in sorted(set(map(int, np.unique(out))) - {bg}):
        for pts in components_mask(out == col, connectivity=4):
            arr = np.asarray(pts, dtype=int)
            r0, c0, r1, c1 = bbox_coords(arr)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            assert h == 1 and len(arr) == w, "generator output pieces must stay single-row and horizontal"
            layer = int(min(r0, 4 - r0, c0, 4 - c0))
            patch = np.full((1, w), bg, dtype=int)
            patch[0, :] = col
            patches[(int(r0), layer)].append(patch)
    return patches


def _scatter(out: np.ndarray, bg: int) -> np.ndarray:
    canvas = np.full((_CANVAS_ROWS, _CANVAS_COLS), bg, dtype=int)
    row_anchors = random.choice(_ROW_ANCHOR_OPTIONS)
    patches_by_group = _extract_patches_by_row_layer(out, bg)

    for row in range(5):
        anchor_r = int(row_anchors[row])
        for layer in range(3):
            pieces = patches_by_group.get((row, layer), [])
            if not pieces:
                continue
            base_c, slots_by_width = _BAND_BASES[layer]
            slots = slots_by_width[int((pieces[0] != bg).sum())]
            assert len(slots) == len(pieces), "row/layer group must fit its input placement slots"
            for patch, local_c in zip(pieces, slots):
                paste(canvas, patch, anchor_r, base_c + local_c, transparent=bg)

    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg = _build_output()
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if len(np.unique(out)) != 5:
        return None
    if _piece_count(out) != 13:
        return None

    try:
        inp = _scatter(out, bg)
        got = main(inp)
        metrics = fusion_metrics(inp)
    except Exception:
        return None

    if not np.array_equal(got, out):
        return None
    if metrics["row_groups"] != 5 or metrics["layer_groups"] != 3:
        return None
    if metrics["row_signal_pieces"] <= 0 or metrics["layer_signal_pieces"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
