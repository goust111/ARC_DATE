# Variant of e8686506
# Dimension: D16 — 碎片间的空间关系约束 (meta)
# Change: 碎片在输入中的相对位置不编码拼合信息——拼合完全由边缘几何关系决定... → 碎片按列分组对应拼合层

# Design analysis
# variant_analysis:
#   concrete_behavior:
#     output:
#       - The assembled target stays a dense portrait rectangle with exact left-right symmetry.
#       - To make "layer" explicit, the generator fixes an 8x5 rectangle and partitions it into 3 concentric rings:
#         layer 0 = outer border, layer 1 = inner border, layer 2 = center spine.
#       - Every piece is generated wholly inside one ring. No piece crosses from one ring to another.
#       - The visible result is a clean nested structure: outer corners + outer edges, inner corners + inner edges,
#         and one center component.
#     input:
#       - Non-background fragments are no longer position-free. They are scattered into 3 vertical column bands.
#       - Left band pieces belong to layer 0, middle band pieces belong to layer 1, right band pieces belong to layer 2.
#       - Inside each band, row placement is still arbitrary; only the band index carries assembly information.
#     trace_example:
#       - Original logic: a 1x1 piece or a short vertical bar could be placed wherever mirror symmetry and edge-fit allow.
#       - Variant logic: the same bar first gets assigned to "outer / inner / center" from its input column band, then it is
#         only allowed to land inside that ring of the final rectangle.
#       - Difference: input x-position now removes cross-layer ambiguity before geometric packing starts.
#   complexity_calibration:
#     - There are exactly 3 layer bands in every sample, so the new rule is visible in 2-3 examples.
#     - Each layer contains repeated piece types (e.g. several singletons, paired bars), so the band assignment matters;
#       otherwise many pieces would have valid placements in multiple rings.
#     - The output stays small and regular enough that pure propagation still solves it.
#   hidden_assumption_audit:
#     - Old frame-locking code assumed 1-2 global frame components whose relative input positions should be preserved.
#       That assumption no longer holds, because the variant deliberately scatters all pieces inside layer bands.
#     - Old singleton restriction assumed 1x1 pieces only belong on the outer border. The variant introduces inner-ring
#       singletons, so that restriction must be removed.
#     - Old grouping by (color, shape) alone would merge identical pieces from different layers. The variant must key piece
#       types by (layer, color, shape).
#     - Old solver did not need any input-position clustering. The variant now assumes exactly 3 non-empty x-clusters of pieces;
#       the generator guarantees this with fixed separated column bands.
#   placement_legality:
#     - Generator scatters each layer into its own band sub-canvas using `random_free_location`, so pieces never overlap,
#       never leave bounds, and keep at least 1-cell Chebyshev separation.
#     - Each band is wider than the largest piece it must hold; retries handle unlucky local packings.
#   visual_observability:
#     - Humans can see three piece columns on the input side and three nested layers on the output side.
#     - The mapping is structural, not color-id based: left/middle/right input bands correspond to outer/middle/inner output rings.
#   cleanliness:
#     - Bands are separated by blank columns, so there is no visual mixing between layers in the input.
#     - The output is a tidy nested rectangle rather than an overdrawn composite with overlapping annotations.
#   color_inference:
#     - Colors are only permuted, never ordered by id. The rule depends on geometry plus band membership, not palette rank.
#   solver_generator_contract:
#     - Solver assumes the fixed 8x5 template, fixed input band cuts, and that every generated component belongs to exactly one layer.
#     - Generator enforces all three facts by construction and validates them through `main(input) == output`.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


_INPUT_BAND_CUTS = (5, 11)


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _layer_of_cell(H: int, W: int, r: int, c: int) -> int:
    return int(min(r, H - 1 - r, c, W - 1 - c))


def _infer_piece_layers(raw_pieces: list[dict]) -> list[int]:
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
    assert all(n > 0 for n in counts), "each of the three input column bands must contain at least one piece"
    return layers


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray | None = None,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        layer = int(t["layer"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if _layer_of_cell(H, W, r, c) != layer:
                return False
            if req is not None and int(req[r, c]) != col:
                return False
            if not mirror_ok(r, c, col):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> None:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            board[r0 + dr, c0 + dc] = col

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(H - int(t["h"]) + 1)
            for c0 in range(W - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert len(places) > 0, "assembly dead end: a piece has no valid placement"
            if len(places) == 1:
                counts[ti] -= 1
                place(ti, places[0][0], places[0][1])
                changed = True
            elif len(places) == counts[ti]:
                for r0, c0 in places:
                    counts[ti] -= 1
                    place(ti, r0, c0)
                changed = True

    assert all(c == 0 for c in counts), "assembly not purely propagation-solvable"
    assert not np.any(board < 0), "assembled board must cover the full rectangle"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def _role_requirements(H: int, W: int, types: list[dict]) -> np.ndarray:
    assert (H, W) == (8, 5), "this D16 variant uses the 8x5 layered template so every role is visually distinct"

    outer = [t for t in types if int(t["layer"]) == 0]
    inner = [t for t in types if int(t["layer"]) == 1]
    center = [t for t in types if int(t["layer"]) == 2]

    outer_corner = [t for t in outer if len(t["coords"]) == 1 and int(t["cnt"]) == 4]
    outer_edge_colors = {int(t["col"]) for t in outer if len(t["coords"]) > 1}
    inner_corner = [t for t in inner if len(t["coords"]) == 1 and int(t["cnt"]) == 4]
    inner_edge_colors = {int(t["col"]) for t in inner if not (len(t["coords"]) == 1 and int(t["cnt"]) == 4)}
    center_colors = {int(t["col"]) for t in center}

    assert len(outer_corner) == 1 and len(outer_edge_colors) == 1, "outer layer roles must be uniquely identifiable"
    assert len(inner_corner) == 1 and len(inner_edge_colors) == 1, "inner layer roles must be uniquely identifiable"
    assert len(center_colors) == 1, "center layer must have a unique color"

    outer_corner_col = int(outer_corner[0]["col"])
    outer_edge_col = next(iter(outer_edge_colors))
    inner_corner_col = int(inner_corner[0]["col"])
    inner_edge_col = next(iter(inner_edge_colors))
    center_col = next(iter(center_colors))

    req = np.full((H, W), -1, dtype=int)
    for r in range(H):
        for c in range(W):
            layer = _layer_of_cell(H, W, r, c)
            if layer == 0:
                req[r, c] = outer_corner_col if (r in (0, H - 1) and c in (0, W - 1)) else outer_edge_col
            elif layer == 1:
                req[r, c] = inner_corner_col if (r in (1, H - 2) and c in (1, W - 2)) else inner_edge_col
            else:
                req[r, c] = center_col
    return req


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble column-banded fragments into a left-right symmetric layered rectangle.

    Input structure:
    - The most common color is background; every other 4-connected component is a puzzle fragment.
    - Fragments are scattered into three vertical column bands.
    - Band order encodes the target layer: left band -> outer ring, middle band -> inner ring, right band -> center spine.

    Rule:
    - Collect all non-background components and infer the output rectangle dimensions from total area.
    - Use each fragment's input column band to decide which concentric layer of the rectangle it may occupy.
    - Pack the fragments without rotation so the result is fully filled, exactly left-right symmetric, and same-color pieces never 4-touch.

    Key insight:
    - In this variant the input x-position is no longer irrelevant: it tells you which ring of the final rectangle a fragment belongs to.
    - Once that layer assignment is fixed, mirror symmetry and edge geometry force the remaining placements by propagation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    raw_pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        color_piece_count[col] = len(comps)
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            _, _, coords = _piece_sig(pts)
            raw_pieces.append(
                {
                    "col": int(col),
                    "coords": coords,
                    "mid_c": (c0 + c1) / 2.0,
                }
            )

    total_area = sum(len(p["coords"]) for p in raw_pieces)
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]
    assert (H, W) == (8, 5), "this D16 variant uses the 8x5 template so the three input bands map to distinct roles"

    piece_layers = _infer_piece_layers(raw_pieces)

    from collections import Counter

    ctr = Counter((int(layer), int(piece["col"]), piece["coords"]) for piece, layer in zip(raw_pieces, piece_layers))
    types = []
    for (layer, col, coords), cnt in sorted(
        ctr.items(),
        key=lambda kv: (-len(kv[0][2]), kv[0][0], kv[0][1], kv[0][2]),
    ):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append(
            {
                "layer": int(layer),
                "col": int(col),
                "coords": coords,
                "h": h,
                "w": w,
                "cnt": int(cnt),
            }
        )

    req = _role_requirements(H, W, types)
    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (D16 layered-column variant) ---------------

_CANVAS_ROWS = 15
_CANVAS_COLS = 15
_GEN_MAX_RETRIES = 300

_BANDS = (
    (0, 5),   # outer layer
    (6, 5),   # inner layer
    (12, 3),  # center layer
)
_VALID_DIMS = [(8, 5)]
_DIM_WEIGHTS = [1]


def _build_output(H: int, W: int) -> tuple[np.ndarray, int]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    outer_corner, outer_edge, inner_corner, inner_edge, center = palette[:5]

    out = np.full((H, W), -1, dtype=int)
    for r in range(H):
        for c in range(W):
            layer = _layer_of_cell(H, W, r, c)
            if layer == 0:
                out[r, c] = outer_corner if (r in (0, H - 1) and c in (0, W - 1)) else outer_edge
            elif layer == 1:
                out[r, c] = inner_corner if (r in (1, H - 2) and c in (1, W - 2)) else inner_edge
            else:
                out[r, c] = center
    return out, bg


def _extract_patches_by_layer(out: np.ndarray, bg: int) -> dict[int, list[np.ndarray]]:
    H, W = out.shape
    patches_by_layer = {0: [], 1: [], 2: []}
    for col in sorted(map(int, np.unique(out))):
        for pts in components_mask(out == col, connectivity=4):
            arr = np.asarray(pts, dtype=int)
            layers = {_layer_of_cell(H, W, int(r), int(c)) for r, c in arr}
            assert len(layers) == 1, "generator pieces must stay inside a single layer"
            layer = next(iter(layers))
            r0, c0, r1, c1 = bbox_coords(arr)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in arr:
                patch[int(r) - r0, int(c) - c0] = col
            patches_by_layer[layer].append(patch)

    for layer in patches_by_layer:
        random.shuffle(patches_by_layer[layer])
        patches_by_layer[layer].sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches_by_layer


def _paste_slots(band: np.ndarray, patches: list[np.ndarray], slots: list[tuple[int, int]], bg: int) -> None:
    assert len(patches) <= len(slots), "not enough placement slots for the layer band"
    order = list(slots)
    random.shuffle(order)
    for patch, (r, c) in zip(patches, order):
        paste(band, patch, r, c, transparent=bg)


def _scatter_band(layer: int, patches: list[np.ndarray], bg: int, width: int) -> np.ndarray:
    band = np.full((_CANVAS_ROWS, width), bg, dtype=int)

    singles = [p for p in patches if int((p != bg).sum()) == 1]
    verticals = [p for p in patches if int((p != bg).sum()) > 1 and p.shape[1] == 1]
    horizontals = [p for p in patches if int((p != bg).sum()) > 1 and p.shape[0] == 1 and p.shape[1] > 1]

    if layer == 0:
        _paste_slots(band, verticals, [(1, 0), (8, 0)], bg)
        _paste_slots(band, horizontals, [(1, 2), (13, 2)], bg)
        _paste_slots(band, singles, [(0, 4), (4, 4), (8, 4), (12, 4)], bg)
    elif layer == 1:
        _paste_slots(band, verticals, [(1, 0), (10, 0)], bg)
        _paste_slots(
            band,
            singles,
            [(0, 4), (2, 4), (4, 4), (6, 4), (8, 4), (10, 4), (12, 4), (14, 4)],
            bg,
        )
        assert not horizontals, "inner band should only contain verticals and singletons"
    else:
        assert not singles or (len(singles) == 1 and not verticals), "center band must contain exactly one center piece"
        assert not horizontals, "center band should not contain horizontal pieces"
        slot = (5, 1) if verticals else (7, 1)
        _paste_slots(band, verticals + singles, [slot], bg)

    return band


def _scatter(out: np.ndarray, bg: int) -> np.ndarray:
    canvas = np.full((_CANVAS_ROWS, _CANVAS_COLS), bg, dtype=int)
    patches_by_layer = _extract_patches_by_layer(out, bg)
    for layer, (c0, width) in enumerate(_BANDS):
        band = _scatter_band(layer, patches_by_layer[layer], bg, width)
        paste(canvas, band, 0, c0)
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _column_band_metric(inp: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    bg = get_bg(inp)
    raw_pieces = []
    for col in sorted(set(map(int, np.unique(inp))) - {bg}):
        for pts in components_mask(inp == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            raw_pieces.append({"mid_c": (c0 + c1) / 2.0, "size": len(pts)})

    layers = _infer_piece_layers(raw_pieces)
    H, W = out.shape
    expected = [0, 0, 0]
    for col in sorted(map(int, np.unique(out))):
        for pts in components_mask(out == col, connectivity=4):
            arr = np.asarray(pts, dtype=int)
            layer = _layer_of_cell(H, W, int(arr[0, 0]), int(arr[0, 1]))
            expected[layer] += 1

    observed = [layers.count(k) for k in range(3)]
    return sum(int(a == b) for a, b in zip(observed, expected)), len(expected)


def _gen_attempt() -> np.ndarray | None:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, bg = _build_output(H, W)
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if len(np.unique(out)) != 5:
        return None
    if _piece_count(out) != 17:
        return None
    if any(len(pts) > 6 for col in np.unique(out) for pts in components_mask(out == col, connectivity=4)):
        return None

    try:
        inp = _scatter(out, bg)
        got = main(inp)
    except Exception:
        return None

    if not np.array_equal(got, out):
        return None
    score, total = _column_band_metric(inp, out)
    if score != total:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D16 input")
