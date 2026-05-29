# Variant of e8686506
# Dimension: D10 — 框架碎片的结构特征 (meta)
# Change: 最大碎片为环形/U形等非凸形状（中心有洞或凹陷），其他小碎片填入空隙... → 框架碎片为完整矩形环
#
# Design analysis
# ---------------
# variant_spec:
#   task: assemble scattered pieces into an LR-symmetric rectangle whose border is already a closed rectangular ring
#   input_core:
#     - one largest 4-connected component is a single-color rectangular ring with bbox HxW
#     - its interior is empty in the input; all other components are scattered outside the ring bbox
#     - the scattered fill pieces are the exact components that tile the ring cavity
#   output_core:
#     - keep the original dense HxW portrait rectangle and exact LR mirror symmetry
#     - border cells stay the ring color; only the strict interior (rows 1..H-2, cols 1..W-2) is filled
#   dimensions:
#     - keep width 5 to stay close to the seed; height is chosen from 7..10 so the cavity is visually obvious
#     - total area is still H*W, so the closest-to-square factorization rule remains valid
#   fill structure:
#     - the cavity is a (H-2)x3 rectangle
#     - generator partitions it into horizontal bands of height 1 or 2
#     - each band is either:
#       1. one full-width 3xh rectangle, or
#       2. three rectangles: left 1xh, center 1xh, right 1xh
#     - this preserves LR symmetry while making the reasoning direction “fill the hole” instead of “match the border”
#
# behavior_trace:
#   original_local_step:
#     - detect 1-2 irregular edge pieces whose union already outlines the final rectangle
#     - use those edge cells as hard requirements, then pack every remaining piece into the whole rectangle
#   variant_local_step:
#     - detect one exact rectangular ring whose bbox already is the final rectangle
#     - pre-place the ring on the board and search only inside the ring cavity
#   concrete_example:
#     - if HxW = 8x5, the ring occupies 22 perimeter cells and the cavity is 6x3
#     - a former zigzag-frame sample could place fill pieces on many boundary-adjacent slots
#     - the new sample forbids that: every non-ring piece must land strictly inside rows 1..6, cols 1..3
#
# complexity_calibration:
#   - using only a closed ring plus interior rectangles would become too trivial if the cavity were tiny, so 5x5 is dropped
#   - using unrestricted polyominoes inside a 3-column cavity would create too many accidental alternative packings
#   - the band vocabulary keeps visual variety while keeping the DFS search small and human-readable
#
# hidden_assumptions_review:
#   - old `singletons_on_border` is no longer valid; interior 1x1 pieces are allowed
#   - old frame detection by “1-2 LR-symmetric 8-connected pieces whose union bbox is HxW” is too weak now,
#     because the variant needs an exact closed perimeter, not merely a symmetric outline
#   - old assembly implicitly allowed pieces anywhere inside the output rectangle; the variant must restrict every
#     non-ring placement to the strict interior, otherwise pieces could spill onto the frame logic
#
# placement_legality:
#   - generator reserves the whole ring bbox before scattering small pieces, so no scattered piece overlaps or enters the cavity
#   - ring is pasted back afterward with its hole left as background
#   - solver uses the ring bbox as the output bbox and only considers placements fully inside the cavity
#
# visual_observability:
#   - across examples, the largest object is always a literal rectangle-with-a-hole, so D10 is immediately visible
#   - the hole is not partly prefilled in input, which keeps the causal story legible: “take outside pieces and fill inside”
#   - color choices never depend on color ids; they only enforce local non-merging between vertically adjacent bands
#
# main_assumption_contract:
#   - exactly one full rectangular ring exists and is the only piece whose bbox equals HxW and whose cells are the full perimeter
#   - all remaining components are meant to tile the ring cavity under LR symmetry
#   - generator guarantees these conditions and validates them by oracle-running `main()`

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _ring_coords(H: int, W: int) -> tuple[tuple[int, int], ...]:
    cells = []
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                cells.append((r, c))
    return tuple(cells)


def _is_full_ring(coords: tuple[tuple[int, int], ...], H: int, W: int) -> bool:
    return len(coords) == (2 * H + 2 * W - 4) and coords == _ring_coords(H, W)


def _build_fill_types(
    comps_by_color: dict[int, list[np.ndarray]],
    H: int,
    W: int,
) -> tuple[list[dict], int]:
    ring_idx: tuple[int, int] | None = None
    ring_color: int | None = None
    pieces: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    color_piece_count: dict[int, int] = {}

    for col, comps in comps_by_color.items():
        color_piece_count[int(col)] = len(comps)
        for ci, pts in enumerate(comps):
            h, w, coords = _piece_sig(pts)
            if h == H and w == W and _is_full_ring(coords, H, W):
                assert ring_idx is None, "ring frame must be unique"
                ring_idx = (int(col), int(ci))
                ring_color = int(col)
                continue
            pieces.append((int(col), coords))

    assert ring_idx is not None and ring_color is not None, "input must contain one complete rectangular ring"

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        types.append(
            {
                "col": int(col),
                "coords": coords,
                "h": int(arr[:, 0].max() + 1),
                "w": int(arr[:, 1].max() + 1),
                "cnt": int(cnt),
            }
        )

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    if ring_color in disallow_touch:
        disallow_touch.remove(ring_color)
    return types, int(ring_color), disallow_touch


def _solve_cavity(
    types: list[dict],
    H: int,
    W: int,
    ring_color: int,
    disallow_touch: set[int],
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    for r, c in _ring_coords(H, W):
        board[r, c] = int(ring_color)

    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = int(board[r, mc])
        return v == -1 or v == col

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if not (1 <= r < H - 1 and 1 <= c < W - 1):
                return False
            if board[r, c] != -1:
                return False
            if not mirror_ok(r, c, col):
                return False
            if col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> list[tuple[int, int]]:
        changed = []
        col = int(types[ti]["col"])
        for dr, dc in types[ti]["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        pos = []
        for r0 in range(1, H - int(t["h"])):
            for c0 in range(1, W - int(t["w"])):
                if can_place(ti, r0, c0):
                    pos.append((r0, c0))
        return pos

    def dfs() -> bool:
        if all(c == 0 for c in counts):
            return True

        best_ti = None
        best_places = None
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            if not places:
                return False
            if best_places is None or len(places) < len(best_places):
                best_ti = ti
                best_places = places
                if len(best_places) == 1:
                    break

        assert best_ti is not None and best_places is not None
        for r0, c0 in best_places:
            counts[best_ti] -= 1
            changed = place(best_ti, r0, c0)
            if dfs():
                return True
            unplace(changed)
            counts[best_ti] += 1
        return False

    assert dfs(), "ring cavity is not tileable by the scattered pieces"
    assert not np.any(board < 0), "solution must fill the entire ring cavity"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a left-right symmetric rectangle whose border is already a complete rectangular ring.

    Input structure:
    - The background is the most common color; one largest non-bg component is a closed rectangular ring.
    - All other colored components are detached fill pieces scattered outside that ring's bounding box.

    Rule:
    - Use the total non-bg area to recover the output rectangle dimensions, then locate the unique full ring with that bbox.
    - Keep the ring fixed as the border of the answer and place every remaining piece strictly inside the ring cavity.
    - The finished rectangle must stay exactly left-right symmetric.

    Key insight:
    - The frame is no longer an irregular outline to extend; it is a finished border with one empty cavity, so the task reduces to tiling that cavity with the scattered pieces under mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[np.ndarray]] = {}
    total_area = 0
    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[int(col)] = comps
        total_area += sum(len(pts) for pts in comps)

    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    types, ring_color, disallow_touch = _build_fill_types(comps_by_color, H, W)
    sol = _solve_cavity(types, H, W, ring_color, disallow_touch)
    return sol.astype(int)


# --------------- generator (variant) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 35
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MIN = 7
_HEIGHT_MAX = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_PIECE_COUNT_MIN = 7
_PIECE_COUNT_MAX = 12
_MIN_SPLIT_BANDS = 2

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 400


def _valid_dims() -> list[tuple[int, int]]:
    out = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and _HEIGHT_MIN <= h <= _HEIGHT_MAX:
                out.append((h, w))
    return out


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _random_partition(total: int) -> list[int]:
    parts = []
    rem = total
    while rem > 0:
        if rem == 1:
            parts.append(1)
            rem = 0
        else:
            step = random.choice([1, 2])
            if step > rem:
                step = 1
            parts.append(step)
            rem -= step
    random.shuffle(parts)
    return parts


def _band_mode_assignments(heights: list[int]) -> list[str] | None:
    idxs = list(range(len(heights)))
    random.shuffle(idxs)
    modes = ["bar"] * len(heights)
    split_count = 0
    for i in idxs:
        if random.random() < 0.6:
            modes[i] = "split"
            split_count += 1
    if split_count < _MIN_SPLIT_BANDS:
        for i in idxs:
            if modes[i] == "bar":
                modes[i] = "split"
                split_count += 1
                if split_count >= _MIN_SPLIT_BANDS:
                    break
    piece_count = 1 + sum(3 if m == "split" else 1 for m in modes)
    if not (_PIECE_COUNT_MIN <= piece_count <= _PIECE_COUNT_MAX):
        return None
    return modes


def _choose_band_pattern(
    fill_pool: list[int],
    prev_triplet: tuple[int, int, int] | None,
    mode: str,
) -> tuple[int, int, int] | None:
    cands = []
    if mode == "bar":
        for col in fill_pool:
            trip = (col, col, col)
            if prev_triplet is not None and any(trip[k] == prev_triplet[k] for k in range(3)):
                continue
            cands.append(trip)
    else:
        for side in fill_pool:
            for center in fill_pool:
                if center == side:
                    continue
                trip = (side, center, side)
                if prev_triplet is not None and any(trip[k] == prev_triplet[k] for k in range(3)):
                    continue
                cands.append(trip)
    if not cands:
        return None
    random.shuffle(cands)
    return cands[0]


def _build_output(H: int, W: int) -> np.ndarray | None:
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)
    ring_col = palette[0]
    fill_pool = palette[1:]
    if len(fill_pool) < 3:
        return None

    inner_h = H - 2
    heights = _random_partition(inner_h)
    modes = _band_mode_assignments(heights)
    if modes is None:
        return None

    out = np.full((H, W), ring_col, dtype=int)
    prev_triplet = None
    used_fill: set[int] = set()
    rr = 1
    for h, mode in zip(heights, modes):
        trip = _choose_band_pattern(fill_pool, prev_triplet, mode)
        if trip is None:
            return None
        for dr in range(h):
            out[rr + dr, 1] = trip[0]
            out[rr + dr, 2] = trip[1]
            out[rr + dr, 3] = trip[2]
        prev_triplet = trip
        used_fill.update(trip)
        rr += h

    if rr != H - 1:
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    if len(used_fill) < 3:
        return None
    return out


def _make_output() -> tuple[np.ndarray | None, int]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out = _build_output(H, W)
    if out is None:
        return None, 0

    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    return np.vectorize(lambda x: cmap[int(x)])(out), int(bg)


def _extract_patches(out: np.ndarray, bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    ring_col = int(out[0, 0])
    ring_patch = np.full(out.shape, bg, dtype=int)
    ring_mask = np.zeros(out.shape, dtype=bool)
    ring_mask[0, :] = True
    ring_mask[-1, :] = True
    ring_mask[:, 0] = True
    ring_mask[:, -1] = True
    ring_patch[ring_mask] = ring_col

    patches = []
    inner = out.copy()
    inner[ring_mask] = bg
    for col in sorted(set(map(int, np.unique(inner))) - {bg}):
        for pts in components_mask(inner == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = int(col)
            patches.append(patch)

    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), p.shape[0], p.shape[1]))
    return ring_patch, patches


def _scatter(out: np.ndarray, bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("ring bbox does not fit the canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    ring_patch, patches = _extract_patches(out, bg)
    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    paste(canvas, ring_patch, off_r, off_c, transparent=bg)
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _is_variant_output(out: np.ndarray) -> bool:
    H, W = out.shape
    if W != 5 or H < 7:
        return False
    ring_col = int(out[0, 0])
    if not np.all(out[0, :] == ring_col):
        return False
    if not np.all(out[-1, :] == ring_col):
        return False
    if not np.all(out[:, 0] == ring_col):
        return False
    if not np.all(out[:, -1] == ring_col):
        return False
    inner = out[1:-1, 1:-1]
    return np.all(inner != ring_col)


def _gen_attempt() -> np.ndarray | None:
    out, bg = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not _is_variant_output(out):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    ring_col = int(out[0, 0])
    for col in np.unique(out):
        for pts in components_mask(out == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            if int(col) == ring_col and h == out.shape[0] and w == out.shape[1] and _is_full_ring(coords, h, w):
                continue
            if len(pts) > _MAX_PIECE_AREA:
                return None
    try:
        inp = _scatter(out, bg)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D10 input")
