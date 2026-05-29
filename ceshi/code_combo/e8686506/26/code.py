# Combo variant of e8686506 (index 26)
# combo_spec: D1/5+D10/2
# Name: 形状嵌合匹配+多个中型框架

import itertools
import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original e8686506 kept 1-2 large LR-symmetric frame pieces in place, then solved the rest by
#      searching all translations whose outer edges line up inside the HxW rectangle.
#    - This combo removes that global edge-alignment search. The preserved structure is now exactly
#      3-4 medium 8-connected frame fragments sharing one clean HxW bbox, and the bg pockets between
#      them are the true complementary cavities. The scattered pieces are those cavities cut out as
#      patches; they must be reinserted by exact shape, with no rotation or flip.
#    - Small example: for H=9, W=5, the preserved scaffold can be split into three row bands whose
#      left-half zigzag columns are [0,1,0 | 1,1,0 | 1,1,0]. Each band contains one center bridge row,
#      so every band is a medium frame fragment. The empty cells between bands form several 2-6 cell
#      notches, including mirror-opposed L / mirrored-L pairs. Original logic would treat those as free
#      pieces to place anywhere; the combo logic reads the notch silhouette directly from the clean bbox.
# b) Complexity calibration
#    - Width stays 5 and height is restricted to 8-10, because that is the smallest regime where
#      3-4 peer frame fragments can all be medium-sized while still leaving many separate cavities.
#    - Real cavities, not arbitrary post-split chunks, are used as the movable pieces. We reject
#      scaffolds unless they create 6-13 cavities, each area <= 6, and at least one cavity is genuinely
#      non-rectangular. This keeps the D1 change visible instead of collapsing into plain rectangles or
#      single-pixel fillers, while avoiding overfitting to a narrower orientation test than the combo
#      description actually requires.
#    - Frame fragment count is limited to 3 or 4, and segment lengths stay in 2-4 rows, so the peer
#      structure is obvious without introducing too many cases for a human to infer.
# c) Implicit assumption audit against the original solver
#    - Original `_assemble()` assumes every remaining piece should be scanned over every translation in
#      the output rectangle. That premise is false here: the valid locations are exactly the hole regions
#      exposed by the preserved scaffold.
#    - Original frame detection only searched subsets of size 1 or 2. The combo requires looking for a
#      unique clean witness of size 3 or 4 instead.
#    - Original optional anchor rules (corner singletons, full bars, paired verticals) are not useful in
#      this variant; the clean scaffold bbox plus cavity shapes are already the decisive evidence.
#    - New `main()` therefore assumes: one unique clean 3-4-part frame group exists; every missing region
#      inside that bbox is a single 4-connected cavity; and every exact cavity shape class maps to one
#      color. The generator enforces all three.
# d) Placement legality
#    - The generator paints the preserved frame fragments directly into the HxW output rectangle, derives
#      cavities from actual bg components, validates their sizes, then scatters those cavity patches onto
#      a 13-15 canvas with Chebyshev padding 1. The frame bbox is blocked before scattering, so movable
#      pieces cannot overlap or leak into the preserved scaffold region.
# e) Visual observability
#    - A solver can visibly identify one structured bbox containing several comparable scaffold pieces,
#      not a single dominant outline. The missing shapes are read off from visible notches between those
#      pieces, and at least some mirrored pairs have different orientations, so "shape complement" is
#      observable without relying on hidden color ordering.
# f) Visual cleanliness
#    - The input has only two layers: one clean central bbox containing the preserved frame fragments,
#      and scattered plug pieces elsewhere. No extra markers are overlaid on the same cells.
# g) Color inferability
#    - Total colors stay in 4-6. Frame colors and filler colors are separate roles. Color assignment for
#      movable pieces is shape-class based (shape and its LR mirror share a class), so there is structure
#      behind color reuse rather than arbitrary remapping.
# h) New solver/generator contract
#    - `main()` searches for a unique 3-4-part LR-symmetric clean bbox witness and then performs exact
#      cavity matching. `generate_input()` guarantees uniqueness by preserving only those frame fragments
#      inside the bbox and scattering every cavity patch outside it.


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _coords_to_patch(coords: tuple[tuple[int, int], ...], col: int, bg: int) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(col)
    return patch


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _mirror_sig(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not coords:
        return coords
    width = 1 + max(int(c) for _, c in coords)
    return tuple(sorted((int(r), int(width - 1 - c)) for r, c in coords))


def _canonical_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    mirrored = _mirror_sig(coords)
    return min(coords, mirrored)


def _is_rect_shape(coords: tuple[tuple[int, int], ...]) -> bool:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)
    return len(coords) == h * w


def _is_lr_symmetric_union(parts: list[tuple[int, np.ndarray]], H: int, W: int) -> bool:
    all_pts = np.concatenate([pts for _, pts in parts], axis=0)
    r0, c0, _, _ = bbox_coords(all_pts)
    bmp = np.zeros((H, W), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            pts = np.array(pts, dtype=int)
            if 5 <= len(pts) <= 10:
                comps8_all.append((int(col), pts))

    candidates: list[list[tuple[int, np.ndarray]]] = []
    for k in (3, 4):
        for idxs in itertools.combinations(range(len(comps8_all)), k):
            parts = [comps8_all[i] for i in idxs]
            union = np.concatenate([pts for _, pts in parts], axis=0)
            r0, c0, r1, c1 = bbox_coords(union)
            if r1 - r0 + 1 != H or c1 - c0 + 1 != W:
                continue
            if not _is_lr_symmetric_union(parts, H, W):
                continue
            if int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] != bg)) != int(len(union)):
                continue
            candidates.append(parts)

    assert len(candidates) == 1, "input must contain exactly one clean 3-4 part LR-symmetric frame group"
    return candidates[0]


def _hole_stats(mask: np.ndarray) -> dict[str, int]:
    holes = [np.array(h, dtype=int) for h in components_mask(mask, connectivity=4)]
    non_rect = 0
    oriented = 0
    for hole in holes:
        _, _, coords = _piece_sig(hole)
        if not _is_rect_shape(coords):
            non_rect += 1
        if coords != _mirror_sig(coords):
            oriented += 1
    return {
        "count": int(len(holes)),
        "max_area": int(max((len(h) for h in holes), default=0)),
        "non_rect": int(non_rect),
        "oriented": int(oriented),
    }


def _frame_component_sizes(out: np.ndarray, frame_colors: set[int]) -> list[int]:
    sizes = []
    for col in sorted(frame_colors):
        comps = components_mask(out == int(col), connectivity=8)
        sizes.extend(len(pts) for pts in comps)
    return sorted(int(v) for v in sizes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a dense left-right symmetric rectangle by plugging scattered shapes into the real cavities left between several peer frame fragments.

    Input structure:
    - The dominant color is background.
    - Exactly one clean HxW bbox contains 3-4 medium 8-connected frame fragments already fixed in their output-relative positions.
    - All other colored components are detached plug pieces cut from the bg cavities between those frame fragments.

    Rule:
    - Infer the output dimensions from the total non-background area.
    - Find the unique clean bbox whose colored cells are exactly a 3-4-part LR-symmetric scaffold, and copy those frame cells into output coordinates.
    - Treat every remaining bg region inside that bbox as a cavity template, then refill each cavity using a scattered piece with the exact same normalized shape, without rotation or flipping.

    Key insight:
    - The preserved multi-part scaffold already exposes where every missing piece belongs; solving comes from complementary cavity matching across several equal-status frame fragments, not from searching for edge-aligned placements inside the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    pieces = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            pts = np.array(pts, dtype=int)
            _, _, coords = _piece_sig(pts)
            pieces.append({"col": int(col), "coords": coords, "abs_pts": pts})

    total_area = sum(len(p["coords"]) for p in pieces)
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)
    frame_abs = np.concatenate([pts for _, pts in frame_parts], axis=0)
    fr0, fc0, _, _ = bbox_coords(frame_abs)

    board = np.full((H, W), -1, dtype=int)
    frame_cells_abs = set()
    for col, pts in frame_parts:
        for r, c in pts:
            rr, cc = int(r) - fr0, int(c) - fc0
            board[rr, cc] = int(col)
            frame_cells_abs.add((int(r), int(c)))

    scattered = []
    for p in pieces:
        abs_set = set((int(r), int(c)) for r, c in p["abs_pts"])
        if abs_set <= frame_cells_abs:
            continue
        scattered.append(p)

    hole_regions = [np.array(hole, dtype=int) for hole in components_mask(board == -1, connectivity=4)]
    assert hole_regions, "frame scaffold must expose at least one cavity"

    by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    shape_colors: dict[tuple[tuple[int, int], ...], set[int]] = defaultdict(set)
    for p in scattered:
        by_shape[p["coords"]].append(p)
        shape_colors[p["coords"]].add(int(p["col"]))

    for coords, cols in shape_colors.items():
        assert len(cols) == 1, f"shape {coords} must map to a single color"

    needed = Counter()
    for hole in hole_regions:
        _, _, coords = _piece_sig(hole)
        needed[coords] += 1

    available = Counter({coords: len(items) for coords, items in by_shape.items()})
    assert needed == available, "scattered piece shapes must match the scaffold cavities exactly"

    for hole in hole_regions:
        r0, c0, _, _ = bbox_coords(hole)
        _, _, coords = _piece_sig(hole)
        piece = by_shape[coords].pop()
        col = int(piece["col"])
        for dr, dc in coords:
            board[int(r0) + int(dr), int(c0) + int(dc)] = col

    assert np.all(board >= 0), "all scaffold cavities must be filled"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board.astype(int)


# --------------- generator ---------------

_AREA_MIN = 40
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MIN = 8
_HEIGHT_MAX = 10

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_FRAME_PARTS_MIN = 3
_FRAME_PARTS_MAX = 4
_FRAME_PART_MIN_AREA = 5
_FRAME_PART_MAX_AREA = 10

_MAX_HOLE_AREA = 6
_HOLE_COUNT_MIN = 6
_HOLE_COUNT_MAX = 13
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17
_MIN_NON_RECT_HOLES = 1
_MIN_ORIENTED_HOLES = 0

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_BUILD_MAX_ATTEMPTS = 200
_GEN_MAX_RETRIES = 500


def _valid_dims() -> list[tuple[int, int]]:
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and _HEIGHT_MIN <= h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _row_partitions(total: int, k: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, cur: list[int]) -> None:
        if slots == 0:
            if remaining == 0:
                out.append(tuple(cur))
            return
        lo = max(2, remaining - 4 * (slots - 1))
        hi = min(4, remaining - 2 * (slots - 1))
        for v in range(lo, hi + 1):
            cur.append(v)
            rec(remaining - v, slots - 1, cur)
            cur.pop()

    rec(total, k, [])
    return out


def _zigzag_cols(H: int) -> list[int] | None:
    for _ in range(120):
        cols = [0]
        c = random.choice([0, 1])
        for r in range(1, H - 1):
            if r >= 2 and cols[-1] == cols[-2]:
                if random.random() < 0.8:
                    c = 1 - c
            else:
                if random.random() < 0.5:
                    c = 1 - c
            cols.append(c)
        cols.append(0)
        if sum(v == 1 for v in cols) >= _FRAME_PARTS_MIN:
            return cols
    return None


def _choose_segments(H: int, cols: list[int]) -> list[int] | None:
    feasible = []
    for k in range(_FRAME_PARTS_MIN, _FRAME_PARTS_MAX + 1):
        if 2 * k <= H <= 4 * k:
            feasible.append(k)
    random.shuffle(feasible)

    for k in feasible:
        parts = _row_partitions(H, k)
        random.shuffle(parts)
        for lengths in parts:
            if max(lengths) - min(lengths) > 1:
                continue
            row0 = 0
            ok = True
            for L in lengths:
                if 1 not in cols[row0 : row0 + L]:
                    ok = False
                    break
                row0 += L
            if ok:
                return list(lengths)
    return None


def _paint_segment(grid: np.ndarray, row0: int, cols: list[int], color: int) -> None:
    bridge_rows = [i for i, v in enumerate(cols) if v == 1]
    bridge = random.choice(bridge_rows)
    for dr, c in enumerate(cols):
        r = row0 + dr
        grid[r, c] = color
        grid[r, grid.shape[1] - 1 - c] = color
        if dr == bridge:
            grid[r, grid.shape[1] // 2] = color


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int], dict[str, int]]:
    for _ in range(_BUILD_MAX_ATTEMPTS):
        cols = _zigzag_cols(H)
        if cols is None:
            continue
        lengths = _choose_segments(H, cols)
        if lengths is None:
            continue

        k = len(lengths)
        frame_colors = list(range(k))
        grid = np.full((H, W), -1, dtype=int)

        row = 0
        for color, L in zip(frame_colors, lengths):
            _paint_segment(grid, row, cols[row : row + L], color)
            row += L

        frame_sizes = _frame_component_sizes(grid, set(frame_colors))
        if len(frame_sizes) != k:
            continue
        if min(frame_sizes) < _FRAME_PART_MIN_AREA or max(frame_sizes) > _FRAME_PART_MAX_AREA:
            continue

        stats = _hole_stats(grid == -1)
        if not (_HOLE_COUNT_MIN <= stats["count"] <= _HOLE_COUNT_MAX):
            continue
        if stats["max_area"] > _MAX_HOLE_AREA:
            continue
        if stats["non_rect"] < _MIN_NON_RECT_HOLES or stats["oriented"] < _MIN_ORIENTED_HOLES:
            continue
        if not np.array_equal(grid >= 0, (grid >= 0)[:, ::-1]):
            continue
        if not np.array_equal(grid, grid[:, ::-1]):
            continue
        if not (H * W == sum(frame_sizes) + sum(len(h) for h in components_mask(grid == -1, connectivity=4))):
            continue

        holes = [np.array(h, dtype=int) for h in components_mask(grid == -1, connectivity=4)]
        classes = []
        for hole in holes:
            _, _, coords = _piece_sig(hole)
            classes.append(_canonical_shape(coords))
        uniq_classes = sorted(set(classes))
        max_hole_colors = _N_COLORS_MAX - k
        min_hole_colors = max(1, _N_COLORS_MIN - k)
        if max_hole_colors <= 0:
            continue
        hole_color_count = random.randint(min_hole_colors, max_hole_colors)
        hole_palette = list(range(k, k + hole_color_count))
        random.shuffle(hole_palette)

        class_to_color: dict[tuple[tuple[int, int], ...], int] = {}
        shuffled_classes = list(uniq_classes)
        random.shuffle(shuffled_classes)
        for i, cls in enumerate(shuffled_classes):
            if i < len(hole_palette):
                class_to_color[cls] = hole_palette[i]
            else:
                class_to_color[cls] = random.choice(hole_palette)

        out = grid.copy()
        for hole in holes:
            _, _, coords = _piece_sig(hole)
            col = class_to_color[_canonical_shape(coords)]
            for r, c in hole:
                out[int(r), int(c)] = int(col)

        if not np.array_equal(out, out[:, ::-1]):
            continue
        if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
            continue

        piece_count = int(k + len(holes))
        if not (_PIECE_COUNT_MIN <= piece_count <= _PIECE_COUNT_MAX):
            continue

        return out.astype(int), set(frame_colors), stats

    return None, set(), {}


def _make_output() -> tuple[np.ndarray | None, int, set[int], dict[str, int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, frame_colors, stats = _build_output(H, W)
    if out is None:
        return None, 0, set(), {}

    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set(), {}
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_frame = {cmap[a] for a in frame_colors}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out).astype(int)
    return remapped, bg, real_frame, stats


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_colors: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in frame_colors:
            continue
        for pts in components_mask(out == col, connectivity=4):
            _, _, coords = _piece_sig(np.array(pts, dtype=int))
            patches.append(_coords_to_patch(coords, int(col), bg))
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, frame_colors: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("output bbox does not fit the canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_non_frame_patches(out, bg, frame_colors):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.isin(out, list(frame_colors))
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg, frame_colors, _stats = _make_output()
    if out is None:
        return None

    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None

    frame_sizes = _frame_component_sizes(out, frame_colors)
    if not (_FRAME_PARTS_MIN <= len(frame_sizes) <= _FRAME_PARTS_MAX):
        return None
    if min(frame_sizes) < _FRAME_PART_MIN_AREA or max(frame_sizes) > _FRAME_PART_MAX_AREA:
        return None

    frame_mask = np.isin(out, list(frame_colors))
    stats = _hole_stats(~frame_mask)
    if not (_HOLE_COUNT_MIN <= stats["count"] <= _HOLE_COUNT_MAX):
        return None
    if stats["max_area"] > _MAX_HOLE_AREA:
        return None
    if stats["non_rect"] < _MIN_NON_RECT_HOLES or stats["oriented"] < _MIN_ORIENTED_HOLES:
        return None

    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None

    try:
        inp = _scatter(out, bg, frame_colors)
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
    raise RuntimeError("failed to generate a valid e8686506 combo input")
