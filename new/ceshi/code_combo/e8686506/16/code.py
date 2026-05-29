# Combo variant of e8686506 (index 16)
# combo_spec: D4/3+D12/1
# Name: 对角线对称+允许水平翻转

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Base task: pack scattered pieces into a portrait rectangle with LR reflection and fixed
#      piece orientation.
#    - Combo task: the assembled target is a square satisfying board[r, c] == board[c, r], and
#      any scattered non-frame piece may need a horizontal flip before it fits.
#    - Small trace: if a 2x3 chunk appears in input as {(0,0),(1,0),(1,1)} but its output slot
#      needs {(0,1),(1,0),(1,1)}, the old solver rejects it because orientation is fixed. Here the
#      solver must accept the hflipped variant, and once it places the chunk at (4,1), diagonal
#      symmetry forces the color at (1,4) rather than at the LR mirror cell (4,4).
# b) Complexity calibration:
#    - A plain diagonal-only fill would be too loose, while unrestricted transforms would branch
#      too much. The generator therefore fixes a visible 2-color X-shaped frame on a 6x6 square,
#      uses only horizontal flip as the extra transform, and requires several horizontally
#      asymmetric fill chunks so the flip degree of freedom is observable in every sample.
# c) Hidden assumptions audited:
#    - The original closest-to-square rectangle logic allowed H != W; diagonal symmetry requires
#      H == W.
#    - Exact piece signatures no longer identify a type because input fragments may be hflipped;
#      types are keyed by a horizontal-flip canonical form.
#    - The old border-only singleton heuristic is invalid because diagonal frame cells are interior
#      4-connected singletons.
#    - LR-specific edge-bar and side-line anchors no longer correspond to the target geometry, so
#      only the corner-singleton hint is retained, and the diagonal frame carries the real anchor.
#    - Pure propagation can stall once flip variants are introduced, so the solver keeps forced
#      placements first and falls back to a small bounded search.
# d) Placement legality:
#    - The generator constructs the output first, extracts 4-connected non-frame patches, and
#      scatters them outside the blocked frame bbox with one-cell 8-neighbor padding.
#    - Horizontal flipping is applied to cropped patches only, so bounds and colors are preserved.
# e) Visual observability:
#    - The output visibly reads as transpose-symmetric because the frame uses different colors on
#      the two diagonals, which also prevents accidental collapse back to LR symmetry.
#    - At least two non-frame patches are horizontally asymmetric, and at least one of them is
#      actually flipped in the input, so the extra transform is not a hidden implementation detail.
# f) Visual cleanliness:
#    - The preserved frame stays inside one clear square bbox, while all other pieces are isolated
#      around it on the background. No markers compete for the same cells.
# g) New solver assumptions guaranteed by the generator:
#    - The frame group is never flipped and always spans the full output bbox.
#    - Non-frame pieces come from strict lower-triangle chunks mirrored by transpose, so every
#      output remains diagonally symmetric and each individual piece area stays small.
#    - There are enough hflip-distinct patches that a no-flip diagonal solver fails on many inputs,
#      which is the intended non-degenerate behavior of the combo.


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _coords_hw(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.array(coords, dtype=int)
    return int(arr[:, 0].max() + 1), int(arr[:, 1].max() + 1)


def _flip_coords_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, w = _coords_hw(coords)
    return tuple(sorted((int(r), int(w - 1 - c)) for r, c in coords))


def _canonical_piece_key(col: int, coords: tuple[tuple[int, int], ...]) -> tuple[int, tuple[tuple[int, int], ...]]:
    return int(col), min(coords, _flip_coords_h(coords))


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _build_requirements(
    n: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell color requirements (req[r, c] == -1 means unconstrained).

    The base task's LR-specific edge bars and side lines are not geometrically stable after
    switching to diagonal symmetry, so this combo keeps only the optional corner-singleton hint.
    """
    req = np.full((n, n), -1, dtype=int)
    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(comp) == 1 for comp in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    if corner_colors:
        col = int(corner_colors[0])
        req[0, 0] = col
        req[0, n - 1] = col
        req[n - 1, 0] = col
        req[n - 1, n - 1] = col
    return req


def _assemble(
    types: list[dict],
    n: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    start_board = np.full((n, n), -1, dtype=int)
    start_counts = [int(t["cnt"]) for t in types]
    node_budget = 5000
    nodes = 0

    def transpose_ok(board: np.ndarray, r: int, c: int, col: int) -> bool:
        v = int(board[c, r])
        if v != -1 and v != col:
            return False
        mirror_req = int(req[c, r])
        return mirror_req in (-1, col)

    def can_place_variant(board: np.ndarray, ti: int, variant: dict, r0: int, c0: int) -> bool:
        col = int(types[ti]["col"])
        for dr, dc in variant["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not transpose_ok(board, r, c, col):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < n and board[rr, cc] == col:
                        return False
        return True

    def placements_for(board: np.ndarray, ti: int) -> list[dict]:
        seen: dict[tuple[tuple[int, int], ...], dict] = {}
        for variant in types[ti]["variants"]:
            for r0 in range(n - int(variant["h"]) + 1):
                for c0 in range(n - int(variant["w"]) + 1):
                    if not can_place_variant(board, ti, variant, r0, c0):
                        continue
                    cells = tuple(sorted((r0 + dr, c0 + dc) for dr, dc in variant["coords"]))
                    if cells not in seen:
                        seen[cells] = {"cells": cells}
        return list(seen.values())

    def place_candidate(board: np.ndarray, ti: int, cand: dict) -> np.ndarray:
        out = board.copy()
        col = int(types[ti]["col"])
        for r, c in cand["cells"]:
            out[r, c] = col
        return out

    def batch_compatible(ti: int, places: list[dict]) -> bool:
        occupied = set()
        col = int(types[ti]["col"])
        for cand in places:
            cells = cand["cells"]
            if any(cell in occupied for cell in cells):
                return False
            if disallow_touch and col in disallow_touch:
                for r, c in cells:
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if (rr, cc) in occupied:
                            return False
            occupied.update(cells)
        return True

    def solve(board: np.ndarray, counts: list[int]) -> np.ndarray | None:
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return None

        cached: dict[int, list[dict]] = {}
        while True:
            progress = False
            cached = {}
            for ti in range(len(types)):
                if counts[ti] <= 0:
                    continue
                places = placements_for(board, ti)
                if not places:
                    return None
                cached[ti] = places
            if not cached:
                return board

            for ti, places in cached.items():
                if len(places) == 1:
                    counts = counts.copy()
                    counts[ti] -= 1
                    board = place_candidate(board, ti, places[0])
                    progress = True
                    break
                if len(places) == counts[ti] and batch_compatible(ti, places):
                    counts = counts.copy()
                    for cand in places:
                        counts[ti] -= 1
                        board = place_candidate(board, ti, cand)
                    progress = True
                    break
            if not progress:
                break

        if all(c == 0 for c in counts):
            return board

        ti = min(
            cached,
            key=lambda idx: (len(cached[idx]), counts[idx], -len(types[idx]["variants"][0]["coords"])),
        )
        for cand in cached[ti]:
            next_counts = counts.copy()
            next_counts[ti] -= 1
            solved = solve(place_candidate(board, ti, cand), next_counts)
            if solved is not None:
                return solved
        return None

    solved = solve(start_board, start_counts)
    assert solved is not None, "assembly dead end"
    assert np.array_equal(solved, solved.T), "assembled board must be diagonally symmetric"
    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a square whose colors match under row/column swap, allowing horizontal flips.

    Input structure:
    - The dominant color is background; every other 4-connected monochrome component is a puzzle piece.
    - A preserved 1-2 part diagonal-symmetric frame already occupies its final relative arrangement and spans the full square bbox.
    - Unlike the base task, scattered non-frame pieces may appear horizontally mirrored relative to the slot they ultimately fill.

    Rule:
    - Extract all non-background pieces and infer the output size from total area; the unique closest-to-square factor pair must be square.
    - Lock the diagonal frame cells, then place each remaining piece either as-is or horizontally flipped.
    - Fill the square so the final board satisfies board[r, c] == board[c, r], while different pieces of the same color never become 4-neighbors.

    Key insight: the anchored X-shaped frame reveals the target orientation, transpose symmetry couples each cell to its swapped coordinate, and irregular chunk contours determine which scattered pieces must be hflipped before they can satisfy those paired constraints.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[int(col)] = comps
        color_piece_count[int(col)] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((int(col), coords))

    ctr = Counter(_canonical_piece_key(col, coords) for col, coords in pieces)
    types = []
    for (col, canon_coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        variant_coords = sorted({canon_coords, _flip_coords_h(canon_coords)})
        variants = []
        for coords in variant_coords:
            h, w = _coords_hw(coords)
            variants.append({"coords": coords, "h": h, "w": w})
        types.append({"col": int(col), "variants": variants, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["variants"][0]["coords"]) for t in types)
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square output shape must be unique"
    h, w = best_dims[0]
    assert h == w, "diagonal symmetry requires a square output"
    n = h

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_diag_symmetric(parts: list[tuple[int, np.ndarray]], size: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((size, size), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp.T)

    frame_parts: list[tuple[int, np.ndarray]] = []
    best_frame_area = -1
    for i, (c1, p1) in enumerate(comps8_all):
        candidates = [[(c1, p1)]]
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            candidates.append([(c1, p1), (c2, p2)])
        for parts in candidates:
            union = np.concatenate([p for _, p in parts])
            r0, c0, r1, c1b = bbox_coords(union)
            if r1 - r0 + 1 != n or c1b - c0 + 1 != n:
                continue
            if not _is_diag_symmetric(parts, n):
                continue
            area = sum(len(p) for _, p in parts)
            if area > best_frame_area:
                best_frame_area = area
                frame_parts = parts
    assert frame_parts, "input must contain 1-2 diagonal-symmetric frame parts spanning the output square"

    req = _build_requirements(n, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < n and 0 <= oc < n:
                req[or_, oc] = int(col)

    disallow_touch = {col for col, count in color_piece_count.items() if count > 1}
    sol = _assemble(types, n, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (diagonal + hflip combo) ---------------

_N = 6

_N_COLORS_MIN = 5
_N_COLORS_MAX = 5
_PIECE_COUNT_MIN = 18
_PIECE_COUNT_MAX = 24
_MAX_PIECE_AREA = 4
_MAX_CHUNK_SIZE = 4
_MIN_FLIPPABLE_PATCHES = 2
_MIN_FLIPPED_PATCHES = 1
_HFLIP_PROB = 0.55

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_GEN_MAX_RETRIES = 400


def _norm_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _cells_hflip_distinct(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> bool:
    coords = _norm_cells(cells)
    return coords != _flip_coords_h(coords)


def _pick_chunk(remaining: set[tuple[int, int]], prefer_asym: bool) -> set[tuple[int, int]]:
    best = None
    best_score = None
    rem_list = list(remaining)
    for _ in range(24):
        seed = random.choice(rem_list)
        target_max = min(_MAX_CHUNK_SIZE, len(remaining))
        target_min = 2 if target_max >= 2 else 1
        target = random.randint(target_min, target_max)
        chunk = {seed}
        frontier = {seed}
        while frontier and len(chunk) < target:
            nbrs = []
            for r, c in frontier:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (rr, cc) in remaining and (rr, cc) not in chunk:
                        nbrs.append((rr, cc))
            if not nbrs:
                break
            nxt = random.choice(nbrs)
            chunk.add(nxt)
            frontier.add(nxt)
        asym = _cells_hflip_distinct(chunk)
        width2 = len({c for _, c in chunk}) >= 2
        score = (int(asym), int(width2), len(chunk))
        if prefer_asym and asym and width2:
            return chunk
        if best_score is None or score > best_score:
            best = chunk
            best_score = score
    return best if best is not None else {random.choice(rem_list)}


def _choose_fill_color(grid: np.ndarray, cells: list[tuple[int, int]], fill_pool: list[int]) -> int:
    avoid = set()
    H, W = grid.shape
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] >= 0:
                avoid.add(int(grid[rr, cc]))
    opts = [col for col in fill_pool if col not in avoid]
    if not opts:
        opts = fill_pool
    return random.choice(opts)


def _build_output() -> tuple[np.ndarray | None, set[int]]:
    palette = list(range(_N_COLORS_MAX))
    random.shuffle(palette)
    main_col, anti_col = palette[:2]
    fill_pool = palette[2:]
    if len(fill_pool) < 2:
        return None, set()

    out = np.full((_N, _N), -1, dtype=int)
    for i in range(_N):
        out[i, i] = main_col
        out[i, _N - 1 - i] = anti_col

    remaining = {
        (r, c)
        for r in range(_N)
        for c in range(_N)
        if r > c and out[r, c] == -1
    }

    asym_chunks = 0
    while remaining:
        chunk = _pick_chunk(remaining, prefer_asym=asym_chunks < _MIN_FLIPPABLE_PATCHES)
        full_cells = [(r, c) for r, c in chunk] + [(c, r) for r, c in chunk]
        col = _choose_fill_color(out, full_cells, fill_pool)
        for r, c in chunk:
            out[r, c] = col
            out[c, r] = col
        asym_chunks += int(_cells_hflip_distinct(chunk))
        remaining -= chunk

    if asym_chunks < _MIN_FLIPPABLE_PATCHES:
        return None, set()
    if np.any(out < 0):
        return None, set()
    if np.array_equal(out, out[:, ::-1]):
        return None, set()
    assert np.array_equal(out, out.T)
    return out, {main_col, anti_col}


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    out, abstract_fcs = _build_output()
    if out is None:
        return None, 0, set()
    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out)
    return remapped.astype(int), bg, real_fcs


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if int(col) in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = int(col)
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _is_hflip_distinct_patch(patch: np.ndarray) -> bool:
    return not np.array_equal(patch, patch[:, ::-1])


def _scatter(out: np.ndarray, bg: int, fcs: set[int], patches: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for preserved frame bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    flippable_idx = [i for i, patch in enumerate(patches) if _is_hflip_distinct_patch(patch)]
    if len(flippable_idx) < _MIN_FLIPPABLE_PATCHES:
        raise ValueError("not enough hflip-distinct patches")

    chosen = {i for i in flippable_idx if random.random() < _HFLIP_PROB}
    while len(chosen) < _MIN_FLIPPED_PATCHES:
        chosen.add(random.choice(flippable_idx))

    for i, patch in enumerate(patches):
        placed_patch = np.fliplr(patch) if i in chosen else patch
        r, c = random_free_location(
            canvas,
            placed_patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, placed_patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas, len(chosen), len(flippable_idx)


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> dict | None:
    out, bg, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out.T):
        return None
    if np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None

    piece_count = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= piece_count <= _PIECE_COUNT_MAX):
        return None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        if int(col) not in fcs
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None

    patches = _extract_non_frame_patches(out, bg, fcs)
    flippable_count = sum(_is_hflip_distinct_patch(patch) for patch in patches)
    if flippable_count < _MIN_FLIPPABLE_PATCHES:
        return None

    try:
        inp, flipped_count, _ = _scatter(out, bg, fcs, patches)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return {
            "input": inp,
            "output": out,
            "flip_count": int(flipped_count),
            "flippable_count": int(flippable_count),
        }
    return None


def _generate_case() -> dict:
    for _ in range(_GEN_MAX_RETRIES):
        case = _gen_attempt()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid diagonal+hflip e8686506 input")


def generate_input() -> np.ndarray:
    return _generate_case()["input"]
