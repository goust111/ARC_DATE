# Combo variant of e8686506 (index 45)
# combo_spec: D5/2+D8/1
# Name: 无颜色相邻约束+仅两个大碎片

import random
from typing import Iterable

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for D5/2 + D8/1
# 1. The original task extracts many single-color fragments and leans on a no-same-color-touching
#    rule to prune placements. This combo removes that pruning entirely and collapses the fragment
#    set into exactly two large multicolor pieces, so grouping by color is no longer valid.
# 2. Concrete behavior shift: instead of packing many small pieces into a mirrored rectangle, we
#    build one LR-symmetric dense rectangle and cut it by a long jagged seam into two connected
#    blobs. The solver therefore matches two translated multicolor patches back into the unique
#    mirrored rectangle.
# 3. Complexity calibration: a straight vertical cut would be too trivial, so the generator uses
#    a row-by-row zigzag seam with multiple switches. The seam is long enough to make edge-fitting
#    the main clue, but its step size is tightly controlled so the objects remain readable.
# 4. Visual observability: the generator forces several same-color contacts across the seam. That
#    makes the merged output visually blur piece boundaries, which directly exhibits the removal of
#    the old color-adjacency restriction instead of quietly degenerating into the original task.
# 5. New solver assumptions, all guaranteed by the generator:
#    - non-background cells form exactly two 4-connected components;
#    - total area has a unique closest-to-square portrait factorization;
#    - each component is used by translation only;
#    - exactly one translated pair tiles the rectangle and yields LR color symmetry.
# 6. Placement legality: each generated piece is cropped from the final rectangle, then scattered
#    onto a larger canvas with transparent background and positive spacing, so pieces never overlap
#    or exit the input bounds.


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _closest_portrait_dims(area: int) -> tuple[int, int]:
    dims = [(h, w) for h, w in _factors(area) if h >= w]
    assert dims, "non-background area must admit a portrait factorization"
    gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == gap]
    assert len(best) == 1, "closest-to-square portrait factor pair must be unique"
    return best[0]


def _crop_piece(grid: np.ndarray, pts: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in pts:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _patch_sig(patch: np.ndarray, bg: int) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
    cells = tuple(
        sorted((int(r), int(c), int(patch[r, c])) for r, c in np.argwhere(patch != bg))
    )
    return int(patch.shape[0]), int(patch.shape[1]), cells


def _patch_area(patch: np.ndarray, bg: int) -> int:
    return int(np.sum(patch != bg))


def _placements(piece: np.ndarray, H: int, W: int, bg: int) -> list[dict]:
    h, w = piece.shape
    assert h <= H and w <= W, "piece bbox must fit inside the reconstructed rectangle"
    cells = [(int(r), int(c), int(piece[r, c])) for r, c in np.argwhere(piece != bg)]
    out = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            occ = np.zeros((H, W), dtype=bool)
            board = np.full((H, W), -1, dtype=int)
            for dr, dc, col in cells:
                occ[r0 + dr, c0 + dc] = True
                board[r0 + dr, c0 + dc] = col
            out.append({"occ": occ, "board": board})
    return out


def _extract_two_pieces(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    comps = components_mask(grid != bg, connectivity=4)
    assert len(comps) == 2, "input must contain exactly two non-background pieces"
    pieces = [_crop_piece(grid, pts, bg) for pts in comps]
    pieces.sort(key=lambda p: (-_patch_area(p, bg), _patch_sig(p, bg)))
    return pieces


def _assemble_two_pieces(pieces: list[np.ndarray], H: int, W: int, bg: int) -> np.ndarray:
    place_a = _placements(pieces[0], H, W, bg)
    place_b = _placements(pieces[1], H, W, bg)
    solutions: dict[bytes, np.ndarray] = {}

    for pa in place_a:
        occ_a = pa["occ"]
        board_a = pa["board"]
        for pb in place_b:
            occ_b = pb["occ"]
            if np.any(occ_a & occ_b):
                continue
            if int(np.sum(occ_a | occ_b)) != H * W:
                continue
            board = np.where(occ_a, board_a, pb["board"])
            if not np.array_equal(board, board[:, ::-1]):
                continue
            solutions.setdefault(board.tobytes(), board.copy())

    assert solutions, "no mirrored rectangle can be formed from the two pieces"
    assert len(solutions) == 1, "assembly must be uniquely determined"
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble two scattered multicolor chunks into one dense rectangle with exact left-right symmetry.

    Input structure:
    - The background is the most common color.
    - Every non-background cell belongs to one of exactly two large 4-connected pieces.
    - Each piece is multicolor, and same-color cells from opposite pieces may legitimately touch after assembly.

    Rule:
    - Treat each non-background connected component as one rigid jigsaw piece.
    - Infer the output rectangle from the total occupied area using the unique closest-to-square portrait factor pair.
    - Translate the two pieces inside that rectangle so they do not overlap, cover it completely, and produce an exact LR mirror image.

    Key insight:
    - Color adjacency is no longer a safe separator. The reliable cue is the long complementary seam between the two large pieces, with symmetry used only to choose the one translation pair that closes the rectangle correctly.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pieces = _extract_two_pieces(g, bg)
    total_area = sum(_patch_area(piece, bg) for piece in pieces)
    H, W = _closest_portrait_dims(total_area)
    out = _assemble_two_pieces(pieces, H, W, bg)
    return out.astype(int)


# --------------- generator ---------------

_OUTPUT_DIMS = [(7, 5), (8, 5), (7, 6), (8, 6), (7, 7), (8, 7)]
_OUTPUT_WEIGHTS = [h * w for h, w in _OUTPUT_DIMS]

_N_COLORS_MIN = 3
_N_COLORS_MAX = 4
_MIN_SEAM_CONTACTS = 2
_MIN_SEAM_SWITCHES = 2

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_GEN_MAX_RETRIES = 400


def _build_cut_path(H: int, W: int) -> list[int] | None:
    choices = sorted({max(1, W // 2 - 1), min(W - 2, W // 2)})
    if len(choices) < 2:
        return None
    path = [random.choice(choices)]
    for _ in range(1, H):
        cur = path[-1]
        opts = [v for v in choices if abs(v - cur) <= 1]
        weights = [1 if v == cur else 2 for v in opts]
        path.append(random.choices(opts, weights=weights, k=1)[0])

    switches = sum(path[i] != path[i - 1] for i in range(1, H))
    left_area = sum(v + 1 for v in path)
    total = H * W
    if switches < _MIN_SEAM_SWITCHES:
        return None
    if len(set(path)) < 2:
        return None
    if abs(2 * left_area - total) > max(4, H):
        return None
    return path


def _owner_from_path(path: Iterable[int], H: int, W: int) -> np.ndarray:
    owner = np.ones((H, W), dtype=int)
    for r, cut in enumerate(path):
        owner[int(r), : int(cut) + 1] = 0
    return owner


def _owner_connected(owner: np.ndarray, label: int) -> bool:
    return len(components_mask(owner == label, connectivity=4)) == 1


def _boundary_edges(owner: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    H, W = owner.shape
    edges = []
    for r in range(H):
        for c in range(W):
            if c + 1 < W and owner[r, c] != owner[r, c + 1]:
                edges.append(((r, c), (r, c + 1)))
            if r + 1 < H and owner[r, c] != owner[r + 1, c]:
                edges.append(((r, c), (r + 1, c)))
    return edges


def _same_color_boundary_contacts(out: np.ndarray, owner: np.ndarray) -> int:
    total = 0
    for (r1, c1), (r2, c2) in _boundary_edges(owner):
        if int(out[r1, c1]) == int(out[r2, c2]):
            total += 1
    return total


def _force_same_color_contacts(out: np.ndarray, owner: np.ndarray, palette: list[int]) -> None:
    H, W = out.shape
    edges = _boundary_edges(owner)
    random.shuffle(edges)
    for (r1, c1), (r2, c2) in edges:
        if _same_color_boundary_contacts(out, owner) >= _MIN_SEAM_CONTACTS:
            return
        col = random.choice(palette)
        cells = {
            (r1, c1),
            (r2, c2),
            (r1, W - 1 - c1),
            (r2, W - 1 - c2),
        }
        for r, c in cells:
            out[int(r), int(c)] = col


def _build_symmetric_output(H: int, W: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = _build_cut_path(H, W)
    if path is None:
        return None, None
    owner = _owner_from_path(path, H, W)
    if not (_owner_connected(owner, 0) and _owner_connected(owner, 1)):
        return None, None

    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    left_w = (W + 1) // 2
    left = np.zeros((H, left_w), dtype=int)
    for r in range(H):
        for c in range(left_w):
            reuse = []
            if r > 0:
                reuse.extend([int(left[r - 1, c])] * 3)
            if c > 0:
                reuse.extend([int(left[r, c - 1])] * 2)
            if r > 0 and c > 0:
                reuse.append(int(left[r - 1, c - 1]))
            if reuse and random.random() < 0.68:
                left[r, c] = random.choice(reuse)
            else:
                left[r, c] = random.choice(palette)

    out = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(left_w):
            val = int(left[r, c])
            out[r, c] = val
            out[r, W - 1 - c] = val

    _force_same_color_contacts(out, owner, palette)
    if len(np.unique(out)) < _N_COLORS_MIN:
        return None, None
    if _same_color_boundary_contacts(out, owner) < _MIN_SEAM_CONTACTS:
        return None, None
    return out, owner


def _remap_colors(out: np.ndarray) -> tuple[np.ndarray | None, int]:
    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    if len(palette) < len(abstract):
        return None, 0
    cmap = {a: palette[i] for i, a in enumerate(abstract)}
    real = np.array([[cmap[int(v)] for v in row] for row in out], dtype=int)
    return real, bg


def _piece_patch_from_owner(out: np.ndarray, owner: np.ndarray, label: int, bg: int) -> np.ndarray:
    pts = np.argwhere(owner == label)
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in pts:
        patch[int(r) - r0, int(c) - c0] = int(out[int(r), int(c)])
    return patch


def _scatter_pieces(patches: list[np.ndarray], bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    order = [0, 1]
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        r0, c0 = random_free_location(
            canvas,
            patch,
            transparent=bg,
            border=1,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r0, c0, transparent=bg)
    return canvas


def _piece_area_balance(owner: np.ndarray) -> tuple[int, int]:
    a = int(np.sum(owner == 0))
    b = int(np.sum(owner == 1))
    return a, b


def _gen_attempt() -> np.ndarray | None:
    H, W = random.choices(_OUTPUT_DIMS, weights=_OUTPUT_WEIGHTS, k=1)[0]
    out_abs, owner = _build_symmetric_output(H, W)
    if out_abs is None or owner is None:
        return None

    a0, a1 = _piece_area_balance(owner)
    if min(a0, a1) < (H * W) // 3:
        return None

    out, bg = _remap_colors(out_abs)
    if out is None:
        return None

    patches = [_piece_patch_from_owner(out, owner, label, bg) for label in (0, 1)]
    try:
        inp = _scatter_pieces(patches, bg)
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
