# Combo variant of f560132c (index 25)
# combo_spec: D1/2+D11/1+D13/1
# Name: 边缘互补匹配+锯齿边界+高度不规则形状

import math
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
SOLVER_DIR = os.path.join(ROOT, "solvers")
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the fused variant:
# a) Behavior trace:
#    Original task: each sprite could be treated mainly as a global polyomino. Once the 4 pieces
#    were packed into the only valid rectangle, recoloring by the legend's 2x2 palette was enough.
#    The interior borders were mostly incidental.
#    Fused task: the output square is still built from 4 scattered sprites, but those sprites are
#    produced by repeatedly transferring boundary cells across the TL/TR/BL/BR interfaces. A small
#    10x10 example starts as four 5x5 quadrants; then a TL cell on the center seam may move into TR,
#    later a neighboring TR cell may move downward into BR, and so on. The resulting pieces develop
#    one-cell teeth, concave bites, narrow bridges, and visible staircase seams. The output still
#    recolors by assembled 2x2 position, but the assembly is now visually justified by local edge
#    complementarity rather than by simple "this piece already looks like a quadrant".
# b) Complexity calibration:
#    Only 4 pieces and one 2x2 palette are kept, so the color rule remains human-readable. The new
#    complexity is concentrated in the silhouettes: many single-cell boundary transfers create enough
#    irregularity to be obvious, but we still preserve the coarse 2x2 corner ownership so examples
#    remain interpretable from 2-4 demonstrations.
# c) Hidden assumptions audit:
#    - The original exact-cover solver assumed only "cover the rectangle exactly". Here we keep that,
#      but we replace the rectangle inference with a guaranteed square side from total area because
#      the generator always builds a square. The generator explicitly enforces that assumption.
#    - The original code could tolerate smooth interior cuts. Here the generator enforces long,
#      jagged shared boundaries and protected corner cores so the legend still has an interior 2x2
#      palette pocket and every piece keeps a stable quadrant identity.
#    - Centroid-to-quadrant recoloring still works, but only because generation checks that the four
#      piece centroids remain in distinct 2x2 cells after roughening.
# d) Placement legality:
#    Generation is output-first. We mutate the partition inside a fixed square while preserving a full
#    partition of the board, then scatter cropped piece patches with random D4 orientation using
#    random_free_location, so pieces never overlap or leave the canvas.
# e) Visual observability:
#    The decisive cue is visible in the outputs: neighboring colors meet on long zigzag seams instead
#    of straight cuts. Inputs also show strongly irregular silhouettes with tabs and bites. The color
#    logic is still recoverable because only four output colors exist and each piece remains tied to a
#    corner-like centroid cell of the assembled square.
# f) Visual cleanliness:
#    No extra markers are added beyond the original legend palette, and the pieces are scattered with
#    padding. Complexity is in the silhouettes, not in layered annotations, so the scene stays legible.
# g) Non-degeneracy target:
#    The generator rejects samples unless seam length significantly exceeds the straight-quadrant
#    baseline and multiple pieces have low bbox fill / multiple degree-1 "tips". That prevents
#    regression to the original smooth-quadrant behavior.


K = 2
NUM_PIECES = K * K
NUM_COLORS = 8

TL = 0
TR = 1
BL = 2
BR = 3
PIECE_IDS = (TL, TR, BL, BR)
ADJ_PAIRS = {frozenset((TL, TR)), frozenset((TL, BL)), frozenset((TR, BR)), frozenset((BL, BR))}

SIDE_CHOICES = [10, 12]
AREA_FRAC_LO = 0.15
AREA_FRAC_HI = 0.38
AREA_ABS_LO = 15
AREA_ABS_HI = 54
MUTATION_ATTEMPTS = 700
TARGET_MOVES = {10: 48, 12: 72}
INPUT_SIZE_LO = 22
INPUT_SIZE_HI = 30
DENSITY_LO = 0.14
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 240


def _normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, _ = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


def _dihedral_orientations(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]] = []
    cur = coords
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), False))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), True))
        cur = _rot90(cur)[0]
    return out


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert hh >= k and ww >= k, "legend must contain a 2x2 palette"
    best_key = None
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best_key is None or key > best_key:
                best_key = key
                best_rc = (rr, cc)
    assert best_rc is not None
    rr, cc = best_rc
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


def _fill_holes_in_component(g: np.ndarray, pts: np.ndarray, bg: int) -> tuple[tuple[tuple[int, int], ...], int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    occ = np.zeros((hh, ww), dtype=bool)
    for r, c in pts:
        occ[r - r0, c - c0] = True

    bgmask = ~occ
    vis = np.zeros_like(bgmask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for rr in range(hh):
        for cc in range(ww):
            if rr not in (0, hh - 1) and cc not in (0, ww - 1):
                continue
            if bgmask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                stack.append((rr, cc))
                while stack:
                    r, c = stack.pop()
                    for r2, c2 in neighbors4(r, c):
                        if 0 <= r2 < hh and 0 <= c2 < ww and bgmask[r2, c2] and not vis[r2, c2]:
                            vis[r2, c2] = True
                            stack.append((r2, c2))

    holes = bgmask & ~vis
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(map(tuple, pts)).union(hole_coords)))
    return filled, int(holes.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _color_patch_variants(patch: np.ndarray) -> list[np.ndarray]:
    opts: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()
    for k in range(4):
        rot = np.rot90(patch, k)
        for oriented in (rot, np.fliplr(rot)):
            pts = np.argwhere(oriented != 0)
            r0, c0, r1, c1 = bbox_coords(pts)
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].copy()
            key = (tuple(crop.shape), tuple(int(x) for x in crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return opts


def _random_patch_orientation(patch: np.ndarray) -> np.ndarray:
    return random.choice(_color_patch_variants(patch))


def _piece_mask_stats(mask: np.ndarray) -> tuple[int, float, int]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    area = int(mask.sum())
    fill = float(area) / float(bbox_area)
    perimeter = 0
    tips = 0
    for r, c in pts:
        deg = 0
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                deg += 1
            else:
                perimeter += 1
        if deg == 1:
            tips += 1
    return perimeter, fill, tips


def _has_hole(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return True
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~sub
    if not empty.any():
        return False
    for comp in components_mask(empty, connectivity=4):
        if not (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == sub.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == sub.shape[1] - 1)
        ):
            return True
    return False


def _assemble_square_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    total = sum(int(p["area"]) for p in pieces)
    side = int(round(total**0.5))
    assert side * side == total, "combined piece area must define a perfect square"
    H = W = side
    nbits = H * W
    all_mask = (1 << nbits) - 1

    other_pis = [i for i in range(len(pieces)) if i != legend_idx]
    placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
    for pi in other_pis:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for coords, ph, pw, _, _ in pieces[pi]["oris"]:
            for r0 in range(0, H - ph + 1):
                for c0 in range(0, W - pw + 1):
                    m = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * W + (int(c0) + int(dc))
                        m |= 1 << idx
                    opts.append((m, int(r0), int(c0), coords))
        placements_other.append(opts)

    def solve_with_fixed_legend(
        legend_coords: tuple[tuple[int, int], ...],
    ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
        legend_mask = 0
        for dr, dc in legend_coords:
            legend_mask |= 1 << (int(dr) * W + int(dc))

        filtered = [[pl for pl in opts if (pl[0] & legend_mask) == 0] for opts in placements_other]
        remaining = set(range(len(other_pis)))
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {
            legend_idx: (legend_mask, 0, 0, legend_coords)
        }

        def search(used: int) -> bool:
            if not remaining:
                return used == all_mask
            empty = (~used) & all_mask
            if empty == 0:
                return False
            cell = (empty & -empty).bit_length() - 1
            candidates = []
            for local_idx in sorted(remaining):
                opts_local = []
                for pl in filtered[local_idx]:
                    m = pl[0]
                    if (m >> cell) & 1 and (m & used) == 0:
                        opts_local.append(pl)
                if opts_local:
                    candidates.append((len(opts_local), local_idx, opts_local))
            if not candidates:
                return False

            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, local_idx, opts_local in candidates:
                remaining.remove(local_idx)
                pi = other_pis[local_idx]
                for pl in opts_local:
                    chosen[pi] = pl
                    if search(used | pl[0]):
                        return True
                chosen.pop(pi, None)
                remaining.add(local_idx)
            return False

        if search(legend_mask):
            return chosen
        return None

    chosen = None
    for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
        assert ph <= H and pw <= W
        sol = solve_with_fixed_legend(coords)
        if sol is not None:
            chosen = sol
            break
    assert chosen is not None, "failed to tile the square with the scattered pieces"

    out = np.zeros((H, W), dtype=int)
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        rs = [r0 + dr for dr, _ in coords]
        cs = [c0 + dc for _, dc in coords]
        cr = float(sum(rs)) / float(len(rs))
        cc = float(sum(cs)) / float(len(cs))
        rr = max(0, min(k - 1, int(cr * float(k) / float(H))))
        cc2 = max(0, min(k - 1, int(cc * float(k) / float(W))))
        col = int(legend_map[rr, cc2])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild four scattered interlocking sprites into a compact square, then recolor each whole piece from the legend tile.

    Input structure:
    - The background is the most common color.
    - Four disconnected non-background sprites are scattered on the canvas.
    - Three sprites are monochrome bodies; one legend sprite contains the same kind of body plus an interior 2x2 palette.
    - The piece silhouettes are highly irregular: tabs, bites, narrow bridges, and staircase-like borders are expected.

    Rule:
    - Treat every sprite as a jigsaw piece and allow rotations and reflections.
    - Anchor the legend piece at the output's top-left corner, and assemble all four pieces into the unique tight square with no gaps or overlaps.
    - After assembly, recolor each entire piece by the legend palette entry selected from the piece centroid's 2x2 cell within the square.

    Key insight:
    - No piece is meant to match a clean quadrant on its own. The decisive cue is the complementary zigzag edge geometry between neighboring pieces, and the final visible seams verify that the assembly is correct.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == NUM_PIECES, "input must contain exactly four scattered pieces"

    legend_idx = None
    legend_score = None
    for i, pts in enumerate(comps):
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        key = (len(cols), -len(pts))
        if legend_score is None or key > legend_score:
            legend_score = key
            legend_idx = i
    assert legend_idx is not None and legend_score[0] > 1, "legend piece must be the only multi-color component"

    legend_map = _extract_legend_map(g, comps[legend_idx], K)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        pts_arr = np.asarray(list(filled_abs), dtype=int)
        r0, c0, _, _ = bbox_coords(pts_arr)
        rel = tuple((int(r) - r0, int(c) - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})

    return _assemble_square_cover(pieces, int(legend_idx), legend_map, K)


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    lo = max(AREA_ABS_LO, math.ceil(total * AREA_FRAC_LO))
    hi = min(AREA_ABS_HI, math.floor(total * AREA_FRAC_HI))
    return lo, hi


def _initial_ownership(side: int) -> np.ndarray:
    half = side // 2
    own = np.zeros((side, side), dtype=int)
    own[:half, :half] = TL
    own[:half, half:] = TR
    own[half:, :half] = BL
    own[half:, half:] = BR
    return own


def _protected_cells(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[:4, :4] = True
    mask[:3, -3:] = True
    mask[-3:, :3] = True
    mask[-3:, -3:] = True
    return mask


def _owner_mask(own: np.ndarray, pid: int) -> np.ndarray:
    return own == pid


def _qcells_for_ownership(own: np.ndarray) -> list[tuple[int, int]]:
    side = own.shape[0]
    qcells = []
    for pid in PIECE_IDS:
        pts = np.argwhere(own == pid)
        rr = min(1, int(float(pts[:, 0].mean()) * K / float(side)))
        cc = min(1, int(float(pts[:, 1].mean()) * K / float(side)))
        qcells.append((rr, cc))
    return qcells


def _ownership_metrics(own: np.ndarray) -> dict[str, float | int | list[float] | list[int]]:
    side = own.shape[0]
    seam_len = 0
    for r in range(side):
        for c in range(side):
            if r + 1 < side and own[r, c] != own[r + 1, c]:
                seam_len += 1
            if c + 1 < side and own[r, c] != own[r, c + 1]:
                seam_len += 1

    fills = []
    tips = []
    perims = []
    for pid in PIECE_IDS:
        perimeter, fill, tip_count = _piece_mask_stats(own == pid)
        fills.append(fill)
        tips.append(tip_count)
        perims.append(perimeter)

    return {
        "seam_len": int(seam_len),
        "min_fill": float(min(fills)),
        "avg_fill": float(sum(fills) / len(fills)),
        "tip_total": int(sum(tips)),
        "max_perimeter": int(max(perims)),
        "fills": fills,
        "tips": tips,
    }


def _valid_piece_mask(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=4)) == 1 and not _has_hole(mask)


def _roughen_partition(side: int) -> np.ndarray | None:
    own = _initial_ownership(side)
    protected = _protected_cells(side)
    lo, hi = _area_bounds(side)
    target_moves = TARGET_MOVES[side]
    accepted = 0

    for _ in range(MUTATION_ATTEMPTS):
        if accepted >= target_moves:
            break
        boundary_cells = []
        for r in range(side):
            for c in range(side):
                if protected[r, c]:
                    continue
                src = int(own[r, c])
                neigh_owners = set()
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < side and 0 <= cc < side:
                        dst = int(own[rr, cc])
                        if dst != src and frozenset((src, dst)) in ADJ_PAIRS:
                            neigh_owners.add(dst)
                if neigh_owners:
                    boundary_cells.append((r, c, src, tuple(sorted(neigh_owners))))
        if not boundary_cells:
            return None

        r, c, src, dsts = random.choice(boundary_cells)
        dst = random.choice(dsts)
        src_area = int(np.sum(own == src))
        dst_area = int(np.sum(own == dst))
        if src_area - 1 < lo or dst_area + 1 > hi:
            continue

        cand = own.copy()
        cand[r, c] = dst
        if not _valid_piece_mask(cand == src):
            continue
        if not _valid_piece_mask(cand == dst):
            continue

        qcells = _qcells_for_ownership(cand)
        if len(set(qcells)) != NUM_PIECES:
            continue
        own = cand
        accepted += 1

    if accepted < target_moves // 2:
        return None

    qcells = _qcells_for_ownership(own)
    if len(set(qcells)) != NUM_PIECES:
        return None

    for pid in PIECE_IDS:
        mask = own == pid
        area = int(mask.sum())
        if area < lo or area > hi or not _valid_piece_mask(mask):
            return None

    metrics = _ownership_metrics(own)
    baseline = 2 * side
    if metrics["seam_len"] < baseline + side // 2 + 6:
        return None
    if metrics["min_fill"] > 0.83:
        return None
    if metrics["tip_total"] < side // 2:
        return None
    return own


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((r, c))
    return spots


def _patch_from_mask(mask: np.ndarray, body_color: int, legend_map: np.ndarray | None = None) -> np.ndarray | None:
    cropped = _crop_mask(mask)
    patch = np.zeros(cropped.shape, dtype=int)
    patch[cropped] = body_color
    if legend_map is not None:
        spots = _palette_spots(cropped)
        if not spots:
            return None
        rr, cc = random.choice(spots)
        patch[rr : rr + K, cc : cc + K] = legend_map
    return patch


def _scatter_patches(patches: list[np.ndarray], side: int) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for idx in random.sample(range(len(patches)), len(patches)):
        patch = _random_patch_orientation(patches[idx])
        try:
            r0, c0 = random_free_location(
                occupied,
                np.where(patch != 0, 1, 0),
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch != 0] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _expected_output(own: np.ndarray, legend_map: np.ndarray) -> np.ndarray:
    side = own.shape[0]
    out = np.zeros((side, side), dtype=int)
    qcells = _qcells_for_ownership(own)
    for pid, (rr, cc) in zip(PIECE_IDS, qcells):
        out[own == pid] = int(legend_map[rr, cc])
    return out


def _gen_attempt(return_metrics: bool = False):
    side = random.choice(SIDE_CHOICES)
    own = _roughen_partition(side)
    if own is None:
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    patches = []
    for pid in PIECE_IDS:
        patch = _patch_from_mask(own == pid, body_colors[pid], legend_map if pid == TL else None)
        if patch is None:
            return None
        patches.append(patch)

    grid = _scatter_patches(patches, side)
    if grid is None:
        return None

    expected = _expected_output(own, legend_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if return_metrics:
        metrics = _ownership_metrics(own)
        metrics["side"] = side
        return grid, metrics
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt(return_metrics=False)
        if out is not None:
            return out
    raise RuntimeError("failed to generate fused f560132c variant")
