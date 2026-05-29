# Combo variant of f560132c (index 14)
# combo_spec: D3/0+D7/0
# Name: 迭代拼装+非矩形轮廓

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the D3/0 + D7/0 combo variant
# a) Concrete behavior:
#    The base task scattered four disconnected polyomino sprites, then asked the solver to assemble them in one shot
#    into a solid square and recolor by a 2x2 legend patch. This combo changes both the destination and the assembly
#    process. The destination is now a visibly irregular silhouette chosen from a small family of left-anchored masks:
#      - H: left bar + right bar + center bridge
#      - U: left bar + right bar + bottom bar
#    The legend piece is still the northwest piece and still carries the interior 2x2 palette, but the solver can no
#    longer solve by plain rectangle factorization. It must anchor the legend at the upper-left corner of one candidate
#    irregular mask, then place the remaining pieces strictly by descending area. After each placement, the already
#    assembled prefix must stay connected, and the still-empty part of the target must remain one connected cavity.
#    Example trace:
#      - Old: all 4 pieces were searched together against a full rectangle; only the final cover mattered.
#      - New: piece 0 (legend, largest) is fixed first; piece 1 is the largest remaining piece that can extend the
#        current partial body while preserving one connected residual cavity; piece 2 repeats that test on the new
#        cavity; piece 3 is the final remainder. The intermediate partial bodies are part of the rule, not an accident.
# b) Complexity calibration:
#    One irregular family would be too repetitive; arbitrary silhouettes would be noisy. Two stroke-based families keep
#    the visual language compact while still making the non-rectangular destination obvious across 2-4 examples.
#    Likewise, the iterative constraint is visible because the four piece areas are generated in strict descending order,
#    so the placement order is stable and inspectable by eye.
# c) Broken assumptions from the original code:
#      - `total_area == H * W` no longer holds, because the cropped output bbox contains background notches/void strips.
#      - A single exact-cover over all pieces is no longer faithful to the task; order now matters.
#      - The old solver relied on any valid cover; the combo solver additionally relies on every prefix leaving one
#        connected remainder cavity.
#      - The old generator only needed a valid final tiling. The combo generator must also guarantee strict area order,
#        legend-largest, connected prefix growth, and unique iterative reconstruction.
# d) Placement legality:
#    The generator builds the output first, carves connected pieces from a connected target while keeping the leftover
#    connected at every carving step, then scatters cropped sprites with `random_free_location` so components never
#    overlap or leave the input canvas.
# e) Visual observability:
#    The D3 change is directly visible because outputs are cropped irregular silhouettes rather than full rectangles.
#    The D7 change is also visible: across examples the largest sprite is always the legend anchor, and the remaining
#    pieces are visibly smaller in a strict chain, matching the staged fill of the silhouette's residual cavity.
# f) Visual cleanliness:
#    Each input sprite stays monochrome except for the legend's palette patch. The output contains only the final
#    assembled silhouette with clean background cutouts; there are no auxiliary markers or overlapping annotations.
# g) Color inferability:
#    Recoloring still comes from a concrete 2x2 palette embedded in the legend piece. Output colors are determined by
#    the assembled piece centroids in the destination bbox, so the human-visible cues remain geometric rather than
#    numeric. The palette uses four distinct colors disjoint from the four body colors.
# h) New solver assumptions and generator guarantees:
#      - Exactly one component is multicolored; it is also the unique largest piece.
#      - Candidate outputs belong to the small H/U target catalog below.
#      - The legend occupies the northwest anchor of the destination and contains a full 4x4 corner block, ensuring an
#        interior 2x2 palette slot exists.
#      - The remaining three pieces have strictly descending areas, and there is exactly one valid area-ordered
#        iterative assembly whose every intermediate remainder cavity is connected.


K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
THICKNESS = 4
LEGEND_CORE = 4
TARGET_FAMILIES = ("H", "U")
TARGET_SIDES = {
    "H": (10, 12),
    "U": (10, 12),
}
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 30
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
SEED_JITTER = 0.35
GROW_NOISE = 0.35
GEN_RETRIES = 700


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
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
            out.append((c, int(h), int(w), int(rot_k), False))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, int(h), int(w), int(rot_k), True))
        cur = _rot90(cur)[0]
    return out


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert k > 0 and hh >= k and ww >= k, "legend component must fit kxk map"
    best = None
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    assert best_rc is not None, "legend component must contain a valid kxk color map window"
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
    st: list[tuple[int, int]] = []

    for rr in range(hh):
        for cc in range(ww):
            if rr in (0, hh - 1) or cc in (0, ww - 1):
                if bgmask[rr, cc] and not vis[rr, cc]:
                    vis[rr, cc] = True
                    st.append((rr, cc))
                    while st:
                        r, c = st.pop()
                        for r2, c2 in neighbors4(r, c):
                            if 0 <= r2 < hh and 0 <= c2 < ww and bgmask[r2, c2] and not vis[r2, c2]:
                                vis[r2, c2] = True
                                st.append((r2, c2))

    holes = bgmask & ~vis
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(map(tuple, pts)).union(hole_coords)))
    return filled, int(holes.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k_rot in range(4):
        rot = np.rot90(arr, k_rot)
        for oriented in (rot, np.fliplr(rot)):
            r0, c0, r1, c1 = bbox_coords(np.argwhere(oriented > 0))
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((int(r), int(c)))
    return spots


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


def _target_mask(family: str, side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mid = (side - THICKNESS) // 2
    if family == "H":
        mask[:, :THICKNESS] = True
        mask[:, -THICKNESS:] = True
        mask[mid : mid + THICKNESS, :] = True
    elif family == "U":
        mask[:, :THICKNESS] = True
        mask[:, -THICKNESS:] = True
        mask[-THICKNESS:, :] = True
    else:
        raise ValueError(f"unknown target family: {family}")
    return mask


def _candidate_target_infos(total: int, max_h: int, max_w: int) -> list[dict]:
    out = []
    for family in TARGET_FAMILIES:
        for side in TARGET_SIDES[family]:
            if side < max_h or side < max_w:
                continue
            mask = _target_mask(family, side)
            if int(mask.sum()) != total:
                continue
            out.append({"family": family, "side": side, "mask": mask})
    return out


def _piece_cell(r0: int, c0: int, coords: tuple[tuple[int, int], ...], side: int) -> tuple[int, int]:
    rs = [r0 + dr for dr, _ in coords]
    cs = [c0 + dc for _, dc in coords]
    rr = min(K - 1, int((float(sum(rs)) / float(len(rs))) * K / side))
    cc = min(K - 1, int((float(sum(cs)) / float(len(cs))) * K / side))
    return int(rr), int(cc)


def _mask_component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return len(components_mask(mask, connectivity=4))


def _placement_contact_count(used: np.ndarray, r0: int, c0: int, coords: tuple[tuple[int, int], ...]) -> int:
    contacts = 0
    for dr, dc in coords:
        rr = r0 + dr
        cc = c0 + dc
        for r2, c2 in neighbors4(rr, cc):
            if 0 <= r2 < used.shape[0] and 0 <= c2 < used.shape[1] and used[r2, c2]:
                contacts += 1
                break
    return int(contacts)


def _placements_for_piece(piece: dict, target_mask: np.ndarray, pos_to_bit: dict[tuple[int, int], int]) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    side = int(target_mask.shape[0])
    opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
    for coords, ph, pw, _, _ in piece["oris"]:
        for r0 in range(0, side - ph + 1):
            for c0 in range(0, side - pw + 1):
                m = 0
                ok = True
                for dr, dc in coords:
                    bit = pos_to_bit.get((int(r0 + dr), int(c0 + dc)))
                    if bit is None:
                        ok = False
                        break
                    m |= 1 << bit
                if ok:
                    opts.append((m, int(r0), int(c0), coords))
    return opts


def _solve_iterative_on_target(
    pieces: list[dict],
    legend_idx: int,
    target_mask: np.ndarray,
    solution_limit: int = 1,
) -> list[dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]]:
    side = int(target_mask.shape[0])
    target_cells = [tuple(map(int, rc)) for rc in np.argwhere(target_mask)]
    pos_to_bit = {rc: i for i, rc in enumerate(target_cells)}
    all_mask = (1 << len(target_cells)) - 1
    remaining_order = sorted((i for i in range(len(pieces)) if i != legend_idx), key=lambda i: (-pieces[i]["area"], i))
    piece_order = [legend_idx] + remaining_order
    placements = {pi: _placements_for_piece(pieces[pi], target_mask, pos_to_bit) for pi in remaining_order}
    solutions: list[dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []

    for coords, _, _, _, _ in pieces[legend_idx]["oris"]:
        legend_mask = 0
        used = np.zeros_like(target_mask, dtype=bool)
        ok = True
        for dr, dc in coords:
            rr = int(dr)
            cc = int(dc)
            bit = pos_to_bit.get((rr, cc))
            if bit is None:
                ok = False
                break
            legend_mask |= 1 << bit
            used[rr, cc] = True
        if not ok:
            continue
        rem = target_mask & ~used
        if rem.any() and _mask_component_count(rem) != 1:
            continue

        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {legend_idx: (legend_mask, 0, 0, coords)}

        def search(step: int, used_bits: int, used_mask: np.ndarray) -> None:
            if len(solutions) >= solution_limit:
                return
            if step == len(piece_order):
                if used_bits == all_mask:
                    solutions.append(chosen.copy())
                return

            pi = piece_order[step]
            options = []
            for pl in placements[pi]:
                m, r0, c0, coords_local = pl
                if m & used_bits:
                    continue
                contact_count = _placement_contact_count(used_mask, r0, c0, coords_local)
                if contact_count == 0:
                    continue
                new_used = used_mask.copy()
                for dr, dc in coords_local:
                    new_used[r0 + dr, c0 + dc] = True
                rem2 = target_mask & ~new_used
                if rem2.any() and _mask_component_count(rem2) != 1:
                    continue
                options.append((-contact_count, r0 + c0, pl, new_used))

            options.sort(key=lambda x: (x[0], x[1]))
            for _, _, pl, new_used in options:
                chosen[pi] = pl
                search(step + 1, used_bits | pl[0], new_used)
                chosen.pop(pi, None)
                if len(solutions) >= solution_limit:
                    return

        search(1, legend_mask, used)
        if len(solutions) >= solution_limit:
            break

    return solutions


def _find_iterative_assemblies(
    pieces: list[dict],
    legend_idx: int,
    solution_limit: int = 1,
) -> list[tuple[dict, dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]]]:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)
    targets = _candidate_target_infos(total, max_h, max_w)
    assert targets, "no candidate irregular target matches the pieces"

    out = []
    for info in targets:
        sols = _solve_iterative_on_target(pieces, legend_idx, info["mask"], solution_limit=max(1, solution_limit - len(out)))
        for sol in sols:
            out.append((info, sol))
            if len(out) >= solution_limit:
                return out
    return out


def _assemble_iterative_irregular_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    assemblies = _find_iterative_assemblies(pieces, legend_idx, solution_limit=1)
    assert assemblies, "iterative irregular assembly failed"
    info, chosen = assemblies[0]
    side = int(info["side"])
    out = np.zeros((side, side), dtype=int)
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        rr, cc = _piece_cell(r0, c0, coords, side)
        col = int(legend_map[rr, cc])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites into an irregular silhouette by staged cavity-filling, then recolor via the legend map.

    Input structure:
    - Background is the most common color.
    - Four disconnected non-background sprites are present; three are monochrome body pieces and one is the legend.
    - The legend is the unique multicolor component and contains an embedded 2x2 palette patch inside a larger body.

    Rule:
    - Reconstruct the four pieces under D4 symmetry into one tight non-rectangular target silhouette from a small family.
    - The legend anchors the northwest corner of that silhouette, and the remaining pieces are inserted in descending area.
    - After each insertion, the assembled prefix must stay connected and the still-empty part of the target must remain a
      single connected cavity; once the silhouette is complete, each piece is recolored by the 2x2 legend entry of the
      bbox quadrant containing that assembled piece's centroid.

    Key insight:
    - The legend still serves as both a piece and a color key, but the decisive clue is no longer "fill the rectangle".
      The solver must infer an irregular destination and respect a specific large-to-small assembly order by tracking how
      each newly placed piece can legally reduce the current cavity.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == NUM_PIECES, "grid must contain exactly four non-background components"

    comp_colors = [set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg) for pts in comps]
    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely identified"
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], K)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})

    largest_piece = max(range(len(pieces)), key=lambda i: pieces[i]["area"])
    assert largest_piece == legend_idx, "legend must be the unique largest piece"
    areas = sorted((int(p["area"]) for p in pieces), reverse=True)
    assert len(set(areas)) == NUM_PIECES, "piece areas must be strictly ordered for iterative assembly"

    return _assemble_iterative_irregular_cover(pieces, legend_idx, legend_map, K)


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    lo = max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total))
    hi = min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))
    return int(lo), int(hi)


def _desired_areas(total: int, min_area: int, max_area: int) -> list[int] | None:
    a0 = int(round(total * 0.35))
    a1 = int(round(total * 0.27))
    a2 = int(round(total * 0.21))
    a3 = int(total - a0 - a1 - a2)
    areas = [a0, a1, a2, a3]
    if any(a < min_area or a > max_area for a in areas):
        return None
    if not (areas[0] > areas[1] > areas[2] > areas[3]):
        return None
    return areas


def _nearest_cell(cells: list[tuple[int, int]], anchor: tuple[int, int]) -> tuple[int, int]:
    ar, ac = anchor
    return min(cells, key=lambda p: abs(p[0] - ar) + abs(p[1] - ac) + random.random() * SEED_JITTER)


def _boundary_cells(mask: np.ndarray, touched: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for r, c in zip(*np.where(mask)):
        for r2, c2 in neighbors4(int(r), int(c)):
            if 0 <= r2 < mask.shape[0] and 0 <= c2 < mask.shape[1] and touched[r2, c2]:
                out.append((int(r), int(c)))
                break
    return out


def _grow_piece(
    remainder: np.ndarray,
    target_area: int,
    anchor: tuple[int, int],
    initial_cells: list[tuple[int, int]],
) -> np.ndarray | None:
    piece = np.zeros_like(remainder, dtype=bool)
    for r, c in initial_cells:
        if not remainder[r, c]:
            return None
        piece[r, c] = True
    if int(piece.sum()) > target_area:
        return None
    leftover = remainder & ~piece
    if leftover.any() and _mask_component_count(leftover) != 1:
        return None

    while int(piece.sum()) < target_area:
        frontier = set()
        for r, c in zip(*np.where(piece)):
            for r2, c2 in neighbors4(int(r), int(c)):
                if 0 <= r2 < remainder.shape[0] and 0 <= c2 < remainder.shape[1] and remainder[r2, c2] and not piece[r2, c2]:
                    frontier.add((int(r2), int(c2)))
        if not frontier:
            return None

        feasible = []
        ar, ac = anchor
        for r, c in frontier:
            leftover[r, c] = False
            ok = (not leftover.any()) or (_mask_component_count(leftover) == 1)
            leftover[r, c] = True
            if not ok:
                continue
            score = abs(r - ar) + abs(c - ac) + random.random() * GROW_NOISE
            feasible.append((score, r, c))
        if not feasible:
            return None

        feasible.sort(key=lambda x: x[0])
        best_pool = feasible[: min(5, len(feasible))]
        _, rr, cc = random.choice(best_pool)
        piece[rr, cc] = True
        leftover[rr, cc] = False

    return piece


def _partition_target_iteratively(target_mask: np.ndarray) -> list[np.ndarray] | None:
    side = int(target_mask.shape[0])
    min_area, max_area = _area_bounds(side)
    areas = _desired_areas(int(target_mask.sum()), min_area, max_area)
    if areas is None:
        return None

    core = [(r, c) for r in range(LEGEND_CORE) for c in range(LEGEND_CORE)]
    if any(not target_mask[r, c] for r, c in core):
        return None

    anchors = [
        (0, 0),
        (0, side - 1),
        (side - 1, 0),
        (side - 1, side - 1),
    ]

    for _ in range(120):
        remainder = target_mask.copy()
        pieces: list[np.ndarray] = []

        p0 = _grow_piece(remainder, areas[0], anchors[0], core)
        if p0 is None:
            continue
        remainder &= ~p0
        pieces.append(p0)

        prefix = p0.copy()
        ok = True
        for idx in (1, 2):
            boundary = _boundary_cells(remainder, prefix)
            if not boundary:
                ok = False
                break
            seed = _nearest_cell(boundary, anchors[idx])
            piece = _grow_piece(remainder, areas[idx], anchors[idx], [seed])
            if piece is None:
                ok = False
                break
            remainder &= ~piece
            prefix |= piece
            pieces.append(piece)
        if not ok:
            continue

        if _mask_component_count(remainder) != 1:
            continue
        pieces.append(remainder.copy())

        if [int(p.sum()) for p in pieces] != areas:
            continue
        if any(_mask_component_count(p) != 1 or _has_hole(p) for p in pieces):
            continue
        if not _palette_spots(_crop_mask(pieces[0])):
            continue

        qcells = []
        for piece in pieces:
            pts = np.argwhere(piece)
            rr = min(K - 1, int(pts[:, 0].mean() * K / side))
            cc = min(K - 1, int(pts[:, 1].mean() * K / side))
            qcells.append((int(rr), int(cc)))
        if qcells[0] != (0, 0):
            continue
        if len(set(qcells)) != NUM_PIECES:
            continue

        prefix = np.zeros_like(target_mask, dtype=bool)
        for piece in pieces[:-1]:
            prefix |= piece
            rem = target_mask & ~prefix
            if rem.any() and _mask_component_count(rem) != 1:
                ok = False
                break
        if not ok:
            continue
        return pieces

    return None


def _scatter_pieces(
    masks: list[np.ndarray],
    body_colors: list[int],
    legend_map: np.ndarray,
) -> np.ndarray | None:
    total = sum(int(m.sum()) for m in masks)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    order = random.sample(range(NUM_PIECES), NUM_PIECES)
    for pi in order:
        oriented = _random_orientation(_crop_mask(masks[pi])) if pi != 0 else _crop_mask(masks[pi])
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _palette_spots(oriented)
            if not spots:
                return None
            rr, cc = random.choice(spots)
            patch[rr : rr + K, cc : cc + K] = legend_map
        try:
            box = np.ones(oriented.shape, dtype=int)
            r0, c0 = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _expected_output(masks: list[np.ndarray], legend_map: np.ndarray) -> np.ndarray:
    side = int(masks[0].shape[0])
    out = np.zeros((side, side), dtype=int)
    for piece in masks:
        pts = np.argwhere(piece)
        rr = min(K - 1, int(pts[:, 0].mean() * K / side))
        cc = min(K - 1, int(pts[:, 1].mean() * K / side))
        out[piece] = int(legend_map[rr, cc])
    return out


def _build_piece_records(grid: np.ndarray) -> tuple[list[dict], int]:
    bg = get_bg(grid)
    comps = components_mask(grid != bg, connectivity=4)
    comp_colors = [set(int(grid[r, c]) for r, c in pts if int(grid[r, c]) != bg) for pts in comps]
    legend_idx = max(range(len(comps)), key=lambda i: len(comp_colors[i]))
    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(grid, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})
    return pieces, legend_idx


def _gen_attempt() -> np.ndarray | None:
    family = random.choice(TARGET_FAMILIES)
    side = random.choice(TARGET_SIDES[family])
    target_mask = _target_mask(family, side)
    masks = _partition_target_iteratively(target_mask)
    if masks is None:
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)
    expected = _expected_output(masks, legend_map)

    grid = _scatter_pieces(masks, body_colors, legend_map)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    pieces, legend_idx = _build_piece_records(grid)
    if legend_idx != max(range(NUM_PIECES), key=lambda i: pieces[i]["area"]):
        return None
    if len(set(int(p["area"]) for p in pieces)) != NUM_PIECES:
        return None
    assemblies = _find_iterative_assemblies(pieces, legend_idx, solution_limit=2)
    if len(assemblies) != 1:
        return None
    info, chosen = assemblies[0]
    if info["family"] != family or info["side"] != side:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo input for f560132c")
