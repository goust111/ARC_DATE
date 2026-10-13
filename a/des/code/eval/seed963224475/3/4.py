from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
GREEN = 3
PALETTE = (0, 1, 2, 4, 5, 6, 7, 9)
RNG = random.Random()
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            dq: deque[tuple[int, int]] = deque([(r, c)])
            while dq:
                rr, cc = dq.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected_mask(mask: np.ndarray) -> bool:
    comps = _component_coords(np.asarray(mask, dtype=bool))
    return len(comps) == 1


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _component_coords(g != BG)
    pieces: list[dict] = []
    for comp in comps:
        comp = sorted((int(r), int(c)) for r, c in comp)
        colors = {int(g[r, c]) for r, c in comp}
        non_green = sorted(col for col in colors if col != GREEN)
        assert len(non_green) == 1, "each fragment must have one main color"
        main_color = int(non_green[0])
        r0, c0, r1, c1 = _bbox(comp)
        rel_coords: list[tuple[int, int]] = []
        rel_colors: list[int] = []
        for r, c in comp:
            rel_coords.append((int(r - r0), int(c - c0)))
            rel_colors.append(int(g[r, c]))
        pieces.append(
            {
                "shape": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
                "coords": tuple(rel_coords),
                "colors": tuple(rel_colors),
                "main_color": main_color,
                "area": len(comp),
            }
        )
    pieces.sort(key=lambda p: (-int(p["area"]), p["shape"], p["coords"]))
    return pieces


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for h in range(1, int(n**0.5) + 1):
        if n % h != 0:
            continue
        w = n // h
        out.append((int(h), int(w)))
        if h != w:
            out.append((int(w), int(h)))
    return out


def _first_set_bit(x: int) -> int:
    return int((x & -x).bit_length() - 1)


def _candidate_placements(piece: dict, H: int, W: int, piece_index: int) -> list[dict]:
    ph, pw = piece["shape"]
    rel_coords = piece["coords"]
    rel_colors = piece["colors"]
    placements: list[dict] = []
    for r0 in range(H - ph + 1):
        for c0 in range(W - pw + 1):
            mask = 0
            cell_items: list[tuple[int, int]] = []
            cell_map: dict[int, int] = {}
            for (dr, dc), color in zip(rel_coords, rel_colors):
                rr = int(r0 + dr)
                cc = int(c0 + dc)
                idx = rr * W + cc
                mask |= 1 << idx
                cell_items.append((idx, int(color)))
                cell_map[idx] = int(color)
            placements.append(
                {
                    "piece": int(piece_index),
                    "origin": (int(r0), int(c0)),
                    "mask": mask,
                    "cell_items": tuple(cell_items),
                    "cell_map": cell_map,
                }
            )
    return placements


def _placements_compatible(a: dict, b: dict, H: int, W: int) -> bool:
    if int(a["mask"]) & int(b["mask"]):
        return False
    bmap = b["cell_map"]
    for idx, color_a in a["cell_items"]:
        r = idx // W
        c = idx % W
        if r > 0:
            nb = idx - W
            color_b = bmap.get(nb)
            if color_b is not None and ((int(color_a) == GREEN) == (int(color_b) == GREEN)):
                return False
        if r + 1 < H:
            nb = idx + W
            color_b = bmap.get(nb)
            if color_b is not None and ((int(color_a) == GREEN) == (int(color_b) == GREEN)):
                return False
        if c > 0:
            nb = idx - 1
            color_b = bmap.get(nb)
            if color_b is not None and ((int(color_a) == GREEN) == (int(color_b) == GREEN)):
                return False
        if c + 1 < W:
            nb = idx + 1
            color_b = bmap.get(nb)
            if color_b is not None and ((int(color_a) == GREEN) == (int(color_b) == GREEN)):
                return False
    return True


def _solve_pieces(pieces: list[dict], solution_limit: int = 1) -> list[dict]:
    total_area = sum(int(piece["area"]) for piece in pieces)
    max_h = max(int(piece["shape"][0]) for piece in pieces)
    max_w = max(int(piece["shape"][1]) for piece in pieces)
    piece_count = len(pieces)
    all_piece_mask = (1 << piece_count) - 1

    dims = [
        (H, W)
        for H, W in _factor_pairs(total_area)
        if H >= max_h and W >= max_w
    ]
    dims.sort(key=lambda hw: (abs(hw[0] - hw[1]), hw[0], hw[1]))

    solutions: list[dict] = []
    for H, W in dims:
        full_mask = (1 << (H * W)) - 1
        candidates: list[dict] = []
        piece_to_candidates: list[list[int]] = [[] for _ in pieces]
        cover_by_cell: list[list[int]] = [[] for _ in range(H * W)]

        for pi, piece in enumerate(pieces):
            start = len(candidates)
            for cand in _candidate_placements(piece, H, W, pi):
                cid = len(candidates)
                candidates.append(cand)
                piece_to_candidates[pi].append(cid)
                for idx, _ in cand["cell_items"]:
                    cover_by_cell[idx].append(cid)
            if len(candidates) == start:
                break
        else:
            compat: list[set[int]] = [set() for _ in candidates]
            for i in range(len(candidates)):
                compat[i].add(i)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if candidates[i]["piece"] == candidates[j]["piece"]:
                        continue
                    if _placements_compatible(candidates[i], candidates[j], H, W):
                        compat[i].add(j)
                        compat[j].add(i)

            chosen: list[int] = []

            def search(used_mask: int, used_pieces_mask: int) -> None:
                if len(solutions) >= solution_limit:
                    return
                if used_mask == full_mask:
                    if used_pieces_mask == all_piece_mask:
                        solutions.append({"H": H, "W": W, "candidates": candidates, "chosen": tuple(chosen)})
                    return

                cell = _first_set_bit(full_mask ^ used_mask)
                grouped: dict[int, list[int]] = {}
                for cid in cover_by_cell[cell]:
                    cand = candidates[cid]
                    pi = int(cand["piece"])
                    if (used_pieces_mask >> pi) & 1:
                        continue
                    if int(cand["mask"]) & used_mask:
                        continue
                    if not all(other in compat[cid] for other in chosen):
                        continue
                    grouped.setdefault(pi, []).append(cid)
                if not grouped:
                    return

                for pi in sorted(grouped, key=lambda p: (len(grouped[p]), p)):
                    for cid in grouped[pi]:
                        chosen.append(cid)
                        search(
                            used_mask | int(candidates[cid]["mask"]),
                            used_pieces_mask | (1 << pi),
                        )
                        chosen.pop()
                        if len(solutions) >= solution_limit:
                            return

            search(0, 0)
            if len(solutions) >= solution_limit:
                return solutions[:solution_limit]
    return solutions


def _assemble_solution(pieces: list[dict], solution: dict) -> tuple[np.ndarray, np.ndarray]:
    H = int(solution["H"])
    W = int(solution["W"])
    candidates = solution["candidates"]
    colors = np.full((H, W), -1, dtype=int)
    mains = np.full((H, W), -1, dtype=int)

    for cid in solution["chosen"]:
        cand = candidates[int(cid)]
        main_color = int(pieces[int(cand["piece"])]["main_color"])
        for idx, color in cand["cell_items"]:
            r = idx // W
            c = idx % W
            colors[r, c] = int(color)
            mains[r, c] = main_color

    assert bool((colors >= 0).all())
    return colors, mains


def _symmetry_kind(green_mask: np.ndarray) -> str:
    sym_h = bool(np.array_equal(green_mask, np.flipud(green_mask)))
    sym_v = bool(np.array_equal(green_mask, np.fliplr(green_mask)))
    if sym_h and sym_v:
        return "both"
    if sym_h:
        return "h"
    if sym_v:
        return "v"
    raise AssertionError("green indicator must expose at least one symmetry")


def _apply_symmetry(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "both":
        return np.rot90(arr, 2).copy()
    if kind == "h":
        return np.flipud(arr).copy()
    if kind == "v":
        return np.fliplr(arr).copy()
    raise ValueError(kind)


def _final_output_from_assembled(assembled: np.ndarray, mains: np.ndarray) -> np.ndarray:
    kind = _symmetry_kind(assembled == GREEN)
    moved = _apply_symmetry(assembled, kind)
    moved_main = _apply_symmetry(mains, kind)
    out = moved.copy()
    out[moved == GREEN] = moved_main[moved == GREEN]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments into a rectangle by complementary green seam fingerprints,
    then transform the assembled rectangle according to the seam pattern's symmetry.
    """
    pieces = _extract_pieces(np.asarray(grid, dtype=int))
    assert len(pieces) >= 2, "puzzle requires multiple fragments"

    solutions = _solve_pieces(pieces, solution_limit=1)
    assert solutions, "failed to assemble fragments into a rectangle"

    assembled, mains = _assemble_solution(pieces, solutions[0])
    return _final_output_from_assembled(assembled, mains)


def _valid_boundaries(cuts: list[int], total: int, min_width: int) -> bool:
    prev = -1
    k = len(cuts)
    for j, x in enumerate(cuts):
        lo = (prev + min_width) if j else (min_width - 1)
        hi = total - 1 - (k - j) * min_width
        if not (lo <= x <= hi):
            return False
        prev = x
    return True


def _random_walk_rows(H: int, W: int, cut_count: int, min_width: int = 1) -> list[list[int]] | None:
    extra = W - min_width * (cut_count + 1)
    if extra < 0:
        return None
    slack = [0] * (cut_count + 1)
    for _ in range(extra):
        slack[RNG.randrange(cut_count + 1)] += 1
    widths = [min_width + x for x in slack]
    cuts: list[int] = []
    acc = 0
    for width in widths[:-1]:
        acc += width
        cuts.append(acc - 1)

    top = [cuts]
    for _ in range(1, H // 2 + 1):
        prev = top[-1]
        nxt = None
        for _ in range(50):
            cand = [int(prev[j] + RNG.choice((-1, 0, 1))) for j in range(cut_count)]
            if _valid_boundaries(cand, W, min_width):
                nxt = cand
                break
        if nxt is None:
            return None
        top.append(nxt)
    return top + top[-2::-1]


def _generate_horizontal_branch() -> np.ndarray | None:
    H = RNG.choice((5, 7, 9))
    W = RNG.choice((6, 8, 10))
    cut_count = RNG.choice((2, 3))
    rows = _random_walk_rows(H, W, cut_count, min_width=1)
    if rows is None:
        return None

    labels = np.zeros((H, W), dtype=int)
    for r, cuts in enumerate(rows):
        prev = -1
        for band, x in enumerate(cuts):
            labels[r, prev + 1 : x + 1] = int(band)
            prev = int(x)
        labels[r, prev + 1 :] = int(cut_count)

    if not bool(np.array_equal(labels, np.flipud(labels))):
        return None
    if any(not _is_connected_mask(labels == lab) for lab in range(cut_count + 1)):
        return None
    if not any(_piece_bbox_area(labels == lab) > int(np.sum(labels == lab)) for lab in range(cut_count + 1)):
        return None
    return labels


def _generate_vertical_branch() -> np.ndarray | None:
    H = RNG.choice((6, 8, 10))
    W = RNG.choice((5, 7, 9))
    cut_count = RNG.choice((2, 3))
    cols = _random_walk_rows(W, H, cut_count, min_width=1)
    if cols is None:
        return None

    labels = np.zeros((H, W), dtype=int)
    for c, cuts in enumerate(cols):
        prev = -1
        for band, y in enumerate(cuts):
            labels[prev + 1 : y + 1, c] = int(band)
            prev = int(y)
        labels[prev + 1 :, c] = int(cut_count)

    if not bool(np.array_equal(labels, np.fliplr(labels))):
        return None
    if any(not _is_connected_mask(labels == lab) for lab in range(cut_count + 1)):
        return None
    if not any(_piece_bbox_area(labels == lab) > int(np.sum(labels == lab)) for lab in range(cut_count + 1)):
        return None
    return labels


def _generate_both_branch() -> np.ndarray | None:
    H = RNG.choice((7, 9))
    W = RNG.choice((7, 9))
    min_mid = 3
    max_x = (W - 2 - min_mid) // 2
    max_y = (H - 2 - min_mid) // 2
    if max_x < 0 or max_y < 0:
        return None

    x_top = [RNG.randint(0, max_x)]
    for _ in range(1, H // 2 + 1):
        prev = x_top[-1]
        nxt = None
        for _ in range(40):
            cand = int(prev + RNG.choice((-1, 0, 1)))
            if 0 <= cand <= max_x:
                nxt = cand
                break
        if nxt is None:
            return None
        x_top.append(nxt)
    x_left = x_top + x_top[-2::-1]

    y_left = [RNG.randint(0, max_y)]
    for _ in range(1, W // 2 + 1):
        prev = y_left[-1]
        nxt = None
        for _ in range(40):
            cand = int(prev + RNG.choice((-1, 0, 1)))
            if 0 <= cand <= max_y:
                nxt = cand
                break
        if nxt is None:
            return None
        y_left.append(nxt)
    y_top = y_left + y_left[-2::-1]

    labels = np.zeros((H, W), dtype=int)
    for r in range(H):
        xl = int(x_left[r])
        xr = int(W - 2 - xl)
        for c in range(W):
            yt = int(y_top[c])
            yb = int(H - 2 - yt)
            if c <= xl:
                labels[r, c] = 0
            elif c > xr:
                labels[r, c] = 1
            elif r <= yt:
                labels[r, c] = 2
            elif r > yb:
                labels[r, c] = 3
            else:
                labels[r, c] = 4

    if any(not _is_connected_mask(labels == lab) for lab in range(5)):
        return None
    if not bool(np.array_equal(labels != 4, np.flipud(labels != 4))):
        return None
    if not bool(np.array_equal(labels != 4, np.fliplr(labels != 4))):
        return None
    return labels


def _piece_bbox_area(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int((rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1))


def _labels_to_case(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    piece_count = int(labels.max()) + 1
    if piece_count < 3:
        return None

    colors = RNG.sample(list(PALETTE), piece_count)
    green_parity = int(RNG.randrange(2))
    H, W = labels.shape
    assembled = np.zeros((H, W), dtype=int)
    mains = np.zeros((H, W), dtype=int)
    seam = np.zeros((H, W), dtype=bool)

    for lab in range(piece_count):
        mask = labels == lab
        if int(mask.sum()) < 3:
            return None
        if not _is_connected_mask(mask):
            return None
        mains[mask] = int(colors[lab])

    for r in range(H):
        for c in range(W):
            lab = int(labels[r, c])
            is_seam = any(
                0 <= nr < H and 0 <= nc < W and int(labels[nr, nc]) != lab
                for nr, nc in _neighbors4(r, c)
            )
            seam[r, c] = bool(is_seam)
            if is_seam and ((r + c) & 1) == green_parity:
                assembled[r, c] = GREEN
            else:
                assembled[r, c] = int(colors[lab])

    for r in range(H):
        for c in range(W):
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W and int(labels[nr, nc]) != int(labels[r, c]):
                    a = int(assembled[r, c] == GREEN)
                    b = int(assembled[nr, nc] == GREEN)
                    if a + b != 1:
                        return None

    if not any(_piece_bbox_area(labels == lab) > int((labels == lab).sum()) for lab in range(piece_count)):
        return None

    _symmetry_kind(assembled == GREEN)
    piece_patches: list[np.ndarray] = []
    for lab in range(piece_count):
        coords = [(int(r), int(c)) for r, c in np.argwhere(labels == lab)]
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        for r, c in coords:
            patch[r - r0, c - c0] = int(assembled[r, c])
        piece_patches.append(patch)

    expected = _final_output_from_assembled(assembled, mains)
    return expected, assembled, piece_patches


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    areas = [int(np.count_nonzero(patch != BG)) for patch in patches]
    total_area = sum(areas)
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)

    h_lo = max(max_h + 2, int(np.sqrt(total_area * 4)))
    w_lo = max(max_w + 2, int(np.sqrt(total_area * 4)))
    H = RNG.randint(h_lo, h_lo + 6)
    W = RNG.randint(w_lo, w_lo + 6)

    order = list(range(len(patches)))
    RNG.shuffle(order)
    order.sort(key=lambda i: -areas[i])

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blocked = np.zeros((H, W), dtype=bool)

    for idx in order:
        patch = patches[idx]
        mask = patch != BG
        ph, pw = patch.shape
        if ph > H or pw > W:
            return None
        placed = False
        for _ in range(250):
            r0 = RNG.randint(0, H - ph)
            c0 = RNG.randint(0, W - pw)
            if bool((blocked[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                continue
            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            sub[mask] = patch[mask]
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
            blocked = _dilate8(occupied)
            placed = True
            break
        if not placed:
            return None

    return grid


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    branch = RNG.choice(("h", "v", "both"))
    if branch == "h":
        labels = _generate_horizontal_branch()
    elif branch == "v":
        labels = _generate_vertical_branch()
    else:
        labels = _generate_both_branch()
    if labels is None:
        return None

    built = _labels_to_case(labels)
    if built is None:
        return None
    expected, _assembled, patches = built

    inp = _scatter_patches(patches)
    if inp is None:
        return None

    found_pieces = _extract_pieces(inp)
    if len(_solve_pieces(found_pieces, solution_limit=2)) != 1:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC instance")
