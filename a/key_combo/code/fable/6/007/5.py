from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg, paste, random_free_location


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = np.array(sorted((int(r), int(c)) for r, c in coords), dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    rel = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), rel


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(1, min(area, 30) + 1):
        if area % h != 0:
            continue
        w = area // h
        if w <= 30:
            dims.append((h, w))
    dims.sort(key=lambda hw: (abs(hw[0] - hw[1]), hw[0], hw[1]))
    return dims


def _extract_state(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    pieces: list[dict] = []
    multicolor_count = 0

    for comp_id, comp in enumerate(comps):
        coords = comp["coords"]
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        counts = Counter(vals)
        colors = sorted(counts)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        rel_all = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))
        stain_rel: tuple[tuple[int, int], ...] = ()

        if len(colors) == 1:
            piece_color = int(colors[0])
        else:
            assert len(colors) == 2, "expected exactly one stained component with two colors"
            common = counts.most_common()
            assert common[0][1] > common[1][1], "stain must be the minority color inside its piece"
            multicolor_count += 1
            piece_color = int(common[0][0])
            stain_color = int(common[1][0])
            stain_rel = tuple(
                sorted(
                    (int(r) - r0, int(c) - c0)
                    for r, c in coords
                    if int(g[int(r), int(c)]) == stain_color
                )
            )
            assert stain_rel, "stained piece must contain at least one stain cell"

        pieces.append(
            {
                "id": int(comp_id),
                "color": int(piece_color),
                "coords": rel_all,
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "area": int(len(rel_all)),
                "stain_coords": stain_rel,
                "is_stained": bool(stain_rel),
            }
        )

    assert pieces, "expected at least one piece"
    assert multicolor_count == 1, "expected exactly one stained multicolor component"
    return pieces, sum(int(piece["area"]) for piece in pieces)


def _placement_data(piece: dict, H: int, W: int) -> list[dict]:
    data = []
    for r0 in range(H - int(piece["h"]) + 1):
        for c0 in range(W - int(piece["w"]) + 1):
            cells = tuple((r0 + dr, c0 + dc) for dr, dc in piece["coords"])
            data.append({"cells": cells, "cell_set": frozenset(cells)})
    return data


def _collect_solutions(
    pieces: list[dict],
    area: int,
    *,
    enforce_symmetry: bool,
    enforce_no_touch: bool,
    limit: int,
) -> list[np.ndarray]:
    solutions: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    def remember(board: np.ndarray) -> None:
        key = (tuple(map(int, board.shape)), board.tobytes())
        if key not in solutions:
            solutions[key] = board.copy()

    for H, W in _factor_pairs(area):
        if len(solutions) >= limit:
            break
        if any(int(piece["h"]) > H or int(piece["w"]) > W for piece in pieces):
            continue

        placements = [_placement_data(piece, H, W) for piece in pieces]
        board = np.full((H, W), -1, dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        remaining = tuple(range(len(pieces)))

        def can_place(pi: int, placement: dict) -> bool:
            piece = pieces[pi]
            col = int(piece["color"])
            cell_set = placement["cell_set"]
            for r, c in placement["cells"]:
                if occupied[r, c]:
                    return False
                if board[r, c] not in (-1, col):
                    return False
                if enforce_symmetry:
                    mc = W - 1 - c
                    if board[r, mc] not in (-1, col):
                        return False
            if enforce_no_touch:
                for r, c in placement["cells"]:
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cell_set:
                            if occupied[nr, nc] and board[nr, nc] == col:
                                return False
            return True

        def place(pi: int, placement: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
            piece = pieces[pi]
            col = int(piece["color"])
            changed: list[tuple[int, int]] = []
            used: list[tuple[int, int]] = []
            for r, c in placement["cells"]:
                if board[r, c] == -1:
                    board[r, c] = col
                    changed.append((r, c))
                occupied[r, c] = True
                used.append((r, c))
                if enforce_symmetry:
                    mc = W - 1 - c
                    if board[r, mc] == -1:
                        board[r, mc] = col
                        changed.append((r, mc))
            return changed, used

        def unplace(changed: list[tuple[int, int]], used: list[tuple[int, int]]) -> None:
            for r, c in used:
                occupied[r, c] = False
            for r, c in changed:
                board[r, c] = -1

        def search(rest: tuple[int, ...]) -> None:
            if len(solutions) >= limit:
                return
            if not rest:
                if not bool(occupied.all()):
                    return
                if enforce_symmetry and not np.array_equal(board, board[:, ::-1]):
                    return
                remember(board.astype(int))
                return

            best_pi = None
            best_places: list[dict] | None = None
            best_key = None
            for pi in rest:
                places = []
                for placement in placements[pi]:
                    if can_place(pi, placement):
                        places.append(placement)
                    if best_places is not None and len(places) > len(best_places):
                        break
                if not places:
                    return
                key = (len(places), -int(pieces[pi]["area"]), int(pi))
                if best_key is None or key < best_key:
                    best_key = key
                    best_pi = pi
                    best_places = places
                    if len(best_places) == 1:
                        break

            assert best_pi is not None and best_places is not None
            new_rest = tuple(pi for pi in rest if pi != best_pi)
            for placement in best_places:
                changed, used = place(best_pi, placement)
                search(new_rest)
                unplace(changed, used)
                if len(solutions) >= limit:
                    return

        search(remaining)

    return list(solutions.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments into a solid left-right symmetric rectangle and restore the stained fragment from its mirror mate.

    Input structure:
    - The background is the dominant color, and each non-background 4-connected component is one scattered puzzle fragment.
    - Exactly one fragment is multicolored because a unique stain color covers a minority subset of that fragment's cells.
    - The stain component still belongs to the covered fragment for area and placement purposes; it is not an output piece.
    - All complete fragments can tile one solid rectangle without overlap.

    Rule:
    - Treat every non-background component as one piece, but reinterpret the multicolored one as a single monochrome fragment whose full footprint includes the stain cells.
    - Determine the rectangle dimensions from the total non-background area and assemble the pieces without rotation so the rectangle is perfectly left-right mirror symmetric.
    - Enforce that different pieces of the same color never 4-touch in the assembly; together with symmetry this makes the packing unique.
    - Replace the stained cells by the fragment's true color, which is forced by the color at the mirror-matched output positions.

    Key insight:
    - The stain does not remove area, only trustworthy color information, so the hidden fragment can be placed by its full occupied footprint and then recolored by the global mirror-symmetric assembly.
    """
    pieces, area = _extract_state(grid)
    solutions = _collect_solutions(
        pieces,
        area,
        enforce_symmetry=True,
        enforce_no_touch=True,
        limit=2,
    )
    assert len(solutions) == 1, "expected a unique symmetric no-touch assembly"
    return solutions[0]


def _grow_left_partition(H: int, W: int, n_center: int, n_pair: int) -> tuple[np.ndarray, set[int]] | None:
    mid = W // 2
    left_w = mid + 1
    labels = np.full((H, left_w), -1, dtype=int)
    center_ids: set[int] = set()
    taken: set[tuple[int, int]] = set()
    next_label = 0

    center_rows = sorted(random.sample(range(H), n_center))
    for row in center_rows:
        labels[row, mid] = next_label
        center_ids.add(next_label)
        taken.add((row, mid))
        next_label += 1

    pair_cells = [(r, c) for r in range(H) for c in range(mid)]
    random.shuffle(pair_cells)
    for r, c in pair_cells:
        if len(taken) >= n_center + n_pair:
            break
        if (r, c) in taken:
            continue
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in taken):
            continue
        labels[r, c] = next_label
        taken.add((r, c))
        next_label += 1

    if next_label != n_center + n_pair:
        return None

    while np.any(labels == -1):
        frontier = []
        for r in range(H):
            for c in range(left_w):
                lab = int(labels[r, c])
                if lab < 0:
                    continue
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < H and 0 <= nc < left_w):
                        continue
                    if labels[nr, nc] != -1:
                        continue
                    if nc == mid and lab not in center_ids:
                        continue
                    frontier.append((lab, nr, nc))
        if not frontier:
            return None
        lab, nr, nc = random.choice(frontier)
        labels[nr, nc] = int(lab)

    areas = Counter(int(v) for v in labels.ravel())
    if min(areas.values()) < 2:
        return None
    if len([lab for lab in areas if lab not in center_ids]) < 2:
        return None
    return labels, center_ids


def _build_piece_layout(H: int, W: int) -> tuple[list[dict], np.ndarray] | None:
    n_center = 1 if H < 6 or random.random() < 0.55 else 2
    n_pair = 2 if random.random() < 0.45 else 3
    grown = _grow_left_partition(H, W, n_center, n_pair)
    if grown is None:
        return None
    labels, center_ids = grown
    owner = np.full((H, W), -1, dtype=int)
    pieces: list[dict] = []
    mid = W // 2

    for lab in sorted(int(v) for v in np.unique(labels)):
        left_cells = {(int(r), int(c)) for r, c in np.argwhere(labels == lab)}
        if lab in center_ids:
            full = set(left_cells)
            full |= {(r, W - 1 - c) for r, c in left_cells}
            piece_id = len(pieces)
            for r, c in full:
                owner[r, c] = piece_id
            pieces.append(
                {
                    "id": piece_id,
                    "group": int(lab),
                    "side": "center",
                    "coords": tuple(sorted(full)),
                }
            )
        else:
            left_id = len(pieces)
            right_id = left_id + 1
            right_cells = {(r, W - 1 - c) for r, c in left_cells}
            for r, c in left_cells:
                owner[r, c] = left_id
            for r, c in right_cells:
                owner[r, c] = right_id
            pieces.append(
                {
                    "id": left_id,
                    "group": int(lab),
                    "side": "left",
                    "coords": tuple(sorted(left_cells)),
                }
            )
            pieces.append(
                {
                    "id": right_id,
                    "group": int(lab),
                    "side": "right",
                    "coords": tuple(sorted(right_cells)),
                }
            )

    if np.any(owner < 0):
        return None
    pair_candidates = [piece for piece in pieces if piece["side"] == "left" and len(piece["coords"]) >= 3]
    if not pair_candidates:
        return None

    by_group = defaultdict(list)
    for piece in pieces:
        by_group[int(piece["group"])].append(piece)
    for group_pieces in by_group.values():
        if len(group_pieces) == 2:
            left_piece = min(group_pieces, key=lambda p: p["id"])
            right_piece = max(group_pieces, key=lambda p: p["id"])
            _, _, sig_left = _normalize(left_piece["coords"])
            _, _, sig_right = _normalize(right_piece["coords"])
            if sig_left == sig_right:
                return None

    return pieces, owner


def _group_adjacency(pieces: list[dict], owner: np.ndarray) -> dict[int, set[int]]:
    H, W = owner.shape
    piece_to_group = {int(piece["id"]): int(piece["group"]) for piece in pieces}
    graph: dict[int, set[int]] = defaultdict(set)
    for r in range(H):
        for c in range(W):
            a = int(owner[r, c])
            ga = piece_to_group[a]
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr >= H or nc >= W:
                    continue
                b = int(owner[nr, nc])
                gb = piece_to_group[b]
                if ga != gb:
                    graph[ga].add(gb)
                    graph[gb].add(ga)
    for piece in pieces:
        graph.setdefault(int(piece["group"]), set())
    return graph


def _assign_group_colors(groups: list[int], graph: dict[int, set[int]], n_colors: int) -> dict[int, int] | None:
    order = sorted(groups, key=lambda g: (-len(graph[g]), int(g)))
    color_of: dict[int, int] = {}
    use_count = Counter()

    def backtrack(i: int) -> bool:
        if i == len(order):
            return len(set(color_of.values())) >= min(3, len(order))
        group = order[i]
        blocked = {color_of[nbr] for nbr in graph[group] if nbr in color_of}
        options = list(range(n_colors))
        random.shuffle(options)
        options.sort(key=lambda col: (use_count[col], col))
        for col in options:
            if col in blocked:
                continue
            color_of[group] = col
            use_count[col] += 1
            if backtrack(i + 1):
                return True
            use_count[col] -= 1
            del color_of[group]
        return False

    return color_of if backtrack(0) else None


def _random_connected_subset(cells: tuple[tuple[int, int], ...], size: int) -> tuple[tuple[int, int], ...] | None:
    cell_set = set(cells)
    start = random.choice(list(cells))
    subset = {start}
    frontier = set()
    for nr, nc in ((start[0] - 1, start[1]), (start[0] + 1, start[1]), (start[0], start[1] - 1), (start[0], start[1] + 1)):
        if (nr, nc) in cell_set:
            frontier.add((nr, nc))
    while len(subset) < size and frontier:
        nxt = random.choice(list(frontier))
        frontier.remove(nxt)
        subset.add(nxt)
        for nr, nc in ((nxt[0] - 1, nxt[1]), (nxt[0] + 1, nxt[1]), (nxt[0], nxt[1] - 1), (nxt[0], nxt[1] + 1)):
            if (nr, nc) in cell_set and (nr, nc) not in subset:
                frontier.add((nr, nc))
    if len(subset) != size:
        return None
    return tuple(sorted(subset))


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(5, 7)
    W = random.choice([5, 7, 9])
    if H * W > 49:
        W = 7

    layout = _build_piece_layout(H, W)
    if layout is None:
        return None
    pieces, owner = layout
    groups = sorted({int(piece["group"]) for piece in pieces})
    color_count = 3 if len(groups) <= 4 else 4
    assignment = _assign_group_colors(groups, _group_adjacency(pieces, owner), color_count)
    if assignment is None:
        return None

    bg = random.choice(list(range(10)))
    available = [c for c in range(10) if c != bg]
    if len(available) < color_count + 1:
        return None
    actual_colors = random.sample(available, color_count + 1)
    stain_color = int(actual_colors[-1])
    real_color_of_group = {group: int(actual_colors[idx]) for group, idx in assignment.items()}

    out = np.full((H, W), bg, dtype=int)
    for piece in pieces:
        col = int(real_color_of_group[int(piece["group"])])
        for r, c in piece["coords"]:
            out[int(r), int(c)] = col
    assert np.all(out != bg)
    assert np.array_equal(out, out[:, ::-1])

    sigs = set()
    for piece in pieces:
        h, w, rel = _normalize(piece["coords"])
        sig = (int(real_color_of_group[int(piece["group"])]), h, w, rel)
        if sig in sigs:
            return None
        sigs.add(sig)

    group_to_pieces = defaultdict(list)
    for piece in pieces:
        group_to_pieces[int(piece["group"])].append(piece)
    stain_options = []
    for group_pieces in group_to_pieces.values():
        if len(group_pieces) != 2:
            continue
        left_piece = [p for p in group_pieces if p["side"] == "left"]
        right_piece = [p for p in group_pieces if p["side"] == "right"]
        if not left_piece or not right_piece:
            continue
        if len(left_piece[0]["coords"]) >= 3:
            stain_options.append((left_piece[0], right_piece[0]))
    if not stain_options:
        return None

    stain_piece, mate_piece = random.choice(stain_options)
    max_stain = min(3, len(stain_piece["coords"]) - 2, (len(stain_piece["coords"]) - 1) // 2)
    if max_stain < 1:
        return None
    stain_size = random.randint(1, max_stain)
    stain_cells = _random_connected_subset(stain_piece["coords"], stain_size)
    if stain_cells is None:
        return None

    mate_set = set(mate_piece["coords"])
    piece_id_to_color = {int(piece["id"]): int(real_color_of_group[int(piece["group"])]) for piece in pieces}
    for r, c in stain_cells:
        mr, mc = int(r), int(W - 1 - c)
        if (mr, mc) not in mate_set:
            return None
        if int(out[mr, mc]) != piece_id_to_color[int(stain_piece["id"])]:
            return None

    rows = random.randint(max(H + 5, 12), min(22, 30))
    cols = random.randint(max(W + 6, 13), min(23, 30))
    canvas = np.full((rows, cols), bg, dtype=int)
    patch_specs = []
    for piece in pieces:
        pts = np.array(piece["coords"], dtype=int)
        r0, c0, r1, c1 = bbox_coords(pts)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        col = int(piece_id_to_color[int(piece["id"])])
        for r, c in piece["coords"]:
            patch[int(r) - r0, int(c) - c0] = col
        if int(piece["id"]) == int(stain_piece["id"]):
            for r, c in stain_cells:
                patch[int(r) - r0, int(c) - c0] = stain_color
        patch_specs.append(
            {
                "area": int(np.count_nonzero(patch != bg)),
                "patch": patch,
            }
        )

    random.shuffle(patch_specs)
    patch_specs.sort(key=lambda item: -int(item["area"]))
    try:
        for spec in patch_specs:
            patch = spec["patch"]
            rr, cc = random_free_location(canvas, patch, transparent=bg, border=1, padding=1, padding_connectivity=8)
            paste(canvas, patch, rr, cc, transparent=bg)
    except ValueError:
        return None

    if int(np.count_nonzero(canvas != bg)) != H * W:
        return None
    if len(Counter(int(v) for v in canvas[canvas != bg].ravel())) < 3:
        return None

    try:
        full_pieces, full_area = _extract_state(canvas)
        full_solutions = _collect_solutions(
            full_pieces,
            full_area,
            enforce_symmetry=True,
            enforce_no_touch=True,
            limit=2,
        )
    except AssertionError:
        return None
    if len(full_solutions) != 1 or not np.array_equal(full_solutions[0], out):
        return None

    relaxed_solutions = _collect_solutions(
        full_pieces,
        full_area,
        enforce_symmetry=True,
        enforce_no_touch=False,
        limit=2,
    )
    if len(relaxed_solutions) < 2:
        return None

    return canvas.astype(int), out.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _make_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid stained symmetric-jigsaw input")
