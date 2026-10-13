import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_RECT_SIDE = 6
MAX_RECT_SIDE = 10
MIN_CHILD_SIDE = 3
MIN_PIECES = 2
MAX_PIECES = 4
CANVAS_MIN = 18
CANVAS_MAX = 28
GEN_ATTEMPTS = 300
SCATTER_ATTEMPTS = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(cells)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _foreign_neighbors(piece_map: np.ndarray, r: int, c: int) -> set[int]:
    pid = int(piece_map[r, c])
    h, w = piece_map.shape
    foreign: set[int] = set()
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            other = int(piece_map[nr, nc])
            if other != pid:
                foreign.add(other)
    return foreign


def _graph_connected(n_nodes: int, edges: list[tuple[int, int]]) -> bool:
    if n_nodes == 0:
        return False
    graph: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == n_nodes


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces: list[dict] = []
    for comp in _components(grid != BLACK):
        r0, c0, _, _ = _bbox(comp)
        rel_cells = tuple(sorted((r - r0, c - c0) for r, c in comp))
        markers: dict[int, tuple[int, int]] = {}
        for r, c in comp:
            color = int(grid[r, c])
            if color not in (BLACK, GREY):
                rel = (r - r0, c - c0)
                assert color not in markers, "each marker color appears at most once per piece"
                markers[color] = rel
        pieces.append({"cells": rel_cells, "markers": markers})
    assert pieces, "expected at least one fragment"
    return pieces


def _marker_pairs(pieces: list[dict]) -> dict[int, tuple[tuple[int, tuple[int, int]], tuple[int, tuple[int, int]]]]:
    occ: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for pid, piece in enumerate(pieces):
        for color, pos in piece["markers"].items():
            occ[int(color)].append((pid, pos))
    pairs: dict[int, tuple[tuple[int, tuple[int, int]], tuple[int, tuple[int, int]]]] = {}
    for color, hits in occ.items():
        assert len(hits) == 2, "each marker color must appear exactly twice"
        assert hits[0][0] != hits[1][0], "paired markers must lie on different pieces"
        pairs[int(color)] = (hits[0], hits[1])
    return pairs


def _solve_rectangle_shapes(grid: np.ndarray, stop_after: int = 2) -> set[tuple[int, int]]:
    pieces = _extract_pieces(grid)
    n_pieces = len(pieces)
    marker_pairs = _marker_pairs(pieces)
    graph_edges = [(a[0], b[0]) for a, b in marker_pairs.values()]
    assert _graph_connected(n_pieces, graph_edges), "marker pair graph must connect all pieces"

    by_piece: dict[int, list[tuple[int, int, tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    for color, ((p0, pos0), (p1, pos1)) in marker_pairs.items():
        by_piece[p0].append((color, p1, pos0, pos1))
        by_piece[p1].append((color, p0, pos1, pos0))

    total_area = sum(len(piece["cells"]) for piece in pieces)
    anchor = max(
        range(n_pieces),
        key=lambda pid: (len(by_piece[pid]), len(pieces[pid]["cells"]), -pid),
    )

    placements: dict[int, tuple[int, int]] = {anchor: (0, 0)}
    occupied = set(pieces[anchor]["cells"])
    anchor_rows = [r for r, _ in pieces[anchor]["cells"]]
    anchor_cols = [c for _, c in pieces[anchor]["cells"]]
    bbox = (min(anchor_rows), min(anchor_cols), max(anchor_rows), max(anchor_cols))
    solutions: set[tuple[int, int]] = set()

    def search(curr_bbox: tuple[int, int, int, int]) -> None:
        if len(solutions) >= stop_after:
            return
        if len(placements) == n_pieces:
            r0, c0, r1, c1 = curr_bbox
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height * width == total_area:
                solutions.add((height, width))
            return

        candidates = [
            pid
            for pid in range(n_pieces)
            if pid not in placements and any(other in placements for _, other, _, _ in by_piece[pid])
        ]
        if not candidates:
            return

        pid = max(
            candidates,
            key=lambda i: (
                sum(1 for _, other, _, _ in by_piece[i] if other in placements),
                len(by_piece[i]),
                len(pieces[i]["cells"]),
                -i,
            ),
        )
        placed_constraints = [
            (other, self_pos, other_pos)
            for _, other, self_pos, other_pos in by_piece[pid]
            if other in placements
        ]
        other, self_pos, other_pos = placed_constraints[0]
        base_r, base_c = placements[other]
        trial_translations: list[tuple[int, int]] = []
        for dr, dc in DIRS4:
            tr = base_r + other_pos[0] + dr - self_pos[0]
            tc = base_c + other_pos[1] + dc - self_pos[1]
            if (tr, tc) not in trial_translations:
                trial_translations.append((tr, tc))

        for tr, tc in trial_translations:
            ok = True
            for other2, self_pos2, other_pos2 in placed_constraints:
                orow, ocol = placements[other2]
                mr = abs((tr + self_pos2[0]) - (orow + other_pos2[0]))
                mc = abs((tc + self_pos2[1]) - (ocol + other_pos2[1]))
                if mr + mc != 1:
                    ok = False
                    break
            if not ok:
                continue

            new_cells = [(r + tr, c + tc) for r, c in pieces[pid]["cells"]]
            if any(cell in occupied for cell in new_cells):
                continue

            nr0 = min(curr_bbox[0], min(r for r, _ in new_cells))
            nc0 = min(curr_bbox[1], min(c for _, c in new_cells))
            nr1 = max(curr_bbox[2], max(r for r, _ in new_cells))
            nc1 = max(curr_bbox[3], max(c for _, c in new_cells))
            if (nr1 - nr0 + 1) * (nc1 - nc0 + 1) > total_area:
                continue

            placements[pid] = (tr, tc)
            occupied.update(new_cells)
            search((nr0, nc0, nr1, nc1))
            for cell in new_cells:
                occupied.remove(cell)
            placements.pop(pid)

    search(bbox)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered marked fragments into the solid grey rectangle they originally tiled.

    Input structure:
    - The background is black.
    - Several disconnected fragments are scattered across the grid; each fragment is a 4-connected region of grey cells with a few non-grey, non-black marker cells embedded on its boundary.
    - Every marker color appears exactly twice in the whole grid, once on each of two different fragments.

    Rule:
    - Treat each non-black connected component as one rigid fragment that may only be translated.
    - Fragments whose boundary markers share a color must be positioned so that those two same-colored cells become 4-neighbors, while fragments do not overlap.
    - Among all placements satisfying the marker pairings, keep the one where the union of fragment cells fills one rectangle exactly, then replace all marker cells by grey.

    Key insight:
    - The colored edge cells are not part of the final picture; they are pairwise connection hints that, together with the fragment outlines, determine the rectangle's dimensions and assembly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    shapes = _solve_rectangle_shapes(g, stop_after=2)
    assert len(shapes) == 1, "expected a unique assembled rectangle shape"
    height, width = next(iter(shapes))
    return np.full((height, width), GREY, dtype=int)


def _partition_rectangles(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
    width = rng.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
    n_pieces = rng.randint(MIN_PIECES, MAX_PIECES)
    rects: list[tuple[int, int, int, int, int]] = [(0, 0, 0, height - 1, width - 1)]
    next_id = 1

    while len(rects) < n_pieces:
        options: list[tuple[int, str]] = []
        for idx, (_, r0, c0, r1, c1) in enumerate(rects):
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if w >= 2 * MIN_CHILD_SIDE:
                options.append((idx, "v"))
            if h >= 2 * MIN_CHILD_SIDE:
                options.append((idx, "h"))
        if not options:
            return None

        idx, orient = rng.choice(options)
        pid, r0, c0, r1, c1 = rects.pop(idx)
        if orient == "v":
            split = rng.randint(c0 + MIN_CHILD_SIDE, c1 - MIN_CHILD_SIDE + 1)
            rects.append((pid, r0, c0, r1, split - 1))
            rects.append((next_id, r0, split, r1, c1))
        else:
            split = rng.randint(r0 + MIN_CHILD_SIDE, r1 - MIN_CHILD_SIDE + 1)
            rects.append((pid, r0, c0, split - 1, c1))
            rects.append((next_id, split, c0, r1, c1))
        next_id += 1

    piece_map = np.empty((height, width), dtype=int)
    for pid, r0, c0, r1, c1 in rects:
        piece_map[r0 : r1 + 1, c0 : c1 + 1] = pid
    return piece_map


def _collect_adjacencies(piece_map: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]:
    h, w = piece_map.shape
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            pid = int(piece_map[r, c])
            if c + 1 < w:
                other = int(piece_map[r, c + 1])
                if other != pid:
                    if pid < other:
                        adj[(pid, other)].append(((r, c), (r, c + 1)))
                    else:
                        adj[(other, pid)].append(((r, c + 1), (r, c)))
            if r + 1 < h:
                other = int(piece_map[r + 1, c])
                if other != pid:
                    if pid < other:
                        adj[(pid, other)].append(((r, c), (r + 1, c)))
                    else:
                        adj[(other, pid)].append(((r + 1, c), (r, c)))
    return adj


def _straight_seams(piece_map: np.ndarray) -> list[dict]:
    seams: list[dict] = []
    for (a, b), pairs in _collect_adjacencies(piece_map).items():
        (r0a, c0a), (r0b, c0b) = pairs[0]
        dr = r0b - r0a
        dc = c0b - c0a
        if dr == 0:
            if dc == 1:
                left, right = a, b
                rows = sorted({ra for (ra, _), _ in pairs})
                left_col = c0a
            else:
                left, right = b, a
                rows = sorted({rb for _, (rb, _) in pairs})
                left_col = c0b
            seams.append(
                {
                    "orientation": "v",
                    "left": left,
                    "right": right,
                    "rows": rows,
                    "left_col": left_col,
                    "right_col": left_col + 1,
                }
            )
        else:
            if dr == 1:
                top, bottom = a, b
                cols = sorted({ca for (_, ca), _ in pairs})
                top_row = r0a
            else:
                top, bottom = b, a
                cols = sorted({cb for _, (_, cb) in pairs})
                top_row = r0b
            seams.append(
                {
                    "orientation": "h",
                    "top": top,
                    "bottom": bottom,
                    "cols": cols,
                    "top_row": top_row,
                    "bottom_row": top_row + 1,
                }
            )
    return seams


def _all_pieces_connected(piece_map: np.ndarray) -> bool:
    for pid in np.unique(piece_map):
        if len(_components(piece_map == int(pid))) != 1:
            return False
    return True


def _has_irregular_piece(piece_map: np.ndarray) -> bool:
    for pid in np.unique(piece_map):
        coords = [(int(r), int(c)) for r, c in np.argwhere(piece_map == int(pid))]
        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) < (r1 - r0 + 1) * (c1 - c0 + 1):
            return True
    return False


def _perturb_partition(piece_map: np.ndarray, rng: random.Random) -> bool:
    seams = _straight_seams(piece_map)
    rng.shuffle(seams)
    changed = False
    for seam in seams:
        if seam["orientation"] == "v":
            rows = seam["rows"]
            if len(rows) < 3:
                continue
            start = rng.randint(rows[1], rows[-2])
            end = rng.randint(start, rows[-2])
            if rng.random() < 0.5:
                piece_map[start : end + 1, seam["right_col"]] = seam["left"]
            else:
                piece_map[start : end + 1, seam["left_col"]] = seam["right"]
            changed = True
        else:
            cols = seam["cols"]
            if len(cols) < 3:
                continue
            start = rng.randint(cols[1], cols[-2])
            end = rng.randint(start, cols[-2])
            if rng.random() < 0.5:
                piece_map[seam["bottom_row"], start : end + 1] = seam["top"]
            else:
                piece_map[seam["top_row"], start : end + 1] = seam["bottom"]
            changed = True
    return changed


def _choose_marker_cells(
    piece_map: np.ndarray,
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]],
    rng: random.Random,
) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    marker_cells: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    used: set[tuple[int, int]] = set()
    for color, pair_key in zip(colors, sorted(adj)):
        a, b = pair_key
        preferred: list[tuple[tuple[int, int], tuple[int, int]]] = []
        fallback: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for cell_a, cell_b in adj[pair_key]:
            if cell_a in used or cell_b in used:
                continue
            fallback.append((cell_a, cell_b))
            if _foreign_neighbors(piece_map, cell_a[0], cell_a[1]) == {b} and _foreign_neighbors(
                piece_map, cell_b[0], cell_b[1]
            ) == {a}:
                preferred.append((cell_a, cell_b))
        choices = preferred or fallback
        if not choices:
            raise ValueError("failed to find unused marker locations")
        chosen = rng.choice(choices)
        marker_cells[color] = chosen
        used.add(chosen[0])
        used.add(chosen[1])
    return marker_cells


def _build_assembled_grid(
    piece_map: np.ndarray, marker_cells: dict[int, tuple[tuple[int, int], tuple[int, int]]]
) -> np.ndarray:
    assembled = np.full(piece_map.shape, GREY, dtype=int)
    for color, (cell_a, cell_b) in marker_cells.items():
        assembled[cell_a] = int(color)
        assembled[cell_b] = int(color)
    return assembled


def _build_patches(piece_map: np.ndarray, assembled: np.ndarray) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for pid in sorted(int(x) for x in np.unique(piece_map)):
        coords = [(int(r), int(c)) for r, c in np.argwhere(piece_map == pid)]
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
        for r, c in coords:
            patch[r - r0, c - c0] = assembled[r, c]
        patches.append(patch)
    return patches


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _scatter_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    canvas_h = rng.randint(max(CANVAS_MIN, max_h + 2), min(30, CANVAS_MAX))
    canvas_w = rng.randint(max(CANVAS_MIN, max_w + 2), min(30, CANVAS_MAX))
    canvas = np.full((canvas_h, canvas_w), BLACK, dtype=int)

    order = list(range(len(patches)))
    rng.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(SCATTER_ATTEMPTS):
            r0 = rng.randint(0, canvas_h - ph)
            c0 = rng.randint(0, canvas_w - pw)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_too_close(box, other) for other in boxes):
                continue
            placements[idx] = (r0, c0)
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    for idx, (r0, c0) in placements.items():
        patch = patches[idx]
        mask = patch != BLACK
        canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
    return canvas


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_map = _partition_rectangles(rng)
    if piece_map is None:
        return None

    changed = _perturb_partition(piece_map, rng)
    if not changed or not _all_pieces_connected(piece_map) or not _has_irregular_piece(piece_map):
        return None

    adj = _collect_adjacencies(piece_map)
    n_pieces = int(np.unique(piece_map).size)
    graph_edges = list(adj.keys())
    if not graph_edges or len(graph_edges) > len(MARKER_COLORS) or not _graph_connected(n_pieces, graph_edges):
        return None

    try:
        marker_cells = _choose_marker_cells(piece_map, adj, rng)
    except ValueError:
        return None

    assembled = _build_assembled_grid(piece_map, marker_cells)
    patches = _build_patches(piece_map, assembled)
    scattered = _scatter_patches(patches, rng)
    if scattered is None:
        return None

    shapes = _solve_rectangle_shapes(scattered, stop_after=2)
    target_shape = {assembled.shape}
    if shapes != target_shape:
        return None
    return scattered


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jigsaw-assembly input")
