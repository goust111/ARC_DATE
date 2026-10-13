import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid, bbox):
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _is_solid_rectangle(grid, color):
    coords = list(map(tuple, np.argwhere(grid == color)))
    if not coords:
        return None
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape[0] * patch.shape[1] != len(coords):
        return None
    if not np.all(patch == color):
        return None
    return r0, c0, r1, c1


def _candidate_occluders(grid):
    h, w = grid.shape
    candidates = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        bbox = _is_solid_rectangle(grid, color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        top_ok = np.all(grid[r0 - 1, c0 : c1 + 1] != BG)
        bottom_ok = np.all(grid[r1 + 1, c0 : c1 + 1] != BG)
        left_ok = np.all(grid[r0 : r1 + 1, c0 - 1] != BG)
        right_ok = np.all(grid[r0 : r1 + 1, c1 + 1] != BG)
        if top_ok and bottom_ok and left_ok and right_ok:
            candidates.append((color, bbox))
    return candidates


def _scene_mask(grid, occluder_bbox):
    h, w = grid.shape
    r0, c0, _r1, _c1 = occluder_bbox
    stack = [(r0, c0)]
    seen = np.zeros((h, w), dtype=bool)
    seen[r0, c0] = True
    while stack:
        rr, cc = stack.pop()
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _extract_pieces(grid, scene_mask):
    pieces_mask = (grid != BG) & (~scene_mask)
    pieces = []
    for coords in _components(pieces_mask):
        color = int(grid[coords[0]])
        assert all(int(grid[r, c]) == color for r, c in coords)
        bbox = _bbox_from_coords(coords)
        patch = _crop_bbox(grid, bbox)
        assert np.all(patch == color)
        pieces.append({"color": color, "shape": patch.shape, "patch": patch})
    assert pieces, "expected scattered pieces outside the scene"
    return pieces


def _boundary_arrays(grid, bbox):
    r0, c0, r1, c1 = bbox
    top = grid[r0 - 1, c0 : c1 + 1].astype(int)
    bottom = grid[r1 + 1, c0 : c1 + 1].astype(int)
    left = grid[r0 : r1 + 1, c0 - 1].astype(int)
    right = grid[r0 : r1 + 1, c1 + 1].astype(int)
    return top, bottom, left, right


def _type_data(pieces):
    counts = Counter()
    for piece in pieces:
        ph, pw = piece["shape"]
        counts[(int(piece["color"]), int(ph), int(pw))] += 1
    return dict(counts)


def _valid_placements(piece_type, window_shape, top, bottom, left, right):
    color, ph, pw = piece_type
    h, w = window_shape
    placements = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if r0 == 0 and not np.all(top[c0 : c0 + pw] == color):
                continue
            if r0 + ph == h and not np.all(bottom[c0 : c0 + pw] == color):
                continue
            if c0 == 0 and not np.all(left[r0 : r0 + ph] == color):
                continue
            if c0 + pw == w and not np.all(right[r0 : r0 + ph] == color):
                continue
            placements.append((r0, c0))
    return placements


def _solve_window(pieces, window_shape, top, bottom, left, right, max_solutions=2):
    h, w = window_shape
    type_counts = _type_data(pieces)
    placements = {
        piece_type: _valid_placements(piece_type, window_shape, top, bottom, left, right)
        for piece_type in sorted(type_counts)
    }
    for piece_type, count in type_counts.items():
        assert placements[piece_type], f"piece type {piece_type} has no valid placement"
        assert count * piece_type[1] * piece_type[2] <= h * w
    board = np.full((h, w), -1, dtype=int)
    solutions = []

    def can_place(piece_type, r0, c0):
        _color, ph, pw = piece_type
        return not np.any(board[r0 : r0 + ph, c0 : c0 + pw] != -1)

    def choose_cell():
        best = None
        for rr in range(h):
            for cc in range(w):
                if board[rr, cc] != -1:
                    continue
                moves = []
                for piece_type in sorted(type_counts):
                    if type_counts[piece_type] <= 0:
                        continue
                    ph, pw = piece_type[1], piece_type[2]
                    for r0, c0 in placements[piece_type]:
                        if not (r0 <= rr < r0 + ph and c0 <= cc < c0 + pw):
                            continue
                        if can_place(piece_type, r0, c0):
                            moves.append((piece_type, r0, c0))
                if best is None or len(moves) < len(best[2]):
                    best = (rr, cc, moves)
                if best is not None and len(best[2]) <= 1:
                    return best
        return best

    def search():
        if len(solutions) >= max_solutions:
            return
        choice = choose_cell()
        if choice is None:
            solutions.append(board.copy())
            return
        _rr, _cc, moves = choice
        if not moves:
            return
        moves.sort(key=lambda item: (-(item[0][1] * item[0][2]), item[1], item[2], item[0][0]))
        for piece_type, r0, c0 in moves:
            color, ph, pw = piece_type
            board[r0 : r0 + ph, c0 : c0 + pw] = color
            type_counts[piece_type] -= 1
            search()
            type_counts[piece_type] += 1
            board[r0 : r0 + ph, c0 : c0 + pw] = -1
            if len(solutions) >= max_solutions:
                return

    search()
    return solutions


def _solve(grid, max_solutions=2):
    g = np.asarray(grid, dtype=int)
    solved = []
    for _occluder_color, occluder_bbox in _candidate_occluders(g):
        scene = _scene_mask(g, occluder_bbox)
        pieces = _extract_pieces(g, scene)
        r0, c0, r1, c1 = occluder_bbox
        window_shape = (r1 - r0 + 1, c1 - c0 + 1)
        piece_area = sum(piece["shape"][0] * piece["shape"][1] for piece in pieces)
        if piece_area != window_shape[0] * window_shape[1]:
            continue
        top, bottom, left, right = _boundary_arrays(g, occluder_bbox)
        solutions = _solve_window(pieces, window_shape, top, bottom, left, right, max_solutions=max_solutions)
        if len(solutions) == 1:
            solved.extend(solutions)
            if len(solved) >= max_solutions:
                break
    assert len(solved) == 1, "expected one occluder candidate with one valid reconstruction"
    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular fragments to repair the uniquely colored window and return only its contents.

    Input structure:
    - A black background contains one connected rectangular scene made from large monochrome rectangles.
    - Inside that scene, one unique solid rectangular occluder color marks the hidden window.
    - Outside the scene, the missing window content appears as scattered monochrome rectangular pieces with no rotation needed.

    Rule:
    - Detect the occluder and read the required window-border colors from the visible scene cells immediately outside it.
    - Collect the scattered rectangular pieces and translate them, without rotation or overlap, until they exactly tile the window while matching those border continuations.
    - Output the reconstructed window crop only.

    Key insight:
    - Symmetry is irrelevant; the unique anchor is that every window-edge cell must continue the color of the scene cell directly beyond that edge.
    """
    solutions = _solve(grid, max_solutions=2)
    assert len(solutions) == 1, "expected a unique reconstruction"
    return solutions[0]


def _rectangles_adjacent(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ac1 + 1 == bc0 or bc1 + 1 == ac0:
        overlap = min(ar1, br1) - max(ar0, br0) + 1
        return overlap > 0
    if ar1 + 1 == br0 or br1 + 1 == ar0:
        overlap = min(ac1, bc1) - max(ac0, bc0) + 1
        return overlap > 0
    return False


def _partition_scene(height, width):
    rects = [(0, 0, height - 1, width - 1)]
    target = random.randint(5, 7)
    while len(rects) < target:
        options = []
        for idx, (r0, c0, r1, c1) in enumerate(rects):
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h >= 4:
                for cut in range(r0 + 2, r1 - 0):
                    if cut <= r1 - 1:
                        options.append((idx, "h", cut))
            if w >= 4:
                for cut in range(c0 + 2, c1 - 0):
                    if cut <= c1 - 1:
                        options.append((idx, "v", cut))
        if not options:
            break
        idx, axis, cut = random.choice(options)
        r0, c0, r1, c1 = rects.pop(idx)
        if axis == "h":
            rects.append((r0, c0, cut - 1, c1))
            rects.append((cut, c0, r1, c1))
        else:
            rects.append((r0, c0, r1, cut - 1))
            rects.append((r0, cut, r1, c1))
    rects.sort()
    return rects


def _color_scene(rects, height, width):
    scene = np.zeros((height, width), dtype=int)
    colors = random.sample(PALETTE, len(rects) + 1)
    assignment = {}
    for idx, rect in enumerate(rects):
        assignment[rect] = colors[idx]
        r0, c0, r1, c1 = rect
        scene[r0 : r1 + 1, c0 : c1 + 1] = colors[idx]
    for i, rect_a in enumerate(rects):
        for rect_b in rects[i + 1 :]:
            if _rectangles_adjacent(rect_a, rect_b):
                assert assignment[rect_a] != assignment[rect_b]
    return scene, colors[-1]


def _patch_pieces(patch):
    pieces = []
    for color in sorted(int(v) for v in np.unique(patch)):
        mask = patch == color
        for coords in _components(mask):
            bbox = _bbox_from_coords(coords)
            piece = _crop_bbox(patch, bbox)
            if piece.shape[0] * piece.shape[1] != len(coords):
                return None
            pr0, pc0, pr1, pc1 = bbox
            touches = {
                "top": pr0 == 0,
                "bottom": pr1 == patch.shape[0] - 1,
                "left": pc0 == 0,
                "right": pc1 == patch.shape[1] - 1,
            }
            pieces.append({"color": color, "shape": piece.shape, "patch": piece, "bbox": bbox, "touches": touches})
    return pieces


def _window_candidates(scene):
    sh, sw = scene.shape
    candidates = []
    for h in range(4, min(8, sh)):
        for w in range(4, min(8, sw)):
            area = h * w
            if area < 12 or area > 24:
                continue
            for r0 in range(1, sh - h):
                for c0 in range(1, sw - w):
                    r1 = r0 + h - 1
                    c1 = c0 + w - 1
                    if not np.all(scene[r0 - 1, c0 : c1 + 1] == scene[r0, c0 : c1 + 1]):
                        continue
                    if not np.all(scene[r1 + 1, c0 : c1 + 1] == scene[r1, c0 : c1 + 1]):
                        continue
                    if not np.all(scene[r0 : r1 + 1, c0 - 1] == scene[r0 : r1 + 1, c0]):
                        continue
                    if not np.all(scene[r0 : r1 + 1, c1 + 1] == scene[r0 : r1 + 1, c1]):
                        continue
                    patch = scene[r0 : r1 + 1, c0 : c1 + 1]
                    pieces = _patch_pieces(patch)
                    if pieces is None:
                        continue
                    if not (3 <= len(pieces) <= 7):
                        continue
                    if len({piece["color"] for piece in pieces}) < 3:
                        continue
                    if any(not any(piece["touches"].values()) for piece in pieces):
                        continue
                    candidates.append((r0, c0, r1, c1, patch.copy(), pieces))
    return candidates


def _scene_is_asymmetric(scene):
    return (
        not np.array_equal(scene, np.fliplr(scene))
        and not np.array_equal(scene, np.flipud(scene))
        and not np.array_equal(scene, np.rot90(scene, 2))
    )


def _can_place_rect(occupied, r0, c0, h, w):
    gh, gw = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(gh, r0 + h + 1)
    cc1 = min(gw, c0 + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _scatter_pieces(grid, occupied, pieces):
    gh, gw = grid.shape
    for piece in sorted(pieces, key=lambda item: -(item["shape"][0] * item["shape"][1])):
        ph, pw = piece["shape"]
        positions = [(r, c) for r in range(1, gh - ph) for c in range(1, gw - pw)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if _can_place_rect(occupied, r0, c0, ph, pw):
                grid[r0 : r0 + ph, c0 : c0 + pw] = piece["color"]
                occupied[r0 : r0 + ph, c0 : c0 + pw] = True
                placed = True
                break
        if not placed:
            return False
    return True


def _make_input():
    sh = random.randint(12, 17)
    sw = random.randint(12, 17)
    rects = _partition_scene(sh, sw)
    if len(rects) < 5:
        return None
    scene, occluder_color = _color_scene(rects, sh, sw)
    if not _scene_is_asymmetric(scene):
        return None
    candidates = _window_candidates(scene)
    if not candidates:
        return None
    wr0, wc0, wr1, wc1, patch, pieces = random.choice(candidates)

    gh = min(30, sh + random.randint(8, 11))
    gw = min(30, sw + random.randint(8, 11))
    if gh - sh < 2 or gw - sw < 2:
        return None
    sr = random.randint(1, gh - sh - 1)
    sc = random.randint(1, gw - sw - 1)

    grid = np.zeros((gh, gw), dtype=int)
    grid[sr : sr + sh, sc : sc + sw] = scene
    grid[sr + wr0 : sr + wr1 + 1, sc + wc0 : sc + wc1 + 1] = occluder_color

    occupied = np.zeros((gh, gw), dtype=bool)
    occupied[max(0, sr - 1) : min(gh, sr + sh + 1), max(0, sc - 1) : min(gw, sc + sw + 1)] = True
    if not _scatter_pieces(grid, occupied, pieces):
        return None

    solved = _solve(grid, max_solutions=2)
    if len(solved) != 1:
        return None
    if not np.array_equal(solved[0], patch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid window-fragment repair input")
