from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_cycle(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords += [(r, w - 1) for r in range(1, h)]
    coords += [(h - 1, c) for c in range(w - 2, -1, -1)]
    coords += [(r, 0) for r in range(h - 2, 0, -1)]
    return coords


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_key(coords: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return (r1 - r0 + 1, c1 - c0 + 1, norm)


def _coords_to_patch(coords: list[tuple[int, int]], color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(color)
    return patch


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[tuple[int, int, int], int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        coords = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
        if not coords:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        if not all(r in (r0, r1) or c in (c0, c1) for r, c in coords):
            continue
        side_touches = [
            any(r == r0 for r, _ in coords),
            any(c == c1 for _, c in coords),
            any(r == r1 for r, _ in coords),
            any(c == c0 for _, c in coords),
        ]
        if not all(side_touches):
            continue
        perimeter = 2 * h + 2 * w - 4
        if len(coords) >= perimeter:
            continue
        score = (h * w, perimeter, len(coords))
        candidates.append((score, int(color), (r0, c0, r1, c1)))
    assert candidates, "failed to detect the gapped frame"
    candidates.sort()
    _score, frame_color, bbox = candidates[-1]
    return int(frame_color), bbox


def _extract_external_pieces(
    grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]
) -> list[dict]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict] = []
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if r0 <= nr <= r1 and c0 <= nc <= c1:
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            pieces.append({"color": color, "coords": sorted(comp), "shape_key": _shape_key(comp)})
    return pieces


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(_most_common_color(g))
    frame_color, (r0, c0, r1, c1) = _find_frame(g, bg)

    border = [(r0 + rr, c0 + cc) for rr, cc in _border_cycle(r1 - r0 + 1, c1 - c0 + 1)]
    index_by_coord = {coord: idx for idx, coord in enumerate(border)}

    gap_mask = np.zeros((h, w), dtype=bool)
    for r, c in border:
        if int(g[r, c]) == int(bg):
            gap_mask[r, c] = True
    gap_comps = _components_from_mask(gap_mask)
    assert gap_comps, "the frame must contain at least one gap"

    pieces = _extract_external_pieces(g, bg, (r0, c0, r1, c1))
    assert len(pieces) == len(gap_comps), "piece count must match gap count"

    pieces_by_shape: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[dict]] = {}
    for piece in pieces:
        pieces_by_shape.setdefault(piece["shape_key"], []).append(piece)

    matched = []
    used_ids: set[int] = set()
    for gap in gap_comps:
        key = _shape_key(gap)
        candidates = pieces_by_shape.get(key, [])
        free = [piece for piece in candidates if id(piece) not in used_ids]
        assert len(free) == 1, "gap-to-piece shape matching must be unique"
        piece = free[0]
        used_ids.add(id(piece))
        matched.append(
            {
                "coords": gap,
                "color": int(piece["color"]),
                "start_idx": min(index_by_coord[(int(r), int(c))] for r, c in gap),
            }
        )

    matched.sort(key=lambda item: item["start_idx"])
    color_seq = [int(item["color"]) for item in matched]
    n = len(color_seq)
    perimeter = len(border)
    segment_of = np.empty(perimeter, dtype=int)
    for seg_idx, idxs in enumerate(np.array_split(np.arange(perimeter), n)):
        segment_of[idxs] = int(seg_idx)

    out = np.full_like(g, int(bg))
    for item in matched:
        for r, c in item["coords"]:
            out[int(r), int(c)] = int(item["color"])

    for idx, (r, c) in enumerate(border):
        if int(g[r, c]) == int(frame_color):
            out[int(r), int(c)] = int(color_seq[int(segment_of[idx])])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame gap with its matching outer fragment, then use the fragment colors to recolor the frame in clockwise segments.

    Input structure:
    - The background is black.
    - One large monochrome hollow rectangular frame has a one-cell-thick border with several missing border fragments.
    - Outside the frame are several 4-connected monochrome fragments, each in a distinct color and each shape-matching exactly one gap without rotation.

    Rule:
    - Detect the frame bbox, its missing border fragments, and the outside fragments.
    - Translate every outside fragment to the unique gap whose shape matches it exactly, preserving fragment orientation and color.
    - Read the filled-fragment colors in clockwise order along the frame border starting from the top-left corner.
    - Split the complete frame border into that many consecutive clockwise segments and recolor only the original frame-border pixels by the ordered fragment colors; the filled gap pixels keep their fragment colors.

    Key insight:
    - The outside pieces serve two roles: they repair the frame by exact shape matching, and their clockwise order on the repaired frame encodes the recoloring sequence for the original border.
    """
    return _solve(grid)


def _side_id(r: int, c: int, h: int, w: int) -> int:
    if r == 0:
        return 0
    if c == w - 1:
        return 1
    if r == h - 1:
        return 2
    return 3


def _gap_candidates(h: int, w: int, max_len: int) -> list[dict]:
    border = _border_cycle(h, w)
    perimeter = len(border)
    candidates = []
    for start in range(1, perimeter):
        for length in range(1, max_len + 1):
            end = start + length
            if end > perimeter:
                break
            coords = border[start:end]
            side_changes = sum(
                _side_id(*coords[i], h, w) != _side_id(*coords[i + 1], h, w)
                for i in range(len(coords) - 1)
            )
            if side_changes > 1:
                continue
            key = _shape_key(coords)
            is_bent = key[0] > 1 and key[1] > 1
            candidates.append(
                {
                    "start": start,
                    "indices": tuple(range(start, end)),
                    "index_set": set(range(start, end)),
                    "coords": coords,
                    "shape_key": key,
                    "bent": is_bent,
                }
            )
    return candidates


def _frame_sides_intact(h: int, w: int, gap_set: set[tuple[int, int]]) -> bool:
    sides = [
        [(0, c) for c in range(w)],
        [(r, w - 1) for r in range(1, h)],
        [(h - 1, c) for c in range(w - 2, -1, -1)],
        [(r, 0) for r in range(h - 2, 0, -1)],
    ]
    return all(any(cell not in gap_set for cell in side) for side in sides)


def _choose_gap_set(h: int, w: int, count: int) -> list[dict] | None:
    max_len = min(5, max(h, w) - 2)
    candidates = _gap_candidates(h, w, max_len)
    if len(candidates) < count:
        return None
    want_bent = bool(random.getrandbits(1)) and any(cand["bent"] for cand in candidates)
    perimeter = len(_border_cycle(h, w))
    for _ in range(250):
        order = candidates[:]
        random.shuffle(order)
        chosen = []
        used_shapes = set()
        forbidden = {0}
        bent_count = 0
        for cand in order:
            if cand["shape_key"] in used_shapes:
                continue
            if cand["index_set"] & forbidden:
                continue
            chosen.append(cand)
            used_shapes.add(cand["shape_key"])
            bent_count += int(cand["bent"])
            for idx in cand["indices"]:
                forbidden.add(int(idx))
                forbidden.add(int((idx - 1) % perimeter))
                forbidden.add(int((idx + 1) % perimeter))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            continue
        if want_bent and bent_count == 0:
            continue
        gap_set = {cell for cand in chosen for cell in cand["coords"]}
        if not _frame_sides_intact(h, w, gap_set):
            continue
        return sorted(chosen, key=lambda item: item["start"])
    return None


def _mark_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = mask.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h - 1, int(r1))
    cc1 = min(w - 1, int(c1))
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
    random.shuffle(positions)
    mask = patch != int(BG)
    for r0, c0 in positions:
        fits = True
        for rr, cc in np.argwhere(mask):
            if blocked[r0 + int(rr), c0 + int(cc)]:
                fits = False
                break
        if not fits:
            continue
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        region[mask] = patch[mask]
        _mark_box(blocked, r0 - 1, c0 - 1, r0 + ph, c0 + pw)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    gap_count = random.randint(2, 5)
    frame_h = random.randint(8, 15)
    frame_w = random.randint(8, 15)
    gaps = _choose_gap_set(frame_h, frame_w, gap_count)
    if gaps is None:
        return None

    frame_color = random.choice(PALETTE)
    piece_colors = random.sample([c for c in PALETTE if c != frame_color], gap_count)

    min_h = max(18, frame_h + 7)
    min_w = max(18, frame_w + 7)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)

    margin = 2
    if grid_h - frame_h - margin - 1 < margin or grid_w - frame_w - margin - 1 < margin:
        return None
    frame_r0 = random.randint(margin, grid_h - frame_h - margin)
    frame_c0 = random.randint(margin, grid_w - frame_w - margin)
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_box(blocked, frame_r0 - 1, frame_c0 - 1, frame_r1 + 1, frame_c1 + 1)

    gap_cells = {cell for gap in gaps for cell in gap["coords"]}
    for rr, cc in _border_cycle(frame_h, frame_w):
        if (rr, cc) in gap_cells:
            continue
        grid[frame_r0 + rr, frame_c0 + cc] = int(frame_color)

    for gap, color in zip(gaps, piece_colors):
        patch = _coords_to_patch(gap["coords"], int(color))
        if not _place_patch(grid, blocked, patch):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-gap sample")
