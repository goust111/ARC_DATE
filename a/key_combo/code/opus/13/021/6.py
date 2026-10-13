from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


WILDCARD_COLOR = 1
HOLE_COLOR = 9
MARKER_COLOR = 6
BACKGROUND_COLOR = 0
GEN_MAX_ATTEMPTS = 400


@dataclass(frozen=True)
class Piece:
    index: int
    array: np.ndarray
    h: int
    w: int


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _marker_corner_from_missing(missing_pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if missing_pos == (r0, c0):
        return "BR"
    if missing_pos == (r0, c1):
        return "BL"
    if missing_pos == (r1, c0):
        return "TR"
    return "TL"


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    counts = Counter(map(int, grid.flatten()))
    marker_color = MARKER_COLOR
    if int(counts.get(int(marker_color), 0)) != 3:
        raise AssertionError("expected one magenta 3-cell L marker")
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(marker_color))]
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    assert r1 - r0 == 1 and c1 - c0 == 1, "marker must occupy a 2x2 box"
    all_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    missing = list(all_cells - set(pts))
    if len(missing) == 1:
        bbox = (r0, c0, r1, c1)
        return int(marker_color), missing[0], _marker_corner_from_missing(missing[0], bbox)
    raise AssertionError("expected one 3-cell L marker")


def _extract_pieces(grid: np.ndarray, bg: int, marker: int) -> list[Piece]:
    mask = (grid != int(bg)) & (grid != int(marker))
    pieces: list[Piece] = []
    for idx, coords in enumerate(_components(mask)):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(crop != bg), "pieces must be solid rectangles"
        pieces.append(Piece(idx, crop, int(crop.shape[0]), int(crop.shape[1])))
    assert len(pieces) == 4, "expected exactly four pieces"
    return pieces


def _rect_origin_for_corner(missing: tuple[int, int], corner: str, h: int, w: int) -> tuple[int, int]:
    mr, mc = map(int, missing)
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - w + 1
    if corner == "BL":
        return mr - h + 1, mc
    return mr - h + 1, mc - w + 1


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h == 0:
            pairs.append((h, area // h))
    return pairs


def _first_empty(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(~mask)
    if coords.size == 0:
        return None
    r, c = coords[0]
    return int(r), int(c)


def _enumerate_tilings(pieces: list[Piece], rect_h: int, rect_w: int) -> list[dict[int, tuple[int, int]]]:
    occupied = np.zeros((rect_h, rect_w), dtype=bool)
    used = [False] * len(pieces)
    placements: dict[int, tuple[int, int]] = {}
    out: list[dict[int, tuple[int, int]]] = []

    def dfs() -> None:
        cell = _first_empty(occupied)
        if cell is None:
            out.append(dict(placements))
            return
        r0, c0 = cell
        for piece in pieces:
            if used[piece.index]:
                continue
            if r0 + piece.h > rect_h or c0 + piece.w > rect_w:
                continue
            if np.any(occupied[r0 : r0 + piece.h, c0 : c0 + piece.w]):
                continue
            occupied[r0 : r0 + piece.h, c0 : c0 + piece.w] = True
            used[piece.index] = True
            placements[piece.index] = (r0, c0)
            dfs()
            placements.pop(piece.index)
            used[piece.index] = False
            occupied[r0 : r0 + piece.h, c0 : c0 + piece.w] = False

    dfs()
    return out


def _edge_compatible(a: int, b: int, wildcard: int) -> bool:
    return int(a) == int(wildcard) or int(b) == int(wildcard) or int(a) == int(b)


def _assemble_candidate(
    pieces: list[Piece],
    placement: dict[int, tuple[int, int]],
    rect_h: int,
    rect_w: int,
    wildcard: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    canvas = np.full((rect_h, rect_w), -1, dtype=int)
    owner = np.full((rect_h, rect_w), -1, dtype=int)
    for piece in pieces:
        r0, c0 = placement[piece.index]
        r1 = r0 + piece.h
        c1 = c0 + piece.w
        canvas[r0:r1, c0:c1] = piece.array
        owner[r0:r1, c0:c1] = piece.index

    for r in range(rect_h):
        for c in range(rect_w):
            if c + 1 < rect_w and owner[r, c] != owner[r, c + 1]:
                if not _edge_compatible(int(canvas[r, c]), int(canvas[r, c + 1]), wildcard):
                    return None
            if r + 1 < rect_h and owner[r, c] != owner[r + 1, c]:
                if not _edge_compatible(int(canvas[r, c]), int(canvas[r + 1, c]), wildcard):
                    return None
    return canvas, owner


def _visible_symmetry_mismatches(grid: np.ndarray, wildcard: int, hole: int = HOLE_COLOR) -> int:
    h, w = grid.shape
    mismatches = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (int(wildcard), int(hole)):
                continue
            hr = h - 1 - r
            vc = w - 1 - c
            if int(grid[hr, c]) not in (int(hole), color):
                mismatches += 1
            if int(grid[r, vc]) not in (int(hole), color):
                mismatches += 1
    return mismatches


def _repair_holes(grid: np.ndarray, hole: int = HOLE_COLOR) -> np.ndarray | None:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    changed = True
    while changed:
        changed = False
        for r, c in np.argwhere(out == int(hole)):
            rr = int(r)
            cc = int(c)
            mates = (
                (h - 1 - rr, cc),
                (rr, w - 1 - cc),
                (h - 1 - rr, w - 1 - cc),
            )
            for mr, mc in mates:
                color = int(out[mr, mc])
                if color != int(hole):
                    out[rr, cc] = color
                    changed = True
                    break
    if np.any(out == int(hole)):
        return None
    return out


def _full_symmetry_mismatches(grid: np.ndarray) -> int:
    h, w = grid.shape
    mismatches = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if int(grid[h - 1 - r, c]) != color:
                mismatches += 1
            if int(grid[r, w - 1 - c]) != color:
                mismatches += 1
    return mismatches


def _candidate_signature(placement: dict[int, tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(placement[idx] for idx in sorted(placement))


def _output_signature(candidate: dict) -> tuple:
    repaired = candidate["repaired"]
    if repaired is None:
        return ("invalid",)
    return (candidate["rect_shape"], candidate["rect_origin"], repaired.tobytes())


def _solve_candidates(grid: np.ndarray) -> list[dict]:
    bg = int(Counter(map(int, grid.flatten())).most_common(1)[0][0])
    marker, missing, corner = _find_marker(grid, bg)
    pieces = _extract_pieces(grid, bg, marker)

    piece_pixels: list[int] = []
    for piece in pieces:
        piece_pixels.extend(map(int, piece.array.flatten()))
    wildcard = int(Counter(piece_pixels).most_common(1)[0][0])

    total_area = int(sum(piece.h * piece.w for piece in pieces))
    grid_h, grid_w = grid.shape
    candidates: list[dict] = []
    for rect_h, rect_w in _factor_pairs(total_area):
        top, left = _rect_origin_for_corner(missing, corner, rect_h, rect_w)
        if top < 0 or left < 0 or top + rect_h > grid_h or left + rect_w > grid_w:
            continue
        for placement in _enumerate_tilings(pieces, rect_h, rect_w):
            assembled = _assemble_candidate(pieces, placement, rect_h, rect_w, wildcard)
            if assembled is None:
                continue
            canvas, owner = assembled
            visible_score = _visible_symmetry_mismatches(canvas, wildcard)
            repaired = _repair_holes(canvas)
            full_score = 10**9 if repaired is None else _full_symmetry_mismatches(repaired)
            candidates.append(
                {
                    "rect_shape": (rect_h, rect_w),
                    "rect_origin": (top, left),
                    "corner": corner,
                    "wildcard": wildcard,
                    "canvas": canvas,
                    "owner": owner,
                    "placement": placement,
                    "placement_sig": _candidate_signature(placement),
                    "visible_score": int(visible_score),
                    "repaired": repaired,
                    "full_score": int(full_score),
                }
            )
    return candidates


def _select_candidate(candidates: list[dict]) -> dict:
    assert candidates, "expected at least one legal assembly"
    best = min(candidates, key=lambda cand: (cand["visible_score"], cand["full_score"], cand["placement_sig"]))
    best_key = (best["visible_score"], best["full_score"])
    ties = [cand for cand in candidates if (cand["visible_score"], cand["full_score"]) == best_key]
    tie_outputs = {_output_signature(cand) for cand in ties}
    assert len(tie_outputs) == 1, "expected the best assemblies to agree on one output"
    assert best["visible_score"] == 0, "valid inputs must have one symmetry-consistent assembly"
    assert best["repaired"] is not None and best["full_score"] == 0, "holes must repair to a fully symmetric rectangle"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four damaged fragments into one rectangle and repair the hidden colors by global mirror symmetry.

    Input structure:
    - A black background contains exactly four non-touching rectangular fragments plus one 3-cell magenta L marker.
    - Fragment edges contain a dominant wildcard color, while a few interior cells use 1-2 stronger colors.
    - Some strong-color fragment cells are replaced by maroon damage pixels.
    - The missing cell of the marker indicates which corner of the final rectangle is pinned in place.

    Rule:
    - Identify the wildcard as the most frequent fragment color.
    - Translate the four fragments, without rotation, so they tile a rectangle anchored at the marker corner.
    - Shared fragment edges must match cell-by-cell, except the wildcard can match any color.
    - If several tilings satisfy edge matching, choose the unique one whose visible non-wildcard, non-maroon cells obey both horizontal and vertical reflection.
    - Repair each maroon cell from the first available color among its horizontal reflection, vertical reflection, and 180-degree partner, then output only the repaired rectangle at the tagged corner.

    Key insight:
    - Wildcard edges make several packings look locally valid, so the real arrangement is the one that becomes globally symmetric and therefore reveals the damaged colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    best = _select_candidate(_solve_candidates(g))
    rect = best["repaired"]
    assert rect is not None
    out = np.full_like(g, bg)
    r0, c0 = best["rect_origin"]
    h, w = rect.shape
    out[r0 : r0 + h, c0 : c0 + w] = rect
    return out


def _partition_template() -> tuple[int, int, list[tuple[int, int, int, int]]]:
    if np.random.rand() < 0.5:
        rect_h = 8
        left_w = int(np.random.choice([3, 4]))
        strip_w = int(np.random.choice([3, 4]))
        rect_w = left_w + strip_w + strip_w
        split = int(np.random.choice([3, 5]))
        return rect_h, rect_w, [
            (0, 0, split - 1, left_w - 1),
            (split, 0, rect_h - 1, left_w - 1),
            (0, left_w, rect_h - 1, left_w + strip_w - 1),
            (0, left_w + strip_w, rect_h - 1, rect_w - 1),
        ]
    top_h = int(np.random.choice([3, 4]))
    strip_h = int(np.random.choice([3, 4]))
    rect_h = top_h + strip_h + strip_h
    rect_w = 8
    split = int(np.random.choice([3, 5]))
    return rect_h, rect_w, [
        (0, 0, top_h - 1, split - 1),
        (0, split, top_h - 1, rect_w - 1),
        (top_h, 0, top_h + strip_h - 1, rect_w - 1),
        (top_h + strip_h, 0, rect_h - 1, rect_w - 1),
    ]


def _build_allowed_mask(rect_h: int, rect_w: int, pieces: list[tuple[int, int, int, int]]) -> np.ndarray:
    allowed = np.ones((rect_h, rect_w), dtype=bool)
    for r0, c0, r1, c1 in pieces:
        allowed[r0, c0 : c1 + 1] = False
        allowed[r1, c0 : c1 + 1] = False
        allowed[r0 : r1 + 1, c0] = False
        allowed[r0 : r1 + 1, c1] = False
    return allowed


def _orbit(rect_h: int, rect_w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (int(r), int(c)),
        (int(r), rect_w - 1 - int(c)),
        (rect_h - 1 - int(r), int(c)),
        (rect_h - 1 - int(r), rect_w - 1 - int(c)),
    }
    return tuple(sorted(pts))


def _piece_index_for_cell(pieces: list[tuple[int, int, int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    for idx, (r0, c0, r1, c1) in enumerate(pieces):
        if r0 <= r <= r1 and c0 <= c <= c1:
            return idx
    raise AssertionError("cell must belong to one piece")


def _symmetric_rectangle() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    strong_colors = np.random.choice([2, 3, 4, 7, 8], size=int(np.random.choice([2, 3])), replace=False)
    rect_h, rect_w, piece_boxes = _partition_template()
    allowed = _build_allowed_mask(rect_h, rect_w, piece_boxes)
    orbits: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(rect_h):
        for c in range(rect_w):
            orb = _orbit(rect_h, rect_w, r, c)
            if orb in seen:
                continue
            seen.add(orb)
            if all(bool(allowed[rr, cc]) for rr, cc in orb):
                orbits.append(orb)
    if len(orbits) < 4:
        raise AssertionError("not enough interior symmetry orbits")

    np.random.shuffle(orbits)
    rect = np.full((rect_h, rect_w), WILDCARD_COLOR, dtype=int)
    coverage = [0, 0, 0, 0]
    chosen: list[tuple[tuple[int, int], ...]] = []
    for orb in orbits:
        touched = {_piece_index_for_cell(piece_boxes, cell) for cell in orb}
        if any(coverage[idx] == 0 for idx in touched):
            chosen.append(orb)
            for idx in touched:
                coverage[idx] += 1
        if all(v > 0 for v in coverage):
            break
    extra_count = int(np.random.randint(2, 5))
    for orb in orbits:
        if orb in chosen:
            continue
        if extra_count <= 0:
            break
        chosen.append(orb)
        extra_count -= 1
    if not all(v > 0 for v in coverage):
        raise AssertionError("every piece must expose strong colors")

    for orb in chosen:
        color = int(np.random.choice(strong_colors))
        for r, c in orb:
            rect[r, c] = color
    return rect, piece_boxes


def _cut_pieces(rect: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    for r0, c0, r1, c1 in boxes:
        parts.append(rect[r0 : r1 + 1, c0 : c1 + 1].copy())
    return parts


def _add_holes(rect: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = rect.copy()
    piece_candidates = []
    for idx, (r0, c0, r1, c1) in enumerate(boxes):
        strong_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(out[r, c]) != WILDCARD_COLOR]
        if strong_cells:
            piece_candidates.append((idx, strong_cells))
    chosen_piece_count = min(len(piece_candidates), int(np.random.choice([1, 2])))
    chosen_piece_rows = np.random.choice(len(piece_candidates), size=chosen_piece_count, replace=False)
    chosen_piece_ids = {piece_candidates[int(row)][0] for row in np.atleast_1d(chosen_piece_rows)}
    occluded_orbits: set[tuple[tuple[int, int], ...]] = set()
    for idx, strong_cells in piece_candidates:
        if idx not in chosen_piece_ids:
            continue
        np.random.shuffle(strong_cells)
        placed = 0
        target = int(np.random.choice([1, 2]))
        for cell in strong_cells:
            orb = _orbit(*out.shape, *cell)
            visible = sum(int(out[r, c] != HOLE_COLOR) for r, c in orb)
            if visible <= 1:
                continue
            if orb in occluded_orbits and placed >= 1:
                continue
            out[cell] = HOLE_COLOR
            occluded_orbits.add(orb)
            placed += 1
            if placed >= target:
                break
    assert np.any(out == HOLE_COLOR), "expected at least one occlusion"
    return out


def _marker_cells(missing: tuple[int, int], corner: str) -> list[tuple[int, int]]:
    mr, mc = missing
    if corner == "BR":
        return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]


def _random_anchor(grid_h: int, grid_w: int, rect_h: int, rect_w: int) -> tuple[tuple[int, int], str]:
    corners = ["TL", "TR", "BL", "BR"]
    np.random.shuffle(corners)
    for corner in corners:
        for _ in range(80):
            if corner == "TL":
                mr = int(np.random.randint(1, grid_h - rect_h))
                mc = int(np.random.randint(1, grid_w - rect_w))
            elif corner == "TR":
                mr = int(np.random.randint(1, grid_h - rect_h))
                mc = int(np.random.randint(rect_w, grid_w - 1))
            elif corner == "BL":
                mr = int(np.random.randint(rect_h, grid_h - 1))
                mc = int(np.random.randint(1, grid_w - rect_w))
            else:
                mr = int(np.random.randint(rect_h, grid_h - 1))
                mc = int(np.random.randint(rect_w, grid_w - 1))
            marker = _marker_cells((mr, mc), corner)
            if all(0 <= r < grid_h and 0 <= c < grid_w for r, c in marker):
                return (mr, mc), corner
    raise AssertionError("failed to place marker anchor")


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _scatter_pieces(
    grid_h: int,
    grid_w: int,
    pieces: list[np.ndarray],
    forbidden: Iterable[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    blocked = list(forbidden)
    positions: list[tuple[int, int]] = []
    order = np.random.permutation(len(pieces))
    pos_by_piece: list[tuple[int, int] | None] = [None] * len(pieces)
    for piece_idx in order:
        piece = pieces[int(piece_idx)]
        h, w = piece.shape
        placed = False
        for _ in range(500):
            r0 = int(np.random.randint(0, grid_h - h + 1))
            c0 = int(np.random.randint(0, grid_w - w + 1))
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_bbox_overlap(box, other, pad=1) for other in blocked):
                continue
            blocked.append(box)
            pos_by_piece[int(piece_idx)] = (r0, c0)
            placed = True
            break
        if not placed:
            raise AssertionError("failed to scatter pieces")
    return [pos for pos in pos_by_piece if pos is not None]


def _compose_input() -> tuple[np.ndarray, np.ndarray, dict]:
    rect, boxes = _symmetric_rectangle()
    rect = _add_holes(rect, boxes)
    pieces = _cut_pieces(rect, boxes)
    rect_h, rect_w = rect.shape

    grid_h = int(np.random.randint(max(20, rect_h + 10), min(28, 31)))
    grid_w = int(np.random.randint(max(20, rect_w + 10), min(28, 31)))
    missing, corner = _random_anchor(grid_h, grid_w, rect_h, rect_w)
    top, left = _rect_origin_for_corner(missing, corner, rect_h, rect_w)
    target_box = (top, left, top + rect_h - 1, left + rect_w - 1)
    marker_cells = _marker_cells(missing, corner)
    marker_bounds = (
        min(r for r, _ in marker_cells),
        min(c for _, c in marker_cells),
        max(r for r, _ in marker_cells),
        max(c for _, c in marker_cells),
    )
    positions = _scatter_pieces(grid_h, grid_w, pieces, [target_box, marker_bounds])

    inp = np.full((grid_h, grid_w), BACKGROUND_COLOR, dtype=int)
    for piece, (r0, c0) in zip(pieces, positions):
        h, w = piece.shape
        inp[r0 : r0 + h, c0 : c0 + w] = piece
    for r, c in marker_cells:
        inp[r, c] = MARKER_COLOR

    out = np.full_like(inp, BACKGROUND_COLOR)
    repaired = _repair_holes(rect)
    assert repaired is not None
    out[top : top + rect_h, left : left + rect_w] = repaired
    meta = {
        "corner": corner,
        "rect_shape": (rect_h, rect_w),
        "target_box": target_box,
        "positions": positions,
    }
    return inp, out, meta


def _is_good_instance(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        candidates = _solve_candidates(inp)
    except AssertionError:
        return False
    if len(candidates) < 2:
        return False
    best_key = min((cand["visible_score"], cand["full_score"]) for cand in candidates)
    if best_key != (0, 0):
        return False
    ties = [cand for cand in candidates if (cand["visible_score"], cand["full_score"]) == best_key]
    if len({_output_signature(cand) for cand in ties}) != 1:
        return False
    best = _select_candidate(candidates)
    produced = np.full_like(inp, BACKGROUND_COLOR)
    rect = best["repaired"]
    assert rect is not None
    r0, c0 = best["rect_origin"]
    h, w = rect.shape
    produced[r0 : r0 + h, c0 : c0 + w] = rect
    return np.array_equal(produced, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            inp, out, _meta = _compose_input()
        except AssertionError:
            continue
        if _is_good_instance(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid puzzle input")
