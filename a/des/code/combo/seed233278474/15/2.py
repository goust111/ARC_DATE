from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPPOSITE_CORNER = {"TL": "BR", "TR": "BL", "BL": "TR", "BR": "TL"}


def _inside(mask: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]


def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


@dataclass
class Piece:
    idx: int
    bbox: tuple[int, int, int, int]
    base_color: int
    mask: np.ndarray
    coords: tuple[tuple[int, int], ...]
    red_cells: tuple[tuple[int, int], tuple[int, int]]
    corner: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def pos_key(self) -> tuple[int, int]:
        return self.bbox[0], self.bbox[1]


def _corner_from_mask(mask: np.ndarray) -> str:
    h, w = mask.shape
    corners = {
        "TL": bool(mask[0, 0]),
        "TR": bool(mask[0, w - 1]),
        "BL": bool(mask[h - 1, 0]),
        "BR": bool(mask[h - 1, w - 1]),
    }
    missing = [name for name, present in corners.items() if not present]
    assert len(missing) == 1, "expected an L shape with exactly one missing bbox corner"
    return missing[0]


def _parse_piece(grid: np.ndarray, comp: list[tuple[int, int]], idx: int) -> Piece:
    vals = {int(grid[r, c]) for r, c in comp}
    assert RED in vals, "each piece must contain a red marker"
    non_red = vals - {RED}
    assert len(non_red) == 1, "each piece must have one body color plus red"
    base_color = next(iter(non_red))

    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    mask = np.zeros((h, w), dtype=bool)
    reds: list[tuple[int, int]] = []
    for r, c in comp:
        rr, cc = r - r0, c - c0
        mask[rr, cc] = True
        if int(grid[r, c]) == RED:
            reds.append((rr, cc))

    assert len(reds) == 2, "each piece must contain exactly two red pixels"
    (r1a, c1a), (r2a, c2a) = reds
    assert abs(r1a - r2a) + abs(c1a - c2a) == 1, "red pixels must be orthogonally adjacent"

    corner = _corner_from_mask(mask)
    coords = tuple((int(r), int(c)) for r, c in zip(*np.where(mask), strict=False))
    return Piece(
        idx=idx,
        bbox=(r0, c0, r1, c1),
        base_color=int(base_color),
        mask=mask,
        coords=coords,
        red_cells=(reds[0], reds[1]),
        corner=corner,
    )


def _marker_directions(piece: Piece) -> list[tuple[int, int]]:
    a, b = piece.red_cells
    candidates: list[tuple[int, int]] = []
    for rod, head in ((a, b), (b, a)):
        dr = head[0] - rod[0]
        dc = head[1] - rod[1]
        prev_r, prev_c = rod[0] - dr, rod[1] - dc
        next_r, next_c = head[0] + dr, head[1] + dc
        prev_ok = _inside(piece.mask, prev_r, prev_c) and bool(piece.mask[prev_r, prev_c])
        next_ok = _inside(piece.mask, next_r, next_c) and bool(piece.mask[next_r, next_c])
        if prev_ok and not next_ok:
            candidates.append((dr, dc))
    if candidates:
        return candidates
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return [(dr, dc), (-dr, -dc)]


def _render_rectangle_if_valid(first: Piece, second: Piece, dr: int, dc: int) -> np.ndarray | None:
    first_cells = set(first.coords)
    second_cells = {(r + dr, c + dc) for r, c in second.coords}
    if first_cells & second_cells:
        return None

    union = first_cells | second_cells
    max_r = max(r for r, _ in union)
    max_c = max(c for _, c in union)
    area = (max_r + 1) * (max_c + 1)
    if len(union) != area:
        return None

    out = np.full((max_r + 1, max_c + 1), BG, dtype=int)
    for r, c in first.coords:
        out[r, c] = first.base_color
    for r, c in second.coords:
        out[r + dr, c + dc] = second.base_color
    return out


def _search_rectangle(first: Piece, second: Piece, axis: str) -> np.ndarray | None:
    if axis == "H":
        limit = first.shape[1] + second.shape[1] + 2
        for dc in range(1, limit + 1):
            rect = _render_rectangle_if_valid(first, second, 0, dc)
            if rect is not None:
                return rect
    else:
        limit = first.shape[0] + second.shape[0] + 2
        for dr in range(1, limit + 1):
            rect = _render_rectangle_if_valid(first, second, dr, 0)
            if rect is not None:
                return rect
    return None


def _assemble_pair(a: Piece, b: Piece) -> np.ndarray | None:
    renders: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for da in _marker_directions(a):
        for db in _marker_directions(b):
            if da[0] != -db[0] or da[1] != -db[1]:
                continue
            rect = None
            if da == (0, 1):
                rect = _search_rectangle(a, b, "H")
            elif da == (0, -1):
                rect = _search_rectangle(b, a, "H")
            elif da == (1, 0):
                rect = _search_rectangle(a, b, "V")
            elif da == (-1, 0):
                rect = _search_rectangle(b, a, "V")
            if rect is not None:
                key = (rect.shape, rect.tobytes())
                renders[key] = rect
    if not renders:
        return None
    assert len(renders) == 1, "marker directions should determine a unique assembly"
    return next(iter(renders.values()))


def _match_one_side(left: list[Piece], right: list[Piece]) -> list[tuple[Piece, Piece, np.ndarray]]:
    assert len(left) == len(right)
    if not left:
        return []

    left = sorted(left, key=lambda p: p.pos_key)
    right = sorted(right, key=lambda p: p.pos_key)
    used = [False] * len(right)

    def rec(i: int) -> list[tuple[Piece, Piece, np.ndarray]] | None:
        if i == len(left):
            return []
        piece = left[i]
        for j, cand in enumerate(right):
            if used[j]:
                continue
            rect = _assemble_pair(piece, cand)
            if rect is None:
                continue
            used[j] = True
            rest = rec(i + 1)
            if rest is not None:
                return [(piece, cand, rect)] + rest
            used[j] = False
        return None

    matched = rec(0)
    assert matched is not None, "failed to match complementary L pieces"
    return matched


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pieces = [_parse_piece(g, comp, idx) for idx, comp in enumerate(_components(g))]
    assert pieces and len(pieces) % 2 == 0, "expected an even number of marked L pieces"

    groups: dict[tuple[int, int], list[Piece]] = {}
    for piece in pieces:
        groups.setdefault(piece.shape, []).append(piece)

    assembled: list[tuple[tuple[int, int], np.ndarray]] = []
    seen_ids: set[int] = set()
    for shape, group in groups.items():
        by_corner: dict[str, list[Piece]] = {"TL": [], "TR": [], "BL": [], "BR": []}
        for piece in group:
            by_corner[piece.corner].append(piece)

        for c1, c2 in (("TL", "BR"), ("TR", "BL")):
            assert len(by_corner[c1]) == len(by_corner[c2]), f"unbalanced complementary corners in {shape}"
            for a, b, rect in _match_one_side(by_corner[c1], by_corner[c2]):
                seen_ids.add(a.idx)
                seen_ids.add(b.idx)
                assembled.append((min(a.pos_key, b.pos_key), rect))

    assert len(seen_ids) == len(pieces), "every piece must belong to exactly one pair"
    assembled.sort(key=lambda item: item[0])
    rects = [rect for _, rect in assembled]
    out_h = max(rect.shape[0] for rect in rects)
    out_w = sum(rect.shape[1] for rect in rects) + len(rects) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for rect in rects:
        h, w = rect.shape
        out[:h, col : col + w] = rect
        col += w + 1
    return out


PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GRID = 30
MIN_GRID = 18
PAIR_COUNT_RANGE = (2, 4)
H_RANGE = (3, 6)
W_RANGE = (3, 7)
PAIR_GAP_RANGE = (2, 5)
GEN_RETRIES = 400


def _build_horizontal_case(h: int, w: int, split: int, cut: int, step_right: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_w = w + cut
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    assembled = np.full((h, total_w), -1, dtype=int)

    if step_right:
        left[:split, :cut] = True
        left[split:, :] = True
        right[:split, :] = True
        right[split:, w - cut :] = True

        assembled[:split, :cut] = 0
        assembled[split:, :w] = 0
        assembled[:split, cut:] = 1
        assembled[split:, w:] = 1
    else:
        left[:split, :] = True
        left[split:, :cut] = True
        right[:split, w - cut :] = True
        right[split:, :] = True

        assembled[:split, :w] = 0
        assembled[split:, :cut] = 0
        assembled[:split, w:] = 1
        assembled[split:, cut:] = 1

    assert np.all(assembled >= 0)
    return left, right, assembled


def _rotate_dir_by_transpose(direction: tuple[int, int]) -> tuple[int, int]:
    return direction[1], direction[0]


def _marker_candidates(mask: np.ndarray, direction: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    dr, dc = direction
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in zip(*np.where(mask), strict=False):
        rod = (int(r), int(c))
        head = (rod[0] + dr, rod[1] + dc)
        prev = (rod[0] - dr, rod[1] - dc)
        nxt = (head[0] + dr, head[1] + dc)
        if not (_inside(mask, head[0], head[1]) and bool(mask[head])):
            continue
        if not (_inside(mask, prev[0], prev[1]) and bool(mask[prev])):
            continue
        if _inside(mask, nxt[0], nxt[1]) and bool(mask[nxt]):
            continue
        out.append((rod, head))
    return out


def _paint_piece(mask: np.ndarray, color: int, direction: tuple[int, int]) -> np.ndarray | None:
    candidates = _marker_candidates(mask, direction)
    if not candidates:
        return None
    rod, head = random.choice(candidates)
    arr = np.full(mask.shape, BG, dtype=int)
    arr[mask] = int(color)
    arr[rod] = RED
    arr[head] = RED
    return arr


def _make_pair_spec(case_name: str, color_a: int, color_b: int) -> dict | None:
    h = random.randint(*H_RANGE)
    w = random.randint(*W_RANGE)
    split = random.randint(1, h - 1)
    cut = random.randint(1, w - 1)

    left_mask, right_mask, assembled_owner = _build_horizontal_case(h, w, split, cut, step_right=case_name in {"HR", "VR"})
    dir_a = (0, 1)
    dir_b = (0, -1)

    if case_name.startswith("V"):
        left_mask = left_mask.T.copy()
        right_mask = right_mask.T.copy()
        assembled_owner = assembled_owner.T.copy()
        dir_a = _rotate_dir_by_transpose(dir_a)
        dir_b = _rotate_dir_by_transpose(dir_b)

    piece_a = _paint_piece(left_mask, color_a, dir_a)
    piece_b = _paint_piece(right_mask, color_b, dir_b)
    if piece_a is None or piece_b is None:
        return None

    assembled = np.full(assembled_owner.shape, BG, dtype=int)
    assembled[assembled_owner == 0] = int(color_a)
    assembled[assembled_owner == 1] = int(color_b)

    orientation = "H" if dir_a[0] == 0 else "V"
    return {
        "piece_a": piece_a,
        "piece_b": piece_b,
        "assembled": assembled,
        "shape": piece_a.shape,
        "orientation": orientation,
    }


def _reserve_ok(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_pairs(specs: list[dict]) -> tuple[np.ndarray, list[tuple[tuple[int, int], np.ndarray]]] | None:
    max_gap = PAIR_GAP_RANGE[1]
    max_pair_h = max(
        spec["piece_a"].shape[0]
        + (spec["piece_b"].shape[0] + max_gap if spec["orientation"] == "V" else 0)
        for spec in specs
    )
    max_pair_w = max(
        spec["piece_a"].shape[1]
        + (spec["piece_b"].shape[1] + max_gap if spec["orientation"] == "H" else 0)
        for spec in specs
    )
    grid_h = random.randint(max(MIN_GRID, max_pair_h + 4), MAX_GRID)
    grid_w = random.randint(max(MIN_GRID, max_pair_w + 4), MAX_GRID)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    outputs: list[tuple[tuple[int, int], np.ndarray]] = []

    random.shuffle(specs)
    for spec in specs:
        a = spec["piece_a"]
        b = spec["piece_b"]
        gap = random.randint(*PAIR_GAP_RANGE)
        if spec["orientation"] == "H":
            pair_h = a.shape[0]
            pair_w = a.shape[1] + gap + b.shape[1]
        else:
            pair_h = a.shape[0] + gap + b.shape[0]
            pair_w = a.shape[1]

        if grid_h - pair_h - 1 < 1 or grid_w - pair_w - 1 < 1:
            return None

        placed = False
        for _ in range(250):
            r0 = random.randint(1, grid_h - pair_h - 1)
            c0 = random.randint(1, grid_w - pair_w - 1)
            if not _reserve_ok(occupied, r0, c0, pair_h, pair_w):
                continue

            if spec["orientation"] == "H":
                ar, ac = r0, c0
                br, bc = r0, c0 + a.shape[1] + gap
            else:
                ar, ac = r0, c0
                br, bc = r0 + a.shape[0] + gap, c0

            grid[ar : ar + a.shape[0], ac : ac + a.shape[1]][a != BG] = a[a != BG]
            grid[br : br + b.shape[0], bc : bc + b.shape[1]][b != BG] = b[b != BG]
            occupied[r0 : r0 + pair_h, c0 : c0 + pair_w] = True
            outputs.append(((min(ar, br), min(ac, bc)), spec["assembled"]))
            placed = True
            break
        if not placed:
            return None
    return grid, outputs


def _pack_expected(outputs: Iterable[tuple[tuple[int, int], np.ndarray]]) -> np.ndarray:
    ordered = sorted(outputs, key=lambda item: item[0])
    rects = [rect for _, rect in ordered]
    out_h = max(rect.shape[0] for rect in rects)
    out_w = sum(rect.shape[1] for rect in rects) + len(rects) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    c = 0
    for rect in rects:
        h, w = rect.shape
        out[:h, c : c + w] = rect
        c += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, 2 * n_pairs)
    cases = [random.choice(("HR", "HL", "VR", "VL")) for _ in range(n_pairs)]

    specs: list[dict] = []
    used_shapes: set[tuple[int, int]] = set()
    for i in range(n_pairs):
        spec = _make_pair_spec(cases[i], colors[2 * i], colors[2 * i + 1])
        if spec is None:
            return None
        if tuple(spec["shape"]) in used_shapes:
            return None
        used_shapes.add(tuple(spec["shape"]))
        specs.append(spec)

    placed = _place_pairs(specs)
    if placed is None:
        return None
    grid, outputs = placed
    expected = _pack_expected(outputs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
