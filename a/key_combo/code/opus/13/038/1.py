from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
VERTICAL_STEPS = (-1, 1)
GEN_ATTEMPTS = 600
PAIR_GAP = 2


@dataclass(frozen=True)
class Domino:
    color: int
    orientation: str
    r: int
    c: int

    def cells(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.orientation == "H":
            return ((self.r, self.c), (self.r, self.c + 1))
        return ((self.r, self.c), (self.r + 1, self.c))

    def sources(self) -> tuple["Source", "Source"]:
        return tuple(Source(self.orientation, self.color, r, c) for r, c in self.cells())


@dataclass(frozen=True)
class Source:
    kind: str
    color: int
    r: int
    c: int


@dataclass(frozen=True)
class Claim:
    kind: str
    color: int
    src_r: int
    src_c: int
    dist: int


@dataclass(frozen=True)
class Placement:
    orientation: str
    r: int
    c: int


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_for_placement(p: Placement) -> tuple[int, int, int, int]:
    if p.orientation == "H":
        return p.r, p.c, p.r, p.c + 1
    return p.r, p.c, p.r + 1, p.c


def _far_enough(a: Placement, b: Placement, gap: int = PAIR_GAP) -> bool:
    ar0, ac0, ar1, ac1 = _bbox_for_placement(a)
    br0, bc0, br1, bc1 = _bbox_for_placement(b)
    return bool(ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _placement_cells(p: Placement) -> tuple[tuple[int, int], tuple[int, int]]:
    if p.orientation == "H":
        return ((p.r, p.c), (p.r, p.c + 1))
    return ((p.r, p.c), (p.r + 1, p.c))


def _placement_sources(p: Placement) -> tuple[Source, Source]:
    return tuple(Source(p.orientation, -1, r, c) for r, c in _placement_cells(p))


def _extract_dominoes(grid: np.ndarray) -> list[Domino]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    dominoes: list[Domino] = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) != BG):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground object must be a 2-pixel marker"
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
        (r0, c0), (r1, c1) = coords
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        if r0 == r1 and c1 == c0 + 1:
            dominoes.append(Domino(color, "H", r0, c0))
        elif c0 == c1 and r1 == r0 + 1:
            dominoes.append(Domino(color, "V", r0, c0))
        else:
            raise AssertionError("every marker must be a horizontal or vertical domino")
    assert dominoes, "expected at least one marker"
    return dominoes


def _iter_claims_for_source(src: Source, shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = [(src.r, src.c, 0)]
    h, w = shape
    if src.kind == "H":
        src_parity = (src.r + src.c) & 1
        for dr in VERTICAL_STEPS:
            rr = src.r + dr
            while 0 <= rr < h:
                if ((rr + src.c) & 1) == src_parity:
                    cells.append((rr, src.c, abs(rr - src.r)))
                rr += dr
    else:
        for dr, dc in DIAGONALS:
            rr = src.r + dr
            cc = src.c + dc
            step = 1
            while 0 <= rr < h and 0 <= cc < w:
                cells.append((rr, cc, 2 * step))
                rr += dr
                cc += dc
                step += 1
    return cells


def _claim_priority(claim: Claim) -> tuple[int, int, int, int]:
    class_priority = 0 if claim.kind == "H" else 1
    return class_priority, claim.dist, claim.src_r, claim.src_c


def _solve_from_dominoes(
    dominoes: list[Domino], shape: tuple[int, int], return_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:
    claim_map: dict[tuple[int, int], list[Claim]] = {}
    marker_cells: dict[tuple[int, int], int] = {}
    sources: list[Source] = []

    for domino in dominoes:
        for cell in domino.cells():
            marker_cells[cell] = domino.color
        for src in domino.sources():
            sources.append(src)
            for r, c, dist in _iter_claims_for_source(src, shape):
                claim = Claim(src.kind, src.color, src.r, src.c, dist)
                claim_map.setdefault((r, c), []).append(claim)

    out = np.zeros(shape, dtype=int)
    winner_map: dict[tuple[int, int], Claim] = {}
    for cell, claims in claim_map.items():
        best = min(claims, key=_claim_priority)
        winner_map[cell] = best
        out[cell] = best.color
    for (r, c), color in marker_cells.items():
        out[r, c] = color

    if not return_meta:
        return out

    meta = {
        "claim_map": claim_map,
        "winner_map": winner_map,
        "marker_cells": marker_cells,
        "sources": sources,
        "dominoes": dominoes,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split domino markers by orientation, then emit diagonal solid rays or vertical periodic rays with horizontal priority at overlaps.

    Input structure:
    - The grid is a black background with 3-5 same-color domino markers.
    - Every foreground object is exactly two 4-connected pixels, either horizontal or vertical.
    - Marker colors are distinct, and the markers sit away from the border.

    Rule:
    - Treat vertical dominoes as diagonal-ray emitters: from each of their two pixels, extend in all four diagonal directions to the grid edge and fill every visited cell with that marker's color.
    - Treat horizontal dominoes as vertical periodic emitters: from each of their two pixels, extend upward and downward to the grid edge, but color only cells whose `(row + col) % 2` matches the source pixel.
    - When multiple rays claim one cell, horizontal-ray claims beat vertical-ray claims; within the same class, the smaller Manhattan distance wins, then smaller source row, then smaller source column.
    - After ray painting, restore every original marker pixel so the input dominoes remain unchanged.

    Key insight:
    - Orientation encodes two things at once: it chooses both the ray geometry (diagonal versus vertical) and the fill density (solid versus parity-skipping), so the output cannot be derived from color alone.
    """
    g = np.asarray(grid, dtype=int)
    dominoes = _extract_dominoes(g)
    return _solve_from_dominoes(dominoes, g.shape)


def _render_input(shape: tuple[int, int], placements: list[Placement]) -> np.ndarray:
    colors = random.sample(COLORS, len(placements))
    random.shuffle(colors)
    dominoes = [Domino(color, p.orientation, p.r, p.c) for color, p in zip(colors, placements)]
    grid = np.zeros(shape, dtype=int)
    for domino in dominoes:
        for r, c in domino.cells():
            grid[r, c] = domino.color
    return grid


def _all_positions(shape: tuple[int, int], orientation: str) -> list[Placement]:
    h, w = shape
    positions: list[Placement] = []
    if orientation == "H":
        for r in range(1, h - 1):
            for c in range(1, w - 2):
                positions.append(Placement("H", r, c))
    else:
        for r in range(1, h - 2):
            for c in range(1, w - 1):
                positions.append(Placement("V", r, c))
    return positions


def _fits(existing: list[Placement], candidate: Placement) -> bool:
    return all(_far_enough(candidate, other) for other in existing)


def _claimed_cells_for_placements(placements: list[Placement], shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for placement in placements:
        for src in _placement_sources(placement):
            for r, c, _ in _iter_claims_for_source(src, shape):
                cells.add((r, c))
    return cells


def _marker_cells_for_placements(placements: list[Placement]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for placement in placements:
        out.update(_placement_cells(placement))
    return out


def _cross_overlap(base: list[Placement], candidate: Placement, shape: tuple[int, int]) -> bool:
    base_cells = _claimed_cells_for_placements(base, shape)
    cand_cells = _claimed_cells_for_placements([candidate], shape)
    blocked = _marker_cells_for_placements(base + [candidate])
    return any(cell not in blocked for cell in base_cells & cand_cells)


def _pick_candidate(
    shape: tuple[int, int],
    orientation: str,
    existing: list[Placement],
    must_cross: list[Placement] | None = None,
) -> Placement | None:
    candidates = _all_positions(shape, orientation)
    random.shuffle(candidates)
    for candidate in candidates:
        if not _fits(existing, candidate):
            continue
        if must_cross is not None and not _cross_overlap(must_cross, candidate, shape):
            continue
        return candidate
    return None


def _build_row_tie(shape: tuple[int, int]) -> list[Placement] | None:
    h, w = shape
    anchors = [(r, c) for r in range(1, h - 5) for c in range(1, w - 2)]
    random.shuffle(anchors)
    for r, c in anchors:
        pair = [Placement("H", r, c), Placement("H", r + 4, c)]
        if not _far_enough(pair[0], pair[1]):
            continue
        vertical = _pick_candidate(shape, "V", pair, must_cross=pair)
        if vertical is not None:
            return pair + [vertical]
    return None


def _build_col_tie(shape: tuple[int, int]) -> list[Placement] | None:
    h, w = shape
    anchors = [(r, c) for r in range(1, h - 2) for c in range(1, w - 5)]
    random.shuffle(anchors)
    for r, c in anchors:
        pair = [Placement("V", r, c), Placement("V", r, c + 4)]
        if not _far_enough(pair[0], pair[1]):
            continue
        horizontal = _pick_candidate(shape, "H", pair, must_cross=pair)
        if horizontal is not None:
            return pair + [horizontal]
    return None


def _add_extras(shape: tuple[int, int], placements: list[Placement], target_count: int) -> bool:
    attempts = 0
    while len(placements) < target_count and attempts < 80:
        attempts += 1
        orientation = random.choice(("H", "V"))
        candidate = _pick_candidate(shape, orientation, placements)
        if candidate is None:
            continue
        placements.append(candidate)
    return len(placements) == target_count


def _has_cross_conflict(meta: dict) -> bool:
    markers = set(meta["marker_cells"])
    for cell, claims in meta["claim_map"].items():
        if cell in markers:
            continue
        if {claim.kind for claim in claims} == {"H", "V"}:
            return True
    return False


def _has_same_class_conflict(meta: dict) -> bool:
    markers = set(meta["marker_cells"])
    for cell, claims in meta["claim_map"].items():
        if cell in markers:
            continue
        for kind in ("H", "V"):
            same_kind = [claim for claim in claims if claim.kind == kind]
            if len({claim.color for claim in same_kind}) >= 2:
                return True
    return False


def _has_row_tie(meta: dict) -> bool:
    markers = set(meta["marker_cells"])
    for cell, claims in meta["claim_map"].items():
        if cell in markers:
            continue
        same_kind = [claim for claim in claims if claim.kind == "H"]
        if len({claim.color for claim in same_kind}) < 2:
            continue
        best_dist = min(claim.dist for claim in same_kind)
        tied = [claim for claim in same_kind if claim.dist == best_dist]
        if len(tied) < 2 or len({claim.src_r for claim in tied}) < 2:
            continue
        expected = min(tied, key=lambda claim: (claim.src_r, claim.src_c))
        if meta["winner_map"].get(cell) == expected:
            return True
    return False


def _has_col_tie(meta: dict) -> bool:
    markers = set(meta["marker_cells"])
    for cell, claims in meta["claim_map"].items():
        if cell in markers:
            continue
        same_kind = [claim for claim in claims if claim.kind == "V"]
        if len({claim.color for claim in same_kind}) < 2:
            continue
        best_dist = min(claim.dist for claim in same_kind)
        tied = [claim for claim in same_kind if claim.dist == best_dist]
        if len(tied) < 2:
            continue
        min_row = min(claim.src_r for claim in tied)
        row_group = [claim for claim in tied if claim.src_r == min_row]
        if len({claim.src_c for claim in row_group}) < 2:
            continue
        expected = min(row_group, key=lambda claim: claim.src_c)
        if meta["winner_map"].get(cell) == expected:
            return True
    return False


def _periodic_visible(out: np.ndarray, meta: dict) -> bool:
    winner_map = meta["winner_map"]
    h, w = out.shape
    for src in meta["sources"]:
        if src.kind != "H":
            continue
        for dr in VERTICAL_STEPS:
            r1 = src.r + dr
            r2 = src.r + 2 * dr
            if not (_in_bounds((h, w), r1, src.c) and _in_bounds((h, w), r2, src.c)):
                continue
            winner = winner_map.get((r2, src.c))
            if out[r1, src.c] == BG and winner is not None and winner.kind == "H":
                return True
    return False


def _solid_diagonal_visible(out: np.ndarray, meta: dict) -> bool:
    winner_map = meta["winner_map"]
    h, w = out.shape
    for src in meta["sources"]:
        if src.kind != "V":
            continue
        for dr, dc in DIAGONALS:
            p1 = (src.r + dr, src.c + dc)
            p2 = (src.r + 2 * dr, src.c + 2 * dc)
            if not (_in_bounds((h, w), *p1) and _in_bounds((h, w), *p2)):
                continue
            w1 = winner_map.get(p1)
            w2 = winner_map.get(p2)
            if w1 is None or w2 is None:
                continue
            if w1.kind == "V" and w2.kind == "V" and out[p1] != BG and out[p2] != BG:
                return True
    return False


def _semantic_checks(inp: np.ndarray, profile: str) -> bool:
    try:
        dominoes = _extract_dominoes(inp)
        out, meta = _solve_from_dominoes(dominoes, inp.shape, return_meta=True)
    except AssertionError:
        return False

    if not np.array_equal(main(inp), out):
        return False

    h_count = sum(domino.orientation == "H" for domino in dominoes)
    v_count = sum(domino.orientation == "V" for domino in dominoes)
    if not (1 <= h_count < len(dominoes) and 1 <= v_count < len(dominoes)):
        return False

    density = float(np.count_nonzero(out)) / out.size
    if not (0.12 <= density <= 0.60):
        return False
    if not _has_cross_conflict(meta):
        return False
    if not _has_same_class_conflict(meta):
        return False
    if not _periodic_visible(out, meta):
        return False
    if not _solid_diagonal_visible(out, meta):
        return False
    if profile == "row_tie" and not _has_row_tie(meta):
        return False
    if profile == "col_tie" and not _has_col_tie(meta):
        return False
    return True


def _build_profile(shape: tuple[int, int], profile: str) -> list[Placement] | None:
    if profile == "row_tie":
        return _build_row_tie(shape)
    return _build_col_tie(shape)


def _attempt() -> np.ndarray | None:
    shape = (random.randint(12, 16), random.randint(12, 16))
    profile = random.choice(("row_tie", "col_tie"))
    placements = _build_profile(shape, profile)
    if placements is None:
        return None

    max_total = 5 if min(shape) >= 13 else 4
    target_count = random.randint(3, max_total)
    if not _add_extras(shape, placements, target_count):
        return None

    inp = _render_input(shape, placements)
    return inp if _semantic_checks(inp, profile) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid orientation-ray puzzle input")
