from __future__ import annotations

from collections import deque
from fractions import Fraction
import random

import numpy as np


BG = 0
GEN_LATTICE = 8
TILE = 5
OUTER_BORDER = 1
ANCHOR_OFFSET = (1, 1)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARKER_COLORS = (2, 3, 4, 6, 7, 9)
GENERATION_RETRIES = 300

# A 15-cell enclosed chamber with a recognizable northwest notch.
CAVITY_SHAPE = frozenset(
    {
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (4, 3),
    }
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _lattice_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid[grid != BG], return_counts=True)
    assert vals.size >= 1, "expected a non-black lattice color"
    return int(vals[int(np.argmax(counts))])


def _matches_anchor_template(
    grid: np.ndarray,
    pair_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
    lattice: int,
) -> bool:
    h, w = grid.shape
    r0 = anchor[0] - ANCHOR_OFFSET[0]
    c0 = anchor[1] - ANCHOR_OFFSET[1]
    if r0 < 0 or c0 < 0 or r0 + TILE > h or c0 + TILE > w:
        return False

    for dr in range(TILE):
        for dc in range(TILE):
            rr = r0 + dr
            cc = c0 + dc
            val = BG if (rr, cc) in pair_cells else int(grid[rr, cc])
            expected = BG if (dr, dc) in CAVITY_SHAPE else lattice
            if val != expected:
                return False
    return True


def _parse_markers(grid: np.ndarray, lattice: int, *, flip_dirs: bool = False) -> list[dict]:
    markers: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in (BG, lattice)):
        for comp in _components(grid == color):
            assert len(comp) == 2, "each marker-direction indicator must be a 2-cell component"
            pair_cells = {tuple(map(int, cell)) for cell in comp.tolist()}
            anchors = [
                cell
                for cell in pair_cells
                if _matches_anchor_template(grid, pair_cells, cell, lattice)
            ]
            assert len(anchors) == 1, "each marker pair must contain one valid anchor"
            anchor = anchors[0]
            indicator = next(cell for cell in pair_cells if cell != anchor)
            dr = indicator[0] - anchor[0]
            dc = indicator[1] - anchor[1]
            assert (dr, dc) in DIRS4, "indicator must be 4-neighbor adjacent to anchor"
            if flip_dirs:
                dr, dc = -dr, -dc
            markers.append(
                {
                    "color": color,
                    "anchor": anchor,
                    "dir": (dr, dc),
                    "cells": tuple(sorted(pair_cells)),
                }
            )

    markers.sort(key=lambda m: (m["anchor"][0], m["anchor"][1], m["color"]))
    assert markers, "expected at least one marker pair"
    return markers


def _erase_markers(grid: np.ndarray, markers: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for marker in markers:
        for r, c in marker["cells"]:
            out[r, c] = BG
    return out


def _find_cavities(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    mask = grid == BG
    comp_id = -np.ones(grid.shape, dtype=int)
    cavities: list[dict] = []
    for cid, coords in enumerate(_components(mask)):
        assert not np.any(coords[:, 0] == 0) and not np.any(coords[:, 0] == grid.shape[0] - 1)
        assert not np.any(coords[:, 1] == 0) and not np.any(coords[:, 1] == grid.shape[1] - 1)
        comp_id[coords[:, 0], coords[:, 1]] = cid
        cavities.append(
            {
                "coords": coords,
                "area": int(len(coords)),
                "sum_r": int(coords[:, 0].sum()),
                "sum_c": int(coords[:, 1].sum()),
            }
        )
    assert cavities, "expected enclosed black cavities after erasing markers"
    return cavities, comp_id


def _distance_to_cavity(anchor: tuple[int, int], cavity: dict) -> Fraction:
    area = int(cavity["area"])
    ar, ac = anchor
    return (
        Fraction(abs(area * int(ar) - int(cavity["sum_r"])), area)
        + Fraction(abs(area * int(ac) - int(cavity["sum_c"])), area)
    )


def _tile_anchor(tile_r: int, tile_c: int) -> tuple[int, int]:
    return OUTER_BORDER + tile_r * TILE + ANCHOR_OFFSET[0], OUTER_BORDER + tile_c * TILE + ANCHOR_OFFSET[1]


def _tile_hits(marker: dict, n_tile_rows: int, n_tile_cols: int) -> set[tuple[int, int]]:
    tr, tc = marker["tile"]
    dr, dc = marker["dir"]
    if (dr, dc) == (0, 1):
        return {(tr, cc) for cc in range(tc, n_tile_cols)}
    if (dr, dc) == (0, -1):
        return {(tr, cc) for cc in range(0, tc + 1)}
    if (dr, dc) == (1, 0):
        return {(rr, tc) for rr in range(tr, n_tile_rows)}
    if (dr, dc) == (-1, 0):
        return {(rr, tc) for rr in range(0, tr + 1)}
    raise AssertionError("unexpected marker direction")


def _marker_hits(marker: dict, comp_id: np.ndarray, *, local_only: bool = False) -> set[int]:
    r, c = marker["anchor"]
    if local_only:
        cid = int(comp_id[r, c])
        return {cid} if cid >= 0 else set()

    dr, dc = marker["dir"]
    h, w = comp_id.shape
    hits: set[int] = set()
    while 0 <= r < h and 0 <= c < w:
        cid = int(comp_id[r, c])
        if cid >= 0:
            hits.add(cid)
        r += dr
        c += dc
    return hits


def _choose_owner(
    candidate_ids: list[int],
    markers: list[dict],
    cavity: dict,
    *,
    conflict_policy: str,
    reverse_tie: bool,
) -> int:
    def tie_key(idx: int) -> tuple[int, int]:
        r, c = markers[idx]["anchor"]
        return (-r, -c) if reverse_tie else (r, c)

    if conflict_policy == "lexicographic_only":
        return min(candidate_ids, key=tie_key)
    if conflict_policy == "nearest":
        return min(candidate_ids, key=lambda idx: (_distance_to_cavity(markers[idx]["anchor"], cavity), tie_key(idx)))
    if conflict_policy == "farthest":
        return min(candidate_ids, key=lambda idx: (-_distance_to_cavity(markers[idx]["anchor"], cavity), tie_key(idx)))
    raise AssertionError(f"unknown conflict policy: {conflict_policy}")


def _solve_core(
    grid: np.ndarray,
    *,
    flip_dirs: bool = False,
    local_only: bool = False,
    conflict_policy: str = "nearest",
    reverse_tie: bool = False,
    return_meta: bool = False,
):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    lattice = _lattice_color(g)
    markers = _parse_markers(g, lattice, flip_dirs=flip_dirs)
    erased = _erase_markers(g, markers)
    cavities, comp_id = _find_cavities(erased)

    cavity_candidates: dict[int, list[int]] = {cid: [] for cid in range(len(cavities))}
    for idx, marker in enumerate(markers):
        hits = _marker_hits(marker, comp_id, local_only=local_only)
        marker["hits"] = hits
        for cid in hits:
            cavity_candidates[cid].append(idx)

    out = np.array(erased, copy=True)
    has_conflict = False
    has_tie = False
    owned_count = 0
    for cid, candidate_ids in cavity_candidates.items():
        if len(candidate_ids) > 1:
            has_conflict = True
            dists = sorted(_distance_to_cavity(markers[idx]["anchor"], cavities[cid]) for idx in candidate_ids)
            if len(dists) >= 2 and dists[0] == dists[1]:
                has_tie = True
        if not candidate_ids:
            continue
        owned_count += 1
        owner = _choose_owner(
            candidate_ids,
            markers,
            cavities[cid],
            conflict_policy=conflict_policy,
            reverse_tie=reverse_tie,
        )
        coords = cavities[cid]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(markers[owner]["color"])

    if not return_meta:
        return out

    meta = {
        "lattice": lattice,
        "markers": markers,
        "cavities": cavities,
        "cavity_candidates": cavity_candidates,
        "has_conflict": has_conflict,
        "has_tie": has_tie,
        "max_candidates": max((len(ids) for ids in cavity_candidates.values()), default=0),
        "owned_count": owned_count,
        "unhit_count": len(cavities) - owned_count,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot color-coded rays through a lattice and fill every chamber the ray traverses.

    Input structure:
    - A cyan lattice forms a periodic grid of enclosed black chambers with a fixed northwest notch pattern.
    - Each active chamber contains one colored anchor pixel and one same-colored 4-neighbor direction pixel.
    - The direction pixel's position relative to the anchor encodes one of the four cardinal ray directions.

    Rule:
    - Erase every colored 2-pixel marker pair back to black.
    - From each anchor position, extend a ray in the encoded cardinal direction all the way to the grid edge.
    - Every enclosed chamber whose black cells are touched by that ray is filled with the anchor color.
    - If multiple rays hit the same chamber, keep the color of the marker with smaller Manhattan distance to the chamber center; break exact ties by the marker position closest to the top-left.

    Key insight:
    - The colored pair is not paint for one nearby chamber: it restores to a black notch, reveals the ray origin and direction, and the ray's full geometric path selects which distant chambers get filled.
    """
    return _solve_core(grid)


def _base_grid(n_tile_rows: int, n_tile_cols: int) -> np.ndarray:
    h = OUTER_BORDER * 2 + n_tile_rows * TILE
    w = OUTER_BORDER * 2 + n_tile_cols * TILE
    grid = np.full((h, w), GEN_LATTICE, dtype=int)
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            r0 = OUTER_BORDER + tr * TILE
            c0 = OUTER_BORDER + tc * TILE
            for dr, dc in CAVITY_SHAPE:
                grid[r0 + dr, c0 + dc] = BG
    return grid


def _render_input(n_tile_rows: int, n_tile_cols: int, markers: list[dict]) -> np.ndarray:
    grid = _base_grid(n_tile_rows, n_tile_cols)
    for marker in markers:
        ar, ac = _tile_anchor(*marker["tile"])
        dr, dc = marker["dir"]
        color = int(marker["color"])
        grid[ar, ac] = color
        grid[ar + dr, ac + dc] = color
    return grid


def _horizontal_candidates(n_tile_rows: int, n_tile_cols: int) -> list[dict]:
    out: list[dict] = []
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols - 1):
            out.append({"tile": (tr, tc), "dir": (0, 1)})
        for tc in range(1, n_tile_cols):
            out.append({"tile": (tr, tc), "dir": (0, -1)})
    return out


def _vertical_candidates(n_tile_rows: int, n_tile_cols: int) -> list[dict]:
    out: list[dict] = []
    for tc in range(n_tile_cols):
        for tr in range(n_tile_rows - 1):
            out.append({"tile": (tr, tc), "dir": (1, 0)})
        for tr in range(1, n_tile_rows):
            out.append({"tile": (tr, tc), "dir": (-1, 0)})
    return out


def _assign_colors(markers: list[dict]) -> list[dict]:
    colors = random.sample(MARKER_COLORS, k=len(markers))
    out: list[dict] = []
    for marker, color in zip(markers, colors):
        clone = dict(marker)
        clone["color"] = int(color)
        out.append(clone)
    return out


def _sample_no_conflict_markers(n_tile_rows: int, n_tile_cols: int) -> list[dict] | None:
    family = random.choice(("horizontal", "vertical"))
    markers: list[dict] = []
    used_tiles: set[tuple[int, int]] = set()

    if family == "horizontal":
        rows = list(range(n_tile_rows))
        random.shuffle(rows)
        count = random.randint(2, min(4, n_tile_rows))
        for tr in rows[:count]:
            choices = [m for m in _horizontal_candidates(n_tile_rows, n_tile_cols) if m["tile"][0] == tr]
            choices = [m for m in choices if m["tile"] not in used_tiles]
            if not choices:
                return None
            chosen = dict(random.choice(choices))
            markers.append(chosen)
            used_tiles.add(chosen["tile"])
    else:
        cols = list(range(n_tile_cols))
        random.shuffle(cols)
        count = random.randint(2, min(4, n_tile_cols))
        for tc in cols[:count]:
            choices = [m for m in _vertical_candidates(n_tile_rows, n_tile_cols) if m["tile"][1] == tc]
            choices = [m for m in choices if m["tile"] not in used_tiles]
            if not choices:
                return None
            chosen = dict(random.choice(choices))
            markers.append(chosen)
            used_tiles.add(chosen["tile"])
    return markers


def _sample_tie_conflict_markers(n_tile_rows: int, n_tile_cols: int) -> list[dict] | None:
    candidates: list[list[dict]] = []
    max_d = min(n_tile_rows - 1, n_tile_cols - 1)
    for d in range(1, max_d + 1):
        for target_r in range(d, n_tile_rows):
            for target_c in range(d, n_tile_cols):
                candidates.append(
                    [
                        {"tile": (target_r, target_c - d), "dir": (0, 1)},
                        {"tile": (target_r - d, target_c), "dir": (1, 0)},
                    ]
                )
        for target_r in range(0, n_tile_rows - d):
            for target_c in range(0, n_tile_cols - d):
                candidates.append(
                    [
                        {"tile": (target_r, target_c + d), "dir": (0, -1)},
                        {"tile": (target_r + d, target_c), "dir": (-1, 0)},
                    ]
                )
    if not candidates:
        return None
    return [dict(marker) for marker in random.choice(candidates)]


def _sample_distance_conflict_markers(n_tile_rows: int, n_tile_cols: int) -> list[dict] | None:
    candidates: list[list[dict]] = []
    for d_h in range(1, n_tile_cols):
        for d_v in range(1, n_tile_rows):
            if d_h == d_v:
                continue
            for target_r in range(d_v, n_tile_rows):
                for target_c in range(d_h, n_tile_cols):
                    candidates.append(
                        [
                            {"tile": (target_r, target_c - d_h), "dir": (0, 1)},
                            {"tile": (target_r - d_v, target_c), "dir": (1, 0)},
                        ]
                    )
            for target_r in range(0, n_tile_rows - d_v):
                for target_c in range(0, n_tile_cols - d_h):
                    candidates.append(
                        [
                            {"tile": (target_r, target_c + d_h), "dir": (0, -1)},
                            {"tile": (target_r + d_v, target_c), "dir": (-1, 0)},
                        ]
                    )
    if not candidates:
        return None
    return [dict(marker) for marker in random.choice(candidates)]


def _semantic_ok(grid: np.ndarray, mode: str) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        out, meta = _solve_core(grid, return_meta=True)
    except Exception:
        return False

    if len(meta["markers"]) < 2:
        return False
    if meta["max_candidates"] > 2:
        return False
    if meta["unhit_count"] < 1:
        return False
    if not any(len(marker["hits"]) >= 2 for marker in meta["markers"]):
        return False

    if mode == "no_conflict" and meta["has_conflict"]:
        return False
    if mode == "distance_conflict" and (not meta["has_conflict"] or meta["has_tie"]):
        return False
    if mode == "tie_conflict" and not meta["has_tie"]:
        return False

    if not np.array_equal(out, main(grid)):
        return False

    try:
        wrong_local = _solve_core(grid, local_only=True)
        wrong_flipped = _solve_core(grid, flip_dirs=True)
    except Exception:
        return False
    if np.array_equal(out, wrong_local):
        return False
    if np.array_equal(out, wrong_flipped):
        return False

    if meta["has_conflict"]:
        if meta["has_tie"]:
            wrong_tie = _solve_core(grid, reverse_tie=True)
            if np.array_equal(out, wrong_tie):
                return False
        else:
            wrong_farthest = _solve_core(grid, conflict_policy="farthest")
            if np.array_equal(out, wrong_farthest):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choices(
        ["no_conflict", "distance_conflict", "tie_conflict"],
        weights=[2, 2, 3],
        k=1,
    )[0]

    if mode == "no_conflict":
        n_tile_rows = random.randint(2, 5)
        n_tile_cols = random.randint(2, 5)
        raw_markers = _sample_no_conflict_markers(n_tile_rows, n_tile_cols)
    elif mode == "distance_conflict":
        n_tile_rows = random.randint(3, 5)
        n_tile_cols = random.randint(3, 5)
        raw_markers = _sample_distance_conflict_markers(n_tile_rows, n_tile_cols)
    elif mode == "tie_conflict":
        n_tile_rows = random.randint(2, 5)
        n_tile_cols = random.randint(2, 5)
        raw_markers = _sample_tie_conflict_markers(n_tile_rows, n_tile_cols)
    else:
        raise AssertionError(f"unexpected generation mode: {mode}")

    if raw_markers is None:
        return None

    markers = _assign_colors(raw_markers)
    random.shuffle(markers)
    grid = _render_input(n_tile_rows, n_tile_cols, markers)

    if not _semantic_ok(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-ray chamber input")
