from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREEN = 3
MIRROR_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
ORIENTATIONS = ("h", "v", "d1", "d2")
GEN_RETRIES = 400
GEN_OVERLAP_PROB = 0.3
GEN_MIN_SIZE = 12
GEN_MAX_SIZE = 24
GEN_MIRROR_COUNTS = (2, 3, 4)


def _mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


SOURCE_SHAPES = (
    _mask(("#.", "##")),
    _mask(("#.", "#.", "##")),
    _mask(("##.", ".##")),
    _mask(("##", "##", "#.")),
    _mask(("##.", ".##", "..#")),
)


DELTAS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DELTAS8 = DELTAS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 2
    deltas = DELTAS4 if connectivity == 4 else DELTAS8
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    out: list[np.ndarray] = []

    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in deltas:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        out.append(np.array(coords, dtype=int))
    return out


def _transform_mask(mask: np.ndarray, op: int) -> np.ndarray:
    if op < 4:
        return np.rot90(mask, k=op)
    return np.fliplr(np.rot90(mask, k=op - 4))


def _coords_from_mask(mask: np.ndarray, top: int = 0, left: int = 0) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.empty((0, 2), dtype=int)
    coords = coords.astype(int)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _mirror_from_cells(coords: np.ndarray, color: int | None = None) -> dict:
    pts = sorted((int(r), int(c)) for r, c in np.asarray(coords, dtype=int))
    assert len(pts) == 2, "each mirror must contain two cells"
    (r0, c0), (r1, c1) = pts
    dr = r1 - r0
    dc = c1 - c0

    if dr == 0 and abs(dc) == 1:
        kind = "h"
        axis = r0
    elif dc == 0 and abs(dr) == 1:
        kind = "v"
        axis = c0
    elif dr == dc and abs(dr) == 1:
        kind = "d1"
        axis = r0 - c0
    elif dr == -dc and abs(dr) == 1:
        kind = "d2"
        axis = r0 + c0
    else:
        raise AssertionError("mirror cells must be adjacent horizontally, vertically, or diagonally")

    return {
        "kind": kind,
        "axis": int(axis),
        "cells": tuple(pts),
        "cell_set": frozenset(pts),
        "color": None if color is None else int(color),
    }


def _signed_axis_values(coords: np.ndarray, mirror: dict) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    kind = mirror["kind"]
    axis = int(mirror["axis"])
    if kind == "h":
        return arr[:, 0] - axis
    if kind == "v":
        return arr[:, 1] - axis
    if kind == "d1":
        return arr[:, 0] - arr[:, 1] - axis
    return arr[:, 0] + arr[:, 1] - axis


def _distance_key(coords: np.ndarray, mirror: dict) -> int:
    signed = _signed_axis_values(coords, mirror).astype(int)
    if mirror["kind"] in ("h", "v"):
        return int(np.min(2 * signed * signed))
    return int(np.min(signed * signed))


def _reflect_coords(coords: np.ndarray, mirror: dict) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    axis = int(mirror["axis"])
    kind = mirror["kind"]
    out = np.empty_like(arr)

    if kind == "h":
        out[:, 0] = 2 * axis - arr[:, 0]
        out[:, 1] = arr[:, 1]
    elif kind == "v":
        out[:, 0] = arr[:, 0]
        out[:, 1] = 2 * axis - arr[:, 1]
    elif kind == "d1":
        out[:, 0] = arr[:, 1] + axis
        out[:, 1] = arr[:, 0] - axis
    else:
        out[:, 0] = axis - arr[:, 1]
        out[:, 1] = axis - arr[:, 0]
    return out


def _isolated_mirror(candidate_cells: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    own = set(candidate_cells)
    for r, c in candidate_cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied and (rr, cc) not in own:
                    return False
    return True


def _source_and_mirrors(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BLACK), "expected black background"

    green_components = _components(g == GREEN, connectivity=8)
    assert len(green_components) == 1, "expected exactly one green source component"
    source = green_components[0]
    assert len(source) >= 3, "expected a small multi-pixel source component"

    mirrors: list[dict] = []
    covered = set(map(tuple, source))
    for color in sorted(set(map(int, np.unique(g))) - {BLACK, GREEN}):
        for comp in _components(g == color, connectivity=8):
            mirror = _mirror_from_cells(comp, color=color)
            mirrors.append(mirror)
            covered.update(mirror["cells"])

    assert mirrors, "expected at least one mirror"
    assert len(covered) == int(np.count_nonzero(g != BLACK)), "unexpected extra foreground pixels"
    return source, mirrors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the green source across each two-pixel mirror axis and recolor every copy with the mirror's color.

    Input structure:
    - The background is black.
    - Exactly one green connected source pattern appears as a small non-symmetric shape.
    - Every other foreground object is a monochromatic two-pixel mirror, arranged
      horizontally, vertically, or diagonally, with each arrangement defining a reflection axis.

    Rule:
    - Keep the source pattern and every mirror in place.
    - For each mirror, reflect the full green source across the axis passing through the
      two mirror pixels: horizontal pairs give horizontal axes, vertical pairs give vertical axes,
      and diagonal pairs give 45-degree or 135-degree axes.
    - Draw the reflected copy in the mirror's color. If reflected copies overlap, a copy
      from a mirror farther from the source overrides a copy from a nearer mirror.

    Key insight:
    - The mirror object does not mark a destination; its two-pixel orientation is the code
      for the reflection axis, and the mirror's own color becomes the reflected copy's color.
    """
    g = np.asarray(grid, dtype=int)
    source, mirrors = _source_and_mirrors(g)

    height, width = g.shape
    out = g.copy()
    for mirror in sorted(
        mirrors,
        key=lambda item: (int(_distance_key(source, item)), item["kind"], item["cells"]),
    ):
        reflected = _reflect_coords(source, mirror)
        assert reflected.size > 0
        assert reflected[:, 0].min() >= 0 and reflected[:, 0].max() < height
        assert reflected[:, 1].min() >= 0 and reflected[:, 1].max() < width
        out[reflected[:, 0], reflected[:, 1]] = int(mirror["color"])

    out[source[:, 0], source[:, 1]] = GREEN
    for mirror in mirrors:
        for r, c in mirror["cells"]:
            out[r, c] = int(mirror["color"])
    return out


def _random_source_mask() -> np.ndarray:
    return _transform_mask(random.choice(SOURCE_SHAPES), random.randrange(8))


def _enumerate_candidates(source: np.ndarray, shape: tuple[int, int]) -> dict[str, list[dict]]:
    height, width = map(int, shape)
    source_set = set(map(tuple, source))
    candidates: dict[str, list[dict]] = {kind: [] for kind in ORIENTATIONS}

    for r in range(height):
        for c in range(width):
            mirror_cells_by_kind = []
            if c + 1 < width:
                mirror_cells_by_kind.append(("h", ((r, c), (r, c + 1))))
            if r + 1 < height:
                mirror_cells_by_kind.append(("v", ((r, c), (r + 1, c))))
            if r + 1 < height and c + 1 < width:
                mirror_cells_by_kind.append(("d1", ((r, c), (r + 1, c + 1))))
            if r + 1 < height and c - 1 >= 0:
                mirror_cells_by_kind.append(("d2", ((r, c), (r + 1, c - 1))))

            for kind, cells in mirror_cells_by_kind:
                mirror = _mirror_from_cells(np.array(cells, dtype=int))
                if mirror["kind"] != kind:
                    continue
                if not _isolated_mirror(mirror["cells"], source_set):
                    continue

                signed = _signed_axis_values(source, mirror)
                if np.any(signed == 0) or not (np.all(signed > 0) or np.all(signed < 0)):
                    continue

                reflected = _reflect_coords(source, mirror)
                if (
                    reflected[:, 0].min() < 0
                    or reflected[:, 0].max() >= height
                    or reflected[:, 1].min() < 0
                    or reflected[:, 1].max() >= width
                ):
                    continue

                copy_set = frozenset((int(rr), int(cc)) for rr, cc in reflected)
                if copy_set & source_set:
                    continue
                if copy_set & mirror["cell_set"]:
                    continue

                mirror["copy_coords"] = reflected
                mirror["copy_set"] = copy_set
                mirror["distance"] = int(_distance_key(source, mirror))
                candidates[kind].append(mirror)
    return candidates


def _pick_one(
    pool: list[dict],
    used_input: set[tuple[int, int]],
    used_copy: set[tuple[int, int]],
    used_distances: set[int],
    allow_copy_overlap: bool,
) -> dict | None:
    choices = pool[:]
    random.shuffle(choices)
    for cand in choices:
        if cand["distance"] in used_distances:
            continue
        if not _isolated_mirror(cand["cells"], used_input):
            continue
        if cand["cell_set"] & used_copy:
            continue
        if cand["copy_set"] & used_input:
            continue
        if not allow_copy_overlap and cand["copy_set"] & used_copy:
            continue
        return cand
    return None


def _pick_overlap_pair(
    candidates: dict[str, list[dict]],
    kinds: list[str],
    source_set: set[tuple[int, int]],
) -> tuple[list[dict], list[str]] | None:
    kind_pairs = list(itertools.combinations(kinds, 2))
    random.shuffle(kind_pairs)
    for kind_a, kind_b in kind_pairs:
        pool_a = candidates[kind_a][:]
        pool_b = candidates[kind_b][:]
        random.shuffle(pool_a)
        random.shuffle(pool_b)
        for first in pool_a:
            used_input = set(source_set) | set(first["cell_set"])
            used_copy = set(first["copy_set"])
            for second in pool_b:
                if second["distance"] == first["distance"]:
                    continue
                if not _isolated_mirror(second["cells"], used_input):
                    continue
                if second["cell_set"] & used_copy:
                    continue
                if second["copy_set"] & used_input:
                    continue
                if not (second["copy_set"] & used_copy):
                    continue
                remaining = [kind for kind in kinds if kind not in {kind_a, kind_b}]
                return [first, second], remaining
    return None


def _select_mirrors(
    candidates: dict[str, list[dict]],
    kinds: list[str],
    source_set: set[tuple[int, int]],
    force_overlap: bool,
) -> list[dict] | None:
    if any(not candidates[kind] for kind in kinds):
        return None

    selected: list[dict] = []
    used_input = set(source_set)
    used_copy: set[tuple[int, int]] = set()
    used_distances: set[int] = set()
    remaining = kinds[:]

    if force_overlap and len(kinds) >= 2:
        pair_result = _pick_overlap_pair(candidates, kinds, source_set)
        if pair_result is None:
            return None
        pair, remaining = pair_result
        for cand in pair:
            selected.append(cand)
            used_input.update(cand["cell_set"])
            used_copy.update(cand["copy_set"])
            used_distances.add(int(cand["distance"]))

    random.shuffle(remaining)
    for kind in remaining:
        cand = _pick_one(
            candidates[kind],
            used_input=used_input,
            used_copy=used_copy,
            used_distances=used_distances,
            allow_copy_overlap=False,
        )
        if cand is None:
            return None
        selected.append(cand)
        used_input.update(cand["cell_set"])
        used_copy.update(cand["copy_set"])
        used_distances.add(int(cand["distance"]))
    return selected


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
    width = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
    source_mask = _random_source_mask()
    src_h, src_w = map(int, source_mask.shape)

    row_margin = 2
    col_margin = 2
    row_lo = row_margin
    row_hi = height - src_h - row_margin
    col_lo = col_margin
    col_hi = width - src_w - col_margin
    if row_lo > row_hi or col_lo > col_hi:
        return None

    top = random.randint(row_lo, row_hi)
    left = random.randint(col_lo, col_hi)
    source = _coords_from_mask(source_mask, top=top, left=left)
    source_set = set(map(tuple, source))

    candidates = _enumerate_candidates(source, (height, width))
    mirror_count = random.choice(GEN_MIRROR_COUNTS)
    kinds = random.sample(list(ORIENTATIONS), mirror_count)
    selected = _select_mirrors(
        candidates,
        kinds,
        source_set=source_set,
        force_overlap=(mirror_count >= 2 and random.random() < GEN_OVERLAP_PROB),
    )
    if selected is None:
        return None

    colors = random.sample(MIRROR_COLORS, len(selected))
    grid = np.full((height, width), BLACK, dtype=int)
    grid[source[:, 0], source[:, 1]] = GREEN

    for cand, color in zip(selected, colors):
        for r, c in cand["cells"]:
            grid[r, c] = int(color)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.count_nonzero(out != BLACK) > np.count_nonzero(grid != BLACK)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-reflection input")
