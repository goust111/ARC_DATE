from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BACKGROUND = 0
ANCHOR = 1
LABEL_COLORS = (2, 3)
SKELETON = 5
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
TRANSFORMS = tuple((rot, flip) for rot in range(4) for flip in (False, True))

GRID_MIN = 17
GRID_MAX = 22
MOTIF_COUNT = 4
GEN_ATTEMPTS = 600
SHAPE_ATTEMPTS = 900
PLACEMENT_ATTEMPTS = 180


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((int(r + dr), int(c + dc)))
    return out


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _apply_transform(offset: tuple[int, int], transform: tuple[int, bool]) -> tuple[int, int]:
    r, c = map(int, offset)
    rot, flip = transform
    for _ in range(int(rot)):
        r, c = -c, r
    if flip:
        c = -c
    return int(r), int(c)


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    r = int(start[0] + dr)
    c = int(start[1] + dc)
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _parse_motifs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    allowed = {BACKGROUND, ANCHOR, SKELETON, *LABEL_COLORS}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors"

    motifs: list[dict] = []
    for coords in _components8(g != BACKGROUND):
        anchor_cells = [(r, c) for r, c in coords if int(g[r, c]) == ANCHOR]
        gray_cells = [(r, c) for r, c in coords if int(g[r, c]) == SKELETON]
        labels = [(r, c, int(g[r, c])) for r, c in coords if int(g[r, c]) in LABEL_COLORS]

        assert len(anchor_cells) == 1, "each motif must contain exactly one anchor"
        anchor = anchor_cells[0]
        assert 4 <= len(gray_cells) + 1 <= 7, "skeleton size must be 4..7 including anchor"

        gray_offsets = frozenset((int(r - anchor[0]), int(c - anchor[1])) for r, c in gray_cells)
        label_infos = []
        for r, c, color in labels:
            dr = int(r - anchor[0])
            dc = int(c - anchor[1])
            assert abs(dr) == 1 and abs(dc) == 1, "labels must be diagonal neighbors of the anchor"
            label_infos.append(
                {
                    "pos": (int(r), int(c)),
                    "offset": (dr, dc),
                    "color": int(color),
                }
            )

        motifs.append(
            {
                "anchor": (int(anchor[0]), int(anchor[1])),
                "gray_offsets": gray_offsets,
                "labels": label_infos,
                "coords": set((int(r), int(c)) for r, c in coords),
            }
        )
    return motifs


def _completed_motifs(grid: np.ndarray) -> list[dict]:
    motifs = _parse_motifs(grid)
    assert len(motifs) == MOTIF_COUNT, "expected exactly four motifs"

    source_indices = [i for i, motif in enumerate(motifs) if motif["labels"]]
    assert len(source_indices) == 1, "expected exactly one labeled source motif"
    source = motifs[source_indices[0]]
    source_labels = [(tuple(label["offset"]), int(label["color"])) for label in source["labels"]]
    assert 1 <= len(source_labels) <= 2, "source must have 1 or 2 labels"

    completed: list[dict] = []
    for motif in motifs:
        if motif["labels"]:
            labels = [
                {
                    "pos": tuple(label["pos"]),
                    "offset": tuple(label["offset"]),
                    "color": int(label["color"]),
                }
                for label in motif["labels"]
            ]
        else:
            matches = []
            for transform in TRANSFORMS:
                transformed = frozenset(_apply_transform(offset, transform) for offset in source["gray_offsets"])
                if transformed == motif["gray_offsets"]:
                    matches.append(transform)
            assert len(matches) == 1, "target motif must have a unique dihedral alignment"

            labels = []
            transform = matches[0]
            ar, ac = motif["anchor"]
            for offset, color in source_labels:
                dr, dc = _apply_transform(offset, transform)
                rr = ar + dr
                cc = ac + dc
                assert _in_bounds(grid.shape, rr, cc), "transferred label must stay in bounds"
                labels.append({"pos": (int(rr), int(cc)), "offset": (int(dr), int(dc)), "color": int(color)})

        completed.append({"anchor": motif["anchor"], "labels": labels})
    return completed


def _render_output_from_motifs(shape: tuple[int, int], motifs: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    ordered = sorted(motifs, key=lambda motif: (motif["anchor"][0], motif["anchor"][1]))
    for motif in ordered:
        labels = sorted(motif["labels"], key=lambda label: (label["pos"][0], label["pos"][1], label["color"]))
        for label in labels:
            for r, c in _ray_cells(label["pos"], label["offset"], shape):
                out[r, c] = int(label["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Transfer the source motif's diagonal labels through dihedral skeleton alignment, then emit only their outward diagonal rays.

    Input structure:
    - A black background contains exactly four 8-connected motifs made from gray skeleton pixels plus one blue anchor pixel each.
    - All motifs share the same skeleton shape up to rotation/reflection, and the blue anchor belongs to that skeleton.
    - Exactly one motif also has one or two red/green label pixels, each diagonally adjacent to its anchor but not part of the skeleton.

    Rule:
    - Use the labeled motif as the source template and, for every unlabeled motif, find the unique dihedral transform that maps the source gray skeleton around its blue anchor onto the target gray skeleton around its blue anchor.
    - Transfer every source label through that transform to the corresponding diagonal position next to the target anchor, preserving the label color.
    - For every original or transferred label, emit a diagonal ray of the same color starting one more step farther from the anchor in that label's diagonal direction, continuing to the grid edge.
    - Erase the gray skeletons, blue anchors, and label pixels themselves, keeping only the rays; if rays overlap, later drawing overwrites earlier drawing, with motifs ordered by anchor position and labels within a motif ordered row-major.

    Key insight:
    - The skeleton is a disposable orientation carrier: only by aligning skeleton plus anchor can you recover where each label lands, and that recovered label position determines the diagonal ray direction.
    """
    motifs = _completed_motifs(np.asarray(grid, dtype=int))
    return _render_output_from_motifs(tuple(map(int, grid.shape)), motifs)


def _anchor_ok(shape_cells: set[tuple[int, int]]) -> bool:
    neighbors = [(r, c) for r, c in shape_cells if (r, c) != (0, 0) and max(abs(r), abs(c)) == 1]
    if len(neighbors) == 1:
        return True
    if len(neighbors) != 2:
        return False
    a = neighbors[0]
    b = neighbors[1]
    return not (a[0] == -b[0] and a[1] == -b[1])


def _shape_variant_count(gray_offsets: set[tuple[int, int]]) -> int:
    variants = {frozenset(_apply_transform(offset, transform) for offset in gray_offsets) for transform in TRANSFORMS}
    return len(variants)


def _clean_label_offsets(shape_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    options = []
    for dr, dc in DIAGONALS:
        if (dr, dc) in shape_cells:
            continue
        touching = [(r, c) for r, c in shape_cells if max(abs(r - dr), abs(c - dc)) == 1]
        if touching == [(0, 0)] or touching == [(0, 0),]:
            options.append((int(dr), int(dc)))
    return options


def _random_shape_cells(total_cells: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    limit = 3
    while len(cells) < int(total_cells):
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) not in cells and max(abs(nr), abs(nc)) <= limit:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            break
        cells.add(random.choice(list(frontier)))
    return cells


def _choose_base_shape() -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], int]]]:
    for _ in range(SHAPE_ATTEMPTS):
        total_cells = random.randint(4, 7)
        shape_cells = _random_shape_cells(total_cells)
        if len(shape_cells) != total_cells:
            continue
        gray_offsets = shape_cells - {(0, 0)}
        if not _anchor_ok(shape_cells):
            continue
        if _shape_variant_count(gray_offsets) != len(TRANSFORMS):
            continue

        label_offsets = _clean_label_offsets(shape_cells)
        if len(label_offsets) < 2:
            continue

        if random.random() < 0.55:
            adjacent_pairs = [
                pair
                for pair in combinations(label_offsets, 2)
                if pair[0][0] == pair[1][0] or pair[0][1] == pair[1][1]
            ]
            if adjacent_pairs:
                pair = list(random.choice(adjacent_pairs))
                colors = list(LABEL_COLORS)
                random.shuffle(colors)
                return gray_offsets, [(pair[0], colors[0]), (pair[1], colors[1])]

        offset = random.choice(label_offsets)
        color = random.choice(LABEL_COLORS)
        return gray_offsets, [(offset, color)]
    raise RuntimeError("failed to build a non-degenerate base skeleton")


def _signature_for_transform(labels: list[tuple[tuple[int, int], int]], transform: tuple[int, bool]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_apply_transform(offset, transform) for offset, _color in labels))


def _choose_transform_subset(labels: list[tuple[tuple[int, int], int]]) -> list[tuple[int, bool]]:
    want_mirror = random.random() < 0.6
    candidates = []
    for subset in combinations(TRANSFORMS, MOTIF_COUNT):
        signatures = {_signature_for_transform(labels, transform) for transform in subset}
        if len(signatures) < 3:
            continue
        if want_mirror and not any(flip for _rot, flip in subset):
            continue
        if want_mirror and not any(not flip for _rot, flip in subset):
            continue
        candidates.append(list(subset))
    if not candidates:
        raise RuntimeError("failed to choose transform subset")
    return random.choice(candidates)


def _anchor_bounds(
    shape: tuple[int, int],
    gray_offsets: set[tuple[int, int]],
    labels: list[tuple[tuple[int, int], int]],
    transform: tuple[int, bool],
) -> tuple[int, int, int, int]:
    needed = [(0, 0)]
    needed.extend(_apply_transform(offset, transform) for offset in gray_offsets)
    transformed_labels = [_apply_transform(offset, transform) for offset, _color in labels]
    needed.extend(transformed_labels)
    needed.extend((2 * dr, 2 * dc) for dr, dc in transformed_labels)
    min_r = min(r for r, _c in needed)
    max_r = max(r for r, _c in needed)
    min_c = min(c for _r, c in needed)
    max_c = max(c for _r, c in needed)
    return -min_r, shape[0] - 1 - max_r, -min_c, shape[1] - 1 - max_c


def _quadrant_boxes(shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = shape
    mid_r = h // 2
    mid_c = w // 2
    return [
        (2, mid_r - 2, 2, mid_c - 2),
        (2, mid_r - 2, mid_c + 1, w - 3),
        (mid_r + 1, h - 3, 2, mid_c - 2),
        (mid_r + 1, h - 3, mid_c + 1, w - 3),
    ]


def _occupied_cells(anchor: tuple[int, int], gray_offsets: set[tuple[int, int]], label_offsets: list[tuple[tuple[int, int], int]], transform: tuple[int, bool]) -> set[tuple[int, int]]:
    ar, ac = anchor
    cells = {(int(ar), int(ac))}
    for dr, dc in (_apply_transform(offset, transform) for offset in gray_offsets):
        cells.add((int(ar + dr), int(ac + dc)))
    for (offset, _color) in label_offsets:
        dr, dc = _apply_transform(offset, transform)
        cells.add((int(ar + dr), int(ac + dc)))
    return cells


def _min_chebyshev(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            best = min(best, max(abs(r0 - r1), abs(c0 - c1)))
    return int(best)


def _build_candidate_motifs(
    shape: tuple[int, int],
    gray_offsets: set[tuple[int, int]],
    labels: list[tuple[tuple[int, int], int]],
    transforms: list[tuple[int, bool]],
) -> list[dict] | None:
    boxes = _quadrant_boxes(shape)
    permuted = list(zip(random.sample(boxes, len(boxes)), random.sample(transforms, len(transforms))))
    motifs: list[dict] = []
    future_occ: list[set[tuple[int, int]]] = []

    for box, transform in permuted:
        r_lo, r_hi, c_lo, c_hi = box
        a_r_lo, a_r_hi, a_c_lo, a_c_hi = _anchor_bounds(shape, gray_offsets, labels, transform)
        r0 = max(r_lo, a_r_lo)
        r1 = min(r_hi, a_r_hi)
        c0 = max(c_lo, a_c_lo)
        c1 = min(c_hi, a_c_hi)
        if r0 > r1 or c0 > c1:
            return None

        anchors = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        random.shuffle(anchors)
        placed = None
        for anchor in anchors:
            occ = _occupied_cells(anchor, gray_offsets, labels, transform)
            if any(_min_chebyshev(occ, other) < 3 for other in future_occ):
                continue
            label_infos = []
            ar, ac = anchor
            for offset, color in labels:
                dr, dc = _apply_transform(offset, transform)
                pos = (int(ar + dr), int(ac + dc))
                label_infos.append({"pos": pos, "offset": (int(dr), int(dc)), "color": int(color)})
            gray_cells = {(int(ar + dr), int(ac + dc)) for dr, dc in (_apply_transform(off, transform) for off in gray_offsets)}
            placed = {
                "anchor": (int(ar), int(ac)),
                "gray_cells": gray_cells,
                "labels": label_infos,
                "transform": transform,
                "all_occ": occ,
            }
            break
        if placed is None:
            return None
        motifs.append(placed)
        future_occ.append(placed["all_occ"])
    return motifs


def _render_input(shape: tuple[int, int], motifs: list[dict], source_index: int) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for i, motif in enumerate(motifs):
        ar, ac = motif["anchor"]
        grid[ar, ac] = ANCHOR
        for r, c in motif["gray_cells"]:
            grid[r, c] = SKELETON
        if i == source_index:
            for label in motif["labels"]:
                lr, lc = label["pos"]
                grid[lr, lc] = int(label["color"])
    return grid


def _overlap_profile(shape: tuple[int, int], motifs: list[dict]) -> tuple[int, int]:
    counts: Counter[tuple[int, int]] = Counter()
    ordered = sorted(motifs, key=lambda motif: (motif["anchor"][0], motif["anchor"][1]))
    for motif in ordered:
        labels = sorted(motif["labels"], key=lambda label: (label["pos"][0], label["pos"][1], label["color"]))
        for label in labels:
            for cell in _ray_cells(label["pos"], label["offset"], shape):
                counts[cell] += 1
    overlap_cells = sum(1 for value in counts.values() if value > 1)
    total_cells = len(counts)
    return int(overlap_cells), int(total_cells)


def _semantic_checks(inp: np.ndarray, out: np.ndarray, motifs: list[dict], source_index: int, want_overlap: bool) -> bool:
    parsed = _parse_motifs(inp)
    if len(parsed) != MOTIF_COUNT:
        return False
    if sum(1 for motif in parsed if motif["labels"]) != 1:
        return False

    signatures = {tuple(sorted(label["offset"] for label in motif["labels"])) for motif in motifs}
    if len(signatures) < 3:
        return False

    overlap_cells, total_cells = _overlap_profile(inp.shape, motifs)
    if want_overlap and overlap_cells == 0:
        return False
    if not want_overlap and overlap_cells != 0:
        return False
    if overlap_cells > max(6, total_cells // 4):
        return False

    unique_ray_pixels = int(np.count_nonzero(out))
    if unique_ray_pixels < len(motifs):
        return False
    if unique_ray_pixels > inp.size // 2:
        return False

    source_labels = motifs[source_index]["labels"]
    if not (1 <= len(source_labels) <= 2):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
        gray_offsets, labels = _choose_base_shape()
        transforms = _choose_transform_subset(labels)
        want_overlap = random.random() < 0.5

        motifs = None
        for _ in range(PLACEMENT_ATTEMPTS):
            candidate = _build_candidate_motifs(shape, gray_offsets, labels, transforms)
            if candidate is None:
                continue
            overlap_cells, total_cells = _overlap_profile(shape, candidate)
            if want_overlap and overlap_cells == 0:
                continue
            if not want_overlap and overlap_cells != 0:
                continue
            if overlap_cells > max(6, total_cells // 4):
                continue
            motifs = candidate
            break
        if motifs is None:
            continue

        source_index = random.randrange(MOTIF_COUNT)
        inp = _render_input(shape, motifs, source_index)
        expected = _render_output_from_motifs(shape, motifs)

        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if not _semantic_checks(inp, expected, motifs, source_index, want_overlap):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 10")
