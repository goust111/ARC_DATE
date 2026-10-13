from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, in_bounds
import itertools
import math
import random


YELLOW = Color.YELLOW
PINK = Color.PINK


def _normalize(cells: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    min_r = min(r for r, _c, _color in cells)
    min_c = min(c for _r, c, _color in cells)
    return tuple(sorted((r - min_r, c - min_c, color) for r, c, color in cells))


def _dihedral_templates(template: np.ndarray) -> list[tuple[tuple[int, int, int], ...]]:
    cells = [
        (r, c, int(template[r, c]))
        for r in range(template.shape[0])
        for c in range(template.shape[1])
    ]
    transforms = [
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    ]

    variants: list[tuple[tuple[int, int, int], ...]] = []
    for transform in transforms:
        transformed = _normalize([(transform(r, c)[0], transform(r, c)[1], color) for r, c, color in cells])
        if transformed not in variants:
            variants.append(transformed)
    return variants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every magenta-marked transformed copy of the yellow/magenta template by painting the missing yellow cells.

    Input structure
    - The existing yellow pixels all lie inside one small rectangular template region.
    - That template region is fully tiled by yellow and magenta pixels.
    - Elsewhere, magenta pixels mark incomplete rotated or reflected copies of the same two-color template.

    Rule
    - Extract the template from the bounding box of all yellow pixels.
    - For every rotated or reflected template placement whose magenta cells are all visible magenta pixels, paint each visible non-magenta yellow-template cell yellow.
    - Allow yellow-template cells to be clipped off the grid edge, but require every magenta-template cell to be inside the grid.

    Key insight
    - The magenta cells are the alignment signature for each oriented copy; the yellow part of the same template is what must be restored.
    """
    g = np.asarray(grid, dtype=int)
    yellow_bbox = bbox(g == YELLOW)
    assert yellow_bbox is not None, "expected a yellow template"
    r0, c0, r1, c1 = yellow_bbox
    template = g[r0 : r1 + 1, c0 : c1 + 1]
    template_colors = set(map(int, np.unique(template)))
    assert template_colors == {YELLOW, PINK}, "yellow bbox must be a complete yellow/magenta template"

    H, W = g.shape
    additions: set[tuple[int, int]] = set()
    for variant in _dihedral_templates(template):
        vh = max(r for r, _c, _color in variant) + 1
        vw = max(c for _r, c, _color in variant) + 1
        pink_cells = [(r, c) for r, c, color in variant if color == PINK]
        yellow_cells = [(r, c) for r, c, color in variant if color == YELLOW]
        assert pink_cells and yellow_cells, "template must contain both colors"

        for top in range(-vh + 1, H):
            for left in range(-vw + 1, W):
                if not all(
                    in_bounds(g.shape, top + r, left + c) and int(g[top + r, left + c]) == PINK
                    for r, c in pink_cells
                ):
                    continue

                visible_yellow = 0
                candidate_additions: list[tuple[int, int]] = []
                for r, c in yellow_cells:
                    rr = top + r
                    cc = left + c
                    if not in_bounds(g.shape, rr, cc):
                        continue
                    visible_yellow += 1
                    value = int(g[rr, cc])
                    if value == PINK:
                        candidate_additions = []
                        break
                    if value != YELLOW:
                        candidate_additions.append((rr, cc))

                if visible_yellow and candidate_additions:
                    additions.update(candidate_additions)

    assert additions, "expected at least one incomplete template copy"
    out = g.copy()
    for r, c in additions:
        out[r, c] = YELLOW
    return out

# --------------- generator (o5x) ---------------

from arc_utils import paste

# Palette and background noise.
_GEN_YELLOW = 4
_GEN_PINK = 6
_GEN_NOISE = (0, 1, 8)
_GEN_OCCUPIED_COLORS = (_GEN_YELLOW, _GEN_PINK)

# Canvas and 3x3 template bounds.
_GEN_ROW_BOUNDS = (16, 19)
_GEN_COL_BOUNDS = (12, 17)
_GEN_TEMPLATE_SIDE = 3
_GEN_TEMPLATE_AREA = _GEN_TEMPLATE_SIDE * _GEN_TEMPLATE_SIDE
_GEN_PINK_COUNT_BOUNDS = (3, 5)
_GEN_MIN_VARIANT_COUNT = 4
_GEN_COMPLETE_EDGE_MARGIN = 1

# Incomplete copies and distractor pink cells.
_GEN_COPY_COUNT_BOUNDS = (1, 3)
_GEN_DISTRACTOR_COUNT_BOUNDS = (13, 22)
_GEN_PINK_DENSITY_BOUNDS = (0.10, 0.13)
_GEN_DISTRACTOR_CLUSTER_BOUNDS = (1, 4)
_GEN_CLUSTER_NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)

# Rejection sampling.
_GEN_ATTEMPT_LIMIT = 300
_GEN_LAST_META: dict[str, int | float] = {}


def _gen_dihedral_arrays(arr: np.ndarray) -> list[np.ndarray]:
    variants = [
        arr,
        np.fliplr(arr),
        np.flipud(arr),
        np.flipud(np.fliplr(arr)),
        arr.T,
        np.fliplr(arr.T),
        np.flipud(arr.T),
        np.flipud(np.fliplr(arr.T)),
    ]
    out: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for variant in variants:
        key = tuple(map(int, variant.ravel()))
        if key not in seen:
            seen.add(key)
            out.append(np.array(variant, dtype=int, copy=True))
    return out


def _gen_cells(arr: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(arr == int(color))]


def _gen_yellow_bbox_full(mask: np.ndarray) -> bool:
    return bbox(mask) == (0, 0, _GEN_TEMPLATE_SIDE - 1, _GEN_TEMPLATE_SIDE - 1)


def _gen_template_pool(pink_count: int) -> list[np.ndarray]:
    templates: list[np.ndarray] = []
    for pink_flat in itertools.combinations(range(_GEN_TEMPLATE_AREA), pink_count):
        arr = np.full((_GEN_TEMPLATE_SIDE, _GEN_TEMPLATE_SIDE), _GEN_YELLOW, dtype=int)
        for idx in pink_flat:
            arr[idx // _GEN_TEMPLATE_SIDE, idx % _GEN_TEMPLATE_SIDE] = _GEN_PINK
        if not _gen_yellow_bbox_full(arr == _GEN_YELLOW):
            continue
        if len(_gen_dihedral_arrays(arr)) < _GEN_MIN_VARIANT_COUNT:
            continue
        templates.append(arr)
    return templates


def _gen_random_template(pink_count: int) -> np.ndarray:
    return random.choice(_gen_template_pool(pink_count)).copy()


def _gen_sample_parameters() -> tuple[int, int, int, int, int]:
    H = random.randint(*_GEN_ROW_BOUNDS)
    W = random.randint(*_GEN_COL_BOUNDS)
    options: list[tuple[int, int, int]] = []
    for pink_count in range(_GEN_PINK_COUNT_BOUNDS[0], _GEN_PINK_COUNT_BOUNDS[1] + 1):
        for copy_count in range(_GEN_COPY_COUNT_BOUNDS[0], _GEN_COPY_COUNT_BOUNDS[1] + 1):
            fixed_pink = pink_count * (1 + copy_count)
            lo = max(
                _GEN_DISTRACTOR_COUNT_BOUNDS[0],
                math.ceil(_GEN_PINK_DENSITY_BOUNDS[0] * H * W - fixed_pink),
            )
            hi = min(
                _GEN_DISTRACTOR_COUNT_BOUNDS[1],
                math.floor(_GEN_PINK_DENSITY_BOUNDS[1] * H * W - fixed_pink),
            )
            for distractor_count in range(lo, hi + 1):
                options.append((pink_count, copy_count, distractor_count))
    pink_count, copy_count, distractor_count = random.choice(options)
    return H, W, pink_count, copy_count, distractor_count


def _gen_stamp_full(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    paste(dst, patch, top, left, clip=True)


def _gen_stamp_pink_only(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    paste(dst, patch, top, left, mask=(patch == _GEN_PINK), clip=True)


def _gen_copy_placements(inp: np.ndarray, out: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    H, W = inp.shape
    pink_cells = _gen_cells(patch, _GEN_PINK)
    yellow_cells = _gen_cells(patch, _GEN_YELLOW)
    min_pr = min(r for r, _c in pink_cells)
    max_pr = max(r for r, _c in pink_cells)
    min_pc = min(c for _r, c in pink_cells)
    max_pc = max(c for _r, c in pink_cells)
    placements: list[tuple[int, int]] = []

    for top in range(-min_pr, H - max_pr):
        for left in range(-min_pc, W - max_pc):
            visible_yellow = 0
            ok = True
            for r, c in pink_cells:
                rr = top + r
                cc = left + c
                if int(out[rr, cc]) in _GEN_OCCUPIED_COLORS:
                    ok = False
                    break
            if not ok:
                continue

            for r, c in yellow_cells:
                rr = top + r
                cc = left + c
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                visible_yellow += 1
                if int(out[rr, cc]) in _GEN_OCCUPIED_COLORS:
                    ok = False
                    break
            if ok and visible_yellow:
                placements.append((top, left))
    random.shuffle(placements)
    return placements


def _gen_cluster_cells(
    available: set[tuple[int, int]], seed: tuple[int, int], target_size: int
) -> list[tuple[int, int]]:
    cluster = [seed]
    used = {seed}
    while len(cluster) < target_size:
        frontier: set[tuple[int, int]] = set()
        for r, c in cluster:
            for dr, dc in _GEN_CLUSTER_NEIGHBOR_OFFSETS:
                nxt = (r + dr, c + dc)
                if nxt in available and nxt not in used:
                    frontier.add(nxt)
        if not frontier:
            break
        chosen = random.choice(tuple(frontier))
        used.add(chosen)
        cluster.append(chosen)
    return cluster


def _gen_scatter_distractors(inp: np.ndarray, out: np.ndarray, count: int) -> bool:
    H, W = inp.shape
    available = {
        (r, c)
        for r in range(H)
        for c in range(W)
        if int(inp[r, c]) not in _GEN_OCCUPIED_COLORS
        and int(out[r, c]) not in _GEN_OCCUPIED_COLORS
    }
    remaining = count
    while remaining:
        if not available:
            return False
        seed = random.choice(tuple(available))
        max_cluster = min(_GEN_DISTRACTOR_CLUSTER_BOUNDS[1], remaining)
        want = random.randint(_GEN_DISTRACTOR_CLUSTER_BOUNDS[0], max_cluster)
        cluster = _gen_cluster_cells(available, seed, want)
        if not cluster:
            return False
        for r, c in cluster:
            inp[r, c] = _GEN_PINK
            out[r, c] = _GEN_PINK
            available.remove((r, c))
        remaining -= len(cluster)
    return True


def _gen_make_canvas_pair(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.random.choice(_GEN_NOISE, size=(H, W)).astype(int)
    return inp, inp.copy()


def _gen_place_complete_template(inp: np.ndarray, out: np.ndarray, template: np.ndarray) -> None:
    H, W = inp.shape
    max_top = H - _GEN_TEMPLATE_SIDE - _GEN_COMPLETE_EDGE_MARGIN
    max_left = W - _GEN_TEMPLATE_SIDE - _GEN_COMPLETE_EDGE_MARGIN
    complete_top = random.randint(_GEN_COMPLETE_EDGE_MARGIN, max_top)
    complete_left = random.randint(_GEN_COMPLETE_EDGE_MARGIN, max_left)
    _gen_stamp_full(inp, template, complete_top, complete_left)
    _gen_stamp_full(out, template, complete_top, complete_left)


def _gen_place_incomplete_copies(
    inp: np.ndarray, out: np.ndarray, variants: list[np.ndarray], copy_count: int
) -> bool:
    for _ in range(copy_count):
        patch = random.choice(variants)
        placements = _gen_copy_placements(inp, out, patch)
        if not placements:
            return False
        top, left = placements[0]
        _gen_stamp_full(out, patch, top, left)
        _gen_stamp_pink_only(inp, patch, top, left)
    return True


def _gen_matches_solver(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _gen_record_meta(
    inp: np.ndarray,
    H: int,
    W: int,
    pink_count: int,
    copy_count: int,
    distractor_count: int,
    variant_count: int,
) -> None:
    global _GEN_LAST_META
    _GEN_LAST_META = {
        "rows": H,
        "cols": W,
        "pink_cells": pink_count,
        "yellow_cells": _GEN_TEMPLATE_AREA - pink_count,
        "copy_count": copy_count,
        "distractor_count": distractor_count,
        "variant_count": variant_count,
        "pink_density": float(np.count_nonzero(inp == _GEN_PINK) / inp.size),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W, pink_count, copy_count, distractor_count = _gen_sample_parameters()
    inp, out = _gen_make_canvas_pair(H, W)

    template = _gen_random_template(pink_count)
    _gen_place_complete_template(inp, out, template)

    variants = _gen_dihedral_arrays(template)
    if not _gen_place_incomplete_copies(inp, out, variants, copy_count):
        return None
    if not _gen_scatter_distractors(inp, out, distractor_count):
        return None
    if not _gen_matches_solver(inp, out):
        return None

    _gen_record_meta(inp, H, W, pink_count, copy_count, distractor_count, len(variants))
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 79369cc6 input")
