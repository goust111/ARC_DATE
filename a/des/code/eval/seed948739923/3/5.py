from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import random

import numpy as np


BLUE = 1
RED = 2
YELLOW = 4
BG = 0

PATTERN_COLORS = (3, 5, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
DIRS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
OPPOSITE = {
    "top": "bottom",
    "right": "left",
    "bottom": "top",
    "left": "right",
}

GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 30
BOX_PAD = 2


@dataclass(frozen=True)
class Fence:
    bbox: tuple[int, int, int, int]
    gap_side: str
    marker: tuple[int, int]
    patch_origin: tuple[int, int]
    patch: np.ndarray


@dataclass(frozen=True)
class FenceSpec:
    side: str
    same_side: bool
    outer_h: int
    outer_w: int
    pattern: np.ndarray
    pattern_pos: tuple[int, int]
    marker_pos: tuple[int, int]
    gap_start: int
    gap_len: int
    bbox: tuple[int, int, int, int]


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIRS.values():
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_side_from_missing(
    missing: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int
) -> str | None:
    if not (1 <= len(missing) <= 3):
        return None

    side: str | None = None
    coords: list[int] = []
    for r, c in missing:
        if r == r0 and c0 < c < c1:
            here = "top"
            coord = c
        elif r == r1 and c0 < c < c1:
            here = "bottom"
            coord = c
        elif c == c0 and r0 < r < r1:
            here = "left"
            coord = r
        elif c == c1 and r0 < r < r1:
            here = "right"
            coord = r
        else:
            return None
        if side is None:
            side = here
        elif side != here:
            return None
        coords.append(coord)

    coords.sort()
    if coords != list(range(coords[0], coords[0] + len(coords))):
        return None
    return side


def _extract_fences(grid: np.ndarray, bg: int) -> list[Fence]:
    g = np.asarray(grid, dtype=int)
    fences: list[Fence] = []
    for pts in _components(g == BLUE):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        pts_set = set(pts)
        perimeter = _perimeter_cells(r0, c0, r1, c1)
        if not pts_set.issubset(perimeter):
            continue
        missing = perimeter - pts_set
        gap_side = _gap_side_from_missing(missing, r0, c0, r1, c1)
        if gap_side is None:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_pts = np.argwhere(interior == YELLOW)
        if marker_pts.shape != (1, 2):
            continue
        marker_r = r0 + 1 + int(marker_pts[0, 0])
        marker_c = c0 + 1 + int(marker_pts[0, 1])

        motif_mask = (interior != bg) & (interior != BLUE) & (interior != YELLOW)
        if not motif_mask.any():
            continue
        motif_pts = np.argwhere(motif_mask)
        pr0, pc0 = motif_pts.min(axis=0)
        pr1, pc1 = motif_pts.max(axis=0)
        pr0 = int(pr0)
        pc0 = int(pc0)
        pr1 = int(pr1)
        pc1 = int(pc1)
        patch = interior[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        patch[patch == YELLOW] = bg
        patch[patch == BLUE] = bg

        fences.append(
            Fence(
                bbox=(int(r0), int(c0), int(r1), int(c1)),
                gap_side=gap_side,
                marker=(int(marker_r), int(marker_c)),
                patch_origin=(int(r0 + 1 + pr0), int(c0 + 1 + pc0)),
                patch=patch,
            )
        )

    return fences


def _marker_side(fence: Fence) -> str:
    r0, c0, r1, c1 = fence.bbox
    mr, mc = fence.marker
    if fence.gap_side in {"top", "bottom"}:
        top_dist = mr - (r0 + 1)
        bottom_dist = (r1 - 1) - mr
        assert top_dist != bottom_dist, "yellow marker must be closer to one axial side"
        return "top" if top_dist < bottom_dist else "bottom"
    left_dist = mc - (c0 + 1)
    right_dist = (c1 - 1) - mc
    assert left_dist != right_dist, "yellow marker must be closer to one axial side"
    return "left" if left_dist < right_dist else "right"


def _project_patch(fence: Fence, bg: int) -> np.ndarray:
    patch = fence.patch.copy()
    if _marker_side(fence) == fence.gap_side:
        return patch
    # Mirror across the cross-section of the beam.
    if fence.gap_side in {"top", "bottom"}:
        return np.fliplr(patch)
    return np.flipud(patch)


def _projection_cells(grid: np.ndarray, fence: Fence, bg: int) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = fence.bbox
    pr0, pc0 = fence.patch_origin
    patch = _project_patch(fence, bg)
    ph, pw = patch.shape
    dr, dc = DIRS[fence.gap_side]

    if fence.gap_side == "top":
        base_r, base_c = r0 - ph, pc0
    elif fence.gap_side == "bottom":
        base_r, base_c = r1 + 1, pc0
    elif fence.gap_side == "left":
        base_r, base_c = pr0, c0 - pw
    else:
        base_r, base_c = pr0, c1 + 1

    local = {(int(rr), int(cc)): int(patch[rr, cc]) for rr, cc in np.argwhere(patch != bg)}
    emitted: dict[tuple[int, int], int] = {}
    step = 0
    while True:
        top = base_r + dr * step
        left = base_c + dc * step
        targets: list[tuple[int, int, int]] = []
        blocked = False
        for (rr, cc), color in local.items():
            tr = top + rr
            tc = left + cc
            if not (0 <= tr < h and 0 <= tc < w):
                blocked = True
                break
            if int(g[tr, tc]) == BLUE:
                blocked = True
                break
            targets.append((int(tr), int(tc), int(color)))
        if blocked:
            break
        for tr, tc, color in targets:
            emitted[(tr, tc)] = color
        step += 1
    return emitted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each fenced inner pattern outward through the fence gap.

    The gap side gives the beam direction. The yellow marker decides whether
    the projected patch is used as-is or mirrored across the beam cross-section.
    Projections stop before leaving the grid or touching any blue fence wall.
    Overlaps between different fences turn red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    fences = _extract_fences(g, bg)
    assert fences, "expected at least one blue fence"

    hits: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, fence in enumerate(fences):
        for pos, color in _projection_cells(g, fence, bg).items():
            hits[pos].append((idx, color))

    out = g.copy()
    for (r, c), entries in hits.items():
        if len({idx for idx, _ in entries}) >= 2:
            out[r, c] = RED
        else:
            out[r, c] = int(entries[0][1])
    return out


def _color_sequence(length: int, rng: random.Random) -> list[int]:
    while True:
        seq = [rng.choice(PATTERN_COLORS) for _ in range(length)]
        if len(set(seq)) < 2:
            continue
        if seq == list(reversed(seq)):
            continue
        return seq


def _bbox_clear(existing: list[tuple[int, int, int, int]], cand: tuple[int, int, int, int], pad: int) -> bool:
    r0, c0, r1, c1 = cand
    for rr0, cc0, rr1, cc1 in existing:
        if not (r1 + pad < rr0 or rr1 + pad < r0 or c1 + pad < cc0 or cc1 + pad < c0):
            return False
    return True


def _random_fence_spec(
    rng: random.Random,
    side: str,
    same_side: bool,
    outer_h: int,
    outer_w: int,
    bbox: tuple[int, int, int, int],
    intercept: int | None = None,
) -> FenceSpec:
    inner_h = outer_h - 2
    inner_w = outer_w - 2
    r0, c0, r1, c1 = bbox

    if side in {"top", "bottom"}:
        length = rng.randint(2, min(6, inner_w))
        if intercept is None:
            start = rng.randint(0, inner_w - length)
        else:
            rel = intercept - (c0 + 1)
            lo = max(0, rel - length + 1)
            hi = min(rel, inner_w - length)
            start = rng.randint(lo, hi)
        pattern = np.array([_color_sequence(length, rng)], dtype=int)
        if side == "top":
            marker_r, pattern_r = (0, 1) if same_side else (inner_h - 1, inner_h - 2)
        else:
            marker_r, pattern_r = (inner_h - 1, inner_h - 2) if same_side else (0, 1)
        marker_c = start + rng.randrange(length)
        gap_len = rng.randint(1, min(3, length))
        gap_start = rng.randint(start, start + length - gap_len)
        return FenceSpec(
            side=side,
            same_side=same_side,
            outer_h=outer_h,
            outer_w=outer_w,
            pattern=pattern,
            pattern_pos=(pattern_r, start),
            marker_pos=(marker_r, marker_c),
            gap_start=gap_start,
            gap_len=gap_len,
            bbox=bbox,
        )

    length = rng.randint(2, min(6, inner_h))
    if intercept is None:
        start = rng.randint(0, inner_h - length)
    else:
        rel = intercept - (r0 + 1)
        lo = max(0, rel - length + 1)
        hi = min(rel, inner_h - length)
        start = rng.randint(lo, hi)
    pattern = np.array(_color_sequence(length, rng), dtype=int)[:, None]
    if side == "left":
        marker_c, pattern_c = (0, 1) if same_side else (inner_w - 1, inner_w - 2)
    else:
        marker_c, pattern_c = (inner_w - 1, inner_w - 2) if same_side else (0, 1)
    marker_r = start + rng.randrange(length)
    gap_len = rng.randint(1, min(3, length))
    gap_start = rng.randint(start, start + length - gap_len)
    return FenceSpec(
        side=side,
        same_side=same_side,
        outer_h=outer_h,
        outer_w=outer_w,
        pattern=pattern,
        pattern_pos=(start, pattern_c),
        marker_pos=(marker_r, marker_c),
        gap_start=gap_start,
        gap_len=gap_len,
        bbox=bbox,
    )


def _paint_fence(grid: np.ndarray, spec: FenceSpec) -> None:
    r0, c0, r1, c1 = spec.bbox
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE

    if spec.side == "top":
        grid[r0, c0 + 1 + spec.gap_start : c0 + 1 + spec.gap_start + spec.gap_len] = BG
    elif spec.side == "bottom":
        grid[r1, c0 + 1 + spec.gap_start : c0 + 1 + spec.gap_start + spec.gap_len] = BG
    elif spec.side == "left":
        grid[r0 + 1 + spec.gap_start : r0 + 1 + spec.gap_start + spec.gap_len, c0] = BG
    else:
        grid[r0 + 1 + spec.gap_start : r0 + 1 + spec.gap_start + spec.gap_len, c1] = BG

    pr, pc = spec.pattern_pos
    ph, pw = spec.pattern.shape
    abs_pr = r0 + 1 + pr
    abs_pc = c0 + 1 + pc
    grid[abs_pr : abs_pr + ph, abs_pc : abs_pc + pw] = spec.pattern

    mr, mc = spec.marker_pos
    grid[r0 + 1 + mr, c0 + 1 + mc] = YELLOW


def _random_bbox(
    rng: random.Random,
    h: int,
    w: int,
    outer_h: int,
    outer_w: int,
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    coords = [
        (r0, c0)
        for r0 in range(1, h - outer_h)
        for c0 in range(1, w - outer_w)
    ]
    rng.shuffle(coords)
    for r0, c0 in coords:
        cand = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if _bbox_clear(existing, cand, BOX_PAD):
            return cand
    return None


def _collision_pair(
    rng: random.Random, h: int, w: int
) -> tuple[list[FenceSpec], list[tuple[int, int, int, int]]] | None:
    cr = rng.randint(7, h - 8)
    cc = rng.randint(7, w - 8)
    existing: list[tuple[int, int, int, int]] = []
    specs: list[FenceSpec] = []

    vertical_side = rng.choice(["top", "bottom"])
    horizontal_side = rng.choice(["left", "right"])

    for _ in range(120):
        existing.clear()
        specs.clear()

        va_h = rng.randint(5, 8)
        va_w = rng.randint(5, 10)
        hb_h = rng.randint(5, 10)
        hb_w = rng.randint(5, 8)

        if vertical_side == "top":
            r_choices = list(range(cr + 1, h - va_h))
        else:
            r_choices = list(range(1, cr - va_h + 1))
        c_lo = max(1, cc - va_w + 2)
        c_hi = min(cc - 1, w - va_w - 1)
        c_choices = list(range(c_lo, c_hi + 1))
        if not r_choices or not c_choices:
            continue
        rng.shuffle(r_choices)
        rng.shuffle(c_choices)
        va_bbox = None
        for r0 in r_choices:
            for c0 in c_choices:
                cand = (r0, c0, r0 + va_h - 1, c0 + va_w - 1)
                if _bbox_clear(existing, cand, BOX_PAD):
                    va_bbox = cand
                    break
            if va_bbox is not None:
                break
        if va_bbox is None:
            continue
        existing.append(va_bbox)
        specs.append(
            _random_fence_spec(
                rng,
                side=vertical_side,
                same_side=bool(rng.randrange(2) == 0),
                outer_h=va_h,
                outer_w=va_w,
                bbox=va_bbox,
                intercept=cc,
            )
        )

        if horizontal_side == "left":
            c_choices = list(range(cc + 1, w - hb_w))
        else:
            c_choices = list(range(1, cc - hb_w + 1))
        r_lo = max(1, cr - hb_h + 2)
        r_hi = min(cr - 1, h - hb_h - 1)
        r_choices = list(range(r_lo, r_hi + 1))
        if not r_choices or not c_choices:
            continue
        rng.shuffle(r_choices)
        rng.shuffle(c_choices)
        hb_bbox = None
        for r0 in r_choices:
            for c0 in c_choices:
                cand = (r0, c0, r0 + hb_h - 1, c0 + hb_w - 1)
                if _bbox_clear(existing, cand, BOX_PAD):
                    hb_bbox = cand
                    break
            if hb_bbox is not None:
                break
        if hb_bbox is None:
            continue
        existing.append(hb_bbox)
        specs.append(
            _random_fence_spec(
                rng,
                side=horizontal_side,
                same_side=bool(rng.randrange(2) == 0),
                outer_h=hb_h,
                outer_w=hb_w,
                bbox=hb_bbox,
                intercept=cr,
            )
        )
        return specs.copy(), existing.copy()

    return None


def _random_layout(rng: random.Random, want_collision: bool) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    specs: list[FenceSpec] = []
    existing: list[tuple[int, int, int, int]] = []

    if want_collision:
        pair = _collision_pair(rng, h, w)
        if pair is None:
            return None
        specs, existing = pair
        extra_count = rng.randint(0, 1)
    else:
        extra_count = rng.randint(2, 4)

    while len(specs) < extra_count + (2 if want_collision else 0):
        side = rng.choice(SIDES)
        same_side = bool(rng.randrange(2) == 0)
        outer_h = rng.randint(5, 10)
        outer_w = rng.randint(5, 10)
        bbox = _random_bbox(rng, h, w, outer_h, outer_w, existing)
        if bbox is None:
            return None
        spec = _random_fence_spec(
            rng,
            side=side,
            same_side=same_side,
            outer_h=outer_h,
            outer_w=outer_w,
            bbox=bbox,
        )
        specs.append(spec)
        existing.append(bbox)

    for spec in specs:
        _paint_fence(grid, spec)
    return grid


def _valid_generated(grid: np.ndarray, want_collision: bool) -> bool:
    fences = _extract_fences(grid, BG)
    if len(fences) < 2:
        return False
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out != grid):
        return False
    has_collision = bool(np.any(out == RED))
    if has_collision != want_collision:
        return False
    if _mode_color(grid) != BG:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    want_collision = bool(rng.randrange(2) == 0)
    for _ in range(GEN_RETRIES):
        grid = _random_layout(rng, want_collision)
        if grid is None:
            continue
        if _valid_generated(grid, want_collision):
            return grid
    raise RuntimeError("failed to generate a valid projection puzzle")
