from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SEED = 3
MOTIF = 4
FRAME = 5

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
QUADS = ("TL", "TR", "BL", "BR")
SIZE_OPTIONS = ((6, 6), (6, 8), (8, 6), (8, 8))
GEN_ATTEMPTS = 500


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _spiral_path(seed_r: int, seed_c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = seed_r, seed_c
    max_delta = max(r, h - 1 - r, c, w - 1 - c)
    max_leg = 2 * (max_delta + 1)
    direction_index = 0
    leg_len = 2
    path: list[tuple[int, int]] = []

    while leg_len <= max_leg:
        dr, dc = DIRS[direction_index]
        for _ in range(leg_len):
            r += dr
            c += dc
            if in_bounds(shape, r, c):
                path.append((r, c))
        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg_len += 2
    return path


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_slices(bbox: tuple[int, int, int, int]) -> dict[str, tuple[slice, slice]]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h % 2 == 0 and inner_w % 2 == 0
    hh = inner_h // 2
    ww = inner_w // 2
    return {
        "TL": (slice(r0 + 1, r0 + 1 + hh), slice(c0 + 1, c0 + 1 + ww)),
        "TR": (slice(r0 + 1, r0 + 1 + hh), slice(c0 + 1 + ww, c1)),
        "BL": (slice(r0 + 1 + hh, r1), slice(c0 + 1, c0 + 1 + ww)),
        "BR": (slice(r0 + 1 + hh, r1), slice(c0 + 1 + ww, c1)),
    }


def _to_canonical(mask: np.ndarray, quad: str) -> np.ndarray:
    if quad == "TL":
        return mask
    if quad == "TR":
        return np.fliplr(mask)
    if quad == "BL":
        return np.flipud(mask)
    return np.flipud(np.fliplr(mask))


def _from_canonical(mask: np.ndarray, quad: str) -> np.ndarray:
    if quad == "TL":
        return mask
    if quad == "TR":
        return np.fliplr(mask)
    if quad == "BL":
        return np.flipud(mask)
    return np.flipud(np.fliplr(mask))


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for coords in components_mask(grid == FRAME):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = _border_mask(patch.shape)
        assert np.array_equal(patch == FRAME, expected), "grey component must be a 1-pixel-thick rectangle"
        frames.append(bbox)
    frames.sort()
    assert 3 <= len(frames) <= 4
    return frames


def _frame_for_cell(frames: Iterable[tuple[int, int, int, int]], r: int, c: int) -> tuple[int, int, int, int] | None:
    for bbox in frames:
        r0, c0, r1, c1 = bbox
        if r0 <= r <= r1 and c0 <= c <= c1:
            return bbox
    return None


def _complete_frame(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    slices = _frame_slices(bbox)
    quad_masks = {
        quad: out[rs, cs] == MOTIF
        for quad, (rs, cs) in slices.items()
    }
    present = [quad for quad, mask in quad_masks.items() if mask.any()]
    missing = [quad for quad, mask in quad_masks.items() if not mask.any()]
    assert len(present) == 3 and len(missing) == 1
    canonical = _to_canonical(quad_masks[present[0]], present[0])
    for quad in present[1:]:
        assert np.array_equal(canonical, _to_canonical(quad_masks[quad], quad))
    pred = _from_canonical(canonical, missing[0])
    rs, cs = slices[missing[0]]
    out[rs, cs][pred] = MOTIF


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    seed_coords = np.argwhere(g == SEED)
    assert seed_coords.shape == (1, 2)
    seed_r, seed_c = map(int, seed_coords[0])
    frames = _find_frames(g)

    out = g.copy()
    activated = None
    for r, c in _spiral_path(seed_r, seed_c, g.shape):
        color = int(g[r, c])
        if color != BG:
            activated = _frame_for_cell(frames, r, c)
            assert activated is not None
            break
        out[r, c] = SEED

    if activated is not None:
        _complete_frame(out, activated)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the green spiral until a frame blocks it, then complete only that touched frame's missing symmetric quadrant.

    Input structure:
    - The grid has a black background, exactly one green seed pixel, and 3-4 disjoint gray rectangular frames.
    - Each frame is a 1-pixel-thick rectangle whose interior contains a yellow motif arranged as four mirrored quadrants with exactly one quadrant missing.
    - The seed lies outside every frame, and untouched background separates the frames from each other.

    Rule:
    - Starting at the green seed, trace a square spiral in direction order up, right, down, left with leg lengths 2, 2, 4, 4, 6, 6, and so on.
    - Paint every in-bounds black spiral cell green; out-of-bounds steps are skipped but still count toward the leg length.
    - If the next in-bounds spiral cell is non-black, stop before painting it and mark the containing frame as activated.
    - Complete only the activated frame by inferring the missing quadrant from the other three mirrored quadrants; leave every untouched frame unchanged.

    Key insight:
    - The spiral is both a drawing process and a spatial selector: the first frame it reaches simultaneously blocks further motion and is the only frame whose interior symmetry is unlocked.
    """
    return _solve(grid)


def _bbox_distance(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _bbox_with_gap(bbox: tuple[int, int, int, int], gap: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - gap, c0 - gap, r1 + gap, c1 + gap


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_contains(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _frame_border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _all_candidate_bboxes(
    shape: tuple[int, int],
    forbidden_cells: set[tuple[int, int]],
    used_bboxes: list[tuple[int, int, int, int]],
    forbid_seed: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    candidates = []
    for fh, fw in SIZE_OPTIONS:
        for r0 in range(h - fh + 1):
            r1 = r0 + fh - 1
            for c0 in range(w - fw + 1):
                c1 = c0 + fw - 1
                bbox = (r0, c0, r1, c1)
                if _bbox_contains(bbox, forbid_seed):
                    continue
                if any(_bbox_contains(bbox, cell) for cell in forbidden_cells):
                    continue
                expanded = _bbox_with_gap(bbox, 3)
                if any(_bboxes_overlap(expanded, other) for other in used_bboxes):
                    continue
                candidates.append(bbox)
    random.shuffle(candidates)
    return candidates


def _active_bbox_candidates(
    shape: tuple[int, int],
    blocker: tuple[int, int],
    pre_path: set[tuple[int, int]],
    seed: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    br, bc = blocker
    candidates = []
    for fh, fw in SIZE_OPTIONS:
        for side in ("top", "bottom", "left", "right"):
            span = fw if side in ("top", "bottom") else fh
            for offset in range(1, span - 1):
                if side == "top":
                    r0, c0 = br, bc - offset
                elif side == "bottom":
                    r0, c0 = br - fh + 1, bc - offset
                elif side == "left":
                    r0, c0 = br - offset, bc
                else:
                    r0, c0 = br - offset, bc - fw + 1
                r1, c1 = r0 + fh - 1, c0 + fw - 1
                bbox = (r0, c0, r1, c1)
                if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                    continue
                if _bbox_contains(bbox, seed):
                    continue
                if any(_bbox_contains(bbox, cell) for cell in pre_path):
                    continue
                border = _frame_border_cells(bbox)
                if blocker not in border:
                    continue
                candidates.append(bbox)
    random.shuffle(candidates)
    return candidates


def _first_path_hit(path: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> int | None:
    for i, cell in enumerate(path):
        if _bbox_contains(bbox, cell):
            return i
    return None


def _make_canonical_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    lo = max(2, (area + 2) // 3)
    hi = max(lo, (2 * area) // 3)
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(100):
        count = random.randint(lo, hi)
        chosen = random.sample(cells, count)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if len({r for r, _ in chosen}) < min(2, h):
            continue
        if len({c for _, c in chosen}) < min(2, w):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to make asymmetric motif")


def _render_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    canonical: np.ndarray,
    missing_quad: str,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    for quad, (rs, cs) in _frame_slices(bbox).items():
        if quad == missing_quad:
            continue
        patch = _from_canonical(canonical, quad)
        grid[rs, cs][patch] = MOTIF


def _potential_completion_delta(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    before = int(np.count_nonzero(grid == MOTIF))
    tmp = grid.copy()
    _complete_frame(tmp, bbox)
    after = int(np.count_nonzero(tmp == MOTIF))
    return after - before


def _build_input_from_specs(
    shape: tuple[int, int],
    seed: tuple[int, int],
    frame_specs: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    sr, sc = seed
    grid[sr, sc] = SEED
    for spec in frame_specs:
        _render_frame(grid, spec["bbox"], spec["canonical"], spec["missing"])
    return grid


def _generate_frame_spec(bbox: tuple[int, int, int, int], forbidden_missing: str | None = None) -> dict:
    inner_h = bbox[2] - bbox[0] - 1
    inner_w = bbox[3] - bbox[1] - 1
    canonical = _make_canonical_mask(inner_h // 2, inner_w // 2)
    choices = [q for q in QUADS if q != forbidden_missing] if forbidden_missing else list(QUADS)
    return {
        "bbox": bbox,
        "canonical": canonical,
        "missing": random.choice(choices),
    }


def _attempt_generate() -> np.ndarray | None:
    rows = random.randint(22, 28)
    cols = random.randint(22, 28)
    frame_count = random.randint(3, 4)
    seed_r = random.randint(6, rows - 7)
    seed_c = random.randint(6, cols - 7)
    seed = (seed_r, seed_c)
    path = _spiral_path(seed_r, seed_c, (rows, cols))
    if len(path) < 35:
        return None

    stop_min = 10
    stop_max = len(path) - 15
    if stop_min >= stop_max:
        return None
    stop_i = random.randint(stop_min, stop_max)
    blocker = path[stop_i]
    pre_path = set(path[:stop_i])
    active_candidates = _active_bbox_candidates((rows, cols), blocker, pre_path, seed)
    if not active_candidates:
        return None

    active_bbox = random.choice(active_candidates)
    used = [active_bbox]
    all_candidates = _all_candidate_bboxes((rows, cols), pre_path, used, seed)
    if len(all_candidates) < 2:
        return None

    later_candidates = []
    for bbox in all_candidates:
        hit = _first_path_hit(path, bbox)
        if hit is not None and hit > stop_i + 6:
            later_candidates.append((hit, bbox))
    if not later_candidates:
        return None
    later_candidates.sort(key=lambda item: item[0])
    later_bbox = random.choice(later_candidates[: min(12, len(later_candidates))])[1]
    used.append(later_bbox)

    active_dist = _bbox_distance(seed, active_bbox)
    remaining = _all_candidate_bboxes((rows, cols), pre_path, used, seed)
    close_options = [bbox for bbox in remaining if _bbox_distance(seed, bbox) <= active_dist]
    if _bbox_distance(seed, later_bbox) <= active_dist:
        close_bbox = None
    else:
        if not close_options:
            return None
        close_bbox = random.choice(close_options)
        used.append(close_bbox)

    while len(used) < frame_count:
        extra_options = _all_candidate_bboxes((rows, cols), pre_path, used, seed)
        if not extra_options:
            return None
        used.append(random.choice(extra_options))

    frame_specs = []
    active_spec = _generate_frame_spec(active_bbox)
    frame_specs.append(active_spec)

    other_bboxes = [bbox for bbox in used if bbox != active_bbox]
    used_missing = {active_spec["missing"]}
    for i, bbox in enumerate(other_bboxes):
        forbid = None
        if i == 0 and len(used_missing) == 1:
            forbid = active_spec["missing"]
        spec = _generate_frame_spec(bbox, forbidden_missing=forbid)
        used_missing.add(spec["missing"])
        frame_specs.append(spec)
    random.shuffle(frame_specs)

    active_index = next(i for i, spec in enumerate(frame_specs) if spec["bbox"] == active_bbox)
    inp = _build_input_from_specs((rows, cols), seed, frame_specs)

    if _frame_for_cell([spec["bbox"] for spec in frame_specs], *blocker) != active_bbox:
        return None

    deltas = [_potential_completion_delta(inp, spec["bbox"]) for spec in frame_specs]
    if deltas[active_index] <= 0:
        return None
    inactive_deltas = [d for i, d in enumerate(deltas) if i != active_index]
    if not inactive_deltas or max(inactive_deltas) <= 0:
        return None

    expected = _solve(inp)
    if np.array_equal(expected, inp):
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None

    all_completed = inp.copy()
    for spec in frame_specs:
        _complete_frame(all_completed, spec["bbox"])
    path_only = inp.copy()
    for r, c in path[:stop_i]:
        path_only[r, c] = SEED
    if np.array_equal(all_completed, expected):
        return None
    if np.array_equal(path_only, expected):
        return None

    changed_frames = 0
    for spec in frame_specs:
        bbox = spec["bbox"]
        r0, c0, r1, c1 = bbox
        before = inp[r0 : r1 + 1, c0 : c1 + 1]
        after = expected[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(before, after):
            changed_frames += 1
    if changed_frames != 1:
        return None

    frame_boxes = [spec["bbox"] for spec in frame_specs]
    if sum(_bbox_distance(seed, bbox) <= active_dist for bbox in frame_boxes if bbox != active_bbox) < 1:
        return None

    later_hits = [(_first_path_hit(path, bbox), bbox) for bbox in frame_boxes if bbox != active_bbox]
    if not any(hit is not None and hit > stop_i + 6 for hit, _ in later_hits):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate valid puzzle input")
