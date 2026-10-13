from __future__ import annotations

import numpy as np
from collections import Counter, deque
from typing import Dict, List, Sequence, Set, Tuple


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate a corner template by frame-gap orientation and stamp it at each frame seed.

    Input structure:
    - Background is the most frequent color.
    - Multiple hollow 1-pixel rectangular grey frames are present, each with exactly one
      missing pixel (the gap) at the midpoint of one border edge.
    - Each frame interior contains exactly one seed pixel of a non-background color.
    - In a corner region, 3x3 template masks are drawn in unique colors that do not repeat.

    Rule:
    - Infer the background by frequency, then detect each frame as a rectangle border with exactly
      one missing border pixel.
    - The single interior seed color of each frame selects the template with the same color.
    - The gap side selects a template rotation: top->0°, right->90° clockwise, bottom->180°,
      left->270° clockwise.
    - Place the rotated template centered on the seed pixel (covering that position) inside the frame,
      preserving the template color and leaving the frame and all other regions unchanged.

    Key insight:
    - The seed color identifies which template to use, while the gap orientation provides an independent
      rotation code.
    """

    arr = np.array(grid, copy=True)
    if arr.ndim != 2:
        return arr

    h, w = arr.shape
    bg = _background_color(arr)

    frames = _find_frames(arr, bg)
    if not frames:
        return arr

    frame_colors = {f["frame_color"] for f in frames}
    templates = _collect_templates(arr, bg, frame_colors, frames)

    out = arr.copy()
    for frame in frames:
        color = frame["seed_color"]
        pattern = templates.get(color)
        if pattern is None:
            continue

        rot = _rotate_template_by_gap(pattern, frame["gap_side"])
        sr, sc = frame["seed"]
        start_r = sr - rot.shape[0] // 2
        start_c = sc - rot.shape[1] // 2
        end_r = start_r + rot.shape[0]
        end_c = start_c + rot.shape[1]

        if start_r < 0 or start_c < 0 or end_r > h or end_c > w:
            continue
        for i in range(rot.shape[0]):
            for j in range(rot.shape[1]):
                if rot[i, j]:
                    out[start_r + i, start_c + j] = color

    return out


def generate_input() -> np.ndarray:
    """Generate ARC-like samples matching the frame-template puzzle description."""

    rng = np.random.default_rng()

    # Keep enough room for templates and multiple frames while respecting ARC limits.
    h = int(rng.integers(18, 29))
    w = int(rng.integers(18, 29))
    bg = 0
    grid = np.full((h, w), bg, dtype=int)

    # Frame color and template colors (all non-background and non-grey).
    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rng.shuffle(candidates)

    grey = candidates[0]
    n_frames = int(rng.integers(1, min(5, len(candidates) - 1)))
    template_colors = candidates[1 : 1 + n_frames]

    # Place 3x3 templates in one randomly chosen corner.
    corner = int(rng.integers(0, 4))
    corner_positions = _corner_template_positions(h, w, corner)
    reserved = np.zeros((h, w), dtype=bool)
    for r0, c0, Hc, Wc in [corner_positions]:
        reserved[r0 : min(h, r0 + Hc), c0 : min(w, c0 + Wc)] = True

    used_template_masks: List[np.ndarray] = []
    template_cells = []
    template_slots = [(0, 0), (0, 4), (4, 0), (4, 4)]
    for idx, color in enumerate(template_colors):
        mask = _random_template_mask(rng, used_template_masks)
        used_template_masks.append(mask)

        base_r, base_c = _template_slot_origin(*corner_positions[0:2], corner)
        slot_r, slot_c = template_slots[idx]
        # keep templates inside the corner pad and non-overlapping.
        tr = base_r + slot_r
        tc = base_c + slot_c
        if tr + 3 > h or tc + 3 > w:
            continue
        if reserved[tr : tr + 3, tc : tc + 3].any():
            continue
        # Slightly random jitter for visual variety, but keep inside bounds.
        tr = min(max(0, tr + int(rng.integers(0, 2))), h - 3)
        tc = min(max(0, tc + int(rng.integers(0, 2))), w - 3)

        for rr in range(3):
            for cc in range(3):
                if mask[rr, cc]:
                    grid[tr + rr, tc + cc] = color
        template_cells.append((tr, tc))
        for rr in range(3):
            for cc in range(3):
                if tr + rr < h and tc + cc < w:
                    reserved[tr + rr, tc + cc] = True

    # Place frames and seeds in the remaining area.
    occupied = np.copy(reserved)
    templates_info: Dict[int, np.ndarray] = {c: None for c in template_colors}
    for idx, color in enumerate(template_colors):
        placed = False
        interior_h = int(rng.choice([5, 7]))
        interior_w = int(rng.choice([5, 7]))
        frame_h = interior_h + 2
        frame_w = interior_w + 2

        # Odd sizes make the gap midpoint exact and seed aligned to frame center.
        gap_side = int(rng.integers(0, 4))
        side = ("top", "right", "bottom", "left")[gap_side]

        for _ in range(500):
            r0 = int(rng.integers(0, h - frame_h + 1))
            c0 = int(rng.integers(0, w - frame_w + 1))
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1

            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            # Keep distance from template corner area and from borders.
            if occupied[r0:r1 + 1, c0:c1 + 1].any():
                continue

            # Draw frame border.
            grid[r0, c0 : c1 + 1] = grey
            grid[r1, c0 : c1 + 1] = grey
            grid[r0:r1 + 1, c0] = grey
            grid[r0:r1 + 1, c1] = grey

            if side == "top":
                gap_r, gap_c = r0, c0 + frame_w // 2
            elif side == "right":
                gap_r, gap_c = r0 + frame_h // 2, c1
            elif side == "bottom":
                gap_r, gap_c = r1, c0 + frame_w // 2
            else:
                gap_r, gap_c = r0 + frame_h // 2, c0

            grid[gap_r, gap_c] = bg

            seed_r = r0 + 1 + interior_h // 2
            seed_c = c0 + 1 + interior_w // 2
            grid[seed_r, seed_c] = color

            if idx in templates_info:
                templates_info[color] = rng.choice([0, 1])

            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            occupied[seed_r - 1 : seed_r + interior_h - 1, seed_c - 1 : seed_c + interior_w - 1] = True
            placed = True
            break

        if not placed:
            # Fallback: reduce this frame to minimum and retry at origin-safe location.
            r0 = 1 + idx * 2
            c0 = 1 + idx * 2
            frame_h = 7
            frame_w = 7
            if r0 + frame_h < h and c0 + frame_w < w:
                r1 = r0 + frame_h - 1
                c1 = c0 + frame_w - 1
                if not occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                    grid[r0, c0 : c1 + 1] = grey
                    grid[r1, c0 : c1 + 1] = grey
                    grid[r0:r1 + 1, c0] = grey
                    grid[r0:r1 + 1, c1] = grey
                    gap_candidates = [
                        (r0, c0 + frame_w // 2),
                        (r0 + frame_h // 2, c1),
                        (r1, c0 + frame_w // 2),
                        (r0 + frame_h // 2, c0),
                    ]
                    gap_r, gap_c = gap_candidates[gap_side]
                    grid[gap_r, gap_c] = bg
                    seed_r = r0 + 3
                    seed_c = c0 + 3
                    grid[seed_r, seed_c] = color
                    occupied[r0:r1 + 1, c0:c1 + 1] = True

    return grid


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _iter_components_same_color(grid: np.ndarray, start: Tuple[int, int], color: int, forbid: np.ndarray | None = None) -> List[Tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w):
        return []
    if grid[sr, sc] != color:
        return []

    q = deque([(sr, sc)])
    comp = [(sr, sc)]
    visited = {(sr, sc)}

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if forbid is not None and forbid[nr, nc]:
                continue
            if (nr, nc) in visited:
                continue
            if grid[nr, nc] != color:
                continue
            visited.add((nr, nc))
            q.append((nr, nc))
            comp.append((nr, nc))
    return comp


def _find_frames(grid: np.ndarray, bg: int) -> List[Dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    frames: List[Dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == bg:
                continue

            color = int(grid[r, c])
            comp = _iter_components_same_color(grid, (r, c), color)
            for cr, cc in comp:
                visited[cr, cc] = True

            comp_set = set(comp)
            if len(comp_set) < 8:
                continue

            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            hh, ww = r1 - r0 + 1, c1 - c0 + 1
            if hh < 3 or ww < 3 or (hh % 2 == 0) or (ww % 2 == 0):
                continue

            perimeter = 2 * hh + 2 * ww - 4
            if len(comp_set) != perimeter - 1:
                continue

            missing: List[Tuple[int, int]] = []
            for rr in range(r0, r1 + 1):
                if (rr, c0) not in comp_set:
                    missing.append((rr, c0))
                if (rr, c1) not in comp_set:
                    missing.append((rr, c1))
            for cc in range(c0 + 1, c1):
                if (r0, cc) not in comp_set:
                    missing.append((r0, cc))
                if (r1, cc) not in comp_set:
                    missing.append((r1, cc))

            # Keep exact one border gap.
            if len(missing) != 1:
                continue
            mr, mc = missing[0]

            if mr == r0:
                if mc != c0 + ww // 2:
                    continue
                gap = "top"
            elif mr == r1:
                if mc != c0 + ww // 2:
                    continue
                gap = "bottom"
            elif mc == c0:
                if mr != r0 + hh // 2:
                    continue
                gap = "left"
            elif mc == c1:
                if mr != r0 + hh // 2:
                    continue
                gap = "right"
            else:
                continue

            # Interior must contain exactly one seed pixel and nothing else.
            seed_positions = []
            for ir in range(r0 + 1, r1):
                for ic in range(c0 + 1, c1):
                    v = int(grid[ir, ic])
                    if v != bg:
                        if v == color:
                            continue
                        seed_positions.append((ir, ic, v))

            if len(seed_positions) != 1:
                continue

            sr, sc, seed_color = seed_positions[0]
            # Seed should be near geometric center.
            if sr != (r0 + r1) // 2 or sc != (c0 + c1) // 2:
                continue

            frames.append(
                {
                    "frame_color": color,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "gap_side": gap,
                    "seed": (sr, sc),
                    "seed_color": seed_color,
                }
            )

    return frames


def _collect_templates(
    grid: np.ndarray,
    bg: int,
    frame_colors: Set[int],
    frames: Sequence[Dict[str, object]],
) -> Dict[int, np.ndarray]:
    h, w = grid.shape
    forbidden = np.zeros((h, w), dtype=bool)

    for frame in frames:
        forbidden[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]] = True

    templates: Dict[int, np.ndarray] = {}
    visited = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if (
                color == bg
                or color in frame_colors
                or forbidden[r, c]
                or visited[r, c]
            ):
                continue

            comp = _iter_components_same_color(grid, (r, c), color, forbid=forbidden)
            for cr, cc in comp:
                visited[cr, cc] = True

            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            hh, ww = r1 - r0 + 1, c1 - c0 + 1
            if hh != 3 or ww != 3:
                continue

            mask = np.zeros((3, 3), dtype=bool)
            for rr, cc in comp:
                mask[rr - r0, cc - c0] = True

            if np.count_nonzero(mask) < 1:
                continue
            # Keep patterns visually informative by rejecting full symmetry under 90/180/270.
            if np.array_equal(mask, np.rot90(mask, 1)) and np.array_equal(mask, np.rot90(mask, 2)) and np.array_equal(mask, np.rot90(mask, 3)):
                continue

            if color not in templates:
                templates[color] = mask

    return templates


def _rotate_template_by_gap(pattern: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return pattern.copy()
    if side == "right":
        return np.rot90(pattern, -1)
    if side == "bottom":
        return np.rot90(pattern, 2)
    if side == "left":
        return np.rot90(pattern, 1)
    return pattern.copy()


def _random_template_mask(rng: np.random.Generator, used: Sequence[np.ndarray]) -> np.ndarray:
    base_candidates = [
        np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 0, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 0],
            ],
            dtype=int,
        ),
    ]

    # deterministic fallback pool with intentional asymmetry.
    for mask in base_candidates:
        if _mask_is_distinct(mask, used):
            return mask.astype(bool)

    for _ in range(1000):
        mask = np.zeros((3, 3), dtype=int)
        mask[1, 1] = 1
        for rr in range(3):
            for cc in range(3):
                if (rr, cc) == (1, 1):
                    continue
                if rng.random() < 0.45:
                    mask[rr, cc] = 1
        if _has_full_rotational_asymmetry(mask) and _mask_is_distinct(mask, used):
            return mask.astype(bool)

    return np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _has_full_rotational_asymmetry(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, np.rot90(mask, 1))
        or np.array_equal(mask, np.rot90(mask, 2))
        or np.array_equal(mask, np.rot90(mask, 3))
    )


def _mask_is_distinct(mask: np.ndarray, used: Sequence[np.ndarray]) -> bool:
    for u in used:
        if np.array_equal(mask, u):
            return False
    return True


def _corner_template_positions(h: int, w: int, corner: int) -> Tuple[int, int, int, int]:
    if corner == 0:  # top-left
        return 0, 0, min(12, h), min(12, w)
    if corner == 1:  # top-right
        return 0, max(0, w - 12), min(12, h), min(12, w)
    if corner == 2:  # bottom-left
        return max(0, h - 12), 0, min(12, h), min(12, w)
    return max(0, h - 12), max(0, w - 12), min(12, h), min(12, w)


def _template_slot_origin(r: int, c: int, corner: int) -> Tuple[int, int]:
    if corner == 0:
        return r + 1, c + 1
    if corner == 1:
        return r + 1, c + 1
    if corner == 2:
        return r + 1, c + 1
    return r + 1, c + 1

