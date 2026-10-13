from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIRS = ("top", "bottom", "left", "right")
DIR_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GAP_LOCAL = {
    "top": (0, 2),
    "bottom": (4, 2),
    "left": (2, 0),
    "right": (2, 4),
}
FRAME_TEMPLATE = {
    side: {
        (r, c)
        for r in range(5)
        for c in range(5)
        if (r in (0, 4) or c in (0, 4)) and (r, c) != gap
    }
    for side, gap in GAP_LOCAL.items()
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == BLACK or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            rmin = int(arr[:, 0].min())
            cmin = int(arr[:, 1].min())
            rmax = int(arr[:, 0].max())
            cmax = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (rmin, cmin, rmax, cmax),
                }
            )

    return comps


def _frame_side(component: dict[str, object]) -> str | None:
    if int(component["area"]) != 15:
        return None

    r0, c0, r1, c1 = component["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
        return None

    local = {(int(r) - r0, int(c) - c0) for r, c in component["coords"]}
    for side, expected in FRAME_TEMPLATE.items():
        if local == expected:
            return side
    return None


def _target_anchor(
    frame_bbox: tuple[int, int, int, int],
    pattern_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int]:
    fr0, fc0, _fr1, _fc1 = frame_bbox
    pr0, pc0, pr1, pc1 = pattern_bbox
    ph = int(pr1 - pr0 + 1)
    pw = int(pc1 - pc0 + 1)
    inner_r0 = fr0 + 1
    inner_c0 = fc0 + 1
    gap_r = fr0 + 2
    gap_c = fc0 + 2

    if side == "top":
        return inner_r0, gap_c - pw // 2
    if side == "bottom":
        return inner_r0 + (3 - ph), gap_c - pw // 2
    if side == "left":
        return gap_r - ph // 2, inner_c0
    if side == "right":
        return gap_r - ph // 2, inner_c0 + (3 - pw)
    raise AssertionError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each outside pattern through a frame's gap into that frame's hollow 3x3 interior.

    Input structure:
    - The background is black.
    - The grid contains one or more monochrome 5x5 square-ring frames whose border has exactly one one-cell gap at the middle of the top, bottom, left, or right side.
    - On the outward ray of each gap there is a separate monochrome 4-connected pattern of another color with bounding-box width and height at most 3.

    Rule:
    - Detect every 5x5 ring frame and determine which side contains its gap.
    - From the gap, scan outward until the first non-black, non-frame-color pixel appears; take that whole monochrome connected component as the paired pattern.
    - Clear the pattern from its original location and translate it straight inward, opposite the gap direction, until its entire bounding box lies inside the frame's 3x3 interior while staying centered on the gap along the perpendicular axis.
    - Keep every frame pixel, including the gap itself, unchanged.

    Key insight:
    - The missing border pixel is both a structural clue and a direction pointer: it identifies which outside component belongs to the frame and how that component must slide into the hollow center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    components = _find_components(g)
    owner = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(components):
        coords = comp["coords"]
        owner[coords[:, 0], coords[:, 1]] = idx

    frames: list[tuple[int, dict[str, object], str]] = []
    for idx, comp in enumerate(components):
        side = _frame_side(comp)
        if side is not None:
            frames.append((idx, comp, side))

    assert frames, "expected at least one gapped square frame"

    moves: list[tuple[np.ndarray, list[tuple[int, int, int]]]] = []
    used_patterns: set[int] = set()
    for frame_idx, frame, side in frames:
        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        gap_lr, gap_lc = GAP_LOCAL[side]
        gap_r = int(fr0 + gap_lr)
        gap_c = int(fc0 + gap_lc)
        step_r, step_c = DIR_STEP[side]

        rr = gap_r + step_r
        cc = gap_c + step_c
        pattern_idx: int | None = None
        while 0 <= rr < h and 0 <= cc < w:
            cell = int(g[rr, cc])
            if cell != BLACK and cell != int(frame["color"]):
                pattern_idx = int(owner[rr, cc])
                break
            rr += step_r
            cc += step_c

        assert pattern_idx is not None, "each frame must point to an outside pattern"
        assert pattern_idx != frame_idx, "frame ray cannot pair with itself"
        assert pattern_idx not in used_patterns, "a pattern may feed only one frame"
        pattern = components[pattern_idx]
        assert _frame_side(pattern) is None, "paired component must be a pattern, not another frame"

        used_patterns.add(pattern_idx)
        pr0, pc0, pr1, pc1 = pattern["bbox"]
        assert pr1 - pr0 + 1 <= 3 and pc1 - pc0 + 1 <= 3, "pattern bbox must fit within 3x3"

        dest_r0, dest_c0 = _target_anchor(frame["bbox"], pattern["bbox"], side)
        dr = int(dest_r0 - pr0)
        dc = int(dest_c0 - pc0)

        translated: list[tuple[int, int, int]] = []
        for r, c in pattern["coords"]:
            nr = int(r + dr)
            nc = int(c + dc)
            assert fr0 + 1 <= nr <= fr0 + 3 and fc0 + 1 <= nc <= fc0 + 3, "translated pattern must stay inside the frame interior"
            translated.append((nr, nc, int(pattern["color"])))
        moves.append((pattern["coords"], translated))

    out = g.copy()
    for coords, _translated in moves:
        out[coords[:, 0], coords[:, 1]] = BLACK

    planned: dict[tuple[int, int], int] = {}
    for _coords, translated in moves:
        for r, c, color in translated:
            assert out[r, c] == BLACK, "frame interior must be empty before insertion"
            prev = planned.get((r, c))
            assert prev is None or prev == color, "translated patterns must not overlap"
            planned[(r, c)] = color

    for (r, c), color in planned.items():
        out[r, c] = color
    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 30
FRAME_COUNT_MIN = 1
FRAME_COUNT_MAX = 4
DISTANCE_MIN = 1
DISTANCE_MAX = 3
MAX_GENERATION_ATTEMPTS = 400


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


@lru_cache(None)
def _shape_specs(side: str) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    vertical = side in ("top", "bottom")
    dims = (
        [(h, w) for h in range(1, 4) for w in (1, 3)]
        if vertical
        else [(h, w) for h in (1, 3) for w in range(1, 4)]
    )
    specs: list[tuple[int, int, tuple[int, ...]]] = []
    for h, w in dims:
        total = h * w
        mid_r = h // 2
        mid_c = w // 2
        for bits in range(1, 1 << total):
            mask = np.zeros((h, w), dtype=bool)
            for idx in range(total):
                if bits & (1 << idx):
                    mask[idx // w, idx % w] = True
            if not mask.any(axis=0).all() or not mask.any(axis=1).all():
                continue
            if not _mask_connected(mask):
                continue
            if vertical:
                if not mask[:, mid_c].any():
                    continue
                if side == "top" and not mask[h - 1, mid_c]:
                    continue
                if side == "bottom" and not mask[0, mid_c]:
                    continue
            else:
                if not mask[mid_r, :].any():
                    continue
                if side == "left" and not mask[mid_r, w - 1]:
                    continue
                if side == "right" and not mask[mid_r, 0]:
                    continue
            specs.append((h, w, tuple(np.flatnonzero(mask.ravel()).tolist())))
    return tuple(specs)


def _shape_from_spec(spec: tuple[int, int, tuple[int, ...]]) -> np.ndarray:
    h, w, flat = spec
    mask = np.zeros(h * w, dtype=bool)
    mask[list(flat)] = True
    return mask.reshape(h, w)


def _pattern_anchor(frame_r0: int, frame_c0: int, side: str, mask: np.ndarray, distance: int) -> tuple[int, int]:
    h, w = mask.shape
    gap_r = frame_r0 + 2
    gap_c = frame_c0 + 2
    if side == "top":
        return frame_r0 - distance - h + 1, gap_c - w // 2
    if side == "bottom":
        return frame_r0 + 4 + distance, gap_c - w // 2
    if side == "left":
        return gap_r - h // 2, frame_c0 - distance - w + 1
    if side == "right":
        return gap_r - h // 2, frame_c0 + 4 + distance
    raise AssertionError(side)


def _footprint(
    frame_r0: int,
    frame_c0: int,
    pattern_r0: int,
    pattern_c0: int,
    pattern_h: int,
    pattern_w: int,
) -> tuple[int, int, int, int]:
    return (
        min(frame_r0, pattern_r0),
        min(frame_c0, pattern_c0),
        max(frame_r0 + 4, pattern_r0 + pattern_h - 1),
        max(frame_c0 + 4, pattern_c0 + pattern_w - 1),
    )


def _placeable_positions(
    h: int,
    w: int,
    blocked: np.ndarray,
    side: str,
    mask: np.ndarray,
    distance: int,
) -> list[tuple[int, int, int, int, tuple[int, int, int, int]]]:
    ph, pw = mask.shape
    candidates: list[tuple[int, int, int, int, tuple[int, int, int, int]]] = []
    for frame_r0 in range(h - 4):
        for frame_c0 in range(w - 4):
            pattern_r0, pattern_c0 = _pattern_anchor(frame_r0, frame_c0, side, mask, distance)
            if pattern_r0 < 0 or pattern_c0 < 0 or pattern_r0 + ph > h or pattern_c0 + pw > w:
                continue

            fp = _footprint(frame_r0, frame_c0, pattern_r0, pattern_c0, ph, pw)
            rr0 = max(0, fp[0] - 1)
            cc0 = max(0, fp[1] - 1)
            rr1 = min(h - 1, fp[2] + 1)
            cc1 = min(w - 1, fp[3] + 1)
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            candidates.append((frame_r0, frame_c0, pattern_r0, pattern_c0, fp))
    return candidates


def _draw_frame(grid: np.ndarray, frame_r0: int, frame_c0: int, color: int, side: str) -> None:
    for lr, lc in FRAME_TEMPLATE[side]:
        grid[frame_r0 + lr, frame_c0 + lc] = color


def _draw_pattern(grid: np.ndarray, pattern_r0: int, pattern_c0: int, mask: np.ndarray, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[pattern_r0 + int(rr), pattern_c0 + int(cc)] = color


def _reserve(blocked: np.ndarray, fp: tuple[int, int, int, int]) -> None:
    rr0 = max(0, fp[0] - 1)
    cc0 = max(0, fp[1] - 1)
    rr1 = min(blocked.shape[0] - 1, fp[2] + 1)
    cc1 = min(blocked.shape[1] - 1, fp[3] + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _attempt() -> np.ndarray | None:
    frame_count = random.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    min_dim = max(GRID_MIN, 11 + 3 * frame_count)
    h = random.randint(min_dim, GRID_MAX)
    w = random.randint(min_dim, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    directions = random.sample(DIRS, k=frame_count)
    colors = random.sample(PALETTE, 2 * frame_count)
    plans = []
    for idx, side in enumerate(directions):
        spec = random.choice(_shape_specs(side))
        mask = _shape_from_spec(spec)
        distance = random.randint(DISTANCE_MIN, DISTANCE_MAX)
        frame_color = colors[2 * idx]
        pattern_color = colors[2 * idx + 1]
        footprint_size = (
            distance + mask.shape[0] + 4 if side in ("top", "bottom") else 5,
            distance + mask.shape[1] + 4 if side in ("left", "right") else 5,
        )
        plans.append(
            {
                "side": side,
                "mask": mask,
                "distance": distance,
                "frame_color": frame_color,
                "pattern_color": pattern_color,
                "score": max(footprint_size),
            }
        )

    random.shuffle(plans)
    plans.sort(key=lambda plan: int(plan["score"]), reverse=True)

    for plan in plans:
        candidates = _placeable_positions(
            h,
            w,
            blocked,
            str(plan["side"]),
            np.asarray(plan["mask"], dtype=bool),
            int(plan["distance"]),
        )
        if not candidates:
            return None

        frame_r0, frame_c0, pattern_r0, pattern_c0, fp = random.choice(candidates)
        _draw_frame(grid, frame_r0, frame_c0, int(plan["frame_color"]), str(plan["side"]))
        _draw_pattern(grid, pattern_r0, pattern_c0, np.asarray(plan["mask"], dtype=bool), int(plan["pattern_color"]))
        _reserve(blocked, fp)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if np.count_nonzero(out) != np.count_nonzero(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-frame slide input")
