from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME):
        coords = np.array(comp, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _border_set(r0, c0, r1, c1)
        if len(comp) != len(border) or set(comp) != border:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or bool(np.any(inner == FRAME)):
            continue
        seeds = np.argwhere((inner != BG) & (inner != FRAME))
        if seeds.shape != (1, 2):
            continue
        sr = int(r0 + 1 + seeds[0, 0])
        sc = int(c0 + 1 + seeds[0, 1])
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed": (sr, sc),
                "color": int(grid[sr, sc]),
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _nearest_side(bbox: tuple[int, int, int, int], sr: int, sc: int) -> str:
    r0, c0, r1, c1 = bbox
    distances = [
        ("top", int(sr - r0)),
        ("bottom", int(r1 - sr)),
        ("left", int(sc - c0)),
        ("right", int(c1 - sc)),
    ]
    distances.sort(key=lambda item: (item[1], SIDES.index(item[0])))
    return distances[0][0]


def _gap_cell(bbox: tuple[int, int, int, int], side: str, sr: int, sc: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return int(r0), int(sc)
    if side == "bottom":
        return int(r1), int(sc)
    if side == "left":
        return int(sr), int(c0)
    return int(sr), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each gray frame on the side nearest to its seed and emit a same-colored ray outward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    frames = _find_frames(g)
    assert frames, "expected at least one gray frame"

    out = g.copy()
    grey_mask = g == FRAME

    for frame in frames:
        bbox = frame["bbox"]
        sr, sc = frame["seed"]
        color = int(frame["color"])
        side = _nearest_side(bbox, sr, sc)
        gap_r, gap_c = _gap_cell(bbox, side, sr, sc)
        dr, dc = DIR_BY_SIDE[side]

        out[sr, sc] = BG
        out[gap_r, gap_c] = BG

        rr, cc = gap_r + dr, gap_c + dc
        while 0 <= rr < h and 0 <= cc < w and not grey_mask[rr, cc]:
            out[rr, cc] = color
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
FRAME_SIZE_RANGE = (5, 9)  # outer size
FRAME_COUNT_CHOICES = (1, 2, 3, 4, 5)
FRAME_COUNT_WEIGHTS = (1, 2, 3, 3, 2)
PAIR_COUNT_BY_MAX = {
    0: ((0,), (1,)),
    1: ((0, 1), (2, 3)),
    2: ((0, 1, 2), (2, 4, 1)),
}
PLACEMENT_TRIES = 120
GEN_RETRIES = 500


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _can_place(box: tuple[int, int, int, int], specs: list[dict]) -> bool:
    return all(not _boxes_too_close(box, spec["bbox"]) for spec in specs)


def _random_size() -> tuple[int, int]:
    return (
        random.randint(*FRAME_SIZE_RANGE),
        random.randint(*FRAME_SIZE_RANGE),
    )


def _sample_line_for_source(box: tuple[int, int, int, int], side: str) -> int | None:
    r0, c0, r1, c1 = box
    if side in ("top", "bottom"):
        lo, hi = c0 + 2, c1 - 2
    else:
        lo, hi = r0 + 2, r1 - 2
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _try_add_blocked_pair(h: int, w: int, specs: list[dict]) -> bool:
    for _ in range(PLACEMENT_TRIES):
        side = random.choice(SIDES)
        src_h, src_w = _random_size()
        blk_h, blk_w = _random_size()

        if side == "top":
            if h < src_h + blk_h + 1:
                continue
            src_r0 = random.randint(blk_h + 1, h - src_h)
            src_c0 = random.randint(0, w - src_w)
            src_box = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            line = _sample_line_for_source(src_box, side)
            if line is None:
                continue
            blk_c0_lo = max(0, line - (blk_w - 2))
            blk_c0_hi = min(w - blk_w, line - 1)
            if blk_c0_lo > blk_c0_hi:
                continue
            blk_c0 = random.randint(blk_c0_lo, blk_c0_hi)
            sep = random.randint(1, src_r0 - blk_h)
            blk_r1 = src_r0 - sep - 1
            blk_r0 = blk_r1 - blk_h + 1
            blk_box = (blk_r0, blk_c0, blk_r1, blk_c0 + blk_w - 1)
        elif side == "bottom":
            max_sep_room = h - src_h - blk_h - 1
            if max_sep_room < 0:
                continue
            src_r0 = random.randint(0, max_sep_room)
            src_c0 = random.randint(0, w - src_w)
            src_box = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            line = _sample_line_for_source(src_box, side)
            if line is None:
                continue
            blk_c0_lo = max(0, line - (blk_w - 2))
            blk_c0_hi = min(w - blk_w, line - 1)
            if blk_c0_lo > blk_c0_hi:
                continue
            blk_c0 = random.randint(blk_c0_lo, blk_c0_hi)
            sep = random.randint(1, h - src_h - src_r0 - blk_h)
            blk_r0 = src_r0 + src_h + sep
            blk_box = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c0 + blk_w - 1)
        elif side == "left":
            if w < src_w + blk_w + 1:
                continue
            src_r0 = random.randint(0, h - src_h)
            src_c0 = random.randint(blk_w + 1, w - src_w)
            src_box = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            line = _sample_line_for_source(src_box, side)
            if line is None:
                continue
            blk_r0_lo = max(0, line - (blk_h - 2))
            blk_r0_hi = min(h - blk_h, line - 1)
            if blk_r0_lo > blk_r0_hi:
                continue
            blk_r0 = random.randint(blk_r0_lo, blk_r0_hi)
            sep = random.randint(1, src_c0 - blk_w)
            blk_c1 = src_c0 - sep - 1
            blk_c0 = blk_c1 - blk_w + 1
            blk_box = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c1)
        else:
            max_sep_room = w - src_w - blk_w - 1
            if max_sep_room < 0:
                continue
            src_r0 = random.randint(0, h - src_h)
            src_c0 = random.randint(0, max_sep_room)
            src_box = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            line = _sample_line_for_source(src_box, side)
            if line is None:
                continue
            blk_r0_lo = max(0, line - (blk_h - 2))
            blk_r0_hi = min(h - blk_h, line - 1)
            if blk_r0_lo > blk_r0_hi:
                continue
            blk_r0 = random.randint(blk_r0_lo, blk_r0_hi)
            sep = random.randint(1, w - src_w - src_c0 - blk_w)
            blk_c0 = src_c0 + src_w + sep
            blk_box = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c0 + blk_w - 1)

        if not _can_place(src_box, specs) or not _can_place(blk_box, specs):
            continue
        if _boxes_too_close(src_box, blk_box):
            continue

        specs.append(
            {
                "bbox": src_box,
                "forced_side": side,
                "forced_line": line,
                "forbid_sides": set(),
                "prefer": "blocked",
                "role": "source",
            }
        )
        specs.append(
            {
                "bbox": blk_box,
                "forced_side": None,
                "forced_line": None,
                "forbid_sides": {OPPOSITE_SIDE[side]},
                "prefer": "boundary",
                "role": "blocker",
            }
        )
        return True
    return False


def _try_add_free_frame(h: int, w: int, specs: list[dict]) -> bool:
    for _ in range(PLACEMENT_TRIES):
        fh, fw = _random_size()
        r0 = random.randint(0, h - fh)
        c0 = random.randint(0, w - fw)
        box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if not _can_place(box, specs):
            continue
        specs.append(
            {
                "bbox": box,
                "forced_side": None,
                "forced_line": None,
                "forbid_sides": set(),
                "prefer": "boundary",
                "role": "free",
            }
        )
        return True
    return False


def _is_exact_center(box: tuple[int, int, int, int], sr: int, sc: int) -> bool:
    r0, c0, r1, c1 = box
    return 2 * sr == r0 + r1 and 2 * sc == c0 + c1


def _candidate_seeds(spec: dict) -> list[tuple[int, int, str]]:
    r0, c0, r1, c1 = spec["bbox"]
    forced_side = spec["forced_side"]
    forced_line = spec["forced_line"]
    forbid_sides = spec["forbid_sides"]

    seeds: list[tuple[int, int, str]] = []
    for sr in range(r0 + 1, r1):
        for sc in range(c0 + 1, c1):
            if _is_exact_center(spec["bbox"], sr, sc):
                continue
            distances = {
                "top": int(sr - r0),
                "bottom": int(r1 - sr),
                "left": int(sc - c0),
                "right": int(c1 - sc),
            }
            best_d = min(distances.values())
            best_sides = [side for side, dist in distances.items() if dist == best_d]
            if len(best_sides) != 1:
                continue
            side = best_sides[0]
            if side in forbid_sides:
                continue
            if forced_side is not None and side != forced_side:
                continue
            if forced_line is not None:
                if side in ("top", "bottom") and sc != forced_line:
                    continue
                if side in ("left", "right") and sr != forced_line:
                    continue
            seeds.append((int(sr), int(sc), side))
    return seeds


def _all_grey_cells(specs: list[dict]) -> set[tuple[int, int]]:
    cells = set()
    for spec in specs:
        cells |= _border_set(*spec["bbox"])
    return cells


def _ray_path(
    bbox: tuple[int, int, int, int],
    side: str,
    sr: int,
    sc: int,
    grey_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], str]:
    gap_r, gap_c = _gap_cell(bbox, side, sr, sc)
    dr, dc = DIR_BY_SIDE[side]
    rr, cc = gap_r + dr, gap_c + dc
    ray: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        if (rr, cc) in grey_cells:
            return ray, "blocked"
        ray.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return ray, "boundary"


def _candidate_options(
    spec: dict,
    grey_cells: set[tuple[int, int]],
    used_rays: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[dict]:
    preferred = spec["prefer"]
    valid: list[dict] = []
    preferred_valid: list[dict] = []
    for sr, sc, side in _candidate_seeds(spec):
        ray, stop = _ray_path(spec["bbox"], side, sr, sc, grey_cells, h, w)
        if not ray or any(cell in used_rays for cell in ray):
            continue
        option = {
            "seed": (sr, sc),
            "side": side,
            "ray": ray,
            "stop": stop,
        }
        valid.append(option)
        if stop == preferred or preferred == "any":
            preferred_valid.append(option)
    if preferred == "blocked":
        return preferred_valid
    return preferred_valid or valid


def _assign_seeds_and_rays(specs: list[dict], h: int, w: int) -> bool:
    grey_cells = _all_grey_cells(specs)
    used_rays: set[tuple[int, int]] = set()
    pending = set(range(len(specs)))

    while pending:
        best_idx = None
        best_options = None
        for idx in pending:
            options = _candidate_options(specs[idx], grey_cells, used_rays, h, w)
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_idx = idx
                best_options = options
        assert best_idx is not None and best_options is not None
        choice = random.choice(best_options)
        specs[best_idx]["seed"] = choice["seed"]
        specs[best_idx]["side"] = choice["side"]
        specs[best_idx]["ray"] = choice["ray"]
        specs[best_idx]["stop"] = choice["stop"]
        used_rays.update(choice["ray"])
        pending.remove(best_idx)

    return True


def _render_input(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(PALETTE, k=len(specs))
    for spec, color in zip(specs, colors):
        r0, c0, r1, c1 = spec["bbox"]
        for rr, cc in _border_set(r0, c0, r1, c1):
            grid[rr, cc] = FRAME
        sr, sc = spec["seed"]
        grid[sr, sc] = int(color)
    return grid


def _branch_requirements(specs: list[dict], pair_count: int) -> bool:
    blocked_count = sum(1 for spec in specs if spec["stop"] == "blocked")
    boundary_count = sum(1 for spec in specs if spec["stop"] == "boundary")
    if blocked_count < pair_count:
        return False
    if pair_count > 0 and boundary_count == 0:
        return False
    return True


def _build_specs(h: int, w: int) -> list[dict] | None:
    frame_count = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    max_pairs = min(frame_count // 2, 2)
    pair_choices, pair_weights = PAIR_COUNT_BY_MAX[max_pairs]
    pair_count = random.choices(pair_choices, weights=pair_weights, k=1)[0]

    specs: list[dict] = []
    for _ in range(pair_count):
        if not _try_add_blocked_pair(h, w, specs):
            return None
    while len(specs) < frame_count:
        if not _try_add_free_frame(h, w, specs):
            return None
    if not _assign_seeds_and_rays(specs, h, w):
        return None
    if not _branch_requirements(specs, pair_count):
        return None
    return specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        specs = _build_specs(h, w)
        if specs is None:
            continue
        grid = _render_input(h, w, specs)
        detected = _find_frames(grid)
        if len(detected) != len(specs):
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
