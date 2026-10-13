from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
NOISE_COLORS = (Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
SIDES = ("up", "down", "left", "right")
GEN_ATTEMPTS = 500


def _touches_left_edge(coords: np.ndarray) -> bool:
    return bool(np.any(coords[:, 1] == 0))


def _touches_any_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _scan_black_windows(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    out: list[tuple[int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            if bool(np.all(grid[r : r + 3, c : c + 3] == BG)):
                out.append((int(r), int(c)))
    return out


def _fill_black_windows(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = map(int, out.shape)
    for r in range(h - 2):
        for c in range(w - 2):
            if bool(np.all(out[r : r + 3, c : c + 3] == BG)):
                out[r : r + 3, c : c + 3] = BLUE
    return out


def _identify_red_parts(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[RED],
        return_format="dict",
    )
    assert len(comps) == 2, "expected exactly two red components"
    left = [obj for obj in comps if _touches_left_edge(obj["coords"])]
    assert len(left) == 1, "expected exactly one left-edge red fragment"
    source = left[0]["coords"]
    anchors = [obj for obj in comps if obj is not left[0]]
    assert len(anchors) == 1, "expected exactly one red anchor"
    anchor = anchors[0]["coords"]
    assert not _touches_any_edge(anchor, grid.shape), "anchor must stay away from the border"
    return source, anchor


def _classify_side(target: np.ndarray, bbox: tuple[int, int, int, int]) -> str | None:
    ur0, uc0, ur1, uc1 = map(int, bbox)
    tr0, tc0, tr1, tc1 = bbox_coords(np.asarray(target, dtype=int))
    if tr0 == ur0 and tr1 < ur1 and tc0 == uc0 and tc1 == uc1:
        return "up"
    if tr1 == ur1 and tr0 > ur0 and tc0 == uc0 and tc1 == uc1:
        return "down"
    if tc0 == uc0 and tc1 < uc1 and tr0 == ur0 and tr1 == ur1:
        return "left"
    if tc1 == uc1 and tc0 > uc0 and tr0 == ur0 and tr1 == ur1:
        return "right"
    return None


def _candidate_translations(grid: np.ndarray, source: np.ndarray, anchor: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    anchor_set = {tuple(map(int, rc)) for rc in anchor}
    source = np.asarray(source, dtype=int)
    candidates: list[dict] = []
    seen_targets: set[tuple[tuple[int, int], ...]] = set()
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            if dr == 0 and dc == 0:
                continue
            target = source + np.array([dr, dc], dtype=int)
            if int(target[:, 0].min()) < 0 or int(target[:, 1].min()) < 0:
                continue
            if int(target[:, 0].max()) >= h or int(target[:, 1].max()) >= w:
                continue

            ok = True
            for rr, cc in target:
                cell = int(grid[int(rr), int(cc)])
                if cell == RED and (int(rr), int(cc)) in anchor_set:
                    continue
                if cell != BG:
                    ok = False
                    break
            if not ok:
                continue

            target_set = {tuple(map(int, rc)) for rc in target}
            union = anchor_set | target_set
            if len(union) <= len(anchor_set):
                continue

            union_arr = np.array(sorted(union), dtype=int)
            bbox = bbox_coords(union_arr)
            area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            if area != len(union):
                continue

            side = _classify_side(target, bbox)
            if side is None:
                continue

            target_key = tuple(sorted(target_set))
            if target_key in seen_targets:
                continue
            seen_targets.add(target_key)
            candidates.append(
                {
                    "dr": int(dr),
                    "dc": int(dc),
                    "side": side,
                    "target": target,
                    "target_set": target_set,
                    "bbox": bbox,
                }
            )
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill black 3x3 holes blue, then move the left-edge red fragment to the unique rectangle-completing position that remains unblocked.

    Input structure:
    - The background is black, with 2-3 embedded all-black 3x3 windows and scattered non-blue noise pixels.
    - Red appears in exactly two 4-connected objects: one touches the left edge and one internal anchor touches no edge.
    - The anchor and fragment can be combined by translation so their union becomes a solid axis-aligned rectangle.

    Rule:
    - Scan every 3x3 window from top to bottom and left to right; whenever the current window is all black, recolor that whole 3x3 area blue.
    - On the filled grid, treat blue cells and noise as obstacles, then enumerate rigid translations of the left-edge red fragment.
    - A translation is legal only if every target cell stays in bounds, lands on black or anchor-red cells, and the translated fragment together with the anchor forms a filled rectangle.
    - Choose the unique legal translated position, erase the fragment from its original place, and paint it at that target while preserving the blue fills, anchor, and noise.

    Key insight:
    - The black-hole fill changes traversable space first, so rectangle completion must be solved after the blue obstacles are created rather than on the raw input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BG, "black must remain the background"

    filled = _fill_black_windows(g)
    source, anchor = _identify_red_parts(filled)
    candidates = _candidate_translations(filled, source, anchor)
    assert len(candidates) == 1, "expected a unique legal rectangle-completing translation"

    out = filled.copy()
    out[source[:, 0], source[:, 1]] = BG
    target = candidates[0]["target"]
    out[target[:, 0], target[:, 1]] = RED
    return out


def _bar_coords(length: int, vertical: bool) -> np.ndarray:
    if vertical:
        return np.array([(r, 0) for r in range(length)], dtype=int)
    return np.array([(0, c) for c in range(length)], dtype=int)


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _window_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, top_left)
    return {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}


def _desired_setup(h: int, w: int, side: str) -> dict | None:
    length = random.randint(3, 5)
    vertical = side in {"left", "right"}

    if vertical:
        if h < length + 8 or w < 10:
            return None
        if side == "right":
            ar = random.randint(3, h - length - 4)
            ac = random.randint(4, w - 5)
            holes = [(ar - 3, ac - 1), (ar + length, ac - 1), (ar, ac - 3)]
            desired_shift = (ar, ac + 1)
        else:
            ar = random.randint(3, h - length - 4)
            ac = random.randint(5, w - 4)
            holes = [(ar - 3, ac - 1), (ar + length, ac - 1), (ar, ac + 1)]
            desired_shift = (ar, ac - 1)

        anchor = _bar_coords(length, vertical=True) + np.array([ar, ac], dtype=int)
        source_row_choices = [r for r in range(1, h - length - 1) if r + length - 1 < ar - 1 or r > ar + length]
        if not source_row_choices:
            return None
        sr = random.choice(source_row_choices)
        source = _bar_coords(length, vertical=True) + np.array([sr, 0], dtype=int)
        desired_target = _bar_coords(length, vertical=True) + np.array(desired_shift, dtype=int)
    else:
        if h < 10 or w < length + 8:
            return None
        if side == "up":
            ar = random.randint(4, h - 5)
            ac = random.randint(3, w - length - 4)
            holes = [(ar - 1, ac - 3), (ar - 1, ac + length), (ar + 1, ac)]
            desired_shift = (ar - 1, ac)
        else:
            ar = random.randint(3, h - 6)
            ac = random.randint(3, w - length - 4)
            holes = [(ar - 1, ac - 3), (ar - 1, ac + length), (ar - 3, ac)]
            desired_shift = (ar + 1, ac)

        anchor = _bar_coords(length, vertical=False) + np.array([ar, ac], dtype=int)
        source_row_choices = [r for r in range(1, h - 2) if abs(r - ar) >= 2]
        if not source_row_choices:
            return None
        sr = random.choice(source_row_choices)
        source = _bar_coords(length, vertical=False) + np.array([sr, 0], dtype=int)
        desired_target = _bar_coords(length, vertical=False) + np.array(desired_shift, dtype=int)

    hole_sets = [_window_cells(pos) for pos in holes]
    red_set = {tuple(map(int, rc)) for rc in np.vstack([source, anchor])}
    if any(red_set & cells for cells in hole_sets):
        return None
    if len(set(holes)) != len(holes):
        return None
    if any(
        not (0 <= r0 <= h - 3 and 0 <= c0 <= w - 3)
        for r0, c0 in holes
    ):
        return None

    return {
        "source": source,
        "anchor": anchor,
        "holes": holes,
        "desired_side": side,
        "desired_target": desired_target,
    }


def _extra_window_breakers(
    grid: np.ndarray,
    reserved_black: set[tuple[int, int]],
    hole_tops: set[tuple[int, int]],
) -> bool:
    h, w = map(int, grid.shape)
    while True:
        extras = [
            (r, c)
            for r in range(h - 2)
            for c in range(w - 2)
            if (r, c) not in hole_tops and bool(np.all(grid[r : r + 3, c : c + 3] == BG))
        ]
        if not extras:
            return True

        best_cell: tuple[int, int] | None = None
        best_score = -1
        for r0, c0 in extras:
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    cell = (rr, cc)
                    if cell in reserved_black or int(grid[rr, cc]) != BG:
                        continue
                    score = 0
                    for sr in range(max(0, rr - 2), min(h - 2, rr) + 1):
                        for sc in range(max(0, cc - 2), min(w - 2, cc) + 1):
                            if (sr, sc) in hole_tops:
                                continue
                            if bool(np.all(grid[sr : sr + 3, sc : sc + 3] == BG)):
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_cell = cell
        if best_cell is None:
            return False
        rr, cc = best_cell
        grid[rr, cc] = random.choice(NOISE_COLORS)


def _add_noise(
    grid: np.ndarray,
    reserved_black: set[tuple[int, int]],
    preferred: list[tuple[int, int]],
    target_noise: int,
) -> None:
    current = int(np.sum((grid != BG) & (grid != RED)))
    if current >= target_noise:
        return

    preferred_cells = [cell for cell in preferred if int(grid[cell]) == BG and cell not in reserved_black]
    random.shuffle(preferred_cells)
    for cell in preferred_cells:
        if current >= target_noise:
            return
        grid[cell] = random.choice(NOISE_COLORS)
        current += 1

    pool = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if int(grid[r, c]) == BG and (r, c) not in reserved_black]
    random.shuffle(pool)
    for cell in pool:
        if current >= target_noise:
            return
        grid[cell] = random.choice(NOISE_COLORS)
        current += 1


def _candidate_signature(cand: dict) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(cand["target_set"]))


def _build_attempt() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(16, 24)
    side = random.choice(SIDES)
    spec = _desired_setup(h, w, side)
    if spec is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    _paint_cells(grid, spec["source"], RED)
    _paint_cells(grid, spec["anchor"], RED)

    source, anchor = _identify_red_parts(grid)
    pre_candidates = _candidate_translations(grid, source, anchor)
    desired_key = tuple(sorted({tuple(map(int, rc)) for rc in spec["desired_target"]}))
    desired_candidates = [cand for cand in pre_candidates if _candidate_signature(cand) == desired_key]
    if len(desired_candidates) != 1:
        return None

    blocked_prefill = [cand for cand in pre_candidates if _candidate_signature(cand) != desired_key]
    if not blocked_prefill:
        return None

    hole_tops = set(spec["holes"])
    hole_cells = set().union(*(_window_cells(pos) for pos in spec["holes"]))
    desired_target_set = {tuple(map(int, rc)) for rc in spec["desired_target"]}
    if desired_target_set & hole_cells:
        return None

    blocked_by_holes = [cand for cand in blocked_prefill if cand["target_set"] & hole_cells]
    if not blocked_by_holes:
        return None

    must_stay_black = {tuple(map(int, rc)) for rc in np.vstack([source, anchor])}
    for cand in pre_candidates:
        must_stay_black |= cand["target_set"]
    must_stay_black |= hole_cells
    must_stay_black -= {tuple(map(int, rc)) for rc in np.vstack([source, anchor])}

    preferred_noise: list[tuple[int, int]] = []
    for r0, c0 in spec["holes"]:
        for rr in range(max(0, r0 - 1), min(h, r0 + 4)):
            for cc in range(max(0, c0 - 1), min(w, c0 + 4)):
                if (rr, cc) not in hole_cells and int(grid[rr, cc]) == BG:
                    preferred_noise.append((rr, cc))

    if not _extra_window_breakers(grid, must_stay_black, hole_tops):
        return None

    area = h * w
    lower = int(np.ceil(area * 0.15))
    upper = int(np.floor(area * 0.25))
    noise_target = random.randint(lower, upper)
    _add_noise(grid, must_stay_black, preferred_noise, noise_target)

    noise_count = int(np.sum((grid != BG) & (grid != RED)))
    if noise_count < lower or noise_count > upper:
        return None

    found_holes = _scan_black_windows(grid)
    if set(found_holes) != hole_tops:
        return None

    source2, anchor2 = _identify_red_parts(grid)
    pre_with_noise = _candidate_translations(grid, source2, anchor2)
    if {_candidate_signature(c) for c in pre_with_noise} != {_candidate_signature(c) for c in pre_candidates}:
        return None

    filled = _fill_black_windows(grid)
    post_candidates = _candidate_translations(filled, source2, anchor2)
    if len(post_candidates) != 1:
        return None
    if _candidate_signature(post_candidates[0]) != desired_key:
        return None

    blue_cells = {tuple(map(int, rc)) for rc in np.argwhere(filled == BLUE)}
    removed = [cand for cand in pre_with_noise if _candidate_signature(cand) != desired_key]
    if not removed:
        return None
    if not any(cand["target_set"] & blue_cells for cand in removed):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid black-hole rectangle-completion input")
