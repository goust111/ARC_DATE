from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
FILL = 4
SIDES = ("left", "right", "above", "below")
GEN_RETRIES = 400
INPUT_MARGIN = 1
PAIR_PADDING = 1
GRID_MIN = 15
GRID_MAX = 22
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
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
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _is_solid_anchor(coords: np.ndarray, grid: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return (
        len(coords) == 4
        and r1 - r0 + 1 == 2
        and c1 - c0 + 1 == 2
        and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    )


def _touches_all_bbox_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_label_components(mask)) == 1


def _pair_mask_from_half(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "lr":
        return np.concatenate([mask, np.fliplr(mask)], axis=1)
    return np.concatenate([mask, np.flipud(mask)], axis=0)


def _hole_regions_from_binary(fg_mask: np.ndarray) -> list[np.ndarray]:
    bg_mask = ~fg_mask
    seen = np.zeros(bg_mask.shape, dtype=bool)
    h, w = bg_mask.shape
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(bg_mask.shape, nr, nc) and bg_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return _label_components(bg_mask & ~seen)


def _half_mask_library() -> tuple[dict[int, list[np.ndarray]], list[np.ndarray]]:
    closing: dict[int, list[np.ndarray]] = {}
    opening: list[np.ndarray] = []
    for h in range(2, 6):
        for w in range(2, 5):
            n = h * w
            for area in range(4, min(7, n) + 1):
                for pick in combinations(range(n), area):
                    mask = np.zeros((h, w), dtype=bool)
                    for idx in pick:
                        mask[idx // w, idx % w] = True
                    if not _touches_all_bbox_edges(mask):
                        continue
                    if not _connected(mask):
                        continue
                    if np.array_equal(mask, np.fliplr(mask)):
                        continue
                    pair = _pair_mask_from_half(mask, "lr")
                    if not _connected(pair):
                        continue
                    holes = _hole_regions_from_binary(pair)
                    if not holes:
                        opening.append(mask)
                        continue
                    if len(holes) != 1:
                        continue
                    area_key = int(len(holes[0]))
                    closing.setdefault(area_key, []).append(mask)
    assert opening, "need at least one non-closing half-mask"
    assert len(closing) >= 2, "need multiple closing half-mask area classes"
    return closing, opening


CLOSING_MASKS, OPEN_MASKS = _half_mask_library()
CLOSING_AREAS = tuple(sorted(CLOSING_MASKS))


def _oriented_mask(base: np.ndarray, side: str) -> np.ndarray:
    mask = base.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if side in ("above", "below"):
        mask = np.rot90(mask, random.choice((1, 3)))
    return mask.copy()


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        cells.add((int(r0 + r), int(c0 + c)))
    return cells


def _mirror_cells(cells: set[tuple[int, int]], anchor_bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    mirrored: set[tuple[int, int]] = set()
    for r, c in cells:
        if side == "left":
            mirrored.add((r, 2 * ac0 - 1 - c))
        elif side == "right":
            mirrored.add((r, 2 * ac1 + 1 - c))
        elif side == "above":
            mirrored.add((2 * ar0 - 1 - r, c))
        else:
            mirrored.add((2 * ar1 + 1 - r, c))
    return mirrored


def _translation_for_side(side: str, gap: int) -> tuple[int, int]:
    if side == "left":
        return 0, gap
    if side == "right":
        return 0, -gap
    if side == "above":
        return gap, 0
    return -gap, 0


def _layout_from_output(mask: np.ndarray, side: str, sr: int, sc: int, gap: int, shift: int):
    h, w = mask.shape
    moved = _coords_from_mask(mask, sr, sc)
    if side == "left":
        ar, ac = sr + shift, sc + w
        ir, ic = sr, sc - gap
    elif side == "right":
        ar, ac = sr + shift, sc - 2
        ir, ic = sr, sc + gap
    elif side == "above":
        ar, ac = sr + h, sc + shift
        ir, ic = sr - gap, sc
    else:
        ar, ac = sr - 2, sc + shift
        ir, ic = sr + gap, sc
    anchor = {(ar + dr, ac + dc) for dr in range(2) for dc in range(2)}
    anchor_bbox = (ar, ac, ar + 1, ac + 1)
    mirrored = _mirror_cells(moved, anchor_bbox, side)
    input_cells = _coords_from_mask(mask, ir, ic)
    step_r = 0 if ir == sr else (1 if sr > ir else -1)
    step_c = 0 if ic == sc else (1 if sc > ic else -1)
    path: set[tuple[int, int]] = set()
    cur_r, cur_c = ir, ic
    for _ in range(gap + 1):
        path |= _coords_from_mask(mask, cur_r, cur_c)
        cur_r += step_r
        cur_c += step_c
    return {
        "input_cells": input_cells,
        "anchor_cells": anchor,
        "moved_cells": moved,
        "mirrored_cells": mirrored,
        "output_cells": moved | mirrored,
        "path_cells": path,
        "anchor_bbox": anchor_bbox,
        "input_origin": (ir, ic),
        "moved_origin": (sr, sc),
        "gap": gap,
        "mask": mask,
        "side": side,
    }


def _inside_margin(cells: set[tuple[int, int]], shape: tuple[int, int], margin: int) -> bool:
    h, w = shape
    return all(margin <= r < h - margin and margin <= c < w - margin for r, c in cells)


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    if radius <= 0:
        return set(cells)
    expanded: set[tuple[int, int]] = set()
    h, w = shape
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _group_candidates(mask: np.ndarray, side: str, shape: tuple[int, int]) -> list[dict]:
    h, w = mask.shape
    H, W = shape
    candidates: list[dict] = []
    gaps = list(range(2, 5))
    random.shuffle(gaps)
    for gap in gaps:
        if side == "left":
            r_range = range(INPUT_MARGIN, H - h - INPUT_MARGIN + 1)
            c_range = range(INPUT_MARGIN + gap, W - 2 * w - INPUT_MARGIN + 1)
            shifts = range(h - 1)
        elif side == "right":
            r_range = range(INPUT_MARGIN, H - h - INPUT_MARGIN + 1)
            c_range = range(w + INPUT_MARGIN, W - w - gap - INPUT_MARGIN + 1)
            shifts = range(h - 1)
        elif side == "above":
            r_range = range(INPUT_MARGIN + gap, H - 2 * h - INPUT_MARGIN + 1)
            c_range = range(INPUT_MARGIN, W - w - INPUT_MARGIN + 1)
            shifts = range(w - 1)
        else:
            r_range = range(h + INPUT_MARGIN, H - h - gap - INPUT_MARGIN + 1)
            c_range = range(INPUT_MARGIN, W - w - INPUT_MARGIN + 1)
            shifts = range(w - 1)
        r_values = list(r_range)
        c_values = list(c_range)
        shift_values = list(shifts)
        random.shuffle(r_values)
        random.shuffle(c_values)
        random.shuffle(shift_values)
        for sr in r_values:
            for sc in c_values:
                for shift in shift_values:
                    layout = _layout_from_output(mask, side, sr, sc, gap, shift)
                    if layout["moved_cells"] & layout["mirrored_cells"]:
                        continue
                    if layout["input_cells"] & layout["anchor_cells"]:
                        continue
                    if not _inside_margin(layout["input_cells"] | layout["anchor_cells"], shape, INPUT_MARGIN):
                        continue
                    if not _inside_margin(layout["output_cells"], shape, INPUT_MARGIN):
                        continue
                    candidates.append(layout)
                    if len(candidates) >= 160:
                        random.shuffle(candidates)
                        return candidates
    return candidates


def _components_by_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _label_components(grid == color)


def _parse_groups(grid: np.ndarray) -> list[dict]:
    colors = [c for c in sorted(set(map(int, np.unique(grid)))) if c not in (BG, FILL)]
    groups: list[dict] = []
    for color in colors:
        comps = _components_by_color(grid, color)
        assert len(comps) == 2, f"color {color} must have exactly two components"
        anchors = [coords for coords in comps if _is_solid_anchor(coords, grid, color)]
        assert len(anchors) == 1, f"color {color} must have one 2x2 anchor"
        anchor = anchors[0]
        sprite = comps[0] if comps[1] is anchor else comps[1]
        area = int(len(sprite))
        assert 4 <= area <= 7, f"sprite area out of range for color {color}"
        sprite_bbox = _bbox(sprite)
        anchor_bbox = _bbox(anchor)
        sr0, sc0, sr1, sc1 = sprite_bbox
        ar0, ac0, ar1, ac1 = anchor_bbox
        candidates: list[tuple[str, int, int]] = []
        if sc1 < ac0 and max(sr0, ar0) <= min(sr1, ar1):
            candidates.append(("left", 0, ac0 - sc1 - 1))
        if sc0 > ac1 and max(sr0, ar0) <= min(sr1, ar1):
            candidates.append(("right", 0, ac1 + 1 - sc0))
        if sr1 < ar0 and max(sc0, ac0) <= min(sc1, ac1):
            candidates.append(("above", ar0 - sr1 - 1, 0))
        if sr0 > ar1 and max(sc0, ac0) <= min(sc1, ac1):
            candidates.append(("below", ar1 + 1 - sr0, 0))
        assert len(candidates) == 1, f"color {color} must have a unique anchor direction"
        side, dr, dc = candidates[0]
        local = _mask_from_coords(sprite)
        if side in ("left", "right"):
            assert not np.array_equal(local, np.fliplr(local)), "sprite must be non-symmetric along the mirror axis"
        else:
            assert not np.array_equal(local, np.flipud(local)), "sprite must be non-symmetric along the mirror axis"
        groups.append(
            {
                "color": color,
                "sprite": sprite,
                "anchor_bbox": anchor_bbox,
                "side": side,
                "dr": dr,
                "dc": dc,
                "gap": abs(dr) + abs(dc),
            }
        )
    assert 2 <= len(groups) <= 3, "expected two or three color groups"
    return groups


def _render_groups(shape: tuple[int, int], groups: list[dict], *, mirror: bool) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for group in groups:
        color = int(group["color"])
        anchor_bbox = group["anchor_bbox"]
        moved: set[tuple[int, int]] = set()
        for r, c in group["sprite"]:
            rr = int(r) + int(group["dr"])
            cc = int(c) + int(group["dc"])
            moved.add((rr, cc))
        for r, c in moved:
            out[r, c] = color
        if mirror:
            for r, c in _mirror_cells(moved, anchor_bbox, group["side"]):
                out[r, c] = color
    return out


def _hole_regions(grid: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(grid.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    h, w = grid.shape
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BG and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return _label_components((grid == BG) & ~seen)


def _region_key(coords: np.ndarray) -> tuple[int, int, int]:
    top_left = tuple(np.min(coords, axis=0))
    return (-int(len(coords)), int(top_left[0]), int(top_left[1]))


def _choose_region(regions: list[np.ndarray]) -> np.ndarray:
    assert regions, "expected at least one enclosed region"
    best = regions[0]
    for region in regions[1:]:
        if _region_key(region) < _region_key(best):
            best = region
    return best


def _path_cells_from_group(group: dict) -> set[tuple[int, int]]:
    dr = int(group["dr"])
    dc = int(group["dc"])
    steps = abs(dr) + abs(dc)
    sprite = {(int(r), int(c)) for r, c in group["sprite"]}
    path: set[tuple[int, int]] = set()
    for t in range(steps + 1):
        rr = dr // steps * t if steps else 0
        cc = dc // steps * t if steps else 0
        for r, c in sprite:
            path.add((r + rr, c + cc))
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each sprite to its same-color square anchor, mirror it across the anchor edge, then fill only the largest enclosed background region.

    Input structure:
    - The grid is a black background with 2 or 3 color groups.
    - Each group contains one irregular 4-connected monochrome sprite of area 4 to 7 and one same-color solid 2x2 anchor square.
    - Within each color group, the sprite and anchor are separated only by black cells along one row-aligned or column-aligned cardinal direction.

    Rule:
    - For each same-color sprite/anchor pair, slide the sprite straight toward the anchor until its facing bbox edge reaches the anchor's near edge.
    - Remove the anchor and place a mirrored copy of the moved sprite across that near edge in the same color.
    - After all groups are transformed, flood-fill black background from the outer border using 4-connectivity.
    - Every unfilled black component is an enclosed region; fill only the region with the largest area using yellow.
    - If multiple enclosed regions share the largest area, choose the one whose top-left pixel is earliest in row-major order.

    Key insight:
    - The anchor only defines where the sprite stops and which mirror axis to use; enclosed regions appear only after that geometric pairing, so the solver must transform first and compare hole areas afterward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    groups = _parse_groups(g)
    base = _render_groups(g.shape, groups, mirror=True)
    regions = _hole_regions(base)
    winner = _choose_region(regions)

    out = base.copy()
    for r, c in winner:
        out[int(r), int(c)] = FILL
    return out


def _build_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for group in placed:
        color = int(group["color"])
        for r, c in group["layout"]["input_cells"] | group["layout"]["anchor_cells"]:
            grid[r, c] = color
    return grid


def _pick_group_specs(n_groups: int) -> list[dict]:
    tie_case = random.random() < 0.3
    specs: list[dict] = []
    if tie_case:
        top_area = random.choice(CLOSING_AREAS)
        specs.append({"kind": "closing", "hole_area": top_area})
        specs.append({"kind": "closing", "hole_area": top_area})
        if n_groups == 3:
            if random.random() < 0.5:
                smaller = random.choice([a for a in CLOSING_AREAS if a <= top_area])
                specs.append({"kind": "closing", "hole_area": smaller})
            else:
                specs.append({"kind": "open"})
    else:
        top_area = random.choice(CLOSING_AREAS[1:] if len(CLOSING_AREAS) > 1 else CLOSING_AREAS)
        smaller_choices = [a for a in CLOSING_AREAS if a < top_area]
        smaller = random.choice(smaller_choices or [top_area])
        specs.append({"kind": "closing", "hole_area": top_area})
        specs.append({"kind": "closing", "hole_area": smaller})
        if n_groups == 3:
            roll = random.random()
            if roll < 0.4:
                specs.append({"kind": "open"})
            elif roll < 0.7:
                specs.append({"kind": "closing", "hole_area": random.choice([a for a in CLOSING_AREAS if a <= smaller] or [smaller])})
            else:
                specs.append({"kind": "closing", "hole_area": random.choice(CLOSING_AREAS)})
    random.shuffle(specs)
    return specs


def _choose_mask(spec: dict) -> np.ndarray:
    if spec["kind"] == "open":
        return random.choice(OPEN_MASKS)
    return random.choice(CLOSING_MASKS[int(spec["hole_area"])])


def _place_groups(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    colors = random.sample(PALETTE, len(specs))
    order = list(range(len(specs)))
    order.sort(key=lambda i: (specs[i]["kind"] != "closing", -int(specs[i].get("hole_area", 0))))

    reserved_input: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()
    reserved_path: set[tuple[int, int]] = set()
    blocked_input: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()
    placed: list[dict | None] = [None] * len(specs)

    for idx in order:
        side = random.choice(SIDES)
        mask = _oriented_mask(_choose_mask(specs[idx]), side)
        candidates = _group_candidates(mask, side, shape)
        if not candidates:
            return None
        picked = None
        for layout in candidates:
            input_cells = layout["input_cells"] | layout["anchor_cells"]
            output_cells = layout["output_cells"]
            path_cells = layout["path_cells"]
            if input_cells & (blocked_input | reserved_path):
                continue
            if path_cells & (blocked_input | reserved_path):
                continue
            if output_cells & blocked_output:
                continue
            picked = {
                "color": colors[idx],
                "layout": layout,
            }
            reserved_input |= input_cells
            reserved_output |= output_cells
            reserved_path |= path_cells
            blocked_input |= _expand(input_cells, shape, PAIR_PADDING)
            blocked_output |= _expand(output_cells, shape, PAIR_PADDING)
            break
        if picked is None:
            return None
        placed[idx] = picked

    assert all(p is not None for p in placed)
    return [p for p in placed if p is not None]


def _generator_constraints(grid: np.ndarray) -> bool:
    groups = _parse_groups(grid)
    if len(groups) not in (2, 3):
        return False
    if any(group["gap"] < 2 for group in groups):
        return False

    base = _render_groups(grid.shape, groups, mirror=True)
    moved_only = _render_groups(grid.shape, groups, mirror=False)
    final_regions = _hole_regions(base)
    if len(final_regions) < 2:
        return False
    if _hole_regions(grid):
        return False
    if _hole_regions(moved_only):
        return False

    winner = _choose_region(final_regions)
    expected = base.copy()
    for r, c in winner:
        expected[int(r), int(c)] = FILL
    if not np.array_equal(main(grid), expected):
        return False

    path_sets = [_path_cells_from_group(group) for group in groups]
    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if path_sets[i] & path_sets[j]:
                return False

    winning_area = max(len(region) for region in final_regions)
    if sum(len(region) == winning_area for region in final_regions) > 1:
        # Explicitly exercise the tie-break condition via row-major top-left choice.
        chosen = tuple(np.min(winner, axis=0))
        tied = [tuple(np.min(region, axis=0)) for region in final_regions if len(region) == winning_area]
        if chosen != min(tied):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_groups = random.choice((2, 3))
    shape = (
        random.randint(max(GRID_MIN, 15 if n_groups == 3 else GRID_MIN), GRID_MAX),
        random.randint(max(GRID_MIN, 15 if n_groups == 3 else GRID_MIN), GRID_MAX),
    )
    specs = _pick_group_specs(n_groups)
    placed = _place_groups(shape, specs)
    if placed is None:
        return None
    grid = _build_input(shape, placed)
    if not _generator_constraints(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
