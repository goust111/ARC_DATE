from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
ANCHOR = 8
PALETTE = [1, 2, 3, 4, 6, 7, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(comp, dtype=int))
    return out


def _find_components(grid: np.ndarray, bg: int, colors: list[int] | None = None) -> list[dict]:
    if colors is None:
        colors = sorted(set(map(int, grid.flat)) - {bg})
    objects: list[dict] = []
    for color in colors:
        for coords in _components_mask(grid == color):
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": _bbox_coords(coords),
                }
            )
    return objects


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(sorted(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    while len(cells) < area:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt not in cells:
                frontier.add(nxt)
    arr = np.asarray(sorted(cells), dtype=int)
    for _ in range(random.randrange(4)):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    if random.random() < 0.5:
        arr[:, 1] *= -1
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _column_profiles(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    profiles: dict[int, tuple[int, int]] = {}
    for c in sorted(set(map(int, coords[:, 1]))):
        rows = coords[coords[:, 1] == c][:, 0]
        profiles[int(c)] = (int(rows.min()), int(rows.max()))
    return profiles


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _extract_anchor(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = _components_mask(grid == ANCHOR)
    assert len(comps) == 1, "expected exactly one anchor component"
    coords = comps[0]
    bbox = _bbox_coords(coords)
    assert coords.shape[0] == 4 and bbox[2] - bbox[0] == 1 and bbox[3] - bbox[1] == 1, "anchor must be 2x2"
    return coords, bbox


def _sort_objects(objects: list[dict], mode: str) -> list[dict]:
    if mode == "correct":
        keys = [(obj["area"], obj["bbox"][0]) for obj in objects]
        assert len(keys) == len(set(keys)), "ambiguous same-area same-top-row ordering"
        return sorted(objects, key=lambda obj: (-obj["area"], obj["bbox"][0]))
    if mode == "ascending":
        return sorted(objects, key=lambda obj: (obj["area"], obj["bbox"][0]))
    if mode == "bbox":
        return sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    raise ValueError(mode)


def _solve(
    grid: np.ndarray,
    *,
    sort_mode: str = "correct",
    collision_mode: str = "correct",
    trail_mode: str = "correct",
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    assert bg == BG, "expected black background"
    anchor_coords, anchor_bbox = _extract_anchor(g)

    objects = _find_components(g, bg=bg, colors=[c for c in sorted(set(map(int, g.flat))) if c not in (bg, ANCHOR)])
    assert 3 <= len(objects) <= 5, "expected 3-5 foreground objects"

    out = np.full_like(g, bg)
    _paint_coords(out, anchor_coords, ANCHOR)
    anchor_mask = np.zeros((h, w), dtype=bool)
    anchor_mask[anchor_coords[:, 0], anchor_coords[:, 1]] = True
    fixed_objects = np.zeros((h, w), dtype=bool)

    ordered = _sort_objects(objects, sort_mode)
    meta_objects = []

    for obj in ordered:
        coords = np.asarray(obj["coords"], dtype=int)
        color = int(obj["color"])
        area = int(obj["area"])
        bbox = obj["bbox"]
        profiles = _column_profiles(coords)

        delta = 0
        block_reason = "full"
        for step in range(1, area + 1):
            moved = coords.copy()
            moved[:, 0] += step
            if int(moved[:, 0].max()) >= h:
                block_reason = "bottom"
                break

            hit_anchor = bool(anchor_mask[moved[:, 0], moved[:, 1]].any())
            hit_object = bool(fixed_objects[moved[:, 0], moved[:, 1]].any())
            if collision_mode == "none":
                hit_anchor = False
                hit_object = False
            elif collision_mode == "anchor_only":
                hit_object = False

            if hit_anchor or hit_object:
                block_reason = "anchor" if hit_anchor and not hit_object else "object"
                if hit_anchor and hit_object:
                    block_reason = "object"
                break
            delta = step

        final_coords = coords.copy()
        final_coords[:, 0] += delta
        _paint_coords(out, final_coords, color)
        fixed_objects[final_coords[:, 0], final_coords[:, 1]] = True

        for c, (top0, bottom0) in profiles.items():
            if trail_mode == "none":
                continue
            if trail_mode == "correct":
                start = bottom0
                stop = top0 + delta
            elif trail_mode == "absolute":
                start = top0
                stop = bottom0 + delta
            elif trail_mode == "bbox":
                start = bbox[2]
                stop = bbox[0] + delta
            else:
                raise ValueError(trail_mode)

            for r in range(start + 1, stop):
                if 0 <= r < h and out[r, c] == bg:
                    parity = (r - start) % 2
                    out[r, c] = color if parity == 0 else GREY

        meta_objects.append(
            {
                "color": color,
                "area": area,
                "bbox": bbox,
                "delta": int(delta),
                "block_reason": block_reason,
                "final_coords": final_coords,
                "orig_coords": coords,
                "profiles": profiles,
            }
        )

    meta = {
        "objects": meta_objects,
        "anchor_bbox": anchor_bbox,
        "shape": (h, w),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each foreground object downward by its area, with larger objects moving first and leaving alternating color/grey tails.

    Input structure:
    - The grid has a black background, one teal 2x2 anchor square near the bottom, and 3-5 separate monochromatic 4-connected objects in the upper half.
    - Each non-anchor object has a distinct color and a small irregular polyomino shape.
    - Grey does not encode an input object; it is reserved for output tails.

    Rule:
    - Compute each object's area and sort the objects by descending area; if two areas tie, the object whose bounding box starts on the higher row moves first.
    - Process the objects in that order. Each object tries to translate straight downward by exactly its own area in rows, but it stops at the last legal position before it would overlap the fixed teal anchor or any object that has already finished moving.
    - After an object stops, keep its translated shape and fill the empty cells strictly between its original lowest cell and translated highest cell in each occupied column with an alternating trail: odd offsets from the original column-bottom become grey, even offsets become the object's own color.
    - The anchor stays unchanged, and trails never overwrite non-black cells.

    Key insight:
    - Area controls three linked decisions at once: motion distance, processing priority, and therefore which moved objects become blockers for later ones, so the same size statistic also determines the final tail lengths and collision pattern.
    """
    return _solve(grid)[0]


def _visited_path(obj: dict) -> set[tuple[int, int]]:
    coords = np.asarray(obj["coords"], dtype=int)
    area = int(obj["area"])
    visited: set[tuple[int, int]] = set()
    for step in range(1, area + 1):
        moved = coords.copy()
        moved[:, 0] += step
        visited |= {tuple(map(int, cell)) for cell in moved}
    return visited


def _choose_areas(n: int, want_tie: bool) -> list[int]:
    if want_tie and n >= 4:
        areas = random.sample([3, 4, 5, 6, 7], n)
        idx1, idx2 = random.sample(range(n), 2)
        tie_area = random.randint(3, 6)
        areas[idx1] = tie_area
        areas[idx2] = tie_area
        if len(set(areas)) == 1:
            areas[random.randrange(n)] = 7
        return areas
    return random.sample([3, 4, 5, 6, 7], n)


def _shape_fits(coords: np.ndarray, top: int, left: int, h: int, w: int, upper_band: int) -> bool:
    placed = coords + np.array([top, left])
    return (
        int(placed[:, 0].min()) >= 0
        and int(placed[:, 1].min()) >= 0
        and int(placed[:, 0].max()) < upper_band
        and int(placed[:, 1].max()) < w
    )


def _try_place(
    occupied: np.ndarray,
    coords: np.ndarray,
    h: int,
    w: int,
    upper_band: int,
    *,
    left_pref: int | None = None,
    top_range: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    coords = np.asarray(coords, dtype=int)
    local_h = int(coords[:, 0].max()) + 1
    local_w = int(coords[:, 1].max()) + 1
    top_lo = 1
    top_hi = upper_band - local_h
    if top_range is not None:
        top_lo = max(top_lo, int(top_range[0]))
        top_hi = min(top_hi, int(top_range[1]))
    if top_lo > top_hi:
        return None

    left_candidates = list(range(0, w - local_w + 1))
    if left_pref is not None:
        left_candidates.sort(key=lambda c: (abs(c - left_pref), random.random()))
    else:
        random.shuffle(left_candidates)

    tops = list(range(top_lo, top_hi + 1))
    random.shuffle(tops)
    for top in tops:
        for left in left_candidates:
            placed = coords + np.array([top, left])
            if occupied[placed[:, 0], placed[:, 1]].any():
                continue
            return int(top), int(left)
    return None


def _scenario_spec(n: int) -> tuple[str, bool]:
    scenario = random.choices(["object", "anchor", "mixed"], weights=[3, 2, 5])[0]
    want_tie = n >= 4 and random.random() < 0.4
    return scenario, want_tie


def _generate_candidate() -> tuple[np.ndarray, str, bool] | None:
    h = random.randint(12, 16)
    w = random.randint(17, 22)
    n = random.randint(3, 5)
    scenario, want_tie = _scenario_spec(n)
    upper_band = h // 2 + 1

    anchor_c = random.randint(max(1, w // 3), w - 4)
    anchor_r = h - random.choice([4, 3])

    areas = _choose_areas(n, want_tie)
    colors = random.sample(PALETTE, n)
    shapes = []
    for area in areas:
        for _ in range(30):
            shape = _random_polyomino(area)
            shape_h = int(shape[:, 0].max()) + 1
            shape_w = int(shape[:, 1].max()) + 1
            if 2 <= shape_h <= 4 and 2 <= shape_w <= 5:
                shapes.append(shape)
                break
        else:
            return None

    biggest = max(range(n), key=lambda i: (areas[i], -i))
    smaller_choices = [i for i in range(n) if areas[i] < areas[biggest]]
    if not smaller_choices:
        return None
    blocker_target = min(smaller_choices, key=lambda i: areas[i])
    anchor_target = random.choice([i for i in range(n) if i != biggest])

    grid = np.full((h, w), BG, dtype=int)
    grid[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = ANCHOR
    occupied = grid != BG

    placements: dict[int, tuple[int, int]] = {}
    lane1 = random.randint(1, max(1, w // 3))
    lane2 = max(0, min(w - 5, anchor_c + random.choice([-2, -1, 0, 1])))

    first = _try_place(
        occupied,
        shapes[biggest],
        h,
        w,
        upper_band,
        left_pref=lane1,
        top_range=(1, 2),
    )
    if first is None:
        return None
    placements[biggest] = first
    big_cells = shapes[biggest] + np.array(first)
    occupied[big_cells[:, 0], big_cells[:, 1]] = True

    second_range = (first[0] + int(shapes[biggest][:, 0].max()) + 1, upper_band - 2)
    second_left_pref = first[1] + random.choice([-1, 0, 1])
    if scenario == "anchor":
        second_left_pref = min(w - 5, max(0, lane2 + 5))
    second = _try_place(
        occupied,
        shapes[blocker_target],
        h,
        w,
        upper_band,
        left_pref=second_left_pref,
        top_range=second_range,
    )
    if second is None:
        return None
    placements[blocker_target] = second
    cells = shapes[blocker_target] + np.array(second)
    occupied[cells[:, 0], cells[:, 1]] = True

    third_left_pref = lane2
    third_top_range = (max(1, upper_band - 4), upper_band - 1)
    if scenario == "object":
        third_left_pref = max(0, min(w - 4, lane1 + 5))
        third_top_range = (1, upper_band - 1)
    third = _try_place(
        occupied,
        shapes[anchor_target],
        h,
        w,
        upper_band,
        left_pref=third_left_pref,
        top_range=third_top_range,
    )
    if third is None:
        return None
    placements[anchor_target] = third
    cells = shapes[anchor_target] + np.array(third)
    occupied[cells[:, 0], cells[:, 1]] = True

    remaining = [i for i in range(n) if i not in placements]
    lane3 = random.randint(max(0, anchor_c - 5), min(w - 4, anchor_c + 2))
    for i in remaining:
        prefs = [lane1, lane2, lane3, None]
        if scenario == "anchor":
            prefs = [lane2 + 4, lane3 + 3, None]
        elif scenario == "object":
            prefs = [lane1, lane1 + 1, lane3, None]
        pos = _try_place(
            occupied,
            shapes[i],
            h,
            w,
            upper_band,
            left_pref=random.choice(prefs),
            top_range=(1, upper_band - 1),
        )
        if pos is None:
            return None
        placements[i] = pos
        cells = shapes[i] + np.array(pos)
        occupied[cells[:, 0], cells[:, 1]] = True

    for i in range(n):
        coords = shapes[i] + np.array(placements[i])
        grid[coords[:, 0], coords[:, 1]] = colors[i]
    return grid, scenario, want_tie


def _semantic_checks(grid: np.ndarray, scenario: str, want_tie: bool) -> bool:
    try:
        correct, meta = _solve(grid)
    except Exception:
        return False

    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if max(h, w) > 30:
        return False
    if np.any(g == GREY):
        return False

    objs = meta["objects"]
    if not (3 <= len(objs) <= 5):
        return False
    if not all(3 <= obj["area"] <= 7 for obj in objs):
        return False

    upper_band = h // 2 + 1
    if any(obj["bbox"][2] >= upper_band for obj in objs):
        return False

    trail_pixels = int(np.count_nonzero(correct == GREY))
    colored_trail = int(np.count_nonzero((correct != g) & (correct != BG) & (correct != ANCHOR) & (correct != GREY)))
    if trail_pixels < 2 or colored_trail < 2:
        return False

    blocked_object = sum(obj["block_reason"] == "object" for obj in objs)
    blocked_anchor = sum(obj["block_reason"] == "anchor" for obj in objs)
    if blocked_object + blocked_anchor == 0:
        return False
    if scenario == "anchor" and (blocked_anchor == 0 or blocked_object != 0):
        return False
    if scenario == "object" and (blocked_object == 0 or blocked_anchor != 0):
        return False
    if scenario == "mixed" and (blocked_object == 0 or blocked_anchor == 0):
        return False

    input_objs = _find_components(g, bg=BG, colors=[c for c in sorted(set(map(int, g.flat))) if c not in (BG, ANCHOR)])
    visited_sets = [_visited_path(obj) for obj in input_objs]
    potential_pairs = 0
    for i in range(len(visited_sets)):
        for j in range(i + 1, len(visited_sets)):
            if visited_sets[i] & visited_sets[j]:
                potential_pairs += 1
    if potential_pairs < 1:
        return False

    wrong_variants = [
        _solve(grid, sort_mode="ascending")[0],
        _solve(grid, sort_mode="bbox")[0],
        _solve(grid, collision_mode="anchor_only")[0],
        _solve(grid, collision_mode="none")[0],
        _solve(grid, trail_mode="none")[0],
        _solve(grid, trail_mode="absolute")[0],
        _solve(grid, trail_mode="bbox")[0],
    ]
    if any(np.array_equal(correct, wrong) for wrong in wrong_variants):
        return False

    density = np.count_nonzero(correct != BG) / float(h * w)
    if density > 0.4:
        return False

    areas = [obj["area"] for obj in objs]
    if len(set(areas)) == 1:
        return False
    if want_tie and len(areas) == len(set(areas)):
        return False
    if not want_tie and len(areas) != len(set(areas)):
        return False
    ties = [area for area in set(areas) if areas.count(area) > 1]
    if ties:
        tied = [obj for obj in objs if obj["area"] in ties]
        tops = [(obj["area"], obj["bbox"][0]) for obj in tied]
        if len(tops) != len(set(tops)):
            return False

    anchor_r0 = meta["anchor_bbox"][0]
    if anchor_r0 < h - 5:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is None:
            continue
        grid, scenario, want_tie = candidate
        if _semantic_checks(grid, scenario, want_tie):
            return grid
    raise RuntimeError("failed to generate a valid area-slide tail input")
