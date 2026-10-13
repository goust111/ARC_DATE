from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [1, 2, 3, 4, 6, 7, 8]
GRID_MIN = 13
GRID_MAX = 16
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in coords) + 1
    return _normalize([(c, h - 1 - r) for r, c in coords])


def _shape_variants() -> list[tuple[tuple[int, int], ...]]:
    seeds = [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ]
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for seed in seeds:
        cur = _normalize(list(seed))
        for _ in range(4):
            if cur not in seen:
                seen.add(cur)
                out.append(cur)
            cur = _rotate(cur)
    return out


SHAPES = _shape_variants()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in coords.tolist()}


def _translate(coords: np.ndarray, dr: int, dc: int, steps: int = 1) -> np.ndarray:
    delta = np.array([dr * steps, dc * steps], dtype=int)
    return coords + delta


def _steps_to_boundary(shape: tuple[int, int], coords: np.ndarray, direction: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    if dr < 0:
        return int(coords[:, 0].min())
    if dr > 0:
        return int(h - 1 - coords[:, 0].max())
    if dc < 0:
        return int(coords[:, 1].min())
    return int(w - 1 - coords[:, 1].max())


def _touching_sides(shape: tuple[int, int], coords: np.ndarray) -> list[str]:
    h, w = shape
    sides = []
    if int(coords[:, 0].min()) == 0:
        sides.append("top")
    if int(coords[:, 0].max()) == h - 1:
        sides.append("bottom")
    if int(coords[:, 1].min()) == 0:
        sides.append("left")
    if int(coords[:, 1].max()) == w - 1:
        sides.append("right")
    return sides


def _expanded(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(coords)
    for r, c in list(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _boundary_place(
    shape: tuple[tuple[int, int], ...], side: str, anchor: int, grid_shape: tuple[int, int]
) -> np.ndarray | None:
    h, w = grid_shape
    arr = np.array(shape, dtype=int)
    max_r = int(arr[:, 0].max())
    max_c = int(arr[:, 1].max())
    if side == "top":
        if anchor < 1 or anchor + max_c > w - 2:
            return None
        placed = arr + np.array([0, anchor], dtype=int)
    elif side == "bottom":
        if anchor < 1 or anchor + max_c > w - 2:
            return None
        placed = arr + np.array([h - 1 - max_r, anchor], dtype=int)
    elif side == "left":
        if anchor < 1 or anchor + max_r > h - 2:
            return None
        placed = arr + np.array([anchor, 0], dtype=int)
    else:
        if anchor < 1 or anchor + max_r > h - 2:
            return None
        placed = arr + np.array([anchor, w - 1 - max_c], dtype=int)
    touched = _touching_sides(grid_shape, placed)
    return placed if touched == [side] else None


def _first_non_overlapping_stack(
    shape: tuple[tuple[int, int], ...],
    side: str,
    anchor: int,
    grid_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    placed = _boundary_place(shape, side, anchor, grid_shape)
    if placed is None:
        return None
    inward = tuple(-v for v in SIDE_TO_DIR[side])
    while True:
        cells = _coords_set(placed)
        if not (cells & occupied):
            return placed
        moved = _translate(placed, inward[0], inward[1], 1)
        if not np.all((0 <= moved[:, 0]) & (moved[:, 0] < grid_shape[0]) & (0 <= moved[:, 1]) & (moved[:, 1] < grid_shape[1])):
            return None
        placed = moved


def _find_lookup_bar(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for r in range(h):
        for c0 in range(w - 3):
            colors = [int(v) for v in g[r, c0 : c0 + 4]]
            if any(v in (BG, GREY, MAROON) for v in colors):
                continue
            if len(set(colors)) != 4:
                continue
            for base_r in (r - 1, r + 1):
                if not (0 <= base_r < h):
                    continue
                if not np.all(g[base_r, c0 : c0 + 4] == GREY):
                    continue
                r0 = min(r, base_r)
                r1 = max(r, base_r)
                touches_horizontal = r0 == 0 or r1 == h - 1
                touches_vertical = c0 == 0 or c0 + 3 == w - 1
                if not (touches_horizontal and touches_vertical):
                    continue
                rect = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c0 + 4)}
                rr0 = max(0, r0 - 1)
                rr1 = min(h - 1, r1 + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w - 1, c0 + 4)
                extras = [
                    (rr, cc)
                    for rr in range(rr0, rr1 + 1)
                    for cc in range(cc0, cc1 + 1)
                    if int(g[rr, cc]) != BG and (rr, cc) not in rect
                ]
                if extras:
                    continue
                candidates.append(
                    {
                        "bar_row": r,
                        "base_row": base_r,
                        "start_col": c0,
                        "colors": colors,
                        "cells": rect,
                    }
                )
    assert len(candidates) == 1, "lookup bar must be unique"
    return candidates[0]


def _extract_objects(grid: np.ndarray, lookup_cells: set[tuple[int, int]]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    temp = g.copy()
    for r, c in lookup_cells:
        temp[r, c] = BG

    out = []
    for color in sorted(set(map(int, temp.ravel())) - {BG, GREY, MAROON}):
        for coords in _component_coords(temp == color):
            directions = set()
            marker_contacts = 0
            for r, c in coords:
                rr = int(r)
                cc = int(c)
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == MAROON:
                        directions.add((-dr, -dc))
                        marker_contacts += 1
            assert directions, "each moving object must touch at least one maroon marker"
            assert len(directions) == 1, "maroon contact must define a unique push direction"
            out.append(
                {
                    "color": color,
                    "coords": coords.astype(int, copy=False),
                    "direction": directions.pop(),
                    "marker_contacts": marker_contacts,
                    "origin": (int(coords[:, 0].min()), int(coords[:, 1].min())),
                }
            )
    assert out, "expected moving objects"
    return out


def _solve(grid: np.ndarray, return_state: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    lookup = _find_lookup_bar(g)
    lookup_cells = set(lookup["cells"])
    moving = _extract_objects(g, lookup_cells)

    out = g.copy()
    for obj in moving:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    static_blockers = set(lookup_cells)
    settled = set(static_blockers)
    placed = []

    moving.sort(
        key=lambda obj: (
            _steps_to_boundary(g.shape, obj["coords"], obj["direction"]),
            obj["origin"][0],
            obj["origin"][1],
        )
    )

    for obj in moving:
        coords = obj["coords"].copy()
        dr, dc = obj["direction"]
        steps = 0
        while True:
            moved = _translate(coords, dr, dc, 1)
            moved_set = _coords_set(moved)
            if not np.all((0 <= moved[:, 0]) & (moved[:, 0] < g.shape[0]) & (0 <= moved[:, 1]) & (moved[:, 1] < g.shape[1])):
                break
            if moved_set & settled:
                break
            coords = moved
            steps += 1
        cells = _coords_set(coords)
        assert not (cells & settled)
        settled |= cells
        out[coords[:, 0], coords[:, 1]] = obj["color"]
        placed.append(
            {
                "color": obj["color"],
                "input_coords": obj["coords"],
                "final_coords": coords,
                "direction": obj["direction"],
                "steps": steps,
                "touched": _touching_sides(g.shape, coords),
            }
        )

    boundary_counts = Counter()
    for obj in placed:
        for side in obj["touched"]:
            boundary_counts[side] += 1

    lookup_map = {i + 1: lookup["colors"][i] for i in range(4)}
    for count in boundary_counts.values():
        assert 1 <= count <= 4, "lookup only covers arrival counts 1..4"

    for obj in placed:
        if not obj["touched"]:
            obj["output_color"] = obj["color"]
            continue
        mapped = {lookup_map[boundary_counts[side]] for side in obj["touched"]}
        assert len(mapped) == 1, "corner-touching recolor would be ambiguous"
        recolor = mapped.pop()
        obj["output_color"] = recolor
        coords = obj["final_coords"]
        out[coords[:, 0], coords[:, 1]] = recolor

    for r, c in lookup_cells:
        out[r, c] = BG

    if not return_state:
        return out
    return out, {"lookup": lookup, "objects": placed, "boundary_counts": dict(boundary_counts)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push sprite objects away from maroon launch markers, count how many reach each boundary, then recolor those boundary arrivals via a 4-cell lookup bar.

    Input structure:
    - The grid has black background, a horizontal 4-pixel color lookup bar in one corner, and a grey 4-pixel base row directly above or below it.
    - Elsewhere are fixed maroon marker pixels and several small monochrome 4-connected objects.
    - Each moving object touches maroon on exactly one cardinal side, which determines its push direction.

    Rule:
    - Read the lookup bar left-to-right as the colors for arrival counts 1, 2, 3, and 4.
    - Clear every moving object from its start position, then settle objects in increasing distance-to-boundary order; each translates rigidly away from its maroon contact until the next step would leave the grid or hit an already settled object.
    - Count, for each boundary, how many settled objects touch that boundary, recolor all objects on that boundary by the lookup color for that count, keep maroon markers fixed, and remove the lookup bar plus its grey base.

    Key insight:
    - The output color of a moved object is not tied to its original color or direction; movement first decides which boundary it belongs to, and that boundary's arrival count becomes the lookup index that drives recoloring.
    """
    return _solve(grid, return_state=False)


def _choose_lookup(grid: np.ndarray) -> dict:
    h, w = grid.shape
    corner = random.choice(["tl", "tr", "bl", "br"])
    colors = random.sample(PALETTE, 4)
    if corner == "tl":
        bar_row, base_row, start_col = 0, 1, 0
    elif corner == "tr":
        bar_row, base_row, start_col = 0, 1, w - 4
    elif corner == "bl":
        bar_row, base_row, start_col = h - 1, h - 2, 0
    else:
        bar_row, base_row, start_col = h - 1, h - 2, w - 4

    cells = set()
    for c in range(start_col, start_col + 4):
        grid[bar_row, c] = colors[c - start_col]
        grid[base_row, c] = GREY
        cells.add((bar_row, c))
        cells.add((base_row, c))
    return {"cells": cells, "colors": colors, "corner": corner}


def _counts_profile(total: int) -> dict[str, int]:
    for _ in range(200):
        active_count = random.randint(2, min(4, total))
        active = random.sample(list(SIDES), active_count)
        counts = [1] * active_count
        remaining = total - active_count
        while remaining > 0:
            idx = random.randrange(active_count)
            if counts[idx] < 4:
                counts[idx] += 1
                remaining -= 1
        if len(set(counts)) == 1:
            continue
        if not any(v == 4 for v in counts) and random.random() < 0.35:
            continue
        return dict(zip(active, counts))
    raise RuntimeError("failed to sample boundary-count profile")


def _possible_anchors(shape: tuple[tuple[int, int], ...], side: str, grid_shape: tuple[int, int]) -> list[int]:
    h, w = grid_shape
    arr = np.array(shape, dtype=int)
    max_r = int(arr[:, 0].max())
    max_c = int(arr[:, 1].max())
    if side in ("top", "bottom"):
        lo = 1
        hi = w - 2 - max_c
    else:
        lo = 1
        hi = h - 2 - max_r
    return list(range(lo, hi + 1))


def _place_leads(
    grid_shape: tuple[int, int], lookup_cells: set[tuple[int, int]], lead_counts: dict[str, int]
) -> list[dict] | None:
    occupied: set[tuple[int, int]] = set(lookup_cells)
    buffer = _expanded(lookup_cells, grid_shape)
    leads: list[dict] = []
    for side in random.sample(list(lead_counts.keys()), len(lead_counts)):
        for _ in range(lead_counts[side]):
            placed = None
            choices = SHAPES[:]
            random.shuffle(choices)
            for shape in choices:
                anchors = _possible_anchors(shape, side, grid_shape)
                random.shuffle(anchors)
                for anchor in anchors:
                    coords = _boundary_place(shape, side, anchor, grid_shape)
                    if coords is None:
                        continue
                    cells = _coords_set(coords)
                    if cells & occupied:
                        continue
                    if _expanded(cells, grid_shape) & buffer:
                        continue
                    placed = {"side": side, "shape": shape, "anchor": anchor, "final_coords": coords}
                    break
                if placed is not None:
                    break
            if placed is None:
                return None
            leads.append(placed)
            cells = _coords_set(placed["final_coords"])
            occupied |= cells
            buffer |= _expanded(cells, grid_shape)
    return leads


def _add_followers(leads: list[dict], grid_shape: tuple[int, int], occupied: set[tuple[int, int]], count: int) -> list[dict] | None:
    if count == 0:
        return []
    if count > len(leads):
        return None
    followers = []
    for lead in random.sample(leads, count):
        coords = _first_non_overlapping_stack(lead["shape"], lead["side"], lead["anchor"], grid_shape, occupied)
        if coords is None:
            return None
        follower = {
            "side": lead["side"],
            "shape": lead["shape"],
            "anchor": lead["anchor"],
            "final_coords": coords,
        }
        followers.append(follower)
        occupied |= _coords_set(coords)
    return followers


def _max_inward_shift(
    coords: np.ndarray, inward: tuple[int, int], grid_shape: tuple[int, int], forbidden: set[tuple[int, int]]
) -> int:
    steps = 0
    cur = coords.copy()
    while True:
        moved = _translate(cur, inward[0], inward[1], 1)
        moved_cells = _coords_set(moved)
        if not np.all((0 <= moved[:, 0]) & (moved[:, 0] < grid_shape[0]) & (0 <= moved[:, 1]) & (moved[:, 1] < grid_shape[1])):
            break
        if moved_cells & forbidden:
            break
        cur = moved
        steps += 1
    return steps


def _marker_candidates(
    coords: np.ndarray,
    direction: tuple[int, int],
    grid_shape: tuple[int, int],
    occupied_objects: set[tuple[int, int]],
    used_markers: set[tuple[int, int]],
    fixed_forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    contact = (-direction[0], -direction[1])
    own = _coords_set(coords)
    candidates = []
    for r, c in own:
        mr = r + contact[0]
        mc = c + contact[1]
        if not _in_bounds(grid_shape, mr, mc):
            continue
        marker = (mr, mc)
        if marker in own or marker in occupied_objects or marker in used_markers or marker in fixed_forbidden:
            continue
        adj = 0
        for nr, nc in _neighbors4(mr, mc):
            if (nr, nc) in occupied_objects or (nr, nc) in own:
                adj += 1
        if adj == 1:
            candidates.append(marker)
    return candidates


def _place_lane_inputs(
    lane_objects: list[dict],
    grid_shape: tuple[int, int],
    fixed_forbidden: set[tuple[int, int]],
    occupied_objects: set[tuple[int, int]],
    used_markers: set[tuple[int, int]],
) -> list[dict] | None:
    inward = tuple(-v for v in SIDE_TO_DIR[lane_objects[0]["side"]])
    max_shifts = [
        _max_inward_shift(obj["final_coords"], inward, grid_shape, fixed_forbidden) for obj in lane_objects
    ]
    if any(limit < 1 for limit in max_shifts):
        return None

    for _ in range(120):
        shifts = []
        minimum = 1
        for limit in max_shifts:
            if limit < minimum:
                break
            shift = random.randint(minimum, limit)
            shifts.append(shift)
            minimum = shift
        if len(shifts) != len(lane_objects):
            continue

        inputs = []
        trial_occupied = set(occupied_objects)
        collision = False
        for obj, shift in zip(lane_objects, shifts):
            coords = _translate(obj["final_coords"], inward[0], inward[1], shift)
            cells = _coords_set(coords)
            if cells & trial_occupied or cells & fixed_forbidden:
                collision = True
                break
            trial_occupied |= cells
            inputs.append(coords)
        if collision:
            continue

        lane_entries = []
        trial_markers = set(used_markers)
        all_input_cells = set(occupied_objects)
        for coords in inputs:
            all_input_cells |= _coords_set(coords)
        for obj, shift, coords in zip(lane_objects, shifts, inputs):
            candidates = _marker_candidates(
                coords,
                SIDE_TO_DIR[obj["side"]],
                grid_shape,
                all_input_cells - _coords_set(coords),
                trial_markers,
                fixed_forbidden,
            )
            if not candidates:
                lane_entries = []
                break
            marker = random.choice(candidates)
            trial_markers.add(marker)
            lane_entries.append(
                {
                    **obj,
                    "direction": SIDE_TO_DIR[obj["side"]],
                    "input_coords": coords,
                    "marker": marker,
                    "shift": shift,
                }
            )
        if lane_entries:
            return lane_entries
    return None


def _validate_generated(inp: np.ndarray, expect_blockers: bool) -> bool:
    try:
        out, state = _solve(inp, return_state=True)
    except AssertionError:
        return False

    objects = state["objects"]
    counts = state["boundary_counts"]
    if len(objects) not in (6, 7):
        return False
    if len([side for side, count in counts.items() if count > 0]) < 2:
        return False
    positives = [count for count in counts.values() if count > 0]
    if len(set(positives)) == 1:
        return False
    if any(count > 4 for count in positives):
        return False
    if any(len(obj["touched"]) > 1 for obj in objects):
        return False
    if not any(obj["steps"] > 0 for obj in objects):
        return False

    blocked = [obj for obj in objects if not obj["touched"]]
    if expect_blockers and not blocked:
        return False
    if not expect_blockers and blocked:
        return False

    lookup_cells = state["lookup"]["cells"]
    if any(int(out[r, c]) != BG for r, c in lookup_cells):
        return False

    lookup_map = {i + 1: state["lookup"]["colors"][i] for i in range(4)}
    for obj in objects:
        coords = obj["final_coords"]
        color = int(out[coords[0, 0], coords[0, 1]])
        if obj["touched"]:
            side = obj["touched"][0]
            if color != lookup_map[counts[side]]:
                return False
        else:
            if color != obj["color"]:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        grid = np.zeros((h, w), dtype=int)
        lookup = _choose_lookup(grid)

        total_objects = random.choice([6, 7])
        expect_blockers = random.random() < 0.65
        follower_count = 0 if not expect_blockers else random.choice([1, 2])
        if total_objects - follower_count < 4:
            follower_count = 1
        lead_count = total_objects - follower_count

        try:
            lead_counts = _counts_profile(lead_count)
        except RuntimeError:
            continue

        leads = _place_leads(grid.shape, lookup["cells"], lead_counts)
        if leads is None:
            continue

        occupied_final = set(lookup["cells"])
        for lead in leads:
            occupied_final |= _coords_set(lead["final_coords"])

        followers = _add_followers(leads, grid.shape, occupied_final, follower_count)
        if followers is None:
            continue

        all_final = leads + followers
        if len(all_final) != total_objects:
            continue

        colors = random.sample(PALETTE, total_objects)
        for obj, color in zip(all_final, colors):
            obj["color"] = color

        lanes: dict[tuple[str, int], list[dict]] = {}
        for obj in all_final:
            key = (obj["side"], obj["anchor"])
            lanes.setdefault(key, []).append(obj)

        for lane in lanes.values():
            direction = lane[0]["side"]
            inward = tuple(-v for v in SIDE_TO_DIR[direction])
            lane.sort(key=lambda obj: _steps_to_boundary(grid.shape, obj["final_coords"], inward))

        fixed_forbidden = set(lookup["cells"])
        placed_entries: list[dict] = []
        occupied_input_objects: set[tuple[int, int]] = set()
        used_markers: set[tuple[int, int]] = set()
        lane_items = list(lanes.values())
        random.shuffle(lane_items)
        failed = False
        for lane in lane_items:
            entries = _place_lane_inputs(
                lane,
                grid.shape,
                fixed_forbidden,
                occupied_input_objects,
                used_markers,
            )
            if entries is None:
                failed = True
                break
            for entry in entries:
                placed_entries.append(entry)
                occupied_input_objects |= _coords_set(entry["input_coords"])
                used_markers.add(entry["marker"])
        if failed:
            continue

        inp = grid.copy()
        for entry in placed_entries:
            coords = entry["input_coords"]
            inp[coords[:, 0], coords[:, 1]] = entry["color"]
            mr, mc = entry["marker"]
            inp[mr, mc] = MAROON

        if len(placed_entries) != total_objects:
            continue

        if _validate_generated(inp, expect_blockers):
            return inp
    raise RuntimeError("failed to generate valid puzzle-8 input")
