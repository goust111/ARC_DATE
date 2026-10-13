from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random
from typing import Iterable

import numpy as np


BG = 0
DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 240
SHAPES = ((18, 18), (18, 20), (20, 18), (20, 20), (18, 22), (22, 18))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _state_cells(state: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r, c, dr, dc = state
    return {(r - dr, c), (r, c - dc), (r - dr, c - dc)}


def _state_bbox(state: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r, c, dr, dc = state
    r0, r1 = sorted((r, r - dr))
    c0, c1 = sorted((c, c - dc))
    return r0, c0, r1, c1


def _valid_state(shape: tuple[int, int], state: tuple[int, int, int, int]) -> bool:
    h, w = shape
    r, c, dr, dc = state
    cells = _state_cells(state) | {(r, c)}
    return all(0 <= rr < h and 0 <= cc < w for rr, cc in cells)


def _advance_state(
    state: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[tuple[int, int, int, int], bool, bool]:
    h, w = shape
    r, c, dr, dc = state
    hit_row = not (0 <= r + dr < h)
    hit_col = not (0 <= c + dc < w)
    if hit_row:
        dr = -dr
    if hit_col:
        dc = -dc
    return (r + dr, c + dc, dr, dc), hit_row, hit_col


def _orbit_steps(shape: tuple[int, int], start: tuple[int, int, int, int]) -> list[dict]:
    steps: list[dict] = []
    state = start
    bounce_total = 0
    corner_seen = False
    while True:
        nxt, hit_row, hit_col = _advance_state(state, shape)
        if nxt == start:
            return steps
        bounce_total += int(hit_row) + int(hit_col)
        corner_seen = corner_seen or (hit_row and hit_col)
        steps.append(
            {
                "state": nxt,
                "coord": nxt[:2],
                "step": len(steps) + 1,
                "bounce_total": bounce_total,
                "corner_seen": corner_seen,
                "had_bounce": bool(hit_row or hit_col),
                "corner_bounce": bool(hit_row and hit_col),
            }
        )
        state = nxt


def _branch_candidates(steps: list[dict], branch: str) -> list[dict]:
    if branch == "direct":
        return [step for step in steps if step["bounce_total"] == 0 and step["step"] >= 2]
    if branch == "bounce":
        return [step for step in steps if step["bounce_total"] >= 1 and not step["corner_seen"]]
    if branch == "corner":
        return [step for step in steps if step["corner_seen"]]
    if branch == "delete":
        return steps if steps else []
    raise ValueError(f"unknown branch: {branch}")


def _parse_l_start(grid: np.ndarray, comp: dict, bg: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "L sprite must occupy a 2x2 box"
    sprite = grid[r0 : r1 + 1, c0 : c1 + 1]
    missing = np.argwhere(sprite == bg)
    assert missing.shape == (1, 2), "L sprite must have exactly one missing corner"
    mr, mc = map(int, missing[0])
    dr = -1 if mr == 0 else 1
    dc = -1 if mc == 0 else 1
    state = (r0 + mr, c0 + mc, dr, dc)
    assert _valid_state(grid.shape, state), "derived L state must be valid"
    return state


def _solve_and_trace(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[int, set[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _find_components(g, bg=bg)

    l_specs: list[dict] = []
    pixels_by_color: dict[int, set[tuple[int, int]]] = {}
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            pixels_by_color.setdefault(color, set()).add((r, c))
            continue
        if area == 3:
            state = _parse_l_start(g, comp, bg)
            l_specs.append({"color": color, "start": state, "bbox": comp["bbox"]})
            continue
        raise AssertionError("every foreground component must be an isolated pixel or a 2x2 L sprite")

    out = np.full_like(g, bg)
    for color, coords in pixels_by_color.items():
        for r, c in coords:
            out[r, c] = int(color)

    traces: list[dict] = []
    for spec in l_specs:
        color = int(spec["color"])
        start = spec["start"]
        state = start
        steps = 0
        bounce_total = 0
        corner_seen = False
        stop_state = None
        while True:
            state, hit_row, hit_col = _advance_state(state, g.shape)
            steps += 1
            bounce_total += int(hit_row) + int(hit_col)
            corner_seen = corner_seen or (hit_row and hit_col)
            coord = state[:2]
            if coord in pixels_by_color.get(color, set()):
                stop_state = state
                break
            if state == start:
                break

        if stop_state is not None:
            r, c, dr, dc = stop_state
            r0, r1 = sorted((r, r - dr))
            c0, c1 = sorted((c, c - dc))
            out[r0 : r1 + 1, c0 : c1 + 1] = color

        traces.append(
            {
                "color": color,
                "start": start,
                "stop_state": stop_state,
                "deleted": stop_state is None,
                "steps": steps,
                "bounce_total": bounce_total,
                "corner_seen": corner_seen,
            }
        )

    return out, traces, pixels_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each missing-corner L sprite diagonally until it docks onto a same-color pixel as a solid square, deleting sprites that loop forever.

    Input structure:
    - The background is black.
    - Foreground objects are monochrome 2x2 L sprites (three cells with one missing corner) and isolated single pixels.
    - Each L sprite's missing corner defines one diagonal travel direction, while isolated pixels stay fixed.

    Rule:
    - Move each L by tracking its missing corner one diagonal cell per step.
    - When that missing corner would leave the grid, reflect the offending velocity component and mirror the L across the touched border so the missing corner keeps facing the new direction.
    - Stop the first time the missing corner lands on an isolated pixel of the same color; that pixel plugs the gap and the result is a solid 2x2 square at that location.
    - Different-color pixels are ignored, and if an L returns to its initial missing-corner position with its initial direction before docking, delete it.
    - Keep every isolated pixel that was not used for docking in place.

    Key insight:
    - The moving reference is the empty corner itself: it simultaneously determines direction, boundary reflections, and the exact docking condition.
    """
    out, _traces, _pixels = _solve_and_trace(grid)
    return out


def _inflate_cells(cells: Iterable[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _all_states(shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = shape
    states = []
    for dr, dc in DIRS:
        for r in range(h):
            for c in range(w):
                state = (r, c, dr, dc)
                if _valid_state(shape, state):
                    states.append(state)
    return states


@lru_cache(maxsize=None)
def _shape_data(shape: tuple[int, int]) -> tuple[
    tuple[tuple[int, int, int, int], ...],
    dict[tuple[int, int, int, int], list[dict]],
    dict[str, list[tuple[int, int, int, int]]],
    tuple[tuple[int, int], ...],
]:
    states = tuple(_all_states(shape))
    orbit_cache = {state: _orbit_steps(shape, state) for state in states}
    branch_pool = {
        "direct": [state for state, steps in orbit_cache.items() if _branch_candidates(steps, "direct")],
        "bounce": [state for state, steps in orbit_cache.items() if _branch_candidates(steps, "bounce")],
        "corner": [state for state, steps in orbit_cache.items() if _branch_candidates(steps, "corner")],
        "delete": [state for state, steps in orbit_cache.items() if len(steps) >= 4],
    }
    cells = tuple((r, c) for r in range(shape[0]) for c in range(shape[1]))
    return states, orbit_cache, branch_pool, cells


def _choose_branches(available: dict[str, list[tuple[int, int, int, int]]], count: int) -> list[str]:
    complex_branches = [name for name in ("bounce", "delete") if available[name]]
    if available["corner"] and random.random() < 0.45:
        complex_branches.append("corner")
    stop_branches = [name for name in ("direct", "bounce", "corner") if available[name]]
    if not complex_branches or not stop_branches:
        raise ValueError("insufficient branch coverage for this grid size")

    branches = [random.choice(complex_branches), random.choice(stop_branches)]
    weighted = []
    for name, copies in (("direct", 4), ("bounce", 3), ("corner", 2), ("delete", 2)):
        if available[name]:
            weighted.extend([name] * copies)
    while len(branches) < count:
        branches.append(random.choice(weighted))
    random.shuffle(branches)
    if all(branch == "delete" for branch in branches):
        branches[0] = random.choice(stop_branches)
    return branches


def _available_orbit_cells(shape: tuple[int, int], orbit_coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in orbit_coords]
    random.shuffle(cells)
    return cells


def _choose_free_pixel(
    candidates: list[tuple[int, int]],
    shape: tuple[int, int],
    blocked_input: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
    local_input: set[tuple[int, int]],
    local_output: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for coord in candidates:
        inflated = _inflate_cells([coord], shape, pad=1)
        if inflated & blocked_input:
            continue
        if coord in blocked_output:
            continue
        if inflated & local_input:
            continue
        if coord in local_output:
            continue
        return coord
    return None


def _place_sprite_plan(
    shape: tuple[int, int],
    color: int,
    branch: str,
    distractor_colors: list[int],
    orbit_cache: dict[tuple[int, int, int, int], list[dict]],
    branch_pool: dict[str, list[tuple[int, int, int, int]]],
    blocked_input: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
) -> dict | None:
    starts = branch_pool[branch][:]
    random.shuffle(starts)
    for start in starts[:PLACEMENT_ATTEMPTS]:
        start_bbox = _state_bbox(start)
        start_block = _inflate_cells(_bbox_cells(start_bbox), shape, pad=1)
        if start_block & blocked_input:
            continue

        steps = orbit_cache[start]
        candidates = _branch_candidates(steps, branch)
        if branch != "delete":
            random.shuffle(candidates)
            for target in candidates:
                stop_state = target["state"]
                final_bbox = _state_bbox(stop_state)
                final_block = _bbox_cells(final_bbox)
                if final_block & blocked_output:
                    continue

                local_input = set(start_block)
                local_output = set(final_block)
                target_coord = target["coord"]
                if _choose_free_pixel([target_coord], shape, blocked_input, set(), local_input, set()) is None:
                    continue
                local_input |= _inflate_cells([target_coord], shape, pad=1)

                pixels = [(target_coord, color)]

                earlier = [step["coord"] for step in steps if step["step"] < target["step"]]
                later = [step["coord"] for step in steps if step["step"] > target["step"]]
                off_orbit = _available_orbit_cells(shape, {step["coord"] for step in steps} | {start[:2]})

                if earlier and random.random() < 0.72 and distractor_colors:
                    random.shuffle(earlier)
                    coord = _choose_free_pixel(earlier, shape, blocked_input, blocked_output, local_input, local_output)
                    if coord is not None:
                        pixels.append((coord, random.choice(distractor_colors)))
                        inflated = _inflate_cells([coord], shape, pad=1)
                        local_input |= inflated
                        local_output.add(coord)

                if later and random.random() < 0.45:
                    random.shuffle(later)
                    coord = _choose_free_pixel(later, shape, blocked_input, blocked_output, local_input, local_output)
                    if coord is not None:
                        pixels.append((coord, color))
                        inflated = _inflate_cells([coord], shape, pad=1)
                        local_input |= inflated
                        local_output.add(coord)

                if off_orbit and random.random() < 0.35:
                    coord = _choose_free_pixel(off_orbit, shape, blocked_input, blocked_output, local_input, local_output)
                    if coord is not None:
                        pixels.append((coord, color))
                        inflated = _inflate_cells([coord], shape, pad=1)
                        local_input |= inflated
                        local_output.add(coord)

                return {
                    "color": color,
                    "branch": branch,
                    "start": start,
                    "pixels": pixels,
                    "final_bbox": final_bbox,
                    "start_block": start_block,
                    "output_block": local_output,
                }
        else:
            orbit_coords = [step["coord"] for step in steps]
            if not orbit_coords:
                continue
            local_input = set(start_block)
            local_output: set[tuple[int, int]] = set()
            pixels: list[tuple[tuple[int, int], int]] = []

            if distractor_colors and random.random() < 0.8:
                path_coords = orbit_coords[:]
                random.shuffle(path_coords)
                coord = _choose_free_pixel(path_coords, shape, blocked_input, blocked_output, local_input, local_output)
                if coord is not None:
                    pixels.append((coord, random.choice(distractor_colors)))
                    inflated = _inflate_cells([coord], shape, pad=1)
                    local_input |= inflated
                    local_output.add(coord)

            off_orbit = _available_orbit_cells(shape, set(orbit_coords) | {start[:2]})
            coord = _choose_free_pixel(off_orbit, shape, blocked_input, blocked_output, local_input, local_output)
            if coord is None:
                continue
            pixels.append((coord, color))
            inflated = _inflate_cells([coord], shape, pad=1)
            local_input |= inflated
            local_output.add(coord)

            return {
                "color": color,
                "branch": branch,
                "start": start,
                "pixels": pixels,
                "final_bbox": None,
                "start_block": start_block,
                "output_block": local_output,
            }

    return None


def _build_grid(shape: tuple[int, int], plans: list[dict], extra_pixels: list[tuple[tuple[int, int], int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for plan in plans:
        for r, c in _state_cells(plan["start"]):
            grid[r, c] = int(plan["color"])
        for coord, color in plan["pixels"]:
            grid[coord] = int(color)
    for coord, color in extra_pixels:
        grid[coord] = int(color)
    return grid


def _semantic_checks(grid: np.ndarray, plans: list[dict], extra_pixels: list[tuple[tuple[int, int], int]]) -> bool:
    try:
        out, traces, pixels_by_color = _solve_and_trace(grid)
    except AssertionError:
        return False

    if len(plans) < 2:
        return False
    if len(traces) != len(plans):
        return False

    trace_by_color = {int(trace["color"]): trace for trace in traces}
    if len(trace_by_color) != len(traces):
        return False

    used_branches = Counter(plan["branch"] for plan in plans)
    if not any(branch in used_branches for branch in ("bounce", "corner", "delete")):
        return False
    if not any(plan["branch"] != "delete" for plan in plans):
        return False

    retained_pixels = {coord for coords in pixels_by_color.values() for coord in coords}
    final_components = _find_components(out, bg=BG)
    for comp in final_components:
        area = int(comp["area"])
        if area == 1:
            continue
        if area != 4:
            return False
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            return False

    for plan in plans:
        color = int(plan["color"])
        trace = trace_by_color.get(color)
        if trace is None:
            return False
        if plan["branch"] == "delete":
            if not trace["deleted"]:
                return False
        else:
            if trace["deleted"]:
                return False
            if plan["final_bbox"] != _state_bbox(trace["stop_state"]):
                return False

        same_color_pixels = [coord for coord, pixel_color in plan["pixels"] if int(pixel_color) == color]
        if plan["branch"] == "delete":
            orbit_coords = {step["coord"] for step in _orbit_steps(grid.shape, plan["start"])}
            if not same_color_pixels:
                return False
            if any(coord in orbit_coords for coord in same_color_pixels):
                return False
        else:
            stop_coord = trace["stop_state"][:2]
            if stop_coord not in same_color_pixels:
                return False
            orbit = _orbit_steps(grid.shape, plan["start"])
            path_index = {step["coord"]: idx for idx, step in enumerate(orbit)}
            stop_idx = path_index.get(stop_coord)
            if stop_idx is None:
                return False
            for coord in same_color_pixels:
                if coord == stop_coord:
                    continue
                idx = path_index.get(coord)
                if idx is not None and idx < stop_idx:
                    return False

    if np.array_equal(grid, out):
        return False
    if int(np.count_nonzero(out)) <= max(4, len(extra_pixels)):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    shape = random.choice(SHAPES)
    _states, orbit_cache, branch_pool, cells = _shape_data(shape)
    if not branch_pool["direct"] or not branch_pool["bounce"] or not branch_pool["delete"]:
        return None

    n_sprites = 2
    all_colors = list(range(1, 10))
    sprite_colors = random.sample(all_colors, n_sprites)
    distractor_colors = [color for color in all_colors if color not in sprite_colors]
    if not distractor_colors:
        return None

    try:
        branches = _choose_branches(branch_pool, n_sprites)
    except ValueError:
        return None

    blocked_input: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()
    plans: list[dict] = []

    placement_order = list(zip(sprite_colors, branches))
    difficulty = {"corner": 0, "delete": 1, "bounce": 2, "direct": 3}
    placement_order.sort(key=lambda item: difficulty[item[1]])

    for color, branch in placement_order:
        plan = _place_sprite_plan(
            shape,
            color,
            branch,
            distractor_colors,
            orbit_cache,
            branch_pool,
            blocked_input,
            blocked_output,
        )
        if plan is None:
            return None
        plans.append(plan)
        blocked_input |= plan["start_block"]
        for coord, _pixel_color in plan["pixels"]:
            blocked_input |= _inflate_cells([coord], shape, pad=1)
        blocked_output |= plan["output_block"]

    extra_pixels: list[tuple[tuple[int, int], int]] = []
    extra_count = random.randint(0, 1)
    shuffled_cells = list(cells)
    random.shuffle(shuffled_cells)
    for coord in shuffled_cells:
        if len(extra_pixels) >= extra_count:
            break
        inflated = _inflate_cells([coord], shape, pad=1)
        if inflated & blocked_input:
            continue
        if coord in blocked_output:
            continue
        extra_pixels.append((coord, random.choice(distractor_colors)))
        blocked_input |= inflated
        blocked_output.add(coord)

    grid = _build_grid(shape, plans, extra_pixels)
    if not _semantic_checks(grid, plans, extra_pixels):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner billiards instance")
