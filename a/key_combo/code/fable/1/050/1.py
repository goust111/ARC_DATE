import random
from collections import deque

import numpy as np


MEETING_COLOR = 2
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if int(connectivity) == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for dr, dc in neighbors:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _diamond_outline(shape: tuple[int, int], cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    if radius < 0:
        return set()
    h, w = map(int, shape)
    if radius == 0:
        return {(cr, cc)} if 0 <= cr < h and 0 <= cc < w else set()

    pts: set[tuple[int, int]] = set()
    r0 = max(0, cr - radius)
    r1 = min(h - 1, cr + radius)
    for r in range(r0, r1 + 1):
        rem = radius - abs(r - cr)
        if rem < 0:
            continue
        c1 = cc - rem
        c2 = cc + rem
        if 0 <= c1 < w:
            pts.add((r, c1))
        if 0 <= c2 < w:
            pts.add((r, c2))
    return pts


def _diamond_fits(coords: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    if not pts:
        return []

    h, w = map(int, shape)
    margin = h + w
    seed_r, seed_c = next(iter(pts))
    fits: list[tuple[int, int, int]] = []
    for cr in range(-margin, h + margin + 1):
        for cc in range(-margin, w + margin + 1):
            radius = abs(seed_r - cr) + abs(seed_c - cc)
            ok = True
            for r, c in pts:
                if abs(r - cr) + abs(c - cc) != radius:
                    ok = False
                    break
            if ok and _diamond_outline(shape, cr, cc, radius) == pts:
                fits.append((cr, cc, radius))
    return fits


def _minimal_diamond_fit(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int]:
    fits = _diamond_fits(coords, shape)
    if not fits:
        raise AssertionError("component is not a clipped Manhattan-diamond outline")
    min_radius = min(radius for _cr, _cc, radius in fits)
    minimal = [(cr, cc, radius) for cr, cc, radius in fits if radius == min_radius]
    if len(minimal) != 1:
        raise AssertionError("diamond fit must have a unique minimal radius")
    return minimal[0]


def _touch_or_overlap(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _extract_specs(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fg_colors = [int(v) for v in np.unique(g) if int(v) != bg]
    if len(fg_colors) != 1:
        raise AssertionError("input must contain exactly one foreground color")
    outline_color = fg_colors[0]

    specs = []
    for comp in _components_mask(g == outline_color, connectivity=8):
        cr, cc, radius = _minimal_diamond_fit(comp, g.shape)
        specs.append({"center": (cr, cc), "radius": radius})
    if len(specs) < 2:
        raise AssertionError("expected several diamond outlines")
    return bg, outline_color, specs


def _simulate_objects(
    shape: tuple[int, int], specs: list[dict]
) -> tuple[list[int], list[set[tuple[int, int]]], set[tuple[int, int]], list[dict]]:
    radii = [int(spec["radius"]) for spec in specs]
    centers = [tuple(map(int, spec["center"])) for spec in specs]
    outlines = [_diamond_outline(shape, cr, cc, radius) for (cr, cc), radius in zip(centers, radii)]
    active = [True] * len(specs)
    red_cells: set[tuple[int, int]] = set()
    events: list[dict] = []

    # Robustness for malformed inputs: objects already touching are treated as already stopped.
    initial_stop = set()
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if not _touch_or_overlap(outlines[i], outlines[j]):
                continue
            initial_stop.add(i)
            initial_stop.add(j)
            overlap = outlines[i] & outlines[j]
            if overlap:
                red_cells |= overlap
            events.append(
                {"step": 0, "pair": (i, j), "overlap": bool(overlap), "kind": "initial-contact"}
            )
    for idx in initial_stop:
        active[idx] = False

    step_limit = 8 * (shape[0] + shape[1])
    for step in range(1, step_limit + 1):
        if not any(active):
            break

        for i, is_active in enumerate(active):
            if not is_active:
                continue
            radii[i] += 1
            cr, cc = centers[i]
            outlines[i] = _diamond_outline(shape, cr, cc, radii[i])

        newly_stopped: set[int] = set()
        step_red: set[tuple[int, int]] = set()
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                if not _touch_or_overlap(outlines[i], outlines[j]):
                    continue
                if not active[i] and not active[j]:
                    continue
                overlap = outlines[i] & outlines[j]
                events.append(
                    {
                        "step": step,
                        "pair": (i, j),
                        "overlap": bool(overlap),
                        "kind": "active-active" if active[i] and active[j] else "active-stopped",
                    }
                )
                if active[i]:
                    newly_stopped.add(i)
                if active[j]:
                    newly_stopped.add(j)
                step_red |= overlap

        for idx in newly_stopped:
            active[idx] = False
        red_cells |= step_red

    if any(active):
        raise AssertionError("all outlines must stop within the simulation horizon")
    return radii, outlines, red_cells, events


def _render_grid(
    shape: tuple[int, int],
    bg: int,
    outline_color: int,
    outlines: list[set[tuple[int, int]]],
    red_cells: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for pts in outlines:
        for r, c in pts:
            out[r, c] = int(outline_color)
    if red_cells is not None:
        for r, c in red_cells:
            out[r, c] = MEETING_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow all same-colored diamond outlines outward in lockstep until each one hits another outline.

    Input structure:
    - A single background color contains several one-cell-thick Manhattan-diamond outlines, all in the same foreground color.
    - Diamond centers and starting radii differ, and some visible outlines may be clipped by the grid boundary.
    - No color code distinguishes which outline stops first; only geometry matters.

    Rule:
    - Recover each diamond's center and current radius from its clipped outline.
    - Simultaneously expand every not-yet-stopped diamond by one radius layer per step, keeping each center fixed.
    - After each growth step, any active diamond that now overlaps or 4-neighbors any other current/frozen outline stops forever; shared overlap cells from such meetings are colored red.

    Key insight:
    - Final radii are not set by a common number of steps: each outline freezes when it first meets another one, and later outlines can stop by colliding with already-frozen neighbors, so the result must be simulated causally.
    """
    g = np.asarray(grid, dtype=int)
    bg, outline_color, specs = _extract_specs(g)
    _radii, outlines, red_cells, _events = _simulate_objects(g.shape, specs)
    return _render_grid(g.shape, bg, outline_color, outlines, red_cells)


def _sort_specs(specs: list[dict]) -> list[tuple[int, int, int]]:
    return sorted((int(spec["center"][0]), int(spec["center"][1]), int(spec["radius"])) for spec in specs)


def _is_clipped(shape: tuple[int, int], center: tuple[int, int], radius: int) -> bool:
    full = 1 if radius == 0 else 4 * radius
    return len(_diamond_outline(shape, center[0], center[1], radius)) < full


def _latent_specs_to_input(shape: tuple[int, int], specs: list[dict]) -> list[set[tuple[int, int]]]:
    return [_diamond_outline(shape, spec["center"][0], spec["center"][1], spec["radius"]) for spec in specs]


def _build_chain_specs() -> tuple[tuple[int, int], int, int, list[dict], bool]:
    count = random.randint(3, 4)
    orientation = random.choice(["horizontal", "vertical"])
    clip_requested = random.random() < 0.65

    bg_choices = [c for c in range(10) if c != MEETING_COLOR]
    bg = random.choice(bg_choices)
    outline_choices = [c for c in range(10) if c not in (bg, MEETING_COLOR)]
    outline_color = random.choice(outline_choices)

    initial_radii = [random.randint(1, 2) for _ in range(count)]
    first_stop = random.randint(1, 2)
    stop_steps = [first_stop, first_stop]
    while len(stop_steps) < count:
        stop_steps.append(stop_steps[-1] + random.randint(1, 2))

    modes = [random.choice(["adjacent", "overlap"]) for _ in range(count - 1)]
    if count - 1 >= 2:
        forced = random.sample(range(count - 1), 2)
        modes[forced[0]] = "adjacent"
        modes[forced[1]] = "overlap"

    final_radii = [initial_radii[i] + stop_steps[i] for i in range(count)]
    gaps = []
    for i in range(1, count):
        gap = final_radii[i - 1] + final_radii[i]
        if modes[i - 1] == "adjacent":
            gap += 1
        gaps.append(gap)

    axis_positions = [0]
    for gap in gaps:
        axis_positions.append(axis_positions[-1] + gap)
    axis_span = axis_positions[-1] - axis_positions[0]

    max_final = max(final_radii)
    axis_len_min = axis_span + 2 * max_final + 1
    if axis_len_min > 30:
        raise AssertionError("chain too large for ARC canvas")
    axis_len = random.randint(axis_len_min, 30)
    axis_offset = random.randint(max_final, axis_len - 1 - max_final - axis_span)

    if orientation == "horizontal":
        width = axis_len
        min_height = max(10, 2 * max_final + 1 if not clip_requested else max_final + 3)
        height = random.randint(min_height, 30)
        if clip_requested:
            row = 0 if random.random() < 0.5 else height - 1
        else:
            row = random.randint(max_final, height - 1 - max_final)
        centers = [(row, axis_offset + pos) for pos in axis_positions]
    else:
        height = axis_len
        min_width = max(10, 2 * max_final + 1 if not clip_requested else max_final + 3)
        width = random.randint(min_width, 30)
        if clip_requested:
            col = 0 if random.random() < 0.5 else width - 1
        else:
            col = random.randint(max_final, width - 1 - max_final)
        centers = [(axis_offset + pos, col) for pos in axis_positions]

    if random.random() < 0.5:
        if orientation == "horizontal":
            centers = [(r, width - 1 - c) for r, c in centers]
        else:
            centers = [(height - 1 - r, c) for r, c in centers]

    specs = [{"center": tuple(map(int, ctr)), "radius": int(rad)} for ctr, rad in zip(centers, initial_radii)]
    return (height, width), bg, outline_color, specs, clip_requested


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            shape, bg, outline_color, specs, clip_requested = _build_chain_specs()
        except AssertionError:
            continue

        input_outlines = _latent_specs_to_input(shape, specs)
        if any(not pts for pts in input_outlines):
            continue

        input_grid = _render_grid(shape, bg, outline_color, input_outlines)
        try:
            parsed_bg, parsed_color, parsed_specs = _extract_specs(input_grid)
        except AssertionError:
            continue
        if parsed_bg != bg or parsed_color != outline_color:
            continue
        if _sort_specs(parsed_specs) != _sort_specs(specs):
            continue

        try:
            final_radii, final_outlines, red_cells, events = _simulate_objects(shape, specs)
        except AssertionError:
            continue

        if any(_touch_or_overlap(input_outlines[i], input_outlines[j]) for i in range(len(specs)) for j in range(i + 1, len(specs))):
            continue

        if not any(event["kind"] == "active-stopped" for event in events):
            continue
        if not any(event["overlap"] for event in events):
            continue
        if not any(not event["overlap"] for event in events if event["step"] > 0):
            continue

        clipped_count = sum(_is_clipped(shape, spec["center"], spec["radius"]) for spec in specs)
        if clip_requested and clipped_count == 0:
            continue

        output_grid = _render_grid(shape, bg, outline_color, final_outlines, red_cells)
        if not np.array_equal(main(input_grid), output_grid):
            continue

        meta = {
            "specs": specs,
            "final_radii": final_radii,
            "events": events,
            "clip_requested": clip_requested,
            "clipped_count": clipped_count,
        }
        return input_grid, output_grid, meta

    raise RuntimeError("failed to generate a valid synchronized-growth case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
