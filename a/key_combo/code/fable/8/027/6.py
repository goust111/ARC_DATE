import random
from collections import Counter

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(10))
NON_BLACK = tuple(range(1, 10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if int(connectivity) == 8:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _rotate90_local(pos: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = pos
    return c, n - 1 - r


def _orbit_local(pos: tuple[int, int], n: int) -> list[tuple[int, int]]:
    cells = []
    cur = pos
    for _ in range(4):
        if cur not in cells:
            cells.append(cur)
        cur = _rotate90_local(cur, n)
    return cells


def _all_orbits(n: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[list[tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            orbit = tuple(sorted(_orbit_local((r, c), n)))
            if orbit in seen:
                continue
            seen.add(orbit)
            out.append(list(orbit))
    out.sort(key=lambda orb: (len(orb), orb))
    return out


def _apply_rows_to_color(color: int, rows: list[tuple[int, int]], stop_before: int | None = None) -> int:
    cur = int(color)
    upto = len(rows) if stop_before is None else int(stop_before)
    for src, dst in rows[:upto]:
        if cur == src:
            cur = dst
    return cur


def _final_preimages(rows: list[tuple[int, int]], colors: list[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for color in colors:
        final = _apply_rows_to_color(color, rows)
        out.setdefault(final, []).append(int(color))
    return out


def _find_structures(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask((g != bg) & (g != BLACK), connectivity=4)
    if len(comps) < 2:
        raise ValueError("expected object and legend components")

    legend_candidates: list[tuple[int, int, int, int]] = []
    for coords in comps:
        b = bbox_coords(coords)
        r0, c0, r1, c1 = b
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        patch = crop_bbox(g, b)
        if w == 2 and h >= 2 and coords.shape[0] == h * w and np.all(patch != bg) and np.all(patch != BLACK):
            legend_candidates.append(b)
    if len(legend_candidates) != 1:
        raise ValueError("legend must be the unique solid two-column component")
    legend_bbox = legend_candidates[0]

    object_bbox = None
    best_area = -1
    for coords in comps:
        b = bbox_coords(coords)
        if b == legend_bbox:
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            object_bbox = b
    if object_bbox is None:
        raise ValueError("missing foreground object")
    return legend_bbox, object_bbox


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor the foreground by a row-ordered two-column legend, then repair black holes by 90-degree rotational completion.

    Input structure:
    - One dominant background surrounds exactly two separate foreground structures: a larger multicolor object and a smaller vertical 2-column legend block.
    - Each legend row encodes one visible replacement rule "left color -> right color"; black cells belong only to missing holes in the object, not to the legend.
    - The completed object becomes 90-degree rotationally color-symmetric only after those replacements are applied in top-to-bottom order.

    Rule:
    - Keep the background and legend unchanged.
    - For every non-background, non-black cell outside the legend, apply the legend rows sequentially from top to bottom, so later rows may rewrite colors produced by earlier rows.
    - On the recolored object, use its square bbox center as the 90-degree rotation center; each orbit must share one color after recoloring.
    - Fill only black holes with that orbit color and leave all already-colored cells unchanged.

    Key insight:
    - Directly comparing pre-recolor rotational counterparts is insufficient because their visible colors can differ; the legend's ordered chain replacement must happen first, and only then do the orbit colors collapse to a unique rotational completion.
    """
    g = np.asarray(input_grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = get_bg(g)
    legend_bbox, object_bbox = _find_structures(g)
    lr0, lc0, lr1, _lc1 = legend_bbox
    or0, oc0, or1, oc1 = object_bbox
    n = or1 - or0 + 1
    if n != oc1 - oc0 + 1 or n % 2 == 0:
        raise ValueError("object bbox must be an odd square")

    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[lr0 : lr1 + 1, lc0 : lc0 + 2] = True

    rows = [(int(g[r, lc0]), int(g[r, lc0 + 1])) for r in range(lr0, lr1 + 1)]
    out = g.copy()
    active = (~legend_mask) & (out != bg) & (out != BLACK)
    for src, dst in rows:
        hit = active & (out == src)
        out[hit] = dst

    patch = out[or0 : or1 + 1, oc0 : oc1 + 1]
    visited: set[tuple[tuple[int, int], ...]] = set()
    for r in range(n):
        for c in range(n):
            orbit = tuple(sorted(_orbit_local((r, c), n)))
            if orbit in visited:
                continue
            visited.add(orbit)
            colors = {int(patch[rr, cc]) for rr, cc in orbit if int(patch[rr, cc]) not in (bg, BLACK)}
            if len(colors) != 1:
                continue
            fill_color = next(iter(colors))
            for rr, cc in orbit:
                if int(patch[rr, cc]) == BLACK:
                    patch[rr, cc] = fill_color
    out[or0 : or1 + 1, oc0 : oc1 + 1] = patch
    return out


def _build_rows(rng: random.Random, bg: int) -> list[tuple[int, int]]:
    colors = [c for c in NON_BLACK if c != bg]
    rng.shuffle(colors)
    chain_a_len = rng.choice([2, 3])
    chain_b_len = rng.choice([1, 2])
    need = chain_a_len + 1 + chain_b_len + 1
    picked = colors[:need]

    rows: list[tuple[int, int]] = []
    idx = 0
    chain_a = picked[idx : idx + chain_a_len + 1]
    idx += chain_a_len + 1
    chain_b = picked[idx : idx + chain_b_len + 1]
    rows.extend((chain_a[i], chain_a[i + 1]) for i in range(chain_a_len))
    rows.extend((chain_b[i], chain_b[i + 1]) for i in range(chain_b_len))
    return rows


def _choose_final_palette(rows: list[tuple[int, int]], bg: int) -> tuple[list[int], dict[int, list[int]]]:
    colors = [c for c in NON_BLACK if c != bg]
    preimages = _final_preimages(rows, colors)
    usable = sorted(final for final, pres in preimages.items() if len(pres) >= 2)
    if len(usable) < 2:
        raise ValueError("need at least two multi-preimage final colors")
    return usable, preimages


def _assign_final_patch(
    rng: random.Random,
    n: int,
    rows: list[tuple[int, int]],
    bg: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict[str, object]]:
    usable_finals, preimages = _choose_final_palette(rows, bg)
    static_choices = [c for c in NON_BLACK if c != bg and _apply_rows_to_color(c, rows) == c]
    all_orbits = _all_orbits(n)
    non_center = [orb for orb in all_orbits if len(orb) == 4]
    boundary_orbits = [orb for orb in non_center if any(r in (0, n - 1) or c in (0, n - 1) for r, c in orb)]
    inner_orbits = [orb for orb in non_center if orb not in boundary_orbits]

    hole_orbits: list[list[tuple[int, int]]] = []
    target_hole_count = min(len(non_center), rng.randint(1, min(4, len(non_center))))
    if boundary_orbits and target_hole_count >= 1:
        hole_orbits.append(rng.choice(boundary_orbits))
    if inner_orbits and target_hole_count >= 2:
        inner_pick = rng.choice(inner_orbits)
        if inner_pick not in hole_orbits:
            hole_orbits.append(inner_pick)
    remaining = [orb for orb in non_center if orb not in hole_orbits]
    rng.shuffle(remaining)
    for orb in remaining:
        if len(hole_orbits) >= target_hole_count:
            break
        hole_orbits.append(orb)
    hole_keys = {tuple(sorted(orb)) for orb in hole_orbits}
    primary_hole_key = tuple(sorted(hole_orbits[0]))

    final_patch = np.zeros((n, n), dtype=int)
    input_patch = np.zeros((n, n), dtype=int)
    chain_sources = [src for src, _dst in rows]
    witness_needed = list(dict.fromkeys(chain_sources))
    forced_orbits = non_center[:]
    rng.shuffle(forced_orbits)
    forced_finals = {
        tuple(sorted(orb)): usable_finals[i % len(usable_finals)]
        for i, orb in enumerate(forced_orbits[: len(usable_finals)])
    }

    for orb in all_orbits:
        key = tuple(sorted(orb))
        if len(orb) == 1:
            center_color = rng.choice(static_choices or usable_finals)
            final_patch[orb[0]] = center_color
            input_patch[orb[0]] = rng.choice(preimages.get(center_color, [center_color]))
            continue

        if key in forced_finals:
            final_color = forced_finals[key]
        else:
            final_color = rng.choice(
                usable_finals if key in hole_keys or rng.random() < 0.75 else usable_finals + (static_choices[:1] if static_choices else [])
            )
        final_patch[tuple(np.array(orb).T)] = final_color
        choices = preimages[final_color][:]
        rng.shuffle(choices)
        originals = []
        required_distinct = 2 if key in hole_keys else (2 if rng.random() < 0.6 and len(choices) >= 2 else 1)
        if key == primary_hole_key and len(choices) >= 2:
            originals = [choices[0], choices[1], choices[0], choices[1]]
        else:
            while len(originals) < len(orb):
                pick = rng.choice(choices)
                originals.append(pick)
        if required_distinct >= 2 and len(set(originals)) == 1 and len(choices) >= 2:
            originals[0] = choices[0]
            originals[1] = choices[1]
        for pos, color in zip(orb, originals):
            input_patch[pos] = color

    witness_assignments: dict[tuple[int, int], int] = {}
    protected: set[tuple[int, int]] = set()
    for src in witness_needed:
        candidates = []
        for orb in non_center:
            final_color = int(final_patch[orb[0]])
            if src not in preimages[final_color]:
                continue
            for pos in orb:
                if pos not in protected:
                    candidates.append(pos)
        if not candidates:
            raise ValueError(f"no witness candidate for source color {src}")
        pos = rng.choice(candidates)
        input_patch[pos] = src
        witness_assignments[pos] = src
        protected.add(pos)

    chosen_holes: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for orb in hole_orbits:
        key = tuple(sorted(orb))
        hole_count = 1 if key == primary_hole_key else rng.randint(1, 3)
        candidates = [pos for pos in orb if pos not in protected]
        if len(candidates) < hole_count:
            raise ValueError("holes would erase required witnesses")
        holes = rng.sample(candidates, hole_count)
        chosen_holes[key] = holes
        for pos in holes:
            input_patch[pos] = BLACK

    meta = {
        "rows": rows,
        "preimages": preimages,
        "hole_keys": hole_keys,
        "chosen_holes": chosen_holes,
        "witness_assignments": witness_assignments,
    }
    return input_patch, final_patch, usable_finals, meta


def _simulate_stepwise(patch: np.ndarray, rows: list[tuple[int, int]]) -> list[np.ndarray]:
    states = [np.asarray(patch, dtype=int).copy()]
    cur = states[0].copy()
    for src, dst in rows:
        hit = cur == src
        cur = cur.copy()
        cur[hit] = dst
        states.append(cur)
    return states


def _validate_case(inp: np.ndarray, out: np.ndarray, meta: dict[str, object], obj_bbox: tuple[int, int, int, int], bg: int) -> None:
    rows = meta["rows"]
    or0, oc0, or1, oc1 = obj_bbox
    input_patch = inp[or0 : or1 + 1, oc0 : oc1 + 1]
    output_patch = out[or0 : or1 + 1, oc0 : oc1 + 1]
    states = _simulate_stepwise(input_patch, rows)

    for i, (src, _dst) in enumerate(rows):
        before = states[i]
        if not np.any(before == src):
            raise AssertionError(f"legend row {i} is inactive")

    multi_change_found = False
    for color in np.unique(input_patch[input_patch != BLACK]):
        step_color = int(color)
        changes = 0
        for src, dst in rows:
            if step_color == src:
                step_color = dst
                changes += 1
        if changes >= 2:
            multi_change_found = True
            break
    if not multi_change_found:
        raise AssertionError("need at least one cell affected by chained replacement")

    recolor_needed = False
    for key, holes in meta["chosen_holes"].items():
        orbit = list(key)
        visible_before = {int(input_patch[r, c]) for r, c in orbit if int(input_patch[r, c]) != BLACK}
        visible_after = {int(output_patch[r, c]) for r, c in orbit if int(output_patch[r, c]) not in (BLACK, bg)}
        if len(visible_before) >= 2 and len(visible_after) == 1 and holes:
            recolor_needed = True
        if len(visible_after) != 1:
            raise AssertionError("hole orbit did not collapse to one color")
    if not recolor_needed:
        raise AssertionError("need at least one hole orbit where recolor is necessary")

    if not np.any(inp == BLACK):
        raise AssertionError("expected black holes")
    if np.any(output_patch == BLACK):
        raise AssertionError("all object holes must be filled")


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        bg = rng.choice(NON_BLACK)
        rows = _build_rows(rng, bg)
        legend_h = len(rows)
        n = rng.choice([5, 7, 9, 11])
        if n + 6 > 30:
            continue

        try:
            input_patch, output_patch, _usable_finals, meta = _assign_final_patch(rng, n, rows, bg)
        except ValueError:
            continue

        min_h = max(n, legend_h) + 4
        min_w = n + 6
        h = rng.randint(min_h, min(30, min_h + 8))
        w = rng.randint(min_w, min(30, min_w + 10))
        if h * w - (n * n + 2 * legend_h) <= n * n:
            continue

        grid = np.full((h, w), bg, dtype=int)

        placed = False
        for _ in range(200):
            or0 = rng.randint(1, h - n - 1)
            oc0 = rng.randint(1, w - n - 1)
            lr0 = rng.randint(1, h - legend_h - 1)
            lc0 = rng.randint(1, w - 3)
            obj_bbox = (or0, oc0, or0 + n - 1, oc0 + n - 1)
            leg_bbox = (lr0, lc0, lr0 + legend_h - 1, lc0 + 1)
            separate = (
                obj_bbox[2] + 2 < leg_bbox[0]
                or leg_bbox[2] + 2 < obj_bbox[0]
                or obj_bbox[3] + 2 < leg_bbox[1]
                or leg_bbox[3] + 2 < obj_bbox[1]
            )
            if separate:
                placed = True
                break
        if not placed:
            continue

        grid[or0 : or0 + n, oc0 : oc0 + n] = input_patch
        for i, (src, dst) in enumerate(rows):
            grid[lr0 + i, lc0] = src
            grid[lr0 + i, lc0 + 1] = dst

        expected = grid.copy()
        expected[or0 : or0 + n, oc0 : oc0 + n] = output_patch

        if get_bg(grid) != bg:
            continue
        try:
            _validate_case(grid, expected, meta, obj_bbox, bg)
            solved = main(grid)
        except (AssertionError, ValueError):
            continue
        if not np.array_equal(solved, expected):
            continue
        return grid, expected
    raise RuntimeError("failed to generate valid case")


def generate_input() -> np.ndarray:
    rng = random.Random()
    return _build_case(rng)[0]
