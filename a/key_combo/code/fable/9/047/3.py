from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BACKGROUND = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
SHAPE_CANVAS = 7
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(cells: set[Coord]) -> tuple[Coord, ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _translate(cells: set[Coord] | tuple[Coord, ...], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in cells}


def _shape_mask(cells: set[Coord]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _components_of_color(grid: np.ndarray, color: int) -> list[set[Coord]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            comp: set[Coord] = set()
            q = deque([(r, c)])
            visited[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_templates(grid: np.ndarray, bg: int) -> list[dict]:
    templates: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, GREY}):
        comps = _components_of_color(grid, color)
        assert len(comps) == 1, "each template color must form exactly one object"
        shape_norm = set(_normalize_shape(comps[0]))
        templates.append({"color": color, "cells": comps[0], "shape": shape_norm})
    assert templates, "expected at least one colored template"
    shapes = {tuple(sorted(t["shape"])) for t in templates}
    assert len(shapes) == len(templates), "template shapes must be distinct"
    return templates


def _subset_offsets(ruin_cells: set[Coord], template_shape: set[Coord]) -> list[Coord]:
    offsets: set[Coord] = set()
    ruin_list = list(ruin_cells)
    templ_list = list(template_shape)
    for rr, rc in ruin_list:
        for tr, tc in templ_list:
            off = (rr - tr, rc - tc)
            if off in offsets:
                continue
            if all((r - off[0], c - off[1]) in template_shape for r, c in ruin_cells):
                offsets.add(off)
    return sorted(offsets)


def _candidate_full_cells(template_shape: set[Coord], offset: Coord) -> set[Coord]:
    return _translate(template_shape, offset[0], offset[1])


def _analyze_ruins(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    templates = _extract_templates(g, bg)
    ruins = _components_of_color(g, GREY)
    analyses: list[dict] = []
    h, w = g.shape
    for ruin_cells in ruins:
        subset_candidates: list[dict] = []
        feasible_candidates: list[dict] = []
        for template in templates:
            for offset in _subset_offsets(ruin_cells, template["shape"]):
                full_cells = _candidate_full_cells(template["shape"], offset)
                missing = full_cells - ruin_cells
                subset_candidates.append(
                    {
                        "template_color": template["color"],
                        "offset": offset,
                        "full_cells": full_cells,
                        "missing": missing,
                    }
                )
                fits = all(0 <= r < h and 0 <= c < w for r, c in missing)
                if fits and all(int(g[r, c]) == bg for r, c in missing):
                    feasible_candidates.append(
                        {
                            "template_color": template["color"],
                            "offset": offset,
                            "full_cells": full_cells,
                            "missing": missing,
                        }
                    )
        analyses.append(
            {
                "ruin_cells": ruin_cells,
                "subset_candidates": subset_candidates,
                "feasible_candidates": feasible_candidates,
            }
        )
    return bg, templates, analyses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each grey ruin by completing it with the unique spatially feasible template, and erase true scraps.

    Input structure:
    - The grid contains several intact monochrome template objects on a shared background.
    - It also contains several grey 4-connected ruin objects left from partially erased template copies.
    - Some ruins are shape-subsets of more than one template under translation, while some grey objects are scraps that match no template at all.

    Rule:
    - For each grey ruin, enumerate every (template, translation) pair whose full template mask contains the ruin mask as a translated subset.
    - Keep only the candidate whose missing pixels would all land on background cells; that unique feasible candidate gives the ruin's owner and the pixels to restore.
    - Fill those missing pixels, recolor the entire repaired object to the chosen template color, and leave every intact template unchanged.
    - If a grey component is not a translated subset of any template under any offset, erase it completely to the background.

    Key insight:
    - Shape containment alone is intentionally ambiguous for some ruins; only the candidate whose completion lands entirely in free space survives, while unmatched grey scraps disappear instead of being forced into a bad template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, _templates, analyses = _analyze_ruins(g)
    out = g.copy()
    claimed_cells: set[Coord] = set()

    for info in analyses:
        ruin_cells = info["ruin_cells"]
        subset_candidates = info["subset_candidates"]
        feasible_candidates = info["feasible_candidates"]
        if not subset_candidates:
            for r, c in ruin_cells:
                out[r, c] = bg
            continue
        assert len(feasible_candidates) == 1, "expected a unique spatially feasible owner for each matched ruin"
        chosen = feasible_candidates[0]
        for cell in chosen["full_cells"]:
            assert cell not in claimed_cells, "repaired ruins must remain disjoint"
            claimed_cells.add(cell)
        for r, c in chosen["full_cells"]:
            out[r, c] = chosen["template_color"]
    return out


def _frontier(shape: set[Coord], canvas_size: int) -> list[Coord]:
    frontier: set[Coord] = set()
    for r, c in shape:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < canvas_size and 0 <= nc < canvas_size and (nr, nc) not in shape:
                frontier.add((nr, nc))
    return list(frontier)


def _random_connected_shape(rng: random.Random, area: int, canvas_size: int = SHAPE_CANVAS) -> set[Coord] | None:
    start = (rng.randrange(1, canvas_size - 1), rng.randrange(1, canvas_size - 1))
    shape = {start}
    while len(shape) < area:
        choices = _frontier(shape, canvas_size)
        if not choices:
            return None
        shape.add(rng.choice(choices))
    r0, c0, r1, c1 = _bbox(shape)
    if r1 - r0 + 1 > 5 or c1 - c0 + 1 > 5:
        return None
    return shape


def _grow_with_distinct_addons(
    base: set[Coord], rng: random.Random, extra_count: int, forbidden: set[Coord]
) -> set[Coord] | None:
    shape = set(base)
    added: set[Coord] = set()
    while len(added) < extra_count:
        choices = [cell for cell in _frontier(shape, SHAPE_CANVAS) if cell not in forbidden and cell not in added]
        if not choices:
            return None
        cell = rng.choice(choices)
        shape.add(cell)
        added.add(cell)
    return added


def _shape_contains(template_cells: set[Coord], sub_cells: set[Coord]) -> bool:
    return bool(_subset_offsets(set(_normalize_shape(sub_cells)), set(_normalize_shape(template_cells))))


def _choose_shape_family(rng: random.Random) -> tuple[set[Coord], set[Coord], set[Coord], set[Coord], set[Coord]] | None:
    base_area = rng.randint(4, 5)
    base = _random_connected_shape(rng, base_area)
    if base is None:
        return None
    addon1 = _grow_with_distinct_addons(base, rng, rng.randint(1, 2), set())
    if addon1 is None:
        return None
    addon2 = _grow_with_distinct_addons(base, rng, rng.randint(1, 2), addon1)
    if addon2 is None:
        return None
    t1 = set(base) | addon1
    t2 = set(base) | addon2
    if tuple(sorted(_normalize_shape(t1))) == tuple(sorted(_normalize_shape(t2))):
        return None

    for _ in range(80):
        t3 = _random_connected_shape(rng, rng.randint(5, 7))
        if t3 is None:
            continue
        if _shape_contains(t3, base):
            continue
        norms = {
            tuple(sorted(_normalize_shape(base))),
            tuple(sorted(_normalize_shape(t1))),
            tuple(sorted(_normalize_shape(t2))),
        }
        if tuple(sorted(_normalize_shape(t3))) in norms:
            continue
        break
    else:
        return None

    t3_residue_choices: list[set[Coord]] = []
    t3_cells = list(t3)
    for remove_count in (1, 2):
        for removed in combinations(t3_cells, remove_count):
            remain = set(t3) - set(removed)
            if len(remain) < 3 or not _is_connected(remain):
                continue
            if _shape_contains(t1, remain) or _shape_contains(t2, remain):
                continue
            t3_residue_choices.append(remain)
    if not t3_residue_choices:
        return None
    t3_residue = rng.choice(t3_residue_choices)
    return base, t1, t2, t3, t3_residue


def _cells_fit(cells: set[Coord], height: int, width: int) -> bool:
    return all(0 <= r < height and 0 <= c < width for r, c in cells)


def _touches_occupied(cells: set[Coord], occupied: set[Coord], padding: int) -> bool:
    if padding <= 0:
        return bool(cells & occupied)
    for r, c in cells:
        for rr in range(r - padding, r + padding + 1):
            for cc in range(c - padding, c + padding + 1):
                if (rr, cc) in occupied:
                    return True
    return False


def _valid_shifts(shape: set[Coord], height: int, width: int) -> list[Coord]:
    r0, c0, r1, c1 = _bbox(shape)
    return [
        (dr, dc)
        for dr in range(-r0, height - r1)
        for dc in range(-c0, width - c1)
    ]


def _place_with_padding(
    shape: set[Coord],
    height: int,
    width: int,
    occupied: set[Coord],
    rng: random.Random,
    padding: int,
) -> set[Coord] | None:
    shifts = _valid_shifts(shape, height, width)
    rng.shuffle(shifts)
    for dr, dc in shifts:
        cells = _translate(shape, dr, dc)
        if not _touches_occupied(cells, occupied, padding):
            return cells
    return None


def _anchor_on_target(
    shape: set[Coord],
    target_cell: Coord,
    height: int,
    width: int,
    occupied: set[Coord],
    rng: random.Random,
) -> set[Coord] | None:
    anchors = list(shape)
    rng.shuffle(anchors)
    for ar, ac in anchors:
        dr = target_cell[0] - ar
        dc = target_cell[1] - ac
        cells = _translate(shape, dr, dc)
        if _cells_fit(cells, height, width) and not _touches_occupied(cells, occupied, 0):
            return cells
    return None


def _paint(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _scene_metadata(grid: np.ndarray) -> dict:
    bg, templates, analyses = _analyze_ruins(grid)
    matched = [a for a in analyses if a["subset_candidates"]]
    scraps = [a for a in analyses if not a["subset_candidates"]]
    return {
        "bg": bg,
        "templates": templates,
        "analyses": analyses,
        "matched": matched,
        "scraps": scraps,
    }


def _make_scrap(rng: random.Random, templates: list[set[Coord]]) -> set[Coord] | None:
    for _ in range(80):
        shape = _random_connected_shape(rng, rng.randint(3, 5))
        if shape is None:
            continue
        if any(_shape_contains(template, shape) for template in templates):
            continue
        return shape
    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    family = _choose_shape_family(rng)
    if family is None:
        return None
    base, t1, t2, t3, t3_residue = family
    template_shapes = [t1, t2, t3]
    colors = rng.sample(COLOR_POOL, 3)
    color_t1, color_t2, color_t3 = colors

    height = rng.randint(18, 24)
    width = rng.randint(18, 24)
    occupied: set[Coord] = set()

    union12 = set(t1) | set(t2)
    ruin1_full = _place_with_padding(union12, height, width, occupied, rng, padding=1)
    if ruin1_full is None:
        return None
    # Use the same shift for both true and decoy templates by aligning the raw union.
    shift1 = next(iter(ruin1_full))[0] - next(iter(union12))[0], next(iter(ruin1_full))[1] - next(iter(union12))[1]
    union12_shifted = _translate(union12, shift1[0], shift1[1])
    if union12_shifted != ruin1_full:
        # Reconstruct the shift explicitly from bbox anchors for stability.
        u0 = min(union12)
        v0 = min(ruin1_full)
        shift1 = (v0[0] - u0[0], v0[1] - u0[1])
    ruin1_true = _translate(t1, shift1[0], shift1[1])
    ruin1_base = _translate(base, shift1[0], shift1[1])
    ruin1_decoy = _translate(t2, shift1[0], shift1[1])
    if not (_cells_fit(ruin1_true, height, width) and _cells_fit(ruin1_decoy, height, width)):
        return None
    occupied |= ruin1_true

    decoy_blockers = list((set(t2) - set(t1)) or (set(t2) - set(base)))
    rng.shuffle(decoy_blockers)
    blocker_template = None
    for blocker_local in decoy_blockers:
        blocker_target = (blocker_local[0] + shift1[0], blocker_local[1] + shift1[1])
        blocker_cells = _anchor_on_target(t3, blocker_target, height, width, occupied, rng)
        if blocker_cells is None:
            continue
        if blocker_cells & ruin1_true:
            continue
        blocker_template = blocker_cells
        break
    if blocker_template is None:
        return None
    occupied |= blocker_template

    template1_ref = _place_with_padding(t1, height, width, occupied, rng, padding=1)
    if template1_ref is None:
        return None
    occupied |= template1_ref

    template2_ref = _place_with_padding(t2, height, width, occupied, rng, padding=1)
    if template2_ref is None:
        return None
    occupied |= template2_ref

    ruin2_full = _place_with_padding(t3, height, width, occupied, rng, padding=1)
    if ruin2_full is None:
        return None
    # Recover the shift from one cell pair.
    shift2 = None
    for cell in ruin2_full:
        for src in t3:
            cand = (cell[0] - src[0], cell[1] - src[1])
            if _translate(t3, cand[0], cand[1]) == ruin2_full:
                shift2 = cand
                break
        if shift2 is not None:
            break
    if shift2 is None:
        return None
    ruin2_residue = _translate(t3_residue, shift2[0], shift2[1])
    occupied |= ruin2_full

    scrap_shape = _make_scrap(rng, template_shapes)
    if scrap_shape is None:
        return None
    scrap_cells = _place_with_padding(scrap_shape, height, width, occupied, rng, padding=1)
    if scrap_cells is None:
        return None

    output = np.full((height, width), BACKGROUND, dtype=int)
    _paint(output, template1_ref, color_t1)
    _paint(output, template2_ref, color_t2)
    _paint(output, blocker_template, color_t3)
    _paint(output, ruin1_true, color_t1)
    _paint(output, ruin2_full, color_t3)

    inp = output.copy()
    for r, c in ruin1_true:
        inp[r, c] = BACKGROUND
    _paint(inp, ruin1_base, GREY)

    for r, c in ruin2_full:
        inp[r, c] = BACKGROUND
    _paint(inp, ruin2_residue, GREY)
    _paint(inp, scrap_cells, GREY)

    meta = _scene_metadata(inp)
    if len(meta["templates"]) != 3:
        return None
    if len(meta["matched"]) != 2 or len(meta["scraps"]) != 1:
        return None
    subset_counts = sorted(len(a["subset_candidates"]) for a in meta["matched"])
    feasible_counts = [len(a["feasible_candidates"]) for a in meta["matched"]]
    if feasible_counts != [1, 1]:
        return None
    if max(subset_counts) < 2 or min(subset_counts) < 1:
        return None
    if not any(len(a["subset_candidates"]) == 1 for a in meta["matched"]):
        return None
    if len(meta["scraps"][0]["subset_candidates"]) != 0:
        return None
    solved = main(inp)
    if not np.array_equal(solved, output):
        return None
    if np.array_equal(inp, output):
        return None
    return inp, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("failed to generate a valid repair-by-template input")
