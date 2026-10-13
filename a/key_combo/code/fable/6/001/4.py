from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, label


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    struct = STRUCT4 if int(connectivity) == 4 else np.ones((3, 3), dtype=int)
    lab, n = label(mask.astype(bool), structure=struct)
    return [np.argwhere(lab == k) for k in range(1, int(n) + 1)]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _cell_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in coords}


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_to_cells(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(map(int, coord)) for coord in np.argwhere(mask))


def _strict_interior(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (
                mask[r, c]
                and mask[r - 1, c]
                and mask[r + 1, c]
                and mask[r, c - 1]
                and mask[r, c + 1]
            ):
                out[r, c] = True
    return out


def _expand_cells(cells: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _mask_to_cells(_trim_mask(mask))


@dataclass(frozen=True)
class Template:
    color: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Candidate:
    template_index: int
    color: int
    top: int
    left: int
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Match:
    hole_index: int
    hole_cells: tuple[tuple[int, int], ...]
    template_index: int
    color: int
    top: int
    left: int
    footprint: tuple[tuple[int, int], ...]
    exact: bool


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    fg_components = components_mask(g != int(bg), connectivity=4)
    assert len(fg_components) >= 3, "expected one large object and at least two templates"

    areas = [int(coords.shape[0]) for coords in fg_components]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "large object must be uniquely largest"
    big_idx = areas.index(max_area)

    big_coords = fg_components[big_idx]
    big_cells = _cell_set(big_coords)
    big_visible_mask = np.zeros_like(g, dtype=bool)
    for r, c in big_cells:
        big_visible_mask[r, c] = True

    big_colors = {int(g[r, c]) for r, c in big_cells}
    assert len(big_colors) == 1, "large object must be monochrome"
    big_color = next(iter(big_colors))

    original_mask = binary_fill_holes(big_visible_mask).astype(bool)
    hole_components = components_mask(original_mask & ~big_visible_mask, connectivity=4)
    assert hole_components, "large object must contain background gaps"
    hole_components.sort(key=lambda coords: bbox_coords(coords))

    templates: list[Template] = []
    for idx, coords in enumerate(fg_components):
        if idx == big_idx:
            continue
        colors = {int(g[r, c]) for r, c in _cell_set(coords)}
        assert len(colors) == 1, "each template must be monochrome"
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in _cell_set(coords):
            mask[r - r0, c - c0] = True
        templates.append(
            Template(
                color=next(iter(colors)),
                bbox=bbox,
                mask=mask,
                cells=_mask_to_cells(mask),
            )
        )
    templates.sort(key=lambda t: (t.mask.sum(), t.bbox[0], t.bbox[1], t.color))
    assert len(templates) >= 2
    return g, bg, big_color, big_visible_mask, original_mask, hole_components, templates


def _candidate_placements(
    grid: np.ndarray,
    bg: int,
    big_color: int,
    original_mask: np.ndarray,
    hole_cells: set[tuple[int, int]],
    template_index: int,
    template: Template,
) -> list[Candidate]:
    h, w = grid.shape
    th, tw = template.mask.shape
    candidates: list[Candidate] = []

    for top in range(h - th + 1):
        for left in range(w - tw + 1):
            cells = tuple((int(top + r), int(left + c)) for r, c in template.cells)
            if any(not original_mask[r, c] for r, c in cells):
                continue
            footprint = set(cells)
            if not hole_cells <= footprint:
                continue

            valid = True
            for r, c in cells:
                value = int(grid[r, c])
                if (r, c) in hole_cells:
                    if value != int(bg):
                        valid = False
                        break
                elif value != int(big_color):
                    valid = False
                    break
            if valid:
                candidates.append(
                    Candidate(
                        template_index=int(template_index),
                        color=int(template.color),
                        top=int(top),
                        left=int(left),
                        cells=cells,
                    )
                )
    return candidates


def _all_matches(grid: np.ndarray) -> tuple[np.ndarray, list[Match]]:
    g, bg, big_color, _big_visible, original_mask, hole_components, templates = _parse_scene(grid)

    matches: list[Match] = []
    planned: dict[tuple[int, int], int] = {}

    for hole_index, hole_coords in enumerate(hole_components):
        hole_cells = tuple(tuple(map(int, coord)) for coord in hole_coords)
        hole_set = set(hole_cells)
        candidates: list[Candidate] = []

        for template_index, template in enumerate(templates):
            candidates.extend(
                _candidate_placements(
                    g,
                    bg,
                    big_color,
                    original_mask,
                    hole_set,
                    template_index,
                    template,
                )
            )

        assert len(candidates) == 1, "each gap must have a unique template placement"
        chosen = candidates[0]
        exact = hole_set == set(chosen.cells)

        for cell in chosen.cells:
            previous = planned.get(cell)
            assert previous is None or previous == chosen.color, "template footprints must not conflict"
            planned[cell] = chosen.color

        matches.append(
            Match(
                hole_index=int(hole_index),
                hole_cells=hole_cells,
                template_index=int(chosen.template_index),
                color=int(chosen.color),
                top=int(chosen.top),
                left=int(chosen.left),
                footprint=chosen.cells,
                exact=bool(exact),
            )
        )

    return g, matches


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each interior background gap inside the large object into its uniquely containing template stamp.

    Input structure:
    - Black is the background.
    - One uniquely largest monochrome object contains several internal background gaps.
    - Outside that object sit several small monochrome template shapes of distinct visible footprints.

    Rule:
    - For each internal gap, find the unique template and untranslated orientation whose translated footprint fully lies inside the large object's range and contains the whole gap.
    - At that position, erase any remaining large-object pixels under the template footprint and repaint the entire footprint with the template's color.
    - Leave the original outside templates and all unrelated pixels unchanged.

    Key insight:
    - Each gap is only a partial erased remnant of a stamp, so matching is by containment under translation, not by exact shape equality.
    """
    g, matches = _all_matches(np.asarray(grid, dtype=int))
    out = g.copy()
    for match in matches:
        for r, c in match.footprint:
            out[r, c] = int(match.color)
    return out


BACKGROUND = 0
GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
BIG_H_RANGE = (12, 17)
BIG_W_RANGE = (12, 17)
TEMPLATE_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 400
OUTSIDE_PLACEMENT_ATTEMPTS = 60

TEMPLATE_LIBRARY = [
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
]


def _random_template_mask(used: set[tuple[tuple[int, int], ...]]) -> np.ndarray | None:
    for _ in range(80):
        mask = TEMPLATE_LIBRARY[random.randrange(len(TEMPLATE_LIBRARY))].copy()
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        mask = np.rot90(mask, random.randrange(4))
        mask = _trim_mask(mask)
        sig = _mask_signature(mask)
        if sig not in used:
            used.add(sig)
            return mask
    return None


def _make_big_mask() -> np.ndarray | None:
    for _ in range(100):
        h = random.randint(*BIG_H_RANGE)
        w = random.randint(*BIG_W_RANGE)
        mask = np.ones((h, w), dtype=bool)
        notch_count = random.choices([0, 1, 2], weights=[1, 3, 4], k=1)[0]

        for _ in range(notch_count):
            side = random.choice(("top", "bottom", "left", "right"))
            if side in {"top", "bottom"}:
                max_nh = min(4, h - 4)
                max_nw = min(6, w - 2)
                if max_nh < 1 or max_nw < 2:
                    continue
                nh = random.randint(1, max_nh)
                nw = random.randint(2, max_nw)
                c0 = random.randint(1, w - nw - 1)
                if side == "top":
                    mask[:nh, c0 : c0 + nw] = False
                else:
                    mask[h - nh :, c0 : c0 + nw] = False
            else:
                max_nh = min(6, h - 2)
                max_nw = min(4, w - 4)
                if max_nh < 2 or max_nw < 1:
                    continue
                nh = random.randint(2, max_nh)
                nw = random.randint(1, max_nw)
                r0 = random.randint(1, h - nh - 1)
                if side == "left":
                    mask[r0 : r0 + nh, :nw] = False
                else:
                    mask[r0 : r0 + nh, w - nw :] = False

        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if not np.array_equal(binary_fill_holes(mask).astype(bool), mask):
            continue
        strict = _strict_interior(mask)
        if int(mask.sum()) < 105 or int(strict.sum()) < 32:
            continue
        return mask
    return None


def _placement_groups(
    big_mask: np.ndarray,
    strict: np.ndarray,
    template_mask: np.ndarray,
) -> list[dict[str, object]]:
    big_h, big_w = big_mask.shape
    th, tw = template_mask.shape
    rel_cells = _mask_to_cells(template_mask)
    out: list[dict[str, object]] = []

    for top in range(big_h - th + 1):
        for left in range(big_w - tw + 1):
            cells = tuple((int(top + r), int(left + c)) for r, c in rel_cells)
            if any(not big_mask[r, c] for r, c in cells):
                continue

            local_interior = np.zeros((th, tw), dtype=bool)
            for r, c in rel_cells:
                if strict[top + r, left + c]:
                    local_interior[r, c] = True

            groups: list[tuple[tuple[int, int], ...]] = []
            for comp in components_mask(local_interior, connectivity=4):
                group = tuple((int(top + r), int(left + c)) for r, c in comp)
                if len(group) >= 2:
                    groups.append(group)

            out.append(
                {
                    "top": int(top),
                    "left": int(left),
                    "cells": cells,
                    "all_interior": bool(local_interior.sum() == template_mask.sum()),
                    "groups": groups,
                }
            )
    return out


def _random_connected_subset(cells: tuple[tuple[int, int], ...], target_size: int) -> tuple[tuple[int, int], ...] | None:
    cell_set = set(cells)
    if not 1 <= int(target_size) <= len(cell_set):
        return None

    for _ in range(40):
        start = random.choice(cells)
        chosen = {start}
        frontier = [nbr for nbr in neighbors4(*start) if nbr in cell_set]

        while frontier and len(chosen) < int(target_size):
            idx = random.randrange(len(frontier))
            cell = frontier.pop(idx)
            if cell in chosen:
                continue
            chosen.add(cell)
            for nbr in neighbors4(*cell):
                if nbr in cell_set and nbr not in chosen:
                    frontier.append(nbr)

        if len(chosen) == int(target_size):
            return tuple(sorted(chosen))
    return None


def _choose_hole(
    placement: dict[str, object],
    kind: str,
) -> tuple[tuple[int, int], ...] | None:
    cells = tuple(placement["cells"])
    if kind == "exact":
        return tuple(sorted(cells)) if placement["all_interior"] else None

    groups = list(placement["groups"])
    random.shuffle(groups)
    full_size = len(cells)
    for group in groups:
        max_size = len(group)
        if max_size >= full_size and placement["all_interior"]:
            max_size = full_size - 1
        if max_size < 2:
            continue
        min_size = 2
        target_size = random.randint(min_size, max_size)
        subset = _random_connected_subset(group, target_size)
        if subset is not None and set(subset) < set(cells):
            return subset
    return None


def _select_placements(big_mask: np.ndarray, template_infos: list[dict[str, object]]) -> list[dict[str, object]] | None:
    count = len(template_infos)
    exact_options = [i for i, info in enumerate(template_infos) if info["exacts"]]
    partial_options = [i for i, info in enumerate(template_infos) if info["partials"]]
    boundary_options = [i for i, info in enumerate(template_infos) if info["boundary_partials"]]
    if not exact_options or not partial_options:
        return None

    exact_idx = random.choice(exact_options)
    growth_pool = [i for i in (boundary_options or partial_options) if i != exact_idx]
    if not growth_pool:
        growth_pool = [i for i in partial_options if i != exact_idx]
    if not growth_pool:
        return None
    growth_idx = random.choice(growth_pool)

    kinds = ["partial"] * count
    kinds[exact_idx] = "exact"
    kinds[growth_idx] = "partial"
    for i in range(count):
        if i in {exact_idx, growth_idx}:
            continue
        if template_infos[i]["exacts"] and random.random() < 0.35:
            kinds[i] = "exact"

    pools: list[list[dict[str, object]]] = []
    for i, info in enumerate(template_infos):
        if i == growth_idx and info["boundary_partials"]:
            pool = list(info["boundary_partials"])
        elif kinds[i] == "exact":
            pool = list(info["exacts"])
        else:
            pool = list(info["partials"])
        random.shuffle(pool)
        if not pool:
            return None
        pools.append(pool)

    order = sorted(range(count), key=lambda i: len(pools[i]))
    chosen: list[dict[str, object] | None] = [None] * count

    def rec(pos: int, blocked: set[tuple[int, int]]) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for placement in pools[idx]:
            cells = tuple(placement["cells"])
            if any(cell in blocked for cell in cells):
                continue
            next_blocked = blocked | _expand_cells(cells, big_mask.shape)
            chosen[idx] = placement
            if rec(pos + 1, next_blocked):
                return True
            chosen[idx] = None
        return False

    if not rec(0, set()):
        return None

    result = []
    for i in range(count):
        result.append(
            {
                "placement": chosen[i],
                "kind": kinds[i],
                "force_boundary_growth": i == growth_idx and bool(template_infos[i]["boundary_partials"]),
            }
        )
    return result


def _place_templates_outside(
    grid_shape: tuple[int, int],
    big_bbox: tuple[int, int, int, int],
    templates: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = map(int, grid_shape)
    br0, bc0, br1, bc1 = map(int, big_bbox)
    blocked = np.zeros((h, w), dtype=bool)
    rr0 = max(0, br0 - 1)
    rr1 = min(h, br1 + 2)
    cc0 = max(0, bc0 - 1)
    cc1 = min(w, bc1 + 2)
    blocked[rr0:rr1, cc0:cc1] = True

    placements: list[tuple[int, int]] = []
    for mask in templates:
        th, tw = mask.shape
        candidates = []
        for top in range(h - th + 1):
            for left in range(w - tw + 1):
                r0 = max(0, top - 1)
                r1 = min(h, top + th + 1)
                c0 = max(0, left - 1)
                c1 = min(w, left + tw + 1)
                if blocked[r0:r1, c0:c1].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements.append((int(top), int(left)))
        r0 = max(0, top - 1)
        r1 = min(h, top + th + 1)
        c0 = max(0, left - 1)
        c1 = min(w, left + tw + 1)
        blocked[r0:r1, c0:c1] = True
    return placements


def _render_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in np.argwhere(mask):
        grid[int(top + r), int(left + c)] = int(color)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    big_mask = _make_big_mask()
    if big_mask is None:
        return None
    strict = _strict_interior(big_mask)

    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    template_masks: list[np.ndarray] = []
    for _ in range(template_count):
        mask = _random_template_mask(used_shapes)
        if mask is None:
            return None
        template_masks.append(mask)

    template_infos: list[dict[str, object]] = []
    for mask in template_masks:
        placements = _placement_groups(big_mask, strict, mask)
        exacts = [p for p in placements if p["all_interior"]]
        partials = [p for p in placements if p["groups"]]
        boundary_partials = [p for p in partials if not p["all_interior"]]
        template_infos.append(
            {
                "mask": mask,
                "placements": placements,
                "exacts": exacts,
                "partials": partials,
                "boundary_partials": boundary_partials,
            }
        )

    selected = _select_placements(big_mask, template_infos)
    if selected is None:
        return None

    hole_records: list[dict[str, object]] = []
    removed: set[tuple[int, int]] = set()
    for info, template_info in zip(selected, template_infos):
        placement = info["placement"]
        hole = _choose_hole(placement, info["kind"])
        if hole is None:
            return None
        if any(cell in removed for cell in hole):
            return None
        removed |= set(hole)
        hole_records.append(
            {
                "template_mask": template_info["mask"],
                "footprint": tuple(sorted(placement["cells"])),
                "hole": tuple(sorted(hole)),
                "all_interior": bool(placement["all_interior"]),
                "top": int(placement["top"]),
                "left": int(placement["left"]),
                "kind": info["kind"],
                "force_boundary_growth": bool(info["force_boundary_growth"]),
            }
        )

    visible_big = big_mask.copy()
    for r, c in removed:
        visible_big[r, c] = False
    if len(components_mask(visible_big, connectivity=4)) != 1:
        return None

    hole_mask = big_mask & ~visible_big
    hole_components = components_mask(hole_mask, connectivity=4)
    if len(hole_components) != template_count:
        return None

    exact_count = sum(record["hole"] == record["footprint"] for record in hole_records)
    partial_count = template_count - exact_count
    if exact_count < 1 or partial_count < 1:
        return None

    big_h, big_w = big_mask.shape
    grid_h = random.randint(max(GRID_H_RANGE[0], big_h + 7), GRID_H_RANGE[1])
    grid_w = random.randint(max(GRID_W_RANGE[0], big_w + 7), GRID_W_RANGE[1])
    big_top = random.randint(2, grid_h - big_h - 2)
    big_left = random.randint(2, grid_w - big_w - 2)

    outside = _place_templates_outside(
        (grid_h, grid_w),
        (big_top, big_left, big_top + big_h - 1, big_left + big_w - 1),
        template_masks,
    )
    if outside is None:
        return None

    colors = [c for c in range(1, 10)]
    big_color = random.choice(colors)
    template_colors = random.sample([c for c in colors if c != big_color], template_count)

    inp = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    out = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    _render_mask(inp, visible_big, big_top, big_left, big_color)
    _render_mask(out, big_mask, big_top, big_left, big_color)

    planned_matches: list[tuple[int, frozenset[tuple[int, int]], frozenset[tuple[int, int]], bool]] = []
    for idx, (template_mask, (ttop, tleft), color, record) in enumerate(
        zip(template_masks, outside, template_colors, hole_records)
    ):
        _render_mask(inp, template_mask, ttop, tleft, color)
        _render_mask(out, template_mask, ttop, tleft, color)

        footprint_global = tuple((big_top + r, big_left + c) for r, c in record["footprint"])
        hole_global = tuple((big_top + r, big_left + c) for r, c in record["hole"])
        for r, c in footprint_global:
            out[r, c] = int(color)
        planned_matches.append(
            (
                int(color),
                frozenset(footprint_global),
                frozenset(hole_global),
                bool(frozenset(footprint_global) == frozenset(hole_global)),
            )
        )

    try:
        solved_grid, inferred_matches = _all_matches(inp)
        assert np.array_equal(solved_grid, inp)
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, out):
        return None

    inferred_signature = sorted(
        (
            int(match.color),
            frozenset(match.footprint),
            frozenset(match.hole_cells),
            bool(match.exact),
        )
        for match in inferred_matches
    )
    if inferred_signature != sorted(planned_matches):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _gen_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate valid input for puzzle 4")
