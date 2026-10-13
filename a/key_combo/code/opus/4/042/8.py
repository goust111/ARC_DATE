from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MASK = 7
CONTROL_PALETTE = (1, 2, 3, 4, 6, 8, 9)
TEMPLATE_H = np.array([[1, 1], [0, 0]], dtype=bool)
TEMPLATE_V = np.array([[1, 0], [1, 0]], dtype=bool)
TEMPLATE_L = np.array([[1, 1], [1, 0]], dtype=bool)
TEMPLATE_C = np.array([[1, 0], [1, 1]], dtype=bool)
TEMPLATE_U = np.array([[1, 1], [0, 1]], dtype=bool)
TEMPLATE_D = np.array([[1, 0], [0, 1]], dtype=bool)
TEMPLATE_LIBRARY = (TEMPLATE_H, TEMPLATE_V, TEMPLATE_L, TEMPLATE_C, TEMPLATE_U, TEMPLATE_D)
SUBGRID_SIZE_CHOICES = (9, 11, 13)
GENERATION_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _orbit4(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = int(n) - 1
    return tuple(
        sorted(
            {
                (int(r), int(c)),
                (int(c), int(last - r)),
                (int(last - r), int(last - c)),
                (int(last - c), int(r)),
            }
        )
    )


def _quarter_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(int(n)):
        for c in range(int(n)):
            orbit = _orbit4(n, r, c)
            if orbit not in seen:
                seen.add(orbit)
                out.append(orbit)
    return out


def _extract_legend(g: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = map(int, g.shape)
    templates: dict[int, np.ndarray] = {}
    marker_cells = np.zeros(g.shape, dtype=bool)
    grey_cover = np.zeros(g.shape, dtype=bool)

    for r in range(h - 2):
        for c in range(w - 2):
            color = int(g[r, c])
            if color in (BG, GREY, MASK):
                continue
            tile = g[r + 1 : r + 3, c + 1 : c + 3]
            if not bool(np.isin(tile, (BG, GREY)).all()):
                continue
            mask = tile == GREY
            if int(mask.sum()) not in (2, 3):
                continue
            if color in templates:
                assert np.array_equal(templates[color], mask), "duplicate legend color must map to the same template"
            else:
                templates[color] = mask.copy()
            marker_cells[r, c] = True
            grey_cover[r + 1 : r + 3, c + 1 : c + 3] |= mask

    assert 3 <= len(templates) <= 4, "expected 3-4 legend entries"
    assert np.array_equal(grey_cover, g == GREY), "grey pixels must belong to legend templates"
    legend_cells = marker_cells | grey_cover
    return templates, legend_cells


def _find_subgrid(g: np.ndarray, legend_cells: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray]:
    active = (g != BG) & ~legend_cells
    box = _bbox(active)
    assert box is not None, "expected a marked lower-right subgrid"
    r0, c0, r1, c1 = box
    n = int(r1 - r0 + 1)
    assert n == int(c1 - c0 + 1), "subgrid must be square"
    assert n % 2 == 1, "subgrid side length must be odd"
    return (r0, c0, n), g[r0 : r1 + 1, c0 : c1 + 1].copy()


def _complete_subgrid(subgrid: np.ndarray) -> tuple[np.ndarray, int, int]:
    n, w = map(int, subgrid.shape)
    assert n == w, "subgrid must be square"
    out = subgrid.copy()
    added_controls = 0
    restored_black = 0

    for orbit in _quarter_orbits(n):
        values = [int(subgrid[r, c]) for r, c in orbit]
        visible = {v for v in values if v != MASK}
        assert len(visible) == 1, "each quarter-turn orbit must expose exactly one visible color"
        fill = visible.pop()
        for r, c in orbit:
            if int(subgrid[r, c]) == MASK:
                out[r, c] = fill
                if fill == BG:
                    restored_black += 1
                else:
                    added_controls += 1
            else:
                assert int(subgrid[r, c]) == fill, "visible orbit members must already match the fill color"

    assert not bool(np.any(out == MASK)), "all masks must be restored"
    return out, int(added_controls), int(restored_black)


def _render_stamps(
    completed: np.ndarray, templates: dict[int, np.ndarray]
) -> tuple[np.ndarray, dict[str, int]]:
    n, w = map(int, completed.shape)
    assert n == w, "completed control area must be square"
    center = n // 2
    cell_candidates: dict[tuple[int, int], list[tuple[int, int, int, tuple[int, int]]]] = {}

    for r, c in np.argwhere(completed != BG):
        rr = int(r)
        cc = int(c)
        color = int(completed[rr, cc])
        assert color in templates, "every control color must have a legend entry"
        mask = templates[color]
        assert bool(mask[0, 0]), "generator expects anchor-preserving templates"
        assert rr + 1 < n and cc + 1 < n, "every control pixel must support an in-bounds 2x2 stamp"
        dist = abs(rr - center) + abs(cc - center)
        for dr, dc in np.argwhere(mask):
            tr = rr + int(dr)
            tc = cc + int(dc)
            cell_candidates.setdefault((tr, tc), []).append((dist, rr, color, (rr, cc)))

    rendered = np.zeros_like(completed)
    overlap_cells = 0
    distance_conflicts = 0
    row_tie_conflicts = 0

    for (r, c), candidates in cell_candidates.items():
        if len(candidates) > 1:
            overlap_cells += 1
            dists = {cand[0] for cand in candidates}
            if len(dists) > 1:
                distance_conflicts += 1
            elif len({cand[1] for cand in candidates}) > 1:
                row_tie_conflicts += 1

        winner = min(candidates, key=lambda cand: (cand[0], cand[1]))
        for cand in candidates:
            if (cand[0], cand[1]) == (winner[0], winner[1]):
                assert cand[3] == winner[3], "conflict remains ambiguous after applying the puzzle tie-breaks"
        rendered[int(r), int(c)] = int(winner[2])

    meta = {
        "overlap_cells": int(overlap_cells),
        "distance_conflicts": int(distance_conflicts),
        "row_tie_conflicts": int(row_tie_conflicts),
    }
    return rendered, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete quarter-turn-symmetric masks to reveal hidden control pixels, then stamp each control color's legend template in place.

    Input structure:
    - A black background contains a top-left legend and a separate lower-right odd-sized square subgrid.
    - Each legend entry is a colored marker with a 2x2 grey template one row below and one column right of it.
    - The subgrid contains colored control pixels, orange mask pixels, and black background; its hidden full pattern is quarter-turn symmetric.

    Rule:
    - Parse the legend as color -> 2x2 template mask.
    - In the lower-right subgrid, replace every orange mask by the unique visible non-orange color in its 0/90/180/270 degree rotation orbit around the subgrid center.
    - Treat every restored non-black cell as a control pixel, and stamp its legend template with that control color using the control pixel as the stamp's top-left anchor.
    - If multiple stamps color the same output cell, keep the stamp whose anchor has smaller Manhattan distance to the subgrid center; if those distances tie, keep the one with the smaller anchor row.
    - Leave the legend unchanged and replace the subgrid by the resolved stamp rendering.

    Key insight:
    - The masks do not hide decoration; they hide the control anchors themselves, and only the quarter-turn completion reveals which legend shape must be stamped where.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"

    templates, legend_cells = _extract_legend(g)
    (r0, c0, n), subgrid = _find_subgrid(g, legend_cells)
    completed, _added_controls, _restored_black = _complete_subgrid(subgrid)
    stamped, _meta = _render_stamps(completed, templates)

    out = g.copy()
    out[r0 : r0 + n, c0 : c0 + n] = stamped
    return out


def _orbit_is_anchor_safe(orbit: tuple[tuple[int, int], ...], n: int) -> bool:
    return all(1 <= r <= n - 2 and 1 <= c <= n - 2 for r, c in orbit)


def _orbits_too_close(
    orbit_a: tuple[tuple[int, int], ...], orbit_b: tuple[tuple[int, int], ...]
) -> bool:
    for r1, c1 in orbit_a:
        for r2, c2 in orbit_b:
            if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) <= 1:
                return True
    return False


def _orbit_matches_seed(
    orbit: tuple[tuple[int, int], ...], seed: tuple[int, int], n: int
) -> bool:
    return orbit == _orbit4(n, int(seed[0]), int(seed[1]))


def _choose_extra_orbits(
    n: int,
    forbidden: list[tuple[tuple[int, int], ...]],
    count: int,
) -> list[tuple[tuple[int, int], ...]]:
    candidates = [
        orbit
        for orbit in _quarter_orbits(n)
        if len(orbit) == 4
        and _orbit_is_anchor_safe(orbit, n)
        and all(orbit != other for other in forbidden)
        and all(not _orbits_too_close(orbit, other) for other in forbidden)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[tuple[int, int], ...]] = []
    for orbit in candidates:
        if all(not _orbits_too_close(orbit, other) for other in chosen):
            chosen.append(orbit)
            if len(chosen) == count:
                break
    return chosen


def _apply_black_masks(
    subgrid: np.ndarray,
    orbit: tuple[tuple[int, int], ...],
    masked_count: int,
) -> None:
    max_masked = len(orbit) - 1
    assert max_masked >= 1, "a masked-black orbit must leave one visible black witness"
    masked_count = min(int(masked_count), int(max_masked))
    masked = random.sample(list(orbit), masked_count)
    for r, c in masked:
        assert int(subgrid[r, c]) == BG
        subgrid[r, c] = MASK


def _visible_only_render(
    subgrid: np.ndarray, templates: dict[int, np.ndarray]
) -> np.ndarray:
    partial = subgrid.copy()
    partial[partial == MASK] = BG
    rendered, _meta = _render_stamps(partial, templates)
    return rendered


def _build_input_and_expected(
    n: int,
    legend_order: list[int],
    templates: dict[int, np.ndarray],
    control_orbits: list[tuple[tuple[int, int], ...]],
    orbit_colors: list[int],
    row_tie_branch: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    complete = np.zeros((n, n), dtype=int)
    for orbit, color in zip(control_orbits, orbit_colors):
        for r, c in orbit:
            complete[int(r), int(c)] = int(color)

    subgrid = np.zeros_like(complete)
    for orbit, color in zip(control_orbits, orbit_colors):
        if len(orbit) == 1:
            r, c = orbit[0]
            subgrid[int(r), int(c)] = int(color)
            continue
        visible = random.choice(list(orbit))
        for r, c in orbit:
            subgrid[int(r), int(c)] = int(color) if (int(r), int(c)) == visible else MASK

    mid = n // 2
    black_orbits = [
        _orbit4(n, 0, 0),
        _orbit4(n, 0, max(1, mid - 1)),
    ]
    for orbit in black_orbits:
        _apply_black_masks(subgrid, orbit, random.choice((2, 3)))

    if random.random() < 0.5:
        extra_black = [
            orbit
            for orbit in _quarter_orbits(n)
            if orbit not in black_orbits
            and len(orbit) >= 2
            and all(int(subgrid[r, c]) == BG for r, c in orbit)
        ]
        random.shuffle(extra_black)
        if extra_black:
            _apply_black_masks(subgrid, extra_black[0], random.choice((1, 2)))

    completed, added_controls, restored_black = _complete_subgrid(subgrid)
    if not np.array_equal(completed, complete):
        return None
    if added_controls < 3 or restored_black < 1:
        return None

    color_usage = {int(color) for color in np.unique(complete) if int(color) != BG}
    if color_usage != set(legend_order):
        return None

    stamped, meta = _render_stamps(completed, templates)
    if meta["overlap_cells"] < 1:
        return None
    if row_tie_branch and meta["row_tie_conflicts"] < 1:
        return None
    if not row_tie_branch and meta["distance_conflicts"] < 1:
        return None
    if meta["overlap_cells"] > 10:
        return None

    if np.array_equal(_visible_only_render(subgrid, templates), stamped):
        return None

    if len({tuple(mask.astype(int).ravel()) for mask in templates.values()}) != len(templates):
        return None

    legend_row = random.randint(1, 2)
    legend_col = random.randint(1, 2)
    spacing = random.randint(3, 4)
    legend_width = spacing * (len(legend_order) - 1) + 3
    gap_rows = random.randint(2, 4)
    gap_cols = random.randint(2, 4)
    sg_r0_min = legend_row + 3 + gap_rows
    sg_c0_min = legend_col + legend_width + gap_cols
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    sg_r0_max = 30 - n - bottom_margin
    sg_c0_max = 30 - n - right_margin
    if sg_r0_min > sg_r0_max or sg_c0_min > sg_c0_max:
        return None
    sg_r0 = random.randint(sg_r0_min, sg_r0_max)
    sg_c0 = random.randint(sg_c0_min, sg_c0_max)
    h = sg_r0 + n + bottom_margin
    w = sg_c0 + n + right_margin

    inp = np.zeros((h, w), dtype=int)
    for idx, color in enumerate(legend_order):
        c = legend_col + idx * spacing
        inp[legend_row, c] = int(color)
        inp[legend_row + 1 : legend_row + 3, c + 1 : c + 3] = np.where(templates[color], GREY, BG)
    inp[sg_r0 : sg_r0 + n, sg_c0 : sg_c0 + n] = subgrid

    expected = inp.copy()
    expected[sg_r0 : sg_r0 + n, sg_c0 : sg_c0 + n] = stamped
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(SUBGRID_SIZE_CHOICES)
    num_entries = random.choice((3, 4))
    colors = random.sample(list(CONTROL_PALETTE), num_entries)
    row_tie_branch = random.random() < 0.45

    colors_for_pair = colors[:2]
    remaining_colors = colors[2:]
    if row_tie_branch:
        pair_templates = [TEMPLATE_V, TEMPLATE_H]
        seed_positions = [(2, 3), (3, 2)]
    else:
        pair_templates = [TEMPLATE_H, TEMPLATE_V]
        seed_positions = [(2, 2), (2, 3)]

    remaining_templates = [mask for mask in TEMPLATE_LIBRARY if all(not np.array_equal(mask, t) for t in pair_templates)]
    random.shuffle(remaining_templates)
    templates: dict[int, np.ndarray] = {}
    for color, mask in zip(colors_for_pair, pair_templates):
        templates[int(color)] = mask.copy()
    for color, mask in zip(remaining_colors, remaining_templates[: len(remaining_colors)]):
        templates[int(color)] = mask.copy()

    orbits = _quarter_orbits(n)
    pair_orbits = []
    for seed in seed_positions:
        matched = [orbit for orbit in orbits if _orbit_matches_seed(orbit, seed, n)]
        if len(matched) != 1 or not _orbit_is_anchor_safe(matched[0], n):
            return None
        pair_orbits.append(matched[0])
    if pair_orbits[0] == pair_orbits[1]:
        return None

    forbidden = pair_orbits.copy()
    extra_needed = len(remaining_colors) + random.randint(0, 1)
    extra_orbits = _choose_extra_orbits(n, forbidden, extra_needed)
    if len(extra_orbits) != extra_needed:
        return None

    control_orbits = pair_orbits + extra_orbits
    orbit_colors = colors_for_pair + remaining_colors + random.choices(colors, k=len(extra_orbits) - len(remaining_colors))
    if len(control_orbits) != len(orbit_colors):
        return None

    legend_order = colors.copy()
    random.shuffle(legend_order)
    built = _build_input_and_expected(n, legend_order, templates, control_orbits, orbit_colors, row_tie_branch)
    if built is None:
        return None
    inp, expected = built

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid symmetry-completion stamp puzzle")
