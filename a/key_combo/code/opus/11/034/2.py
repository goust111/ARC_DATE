from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
BLACK = 0
VOTE_COLORS = (1, 2, 3, 4)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            comp_id = len(comps)
            stack = [(r, c)]
            labels[r, c] = comp_id
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
            comps.append(cells)
    return labels, comps


def _region_data(grid: np.ndarray) -> tuple[np.ndarray, list[dict], int]:
    labels, comps = _components(grid != GREY)
    regions: list[dict] = []
    outer_candidates: list[int] = []
    h, w = grid.shape

    for region_id, cells in enumerate(comps):
        touches_border = any(
            r == 0 or c == 0 or r == h - 1 or c == w - 1
            for r, c in cells
        )
        if touches_border:
            outer_candidates.append(region_id)
        regions.append(
            {
                "id": region_id,
                "cells": cells,
                "touches_border": touches_border,
                "area": len(cells),
            }
        )

    assert len(outer_candidates) == 1, "expected exactly one outer region"
    return labels, regions, outer_candidates[0]


def _region_vote(grid: np.ndarray, cells: list[tuple[int, int]]) -> int:
    counts = Counter(
        int(grid[r, c])
        for r, c in cells
        if int(grid[r, c]) not in (BLACK, GREY)
    )
    assert counts, "each region must contain colored dots"
    assert len(counts) >= 2, "each region must contain at least two dot colors"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "vote color must be unique"
    return int(winners[0])


def _analyze(grid: np.ndarray) -> dict:
    labels, regions, outer_id = _region_data(grid)
    votes = {region["id"]: _region_vote(grid, region["cells"]) for region in regions}

    erasable: list[tuple[int, int]] = []
    kept_internal: list[tuple[int, int]] = []
    kept_outer: list[tuple[int, int]] = []

    h, w = grid.shape
    for r, c in np.argwhere(grid == GREY):
        rr, cc = int(r), int(c)
        adjacent = {
            int(labels[nr, nc])
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] != -1
        }
        assert adjacent, "every contour pixel must border at least one region"
        if outer_id in adjacent:
            kept_outer.append((rr, cc))
            continue
        if len(adjacent) != 2:
            kept_internal.append((rr, cc))
            continue
        a, b = sorted(adjacent)
        if votes[a] == votes[b]:
            erasable.append((rr, cc))
        else:
            kept_internal.append((rr, cc))

    return {
        "labels": labels,
        "regions": regions,
        "outer_id": outer_id,
        "votes": votes,
        "erasable": erasable,
        "kept_internal": kept_internal,
        "kept_outer": kept_outer,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each contour-defined region by majority vote, then erase internal boundaries between equal-vote regions.

    Input structure:
    - Gray 1-pixel closed contour lines partition the black background into several 4-connected regions.
    - Every region, including the outer region outside all loops, contains scattered single-pixel dots in colors 1/2/3/4.
    - Inside each region exactly one dot color appears strictly most often, while gray contour pixels never overlap those dots.

    Rule:
    - Treat gray as walls and find all 4-connected non-gray regions, including the outer region touching the grid border.
    - For each region, count its non-black non-gray dots and use the unique majority color as that region's vote color.
    - Fill only that region's black cells with its vote color, leaving the original colored dots unchanged.
    - For every gray contour pixel, erase it to the shared vote color only when it separates two non-outer regions with the same vote color; otherwise keep it gray.
    - Any contour pixel bordering the outer region is always preserved.

    Key insight:
    - Boundary removal depends on comparing two independently inferred region votes, so the solver must first segment regions and compute votes before deciding which contours disappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    info = _analyze(g)
    labels = info["labels"]
    votes = info["votes"]
    outer_id = info["outer_id"]

    out = g.copy()
    for region in info["regions"]:
        vote = votes[region["id"]]
        for r, c in region["cells"]:
            if int(out[r, c]) == BLACK:
                out[r, c] = vote

    h, w = g.shape
    for r, c in np.argwhere(g == GREY):
        rr, cc = int(r), int(c)
        adjacent = {
            int(labels[nr, nc])
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] != -1
        }
        if outer_id in adjacent or len(adjacent) != 2:
            continue
        a, b = sorted(adjacent)
        if votes[a] == votes[b]:
            out[rr, cc] = votes[a]

    return out


def _draw_rect_loop(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _sample_vote_assignment(
    inner_ids: list[int],
    annulus_id: int,
    outer_id: int,
    force_outer_match: bool,
) -> dict[int, int]:
    annulus_vote = random.choice(VOTE_COLORS)
    votes = {annulus_id: annulus_vote}

    same_index = random.randrange(len(inner_ids))
    diff_candidates = [i for i in range(len(inner_ids)) if i != same_index]
    diff_index = random.choice(diff_candidates) if diff_candidates else same_index

    for idx, region_id in enumerate(inner_ids):
        if idx == same_index:
            votes[region_id] = annulus_vote
        elif idx == diff_index:
            votes[region_id] = random.choice([c for c in VOTE_COLORS if c != annulus_vote])
        else:
            if random.random() < 0.5:
                votes[region_id] = annulus_vote
            else:
                votes[region_id] = random.choice([c for c in VOTE_COLORS if c != annulus_vote])

    if force_outer_match:
        votes[outer_id] = annulus_vote
    else:
        if random.random() < 0.5:
            votes[outer_id] = annulus_vote
        else:
            votes[outer_id] = random.choice([c for c in VOTE_COLORS if c != annulus_vote])
    return votes


def _region_dot_counts(capacity: int, vote_color: int) -> dict[int, int]:
    total = random.randint(3, min(8, capacity))
    others = [c for c in VOTE_COLORS if c != vote_color]
    random.shuffle(others)
    n_other_colors = random.randint(1, min(3, total - 1, len(others)))
    chosen_others = others[:n_other_colors]

    counts = {vote_color: 1}
    for color in chosen_others:
        counts[color] = 1

    remaining = total - len(counts)
    while remaining > 0:
        chosen = random.choice([vote_color] + chosen_others)
        counts[chosen] += 1
        remaining -= 1

    max_other = max(counts[color] for color in chosen_others)
    while counts[vote_color] <= max_other:
        donor_choices = [color for color in chosen_others if counts[color] > 1]
        if donor_choices:
            donor = random.choice(donor_choices)
            counts[donor] -= 1
            counts[vote_color] += 1
        else:
            counts[vote_color] += 1
            total += 1
            assert total <= capacity
        max_other = max(counts[color] for color in chosen_others)

    return counts


def _place_region_dots(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    vote_color: int,
) -> None:
    counts = _region_dot_counts(len(cells), vote_color)
    chosen_cells = random.sample(cells, sum(counts.values()))
    cursor = 0
    for color, count in counts.items():
        for r, c in chosen_cells[cursor : cursor + count]:
            grid[r, c] = color
        cursor += count


def _build_structure() -> np.ndarray | None:
    inner_count = random.randint(2, 4)

    slot_h = [random.randint(5, 7), random.randint(5, 7)]
    slot_w = [random.randint(5, 7), random.randint(5, 7)]
    gap_r = random.randint(1, 2)
    gap_c = random.randint(1, 2)
    margin_in = [random.randint(1, 2) for _ in range(4)]
    margin_out = [random.randint(2, 3) for _ in range(4)]

    outer_inner_h = margin_in[0] + slot_h[0] + gap_r + slot_h[1] + margin_in[1]
    outer_inner_w = margin_in[2] + slot_w[0] + gap_c + slot_w[1] + margin_in[3]
    grid_h = margin_out[0] + outer_inner_h + 2 + margin_out[1]
    grid_w = margin_out[2] + outer_inner_w + 2 + margin_out[3]
    if max(grid_h, grid_w) > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    outer_r0 = margin_out[0]
    outer_c0 = margin_out[2]
    outer_r1 = outer_r0 + outer_inner_h + 1
    outer_c1 = outer_c0 + outer_inner_w + 1
    _draw_rect_loop(grid, outer_r0, outer_c0, outer_r1, outer_c1)

    base_r = outer_r0 + 1 + margin_in[0]
    base_c = outer_c0 + 1 + margin_in[2]
    row_starts = [base_r, base_r + slot_h[0] + gap_r]
    col_starts = [base_c, base_c + slot_w[0] + gap_c]

    all_slots = [
        (row_starts[0], col_starts[0], row_starts[0] + slot_h[0] - 1, col_starts[0] + slot_w[0] - 1),
        (row_starts[0], col_starts[1], row_starts[0] + slot_h[0] - 1, col_starts[1] + slot_w[1] - 1),
        (row_starts[1], col_starts[0], row_starts[1] + slot_h[1] - 1, col_starts[0] + slot_w[0] - 1),
        (row_starts[1], col_starts[1], row_starts[1] + slot_h[1] - 1, col_starts[1] + slot_w[1] - 1),
    ]
    active_slots = random.sample(all_slots, inner_count)
    for r0, c0, r1, c1 in active_slots:
        _draw_rect_loop(grid, r0, c0, r1, c1)
    return grid


def _validate_generated_case(inp: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30

    info = _analyze(inp)
    regions = info["regions"]
    outer_id = info["outer_id"]
    votes = info["votes"]
    assert 4 <= len(regions) <= 6

    non_outer = [r for r in regions if r["id"] != outer_id]
    annulus = max(non_outer, key=lambda region: region["area"])
    annulus_id = annulus["id"]
    inner_ids = [r["id"] for r in non_outer if r["id"] != annulus_id]
    assert len(inner_ids) >= 2

    for region in regions:
        dot_cells = [(r, c) for r, c in region["cells"] if int(inp[r, c]) not in (BLACK, GREY)]
        assert 3 <= len(dot_cells) <= 8
        dot_colors = {int(inp[r, c]) for r, c in dot_cells}
        assert len(dot_colors) >= 2
        _ = _region_vote(inp, region["cells"])

    pair_votes = {votes[region_id] for region_id in inner_ids}
    assert votes[annulus_id] in pair_votes
    assert any(votes[region_id] == votes[annulus_id] for region_id in inner_ids)
    assert any(votes[region_id] != votes[annulus_id] for region_id in inner_ids)
    assert info["erasable"], "need at least one erased internal contour"
    assert info["kept_internal"], "need at least one preserved internal contour"
    assert info["kept_outer"], "outer contour must exist"

    out = main(inp)
    assert out.shape == inp.shape
    assert not np.array_equal(out, inp)

    for r, c in info["erasable"]:
        assert int(out[r, c]) != GREY
    for r, c in info["kept_internal"] + info["kept_outer"]:
        assert int(out[r, c]) == GREY

    if votes[outer_id] == votes[annulus_id]:
        # This is the non-degenerate exception branch from rule 5.
        assert any(int(out[r, c]) == GREY for r, c in info["kept_outer"])


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_structure()
        if grid is None:
            continue

        _labels, regions, outer_id = _region_data(grid)
        non_outer = [r for r in regions if r["id"] != outer_id]
        annulus = max(non_outer, key=lambda region: region["area"])
        annulus_id = annulus["id"]
        inner_ids = [r["id"] for r in non_outer if r["id"] != annulus_id]
        if len(inner_ids) < 2:
            continue

        force_outer_match = random.random() < 0.5
        votes = _sample_vote_assignment(inner_ids, annulus_id, outer_id, force_outer_match)

        for region in regions:
            if len(region["cells"]) < 3:
                break
            _place_region_dots(grid, region["cells"], votes[region["id"]])
        else:
            try:
                _validate_generated_case(grid)
            except Exception:
                continue
            return grid

    raise RuntimeError("failed to generate a valid contour-vote-erase input")
