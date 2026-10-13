from __future__ import annotations

import numpy as np


BG = 0
FRAME = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_RETRIES = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if _in_bounds(h, w, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _draw_rect_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    pts = {(int(r), int(c)) for r, c in coords}
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((int(r0), int(c)))
        expected.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        expected.add((int(r), int(c0)))
        expected.add((int(r), int(c1)))
    return pts == expected


def _find_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fences: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            if not _is_rect_frame(comp):
                continue
            bbox = _bbox(comp)
            fences.append(
                {
                    "color": int(color),
                    "bbox": tuple(map(int, bbox)),
                    "cells": {(int(r), int(c)) for r, c in comp},
                }
            )
    fences.sort(key=lambda item: item["bbox"])
    return fences


def _inside_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 <= r <= r1 and c0 <= c <= c1


def _inside_interior(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 < r < r1 and c0 < c < c1


def _distance_sq_to_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    dr = 0
    dc = 0
    if r < r0:
        dr = int(r0 - r)
    elif r > r1:
        dr = int(r - r1)
    if c < c0:
        dc = int(c0 - c)
    elif c > c1:
        dc = int(c - c1)
    return int(dr * dr + dc * dc)


def _find_arbitrators(grid: np.ndarray, fences: list[dict]) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    frame_cells = set()
    for fence in fences:
        frame_cells |= fence["cells"]

    arbitrators: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or (r, c) in frame_cells:
                continue
            if any(_inside_interior(r, c, fence["bbox"]) for fence in fences):
                continue
            isolated = True
            for dr, dc in DIR8:
                nr = int(r + dr)
                nc = int(c + dc)
                if not _in_bounds(h, w, nr, nc):
                    continue
                if int(g[nr, nc]) != BG:
                    isolated = False
                    break
            if isolated:
                arbitrators.append((int(r), int(c), int(color)))
    return arbitrators


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each fenced rectangle by the interior vote winner, using the nearest exterior pixel to break ties.

    Input structure:
    - Black background fills the grid outside all objects.
    - One-pixel-thick rectangular fence components enclose separate interior regions.
    - Inside each fence are scattered non-black, non-fence pixels in 2-4 colors.
    - Outside the fences are isolated non-black arbitration pixels.

    Rule:
    - Detect every closed rectangular fence and inspect the colors strictly inside it.
    - If one interior color has the unique highest count, repaint the whole fence border with that color.
    - If several interior colors tie for the highest count, use the color of the exterior arbitration pixel with the smallest geometric distance to that fence.
    - Remove every non-border colored pixel, leaving only the recolored fence borders on black.

    Key insight: The border color is not determined by geometry alone; it comes from a two-stage decision, with interior majority vote first and an exterior nearest-pixel tie-breaker only when the vote is tied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fences = _find_fences(g)
    assert fences, "expected at least one rectangular fence"
    arbitrators = _find_arbitrators(g, fences)

    out = np.full_like(g, BG)
    for fence in fences:
        r0, c0, r1, c1 = map(int, fence["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        vals = interior[(interior != BG) & (interior != int(fence["color"]))]
        assert vals.size > 0, "fence interior must contain colored pixels"

        colors, counts = np.unique(vals, return_counts=True)
        top = int(counts.max())
        winners = [int(color) for color, count in zip(colors, counts) if int(count) == top]
        if len(winners) == 1:
            winner = int(winners[0])
        else:
            assert arbitrators, "tied fence needs at least one arbitration pixel"
            ranked = sorted(
                arbitrators,
                key=lambda item: (
                    _distance_sq_to_bbox(int(item[0]), int(item[1]), fence["bbox"]),
                    abs(int(item[0]) - r0) + abs(int(item[1]) - c0),
                    int(item[0]),
                    int(item[1]),
                ),
            )
            winner = int(ranked[0][2])
        _draw_rect_border(out, fence["bbox"], winner)

    return out


def _boxes_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _choose_branches(rng: np.random.Generator, n_fences: int) -> list[str]:
    if n_fences == 1:
        return [str(rng.choice(np.array(["majority", "tie"], dtype=object)))]
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return ["majority"] * n_fences
    if mode == 1:
        return ["tie"] * n_fences
    n_ties = int(rng.integers(1, n_fences))
    branches = ["tie"] * n_ties + ["majority"] * (n_fences - n_ties)
    rng.shuffle(branches)
    return branches


def _positive_composition(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    out: list[int] = []
    prev = 0
    for cut in list(cuts) + [int(total)]:
        out.append(int(cut - prev))
        prev = int(cut)
    return out


def _sample_counts(
    rng: np.random.Generator,
    n_colors: int,
    capacity: int,
    tie: bool,
) -> list[int]:
    min_total = n_colors + (0 if tie else 1)
    max_total = min(int(capacity), int(n_colors + 8))
    if max_total < min_total:
        raise RuntimeError("impossible interior count request")
    for _ in range(200):
        total = int(rng.integers(min_total, max_total + 1))
        counts = _positive_composition(rng, total, n_colors)
        top = max(counts)
        n_top = sum(int(count == top) for count in counts)
        if tie and n_top >= 2:
            rng.shuffle(counts)
            return [int(x) for x in counts]
        if not tie and n_top == 1:
            rng.shuffle(counts)
            return [int(x) for x in counts]
    raise RuntimeError("failed to sample interior counts")


def _place_fences(
    rng: np.random.Generator,
    height: int,
    width: int,
    branches: list[str],
) -> list[dict] | None:
    fences: list[dict] = []
    for branch in branches:
        placed = False
        for _ in range(400):
            outer_h = int(rng.integers(6, min(11, height - 4) + 1))
            outer_w = int(rng.integers(6, min(11, width - 4) + 1))
            r0 = int(rng.integers(2, height - outer_h - 1))
            c0 = int(rng.integers(2, width - outer_w - 1))
            bbox = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))
            if all(_boxes_far_apart(bbox, other["bbox"], margin=1) for other in fences):
                fences.append({"bbox": bbox, "branch": str(branch)})
                placed = True
                break
        if not placed:
            return None
    fences.sort(key=lambda item: item["bbox"])
    return fences


def _safe_interior_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    return [
        (int(r), int(c))
        for r in range(r0 + 2, r1 - 1)
        for c in range(c0 + 2, c1 - 1)
    ]


def _is_valid_arbitrator_cell(
    grid: np.ndarray,
    fences: list[dict],
    r: int,
    c: int,
) -> bool:
    h, w = map(int, grid.shape)
    if not _in_bounds(h, w, r, c) or int(grid[r, c]) != BG:
        return False
    if any(_inside_bbox(r, c, fence["bbox"]) for fence in fences):
        return False
    for dr, dc in DIR8:
        nr = int(r + dr)
        nc = int(c + dc)
        if _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) != BG:
            return False
    return True


def _side_candidates(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    gap: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cells = set()
    for c in range(c0, c1 + 1):
        if _in_bounds(height, width, r0 - gap, c):
            cells.add((int(r0 - gap), int(c)))
        if _in_bounds(height, width, r1 + gap, c):
            cells.add((int(r1 + gap), int(c)))
    for r in range(r0, r1 + 1):
        if _in_bounds(height, width, r, c0 - gap):
            cells.add((int(r), int(c0 - gap)))
        if _in_bounds(height, width, r, c1 + gap):
            cells.add((int(r), int(c1 + gap)))
    return sorted(cells)


def _dedicated_arbitrator(
    rng: np.random.Generator,
    grid: np.ndarray,
    fences: list[dict],
    fence_idx: int,
    placed: list[dict],
    target_dist: dict[int, int],
) -> tuple[int, int] | None:
    height, width = map(int, grid.shape)
    bbox = fences[fence_idx]["bbox"]
    for gap in (2, 3, 4, 5):
        candidates = []
        for r, c in _side_candidates(bbox, height, width, gap):
            if not _is_valid_arbitrator_cell(grid, fences, r, c):
                continue
            dist_here = _distance_sq_to_bbox(r, c, bbox)
            if any(
                _distance_sq_to_bbox(int(item["r"]), int(item["c"]), bbox) <= dist_here
                for item in placed
            ):
                continue
            ok = True
            for other_idx, best_dist in target_dist.items():
                other_bbox = fences[int(other_idx)]["bbox"]
                if _distance_sq_to_bbox(r, c, other_bbox) <= int(best_dist):
                    ok = False
                    break
            if ok:
                candidates.append((int(r), int(c)))
        if candidates:
            return tuple(map(int, candidates[int(rng.integers(0, len(candidates)))]))
    return None


def _far_arbitrator_candidates(
    grid: np.ndarray,
    fences: list[dict],
    target_dist: dict[int, int],
) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    out: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if not _is_valid_arbitrator_cell(grid, fences, r, c):
                continue
            if any(
                _distance_sq_to_bbox(r, c, fences[idx]["bbox"]) <= int(best_dist)
                for idx, best_dist in target_dist.items()
            ):
                continue
            out.append((int(r), int(c)))
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()

    n_fences = int(rng.integers(1, 5))
    min_dim = min(30, 12 + 4 * n_fences)
    if min_dim > 30:
        return None
    height = int(rng.integers(min_dim, 31))
    width = int(rng.integers(min_dim, 31))

    branches = _choose_branches(rng, n_fences)
    fences = _place_fences(rng, height, width, branches)
    if fences is None:
        return None

    inp = np.full((height, width), BG, dtype=int)
    out = np.full_like(inp, BG)

    for fence in fences:
        _draw_rect_border(inp, fence["bbox"], FRAME)

    for fence in fences:
        safe_positions = _safe_interior_positions(fence["bbox"])
        capacity = len(safe_positions)
        if capacity < 4:
            return None

        max_colors = min(4, capacity - (0 if fence["branch"] == "tie" else 1))
        if max_colors < 2:
            return None
        n_colors = int(rng.integers(2, max_colors + 1))
        colors = [int(x) for x in rng.choice(PALETTE, size=n_colors, replace=False)]
        counts = _sample_counts(rng, n_colors, capacity, tie=fence["branch"] == "tie")
        total = int(sum(counts))

        chosen_idx = rng.choice(len(safe_positions), size=total, replace=False)
        chosen_positions = [safe_positions[int(i)] for i in chosen_idx]
        assignments = []
        for color, count in zip(colors, counts):
            assignments.extend([int(color)] * int(count))
        rng.shuffle(assignments)
        for (r, c), color in zip(chosen_positions, assignments):
            inp[int(r), int(c)] = int(color)

        fence["interior_colors"] = [int(x) for x in colors]
        fence["counts"] = [int(x) for x in counts]

        top = max(counts)
        winners = [int(color) for color, count in zip(colors, counts) if int(count) == top]
        if fence["branch"] == "majority":
            assert len(winners) == 1
            fence["winner"] = int(winners[0])
        else:
            fence["tied_colors"] = winners

    placed_arbitrators: list[dict] = []
    tie_dist: dict[int, int] = {}
    tie_indices = [idx for idx, fence in enumerate(fences) if fence["branch"] == "tie"]
    rng.shuffle(tie_indices)

    for idx in tie_indices:
        cell = _dedicated_arbitrator(rng, inp, fences, idx, placed_arbitrators, tie_dist)
        if cell is None:
            return None

        fence = fences[idx]
        outside_colors = [color for color in PALETTE if color not in fence["interior_colors"]]
        if outside_colors and bool(rng.random() < 0.7):
            arb_color = int(rng.choice(np.array(outside_colors, dtype=int)))
        else:
            arb_color = int(rng.choice(PALETTE))

        rr, cc = map(int, cell)
        inp[rr, cc] = int(arb_color)
        dist = _distance_sq_to_bbox(rr, cc, fence["bbox"])
        tie_dist[int(idx)] = int(dist)
        placed_arbitrators.append({"r": int(rr), "c": int(cc), "color": int(arb_color), "fence": int(idx)})
        fence["winner"] = int(arb_color)

    if tie_indices:
        max_extra = 3
        n_extra = int(rng.integers(0, max_extra + 1))
    else:
        n_extra = int(rng.integers(1, 4))

    extra_candidates = _far_arbitrator_candidates(inp, fences, tie_dist)
    rng.shuffle(extra_candidates)
    for r, c in extra_candidates[:n_extra]:
        color = int(rng.choice(PALETTE))
        inp[int(r), int(c)] = int(color)
        placed_arbitrators.append({"r": int(r), "c": int(c), "color": int(color), "fence": -1})

    if not placed_arbitrators:
        return None

    for fence in fences:
        _draw_rect_border(out, fence["bbox"], int(fence["winner"]))

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
