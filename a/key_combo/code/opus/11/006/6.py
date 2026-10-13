from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

RECT_COLORS = (BLUE, RED, YELLOW)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 500


Rect = tuple[int, int, int, int]
Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sorted_seeds(seeds: list[Coord]) -> list[Coord]:
    # The statement says ties are broken from left to right, then top to bottom.
    return sorted(seeds, key=lambda rc: (rc[1], rc[0]))


def _rect_overlap(a: Rect, b: Rect, gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _extract_rectangles(grid: np.ndarray) -> list[tuple[int, Rect]]:
    rectangles: list[tuple[int, Rect]] = []
    for color in RECT_COLORS:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        assert coords.shape == (2, 2), "each rectangle color must appear exactly twice"
        rectangles.append((color, _bbox(coords)))
    assert len(rectangles) >= 1, "expected at least one rectangle-defining color pair"
    return rectangles


def _extract_seeds(grid: np.ndarray) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]


def _render_filled_rectangles(shape: tuple[int, int], rectangles: list[tuple[int, Rect]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, (r0, c0, r1, c1) in rectangles:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _barrier_mask(shape: tuple[int, int], rectangles: list[tuple[int, Rect]]) -> np.ndarray:
    barrier = np.zeros(shape, dtype=bool)
    for _color, (r0, c0, r1, c1) in rectangles:
        barrier[r0, c0 : c1 + 1] = True
        barrier[r1, c0 : c1 + 1] = True
        barrier[r0 : r1 + 1, c0] = True
        barrier[r0 : r1 + 1, c1] = True
    return barrier


def _analyze_growth(
    shape: tuple[int, int], rectangles: list[tuple[int, Rect]], seeds: list[Coord]
) -> dict:
    barrier = _barrier_mask(shape, rectangles)
    seed_set = set(seeds)
    sorted_seeds = _sorted_seeds(seeds)
    priority = {seed: idx for idx, seed in enumerate(sorted_seeds)}

    claims_by_cell: dict[Coord, list[tuple[int, int, int, Coord, Coord]]] = {}
    direction_stops: list[dict] = []
    per_seed_claims = {seed: 0 for seed in seeds}

    for seed in sorted_seeds:
        sr, sc = seed
        for dr, dc in DIRECTIONS:
            step = 1
            length = 0
            while True:
                r = sr + dr * step
                c = sc + dc * step
                if not _in_bounds(shape, r, c):
                    direction_stops.append(
                        {
                            "seed": seed,
                            "direction": (dr, dc),
                            "stop": "edge",
                            "length": length,
                        }
                    )
                    break
                if barrier[r, c]:
                    direction_stops.append(
                        {
                            "seed": seed,
                            "direction": (dr, dc),
                            "stop": "rectangle",
                            "length": length,
                            "barrier_cell": (r, c),
                        }
                    )
                    break

                length += 1
                if (r, c) not in seed_set:
                    color = PINK if step % 2 == 1 else ORANGE
                    claims_by_cell.setdefault((r, c), []).append(
                        (step, priority[seed], color, seed, (dr, dc))
                    )
                    per_seed_claims[seed] += 1
                step += 1

    chosen: dict[Coord, tuple[int, int, int, Coord, Coord]] = {}
    contested_cells: list[Coord] = []
    tie_cells: list[Coord] = []
    for cell, claims in claims_by_cell.items():
        claims.sort(key=lambda item: (item[0], item[1]))
        chosen[cell] = claims[0]
        if len(claims) > 1:
            contested_cells.append(cell)
            best_step = claims[0][0]
            if sum(claim[0] == best_step for claim in claims) > 1:
                tie_cells.append(cell)

    return {
        "barrier": barrier,
        "claims_by_cell": claims_by_cell,
        "chosen": chosen,
        "direction_stops": direction_stops,
        "contested_cells": contested_cells,
        "tie_cells": tie_cells,
        "per_seed_claims": per_seed_claims,
    }


def _compose_output(
    shape: tuple[int, int],
    rectangles: list[tuple[int, Rect]],
    seeds: list[Coord],
    chosen: dict[Coord, tuple[int, int, int, Coord, Coord]],
) -> np.ndarray:
    out = _render_filled_rectangles(shape, rectangles)
    for (r, c), claim in chosen.items():
        out[r, c] = claim[2]
    for r, c in seeds:
        out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each same-color diagonal pair into a solid rectangle, then let the green seeds grow in four directions until those rectangle borders stop them.

    Input structure:
    - A black background contains 2-3 colors chosen from blue, red, and yellow; each such color appears exactly twice as isolated pixels.
    - Each equal-color pair lies on different rows and columns and defines one axis-aligned rectangle; the rectangles do not overlap and stay away from the grid edge.
    - The grid also contains 1-2 isolated green seed pixels outside all of those rectangles.

    Rule:
    - For every non-green foreground color, fill the inclusive bounding rectangle spanned by its two pixels with that color.
    - Starting from every green seed simultaneously, grow rays upward, downward, leftward, and rightward; step 1 is pink, step 2 is orange, then the colors alternate.
    - A ray stops before entering a filled rectangle border cell or before leaving the grid; if several seeds claim the same cell, keep the one with the smaller step count, and break equal-step ties by seed order from left to right, then top to bottom.
    - After all growth is resolved, erase the original green seed cells back to black.

    Key insight:
    - The same-color pairs are not the final target by themselves: once filled, their rectangle borders become walls that carve the seed growth into obstacle-shaped cross traces.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _extract_rectangles(g)
    seeds = _extract_seeds(g)
    assert 1 <= len(seeds) <= 2, "expected one or two green seeds"

    filled = _render_filled_rectangles(g.shape, rectangles)
    for r, c in seeds:
        assert filled[r, c] == BLACK, "seed must be outside every rectangle"

    analysis = _analyze_growth(g.shape, rectangles, seeds)
    return _compose_output(g.shape, rectangles, seeds, analysis["chosen"])


def _all_input_points(rectangles: list[tuple[int, Rect]], seeds: list[Coord]) -> list[Coord]:
    pts: list[Coord] = []
    for _color, (r0, c0, r1, c1) in rectangles:
        pts.extend([(r0, c0), (r1, c1)])
    pts.extend(seeds)
    return pts


def _min_pairwise_manhattan(points: list[Coord]) -> int:
    if len(points) < 2:
        return 99
    return min(abs(r1 - r2) + abs(c1 - c2) for (r1, c1), (r2, c2) in combinations(points, 2))


def _validate_input(grid: np.ndarray) -> tuple[list[tuple[int, Rect]], list[Coord]]:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED, GREEN, YELLOW})

    rectangles = _extract_rectangles(grid)
    seeds = _extract_seeds(grid)
    assert 2 <= len(rectangles) <= 3
    assert 1 <= len(seeds) <= 2

    for idx, (_color, rect) in enumerate(rectangles):
        r0, c0, r1, c1 = rect
        assert r0 >= 1 and c0 >= 1 and r1 <= grid.shape[0] - 2 and c1 <= grid.shape[1] - 2
        assert r0 < r1 and c0 < c1
        for _other_color, other in rectangles[idx + 1 :]:
            assert not _rect_overlap(rect, other, gap=0)

    filled = _render_filled_rectangles(grid.shape, rectangles)
    for r, c in seeds:
        assert filled[r, c] == BLACK

    assert _min_pairwise_manhattan(_all_input_points(rectangles, seeds)) >= 2
    return rectangles, seeds


def _seed_metrics(shape: tuple[int, int], rectangles: list[tuple[int, Rect]], seed: Coord) -> dict:
    analysis = _analyze_growth(shape, rectangles, [seed])
    stops = analysis["direction_stops"]
    lengths = [stop["length"] for stop in stops]
    return {
        "seed": seed,
        "rect_stops": sum(stop["stop"] == "rectangle" for stop in stops),
        "edge_stops": sum(stop["stop"] == "edge" for stop in stops),
        "max_length": max(lengths),
        "total_length": sum(lengths),
        "output_cells": len(analysis["chosen"]),
    }


def _candidate_seeds(
    shape: tuple[int, int], rectangles: list[tuple[int, Rect]], marker_points: list[Coord]
) -> list[dict]:
    filled = _render_filled_rectangles(shape, rectangles)
    candidates: list[dict] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if filled[r, c] != BLACK:
                continue
            if min(abs(r - mr) + abs(c - mc) for mr, mc in marker_points) < 2:
                continue
            metrics = _seed_metrics(shape, rectangles, (r, c))
            if metrics["rect_stops"] < 1:
                continue
            if metrics["max_length"] < 2:
                continue
            if metrics["output_cells"] < 5:
                continue
            candidates.append(metrics)
    random.shuffle(candidates)
    return candidates


def _pair_quality(shape: tuple[int, int], rectangles: list[tuple[int, Rect]], seeds: list[Coord]) -> dict:
    analysis = _analyze_growth(shape, rectangles, seeds)
    output = _compose_output(shape, rectangles, seeds, analysis["chosen"])
    filled_only = _render_filled_rectangles(shape, rectangles)

    colors_after_growth = set(map(int, np.unique(output))) - {BLACK, BLUE, RED, YELLOW}
    barrier_hits = sum(stop["stop"] == "rectangle" for stop in analysis["direction_stops"])
    edge_hits = sum(stop["stop"] == "edge" for stop in analysis["direction_stops"])
    contributions = analysis["per_seed_claims"]

    return {
        "analysis": analysis,
        "output": output,
        "growth_cells": len(analysis["chosen"]),
        "contested": len(analysis["contested_cells"]),
        "ties": len(analysis["tie_cells"]),
        "barrier_hits": barrier_hits,
        "edge_hits": edge_hits,
        "colors_after_growth": colors_after_growth,
        "filled_only_diff": int(np.count_nonzero(output != filled_only)),
        "per_seed_claims": contributions,
    }


def _sample_rectangles(rng: np.random.Generator) -> tuple[tuple[int, int], list[tuple[int, Rect]]] | None:
    h = int(rng.integers(13, 19))
    w = int(rng.integers(13, 19))
    n_rects = int(rng.integers(2, 4))
    colors = list(rng.choice(np.array(RECT_COLORS), size=n_rects, replace=False))

    for _ in range(200):
        rectangles: list[tuple[int, Rect]] = []
        ok = True
        for color in colors:
            placed = False
            for _ in range(200):
                rh = int(rng.integers(2, min(7, h - 2) + 1))
                rw = int(rng.integers(2, min(7, w - 2) + 1))
                r0 = int(rng.integers(1, h - rh))
                c0 = int(rng.integers(1, w - rw))
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if any(_rect_overlap(rect, other_rect, gap=1) for _other_color, other_rect in rectangles):
                    continue
                rectangles.append((int(color), rect))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        marker_points = _all_input_points(rectangles, [])
        if _min_pairwise_manhattan(marker_points) < 2:
            continue
        return (h, w), rectangles
    return None


def _pick_single_seed(
    shape: tuple[int, int], rectangles: list[tuple[int, Rect]], candidates: list[dict]
) -> list[Coord] | None:
    good = [
        cand
        for cand in candidates
        if cand["rect_stops"] >= 1
        and cand["edge_stops"] >= 1
        and cand["max_length"] >= 3
        and cand["total_length"] >= 8
    ]
    if not good:
        return None
    chosen = random.choice(good)
    return [chosen["seed"]]


def _pick_seed_pair(
    shape: tuple[int, int], rectangles: list[tuple[int, Rect]], candidates: list[dict], want_tie: bool
) -> list[Coord] | None:
    pairs: list[tuple[list[Coord], dict]] = []
    for left, right in combinations(candidates, 2):
        s0 = left["seed"]
        s1 = right["seed"]
        if s0[0] == s1[0] or s0[1] == s1[1]:
            continue
        if abs(s0[0] - s1[0]) + abs(s0[1] - s1[1]) < 2:
            continue
        quality = _pair_quality(shape, rectangles, [s0, s1])
        if quality["growth_cells"] < 9:
            continue
        if quality["barrier_hits"] < 2:
            continue
        if quality["edge_hits"] < 2:
            continue
        if len(quality["colors_after_growth"]) < 2:
            continue
        if any(v < 2 for v in quality["per_seed_claims"].values()):
            continue
        if want_tie:
            if quality["ties"] < 1:
                continue
        else:
            if quality["contested"] < 1 or quality["ties"] != 0:
                continue
        pairs.append(([s0, s1], quality))

    if not pairs:
        return None

    if want_tie:
        pairs.sort(
            key=lambda item: (
                item[1]["ties"],
                item[1]["contested"],
                item[1]["growth_cells"],
                item[1]["filled_only_diff"],
            ),
            reverse=True,
        )
    else:
        pairs.sort(
            key=lambda item: (
                item[1]["contested"],
                item[1]["growth_cells"],
                item[1]["filled_only_diff"],
            ),
            reverse=True,
        )
    top = pairs[: min(8, len(pairs))]
    return random.choice(top)[0]


def _build_input(shape: tuple[int, int], rectangles: list[tuple[int, Rect]], seeds: list[Coord]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, (r0, c0, r1, c1) in rectangles:
        grid[r0, c0] = color
        grid[r1, c1] = color
    for r, c in seeds:
        grid[r, c] = GREEN
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    rectangles, seeds = _validate_input(grid)
    quality = _pair_quality(grid.shape, rectangles, seeds)
    output = main(grid)

    if not np.array_equal(output, quality["output"]):
        return False
    if quality["growth_cells"] < 5:
        return False
    if quality["barrier_hits"] < 1:
        return False
    if quality["filled_only_diff"] < 5:
        return False
    if len(quality["colors_after_growth"]) < 2:
        return False
    if len(seeds) == 2 and quality["contested"] < 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout = _sample_rectangles(np.random.default_rng())
        if layout is None:
            continue
        shape, rectangles = layout
        marker_points = _all_input_points(rectangles, [])
        candidates = _candidate_seeds(shape, rectangles, marker_points)
        if len(candidates) < 1:
            continue

        want_two = random.random() < 0.7
        want_tie = random.random() < 0.45
        seeds = None
        if want_two and len(candidates) >= 2:
            seeds = _pick_seed_pair(shape, rectangles, candidates, want_tie=want_tie)
        if seeds is None:
            seeds = _pick_single_seed(shape, rectangles, candidates)
        if seeds is None:
            continue

        if _min_pairwise_manhattan(_all_input_points(rectangles, seeds)) < 2:
            continue

        grid = _build_input(shape, rectangles, seeds)
        if _validate_semantics(grid):
            return grid

    raise RuntimeError("failed to generate valid ARC input")
