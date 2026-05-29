# Combo variant of 28a6681f (index 16)
# combo_spec: D1/3+D7/3
# Name: 悬浮平台+BFS洪水扩散
#
# Design analysis
# ---------------
# a) Concrete fused behavior and traced example:
#    - Original family: water is a set of supported blue pixels that later move one-by-one under
#      gravity into bottom-anchored basins. The visible answer is always a low, supported puddle.
#    - Fused family: the water source is still a supported side pocket, but the receiving structure
#      is now a suspended midair chamber with multiple floating horizontal bars. The solver no longer
#      treats water as independent falling particles. Instead, every input water pixel contributes
#      one unit of mass and one BFS source location; the source pocket is cleared, then a 4-neighbor
#      wave expands through background cells.
#    - Concrete canonical trace:
#      * the input contains a 4x3 water block packed into the left edge pocket, with only one exit
#        opening into a floating chamber;
#      * inside the chamber, the first horizontal bar blocks the direct route downward, so the BFS
#        wave reaches both side corridors and visibly splits into a left and a right branch;
#      * one row lower, those branches reconnect in the middle band, then a second suspended bar
#        forces the frontier to progress through a narrower lower path;
#      * under the original gravity rule, the same 12 water pixels slide and fall to the lower rows
#        of the chamber; under the fused rule, the nearest reachable cells in the upper chamber fill
#        first, so the output remains high, spread-out, and partly unsupported.
# b) Complexity calibration:
#    - The geometry is limited to one clear repeated family: a side source pocket plus one suspended
#      rectangular chamber containing three floating bars. A horizontal mirror produces the right-fed
#      counterpart. This is enough variation for ARC-style inference without introducing unrelated
#      shape grammars.
#    - The source volume is fixed at 12 pixels because this exactly fills seven complete BFS layers
#      in the canonical chamber. Avoiding partial final layers removes low-signal reading-order ties
#      and makes the wavefront shape easier to infer from a few examples.
#    - Three solid colors are used per sample: one for the source pocket support, one for the outer
#      floating chamber, and one for the internal bars. Colors remain decorative only.
# c) Hidden-assumption audit of the original main():
#    - The original `component_count(...)` / guide-detection block assumes that upper non-bottom
#      solids are special guide pieces and that important structure identity can be inferred from
#      whether a color appears on the bottom row. That is invalid here because the main obstacles are
#      intentionally floating and never touch the bottom row. The entire parsing block must be removed.
#    - The original `settle(...)` loop assumes each water pixel is an independent particle that can
#      repeatedly fall, slide diagonally, and slide horizontally until stable. D1/3 breaks that core
#      assumption: water now moves as a synchronous BFS frontier, so `settle(...)` and the outer
#      relaxation passes are replaced by a single layer-by-layer diffusion plan.
#    - The original output check requires every water pixel to have support on the left, below, and
#      right. That is a consequence of gravity settling and is no longer valid. In the fused task,
#      unsupported high water is expected and is one of the main visual cues.
#    - The original input-side support check still survives. The generator keeps the source as a
#      stable packed pocket so the input visually remains "stored water waiting to be released".
# d) Placement legality:
#    - The source pocket, suspended chamber, and internal bars are drawn from explicit coordinates on
#      a 10x10 board. They never overlap illegally, and the only intended connection between source
#      and chamber is a single side opening.
#    - No solid pixel touches the bottom row. The floating nature is therefore visually explicit and
#      never accidental.
#    - Internal bars are placed strictly inside the chamber walls, leaving enough corridor cells on
#      both sides of the first bar so that the BFS wave can visibly split and later rejoin.
# e) Visual observability:
#    - From examples alone, a human can see that the source pocket empties into a suspended maze and
#      that the output prefers the nearest chamber cells instead of the deepest ones.
#    - The split/merge behavior is directly visible because the output occupies both side corridors
#      around the first bar and also occupies the central band below that split.
#    - The rule does not rely on hidden numeric ordering of colors or coordinates; everything needed
#      for inference is encoded by visible geometry.
# f) Visual cleanliness:
#    - The board is divided into three readable regions: source pocket on one side, floating chamber
#      in the center, and empty border around them. This keeps the scene from collapsing into a dense
#      all-over flood-fill puzzle.
#    - Bars are simple rectilinear segments with one row of separation between them, so the layered
#      split/merge pattern stays legible.
# g) Change inferability / color discipline:
#    - Non-zero colors do not change role across space inside a sample. They are all rigid solids.
#    - The important learnable changes are geometric: suspended obstacles replace bottom basins, and
#      BFS distance replaces gravity settling. Because both cues are repeated on every accepted sample,
#      the fusion is inferable from a small set of demonstrations.
# 9) New solver assumptions guaranteed by generate_input():
#    - Assumption: at least one source water pixel touches background through the intended outlet.
#      Guarantee: the source pocket is sealed everywhere except a single side opening.
#    - Assumption: the reachable background region contains at least as many cells as the source water
#      count. Guarantee: candidates are validated by running the solver and rejecting any shortage.
#    - Assumption: partial final BFS layers would create a weaker visual signal. Guarantee: the source
#      size and chamber geometry are chosen so accepted samples fill only whole BFS layers.
#    - Assumption: the fused behavior should not collapse back to the original gravity output.
#      Guarantee: the generator measures the difference versus a gravity baseline and rejects any
#      candidate with too little divergence.

from __future__ import annotations

import random

import numpy as np


BG = 0
WATER = 1
GRID_SHAPE = (10, 10)
COLOR_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 200

SOURCE_ROWS = slice(0, 4)
SOURCE_COLS = slice(0, 3)
SOURCE_WATER_COUNT = 12

CANONICAL_SPLIT_LEFT = ((3, 4), (4, 4))
CANONICAL_SPLIT_RIGHT = ((3, 7), (4, 7))
CANONICAL_MERGE_ZONE = ((4, 5), (4, 6))
CANONICAL_LOWER_PROGRESS = ((5, 6), (6, 6), (5, 7))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _water_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(np.asarray(grid) == int(WATER))]


def _mirror_points(points: tuple[tuple[int, int], ...], width: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(width - 1 - c)) for r, c in points)


def _source_is_mirrored(grid: np.ndarray) -> bool:
    cols = [c for _r, c in _water_positions(grid)]
    assert cols, "expected source water"
    return float(np.mean(cols)) > (grid.shape[1] - 1) / 2.0


def _feature_points(grid: np.ndarray) -> dict[str, tuple[tuple[int, int], ...]]:
    if _source_is_mirrored(grid):
        width = int(grid.shape[1])
        return {
            "split_left": _mirror_points(CANONICAL_SPLIT_RIGHT, width),
            "split_right": _mirror_points(CANONICAL_SPLIT_LEFT, width),
            "merge_zone": _mirror_points(CANONICAL_MERGE_ZONE, width),
            "lower_progress": _mirror_points(CANONICAL_LOWER_PROGRESS, width),
        }
    return {
        "split_left": CANONICAL_SPLIT_LEFT,
        "split_right": CANONICAL_SPLIT_RIGHT,
        "merge_zone": CANONICAL_MERGE_ZONE,
        "lower_progress": CANONICAL_LOWER_PROGRESS,
    }


def _diffusion_plan(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]], bool, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources = _water_positions(g)
    remaining = len(sources)

    seen = np.zeros((h, w), dtype=bool)
    frontier = sorted(sources)
    for r, c in frontier:
        seen[r, c] = True

    filled: list[tuple[int, int]] = []
    layers: list[list[tuple[int, int]]] = []
    partial_last_layer = False

    while frontier and remaining > 0:
        layer: set[tuple[int, int]] = set()
        for r, c in frontier:
            for rr, cc in _neighbors4(r, c, h, w):
                if seen[rr, cc]:
                    continue
                if int(g[rr, cc]) != int(BG):
                    continue
                seen[rr, cc] = True
                layer.add((rr, cc))

        if not layer:
            break

        ordered = sorted(layer)
        if remaining >= len(ordered):
            chosen = ordered
            frontier = ordered
            remaining -= len(ordered)
        else:
            chosen = ordered[:remaining]
            frontier = []
            remaining = 0
            partial_last_layer = True

        layers.append(chosen)
        filled.extend(chosen)

    return filled, layers, partial_last_layer, remaining


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Redistribute a supported source of water by breadth-first diffusion through a suspended chamber.

    Input structure:
    - The board is a 10x10 grid with background 0 and water color 1.
    - Every other non-zero color is a rigid obstacle, including a side source pocket, a floating
      chamber, and suspended horizontal bars inside that chamber.
    - Water starts as a supported packed source block beside the chamber rather than as already
      settled basin water.

    Rule:
    - Treat each input water pixel as one unit of available water and as part of the BFS source
      frontier, then clear the source block from the board.
    - Expand a 4-neighbor wave through background cells only; every full BFS layer fills at once.
    - If the last required layer is only partly needed, take its cells in reading order so the result
      stays deterministic.

    Key insight: The answer is controlled by graph distance inside a floating obstacle maze, not by
    gravity. Suspended bars split the BFS front into multiple branches and those branches can merge
    again lower in the chamber, so upper nearby cells fill before deeper cells.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    assert (h, w) == GRID_SHAPE, f"expected {GRID_SHAPE}, got {(h, w)}"

    sources = _water_positions(g)
    if check_input:
        assert sources, "expected source water"
        assert not np.any((g[h - 1] != int(BG)) & (g[h - 1] != int(WATER))), (
            "suspended obstacles should not touch the bottom row"
        )
        for r, c in sources:
            assert r == h - 1 or int(g[r + 1, c]) != int(BG), "input water must be supported"
            left_bg = c > 0 and int(g[r, c - 1]) == int(BG)
            right_bg = c + 1 < w and int(g[r, c + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"
        assert any(
            int(g[rr, cc]) == int(BG)
            for r, c in sources
            for rr, cc in _neighbors4(r, c, h, w)
        ), "source water must touch background through at least one side"

    source_count = len(sources)
    out = g.copy()
    out[out == int(WATER)] = int(BG)

    filled, _layers, _partial_last_layer, remaining = _diffusion_plan(g)
    assert remaining == 0, "not enough reachable background cells for BFS diffusion"

    for r, c in filled:
        out[r, c] = int(WATER)

    if check_output:
        assert int(np.sum(out == int(WATER))) == source_count, "water count must be conserved"
        solid_mask = (g != int(BG)) & (g != int(WATER))
        assert np.array_equal(out[solid_mask], g[solid_mask]), "solid structure must stay unchanged"
        for r, c in np.argwhere(out == int(WATER)):
            assert int(g[int(r), int(c)]) == int(BG), "output water must occupy former background cells"

    return out


def _gravity_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            options: list[tuple[int, int]] = []
            if (
                rr + 1 < h
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                options.append((rr + 1, cc - 1))
            if (
                rr + 1 < h
                and cc + 1 < w
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                options.append((rr + 1, cc + 1))
            if options:
                if len(options) != 1:
                    raise AssertionError(f"ambiguous diagonal moves: {options}")
                rr, cc = options[0]
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < w and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                if len(nearest) != 1:
                    raise AssertionError(f"ambiguous nearest drop columns: {nearest}")
                best = nearest[0]
                step = -1 if best < cc else 1
                nx = int(cc + step)
                if int(g[rr, nx]) != int(BG):
                    raise AssertionError("horizontal slide blocked")
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")

        return int(rr), int(cc)

    for _pass in range(h * w):
        moved = False
        pts = sorted(_water_positions(g), reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = int(WATER)
        if not moved:
            return g

    raise AssertionError("gravity baseline did not converge")


def _build_candidate(rng: random.Random) -> np.ndarray:
    pocket_color, chamber_color, bar_color = rng.sample(list(COLOR_POOL), 3)
    mirror = rng.choice((False, True))

    g = np.zeros(GRID_SHAPE, dtype=int)

    # Side source pocket with one outlet into the chamber.
    g[SOURCE_ROWS, SOURCE_COLS] = int(WATER)
    g[0, 3] = int(pocket_color)
    g[4, 0:4] = int(pocket_color)

    # Suspended outer chamber.
    g[1, 3:9] = int(chamber_color)
    g[8, 3:9] = int(chamber_color)
    g[1:9, 8] = int(chamber_color)
    g[1:9, 3] = int(chamber_color)
    g[2, 3] = int(BG)

    # Internal floating bars that create one clear split and a later reconnection.
    g[3, 5:7] = int(bar_color)
    g[5, 4:6] = int(bar_color)
    g[7, 5:7] = int(bar_color)

    if mirror:
        g = np.fliplr(g).copy()

    assert int(np.count_nonzero(g == int(WATER))) == int(SOURCE_WATER_COUNT)
    return g


def _count_unsupported(out: np.ndarray) -> int:
    h, _w = out.shape
    total = 0
    for r, c in _water_positions(out):
        if r + 1 < h and int(out[r + 1, c]) == int(BG):
            total += 1
    return total


def _zone_fill(out: np.ndarray, points: tuple[tuple[int, int], ...]) -> int:
    return int(sum(int(out[r, c]) == int(WATER) for r, c in points))


def _behavior_metrics(grid: np.ndarray) -> dict[str, int | float | bool]:
    new_out = main(grid)
    old_out = _gravity_baseline(grid)
    _filled, layers, partial_last_layer, remaining = _diffusion_plan(grid)
    assert remaining == 0

    features = _feature_points(grid)
    new_rows = [r for r, _c in _water_positions(new_out)]
    old_rows = [r for r, _c in _water_positions(old_out)]

    return {
        "delta_vs_gravity": int(np.sum(new_out != old_out)),
        "unsupported_water": int(_count_unsupported(new_out)),
        "height_gap": float(np.mean(old_rows) - np.mean(new_rows)),
        "layer_count": int(len(layers)),
        "partial_last_layer": bool(partial_last_layer),
        "split_left": int(_zone_fill(new_out, features["split_left"])),
        "split_right": int(_zone_fill(new_out, features["split_right"])),
        "merge_fill": int(_zone_fill(new_out, features["merge_zone"])),
        "lower_progress": int(_zone_fill(new_out, features["lower_progress"])),
        "bottom_solid_count": int(np.count_nonzero((grid[-1] != int(BG)) & (grid[-1] != int(WATER)))),
    }


def generate_example(
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | bool]]:
    if rng is None:
        rng = random.Random()

    candidate = _build_candidate(rng)
    output = main(candidate)
    metadata = _behavior_metrics(candidate)
    return candidate, output, metadata


def generate_input() -> np.ndarray:
    rng = random.Random()
    last_error: Exception | None = None

    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate(rng)
        try:
            metrics = _behavior_metrics(candidate)
        except Exception as exc:
            last_error = exc
            continue

        if metrics["delta_vs_gravity"] < 14:
            continue
        if metrics["unsupported_water"] < 2:
            continue
        if metrics["height_gap"] < 1.25:
            continue
        if metrics["layer_count"] < 7:
            continue
        if metrics["partial_last_layer"]:
            continue
        if metrics["split_left"] < 2 or metrics["split_right"] < 2:
            continue
        if metrics["merge_fill"] < 2:
            continue
        if metrics["lower_progress"] < 1:
            continue
        if metrics["bottom_solid_count"] != 0:
            continue

        return candidate

    if last_error is not None:
        raise RuntimeError(f"failed to generate fused suspended-BFS sample: {last_error}")
    raise RuntimeError("failed to generate fused suspended-BFS sample")
