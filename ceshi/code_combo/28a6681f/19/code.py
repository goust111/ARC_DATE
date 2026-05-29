# Combo variant of 28a6681f (index 19)
# combo_spec: D8/1+D13/0
# Name: 大网格+多盆地

from __future__ import annotations

import random
from typing import Any

import numpy as np


# Design analysis
# a) Concrete realization and behavior trace
#    - Keep the original visual language: background 0, water 1, and rigid colored basin walls.
#      The fusion does not change the local water physics; it changes the scene scale and the lower
#      obstacle vocabulary.
#    - The board is now 15x15 or 20x20. The lower structure is a single adjacent basin chain:
#      four small basins on 15x15, or five small basins on 20x20.
#    - Each basin is only 1-2 columns wide, so none of them can absorb much water alone. The
#      source reservoir is a supported edge block on a shelf above one end of the chain.
#    - Concrete 15x15 trace: suppose the shelf is on the left, the chain widths are [1, 2, 1, 2],
#      and the four divider heights are 6/3/2/3/5. Under the original 10x10 family there would be
#      room for only 2-3 basins, so the solver would only need to track one or two overflows. Here,
#      the same local rule repeatedly fills the leftmost cup, then spills over the next lower divider,
#      then the next one, until all four cups hold water. The output therefore encodes a longer
#      cascade even though each individual step is still ordinary fall + slide.
# b) Complexity calibration
#    - The generator uses only a few vetted templates per board size. That keeps the family readable
#      while still repeating the new large-grid / multi-basin signal across examples.
#    - Basin widths are restricted to 1 or 2 columns and wall heights to a small set of 2-3 levels.
#      This makes the chain visibly "many small cups" instead of one wide container.
#    - Source fill heights are also restricted to a short list. Every accepted sample still reaches
#      the far end of the chain, but the final per-basin counts vary enough that the outputs do not
#      collapse to one identical picture.
# c) Implicit-assumption audit against the original solver
#    - The actual settle loop only depends on occupancy, so it naturally generalizes from 10x10 to
#      15x15/20x20. No change is needed in the fall / diagonal / horizontal mechanics.
#    - The original guide-parsing block scanned the top 7 rows for special diagonal components.
#      That was a quiet 10x10-era assumption and is unnecessary here, because this fused family uses
#      a simple supported shelf source instead of edge-entering guide objects. The block is removed.
#    - The original input-support assertion still matters: every input water pixel must be supported
#      from below. The generator therefore uses packed rectangles resting on a shelf or on other water.
#    - The horizontal-slide branch assumes a unique nearest drop. The templates keep exactly one
#      shelf opening and use asymmetric wall heights so a water pixel never sees symmetric choices.
#    - Output stability checks from the original task remain valid and are intentionally preserved.
# d) Placement legality
#    - Every basin chain is drawn from explicit wall columns and explicit heights, so the obstacles
#      are always in bounds and non-overlapping.
#    - The source shelf is placed well above the tallest basin wall, leaving empty space between the
#      source region and the lower structure. This prevents accidental merging of the two objects.
#    - The shelf opening is aligned with the first or last basin interior column, so the water has a
#      legal entry path into the chain without requiring any extra hidden rule.
# e) Visual observability
#    - The fusion is visible from raw pixels alone: the board is obviously larger than 10x10, and
#      the lower obstacle field obviously contains 4-5 repeated cups rather than 2-3 larger ones.
#    - Because accepted samples always send water to the far end of the chain, a human can see that
#      the puzzle is about longer cascade tracking, not just about a larger empty margin.
# f) Visual cleanliness
#    - The picture has two separated vertical bands: an upper shelf reservoir and a lower basin comb.
#      Water never overlaps the shelf in the output, so source and destination remain easy to parse.
#    - Only one extra solid role is added beyond the basin walls: the shelf itself. No markers,
#      hidden codes, or color-dependent behavior are introduced.
# g) Change inferability
#    - Colors are purely structural. They help separate neighboring walls and the shelf, but the rule
#      does not depend on color ids or any non-visual ordering.
#    - The repeated evidence is geometric: bigger canvas, more cups, longer spill chain, and final
#      water occupying the whole chain.
# h) Fusion metric used after implementation
#    - For generated samples we measure `(size, basin_count, active_basins, cascade_steps,
#      top_row_spread)`.
#    - `cascade_steps` is the length of the contiguous occupied basin prefix from the left or right
#      edge source, minus one. This fusion targets `cascade_steps == basin_count - 1`, meaning water
#      reached the far end of the chain and therefore crossed every intermediate basin handoff.
#    - `top_row_spread <= 1` keeps the final waterline approximately leveled across the whole chain,
#      which is the visible signature of repeated overflow through many small basins.
# i) New solver assumptions re-checked against the generator
#    - `main()` now assumes the task family uses a 15x15 or 20x20 square canvas. The generator only
#      emits those two sizes.
#    - `_parse_basin_chain()` assumes one continuous bottom base with 4-5 wall columns taller than
#      one cell. The generator constructs exactly that comb structure.
#    - The stability asserts assume all output water ends in the lower basin chain. The generator
#      rejects any candidate with residual upper water, so this is guaranteed before return.


BG = 0
WATER = 1
SOLID_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
SIZE_OPTIONS = (15, 20)
MAX_RETRIES = 400

TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "size": 15,
        "widths": (1, 1, 1, 2),
        "wall_heights": (6, 4, 2, 3, 7),
        "source_margin": 2,
        "floor_row": 7,
        "fill_heights": (6,),
    },
    {
        "size": 15,
        "widths": (1, 1, 2, 1),
        "wall_heights": (7, 4, 4, 2, 7),
        "source_margin": 4,
        "floor_row": 6,
        "fill_heights": (5,),
    },
    {
        "size": 20,
        "widths": (1, 1, 2, 2, 2),
        "wall_heights": (6, 3, 2, 3, 2, 7),
        "source_margin": 4,
        "floor_row": 8,
        "fill_heights": (6, 7, 8),
    },
    {
        "size": 20,
        "widths": (2, 1, 2, 1, 2),
        "wall_heights": (7, 2, 3, 2, 3, 6),
        "source_margin": 4,
        "floor_row": 8,
        "fill_heights": (6, 7, 8),
    },
)


def _stack_height(structure: np.ndarray, c: int) -> int:
    h = int(structure.shape[0])
    if int(structure[h - 1, c]) == BG:
        return 0
    height = 1
    r = h - 2
    while r >= 0 and int(structure[r, c]) != BG:
        height += 1
        r -= 1
    return int(height)


def _parse_basin_chain(grid: np.ndarray) -> dict[str, Any]:
    structure = np.where(np.asarray(grid, dtype=int) == int(WATER), int(BG), np.asarray(grid, dtype=int))
    h, w = structure.shape
    bottom_cols = [int(c) for c in range(w) if int(structure[h - 1, c]) != BG]
    assert bottom_cols, "expected a bottom-anchored basin chain"
    left = int(min(bottom_cols))
    right = int(max(bottom_cols))
    assert all(int(structure[h - 1, c]) != BG for c in range(left, right + 1)), "bottom base must be continuous"

    wall_cols = [int(c) for c in range(left, right + 1) if _stack_height(structure, int(c)) >= 2]
    basins = [(int(a) + 1, int(b)) for a, b in zip(wall_cols, wall_cols[1:]) if int(b) - int(a) >= 2]
    widths = [int(b - a) for a, b in basins]
    wall_heights = [_stack_height(structure, int(c)) for c in wall_cols]
    return {
        "left": int(left),
        "right": int(right),
        "wall_cols": wall_cols,
        "basins": basins,
        "widths": widths,
        "wall_heights": wall_heights,
    }


def _settle(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = g.shape
    rr = int(r)
    cc = int(c)
    for _ in range(h * w):
        if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
            rr += 1
            continue

        opts: list[tuple[int, int]] = []
        if (
            rr + 1 < h
            and cc - 1 >= 0
            and int(g[rr, cc - 1]) == int(BG)
            and int(g[rr + 1, cc - 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc - 1)))
        if (
            rr + 1 < h
            and cc + 1 < w
            and int(g[rr, cc + 1]) == int(BG)
            and int(g[rr + 1, cc + 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc + 1)))
        if opts:
            assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
            rr, cc = opts[0]
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
            assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
            best = nearest[0]
            assert best != int(cc)
            step = -1 if best < int(cc) else 1
            nx = int(cc + step)
            assert int(g[rr, nx]) == int(BG)
            cc = nx
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")

    return int(rr), int(cc)


def _simulate(grid: np.ndarray, *, check_input: bool, check_output: bool) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    assert h == w and h in SIZE_OPTIONS, "grid must be 15x15 or 20x20"

    if check_input:
        parsed = _parse_basin_chain(g)
        basin_count = int(len(parsed["basins"]))
        assert basin_count in (4, 5), f"expected 4-5 small basins, got {basin_count}"
        assert all(width in (1, 2) for width in parsed["widths"]), "basins must stay small"
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    for _ in range(h * w):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = _settle(g, int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == w - 1 or int(g[rr, cc + 1]) != int(BG)

    return g


def _occupied_counts(grid: np.ndarray, basins: list[tuple[int, int]]) -> list[int]:
    counts: list[int] = []
    for left, right in basins:
        counts.append(int(np.count_nonzero(grid[:, int(left) : int(right)] == int(WATER))))
    return counts


def _top_rows(grid: np.ndarray, basins: list[tuple[int, int]]) -> list[int]:
    rows: list[int] = []
    for left, right in basins:
        pts = np.argwhere(grid[:, int(left) : int(right)] == int(WATER))
        if len(pts) > 0:
            rows.append(int(pts[:, 0].min()))
    return rows


def _fusion_signature(grid: np.ndarray) -> dict[str, Any]:
    parsed = _parse_basin_chain(grid)
    basins = [(int(left), int(right)) for left, right in parsed["basins"]]
    counts = _occupied_counts(grid, basins)
    active = [count > 0 for count in counts]

    left_reach = 0
    for flag in active:
        if not flag:
            break
        left_reach += 1

    right_reach = 0
    for flag in reversed(active):
        if not flag:
            break
        right_reach += 1

    top_rows = _top_rows(grid, basins)
    return {
        "size": int(grid.shape[0]),
        "basin_count": int(len(basins)),
        "active_basins": int(sum(active)),
        "occupied_counts": counts,
        "cascade_steps": int(max(left_reach, right_reach) - 1),
        "top_rows": top_rows,
        "top_row_spread": int(max(top_rows) - min(top_rows)) if top_rows else 0,
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate gravity-driven water settling on a larger board that contains a longer chain of small basins.

    Input structure:
    - The canvas is a 15x15 or 20x20 square instead of the original 10x10 board.
    - Background is 0, water is 1, and every other color is a rigid wall or shelf.
    - The lower structure is one bottom-anchored chain of 4-5 narrow basins, while the input water starts
      as a supported edge reservoir on an upper shelf near the left or right side.

    Rule:
    - Process water with the same local motion law as the source task: fall straight down through empty cells,
      otherwise take the unique open diagonal, otherwise slide along a supported row toward the unique nearest drop.
    - Repeat global bottom-first passes until no water pixel moves.
    - Walls and shelf stay fixed; only the water redistributes, often through several consecutive basin overflows.

    Key insight: The physics is unchanged, but the larger canvas packs in 4-5 small cups, so the hard part is
    tracking a longer cascade chain as one reservoir successively fills and spills across many neighboring basins.
    """
    return _simulate(
        grid,
        check_input=check_input,
        check_output=check_output,
    )


def _draw_structure(
    size: int,
    widths: tuple[int, ...],
    wall_heights: tuple[int, ...],
    source_margin: int,
    floor_row: int,
    source_side: str,
    colors: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    basin_colors = colors[:3]
    shelf_color = int(colors[3])

    span = int(sum(widths) + len(widths) + 1)
    if source_side == "left":
        left_margin = int(source_margin)
    else:
        left_margin = int(size - span - source_margin)
    wall_cols = [int(left_margin)]
    for width in widths:
        wall_cols.append(int(wall_cols[-1] + int(width) + 1))

    g = np.zeros((size, size), dtype=int)
    for idx in range(len(wall_cols) - 1):
        color = int(basin_colors[idx % len(basin_colors)])
        g[size - 1, int(wall_cols[idx]) : int(wall_cols[idx + 1]) + 1] = int(color)
    for idx, (col, height) in enumerate(zip(wall_cols, wall_heights)):
        color = int(basin_colors[idx % len(basin_colors)])
        g[size - int(height) :, int(col)] = int(color)

    if source_side == "left":
        drop_col = int(wall_cols[0] + 1)
        shelf_end = min(int(size - 2), int(drop_col + 3))
        for c in range(0, int(shelf_end) + 1):
            if c != int(drop_col):
                g[int(floor_row), int(c)] = int(shelf_color)
        source_cols = (0, int(drop_col))
    else:
        drop_col = int(wall_cols[-1] - 1)
        shelf_start = max(1, int(drop_col - 3))
        for c in range(int(shelf_start), int(size)):
            if c != int(drop_col):
                g[int(floor_row), int(c)] = int(shelf_color)
        source_cols = (int(drop_col + 1), int(size))

    return g, {
        "wall_cols": wall_cols,
        "drop_col": int(drop_col),
        "source_cols": source_cols,
        "floor_row": int(floor_row),
    }


def _add_source_water(grid: np.ndarray, fill_height: int, source_cols: tuple[int, int], floor_row: int) -> None:
    left, right = source_cols
    g = grid
    g[int(floor_row - fill_height) : int(floor_row), int(left) : int(right)] = int(WATER)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        tpl = dict(rng.choice(TEMPLATES))
        source_side = str(rng.choice(("left", "right")))
        fill_height = int(rng.choice(tuple(int(x) for x in tpl["fill_heights"])))
        colors = tuple(int(c) for c in rng.sample(SOLID_POOL, 4))

        base, meta = _draw_structure(
            size=int(tpl["size"]),
            widths=tuple(int(x) for x in tpl["widths"]),
            wall_heights=tuple(int(x) for x in tpl["wall_heights"]),
            source_margin=int(tpl["source_margin"]),
            floor_row=int(tpl["floor_row"]),
            source_side=source_side,
            colors=colors,
        )
        inp = np.array(base, dtype=int).copy()
        _add_source_water(
            inp,
            fill_height=int(fill_height),
            source_cols=tuple(int(x) for x in meta["source_cols"]),
            floor_row=int(meta["floor_row"]),
        )

        try:
            out = main(inp, check_input=True, check_output=True)
        except AssertionError:
            continue

        sig = _fusion_signature(out)
        expected_basins = int(len(tpl["widths"]))
        if sig["basin_count"] != expected_basins:
            continue
        if sig["active_basins"] != expected_basins:
            continue
        if sig["cascade_steps"] != expected_basins - 1:
            continue
        if sig["top_row_spread"] > 1:
            continue
        if np.any(out[: int(meta["floor_row"]), :] == int(WATER)):
            continue
        if np.array_equal(inp, out):
            continue

        return inp

    raise RuntimeError("failed to generate a valid larger-grid multi-basin cascade sample")
