# Combo variant of 28a6681f (index 39)
# combo_spec: D11/0+D16/1
# Name: 裁剪到盆地区域+更大网格
#
# Design analysis
# ---------------
# a) Concrete fused behavior and traced example:
#    - Source task: a 10x10 board shows both the unstable source water and the bottom basin system,
#      and the output keeps that full frame after water settles.
#    - Fused task: the simulation still happens on the full board, but the board is now a larger
#      15x15 or 20x20 square. After settling, the output is cropped to the minimal rectangle of the
#      bottom-anchored basin structure. That crop keeps the basin walls and the water inside them,
#      while discarding the empty headroom and the upper side source shelf.
#    - Concrete 15x15 trace:
#      * columns 3, 7, 11 form three basin walls with heights 6, 4, 6, so the crop later spans
#        only columns 3..11 and rows 9..14;
#      * a supported 3x4 water block sits on a shelf near the top-left, with one leak at column 4;
#      * under the original family, the full 15x15 settled scene would still show the upper shelf and
#        large empty margins;
#      * under the fused family, the same water must still be simulated on the full board first, but
#        the visible answer is just the compact lower basin rectangle.
# b) Complexity calibration:
#    - The geometry stays in one clear subfamily: one side-fed upper shelf plus 2-3 linked lower
#      basins. That keeps the water law recognizable.
#    - Freedom is limited to a few repeated axes: board size (15 or 20 only), left/right source side,
#      basin count, basin widths, wall heights, and source water volume. This is enough variation to
#      make the larger-grid and crop effects visible without making the family visually messy.
#    - Accepted samples always shrink after cropping and always use a 15x15 or 20x20 full board, so
#      both fused dimensions are active on every instance rather than appearing accidentally.
# c) Hidden assumption review line-by-line against the original solver:
#    - The original upper-guide validation scanned `g[:7]` and inferred guide identity from colors
#      absent on the bottom row. That quietly assumes a 10-row board and a special upper color split.
#      Both assumptions break here: larger boards can be 20 rows tall, and source shelf colors may
#      also appear in the basin. The fused solver therefore removes that parsing block entirely.
#    - The actual settle loop only depends on occupancy (`bg` vs water vs solid), so its straight
#      fall, unique diagonal, and nearest-drop horizontal slide remain valid on larger boards.
#    - New assumption introduced by cropping: every final water pixel must lie inside the basin crop,
#      otherwise cropping would silently discard water. The generator guarantees this by placing all
#      source water on a one-leak shelf that fully drains into the lower basin system.
# d) Placement legality:
#    - Basin walls are generated from explicit wall columns and heights, then drawn inside the board
#      with positive outer margins, so they neither overlap illegally nor leave the canvas.
#    - The source shelf is placed strictly above the tallest wall with at least one blank row of
#      separation, so the source structure never merges with the bottom-anchored basin component.
#    - Source water is always a side-packed rectangular block resting on that shelf, so every input
#      water pixel is supported and no water pixel starts with background on both horizontal sides.
# e) Visual observability:
#    - A human solver can see the fusion from 2-4 examples because two cues repeat directly:
#      1. the input board is much larger than 10x10;
#      2. the output is a compact crop showing only the lower basin rectangle.
#    - The crop is geometric, not based on hidden color order or any non-visual metadata. Black/0 is
#      only visible empty space.
# f) Visual cleanliness:
#    - The picture is split into two bands: source shelf high above, basin comb low below. That keeps
#      the cropped output compact and readable instead of mixing symbols in the same area.
#    - Basin walls stay rectilinear and adjacent, so even overflow cases look like one coherent basin
#      system rather than scattered obstacles.
# g) Inferability of the change:
#    - No new color mapping is introduced. The only new rule beyond the original motion law is that
#      one must identify the basin region after simulating on the large board and return only that
#      region.
#    - Because 15x15 and 20x20 examples both recur, and every accepted sample has strict crop margins,
#      the size increase and the basin-only crop are both learnable from a few demonstrations.
# 9) New solver assumptions audited against the generator:
#    - Assumption: there is at least one bottom-anchored solid component to define the crop. Guarantee:
#      `generate_input()` always draws a bottom floor plus outer walls.
#    - Assumption: water never remains outside the basin bbox after settling. Guarantee: the source is
#      a one-gap shelf, and candidates with source-side residual water are rejected.
#    - Assumption: diagonal and nearest-drop choices stay unique. Guarantee: the comb basin and single
#      shelf leak are simple enough that all candidates are validated by re-running `main()`.

from __future__ import annotations

import random

import numpy as np


BG = 0
WATER = 1
SOLID_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
SIZE_OPTIONS = (15, 20)
MAX_RETRIES = 600


def _solid_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _bottom_anchored_mask(grid: np.ndarray) -> np.ndarray:
    solids = (grid != int(BG)) & (grid != int(WATER))
    mask = np.zeros_like(solids, dtype=bool)
    for comp in _solid_components(solids):
        if any(r == grid.shape[0] - 1 for r, _c in comp):
            for r, c in comp:
                mask[r, c] = True
    return mask


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size, "mask must be non-empty"
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return r0, c0, r1, c1


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


def _simulate_water(grid: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int).copy()
    moved_total = 0
    h, w = out.shape
    for _ in range(h * w):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(out == int(WATER))], reverse=True)
        for r, c in pts:
            if int(out[r, c]) != int(WATER):
                continue
            out[r, c] = int(BG)
            rr, cc = _settle(out, int(r), int(c))
            moved = bool(moved or ((rr, cc) != (int(r), int(c))))
            moved_total += int((rr, cc) != (int(r), int(c)))
            out[rr, cc] = int(WATER)
        if not moved:
            return out, int(moved_total)
    raise AssertionError("water simulation did not converge")


def _occupied_basin_count(full: np.ndarray, wall_cols: list[int]) -> int:
    occupied = 0
    for left, right in zip(wall_cols, wall_cols[1:]):
        if np.any(full[:, int(left) + 1 : int(right)] == int(WATER)):
            occupied += 1
    return int(occupied)


def _source_water_remaining(full: np.ndarray, floor_row: int, source_side: str) -> int:
    if source_side == "left":
        return int(np.count_nonzero(full[: floor_row + 1, : full.shape[1] // 2] == int(WATER)))
    return int(np.count_nonzero(full[: floor_row + 1, full.shape[1] // 2 :] == int(WATER)))


def _simulate_full(
    grid: np.ndarray,
    *,
    crop_output: bool,
    check_input: bool,
    check_output: bool,
    return_info: bool = False,
):
    base = np.array(grid, dtype=int).copy()
    h, w = base.shape
    assert h == w and h in SIZE_OPTIONS, "grid must be 15x15 or 20x20"

    basin_mask = _bottom_anchored_mask(base)
    basin_bbox = _bbox_from_mask(basin_mask)

    if check_input:
        assert np.any(base == int(WATER)), "input must contain water"
        assert bool(basin_mask.any()), "input must contain a bottom-anchored basin structure"
        for r, c in np.argwhere(base == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(base[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(base[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < w and int(base[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"
        assert basin_bbox[0] >= 2 and basin_bbox[1] >= 1 and basin_bbox[3] <= w - 2, "crop margins must be strict"

    settled, moved_total = _simulate_water(base)

    if check_output:
        for r, c in np.argwhere(settled == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(settled[rr, cc - 1]) != int(BG)
            assert rr == h - 1 or int(settled[rr + 1, cc]) != int(BG)
            assert cc == w - 1 or int(settled[rr, cc + 1]) != int(BG)
            assert basin_bbox[0] <= rr <= basin_bbox[2] and basin_bbox[1] <= cc <= basin_bbox[3], "cropping must not discard water"

    out = settled[basin_bbox[0] : basin_bbox[2] + 1, basin_bbox[1] : basin_bbox[3] + 1].copy() if crop_output else settled

    info = {
        "full_shape": (int(h), int(w)),
        "crop_bbox": tuple(int(v) for v in basin_bbox),
        "cropped_shape": tuple(int(v) for v in out.shape),
        "crop_removed_rows": int(h - out.shape[0]),
        "crop_removed_cols": int(w - out.shape[1]),
        "crop_area_ratio": float((out.shape[0] * out.shape[1]) / float(h * w)),
        "water_count": int(np.count_nonzero(settled == int(WATER))),
        "moved_water_steps": int(moved_total),
    }
    if return_info:
        return out, settled, info
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate the original gravity-driven water behavior on a larger square board, then crop to the basin region only.

    Input structure:
    - The canvas is 15x15 or 20x20 instead of 10x10.
    - Background is 0, water is color 1, and every other non-zero pixel is rigid structure.
    - A supported side source shelf or run appears in the upper area, while the actual linked basin
      system is the bottom-anchored structure near the lower part of the board.

    Rule:
    - Water first follows the same local motion law as the source task: fall straight down through
      empty cells, otherwise take the unique open downward diagonal, otherwise slide along a supported
      row toward the unique nearest drop column.
    - Repeat that per-pixel settling until no water moves.
    - After settling on the full large board, locate the minimal bounding rectangle of the
      bottom-anchored basin structure and return only that rectangle, keeping both basin walls and the
      water now resting inside them.

    Key insight:
    The hard part is no longer just simulating the liquid. One must simulate on the full 15x15/20x20
    scene first, then recognize that the answer discards the source region and all surrounding empty
    space, preserving only the compact basin area.
    """
    return _simulate_full(
        grid,
        crop_output=True,
        check_input=check_input,
        check_output=check_output,
    )


def _distribute(total: int, parts: int, min_each: int, rng: random.Random) -> list[int]:
    vals = [int(min_each)] * int(parts)
    rem = int(total - min_each * parts)
    assert rem >= 0
    for _ in range(rem):
        vals[rng.randrange(parts)] += 1
    rng.shuffle(vals)
    return vals


def _build_basin_geometry(
    n: int,
    source_side: str,
    target_overflow: bool,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int]]:
    margin_lo = 2 if n == 15 else 3
    margin_hi = 4 if n == 15 else 5
    left_margin = rng.randint(margin_lo, margin_hi)
    right_margin = rng.randint(margin_lo, margin_hi)
    span = n - left_margin - right_margin

    max_basins = 3 if span >= 10 else 2
    n_basins = rng.randint(2, max_basins)
    total_interior = span - (n_basins + 1)
    widths = _distribute(total_interior, n_basins, 2, rng)

    wall_cols = [int(left_margin)]
    for width in widths:
        wall_cols.append(int(wall_cols[-1] + width + 1))

    outer_base = rng.randint(5, 7) if n == 15 else rng.randint(7, 10)
    heights = [0] * (n_basins + 1)
    heights[0] = outer_base + rng.randint(0, 1)
    heights[-1] = outer_base + rng.randint(0, 1)
    for i in range(1, n_basins):
        heights[i] = rng.randint(max(3, outer_base - 3), outer_base)

    if source_side == "left":
        critical = 1
        other = max(1, n_basins - 1)
    else:
        critical = n_basins - 1
        other = min(n_basins - 1, 1)

    if target_overflow:
        heights[critical] = rng.randint(max(3, outer_base - 3), max(3, outer_base - 1))
        if n_basins >= 3 and other != critical:
            heights[other] = rng.randint(max(3, outer_base - 2), outer_base)
    else:
        heights[critical] = outer_base + rng.randint(0, 1)

    return wall_cols, heights, widths


def _draw_basin(
    n: int,
    wall_cols: list[int],
    heights: list[int],
    basin_colors: list[int],
) -> np.ndarray:
    g = np.zeros((n, n), dtype=int)
    for i in range(len(wall_cols) - 1):
        color = int(basin_colors[i % len(basin_colors)])
        g[n - 1, wall_cols[i] : wall_cols[i + 1] + 1] = color
    for i, (col, height) in enumerate(zip(wall_cols, heights)):
        color = int(basin_colors[i % len(basin_colors)])
        g[n - int(height) : n, int(col)] = color
    return g


def _place_source(
    g: np.ndarray,
    wall_cols: list[int],
    heights: list[int],
    source_color: int,
    source_side: str,
    target_overflow: bool,
    rng: random.Random,
) -> dict[str, int]:
    n = g.shape[0]
    max_height = max(int(h) for h in heights)
    floor_row_max = n - max_height - 3
    assert floor_row_max >= 3
    floor_row = rng.randint(3, floor_row_max)

    if source_side == "left":
        drop_col = int(wall_cols[0] + 1)
        shelf_right = min(n - 2, drop_col + rng.randint(3, 5 if n == 20 else 4))
        for c in range(0, shelf_right + 1):
            if c != drop_col:
                g[floor_row, int(c)] = int(source_color)
        max_width = min(6 if n == 20 else 5, drop_col)
        block_width = max_width if target_overflow else rng.randint(2, max_width)
        block_height = rng.randint(2, min(4, floor_row))
        g[floor_row - block_height : floor_row, 0:block_width] = int(WATER)
    else:
        drop_col = int(wall_cols[-1] - 1)
        shelf_left = max(1, drop_col - rng.randint(3, 5 if n == 20 else 4))
        for c in range(shelf_left, n):
            if c != drop_col:
                g[floor_row, int(c)] = int(source_color)
        max_width = min(6 if n == 20 else 5, n - 1 - drop_col)
        block_width = max_width if target_overflow else rng.randint(2, max_width)
        block_height = rng.randint(2, min(4, floor_row))
        g[floor_row - block_height : floor_row, n - block_width : n] = int(WATER)

    return {
        "floor_row": int(floor_row),
        "drop_col": int(drop_col),
        "water_count": int(np.count_nonzero(g == int(WATER))),
    }


def _build_candidate(rng: random.Random, target_overflow: bool) -> tuple[np.ndarray, dict[str, object]]:
    n = int(rng.choice(SIZE_OPTIONS))
    source_side = "left" if rng.random() < 0.5 else "right"
    wall_cols, heights, widths = _build_basin_geometry(n, source_side, target_overflow, rng)

    basin_color_count = 2 if rng.random() < 0.55 else 3
    palette = rng.sample(list(SOLID_POOL), basin_color_count + 1)
    source_color = int(palette[0])
    basin_colors = [int(x) for x in palette[1:]]

    g = _draw_basin(n, wall_cols, heights, basin_colors)
    source_meta = _place_source(g, wall_cols, heights, source_color, source_side, target_overflow, rng)
    meta: dict[str, object] = {
        "size": int(n),
        "source_side": source_side,
        "wall_cols": wall_cols,
        "heights": heights,
        "widths": widths,
        "source_color": int(source_color),
        "basin_colors": basin_colors,
        **source_meta,
    }
    return g, meta


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        target_overflow = rng.random() < 0.6
        grid, meta = _build_candidate(rng, target_overflow)
        try:
            cropped, full, info = _simulate_full(
                grid,
                crop_output=True,
                check_input=True,
                check_output=True,
                return_info=True,
            )
        except AssertionError:
            continue

        if np.array_equal(grid, full):
            continue

        basin_count = _occupied_basin_count(full, list(meta["wall_cols"]))
        source_residual = _source_water_remaining(full, int(meta["floor_row"]), str(meta["source_side"]))
        crop_is_strict = bool(cropped.shape[0] < grid.shape[0] and cropped.shape[1] < grid.shape[1])
        moved_far_enough = info["moved_water_steps"] >= int(meta["water_count"])
        crop_visible = info["crop_area_ratio"] <= 0.65
        bottom_component_top = int(info["crop_bbox"][0])
        no_top_water = not np.any(full[:bottom_component_top, :] == int(WATER))

        if basin_count < 1:
            continue
        if target_overflow and basin_count < 2:
            continue
        if source_residual != 0 or not no_top_water:
            continue
        if not crop_is_strict or not crop_visible:
            continue
        if not moved_far_enough:
            continue

        original_style = _simulate_full(
            grid,
            crop_output=False,
            check_input=False,
            check_output=False,
        )
        if original_style.shape == cropped.shape:
            continue

        return grid

    raise RuntimeError("generate_input failed to produce a valid D11/0 + D16/1 sample")
