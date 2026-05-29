# Variant of 28a6681f
# Dimension: D16 — 盆壁颜色的功能意义 (obstacle)
# Change: 使用2-3种非水、非背景颜色构成盆壁结构，颜色仅用于视觉区分，不影响水流行为... → 不同颜色壁面具有不同摩擦力

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) Concrete form and behavior trace:
#    - The basin still lives in a 10x10 grid with bg=0 and water=1.
#    - Color 2 is a rough obstacle: if a water pixel is resting on top of it, any
#      horizontal spill attempt is cancelled immediately.
#    - Color 3 is a smooth obstacle: on top of it the old horizontal search for the
#      nearest drop is still allowed. Color 4 is a neutral guide shelf used only to
#      hold the source water; it behaves like a smooth solid.
#    - Each generated sample shows an edge-packed source block on rows 5-6 and a
#      bottom-anchored terrace system on rows 7-9. The source spans both guide cells
#      and rough cells, so one part of the block can keep flowing while another part
#      visibly sticks.
#    - Trace example: with the old rule, a pixel above cols 2-4 would slide across the
#      rough launch strip, pass the gap, and continue toward the smooth basin. Under
#      D16 the same pixel stops as soon as its current support is color 2. Only pixels
#      still resting on color 4 or color 3 keep sliding into the lower basin.
# b) Complexity calibration:
#    - The friction signal is binary and repeated: every sample contains multiple rough
#      support cells and multiple smooth support cells. The rule is visible without
#      needing many colors or many special cases.
#    - Left and right mirrored layouts are both used so the task does not collapse into
#      "always move right" or "always move left".
# c) Hidden assumptions from the original solver:
#    - Original horizontal motion assumed every supported pixel may keep searching for a
#      drop. That is no longer true: support color now matters, so the horizontal phase
#      must branch on the support tile.
#    - Original output validation assumed every final water pixel has no background on
#      either horizontal side. That was only harmless because old water never stopped on
#      a ledge with a side opening. In D16 that assumption is false, so output checking
#      must be replaced with a local re-simulation stability test.
# d) Placement legality:
#    - All templates are fully inside the 10x10 frame.
#    - Source water is always a contiguous edge-packed block; every source pixel is
#      vertically supported by water or a solid guide/terrace tile.
#    - The lower structures are bottom-anchored and use a single visible gap so the
#      nearest-drop rule stays unambiguous.
# e) Visual observability:
#    - Every accepted sample must exhibit both a rough block event and at least one
#      smooth horizontal slide event. The generator rejects candidates that do not use
#      both branches.
#    - The generator also compares the D16 output against the friction-free output and
#      rejects candidates unless they differ by several cells. That prevents degeneration
#      back into the original task.
# f) Visual cleanliness:
#    - The source occupies the upper-left or upper-right terrace only; the receiving
#      basin occupies the lower rows. Colors are not overlaid, and each region has a
#      single purpose.
# g) Color-role inferability:
#    - The mapping is fixed and visible: color 2 is always rough, color 3 is always
#      smooth, and color 4 is only a guide shelf. No invisible ordering of color ids is
#      used.
# 9) New solver assumptions and generator guarantees:
#    - Rough stopping is only defined for water that is already vertically supported.
#      The generator therefore places rough tiles under active launch ledges, not only in
#      dead-end pits.
#    - Smooth sliding still relies on a unique reachable drop. The templates reserve one
#      clear gap between the rough launch strip and the smooth receiving strip.
#    - Because local stability now depends on support color, output validation replays
#      each water pixel in isolation. The generator only keeps templates that satisfy
#      this stronger check.

BG = 0
WATER = 1
ROUGH = 2
SMOOTH = 3
GUIDE = 4

_TEMPLATE_SPECS = (
    {
        "bottom_rows": ("4422200000", "0022200300", "0022203330"),
        "source_widths": (5, 5),
    },
    {
        "bottom_rows": ("4422200000", "0022203000", "0022203330"),
        "source_widths": (4, 5),
    },
    {
        "bottom_rows": ("4422200000", "0022200000", "0022203330"),
        "source_widths": (5, 5),
    },
    {
        "bottom_rows": ("4422200000", "0022203030", "0022203330"),
        "source_widths": (4, 4),
    },
    {
        "bottom_rows": ("4422000300", "0022200300", "0022230330"),
        "source_widths": (4, 4),
    },
)


def _grid_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _build_candidate(spec: dict[str, object], mirror: bool) -> np.ndarray:
    rows = ["0000000000"] * 7 + list(spec["bottom_rows"])
    g = _grid_from_rows(rows)
    top_width, bottom_width = spec["source_widths"]
    if bottom_width:
        g[6, : int(bottom_width)] = WATER
    if top_width:
        g[5, : int(top_width)] = WATER
    if mirror:
        g = np.fliplr(g).copy()
    return g


def _simulate(
    grid: np.ndarray,
    *,
    friction_enabled: bool,
    check_input: bool,
    check_output: bool,
    collect_metrics: bool = False,
) -> tuple[np.ndarray, dict[str, int]] | np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    metrics = {"rough_blocks": 0, "smooth_slides": 0} if collect_metrics else None

    if check_input:
        assert np.any(g == ROUGH), "input must contain rough obstacles"
        assert np.any(g == SMOOTH), "input must contain smooth obstacles"
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    def horizontal_run(rr: int, cc: int) -> tuple[int, int, list[int]]:
        left = int(cc)
        while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
            left -= 1
        right = int(cc)
        while right + 1 < W and int(g[rr, right + 1]) == int(BG):
            right += 1
        drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
        return int(left), int(right), drops

    def settle(r: int, c: int, *, record_metrics: bool = True) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                _left, _right, drops = horizontal_run(int(rr), int(cc))
                support_color = int(g[rr + 1, cc])
                if friction_enabled and support_color == int(ROUGH):
                    if record_metrics and metrics is not None and drops:
                        metrics["rough_blocks"] += 1
                    break
                if not drops:
                    break
                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = int(nearest[0])
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                if record_metrics and metrics is not None:
                    metrics["smooth_slides"] += 1
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c), record_metrics=True)
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
            g[rr, cc] = int(BG)
            sr, sc = settle(int(rr), int(cc), record_metrics=False)
            g[rr, cc] = int(WATER)
            assert (int(sr), int(sc)) == (int(rr), int(cc)), "output water is not locally stable"

    if metrics is None:
        return g
    return g, metrics


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    out_variant, metrics = _simulate(
        grid,
        friction_enabled=True,
        check_input=True,
        check_output=True,
        collect_metrics=True,
    )
    out_original = _simulate(
        grid,
        friction_enabled=False,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    result = dict(metrics)
    result["diff_cells"] = int(np.count_nonzero(out_variant != out_original))
    result["top_water"] = int(np.count_nonzero(out_variant[:7] == WATER))
    result["bottom_water"] = int(np.count_nonzero(out_variant[7:] == WATER))
    return result


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate gravity-driven water settling in bottom-anchored basins whose colors now encode slide resistance.

    Input structure:
    - Background is 0 and water is color 1.
    - Color 2 marks rough basin tiles, color 3 marks smooth basin tiles, and color 4 may appear as a guide shelf.
    - Water begins as a supported edge-packed source above adjacent terraces and lower receiving basins.

    Rule:
    - Water still falls straight down through background cells and may descend diagonally through the unique open shoulder.
    - When a water pixel is resting on a smooth support tile, it keeps searching horizontally for the nearest reachable drop.
    - When it is resting on a rough support tile, that horizontal spill is cancelled and the pixel stays there until later passes move other water around it.
    - Repeat full settling passes until no water pixel changes position.

    Key insight: Basin geometry alone is no longer enough; the support color decides which ledges behave like spillways and which ledges pin the water in place.
    """
    return _simulate(
        grid,
        friction_enabled=True,
        check_input=check_input,
        check_output=check_output,
        collect_metrics=False,
    )


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = random.Random() if rng is None else rng
    for _ in range(64):
        spec = rng.choice(_TEMPLATE_SPECS)
        mirror = bool(rng.randrange(2))
        candidate = _build_candidate(spec, mirror)
        try:
            stats = variant_metrics(candidate)
        except AssertionError:
            continue
        if stats["rough_blocks"] < 1:
            continue
        if stats["smooth_slides"] < 1:
            continue
        if stats["diff_cells"] < 4:
            continue
        if stats["top_water"] < 3:
            continue
        if stats["bottom_water"] < 3:
            continue
        return candidate
    raise AssertionError("failed to generate a non-degenerate D16 sample")
