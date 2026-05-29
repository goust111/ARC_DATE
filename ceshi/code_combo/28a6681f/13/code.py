# Combo variant of 28a6681f (index 13)
# combo_spec: D12/0+D16/0
# Name: 水蒸发+壁面摩擦力

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ---------------
# a) Concrete fused behavior and traced example:
#    - The board remains 10x10 with background 0 and water 1, but basin-wall colors are now
#      functional: color 2 is a rough high-friction support and color 3 is a smooth low-friction
#      support.
#    - The generated family uses an upper supported source slab on rows 5-6, a mixed rough/smooth
#      launch strip on row 7, and a small carried-over floor puddle on row 9 so the lower basin
#      stays visually active.
#    - Example trace on the unmirrored template `2222233000 / 0000200003 / 2222200003`:
#      * under the original task, source water above both colors would keep sliding toward the right
#        drop and would eventually survive as redistributed basin water;
#      * under D16 alone, water above rough color 2 would stop instead of sliding, while water above
#        smooth color 3 would still spill rightward;
#      * under the fused rule here, that smooth spill is no longer conserved: the instant a droplet
#        would take a diagonal or horizontal step, that one-pixel droplet evaporates and disappears;
#      * therefore rough-supported droplets survive on the launch strip, smooth-supported droplets
#        vanish, and only the pre-existing bottom puddle remains in the lower basin.
#    - The visible difference is not just "less water": the surviving upper water aligns with rough
#      supports, showing that friction decides where evaporation can even happen.
# b) Complexity calibration:
#    - The friction vocabulary is intentionally binary and repeated across every sample: color 2
#      always means rough, color 3 always means smooth. No hidden color ordering is used.
#    - The geometry stays in one clean subfamily with a left/right mirrored option and a handful of
#      closely related terrace templates. This keeps the puzzle learnable from 2-4 examples while
#      still showing several repeated rough and smooth support cells in each instance.
#    - A carried-over floor puddle is kept in every sample so the scene still reads as a basin task
#      instead of collapsing into "water just disappears in the sky".
# c) Hidden-assumption audit of the original main():
#    - Original `settle()` assumed every non-vertical move relocates the same water pixel. That
#      assumption breaks here: a diagonal or horizontal branch now destroys the moving droplet, so
#      `settle()` must be able to return "evaporated" instead of a destination cell.
#    - Original horizontal motion ignored support color. That is false now because rough supports
#      cancel horizontal spill while smooth supports allow it.
#    - Original outer simulation loop implicitly conserved water count because every removed input
#      pixel was always written back somewhere. That is false here; evaporated droplets are simply
#      not restored.
#    - Original output validation checked for no background on left/below/right, but rough stopping
#      makes side-open ledges legal final states. Local replay stability is the correct invariant.
# d) Placement legality:
#    - Every template is fully inside the 10x10 canvas and uses only bottom-anchored solids.
#    - Source water occupies only rows 5-6 and never overlaps solids. The carried-over puddle is
#      placed only on pre-cleared floor cells on row 9, so it never overwrites a wall pixel.
#    - The rough/smooth launch strip exposes exactly one nearest drop direction, keeping the original
#      "unique reachable drop" assumption valid whenever a horizontal branch is allowed.
# e) Visual observability:
#    - Each accepted sample must contain at least one rough-block event and at least one evaporation
#      event, and the fused output must differ from both the friction-only and evaporation-only
#      baselines. That prevents silent degeneration back to a single-dimension variant.
#    - Because the source slab spans several repeated rough cells and several repeated smooth cells,
#      a human can match "water survives above color 2 but disappears above color 3" directly from
#      the input/output pictures without needing intermediate animation.
# f) Visual cleanliness:
#    - The layout is split into two readable bands: source slab above, basin structure below. No
#      legends or extra markers are added.
#    - Only three visible object types appear: water, rough walls, smooth walls. The lower puddle is
#      contiguous and the upper survivors stay in one strip, so outputs remain uncluttered.
# g) Change inferability:
#    - The crucial clue is structural, not numeric: the same wall color that blocks horizontal motion
#      also preserves water by preventing evaporation. Smooth supports show the complementary pattern.
#    - The generator keeps that clue redundant by repeating both colors multiple times per sample and
#      mirroring the whole scene left/right across the dataset.
# 9) New solver assumptions guaranteed by generate_input():
#    - Assumption: color 2 and color 3 are both present and play stable roles. Guarantee: every
#      template contains repeated cells of both colors.
#    - Assumption: any allowed horizontal spill has a unique nearest drop. Guarantee: templates use a
#      single open chute beside the launch strip, and candidates are validated by re-simulation.
#    - Assumption: evaporation is the only source of water loss. Guarantee: all generated samples are
#      checked so `water_lost == horizontal_evaps + diag_evaps`.
#    - Assumption: the fused behavior is visible, not accidental. Guarantee: candidates are rejected
#      unless they preserve rough-supported survivors, lose several pixels to evaporation, and differ
#      from both relevant one-dimension baselines.


BG = 0
WATER = 1
ROUGH = 2
SMOOTH = 3
GRID_SHAPE = (10, 10)
MAX_GENERATION_ATTEMPTS = 128

_TEMPLATE_SPECS = (
    {
        "bottom_rows": ("2222233000", "0000200003", "2222200003"),
        "source_widths": (7, 7),
        "carry_span": (6, 3),
    },
    {
        "bottom_rows": ("2222333000", "0002200003", "2222200003"),
        "source_widths": (7, 7),
        "carry_span": (6, 3),
    },
    {
        "bottom_rows": ("2222233300", "0000200003", "2222200003"),
        "source_widths": (7, 7),
        "carry_span": (6, 3),
    },
    {
        "bottom_rows": ("2222223300", "0000200003", "2222200003"),
        "source_widths": (8, 8),
        "carry_span": (6, 3),
    },
    {
        "bottom_rows": ("2222233000", "0000200303", "2222200003"),
        "source_widths": (7, 7),
        "carry_span": (6, 3),
    },
    {
        "bottom_rows": ("2222333000", "0000200303", "2222200003"),
        "source_widths": (7, 7),
        "carry_span": (6, 3),
    },
)


def _grid_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _build_candidate(spec: dict[str, object], mirror: bool) -> np.ndarray:
    rows = ["0000000000"] * 7 + list(spec["bottom_rows"])
    g = _grid_from_rows(rows)
    top_width, bottom_width = spec["source_widths"]
    carry_start, carry_len = spec["carry_span"]
    if top_width:
        g[5, : int(top_width)] = WATER
    if bottom_width:
        g[6, : int(bottom_width)] = WATER
    if carry_len:
        g[9, int(carry_start) : int(carry_start) + int(carry_len)] = WATER
    if mirror:
        g = np.fliplr(g).copy()
    return g


def _simulate(
    grid: np.ndarray,
    *,
    friction_enabled: bool,
    evaporation_enabled: bool,
    check_input: bool,
    check_output: bool,
    collect_metrics: bool = False,
) -> tuple[np.ndarray, dict[str, int]] | np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    metrics = (
        {
            "rough_blocks": 0,
            "horizontal_evaps": 0,
            "diag_evaps": 0,
        }
        if collect_metrics
        else None
    )

    if check_input:
        assert tuple(map(int, g.shape)) == GRID_SHAPE, "grid must stay 10x10"
        assert np.any(g == int(ROUGH)), "input must contain rough supports"
        assert np.any(g == int(SMOOTH)), "input must contain smooth supports"
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

    def settle(r: int, c: int, *, record_metrics: bool = True) -> tuple[int, int] | None:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
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
                if evaporation_enabled:
                    if record_metrics and metrics is not None:
                        metrics["diag_evaps"] += 1
                    return None
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
                if evaporation_enabled:
                    if record_metrics and metrics is not None:
                        metrics["horizontal_evaps"] += 1
                    return None
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
            dest = settle(int(r), int(c), record_metrics=True)
            if dest is None:
                moved = True
                continue
            rr, cc = dest
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
            dest = settle(int(rr), int(cc), record_metrics=False)
            g[rr, cc] = int(WATER)
            assert dest == (int(rr), int(cc)), "output water is not locally stable"

    if metrics is None:
        return g
    return g, metrics


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    out_variant, metrics = _simulate(
        grid,
        friction_enabled=True,
        evaporation_enabled=True,
        check_input=True,
        check_output=True,
        collect_metrics=True,
    )
    out_friction_only = _simulate(
        grid,
        friction_enabled=True,
        evaporation_enabled=False,
        check_input=True,
        check_output=True,
        collect_metrics=False,
    )
    out_evap_only = _simulate(
        grid,
        friction_enabled=False,
        evaporation_enabled=True,
        check_input=True,
        check_output=True,
        collect_metrics=False,
    )
    out_original = _simulate(
        grid,
        friction_enabled=False,
        evaporation_enabled=False,
        check_input=True,
        check_output=True,
        collect_metrics=False,
    )

    water_in = int(np.count_nonzero(np.asarray(grid) == int(WATER)))
    water_out = int(np.count_nonzero(out_variant == int(WATER)))
    rough_supported_output = 0
    smooth_supported_output = 0
    for r, c in np.argwhere(out_variant == int(WATER)):
        rr = int(r)
        cc = int(c)
        if rr + 1 >= out_variant.shape[0]:
            continue
        support = int(grid[rr + 1, cc])
        if support == int(ROUGH):
            rough_supported_output += 1
        elif support == int(SMOOTH):
            smooth_supported_output += 1

    result = dict(metrics)
    result["water_in"] = int(water_in)
    result["water_out"] = int(water_out)
    result["water_lost"] = int(water_in - water_out)
    result["evap_events"] = int(metrics["horizontal_evaps"] + metrics["diag_evaps"])
    result["top_water"] = int(np.count_nonzero(out_variant[:7] == int(WATER)))
    result["bottom_water"] = int(np.count_nonzero(out_variant[7:] == int(WATER)))
    result["rough_supported_output"] = int(rough_supported_output)
    result["smooth_supported_output"] = int(smooth_supported_output)
    result["diff_vs_friction_only"] = int(np.count_nonzero(out_variant != out_friction_only))
    result["diff_vs_evap_only"] = int(np.count_nonzero(out_variant != out_evap_only))
    result["diff_vs_original"] = int(np.count_nonzero(out_variant != out_original))
    return result


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate gravity-driven water in linked basins where sideways motion consumes the droplet and wall color controls whether that motion is allowed.

    Input structure:
    - Background is 0 and water is color 1.
    - Color 2 marks rough high-friction supports, while color 3 marks smooth low-friction supports.
    - Water starts as a supported upper source slab above those mixed supports, and some samples also retain a small stable puddle on the basin floor.

    Rule:
    - A water pixel may still fall straight down through background without any loss.
    - If its next move would be diagonal, the droplet evaporates instead of appearing in the diagonal cell.
    - When the pixel is supported and searches horizontally for the unique nearest drop, a rough support tile blocks that slide completely, but any allowed horizontal slide on a non-rough support also evaporates the droplet.
    - Repeat full settling passes until no surviving water pixel changes position and no further droplets evaporate.

    Key insight: Support color and evaporation are coupled: rough walls preserve water precisely because they prevent the sideways move that would otherwise make that droplet disappear.
    """
    return _simulate(
        grid,
        friction_enabled=True,
        evaporation_enabled=True,
        check_input=check_input,
        check_output=check_output,
        collect_metrics=False,
    )


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = random.Random() if rng is None else rng
    for _ in range(MAX_GENERATION_ATTEMPTS):
        spec = rng.choice(_TEMPLATE_SPECS)
        mirror = bool(rng.randrange(2))
        candidate = _build_candidate(spec, mirror)
        try:
            stats = variant_metrics(candidate)
        except AssertionError:
            continue
        if stats["rough_blocks"] < 1:
            continue
        if stats["horizontal_evaps"] < 1:
            continue
        if stats["water_lost"] != stats["evap_events"]:
            continue
        if stats["water_lost"] < 6:
            continue
        if stats["top_water"] < 4:
            continue
        if stats["bottom_water"] < 3:
            continue
        if stats["rough_supported_output"] < 4:
            continue
        if stats["smooth_supported_output"] != 0:
            continue
        if stats["diff_vs_friction_only"] < 8:
            continue
        if stats["diff_vs_evap_only"] < 4:
            continue
        if stats["diff_vs_original"] < 8:
            continue
        return candidate
    raise AssertionError("failed to generate a non-degenerate D12+D16 sample")
