# Combo variant of 28a6681f (index 15)
# combo_spec: D17/2+D23/0
# Name: 按壁面距离排序+弹跳机制

from __future__ import annotations

import random

import numpy as np

BG = 0
WATER = 1
STRUCTURE_TEMPLATE_COLORS = (2, 3, 4)
GENERATION_ATTEMPTS = 200


# Design analysis for combo D17/2 + D23/0
# ----------------------------------------
# a) Concretization + behavior trace:
#    - The scene stays a 10x10 water-settling puzzle with rigid colored solids and water color 1.
#      The generator uses one compact basin family: a bottom-anchored left gutter, a center
#      divider, a right chamber, and a short upper post beside a two-pixel near-wall source.
#      A third water pixel starts lower down inside a shelf pocket, farther from any side wall.
#    - Concrete unmirrored trace:
#        * Near-wall waters start at (3,4) and (4,4), each with nearest side-wall distance 1.
#        * The lower blocker water starts at (6,4) with nearest side-wall distance 2.
#        * Original D23-only order (`reverse`): the lower blocker moves away first to (7,2),
#          opening a direct route. Only one upper pixel later gets a bounce; the other settles
#          inside the basin.
#        * Fused order (`wall_distance`): the two near-wall pixels are processed before the farther
#          blocker. Because the blocker is still sitting at (6,4), both upper pixels are rerouted
#          left, reach the low left wall, and each bounce two rows upward over it into the edge
#          gutter, then fall to the bottom edge.
#      The visible difference is therefore not "sorting in isolation" and not "bounce in isolation":
#      the early near-wall settle order changes whether the blocker is still present when a bounce
#      path is evaluated.
#
# b) Complexity calibration:
#    - The family is intentionally narrow: one validated landform, horizontal mirroring, and
#      structure recoloring. This keeps the rule readable while still giving multiple distinct
#      input/output pairs.
#    - Water count is fixed at three. Two repeated near-wall pixels provide a stable visual cue
#      for the distance rule, while one farther pixel provides the ordering conflict.
#    - Bounce height is fixed at 2 in generated samples so the hop is easy to read.
#
# c) Hidden-assumption audit of the original main():
#    - The original per-pass water order was pure coordinate order. That assumption is false here;
#      we must compute a new deterministic order from distance to the nearest wall surface.
#    - Using "distance to any solid" would collapse the rule, because every supported water pixel
#      would be distance 1 from the floor beneath it. The solver therefore measures distance only
#      to side-exposed solid cells, i.e. visible wall faces rather than generic supports.
#    - The D23 bounce rule still needs state: whether a vertical fall just ended, and whether the
#      pixel has already spent its single bounce.
#    - The old diagonal/horizontal uniqueness assumptions remain useful and are preserved. The
#      generator keeps all bounce choices unique as well.
#    - Tie breaking inside one distance shell is top-to-bottom then left-to-right. The generator
#      does not rely on that tie break for the key combo effect: the decisive blocker water is at
#      distance 2 while the decisive near-wall waters are at distance 1.
#
# d) Placement legality:
#    - The template is hand-validated and stays strictly in-bounds.
#    - Mirroring preserves non-overlap and bottom anchoring.
#    - Recoloring only remaps structure colors and never changes background or water.
#
# e) Visual observability:
#    - In every accepted sample, the fused solver sends two near-wall waters into the edge gutter,
#      the bounce-only ablation sends only one, and the no-bounce ablation sends none.
#    - The ordering cue is geometric and human-visible: the two upper waters are visibly pressed
#      against the short post, while the blocker water sits one cell farther from any side wall.
#    - No invisible attribute such as color id or generation-time randomness participates in the
#      sort key.
#
# f) Visual cleanliness:
#    - The active interaction is localized to one compact chute-and-basin cluster.
#    - There are only three structure colors plus water, with no extra markers or overlays.
#
# g) Color inferability:
#    - Colors remain purely structural. Water is always 1 and background is always 0.
#    - Random recoloring uses only three remapped solid colors, so the rule stays geometric.
#
# 9) New solver assumptions guaranteed by generate_input():
#    - At least one side-exposed wall cell exists, so wall-distance ordering is defined.
#    - The input always contains exactly three water pixels in the validated template family.
#    - The key waters occupy two distinct wall-distance shells: two pixels at distance 1 and one
#      blocker pixel at distance 2.
#    - Generated samples satisfy these behavior metrics:
#        combo (`wall_distance` + bounce) -> exactly 2 bounces and 2 edge-gutter waters
#        D23-only (`reverse` + bounce)    -> exactly 1 bounce and 0 edge-gutter waters
#        D17-only (`wall_distance`, no bounce) -> 0 bounces
#      This prevents silent regression to either single-variant behavior.


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _side_wall_cells(g: np.ndarray) -> list[tuple[int, int]]:
    solid = (g != int(BG)) & (g != int(WATER))
    H, W = solid.shape
    walls: list[tuple[int, int]] = []
    for r, c in np.argwhere(solid):
        rr = int(r)
        cc = int(c)
        left_open = bool(cc == 0 or not bool(solid[rr, cc - 1]))
        right_open = bool(cc == W - 1 or not bool(solid[rr, cc + 1]))
        if left_open or right_open:
            walls.append((rr, cc))
    if walls:
        return walls
    return [(int(r), int(c)) for r, c in np.argwhere(solid)]


def _nearest_wall_distance(g: np.ndarray, r: int, c: int) -> int:
    rr = int(r)
    cc = int(c)
    return min(abs(int(rr - wr)) + abs(int(cc - wc)) for wr, wc in _side_wall_cells(g))


def _ordered_water_positions(g: np.ndarray, order_mode: str) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))]
    if order_mode == "reverse":
        return sorted(pts, reverse=True)
    if order_mode == "wall_distance":
        return sorted(pts, key=lambda rc: (_nearest_wall_distance(g, rc[0], rc[1]), rc[0], rc[1]))
    raise AssertionError(f"unknown order mode: {order_mode}")


def _attempt_bounce(g: np.ndarray, rr: int, cc: int) -> tuple[int, int, int] | None:
    """Return (new_r, new_c, rise) for a unique bounce-over-wall move."""
    H, W = g.shape
    if rr + 1 >= H:
        return None
    below = int(g[rr + 1, cc])
    if below in {BG, WATER}:
        return None

    options: list[tuple[int, int, int]] = []
    for dc in (-1, 1):
        wall_c = int(cc + dc)
        land_c = int(cc + 2 * dc)
        if not (0 <= wall_c < W and 0 <= land_c < W):
            continue
        if int(g[rr, wall_c]) in {BG, WATER}:
            continue

        rise = None
        if rr - 1 >= 0 and int(g[rr - 1, wall_c]) == int(BG):
            rise = 1
        elif rr - 2 >= 0 and int(g[rr - 1, wall_c]) not in {BG, WATER} and int(g[rr - 2, wall_c]) == int(BG):
            rise = 2

        if rise is None:
            continue
        land_r = int(rr - rise)
        if int(g[land_r, land_c]) != int(BG):
            continue
        options.append((int(land_r), int(land_c), int(rise)))

    if not options:
        return None
    assert len(options) == 1, f"ambiguous bounce moves: {options}"
    return options[0]


def _settle_pixel(
    g: np.ndarray,
    r: int,
    c: int,
    *,
    allow_bounce: bool,
    stats: dict[str, object] | None,
) -> tuple[int, int]:
    H, W = g.shape
    rr = int(r)
    cc = int(c)
    bounce_used = not bool(allow_bounce)
    fell_since_event = False

    for _ in range(H * W * 2):
        if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
            rr += 1
            fell_since_event = True
            continue

        if fell_since_event and not bounce_used:
            bounced = _attempt_bounce(g, rr, cc)
            if bounced is not None:
                rr, cc, rise = bounced
                bounce_used = True
                fell_since_event = False
                if stats is not None:
                    stats["bounce_count"] = int(stats["bounce_count"]) + 1
                    cast_heights = stats["bounce_heights"]
                    assert isinstance(cast_heights, list)
                    cast_heights.append(int(rise))
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
            fell_since_event = False
            continue

        if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
            left = int(cc)
            while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                left -= 1
            right = int(cc)
            while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                right += 1

            drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
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
            fell_since_event = False
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")

    return int(rr), int(cc)


def _simulate(
    grid: np.ndarray,
    *,
    order_mode: str,
    allow_bounce: bool,
    check_input: bool,
    check_output: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    stats: dict[str, object] = {"bounce_count": 0, "bounce_heights": []}

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    assert _side_wall_cells(g), "scene must contain at least one side-exposed wall"

    for _pass in range(H * W):
        moved = False
        pts = _ordered_water_positions(g, order_mode=order_mode)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = _settle_pixel(g, int(r), int(c), allow_bounce=allow_bounce, stats=stats)
            assert int(g[rr, cc]) == int(BG), "water paths should not overlap"
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    return g, stats


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin water that is processed by wall proximity and can rebound over low walls.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors are rigid basin parts, ledges, dividers, and short posts that act as wall faces.
    - Water pixels start in supported source positions, with some pressed directly against a wall and
      others parked farther inside a shelf pocket.

    Rule:
    - In each pass, process water pixels from nearest wall face to farthest wall face, using a fixed
      top-to-bottom, left-to-right tie break inside one distance shell.
    - Each pixel still follows the original local physics: fall vertically through background, slip
      diagonally only through an open side channel, and slide horizontally along a supported run
      toward the nearest unique drop.
    - After a vertical fall hits a solid floor, the pixel may use one bounce of height 1 or 2 to hop
      over the unique adjacent low wall, then continue settling normally.

    Key insight:
    The bounce is path-dependent and the path is order-dependent: near-wall water settles first while
    farther water may still be sitting in a would-be route, so early wall-adjacent pixels can bounce
    into regions that later pixels can no longer reach.
    """
    out, _stats = _simulate(
        grid,
        order_mode="wall_distance",
        allow_bounce=True,
        check_input=bool(check_input),
        check_output=bool(check_output),
    )
    return out


def _build_template() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 3, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 3, 0, 0, 0, 0],
            [0, 2, 0, 0, 1, 3, 3, 0, 4, 0],
            [0, 2, 0, 2, 3, 3, 2, 0, 4, 0],
            [0, 2, 2, 2, 2, 3, 4, 4, 4, 0],
            [0, 2, 2, 2, 2, 3, 4, 4, 4, 0],
        ],
        dtype=int,
    )


def _remap_structure_colors(g: np.ndarray) -> np.ndarray:
    targets = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)
    mapping = dict(zip(STRUCTURE_TEMPLATE_COLORS, targets))
    out = g.copy()
    for src, dst in mapping.items():
        out[g == int(src)] = int(dst)
    return out


def _water_signature(g: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))}


def _edge_water_count(g: np.ndarray) -> int:
    H, W = g.shape
    count = 0
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        if cc in (0, W - 1):
            count += 1
        assert 0 <= rr < H
    return int(count)


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    combo_out, combo_stats = _simulate(
        grid,
        order_mode="wall_distance",
        allow_bounce=True,
        check_input=True,
        check_output=True,
    )
    reverse_bounce_out, reverse_bounce_stats = _simulate(
        grid,
        order_mode="reverse",
        allow_bounce=True,
        check_input=True,
        check_output=False,
    )
    distance_only_out, distance_only_stats = _simulate(
        grid,
        order_mode="wall_distance",
        allow_bounce=False,
        check_input=True,
        check_output=False,
    )
    plain_out, plain_stats = _simulate(
        grid,
        order_mode="reverse",
        allow_bounce=False,
        check_input=True,
        check_output=False,
    )

    combo_water = _water_signature(combo_out)
    reverse_bounce_water = _water_signature(reverse_bounce_out)
    distance_only_water = _water_signature(distance_only_out)
    plain_water = _water_signature(plain_out)
    return {
        "combo_bounce_count": int(combo_stats["bounce_count"]),
        "combo_bounce_heights": tuple(int(x) for x in combo_stats["bounce_heights"]),
        "reverse_bounce_count": int(reverse_bounce_stats["bounce_count"]),
        "reverse_bounce_heights": tuple(int(x) for x in reverse_bounce_stats["bounce_heights"]),
        "distance_only_bounce_count": int(distance_only_stats["bounce_count"]),
        "plain_bounce_count": int(plain_stats["bounce_count"]),
        "combo_edge_water": _edge_water_count(combo_out),
        "reverse_edge_water": _edge_water_count(reverse_bounce_out),
        "distance_only_edge_water": _edge_water_count(distance_only_out),
        "combo_vs_reverse_diff": len(combo_water.symmetric_difference(reverse_bounce_water)),
        "combo_vs_distance_only_diff": len(combo_water.symmetric_difference(distance_only_water)),
        "combo_vs_plain_diff": len(combo_water.symmetric_difference(plain_water)),
        "combo_output_differs": bool(np.any(combo_out != reverse_bounce_out)),
        "combo_vs_distance_only_output_differs": bool(np.any(combo_out != distance_only_out)),
        "reverse_vs_plain_output_differs": bool(np.any(reverse_bounce_out != plain_out)),
    }


def generate_input() -> np.ndarray:
    """Generate a wall-distance-order-dependent bounce sample for the fused variant."""
    for _ in range(GENERATION_ATTEMPTS):
        g = _build_template()
        if random.random() < 0.5:
            g = np.fliplr(g)
        g = _remap_structure_colors(g)

        metrics = _behavior_metrics(g)
        if int(metrics["combo_bounce_count"]) != 2:
            continue
        if tuple(metrics["combo_bounce_heights"]) != (2, 2):
            continue
        if int(metrics["reverse_bounce_count"]) != 1:
            continue
        if tuple(metrics["reverse_bounce_heights"]) != (2,):
            continue
        if int(metrics["distance_only_bounce_count"]) != 0:
            continue
        if int(metrics["combo_edge_water"]) != 2:
            continue
        if int(metrics["reverse_edge_water"]) != 0:
            continue
        if int(metrics["distance_only_edge_water"]) != 0:
            continue
        if not bool(metrics["combo_output_differs"]):
            continue
        if not bool(metrics["combo_vs_distance_only_output_differs"]):
            continue
        if int(metrics["combo_vs_reverse_diff"]) < 4:
            continue
        if int(metrics["combo_vs_distance_only_diff"]) < 4:
            continue
        if int(metrics["combo_vs_plain_diff"]) < 2:
            continue
        return g

    raise RuntimeError("failed to generate a valid D17/2+D23/0 combo input")
