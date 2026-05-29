# Variant of 28a6681f
# Dimension: D23 — 反弹/弹跳机制 (path)
# Change: 无反弹——水碰到阻挡后尝试对角线或水平滑动，最终停在支撑位置... → 水碰到壁面后弹起

from __future__ import annotations

import random

import numpy as np

BG = 0
WATER = 1
STRUCTURE_TEMPLATE_COLORS = (2, 3, 4)

# Design analysis for the D23 variant
# a) Concrete behavior and trace
#    The original task is a settling puzzle on a 10x10 basin scene. D23 changes only the path
#    rule at the instant when a water pixel finishes a vertical drop and strikes a solid floor.
#    In this variant, the first such solid-floor impact can launch the pixel over one adjacent
#    one-cell-thick wall. The ARC-grid analogue is: from (r, c), if the side cell (r, c+dir) is a
#    solid wall whose top is 1 or 2 cells above the impact row, and the cell beyond the wall
#    (r-rise, c+2*dir) is background, then the water bounces to that beyond-wall cell and resumes
#    ordinary settling. A concrete trace for the generator template is:
#    input source pixel on the upper shelf -> diagonal release into a shaft -> vertical fall to
#    the left impact chamber -> original logic would stop there -> D23 instead bounces over the
#    center divider -> falls into the right chamber and packs into a 2x2 block.
# b) Complexity calibration
#    Unlimited bouncing would create ping-pong loops and make examples unreadable. The variant
#    therefore allows at most one bounce per settling path, which is still visibly different from
#    the original task and enough to cross a low divider. Bounce height is limited to 1 or 2 by
#    the prompt; the generator uses the 2-cell case because it supports a stable 2x2 destination
#    pocket with four source pixels.
# c) Audit of original assumptions
#    - The original input-support assertions still hold and remain useful.
#    - The original diagonal and horizontal uniqueness assertions still hold, but now the solver
#      also needs the bounce choice to be unique. The generator builds exactly one low divider
#      next to the impact shaft, so the bounce branch is deterministic.
#    - The old settle() was stateless apart from position. D23 needs two extra pieces of local
#      state: whether the pixel has just completed a vertical fall, and whether its single bounce
#      has already been spent.
#    - Original code could safely treat every blocked-below event the same. Here we must separate
#      solid-floor impacts from water support; only the former can trigger a bounce.
# d) Placement legality
#    The template keeps all solids inside the 10x10 frame and bottom-anchors the basin walls via a
#    double base on rows 8-9. The upper source shelf and the top post use only colors that also
#    appear on the bottom row, so the original guide-color validation remains satisfied.
# e) Visual observability
#    The generator rejects any sample whose output under the new solver matches the old no-bounce
#    solver. In accepted samples, all four water pixels move from the near-left chamber to the
#    far-right chamber, so the bounce is readable from a few examples without inspecting IDs.
# f) Visual cleanliness
#    Only three solid colors are used, and the active interaction happens in one compact basin
#    cluster. There are no overlapping markers, hidden counters, or color-coded subrules.
# g) Color inferability
#    Colors are structural only. Water remains color 1, background remains 0, and solid colors are
#    randomly remapped among three distinct nonzero colors without changing their roles.
# h) Generator guarantees for new main() assumptions
#    - Every source pixel is supported and has at most one open horizontal side.
#    - The impact shaft has exactly one bounce-capable divider side.
#    - The destination chamber is a stable 2x2 pocket, so post-bounce water always has a valid
#      packed resting shape and the original output stability check remains meaningful.
# i) Reference-task note
#    A standalone local solver file for source task 36a08778 was not present; the closest
#    task-level code artifact in this workspace was
#    /home/lixinyu/snap/new/ceshi/code/36a08778/_variant_common.py. Its explicit impact-branching
#    style was used only as a complexity baseline; this file implements only the D23 change above.


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
            L = int(cc)
            while L - 1 >= 0 and int(g[rr, L - 1]) == int(BG):
                L -= 1
            R = int(cc)
            while R + 1 < W and int(g[rr, R + 1]) == int(BG):
                R += 1

            drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
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

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = _settle_pixel(g, int(r), int(c), allow_bounce=allow_bounce, stats=stats)
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
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    return g, stats


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin-filling water with a single rebound over low walls.

    Input structure:
    - bg = 0 and water = 1.
    - Other colors are rigid basin pieces, shelves, dividers, and optional upper guides.
    - Water pixels start supported on shelves, against walls, or stacked on other water.

    Rule:
    - A water pixel still falls straight down through background cells.
    - On the first solid-floor impact after a vertical drop, it may rebound over the unique
      adjacent one-cell-thick wall whose top is 1-2 cells above the impact row.
    - After that one bounce, the original diagonal slipping and supported horizontal sliding
      rules continue until the pixel reaches a packed resting place.

    Key insight:
    A low divider is no longer a permanent separator: a falling pixel can use one elastic hop to
    clear it, so the deepest reachable chamber may now lie behind a wall that the original solver
    treated as final.
    """
    out, _stats = _simulate(
        grid,
        allow_bounce=True,
        check_input=bool(check_input),
        check_output=bool(check_output),
    )
    return out


def _build_template() -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    a, b, c = STRUCTURE_TEMPLATE_COLORS

    g[0:2, 0:2] = WATER
    g[2, 0:2] = a
    g[0:3, 3] = a

    g[6:10, 0] = a
    g[6:10, 3] = b
    g[6:10, 6] = c

    g[8:10, 0:3] = a
    g[8:10, 3] = b
    g[8:10, 4:7] = c
    return g


def _remap_structure_colors(g: np.ndarray) -> np.ndarray:
    targets = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)
    mapping = dict(zip(STRUCTURE_TEMPLATE_COLORS, targets))
    out = g.copy()
    for src, dst in mapping.items():
        out[g == int(src)] = int(dst)
    return out


def _water_signature(g: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))}


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    bounced_out, bounce_stats = _simulate(grid, allow_bounce=True, check_input=True, check_output=True)
    plain_out, _plain_stats = _simulate(grid, allow_bounce=False, check_input=True, check_output=True)
    bounced_water = _water_signature(bounced_out)
    plain_water = _water_signature(plain_out)
    return {
        "bounce_count": int(bounce_stats["bounce_count"]),
        "bounce_heights": tuple(int(x) for x in bounce_stats["bounce_heights"]),
        "water_symmetric_difference": len(bounced_water.symmetric_difference(plain_water)),
        "outputs_differ": bool(np.any(bounced_out != plain_out)),
    }


def generate_input() -> np.ndarray:
    """Generate a bounce-dependent input for the D23 variant."""
    source_water = 4
    for _ in range(200):
        g = _build_template()
        if random.random() < 0.5:
            g = np.fliplr(g)
        g = _remap_structure_colors(g)

        metrics = _behavior_metrics(g)
        if int(metrics["bounce_count"]) != source_water:
            continue
        if tuple(metrics["bounce_heights"]) != (2, 2, 2, 2):
            continue
        if not bool(metrics["outputs_differ"]):
            continue
        if int(metrics["water_symmetric_difference"]) < 2 * source_water:
            continue
        return g

    raise RuntimeError("failed to generate a bounce-distinguishing D23 input")
