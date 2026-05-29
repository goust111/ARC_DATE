# Variant of 409aa875
# Dimension: D18 — 终止/目标条件 (dst)
# Change: 无显式目标；投射物飞行固定距离后落地，效果取决于落点内容... → 在特定颜色线上停止

import random
import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D18 variant
# a) Concrete behavior:
#    The original grid only had 3-pixel triangle sprites. This variant adds
#    long 1-cell-thick horizontal/vertical line segments. A triangle still
#    encodes a direction, but the projectile no longer lands at apex+5*dir.
#    Instead it advances cell by cell until it first touches a colored line,
#    then the entire struck line segment is recolored to maroon (9).
#    Example: if an L-triangle fires up-right from apex (8,4), the old rule
#    always inspected (3,9). The new rule visits (7,5),(6,6),(5,7),... and if
#    a horizontal segment occupies row 5 through col 7, the stop cell is (5,7)
#    and that whole segment changes color.
# b) Complexity calibration:
#    Line segments are length 4-8 so they are visually distinct from 3-pixel
#    triangles. Each grid has 2-4 lines and 4-8 triangles. At least one line
#    is hit and at least one line is left untouched, so the rule is visible
#    from a few examples without cluttering the scene.
# c) Hidden assumptions from the old solver:
#    - Old code assumed every non-bg monochrome component could be treated as a
#      projectile source or collision target; this fails once lines exist.
#    - Old code hard-coded apex+5*dir; that premise is intentionally broken.
#    - Old output logic painted bg markers or recolored hit sprites. In this
#      variant only line segments can change, so destination aggregation and
#      collision coloring are removed.
#    - The original solver would accidentally match this variant if many lines
#      were placed exactly 5 steps away. The generator forbids that by only
#      using stop distances in {3,4,6,7}.
# d) Placement legality:
#    The generator places isolated straight segments first, then adds triangles
#    whose rays reach a chosen line with no occupied cell earlier on the path.
#    Objects never overlap or touch in the 8-neighborhood, and lines never
#    intersect each other, so component classification stays stable.
# e) Visual observability:
#    Inputs visibly contain triangles plus long colored bars. Outputs keep the
#    triangles unchanged and recolor only the bars that lie on the firing rays.
#    Because at least one untouched bar remains, the stopping condition is
#    inferable by comparison.
# f) Visual cleanliness:
#    Lines are axis-aligned, one cell thick, and separated from triangles. That
#    avoids ambiguous merged shapes and keeps the recolored regions legible.
# g) Color-change inferability:
#    The output uses a single effect color (9) for hit line segments. Input
#    line colors are chosen from a small palette excluding 9, so the viewer can
#    attribute the recoloring to the ray/line interaction rather than to an
#    opaque color remapping rule.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        corner = corners[0]
        cr, cc = corner
        arms = []
        for r, c in comp:
            if (r, c) == corner:
                continue
            dr, dc = r - cr, c - cc
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr_sum = arms[0][0] + arms[1][0]
        dc_sum = arms[0][1] + arms[1][1]
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
        return corner, (dr, dc)

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    dr = -((d1[0] + d2[0]) // 2)
    dc = -((d1[1] + d2[1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _line_orientation(comp: list[tuple[int, int]]) -> str | None:
    if len(comp) < 4:
        return None
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1:
        ordered = sorted(c for _, c in comp)
        return "h" if ordered == list(range(ordered[0], ordered[-1] + 1)) else None
    if len(cols) == 1:
        ordered = sorted(r for r, _ in comp)
        return "v" if ordered == list(range(ordered[0], ordered[-1] + 1)) else None
    return None


def _extract_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    triangles: list[dict] = []
    lines: list[dict] = []
    for col in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for comp in _cc8(grid == col):
            ad = _triangle_apex_and_dir(comp)
            if ad is not None:
                triangles.append(
                    {
                        "cells": tuple(sorted(comp)),
                        "cell_set": frozenset(comp),
                        "apex": ad[0],
                        "dir": ad[1],
                        "color": col,
                    }
                )
                continue
            orient = _line_orientation(comp)
            if orient is not None:
                lines.append(
                    {
                        "cells": tuple(sorted(comp)),
                        "cell_set": frozenset(comp),
                        "orientation": orient,
                        "color": col,
                    }
                )
    return triangles, lines


def _trace_hits(triangles: list[dict], lines: list[dict], shape: tuple[int, int]) -> list[dict]:
    line_cell_to_id: dict[tuple[int, int], int] = {}
    for lid, line in enumerate(lines):
        for rc in line["cells"]:
            line_cell_to_id[rc] = lid

    h, w = shape
    hits: list[dict] = []
    for tid, tri in enumerate(triangles):
        ar, ac = tri["apex"]
        dr, dc = tri["dir"]
        rr, cc = ar, ac
        dist = 0
        while True:
            rr += dr
            cc += dc
            dist += 1
            if not (0 <= rr < h and 0 <= cc < w):
                break
            lid = line_cell_to_id.get((rr, cc))
            if lid is not None:
                hits.append({"triangle_id": tid, "line_id": lid, "cell": (rr, cc), "distance": dist})
                break
    return hits


def _analyze_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, lines = _extract_objects(g, bg)
    hits = _trace_hits(triangles, lines, g.shape)
    hit_line_ids = {hit["line_id"] for hit in hits}
    return {
        "triangle_count": len(triangles),
        "line_count": len(lines),
        "hit_count": len(hits),
        "hit_line_count": len(hit_line_ids),
        "untouched_line_count": len(lines) - len(hit_line_ids),
        "distances": [hit["distance"] for hit in hits],
        "hit_line_ids": sorted(hit_line_ids),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire triangle rays until they strike a colored line segment, then repaint that segment.

    Input structure:
    - bg = most common color.
    - Non-bg objects are of two kinds: 3-pixel triangle sprites (axis-aligned L or diagonal V)
      and longer 1-cell-thick horizontal/vertical line segments.

    Rule:
    - Each triangle still determines an emission direction from its shape, exactly as in the
      original task.
    - Starting from the apex, advance one step at a time along that direction until the ray first
      reaches a colored line cell; triangles do not move and no intermediate trail is drawn.
    - When a line is struck, recolor the entire contiguous line segment containing that hit cell
      to maroon (9).
    - Rays that never meet a line do nothing.

    Key insight: the triangle orientation still encodes the ray direction, but the destination is
    no longer a fixed offset; the first blocking line on that path determines where the action
    stops and which whole segment changes color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, lines = _extract_objects(g, bg)
    hits = _trace_hits(triangles, lines, g.shape)

    out = g.copy()
    for lid in {hit["line_id"] for hit in hits}:
        for r, c in lines[lid]["cells"]:
            out[r, c] = 9
    return out


# --------------- generator (D18 line-stop variant) ---------------

H = W = 16
BG = 7
HIT_COLOR = 9
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8]
LINE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8]

SPRITE_COUNT_RANGE = (4, 8)
LINE_COUNT_RANGE = (2, 4)
LINE_LEN_RANGE = (4, 8)
RAY_STEPS = (3, 4, 6, 7)

MAX_ATTEMPTS = 500


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _line_cells(start: tuple[int, int], orientation: str, length: int) -> tuple[tuple[int, int], ...]:
    r0, c0 = start
    if orientation == "h":
        return tuple((r0, c0 + k) for k in range(length))
    return tuple((r0 + k, c0) for k in range(length))


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_lines(rng: random.Random, sprite_color: int, n_lines: int) -> list[dict] | None:
    line_palette = [c for c in LINE_COLORS if c != sprite_color and c != HIT_COLOR]
    if len(line_palette) < n_lines:
        return None
    colors = rng.sample(line_palette, n_lines)

    lines: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for color in colors:
        placed = False
        for _ in range(200):
            orientation = rng.choice(["h", "v"])
            length = rng.randint(*LINE_LEN_RANGE)
            if orientation == "h":
                r0 = rng.randint(1, H - 2)
                c0 = rng.randint(0, W - length)
            else:
                r0 = rng.randint(0, H - length)
                c0 = rng.randint(1, W - 2)
            cells = _line_cells((r0, c0), orientation, length)
            if _halo(cells) & occupied:
                continue
            record = {
                "cells": cells,
                "cell_set": frozenset(cells),
                "halo": _halo(cells),
                "orientation": orientation,
                "color": color,
            }
            lines.append(record)
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None
    return lines


def _dirs_for_line(orientation: str) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [(0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _trace_path(apex: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    ar, ac = apex
    dr, dc = direction
    return [(ar + k * dr, ac + k * dc) for k in range(1, steps + 1)]


def _place_sprite_to_line(
    rng: random.Random,
    line: dict,
    occupied: set[tuple[int, int]],
) -> dict | None:
    line_cells = list(line["cells"])
    directions = _dirs_for_line(line["orientation"])
    rng.shuffle(line_cells)
    rng.shuffle(directions)

    for hit_cell in line_cells:
        for direction in directions:
            step_options = list(RAY_STEPS)
            rng.shuffle(step_options)
            for steps in step_options:
                hr, hc = hit_cell
                dr, dc = direction
                apex = (hr - steps * dr, hc - steps * dc)
                cells = _sprite_cells(apex, direction)
                if not _in_bounds_cells(cells):
                    continue
                if _halo(cells) & occupied:
                    continue
                ray = _trace_path(apex, direction, steps)
                if any(rc in occupied for rc in ray[:-1]):
                    continue
                return {
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "apex": apex,
                    "dir": direction,
                    "target_line_color": line["color"],
                    "target_hit_cell": hit_cell,
                    "steps": steps,
                }
    return None


def _validate_input(grid: np.ndarray, expected_sprites: int, expected_lines: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    stats = _analyze_instance(grid)
    if stats["triangle_count"] != expected_sprites:
        return False
    if stats["line_count"] != expected_lines:
        return False
    if stats["hit_count"] != expected_sprites:
        return False
    if stats["hit_line_count"] < 1 or stats["untouched_line_count"] < 1:
        return False
    if any(dist == 5 for dist in stats["distances"]):
        return False
    if any(dist not in RAY_STEPS for dist in stats["distances"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_lines = rng.randint(*LINE_COUNT_RANGE)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)

    lines = _place_lines(rng, sprite_color, n_lines)
    if lines is None:
        return None

    target_line_count = rng.randint(1, n_lines - 1)
    target_ids = rng.sample(range(n_lines), target_line_count)
    assignments = [target_ids[i % target_line_count] for i in range(n_sprites)]
    rng.shuffle(assignments)

    occupied = set()
    for line in lines:
        occupied.update(line["cells"])

    sprites: list[dict] = []
    for line_id in assignments:
        sprite = _place_sprite_to_line(rng, lines[line_id], occupied)
        if sprite is None:
            return None
        sprites.append(sprite)
        occupied.update(sprite["cells"])

    inp = np.full((H, W), BG, dtype=int)
    for line in lines:
        _paint_cells(inp, line["cells"], line["color"])
    for sprite in sprites:
        _paint_cells(inp, sprite["cells"], sprite_color)

    if not _validate_input(inp, n_sprites, n_lines):
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    if not np.any(out == HIT_COLOR):
        return None

    stats = _analyze_instance(inp)
    if stats["hit_line_count"] >= stats["line_count"]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
