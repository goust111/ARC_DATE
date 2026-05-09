# Combo variant of 409aa875 (index 21)
# combo_spec: D6/0+D18/2
# Name: 连续轨迹+彩色线上停止

# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace
#    - Base task: each 3-pixel triangle encodes a direction, then "teleports" its
#      projectile to apex + 5*dir and only that destination cell/sprite changes.
#    - D18/2 replaces the fixed-distance destination with the first colored line met
#      along the ray. D6/0 makes the whole flight visible instead of only the final
#      effect.
#    - Fused rule: triangles still determine direction exactly as before. Starting
#      from the apex, move one step at a time until the first line cell is reached.
#      Every background cell on that path is painted maroon 9, and the entire struck
#      line segment is recolored to 9.
#    - Example: suppose an L-sprite has apex (10,4) and dir=(-1,1), and a horizontal
#      line occupies row 6, cols 6..10. The base task would inspect only (5,9). The
#      fused task visits (9,5),(8,6),(7,7),(6,8): cells (9,5),(8,6),(7,7) become 9,
#      then the ray stops at line cell (6,8) and the whole line segment turns 9.
#
# b) Complexity calibration
#    - The board now contains two visually distinct object classes: 3-pixel triangles
#      and 4-8 cell straight bars. That is enough structure for a human to separate
#      emitters from blockers without extra metadata.
#    - Trail color is fixed to a single effect color (9). This keeps the new visual
#      signal strong without introducing a second mapping problem.
#    - Stop distances are sampled from {3,4,6,7} rather than 5. This makes the line-
#      stop rule visibly different from the original fixed-offset task and prevents a
#      degenerate sample from accidentally agreeing with the base logic.
#
# c) Hidden assumptions from the original main()
#    - Old assumption: every non-bg component is a triangle. Broken once long lines
#      are added, so the solver must explicitly classify components as triangle vs line.
#    - Old assumption: the destination is known from apex + 5*dir. Broken by D18/2,
#      so the solver must trace cell-by-cell until it hits a line or exits the grid.
#    - Old assumption: only the endpoint is painted. Broken by D6/0, so write order
#      now matters: paint trail cells first, then recolor hit line segments.
#    - Old assumption: bg collisions / hit-sprite recolors are the only outcomes.
#      In the fused rule there are no bg collision markers at all; the struck line is
#      the terminal object and triangles themselves stay unchanged.
#
# d) Placement legality in the generator
#    - Lines are placed first, axis-aligned, 1 cell thick, length 4-8, mutually
#      non-touching, and separated from later triangles by at least one cell.
#    - Each triangle is placed so that its ray reaches a chosen line at distance
#      3/4/6/7, with all intermediate cells currently empty background.
#    - Intermediate trail cells are reserved so later sprites cannot occupy or cross
#      them. This keeps the input legal and the output legible.
#
# e) Visual observability
#    - From a few examples, a human can read the firing direction from the triangle
#      geometry, then see a straight maroon trail stop exactly when it reaches a
#      pre-existing colored bar. Hit bars change to maroon while untouched bars keep
#      their input colors, making the stopping condition explicit.
#    - The generator guarantees at least one hit line and at least one untouched line,
#      so the line-stop criterion is contrastive rather than implicit.
#
# f) Visual cleanliness
#    - Trails are only drawn on background cells before the hit cell, never through
#      objects. Different trails are kept disjoint off the target line to avoid visual
#      tangles, and lines themselves never intersect.
#    - Sprite color and line colors exclude 9 so the added maroon effects remain easy
#      to spot.
#
# g) Change inferability
#    - Output color changes are structurally tied to geometry: maroon 9 means either
#      "ray path" or "line that the ray first hit". There is no hidden per-color
#      remapping and no ordering by numeric color id.
#
# Post-change main() assumptions checked against generate_input():
#    - `_extract_objects()` assumes every non-bg component is either a 3-pixel
#      triangle or a straight line segment of length >= 4; the generator constructs
#      only those two kinds and keeps them disconnected.
#    - `_trace_hits()` assumes the first occupied cell on a ray, if any, is a line
#      cell rather than another triangle; the generator forbids intermediate sprite or
#      trail crossings.
#    - Trail painting assumes each stored trail cell is background in the input; the
#      generator reserves those cells before placing later objects.
#    - Whole-line recoloring assumes contiguous line segments can be recovered from one
#      hit cell; the generator never lets two lines touch or intersect.

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# ---- canvas / palette ----
H = W = 16
BG = 7
EFFECT_COLOR = 9
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8]
LINE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8]

# ---- geometry ----
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
RAY_STEPS = (3, 4, 6, 7)

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 8)
LINE_COUNT_RANGE = (2, 4)
LINE_LEN_RANGE = (4, 8)

# ---- generation retries ----
MAX_ATTEMPTS = 800


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
        if abs(dr) + abs(dc) != 2:
            return None
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
        trail_mid: list[tuple[int, int]] = []
        dist = 0
        while True:
            rr += dr
            cc += dc
            dist += 1
            if not (0 <= rr < h and 0 <= cc < w):
                break
            lid = line_cell_to_id.get((rr, cc))
            if lid is not None:
                hits.append(
                    {
                        "triangle_id": tid,
                        "line_id": lid,
                        "hit_cell": (rr, cc),
                        "distance": dist,
                        "trail_mid": tuple(trail_mid),
                    }
                )
                break
            trail_mid.append((rr, cc))
    return hits


def _analyze_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, lines = _extract_objects(g, bg)
    hits = _trace_hits(triangles, lines, g.shape)
    hit_line_ids = {hit["line_id"] for hit in hits}
    trail_lengths = [len(hit["trail_mid"]) for hit in hits]
    return {
        "triangle_count": len(triangles),
        "line_count": len(lines),
        "hit_count": len(hits),
        "hit_line_count": len(hit_line_ids),
        "untouched_line_count": len(lines) - len(hit_line_ids),
        "distances": [hit["distance"] for hit in hits],
        "trail_lengths": trail_lengths,
        "trail_cells_total": sum(trail_lengths),
        "hit_line_ids": sorted(hit_line_ids),
    }


def _render_line_hits_only(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    bg = _bg_color(g)
    _, lines = _extract_objects(g, bg)
    hits = _trace_hits(*_extract_objects(g, bg), g.shape)
    out = g.copy()
    for lid in {hit["line_id"] for hit in hits}:
        for r, c in lines[lid]["cells"]:
            out[r, c] = EFFECT_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire triangle rays until they strike a colored line, while drawing the full path.

    Input structure:
    - bg is the most common color.
    - Non-bg objects are either 3-pixel triangle sprites (axis-aligned L or diagonal V)
      or longer 1-cell-thick horizontal/vertical line segments.

    Rule:
    - Each triangle still determines a single firing direction from its geometry: L-shapes
      emit diagonally away from the missing 2x2 corner, and V-shapes emit axially away from
      their opening.
    - Starting at the apex, advance cell by cell along that direction until the ray first
      reaches a colored line cell. Every background cell before that hit is painted maroon (9).
    - When a line is struck, recolor the entire contiguous line segment containing that hit
      cell to maroon (9). Triangles do not change, and rays that never meet a line do nothing.

    Key insight:
    - Triangle orientation still tells you which way the projectile travels, but the final
      effect is now determined by the first blocking line on that path, and the visible trail
      shows the whole run-up to that stop.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    triangles, lines = _extract_objects(g, bg)
    hits = _trace_hits(triangles, lines, g.shape)

    out = g.copy()
    for hit in hits:
        for r, c in hit["trail_mid"]:
            if g[r, c] == bg:
                out[r, c] = EFFECT_COLOR

    for lid in {hit["line_id"] for hit in hits}:
        for r, c in lines[lid]["cells"]:
            out[r, c] = EFFECT_COLOR
    return out


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


def _dirs_for_line(orientation: str) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [(0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _direction_family(direction: tuple[int, int]) -> str:
    return "diag" if abs(direction[0]) + abs(direction[1]) == 2 else "axis"


def _trace_path(apex: tuple[int, int], direction: tuple[int, int], steps: int) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    return tuple((ar + k * dr, ac + k * dc) for k in range(1, steps + 1))


def _place_lines(rng: random.Random, sprite_color: int, n_lines: int) -> list[dict] | None:
    line_palette = [c for c in LINE_COLORS if c != sprite_color]
    if len(line_palette) < n_lines:
        return None
    colors = rng.sample(line_palette, n_lines)

    lines: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    orientations = ["h", "v"]
    while len(orientations) < n_lines:
        orientations.append(rng.choice(["h", "v"]))
    rng.shuffle(orientations)

    for color, orientation in zip(colors, orientations):
        placed = False
        for _ in range(250):
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


def _place_sprite_to_line(
    rng: random.Random,
    line_id: int,
    lines: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    required_family: str | None = None,
) -> dict | None:
    line = lines[line_id]
    hit_cells = list(line["cells"])
    directions = _dirs_for_line(line["orientation"])
    if required_family is not None:
        directions = [d for d in directions if _direction_family(d) == required_family]
        if not directions:
            return None
    rng.shuffle(hit_cells)
    rng.shuffle(directions)

    for hit_cell in hit_cells:
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
                cell_set = frozenset(cells)
                if _halo(cells) & occupied:
                    continue
                if cell_set & reserved_bg:
                    continue
                path = _trace_path(apex, direction, steps)
                trail_mid = path[:-1]
                trail_mid_set = frozenset(trail_mid)
                if trail_mid_set & occupied:
                    continue
                if trail_mid_set & reserved_bg:
                    continue
                return {
                    "cells": cells,
                    "cell_set": cell_set,
                    "halo": _halo(cells),
                    "apex": apex,
                    "dir": direction,
                    "steps": steps,
                    "trail_mid": trail_mid,
                    "trail_mid_set": trail_mid_set,
                    "target_line_id": line_id,
                    "target_hit_cell": hit_cell,
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
    if any(dist not in RAY_STEPS for dist in stats["distances"]):
        return False
    if len(set(stats["distances"])) < 2:
        return False
    if stats["trail_cells_total"] < expected_sprites * 2:
        return False
    triangles, lines = _extract_objects(grid, BG)
    if {line["orientation"] for line in lines} != {"h", "v"}:
        return False
    if {_direction_family(tri["dir"]) for tri in triangles} != {"axis", "diag"}:
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
    assignments = list(target_ids)
    while len(assignments) < n_sprites:
        assignments.append(rng.choice(target_ids))
    rng.shuffle(assignments)
    families = ["axis", "diag"]
    while len(families) < n_sprites:
        families.append(rng.choice(["axis", "diag"]))
    rng.shuffle(families)

    occupied: set[tuple[int, int]] = set()
    for line in lines:
        occupied.update(line["cells"])
    reserved_bg: set[tuple[int, int]] = set()

    sprites: list[dict] = []
    for line_id, family in zip(assignments, families):
        sprite = _place_sprite_to_line(rng, line_id, lines, occupied, reserved_bg, required_family=family)
        if sprite is None:
            return None
        sprites.append(sprite)
        occupied.update(sprite["cells"])
        reserved_bg.update(sprite["trail_mid"])

    inp = np.full((H, W), BG, dtype=int)
    for line in lines:
        _paint_cells(inp, line["cells"], line["color"])
    for sprite in sprites:
        _paint_cells(inp, sprite["cells"], sprite_color)

    if not _validate_input(inp, n_sprites, n_lines):
        return None

    out = main(inp)
    line_only = _render_line_hits_only(inp)
    stats = _analyze_instance(inp)

    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, line_only):
        return None
    if not np.any(out == EFFECT_COLOR):
        return None
    if stats["hit_line_count"] >= stats["line_count"]:
        return None
    if min(stats["distances"]) >= max(stats["distances"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
