# Combo variant of 8b9c3697 (index 19)
# combo_spec: D1/0+D2/1+D12/0
# Name: 全色彩路由系统

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
# - Base task: monochrome structures wait for pure-red blocks that happen to lie on the
#   correct opening ray; the matched red slides inward and leaves a black trail.
# - Fused task: every structure has two visible colors. Its wall/body color is unique to
#   that structure, while its teeth use one shared guide color. Every movable object is a
#   2-row sprite made of a route-color stripe plus a red stripe. Matching now ignores
#   spatial ray hits and instead pairs a sprite to the structure whose wall color matches
#   the sprite's route color.
# - Small trace:
#   1. A blue-wall comb and an orange-wall comb are both present.
#   2. In front of the blue comb, there are two sprites on the opening ray: a nearer
#      orange+red sprite and a farther blue+red sprite.
#   3. The original task would choose the nearer on-ray red object. The combo variant
#      does not: it reads the comb's blue wall color, selects the blue+red sprite, and
#      slides that entire sprite into the socket under the grey teeth.
#   4. Every intermediate placement keeps the sprite's own colors, so the trail is
#      multicolored instead of black. The nearer orange+red decoy disappears.
#
# b) Complexity calibration
# - Structures still use the original comb geometry, but teeth are restricted to 2 or 3
#   cells so both the teeth block and the 2-color sprite stripe stay clearly visible.
# - Each grid has 2 or 3 structures with distinct wall colors, providing repeated evidence
#   that the correspondence is color-based rather than positional.
# - Each structure gets a closer wrong-color sprite on the exact same opening ray. That
#   makes the D1 change human-visible and prevents accidental regression to the original
#   "nearest visible red" rule.
#
# c) Implicit-assumption review of main()
# - Original assumption "movable objects are pure red components" is invalid. The solver
#   now classifies connected components by whether they contain red (sprites) or not
#   (structures), and extracts the sprite's route color from its non-red stripe.
# - Original assumption "the hit ray identifies both the object and the slide direction"
#   is only half-valid now. The structure geometry still supplies the slide direction, but
#   the chosen object comes from wall-color equality, not from the visible ray.
# - Original assumption "trail color is black" is invalid. The fused rule replays the
#   sprite's full 2-color pattern at every intermediate placement.
# - New assumption: structures have exactly two non-background colors, with the rarer one
#   being the teeth color. The generator enforces that by construction.
#
# d) Placement legality
# - Generation stays output-first. Each local patch contains one structure, its farther
#   true sprite, and its nearer wrong-color decoy, so the geometry is internally legal
#   before global placement.
# - Rotated patches are placed with padding on a fresh canvas, and extra distractor sprites
#   are kept off all structure rays with a 1-cell buffer, preventing overlaps or merged
#   components.
# - All matched sprites share the same socket dimensions as their target teeth block, so
#   docking requires only an axis-aligned translation with no clipping or partial overlap.
#
# e) Visual observability
# - The rule is visible from pixels alone: the shared cue is the repeated wall-color /
#   sprite-stripe color pairing. Color ids are not used numerically; only equality matters.
# - The closer wrong-color ray decoys make it clear that the output is not choosing by
#   distance or by "first thing on the corridor". A human can see one sprite being ignored
#   while a farther same-color sprite moves.
# - The trail also changes in an obvious way: it inherits the sprite's colors instead of
#   turning black, so the combo does not look like a minor recolor of the base task.
#
# f) Visual cleanliness
# - Structures remain separated, and each ray corridor carries at most one true sprite and
#   one false sprite before any global distractors are added.
# - All distractors use the same 2-row sprite template as the matched objects, avoiding a
#   mixture of unrelated glyph styles.
# - Teeth keep one shared guide color across the grid, so the eye can quickly isolate the
#   docking zone without having to learn several independent tooth-color rules.
#
# g) Degeneracy checks enforced by the generator
# - `main(inp)` must exactly reproduce the intended output.
# - Every generated structure must have a nearer wrong-color on-ray sprite and a farther
#   same-shape sprite with the correct route color.
# - Each structure wall color must appear on exactly one matched sprite and on no other
#   matched target, keeping the routing one-to-one and inferable.
#
# h) New solver assumptions audited against generation
# - The solver infers a structure's route color as its majority color and its teeth as the
#   minority color; the generator guarantees a strong count gap.
# - The solver aligns sprites by their route-color stripe bbox, so matched sprites are
#   always generated with a stripe shape identical to the tooth span.
# - The solver expects sprites to move only along one axis; the generator preserves the
#   original straight-corridor slide model under rotation.


RED = int(Color.RED)
BLACK = int(Color.BLACK)
COMMON_TOOTH_COLOR = int(Color.GREY)
BG_COLORS = (int(Color.GREEN), int(Color.YELLOW))
ROUTE_COLOR_POOL = (
    int(Color.BLUE),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _component_colors(g: np.ndarray, coords: np.ndarray | list[tuple[int, int]]) -> set[int]:
    return {int(g[int(r), int(c)]) for r, c in coords}


def _piece_route_color(g: np.ndarray, coords: np.ndarray, bg: int) -> int | None:
    colors = _component_colors(g, coords) - {int(bg), int(RED), int(BLACK)}
    if len(colors) != 1:
        return None
    return next(iter(colors))


def _piece_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    pts = [(int(r), int(c)) for r, c in coords]
    route_color = _piece_route_color(g, coords, bg)
    info = {
        "coords": pts,
        "bbox": tuple(map(int, bbox_coords(coords))),
        "route_color": None if route_color is None else int(route_color),
        "route_bbox": None,
    }
    if route_color is not None:
        route_pts = np.array([(r, c) for r, c in pts if int(g[r, c]) == int(route_color)], dtype=int)
        info["route_bbox"] = tuple(map(int, bbox_coords(route_pts)))
    return info


def _structure_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    pts = [(int(r), int(c)) for r, c in coords]
    colors = _component_colors(g, coords) - {int(bg), int(BLACK), int(RED)}
    if len(colors) != 2:
        return None

    counts = Counter(int(g[r, c]) for r, c in pts)
    tooth_color = min(counts.items(), key=lambda kv: int(kv[1]))[0]
    wall_color = max(counts.items(), key=lambda kv: int(kv[1]))[0]
    tooth_cells = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == int(tooth_color)]
    if not tooth_cells:
        return None

    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    dir_hits: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in tooth_cells:
        for dr, dc in _DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                continue
            if int(g[nr, nc]) != int(bg):
                continue
            if not (r0 < nr < r1 and c0 < nc < c1):
                continue
            dir_hits.setdefault((int(dr), int(dc)), []).append((int(r), int(c)))

    if not dir_hits:
        return None

    direction = max(dir_hits.keys(), key=lambda d: len(dir_hits[d]))
    tooth_bbox = tuple(map(int, bbox_coords(np.array(tooth_cells, dtype=int))))
    dr, dc = direction
    target_route_bbox = (
        int(tooth_bbox[0] + dr),
        int(tooth_bbox[1] + dc),
        int(tooth_bbox[2] + dr),
        int(tooth_bbox[3] + dc),
    )
    return {
        "coords": set(pts),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "wall_color": int(wall_color),
        "tooth_color": int(tooth_color),
        "tooth_bbox": tooth_bbox,
        "direction": (int(dr), int(dc)),
        "target_route_bbox": target_route_bbox,
    }


def _paint_sprite(out: np.ndarray, sprite_cells: list[tuple[int, int, int]]) -> None:
    h, w = out.shape
    for r, c, color in sprite_cells:
        if 0 <= int(r) < h and 0 <= int(c) < w:
            out[int(r), int(c)] = int(color)


def _shift_sprite_cells(sprite_cells: list[tuple[int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int]]:
    return [(int(r + dr), int(c + dc), int(color)) for r, c, color in sprite_cells]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route multicolor sprites into same-color comb structures, leaving a color-preserving trail.

    Input structure:
    - The background is the most common color.
    - Each comb-like structure uses a unique wall color plus one shared tooth color.
    - Each movable object is a 2-color sprite: one stripe is red and the other stripe is a route color.

    Rule:
    - For every structure, read its opening direction from the tooth geometry and read its target color from the wall color.
    - Find the unique sprite whose non-red route stripe has that same wall color.
    - Slide that full sprite straight into the socket under the teeth, repainting every intermediate placement with the sprite's own colors; sprites with no matching structure color are erased.

    Key insight: Geometry still determines where a matched sprite docks, but color equality between the structure body and the sprite's route stripe determines which sprite moves, even when a closer wrong-color sprite sits on the same ray.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    assert int(RED) in set(map(int, np.unique(g)))

    comps = components_mask(g != int(bg), connectivity=4)
    assert comps

    pieces: list[dict] = []
    structures: list[dict] = []
    for coords in comps:
        colors = _component_colors(g, coords)
        if int(RED) in colors:
            pieces.append(_piece_info(g, coords, bg))
        else:
            info = _structure_info(g, coords, bg)
            if info is not None:
                structures.append(info)

    assert pieces, "expected at least one multicolor sprite"
    assert structures, "expected at least one multicolor comb structure"

    pieces_by_route: dict[int, list[int]] = {}
    for pid, piece in enumerate(pieces):
        route_color = piece["route_color"]
        if route_color is None:
            continue
        pieces_by_route.setdefault(int(route_color), []).append(int(pid))

    assignments: list[tuple[int, dict]] = []
    for struct in structures:
        candidates = pieces_by_route.get(int(struct["wall_color"]), [])
        if len(candidates) != 1:
            continue
        pid = int(candidates[0])
        if pieces[pid]["route_bbox"] is None:
            continue
        assignments.append((pid, struct))

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)

    used_pids: set[int] = set()
    for pid, struct in assignments:
        if pid in used_pids:
            continue
        used_pids.add(pid)

        piece = pieces[pid]
        route_bbox = piece["route_bbox"]
        assert route_bbox is not None
        tr0, tc0, _tr1, _tc1 = map(int, struct["target_route_bbox"])
        pr0, pc0, _pr1, _pc1 = map(int, route_bbox)
        shift_r = int(tr0 - pr0)
        shift_c = int(tc0 - pc0)
        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = 0 if int(shift_r) == 0 else (1 if int(shift_r) > 0 else -1)
        step_dc = 0 if int(shift_c) == 0 else (1 if int(shift_c) > 0 else -1)
        sprite_cells = [(int(r), int(c), int(g[int(r), int(c)])) for r, c in piece["coords"]]

        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            _paint_sprite(out, _shift_sprite_cells(sprite_cells, off_r, off_c))

        _paint_sprite(out, _shift_sprite_cells(sprite_cells, shift_r, shift_c))

    return out


# --------------- generator (codex-att2 combo variant) ---------------

GRID_H_RANGE = (20, 25)
GRID_W_RANGE = (24, 30)

STRUCT_WIDTH_TEETH = ((8, 2), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
TRUE_SLIDE_RANGE = (5, 8)
FALSE_SLIDE_MIN = 2
STRUCT_PADDING = 4

TOTAL_PIECES_MIN = 5
TOTAL_PIECES_MAX = 11
DISTRACTOR_WIDTHS = (2, 3)

MAX_RETRIES = 500


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, wall_color: int, tooth_color: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = int(wall_color)
    g[1, 1] = g[1, width - 2] = int(wall_color)
    g[2:4, 1] = g[2:4, width - 2] = int(wall_color)
    g[1, start : start + teeth] = int(tooth_color)
    return g


def _piece_array(width: int, bg: int, route_color: int) -> np.ndarray:
    piece = np.full((2, width), bg, dtype=int)
    piece[0, :] = int(route_color)
    piece[1, :] = int(RED)
    return piece


def _paint_slide_patch(patch: np.ndarray, piece: np.ndarray, start_r: int, start_c: int, end_r: int, end_c: int, bg: int) -> None:
    shift_r = int(end_r - start_r)
    shift_c = int(end_c - start_c)
    steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
    step_dr = 0 if int(shift_r) == 0 else (1 if int(shift_r) > 0 else -1)
    step_dc = 0 if int(shift_c) == 0 else (1 if int(shift_c) > 0 else -1)

    for step in range(0, steps):
        rr = int(start_r + step * step_dr)
        cc = int(start_c + step * step_dc)
        paste(patch, piece, rr, cc, transparent=bg)
    paste(patch, piece, end_r, end_c, transparent=bg)


def _matched_patches(
    width: int,
    teeth: int,
    true_slide: int,
    false_slide: int,
    rot_k: int,
    bg: int,
    wall_color: int,
    tooth_color: int,
    fake_route_color: int,
) -> dict:
    base = _structure_array(width, teeth, bg, wall_color, tooth_color)
    true_piece = _piece_array(teeth, bg, wall_color)
    fake_piece = _piece_array(teeth, bg, fake_route_color)
    start = (width - teeth) // 2
    patch_h = 2 + true_slide + true_piece.shape[0]

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)

    input_patch = structure.copy()
    output_patch = structure.copy()

    paste(input_patch, true_piece, 2 + true_slide, start, transparent=bg)
    paste(input_patch, fake_piece, 2 + false_slide, start, transparent=bg)
    _paint_slide_patch(output_patch, true_piece, 2 + true_slide, start, 2, start, bg)

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "wall_color": int(wall_color),
        "tooth_color": int(tooth_color),
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(
    specs: list[dict],
    h: int,
    w: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] != bg)
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        tooth_arr = np.argwhere(spec["structure"] == int(spec["tooth_color"]))
        tooth_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in tooth_arr]
        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "tooth_coords": tooth_coords,
                "tooth_color": int(spec["tooth_color"]),
            }
        )

    return inp, out, struct_grid, placed


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for r, c in item["tooth_coords"]:
            for dr, dc in _DIRS:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(struct_grid[nr, nc]) != int(bg):
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                    forbidden[rr, cc] = True
                    rr += int(dr)
                    cc += int(dc)
    return forbidden


def _parse_input_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    structures: list[dict] = []
    pieces: list[dict] = []
    for coords in components_mask(grid != int(bg), connectivity=4):
        colors = _component_colors(grid, coords)
        if int(RED) in colors:
            pieces.append(_piece_info(grid, coords, bg))
        else:
            info = _structure_info(grid, coords, bg)
            if info is not None:
                structures.append(info)
    return structures, pieces


def _valid_color_mapping(grid: np.ndarray, bg: int, expected_structs: int) -> bool:
    structures, pieces = _parse_input_objects(grid, bg)
    if len(structures) != expected_structs:
        return False

    wall_colors = [int(s["wall_color"]) for s in structures]
    if len(set(wall_colors)) != len(wall_colors):
        return False

    route_counts: Counter[int] = Counter()
    for piece in pieces:
        if piece["route_color"] is not None:
            route_counts[int(piece["route_color"])] += 1

    for wall_color in wall_colors:
        if int(route_counts.get(int(wall_color), 0)) != 1:
            return False
    return True


def _sprite_patch(route_color: int, width: int, bg: int) -> np.ndarray:
    return _piece_array(width, bg, route_color)


def _place_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    count: int,
    blocked_route_colors: set[int],
) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    piece_mask = inp != bg

    route_choices = [
        c
        for c in ROUTE_COLOR_POOL
        if c not in (int(bg), int(RED), int(BLACK), int(COMMON_TOOTH_COLOR)) and c not in blocked_route_colors
    ]
    if count > 0 and not route_choices:
        return False
    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(piece_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = int(COMMON_TOOTH_COLOR)
        random.shuffle(route_choices)
        widths = list(DISTRACTOR_WIDTHS)
        random.shuffle(widths)
        placed_one = False
        for route_color in route_choices:
            for width in widths:
                patch = _sprite_patch(route_color, width, bg)
                loc = _place_patch(canvas, patch, bg, padding=0)
                if loc is None:
                    continue
                paste(inp, patch, loc[0], loc[1], transparent=bg)
                piece_mask[loc[0] : loc[0] + patch.shape[0], loc[1] : loc[1] + patch.shape[1]] = True
                placed_one = True
                break
            if placed_one:
                break
        if not placed_one:
            return False
    return True


def _is_on_ray(piece_bbox: tuple[int, int, int, int] | None, target_bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> int | None:
    if piece_bbox is None:
        return None
    pr0, pc0, pr1, pc1 = map(int, piece_bbox)
    tr0, tc0, tr1, tc1 = map(int, target_bbox)
    dr, dc = map(int, direction)

    if (pr1 - pr0) != (tr1 - tr0) or (pc1 - pc0) != (tc1 - tc0):
        return None

    if dr == 1 and dc == 0 and pc0 == tc0 and pc1 == tc1 and pr0 > tr0:
        return int(pr0 - tr0)
    if dr == -1 and dc == 0 and pc0 == tc0 and pc1 == tc1 and pr0 < tr0:
        return int(tr0 - pr0)
    if dr == 0 and dc == 1 and pr0 == tr0 and pr1 == tr1 and pc0 > tc0:
        return int(pc0 - tc0)
    if dr == 0 and dc == -1 and pr0 == tr0 and pr1 == tr1 and pc0 < tc0:
        return int(tc0 - pc0)
    return None


def _fusion_metric(grid: np.ndarray, bg: int) -> int:
    structures, pieces = _parse_input_objects(grid, bg)
    score = 0
    for struct in structures:
        ray_pieces: list[tuple[int, int]] = []
        for pid, piece in enumerate(pieces):
            dist = _is_on_ray(piece["route_bbox"], struct["target_route_bbox"], struct["direction"])
            if dist is not None:
                ray_pieces.append((int(dist), int(pid)))
        if not ray_pieces:
            continue
        ray_pieces.sort()
        nearest_piece = pieces[ray_pieces[0][1]]
        farther_match = any(
            int(pieces[pid]["route_color"]) == int(struct["wall_color"]) and int(dist) > int(ray_pieces[0][0])
            for dist, pid in ray_pieces
            if pieces[pid]["route_color"] is not None
        )
        if nearest_piece["route_color"] != int(struct["wall_color"]) and farther_match:
            score += 1
    return score


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    route_pool = [c for c in ROUTE_COLOR_POOL if c not in (int(bg), int(RED), int(BLACK), int(COMMON_TOOTH_COLOR))]
    # Three-structure layouts are geometrically harder to place, so bias attempts toward
    # them to keep accepted samples closer to balanced branch coverage.
    n_structs = 3 if random.random() < 0.8 else 2
    if len(route_pool) < 2 * n_structs:
        return None

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    random.shuffle(route_pool)
    wall_colors = route_pool[:n_structs]
    fake_colors = route_pool[n_structs : 2 * n_structs]

    base_pieces = 2 * n_structs
    extra_min = max(0, TOTAL_PIECES_MIN - base_pieces)
    extra_max = max(0, TOTAL_PIECES_MAX - base_pieces)
    if extra_min > extra_max:
        return None
    distractor_count = random.randint(extra_min, extra_max)

    specs = []
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        true_slide = random.randint(*TRUE_SLIDE_RANGE)
        false_slide = random.randint(FALSE_SLIDE_MIN, true_slide - 3)
        specs.append(
            _matched_patches(
                width,
                teeth,
                true_slide,
                false_slide,
                rot_k,
                bg,
                wall_colors[idx],
                COMMON_TOOTH_COLOR,
                fake_colors[idx],
            )
        )

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    placed = _place_structures(specs, h, w, bg)
    if placed is None:
        return None
    inp, out, struct_grid, placed_items = placed

    if not _valid_color_mapping(inp, bg, n_structs):
        return None

    if _fusion_metric(inp, bg) != n_structs:
        return None

    if not _place_distractors(inp, struct_grid, placed_items, bg, distractor_count, set(map(int, wall_colors))):
        return None

    if not _valid_color_mapping(inp, bg, n_structs):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if _fusion_metric(inp, bg) != n_structs:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant 19")
