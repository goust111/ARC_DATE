# Combo variant of 8b9c3697 (index 15)
# combo_spec: D1/0+D10/0+D11/1
# Name: 颜色路由多对一标记干扰

from __future__ import annotations

# Design analysis for this combo variant
# --------------------------------------
# 1. Concrete visual change
#    - The original comb/bracket geometry is kept, but each structure now has two colors:
#      one shared wall color and one tooth color that acts as a routing key.
#    - Each movable object is a two-cell-deep red strip whose contact face is recolored to
#      that routing color. Several strips may share one routing color.
#    - A structure may now have 1-3 stacked interior bays. Same-color strips enter the same
#      structure and end up stacked in those bays from the tooth side outward.
#    - Distractors are no longer red. They are small monochrome blocks in one special color,
#      clearly separated from both the structures and the red moving pieces, and they remain
#      unchanged in the output.
#
# 2. Behavior trace on the changed step
#    - Original: each structure ray-casts outward from its socket and chooses one visible red
#      component, so matching is 1:1 and driven by alignment.
#    - Here: a structure first reads its tooth color and its opening direction from geometry.
#      Then it collects every red piece whose marker stripe has that same color. Those pieces
#      are sorted by distance to the structure and inserted in that order.
#    - Example with a downward-opening blue-tooth structure and three blue-marked pieces:
#      the nearest blue-marked piece takes the innermost bay, the next one stops in the middle
#      bay, and the farthest one stops in the outer bay. A yellow block nearby does nothing.
#      The contrast to the original task is visible because one structure receives multiple
#      red pieces, and the yellow distractor would have been irrelevant even if placed nearby.
#
# 3. Complexity calibration
#    - Tooth colors are unique per structure, which avoids ambiguous many-to-many routing while
#      still making repeated same-color pieces necessary for the multi-slot behavior.
#    - Teeth counts are restricted to 2 or 3 so the colored face is always easy to see.
#    - Slot counts are restricted to 1-3, and at least one structure in each generated grid has
#      2 or 3 slots, so the many-to-one fusion is present in every example without turning the
#      scene into a dense unreadable pileup.
#
# 4. Old implicit assumptions that no longer hold
#    - The original solver assumed every movable object was pure red and every non-red component
#      was a structure. That fails here because valid movers contain both red and a marker color,
#      while distractors are non-red objects that are not structures. The new solver therefore
#      classifies connected components by whether they contain red and by whether their geometry
#      supports a comb-like tooth/cavity interpretation.
#    - The original solver assumed one matched red per structure. That fails here because one
#      tooth color may appear on several pieces, and they all belong to the same structure.
#    - The original solver inferred the match itself from ray hits. That fails here because the
#      ray is only used to infer the opening direction and slot stack; color performs the match.
#
# 5. Placement legality and cleanliness
#    - Generation stays output-first: each local patch already contains the final docked pieces
#      and black trails, then the input is derived by translating those pieces outward.
#    - Each structure patch includes its full trail extent, so patch placement on the global
#      canvas guarantees non-overlap in both the input and the output.
#    - Special-color distractors are placed only on cells that are background in both input and
#      output, so they do not block motion paths or get partially overwritten by trails.
#
# 6. Visual observability
#    - A solver looking only for red-on-ray matches cannot explain these examples well because
#      one structure may absorb several same-color pieces while another of different tooth color
#      absorbs a different color family.
#    - The yellow distractors stay fixed, making the color-category split visible: red-with-tag
#      objects move, special-color objects do not.
#    - Distance order is human-visible because pieces on the same lane start separated in depth
#      and end up stacked from the tooth side outward.
#
# 7. New solver assumptions and matching generator guarantees
#    - Every valid moving piece has exactly one non-red marker color, and that color is equal to
#      exactly one structure tooth color in the grid.
#    - For each structure, the number of same-color pieces equals the number of interior bays
#      detected from the cavity depth.
#    - Same-color pieces assigned to one structure are pre-aligned with that structure's opening
#      axis and have strictly increasing distances, so the distance sort is deterministic.

from collections import Counter
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


RED = int(Color.RED)
BLACK = int(Color.BLACK)
PIECE_DEPTH = 2
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _component_colors(g: np.ndarray, coords: np.ndarray) -> set[int]:
    return {int(g[int(r), int(c)]) for r, c in coords}


def _piece_marker_color(g: np.ndarray, coords: np.ndarray, bg: int) -> int | None:
    cols = _component_colors(g, coords) - {int(bg), int(RED), int(BLACK)}
    if len(cols) != 1:
        return None
    return int(next(iter(cols)))


def _piece_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    pts = [(int(r), int(c)) for r, c in coords]
    marker = _piece_marker_color(g, coords, bg)
    if marker is None:
        return None
    marker_pts = np.array([(r, c) for r, c in pts if int(g[r, c]) == int(marker)], dtype=int)
    if marker_pts.size == 0:
        return None
    return {
        "coords": pts,
        "bbox": tuple(map(int, bbox_coords(coords))),
        "marker_color": int(marker),
        "marker_bbox": tuple(map(int, bbox_coords(marker_pts))),
    }


def _shift_bbox(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    return (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc))


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = map(int, inner)
    or0, oc0, or1, oc1 = map(int, outer)
    return or0 <= ir0 <= ir1 <= or1 and oc0 <= ic0 <= ic1 <= oc1


def _bbox_all_color(g: np.ndarray, box: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = map(int, box)
    return bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _structure_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    pts = [(int(r), int(c)) for r, c in coords]
    counts = Counter(int(g[r, c]) for r, c in pts)
    if len(counts) < 2:
        return None

    tooth_color = min(counts.items(), key=lambda kv: int(kv[1]))[0]
    tooth_cells = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == int(tooth_color)]
    if not tooth_cells:
        return None

    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    tooth_bbox = tuple(map(int, bbox_coords(np.array(tooth_cells, dtype=int))))

    dir_hits: dict[tuple[int, int], int] = {}
    for r, c in tooth_cells:
        for dr, dc in _DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                continue
            if int(g[nr, nc]) != int(bg):
                continue
            if not (r0 < nr < r1 and c0 < nc < c1):
                continue
            dir_hits[(int(dr), int(dc))] = int(dir_hits.get((int(dr), int(dc)), 0) + 1)

    best: tuple[int, int, tuple[int, int], list[tuple[int, int, int, int]]] | None = None
    for direction, hit_count in dir_hits.items():
        dr, dc = map(int, direction)
        slot_targets: list[tuple[int, int, int, int]] = []
        offset = 1
        while True:
            marker_box = _shift_bbox(tooth_bbox, int(dr * offset), int(dc * offset))
            red_box = _shift_bbox(tooth_bbox, int(dr * (offset + 1)), int(dc * (offset + 1)))
            if not (_bbox_inside(marker_box, (r0, c0, r1, c1)) and _bbox_inside(red_box, (r0, c0, r1, c1))):
                break
            if not (_bbox_all_color(g, marker_box, bg) and _bbox_all_color(g, red_box, bg)):
                break
            slot_targets.append(marker_box)
            offset += PIECE_DEPTH

        cand = (int(len(slot_targets)), int(hit_count), (int(dr), int(dc)), slot_targets)
        if best is None or cand > best:
            best = cand

    if best is None:
        return None

    _slot_count, _hit_count, direction, slot_targets = best
    dr, dc = map(int, direction)

    if not slot_targets:
        return None

    return {
        "coords": set(pts),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "tooth_color": int(tooth_color),
        "direction": (int(dr), int(dc)),
        "tooth_bbox": tuple(map(int, tooth_bbox)),
        "slot_targets": slot_targets,
    }


def _projected_distance(piece: dict, struct: dict) -> int:
    mr0, mc0, _mr1, _mc1 = map(int, piece["marker_bbox"])
    tr0, tc0, _tr1, _tc1 = map(int, struct["tooth_bbox"])
    dr, dc = map(int, struct["direction"])
    return abs(int((mr0 - tr0) * dr + (mc0 - tc0) * dc))


def _shifted_offset(step: int, shift_r: int, shift_c: int, steps: int) -> tuple[int, int]:
    if steps == 0:
        return 0, 0
    return int(round(step * shift_r / steps)), int(round(step * shift_c / steps))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route color-tagged red strips into tooth-matched stacked bays, leaving black slide trails.

    Input structure:
    - The background is the most frequent color.
    - Each large comb-like structure uses one wall color and one smaller tooth color.
    - Movable pieces are red strips with a same-width colored contact face; small special-color blocks are inert distractors.

    Rule:
    - Read each structure's tooth color and opening direction from its geometry.
    - Match every red piece whose colored face has that tooth color to that structure.
    - Sort those same-color pieces by distance to the structure and slide them into the structure's bays from the tooth side outward.
    - Paint every intermediate placement black; keep the special-color distractors unchanged.

    Key insight:
    - Matching is controlled by color categories, not by ray-hit alignment, and one structure may absorb several same-color pieces
      in distance order.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    assert int(RED) in set(map(int, np.unique(g)))

    comps = components_mask(g != int(bg), connectivity=4)
    assert comps

    pieces: list[dict] = []
    structures: list[dict] = []
    for coords in comps:
        cols = _component_colors(g, coords)
        if int(RED) in cols:
            info = _piece_info(g, coords, bg)
            if info is not None:
                pieces.append(info)
        else:
            info = _structure_info(g, coords, bg)
            if info is not None:
                structures.append(info)

    assert pieces
    assert structures, "expected at least one colored multi-slot structure"

    pieces_by_marker: dict[int, list[dict]] = {}
    for piece in pieces:
        pieces_by_marker.setdefault(int(piece["marker_color"]), []).append(piece)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)

    for struct in structures:
        candidates = pieces_by_marker.get(int(struct["tooth_color"]), [])
        if not candidates:
            continue
        ordered = sorted(
            candidates,
            key=lambda piece: (int(_projected_distance(piece, struct)), tuple(map(int, piece["bbox"]))),
        )
        for slot_idx, piece in enumerate(ordered[: len(struct["slot_targets"])]):
            target_box = struct["slot_targets"][slot_idx]
            tr0, tc0, _tr1, _tc1 = map(int, target_box)
            mr0, mc0, _mr1, _mc1 = map(int, piece["marker_bbox"])
            shift_r = int(tr0 - mr0)
            shift_c = int(tc0 - mc0)
            steps = int(max(abs(int(shift_r)), abs(int(shift_c))))

            for step in range(0, steps):
                off_r, off_c = _shifted_offset(step, shift_r, shift_c, steps)
                for r, c in piece["coords"]:
                    rr = int(r + off_r)
                    cc = int(c + off_c)
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr, cc] = int(BLACK)

            for r, c in piece["coords"]:
                rr = int(r + shift_r)
                cc = int(c + shift_c)
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator (codex-att2 combo D1/0+D10/0+D11/1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────────────
GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (28, 36)
BG_COLORS = (3, 4)
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)

# ── CFG: structures ─────────────────────────────────────────────────────────
STRUCT_WIDTH_TEETH = ((8, 2), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
SLOT_COUNT_CHOICES = (1, 2, 3)
STRUCT_PADDING = 3
SLIDE_PATTERNS = {
    1: ((4,), (5,), (6,), (7,)),
    2: ((4, 6), (5, 7), (6, 8)),
    3: ((6, 7, 8), (6, 8, 9), (7, 8, 9)),
}

# ── CFG: distractors ────────────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
DISTRACTOR_COUNT_RANGE = (2, 4)

# ── CFG: retry ──────────────────────────────────────────────────────────────
MAX_RETRIES = 500
PROFILE_CYCLE = (
    (2, (2, 1)),
    (2, (3, 1)),
    (2, (3, 2)),
    (3, (2, 1, 1)),
    (3, (2, 2, 1)),
    (3, (3, 2, 1)),
)
_PROFILE_CURSOR = 0


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, slots: int, bg: int, wall_color: int, tooth_color: int) -> np.ndarray:
    h = int(2 + PIECE_DEPTH * slots)
    g = np.full((h, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = int(wall_color)
    g[1, 1] = g[1, width - 2] = int(wall_color)
    g[2:, 1] = g[2:, width - 2] = int(wall_color)
    g[1, start : start + teeth] = int(tooth_color)
    return g


def _piece_array(teeth: int, bg: int, marker_color: int) -> np.ndarray:
    piece = np.full((PIECE_DEPTH, teeth), bg, dtype=int)
    piece[0, :] = int(marker_color)
    piece[1, :] = int(RED)
    return piece


def _matched_patches(
    width: int,
    teeth: int,
    slots: int,
    slides: tuple[int, ...],
    rot_k: int,
    bg: int,
    wall_color: int,
    tooth_color: int,
) -> dict:
    structure = _structure_array(width, teeth, slots, bg, wall_color, tooth_color)
    piece = _piece_array(teeth, bg, tooth_color)
    start = (width - teeth) // 2
    target_rows = [int(2 + PIECE_DEPTH * idx) for idx in range(slots)]
    patch_h = int(target_rows[-1] + slides[-1] + PIECE_DEPTH)

    structure_patch = np.full((patch_h, width), bg, dtype=int)
    paste(structure_patch, structure, 0, 0, transparent=bg)
    input_patch = structure_patch.copy()
    output_patch = structure_patch.copy()

    for slot_idx, slide in enumerate(slides):
        target_r = int(target_rows[slot_idx])
        start_r = int(target_r + slide)
        paste(input_patch, piece, start_r, start, transparent=bg)
        for off in range(1, int(slide) + 1):
            output_patch[target_r + off : target_r + off + PIECE_DEPTH, start : start + teeth] = int(BLACK)

    for target_r in target_rows:
        paste(output_patch, piece, int(target_r), start, transparent=bg)

    return {
        "structure": np.rot90(structure_patch, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "tooth_color": int(tooth_color),
        "slot_count": int(slots),
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _parse_input_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    structures: list[dict] = []
    pieces: list[dict] = []
    for coords in components_mask(grid != int(bg), connectivity=4):
        cols = _component_colors(grid, coords)
        if int(RED) in cols:
            info = _piece_info(grid, coords, bg)
            if info is not None:
                pieces.append(info)
        else:
            info = _structure_info(grid, coords, bg)
            if info is not None:
                structures.append(info)
    return structures, pieces


def _valid_mapping(
    grid: np.ndarray,
    bg: int,
    expected_slots_by_color: dict[int, int],
    expected_structs: int,
) -> bool:
    structures, pieces = _parse_input_objects(grid, bg)
    if len(structures) != expected_structs:
        return False

    if len(pieces) != sum(expected_slots_by_color.values()):
        return False

    tooth_colors = [int(s["tooth_color"]) for s in structures]
    if len(set(tooth_colors)) != len(tooth_colors):
        return False

    marker_count: Counter[int] = Counter()
    for piece in pieces:
        marker = piece["marker_color"]
        if marker is None:
            return False
        marker_count[int(marker)] += 1

    for struct in structures:
        tooth_color = int(struct["tooth_color"])
        expected_slots = int(expected_slots_by_color.get(tooth_color, -1))
        if expected_slots <= 0:
            return False
        if len(struct["slot_targets"]) != expected_slots:
            return False
        if int(marker_count.get(tooth_color, 0)) != expected_slots:
            return False

    return True


def _place_structures(
    specs: list[dict],
    h: int,
    w: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    expected_slots_by_color: dict[int, int] = {}

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        expected_slots_by_color[int(spec["tooth_color"])] = int(spec["slot_count"])

    return inp, out, expected_slots_by_color


def _place_distractors(
    inp: np.ndarray,
    out: np.ndarray,
    bg: int,
    block_color: int,
    distractor_color: int,
    count: int,
) -> bool:
    h, w = inp.shape
    occ = (inp != bg) | (out != bg)
    occ = _dilate4(occ)
    shapes = list(DISTRACTOR_SHAPES)

    for _ in range(count):
        canvas = np.full((h, w), bg, dtype=int)
        canvas[occ] = int(block_color)
        random.shuffle(shapes)
        placed = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(distractor_color), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            paste(inp, patch, loc[0], loc[1], transparent=bg)
            paste(out, patch, loc[0], loc[1], transparent=bg)
            occ[loc[0] : loc[0] + rh, loc[1] : loc[1] + rw] = True
            occ = _dilate4(occ)
            placed = True
            break
        if not placed:
            return False
    return True


def _choose_slot_counts(n_structs: int) -> list[int]:
    counts = [random.choice(SLOT_COUNT_CHOICES) for _ in range(n_structs)]
    if all(int(v) == 1 for v in counts):
        counts[random.randrange(n_structs)] = random.choice((2, 3))
    return counts


def _next_profile() -> tuple[int, tuple[int, ...]]:
    global _PROFILE_CURSOR
    profile = PROFILE_CYCLE[_PROFILE_CURSOR % len(PROFILE_CYCLE)]
    _PROFILE_CURSOR += 1
    return int(profile[0]), tuple(int(v) for v in profile[1])


def _gen_instance(
    n_structs: int | None = None,
    slot_counts: tuple[int, ...] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    bg = random.choice(BG_COLORS)
    if n_structs is None:
        n_structs = random.randint(*STRUCT_COUNT_RANGE)
    else:
        n_structs = int(n_structs)

    if slot_counts is None:
        slot_counts = _choose_slot_counts(n_structs)
    else:
        slot_counts = [int(v) for v in slot_counts]
        if len(slot_counts) != n_structs:
            return None

    total_needed_colors = int(n_structs + 2)
    palette = [c for c in PALETTE if c not in (bg, RED, BLACK)]
    if len(palette) < total_needed_colors:
        return None
    random.shuffle(palette)

    wall_color = int(palette[0])
    distractor_color = int(palette[1])
    tooth_colors = [int(c) for c in palette[2 : 2 + n_structs]]
    if len(set(tooth_colors)) != n_structs:
        return None

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    specs = []
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        slots = int(slot_counts[idx])
        slides = random.choice(SLIDE_PATTERNS[slots])
        rot_k = random.randrange(4)
        specs.append(
            _matched_patches(
                width,
                teeth,
                slots,
                slides,
                rot_k,
                bg,
                wall_color,
                tooth_colors[idx],
            )
        )

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    placed = _place_structures(specs, h, w, bg)
    if placed is None:
        return None
    inp, out, expected_slots_by_color = placed

    if not _valid_mapping(inp, bg, expected_slots_by_color, n_structs):
        return None

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    if not _place_distractors(inp, out, bg, wall_color, distractor_color, distractor_count):
        return None

    if not np.array_equal(main(inp), out):
        return None

    meta = {
        "n_structs": int(n_structs),
        "slot_counts": tuple(int(v) for v in slot_counts),
        "distractor_color": int(distractor_color),
        "expected_slots_by_color": dict(expected_slots_by_color),
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    attempts_per_profile = max(1, MAX_RETRIES // len(PROFILE_CYCLE))
    visited_profiles = 0
    while visited_profiles < len(PROFILE_CYCLE):
        n_structs, slot_counts = _next_profile()
        for _ in range(attempts_per_profile):
            instance = _gen_instance(n_structs=n_structs, slot_counts=slot_counts)
            if instance is not None:
                return instance[0]
        visited_profiles += 1

    for _ in range(attempts_per_profile):
        instance = _gen_instance()
        if instance is not None:
            return instance[0]
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo D1/0+D10/0+D11/1")
