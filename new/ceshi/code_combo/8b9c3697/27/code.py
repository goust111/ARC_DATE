# Combo variant of 8b9c3697 (index 27)
# combo_spec: D10/1+D13/2
# Name: 链式频率消歧

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# =====================================
# spec_variant:
#   visible_layout:
#     - Keep the original comb/bracket structures, but use three structures in a left-to-right chain.
#     - Warm-colored movable pieces come in exactly two visibly related shades: red and maroon.
#     - The first structure has a private majority-shade piece.
#     - The second and third structures each see two candidates on their opening side:
#       a farther majority-shade bar and a nearer minority-shade bar, both with the same hit count.
#   concrete_behavior_trace:
#     - Baseline geometry-only logic:
#       1. For an ambiguous structure, the nearer bar wins.
#       2. Therefore both the second and third structures would choose the nearer minority-shade bars.
#     - Static frequency-only logic:
#       1. Count shade frequencies once from the full grid.
#       2. The majority shade stays preferred forever, so both later structures would choose the farther majority bars.
#     - Fused combo logic:
#       1. Process structures strictly left to right.
#       2. Structure 1 consumes one majority-shade piece, reducing that shade's remaining count.
#       3. Structure 2 still sees majority as more frequent, so frequency overrides distance and it takes the farther majority bar.
#       4. That second consumption changes the remaining counts again.
#       5. Structure 3 now sees tied shade frequencies, so distance becomes decisive and it takes the nearer minority bar.
#     - This yields a visible `major -> major -> minor` chain that neither single change can reproduce.
#   complexity_calibration:
#     - Only two red-family shades are used, so humans can count them directly.
#     - Ambiguity is repeated twice with the same visual motif, which makes the tie-break rule observable.
#     - The generator fixes the three-structure chain pattern instead of introducing many free layouts;
#       this keeps the examples readable while still exposing the stateful interaction.
#   implicit_assumption_audit:
#     - Original code assumed movable pieces all share one exact red id; that no longer holds.
#     - Original code scored candidates once per structure and then greedily consumed them; here the
#       available shade frequencies must be recomputed after each earlier match.
#     - Original nearest-distance tie-break is no longer sufficient when same-score candidates use
#       different red-family shades.
#     - Therefore the solver must:
#       1. detect red-family components across multiple colors,
#       2. preserve each component's own shade in the output,
#       3. rescan each structure against the remaining pool,
#       4. use current shade counts as a tie-break before distance.
#   placement_legality:
#     - Structures are placed with fixed orientation templates and non-overlapping coordinates.
#     - Every candidate component stays in bounds and separated by at least one background cell,
#       so connected-component parsing remains stable.
#     - Later-structure candidates are placed on different rows, so both remain simultaneously visible
#       from separate socket rays instead of one occluding the other.
#   observability:
#     - The ordering cue is purely spatial: left to right by structure position.
#     - The shade cue is purely visual: red versus maroon component counts.
#     - No hidden ordering by color id is used; numeric ids only back a deterministic fallback after
#       visible spatial ordering has already been fixed by bbox sorting.
#   cleanliness:
#     - The three structures occupy separated vertical bands, avoiding ray collisions and visual clutter.
#     - Only one extra filler component is added, just enough to establish the initial majority shade.
#   anti_degeneracy_checks:
#     - The generator rejects any sample unless the combo solver picks shades `[major, major, minor]`.
#     - It also rejects any sample unless:
#       1. chain-only solving differs at the second structure, and
#       2. static-frequency solving differs at the third structure.


RED_VARIANTS = (int(Color.RED), int(Color.MAROON))
BLACK = int(Color.BLACK)

GRID_H_RANGE = (22, 23)
GRID_W_RANGE = (26, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 8)

WIDTH = 8
TEETH = 2
S1_TOP_CHOICES = (1, 2)
S2_TOP_CHOICES = (7, 8)
S3_TOP_CHOICES = (12, 13, 14)
S1_LEFT_CHOICES = (1, 2)
S2_LEFT_CHOICES = (12, 13)
S3_RIGHT_MARGIN = 5

S1_SLIDE_CHOICES = (5, 6)
FAR_GAP_CHOICES = (5, 6)
NEAR_GAP_CHOICES = (2, 3)

MAX_RETRIES = 200
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _rotate_dir(dr: int, dc: int, rot_k: int) -> tuple[int, int]:
    for _ in range(int(rot_k) % 4):
        dr, dc = -int(dc), int(dr)
    return int(dr), int(dc)


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (int(width) - int(teeth)) // 2
    g[0, :] = int(sc)
    g[1, 1] = int(sc)
    g[1, width - 2] = int(sc)
    g[2:4, 1] = int(sc)
    g[2:4, width - 2] = int(sc)
    g[1, start : start + teeth] = int(sc)
    return g


def _structure_spec(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    base = _structure_array(int(width), int(teeth), int(bg), int(sc))
    red_h = min(int(teeth), 2)
    start = (int(width) - int(teeth)) // 2

    dock = np.zeros_like(base, dtype=bool)
    dock[2 : 2 + red_h, start : start + teeth] = True

    patch = np.rot90(base, k=int(rot_k))
    dock_mask = np.rot90(dock, k=int(rot_k))
    dock_coords = [(int(r), int(c)) for r, c in np.argwhere(dock_mask)]
    dock_bbox = tuple(map(int, bbox_coords(np.argwhere(dock_mask))))

    return {
        "patch": patch,
        "shape": tuple(map(int, patch.shape)),
        "dir": _rotate_dir(1, 0, int(rot_k)),
        "dock_rel": dock_coords,
        "dock_bbox_rel": dock_bbox,
    }


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    h, w = grid.shape
    for r, c in coords:
        if 0 <= int(r) < h and 0 <= int(c) < w:
            grid[int(r), int(c)] = int(color)


def _apply_slide(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    shift_r: int,
    shift_c: int,
    color: int,
    black_val: int,
) -> None:
    h, w = out.shape
    steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
    step_dr = 0 if int(shift_r) == 0 else (1 if int(shift_r) > 0 else -1)
    step_dc = 0 if int(shift_c) == 0 else (1 if int(shift_c) > 0 else -1)

    for step in range(0, steps):
        off_r = int(step * step_dr)
        off_c = int(step * step_dc)
        for r, c in coords:
            rr = int(r + off_r)
            cc = int(c + off_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(black_val)

    for r, c in coords:
        rr = int(r + shift_r)
        cc = int(c + shift_c)
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = int(color)


def _collect_red_components(g: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], list[int], dict[tuple[int, int], int]]:
    records: list[dict] = []
    for color in RED_VARIANTS:
        for coords in components_mask(g == int(color), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            records.append(
                {
                    "coords": pts,
                    "bbox": tuple(map(int, bbox_coords(coords))),
                    "color": int(color),
                }
            )

    records.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][1], rec["bbox"][2], rec["bbox"][3], rec["color"]))

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_color: list[int] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, rec in enumerate(records):
        red_coords.append(rec["coords"])
        red_bbox.append(rec["bbox"])
        red_color.append(int(rec["color"]))
        for rc in rec["coords"]:
            red_at[rc] = int(rid)

    return red_coords, red_bbox, red_color, red_at


def _collect_structures(
    g: np.ndarray, bg: int
) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    struct_mask = g != int(bg)
    for color in RED_VARIANTS:
        struct_mask &= g != int(color)

    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg structure"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    return struct_coords, struct_bbox


def _best_candidate_for_structure(
    g: np.ndarray,
    sid: int,
    struct_coords: list[set[tuple[int, int]]],
    struct_bbox: list[tuple[int, int, int, int]],
    red_at: dict[tuple[int, int], int],
    red_color: list[int],
    available: set[int],
    color_freq: Counter[int],
    use_frequency: bool,
    bg: int,
) -> dict | None:
    h, w = g.shape
    sr0, sc0, sr1, sc1 = map(int, struct_bbox[int(sid)])
    best: dict | None = None

    for dr, dc in _DIRS:
        hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
        for r, c in struct_coords[int(sid)]:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(g[nr, nc]) != int(bg):
                continue
            if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                continue

            rr, cc = int(nr), int(nc)
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                rr += int(dr)
                cc += int(dc)

            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) in RED_VARIANTS:
                rid = int(red_at[(int(rr), int(cc))])
                if int(rid) not in available:
                    continue
                rec = hits[rid]
                rec[0] += 1
                dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                if int(dist) < int(rec[1]):
                    rec[1] = int(dist)
                    rec[2] = (int(r), int(c))

        for rid, (cnt, mind, anchor) in hits.items():
            freq_score = int(color_freq[int(red_color[int(rid)])]) if use_frequency else 0
            score = (int(cnt), int(freq_score), -int(mind), -int(rid))
            cand = {
                "sid": int(sid),
                "rid": int(rid),
                "dir": (int(dr), int(dc)),
                "anchor": anchor,
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

    return best


def _solve_assignments(
    g: np.ndarray,
    *,
    use_frequency: bool,
    dynamic_frequency: bool,
) -> tuple[dict, list[dict]]:
    grid = np.array(g, dtype=int)
    bg = int(get_bg(grid))

    red_coords, red_bbox, red_color, red_at = _collect_red_components(grid)
    assert red_coords, "expected at least one red-family component"

    struct_coords, struct_bbox = _collect_structures(grid, bg)
    order = sorted(
        range(len(struct_coords)),
        key=lambda sid: (
            int(struct_bbox[int(sid)][1]),
            int(struct_bbox[int(sid)][0]),
            int(struct_bbox[int(sid)][3]),
            int(struct_bbox[int(sid)][2]),
        ),
    )

    scene = {
        "grid": grid,
        "bg": int(bg),
        "red_coords": red_coords,
        "red_bbox": red_bbox,
        "red_color": red_color,
        "red_at": red_at,
        "struct_coords": struct_coords,
        "struct_bbox": struct_bbox,
        "order": order,
    }

    base_freq = Counter(int(c) for c in red_color)
    available: set[int] = set(range(len(red_coords)))
    assignments: list[dict] = []

    for sid in order:
        color_freq = (
            Counter(int(red_color[rid]) for rid in available)
            if dynamic_frequency
            else Counter(base_freq)
        )
        best = _best_candidate_for_structure(
            grid,
            int(sid),
            struct_coords,
            struct_bbox,
            red_at,
            red_color,
            available,
            color_freq,
            bool(use_frequency),
            int(bg),
        )
        if best is None or best["anchor"] is None:
            continue
        assignments.append(best)
        available.remove(int(best["rid"]))

    return scene, assignments


def _render_output(scene: dict, assignments: list[dict]) -> np.ndarray:
    g = np.array(scene["grid"], dtype=int)
    out = g.copy()
    bg = int(scene["bg"])
    out[np.isin(out, np.array(RED_VARIANTS, dtype=int))] = int(bg)

    for item in assignments:
        rid = int(item["rid"])
        dr, dc = map(int, item["dir"])
        ar, ac = map(int, item["anchor"])
        coords = scene["red_coords"][rid]
        color = int(scene["red_color"][rid])
        r0, c0, r1, c1 = map(int, scene["red_bbox"][rid])
        tr, tc = int(ar + dr), int(ac + dc)

        if dr == 0 and dc == 1:
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        _apply_slide(out, coords, int(shift_r), int(shift_c), int(color), int(BLACK))

    return out


def _assignment_colors(scene: dict, assignments: list[dict]) -> list[int]:
    return [int(scene["red_color"][int(item["rid"])]) for item in assignments]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide red-family components into comb sockets using chain order and live shade frequencies.

    Input structure:
    - The background is the most common color. Large non-background comb/bracket structures appear in one color.
    - Outside the structures sit rectangular warm-colored components in two red-toned shades.
    - Later structures may see multiple same-score candidates in different red shades.

    Rule:
    - Find each structure's interior socket cells and ray-cast outward through background cells.
    - Process structures strictly from left to right. Earlier matches remove components from the available pool.
    - For each structure, prefer the candidate with the best hit count; when candidates are otherwise ambiguous,
      prefer the shade that is currently most frequent among the remaining red-family components, then use distance.
    - Slide the chosen component into the socket, paint its travel path black, and erase every unused red-family piece.

    Key insight: The tie-break is stateful twice: the chain order changes which pieces remain, and those removals
    change the remaining shade frequencies, which can flip how later ambiguities are resolved.
    """
    scene, assignments = _solve_assignments(
        np.array(grid, dtype=int),
        use_frequency=True,
        dynamic_frequency=True,
    )
    return _render_output(scene, assignments)


def _place_structure(inp: np.ndarray, spec: dict, top: int, left: int, bg: int) -> None:
    paste(inp, spec["patch"], int(top), int(left), transparent=int(bg))


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(BG_COLORS))
    sc = int(random.choice([c for c in STRUCT_COLORS if c != bg]))
    h = int(random.randint(*GRID_H_RANGE))
    w = int(random.randint(*GRID_W_RANGE))

    major = int(random.choice(RED_VARIANTS))
    minor = int([c for c in RED_VARIANTS if c != major][0])

    s1 = _structure_spec(WIDTH, TEETH, 1, bg, sc)  # opening right
    s2 = _structure_spec(WIDTH, TEETH, 3, bg, sc)  # opening left
    s3 = _structure_spec(WIDTH, TEETH, 3, bg, sc)  # opening left

    s1_top = int(random.choice(S1_TOP_CHOICES))
    s2_top = int(random.choice(S2_TOP_CHOICES))
    s3_top_choices = [t for t in S3_TOP_CHOICES if t + s3["shape"][0] <= h]
    if not s3_top_choices:
        return None
    s3_top = int(random.choice(s3_top_choices))

    s1_left = int(random.choice(S1_LEFT_CHOICES))
    s2_left = int(random.choice(S2_LEFT_CHOICES))
    s3_left = int(w - S3_RIGHT_MARGIN)

    if s2_left <= s1_left + s1["shape"][1]:
        return None
    if s3_left <= s2_left + s2["shape"][1]:
        return None

    inp = np.full((h, w), bg, dtype=int)
    _place_structure(inp, s1, s1_top, s1_left, bg)
    _place_structure(inp, s2, s2_top, s2_left, bg)
    _place_structure(inp, s3, s3_top, s3_left, bg)

    slide1 = int(random.choice(S1_SLIDE_CHOICES))
    far2 = int(random.choice(FAR_GAP_CHOICES))
    near2 = int(random.choice(NEAR_GAP_CHOICES))
    far3 = int(random.choice(FAR_GAP_CHOICES))
    near3 = int(random.choice(NEAR_GAP_CHOICES))
    if not (far2 > near2 and far3 > near3):
        return None

    s1_major = _translate(
        _translate(s1["dock_rel"], int(s1_top), int(s1_left)),
        int(slide1 * s1["dir"][0]),
        int(slide1 * s1["dir"][1]),
    )
    s2_major = [(int(s2_top + 3), int(s2_left - far2 - 1)), (int(s2_top + 3), int(s2_left - far2))]
    s2_minor = [(int(s2_top + 4), int(s2_left - near2 - 1)), (int(s2_top + 4), int(s2_left - near2))]
    s3_major = [(int(s3_top + 3), int(s3_left - far3 - 1)), (int(s3_top + 3), int(s3_left - far3))]
    s3_minor = [(int(s3_top + 4), int(s3_left - near3 - 1)), (int(s3_top + 4), int(s3_left - near3))]
    filler_major = [(0, int(w - 2))]

    all_components = [
        (s1_major, major),
        (s2_major, major),
        (s2_minor, minor),
        (s3_major, major),
        (s3_minor, minor),
        (filler_major, major),
    ]

    occupied: set[tuple[int, int]] = set()
    for coords, color in all_components:
        for r, c in coords:
            if not (0 <= int(r) < h and 0 <= int(c) < w):
                return None
            if int(inp[int(r), int(c)]) != int(bg):
                return None
            if (int(r), int(c)) in occupied:
                return None
        for r, c in coords:
            occupied.add((int(r), int(c)))
            inp[int(r), int(c)] = int(color)

    scene, combo_assign = _solve_assignments(inp, use_frequency=True, dynamic_frequency=True)
    _, chain_only_assign = _solve_assignments(inp, use_frequency=False, dynamic_frequency=True)
    _, static_freq_assign = _solve_assignments(inp, use_frequency=True, dynamic_frequency=False)

    combo_colors = _assignment_colors(scene, combo_assign)
    chain_colors = _assignment_colors(scene, chain_only_assign)
    static_colors = _assignment_colors(scene, static_freq_assign)

    if combo_colors != [major, major, minor]:
        return None
    if len(chain_colors) < 3 or int(chain_colors[1]) == int(combo_colors[1]):
        return None
    if len(static_colors) < 3 or int(static_colors[2]) == int(combo_colors[2]):
        return None

    expected = _render_output(scene, combo_assign)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo 8b9c3697/27")
