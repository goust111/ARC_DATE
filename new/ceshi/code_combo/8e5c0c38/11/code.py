# Combo variant of 8e5c0c38 (index 11)
# combo_spec: D2/3+D5/0
# Name: 标记指示轴方向与位置

from __future__ import annotations

import itertools
import math
import random

import numpy as np

from arc_utils import get_bg, label_mask, paste

"""
design_analysis:
  a_specific_behavior:
    input_visual:
      - 22x22 grid with one dominant background color.
      - Exactly 3 noisy sprites, each a distinct non-red color.
      - Red (2) is reserved for detached metadata markers and never used by payload sprites or bg.
      - Every sprite has one nearby 3-cell red line:
        - horizontal red line => the sprite must be repaired by vertical reflection
        - vertical red line => the sprite must be repaired by horizontal reflection
      - The marker's midpoint is also the exact axis coordinate:
        - horizontal line midpoint column = vertical symmetry axis column
        - vertical line midpoint row = horizontal symmetry axis row
      - Output removes both the red markers and the unmatched same-color noise.
    flow_trace:
      - Original task per sprite:
        1. collect one color layer
        2. search every vertical axis position
        3. keep only pixels that have a mirror partner under the best axis
      - Combo task per sprite:
        1. find the detached red marker paired with that sprite
        2. decode orientation to choose vertical-vs-horizontal reflection
        3. decode the marker midpoint as the exact axis position
        4. keep only pixels that have a same-color partner under that given axis
      - Difference:
        - neither axis family nor axis position is inferred from the noisy sprite anymore
        - both are read from a single external symbol system
  b_complexity_calibration:
    - Marker vocabulary is intentionally tiny: only two line orientations, both always length 3.
    - Every generated sample contains both axis families, so the orientation cue repeats inside one grid.
    - Marker position is still visually salient because the line midpoint sits exactly on the intended axis.
    - Half-integer axes are removed. With explicit marker midpoints, integer row/column axes keep the encoding learnable.
  c_implicit_assumption_audit:
    - Original `main()` treated every non-bg color as payload. That breaks because red is now metadata and must disappear.
    - Original search over vertical axes no longer applies. The combo solver must read both family and position from the marker before denoising.
    - Original per-color iteration assumed color alone identifies the work unit. The combo solver instead pairs red-marker components with object components.
    - Original `out = g.copy()` was harmless. Here it would incorrectly preserve markers, so the solver clears red first.
    - New solver assumption: each payload sprite is one 4-connected monochrome component and each has exactly one nearby straight odd-length marker.
  d_placement_legality:
    - The grid uses three separated slots. Each slot contains one sprite-marker bundle only.
    - Vertical-axis sprites reserve one extra marker row above or below; horizontal-axis sprites reserve one extra marker column left or right.
    - A one-cell background gap is blocked from noise placement so markers never fuse with sprites.
    - Noise is placed only on background cells 4-adjacent to the clean sprite and never on the marked axis itself.
  e_visual_observability:
    - From 2-4 examples, a human can see that red marks vanish in the output and that horizontal-vs-vertical red lines predict left-right vs up-down cleanup.
    - Because the marker midpoint coincides with the repair axis, the same symbol also explains why specific off-axis noise disappears.
    - The generator rejects cases where "use marker direction but still search for axis position" gives the same result, so the D5 part remains visible.
  f_visual_cleanliness:
    - Markers stay detached and short, with one clean bg cell of separation.
    - Slot spacing prevents cross-object ambiguity and keeps the pairing cue local.
    - Only one metadata color is introduced, so the scene stays readable.
  g_change_inference:
    - No payload recoloring occurs.
    - Red always means metadata, never object content.
    - The marker system is structurally grounded by orientation and proximity, not by numeric color ids.
  solver_generator_contract:
    - `main()` assumes every red component is a straight contiguous odd-length line and decodes its midpoint as the axis coordinate.
    - `main()` pairs markers to sprites by minimum total center distance; the generator's slot layout makes that assignment unique and stable.
    - `main()` assumes each sprite stays a single monochrome 4-connected component; the generator enforces this for clean cores and only adds 4-adjacent noise.
    - Degeneration guard: generation is rejected unless
      1. combo output differs from a D2-only ablation (read direction, still search axis position), and
      2. combo output differs from a D5-only-style ablation (read position, force vertical reflection).
"""

MARK = 2


def _bbox_center(coords: np.ndarray) -> tuple[float, float]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return (float(rs.min() + rs.max()) / 2.0, float(cs.min() + cs.max()) / 2.0)


def _mirror_cell(axis_dir: str, k: int, r: int, c: int) -> tuple[int, int]:
    if axis_dir == "vertical":
        return int(r), int(k) - int(c)
    return int(k) - int(r), int(c)


def _axis_scores(coords: list[tuple[int, int]], axis_dir: str, height: int, width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    limit = width if axis_dir == "vertical" else height
    scores: dict[int, int] = {}
    for k in range(0, 2 * (limit - 1) + 1):
        score = 0
        for r, c in pts:
            rr, cc = _mirror_cell(axis_dir, k, r, c)
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) in pts:
                score += 1
        scores[int(k)] = int(score)
    return scores


def _best_axis(coords: list[tuple[int, int]], axis_dir: str, height: int, width: int) -> tuple[int, int]:
    best_k = 0
    best_score = -1
    for k, score in _axis_scores(coords, axis_dir, height, width).items():
        if score > best_score:
            best_k = int(k)
            best_score = int(score)
    return int(best_k), int(best_score)


def _second_best_score(
    coords: list[tuple[int, int]], axis_dir: str, true_k: int, height: int, width: int
) -> int:
    scores = _axis_scores(coords, axis_dir, height, width)
    return max(score for k, score in scores.items() if int(k) != int(true_k))


def _best_score(coords: list[tuple[int, int]], axis_dir: str, height: int, width: int) -> int:
    return max(_axis_scores(coords, axis_dir, height, width).values())


def _other_axis(axis_dir: str) -> str:
    return "horizontal" if axis_dir == "vertical" else "vertical"


def _parse_marker_component(coords: list[tuple[int, int]]) -> dict:
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    if (
        len(rows) == 1
        and len(cols) >= 3
        and len(cols) % 2 == 1
        and cols[-1] - cols[0] + 1 == len(cols)
    ):
        center_r = int(rows[0])
        center_c = int(cols[len(cols) // 2])
        return {
            "coords": coords,
            "axis_dir": "vertical",
            "axis_k": int(2 * center_c),
            "center": (float(center_r), float(center_c)),
        }
    if (
        len(cols) == 1
        and len(rows) >= 3
        and len(rows) % 2 == 1
        and rows[-1] - rows[0] + 1 == len(rows)
    ):
        center_r = int(rows[len(rows) // 2])
        center_c = int(cols[0])
        return {
            "coords": coords,
            "axis_dir": "horizontal",
            "axis_k": int(2 * center_r),
            "center": (float(center_r), float(center_c)),
        }
    raise ValueError("marker must be a straight odd-length line")


def _scene_records(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    obj_lab, obj_n = label_mask((grid != int(bg)) & (grid != int(MARK)), connectivity=4)
    marker_lab, marker_n = label_mask(grid == int(MARK), connectivity=4)

    obj_comps: list[dict] = []
    for lbl in range(1, int(obj_n) + 1):
        pts_arr = np.argwhere(obj_lab == int(lbl))
        if pts_arr.size == 0:
            continue
        coords = [(int(r), int(c)) for r, c in pts_arr]
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            raise ValueError("payload component must be monochrome")
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        obj_comps.append(
            {
                "id": int(lbl),
                "coords": coords,
                "color": int(next(iter(colors))),
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
                "center": _bbox_center(pts_arr),
            }
        )

    marker_comps: list[dict] = []
    for lbl in range(1, int(marker_n) + 1):
        pts_arr = np.argwhere(marker_lab == int(lbl))
        if pts_arr.size == 0:
            continue
        coords = [(int(r), int(c)) for r, c in pts_arr]
        marker_comps.append(_parse_marker_component(coords))

    return obj_comps, marker_comps


def _assign_markers_to_objects(obj_comps: list[dict], marker_comps: list[dict]) -> dict[int, dict]:
    assert len(obj_comps) == len(marker_comps), "each sprite must have exactly one marker"
    best_perm: tuple[int, ...] | None = None
    best_cost: float | None = None
    for perm in itertools.permutations(range(len(marker_comps))):
        cost = 0.0
        for obj, marker_idx in zip(obj_comps, perm):
            orow, ocol = obj["center"]
            mrow, mcol = marker_comps[int(marker_idx)]["center"]
            cost += abs(orow - mrow) + abs(ocol - mcol)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    assert best_perm is not None
    return {int(obj["id"]): marker_comps[int(marker_idx)] for obj, marker_idx in zip(obj_comps, best_perm)}


def _solve_with_markers(
    grid: np.ndarray, *, use_direction: bool, use_explicit_position: bool
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    obj_comps, marker_comps = _scene_records(g, bg)
    marker_for_obj = _assign_markers_to_objects(obj_comps, marker_comps)

    for obj in obj_comps:
        pts = obj["coords"]
        marker = marker_for_obj[int(obj["id"])]

        axis_dir = str(marker["axis_dir"]) if use_direction else "vertical"
        if use_explicit_position:
            if axis_dir == "vertical":
                axis_k = int(2 * int(marker["center"][1]))
            else:
                axis_k = int(2 * int(marker["center"][0]))
        else:
            axis_k, _score = _best_axis(pts, axis_dir, height, width)

        for r, c in pts:
            rr, cc = _mirror_cell(axis_dir, axis_k, r, c)
            if not (0 <= rr < height and 0 <= cc < width and int(g[rr, cc]) == int(obj["color"])):
                out[int(r), int(c)] = int(bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each sprite using an external marker that jointly specifies axis direction and axis location.

    Input structure:
    - The most common color is background. Red (2) is reserved for detached metadata markers, not payload.
    - Each non-red sprite is nearly mirror-symmetric but contains same-color noise pixels, and each sprite has one nearby red 3-cell line.

    Rule:
    - Pair each sprite with its nearby red line. A horizontal red line means use a vertical mirror axis, and a vertical red line means use a horizontal mirror axis.
    - The midpoint of that red line gives the exact axis position. Delete every payload pixel that lacks a same-color mirror counterpart across the marked axis, and erase the red marker itself.

    Key insight:
    - The red marker is a unified instruction: its orientation chooses the reflection family, and its midpoint fixes the axis coordinate. The sprite no longer determines either quantity by search.
    """
    return _solve_with_markers(grid, use_direction=True, use_explicit_position=True)


# --------------- generator ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 8, 8), (0, 13, 8, 21), (13, 4, 21, 17)],
    [(0, 0, 8, 9), (0, 12, 8, 21), (13, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
TRAP_PIXEL_COUNT_MIN = 14
HEIGHT_MIN = 2
HEIGHT_MAX = 9
ODD_SIDE_W_RANGE = range(1, 6)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
TRAP_MIN_NOISE = 4
DECOY_SHIFTS = (1, 2)

# ── CFG: symmetry quality ──
SPRITE_AXIS_GAP_MIN = 3
ORTHO_AXIS_GAP_MIN = 2

# ── CFG: markers ──
MARKER_LEN = 3
MARKER_GAP = 1

# ── CFG: retry limits ──
SPRITE_RETRIES = 200
GEN_RETRIES = 500

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _choose_shape_spec(max_h: int, max_w: int, min_pixels: int) -> tuple[int, int, int, int]:
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(max(int(min_pixels), PIXEL_COUNT_MIN), PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = 2 * side_w + 1
                if w > max_w:
                    continue
                area = h * w
                lo = max(int(min_pixels), math.ceil(DENSITY_LO * area))
                hi = min(PIXEL_COUNT_MAX, math.floor(DENSITY_HI * area))
                if not (lo <= target_count <= hi):
                    continue
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((h, w, target_count, center_count))
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _h, _w, target_count, _cc in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _target, _cc in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    inner_anchor_row = random.randrange(h)
    for r in range(h):
        if r == 0:
            c = 0
        elif side_w > 1 and r == inner_anchor_row:
            c = side_w - 1
        else:
            c = random.randrange(side_w)
        cells.add((r, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(r, c) for r in range(h) for c in range(side_w) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_sprite_vertical_raw(
    color: int, max_h: int, max_w: int, min_pixels: int
) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w, min_pixels)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mirror_c = w - 1 - c
            sprite[r, c] = int(color)
            sprite[r, mirror_c] = int(color)
        sprite[centers, side_w] = int(color)

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        _lab, n_comp = label_mask(sprite == int(color), connectivity=4)
        if int(n_comp) != 1:
            continue
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if _second_best_score(coords, "vertical", true_k, h, w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, int(true_k)
    raise ValueError("failed to build clean sprite")


def _build_clean_sprite(
    color: int, max_h: int, max_w: int, axis_dir: str, min_pixels: int
) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        if axis_dir == "vertical":
            sprite, true_k = _build_clean_sprite_vertical_raw(color, max_h, max_w, min_pixels)
        else:
            tmp, _tmp_k = _build_clean_sprite_vertical_raw(color, max_w, max_h, min_pixels)
            sprite = tmp.T.copy()
            true_k = sprite.shape[0] - 1

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        ortho_best = _best_score(coords, _other_axis(axis_dir), sprite.shape[0], sprite.shape[1])
        if ortho_best > len(coords) - ORTHO_AXIS_GAP_MIN:
            continue
        return sprite, int(true_k)
    raise ValueError("failed to build clean sprite with directional asymmetry")


def _marker_placement(
    axis_dir: str, sprite_shape: tuple[int, int], slot: tuple[int, int, int, int]
) -> tuple[int, int, list[tuple[int, int]], set[tuple[int, int]], int]:
    h, w = sprite_shape
    r0s, c0s, r1s, c1s = slot
    slot_h = int(r1s - r0s + 1)
    slot_w = int(c1s - c0s + 1)

    if axis_dir == "vertical":
        if h + MARKER_GAP + 1 > slot_h or w > slot_w:
            raise ValueError("slot too small for vertical-axis sprite")
        side = random.choice(["top", "bottom"])
        c0 = random.randint(int(c0s), int(c1s - w + 1))
        if side == "top":
            r0 = random.randint(int(r0s + MARKER_GAP + 1), int(r1s - h + 1))
            marker_r = int(r0 - MARKER_GAP - 1)
            gap_r = int(r0 - 1)
        else:
            r0 = random.randint(int(r0s), int(r1s - h - MARKER_GAP))
            marker_r = int(r0 + h + MARKER_GAP)
            gap_r = int(r0 + h)
        axis_c = int(c0 + w // 2)
        marker_cells = [(int(marker_r), int(axis_c + dc)) for dc in (-1, 0, 1)]
        blockers = set(marker_cells) | {(int(gap_r), int(axis_c + dc)) for dc in (-1, 0, 1)}
        return int(r0), int(c0), marker_cells, blockers, int(2 * axis_c)

    if h > slot_h or w + MARKER_GAP + 1 > slot_w:
        raise ValueError("slot too small for horizontal-axis sprite")
    side = random.choice(["left", "right"])
    r0 = random.randint(int(r0s), int(r1s - h + 1))
    if side == "left":
        c0 = random.randint(int(c0s + MARKER_GAP + 1), int(c1s - w + 1))
        marker_c = int(c0 - MARKER_GAP - 1)
        gap_c = int(c0 - 1)
    else:
        c0 = random.randint(int(c0s), int(c1s - w - MARKER_GAP))
        marker_c = int(c0 + w + MARKER_GAP)
        gap_c = int(c0 + w)
    axis_r = int(r0 + h // 2)
    marker_cells = [(int(axis_r + dr), int(marker_c)) for dr in (-1, 0, 1)]
    blockers = set(marker_cells) | {(int(axis_r + dr), int(gap_c)) for dr in (-1, 0, 1)}
    return int(r0), int(c0), marker_cells, blockers, int(2 * axis_r)


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    placed: list[dict] = []

    for spec, slot in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        try:
            r0, c0, marker_cells, blockers, true_k = _marker_placement(
                spec["axis_dir"], sprite.shape, slot
            )
        except ValueError:
            return None

        paste(out, sprite, int(r0), int(c0), transparent=0)
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["marker_cells"] = marker_cells
        spec["noise_blockers"] = blockers
        spec["true_k"] = int(true_k)
        placed.append(spec)

    return out, placed


def _noise_candidates(
    clean_mask: np.ndarray,
    axis_dir: str,
    true_k: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    height, width = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    if forbidden:
        for r, c in forbidden:
            if 0 <= int(r) < height and 0 <= int(c) < width:
                adj[int(r), int(c)] = False

    if axis_dir == "vertical":
        cols = np.arange(width)
        mirror_cols = int(true_k) - cols
        on_axis = mirror_cols == cols
        adj[:, on_axis] = False
        valid = (mirror_cols >= 0) & (mirror_cols < width) & ~on_axis
        for c in np.where(valid)[0]:
            adj[:, int(c)] &= ~clean_mask[:, int(mirror_cols[int(c)])]
    else:
        rows = np.arange(height)
        mirror_rows = int(true_k) - rows
        on_axis = mirror_rows == rows
        adj[on_axis, :] = False
        valid = (mirror_rows >= 0) & (mirror_rows < height) & ~on_axis
        for r in np.where(valid)[0]:
            adj[int(r), :] &= ~clean_mask[int(mirror_rows[int(r)]), :]

    rows, cols = np.where(adj)
    return list(zip(rows.tolist(), cols.tolist()))


def _candidate_decoy_pairs(
    clean_mask: np.ndarray,
    axis_dir: str,
    true_k: int,
    forbidden: set[tuple[int, int]],
) -> dict[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
    candidates = _noise_candidates(clean_mask, axis_dir, true_k, forbidden=forbidden)
    cand_set = set(candidates)
    limit = clean_mask.shape[1] if axis_dir == "vertical" else clean_mask.shape[0]

    pairs_by_k: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for shift in DECOY_SHIFTS:
        for sign in (-1, 1):
            wrong_k = int(true_k + sign * 2 * shift)
            if not (0 <= wrong_k <= 2 * (limit - 1)):
                continue
            seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for cell in candidates:
                mate = _mirror_cell(axis_dir, wrong_k, cell[0], cell[1])
                mate = (int(mate[0]), int(mate[1]))
                if mate not in cand_set or mate == cell:
                    continue
                if _mirror_cell(axis_dir, true_k, cell[0], cell[1]) == mate:
                    continue
                pair = tuple(sorted([cell, mate]))
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
            if pairs:
                pairs_by_k[int(wrong_k)] = pairs
    return pairs_by_k


def _choose_decoy_cells(
    clean_coords: list[tuple[int, int]],
    clean_mask: np.ndarray,
    axis_dir: str,
    true_k: int,
    forbidden: set[tuple[int, int]],
    max_pairs: int,
) -> list[tuple[int, int]] | None:
    pairs_by_k = _candidate_decoy_pairs(clean_mask, axis_dir, true_k, forbidden)
    best_cells: list[tuple[int, int]] | None = None
    best_metric: tuple[int, int, int, int, float] | None = None

    for wrong_k, pairs in pairs_by_k.items():
        max_combo = min(int(max_pairs), len(pairs))
        for combo_size in range(max_combo, 0, -1):
            for combo in itertools.combinations(pairs, combo_size):
                cells = [cell for pair in combo for cell in pair]
                if len(set(cells)) != len(cells):
                    continue
                test_coords = clean_coords + cells
                scores = _axis_scores(test_coords, axis_dir, GRID_SIZE, GRID_SIZE)
                best_k, _best_score_here = _best_axis(test_coords, axis_dir, GRID_SIZE, GRID_SIZE)
                metric = (
                    int(best_k != true_k),
                    int(scores[int(wrong_k)] - scores[int(true_k)]),
                    int(scores[int(wrong_k)]),
                    int(combo_size),
                    random.random(),
                )
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_cells = cells
    return best_cells


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict, force_decoy: bool) -> bool:
    color = int(spec["color"])
    axis_dir = str(spec["axis_dir"])
    true_k = int(spec["true_k"])
    clean_mask = out == int(color)
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if force_decoy:
        lo = max(lo, TRAP_MIN_NOISE)
    if lo > hi:
        return False

    noise_target = random.randint(lo, hi)
    chosen: list[tuple[int, int]] = []
    forbidden = set(spec["noise_blockers"])
    current_coords = list(clean_coords)

    if force_decoy:
        decoy_cells = _choose_decoy_cells(
            clean_coords=current_coords,
            clean_mask=clean_mask,
            axis_dir=axis_dir,
            true_k=true_k,
            forbidden=forbidden,
            max_pairs=max(1, noise_target // 2),
        )
        if decoy_cells is None or len(decoy_cells) < 2:
            return False
        for cell in decoy_cells:
            chosen.append(cell)
            current_coords.append(cell)
            forbidden.add(cell)
            mirror = _mirror_cell(axis_dir, true_k, cell[0], cell[1])
            if 0 <= mirror[0] < GRID_SIZE and 0 <= mirror[1] < GRID_SIZE:
                forbidden.add((int(mirror[0]), int(mirror[1])))

    while len(chosen) < noise_target:
        candidates = _noise_candidates(clean_mask, axis_dir, true_k, forbidden=forbidden)
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
        if not viable:
            return False
        cell = random.choice(viable)
        chosen.append(cell)
        current_coords.append(cell)
        forbidden.add(cell)
        mirror = _mirror_cell(axis_dir, true_k, cell[0], cell[1])
        if 0 <= mirror[0] < GRID_SIZE and 0 <= mirror[1] < GRID_SIZE:
            forbidden.add((int(mirror[0]), int(mirror[1])))

    if force_decoy:
        best_k, _best_score_here = _best_axis(current_coords, axis_dir, GRID_SIZE, GRID_SIZE)
        if int(best_k) == int(true_k):
            return False

    for r, c in chosen:
        inp[int(r), int(c)] = int(color)
    return True


def _add_markers(inp: np.ndarray, placed_specs: list[dict]) -> None:
    for spec in placed_specs:
        for r, c in spec["marker_cells"]:
            inp[int(r), int(c)] = int(MARK)


def _gen_attempt() -> np.ndarray | None:
    palette = [color for color in range(10) if color != MARK]
    bg, *obj_colors = random.sample(palette, NUM_SPRITES + 1)
    slots = random.choice(LAYOUTS)

    num_horizontal = random.choice([1, 2])
    axis_dirs = ["horizontal"] * num_horizontal + ["vertical"] * (NUM_SPRITES - num_horizontal)
    random.shuffle(axis_dirs)
    trap_idx = random.randrange(NUM_SPRITES)

    sprite_specs: list[dict] = []
    try:
        for idx, (color, slot, axis_dir) in enumerate(zip(obj_colors, slots, axis_dirs)):
            slot_h = int(slot[2] - slot[0] + 1)
            slot_w = int(slot[3] - slot[1] + 1)
            max_h = slot_h - (MARKER_GAP + 1 if axis_dir == "vertical" else 0)
            max_w = slot_w - (MARKER_GAP + 1 if axis_dir == "horizontal" else 0)
            min_pixels = TRAP_PIXEL_COUNT_MIN if idx == trap_idx else PIXEL_COUNT_MIN
            sprite, axis_k_local = _build_clean_sprite(
                int(color), max_h, max_w, axis_dir, min_pixels=min_pixels
            )
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "axis_dir": axis_dir,
                    "axis_k_local": int(axis_k_local),
                    "is_trap": bool(idx == trap_idx),
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(int(bg), sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec, force_decoy=bool(spec["is_trap"])):
            return None
    _add_markers(inp, placed_specs)

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    try:
        combo_out = main(inp)
        direction_only_out = _solve_with_markers(inp, use_direction=True, use_explicit_position=False)
        position_only_out = _solve_with_markers(inp, use_direction=False, use_explicit_position=True)
    except (AssertionError, ValueError):
        return None

    if not np.array_equal(combo_out, out):
        return None
    if np.array_equal(combo_out, direction_only_out):
        return None
    if np.array_equal(combo_out, position_only_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
