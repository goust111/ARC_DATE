# Combo variant of 8e5c0c38 (index 35)
# combo_spec: D2/1+D5/0+D15/2
# Name: 标记系统全指示修复

from __future__ import annotations

"""
Design analysis for combo variant D2/1 + D5/0 + D15/2.

a) Concrete behavior trace:
  - Original task behavior on one sprite:
    1. Take one non-background color layer.
    2. Search only vertical axes.
    3. Keep pixels that have same-color mirrors across the best vertical axis.
  - Combo behavior on one sprite:
    1. Detect the nearby red marker component.
    2. Read its geometry: a vertical red bar means the sprite should be repaired by left-right mirroring, while a
       horizontal red bar means it should be repaired by up-down mirroring.
    3. Read its position: for a vertical bar, the bar's column is the exact axis column; for a horizontal bar, the
       bar's row is the exact axis row.
    4. Pair that marker with the nearby sprite whose bbox spans the marked row/column, then delete every sprite pixel
       lacking a same-color mirror partner across the marked axis. The marker itself disappears in the output.
  - Concrete mini-example:
    - Suppose a blue blob sits at rows 8..12, cols 3..7, and a red bar of cells {(10,0),(10,1),(10,2)} sits to its
      left. The old solver would still search vertical axes and could keep the wrong side if the blob is horizontally
      symmetric. The combo solver instead sees a horizontal red bar, interprets that as a horizontal axis, uses row 10
      exactly, and removes only the blue pixels that do not have top/bottom partners.
  - Difference summary:
    - D2 changes "which axis direction is valid".
    - D5 changes "how the axis location is obtained".
    - D15 provides the visible symbol system that carries both pieces of metadata.

b) Complexity calibration:
  - Only two marker geometries are allowed: vertical bar and horizontal bar. This keeps the symbol vocabulary small.
  - Every generated grid contains both axis directions, so the geometry-to-direction rule is repeated inside a single
    sample and becomes visually learnable.
  - Marker length varies only within {2, 3}. That gives repetition without making the symbol system too brittle.
  - Clean sprites are still near-symmetric noisy objects, so the repair action stays close to the original task rather
    than degenerating into arbitrary marker-driven editing.

c) Hidden-assumption audit of the original main():
  - Original assumption: every non-background color is payload.
    - Broken: red (2) is now metadata, so the solver must erase it instead of denoising it.
  - Original assumption: processing by color is enough.
    - Broken: the decisive grouping is marker-to-object pairing, not blind color layers.
  - Original assumption: all candidate axes are vertical and must be searched.
    - Broken twice: orientation now comes from marker shape, and axis position comes from marker coordinates.
  - Original assumption: copying the whole grid to output is harmless.
    - Broken: marker pixels must always vanish from the output.
  - New combo-specific assumption:
    - A horizontal marker must sit left or right of its sprite on the actual axis row; otherwise position would not
      specify the row unambiguously. The generator enforces that placement convention directly.

d) Placement legality:
  - The generator uses three roomy slot regions so sprite bundles stay disjoint.
  - Vertical-axis bundles reserve extra height for "marker + one-cell gap + sprite".
  - Horizontal-axis bundles reserve extra width for "marker + one-cell gap + sprite".
  - Marker cells and the one-cell axis gap are blocked during noise placement, preventing accidental marker-object
    merging.
  - Each sprite color is unique and each sprite is generated as one 4-connected component, making marker pairing stable.

e) Visual observability:
  - The same special color always disappears in the output, so viewers can infer that red is instructional.
  - Bar orientation is directly visible and repeated across objects; some objects repair left-right and others repair
    up-down within the same grid.
  - Because the marker lies on the exact reflected row/column, the input-output relation is recoverable from 2-4
    examples without relying on hidden numeric properties of colors.

f) Visual cleanliness:
  - Markers are external to sprites rather than embedded through them, so they do not obscure payload shape.
  - One empty gap cell separates each marker from its sprite along the marked axis, keeping the components legible.
  - Objects live in separated slots, avoiding cross-object ambiguity when matching markers to targets.

g) Change inferability:
  - No payload recoloring is used.
  - The only symbolic color is red, and its role is structural rather than numeric.
  - Axis direction is inferred from visible geometry, not from color id, length parity, or generation order.

9) New solver assumptions checked against the generator:
  - Assumption: every marker is a 1-cell-thick straight bar of length >= 2.
    - Guarantee: the generator only emits 2- or 3-cell vertical/horizontal red bars.
  - Assumption: each object has exactly one compatible nearby marker spanning its true axis row/column.
    - Guarantee: placement is one marker per sprite inside separate slots.
  - Assumption: each object remains a single 4-connected non-marker component even after noise.
    - Guarantee: base sprites are connectivity-checked, and noise is sampled only from 4-neighbors.
  - Assumption: unmatched noise never occupies marker or reserved gap cells.
    - Guarantee: those cells are blocked globally before noise sampling.
"""

import math
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import get_bg, label_mask, paste


MARK = 2
GRID_SIZE = 22
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 9, 8), (0, 13, 9, 21), (12, 4, 21, 17)],
    [(0, 1, 9, 9), (0, 12, 9, 20), (12, 3, 21, 16)],
]

PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 6
ODD_SIDE_W_RANGE = range(1, 6)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

MARKER_GAP = 1
MARKER_LENGTHS = (2, 3)

SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2
SPRITE_RETRIES = 220
GEN_RETRIES = 500
FLIP_PROB = 0.5


def _axis_scores(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    if direction == "vertical":
        for k in range(2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k - c)
                if 0 <= cc < width and (int(r), int(cc)) in pts:
                    score += 1
            scores[int(k)] = int(score)
    elif direction == "horizontal":
        for k in range(2 * (height - 1) + 1):
            score = 0
            for r, c in pts:
                rr = int(k - r)
                if 0 <= rr < height and (int(rr), int(c)) in pts:
                    score += 1
            scores[int(k)] = int(score)
    else:
        raise ValueError(f"unknown direction: {direction}")
    return scores


def _best_axis(coords: list[tuple[int, int]], height: int, width: int) -> tuple[str, int, int]:
    best_direction = "vertical"
    best_k = 0
    best_score = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if score > best_score:
                best_score = int(score)
                best_direction = direction
                best_k = int(k)
    return best_direction, best_k, best_score


def _best_other_score(
    coords: list[tuple[int, int]], height: int, width: int, true_direction: str, true_k: int
) -> int:
    best = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if direction == true_direction and int(k) == int(true_k):
                continue
            if score > best:
                best = int(score)
    return best


def _paired_under_axis(
    pts: set[tuple[int, int]], r: int, c: int, height: int, width: int, direction: str, axis_pos: int
) -> bool:
    if direction == "vertical":
        cc = int(2 * axis_pos - c)
        return 0 <= cc < width and (int(r), int(cc)) in pts
    rr = int(2 * axis_pos - r)
    return 0 <= rr < height and (int(rr), int(c)) in pts


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    r0a, c0a, r1a, c1a = map(int, b1)
    r0b, c0b, r1b, c1b = map(int, b2)
    if r1a < r0b:
        dr = int(r0b - r1a - 1)
    elif r1b < r0a:
        dr = int(r0a - r1b - 1)
    else:
        dr = 0
    if c1a < c0b:
        dc = int(c0b - c1a - 1)
    elif c1b < c0a:
        dc = int(c0a - c1b - 1)
    else:
        dc = 0
    return int(dr + dc)


def _component_records(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    obj_lab, obj_n = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=4)
    marker_lab, marker_n = label_mask(g == int(MARK), connectivity=4)

    obj_comps: list[dict] = []
    for lbl in range(1, int(obj_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(obj_lab == int(lbl))]
        if not coords:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        obj_comps.append(
            {
                "id": int(lbl),
                "coords": coords,
                "color": int(next(iter(colors))),
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    marker_comps: list[dict] = []
    for lbl in range(1, int(marker_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(marker_lab == int(lbl))]
        if not coords:
            continue
        rs = sorted({int(r) for r, _c in coords})
        cs = sorted({int(c) for _r, c in coords})
        if len(cs) == 1 and len(rs) >= 2:
            direction = "vertical"
            axis_pos = int(cs[0])
        elif len(rs) == 1 and len(cs) >= 2:
            direction = "horizontal"
            axis_pos = int(rs[0])
        else:
            continue
        marker_comps.append(
            {
                "id": int(lbl),
                "coords": coords,
                "direction": direction,
                "axis_pos": int(axis_pos),
                "bbox": (
                    int(min(r for r, _c in coords)),
                    int(min(c for _r, c in coords)),
                    int(max(r for r, _c in coords)),
                    int(max(c for _r, c in coords)),
                ),
            }
        )
    return obj_comps, marker_comps


def _pair_markers_to_objects(obj_comps: list[dict], marker_comps: list[dict]) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    used_obj_ids: set[int] = set()

    for marker in sorted(marker_comps, key=lambda rec: (rec["direction"], rec["bbox"])):
        direction = str(marker["direction"])
        axis_pos = int(marker["axis_pos"])
        if direction == "vertical":
            candidates = [
                comp
                for comp in obj_comps
                if comp["id"] not in used_obj_ids and int(comp["bbox"][1]) <= axis_pos <= int(comp["bbox"][3])
            ]
            if not candidates:
                continue
            comp = min(
                candidates,
                key=lambda rec: (
                    _bbox_gap(marker["bbox"], rec["bbox"]),
                    abs(axis_pos - ((int(rec["bbox"][1]) + int(rec["bbox"][3])) // 2)),
                    abs(((int(marker["bbox"][0]) + int(marker["bbox"][2])) // 2) - ((int(rec["bbox"][0]) + int(rec["bbox"][2])) // 2)),
                ),
            )
        else:
            candidates = [
                comp
                for comp in obj_comps
                if comp["id"] not in used_obj_ids and int(comp["bbox"][0]) <= axis_pos <= int(comp["bbox"][2])
            ]
            if not candidates:
                continue
            comp = min(
                candidates,
                key=lambda rec: (
                    _bbox_gap(marker["bbox"], rec["bbox"]),
                    abs(axis_pos - ((int(rec["bbox"][0]) + int(rec["bbox"][2])) // 2)),
                    abs(((int(marker["bbox"][1]) + int(marker["bbox"][3])) // 2) - ((int(rec["bbox"][1]) + int(rec["bbox"][3])) // 2)),
                ),
            )

        used_obj_ids.add(int(comp["id"]))
        pairs.append((marker, comp))

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each noisy sprite using its explicit marker-defined symmetry axis.

    Input structure:
    - bg is the dominant background color, while red (2) is a dedicated marker color rather than payload.
    - Three colored sprites appear as separate near-symmetric objects with a nearby red bar for each one.
    - A vertical red bar marks a vertical axis on its column; a horizontal red bar marks a horizontal axis on its row.

    Rule:
    - Match each red bar to the nearby sprite whose bbox spans the bar's marked row or column.
    - Read the bar geometry to choose reflection direction, and read its row/column position as the exact axis.
    - Erase the markers and delete every sprite pixel that lacks a same-color mirror partner across that marked axis.

    Key insight: The solver no longer has to infer axes from noisy shape statistics; the marker system fully specifies
    both the axis direction and its location, so the real task is marker-object association plus mirror-based cleanup.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    obj_comps, marker_comps = _component_records(g, bg)
    for marker, comp in _pair_markers_to_objects(obj_comps, marker_comps):
        direction = str(marker["direction"])
        axis_pos = int(marker["axis_pos"])
        color = int(comp["color"])
        pts = {(int(r), int(c)) for r, c in comp["coords"]}
        for r, c in comp["coords"]:
            if not _paired_under_axis(pts, int(r), int(c), height, width, direction, axis_pos):
                out[int(r), int(c)] = int(bg)
            else:
                rr = int(r)
                cc = int(c)
                if int(g[rr, cc]) == color:
                    out[rr, cc] = color

    return out


def _choose_vertical_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    options: list[tuple[int, int, int, int]] = []
    for h in range(HEIGHT_MIN, min(HEIGHT_MAX, max_h) + 1):
        for side_w in ODD_SIDE_W_RANGE:
            w = 2 * side_w + 1
            if w > max_w:
                continue
            area = h * w
            lo = max(PIXEL_COUNT_MIN, math.ceil(DENSITY_LO * area))
            hi = min(PIXEL_COUNT_MAX, math.floor(DENSITY_HI * area))
            if lo > hi:
                continue
            for target_count in range(lo, hi + 1):
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((h, w, target_count, center_count))
    if not options:
        raise ValueError("no feasible odd-axis sprite spec")
    weights = [1.0 / (h * w) for h, w, _tc, _cc in options]
    return random.choices(options, weights=weights, k=1)[0]


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
        cells.add((int(r), int(c)))

    remaining = int(left_count - len(cells))
    if remaining < 0:
        raise ValueError("left_count too small")
    pool = [(int(r), int(c)) for r in range(h) for c in range(side_w) if (int(r), int(c)) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    rows[random.sample(range(h), center_count)] = True
    return rows


def _crop_to_bbox(sprite: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(sprite == int(color))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _build_vertical_candidate(color: int, max_h: int, max_w: int) -> np.ndarray:
    h, w, target_count, center_count = _choose_vertical_shape_spec(max_h, max_w)
    side_w = w // 2
    left_count = (target_count - center_count) // 2
    left = _left_cells(h, side_w, left_count)
    centers = _center_rows(h, center_count)

    sprite = np.zeros((h, w), dtype=int)
    for r, c in left:
        mirror_c = int(w - 1 - c)
        sprite[int(r), int(c)] = int(color)
        sprite[int(r), int(mirror_c)] = int(color)
    sprite[centers, int(side_w)] = int(color)
    return _crop_to_bbox(sprite, color)


def _build_clean_sprite(color: int, max_h: int, max_w: int, direction: str) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        if direction == "vertical":
            sprite = _build_vertical_candidate(color, max_h, max_w)
            axis_pos = int(sprite.shape[1] // 2)
            true_k = int(2 * axis_pos)
        elif direction == "horizontal":
            base = _build_vertical_candidate(color, max_w, max_h)
            sprite = base.T.copy()
            axis_pos = int(sprite.shape[0] // 2)
            true_k = int(2 * axis_pos)
        else:
            raise ValueError(f"unknown direction: {direction}")

        if sprite.shape[0] > max_h or sprite.shape[1] > max_w:
            continue
        coords = [tuple(map(int, x)) for x in np.argwhere(sprite == int(color))]
        if not coords:
            continue
        _lab, n = label_mask(sprite == int(color), connectivity=4)
        if int(n) != 1:
            continue
        best_direction, best_k, best_score = _best_axis(coords, *sprite.shape)
        if best_direction != direction or int(best_k) != int(true_k):
            continue
        if best_score != len(coords):
            continue
        other_score = _best_other_score(coords, *sprite.shape, direction, true_k)
        if best_score - other_score < SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, axis_pos
    raise ValueError("failed to build clean sprite")


def _sample_orientations() -> list[str]:
    horizontal_count = random.choice([1, 2])
    horizontal_idx = set(random.sample(range(NUM_SPRITES), horizontal_count))
    return ["horizontal" if i in horizontal_idx else "vertical" for i in range(NUM_SPRITES)]


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        direction = str(spec["axis_dir"])
        marker_len = int(spec["marker_len"])
        marker_side = str(spec["marker_side"])
        h, w = sprite.shape
        slot_h = int(r1s - r0s + 1)
        slot_w = int(c1s - c0s + 1)

        if direction == "vertical":
            bundle_h = int(h + MARKER_GAP + marker_len)
            if bundle_h > slot_h or w > slot_w:
                return None
            top = random.randint(int(r0s), int(r1s - bundle_h + 1))
            c0 = random.randint(int(c0s), int(c1s - w + 1))
            if marker_side == "above":
                marker_r0 = int(top)
                sprite_r0 = int(top + marker_len + MARKER_GAP)
                gap_cells = [(int(sprite_r0 - 1), int(c0 + spec["axis_local"]))]
            else:
                sprite_r0 = int(top)
                marker_r0 = int(sprite_r0 + h + MARKER_GAP)
                gap_cells = [(int(sprite_r0 + h), int(c0 + spec["axis_local"]))]
            sprite_c0 = int(c0)
            axis_pos = int(c0 + spec["axis_local"])
            marker_cells = [(int(marker_r0 + i), int(axis_pos)) for i in range(marker_len)]
        else:
            bundle_w = int(w + MARKER_GAP + marker_len)
            if h > slot_h or bundle_w > slot_w:
                return None
            r0 = random.randint(int(r0s), int(r1s - h + 1))
            left = random.randint(int(c0s), int(c1s - bundle_w + 1))
            if marker_side == "left":
                marker_c0 = int(left)
                sprite_c0 = int(left + marker_len + MARKER_GAP)
                gap_cells = [(int(r0 + spec["axis_local"]), int(sprite_c0 - 1))]
            else:
                sprite_c0 = int(left)
                marker_c0 = int(sprite_c0 + w + MARKER_GAP)
                gap_cells = [(int(r0 + spec["axis_local"]), int(sprite_c0 + w))]
            sprite_r0 = int(r0)
            axis_pos = int(r0 + spec["axis_local"])
            marker_cells = [(int(axis_pos), int(marker_c0 + i)) for i in range(marker_len)]

        paste(out, sprite, int(sprite_r0), int(sprite_c0), transparent=0)
        for r, c in marker_cells + gap_cells:
            blocked[int(r), int(c)] = True

        spec["r0"] = int(sprite_r0)
        spec["c0"] = int(sprite_c0)
        spec["axis_pos"] = int(axis_pos)
        spec["marker_cells"] = marker_cells
        spec["gap_cells"] = gap_cells
        placed.append(spec)

    return out, blocked, placed


def _noise_candidates(
    clean_mask: np.ndarray, direction: str, axis_pos: int, blocked: np.ndarray
) -> list[tuple[int, int]]:
    height, width = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask
    adj &= ~blocked

    if direction == "vertical":
        cols = np.arange(width)
        mirror_cols = 2 * int(axis_pos) - cols
        on_axis = mirror_cols == cols
        adj[:, on_axis] = False
        valid = (mirror_cols >= 0) & (mirror_cols < width) & ~on_axis
        for c in np.where(valid)[0]:
            adj[:, int(c)] &= ~clean_mask[:, int(mirror_cols[int(c)])]
    elif direction == "horizontal":
        rows = np.arange(height)
        mirror_rows = 2 * int(axis_pos) - rows
        on_axis = mirror_rows == rows
        adj[on_axis, :] = False
        valid = (mirror_rows >= 0) & (mirror_rows < height) & ~on_axis
        for r in np.where(valid)[0]:
            adj[int(r), :] &= ~clean_mask[int(mirror_rows[int(r)]), :]
    else:
        raise ValueError(f"unknown direction: {direction}")

    rows_arr, cols_arr = np.where(adj)
    return list(zip(rows_arr.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict, blocked: np.ndarray) -> bool:
    color = int(spec["color"])
    direction = str(spec["axis_dir"])
    axis_pos = int(spec["axis_pos"])
    true_k = int(2 * axis_pos)
    clean_mask = out == int(color)
    clean_coords = [tuple(map(int, x)) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)
    if clean_count == 0:
        return False

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, direction, axis_pos, blocked)
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in chosen and cell not in forbidden]
        if not viable:
            return False

        scored = []
        for cell in viable:
            alt = _best_other_score(current_coords + [cell], GRID_SIZE, GRID_SIZE, direction, true_k)
            scored.append((int(alt), random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = int(scored[0][0])
        pool = [cell for alt, _junk, cell in scored if int(alt) <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        r, c = map(int, cell)
        forbidden.add((int(r), int(c)))
        if direction == "vertical":
            mirror = (int(r), int(2 * axis_pos - c))
            if 0 <= mirror[1] < GRID_SIZE:
                forbidden.add(mirror)
        else:
            mirror = (int(2 * axis_pos - r), int(c))
            if 0 <= mirror[0] < GRID_SIZE:
                forbidden.add(mirror)

    best_direction, best_k, best_score = _best_axis(current_coords, GRID_SIZE, GRID_SIZE)
    other_score = _best_other_score(current_coords, GRID_SIZE, GRID_SIZE, direction, true_k)
    if best_direction != direction or int(best_k) != int(true_k):
        return False
    if best_score != clean_count:
        return False
    if best_score - other_score < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[int(r), int(c)] = int(color)
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    palette = [c for c in range(10) if c != MARK]
    bg, *obj_colors = random.sample(palette, NUM_SPRITES + 1)
    slots = random.choice(LAYOUTS)
    directions = _sample_orientations()

    sprite_specs: list[dict] = []
    try:
        for color, slot, direction in zip(obj_colors, slots, directions):
            slot_h = int(slot[2] - slot[0] + 1)
            slot_w = int(slot[3] - slot[1] + 1)
            marker_len = int(random.choice(MARKER_LENGTHS))
            if direction == "vertical":
                max_h = int(slot_h - MARKER_GAP - marker_len)
                max_w = int(slot_w)
                marker_side = random.choice(["above", "below"])
            else:
                max_h = int(slot_h)
                max_w = int(slot_w - MARKER_GAP - marker_len)
                marker_side = random.choice(["left", "right"])
            sprite, axis_local = _build_clean_sprite(int(color), max_h, max_w, direction)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "axis_dir": str(direction),
                    "axis_local": int(axis_local),
                    "marker_len": int(marker_len),
                    "marker_side": str(marker_side),
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(int(bg), sprite_specs, slots)
    if result is None:
        return None
    out, blocked, placed_specs = result

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec, blocked):
            return None
        for r, c in spec["marker_cells"]:
            inp[int(r), int(c)] = int(MARK)

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out, placed_specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
