# Combo variant of 8e5c0c38 (index 9)
# combo_spec: D2/1+D5/0
# Name: 标记指示自适应轴

from __future__ import annotations

"""
Design analysis for the combo variant.

a) Concrete behavior trace:
  - Original task:
    1. Treat each non-background color as one noisy sprite.
    2. Search only vertical axes.
    3. Keep pixels that have same-color mirrors across the best vertical axis.
  - New fused task:
    1. Input still has 3 noisy sprites, but now each sprite also has an external
       red marker.
    2. Marker orientation is semantic:
       * vertical red segment -> use a vertical symmetry axis
       * horizontal red segment -> use a horizontal symmetry axis
    3. Marker position gives the axis coordinate directly:
       * vertical marker column = reflection column
       * horizontal marker row = reflection row
    4. The solver pairs each marker with the nearby sprite whose bbox spans that
       axis coordinate, then deletes sprite pixels lacking a same-color mirror
       across the marker-specified axis. Markers themselves disappear.
  - Small example:
    * Suppose a blue sprite is almost left-right symmetric around column 6 and a
      red 2-pixel vertical segment sits one cell above it in column 6. Original
      logic would search many vertical axes. New logic reads column 6 directly.
    * Suppose a green sprite is almost up-down symmetric around row 15 and a red
      2-pixel horizontal segment sits one cell left of it on row 15. Original
      logic could not even consider horizontal repair. New logic reads "horizontal"
      from the marker's shape and row 15 from the marker's position.

b) Complexity calibration:
  - Only two marker orientations exist, so the rule class stays compact.
  - Every generated grid mixes vertical-marked and horizontal-marked sprites, so
    the D2 change is visible in each sample instead of being a rare corner case.
  - Marker color is fixed and always erased in the output, giving repeated
    evidence that red is metadata rather than payload.
  - Marker length is kept short (2 cells) so the visual language is consistent;
    the only meaningful difference is orientation.

c) Hidden assumption audit against the original main():
  - "all non-bg colors are payload" is no longer true: red is control metadata.
    The new solver must erase markers and exclude them from object extraction.
  - "one color layer == one reasoning unit" is no longer sufficient. The right
    unit is marker-object pairing, because the axis comes from a nearby marker.
  - "all symmetry is vertical" is false. Reflection must branch on marker
    orientation, using mirrored columns for vertical markers and mirrored rows
    for horizontal markers.
  - "axis must be inferred from noisy pixels" is false. Axis location is read
    from marker geometry, so generator and solver must agree on marker placement.
  - D5 originally used single-pixel markers too, but that would be ambiguous
    once orientation carries meaning. This combo therefore restricts markers to
    2-cell line segments only.
  - The explicit-marker design also removes half-integer axes: a one-cell-wide
    or one-cell-tall marker can only point cleanly to an integer row/column.

d) Placement legality:
  - Objects are still placed in separated slots on a 22x22 canvas.
  - Vertical markers live above or below their sprite with a reserved 1-cell gap.
  - Horizontal markers live left or right of their sprite with a reserved 1-cell
    gap.
  - Noise sampling blocks all marker and gap cells, so metadata never merges with
    payload and markers remain visually separate connected components.

e) Visual observability:
  - Across 2-4 examples, a human can see that vertical red segments correlate
    with left-right cleanup while horizontal red segments correlate with up-down
    cleanup.
  - Because red disappears in the output, it is natural to infer that the red
    segments are instructions rather than part of the target pattern.

f) Visual cleanliness:
  - There is only one metadata color and it never overlaps sprites.
  - Objects remain far apart, and markers sit in a reserved adjacent lane.
  - Noise stays 4-adjacent to its own sprite, so each local scene remains easy
    to parse.

g) Change inferability:
  - No payload recoloring is used.
  - Color ids are never sorted or interpreted numerically. Only marker geometry
    matters, which is directly visible.

h) Quantified non-degeneration target:
  - The generator forces both marker orientations in every sample.
  - A baseline that behaves like the original task (vertical search, no marker
    semantics) should disagree with the combo output on most samples, because
    horizontal-marked sprites and removable red markers are present by design.

i) New solver assumptions and generator guarantees:
  - Assumption: each marker component is a straight 2-cell segment.
    Guarantee: the generator only draws 1x2 or 2x1 red segments.
  - Assumption: each sprite has exactly one nearby marker whose axis coordinate
    lies inside that sprite bbox in the relevant dimension.
    Guarantee: placement code constructs one marker per sprite after sprite bbox
    is fixed.
  - Assumption: each sprite remains one connected component even after noise is
    added.
    Guarantee: clean sprites are connected and all noise is 4-adjacent.
  - Assumption: marker-object matching by bbox-span + nearest gap is unique.
    Guarantee: slots are well separated and the marker gap to its own sprite is
    always 1, much smaller than any cross-object gap.
"""

import math
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import get_bg, label_mask, paste


GRID_SIZE = 22
MARK = 2
NUM_SPRITES = 3

# ── CFG: layout ──
LAYOUTS = [
    [(0, 0, 9, 8), (0, 13, 9, 21), (12, 4, 21, 17)],
    [(0, 1, 9, 9), (0, 12, 9, 20), (12, 3, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 34
HEIGHT_MIN = 2
HEIGHT_MAX = 6
ODD_SIDE_W_RANGE = range(1, 4)  # widths 3, 5, 7 only; explicit markers need integer axes
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: markers ──
MARKER_GAP = 1
MARKER_LENGTH = 2

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# ── CFG: axis distinctness ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retries ──
SPRITE_RETRIES = 220
GEN_RETRIES = 500

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _axis_scores(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> dict[int, int]:
    """Return reflection-partner counts for every axis in one direction."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    if direction == "vertical":
        for k in range(0, 2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < width and (r, cc) in pts:
                    score += 1
            scores[int(k)] = int(score)
    elif direction == "horizontal":
        for k in range(0, 2 * (height - 1) + 1):
            score = 0
            for r, c in pts:
                rr = int(k) - int(r)
                if 0 <= rr < height and (rr, c) in pts:
                    score += 1
            scores[int(k)] = int(score)
    else:
        raise ValueError(f"unknown direction: {direction}")
    return scores


def _best_axis(coords: list[tuple[int, int]], height: int, width: int) -> tuple[str, int, int]:
    """Pick the best axis over both directions; ties keep the earliest vertical candidate."""
    best_direction = "vertical"
    best_k = 0
    best_score = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if int(score) > best_score:
                best_score = int(score)
                best_direction = str(direction)
                best_k = int(k)
    return best_direction, best_k, best_score


def _best_other_score(
    coords: list[tuple[int, int]], height: int, width: int, true_direction: str, true_k: int
) -> int:
    """Highest axis score excluding the supplied true axis."""
    best = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if direction == true_direction and int(k) == int(true_k):
                continue
            if int(score) > best:
                best = int(score)
    return int(best)


def _paired_under_axis(
    pts: set[tuple[int, int]], r: int, c: int, height: int, width: int, direction: str, k: int
) -> bool:
    """Return True iff (r,c) has a same-color reflection partner under the chosen axis."""
    if direction == "vertical":
        cc = int(k) - int(c)
        return 0 <= cc < width and (int(r), int(cc)) in pts
    rr = int(k) - int(r)
    return 0 <= rr < height and (int(rr), int(c)) in pts


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    """L1-style gap between two inclusive bboxes."""
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


def _extract_objects(g: np.ndarray, bg: int) -> list[dict]:
    """Return non-marker object components as records."""
    obj_lab, obj_n = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=4)
    objects: list[dict] = []
    for lbl in range(1, int(obj_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(obj_lab == int(lbl))]
        if not coords:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        objects.append(
            {
                "id": int(lbl),
                "coords": coords,
                "color": int(next(iter(colors))),
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )
    return objects


def _extract_markers(g: np.ndarray) -> list[dict]:
    """Return valid marker components with decoded axis direction/position."""
    marker_lab, marker_n = label_mask(g == int(MARK), connectivity=4)
    markers: list[dict] = []
    for lbl in range(1, int(marker_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(marker_lab == int(lbl))]
        if not coords:
            continue
        rows = sorted({r for r, _c in coords})
        cols = sorted({c for _r, c in coords})
        if len(rows) == 1 and len(cols) == MARKER_LENGTH:
            markers.append(
                {
                    "id": int(lbl),
                    "direction": "horizontal",
                    "axis_pos": int(rows[0]),
                    "bbox": (int(rows[0]), int(cols[0]), int(rows[0]), int(cols[-1])),
                    "coords": coords,
                }
            )
        elif len(cols) == 1 and len(rows) == MARKER_LENGTH:
            markers.append(
                {
                    "id": int(lbl),
                    "direction": "vertical",
                    "axis_pos": int(cols[0]),
                    "bbox": (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[0])),
                    "coords": coords,
                }
            )
    return markers


def _match_marker_to_object(marker: dict, objects: list[dict], used_ids: set[int]) -> dict | None:
    """Match one marker to the unique nearby object consistent with its axis coordinate."""
    direction = str(marker["direction"])
    axis_pos = int(marker["axis_pos"])
    if direction == "vertical":
        candidates = [
            obj
            for obj in objects
            if obj["id"] not in used_ids and int(obj["bbox"][1]) <= axis_pos <= int(obj["bbox"][3])
        ]
        center_key = lambda rec: abs(axis_pos - ((int(rec["bbox"][1]) + int(rec["bbox"][3])) // 2))
    else:
        candidates = [
            obj
            for obj in objects
            if obj["id"] not in used_ids and int(obj["bbox"][0]) <= axis_pos <= int(obj["bbox"][2])
        ]
        center_key = lambda rec: abs(axis_pos - ((int(rec["bbox"][0]) + int(rec["bbox"][2])) // 2))
    if not candidates:
        return None
    return min(candidates, key=lambda rec: (_bbox_gap(marker["bbox"], rec["bbox"]), center_key(rec)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each sprite using the axis explicitly encoded by a nearby marker.

    Input structure:
    - One background color fills most of the 22x22 grid. Red (2) is reserved for external markers.
    - There are three non-red sprites, one connected component per sprite, and each has a few attached same-color noise pixels.
    - Every sprite has exactly one nearby red 2-cell segment: vertical segments indicate a vertical reflection axis, while horizontal segments indicate a horizontal reflection axis.

    Rule:
    - Find each red marker, decode its orientation, and read its row or column as the sprite's symmetry axis.
    - Pair that marker with the nearby sprite whose bounding box spans the marker's axis coordinate.
    - Remove every sprite pixel that lacks a same-color mirror counterpart across the marker-specified axis, and erase all red markers.

    Key insight:
    - The task is no longer "search for the best axis from the noisy shape". The marker already tells both which direction to reflect and where the axis lies, so the real work is marker-to-sprite association plus mirror-based denoising under that explicit instruction.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    objects = _extract_objects(g, bg)
    markers = _extract_markers(g)

    used_ids: set[int] = set()
    for marker in sorted(markers, key=lambda rec: (rec["bbox"][0], rec["bbox"][1])):
        obj = _match_marker_to_object(marker, objects, used_ids)
        if obj is None:
            continue
        used_ids.add(int(obj["id"]))

        direction = str(marker["direction"])
        axis_pos = int(marker["axis_pos"])
        k = int(2 * axis_pos)
        pts = {(int(r), int(c)) for r, c in obj["coords"]}
        color = int(obj["color"])
        for r, c in obj["coords"]:
            if direction == "vertical":
                cc = int(k) - int(c)
                ok = 0 <= cc < width and int(g[int(r), int(cc)]) == color
            else:
                rr = int(k) - int(r)
                ok = 0 <= rr < height and int(g[int(rr), int(c)]) == color
            if not ok:
                out[int(r), int(c)] = int(bg)

    return out


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Pick (h, w, target_count, center_count) for an odd-width vertically symmetric sprite."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, int(max_h))
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = int(2 * side_w + 1)
                if w > int(max_w):
                    continue
                area = int(h * w)
                lo = int(math.ceil(DENSITY_LO * area))
                hi = int(math.floor(DENSITY_HI * area))
                if not (lo <= target_count <= hi):
                    continue
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((int(h), int(w), int(target_count), int(center_count)))
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _h, _w, target_count, _c in options})
    target_count = int(random.choice(counts))
    target_options = [opt for opt in options if int(opt[2]) == target_count]
    weights = [1.0 / (int(h) * int(w)) for h, w, _t, _c in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left-half cells while touching every row, which helps connectivity."""
    cells: set[tuple[int, int]] = set()
    inner_anchor_row = random.randrange(int(h))
    for r in range(int(h)):
        if r == 0:
            c = 0
        elif int(side_w) > 1 and r == inner_anchor_row:
            c = int(side_w) - 1
        else:
            c = random.randrange(int(side_w))
        cells.add((int(r), int(c)))

    remaining = int(left_count) - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")
    pool = [(int(r), int(c)) for r in range(int(h)) for c in range(int(side_w)) if (int(r), int(c)) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    """Return which rows receive the center-column pixels."""
    rows = np.zeros(int(h), dtype=bool)
    rows[random.sample(range(int(h)), int(center_count))] = True
    return rows


def _vertical_candidate(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build one perfectly vertically symmetric odd-width candidate sprite."""
    h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
    side_w = int(w // 2)
    left_count = int((target_count - center_count) // 2)
    left = _left_cells(h, side_w, left_count)
    centers = _center_rows(h, center_count)

    sprite = np.zeros((int(h), int(w)), dtype=int)
    for r, c in left:
        mirror_c = int(w - 1 - c)
        sprite[int(r), int(c)] = int(color)
        sprite[int(r), int(mirror_c)] = int(color)
    sprite[centers, int(side_w)] = int(color)
    return sprite, int(w - 1)


def _build_clean_sprite(color: int, max_h: int, max_w: int, direction: str) -> tuple[np.ndarray, int]:
    """Build a connected clean sprite whose unique best axis matches the requested direction."""
    for _ in range(SPRITE_RETRIES):
        if direction == "vertical":
            sprite, true_k = _vertical_candidate(color, max_h, max_w)
        elif direction == "horizontal":
            base, true_k = _vertical_candidate(color, max_w, max_h)
            sprite = base.T.copy()
        else:
            raise ValueError(f"unknown direction: {direction}")

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        if len(coords) == 0:
            continue
        _lab, n = label_mask(sprite == int(color), connectivity=4)
        if int(n) != 1:
            continue
        best_direction, best_k, best_score = _best_axis(coords, *sprite.shape)
        if best_direction != direction or int(best_k) != int(true_k):
            continue
        if int(best_score) != len(coords):
            continue
        other_score = _best_other_score(coords, *sprite.shape, direction, true_k)
        if int(best_score) - int(other_score) < SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, int(true_k)
    raise ValueError("failed to build clean sprite")


def _sample_orientations() -> list[str]:
    """Guarantee that both axis directions appear in every grid."""
    horizontal_count = int(random.choice([1, 2]))
    horizontal_ids = set(random.sample(range(NUM_SPRITES), horizontal_count))
    return ["horizontal" if i in horizontal_ids else "vertical" for i in range(NUM_SPRITES)]


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict], np.ndarray] | None:
    """Place clean sprites, their markers, and the blocked metadata lanes."""
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = map(int, sprite.shape)
        slot_h = int(r1s - r0s + 1)
        slot_w = int(c1s - c0s + 1)
        direction = str(spec["axis_dir"])
        side = str(spec["marker_side"])

        if direction == "vertical":
            bundle_h = int(h + MARKER_GAP + MARKER_LENGTH)
            if bundle_h > slot_h or w > slot_w:
                return None
            top = random.randint(int(r0s), int(r1s - bundle_h + 1))
            c0 = random.randint(int(c0s), int(c1s - w + 1))
            axis_pos = int(c0 + w // 2)
            if side == "above":
                marker_r0 = int(top)
                sprite_r0 = int(top + MARKER_LENGTH + MARKER_GAP)
                gap_cells = [(int(sprite_r0 - 1), int(axis_pos))]
            else:
                sprite_r0 = int(top)
                marker_r0 = int(sprite_r0 + h + MARKER_GAP)
                gap_cells = [(int(sprite_r0 + h), int(axis_pos))]
            marker_cells = [(int(marker_r0 + i), int(axis_pos)) for i in range(MARKER_LENGTH)]
            paste(out, sprite, int(sprite_r0), int(c0), transparent=0)
            spec["true_k"] = int(2 * axis_pos)
            spec["axis_pos"] = int(axis_pos)
            spec["sprite_r0"] = int(sprite_r0)
            spec["sprite_c0"] = int(c0)
        else:
            bundle_w = int(w + MARKER_GAP + MARKER_LENGTH)
            if h > slot_h or bundle_w > slot_w:
                return None
            r0 = random.randint(int(r0s), int(r1s - h + 1))
            left = random.randint(int(c0s), int(c1s - bundle_w + 1))
            axis_pos = int(r0 + h // 2)
            if side == "left":
                marker_c0 = int(left)
                sprite_c0 = int(left + MARKER_LENGTH + MARKER_GAP)
                gap_cells = [(int(axis_pos), int(sprite_c0 - 1))]
            else:
                sprite_c0 = int(left)
                marker_c0 = int(sprite_c0 + w + MARKER_GAP)
                gap_cells = [(int(axis_pos), int(sprite_c0 + w))]
            marker_cells = [(int(axis_pos), int(marker_c0 + i)) for i in range(MARKER_LENGTH)]
            paste(out, sprite, int(r0), int(sprite_c0), transparent=0)
            spec["true_k"] = int(2 * axis_pos)
            spec["axis_pos"] = int(axis_pos)
            spec["sprite_r0"] = int(r0)
            spec["sprite_c0"] = int(sprite_c0)

        for r, c in marker_cells + gap_cells:
            blocked[int(r), int(c)] = True

        spec["marker_cells"] = marker_cells
        spec["gap_cells"] = gap_cells
        placed.append(spec)

    return out, placed, blocked


def _noise_candidates(
    clean_mask: np.ndarray,
    direction: str,
    true_k: int,
    blocked: np.ndarray,
    free_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Find valid 4-adjacent background cells whose mirrors are absent on the clean shape."""
    height, width = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask
    adj &= blocked == 0
    adj &= free_mask

    if direction == "vertical":
        cols = np.arange(width)
        mirror_cols = int(true_k) - cols
        on_axis = mirror_cols == cols
        adj[:, on_axis] = False
        valid = (mirror_cols >= 0) & (mirror_cols < width) & ~on_axis
        for c in np.where(valid)[0]:
            adj[:, int(c)] &= ~clean_mask[:, int(mirror_cols[int(c)])]
    elif direction == "horizontal":
        rows = np.arange(height)
        mirror_rows = int(true_k) - rows
        on_axis = mirror_rows == rows
        adj[on_axis, :] = False
        valid = (mirror_rows >= 0) & (mirror_rows < height) & ~on_axis
        for r in np.where(valid)[0]:
            adj[int(r), :] &= ~clean_mask[int(mirror_rows[int(r)]), :]
    else:
        raise ValueError(f"unknown direction: {direction}")

    rows_arr, cols_arr = np.where(adj)
    return list(zip(rows_arr.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict, blocked: np.ndarray, bg: int) -> bool:
    """Add local, unpaired noise around one sprite while preserving the intended marked axis."""
    color = int(spec["color"])
    direction = str(spec["axis_dir"])
    true_k = int(spec["true_k"])
    clean_mask = out == int(color)
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)
    if clean_count == 0:
        return False

    lo = max(1, int(math.ceil(NOISE_FRAC_LO * clean_count)))
    hi = min(NOISE_MAX, int(math.floor(NOISE_FRAC_HI * clean_count)))
    if lo > hi:
        return False

    noise_target = int(random.randint(lo, hi))
    candidates = _noise_candidates(clean_mask, direction, true_k, blocked, inp == int(bg))
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
        if not viable:
            return False

        scored: list[tuple[int, float, tuple[int, int]]] = []
        for cell in viable:
            alt = _best_other_score(current_coords + [cell], GRID_SIZE, GRID_SIZE, direction, true_k)
            scored.append((int(alt), random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = int(scored[0][0])
        pool = [cell for alt, _rnd, cell in scored if int(alt) <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        r, c = cell
        forbidden.add((int(r), int(c)))
        if direction == "vertical":
            mirror = (int(r), int(true_k - c))
            if 0 <= int(mirror[1]) < GRID_SIZE:
                forbidden.add(mirror)
        else:
            mirror = (int(true_k - r), int(c))
            if 0 <= int(mirror[0]) < GRID_SIZE:
                forbidden.add(mirror)

    best_direction, best_k, best_score = _best_axis(current_coords, GRID_SIZE, GRID_SIZE)
    other_score = _best_other_score(current_coords, GRID_SIZE, GRID_SIZE, direction, true_k)
    if best_direction != direction or int(best_k) != int(true_k):
        return False
    if int(best_score) != clean_count:
        return False
    if int(best_score) - int(other_score) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[int(r), int(c)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Output-first generation with marker-guided axis direction and position."""
    palette = [c for c in range(10) if c != MARK]
    bg, *obj_colors = random.sample(palette, NUM_SPRITES + 1)
    slots = random.choice(LAYOUTS)
    orientations = _sample_orientations()

    sprite_specs: list[dict] = []
    try:
        for color, slot, direction in zip(obj_colors, slots, orientations):
            slot_h = int(slot[2] - slot[0] + 1)
            slot_w = int(slot[3] - slot[1] + 1)
            if direction == "vertical":
                max_h = int(slot_h - MARKER_GAP - MARKER_LENGTH)
                max_w = int(slot_w)
                marker_side = random.choice(["above", "below"])
            else:
                max_h = int(slot_h)
                max_w = int(slot_w - MARKER_GAP - MARKER_LENGTH)
                marker_side = random.choice(["left", "right"])
            sprite, axis_k_local = _build_clean_sprite(int(color), max_h, max_w, direction)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "axis_dir": str(direction),
                    "axis_k_local": int(axis_k_local),
                    "marker_side": str(marker_side),
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(int(bg), sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs, blocked = result

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec, blocked, int(bg)):
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
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
