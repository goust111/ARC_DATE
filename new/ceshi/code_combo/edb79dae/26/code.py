# Combo variant of edb79dae (index 26)
# combo_spec: D4/2+D12/1
# Name: 色块簇目标+最近匹配

# Design analysis for the combo variant
# ------------------------------------
# a) Concrete behavior and trace
#    - Original task:
#      The frame interior contains solid color blocks. Every inside block of color src receives the
#      outside sprite of the same color, and that sprite is then recolored by the outside legend
#      mapping src -> tgt.
#    - This combo:
#      Each inside target is now a same-color connected cluster with area > 1 instead of a solid
#      placement square, and a source color can still appear on multiple interior targets. The solver
#      must compute every cluster centroid, compare those centroids with the outside sprite center,
#      keep only the nearest same-color cluster, then place the sprite with its center aligned to the
#      chosen centroid and recolor it through the legend.
#    - Small trace:
#      Suppose color 2 appears on two adjacent clusters inside the frame. Their areas and shapes are
#      different, so their true centroids are shifted differently inside their local 5x5 target zones.
#      A sprite placed outside the frame can therefore be closer to the left cluster by centroid but
#      closer to the right cluster by bbox-center reasoning. The combo output keeps only the
#      centroid-nearest target and clears the other one.
#
# b) Complexity calibration
#    - Every generated sample repeats every source color at least twice, so the D12/1 "nearest only"
#      rule is active in all examples.
#    - The interior clusters come from a small library of asymmetric 6-cell templates with integral
#      centroids. This keeps the visual vocabulary small enough to learn while still making the D4/2
#      change obvious.
#    - One designated source color is always assigned to a forced left/right pair whose centroid
#      winner differs from the bbox-center winner. This prevents the generator from accidentally
#      collapsing back to the simpler nearest-by-box variant.
#
# c) Hidden assumptions from the original solver that break
#    - Placeholder bbox size no longer identifies the sprite, because the inside targets are small
#      irregular clusters instead of full sprite-sized blocks.
#    - Bbox top-left alignment is invalid. Placement must now be derived from the target centroid.
#    - "Stamp at every same-color target" is invalid. Repeated colors require a nearest-target
#      selection step before any stamping happens.
#    - The outside legend logic still works, but sprite recovery must merge all non-legend components
#      of a source color into one odd-sized sprite bbox.
#
# d) Placement legality
#    - The frame interior uses a coarse lattice with explicit background gaps, so clusters never touch
#      each other or the border even when their shapes reach two cells away from the local zone center.
#    - Outside sprites and legends are placed only in reserve-checked free positions with padding, so
#      they do not overlap one another or the frame.
#    - All cluster templates have integral centroids compatible with centering a 3x3 sprite exactly on
#      a grid cell.
#
# e) Visual observability
#    - The D12 effect is obvious because repeated same-color clusters reduce to one output sprite.
#    - The D4 effect is visible because the targets are irregular color blobs, not solid placement
#      blocks, and at least one color in every sample truly requires centroid reasoning.
#    - No hidden ordering uses raw color ids; all choices are explained by visible geometry plus the
#      visible legend.
#
# f) Visual cleanliness
#    - The frame interior remains sparse and grid-like, so the irregular clusters stay readable.
#    - Outside sprites and legends live in the background-only region around the frame, which keeps
#      matching and recoloring cues spatially separated.
#
# g) New solver assumptions reviewed against the generator
#    - Each source color has exactly one outside sprite after legend components are excluded.
#    - Each repeated color has a unique centroid-nearest cluster with a minimum distance gap.
#    - At least one color per sample also has centroid-nearest != bbox-center-nearest.
#    - All target clusters use templates whose centroids allow exact center alignment for a 3x3
#      sprite, and the generator builds only such inputs.

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _component_centroid2(pts: list[tuple[int, int]]) -> tuple[int, int]:
    area = len(pts)
    r2_num = 2 * sum(int(r) for r, _ in pts)
    c2_num = 2 * sum(int(c) for _, c in pts)
    assert r2_num % area == 0 and c2_num % area == 0, "cluster centroid must lie on an integer cell"
    return r2_num // area, c2_num // area


def _offset_centroid2(offsets: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return _component_centroid2([(int(r), int(c)) for r, c in offsets])


def _bbox_center2(b: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = b
    return r0 + r1, c0 + c1


def _center_manhattan2(a2: tuple[int, int], b2: tuple[int, int]) -> int:
    return abs(int(a2[0]) - int(b2[0])) + abs(int(a2[1]) - int(b2[1]))


def _sprite_top_left_from_centroid2(
    centroid2: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    sh, sw = shape
    assert sh % 2 == 1 and sw % 2 == 1, f"sprite shape must be odd, got {shape}"
    assert (centroid2[0] - (sh - 1)) % 2 == 0 and (centroid2[1] - (sw - 1)) % 2 == 0, (
        f"centroid {centroid2} cannot center sprite shape {shape}"
    )
    return (centroid2[0] - (sh - 1)) // 2, (centroid2[1] - (sw - 1)) // 2


def _distance_profile(
    entries: list[dict[str, object]],
    sprite_center2: tuple[int, int],
    *,
    use_bbox_center: bool,
) -> list[tuple[int, int, int, dict[str, object]]]:
    scored = []
    for entry in entries:
        ref_center2 = _bbox_center2(entry["abs_bbox"]) if use_bbox_center else entry["abs_centroid2"]
        bbox_rel = entry["rel_bbox"]
        dist = _center_manhattan2(sprite_center2, ref_center2)
        scored.append((dist, bbox_rel[0], bbox_rel[1], entry))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored


def _choose_nearest_cluster(entries: list[dict[str, object]], sprite_center2: tuple[int, int]) -> dict[str, object]:
    return _distance_profile(entries, sprite_center2, use_bbox_center=False)[0][3]


def _cluster_bbox(center: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [center[0] + dr for dr, _ in offsets]
    cs = [center[1] + dc for _, dc in offsets]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace repeated target clusters by one nearest recolored sprite inside the frame.

    Input structure:
    - A gray rectangular frame encloses several small same-color connected clusters instead of solid
      placement blocks, and a source color can appear on multiple interior clusters.
    - Outside the frame there is one sprite template per source color plus the usual overlapping
      two-color legend that tells which final color each source color should become.

    Rule:
    - Find the frame, group the interior clusters by color, and compute each cluster centroid.
    - For each source color, recover the matching outside sprite and legend mapping, compare the
      outside sprite center to all same-color cluster centroids, and keep only the nearest cluster.
    - Center that sprite on the chosen cluster centroid, recolor it by the legend, and clear all
      non-chosen same-color clusters to background.

    Key insight:
    - Color still selects the sprite shape and the recolor legend, but destination selection is now a
      two-step geometric problem: irregular target clusters require centroid reasoning, and repeated
      same-color targets are filtered down to the unique nearest one before stamping.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no target colors found inside the frame"

    clusters_by_color: dict[int, list[dict[str, object]]] = {col: [] for col in sorted(placeholder_colors)}
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            bbox_rel = bbox_coords(np.array(pts_tuples, dtype=int))
            centroid2_rel = _component_centroid2(pts_tuples)
            clusters_by_color[col].append(
                {
                    "rel_bbox": bbox_rel,
                    "abs_bbox": (fr0 + bbox_rel[0], fc0 + bbox_rel[1], fr0 + bbox_rel[2], fc0 + bbox_rel[3]),
                    "rel_centroid2": centroid2_rel,
                    "abs_centroid2": (2 * fr0 + centroid2_rel[0], 2 * fc0 + centroid2_rel[1]),
                }
            )

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts_tuples),
                }
            )
            cid += 1

    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            if ci["col"] not in sub_colors or cj["col"] not in sub_colors:
                continue

            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]

            assert src in placeholder_colors, f"legend source color {src} is not used inside the frame"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match frame colors {sorted(placeholder_colors)}"
    )

    sprite_data: dict[int, dict[str, object]] = {}
    for src_col in sorted(placeholder_colors):
        sprite_pts = set()
        for comp in comps:
            if comp["col"] == src_col and comp["id"] not in legend_ids:
                sprite_pts.update(comp["pts"])
        assert sprite_pts, f"no outside sprite found for color {src_col}"
        coords_arr = np.array(sorted(sprite_pts), dtype=int)
        sr0, sc0, sr1, sc1 = bbox_coords(coords_arr)
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        assert sh == sw and sh % 2 == 1, f"sprite bbox for color {src_col} must be odd square, got {(sh, sw)}"
        coords = tuple(sorted((int(r - sr0), int(c - sc0)) for r, c in sprite_pts))
        sprite_data[src_col] = {
            "coords": coords,
            "shape": (sh, sw),
            "center2": (sr0 + sr1, sc0 + sc1),
        }

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col in sorted(placeholder_colors):
        entries = clusters_by_color[src_col]
        assert entries, f"no interior cluster found for color {src_col}"
        chosen = _choose_nearest_cluster(entries, sprite_data[src_col]["center2"])
        tgt_col = int(color_map[src_col])
        coords = sprite_data[src_col]["coords"]
        sh, sw = sprite_data[src_col]["shape"]
        off_r, off_c = _sprite_top_left_from_centroid2(chosen["rel_centroid2"], (sh, sw))
        assert off_r >= 1 and off_c >= 1, f"sprite for color {src_col} would cross the frame border"
        assert off_r + sh <= h - 1 and off_c + sw <= w - 1, (
            f"sprite for color {src_col} does not fit at centroid {chosen['rel_centroid2']}"
        )
        for dr, dc in coords:
            out[off_r + dr, off_c + dc] = tgt_col

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N = 3
GRID_ROWS_RANGE = (2, 4)  # rows = 2 or 3
GRID_COLS = 3
MAX_EMPTY_SLOTS = 1
NUM_COLORS_RANGE = (2, 4)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
CANVAS_H_RANGE = (28, 31)
CANVAS_W_RANGE = (28, 31)
PATCH_PADDING = 1
GEN_RETRIES = 300
MIN_NEAREST_GAP = 2
PITCH = 6

TEMPLATE_UP = ((-2, -1), (-2, 0), (-2, 1), (-1, 0), (0, 0), (1, 0))
TEMPLATE_DOWN = ((-1, 0), (0, 0), (1, 0), (2, -1), (2, 0), (2, 1))
TEMPLATE_LEFT = ((-1, -2), (0, -2), (0, -1), (0, 0), (0, 1), (1, -2))
TEMPLATE_RIGHT = ((-1, 2), (0, -1), (0, 0), (0, 1), (0, 2), (1, 2))
TEMPLATE_CENTER = ((-2, 0), (-1, 0), (0, 0), (1, -1), (1, 0), (1, 1))
TEMPLATE_UP_LEFT = ((-2, -2), (-2, -1), (-2, 0), (-1, -1), (0, -1), (1, -1))
TEMPLATE_DOWN_RIGHT = ((-1, 1), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2))
CLUSTER_TEMPLATES: tuple[tuple[tuple[int, int], ...], ...] = (
    TEMPLATE_UP,
    TEMPLATE_DOWN,
    TEMPLATE_LEFT,
    TEMPLATE_RIGHT,
    TEMPLATE_CENTER,
    TEMPLATE_UP_LEFT,
    TEMPLATE_DOWN_RIGHT,
)


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    mask = np.zeros((n, n), dtype=bool)
    c = random.randrange(n)
    for r in range(n):
        c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(n), :] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        r, c = random.choice(frontier)
        mask[r, c] = True
    return mask


def _distinct_sprite_masks(colors: list[int]) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    seen = set()
    for src in colors:
        for _ in range(100):
            mask = _sprite_mask(SPRITE_N)
            key = tuple(map(tuple, mask.astype(int)))
            if key not in seen:
                seen.add(key)
                masks[src] = mask
                break
        else:
            return {}
    return masks


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _choose_structure() -> tuple[int, int, int]:
    rows = random.choice(list(range(*GRID_ROWS_RANGE)))
    cols = GRID_COLS
    empty_count = 0 if rows == 2 else random.randint(0, MAX_EMPTY_SLOTS)
    return rows, cols, empty_count


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int, required_filled: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        if required_filled & empty:
            continue
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    assert valid, "no valid empty-slot pattern leaves the required pair filled"
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _repeat_counts(total: int, k: int) -> list[int]:
    counts = [2] * k
    extras = total - 2 * k
    choices = list(range(1, k)) if k > 1 else [0]
    for _ in range(extras):
        counts[random.choice(choices)] += 1
    return counts


def _mark_occupied(reserve: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    reserve[r : r + h, c : c + w] = FRAME_COL


def _can_place(reserve: np.ndarray, r: int, c: int, h: int, w: int, bg: int) -> bool:
    rh, rw = reserve.shape
    if r < 0 or c < 0 or r + h > rh or c + w > rw:
        return False
    rr0 = max(0, r - PATCH_PADDING)
    cc0 = max(0, c - PATCH_PADDING)
    rr1 = min(rh, r + h + PATCH_PADDING)
    cc1 = min(rw, c + w + PATCH_PADDING)
    return bool(np.all(reserve[rr0:rr1, cc0:cc1] == bg))


def _enumerate_free_positions(
    reserve: np.ndarray, patch_shape: tuple[int, int], bg: int
) -> list[tuple[int, int]]:
    ph, pw = patch_shape
    rh, rw = reserve.shape
    return [(r, c) for r in range(rh - ph + 1) for c in range(rw - pw + 1) if _can_place(reserve, r, c, ph, pw, bg)]


def _place_patch_at(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    r: int,
    c: int,
    bg: int,
    *,
    transparent: bool,
) -> None:
    _mark_occupied(reserve, r, c, patch.shape[0], patch.shape[1])
    paste(grid, patch, r, c, transparent=bg if transparent else None)


def _place_patch_anywhere(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    candidates = _enumerate_free_positions(reserve, patch.shape, bg)
    if not candidates:
        return False
    r, c = random.choice(candidates)
    _place_patch_at(grid, reserve, patch, r, c, bg, transparent=transparent)
    return True


def _build_cluster_entry(
    fr0: int,
    fc0: int,
    rel_center: tuple[int, int],
    offsets: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    centroid_shift2 = _offset_centroid2(offsets)
    rel_centroid2 = (2 * rel_center[0] + centroid_shift2[0], 2 * rel_center[1] + centroid_shift2[1])
    rel_bbox = _cluster_bbox(rel_center, offsets)
    abs_bbox = (fr0 + rel_bbox[0], fc0 + rel_bbox[1], fr0 + rel_bbox[2], fc0 + rel_bbox[3])
    return {
        "rel_bbox": rel_bbox,
        "abs_bbox": abs_bbox,
        "rel_centroid2": rel_centroid2,
        "abs_centroid2": (2 * fr0 + rel_centroid2[0], 2 * fc0 + rel_centroid2[1]),
        "rel_sprite_tl": _sprite_top_left_from_centroid2(rel_centroid2, (SPRITE_N, SPRITE_N)),
        "area": len(offsets),
    }


def _select_sprite_position(
    reserve: np.ndarray,
    entries: list[dict[str, object]],
    bg: int,
    *,
    require_bbox_disagreement: bool,
) -> tuple[tuple[int, int], dict[str, object]] | None:
    candidates = _enumerate_free_positions(reserve, (SPRITE_N, SPRITE_N), bg)
    random.shuffle(candidates)
    scored = []
    fallback_scored = []
    for r, c in candidates:
        bbox = (r, c, r + SPRITE_N - 1, c + SPRITE_N - 1)
        sprite_center2 = _bbox_center2(bbox)
        centroid_profile = _distance_profile(entries, sprite_center2, use_bbox_center=False)
        assert len(centroid_profile) >= 2, "generator expects repeated targets per color"
        gap = centroid_profile[1][0] - centroid_profile[0][0]
        if gap < MIN_NEAREST_GAP:
            continue
        bbox_profile = _distance_profile(entries, sprite_center2, use_bbox_center=True)
        disagree = centroid_profile[0][3] is not bbox_profile[0][3]
        best_dist = centroid_profile[0][0]
        record = (-gap, best_dist, 0 if disagree else 1, r, c, centroid_profile[0][3])
        fallback_scored.append(record)
        if require_bbox_disagreement and not disagree:
            continue
        scored.append(record)
    if not scored and require_bbox_disagreement:
        scored = fallback_scored
    if not scored:
        return None
    _, _, _, r, c, chosen_entry = min(scored)
    return (r, c), chosen_entry


def _gen_attempt() -> np.ndarray | None:
    rows, cols, empty_count = _choose_structure()
    total = rows * cols - empty_count
    max_k = min(NUM_COLORS_RANGE[1], total // 2)
    if max_k < NUM_COLORS_RANGE[0]:
        return None

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(NUM_COLORS_RANGE[0], max_k)
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = _distinct_sprite_masks(src_cols)
    if len(masks) != k:
        return None

    special_row = random.randrange(rows)
    special_pair = ((special_row, 0), (special_row, 1)) if random.random() < 0.5 else ((special_row, 1), (special_row, 2))
    _, filled_slots = _choose_empty_slots(rows, cols, empty_count, set(special_pair))

    counts = _repeat_counts(total, k)
    special_color = src_cols[0]
    remaining_colors = []
    for src, count in zip(src_cols, counts):
        use = count if src != special_color else count - 2
        remaining_colors.extend([src] * use)
    random.shuffle(remaining_colors)

    slot_to_color: dict[tuple[int, int], int] = {special_pair[0]: special_color, special_pair[1]: special_color}
    for slot in filled_slots:
        if slot in slot_to_color:
            continue
        slot_to_color[slot] = remaining_colors.pop()
    assert not remaining_colors, "every color count must be assigned exactly once"

    template_by_slot: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    template_by_slot[special_pair[0]] = TEMPLATE_LEFT
    template_by_slot[special_pair[1]] = TEMPLATE_RIGHT
    for slot in slot_to_color:
        if slot not in template_by_slot:
            template_by_slot[slot] = random.choice(CLUSTER_TEMPLATES)

    frame_h = rows * PITCH + 3
    frame_w = cols * PITCH + 3
    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)

    canvas_options = [(h, w) for h in range(*CANVAS_H_RANGE) for w in range(*CANVAS_W_RANGE) if h >= frame_h and w >= frame_w]
    if not canvas_options:
        return None
    h, w = random.choice(canvas_options)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    corners = [(0, 0), (0, w - frame_w), (h - frame_h, 0), (h - frame_h, w - frame_w)]
    fr0, fc0 = random.choice(corners)
    paste(grid, input_frame, fr0, fc0)
    _mark_occupied(reserve, fr0, fc0, frame_h, frame_w)

    cluster_entries_by_color: dict[int, list[dict[str, object]]] = {src: [] for src in src_cols}
    for (rr, cc), src in slot_to_color.items():
        rel_tl = (3 + rr * PITCH, 3 + cc * PITCH)
        rel_center = (rel_tl[0] + 1, rel_tl[1] + 1)
        offsets = template_by_slot[(rr, cc)]
        for dr, dc in offsets:
            input_frame[rel_center[0] + dr, rel_center[1] + dc] = src
        cluster_entries_by_color[src].append(_build_cluster_entry(fr0, fc0, rel_center, offsets))

    paste(grid, input_frame, fr0, fc0)

    chosen_entry_by_color: dict[int, dict[str, object]] = {}
    sprite_order = [special_color] + [src for src in src_cols if src != special_color]
    for src in sprite_order:
        pick = _select_sprite_position(
            reserve,
            cluster_entries_by_color[src],
            bg,
            require_bbox_disagreement=(src == special_color),
        )
        if pick is None:
            return None
        (r, c), chosen_entry = pick
        sprite_patch = _colored_sprite(masks[src], src, bg)
        _place_patch_at(grid, reserve, sprite_patch, r, c, bg, transparent=True)
        chosen_entry_by_color[src] = chosen_entry

    for src, tgt in zip(src_cols, tgt_cols):
        if not _place_patch_anywhere(grid, reserve, _legend(src, tgt), bg, transparent=False):
            return None

    for src, entry in chosen_entry_by_color.items():
        r0, c0 = entry["rel_sprite_tl"]
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae combo input")
