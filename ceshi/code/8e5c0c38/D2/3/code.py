# Variant of 8e5c0c38
# Dimension: D2 — 对称轴方向 (transform)
# Change: 仅搜索垂直对称轴（遍历所有可能的垂直轴位置，选择得分最高的轴）...
#         → 使用外部标记指示对称轴方向

from __future__ import annotations

import itertools
import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, label_mask, paste

"""
design_analysis:
  a_specific_behavior:
    input_visual:
      - 22x22 grid, one background color.
      - Exactly 3 noisy non-background sprites, each with its own color.
      - Red (2) is reserved as an external marker color and never used by sprites or bg.
      - Each sprite has one detached red line nearby:
        - horizontal red line => this sprite should be denoised under a vertical axis
        - vertical red line => this sprite should be denoised under a horizontal axis
      - Output removes both the noise pixels and the red markers, keeping only the clean symmetric sprites.
    flow_delta_example:
      - Original logic: take a horizontally symmetric sprite, still search only vertical axes, so the kept pixels are governed by the best vertical pairing score.
      - Variant logic: first read the nearby red marker; if it is vertical, switch the search family to horizontal axes, then remove only pixels lacking a row-mirror counterpart.
      - Core difference: axis direction is no longer inferred from the sprite alone; it is decoded from a separate symbol and only the axis position is searched.
  b_complexity_calibration:
    - Keep just two marker categories (horizontal-line, vertical-line). This is enough to be learnable from 2-4 examples.
    - Every generated grid contains both axis directions (1 or 2 horizontal-axis sprites, remainder vertical-axis sprites), so the semantic contrast repeats inside each sample.
    - Marker placement side may vary (top/bottom or left/right), but its orientation is the only stable cue. This makes the D2 change visible instead of degenerating into a fixed layout trick.
  c_implicit_assumption_audit:
    - Original `main()` iterated per non-bg color and assumed every such color was a denoise target. That is now false because red is a marker, not a sprite.
    - Original axis search loop assumed the search family was always vertical. That assumption is the main D2 change and must become conditional on decoded marker orientation.
    - Original output started as a copy of the input and only removed unpaired sprite pixels. That would incorrectly preserve red markers in the variant, so markers must be erased up front.
    - Original per-color processing was safe because each non-bg color corresponded to exactly one sprite. The generator keeps that invariant so color-layer denoising still works.
  d_placement_legality:
    - The grid is still split into 3 large slots to prevent overlap and ambiguous nearest-marker matches.
    - A sprite and its marker are placed in the same slot with a one-cell bg gap, so they are close enough to pair but remain visually distinct.
    - Vertical-axis sprites reserve two rows for an above/below horizontal marker; horizontal-axis sprites reserve two columns for a left/right vertical marker.
    - Noise is forbidden from entering the marker-gap strip, preventing accidental visual merging.
  e_visual_observability:
    - A human can compare examples and see that the red segment is removed in the output and that its orientation predicts whether the sprite is cleaned left-right or up-down.
    - The solver still searches axis position from the sprite itself, so the marker is a semantic cue, not an exact ruler; this keeps the task close to the original while making D2 observable.
  f_visual_cleanliness:
    - Markers are short straight lines of length 3, always detached, and never overlap sprites.
    - Objects remain in separated regions, so no area contains multiple unrelated structures layered together.
  g_color_inference:
    - No color remapping is introduced.
    - Only one special color is added: red consistently means marker, not payload.
  solver_generator_contract:
    - `main()` assumes exactly one straight red-marker component per sprite and pairs markers to sprite colors by minimum-total center distance.
    - The generator enforces equal counts, slot-separated placement, and strong marker proximity so that this pairing is stable.
    - `main()` also assumes sprites are not perfectly symmetric in both axis families; the generator rejects cross-axis-perfect sprites to keep the marker meaningful.
"""

MARK = 2

# --------------- solver ---------------


def _marker_orientation(coords: list[tuple[int, int]]) -> str:
    rows = {int(r) for r, _c in coords}
    cols = {int(c) for _r, c in coords}
    if len(rows) == 1 and len(cols) >= 2:
        return "vertical"
    if len(cols) == 1 and len(rows) >= 2:
        return "horizontal"
    raise ValueError("marker must be a straight line")


def _bbox_center(coords: np.ndarray) -> tuple[float, float]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return float(rs.min() + rs.max()) / 2.0, float(cs.min() + cs.max()) / 2.0


def _mirror_cell(axis_dir: str, k: int, r: int, c: int) -> tuple[int, int]:
    if axis_dir == "vertical":
        return int(r), int(k) - int(c)
    return int(k) - int(r), int(c)


def _axis_scores(coords: list[tuple[int, int]], axis_dir: str, height: int, width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    limit = width if axis_dir == "vertical" else height
    for k in range(0, 2 * (limit - 1) + 1):
        score = 0
        for r, c in pts:
            rr, cc = _mirror_cell(axis_dir, k, r, c)
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) in pts:
                score += 1
        scores[int(k)] = int(score)
    return scores


def _best_axis(coords: list[tuple[int, int]], axis_dir: str, height: int, width: int) -> tuple[int, int]:
    best_score = -1
    best_k = 0
    for k, score in _axis_scores(coords, axis_dir, height, width).items():
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return int(best_k), int(best_score)


def _assign_markers_to_colors(
    color_pts: dict[int, np.ndarray], marker_comps: list[dict]
) -> dict[int, dict]:
    colors = sorted(color_pts)
    assert len(colors) == len(marker_comps), "each sprite must have one marker"

    color_centers = {col: _bbox_center(pts) for col, pts in color_pts.items()}
    marker_centers = [comp["center"] for comp in marker_comps]

    best_perm: tuple[int, ...] | None = None
    best_cost: float | None = None
    for perm in itertools.permutations(range(len(marker_comps))):
        cost = 0.0
        for col, marker_idx in zip(colors, perm):
            cr, cc = color_centers[int(col)]
            mr, mc = marker_centers[int(marker_idx)]
            cost += abs(cr - mr) + abs(cc - mc)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    assert best_perm is not None
    return {int(col): marker_comps[int(marker_idx)] for col, marker_idx in zip(colors, best_perm)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each sprite by reading a nearby marker that tells which mirror-axis family to use.

    Input structure:
    - bg is the most common color. Red (2) is a detached marker color, not part of the sprite payload.
    - Each non-red sprite is almost symmetric, but some same-color pixels are noise. A short red line sits next to each sprite.

    Rule:
    - Pair every sprite color with its nearby red line. A horizontal red line means search vertical reflection axes; a vertical red line means search horizontal reflection axes.
    - Within that axis family, choose the best-scoring axis for the sprite and delete pixels that do not have a same-color mirror partner.

    Key insight:
    - The red marker does not specify the exact axis location; it only selects the direction of the symmetry search. The sprite still determines where the best axis lies.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    marker_lab, n_markers = label_mask(g == int(MARK), connectivity=4)
    marker_comps: list[dict] = []
    for lbl in range(1, int(n_markers) + 1):
        pts = np.argwhere(marker_lab == int(lbl))
        if pts.size == 0:
            continue
        coords = [(int(r), int(c)) for r, c in pts]
        marker_comps.append(
            {
                "coords": coords,
                "axis_dir": _marker_orientation(coords),
                "center": _bbox_center(pts),
            }
        )

    colors = colors_present(g, exclude={bg, MARK})
    assert colors, "grid must contain non-marker sprites"

    color_pts = {int(col): np.argwhere(g == int(col)) for col in colors}
    marker_for_color = _assign_markers_to_colors(color_pts, marker_comps)

    for col in colors:
        pts_arr = color_pts[int(col)]
        if pts_arr.size == 0:
            continue
        pts = [(int(r), int(c)) for r, c in pts_arr]
        axis_dir = marker_for_color[int(col)]["axis_dir"]
        best_k, _best_score = _best_axis(pts, axis_dir, height, width)
        for r, c in pts:
            rr, cc = _mirror_cell(axis_dir, best_k, r, c)
            if not (0 <= rr < height and 0 <= cc < width and int(g[rr, cc]) == int(col)):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 3
LAYOUTS = [
    [(0, 0, 9, 8), (0, 13, 9, 21), (12, 4, 21, 17)],
    [(0, 0, 9, 9), (0, 12, 9, 21), (11, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# ── CFG: symmetry quality ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2
ORTHO_AXIS_GAP_MIN = 2

# ── CFG: markers ──
MARKER_LEN = 3
MARKER_GAP = 1

# ── CFG: retry limits ──
SPRITE_RETRIES = 120
GEN_RETRIES = 300

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _second_best_score(
    coords: list[tuple[int, int]], axis_dir: str, true_k: int, height: int, width: int
) -> int:
    scores = _axis_scores(coords, axis_dir, height, width)
    return max(score for k, score in scores.items() if int(k) != int(true_k))


def _best_score(coords: list[tuple[int, int]], axis_dir: str, height: int, width: int) -> int:
    return max(_axis_scores(coords, axis_dir, height, width).values())


def _other_axis(axis_dir: str) -> str:
    return "horizontal" if axis_dir == "vertical" else "vertical"


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = 2 * side_w + 1
                if w > max_w:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((h, w, target_count, center_count))

            if target_count % 2 != 0:
                continue
            for side_w in EVEN_SIDE_W_RANGE:
                w = 2 * side_w
                if w > max_w:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                left_count = target_count // 2
                if h <= left_count <= h * side_w:
                    options.append((h, w, target_count, 0))
    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({tc for _h, _w, tc, _cc in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _tc, _cc in target_options]
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
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_sprite_vertical_raw(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mirror_c = w - 1 - c
            sprite[r, c] = int(color)
            sprite[r, mirror_c] = int(color)
        if center_count > 0:
            sprite[centers, side_w] = int(color)

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if _second_best_score(coords, "vertical", true_k, h, w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _build_clean_sprite(color: int, max_h: int, max_w: int, axis_dir: str) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        if axis_dir == "vertical":
            sprite, true_k = _build_clean_sprite_vertical_raw(color, max_h, max_w)
        else:
            tmp, _tmp_k = _build_clean_sprite_vertical_raw(color, max_w, max_h)
            sprite = tmp.T.copy()
            true_k = sprite.shape[0] - 1

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        ortho = _best_score(coords, _other_axis(axis_dir), sprite.shape[0], sprite.shape[1])
        if ortho > len(coords) - ORTHO_AXIS_GAP_MIN:
            continue
        return sprite, int(true_k)
    raise ValueError("failed to build clean sprite with directional asymmetry")


def _marker_placement(
    axis_dir: str, sprite_shape: tuple[int, int], slot: tuple[int, int, int, int]
) -> tuple[int, int, list[tuple[int, int]], set[tuple[int, int]]]:
    h, w = sprite_shape
    r0s, c0s, r1s, c1s = slot
    slot_h = r1s - r0s + 1
    slot_w = c1s - c0s + 1

    if axis_dir == "vertical":
        if h + MARKER_GAP + 1 > slot_h or w > slot_w:
            raise ValueError("slot too small for vertical-axis sprite")
        side = random.choice(["top", "bottom"])
        c0 = random.randint(c0s, c1s - w + 1)
        if side == "top":
            r0 = random.randint(r0s + MARKER_GAP + 1, r1s - h + 1)
            marker_r = r0 - (MARKER_GAP + 1)
            gap_r = r0 - 1
        else:
            r0 = random.randint(r0s, r1s - h - (MARKER_GAP + 1) + 1)
            marker_r = r0 + h + MARKER_GAP
            gap_r = r0 + h
        marker_c0 = c0 + (w - MARKER_LEN) // 2
        marker_cells = [(int(marker_r), int(marker_c0 + dc)) for dc in range(MARKER_LEN)]
        blockers = {(int(gap_r), int(marker_c0 + dc)) for dc in range(MARKER_LEN)}
        return int(r0), int(c0), marker_cells, blockers

    if h > slot_h or w + MARKER_GAP + 1 > slot_w:
        raise ValueError("slot too small for horizontal-axis sprite")
    side = random.choice(["left", "right"])
    r0 = random.randint(r0s, r1s - h + 1)
    if side == "left":
        c0 = random.randint(c0s + MARKER_GAP + 1, c1s - w + 1)
        marker_c = c0 - (MARKER_GAP + 1)
        gap_c = c0 - 1
    else:
        c0 = random.randint(c0s, c1s - w - (MARKER_GAP + 1) + 1)
        marker_c = c0 + w + MARKER_GAP
        gap_c = c0 + w
    marker_r0 = r0 + (h - MARKER_LEN) // 2
    marker_cells = [(int(marker_r0 + dr), int(marker_c)) for dr in range(MARKER_LEN)]
    blockers = {(int(marker_r0 + dr), int(gap_c)) for dr in range(MARKER_LEN)}
    return int(r0), int(c0), marker_cells, blockers


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for spec, slot in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        try:
            r0, c0, marker_cells, blockers = _marker_placement(spec["axis_dir"], (h, w), slot)
        except ValueError:
            return None
        paste(out, sprite, r0, c0, transparent=0)
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["marker_cells"] = marker_cells
        spec["noise_blockers"] = blockers
        if spec["axis_dir"] == "vertical":
            spec["true_k"] = int(spec["axis_k_local"] + 2 * c0)
        else:
            spec["true_k"] = int(spec["axis_k_local"] + 2 * r0)
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
            adj[:, c] &= ~clean_mask[:, int(mirror_cols[c])]
    else:
        rows = np.arange(height)
        mirror_rows = int(true_k) - rows
        on_axis = mirror_rows == rows
        adj[on_axis, :] = False
        valid = (mirror_rows >= 0) & (mirror_rows < height) & ~on_axis
        for r in np.where(valid)[0]:
            adj[r, :] &= ~clean_mask[int(mirror_rows[r]), :]

    rows, cols = np.where(adj)
    return list(zip(rows.tolist(), cols.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    color = int(spec["color"])
    axis_dir = str(spec["axis_dir"])
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, axis_dir, true_k, forbidden=spec["noise_blockers"])
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int]] = []
    forbidden = set(spec["noise_blockers"])
    current_coords = list(clean_coords)
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
        if not viable:
            return False

        scored = []
        for cell in viable:
            alt = _second_best_score(
                current_coords + [cell], axis_dir, true_k, inp.shape[0], inp.shape[1]
            )
            scored.append((int(alt), random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _rand, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        forbidden.add(cell)
        mirror = _mirror_cell(axis_dir, true_k, cell[0], cell[1])
        if 0 <= mirror[0] < GRID_SIZE and 0 <= mirror[1] < GRID_SIZE:
            forbidden.add((int(mirror[0]), int(mirror[1])))

    if clean_count - _second_best_score(current_coords, axis_dir, true_k, GRID_SIZE, GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[int(r), int(c)] = color
    return True


def _add_markers(inp: np.ndarray, placed_specs: list[dict]) -> None:
    for spec in placed_specs:
        for r, c in spec["marker_cells"]:
            inp[int(r), int(c)] = int(MARK)


def _gen_attempt() -> np.ndarray | None:
    palette = [c for c in range(10) if c != MARK]
    bg, *obj_colors = random.sample(palette, NUM_OBJECTS + 1)
    slots = random.choice(LAYOUTS)

    num_horizontal = random.choice([1, 2])
    axis_dirs = ["horizontal"] * num_horizontal + ["vertical"] * (NUM_OBJECTS - num_horizontal)
    random.shuffle(axis_dirs)

    sprite_specs = []
    try:
        for color, slot, axis_dir in zip(obj_colors, slots, axis_dirs):
            slot_h = slot[2] - slot[0] + 1
            slot_w = slot[3] - slot[1] + 1
            max_h = slot_h - (MARKER_GAP + 1 if axis_dir == "vertical" else 0)
            max_w = slot_w - (MARKER_GAP + 1 if axis_dir == "horizontal" else 0)
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w, axis_dir)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "axis_dir": axis_dir,
                    "axis_k_local": int(axis_k_local),
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec):
            return None
    _add_markers(inp, placed_specs)

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
