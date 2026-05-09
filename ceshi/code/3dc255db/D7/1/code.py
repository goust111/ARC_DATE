# Variant of 3dc255db
# Dimension: D7 — 输出网格尺寸 (grid_shape)
# Change: same_size — 输出网格与输入网格尺寸相同... → 仅保留杯内区域
#
# Design analysis
# ----------------
# reference_task:
#   requested_source: a6f40cea
#   availability: no standalone `a6f40cea.py` solver exists in the available workspace
#   fallback: only the evaluation JSON can be inspected, so the crop baseline here is limited to
#     "remove unrelated outer background and keep only the task-relevant rectangular region(s)"
#
# variant_spec:
#   unchanged_core:
#     - Detect each hollow cup frame exactly as in the original task.
#     - Identify the unique tail spike by bbox-edge counting plus the mouth-side flood test.
#     - Remove all content pixels from the mouth and conceptually eject them outward.
#   D7_change:
#     - After the original same-size output is formed, discard the global canvas.
#     - Keep only each cup's own bbox crop from that full output.
#     - If there are multiple cups, concatenate those cropped panels along the cups' dominant
#       layout axis in the input:
#         - horizontally arranged cups -> left-to-right concatenation
#         - vertically arranged cups -> top-to-bottom concatenation
#     - The orthogonal axis is padded with background to make one rectangular output grid.
#   small_example_trace:
#     - Original logic: a right-facing cup inside a 12x12 grid clears its interior content and
#       paints a short line outside the bbox to the right. The returned output stays 12x12.
#     - Variant logic: compute the same 12x12 intermediate scene first, then crop the cup's bbox
#       only. The outside ejection line is therefore dropped by cropping, so the visible result is
#       the emptied cup patch itself.
#     - Multi-cup difference: instead of keeping several far-apart cups on one large canvas, the
#       solver packs their bbox crops together in one compact strip/stack.
#   complexity_calibration:
#     - The new rule is still visually inferable because input cups are laid out in clear row-like
#       or column-like groups, so the splice axis is visible from geometry rather than color ids.
#     - Multi-cup cases are biased upward, and content counts are biased away from 1 when possible,
#       so the crop/splice change is repeatedly observable.
#   implicit_assumption_audit:
#     - `_flood_from_bbox_side(...)` still assumes the cup mouth is reachable from the edge
#       opposite the tail candidate. The variant does not change cup topology, so this remains
#       valid.
#     - Candidate spike detection still assumes "single non-corner border cell on a bbox edge".
#       The generator preserves that.
#     - The original code wrote ejection pixels directly into the global output. Under D7 this is
#       no longer the final return value, so the solver must not return `out` directly; it must
#       crop after the full scene render.
#     - Concatenation adds a new assumption: cups in the input have a clear dominant horizontal or
#       vertical arrangement. The generator now enforces that and rejects ambiguous placements.
#   placement_legality:
#     - Cups still keep a 1-cell gap between bboxes and clear ejection paths to the canvas edge.
#     - For multi-cup inputs the generator constrains cup centers to a narrow shared row band
#       (horizontal layout) or column band (vertical layout), then verifies that the solver's
#       dominant-axis heuristic recovers the intended axis.
#   visual_observability:
#     - Output size is often strictly smaller than input size.
#     - Multi-cup samples show either a left-to-right strip or a top-to-bottom stack of cropped
#       cup panels; the branch depends on visible cup placement, not hidden metadata.
#   visual_cleanliness:
#     - Stray pixels remain possible but less frequent; they disappear in the crop and therefore do
#       not clutter the compact output.
#   color_inference:
#     - No new color mapping is introduced. Colors only preserve the original border/content roles.

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _flood_from_bbox_side(border_set: set, bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood inside ``bbox`` seeded from a chosen bbox edge, stopping at border cells."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:  # "R"
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _detect_cups(grid: np.ndarray) -> list[dict]:
    """Recover all cups that the original solver would act on."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cups: list[dict] = []

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [
                (d, cells[0]) for d, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = sorted(p for p in mouth if int(g[p[0], p[1]]) == content_col)
                passing.append((d, spike, content_col, content_cells, sorted(mouth)))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells, mouth = passing[0]
            cups.append(
                {
                    "border_color": int(border_col),
                    "content_color": content_col,
                    "bbox": bbox,
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "content_cells": content_cells,
                    "mouth": mouth,
                }
            )

    cups.sort(key=lambda cup: (cup["bbox"][0], cup["bbox"][1], cup["border_color"]))
    return cups


def _full_scene_after_ejection(grid: np.ndarray, cups: list[dict]) -> np.ndarray:
    """Render the original same-size output before the D7 crop step."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape
    bg = int(get_bg(g))

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        for t in range(1, len(cup["content_cells"]) + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = cup["content_color"]

    return out


def _layout_axis_from_bboxes(bboxes: list[tuple[int, int, int, int]]) -> str:
    """Infer whether the cups are arranged more horizontally or vertically."""
    if len(bboxes) <= 1:
        return "V"
    row_centers = [(r0 + r1) / 2.0 for r0, _c0, r1, _c1 in bboxes]
    col_centers = [(c0 + c1) / 2.0 for _r0, c0, _r1, c1 in bboxes]
    row_span = max(row_centers) - min(row_centers)
    col_span = max(col_centers) - min(col_centers)
    return "H" if col_span > row_span else "V"


def _assemble_bbox_crops(full_out: np.ndarray, cups: list[dict], bg: int) -> np.ndarray:
    """Crop each cup bbox from the full output and concatenate the crops."""
    if not cups:
        return np.array(full_out, dtype=int)

    bboxes = [cup["bbox"] for cup in cups]
    axis = _layout_axis_from_bboxes(bboxes)
    if axis == "H":
        ordered = sorted(cups, key=lambda cup: (cup["bbox"][1], cup["bbox"][0]))
    else:
        ordered = sorted(cups, key=lambda cup: (cup["bbox"][0], cup["bbox"][1]))
    patches = [full_out[r0:r1 + 1, c0:c1 + 1].copy() for r0, c0, r1, c1 in (cup["bbox"] for cup in ordered)]

    if len(patches) == 1:
        return patches[0]

    if axis == "H":
        max_h = max(p.shape[0] for p in patches)
        total_w = sum(p.shape[1] for p in patches)
        out = np.full((max_h, total_w), bg, dtype=int)
        cursor = 0
        for patch in patches:
            h, w = patch.shape
            out[:h, cursor:cursor + w] = patch
            cursor += w
        return out

    max_w = max(p.shape[1] for p in patches)
    total_h = sum(p.shape[0] for p in patches)
    out = np.full((total_h, max_w), bg, dtype=int)
    cursor = 0
    for patch in patches:
        h, w = patch.shape
        out[cursor:cursor + h, :w] = patch
        cursor += h
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Original task behavior plus D7 crop:
    1. detect each cup and clear/eject its content as before,
    2. crop the final scene to the cup bboxes only,
    3. concatenate those crops along the cups' dominant input layout axis.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cups = _detect_cups(g)
    full_out = _full_scene_after_ejection(g, cups)
    return _assemble_bbox_crops(full_out, cups, bg)


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)

# Bias toward multi-cup cases so the D7 splice is well represented.
NUM_CUPS_CHOICES = (1, 2, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
COMPACT_ARM_LEN_RANGE = (1, 2)
COMPACT_WALL_EXT_RANGE = (0, 2)

CONTENT_COUNT_MAX = 7

STRAY_PROB = 0.25
STRAY_USE_BORDER_PROB = 0.5

LAYOUT_BAND_TOL = 1.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

_STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1:L + hL + 1, 0] = color
    if hR:
        g[R + 1:R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the cup interior from the mouth side."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:  # "R"
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]
    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical (tail=U) patch to the target tail direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(
    rng: np.random.Generator,
    direction: str,
    compact: bool,
) -> tuple[int, int, int, int] | None:
    """Pick canonical cup parameters that fit the bbox limits."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    arm_lo, arm_hi = COMPACT_ARM_LEN_RANGE if compact else ARM_LEN_RANGE
    wall_lo, wall_hi = COMPACT_WALL_EXT_RANGE if compact else WALL_EXT_RANGE

    L_arm = int(rng.integers(arm_lo, arm_hi + 1))
    R_arm = int(rng.integers(arm_lo, arm_hi + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    hL = int(rng.integers(wall_lo, wall_hi + 1))
    hR = int(rng.integers(wall_lo, wall_hi + 1))
    if hL + hR == 0:
        hL = 1
    while max(L_arm + hL, R_arm + hR) + 1 > max_h_canon:
        if L_arm + hL >= R_arm + hR and hL > 0:
            hL -= 1
        elif hR > 0:
            hR -= 1
        else:
            break
    if hL + hR == 0:
        return None
    return L_arm, R_arm, hL, hR


def _candidate_matches_layout(
    r0: int,
    c0: int,
    ph: int,
    pw: int,
    layout: dict,
) -> bool:
    """Constrain later cups to a clear shared row-band or column-band."""
    if layout["anchor"] is None:
        return True
    center_r = r0 + (ph - 1) / 2.0
    center_c = c0 + (pw - 1) / 2.0
    if layout["axis"] == "H":
        return abs(center_r - layout["anchor"]) <= LAYOUT_BAND_TOL
    return abs(center_c - layout["anchor"]) <= LAYOUT_BAND_TOL


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
    layout: dict,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup with spacing, path-clearance, and layout-axis constraints."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if not _candidate_matches_layout(r0, c0, ph, pw, layout):
            continue
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((rr, cc))
            rr += dr
            cc += dc
        if not path_cells:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Mask of cells lying on the mouth side of a cup bbox."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif tail_dir == "D":
        mask[:r0, c0:c1 + 1] = True
    else:  # "U"
        mask[r1 + 1:, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    """Optionally place one isolated stray pixel on the mouth side of a cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in cup["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
    layout: dict,
    compact: bool,
) -> dict | None:
    """Build, place, and commit a cup."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction, compact=compact)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 2:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    count_lo = 2 if len(mouth) >= 2 else 1
    content_count = int(rng.integers(count_lo, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(
        rng,
        H,
        W,
        patch,
        spike,
        direction,
        content_count,
        occ_cup,
        occ_path,
        layout,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    center = (r0 + (ph - 1) / 2.0, c0 + (pw - 1) / 2.0)
    if layout["anchor"] is None:
        layout["anchor"] = center[0] if layout["axis"] == "H" else center[1]

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's content pixels inside its mouth."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _assemble_expected_from_meta(canvas_out: np.ndarray, cups_meta: list[dict]) -> np.ndarray:
    """Build the expected D7 output from generation metadata."""
    bg = int(get_bg(canvas_out))
    cups = [{"bbox": meta["bbox"]} for meta in cups_meta]
    return _assemble_bbox_crops(canvas_out, cups, bg)


def _layout_axis_matches(cups_meta: list[dict], intended_axis: str) -> bool:
    """Ensure the solver's dominant-axis heuristic agrees with the generator layout choice."""
    if len(cups_meta) <= 1:
        return True
    inferred = _layout_axis_from_bboxes([meta["bbox"] for meta in cups_meta])
    return inferred == intended_axis


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single end-to-end generation attempt with oracle verification."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    intended_axis = "V" if num_cups >= 3 else str(rng.choice(["H", "V"]))
    layout = {"axis": intended_axis, "anchor": None}
    compact = num_cups >= 2

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if len(palette) < 2:
            return None
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        palette.append(content_color)

        meta = _add_cup(
            rng,
            H,
            W,
            border_color,
            content_color,
            canvas_in,
            canvas_out,
            occ_cup,
            occ_path,
            layout,
            compact=compact,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _layout_axis_matches(cups_meta, intended_axis):
        return None

    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

    expected = _assemble_expected_from_meta(canvas_out, cups_meta)
    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
