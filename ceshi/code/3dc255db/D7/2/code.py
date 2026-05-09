# Variant of 3dc255db
# Dimension: D7 — 输出网格尺寸 (grid_shape)
# Change: same_size — 输出网格与输入网格尺寸相同... → 扩展网格以容纳全部弹射

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage

# Design analysis for the D7 variant
# a) Concrete behavior and trace:
#    The original task already identifies each hollow cup, erases the interior content, and
#    ejects a straight tail-directed line whose length equals the number of erased content cells.
#    The only changed behavior is what happens when that line crosses the input boundary.
#    Here the input grid is treated as a finite window cut out from an infinite background plane.
#    We first infer all cups exactly as before, then draw every ejection pixel on that infinite
#    plane, even when the coordinates become negative or exceed the input height/width.
#    The output is the minimal axis-aligned bbox that contains both:
#    - the entire original input rectangle [0..H-1] x [0..W-1]
#    - every ejected pixel
#    Small trace: if a tail spike is at (0, 4), points upward, and the cup contains 3 content
#    cells, the original solver keeps only in-bounds cells and therefore draws nothing. The
#    variant instead creates ejection pixels at (-1,4), (-2,4), (-3,4), then prepends 3 rows so
#    the whole original grid shifts downward in the output.
# b) Complexity calibration:
#    The cup-recognition rule stays unchanged, so the task is still "find tail direction + count
#    content". The new difficulty is only output sizing and coordinate shifting. To make this
#    visually observable, the generator forces at least one cup per sample to overflow, and the
#    overflow direction varies across U/D/L/R so both prepend-padding (U/L) and append-padding
#    (D/R) cases appear.
# c) Hidden-assumption audit against the original main():
#    - `out = g.copy()` relied on fixed output size. No longer valid; the variant must allocate
#      after all ejection endpoints are known.
#    - `if not (0 <= rr < H and 0 <= cc < W): break` encoded truncation at the boundary. That is
#      precisely the old D7 behavior and must be replaced by unbounded coordinate collection.
#    - The cup-detection loop itself only reads from the input and does not assume fixed output
#      size, so it can remain intact.
#    - Replaying mutations still needs the original per-cup order (erase then draw) so the new
#      solver keeps the same sequential write semantics after embedding the input into the larger
#      canvas.
# d) Placement legality in the generator:
#    Cups still have non-overlapping bboxes with a 1-cell gap. Their in-grid path from the spike
#    to the relevant boundary must remain clear, but unlike the original generator this path may
#    now be empty (for example a U-tail cup can sit on the top edge and eject entirely outside the
#    original canvas). This is necessary because "fully outside" ejections are now valid outputs.
# e) Visual observability:
#    A human can see that the same cup logic applies, except some outputs become taller/wider and
#    preserve the overflowing line rather than clipping it. Because at least one overflow is forced
#    in every generated sample, the D7 change is present in the visible I/O pair rather than being
#    a latent possibility.
# f) Clean visualization:
#    The generator keeps the original sparse layout discipline: cups do not overlap, in-grid path
#    cells stay clear, and stray pixels remain isolated on the mouth side. Expansion happens only
#    outside the original canvas, so the added pixels do not clutter the cup interiors.
# g) Color inferability:
#    Colors behave exactly as in the source task: each cup has one border color and one content
#    color, and the ejected line keeps that content color. No new color-based rule is introduced.


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from non-border cells on the given bbox edge."""
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


def _extract_cup_actions(g: np.ndarray, bg: int) -> list[dict]:
    """Detect actionable cups exactly as in the source task, but keep full ejection coordinates."""
    actions: list[dict] = []
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
                (direction, cells[0]) for direction, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for direction, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[direction])
                counts = Counter()
                for r, c in mouth:
                    value = int(g[r, c])
                    if value != bg and value != int(border_col):
                        counts[value] += 1
                if not counts:
                    continue
                content_col = int(counts.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((direction, spike, content_col, content_cells))

            if len(passing) != 1:
                continue

            tail_dir, spike, content_col, content_cells = passing[0]
            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            ejection_cells = [(sr + dr * t, sc + dc * t) for t in range(1, len(content_cells) + 1)]
            actions.append(
                {
                    "tail_dir": tail_dir,
                    "content_col": content_col,
                    "content_cells": content_cells,
                    "ejection_cells": ejection_cells,
                }
            )
    return actions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Source-task rule plus D7 variant:
    draw the full ejection line on an expanded canvas instead of clipping it at the boundary.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    actions = _extract_cup_actions(g, bg)

    min_r, min_c = 0, 0
    max_r, max_c = H - 1, W - 1
    for action in actions:
        for rr, cc in action["ejection_cells"]:
            min_r = min(min_r, int(rr))
            min_c = min(min_c, int(cc))
            max_r = max(max_r, int(rr))
            max_c = max(max_c, int(cc))

    row_shift = -min_r
    col_shift = -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    out[row_shift:row_shift + H, col_shift:col_shift + W] = g

    for action in actions:
        for r, c in action["content_cells"]:
            out[row_shift + int(r), col_shift + int(c)] = bg
        for rr, cc in action["ejection_cells"]:
            out[row_shift + int(rr), col_shift + int(cc)] = int(action["content_col"])

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical Y/funnel cup whose tail points upward."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    width = L + R + 1
    height = max(L + hL, R + hR) + 1
    patch = np.zeros((height, width), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1:L + hL + 1, 0] = color
    if hR:
        patch[R + 1:R + hR + 1, width - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the cup interior from the mouth side."""
    height, width = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(width) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(height - 1, c) for c in range(width) if not border_mask[height - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(height) if not border_mask[r, 0]]
    else:  # "R"
        seeds = [(r, width - 1) for r in range(height) if not border_mask[r, width - 1]]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical cup patch to the target tail direction."""
    height, width = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (height - 1 - sr, sc), [(height - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (width - 1 - sc, sr), [(width - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, height - 1 - sr), [(c, height - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Pick one feasible canonical cup shape under the original bbox constraints."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    L_arm = int(rng.integers(lo, hi + 1))
    R_arm = int(rng.integers(lo, hi + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    wlo, whi = WALL_EXT_RANGE
    hL = int(rng.integers(wlo, whi + 1))
    hR = int(rng.integers(wlo, whi + 1))
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
    require_overflow: bool,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int] | None:
    """Find one valid placement; in-grid path may be empty in this D7 variant."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    placements = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(placements)

    for r0, c0 in placements:
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
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        full_ejection = [(sr_abs + dr * t, sc_abs + dc * t) for t in range(1, content_count + 1)]
        overflow = content_count - len(path_cells)
        if require_overflow and overflow <= 0:
            continue
        return r0, c0, path_cells, full_ejection, overflow

    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of one cup, restricted to the cup's perpendicular extent."""
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


def _try_place_stray(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> None:
    """Optionally place one isolated stray pixel on the mouth side of a random cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for other in cups_meta:
        br0, bc0, br1, bc1 = other["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in other["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[int(r), int(c)] = stray_color


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
    require_overflow: bool,
) -> dict | None:
    """Build one cup, place it, and record the full ejection metadata."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if not mouth_canon:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    content_cap = min(CONTENT_COUNT_MAX, len(mouth))
    if content_cap < 1:
        return None
    if require_overflow and content_cap >= 4:
        content_count = int(rng.integers(4, content_cap + 1))
    else:
        content_count = int(rng.integers(1, content_cap + 1))

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
        require_overflow=require_overflow,
    )
    if placement is None:
        return None
    r0, c0, path_cells, full_ejection_cells, overflow = placement

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "spike_abs": (r0 + spike[0], c0 + spike[1]),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "full_ejection_cells": full_ejection_cells,
        "overflow": overflow,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's content pixels inside its mouth and record their exact locations."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        chosen = []
        for idx in idxs:
            r, c = mouth_abs[int(idx)]
            canvas_in[int(r), int(c)] = meta["content_color"]
            chosen.append((int(r), int(c)))
        meta["content_cells_abs"] = chosen
    return True


def _render_expected_output(canvas_in: np.ndarray, cups_meta: list[dict]) -> np.ndarray:
    """Render the intended expanded output directly from generator metadata."""
    H, W = canvas_in.shape
    bg = int(get_bg(canvas_in))

    min_r, min_c = 0, 0
    max_r, max_c = H - 1, W - 1
    for meta in cups_meta:
        for rr, cc in meta["full_ejection_cells"]:
            min_r = min(min_r, int(rr))
            min_c = min(min_c, int(cc))
            max_r = max(max_r, int(rr))
            max_c = max(max_c, int(cc))

    row_shift = -min_r
    col_shift = -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    out[row_shift:row_shift + H, col_shift:col_shift + W] = canvas_in

    for meta in cups_meta:
        for r, c in meta["content_cells_abs"]:
            out[row_shift + int(r), col_shift + int(c)] = bg
        for rr, cc in meta["full_ejection_cells"]:
            out[row_shift + int(rr), col_shift + int(cc)] = int(meta["content_color"])

    return out


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single generation attempt with at least one forced overflowing cup."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    overflow_idx = int(rng.integers(0, num_cups))
    cups_meta: list[dict] = []
    for cup_idx in range(num_cups):
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
            occ_cup,
            occ_path,
            require_overflow=(cup_idx == overflow_idx),
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if not cups_meta or max(meta["overflow"] for meta in cups_meta) <= 0:
        return None

    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    _try_place_stray(rng, canvas_in, cups_meta)
    expected = _render_expected_output(canvas_in, cups_meta)
    if expected.shape == canvas_in.shape:
        return None

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
