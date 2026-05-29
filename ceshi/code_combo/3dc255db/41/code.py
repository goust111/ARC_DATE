# Combo variant of 3dc255db (index 41)
# combo_spec: D10/2+D16/0
# Name: 倍数长度弹射+溢出回填杯内

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage

# -----------------------------------------------------------------------------
# Design analysis for combo D10/2 + D16/0
#
# a) Concrete behavior and trace
# - Visual core stays the same: 2-3 hollow cup/Y/funnel frames sit on a 10..13 grid.
#   Each cup has one spike on a bbox edge and one open mouth on the opposite edge.
# - Original task:
#   1. Count the content pixels inside one cup.
#   2. Erase them.
#   3. Shoot exactly that many pixels from the spike to the grid edge; overflow is lost.
# - Combo task:
#   1. Detect all valid cups first, so the solver knows the global cup count K.
#   2. For one cup with C content pixels, eject length becomes C * K.
#   3. Draw the visible prefix on the tail ray.
#   4. If C * K exceeds the on-grid ray length, the remaining pixels refill the same
#      cup interior from deepest cells first.
# - Example: two cups total (K=2). One upward cup contains C=3 red pixels and has only
#   one visible cell above the spike before the edge. Original output would draw 1 red
#   pixel and drop 2. Combo output draws 1 red pixel, then refills 5 red pixels back
#   into the cup interior because total eject length is 6 and overflow is 5.
#
# b) Complexity calibration
# - The multiplier is fixed to a tiny set of values: K in {2, 3}. This is enough to
#   create a genuine cross-cup dependency without making the inference search space large.
# - Generation always includes at least one non-overflow cup and at least one overflow cup.
#   The first makes "line longer than the visible content count" observable; the second
#   makes "overflow refills the cup" observable.
# - One-cup grids are disallowed here because K=1 would collapse the D10 change and
#   create many degenerate samples identical to the original task.
#
# c) Implicit assumption audit against the original solver
# - Original assumption: `len(content_cells)` controls both erasure and ejection length.
#   Broken. Erasure still depends on the true content cells, but ejection length now
#   depends on `len(content_cells) * number_of_cups`.
# - Original assumption: each cup can be processed independently as soon as it is found.
#   Broken. The solver must first collect all cups, then apply the transform using the
#   global cup count.
# - Original assumption: once the ray leaves the grid, the remaining pixels disappear.
#   Broken by D16. Overflow must be counted and redirected into a deterministic mouth order.
# - Preserved assumptions:
#   1. The mouth interior is still the 4-connected flood from the side opposite the spike.
#   2. Each acted-on cup still has a single content color inside its mouth.
#   3. Tail-spike disambiguation still uses "unique non-corner border cell on a bbox edge"
#      plus the content-reaching flood test.
#
# d) Placement legality
# - Cup bboxes still keep a 1-cell gap via 8-neighborhood dilation.
# - Tail rays remain fully background in the input and reserved globally, so cups and
#   stray pixels never block another cup's ejection path.
# - For every generated cup, content_count is chosen so that:
#     content_count * K <= path_len + mouth_size
#   Therefore overflow can always be fully represented by distinct mouth cells.
#
# e) Visual observability
# - Because K is the total number of cups, examples with two or three cups visibly show
#   that each cup's output depends on the presence of the others.
# - The no-overflow cup shows a longer-than-expected straight line.
# - The overflow cup shows a truncated line plus a compact refill deeper inside the cup.
# - Generator also checks that the combo output differs from the original rule output on
#   every accepted sample, preventing silent regression to the base task.
#
# f) Visual cleanliness
# - Overflow refill is confined to the cup's own mouth, so no cross-object clutter is added.
# - Overflow size is biased toward small values when possible, which keeps the refill pattern
#   readable instead of flooding the entire mouth in every example.
# - Content counts are capped at 4, which avoids large monochrome blobs in the input and
#   reduces accidental fake "border-like" components for reused colors.
#
# g) Change inferability
# - No new color mapping exists. The only new logic is spatial and count-based.
# - The multiplier uses a human-visible quantity (number of cups), not hidden metadata.
# - The refill order uses only cup geometry: deepest cells first, with a simple deterministic
#   raster tie-break on the perpendicular axis.
#
# New solver assumptions introduced by the combo and how the generator satisfies them
# - Assumption: there are at least two valid cups, so the cross-cup multiplier matters.
#   Generator samples only 2 or 3 cups.
# - Assumption: every valid cup is detectable before any transform is applied.
#   Generator preserves the original cup geometry and keeps tail rays empty in the input.
# - Assumption: overflow never exceeds mouth capacity.
#   Generator enforces `overflow_count <= mouth_size` when choosing content_count.
# -----------------------------------------------------------------------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)

# Restrict to 2..3 cups so the global multiplier is always visible.
NUM_CUPS_CHOICES = (2, 3)

CUP_BBOX_MAX_UD = (8, 7)  # (max_h, max_w) when tail is U or D
CUP_BBOX_MAX_LR = (7, 8)  # (max_h, max_w) when tail is L or R

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

# Keep contents small so reused colors do not form large fake components in the input.
CONTENT_COUNT_MAX = 4

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

PREFERRED_OVERFLOW_MAX = 3

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400
# ==================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood within bbox, blocked by border cells."""
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
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _mouth_backfill_order(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    tail_dir: str,
) -> list[tuple[int, int]]:
    """Deepest mouth cells first; ties use raster order on the perpendicular axis."""
    coords = list(cells)
    if tail_dir == "U":
        return sorted(coords, key=lambda rc: (rc[0], rc[1]))
    if tail_dir == "D":
        return sorted(coords, key=lambda rc: (-rc[0], rc[1]))
    if tail_dir == "L":
        return sorted(coords, key=lambda rc: (rc[1], rc[0]))
    if tail_dir == "R":
        return sorted(coords, key=lambda rc: (-rc[1], rc[0]))
    raise ValueError(tail_dir)


def _find_cups(grid: np.ndarray, bg: int) -> list[dict]:
    """Detect all valid cups together with their mouth and content metadata."""
    cups: list[dict] = []
    for border_col in colors_present(grid, exclude={bg}):
        for border_coords in components_mask(grid == int(border_col), connectivity=8):
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
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    v = int(grid[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(grid[p[0], p[1]]) == content_col]
                passing.append(
                    {
                        "border_color": int(border_col),
                        "bbox": bbox,
                        "tail_dir": tail_dir,
                        "spike": spike,
                        "content_col": content_col,
                        "content_cells": content_cells,
                        "mouth": mouth,
                    }
                )

            if len(passing) == 1:
                cups.append(passing[0])
    return cups


def _transform(grid: np.ndarray, *, length_factor: int, use_backfill: bool) -> np.ndarray:
    """Shared transform core for both the combo rule and the original baseline rule."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    cups = _find_cups(g, bg)
    for cup in cups:
        content_cells = cup["content_cells"]
        for r, c in content_cells:
            out[r, c] = bg

        eject_len = len(content_cells) * length_factor
        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        visible_count = 0
        for t in range(1, eject_len + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = cup["content_col"]
            visible_count += 1

        overflow = eject_len - visible_count
        if use_backfill and overflow > 0:
            order = _mouth_backfill_order(cup["mouth"], cup["tail_dir"])
            for r, c in order[:min(overflow, len(order))]:
                out[r, c] = cup["content_col"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Combo rule: eject `content_count * number_of_cups`, then refill overflow into the cup."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cup_count = len(_find_cups(g, bg))
    return _transform(g, length_factor=cup_count, use_backfill=True)


def _build_canonical_cup(
    L: int,
    R: int,
    hL: int,
    hR: int,
    color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
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


def _mouth_interior_coords(
    patch: np.ndarray,
    border_color: int,
    opposite_side: str,
) -> list[tuple[int, int]]:
    """Flood-fill the mouth interior of one patch from the open side."""
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
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
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
    """Rotate a canonical UP-tail patch to the target tail direction."""
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


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Pick cup parameters while respecting the original bbox envelope."""
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


def _count_choices_for_role(
    num_cups: int,
    mouth_size: int,
    path_len: int,
    role: str,
) -> list[tuple[int, int, int]]:
    """Return feasible (content_count, eject_len, overflow_count) tuples for one role."""
    cap = min(CONTENT_COUNT_MAX, mouth_size, (path_len + mouth_size) // num_cups)
    choices: list[tuple[int, int, int]] = []
    for content_count in range(1, cap + 1):
        eject_len = content_count * num_cups
        overflow = max(0, eject_len - path_len)
        if overflow > mouth_size:
            continue
        if role == "overflow" and overflow == 0:
            continue
        if role == "visible" and overflow != 0:
            continue
        choices.append((content_count, eject_len, overflow))
    return choices


def _choose_count_for_role(
    rng: np.random.Generator,
    num_cups: int,
    mouth_size: int,
    path_len: int,
    role: str,
) -> tuple[int, int, int] | None:
    """Pick one feasible count, favoring readable examples."""
    choices = _count_choices_for_role(num_cups, mouth_size, path_len, role)
    if not choices:
        return None

    if role == "overflow":
        preferred = [x for x in choices if 1 <= x[2] <= PREFERRED_OVERFLOW_MAX]
        if preferred:
            choices = preferred
    else:  # visible
        preferred = [x for x in choices if x[0] >= 2 or x[1] >= 4]
        if preferred:
            choices = preferred

    idx = int(rng.integers(0, len(choices)))
    return choices[idx]


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    num_cups: int,
    mouth_size: int,
    role: str,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], int, int, int] | None:
    """Place one cup and simultaneously choose a role-compatible content count."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
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

        choice = _choose_count_for_role(rng, num_cups, mouth_size, len(path_cells), role)
        if choice is None:
            continue
        content_count, eject_len, overflow_count = choice
        return r0, c0, path_cells, content_count, eject_len, overflow_count

    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of the cup bbox, restricted to the perpendicular extent."""
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
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["path_cells"]:
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
    num_cups: int,
    role: str,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build, place, and commit one cup plus its combo-rule output."""
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
    placement = _try_place_cup(
        rng,
        H,
        W,
        patch,
        spike,
        direction,
        num_cups,
        len(mouth),
        role,
        occ_cup,
        occ_path,
    )
    if placement is None:
        return None

    r0, c0, path_cells, content_count, eject_len, overflow_count = placement
    ph, pw = patch.shape
    visible_count = eject_len - overflow_count

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in path_cells:
        occ_path[r, c] = True
    for r, c in path_cells[:visible_count]:
        canvas_out[r, c] = content_color

    mouth_abs = [(r0 + r, c0 + c) for r, c in mouth]
    backfill_cells = _mouth_backfill_order(mouth_abs, direction)[:overflow_count]
    for r, c in backfill_cells:
        canvas_out[r, c] = content_color

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": mouth_abs,
        "content_count": content_count,
        "eject_len": eject_len,
        "path_cells": path_cells,
        "visible_count": visible_count,
        "overflow_count": overflow_count,
        "backfill_cells": backfill_cells,
        "role": role,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's input content inside its mouth interior."""
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


def _sample_roles(rng: np.random.Generator, num_cups: int) -> list[str]:
    """Ensure every example shows both parts of the combo."""
    roles = ["overflow", "visible"]
    while len(roles) < num_cups:
        roles.append(str(rng.choice(["overflow", "visible"])))
    rng.shuffle(roles)
    return roles


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single output-first generation attempt with anti-degeneracy checks."""
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

    roles = _sample_roles(rng, num_cups)
    cups_meta: list[dict] = []
    for role in roles:
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
            num_cups,
            role,
            canvas_in,
            canvas_out,
            occ_cup,
            occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not any(m["overflow_count"] > 0 for m in cups_meta):
        return None
    if not any(m["overflow_count"] == 0 and m["visible_count"] > m["content_count"] for m in cups_meta):
        return None
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

    try:
        produced = main(canvas_in)
        baseline = _transform(canvas_in, length_factor=1, use_backfill=False)
    except Exception:
        return None
    if not np.array_equal(produced, canvas_out):
        return None
    if np.array_equal(produced, baseline):
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
    raise RuntimeError(
        f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts"
    )
