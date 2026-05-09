# Combo variant of 3dc255db (index 5)
# combo_spec: D1/1+D4/0+D5/1
# Name: 反弹+障碍停止+渐变轨迹
#
# Design analysis for this combo
# a) Concrete behavior and trace:
#    - Base task: erase the mouth contents and shoot them straight out from the spike, one
#      monochrome cell per content pixel, until the ray leaves the grid.
#    - Combo task here:
#      1. The ray is simulated step-by-step and reflects off grid boundaries.
#      2. The ray ignores its own border and its own content pixels, so after a bounce it can pass
#         back through the spike and the central mouth corridor just like the D1 reflection variant.
#      3. The first other non-bg pixel on that bounced path is an obstacle; the ray stops before it
#         and drops all remaining content pixels.
#      4. Because ARC colors are categorical rather than numeric, the "gradient" is discretized per
#         straight segment into three ordered zones: first cell = content color, last cell = border
#         color, interior cells (if any) = the obstacle color. This still gives a clear
#         content→middle→border progression, and ties the trajectory colors to a visible input cue.
#    - Small trace (tail points up, direct run to top edge has length 2):
#      input ray-axis cells away from the spike: [bg, bg, bg, bg, obstacle, ...]
#      bounced walk: step1=row1, step2=row0, bounce, step3=row1, step4=spike, step5=mouth cell,
#      then stop because step6 would enter the obstacle.
#      output colors by segment:
#        segment 1 (rows 1,0): [content, border]
#        segment 2 (rows 1,spike,mouth): [content, obstacle-color, border]
#      The obstacle itself remains unchanged.
# b) Complexity calibration:
#    - Reflection is forced to matter: each generated cup is placed so the blocker is reachable only
#      after at least one boundary bounce.
#    - Obstacle stopping is also forced to matter: content_count is always strictly larger than the
#      number of cells written before the blocker, so some content is dropped.
#    - Gradient visibility is forced by requiring at least one segment of length >= 3, so at least
#      one middle-colored cell appears in the output.
#    - To avoid visual clutter after adding per-cup obstacles and recolored bounced paths, the
#      generator uses 1-2 cups and removes unrelated stray pixels from the base task.
# c) Implicit-assumption audit of the original main():
#    - Cup detection via unique border component, bbox-edge singleton spike search, and mouth flood
#      remains valid.
#    - The original straight-ray loop assumed "move in one direction until out of bounds"; that is
#      replaced by a stateful bounced walk.
#    - The original code assumed every non-bg cell on the path was writable. That breaks here:
#      obstacle detection must consult the original input grid `g`.
#    - The original code assumed the ray never revisits old cells. That is false after reflection, so
#      the solver now tracks an ordered path segmented by bounces rather than a single simple run.
#    - The obstacle can sit inside the same cup's mouth corridor. Therefore "any non-bg pixel" is too
#      broad a blocker rule; the solver must explicitly exempt the current cup's border cells and the
#      current cup's own content cells, while still treating any other non-bg pixel as blocking.
# d) Placement legality:
#    - Cup bboxes still keep a 1-cell gap.
#    - Each cup reserves any external part of its bounced prefix so later cups cannot overlap it.
#    - The dedicated obstacle cell is chosen on the reflected mouth-side corridor, never on the cup
#      border, and is excluded from content scattering so the solver sees one clear blocker.
# e) Visual observability:
#    - Input already shows the obstacle color on the eventual stopping point, so the middle trajectory
#      color is not arbitrary.
#    - Output makes the reflection visible because the second segment returns through the spike/mouth
#      side, and the stop is visible because the obstacle cell is not overwritten.
#    - The segment-wise color reset at each bend makes the bend structure readable from 2-3 examples.
# f) Visual cleanliness:
#    - Only one special obstacle is added per cup, on the ray corridor.
#    - Colors are unique per cup triplet (border, content, obstacle) within a sample, preventing
#      accidental mergers and reducing ambiguity.
# g) New solver assumptions introduced here:
#    - The dominant non-border color in a cup mouth is still the true content color.
#      Generator guarantee: content_count is always much larger than the single obstacle pixel count.
#    - The first blocker encountered after at least one bounce is the intended stop point.
#      Generator guarantee: the reflected prefix before that blocker is kept free of other blockers.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood inside the bbox, seeded from the chosen bbox edge."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
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


def _bounced_path_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: str, steps: int
) -> list[tuple[int, int]]:
    """Ordered cells visited by a 4-neighbor ray that reflects off the grid boundary."""
    H, W = shape
    r, c = int(start[0]), int(start[1])
    dr, dc = DIR4[direction]
    path: list[tuple[int, int]] = []
    for _ in range(int(steps)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            dr, dc = -dr, -dc
            nr, nc = r + dr, c + dc
        r, c = int(nr), int(nc)
        path.append((r, c))
    return path


def _direct_run_to_edge(H: int, W: int, spike: tuple[int, int], direction: str) -> int:
    sr, sc = spike
    if direction == "U":
        return int(sr)
    if direction == "D":
        return int(H - 1 - sr)
    if direction == "L":
        return int(sc)
    return int(W - 1 - sc)


def _paint_segment_gradient(
    out: np.ndarray,
    segment: list[tuple[int, int]],
    content_col: int,
    border_col: int,
    mid_col: int,
) -> None:
    """Paint one straight segment with a discrete content->mid->border progression."""
    n = len(segment)
    if n == 0:
        return
    if n == 1:
        colors = [content_col]
    elif n == 2:
        colors = [content_col, border_col]
    else:
        colors = [content_col] + [mid_col] * (n - 2) + [border_col]
    for (r, c), col in zip(segment, colors):
        out[r, c] = int(col)


def _trace_bounced_segments_until_blocker(
    g: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    max_steps: int,
    pass_through: set[tuple[int, int]],
    bg: int,
) -> tuple[list[list[tuple[int, int]]], int | None]:
    """
    Trace the bounced ray for at most ``max_steps`` placements.

    The ray may move through cells listed in ``pass_through`` even if they are non-bg in the input.
    The first other non-bg cell is returned as the blocker color; that blocker cell itself is not
    written into any segment.
    """
    H, W = g.shape
    r, c = int(spike[0]), int(spike[1])
    dr, dc = DIR4[direction]
    segments: list[list[tuple[int, int]]] = []
    cur_segment: list[tuple[int, int]] = []
    blocker_color: int | None = None

    for _ in range(int(max_steps)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            if cur_segment:
                segments.append(cur_segment)
                cur_segment = []
            dr, dc = -dr, -dc
            nr, nc = r + dr, c + dc

        if int(g[nr, nc]) != int(bg) and (int(nr), int(nc)) not in pass_through:
            blocker_color = int(g[nr, nc])
            break

        r, c = int(nr), int(nc)
        cur_segment.append((r, c))

    if cur_segment:
        segments.append(cur_segment)
    return segments, blocker_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule for 3dc255db:
    - identify cups exactly as in the base task,
    - erase the true content pixels,
    - trace a bounced ray from the spike,
    - stop before the first obstacle that is neither this cup's border nor this cup's own content,
    - color each straight segment with a discrete content->obstacle->border gradient.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

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
                (d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != int(bg) and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            pass_through = border_set | set(content_cells)
            segments, blocker_color = _trace_bounced_segments_until_blocker(
                g, spike, tail_dir, len(content_cells), pass_through, bg
            )
            mid_col = int(blocker_color) if blocker_color is not None else int(border_col)
            for segment in segments:
                _paint_segment_gradient(out, segment, content_col, int(border_col), mid_col)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
DIRECT_RUN_CHOICES = (2, 3)
SECOND_SEG_LEN_CHOICES = (2, 3)

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical tail-up cup patch with background 0."""
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
    """Flood-fill the interior of the cup patch from the mouth side."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
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
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate the canonical tail-up patch to the target tail direction."""
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
    """Sample canonical cup parameters that still fit the original bbox limits."""
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


def _external_path_mask(
    H: int,
    W: int,
    prefix: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Mask of path cells that lie outside the source cup bbox."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    for r, c in set(prefix):
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            mask[r, c] = True
    return mask


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    mouth: list[tuple[int, int]],
    occ_bbox: np.ndarray,
    occ_external_path: np.ndarray,
) -> tuple[int, int, int, tuple[int, int], list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    """
    Find a placement whose blocker is hit only after one visible bounce.

    Returns:
      (r0, c0, content_count, obstacle_cell, segments, prefix_cells)
    """
    ph, pw = patch.shape
    mouth_rel = set(mouth)
    occ_bbox_dilated = ndimage.binary_dilation(occ_bbox, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_bbox_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_external_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        spike_abs = (r0 + spike[0], c0 + spike[1])
        direct_run = _direct_run_to_edge(H, W, spike_abs, direction)
        if direct_run not in DIRECT_RUN_CHOICES:
            continue

        seq = _bounced_path_cells((H, W), spike_abs, direction, CONTENT_COUNT_MAX + 2)
        bbox_abs = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        for second_len in SECOND_SEG_LEN_CHOICES:
            placed_len = direct_run + second_len
            obstacle_idx = placed_len
            content_count = placed_len + 1
            if content_count > CONTENT_COUNT_MAX:
                continue
            if content_count > len(mouth_rel) - 1:
                continue
            if obstacle_idx >= len(seq):
                continue
            obstacle_cell = seq[obstacle_idx]
            if obstacle_cell in seq[:obstacle_idx]:
                continue
            obs_rel = (obstacle_cell[0] - r0, obstacle_cell[1] - c0)
            if obs_rel not in mouth_rel:
                continue
            if max(direct_run, second_len) < 3:
                continue

            prefix = seq[:placed_len]
            prefix_ok = True
            for rr, cc in set(prefix):
                inside_own_bbox = bbox_abs[0] <= rr <= bbox_abs[2] and bbox_abs[1] <= cc <= bbox_abs[3]
                if not inside_own_bbox and occ_external_path[rr, cc]:
                    prefix_ok = False
                    break
                if not inside_own_bbox and occ_bbox[rr, cc]:
                    prefix_ok = False
                    break
            if not prefix_ok:
                continue

            segments = [prefix[:direct_run], prefix[direct_run:]]
            if any(len(seg) < 2 for seg in segments):
                continue
            return r0, c0, content_count, obstacle_cell, segments, prefix

    return None


def _paint_segments_for_output(
    canvas_out: np.ndarray,
    segments: list[list[tuple[int, int]]],
    content_color: int,
    border_color: int,
    obstacle_color: int,
) -> None:
    for segment in segments:
        _paint_segment_gradient(canvas_out, segment, content_color, border_color, obstacle_color)


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    colors_triplet: tuple[int, int, int],
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_bbox: np.ndarray,
    occ_external_path: np.ndarray,
) -> dict | None:
    """Build one cup, place an internal blocker on the reflected corridor, and commit it."""
    border_color, content_color, obstacle_color = (int(v) for v in colors_triplet)
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 6:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, mouth, occ_bbox, occ_external_path
    )
    if placement is None:
        return None
    r0, c0, content_count, obstacle_cell, segments, prefix = placement
    ph, pw = patch.shape
    bbox_abs = (r0, c0, r0 + ph - 1, c0 + pw - 1)

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_bbox[r0:r0 + ph, c0:c0 + pw] = True

    orow, ocol = obstacle_cell
    canvas_in[orow, ocol] = obstacle_color
    canvas_out[orow, ocol] = obstacle_color

    _paint_segments_for_output(canvas_out, segments, content_color, border_color, obstacle_color)
    occ_external_path |= _external_path_mask(H, W, prefix, bbox_abs)

    return {
        "border_color": border_color,
        "content_color": content_color,
        "obstacle_color": obstacle_color,
        "direction": direction,
        "bbox": bbox_abs,
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "obstacle_cell": obstacle_cell,
        "segments": segments,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's true content inside its mouth, excluding the reserved obstacle cell."""
    for meta in cups_meta:
        candidates = [p for p in meta["mouth_abs"] if p != meta["obstacle_cell"]]
        count = int(meta["content_count"])
        if len(candidates) < count:
            return False
        idxs = rng.choice(len(candidates), size=count, replace=False)
        for i in idxs:
            r, c = candidates[int(i)]
            canvas_in[r, c] = int(meta["content_color"])
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Output-first generation validated by the combo solver itself."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if len(palette) < 3 * num_cups:
        return None

    occ_bbox = np.zeros((H, W), dtype=bool)
    occ_external_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for i in range(num_cups):
        triplet = tuple(int(v) for v in palette[3 * i:3 * i + 3])
        meta = _add_cup(
            rng, H, W, triplet, canvas_in, canvas_out, occ_bbox, occ_external_path
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, canvas_out):
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
