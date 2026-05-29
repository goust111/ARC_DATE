# Combo variant of 3dc255db (index 4)
# combo_spec: D1/1+D4/0
# Name: 反弹弹射+障碍物停止
#
# Design analysis for this combo
# a) Concrete behavior and trace:
#    - Base task: each cup is detected from its monochrome border component; the mouth content is
#      erased, then the content color is emitted from the spike in a straight ray until the grid
#      edge or until content runs out.
#    - D1/1 changes the path: after the ray reaches a boundary, it reflects and continues along
#      the same axis. This makes the path re-enter the spike and the cup's mouth corridor.
#    - D4/0 changes the stop rule: the first obstacle already present on the path is a hard stop;
#      the obstacle stays unchanged and leftover content is dropped.
#    - Combo rule: simulate the bounced path one step at a time, but stop before stepping onto the
#      first obstacle. Example with tail=U, spike row=2, content count=6:
#        original straight ray: rows 1,0 then stop/drop the rest
#        pure bounce: rows 1,0,1,2,3,4
#        combo with an obstacle at row 4: rows 1,0,1,2,3 then stop before row 4
#      The visible signature is therefore "boundary bounce happened, but a later obstacle cut the
#      reflected continuation short".
# b) Complexity calibration:
#    - The generator forces each cup to have a short direct run to the launch boundary and places
#      its obstacle on the mouth side of the cup, after at least one bounce. This prevents
#      degeneration to either the base task or the obstacle-only straight-ray variant.
#    - Obstacles remain isolated single pixels; cups still use 1..3 objects on 10..13 grids, so
#      the scene stays legible.
# c) Implicit-assumption audit of the original main():
#    - Cup detection via bbox-edge singleton spikes and mouth flood-fill remains valid because cup
#      geometry is unchanged.
#    - The original ejection loop assumed "outward cells are written once until boundary". That no
#      longer holds: the combo needs stepwise simulation with reflection and an early-stop check.
#    - The original obstacle-only variant could treat every non-bg pixel as a blocker. That is no
#      longer valid once bounce can revisit the source cup: the ray must be allowed to pass through
#      its own border spike and its own mouth content cells, otherwise it would stop at the spike on
#      the first reflection and the bounce would become invisible.
#    - Blocker checks must therefore use the original input grid `g`, but exclude the current cup's
#      border cells and its identified content cells.
# d) Placement legality:
#    - Cup bboxes keep a 1-cell gap.
#    - Every obstacle is outside all cup bboxes, on the mouth side of its source cup, and at least
#      8-neighbor separated from earlier cups/obstacles.
#    - The traversed bounced prefix outside the source bbox is also reserved, so later cups do not
#      overlap the path that the solver will actually paint.
# e) Visual observability:
#    - Each example contains a visible reflected continuation that reaches at least the spike and
#      then heads toward the mouth side, where an isolated obstacle halts it.
#    - This lets a human distinguish the combo from both "straight ray stops at obstacle" and
#      "bounce forever until content runs out".
# f) Visual cleanliness:
#    - Unrelated stray pixels are removed; only the cups, their interior content, and one obstacle
#      per cup remain.
#    - Obstacles are placed on a structured axis-aligned location tied to one cup, so they read as
#      meaningful markers rather than noise.
# g) New solver assumptions and generator guarantees:
#    - Assumption: blockers are non-bg cells not belonging to the active cup.
#      Guarantee: generator puts exactly one such obstacle on the traversed bounced prefix.
#    - Assumption: the first external blocker on that prefix is on the mouth side after a bounce.
#      Guarantee: generator chooses obstacle cells only from mouth-side positions on the reflected
#      sequence and reserves the earlier prefix.
#    - Assumption: cup/content disambiguation remains unique.
#      Guarantee: obstacles stay outside cup bboxes and never touch cup components.

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
    """4-connected flood inside ``bbox`` from the chosen edge, stopping at border cells."""
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


def _detect_cups(g: np.ndarray, bg: int) -> list[dict]:
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
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]
            cups.append(
                {
                    "border_col": int(border_col),
                    "bbox": bbox,
                    "border_set": border_set,
                    "tail_dir": tail_dir,
                    "spike": (int(spike[0]), int(spike[1])),
                    "content_col": int(content_col),
                    "content_cells": [(int(r), int(c)) for r, c in content_cells],
                }
            )
    return cups


def _bounced_path_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: str, steps: int
) -> list[tuple[int, int]]:
    """Return the next ``steps`` visited cells of a cardinal ray that reflects off boundaries."""
    H, W = shape
    r, c = (int(start[0]), int(start[1]))
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


def _bounced_prefix_until_block(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: str,
    max_steps: int,
    blockers: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    """Trace a bounced ray until it runs out of steps or the next cell is a blocker."""
    H, W = shape
    r, c = (int(start[0]), int(start[1]))
    dr, dc = DIR4[direction]
    path: list[tuple[int, int]] = []
    stop_cell: tuple[int, int] | None = None

    for _ in range(int(max_steps)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            dr, dc = -dr, -dc
            nr, nc = r + dr, c + dc
        if (int(nr), int(nc)) in blockers:
            stop_cell = (int(nr), int(nc))
            break
        r, c = int(nr), int(nc)
        path.append((r, c))

    return path, stop_cell


def _direct_run_to_edge(H: int, W: int, spike: tuple[int, int], direction: str) -> int:
    sr, sc = spike
    if direction == "U":
        return int(sr)
    if direction == "D":
        return int(H - 1 - sr)
    if direction == "L":
        return int(sc)
    return int(W - 1 - sc)


def _is_mouth_side_cell(cell: tuple[int, int], bbox: tuple[int, int, int, int], tail_dir: str) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    if tail_dir == "U":
        return r > r1 and c0 <= c <= c1
    if tail_dir == "D":
        return r < r0 and c0 <= c <= c1
    if tail_dir == "L":
        return c > c1 and r0 <= r <= r1
    return c < c0 and r0 <= r <= r1


def _analyze_cup_paths(g: np.ndarray) -> list[dict]:
    """Internal helper for validation scripts: detect cups and summarize combo-path behavior."""
    g = np.array(g, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    all_non_bg = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}
    analysis: list[dict] = []

    for cup in _detect_cups(g, bg):
        content_cells = cup["content_cells"]
        allowed = cup["border_set"] | set(content_cells)
        blockers = all_non_bg - allowed
        path, stop_cell = _bounced_prefix_until_block(
            (H, W), cup["spike"], cup["tail_dir"], len(content_cells), blockers
        )
        direct_run = _direct_run_to_edge(H, W, cup["spike"], cup["tail_dir"])
        analysis.append(
            {
                **cup,
                "direct_run": int(direct_run),
                "emitted_len": int(len(path)),
                "stop_cell": stop_cell,
                "bounce_visible": bool(len(path) > direct_run),
                "mouth_side_stop": bool(stop_cell is not None and _is_mouth_side_cell(stop_cell, cup["bbox"], cup["tail_dir"])),
            }
        )
    return analysis


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from cup-shaped frames outward along each cup's tail direction.

    Combo rule:
    - Detect cups exactly as in the source task.
    - Erase the mouth content.
    - Trace the outward ray step-by-step with boundary reflection.
    - Stop immediately before the first non-bg obstacle from the original input, excluding the
      source cup's own border and its own content cells so the reflected ray can pass through the
      spike and mouth corridor.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()
    all_non_bg = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}

    for cup in _detect_cups(g, bg):
        content_cells = cup["content_cells"]
        for r, c in content_cells:
            out[r, c] = bg

        allowed = cup["border_set"] | set(content_cells)
        blockers = all_non_bg - allowed
        path, _stop_cell = _bounced_prefix_until_block(
            (H, W), cup["spike"], cup["tail_dir"], len(content_cells), blockers
        )
        for rr, cc in path:
            out[rr, cc] = cup["content_col"]

    return out


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
VISIBLE_DIRECT_RUN_MAX = 3

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ================================


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP (spike at row 0)."""
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
    """Rotate a canonical tail-UP cup patch to the requested tail direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Pick canonical cup parameters that fit the original bbox constraints after rotation."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    L_arm = int(rng.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1] + 1))
    R_arm = int(rng.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1] + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    hL = int(rng.integers(WALL_EXT_RANGE[0], WALL_EXT_RANGE[1] + 1))
    hR = int(rng.integers(WALL_EXT_RANGE[0], WALL_EXT_RANGE[1] + 1))
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


def _choose_obstacle_color(rng: np.random.Generator, content_color: int) -> int:
    choices = [c for c in range(1, 10) if c != content_color]
    return int(rng.choice(choices))


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    max_content_count: int,
    occ_bbox: np.ndarray,
    occ_trace: np.ndarray,
    occ_obstacle: np.ndarray,
) -> tuple[int, int, int, tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    """
    Find a placement whose actual bounced prefix is clear and whose first obstacle lies on the
    mouth side after a bounce.
    """
    ph, pw = patch.shape
    occ_bbox_dilated = ndimage.binary_dilation(occ_bbox, structure=_STRUCT8)
    occ_obstacle_dilated = ndimage.binary_dilation(occ_obstacle, structure=_STRUCT8)
    occ_external_block = ndimage.binary_dilation(occ_bbox | occ_trace | occ_obstacle, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_bbox_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_obstacle_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_trace[r0:r0 + ph, c0:c0 + pw].any():
            continue

        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        direct_run = _direct_run_to_edge(H, W, (sr_abs, sc_abs), direction)
        if direct_run < 1 or direct_run > VISIBLE_DIRECT_RUN_MAX:
            continue

        seq = _bounced_path_cells((H, W), (sr_abs, sc_abs), direction, max_content_count)
        candidate_indices: list[int] = []
        blocked = False
        for idx, (r, c) in enumerate(seq):
            inside_own_bbox = r0 <= r < r0 + ph and c0 <= c < c0 + pw
            if not inside_own_bbox and occ_external_block[r, c]:
                blocked = True
                break
            if not inside_own_bbox and _is_mouth_side_cell((r, c), bbox, direction):
                candidate_indices.append(idx)
        if blocked or not candidate_indices:
            continue

        feasible = [idx for idx in candidate_indices if idx + 1 <= max_content_count]
        if not feasible:
            continue
        obstacle_idx = int(rng.choice(feasible))
        free_run = obstacle_idx
        min_content_count = free_run + 1
        if min_content_count > max_content_count:
            continue

        content_count = int(rng.integers(min_content_count, max_content_count + 1))
        obstacle_cell = seq[obstacle_idx]
        ejection_seq = seq[:free_run]
        path_cells = sorted(set(ejection_seq))
        return r0, c0, content_count, obstacle_cell, path_cells, ejection_seq

    return None


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_bbox: np.ndarray,
    occ_trace: np.ndarray,
    occ_obstacle: np.ndarray,
) -> dict | None:
    """Build one cup, place one mouth-side obstacle on its bounced path, and commit it."""
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
    max_content_count = min(CONTENT_COUNT_MAX, len(mouth))
    if max_content_count < 3:
        return None

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, max_content_count, occ_bbox, occ_trace, occ_obstacle
    )
    if placement is None:
        return None
    r0, c0, content_count, obstacle_cell, path_cells, ejection_seq = placement

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_bbox[r0:r0 + ph, c0:c0 + pw] = True

    obstacle_color = _choose_obstacle_color(rng, content_color)
    orow, ocol = obstacle_cell
    canvas_in[orow, ocol] = obstacle_color
    canvas_out[orow, ocol] = obstacle_color
    occ_obstacle[orow, ocol] = True

    for r, c in ejection_seq:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_trace[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "obstacle_cell": obstacle_cell,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's content pixels at random inside its mouth interior."""
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


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single output-first generation attempt, validated against ``main``."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)

    occ_bbox = np.zeros((H, W), dtype=bool)
    occ_trace = np.zeros((H, W), dtype=bool)
    occ_obstacle = np.zeros((H, W), dtype=bool)

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
            occ_bbox,
            occ_trace,
            occ_obstacle,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    analysis = _analyze_cup_paths(canvas_in)
    if len(analysis) != num_cups:
        return None
    if not all(item["bounce_visible"] and item["mouth_side_stop"] for item in analysis):
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
