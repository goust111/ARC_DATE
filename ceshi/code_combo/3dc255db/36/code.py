# Combo variant of 3dc255db (index 36)
# combo_spec: D1/3+D4/0
# Name: BFS扩散+障碍物停止
#
# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Cup detection stays the same as the source task: find one 8-connected border component,
#      infer the tail from the bbox edge that has exactly one non-corner border cell, then flood
#      from the opposite side to recover the cup mouth and its content color.
#    - The fused change starts after counting the mouth content pixels. Suppose tail='U',
#      spike=(5,5), seed=(4,5), content_count=5, and an obstacle sits at (3,5).
#        original output: (4,5),(3,5),(2,5),(1,5),(0,5) on a straight ray
#        D1-only output : BFS from (4,5), so early cells are (4,5),(3,5),(4,4),(4,6),(2,5)
#        fused output   : obstacle blocks (3,5), so BFS must detour:
#                         (4,5),(4,4),(4,6),(5,5?) forbidden by bbox, then around the obstacle
#                         through other reachable background cells such as (3,4),(3,6),...
#      The visible result is: content still leaves from the spike, but it fans out as a BFS cloud
#      whose forward branch is stopped by an obstacle and whose remaining pixels route around it.
#    - In this fusion, D4's "stop at obstacle" applies to each local propagation branch, not to
#      the entire object globally. Once one frontier edge meets a blocker, that edge dies, but
#      other queued BFS branches continue if they can still reach background cells.
#
# b) Complexity calibration
#    - To keep the BFS nature visible, generator uses content counts 4..7 and requires the first
#      four filled cells to include at least two off-axis cells.
#    - To keep the obstacle contribution visible, every cup gets one obstacle exactly on the
#      forward cell after the seed. That guarantees the fused output differs from the obstacle-free
#      BFS variant on the same input instead of relying on rare incidental blockers.
#    - A single obstacle per cup is enough: it produces a repeated, learnable cue without making
#      placement on 10..13 grids too fragile.
#
# c) Implicit-assumption audit against the original main()
#    - Original line drawing loop assumed one total order along tail_dir. Fusion breaks that:
#      the emitted region is now a queue-driven 4-neighbor BFS, so visited-state and queue order
#      become mandatory.
#    - Original `break` on out-of-bounds was safe for a straight ray. Under BFS, one blocked or
#      out-of-bounds branch must not terminate the whole fill.
#    - Original code could safely write through any in-bounds tail cell. Fusion must instead read
#      blockers from the original input grid `g`; otherwise earlier cups painted into `out` would
#      become fake obstacles for later cups.
#    - Original code had no need to distinguish "inside current cup bbox" from "outside cup".
#      Fusion must forbid BFS from re-entering the current bbox, or the cloud can leak back toward
#      the mouth through exterior routes.
#    - D4/0 on its own assumed "the first obstacle truncates the whole ray". That assumption does
#      not transfer literally to BFS, so the fused solver interprets obstacles as impassable cells
#      inside the BFS graph rather than a global stop signal.
#
# d) Placement legality
#    - Generator remains output-first: place cup borders, decide obstacle position, compute the
#      actual BFS fill order under those blockers, then reconstruct the input by scattering the
#      content into the mouth.
#    - Cup bboxes keep at least a 1-cell gap via bbox occupancy plus 8-neighbor dilation checks.
#    - Obstacles are outside every cup bbox and kept 8-neighbor separated from previously placed
#      cups and obstacles, so they cannot merge into a misleading larger border component.
#    - Filled BFS cells may not overlap any prior cup, fill cloud, or obstacle.
#
# e) Visual observability
#    - Across a few examples, a human sees three repeated cues together:
#        1. mouth content disappears,
#        2. output is a BFS-shaped cloud instead of a line,
#        3. a lone obstacle pixel sits on the spike-forward branch and the cloud bends around it.
#      That is enough to infer "BFS diffusion with blockers" rather than either the original ray
#      rule or a plain obstacle-free diffusion rule.
#
# f) Visual cleanliness
#    - The generator drops the original stray mouth-side noise. In this combo the obstacle is the
#      only extra non-cup marker, so keeping the rest of the canvas clean makes the causal role of
#      that obstacle legible.
#
# g) Color inferability
#    - No color remapping is introduced. The content color is copied to the BFS cloud; obstacle
#      colors are arbitrary non-bg colors but their role is inferred geometrically, not numerically.
#
# Audit of new solver assumptions
# - Each acted-on cup still has exactly one candidate tail whose opposite-side flood sees content.
# - The cell immediately outside the spike is background and lies outside the cup bbox.
# - Every cup has an obstacle on its forward BFS branch, and there remain at least 4 reachable BFS
#   cells despite that obstacle.
# - Obstacles are isolated from cup-border components, so component-based cup detection is stable.

from __future__ import annotations

from collections import Counter, deque
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
LEFT_TURN = {"U": "L", "L": "D", "D": "R", "R": "U"}
RIGHT_TURN = {"U": "R", "R": "D", "D": "L", "L": "U"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood inside the bbox, seeded from one edge and blocked by border cells."""
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


def _expansion_dirs(tail_dir: str) -> list[tuple[int, int]]:
    """Deterministic tail-relative neighbor order for BFS."""
    order = [tail_dir, LEFT_TURN[tail_dir], RIGHT_TURN[tail_dir], OPPOSITE[tail_dir]]
    return [DIR4[d] for d in order]


def _bfs_fill_cells(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    tail_dir: str,
    blocked_mask: np.ndarray,
    limit: int | None = None,
) -> list[tuple[int, int]]:
    """BFS over background cells outside the current cup bbox."""
    H, W = shape
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    if not (0 <= sr < H and 0 <= sc < W):
        return []
    if r0 <= sr <= r1 and c0 <= sc <= c1:
        return []
    if bool(blocked_mask[sr, sc]):
        return []

    out: list[tuple[int, int]] = []
    visited = {seed}
    q = deque([seed])
    nbrs = _expansion_dirs(tail_dir)

    while q and (limit is None or len(out) < limit):
        r, c = q.popleft()
        out.append((r, c))
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                continue
            if bool(blocked_mask[nr, nc]):
                continue
            visited.add((nr, nc))
            q.append((nr, nc))
    return out


def _has_off_axis_spread(
    fill_cells: list[tuple[int, int]],
    seed: tuple[int, int],
    tail_dir: str,
) -> bool:
    """Require at least two filled cells off the main tail axis."""
    sr, sc = seed
    if tail_dir in ("U", "D"):
        off_axis = sum(c != sc for _, c in fill_cells)
    else:
        off_axis = sum(r != sr for r, _ in fill_cells)
    return off_axis >= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - detect each cup exactly as in the original task
    - erase its mouth content
    - launch BFS from the cell just outside the spike
    - BFS can only traverse background cells outside the cup bbox
    - any non-bg pixel in the input acts as an obstacle; blocked branches stop there
    - paint the first N reachable cells, where N is the number of erased content pixels
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
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
                (direction, cells[0]) for direction, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for direction, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[direction])
                cnt = Counter()
                for r, c in mouth:
                    value = int(g[r, c])
                    if value != bg and value != int(border_col):
                        cnt[value] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((direction, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIR4[tail_dir]
            seed = (spike[0] + dr, spike[1] + dc)
            blocked_mask = (g != bg)
            fill_cells = _bfs_fill_cells(
                (H, W),
                bbox,
                seed,
                tail_dir,
                blocked_mask,
                limit=len(content_cells),
            )
            for r, c in fill_cells:
                out[r, c] = content_col

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MIN = 4
CONTENT_COUNT_MAX = 7

GEN_RETRIES_PRIMARY = 600
GEN_RETRIES_FALLBACK = 600


def _build_canonical_cup(
    L: int,
    R: int,
    hL: int,
    hR: int,
    color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical cup whose tail points upward."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1:L + hL + 1, 0] = color
    if hR:
        patch[R + 1:R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(
    patch: np.ndarray,
    border_color: int,
    opposite_side: str,
) -> list[tuple[int, int]]:
    """Flood-fill the cup mouth inside a canonical or rotated patch."""
    h, w = patch.shape
    border_mask = (patch == border_color)
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
    """Rotate the canonical upward cup to the desired tail direction."""
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
    """Sample canonical cup parameters that fit the bbox bounds after rotation."""
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


def _choose_obstacle_color(
    rng: np.random.Generator,
    border_color: int,
    content_color: int,
) -> int:
    choices = [c for c in range(1, 10) if c not in {border_color, content_color}]
    return int(rng.choice(choices))


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    occ_cup: np.ndarray,
    occ_fill: np.ndarray,
    occ_obstacle: np.ndarray,
) -> tuple[int, int, tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    """Find a placement whose BFS fill is visibly bent by an obstacle."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    occ_fill_dilated = ndimage.binary_dilation(occ_fill, structure=_STRUCT8)
    occ_obstacle_dilated = ndimage.binary_dilation(occ_obstacle, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_fill_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_obstacle_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue

        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        seed = (sr_abs + dr, sc_abs + dc)
        obstacle_cell = (sr_abs + 2 * dr, sc_abs + 2 * dc)

        if not (0 <= seed[0] < H and 0 <= seed[1] < W):
            continue
        if not (0 <= obstacle_cell[0] < H and 0 <= obstacle_cell[1] < W):
            continue
        if bool(occ_cup[seed]) or bool(occ_fill[seed]) or bool(occ_obstacle[seed]):
            continue
        if bool(occ_cup_dilated[obstacle_cell]) or bool(occ_fill_dilated[obstacle_cell]) or bool(occ_obstacle_dilated[obstacle_cell]):
            continue

        blocked_open = occ_cup | occ_fill | occ_obstacle
        if bool(blocked_open[obstacle_cell]):
            continue

        blocked_with_obstacle = blocked_open.copy()
        blocked_with_obstacle[obstacle_cell] = True

        open_order = _bfs_fill_cells(
            (H, W),
            bbox,
            seed,
            direction,
            blocked_open,
            limit=CONTENT_COUNT_MAX,
        )
        fill_order = _bfs_fill_cells(
            (H, W),
            bbox,
            seed,
            direction,
            blocked_with_obstacle,
            limit=CONTENT_COUNT_MAX,
        )

        if len(open_order) < CONTENT_COUNT_MIN or len(fill_order) < CONTENT_COUNT_MIN:
            continue
        if open_order[:CONTENT_COUNT_MIN] == fill_order[:CONTENT_COUNT_MIN]:
            continue
        if not _has_off_axis_spread(fill_order[:CONTENT_COUNT_MIN], seed, direction):
            continue

        preview_cells = fill_order[:CONTENT_COUNT_MAX]
        if any(bool(occ_cup_dilated[r, c]) or bool(occ_fill_dilated[r, c]) or bool(occ_obstacle_dilated[r, c]) for r, c in preview_cells):
            continue

        return r0, c0, seed, obstacle_cell, fill_order
    return None


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_fill: np.ndarray,
    occ_obstacle: np.ndarray,
) -> dict | None:
    """Build, place, and commit one cup plus its obstacle-bent BFS cloud."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < CONTENT_COUNT_MIN:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    placement = _try_place_cup(rng, H, W, patch, spike, direction, occ_cup, occ_fill, occ_obstacle)
    if placement is None:
        return None
    r0, c0, seed, obstacle_cell, fill_order = placement

    max_count = min(CONTENT_COUNT_MAX, len(mouth), len(fill_order))
    if max_count < CONTENT_COUNT_MIN:
        return None
    content_count = int(rng.integers(CONTENT_COUNT_MIN, max_count + 1))
    fill_cells = fill_order[:content_count]
    if not _has_off_axis_spread(fill_cells[:CONTENT_COUNT_MIN], seed, direction):
        return None

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True

    obstacle_color = _choose_obstacle_color(rng, border_color, content_color)
    orow, ocol = obstacle_cell
    canvas_in[orow, ocol] = obstacle_color
    canvas_out[orow, ocol] = obstacle_color
    occ_obstacle[orow, ocol] = True

    for r, c in fill_cells:
        canvas_out[r, c] = content_color
        occ_fill[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "obstacle_cell": obstacle_cell,
        "fill_cells": fill_cells,
    }


def _scatter_content(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    cups_meta: list[dict],
) -> bool:
    """Scatter each cup's content pixels inside its mouth."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for idx in idxs:
            r, c = mouth_abs[int(idx)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single validated output-first generation attempt."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_fill = np.zeros((H, W), dtype=bool)
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
            occ_cup,
            occ_fill,
            occ_obstacle,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
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
    raise RuntimeError(
        f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts"
    )
