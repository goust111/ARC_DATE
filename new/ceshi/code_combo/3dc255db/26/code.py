# Combo variant of 3dc255db (index 26)
# combo_spec: D9/0+D17/1
# Name: 矩形框架+特殊色尖刺

"""
Design analysis for the fused variant
=====================================

a) Concrete behavior and trace
- Original task: a same-color Y/funnel border component contains content, and the solver
  infers the tail by finding the bbox edge that has exactly one border pixel.
- D9/0 change: the cup becomes a strict 3-sided rectangular frame with one open side.
  Example for tail=U: top side closed, left/right sides closed, bottom side open except
  for the two corner pixels contributed by the vertical walls.
- D17/1 fusion: the tail spike is no longer inferred from edge counts. Instead, the
  center cell of the closed side is recolored to a shared special spike color.
  Example for tail=U, width=5, height=4:
      input top row: [border, border, spike, border, border]
      side walls: left/right columns in border color
      content: a few pixels inside the open mouth
  The output erases the interior content and emits a straight ray from the spike outward.
- Key difference: after recoloring the center spike, the border-color cells split into two
  disconnected pieces. Therefore the original "one connected border component per cup"
  assumption is no longer valid; detection has to start from the spike marker.

b) Complexity calibration
- The rectangular frame is visually distinct from the original funnel, so the D9 change is
  obvious in 2-3 examples.
- The special spike color is global and reused across all cups in one grid. This makes the
  D17 change observable without exploding the palette.
- Width/height/orientation/content count/number of cups still vary, so the task keeps
  enough freedom and does not collapse to a single rigid template.

c) Hidden-assumption audit against the original solver
- Iterating same-color connected border components is invalid now because removing the
  center spike from the border color disconnects the frame halves.
- "Candidate spike = unique border pixel on one bbox edge" is invalid. The spike is a
  dedicated color marker and the closed side may have many border pixels.
- Flooding from the open side is still valid, but the flood barrier must include both the
  border color and the special spike cell.
- A new solver assumption appears: for a real cup, the special-colored cell is exactly the
  center of the closed side of a strict 3-sided rectangle. The generator guarantees that.

d) Placement legality
- Cups keep the original 10-13 canvas and 1-3 object count.
- Cup bboxes remain non-overlapping with at least one cell of separation.
- Each spike's outward ray is checked to be clear to the grid edge before placement.
- Stray pixels never use the spike color, so spike markers stay unambiguous.

e) Visual observability
- A human sees a U/C-shaped rectangular frame instead of a funnel, and also sees a single
  conspicuous off-color pixel sitting at the center of the closed side.
- Across examples, that special-colored center is always where the output ray begins.
  No inference depends on numeric color ids or hidden ordering.

f) Visual cleanliness
- Only three semantic roles are present per cup: border, spike marker, content.
- The spike color is forbidden from content/stray usage, so there is no color-role overlap.
- Paths and cup bboxes are kept disjoint, preventing merged rays or ambiguous frames.

g) Color inferability
- The special spike color is reused as a consistent marker role, not a random recolor.
- Border colors remain unique per cup; content colors may repeat, but spike color may not
  be reused as content or stray. That keeps the new color role easy to infer.

Post-change solver assumptions checked against the generator
------------------------------------------------------------
- Every true cup contains exactly one spike marker pixel. `generate_input()` constructs one.
- The spike sits on the center of the closed side, so that side length is odd. The generator
  samples only canonical widths that preserve a unique center after rotation.
- All non-background interior pixels in a cup mouth share one content color. The generator
  scatters only one content color per cup.
- No other special-color pixels appear outside spike positions. Stray pixels reuse only
  border/content colors, never the spike color.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, get_bg


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood inside ``bbox``, seeded from non-border cells on one bbox edge."""
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


def _rect_frame_cells(
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> set[tuple[int, int]]:
    """All cells belonging to the rectangular 3-sided cup, including the spike slot."""
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    if tail_dir == "U":
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
    elif tail_dir == "D":
        cells.update((r1, c) for c in range(c0, c1 + 1))
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
    elif tail_dir == "L":
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
    else:
        cells.update((r, c1) for r in range(r0, r1 + 1))
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
    return cells


def _spike_from_bbox_center(
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> tuple[int, int] | None:
    """Center cell of the closed side; returns None when that side length is even."""
    r0, c0, r1, c1 = bbox
    if tail_dir in ("U", "D"):
        width = c1 - c0 + 1
        if width % 2 == 0:
            return None
        return (r0 if tail_dir == "U" else r1, c0 + width // 2)

    height = r1 - r0 + 1
    if height % 2 == 0:
        return None
    return (r0 + height // 2, c0 if tail_dir == "L" else c1)


def _run_len(
    grid: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    color: int,
) -> int:
    """Count contiguous cells of ``color`` starting from the next step along ``(dr, dc)``."""
    H, W = grid.shape
    rr, cc = r + dr, c + dc
    n = 0
    while 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == int(color):
        n += 1
        rr += dr
        cc += dc
    return n


def _match_rect_cup_at_spike(
    grid: np.ndarray,
    spike: tuple[int, int],
    tail_dir: str,
    bg: int,
) -> dict | None:
    """
    Validate whether ``spike`` is the special-colored center of a 3-sided rectangular cup.

    Detection starts from the spike marker because the border-color cells alone are no longer
    a single connected component after the center cell is recolored.
    """
    H, W = grid.shape
    sr, sc = spike
    spike_color = int(grid[sr, sc])
    if spike_color == int(bg):
        return None

    if tail_dir in ("U", "D"):
        if not (0 <= sc - 1 < W and 0 <= sc + 1 < W):
            return None
        border_color = int(grid[sr, sc - 1])
        if border_color in (bg, spike_color) or int(grid[sr, sc + 1]) != border_color:
            return None
        left_run = _run_len(grid, sr, sc, 0, -1, border_color)
        right_run = _run_len(grid, sr, sc, 0, 1, border_color)
        if left_run != right_run or left_run < 1:
            return None
        c0, c1 = sc - left_run, sc + right_run

        if tail_dir == "U":
            left_ext = _run_len(grid, sr, c0, 1, 0, border_color)
            right_ext = _run_len(grid, sr, c1, 1, 0, border_color)
            if left_ext != right_ext or left_ext < 2:
                return None
            r0, r1 = sr, sr + left_ext
        else:
            left_ext = _run_len(grid, sr, c0, -1, 0, border_color)
            right_ext = _run_len(grid, sr, c1, -1, 0, border_color)
            if left_ext != right_ext or left_ext < 2:
                return None
            r0, r1 = sr - left_ext, sr
    else:
        if not (0 <= sr - 1 < H and 0 <= sr + 1 < H):
            return None
        border_color = int(grid[sr - 1, sc])
        if border_color in (bg, spike_color) or int(grid[sr + 1, sc]) != border_color:
            return None
        up_run = _run_len(grid, sr, sc, -1, 0, border_color)
        down_run = _run_len(grid, sr, sc, 1, 0, border_color)
        if up_run != down_run or up_run < 1:
            return None
        r0, r1 = sr - up_run, sr + down_run

        if tail_dir == "L":
            top_ext = _run_len(grid, r0, sc, 0, 1, border_color)
            bot_ext = _run_len(grid, r1, sc, 0, 1, border_color)
            if top_ext != bot_ext or top_ext < 2:
                return None
            c0, c1 = sc, sc + top_ext
        else:
            top_ext = _run_len(grid, r0, sc, 0, -1, border_color)
            bot_ext = _run_len(grid, r1, sc, 0, -1, border_color)
            if top_ext != bot_ext or top_ext < 2:
                return None
            c0, c1 = sc - top_ext, sc

    bbox = (r0, c0, r1, c1)
    spike_expected = _spike_from_bbox_center(bbox, tail_dir)
    if spike_expected != spike:
        return None

    frame_all = _rect_frame_cells(bbox, tail_dir)
    border_cells = frame_all - {spike}

    observed_border = set()
    observed_spike = set()
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            v = int(grid[rr, cc])
            if v == border_color:
                observed_border.add((rr, cc))
            elif v == spike_color:
                observed_spike.add((rr, cc))
    if observed_border != border_cells or observed_spike != {spike}:
        return None

    mouth = _flood_from_bbox_side(frame_all, bbox, OPPOSITE[tail_dir])
    if len(mouth) < 3:
        return None

    cnt = Counter()
    for rr, cc in mouth:
        v = int(grid[rr, cc])
        if v != bg:
            cnt[v] += 1
    if len(cnt) != 1:
        return None

    content_color = int(next(iter(cnt)))
    if content_color in (border_color, spike_color):
        return None
    content_cells = sorted((rr, cc) for rr, cc in mouth if int(grid[rr, cc]) == content_color)
    if not content_cells:
        return None

    return {
        "tail_dir": tail_dir,
        "spike": spike,
        "bbox": bbox,
        "border_color": border_color,
        "spike_color": spike_color,
        "content_color": content_color,
        "content_cells": content_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Cups are 3-sided rectangular frames.
    - The spike is a special-colored pixel at the center of the closed side.
    - Content inside the open mouth is erased and ejected outward from that spike.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for sr, sc in map(tuple, np.argwhere(g != bg)):
        matches = []
        for tail_dir in ("U", "D", "L", "R"):
            match = _match_rect_cup_at_spike(g, (int(sr), int(sc)), tail_dir, bg)
            if match is not None:
                matches.append(match)
        if len(matches) != 1:
            continue

        info = matches[0]
        for rr, cc in info["content_cells"]:
            out[rr, cc] = bg

        dr, dc = DIR4[info["tail_dir"]]
        sr0, sc0 = info["spike"]
        for step in range(1, len(info["content_cells"]) + 1):
            rr = sr0 + dr * step
            cc = sc0 + dc * step
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = info["content_color"]

    return out


def _build_canonical_cup(
    height: int,
    width: int,
    border_color: int,
    spike_color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a 3-sided rectangular cup with tail pointing UP and a marked spike."""
    assert height >= 3 and width >= 3 and width % 2 == 1
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = border_color
    patch[:, 0] = border_color
    patch[:, width - 1] = border_color
    spike = (0, width // 2)
    patch[spike] = spike_color
    return patch, spike


def _mouth_interior_coords(
    patch: np.ndarray,
    opposite_side: str,
) -> list[tuple[int, int]]:
    """Flood-fill the cup interior from the open side; any non-zero cell acts as a wall."""
    h, w = patch.shape
    wall = patch != 0
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not wall[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not wall[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not wall[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not wall[r, w - 1]]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not wall[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate the canonical UP-oriented patch and its metadata."""
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


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int] | None:
    """Pick canonical (height, width) for a 3-sided rectangular cup."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    shapes = [
        (h, w)
        for h in range(3, max_h_canon + 1)
        for w in range(3, max_w_canon + 1, 2)
        if (h - 1) * (w - 2) >= 3
    ]
    if not shapes:
        return None
    return shapes[int(rng.integers(0, len(shapes)))]


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
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a placement with a 1-cell bbox gap and a clear outward ray."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=STRUCT8)

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
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


def _stray_candidate_mask(
    H: int,
    W: int,
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> np.ndarray:
    """Cells on the mouth side of the cup bbox, restricted to its perpendicular span."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif tail_dir == "D":
        mask[:r0, c0:c1 + 1] = True
    else:
        mask[r1 + 1:, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    """Optionally place one isolated stray pixel; spike color is never used here."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    if rng.random() < STRAY_USE_BORDER_PROB:
        stray_color = meta["border_color"]
    else:
        stray_color = meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=STRUCT8)
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
    spike_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build, place, and commit one cup plus its output ray."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    height, width = shape

    patch_canon, spike_canon = _build_canonical_cup(height, width, border_color, spike_color)
    mouth_canon = _mouth_interior_coords(patch_canon, opposite_side="D")
    if len(mouth_canon) < 3:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_in[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]][patch_mask] = patch[patch_mask]
    canvas_out[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]][patch_mask] = patch[patch_mask]
    occ_cup[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]] |= patch_mask

    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "spike_color": spike_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1),
        "spike_abs": (r0 + spike[0], c0 + spike[1]),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    cups_meta: list[dict],
) -> bool:
    """Scatter each cup's content pixels inside its mouth interior."""
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
    """Single output-first generation attempt with oracle verification."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    spike_color = int(palette.pop())

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    cups_meta: list[dict] = []

    for _ in range(num_cups):
        if not palette:
            return None
        border_color = int(palette.pop())
        content_choices = [c for c in range(1, 10) if c not in {0, spike_color, border_color}]
        if not content_choices:
            return None
        content_color = int(content_choices[int(rng.integers(0, len(content_choices)))])

        meta = _add_cup(
            rng,
            H,
            W,
            border_color,
            content_color,
            spike_color,
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
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

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
