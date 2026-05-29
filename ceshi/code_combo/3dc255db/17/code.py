# Combo variant of 3dc255db (index 17)
# combo_spec: D5/1+D6/0
# Name: 渐变色轨迹+内容变色保留
#
# Design analysis for the fused variant:
# - Concrete behavior:
#   Original: content pixels inside a cup are erased to bg, and the emitted line is a flat
#   content-colored segment.
#   New: those interior content pixels stay visible but recolor to the cup border color, while the
#   emitted segment becomes a distance-ordered ramp: near-spike cells keep the content color,
#   far-end cells become the border color, and middle cells use a bridge color.
#   Example with 4 content pixels and a downward tail:
#   input mouth = four red pixels inside a yellow cup.
#   original output = those four mouth cells become bg; emitted line = red red red red.
#   fused output = those four mouth cells become yellow; emitted line = red grey grey yellow.
# - Complexity calibration:
#   A fully free-form "interpolate any pair of ARC colors" rule would be hard to infer from only a
#   few examples. To keep the change visible and learnable, the generator reserves a single bridge
#   hue (prefer grey=5) and enforces emitted length >= 3, so examples repeatedly show the same
#   three-stage pattern: content -> bridge -> border.
# - Hidden assumptions checked against main():
#   1. Tail detection still relies on a unique candidate edge whose opposite-side flood reaches the
#      mouth content. The transform change does not alter the input, so this remains valid.
#   2. The mouth still contains exactly one non-border content color per cup. Generator keeps that.
#   3. Output-only bridge colors must not appear as input cup colors, or the visual cue becomes
#      muddled. Generator excludes the preferred bridge color from border/content sampling.
#   4. Gradient visibility needs actual run length >= 3; otherwise D5/1 degenerates to the old
#      behavior. Generator therefore requires both content_count >= 3 and visible path >= 3.
#   5. Since D6/0 depends on which exact mouth cells were occupied, output can no longer be fixed
#      before content scattering. Generator now chooses mouth positions and writes those cells to
#      input/output simultaneously.
# - Placement legality:
#   Cup bboxes still maintain a 1-cell gap, paths stay obstacle-free, and the input keeps emitted
#   path cells empty. The added recolor step only touches selected mouth cells, so it cannot create
#   overlap or out-of-bounds issues.
# - Visual observability / neatness:
#   The bridge color is reused consistently, the ejected line is straight and isolated, and the
#   recolored mouth cells sit inside the existing frame. That makes the two fused changes legible
#   without introducing extra marks or clutter.

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
CONTENT_COUNT_MIN = 3
VISIBLE_EJECT_MIN = 3
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
PREFERRED_BRIDGE_COLOR = 5
CUP_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
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


def _bridge_color(content_col: int, border_col: int) -> int:
    """Prefer a consistent bridge hue for learnability, but stay robust on arbitrary inputs."""
    for cand in (PREFERRED_BRIDGE_COLOR, 8, 7, 4, 3, 2, 1, 6, 9):
        if cand not in {int(content_col), int(border_col)}:
            return int(cand)
    return int(content_col)


def _trajectory_colors(content_col: int, border_col: int, length: int) -> list[int]:
    """Distance-ordered 3-stage ramp: content -> bridge -> border."""
    if length <= 0:
        return []
    if length == 1:
        return [int(content_col)]
    if length == 2:
        return [int(content_col), int(border_col)]

    lead = max(1, length // 3)
    trail = max(1, length // 3)
    if lead + trail >= length:
        lead, trail = 1, 1
    middle = length - lead - trail
    bridge = _bridge_color(content_col, border_col)
    return [int(content_col)] * lead + [int(bridge)] * middle + [int(border_col)] * trail


def _collect_cups(g: np.ndarray, bg: int) -> list[dict]:
    """Detect cup components together with their tail, content color, and occupied mouth cells."""
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
            candidates = [(d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners]
            if not candidates:
                continue

            passing = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != int(bg) and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == int(content_col)]
                passing.append(
                    {
                        "border_col": int(border_col),
                        "tail_dir": tail_dir,
                        "spike": (int(spike[0]), int(spike[1])),
                        "content_col": content_col,
                        "content_cells": content_cells,
                    }
                )

            if len(passing) == 1:
                cups.append(passing[0])
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject cup contents along the tail direction, but keep the mouth contents as recolored pixels.

    Fused rule:
    - Every detected content pixel inside the cup mouth becomes the cup border color.
    - The emitted straight segment uses a distance-ordered ramp from content color to border color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for cup in _collect_cups(g, bg):
        border_col = int(cup["border_col"])
        content_col = int(cup["content_col"])
        content_cells = list(cup["content_cells"])

        for r, c in content_cells:
            out[r, c] = border_col

        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        path = []
        for step in range(1, len(content_cells) + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < H and 0 <= cc < W):
                break
            path.append((int(rr), int(cc)))

        for (rr, cc), col in zip(path, _trajectory_colors(content_col, border_col, len(path))):
            out[rr, cc] = int(col)

    return out


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
        g[L + 1 : L + hL + 1, 0] = color
    if hR:
        g[R + 1 : R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the cup interior from the mouth side."""
    h, w = patch.shape
    border_mask = patch == int(border_color)
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
    """Rotate a canonical tail-UP patch to the requested tail direction."""
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
    """Sample a canonical cup shape that still fits the bbox limits after rotation."""
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
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place one cup so bbox gaps and the emitted path remain legal and visible."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((int(rr), int(cc)))
            rr += dr
            cc += dc
        if len(path_cells) < VISIBLE_EJECT_MIN:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        eject_len = min(int(content_count), len(path_cells))
        if eject_len < VISIBLE_EJECT_MIN:
            continue
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells past the mouth opening of a cup, restricted to its perpendicular span."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0 : r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0 : r1 + 1, c1 + 1 :] = True
    elif tail_dir == "D":
        mask[:r0, c0 : c1 + 1] = True
    else:
        mask[r1 + 1 :, c0 : c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> None:
    """Optionally add one isolated stray pixel on the mouth side of a cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = int(stray_color)
    canvas_out[r, c] = int(stray_color)


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
) -> dict | None:
    """Build and place one cup, writing border and emitted trajectory to the canvases."""
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
    content_count = int(rng.integers(CONTENT_COUNT_MIN, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = int(border_color)
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = int(border_color)
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True

    for (r, c), col in zip(ejection_cells, _trajectory_colors(content_color, border_color, len(ejection_cells))):
        canvas_out[r, c] = int(col)
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": int(border_color),
        "content_color": int(content_color),
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": int(content_count),
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter input content pixels and write their recolored output counterparts."""
    for meta in cups_meta:
        mouth_abs = list(meta["mouth_abs"])
        count = int(meta["content_count"])
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = int(meta["content_color"])
            canvas_out[r, c] = int(meta["border_color"])
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single generation attempt with oracle verification against main()."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = CUP_PALETTE.copy()
    rng.shuffle(palette)
    border_colors = palette[:num_cups]
    content_pool = palette[num_cups : num_cups + 2]
    if len(border_colors) < num_cups or len(content_pool) < 2:
        return None

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    cups_meta: list[dict] = []

    for border_color in border_colors:
        content_color = int(rng.choice(content_pool))
        meta = _add_cup(
            rng,
            H,
            W,
            int(border_color),
            content_color,
            canvas_in,
            canvas_out,
            occ_cup,
            occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if not _scatter_content(rng, canvas_in, canvas_out, cups_meta):
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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
