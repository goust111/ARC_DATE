# Combo variant of 3dc255db (index 30)
# combo_spec: D13/2+D14/2
# Name: 锥形弹射+双向弹射
#
# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Cup detection stays the same as the original task: find a hollow border component, infer
#      the unique spike direction from the bbox edge with exactly one non-corner border cell, and
#      flood from the opposite side to recover the mouth interior plus its single content color.
#    - The fused change happens only after content counting. If a cup contains n content pixels:
#        tail_count = ceil(n / 2)
#        mouth_count = floor(n / 2)
#      Odd counts still favor the tail by one pixel group.
#    - Unlike D14/2's thin bidirectional rays, each side now uses its allocated count as cone
#      depth. Layer t sits t cells away from the launch anchor and has width 2*t-1 on the
#      perpendicular axis. The tail cone launches from the spike; the mouth cone launches from
#      the mouth-edge anchor opposite the spike and expands outward from the cup.
#    - Example, tail='U', n=5:
#        original: erase 5 interior pixels, draw 5 cells upward from the spike
#        fused   : erase 5 interior pixels
#                  tail 3-layer cone above the spike with widths 1,3,5
#                  mouth 2-layer cone below the mouth with widths 1,3
#      The output becomes a bow-tie / hourglass silhouette rather than a single emitted line.
#
# b) Complexity calibration
#    - If n were allowed to be 2 or 3, one side could remain a single-pixel line and the sample
#      could visually collapse toward the plain bidirectional variant. To keep the fusion visible,
#      the generator uses content counts 4..5 whenever possible, guaranteeing both sides have at
#      least two visible layers and width growth to 3+ pixels.
#    - Depth 4+ on both sides would make multi-cup placement unnecessarily fragile on 10..13-ish
#      canvases, so the count range is intentionally narrow.
#
# c) Implicit-assumption audit against the original main()
#    - Tail-candidate detection still relies on "one border pixel on one bbox edge". Cup borders
#      are unchanged, so this remains valid.
#    - Mouth recovery still assumes the opposite side is the only open side. The generator keeps
#      the original cup construction, so the flood remains reliable.
#    - The original draw loop assumed a one-cell-wide path and "one erased pixel -> one emitted
#      cell". That assumption is now false twice over: the count is first split into two branches,
#      then each branch expands into multi-cell cone layers.
#    - The original code did not need a mouth-side launch anchor. The fused solver now assumes a
#      deterministic mouth anchor can be derived from the spike axis and flooded mouth edge; the
#      generator guarantees this anchor exists.
#
# d) Placement legality
#    - Generator remains output-first. For each cup it picks a content count, computes both cones,
#      and rejects placements unless the cup bbox and both full cone footprints are inside bounds,
#      unclipped laterally, non-overlapping with prior cups, and non-overlapping with previously
#      reserved cone footprints.
#    - Stray pixels are forbidden from every reserved cone cell, not just the cup bbox.
#
# e) Visual observability
#    - Human solvers can see three repeated cues at once: content disappears from the cup,
#      emission happens on both sides, and each side widens with distance. Across 2-4 examples,
#      that supports the inference "split the count, then build two cones" rather than merely
#      "draw two lines".
#
# f) Visual cleanliness
#    - The generator favors larger grids and rejects clipped cones, so emitted shapes stay clean.
#      Each cone uses the cup's own content color, and cones are spatially separated by the cup
#      itself, preventing mixed-object tangles.
#
# g) Color inferability
#    - No color remapping is introduced. The only rule-bearing feature is geometry: the same
#      content color fills both cones for that cup.
#
# Audit of new solver assumptions
# - Every acted-on cup has exactly one candidate direction whose opposite-side flood sees content.
# - The mouth-side anchor derived from the spike axis lies on the flooded mouth edge.
# - Cone cells intended for a cup are background in the input; the generator enforces this by
#   reserving both full cone footprints before scattering content or placing stray pixels.

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
PERP4 = {"U": (0, 1), "D": (0, 1), "L": (1, 0), "R": (1, 0)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood inside a bbox, seeded from one edge and blocked by border cells."""
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


def _split_content_count(n: int) -> tuple[int, int]:
    """Return (tail_count, mouth_count); odd counts favor the tail."""
    return (n + 1) // 2, n // 2


def _cone_layers_from_anchor(
    anchor: tuple[int, int],
    direction: str,
    depth: int,
    shape: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    """Build widening cone layers outward from an anchor cell."""
    H, W = shape
    ar, ac = anchor
    dr, dc = DIR4[direction]
    pr, pc = PERP4[direction]
    layers: list[list[tuple[int, int]]] = []
    for t in range(1, depth + 1):
        cr = ar + dr * t
        cc = ac + dc * t
        if not (0 <= cr < H and 0 <= cc < W):
            break
        layer: list[tuple[int, int]] = []
        for offset in range(-(t - 1), t):
            rr = cr + pr * offset
            rc = cc + pc * offset
            if 0 <= rr < H and 0 <= rc < W:
                layer.append((rr, rc))
        if layer:
            layers.append(layer)
    return layers


def _flatten_layers(layers: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for layer in layers for cell in layer]


def _layers_are_full(layers: list[list[tuple[int, int]]], depth: int) -> bool:
    """Require every intended layer to exist and keep its full 1,3,5,... width."""
    if len(layers) != depth:
        return False
    return all(len(layer) == 2 * idx - 1 for idx, layer in enumerate(layers, start=1))


def _mouth_anchor_from_spike(
    spike: tuple[int, int],
    bbox: tuple[int, int, int, int],
    tail_dir: str,
    mouth_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find the mouth-edge launch anchor opposite the spike, preferring exact axis alignment."""
    sr, sc = spike
    r0, c0, r1, c1 = bbox
    if tail_dir == "U":
        preferred = (r1, sc)
        edge_cells = [p for p in mouth_cells if p[0] == r1]
        key = lambda p: (abs(p[1] - sc), p[1])
    elif tail_dir == "D":
        preferred = (r0, sc)
        edge_cells = [p for p in mouth_cells if p[0] == r0]
        key = lambda p: (abs(p[1] - sc), p[1])
    elif tail_dir == "L":
        preferred = (sr, c1)
        edge_cells = [p for p in mouth_cells if p[1] == c1]
        key = lambda p: (abs(p[0] - sr), p[0])
    else:  # "R"
        preferred = (sr, c0)
        edge_cells = [p for p in mouth_cells if p[1] == c0]
        key = lambda p: (abs(p[0] - sr), p[0])

    if preferred in mouth_cells:
        return preferred
    if not edge_cells:
        return None
    return min(edge_cells, key=key)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule: each cup's content is split across tail-side and mouth-side cones.

    For a cup with n content pixels:
    - erase the n interior pixels
    - tail side gets ceil(n/2) cone layers launched from the spike
    - mouth side gets floor(n/2) cone layers launched from the opposite mouth-edge anchor
    - each layer widens by 2 each step away from the launch anchor
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
                mouth_anchor = _mouth_anchor_from_spike(spike, bbox, direction, mouth)
                if mouth_anchor is None:
                    continue
                passing.append((direction, spike, mouth_anchor, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, mouth_anchor, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            tail_count, mouth_count = _split_content_count(len(content_cells))
            tail_layers = _cone_layers_from_anchor(spike, tail_dir, tail_count, (H, W))
            mouth_layers = _cone_layers_from_anchor(mouth_anchor, OPPOSITE[tail_dir], mouth_count, (H, W))

            for layer in tail_layers:
                for r, c in layer:
                    out[r, c] = content_col
            for layer in mouth_layers:
                for r, c in layer:
                    out[r, c] = content_col

    return out


# ==== Generator configuration ====
GRID_H_RANGE = (11, 13)
GRID_W_RANGE = (11, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MIN = 4
CONTENT_COUNT_MAX = 5

STRAY_PROB = 0.35
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 500
GEN_RETRIES_FALLBACK = 500
ADD_CUP_ATTEMPTS = 80
# ==================================


def _build_canonical_cup(
    L: int,
    R: int,
    hL: int,
    hR: int,
    color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical cup whose tail points up."""
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
    """Rotate a canonical up-tail cup to a chosen tail direction."""
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
    """Pick a canonical cup shape that fits the original bbox limits after rotation."""
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
    mouth_anchor_rel: tuple[int, int],
    direction: str,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[list[tuple[int, int]]], list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    """Find a placement where the cup bbox and both full cones are conflict-free."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    tail_count, mouth_count = _split_content_count(content_count)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    mouth_dir = OPPOSITE[direction]

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        spike_abs = (r0 + spike[0], c0 + spike[1])
        mouth_anchor_abs = (r0 + mouth_anchor_rel[0], c0 + mouth_anchor_rel[1])

        tail_layers = _cone_layers_from_anchor(spike_abs, direction, tail_count, (H, W))
        mouth_layers = _cone_layers_from_anchor(mouth_anchor_abs, mouth_dir, mouth_count, (H, W))
        if not _layers_are_full(tail_layers, tail_count):
            continue
        if not _layers_are_full(mouth_layers, mouth_count):
            continue

        cone_cells = _flatten_layers(tail_layers) + _flatten_layers(mouth_layers)
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in cone_cells):
            continue
        return r0, c0, tail_layers, mouth_layers, cone_cells
    return None


def _stray_candidate_mask(
    H: int,
    W: int,
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, restricted to the perpendicular span."""
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
    """Optionally place one isolated stray pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in cup["cone_cells"]:
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
) -> dict | None:
    """Build one cup, place it, and commit its two cone footprints."""
    for _ in range(ADD_CUP_ATTEMPTS):
        direction = str(rng.choice(["U", "D", "L", "R"]))
        shape = _sample_cup_shape(rng, direction)
        if shape is None:
            continue
        L_arm, R_arm, hL, hR = shape

        patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
        mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
        max_count = min(CONTENT_COUNT_MAX, len(mouth_canon))
        if max_count < CONTENT_COUNT_MIN:
            continue

        patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
        ph, pw = patch.shape
        mouth_anchor_rel = _mouth_anchor_from_spike(
            spike,
            (0, 0, ph - 1, pw - 1),
            direction,
            set(mouth),
        )
        if mouth_anchor_rel is None:
            continue

        counts = list(range(CONTENT_COUNT_MIN, max_count + 1))
        rng.shuffle(counts)
        for content_count in counts:
            placement = _try_place_cup(
                rng,
                H,
                W,
                patch,
                spike,
                mouth_anchor_rel,
                direction,
                content_count,
                occ_cup,
                occ_path,
            )
            if placement is None:
                continue
            r0, c0, tail_layers, mouth_layers, cone_cells = placement

            patch_mask = patch != 0
            canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
            canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
            occ_cup[r0:r0 + ph, c0:c0 + pw] = True
            for r, c in cone_cells:
                canvas_out[r, c] = content_color
                occ_path[r, c] = True

            tail_count, mouth_count = _split_content_count(content_count)
            return {
                "border_color": border_color,
                "content_color": content_color,
                "direction": direction,
                "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
                "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
                "content_count": content_count,
                "tail_count": tail_count,
                "mouth_count": mouth_count,
                "tail_layers": tail_layers,
                "mouth_layers": mouth_layers,
                "cone_cells": cone_cells,
            }
    return None


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's content pixels inside its mouth interior."""
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
    """Single output-first generation attempt for the fused cone-on-both-sides rule."""
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
