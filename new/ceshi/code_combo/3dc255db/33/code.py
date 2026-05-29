# Combo variant of 3dc255db (index 33)
# combo_spec: D15/2+D17/1
# Name: 边框色与内容色相同+特殊色尖刺
#
# Design analysis for this combo
# a) Concretization and behavior trace
#    - Original task: each cup uses one border color and one different content color. The solver
#      groups by border color, finds the unique spike edge, floods from the opposite side, erases
#      the differently colored content, then ejects a straight ray from the spike.
#    - Combo task: each cup body and its interior dots share one cup color, while the single
#      spike pixel uses a reserved special color. Example: a right-pointing cup may have a red
#      tip, blue frame arms/walls, and blue detached dots in the mouth. The output keeps the red
#      tip fixed, removes the blue interior dots, and writes a blue ray to the right of the tip.
#    - Difference: color no longer separates frame vs content; the solver must recover the frame
#      from spatial structure, starting from the specially colored tip.
# b) Complexity calibration
#    - If same-color content touched the frame, border and content would merge into the same
#      8-component and become ambiguous. The generator therefore keeps content cells 8-separated
#      from every border cell and from each other.
#    - To keep the new spike role obvious across 2-4 ARC examples, the generator uses one global
#      spike color per grid and reserves that color for spikes only.
#    - Different cups still keep different cup colors. This avoids an unnecessary extra source of
#      ambiguity; the fusion burden stays focused on "same-color body/content + special-color tip".
# c) Implicit-assumption audit of the original main()
#    - Iterating by border color is no longer valid: the true cup border is split because the tip
#      is a different color and the remaining cup-colored border halves are disconnected.
#    - "Content is non-bg and non-border-color" is invalid because content now equals the border
#      body color.
#    - The old spike disambiguation rule based on bbox-edge counts is unnecessary here. Instead,
#      a spike is a singleton non-bg component whose 8-neighborhood has exactly two non-bg cells,
#      both the same color, arranged in one of four diagonal patterns that encode direction.
#    - After the spike is known, the solver unions the cup-color components containing those two
#      arm seeds. That reconstructs the border body while excluding detached same-color content.
# d) Placement legality
#    - Cup bboxes still keep a 1-cell gap and all ejection paths remain clear to the edge.
#    - The global spike color is never reused for strays or content, so there are no fake tips.
#    - Same-color content dots are chosen only from mouth cells that are not 8-neighbors of the
#      border, so the arm-seed component union remains valid.
# e) Visual observability
#    - A human can directly see that every cup has a specially colored one-pixel tip and that the
#      removed dots have the same color as the surrounding cup body. The output preserves the tip
#      but moves only the body-colored dots into a ray.
# f) Visual cleanliness
#    - Reserving one shared spike color reduces color noise and makes the marked tip easy to scan.
#      Detached mouth dots and isolated strays stay outside cup borders and ejection paths.
# g) Color-rule inferability
#    - There is a simple color story: one reserved spike color marks all tips; each cup has one
#      body/content color; the ejected ray uses that body/content color. No hidden numeric color
#      ordering is used.
# h) Verification target
#    - Useful metrics are: (1) every generated cup has same-color content and a different-color
#      tip, and (2) the original solver should usually disagree because it expects the content to
#      have a color distinct from the border.
# i) New main() assumptions and generator guarantees
#    - Assumption: each real spike is a singleton component with exactly two same-colored diagonal
#      neighbors in one of four allowed patterns. Generator guarantee: every cup is built that
#      way and no stray uses the spike color.
#    - Assumption: the union of cup-color components containing the two arm seeds is the full
#      border body. Generator guarantee: same-color content dots are detached from the border.
#    - Assumption: cup-color cells inside the mouth flood but outside the recovered border body
#      are exactly the content dots. Generator guarantee: strays are outside all cup bboxes.

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
SPIKE_PATTERNS = {
    "U": ((1, -1), (1, 1)),
    "D": ((-1, -1), (-1, 1)),
    "L": ((-1, 1), (1, 1)),
    "R": ((-1, -1), (1, -1)),
}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood within bbox from the given side, stopping at border cells."""
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


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    H, W = shape
    return 0 <= r < H and 0 <= c < W


def _singleton_coords_by_color(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        for comp in components_mask(g == int(col), connectivity=8):
            if int(comp.shape[0]) == 1:
                coords.append((int(comp[0, 0]), int(comp[0, 1])))
    return coords


def _detect_spike(
    g: np.ndarray,
    spike: tuple[int, int],
) -> tuple[str, int, list[tuple[int, int]]] | None:
    """Recognize the special-color tip by its two same-colored diagonal arm neighbors."""
    r, c = spike
    spike_color = int(g[r, c])
    nbrs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) != 0:
                nbrs.append((rr, cc))
    if len(nbrs) != 2:
        return None

    neighbor_colors = {int(g[rr, cc]) for rr, cc in nbrs}
    if len(neighbor_colors) != 1:
        return None
    cup_color = next(iter(neighbor_colors))
    if cup_color == spike_color:
        return None

    rels = tuple(sorted((rr - r, cc - c) for rr, cc in nbrs))
    for direction, pattern in SPIKE_PATTERNS.items():
        if rels == tuple(sorted(pattern)):
            return direction, cup_color, nbrs
    return None


def _cup_components_for_color(g: np.ndarray, cup_color: int) -> list[set[tuple[int, int]]]:
    return [
        {(int(r), int(c)) for r, c in comp}
        for comp in components_mask(g == int(cup_color), connectivity=8)
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule: the cup body and content share one color; the single spike pixel has a reserved
    special color. The spike stays fixed, content disappears from the mouth, and a same-colored
    ray is ejected outward from the spike.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    processed_spikes: set[tuple[int, int]] = set()
    singleton_coords = _singleton_coords_by_color(g, bg)

    for spike in singleton_coords:
        if spike in processed_spikes:
            continue

        detected = _detect_spike(g, spike)
        if detected is None:
            continue
        tail_dir, cup_color, arm_seeds = detected

        border_body: set[tuple[int, int]] = set()
        for comp in _cup_components_for_color(g, cup_color):
            if any(seed in comp for seed in arm_seeds):
                border_body |= comp
        if not all(seed in border_body for seed in arm_seeds):
            continue

        full_border = set(border_body)
        full_border.add(spike)
        if len(full_border) < 5:
            continue

        border_coords = np.array(sorted(full_border), dtype=int)
        r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
        bbox = (r0, c0, r1, c1)

        if tail_dir == "U":
            tail_edge = [p for p in full_border if p[0] == r0]
        elif tail_dir == "D":
            tail_edge = [p for p in full_border if p[0] == r1]
        elif tail_dir == "L":
            tail_edge = [p for p in full_border if p[1] == c0]
        else:
            tail_edge = [p for p in full_border if p[1] == c1]
        if tail_edge != [spike]:
            continue

        mouth = _flood_from_bbox_side(full_border, bbox, OPPOSITE[tail_dir])
        content_cells = [p for p in mouth if int(g[p[0], p[1]]) == cup_color and p not in border_body]
        if not content_cells:
            continue

        for r, c in content_cells:
            out[r, c] = bg

        dr, dc = DIR4[tail_dir]
        sr, sc = spike
        for t in range(1, len(content_cells) + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = cup_color

        processed_spikes.add(spike)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300


def _build_canonical_cup(
    L: int,
    R: int,
    hL: int,
    hR: int,
    cup_color: int,
    spike_color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP and a special-color spike."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = spike_color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = cup_color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = cup_color
    if hL:
        g[L + 1:L + hL + 1, 0] = cup_color
    if hR:
        g[R + 1:R + hR + 1, w - 1] = cup_color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill cup interior from the mouth side, stopping at any nonzero border cell."""
    h, w = patch.shape
    border_mask = patch != 0
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
    """Rotate the canonical tail-up cup patch to the requested direction."""
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


def _best_isolated_subset(
    rng: np.random.Generator,
    cells: list[tuple[int, int]],
    tries: int = 8,
) -> list[tuple[int, int]]:
    """Greedy-search for a large subset with pairwise Chebyshev distance >= 2."""
    best: list[tuple[int, int]] = []
    for _ in range(tries):
        order = list(cells)
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for r, c in order:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
        if len(chosen) > len(best):
            best = chosen
    return best


def _safe_mouth_coords(patch: np.ndarray, mouth: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep mouth cells that are not 8-neighbors of the border, so content stays detached."""
    dilated = ndimage.binary_dilation(patch != 0, structure=_STRUCT8)
    return [(r, c) for r, c in mouth if not dilated[r, c]]


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
    """Find a placement with a 1-cell bbox gap and a clear ejection path."""
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
        path_cells: list[tuple[int, int]] = []
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
    """Cells on the mouth side of a cup bbox, restricted to the cup's perpendicular extent."""
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
    """Optionally place an isolated cup-color stray pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["cup_color"]

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
    cup_color: int,
    spike_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, choose detached same-color content dots, and commit the output state."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, cup_color, spike_color)
    mouth_canon = _mouth_interior_coords(patch_canon, opposite_side="D")
    if not mouth_canon:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    safe_mouth = _safe_mouth_coords(patch, mouth)
    isolated_pool = _best_isolated_subset(rng, safe_mouth)
    if not isolated_pool:
        return None

    max_content = min(CONTENT_COUNT_MAX, len(isolated_pool))
    min_content = 2 if max_content >= 2 else 1
    content_count = int(rng.integers(min_content, max_content + 1))
    chosen_idxs = rng.choice(len(isolated_pool), size=content_count, replace=False)
    content_local = [isolated_pool[int(i)] for i in chosen_idxs]

    ph, pw = patch.shape
    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = patch[patch_mask]
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = patch[patch_mask]
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in ejection_cells:
        canvas_out[r, c] = cup_color
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "cup_color": cup_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "content_abs": [(r0 + r, c0 + c) for r, c in content_local],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(canvas_in: np.ndarray, cups_meta: list[dict]) -> None:
    """Place each cup's detached same-color content dots."""
    for meta in cups_meta:
        for r, c in meta["content_abs"]:
            canvas_in[r, c] = meta["cup_color"]


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Generate one valid combo input grid and verify it with the solver."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if len(palette) < num_cups + 1:
        return None
    spike_color = int(palette.pop())

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if not palette:
            return None
        cup_color = int(palette.pop())
        meta = _add_cup(
            rng,
            H,
            W,
            cup_color,
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

    _scatter_content(canvas_in, cups_meta)
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
