import random
from collections import Counter

import numpy as np


BG = 0
ANCHOR_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
PATCH_SIZES = (5, 7)
GRID_SIDE_RANGE = (18, 31)  # numpy-style exclusive high is not used here
SLOT_GAP = 2
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_symmetric(patch: np.ndarray, axis: str) -> bool:
    if axis == "vertical":
        return np.array_equal(patch, np.fliplr(patch))
    return np.array_equal(patch, np.flipud(patch))


def _mismatch_pairs(
    patch: np.ndarray, axis: str
) -> list[tuple[tuple[int, int], int, tuple[int, int], int]]:
    h, w = patch.shape
    pairs: list[tuple[tuple[int, int], int, tuple[int, int], int]] = []
    if axis == "vertical":
        for r in range(h):
            for c in range(w // 2):
                mc = w - 1 - c
                left = int(patch[r, c])
                right = int(patch[r, mc])
                if left != right:
                    pairs.append(((r, c), left, (r, mc), right))
    else:
        for r in range(h // 2):
            mr = h - 1 - r
            for c in range(w):
                top = int(patch[r, c])
                bottom = int(patch[mr, c])
                if top != bottom:
                    pairs.append(((r, c), top, (mr, c), bottom))
    return pairs


def _try_candidate(patch: np.ndarray, axis: str, side: int):
    mismatches = _mismatch_pairs(patch, axis)
    if not (1 <= len(mismatches) <= 3):
        return None

    coords = [entry[0] if side < 0 else entry[2] for entry in mismatches]
    colors = [entry[1] if side < 0 else entry[3] for entry in mismatches]
    if len(set(colors)) != 1:
        return None

    defect_color = colors[0]
    if defect_color == BG:
        return None

    defect_positions = [tuple(map(int, rc)) for rc in np.argwhere(patch == defect_color)]
    if sorted(defect_positions) != sorted(coords):
        return None

    repaired = patch.copy()
    for (neg_pos, neg_color, pos_pos, pos_color) in mismatches:
        if side < 0:
            repaired[neg_pos] = pos_color
        else:
            repaired[pos_pos] = neg_color

    if not _is_symmetric(repaired, axis):
        return None

    other_axis = "horizontal" if axis == "vertical" else "vertical"
    score = (len(mismatches), int(_is_symmetric(repaired, other_axis)))
    return {
        "axis": axis,
        "side": side,
        "defect_color": defect_color,
        "mismatch_count": len(mismatches),
        "repaired": repaired,
        "score": score,
    }


def _infer_repaired_patch(patch: np.ndarray):
    candidates = []
    for axis in ("vertical", "horizontal"):
        for side in (-1, 1):
            cand = _try_candidate(patch, axis, side)
            if cand is not None:
                candidates.append(cand)
    assert candidates, "failed to infer a unique near-symmetry repair"
    candidates.sort(key=lambda item: item["score"])
    best = candidates[0]
    if len(candidates) > 1:
        assert candidates[1]["score"] != best["score"], "ambiguous symmetry repair"
    return best


def _stamp_centered(out: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    h, w = patch.shape
    cr, cc = center
    top = cr - h // 2
    left = cc - w // 2
    assert 0 <= top and 0 <= left and top + h <= out.shape[0] and left + w <= out.shape[1]
    target = out[top : top + h, left : left + w]
    mask = patch != BG
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the almost-symmetric seed pattern, then copy it onto each anchor as a direction-flipped replica.

    Input structure:
    - The background is black.
    - One multicolor connected seed pattern occupies a roughly 5x5 to 7x7 area.
    - The seed is nearly mirror-symmetric horizontally or vertically, except for
      1-3 pixels of one unique defect color that all lie on the same side of the
      hidden symmetry axis.
    - Several isolated grey anchor pixels lie away from the seed.

    Rule:
    - Treat the largest non-black connected component as the seed and the other
      singleton components as anchors.
    - Compare the seed against its horizontal and vertical mirror pairings to
      find the axis where every mismatch can be explained by one defect color on
      one side only.
    - Repair the seed by replacing each defect-color pixel with the color at its
      mirrored counterpart, yielding a perfectly symmetric patch.
    - Replace the original seed with that repaired patch.
    - Center a copy of the repaired patch on every anchor, mirrored once along
      the inferred growth axis, and let the copy overwrite the grey anchor.

    Key insight:
    - The anomalous color is not noise; its shared side relative to the best
      mirror axis identifies how the seed should be repaired before duplication.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != BG)
    assert comps, "expected foreground pixels"
    comps.sort(key=len, reverse=True)
    assert len(comps[0]) > 1, "largest foreground component should be the seed"
    assert all(len(comp) == 1 for comp in comps[1:]), "non-seed objects must be singleton anchors"

    seed = comps[0]
    anchors = [comp[0] for comp in comps[1:]]
    assert anchors, "expected at least one anchor"

    r0, c0, r1, c1 = _bbox(seed)
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    info = _infer_repaired_patch(patch)
    repaired = info["repaired"]
    flipped = np.fliplr(repaired) if info["axis"] == "vertical" else np.flipud(repaired)

    out = g.copy()
    for r, c in anchors:
        out[r, c] = BG
    out[r0 : r1 + 1, c0 : c1 + 1] = repaired
    for anchor in anchors:
        _stamp_centered(out, flipped, anchor)
    return out


def _build_clean_patch(size: int, axis: str, colors: list[int]) -> np.ndarray | None:
    for _ in range(120):
        patch = np.empty((size, size), dtype=int)
        if axis == "vertical":
            half = np.array(
                [[random.choice(colors) for _ in range(size // 2)] for _ in range(size)],
                dtype=int,
            )
            center = np.array([random.choice(colors) for _ in range(size)], dtype=int)
            patch[:, : size // 2] = half
            patch[:, size // 2] = center
            patch[:, size // 2 + 1 :] = half[:, ::-1]
        else:
            half = np.array(
                [[random.choice(colors) for _ in range(size)] for _ in range(size // 2)],
                dtype=int,
            )
            center = np.array([random.choice(colors) for _ in range(size)], dtype=int)
            patch[: size // 2, :] = half
            patch[size // 2, :] = center
            patch[size // 2 + 1 :, :] = half[::-1, :]

        if len(set(map(int, patch.ravel()))) < 3:
            continue
        other = "horizontal" if axis == "vertical" else "vertical"
        if len(_mismatch_pairs(patch, other)) < 7:
            continue
        return patch
    return None


def _apply_defects(
    clean: np.ndarray, axis: str, side: int, defect_color: int, defect_count: int
) -> np.ndarray | None:
    patch = clean.copy()
    size = clean.shape[0]
    mid = size // 2
    if axis == "vertical":
        cols = range(0, mid) if side < 0 else range(mid + 1, size)
        cells = [(r, c) for r in range(size) for c in cols]
    else:
        rows = range(0, mid) if side < 0 else range(mid + 1, size)
        cells = [(r, c) for r in rows for c in range(size)]
    if len(cells) < defect_count:
        return None
    chosen = random.sample(cells, defect_count)
    for r, c in chosen:
        patch[r, c] = defect_color
    return patch


def _slot_starts(limit: int, size: int) -> list[int]:
    offset = random.randint(0, SLOT_GAP)
    starts = list(range(offset, limit - size + 1, size + SLOT_GAP))
    return starts


def _build_layout(size: int):
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    if h > 30 or w > 30:
        return None
    row_starts = _slot_starts(h, size)
    col_starts = _slot_starts(w, size)
    slots = [(r, c) for r in row_starts for c in col_starts]
    if len(slots) < 3:
        return None
    max_anchors = min(4, len(slots) - 1)
    if max_anchors < 2:
        return None
    anchor_count = random.randint(2, max_anchors)
    chosen = random.sample(slots, anchor_count + 1)
    seed_top_left = chosen[0]
    anchor_centers = [(r + size // 2, c + size // 2) for r, c in chosen[1:]]
    return h, w, seed_top_left, anchor_centers


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(PATCH_SIZES)
    axis = random.choice(["vertical", "horizontal"])
    side = random.choice([-1, 1])

    motif_colors = random.sample(PALETTE, random.randint(3, 5))
    defect_choices = [color for color in PALETTE if color not in motif_colors and color != ANCHOR_COLOR]
    if not defect_choices:
        return None
    defect_color = random.choice(defect_choices)

    clean = _build_clean_patch(size, axis, motif_colors)
    if clean is None:
        return None

    defect_count = random.randint(1, 3)
    defective = _apply_defects(clean, axis, side, defect_color, defect_count)
    if defective is None:
        return None

    inferred = _infer_repaired_patch(defective)
    if inferred["axis"] != axis or inferred["side"] != side:
        return None
    if not np.array_equal(inferred["repaired"], clean):
        return None

    layout = _build_layout(size)
    if layout is None:
        return None
    h, w, seed_top_left, anchor_centers = layout

    grid = np.full((h, w), BG, dtype=int)
    sr, sc = seed_top_left
    grid[sr : sr + size, sc : sc + size] = defective
    for r, c in anchor_centers:
        grid[r, c] = ANCHOR_COLOR

    expected = np.full((h, w), BG, dtype=int)
    expected[sr : sr + size, sc : sc + size] = clean
    flipped = np.fliplr(clean) if axis == "vertical" else np.flipud(clean)
    for r, c in anchor_centers:
        _stamp_centered(expected, flipped, (r, c))

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
