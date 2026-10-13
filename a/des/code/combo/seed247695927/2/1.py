from __future__ import annotations

import random

import numpy as np


BG = 0
OBJ = 5
LEGEND_COLORS = (2, 1, 3)  # left-to-right: horizontal only, vertical only, both
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 400


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    out = []
    for coords in comps:
        arr = np.asarray(coords, dtype=int)
        out.append(arr)
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _symmetry_type(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    if horizontal and vertical:
        return "both"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    raise AssertionError("object must match one of the three supported symmetry classes")


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _find_legend(grid: np.ndarray, object_color: int) -> dict[str, int]:
    h, w = grid.shape
    mask = (grid != BG) & (grid != object_color)
    comps = _find_components(mask)
    markers: list[tuple[int, int]] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        color = int(grid[coords[0, 0], coords[0, 1]])
        if len(coords) == 3 and r1 - r0 + 1 == 3 and c1 == c0 and r1 == h - 1:
            markers.append((c0, color))
    assert len(markers) == 3, "expected three bottom legend markers"
    markers.sort()
    return {
        "horizontal": markers[0][1],
        "vertical": markers[1][1],
        "both": markers[2][1],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object's outer contour using the bottom legend keyed by reflection symmetry.

    The bottom legend has three 3x1 colored markers ordered left-to-right:
    horizontal-only, vertical-only, both-axes. Each gray foreground object is classified by the
    reflection symmetries of its tight bounding-box mask. Only its 4-neighbor-exposed contour
    pixels are recolored; interior gray pixels and the legend stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    object_colors = [int(c) for c in np.unique(g) if c not in (BG, *LEGEND_COLORS)]
    assert object_colors == [OBJ], "expected a unique gray object color"
    legend = _find_legend(g, OBJ)

    out = g.copy()
    object_mask = g == OBJ
    comps = _find_components(object_mask)
    assert comps, "expected at least one gray object"

    for coords in comps:
        mask = _crop_mask(coords)
        sym = _symmetry_type(mask)
        outline = _outline_mask(mask)
        r0, c0, _r1, _c1 = _bbox(coords)
        local_rows, local_cols = np.nonzero(outline)
        out[r0 + local_rows, c0 + local_cols] = legend[sym]

    return out


def _connected_random_mask(h: int, w: int, target_area: int, must_touch: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = random.choice(tuple(must_touch))
    mask[start] = True
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    required = [cell for cell in must_touch if cell != start]
    random.shuffle(required)
    while required:
        candidates = []
        goal_r, goal_c = required[0]
        for r, c in frontier:
            dist = abs(r - goal_r) + abs(c - goal_c)
            candidates.append((dist, random.random(), r, c))
        _dist, _tie, r, c = min(candidates)
        frontier.discard((r, c))
        mask[r, c] = True
        add_frontier(r, c)
        if mask[goal_r, goal_c]:
            required.pop(0)

    while int(mask.sum()) < target_area and frontier:
        weighted = []
        for r, c in frontier:
            score = 0
            if r in (0, h - 1):
                score += 1
            if c in (0, w - 1):
                score += 1
            weighted.append((score, random.random(), r, c))
        best_score = max(item[0] for item in weighted)
        choices = [item for item in weighted if item[0] == best_score or random.random() < 0.35]
        _score, _tie, r, c = random.choice(choices)
        frontier.discard((r, c))
        mask[r, c] = True
        add_frontier(r, c)
    return mask


def _gen_both_mask() -> np.ndarray | None:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    qh = (h + 1) // 2
    qw = (w + 1) // 2
    quarter_area = random.randint(max(2, (qh * qw) // 3), qh * qw)
    must_touch = {(0, 0), (qh - 1, 0), (0, qw - 1)}
    quarter = _connected_random_mask(qh, qw, quarter_area, must_touch)
    top = np.concatenate([quarter, quarter[:, -2::-1] if w % 2 else quarter[:, ::-1]], axis=1)
    full = np.concatenate([top, top[-2::-1, :] if h % 2 else top[::-1, :]], axis=0)
    if _symmetry_type(full) != "both":
        return None
    return full


def _gen_horizontal_mask() -> np.ndarray | None:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    top_h = (h + 1) // 2
    area = random.randint(max(2, top_h), max(2, top_h * w - 1))
    must_touch = {(0, 0), (top_h - 1, 0), (0, w - 1)}
    half = _connected_random_mask(top_h, w, area, must_touch)
    if h % 2:
        top = half[:-1, :]
        middle = half[-1:, :]
        full = np.concatenate([top, middle, top[::-1, :]], axis=0)
    else:
        full = np.concatenate([half, half[::-1, :]], axis=0)
    if _symmetry_type(full) != "horizontal":
        return None
    return full


def _gen_vertical_mask() -> np.ndarray | None:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    left_w = (w + 1) // 2
    area = random.randint(max(2, left_w), max(2, h * left_w - 1))
    must_touch = {(0, 0), (h - 1, 0), (0, left_w - 1)}
    half = _connected_random_mask(h, left_w, area, must_touch)
    if w % 2:
        left = half[:, :-1]
        middle = half[:, -1:]
        full = np.concatenate([left, middle, left[:, ::-1]], axis=1)
    else:
        full = np.concatenate([half, half[:, ::-1]], axis=1)
    if _symmetry_type(full) != "vertical":
        return None
    return full


def _gen_mask(kind: str) -> np.ndarray:
    for _ in range(80):
        if kind == "both":
            mask = _gen_both_mask()
        elif kind == "horizontal":
            mask = _gen_horizontal_mask()
        else:
            mask = _gen_vertical_mask()
        if mask is not None and int(mask.sum()) >= 5:
            return mask
    raise RuntimeError(f"failed to generate {kind} mask")


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    return not occ[rr0:rr1, cc0:cc1].any()


def _place_patch(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray) -> bool:
    H, W = grid.shape
    h, w = patch.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _can_place(occ, r0, c0, h, w):
            continue
        grid[r0 : r0 + h, c0 : c0 + w][patch] = OBJ
        occ[r0 : r0 + h, c0 : c0 + w][patch] = True
        return True
    return False


def _render_legend(grid: np.ndarray, occ: np.ndarray) -> None:
    h, w = grid.shape
    cols = list(range(1, w - 1))
    random.shuffle(cols)
    for start in cols:
        pos = [start, start + 2, start + 4]
        if pos[-1] >= w - 1:
            continue
        ok = True
        for c in pos:
            if occ[h - 3 : h, c].any():
                ok = False
                break
        if not ok:
            continue
        for c, color in zip(pos, LEGEND_COLORS):
            grid[h - 3 : h, c] = color
            occ[h - 3 : h, c] = True
        return
    raise RuntimeError("failed to place legend")


def _expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 24)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    # Reserve the bottom legend band and one spacer row above it.
    occ[h - 4 :, :] = True
    occ[h - 4, :] = True
    occ[h - 3 :, :] = False

    base_kinds = ["horizontal", "vertical", "both"]
    extra_count = random.randint(0, 3)
    extras = [random.choice(base_kinds) for _ in range(extra_count)]
    kinds = base_kinds + extras
    random.shuffle(kinds)

    for kind in kinds:
        mask = _gen_mask(kind)
        if not _place_patch(grid[: h - 4, :], occ[: h - 4, :], mask):
            return None

    _render_legend(grid, occ)

    try:
        out = _expected_output(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
