from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
FRAME_COLORS = (1, 2, 3, 4, 6, 8, 9)
GEN_RETRIES = 400
ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _rotation_signatures(mask: np.ndarray) -> tuple[tuple[tuple[int, int], ...], ...]:
    tight = _tight_mask(mask)
    return tuple(_shape_signature(np.rot90(tight, k)) for k in range(4))


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_rotation_signatures(mask))


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _is_rect_ring(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    return set(points) == border


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY, ORANGE):
            continue
        for pts in _components(grid == color):
            if not _is_rect_ring(pts):
                continue
            r0, c0, r1, c1 = _bbox(pts)
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            assert interior.size > 0
            assert np.all((interior == BG) | (interior == GREY))
            grey_mask = interior == GREY
            comp_mask = interior == BG
            assert grey_mask.any() and comp_mask.any()
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "comp_mask": comp_mask.copy(),
                    "sig": _canonical_signature(comp_mask),
                }
            )
    assert frames, "expected at least one rectangular frame"
    inner_sizes = {(frame["comp_mask"].shape[0], frame["comp_mask"].shape[1]) for frame in frames}
    assert len(inner_sizes) == 1, "all frame interiors must share the same size"
    sigs = [frame["sig"] for frame in frames]
    assert len(set(sigs)) == len(sigs), "frame complement shapes must be unique up to rotation"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each orange cluster into the frame whose grey key leaves that cluster's
    complementary shape inside the frame, allowing rotation but not reflection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)

    orange_comps = _components(g == ORANGE)
    assert len(orange_comps) == len(frames), "orange cluster count must match frame count"

    clusters_by_sig: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for pts in orange_comps:
        tight = np.zeros((_bbox(pts)[2] - _bbox(pts)[0] + 1, _bbox(pts)[3] - _bbox(pts)[1] + 1), dtype=bool)
        r0, c0, _, _ = _bbox(pts)
        for r, c in pts:
            tight[r - r0, c - c0] = True
        sig = _canonical_signature(tight)
        assert sig not in clusters_by_sig, "orange cluster shapes must be unique up to rotation"
        clusters_by_sig[sig] = pts

    matched_cells: set[tuple[int, int]] = set()
    for frame in frames:
        sig = frame["sig"]
        assert sig in clusters_by_sig, "missing orange cluster for frame complement"
        matched_cells.update(clusters_by_sig[sig])

        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frame["comp_mask"]] = int(frame["color"])

    for r, c in matched_cells:
        out[r, c] = BG

    assert len(matched_cells) == int((g == ORANGE).sum())
    return out


def _grow_connected_mask(h: int, w: int, area: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    mask[start] = True

    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        seen_frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in ORTHO_DIRS:
                nr, nc = int(r + dr), int(c + dc)
                cell = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and cell not in seen_frontier:
                    frontier.append(cell)
                    seen_frontier.add(cell)
        if not frontier:
            return None
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True
    return mask


def _sample_complement_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray | None:
    total = h * w
    low = max(3, total // 4)
    high = min(total - 3, (3 * total) // 4)
    if low > high:
        return None
    for _ in range(300):
        area = int(rng.integers(low, high + 1))
        mask = _grow_connected_mask(h, w, area, rng)
        if mask is None:
            continue
        if not _is_connected(mask):
            continue
        if not _is_connected(~mask):
            continue
        if len(set(_rotation_signatures(mask))) == 1:
            continue
        return mask
    return None


def _can_place(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occ.shape
    if top < 1 or left < 1 or top + h >= H or left + w >= W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not occ[r0:r1, c0:c1].any()


def _place_box(occ: np.ndarray, h: int, w: int, rng: np.random.Generator) -> tuple[int, int] | None:
    H, W = occ.shape
    if H - h - 1 < 1 or W - w - 1 < 1:
        return None
    for _ in range(300):
        top = int(rng.integers(1, H - h))
        left = int(rng.integers(1, W - w))
        if not _can_place(occ, top, left, h, w, pad=1):
            continue
        occ[top : top + h, left : left + w] = True
        return top, left
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, color: int, grey_mask: np.ndarray) -> None:
    ih, iw = grey_mask.shape
    oh, ow = ih + 2, iw + 2
    grid[top, left : left + ow] = color
    grid[top + oh - 1, left : left + ow] = color
    grid[top : top + oh, left] = color
    grid[top : top + oh, left + ow - 1] = color
    interior = grid[top + 1 : top + 1 + ih, left + 1 : left + 1 + iw]
    interior[grey_mask] = GREY


def _expected_output(
    grid: np.ndarray,
    frame_positions: list[tuple[int, int]],
    frame_colors: list[int],
    complement_masks: list[np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    out[out == ORANGE] = BG
    for (top, left), color, comp_mask in zip(frame_positions, frame_colors, complement_masks):
        interior = out[top + 1 : top + 1 + comp_mask.shape[0], left + 1 : left + 1 + comp_mask.shape[1]]
        interior[comp_mask] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    ih, iw = ((4, 4), (4, 5), (5, 4), (5, 5))[int(rng.integers(4))]
    num_frames = int(rng.integers(2, 5))
    frame_colors = list(map(int, rng.choice(FRAME_COLORS, size=num_frames, replace=False)))

    complement_masks: list[np.ndarray] = []
    used_sigs: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(num_frames):
        chosen = None
        for _ in range(200):
            comp_mask = _sample_complement_mask(ih, iw, rng)
            if comp_mask is None:
                continue
            sig = _canonical_signature(comp_mask)
            if sig in used_sigs:
                continue
            used_sigs.add(sig)
            chosen = comp_mask
            break
        if chosen is None:
            return None
        complement_masks.append(chosen)

    cluster_masks: list[np.ndarray] = []
    for comp_mask in complement_masks:
        tight = _tight_mask(comp_mask)
        k = int(rng.integers(4))
        cluster_masks.append(np.rot90(tight, k))

    outer_h, outer_w = ih + 2, iw + 2
    padded_area = num_frames * (outer_h + 2) * (outer_w + 2)
    padded_area += sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in cluster_masks)
    min_side = max(18, int(np.ceil(np.sqrt(padded_area))) + 3)
    if min_side > 30:
        return None
    max_side = min(30, min_side + 4)
    H = int(rng.integers(min_side, max_side + 1))
    W = int(rng.integers(min_side, max_side + 1))

    occ = np.zeros((H, W), dtype=bool)
    frame_positions: list[tuple[int, int]] = []
    for _ in range(num_frames):
        pos = _place_box(occ, outer_h, outer_w, rng)
        if pos is None:
            return None
        frame_positions.append(pos)

    cluster_positions: list[tuple[int, int] | None] = [None] * num_frames
    order = sorted(range(num_frames), key=lambda i: cluster_masks[i].shape[0] * cluster_masks[i].shape[1], reverse=True)
    for idx in order:
        ch, cw = cluster_masks[idx].shape
        pos = _place_box(occ, ch, cw, rng)
        if pos is None:
            return None
        cluster_positions[idx] = pos

    grid = np.full((H, W), BG, dtype=int)
    for pos, color, comp_mask in zip(frame_positions, frame_colors, complement_masks):
        grey_mask = ~comp_mask
        _draw_frame(grid, pos[0], pos[1], color, grey_mask)
    for pos, mask in zip(cluster_positions, cluster_masks):
        assert pos is not None
        top, left = pos
        sub = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
        sub[mask] = ORANGE

    expected = _expected_output(grid, frame_positions, frame_colors, complement_masks)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-and-lock frame puzzle")
