from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS = ("top", "bottom", "left", "right")
MAX_WINDOW_ATTEMPTS = 160
MAX_PLACEMENT_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _contiguous_internal_gap(edge_mask: np.ndarray) -> tuple[int, int] | None:
    missing = np.flatnonzero(~edge_mask)
    if missing.size == 0:
        return None
    if missing[0] == 0 or missing[-1] == len(edge_mask) - 1:
        return None
    if np.any(np.diff(missing) != 1):
        return None
    return int(missing[0]), int(missing[-1])


def _find_windows(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    windows: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue

            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            rr, cc = np.nonzero(local)
            if np.any((rr != 0) & (rr != local.shape[0] - 1) & (cc != 0) & (cc != local.shape[1] - 1)):
                continue

            sides = {
                "top": local[0, :],
                "bottom": local[-1, :],
                "left": local[:, 0],
                "right": local[:, -1],
            }
            side_gaps: list[tuple[str, tuple[int, int]]] = []
            valid = True
            for side, edge in sides.items():
                gap = _contiguous_internal_gap(edge)
                if gap is not None:
                    side_gaps.append((side, gap))
                elif not np.all(edge):
                    valid = False
                    break
            if not valid or len(side_gaps) != 1:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            if np.any(interior == color):
                continue
            pattern_coords = np.argwhere(interior != BG)
            if pattern_coords.size == 0:
                continue
            pattern_coords = pattern_coords + np.array([r0 + 1, c0 + 1], dtype=int)
            pattern_colors = {int(g[r, c]) for r, c in pattern_coords}
            if len(pattern_colors) != 1:
                continue

            side, gap = side_gaps[0]
            windows.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "gap_side": side,
                    "gap_span": gap,
                    "pattern_coords": pattern_coords,
                }
            )

    windows.sort(key=lambda item: tuple(item["bbox"]))  # deterministic application order
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each window's interior pattern straight out through its border gap and backfill the vacated cells with the frame color.

    Input structure:
    - The background is Black.
    - Several one-pixel-thick hollow rectangular frames of different colors appear on the grid.
    - Each frame has exactly one contiguous gap on one side, and its interior contains one smaller non-Black pattern of a different color.

    Rule:
    - Detect every frame by its rectangular border and identify which side contains the unique contiguous gap.
    - Treat that gap as the push direction and translate the whole interior pattern along that axis until its leading edge crosses the gapped side by one cell.
    - Before drawing the translated pattern, repaint the pattern's original cells with the frame color so the border appears to expand inward.
    - Draw the translated pattern in the same color outside or across the gap, clipping any cells that leave the grid, while keeping all intact border pixels unchanged.

    Key insight:
    - The missing border segment is not damage but a direction marker: it tells which way the interior sprite moves, and the emptied cells are swallowed by the frame color rather than restored to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    h, w = g.shape
    windows = _find_windows(g)
    assert windows, "expected at least one gapped rectangular window"

    for window in windows:
        frame_color = int(window["color"])
        r0, c0, r1, c1 = window["bbox"]
        coords = np.asarray(window["pattern_coords"], dtype=int)
        pr0, pc0, pr1, pc1 = _bbox(coords)

        dy = dx = 0
        side = str(window["gap_side"])
        if side == "top":
            dy = r0 - pr0 - 1
        elif side == "bottom":
            dy = r1 - pr1 + 1
        elif side == "left":
            dx = c0 - pc0 - 1
        else:
            dx = c1 - pc1 + 1

        out[coords[:, 0], coords[:, 1]] = frame_color
        for r, c in coords:
            rr = int(r + dy)
            cc = int(c + dx)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(g[r, c])

    return out


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    target_lo = max(2, (height * width + 2) // 3)
    target_hi = height * width
    for _ in range(80):
        target = random.randint(target_lo, target_hi)
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any():
            return mask

    return np.ones((height, width), dtype=bool)


def _make_window_patch(frame_color: int, pattern_color: int, direction: str) -> np.ndarray | None:
    for _ in range(MAX_WINDOW_ATTEMPTS):
        frame_h = random.randint(6, 10)
        frame_w = random.randint(6, 10)
        max_ph = min(4, frame_h - 4)
        max_pw = min(4, frame_w - 4)
        if max_ph < 1 or max_pw < 1:
            continue

        if direction in ("top", "bottom"):
            if max_ph < 2:
                continue
            pattern_h = random.randint(2, max_ph)
            pattern_w = random.randint(1, max_pw)
        else:
            if max_pw < 2:
                continue
            pattern_h = random.randint(1, max_ph)
            pattern_w = random.randint(2, max_pw)

        mask = _random_connected_mask(pattern_h, pattern_w)

        extra_top = 1 if direction == "top" else 0
        extra_bottom = 1 if direction == "bottom" else 0
        extra_left = 1 if direction == "left" else 0
        extra_right = 1 if direction == "right" else 0

        patch = np.zeros(
            (frame_h + extra_top + extra_bottom, frame_w + extra_left + extra_right),
            dtype=int,
        )
        r0 = extra_top
        c0 = extra_left
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        _draw_frame(patch, (r0, c0, r1, c1), frame_color)

        pr_lo = r0 + 2
        pr_hi = r1 - pattern_h - 1
        pc_lo = c0 + 2
        pc_hi = c1 - pattern_w - 1
        if pr_lo > pr_hi or pc_lo > pc_hi:
            continue

        pr = random.randint(pr_lo, pr_hi)
        pc = random.randint(pc_lo, pc_hi)

        if direction in ("top", "bottom"):
            max_left_slack = pc - (c0 + 1)
            max_right_slack = (c1 - 1) - (pc + pattern_w - 1)
            gap_start = pc - random.randint(0, min(2, max_left_slack))
            gap_end = pc + pattern_w - 1 + random.randint(0, min(2, max_right_slack))
            gap_row = r0 if direction == "top" else r1
            patch[gap_row, gap_start : gap_end + 1] = BG
        else:
            max_top_slack = pr - (r0 + 1)
            max_bottom_slack = (r1 - 1) - (pr + pattern_h - 1)
            gap_start = pr - random.randint(0, min(2, max_top_slack))
            gap_end = pr + pattern_h - 1 + random.randint(0, min(2, max_bottom_slack))
            gap_col = c0 if direction == "left" else c1
            patch[gap_start : gap_end + 1, gap_col] = BG

        patch_slice = patch[pr : pr + pattern_h, pc : pc + pattern_w]
        patch_slice[mask] = pattern_color

        windows = _find_windows(patch)
        if len(windows) != 1:
            continue
        return patch

    return None


def _place_patches(height: int, width: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
        random.shuffle(candidates)

        placed = False
        for r, c in candidates:
            rr0 = max(0, r - 1)
            rr1 = min(height, r + ph + 1)
            cc0 = max(0, c - 1)
            cc1 = min(width, c + pw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue

            mask = patch != BG
            grid[r : r + ph, c : c + pw][mask] = patch[mask]
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_windows = random.randint(2, 4)
    directions = random.sample(list(DIRECTIONS), n_windows)
    colors = random.sample(list(PALETTE), 2 * n_windows)
    frame_colors = colors[:n_windows]
    pattern_colors = colors[n_windows:]

    patches: list[np.ndarray] = []
    for frame_color, pattern_color, direction in zip(frame_colors, pattern_colors, directions):
        patch = _make_window_patch(frame_color, pattern_color, direction)
        if patch is None:
            return None
        patches.append(patch)

    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    base = 16 + 2 * (n_windows - 2)
    h_lo = max(max_h + 2, base)
    w_lo = max(max_w + 2, base)
    if h_lo > 30 or w_lo > 30:
        return None

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        height = random.randint(h_lo, 30)
        width = random.randint(w_lo, 30)
        grid = _place_patches(height, width, patches)
        if grid is None:
            continue

        windows = _find_windows(grid)
        if len(windows) != n_windows:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-window ARC input")
