from __future__ import annotations

import numpy as np

BG = 0
YELLOW = 4
FRAME_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _side_gap(side: np.ndarray) -> tuple[int, int] | None:
    if side.size < 4 or (not side[0]) or (not side[-1]):
        return None
    interior = side[1:-1]
    missing = np.flatnonzero(~interior)
    if missing.size == 0:
        return None
    if int(missing[-1] - missing[0] + 1) != int(missing.size):
        return None
    if not interior[: missing[0]].all():
        return None
    if not interior[missing[-1] + 1 :].all():
        return None
    start = int(missing[0] + 1)
    end = int(missing[-1] + 1)
    if end - start + 1 < 2:
        return None
    return start, end


def _detect_windows(grid: np.ndarray) -> list[dict]:
    windows: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, YELLOW)):
        coords_arr = np.argwhere(grid == int(color))
        if coords_arr.size == 0:
            continue
        coords = [(int(r), int(c)) for r, c in coords_arr]
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in coords):
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)
        if np.any(sub[1:-1, 1:-1]):
            continue

        top_gap = _side_gap(sub[0, :])
        bottom_gap = _side_gap(sub[-1, :])
        left_gap = _side_gap(sub[:, 0])
        right_gap = _side_gap(sub[:, -1])
        gaps = {
            "up": top_gap,
            "down": bottom_gap,
            "left": left_gap,
            "right": right_gap,
        }
        gap_sides = [name for name, gap in gaps.items() if gap is not None]
        if len(gap_sides) != 1:
            continue

        full_sides = {
            "up": bool(sub[0, :].all()),
            "down": bool(sub[-1, :].all()),
            "left": bool(sub[:, 0].all()),
            "right": bool(sub[:, -1].all()),
        }
        gap_side = gap_sides[0]
        if any(name != gap_side and not ok for name, ok in full_sides.items()):
            continue

        windows.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "direction": gap_side,
                "gap": gaps[gap_side],
            }
        )
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each window-frame gap as an outward direction and stamp colored exterior shadows of the yellow interior motifs.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular window frames appear in distinct colors.
    - Each frame has exactly one continuous gap of width at least 2 on one side; the gap side encodes the window's orientation.
    - Inside each frame are one or more disconnected yellow motifs that do not touch the frame.

    Rule:
    - For every colored frame, find which side contains the unique gap; that side gives the projection direction.
    - For each yellow connected component inside that frame, translate the whole component outward along that direction.
    - The translation magnitude is the component's distance to the gap side plus 1, so its near edge lands just outside the opening.
    - Draw the translated copy in the frame color, while keeping the original yellow motifs, the frame, and the gap unchanged.

    Key insight:
    - The missing frame segment is not damage; it is an exit cue that simultaneously chooses the axis of motion and the place where the colored shadow appears.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for window in _detect_windows(g):
        r0, c0, r1, c1 = window["bbox"]
        direction = window["direction"]
        gap0, gap1 = window["gap"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        for comp in _components(interior == YELLOW):
            abs_comp = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in comp]
            cr0, cc0, cr1, cc1 = _bbox(abs_comp)

            if direction in ("up", "down"):
                gap_c0 = c0 + gap0
                gap_c1 = c0 + gap1
                if cc0 < gap_c0 or cc1 > gap_c1:
                    continue
                dy = -(cr0 - r0 + 1) if direction == "up" else (r1 - cr1 + 1)
                dx = 0
            else:
                gap_r0 = r0 + gap0
                gap_r1 = r0 + gap1
                if cr0 < gap_r0 or cr1 > gap_r1:
                    continue
                dy = 0
                dx = -(cc0 - c0 + 1) if direction == "left" else (c1 - cc1 + 1)

            for rr, cc in abs_comp:
                nr, nc = rr + dy, cc + dx
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(window["color"])

    return out


def _scatter(rng: np.random.Generator, total: int, count: int) -> list[int]:
    parts = [0] * count
    for _ in range(int(total)):
        parts[int(rng.integers(0, count))] += 1
    return parts


def _sample_intervals(
    rng: np.random.Generator,
    usable_len: int,
    count: int,
    max_len: int,
) -> list[tuple[int, int]] | None:
    if count <= 0:
        return []
    min_required = int(count) + int(count - 1)
    if min_required > usable_len:
        return None
    for _ in range(100):
        lengths = [int(rng.integers(1, max_len + 1)) for _ in range(count)]
        required = sum(lengths) + count - 1
        if required > usable_len:
            continue
        extras = _scatter(rng, usable_len - required, count + 1)
        intervals: list[tuple[int, int]] = []
        pos = extras[0]
        for idx, length in enumerate(lengths):
            start = pos
            end = start + length - 1
            intervals.append((int(start), int(end)))
            pos = end + 1
            if idx < count - 1:
                pos += 1 + extras[idx + 1]
        return intervals
    return None


def _make_window_patch(
    rng: np.random.Generator,
    direction: str,
    color: int,
) -> np.ndarray | None:
    for _ in range(100):
        if direction in ("up", "down"):
            frame_h = int(rng.integers(6, 11))
            frame_w = int(rng.integers(7, 13))
            safe_rows = list(range(2, frame_h - 2))
            usable_cols = frame_w - 4
            max_count = min(3, len(safe_rows), (usable_cols + 1) // 2)
            if max_count <= 0:
                continue
            count = int(rng.integers(1, max_count + 1))
            intervals = _sample_intervals(rng, usable_cols, count, max_len=min(4, usable_cols))
            if intervals is None:
                continue
            rows = [int(x) for x in rng.choice(np.array(safe_rows), size=count, replace=False)]
            segs = [(row, 2 + a, 2 + b) for row, (a, b) in zip(rows, intervals)]
            min_c = min(a for _, a, _ in segs)
            max_c = max(b for _, _, b in segs)
            gap0 = int(rng.integers(1, min_c + 1))
            gap1_lo = max(max_c, gap0 + 1)
            if gap1_lo > frame_w - 2:
                continue
            gap1 = int(rng.integers(gap1_lo, frame_w - 1))

            patch = np.full((frame_h, frame_w), BG, dtype=int)
            patch[0, :] = color
            patch[-1, :] = color
            patch[:, 0] = color
            patch[:, -1] = color
            if direction == "up":
                patch[0, gap0 : gap1 + 1] = BG
            else:
                patch[-1, gap0 : gap1 + 1] = BG
            for row, a, b in segs:
                patch[row, a : b + 1] = YELLOW
            return patch

        frame_h = int(rng.integers(7, 13))
        frame_w = int(rng.integers(6, 11))
        safe_cols = list(range(2, frame_w - 2))
        usable_rows = frame_h - 4
        max_count = min(3, len(safe_cols), (usable_rows + 1) // 2)
        if max_count <= 0:
            continue
        count = int(rng.integers(1, max_count + 1))
        intervals = _sample_intervals(rng, usable_rows, count, max_len=min(4, usable_rows))
        if intervals is None:
            continue
        cols = [int(x) for x in rng.choice(np.array(safe_cols), size=count, replace=False)]
        segs = [(2 + a, 2 + b, col) for col, (a, b) in zip(cols, intervals)]
        min_r = min(a for a, _, _ in segs)
        max_r = max(b for _, b, _ in segs)
        gap0 = int(rng.integers(1, min_r + 1))
        gap1_lo = max(max_r, gap0 + 1)
        if gap1_lo > frame_h - 2:
            continue
        gap1 = int(rng.integers(gap1_lo, frame_h - 1))

        patch = np.full((frame_h, frame_w), BG, dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        if direction == "left":
            patch[gap0 : gap1 + 1, 0] = BG
        else:
            patch[gap0 : gap1 + 1, -1] = BG
        for a, b, col in segs:
            patch[a : b + 1, col] = YELLOW
        return patch
    return None


def _reserve_box(
    frame_box: tuple[int, int, int, int],
    direction: str,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = frame_box
    if direction == "up":
        r0 -= 1
    elif direction == "down":
        r1 += 1
    elif direction == "left":
        c0 -= 1
    else:
        c1 += 1
    return max(0, r0), max(0, c0), min(h - 1, r1), min(w - 1, c1)


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _place_patch(
    rng: np.random.Generator,
    grid: np.ndarray,
    patch: np.ndarray,
    direction: str,
    reserved: list[tuple[int, int, int, int]],
) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape

    r_min = 1 if direction == "up" else 0
    r_max = gh - ph - (1 if direction == "down" else 0)
    c_min = 1 if direction == "left" else 0
    c_max = gw - pw - (1 if direction == "right" else 0)
    if r_min > r_max or c_min > c_max:
        return False

    positions = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
    order = rng.permutation(len(positions))
    for idx in order:
        r0, c0 = positions[int(idx)]
        frame_box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        reserve = _reserve_box(frame_box, direction, grid.shape)
        if any(_boxes_overlap(reserve, old, pad=1) for old in reserved):
            continue
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((patch != BG) & (region != BG)):
            continue
        region[patch != BG] = patch[patch != BG]
        reserved.append(reserve)
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_windows = int(rng.integers(2, 5))
    grid_h = int(rng.integers(24, 31))
    grid_w = int(rng.integers(24, 31))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    directions = list(rng.choice(np.array(DIRS), size=n_windows, replace=False))
    colors = [int(x) for x in rng.choice(np.array(FRAME_COLORS), size=n_windows, replace=False)]
    order = [int(x) for x in rng.permutation(n_windows)]

    reserved: list[tuple[int, int, int, int]] = []
    for idx in order:
        patch = _make_window_patch(rng, directions[idx], colors[idx])
        if patch is None:
            return None
        if not _place_patch(rng, grid, patch, directions[idx], reserved):
            return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
