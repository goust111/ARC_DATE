from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 5
MARKER = 7
FOREGROUND_COLORS = [1, 2, 3, 4, 6, 8, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[tuple[int, int]] = deque([(r, c)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(pts)
    return comps


def _find_frames(g: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    frame_mask = (g == FRAME) | (g == MARKER)
    frames: list[dict[str, int | tuple[int, int]]] = []
    for pts in _components(frame_mask):
        markers = [(r, c) for r, c in pts if int(g[r, c]) == MARKER]
        if len(markers) != 1:
            continue

        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_ring = r in (r0, r1) or c in (c0, c1)
                is_frame_cell = int(g[r, c]) in (FRAME, MARKER)
                if on_ring and not is_frame_cell:
                    ok = False
                if not on_ring and is_frame_cell:
                    ok = False
        if not ok:
            continue

        marker = markers[0]
        mr, mc = marker
        if not ((mr == r0 and c0 < mc < c1) or (mc == c0 and r0 < mr < r1)):
            continue

        frames.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "marker": marker})

    frames.sort(key=lambda f: (int(f["r0"]), int(f["c0"])))
    return frames


def _seed_patch(g: np.ndarray, frame: dict[str, int | tuple[int, int]]) -> np.ndarray:
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    pts = np.argwhere(interior != BG)
    assert pts.size, "each frame must contain a seed pattern"

    sr0, sc0 = map(int, pts.min(axis=0))
    sr1, sc1 = map(int, pts.max(axis=0))
    assert sr0 == 0 and sc0 == 0, "the seed must occupy the upper-left tile position"
    patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()

    inner_h, inner_w = interior.shape
    th, tw = patch.shape
    assert inner_h % th == 0 and inner_w % tw == 0, "seed size must divide the frame interior"
    return patch


def _turn_to_unfilled(
    filled: np.ndarray, r: int, c: int, dr: int, dc: int
) -> tuple[int, int]:
    if dc != 0:
        options = ((1, 0), (-1, 0))
    else:
        options = ((0, 1), (0, -1))

    valid: list[tuple[int, int]] = []
    rows, cols = filled.shape
    for rr, cc in options:
        nr = r + rr
        nc = c + cc
        if 0 <= nr < rows and 0 <= nc < cols and not filled[nr, nc]:
            valid.append((rr, cc))
    assert len(valid) == 1, "reflection must lead to a unique unfilled tile"
    return valid[0]


def _tile_with_reflections(
    out: np.ndarray, frame: dict[str, int | tuple[int, int]], patch: np.ndarray
) -> None:
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    mr, mc = frame["marker"]  # type: ignore[misc]
    mr = int(mr)
    mc = int(mc)

    ir0 = r0 + 1
    ic0 = c0 + 1
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    th, tw = patch.shape
    tile_rows = inner_h // th
    tile_cols = inner_w // tw

    if mr == r0:
        dr, dc = 0, 1
    else:
        dr, dc = 1, 0

    filled = np.zeros((tile_rows, tile_cols), dtype=bool)
    r = c = 0
    cur = patch.copy()
    total = int(tile_rows * tile_cols)
    done = 0
    while done < total:
        rr = ir0 + r * th
        cc = ic0 + c * tw
        out[rr : rr + th, cc : cc + tw] = cur
        if not filled[r, c]:
            filled[r, c] = True
            done += 1
        if done == total:
            break

        nr = r + dr
        nc = c + dc
        if not (0 <= nr < tile_rows and 0 <= nc < tile_cols) or filled[nr, nc]:
            if dc != 0:
                cur = np.fliplr(cur).copy()
            else:
                cur = np.flipud(cur).copy()
            dr, dc = _turn_to_unfilled(filled, r, c, dr, dc)
            nr = r + dr
            nc = c + dc

        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each framed seed pattern through its rectangle in a bouncing spiral.

    Input structure:
    - Black background contains one or more grey one-pixel rectangular closed frames.
    - Each frame has exactly one orange marker replacing a grey pixel on its top or left border.
    - The upper-left tile of each frame interior contains a small multi-color seed pattern whose bounding box evenly divides the interior height and width.

    Rule:
    - A top-border marker starts the seed moving right; a left-border marker starts it moving down.
    - Copy the seed by whole seed-bounding-box steps through the tiled interior.
    - Whenever the next step would hit the frame wall or an already filled tile, rotate to the unique perpendicular unfilled direction.
    - After a horizontal collision, horizontally flip the copied pattern; after a vertical collision, vertically flip it.

    Key insight:
    The seed bounding box defines the hidden tile grid, and the border marker supplies the first direction for the reflection-and-flip fill path.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one marked grey frame"
    for frame in frames:
        patch = _seed_patch(g, frame)
        _tile_with_reflections(out, frame, patch)
    return out


def _random_patch(rng: random.Random, h: int, w: int) -> np.ndarray:
    area = h * w
    n_colors = rng.randint(2, min(4, area, len(FOREGROUND_COLORS)))
    colors = rng.sample(FOREGROUND_COLORS, n_colors)
    for _ in range(100):
        vals = [colors[i] for i in range(n_colors)]
        vals.extend(rng.choice(colors) for _ in range(area - n_colors))
        rng.shuffle(vals)
        patch = np.array(vals, dtype=int).reshape(h, w)
        if not np.array_equal(patch, np.fliplr(patch)) and not np.array_equal(patch, np.flipud(patch)):
            return patch

    vals = [colors[(r * 2 + c) % n_colors] for r in range(h) for c in range(w)]
    patch = np.array(vals, dtype=int).reshape(h, w)
    if np.array_equal(patch, np.fliplr(patch)):
        patch[0, 0], patch[0, -1] = patch[0, -1], patch[0, 0]
    if np.array_equal(patch, np.flipud(patch)):
        patch[0, 0], patch[-1, 0] = patch[-1, 0], patch[0, 0]
    return patch


def _sample_frame_spec(rng: random.Random, max_h: int, max_w: int) -> dict[str, int | str | np.ndarray] | None:
    for _ in range(80):
        tile_h = rng.randint(2, 4)
        tile_w = rng.randint(2, 4)
        tile_rows = rng.randint(2, 5)
        tile_cols = rng.randint(2, 5)
        frame_h = tile_h * tile_rows + 2
        frame_w = tile_w * tile_cols + 2
        if frame_h > max_h or frame_w > max_w:
            continue
        side = "top" if rng.random() < 0.5 else "left"
        return {
            "tile_h": tile_h,
            "tile_w": tile_w,
            "tile_rows": tile_rows,
            "tile_cols": tile_cols,
            "frame_h": frame_h,
            "frame_w": frame_w,
            "side": side,
            "patch": _random_patch(rng, tile_h, tile_w),
        }
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rows, cols = occupied.shape
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(rows, r0 + h + 1)
    pc1 = min(cols, c0 + w + 1)
    return not bool(occupied[pr0:pr1, pc0:pc1].any())


def _draw_frame(
    g: np.ndarray, r0: int, c0: int, spec: dict[str, int | str | np.ndarray], rng: random.Random
) -> None:
    frame_h = int(spec["frame_h"])
    frame_w = int(spec["frame_w"])
    tile_h = int(spec["tile_h"])
    tile_w = int(spec["tile_w"])
    patch = np.asarray(spec["patch"], dtype=int)

    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    g[r0, c0 : c1 + 1] = FRAME
    g[r1, c0 : c1 + 1] = FRAME
    g[r0 : r1 + 1, c0] = FRAME
    g[r0 : r1 + 1, c1] = FRAME

    if str(spec["side"]) == "top":
        mc = c0 + rng.choice((1, tile_w))
        g[r0, mc] = MARKER
    else:
        mr = r0 + rng.choice((1, tile_h))
        g[mr, c0] = MARKER

    g[r0 + 1 : r0 + 1 + tile_h, c0 + 1 : c0 + 1 + tile_w] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(12, 30)
    cols = rng.randint(12, 30)
    n_frames = rng.choices([1, 2, 3], weights=[2, 4, 3], k=1)[0]
    g = np.zeros((rows, cols), dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    for _ in range(n_frames):
        spec = _sample_frame_spec(rng, rows - 2, cols - 2)
        if spec is None:
            return None
        fh = int(spec["frame_h"])
        fw = int(spec["frame_w"])

        placed = False
        for _try in range(120):
            if rows - fh <= 1 or cols - fw <= 1:
                break
            r0 = rng.randrange(1, rows - fh)
            c0 = rng.randrange(1, cols - fw)
            if not _can_place(occupied, r0, c0, fh, fw):
                continue
            _draw_frame(g, r0, c0, spec, rng)
            occupied[max(0, r0 - 1) : min(rows, r0 + fh + 1), max(0, c0 - 1) : min(cols, c0 + fw + 1)] = True
            placed = True
            break
        if not placed:
            return None

    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marked reflection tiling input")
