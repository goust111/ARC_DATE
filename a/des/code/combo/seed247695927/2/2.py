import random
from collections import Counter

import numpy as np


BG = 7  # orange
MAX_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_info(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    if coords.shape[0] < 8:
        return None
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return None
    color = int(grid[r0, c0])
    expected = 2 * (h + w) - 4
    if coords.shape[0] != expected:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == color):
        return None
    if np.any(patch[~border] == color):
        return None
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "height": h,
        "width": w,
        "shorter": min(h, w),
        "longer": max(h, w),
        "center": ((r0 + r1) // 2, (c0 + c1) // 2),
    }


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    non_bg_colors = [int(x) for x in np.unique(g) if int(x) != bg]

    frames = []
    marker_candidates = []
    for color in non_bg_colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 1
        comp = comps[0]
        if comp.shape[0] == 1:
            marker_candidates.append((color, int(comp[0, 0]), int(comp[0, 1])))
            continue
        info = _frame_info(g, comp)
        assert info is not None
        frames.append(info)

    assert len(frames) >= 1
    short_sizes = [frame["shorter"] for frame in frames]
    assert len(set(short_sizes)) == len(short_sizes)
    min_short = min(short_sizes)
    smallest = [frame for frame in frames if frame["shorter"] == min_short]
    assert len(smallest) == 1
    smallest = smallest[0]

    sr, sc = smallest["center"]
    marker_matches = [(color, r, c) for color, r, c in marker_candidates if (r, c) == (sr, sc)]
    assert len(marker_matches) == 1
    marker_color = int(marker_matches[0][0])
    assert marker_color != smallest["color"]

    side = max(frame["longer"] for frame in frames)
    out = np.full((side, side), bg, dtype=int)
    ordered = sorted(frames, key=lambda frame: frame["shorter"], reverse=True)
    for frame in ordered:
        h = frame["height"]
        w = frame["width"]
        r0 = (side - h) // 2
        c0 = (side - w) // 2
        _draw_frame(out, r0, c0, h, w, frame["color"])

    out[side // 2, side // 2] = marker_color
    return out


def _sample_frame_sizes(rng: random.Random, n_frames: int) -> list[tuple[int, int]]:
    sizes = []
    used_short = set()
    for _ in range(n_frames):
        for _ in range(100):
            h = rng.randrange(3, 14, 2)
            w = rng.randrange(3, 14, 2)
            shorter = min(h, w)
            if shorter in used_short:
                continue
            sizes.append((h, w))
            used_short.add(shorter)
            break
        else:
            return []
    return sizes


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h0, w0 = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > h0 or c0 + w > w0:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h0, r0 + h + 1)
    cc1 = min(w0, c0 + w + 1)
    return not occ[rr0:rr1, cc0:cc1].any()


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    n_frames = rng.randint(2, 5)
    frame_sizes = _sample_frame_sizes(rng, n_frames)
    if len(frame_sizes) != n_frames:
        return None

    palette = [c for c in range(10) if c != BG]
    rng.shuffle(palette)
    frame_colors = palette[:n_frames]
    marker_color = palette[n_frames]

    placed = []
    for (fh, fw), color in zip(frame_sizes, frame_colors):
        for _ in range(200):
            r0 = rng.randint(0, h - fh)
            c0 = rng.randint(0, w - fw)
            if not _can_place(occ, r0, c0, fh, fw):
                continue
            _draw_frame(grid, r0, c0, fh, fw, color)
            _mark_occ(occ, r0, c0, fh, fw)
            placed.append((fh, fw, color, r0, c0))
            break
        else:
            return None

    smallest = min(placed, key=lambda item: min(item[0], item[1]))
    fh, fw, _, r0, c0 = smallest
    mr = r0 + fh // 2
    mc = c0 + fw // 2
    if grid[mr, mc] != BG:
        return None
    grid[mr, mc] = marker_color

    counts = Counter(map(int, grid.flat))
    if counts[BG] <= max(count for color, count in counts.items() if color != BG):
        return None

    frame_shorts = [min(item[0], item[1]) for item in placed]
    if len(set(frame_shorts)) != len(frame_shorts):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
