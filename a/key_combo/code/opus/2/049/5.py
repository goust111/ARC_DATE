from __future__ import annotations

import random
import numpy as np

BG = 0
RED = 2
BLUE = 1
YELLOW = 4
GREY = 5
GREEN = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _find_frame_center(g: np.ndarray) -> tuple[int, int]:
    H, W = g.shape
    candidates = []
    reds = np.argwhere(g == RED)
    for r, c in reds:
        if r == 0 or c == 0 or r == H - 1 or c == W - 1:
            continue
        window = g[r - 1 : r + 2, c - 1 : c + 2]
        if window.shape != (3, 3):
            continue
        if np.sum(window == YELLOW) == 8 and int(window[1, 1]) == RED:
            candidates.append((int(r), int(c)))
    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift a 3x3 yellow frame by a path-encoded displacement and draw concentric rings from seeds.

    Input structure:
    - A 3x3 yellow frame with a red center.
    - Another red pixel and one isolated yellow pixel connected through a gray path.
    - 2~4 blue seed pixels in black background.

    Rule:
    - Compute displacement vector from the second red to the isolated yellow.
    - Remove the gray path, second red, and isolated yellow.
    - Translate the frame by that vector, erasing the old frame.
    - Draw concentric square ring borders around each blue seed with side length determined by seed count S (distance 1..S).
    - If a ring pixel overlaps the translated frame border, paint it green; otherwise paint it blue.
    - Preserve the translated red frame center.

    Key insight:
    - Translation comes from the endpoint relation, but the full output shape is formed by combining transformed geometry and seed-indexed concentric growth.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    fr, fc = _find_frame_center(g)

    frame_border = {(fr - 1, fc - 1), (fr - 1, fc), (fr - 1, fc + 1),
                    (fr, fc - 1), (fr, fc + 1),
                    (fr + 1, fc - 1), (fr + 1, fc), (fr + 1, fc + 1)}
    reds = {(int(r), int(c)) for r, c in np.argwhere(g == RED)}
    assert len(reds) == 2
    second_red = next(iter(reds - {(fr, fc)}))
    ys = {(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)} - frame_border
    assert len(ys) == 1
    endpoint = next(iter(ys))

    dr = int(endpoint[0] - second_red[0])
    dc = int(endpoint[1] - second_red[1])

    seeds = [(int(r), int(c)) for r, c in np.argwhere(g == BLUE)]
    S = len(seeds)
    assert S >= 2

    out = np.full((H, W), BG, dtype=int)
    border_mask = np.zeros((H, W), dtype=bool)

    # Translate and draw frame.
    for r, c in frame_border | {(fr, fc)}:
        nr, nc = r + dr, c + dc
        if not _in_bounds((H, W), nr, nc):
            continue
        if (r, c) == (fr, fc):
            out[nr, nc] = RED
        else:
            out[nr, nc] = YELLOW
            border_mask[nr, nc] = True

    # Erase path markers by starting from out-black canvas.

    # Draw rings.
    for sr, sc in seeds:
        for d in range(1, S + 1):
            cells = []
            for rr in range(sr - d, sr + d + 1):
                if _in_bounds((H, W), rr, sc - d):
                    cells.append((rr, sc - d))
                if _in_bounds((H, W), rr, sc + d):
                    cells.append((rr, sc + d))
            for cc in range(sc - d + 1, sc + d):
                if _in_bounds((H, W), sr - d, cc):
                    cells.append((sr - d, cc))
                if _in_bounds((H, W), sr + d, cc):
                    cells.append((sr + d, cc))

            for r, c in cells:
                if border_mask[r, c]:
                    out[r, c] = GREEN
                else:
                    out[r, c] = BLUE

    # Ensure center always red if translation stayed in bounds.
    if 0 <= fr + dr < H and 0 <= fc + dc < W:
        out[fr + dr, fc + dc] = RED
    return out


def _path_points(start: tuple[int, int], end: tuple[int, int], orientation: str) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = r1 - r0
    dc = c1 - c0
    if orientation == "hv":
        seq: list[tuple[int, int]] = []
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
        for rr in range(r0, r1 + step_r, step_r) if step_r != 0 else [r0]:
            seq.append((rr, c0))
        for cc in range(c0 + step_c, c1 + step_c, step_c) if step_c != 0 else [c0]:
            seq.append((r1, cc))
        return seq
    raise ValueError("unsupported orientation")


def generate_input() -> np.ndarray:
    for _ in range(1200):
        H = random.randint(12, 16)
        W = random.randint(12, 16)
        g = np.full((H, W), BG, dtype=int)

        fr = random.randint(1, H - 4)
        fc = random.randint(1, W - 4)
        frame = [
            (fr - 1, fc - 1),
            (fr - 1, fc),
            (fr - 1, fc + 1),
            (fr, fc - 1),
            (fr, fc + 1),
            (fr + 1, fc - 1),
            (fr + 1, fc),
            (fr + 1, fc + 1),
        ]
        for r, c in frame:
            g[r, c] = YELLOW
        center = (fr, fc)
        g[center] = RED

        # choose displacement and second red
        attempts = 0
        placed = False
        while attempts < 200 and not placed:
            attempts += 1
            if H - fr - 3 <= 0 or fr < 3 or W - fc - 3 <= 0:
                fr = random.randint(3, H - 4)
                fc = random.randint(3, W - 4)

            dr = random.randint(-4, 4)
            dc = random.randint(-4, 4)
            if dr == 0 and dc == 0:
                continue
            sr = random.randint(1, H - 2)
            sc = random.randint(1, W - 2)
            if abs(sr - fr) < 2 and abs(sc - fc) < 2:
                continue
            er, ec = sr + dr, sc + dc
            if not (1 <= er < H - 1 and 1 <= ec < W - 1):
                continue
            if (sr, sc) == center or (er, ec) == center:
                continue

            # translated frame should stay inside
            if not (0 <= fr + dr < H and 0 <= fc + dc < W):
                continue
            if not (1 <= fr + dr <= H - 2 and 1 <= fc + dc <= W - 2):
                continue
            g[sr, sc] = RED
            g[er, ec] = YELLOW
            placed = True
            red2 = (sr, sc)

            # route path (L-shaped)
            steps = "hv"
            for r, c in _path_points(red2, (er, ec), steps):
                if (r, c) in {red2, (fr, fc), center, (er, ec)}:
                    continue
                if int(g[r, c]) in {RED, YELLOW}:
                    continue
                g[r, c] = GREY

        if not placed:
            continue

        # Blue seeds
        S = random.randint(2, 4)
        seed_candidates = [
            (r, c)
            for r in range(1, H - 1)
            for c in range(1, W - 1)
            if g[r, c] == BG
        ]
        random.shuffle(seed_candidates)
        if len(seed_candidates) < S:
            continue
        seeds = seed_candidates[:S]
        for r, c in seeds:
            g[r, c] = BLUE

        try:
            out = main(g)
            if int(np.sum(out == GREEN)) == 0:
                continue
            if out.shape != g.shape:
                continue
        except Exception:
            continue
        return g

    raise RuntimeError("failed to generate valid puzzle 5 input")
