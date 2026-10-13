from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


QUADRANTS = ("tl", "tr", "bl", "br")
BRANCH_CASES = ("diag", "horiz", "vert")
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.ravel(grid))).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_frames(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g) if bg is None else int(bg)

    visited = np.zeros((h, w), dtype=bool)
    frames: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((int(nr), int(nc)))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            side = int(r1 - r0 + 1)
            if side != int(c1 - c0 + 1) or side < 4:
                continue

            perimeter = set()
            for cc in range(c0, c1 + 1):
                perimeter.add((int(r0), int(cc)))
                perimeter.add((int(r1), int(cc)))
            for rr in range(r0 + 1, r1):
                perimeter.add((int(rr), int(c0)))
                perimeter.add((int(rr), int(c1)))

            cell_set = set(cells)
            if cell_set != perimeter:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_pts = np.argwhere(interior != bg)
            if int(seed_pts.shape[0]) != 1:
                continue

            sr = int(seed_pts[0, 0] + r0 + 1)
            sc = int(seed_pts[0, 1] + c0 + 1)
            seed_color = int(g[sr, sc])
            if seed_color == color:
                continue

            frames.append(
                {
                    "frame_color": int(color),
                    "seed_color": int(seed_color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "seed": (int(sr), int(sc)),
                }
            )

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    return frames


def _grow_branch(out: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    sr, sc = map(int, frame["seed"])
    seed_color = int(frame["seed_color"])
    frame_color = int(frame["frame_color"])

    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    if sr < center_r:
        row_step = 1
        dr = int(sr - r0)
    elif sr > center_r:
        row_step = -1
        dr = int(r1 - sr)
    else:
        raise AssertionError("seed cannot lie on the frame's horizontal center line")

    if sc < center_c:
        col_step = 1
        dc = int(sc - c0)
    elif sc > center_c:
        col_step = -1
        dc = int(c1 - sc)
    else:
        raise AssertionError("seed cannot lie on the frame's vertical center line")

    stem = min(dr, dc)

    def paint(rr: int, cc: int) -> None:
        if out[rr, cc] != frame_color:
            out[rr, cc] = seed_color

    for step in range(stem + 1):
        paint(int(sr + row_step * step), int(sc + col_step * step))

    end_r = int(sr + row_step * stem)
    end_c = int(sc + col_step * stem)

    for step in range(1, int(dc - stem) + 1):
        paint(int(end_r), int(end_c + col_step * step))

    for step in range(1, int(dr - stem) + 1):
        paint(int(end_r + row_step * step), int(end_c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored diagonal-and-L branch inside each hollow square frame.

    Input structure:
    - bg is the most common color.
    - The grid contains several 1-pixel-thick monochrome square frames on bg.
    - Each frame encloses exactly one interior seed pixel whose color differs from both bg and the frame.

    Rule:
    - Detect the quadrant of the seed relative to its frame center; that quadrant fixes the inward diagonal growth direction toward the center.
    - Measure dr and dc as the seed's distances to the quadrant's nearest horizontal and vertical frame edges.
    - Draw the seed color from the seed along the inward diagonal for min(dr, dc) steps, then continue the remaining horizontal and vertical distances as two perpendicular branches from the diagonal endpoint.
    - Keep the original frames unchanged and leave the grid size unchanged.

    Key insight:
    - The seed's quadrant determines the diagonal direction, while the two edge distances determine whether the result is a pure diagonal or an L-shaped branch with one or two leftover arms.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one framed seed object"

    out = g.copy()
    for frame in frames:
        _grow_branch(out, frame)
    return out


def _choose_distances(limit: int, branch_case: str, rng: random.Random) -> tuple[int, int]:
    if branch_case == "diag":
        d = rng.randint(1, limit)
        return int(d), int(d)
    if branch_case == "horiz":
        dr = rng.randint(1, limit - 1)
        dc = rng.randint(dr + 1, limit)
        return int(dr), int(dc)
    if branch_case == "vert":
        dc = rng.randint(1, limit - 1)
        dr = rng.randint(dc + 1, limit)
        return int(dr), int(dc)
    raise ValueError(f"unknown branch case: {branch_case}")


def _seed_from_quadrant(
    bbox: tuple[int, int, int, int], quadrant: str, dr: int, dc: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    if quadrant == "tl":
        return int(r0 + dr), int(c0 + dc)
    if quadrant == "tr":
        return int(r0 + dr), int(c1 - dc)
    if quadrant == "bl":
        return int(r1 - dr), int(c0 + dc)
    if quadrant == "br":
        return int(r1 - dr), int(c1 - dc)
    raise ValueError(f"unknown quadrant: {quadrant}")


def _place_specs(specs: list[dict], h: int, w: int, rng: random.Random) -> bool:
    occupied = np.zeros((h, w), dtype=bool)

    for spec in sorted(specs, key=lambda item: int(item["side"]), reverse=True):
        side = int(spec["side"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - side):
            for c0 in range(1, w - side):
                rr0 = max(0, int(r0 - 1))
                rr1 = min(h, int(r0 + side + 1))
                cc0 = max(0, int(c0 - 1))
                cc1 = min(w, int(c0 + side + 1))
                if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                    candidates.append((int(r0), int(c0)))

        if not candidates:
            return False

        r0, c0 = rng.choice(candidates)
        spec["bbox"] = (int(r0), int(c0), int(r0 + side - 1), int(c0 + side - 1))
        occupied[r0 : r0 + side, c0 : c0 + side] = True

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randint(0, 9)
    palette = [color for color in range(10) if color != bg]

    num_frames = rng.choice((3, 4))
    colors = rng.sample(palette, num_frames * 2)
    frame_colors = colors[:num_frames]
    seed_colors = colors[num_frames:]

    sides = rng.sample(range(6, 13), num_frames)
    max_side = max(sides)
    min_dim = max(18, max_side * 2 + 2)
    if min_dim > 30:
        return None

    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)

    quadrants = rng.sample(list(QUADRANTS), num_frames)
    branch_cases = list(BRANCH_CASES)
    if num_frames == 4:
        branch_cases.append(rng.choice(BRANCH_CASES))
    rng.shuffle(branch_cases)

    specs: list[dict] = []
    for idx in range(num_frames):
        side = int(sides[idx])
        limit = (side - 2) // 2
        branch_case = str(branch_cases[idx])
        dr, dc = _choose_distances(limit, branch_case, rng)
        specs.append(
            {
                "side": int(side),
                "frame_color": int(frame_colors[idx]),
                "seed_color": int(seed_colors[idx]),
                "quadrant": str(quadrants[idx]),
                "branch_case": str(branch_case),
                "dr": int(dr),
                "dc": int(dc),
            }
        )

    if not _place_specs(specs, h, w, rng):
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        frame_color = int(spec["frame_color"])

        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        sr, sc = _seed_from_quadrant(spec["bbox"], str(spec["quadrant"]), int(spec["dr"]), int(spec["dc"]))
        spec["seed"] = (int(sr), int(sc))
        grid[sr, sc] = int(spec["seed_color"])

    counts = Counter(map(int, np.ravel(grid)))
    if counts[int(bg)] <= max(count for color, count in counts.items() if int(color) != int(bg)):
        return None

    detected = _find_frames(grid, bg)
    if len(detected) != num_frames:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid framed-branch input")
