import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BG = 0
TRACK = 5
GEAR = 3
TOKEN_COLORS = [1, 2, 4, 6, 7, 8, 9]

MAX_ARM = 3
MIN_TRACKS = 2
MAX_TRACKS = 4
BASE_CANVAS = 30
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_clockwise(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _find_gear_center(points: list[tuple[int, int]]) -> tuple[int, int]:
    pts = list(points)
    for r, c in pts:
        if not all(pr == r or pc == c for pr, pc in pts):
            continue
        row_cols = sorted(pc for pr, pc in pts if pr == r)
        col_rows = sorted(pr for pr, pc in pts if pc == c)
        if row_cols != list(range(min(row_cols), max(row_cols) + 1)):
            continue
        if col_rows != list(range(min(col_rows), max(col_rows) + 1)):
            continue
        if len(pts) == len(row_cols) + len(col_rows) - 1:
            return r, c
    raise AssertionError("failed to locate gear center")


def _gear_lengths(points: list[tuple[int, int]], center: tuple[int, int]) -> tuple[int, int, int, int]:
    gear = set(points)
    cr, cc = center

    def arm_len(dr: int, dc: int) -> int:
        steps = 0
        r, c = cr + dr, cc + dc
        while (r, c) in gear:
            steps += 1
            r += dr
            c += dc
        return steps

    up = arm_len(-1, 0)
    down = arm_len(1, 0)
    left = arm_len(0, -1)
    right = arm_len(0, 1)
    assert len(gear) == 1 + up + down + left + right
    return up, down, left, right


def _extract_tracks(grid: np.ndarray, bg: int) -> list[dict]:
    tracks: list[dict] = []
    for pts in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = _border_clockwise(r0, c0, r1, c1)
        if len(pts) != len(border):
            continue
        if set(pts) != set(border):
            continue

        border_colors = [int(grid[r, c]) for r, c in border]
        track_color, _count = Counter(border_colors).most_common(1)[0]
        tokens = [(r, c, int(grid[r, c])) for r, c in border if int(grid[r, c]) != track_color]
        if not (1 <= len(tokens) <= 2):
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        gear_comps = _components(interior != bg)
        if len(gear_comps) != 1:
            continue
        gear_pts = [(r0 + 1 + r, c0 + 1 + c) for r, c in gear_comps[0]]
        expected_center = ((r0 + r1) // 2, (c0 + c1) // 2)
        center = expected_center if expected_center in set(gear_pts) else _find_gear_center(gear_pts)
        up, down, left, right = _gear_lengths(gear_pts, center)
        if max(up, down, left, right) > MAX_ARM:
            continue

        tracks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border": border,
                "track_color": int(track_color),
                "tokens": tokens,
                "gear": gear_pts,
                "center": center,
                "arms": (up, down, left, right),
            }
        )
    return tracks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each token around its rectangular track by the amount and direction
    encoded by the interior gear.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    tracks = _extract_tracks(g, bg)
    assert tracks, "expected at least one rectangular track"

    out = g.copy()
    for track in tracks:
        up, down, left, right = track["arms"]
        steps = up
        total_teeth = up + down + left + right
        delta = steps if total_teeth % 2 == 1 else -steps

        border = track["border"]
        border_index = {coord: idx for idx, coord in enumerate(border)}
        border_len = len(border)

        for r, c, _color in track["tokens"]:
            out[r, c] = track["track_color"]

        for r, c, color in track["tokens"]:
            dst = (border_index[(r, c)] + delta) % border_len
            nr, nc = border[dst]
            out[nr, nc] = color

    return out


def _rectangles_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_arms(rng: random.Random, parity: int) -> tuple[int, int, int, int]:
    up = rng.randint(0, MAX_ARM)
    options = [
        (up, down, left, right)
        for down in range(MAX_ARM + 1)
        for left in range(MAX_ARM + 1)
        for right in range(MAX_ARM + 1)
        if (up + down + left + right) % 2 == parity
    ]
    return rng.choice(options)


def _make_track_spec(rng: random.Random, parity: int) -> dict:
    up, down, left, right = _sample_arms(rng, parity)
    # Keep at least one background cell between the gear and the frame.
    half_h = max(1, max(up, down) + 1) + rng.randint(0, 1)
    half_w = max(1, max(left, right) + 1) + rng.randint(0, 1)
    outer_h = 2 * half_h + 3
    outer_w = 2 * half_w + 3
    return {
        "arms": (up, down, left, right),
        "outer_h": outer_h,
        "outer_w": outer_w,
        "token_count": rng.choice([1, 2]),
    }


def _place_specs(rng: random.Random, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    order = sorted(specs, key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)

    for spec in order:
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, BASE_CANVAS - oh):
            for c0 in range(1, BASE_CANVAS - ow):
                bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                if all(not _rectangles_too_close(bbox, other["bbox"], pad=1) for other in placed):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = rng.choice(candidates)
        bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
        placed.append({**spec, "bbox": bbox})

    return placed


def _draw_track(grid: np.ndarray, rng: random.Random, spec: dict) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    border = _border_clockwise(r0, c0, r1, c1)
    for r, c in border:
        grid[r, c] = TRACK

    token_indices = rng.sample(range(len(border)), int(spec["token_count"]))
    token_colors = rng.sample(TOKEN_COLORS, int(spec["token_count"]))
    for idx, color in zip(token_indices, token_colors):
        r, c = border[idx]
        grid[r, c] = color

    up, down, left, right = spec["arms"]
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    grid[cr, cc] = GEAR
    for step in range(1, up + 1):
        grid[cr - step, cc] = GEAR
    for step in range(1, down + 1):
        grid[cr + step, cc] = GEAR
    for step in range(1, left + 1):
        grid[cr, cc - step] = GEAR
    for step in range(1, right + 1):
        grid[cr, cc + step] = GEAR


def _crop_with_margin(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    points = np.argwhere(grid != BG)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    top = rng.randint(1, min(3, int(r0)))
    left = rng.randint(1, min(3, int(c0)))
    bottom = rng.randint(1, min(3, int(grid.shape[0] - 1 - r1)))
    right = rng.randint(1, min(3, int(grid.shape[1] - 1 - c1)))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()


def _validate_generated(grid: np.ndarray, expected_count: int) -> None:
    tracks = _extract_tracks(grid, _get_bg(grid))
    assert len(tracks) == expected_count
    assert any(track["arms"][0] > 0 for track in tracks)
    for track in tracks:
        assert track["track_color"] == TRACK
        assert 1 <= len(track["tokens"]) <= 2
        token_colors = {color for _r, _c, color in track["tokens"]}
        assert BG not in token_colors and TRACK not in token_colors and GEAR not in token_colors
    out = main(grid)
    assert out.shape == grid.shape


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    track_count = rng.randint(MIN_TRACKS, MAX_TRACKS)
    parities = [i % 2 for i in range(track_count)]
    if track_count % 2 == 1:
        parities[-1] = rng.randint(0, 1)
    rng.shuffle(parities)

    specs = [_make_track_spec(rng, parity) for parity in parities]
    if not any(spec["arms"][0] > 0 for spec in specs):
        return None

    placed = _place_specs(rng, specs)
    if placed is None:
        return None

    grid = np.full((BASE_CANVAS, BASE_CANVAS), BG, dtype=int)
    for spec in placed:
        _draw_track(grid, rng, spec)

    grid = _crop_with_margin(grid, rng)
    _validate_generated(grid, track_count)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
