from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
SEED = 4
OUTER_COLOR = 2
MIDDLE_COLOR = 3
INNER_COLOR = 1

FRAME_COLORS = (OUTER_COLOR, MIDDLE_COLOR, INNER_COLOR)
GEN_RETRIES = 400
INF = 10**9


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _boundary_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _infer_frame(mask: np.ndarray, color: int) -> dict:
    coords = np.argwhere(mask)
    assert coords.size > 0, f"missing frame color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox = (int(r0), int(c0), int(r1), int(c1))
    boundary = set(_boundary_cells(bbox))
    colored = {tuple(map(int, rc)) for rc in coords}
    assert colored <= boundary, "frame must lie on a rectangle boundary"
    gap = list(boundary - colored)
    assert len(gap) == 1, "each frame must have exactly one missing boundary cell"
    gap_r, gap_c = gap[0]
    if gap_r == r0:
        side = "top"
    elif gap_r == r1:
        side = "bottom"
    elif gap_c == c0:
        side = "left"
    else:
        side = "right"
    assert len(colored) == len(boundary) - 1, "frame must be 1-pixel thick with one gap"
    return {"color": int(color), "bbox": bbox, "gap": (gap_r, gap_c), "side": side}


def _parse_frames(grid: np.ndarray) -> list[dict]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BLACK, SEED)]
    frames = [_infer_frame(grid == color, color) for color in colors]
    assert len(frames) == 3, "expected exactly three frame colors"
    frames.sort(
        key=lambda frame: (
            -(frame["bbox"][2] - frame["bbox"][0] + 1) * (frame["bbox"][3] - frame["bbox"][1] + 1),
            frame["bbox"][0],
            frame["bbox"][1],
        )
    )
    for outer, inner in zip(frames, frames[1:]):
        r0, c0, r1, c1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert r0 < ir0 < ir1 < r1 and c0 < ic0 < ic1 < c1, "frames must be strictly nested"
        assert ir0 - r0 >= 2 and ic0 - c0 >= 2, "annulus between frames must have positive width"
    return frames


def _containment_masks(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    h, w = shape
    contain = np.zeros((h, w), dtype=np.int8)
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        contain[r0 + 1 : r1, c0 + 1 : c1] |= 1 << idx
    return contain


def _transition_last(cur_mask: int, next_mask: int, last_color: int, frames: list[dict]) -> int:
    gain = int(next_mask & ~cur_mask)
    lose = int(cur_mask & ~next_mask)
    changed = gain | lose
    if changed == 0:
        return int(last_color)
    assert changed & (changed - 1) == 0, "a 4-neighbor step may cross at most one frame boundary"
    layer_idx = changed.bit_length() - 1
    return int(frames[layer_idx]["color"])


def _trace_flood(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seeds = np.argwhere(g == SEED)
    assert len(seeds) == 1, "expected exactly one yellow seed"
    seed_r, seed_c = map(int, seeds[0])

    frames = _parse_frames(g)
    contain = _containment_masks(g.shape, frames)
    passable = (g == BLACK) | (g == SEED)

    state_colors = [0] + [int(frame["color"]) for frame in frames]
    state_index = {color: idx for idx, color in enumerate(state_colors)}
    dist = np.full((h, w, len(state_colors)), INF, dtype=int)
    dist[seed_r, seed_c, 0] = 0
    q: deque[tuple[int, int, int]] = deque([(seed_r, seed_c, 0)])

    while q:
        r, c, sid = q.popleft()
        cur_d = int(dist[r, c, sid])
        cur_mask = int(contain[r, c])
        last_color = int(state_colors[sid])
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or not passable[nr, nc]:
                continue
            next_last = _transition_last(cur_mask, int(contain[nr, nc]), last_color, frames)
            next_sid = state_index[next_last]
            if dist[nr, nc, next_sid] != INF:
                continue
            dist[nr, nc, next_sid] = cur_d + 1
            q.append((nr, nc, next_sid))

    best = dist.min(axis=2)
    last_sets: list[list[set[int]]] = [[set() for _ in range(w)] for _ in range(h)]
    for sid, color in enumerate(state_colors):
        mask = dist[:, :, sid] == best
        for r, c in np.argwhere(mask & (best < INF)):
            last_sets[int(r)][int(c)].add(int(color))

    return {
        "frames": frames,
        "contain": contain,
        "best": best,
        "last_sets": last_sets,
        "seed": (seed_r, seed_c),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood from the outer seed through nested frame gaps and recolor each reached enclosed region by the color of the last wall crossed.

    Input structure:
    - The grid has black background, one yellow seed in the area outside the outer frame, and exactly three 1-pixel-thick rectangular frames with one gap each.
    - The three frames are strictly nested, use distinct colors, and their gaps are black cells on the corresponding rectangle boundaries.
    - Only black cells and the yellow seed are passable; colored frame pixels are walls.

    Rule:
    - Start a 4-connected flood from the yellow seed through black cells only.
    - Whenever a shortest flood path steps across a frame boundary through that frame's gap, update the path's "last crossed wall color" to that frame's color.
    - Recolor every reachable black cell that lies inside at least one frame to its unique shortest-path last crossed wall color; keep frame pixels, the yellow seed, and all outside black cells unchanged.

    Key insight:
    - The target color is not tied to raw depth alone in the implementation: it is carried by the flood state and changes only when a path actually crosses a specific frame boundary through a gap.
    """
    g = np.asarray(grid, dtype=int)
    trace = _trace_flood(g)
    contain = trace["contain"]
    best = trace["best"]
    last_sets = trace["last_sets"]

    out = g.copy()
    target = (g == BLACK) & (contain != 0) & (best < INF)
    for r, c in np.argwhere(target):
        colors = last_sets[int(r)][int(c)]
        assert colors, "reachable internal black cell must have at least one shortest-path state"
        assert len(colors) == 1, "generator must avoid ambiguous shortest paths with different last walls"
        fill_color = next(iter(colors))
        assert fill_color != 0, "reachable internal region must be entered by crossing a wall"
        out[int(r), int(c)] = int(fill_color)
    return out


def _gap_from_side(
    bbox: tuple[int, int, int, int],
    side: str,
    forced_coord: int | None = None,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        col = forced_coord if forced_coord is not None else random.randint(c0 + 1, c1 - 1)
        row = r0 if side == "top" else r1
        return row, int(col)
    row = forced_coord if forced_coord is not None else random.randint(r0 + 1, r1 - 1)
    col = c0 if side == "left" else c1
    return int(row), col


def _shared_gap_coord(frames: list[dict], side: str, indices: tuple[int, int]) -> int:
    a = frames[indices[0]]["bbox"]
    b = frames[indices[1]]["bbox"]
    if side in ("top", "bottom"):
        lo = max(a[1], b[1]) + 1
        hi = min(a[3], b[3]) - 1
    else:
        lo = max(a[0], b[0]) + 1
        hi = min(a[2], b[2]) - 1
    assert lo <= hi
    return random.randint(lo, hi)


def _build_frames(
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict], str]:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)

    for _ in range(200):
        outer_h = random.randint(10, min(16, h - 2))
        outer_w = random.randint(10, min(16, w - 2))
        gap1 = random.randint(2, 4)
        gap2 = random.randint(2, 3)
        inner_h = outer_h - 2 * (gap1 + gap2)
        inner_w = outer_w - 2 * (gap1 + gap2)
        if inner_h < 5 or inner_w < 5:
            continue
        top = random.randint(1, h - outer_h - 1)
        left = random.randint(1, w - outer_w - 1)
        bboxes = [
            (top, left, top + outer_h - 1, left + outer_w - 1),
            (top + gap1, left + gap1, top + outer_h - gap1 - 1, left + outer_w - gap1 - 1),
            (
                top + gap1 + gap2,
                left + gap1 + gap2,
                top + outer_h - gap1 - gap2 - 1,
                left + outer_w - gap1 - gap2 - 1,
            ),
        ]
        break
    else:
        raise RuntimeError("failed to sample nested frame sizes")

    frames = [{"color": color, "bbox": bbox} for color, bbox in zip(FRAME_COLORS, bboxes)]

    branch = random.choices(
        ["distinct", "outer_mid_aligned", "mid_inner_aligned", "same_side_shifted"],
        weights=[2, 3, 3, 2],
        k=1,
    )[0]
    sides = ["top", "bottom", "left", "right"]

    if branch == "distinct":
        chosen_sides = random.sample(sides, 3)
        for frame, side in zip(frames, chosen_sides):
            frame["side"] = side
            frame["gap"] = _gap_from_side(frame["bbox"], side)
    elif branch == "outer_mid_aligned":
        shared_side = random.choice(sides)
        shared_coord = _shared_gap_coord(frames, shared_side, (0, 1))
        third_side = random.choice([side for side in sides if side != shared_side])
        assignments = [shared_side, shared_side, third_side]
        for idx, (frame, side) in enumerate(zip(frames, assignments)):
            frame["side"] = side
            forced = shared_coord if idx in (0, 1) else None
            frame["gap"] = _gap_from_side(frame["bbox"], side, forced_coord=forced)
    elif branch == "mid_inner_aligned":
        shared_side = random.choice(sides)
        shared_coord = _shared_gap_coord(frames, shared_side, (1, 2))
        first_side = random.choice([side for side in sides if side != shared_side])
        assignments = [first_side, shared_side, shared_side]
        for idx, (frame, side) in enumerate(zip(frames, assignments)):
            frame["side"] = side
            forced = shared_coord if idx in (1, 2) else None
            frame["gap"] = _gap_from_side(frame["bbox"], side, forced_coord=forced)
    else:
        shared_side = random.choice(sides)
        assignments = [shared_side, shared_side, random.choice(sides)]
        for frame, side in zip(frames, assignments):
            frame["side"] = side
            frame["gap"] = _gap_from_side(frame["bbox"], side)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        for r, c in _boundary_cells(frame["bbox"]):
            grid[r, c] = frame["color"]
        gr, gc = frame["gap"]
        grid[gr, gc] = BLACK

    return grid, frames, branch


def _place_seed(grid: np.ndarray, outer_frame: dict, align_with_gap: bool) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = outer_frame["bbox"]
    gap_r, gap_c = outer_frame["gap"]
    side = outer_frame["side"]

    if align_with_gap:
        if side == "top":
            row = random.randint(0, r0 - 1)
            grid[row, gap_c] = SEED
            return
        if side == "bottom":
            row = random.randint(r1 + 1, h - 1)
            grid[row, gap_c] = SEED
            return
        if side == "left":
            col = random.randint(0, c0 - 1)
            grid[gap_r, col] = SEED
            return
        col = random.randint(c1 + 1, w - 1)
        grid[gap_r, col] = SEED
        return

    options: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if grid[r, c] == BLACK:
                options.append((r, c))
    assert options
    sr, sc = random.choice(options)
    grid[sr, sc] = SEED


def _ring_masks(frames: list[dict], contain: np.ndarray) -> list[np.ndarray]:
    masks = []
    for depth in range(1, 4):
        need = (1 << depth) - 1
        masks.append(contain == need)
    return masks


def _semantic_checks(inp: np.ndarray, out: np.ndarray, branch: str) -> None:
    trace = _trace_flood(inp)
    frames = trace["frames"]
    contain = trace["contain"]
    best = trace["best"]
    last_sets = trace["last_sets"]

    assert np.array_equal(main(inp), out)
    assert np.count_nonzero(out != inp) >= 12, "transformation must be visibly non-trivial"
    assert np.count_nonzero(inp == SEED) == 1
    assert np.count_nonzero(out == SEED) == 1

    ring_masks = _ring_masks(frames, contain)
    expected_colors = [frame["color"] for frame in frames]
    for idx, ring_mask in enumerate(ring_masks):
        target = ring_mask & (inp == BLACK) & (best < INF)
        assert np.any(target), "every annulus and core must contain reachable black cells"
        colors = {next(iter(last_sets[int(r)][int(c)])) for r, c in np.argwhere(target)}
        assert colors == {expected_colors[idx]}, "each chamber must get the correct last-crossed wall color"

    outer_gap = frames[0]["gap"]
    assert out[outer_gap] == BLACK, "outer opening itself should stay outside and remain black"
    if branch in {"outer_mid_aligned", "mid_inner_aligned"}:
        aligned = False
        for a, b in ((frames[0], frames[1]), (frames[1], frames[2])):
            same_side = a["side"] == b["side"]
            same_coord = (
                a["gap"][0] == b["gap"][0] if a["side"] in ("left", "right") else a["gap"][1] == b["gap"][1]
            )
            aligned = aligned or (same_side and same_coord)
        assert aligned, "aligned-branch examples must really align at least one adjacent gap pair"


def _generate_once() -> np.ndarray:
    h = random.randint(13, 19)
    w = random.randint(13, 19)
    grid, frames, branch = _build_frames((h, w))
    _place_seed(grid, frames[0], align_with_gap=random.random() < 0.7)

    out = main(grid)
    _semantic_checks(grid, out, branch)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _generate_once()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid nested-gap ARC input")
