from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape

    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _legend_from_component(g: np.ndarray, coords: np.ndarray) -> tuple[list[tuple[int, int]], list[int]] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height, width = r1 - r0 + 1, c1 - c0 + 1
    if min(height, width) != 1:
        return None
    if int(coords.shape[0]) != max(height, width):
        return None
    if int(coords.shape[0]) < 3:
        return None

    if height == 1:
        cells = [(r0, c) for c in range(c0, c1 + 1)]
    else:
        cells = [(r, c0) for r in range(r0, r1 + 1)]

    colors = [int(g[r, c]) for r, c in cells]
    if any(color == BG for color in colors):
        return None
    if len(colors) != len(set(colors)):
        return None
    return cells, colors


def _frame_from_component(
    g: np.ndarray,
    coords: np.ndarray,
    legend_cells: set[tuple[int, int]],
    cycle_colors: set[int],
) -> dict | None:
    pts = {(int(r), int(c)) for r, c in coords}
    if pts & legend_cells:
        return None
    colors = {int(g[r, c]) for r, c in pts}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    if color not in cycle_colors:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    side_h, side_w = r1 - r0 + 1, c1 - c0 + 1
    if side_h != side_w or side_h < 3:
        return None
    side = side_h

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if pts != border:
        return None
    if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG):
        return None
    return {"bbox": (r0, c0, r1, c1), "side": side, "color": color}


def _apply_frames(g: np.ndarray, cycle: list[int], frames: list[dict]) -> np.ndarray:
    color_to_pos = {int(color): i for i, color in enumerate(cycle)}
    out = g.copy()
    n = len(cycle)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        phase = color_to_pos[int(frame["color"])]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                layer = min(r - r0, r1 - r, c - c0, c1 - c)
                out[r, c] = int(cycle[(phase + layer) % n])
    return out


def _parse_input(g: np.ndarray) -> tuple[list[int], list[dict]]:
    comps = _components(g != BG)
    legend_candidates: list[tuple[list[tuple[int, int]], list[int]]] = []
    for comp in comps:
        legend = _legend_from_component(g, comp)
        if legend is not None:
            legend_candidates.append(legend)
    assert len(legend_candidates) == 1, "expected exactly one isolated 1-by-n legend strip"

    legend_cells, cycle = legend_candidates[0]
    legend_cell_set = set(legend_cells)
    cycle_set = set(cycle)
    frames: list[dict] = []
    for comp in comps:
        frame = _frame_from_component(g, comp, legend_cell_set, cycle_set)
        if frame is not None:
            frames.append(frame)

    assert frames, "expected at least one hollow square frame"
    frame_cells = sum(4 * (int(frame["side"]) - 1) for frame in frames)
    assert frame_cells + len(legend_cells) == int((g != BG).sum()), "unexpected foreground object"
    return cycle, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow square shells from a global cyclic legend, phased by each frame color.

    Input structure:
    - The background is black.
    - One isolated 1-by-n straight legend strip contains n distinct non-black colors; its
      left-to-right or top-to-bottom order is a cyclic color ring.
    - The other foreground objects are one-cell-thick hollow square frames whose single
      frame color appears in the legend.

    Rule:
    - Preserve the legend strip and every frame border.
    - For each frame, group its interior cells by their distance to that frame's border.
    - The first interior shell uses the legend color after the frame color, the second
      shell uses the next legend color, and so on with cyclic wrap through the center.

    Key insight:
    - The strip is a global color cycle, but every frame starts walking that same cycle
      from its own border color, so identical-sized frames with different border colors
      receive phase-shifted concentric fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    cycle, frames = _parse_input(g)
    return _apply_frames(g, cycle, frames)


# ----------------------------- generator -----------------------------


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _pad_rect(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _place_rect(
    rows: int,
    cols: int,
    height: int,
    width: int,
    occupied: list[tuple[int, int, int, int]],
    *,
    pad: int = 1,
) -> tuple[int, int] | None:
    candidates = [(r, c) for r in range(rows - height + 1) for c in range(cols - width + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rect = (r, c, r + height - 1, c + width - 1)
        padded = _pad_rect(rect, pad)
        if any(_rects_overlap(padded, other) for other in occupied):
            continue
        return r, c
    return None


def _draw_frame(grid: np.ndarray, r0: int, c0: int, side: int, color: int) -> None:
    r1, c1 = r0 + side - 1, c0 + side - 1
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _known_expected(grid: np.ndarray, cycle: list[int], frame_specs: list[dict]) -> np.ndarray:
    frames = [
        {"bbox": spec["bbox"], "side": spec["side"], "color": spec["color"]}
        for spec in frame_specs
    ]
    return _apply_frames(grid, cycle, frames)


def _build_frame_specs(cycle: list[int]) -> list[dict]:
    n = len(cycle)
    ordered_colors = random.sample(cycle, n)
    common_side = random.choice([5, 6])
    wrap_side = 2 * n + 1

    specs = [
        {"side": common_side, "color": ordered_colors[0]},
        {"side": common_side, "color": ordered_colors[1]},
        {"side": wrap_side, "color": ordered_colors[2]},
    ]

    remaining_colors = ordered_colors[3:]
    if not remaining_colors:
        remaining_colors = [random.choice(cycle)]

    even_used = common_side % 2 == 0
    for color in remaining_colors:
        if not even_used:
            side = random.choice([4, 6, 8])
            even_used = True
        else:
            side = random.choice([3, 4, 5, 6, 7, 8])
        specs.append({"side": side, "color": int(color)})

    while len(specs) < 4:
        side = random.choice([4, 6, 8]) if not even_used else random.choice([3, 4, 5, 6, 7])
        even_used = even_used or side % 2 == 0
        specs.append({"side": side, "color": int(random.choice(cycle))})

    random.shuffle(specs)
    return specs


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, cycle: list[int], frames: list[dict]) -> bool:
    if not np.any(inp != expected):
        return False
    frame_colors = [int(frame["color"]) for frame in frames]
    if set(frame_colors) != set(cycle):
        return False
    if len(set(frame_colors)) < 3:
        return False
    if max((int(frame["side"]) - 1) // 2 for frame in frames) < len(cycle):
        return False
    if not any(int(frame["side"]) % 2 == 0 for frame in frames):
        return False
    if not any(int(frame["side"]) % 2 == 1 for frame in frames):
        return False

    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            if int(a["side"]) != int(b["side"]) or int(a["color"]) == int(b["color"]):
                continue
            ar0, ac0, _ar1, _ac1 = a["bbox"]
            br0, bc0, _br1, _bc1 = b["bbox"]
            if int(expected[ar0 + 1, ac0 + 1]) != int(expected[br0 + 1, bc0 + 1]):
                break
        else:
            continue
        break
    else:
        return False

    reversed_cycle = list(reversed(cycle))
    if np.array_equal(_known_expected(inp, reversed_cycle, frames), expected):
        return False
    first_phase_cycle = [cycle[0]] + [c for c in cycle if c != cycle[0]]
    phase_erased = inp.copy()
    for frame in frames:
        fake = [{"bbox": frame["bbox"], "side": frame["side"], "color": first_phase_cycle[0]}]
        phase_erased = _apply_frames(phase_erased, first_phase_cycle, fake)
    if np.array_equal(phase_erased, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 5)
    cycle = random.sample(PALETTE, n)
    frame_specs = _build_frame_specs(cycle)

    rows = random.randint(23, 30)
    cols = random.randint(max(23, n + 4), 30)
    grid = np.full((rows, cols), BG, dtype=int)
    occupied: list[tuple[int, int, int, int]] = []

    legend_pos = _place_rect(rows, cols, 1, n, occupied, pad=1)
    if legend_pos is None:
        return None
    lr, lc = legend_pos
    grid[lr, lc : lc + n] = np.array(cycle, dtype=int)
    occupied.append((lr, lc, lr, lc + n - 1))

    placed_frames: list[dict] = []
    for spec in sorted(frame_specs, key=lambda item: int(item["side"]), reverse=True):
        side = int(spec["side"])
        pos = _place_rect(rows, cols, side, side, occupied, pad=1)
        if pos is None:
            return None
        r0, c0 = pos
        _draw_frame(grid, r0, c0, side, int(spec["color"]))
        bbox = (r0, c0, r0 + side - 1, c0 + side - 1)
        occupied.append(bbox)
        placed_frames.append({"bbox": bbox, "side": side, "color": int(spec["color"])})

    expected = _known_expected(grid, cycle, placed_frames)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid, expected, cycle, placed_frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-cycle hollow-frame input")
