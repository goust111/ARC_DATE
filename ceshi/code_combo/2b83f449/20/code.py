# Combo variant of 2b83f449 (index 20)
# combo_spec: D1/1+D10/0
# Name: 矩形门+十字扩展
#
# Design analysis
# ---------------
# 1. Concrete fused geometry:
#    - The seed task alternates `pipe / wall / pipe`. A gate is a 1x3 bar in one wall
#      row, and opening it leaves two bg doorways plus a thin 3x1 center post.
#    - This combo needs a true 2x3 source gate, so the lattice becomes
#      `pipe / wall / wall / pipe` (period 3 rows). Each input gate is a solid 2x3
#      rectangle on the two wall rows between two neighboring pipe rows.
#    - A literal discrete 90-degree rotation of a 2x3 block to a 3x2 pillar is
#      awkward inside a 2-row wall band: it becomes off-center and visually noisy.
#      The implementation therefore uses the minimal symmetric discrete analogue that
#      still expresses the requested "rectangle gate + larger cross blocker":
#      the 2x3 gate core turns into a 2x3 open-color blocker, and it grows one extra
#      center pixel into the pipe row above and the pipe row below. That produces an
#      8-pixel thick cross. The wall band simultaneously opens two bg side doorways at
#      center-2 and center+2 on both wall rows, so tokens can still pass around the
#      cross.
#    - Small example with one gate centered at column 8:
#      input wall band rows:    `00000777000` / `00000777000`
#      output local pattern:    row above has `6` at col 8,
#                               wall rows have `66 666 66` style core with bg only
#                               at cols 6 and 10,
#                               row below has `6` at col 8.
#      Compared with the seed task, the blocker is visibly wider and it splits the
#      neighboring pipe rows themselves, not only the wall row.
#
# 2. Complexity calibration / observability:
#    - One rectangular gate is placed in every wall band. Consecutive gate centers
#      move by exactly one column left or right, so the left doorway chain and right
#      doorway chain both stay connected for many levels. This gives repeated evidence
#      of the new geometry across a sample instead of a one-off special case.
#    - The only colors remain the seed colors {0,3,6,7,8}. The new rule is conveyed
#      by shape and placement, not by hidden color arithmetic.
#    - The output is still easy to parse by eye: black wall, teal pipe, orange input
#      rectangles, magenta output crosses, green tokens.
#
# 3. Original implicit assumptions that no longer hold:
#    - Gate detection can no longer search for 1x3 components; it must search for
#      isolated 2x3 components.
#    - A token no longer drops across 2 rows; it crosses a 2-row wall band, so each
#      vertical step is 3 rows.
#    - The opened blocker is no longer a single center post. Pipe rows can contain
#      blocker pixels from both the band above and the band below.
#    - "Every segment has <= 1 exit" is not generally true anymore, because each thick
#      cross creates two side doorways. The solver therefore uses the side that the
#      token started on: left-edge tokens always choose the leftmost reachable doorway,
#      right-edge tokens choose the rightmost one.
#
# 4. Generator legality guarantees:
#    - Gate centers stay in `[max(4, n_bands), min(W-5, W-n_bands-1)]`, so the gate
#      rectangle, both side doorways, and the final token stacks all fit in-bounds.
#    - Consecutive centers differ by exactly 1. This keeps both side doorway chains
#      connected across adjacent bands; without that, a token could fall into a segment
#      that cannot reach the next doorway.
#    - The bottom left/right basin capacities are at least the number of left/right
#      tokens, so packing never truncates tokens.
#
# 5. Visual cleanliness / inferability:
#    - The side doorways are outside the original 2x3 input rectangle, making the
#      expansion visually obvious in the output.
#    - No overlapping object types share the same color. Open-color cells are always
#      blocker material; bg doorways remain teal.
#
# 6. New solver assumptions and generator support:
#    - `main()` assumes every true gate is an isolated 2x3 component. The generator
#      uses exactly one gate per wall band and keeps them separated by full pipe rows.
#    - `main()` assumes tokens start on edge cells of pipe rows and use a fixed source
#      side throughout the fall. The generator uses the same edge-token placement for
#      every non-bottom pipe row.
#    - `main()` assumes the intended path exists through consecutive left/right
#      doorways. The generator enforces center deltas of exactly 1 to guarantee that.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask


WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8
BAND = 3


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))

    gate = None
    for col in sorted(colors):
        if col == WALL:
            continue
        lab, n = label_mask(g == int(col), connectivity=4)
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(k))
            if int(pts.shape[0]) != 6:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 2 and int(c1 - c0 + 1) == 3:
                gate = int(col)
                break
        if gate is not None:
            break
    if gate is None:
        raise ValueError("failed to infer 2x3 gate color")

    counts = Counter(map(int, g.flatten()))
    token_candidates = [c for c in colors if int(c) not in (WALL, gate)]
    if not token_candidates:
        raise ValueError("failed to infer token color")
    token = int(min(token_candidates, key=lambda c: (int(counts[int(c)]), int(c))))

    bg_candidates = [c for c in colors if int(c) not in (WALL, gate, token)]
    if not bg_candidates:
        raise ValueError("failed to infer bg color")
    bg = int(max(bg_candidates, key=lambda c: (int(counts[int(c)]), -int(c))))

    open_col = OPEN if gate == GATE and OPEN not in colors else int(gate - 1 if gate > 0 else OPEN)
    return gate, token, bg, open_col


def _find_gates(grid: np.ndarray, gate: int) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    lab, n = label_mask(g == int(gate), connectivity=4)
    gates: list[tuple[int, int]] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        if int(pts.shape[0]) != 6:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 2 or int(c1 - c0 + 1) != 3:
            continue
        gates.append((int(r0), int(c0 + 1)))
    gates.sort()
    return gates


def _open_from_input(grid: np.ndarray, gate: int, bg: int, open_col: int) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    height, width = out.shape
    for top, center in _find_gates(out, gate):
        if not (0 <= top - 1 < height and 0 <= top + 2 < height):
            raise ValueError("gate band does not have adjacent pipe rows")
        if not (0 <= center - 2 and center + 2 < width):
            raise ValueError("gate does not have room for side doorways")
        # Thick cross core.
        out[top : top + 2, center - 1 : center + 2] = int(open_col)
        out[top - 1, center] = int(open_col)
        out[top + 2, center] = int(open_col)
        # Symmetric side doorways through the 2-row wall band.
        out[top : top + 2, center - 2] = int(bg)
        out[top : top + 2, center + 2] = int(bg)
    return out


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != int(bg):
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == int(bg):
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + BAND >= grid.shape[0]:
        return []
    a, b = seg
    exits: list[int] = []
    for c in range(a, b + 1):
        if (
            int(grid[r + 1, c]) == int(bg)
            and int(grid[r + 2, c]) == int(bg)
            and int(grid[r + 3, c]) == int(bg)
        ):
            exits.append(c)
    return exits


def _simulate_on_opened(
    opened_with_tokens: np.ndarray, token: int, bg: int
) -> tuple[np.ndarray, list[int], int]:
    out = np.array(opened_with_tokens, dtype=int).copy()
    height, width = out.shape

    token_pts = sorted((int(r), int(c)) for r, c in np.argwhere(out == int(token)))
    out[out == int(token)] = int(bg)

    finals: list[tuple[int, int, int | None, str]] = []
    drop_counts: list[int] = []
    for start_r, start_c in token_pts:
        cur_r, cur_c = int(start_r), int(start_c)
        side = "left" if cur_c <= width // 2 else "right"
        last_entry: int | None = None
        drops = 0

        if int(out[cur_r, cur_c]) == int(bg):
            while True:
                seg = _find_segment(out, cur_r, cur_c, bg)
                if seg is None:
                    break
                exits = _exits_below(out, cur_r, seg, bg)
                if not exits:
                    break
                choice = int(min(exits) if side == "left" else max(exits))
                cur_r += BAND
                cur_c = choice
                last_entry = choice
                drops += 1

        finals.append((cur_r, cur_c, last_entry, side))
        drop_counts.append(drops)

    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)
    for r, c, last_entry, side in finals:
        seg = _find_segment(out, r, c, bg)
        if seg is None:
            continue
        a, b = seg
        if last_entry is not None:
            pack_end = b if (last_entry - a) <= (b - last_entry) else a
        else:
            pack_end = a if side == "left" else b
        buckets[(r, seg, pack_end)] += 1

    placed = 0
    for (r, (a, b), pack_end), count in buckets.items():
        if pack_end == a:
            for i in range(count):
                if a + i <= b:
                    out[r, a + i] = int(token)
                    placed += 1
        else:
            for i in range(count):
                if b - i >= a:
                    out[r, b - i] = int(token)
                    placed += 1

    return out, drop_counts, placed


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gate, token, bg, open_col = _infer_colors(g)
    opened = _open_from_input(g, gate, bg, open_col)
    out, _drop_counts, _placed = _simulate_on_opened(opened, token, bg)
    return out


def _build_opened_grid(
    centers: list[int], width: int, *, with_spikes: bool
) -> np.ndarray:
    n_bands = len(centers)
    height = BAND * n_bands + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(n_bands + 1):
        grid[BAND * pipe_idx, :] = BG

    for band_idx, center in enumerate(centers):
        top = BAND * band_idx + 1
        grid[top : top + 2, center - 1 : center + 2] = OPEN
        grid[top : top + 2, center - 2] = BG
        grid[top : top + 2, center + 2] = BG
        if with_spikes:
            grid[top - 1, center] = OPEN
            grid[top + 2, center] = OPEN

    return grid


def _build_input(centers: list[int], width: int) -> np.ndarray:
    n_bands = len(centers)
    height = BAND * n_bands + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(n_bands + 1):
        r = BAND * pipe_idx
        grid[r, :] = BG

    for band_idx, center in enumerate(centers):
        top = BAND * band_idx + 1
        grid[top : top + 2, center - 1 : center + 2] = GATE

    for pipe_idx in range(n_bands):
        r = BAND * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _simulate_output(opened: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    n_bands = (opened.shape[0] - 1) // BAND
    seeded = np.array(opened, dtype=int).copy()
    for pipe_idx in range(n_bands):
        seeded[BAND * pipe_idx, 0] = TOKEN
        seeded[BAND * pipe_idx, seeded.shape[1] - 1] = TOKEN
    return _simulate_on_opened(seeded, TOKEN, BG)


def _sample_centers(rng: np.random.Generator, n_bands: int, width: int) -> list[int] | None:
    lo = max(4, n_bands)
    hi = min(width - 5, width - n_bands - 1)
    if lo > hi:
        return None

    start_lo = lo + 1 if lo + 1 <= hi - 1 else lo
    start_hi = hi - 1 if lo + 1 <= hi - 1 else hi
    center = int(rng.integers(start_lo, start_hi + 1))
    centers = [center]

    for _ in range(1, n_bands):
        options = [center + dc for dc in (-1, 1) if lo <= center + dc <= hi]
        if not options:
            return None
        center = int(rng.choice(options))
        centers.append(center)

    if len(set(centers)) < 2:
        return None
    return centers


def _behavior_metrics(
    centers: list[int], width: int, combo_output: np.ndarray
) -> dict[str, int]:
    combo_opened = _build_opened_grid(centers, width, with_spikes=True)
    rect_opened = _build_opened_grid(centers, width, with_spikes=False)
    rect_output, _rect_drops, rect_placed = _simulate_output(rect_opened)

    metrics = {
        "gate_count": int(len(centers)),
        "extra_cross_pixels": int(np.sum((combo_opened == OPEN) & (rect_opened != OPEN))),
        "full_output_diff": int(np.sum(combo_output != rect_output)),
        "token_output_diff": int(np.sum((combo_output == TOKEN) != (rect_output == TOKEN))),
        "rect_tokens_placed": int(rect_placed),
    }
    return metrics


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_bands = int(rng.integers(5, 7))
    width_min = max(17, 2 * n_bands + 6)
    width = int(rng.integers(width_min, 21))

    centers = _sample_centers(rng, n_bands, width)
    if centers is None:
        return None

    opened = _build_opened_grid(centers, width, with_spikes=True)
    output, drop_counts, placed = _simulate_output(opened)
    if placed != 2 * n_bands:
        return None
    if max(drop_counts, default=0) < 3:
        return None

    inp = _build_input(centers, width)
    if not np.array_equal(main(inp), output):
        return None

    metrics = _behavior_metrics(centers, width, output)
    if metrics["extra_cross_pixels"] < 2 * n_bands:
        return None
    if metrics["full_output_diff"] <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/20")
