# Combo variant of 1818057f (index 5)
# combo_spec: D2/1+D4/0
# Name: 双色层序号编码着色

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


"""
Design analysis for the combo variant
-------------------------------------
a) Concrete behavior and trace:
   - The seed task has one foreground layer: yellow pixels on red. A plus is any
     yellow center whose 4 orthogonal neighbors are also yellow, and every such
     5-cell footprint becomes teal.
   - This combo keeps the same local plus geometry but splits the foreground into
     two independent layers, yellow and green. A yellow pixel only forms a plus
     with yellow neighbors; a green pixel only forms a plus with green neighbors.
   - After detecting both layers, the solver does not color them per source layer.
     Instead it merges every detected plus center, sorts them in row-major order,
     and assigns colors from a short repeating palette.
   - Example trace: suppose row-major detected centers are
       (2, 6, yellow), (4, 3, green), (7, 8, yellow), (9, 5, green), (11, 10, green).
     The output cross colors become
       palette[0], palette[1], palette[2], palette[0], palette[1].
     Under the seed task all five crosses would have become the same teal color;
     under D4/0 alone the color would depend only on source layer; here the final
     color depends on the global merged rank after per-layer detection.
   - The changed reasoning chain is therefore:
       detect pluses separately per color -> union all centers -> sort by position
       -> color by sequence index.
     None of those steps can be skipped without changing the answer.

b) Complexity calibration:
   - The palette is kept short, `PALETTE = (8, 6, 7)`, so the output introduces
     only three transformed colors. This is enough to expose a cycle without making
     the color rule arbitrary.
   - The generator uses 5-7 planted pluses when possible. That makes a single
     sample likely to show both multiple palette colors and at least one wraparound.
   - Both yellow and green pluses are always present, and the accepted samples
     require that each source color appears with at least two different output
     colors. This prevents the task from visually collapsing into a simple fixed
     source->target mapping.

c) Hidden assumption audit for the original main():
   - Seed assumption: there is only one non-background foreground color, so
     `colors[0]` is enough. Invalid now. The combo solver must explicitly inspect
     yellow and green as separate detection layers.
   - Seed assumption: every detected plus is recolored to the same target. Invalid
     now. The target color depends on the plus's position in the global row-major
     order across both layers.
   - Seed assumption: recoloring during scanning is harmless because all matches
     share one target color. Risky now. The combo solver first gathers centers from
     the immutable input grid and only then recolors by sorted index.
   - Seed assumption: generator only has to avoid accidental pluses for one color.
     Invalid now. Any added yellow distractor must avoid creating an unplanned
     yellow plus, and likewise for green.
   - New combo assumption: only yellow and green act as meaningful source layers.
     `generate_input()` guarantees that palette so `main()` can stay simple.

d) Placement legality:
   - Planted centers stay at least two cells away from the border so every 5-cell
     cross fits and each arm can potentially grow further.
   - All centers, regardless of color, keep Manhattan distance >= 4, which prevents
     cross-footprint overlap.
   - Extensions and distractors are only written onto background cells, never over
     another planted structure.
   - Final validation checks that the actually detectable yellow and green centers
     match the planned ones exactly, so no accidental extra plus survives.

e) Visual observability:
   - Row-major order is spatial and human-visible: higher pluses are colored earlier,
     and ties break left-to-right.
   - Because both source colors participate in the same global ordering, nearby
     yellow and green pluses can receive consecutive palette colors. This visibly
     signals that source color does not determine the output color by itself.
   - The generator rejects layouts where one source color would appear with only one
     target color, because such samples would weaken the evidence for the sequence rule.

f) Visual cleanliness:
   - Crosses remain thin and spaced apart. Only their 5-cell footprints are recolored;
     extensions and distractors keep their input color.
   - The grid therefore still reads as "mixed yellow/green shapes on red", with a
     clean overlay of small recolored plus footprints rather than a noisy full-scene
     recoloring.

g) Change inferability:
   - The output palette is intentionally small and repeats. A human can compare
     several examples and notice that the same source color does not map to one
     fixed output color, while the top-to-bottom / left-to-right sequence does.
   - The generator further ensures at least one palette color is reused on both
     source layers within a sample, so the color change is structurally tied to
     order rather than to hidden color ids.

h) New solver assumptions and generator guarantees:
   - `main()` assumes plus detection is color-layer-local and only depends on the
     immutable input grid. The generator never requires chained recoloring effects.
   - `main()` assumes every meaningful transformed plus belongs to yellow or green.
     The generator uses exactly those two input layers.
   - `main()` assumes row-major ordering over all detected centers is the intended
     tie-break. The generator builds expected outputs with that exact ordering.
"""

BG = 2
YELLOW = 4
GREEN = 3
SOURCE_COLORS = (YELLOW, GREEN)
PALETTE = (8, 6, 7)
PLUS_DIRS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ORTH_DIRS = PLUS_DIRS[1:]


def _cross_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in PLUS_DIRS]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != fg:
        return False
    return all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTH_DIRS)


def _plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def _collect_sorted_pluses(grid: np.ndarray) -> list[tuple[int, int, int]]:
    H, W = grid.shape
    pluses: list[tuple[int, int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            fg = int(grid[r, c])
            if fg not in SOURCE_COLORS:
                continue
            if _is_plus_center(grid, r, c, fg):
                pluses.append((r, c, fg))
    return pluses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect yellow and green '+' motifs independently, then recolor all of them by global scan rank.

    Input structure:
    - The background is the most common color.
    - Two foreground colors, yellow and green, form mixed blobs, lines, and embedded thin '+' motifs on that background.

    Rule:
    - A '+' center is any yellow or green pixel whose 4 orthogonal neighbors share that same color; the two colors are checked independently.
    - Merge every detected plus center across both colors, sort them from top to bottom and left to right, and recolor each 5-cell cross with the repeating palette (8, 6, 7) by that global index.
    - Leave every non-plus pixel unchanged.

    Key insight:
    - Local same-color detection decides which centers qualify, but source color does not decide the final output color; the transformed color comes from the plus's row-major rank within the combined yellow-and-green hit list.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    if np.all(g == bg):
        return out

    pluses = _collect_sorted_pluses(g)
    for idx, (r, c, _) in enumerate(pluses):
        color = PALETTE[idx % len(PALETTE)]
        for dr, dc in PLUS_DIRS:
            out[r + dr, c + dc] = color
    return out


def _would_create_unplanned_plus(
    grid: np.ndarray, r: int, c: int, fg: int, allowed_centers: set[tuple[int, int]]
) -> bool:
    """Return True iff writing fg at (r,c) creates a same-color plus outside allowed_centers."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_DIRS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg and (cr, cc) not in allowed_centers:
            return True
    return False


def _pick_source_sequence(count: int, rng: np.random.Generator) -> list[int] | None:
    if count < 5:
        return None
    for _ in range(200):
        seq = [YELLOW, YELLOW, GREEN, GREEN]
        extra = [int(rng.choice(SOURCE_COLORS)) for _ in range(count - 4)]
        seq.extend(extra)
        rng.shuffle(seq)

        if not any(seq[i] != seq[i + 1] for i in range(count - 1)):
            continue

        per_fg_targets = {
            fg: {PALETTE[i % len(PALETTE)] for i, src in enumerate(seq) if src == fg}
            for fg in SOURCE_COLORS
        }
        if any(len(targets) < 2 for targets in per_fg_targets.values()):
            continue

        palette_sources = {color: set() for color in PALETTE}
        for i, src in enumerate(seq):
            palette_sources[PALETTE[i % len(PALETTE)]].add(src)
        if not any(len(sources) >= 2 for sources in palette_sources.values()):
            continue

        return seq
    return None


def _build_expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for idx, (r, c, _) in enumerate(_collect_sorted_pluses(inp)):
        color = PALETTE[idx % len(PALETTE)]
        for dr, dc in PLUS_DIRS:
            out[r + dr, c + dc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    total = H * W

    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    separable: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - er) + abs(c - ec) >= 4 for er, ec in separable):
            separable.append((r, c))

    max_k = min(7, len(separable))
    if max_k < 5:
        return None
    K = int(rng.integers(5, max_k + 1))
    centers = sorted(separable[:K])

    source_sequence = _pick_source_sequence(K, rng)
    if source_sequence is None:
        return None
    center_specs = [(r, c, fg) for (r, c), fg in zip(centers, source_sequence)]

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)

    for idx, (r, c, fg) in enumerate(center_specs):
        dst = PALETTE[idx % len(PALETTE)]
        cross = _cross_cells(r, c)
        if any(int(inp[rr, cc]) != BG for rr, cc in cross):
            return None
        for rr, cc in cross:
            inp[rr, cc] = fg
            out[rr, cc] = dst

    rng.shuffle(center_specs)
    for r, c, fg in center_specs:
        feasible_runs: list[list[tuple[int, int]]] = []
        for dr, dc in ORTH_DIRS:
            run: list[tuple[int, int]] = []
            nr, nc = r + 2 * dr, c + 2 * dc
            while 0 <= nr < H and 0 <= nc < W and int(inp[nr, nc]) == BG and len(run) < 3:
                run.append((nr, nc))
                nr += dr
                nc += dc
            if run:
                feasible_runs.append(run)
        if not feasible_runs:
            return None

        rng.shuffle(feasible_runs)
        n_dirs = 1 if len(feasible_runs) == 1 else int(rng.integers(1, min(2, len(feasible_runs)) + 1))
        added = 0
        for run in feasible_runs[:n_dirs]:
            ext_len = int(rng.integers(1, min(3, len(run)) + 1))
            for rr, cc in run[:ext_len]:
                if int(inp[rr, cc]) != BG:
                    break
                inp[rr, cc] = fg
                out[rr, cc] = fg
                added += 1
        if added == 0:
            return None

    current_fg = int(np.sum(inp != BG))
    if current_fg >= total // 2:
        return None

    min_density = max(0.26, current_fg / total)
    if min_density > 0.41:
        return None
    target_density = float(rng.uniform(min_density, 0.41))
    target_fg = int(total * target_density)

    allowed_centers = {
        fg: {(r, c) for r, c, src in center_specs if src == fg}
        for fg in SOURCE_COLORS
    }
    color_counts = {
        fg: int(np.sum(inp == fg))
        for fg in SOURCE_COLORS
    }

    bg_positions = list(zip(*np.where(inp == BG)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        preferred = [YELLOW, GREEN] if color_counts[YELLOW] <= color_counts[GREEN] else [GREEN, YELLOW]
        if rng.random() < 0.35:
            preferred.reverse()
        for fg in preferred:
            if _would_create_unplanned_plus(inp, int(r), int(c), fg, allowed_centers[fg]):
                continue
            inp[int(r), int(c)] = fg
            out[int(r), int(c)] = fg
            color_counts[fg] += 1
            current_fg += 1
            break

    if not (0.26 <= current_fg / total <= 0.41):
        return None
    if current_fg >= total // 2:
        return None

    actual_by_color = {
        fg: _plus_centers(inp, fg)
        for fg in SOURCE_COLORS
    }
    for fg in SOURCE_COLORS:
        if sorted(actual_by_color[fg]) != sorted(allowed_centers[fg]):
            return None

    if not actual_by_color[YELLOW] or not actual_by_color[GREEN]:
        return None

    expected = _build_expected_output(inp)
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, inp):
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
