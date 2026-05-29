# Combo variant of e12f9a14 (index 31)
# combo_spec: D1/3+D3/3
# Name: 螺旋路径+距离递减涂色

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Original e12f9a14 launches straight diagonal/cardinal beams from frame gaps and spends most
#    of its logic on beam/beam collisions. This combo replaces the motion itself. Example:
#    for a top-side gap, the original would shoot outward in one fixed direction; here the gap
#    first paints itself, steps one cell outward, then grows a clockwise self-avoiding spiral.
#    The hidden spiral path is continuous, but the visible paint is distance-banded:
#    the first few cells are solid, the next band paints every other step, then every third step,
#    etc. The visible difference is therefore twofold: many right-angle turns, and increasing gaps
#    on outer rings.
# b) Complexity calibration:
#    To keep the rule legible in 2-4 examples, the generator only uses side gaps, not corner gaps.
#    That keeps the launch heading cardinal, so the spiral direction is easy to read by eye.
#    Multiple framed blocks still appear, but each frame emits exactly one spiral; this preserves
#    repetition without turning the output into an unreadable mesh.
# c) Hidden-assumption audit:
#    The original solver assumed gaps could launch collision-capable beams in any of the 8 outward
#    directions, and that later state depended on beam interactions. None of those assumptions hold
#    after switching to independent spirals. The new solver assumes:
#    - every generated launch gap is on a side cell of the 4x4 frame, so the outward heading is
#      cardinal;
#    - spirals are solved independently, with no hidden inter-spiral collision rules;
#    - self-avoidance matters for the full hidden path, while the visible sparse paint is derived
#      afterward from that path.
#    The generator explicitly enforces all of these assumptions.
# d) Placement legality:
#    Frames are placed with generous clearance, and the generator pre-traces every candidate spiral
#    on the obstacle grid before committing to a gap. It rejects short traces, low-turn traces, and
#    any pair of traces whose full paths overlap or touch orthogonally.
# e) Visual observability / cleanliness / inferability:
#    The human-visible signal is structural, not color-id based: each output trail still uses the
#    inner block color, but the distinguishing information is the repeated spiral turn pattern plus
#    the outer-band sparsification. The generator also rejects samples where all spirals face the
#    same way or where the decay is too weak, so the combo does not collapse back into near-straight
#    rays or dense orthogonal walls.


_DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_SIDE_GAPS = (
    (0, 1), (0, 2),
    (1, 3), (2, 3),
    (3, 2), (3, 1),
    (2, 0), (1, 0),
)


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))
    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        ok = True
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c < w
                and 0 <= r < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    out_set = set(out)
    if len(out_set) >= 3 and (fr1, fc1) in out_set and (fr0, bc + 1) in out_set:
        out = [p for p in out if p != (fr1, fc1)]
    return out


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS4[(_DIRS4.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS4[(_DIRS4.index(d) + 3) % 4]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(cell: tuple[int, int]):
    r, c = cell
    yield (r - 1, c)
    yield (r + 1, c)
    yield (r, c - 1)
    yield (r, c + 1)


def _trace_spiral_path(
    g: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    out_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    assert abs(out_dir[0]) + abs(out_dir[1]) == 1, "spiral variant expects side gaps only"

    start = (seed[0] + out_dir[0], seed[1] + out_dir[1])
    path = [seed]
    if not _in_bounds(g.shape, start) or int(g[start]) != bg:
        return path

    path.append(start)
    local = {seed, start}
    cur = start
    heading = _right(out_dir)

    def can_step(nxt: tuple[int, int], prev: tuple[int, int]) -> bool:
        if not _in_bounds(g.shape, nxt):
            return False
        if int(g[nxt]) != bg:
            return False
        if nxt in local:
            return False
        for nbr in _neighbors4(nxt):
            if nbr == prev:
                continue
            if nbr in local:
                return False
        return True

    guard = g.shape[0] * g.shape[1] * 4
    for _ in range(guard):
        moved = False
        for nd in (_right(heading), heading, _left(heading)):
            nxt = (cur[0] + nd[0], cur[1] + nd[1])
            if not can_step(nxt, cur):
                continue
            path.append(nxt)
            local.add(nxt)
            cur = nxt
            heading = nd
            moved = True
            break
        if not moved:
            break
    else:
        raise AssertionError("spiral trace did not converge")

    return path


def _decay_band(step_idx: int) -> tuple[int, int]:
    band = 1
    span = 4
    remaining = step_idx
    while remaining > span:
        remaining -= span
        band += 1
        span += 2
    return band, remaining


def _painted_cells_from_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    painted: list[tuple[int, int]] = []
    for i, cell in enumerate(path):
        if i == 0:
            painted.append(cell)
            continue
        band, offset = _decay_band(i)
        if (offset - 1) % band == 0:
            painted.append(cell)
    return painted


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_dir: tuple[int, int] | None = None
    for a, b in zip(path, path[1:]):
        d = (b[0] - a[0], b[1] - a[1])
        if prev_dir is not None and d != prev_dir:
            turns += 1
        prev_dir = d
    return turns


def _collect_launches(g: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    launches: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, br, bc in _find_blocks_2x2(g, bg):
        frame_col = _infer_frame_color(g, bg, br, bc, color)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        for gr, gc in gaps:
            out_dir = _gap_dir((gr - (br - 1), gc - (bc - 1)))
            if abs(out_dir[0]) + abs(out_dir[1]) != 1:
                continue
            launches.append((color, (gr, gc), out_dir))
    assert launches, "expected at least one side gap that can launch a spiral"
    return launches


def _analyze_input(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    info: list[dict[str, object]] = []
    for color, seed, out_dir in _collect_launches(g, bg):
        path = _trace_spiral_path(g, bg, seed, out_dir)
        painted = _painted_cells_from_path(path)
        info.append(
            {
                "color": color,
                "seed": seed,
                "out_dir": out_dir,
                "path": path,
                "painted": painted,
                "path_len": len(path),
                "painted_len": len(painted),
                "turns": _turn_count(path),
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch sparse colored spirals from framed 2x2 blocks through their side gaps.

    Input structure:
    - The background is the most common color.
    - Each object is a monochrome 2x2 block inside a shared-color 4x4 frame.
    - Generated instances use side gaps on the frame border as the spiral exits.

    Rule:
    - Each side gap paints its own cell, takes one step outward, then follows a clockwise
      self-avoiding spiral in the surrounding background.
    - The hidden spiral path is continuous, but the visible trail decays with distance:
      an inner band paints every step, the next band every other step, then every third step,
      and so on.
    - Spirals stop when the next move would leave the grid, enter a non-background cell,
      or touch an earlier cell of the same hidden path.

    Key insight:
    - The frame gap still determines where color escapes, but the output is no longer a straight
      beam. The solver must infer a full square spiral first, then sparsify outer rings according
      to distance from the source.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for color, seed, out_dir in _collect_launches(g, bg):
        path = _trace_spiral_path(g, bg, seed, out_dir)
        for r, c in _painted_cells_from_path(path):
            out[r, c] = color

    return out


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (2, 4)  # uniform draw [lo, hi)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 18, 3: 24}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 7
PLACEMENT_MARGIN = 2
MIN_PATH_LEN = 18
MIN_TURNS = 4
MIN_PAINTED = 10
MIN_TOTAL_ADDED = 24
GEN_RETRIES = 400


def _grid_size_min(block_count: int) -> int:
    result = 18
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    all_positions = [
        (r, c)
        for r in range(PLACEMENT_MARGIN, n - 5)
        for c in range(PLACEMENT_MARGIN, n - 5)
    ]
    rng.shuffle(all_positions)
    selected: list[tuple[int, int]] = []
    for p in all_positions:
        if all(_frames_separated(p, q) for q in selected):
            selected.append(p)
            if len(selected) == count:
                return sorted(selected)
    return None


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _path_compatible(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
) -> bool:
    aset = set(a)
    bset = set(b)
    if not aset.isdisjoint(bset):
        return False
    for cell in aset:
        for nbr in _neighbors4(cell):
            if nbr in bset:
                return False
    return True


def _candidate_launches(
    obstacles: np.ndarray,
    bg: int,
    block: tuple[int, int, int],
) -> list[dict[str, object]]:
    color, br, bc = block
    r0, c0 = br - 1, bc - 1
    candidates: list[dict[str, object]] = []
    for gap in _SIDE_GAPS:
        seed = (r0 + gap[0], c0 + gap[1])
        out_dir = _gap_dir(gap)
        path = _trace_spiral_path(obstacles, bg, seed, out_dir)
        painted = _painted_cells_from_path(path)
        turns = _turn_count(path)
        if len(path) < MIN_PATH_LEN:
            continue
        if len(painted) < MIN_PAINTED:
            continue
        if turns < MIN_TURNS:
            continue
        if len(painted) >= len(path) - 1:
            continue
        candidates.append(
            {
                "gap": gap,
                "seed": seed,
                "out_dir": out_dir,
                "path": path,
                "painted": painted,
                "path_len": len(path),
                "painted_len": len(painted),
                "turns": turns,
            }
        )
    return candidates


def _choose_gap_combo(
    candidate_lists: list[list[dict[str, object]]],
    rng: np.random.Generator,
) -> list[dict[str, object]] | None:
    all_combos: list[tuple[int, list[dict[str, object]]]] = []
    for combo in itertools.product(*candidate_lists):
        paths = [entry["path"] for entry in combo]
        ok = True
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if not _path_compatible(paths[i], paths[j]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        dirs = {entry["out_dir"] for entry in combo}
        if len(dirs) < 2:
            continue
        score = sum(int(entry["path_len"]) + 2 * int(entry["turns"]) for entry in combo)
        all_combos.append((score, list(combo)))
    if not all_combos:
        return None
    all_combos.sort(key=lambda x: x[0], reverse=True)
    top = all_combos[: min(12, len(all_combos))]
    return top[int(rng.integers(len(top)))][1]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)
    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    blocks = [(int(color), r0 + 1, c0 + 1) for (r0, c0), color in zip(positions, block_colors)]
    candidate_lists = [_candidate_launches(obstacles, bg, block) for block in blocks]
    if any(not cand for cand in candidate_lists):
        return None

    chosen = _choose_gap_combo(candidate_lists, rng)
    if chosen is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, choice in zip(positions, block_colors, chosen):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), (choice["gap"],))

    try:
        out = main(grid)
    except Exception:
        return None

    info = _analyze_input(grid)
    if len(info) != block_count:
        return None
    if sum(int(item["painted_len"]) - 1 for item in info) < MIN_TOTAL_ADDED:
        return None
    if any(int(item["turns"]) < MIN_TURNS for item in info):
        return None
    if any(int(item["painted_len"]) >= int(item["path_len"]) - 1 for item in info):
        return None
    if len({tuple(item["out_dir"]) for item in info}) < 2:
        return None
    if np.count_nonzero(out != grid) < MIN_TOTAL_ADDED:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo-variant e12f9a14 input")
