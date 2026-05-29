# Variant of e12f9a14
# Dimension: D11 — 框架数量与基数关系 (meta)
# Change: 多个框架（1-5个），每个框架独立发射一条射线，射线之间可能碰撞（多源独立发射模型）... → 链式传递——射线到达下一个框架后触发该框架发射

# Design analysis for the D11 chain-reaction variant
# ---------------------------------------------------
# concretization:
# - Keep the original visual vocabulary: square grid, one shared frame color, 2x2 monochrome inner blocks,
#   and frame-border gaps that define beam directions.
# - Change only the source model. There is now exactly one auto-start frame: the unique frame with 1 gap.
# - Every later frame has exactly 2 non-adjacent gaps:
#   one entry gap that faces the predecessor ray, and one exit gap that emits after activation.
# - A ray segment stops when it lands on the next frame's entry gap. That landing gap is painted in the
#   predecessor's color; the triggered frame then emits from its other gap in its own color.
#
# behavior_trace:
# - Original logic at the key step: after parsing all frames, every gap becomes an immediate beam seed.
#   If frame B has two gaps, both colors from B appear at once, regardless of whether another beam reaches B.
# - New logic at the key step: only the unique 1-gap source frame emits at time 0. Suppose frame A points to
#   frame B's west-facing gap. The A-colored ray paints from A's exit gap to B's entry gap and stops there.
#   Only then does frame B emit from its second gap in B's color toward frame C or the boundary.
# - Observable difference: middle frames no longer produce a backward/out-of-order beam. Their entry gap gets
#   the predecessor color, while their exit gap starts a new differently colored segment.
#
# complexity_calibration:
# - A completely free graph of frames would be too ambiguous from 2-3 examples, so the generator restricts the
#   structure to one directed chain. This keeps the new reasoning step (ordered activation) while avoiding
#   underconstrained graph search.
# - Chain length is 2-5, weighted toward 3-5 so the variant is usually visible.
# - Rays may use any of the 8 border directions supported by the original task, so the chain is not visually
#   trivial, but each frame still has only 1 or 2 gaps to keep inference crisp.
#
# implicit_assumptions_review:
# - The original solver assumes "every structural gap is an active seed" inside the frame-parsing loop.
#   That is invalid here, so seed creation is removed and replaced by source detection plus sequential tracing.
# - The original beam simulator treats every non-bg cell as a hard obstacle. That remains valid, but now a
#   bg gap on another frame is a legal stopping target rather than a place to pass through.
# - The new solver assumes exactly one 1-gap source frame, every other frame has exactly 2 gaps, and the chain
#   is acyclic. The generator explicitly constructs inputs that satisfy these conditions.
#
# placement_legality:
# - Frames still use 4x4 bounding boxes around 2x2 blocks and stay within the grid with a margin.
# - Consecutive frames are positioned so that an exit gap and the next entry gap lie on one clear ray with at
#   least a few background cells in between. Non-consecutive frames remain separated by >=2 cells.
# - The generator rejects layouts where any connector would cross another connector or any frame cell.
#
# visual_observability_and_tidiness:
# - The source frame is visually unique because it has only 1 gap; all other frames have 2. This is a visible,
#   color-independent cue for where the chain starts.
# - Output colors change exactly at frames, which makes the handoff easy to see: predecessor color fills the
#   incoming connector; current frame color fills the outgoing connector.
# - Intersections and heavy overdraw are rejected during generation, so the final grids remain readable.
#
# color_inference:
# - No color id is used for sorting or ordering. Order comes only from visible geometry: gap count and gap
#   alignment. Colors only label which frame emitted which segment, matching the original task's palette role.
#
# solver_generator_contract:
# - main() assumes: one source frame, all other frames are 2-gap relays, every emitted segment either reaches
#   the next frame's gap or leaves the grid, and relay segments do not overlap previous ones.
# - generate_input() enforces exactly those conditions and validates each sample by running main().

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}
_DIR_TO_LOCAL_GAPS = {
    (-1, -1): [(0, 0)],
    (-1, 0): [(0, 1), (0, 2)],
    (-1, 1): [(0, 3)],
    (0, 1): [(1, 3), (2, 3)],
    (1, 1): [(3, 3)],
    (1, 0): [(3, 2), (3, 1)],
    (1, -1): [(3, 0)],
    (0, -1): [(2, 0), (1, 0)],
}


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
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    return out


def _gap_dir(local_gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = local_gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return -d[0], -d[1]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _scale(d: tuple[int, int], k: int) -> tuple[int, int]:
    return d[0] * k, d[1] * k


def _frame_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(4) for dc in range(4)}


def _frames_separated(a: tuple[int, int], b: tuple[int, int], gap: int) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _non_adjacent_local(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ia = _BORDER_INDEX[a]
    ib = _BORDER_INDEX[b]
    d = (ib - ia) % len(_BORDER_ORDER)
    return d not in (1, len(_BORDER_ORDER) - 1)


_PAIR_OPTIONS: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] = {}
for din, in_gaps in _DIR_TO_LOCAL_GAPS.items():
    for dout, out_gaps in _DIR_TO_LOCAL_GAPS.items():
        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for gin in in_gaps:
            for gout in out_gaps:
                if gin != gout and _non_adjacent_local(gin, gout):
                    pairs.append((gin, gout))
        _PAIR_OPTIONS[(din, dout)] = pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay colored rays between framed 2x2 blocks; only the source frame fires immediately.

    Input structure:
    - bg is the most common color. Each colored 2x2 block sits inside a shared-color 4x4 frame.
    - Exactly one frame has a single gap; every other frame has two non-adjacent gaps.
    - The 1-gap frame is the source, while each 2-gap frame has one entry gap facing a predecessor and one exit gap.

    Rule:
    - Start from the unique 1-gap frame and paint a ray from that gap in the gap's outward direction.
    - When the ray lands on another frame's gap, stop that segment there and trigger the reached frame.
    - The triggered frame emits from its other gap in its own block color; continue the chain until the last ray leaves the grid.

    Key insight:
    - Gaps no longer act as simultaneous independent emitters. They encode a directed handoff chain: incoming color marks
      the reached gap, while the reached frame's own color starts the next segment from its remaining gap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    blocks = _find_blocks_2x2(g, bg)
    assert len(blocks) >= 2, "expected a visible chain with at least two frames"

    frames: list[dict[str, object]] = []
    gap_to_frame: dict[tuple[int, int], tuple[int, int]] = {}

    for color, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, color)
        gaps = sorted(_frame_gaps_for_block(g, bg, frame_col, br, bc))
        assert 1 <= len(gaps) <= 2, "variant expects one source gap or one entry/one exit"
        r0, c0 = br - 1, bc - 1
        local_gaps = [(gr - r0, gc - c0) for gr, gc in gaps]
        dirs = [_gap_dir(local_gap) for local_gap in local_gaps]
        frames.append(
            {
                "color": int(color),
                "pos": (r0, c0),
                "gaps": gaps,
                "dirs": dirs,
            }
        )
        for gi, gap in enumerate(gaps):
            assert gap not in gap_to_frame, "gap positions must be unique"
            gap_to_frame[gap] = (len(frames) - 1, gi)

    source_ids = [fi for fi, frame in enumerate(frames) if len(frame["gaps"]) == 1]
    assert len(source_ids) == 1, "expected exactly one auto-start frame"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def trace_segment(frame_id: int, gap_id: int) -> tuple[int, int] | None:
        frame = frames[frame_id]
        color = int(frame["color"])
        start = tuple(frame["gaps"][gap_id])
        dr, dc = tuple(frame["dirs"][gap_id])

        if out[start[0], start[1]] == bg:
            out[start[0], start[1]] = color

        cur = start
        while True:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not in_bounds(nxt):
                return None
            owner = gap_to_frame.get(nxt)
            if owner is not None and owner[0] != frame_id:
                assert out[nxt[0], nxt[1]] == bg, "relay entry gaps should be untouched before activation"
                out[nxt[0], nxt[1]] = color
                return owner
            assert int(g[nxt[0], nxt[1]]) == bg, "connector must pass only through background"
            assert int(out[nxt[0], nxt[1]]) == bg, "chain connectors must not overlap"
            out[nxt[0], nxt[1]] = color
            cur = nxt

    activated: set[int] = set()
    current_frame = source_ids[0]
    current_gap = 0

    while True:
        assert current_frame not in activated, "chain must be acyclic"
        activated.add(current_frame)
        hit = trace_segment(current_frame, current_gap)
        if hit is None:
            break
        next_frame, entry_gap = hit
        assert len(frames[next_frame]["gaps"]) == 2, "every non-source frame must be a 2-gap relay"
        current_frame = next_frame
        current_gap = 1 - entry_gap

    assert len(activated) == len(frames), "every frame should participate in the chain"
    return out


# --------------- generator (d11_chain_relay) ---------------

BLOCK_COUNT_CHOICES = [2, 3, 4, 5]
BLOCK_COUNT_WEIGHTS = [1, 3, 3, 2]
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 3: 18, 4: 22, 5: 25}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
PLACEMENT_MARGIN = 2
FRAME_BBOX_GAP = 2
MIN_LINK_SPAN = 4
MAX_LINK_SPAN = 8
MIN_TAIL_CELLS = 3
GEN_RETRIES = 500


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


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 14
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    cells = [start]
    cur = start
    for _ in range(steps):
        cur = (cur[0] + direction[0], cur[1] + direction[1])
        cells.append(cur)
    return cells


def _trace_to_edge(start: tuple[int, int], direction: tuple[int, int], n: int) -> list[tuple[int, int]]:
    cells = [start]
    cur = start
    while True:
        nxt = (cur[0] + direction[0], cur[1] + direction[1])
        if not (0 <= nxt[0] < n and 0 <= nxt[1] < n):
            return cells
        cells.append(nxt)
        cur = nxt


def _choose_directions(count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    dirs = list(_DIRS8)
    out_dirs = [dirs[int(rng.integers(len(dirs)))]]
    for i in range(1, count):
        entry_dir = _opp(out_dirs[-1])
        choices = [d for d in dirs if d != entry_dir and _PAIR_OPTIONS[(entry_dir, d)]]
        if not choices:
            return None
        out_dirs.append(choices[int(rng.integers(len(choices)))])
    return out_dirs


def _build_relative_chain(rng: np.random.Generator, count: int) -> list[dict[str, object]] | None:
    out_dirs = _choose_directions(count, rng)
    if out_dirs is None:
        return None

    frames: list[dict[str, object]] = []
    first_out = tuple(_DIR_TO_LOCAL_GAPS[out_dirs[0]][int(rng.integers(len(_DIR_TO_LOCAL_GAPS[out_dirs[0]])))])
    frames.append({"entry": None, "exit": first_out, "out_dir": out_dirs[0], "pos": (0, 0)})

    for i in range(1, count):
        entry_dir = _opp(out_dirs[i - 1])
        exit_dir = out_dirs[i]
        pairs = _PAIR_OPTIONS[(entry_dir, exit_dir)]
        if not pairs:
            return None
        entry_gap, exit_gap = pairs[int(rng.integers(len(pairs)))]
        span = int(rng.integers(MIN_LINK_SPAN, MAX_LINK_SPAN + 1))
        prev = frames[-1]
        prev_exit = _add(tuple(prev["pos"]), tuple(prev["exit"]))
        next_pos = _sub(_add(prev_exit, _scale(out_dirs[i - 1], span)), entry_gap)
        frames.append({"entry": entry_gap, "exit": exit_gap, "out_dir": exit_dir, "pos": next_pos})

    positions = [tuple(frame["pos"]) for frame in frames]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if not _frames_separated(positions[i], positions[j], FRAME_BBOX_GAP):
                return None
    return frames


def _frame_fit_ranges(positions: list[tuple[int, int]], n: int) -> tuple[range, range] | None:
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    sr_lo = PLACEMENT_MARGIN - min(rows)
    sr_hi = n - 4 - PLACEMENT_MARGIN - max(rows)
    sc_lo = PLACEMENT_MARGIN - min(cols)
    sc_hi = n - 4 - PLACEMENT_MARGIN - max(cols)
    if sr_lo > sr_hi or sc_lo > sc_hi:
        return None
    return range(sr_lo, sr_hi + 1), range(sc_lo, sc_hi + 1)


def _realize_chain(
    frames: list[dict[str, object]],
    block_colors: list[int],
    bg: int,
    frame_color: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    positions = [tuple(frame["pos"]) for frame in frames]
    shift_ranges = _frame_fit_ranges(positions, n)
    if shift_ranges is None:
        return None
    row_range, col_range = shift_ranges
    shifts = [(sr, sc) for sr in row_range for sc in col_range]
    if not shifts:
        return None
    rng.shuffle(shifts)

    for shift in shifts[:40]:
        placed = [{**frame, "pos": _add(tuple(frame["pos"]), shift)} for frame in frames]
        grid = np.full((n, n), bg, dtype=int)
        frame_cells: set[tuple[int, int]] = set()
        gap_cells: set[tuple[int, int]] = set()
        gap_map: dict[tuple[int, int], int] = {}

        for frame, color in zip(placed, block_colors):
            r0, c0 = tuple(frame["pos"])
            entry_gap = frame["entry"]
            gaps = [tuple(frame["exit"])]
            if entry_gap is not None:
                gaps.append(tuple(entry_gap))
            _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), tuple(gaps))
            local_gap_cells = {_add((r0, c0), gap) for gap in gaps}
            gap_cells |= local_gap_cells
            for cell in local_gap_cells:
                gap_map[cell] = gap_map.get(cell, 0) + 1
            frame_cells |= _frame_cells(r0, c0)

        if any(v != 1 for v in gap_map.values()):
            continue

        used_cells: set[tuple[int, int]] = set()
        ok = True

        for i in range(len(placed) - 1):
            start = _add(tuple(placed[i]["pos"]), tuple(placed[i]["exit"]))
            target = _add(tuple(placed[i + 1]["pos"]), tuple(placed[i + 1]["entry"]))
            direction = tuple(placed[i]["out_dir"])
            diff = _sub(target, start)
            if direction[0] == 0:
                if diff[0] != 0 or diff[1] == 0 or diff[1] % direction[1] != 0:
                    ok = False
                    break
                steps = diff[1] // direction[1]
            elif direction[1] == 0:
                if diff[1] != 0 or diff[0] == 0 or diff[0] % direction[0] != 0:
                    ok = False
                    break
                steps = diff[0] // direction[0]
            else:
                if diff[0] == 0 or diff[1] == 0 or diff[0] != diff[1] * direction[0] // direction[1]:
                    ok = False
                    break
                if diff[0] % direction[0] != 0 or diff[1] % direction[1] != 0:
                    ok = False
                    break
                steps = diff[0] // direction[0]
            if steps < MIN_LINK_SPAN:
                ok = False
                break
            cells = _segment_cells(start, direction, steps)
            if cells[-1] != target:
                ok = False
                break
            for j, cell in enumerate(cells):
                if not (0 <= cell[0] < n and 0 <= cell[1] < n):
                    ok = False
                    break
                if j == 0:
                    if cell in used_cells:
                        ok = False
                        break
                    continue
                if j == len(cells) - 1:
                    if cell in used_cells:
                        ok = False
                        break
                    if cell not in gap_cells:
                        ok = False
                        break
                    continue
                if cell in used_cells or cell in frame_cells:
                    ok = False
                    break
            if not ok:
                break
            used_cells.update(cells)

        if not ok:
            continue

        last_start = _add(tuple(placed[-1]["pos"]), tuple(placed[-1]["exit"]))
        tail_cells = _trace_to_edge(last_start, tuple(placed[-1]["out_dir"]), n)
        if len(tail_cells) - 1 < MIN_TAIL_CELLS:
            continue
        for j, cell in enumerate(tail_cells):
            if j == 0:
                if cell in used_cells:
                    ok = False
                    break
                continue
            if cell in used_cells or cell in frame_cells:
                ok = False
                break
        if not ok:
            continue
        used_cells.update(tail_cells)
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    block_count = int(rng.choice(BLOCK_COUNT_CHOICES, p=np.array(BLOCK_COUNT_WEIGHTS) / sum(BLOCK_COUNT_WEIGHTS)))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    relative = _build_relative_chain(rng, block_count)
    if relative is None:
        return None

    grid = _realize_chain(relative, block_colors, bg, frame_color, n, rng)
    if grid is None:
        return None

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < block_count * 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D11 input")
