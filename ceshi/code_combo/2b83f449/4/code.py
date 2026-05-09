# Combo variant of 2b83f449 (index 4)
# combo_spec: D1/2+D9/0
# Name: 条件偏移旋转+逐门处理

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask

# Design analysis for combo D1/2 + D9/0
# a) Concrete behavior and trace:
#    - Original gate centered at columns [c-1,c,c+1] opens into a centered 3x1 post at c.
#      A token only needs the final fully-open grid; all gates open before any packing.
#    - Fused rule:
#      1. If gate center is on the left half, rotate clockwise: the 3x1 post appears at c+1.
#         The wall row becomes open at c-1 and c.
#      2. If gate center is on the right half, rotate counterclockwise: the post appears at c-1.
#         The wall row becomes open at c and c+1.
#      3. Gates are processed one by one from top to bottom. After each single gate opens,
#         all tokens fall as far as they currently can and then immediately repack to segment ends.
#      Example: width=17, a left gate centered at c=5 opens to post=6, then a token that was
#      packed at row 2, col 4 may stay on the left child segment when a later post appears below;
#      in the all-open version that same token would have entered the deeper row before the split.
#      This is the intended D9 interaction with the D1 offset topology.
# b) Complexity calibration:
#    - Only two rotation outcomes exist: left-half gates shift right, right-half gates shift left.
#      The mapping is repeated across every sample, so humans get multiple left/right examples.
#    - We intentionally keep one gate per wall row in generated samples. That removes same-row
#      order ambiguity while still giving 6-8 repeated gate events, enough to expose the rule.
# c) Broken original assumptions:
#    - The original solver assumed each pipe segment has <=1 exit and <=1 entry. That is no
#      longer safe: an offset post naturally leaves two adjacent wall openings on one side.
#      Therefore this combo solver explicitly enumerates exit columns and chooses the extreme one
#      for the token side (left token -> leftmost exit, right token -> rightmost exit).
#    - The original "pack only once at the very end" assumption also breaks the D9 change.
#      Here stagewise packing is part of the state update after every opened gate.
# d) Placement legality:
#    - Generated bars stay strictly inside the wall row (center in 3..W-4), so both shifted posts
#      remain inside 1..W-2.
#    - No holes are used; this keeps the geometry clean and avoids accidental over-constraining.
#    - Samples are rejected if a newly opened post would land on an already packed token.
# e) Visual observability:
#    - Left-half bars and right-half bars visibly open to different offset columns in the output.
#    - Stagewise behavior is forced to matter: the generator rejects any sample where the fused
#      output equals either the "all gates open at once with shifted posts" control or the
#      "progressive opening but centered posts" control.
# f) Visual cleanliness:
#    - One gate per wall row and no extra hole noise keeps each row readable.
#    - Tokens remain color-3, gates color-7, opened posts color-6, background color-8.
# g) New solver assumptions and generator guarantees:
#    - `main()` assumes every token starts on a pipe row and every gate is a 1x3 horizontal bar.
#      The generator constructs exactly that.
#    - `main()` also assumes the left/right classification comes from visible geometric half-plane,
#      not from color or hidden metadata. The generator avoids midpoint bars so the class is clear.

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _infer_gate_color(grid: np.ndarray) -> int:
    colors = sorted(set(map(int, grid.flatten())))
    for color in colors:
        if color == WALL:
            continue
        labels, n = label_mask(grid == color, connectivity=4)
        for k in range(1, int(n) + 1):
            pts = np.argwhere(labels == k)
            if int(pts.shape[0]) != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                return int(color)
    raise AssertionError("gate color not found")


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    gate = _infer_gate_color(grid)
    colors = set(map(int, grid.flatten()))
    counts = Counter(map(int, grid.flatten()))
    token_candidates = [c for c in colors if c not in (WALL, gate)]
    if not token_candidates:
        raise AssertionError("token color not found")
    token = int(min(token_candidates, key=lambda c: (int(counts[c]), int(c))))
    bg_candidates = [c for c in colors if c not in (WALL, gate, token)]
    if not bg_candidates:
        raise AssertionError("bg color not found")
    bg = int(max(bg_candidates, key=lambda c: (int(counts[c]), -int(c))))
    open_col = OPEN if gate == GATE else int(gate - 1) if gate > 0 else OPEN
    return gate, token, bg, open_col


def _gate_direction(center: int, width: int) -> int:
    return 1 if (2 * center) < (width - 1) else -1


def _post_col(center: int, width: int, shifted: bool) -> int:
    return int(center + (_gate_direction(center, width) if shifted else 0))


def _parse_gates(grid: np.ndarray, gate: int) -> list[dict[str, int]]:
    labels, n = label_mask(grid == int(gate), connectivity=4)
    gates: list[dict[str, int]] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(labels == int(k))
        if int(pts.shape[0]) != 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        gates.append(
            {
                "wr": int(r0),
                "wall_idx": int((int(r0) - 1) // 2),
                "c0": int(c0),
                "center": int(c0 + 1),
            }
        )
    gates.sort(key=lambda g: (int(g["wall_idx"]), int(g["center"])))
    return gates


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
        segs.append((int(start), int(c)))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exit_columns(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [
        int(c)
        for c in range(a, b + 1)
        if int(grid[r + 1, c]) == int(bg) and int(grid[r + 2, c]) == int(bg)
    ]


def _choose_exit(exits: list[int], side: str, current_c: int) -> int | None:
    if not exits:
        return None
    if side == "left":
        return int(min(exits))
    if side == "right":
        return int(max(exits))
    return int(min(exits, key=lambda c: (abs(int(c) - int(current_c)), int(c))))


def _pack_end(entry: int | None, seg: tuple[int, int], side: str, current_c: int) -> str:
    a, b = seg
    if entry is None:
        if side == "left":
            return "left"
        if side == "right":
            return "right"
        return "left" if current_c <= (a + b) // 2 else "right"
    return "right" if (entry - a) <= (b - entry) else "left"


def _snap_to_bg(grid: np.ndarray, r: int, c: int, bg: int, side: str) -> int | None:
    width = grid.shape[1]
    if int(grid[r, c]) == int(bg):
        return int(c)
    left = [cc for cc in range(int(c), -1, -1) if int(grid[r, cc]) == int(bg)]
    right = [cc for cc in range(int(c), width) if int(grid[r, cc]) == int(bg)]
    ordered: list[int]
    if side == "left":
        ordered = left + right
    elif side == "right":
        ordered = right + left
    else:
        all_bg = [cc for cc in range(width) if int(grid[r, cc]) == int(bg)]
        ordered = sorted(all_bg, key=lambda cc: (abs(int(cc) - int(c)), int(cc)))
    return int(ordered[0]) if ordered else None


def _render_states(base: np.ndarray, states: list[dict[str, int]], token: int) -> np.ndarray:
    out = base.copy()
    for state in states:
        out[int(state["r"]), int(state["c"])] = int(token)
    return out


def _pack_states(
    base: np.ndarray,
    landed: list[dict[str, int | str | None]],
    bg: int,
) -> list[dict[str, int | str]]:
    buckets: dict[tuple[int, tuple[int, int]], list[dict[str, int | str | None]]] = defaultdict(list)
    for state in landed:
        seg = _find_segment(base, int(state["r"]), int(state["c"]), bg)
        if seg is None:
            seg = (int(state["c"]), int(state["c"]))
        buckets[(int(state["r"]), seg)].append({**state, "seg": seg})

    packed: list[dict[str, int | str]] = []
    for (r, seg), recs in buckets.items():
        a, b = seg
        used: set[int] = set()
        for rec in sorted(recs, key=lambda item: int(item["id"])):
            pref = _pack_end(
                int(rec["entry"]) if rec["entry"] is not None else None,
                seg,
                str(rec["side"]),
                int(rec["c"]),
            )
            candidates = range(a, b + 1) if pref == "left" else range(b, a - 1, -1)
            placed = None
            for cc in candidates:
                if cc not in used:
                    used.add(int(cc))
                    placed = int(cc)
                    break
            if placed is None:
                continue
            packed.append(
                {
                    "id": int(rec["id"]),
                    "r": int(r),
                    "c": int(placed),
                    "side": str(rec["side"]),
                    "source_r": int(rec["source_r"]),
                }
            )
    packed.sort(key=lambda item: int(item["id"]))
    return packed


def _settle_once(
    base: np.ndarray,
    states: list[dict[str, int | str]],
    bg: int,
    strict: bool,
) -> tuple[list[dict[str, int | str]], dict[int, bool]] | tuple[None, None]:
    landed: list[dict[str, int | str | None]] = []
    moved_flags: dict[int, bool] = {}
    for state in states:
        r = int(state["r"])
        c = int(state["c"])
        side = str(state["side"])
        snapped = _snap_to_bg(base, r, c, bg, side)
        if snapped is None:
            if strict:
                return None, None
            snapped = c
        if strict and snapped != c and int(base[r, c]) != int(bg):
            return None, None
        c = int(snapped)
        last_entry: int | None = None
        moved = False
        while True:
            seg = _find_segment(base, r, c, bg)
            if seg is None:
                break
            exits = _exit_columns(base, r, seg, bg)
            exit_c = _choose_exit(exits, side, c)
            if exit_c is None:
                break
            r += 2
            c = int(exit_c)
            last_entry = int(exit_c)
            moved = True
        landed.append(
            {
                "id": int(state["id"]),
                "r": int(r),
                "c": int(c),
                "side": side,
                "entry": last_entry,
                "source_r": int(state["source_r"]),
            }
        )
        moved_flags[int(state["id"])] = moved or (int(state["r"]) != int(r)) or (int(state["c"]) != int(c))
    packed = _pack_states(base, landed, bg)
    return packed, moved_flags


def _open_gate(base: np.ndarray, gate: dict[str, int], bg: int, open_col: int, shifted: bool) -> None:
    wr = int(gate["wr"])
    c0 = int(gate["c0"])
    center = int(gate["center"])
    post = _post_col(center, base.shape[1], shifted)
    base[wr, c0 : c0 + 3] = int(bg)
    for rr in (wr - 1, wr, wr + 1):
        if 0 <= rr < base.shape[0]:
            base[rr, post] = int(open_col)


def _initial_states(grid: np.ndarray, token: int) -> list[dict[str, int | str]]:
    states: list[dict[str, int | str]] = []
    idx = 0
    for r, c in np.argwhere(grid == int(token)):
        side = "left" if int(c) <= 1 else "right" if int(c) >= grid.shape[1] - 2 else "internal"
        states.append(
            {
                "id": int(idx),
                "r": int(r),
                "c": int(c),
                "side": side,
                "source_r": int(r),
            }
        )
        idx += 1
    states.sort(key=lambda item: (int(item["source_r"]), int(item["c"]), int(item["id"])))
    for idx, state in enumerate(states):
        state["id"] = int(idx)
    return states


def _solve_model(
    grid: np.ndarray,
    progressive: bool,
    shifted: bool,
    strict: bool = False,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int)
    gate, token, bg, open_col = _infer_colors(g)
    gates = _parse_gates(g, gate)
    base = g.copy()
    base[base == int(token)] = int(bg)
    states = _initial_states(g, token)
    move_counts = {int(state["id"]): 0 for state in states}

    if progressive:
        for gate_info in gates:
            _open_gate(base, gate_info, bg, open_col, shifted)
            settled, moved_flags = _settle_once(base, states, bg, strict)
            if settled is None:
                raise AssertionError("invalid progressive state")
            states = settled
            for token_id, moved in moved_flags.items():
                if moved:
                    move_counts[int(token_id)] += 1
    else:
        for gate_info in gates:
            _open_gate(base, gate_info, bg, open_col, shifted)
        settled, moved_flags = _settle_once(base, states, bg, strict)
        if settled is None:
            raise AssertionError("invalid static state")
        states = settled
        for token_id, moved in moved_flags.items():
            if moved:
                move_counts[int(token_id)] += 1

    out = _render_states(base, states, token)
    if not return_meta:
        return out
    meta = {
        "late_movers": int(sum(1 for cnt in move_counts.values() if int(cnt) >= 2)),
        "token_count": int(len(states)),
        "gate_count": int(len(gates)),
        "left_gates": int(sum(1 for info in gates if _gate_direction(int(info["center"]), g.shape[1]) > 0)),
        "right_gates": int(sum(1 for info in gates if _gate_direction(int(info["center"]), g.shape[1]) < 0)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_model(np.array(grid, dtype=int), progressive=True, shifted=True)


def _build_input(centers: list[int], width: int) -> np.ndarray:
    n_walls = len(centers)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        grid[2 * pipe_idx, :] = BG
    for wall_idx, center in enumerate(centers):
        wr = 2 * wall_idx + 1
        grid[wr, center - 1 : center + 2] = GATE
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN
    return grid


def _valid_center_pool(width: int) -> tuple[list[int], list[int]]:
    mid = (width - 1) / 2.0
    left = [c for c in range(3, width - 3) if c < mid - 0.75]
    right = [c for c in range(3, width - 3) if c > mid + 0.75]
    return left, right


def _make_side_pattern(rng: np.random.Generator, n_walls: int) -> list[str] | None:
    for _ in range(60):
        pattern = [str(rng.choice(["L", "R"])) for _ in range(n_walls)]
        if pattern.count("L") < 2 or pattern.count("R") < 2:
            continue
        switches = sum(1 for i in range(1, n_walls) if pattern[i] != pattern[i - 1])
        if switches < 2:
            continue
        return pattern
    return None


def _sample_centers(rng: np.random.Generator, width: int, n_walls: int) -> list[int] | None:
    left_pool, right_pool = _valid_center_pool(width)
    if not left_pool or not right_pool:
        return None
    pattern = _make_side_pattern(rng, n_walls)
    if pattern is None:
        return None

    centers: list[int] = []
    prev: int | None = None
    for side in pattern:
        pool = left_pool if side == "L" else right_pool
        if prev is None:
            center = int(rng.choice(pool))
        else:
            near = [c for c in pool if 1 <= abs(int(c) - int(prev)) <= 4]
            center = int(rng.choice(near if near else pool))
        centers.append(center)
        prev = center
    return centers


def _control_deltas(inp: np.ndarray) -> tuple[int, int, dict[str, int]]:
    combo, meta = _solve_model(inp, progressive=True, shifted=True, strict=True, return_meta=True)
    shifted_static = _solve_model(inp, progressive=False, shifted=True, strict=True)
    centered_progressive = _solve_model(inp, progressive=True, shifted=False, strict=True)
    delta_static = int(np.count_nonzero(combo != shifted_static))
    delta_centered = int(np.count_nonzero(combo != centered_progressive))
    return delta_static, delta_centered, meta


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    width = int(rng.choice([17, 19]))
    n_walls = int(rng.integers(6, 9))
    centers = _sample_centers(rng, width, n_walls)
    if centers is None:
        return None

    inp = _build_input(centers, width)
    try:
        combo_out = main(inp)
        if int(np.count_nonzero(combo_out == TOKEN)) != 2 * n_walls:
            return None
        delta_static, delta_centered, meta = _control_deltas(inp)
    except Exception:
        return None

    if meta["left_gates"] < 2 or meta["right_gates"] < 2:
        return None
    if meta["late_movers"] < 2:
        return None
    if delta_static <= 0 or delta_centered <= 0:
        return None
    if delta_static + delta_centered < 6:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo 2b83f449 D1/2+D9/0")
