# Combo variant of 2b83f449 (index 14)
# combo_spec: D4/1+D7/0
# Name: 到达顺序排列+多出口

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask

"""
Design analysis for the fused variant.

1. Concrete fused behavior on the pixel grid
   - Geometry remains the seed task's pipe-and-wall lattice: even rows are pipe
     rows, odd rows are wall rows, input gates are horizontal 1x3 bars, and
     output gates become vertical 3-cell posts.
   - D7/0 changes routing: some segments have two downward exits. Tokens remain
     deterministic because each token keeps its origin side. Left-edge tokens
     choose the leftmost available exit; right-edge tokens choose the rightmost.
   - D4/1 changes settling: when multiple tokens stop in the same destination
     segment, they are written from that segment's left edge toward the right in
     arrival order, not packed toward a far endpoint.
   - Traceable mini-example:
       * A right-edge token can reach a segment whose exits are [4, 7] and take
         exit 7 because it came from the right.
       * Another token can reach the same final segment by a shorter route and
         therefore arrive earlier.
       * In the seed task both tokens would far-pack on the same side. In this
         combo they occupy the leftmost cells of the destination segment in
         increasing arrival-time order.

2. Complexity calibration / observability
   - The generator keeps colors unchanged from the seed task and introduces only
     one new structural degree of freedom: exactly one real two-exit gadget.
   - The generator rejects samples where the combo output matches the old
     far-pack behavior, so the D4 part stays visually observable.
   - It also rejects samples unless at least one merged destination bucket mixes
     path lengths and includes a token that actually traversed a two-exit
     segment. That makes the fusion behavior real rather than latent.

3. Original solver assumptions that break here
   - "Each segment has <=1 exit" is false now, so the solver must keep the full
     exit list and select by source side instead of asserting.
   - "Packing can be decided from a unique entry above" is no longer central.
     The combo packs by arrival order, so merged multi-entry segments are fine.
   - The original packing bucket key used a destination endpoint; the combo
     buckets by final segment only and lays tokens left-to-right.

4. Placement legality
   - Gate centers stay interior and same-row centers stay at least 5 columns
     apart. The generator constructs 5-7 wall rows and 16-18 columns, which is
     consistent with the seed task's scale.
   - No holes are used, which keeps the picture clean and avoids introducing an
     unrelated extra cue.

5. Visual cleanliness / human learnability
   - Input colors remain wall 0, token 3, gate 7, background 8.
   - Output colors remain wall 0, token 3, opened post 6, background 8.
   - Across examples humans repeatedly see one wall row with two gates, side-
     dependent routing there, and merged destination segments filling from the
     left instead of from a far endpoint.

6. New solver assumptions introduced by the combo
   - Tokens are still edge sources only. The generator preserves that.
   - Tie-breaking between equal-arrival tokens is deterministic: lower source
     rows first, then input scan order.
   - Traversed segments are filtered so no token ever faces >2 exits. The
     generator explicitly enforces that.
"""

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _infer_gate_token_bg(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))

    gate = None
    for col in sorted(colors):
        if col == WALL:
            continue
        lab, n = label_mask(g == col, connectivity=4)
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == k)
            if int(pts.shape[0]) != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                gate = int(col)
                break
        if gate is not None:
            break
    assert gate is not None

    counts = Counter(map(int, g.flatten()))
    token_candidates = [c for c in colors if c not in (WALL, gate)]
    assert token_candidates
    token = int(min(token_candidates, key=lambda c: (int(counts[int(c)]), int(c))))

    bg_candidates = [c for c in colors if c not in (WALL, gate, token)]
    assert bg_candidates
    bg = int(max(bg_candidates, key=lambda c: (int(counts[int(c)]), -int(c))))
    return int(gate), int(token), int(bg)


def _open_gates(grid: np.ndarray, gate: int, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    colors = set(map(int, out.flatten()))
    open_col = OPEN if gate == GATE and OPEN not in colors else (gate - 1 if gate > 0 else OPEN)

    lab, n = label_mask(out == int(gate), connectivity=4)
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        if int(pts.shape[0]) != 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        r = int(r0)
        left = int(c0)
        center = int(left + 1)
        out[r, left : left + 3] = int(bg)
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < out.shape[0]:
                out[rr, center] = int(open_col)
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
        segs.append((int(start), int(c)))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [
        int(c)
        for c in range(a, b + 1)
        if int(grid[r + 1, c]) == int(bg) and int(grid[r + 2, c]) == int(bg)
    ]


def _entry_from_above(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> int | None:
    if r < 2:
        return None
    a, b = seg
    entries = [
        int(c)
        for c in range(a, b + 1)
        if int(grid[r - 1, c]) == int(bg) and int(grid[r - 2, c]) == int(bg)
    ]
    if len(entries) == 1:
        return int(entries[0])
    return None


def _source_side(c: int, width: int) -> str:
    if c <= 1:
        return "left"
    if c >= width - 2:
        return "right"
    return "left" if c < width // 2 else "right"


def _parse_sources(grid: np.ndarray, token: int) -> list[dict[str, int | str]]:
    sources: list[dict[str, int | str]] = []
    width = grid.shape[1]
    order = 0
    for pipe_idx, r in enumerate(range(0, grid.shape[0], 2)):
        for c in range(width):
            if int(grid[r, c]) != int(token):
                continue
            sources.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "pipe_idx": int(pipe_idx),
                    "side": _source_side(int(c), width),
                    "order": int(order),
                }
            )
            order += 1
    return sources


def _choose_exit(exits: list[int], side: str) -> int | None:
    if not exits:
        return None
    if side == "right":
        return int(max(exits))
    return int(min(exits))


def _simulate_records(opened: np.ndarray, sources: list[dict[str, int | str]], bg: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source in sources:
        r = int(source["r"])
        c = int(source["c"])
        depth = 0
        used_multi_exit = False
        max_exit_arity = 0

        while True:
            seg = _find_segment(opened, r, c, bg)
            if seg is None:
                break
            exits = _exits_below(opened, r, seg, bg)
            max_exit_arity = max(max_exit_arity, len(exits))
            if len(exits) >= 2:
                used_multi_exit = True
            exit_col = _choose_exit(exits, str(source["side"]))
            if exit_col is None:
                break
            r += 2
            c = int(exit_col)
            depth += 1
            if r >= opened.shape[0]:
                break

        final_seg = _find_segment(opened, r, c, bg)
        if final_seg is None:
            final_seg = (int(c), int(c))
        records.append(
            {
                "r": int(r),
                "c": int(c),
                "depth": int(depth),
                "seg": final_seg,
                "used_multi_exit": bool(used_multi_exit),
                "max_exit_arity": int(max_exit_arity),
                "source": source,
            }
        )
    return records


def _arrival_key(rec: dict[str, object]) -> tuple[int, int, int]:
    source = rec["source"]
    assert isinstance(source, dict)
    return (int(rec["depth"]), -int(source["pipe_idx"]), int(source["order"]))


def _render_combo_output(opened: np.ndarray, records: list[dict[str, object]], token: int) -> np.ndarray:
    out = np.array(opened, dtype=int, copy=True)
    buckets: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        a, b = tuple(rec["seg"])
        buckets[(int(rec["r"]), int(a), int(b))].append(rec)

    for (r, a, b), bucket in buckets.items():
        ordered = sorted(bucket, key=_arrival_key)
        for idx, _rec in enumerate(ordered):
            c = a + idx
            if c <= b:
                out[r, c] = int(token)
    return out


def _render_far_pack_output(opened: np.ndarray, records: list[dict[str, object]], token: int, bg: int) -> np.ndarray:
    out = np.array(opened, dtype=int, copy=True)
    buckets: dict[tuple[int, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        r = int(rec["r"])
        c = int(rec["c"])
        a, b = tuple(rec["seg"])
        entry = _entry_from_above(opened, r, (a, b), bg)
        if entry is not None:
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if c <= a else b
        buckets[(r, a, b, int(pack_end))].append(rec)

    for (r, a, b, pack_end), bucket in buckets.items():
        ordered = sorted(bucket, key=lambda rec: int(rec["source"]["order"]))  # type: ignore[index]
        if pack_end == a:
            for idx, _rec in enumerate(ordered):
                c = a + idx
                if c <= b:
                    out[r, c] = int(token)
        else:
            for idx, _rec in enumerate(ordered):
                c = b - idx
                if c >= a:
                    out[r, c] = int(token)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gate, token, bg = _infer_gate_token_bg(g)
    opened = _open_gates(g, gate, bg)
    sources = _parse_sources(g, token)
    for source in sources:
        opened[int(source["r"]), int(source["c"])] = int(bg)
    records = _simulate_records(opened, sources, bg)
    return _render_combo_output(opened, records, token)


def _build_opened_grid(gates: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        grid[2 * pipe_idx, :] = BG
    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center in centers:
            grid[wr - 1, center] = OPEN
            grid[wr, center - 1] = BG
            grid[wr, center] = OPEN
            grid[wr, center + 1] = BG
            grid[wr + 1, center] = OPEN
    return grid


def _build_input(gates: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center in centers:
            grid[wr, center - 1 : center + 2] = GATE
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN
    return grid


def _build_multi_exit_structure(rng: np.random.Generator) -> tuple[list[list[int]], int]:
    n_walls = int(rng.integers(5, 8))
    width = int(rng.integers(16, 19))

    centers = [int(rng.integers(4, width - 4))]
    for _ in range(1, n_walls):
        shift = int(rng.choice([-2, -1, 1, 2]))
        centers.append(int(np.clip(centers[-1] + shift, 3, width - 4)))

    special_row = n_walls // 2
    special_center = centers[special_row]
    if special_center + 5 <= width - 3:
        extra = special_center + 5
    else:
        extra = special_center - 5
    gates = [[center] for center in centers]
    gates[special_row] = sorted([special_center, extra])
    return gates, width


def _combo_metrics(opened: np.ndarray, records: list[dict[str, object]]) -> dict[str, int]:
    buckets: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        a, b = tuple(rec["seg"])
        buckets[(int(rec["r"]), int(a), int(b))].append(rec)

    traversed_multi_exit = sum(1 for rec in records if bool(rec["used_multi_exit"]))
    max_exit_arity = max((int(rec["max_exit_arity"]) for rec in records), default=0)
    longest_depth = max((int(rec["depth"]) for rec in records), default=0)
    mixed_depth_merged_buckets = 0
    max_bucket_size = 0

    for bucket in buckets.values():
        max_bucket_size = max(max_bucket_size, len(bucket))
        if len(bucket) < 2:
            continue
        if len({int(rec["depth"]) for rec in bucket}) >= 2 and any(bool(rec["used_multi_exit"]) for rec in bucket):
            mixed_depth_merged_buckets += 1

    return {
        "traversed_multi_exit": int(traversed_multi_exit),
        "max_exit_arity": int(max_exit_arity),
        "longest_depth": int(longest_depth),
        "mixed_depth_merged_buckets": int(mixed_depth_merged_buckets),
        "max_bucket_size": int(max_bucket_size),
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(1200):
        gates, width = _build_multi_exit_structure(rng)
        opened = _build_opened_grid(gates, width)
        inp = _build_input(gates, width)

        records = _simulate_records(opened, _parse_sources(inp, TOKEN), BG)
        metrics = _combo_metrics(opened, records)
        if metrics["traversed_multi_exit"] < 1:
            continue
        if metrics["max_exit_arity"] > 2:
            continue
        if metrics["mixed_depth_merged_buckets"] < 1:
            continue
        if metrics["max_bucket_size"] < 2:
            continue
        if metrics["longest_depth"] < 3:
            continue

        combo_out = _render_combo_output(opened, records, TOKEN)
        far_out = _render_far_pack_output(opened, records, TOKEN, BG)
        if np.array_equal(combo_out, far_out):
            continue
        if not np.array_equal(main(inp), combo_out):
            continue
        return inp
    raise RuntimeError("failed to generate combo input for task 2b83f449")
