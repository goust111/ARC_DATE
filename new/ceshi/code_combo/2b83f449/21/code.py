# Combo variant of 2b83f449 (index 21)
# combo_spec: D7/0+D10/2
# Name: 可变宽度门+多出口

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask, paste

"""
Design analysis for combo D7/0 + D10/2

a) Concrete fused behavior on the ARC grid
   - The base task keeps the alternating lattice: even rows are pipe rows filled with
     bg=8, odd rows are wall rows filled with 0 except for gate bars in color 7, and
     every pipe row except the last starts with two edge tokens (color 3).
   - This combo introduces two gate widths in the input: 1x3 and 1x5. Both still use
     the same gate color. After opening, both become a 3-tall vertical hinge in color 6,
     but the wall-row opening that remains around that hinge preserves the source width:
       * 1x3 -> wall row shows [bg, 6, bg]
       * 1x5 -> wall row shows [bg, bg, 6, bg, bg]
   - That width difference is exactly what creates the D7 interaction. Under a wide gate,
     one side of a segment can now see two adjacent downward exits instead of one. Tokens
     stay deterministic because origin side is remembered: left-source tokens always take
     the leftmost available exit in their current segment, right-source tokens take the
     rightmost.
   - Traceable mini-example:
       current pipe segment columns = [0..7], opened wide gate centered at 8
       exit columns under that segment = [6, 7]
       a left token drops through 6, while under the old narrow-gate rule it would have
       dropped through only 7. The next rows then diverge, so width and multi-exit routing
       jointly affect the final token pile-up.

b) Complexity calibration
   - If wide gates were rare, the task would visually collapse back to the seed puzzle.
     The generator therefore enforces at least two wide gates and at least two narrow gates.
   - If wide gates existed but never changed any route, the variant would still degenerate.
     The generator rejects any sample whose output matches the baseline obtained by treating
     every wide gate as narrow.
   - To keep the rule learnable, I do not add holes or extra colors. The only new visible
     degree of freedom is width 3 vs width 5, repeated across several wall rows.

c) Hidden assumptions in the original main() that break here
   - Gate detection by "size exactly 3" is no longer valid. The new solver accepts 4-connected
     horizontal gate components of size 3 or 5.
   - "Each segment has <=1 exit" is false now. The old `seg_exit()` assertion would fail on
     every genuine wide-gate sample. The new solver enumerates all exits in the current
     segment and selects an extreme by source side.
   - "Packing side can be inferred from a unique entry above" is also false. A final segment
     may receive tokens from different entry columns. The new solver records each token's
     actual last chosen exit column and packs to the opposite end from that specific entry.

d) Placement legality
   - All gates stay fully inside the grid. For width 3 the center is in [2, W-3]; for width 5
     it is in [3, W-4].
   - The generator uses one gate per wall row, so gates never overlap or merge into a larger
     same-color component. That keeps width detection unambiguous.
   - The zig-zag spine is generated first in opened-state coordinates, then inverted into the
     input bars. Every candidate is validated by running `main()` on the produced input.

e) Visual observability
   - Humans can directly see that some wall rows contain short 1x3 bars and others long 1x5
     bars, and the corresponding output rows show narrow vs wide opened passages around the
     vertical hinge.
   - Because accepted samples must differ from the "all narrow" baseline, the width signal is
     not decorative: at least one token route genuinely changes because a wide gate offers more
     than one exit.
   - Because the generator also requires actual multi-exit traversals and a path depth >= 3,
     the combo behavior is not a tiny corner case hidden in one local row.

f) Visual cleanliness
   - The palette remains the seed task's palette: wall 0, token 3, opened hinge 6, gate 7,
     background 8.
   - I deliberately avoid holes and extra markers. The only objects are the original walls,
     gates, hinges, and tokens, so the scene stays legible despite the new routing behavior.

g) Color / change inferability
   - No new color mapping is introduced. Width alone encodes the extra parameter, and width
     is a human-visible geometric property repeated across examples.

h) Why this specific wide-gate geometry
   - The prompt asks for variable-width gates whose opened form changes the width of the
     resulting vertical passage and can create multiple exits. In this lattice there are only
     three local rows available around a gate. Keeping the same 3-tall hinge while preserving
     a wider wall-row opening is the minimal realizable geometry that both reflects the source
     width and reliably creates extra exits without introducing new object types.

i) New solver assumptions introduced by this combo, and how the generator guarantees them
   - Tokens still only originate from the left and right borders. The generator preserves that,
     so "left" and "right" remain stable identities for exit choice.
   - Only horizontal gate components of width 3 or 5 are considered real gates. The generator
     produces exactly those components and never lets same-row gates touch.
   - Final packing assumes the last chosen exit column lies inside the final segment when a token
     has dropped at least once. The generated networks satisfy that by construction, and all
     candidates are solver-validated before acceptance.
"""

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8
NARROW = 3
WIDE = 5


def _infer_gate_token_bg(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))

    gate = None
    for col in sorted(colors):
        if col == WALL:
            continue
        lab, n = label_mask(g == col, connectivity=4)
        found = False
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == k)
            size = int(pts.shape[0])
            if size not in (NARROW, WIDE):
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == size:
                found = True
                break
        if found:
            gate = int(col)
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


def _extract_gates(grid: np.ndarray, gate: int) -> list[tuple[int, int, int]]:
    lab, n = label_mask(np.array(grid, dtype=int) == int(gate), connectivity=4)
    gates: list[tuple[int, int, int]] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == k)
        size = int(pts.shape[0])
        if size not in (NARROW, WIDE):
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != size:
            continue
        gates.append((int(r0), int(c0), int(size)))
    gates.sort()
    return gates


def _open_from_input(grid: np.ndarray, gate: int, bg: int) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int, copy=True)
    height = out.shape[0]
    colors = set(map(int, out.flatten()))
    open_col = OPEN if gate == GATE and OPEN not in colors else (gate - 1 if gate > 0 else OPEN)

    for r, left, span in _extract_gates(out, gate):
        center = int(left + span // 2)
        out[r, left : left + span] = int(bg)
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < height:
                out[rr, center] = int(open_col)

    return out, int(open_col)


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


def _token_points(grid: np.ndarray, token: int) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in np.argwhere(np.array(grid, dtype=int) == int(token)))


def _source_side(c: int, width: int) -> str:
    if c <= 1:
        return "left"
    if c >= width - 2:
        return "right"
    return "left" if c < width // 2 else "right"


def _choose_exit(exits: list[int], side: str) -> int | None:
    if not exits:
        return None
    return int(min(exits) if side == "left" else max(exits))


def _simulate_opened(
    opened: np.ndarray,
    token_points: list[tuple[int, int]],
    bg: int,
    token: int,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    out = np.array(opened, dtype=int, copy=True)
    for r, c in token_points:
        if int(out[r, c]) == int(token):
            out[r, c] = int(bg)

    records: list[dict[str, object]] = []
    first_choices: dict[int, dict[str, int]] = defaultdict(dict)
    stats = {
        "max_depth": 0,
        "multi_steps": 0,
        "tokens_using_multi": 0,
        "paired_row_splits": 0,
    }

    for order, (start_r, start_c) in enumerate(token_points):
        side = _source_side(start_c, out.shape[1])
        r = int(start_r)
        c = int(start_c)
        depth = 0
        last_entry: int | None = None
        used_multi = False

        if int(out[r, c]) == int(bg):
            while True:
                seg = _find_segment(out, r, c, bg)
                if seg is None:
                    break
                exits = _exits_below(out, r, seg, bg)
                if not exits:
                    break
                if len(exits) > 1:
                    stats["multi_steps"] += 1
                    used_multi = True
                chosen = _choose_exit(exits, side)
                assert chosen is not None
                if depth == 0:
                    first_choices[start_r][side] = int(chosen)
                last_entry = int(chosen)
                r += 2
                c = int(chosen)
                depth += 1
                if r >= out.shape[0]:
                    break

        final_seg = _find_segment(out, r, c, bg)
        if final_seg is None:
            final_seg = (int(c), int(c))
        if used_multi:
            stats["tokens_using_multi"] += 1
        stats["max_depth"] = max(int(stats["max_depth"]), int(depth))
        records.append(
            {
                "order": int(order),
                "start_r": int(start_r),
                "start_c": int(start_c),
                "side": side,
                "r": int(r),
                "c": int(c),
                "depth": int(depth),
                "seg": final_seg,
                "entry_col": last_entry,
            }
        )

    for row_choices in first_choices.values():
        if "left" in row_choices and "right" in row_choices and int(row_choices["left"]) != int(row_choices["right"]):
            stats["paired_row_splits"] += 1

    buckets: dict[tuple[int, tuple[int, int], int], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        seg = rec["seg"]
        assert isinstance(seg, tuple)
        a, b = int(seg[0]), int(seg[1])
        entry_col = rec["entry_col"]
        if entry_col is not None:
            entry = int(entry_col)
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if str(rec["side"]) == "left" else b
        buckets[(int(rec["r"]), (a, b), int(pack_end))].append(rec)

    for (r, (a, b), pack_end), bucket in buckets.items():
        ordered = sorted(bucket, key=lambda item: int(item["order"]))
        if pack_end == a:
            for idx, _rec in enumerate(ordered):
                if a + idx <= b:
                    out[r, a + idx] = int(token)
        else:
            for idx, _rec in enumerate(ordered):
                if b - idx >= a:
                    out[r, b - idx] = int(token)

    return out, records, stats


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gate, token, bg = _infer_gate_token_bg(g)
    opened, _open_col = _open_from_input(g, gate, bg)
    token_points = _token_points(g, token)
    out, _records, _stats = _simulate_opened(opened, token_points, bg, token)
    return out


def _build_opened_grid(gates: list[list[tuple[int, int]]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        grid[2 * pipe_idx, :] = BG

    for wall_idx, row_gates in enumerate(gates):
        r = 2 * wall_idx + 1
        for center, span in row_gates:
            left = int(center - span // 2)
            grid[r, left : left + span] = BG
            grid[r - 1, center] = OPEN
            grid[r, center] = OPEN
            grid[r + 1, center] = OPEN
    return grid


def _build_input(gates: list[list[tuple[int, int]]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG

    for wall_idx, row_gates in enumerate(gates):
        r = 2 * wall_idx + 1
        for center, span in row_gates:
            left = int(center - span // 2)
            gate_patch = np.full((1, span), GATE, dtype=int)
            paste(grid, gate_patch, r, left)

    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _baseline_narrow_gates(gates: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    return [[(int(center), NARROW) for center, _span in row] for row in gates]


def _spine_centers(rng: np.random.Generator, n_walls: int, width: int) -> list[int] | None:
    centers: list[int] = []
    prev_center: int | None = None
    go_left = bool(rng.integers(2))

    for row in range(n_walls):
        if prev_center is None:
            options = list(range(5, width - 5))
        elif go_left:
            options = [c for c in range(3, prev_center - 1)]
        else:
            options = [c for c in range(prev_center + 2, width - 3)]
        if not options:
            return None

        rem = n_walls - row - 1
        filtered: list[int] = []
        for c in options:
            # Keep some room for future zig-zag steps so depth>=3 remains likely.
            left_room = c - 3
            right_room = (width - 4) - c
            if rem <= 0 or min(left_room, right_room) >= 1:
                filtered.append(int(c))
        if filtered:
            options = filtered

        if prev_center is None:
            options.sort(key=lambda c: (abs(c - width // 2), c))
        else:
            options.sort(key=lambda c: (abs(c - prev_center), c))
        band = options[: min(8, len(options))]
        center = int(band[int(rng.integers(len(band)))])
        centers.append(int(center))
        prev_center = int(center)
        go_left = not go_left

    return centers


def _assign_widths(rng: np.random.Generator, n_walls: int) -> list[int]:
    spans = [NARROW] * n_walls
    spans[0] = WIDE
    wide_target = int(rng.integers(2, min(5, n_walls) + 1))
    wide_rows = {0}
    while len(wide_rows) < wide_target:
        wide_rows.add(int(rng.integers(0, n_walls)))
    for row in wide_rows:
        spans[row] = WIDE

    if spans.count(NARROW) < 2 and n_walls >= 2:
        candidates = [i for i in range(1, n_walls)]
        rng.shuffle(candidates)
        for row in candidates[:2]:
            spans[row] = NARROW
    return [int(x) for x in spans]


def _build_gate_rows(centers: list[int], spans: list[int], width: int) -> list[list[tuple[int, int]]] | None:
    gates: list[list[tuple[int, int]]] = []
    for center, span in zip(centers, spans):
        min_center = 3 if span == WIDE else 2
        max_center = width - 4 if span == WIDE else width - 3
        if not (min_center <= center <= max_center):
            return None
        gates.append([(int(center), int(span))])
    return gates


def _variant_metrics(gates: list[list[tuple[int, int]]], width: int) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    opened = _build_opened_grid(gates, width)
    token_points = _token_points(_build_input(gates, width), TOKEN)
    fused_out, _records, stats = _simulate_opened(opened, token_points, BG, TOKEN)

    baseline_opened = _build_opened_grid(_baseline_narrow_gates(gates), width)
    baseline_out, _records_base, _stats_base = _simulate_opened(baseline_opened, token_points, BG, TOKEN)

    topology_multi = 0
    for r in range(0, opened.shape[0] - 2, 2):
        for seg in _segments_in_row(opened, r, BG):
            if len(_exits_below(opened, r, seg, BG)) > 1:
                topology_multi += 1

    enriched_stats = dict(stats)
    enriched_stats["wide_gates"] = int(sum(span == WIDE for row in gates for _center, span in row))
    enriched_stats["narrow_gates"] = int(sum(span == NARROW for row in gates for _center, span in row))
    enriched_stats["topology_multi_segments"] = int(topology_multi)
    enriched_stats["diff_vs_narrow"] = int(np.count_nonzero(fused_out != baseline_out))
    return fused_out, enriched_stats, baseline_out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(6, 9))
    width = int(rng.integers(19, 23))

    centers = _spine_centers(rng, n_walls, width)
    if centers is None:
        return None
    spans = _assign_widths(rng, n_walls)
    gates = _build_gate_rows(centers, spans, width)
    if gates is None:
        return None

    inp = _build_input(gates, width)
    fused_out, stats, _baseline_out = _variant_metrics(gates, width)

    if int(stats["wide_gates"]) < 2:
        return None
    if int(stats["narrow_gates"]) < 2:
        return None
    if int(stats["topology_multi_segments"]) < 2:
        return None
    if int(stats["multi_steps"]) < 2:
        return None
    if int(stats["tokens_using_multi"]) < 2:
        return None
    if int(stats["max_depth"]) < 3:
        return None
    if int(stats["diff_vs_narrow"]) < 1:
        return None

    if not np.array_equal(main(inp), fused_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/21")
