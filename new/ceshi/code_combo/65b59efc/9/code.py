# Combo variant of 65b59efc (index 9)
# combo_spec: D6/0+D16/0
# Name: 两轮优先级叠加

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import paste, random_sprite


# Design analysis for the fused variant:
# - Original task assumptions broken here:
#   1) slot exclusivity is gone: different templates may target the same (lr, lc) slot;
#   2) rendering is no longer one-pass: templates split into two visible priority tiers.
# - Concrete example:
#   N=3, template A has its palette swatch on the upper palette row (high priority),
#   template B has its swatch on the lower palette row (low priority), and both place
#   a marker at local slot (1, 2). The output block at (1, 2) is first filled with A's
#   recolored stamp, then replaced by B's recolored stamp. In the original task this
#   collision was illegal.
# - To keep the rule inferable rather than noisy, the generator restricts each output
#   slot to at most one high-priority stamp and at most one low-priority stamp. This
#   still exercises overlap while avoiding deep multi-layer clutter.
# - Priority must be human-visible. Using palette-swatch height is explicit in the input:
#   upper swatch row means "first round", lower swatch row means "second round".
# - Generator-side legality guarantees:
#   1) both priority tiers are present in every sample;
#   2) at least one cross-priority collision exists;
#   3) at least one collision would resolve differently under naive column order, so the
#      sample cannot collapse back to a plain D6-only overlay task;
#   4) every template appears at least once, keeping color/template/priority evidence repeated.


# -- CFG: colors --
_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- CFG: grid dimensions --
_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H = 2
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

# -- CFG: template generation --
_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

# -- CFG: marker placement --
_MIN_SLOT_COVERAGE = 0.65
_MAX_CONFLICT_FRAC = 0.35

# -- CFG: retry --
_MAX_RETRIES = 400


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _sample_template(n: int, color: int) -> np.ndarray:
    sprite = random_sprite(
        n,
        n,
        density=random.uniform(*_DENSITY_RANGE),
        symmetry=random.choice(_SYMMETRY_OPTIONS),
        color_palette=[color],
        connectivity=_TEMPLATE_CONNECTIVITY,
        background=_BG,
    )
    return sprite == color


def _sample_distinct_templates(n: int, colors: list[int]) -> list[np.ndarray]:
    templates: list[np.ndarray] = []
    while len(templates) < len(colors):
        cand = _sample_template(n, colors[len(templates)])
        if any(np.array_equal(cand, prev) for prev in templates):
            continue
        templates.append(cand)
    return templates


def _detect_bg_sep(g: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    row_thr = max(2, g.shape[1] // 3)
    col_thr = max(2, g.shape[0] // 3)

    def score(col: int) -> tuple[int, int, int, int]:
        mask = g == col
        row_hits = int(np.sum(mask.sum(axis=1) >= row_thr))
        col_hits = int(np.sum(mask.sum(axis=0) >= col_thr))
        return (int(row_hits >= 2), int(col_hits >= 1), row_hits + col_hits, counts[col])

    sep = max((int(c) for c in counts if int(c) != bg), key=score)
    return bg, sep


def _decode_components(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, sep = _detect_bg_sep(g)

    row_thr = max(2, W // 3)
    col_thr = max(2, H // 3)
    sep_rows = [r for r in range(H) if int(np.sum(g[r, :] == sep)) >= row_thr]
    sep_cols = [c for c in range(W) if int(np.sum(g[:, c] == sep)) >= col_thr]
    assert len(sep_rows) >= 2 and len(sep_cols) >= 1

    n = sep_cols[0]
    k = len(sep_cols) + 1
    starts = [i * (n + 1) for i in range(k)]
    mid_r0 = n + 1
    pal_r0 = sep_rows[-1] + 1

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st:st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pcols: list[int] = []
    swatch_rows: list[int] = []
    for st in starts:
        sub = g[pal_r0:H, st:st + n]
        coords = np.argwhere((sub != bg) & (sub != sep))
        assert coords.size > 0
        vals = {int(sub[r, c]) for r, c in coords}
        assert len(vals) == 1
        pcols.append(vals.pop())
        swatch_rows.append(int(coords[:, 0].min()))

    unique_swatch_rows = sorted(set(swatch_rows))
    assert len(unique_swatch_rows) == 2
    priority_rank = {row: rank for rank, row in enumerate(unique_swatch_rows)}
    priorities = [priority_rank[row] for row in swatch_rows]
    tiles = [np.where(mask, pcols[j], bg) for j, mask in enumerate(tmasks)]

    markers: list[tuple[int, int, int, int, int, int]] = []
    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                abs_r = mid_r0 + lr
                abs_c = st + lc
                v = int(g[abs_r, abs_c])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = tcol_to_j[v]
                markers.append((priorities[j], abs_c, abs_r, j, lr, lc))

    return {
        "grid": g,
        "bg": bg,
        "sep": sep,
        "n": n,
        "k": k,
        "starts": starts,
        "mid_r0": mid_r0,
        "pal_r0": pal_r0,
        "tcols": tcols,
        "tmasks": tmasks,
        "pcols": pcols,
        "priorities": priorities,
        "tiles": tiles,
        "markers": markers,
    }


def _render_from_components(comp: dict, *, use_priority: bool) -> np.ndarray:
    n = comp["n"]
    bg = comp["bg"]
    out = np.full((n * n, n * n), bg, dtype=int)

    if use_priority:
        ordered = sorted(comp["markers"], key=lambda x: (x[0], x[1], x[2]))
    else:
        ordered = sorted(comp["markers"], key=lambda x: (x[1], x[2]))

    for _, _, _, j, lr, lc in ordered:
        tile = comp["tiles"][j]
        out[lr * n:(lr + 1) * n, lc * n:(lc + 1) * n] = tile
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    comp = _decode_components(grid)
    n = comp["n"]
    slot_priorities: dict[tuple[int, int], set[int]] = defaultdict(set)
    for pr, _, _, _, lr, lc in comp["markers"]:
        slot_priorities[(lr, lc)].add(pr)
    overlap_slots = sum(len(prs) >= 2 for prs in slot_priorities.values())

    priority_out = _render_from_components(comp, use_priority=True)
    naive_out = _render_from_components(comp, use_priority=False)
    changed_blocks = 0
    for rr in range(n):
        for cc in range(n):
            a = priority_out[rr * n:(rr + 1) * n, cc * n:(cc + 1) * n]
            b = naive_out[rr * n:(rr + 1) * n, cc * n:(cc + 1) * n]
            if not np.array_equal(a, b):
                changed_blocks += 1
    return {"overlap_slots": overlap_slots, "priority_sensitive_blocks": changed_blocks}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a two-round stamp code: read templates, markers, and palette cues, then render layered stamps.

    Input structure:
    - The grid is split into K tile columns of width N with dashed separator rows/columns.
    - Top band: K template masks, one colored mask per tile column.
    - Middle band: marker pixels; a marker's color still identifies which template is requested.
    - Bottom band: K palette swatches; the swatch color recolors the template, and the swatch's vertical row indicates priority tier.

    Rule:
    - A marker's local coordinates (lr, lc) within its tile column choose which N x N output block receives the stamp.
    - Templates whose swatches sit on the higher palette row stamp first; templates whose swatches sit on the lower palette row stamp second.
    - Overlap is allowed. If several markers target the same output block, later stamps replace earlier ones; within a tier, markers follow left-to-right then top-to-bottom order in the middle band.

    Key insight:
    - The palette band now carries two signals at once: its color still picks the recoloring, while its height splits templates into the high and low priority rounds that resolve collisions.
    """
    comp = _decode_components(grid)
    return _render_from_components(comp, use_priority=True)


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _sample_priorities(k: int) -> list[int]:
    js = list(range(k))
    while True:
        high_count = random.randint(1, k - 1)
        high_set = set(random.sample(js, high_count))
        low_set = set(js) - high_set
        if any(low < high for high in high_set for low in low_set):
            return [0 if j in high_set else 1 for j in js]


def _sample_slot_partition(n: int, high_count: int, low_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    min_union = max(int(np.ceil(_MIN_SLOT_COVERAGE * n * n)), 2)

    while True:
        union_count = random.randint(min_union, n * n)
        conflict_cap = min(union_count - 1, max(1, int(np.ceil(_MAX_CONFLICT_FRAC * n * n))))
        conflict_count = random.randint(1, conflict_cap)
        min_high_only = max(0, high_count - conflict_count)
        min_low_only = max(0, low_count - conflict_count)
        remainder = union_count - conflict_count - min_high_only - min_low_only
        if remainder < 0:
            continue
        extra_high = random.randint(0, remainder)
        extra_low = remainder - extra_high
        high_only_count = min_high_only + extra_high
        low_only_count = min_low_only + extra_low

        random.shuffle(all_slots)
        conflict_slots = all_slots[:conflict_count]
        high_only_slots = all_slots[conflict_count:conflict_count + high_only_count]
        low_only_slots = all_slots[conflict_count + high_only_count:conflict_count + high_only_count + low_only_count]
        return conflict_slots, high_only_slots, low_only_slots


def _assign_group_slots(
    slots: list[tuple[int, int]],
    group_js: list[int],
    *,
    preset: dict[tuple[int, int], int] | None = None,
) -> dict[tuple[int, int], int]:
    preset = {} if preset is None else dict(preset)
    free_slots = [slot for slot in slots if slot not in preset]

    assigned = set(preset.values())
    need = [j for j in group_js if j not in assigned]
    assert len(need) <= len(free_slots)

    fill = need[:]
    while len(fill) < len(free_slots):
        fill.append(random.choice(group_js))
    random.shuffle(fill)

    for slot, j in zip(free_slots, fill):
        preset[slot] = j
    return preset


def _build_marker_records(
    n: int,
    priorities: list[int],
    high_map: dict[tuple[int, int], int],
    low_map: dict[tuple[int, int], int],
) -> list[tuple[int, int, int, int, int, int]]:
    markers: list[tuple[int, int, int, int, int, int]] = []
    for (rr, cc), j in high_map.items():
        markers.append((priorities[j], _raw_col(j, cc, n), n + 1 + rr, j, rr, cc))
    for (rr, cc), j in low_map.items():
        markers.append((priorities[j], _raw_col(j, cc, n), n + 1 + rr, j, rr, cc))
    return markers


def _render_expected(
    n: int,
    bg: int,
    tiles: list[np.ndarray],
    markers: list[tuple[int, int, int, int, int, int]],
    *,
    use_priority: bool,
) -> np.ndarray:
    out = np.full((n * n, n * n), bg, dtype=int)
    if use_priority:
        ordered = sorted(markers, key=lambda x: (x[0], x[1], x[2]))
    else:
        ordered = sorted(markers, key=lambda x: (x[1], x[2]))
    for _, _, _, j, rr, cc in ordered:
        out[rr * n:(rr + 1) * n, cc * n:(cc + 1) * n] = tiles[j]
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    pcols = random.sample(_AVAILABLE_COLORS, k)
    priorities = _sample_priorities(k)

    pal_h = _PAL_H
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates = _sample_distinct_templates(n, tcols)
    tiles = [np.where(mask, pcols[j], _BG) for j, mask in enumerate(templates)]

    high_js = [j for j, pr in enumerate(priorities) if pr == 0]
    low_js = [j for j, pr in enumerate(priorities) if pr == 1]
    conflict_slots, high_only_slots, low_only_slots = _sample_slot_partition(n, len(high_js), len(low_js))

    candidate_pairs = [(hj, lj) for hj in high_js for lj in low_js if lj < hj]
    if not candidate_pairs or not conflict_slots:
        return None
    special_slot = random.choice(conflict_slots)
    special_high, special_low = random.choice(candidate_pairs)

    high_slots = conflict_slots + high_only_slots
    low_slots = conflict_slots + low_only_slots
    high_map = _assign_group_slots(high_slots, high_js, preset={special_slot: special_high})
    low_map = _assign_group_slots(low_slots, low_js, preset={special_slot: special_low})

    markers = _build_marker_records(n, priorities, high_map, low_map)
    expected = _render_expected(n, _BG, tiles, markers, use_priority=True)
    naive = _render_expected(n, _BG, tiles, markers, use_priority=False)
    if np.array_equal(expected, naive):
        return None

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    for (rr, cc), j in high_map.items():
        grid[n + 1 + rr, _raw_col(j, cc, n)] = tcols[j]
    for (rr, cc), j in low_map.items():
        grid[n + 1 + rr, _raw_col(j, cc, n)] = tcols[j]

    pal_r0 = 2 * n + 2
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        if priorities[j] == 0:
            swatch_r = pal_r0
            size = 2 if n >= 4 else 1
        else:
            swatch_r = pal_r0 + 1
            size = 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0:c0 + size] = col

    try:
        solved = main(grid)
        if not np.array_equal(solved, expected):
            return None
        metrics = _behavior_metrics(grid)
        if metrics["overlap_slots"] <= 0 or metrics["priority_sensitive_blocks"] <= 0:
            return None
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
