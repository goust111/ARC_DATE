from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 800


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _separator_info(g: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    sep = sub == int(color)
    h, w = sep.shape

    full_rows = [r for r in range(h) if bool(np.all(sep[r, :]))]
    full_cols = [c for c in range(w) if bool(np.all(sep[:, c]))]
    if len(full_rows) < 2 or len(full_cols) < 2:
        return None
    if full_rows[0] != 0 or full_rows[-1] != h - 1:
        return None
    if full_cols[0] != 0 or full_cols[-1] != w - 1:
        return None
    if any(full_rows[i + 1] - full_rows[i] < 2 for i in range(len(full_rows) - 1)):
        return None
    if any(full_cols[i + 1] - full_cols[i] < 2 for i in range(len(full_cols) - 1)):
        return None

    union = np.zeros_like(sep, dtype=bool)
    union[full_rows, :] = True
    union[:, full_cols] = True
    if not np.array_equal(sep, union):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "row_lines": [r0 + r for r in full_rows],
        "col_lines": [c0 + c for c in full_cols],
    }


def _extract_tables(g: np.ndarray) -> tuple[int, int, list[dict]]:
    candidates: list[tuple[int, list[dict]]] = []
    for color in sorted(map(int, np.unique(g))):
        infos = []
        for coords in _components(g == color):
            info = _separator_info(g, color, coords)
            if info is not None:
                infos.append(info)
        if len(infos) == 3:
            candidates.append((int(color), sorted(infos, key=lambda x: x["bbox"])))

    assert len(candidates) == 1, "expected exactly one separator color forming three table frames"
    sep_color, infos = candidates[0]

    outside = np.ones(g.shape, dtype=bool)
    for info in infos:
        r0, c0, r1, c1 = info["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False
    outside_colors = [int(v) for v in np.unique(g[outside])] if bool(np.any(outside)) else []
    if outside_colors:
        assert len(outside_colors) == 1, "area outside the three tables must be a single background color"
        bg = outside_colors[0]
    else:
        counts = Counter(map(int, g.flat))
        bg = counts.most_common(1)[0][0]
    assert bg != sep_color

    tables = [_parse_table(g, bg, sep_color, info) for info in infos]
    return int(bg), int(sep_color), tables


def _parse_table(g: np.ndarray, bg: int, sep_color: int, info: dict) -> dict:
    row_lines = list(map(int, info["row_lines"]))
    col_lines = list(map(int, info["col_lines"]))
    row_spans = [(row_lines[i] + 1, row_lines[i + 1] - 1) for i in range(len(row_lines) - 1)]
    col_spans = [(col_lines[i] + 1, col_lines[i + 1] - 1) for i in range(len(col_lines) - 1)]
    assert row_spans and col_spans
    assert all(r0 <= r1 for r0, r1 in row_spans)
    assert all(c0 <= c1 for c0, c1 in col_spans)

    compact = np.full((len(row_spans), len(col_spans)), int(bg), dtype=int)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            vals = set(map(int, np.unique(cell)))
            assert int(sep_color) not in vals, "separator color may not appear inside a region"
            assert len(vals) == 1, "each separator-delimited region must be uniform"
            compact[i, j] = vals.pop()

    return {
        "bbox": tuple(map(int, info["bbox"])),
        "row_spans": row_spans,
        "col_spans": col_spans,
        "compact": compact,
    }


def _orientations(arr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    a = np.asarray(arr, dtype=int)
    return [
        ("identity", a.copy()),
        ("rot90", np.rot90(a, 1).copy()),
        ("rot180", np.rot90(a, 2).copy()),
        ("rot270", np.rot90(a, 3).copy()),
        ("reflect_lr", np.fliplr(a).copy()),
        ("reflect_ud", np.flipud(a).copy()),
        ("reflect_main_diag", a.T.copy()),
        ("reflect_anti_diag", np.flipud(np.fliplr(a)).T.copy()),
    ]


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, sep_color, tables = _extract_tables(g)

    sources = [table for table in tables if not np.any(table["compact"] == bg)]
    targets = [table for table in tables if np.any(table["compact"] == bg)]
    assert len(sources) == 2 and len(targets) == 1, "expected two complete sources and one partial target"
    target = targets[0]
    target_compact = target["compact"]
    assert target_compact.shape[0] == target_compact.shape[1], "target table must compress to k by k"
    k = int(target_compact.shape[0])
    assert k >= 3
    known = target_compact != bg
    assert 0 < int(np.count_nonzero(known)) < k * k, "target must contain both clues and empty regions"

    for source in sources:
        compact = source["compact"]
        assert compact.shape == (k, k), "all three tables must compress to the same k by k shape"
        assert not np.any(compact == bg), "source tables must be fully colored"

    matches: list[tuple[int, int, str, np.ndarray]] = []
    for source_index, source in enumerate(sources):
        for orientation_index, (name, candidate) in enumerate(_orientations(source["compact"])):
            if candidate.shape == target_compact.shape and bool(np.all(candidate[known] == target_compact[known])):
                matches.append((source_index, orientation_index, name, candidate))

    assert len(matches) == 1, "filled target clues must select one source and one orientation"
    source_index, orientation_index, orientation_name, chosen = matches[0]
    return {
        "grid": g,
        "bg": bg,
        "sep": sep_color,
        "tables": tables,
        "sources": sources,
        "target": target,
        "known": known,
        "source_index": source_index,
        "orientation_index": orientation_index,
        "orientation_name": orientation_name,
        "chosen": chosen,
        "matches": matches,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the sparse target table by choosing the uniquely matching source table and orientation.

    Input structure:
    - The grid contains three separator-framed tables on one background color.
    - Two tables are complete k by k region color maps after compression; the third has only sparse colored clues.
    - The three tables have their own uneven row heights and column widths, so pixel-scale sizes do not match.

    Rule:
    - Compress each separator-delimited region to one color cell.
    - Test both complete source maps under all eight rotations/reflections against every filled target clue.
    - Fill the target's empty regions from the single matching transformed source, using the target table's own spans.

    Key insight:
    - The sparse target clues are a two-part signature: they identify both which similar source table was copied
      and which dihedral orientation was used, but only after comparing at region scale.
    """
    details = _solve_details(grid)
    out = details["grid"].copy()
    target = details["target"]
    chosen = details["chosen"]
    for i, (r0, r1) in enumerate(target["row_spans"]):
        for j, (c0, c1) in enumerate(target["col_spans"]):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(chosen[i, j])
    return out


def _vectors_proportional(a: list[int], b: list[int]) -> bool:
    return all(int(x) * int(b[0]) == int(y) * int(a[0]) for x, y in zip(a, b))


def _sample_spans(k: int) -> tuple[list[int], list[int]]:
    max_cell = 3 if k <= 4 else 2
    for _ in range(200):
        rows = [random.randint(1, max_cell) for _ in range(k)]
        cols = [random.randint(1, max_cell) for _ in range(k)]
        if len(set(rows)) > 1 and len(set(cols)) > 1:
            return rows, cols
    raise RuntimeError("failed to sample uneven table spans")


def _sample_three_span_sets(k: int) -> list[tuple[list[int], list[int]]]:
    sets: list[tuple[list[int], list[int]]] = []
    for _ in range(500):
        rows, cols = _sample_spans(k)
        ok = True
        for prev_rows, prev_cols in sets:
            if _vectors_proportional(rows, prev_rows) and _vectors_proportional(cols, prev_cols):
                ok = False
                break
        if ok:
            sets.append((rows, cols))
            if len(sets) == 3:
                return sets
    raise RuntimeError("failed to sample non-proportional table spans")


def _render_table(compact: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    k = int(compact.shape[0])
    h = int(sum(row_heights) + k + 1)
    w = int(sum(col_widths) + k + 1)
    table = np.full((h, w), SEP, dtype=int)
    r = 1
    for i, rh in enumerate(row_heights):
        c = 1
        for j, cw in enumerate(col_widths):
            table[r : r + rh, c : c + cw] = int(compact[i, j])
            c += cw + 1
        r += rh + 1
    return table


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _pairwise_distinct_oriented(sources: list[np.ndarray]) -> bool:
    keys: set[bytes] = set()
    for source in sources:
        for _name, oriented in _orientations(source):
            key = oriented.tobytes()
            if key in keys:
                return False
            keys.add(key)
    return True


def _make_source_pair(k: int) -> tuple[np.ndarray, np.ndarray] | None:
    color_count = random.randint(4, min(6, len(PALETTE)))
    colors = random.sample(PALETTE, color_count)
    for _ in range(300):
        a = np.array([[random.choice(colors) for _ in range(k)] for _ in range(k)], dtype=int)
        if len(set(map(int, a.flat))) < min(4, k * k):
            continue
        b = a.copy()
        diff_count = random.randint(2, max(2, min(5, k)))
        for flat_index in random.sample(range(k * k), diff_count):
            r, c = divmod(flat_index, k)
            choices = [color for color in colors if color != int(b[r, c])]
            b[r, c] = random.choice(choices)
        if int(np.count_nonzero(a != b)) != diff_count:
            continue
        if _pairwise_distinct_oriented([a, b]):
            return a, b
    return None


def _candidate_maps(sources: list[np.ndarray]) -> list[tuple[int, int, str, np.ndarray]]:
    candidates = []
    for source_index, source in enumerate(sources):
        for orientation_index, (name, oriented) in enumerate(_orientations(source)):
            candidates.append((source_index, orientation_index, name, oriented))
    return candidates


def _signature_positions(
    target_full: np.ndarray,
    candidates: list[tuple[int, int, str, np.ndarray]],
    true_source: int,
    true_orientation: int,
) -> list[tuple[int, int]] | None:
    k = int(target_full.shape[0])
    all_positions = [(r, c) for r in range(k) for c in range(k)]
    wrong = [
        (source_index, orientation_index, candidate)
        for source_index, orientation_index, _name, candidate in candidates
        if not (source_index == true_source and orientation_index == true_orientation)
    ]
    if any(np.array_equal(candidate, target_full) for _si, _oi, candidate in wrong):
        return None

    remaining = set(range(len(wrong)))
    chosen: list[tuple[int, int]] = []
    max_clues = min(k * k - 2, 8 if k <= 4 else 10)
    while remaining:
        scored: list[tuple[int, tuple[int, int]]] = []
        for pos in all_positions:
            if pos in chosen:
                continue
            r, c = pos
            score = sum(
                int(wrong[idx][2][r, c]) != int(target_full[r, c])
                for idx in remaining
            )
            if score:
                scored.append((score, pos))
        if not scored:
            return None
        best = max(score for score, _pos in scored)
        top = [pos for score, pos in scored if score == best]
        pos = random.choice(top)
        chosen.append(pos)
        r, c = pos
        remaining = {
            idx
            for idx in remaining
            if int(wrong[idx][2][r, c]) == int(target_full[r, c])
        }
        if len(chosen) > max_clues:
            return None

    same_orientation_wrong = [
        idx
        for idx, (source_index, orientation_index, _candidate) in enumerate(wrong)
        if source_index != true_source and orientation_index == true_orientation
    ]
    same_source_wrong = [
        idx
        for idx, (source_index, orientation_index, _candidate) in enumerate(wrong)
        if source_index == true_source and orientation_index != true_orientation
    ]

    def hit(indices: list[int]) -> bool:
        return any(
            int(wrong[idx][2][r, c]) != int(target_full[r, c])
            for idx in indices
            for r, c in chosen
        )

    if same_orientation_wrong and not hit(same_orientation_wrong):
        return None
    if same_source_wrong and not hit(same_source_wrong):
        return None

    min_clues = min(3, k * k - 2)
    extras = [pos for pos in all_positions if pos not in chosen]
    random.shuffle(extras)
    while len(chosen) < min_clues:
        chosen.append(extras.pop())

    extra_cap = random.randint(len(chosen), max_clues)
    while extras and len(chosen) < extra_cap and random.random() < 0.35:
        chosen.append(extras.pop())
    random.shuffle(chosen)
    return chosen


def _place_tables(tables: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    src_a, src_b, target = tables
    source_order = [0, 1]
    random.shuffle(source_order)
    top_tables = [tables[source_order[0]], tables[source_order[1]]]
    gap = random.randint(1, 3)

    top_h = max(int(top_tables[0].shape[0]), int(top_tables[1].shape[0]))
    top_w = int(top_tables[0].shape[1]) + gap + int(top_tables[1].shape[1])
    min_h = top_h + gap + int(target.shape[0])
    min_w = max(top_w, int(target.shape[1]))
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.full((h, w), BG, dtype=int)

    top_base = random.randint(0, h - min_h)
    left_base = random.randint(0, w - top_w)
    loc_top_0 = (
        top_base + random.randint(0, top_h - int(top_tables[0].shape[0])),
        left_base,
    )
    loc_top_1 = (
        top_base + random.randint(0, top_h - int(top_tables[1].shape[0])),
        left_base + int(top_tables[0].shape[1]) + gap,
    )
    target_r = top_base + top_h + gap
    target_c = random.randint(0, w - int(target.shape[1]))

    locations_by_original: list[tuple[int, int] | None] = [None, None, None]
    for original_index, table, loc in (
        (source_order[0], top_tables[0], loc_top_0),
        (source_order[1], top_tables[1], loc_top_1),
        (2, target, (target_r, target_c)),
    ):
        _paste(grid, table, loc[0], loc[1])
        locations_by_original[original_index] = loc

    assert all(loc is not None for loc in locations_by_original)
    return grid, [loc for loc in locations_by_original if loc is not None]


def _gen_attempt() -> np.ndarray | None:
    k = random.choice([3, 3, 4, 4, 5])
    pair = _make_source_pair(k)
    if pair is None:
        return None
    source_a, source_b = pair
    sources = [source_a, source_b]
    candidates = _candidate_maps(sources)

    true_source = random.randrange(2)
    true_orientation = random.randrange(8)
    target_full = [
        candidate
        for source_index, orientation_index, _name, candidate in candidates
        if source_index == true_source and orientation_index == true_orientation
    ][0]

    signature = _signature_positions(target_full, candidates, true_source, true_orientation)
    if signature is None:
        return None
    target_sparse = np.full((k, k), BG, dtype=int)
    for r, c in signature:
        target_sparse[r, c] = int(target_full[r, c])
    if not (0 < int(np.count_nonzero(target_sparse != BG)) < k * k):
        return None

    span_sets = _sample_three_span_sets(k)
    source_a_table = _render_table(source_a, *span_sets[0])
    source_b_table = _render_table(source_b, *span_sets[1])
    target_sparse_table = _render_table(target_sparse, *span_sets[2])
    target_full_table = _render_table(target_full, *span_sets[2])

    placed = _place_tables([source_a_table, source_b_table, target_sparse_table])
    if placed is None:
        return None
    grid, locations = placed
    expected = grid.copy()
    target_r, target_c = locations[2]
    _paste(expected, target_full_table, target_r, target_c)

    try:
        actual = main(grid)
        details = _solve_details(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(details["matches"]) != 1:
        return None
    filled_target = details["chosen"]
    if not np.array_equal(filled_target, target_full):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-source orientation signature input")
