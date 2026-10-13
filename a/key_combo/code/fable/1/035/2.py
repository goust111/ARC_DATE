from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _extract_rows(grid: np.ndarray, bg: int) -> list[dict]:
    rows = []
    height, _width = map(int, grid.shape)
    for r in range(height):
        cols = np.flatnonzero(grid[r] != bg)
        if cols.size == 0:
            continue
        rows.append(
            {
                "row": int(r),
                "cols": tuple(int(c) for c in cols),
                "values": tuple(int(grid[r, c]) for c in cols),
                "min_col": int(cols[0]),
                "max_col": int(cols[-1]),
            }
        )
    assert rows, "expected at least one damaged horizontal string"
    return rows


def _candidate_placements(row: dict, width: int, length: int) -> list[dict]:
    start_lo = row["max_col"] - length + 1
    start_hi = row["min_col"]
    candidates = []
    for start in range(start_lo, start_hi + 1):
        for mirrored in (False, True):
            constraints = []
            for c, color in zip(row["cols"], row["values"]):
                offset = c - start
                pos = length - 1 - offset if mirrored else offset
                if not (0 <= pos < length):
                    constraints = None
                    break
                constraints.append((int(pos), int(color)))
            if constraints is None:
                continue
            candidates.append(
                {
                    "start": int(start),
                    "mirrored": bool(mirrored),
                    "constraints": tuple(constraints),
                    "covered": frozenset(pos for pos, _color in constraints),
                }
            )
    return candidates


def _render_output(
    shape: tuple[int, int],
    bg: int,
    master: tuple[int, ...],
    rows: list[dict],
    placements: list[dict],
) -> np.ndarray:
    height, width = map(int, shape)
    length = len(master)
    out = np.full((height, width), int(bg), dtype=int)
    master_arr = np.asarray(master, dtype=int)
    mirrored_arr = master_arr[::-1]

    for row, placement in zip(rows, placements):
        seq = mirrored_arr if placement["mirrored"] else master_arr
        start = int(placement["start"])
        left = max(0, start)
        right = min(width - 1, start + length - 1)
        if left > right:
            continue
        out[int(row["row"]), left : right + 1] = seq[left - start : right - start + 1]

    return out


def _enumerate_outputs(grid: np.ndarray, limit: int = 2) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    rows = _extract_rows(g, bg)
    height, width = map(int, g.shape)
    min_length = max(row["max_col"] - row["min_col"] + 1 for row in rows)
    unique_outputs: dict[bytes, np.ndarray] = {}

    for length in range(min_length, width + 1):
        row_candidates = [_candidate_placements(row, width, length) for row in rows]
        if any(not candidates for candidates in row_candidates):
            continue

        possible_cover = [set().union(*(cand["covered"] for cand in candidates)) for candidates in row_candidates]
        if set().union(*possible_cover) != set(range(length)):
            continue

        order = sorted(
            range(len(rows)),
            key=lambda idx: (len(row_candidates[idx]), -len(rows[idx]["cols"]), rows[idx]["row"]),
        )
        suffix_cover = [set() for _ in range(len(rows) + 1)]
        running: set[int] = set()
        for depth in range(len(rows) - 1, -1, -1):
            running = running | possible_cover[order[depth]]
            suffix_cover[depth] = running.copy()

        master = [-1] * length
        covered = [False] * length
        placements: list[dict | None] = [None] * len(rows)

        def dfs(depth: int) -> None:
            if len(unique_outputs) >= limit:
                return
            if depth == len(rows):
                if not all(covered):
                    return
                assert all(p is not None for p in placements)
                out = _render_output(
                    g.shape,
                    bg,
                    tuple(int(v) for v in master),
                    rows,
                    [p for p in placements if p is not None],
                )
                key = out.tobytes()
                if key not in unique_outputs:
                    unique_outputs[key] = out
                return

            row_idx = order[depth]
            remaining_cover = suffix_cover[depth + 1]
            for candidate in row_candidates[row_idx]:
                changed_master = []
                changed_cover = []
                ok = True
                for pos, color in candidate["constraints"]:
                    if master[pos] == -1:
                        master[pos] = int(color)
                        changed_master.append(pos)
                    elif master[pos] != int(color):
                        ok = False
                        break
                    if not covered[pos]:
                        covered[pos] = True
                        changed_cover.append(pos)

                if ok and all(covered[pos] or pos in remaining_cover for pos in range(length)):
                    placements[row_idx] = candidate
                    dfs(depth + 1)
                    placements[row_idx] = None

                for pos in changed_cover:
                    covered[pos] = False
                for pos in changed_master:
                    master[pos] = -1

        dfs(0)
        if len(unique_outputs) >= limit:
            break

    return list(unique_outputs.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden master color string and use it to complete every broken translated copy.

    Input structure
    - The background is the most common color.
    - Foreground appears only on several separated rows, each row containing one damaged horizontal copy.
    - Every copy comes from the same unseen master string, but each row may use either the master or its left-right mirror, shifted and optionally clipped by the grid edges.

    Rule
    - Simultaneously infer the master string length, each row's horizontal placement, and whether that row is forward or mirrored.
    - Accept only arrangements where all visible colors agree on one master string and the visible union covers every master position.
    - Fill each row with its entire in-bounds copy of the solved string, preserving clipping at the left and right borders.

    Key insight
    - Orientation and content are entangled: a single residue row is ambiguous, but a globally consistent set of alignments makes all partial traces snap together into one shared string.
    """
    outputs = _enumerate_outputs(grid, limit=2)
    assert outputs, "no consistent master string reconstruction found"
    assert len(outputs) == 1, "expected a unique completed output"
    return outputs[0]


def _rand_master(length: int, bg: int) -> tuple[int, ...]:
    palette_pool = [color for color in range(10) if color != bg]
    for _ in range(200):
        n_colors = random.randint(3, min(5, length - 1))
        palette = random.sample(palette_pool, n_colors)
        seq = palette.copy()
        while len(seq) < length:
            seq.append(random.choice(palette))
        random.shuffle(seq)
        if len(set(seq)) == length:
            continue
        if seq == list(reversed(seq)):
            continue
        return tuple(int(v) for v in seq)
    raise RuntimeError("failed to sample a nontrivial master string")


def _pick_rows(height: int, count: int) -> list[int]:
    choices = list(range(1, height - 1, 2))
    return sorted(random.sample(choices, count))


def _full_start(width: int, length: int) -> int:
    starts = list(range(0, width - length + 1))
    inner = [start for start in starts if 0 < start < width - length]
    return random.choice(inner or starts)


def _clipped_start(width: int, length: int, side: str) -> int:
    overlap = random.randint(4, length - 1)
    if side == "left":
        return overlap - length
    if side == "right":
        return width - overlap
    raise ValueError(f"unknown side: {side}")


def _master_to_col(length: int, start: int, mirrored: bool, pos: int) -> int:
    return int(start + (length - 1 - pos if mirrored else pos))


def _visible_positions(length: int, width: int, start: int, mirrored: bool) -> list[int]:
    return [
        pos
        for pos in range(length)
        if 0 <= _master_to_col(length, start, mirrored, pos) < width
    ]


def _boundary_position(length: int, width: int, start: int, mirrored: bool) -> int | None:
    boundary_col = 0 if start < 0 else width - 1 if start + length > width else None
    if boundary_col is None:
        return None
    for pos in range(length):
        if _master_to_col(length, start, mirrored, pos) == boundary_col:
            return int(pos)
    return None


def _allocate_visible_sets(
    row_specs: list[dict],
    length: int,
) -> list[set[int]] | None:
    covers = [set(spec["cover"]) for spec in row_specs]
    visible = [set() for _ in row_specs]
    min_vis = 3

    full_rows = [idx for idx, spec in enumerate(row_specs) if spec["kind"] == "full"]
    assert full_rows, "expected at least one full row"
    anchor_full = full_rows[0]
    visible[anchor_full].update({0, length - 1})

    for idx, spec in enumerate(row_specs):
        boundary_pos = spec["boundary_pos"]
        if boundary_pos is not None:
            visible[idx].add(boundary_pos)

    for pos in range(length):
        if any(pos in vis for vis in visible):
            continue
        eligible = [
            idx
            for idx, cover in enumerate(covers)
            if pos in cover and len(visible[idx]) < len(cover) - 1
        ]
        if not eligible:
            return None
        idx = min(eligible, key=lambda item: (len(visible[item]), random.random()))
        visible[idx].add(pos)

    for idx, cover in enumerate(covers):
        while len(visible[idx]) < min_vis:
            choices = [
                pos
                for pos in cover
                if pos not in visible[idx] and len(visible[idx]) < len(cover) - 1
            ]
            if not choices:
                return None
            visible[idx].add(random.choice(choices))

    for idx, cover in enumerate(covers):
        max_keep = len(cover) - 1
        extra_choices = [pos for pos in cover if pos not in visible[idx]]
        max_add = max_keep - len(visible[idx])
        if max_add > 0 and extra_choices:
            add_count = random.randint(0, max_add)
            if add_count:
                for pos in random.sample(extra_choices, add_count):
                    visible[idx].add(pos)
        if len(visible[idx]) >= len(cover):
            return None

    union = set().union(*visible)
    if union != set(range(length)):
        return None
    return visible


def _build_hidden_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.randint(0, 9)
    width = random.randint(12, 16)
    length = random.randint(6, min(9, width - 2))
    n_rows = random.randint(3, 5)
    height = random.randint(max(10, 2 * n_rows + 3), 18)
    master = _rand_master(length, bg)

    descriptors = [
        {"kind": "full", "mirrored": False},
        {"kind": "full", "mirrored": True},
    ]
    if n_rows > 2:
        descriptors.append({"kind": random.choice(["left", "right"]), "mirrored": random.choice([False, True])})
    while len(descriptors) < n_rows:
        descriptors.append(
            {
                "kind": random.choice(["full", "left", "right"]),
                "mirrored": random.choice([False, True]),
            }
        )
    random.shuffle(descriptors)

    rows = _pick_rows(height, n_rows)
    full_starts_used = []
    row_specs = []
    for row, spec in zip(rows, descriptors):
        kind = spec["kind"]
        if kind == "full":
            start = _full_start(width, length)
            if len(set(range(0, width - length + 1))) > 1:
                retries = 0
                while start in full_starts_used and retries < 10:
                    start = _full_start(width, length)
                    retries += 1
            full_starts_used.append(start)
        else:
            start = _clipped_start(width, length, kind)

        cover = _visible_positions(length, width, start, spec["mirrored"])
        if len(cover) < 4:
            return None
        row_specs.append(
            {
                "row": int(row),
                "kind": kind,
                "mirrored": bool(spec["mirrored"]),
                "start": int(start),
                "cover": tuple(int(pos) for pos in cover),
                "boundary_pos": _boundary_position(length, width, start, spec["mirrored"]),
            }
        )

    if not any(spec["mirrored"] for spec in row_specs) or not any(not spec["mirrored"] for spec in row_specs):
        return None

    visible_sets = _allocate_visible_sets(row_specs, length)
    if visible_sets is None:
        return None

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    for spec, visible in zip(row_specs, visible_sets):
        r = int(spec["row"])
        for pos in spec["cover"]:
            c = _master_to_col(length, int(spec["start"]), bool(spec["mirrored"]), int(pos))
            out[r, c] = int(master[pos])
        for pos in visible:
            c = _master_to_col(length, int(spec["start"]), bool(spec["mirrored"]), int(pos))
            inp[r, c] = int(master[pos])

    if np.array_equal(inp, out):
        return None
    if np.bincount(inp.ravel(), minlength=10)[bg] <= int(np.max(np.bincount(inp.ravel(), minlength=10)[[c for c in range(10) if c != bg]])):
        return None

    solved = _enumerate_outputs(inp, limit=2)
    if len(solved) != 1 or not np.array_equal(solved[0], out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _build_hidden_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid damaged-string input")
