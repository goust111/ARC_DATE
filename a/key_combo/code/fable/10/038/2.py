from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _ring_coords(r0: int, c0: int, r1: int, c1: int) -> list[Coord]:
    ring: list[Coord] = []
    for c in range(c0, c1 + 1):
        ring.append((r0, c))
    for r in range(r0 + 1, r1):
        ring.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        ring.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        ring.append((r, c0))
    return ring


def _positive_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    assert total >= parts >= 1
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    out: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _find_frame_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != bg)
    assert coords.size > 0, "expected foreground pixels"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have an interior"
    ring_set = set(_ring_coords(r0, c0, r1, c1))
    for rr, cc in coords:
        assert (int(rr), int(cc)) in ring_set, "all non-bg pixels must lie on the frame ring"
    return r0, c0, r1, c1


def _move_order(length: int, direction: int) -> list[int]:
    if direction == 1:
        return list(range(length))
    return list(range(length - 1, -1, -1))


def _rotate_events(events: list[tuple[int, str, int, int | None]]) -> list[tuple[int, str, int, int | None]]:
    balance = 0
    min_balance = 0
    cut = 0
    for i, (_rank, kind, _idx, _color) in enumerate(events):
        balance += 1 if kind == "H" else -1
        if balance < min_balance:
            min_balance = balance
            cut = i + 1
    return events[cut:] + events[:cut]


def _next_gap_by_rank(head_rank: int, gap_ranks: list[int]) -> int:
    for gr in gap_ranks:
        if gr > head_rank:
            return int(gr)
    return int(gap_ranks[0])


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    r0, c0, r1, c1 = _find_frame_bbox(g, bg)
    ring = _ring_coords(r0, c0, r1, c1)
    ring_vals = [int(g[r, c]) for r, c in ring]
    length = len(ring)

    non_bg_counts = Counter(v for v in ring_vals if v != bg)
    assert non_bg_counts, "expected non-background ring pixels"
    frame_color, frame_count = non_bg_counts.most_common(1)[0]
    assert frame_count >= 4, "frame color must dominate the ring"

    gap_indices = [i for i, v in enumerate(ring_vals) if v == bg]
    assert gap_indices, "expected at least one gap"

    special_counts = Counter(v for v in ring_vals if v not in {bg, frame_color})
    assert special_counts, "expected colored marker cells"
    tail_candidates = [color for color, count in special_counts.items() if count > 1]
    assert len(tail_candidates) == 1, "expected a unique repeated tail color"
    tail_color = int(tail_candidates[0])

    head_indices_by_tail: dict[int, int] = {}
    directions: set[int] = set()
    head_colors: dict[int, int] = {}
    for idx, value in enumerate(ring_vals):
        if value != tail_color:
            continue
        prev_idx = (idx - 1) % length
        next_idx = (idx + 1) % length
        prev_is_head = ring_vals[prev_idx] not in {bg, frame_color, tail_color}
        next_is_head = ring_vals[next_idx] not in {bg, frame_color, tail_color}
        assert prev_is_head ^ next_is_head, "each tail must touch exactly one head"
        if next_is_head:
            directions.add(1)
            head_idx = next_idx
        else:
            directions.add(-1)
            head_idx = prev_idx
        head_indices_by_tail[idx] = head_idx
        head_colors[head_idx] = int(ring_vals[head_idx])

    assert len(directions) == 1, "all markers must share one head-tail direction"
    direction = int(next(iter(directions)))
    head_indices = sorted(head_colors)
    assert len(head_indices) == len(gap_indices), "marker count must equal gap count"
    assert len(set(head_colors.values())) == len(head_indices), "head colors must be distinct"

    special_positions = {
        i for i, v in enumerate(ring_vals) if v not in {bg, frame_color}
    }
    expected_positions = set(head_indices_by_tail) | set(head_indices)
    assert special_positions == expected_positions, "unexpected special cells on the ring"

    move_order = _move_order(length, direction)
    rank = {idx: pos for pos, idx in enumerate(move_order)}
    gap_ranks = sorted(rank[idx] for idx in gap_indices)

    events: list[tuple[int, str, int, int | None]] = []
    first_gap_by_head: dict[int, int] = {}
    for head_idx in head_indices:
        hr = rank[head_idx]
        events.append((hr, "H", head_idx, head_colors[head_idx]))
        next_gap_rank = _next_gap_by_rank(hr, gap_ranks)
        first_gap_by_head[head_idx] = move_order[next_gap_rank]
    for gap_idx in gap_indices:
        events.append((rank[gap_idx], "G", gap_idx, None))
    events.sort(key=lambda item: int(item[0]))

    rotated_events = _rotate_events(events)
    stack: list[tuple[int, int]] = []
    assigned_gap_by_head: dict[int, int] = {}
    for _rank, kind, idx, color in rotated_events:
        if kind == "H":
            assert color is not None
            stack.append((idx, int(color)))
            continue
        assert stack, "cyclic queue matching must have a pending marker at each served gap"
        head_idx, _head_color = stack.pop()
        assigned_gap_by_head[head_idx] = idx

    assert not stack, "all markers must be assigned"
    assert len(assigned_gap_by_head) == len(head_indices)

    contender_counts = Counter(first_gap_by_head.values())
    assert max(contender_counts.values()) >= 2, "need at least one queued first-gap collision"
    assert any(
        assigned_gap_by_head[head_idx] != first_gap_by_head[head_idx]
        for head_idx in head_indices
    ), "at least one marker must skip an occupied first gap"

    return {
        "bg": bg,
        "bbox": (r0, c0, r1, c1),
        "ring": ring,
        "ring_vals": ring_vals,
        "frame_color": int(frame_color),
        "tail_color": tail_color,
        "direction": direction,
        "gap_indices": gap_indices,
        "head_indices": head_indices,
        "head_colors": head_colors,
        "first_gap_by_head": first_gap_by_head,
        "assigned_gap_by_head": assigned_gap_by_head,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each two-cell marker around the frame so queued markers fill successive gaps with their head colors.

    Input structure:
    - A one-cell-thick hollow rectangular frame sits on the background, with a single frame color.
    - Some ring cells are missing as one-cell background gaps.
    - Other ring cells form markers: each marker occupies two adjacent ring cells, with a unique colored head and a shared tail color; all markers face the same cyclic direction around the ring.

    Rule:
    - Read the common travel direction from the head-tail orientation.
    - Each marker moves forward along the ring toward the first gap ahead of its head.
    - If a nearer marker already occupies that gap, the later marker skips it and continues to the next gap, until all gaps are filled one-to-one.
    - Replace every original marker cell with the frame color, and paint each claimed gap cell with its marker's head color.

    Key insight:
    - The pairing is not plain nearest-gap matching: in each direction-ordered queue, the closest head to a gap claims it first and earlier heads spill forward to later gaps.
    """
    meta = _analyze_grid(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    ring = meta["ring"]
    frame_color = int(meta["frame_color"])
    for r, c in ring:
        out[r, c] = frame_color

    for head_idx, gap_idx in meta["assigned_gap_by_head"].items():
        color = int(meta["head_colors"][head_idx])
        rr, cc = ring[gap_idx]
        out[rr, cc] = color

    return out


def _sample_counts(n_markers: int, rng: random.Random) -> list[int]:
    while True:
        counts = [0] * n_markers
        counts[rng.randrange(n_markers)] = 2
        for _ in range(n_markers - 2):
            counts[rng.randrange(n_markers)] += 1
        if max(counts) > 3:
            continue
        if sum(counts) != n_markers or max(counts) < 2:
            continue
        return counts


def _build_token_sequence(length: int, counts: list[int], rng: random.Random) -> list[tuple[str, int | None]]:
    n_markers = sum(counts)
    free_cells = length - 3 * n_markers
    assert free_cells >= 2 * n_markers
    pads = _positive_composition(free_cells, 2 * n_markers, rng)
    gap_pads = pads[:n_markers]
    marker_pads = pads[n_markers:]

    tokens: list[tuple[str, int | None]] = []
    marker_id = 0
    for gap_id, marker_count in enumerate(counts):
        tokens.append(("G", gap_id))
        tokens.extend([("F", None)] * gap_pads[gap_id])
        for _ in range(marker_count):
            tokens.append(("T", marker_id))
            tokens.append(("H", marker_id))
            tokens.extend([("F", None)] * marker_pads[marker_id])
            marker_id += 1

    assert marker_id == n_markers
    assert len(tokens) == length
    return tokens


def _corner_indices_for_bbox(r0: int, c0: int, r1: int, c1: int) -> set[int]:
    ring = _ring_coords(r0, c0, r1, c1)
    corners = {(r0, c0), (r0, c1), (r1, c1), (r1, c0)}
    return {i for i, coord in enumerate(ring) if coord in corners}


def _embed_tokens(
    tokens: list[tuple[str, int | None]],
    direction: int,
    corner_indices: set[int],
    rng: random.Random,
) -> dict | None:
    length = len(tokens)
    move_order = _move_order(length, direction)
    shifts = list(range(length))
    rng.shuffle(shifts)

    for shift in shifts:
        placement: dict[str, dict[int, int]] = {"G": {}, "T": {}, "H": {}}
        ok = True
        for pos, (kind, ident) in enumerate(tokens):
            cw_idx = move_order[(shift + pos) % length]
            if kind in {"G", "T", "H"}:
                assert ident is not None
                if cw_idx in corner_indices:
                    ok = False
                    break
                placement[kind][int(ident)] = cw_idx
        if ok:
            return placement
    return None


def _make_candidate(rng: random.Random) -> np.ndarray | None:
    n_markers = rng.choice([3, 4, 5])
    direction = rng.choice([-1, 1])

    valid_sizes: list[tuple[int, int]] = []
    for frame_h in range(6, 13):
        for frame_w in range(6, 13):
            ring_len = 2 * (frame_h + frame_w) - 4
            if ring_len >= 5 * n_markers:
                valid_sizes.append((frame_h, frame_w))
    frame_h, frame_w = rng.choice(valid_sizes)

    top_margin = rng.randint(1, 4)
    left_margin = rng.randint(1, 4)
    bottom_margin = rng.randint(1, 4)
    right_margin = rng.randint(1, 4)
    height = frame_h + top_margin + bottom_margin
    width = frame_w + left_margin + right_margin
    if height > 30 or width > 30:
        return None

    colors = rng.sample(list(range(10)), n_markers + 3)
    bg = colors[0]
    frame_color = colors[1]
    tail_color = colors[2]
    head_palette = colors[3:]

    grid = np.full((height, width), bg, dtype=int)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    ring = _ring_coords(r0, c0, r1, c1)
    ring_len = len(ring)
    corner_indices = _corner_indices_for_bbox(r0, c0, r1, c1)

    counts = _sample_counts(n_markers, rng)
    tokens = _build_token_sequence(ring_len, counts, rng)
    placement = _embed_tokens(tokens, direction, corner_indices, rng)
    if placement is None:
        return None

    for r, c in ring:
        grid[r, c] = frame_color

    gap_positions = placement["G"]
    tail_positions = placement["T"]
    head_positions = placement["H"]
    if not (len(gap_positions) == len(tail_positions) == len(head_positions) == n_markers):
        return None

    for marker_id, cw_idx in gap_positions.items():
        rr, cc = ring[cw_idx]
        grid[rr, cc] = bg

    for marker_id, cw_idx in tail_positions.items():
        rr, cc = ring[cw_idx]
        grid[rr, cc] = tail_color

    for marker_id, cw_idx in head_positions.items():
        rr, cc = ring[cw_idx]
        grid[rr, cc] = head_palette[marker_id]

    if _get_bg(grid) != bg:
        return None

    try:
        meta = _analyze_grid(grid)
    except AssertionError:
        return None

    out = main(grid)
    out_ring_vals = [int(out[r, c]) for r, c in meta["ring"]]
    if meta["bg"] in out_ring_vals:
        return None
    if meta["tail_color"] in out_ring_vals:
        return None
    if Counter(meta["first_gap_by_head"].values()).most_common(1)[0][1] < 2:
        return None
    if not any(
        meta["assigned_gap_by_head"][head_idx] != meta["first_gap_by_head"][head_idx]
        for head_idx in meta["head_indices"]
    ):
        return None
    if len(set(out_ring_vals)) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _make_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-gap repair puzzle")
