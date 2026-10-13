from __future__ import annotations

import itertools

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5

GRID_H = 5
SLOT_W = 3
SLOT_COUNT = 6
GRID_W = SLOT_COUNT * SLOT_W + (SLOT_COUNT - 1)
TOP_ROW = 0
BOTTOM_ROW = GRID_H - 1
CONTENT_ROWS = (1, 2, 3)

SLOT_BANDS = [(i * (SLOT_W + 1), i * (SLOT_W + 1) + SLOT_W - 1) for i in range(SLOT_COUNT)]
SEP_COLS = [c1 + 1 for _c0, c1 in SLOT_BANDS[:-1]]


def _minimal_period(bits: tuple[int, ...]) -> int:
    n = len(bits)
    for p in range(1, n):
        if n % p == 0 and bits == bits[:p] * (n // p):
            return p
    return n


def _periodic_patterns(period: int) -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    for bits in itertools.product((0, 1), repeat=period):
        if 0 not in bits or 1 not in bits:
            continue
        if _minimal_period(bits) != period:
            continue
        patterns.append(tuple(map(int, bits)))
    return patterns


def _slot_view(grid: np.ndarray, slot_idx: int) -> np.ndarray:
    c0, c1 = SLOT_BANDS[slot_idx]
    return grid[:, c0 : c1 + 1]


def _parse_grid(grid: np.ndarray) -> tuple[list[np.ndarray], tuple[tuple[int, int], ...], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W), f"expected {GRID_H}x{GRID_W}, got {g.shape}"
    assert np.all(g[TOP_ROW, :] == GREY), "top border must be grey"
    assert np.all(g[BOTTOM_ROW, :] == GREY), "bottom border must be grey"
    assert np.all(g[:, SEP_COLS] == GREY), "internal separator columns must be grey"
    assert not np.any(g == BLUE), "input must not contain blue"

    allowed = {BG, RED, GREY}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input colors"
    for c in range(GRID_W):
        if c not in SEP_COLS:
            assert np.all(g[1:4, c] != GREY), "grey may only appear on outer rows or separator columns"

    slots = [_slot_view(g, idx).copy() for idx in range(SLOT_COUNT)]
    red_masks = [tuple(map(tuple, np.argwhere(slot == RED))) for slot in slots]
    nonempty = [mask for mask in red_masks if mask]
    assert nonempty, "at least one slot must contain the sprite"
    sprite_coords = nonempty[0]
    for mask in nonempty:
        assert mask == sprite_coords, "all present sprite copies must be identical"
    for r, c in sprite_coords:
        assert int(r) in CONTENT_ROWS, "sprite must lie inside the slot body"
        assert 0 <= int(c) < SLOT_W

    seq = [1 if mask else 0 for mask in red_masks]
    assert 0 in seq and 1 in seq, "sequence must contain both present and empty slots"
    return slots, tuple((int(r), int(c)) for r, c in sprite_coords), seq


def _solve_sequence(seq: list[int]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for period in (2, 3):
        for base in _periodic_patterns(period):
            expected = [int(base[i % period]) for i in range(len(seq))]
            mismatches = [i for i, (a, b) in enumerate(zip(seq, expected)) if a != b]
            if len(mismatches) != 1:
                continue
            idx = mismatches[0]
            candidates.append(
                {
                    "period": period,
                    "base": base,
                    "expected": expected,
                    "idx": idx,
                    "kind": "missing" if seq[idx] == 0 else "extra",
                }
            )
    assert len(candidates) == 1, f"expected a unique periodic explanation, got {candidates}"
    return candidates[0]


def _repair_slot(slot: np.ndarray, sprite_coords: tuple[tuple[int, int], ...], should_have_sprite: bool) -> np.ndarray:
    out = slot.copy()
    out[(out == RED)] = BG
    if should_have_sprite:
        for r, c in sprite_coords:
            out[r, c] = RED
    return out


def _highlight_slot(out: np.ndarray, slot_idx: int) -> None:
    c0, c1 = SLOT_BANDS[slot_idx]
    out[TOP_ROW, c0 : c1 + 1] = BLUE
    out[BOTTOM_ROW, c0 : c1 + 1] = BLUE

    left_col = c0 - 1 if c0 > 0 and np.all(out[:, c0 - 1] == GREY) else c0
    right_col = c1 + 1 if c1 + 1 < GRID_W and np.all(out[:, c1 + 1] == GREY) else c1
    out[:, left_col] = BLUE
    out[:, right_col] = BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the single slot that breaks a binary sprite-appearance period and mark that slot with a blue frame.

    Input structure:
    - The grid is a 5x23 black canvas with grey top/bottom border rows and five full-height
      grey separator columns that split it into six 5x3 slots.
    - A fixed red sprite appears in some slots at one shared local position, while other slots
      are empty except for the grey border rows.
    - Across the six slots, sprite presence/absence follows a period-2 or period-3 binary pattern
      except at exactly one slot, which is either missing the sprite or contains one extra copy.

    Rule:
    - Read the six-slot presence sequence and test every strict binary period of length 2 and 3.
    - Find the unique periodic sequence that differs from the observed one in exactly one slot.
    - Repair that slot by stamping in the missing red sprite or erasing the extra one, then mark
      the repaired slot's border blue using the adjacent separator columns and the slot's top/bottom edge.

    Key insight:
    - The anomaly is not the minority state of a slot; it is the lone slot whose presence bit breaks
      an otherwise repeatable period, so period inference and anomaly localization must validate each other.
    """
    g = np.asarray(grid, dtype=int)
    slots, sprite_coords, seq = _parse_grid(g)
    solution = _solve_sequence(seq)
    idx = int(solution["idx"])
    expected = solution["expected"]

    out = g.copy()
    c0, c1 = SLOT_BANDS[idx]
    repaired = _repair_slot(slots[idx], sprite_coords, bool(expected[idx]))
    out[:, c0 : c1 + 1] = repaired
    _highlight_slot(out, idx)
    return out


SHAPES = [
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
]

GEN_PATTERNS = {
    2: _periodic_patterns(2),
    3: _periodic_patterns(3),
}

GEN_RETRY_LIMIT = 400


def _shift_shape(
    shape: tuple[tuple[int, int], ...], anchor_r: int, anchor_c: int
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((anchor_r + dr, anchor_c + dc) for dr, dc in shape))


def _choose_sprite(anomaly_idx: int) -> tuple[tuple[int, int], ...]:
    anchor_r = int(np.random.choice([1, 2]))
    if anomaly_idx == 0:
        anchor_c = 1
    elif anomaly_idx == SLOT_COUNT - 1:
        anchor_c = 0
    else:
        anchor_c = int(np.random.choice([0, 1]))
    shape = SHAPES[int(np.random.randint(len(SHAPES)))]
    return _shift_shape(shape, anchor_r, anchor_c)


def _render_grid(seq: list[int], sprite_coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_ROW, :] = GREY
    grid[BOTTOM_ROW, :] = GREY
    grid[:, SEP_COLS] = GREY
    for idx, bit in enumerate(seq):
        if not bit:
            continue
        c0, _c1 = SLOT_BANDS[idx]
        for r, c in sprite_coords:
            grid[r, c0 + c] = RED
    return grid


def _build_expected(
    ideal_seq: list[int], anomaly_idx: int, sprite_coords: tuple[tuple[int, int], ...]
) -> np.ndarray:
    out = _render_grid(ideal_seq, sprite_coords)
    _highlight_slot(out, anomaly_idx)
    return out


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    ideal_seq: list[int],
    observed_seq: list[int],
    period: int,
    anomaly_idx: int,
    sprite_coords: tuple[tuple[int, int], ...],
) -> None:
    slots, parsed_sprite, parsed_seq = _parse_grid(inp)
    assert parsed_sprite == sprite_coords
    assert parsed_seq == observed_seq
    assert sum(int(a != b) for a, b in zip(observed_seq, ideal_seq)) == 1
    assert observed_seq[anomaly_idx] != ideal_seq[anomaly_idx]
    assert 0 in ideal_seq and 1 in ideal_seq
    assert _minimal_period(tuple(ideal_seq[:period])) == period

    solutions = []
    for cand_period in (2, 3):
        for base in GEN_PATTERNS[cand_period]:
            cand = [int(base[i % cand_period]) for i in range(SLOT_COUNT)]
            mismatch = [i for i, (a, b) in enumerate(zip(observed_seq, cand)) if a != b]
            if len(mismatch) == 1:
                solutions.append((cand_period, base, mismatch[0], tuple(cand)))
    assert len(solutions) == 1
    only_period, _only_base, only_idx, only_seq = solutions[0]
    assert only_period == period
    assert only_idx == anomaly_idx
    assert list(only_seq) == ideal_seq

    repaired = main(inp)
    assert np.array_equal(repaired, expected)

    c0, c1 = SLOT_BANDS[anomaly_idx]
    slot_after = repaired[:, c0 : c1 + 1]
    if ideal_seq[anomaly_idx]:
        for r, c in sprite_coords:
            assert slot_after[r, c] == RED or (c in (0, SLOT_W - 1) and anomaly_idx in (0, SLOT_COUNT - 1))
    else:
        assert not np.any(slot_after == RED)

    for idx in range(SLOT_COUNT):
        if idx == anomaly_idx:
            continue
        s0, s1 = SLOT_BANDS[idx]
        assert np.array_equal(repaired[:, s0 : s1 + 1], inp[:, s0 : s1 + 1])


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        period = int(np.random.choice([2, 3]))
        base = GEN_PATTERNS[period][int(np.random.randint(len(GEN_PATTERNS[period])))]
        ideal_seq = [int(base[i % period]) for i in range(SLOT_COUNT)]

        kind = str(np.random.choice(["missing", "extra"]))
        valid_indices = [i for i, bit in enumerate(ideal_seq) if (bit == 1) == (kind == "missing")]
        anomaly_idx = int(np.random.choice(valid_indices))
        observed_seq = ideal_seq.copy()
        observed_seq[anomaly_idx] = 1 - observed_seq[anomaly_idx]

        sprite_coords = _choose_sprite(anomaly_idx)
        inp = _render_grid(observed_seq, sprite_coords)
        expected = _build_expected(ideal_seq, anomaly_idx, sprite_coords)

        try:
            _validate_semantics(
                inp=inp,
                expected=expected,
                ideal_seq=ideal_seq,
                observed_seq=observed_seq,
                period=period,
                anomaly_idx=anomaly_idx,
                sprite_coords=sprite_coords,
            )
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a non-degenerate periodic anomaly instance")
