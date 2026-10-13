import random
import sys
from pathlib import Path

import numpy as np


UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, paste


TRANSFORMS = ("cw90", "rot180", "flip_lr", "flip_ud")
SPRITE_BACKGROUND = Color.BLACK
DIVIDER_COLOR = Color.GREY
PALETTE = tuple(c for c in Color.NOT_BLACK if c != DIVIDER_COLOR)
MAX_GENERATION_ATTEMPTS = 400


def _apply_transform(slot: np.ndarray, name: str) -> np.ndarray:
    if name == "cw90":
        return np.rot90(slot, k=-1)
    if name == "rot180":
        return np.rot90(slot, k=2)
    if name == "flip_lr":
        return np.fliplr(slot)
    if name == "flip_ud":
        return np.flipud(slot)
    raise ValueError(f"unknown transform {name}")


def _apply_inverse_transform(slot: np.ndarray, name: str) -> np.ndarray:
    if name == "cw90":
        return np.rot90(slot, k=1)
    return _apply_transform(slot, name)


def _apply_transform_steps(slot: np.ndarray, name: str, steps: int) -> np.ndarray:
    out = np.asarray(slot, dtype=int).copy()
    for _ in range(int(steps)):
        out = _apply_transform(out, name)
    return out


def _parse_slots(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    divider_cols = [c for c in range(w) if np.all(g[:, c] == DIVIDER_COLOR)]
    assert divider_cols, "expected vertical divider columns"
    spans: list[tuple[int, int]] = []
    start = 0
    for c in range(w + 1):
        is_divider = c < w and c in divider_cols
        if is_divider:
            if start < c:
                spans.append((start, c))
            start = c + 1
        elif c == w and start < c:
            spans.append((start, c))
    assert len(spans) >= 3, "expected at least three slots"
    widths = [c1 - c0 for c0, c1 in spans]
    assert len(set(widths)) == 1, "slots must have equal width"
    assert widths[0] == h, "slots must be square"
    for c in divider_cols:
        assert c not in {cc for c0, c1 in spans for cc in range(c0, c1)}
    return spans


def _background_color(grid: np.ndarray) -> int:
    values = grid[grid != DIVIDER_COLOR]
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _slot_arrays(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    spans = _parse_slots(grid)
    bg = _background_color(grid)
    slots = [grid[:, c0:c1].copy() for c0, c1 in spans]
    for slot in slots:
        assert slot.shape[0] == slot.shape[1]
        assert DIVIDER_COLOR not in set(map(int, slot.flat))
    return slots, bg


def _infer_transform(left_slot: np.ndarray, right_slot: np.ndarray, steps: int) -> str:
    matches = [name for name in TRANSFORMS if np.array_equal(_apply_transform_steps(left_slot, name, steps), right_slot)]
    assert len(matches) == 1, f"ambiguous generator inference: {matches}"
    return matches[0]


def _compose_row(slots: list[np.ndarray]) -> np.ndarray:
    size = int(slots[0].shape[0])
    n_slots = len(slots)
    out = np.full((size, n_slots * size + (n_slots - 1)), SPRITE_BACKGROUND, dtype=int)
    for idx, slot in enumerate(slots):
        paste(out, np.asarray(slot, dtype=int), 0, idx * (size + 1))
        if idx < n_slots - 1:
            out[:, (idx + 1) * size + idx] = DIVIDER_COLOR
    return out


def _filled_slot_indices(slots: list[np.ndarray], bg: int) -> list[int]:
    return [idx for idx, slot in enumerate(slots) if np.any(slot != bg)]


def _solve_row(grid: np.ndarray) -> tuple[np.ndarray, str, tuple[int, int], int]:
    slots, bg = _slot_arrays(grid)
    filled = _filled_slot_indices(slots, bg)
    assert len(filled) == 2, "expected exactly two filled slots"
    left_idx, right_idx = filled
    assert left_idx < right_idx
    steps = right_idx - left_idx
    transform = _infer_transform(slots[left_idx], slots[right_idx], steps)

    solved = [slot.copy() for slot in slots]
    for idx in range(left_idx + 1, len(solved)):
        solved[idx] = _apply_transform(solved[idx - 1], transform)
    for idx in range(left_idx - 1, -1, -1):
        solved[idx] = _apply_inverse_transform(solved[idx + 1], transform)
    return _compose_row(solved), transform, (left_idx, right_idx), bg


def _random_connected_cells(size: int, count: int) -> list[tuple[int, int]]:
    start = (random.randrange(size), random.randrange(size))
    cells = {start}
    frontier = {start}
    while len(cells) < count:
        r, c = random.choice(tuple(frontier))
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in cells:
                nbrs.append((rr, cc))
        if not nbrs:
            frontier.remove((r, c))
            if not frontier:
                frontier = set(cells)
            continue
        cell = random.choice(nbrs)
        cells.add(cell)
        frontier.add(cell)
    return list(cells)


def _is_valid_base_slot(slot: np.ndarray) -> bool:
    if not np.any(slot != SPRITE_BACKGROUND):
        return False
    seen = []
    for variant in (
        slot,
        np.rot90(slot, k=-1),
        np.rot90(slot, k=2),
        np.rot90(slot, k=1),
        np.fliplr(slot),
        np.flipud(slot),
    ):
        if any(np.array_equal(variant, prior) for prior in seen):
            return False
        seen.append(variant.copy())
    return True


def _make_base_slot(size: int) -> np.ndarray:
    min_cells = max(5, size)
    max_cells = min(size * size - 2, size + 4)
    for _ in range(300):
        count = random.randint(min_cells, max_cells)
        cells = _random_connected_cells(size, count)
        slot = np.full((size, size), SPRITE_BACKGROUND, dtype=int)
        palette = random.sample(PALETTE, k=random.randint(2, min(4, len(PALETTE))))
        for r, c in cells:
            slot[r, c] = random.choice(palette)
        anchors = random.sample(cells, k=min(3, len(cells)))
        for idx, (r, c) in enumerate(anchors):
            slot[r, c] = palette[idx % len(palette)]
        if np.count_nonzero(slot != SPRITE_BACKGROUND) < min_cells:
            continue
        if _is_valid_base_slot(slot):
            return slot
    raise RuntimeError("failed to generate an asymmetric base slot")


def _choose_layout() -> tuple[int, int]:
    choices = []
    for size in range(4, 7):
        for n_slots in range(4, 7):
            width = n_slots * size + (n_slots - 1)
            if size <= 30 and width <= 30:
                choices.append((size, n_slots))
    return random.choice(choices)


def _choose_transform_and_gap(n_slots: int) -> tuple[str, int]:
    transform = random.choices(
        population=TRANSFORMS,
        weights=(4, 2, 2, 2),
        k=1,
    )[0]
    if transform == "cw90":
        gaps = [gap for gap in range(2, n_slots) if gap % 4 != 0]
        preferred = [gap for gap in gaps if gap in (2, 3)]
        return transform, random.choice(preferred or gaps)
    gaps = [gap for gap in range(2, n_slots) if gap % 2 == 1]
    return transform, random.choice(gaps)


def _make_full_sequence(base_slot: np.ndarray, transform: str, n_slots: int) -> list[np.ndarray]:
    seq = [base_slot.copy()]
    for _ in range(1, n_slots):
        seq.append(_apply_transform(seq[-1], transform))
    return seq


def _semantic_signature(grid: np.ndarray) -> tuple:
    slots, bg = _slot_arrays(grid)
    filled = _filled_slot_indices(slots, bg)
    return (grid.shape, len(slots), tuple(filled), tuple(int(np.count_nonzero(slot != bg)) for slot in slots))


def _validate_generated_case(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30
    assert max(out.shape) <= 30

    solved, transform, (left_idx, right_idx), bg = _solve_row(inp)
    assert np.array_equal(out, solved)

    in_slots, _ = _slot_arrays(inp)
    out_slots, _ = _slot_arrays(out)
    filled = _filled_slot_indices(in_slots, bg)
    assert filled == [left_idx, right_idx]
    assert len(_filled_slot_indices(out_slots, bg)) == len(out_slots)

    steps = right_idx - left_idx
    matches = [name for name in TRANSFORMS if np.array_equal(_apply_transform_steps(in_slots[left_idx], name, steps), in_slots[right_idx])]
    assert matches == [transform]

    for idx in range(1, len(out_slots)):
        assert np.array_equal(out_slots[idx], _apply_transform(out_slots[idx - 1], transform))


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Infer the constant slot-to-slot transform from two observed poses, then fill the entire slot row.

    Input structure:
    - Gray full-height divider columns split the grid into one row of equal square slots.
    - Exactly two slots contain the same non-symmetric sprite on a black background; all other slots are empty.
    - The two filled slots may be separated by multiple empty slots.

    Rule:
    - Parse the ordered slots between gray dividers.
    - Let the filled slots be k steps apart; among {clockwise 90 degrees, 180 degrees, left-right mirror, up-down mirror}, find the unique transform T whose k repeated applications map the left filled slot to the right filled slot.
    - Propagate T to the right and T inverse to the left until every slot is filled, keeping the divider columns unchanged.

    Key insight:
    - The observed pair does not necessarily show adjacent slots, so the task is to solve a transform-power equation T^k from the slot distance before completing the full sequence.
    """
    solved, _transform, _pair, _bg = _solve_row(np.asarray(input_grid, dtype=int))
    return solved


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        size, n_slots = _choose_layout()
        transform, gap = _choose_transform_and_gap(n_slots)
        left_idx = random.randint(0, n_slots - gap - 1)
        right_idx = left_idx + gap

        base = _make_base_slot(size)
        full_sequence = _make_full_sequence(base, transform, n_slots)
        assert np.array_equal(full_sequence[right_idx], _apply_transform_steps(full_sequence[left_idx], transform, gap))

        empty_slot = np.full((size, size), SPRITE_BACKGROUND, dtype=int)
        visible_slots = [empty_slot.copy() for _ in range(n_slots)]
        visible_slots[left_idx] = full_sequence[left_idx]
        visible_slots[right_idx] = full_sequence[right_idx]
        inp = _compose_row(visible_slots)
        out = _compose_row(full_sequence)

        if len({slot.tobytes() for slot in full_sequence}) < 2:
            continue
        if np.array_equal(inp, out):
            continue
        try:
            _validate_generated_case(inp, out)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid slot-transform puzzle input")
