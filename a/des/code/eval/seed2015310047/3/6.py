from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
FENCE_COLOR = 5
PALETTE = tuple(color for color in range(1, 10) if color != FENCE_COLOR)
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FENCE_COUNT_RANGE = (2, 5)
OUTER_H_RANGE = (6, 12)
OUTER_W_RANGE = (6, 12)
PLACEMENT_PADDING = 2
PLACEMENT_ATTEMPTS = 200
INTERIOR_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _side_slots(r0: int, c0: int, r1: int, c1: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown side: {side}")


def _is_contiguous_run(cells: list[tuple[int, int]], side: str) -> bool:
    if not cells:
        return False
    if side in ("top", "bottom"):
        coords = sorted(c for _, c in cells)
    else:
        coords = sorted(r for r, _ in cells)
    return all(b - a == 1 for a, b in zip(coords, coords[1:]))


def _gap_side(cells: set[tuple[int, int]], bbox_: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox_
    missing = _perimeter_cells(r0, c0, r1, c1) - cells
    if not (1 <= len(missing) <= 2):
        return None
    for side in SIDES:
        slots = set(_side_slots(r0, c0, r1, c1, side))
        if missing <= slots and _is_contiguous_run(sorted(missing), side):
            return side
    return None


def _voting_mask(shape: tuple[int, int], gap_side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if gap_side == "top":
        mask[h // 2 :, :] = True
    elif gap_side == "bottom":
        mask[: h // 2, :] = True
    elif gap_side == "left":
        mask[:, w // 2 :] = True
    elif gap_side == "right":
        mask[:, : w // 2] = True
    else:
        raise ValueError(f"unknown gap side: {gap_side}")
    return mask


def _majority_color(values: list[int]) -> int:
    counts = Counter(values)
    assert counts, "expected at least one voting pixel"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "voting colors must have a unique winner"
    return int(winners[0])


def _unique_winner(counts: Counter[int] | dict[int, int]) -> int | None:
    if not counts:
        return None
    best = max(int(count) for count in counts.values())
    winners = [int(color) for color, count in counts.items() if int(count) == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _find_fences(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    fences: list[dict[str, object]] = []
    for color in sorted({int(v) for v in g.flat if int(v) != BLACK}):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
                continue
            if len(comp) != len(set(comp)):
                continue
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(patch[1:-1, 1:-1] == color):
                continue

            side = _gap_side(set(comp), (r0, c0, r1, c1))
            if side is None:
                continue

            gap_cells = _perimeter_cells(r0, c0, r1, c1) - set(comp)
            if any(int(g[r, c]) != BLACK for r, c in gap_cells):
                continue

            interior = patch[1:-1, 1:-1]
            inside_colors = {int(v) for v in interior.flat if int(v) not in (BLACK, color)}
            if len(inside_colors) < 2:
                continue

            fences.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": int(color),
                    "gap_side": side,
                }
            )
    fences.sort(key=lambda fence: tuple(fence["bbox"]))
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gap-frame interior with the unique majority color from the half
    opposite its border gap, while keeping the frame and the gap unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fences = _find_fences(g)
    assert fences, "expected at least one gapped rectangular fence"

    out = g.copy()
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        frame_color = int(fence["color"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        mask = _voting_mask(interior.shape, str(fence["gap_side"]))
        voters = [
            int(value)
            for value in interior[mask].flat
            if int(value) not in (BLACK, frame_color)
        ]
        fill_color = _majority_color(voters)
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    return out


def _gap_positions(bbox_: tuple[int, int, int, int], side: str, rng: random.Random) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox_
    slots = _side_slots(r0, c0, r1, c1, side)
    gap_len = rng.choice([1, 2])
    gap_len = min(gap_len, len(slots))
    start = rng.randint(0, len(slots) - gap_len)
    return slots[start : start + gap_len]


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _shuffled_sides(count: int, rng: random.Random) -> list[str]:
    pool = list(SIDES)
    rng.shuffle(pool)
    if count <= len(pool):
        return pool[:count]
    extra = [rng.choice(SIDES) for _ in range(count - len(pool))]
    return pool + extra


def _voter_coords(shape: tuple[int, int], side: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mask = _voting_mask(shape, side)
    voters: list[tuple[int, int]] = []
    others: list[tuple[int, int]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if mask[r, c]:
                voters.append((r, c))
            else:
                others.append((r, c))
    return voters, others


def _allocate_vote_counts(
    colors: list[int], vote_count: int, rng: random.Random
) -> dict[int, int] | None:
    winner = colors[0]
    runner_up = colors[1]
    extras = colors[2:]
    counts = {color: 0 for color in colors}

    if len(colors) == 2:
        low = vote_count // 2 + 1
        high = vote_count - 1
    else:
        low = max(2, vote_count // 3 + 1)
        high = min(vote_count - 1, (vote_count + 1) // 2)
    if low > high:
        return None

    counts[winner] = rng.randint(low, high)
    remaining = vote_count - counts[winner]
    if remaining <= 0:
        return None

    counts[runner_up] = 1
    remaining -= 1
    recipients = [runner_up] + extras
    if remaining < 0 or not recipients:
        return None

    while remaining > 0:
        capped = [color for color in recipients if counts[color] < counts[winner] - 1]
        if not capped:
            return None
        chosen = rng.choice(capped)
        counts[chosen] += 1
        remaining -= 1

    best = max(counts.values())
    if sum(count == best for count in counts.values()) != 1 or counts[winner] != best:
        return None
    return counts


def _build_interior_patch(
    shape: tuple[int, int], gap_side: str, rng: random.Random
) -> tuple[np.ndarray, int] | None:
    voters, others = _voter_coords(shape, gap_side)
    if len(voters) < 4 or len(others) < 4:
        return None

    for _ in range(INTERIOR_ATTEMPTS):
        color_count = rng.randint(2, 4)
        colors = rng.sample(list(PALETTE), color_count)
        winner = colors[0]
        decoy = colors[1]
        extras = colors[2:]

        vote_count = rng.randint(max(4, len(voters) // 2), len(voters))
        other_count = rng.randint(max(4, (2 * len(others)) // 3), len(others))
        vote_counts = _allocate_vote_counts(colors, vote_count, rng)
        if vote_counts is None:
            continue

        a_other = rng.randint(0, min(2, other_count - 1 - len(extras)))
        other_counts = {color: 0 for color in colors}
        other_counts[winner] = a_other
        for color in extras:
            other_counts[color] = 1
        other_counts[decoy] = other_count - other_counts[winner] - sum(other_counts[color] for color in extras)
        if other_counts[decoy] <= 0:
            continue

        overall = Counter()
        overall.update(vote_counts)
        overall.update(other_counts)
        if _unique_winner(overall) != decoy:
            continue
        if max(other_counts.values()) != other_counts[decoy]:
            continue
        if sum(count == other_counts[decoy] for count in other_counts.values()) != 1:
            continue

        patch = np.zeros(shape, dtype=int)
        vote_positions = rng.sample(voters, vote_count)
        other_positions = rng.sample(others, other_count)

        vote_colors: list[int] = []
        for color, count in vote_counts.items():
            vote_colors.extend([color] * count)
        other_colors: list[int] = []
        for color, count in other_counts.items():
            other_colors.extend([color] * count)
        rng.shuffle(vote_colors)
        rng.shuffle(other_colors)

        for (r, c), color in zip(vote_positions, vote_colors):
            patch[r, c] = color
        for (r, c), color in zip(other_positions, other_colors):
            patch[r, c] = color

        return patch, winner

    return None


def _draw_fence(grid: np.ndarray, spec: dict[str, object], rng: random.Random) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    grid[r0, c0 : c1 + 1] = FENCE_COLOR
    grid[r1, c0 : c1 + 1] = FENCE_COLOR
    grid[r0 : r1 + 1, c0] = FENCE_COLOR
    grid[r0 : r1 + 1, c1] = FENCE_COLOR
    for r, c in _gap_positions(spec["bbox"], str(spec["gap_side"]), rng):
        grid[r, c] = BLACK


def _place_specs(rows: int, cols: int, rng: random.Random) -> list[dict[str, object]] | None:
    count = rng.randint(*FENCE_COUNT_RANGE)
    sides = _shuffled_sides(count, rng)
    specs: list[dict[str, object]] = []
    for side in sides:
        for _ in range(PLACEMENT_ATTEMPTS):
            height = rng.randint(*OUTER_H_RANGE)
            width = rng.randint(*OUTER_W_RANGE)
            if height >= rows - 2 or width >= cols - 2:
                continue
            r0 = rng.randint(1, rows - height - 1)
            c0 = rng.randint(1, cols - width - 1)
            bbox_ = (r0, c0, r0 + height - 1, c0 + width - 1)
            if any(_expanded_overlap(bbox_, tuple(spec["bbox"]), PLACEMENT_PADDING) for spec in specs):
                continue
            specs.append({"bbox": bbox_, "gap_side": side})
            break
        else:
            return None
    return specs


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(spec["winner"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(*GRID_H_RANGE)
    cols = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((rows, cols), dtype=int)

    specs = _place_specs(rows, cols, rng)
    if specs is None:
        return None

    for spec in specs:
        _draw_fence(grid, spec, rng)
        r0, c0, r1, c1 = spec["bbox"]
        patch_info = _build_interior_patch((r1 - r0 - 1, c1 - c0 - 1), str(spec["gap_side"]), rng)
        if patch_info is None:
            return None
        patch, winner = patch_info
        spec["winner"] = int(winner)
        grid[r0 + 1 : r1, c0 + 1 : c1] = patch

    expected = _expected_output(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-fence voting puzzle")
