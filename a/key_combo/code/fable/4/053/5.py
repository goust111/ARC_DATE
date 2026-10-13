from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PLACEHOLDER = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            out.append(comp)
    return out


def _ring_color(patch: np.ndarray, layer: int) -> int:
    side = int(patch.shape[0])
    end = side - layer
    cells = []
    for c in range(layer, end):
        cells.append(int(patch[layer, c]))
        cells.append(int(patch[end - 1, c]))
    for r in range(layer + 1, end - 1):
        cells.append(int(patch[r, layer]))
        cells.append(int(patch[r, end - 1]))
    colors = set(cells)
    assert len(colors) == 1
    return cells[0]


def _paint_ring(patch: np.ndarray, layer: int, color: int) -> None:
    side = int(patch.shape[0])
    end = side - layer
    patch[layer, layer:end] = color
    patch[end - 1, layer:end] = color
    patch[layer:end, layer] = color
    patch[layer:end, end - 1] = color


def _make_target(sequence: tuple[int, ...], missing_idx: int | None) -> np.ndarray:
    side = 2 * len(sequence) - 1
    patch = np.full((side, side), BG, dtype=int)
    for layer, color in enumerate(sequence):
        _paint_ring(patch, layer, PLACEHOLDER if missing_idx == layer else int(color))
    return patch


def _extract_targets(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    targets = []
    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0) == (c1 - c0)
        side = int(r1 - r0 + 1)
        assert side % 2 == 1
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        layers = (side + 1) // 2
        sequence: list[int | None] = []
        missing_idx = None
        for layer in range(layers):
            color = _ring_color(patch, layer)
            if color == PLACEHOLDER:
                assert missing_idx is None
                missing_idx = layer
                sequence.append(None)
            else:
                sequence.append(int(color))
        assert missing_idx is not None
        targets.append(
            {
                "bbox": (r0, c0, r1, c1),
                "sequence": tuple(sequence),
                "missing_idx": int(missing_idx),
            }
        )
    assert targets
    targets.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return targets


def _merge_known(
    known: dict[int, int], sequence: tuple[int | None, ...], start: int
) -> dict[int, int] | None:
    merged = dict(known)
    for idx, color in enumerate(sequence):
        if color is None:
            continue
        pos = int(start + idx)
        prev = merged.get(pos)
        if prev is not None and prev != color:
            return None
        merged[pos] = int(color)
    return merged


def _candidate_shift(seq_a: tuple[int | None, ...], seq_b: tuple[int | None, ...]) -> list[int]:
    out = []
    len_a = len(seq_a)
    len_b = len(seq_b)
    for delta in range(-(len_b - 1), len_a):
        overlaps = False
        ok = True
        for idx_a, color_a in enumerate(seq_a):
            idx_b = idx_a - delta
            if not (0 <= idx_b < len_b):
                continue
            overlaps = True
            color_b = seq_b[idx_b]
            if color_a is not None and color_b is not None and color_a != color_b:
                ok = False
                break
        if overlaps and ok:
            out.append(int(delta))
    return out


def _infer_missing_colors(sequences: list[tuple[int | None, ...]]) -> tuple[int, ...] | None:
    assert sequences
    missing_positions = [seq.index(None) for seq in sequences]
    lengths = [len(seq) for seq in sequences]
    span_limit = sum(lengths)
    results: set[tuple[int, ...]] = set()

    def record(starts: dict[int, int], known: dict[int, int]) -> None:
        colors = []
        for idx, miss in enumerate(missing_positions):
            color = known.get(int(starts[idx] + miss))
            if color is None:
                return
            colors.append(int(color))
        results.add(tuple(colors))

    def dfs(starts: dict[int, int], known: dict[int, int]) -> None:
        if len(results) > 1:
            return
        if len(starts) == len(sequences):
            record(starts, known)
            return

        frontier = []
        for idx in range(len(sequences)):
            if idx in starts:
                continue
            valid = []
            for start in range(span_limit - lengths[idx] + 1):
                merged = _merge_known(known, sequences[idx], start)
                if merged is not None:
                    valid.append((start, merged))
            if valid:
                frontier.append((len(valid), -len(sequences[idx]), idx, valid))

        if not frontier:
            return

        best_count = min(item[0] for item in frontier)
        candidates = [item for item in frontier if item[0] == best_count]
        for _count, _neg_len, idx, valid in candidates:
            for start, merged in valid:
                starts[idx] = int(start)
                dfs(starts, merged)
                del starts[idx]
                if len(results) > 1:
                    return

    for anchor in range(len(sequences)):
        initial_known = _merge_known({}, sequences[anchor], 0)
        assert initial_known is not None
        dfs({anchor: 0}, initial_known)
        if len(results) > 1:
            return None
    if len(results) != 1:
        return None
    return next(iter(results))


def main(grid: np.ndarray) -> np.ndarray:
    """Restore each grey ring layer so every target becomes a contiguous window of one hidden mother spectrum.

    Input structure:
    - A background grid contains several separated odd-sized concentric square targets.
    - Every target has one-cell-thick monochrome rings read from outermost to innermost.
    - Exactly one ring per target is replaced by a full grey placeholder layer.

    Rule:
    - Convert each target into its outer-to-inner color sequence, treating the grey ring as unknown.
    - Find a globally consistent alignment where all targets are overlapping contiguous windows of the same hidden mother sequence.
    - If the overlaps force a unique color for every unknown position, recolor each grey ring with that forced color and leave everything else unchanged.

    Key insight:
    - A single target is ambiguous, but the shared sliding-window overlaps between multiple targets pin each missing layer to the color already witnessed by another aligned target.
    """
    g = np.asarray(grid, dtype=int)
    targets = _extract_targets(g)
    sequences = [target["sequence"] for target in targets]
    inferred = _infer_missing_colors(sequences)
    assert inferred is not None

    out = g.copy()
    for target, color in zip(targets, inferred):
        r0, c0, r1, c1 = target["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        _paint_ring(patch, target["missing_idx"], int(color))
    return out


def _intervals_connected(intervals: list[tuple[int, int]]) -> bool:
    seen = {0}
    stack = [0]
    while stack:
        idx = stack.pop()
        s0, l0 = intervals[idx]
        e0 = s0 + l0
        for j, (s1, l1) in enumerate(intervals):
            if j in seen:
                continue
            e1 = s1 + l1
            if max(s0, s1) < min(e0, e1):
                seen.add(j)
                stack.append(j)
    return len(seen) == len(intervals)


def _sample_sequences() -> tuple[list[tuple[int | None, ...]], list[tuple[int, ...]]]:
    for _ in range(GEN_ATTEMPTS):
        mother_len = random.randint(5, 8)
        mother = tuple(random.sample(COLOR_POOL, mother_len))
        target_count = random.randint(3, min(5, mother_len))

        intervals: list[tuple[int, int]] = []
        for _idx in range(target_count):
            length = random.randint(3, min(5, mother_len))
            start = random.randint(0, mother_len - length)
            intervals.append((start, length))

        if len(set(intervals)) < 2 or not _intervals_connected(intervals):
            continue

        hidden_positions = [random.randint(0, length - 1) for _start, length in intervals]
        full_sequences = []
        masked_sequences = []
        for (start, length), hidden in zip(intervals, hidden_positions):
            seq = mother[start : start + length]
            full_sequences.append(tuple(seq))
            masked = list(seq)
            masked[hidden] = None
            masked_sequences.append(tuple(masked))

        inferred = _infer_missing_colors(masked_sequences)
        expected = tuple(seq[hidden] for seq, hidden in zip(full_sequences, hidden_positions))
        if inferred != expected:
            continue
        return masked_sequences, full_sequences
    raise RuntimeError("failed to sample a uniquely solvable spectrum family")


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_patches(patches: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    max_side = max(int(p.shape[0]) for p in patches)
    for _ in range(GEN_ATTEMPTS):
        h_min = max(max_side + 2, int(np.ceil(np.sqrt(total_area))) + 6)
        w_min = max_side + 2
        h = random.randint(h_min, 30)
        w = random.randint(max(w_min, h_min - 3), 30)
        canvas = np.full((h, w), BG, dtype=int)

        positions: list[tuple[int, int] | None] = [None] * len(patches)
        placed_boxes: list[tuple[int, int, int, int]] = []
        order = sorted(range(len(patches)), key=lambda idx: int(patches[idx].shape[0]), reverse=True)
        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            placed = False
            for _trial in range(300):
                r0 = random.randint(1, h - ph - 1)
                c0 = random.randint(1, w - pw - 1)
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if all(_separated(box, other, gap=1) for other in placed_boxes):
                    canvas[r0 : r0 + ph, c0 : c0 + pw] = patch
                    positions[idx] = (r0, c0)
                    placed_boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return canvas, [pos for pos in positions if pos is not None]
    raise RuntimeError("failed to place targets")


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        masked_sequences, full_sequences = _sample_sequences()
        missing_indices = [seq.index(None) for seq in masked_sequences]

        masked_patches = []
        solved_patches = []
        for masked, full, miss in zip(masked_sequences, full_sequences, missing_indices):
            solved = tuple(int(color) for color in full)
            masked_patches.append(_make_target(solved, miss))
            solved_patches.append(_make_target(solved, None))

        grid, positions = _place_patches(masked_patches)
        expected = grid.copy()
        for patch, (r0, c0) in zip(solved_patches, positions):
            h, w = patch.shape
            expected[r0 : r0 + h, c0 : c0 + w] = patch

        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
