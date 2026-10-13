from __future__ import annotations

from collections import Counter
import numpy as np


BG = 7  # orange
COLOR_POOL = [1, 2, 3, 4, 5, 6, 8, 9]
EDGE_ORDER = ("top", "right", "bottom", "left")
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_segments(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    segments: list[dict[str, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = len(coords)

            if r0 == r1 and r0 in (0, h - 1):
                edge = "top" if r0 == 0 else "bottom"
                assert area == c1 - c0 + 1
                assert all(int(g[r0, cc]) == color for cc in range(c0, c1 + 1))
                pos = int(c0 if edge == "top" else (w - 1 - c1))
            elif c0 == c1 and c0 in (0, w - 1):
                edge = "left" if c0 == 0 else "right"
                assert area == r1 - r0 + 1
                assert all(int(g[rr, c0]) == color for rr in range(r0, r1 + 1))
                pos = int(r0 if edge == "right" else (h - 1 - r1))
            elif area == 1 and (r0, c0) in ((0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)):
                edge = {
                    (0, 0): "top",
                    (0, w - 1): "right",
                    (h - 1, w - 1): "bottom",
                    (h - 1, 0): "left",
                }[(r0, c0)]
                pos = 0
            else:
                raise AssertionError("foreground components must be straight edge segments")

            segments.append(
                {
                    "color": color,
                    "length": int(area),
                    "edge_rank": EDGE_ORDER.index(edge),
                    "pos": int(pos),
                }
            )

    assert segments, "expected at least one edge segment"
    return segments


def _paint_output(shape: tuple[int, int], ordered: list[dict[str, int]], bg: int) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    edge_buckets = {edge: [] for edge in EDGE_ORDER}
    for idx, seg in enumerate(ordered):
        edge_buckets[EDGE_ORDER[idx % 4]].append(seg)

    nonempty = {edge: bool(edge_buckets[edge]) for edge in EDGE_ORDER}
    capacities = {
        "top": w - int(nonempty["right"]),
        "right": h - int(nonempty["bottom"]),
        "bottom": w - int(nonempty["left"]),
        "left": h - int(nonempty["top"]),
    }

    for edge in EDGE_ORDER:
        need = sum(int(seg["length"]) for seg in edge_buckets[edge])
        assert need <= capacities[edge], f"segments do not fit on {edge} edge"

    cursors = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    for edge in EDGE_ORDER:
        for seg in edge_buckets[edge]:
            color = int(seg["color"])
            length = int(seg["length"])
            start = int(cursors[edge])
            if edge == "top":
                out[0, start : start + length] = color
            elif edge == "right":
                out[start : start + length, w - 1] = color
            elif edge == "bottom":
                c1 = w - 1 - start
                out[h - 1, c1 - length + 1 : c1 + 1] = color
            else:
                r1 = h - 1 - start
                out[r1 - length + 1 : r1 + 1, 0] = color
            cursors[edge] += length

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort edge segments by length and reassign them to clockwise edge slots."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    segments = _find_segments(g, bg)
    segments.sort(key=lambda seg: (int(seg["length"]), int(seg["edge_rank"]), int(seg["pos"])))
    return _paint_output(g.shape, segments, bg)


def _sample_sorted_lengths(rng: np.random.Generator, h: int, w: int, n: int) -> list[int] | None:
    max_len = int(max(h - 2, w - 2))
    if max_len < n:
        return None

    counts = [sum(1 for i in range(n) if i % 4 == edge_idx) for edge_idx in range(4)]
    nonempty = [count > 0 for count in counts]
    capacities = [
        w - int(nonempty[1]),
        h - int(nonempty[2]),
        w - int(nonempty[3]),
        h - int(nonempty[0]),
    ]

    for _ in range(300):
        lengths = sorted(int(x) for x in rng.choice(np.arange(1, max_len + 1), size=n, replace=False))
        if any(sum(lengths[i] for i in range(edge_idx, n, 4)) > capacities[edge_idx] for edge_idx in range(4)):
            continue
        return lengths
    return None


def _sample_input_edges(
    rng: np.random.Generator, lengths: list[int], h: int, w: int
) -> list[list[int]] | None:
    capacities = [w - 2, h - 2, w - 2, h - 2]
    for _ in range(300):
        buckets = [[] for _ in range(4)]
        for idx in rng.permutation(len(lengths)):
            edge = int(rng.integers(0, 4))
            buckets[edge].append(int(lengths[idx]))
        if any(sum(bucket) + max(0, len(bucket) - 1) > capacities[edge] for edge, bucket in enumerate(buckets)):
            continue
        return buckets
    return None


def _place_edge_segments(
    grid: np.ndarray,
    edge_idx: int,
    segments: list[tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    h, w = grid.shape
    capacity = (w - 2) if edge_idx in (0, 2) else (h - 2)
    total = sum(length for _, length in segments)
    gaps = len(segments) - 1
    slack = capacity - total - gaps
    extras = rng.multinomial(slack, np.ones(len(segments) + 1) / (len(segments) + 1)) if segments else []

    cursor = 1 + (int(extras[0]) if segments else 0)
    for i, (color, length) in enumerate(segments):
        if edge_idx == 0:
            grid[0, cursor : cursor + length] = color
        elif edge_idx == 1:
            grid[cursor : cursor + length, w - 1] = color
        elif edge_idx == 2:
            grid[h - 1, cursor : cursor + length] = color
        else:
            grid[cursor : cursor + length, 0] = color
        cursor += length
        if i < len(segments) - 1:
            cursor += 1 + int(extras[i + 1])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 19))
    w = int(rng.integers(10, 19))
    max_segments = min(8, len(COLOR_POOL), max(h - 2, w - 2))
    if max_segments < 2:
        return None

    n = int(rng.integers(2, max_segments + 1))
    lengths = _sample_sorted_lengths(rng, h, w, n)
    if lengths is None:
        return None

    colors = [int(c) for c in rng.choice(np.array(COLOR_POOL), size=n, replace=False)]
    color_for_length = {int(length): int(color) for length, color in zip(lengths, colors)}

    input_edges = _sample_input_edges(rng, lengths, h, w)
    if input_edges is None:
        return None

    buckets: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    for edge_idx, bucket_lengths in enumerate(input_edges):
        for length in bucket_lengths:
            buckets[edge_idx].append((int(color_for_length[int(length)]), int(length)))
        rng.shuffle(buckets[edge_idx])

    grid = np.full((h, w), BG, dtype=int)
    for edge_idx in range(4):
        _place_edge_segments(grid, edge_idx, buckets[edge_idx], rng)

    # Ensure the instance matches the intended rule and is not degenerate.
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if len({length for bucket in buckets for _, length in bucket}) != n:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
