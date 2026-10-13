from __future__ import annotations

from collections import Counter, deque
import hashlib
import random

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
PALETTE = tuple(range(1, 10))


def _background(g: np.ndarray) -> int:
    """Return the uniquely most frequent color (the ARC background convention)."""
    counts = Counter(map(int, g.reshape(-1)))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    result: list[np.ndarray] = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    q = (r + dr, c + dc)
                    if q in unseen:
                        unseen.remove(q)
                        stack.append(q)
                        component.append(q)
        result.append(np.asarray(component, dtype=int))
    return result


def _mouth_region(
    frame: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> set[tuple[int, int]]:
    """Flood the open side of a frame, staying in its inclusive bounding box."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in frame]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in frame]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in frame]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in frame]

    reached = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS.values():
            q = (r + dr, c + dc)
            if (
                r0 <= q[0] <= r1
                and c0 <= q[1] <= c1
                and q not in frame
                and q not in reached
            ):
                reached.add(q)
                stack.append(q)
    return reached


def _find_cups(g: np.ndarray, bg: int) -> list[dict]:
    """Recognize frames by a unique non-corner singleton edge spike."""
    cups: list[dict] = []
    for frame_color in sorted(set(map(int, g.reshape(-1))) - {bg}):
        for coords in _components8(g == frame_color):
            if len(coords) < 5:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            bbox = (int(r0), int(c0), int(r1), int(c1))
            frame = {tuple(map(int, rc)) for rc in coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            edge_cells = {
                "U": [p for p in frame if p[0] == r0],
                "D": [p for p in frame if p[0] == r1],
                "L": [p for p in frame if p[1] == c0],
                "R": [p for p in frame if p[1] == c1],
            }
            spike_candidates = [
                (direction, cells[0])
                for direction, cells in edge_cells.items()
                if len(cells) == 1 and cells[0] not in corners
            ]

            valid = []
            for direction, spike in spike_candidates:
                mouth = _mouth_region(frame, bbox, OPPOSITE[direction])
                payload = Counter(
                    int(g[r, c])
                    for r, c in mouth
                    if int(g[r, c]) not in (bg, frame_color)
                )
                if len(payload) != 1:
                    continue
                content_color, count = next(iter(payload.items()))
                if count <= 0:
                    continue
                cells = {p for p in mouth if int(g[p]) == content_color}
                valid.append((direction, spike, mouth, int(content_color), cells))

            if len(valid) == 1:
                direction, spike, mouth, content_color, content_cells = valid[0]
                cups.append(
                    {
                        "frame_color": int(frame_color),
                        "frame": frame,
                        "bbox": bbox,
                        "direction": direction,
                        "spike": tuple(map(int, spike)),
                        "mouth": mouth,
                        "content_color": content_color,
                        "content_cells": content_cells,
                        "content_count": len(content_cells),
                    }
                )
    return cups


def _successors(cups: list[dict], shape: tuple[int, int]) -> list[int | None]:
    """Find the first cup mouth met by each spike ray in the grid reference frame."""
    h, w = shape
    successors: list[int | None] = []
    for i, cup in enumerate(cups):
        dr, dc = DIRS[cup["direction"]]
        r, c = cup["spike"]
        found = None
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            hits = [j for j, other in enumerate(cups) if j != i and (r, c) in other["mouth"]]
            assert len(hits) <= 1, "an ejection ray cannot enter two mouths at once"
            if hits:
                found = hits[0]
                break
        successors.append(found)
    return successors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade cup contents through downstream mouths and eject each chain from its final spike.

    Input structure:
    - A dominant background contains several hollow, differently colored cup frames.
    - Each frame has one convergent tail spike, an opposite open mouth, and one uniquely
      colored group of content pixels in that mouth; different cups use different content colors.
    - A spike ray either leaves the grid or enters exactly one other cup mouth; these links
      form acyclic chains and every cup receives at most one upstream ray.

    Rule:
    - Erase every cup's original content.  Follow each spike ray in grid row-column
      coordinates.  A ray entering another mouth transfers its entire ordered payload and
      paints nothing there.  When a cup later ejects, its own content is first, followed by
      the upstream payload.  Only a chain-ending ray is painted from the cell beyond its
      spike, clipped by the grid boundary, with every source segment retaining its color.

    Key insight:
    - Cups are nodes of a directed transfer chain, not independent launchers; the final
      ray's near-to-far color segments encode the chain in reverse traversal order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background(g)
    cups = _find_cups(g, bg)
    assert cups, "expected at least one unambiguous cup"

    successors = _successors(cups, g.shape)
    indegree = [0] * len(cups)
    for target in successors:
        if target is not None:
            indegree[target] += 1
            assert indegree[target] <= 1, "a cup may receive at most one stream"

    out = g.copy()
    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    incoming: list[list[tuple[int, int]]] = [[] for _ in cups]
    queue = deque(i for i, degree in enumerate(indegree) if degree == 0)
    processed = 0
    h, w = g.shape
    while queue:
        i = queue.popleft()
        processed += 1
        cup = cups[i]
        packet = [(cup["content_color"], cup["content_count"]), *incoming[i]]
        target = successors[i]
        if target is not None:
            incoming[target] = packet
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
            continue

        dr, dc = DIRS[cup["direction"]]
        r, c = cup["spike"]
        for color, count in packet:
            for _ in range(count):
                r += dr
                c += dc
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = color

    assert processed == len(cups), "cup transfer graph must be acyclic"
    return out


# ------------------------------ generator ------------------------------

_SEEN_OUTPUTS: set[bytes] = set()


def _canonical_frame(color: int) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Five-by-four cup with tail right and mouth left."""
    patch = np.zeros((5, 4), dtype=int)
    for r, c in ((0, 0), (1, 1), (2, 2), (2, 3), (3, 1), (4, 0)):
        patch[r, c] = color
    frame = {tuple(map(int, rc)) for rc in np.argwhere(patch == color)}
    mouth = sorted(_mouth_region(frame, (0, 0, 4, 3), "L"))
    return patch, (2, 3), mouth


def _rotate_grid(g: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(g, k=turns).copy()


def _oracle_horizontal(
    inp: np.ndarray, chains: list[list[dict]], bg: int = 0
) -> np.ndarray:
    """Render expected output directly from generator metadata, independently of main."""
    out = inp.copy()
    for chain in chains:
        for cup in chain:
            for r, c in cup["content_cells"]:
                out[r, c] = bg
        sink = chain[-1]
        r, c = sink["spike"]
        for cup in reversed(chain):
            for _ in range(cup["count"]):
                c += 1
                if c < out.shape[1]:
                    out[r, c] = cup["content_color"]
    return out


def _independent_horizontal(inp: np.ndarray, chains: list[list[dict]]) -> np.ndarray:
    """Deliberately wrong shortcut: erase, then launch every cup independently."""
    out = inp.copy()
    for chain in chains:
        for cup in chain:
            for r, c in cup["content_cells"]:
                out[r, c] = 0
            r, c = cup["spike"]
            for _ in range(cup["count"]):
                c += 1
                if c < out.shape[1]:
                    out[r, c] = cup["content_color"]
    return out


def _partition_chains(n: int) -> list[int]:
    if n == 4 and random.random() < 0.40:
        return [2, 2]
    if n == 3 and random.random() < 0.25:
        return [2, 1]
    return [n]


def _generate_attempt() -> np.ndarray | None:
    n_cups = random.choice((2, 3, 4))
    lengths = _partition_chains(n_cups)
    height = random.randint(7 if len(lengths) == 1 else 13, 18)
    width = random.randint(22, 30)
    inp = np.zeros((height, width), dtype=int)
    colors = random.sample(PALETTE, 2 * n_cups)
    frame_colors = colors[:n_cups]
    content_colors = colors[n_cups:]
    color_index = 0
    chains: list[list[dict]] = []

    lane_tops = random.sample(range(1, height - 5), len(lengths))
    lane_tops.sort()
    if any(b - a < 6 for a, b in zip(lane_tops, lane_tops[1:])):
        return None

    for lane, chain_len in enumerate(lengths):
        gaps = [random.randint(1, 3) for _ in range(chain_len - 1)]
        span = 4 * chain_len + sum(gaps)
        counts = [random.randint(1, 3) for _ in range(chain_len)]
        total = sum(counts)
        # Usually preserve the whole packet; sometimes clip only its far end.
        desired_room = total if random.random() < 0.72 else max(2, total - random.randint(1, 2))
        max_start = width - span - desired_room
        if max_start < 1:
            return None
        x = random.randint(1, max_start)
        y = lane_tops[lane]
        chain: list[dict] = []
        for j in range(chain_len):
            frame_color = frame_colors[color_index]
            content_color = content_colors[color_index]
            patch, spike_local, mouth_local = _canonical_frame(frame_color)
            mask = patch != 0
            view = inp[y : y + 5, x : x + 4]
            if np.any(view[mask] != 0):
                return None
            view[mask] = patch[mask]

            # Scatter content across the true mouth, always including the axis cell.
            available = [(r, c) for r, c in mouth_local if (r, c) != (2, 0)]
            chosen = [(2, 0)]
            if counts[j] > 1:
                chosen += random.sample(available, counts[j] - 1)
            content_cells = [(y + r, x + c) for r, c in chosen]
            for r, c in content_cells:
                assert inp[r, c] == 0
                inp[r, c] = content_color

            chain.append(
                {
                    "frame_color": frame_color,
                    "content_color": content_color,
                    "content_cells": content_cells,
                    "count": counts[j],
                    "spike": (y + spike_local[0], x + spike_local[1]),
                }
            )
            color_index += 1
            if j < chain_len - 1:
                x += 4 + gaps[j]
        chains.append(chain)

    expected = _oracle_horizontal(inp, chains)
    wrong_independent = _independent_horizontal(inp, chains)
    erased_only = inp.copy()
    for chain in chains:
        for cup in chain:
            for r, c in cup["content_cells"]:
                erased_only[r, c] = 0
    if np.array_equal(expected, wrong_independent) or np.array_equal(expected, erased_only):
        return None

    turns = random.randrange(4)
    rotated_input = _rotate_grid(inp, turns)
    rotated_expected = _rotate_grid(expected, turns)
    if max(rotated_input.shape) > 30:
        return None

    # Semantic checks: intended graph, source order, uniqueness, and independent oracle.
    try:
        cups = _find_cups(rotated_input, _background(rotated_input))
        succ = _successors(cups, rotated_input.shape)
        if sum(x is not None for x in succ) != sum(length - 1 for length in lengths):
            return None
        actual = main(rotated_input)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, rotated_expected):
        return None
    if np.array_equal(actual, rotated_input):
        return None

    signature = hashlib.sha256(actual.tobytes() + repr(actual.shape).encode()).digest()
    if signature in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(signature)
    return rotated_input


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate chains in all four orientations."""
    for _ in range(500):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid cascade-ejection input")
