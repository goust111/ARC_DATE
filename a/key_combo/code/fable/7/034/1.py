"""ARC task: propagate a cyclically recolored flow through aligned frame gaps."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BG, FRAME = 0, 1
RED, TEAL, GREY = 2, 8, 5
PALETTE = (RED, TEAL, GREY)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components using the common ARC (row, column) convention."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        result.append(comp)
    return result


def _border(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    ]


def _outward(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return -1, 0
    if r == r1 and c0 < c < c1:
        return 1, 0
    if c == c0 and r0 < r < r1:
        return 0, -1
    if c == c1 and r0 < r < r1:
        return 0, 1
    raise AssertionError("every gap must be a non-corner border cell")


def _read_frames(g: np.ndarray) -> list[dict]:
    # Two gaps split a rectangular cycle into two connected components, so frame
    # identity must be recovered in bbox-border coordinates, not CC coordinates.
    blue = {tuple(map(int, p)) for p in np.argwhere(g == FRAME)}
    candidates: list[dict] = []
    points = sorted(blue)
    for r0, c0 in points:
        for r1, c1 in points:
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            if (r0, c1) not in blue or (r1, c0) not in blue:
                continue
            bbox = (r0, c0, r1, c1)
            border = _border(bbox)
            border_blue = {p for p in border if p in blue}
            gaps = [p for p in border if p not in blue]
            if len(gaps) not in (1, 2) or len(border_blue) != len(border) - len(gaps):
                continue
            if any(int(g[p]) != BG for p in gaps):
                continue
            if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == FRAME):
                continue
            try:
                for gap in gaps:
                    _outward(gap, bbox)
            except AssertionError:
                continue
            candidates.append({"bbox": bbox, "gaps": gaps, "blue": border_blue})

    by_pixel: dict[tuple[int, int], list[int]] = {p: [] for p in blue}
    for i, candidate in enumerate(candidates):
        for p in candidate["blue"]:
            by_pixel[p].append(i)
    solutions: list[list[int]] = []

    def cover(remaining: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return
        pivot = min(remaining, key=lambda p: len([i for i in by_pixel[p] if candidates[i]["blue"] <= remaining]))
        for i in by_pixel[pivot]:
            cells = candidates[i]["blue"]
            if cells <= remaining:
                cover(remaining - cells, chosen + [i])

    cover(set(blue), [])
    assert len(solutions) == 1, "blue pixels must determine a unique set of rectangular frames"
    return [{"bbox": candidates[i]["bbox"], "gaps": candidates[i]["gaps"]} for i in solutions[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """Send the red seed flow through a chain of aligned blue-frame gaps.

    Input structure:
    - A black background contains separated one-cell-thick blue rectangular frames.
    - Every frame has one or two non-corner border gaps, and exactly one source frame
      contains a single red seed pixel in its otherwise empty interior.
    - Some non-source frames may be unaligned spectators.

    Rule:
    - Starting red, send a straight outward ray from the source frame's sole gap.
    - When the ray enters another frame through an exactly aligned gap, advance through
      the red, teal, grey cycle, fill that frame's interior, and leave through its other
      gap if present. A wall, terminal frame, or grid edge ends propagation.
    - Preserve every frame, gap, and background cell not visited by this flow.

    Key insight:
    - Gaps are directed doors: the hidden visit chain, rather than a frame property,
      determines both which interiors change and their cyclic visit-order colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BG, FRAME, RED}, "input uses black, blue, and one red seed"
    frames = _read_frames(g)

    sources = []
    for i, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        reds = np.argwhere(g[r0 + 1 : r1, c0 + 1 : c1] == RED)
        if len(reds):
            sources.append(i)
    assert len(sources) == 1 and int(np.count_nonzero(g == RED)) == 1
    source = sources[0]
    assert len(frames[source]["gaps"]) == 1

    gap_owner: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for i, frame in enumerate(frames):
        for gap in frame["gaps"]:
            assert gap not in gap_owner, "separated frames cannot share a gap"
            gap_owner[gap] = (i, _outward(gap, frame["bbox"]))

    out = g.copy()
    current_frame = source
    color_index = 0
    exit_gap = frames[source]["gaps"][0]
    direction = _outward(exit_gap, frames[source]["bbox"])
    visited = {source}
    r, c = exit_gap
    h, w = g.shape

    while 0 <= r < h and 0 <= c < w:
        owner = gap_owner.get((r, c))
        if owner is not None and owner[0] != current_frame and direction == (-owner[1][0], -owner[1][1]):
            next_frame = owner[0]
            assert next_frame not in visited, "the gap chain must not loop"
            out[r, c] = PALETTE[color_index]
            color_index = (color_index + 1) % len(PALETTE)
            color = PALETTE[color_index]
            r0, c0, r1, c1 = frames[next_frame]["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = color
            visited.add(next_frame)
            other = [p for p in frames[next_frame]["gaps"] if p != (r, c)]
            if not other:
                break
            assert len(other) == 1
            current_frame = next_frame
            r, c = other[0]
            direction = _outward((r, c), frames[next_frame]["bbox"])
            continue

        if int(g[r, c]) != BG:
            break
        out[r, c] = PALETTE[color_index]
        r += direction[0]
        c += direction[1]

    return out


def _side_gap(b: tuple[int, int, int, int], side: int, offset: int | None = None) -> tuple[int, int]:
    r0, c0, r1, c1 = b
    if side < 2:
        c = c0 + (offset if offset is not None else random.randint(1, c1 - c0 - 1))
        return (r0 if side == 0 else r1, c)
    r = r0 + (offset if offset is not None else random.randint(1, r1 - r0 - 1))
    return (r, c0 if side == 2 else c1)


def _separated(b: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], pad: int = 1) -> bool:
    r0, c0, r1, c1 = b
    for a0, b0, a1, b1 in boxes:
        if not (r1 + pad < a0 or a1 + pad < r0 or c1 + pad < b0 or b1 + pad < c0):
            return False
    return True


def _place_next(
    shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], out_gap: tuple[int, int], side: int
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    """Place a frame ahead of out_gap and return its bbox and incoming gap."""
    h_grid, w_grid = shape
    gr, gc = out_gap
    for _ in range(100):
        hh, ww = random.randint(4, 7), random.randint(4, 8)
        distance = random.randint(2, 5)
        incoming_side = side ^ 1
        if side == 0:
            r1 = gr - distance
            r0 = r1 - hh + 1
            off = random.randint(1, ww - 2)
            c0, c1 = gc - off, gc - off + ww - 1
        elif side == 1:
            r0 = gr + distance
            r1 = r0 + hh - 1
            off = random.randint(1, ww - 2)
            c0, c1 = gc - off, gc - off + ww - 1
        elif side == 2:
            c1 = gc - distance
            c0 = c1 - ww + 1
            off = random.randint(1, hh - 2)
            r0, r1 = gr - off, gr - off + hh - 1
        else:
            c0 = gc + distance
            c1 = c0 + ww - 1
            off = random.randint(1, hh - 2)
            r0, r1 = gr - off, gr - off + hh - 1
        b = (r0, c0, r1, c1)
        if r0 >= 1 and c0 >= 1 and r1 < h_grid - 1 and c1 < w_grid - 1 and _separated(b, boxes):
            return b, _side_gap(b, incoming_side, off)
    return None


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], gaps: list[tuple[int, int]]) -> None:
    for p in _border(bbox):
        grid[p] = BG if p in gaps else FRAME


def _oracle(
    inp: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    gaps: list[list[tuple[int, int]]],
    chain_length: int,
) -> np.ndarray:
    """Independent metadata-based renderer used to validate generation semantics."""
    out = inp.copy()
    color_index = 0
    for i in range(chain_length):
        if i == 0:
            start = gaps[0][0]
        else:
            color_index = (color_index + 1) % 3
            r0, c0, r1, c1 = boxes[i]
            out[r0 + 1 : r1, c0 + 1 : c1] = PALETTE[color_index]
            start = gaps[i][1] if len(gaps[i]) == 2 else None
        if start is None:
            break
        dr, dc = _outward(start, boxes[i])
        r, c = start
        stop = gaps[i + 1][0] if i + 1 < chain_length else None
        while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            if stop is not None and (r, c) == stop:
                out[r, c] = PALETTE[color_index]
                break
            if int(inp[r, c]) != BG:
                break
            out[r, c] = PALETTE[color_index]
            r, c = r + dr, c + dc
    return out


def _attempt() -> np.ndarray | None:
    shape = (random.randint(24, 30), random.randint(24, 30))
    chain_length = random.choice((3, 4, 4))
    hh, ww = random.randint(4, 6), random.randint(5, 8)
    r0 = random.randint(7, shape[0] - hh - 7)
    c0 = random.randint(7, shape[1] - ww - 7)
    boxes = [(r0, c0, r0 + hh - 1, c0 + ww - 1)]
    first_side = random.randrange(4)
    gaps: list[list[tuple[int, int]]] = [[_side_gap(boxes[0], first_side)]]

    for i in range(1, chain_length):
        prev_gap = gaps[i - 1][-1]
        prev_side = DIRS.index(_outward(prev_gap, boxes[i - 1]))
        placed = _place_next(shape, boxes, prev_gap, prev_side)
        if placed is None:
            return None
        b, incoming = placed
        boxes.append(b)
        if i == chain_length - 1:
            gaps.append([incoming])
        else:
            choices = [s for s in range(4) if s != (prev_side ^ 1)]
            random.shuffle(choices)
            outgoing = _side_gap(b, choices[0])
            gaps.append([incoming, outgoing])

    # Add one or two spectators. Their deliberately offset gaps are near-miss evidence.
    spectator_count = random.choice((1, 2))
    for _ in range(spectator_count):
        found = False
        for _ in range(200):
            hh, ww = random.randint(4, 7), random.randint(4, 8)
            b = (
                random.randint(1, shape[0] - hh - 1),
                random.randint(1, shape[1] - ww - 1),
                0,
                0,
            )
            b = (b[0], b[1], b[0] + hh - 1, b[1] + ww - 1)
            if not _separated(b, boxes, pad=2):
                continue
            n_gaps = random.choice((1, 2))
            sides = random.sample(range(4), n_gaps)
            sg = [_side_gap(b, s) for s in sides]
            boxes.append(b)
            gaps.append(sg)
            found = True
            break
        if not found:
            return None

    inp = np.zeros(shape, dtype=int)
    for b, gs in zip(boxes, gaps):
        _paint_frame(inp, b, gs)
    sr0, sc0, sr1, sc1 = boxes[0]
    inp[random.randint(sr0 + 1, sr1 - 1), random.randint(sc0 + 1, sc1 - 1)] = RED

    expected = _oracle(inp, boxes, gaps, chain_length)
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None

    # Semantic/non-degeneracy checks: every chain link matters; spectators do not.
    for i in range(1, chain_length):
        r0, c0, r1, c1 = boxes[i]
        if not np.all(actual[r0 + 1 : r1, c0 + 1 : c1] == PALETTE[i % 3]):
            return None
    for i in range(chain_length, len(boxes)):
        r0, c0, r1, c1 = boxes[i]
        if np.any(actual[r0:r1 + 1, c0:c1 + 1] != inp[r0:r1 + 1, c0:c1 + 1]):
            return None
    if np.count_nonzero(actual != inp) < 8:
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate varied, unambiguous chains with terminal and spectator counterexamples."""
    for _ in range(800):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-chain-flow input")
