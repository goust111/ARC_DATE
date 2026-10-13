from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

# Aligned hole cycle. The final blank template panel is the missing fourth slot.
HOLE_CYCLE = (RED, BG, BLUE, BG)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_key(comp: list[tuple[int, int]]) -> tuple[int, int, int, int, int, tuple[tuple[int, int], ...]]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    pts = tuple(sorted(comp))
    return (min(rows), min(cols), max(rows), max(cols), len(comp), pts)


def _hole_components(panel: np.ndarray) -> list[list[tuple[int, int]]]:
    holes = []
    for comp in _components(panel == BG):
        if any(r == 0 or c == 0 or r == panel.shape[0] - 1 or c == panel.shape[1] - 1 for r, c in comp):
            continue
        holes.append(comp)
    holes.sort(key=_component_key)
    return holes


def _fill_state(panel: np.ndarray, hole: list[tuple[int, int]]) -> int:
    vals = {int(panel[r, c]) for r, c in hole}
    vals.discard(BG)
    vals.discard(GREEN)
    if not vals:
        return BG
    assert len(vals) == 1
    color = vals.pop()
    assert color in (BLUE, RED)
    return color


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    if w == 4 * h + 3:
        size = h
        cuts = [size, 2 * size + 1, 3 * size + 2]
        assert all(np.all(grid[:, c] == BG) for c in cuts)
        return [grid[:, i * (size + 1): i * (size + 1) + size].copy() for i in range(4)]
    if h == 4 * w + 3:
        size = w
        cuts = [size, 2 * size + 1, 3 * size + 2]
        assert all(np.all(grid[r, :] == BG) for r in cuts)
        return [grid[i * (size + 1): i * (size + 1) + size, :].copy() for i in range(4)]
    raise AssertionError("grid is not a 4-panel square strip")


def _infer_rotation_step(panels: list[np.ndarray]) -> int:
    masks = [p == GREEN for p in panels]
    candidates = []
    for k in (1, 3):
        if all(np.array_equal(np.rot90(masks[i], k), masks[i + 1]) for i in range(3)):
            candidates.append(k)
    assert len(candidates) == 1
    return int(candidates[0])


def _align_to_template(panels: list[np.ndarray], rot_step: int) -> list[np.ndarray]:
    last = len(panels) - 1
    return [np.rot90(panel, ((last - i) * rot_step) % 4).copy() for i, panel in enumerate(panels)]


def _predict_next_state(seq: tuple[int, int, int]) -> int:
    matches = []
    for phase in range(4):
        cycle = tuple(HOLE_CYCLE[(phase + t) % 4] for t in range(4))
        if cycle[:3] == seq:
            matches.append(cycle[3])
    assert len(matches) == 1
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the missing colored-hole panel on the final rotated template.

    Input structure:
    - The grid is a 1x4 or 4x1 strip of equal square panels separated by 1-pixel black lines.
    - Every panel contains the same green outline shape in a different 90-degree orientation.
    - The last panel is an empty template of that outline; earlier panels may color enclosed holes in red or blue.

    Rule:
    - Identify the rotation step between consecutive outline panels.
    - Rotate the earlier panels into the last panel's orientation so the same hole locations line up.
    - For each enclosed hole, read its first three aligned fill states and match them to the repeating cycle red -> empty -> blue -> empty.
    - Paint the uniquely implied fourth state onto the blank final template and return that single panel.

    Key insight:
    - Hole identities move with the rotating outline, so the hole schedule is only readable after undoing the inter-panel rotations.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    rot_step = _infer_rotation_step(panels)
    aligned = _align_to_template(panels, rot_step)
    template = aligned[-1].copy()
    holes = _hole_components(template)
    assert 2 <= len(holes) <= 3

    for hole in holes:
        seq = tuple(_fill_state(aligned[t], hole) for t in range(3))
        color = _predict_next_state(seq)
        if color != BG:
            for r, c in hole:
                template[r, c] = color
    return template


SHAPE_VARIANTS = [
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
]


def _render_empty_template(size: int) -> np.ndarray:
    local = random.choice(SHAPE_VARIANTS)
    panel = np.zeros((size, size), dtype=int)
    top = 0 if size == 5 else random.randint(0, 1)
    left = 0 if size == 5 else random.randint(0, 1)
    panel[top: top + 5, left: left + 5] = np.where(local == 1, GREEN, BG)
    panel = np.rot90(panel, random.randrange(4)).copy()
    mask = panel == GREEN
    assert all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))
    holes = _hole_components(panel)
    assert 2 <= len(holes) <= 3
    return panel


def _hole_sequences(n_holes: int) -> list[tuple[int, int, int, int]]:
    while True:
        phases = [random.randrange(4) for _ in range(n_holes)]
        seqs = [tuple(HOLE_CYCLE[(phase + t) % 4] for t in range(4)) for phase in phases]
        if len(set(phases)) < 2:
            continue
        predicted = tuple(seq[3] for seq in seqs)
        observed = [tuple(seq[t] for seq in seqs) for t in range(3)]
        if predicted in observed:
            continue
        if all(v == BG for v in predicted):
            continue
        return seqs


def _paint_states(template: np.ndarray, holes: list[list[tuple[int, int]]], states: list[int]) -> np.ndarray:
    out = template.copy()
    for hole, color in zip(holes, states):
        if color == BG:
            continue
        for r, c in hole:
            out[r, c] = int(color)
    return out


def _assemble_strip(panels: list[np.ndarray], horizontal: bool) -> np.ndarray:
    size = panels[0].shape[0]
    if horizontal:
        out = np.zeros((size, 4 * size + 3), dtype=int)
        for i, panel in enumerate(panels):
            c0 = i * (size + 1)
            out[:, c0:c0 + size] = panel
        return out
    out = np.zeros((4 * size + 3, size), dtype=int)
    for i, panel in enumerate(panels):
        r0 = i * (size + 1)
        out[r0:r0 + size, :] = panel
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    size = random.choice((5, 6))
    template = _render_empty_template(size)
    holes = _hole_components(template)
    seqs = _hole_sequences(len(holes))

    aligned_frames = []
    for t in range(4):
        states = [seq[t] for seq in seqs]
        aligned_frames.append(_paint_states(template, holes, states))

    panel1 = np.rot90(aligned_frames[0], 3).copy()
    panel2 = np.rot90(aligned_frames[1], 2).copy()
    panel3 = np.rot90(aligned_frames[2], 1).copy()
    panel4 = template.copy()
    output = aligned_frames[3].copy()

    panels = [panel1, panel2, panel3, panel4]
    horizontal = random.choice((False, True))
    inp = _assemble_strip(panels, horizontal)
    return inp, output


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    panels = _split_panels(inp)
    rot_step = _infer_rotation_step(panels)
    if rot_step != 3:
        return False
    aligned = _align_to_template(panels, rot_step)
    holes = _hole_components(aligned[-1])
    if not (2 <= len(holes) <= 3):
        return False

    vectors = []
    for t in range(3):
        vectors.append(tuple(_fill_state(aligned[t], hole) for hole in holes))
    predicted = tuple(_fill_state(expected, hole) for hole in holes)
    if predicted in vectors:
        return False
    if len(set(vectors)) != 3:
        return False

    for hole in holes:
        seq = tuple(_fill_state(aligned[t], hole) for t in range(3))
        if _predict_next_state(seq) != _fill_state(expected, hole):
            return False

    green = aligned[-1] == GREEN
    if any(np.array_equal(green, np.rot90(green, k)) for k in (1, 2, 3)):
        return False
    if np.array_equal(expected, aligned[-1]):
        return False
    for panel in panels[:3]:
        for k in range(4):
            if np.array_equal(expected, np.rot90(panel, k)):
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        inp, out = _generate_case()
        if _validate_case(inp, out):
            return inp
    raise RuntimeError("failed to generate valid input")
