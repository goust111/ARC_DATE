"""ARC task: extrapolate a bouncing sprite animation by one frame."""

from __future__ import annotations

import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox in the common ARC ``(row, column)`` convention."""
    points = np.argwhere(mask)
    assert len(points), "expected a nonempty sprite"
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _split(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"
    total_h, width = g.shape
    assert total_h >= 7 and width >= 3 and (total_h - 3) % 4 == 0
    panel_h = (total_h - 3) // 4
    separators = [panel_h, 2 * panel_h + 1, 3 * panel_h + 2]
    sep_color = int(g[separators[0], 0])
    assert all(np.all(g[r] == sep_color) for r in separators)

    panels = []
    start = 0
    for r in separators:
        panels.append(g[start:r].copy())
        start = r + 1
    panels.append(g[start:].copy())
    assert all(p.shape == (panel_h, width) for p in panels)

    values, counts = np.unique(np.concatenate([p.ravel() for p in panels]), return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    assert sep_color != bg
    return panels, bg, sep_color


def _state(panel: np.ndarray, bg: int) -> tuple[int, int, np.ndarray, int]:
    mask = panel != bg
    r0, c0, r1, c1 = _bbox(mask)
    colors = np.unique(panel[mask])
    assert len(colors) == 1, "the sprite must be monochromatic"
    local = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    return r0, c0, local, int(colors[0])


def _advance_axis(pos: int, speed: int, limit: int) -> tuple[int, int, bool]:
    """Move one bbox anchor, reflecting it at the inclusive anchor interval."""
    assert limit > 0
    nxt = pos + speed
    bounced = False
    # Candidate speeds are no larger than the free anchor span, so one reflection
    # normally suffices; the loop also makes the helper safe at an endpoint.
    while nxt < 0 or nxt > limit:
        if nxt < 0:
            nxt = -nxt
        else:
            nxt = 2 * limit - nxt
        speed = -speed
        bounced = True
    return nxt, speed, bounced


def _advance(
    state: tuple[int, int, np.ndarray, int],
    velocity: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int, np.ndarray, int], tuple[int, int], tuple[bool, bool]]:
    r, c, local, color = state
    height, width = shape
    local_h, local_w = local.shape
    nr, vr, bounce_r = _advance_axis(r, velocity[0], height - local_h)
    nc, vc, bounce_c = _advance_axis(c, velocity[1], width - local_w)
    if bounce_r:
        local = np.flip(local, axis=0).copy()
    if bounce_c:
        local = np.flip(local, axis=1).copy()
    return (nr, nc, local, color), (vr, vc), (bounce_r, bounce_c)


def _render(state: tuple[int, int, np.ndarray, int], shape: tuple[int, int], bg: int) -> np.ndarray:
    r, c, local, color = state
    out = np.full(shape, bg, dtype=int)
    patch = out[r : r + local.shape[0], c : c + local.shape[1]]
    patch[local] = color
    return out


def _candidate_velocities(panels: list[np.ndarray], bg: int) -> list[tuple[int, int]]:
    initial = _state(panels[0], bg)
    h, w = panels[0].shape
    local_h, local_w = initial[2].shape
    row_span, col_span = h - local_h, w - local_w
    candidates: list[tuple[int, int]] = []
    for vr in range(-row_span, row_span + 1):
        for vc in range(-col_span, col_span + 1):
            if vr == vc == 0:
                continue
            state = (initial[0], initial[1], initial[2].copy(), initial[3])
            velocity = (vr, vc)
            matches = True
            for observed in panels[1:]:
                state, velocity, _ = _advance(state, velocity, (h, w))
                if not np.array_equal(_render(state, (h, w), bg), observed):
                    matches = False
                    break
            if matches:
                candidates.append((vr, vc))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate one frame of a rigid sprite moving with elastic boundary bounces.

    Input structure:
    - Four equal-sized animation panels are stacked vertically, separated by three
      solid full-width rows; every panel contains one asymmetric monochrome sprite
      on a shared background.

    Rule:
    - Find the unique constant row/column velocity that reproduces all four frames.
      Move the sprite bbox by that velocity; when an anchor passes a panel edge,
      fold its overshoot back by the same distance, reverse that velocity component,
      and flip the sprite mask along the collided axis. Return the fifth panel.

    Key insight:
    - A collision changes both trajectory and orientation, so the four-frame history
      must be interpreted as elastic motion rather than unrestricted translation.
    """
    panels, bg, _ = _split(grid)
    first_states = [_state(panel, bg) for panel in panels]
    colors = {state[3] for state in first_states}
    cell_counts = {int(state[2].sum()) for state in first_states}
    sizes = {state[2].shape for state in first_states}
    assert len(colors) == len(cell_counts) == len(sizes) == 1

    candidates = _candidate_velocities(panels, bg)
    assert len(candidates) == 1, "four panels must determine one velocity"
    state = first_states[0]
    velocity = candidates[0]
    for _ in range(4):
        state, velocity, _ = _advance(state, velocity, panels[0].shape)
    return _render(state, panels[0].shape, bg)


def _random_sprite() -> np.ndarray:
    for _ in range(100):
        h = random.randint(2, 3)
        w = random.randint(2, 3)
        mask = np.zeros((h, w), dtype=bool)
        r, c = random.randrange(h), random.randrange(w)
        mask[r, c] = True
        target = random.randint(3, h * w - 1)
        while int(mask.sum()) < target:
            cells = list(map(tuple, np.argwhere(mask)))
            r, c = random.choice(cells)
            dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
            if 0 <= r + dr < h and 0 <= c + dc < w:
                mask[r + dr, c + dc] = True
        if (np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))
                and not np.array_equal(mask, np.flip(mask, 0))
                and not np.array_equal(mask, np.flip(mask, 1))):
            return mask
    raise RuntimeError("failed to make an asymmetric sprite")


def _stack(frames: list[np.ndarray], separator: int) -> np.ndarray:
    row = np.full((1, frames[0].shape[1]), separator, dtype=int)
    pieces: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        if i:
            pieces.append(row)
        pieces.append(frame)
    return np.vstack(pieces)


def generate_input() -> np.ndarray:
    """Generate varied, nondegenerate histories with visible past and next bounces."""
    for _ in range(4000):
        h = random.randint(5, 6)                 # stacked height is at most 27
        w = random.randint(7, 12)
        bg, color, separator = random.sample(range(10), 3)
        local = _random_sprite()
        row_span, col_span = h - local.shape[0], w - local.shape[1]
        if row_span < 2 or col_span < 2:
            continue

        # Zero components are represented, but diagonal motion is more common and
        # makes independent horizontal/vertical reflection evidence easy to see.
        vr = 0 if random.random() < 0.18 else random.choice(
            [v for v in range(-min(3, row_span), min(3, row_span) + 1) if v])
        vc = 0 if random.random() < 0.18 else random.choice(
            [v for v in range(-min(3, col_span), min(3, col_span) + 1) if v])
        if vr == vc == 0:
            continue
        state = (random.randint(0, row_span), random.randint(0, col_span), local, color)
        velocity = (vr, vc)
        frames = [_render(state, (h, w), bg)]
        bounce_flags = []
        for _step in range(4):
            state, velocity, flags = _advance(state, velocity, (h, w))
            bounce_flags.append(flags)
            frames.append(_render(state, (h, w), bg))

        # Both the historical inference and the requested extrapolation must need
        # the bounce rule. The asymmetric mask makes every indicated flip visible.
        if not any(a or b for a, b in bounce_flags[:3]):
            continue
        if not any(bounce_flags[3]):
            continue
        if any(np.array_equal(frames[4], frame) for frame in frames[:4]):
            continue

        inp = _stack(frames[:4], separator)
        candidates = _candidate_velocities(frames[:4], bg)
        if candidates != [(vr, vc)]:
            continue
        if not np.array_equal(main(inp), frames[4]):
            continue
        return inp
    raise RuntimeError("failed to generate a unique bouncing animation")
