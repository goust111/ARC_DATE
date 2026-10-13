from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
WALL_COLOR = 5
SEP_COLOR = 8
SLIDER_COLORS = [1, 2, 3, 4, 6, 7, 9]
SIDES = ("top", "bottom", "left", "right")


def _bg_color(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    rel = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in rel))


def _edge_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "bottom":
        mask[h - 1, :] = True
    elif side == "left":
        mask[:, 0] = True
    elif side == "right":
        mask[:, w - 1] = True
    else:
        raise ValueError(side)
    return mask


def _split_panels(g: np.ndarray) -> list[np.ndarray]:
    h, w = g.shape
    bg = _bg_color(g)
    candidates: list[list[np.ndarray]] = []

    for n in range(3, 7):
        if (w - (n - 1)) % n == 0:
            pw = (w - (n - 1)) // n
            if pw >= 4:
                divs = [i * (pw + 1) - 1 for i in range(1, n)]
                colors = [int(g[0, c]) for c in divs]
                if len(set(colors)) == 1 and colors[0] != bg:
                    if all(np.all(g[:, c] == colors[0]) for c in divs):
                        panels = [g[:, i * (pw + 1) : i * (pw + 1) + pw].copy() for i in range(n)]
                        candidates.append(panels)

        if (h - (n - 1)) % n == 0:
            ph = (h - (n - 1)) // n
            if ph >= 4:
                divs = [i * (ph + 1) - 1 for i in range(1, n)]
                colors = [int(g[r, 0]) for r in divs]
                if len(set(colors)) == 1 and colors[0] != bg:
                    if all(np.all(g[r, :] == colors[0]) for r in divs):
                        panels = [g[i * (ph + 1) : i * (ph + 1) + ph, :].copy() for i in range(n)]
                        candidates.append(panels)

    assert len(candidates) == 1, "expected one unambiguous row or column of panels"
    return candidates[0]


def _find_wall(panel: np.ndarray, bg: int) -> tuple[int, tuple[int, int], np.ndarray, str]:
    h, w = panel.shape
    specs = [
        ("top", (-1, 0), panel[0, :]),
        ("bottom", (1, 0), panel[h - 1, :]),
        ("left", (0, -1), panel[:, 0]),
        ("right", (0, 1), panel[:, w - 1]),
    ]
    candidates = []
    for side, direction, values in specs:
        color = int(values[0])
        if color == bg or not np.all(values == color):
            continue
        mask = _edge_mask(panel.shape, side)
        if np.array_equal(panel == color, mask):
            candidates.append((color, direction, mask, side))
    assert len(candidates) == 1, "expected exactly one full edge wall"
    return candidates[0]


def _panel_objects(panel: np.ndarray, bg: int, wall_color: int) -> dict[int, dict]:
    colors = sorted(set(map(int, np.unique(panel))) - {bg, wall_color})
    objects: dict[int, dict] = {}
    for color in colors:
        coords = np.argwhere(panel == color).astype(int)
        assert coords.size > 0
        objects[color] = {
            "coords": coords,
            "bbox": _bbox(coords),
            "anchor": tuple(map(int, coords.min(axis=0))),
            "shape": _norm_shape(coords),
        }
    return objects


def _velocity_from_anchors(
    anchors: list[tuple[int, int]], direction: tuple[int, int]
) -> tuple[int, int]:
    dr, dc = direction
    deltas = [
        (anchors[i + 1][0] - anchors[i][0], anchors[i + 1][1] - anchors[i][1])
        for i in range(len(anchors) - 1)
    ]
    last = deltas[-1]
    if last == (0, 0):
        return (0, 0)

    if dr:
        assert last[1] == 0 and last[0] * dr > 0
    else:
        assert last[0] == 0 and last[1] * dc > 0
    for delta in deltas:
        assert delta == last or delta == (0, 0), "visible motion must be constant until it stops"
    return last


def _settle_key(obj: dict, direction: tuple[int, int]) -> tuple[int, int, int, int]:
    dr, dc = direction
    r0, c0, r1, c1 = obj["bbox"]
    if dr == -1:
        return (r0, c0, c1, r1)
    if dr == 1:
        return (-r1, c0, c1, r0)
    if dc == -1:
        return (c0, r0, r1, c1)
    assert dc == 1
    return (-c1, r0, r1, c0)


def _advance(
    coords: np.ndarray, velocity: tuple[int, int], occupied: np.ndarray
) -> tuple[np.ndarray, bool]:
    vr, vc = velocity
    mag = max(abs(vr), abs(vc))
    if mag == 0:
        return coords.copy(), False
    step = np.array([0 if vr == 0 else int(vr / abs(vr)), 0 if vc == 0 else int(vc / abs(vc))])
    h, w = occupied.shape
    current = coords.astype(int).copy()
    truncated = False
    for _ in range(mag):
        nxt = current + step
        in_bounds = (
            np.all(0 <= nxt[:, 0])
            and np.all(nxt[:, 0] < h)
            and np.all(0 <= nxt[:, 1])
            and np.all(nxt[:, 1] < w)
        )
        if not in_bounds or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
            truncated = True
            break
        current = nxt
    return current, truncated


def _next_panel(panels: list[np.ndarray]) -> np.ndarray:
    last = panels[-1]
    bg = _bg_color(last)
    wall_color, direction, wall_mask, _side = _find_wall(last, bg)

    for panel in panels[:-1]:
        color, other_direction, other_mask, _ = _find_wall(panel, bg)
        assert color == wall_color and other_direction == direction
        assert np.array_equal(other_mask, wall_mask)

    states = [_panel_objects(panel, bg, wall_color) for panel in panels]
    colors = set(states[-1])
    assert colors and all(set(state) == colors for state in states)

    moving: list[dict] = []
    settled: list[dict] = []
    for color in sorted(colors):
        shapes = [state[color]["shape"] for state in states]
        assert all(shape == shapes[0] for shape in shapes), "slider shape must not change"
        anchors = [state[color]["anchor"] for state in states]
        velocity = _velocity_from_anchors(anchors, direction)
        obj = dict(states[-1][color])
        obj["color"] = color
        obj["velocity"] = velocity
        if velocity == (0, 0):
            settled.append(obj)
        else:
            moving.append(obj)

    out = np.full_like(last, bg)
    out[wall_mask] = wall_color
    occupied = wall_mask.copy()

    for obj in sorted(settled, key=lambda item: _settle_key(item, direction)):
        coords = obj["coords"]
        assert not np.any(occupied[coords[:, 0], coords[:, 1]])
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
        occupied[coords[:, 0], coords[:, 1]] = True

    for obj in sorted(moving, key=lambda item: _settle_key(item, direction)):
        coords, _truncated = _advance(obj["coords"], obj["velocity"], occupied)
        assert not np.any(occupied[coords[:, 0], coords[:, 1]])
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
        occupied[coords[:, 0], coords[:, 1]] = True

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next panel after sliding blocks settle their first collision.

    Input structure:
    - The input is a row or column of equal-size panels separated by solid divider lines.
    - Each panel contains the same fixed full edge wall and several uniquely colored rigid sliders.
    - Slider colors identify the same object across panels; each moving object advances by a constant
      frame velocity toward the wall, while already settled sliders remain fixed.

    Rule:
    - Return one panel: the next frame after the last visible panel.
    - Read every slider's current velocity from the panel sequence.
    - Starting with the wall and zero-velocity sliders fixed, settle moving sliders from nearest to
      farthest from the wall; a slider moves at most one frame step and stops early if the unit path
      would hit the wall or a previously settled slider.

    Key insight:
    - Pure linear extrapolation is tempting, but the requested next frame is exactly the collision
      frame, so per-slider velocity must be combined with nearest-first collision truncation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    panels = _split_panels(g)
    return _next_panel(panels)


def _rect_coords(r0: int, c0: int, height: int, width: int) -> np.ndarray:
    coords = [(r, c) for r in range(r0, r0 + height) for c in range(c0, c0 + width)]
    return np.array(coords, dtype=int)


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _transform_panel(panel: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return panel.copy()
    if side == "bottom":
        return np.flipud(panel).copy()
    if side == "left":
        return np.rot90(panel, 1).copy()
    if side == "right":
        return np.rot90(panel, -1).copy()
    raise ValueError(side)


def _assemble(panels: list[np.ndarray], horizontal: bool) -> np.ndarray:
    ph, pw = panels[0].shape
    n = len(panels)
    if horizontal:
        grid = np.full((ph, n * pw + n - 1), SEP_COLOR, dtype=int)
        for i, panel in enumerate(panels):
            c0 = i * (pw + 1)
            grid[:, c0 : c0 + pw] = panel
        return grid

    grid = np.full((n * ph + n - 1, pw), SEP_COLOR, dtype=int)
    for i, panel in enumerate(panels):
        r0 = i * (ph + 1)
        grid[r0 : r0 + ph, :] = panel
    return grid


def _top_wall_grid(h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = WALL_COLOR
    return grid


def _lane_spans(w: int) -> dict[str, tuple[int, int]]:
    roles = ["chain", "blocker", "free"]
    chain_width = random.choice([1, 2])
    free_width = random.choice([1, 2])
    blocker_width = random.choice([1, 2])
    widths = {"chain": chain_width, "blocker": blocker_width, "free": free_width}
    while sum(widths.values()) + 2 > w - 2:
        role = random.choice([role for role in roles if widths[role] > 1])
        widths[role] -= 1

    order = roles[:]
    random.shuffle(order)
    total = sum(widths.values()) + 2
    spare = (w - 2) - total
    left_pad = random.randint(0, spare)
    gaps = [1, 1]
    for _ in range(spare - left_pad):
        gaps[random.randrange(2)] += 1

    spans: dict[str, tuple[int, int]] = {}
    c = 1 + left_pad
    for idx, role in enumerate(order):
        width = widths[role]
        spans[role] = (c, c + width - 1)
        c += width
        if idx < 2:
            c += gaps[idx]
    return spans


def _build_objects(h: int, w: int) -> list[dict]:
    spans = _lane_spans(w)
    colors = random.sample(SLIDER_COLORS, 5)
    objects: list[dict] = []

    chain_c0, chain_c1 = spans["chain"]
    chain_w = chain_c1 - chain_c0 + 1
    objects.append(
        {"color": colors[0], "last": _rect_coords(2, chain_c0, 1, chain_w), "velocity": (-2, 0)}
    )
    objects.append(
        {"color": colors[1], "last": _rect_coords(3, chain_c0, 1, chain_w), "velocity": (-2, 0)}
    )

    block_c0, block_c1 = spans["blocker"]
    block_w = block_c1 - block_c0 + 1
    objects.append(
        {"color": colors[2], "last": _rect_coords(1, block_c0, 1, block_w), "velocity": (0, 0)}
    )
    objects.append(
        {"color": colors[3], "last": _rect_coords(3, block_c0, 1, block_w), "velocity": (-2, 0)}
    )

    free_c0, free_c1 = spans["free"]
    free_w = free_c1 - free_c0 + 1
    free_h = 2 if h == 9 and random.random() < 0.45 else 1
    free_top = random.choice([4, 5])
    objects.append(
        {"color": colors[4], "last": _rect_coords(free_top, free_c0, free_h, free_w), "velocity": (-1, 0)}
    )
    return objects


def _render_top_frame(h: int, w: int, objects: list[dict], frame: int, n_frames: int) -> np.ndarray:
    grid = _top_wall_grid(h, w)
    back_steps = n_frames - 1 - frame
    occupied = grid != BG
    for obj in objects:
        velocity = np.array(obj["velocity"], dtype=int)
        coords = obj["last"] - back_steps * velocity
        assert np.all(0 <= coords[:, 0]) and np.all(coords[:, 0] < h)
        assert np.all(0 <= coords[:, 1]) and np.all(coords[:, 1] < w)
        assert not np.any(occupied[coords[:, 0], coords[:, 1]])
        _paint(grid, coords, int(obj["color"]))
        occupied[coords[:, 0], coords[:, 1]] = True
    return grid


def _expected_top_next(h: int, w: int, objects: list[dict]) -> tuple[np.ndarray, int, int]:
    out = _top_wall_grid(h, w)
    occupied = out != BG
    settled = []
    moving = []
    for obj in objects:
        coords = obj["last"]
        entry = {"coords": coords, "bbox": _bbox(coords), **obj}
        if obj["velocity"] == (0, 0):
            settled.append(entry)
        else:
            moving.append(entry)

    full_moves = 0
    truncated_moves = 0
    for obj in sorted(settled, key=lambda item: _settle_key(item, (-1, 0))):
        coords = obj["coords"]
        _paint(out, coords, int(obj["color"]))
        occupied[coords[:, 0], coords[:, 1]] = True

    for obj in sorted(moving, key=lambda item: _settle_key(item, (-1, 0))):
        coords, truncated = _advance(obj["coords"], obj["velocity"], occupied)
        if truncated:
            truncated_moves += 1
        else:
            full_moves += 1
        _paint(out, coords, int(obj["color"]))
        occupied[coords[:, 0], coords[:, 1]] = True
    return out, truncated_moves, full_moves


def _gen_attempt() -> np.ndarray | None:
    h = random.choice([8, 9])
    w = random.choice([8, 9])
    n_frames = 3
    side = random.choice(SIDES)
    horizontal = random.choice([False, True])
    objects = _build_objects(h, w)

    try:
        top_panels = [_render_top_frame(h, w, objects, frame, n_frames) for frame in range(n_frames)]
        expected_top, truncated, full_moves = _expected_top_next(h, w, objects)
    except AssertionError:
        return None

    if truncated < 3 or full_moves < 1:
        return None

    panels = [_transform_panel(panel, side) for panel in top_panels]
    expected = _transform_panel(expected_top, side)
    inp = _assemble(panels, horizontal)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, panels[-1]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid collision-settlement sequence")
