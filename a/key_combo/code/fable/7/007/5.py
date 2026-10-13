from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared_007_5", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


VISIBLE_FRAMES = 3
TOTAL_PANELS = VISIBLE_FRAMES + 1
PANEL_W = 6
PANEL_H_CHOICES = (6, 7, 8)
PANEL_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))

SPRITE_LIBRARY = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
)


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    border = int(grid[0, 0])
    top = grid[0]
    runs: list[tuple[int, int]] = []
    c = 0
    while c < grid.shape[1]:
        if int(top[c]) != border:
            c += 1
            continue
        start = c
        while c < grid.shape[1] and int(top[c]) == border:
            c += 1
        runs.append((start, c))

    assert len(runs) >= 2
    widths = {end - start for start, end in runs}
    assert len(widths) == 1
    panel_w = widths.pop()
    assert panel_w >= 4

    panels = [grid[:, start:end].copy() for start, end in runs]
    assert all(panel.shape == panels[0].shape for panel in panels)
    for panel in panels:
        assert np.all(panel[0, :] == border)
        assert np.all(panel[-1, :] == border)
        assert np.all(panel[:, 0] == border)
        assert np.all(panel[:, -1] == border)
    return panels, border


def _panel_inner(panel: np.ndarray) -> np.ndarray:
    h, w = panel.shape
    return crop_bbox(panel, (1, 1, h - 2, w - 2))


def _extract_sprite(panel: np.ndarray, bg: int) -> dict:
    inner = _panel_inner(panel)
    mask = inner != int(bg)
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    patch = crop_bbox(inner, box)
    sprite_mask = patch != int(bg)
    assert bool(sprite_mask.any())
    colors = set(map(int, patch[sprite_mask].ravel()))
    assert len(colors) == 1
    color = colors.pop()
    return {
        "mask": sprite_mask.astype(bool),
        "color": int(color),
        "pos": (int(r0), int(c0)),
        "shape": tuple(map(int, sprite_mask.shape)),
    }


def _render_panel(
    panel_h: int,
    panel_w: int,
    *,
    border: int,
    bg: int,
    sprite_color: int,
    sprite_mask: np.ndarray | None = None,
    pos: tuple[int, int] | None = None,
) -> np.ndarray:
    panel = np.full((panel_h, panel_w), int(border), dtype=int)
    panel[1:-1, 1:-1] = int(bg)
    if sprite_mask is not None:
        assert pos is not None
        patch = np.full(sprite_mask.shape, int(bg), dtype=int)
        patch[sprite_mask] = int(sprite_color)
        paste(panel[1:-1, 1:-1], patch, int(pos[0]), int(pos[1]), transparent=int(bg))
    return panel


def _advance(
    sprite_mask: np.ndarray,
    pos: tuple[int, int],
    vel: tuple[int, int],
    inner_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[bool, bool]]:
    r, c = map(int, pos)
    dr, dc = map(int, vel)
    sprite_h, sprite_w = map(int, sprite_mask.shape)
    max_r = int(inner_shape[0] - sprite_h)
    max_c = int(inner_shape[1] - sprite_w)
    assert max_r >= 0 and max_c >= 0

    bounce_r = not (0 <= r + dr <= max_r)
    bounce_c = not (0 <= c + dc <= max_c)

    next_mask = sprite_mask.copy()
    if bounce_r:
        dr = -dr
        next_mask = np.flipud(next_mask)
    if bounce_c:
        dc = -dc
        next_mask = np.fliplr(next_mask)

    nr = r + dr
    nc = c + dc
    assert 0 <= nr <= max_r
    assert 0 <= nc <= max_c
    return next_mask, (int(nr), int(nc)), (int(dr), int(dc)), (bool(bounce_r), bool(bounce_c))


def _state_equal(state: tuple[np.ndarray, tuple[int, int]], info: dict) -> bool:
    mask, pos = state
    return pos == info["pos"] and np.array_equal(mask, info["mask"])


def _infer_post_velocity(a: dict, b: dict, inner_shape: tuple[int, int]) -> tuple[int, int]:
    delta = (int(b["pos"][0] - a["pos"][0]), int(b["pos"][1] - a["pos"][1]))
    assert abs(delta[0]) == 1 and abs(delta[1]) == 1

    candidates: list[tuple[int, int]] = []
    for bounce_r in (False, True):
        for bounce_c in (False, True):
            expected_mask = a["mask"]
            if bounce_r:
                expected_mask = np.flipud(expected_mask)
            if bounce_c:
                expected_mask = np.fliplr(expected_mask)
            if not np.array_equal(expected_mask, b["mask"]):
                continue

            pre_vel = (
                -delta[0] if bounce_r else delta[0],
                -delta[1] if bounce_c else delta[1],
            )
            next_mask, next_pos, post_vel, bounce = _advance(a["mask"], a["pos"], pre_vel, inner_shape)
            if _state_equal((next_mask, next_pos), b) and bounce == (bounce_r, bounce_c):
                candidates.append((int(post_vel[0]), int(post_vel[1])))

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the next animation panel after a diagonally moving sprite bounces and flips at borders.

    Input structure:
    - The grid is a single horizontal strip of equal framed panels separated by one-color gaps.
    - Every visible panel contains the same monochrome asymmetric sprite on the same background; the final panel is blank.
    - Across panels the sprite moves one cell diagonally per frame inside each panel's inner area.

    Rule:
    - Read the visible frames in order and infer the current sprite position, orientation, and diagonal velocity.
    - A step that would leave the panel first bounces: the corresponding velocity component reverses, and the sprite is mirrored across the axis parallel to the struck border.
    - Continue the motion by one more step from the last visible frame and draw that predicted sprite in the blank panel.

    Key insight:
    - Border contact changes both trajectory and pose, so the history must be interpreted as a bounce-and-flip animation rather than as pure translation of a fixed shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panels, _border = _split_panels(g)
    assert len(panels) >= 3

    blank = panels[-1]
    bg = int(get_bg(_panel_inner(blank)))
    visible = [_extract_sprite(panel, bg) for panel in panels[:-1]]
    sprite_color = int(visible[0]["color"])
    assert all(info["color"] == sprite_color for info in visible)
    assert all(info["shape"] == visible[0]["shape"] for info in visible)

    inner_shape = _panel_inner(blank).shape
    velocities = [
        _infer_post_velocity(a, b, inner_shape)
        for a, b in zip(visible, visible[1:])
    ]
    vel = velocities[-1]
    next_mask, next_pos, _next_vel, _next_bounce = _advance(visible[-1]["mask"], visible[-1]["pos"], vel, inner_shape)

    out = blank.copy()
    patch = np.full(next_mask.shape, int(bg), dtype=int)
    patch[next_mask] = sprite_color
    paste(out[1:-1, 1:-1], patch, int(next_pos[0]), int(next_pos[1]), transparent=int(bg))
    return out


def _random_orientation(sprite_mask: np.ndarray) -> np.ndarray:
    out = sprite_mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _bounce_kind(bounce: tuple[bool, bool]) -> str:
    if bounce == (False, False):
        return "none"
    if bounce == (True, True):
        return "corner"
    return "single"


def _sequence_signature(states: list[tuple[np.ndarray, tuple[int, int]]]) -> tuple:
    return tuple((tuple(state[0].astype(int).ravel()), state[1]) for state in states)


def _build_candidate() -> tuple[np.ndarray, dict] | None:
    panel_h = random.choice(PANEL_H_CHOICES)
    panel_w = PANEL_W
    inner_shape = (panel_h - 2, panel_w - 2)
    sprite_base = random.choice(SPRITE_LIBRARY)
    sprite_mask = _random_orientation(sprite_base)
    sprite_h, sprite_w = sprite_mask.shape
    max_r = inner_shape[0] - sprite_h
    max_c = inner_shape[1] - sprite_w
    if max_r < 1 or max_c < 1:
        return None

    pos = (random.randint(0, max_r), random.randint(0, max_c))
    vel = random.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))

    states: list[tuple[np.ndarray, tuple[int, int]]] = [(sprite_mask.copy(), pos)]
    bounces: list[tuple[bool, bool]] = []
    cur_mask = sprite_mask.copy()
    cur_pos = pos
    cur_vel = vel
    for _ in range(TOTAL_PANELS - 1):
        cur_mask, cur_pos, cur_vel, bounce = _advance(cur_mask, cur_pos, cur_vel, inner_shape)
        states.append((cur_mask.copy(), cur_pos))
        bounces.append(bounce)

    if not any(any(bounce) for bounce in bounces[:VISIBLE_FRAMES - 1]):
        return None

    target_need_bounce = random.random() < 0.65
    if any(bounces[-1]) != target_need_bounce:
        return None

    history_mode = random.choice(("any", "single", "corner"))
    history_kinds = {_bounce_kind(bounce) for bounce in bounces[:VISIBLE_FRAMES - 1]}
    if history_mode == "single" and "single" not in history_kinds:
        return None
    if history_mode == "corner" and "corner" not in history_kinds:
        return None

    if len(set(state[1] for state in states)) != len(states):
        return None

    if len({_sequence_signature(states[:VISIBLE_FRAMES]), _sequence_signature(states[1:])}) != 2:
        return None

    bg, border, separator, sprite_color = random.sample(PALETTE, 4)
    panels = [
        _render_panel(
            panel_h,
            panel_w,
            border=border,
            bg=bg,
            sprite_color=sprite_color,
            sprite_mask=state_mask,
            pos=state_pos,
        )
        for state_mask, state_pos in states[:VISIBLE_FRAMES]
    ]
    panels.append(_render_panel(panel_h, panel_w, border=border, bg=bg, sprite_color=sprite_color))

    total_w = TOTAL_PANELS * panel_w + (TOTAL_PANELS - 1) * PANEL_GAP
    grid = np.full((panel_h, total_w), int(separator), dtype=int)
    for idx, panel in enumerate(panels):
        paste(grid, panel, 0, idx * (panel_w + PANEL_GAP))

    expected = _render_panel(
        panel_h,
        panel_w,
        border=border,
        bg=bg,
        sprite_color=sprite_color,
        sprite_mask=states[-1][0],
        pos=states[-1][1],
    )
    return grid, {"expected": expected, "bounces": bounces, "inner_shape": inner_shape}


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, meta = built
        try:
            out = main(grid)
        except Exception:
            continue
        if not np.array_equal(out, meta["expected"]):
            continue
        return grid
    raise RuntimeError("failed to generate a valid bouncing-panel ARC input")
