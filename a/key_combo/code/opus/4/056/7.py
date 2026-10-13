from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BORDER = 5
BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SPRITE_MASKS = (
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
)
MAX_GENERATION_ATTEMPTS = 400


def _panel_slices(grid: np.ndarray) -> list[tuple[int, int]]:
    grey_cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == BORDER)]
    assert len(grey_cols) == 4, "expected four shared vertical borders"
    gaps = [grey_cols[i + 1] - grey_cols[i] for i in range(3)]
    assert len(set(gaps)) == 1, "expected equal-width panels"
    panel_w = int(gaps[0] + 1)
    assert grey_cols == [0, panel_w - 1, 2 * (panel_w - 1), 3 * (panel_w - 1)]
    assert grid.shape[1] == 3 * (panel_w - 1) + 1
    assert np.all(grid[0, :] == BORDER) and np.all(grid[-1, :] == BORDER)
    return [(grey_cols[i], grey_cols[i + 1]) for i in range(3)]


def _extract_panel_sprite(panel: np.ndarray) -> dict:
    assert panel.shape[0] >= 4 and panel.shape[1] >= 4
    assert np.all(panel[0, :] == BORDER) and np.all(panel[-1, :] == BORDER)
    assert np.all(panel[:, 0] == BORDER) and np.all(panel[:, -1] == BORDER)
    interior = panel[1:-1, 1:-1]
    components = find_components(interior, bg=BG, connectivity=4, monochromatic=True)
    assert len(components) == 1, "expected one monochrome sprite per panel"
    comp = components[0]
    r0, c0, r1, c1 = comp["bbox"]
    sprite = crop_bbox(interior, comp["bbox"])
    assert np.all((sprite == BG) | (sprite == int(comp["color"])))
    return {
        "color": int(comp["color"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "sprite": sprite,
        "mask": sprite != BG,
        "shape": sprite.shape,
        "interior_shape": interior.shape,
    }


def _minimal_period(values: list[int]) -> int:
    for period in range(1, len(values) + 1):
        if all(values[i] == values[i % period] for i in range(len(values))):
            return period
    raise AssertionError("sequence must have a period")


def _next_cycle_value(values: list[int]) -> int:
    period = _minimal_period(values)
    return int(values[len(values) % period])


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    slices = _panel_slices(g)
    panels = [_extract_panel_sprite(g[:, c0 : c1 + 1]) for c0, c1 in slices]

    ref_mask = panels[0]["mask"]
    ref_shape = panels[0]["shape"]
    for panel in panels[1:]:
        assert panel["shape"] == ref_shape, "sprite bbox must stay constant"
        assert np.array_equal(panel["mask"], ref_mask), "sprite shape must be identical"

    positions = [(panel["bbox"][0], panel["bbox"][1]) for panel in panels]
    dr1 = int(positions[1][0] - positions[0][0])
    dc1 = int(positions[1][1] - positions[0][1])
    dr2 = int(positions[2][0] - positions[1][0])
    dc2 = int(positions[2][1] - positions[1][1])
    assert (dr1, dc1) == (dr2, dc2), "expected constant motion"

    next_r = int(positions[2][0] + dr1)
    next_c = int(positions[2][1] + dc1)
    ih, iw = panels[0]["interior_shape"]
    sh, sw = ref_shape
    assert 0 <= next_r <= ih - sh and 0 <= next_c <= iw - sw, "extrapolated sprite must stay inside"

    colors = [int(panel["color"]) for panel in panels]
    next_color = _next_cycle_value(colors)

    return {
        "panel_h": int(g.shape[0]),
        "panel_w": int(slices[0][1] - slices[0][0] + 1),
        "sprite_mask": ref_mask,
        "sprite_shape": ref_shape,
        "positions": positions,
        "delta": (dr1, dc1),
        "colors": colors,
        "next_color": next_color,
        "next_pos": (next_r, next_c),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the next framed sprite by continuing both its motion and its palette cycle.

    Input structure:
    - The input is a single row of three equal panels that share grey vertical borders.
    - Each panel has a grey one-cell frame, a black interior, and exactly one small
      monochrome sprite of the same shape.
    - Across the three panels, the sprite's bbox moves by a constant panel-to-panel
      offset and its color follows a short periodic cycle.

    Rule:
    - Detect the three panel interiors from the shared grey borders.
    - In each panel, locate the sprite and read its top-left bbox offset relative to
      the interior origin, then recover the constant motion vector.
    - Infer the next sprite color by extending the shortest visible color period.
    - Output one standalone framed panel whose sprite is placed at the extrapolated
      next position and repainted with the extrapolated next color.

    Key insight:
    - Position and color are encoded by two independent panel-wise sequences; the
      answer is only correct when the next frame satisfies both at once.
    """
    info = _analyze(grid)

    out = np.full((info["panel_h"], info["panel_w"]), BG, dtype=int)
    out[0, :] = BORDER
    out[-1, :] = BORDER
    out[:, 0] = BORDER
    out[:, -1] = BORDER

    sprite = np.where(info["sprite_mask"], info["next_color"], BG).astype(int)
    paste(out, sprite, 1 + info["next_pos"][0], 1 + info["next_pos"][1], transparent=BG)
    return out


def _build_input(
    panel_h: int,
    panel_w: int,
    sprite_mask: np.ndarray,
    start: tuple[int, int],
    delta: tuple[int, int],
    colors: list[int],
) -> np.ndarray:
    width = 3 * (panel_w - 1) + 1
    grid = np.full((panel_h, width), BG, dtype=int)
    grid[0, :] = BORDER
    grid[-1, :] = BORDER
    for c in (0, panel_w - 1, 2 * (panel_w - 1), 3 * (panel_w - 1)):
        grid[:, c] = BORDER

    sh, sw = sprite_mask.shape
    for idx in range(3):
        pr = int(start[0] + idx * delta[0])
        pc = int(start[1] + idx * delta[1])
        patch = np.where(sprite_mask, int(colors[idx]), BG).astype(int)
        panel_left = idx * (panel_w - 1)
        paste(grid, patch, 1 + pr, panel_left + 1 + pc, transparent=BG)
        assert 0 <= pr <= panel_h - 2 - sh
        assert 0 <= pc <= panel_w - 2 - sw

    return grid


def _candidate_deltas(ih: int, iw: int, sh: int, sw: int) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    max_r = ih - sh
    max_c = iw - sw
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            starts = []
            for r0 in range(max_r + 1):
                for c0 in range(max_c + 1):
                    ok = True
                    for step in range(4):
                        rr = r0 + step * dr
                        cc = c0 + step * dc
                        if not (0 <= rr <= max_r and 0 <= cc <= max_c):
                            ok = False
                            break
                    if ok:
                        starts.append((int(r0), int(c0)))
            if starts:
                candidates.append(((int(dr), int(dc)), starts))
    return candidates


def _sample_color_sequence() -> list[int]:
    if random.random() < 0.5:
        a, b = random.sample(PALETTE, 2)
        return [int(a), int(b), int(a)]
    a, b, c = random.sample(PALETTE, 3)
    return [int(a), int(b), int(c)]


def _validate_example(inp: np.ndarray) -> bool:
    try:
        info = _analyze(inp)
    except AssertionError:
        return False

    if info["delta"] == (0, 0):
        return False
    if info["sprite_shape"] != (2, 2):
        return False

    period = _minimal_period(info["colors"])
    if period not in (2, 3):
        return False

    out = main(inp)
    if out.shape != (info["panel_h"], info["panel_w"]):
        return False

    observed_panels = []
    panel_w = info["panel_w"]
    for idx in range(3):
        left = idx * (panel_w - 1)
        observed_panels.append(inp[:, left : left + panel_w])
    if any(np.array_equal(out, panel) for panel in observed_panels):
        return False

    panel_sprite_cells = int(np.sum(info["sprite_mask"]))
    if int(np.sum(out == info["next_color"])) != panel_sprite_cells:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        panel_h = random.randint(6, 10)
        panel_w = random.randint(6, 10)
        if 3 * (panel_w - 1) + 1 > 30:
            continue

        sprite_mask = random.choice(SPRITE_MASKS).astype(bool)
        ih, iw = panel_h - 2, panel_w - 2
        sh, sw = sprite_mask.shape
        candidates = _candidate_deltas(ih, iw, sh, sw)
        if not candidates:
            continue

        delta, starts = random.choice(candidates)
        start = random.choice(starts)
        colors = _sample_color_sequence()
        grid = _build_input(panel_h, panel_w, sprite_mask, start, delta, colors)

        if _validate_example(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
