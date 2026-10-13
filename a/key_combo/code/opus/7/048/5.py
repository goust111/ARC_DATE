from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7

PANEL_H = 8
PANEL_W = 7
INPUT_W = PANEL_W * 3 + 2
LAYER_COLORS = {0: YELLOW, 1: ORANGE, 2: BLUE}
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    red_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]
    assert len(red_cols) == 2, "expected exactly two full-height red divider columns"
    c0, c1 = red_cols
    assert c0 < c1
    left = grid[:, :c0]
    middle = grid[:, c0 + 1 : c1]
    right = grid[:, c1 + 1 :]
    assert left.shape == middle.shape == right.shape == (PANEL_H, PANEL_W)
    return [left, middle, right]


def _composite_with_sources(panels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    composite = np.zeros((PANEL_H, PANEL_W), dtype=int)
    sources = -np.ones((PANEL_H, PANEL_W), dtype=int)
    for layer, panel in enumerate(panels):
        mask = panel != BLACK
        composite[mask] = panel[mask]
        sources[mask] = layer
    return composite, sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each overlaid connected component by the source panel that contributes the most visible pixels.

    Input structure:
    - The grid is split into three 8x7 panels by two full-height red divider columns.
    - The left, middle, and right panels contain yellow, orange, and blue patterns respectively on black background.
    - Black is transparent inside each panel; panel order encodes back-to-front layering.

    Rule:
    - Remove the red divider columns and overlay the three panels onto a single 8x7 canvas: left first, then middle, then right.
    - Record which layer supplies each final visible non-black pixel, then find all 4-connected components in the composite image.
    - Repaint each component to the color of the layer with the largest visible contribution, breaking ties in favor of the frontmost layer; erase singleton components to black.

    Key insight:
    - The final color of a visible component depends on source-layer majority after occlusion, not on the surface color seen at any individual cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (PANEL_H, INPUT_W)

    panels = _split_panels(g)
    composite, sources = _composite_with_sources(panels)

    out = np.zeros((PANEL_H, PANEL_W), dtype=int)
    for comp in _components(composite != BLACK):
        if len(comp) == 1:
            continue
        counts = [0, 0, 0]
        for r, c in comp:
            counts[int(sources[r, c])] += 1
        winner = max(range(3), key=lambda layer: (counts[layer], layer))
        color = LAYER_COLORS[winner]
        for r, c in comp:
            out[r, c] = color
    return out


def _build_input_grid(panels: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [panels[0], np.full((PANEL_H, 1), RED, dtype=int), panels[1], np.full((PANEL_H, 1), RED, dtype=int), panels[2]],
        axis=1,
    )


def _candidate_blue_shapes(bridge: tuple[int, int], direction: int) -> list[set[tuple[int, int]]]:
    r, c = bridge
    shapes: list[set[tuple[int, int]]] = []

    vertical = {(r, c)}
    if r - 1 >= 0:
        vertical.add((r - 1, c))
    if r + 1 < PANEL_H:
        vertical.add((r + 1, c))
    if len(vertical) >= 3:
        shapes.append(vertical)

    horizontal = {(r, c)}
    if c - 1 >= 0:
        horizontal.add((r, c - 1))
    if c + 1 < PANEL_W:
        horizontal.add((r, c + 1))
    if len(horizontal) >= 3:
        shapes.append(horizontal)

    bent = set(vertical if len(vertical) >= 3 else horizontal)
    arm_r = r + direction
    if 0 <= arm_r < PANEL_H:
        bent.add((arm_r, c))
        side_c = c + random.choice([-1, 1])
        if 0 <= side_c < PANEL_W:
            bent.add((arm_r, side_c))
        if 3 <= len(bent) <= 5:
            shapes.append(bent)

    side_bent = {(r, c)}
    for dc in (-1, 1):
        nc = c + dc
        if 0 <= nc < PANEL_W:
            side_bent.add((r, nc))
    if 0 <= r + direction < PANEL_H:
        side_bent.add((r + direction, c))
    if len(side_bent) >= 3:
        shapes.append(side_bent)

    unique: list[set[tuple[int, int]]] = []
    seen = set()
    for shape in shapes:
        key = tuple(sorted(shape))
        if key not in seen:
            seen.add(key)
            unique.append(shape)
    return unique


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                    out[nr, nc] = True
    return out


def _random_connected_patch(rng: random.Random, allowed: np.ndarray, size: int, tries: int = 80) -> set[tuple[int, int]] | None:
    cells = [(r, c) for r in range(PANEL_H) for c in range(PANEL_W) if allowed[r, c]]
    if not cells:
        return None
    for _ in range(tries):
        patch = {rng.choice(cells)}
        frontier = set()
        while len(patch) < size:
            frontier.clear()
            for r, c in patch:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < PANEL_H and 0 <= nc < PANEL_W and allowed[nr, nc] and (nr, nc) not in patch:
                        frontier.add((nr, nc))
            if not frontier:
                break
            patch.add(rng.choice(tuple(frontier)))
        if len(patch) != size:
            continue
        rows = {r for r, _ in patch}
        cols = {c for _, c in patch}
        if len(rows) == 1 or len(cols) == 1:
            if rng.random() < 0.7:
                continue
        return patch
    return None


def _apply_patch(panel: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        panel[r, c] = color


def _maybe_add_extra_region(rng: random.Random, panels: list[np.ndarray]) -> None:
    choice = rng.random()
    if choice < 0.25:
        return
    layer = 1 if choice < 0.7 else 2
    panel = panels[layer]
    upper_mask = np.zeros((PANEL_H, PANEL_W), dtype=bool)
    for upper in panels[layer + 1 :]:
        upper_mask |= upper != BLACK
    composite, _ = _composite_with_sources(panels)
    blocked = upper_mask | _dilate4(composite != BLACK) | _dilate4(panel != BLACK)
    allowed = ~blocked
    patch = _random_connected_patch(rng, allowed, rng.randint(3, 5))
    if patch is not None:
        _apply_patch(panel, patch, LAYER_COLORS[layer])


def _place_singleton(rng: random.Random, panels: list[np.ndarray]) -> bool:
    composite, _ = _composite_with_sources(panels)
    blocked = _dilate4(composite != BLACK) | _dilate4(panels[2] != BLACK)
    candidates = [(r, c) for r in range(PANEL_H) for c in range(PANEL_W) if not blocked[r, c]]
    if not candidates:
        return False
    r, c = rng.choice(candidates)
    panels[2][r, c] = BLUE
    return True


def _flip_panels(rng: random.Random, panels: list[np.ndarray]) -> list[np.ndarray]:
    out = [panel.copy() for panel in panels]
    if rng.random() < 0.5:
        out = [np.fliplr(panel).copy() for panel in out]
    if rng.random() < 0.5:
        out = [np.flipud(panel).copy() for panel in out]
    return out


def _panel_component_sizes(panel: np.ndarray) -> list[int]:
    return sorted(len(comp) for comp in _components(panel != BLACK))


def _has_split_due_to_occlusion(panels: list[np.ndarray]) -> bool:
    for layer in range(2):
        upper_mask = np.zeros((PANEL_H, PANEL_W), dtype=bool)
        for upper in panels[layer + 1 :]:
            upper_mask |= upper != BLACK
        for comp in _components(panels[layer] != BLACK):
            visible = np.zeros((PANEL_H, PANEL_W), dtype=bool)
            for r, c in comp:
                if not upper_mask[r, c]:
                    visible[r, c] = True
            if np.count_nonzero(visible) == 0:
                continue
            if len(_components(visible)) > 1:
                return True
    return False


def _scene_constraints_hold(panels: list[np.ndarray]) -> bool:
    sizes = [_panel_component_sizes(panel) for panel in panels]
    if not all(1 <= len(s) <= 3 for s in sizes):
        return False
    if any(size < 3 or size > 10 for size in sizes[0] + sizes[1]):
        return False

    blue_sizes = sizes[2]
    if blue_sizes.count(1) != 1:
        return False
    if any(size > 10 for size in blue_sizes):
        return False
    if any(size < 3 for size in blue_sizes if size != 1):
        return False

    composite, sources = _composite_with_sources(panels)
    comps = _components(composite != BLACK)
    if sum(1 for comp in comps if len(comp) == 1) != 1:
        return False

    if not any(len({int(sources[r, c]) for r, c in comp}) >= 2 for comp in comps if len(comp) > 1):
        return False

    if not _has_split_due_to_occlusion(panels):
        return False

    out = main(_build_input_grid(panels))
    if out.shape != (PANEL_H, PANEL_W):
        return False
    if np.array_equal(out, composite):
        return False
    if np.count_nonzero(out) == 0:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    left = np.zeros((PANEL_H, PANEL_W), dtype=int)
    middle = np.zeros((PANEL_H, PANEL_W), dtype=int)
    right = np.zeros((PANEL_H, PANEL_W), dtype=int)
    panels = [left, middle, right]

    connector_below = rng.random() < 0.5
    direction = 1 if connector_below else -1
    r0 = rng.randint(1, 4)
    c0 = rng.randint(0, 2)
    bridge_r = r0 + rng.randint(0, 1)
    bridge = (bridge_r, c0 + 2)

    yellow_main = {
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0 + 1, c0 + 1),
        (bridge_r, c0 + 2),
        (r0, c0 + 3),
        (r0, c0 + 4),
        (r0 + 1, c0 + 3),
        (r0 + 1, c0 + 4),
    }
    _apply_patch(left, yellow_main, YELLOW)

    connector_r = r0 + 2 if connector_below else r0 - 1
    connector_cols = [c0 + 1, c0 + 2, c0 + 3]
    if c0 > 0 and rng.random() < 0.5:
        connector_cols.insert(0, c0)
    if c0 + 4 < PANEL_W and rng.random() < 0.5:
        connector_cols.append(c0 + 4)
    orange_main = {(connector_r, c) for c in connector_cols}
    kink_candidates = [
        (connector_r + direction, connector_cols[0]),
        (connector_r + direction, connector_cols[-1]),
        (connector_r - direction, connector_cols[len(connector_cols) // 2]),
    ]
    rng.shuffle(kink_candidates)
    for cell in kink_candidates:
        rr, cc = cell
        if 0 <= rr < PANEL_H and 0 <= cc < PANEL_W and rng.random() < 0.45:
            orange_main.add(cell)
            break
    if not (3 <= len(orange_main) <= 10):
        return None
    _apply_patch(middle, orange_main, ORANGE)

    blue_shapes = _candidate_blue_shapes(bridge, direction)
    if not blue_shapes:
        return None
    blue_main = rng.choice(blue_shapes)
    if not (3 <= len(blue_main) <= 10):
        return None
    _apply_patch(right, blue_main, BLUE)

    _maybe_add_extra_region(rng, panels)
    if not _place_singleton(rng, panels):
        return None

    panels = _flip_panels(rng, panels)
    if not _scene_constraints_hold(panels):
        return None
    return _build_input_grid(panels)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
