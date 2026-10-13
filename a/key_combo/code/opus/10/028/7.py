from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIVIDER = 4
PALETTE = (1, 2, 3)
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mono_components(grid: np.ndarray, colors: set[int] | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            if colors is not None and color not in colors:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "touch_right": bool(np.any(arr[:, 1] == w - 1)),
                    "touch_bottom": bool(np.any(arr[:, 0] == h - 1)),
                }
            )
    return comps


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _extract_cues(panel: np.ndarray, *, allow_bottom: bool) -> tuple[dict[int, int], np.ndarray]:
    stripped = panel.copy()
    cues: dict[int, int] = {}
    for comp in _mono_components(panel):
        is_cue = comp["area"] <= 3 and (comp["touch_right"] or (allow_bottom and comp["touch_bottom"]))
        if not is_cue:
            continue
        color = int(comp["color"])
        assert color not in cues, f"duplicate cue color {color}"
        cues[color] = int(comp["area"])
        rr, cc = comp["coords"].T
        stripped[rr, cc] = BG
    return cues, stripped


def _resolve_panel_from_masks(
    layer_masks: dict[int, np.ndarray],
    layer_areas: dict[int, int],
    angles: dict[int, int],
) -> np.ndarray:
    colors = sorted(layer_masks)
    shape = next(iter(layer_masks.values())).shape
    out = np.zeros(shape, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            contenders = [color for color in colors if layer_masks[color][r, c]]
            if not contenders:
                continue
            scored = sorted(
                contenders,
                key=lambda color: (int(angles.get(color, 0)) % 4, -int(layer_areas[color])),
                reverse=True,
            )
            if len(scored) >= 2:
                a0 = (int(angles.get(scored[0], 0)) % 4, -int(layer_areas[scored[0]]))
                a1 = (int(angles.get(scored[1], 0)) % 4, -int(layer_areas[scored[1]]))
                assert a0 != a1, "ambiguous same-panel collision"
            out[r, c] = scored[0]
    return out


def _rotate_panel_layers(panel: np.ndarray, angles: dict[int, int]) -> np.ndarray:
    colors = sorted(int(c) for c in np.unique(panel) if int(c) != BG)
    layer_masks = {
        color: np.rot90(panel == color, -(int(angles.get(color, 0)) % 4)) for color in colors
    }
    layer_areas = {color: int(np.count_nonzero(panel == color)) for color in colors}
    return _resolve_panel_from_masks(layer_masks, layer_areas, angles)


def _overlay_panels(top: np.ndarray, bottom: np.ndarray, angles: dict[int, int]) -> np.ndarray:
    assert top.shape == bottom.shape
    out = np.zeros_like(top)
    h, w = top.shape
    for r in range(h):
        for c in range(w):
            a = int(top[r, c])
            b = int(bottom[r, c])
            if a == BG:
                out[r, c] = b
            elif b == BG:
                out[r, c] = a
            else:
                aa = int(angles.get(a, 0)) % 4
                bb = int(angles.get(b, 0)) % 4
                out[r, c] = a if aa >= bb else b
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each panel's color layers from edge cues, then overlay the two panels after removing the divider.

    Input structure:
    - The grid has black background and exactly one full-width yellow divider row.
    - The divider splits two equal-size square panels using the same 2-3 foreground colors.
    - Each foreground color forms one board layer per panel, while tiny same-color cue sprites sit on the right edge or bottom edge outside the board mass.

    Rule:
    - Read each cue sprite as a monochrome 4-connected component; its area P gives a clockwise rotation of (P mod 4) quarter-turns for that color.
    - Remove cue sprites, then rotate every color layer inside each panel in the panel's own coordinate system.
    - If rotated layers collide within one panel, keep the color with the larger rotation; equal rotations keep the color whose panel layer had fewer pixels.
    - Delete the yellow divider row and transparently overlay the rotated top and bottom panels; cross-panel conflicts keep the larger-rotation color, and equal rotations keep the top-panel color.

    Key insight:
    - Cue sprite size is both the motion code and the conflict priority, so the solver must decode cues before either panel rotation or panel overlay can be resolved correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    divider_rows = [r for r in range(h) if np.all(g[r, :] == DIVIDER)]
    assert len(divider_rows) == 1, "expected exactly one full-width yellow divider"
    div = divider_rows[0]

    top = g[:div, :]
    bottom = g[div + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "divider must split equal panels"
    assert abs(top.shape[0] - top.shape[1]) <= 1, "panels must be square or near-square"
    assert top.shape[0] == top.shape[1], "90-degree layer rotation is generated only on square panels"

    top_cues, top_board = _extract_cues(top, allow_bottom=False)
    bottom_cues, bottom_board = _extract_cues(bottom, allow_bottom=True)
    cue_sizes = dict(top_cues)
    for color, size in bottom_cues.items():
        if color in cue_sizes:
            assert cue_sizes[color] == size, f"conflicting cue sizes for color {color}"
        cue_sizes[color] = size
    angles = {color: size % 4 for color, size in cue_sizes.items()}

    top_rot = _rotate_panel_layers(top_board, angles)
    bottom_rot = _rotate_panel_layers(bottom_board, angles)
    return _overlay_panels(top_rot, bottom_rot, angles)


def _grow_connected_mask(n: int, area: int) -> np.ndarray:
    usable = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
    center = (n - 1) / 2.0
    weights = [1.0 / (1.0 + abs(r - center) + abs(c - center)) for r, c in usable]
    seed = random.choices(usable, weights=weights, k=1)[0]
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in list(cells):
            for nr, nc in _neighbors4(r, c):
                if 1 <= nr < n - 1 and 1 <= nc < n - 1 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier
        frontier.sort(key=lambda rc: abs(rc[0] - center) + abs(rc[1] - center))
        pick = frontier[random.randrange(min(len(frontier), 6))]
        cells.add(pick)
    mask = np.zeros((n, n), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mask_with_variation(base: np.ndarray) -> np.ndarray:
    mask = base.copy()
    if random.random() < 0.55:
        on = list(map(tuple, np.argwhere(mask)))
        off = []
        for r, c in on:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 1 <= nr < mask.shape[0] - 1 and 1 <= nc < mask.shape[1] - 1 and not mask[nr, nc]:
                    off.append((nr, nc))
        if off:
            nr, nc = random.choice(off)
            mask[nr, nc] = True
            rr, cc = random.choice(on)
            if mask.sum() > 4:
                trial = mask.copy()
                trial[rr, cc] = False
                if _component_count(trial) == 1:
                    mask = trial
    return mask


def _random_rotated_layers(n: int, colors: list[int]) -> dict[int, np.ndarray]:
    base = _grow_connected_mask(n, random.randint(4, min(6, (n - 2) * (n - 2) - 1)))
    layers = {colors[0]: base}
    for color in colors[1:]:
        if random.random() < 0.65:
            shifted = np.roll(base, shift=(random.choice([-1, 0, 1]), random.choice([-1, 0, 1])), axis=(0, 1))
            shifted[[0, -1], :] = False
            shifted[:, [0, -1]] = False
            if shifted.sum() >= 4 and _component_count(shifted) == 1:
                base = shifted
        layers[color] = _mask_with_variation(base if random.random() < 0.5 else _grow_connected_mask(n, random.randint(4, 6)))
    return layers


def _panel_from_masks(masks: dict[int, np.ndarray]) -> np.ndarray:
    panel = np.zeros(next(iter(masks.values())).shape, dtype=int)
    for color, mask in masks.items():
        panel[mask] = color
    return panel


def _inverse_rotate_masks(rotated: dict[int, np.ndarray], angles: dict[int, int]) -> dict[int, np.ndarray]:
    return {
        color: np.rot90(mask, int(angles[color]) % 4)
        for color, mask in rotated.items()
    }


def _nonoverlap_panel(masks: dict[int, np.ndarray]) -> bool:
    acc = np.zeros(next(iter(masks.values())).shape, dtype=int)
    for mask in masks.values():
        acc += mask.astype(int)
    return bool(np.max(acc) <= 1)


def _same_panel_conflicts(resolved_source: dict[int, np.ndarray]) -> int:
    acc = np.zeros(next(iter(resolved_source.values())).shape, dtype=int)
    for mask in resolved_source.values():
        acc += mask.astype(int)
    return int(np.count_nonzero(acc > 1))


def _board_layer_valid(mask: np.ndarray) -> bool:
    if int(mask.sum()) < 4:
        return False
    if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        return False
    return _component_count(mask) == 1


def _adjacent_to_mask(coords: set[tuple[int, int]], mask: np.ndarray) -> bool:
    for r, c in coords:
        if mask[r, c]:
            return True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                return True
    return False


def _candidate_cue_shapes(n: int, size: int, *, allow_bottom: bool) -> list[set[tuple[int, int]]]:
    out: list[set[tuple[int, int]]] = []
    if size == 1:
        for r in range(n):
            out.append({(r, n - 1)})
        if allow_bottom:
            for c in range(n):
                out.append({(n - 1, c)})
    elif size == 2:
        for r in range(n - 1):
            out.append({(r, n - 1), (r + 1, n - 1)})
        for c in range(n - 1):
            out.append({(c, n - 2), (c, n - 1)})
        if allow_bottom:
            for c in range(n - 1):
                out.append({(n - 1, c), (n - 1, c + 1)})
            for r in range(n - 1):
                out.append({(n - 2, r), (n - 1, r)})
    else:
        for r in range(n - 2):
            out.append({(r, n - 1), (r + 1, n - 1), (r + 2, n - 1)})
        for r in range(n - 1):
            out.append({(r, n - 1), (r + 1, n - 1), (r + 1, n - 2)})
        for r in range(n - 1):
            out.append({(r, n - 1), (r + 1, n - 1), (r, n - 2)})
        if allow_bottom:
            for c in range(n - 2):
                out.append({(n - 1, c), (n - 1, c + 1), (n - 1, c + 2)})
            for c in range(n - 1):
                out.append({(n - 1, c), (n - 1, c + 1), (n - 2, c)})
            for c in range(n - 1):
                out.append({(n - 1, c), (n - 1, c + 1), (n - 2, c + 1)})
    uniq = []
    seen = set()
    for shape in out:
        key = tuple(sorted(shape))
        if key not in seen:
            seen.add(key)
            uniq.append(shape)
    random.shuffle(uniq)
    return uniq


def _place_cues(
    n: int,
    cue_sizes: dict[int, int],
    top_board: np.ndarray,
    bottom_board: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    top = np.zeros((n, n), dtype=int)
    bottom = np.zeros((n, n), dtype=int)
    top_occ: set[tuple[int, int]] = set()
    bottom_occ: set[tuple[int, int]] = set()
    plan = []
    for color in cue_sizes:
        mode = random.choice(["top_right", "bottom_right", "bottom_bottom"])
        if mode == "top_right":
            plan.append((color, size := cue_sizes[color], top, top_occ, top_board, False))
        elif mode == "bottom_right":
            plan.append((color, size := cue_sizes[color], bottom, bottom_occ, bottom_board, False))
        else:
            plan.append((color, size := cue_sizes[color], bottom, bottom_occ, bottom_board, True))

    for color, size, panel, occ, board_mask, allow_bottom in plan:
        placed = False
        for coords in _candidate_cue_shapes(n, size, allow_bottom=allow_bottom):
            if coords & occ:
                continue
            if any((r, c) in top_occ and panel is bottom for r, c in coords):
                continue
            if _adjacent_to_mask(coords, board_mask):
                continue
            if panel is top and any(bottom_board[r, c] or bottom[r, c] for r, c in coords):
                continue
            if panel is bottom and any(top_board[r, c] or top[r, c] for r, c in coords):
                continue
            for r, c in coords:
                panel[r, c] = color
            occ |= coords
            placed = True
            break
        if not placed:
            return None
    return top, bottom


def _build_input_from_case(
    n: int,
    colors: list[int],
    cue_sizes: dict[int, int],
    rotated_top: dict[int, np.ndarray],
    rotated_bottom: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    angles = {color: cue_sizes[color] % 4 for color in colors}
    input_top_masks = _inverse_rotate_masks(rotated_top, angles)
    input_bottom_masks = _inverse_rotate_masks(rotated_bottom, angles)
    if not _nonoverlap_panel(input_top_masks) or not _nonoverlap_panel(input_bottom_masks):
        return None
    if not all(_board_layer_valid(mask) for mask in list(input_top_masks.values()) + list(input_bottom_masks.values())):
        return None

    top_board = _panel_from_masks(input_top_masks)
    bottom_board = _panel_from_masks(input_bottom_masks)
    if np.any((top_board != BG) & (bottom_board != BG)):
        return None

    cues = _place_cues(n, cue_sizes, top_board != BG, bottom_board != BG)
    if cues is None:
        return None
    top_cues, bottom_cues = cues
    if np.any((top_cues != BG) & (bottom_cues != BG)):
        return None

    top_input = top_board.copy()
    bottom_input = bottom_board.copy()
    top_input[top_cues != BG] = top_cues[top_cues != BG]
    bottom_input[bottom_cues != BG] = bottom_cues[bottom_cues != BG]
    if np.any((top_input != BG) & (bottom_input != BG)):
        return None

    expected_top = _resolve_panel_from_masks(rotated_top, {c: int(rotated_top[c].sum()) for c in colors}, angles)
    expected_bottom = _resolve_panel_from_masks(rotated_bottom, {c: int(rotated_bottom[c].sum()) for c in colors}, angles)
    expected = _overlay_panels(expected_top, expected_bottom, angles)
    cross_conflicts = int(np.count_nonzero((expected_top != BG) & (expected_bottom != BG)))
    inner_conflicts = _same_panel_conflicts(rotated_top) + _same_panel_conflicts(rotated_bottom)

    inp = np.zeros((2 * n + 1, n), dtype=int)
    inp[:n, :] = top_input
    inp[n, :] = DIVIDER
    inp[n + 1 :, :] = bottom_input
    return inp, expected, expected_top, cross_conflicts, inner_conflicts


def _grow_mask_in_box(n: int, area: int, box: tuple[int, int, int, int]) -> np.ndarray | None:
    r0, r1, c0, c1 = box
    cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
    if len(cells) < area:
        return None
    for _ in range(60):
        seed = random.choice(cells)
        chosen = {seed}
        while len(chosen) < area:
            frontier = []
            for r, c in list(chosen):
                for nr, nc in _neighbors4(r, c):
                    if r0 <= nr < r1 and c0 <= nc < c1 and (nr, nc) not in chosen:
                        frontier.append((nr, nc))
            if not frontier:
                break
            chosen.add(random.choice(frontier))
        if len(chosen) != area:
            continue
        mask = np.zeros((n, n), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if _board_layer_valid(mask):
            return mask
    return None


def _sample_input_masks(n: int, colors: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    mid = n // 2
    base_boxes = [
        (1, mid + 1, 1, mid + 1),
        (1, mid + 1, mid - 1, n - 1),
        (mid - 1, n - 1, 1, mid + 1),
        (mid - 1, n - 1, mid - 1, n - 1),
    ]
    for _ in range(40):
        boxes = base_boxes[:]
        random.shuffle(boxes)
        top: dict[int, np.ndarray] = {}
        bottom: dict[int, np.ndarray] = {}
        for i, color in enumerate(colors):
            top_mask = _grow_mask_in_box(n, random.randint(4, 5), boxes[i % 4])
            bottom_mask = _grow_mask_in_box(n, random.randint(4, 5), boxes[(i + 1) % 4])
            if top_mask is None or bottom_mask is None:
                top = {}
                break
            top[color] = top_mask
            bottom[color] = bottom_mask
        if len(top) != len(colors):
            continue
        if not _nonoverlap_panel(top) or not _nonoverlap_panel(bottom):
            continue
        top_panel = _panel_from_masks(top)
        bottom_panel = _panel_from_masks(bottom)
        if np.any((top_panel != BG) & (bottom_panel != BG)):
            continue
        return top, bottom
    return None


def _build_from_input_masks(
    n: int,
    colors: list[int],
    cue_sizes: dict[int, int],
    top_input_masks: dict[int, np.ndarray],
    bottom_input_masks: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    angles = {color: cue_sizes[color] % 4 for color in colors}
    top_board = _panel_from_masks(top_input_masks)
    bottom_board = _panel_from_masks(bottom_input_masks)
    cues = _place_cues(n, cue_sizes, top_board != BG, bottom_board != BG)
    if cues is None:
        return None
    top_cues, bottom_cues = cues

    top_input = top_board.copy()
    bottom_input = bottom_board.copy()
    top_input[top_cues != BG] = top_cues[top_cues != BG]
    bottom_input[bottom_cues != BG] = bottom_cues[bottom_cues != BG]
    if np.any((top_input != BG) & (bottom_input != BG)):
        return None

    rotated_top = {color: np.rot90(mask, -angles[color]) for color, mask in top_input_masks.items()}
    rotated_bottom = {color: np.rot90(mask, -angles[color]) for color, mask in bottom_input_masks.items()}
    top_areas = {color: int(mask.sum()) for color, mask in top_input_masks.items()}
    bottom_areas = {color: int(mask.sum()) for color, mask in bottom_input_masks.items()}
    expected_top = _resolve_panel_from_masks(rotated_top, top_areas, angles)
    expected_bottom = _resolve_panel_from_masks(rotated_bottom, bottom_areas, angles)
    expected = _overlay_panels(expected_top, expected_bottom, angles)

    cross_conflicts = int(np.count_nonzero((expected_top != BG) & (expected_bottom != BG)))
    inner_conflicts = _same_panel_conflicts(rotated_top) + _same_panel_conflicts(rotated_bottom)
    inp = np.zeros((2 * n + 1, n), dtype=int)
    inp[:n, :] = top_input
    inp[n, :] = DIVIDER
    inp[n + 1 :, :] = bottom_input
    return inp, expected, cross_conflicts, inner_conflicts


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n = random.randint(7, 9)
    k = 3 if random.random() < 0.2 else 2
    colors = random.sample(list(PALETTE), k)
    sizes = random.sample([1, 2, 3], k)
    cue_sizes = {color: size for color, size in zip(colors, sizes)}
    want_inner_conflict = random.random() < 0.05

    for _ in range(80):
        sampled = _sample_input_masks(n, colors)
        if sampled is None:
            continue
        top_masks, bottom_masks = sampled
        built = _build_from_input_masks(n, colors, cue_sizes, top_masks, bottom_masks)
        if built is None:
            continue
        inp, expected, cross_conflicts, inner_conflicts = built
        if cross_conflicts < 2 or cross_conflicts > max(8, n):
            continue
        if want_inner_conflict and inner_conflicts == 0:
            continue
        top_cues, top_board = _extract_cues(inp[:n, :], allow_bottom=False)
        bottom_cues, bottom_board = _extract_cues(inp[n + 1 :, :], allow_bottom=True)
        if set(top_cues) | set(bottom_cues) != set(colors):
            continue
        if len(_mono_components(top_board, set(colors))) != len(colors):
            continue
        if len(_mono_components(bottom_board, set(colors))) != len(colors):
            continue
        assert np.all(inp[n, :] == DIVIDER)
        assert len(set(cue_sizes.values())) == len(cue_sizes)
        assert np.array_equal(main(inp), expected)
        assert expected.shape == (n, n)
        assert not np.array_equal(inp[:n, :], expected)
        return inp, expected

    raise AssertionError("failed to build a valid case")


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, _expected = _generate_case()
        except AssertionError:
            continue
        return inp
    raise AssertionError("failed to generate a valid input")
