import random
from typing import Any

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500
COMMON_PATTERNS = {
    3: (
        (0, 2, 1),
        (2, 0, 1),
    ),
    4: (
        (0, 1, 2, 3),
        (2, 3, 0, 1),
        (0, 1, 2, 1),
        (2, 1, 0, 1),
    ),
}
RARE_PATTERNS = {
    3: (
        (2, 1, 1),
        (1, 2, 1),
    ),
    4: (
        (2, 1, 1, 1),
        (1, 2, 1, 1),
        (1, 1, 2, 1),
    ),
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert len(sep_cols) in (2, 3), "expected 3 or 4 panels"
    assert all(sep_cols[i + 1] - sep_cols[i] > 1 for i in range(len(sep_cols) - 1)), "separator columns must be isolated"

    cuts = [-1, *sep_cols, g.shape[1]]
    panels = [g[:, cuts[i] + 1 : cuts[i + 1]] for i in range(len(cuts) - 1)]
    widths = {panel.shape[1] for panel in panels}
    assert len(widths) == 1 and next(iter(widths)) > 0, "panels must be equal width"
    assert 3 <= len(panels) <= 4
    for panel in panels:
        assert not np.any(panel == SEP), "separator color must only appear in full divider columns"
    return panels


def _extract_sprite(panel: np.ndarray) -> dict[str, Any]:
    fg = (panel != BG)
    box = _bbox(fg)
    assert box is not None, "each panel must contain one sprite"
    colors = np.unique(panel[fg])
    assert len(colors) == 1, "each panel sprite must be monochromatic"
    color = int(colors[0])
    sprite = _crop(panel, box)
    mask = sprite != BG
    return {
        "color": color,
        "mask": mask,
        "height": mask.shape[0],
        "width": mask.shape[1],
    }


def _find_alignment(mask: np.ndarray, target: np.ndarray) -> tuple[int, np.ndarray]:
    matches: list[tuple[int, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(mask, k=k)
        if rotated.shape == target.shape and np.array_equal(rotated, target):
            matches.append((k, rotated))
    assert len(matches) == 1, "sprite rotations must map uniquely to the reference orientation"
    return matches[0]


def _analyze(grid: np.ndarray) -> dict[str, Any]:
    panels = _split_panels(grid)
    sprites = [_extract_sprite(panel) for panel in panels]

    heights = [sprite["height"] for sprite in sprites]
    ref_height = max(heights)
    ref_index = next(i for i, height in enumerate(heights) if height == ref_height)
    ref_mask = sprites[ref_index]["mask"]

    aligned_masks: list[np.ndarray] = []
    align_steps: list[int] = []
    for sprite in sprites:
        k, aligned = _find_alignment(sprite["mask"], ref_mask)
        aligned_masks.append(aligned)
        align_steps.append(k)

    order = sorted(range(len(sprites)), key=lambda i: (sprites[i]["height"], i))
    return {
        "sprites": sprites,
        "aligned_masks": aligned_masks,
        "align_steps": align_steps,
        "ref_index": ref_index,
        "ref_mask": ref_mask,
        "order": order,
    }


def _overlay(aligned_masks: list[np.ndarray], colors: list[int], order: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx in order:
        out[aligned_masks[idx]] = int(colors[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate same-template panel sprites into a reference orientation, then overlay them by original bbox height.

    Input structure:
    - A black background grid is split into 3 or 4 equal-width panels by full-height grey separator columns.
    - Each panel contains exactly one monochromatic sprite in a distinct color.
    - The sprites are 90-degree rotation variants of the same non-symmetric template.

    Rule:
    - Measure each panel sprite's actual bounding-box height and choose the leftmost tallest sprite as the reference orientation.
    - Rotate every sprite so its cropped shape matches that reference orientation while keeping its original color.
    - Sort sprites by their original bounding-box height from short to tall, breaking ties by panel position from left to right.
    - Overlay the aligned sprites on one blank canvas with top-left corners all at (0,0), letting later non-background pixels overwrite earlier ones.

    Key insight:
    - Bounding-box height carries two meanings at once: it identifies the reference orientation and also fixes the overwrite order for the aligned stack.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    colors = [sprite["color"] for sprite in info["sprites"]]
    return _overlay(info["aligned_masks"], colors, info["order"], info["ref_mask"].shape)


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            frontier.update((nr, nc) for nr, nc in _neighbors4(int(r), int(c), h, w) if not mask[nr, nc])
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _make_template() -> np.ndarray:
    for _ in range(300):
        h = random.choice((4, 5))
        w = random.choice((3, 4))
        if h <= w:
            continue
        min_area = max(h + w - 1, int(np.ceil(0.55 * h * w)))
        max_area = h * w - 2
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(h, w, area)
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        rots = {_signature(np.rot90(mask, k)) for k in range(4)}
        if len(rots) != 4:
            continue
        if len(set(mask.sum(axis=1).tolist())) == 1 and len(set(mask.sum(axis=0).tolist())) == 1:
            continue
        return mask
    raise RuntimeError("failed to construct a non-symmetric template")


def _choose_orientations(n_panels: int) -> tuple[list[int], bool]:
    use_rare = random.random() < 0.25
    pool = RARE_PATTERNS[n_panels] if use_rare else COMMON_PATTERNS[n_panels]
    return list(random.choice(pool)), use_rare


def _render_input(template: np.ndarray, orientations: list[int], colors: list[int], panel_h: int, panel_w: int) -> np.ndarray:
    n_panels = len(orientations)
    width = n_panels * panel_w + (n_panels - 1)
    grid = np.full((panel_h, width), BG, dtype=int)
    for idx in range(n_panels - 1):
        grid[:, (idx + 1) * panel_w + idx] = SEP

    for idx, (orientation, color) in enumerate(zip(orientations, colors)):
        sprite_mask = np.rot90(template, k=orientation)
        sprite = np.where(sprite_mask, color, BG)
        panel = np.full((panel_h, panel_w), BG, dtype=int)
        r0 = (panel_h - sprite.shape[0]) // 2
        c0 = (panel_w - sprite.shape[1]) // 2
        panel[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]] = sprite

        c0_grid = idx * (panel_w + 1)
        grid[:, c0_grid : c0_grid + panel_w] = panel
    return grid


def _semantic_checks(grid: np.ndarray, expect_winner_rotated: bool) -> bool:
    try:
        info = _analyze(grid)
        expected = main(grid)
    except AssertionError:
        return False

    if max(grid.shape) > 30 or grid.ndim != 2:
        return False

    sprites = info["sprites"]
    heights = [sprite["height"] for sprite in sprites]
    if len(set(heights)) < 2:
        return False
    if len({sprite["color"] for sprite in sprites}) != len(sprites):
        return False
    if not any(step != 0 for step in info["align_steps"]):
        return False

    winner = info["order"][-1]
    if expect_winner_rotated:
        if info["align_steps"][winner] == 0:
            return False
    else:
        if winner != info["ref_index"]:
            return False

    left_to_right = _overlay(info["aligned_masks"], [sprite["color"] for sprite in sprites], list(range(len(sprites))), info["ref_mask"].shape)
    if np.array_equal(left_to_right, expected):
        return False

    raw_winner = np.where(sprites[winner]["mask"], sprites[winner]["color"], BG)
    if expect_winner_rotated and raw_winner.shape == expected.shape and np.array_equal(raw_winner, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_panels = random.choice((3, 4))
    template = _make_template()
    orientations, rare_case = _choose_orientations(n_panels)
    colors = random.sample(PALETTE, n_panels)

    max_dim = max(template.shape)
    max_panel_w = (30 - (n_panels - 1)) // n_panels
    min_panel_w = max_dim + 1
    if min_panel_w > max_panel_w:
        return None
    panel_w = random.randint(min_panel_w, max_panel_w)
    panel_h = random.randint(max_dim + 2, min(12, 30))

    grid = _render_input(template, orientations, colors, panel_h, panel_w)
    expect_winner_rotated = not rare_case
    if not _semantic_checks(grid, expect_winner_rotated=expect_winner_rotated):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-overlay ARC input")
