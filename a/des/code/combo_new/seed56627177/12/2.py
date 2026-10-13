import random
from collections import deque

import numpy as np


BG = 0
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 400

CORE_TEMPLATES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
)

ROW_PATTERNS = {
    1: ((1,),),
    2: ((2,), (1, 1)),
    3: ((3,), (2, 1), (1, 2), (1, 1, 1)),
}


def _components(mask: np.ndarray) -> list[dict[str, np.ndarray | tuple[int, int, int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append({"coords": coords, "bbox": (int(r0), int(c0), int(r1), int(c1))})
    return comps


def _crop_nonzero(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    exposed = (
        ~padded[:-2, 1:-1]
        | ~padded[2:, 1:-1]
        | ~padded[1:-1, :-2]
        | ~padded[1:-1, 2:]
    )
    return mask & exposed


def _dilate4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _extract_layers(sprite: np.ndarray) -> list[np.ndarray]:
    remaining = sprite != BG
    layers: list[np.ndarray] = []
    while remaining.any():
        shell = _boundary(remaining)
        layer = np.where(shell, sprite, BG)
        layers.append(_crop_nonzero(layer))
        remaining &= ~shell
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel each nested multicolor sprite into successive outer contours and pack those contour snapshots from left to right.

    Input structure:
    - Background is black.
    - The non-black pixels form several disconnected 4-connected sprites.
    - Each sprite is a concentric multicolor object: removing the current outer contour reveals another fully enclosed layer inside it.

    Rule:
    - Find each sprite and sort the sprites by their appearance order, top to bottom and then left to right.
    - For every sprite, repeatedly remove the currently exposed 4-neighbor outer contour; each peeled contour becomes one output patch with only that contour's original color and black elsewhere.
    - Pack the peeled patches for one sprite from outermost to innermost, using one black separator column between consecutive layers and two black separator columns between consecutive sprites.

    Key insight:
    - The same object must be interpreted as a stack of erosion depths rather than as one solid sprite: every newly exposed contour is the next item in the left-to-right output sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _components(g != BG)
    assert components, "expected at least one sprite"
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    sprite_layers: list[list[np.ndarray]] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        sprite = g[r0 : r1 + 1, c0 : c1 + 1]
        layers = _extract_layers(sprite)
        sprite_layers.append(layers)

    out_h = max(layer.shape[0] for layers in sprite_layers for layer in layers)
    out_w = sum(
        sum(layer.shape[1] for layer in layers) + max(0, len(layers) - 1)
        for layers in sprite_layers
    ) + max(0, 2 * (len(sprite_layers) - 1))
    out = np.full((out_h, out_w), BG, dtype=int)

    cursor = 0
    for sprite_index, layers in enumerate(sprite_layers):
        for layer_index, layer in enumerate(layers):
            h, w = layer.shape
            out[:h, cursor : cursor + w] = layer
            cursor += w
            if layer_index + 1 < len(layers):
                cursor += 1
        if sprite_index + 1 < len(sprite_layers):
            cursor += 2

    assert cursor == out_w
    return out


def _random_core() -> np.ndarray:
    mask = random.choice(CORE_TEMPLATES).copy()
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return _crop_nonzero(mask.astype(int)).astype(bool)


def _build_sprite(n_layers: int) -> np.ndarray:
    core = _random_core()
    pad = n_layers - 1
    h, w = core.shape
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    canvas[pad : pad + h, pad : pad + w] = core

    masks = [canvas.copy()]
    for _ in range(n_layers - 1):
        masks.append(_dilate4(masks[-1]))

    colors = random.sample(range(1, 10), n_layers)
    inner_to_outer = list(reversed(colors))

    sprite = np.full_like(canvas, BG, dtype=int)
    prev = np.zeros_like(canvas, dtype=bool)
    for color, mask in zip(inner_to_outer, masks):
        band = mask & ~prev
        sprite[band] = color
        prev = mask

    return _crop_nonzero(sprite)


def _sample_sprites() -> list[np.ndarray]:
    n_sprites = random.choices((1, 2, 3), weights=(3, 4, 3), k=1)[0]
    sprites = []
    for _ in range(n_sprites):
        n_layers = random.choice((2, 3, 4))
        sprites.append(_build_sprite(n_layers))
    return sprites


def _place_sprites(sprites: list[np.ndarray]) -> np.ndarray | None:
    row_pattern = random.choice(ROW_PATTERNS[len(sprites)])
    rows: list[list[np.ndarray]] = []
    idx = 0
    for count in row_pattern:
        rows.append(sprites[idx : idx + count])
        idx += count

    col_gaps = [random.randint(2, 4) for _ in rows]
    row_gaps = [random.randint(2, 4) for _ in range(max(0, len(rows) - 1))]
    row_heights = [max(sprite.shape[0] for sprite in row) for row in rows]
    row_widths = [
        sum(sprite.shape[1] for sprite in row) + gap * max(0, len(row) - 1)
        for row, gap in zip(rows, col_gaps)
    ]

    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)

    grid_h = top_margin + sum(row_heights) + sum(row_gaps) + bottom_margin
    grid_w = left_margin + max(row_widths) + right_margin
    if grid_h > ARC_MAX or grid_w > ARC_MAX:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    row_top = top_margin
    max_row_width = max(row_widths)
    for row_index, (row, row_h, row_w, gap) in enumerate(zip(rows, row_heights, row_widths, col_gaps)):
        slack = max_row_width - row_w
        col = left_margin + random.randint(0, slack)
        for sprite in row:
            h, w = sprite.shape
            grid[row_top : row_top + h, col : col + w] = sprite
            col += w + gap
        row_top += row_h
        if row_index + 1 < len(rows):
            row_top += row_gaps[row_index]
    return grid


def _valid_generated_case(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > ARC_MAX:
        return False
    comps = _components(grid != BG)
    if not (1 <= len(comps) <= 3):
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.ndim != 2 or max(out.shape) > ARC_MAX:
        return False
    return len(np.unique(grid)) >= 3


def _gen_attempt() -> np.ndarray | None:
    sprites = _sample_sprites()
    random.shuffle(sprites)
    grid = _place_sprites(sprites)
    if grid is None:
        return None
    if not _valid_generated_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
