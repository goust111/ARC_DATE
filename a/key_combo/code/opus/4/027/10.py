from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_path = str(_ARC_UTILS_DIR)
    if arc_utils_path not in sys.path:
        sys.path.append(arc_utils_path)

try:
    from arc_utils import Color, bbox_h, bbox_w, colors_present, crop_bbox, find_components, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        ORANGE = 7
        TEAL = 8

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        colors = set(map(int, grid.reshape(-1)))
        if exclude is not None:
            colors -= set(map(int, exclude))
        return sorted(colors)

    def bbox_h(box: tuple[int, int, int, int]) -> int:
        return int(box[2] - box[0] + 1)

    def bbox_w(box: tuple[int, int, int, int]) -> int:
        return int(box[3] - box[1] + 1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = False,
        return_format: str = "dict",
        **_kwargs,
    ) -> list[dict]:
        if monochromatic:
            raise NotImplementedError("fallback only supports multicolor components")
        if return_format != "dict":
            raise NotImplementedError("fallback only supports dict return format")
        if bg is None:
            bg = get_bg(grid)
        fg = grid != int(bg)
        seen = np.zeros_like(fg, dtype=bool)
        out: list[dict] = []
        if connectivity == 4:
            steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
        else:
            steps = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
        for r, c in np.argwhere(fg):
            rr = int(r)
            cc = int(c)
            if seen[rr, cc]:
                continue
            stack = [(rr, cc)]
            coords = []
            seen[rr, cc] = True
            while stack:
                cr, cc2 = stack.pop()
                coords.append((cr, cc2))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc2 + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            out.append(
                {
                    "color": None,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
        return out


BACKGROUND = Color.BLACK
MARKER = Color.TEAL
MAIN_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.ORANGE)
SPRITE_SIZE = 4
GRID_SIZE_RANGE = (14, 20)
MARKER_COUNT_OPTIONS = (
    (1, 1, 1, 2),
    (1, 1, 2, 3),
    (1, 2, 2, 3),
)
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 300


def _all_row_rolls_distinct(mask: np.ndarray) -> bool:
    signatures = {np.roll(mask, shift, axis=0).tobytes() for shift in range(SPRITE_SIZE)}
    return len(signatures) == SPRITE_SIZE


def _sprite_signature(sprite: np.ndarray) -> bytes:
    return sprite.astype(np.int8, copy=False).tobytes()


def _make_sprite(main_color: int, marker_count: int, rng: random.Random) -> np.ndarray:
    while True:
        sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), int(main_color), dtype=int)
        chosen = rng.sample(range(SPRITE_SIZE * SPRITE_SIZE), int(marker_count))
        for idx in chosen:
            sprite[idx // SPRITE_SIZE, idx % SPRITE_SIZE] = int(MARKER)
        marker_mask = sprite == int(MARKER)
        if _all_row_rolls_distinct(marker_mask):
            return sprite


def _sample_positions(height: int, width: int, rng: random.Random) -> list[tuple[int, int]] | None:
    valid_rows = list(range(1, height - SPRITE_SIZE))
    valid_cols = list(range(1, width - SPRITE_SIZE))
    if not valid_rows or not valid_cols:
        return None

    blocked = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int]] = []
    candidates = [(r, c) for r in valid_rows for c in valid_cols]
    rng.shuffle(candidates)
    for r, c in candidates:
        if blocked[r : r + SPRITE_SIZE, c : c + SPRITE_SIZE].any():
            continue
        placements.append((int(r), int(c)))
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(height, r + SPRITE_SIZE + 1)
        c1 = min(width, c + SPRITE_SIZE + 1)
        blocked[r0:r1, c0:c1] = True
        if len(placements) == len(MAIN_COLORS):
            return placements
    return None


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    non_bg = colors_present(g, exclude={bg})
    assert non_bg, "expected foreground colors"

    counts = Counter(int(v) for v in g[g != bg].reshape(-1))
    min_count = min(counts[color] for color in non_bg)
    marker_candidates = [color for color in non_bg if counts[color] == min_count]
    assert len(marker_candidates) == 1, "marker color must be uniquely least frequent"
    marker = int(marker_candidates[0])

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert components, "expected at least one sprite component"

    sprites: list[dict] = []
    for comp in components:
        box = tuple(map(int, comp["bbox"]))
        assert bbox_h(box) == SPRITE_SIZE and bbox_w(box) == SPRITE_SIZE, "every sprite must be 4x4"
        assert int(comp["area"]) == SPRITE_SIZE * SPRITE_SIZE, "every sprite must fill its 4x4 bbox"
        sprite = crop_bbox(g, box)
        palette = colors_present(sprite)
        assert len(palette) == 2 and marker in palette, "sprite must use one main color plus the marker color"
        marker_count = int(np.count_nonzero(sprite == marker))
        assert 1 <= marker_count <= 3, "marker count must be between 1 and 3"
        sprites.append(
            {
                "bbox": box,
                "sprite": sprite,
                "marker_count": marker_count,
                "main_color": int(next(color for color in palette if color != marker)),
            }
        )
    return marker, sprites


def _expected_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    marker, sprites = _extract_sprites(grid)
    scores = [item["marker_count"] for item in sprites]
    best = max(scores)
    winners = [item for item in sprites if item["marker_count"] == best]
    assert len(winners) == 1, "expected a unique maximum marker count"
    winner = winners[0]
    rolled = np.roll(winner["sprite"], shift=int(winner["marker_count"]), axis=0)
    return rolled, {"marker": marker, "sprites": sprites, "winner": winner}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the 4x4 sprite with the most pixels of the globally rare marker color, then roll it downward by that same count.

    Input structure:
    - The background is black, and the foreground consists of several separated solid 4x4 sprites.
    - Each sprite is filled with exactly two colors: one dominant main color and one globally rare marker color.
    - The marker color is the unique least frequent non-background color in the whole grid.

    Rule:
    - Infer the marker color as the unique least frequent non-background color.
    - Extract every 4-connected foreground sprite and count how many marker pixels it contains.
    - Select the unique sprite with the largest marker-color count; call that count K.
    - Crop that sprite and cyclically shift its rows downward by K.

    Key insight:
    - The same marker-count measurement is used twice: first to decide which sprite wins, and then again to decide how far the winning sprite is rolled.
    """
    out, _meta = _expected_output(np.asarray(grid, dtype=int))
    return out


def _validate_instance(grid: np.ndarray) -> None:
    expected, meta = _expected_output(grid)
    sprites = meta["sprites"]
    winner = meta["winner"]

    assert grid.ndim == 2
    assert 2 <= len(sprites) <= len(MAIN_COLORS)
    assert len(sprites) == len(MAIN_COLORS), "generator should place four sprites with distinct main colors"
    assert len({item["main_color"] for item in sprites}) == len(sprites), "main colors must be distinct"
    assert meta["marker"] == int(MARKER), "teal should be the globally rare marker color"
    assert winner["marker_count"] in {2, 3}, "winner count should vary the roll amount meaningfully"

    unrolled = winner["sprite"]
    shifts = [np.roll(unrolled, shift, axis=0) for shift in range(SPRITE_SIZE)]
    assert len({_sprite_signature(sprite) for sprite in shifts}) == SPRITE_SIZE, "winner must not be row-periodic"
    assert _sprite_signature(expected) != _sprite_signature(unrolled), "roll must change the winner sprite"

    non_winner_counts = [item["marker_count"] for item in sprites if item is not winner]
    assert all(count < winner["marker_count"] for count in non_winner_counts), "winner must be unique"
    assert sum(item["marker_count"] for item in sprites) < min(
        int(np.count_nonzero(grid == color)) for color in MAIN_COLORS
    ), "marker color must remain globally rarer than every main color"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIZE_RANGE)
    width = rng.randint(*GRID_SIZE_RANGE)

    positions = None
    for _ in range(PLACEMENT_RETRIES):
        positions = _sample_positions(height, width, rng)
        if positions is not None:
            break
    if positions is None:
        return None

    grid = np.full((height, width), int(BACKGROUND), dtype=int)
    counts = list(rng.choice(MARKER_COUNT_OPTIONS))
    rng.shuffle(counts)
    colors = list(MAIN_COLORS)
    rng.shuffle(colors)

    used_signatures: set[bytes] = set()
    for (r, c), main_color, marker_count in zip(positions, colors, counts):
        for _ in range(50):
            sprite = _make_sprite(main_color, marker_count, rng)
            sig = _sprite_signature(sprite)
            if sig not in used_signatures:
                used_signatures.add(sig)
                break
        else:
            return None
        grid[r : r + SPRITE_SIZE, c : c + SPRITE_SIZE] = sprite

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
