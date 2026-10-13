from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NOTCH_TO_DIR = {
    (0, 1): "U",
    (2, 1): "D",
    (1, 0): "L",
    (1, 2): "R",
}
DIR_TO_NOTCH = {direction: cell for cell, direction in NOTCH_TO_DIR.items()}

BG = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (12, 28)
GRID_W_RANGE = (12, 28)
PAIR_COUNT_MAX = 4
SHIFT_RANGE = (1, 4)
PADDING = 1
MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr, sc in np.argwhere(grid != int(bg)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        color = int(grid[sr, sc])
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < h
                    and 0 <= cc < w
                    and not seen[rr, cc]
                    and int(grid[rr, cc]) == color
                ):
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        arr = np.array(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        comps.append(
            {
                "color": color,
                "coords": arr,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    return comps


def _sprite_info(comp: dict) -> dict:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    assert int(coords.shape[0]) == 8, "each sprite must contain 8 cells"
    assert int(r1 - r0) == 2 and int(c1 - c0) == 2, "each sprite must fit a 3x3 bbox"

    local = {(int(r - r0), int(c - c0)) for r, c in coords}
    full = {(r, c) for r in range(3) for c in range(3)}
    missing = list(full - local)
    assert len(missing) == 1 and missing[0] in NOTCH_TO_DIR, "missing cell must be an edge midpoint"

    return {
        "color": int(comp["color"]),
        "top": int(r0),
        "left": int(c0),
        "direction": NOTCH_TO_DIR[missing[0]],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each notched 3x3 sprite toward its opposite-facing partner and merge the pair into a solid bar.

    Input structure:
    - The background is black.
    - The foreground consists of separated same-color pairs of 3x3 sprites.
    - Every sprite is a solid 3x3 square missing exactly one edge-middle pixel, which encodes one of the four directions.
    - The two sprites of a color share one row or one column, and their missing-pixel directions face each other.

    Rule:
    - Detect the missing-pixel direction of every sprite from its 3x3 bounding box.
    - Pair sprites by color; horizontal `R/L` pairs close their horizontal gap, and vertical `D/U` pairs close their vertical gap.
    - After the gap is closed, replace the pair by one solid same-color rectangle spanning both adjacent 3x3 blocks, which repairs the two complementary notches.
    - Clear all original sprite positions; only the merged rectangles remain on the background.

    Key insight:
    - The missing edge cell is both a motion cue and a pairing cue: opposite notches identify which same-color sprites move together and along which axis they collapse into one repaired object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    sprites = [_sprite_info(comp) for comp in _components_by_color(g, bg)]

    by_color: dict[int, list[dict]] = {}
    for sprite in sprites:
        by_color.setdefault(int(sprite["color"]), []).append(sprite)

    out = np.full_like(g, int(bg))
    for color, pair in by_color.items():
        assert len(pair) == 2, "each color must appear in exactly one pair"
        directions = {sprite["direction"] for sprite in pair}

        if directions == {"L", "R"}:
            left_sprite, right_sprite = sorted(pair, key=lambda sprite: (sprite["left"], sprite["top"]))
            assert left_sprite["direction"] == "R" and right_sprite["direction"] == "L"
            assert left_sprite["top"] == right_sprite["top"], "horizontal pair must share a row"

            gap = int(right_sprite["left"] - left_sprite["left"] - 3)
            assert gap >= 0, "paired sprites must not overlap"
            left_move = gap // 2
            right_move = gap - left_move
            r0 = int(left_sprite["top"])
            c0 = int(left_sprite["left"] + left_move)
            c1 = int(right_sprite["left"] - right_move + 2)
            region = out[r0 : r0 + 3, c0 : c1 + 1]
            assert np.all((region == int(bg)) | (region == int(color))), "merged objects must not conflict"
            out[r0 : r0 + 3, c0 : c1 + 1] = int(color)
            continue

        assert directions == {"U", "D"}, "each pair must face along one axis"
        top_sprite, bottom_sprite = sorted(pair, key=lambda sprite: (sprite["top"], sprite["left"]))
        assert top_sprite["direction"] == "D" and bottom_sprite["direction"] == "U"
        assert top_sprite["left"] == bottom_sprite["left"], "vertical pair must share a column"

        gap = int(bottom_sprite["top"] - top_sprite["top"] - 3)
        assert gap >= 0, "paired sprites must not overlap"
        top_move = gap // 2
        bottom_move = gap - top_move
        r0 = int(top_sprite["top"] + top_move)
        r1 = int(bottom_sprite["top"] - bottom_move + 2)
        c0 = int(top_sprite["left"])
        region = out[r0 : r1 + 1, c0 : c0 + 3]
        assert np.all((region == int(bg)) | (region == int(color))), "merged objects must not conflict"
        out[r0 : r1 + 1, c0 : c0 + 3] = int(color)

    return out


def _draw_sprite(grid: np.ndarray, top: int, left: int, color: int, direction: str) -> None:
    patch = np.full((3, 3), int(color), dtype=int)
    notch_r, notch_c = DIR_TO_NOTCH[direction]
    patch[notch_r, notch_c] = int(BG)
    grid[top : top + 3, left : left + 3] = patch


def _max_pair_count(h: int, w: int) -> int:
    area = int(h * w)
    if area < 180:
        return 1
    if area < 260:
        return 2
    if area < 340:
        return 3
    return PAIR_COUNT_MAX


def _footprint_shape(orientation: str, shift: int) -> tuple[int, int]:
    if orientation == "H":
        return 3, 6 + 2 * int(shift)
    return 6 + 2 * int(shift), 3


def _place_specs(
    h: int,
    w: int,
    specs: list[dict],
    rng: random.Random,
) -> list[dict] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []

    ordered = specs[:]
    rng.shuffle(ordered)
    ordered.sort(key=lambda spec: _footprint_shape(spec["orientation"], spec["shift"])[0] * _footprint_shape(spec["orientation"], spec["shift"])[1], reverse=True)

    for spec in ordered:
        fh, fw = _footprint_shape(spec["orientation"], spec["shift"])
        candidates: list[tuple[int, int]] = []
        for top in range(h - fh + 1):
            for left in range(w - fw + 1):
                if occupied[top : top + fh, left : left + fw].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return None

        top, left = rng.choice(candidates)
        r0 = max(0, top - PADDING)
        c0 = max(0, left - PADDING)
        r1 = min(h, top + fh + PADDING)
        c1 = min(w, left + fw + PADDING)
        occupied[r0:r1, c0:c1] = True
        placements.append({**spec, "top": int(top), "left": int(left)})

    return placements


def _build_pair_grids(h: int, w: int, placements: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for spec in placements:
        top = int(spec["top"])
        left = int(spec["left"])
        color = int(spec["color"])
        shift = int(spec["shift"])

        if spec["orientation"] == "H":
            right_left = left + 3 + 2 * shift
            _draw_sprite(inp, top, left, color, "R")
            _draw_sprite(inp, top, right_left, color, "L")
            out[top : top + 3, left + shift : left + shift + 6] = color
        else:
            bottom_top = top + 3 + 2 * shift
            _draw_sprite(inp, top, left, color, "D")
            _draw_sprite(inp, bottom_top, left, color, "U")
            out[top + shift : top + shift + 6, left : left + 3] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    pair_count = rng.randint(1, _max_pair_count(h, w))
    colors = rng.sample(PALETTE, pair_count)

    orientations = [rng.choice(("H", "V")) for _ in range(pair_count)]
    if pair_count >= 2 and rng.random() < 0.5:
        orientations[0] = "H"
        orientations[1] = "V"
        rng.shuffle(orientations)

    specs = [
        {
            "color": int(color),
            "orientation": orientation,
            "shift": rng.randint(*SHIFT_RANGE),
        }
        for color, orientation in zip(colors, orientations)
    ]

    placements = _place_specs(h, w, specs, rng)
    if placements is None:
        return None

    inp, expected = _build_pair_grids(h, w, placements)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo_new seed1315637744/9/5")
