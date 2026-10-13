from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BLACK = 0
TEAL = 8
NON_ANCHOR_COLORS = (1, 2, 3, 4, 6, 7)
ALL_SPRITE_COLORS = NON_ANCHOR_COLORS + (TEAL,)

GRID_ROWS = (10, 16)
GRID_COLS = (20, 30)
SPRITE_COUNT = (4, 6)
MAX_ATTEMPTS = 400

_MASK_LIBRARY: list[np.ndarray] | None = None


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask, bbox


def _paste_sprite(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = out[top : top + h, left : left + w]
    patch[mask] = int(color)


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert TEAL in colors, "expected exactly one teal anchor sprite"
    assert len(colors) >= 2, "expected multiple colored sprites"

    sprites: list[dict] = []
    ref_mask = None
    anchor = None

    for color in colors:
        comps = _components_mask(g == color)
        assert len(comps) == 1, "each color must form one 4-connected sprite"
        coords = comps[0]
        mask, bbox = _mask_from_coords(coords)
        if ref_mask is None:
            ref_mask = mask
        else:
            assert mask.shape == ref_mask.shape, "all sprites must share one bbox size"
            assert np.array_equal(mask, ref_mask), "all sprites must share one normalized mask"
        sprite = {
            "color": int(color),
            "bbox": bbox,
            "top": int(bbox[0]),
            "left": int(bbox[1]),
            "right": int(bbox[3]),
        }
        sprites.append(sprite)
        if color == TEAL:
            anchor = sprite

    assert ref_mask is not None and anchor is not None
    return bg, sprites, ref_mask, anchor


def _render_output(
    shape: tuple[int, int],
    bg: int,
    sprites: list[dict],
    mask: np.ndarray,
    top_row: int,
    *,
    parity_mode: str = "puzzle",
) -> np.ndarray:
    h, w = mask.shape
    out = np.full(shape, bg, dtype=int)
    ordered = sorted(sprites, key=lambda s: s["left"])

    for sprite in ordered:
        _paste_sprite(out, mask, top_row, int(sprite["left"]), int(sprite["color"]))

    for left_sprite, right_sprite in zip(ordered, ordered[1:]):
        gap_l = int(left_sprite["right"]) + 1
        gap_r = int(right_sprite["left"]) - 1
        width = gap_r - gap_l + 1
        assert width >= 1, "adjacent sprites must have a positive gap"

        if parity_mode == "left":
            fill_color = int(left_sprite["color"])
        elif parity_mode == "right":
            fill_color = int(right_sprite["color"])
        else:
            fill_color = int(left_sprite["color"]) if width % 2 == 1 else int(right_sprite["color"])

        out[top_row : top_row + h, gap_l : gap_r + 1] = fill_color

    return out


def _gap_widths(sprites: list[dict]) -> list[int]:
    ordered = sorted(sprites, key=lambda s: s["left"])
    return [int(b["left"]) - int(a["right"]) - 1 for a, b in zip(ordered, ordered[1:])]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vertically align repeated sprites to the teal anchor, then color each gap rectangle by the aligned gap width parity.

    Input structure:
    - The grid has a black background and 4-connected monochromatic sprites in distinct colors.
    - Exactly one sprite is teal and serves as the anchor; every sprite shares the same normalized mask.
    - Sprite column ranges are disjoint, while input top rows may differ.

    Rule:
    - Find the teal sprite and record its top row R.
    - Clear the canvas, then paste every sprite unchanged in its original columns but with top row R.
    - Sort sprites from left to right. For each adjacent pair, measure the horizontal gap width W between their bounding boxes.
    - Fill the aligned gap rectangle spanning rows R..R+sprite_height-1. If W is odd, use the left sprite's color; if W is even, use the right sprite's color.

    Key insight:
    - The teal sprite fixes the only valid shared row, and the parity rule applies to the between-sprite distance after alignment, not to any sprite-intrinsic property.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites, mask, anchor = _extract_sprites(g)
    return _render_output(g.shape, bg, sprites, mask, int(anchor["top"]), parity_mode="puzzle")


def _build_mask_library() -> list[np.ndarray]:
    masks: dict[bytes, np.ndarray] = {}
    for h in range(2, 5):
        for w in range(2, 5):
            total = h * w
            for area in range(4, min(6, total - 1) + 1):
                for cells in combinations(range(total), area):
                    mask = np.zeros((h, w), dtype=bool)
                    for idx in cells:
                        mask[idx // w, idx % w] = True
                    if len(_components_mask(mask)) != 1:
                        continue
                    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                        continue
                    if not any(0 < int(row.sum()) < w for row in mask):
                        continue
                    if not any(0 < int(col.sum()) < h for col in mask.T):
                        continue
                    masks.setdefault(mask.tobytes(), mask.copy())
    library = list(masks.values())
    assert library, "mask library must not be empty"
    return library


def _mask_library() -> list[np.ndarray]:
    global _MASK_LIBRARY
    if _MASK_LIBRARY is None:
        _MASK_LIBRARY = _build_mask_library()
    return _MASK_LIBRARY


def _sample_gap_widths(rng: random.Random, count: int, max_total: int) -> list[int] | None:
    min_total = count
    max_total = min(max_total, 6 * count)
    if max_total < min_total:
        return None

    for _ in range(200):
        total = rng.randint(min_total, max_total)
        remaining = total
        gaps: list[int] = []
        for i in range(count):
            slots_left = count - i - 1
            lo = max(1, remaining - 6 * slots_left)
            hi = min(6, remaining - slots_left)
            if lo > hi:
                break
            value = rng.randint(lo, hi)
            gaps.append(value)
            remaining -= value
        if remaining == 0 and any(g % 2 == 0 for g in gaps) and any(g % 2 == 1 for g in gaps):
            return gaps
    return None


def _sample_tops(rng: random.Random, rows: int, sprite_h: int, count: int) -> tuple[int, list[int]] | None:
    candidates = list(range(rows - sprite_h + 1))
    if len(candidates) < count:
        return None
    for _ in range(200):
        chosen = sorted(rng.sample(candidates, count))
        if chosen[-1] - chosen[0] < 3:
            continue
        anchor_top = rng.choice(chosen[1:-1])
        others = [top for top in chosen if top != anchor_top]
        rng.shuffle(others)
        return int(anchor_top), [int(v) for v in others]
    return None


def _validate_example(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert inp.shape == out.shape
    assert max(inp.shape) <= 30
    assert max(out.shape) <= 30

    bg, sprites, mask, anchor = _extract_sprites(inp)
    assert bg == BLACK
    assert 4 <= len(sprites) <= 6
    assert sorted(s["color"] for s in sprites).count(TEAL) == 1

    h, _w = mask.shape
    tops = sorted(int(s["top"]) for s in sprites)
    assert len(set(tops)) == len(tops), "input top rows must be distinct"
    assert int(anchor["top"]) != tops[0] and int(anchor["top"]) != tops[-1], "anchor row must be non-extremal"

    gaps = _gap_widths(sprites)
    assert all(1 <= width <= 6 for width in gaps)
    assert any(width % 2 == 0 for width in gaps)
    assert any(width % 2 == 1 for width in gaps)

    expected = _render_output(inp.shape, bg, sprites, mask, int(anchor["top"]), parity_mode="puzzle")
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out), "transformation must change the grid"

    left_only = _render_output(inp.shape, bg, sprites, mask, int(anchor["top"]), parity_mode="left")
    right_only = _render_output(inp.shape, bg, sprites, mask, int(anchor["top"]), parity_mode="right")
    min_top = _render_output(inp.shape, bg, sprites, mask, tops[0], parity_mode="puzzle")
    max_top = _render_output(inp.shape, bg, sprites, mask, tops[-1], parity_mode="puzzle")

    assert not np.array_equal(out, left_only), "odd/even split must matter"
    assert not np.array_equal(out, right_only), "odd/even split must matter"
    assert not np.array_equal(out, min_top), "teal anchor must matter"
    assert not np.array_equal(out, max_top), "teal anchor must matter"

    ordered = sorted(sprites, key=lambda s: s["left"])
    for left_sprite, right_sprite in zip(ordered, ordered[1:]):
        gap_l = int(left_sprite["right"]) + 1
        gap_r = int(right_sprite["left"]) - 1
        width = gap_r - gap_l + 1
        fill_color = int(left_sprite["color"]) if width % 2 == 1 else int(right_sprite["color"])
        block = out[int(anchor["top"]) : int(anchor["top"]) + h, gap_l : gap_r + 1]
        assert block.shape == (h, width)
        assert np.all(block == fill_color), "gap rectangle must be uniformly recolored"


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    mask = rng.choice(_mask_library()).copy()
    sprite_h, sprite_w = mask.shape
    count = rng.randint(*SPRITE_COUNT)

    rows_lo = max(GRID_ROWS[0], sprite_h + count - 1)
    if rows_lo > GRID_ROWS[1]:
        return None
    rows = rng.randint(rows_lo, GRID_ROWS[1])

    gap_budget = GRID_COLS[1] - count * sprite_w
    gaps = _sample_gap_widths(rng, count - 1, gap_budget)
    if gaps is None:
        return None

    min_cols = count * sprite_w + sum(gaps)
    cols_lo = max(GRID_COLS[0], min_cols)
    if cols_lo > GRID_COLS[1]:
        return None
    cols = rng.randint(cols_lo, GRID_COLS[1])
    slack = cols - min_cols
    left_margin = rng.randint(0, slack)
    right_margin = slack - left_margin
    if left_margin + right_margin < 0:
        return None

    lefts: list[int] = []
    cursor = left_margin
    for i in range(count):
        lefts.append(cursor)
        cursor += sprite_w
        if i < count - 1:
            cursor += gaps[i]
    assert cursor + right_margin == cols

    top_sample = _sample_tops(rng, rows, sprite_h, count)
    if top_sample is None:
        return None
    anchor_top, other_tops = top_sample

    anchor_idx = rng.randrange(count)
    colors = rng.sample(list(NON_ANCHOR_COLORS), count - 1)
    tops_by_index = [None] * count
    tops_by_index[anchor_idx] = anchor_top
    j = 0
    for i in range(count):
        if i == anchor_idx:
            continue
        tops_by_index[i] = other_tops[j]
        j += 1

    sprite_colors = []
    non_anchor_iter = iter(colors)
    for i in range(count):
        sprite_colors.append(TEAL if i == anchor_idx else next(non_anchor_iter))

    inp = np.full((rows, cols), BLACK, dtype=int)
    sprites: list[dict] = []
    for i in range(count):
        top = int(tops_by_index[i])
        left = int(lefts[i])
        color = int(sprite_colors[i])
        _paste_sprite(inp, mask, top, left, color)
        sprites.append(
            {
                "color": color,
                "bbox": (top, left, top + sprite_h - 1, left + sprite_w - 1),
                "top": top,
                "left": left,
                "right": left + sprite_w - 1,
            }
        )

    out = _render_output(inp.shape, BLACK, sprites, mask, anchor_top, parity_mode="puzzle")

    try:
        solved = main(inp)
        if not np.array_equal(solved, out):
            return None
        _validate_example(inp, out)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
