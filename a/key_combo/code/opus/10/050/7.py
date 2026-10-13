from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
ANCHOR_COLOR = 8
SPRITE_COLOR_POOL = [1, 2, 3, 4, 6]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _parse_palette(row: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    w = int(row.shape[0])
    c = 0
    while c < w:
        if int(row[c]) == BG:
            c += 1
            continue
        assert c + 1 < w and int(row[c + 1]) != BG, "palette entries must be adjacent non-black pairs"
        src = int(row[c])
        dst = int(row[c + 1])
        assert src != dst, "palette mapping must change the color"
        assert src not in mapping, "each source color must appear in exactly one palette pair"
        mapping[src] = dst
        c += 2
        if c < w:
            assert int(row[c]) == BG, "palette pairs must be separated by black cells"
    return mapping


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    body = np.asarray(grid[:-1], dtype=int)
    colors = [int(c) for c in np.unique(body) if int(c) != BG]
    sprites = []
    for color in colors:
        mask = body == color
        comps = _components(mask)
        assert len(comps) == 1, "each sprite color must form exactly one body component"
        coords = comps[0]
        b = _bbox(mask)
        assert b is not None
        sprites.append(
            {
                "color": color,
                "bbox": b,
                "mask": _crop_mask(mask, b),
                "top": int(b[0]),
                "left": int(b[1]),
                "count": int(np.count_nonzero(grid == color)),
            }
        )
    return sprites


def _compose(
    sprites: list[dict],
    shape: tuple[int, int],
    anchor_top: int,
    palette_map: dict[int, int],
    *,
    priority_mode: str = "correct",
    apply_palette: bool = True,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    best_count = np.full(shape, -1, dtype=int)
    best_top = np.full(shape, 10**9, dtype=int)
    best_left = np.full(shape, -10**9, dtype=int)
    chosen = np.full(shape, BG, dtype=int)

    for sprite in sprites:
        mask = sprite["mask"]
        top = int(anchor_top)
        left = int(sprite["left"])
        count = int(sprite["count"])
        original_top = int(sprite["top"])
        color = int(sprite["color"])
        for rr, cc in np.argwhere(mask):
            r = top + int(rr)
            c = left + int(cc)
            if priority_mode == "correct":
                better = count > best_count[r, c] or (
                    count == best_count[r, c] and original_top < best_top[r, c]
                )
            elif priority_mode == "top_only":
                better = original_top < best_top[r, c] or (
                    original_top == best_top[r, c] and count > best_count[r, c]
                )
            elif priority_mode == "rightmost_tie":
                better = count > best_count[r, c] or (
                    count == best_count[r, c] and left > best_left[r, c]
                )
            else:
                raise ValueError(f"unknown priority mode: {priority_mode}")

            if better:
                best_count[r, c] = count
                best_top[r, c] = original_top
                best_left[r, c] = left
                chosen[r, c] = color
            elif (
                count == best_count[r, c]
                and original_top == best_top[r, c]
                and left == best_left[r, c]
                and chosen[r, c] not in (BG, color)
            ):
                raise AssertionError("priority must remain unique on every overlap cell")

    out[:, :] = chosen
    if apply_palette:
        for src, dst in palette_map.items():
            out[out == int(src)] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align same-shaped sprites to the teal sprite, resolve aligned overlaps by original area priority, then recolor via the bottom palette.

    Input structure:
    - The grid is a black background with one bottom palette row.
    - Above the palette are several monochrome 4-connected sprite bodies, including one teal anchor sprite.
    - All sprite bodies share the same normalized mask after cropping to their bounding boxes.
    - The bottom row contains adjacent 1x1 palette pairs: left color = sprite color, right color = output color.

    Rule:
    - Find the teal sprite body and use its top row as the shared alignment row.
    - Vertically translate every non-teal sprite body so its top row matches the teal top row, keeping each sprite's columns fixed.
    - When aligned sprites cover the same cell, keep the sprite whose input color occupies more total pixels in the whole input; if tied, keep the sprite whose original top row is higher.
    - Apply the palette-row recoloring to every non-teal sprite color and return the grid without the bottom palette row.

    Key insight:
    - Geometry, visibility, and recoloring are separate clues: the teal sprite fixes position, total input area fixes overlap priority, and the palette row fixes only the final colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2
    body = g[:-1]
    palette_map = _parse_palette(g[-1])
    sprites = _extract_sprites(g)

    anchors = [sprite for sprite in sprites if int(sprite["color"]) == ANCHOR_COLOR]
    assert len(anchors) == 1, "expected exactly one teal anchor sprite"
    anchor = anchors[0]
    reference_mask = anchor["mask"]
    for sprite in sprites:
        assert sprite["mask"].shape == reference_mask.shape
        assert np.array_equal(sprite["mask"], reference_mask), "all sprite bodies must share the same normalized mask"

    body_colors = {int(sprite["color"]) for sprite in sprites}
    non_anchor_colors = body_colors - {ANCHOR_COLOR}
    assert set(palette_map) == non_anchor_colors, "palette must map every non-teal sprite color exactly once"
    assert ANCHOR_COLOR not in palette_map
    target_colors = set(palette_map.values())
    assert len(target_colors) == len(palette_map)
    assert not (target_colors & body_colors), "palette targets must differ from all sprite body colors"

    return _compose(sprites, body.shape, int(anchor["top"]), palette_map, priority_mode="correct", apply_palette=True)


def _mask_overlap(mask: np.ndarray, shift: int) -> bool:
    h, w = mask.shape
    if shift <= 0 or shift >= w:
        return False
    return bool(np.any(mask[:, : w - shift] & mask[:, shift:]))


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if _bbox(mask) != (0, 0, h - 1, w - 1):
        return None
    return mask


def _generate_mask() -> tuple[np.ndarray, int] | None:
    for _ in range(120):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        lo = max(4, h + w - 2)
        hi = min(6, h * w)
        if lo > hi:
            continue
        area = random.randint(lo, hi)
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue
        options = [d for d in range(1, w) if _mask_overlap(mask, d) and _mask_overlap(mask, w - d)]
        if options:
            return mask, random.choice(options)
    return None


def _place_palette(width: int, mappings: list[tuple[int, int]]) -> np.ndarray | None:
    need = 2 * len(mappings) + max(0, len(mappings) - 1)
    if need > width:
        return None
    slack = width - need
    extras = [0] * (len(mappings) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1
    row = np.zeros(width, dtype=int)
    c = extras[0]
    for i, (src, dst) in enumerate(mappings):
        row[c] = int(src)
        row[c + 1] = int(dst)
        c += 2
        if i < len(mappings) - 1:
            c += 1 + extras[i + 1]
    return row


def _semantic_checks(sprites: list[dict], shape: tuple[int, int], anchor_top: int, palette_map: dict[int, int]) -> bool:
    correct = _compose(sprites, shape, anchor_top, palette_map, priority_mode="correct", apply_palette=True)
    if np.array_equal(correct, _compose(sprites, shape, anchor_top, palette_map, priority_mode="top_only", apply_palette=True)):
        return False
    if np.array_equal(correct, _compose(sprites, shape, anchor_top, palette_map, priority_mode="rightmost_tie", apply_palette=True)):
        return False
    if np.array_equal(correct, _compose(sprites, shape, anchor_top, palette_map, priority_mode="correct", apply_palette=False)):
        return False
    return True


def _build_attempt() -> np.ndarray | None:
    total_sprites = random.randint(4, 5)
    non_anchor_count = total_sprites - 1

    generated = _generate_mask()
    if generated is None:
        return None
    mask, overlap_delta = generated
    h, w = mask.shape
    body_area = int(mask.sum())

    used_colors = random.sample(SPRITE_COLOR_POOL, non_anchor_count)
    available_targets = [c for c in range(1, 10) if c not in set(used_colors) | {ANCHOR_COLOR}]
    if len(available_targets) < non_anchor_count:
        return None
    target_colors = random.sample(available_targets, non_anchor_count)

    base_body_width = 2 * w + max(0, total_sprites - 3) * (w + 1)
    min_palette_width = 2 * non_anchor_count + max(0, non_anchor_count - 1)
    min_width = max(base_body_width, min_palette_width)
    max_width = min(20, min_width + 5)
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)

    base_left = random.randint(0, width - base_body_width)
    lefts = [base_left, base_left + overlap_delta, base_left + w]
    for i in range(total_sprites - 3):
        lefts.append(base_left + 2 * w + 1 + i * (w + 1))

    step = h + random.randint(0, 1)
    min_body_rows = h + step * (total_sprites - 1)
    body_rows = random.randint(min_body_rows, min(24, min_body_rows + 4))
    base_top = random.randint(0, body_rows - min_body_rows)
    tops = [base_top + i * step for i in range(total_sprites)]
    anchor_top = tops[0]

    sprites = [
        {
            "color": ANCHOR_COLOR,
            "mask": mask,
            "top": tops[0],
            "left": lefts[0],
            "count": body_area,
        },
        {
            "color": used_colors[0],
            "mask": mask,
            "top": tops[1],
            "left": lefts[1],
            "count": body_area + 1,
        },
        {
            "color": used_colors[1],
            "mask": mask,
            "top": tops[2],
            "left": lefts[2],
            "count": body_area + 1,
        },
    ]

    for idx in range(3, total_sprites):
        sprites.append(
            {
                "color": used_colors[idx - 1],
                "mask": mask,
                "top": tops[idx],
                "left": lefts[idx],
                "count": body_area + 1,
            }
        )

    random.shuffle(target_colors)
    palette_items = list(zip(used_colors, target_colors))
    random.shuffle(palette_items)
    palette_map = {int(src): int(dst) for src, dst in palette_items}
    if not _semantic_checks(sprites, (body_rows, width), anchor_top, palette_map):
        return None

    body = np.zeros((body_rows, width), dtype=int)
    for sprite in sprites:
        patch = np.where(mask, int(sprite["color"]), BG)
        r0 = int(sprite["top"])
        c0 = int(sprite["left"])
        view = body[r0 : r0 + h, c0 : c0 + w]
        if view.shape != patch.shape or np.any(view[mask] != BG):
            return None
        view[mask] = int(sprite["color"])

    palette_row = _place_palette(width, palette_items)
    if palette_row is None:
        return None

    grid = np.vstack([body, palette_row])
    try:
        solved = main(grid)
    except Exception:
        return None

    expected = _compose(sprites, body.shape, anchor_top, palette_map, priority_mode="correct", apply_palette=True)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
