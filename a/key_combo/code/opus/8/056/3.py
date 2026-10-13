from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


YELLOW = 4
BG = 0
DIRS = ("top", "bottom", "left", "right")
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _analyze_cup(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    mask = _component_mask(comp)
    h, w = mask.shape
    if h < 5 or w < 5:
        return None

    def expected(direction: str) -> np.ndarray:
        out = np.zeros((h, w), dtype=bool)
        out[:, 0] = True
        out[:, -1] = True
        out[0, :] = True
        out[-1, :] = True
        if direction == "top":
            out[0, 1:-1] = False
        elif direction == "bottom":
            out[-1, 1:-1] = False
        elif direction == "left":
            out[1:-1, 0] = False
        else:
            out[1:-1, -1] = False
        return out

    matches = [direction for direction in DIRS if np.array_equal(mask, expected(direction))]
    if len(matches) != 1:
        return None
    direction = matches[0]
    return {
        "direction": direction,
        "bbox": (r0, c0, r1, c1),
        "cavity": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "color": int(comp["color"]),
    }


def _reflect_coords(coords: list[tuple[int, int]], cup: dict) -> list[tuple[int, int]]:
    direction = cup["direction"]
    r0, c0, r1, c1 = cup["bbox"]
    reflected: list[tuple[int, int]] = []
    for r, c in coords:
        if direction == "top":
            reflected.append((2 * r0 + 1 - r, c))
        elif direction == "bottom":
            reflected.append((2 * r1 - 1 - r, c))
        elif direction == "left":
            reflected.append((r, 2 * c0 + 1 - c))
        else:
            reflected.append((r, 2 * c1 - 1 - c))
    return reflected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the outside sprite to the cup opening, reflect it through the rim, and fill the remaining cavity yellow.

    Input structure:
    - The background is black.
    - One larger non-background component is a one-pixel-thick cup outline: a rectangular frame missing the middle of exactly one side.
    - One smaller non-background component is a 4-connected irregular sprite placed outside the cup along the opening direction in a different color.

    Rule:
    - Detect which side of the cup is open and treat that cup-edge line as the entry side.
    - Slide the sprite straight toward the opening until its front touches the outer opening edge of the cup.
    - Reflect the slid sprite across that opening edge so the mirrored copy lands inside the cup cavity in the same sprite color.
    - Erase the original outside sprite, keep the cup walls unchanged, and fill every remaining black cavity cell with yellow.

    Key insight:
    - The cup opening determines both the only valid slide direction and the mirror axis, so the sprite must be moved relative to the cup before it can be placed correctly inside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert len(comps) == 2, "expected exactly one cup and one sprite"

    cup_candidates = []
    for comp in comps:
        info = _analyze_cup(comp)
        if info is not None:
            cup_candidates.append((comp, info))
    assert len(cup_candidates) == 1, "cup structure must be unique"

    cup_comp, cup = cup_candidates[0]
    sprite = comps[0] if comps[1] is cup_comp else comps[1]
    assert 3 <= int(sprite["area"]) <= 5
    assert int(sprite["color"]) != int(cup["color"])

    sr0, sc0, sr1, sc1 = sprite["bbox"]
    cr0, cc0, cr1, cc1 = cup["bbox"]
    cav_r0, cav_c0, cav_r1, cav_c1 = cup["cavity"]

    if cup["direction"] == "top":
        assert sr1 < cr0 and cav_c0 <= sc0 and sc1 <= cav_c1
        dr, dc = cr0 - sr1, 0
    elif cup["direction"] == "bottom":
        assert sr0 > cr1 and cav_c0 <= sc0 and sc1 <= cav_c1
        dr, dc = cr1 - sr0, 0
    elif cup["direction"] == "left":
        assert sc1 < cc0 and cav_r0 <= sr0 and sr1 <= cav_r1
        dr, dc = 0, cc0 - sc1
    else:
        assert sc0 > cc1 and cav_r0 <= sr0 and sr1 <= cav_r1
        dr, dc = 0, cc1 - sc0

    moved = [(r + dr, c + dc) for r, c in sprite["coords"]]
    reflected = _reflect_coords(moved, cup)
    reflected_set = set(reflected)
    assert len(reflected_set) == len(reflected)
    assert all(cav_r0 <= r <= cav_r1 and cav_c0 <= c <= cav_c1 for r, c in reflected)

    out = g.copy()
    for r, c in sprite["coords"]:
        out[r, c] = bg
    for r, c in reflected:
        out[r, c] = int(sprite["color"])
    for r in range(cav_r0, cav_r1 + 1):
        for c in range(cav_c0, cav_c1 + 1):
            if int(out[r, c]) == bg:
                out[r, c] = YELLOW
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q = deque([tuple(map(int, coords[0]))])
    cells = {tuple(map(int, cell)) for cell in coords}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _touches_all_bbox_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _enumerate_sprite_masks(axis: str) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    flip = np.flipud if axis == "ud" else np.fliplr
    for h in range(2, 5):
        for w in range(2, 5):
            for area in range(3, min(5, h * w) + 1):
                for cells in itertools.combinations(range(h * w), area):
                    mask = np.zeros((h, w), dtype=bool)
                    for idx in cells:
                        mask[idx // w, idx % w] = True
                    if mask.all():
                        continue
                    if not _touches_all_bbox_edges(mask):
                        continue
                    if not _is_connected(mask):
                        continue
                    if np.array_equal(mask, flip(mask)):
                        continue
                    out.append(mask)
    return tuple(out)


SPRITE_MASKS = {
    "ud": _enumerate_sprite_masks("ud"),
    "lr": _enumerate_sprite_masks("lr"),
}


def _cup_patch(cavity_h: int, cavity_w: int, direction: str, color: int) -> np.ndarray:
    patch = np.full((cavity_h + 2, cavity_w + 2), BG, dtype=int)
    patch[:, 0] = color
    patch[:, -1] = color
    patch[0, :] = color
    patch[-1, :] = color
    if direction == "top":
        patch[0, 1:-1] = BG
    elif direction == "bottom":
        patch[-1, 1:-1] = BG
    elif direction == "left":
        patch[1:-1, 0] = BG
    else:
        patch[1:-1, -1] = BG
    return patch


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    view = grid[r0 : r0 + h, c0 : c0 + w]
    if np.any((view != BG) & mask):
        return False
    view[mask] = color
    return True


def _validate_generated(inp: np.ndarray) -> None:
    out = main(inp)
    comps = _find_components(inp, BG)
    assert len(comps) == 2
    cup_info = [_analyze_cup(comp) for comp in comps]
    idxs = [i for i, info in enumerate(cup_info) if info is not None]
    assert len(idxs) == 1
    cup = cup_info[idxs[0]]
    assert cup is not None
    sprite = comps[1 - idxs[0]]
    cav_r0, cav_c0, cav_r1, cav_c1 = cup["cavity"]
    cavity = out[cav_r0 : cav_r1 + 1, cav_c0 : cav_c1 + 1]
    assert np.all(cavity != BG)
    assert np.any(cavity == YELLOW)
    assert np.any(cavity == int(sprite["color"]))
    assert not np.any((out == int(sprite["color"])) & (inp == int(sprite["color"])))
    if cup["direction"] == "top":
        assert np.any(out[cav_r0, cav_c0 : cav_c1 + 1] == int(sprite["color"]))
    elif cup["direction"] == "bottom":
        assert np.any(out[cav_r1, cav_c0 : cav_c1 + 1] == int(sprite["color"]))
    elif cup["direction"] == "left":
        assert np.any(out[cav_r0 : cav_r1 + 1, cav_c0] == int(sprite["color"]))
    else:
        assert np.any(out[cav_r0 : cav_r1 + 1, cav_c1] == int(sprite["color"]))
    assert not np.array_equal(inp, out)


def _generate_attempt() -> np.ndarray | None:
    direction = random.choice(DIRS)
    cavity_h = random.randint(3, 4)
    cavity_w = random.randint(3, 4)
    axis = "ud" if direction in ("top", "bottom") else "lr"
    candidates = [mask for mask in SPRITE_MASKS[axis] if mask.shape[0] <= cavity_h and mask.shape[1] <= cavity_w]
    if not candidates:
        return None
    inside_mask = random.choice(candidates)
    sprite_h, sprite_w = inside_mask.shape
    outside_mask = np.flipud(inside_mask) if axis == "ud" else np.fliplr(inside_mask)
    gap = random.randint(1, 4)

    H = random.randint(14, 18)
    W = random.randint(14, 18)
    cup_h = cavity_h + 2
    cup_w = cavity_w + 2

    if direction == "top":
        row_min = gap + sprite_h - 1
        row_max = H - cup_h
        col_min = 0
        col_max = W - cup_w
    elif direction == "bottom":
        row_min = 0
        row_max = H - cup_h - gap - sprite_h + 1
        col_min = 0
        col_max = W - cup_w
    elif direction == "left":
        row_min = 0
        row_max = H - cup_h
        col_min = gap + sprite_w - 1
        col_max = W - cup_w
    else:
        row_min = 0
        row_max = H - cup_h
        col_min = 0
        col_max = W - cup_w - gap - sprite_w + 1
    if row_min > row_max or col_min > col_max:
        return None

    cup_r0 = random.randint(row_min, row_max)
    cup_c0 = random.randint(col_min, col_max)

    if direction in ("top", "bottom"):
        offset = random.randint(0, cavity_w - sprite_w)
        sprite_r = cup_r0 + 1 if direction == "top" else cup_r0 + cavity_h - sprite_h + 1
        sprite_c = cup_c0 + 1 + offset
        input_r = cup_r0 - gap - sprite_h + 1 if direction == "top" else cup_r0 + cup_h - 1 + gap
        input_c = sprite_c
    else:
        offset = random.randint(0, cavity_h - sprite_h)
        sprite_r = cup_r0 + 1 + offset
        sprite_c = cup_c0 + 1 if direction == "left" else cup_c0 + cavity_w - sprite_w + 1
        input_r = sprite_r
        input_c = cup_c0 - gap - sprite_w + 1 if direction == "left" else cup_c0 + cup_w - 1 + gap

    cup_color, sprite_color = random.sample(PALETTE, 2)
    inp = np.full((H, W), BG, dtype=int)
    if not _stamp_mask(inp, _cup_patch(cavity_h, cavity_w, direction, cup_color) != BG, cup_r0, cup_c0, cup_color):
        return None
    if not _stamp_mask(inp, outside_mask, input_r, input_c, sprite_color):
        return None

    expected = np.full((H, W), BG, dtype=int)
    _stamp_mask(expected, _cup_patch(cavity_h, cavity_w, direction, cup_color) != BG, cup_r0, cup_c0, cup_color)
    _stamp_mask(expected, inside_mask, sprite_r, sprite_c, sprite_color)
    cavity = expected[cup_r0 + 1 : cup_r0 + 1 + cavity_h, cup_c0 + 1 : cup_c0 + 1 + cavity_w]
    cavity[cavity == BG] = YELLOW

    if not np.array_equal(main(inp), expected):
        return None
    _validate_generated(inp)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid ARC instance")
