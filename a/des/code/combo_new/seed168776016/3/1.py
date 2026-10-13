from __future__ import annotations

import random

import numpy as np


BG = 0
ENTRY_H = 3
ENTRY_W = 4
ENTRY_GAP = 1
TEXTURE_SIZE = 3
COLOR_POOL = tuple(range(1, 10))
GRID_MIN = 15
GRID_MAX = 30
PALETTE_ENTRY_RANGE = (2, 4)
FRAME_COUNT_RANGE = (1, 4)
FRAME_SIZE_RANGE = (5, 11)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 300

SIDE_TO_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
CORNERS = ("tl", "tr", "bl", "br")


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _parse_palette_entry(block: np.ndarray) -> tuple[int, np.ndarray] | None:
    if block.shape != (ENTRY_H, ENTRY_W):
        return None
    if np.any(block == BG):
        return None
    key = int(block[0, 0])
    if key == BG or not np.all(block[:, 0] == key):
        return None
    texture = block[:, 1:].copy()
    colors = set(map(int, np.unique(texture)))
    if BG in colors or len(colors) != 2:
        return None
    return key, texture


def _scan_corner_palette(grid: np.ndarray, corner: str) -> dict | None:
    h, w = grid.shape
    if w < ENTRY_W or h < ENTRY_H:
        return None

    c0 = 0 if corner.endswith("l") else w - ENTRY_W
    entries: list[tuple[int, np.ndarray]] = []
    positions: list[tuple[int, int]] = []

    if corner.startswith("t"):
        r0 = 0
        while r0 + ENTRY_H <= h:
            parsed = _parse_palette_entry(grid[r0 : r0 + ENTRY_H, c0 : c0 + ENTRY_W])
            if parsed is None:
                break
            entries.append(parsed)
            positions.append((r0, c0))
            if r0 + ENTRY_H < h and np.any(grid[r0 + ENTRY_H, c0 : c0 + ENTRY_W] != BG):
                break
            r0 += ENTRY_H + ENTRY_GAP
    else:
        r0 = h - ENTRY_H
        while r0 >= 0:
            parsed = _parse_palette_entry(grid[r0 : r0 + ENTRY_H, c0 : c0 + ENTRY_W])
            if parsed is None:
                break
            entries.append(parsed)
            positions.append((r0, c0))
            if r0 - 1 >= 0 and np.any(grid[r0 - 1, c0 : c0 + ENTRY_W] != BG):
                break
            r0 -= ENTRY_H + ENTRY_GAP

    if not entries:
        return None

    keys = [key for key, _texture in entries]
    if len(set(keys)) != len(keys):
        return None

    mask = np.zeros_like(grid, dtype=bool)
    mapping: dict[int, np.ndarray] = {}
    for (key, texture), (r0, c0) in zip(entries, positions):
        mask[r0 : r0 + ENTRY_H, c0 : c0 + ENTRY_W] = True
        mapping[int(key)] = texture

    return {
        "corner": corner,
        "entries": entries,
        "positions": positions,
        "mapping": mapping,
        "mask": mask,
    }


def _extract_palette(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    candidates = [_scan_corner_palette(grid, corner) for corner in CORNERS]
    candidates = [cand for cand in candidates if cand is not None]
    assert candidates, "expected a corner palette made of 3x4 entries"
    best = max(candidates, key=lambda cand: len(cand["entries"]))
    return best["mapping"], best["mask"]


def _extract_frame(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_color = int(patch[0, 0])
    if border_color == BG:
        return None
    if not (
        np.all(patch[0, :] == border_color)
        and np.all(patch[-1, :] == border_color)
        and np.all(patch[:, 0] == border_color)
        and np.all(patch[:, -1] == border_color)
    ):
        return None

    perimeter = 2 * h + 2 * w - 4
    if int(np.count_nonzero(patch)) != perimeter + 1:
        return None

    interior = patch[1:-1, 1:-1]
    marker_positions = np.argwhere(interior != BG)
    if len(marker_positions) != 1:
        return None
    mr, mc = map(int, marker_positions[0] + 1)
    marker_color = int(patch[mr, mc])
    if marker_color == border_color:
        return None

    turns: list[int] = []
    if mr == 1:
        turns.append(0)
    if mc == w - 2:
        turns.append(1)
    if mr == h - 2:
        turns.append(2)
    if mc == 1:
        turns.append(3)
    if len(turns) != 1:
        return None

    return {
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "border_color": border_color,
        "marker_color": marker_color,
        "turns": turns[0],
    }


def _tile_texture(texture: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % TEXTURE_SIZE
    cc = np.arange(width)[None, :] % TEXTURE_SIZE
    return texture[rr, cc]


def _apply_frames(grid: np.ndarray, palette: dict[int, np.ndarray], palette_mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    frame_mask = (grid != BG) & (~palette_mask)
    frames: list[dict] = []
    for comp in _components(frame_mask):
        info = _extract_frame(grid, comp)
        if info is not None:
            frames.append(info)

    assert frames, "expected at least one framed rectangle"
    for frame in frames:
        texture = palette.get(int(frame["border_color"]))
        assert texture is not None, "missing palette entry for frame color"
        rotated = np.rot90(texture, -int(frame["turns"]))
        inner_h = int(frame["h"]) - 2
        inner_w = int(frame["w"]) - 2
        tiled = _tile_texture(rotated, inner_h, inner_w)
        r0 = int(frame["r0"]) + 1
        c0 = int(frame["c0"]) + 1
        out[r0 : r0 + inner_h, c0 : c0 + inner_w] = tiled
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up each frame color in the corner palette, rotate its texture by the marker side, and tile-fill the framed interior.

    Input structure:
    - The background is black.
    - One corner contains a palette made of one or more 3x4 entries; each entry has a solid color key column on the left and a 3x3 two-color texture on the right, with black separator rows between entries.
    - Elsewhere, the grid contains several one-pixel-thick rectangular frames in palette key colors, and each frame contains exactly one non-black single-pixel marker on one inner side.

    Rule:
    - Read the corner palette to build a mapping from frame color to its 3x3 texture.
    - For each rectangular frame, find the unique interior marker and interpret its side as a clockwise rotation code: top=0, right=90, bottom=180, left=270.
    - Rotate the looked-up texture by that amount and tile it from the interior top-left corner across the whole inner rectangle, cropping the last partial tiles as needed.
    - Keep every frame border and the palette unchanged; the marker disappears because the tiled texture overwrites the interior.

    Key insight:
    - The puzzle combines two independent inferences: the corner palette links border colors to textures, and the marker position independently encodes the rotation to apply before tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    palette, palette_mask = _extract_palette(g)
    return _apply_frames(g, palette, palette_mask)


def _palette_top_left(corner: str, h: int, w: int, palette_h: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - ENTRY_W
    if corner == "bl":
        return h - palette_h, 0
    if corner == "br":
        return h - palette_h, w - ENTRY_W
    raise ValueError(corner)


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _random_texture(rng: random.Random, key_color: int, seen: set[tuple[int, ...]]) -> np.ndarray | None:
    choices = [color for color in COLOR_POOL if color != key_color]
    for _ in range(200):
        a, b = rng.sample(choices, 2)
        bits = np.array([[rng.randint(0, 1) for _ in range(TEXTURE_SIZE)] for _ in range(TEXTURE_SIZE)], dtype=int)
        if int(bits.sum()) in (0, TEXTURE_SIZE * TEXTURE_SIZE):
            continue
        texture = np.where(bits == 0, a, b).astype(int)
        rots = [tuple(np.rot90(texture, k).ravel()) for k in range(4)]
        if len(set(rots)) != 4:
            continue
        sig = tuple(texture.ravel())
        if sig in seen:
            continue
        seen.add(sig)
        return texture
    return None


def _draw_palette(grid: np.ndarray, corner: str, entries: list[dict]) -> None:
    palette_h = len(entries) * ENTRY_H + (len(entries) - 1) * ENTRY_GAP
    pr, pc = _palette_top_left(corner, grid.shape[0], grid.shape[1], palette_h)
    for idx, entry in enumerate(entries):
        r0 = pr + idx * (ENTRY_H + ENTRY_GAP)
        grid[r0 : r0 + ENTRY_H, pc] = int(entry["key"])
        grid[r0 : r0 + ENTRY_H, pc + 1 : pc + ENTRY_W] = entry["texture"]


def _marker_position(side: str, h: int, w: int, rng: random.Random) -> tuple[int, int]:
    if side == "top":
        return 1, rng.randint(2, w - 3)
    if side == "right":
        return rng.randint(2, h - 3), w - 2
    if side == "bottom":
        return h - 2, rng.randint(2, w - 3)
    if side == "left":
        return rng.randint(2, h - 3), 1
    raise ValueError(side)


def _draw_frame(grid: np.ndarray, spec: dict) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    h = int(spec["h"])
    w = int(spec["w"])
    key = int(spec["key"])
    grid[r0, c0 : c0 + w] = key
    grid[r0 + h - 1, c0 : c0 + w] = key
    grid[r0 : r0 + h, c0] = key
    grid[r0 : r0 + h, c0 + w - 1] = key
    mr, mc = _marker_position(str(spec["side"]), h, w, random.Random(spec["marker_seed"]))
    grid[r0 + mr, c0 + mc] = int(spec["marker_color"])


def _fill_expected(grid: np.ndarray, placed_frames: list[dict], palette_by_key: dict[int, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for frame in placed_frames:
        texture = palette_by_key[int(frame["key"])]
        turns = SIDE_TO_TURNS[str(frame["side"])]
        rotated = np.rot90(texture, -turns)
        inner_h = int(frame["h"]) - 2
        inner_w = int(frame["w"]) - 2
        tiled = _tile_texture(rotated, inner_h, inner_w)
        r0 = int(frame["r0"]) + 1
        c0 = int(frame["c0"]) + 1
        out[r0 : r0 + inner_h, c0 : c0 + inner_w] = tiled
    return out


def _place_frames(
    rng: random.Random,
    occupied: np.ndarray,
    frame_specs: list[dict],
) -> list[dict] | None:
    h, w = occupied.shape
    placed: list[dict] = []
    for spec in sorted(frame_specs, key=lambda item: item["h"] * item["w"], reverse=True):
        fh = int(spec["h"])
        fw = int(spec["w"])
        if h - fh - 2 < 1 or w - fw - 2 < 1:
            return None
        for _ in range(300):
            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            if not _can_place(occupied, r0, c0, fh, fw, PLACEMENT_PADDING):
                continue
            placed_spec = dict(spec)
            placed_spec["r0"] = r0
            placed_spec["c0"] = c0
            placed.append(placed_spec)
            _reserve(occupied, r0, c0, fh, fw, PLACEMENT_PADDING)
            break
        else:
            return None
    return placed


def _sample_palette_entries(rng: random.Random) -> list[dict] | None:
    n_entries = rng.randint(*PALETTE_ENTRY_RANGE)
    keys = rng.sample(COLOR_POOL, n_entries)
    seen: set[tuple[int, ...]] = set()
    entries: list[dict] = []
    for key in keys:
        texture = _random_texture(rng, key, seen)
        if texture is None:
            return None
        entries.append({"key": int(key), "texture": texture})
    return entries


def _sample_frame_specs(rng: random.Random, palette_keys: list[int]) -> list[dict]:
    n_frames = rng.randint(*FRAME_COUNT_RANGE)
    specs: list[dict] = []
    for _ in range(n_frames):
        h = rng.randint(*FRAME_SIZE_RANGE)
        w = rng.randint(*FRAME_SIZE_RANGE)
        key = rng.choice(palette_keys)
        marker_choices = [color for color in COLOR_POOL if color != key]
        specs.append(
            {
                "h": h,
                "w": w,
                "key": int(key),
                "side": rng.choice(tuple(SIDE_TO_TURNS)),
                "marker_color": int(rng.choice(marker_choices)),
                "marker_seed": rng.randrange(1 << 30),
            }
        )
    return specs


def _grid_bounds(corner: str, palette_h: int, frame_specs: list[dict]) -> tuple[int, int]:
    max_frame_h = max(int(spec["h"]) for spec in frame_specs)
    max_frame_w = max(int(spec["w"]) for spec in frame_specs)
    min_h = max(GRID_MIN, palette_h + 2, max_frame_h + 2)
    min_w = max(GRID_MIN, ENTRY_W + 2, max_frame_w + 2)
    if corner.endswith("l") or corner.endswith("r"):
        min_w = max(min_w, ENTRY_W + 6)
    if corner.startswith("t") or corner.startswith("b"):
        min_h = max(min_h, palette_h + 2)
    return min_h, min_w


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    entries = _sample_palette_entries(rng)
    if entries is None:
        return None

    palette_h = len(entries) * ENTRY_H + (len(entries) - 1) * ENTRY_GAP
    corner = rng.choice(CORNERS)
    frame_specs = _sample_frame_specs(rng, [int(entry["key"]) for entry in entries])
    min_h, min_w = _grid_bounds(corner, palette_h, frame_specs)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = rng.randint(min_h, GRID_MAX)
    grid_w = rng.randint(min_w, GRID_MAX)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    _draw_palette(grid, corner, entries)
    pr, pc = _palette_top_left(corner, grid_h, grid_w, palette_h)
    _reserve(occupied, pr, pc, palette_h, ENTRY_W, PLACEMENT_PADDING)

    placed_frames = _place_frames(rng, occupied, frame_specs)
    if placed_frames is None:
        return None
    for spec in placed_frames:
        _draw_frame(grid, spec)

    expected = _fill_expected(
        grid,
        placed_frames,
        {int(entry["key"]): entry["texture"] for entry in entries},
    )

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-frame ARC input")
