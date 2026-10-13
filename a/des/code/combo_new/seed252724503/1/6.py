from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
OPENINGS = ("up", "down", "left", "right")
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_MIN = 1
OBJECT_COUNT_MAX = 3
ENVELOPE_SIZE_MIN = 6
ENVELOPE_SIZE_MAX = 9
STAMP_GAPS = (2, 3)
PLACEMENT_ATTEMPTS = 250
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _frame_coords_for_bbox(
    bbox: tuple[int, int, int, int], opening: str
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords: set[tuple[int, int]] = set()
    if opening != "up":
        for c in range(c0, c1 + 1):
            coords.add((r0, c))
    if opening != "down":
        for c in range(c0, c1 + 1):
            coords.add((r1, c))
    if opening != "left":
        for r in range(r0, r1 + 1):
            coords.add((r, c0))
    if opening != "right":
        for r in range(r0, r1 + 1):
            coords.add((r, c1))
    return coords


def _detect_opening(coords: np.ndarray) -> str | None:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    coord_set = {tuple(map(int, rc)) for rc in coords.tolist()}
    matches = [opening for opening in OPENINGS if coord_set == _frame_coords_for_bbox(bbox, opening)]
    if len(matches) != 1:
        return None
    return matches[0]


def _stamp_matches(
    stamp: tuple[int, int], bbox: tuple[int, int, int, int], opening: str
) -> bool:
    r, c = stamp
    r0, c0, r1, c1 = bbox
    if opening == "up":
        return r > r1
    if opening == "down":
        return r < r0
    if opening == "left":
        return c > c1
    return c < c0


def _find_envelopes(grid: np.ndarray) -> list[dict]:
    envelopes: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        components = _connected_components(grid == color)
        stamps = [tuple(map(int, comp[0])) for comp in components if int(comp.shape[0]) == 1]
        for comp in components:
            if int(comp.shape[0]) <= 1:
                continue
            opening = _detect_opening(comp)
            if opening is None:
                continue
            bbox = _bbox_from_coords(comp)
            if not any(_stamp_matches(stamp, bbox, opening) for stamp in stamps):
                continue
            r0, c0, r1, c1 = bbox
            letter: list[tuple[int, int, int]] = []
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    value = int(grid[r, c])
                    if value != BG:
                        letter.append((r, c, value))
            if not letter:
                continue
            envelopes.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "opening": opening,
                    "letter": letter,
                }
            )
    envelopes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    return envelopes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each letter out of its open envelope and place it just beyond the gap.

    Input structure:
    - The background is black.
    - Each envelope is a single-color rectangular border with exactly one missing side.
    - A multi-color letter pattern sits strictly inside the envelope, not touching the border.
    - A same-color singleton stamp sits outside the envelope on the side opposite the opening.

    Rule:
    - Detect the missing border side of each envelope to determine the opening direction.
    - Erase the letter from inside the envelope and translate the whole letter straight through the opening.
    - Stop when the letter's innermost edge lands exactly one cell outside the opening; keep the envelope and stamp unchanged.
    - If the moved letter lands on top of other objects, the moved letter overwrites them.

    Key insight:
    - The true cue is the missing edge of the rectangular frame: the stamp is only a distractor sharing the frame color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    moves: list[tuple[list[tuple[int, int, int]], int, int]] = []

    for envelope in _find_envelopes(g):
        letter = envelope["letter"]
        rows = [r for r, _c, _v in letter]
        cols = [c for _r, c, _v in letter]
        r0, c0, r1, c1 = envelope["bbox"]
        opening = envelope["opening"]

        if opening == "up":
            dr, dc = r0 - 1 - max(rows), 0
        elif opening == "down":
            dr, dc = r1 + 1 - min(rows), 0
        elif opening == "left":
            dr, dc = 0, c0 - 1 - max(cols)
        else:
            dr, dc = 0, c1 + 1 - min(cols)

        moves.append((letter, dr, dc))
        for r, c, _value in letter:
            out[r, c] = BG

    h, w = out.shape
    for letter, dr, dc in moves:
        for r, c, value in letter:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = value
    return out


def _grow_shape(height: int, width: int, cells_needed: int, rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(30):
        start = (rng.randrange(height), rng.randrange(width))
        cells = {start}
        frontier = [start]
        while len(cells) < cells_needed and frontier:
            r, c = rng.choice(frontier)
            candidates = _neighbors4(r, c)
            rng.shuffle(candidates)
            added = False
            for nr, nc in candidates:
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    cells.add((nr, nc))
                    frontier.append((nr, nc))
                    added = True
                    break
            if not added:
                frontier.remove((r, c))
        if len(cells) == cells_needed:
            return sorted(cells)
    return None


def _make_letter(
    max_height: int, max_width: int, palette: list[int], rng: random.Random
) -> list[tuple[int, int, int]] | None:
    if max_height < 2 or max_width < 2 or len(palette) < 2:
        return None
    for _ in range(40):
        height = rng.randint(2, max_height)
        width = rng.randint(2, max_width)
        max_cells = min(8, height * width)
        cells_needed = rng.randint(3, max_cells)
        shape = _grow_shape(height, width, cells_needed, rng)
        if shape is None:
            continue
        color_count = rng.randint(2, min(4, len(palette), len(shape)))
        colors = rng.sample(palette, color_count)
        assigned = colors[:]
        while len(assigned) < len(shape):
            assigned.append(rng.choice(colors))
        rng.shuffle(assigned)
        return [(r, c, color) for (r, c), color in zip(shape, assigned)]
    return None


def _make_local_object(
    envelope_color: int, opening: str, letter_palette: list[int], rng: random.Random
) -> dict | None:
    height = rng.randint(ENVELOPE_SIZE_MIN, ENVELOPE_SIZE_MAX)
    width = rng.randint(ENVELOPE_SIZE_MIN, ENVELOPE_SIZE_MAX)

    safe_height = height - 4
    safe_width = width - 4
    letter_patch = _make_letter(safe_height, safe_width, letter_palette, rng)
    if letter_patch is None:
        return None

    patch_max_r = max(r for r, _c, _v in letter_patch)
    patch_max_c = max(c for _r, c, _v in letter_patch)
    top = rng.randint(2, height - patch_max_r - 3)
    left = rng.randint(2, width - patch_max_c - 3)
    letter = [(top + r, left + c, value) for r, c, value in letter_patch]

    rows = [r for r, _c, _v in letter]
    cols = [c for _r, c, _v in letter]
    if opening == "up":
        dr, dc = -1 - max(rows), 0
    elif opening == "down":
        dr, dc = height - min(rows), 0
    elif opening == "left":
        dr, dc = 0, -1 - max(cols)
    else:
        dr, dc = 0, width - min(cols)
    moved_letter = [(r + dr, c + dc, value) for r, c, value in letter]

    frame = [(r, c, envelope_color) for r, c in sorted(_frame_coords_for_bbox((0, 0, height - 1, width - 1), opening))]
    gap = rng.choice(STAMP_GAPS)
    if opening == "up":
        stamp = (height - 1 + gap, rng.randrange(width), envelope_color)
    elif opening == "down":
        stamp = (-gap, rng.randrange(width), envelope_color)
    elif opening == "left":
        stamp = (rng.randrange(height), width - 1 + gap, envelope_color)
    else:
        stamp = (rng.randrange(height), -gap, envelope_color)

    input_pixels = frame + [stamp] + letter
    output_pixels = frame + [stamp] + moved_letter
    all_coords = [(r, c) for r, c, _v in input_pixels + output_pixels]
    min_r = min(r for r, _c in all_coords)
    max_r = max(r for r, _c in all_coords)
    min_c = min(c for _r, c in all_coords)
    max_c = max(c for _r, c in all_coords)
    return {
        "input_pixels": input_pixels,
        "output_pixels": output_pixels,
        "bounds": (min_r, max_r, min_c, max_c),
        "span_area": (max_r - min_r + 1) * (max_c - min_c + 1),
    }


def _translate_pixels(
    pixels: list[tuple[int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int]]:
    return [(r + dr, c + dc, value) for r, c, value in pixels]


def _expand_coords(coords: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _place_objects(specs: list[dict], h: int, w: int, rng: random.Random) -> list[dict] | None:
    placed: list[dict] = []
    input_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()
    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()

    for spec in sorted(specs, key=lambda item: item["span_area"], reverse=True):
        min_r, max_r, min_c, max_c = spec["bounds"]
        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            dr = rng.randint(row_lo, row_hi)
            dc = rng.randint(col_lo, col_hi)
            input_pixels = _translate_pixels(spec["input_pixels"], dr, dc)
            output_pixels = _translate_pixels(spec["output_pixels"], dr, dc)
            input_coords = {(r, c) for r, c, _v in input_pixels}
            output_coords = {(r, c) for r, c, _v in output_pixels}

            if input_coords & input_occ or output_coords & output_occ:
                continue
            if any(coord in input_blocked for coord in input_coords):
                continue
            if any(coord in output_blocked for coord in output_coords):
                continue

            placed.append({"input_pixels": input_pixels, "output_pixels": output_pixels})
            input_occ |= input_coords
            output_occ |= output_coords
            input_blocked |= _expand_coords(input_coords, h, w)
            output_blocked |= _expand_coords(output_coords, h, w)
            success = True
            break
        if not success:
            return None
    return placed


def _render_scene(h: int, w: int, objects: list[dict], key: str) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c, value in obj[key]:
            grid[r, c] = value
    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    envelope_colors = rng.sample(list(PALETTE), object_count)
    letter_palette = [color for color in PALETTE if color not in envelope_colors]
    if len(letter_palette) < 2:
        return None

    specs: list[dict] = []
    for envelope_color in envelope_colors:
        opening = rng.choice(OPENINGS)
        spec = _make_local_object(envelope_color, opening, letter_palette, rng)
        if spec is None:
            return None
        specs.append(spec)

    max_span_h = max(spec["bounds"][1] - spec["bounds"][0] + 1 for spec in specs)
    max_span_w = max(spec["bounds"][3] - spec["bounds"][2] + 1 for spec in specs)
    min_h = max(GRID_MIN, max_span_h + 2, 8 + object_count * 5)
    min_w = max(GRID_MIN, max_span_w + 2, 8 + object_count * 5)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)
    placed = _place_objects(specs, h, w, rng)
    if placed is None:
        return None

    inp = _render_scene(h, w, placed, "input_pixels")
    expected = _render_scene(h, w, placed, "output_pixels")
    if np.array_equal(inp, expected):
        return None

    counts = np.bincount(inp.ravel(), minlength=10)
    if int(counts[BG]) <= int(np.max(counts[1:])):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATION_ATTEMPTS} attempts")
