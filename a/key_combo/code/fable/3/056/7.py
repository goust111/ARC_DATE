from __future__ import annotations

import random

import numpy as np


GLYPH_POOL = [
    np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
]

BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    trimmed = _trim_mask(np.asarray(mask, dtype=bool))
    flat = tuple(int(v) for v in trimmed.ravel())
    return int(trimmed.shape[0]), int(trimmed.shape[1]), flat


def _is_connected(mask: np.ndarray) -> bool:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _coords_to_sprite(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    sprite = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return sprite


def _fragment_signature_sets(glyph_mask: np.ndarray) -> set[tuple[int, int, tuple[int, ...]]]:
    sigs: set[tuple[int, int, tuple[int, ...]]] = set()
    for r0 in range(4):
        for r1 in range(r0, 4):
            for c0 in range(4):
                for c1 in range(c0, 4):
                    fragment = glyph_mask[r0 : r1 + 1, c0 : c1 + 1]
                    if not np.any(fragment):
                        continue
                    trimmed = _trim_mask(fragment)
                    if _is_connected(trimmed):
                        sigs.add(_mask_signature(trimmed))
    return sigs


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, np.ndarray]]]:
    occupied_rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert occupied_rows.size >= 5
    label_row = int(occupied_rows[-1])
    legend_top = label_row - 4
    assert legend_top >= 0

    entries: list[tuple[int, np.ndarray, int]] = []
    for c in np.flatnonzero(grid[label_row] != bg):
        c = int(c)
        if c < 4:
            continue
        color = int(grid[label_row, c])
        glyph_mask = np.asarray(grid[legend_top : legend_top + 4, c - 4 : c] == color, dtype=bool)
        if np.any(glyph_mask):
            entries.append((c, glyph_mask, color))

    assert entries
    entries.sort(key=lambda item: item[0])
    return legend_top, [(color, glyph_mask) for _c, glyph_mask, color in entries]


def _build_histogram(colors: list[int], counts: list[int], bg: int = BG) -> np.ndarray:
    height = max(counts)
    assert height > 0
    out = np.full((height, len(colors)), bg, dtype=int)
    for c, (color, count) in enumerate(zip(colors, counts)):
        if count:
            out[height - count :, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode clipped monochrome glyph instances via the legend and output their counts as a color histogram.

    Input structure:
    - A bottom legend band contains several 4x4 sample glyphs, each painted in its own code color,
      with a same-color label pixel immediately below-right of the glyph.
    - Above the legend, the main area contains separated instances of those glyphs, all drawn in one
      shared ink color; some instances may be cropped by the canvas edge.
    - A monochrome distractor shape may also appear in the main area and should be ignored.

    Rule:
    - Read the legend from left to right to recover the ordered list of full glyph shapes and their code colors.
    - For each connected monochrome object in the main area, compare its visible fragment against every
      edge-cropped fragment that any legend glyph could produce; count it only when exactly one glyph matches.
    - Return a new grid whose width equals the number of legend glyphs and whose kth column is filled
      bottom-up with the kth code color exactly as many times as that glyph was counted.

    Key insight:
    - The legend is used in reverse: shape identifies the glyph, glyph identifies the output color, and
      even a clipped edge fragment is enough as long as it uniquely matches one legend template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_top, legend_entries = _extract_legend(g, bg)
    colors = [color for color, _glyph in legend_entries]
    glyphs = [glyph for _color, glyph in legend_entries]
    signature_sets = [_fragment_signature_sets(glyph) for glyph in glyphs]

    counts = [0] * len(glyphs)
    scene_mask = np.asarray(g[:legend_top] != bg, dtype=bool)
    for coords in _find_components(scene_mask):
        sprite = _coords_to_sprite(coords)
        sig = _mask_signature(sprite)
        matches = [i for i, sigs in enumerate(signature_sets) if sig in sigs]
        if len(matches) == 1:
            counts[matches[0]] += 1

    return _build_histogram(colors, counts, bg=bg)


def _candidate_fragments(
    glyphs: list[np.ndarray], glyph_index: int, main_h: int, width: int
) -> list[dict[str, object]]:
    signature_sets = [_fragment_signature_sets(glyph) for glyph in glyphs]
    glyph = glyphs[glyph_index]
    by_coords: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}
    for top in range(-3, main_h - 3):
        for left in range(-3, width):
            coords = []
            for r, c in np.argwhere(glyph):
                rr = top + int(r)
                cc = left + int(c)
                if 0 <= rr < main_h and 0 <= cc < width:
                    coords.append((rr, cc))
            if not coords:
                continue
            arr = np.array(coords, dtype=int)
            sprite = _coords_to_sprite(arr)
            if not _is_connected(sprite):
                continue
            sig = _mask_signature(sprite)
            matches = [i for i, sigs in enumerate(signature_sets) if sig in sigs]
            if matches != [glyph_index]:
                continue
            coords_key = tuple(sorted(coords))
            clipped = top < 0 or left < 0 or left + 4 > width
            existing = by_coords.get(coords_key)
            if existing is None:
                by_coords[coords_key] = {"coords": coords_key, "clipped": clipped}
            elif clipped:
                existing["clipped"] = True
    candidates = list(by_coords.values())
    random.shuffle(candidates)
    return candidates


def _fits(coords: tuple[tuple[int, int], ...], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        r0 = max(0, r - 1)
        r1 = min(h - 1, r + 1)
        c0 = max(0, c - 1)
        c1 = min(w - 1, c + 1)
        if np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return True


def _mark(coords: tuple[tuple[int, int], ...], occupied: np.ndarray) -> None:
    for r, c in coords:
        occupied[r, c] = True


def _random_connected_shape() -> np.ndarray | None:
    target = random.randint(3, 6)
    cells = {(random.randrange(4), random.randrange(4))}
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        random.shuffle(neighbors)
        placed = False
        for nr, nc in neighbors:
            if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in cells:
                trial = cells | {(nr, nc)}
                arr = np.zeros((4, 4), dtype=bool)
                for rr, cc in trial:
                    arr[rr, cc] = True
                if _trim_mask(arr).shape[0] <= 4 and _trim_mask(arr).shape[1] <= 4:
                    cells = trial
                    placed = True
                    break
        if not placed:
            return None

    arr = np.zeros((4, 4), dtype=bool)
    for r, c in cells:
        arr[r, c] = True
    trimmed = _trim_mask(arr)
    return trimmed if _is_connected(trimmed) else None


def _place_distractor(
    scene: np.ndarray, occupied: np.ndarray, signature_union: set[tuple[int, int, tuple[int, ...]]], ink: int
) -> bool:
    for _ in range(80):
        shape = _random_connected_shape()
        if shape is None or _mask_signature(shape) in signature_union:
            continue
        h, w = shape.shape
        top = random.randint(0, scene.shape[0] - h)
        left = random.randint(0, scene.shape[1] - w)
        coords = tuple(
            sorted(
                (top + int(r), left + int(c))
                for r, c in np.argwhere(shape)
            )
        )
        if not _fits(coords, occupied):
            continue
        _mark(coords, occupied)
        for r, c in coords:
            scene[r, c] = ink
        return True
    return False


def _gen_counts(n_glyphs: int) -> list[int]:
    for _ in range(100):
        counts = [random.randint(0, 3) for _ in range(n_glyphs)]
        total = sum(counts)
        if 3 <= total <= 8 and sum(count > 0 for count in counts) >= 2:
            return counts
    counts = [0] * n_glyphs
    for _ in range(random.randint(3, 6)):
        counts[random.randrange(n_glyphs)] += 1
    return counts


def _legend_layout(n_glyphs: int) -> tuple[list[int], int]:
    starts = [0]
    for _ in range(1, n_glyphs):
        starts.append(starts[-1] + 5 + random.randint(1, 2))
    legend_width = starts[-1] + 5
    return starts, legend_width


def _gen_attempt() -> np.ndarray | None:
    n_glyphs = random.randint(3, 4)
    glyphs = random.sample(GLYPH_POOL, n_glyphs)
    counts = _gen_counts(n_glyphs)
    starts, legend_width = _legend_layout(n_glyphs)

    width_min = max(legend_width, 18)
    if width_min > 30:
        return None
    width = random.randint(width_min, 30)
    main_h = random.randint(9, 18)
    total_h = main_h + 5

    palette = random.sample(COLOR_POOL, n_glyphs + 1)
    legend_colors = palette[:n_glyphs]
    ink = palette[-1]

    candidates_by_glyph = [_candidate_fragments(glyphs, i, main_h, width) for i in range(n_glyphs)]
    if any(counts[i] > 0 and not candidates_by_glyph[i] for i in range(n_glyphs)):
        return None

    clipped_sources = [i for i, count in enumerate(counts) if count > 0 and any(c["clipped"] for c in candidates_by_glyph[i])]
    if not clipped_sources:
        return None
    forced_clipped = random.choice(clipped_sources)

    scene = np.full((main_h, width), BG, dtype=int)
    occupied = np.zeros((main_h, width), dtype=bool)

    placements: list[dict[str, object]] = [{"glyph": forced_clipped, "must_clipped": True}]
    remaining = counts[:]
    remaining[forced_clipped] -= 1
    for glyph_index, count in enumerate(remaining):
        placements.extend({"glyph": glyph_index, "must_clipped": False} for _ in range(count))
    random.shuffle(placements[1:])
    placements[1:] = sorted(
        placements[1:],
        key=lambda item: len(candidates_by_glyph[int(item["glyph"])]),
    )

    saw_full = False
    saw_clipped = False
    for request in placements:
        glyph_index = int(request["glyph"])
        must_clipped = bool(request["must_clipped"])
        options = candidates_by_glyph[glyph_index]
        if must_clipped:
            options = [option for option in options if bool(option["clipped"])]
        else:
            prefer_full = [option for option in options if not bool(option["clipped"])]
            if prefer_full and not saw_full:
                options = prefer_full + [option for option in options if bool(option["clipped"])]

        placed = False
        for option in options:
            coords = option["coords"]
            if _fits(coords, occupied):
                _mark(coords, occupied)
                for r, c in coords:
                    scene[r, c] = ink
                saw_clipped |= bool(option["clipped"])
                saw_full |= not bool(option["clipped"])
                placed = True
                break
        if not placed:
            return None

    if not saw_clipped or not saw_full:
        return None

    signature_union: set[tuple[int, int, tuple[int, ...]]] = set()
    for glyph in glyphs:
        signature_union |= _fragment_signature_sets(glyph)
    if random.random() < 0.65 and not _place_distractor(scene, occupied, signature_union, ink):
        return None

    grid = np.full((total_h, width), BG, dtype=int)
    grid[:main_h, :] = scene

    legend_top = main_h
    if width > legend_width:
        shift = random.randint(0, width - legend_width)
        starts = [start + shift for start in starts]
    for start, glyph_mask, color in zip(starts, glyphs, legend_colors):
        glyph_patch = np.asarray(glyph_mask, dtype=bool)
        grid[legend_top : legend_top + 4, start : start + 4][glyph_patch] = color
        grid[legend_top + 4, start + 4] = color

    expected = _build_histogram(legend_colors, counts, bg=BG)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
