import random

import numpy as np


BLACK = 0
FRAME_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 400
WINDOW_COUNT_CHOICES = (2, 3, 4)
REFERENCE_H_RANGE = (8, 14)
REFERENCE_W_RANGE = (8, 14)
WINDOW_DIM_MIN = 2
WINDOW_DIM_MAX = 6
PALETTE_SIZE_CHOICES = (4, 5)


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
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    frame = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        frame[r - r0, c - c0] = True
    if not np.all(frame[0, :]) or not np.all(frame[-1, :]):
        return False
    if not np.all(frame[:, 0]) or not np.all(frame[:, -1]):
        return False
    return not np.any(frame[1:-1, 1:-1])


def _find_windows(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    windows = []
    for comp in _components(grid == FRAME_COLOR):
        if _is_rectangular_frame(comp):
            windows.append(_bbox(comp))
    windows.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    assert windows, "expected at least one grey window frame"
    return windows


def _find_reference_bbox(
    grid: np.ndarray, windows: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    mask = grid != BLACK
    for r0, c0, r1, c1 in windows:
        mask[r0 : r1 + 1, c0 : c1 + 1] = False

    comps = _components(mask)
    assert len(comps) == 1, f"expected one reference component, found {len(comps)}"
    ref_box = _bbox(comps[0])
    r0, c0, r1, c1 = ref_box
    ref = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(ref != BLACK), "reference pattern must be a filled rectangle"
    assert int(ref.size) == len(comps[0]), "reference component must fill its bbox"
    return ref_box


def _candidate_matches(
    inner: np.ndarray, reference: np.ndarray
) -> list[tuple[int, int, int, int]]:
    ih, iw = inner.shape
    rh, rw = reference.shape
    matches = []
    for r0 in range(rh - ih + 1):
        for c0 in range(rw - iw + 1):
            sub = reference[r0 : r0 + ih, c0 : c0 + iw]
            diff = sub != inner
            if not np.any(diff):
                continue
            originals = np.unique(sub[diff])
            replacements = np.unique(inner[diff])
            if len(originals) != 1 or len(replacements) != 1:
                continue
            original = int(originals[0])
            replacement = int(replacements[0])
            if original == replacement:
                continue
            converted = sub.copy()
            converted[converted == original] = replacement
            if np.array_equal(converted, inner):
                matches.append((r0, c0, original, replacement))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each grey-framed window patch from the large reference pattern and recolor the frame with the replaced source color.

    Input structure:
    - The background is black.
    - One large filled rectangular reference pattern occupies the left side of the grid and uses several non-black colors.
    - Several smaller rectangular windows appear elsewhere; each window has a one-cell grey frame and contains a copied reference subregion.
    - Inside each window, exactly one original reference color has been replaced everywhere by a different color.

    Rule:
    - Detect every grey rectangular frame and treat its interior as a corrupted reference fragment.
    - Find the unique same-sized subregion of the large reference rectangle that matches after undoing one global color replacement.
    - Replace the window interior with the clean reference subregion.
    - Recolor the whole grey frame to the original source color that had been replaced inside that window.

    Key insight:
    - Solving requires matching each window spatially against the reference while allowing one color-wide substitution, then using the removed source color, not the replacement color, as the new frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    windows = _find_windows(g)
    ref_r0, ref_c0, ref_r1, ref_c1 = _find_reference_bbox(g, windows)
    reference = g[ref_r0 : ref_r1 + 1, ref_c0 : ref_c1 + 1]

    out = g.copy()
    for r0, c0, r1, c1 in windows:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        matches = _candidate_matches(inner, reference)
        assert len(matches) == 1, f"expected unique reference match, found {len(matches)}"
        rr, cc, original, _replacement = matches[0]
        clean = reference[rr : rr + inner.shape[0], cc : cc + inner.shape[1]]
        out[r0 : r1 + 1, c0 : c1 + 1] = original
        out[r0 + 1 : r1, c0 + 1 : c1] = clean

    return out


def _make_reference_pattern(
    rng: random.Random, height: int, width: int, palette: tuple[int, ...]
) -> np.ndarray:
    ref = np.array(
        [[rng.choice(palette) for _ in range(width)] for _ in range(height)],
        dtype=int,
    )

    for _ in range(rng.randint(2, 4)):
        color = rng.choice(palette)
        kind = rng.choice(("rect", "rows", "cols"))
        if kind == "rect":
            r0 = rng.randint(0, height - 2)
            r1 = rng.randint(r0 + 1, height - 1)
            c0 = rng.randint(0, width - 2)
            c1 = rng.randint(c0 + 1, width - 1)
            ref[r0 : r1 + 1, c0 : c1 + 1] = color
        elif kind == "rows":
            r0 = rng.randint(0, height - 1)
            r1 = rng.randint(r0, height - 1)
            ref[r0 : r1 + 1, :] = color
        else:
            c0 = rng.randint(0, width - 1)
            c1 = rng.randint(c0, width - 1)
            ref[:, c0 : c1 + 1] = color

    accent_low = max(2, (height * width) // 12)
    accent_high = max(accent_low, (height * width) // 6)
    for _ in range(rng.randint(accent_low, accent_high)):
        ref[rng.randrange(height), rng.randrange(width)] = rng.choice(palette)

    missing = [color for color in palette if not np.any(ref == color)]
    for color in missing:
        ref[rng.randrange(height), rng.randrange(width)] = color
    return ref


def _build_window_specs(
    rng: random.Random, reference: np.ndarray, ref_palette: tuple[int, ...], count: int
) -> list[dict[str, np.ndarray | int]] | None:
    ref_h, ref_w = reference.shape
    dims = [
        (h, w)
        for h in range(WINDOW_DIM_MIN, min(WINDOW_DIM_MAX, ref_h) + 1)
        for w in range(WINDOW_DIM_MIN, min(WINDOW_DIM_MAX, ref_w) + 1)
    ]
    replacement_pool = [color for color in COLOR_POOL if color not in ref_palette]
    if not dims or not replacement_pool:
        return None

    for _ in range(120):
        specs: list[dict[str, np.ndarray | int]] = []
        seen_windows: set[tuple[int, int, bytes, int]] = set()
        for _window_index in range(count):
            found = None
            for _ in range(180):
                inner_h, inner_w = rng.choice(dims)
                src_r = rng.randint(0, ref_h - inner_h)
                src_c = rng.randint(0, ref_w - inner_w)
                clean = reference[src_r : src_r + inner_h, src_c : src_c + inner_w]
                colors = [int(color) for color in np.unique(clean)]
                if len(colors) < 2:
                    continue
                original_choices = [color for color in colors if np.any(clean != color)]
                if not original_choices:
                    continue
                original = rng.choice(original_choices)
                replacement = rng.choice(replacement_pool)
                corrupted = clean.copy()
                corrupted[corrupted == original] = replacement
                matches = _candidate_matches(corrupted, reference)
                if len(matches) != 1:
                    continue
                match_r, match_c, match_original, match_replacement = matches[0]
                if (
                    match_r != src_r
                    or match_c != src_c
                    or match_original != original
                    or match_replacement != replacement
                ):
                    continue
                key = (inner_h, inner_w, corrupted.tobytes(), original)
                if key in seen_windows:
                    continue
                found = {
                    "clean": clean.copy(),
                    "corrupted": corrupted,
                    "original": original,
                }
                seen_windows.add(key)
                break
            if found is None:
                break
            specs.append(found)
        if len(specs) == count:
            return specs
    return None


def _place_windows(
    rng: random.Random,
    grid_shape: tuple[int, int],
    ref_box: tuple[int, int, int, int],
    specs: list[dict[str, np.ndarray | int]],
) -> list[tuple[dict[str, np.ndarray | int], int, int]] | None:
    height, width = grid_shape
    ref_r0, ref_c0, ref_r1, ref_c1 = ref_box
    min_col = ref_c1 + 3
    occupied = np.zeros((height, width), dtype=bool)
    occupied[max(0, ref_r0 - 1) : min(height, ref_r1 + 2), max(0, ref_c0 - 1) : min(width, ref_c1 + 2)] = True

    placements: list[tuple[dict[str, np.ndarray | int], int, int]] = []
    for spec in sorted(specs, key=lambda item: int(np.prod(item["corrupted"].shape)), reverse=True):
        corrupted = spec["corrupted"]
        inner_h, inner_w = corrupted.shape
        box_h = inner_h + 2
        box_w = inner_w + 2
        if min_col > width - box_w - 1:
            return None
        candidates = [
            (r0, c0)
            for r0 in range(1, height - box_h)
            for c0 in range(min_col, width - box_w)
        ]
        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            pad_r0 = r0 - 1
            pad_c0 = c0 - 1
            pad_r1 = r0 + box_h + 1
            pad_c1 = c0 + box_w + 1
            if occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
                continue
            occupied[pad_r0:pad_r1, pad_c0:pad_c1] = True
            chosen = (r0, c0)
            break
        if chosen is None:
            return None
        placements.append((spec, chosen[0], chosen[1]))
    return placements


def _expected_output(
    grid: np.ndarray, placements: list[tuple[dict[str, np.ndarray | int], int, int]]
) -> np.ndarray:
    out = grid.copy()
    for spec, r0, c0 in placements:
        clean = spec["clean"]
        color = int(spec["original"])
        inner_h, inner_w = clean.shape
        out[r0 : r0 + inner_h + 2, c0 : c0 + inner_w + 2] = color
        out[r0 + 1 : r0 + inner_h + 1, c0 + 1 : c0 + inner_w + 1] = clean
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    ref_h = rng.randint(*REFERENCE_H_RANGE)
    ref_w = rng.randint(*REFERENCE_W_RANGE)
    palette_size = rng.choice(PALETTE_SIZE_CHOICES)
    ref_palette = tuple(rng.sample(COLOR_POOL, palette_size))
    reference = _make_reference_pattern(rng, ref_h, ref_w, ref_palette)

    window_count = rng.choice(WINDOW_COUNT_CHOICES)
    specs = _build_window_specs(rng, reference, ref_palette, window_count)
    if specs is None:
        return None

    max_box_h = max(spec["corrupted"].shape[0] + 2 for spec in specs)
    max_box_w = max(spec["corrupted"].shape[1] + 2 for spec in specs)
    min_h = max(ref_h + 2, max_box_h + 2, 16)
    min_w = max(ref_w + max_box_w + 6, 20)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    max_ref_left = min(3, grid_w - ref_w - max_box_w - 4)
    if max_ref_left < 1:
        return None
    ref_top = rng.randint(1, grid_h - ref_h - 1)
    ref_left = rng.randint(1, max_ref_left)
    ref_box = (ref_top, ref_left, ref_top + ref_h - 1, ref_left + ref_w - 1)

    placements = _place_windows(rng, (grid_h, grid_w), ref_box, specs)
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[ref_top : ref_top + ref_h, ref_left : ref_left + ref_w] = reference

    for spec, r0, c0 in placements:
        corrupted = spec["corrupted"]
        inner_h, inner_w = corrupted.shape
        grid[r0 : r0 + inner_h + 2, c0 : c0 + inner_w + 2] = FRAME_COLOR
        grid[r0 + 1 : r0 + inner_h + 1, c0 + 1 : c0 + inner_w + 1] = corrupted

    expected = _expected_output(grid, placements)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
