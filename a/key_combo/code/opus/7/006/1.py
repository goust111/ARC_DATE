from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
SEP = 5
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp.sort()
            comps.append(comp)
    comps.sort(key=lambda comp: comp[0])
    return comps


def _crop_key(crop: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, crop.shape)), tuple(int(x) for x in crop.flatten())


def _non_separator_spans(length: int, blocked: set[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx in range(length):
        if idx in blocked:
            if start is not None:
                spans.append((start, idx - 1))
                start = None
            continue
        if start is None:
            start = idx
    if start is not None:
        spans.append((start, length - 1))
    return spans


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    sep_rows = {r for r in range(grid.shape[0]) if np.all(grid[r] == SEP)}
    sep_cols = {c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)}
    row_spans = _non_separator_spans(grid.shape[0], sep_rows)
    col_spans = _non_separator_spans(grid.shape[1], sep_cols)
    assert len(row_spans) == 2 and len(col_spans) == 2

    panels: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            assert not np.any(panel == SEP)
            panels.append(panel)
    shapes = {tuple(map(int, panel.shape)) for panel in panels}
    assert len(shapes) == 1
    return panels


def _overlay_panels(panels: list[np.ndarray]) -> np.ndarray:
    h, w = panels[0].shape
    canvas = np.full((h, w), BG, dtype=int)
    for panel in panels:
        assert tuple(map(int, panel.shape)) == (h, w)
        mask = panel != BG
        conflict = mask & (canvas != BG) & (canvas != panel)
        assert not np.any(conflict)
        canvas[mask] = panel[mask]
    return canvas


def _find_repeated_crop(canvas: np.ndarray) -> np.ndarray:
    entries = []
    for comp in _components(canvas != BG):
        box = _bbox(comp)
        crop = _crop_bbox(canvas, box)
        entries.append({"coords": comp, "bbox": box, "crop": crop})

    assert len(entries) >= 3
    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = defaultdict(list)
    for entry in entries:
        groups[_crop_key(entry["crop"])].append(entry)

    repeated = [group for group in groups.values() if len(group) == 2]
    singles = [group for group in groups.values() if len(group) == 1]
    assert len(repeated) == 1
    assert len(singles) + 1 == len(groups)
    assert sum(len(group) for group in groups.values()) == len(entries)

    chosen = sorted(repeated[0], key=lambda entry: (entry["bbox"][0], entry["bbox"][1]))[0]
    return chosen["crop"].copy()


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Overlay sparse panel marks, then extract the only emergent connected pattern whose tight crop appears twice.

    Input structure:
    - A grey separator row and a grey separator column split the grid into four equal rectangular panels.
    - Each panel has black background plus 3-5 sparse colored pixels from the set {blue, red, green, yellow}.
    - Inside any single panel, foreground pixels are pairwise non-4-connected, so no complete shape exists before overlay.

    Rule:
    - Remove the grey separators and read the four same-sized panels in row-major order.
    - Transparently overlay every non-black panel pixel onto one black canvas at the same relative coordinates.
    - Find all 4-connected foreground components in the overlaid canvas, tight-crop each component while keeping black cells inside its bbox, and compare those crops exactly.
    - Return the unique crop that occurs exactly twice among the overlaid components.

    Key insight:
    - The target shape is invisible in every single panel; only the transparent overlay creates full connected components, and the answer is chosen by duplicate counting at that emergent level.
    """
    grid = np.asarray(input_grid, dtype=int)
    panels = _extract_panels(grid)
    canvas = _overlay_panels(panels)
    return _find_repeated_crop(canvas)


BASE_PATTERNS = [
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
]


def _pattern_variants() -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for base in BASE_PATTERNS:
        for flip in (False, True):
            work = np.fliplr(base) if flip else base
            for k in range(4):
                variant = np.rot90(work, k)
                key = _crop_key(variant)
                variants[key] = variant.copy()
    return list(variants.values())


PATTERN_VARIANTS = _pattern_variants()


def _random_colored_pattern() -> np.ndarray:
    variant = random.choice(PATTERN_VARIANTS)
    out = np.zeros_like(variant, dtype=int)
    cells = np.argwhere(variant != 0)
    for r, c in cells:
        out[int(r), int(c)] = random.choice(PALETTE)
    return out


def _pattern_area(pattern: np.ndarray) -> int:
    return int(np.count_nonzero(pattern))


def _can_place(occupied: np.ndarray, pattern: np.ndarray, r0: int, c0: int) -> bool:
    h, w = pattern.shape
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    mask = pattern != BG
    for rr in range(h):
        for cc in range(w):
            if not mask[rr, cc]:
                continue
            gr = r0 + rr
            gc = c0 + cc
            if occupied[gr, gc]:
                return False
            for nr, nc in _neighbors4(gr, gc, occupied.shape):
                if occupied[nr, nc]:
                    return False
    return True


def _place_pattern(canvas: np.ndarray, occupied: np.ndarray, pattern: np.ndarray, r0: int, c0: int) -> None:
    h, w = pattern.shape
    mask = pattern != BG
    for rr in range(h):
        for cc in range(w):
            if mask[rr, cc]:
                canvas[r0 + rr, c0 + cc] = int(pattern[rr, cc])
                occupied[r0 + rr, c0 + cc] = True


def _class_index(r: int, c: int) -> int:
    return (r % 2) * 2 + (c % 2)


def _panel_counts(canvas: np.ndarray) -> list[int]:
    counts = [0, 0, 0, 0]
    for r, c in np.argwhere(canvas != BG):
        counts[_class_index(int(r), int(c))] += 1
    return counts


def _split_into_panels(canvas: np.ndarray) -> list[np.ndarray]:
    panels = [np.full_like(canvas, BG) for _ in range(4)]
    for r, c in np.argwhere(canvas != BG):
        idx = _class_index(int(r), int(c))
        panels[idx][int(r), int(c)] = int(canvas[int(r), int(c)])
    return panels


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    h, w = panels[0].shape
    grid = np.full((2 * h + 1, 2 * w + 1), SEP, dtype=int)
    grid[:h, :w] = panels[0]
    grid[:h, w + 1 :] = panels[1]
    grid[h + 1 :, :w] = panels[2]
    grid[h + 1 :, w + 1 :] = panels[3]
    return grid


def _isolated_within_panel(panel: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(panel != BG)]
    for r, c in coords:
        for nr, nc in _neighbors4(r, c, panel.shape):
            if panel[nr, nc] != BG:
                return False
    return True


def _build_overlay_canvas() -> np.ndarray | None:
    h = random.randint(7, 10)
    w = random.randint(7, 11)

    repeated = _random_colored_pattern()
    unique1 = _random_colored_pattern()
    unique2 = _random_colored_pattern()

    patterns = [repeated, repeated.copy(), unique1, unique2]
    if not 13 <= sum(_pattern_area(pattern) for pattern in patterns) <= 18:
        return None

    seen_keys = [_crop_key(pattern) for pattern in patterns]
    if len(set(seen_keys)) != 3:
        return None

    order = list(range(len(patterns)))
    order.sort(key=lambda idx: _pattern_area(patterns[idx]), reverse=True)

    for _ in range(180):
        canvas = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        ok = True
        for idx in order:
            pattern = patterns[idx]
            ph, pw = pattern.shape
            options = []
            for r0 in range(h - ph + 1):
                for c0 in range(w - pw + 1):
                    if _can_place(occupied, pattern, r0, c0):
                        options.append((r0, c0))
            if not options:
                ok = False
                break
            r0, c0 = random.choice(options)
            _place_pattern(canvas, occupied, pattern, r0, c0)
        if not ok:
            continue
        counts = _panel_counts(canvas)
        if all(3 <= count <= 5 for count in counts):
            return canvas
    return None


def _validate_semantics(grid: np.ndarray) -> np.ndarray:
    panels = _extract_panels(grid)
    assert len(panels) == 4
    for panel in panels:
        count = int(np.count_nonzero(panel != BG))
        assert 3 <= count <= 5
        assert _isolated_within_panel(panel)
        assert set(np.unique(panel)).issubset({BG, *PALETTE})

    canvas = _overlay_panels(panels)
    entries = []
    for comp in _components(canvas != BG):
        box = _bbox(comp)
        crop = _crop_bbox(canvas, box)
        entries.append({"coords": comp, "bbox": box, "crop": crop})

    assert len(entries) >= 3
    for entry in entries:
        assert len(entry["coords"]) >= 2
        panel_ids = {_class_index(r, c) for r, c in entry["coords"]}
        assert len(panel_ids) >= 2

    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = defaultdict(list)
    for entry in entries:
        groups[_crop_key(entry["crop"])].append(entry)

    repeated_groups = [group for group in groups.values() if len(group) == 2]
    assert len(repeated_groups) == 1
    assert all(len(group) == 1 for group in groups.values() if group not in repeated_groups)

    expected = sorted(repeated_groups[0], key=lambda entry: (entry["bbox"][0], entry["bbox"][1]))[0]["crop"]
    actual = main(grid)
    assert np.array_equal(actual, expected)
    return expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        canvas = _build_overlay_canvas()
        if canvas is None:
            continue
        panels = _split_into_panels(canvas)
        grid = _assemble_grid(panels)
        try:
            _validate_semantics(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
