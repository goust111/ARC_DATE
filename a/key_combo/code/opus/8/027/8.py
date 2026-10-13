from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

FRAME_COLORS = (BLUE, RED, GREEN, ORANGE)

TEMPLATE_LIBRARY: dict[str, tuple[tuple[int, int], ...]] = {
    "domino": ((0, 0), (0, 1)),
    "line3": ((0, 0), (0, 1), (0, 2)),
    "ell3": ((0, 0), (1, 0), (1, 1)),
}

MAX_ATTEMPTS = 300


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                if connectivity == 4:
                    nbrs = neighbors4(rr, cc)
                else:
                    nbrs = (
                        (rr + dr, cc + dc)
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if (dr, dc) != (0, 0)
                    )
                for nr, nc in nbrs:
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    out = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
    for r, c in coords:
        out[r, c] = True
    return out


def canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    best = None
    for variant in dihedral_variants(mask):
        key = (variant.tobytes(), tuple(map(int, variant.shape)))
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    base = np.asarray(mask, dtype=bool)
    for k in range(4):
        rot = np.rot90(base, k)
        for variant in (rot, np.fliplr(rot)):
            key = (variant.tobytes(), tuple(map(int, variant.shape)))
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def shape_key_from_cells(cells: tuple[tuple[int, int], ...]) -> tuple[bytes, tuple[int, int]]:
    return canonical_shape_key(mask_from_coords(normalize_coords(cells)))


def component_mask_key(grid: np.ndarray, coords: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return canonical_shape_key(mask)


def is_frame_component(grid: np.ndarray, coords: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    expected_area = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected_area:
        return False
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    inside = patch[1:-1, 1:-1]
    if inside.size == 0:
        return False
    return not np.any(inside == color)


def extract_frame_info(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in FRAME_COLORS:
        for coords in components_mask(grid == color, connectivity=4):
            if not is_frame_component(grid, coords, color):
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            grey_comps = components_mask(inner == GREY, connectivity=4)
            assert len(grey_comps) == 1, "each frame must contain exactly one grey template"
            rel = grey_comps[0]
            tr0, tc0, tr1, tc1 = bbox_coords(rel)
            template_mask = np.zeros((tr1 - tr0 + 1, tc1 - tc0 + 1), dtype=bool)
            abs_template: list[tuple[int, int]] = []
            for rr, cc in rel:
                template_mask[int(rr) - tr0, int(cc) - tc0] = True
                abs_template.append((int(r0 + 1 + rr), int(c0 + 1 + cc)))
            frames.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "template_mask": template_mask,
                    "shape_key": canonical_shape_key(template_mask),
                    "template_cells": tuple(sorted(abs_template)),
                }
            )
    frames.sort(key=lambda fr: fr["bbox"])
    return frames


def extract_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for color in FRAME_COLORS:
        for coords in components_mask(grid == color, connectivity=4):
            if is_frame_component(grid, coords, color):
                continue
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "shape_key": component_mask_key(grid, coords),
                }
            )
    sprites.sort(key=lambda sp: bbox_coords(sp["coords"]))
    return sprites


def classify_sprites(frames: list[dict], sprites: list[dict]) -> tuple[dict[int, int], dict[int, int], dict[int, int], list[int]]:
    true_counts = {int(fr["color"]): 0 for fr in frames}
    color_only_counts = {int(fr["color"]): 0 for fr in frames}
    shape_only_counts = {int(fr["color"]): 0 for fr in frames}
    match_cardinalities: list[int] = []
    for sp in sprites:
        matches = []
        for fr in frames:
            same_color = int(sp["color"]) == int(fr["color"])
            same_shape = sp["shape_key"] == fr["shape_key"]
            if same_color:
                color_only_counts[int(fr["color"])] += 1
            if same_shape:
                shape_only_counts[int(fr["color"])] += 1
            if same_color and same_shape:
                true_counts[int(fr["color"])] += 1
                matches.append(int(fr["color"]))
        match_cardinalities.append(len(matches))
    return true_counts, color_only_counts, shape_only_counts, match_cardinalities


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each frame's embedded grey template to count matching same-color sprites and show the tally as dots.

    Input structure:
    - The background is black.
    - Two or three hollow rectangular frames of distinct colors appear on the grid.
    - Each frame contains one small grey template inside its upper-left interior, while small monochrome sprites of frame colors are scattered outside the frames.

    Rule:
    - Detect every colored hollow frame and crop its interior grey template to a bbox-normalized binary mask.
    - For each outside sprite, assign it to a frame only when both its color matches the frame border color and its bbox-normalized mask matches the frame template up to rotation or reflection.
    - Erase all outside sprites and erase the grey template inside each frame.
    - In each frame interior, draw the assigned-count as border-colored dots on the interior center row, starting from the right and stepping left by two columns.

    Key insight:
    - The grey shape inside the frame is not decoration; it is the shape key that disambiguates same-color sprites before the tally is rendered.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = extract_frame_info(g)
    sprites = extract_sprites(g)
    counts, _color_only, _shape_only, _card = classify_sprites(frames, sprites)

    for sp in sprites:
        for r, c in sp["coords"]:
            out[int(r), int(c)] = BLACK

    for fr in frames:
        for r, c in fr["template_cells"]:
            out[int(r), int(c)] = BLACK

        r0, c0, r1, c1 = fr["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        n = int(counts[int(fr["color"])])
        max_dots = (inner_w + 1) // 2
        assert n <= max_dots, "frame interior too narrow for requested dots"
        mid_r = r0 + 1 + (inner_h - 1) // 2
        for i in range(n):
            cc = c1 - 1 - 2 * i
            out[mid_r, cc] = int(fr["color"])

    return out


def _random_variant_cells(rng: np.random.Generator, shape_name: str) -> tuple[tuple[int, int], ...]:
    base = mask_from_coords(TEMPLATE_LIBRARY[shape_name])
    variants = dihedral_variants(base)
    chosen = variants[int(rng.integers(len(variants)))]
    cells = tuple(sorted(map(tuple, np.argwhere(chosen))))
    return normalize_coords(cells)


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_frames(
    rng: np.random.Generator,
    h: int,
    w: int,
    frame_specs: list[dict],
) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    for fr in frame_specs:
        fh = int(fr["outer_h"])
        fw = int(fr["outer_w"])
        for _ in range(200):
            r0 = int(rng.integers(1, h - fh))
            c0 = int(rng.integers(1, w - fw))
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if all(_rects_separated(bbox, other, gap=2) for other in placed):
                fr["bbox"] = bbox
                placed.append(bbox)
                break
        else:
            return False
    return True


def _mark_padded(occ: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = occ.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_shape(
    rng: np.random.Generator,
    occ: np.ndarray,
    h: int,
    w: int,
    cells: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    sh = max(rows) + 1
    sw = max(cols) + 1
    positions = [
        (r0, c0)
        for r0 in range(1, h - sh)
        for c0 in range(1, w - sw)
    ]
    rng.shuffle(positions)
    for r0, c0 in positions:
        bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r0 + sh)
        cc1 = min(w - 1, c0 + sw)
        if np.any(occ[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue
        _mark_padded(occ, bbox, pad=1)
        return tuple(sorted((int(r0 + r), int(c0 + c)) for r, c in cells))
    return None


def _build_expected_output(inp: np.ndarray, frame_specs: list[dict], sprite_specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for sp in sprite_specs:
        for r, c in sp["placed_cells"]:
            out[int(r), int(c)] = BLACK
    for fr in frame_specs:
        for r, c in fr["template_cells"]:
            out[int(r), int(c)] = BLACK
        r0, c0, r1, c1 = fr["bbox"]
        inner_h = r1 - r0 - 1
        mid_r = r0 + 1 + (inner_h - 1) // 2
        for i in range(int(fr["true_count"])):
            out[mid_r, c1 - 1 - 2 * i] = int(fr["color"])
    return out


def _validate_instance(inp: np.ndarray, frame_specs: list[dict], sprite_specs: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    parsed_frames = extract_frame_info(inp)
    parsed_sprites = extract_sprites(inp)
    if len(parsed_frames) != len(frame_specs):
        return False
    if len(parsed_sprites) != len(sprite_specs):
        return False

    meta_by_color = {int(fr["color"]): fr for fr in frame_specs}
    for fr in parsed_frames:
        color = int(fr["color"])
        if color not in meta_by_color:
            return False
        if fr["shape_key"] != meta_by_color[color]["shape_key"]:
            return False

    true_counts, color_only_counts, shape_only_counts, match_cardinalities = classify_sprites(parsed_frames, parsed_sprites)

    for fr in frame_specs:
        color = int(fr["color"])
        if int(true_counts[color]) != int(fr["true_count"]):
            return False
        if int(color_only_counts[color]) <= int(true_counts[color]):
            return False
        if int(shape_only_counts[color]) <= int(true_counts[color]):
            return False

    genuine_by_cell = {
        tuple(sorted(map(tuple, sp["placed_cells"]))): bool(sp["is_genuine"])
        for sp in sprite_specs
    }
    for sp, card in zip(parsed_sprites, match_cardinalities):
        key = tuple(sorted(map(tuple, sp["coords"])))
        expected_genuine = genuine_by_cell.get(key)
        if expected_genuine is None:
            return False
        if expected_genuine and card != 1:
            return False
        if (not expected_genuine) and card != 0:
            return False

    expected = _build_expected_output(inp, frame_specs, sprite_specs)
    actual = main(inp)
    if not np.array_equal(expected, actual):
        return False

    for fr in frame_specs:
        r0, c0, r1, c1 = fr["bbox"]
        inner = actual[r0 + 1 : r1, c0 + 1 : c1]
        color = int(fr["color"])
        true_count = int(fr["true_count"])
        mid_r = (r1 - r0 - 2 - 1) // 2
        expected_inner = np.zeros_like(inner)
        for i in range(true_count):
            expected_inner[mid_r, inner.shape[1] - 1 - 2 * i] = color
        if not np.array_equal(inner, expected_inner):
            return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 25))
    w = int(rng.integers(18, 25))

    n_frames = int(rng.choice([2, 3]))
    shape_names = list(rng.choice(list(TEMPLATE_LIBRARY.keys()), size=n_frames, replace=False))
    colors = list(rng.choice(FRAME_COLORS, size=n_frames, replace=False))
    rng.shuffle(shape_names)
    rng.shuffle(colors)

    frame_specs: list[dict] = []
    counts = [int(rng.integers(1, 4)) for _ in range(n_frames)]
    if max(counts) == 1:
        counts[int(rng.integers(n_frames))] = 2

    for shape_name, color, true_count in zip(shape_names, colors, counts):
        outer_w_min = max(7, 2 * true_count + 2)
        outer_w = int(rng.integers(outer_w_min, 10))
        outer_h = int(rng.integers(6, 9))
        max_dots = ((outer_w - 2) + 1) // 2
        if true_count > max_dots:
            return None
        frame_specs.append(
            {
                "shape_name": shape_name,
                "shape_key": shape_key_from_cells(TEMPLATE_LIBRARY[shape_name]),
                "color": int(color),
                "true_count": int(true_count),
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )

    if not _place_frames(rng, h, w, frame_specs):
        return None

    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    for fr in frame_specs:
        r0, c0, r1, c1 = fr["bbox"]
        color = int(fr["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        _mark_padded(occ, fr["bbox"], pad=1)

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        variant_cells = _random_variant_cells(rng, fr["shape_name"])
        sh = max(r for r, _ in variant_cells) + 1
        sw = max(c for _, c in variant_cells) + 1
        if sh > inner_h or sw > inner_w:
            return None
        max_off_r = min(1, inner_h - sh)
        max_off_c = min(1, inner_w - sw)
        off_r = int(rng.integers(max_off_r + 1))
        off_c = int(rng.integers(max_off_c + 1))
        placed_template = tuple(sorted((r0 + 1 + off_r + r, c0 + 1 + off_c + c) for r, c in variant_cells))
        fr["template_cells"] = placed_template
        for r, c in placed_template:
            grid[int(r), int(c)] = GREY

    sprite_specs: list[dict] = []
    for fr in frame_specs:
        for _ in range(int(fr["true_count"])):
            sprite_specs.append(
                {
                    "shape_name": fr["shape_name"],
                    "color": int(fr["color"]),
                    "is_genuine": True,
                }
            )

    for i, fr in enumerate(frame_specs):
        wrong_color = int(frame_specs[(i + 1) % n_frames]["color"])
        sprite_specs.append(
            {
                "shape_name": fr["shape_name"],
                "color": wrong_color,
                "is_genuine": False,
            }
        )

    rng.shuffle(sprite_specs)

    for sp in sprite_specs:
        cells = _random_variant_cells(rng, sp["shape_name"])
        placed = _place_shape(rng, occ, h, w, cells)
        if placed is None:
            return None
        sp["placed_cells"] = placed
        for r, c in placed:
            grid[int(r), int(c)] = int(sp["color"])

    if not _validate_instance(grid, frame_specs, sprite_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
