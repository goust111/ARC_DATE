from __future__ import annotations

import random

import numpy as np


BG = 0


def _parse_mapping_band(top_row: np.ndarray) -> dict[int, int]:
    """Read the top-row key/value stripe as adjacent non-black pairs."""
    top = np.asarray(top_row, dtype=int)
    mapping: dict[int, int] = {}
    c = 0
    width = top.shape[0]

    while c < width:
        if int(top[c]) == BG:
            c += 1
            continue

        assert c + 1 < width, "mapping band ended with an unmatched key pixel"
        key = int(top[c])
        value = int(top[c + 1])
        assert value != BG, "mapping pairs must contain two colored pixels"
        assert key not in mapping, "mapping keys must be unique"
        mapping[key] = value
        c += 2
        if c < width:
            assert int(top[c]) == BG, "adjacent mapping pairs must be separated by black"

    assert mapping, "expected a non-empty mapping band"
    return mapping


def _color_pairs(body: np.ndarray) -> dict[int, np.ndarray]:
    """Return the two coordinates for each non-black body color."""
    colors = [int(color) for color in np.unique(body) if int(color) != BG]
    pairs: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(body == color)
        assert coords.shape == (2, 2), "each body color must appear exactly twice"
        pairs[color] = coords
    assert pairs, "expected at least one paired body color"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill pair-defined rectangles using the color remapping encoded in the top-row legend.

    Input structure:
    - The background is black.
    - The top row is a mapping band made of adjacent colored key/value pairs, with black separators between neighboring pairs.
    - In the remaining rows, each non-black color appears exactly twice; those two pixels are markers for one rectangle and their color is a mapping key.

    Rule:
    - Parse the top row as key→value color mappings, where each pair contributes left_color → right_color.
    - For every paired body color C, take the axis-aligned bounding rectangle of its two marker pixels.
    - Look up the mapped fill color C' for C, and recolor every black cell inside that rectangle to C'.
    - Keep all original colored marker pixels unchanged, then remove the top mapping row from the final output.

    Key insight:
    - The marker color does not determine the fill directly; it first selects a hidden value color from the edge legend, and that mapped color is used to fill the pair's rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2, "expected a top mapping row plus a body"

    mapping = _parse_mapping_band(g[0])
    body = g[1:].copy()
    pairs = _color_pairs(body)

    for color, coords in pairs.items():
        assert color in mapping, "every body color must be a mapping key"
        fill_color = mapping[color]
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        patch = body[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = fill_color

    return body


# --------------- generator ---------------

GRID_H_RANGE = (7, 18)
GRID_W_RANGE = (10, 24)
KEY_COUNT_RANGE = (2, 4)
OBJECT_KIND_CHOICES = ("horizontal", "vertical", "rectangle")
TYPE_COVER_PROB = 0.6

H_LINE_W_RANGE = (3, 8)
V_LINE_H_RANGE = (3, 8)
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 7)

PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400


def _derangement(values: list[int], rng: random.Random) -> list[int] | None:
    for _ in range(50):
        perm = values[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(values, perm)):
            return perm
    return None


def _sample_mapping(rng: random.Random, key_count: int) -> dict[int, int] | None:
    keys = rng.sample(list(range(1, 10)), key_count)
    values = _derangement(keys, rng)
    if values is None:
        return None
    return dict(zip(keys, values))


def _build_mapping_row(rng: random.Random, mapping: dict[int, int], width: int) -> np.ndarray | None:
    order = list(mapping)
    rng.shuffle(order)
    band: list[int] = []
    for idx, key in enumerate(order):
        band.extend([key, mapping[key]])
        if idx != len(order) - 1:
            band.append(BG)
    if len(band) > width:
        return None

    row = np.zeros(width, dtype=int)
    start = rng.randint(0, width - len(band))
    row[start : start + len(band)] = np.array(band, dtype=int)
    return row


def _sample_object_kinds(rng: random.Random, object_count: int) -> list[str]:
    if object_count >= 3 and rng.random() < TYPE_COVER_PROB:
        kinds = list(OBJECT_KIND_CHOICES)
        while len(kinds) < object_count:
            kinds.append(rng.choice(OBJECT_KIND_CHOICES))
        rng.shuffle(kinds)
        return kinds
    return [rng.choice(OBJECT_KIND_CHOICES) for _ in range(object_count)]


def _sample_box(rng: random.Random, kind: str, body_h: int, body_w: int) -> tuple[int, int] | None:
    if kind == "horizontal":
        max_w = min(H_LINE_W_RANGE[1], body_w)
        if max_w < H_LINE_W_RANGE[0]:
            return None
        return 1, rng.randint(H_LINE_W_RANGE[0], max_w)

    if kind == "vertical":
        max_h = min(V_LINE_H_RANGE[1], body_h)
        if max_h < V_LINE_H_RANGE[0]:
            return None
        return rng.randint(V_LINE_H_RANGE[0], max_h), 1

    max_h = min(RECT_H_RANGE[1], body_h)
    max_w = min(RECT_W_RANGE[1], body_w)
    if max_h < RECT_H_RANGE[0] or max_w < RECT_W_RANGE[0]:
        return None
    return rng.randint(RECT_H_RANGE[0], max_h), rng.randint(RECT_W_RANGE[0], max_w)


def _marker_positions(kind: str, r0: int, c0: int, h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    if kind == "horizontal":
        return [(r0, c0), (r0, c0 + w - 1)]
    if kind == "vertical":
        return [(r0, c0), (r0 + h - 1, c0)]
    if rng.random() < 0.5:
        return [(r0, c0), (r0 + h - 1, c0 + w - 1)]
    return [(r0, c0 + w - 1), (r0 + h - 1, c0)]


def _place_objects(
    rng: random.Random,
    body_h: int,
    body_w: int,
    keys: list[int],
    kinds: list[str],
) -> list[dict] | None:
    occupied = np.zeros((body_h, body_w), dtype=bool)
    specs: list[dict] = []

    samples: list[dict] = []
    for key, kind in zip(keys, kinds):
        box = _sample_box(rng, kind, body_h, body_w)
        if box is None:
            return None
        h, w = box
        samples.append({"key": key, "kind": kind, "h": h, "w": w})
    samples.sort(key=lambda spec: spec["h"] * spec["w"], reverse=True)

    for spec in samples:
        h = spec["h"]
        w = spec["w"]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(0, body_h - h)
            c0 = rng.randint(0, body_w - w)
            r_lo = max(0, r0 - 1)
            r_hi = min(body_h, r0 + h + 1)
            c_lo = max(0, c0 - 1)
            c_hi = min(body_w, c0 + w + 1)
            if occupied[r_lo:r_hi, c_lo:c_hi].any():
                continue

            spec = dict(spec)
            spec["r0"] = r0
            spec["c0"] = c0
            spec["markers"] = _marker_positions(spec["kind"], r0, c0, h, w, rng)
            specs.append(spec)
            occupied[r_lo:r_hi, c_lo:c_hi] = True
            placed = True
            break
        if not placed:
            return None

    return specs


def _render_example(
    width: int,
    mapping_row: np.ndarray,
    mapping: dict[int, int],
    body_h: int,
    specs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    body_in = np.zeros((body_h, width), dtype=int)
    body_out = np.zeros((body_h, width), dtype=int)

    for spec in specs:
        color = spec["key"]
        for r, c in spec["markers"]:
            body_in[r, c] = color
            body_out[r, c] = color

    for spec in specs:
        color = spec["key"]
        fill = mapping[color]
        r0 = spec["r0"]
        c0 = spec["c0"]
        h = spec["h"]
        w = spec["w"]
        patch = body_out[r0 : r0 + h, c0 : c0 + w]
        patch[patch == BG] = fill

    inp = np.vstack([mapping_row, body_in])
    return inp, body_out


def _valid_input(mapping: dict[int, int], body: np.ndarray, specs: list[dict]) -> bool:
    if not mapping:
        return False
    body_colors = [int(color) for color in np.unique(body) if int(color) != BG]
    if len(body_colors) != len(specs):
        return False
    for color in body_colors:
        if int(np.count_nonzero(body == color)) != 2 or color not in mapping:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    body_h = height - 1

    key_count = rng.randint(*KEY_COUNT_RANGE)
    mapping = _sample_mapping(rng, key_count)
    if mapping is None:
        return None
    mapping_row = _build_mapping_row(rng, mapping, width)
    if mapping_row is None:
        return None

    object_count = rng.randint(2, key_count)
    used_keys = rng.sample(list(mapping), object_count)
    kinds = _sample_object_kinds(rng, object_count)
    specs = _place_objects(rng, body_h, width, used_keys, kinds)
    if specs is None:
        return None

    inp, expected = _render_example(width, mapping_row, mapping, body_h, specs)
    if not _valid_input(mapping, inp[1:], specs):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
