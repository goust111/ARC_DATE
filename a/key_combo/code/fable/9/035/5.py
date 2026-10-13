from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _inside_bbox(b: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = b
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _card_border_cells(card: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = card["bbox"]
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _extract_cards(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    cards: list[dict] = []
    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 3, "each card needs two corners and one remnant cell"
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 1 and c1 - c0 >= 1, "card bbox must span both axes"
        border = _card_border_cells({"bbox": bbox})
        clues = {tuple(map(int, xy)) for xy in coords}
        assert clues <= border, "all clue pixels must lie on the reconstructed border"
        cards.append(
            {
                "color": int(color),
                "bbox": bbox,
                "clues": clues,
            }
        )
    assert cards, "expected at least one colored card"
    return cards


def _unique_top_order(colors: list[int], above: dict[int, set[int]]) -> list[int]:
    indeg = {color: 0 for color in colors}
    for src in colors:
        for dst in above[src]:
            indeg[dst] += 1

    available = sorted(color for color in colors if indeg[color] == 0)
    order: list[int] = []
    while available:
        assert len(available) == 1, "stack order is not uniquely determined"
        cur = available.pop()
        order.append(cur)
        for nxt in sorted(above[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)
        available.sort()

    assert len(order) == len(colors), "occlusion constraints must form an acyclic total order"
    return order


def _infer_top_order(cards: list[dict]) -> list[int]:
    colors = [card["color"] for card in cards]
    by_color = {card["color"]: card for card in cards}
    above = {color: set() for color in colors}

    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            a = cards[i]
            b = cards[j]
            overlap = _overlap_bbox(a["bbox"], b["bbox"])
            if overlap is None:
                continue

            a_visible = any(_inside_bbox(overlap, cell) for cell in a["clues"])
            b_visible = any(_inside_bbox(overlap, cell) for cell in b["clues"])
            assert a_visible ^ b_visible, "each overlapping pair must expose exactly one upper remnant"
            if a_visible:
                above[a["color"]].add(b["color"])
            else:
                above[b["color"]].add(a["color"])

    order = _unique_top_order(colors, above)

    # Every visible remnant cell that falls inside another card's area must agree with the solved order.
    rank = {color: idx for idx, color in enumerate(order)}
    for card in cards:
        for other in cards:
            if card["color"] == other["color"]:
                continue
            overlap = _overlap_bbox(card["bbox"], other["bbox"])
            if overlap is None:
                continue
            visible_here = any(_inside_bbox(overlap, cell) for cell in card["clues"])
            if visible_here:
                assert rank[card["color"]] < rank[other["color"]]

    return order


def _draw_card(canvas: np.ndarray, card: dict) -> None:
    r0, c0, r1, c1 = card["bbox"]
    color = int(card["color"])
    canvas[r0 : r1 + 1, c0 : c1 + 1] = 0
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the opaque-card stack order from sparse border remnants, then redraw the full stacked scene.

    Input structure
    - The grid has a black background and one color per card.
    - For each card, the visible clues are exactly two opposite corner pixels plus a short remnant segment on that card's border.
    - The full card is an opaque hollow rectangle: the border has the card color and the interior is black, so a higher card covers both borders and interiors below it.

    Rule
    - Group pixels by color and reconstruct each card as the axis-aligned rectangle spanning that color's clues.
    - For every overlapping pair of cards, whichever card still shows clue pixels inside the overlap must lie above the other there; use all such constraints to recover the unique top-to-bottom order.
    - Redraw the cards from bottom to top, letting each later card overwrite earlier content with its colored frame and black interior.

    Key insight
    - The sparse border remnants are depth evidence rather than decoration: a remnant visible inside another card's footprint proves its owner is on top at that overlap, so solving the task requires causal inversion first and forward restacking second.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert _get_bg(g) == 0, "background must be black"

    cards = _extract_cards(g)
    top_to_bottom = _infer_top_order(cards)
    by_color = {card["color"]: card for card in cards}

    out = np.zeros_like(g)
    for color in reversed(top_to_bottom):
        _draw_card(out, by_color[color])
    return out


# --------------- generator (att1) ---------------

CARD_COUNT_CHOICES = (3, 4)
COLOR_POOL = tuple(range(1, 10))
CARD_SIZE_RANGE = (7, 9)
OVERLAP_RANGE = (2, 3)
SLACK_RANGE = (0, 2)
REMNANT_MAX_LEN = 3
GEN_ATTEMPTS = 500


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _flip_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], flip_v: bool, flip_h: bool
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    if flip_v:
        r0, r1 = h - 1 - r1, h - 1 - r0
    if flip_h:
        c0, c1 = w - 1 - c1, w - 1 - c0
    return int(r0), int(c0), int(r1), int(c1)


def _flip_cells(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    shape: tuple[int, int],
    flip_v: bool,
    flip_h: bool,
) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for r, c in cells:
        if flip_v:
            r = h - 1 - r
        if flip_h:
            c = w - 1 - c
        out.append((int(r), int(c)))
    return out


def _segment_from_line(cells: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if not cells:
        return None
    length = random.randint(1, min(REMNANT_MAX_LEN, len(cells)))
    start = random.randint(0, len(cells) - length)
    return cells[start : start + length]


def _upper_remnant(card: dict, lower: dict) -> list[tuple[int, int]] | None:
    overlap = _overlap_bbox(card["bbox"], lower["bbox"])
    assert overlap is not None
    r0, c0, r1, c1 = overlap
    cr0, cc0, cr1, cc1 = card["bbox"]

    bottom_cells = [(cr1, c) for c in range(c0, c1 + 1) if c not in (cc0, cc1)]
    right_cells = [(r, cc1) for r in range(r0, r1 + 1) if r not in (cr0, cr1)]
    candidates = [cells for cells in (bottom_cells, right_cells) if cells]
    if not candidates:
        return None
    return _segment_from_line(random.choice(candidates))


def _bottom_remnant(card: dict) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = card["bbox"]
    bottom_cells = [(r1, c) for c in range(c0 + 1, c1)]
    right_cells = [(r, c1) for r in range(r0 + 1, r1)]
    candidates = [cells for cells in (bottom_cells, right_cells) if cells]
    if not candidates:
        return None
    return _segment_from_line(random.choice(candidates))


def _render_scene(shape: tuple[int, int], cards: list[dict], top_to_bottom: list[int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=int)
    by_color = {card["color"]: card for card in cards}
    for color in reversed(top_to_bottom):
        _draw_card(canvas, by_color[color])
    return canvas


def _build_input(shape: tuple[int, int], cards: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for card in cards:
        for r, c in card["clues"]:
            grid[r, c] = int(card["color"])
    return grid


def _validate_scene(inp: np.ndarray, out: np.ndarray, top_to_bottom: list[int]) -> bool:
    cards = _extract_cards(inp)
    if sorted(card["color"] for card in cards) != sorted(top_to_bottom):
        return False

    by_color = {card["color"]: card for card in cards}
    rank = {color: idx for idx, color in enumerate(top_to_bottom)}

    for card in cards:
        color = card["color"]
        clues = card["clues"]
        if len(clues) < 3:
            return False
        if not all(int(out[r, c]) == color for r, c in clues):
            return False

    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            a = cards[i]
            b = cards[j]
            overlap = _overlap_bbox(a["bbox"], b["bbox"])
            if overlap is None:
                continue
            a_visible = any(_inside_bbox(overlap, cell) for cell in a["clues"])
            b_visible = any(_inside_bbox(overlap, cell) for cell in b["clues"])
            if not (a_visible ^ b_visible):
                return False
            upper = a["color"] if a_visible else b["color"]
            lower = b["color"] if a_visible else a["color"]
            if rank[upper] >= rank[lower]:
                return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    n_cards = random.choice(CARD_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, n_cards)
    heights = [_randint(CARD_SIZE_RANGE) for _ in range(n_cards)]
    widths = [_randint(CARD_SIZE_RANGE) for _ in range(n_cards)]
    row_overlaps = [_randint(OVERLAP_RANGE) for _ in range(n_cards - 1)]
    col_overlaps = [_randint(OVERLAP_RANGE) for _ in range(n_cards - 1)]

    cards: list[dict] = []
    r0 = 0
    c0 = 0
    for i, color in enumerate(colors):
        h = heights[i]
        w = widths[i]
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        cards.append({"color": color, "bbox": bbox})
        if i < n_cards - 1:
            r0 += h - row_overlaps[i]
            c0 += w - col_overlaps[i]

    for i in range(n_cards):
        for j in range(i + 2, n_cards):
            if _overlap_bbox(cards[i]["bbox"], cards[j]["bbox"]) is not None:
                return None

    span_h = max(card["bbox"][2] for card in cards) + 1
    span_w = max(card["bbox"][3] for card in cards) + 1
    if span_h > 30 or span_w > 30:
        return None

    extra_h = random.randint(SLACK_RANGE[0], min(SLACK_RANGE[1], 30 - span_h))
    extra_w = random.randint(SLACK_RANGE[0], min(SLACK_RANGE[1], 30 - span_w))
    shape = (span_h + extra_h, span_w + extra_w)
    shift_r = random.randint(0, shape[0] - span_h)
    shift_c = random.randint(0, shape[1] - span_w)

    shifted_cards: list[dict] = []
    for card in cards:
        r0, c0, r1, c1 = card["bbox"]
        shifted_cards.append(
            {
                "color": card["color"],
                "bbox": (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c),
            }
        )

    for i in range(n_cards - 1):
        remnant = _upper_remnant(shifted_cards[i], shifted_cards[i + 1])
        if remnant is None:
            return None
        shifted_cards[i]["clues"] = {
            (shifted_cards[i]["bbox"][0], shifted_cards[i]["bbox"][3]),
            (shifted_cards[i]["bbox"][2], shifted_cards[i]["bbox"][1]),
            *remnant,
        }
    remnant = _bottom_remnant(shifted_cards[-1])
    if remnant is None:
        return None
    shifted_cards[-1]["clues"] = {
        (shifted_cards[-1]["bbox"][0], shifted_cards[-1]["bbox"][3]),
        (shifted_cards[-1]["bbox"][2], shifted_cards[-1]["bbox"][1]),
        *remnant,
    }

    flip_v = bool(random.getrandbits(1))
    flip_h = bool(random.getrandbits(1))
    final_cards: list[dict] = []
    for card in shifted_cards:
        final_cards.append(
            {
                "color": card["color"],
                "bbox": _flip_bbox(card["bbox"], shape, flip_v, flip_h),
                "clues": set(_flip_cells(card["clues"], shape, flip_v, flip_h)),
            }
        )

    top_to_bottom = [card["color"] for card in final_cards]
    out = _render_scene(shape, final_cards, top_to_bottom)
    inp = _build_input(shape, final_cards)

    if not _validate_scene(inp, out, top_to_bottom):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the remnant stack puzzle")
