from __future__ import annotations

from typing import Iterable

import numpy as np


BG = 0
FOREGROUND_COLORS = tuple(range(1, 10))
MISSING_TO_OCCUPIED = {
    "tl": ((0, 1), (1, 0), (1, 1)),
    "tr": ((0, 0), (1, 0), (1, 1)),
    "bl": ((0, 0), (0, 1), (1, 1)),
    "br": ((0, 0), (0, 1), (1, 0)),
}
MISSING_TO_CELL = {
    "tl": (0, 0),
    "tr": (0, 1),
    "bl": (1, 0),
    "br": (1, 1),
}
OPPOSITE_MISSING = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    pieces: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pieces.append(np.array(coords, dtype=int))
    return pieces


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _describe_piece(coords: np.ndarray) -> dict[str, object]:
    assert coords.shape == (3, 2)
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 == 1 and c1 - c0 == 1
    occupied = {(int(r - r0), int(c - c0)) for r, c in coords}
    assert len(occupied) == 3
    all_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing_cells = list(all_cells - occupied)
    assert len(missing_cells) == 1
    missing = {
        (0, 0): "tl",
        (0, 1): "tr",
        (1, 0): "bl",
        (1, 1): "br",
    }[missing_cells[0]]
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "key": (r0, c0),
        "missing": missing,
    }


def _seam_kind(missing_a: str, missing_b: str) -> str:
    pair = frozenset((missing_a, missing_b))
    if pair == frozenset(("tl", "br")):
        return "\\"
    if pair == frozenset(("tr", "bl")):
        return "/"
    raise AssertionError("same-color pieces must be opposite missing-corner Ls")


def _square_layout(first_missing: str, second_missing: str) -> np.ndarray:
    seam = _seam_kind(first_missing, second_missing)
    first = np.zeros((2, 2), dtype=int)
    second = np.zeros((2, 2), dtype=int)
    for rr, cc in MISSING_TO_OCCUPIED[first_missing]:
        first[rr, cc] = 1
    for rr, cc in MISSING_TO_OCCUPIED[second_missing]:
        second[rr, cc] = 2
    layout = first + second
    assert bool((layout > 0).all())
    if seam == "\\":
        layout = np.flipud(layout)
    else:
        layout = np.fliplr(layout)
    return layout


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color L triominoes into 2x2 squares at the upper-left fragment position.

    The internal seam orientation is inferred from the two opposite missing corners:
    {"tl","br"} gives a "\" seam and triggers a horizontal-axis flip;
    {"tr","bl"} gives a "/" seam and triggers a vertical-axis flip.
    The visible square is monochrome, but the seam-dependent flip is still applied
    to the abstract ownership layout before the final color fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = sorted(int(color) for color in np.unique(g) if int(color) != BG)
    out = g.copy()

    for color in colors:
        pieces = [_describe_piece(coords) for coords in _components(g == color)]
        assert len(pieces) == 2
        pieces.sort(key=lambda piece: piece["key"])
        first, second = pieces
        assert first["missing"] == OPPOSITE_MISSING[second["missing"]]

        for piece in pieces:
            coords = piece["coords"]
            out[coords[:, 0], coords[:, 1]] = BG

        layout = _square_layout(str(first["missing"]), str(second["missing"]))
        r0, c0, _r1, _c1 = first["bbox"]
        out[r0 : r0 + 2, c0 : c0 + 2][layout > 0] = color

    return out


def _boxes_separated(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    return (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _piece_box(top: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = top
    return int(r), int(c), int(r + 1), int(c + 1)


def _paint_piece(grid: np.ndarray, color: int, top: tuple[int, int], missing: str) -> None:
    r0, c0 = map(int, top)
    for dr, dc in MISSING_TO_OCCUPIED[missing]:
        grid[r0 + dr, c0 + dc] = int(color)


def _balanced_seams(count: int, rng: np.random.Generator) -> list[str]:
    slash_count = count // 2
    if count % 2 == 1 and bool(rng.integers(0, 2)):
        slash_count += 1
    seams = ["/"] * slash_count + ["\\"] * (count - slash_count)
    rng.shuffle(seams)
    return seams


def _missing_pair_for_seam(seam: str, rng: np.random.Generator) -> tuple[str, str]:
    if seam == "\\":
        options = (("tl", "br"), ("br", "tl"))
    else:
        options = (("tr", "bl"), ("bl", "tr"))
    return options[int(rng.integers(0, len(options)))]


def _candidate_positions(
    shape: tuple[int, int],
    used_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    positions: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            box = _piece_box((r, c))
            if all(_boxes_separated(box, other, gap=1) for other in used_boxes):
                positions.append((r, c))
    return positions


def _pair_far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) >= 5


def _place_pair(
    shape: tuple[int, int],
    used_boxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    legal = _candidate_positions(shape, used_boxes)
    if len(legal) < 2:
        return None

    order = list(rng.permutation(len(legal)))
    for idx in order[: min(len(order), 60)]:
        first = legal[int(idx)]
        first_box = _piece_box(first)
        second_choices = [
            pos
            for pos in legal
            if pos != first
            and _boxes_separated(first_box, _piece_box(pos), gap=1)
            and _pair_far_enough(first, pos)
        ]
        if second_choices:
            second = second_choices[int(rng.integers(0, len(second_choices)))]
            return first, second
    return None


def _valid_output(grid: np.ndarray, out: np.ndarray, n_pairs: int) -> bool:
    if np.array_equal(grid, out):
        return False
    colors_in = {int(c) for c in np.unique(grid) if int(c) != BG}
    colors_out = {int(c) for c in np.unique(out) if int(c) != BG}
    if colors_in != colors_out:
        return False
    for color in colors_in:
        comps = _components(out == color)
        if len(comps) != 1:
            return False
        coords = comps[0]
        if coords.shape != (4, 2):
            return False
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 != 1 or c1 - c0 != 1:
            return False
    return len(colors_in) == n_pairs


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.integers(2, 6))
    h = int(rng.integers(8 + n_pairs, 13 + n_pairs))
    w = int(rng.integers(8 + n_pairs, 15 + n_pairs))
    colors = [int(c) for c in rng.choice(FOREGROUND_COLORS, size=n_pairs, replace=False)]
    seams = _balanced_seams(n_pairs, rng)

    grid = np.full((h, w), BG, dtype=int)
    used_boxes: list[tuple[int, int, int, int]] = []

    for color, seam in zip(colors, seams):
        placement = _place_pair(grid.shape, used_boxes, rng)
        if placement is None:
            return None
        first_top, second_top = placement
        missing_first, missing_second = _missing_pair_for_seam(seam, rng)
        _paint_piece(grid, color, first_top, missing_first)
        _paint_piece(grid, color, second_top, missing_second)
        used_boxes.append(_piece_box(first_top))
        used_boxes.append(_piece_box(second_top))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not _valid_output(grid, out, n_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
