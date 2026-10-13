from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox_coords, components_mask
except Exception:
    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        if connectivity == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
        comps: list[np.ndarray] = []
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
                    for dr, dc in steps:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.asarray(cells, dtype=int))
        return comps


BLACK = 0
DIVIDER = 5
UNKNOWN = 8
COLOR_CHOICES = (1, 2, 3, 4, 6, 7, 9)
OPS = ("or", "and", "diff", "xor")
MAX_GENERATION_ATTEMPTS = 400


def _parse_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 11, "grid is too small for three panels and two dividers"
    assert (w - 2) % 3 == 0, "three equal-width panels plus two dividers are required"
    pw = (w - 2) // 3
    d0 = pw
    d1 = 2 * pw + 1
    assert np.all(g[:, d0] == DIVIDER) and np.all(g[:, d1] == DIVIDER), "divider columns must be solid grey"
    left = g[:, :pw]
    middle = g[:, d0 + 1 : d1]
    right = g[:, d1 + 1 :]
    assert left.shape == middle.shape == right.shape
    return left, middle, right


def _single_foreground_color(panel: np.ndarray, forbidden: set[int]) -> int:
    colors = {int(v) for v in np.unique(panel) if int(v) != BLACK}
    colors -= forbidden
    assert len(colors) == 1, "panel must contain exactly one foreground color"
    return next(iter(colors))


def _result_color(panel: np.ndarray) -> int:
    colors = {int(v) for v in np.unique(panel) if int(v) not in {BLACK, UNKNOWN}}
    assert len(colors) == 1, "third panel must reveal exactly one output foreground color"
    return next(iter(colors))


def _apply_op(name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if name == "or":
        return a | b
    if name == "and":
        return a & b
    if name == "diff":
        return a & (~b)
    assert name == "xor"
    return a ^ b


def _consistent_ops(a: np.ndarray, b: np.ndarray, partial: np.ndarray, out_color: int) -> list[str]:
    known_true = partial == out_color
    known_false = partial == BLACK
    known = known_true | known_false
    assert np.any(known), "third panel must reveal at least one cell"
    assert not np.any((partial != UNKNOWN) & (partial != BLACK) & (partial != out_color)), "unexpected reveal color"

    consistent: list[str] = []
    for name in OPS:
        result = _apply_op(name, a, b)
        if np.all(result[known_true]) and not np.any(result[known_false]):
            consistent.append(name)
    return consistent


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden boolean operation from a few revealed cells and complete the damaged result panel.

    Input structure:
    - Three equal-size panels are arranged side by side, separated by two solid grey divider columns.
    - The first two panels contain black background plus one foreground color each; presence versus absence is the boolean signal.
    - The third panel is mostly color 8 unknown-mask cells, with only a few revealed black or result-color cells.

    Rule:
    - Align the three panels cell by cell and treat the first two as boolean masks.
    - Test the candidate operations union, intersection, first-minus-second, and symmetric difference against the revealed cells in the third panel.
    - Keep the unique operation consistent with every reveal, then fill the entire third panel with black for false cells and the third-panel foreground color for true cells.

    Key insight:
    - The sparse visible cells in the damaged panel are evidence for which boolean truth table is active, not decorative fragments to copy locally.
    """
    left, middle, partial = _parse_panels(grid)
    _single_foreground_color(left, {DIVIDER, UNKNOWN})
    _single_foreground_color(middle, {DIVIDER, UNKNOWN})
    out_color = _result_color(partial)

    a = left != BLACK
    b = middle != BLACK
    candidates = _consistent_ops(a, b, partial, out_color)
    assert len(candidates) == 1, "revealed cells must determine a unique boolean operation"
    result = _apply_op(candidates[0], a, b)
    return np.where(result, out_color, BLACK).astype(int)


def _random_rectilinear_mask(h: int, w: int, rng: np.random.Generator, seed: tuple[int, int] | None = None) -> np.ndarray:
    for _ in range(60):
        mask = np.zeros((h, w), dtype=bool)
        n_rects = int(rng.integers(2, 5))

        if seed is None:
            rh = int(rng.integers(2, min(h, max(3, h - 1)) + 1))
            rw = int(rng.integers(2, min(w, max(3, w - 1)) + 1))
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
        else:
            sr, sc = seed
            rh = int(rng.integers(2, min(h, max(3, h - 1)) + 1))
            rw = int(rng.integers(2, min(w, max(3, w - 1)) + 1))
            r_lo = max(0, sr - rh + 1)
            r_hi = min(sr, h - rh)
            c_lo = max(0, sc - rw + 1)
            c_hi = min(sc, w - rw)
            r0 = int(rng.integers(r_lo, r_hi + 1))
            c0 = int(rng.integers(c_lo, c_hi + 1))
        mask[r0 : r0 + rh, c0 : c0 + rw] = True

        for _ in range(n_rects - 1):
            cells = np.argwhere(mask)
            ar, ac = map(int, cells[int(rng.integers(len(cells)))])
            rh = int(rng.integers(1, min(h, max(2, h // 2 + 1)) + 1))
            rw = int(rng.integers(1, min(w, max(2, w // 2 + 1)) + 1))
            r_lo = max(0, ar - rh + 1)
            r_hi = min(ar, h - rh)
            c_lo = max(0, ac - rw + 1)
            c_hi = min(ac, w - rw)
            r0 = int(rng.integers(r_lo, r_hi + 1))
            c0 = int(rng.integers(c_lo, c_hi + 1))
            mask[r0 : r0 + rh, c0 : c0 + rw] = True

        if _mask_is_reasonable(mask):
            return mask
    raise RuntimeError("failed to build a mask")


def _mask_is_reasonable(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    if area < max(6, (h * w) // 6) or area > int(0.72 * h * w):
        return False
    comps = components_mask(mask, connectivity=4)
    if len(comps) != 1:
        return False
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    return True


def _build_mask_pair(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(120):
        a = _random_rectilinear_mask(h, w, rng)
        overlap_seeds = np.argwhere(a)
        seed = None
        if len(overlap_seeds) and rng.random() < 0.85:
            seed = tuple(map(int, overlap_seeds[int(rng.integers(len(overlap_seeds)))]))
        b = _random_rectilinear_mask(h, w, rng, seed=seed)

        cats = _truth_categories(a, b)
        if all(len(cats[name]) > 0 for name in ("00", "10", "01", "11")):
            return a, b
    raise RuntimeError("failed to build a non-degenerate mask pair")


def _truth_categories(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "00": np.argwhere((~a) & (~b)),
        "10": np.argwhere(a & (~b)),
        "01": np.argwhere((~a) & b),
        "11": np.argwhere(a & b),
    }


def _required_evidence(op_name: str) -> list[tuple[str, bool]]:
    if op_name == "or":
        return [("01", True), ("11", True)]
    if op_name == "and":
        return [("01", False), ("11", True)]
    if op_name == "diff":
        return [("10", True), ("01", False)]
    assert op_name == "xor"
    return [("01", True), ("11", False)]


def _manhattan(p: tuple[int, int], q: tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def _pick_evidence_cells(cats: dict[str, np.ndarray], spec: list[tuple[str, bool]]) -> list[tuple[int, int, bool]]:
    first_cells = [tuple(map(int, x)) for x in cats[spec[0][0]]]
    second_cells = [tuple(map(int, x)) for x in cats[spec[1][0]]]
    best: tuple[int, tuple[int, int], tuple[int, int]] | None = None
    for p in first_cells:
        for q in second_cells:
            score = _manhattan(p, q)
            if p == q:
                continue
            if best is None or score > best[0]:
                best = (score, p, q)
    assert best is not None, "evidence categories must offer distinct cells"
    return [
        (best[1][0], best[1][1], spec[0][1]),
        (best[2][0], best[2][1], spec[1][1]),
    ]


def _render_input_from_parts(
    a: np.ndarray,
    b: np.ndarray,
    a_color: int,
    b_color: int,
    out_color: int,
    op_name: str,
    evidence: list[tuple[int, int, bool]],
) -> np.ndarray:
    h, w = a.shape
    left = np.where(a, a_color, BLACK).astype(int)
    middle = np.where(b, b_color, BLACK).astype(int)
    partial = np.full((h, w), UNKNOWN, dtype=int)
    for r, c, is_true in evidence:
        partial[r, c] = out_color if is_true else BLACK
    return np.concatenate(
        [left, np.full((h, 1), DIVIDER, dtype=int), middle, np.full((h, 1), DIVIDER, dtype=int), partial],
        axis=1,
    )


def _nondegenerate_outputs(a: np.ndarray, b: np.ndarray) -> bool:
    rendered = []
    for name in OPS:
        rendered.append(_apply_op(name, a, b).tobytes())
    return len(set(rendered)) == len(rendered)


def _validate_semantics(grid: np.ndarray) -> None:
    left, middle, partial = _parse_panels(grid)
    a_color = _single_foreground_color(left, {DIVIDER, UNKNOWN})
    b_color = _single_foreground_color(middle, {DIVIDER, UNKNOWN})
    out_color = _result_color(partial)
    assert len({a_color, b_color, out_color, DIVIDER, UNKNOWN, BLACK}) == 6, "all semantic colors must be distinct"

    a = left != BLACK
    b = middle != BLACK
    cats = _truth_categories(a, b)
    assert all(len(cats[name]) > 0 for name in cats), "all four truth categories must appear"
    assert _nondegenerate_outputs(a, b), "different boolean operations must produce different outputs"

    candidates = _consistent_ops(a, b, partial, out_color)
    assert len(candidates) == 1, "reveals must identify a unique operation"

    reveal_coords = [(int(r), int(c)) for r, c in np.argwhere(partial != UNKNOWN)]
    assert len(reveal_coords) == 2, "this generator should expose exactly two indispensable clues"
    for r, c in reveal_coords:
        reduced = partial.copy()
        reduced[r, c] = UNKNOWN
        assert len(_consistent_ops(a, b, reduced, out_color)) > 1, "every clue must be necessary"

    expected = np.where(_apply_op(candidates[0], a, b), out_color, BLACK).astype(int)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the inferred boolean result"


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(6, 11))
    w = int(rng.integers(6, 10))
    a, b = _build_mask_pair(h, w, rng)
    if not _nondegenerate_outputs(a, b):
        return None

    op_name = OPS[int(rng.integers(len(OPS)))]
    spec = _required_evidence(op_name)
    evidence = _pick_evidence_cells(_truth_categories(a, b), spec)

    colors = list(rng.choice(np.asarray(COLOR_CHOICES, dtype=int), size=3, replace=False))
    grid = _render_input_from_parts(a, b, int(colors[0]), int(colors[1]), int(colors[2]), op_name, evidence)

    try:
        _validate_semantics(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-boolean ARC instance")
