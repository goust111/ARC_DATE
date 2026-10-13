import random

import numpy as np


RIGHT_BG = 0
SEPARATOR_COLOR = 5
VISIBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_HEIGHT_RANGE = (6, 18)
GEN_LEFT_WIDTH_RANGE = (6, 14)
GEN_OCCLUDER_RETRIES = 60
GEN_TOTAL_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _checker_grid(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where(((rows + cols) & 1) == 0, even_color, odd_color).astype(int)


def _component_penalties(left: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int]:
    colors = {int(left[r, c]) for r, c in comp}
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    multi_color_penalty = max(0, len(colors) - 1)
    rect_penalty = area - len(comp)
    return multi_color_penalty, rect_penalty


def _infer_checker(left: np.ndarray) -> np.ndarray:
    colors = [int(color) for color in np.unique(left)]
    best_key = None
    best_expected = None

    for even_color in colors:
        for odd_color in colors:
            if even_color == odd_color:
                continue
            expected = _checker_grid(left.shape, even_color, odd_color)
            mismatch = left != expected
            comps = _components(mismatch)
            multi_color_penalty = 0
            rect_penalty = 0
            for comp in comps:
                comp_multi, comp_rect = _component_penalties(left, comp)
                multi_color_penalty += comp_multi
                rect_penalty += comp_rect

            key = (
                multi_color_penalty > 0,
                rect_penalty > 0,
                int(mismatch.sum()),
                multi_color_penalty,
                rect_penalty,
                len(comps),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_expected = expected

    assert best_expected is not None, "expected at least two colors on the left side"
    return best_expected


def _sorted_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    comps = _components(mask)
    return sorted(comps, key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp), len(comp)))


def _stamp_border(
    out: np.ndarray,
    comp: list[tuple[int, int]],
    color: int,
    right_col0: int,
    right_width: int,
) -> None:
    h = out.shape[0]
    comp_set = set(comp)
    border: set[tuple[int, int]] = set()
    for r, c in comp:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < right_width and (nr, nc) not in comp_set:
                    border.add((nr, nc))
    for r, c in border:
        out[r, right_col0 + c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the checkerboard hidden under left-side occluders and stamp those hidden pixels into the right panel with colored outlines.

    Input structure:
    - The grid is split into equal-width left and right panels by a one-column gray separator.
    - The left panel contains a full 2-color checkerboard with several solid rectangular occluders painted on top.
    - The right panel starts as a blank black canvas, while occluder colors differ from both checker colors and from the separator.

    Rule:
    - Infer the left checkerboard's two colors and phase from the visible cells.
    - Mark every left-panel cell that disagrees with that checkerboard as part of an occluder, then split those cells into 4-connected components.
    - Copy the hidden checkerboard colors under each occluder onto the same relative coordinates in the right panel.
    - Around each copied shape, draw a one-pixel outer border on the right panel by expanding the shape by one cell in all 8 directions and coloring only the outside ring with the occluder's color.

    Key insight:
    - The non-checker cells are not the answer themselves: they define stamp masks, while the missing checkerboard pixels become the fill and the occluder color is reused only for the outline.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w % 2 == 1, "grid width must be odd so the separator can sit in the middle"

    sep_col = w // 2
    left = g[:, :sep_col]
    right_col0 = sep_col + 1
    right_width = w - right_col0
    assert right_width == left.shape[1], "left and right panels must be equally wide"

    expected = _infer_checker(left)
    mismatch = left != expected
    occluders = _sorted_components(mismatch)

    out = g.copy()
    for comp in occluders:
        for r, c in comp:
            out[r, right_col0 + c] = int(expected[r, c])

    for comp in occluders:
        color = int(left[comp[0][0], comp[0][1]])
        _stamp_border(out, comp, color, right_col0, right_width)

    return out


def _render_expected(
    inp: np.ndarray,
    checker: np.ndarray,
    occluders: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    left_width = checker.shape[1]
    right_col0 = left_width + 1

    for r0, c0, rh, rw, _color in occluders:
        for r in range(r0, r0 + rh):
            for c in range(c0, c0 + rw):
                out[r, right_col0 + c] = int(checker[r, c])

    for r0, c0, rh, rw, color in occluders:
        comp = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]
        _stamp_border(out, comp, color, right_col0, left_width)

    return out


def _visible_parity_ok(h: int, w: int, occluded: np.ndarray) -> bool:
    rows, cols = np.indices((h, w))
    even_visible = int(((((rows + cols) & 1) == 0) & ~occluded).sum())
    odd_visible = int(((((rows + cols) & 1) == 1) & ~occluded).sum())
    return even_visible >= 4 and odd_visible >= 4


def _candidate_positions(
    blocked: np.ndarray,
    rh: int,
    rw: int,
    *,
    edge_only: bool,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - rh + 1):
        for c0 in range(w - rw + 1):
            if blocked[max(0, r0 - 1) : min(h, r0 + rh + 1), max(0, c0 - 1) : min(w, c0 + rw + 1)].any():
                continue
            if edge_only and not (r0 == 0 or c0 == 0 or r0 + rh == h or c0 + rw == w):
                continue
            positions.append((r0, c0))
    return positions


def _choose_rect_dims(h: int, w: int) -> tuple[int, int]:
    mode = random.random()
    if mode < 0.3:
        return 1, random.randint(1, min(5, w))
    if mode < 0.6:
        return random.randint(1, min(5, h)), 1
    return random.randint(2, min(5, h)), random.randint(2, min(5, w))


def _gen_occluders(h: int, w: int, colors: list[int]) -> list[tuple[int, int, int, int, int]] | None:
    area = h * w
    max_count = min(4, len(colors), max(1, area // 24 + 1))
    target_count = random.randint(1, max_count)
    blocked = np.zeros((h, w), dtype=bool)
    occluded = np.zeros((h, w), dtype=bool)
    occluders: list[tuple[int, int, int, int, int]] = []
    want_edge = random.random() < 0.45
    edge_used = False

    for idx in range(target_count):
        placed = False
        for _ in range(GEN_OCCLUDER_RETRIES):
            rh, rw = _choose_rect_dims(h, w)
            edge_only = want_edge and not edge_used and idx == 0
            positions = _candidate_positions(blocked, rh, rw, edge_only=edge_only)
            if not positions and edge_only:
                positions = _candidate_positions(blocked, rh, rw, edge_only=False)
            if not positions:
                continue
            r0, c0 = random.choice(positions)
            trial = occluded.copy()
            trial[r0 : r0 + rh, c0 : c0 + rw] = True
            if int(trial.sum()) > area // 2:
                continue
            if not _visible_parity_ok(h, w, trial):
                continue
            color = colors[len(occluders)]
            occluders.append((r0, c0, rh, rw, color))
            blocked[max(0, r0 - 1) : min(h, r0 + rh + 1), max(0, c0 - 1) : min(w, c0 + rw + 1)] = True
            occluded = trial
            edge_used = edge_used or (r0 == 0 or c0 == 0 or r0 + rh == h or c0 + rw == w)
            placed = True
            break
        if not placed:
            return None

    return occluders if occluders else None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    left_width = random.randint(*GEN_LEFT_WIDTH_RANGE)
    if 2 * left_width + 1 > 30:
        return None

    checker_colors = random.sample(VISIBLE_COLORS, 2)
    phase = random.randint(0, 1)
    even_color = checker_colors[phase]
    odd_color = checker_colors[1 - phase]
    checker = _checker_grid((h, left_width), even_color, odd_color)

    occluder_palette = [color for color in VISIBLE_COLORS if color not in checker_colors]
    random.shuffle(occluder_palette)
    occluders = _gen_occluders(h, left_width, occluder_palette)
    if occluders is None:
        return None

    inp = np.full((h, 2 * left_width + 1), RIGHT_BG, dtype=int)
    inp[:, :left_width] = checker
    inp[:, left_width] = SEPARATOR_COLOR

    for r0, c0, rh, rw, color in occluders:
        inp[r0 : r0 + rh, c0 : c0 + rw] = int(color)

    out = _render_expected(inp, checker, occluders)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_TOTAL_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checker-stamp ARC sample")
