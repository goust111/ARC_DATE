from collections import deque
import random

import numpy as np


_BG = 0
_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# Solver / generator shared assumptions.
_GAP_OPTIONS = (2, 4, 6)
_GEN_ATTEMPTS = 400

# Generator ranges.
_PAIR_COUNT_OPTIONS = (2, 3, 4)
_PAIR_COUNT_WEIGHTS = (2, 3, 2)
_GRID_MIN = 12
_GRID_MAX = 30
_OUTPUT_FILL_MAX = 0.42
_PADDING = 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _solid_rectangle(coords: np.ndarray) -> bool:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    probe = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    if probe[local[:, 0], local[:, 1]].any():
        return False
    probe[local[:, 0], local[:, 1]] = True
    return bool(probe.all())


def _merge_pair(comp_a: np.ndarray, comp_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows_a = np.unique(comp_a[:, 0])
    rows_b = np.unique(comp_b[:, 0])
    cols_a = np.unique(comp_a[:, 1])
    cols_b = np.unique(comp_b[:, 1])
    same_rows = np.array_equal(rows_a, rows_b)
    same_cols = np.array_equal(cols_a, cols_b)
    assert same_rows != same_cols, "pair must align on exactly one axis"

    if same_rows:
        left, right = sorted((comp_a, comp_b), key=lambda pts: int(pts[:, 1].min()))
        gap = None
        for r in rows_a:
            left_cols = left[left[:, 0] == r, 1]
            right_cols = right[right[:, 0] == r, 1]
            assert left_cols.size > 0 and right_cols.size > 0
            row_gap = int(right_cols.min() - left_cols.max() - 1)
            assert row_gap >= 0
            gap = row_gap if gap is None else gap
            assert row_gap == gap, "horizontal gap must be constant across rows"
        assert gap is not None and gap % 2 == 0, "generated pairs use even symmetric gaps"
        shift = gap // 2
        left = left.copy()
        right = right.copy()
        left[:, 1] += shift
        right[:, 1] -= shift
        return left, right

    top, bottom = sorted((comp_a, comp_b), key=lambda pts: int(pts[:, 0].min()))
    gap = None
    for c in cols_a:
        top_rows = top[top[:, 1] == c, 0]
        bottom_rows = bottom[bottom[:, 1] == c, 0]
        assert top_rows.size > 0 and bottom_rows.size > 0
        col_gap = int(bottom_rows.min() - top_rows.max() - 1)
        assert col_gap >= 0
        gap = col_gap if gap is None else gap
        assert col_gap == gap, "vertical gap must be constant across columns"
    assert gap is not None and gap % 2 == 0, "generated pairs use even symmetric gaps"
    shift = gap // 2
    top = top.copy()
    bottom = bottom.copy()
    top[:, 0] += shift
    bottom[:, 0] -= shift
    return top, bottom


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-colored fragment pair toward each other until they re-form a solid rectangle.

    Input structure:
    - The background is black.
    - Every non-black color appears in exactly two 4-connected fragments.
    - Within one color pair, the two fragments share either the same occupied rows or the same occupied columns, with a black gap between their facing edges.

    Rule:
    - Pair the two fragments of each color.
    - Use the shared occupied-row set to detect a horizontal slide, or the shared occupied-column set to detect a vertical slide.
    - Move the two fragments symmetrically along that axis until the constant black gap disappears, then paint the translated cells on a clean black canvas.

    Key insight:
    - Color identifies which two pieces belong together, while exact row/column alignment reveals the slide axis; after that, solving is pure translation until the complementary pieces touch and fill one rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != _BG):
        comps = _components(g == color)
        assert len(comps) == 2, "each foreground color must form exactly two fragments"
        moved_a, moved_b = _merge_pair(comps[0], comps[1])
        merged = np.vstack((moved_a, moved_b))
        assert _solid_rectangle(merged), "merged pair must be a filled rectangle"
        for coords in (moved_a, moved_b):
            rr = coords[:, 0]
            cc = coords[:, 1]
            assert np.all((0 <= rr) & (rr < g.shape[0]) & (0 <= cc) & (cc < g.shape[1]))
            assert not np.any(out[rr, cc] != _BG), "merged pairs must not overlap"
            out[rr, cc] = color
    return out


def _sample_profile(length: int, total: int) -> list[int]:
    assert length >= 2 and total >= 2
    mode = random.choices(("constant", "walk", "mirror"), weights=(1, 3, 2), k=1)[0]

    if mode == "constant":
        cuts = [random.randint(1, total - 1)] * length
    elif mode == "mirror":
        half = []
        cur = random.randint(1, total - 1)
        for _ in range((length + 1) // 2):
            half.append(cur)
            cur = max(1, min(total - 1, cur + random.choice((-1, 0, 1))))
        cuts = half + half[: length // 2][::-1]
    else:
        cur = random.randint(1, total - 1)
        cuts = [cur]
        for _ in range(length - 1):
            cur = max(1, min(total - 1, cur + random.choice((-2, -1, 0, 1, 2))))
            cuts.append(cur)

    if len(set(cuts)) == 1 and total > 2 and length > 1 and random.random() < 0.75:
        idx = random.randrange(length)
        options = [v for v in (cuts[idx] - 1, cuts[idx] + 1) if 1 <= v <= total - 1]
        if options:
            cuts[idx] = random.choice(options)
    return cuts


def _sample_pair_specs() -> list[dict[str, object]]:
    pair_count = random.choices(_PAIR_COUNT_OPTIONS, weights=_PAIR_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(_COLORS, pair_count)

    if pair_count == 2 and random.random() < 0.75:
        orientations = random.sample(("H", "V"), 2)
    else:
        orientations = [random.choice(("H", "V")) for _ in range(pair_count)]
        if pair_count >= 3 and len(set(orientations)) == 1:
            orientations[random.randrange(pair_count)] = "V" if orientations[0] == "H" else "H"

    specs: list[dict[str, object]] = []
    for color, orientation in zip(colors, orientations):
        gap = random.choice(_GAP_OPTIONS)
        if orientation == "H":
            rect_h = random.randint(2, 6)
            rect_w = random.randint(4, 8)
            cuts = _sample_profile(rect_h, rect_w)
            specs.append(
                {
                    "color": color,
                    "orientation": orientation,
                    "rect_h": rect_h,
                    "rect_w": rect_w,
                    "gap": gap,
                    "cuts": cuts,
                    "corridor_h": rect_h,
                    "corridor_w": rect_w + gap,
                }
            )
        else:
            rect_h = random.randint(4, 8)
            rect_w = random.randint(2, 6)
            cuts = _sample_profile(rect_w, rect_h)
            specs.append(
                {
                    "color": color,
                    "orientation": orientation,
                    "rect_h": rect_h,
                    "rect_w": rect_w,
                    "gap": gap,
                    "cuts": cuts,
                    "corridor_h": rect_h + gap,
                    "corridor_w": rect_w,
                }
            )
    return specs


def _choose_canvas(specs: list[dict[str, object]]) -> tuple[int, int] | None:
    min_h = max(_GRID_MIN, max(int(spec["corridor_h"]) for spec in specs))
    min_w = max(_GRID_MIN, max(int(spec["corridor_w"]) for spec in specs))
    if min_h > _GRID_MAX or min_w > _GRID_MAX:
        return None

    total_output_area = sum(int(spec["rect_h"]) * int(spec["rect_w"]) for spec in specs)
    total_corridor_area = sum(int(spec["corridor_h"]) * int(spec["corridor_w"]) for spec in specs)

    for _ in range(80):
        h = random.randint(min_h, _GRID_MAX)
        w = random.randint(min_w, _GRID_MAX)
        if total_output_area > _OUTPUT_FILL_MAX * h * w:
            continue
        if total_corridor_area > 0.72 * h * w:
            continue
        return h, w
    return None


def _place_corridors(h: int, w: int, specs: list[dict[str, object]]) -> bool:
    blocked = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["corridor_h"]) * int(specs[i]["corridor_w"]), reverse=True)
    for idx in order:
        spec = specs[idx]
        ch = int(spec["corridor_h"])
        cw = int(spec["corridor_w"])
        candidates = []
        for top in range(h - ch + 1):
            for left in range(w - cw + 1):
                r0 = max(0, top - _PADDING)
                r1 = min(h, top + ch + _PADDING)
                c0 = max(0, left - _PADDING)
                c1 = min(w, left + cw + _PADDING)
                if blocked[r0:r1, c0:c1].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return False
        top, left = random.choice(candidates)
        spec["top"] = top
        spec["left"] = left
        r0 = max(0, top - _PADDING)
        r1 = min(h, top + ch + _PADDING)
        c0 = max(0, left - _PADDING)
        c1 = min(w, left + cw + _PADDING)
        blocked[r0:r1, c0:c1] = True
    return True


def _render_pair(inp: np.ndarray, out: np.ndarray, spec: dict[str, object]) -> None:
    color = int(spec["color"])
    orientation = str(spec["orientation"])
    rect_h = int(spec["rect_h"])
    rect_w = int(spec["rect_w"])
    gap = int(spec["gap"])
    cuts = list(spec["cuts"])
    top = int(spec["top"])
    left = int(spec["left"])
    half = gap // 2

    if orientation == "H":
        final_left = left + half
        out[top : top + rect_h, final_left : final_left + rect_w] = color
        for r, cut in enumerate(cuts):
            inp[top + r, left : left + cut] = color
            inp[top + r, left + gap + cut : left + gap + rect_w] = color
        return

    final_top = top + half
    out[final_top : final_top + rect_h, left : left + rect_w] = color
    for c, cut in enumerate(cuts):
        inp[top : top + cut, left + c] = color
        inp[top + gap + cut : top + gap + rect_h, left + c] = color


def _input_is_valid(inp: np.ndarray, specs: list[dict[str, object]]) -> bool:
    for spec in specs:
        color = int(spec["color"])
        if len(_components(inp == color)) != 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_pair_specs()
    canvas = _choose_canvas(specs)
    if canvas is None:
        return None
    h, w = canvas
    if not _place_corridors(h, w, specs):
        return None

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for spec in specs:
        _render_pair(inp, out, spec)

    if not _input_is_valid(inp, specs):
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragment-slide sample")
