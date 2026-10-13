import random
from collections import defaultdict

import numpy as np


GEN_RETRIES = 1000
SPEC_RETRIES = 300
RECT_RETRIES = 300
COUNT_OPTIONS = [(1, 1), (1, 1), (2, 1), (1, 2), (2, 2)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rect_area(r0: int, c0: int, r1: int, c1: int) -> int:
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _full_border_mask(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _draw_full_border(out: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> None:
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _draw_open_border(out: np.ndarray, bg: int, color: int, r0: int, c0: int, r1: int, c1: int) -> None:
    if c1 - c0 >= 2:
        out[r0, c0 + 1 : c1] = color
        out[r1, c0 + 1 : c1] = color
    if r1 - r0 >= 2:
        out[r0 + 1 : r1, c0] = color
        out[r0 + 1 : r1, c1] = color
    # Open-corner rectangles actively restore their corners to background,
    # even if a larger rectangle painted those cells earlier.
    out[r0, c0] = bg
    out[r0, c1] = bg
    out[r1, c0] = bg
    out[r1, c1] = bg


def _sorted_specs(specs: list[dict]) -> list[dict]:
    return sorted(
        specs,
        key=lambda spec: (
            -int(spec["area"]),
            int(spec["r0"]),
            int(spec["c0"]),
            int(spec["r1"]),
            int(spec["c1"]),
        ),
    )


def _render_specs(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in _sorted_specs(specs):
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = int(spec["r1"])
        c1 = int(spec["c1"])
        color = int(spec["color"])
        if spec["style"] == "full":
            _draw_full_border(out, color, r0, c0, r1, c1)
        else:
            _draw_open_border(out, bg, color, r0, c0, r1, c1)
    return out


def _collect_specs(grid: np.ndarray, bg: int) -> list[dict]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(grid != bg):
        groups[int(grid[r, c])].append((int(r), int(c)))

    specs: list[dict] = []
    for color, pts in groups.items():
        count = len(pts)
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        area = _rect_area(r0, c0, r1, c1)

        if count == 2:
            assert r0 < r1 and c0 < c1, "two-point markers must be opposite rectangle corners"
            specs.append(
                {
                    "color": color,
                    "style": "full",
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "area": area,
                }
            )
            continue

        assert count == 4, "each non-background color must appear exactly twice or four times"
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        assert set(pts) == corners, "four-point markers must occupy the rectangle corners"
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "open-corner rectangles need non-corner edge cells"
        specs.append(
            {
                "color": color,
                "style": "open",
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "area": area,
            }
        )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group singleton markers by color, infer either a diagonal pair or four corners,
    then redraw the corresponding rectangle border style.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    specs = _collect_specs(g, bg)
    return _render_specs(g.shape, bg, specs)


def _random_rect(style: str, h: int, w: int, rng: random.Random) -> tuple[int, int, int, int]:
    min_span = 1 if style == "full" else 2
    rect_h = rng.randint(min_span + 1, h)
    rect_w = rng.randint(min_span + 1, w)
    r0 = rng.randint(0, h - rect_h)
    c0 = rng.randint(0, w - rect_w)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    return r0, c0, r1, c1


def _marker_cells(style: str, r0: int, c0: int, r1: int, c1: int, rng: random.Random) -> list[tuple[int, int]]:
    if style == "full":
        if rng.random() < 0.5:
            return [(r0, c0), (r1, c1)]
        return [(r0, c1), (r1, c0)]
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _try_build_specs(h: int, w: int, bg: int, rng: random.Random) -> list[dict] | None:
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)

    pair_count, open_count = rng.choice(COUNT_OPTIONS)
    total = pair_count + open_count
    if total > len(colors):
        return None

    styles = ["full"] * pair_count + ["open"] * open_count
    rng.shuffle(styles)
    want_overlap = total >= 3 and rng.random() < 0.5

    for _ in range(SPEC_RETRIES):
        specs: list[dict] = []
        used_areas: set[int] = set()
        used_markers: set[tuple[int, int]] = set()
        footprint_masks: list[np.ndarray] = []
        overlap_pairs = 0
        overlap_points = 0

        ok = True
        for style in styles:
            placed = False
            for _ in range(RECT_RETRIES):
                r0, c0, r1, c1 = _random_rect(style, h, w, rng)
                area = _rect_area(r0, c0, r1, c1)
                if area in used_areas:
                    continue

                markers = _marker_cells(style, r0, c0, r1, c1, rng)
                if any(cell in used_markers for cell in markers):
                    continue

                mask = _full_border_mask((h, w), r0, c0, r1, c1)
                local_overlap_pairs = 0
                local_overlap_points = 0
                for old_mask in footprint_masks:
                    inter = int(np.count_nonzero(mask & old_mask))
                    if inter > 0:
                        local_overlap_pairs += 1
                        local_overlap_points += inter
                if local_overlap_pairs > 1:
                    continue
                if overlap_points + local_overlap_points > max(6, total * 2):
                    continue

                specs.append(
                    {
                        "color": colors[len(specs)],
                        "style": style,
                        "r0": r0,
                        "c0": c0,
                        "r1": r1,
                        "c1": c1,
                        "area": area,
                        "markers": markers,
                    }
                )
                used_areas.add(area)
                used_markers.update(markers)
                footprint_masks.append(mask)
                overlap_pairs += local_overlap_pairs
                overlap_points += local_overlap_points
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        if want_overlap and overlap_pairs == 0:
            continue

        expected = _render_specs((h, w), bg, specs)
        colored = int(np.count_nonzero(expected != bg))
        if colored < 8 or colored > int(0.45 * h * w):
            continue

        return specs
    return None


def _build_input(h: int, w: int, bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        for r, c in spec["markers"]:
            grid[r, c] = int(spec["color"])
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 20)
    w = rng.randint(10, 20)
    bg = rng.randint(0, 9)

    specs = _try_build_specs(h, w, bg, rng)
    if specs is None:
        return None

    inp = _build_input(h, w, bg, specs)
    expected = _render_specs((h, w), bg, specs)

    if np.array_equal(inp, expected):
        return None
    if _get_bg(inp) != bg:
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
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
