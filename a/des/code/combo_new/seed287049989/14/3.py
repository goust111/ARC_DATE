import numpy as np


_RNG = np.random.default_rng()
_PALETTE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_STRIPE_WIDTH_OPTIONS = (2, 3)
_MAX_INFER_WIDTH = 6
_GEN_RETRIES = 300
_RECT_RETRIES = 120


def _touches_edge(bbox: tuple[int, int, int, int], height: int, width: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    height, width = map(int, grid.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = int(r)
            c0 = c1 = int(c)

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                r0 = min(r0, int(rr))
                r1 = max(r1, int(rr))
                c0 = min(c0, int(cc))
                c1 = max(c1, int(cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            comps.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                    "area": len(coords),
                }
            )

    return comps


def _stripe_grid(
    shape: tuple[int, int], orientation: str, stripe_width: int, stripe_colors: tuple[int, int]
) -> np.ndarray:
    height, width = map(int, shape)
    if orientation == "horizontal":
        bands = (np.arange(height) // int(stripe_width)) % 2
        row_colors = np.where(bands == 0, int(stripe_colors[0]), int(stripe_colors[1])).astype(int)
        return np.broadcast_to(row_colors[:, None], (height, width)).copy()

    bands = (np.arange(width) // int(stripe_width)) % 2
    col_colors = np.where(bands == 0, int(stripe_colors[0]), int(stripe_colors[1])).astype(int)
    return np.broadcast_to(col_colors[None, :], (height, width)).copy()


def _rectangle_components(grid: np.ndarray, stripe_colors: tuple[int, int]) -> list[dict] | None:
    height, width = map(int, grid.shape)
    rectangles: list[dict] = []
    for color in map(int, np.unique(grid)):
        if int(color) in stripe_colors:
            continue
        for comp in _components_of_color(grid, int(color)):
            r0, c0, r1, c1 = comp["bbox"]
            comp_h = int(r1 - r0 + 1)
            comp_w = int(c1 - c0 + 1)
            if int(comp["area"]) != comp_h * comp_w:
                return None
            if _touches_edge(comp["bbox"], height, width):
                return None
            if comp_h == comp_w:
                return None
            comp["height"] = comp_h
            comp["width"] = comp_w
            rectangles.append(comp)

    return rectangles if rectangles else None


def _candidate_model(
    grid: np.ndarray, orientation: str, stripe_width: int, stripe_colors: tuple[int, int]
) -> dict | None:
    color_a, color_b = map(int, stripe_colors)
    mask = (grid == int(color_a)) | (grid == int(color_b))
    score = int(mask.sum())
    if score == 0 or not bool((grid == int(color_a)).any()) or not bool((grid == int(color_b)).any()):
        return None

    pattern = _stripe_grid(grid.shape, orientation, stripe_width, stripe_colors)
    if not np.array_equal(grid[mask], pattern[mask]):
        return None

    rectangles = _rectangle_components(grid, stripe_colors)
    if rectangles is None:
        return None

    return {
        "orientation": orientation,
        "width": int(stripe_width),
        "colors": (int(color_a), int(color_b)),
        "pattern": pattern,
        "rectangles": rectangles,
        "score": score,
    }


def _infer_model(grid: np.ndarray) -> dict:
    colors = [int(c) for c in np.unique(grid)]
    counts = {color: int(np.count_nonzero(grid == int(color))) for color in colors}
    ordered_colors = sorted(colors, key=lambda color: (-counts[int(color)], int(color)))

    candidates: list[dict] = []
    for i, color_a in enumerate(ordered_colors):
        for color_b in ordered_colors[i + 1 :]:
            for orientation in ("horizontal", "vertical"):
                axis_len = int(grid.shape[0] if orientation == "horizontal" else grid.shape[1])
                for stripe_width in range(1, min(axis_len, _MAX_INFER_WIDTH) + 1):
                    for stripe_colors in ((int(color_a), int(color_b)), (int(color_b), int(color_a))):
                        candidate = _candidate_model(grid, orientation, stripe_width, stripe_colors)
                        if candidate is not None:
                            candidates.append(candidate)

    assert candidates, "failed to infer a stripe model"
    candidates.sort(
        key=lambda item: (
            int(item["score"]),
            len(item["rectangles"]),
            -int(item["width"]),
            0 if item["orientation"] == "horizontal" else 1,
            -int(item["colors"][0]),
            -int(item["colors"][1]),
        ),
        reverse=True,
    )
    return candidates[0]


def _should_restore(orientation: str, rect: dict) -> bool:
    rect_h = int(rect["height"])
    rect_w = int(rect["width"])
    return (orientation == "horizontal" and rect_w > rect_h) or (
        orientation == "vertical" and rect_h > rect_w
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden alternating stripe field and restore it only through rectangles aligned with the stripe direction.

    Input structure:
    - The whole grid is covered by a two-color horizontal or vertical stripe pattern with a fixed stripe width.
    - Several interior solid monochrome rectangles in other colors occlude parts of that stripe field.
    - Every occluder is a non-square axis-aligned rectangle, so each one is either wider than tall or taller than wide.

    Rule:
    - Recover the global stripe orientation, stripe width, and the two stripe colors from the visible non-occluded cells.
    - Treat every other color component as one rectangular occluder.
    - If the stripes are horizontal, replace only wide occluders with the stripe colors underneath them; if the stripes are vertical, replace only tall occluders.
    - Leave occluders whose long axis is perpendicular to the stripe direction unchanged.

    Key insight:
    The first step is to reconstruct the single periodic stripe template spanning the entire grid; only then can each rectangle be filtered by whether its long axis runs parallel to those stripes.
    """
    g = np.asarray(grid, dtype=int)
    model = _infer_model(g)
    out = g.copy()

    for rect in model["rectangles"]:
        if not _should_restore(str(model["orientation"]), rect):
            continue
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = model["pattern"][r0 : r1 + 1, c0 : c1 + 1]

    return out


def _model_key(model: dict) -> tuple:
    return (str(model["orientation"]), int(model["width"]), tuple(map(int, model["colors"])))


def _rectangle_kinds(orientation: str, n_rectangles: int) -> list[str]:
    parallel = "wide" if orientation == "horizontal" else "tall"
    perpendicular = "tall" if orientation == "horizontal" else "wide"
    kinds = [parallel, perpendicular]
    for _ in range(n_rectangles - 2):
        kinds.append(parallel if int(_RNG.integers(2)) == 0 else perpendicular)
    _RNG.shuffle(kinds)
    return kinds


def _sample_rect_dims(kind: str, stripe_width: int, height: int, width: int) -> tuple[int, int] | None:
    if kind == "wide":
        rect_h_lo = int(stripe_width + 1)
        rect_h_hi = int(min(5, height - 3))
        if rect_h_lo > rect_h_hi:
            return None
        rect_h = int(_RNG.integers(rect_h_lo, rect_h_hi + 1))

        rect_w_lo = int(max(rect_h + 2, 6))
        rect_w_hi = int(min(10, width - 2))
        if rect_w_lo > rect_w_hi:
            return None
        rect_w = int(_RNG.integers(rect_w_lo, rect_w_hi + 1))
        return rect_h, rect_w

    rect_w_lo = int(stripe_width + 1)
    rect_w_hi = int(min(5, width - 3))
    if rect_w_lo > rect_w_hi:
        return None
    rect_w = int(_RNG.integers(rect_w_lo, rect_w_hi + 1))

    rect_h_lo = int(max(rect_w + 2, 6))
    rect_h_hi = int(min(10, height - 2))
    if rect_h_lo > rect_h_hi:
        return None
    rect_h = int(_RNG.integers(rect_h_lo, rect_h_hi + 1))
    return rect_h, rect_w


def _place_rectangles(
    shape: tuple[int, int], kinds: list[str], colors: list[int], stripe_width: int
) -> list[dict] | None:
    height, width = map(int, shape)
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    order = sorted(range(len(kinds)), key=lambda idx: 0 if kinds[idx] == "tall" else 1)
    for idx in order:
        kind = str(kinds[idx])
        color = int(colors[idx])
        success = False
        for _ in range(_RECT_RETRIES):
            dims = _sample_rect_dims(kind, stripe_width, height, width)
            if dims is None:
                return None
            rect_h, rect_w = map(int, dims)
            r0 = int(_RNG.integers(1, height - rect_h))
            c0 = int(_RNG.integers(1, width - rect_w))
            r1 = int(r0 + rect_h - 1)
            c1 = int(c0 + rect_w - 1)

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height - 1, r1 + 1)
            cc1 = min(width - 1, c1 + 1)
            if bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue

            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed.append(
                {
                    "color": int(color),
                    "kind": kind,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(rect_h),
                    "width": int(rect_w),
                }
            )
            success = True
            break

        if not success:
            return None

    return placed


def _gen_attempt() -> np.ndarray | None:
    n_rectangles = int(_RNG.integers(2, 5))
    min_side = 16 + 2 * max(0, n_rectangles - 2)
    height = int(_RNG.integers(min_side, 31))
    width = int(_RNG.integers(min_side, 31))

    orientation = "horizontal" if int(_RNG.integers(2)) == 0 else "vertical"
    stripe_width = int(_RNG.choice(_STRIPE_WIDTH_OPTIONS))

    palette = [int(c) for c in _RNG.choice(_PALETTE, size=n_rectangles + 2, replace=False)]
    stripe_colors = (int(palette[0]), int(palette[1]))
    rect_colors = [int(c) for c in palette[2:]]

    kinds = _rectangle_kinds(orientation, n_rectangles)
    rectangles = _place_rectangles((height, width), kinds, rect_colors, stripe_width)
    if rectangles is None:
        return None

    base = _stripe_grid((height, width), orientation, stripe_width, stripe_colors)
    inp = base.copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])

    inferred = _infer_model(inp)
    if _model_key(inferred) != (orientation, stripe_width, stripe_colors):
        return None

    expected = inp.copy()
    for rect in rectangles:
        if not _should_restore(orientation, rect):
            continue
        r0, c0, r1, c1 = rect["bbox"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = base[r0 : r1 + 1, c0 : c1 + 1]

    if np.array_equal(expected, inp):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
