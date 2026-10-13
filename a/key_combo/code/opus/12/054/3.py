import hashlib
import itertools
import random

import numpy as np


PANEL_SIZE = 5
INPUT_SHAPE = (5, 23)
PANEL_STARTS = (0, 6, 12, 18)
SEPARATOR_COLS = (5, 11, 17)

BG = 0
PANEL_COLORS = (1, 2, 3, 4)
SEPARATOR_COLOR = 5
MARKER_COLOR = 7

GEN_MAX_ATTEMPTS = 600
GEN_LOCAL_ATTEMPTS = 80
_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _count_intersections(panel: np.ndarray, color: int, marker_cols: tuple[int, ...]) -> int:
    total = 0
    for c in marker_cols:
        total += int(np.sum(panel[1:, c] == color))
    return total


def _overlay_panels(panels: list[np.ndarray], front_order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    owner = np.full((PANEL_SIZE, PANEL_SIZE), -1, dtype=int)
    for idx in reversed(front_order):
        mask = panels[idx] != BG
        out[mask] = panels[idx][mask]
        owner[mask] = idx
    return out, owner


def _extract_panel_infos(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE, "expected a 5x23 grid with four 5x5 panels"

    separators = g[:, SEPARATOR_COLS]
    assert np.all(separators == SEPARATOR_COLOR), "separator columns must all be gray"

    infos = []
    used_colors = set()
    for start in PANEL_STARTS:
        panel = g[:, start : start + PANEL_SIZE].copy()
        assert not np.any(panel[:, :] == SEPARATOR_COLOR), "separator color may only appear in divider columns"

        marker_mask = panel == MARKER_COLOR
        assert not np.any(marker_mask[1:, :]), "orange markers must lie only on the top row"
        marker_cols = tuple(int(c) for c in np.flatnonzero(marker_mask[0]))
        assert 1 <= len(marker_cols) <= 3, "each panel needs 1-3 markers"

        sprite_colors = set(map(int, np.unique(panel))) - {BG, MARKER_COLOR}
        assert len(sprite_colors) == 1, "each panel must contain exactly one foreground sprite color"
        color = int(next(iter(sprite_colors)))
        assert color in PANEL_COLORS, "panel sprite colors must come from blue/red/green/yellow"
        assert color not in used_colors, "panel sprite colors must be unique"
        used_colors.add(color)

        sprite_mask = panel == color
        area = int(sprite_mask.sum())
        assert 3 <= area <= 8, "sprite size must be 3-8 pixels"
        assert _is_connected(sprite_mask), "sprite must be 4-connected"
        assert not np.any(sprite_mask[0, list(marker_cols)]), "markers may not overlap the sprite"

        count = _count_intersections(panel, color, marker_cols)
        infos.append(
            {
                "panel": panel,
                "color": color,
                "marker_cols": marker_cols,
                "marker_count": len(marker_cols),
                "area": area,
                "count": count,
            }
        )

    counts = [info["count"] for info in infos]
    assert len(set(counts)) == 4, "panel intersection counts must be distinct"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank four aligned panels by invisible ray-hit counts, then overlay them in that depth order.

    Input structure:
    - The grid is 5x23 and is split by three gray separator columns into four aligned 5x5 panels.
    - Inside each panel, black is background, one unique non-orange color forms a 4-connected sprite, and 1-3 orange marker pixels sit on the top row.
    - The four sprite colors are all different and come from blue, red, green, and yellow.

    Rule:
    - For each panel, each orange marker emits a vertical downward probe starting one row below the marker and continuing to the panel bottom.
    - Count how many sprite-colored cells are crossed by all probes in that panel; this sum is the panel's intersection count.
    - Sort panels by increasing intersection count so the smallest count becomes the front layer, then transparently overlay the four 5x5 panels on one canvas, treating black as transparent and any non-black cell as opaque.

    Key insight:
    - The orange markers do not draw anything themselves; they define hidden measurement rays whose hit totals uniquely encode the layer order that controls the visible overlay.
    """
    infos = _extract_panel_infos(np.asarray(grid, dtype=int))
    front_order = [idx for idx, _ in sorted(enumerate(infos), key=lambda item: item[1]["count"])]
    panels = [info["panel"] for info in infos]
    out, _owner = _overlay_panels(panels, front_order)
    return out


def _random_connected_mask(rng: random.Random, size: int) -> np.ndarray | None:
    start = (rng.randrange(PANEL_SIZE), rng.randrange(PANEL_SIZE))
    cells = {start}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        frontier = sorted(frontier)
        degrees = []
        for nr, nc in frontier:
            degree = sum((ar, ac) in cells for ar, ac in _neighbors4(nr, nc))
            degrees.append((degree, nr, nc))
        max_degree = max(degree for degree, _nr, _nc in degrees)
        if rng.random() < 0.55:
            choices = [(nr, nc) for degree, nr, nc in degrees if degree == max_degree]
        else:
            choices = [(nr, nc) for _degree, nr, nc in degrees]
        cells.add(rng.choice(choices))

    mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask if _is_connected(mask) else None


def _marker_options(mask: np.ndarray) -> list[dict]:
    empty_top_cols = [c for c in range(PANEL_SIZE) if not bool(mask[0, c])]
    options = []
    for k in range(1, min(3, len(empty_top_cols)) + 1):
        for cols in itertools.combinations(empty_top_cols, k):
            count = int(sum(int(np.sum(mask[1:, c])) for c in cols))
            options.append({"cols": tuple(int(c) for c in cols), "count": count})
    return options


def _build_panel_base(color: int, rng: random.Random) -> dict | None:
    for _ in range(GEN_LOCAL_ATTEMPTS):
        size = rng.randint(3, 8)
        mask = _random_connected_mask(rng, size)
        if mask is None:
            continue
        options = _marker_options(mask)
        unique_counts = {opt["count"] for opt in options}
        if not options or len(unique_counts) < 2:
            continue
        if np.sum(mask[1:, :]) < 2:
            continue
        return {"color": color, "mask": mask, "area": int(mask.sum()), "options": options}
    return None


def _choose_options(panel_bases: list[dict], rng: random.Random, prefer_zero: bool) -> list[dict] | None:
    if prefer_zero and not any(any(opt["count"] == 0 for opt in base["options"]) for base in panel_bases):
        return None

    ordered_indices = sorted(range(4), key=lambda idx: len(panel_bases[idx]["options"]))
    chosen = [None, None, None, None]
    used_counts = set()

    def backtrack(pos: int) -> bool:
        if pos == 4:
            return (not prefer_zero) or any(opt["count"] == 0 for opt in chosen if opt is not None)

        idx = ordered_indices[pos]
        options = list(panel_bases[idx]["options"])
        rng.shuffle(options)
        for opt in options:
            if opt["count"] in used_counts:
                continue
            chosen[idx] = opt
            used_counts.add(opt["count"])
            if backtrack(pos + 1):
                return True
            used_counts.remove(opt["count"])
            chosen[idx] = None
        return False

    if backtrack(0):
        return chosen
    return None


def _assemble_panel(mask: np.ndarray, color: int, marker_cols: tuple[int, ...]) -> np.ndarray:
    panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    panel[mask] = color
    for c in marker_cols:
        panel[0, c] = MARKER_COLOR
    return panel


def _candidate_valid(specs: list[dict], expected: np.ndarray) -> bool:
    counts = [spec["count"] for spec in specs]
    areas = [spec["area"] for spec in specs]
    marker_counts = [len(spec["marker_cols"]) for spec in specs]
    panels = [spec["panel"] for spec in specs]

    front_order = sorted(range(4), key=lambda idx: counts[idx])
    if front_order == [0, 1, 2, 3]:
        return False

    area_order = sorted(range(4), key=lambda idx: (areas[idx], idx))
    nonzero_order = sorted(range(4), key=lambda idx: (int(np.count_nonzero(panels[idx])), idx))
    marker_order = sorted(range(4), key=lambda idx: (marker_counts[idx], idx))
    if front_order in (area_order, nonzero_order, marker_order):
        return False

    stack_count = np.sum(np.stack([panel != BG for panel in panels], axis=0), axis=0)
    if int(np.sum(stack_count >= 2)) < 4:
        return False
    if not np.any(stack_count[1:, :] >= 2):
        return False

    _, owner = _overlay_panels(panels, front_order)
    if any(int(np.sum(owner == idx)) == 0 for idx in range(4)):
        return False

    visible_colors = set(map(int, np.unique(expected))) - {BG}
    if MARKER_COLOR not in visible_colors:
        return False
    if len(visible_colors) < 3:
        return False

    for alt_order in ([0, 1, 2, 3], [3, 2, 1, 0], area_order, nonzero_order, marker_order):
        if alt_order == front_order:
            return False
        alt, _owner = _overlay_panels(panels, alt_order)
        if np.array_equal(alt, expected):
            return False

    sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
    if sig in _SEEN_OUTPUT_SIGS:
        return False
    return True


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    colors = list(PANEL_COLORS)
    rng.shuffle(colors)

    panel_bases = []
    for color in colors:
        base = _build_panel_base(color, rng)
        if base is None:
            return None
        panel_bases.append(base)

    chosen = _choose_options(panel_bases, rng, prefer_zero=rng.random() < 0.35)
    if chosen is None:
        return None

    specs = []
    for base, opt in zip(panel_bases, chosen):
        panel = _assemble_panel(base["mask"], base["color"], opt["cols"])
        specs.append(
            {
                "panel": panel,
                "color": base["color"],
                "marker_cols": opt["cols"],
                "count": opt["count"],
                "area": base["area"],
            }
        )

    front_order = sorted(range(4), key=lambda idx: specs[idx]["count"])
    expected, _owner = _overlay_panels([spec["panel"] for spec in specs], front_order)
    if not _candidate_valid(specs, expected):
        return None

    grid = np.zeros(INPUT_SHAPE, dtype=int)
    grid[:, SEPARATOR_COLS] = SEPARATOR_COLOR
    for start, spec in zip(PANEL_STARTS, specs):
        grid[:, start : start + PANEL_SIZE] = spec["panel"]

    parsed = _extract_panel_infos(grid)
    parsed_expected, _owner = _overlay_panels(
        [info["panel"] for info in parsed],
        sorted(range(4), key=lambda idx: parsed[idx]["count"]),
    )
    if not np.array_equal(parsed_expected, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
    _SEEN_OUTPUT_SIGS.add(sig)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate ray-order overlay input")
