import numpy as np
from arc_utils import components_mask


def _find_full_cross(g: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        rows = np.where(np.all(g == color, axis=1))[0]
        cols = np.where(np.all(g == color, axis=0))[0]
        for r in rows:
            for c in cols:
                candidates.append((color, int(r), int(c)))
    assert len(candidates) == 1
    return candidates[0]


def _unique_common_bg(panels: list[np.ndarray]) -> int:
    color_sets = [set(map(int, np.unique(panel))) for panel in panels]
    common = set.intersection(*color_sets)
    assert len(common) == 1
    return common.pop()


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the four panels split by the full cross into one panel using a fixed layer order.

    Input structure
    A single full-row/full-column cross divides the grid into four equal-size rectangular panels; each panel contains one transparent background color plus one foreground color.

    Rule
    Remove the cross and align the four panels by local coordinates. Paint non-background pixels from bottom layer to top layer in panel-position order: bottom-left, top-right, top-left, bottom-right.

    Key insight
    The output is the four quadrants collapsed onto the same canvas, with conflicts resolved by the quadrant layer stack rather than by color id.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    cross_color, cross_r, cross_c = _find_full_cross(g)
    assert H == 2 * cross_r + 1
    assert W == 2 * cross_c + 1

    top_left = g[:cross_r, :cross_c]
    top_right = g[:cross_r, cross_c + 1 :]
    bottom_left = g[cross_r + 1 :, :cross_c]
    bottom_right = g[cross_r + 1 :, cross_c + 1 :]
    panels = [top_left, top_right, bottom_left, bottom_right]
    assert len({panel.shape for panel in panels}) == 1

    bg = _unique_common_bg(panels)
    for panel in panels:
        colors = set(map(int, np.unique(panel)))
        assert cross_color not in colors
        assert len(colors - {bg}) == 1

    out = np.full(top_left.shape, bg, dtype=int)
    for panel in (bottom_left, top_right, top_left, bottom_right):
        mask = panel != bg
        out[mask] = panel[mask]
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
PANEL_SIDE_MIN = 3
PANEL_SIDE_MAX_EXCLUSIVE = 8
PALETTE_SIZE = 10
INPUT_COLOR_COUNT = 6

# Panel fill and connected-component constraints.
MIN_FILL_RATIO = 0.30
MAX_FILL_RATIO = 0.80
FOUR_CONNECTIVITY = 4
MAX_PANEL_COMPONENTS = 5
LOWER_OUTPUT_MAX_COMPONENTS = 4

# Layer names and stochastic controls.
PANEL_NAMES = ("TL", "TR", "BL", "BR")
LOWER_PANEL_NAMES = ("TL", "TR", "BL")
LOWER_ORDER_SHUFFLE_PROB = 0.25
GENERATION_RETRY_LIMIT = 300


def _gen_component_count(mask: np.ndarray) -> int:
    return len(components_mask(mask.astype(bool), connectivity=FOUR_CONNECTIVITY))


def _gen_dilate4(mask: np.ndarray) -> np.ndarray:
    near = np.zeros(mask.shape, dtype=bool)
    near[1:, :] |= mask[:-1, :]
    near[:-1, :] |= mask[1:, :]
    near[:, 1:] |= mask[:, :-1]
    near[:, :-1] |= mask[:, 1:]
    return near


def _gen_frontier(mask: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    near = _gen_dilate4(mask)
    return near & allowed & ~mask


def _gen_neighbor_mask(mask: np.ndarray) -> np.ndarray:
    return _gen_dilate4(mask)


def _gen_pick(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r, c = coords[np.random.randint(len(coords))]
    return int(r), int(c)


def _gen_blob_mask(
    shape: tuple[int, int],
    target: int,
    max_components: int,
    allowed: np.ndarray | None = None,
) -> np.ndarray | None:
    if allowed is None:
        allowed = np.ones(shape, dtype=bool)
    else:
        allowed = allowed.astype(bool).copy()
    if target < 0 or target > int(allowed.sum()):
        return None
    mask = np.zeros(shape, dtype=bool)
    if target == 0:
        return mask

    options = np.arange(1, min(max_components, target) + 1)
    weights = options.astype(float)
    weights /= weights.sum()
    component_goal = int(np.random.choice(options, p=weights))
    labels = np.zeros(shape, dtype=int)
    for label in range(1, component_goal + 1):
        choices = allowed & (labels == 0) & ~_gen_neighbor_mask(labels > 0)
        picked = _gen_pick(choices)
        if picked is None:
            return None
        labels[picked] = label

    sizes = np.ones(component_goal, dtype=int)
    for _ in range(target - component_goal):
        sizes[np.random.randint(component_goal)] += 1

    stuck: set[int] = set()
    while int((labels > 0).sum()) < target:
        growable = [
            label
            for label in range(1, component_goal + 1)
            if int((labels == label).sum()) < int(sizes[label - 1]) and label not in stuck
        ]
        if not growable:
            return None
        label = int(growable[np.random.randint(len(growable))])
        own = labels == label
        other = (labels > 0) & ~own
        frontier = _gen_frontier(own, allowed & (labels == 0))
        choices = frontier & ~_gen_neighbor_mask(other)
        picked = _gen_pick(choices)
        if picked is None:
            stuck.add(label)
            continue
        labels[picked] = label
        stuck.clear()

    mask = labels > 0
    if _gen_component_count(mask) != component_goal:
        return None
    return mask


def _gen_grow_to_count(
    mask: np.ndarray,
    allowed: np.ndarray,
    target: int,
    max_components: int,
) -> np.ndarray | None:
    mask = mask.astype(bool).copy()
    allowed = allowed.astype(bool).copy()
    if target < int(mask.sum()) or target > int(mask.sum()) + int((allowed & ~mask).sum()):
        return None
    if _gen_component_count(mask) > max_components:
        return None

    while int(mask.sum()) < target:
        frontier = _gen_frontier(mask, allowed)
        if frontier.any():
            choices = frontier
        else:
            if _gen_component_count(mask) >= max_components:
                return None
            choices = allowed & ~mask

        picked = _gen_pick(choices)
        if picked is None:
            return None
        mask[picked] = True
        if _gen_component_count(mask) > max_components:
            return None
    return mask


def _gen_grow_hidden_to_count(
    visible: np.ndarray,
    anchor: np.ndarray,
    higher: np.ndarray,
    target: int,
    max_components: int,
) -> np.ndarray | None:
    visible = visible.astype(bool)
    hidden_target = target - int(visible.sum())
    visible_components = _gen_component_count(visible)
    hidden_budget = max_components - visible_components
    if hidden_target < int(anchor.sum()) or hidden_budget < 1:
        return None

    hidden_allowed = higher & ~visible
    quiet_allowed = hidden_allowed & ~_gen_neighbor_mask(visible)
    for allowed in (quiet_allowed, hidden_allowed):
        hidden = _gen_grow_to_count(anchor, allowed, hidden_target, hidden_budget)
        if hidden is None:
            continue
        combined = visible | hidden
        if int(combined.sum()) == target and _gen_component_count(combined) <= max_components:
            return combined
    return None


def _gen_positive_split(total: int, parts: int) -> list[int] | None:
    if total < parts:
        return None
    if parts == 1:
        return [total]
    cuts = sorted(np.random.choice(np.arange(1, total), size=parts - 1, replace=False))
    values = [cuts[0]]
    values.extend(cuts[i] - cuts[i - 1] for i in range(1, len(cuts)))
    values.append(total - cuts[-1])
    return [int(v) for v in values]


def _gen_output_counts(n_cells: int, min_fill: int, max_fill: int) -> dict[str, int] | None:
    lower_count = len(LOWER_PANEL_NAMES)
    br_hi = min(max_fill, n_cells - lower_count)
    if br_hi < min_fill:
        return None
    br_count = int(np.random.randint(min_fill, br_hi + 1))
    remaining = n_cells - br_count
    bg_count = int(np.random.randint(0, remaining - lower_count + 1))
    lower = _gen_positive_split(remaining - bg_count, lower_count)
    if lower is None:
        return None
    np.random.shuffle(lower)
    return {
        "TL": lower[0],
        "TR": lower[1],
        "BL": lower[2],
        "BR": br_count,
        "BG": bg_count,
    }


def _gen_output_grid(
    shape: tuple[int, int],
    bg: int,
    fg: dict[str, int],
    min_fill: int,
    max_fill: int,
) -> np.ndarray | None:
    n_cells = shape[0] * shape[1]
    counts = _gen_output_counts(n_cells, min_fill, max_fill)
    if counts is None:
        return None

    out = np.full(shape, bg, dtype=int)
    free = np.ones(shape, dtype=bool)

    br_mask = _gen_blob_mask(
        shape,
        counts["BR"],
        int(np.random.randint(1, min(MAX_PANEL_COMPONENTS, counts["BR"]) + 1)),
        allowed=free,
    )
    if br_mask is None:
        return None
    out[br_mask] = fg["BR"]
    free &= ~br_mask

    items = [(name, counts[name]) for name in LOWER_PANEL_NAMES]
    items.sort(key=lambda item: item[1], reverse=True)
    if np.random.random() < LOWER_ORDER_SHUFFLE_PROB:
        np.random.shuffle(items)

    for name, count in items:
        max_components = min(LOWER_OUTPUT_MAX_COMPONENTS, count)
        mask = _gen_blob_mask(
            shape,
            count,
            int(np.random.randint(1, max_components + 1)),
            allowed=free,
        )
        if mask is None:
            return None
        out[mask] = fg[name]
        free &= ~mask

    return out


def _gen_layer_masks(
    out: np.ndarray,
    fg: dict[str, int],
    min_fill: int,
    max_fill: int,
) -> dict[str, np.ndarray] | None:
    masks: dict[str, np.ndarray] = {}
    br_mask = out == fg["BR"]
    if not (min_fill <= int(br_mask.sum()) <= max_fill):
        return None
    if _gen_component_count(br_mask) > MAX_PANEL_COMPONENTS:
        return None
    masks["BR"] = br_mask

    anchor = np.zeros(out.shape, dtype=bool)
    lower_visible = (out == fg["TL"]) | (out == fg["TR"]) | (out == fg["BL"])
    picked = _gen_pick(br_mask & ~_gen_neighbor_mask(lower_visible))
    if picked is None:
        picked = _gen_pick(br_mask)
    if picked is None:
        return None
    anchor[picked] = True

    higher = masks["BR"].copy()
    for name in LOWER_PANEL_NAMES:
        visible = out == fg[name]

        base_count = int(visible.sum()) + int(anchor.sum())
        available = int((higher & ~visible & ~anchor).sum())
        low = max(min_fill, base_count)
        high = min(max_fill, base_count + available)
        if high < low:
            return None
        target = int(np.random.randint(low, high + 1))
        grown = _gen_grow_hidden_to_count(
            visible,
            anchor,
            higher,
            target,
            max_components=MAX_PANEL_COMPONENTS,
        )
        if grown is None:
            return None
        if not (min_fill <= int(grown.sum()) <= max_fill):
            return None
        if _gen_component_count(grown) > MAX_PANEL_COMPONENTS:
            return None
        masks[name] = grown
        higher |= grown

    return masks


def _gen_assemble_input(
    shape: tuple[int, int],
    bg: int,
    cross: int,
    fg: dict[str, int],
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    p_h, p_w = shape
    grid = np.full((2 * p_h + 1, 2 * p_w + 1), cross, dtype=int)
    panels = {}
    for name in PANEL_NAMES:
        panel = np.full(shape, bg, dtype=int)
        panel[masks[name]] = fg[name]
        panels[name] = panel

    grid[:p_h, :p_w] = panels["TL"]
    grid[:p_h, p_w + 1 :] = panels["TR"]
    grid[p_h + 1 :, :p_w] = panels["BL"]
    grid[p_h + 1 :, p_w + 1 :] = panels["BR"]
    return grid


def _gen_panel_shape_and_fill() -> tuple[tuple[int, int], int, int]:
    p_h = int(np.random.randint(PANEL_SIDE_MIN, PANEL_SIDE_MAX_EXCLUSIVE))
    p_w = int(np.random.randint(PANEL_SIDE_MIN, PANEL_SIDE_MAX_EXCLUSIVE))
    n_cells = p_h * p_w
    min_fill = int(np.ceil(MIN_FILL_RATIO * n_cells))
    max_fill = int(np.floor(MAX_FILL_RATIO * n_cells))
    return (p_h, p_w), min_fill, max_fill


def _gen_palette() -> tuple[int, int, dict[str, int]]:
    colors = [int(c) for c in np.random.permutation(PALETTE_SIZE)[:INPUT_COLOR_COUNT]]
    bg, cross, tl_fg, tr_fg, bl_fg, br_fg = colors
    fg = dict(zip(PANEL_NAMES, (tl_fg, tr_fg, bl_fg, br_fg)))
    return bg, cross, fg


def _gen_attempt() -> np.ndarray | None:
    shape, min_fill, max_fill = _gen_panel_shape_and_fill()
    bg, cross, fg = _gen_palette()

    out = _gen_output_grid(shape, bg, fg, min_fill, max_fill)
    if out is None:
        return None

    masks = _gen_layer_masks(out, fg, min_fill, max_fill)
    if masks is None:
        return None

    inp = _gen_assemble_input(shape, bg, cross, fg, masks)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e99362f0 input")
