import numpy as np


BG = 0
SEP = 5
RANK_COLORS = [2, 3, 1, 4, 6, 8, 9, 7]
FG_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_RNG = np.random.default_rng()
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "panel must contain a pattern"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    cropped = _normalize_mask(mask)
    keys = []
    for k in range(4):
        rot = np.rot90(cropped, k)
        coords = np.argwhere(rot)
        key = tuple(tuple(map(int, rc)) for rc in coords)
        keys.append(key)
    return min(keys)


def _separator_columns(grid: np.ndarray) -> list[int]:
    cols = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) == SEP and np.all(col == SEP):
            cols.append(c)
    return cols


def _panel_spans(width: int, sep_cols: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for c in sep_cols + [width]:
        if c > start:
            spans.append((start, c))
        start = c + 1
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_cols = _separator_columns(g)
    assert sep_cols, "expected grey separator columns"
    spans = _panel_spans(g.shape[1], sep_cols)
    assert len(spans) >= 2, "expected multiple panels"

    widths = {c1 - c0 for c0, c1 in spans}
    assert len(widths) == 1, "panels must have equal width"

    panel_info = []
    for idx, (c0, c1) in enumerate(spans):
        panel = g[:, c0:c1]
        fg_mask = panel != BG
        fg_mask &= panel != SEP
        assert fg_mask.any(), "each panel must contain one pattern"
        colors = set(map(int, panel[fg_mask]))
        assert len(colors) == 1, "pattern must be monochrome within a panel"
        color = colors.pop()
        norm = _normalize_mask(fg_mask)
        key = _canonical_key(norm)
        panel_info.append(
            {
                "index": idx,
                "c0": c0,
                "c1": c1,
                "color": color,
                "mask": fg_mask,
                "norm": norm,
                "key": key,
            }
        )

    counts: dict[tuple[tuple[int, int], ...], int] = {}
    first_seen: dict[tuple[tuple[int, int], ...], int] = {}
    for info in panel_info:
        key = info["key"]
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, info["index"])

    ranked_keys = sorted(counts, key=lambda key: (-counts[key], first_seen[key]))
    assert len(ranked_keys) <= len(RANK_COLORS), "too many distinct shapes for rank palette"
    color_by_key = {key: RANK_COLORS[i] for i, key in enumerate(ranked_keys)}

    out = np.full_like(g, BG)
    for c in sep_cols:
        out[:, c] = SEP
    for info in panel_info:
        panel = out[:, info["c0"] : info["c1"]]
        panel[info["mask"]] = color_by_key[info["key"]]
    return out


_BASE_SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
]


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = tuple(tuple(map(int, rc)) for rc in np.argwhere(rot))
        if key not in seen:
            variants.append(rot.copy())
            seen.add(key)
    return variants


def _choose_frequencies(kinds: int, panels: int) -> list[int] | None:
    if panels < kinds:
        return None
    remaining = panels
    counts = [1] * kinds
    remaining -= kinds
    while remaining > 0:
        idx = int(_RNG.integers(kinds))
        counts[idx] += 1
        remaining -= 1
    _RNG.shuffle(counts)
    if len(set(counts)) == 1:
        i = int(_RNG.integers(kinds))
        j = (i + 1) % kinds
        if counts[j] > 1:
            counts[i] += 1
            counts[j] -= 1
    return counts


def _sample_shape_pool(kinds: int) -> list[np.ndarray]:
    ids = _RNG.choice(len(_BASE_SHAPES), size=kinds, replace=False)
    return [_BASE_SHAPES[int(i)].copy() for i in ids]


def _build_panel_sequence(shape_pool: list[np.ndarray], counts: list[int]) -> list[np.ndarray]:
    order = list(range(len(shape_pool)))
    _RNG.shuffle(order)
    panels = []
    for idx in order:
        for _ in range(counts[idx]):
            variants = _rotations(shape_pool[idx])
            panels.append(variants[int(_RNG.integers(len(variants)))])
    _RNG.shuffle(panels)
    return panels


def _render_panel(panel_h: int, panel_w: int, mask: np.ndarray, color: int) -> np.ndarray | None:
    h, w = mask.shape
    if h > panel_h - 2 or w > panel_w - 2:
        return None
    r0 = int(_RNG.integers(1, panel_h - h))
    c0 = int(_RNG.integers(1, panel_w - w))
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    panel[r0 : r0 + h, c0 : c0 + w][mask] = color
    return panel


def _gen_attempt() -> np.ndarray | None:
    panel_count = int(_RNG.integers(4, 9))
    kind_count = int(_RNG.integers(2, min(5, panel_count) + 1))
    counts = _choose_frequencies(kind_count, panel_count)
    if counts is None or max(counts) == 1:
        return None

    shape_pool = _sample_shape_pool(kind_count)
    sequence = _build_panel_sequence(shape_pool, counts)
    if len(sequence) != panel_count:
        return None

    max_h = max(mask.shape[0] for mask in sequence)
    max_w = max(mask.shape[1] for mask in sequence)
    panel_h = int(_RNG.integers(max_h + 3, max(max_h + 4, 11)))
    panel_w = int(_RNG.integers(max_w + 3, max(max_w + 4, 10)))

    used_colors = _RNG.choice(FG_COLORS, size=panel_count, replace=panel_count > len(FG_COLORS))
    rendered = []
    for mask, color in zip(sequence, used_colors):
        panel = _render_panel(panel_h, panel_w, mask, int(color))
        if panel is None:
            return None
        rendered.append(panel)

    width = panel_count * panel_w + (panel_count - 1)
    grid = np.full((panel_h, width), BG, dtype=int)
    c = 0
    for i, panel in enumerate(rendered):
        grid[:, c : c + panel_w] = panel
        c += panel_w
        if i < panel_count - 1:
            grid[:, c] = SEP
            c += 1

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if len({hash(_canonical_key(mask)) for mask in sequence}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
