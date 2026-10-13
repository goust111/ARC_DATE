import itertools
from collections import Counter, deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend(grid: np.ndarray) -> dict[int, int]:
    top = grid[-2]
    bottom = grid[-1]
    mapping: dict[int, int] = {}
    for c in range(grid.shape[1]):
        border = int(top[c])
        fill = int(bottom[c])
        if border == 0 and fill == 0:
            continue
        assert border != 0 and fill != 0, "legend entries must be vertically aligned"
        assert border not in mapping, "legend border colors must be unique"
        mapping[border] = fill
    assert len(mapping) >= 2
    return mapping


def _extract_sprites(body: np.ndarray) -> list[dict]:
    sprites = []
    for comp in _components_mask(body != 0):
        r0, c0, r1, c1 = _bbox_coords(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h not in (3, 4):
            continue
        colors = {int(body[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        patch = body[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros((h, h), dtype=int)
        expected[0, :] = color
        expected[-1, :] = color
        expected[:, 0] = color
        expected[:, -1] = color
        if not np.array_equal(patch, expected):
            continue
        sprites.append(
            {
                "r0": r0,
                "c0": c0,
                "size": h,
                "color": color,
                "center": (r0 + (h - 1) / 2.0, c0 + (h - 1) / 2.0),
            }
        )
    assert sprites, "no hollow sprites found"
    return sprites


def _axis_support_counts(indices: list[tuple[int, int]], axis: str) -> dict[int, int]:
    assert axis in {"h", "v"}
    counts: Counter[int] = Counter()
    for r1, c1 in indices:
        for r2, c2 in indices:
            if axis == "h":
                if r1 == r2 and c2 > c1:
                    counts[c2 - c1] += 1
            else:
                if c1 == c2 and r2 > r1:
                    counts[r2 - r1] += 1
    assert counts, f"no candidate {axis} shifts"
    return dict(counts)


def _axis_support(indices: list[tuple[int, int]], axis: str) -> tuple[int, dict[int, int]]:
    counts = _axis_support_counts(indices, axis)
    best = max(counts.values())
    winners = [step for step, support in counts.items() if support == best]
    assert len(winners) == 1, f"ambiguous best {axis} shift"
    return winners[0], dict(counts)


def _build_observed_color_grid(sprites: list[dict], dy: int, dx: int) -> tuple[np.ndarray, list[int], list[int], int]:
    sizes = {sprite["size"] for sprite in sprites}
    assert len(sizes) == 1
    size = next(iter(sizes))

    rows = sorted({sprite["r0"] for sprite in sprites})
    cols = sorted({sprite["c0"] for sprite in sprites})
    assert len(rows) >= 2 and len(cols) >= 2
    full_rows = list(range(rows[0], rows[-1] + 1, dy))
    full_cols = list(range(cols[0], cols[-1] + 1, dx))
    assert set(rows).issubset(full_rows)
    assert set(cols).issubset(full_cols)

    observed = np.full((len(full_rows), len(full_cols)), -1, dtype=int)
    row_to_idx = {r: i for i, r in enumerate(full_rows)}
    col_to_idx = {c: j for j, c in enumerate(full_cols)}
    for sprite in sprites:
        observed[row_to_idx[sprite["r0"]], col_to_idx[sprite["c0"]]] = sprite["color"]
    return observed, full_rows, full_cols, size


def _infer_full_color_grid(observed: np.ndarray) -> np.ndarray:
    nr, nc = observed.shape
    predictions = {}
    for pr in range(1, nr + 1):
        for pc in range(1, nc + 1):
            classes: dict[tuple[int, int], int] = {}
            ok = True
            for r in range(nr):
                for c in range(nc):
                    color = int(observed[r, c])
                    if color == -1:
                        continue
                    key = (r % pr, c % pc)
                    prev = classes.get(key)
                    if prev is None:
                        classes[key] = color
                    elif prev != color:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or len(classes) != pr * pc:
                continue
            full = np.zeros((nr, nc), dtype=int)
            for r in range(nr):
                for c in range(nc):
                    full[r, c] = classes[(r % pr, c % pc)]
            key = tuple(map(tuple, full.tolist()))
            predictions[key] = full
    assert len(predictions) == 1, "periodic color pattern is ambiguous"
    return next(iter(predictions.values()))


def _draw_sprite(out: np.ndarray, r0: int, c0: int, size: int, border: int, fill: int) -> None:
    out[r0 : r0 + size, c0 : c0 + size] = 0
    out[r0, c0 : c0 + size] = border
    out[r0 + size - 1, c0 : c0 + size] = border
    out[r0 : r0 + size, c0] = border
    out[r0 : r0 + size, c0 + size - 1] = border
    out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = fill


def _solve_core(grid: np.ndarray) -> np.ndarray:
    mapping = _parse_legend(grid)
    body = np.asarray(grid[:-2], dtype=int)
    sprites = _extract_sprites(body)

    positions = [(sprite["r0"], sprite["c0"]) for sprite in sprites]
    dx, _ = _axis_support(positions, "h")
    dy, _ = _axis_support(positions, "v")
    observed, rows, cols, size = _build_observed_color_grid(sprites, dy, dx)
    full_colors = _infer_full_color_grid(observed)

    out = np.zeros_like(body)
    for i, r0 in enumerate(rows):
        for j, c0 in enumerate(cols):
            border = int(full_colors[i, j])
            assert border in mapping, "every border color must appear in the legend"
            _draw_sprite(out, r0, c0, size, border, int(mapping[border]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a periodic array of missing hollow sprites, then fill every sprite interior from the bottom legend.

    Input structure:
    - Black background with a 3x3 or 4x4 array of same-shaped hollow square sprites in the top area.
    - Sprite top-left positions lie on an axis-aligned lattice with horizontal and vertical periods of 5 to 7 cells.
    - Some whole sprites are removed to black, while the last two rows contain a horizontal border-color to fill-color legend.

    Rule:
    - Ignore the last two legend rows when locating sprites.
    - Recover the horizontal and vertical lattice periods from the most-supported same-row and same-column sprite translations.
    - Use the visible sprites to reconstruct the full lattice of sprite positions and infer the repeating border-color pattern.
    - Draw any missing sprites with their inferred border colors, then fill the enclosed interior of every sprite using the legend mapping.
    - Return only the restored sprite field, omitting the bottom legend rows.

    Key insight:
    - The visible hollow sprites provide two independent clues: lattice repetition determines where missing sprites go and what border color class they belong to, while the separate legend determines how each recovered border color should fill its enclosed hole.
    """
    return _solve_core(np.asarray(grid, dtype=int))


def _render_output_body(
    body_shape: tuple[int, int],
    top: int,
    left: int,
    size: int,
    dy: int,
    dx: int,
    full_colors: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.zeros(body_shape, dtype=int)
    nr, nc = full_colors.shape
    for r in range(nr):
        for c in range(nc):
            _draw_sprite(out, top + r * dy, left + c * dx, size, int(full_colors[r, c]), int(mapping[int(full_colors[r, c])]))
    return out


def _render_input_body(
    body_shape: tuple[int, int],
    top: int,
    left: int,
    size: int,
    dy: int,
    dx: int,
    full_colors: np.ndarray,
    missing: set[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(body_shape, dtype=int)
    nr, nc = full_colors.shape
    for r in range(nr):
        for c in range(nc):
            if (r, c) in missing:
                continue
            border = int(full_colors[r, c])
            rr = top + r * dy
            cc = left + c * dx
            out[rr, cc : cc + size] = border
            out[rr + size - 1, cc : cc + size] = border
            out[rr : rr + size, cc] = border
            out[rr : rr + size, cc + size - 1] = border
    return out


def _choose_pattern(nr: int, nc: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    templates = [
        ("col_alt_2", 2),
        ("row_alt_2", 2),
        ("checker_2", 2),
        ("col_cycle_3", 3),
        ("diag_shift_3", 3),
    ]
    name, color_count = templates[int(rng.integers(len(templates)))]
    colors = list(rng.choice(np.arange(1, 10), size=color_count, replace=False))
    full = np.zeros((nr, nc), dtype=int)
    for r in range(nr):
        for c in range(nc):
            if name == "col_alt_2":
                idx = c % 2
            elif name == "row_alt_2":
                idx = r % 2
            elif name == "checker_2":
                idx = (r + c) % 2
            elif name == "col_cycle_3":
                idx = c % 3
            else:
                idx = (r + c) % 3
            full[r, c] = colors[idx]
    assert len(set(map(int, full.flatten()))) == color_count
    return full, colors


def _legend_mapping(border_colors: list[int], rng: np.random.Generator) -> dict[int, int]:
    remaining = [c for c in range(1, 10) if c not in border_colors]
    fill_colors = list(rng.choice(np.array(remaining), size=len(border_colors), replace=False))
    return {int(b): int(f) for b, f in zip(border_colors, fill_colors)}


def _sample_missing(nr: int, nc: int, count: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    for _ in range(200):
        rng.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return set(chosen)
    raise RuntimeError("failed to sample non-adjacent missing sprites")


def _best_shift_with_uniqueness(indices: list[tuple[int, int]], axis: str) -> tuple[int, bool]:
    counts = _axis_support_counts(indices, axis)
    best_support = max(counts.values())
    winners = [step for step, support in counts.items() if support == best_support]
    unique = len(winners) == 1
    best = min(winners)
    return best, unique


def _place_legend(full_grid: np.ndarray, mapping: dict[int, int], rng: np.random.Generator) -> None:
    keys = list(mapping.keys())
    rng.shuffle(keys)
    span = 2 * len(keys) - 1
    start = int(rng.integers(0, full_grid.shape[1] - span + 1))
    for i, border in enumerate(keys):
        c = start + 2 * i
        full_grid[-2, c] = border
        full_grid[-1, c] = mapping[border]


def _input_signature_conditions(
    body: np.ndarray,
    full_colors: np.ndarray,
    missing: set[tuple[int, int]],
    top: int,
    left: int,
    size: int,
    dy: int,
    dx: int,
    mapping: dict[int, int],
) -> bool:
    sprites = _extract_sprites(body)
    positions = [(sprite["r0"], sprite["c0"]) for sprite in sprites]
    best_dx, uniq_dx = _best_shift_with_uniqueness(positions, "h")
    best_dy, uniq_dy = _best_shift_with_uniqueness(positions, "v")
    if not (uniq_dx and uniq_dy and best_dx == dx and best_dy == dy):
        return False

    nr, nc = full_colors.shape
    if any(all((r, c) in missing for c in range(nc)) for r in range(nr)):
        return False
    if any(all((r, c) in missing for r in range(nr)) for c in range(nc)):
        return False

    observed = np.full((nr, nc), -1, dtype=int)
    for r in range(nr):
        for c in range(nc):
            if (r, c) not in missing:
                observed[r, c] = int(full_colors[r, c])
    try:
        inferred = _infer_full_color_grid(observed)
    except AssertionError:
        return False
    if not np.array_equal(inferred, full_colors):
        return False

    output_body = _render_output_body(body.shape, top, left, size, dy, dx, full_colors, mapping)
    visible_filled = np.zeros_like(body)
    for r in range(nr):
        for c in range(nc):
            if (r, c) in missing:
                continue
            border = int(full_colors[r, c])
            _draw_sprite(
                visible_filled,
                top + r * dy,
                left + c * dx,
                size,
                border,
                int(mapping[border]),
            )
    if np.array_equal(visible_filled, output_body):
        return False

    wrong_fill_mapping = {border: border for border in mapping}
    wrong_fill_output = _render_output_body(body.shape, top, left, size, dy, dx, full_colors, wrong_fill_mapping)
    if np.array_equal(wrong_fill_output, output_body):
        return False

    if any(border == fill for border, fill in mapping.items()):
        return False
    return True


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    size = int(rng.choice([3, 4]))
    nr = int(rng.choice([3, 4]))
    nc = int(rng.choice([3, 4]))
    dy = int(rng.integers(max(size + 1, 5), 8))
    dx = int(rng.integers(max(size + 1, 5), 8))

    top_pad = int(rng.integers(0, 3))
    left_pad = int(rng.integers(0, 3))
    bottom_pad = int(rng.integers(1, 3))
    right_pad = int(rng.integers(0, 3))

    span_h = size + (nr - 1) * dy
    span_w = size + (nc - 1) * dx
    body_h = top_pad + span_h + bottom_pad
    body_w = left_pad + span_w + right_pad
    if body_h + 2 > 30 or body_w > 30:
        return None

    full_colors, border_colors = _choose_pattern(nr, nc, rng)
    mapping = _legend_mapping(border_colors, rng)

    max_missing = min(4, (nr * nc + 1) // 2)
    min_missing = 2
    count = int(rng.integers(min_missing, max_missing + 1))

    for _ in range(120):
        missing = _sample_missing(nr, nc, count, rng)
        body = _render_input_body((body_h, body_w), top_pad, left_pad, size, dy, dx, full_colors, missing)
        if not _input_signature_conditions(body, full_colors, missing, top_pad, left_pad, size, dy, dx, mapping):
            continue

        full_grid = np.zeros((body_h + 2, body_w), dtype=int)
        full_grid[:-2] = body
        _place_legend(full_grid, mapping, rng)

        expected = _render_output_body((body_h, body_w), top_pad, left_pad, size, dy, dx, full_colors, mapping)
        try:
            actual = main(full_grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return full_grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic hollow-sprite puzzle")
