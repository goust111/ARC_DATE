import numpy as np


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_full_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _extract_instance(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    bg = _most_common_color(g)
    comps = _components(g != bg)
    payload_candidates = [comp for comp in comps if comp.shape[0] > 1 and _is_full_rectangle(comp)]
    assert len(payload_candidates) == 1, "expected one solid rectangular payload"

    payload_coords = payload_candidates[0]
    r0, c0, r1, c1 = _bbox(payload_coords)
    payload = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all(payload != bg), "payload must be fully filled"

    side_coords: list[tuple[int, int]] = []
    top_bottom_coords: list[tuple[int, int]] = []
    for comp in comps:
        comp_list = [tuple(map(int, rc)) for rc in comp.tolist()]
        if comp is payload_coords:
            continue
        assert len(comp_list) == 1, "all non-payload foreground objects must be isolated singletons"
        r, c = comp_list[0]
        if r0 <= r <= r1 and (c < c0 or c > c1):
            side_coords.append((r, c))
        elif c0 <= c <= c1 and (r < r0 or r > r1):
            top_bottom_coords.append((r, c))
        else:
            raise AssertionError("singleton signals must align with payload rows or payload columns")

    assert side_coords and top_bottom_coords, "expected both signal groups"

    side_colors = {int(g[r, c]) for r, c in side_coords}
    top_bottom_colors = {int(g[r, c]) for r, c in top_bottom_coords}
    assert len(side_colors) == 1 and len(top_bottom_colors) == 1, "each signal group must use one color"
    assert side_colors != top_bottom_colors, "the two signal colors must be distinct"

    payload_colors = set(map(int, np.unique(payload)))
    assert not (payload_colors & side_colors) and not (payload_colors & top_bottom_colors), (
        "payload must exclude both signal colors"
    )

    # Every signal pixel must be isolated from all other foreground pixels.
    fg = g != bg
    for r, c in side_coords + top_bottom_coords:
        neighbors = 0
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and fg[nr, nc]:
                neighbors += 1
        assert neighbors == 0, "signal pixels must be isolated in the full foreground mask"

    down_shift = len(side_coords)
    right_shift = len(top_bottom_coords)
    return payload, (r0, c0, r1, c1), down_shift, right_shift


def _has_cyclic_symmetry(payload: np.ndarray) -> bool:
    h, w = payload.shape
    for dr in range(h):
        for dc in range(w):
            if dr == 0 and dc == 0:
                continue
            if np.array_equal(np.roll(payload, shift=(dr, dc), axis=(0, 1)), payload):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the filled payload rectangle, then cyclically shift it by counts encoded as isolated noise pixels.

    Input structure:
    - One non-background object is a fully filled multicolor rectangle, the payload.
    - All other non-background objects are isolated single pixels of exactly two signal colors.
    - One signal color appears only in side strips aligned with the payload rows, and the other appears only in top/bottom strips aligned with the payload columns.

    Rule:
    - Crop out the payload rectangle.
    - Count the side-strip signal pixels and roll the payload downward by that count modulo the payload height.
    - Count the top/bottom-strip signal pixels and roll the payload rightward by that count modulo the payload width.

    Key insight:
    - The scattered pixels are not decoration to remove; their populations encode the two cyclic shift amounts.
    """
    payload, _bbox_payload, down_shift, right_shift = _extract_instance(grid)
    return np.roll(payload, shift=(down_shift, right_shift), axis=(0, 1))


def _make_payload(height: int, width: int, palette: list[int]) -> np.ndarray:
    num_colors = int(np.random.randint(3, min(6, len(palette)) + 1))
    colors = [int(c) for c in np.random.choice(palette, size=num_colors, replace=False)]

    seed_count = num_colors + int(np.random.randint(0, 3))
    seed_positions = set()
    while len(seed_positions) < seed_count:
        seed_positions.add((int(np.random.randint(0, height)), int(np.random.randint(0, width))))
    seed_positions = list(seed_positions)

    seed_colors = [colors[i % num_colors] for i in range(seed_count)]
    np.random.shuffle(seed_colors)
    payload = np.zeros((height, width), dtype=int)

    for r in range(height):
        for c in range(width):
            best_idx = None
            best_key = None
            for idx, (sr, sc) in enumerate(seed_positions):
                key = (abs(r - sr) + abs(c - sc), idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            payload[r, c] = seed_colors[int(best_idx)]

    extra_mutations = int(np.random.randint(1, max(2, (height * width) // 6)))
    for _ in range(extra_mutations):
        rr = int(np.random.randint(0, height))
        cc = int(np.random.randint(0, width))
        payload[rr, cc] = int(np.random.choice(colors))

    used = set(map(int, np.unique(payload)))
    if len(used) < 3:
        return _make_payload(height, width, palette)
    return payload


def _checker_candidates(rows: list[int], cols: list[int]) -> list[tuple[int, int]]:
    if not rows or not cols:
        return []
    row_parity = int(np.random.randint(0, 2))
    col_parity = int(np.random.randint(0, 2))
    candidates = [(r, c) for r in rows for c in cols if r % 2 == row_parity and c % 2 == col_parity]
    np.random.shuffle(candidates)
    return candidates


def _choose_signal_count(capacity: int, modulus: int) -> int:
    upper = min(capacity, max(modulus + 4, 2 * modulus))
    choices = [n for n in range(1, upper + 1) if n % modulus != 0]
    assert choices, "need a non-zero shift modulo the payload size"
    return int(np.random.choice(choices))


def _semantic_checks(grid: np.ndarray) -> None:
    payload, _bbox_payload, down_shift, right_shift = _extract_instance(grid)
    h, w = payload.shape
    assert down_shift % h != 0 and right_shift % w != 0, "both shifts must be non-trivial modulo payload size"
    assert not _has_cyclic_symmetry(payload), "payload must not be cyclically periodic"

    rolled = np.roll(payload, shift=(down_shift, right_shift), axis=(0, 1))
    assert not np.array_equal(rolled, payload), "full transform must change the payload"
    assert not np.array_equal(rolled, np.roll(payload, shift=(down_shift, 0), axis=(0, 1))), (
        "horizontal count must affect the result"
    )
    assert not np.array_equal(rolled, np.roll(payload, shift=(0, right_shift), axis=(0, 1))), (
        "vertical count must affect the result"
    )


def generate_input() -> np.ndarray:
    for _ in range(400):
        payload_h = int(np.random.randint(4, 9))
        payload_w = int(np.random.randint(4, 9))

        top_margin = int(np.random.randint(3, 8))
        bottom_margin = int(np.random.randint(3, 8))
        left_margin = int(np.random.randint(3, 8))
        right_margin = int(np.random.randint(3, 8))

        height = top_margin + payload_h + bottom_margin
        width = left_margin + payload_w + right_margin
        if height > 30 or width > 30:
            continue

        bg = int(np.random.randint(0, 10))
        palette = [c for c in range(10) if c != bg]
        signal_colors = [int(c) for c in np.random.choice(palette, size=2, replace=False)]
        payload_palette = [c for c in palette if c not in signal_colors]
        if len(payload_palette) < 3:
            continue

        payload = _make_payload(payload_h, payload_w, payload_palette)
        if _has_cyclic_symmetry(payload):
            continue

        grid = np.full((height, width), bg, dtype=int)
        r0, c0 = top_margin, left_margin
        r1, c1 = r0 + payload_h - 1, c0 + payload_w - 1
        grid[r0 : r1 + 1, c0 : c1 + 1] = payload

        side_rows = list(range(r0, r1 + 1))
        side_cols = list(range(0, max(0, c0 - 1))) + list(range(c1 + 2, width))
        tb_rows = list(range(0, max(0, r0 - 1))) + list(range(r1 + 2, height))
        tb_cols = list(range(c0, c1 + 1))

        side_candidates = _checker_candidates(side_rows, side_cols)
        tb_candidates = _checker_candidates(tb_rows, tb_cols)
        if len(side_candidates) < 2 or len(tb_candidates) < 2:
            continue

        down_count = _choose_signal_count(len(side_candidates), payload_h)
        right_count = _choose_signal_count(len(tb_candidates), payload_w)

        side_pick = side_candidates[:down_count]
        tb_pick = tb_candidates[:right_count]
        if set(side_pick) & set(tb_pick):
            continue

        valid = True
        all_fg = set(side_pick) | set(tb_pick) | {
            (rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
        }
        for r, c in side_pick + tb_pick:
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) in all_fg and (nr, nc) != (r, c):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        for r, c in side_pick:
            grid[r, c] = signal_colors[0]
        for r, c in tb_pick:
            grid[r, c] = signal_colors[1]

        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
