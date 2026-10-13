import numpy as np


BG = 0
FRAME_SIZES = (6, 5)
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray, tuple[int, int, int, int]]:
    best_bbox = None
    best_score = None
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if c1 - c0 + 1 != 2 or r1 - r0 + 1 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg):
            continue
        score = (r1 + c1, r1, c1, -(r1 - r0))
        if best_score is None or score > best_score:
            best_score = score
            best_bbox = (r0, c0, r1, c1)

    assert best_bbox is not None, "legend block not found"
    r0, c0, r1, c1 = best_bbox
    mapping: dict[int, int] = {}
    for rr in range(r1 - r0 + 1):
        src = int(grid[r0 + rr, c0])
        dst = int(grid[r0 + rr, c0 + 1])
        assert src != bg and dst != bg
        assert src not in mapping, "legend source colors must be unique"
        mapping[src] = dst

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mapping, legend_mask, best_bbox


def _parse_frame(patch: np.ndarray, bg: int) -> dict[str, object] | None:
    h, w = patch.shape
    if h != w or h not in FRAME_SIZES:
        return None

    border_mask = np.zeros_like(patch, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    border_vals = patch[border_mask]
    if np.count_nonzero(border_vals == bg) != 1:
        return None

    border_colors = sorted({int(v) for v in border_vals if int(v) != bg})
    if len(border_colors) != 1:
        return None
    border_color = border_colors[0]

    interior = patch[1:-1, 1:-1]
    if interior.shape[0] not in (3, 4) or interior.shape[0] != interior.shape[1]:
        return None
    if np.any(interior == bg):
        return None

    interior_colors = sorted(int(v) for v in np.unique(interior))
    if len(interior_colors) != 2 or border_color in interior_colors:
        return None

    gap = np.argwhere(border_mask & (patch == bg))
    gr, gc = map(int, gap[0])
    if gr in (0, h - 1) and gc in (0, w - 1):
        return None
    if gr == 0:
        side = "top"
    elif gc == w - 1:
        side = "right"
    elif gr == h - 1:
        side = "bottom"
    else:
        side = "left"

    return {
        "side": side,
        "size": h,
        "border_color": border_color,
        "interior": interior.copy(),
    }


def _find_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    used = np.zeros_like(grid, dtype=bool)
    frames: list[dict[str, object]] = []

    for size in FRAME_SIZES:
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                if legend_mask[r0 : r0 + size, c0 : c0 + size].any():
                    continue
                if used[r0 : r0 + size, c0 : c0 + size].any():
                    continue
                parsed = _parse_frame(grid[r0 : r0 + size, c0 : c0 + size], bg)
                if parsed is None:
                    continue
                frames.append({"r0": r0, "c0": c0, **parsed})
                used[r0 : r0 + size, c0 : c0 + size] = True

    assert frames, "no frames found"
    frames.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    return frames


def _recolor(pattern: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = pattern.copy()
    for src, dst in mapping.items():
        out[pattern == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    mapping, legend_mask, _ = _find_legend(g, bg)
    frames = _find_frames(g, bg, legend_mask)

    out = g.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        size = int(frame["size"])
        turns = SIDE_TO_TURNS[str(frame["side"])]
        interior = np.asarray(frame["interior"], dtype=int)
        rotated = np.rot90(interior, -turns)
        out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = _recolor(rotated, mapping)
    return out


def _mark_occupied(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    r1 = min(mask.shape[0], r0 + h + pad)
    c1 = min(mask.shape[1], c0 + w + pad)
    mask[max(0, r0 - pad) : r1, max(0, c0 - pad) : c1] = True


def _random_pair(pool: list[int], rng: np.random.Generator) -> tuple[int, int]:
    chosen = rng.choice(pool, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _make_pattern(size: int, colors: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    a, b = colors
    total = size * size
    for _ in range(200):
        prob = float(rng.uniform(0.3, 0.7))
        pick_a = rng.random((size, size)) < prob
        count_a = int(pick_a.sum())
        count_b = total - count_a
        if count_a < 2 or count_b < 2:
            continue
        pattern = np.where(pick_a, a, b).astype(int)
        rotations = {np.rot90(pattern, k).tobytes() for k in range(4)}
        if len(rotations) == 4:
            return pattern

    pattern = np.full((size, size), b, dtype=int)
    pattern[0, 0] = a
    pattern[min(1, size - 1), -1] = a
    pattern[-1, max(0, size - 2)] = a
    pattern[size // 2, (size // 2 + 1) % size] = a
    assert len({np.rot90(pattern, k).tobytes() for k in range(4)}) == 4
    return pattern


def _gap_cell(side: str, size: int, rng: np.random.Generator) -> tuple[int, int]:
    pos = int(rng.integers(1, size - 1))
    if side == "top":
        return 0, pos
    if side == "right":
        return pos, size - 1
    if side == "bottom":
        return size - 1, pos
    return pos, 0


def _frame_patch(
    border_color: int,
    pattern: np.ndarray,
    side: str,
    rng: np.random.Generator,
) -> np.ndarray:
    size = pattern.shape[0] + 2
    patch = np.full((size, size), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    gr, gc = _gap_cell(side, size, rng)
    patch[gr, gc] = BG
    patch[1:-1, 1:-1] = pattern
    return patch


def _choose_pattern_colors(
    mode: str,
    mapped: list[int],
    unmapped: list[int],
    rng: np.random.Generator,
) -> tuple[int, int]:
    if mode == "mapped":
        return _random_pair(mapped, rng)
    if mode == "mixed":
        src = int(rng.choice(mapped))
        other = int(rng.choice(unmapped))
        return (src, other) if rng.random() < 0.5 else (other, src)
    return _random_pair(unmapped, rng)


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 : r0 + ph, c0 : c0 + pw].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = candidates[int(rng.integers(len(candidates)))]
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    _mark_occupied(occupied, r0, c0, ph, pw, pad=1)
    return r0, c0


def _expected_output(grid: np.ndarray, frames: list[dict[str, object]], mapping: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        pattern = np.asarray(frame["pattern"], dtype=int)
        turns = SIDE_TO_TURNS[str(frame["side"])]
        rotated = np.rot90(pattern, -turns)
        out[r0 + 1 : r0 + 1 + pattern.shape[0], c0 + 1 : c0 + 1 + pattern.shape[1]] = _recolor(rotated, mapping)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.integers(2, 6))
    legend_rows = int(rng.integers(2, 5))
    min_side = 17 + frame_count
    h = int(rng.integers(min_side, 27))
    w = int(rng.integers(min_side, 27))

    palette = rng.permutation(np.arange(1, 10)).tolist()
    mapped = [int(v) for v in palette[:legend_rows]]
    targets = [int(v) for v in palette[legend_rows : 2 * legend_rows]]
    mapping = dict(zip(mapped, targets))
    unmapped = [color for color in range(1, 10) if color not in mapping]

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    legend = np.column_stack([mapped, targets]).astype(int)
    bottom_margin = int(rng.integers(0, 2))
    right_margin = int(rng.integers(0, 2))
    lr0 = h - legend.shape[0] - bottom_margin
    lc0 = w - legend.shape[1] - right_margin
    if lr0 < 1 or lc0 < 1:
        return None
    grid[lr0 : lr0 + legend.shape[0], lc0 : lc0 + legend.shape[1]] = legend
    _mark_occupied(occupied, lr0, lc0, legend.shape[0], legend.shape[1], pad=1)

    if frame_count <= 4:
        sides = [str(v) for v in rng.permutation(np.array(SIDES, dtype=object))[:frame_count]]
    else:
        sides = [str(v) for v in rng.permutation(np.array(SIDES, dtype=object))]
        sides.append(str(rng.choice(np.array(SIDES, dtype=object))))
    rng.shuffle(sides)

    modes = ["mapped", "mixed"]
    for _ in range(frame_count - 2):
        modes.append(str(rng.choice(np.array(["mapped", "mixed", "unmapped"], dtype=object))))
    rng.shuffle(modes)

    frame_specs: list[dict[str, object]] = []
    for idx in range(frame_count):
        outer_size = int(rng.choice(np.array(FRAME_SIZES)))
        interior_size = outer_size - 2
        mode = modes[idx]
        colors = _choose_pattern_colors(mode, mapped, unmapped, rng)
        border_choices = [color for color in range(1, 10) if color not in colors]
        border_color = int(rng.choice(border_choices))
        pattern = _make_pattern(interior_size, colors, rng)
        patch = _frame_patch(border_color, pattern, sides[idx], rng)
        frame_specs.append(
            {
                "side": sides[idx],
                "pattern": pattern,
                "patch": patch,
                "border_color": border_color,
            }
        )

    frame_specs.sort(key=lambda item: int(np.asarray(item["patch"]).shape[0]), reverse=True)
    placed_frames: list[dict[str, object]] = []
    for spec in frame_specs:
        patch = np.asarray(spec["patch"], dtype=int)
        pos = _place_patch(grid, occupied, patch, rng)
        if pos is None:
            return None
        placed_frames.append({**spec, "r0": pos[0], "c0": pos[1]})

    expected = _expected_output(grid, placed_frames, mapping)
    parsed_mapping, legend_mask, _ = _find_legend(grid, BG)
    if parsed_mapping != mapping:
        return None
    found_frames = _find_frames(grid, BG, legend_mask)
    if len(found_frames) != frame_count:
        return None
    found_signature = {
        (int(frame["r0"]), int(frame["c0"]), int(frame["size"]), str(frame["side"]))
        for frame in found_frames
    }
    expected_signature = {
        (int(frame["r0"]), int(frame["c0"]), int(np.asarray(frame["patch"]).shape[0]), str(frame["side"]))
        for frame in placed_frames
    }
    if found_signature != expected_signature:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
