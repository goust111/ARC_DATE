from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4)
SLOT_ROWS = 2
SLOT_COLS = 3
SLOT_POSITIONS = [(sr, sc) for sr in range(SLOT_ROWS) for sc in range(SLOT_COLS)]
FRAME_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _line_indices(lengths: tuple[int, ...]) -> list[int]:
    out = [0]
    acc = 0
    for length in lengths:
        acc += int(length) + 1
        out.append(acc)
    return out


def _parse_frames(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _components(grid == FRAME)
    frames: list[dict[str, object]] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == FRAME
        row_lines = [r for r in range(sub.shape[0]) if bool(sub[r, :].all())]
        col_lines = [c for c in range(sub.shape[1]) if bool(sub[:, c].all())]
        if len(row_lines) != SLOT_ROWS + 1 or len(col_lines) != SLOT_COLS + 1:
            raise AssertionError("expected 2x3 lattice frames")
        if row_lines[0] != 0 or row_lines[-1] != sub.shape[0] - 1:
            raise AssertionError("frame must have top/bottom borders")
        if col_lines[0] != 0 or col_lines[-1] != sub.shape[1] - 1:
            raise AssertionError("frame must have left/right borders")
        if not all(a + 1 < b for a, b in zip(row_lines, row_lines[1:])):
            raise AssertionError("row dividers must enclose non-empty slot interiors")
        if not all(a + 1 < b for a, b in zip(col_lines, col_lines[1:])):
            raise AssertionError("col dividers must enclose non-empty slot interiors")

        expected = np.zeros_like(sub, dtype=bool)
        expected[row_lines, :] = True
        expected[:, col_lines] = True
        if not np.array_equal(sub, expected):
            raise AssertionError("frame component must be a complete rectangular lattice")

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "rows": [r0 + r for r in row_lines],
                "cols": [c0 + c for c in col_lines],
            }
        )

    frames.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    if len(frames) != 3:
        raise AssertionError("expected exactly three frames")
    return frames


def _slot_slice(frame: dict[str, object], sr: int, sc: int) -> tuple[slice, slice]:
    rows = frame["rows"]
    cols = frame["cols"]
    return slice(rows[sr] + 1, rows[sr + 1]), slice(cols[sc] + 1, cols[sc + 1])


def _drop_slot(slot: np.ndarray) -> np.ndarray:
    h, w = slot.shape
    out = np.zeros_like(slot)
    for c in range(w):
        colors = [int(slot[r, c]) for r in range(h) if int(slot[r, c]) != BG]
        if colors:
            out[h - len(colors) :, c] = np.array(colors, dtype=int)
    return out


def _bottom_color(slot_after_drop: np.ndarray) -> int | None:
    vals = {int(v) for v in slot_after_drop[-1, :] if int(v) != BG}
    if len(vals) > 1:
        raise AssertionError("bottom row color must be unique after gravity")
    return next(iter(vals)) if vals else None


def _infer_colors_after_gravity(grid: np.ndarray, frames: list[dict[str, object]]) -> dict[tuple[int, int], int]:
    inferred: dict[tuple[int, int], int] = {}
    for sr, sc in SLOT_POSITIONS:
        seen: set[int] = set()
        for frame in frames:
            rs, cs = _slot_slice(frame, sr, sc)
            dropped = _drop_slot(grid[rs, cs])
            color = _bottom_color(dropped)
            if color is not None:
                seen.add(color)
        if len(seen) != 1:
            raise AssertionError("each logical slot position must infer exactly one color")
        inferred[(sr, sc)] = next(iter(seen))
    return inferred


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop sparse slot pixels inside each lattice frame, infer each logical slot's color from the post-gravity bottom row, then fill every slot.

    Input structure:
    - The grid has a black background and exactly three disconnected gray rectangular lattice frames.
    - Every frame is a one-pixel-thick 2x3 slot layout: two slot rows and three slot columns.
    - Some slot interiors contain one or two colored pixels from red, blue, green, or yellow; many slots are empty.

    Rule:
    - Inside each slot independently, apply downward gravity to the colored pixels while keeping frame lines and background fixed.
    - For each logical slot position `(slot_row, slot_col)`, inspect the three corresponding slots after gravity and read the unique non-background color that reaches the slot's bottom row.
    - Fill the entire interior of every corresponding slot in all three frames with that inferred color.
    - Keep the gray frame lines and all outside background unchanged.

    Key insight:
    - The raw pixel locations are misleading; only the color that settles onto a slot's bottom row after gravity is the true cross-frame label for that logical slot.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("input must be a 2D grid")

    frames = _parse_frames(g)
    slot_colors = _infer_colors_after_gravity(g, frames)

    out = g.copy()
    for frame in frames:
        for sr, sc in SLOT_POSITIONS:
            rs, cs = _slot_slice(frame, sr, sc)
            out[rs, cs] = slot_colors[(sr, sc)]
    return out


ROW_OPTIONS = ((3, 3), (3, 4), (4, 3), (4, 4))
COL_OPTIONS = (
    (1, 1, 2),
    (1, 2, 1),
    (2, 1, 1),
    (1, 1, 3),
    (1, 3, 1),
    (3, 1, 1),
    (1, 2, 2),
    (2, 1, 2),
    (2, 2, 1),
    (1, 2, 3),
    (1, 3, 2),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
    (3, 2, 1),
    (2, 2, 2),
)


def _choose_slot_colors() -> dict[tuple[int, int], int]:
    values = [1, 2, 3, 4, random.choice(PALETTE), random.choice(PALETTE)]
    random.shuffle(values)
    colors = {pos: values[i] for i, pos in enumerate(SLOT_POSITIONS)}
    top = tuple(colors[(0, sc)] for sc in range(SLOT_COLS))
    bottom = tuple(colors[(1, sc)] for sc in range(SLOT_COLS))
    if top == bottom:
        raise ValueError("degenerate color pattern")
    return colors


def _build_frame_specs() -> list[dict[str, object]] | None:
    for _ in range(80):
        geom = []
        total_width = FRAME_GAP * 2
        for _frame_idx in range(3):
            row_heights = random.choice(ROW_OPTIONS)
            col_widths = random.choice(COL_OPTIONS)
            frame_h = sum(row_heights) + SLOT_ROWS + 1
            frame_w = sum(col_widths) + SLOT_COLS + 1
            geom.append((row_heights, col_widths, frame_h, frame_w))
            total_width += frame_w
        if total_width > 30:
            continue

        left_margin = random.randint(0, 30 - total_width)
        right_margin = 30 - total_width - left_margin
        top_margin = random.randint(0, 2)
        bottom_margin = random.randint(0, 2)
        height = top_margin + max(item[2] for item in geom) + bottom_margin
        if height > 30:
            continue

        specs: list[dict[str, object]] = []
        c = left_margin
        for row_heights, col_widths, frame_h, frame_w in geom:
            specs.append(
                {
                    "r0": top_margin,
                    "c0": c,
                    "row_heights": row_heights,
                    "col_widths": col_widths,
                    "frame_h": frame_h,
                    "frame_w": frame_w,
                    "rows": [top_margin + x for x in _line_indices(row_heights)],
                    "cols": [c + x for x in _line_indices(col_widths)],
                }
            )
            c += frame_w + FRAME_GAP
        if right_margin < 0 or c - FRAME_GAP + right_margin != 30:
            continue
        return specs
    return None


def _draw_frame(grid: np.ndarray, frame: dict[str, object]) -> None:
    rows = frame["rows"]
    cols = frame["cols"]
    grid[rows, cols[0] : cols[-1] + 1] = FRAME
    grid[rows[0] : rows[-1] + 1, cols] = FRAME


def _filled_output(specs: list[dict[str, object]], slot_colors: dict[tuple[int, int], int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame in specs:
        _draw_frame(out, frame)
        for sr, sc in SLOT_POSITIONS:
            rows = frame["rows"]
            cols = frame["cols"]
            out[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]] = slot_colors[(sr, sc)]
    return out


def _choose_providers() -> dict[tuple[int, int], list[int]]:
    positions = SLOT_POSITIONS[:]
    random.shuffle(positions)
    special_single = positions[0]
    special_multi = positions[1]

    providers: dict[tuple[int, int], list[int]] = {}
    for pos in SLOT_POSITIONS:
        if pos == special_single:
            k = 1
        elif pos == special_multi:
            k = random.choice((2, 3))
        else:
            k = random.choice((1, 2, 3))
        providers[pos] = sorted(random.sample(range(3), k))

    covered = Counter(frame for frames in providers.values() for frame in frames)
    for frame_idx in range(3):
        if covered[frame_idx] > 0:
            continue
        choices = [pos for pos in SLOT_POSITIONS if pos != special_single and frame_idx not in providers[pos]]
        if not choices:
            return {}
        pos = random.choice(choices)
        providers[pos].append(frame_idx)
        providers[pos].sort()
    return providers


def _pick_pattern_kind(h: int, w: int, *, special: str | None = None) -> str:
    if special == "airborne_distractor":
        return "airborne_distractor"
    choices = ["plain", "airborne", "double_stack"]
    if w >= 2:
        choices.append("double_split")
    if h >= 3:
        choices.append("airborne_distractor")
    return random.choice(choices)


def _make_slot_hint(h: int, w: int, true_color: int, kind: str) -> np.ndarray:
    slot = np.zeros((h, w), dtype=int)
    if kind == "plain":
        slot[random.randrange(h), random.randrange(w)] = true_color
        return slot
    if kind == "airborne":
        slot[random.randrange(h - 1), random.randrange(w)] = true_color
        return slot
    if kind == "double_stack":
        c = random.randrange(w)
        r_top, r_low = sorted(random.sample(range(h), 2))
        slot[r_top, c] = true_color
        slot[r_low, c] = true_color
        return slot
    if kind == "double_split":
        cols = random.sample(range(w), 2)
        slot[random.randrange(h), cols[0]] = true_color
        slot[random.randrange(h), cols[1]] = true_color
        return slot
    if kind == "airborne_distractor":
        distractor = random.choice([color for color in PALETTE if color != true_color])
        c = random.randrange(w)
        low = random.randrange(1, h - 1)
        high = random.randrange(low)
        slot[low, c] = true_color
        slot[high, c] = distractor
        return slot
    raise ValueError(f"unknown slot hint kind: {kind}")


def _place_hints(
    specs: list[dict[str, object]],
    slot_colors: dict[tuple[int, int], int],
    providers: dict[tuple[int, int], list[int]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[tuple[int, int, int], str]] | None:
    inp = np.zeros(shape, dtype=int)
    kinds: dict[tuple[int, int, int], str] = {}
    for frame in specs:
        _draw_frame(inp, frame)

    special_pos = None
    for pos, frames in providers.items():
        if len(frames) == 1:
            special_pos = pos
            break
    if special_pos is None:
        return None

    made_double_split = False
    for sr, sc in SLOT_POSITIONS:
        for frame_idx in providers[(sr, sc)]:
            frame = specs[frame_idx]
            rows = frame["rows"]
            cols = frame["cols"]
            h = rows[sr + 1] - rows[sr] - 1
            w = cols[sc + 1] - cols[sc] - 1

            special = "airborne_distractor" if (sr, sc) == special_pos else None
            kind = _pick_pattern_kind(h, w, special=special)
            if not made_double_split and w >= 2 and (sr, sc) != special_pos and random.random() < 0.55:
                kind = "double_split"
                made_double_split = True

            hint = _make_slot_hint(h, w, slot_colors[(sr, sc)], kind)
            rs = slice(rows[sr] + 1, rows[sr + 1])
            cs = slice(cols[sc] + 1, cols[sc + 1])
            inp[rs, cs] = hint
            kinds[(frame_idx, sr, sc)] = kind

    return inp, kinds


def _semantic_profile(grid: np.ndarray) -> dict[str, object]:
    frames = _parse_frames(grid)
    coverage = {pos: 0 for pos in SLOT_POSITIONS}
    inferred_sets = {pos: set() for pos in SLOT_POSITIONS}
    frame_hint_counts = [0, 0, 0]
    single_provider_multicolor = False
    bottom_empty_then_filled = False
    multi_provider_exists = False

    per_position_features: dict[tuple[int, int], list[dict[str, object]]] = {pos: [] for pos in SLOT_POSITIONS}
    for frame_idx, frame in enumerate(frames):
        for sr, sc in SLOT_POSITIONS:
            rs, cs = _slot_slice(frame, sr, sc)
            raw = grid[rs, cs]
            dropped = _drop_slot(raw)
            raw_colors = {int(v) for v in raw.ravel() if int(v) != BG}
            bottom_raw = {int(v) for v in raw[-1, :] if int(v) != BG}
            bottom_drop = {int(v) for v in dropped[-1, :] if int(v) != BG}

            if bottom_drop:
                frame_hint_counts[frame_idx] += 1
                coverage[(sr, sc)] += 1
                inferred_sets[(sr, sc)].update(bottom_drop)
            if not bottom_raw and bottom_drop:
                bottom_empty_then_filled = True
            per_position_features[(sr, sc)].append(
                {
                    "raw_colors": raw_colors,
                    "bottom_drop": bottom_drop,
                }
            )

    for pos in SLOT_POSITIONS:
        if coverage[pos] >= 2:
            multi_provider_exists = True
        if coverage[pos] == 1:
            for feat in per_position_features[pos]:
                if len(feat["raw_colors"]) >= 2 and feat["bottom_drop"]:
                    single_provider_multicolor = True

    return {
        "coverage": coverage,
        "inferred_sets": inferred_sets,
        "frame_hint_counts": frame_hint_counts,
        "single_provider_multicolor": single_provider_multicolor,
        "bottom_empty_then_filled": bottom_empty_then_filled,
        "multi_provider_exists": multi_provider_exists,
    }


def _validate_generated_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    try:
        frames = _parse_frames(inp)
        if len(frames) != 3:
            return False
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    profile = _semantic_profile(inp)
    if any(count == 0 for count in profile["frame_hint_counts"]):
        return False
    if any(profile["coverage"][pos] == 0 for pos in SLOT_POSITIONS):
        return False
    if any(len(profile["inferred_sets"][pos]) != 1 for pos in SLOT_POSITIONS):
        return False
    if not profile["single_provider_multicolor"]:
        return False
    if not profile["bottom_empty_then_filled"]:
        return False
    if not profile["multi_provider_exists"]:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        slot_colors = _choose_slot_colors()
    except ValueError:
        return None
    specs = _build_frame_specs()
    if specs is None:
        return None

    providers = _choose_providers()
    if not providers:
        return None

    shape = (
        max(int(frame["rows"][-1]) for frame in specs) + 1,
        max(int(frame["cols"][-1]) for frame in specs) + 1,
    )
    placed = _place_hints(specs, slot_colors, providers, shape)
    if placed is None:
        return None
    inp, _kinds = placed
    out = _filled_output(specs, slot_colors, shape)

    if not _validate_generated_pair(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
