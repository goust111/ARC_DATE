from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from arc_utils import find_components, get_bg
except Exception:  # pragma: no cover - fallback when working directory is not ARC root
    _ROOT = Path(__file__).resolve()
    _utils_dir = None
    for parent in _ROOT.parents:
        candidate = parent / "new" / "solvers"
        if (candidate / "arc_utils.py").exists():
            _utils_dir = candidate
            break
    if _utils_dir is None:
        _utils_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(_utils_dir) not in sys.path:
        sys.path.append(str(_utils_dir))
    from arc_utils import find_components, get_bg  # type: ignore


def _layout_from_components(components: list[dict]) -> dict | None:
    """Select component group that forms a rectangular slot array."""
    by_size: dict[tuple[int, int], list[dict]] = {}
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h, w = r1 - r0 + 1, c1 - c0 + 1
        area = int(comp["area"])

        if h < 2 or w < 2 or area < 4:
            continue
        if area < (h * w) * 0.5:
            continue

        by_size.setdefault((h, w), []).append(comp)

    best: dict | None = None
    best_score = -1

    for (h, w), comps in by_size.items():
        if len(comps) < 2:
            continue

        top_lefts = [(int(item["bbox"][0]), int(item["bbox"][1])) for item in comps]
        if len(top_lefts) != len(set(top_lefts)):
            continue

        rows = sorted(set(r for r, _ in top_lefts))
        cols = sorted(set(c for _, c in top_lefts))
        if len(rows) * len(cols) != len(top_lefts):
            continue
        if len(rows) < 2 and len(cols) < 2:
            continue

        if len(rows) > 1:
            row_step = rows[1] - rows[0]
            if row_step <= 0 or any(rows[i] - rows[i - 1] != row_step for i in range(1, len(rows))):
                continue
        else:
            row_step = 0

        if len(cols) > 1:
            col_step = cols[1] - cols[0]
            if col_step <= 0 or any(cols[i] - cols[i - 1] != col_step for i in range(1, len(cols))):
                continue
        else:
            col_step = 0

        lookup = {(r, c): comp for comp, (r, c) in zip(comps, top_lefts)}
        ordered: list[dict] = []
        valid = True
        for r in rows:
            for c in cols:
                comp = lookup.get((r, c))
                if comp is None:
                    valid = False
                    break
                ordered.append(comp)
            if not valid:
                break
        if not valid:
            continue

        if len(ordered) * h * w <= best_score:
            continue

        best = {
            "height": int(h),
            "width": int(w),
            "rows": rows,
            "cols": cols,
            "row_step": int(row_step),
            "col_step": int(col_step),
            "components": ordered,
        }
        best_score = len(ordered) * h * w

    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden deviation map from a regular array of copied sprites.

    Input structure:
    - The background is the most frequent color.
    - Non-background pixels form a regular rectangular array of slot copies.
    - Each slot is a copy of one base sprite, with sparse pixel deviations.
    - There are free floating noise pixels outside the slot array.

    Rule:
    - Detect all slot components, identify their common bbox and the lattice layout.
    - For each slot-relative cell, choose the majority color across all slots as the base sprite color.
    - Emit a sprite-sized grid where only non-majority colors are kept at that relative cell; all other cells are background.

    Key insight:
    - The signal is inside the copy offsets: align every slot by bbox and keep the disagreement colors at each relative coordinate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, min_area=1)
    layout = _layout_from_components(components)
    assert layout is not None, "unable to detect a regular slot layout"

    h = int(layout["height"])
    w = int(layout["width"])

    patch_list: list[np.ndarray] = []
    for comp in layout["components"]:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (h, w), "slot bbox must align to sprite size"
        patch_list.append(patch)

    out = np.full((h, w), int(bg), dtype=int)
    for rr in range(h):
        for cc in range(w):
            freq = Counter(int(patch[rr, cc]) for patch in patch_list)
            base_color, base_count = freq.most_common(1)[0]
            if len(freq) > 1 and freq.most_common(2)[1][1] == base_count:
                raise AssertionError("ambiguous base color at slot-relative cell")

            deviants = [(color, cnt) for color, cnt in freq.items() if color != base_color]
            if deviants:
                assert len(deviants) == 1, "each slot-relative cell should have at most one deviation color"
                out[rr, cc] = int(deviants[0][0])

    return out


# -------------------------- generator --------------------------

def _sample_slot_layout(sprite_h: int, sprite_w: int) -> tuple[int, int, int, int] | None:
    """Return (rows, cols, gap_r, gap_c)."""
    max_slots = sprite_h * sprite_w
    options = [(r, c) for r in range(2, 5) for c in range(2, 5) if r * c <= max_slots]

    if random.random() < 0.35:
        options.extend([(1, c) for c in range(2, min(7, max_slots) + 1)])
        options.extend([(r, 1) for r in range(2, min(7, max_slots) + 1)])

    if not options:
        return None

    rows, cols = random.choice(options)
    gap_r = random.randint(1 if rows > 1 else 0, 2)
    gap_c = random.randint(1 if cols > 1 else 0, 2)
    return rows, cols, gap_r, gap_c


def _is_free_for_noise(r: int, c: int, occupied: set[tuple[int, int]], H: int, W: int) -> bool:
    if (r, c) in occupied:
        return False
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and (nr, nc) in occupied:
            return False
    return True


def _place_isolated_noise_pairs(grid: np.ndarray, occupied: set[tuple[int, int]], color: int, attempts: int = 32) -> None:
    H, W = grid.shape
    for _ in range(attempts):
        candidates = [
            (r, c)
            for r in range(H)
            for c in range(W)
            if _is_free_for_noise(r, c, occupied, H, W)
        ]
        if not candidates:
            return
        r0, c0 = random.choice(candidates)

        for dr, dc in ((1, 0), (0, 1)):
            r1, c1 = r0 + dr, c0 + dc
            if not (0 <= r1 < H and 0 <= c1 < W):
                continue
            if not _is_free_for_noise(r1, c1, occupied, H, W):
                continue
            grid[r0, c0] = color
            grid[r1, c1] = color
            occupied.add((r0, c0))
            occupied.add((r1, c1))
            return


def _render_expected_and_grid(
    sprite_h: int,
    sprite_w: int,
    rows: int,
    cols: int,
    gap_r: int,
    gap_c: int,
    margin_t: int,
    margin_b: int,
    margin_l: int,
    margin_r: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[list[tuple[int, int, int]]], np.ndarray]:
    """Create slot array plus per-slot deviation map from random choices."""
    base_colors = [c for c in range(10)]
    bg = random.choice(base_colors)
    base_colors.remove(bg)
    base_color = random.choice(base_colors)
    dev_pool = [c for c in base_colors if c != base_color]

    height = rows * sprite_h + max(0, rows - 1) * gap_r + margin_t + margin_b
    width = cols * sprite_w + max(0, cols - 1) * gap_c + margin_l + margin_r

    grid = np.full((height, width), int(bg), dtype=int)
    expected = np.full((sprite_h, sprite_w), int(bg), dtype=int)

    top_lefts: list[tuple[int, int]] = []
    for ri in range(rows):
        for ci in range(cols):
            top_lefts.append((margin_t + ri * (sprite_h + gap_r), margin_l + ci * (sprite_w + gap_c)))

    n_slots = rows * cols
    rel_positions = [(r, c) for r in range(sprite_h) for c in range(sprite_w)]
    random.shuffle(rel_positions)

    remaining = rel_positions[:]  # global per-slot disjointness over relative positions
    assignments: list[list[tuple[int, int, int]]] = [[] for _ in range(n_slots)]

    force_unique = n_slots <= len(remaining) and random.random() < 0.55
    if force_unique:
        force_n = random.randint(max(2, n_slots // 2), min(n_slots, len(remaining)))
        forced_slots = random.sample(range(n_slots), force_n)
        for slot_idx in forced_slots:
            rr, cc = remaining.pop()
            color = random.choice(dev_pool)
            assignments[slot_idx].append((rr, cc, color))

    for slot_idx in range(n_slots):
        extra = random.randint(0, 2)
        if force_unique:
            extra = max(0, extra - 1)
        if force_unique and random.random() < 0.3:
            extra += 1

        for _ in range(extra):
            if not remaining:
                break
            if random.random() < 0.15:
                break
            rr, cc = remaining.pop()
            color = random.choice(dev_pool)
            assignments[slot_idx].append((rr, cc, color))

    if not any(assignments):
        slot_idx = random.randrange(n_slots)
        rr, cc = remaining.pop() if remaining else (random.randrange(sprite_h), random.randrange(sprite_w))
        assignments[slot_idx].append((rr, cc, random.choice(dev_pool)))

    for slot_idx, (r0, c0) in enumerate(top_lefts):
        patch = np.full((sprite_h, sprite_w), int(base_color), dtype=int)
        for rr, cc, color in assignments[slot_idx]:
            patch[rr, cc] = color
            if expected[rr, cc] == bg:
                expected[rr, cc] = color
        grid[r0 : r0 + sprite_h, c0 : c0 + sprite_w] = patch

    return grid, top_lefts, assignments, expected


def _add_outside_noise(grid: np.ndarray, top_lefts: list[tuple[int, int]], sprite_h: int, sprite_w: int, bg: int, base_color: int) -> None:
    H, W = grid.shape
    occupied = {
        (r, c)
        for top_r, top_c in top_lefts
        for r in range(top_r, top_r + sprite_h)
        for c in range(top_c, top_c + sprite_w)
    }

    noise_colors = [c for c in range(10) if c not in (bg, base_color)]

    singles = random.randint(2, 12)
    for _ in range(singles):
        candidates = [(r, c) for r in range(H) for c in range(W) if _is_free_for_noise(r, c, occupied, H, W)]
        if not candidates:
            break
        r, c = random.choice(candidates)
        grid[r, c] = random.choice(noise_colors)
        occupied.add((r, c))

    pair_blocks = random.randint(0, 3)
    for _ in range(pair_blocks):
        _place_isolated_noise_pairs(grid, occupied, random.choice(noise_colors))


def _candidate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        sprite_h = random.randint(3, 8)
        sprite_w = random.randint(3, 8)
        layout = _sample_slot_layout(sprite_h, sprite_w)
        if layout is None:
            continue

        rows, cols, gap_r, gap_c = layout
        margin_t = random.randint(0, 2)
        margin_b = random.randint(0, 2)
        margin_l = random.randint(0, 2)
        margin_r = random.randint(0, 2)

        base_colors = list(range(10))
        bg = random.choice(base_colors)
        base_colors.remove(bg)
        if len(base_colors) < 2:
            continue

        # Re-generate with explicit base/noise color control for robust output reconstruction.
        height = rows * sprite_h + max(0, rows - 1) * gap_r + margin_t + margin_b
        width = cols * sprite_w + max(0, cols - 1) * gap_c + margin_l + margin_r
        if height > 30 or width > 30:
            continue

        n_slots = rows * cols
        rel_positions = [(r, c) for r in range(sprite_h) for c in range(sprite_w)]
        if n_slots > len(rel_positions):
            # Hard to guarantee per-slot disjoint coordinates.
            continue

        base_color = random.choice(base_colors)
        dev_pool = [c for c in base_colors if c != base_color]

        top_lefts: list[tuple[int, int]] = []
        for ri in range(rows):
            for ci in range(cols):
                top_lefts.append((margin_t + ri * (sprite_h + gap_r), margin_l + ci * (sprite_w + gap_c)))

        remaining = rel_positions[:]
        random.shuffle(remaining)
        assignments: list[list[tuple[int, int, int]]] = [[] for _ in range(n_slots)]

        force_unique = random.random() < 0.55
        if force_unique:
            force_n = random.randint(max(2, n_slots // 2), n_slots)
            force_slots = random.sample(range(n_slots), force_n)
            for slot_idx in force_slots:
                if not remaining:
                    break
                rr, cc = remaining.pop()
                assignments[slot_idx].append((rr, cc, random.choice(dev_pool)))

        for slot_idx in range(n_slots):
            extra = random.randint(0, 2)
            if force_unique:
                extra = max(0, extra - 1)
            if force_unique and random.random() < 0.35:
                extra += 1

            for _ in range(extra):
                if not remaining or random.random() < 0.25:
                    break
                rr, cc = remaining.pop()
                assignments[slot_idx].append((rr, cc, random.choice(dev_pool)))

        if not any(assignments):
            # keep at least one deviation
            if not remaining:
                continue
            slot_idx = random.randrange(n_slots)
            rr, cc = remaining.pop()
            assignments[slot_idx].append((rr, cc, random.choice(dev_pool)))

        grid = np.full((height, width), int(bg), dtype=int)
        expected = np.full((sprite_h, sprite_w), int(bg), dtype=int)

        for slot_idx, (r0, c0) in enumerate(top_lefts):
            patch = np.full((sprite_h, sprite_w), int(base_color), dtype=int)
            for rr, cc, color in assignments[slot_idx]:
                patch[rr, cc] = color
                expected[rr, cc] = color
            grid[r0 : r0 + sprite_h, c0 : c0 + sprite_w] = patch

        # Add drift noise outside array.
        _add_outside_noise(grid, top_lefts, sprite_h, sprite_w, bg, base_color)

        return grid, expected

    return None


def generate_input() -> np.ndarray:
    """Generate one input instance for this puzzle family."""
    for _ in range(300):
        candidate = _candidate_attempt()
        if candidate is None:
            continue

        inp, expected = candidate
        try:
            out = main(inp)
        except Exception:
            continue
        if out.shape != expected.shape or not np.array_equal(out, expected):
            continue

        bg = get_bg(inp)
        if np.count_nonzero(inp == bg) <= inp.size // 2:
            continue
        return inp

    raise RuntimeError("failed to generate valid 014/1 input")
