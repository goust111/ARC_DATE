from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import get_bg


MAX_GEN_ATTEMPTS = 400
HEIGHT_RANGE = (5, 9)
EVEN_WIDTHS = (8, 10, 12, 14)
ODD_WIDTHS = (9, 11, 13, 15)
EVEN_BRANCH_PROB = 0.75
PALETTE = tuple(range(10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _palindrome_candidates(row: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """All distinct cyclic shifts of a row that are palindromic about the vertical axis."""
    width = int(row.shape[0])
    best_by_bytes: dict[bytes, tuple[int, np.ndarray]] = {}
    for shift in range(width):
        restored = np.roll(row, -shift)
        if not np.array_equal(restored, restored[::-1]):
            continue
        key = restored.tobytes()
        prev = best_by_bytes.get(key)
        if prev is None or shift < prev[0]:
            best_by_bytes[key] = (int(shift), restored.copy())
    return sorted(best_by_bytes.values(), key=lambda item: item[0])


def _mask_component_stats(mask: np.ndarray) -> tuple[int, int]:
    """Return (largest_area, component_count) for a 4-connected boolean mask."""
    seen = np.zeros(mask.shape, dtype=bool)
    largest = 0
    component_count = 0
    height, width = mask.shape

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            component_count += 1
            area = 0
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                area += 1
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            largest = max(largest, area)

    return int(largest), int(component_count)


def _vertical_contacts(grid: np.ndarray, bg: int) -> int:
    fg = grid != int(bg)
    return int(np.sum(fg[:-1] & fg[1:])) if grid.shape[0] > 1 else 0


def _grid_score(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    """Prefer restorations whose foreground forms the strongest vertical 4-connected object."""
    largest, components = _mask_component_stats(grid != int(bg))
    contacts = _vertical_contacts(grid, bg)
    return int(largest), int(contacts), -int(components)


def _best_restoration(
    grid: np.ndarray, bg: int, *, return_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, tuple[int, int, int], int, int]:
    candidates_per_row = [_palindrome_candidates(grid[r]) for r in range(grid.shape[0])]
    assert all(candidates_per_row), "every row must have at least one palindromic cyclic restoration"

    best_grid = None
    best_score = None
    best_shifts = None
    best_count = 0
    total_variants = 1
    for row_candidates in candidates_per_row:
        total_variants *= len(row_candidates)

    for combo in product(*candidates_per_row):
        shifts = tuple(int(shift) for shift, _row in combo)
        restored = np.stack([row for _shift, row in combo], axis=0)
        score = _grid_score(restored, bg)
        if (
            best_score is None
            or score > best_score
            or (score == best_score and shifts < best_shifts)
        ):
            best_grid = restored
            best_score = score
            best_shifts = shifts
            best_count = 1
        elif score == best_score:
            best_count += 1

    assert best_grid is not None and best_score is not None
    if return_meta:
        return best_grid, best_score, int(best_count), int(total_variants)
    return best_grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cyclically unshift each row until the full picture becomes vertically mirror-symmetric again.

    Input structure:
    - The grid has one dominant background color and every row contains some foreground cells.
    - Each row comes from a mirror-symmetric target row after an independent horizontal cyclic shift.
    - Some rows may admit more than one palindrome-preserving cyclic restoration.

    Rule:
    - For each row, enumerate the cyclic shifts that make that row a left-right palindrome.
    - Combine those row-wise candidates into whole-grid restorations.
    - Choose the restoration whose foreground forms the strongest vertical 4-connected structure with adjacent rows, then output that restored grid.

    Key insight:
    - Row-wise palindrome is necessary but not always sufficient: the correct phase is the one that keeps the foreground connected from row to row, rather than splitting it into disconnected wrapped fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    return _best_restoration(g, bg).astype(int)


def _max_width_for_branch(width: int) -> int:
    max_width = width // 2 if width % 2 == 0 else width // 2 + 1
    if max_width % 2 != width % 2:
        max_width -= 1
    return max_width


def _width_choices(width: int) -> list[int]:
    parity = width % 2
    min_width = 2 if parity == 0 else 1
    max_width = _max_width_for_branch(width)
    return list(range(min_width, max_width + 1, 2))


def _sample_width_profile(rng: np.random.Generator, height: int, width: int) -> list[int]:
    choices = _width_choices(width)
    profile = [int(rng.choice(choices))]
    for _ in range(height - 1):
        prev = profile[-1]
        nearby = [w for w in choices if abs(w - prev) <= 4]
        if rng.random() < 0.2:
            nearby = choices
        profile.append(int(rng.choice(nearby)))
    return profile


def _centered_target(height: int, width: int, bg: int, fg: int, widths: list[int]) -> np.ndarray:
    out = np.full((height, width), int(bg), dtype=int)
    for r, span in enumerate(widths):
        c0 = (width - span) // 2
        out[r, c0 : c0 + span] = int(fg)
    return out


def _sample_scramble_shifts(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    while True:
        shifts = rng.integers(0, width, size=height, dtype=int)
        if len(set(map(int, shifts))) < 2:
            continue
        if np.count_nonzero(shifts) < max(1, height // 2):
            continue
        return shifts


def _scramble_rows(target: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    rows = [np.roll(target[r], int(shifts[r])) for r in range(target.shape[0])]
    return np.stack(rows, axis=0)


def _semantic_checks(
    inp: np.ndarray, out: np.ndarray, *, expect_ambiguity: bool
) -> bool:
    if np.array_equal(inp, out):
        return False

    bg = int(get_bg(inp))
    if bg != int(get_bg(out)):
        return False

    fg_cells = int(np.sum(out != bg))
    if fg_cells == 0 or fg_cells * 2 >= out.size:
        return False

    row_candidates = [_palindrome_candidates(inp[r]) for r in range(inp.shape[0])]
    candidate_counts = [len(cands) for cands in row_candidates]
    if min(candidate_counts) < 1:
        return False

    has_ambiguity = any(count > 1 for count in candidate_counts)
    if expect_ambiguity and not has_ambiguity:
        return False
    if not expect_ambiguity and has_ambiguity:
        return False

    solved, best_score, best_count, total_variants = _best_restoration(inp, bg, return_meta=True)
    if not np.array_equal(solved, out):
        return False
    if best_count != 1:
        return False
    if expect_ambiguity and total_variants <= 1:
        return False

    target_score = _grid_score(out, bg)
    if target_score != best_score:
        return False

    largest, components = _mask_component_stats(out != bg)
    if largest != fg_cells or components != 1:
        return False

    changed_rows = int(np.sum(np.any(inp != out, axis=1)))
    return changed_rows >= max(1, inp.shape[0] // 2)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    use_even_branch = bool(rng.random() < EVEN_BRANCH_PROB)
    width = int(rng.choice(EVEN_WIDTHS if use_even_branch else ODD_WIDTHS))
    height = int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))

    colors = rng.choice(PALETTE, size=2, replace=False)
    bg = int(colors[0])
    fg = int(colors[1])

    widths = _sample_width_profile(rng, height, width)
    if len(set(widths)) < 2:
        return None

    target = _centered_target(height, width, bg, fg, widths)
    shifts = _sample_scramble_shifts(rng, height, width)
    inp = _scramble_rows(target, shifts)

    if not _semantic_checks(inp, target, expect_ambiguity=use_even_branch):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated semantic-validation retries")
