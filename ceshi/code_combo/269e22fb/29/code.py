# Combo variant of 269e22fb (index 29)
# combo_spec: D1/0+D7/0+D13/0
# Name: 多小窗口联合消歧

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# spec_variant:
#   core_fusion:
#     original_input: one fully observed 2-color crop whose size alone guarantees a
#       unique (variant, offset, polarity) match.
#     fused_input: one clue row plus one 3-color masked body. The body reveals exactly
#       two tiny rectangles cut from the same 20x20 dihedral motif, while the clue row
#       contributes one extra parity bit.
#     fused_uniqueness_chain:
#       - each tiny window is locally ambiguous
#       - the two windows together still leave exactly 2 global candidates
#       - the clue row chooses between the even-indexed and odd-indexed candidate
#   behavior_trace:
#     concrete_example:
#       - body size might be 12x13, with windows at (1,1,5,6) and (7,8,6,5)
#       - window A may appear in 4 places over all variants/offsets
#       - window B may appear in 3 places over all variants/offsets
#       - matching both windows inside the same outer frame may shrink that to 2 full
#         candidates, for example one even variant and one odd variant
#       - the top endpoint marker then keeps only the parity-consistent candidate
#     difference_from_original:
#       original logic accepted a single rectangle match; fused logic must combine:
#       1) clue-row parsing, 2) mask-aware component extraction, 3) joint spatial
#       matching of two windows, and 4) parity filtering of the remaining candidates
#   complexity_calibration:
#     body_size: 11..15 in each dimension so the two windows and their spacing are
#       visually legible without filling the entire crop
#     window_sizes: heights 5..6 and widths 5..7; this is much smaller than the
#       original 8..16 by 8..12 regime and makes standalone ambiguity common
#     ambiguity_target: generator rejects samples unless body-only matching returns
#       exactly 2 candidates split across parities {0,1}; this makes the clue visibly
#       necessary and prevents a silent fallback to the D1-only regime
#     branch_balance: candidate precomputation stores separate even/odd buckets so the
#       clue-row branch is exercised in a roughly balanced way over many samples
#   old_main_assumptions_that_break:
#     - `len(colors) == 2` on the whole input is false: the fused grid uses 4 colors
#       overall (2 motif colors + mask color + clue-strip color)
#     - every non-clue cell is no longer observed; mask-colored cells in the body must
#       be ignored during pattern matching
#     - one rectangular crop is no longer present; the solver must recover exactly 2
#       disjoint rectangles and preserve their shared coordinates inside the body
#     - the raw match set is no longer unique before reading the clue row
#   new_solver_assumptions_and_generator_guarantees:
#     - row 0 is always the clue strip, containing one majority clue color and exactly
#       one endpoint marker at column 0 or column W-1
#     - the clue-strip majority color does not appear in the body, so the strip is
#       structurally separable from the masked windows
#     - the body uses exactly 3 colors and its non-mask cells form exactly 2 connected
#       components, each of which is a filled rectangle
#     - each visible window contains both motif colors and has at least 2 standalone
#       matches over all variants/offsets
#     - body-only matching returns exactly 2 candidates, and their variant parities are
#       {0, 1}; filtering by the clue parity yields exactly 1 solution
#   placement_legality:
#     - both windows are sampled within one valid body rectangle taken from the 20x20
#       motif, so no visible patch can go out of bounds
#     - windows are non-overlapping and separated by at least one full masked row or
#       column, which keeps component extraction unambiguous
#     - the clue row sits above the masked body and never overwrites motif pixels
#   visual_observability:
#     - humans can see 3 separate evidence layers: endpoint clue, hidden space, and 2
#       tiny revealed windows
#     - the windows are deliberately small enough that the extra clue feels necessary
#     - parity is encoded by left vs right endpoint position, not by any numeric color id
#   visual_cleanliness:
#     - the layout is sparse: 1 clue strip plus 2 small rectangular reveals
#     - no overlapping markers or dense decorative noise are introduced
#   color_reasoning:
#     - colors keep structural roles only; the marker color reuses one motif color, but
#       the rule depends on the marker's side, not on which color id was chosen


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)

Window = tuple[int, int, int, int]
BodyMatch = tuple[int, int, int, int, int, np.ndarray]
Record = tuple[int, int, int, int, int, Window, Window]


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


@lru_cache(maxsize=1)
def _variants() -> tuple[np.ndarray, ...]:
    return tuple(_dihedral_variants(_pattern_mask()))


@lru_cache(maxsize=1)
def _single_window_counts() -> dict[tuple[int, int], dict[bytes, int]]:
    counts: dict[tuple[int, int], dict[bytes, int]] = {}
    for h in range(5, 7):
        for w in range(5, 8):
            bucket: dict[bytes, int] = {}
            for pattern in _variants():
                for r in range(20 - h + 1):
                    for c in range(20 - w + 1):
                        key = pattern[r : r + h, c : c + w].tobytes()
                        bucket[key] = bucket.get(key, 0) + 1
            counts[h, w] = bucket
    return counts


def _component_boxes(mask: np.ndarray) -> list[Window]:
    H, W = map(int, mask.shape)
    seen = np.zeros((H, W), dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            area = 0
            while stack:
                cr, cc = stack.pop()
                area += 1
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            boxes.append((r0, c0, r1, c1, area))
    boxes.sort()

    rects: list[Window] = []
    for r0, c0, r1, c1, area in boxes:
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if area != h * w:
            return []
        rects.append((r0, c0, h, w))
    return rects


def _extract_windows(body: np.ndarray, mask_color: int) -> list[Window] | None:
    rects = _component_boxes(body != int(mask_color))
    if len(rects) != 2:
        return None
    return rects


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= 12 and W >= 11

    clue_row = g[0]
    clue_colors = colors_present(clue_row)
    assert len(clue_colors) == 2
    counts = {color: int(np.sum(clue_row == color)) for color in clue_colors}
    clue_color = max(counts, key=counts.get)
    marker_positions = np.flatnonzero(clue_row != int(clue_color))
    assert len(marker_positions) == 1
    marker_col = int(marker_positions[0])
    assert marker_col in (0, W - 1)
    parity = 0 if marker_col == 0 else 1

    body = g[1:, :]
    body_colors = colors_present(body)
    assert len(body_colors) == 3
    assert int(clue_color) not in body_colors
    return body, parity


def _count_single_window_matches(visible: np.ndarray) -> int:
    h, w = map(int, visible.shape)
    bucket = _single_window_counts().get((h, w))
    if bucket is not None:
        return bucket.get(visible.tobytes(), 0)

    total = 0
    for pattern in _variants():
        for r in range(20 - h + 1):
            for c in range(20 - w + 1):
                total += int(np.array_equal(pattern[r : r + h, c : c + w], visible))
    return total


def _body_matches(body: np.ndarray) -> list[BodyMatch]:
    g = np.asarray(body, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 3

    matches: list[BodyMatch] = []
    for mask_color in colors:
        windows = _extract_windows(g, int(mask_color))
        if windows is None:
            continue
        motif_colors = [color for color in colors if color != int(mask_color)]
        for fg in motif_colors:
            bg = int(motif_colors[0] if motif_colors[1] == fg else motif_colors[1])
            visible_patches = [
                g[r0 : r0 + h, c0 : c0 + w] == int(fg)
                for r0, c0, h, w in windows
            ]
            for variant_index, pattern in enumerate(_variants()):
                for top in range(20 - H + 1):
                    for left in range(20 - W + 1):
                        ok = True
                        for (r0, c0, h, w), patch in zip(windows, visible_patches):
                            if not np.array_equal(
                                pattern[top + r0 : top + r0 + h, left + c0 : left + c0 + w],
                                patch,
                            ):
                                ok = False
                                break
                        if ok:
                            matches.append((int(fg), int(bg), variant_index, top, left, pattern))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from two tiny masked windows plus a one-bit endpoint clue.

    Input structure:
    - The first row is a clue strip in a dedicated clue color, with one endpoint cell recolored as a marker.
    - The remaining rows form a three-color masked body: two motif colors appear only inside exactly two tiny rectangular windows, and the third body color masks everything else.
    - A marker at the left endpoint means the correct dihedral variant index is even; a marker at the right endpoint means it is odd.

    Rule:
    - Read the top strip to obtain the required variant-index parity, then ignore that strip for motif matching.
    - In the masked body, try each body color as the possible mask, recover the two visible rectangles, and test both remaining colors as foreground/background candidates.
    - Match those two window patches jointly against every offset of every dihedral motif variant, keep only matches whose variant parity agrees with the clue, and repaint the full 20x20 motif with the recovered motif colors.

    Key insight:
    - Neither tiny window is distinctive on its own, and even the window pair is intentionally left with two global candidates; the endpoint clue supplies the final bit needed to choose the correct orientation.
    """
    body, parity = _parse_input(np.asarray(grid, dtype=int))
    matches = _body_matches(body)
    filtered = [match for match in matches if match[2] % 2 == parity]
    assert len(filtered) == 1
    fg, bg, _variant_index, _top, _left, pattern = filtered[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _windows_have_gap(a: Window, b: Window) -> bool:
    r1, c1, h1, w1 = a
    r2, c2, h2, w2 = b
    return (
        r1 + h1 < r2
        or r2 + h2 < r1
        or c1 + w1 < c2
        or c2 + w2 < c1
    )


def _render_body(
    pattern: np.ndarray,
    fg: int,
    bg: int,
    mask_color: int,
    outer_top: int,
    outer_left: int,
    outer_shape: tuple[int, int],
    windows: list[Window],
) -> np.ndarray:
    H, W = map(int, outer_shape)
    body = np.full((H, W), int(mask_color), dtype=int)
    for r0, c0, h, w in windows:
        patch = pattern[outer_top + r0 : outer_top + r0 + h, outer_left + c0 : outer_left + c0 + w]
        body[r0 : r0 + h, c0 : c0 + w] = np.where(patch, int(fg), int(bg))
    return body


@lru_cache(maxsize=1)
def _candidate_records() -> tuple[Record, ...]:
    rng = np.random.default_rng(0)
    records_by_parity: dict[int, list[Record]] = {0: [], 1: []}
    seen: set[Record] = set()
    target_per_parity = 12

    for _ in range(200000):
        variant_index = int(rng.integers(len(_variants())))
        pattern = _variants()[variant_index]
        H = int(rng.integers(11, 16))
        W = int(rng.integers(11, 16))
        outer_top = int(rng.integers(0, 20 - H + 1))
        outer_left = int(rng.integers(0, 20 - W + 1))

        h1 = int(rng.integers(5, 7))
        w1 = int(rng.integers(5, 8))
        h2 = int(rng.integers(5, 7))
        w2 = int(rng.integers(5, 8))
        r1 = int(rng.integers(0, H - h1 + 1))
        c1 = int(rng.integers(0, W - w1 + 1))
        r2 = int(rng.integers(0, H - h2 + 1))
        c2 = int(rng.integers(0, W - w2 + 1))
        windows = sorted([(r1, c1, h1, w1), (r2, c2, h2, w2)])
        if not _windows_have_gap(windows[0], windows[1]):
            continue

        visible_patches = [
            pattern[outer_top + r0 : outer_top + r0 + h, outer_left + c0 : outer_left + c0 + w]
            for r0, c0, h, w in windows
        ]
        if any(patch.min() == patch.max() for patch in visible_patches):
            continue

        single_counts = [_count_single_window_matches(patch) for patch in visible_patches]
        if min(single_counts) < 2:
            continue

        probe_body = _render_body(pattern, 1, 2, 3, outer_top, outer_left, (H, W), windows)
        matches = _body_matches(probe_body)
        if len(matches) != 2:
            continue
        if {match[2] % 2 for match in matches} != {0, 1}:
            continue

        record: Record = (H, W, variant_index, outer_top, outer_left, windows[0], windows[1])
        if record in seen:
            continue
        seen.add(record)
        parity = variant_index % 2
        records_by_parity[parity].append(record)
        if min(len(records_by_parity[0]), len(records_by_parity[1])) >= target_per_parity:
            break

    assert records_by_parity[0] and records_by_parity[1]
    return tuple(records_by_parity[0] + records_by_parity[1])


@lru_cache(maxsize=1)
def _candidate_records_by_parity() -> dict[int, tuple[Record, ...]]:
    grouped: dict[int, list[Record]] = {0: [], 1: []}
    for record in _candidate_records():
        grouped[record[2] % 2].append(record)
    assert grouped[0] and grouped[1]
    return {parity: tuple(records) for parity, records in grouped.items()}


def generate_input() -> np.ndarray:
    for _ in range(200):
        fg, bg, mask_color, clue_color = random.sample(Color.ALL_COLORS, 4)
        parity = random.randint(0, 1)
        H, W, variant_index, outer_top, outer_left, win_a, win_b = random.choice(
            _candidate_records_by_parity()[parity]
        )
        pattern = _variants()[variant_index]
        windows = [win_a, win_b]
        body = _render_body(
            pattern,
            int(fg),
            int(bg),
            int(mask_color),
            outer_top,
            outer_left,
            (H, W),
            windows,
        )

        clue_row = np.full((1, W), int(clue_color), dtype=int)
        clue_row[0, 0 if variant_index % 2 == 0 else W - 1] = int(fg)
        grid = np.vstack([clue_row, body])

        expected = np.where(pattern, int(fg), int(bg)).astype(int)
        if np.array_equal(main(grid), expected):
            return grid

    raise AssertionError("failed to construct a valid two-window parity-disambiguated sample")


def fusion_behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    body, parity = _parse_input(np.asarray(grid, dtype=int))
    colors = colors_present(body)
    mask_color = next(color for color in colors if _extract_windows(body, color) is not None)
    windows = _extract_windows(body, int(mask_color))
    assert windows is not None
    matches = _body_matches(body)
    chosen = [match for match in matches if match[2] % 2 == parity]
    assert len(chosen) == 1
    fg = int(chosen[0][0])
    single_counts = [
        _count_single_window_matches(body[r0 : r0 + h, c0 : c0 + w] == fg)
        for r0, c0, h, w in windows
    ]
    return {
        "body_match_count": len(matches),
        "parity_options": sorted({match[2] % 2 for match in matches}),
        "single_window_counts": tuple(int(v) for v in single_counts),
        "required_parity": int(parity),
    }
